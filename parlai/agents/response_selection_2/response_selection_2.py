# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Example sequence to sequence agent for ParlAI "Creating an Agent" tutorial.
http://parl.ai/static/docs/tutorial_seq2seq.html
"""

from parlai.core.torch_agent import TorchAgent, Output
from parlai.core.utils import padded_tensor
from pytorch_transformers.tokenization_distilbert import DistilBertTokenizer
from pytorch_transformers.modeling_distilbert import DistilBertModel
import torch
import random
from torch import optim
import torch.nn as nn
import torch.nn.functional as F

class EncoderRNN(nn.Module):
    """Encodes the input context."""

    def __init__(self, input_size, hidden_size, numlayers):
        """Initialize encoder.

        :param input_size: size of embedding
        :param hidden_size: size of GRU hidden layers
        :param numlayers: number of GRU layers
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.embedder = DistilBertModel.from_pretrained('distilbert-base-uncased')

        self.gru = nn.GRU(
            input_size, hidden_size, num_layers=numlayers, batch_first=True
        )

    def forward(self, input, hidden=None):
        """Return encoded state.

        :param input: (batchsize x seqlen x embedding_dim) tensor of token indices.
        :param hidden: optional past hidden state
        """
        input = self.embedder(torch.squeeze(input))[0]
        output, hidden = self.gru(input, hidden)
        return output, hidden

class FF(nn.Module):
    """Encodes the input context."""

    def __init__(self, hidden_size, embeddingsize, numcandidates):
        """Initialize encoder.

        :param input_size: size of embedding
        :param hidden_size: size of GRU hidden layers
        :param numlayers: number of GRU layers
        """
        super().__init__()
        self.input_size = 2 * embeddingsize
        self.hidden_size = hidden_size

        self.dense = nn.Linear(
            self.input_size, hidden_size,
        )
        self.activation = nn.ReLU()
        self.linear = nn.Linear(hidden_size, 2)

    def forward(self, input):
        """Return encoded state.

        :param input: (batchsize x seqlen) tensor of token indices.
        :param hidden: optional past hidden state
        """
        # use the last sequence output as the input
        output = self.linear(
          self.activation(
            self.dense(
              input
            )
          )
        )
        return output

class ResponseSelection2Agent(TorchAgent):
    """Agent which takes an input sequence and produces an output sequence.

    This model is based on Sean Robertson's `seq2seq tutorial
    <http://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html>`_.
    """

    def save(self, path=None):
      print("Saving!")
      super().save(path)

    def state_dict(self):
        """
        Get the state dict for saving.
        Override this method for more specific saving.
        """
        states = {}
        states['encoder'] = self.encoder.state_dict()
        states['ff'] = self.ff1.state_dict()

        if hasattr(self, 'optimizer'):
            # save optimizer params
            states['optimizer'] = self.optimizer.state_dict()
            states['optimizer_type'] = self.opt['optimizer']

        # lr scheduler
        if torch.__version__.startswith('0.'):
            warn_once(
                "Must upgrade to Pytorch 1.0 to save the state of your " "LR scheduler."
            )
        else:
            states['number_training_updates'] = self._number_training_updates
            if getattr(self, 'scheduler', None):
                states['lr_scheduler'] = self.scheduler.state_dict()
                states['lr_scheduler_type'] = self.opt['lr_scheduler']
            if getattr(self, 'warmup_scheduler', None):
                states['warmup_scheduler'] = self.warmup_scheduler.state_dict()

        return states
    @classmethod
    def add_cmdline_args(cls, argparser):
        """Add command-line arguments specifically for this agent."""
        super(ResponseSelection2Agent, cls).add_cmdline_args(argparser)
        agent = argparser.add_argument_group('ResponseSelection2 Arguments')
        agent.add_argument(
            '-hs',
            '--hiddensize',
            type=int,
            default=128,
            help='size of the hidden layers',
        )
        agent.add_argument(
            '-esz',
            '--embeddingsize',
            type=int,
            default=768,
            help='size of the token embeddings',
        )
        agent.add_argument(
            '-nl', '--numlayers', type=int, default=2, help='number of hidden layers'
        )
        agent.add_argument(
            '-nc', '--numcandidates', type=int, default=3, help='number of candidates to select from as output, inclusive of the correct candidate'
        )
        agent.add_argument(
            '-lr', '--learningrate', type=float, default=1, help='learning rate'
        )
        agent.add_argument(
            '-dr', '--dropout', type=float, default=0.1, help='dropout rate'
        )
        agent.add_argument(
            '--gpu', type=int, default=-1, help='which GPU device to use'
        )
        agent.add_argument(
            '-msl', '--maxseqlen', type=int, default=20, help='maximum sequence length'
        )
        agent.add_argument(
            '-rf',
            '--report-freq',
            type=float,
            default=0.001,
            help='Report frequency of prediction during eval.',
        )
        ResponseSelection2Agent.dictionary_class().add_cmdline_args(argparser)
        return agent

    def __init__(self, opt, shared=None):
        """Initialize example seq2seq agent.

        :param opt: options dict generated by parlai.core.params:ParlaiParser
        :param shared: optional shared dict with preinitialized model params
        """
        super().__init__(opt, shared)

        self.id = 'ResponseSelection2Agent'

        if not shared:
            # set up model from scratch
            hsz = opt['hiddensize']
            nl = opt['numlayers']
            esz = opt['embeddingsize']
            nc = opt["numcandidates"]
            # encoder converts batch_size * seqlen * embedding_dim to batch_size * embedding_dim
            self.encoder = EncoderRNN(esz, hsz, nl)
            self.model = self.encoder
            # FF1 converts batch_size * embedding_dim to batch_size * num_candidates
            self.ff1 = FF(hsz, esz, nc)

            if self.use_cuda:  # set in parent class
                self.encoder = self.encoder.cuda()
                self.ff1 = self.ff1.cuda()

            if opt.get('numthreads', 1) > 1:
                self.encoder.share_memory()
        elif 'encoder' in shared:
            # copy initialized data from shared table
            self.encoder = shared['encoder']

        # set up the criterion
        self.criterion = nn.CrossEntropyLoss()

        # set up optims for each module
        lr = opt['learningrate']
        self.optims = {
            'encoder': optim.SGD(self.encoder.parameters(), lr=lr),
        }

        self.longest_label = 1
        self.hiddensize = opt['hiddensize']
        self.numlayers = opt['numlayers']
        self.numcandidates = opt['numcandidates']
        self.embeddingsize = opt['embeddingsize']
        self.maxseqlen = opt['maxseqlen']
        self.START = torch.LongTensor([self.START_IDX])
        if self.use_cuda:
            self.START = self.START.cuda()

        self.reset()
  
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

        self.agg_loss = 0

    def _vectorize(self, text):
        tokens = self.tokenizer.encode(text)
        tokens = torch.LongTensor(tokens)
        return tokens

    def _set_text_vec(self, obs, history, truncate):
        """
        Set the 'text_vec' field in the observation.
        Useful to override to change vectorization behavior
        """
        obs["text_vec"] = self._vectorize(obs["text"])
        return obs

    def _set_label_vec(self, obs, add_start, add_end, truncate):
        """
        Set the 'label_vec' field in the observation.
        Useful to override to change vectorization behavior
        """
        obs["labels_vec"] = self._vectorize(obs["labels"][0])
        return obs

    def zero_grad(self):
        """Zero out optimizer."""
        for optimizer in self.optims.values():
            optimizer.zero_grad()

    def update_params(self):
        """Do one optimization step."""
        for optimizer in self.optims.values():
            optimizer.step()

    def share(self):
        """Share internal states."""
        shared = super().share()
        shared['encoder'] = self.encoder
        shared['ff1'] = self.ff1
      
        return shared

    def report(self):
      print(self.metrics)
      metrics = super().report()
      metrics['loss'] = self.agg_loss.item()
      self.agg_loss = 0
      return metrics

    def v2t(self, vector):
        """Convert vector to text.

        :param vector: tensor of token indices.
            1-d tensors will return a string, 2-d will return a list of strings
        """
        if vector.dim() == 1:
            output_tokens = []
            # Remove the final END_TOKEN that is appended to predictions
            for token in vector:
                if token == self.END_IDX:
                    break
                else:
                    output_tokens.append(token)
            return self.dict.vec2txt(output_tokens)
        elif vector.dim() == 2:
            return [self.v2t(vector[i]) for i in range(vector.size(0))]
        raise RuntimeError(
            'Improper input to v2t with dimensions {}'.format(vector.size())
        )

    def vectorize(self, *args, **kwargs):
        """Call vectorize without adding start tokens to labels."""
        kwargs['add_start'] = False
        return super().vectorize(*args, **kwargs)

    def _extract_and_tokenize(self, candidates, correct):
      candidates_toks = []
      selected = []
      for j in range(self.numcandidates - 1):
        candidate = None
        while candidate is None or candidate in selected or candidate == correct:
          candidate = random.choice(candidates)
        selected.append(candidate)
        candidate_toks = self.tokenizer.encode(candidate)
        candidates_toks.append(candidate_toks)
      # add the correct candidate as the last entry
      correct_toks = self.tokenizer.encode(correct)
      candidates_toks.append(correct_toks)
      candidates_toks = padded_tensor(candidates_toks)[0]
      selected.append(correct)
      return candidates_toks, selected

    def train_step(self, batch):
        """Train model to produce ys given xs.

        :param batch: parlai.core.torch_agent.Batch, contains tensorized
                      version of observations.

        Return estimated responses, with teacher forcing on the input sequence
        (list of strings of length batchsize).
        """
        xs = batch.text_vec

        if xs is None:
            return

        bsz = xs.size(0)

        # xs are the tokenized utterances
        xs = torch.unsqueeze(xs, 1) # transform to bsz x 1 x seqlen
        # now encode and take the last RNN output
        xs = self.encoder(xs)[0]
        xs = xs[:,-1] # bsz x esz
        xs = torch.unsqueeze(xs, 1) # bsz x 1 x esz, so we can tile on dim1 later
        
        inputs = torch.zeros([bsz, self.numcandidates, self.embeddingsize * 2])

        candidates = []

        for i in range(bsz):
          obs = batch.observations[i]
          # tokenize/embed the responses
          toks, selected = self._extract_and_tokenize(obs["label_candidates"], obs["labels"][0]) # returns numcandidates x seqlen
          candidates.append(selected)
          encoded = self.encoder(toks.cuda())[0]
          encoded = torch.squeeze(encoded[:,-1], 1) # should be numcandidates x 1 x esz
          
          # tile the utterance to numcandidates
          # xs[i] is 1 x esz
          x = xs[i].expand(self.numcandidates, self.embeddingsize) # convert esz to (numcandidates x esz
          encoded = torch.cat([x, encoded], 1) # numcandidates x esz
          inputs[i] = encoded

        # ys will be the correct (embedded) candidate responses
        ys = [0] * (self.numcandidates - 1)
        ys += [1]
        ys = [ys] * bsz
        ys = torch.LongTensor(ys)
        

        if self.use_cuda:  # set in parent class
          ys = ys.cuda()
          inputs.cuda()

        loss = 0
        self.zero_grad()
        self.encoder.train()

        # Feed the concatenated inputs to the input to the FF network
        # the output will be trained to minimize MSE against the correct candidate vector
        ff_output = self.ff1(inputs.cuda())
        # CrossEntropyLoss expects bsz x num_classes x ...., to transpose
        ff_output = torch.transpose(ff_output, 1, 2)
        #print("ys")
        #print(ys.size())
        #print(ys)
        loss = self.criterion(ff_output, ys)
        self.agg_loss += loss
        loss.backward()
        self.update_params()

        preds = []
        #print("ff_output")
        #print(ff_output.size()) # should be bsz x numcandidates x 2
        #print(ff_output)
        pred_text = []

        for i in range(bsz):
          #print("logits")
          #print(ff_output[i])  # is numcandidates x 2
          pred_idx = torch.max(ff_output[i], 0)[0] # to get max logit per candidate
          print("pred_idx1")
          print(pred_idx.size())
          pred_idx = torch.max(pred_idx, 0)[1].item() # to get max candidate index per utterance
          #print("pred_idx")
          #print(pred_idx)
          prediction = candidates[i][pred_idx]

        #  print("###### utterance: " + batch.observations[i]["text"])
        #  print("####### correct response: " + batch.observations[i]["labels"][0])
        #  print("######### predicted response: " + prediction)
        #  print("########## predicted index: " + str(pred_idx))
          pred_text.append(prediction)
         #        print(pred_text)
        return Output(pred_text)

    def eval_step(self, batch):
        """Generate a response to the input tokens.

        :param batch: parlai.core.torch_agent.Batch, contains tensorized
                      version of observations.

        Return predicted responses (list of strings of length batchsize).
        """
        xs = batch.text_vec
        if xs is None:
            return
        bsz = xs.size(0)
        starts = self.START.expand(bsz, 1)  # expand to batch size
        # just predict
        self.encoder.eval()
        _encoder_output, encoder_hidden = self.encoder(xs)
        ff_output, _ = self.ff(_encoder_output)

        _, output_indices = torch.max(ff_output,1)

        scores = ff_output.view(-1, ff_output.size(-1))
        pred_text = torch.gather(batch.text, output_indices)
        return Output(pred_text)


