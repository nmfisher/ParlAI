# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Example sequence to sequence agent for ParlAI "Creating an Agent" tutorial.
http://parl.ai/static/docs/tutorial_seq2seq.html
"""

from parlai.core.torch_agent import TorchAgent, Output, Batch
from parlai.core.utils import padded_tensor
from pytorch_transformers.tokenization_distilbert import DistilBertTokenizer
from pytorch_transformers.modeling_distilbert import DistilBertModel
import torch
import random
from torch import optim
import torch.nn as nn
import torch.nn.functional as F

class AttentionFF(nn.Module):
    """Encodes the input context."""

    def __init__(self, hidden_size, numlayers):
        """Initialize encoder.
        :param hidden_size: size of GRU hidden layers
        :param numlayers: number of GRU layers
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.embedder = DistilBertModel.from_pretrained('distilbert-base-uncased')

        self.attn = nn.MultiheadAttention(
            768, 2
        )
      
        self.key = nn.Linear(768, hidden_size)
        self.value = nn.Linear(768, hidden_size)

        self.dense = nn.Linear(
            768, hidden_size,
        )
        self.activation = nn.ReLU()
        self.linear = nn.Linear(hidden_size, 3)

    def forward(self, input, hidden=None):
        """Return encoded state.

        :param input: (batchsize x seqlen x embedding_dim) tensor of token indices.
        :param hidden: optional past hidden state
        """
        embedded1 = self.embedder(input[:,0])[0]
        embedded2 = self.embedder(input[:,1])[0]
        embedded3 = self.embedder(input[:,1])[0]
        attended1 = self.attn(embedded1, self.key(embedded1), self.value(embedded1))[0].sum(1)
        attended2 = self.attn(embedded2, self.key(embedded2), self.value(embedded2))[0].sum(1)

        attended = torch.stack([attended1, attended2], 1)

        output = self.linear(
          self.activation(
            self.dense(
              attended
            )
          )
        )
        return output

class ResponseSelection2Agent(TorchAgent):
    """Agent which takes an input sequence and produces an output sequence.

    This model is based on Sean Robertson's `seq2seq tutorial
    <http://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html>`_.
    """

    def state_dict(self):
        """
        Get the state dict for saving.
        Override this method for more specific saving.
        """
        states = {}
        states['model'] = self.model.state_dict()

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

        self.step = 0

        self.id = 'ResponseSelection2Agent'

        if not shared:
            # set up model from scratch
            hsz = opt['hiddensize']
            nl = opt['numlayers']
            esz = opt['embeddingsize']
            nc = opt["numcandidates"]
            self.model = AttentionFF(hsz, nl)

            if self.use_cuda:  # set in parent class
                self.model = self.model.cuda()

            if opt.get('numthreads', 1) > 1:
                self.model.share_memory()
        elif 'model' in shared:
            # copy initialized data from shared table
            self.model = shared['model']

        # set up the criterion
        self.criterion = nn.CrossEntropyLoss()

        # set up optims for each module
        lr = opt['learningrate']
        self.optims = {
            'model': optim.SGD(self.model.parameters(), lr=lr),
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
        if "labels" in obs:
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
        shared['model'] = self.model
        return shared

    def report(self):
      metrics = super().report()
      metrics['loss'] = self.agg_loss.item()
      metrics['step'] = self.step
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

    def _tokenize_observation(self, batch):
        bsz = len(batch.observations)
        inputs = []
        max_seq_len = 0
        candidates = []
        for i in range(bsz):
          utterance = batch.observations[i]["text"]
          c_t = batch.observations[i]["labels"][0]
          c_f = batch.observations[i]["label_candidates"][random.randint(0, len(batch.observations[i]["label_candidates"]) - 1)]
          c1 = utterance + " [SEP] " + c_t
          c2 = utterance + " [SEP] " + c_f
          c1_t = self.tokenizer.encode(c1)
          c2_t = self.tokenizer.encode(c2)
          max_seq_len = max(max_seq_len, max(len(c1_t), len(c2_t)))
          inputs.append([c1_t, c2_t])
          candidates.append([c_t, c_f])
        for c in inputs:
          while len(c[0]) < max_seq_len:
            c[0].append(0)
          while len(c[1]) < max_seq_len:
            c[1].append(0)
        inputs = torch.LongTensor(inputs)
        return inputs, candidates

    def _get_predictions(self, output, candidates, bsz):  
        preds = []
        pred_text = []

        for i in range(bsz):
          pred_idx = torch.max(output[i], 0)[1] # predicted binary value for each candidate, should be [1,0]
          if pred_idx[0].item() == 1 and pred_idx[1].item() == 0:
            pred_text.append(candidates[i][0])
          else:
            pred_text.append(candidates[i][1])
          #print("###### utterance: " + batch.observations[i]["text"])
          #print("####### correct response: " + batch.observations[i]["labels"][0])
          #print("######### predicted response: " + pred_text[-1])
        #  print("########## predicted index: " + str(pred_idx))
        return pred_text

    def train_step(self, batch):
        """Train model to produce ys given xs.

        :param batch: parlai.core.torch_agent.Batch, contains tensorized
                      version of observations.

        Return estimated responses, with teacher forcing on the input sequence
        (list of strings of length batchsize).
        """
        self.step += 1

        if self.step % 1000 == 0:
          self.eval_step(batch)

        xs = batch.text_vec

        if xs is None:
            return

        bsz = xs.size(0)
        inputs, candidates = self._tokenize_observation(batch)

        # ys will be the correct (embedded) candidate responses
        ys = [[1,0]] * bsz
        ys = torch.LongTensor(ys)

        if self.use_cuda:  # set in parent class
          ys = ys.cuda()
          inputs = inputs.cuda()

        loss = 0
        self.zero_grad()
        self.model.train()

        output = self.model(inputs) # returns bsz x 2 x 2

        # CrossEntropyLoss expects bsz x num_classes x ...., so first transpose
        output = torch.transpose(output, 1, 2)
        loss = self.criterion(output, ys)
        self.agg_loss += loss
        loss.backward()
        self.update_params()

        pred_text = self._get_predictions(output, candidates, bsz)

        return Output(pred_text)

    def eval_step(self, batch):
        """Generate a response to the input tokens.

        :param batch: parlai.core.torch_agent.Batch, contains tensorized
                      version of observations.

        Return predicted responses (list of strings of length batchsize).
        """
        item1 = { 
          "text":"Do you like playing, or watching sports?",
          "labels":["I like watching sports."],
          "label_candidates":list(batch.observations[0]["label_candidates"])
        }
        item1["label_candidates"].append(item1["labels"][0])

        item2 = { 
          "text":"Do you think chess counts as a sport?",
          "labels":["Yes, I think it does."],
          "label_candidates":list(batch.observations[0]["label_candidates"])
        }
        item2["label_candidates"].append(item2["labels"][0])
        batch = Batch(observations=[item1, item2])
        inputs, candidates = self._tokenize_observation(batch)
        # just predict
        self.model.eval()
        output = self.model(inputs.cuda())
        pred_text = self._get_predictions(output, candidates, 2)    
        print(pred_text)
        print("EVALUATING")
        return Output(pred_text)


