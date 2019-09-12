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

    def __init__(self, hidden_size, numlayers, vocab_size):
        """Initialize encoder.
        :param hidden_size: size of GRU hidden layers
        :param numlayers: number of GRU layers
        """
        super().__init__()


        self.hidden_size = hidden_size

        self.numlayers = numlayers
        self.embedder = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.attn = []
        self.key = []
        self.value = []
        self.dense = []
        self.activation = []

        self.lstm = torch.nn.LSTM(768, 768, 1)

        for i in range(numlayers):
          self.attn.append(nn.MultiheadAttention(
              768, 2
          ).cuda())
        
          self.key.append(nn.Linear(768, hidden_size).cuda())
          self.value.append(nn.Linear(768, hidden_size).cuda())
  
          self.dense.append(nn.Linear(
              768, hidden_size,
          ).cuda())
          self.activation.append(nn.ReLU().cuda())
        self.linear = nn.Linear(hidden_size, vocab_size).cuda()

    def forward(self, input, hidden=None):
        """Return encoded state.

        :param input: (batchsize x seqlen x embedding_dim) tensor of token indices.
        :param hidden: optional past hidden state
        """
        embedded = self.embedder(input)[0]
#        embedded = self.lstm(embedded)[0]

#        print("lstme")
#        print(embedded.size())

#        embedded = embedded[-1]
        #for i in range(self.numlayers):
          #print(str(i))
        #  k = self.key[i](embedded)
        #  v = self.value[i](embedded)
        #  embedded = self.attn[i](embedded, k, v)[0]
        #  embedded = self.activation[i](
        #      self.dense[i](
        #          embedded
        #      )
        #  )
        
        return self.linear(embedded)

class AutoencoderAgent(TorchAgent):
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
        super(AutoencoderAgent, cls).add_cmdline_args(argparser)
        agent = argparser.add_argument_group('Autoencoder Arguments')
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
        AutoencoderAgent.dictionary_class().add_cmdline_args(argparser)
        return agent

    def __init__(self, opt, shared=None):
        """Initialize example seq2seq agent.

        :param opt: options dict generated by parlai.core.params:ParlaiParser
        :param shared: optional shared dict with preinitialized model params
        """
        super().__init__(opt, shared)


        self.step = 0

        self.id = 'AutoencoderAgent'

        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

        if not shared:
            # set up model from scratch
            hsz = opt['hiddensize']
            nl = opt['numlayers']
            esz = opt['embeddingsize']
            nc = opt["numcandidates"]
            self.model = AttentionFF(hsz, nl, self.tokenizer.vocab_size)

            if self.use_cuda:  # set in parent class
                self.model = self.model.cuda()

            if opt.get('numthreads', 1) > 1:
                self.model.share_memory()
        elif 'model' in shared:
            # copy initialized data from shared table
            self.model = shared['model']

        init_model, is_finetune = self._get_init_model(opt, shared)

        if init_model is not None:
          # load model parameters if available
          print('[ Loading existing model params from {} ]' ''.format(init_model))
          states = self.load(init_model)
        else:
          states = {}

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

    def _tokenize_observation(self, batch):
        bsz = len(batch.observations)
        inputs = []
        max_seq_len = 0
        for i in range(bsz):
          utterance = batch.observations[i]["labels"][0]
          c_t = self.tokenizer.encode(utterance)
          max_seq_len = max(max_seq_len, len(c_t))
          inputs.append(c_t)
        for c in inputs:
          while len(c) < max_seq_len:
            c.append(self.tokenizer.encode("[PAD]")[0])
        inputs = torch.LongTensor(inputs)
        return inputs

    def _get_predictions(self, output):  
        pred_text = []
#        for i in range(output.size(0)):
#        print(self.tokenizer.convert_ids_to_tokens([2821]))
       # print(output.size())
        indices = torch.argmax(output, 1) # get token index for each step in sequence 
#        print(indices.size())
        for i in range(indices.size(0)):
          tokens = self.tokenizer.convert_ids_to_tokens(indices[i].tolist())
          pred_text.append(" ".join(tokens).replace("[PAD]","").strip())
#        print("pred_text len : %d" % len(pred_text))
        return pred_text

    def load(self, path):
      states = super().load(path)
      return states

    def train_step(self, batch):
        """Train model to produce ys given xs.

        :param batch: parlai.core.torch_agent.Batch, contains tensorized
                      version of observations.

        Return estimated responses, with teacher forcing on the input sequence
        (list of strings of length batchsize).
        """
        self.step += 1

        if self.step % 100 == 0:
          self.eval_step(batch)

        xs = batch.text_vec

        if xs is None:
            return

        bsz = xs.size(0)
        inputs = self._tokenize_observation(batch)

        if self.use_cuda:  # set in parent class
          inputs = inputs.cuda()

        loss = 0
        self.zero_grad()
        self.model.train()

        output = self.model(inputs) # returns bsz x seqlen x vocab_size
        # CrossEntropyLoss expects bsz x num_classes x ...., so first transpose
#        output = output[:,0,:]
        output = torch.transpose(output, 1, 2)
        loss = self.criterion(output, inputs)
        self.agg_loss += loss
        loss.backward()
        self.update_params()
        pred_text = self._get_predictions(output)

        return Output(pred_text)

    def eval_step(self, batch):
        """Generate a response to the input tokens.

        :param batch: parlai.core.torch_agent.Batch, contains tensorized
                      version of observations.

        Return predicted responses (list of strings of length batchsize).
        """
        inputs = self._tokenize_observation(batch)
        print(inputs.size())
        self.model.eval()
        output = self.model(inputs.cuda())
        #output = output[:,0,:]
        output = torch.transpose(output, 1, 2)
        pred_text = self._get_predictions(output)
        print("Input: " + batch.observations[0]["labels"][0])
        print("Prediction: " + pred_text[0])
        return Output(pred_text)

