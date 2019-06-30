#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""A self-feeding chatbot (a chatbot wrapped with an NRC and QGen)

A self-feeding chatbot consists of three components:
- dialog agent: this is a typical model for handlin a conversastion(e.g. Transformer)
- feedback classifier: this classifies incoming user responses
- question generator: when the feedback classifier predicts a sufficiently negative
    feedback score, this module asks for an explanation from the from the user.
"""

import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from parlai.core.torch_agent import TorchAgent, Output
from parlai.core.utils import round_sigfigs, warn_once, padded_tensor

from parlai.agents.transformer.transformer import TransformerRankerAgent

from .feedback_classifier.feedback_classifier import (
    FeedbackClassifierRegex,
)
from .modules import SelfFeedingModel
from .utils import add_person_tokens

EPS = 1e-9

# Status options
NORMAL = 1
EXPLANATION_REQUESTED = 2
NEWTOPIC_REQUESTED = 3
RATING_REQUESTED = 4
RATING_ACCEPTED = 5

# Hardcoded responses
RAT_REQUEST = ("Just checking: I'm not sure how positive/negative the response you "
               "just gave was. Could you help me out? (Write positive, negative, or "
               "neutral)")
CONTINUE = "And in response to what you were saying before"
EXP_REQUEST = ("Oops! I think I messed up. Whether I messed up or not, what could I "
               "have said (in response to <response>)?")
THANKS = "Thanks! I'll try to remember that."
NEWTOPIC = "Can you pick a new topic for us to talk about now?"


class SelfFeedingAgent(TransformerRankerAgent):
    @classmethod
    def add_cmdline_args(cls, argparser):
        """Add command-line arguments specifically for this agent."""
        super().add_cmdline_args(argparser)
        SelfFeedingModel.add_cmdline_args(argparser)

        agent = argparser.add_argument_group('Self-feeding Agent')
        agent.add_argument('--request-rating', type='bool', default=False,
                           help="If True, ocassionally request ratings")
        agent.add_argument('--rating-frequency', type=float, default=0.01,
                           help="The fraction of the time that a rating will be "
                           "randomly requested, regardless of classifier confidence")
        agent.add_argument('--rating-gap', type=float, default=0.05,
                           help="A positivity rating must be within this amount of "
                           "the rating-threshold to trigger the confidence-based "
                           "rating request")

        agent.add_argument('--request-explanation', type='bool', default=False,
                           help="If True, recognize mistakes and request explanations")
        agent.add_argument('--rating-threshold', type=float, default=0.5,
                           help="Treat feedback below this threshold as negative")
        agent.add_argument('--target-class', type=int, default=0,
                           help="The label (in [0,1]) to treat as the target rarer "
                           "class when calculating tp/fp/tn/fn")

        agent.add_argument('--freeze-base', type='bool', default=False,
                           help="If True, freeze all but the sentiment linear layer")
        agent.add_argument('--partial-load', type='bool', default=False,
                           help="If True, allow a model to be partially loaded (i.e., "
                           "a model need not be a perfect match to be loaded")

        agent.add_argument('-subtasks', '--subtasks', type=str,
                           help="A comma-separated list of active subtasks")

        agent.add_argument('--dia-weight', type=float, default=1.0,
                           help="The dialog task loss is multiplied by this")
        agent.add_argument('--exp-weight', type=float, default=1.0,
                           help="The explanation task loss is multiplied by this")
        agent.add_argument('--sen-weight', type=float, default=1.0,
                           help="The sentiment task loss is multiplied by this")
        agent.add_argument(
            '--prev-response-negatives', type='bool', default=False,
            help="If True, during training add the previous agent response as a "
                 "negative candidate for the current response (as a way to "
                 "organically teach the model to not repeat itself). This option is "
                 "ignored when candidate set is 'fixed' or 'vocab', as these already "
                 "include all candidates")
        agent.add_argument(
            '--prev-response-filter', type='bool', default=False,
            help="If True and --interactive=True, do not allow the model to output its "
                 "previous response. This is a hackier solution than using "
                 "--prev-response-negatives, but MUCH faster/simpler")
        agent.add_argument(
            '--interactive', type='bool', default=False,
            help="Mark as true if you are in a setting where you are only doing "
                 "evaluation, and always with the same fixed candidate set.")

        variants = argparser.add_argument_group('Self-feeding Variants')
        variants.add_argument('-rgx', '--regex', type='bool', default=False,
                              help="If True, classify sentiment using regexes instead "
                              "of model")
        variants.add_argument('-up', '--uncertainty-predictor', type='bool',
                              default=False,
                              help="If True, classify sentiment using uncertainty of "
                              "dialog models predictions instead of classifer"
                              "model")
        variants.add_argument('-ut', '--uncertainty-threshold', type=float, default=0.5,
                              help="If model confidence is smaller than this number and"
                              " --uncertainty-predictor=True, classify as bot mistake")
        variants.add_argument('-us', '--uncertainty-style', type=str, default='gap',
                              choices=['gap', 'mag'],
                              help="Whether the uncertainty threshold measures the "
                              "magnitude of the top confidence, or the gap between the "
                              "two most confident answers")
        return agent

    def __init__(self, opt, shared=None):
        if opt['interactive']:
            print("[ Setting interactive mode defaults... ]")
            opt['prev_response_filter'] = True
            opt['person_tokens'] = True

        # Set subtasks first so that opt['subtasks'] is set before build_model()
        self.set_subtasks(opt)
        self.multitask = len(self.subtasks) > 1
        if not self.multitask:
            self.subtask = self.subtasks[0]

        if opt['prev_response_negatives']:
            if opt['candidates'] in ['fixed', 'vocab']:
                msg = ("[ Option --prev-response-negatives=True is incompatible with "
                       "--candidates=['fixed','vocab']. Overriding it to False. ]")
                warn_once(msg)
                self.opt['prev_response_negatives'] = False
            self.prev_responses = None
        if opt['prev_response_filter']:
            if not opt['interactive']:
                msg = ("[ Option --prev-response-filter=True can only be used when "
                       "--interactive=True. Overriding it to False. ]")
                warn_once(msg)
                self.opt['prev_response_filter'] = False
            self.prev_response = None

        super().__init__(opt, shared)

        self.status = NORMAL
        if self.opt['interactive']:
            assert('dialog' in self.subtasks)
            assert('sentiment' in self.subtasks)
        else:
            assert(not self.opt['request_explanation'])
            assert(not self.opt['request_rating'])

        self.task_weight = {
            'dialog': opt['dia_weight'],
            'explanation': opt['exp_weight'],
            'sentiment': opt['sen_weight'],
        }

        # dialog/explanation tasks use self.rank_loss from TorchRankerAgent
        # Don't do BCEWithLogitsLoss since we need the probs from the sigmoid anyway
        self.sentiment_criterion = nn.BCELoss(reduce=True, size_average=False)

        # Set rating classifier
        if opt['regex']:
            self.rating_classifier = FeedbackClassifierRegex()

        random.seed()
        self.reset()

    # NOTE: This is the only method of TransformerAgent being overwritten
    def build_model(self):
        self.model = SelfFeedingModel(self.opt, self.dict)
        if self.opt['embedding_type'] != 'random':
            for embeddings in [getattr(self.model, 'dia_embeddings', None),
                               getattr(self.model, 'exp_embeddings', None),
                               getattr(self.model, 'sen_embeddings', None)]:
                if embeddings is not None:
                    self._copy_embeddings(embeddings.weight, self.opt['embedding_type'])
        return self.model

    def act(self):
        if self.opt['request_explanation'] or self.opt['request_rating']:
            # rating request ->
            #   0/+1 -> normal
            #   -1 -> explanation request
            # explanation request -> topic request -> normal
            positivity = -1  # actual positivity values are in [0, 1]

            # Preprocessing (for special states that transition to NORMAL)
            if self.status in [NEWTOPIC_REQUESTED, RATING_ACCEPTED]:
                self.status = NORMAL

            if self.status == NORMAL:
                positivity = self.predict_sentiment(self.observation)
                # print(f"[ positivity ]: {positivity}")
                if self.do_request_explanation(positivity):
                    action = self.make_action(self.make_explanation_request())
                    self.status = EXPLANATION_REQUESTED
                elif self.do_request_rating(positivity):
                    action = self.make_action(self.make_rating_request())
                    self.requested_rating = True
                    self.status = RATING_REQUESTED

            elif self.status == RATING_REQUESTED:
                rating = self.extract_rating()
                if rating == -1:
                    action = self.make_action(
                        self.make_explanation_request(), reward=rating)
                    self.status = EXPLANATION_REQUESTED

                else:
                    action, reply = self.make_rating_response(rating)
                    self.status = RATING_ACCEPTED
                    # Store just the non-template part of the response in the history
                    self.history.update_history({"text": reply})

            elif self.status == EXPLANATION_REQUESTED:
                action = self.make_action(THANKS + ' ' + NEWTOPIC)
                self.status = NEWTOPIC_REQUESTED
                self.history.reset()

            else:
                raise Exception(f"Unrecognized status: {self.status}")

        if self.status == NORMAL:
            action = super().act()
            self.history.update_history(action)

        return action

    def observe(self, observation):
        """Add to history, concatenate history-size utterances, and add person tokens"""
        # If their response is a response to a rating request, no work required
        if self.status == RATING_REQUESTED:
            self.last_rating = observation['text']
        else:
            if 'text' in observation:
                self.history.update_history(observation)

        if len(self.history.history_strings) > 0:
            # WARNING: Be suspicious of this; is history_size being applied correctly?
            observation['text'] = add_person_tokens(
                self.history.history_strings[-self.opt['history_size']:], last_speaker=1)

        self.observation = observation

        if observation.get('episode_done', True):
            self.history.reset()

        return self.vectorize(self.observation, self.history)

    def batchify(self, observations):
        batch = super().batchify(observations)
        if self.multitask and batch.text_vec is not None:
            # Dialog is the default task; e.g., in interactive mode
            subtasks = [o.get('subtask', 'dialog') for o in observations]
            # Catch error here for debugging
            if len(set(subtasks)) > 1:
                import pdb
                pdb.set_trace()
            self.subtask = subtasks[0]
        # print(f"[ context ]: {observations[0]['text']}")
        return batch

    def train_step(self, batch):
        """Train on a single batch of examples."""
        import pdb; pdb.set_trace()
        if batch.text_vec is None:
            return

        self.model.train()
        self.optimizer.zero_grad()

        batchsize = batch.text_vec.size(0)
        self.metrics['examples'] += batchsize

        if self.subtask == 'dialog':
            loss, preds, _ = self.dialog_step(batch)
        elif self.subtask == 'explanation':
            loss, preds, _ = self.explanation_step(batch)
        elif self.subtask == 'sentiment':
            loss, preds = self.sentiment_step(batch)
            preds = [str(p) for p in preds]

        # Weight loss by task-weight
        loss *= self.task_weight[self.subtask]

        loss.backward()
        self.update_params()

        return Output(preds)

    def eval_step(self, batch):
        if batch.text_vec is None:
            return

        self.model.eval()

        batchsize = batch.text_vec.size(0)
        self.metrics['examples'] += batchsize

        if self.subtask == 'dialog':
            loss, preds, cand_ranked = self.dialog_step(batch)
            if self.opt['interactive']:
                if self.opt['prev_response_filter']:
                    preds = self.check_prev_response(preds, cand_ranked)
                return Output(preds)
            else:
                return Output(preds, cand_ranked)

        elif self.subtask == 'explanation':
            loss, preds, cand_ranked = self.explanation_step(batch)
            return Output(preds, cand_ranked)

        elif self.subtask == 'sentiment':
            if self.opt['uncertainty_predictor']:
                # Use uncertainty of dialog model to classify bot's previous utterance
                preds = self.predict_sentiment_by_uncertainty(batch)
            else:
                # Use sentiment of user response to classify bot's previous response
                loss, preds = self.sentiment_step(batch)
            preds = [str(p) for p in preds]
            return Output(preds)

    def dialog_step(self, batch):
        batchsize = batch.text_vec.size(0)
        if self.model.training:
            cands, cand_vecs, label_inds = self._build_candidates(
                batch, source=self.opt['candidates'], mode='train')
        else:
            cands, cand_vecs, label_inds = self._build_candidates(
                batch, source=self.opt['eval_candidates'], mode='eval')

        scores = self.model.score_dialog(batch.text_vec, cand_vecs)
        _, ranks = scores.sort(1, descending=True)

        if self.model.training:
            # Get predictions but not full rankings for the sake of speed
            cand_ranked = None
            if cand_vecs.dim() == 2:
                preds = [cands[ordering[0]] for ordering in ranks]
            elif cand_vecs.dim() == 3:
                preds = [cands[i][ordering[0]] for i, ordering in enumerate(ranks)]
        else:
            # Return full rankings to calculate hits@ metrics
            cand_ranked = []
            for i, ordering in enumerate(ranks):
                if cand_vecs.dim() == 2:
                    cand_list = cands
                elif cand_vecs.dim() == 3:
                    cand_list = cands[i]
                cand_ranked.append([cand_list[rank] for rank in ordering])
            preds = [cand_ranked[i][0] for i in range(batchsize)]

        if label_inds is None:
            loss = None
        else:
            loss = self.rank_loss(scores, label_inds)
            self.update_dia_metrics(loss, ranks, label_inds, batchsize)
        return loss, preds, cand_ranked

    def explanation_step(self, batch):
        batchsize = batch.text_vec.size(0)

        warn_once("WARNING: explanation candidates are hardcoded to batch")
        if self.model.training:
            cands, cand_vecs, label_inds = self._build_candidates(
                batch, source='batch', mode='train')
        else:
            cands, cand_vecs, label_inds = self._build_candidates(
                batch, source='batch', mode='eval')

        scores = self.model.score_explanation(batch.text_vec, cand_vecs)
        _, ranks = scores.sort(1, descending=True)

        if self.model.training:
            # Get predictions but not full rankings for the sake of speed
            cand_ranked = None
            preds = [cands[ordering[0]] for ordering in ranks]
        else:
            # Return full rankings to calculate hits@ metrics
            cand_ranked = []
            for i, ordering in enumerate(ranks):
                cand_ranked.append([cands[rank] for rank in ordering])
            preds = [cand_ranked[i][0] for i in range(batchsize)]

        if label_inds is None:
            loss = None
        else:
            loss = self.rank_loss(scores, label_inds)
            self.update_exp_metrics(loss, ranks, label_inds, batchsize)
        return loss, preds, cand_ranked

    def sentiment_step(self, batch):
        batchsize = batch.text_vec.size(0)
        # probs is a [batchsize] torch.FloatTensor with values in [0, 1]
        if self.opt['regex']:
            contexts = [self.dict.vec2txt(vec) for vec in batch.text_vec]
            # HACK: --use-cuda flag
            probs = self.rating_classifier.predict_proba(contexts).cuda()
        else:
            probs = self.model.score_sentiment(batch.text_vec)
        # preds is a [batchsize] torch.LongTensor with values in {0, 1}
        preds = (probs > self.opt['rating_threshold']).long()

        if batch.labels is None:  # TODO: not sure if it's None, list of Nones, etc.
            loss = None
        else:
            # labels will be a [batchsize] torch.LongTensor with values in {0, 1}
            labels = torch.LongTensor([int(l) == 1 for l in batch.labels]).cuda()

            loss = self.sentiment_criterion(probs, labels.float())
            self.update_sen_metrics(loss, preds, labels, batchsize)

        return loss, preds

    def update_dia_metrics(self, loss, ranks, label_inds, batchsize):
        self.metrics['dia_exs'] += batchsize
        self.metrics['dia_loss'] += loss.item()
        if label_inds is not None:
            for b in range(batchsize):
                rank = (ranks[b] == label_inds[b]).nonzero().item()
                self.metrics['dia_rank'] += 1 + rank
                self.metrics['dia_correct'] += (rank == 0)

    def update_exp_metrics(self, loss, ranks, label_inds, batchsize):
        self.metrics['exp_exs'] += batchsize
        self.metrics['exp_loss'] += loss.item()
        if label_inds is not None:
            for b in range(batchsize):
                rank = (ranks[b] == label_inds[b]).nonzero().item()
                self.metrics['exp_rank'] += 1 + rank
                self.metrics['exp_correct'] += (rank == 0)

    def update_sen_metrics(self, loss, preds, labels, batchsize):
        # tp/fp/tn/fn are w/r/t the rarer class (negative responses)
        self.metrics['sen_exs'] += batchsize
        self.metrics['sen_loss'] += loss.item()
        a = self.opt['target_class']
        b = not self.opt['target_class']
        assert(a in [0, 1])
        assert(b in [0, 1])
        self.metrics['sen_tp'] += ((preds == labels) * (labels == a)).sum().item()
        self.metrics['sen_fp'] += ((preds != labels) * (labels == b)).sum().item()
        self.metrics['sen_tn'] += ((preds == labels) * (labels == b)).sum().item()
        self.metrics['sen_fn'] += ((preds != labels) * (labels == a)).sum().item()

    def reset(self):
        super().reset()
        self.requested_rating = False

    def reset_metrics(self):
        """Reset metrics."""
        super().reset_metrics()
        # self.metrics['examples'] = 0
        if 'dialog' in self.subtasks:
            self.metrics['dia_exs'] = 0
            self.metrics['dia_loss'] = 0.0
            self.metrics['dia_rank'] = 0
            self.metrics['dia_correct'] = 0
        if 'explanation' in self.subtasks:
            self.metrics['exp_exs'] = 0
            self.metrics['exp_loss'] = 0.0
            self.metrics['exp_rank'] = 0
            self.metrics['exp_correct'] = 0.0
        if 'sentiment' in self.subtasks:
            self.metrics['sen_exs'] = 0
            self.metrics['sen_loss'] = 0
            self.metrics['sen_tp'] = 0
            self.metrics['sen_fp'] = 0
            self.metrics['sen_tn'] = 0
            self.metrics['sen_fn'] = 0

    def report(self):
        """Report metrics from model's perspective."""
        m = TorchAgent.report(self)  # Skip TorchRankerAgent; totally redundant
        # examples = self.metrics['examples']
        examples = self.metrics.get('exs', 0)
        if examples > 0:
            m['examples'] = examples
            if 'dialog' in self.subtasks and self.metrics['dia_exs'] > 0:
                m['dia_loss'] = self.metrics['dia_loss'] / self.metrics['dia_exs']
                m['dia_rank'] = self.metrics['dia_rank'] / self.metrics['dia_exs']
                m['dia_acc'] = self.metrics['dia_correct'] / self.metrics['dia_exs']
                m['dia_exs'] = self.metrics['dia_exs']
            if 'explanation' in self.subtasks and self.metrics['exp_exs'] > 0:
                m['exp_loss'] = self.metrics['exp_loss'] / self.metrics['exp_exs']
                m['exp_rank'] = self.metrics['exp_rank'] / self.metrics['exp_exs']
                m['exp_acc'] = self.metrics['exp_correct'] / self.metrics['exp_exs']
                m['exp_exs'] = self.metrics['exp_exs']
                m['exp_exs'] = self.metrics['exp_exs']
            if 'sentiment' in self.subtasks and self.metrics['sen_exs'] > 0:
                tp = self.metrics['sen_tp']
                tn = self.metrics['sen_tn']
                fp = self.metrics['sen_fp']
                fn = self.metrics['sen_fn']
                assert(tp + tn + fp + fn == self.metrics['sen_exs'])
                m['sen_loss'] = self.metrics['sen_loss'] / self.metrics['sen_exs']
                m['sen_pr'] = tp / (tp + fp + EPS)
                m['sen_re'] = tp / (tp + fn + EPS)
                pr = m['sen_pr']
                re = m['sen_re']
                m['sen_f1'] = (2 * pr * re) / (pr + re) if (pr and re) else 0.0
                m['sen_acc'] = (tp + tn) / self.metrics['sen_exs']
                m['sen_exs'] = self.metrics['sen_exs']
        for k, v in m.items():
            # clean up: rounds to sigfigs and converts tensors to floats
            if isinstance(v, float):
                m[k] = round_sigfigs(v, 4)
            else:
                m[k] = v
        return m

    def do_request_explanation(self, positivity):
        """Decide whether to request an explanation this turn"""
        # If --request-explanation=False, then don't request an explanation
        if not self.opt['request_explanation'] or len(self.history.history_strings) < 3:
            return False
        else:
            return positivity < self.opt['rating_threshold']

    def do_request_rating(self, positivity):
        """Decide whether to request a rating this turn"""
        # If --request-rating=False, then don't request a rating
        if not self.opt['request_rating']:
            return False
        # If no previous bot utterance is on record, can't ask about it
        elif len(self.history.history_strings) < 2:
            return False
        # Only request a rating a maximum of once per episode
        elif self.requested_rating:
            return False
        # Some percentage of the time, we ask for a rating randomly (to aid exploration)
        elif random.random() < self.opt['rating_frequency']:
            return True
        # Lastly, ask for a rating based on confidence (proximity to threshold)
        else:
            gap = abs(positivity - self.opt['rating_threshold'])
            return gap < self.opt['rating_gap']

    def extract_rating(self):
        """Convert user response to rating request from text to an integer rating"""
        # TODO: Make this more robust!
        if self.last_rating == 'positive':
            return 1
        elif self.last_rating == 'negative':
            return -1
        else:
            return 0

    def predict_sentiment(self, observation):
        if self.opt['regex']:
            prob = self.rating_classifier.predict_proba(observation['text_vec'])
        else:
            prob = self.model.score_sentiment(observation['text_vec'].reshape(1, -1))
        return prob.item()

    def predict_sentiment_by_uncertainty(self, batch):
        # Use the dialog model's confidence to predict the rating on previous response
        # HACK: this test is run using a model that was trained on dialog but is now
        # being evaluated on sentiment. We use a sentiment dataset so that we have
        # access to the sentiment labels. Therefore, we do a sloppy hack here to pull
        # out what would have been the candidates for the dialog task. We use histsz=4
        # and ignore the last response (the user's feedback) and use the penultimate
        # utterances as the candidates. The (up to) two utterances before that are
        # context.

        # pull out dialog candidates from text_vecs since this is a sentiment task
        assert(self.opt['history_size'] > 2)
        text_vecs = []
        cand_vecs = []
        for vec in batch.text_vec:
            last_p1 = (vec == self.dict.txt2vec('__p1__')[0]).nonzero()[-1].item()
            last_p2 = (vec == self.dict.txt2vec('__p2__')[0]).nonzero()[-1].item()
            text_vecs.append(vec[:last_p2])
            cand_vecs.append(vec[last_p2 + 1:last_p1])
        text_padded, _ = padded_tensor(text_vecs)
        cand_padded, _ = padded_tensor(cand_vecs)
        scores = self.model.score_dialog(text_padded, cand_padded)
        confidences = F.softmax(scores, dim=1).cpu().detach().numpy()

        preds = []
        for example in confidences:
            ranked_confidences = sorted(list(example), reverse=True)
            if self.opt['uncertainty_style'] == 'mag':
                # If the most confident choice isn't confident enough, predict that
                # the response the bot gives will be bad (pred=0)
                mag = ranked_confidences[0]
                preds.append(mag > self.opt['uncertainty_threshold'])
            elif self.opt['uncertainty_style'] == 'gap':
                # If the gap between the first and second most confident choices isn't
                # large enough, predict that the response the bot gives will be bad
                # (pred=0)
                gap = ranked_confidences[0] - ranked_confidences[1]
                preds.append(gap > self.opt['uncertainty_threshold'])

        loss = torch.tensor(0)
        preds = torch.LongTensor(preds)
        labels = torch.LongTensor([int(l) == 1 for l in batch.labels])
        batchsize = len(labels)
        self.update_sen_metrics(loss, preds, labels, batchsize)
        return preds

    def check_prev_response(self, preds, cand_ranked):
        # Compare current prediction to previous (replacing if necessary)
        if self.prev_response and (preds[0] == self.prev_response):
            preds[0] = cand_ranked[0][1]
        # Save current prediction for next turn
        self.prev_response = preds[0]
        return preds

    def make_action(self, text, reward=0):
        action = {
            'id': self.id,
            'text': text,
        }
        if reward:
            action['reward'] = reward
        return action

    def make_explanation_request(self):
        orig_prompt = self.history.history_strings[-3]
        return (f'Oops! I think I messed up. '
                f'Whether I messed up or not, what could I have said '
                f'(in response to "{orig_prompt}")?')

    def make_rating_request(self):
        # last_response = self.history[-2]
        # return (f'Just checking: did my last response ("{last_response}") make sense '
        #         'in the conversation? (Choose one of: [yes, no, maybe])')
        return RAT_REQUEST

    def make_rating_response(self, rating):
        action = super().act()
        reply = str(action['text'])
        action['reward'] = rating
        last_message = self.history.history_strings[-1]
        if rating == 0:
            action['text'] = f'Okay, thanks! {CONTINUE} ("{last_message}"): {reply}'
        elif rating == 1:
            action['text'] = f'Great, thanks! {CONTINUE} ("{last_message}"): {reply}'
        return action, reply

    def set_subtasks(self, opt):
        # Find assigned subtasks
        if opt.get('subtasks', None):
            # First check if it was passed explicitly as a flag or otherwise in opt
            if isinstance(opt['subtasks'], list):
                subtasks = opt['subtasks']
            else:
                subtasks = opt['subtasks'].split(',')
        else:
            # Otherwise, try to infer from the task name.
            subtasks = [task.split(':')[-2] for task in opt['task'].split(',')]

        # Expand any abbreviations
        if subtasks[0] == 'diaexp':
            subtasks = ['dialog', 'explanation']
        elif subtasks[0] == 'diasen':
            subtasks = ['dialog', 'sentiment']
        elif subtasks[0] == 'all':
            subtasks = ['dialog', 'explanation', 'sentiment']

        self.subtasks = subtasks
        opt['subtasks'] = subtasks  # Add to opt so that model module can see the list

    def load(self, path):
        """Return opt and model states.

        Overriding TorchAgent.load() to enable partial loading
        """
        states = torch.load(path, map_location=lambda cpu, _: cpu)
        if 'model' in states:
            try:
                self.model.load_state_dict(states['model'])
            except RuntimeError as e:
                if self.opt['partial_load']:
                    print("WARNING: could not load entire --init-model; "
                          "loading partially instead.")
                    pretrained_state = states['model']
                    current_state = self.model.state_dict()
                    # 1. filter out unnecessary keys
                    pretrained_dict = {
                        k: v for k, v in pretrained_state.items() if k in current_state}
                    # 2. overwrite entries in the existing state dict
                    current_state.update(pretrained_dict)
                    # 3. load the new state dict
                    self.model.load_state_dict(current_state)
                else:
                    raise Exception(f"The designated model could not be loaded. "
                                    f"Consider using --partial-load=True.\n {e}")

        if 'optimizer' in states and hasattr(self, 'optimizer'):
            self.optimizer.load_state_dict(states['optimizer'])
        return states

    def _add_prev_responses(self, batch, cands, cand_vecs, label_inds, source):
        assert(source not in ['fixed', 'vocab'])
        self._extract_prev_responses(batch)

        # Add prev_responses as negatives
        prev_cands = self.prev_responses
        prev_cand_vecs = [torch.Tensor(self.dict.txt2vec(cand)) for cand in prev_cands]
        if source == 'batch':
            # Option 1: Change from one set of shared candidates to separate per example
            # cands = [cands + [prev_cand] for prev_cand in prev_cands]
            # list_of_lists_of_cand_vecs = [[vec for vec in cand_vecs] + [prev_cand_vec]
            #                            for prev_cand_vec in prev_cand_vecs]
            # cand_vecs = padded_3d(list_of_lists_of_cand_vecs, use_cuda=self.use_cuda,
            #                       dtype=cand_vecs[0].dtype)

            # Option 2: Just add all prev responses for the whole batch (doubles bs)
            cands += prev_cands
            cand_vecs, _ = padded_tensor([vec for vec in cand_vecs] + prev_cand_vecs,
                                         use_cuda=self.use_cuda)
        elif source == 'inline':
            raise NotImplementedError

        return cands, cand_vecs

    def _extract_prev_responses(self, batch):
        # Extract prev_responses for self-feeding formatted examples
        warn_once("WARNING: This code is specific to self-feeding formatted examples")

        # TODO: Pull out p1/p2 once elsewhere, not every time
        p1 = self.dict.txt2vec('__p1__')[0]
        p2 = self.dict.txt2vec('__p2__')[0]
        self.prev_responses = []
        # Do naively for now with a for loop
        for text_vec in batch.text_vec:
            p1s = (text_vec == p1).nonzero()
            p2s = (text_vec == p2).nonzero()
            if len(p1s) and len(p2s):
                response_vec = text_vec[p2s[-1] + 1:p1s[-1]]
            else:
                response_vec = [self.NULL_IDX]  # TODO: pull in actual N
            response = self.dict.vec2txt(response_vec)
            self.prev_responses.append(response)

    def _build_candidates(self, batch, source, mode):
        cands, cand_vecs, label_inds = super()._build_candidates(batch, source, mode)
        if self.opt['prev_response_negatives'] and mode == 'train':
            cands, cand_vecs = self._add_prev_responses(
                batch, cands, cand_vecs, label_inds, source)
        return cands, cand_vecs, label_inds
