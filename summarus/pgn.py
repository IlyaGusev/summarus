from typing import Dict, Tuple, List
import numpy as np

import torch
import torch.nn.functional as F
from torch.nn.modules.linear import Linear
from torch.nn.modules.rnn import LSTMCell

from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.vocabulary import Vocabulary, DEFAULT_OOV_TOKEN
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder
from allennlp.models.model import Model
from allennlp.modules.token_embedders import Embedding
from allennlp.modules import Attention
from allennlp.nn.beam_search import BeamSearch
from allennlp.models.encoder_decoders.simple_seq2seq import SimpleSeq2Seq
from allennlp.nn import util


@Model.register("pgn")
class PGN(SimpleSeq2Seq):
    def __init__(self,
                 vocab: Vocabulary,
                 source_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 max_decoding_steps: int,
                 attention: Attention = None,
                 beam_size: int = None,
                 target_namespace: str = "tokens",
                 target_embedding_dim: int = None,
                 projection_dim: int = None,
                 tie_embeddings: bool = False) -> None:
        super(SimpleSeq2Seq, self).__init__(vocab)
        self._target_namespace = target_namespace
        self._tie_embeddings = tie_embeddings
        self._scheduled_sampling_ratio = 0.

        self._start_index = self.vocab.get_token_index(START_SYMBOL, self._target_namespace)
        self._end_index = self.vocab.get_token_index(END_SYMBOL, self._target_namespace)

        # Source embedder
        self._source_embedder = source_embedder

        # Encoder
        self._encoder = encoder
        self._encoder_output_dim = self._encoder.get_output_dim()
        self._decoder_output_dim = self._encoder_output_dim
        self.reduce_h = Linear(self._encoder_output_dim, self._decoder_output_dim)
        self.reduce_c = Linear(self._encoder_output_dim, self._decoder_output_dim)

        # Target embedder
        num_classes = self.vocab.get_vocab_size(self._target_namespace)
        target_embedding_dim = target_embedding_dim or source_embedder.get_output_dim()
        self._target_embedder = Embedding(num_classes, target_embedding_dim)

        # Decoder
        self._decoder_input_dim = target_embedding_dim
        self._decoder_input_projection = Linear(self._decoder_output_dim + target_embedding_dim,
                                                self._decoder_input_dim)
        self._decoder_cell = LSTMCell(self._decoder_input_dim, self._decoder_output_dim)

        # Attention
        self._attention = attention

        # Prediction
        self._projection_dim = projection_dim or self._source_embedder.get_output_dim()
        self._hidden_projection_layer = Linear(self._decoder_output_dim * 2, self._projection_dim)
        self._output_projection_layer = Linear(self._projection_dim, num_classes)

        # Misc
        self._max_decoding_steps = max_decoding_steps
        self._beam_search = BeamSearch(self._end_index, max_steps=max_decoding_steps, beam_size=beam_size) if beam_size else None

        self._bleu = False

        self._p_gen_layer = Linear(self._decoder_output_dim * 3 + self._decoder_input_dim, 1)

    def forward(self,  # type: ignore
                source_tokens: Dict[str, torch.LongTensor],
                source_token_ids: torch.Tensor,
                target_tokens: Dict[str, torch.LongTensor] = None,
                target_token_ids: torch.Tensor = None,
                source_to_target=None,
                metadata=None) -> Dict[str, torch.Tensor]:
        state = self._encode(source_tokens)

        extra_zeros, tokens, modified_target_tokens = self._pgn_prepare(source_tokens, source_token_ids,
                                                                        target_tokens, target_token_ids)

        state["tokens"] = tokens
        state["extra_zeros"] = extra_zeros

        if target_tokens:
            state["target_tokens"] = modified_target_tokens
            state = self._init_decoder_state(state)
            output_dict = self._forward_loop(state, target_tokens)
        else:
            output_dict = {}

        output_dict["metadata"] = metadata
        output_dict["source_tokens"] = source_tokens["tokens"]

        if not self.training:
            state = self._init_decoder_state(state)
            predictions = self._forward_beam_search(state)
            output_dict.update(predictions)

        return output_dict

    def _pgn_prepare(self,
                     source_tokens: Dict[str, torch.LongTensor],
                     source_token_ids: torch.Tensor,
                     target_tokens: Dict[str, torch.LongTensor] = None,
                     target_token_ids: torch.Tensor = None):
        tokens = source_tokens["tokens"].long()
        token_ids = source_token_ids.long()
        batch_size = tokens.size(0)
        source_max_length = tokens.size(1)
        if target_tokens:
            tokens = torch.cat((tokens, target_tokens["tokens"]), 1)
            token_ids = torch.cat((token_ids, target_token_ids.long()), 1)

        vocab_unk_index = self.vocab.get_token_index(DEFAULT_OOV_TOKEN)
        is_unk_token = torch.eq(tokens, vocab_unk_index).long()
        tokens = tokens - tokens * is_unk_token + (self.vocab.get_vocab_size() - 1) * is_unk_token
        unk_token_nums = token_ids.new_zeros((token_ids.size(0), token_ids.size(1)))
        unk_only = token_ids * is_unk_token
        for i in range(batch_size):
            unique = torch.unique(unk_only[i, :],
                                  return_inverse=True, sorted=True)[1]
            unk_token_nums[i, :] = unique
        tokens += unk_token_nums

        modified_target_tokens = None
        if target_tokens:
            for i in range(batch_size):
                max_source_num = self.vocab.get_vocab_size() - 1 + torch.max(unk_token_nums[i, :source_max_length])
                unk_target_tokens_mask = torch.gt(tokens[i, :], max_source_num).long()

                tokens[i, :] += -tokens[i, :] * unk_target_tokens_mask + vocab_unk_index * unk_target_tokens_mask
            modified_target_tokens = tokens[:, source_max_length:]
            modified_source_tokens = tokens[:, :source_max_length]
        else:
            modified_source_tokens = tokens

        extra_zeros = unk_token_nums.new_zeros((unk_token_nums.size(0), torch.max(unk_token_nums[:, :source_max_length]))).float()
        return extra_zeros, modified_source_tokens, modified_target_tokens

    def _prepare_output_projections(self,
                                    last_predictions: torch.Tensor,
                                    state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:  # pylint: disable=line-too-long
        # shape: (group_size, max_input_sequence_length, encoder_output_dim)
        encoder_outputs = state["encoder_outputs"]
        # shape: (group_size, max_input_sequence_length)
        source_mask = state["source_mask"]
        batch_size = source_mask.size()[0]
        # shape: (group_size, decoder_output_dim)
        decoder_hidden = state["decoder_hidden"]
        # shape: (group_size, decoder_output_dim)
        decoder_context = state["decoder_context"]

        # shape: (group_size, target_embedding_dim)
        is_unk = (last_predictions >= self.vocab.get_vocab_size(self._target_namespace)).long()
        vocab_unk_index = self.vocab.get_token_index(DEFAULT_OOV_TOKEN)
        last_predictions_fixed = last_predictions + vocab_unk_index * is_unk - last_predictions*is_unk
        embedded_input = self._target_embedder.forward(last_predictions_fixed)

        if "attn_context" in state:
            context = state["attn_context"]
        else:
            context = decoder_context.new_zeros(batch_size, self._decoder_output_dim)

        decoder_input = self._decoder_input_projection(torch.cat((context, embedded_input), 1))
        decoder_hidden, decoder_context = self._decoder_cell(decoder_input, (decoder_hidden, decoder_context))

        attn_scores = self._attention.forward(decoder_hidden, encoder_outputs, source_mask)
        context = util.weighted_sum(encoder_outputs, attn_scores)
        output = torch.cat((decoder_hidden.view(-1, self._decoder_output_dim), context), 1)  # B x hidden_dim
        output_projections = self._output_projection_layer(self._hidden_projection_layer(output))

        state["decoder_input"] = decoder_input
        state["decoder_hidden"] = decoder_hidden
        state["decoder_context"] = decoder_context
        state["attn_context"] = context
        state["attn_scores"] = attn_scores

        return output_projections, state

    def _forward_beam_search(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Make forward pass during prediction using a beam search."""
        batch_size = state["source_mask"].size()[0]
        start_predictions = state["source_mask"].new_full((batch_size,), fill_value=self._start_index)

        # shape (all_top_k_predictions): (batch_size, beam_size, num_decoding_steps)
        # shape (log_probabilities): (batch_size, beam_size)
        all_top_k_predictions, log_probabilities = self._beam_search.search(
            start_predictions, state, self.take_step)

        output_dict = {
            "class_log_probabilities": log_probabilities,
            "predictions": all_top_k_predictions,
        }
        return output_dict

    def take_step(self,
                  last_predictions: torch.Tensor,
                  state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # shape: (group_size, num_classes)
        output_projections, state = self._prepare_output_projections(last_predictions, state)
        final_dist = self._get_final_dist(state, output_projections)

        return torch.log(final_dist), state

    def _get_final_dist(self, state: Dict[str, torch.Tensor], output_projections):
        attn_scores = state["attn_scores"]
        tokens = state["tokens"]
        extra_zeros = state["extra_zeros"]

        # shape: (group_size, num_classes)
        decoder_state = torch.cat((state["decoder_hidden"].view(-1, self._decoder_output_dim),
                                   state["decoder_context"].view(-1, self._decoder_output_dim)), 1)
        p_gen = self._p_gen_layer(torch.cat((state["attn_context"], decoder_state, state["decoder_input"]), 1))
        p_gen = torch.sigmoid(p_gen)
        vocab_dist = F.softmax(output_projections, dim=-1)
        vocab_dist = vocab_dist * p_gen
        attn_dist = attn_scores * (1 - p_gen)
        vocab_dist = torch.cat((vocab_dist, extra_zeros), 1)
        final_dist = vocab_dist.scatter_add(1, tokens, attn_dist)

        return final_dist

    def _forward_loop(self,
                      state: Dict[str, torch.Tensor],
                      target_tokens: Dict[str, torch.LongTensor] = None) -> Dict[str, torch.Tensor]:
        # shape: (batch_size, max_input_sequence_length)
        source_mask = state["source_mask"]

        batch_size = source_mask.size()[0]

        if target_tokens:
            # shape: (batch_size, max_target_sequence_length)
            targets = target_tokens["tokens"]

            _, target_sequence_length = targets.size()

            # The last input from the target is either padding or the end symbol.
            # Either way, we don't have to process it.
            num_decoding_steps = target_sequence_length - 1
        else:
            num_decoding_steps = self._max_decoding_steps

        # Initialize target predictions with the start index.
        # shape: (batch_size,)
        last_predictions = source_mask.new_full((batch_size,), fill_value=self._start_index)

        step_proba: List[torch.Tensor] = []
        step_predictions: List[torch.Tensor] = []
        for timestep in range(num_decoding_steps):
            if self.training and torch.rand(1).item() < self._scheduled_sampling_ratio:
                # Use gold tokens at test time and at a rate of 1 - _scheduled_sampling_ratio
                # during training.
                # shape: (batch_size,)
                input_choices = last_predictions
            elif not target_tokens:
                # shape: (batch_size,)
                input_choices = last_predictions
            else:
                # shape: (batch_size,)
                input_choices = targets[:, timestep]

            # shape: (batch_size, num_classes)
            output_projections, state = self._prepare_output_projections(input_choices, state)
            final_dist = self._get_final_dist(state, output_projections)
            if torch.isinf(final_dist).sum() != 0 or torch.isnan(final_dist).sum() != 0:
                raise ValueError("bad final dist")
            # final_dist = F.softmax(output_projections, dim=1)
            step_proba.append(final_dist)

            # list of tensors, shape: (batch_size, 1, num_classes)
            # step_logits.append(output_projections.unsqueeze(1))

            # shape (predicted_classes): (batch_size,)
            _, predicted_classes = torch.max(final_dist, 1)

            # shape (predicted_classes): (batch_size,)
            last_predictions = predicted_classes

            step_predictions.append(last_predictions.unsqueeze(1))

        # shape: (batch_size, num_decoding_steps)
        predictions = torch.cat(step_predictions, 1)

        output_dict = {"predictions": predictions}

        if target_tokens:
            # shape: (batch_size, num_decoding_steps, num_classes)
            proba = step_proba[0].new_zeros((batch_size, len(step_proba), step_proba[0].size(1)))
            for i, p in enumerate(step_proba):
                proba[:, i, :] = p

            # Compute loss.
            target_mask = util.get_text_field_mask(target_tokens)
            target_tokens = state["target_tokens"]
            # target_tokens = target_tokens["tokens"]
            loss = self._get_loss(proba, target_tokens, target_mask)
            output_dict["loss"] = loss

        return output_dict

    @staticmethod
    def _get_loss(proba: torch.LongTensor,
                  targets: torch.LongTensor,
                  target_mask: torch.LongTensor) -> torch.Tensor:
        targets = targets[:, 1:]
        proba = torch.log(proba.transpose(1, 2))
        loss = torch.nn.NLLLoss(ignore_index=0)(proba, targets)
        return loss

    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        predicted_indices = output_dict["predictions"]
        if not isinstance(predicted_indices, np.ndarray):
            predicted_indices = predicted_indices.detach().cpu().numpy()
        all_predicted_tokens = []
        for (indices, metadata), source_tokens in zip(zip(predicted_indices, output_dict["metadata"]), output_dict["source_tokens"]):
            # Beam search gives us the top k results for each source sentence in the batch
            # but we just want the single best.
            if len(indices.shape) > 1:
                indices = indices[0]
            indices = list(indices)
            # Collect indices till the first end_symbol
            if self._end_index in indices:
                indices = indices[:indices.index(self._end_index)]
            predicted_tokens = []
            for x in indices:
                if x < self.vocab.get_vocab_size():
                    token = self.vocab.get_token_from_index(x, namespace=self._target_namespace)
                else:
                    unk_number = x - self.vocab.get_vocab_size()
                    vocab_unk_index = self.vocab.get_token_index(DEFAULT_OOV_TOKEN)
                    unk_index = 0
                    result = 0
                    for i, t in enumerate(source_tokens):
                        if t == vocab_unk_index:
                            if unk_index == unk_number:
                                result = i
                                break
                            unk_index += 1
                    token = metadata["source_tokens"][result]
                predicted_tokens.append(token)
            all_predicted_tokens.append(predicted_tokens)
        output_dict["predicted_tokens"] = all_predicted_tokens
        return output_dict
