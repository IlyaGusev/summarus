from typing import Dict, Tuple, List
import numpy as np
from overrides import overrides

import torch
import torch.nn.functional as F
from torch.nn.modules.linear import Linear
from torch.nn.modules.rnn import LSTMCell
from torch.nn.functional import relu

from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.vocabulary import Vocabulary, DEFAULT_OOV_TOKEN
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder
from allennlp.models.model import Model
from allennlp.modules.token_embedders import Embedding
from allennlp.modules import Attention
from allennlp.nn.beam_search import BeamSearch
from allennlp.nn import util


@Model.register("pgn")
class PointerGeneratorNetwork(Model):
    """
    Based on https://arxiv.org/pdf/1704.04368.pdf
    """
    def __init__(self,
                 vocab: Vocabulary,
                 source_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 attention: Attention,
                 max_decoding_steps: int,
                 beam_size: int = None,
                 target_namespace: str = "tokens",
                 target_embedding_dim: int = None,
                 scheduled_sampling_ratio: float = 0.,
                 projection_dim: int = None,
                 use_coverage: bool = False,
                 coverage_shift: float = 0.,
                 coverage_loss_weight: float = None,
                 embed_attn_to_output: bool = False) -> None:
        super(PointerGeneratorNetwork, self).__init__(vocab)

        self._target_namespace = target_namespace
        self._start_index = self.vocab.get_token_index(START_SYMBOL, target_namespace)
        self._end_index = self.vocab.get_token_index(END_SYMBOL, target_namespace)
        self._unk_index = self.vocab.get_token_index(DEFAULT_OOV_TOKEN, target_namespace)
        self._vocab_size = self.vocab.get_vocab_size(target_namespace)
        assert self._vocab_size > 2, \
            "Target vocabulary is empty. Make sure 'target_namespace' option of the model is correct."

        # Encoder
        self._source_embedder = source_embedder
        self._encoder = encoder
        self._encoder_output_dim = self._encoder.get_output_dim()

        # Decoder
        self._target_embedding_dim = target_embedding_dim or source_embedder.get_output_dim()
        self._num_classes = self.vocab.get_vocab_size(target_namespace)
        self._target_embedder = Embedding(self._target_embedding_dim, self._num_classes)

        self._decoder_input_dim = self._encoder_output_dim + self._target_embedding_dim
        self._decoder_output_dim = self._encoder_output_dim
        self._decoder_cell = LSTMCell(self._decoder_input_dim, self._decoder_output_dim)

        self._projection_dim = projection_dim or self._source_embedder.get_output_dim()
        hidden_projection_dim = self._decoder_output_dim if not embed_attn_to_output else self._decoder_output_dim * 2
        self._hidden_projection_layer = Linear(hidden_projection_dim, self._projection_dim)
        self._output_projection_layer = Linear(self._projection_dim, self._num_classes)

        self._p_gen_layer = Linear(self._decoder_output_dim * 3 + self._decoder_input_dim, 1)
        self._attention = attention
        self._use_coverage = use_coverage
        self._coverage_loss_weight = coverage_loss_weight
        self._eps = 1e-31
        self._embed_attn_to_output = embed_attn_to_output
        self._coverage_shift = coverage_shift

        # Metrics
        self._p_gen_sum = 0.0
        self._p_gen_iterations = 0
        self._coverage_loss_sum = 0.0
        self._coverage_iterations = 0

        # Decoding
        self._scheduled_sampling_ratio = scheduled_sampling_ratio
        self._max_decoding_steps = max_decoding_steps
        self._beam_search = BeamSearch(self._end_index, max_steps=max_decoding_steps, beam_size=beam_size or 1)

    def forward(self,
                source_tokens: Dict[str, Dict[str, torch.LongTensor]],
                source_token_ids: torch.Tensor,
                source_to_target: torch.LongTensor,
                target_tokens: Dict[str, Dict[str, torch.LongTensor]] = None,
                target_token_ids: torch.Tensor = None,
                metadata=None) -> Dict[str, torch.Tensor]:
        state = self._encode(source_tokens)
        target_tokens_tensor = target_tokens["tokens"]["tokens"].long() if target_tokens else None
        extra_zeros, modified_source_tokens, modified_target_tokens = self._prepare(
            source_to_target, source_token_ids, target_tokens_tensor, target_token_ids)

        state["tokens"] = modified_source_tokens
        state["extra_zeros"] = extra_zeros

        output_dict = {}
        if target_tokens:
            state["target_tokens"] = modified_target_tokens
            state = self._init_decoder_state(state)
            output_dict = self._forward_loop(state, target_tokens)
        output_dict["metadata"] = metadata
        output_dict["source_to_target"] = source_to_target

        if not self.training:
            state = self._init_decoder_state(state)
            predictions = self._forward_beam_search(state)
            output_dict.update(predictions)

        return output_dict

    def _prepare(self,
                 source_tokens: torch.LongTensor,
                 source_token_ids: torch.Tensor,
                 target_tokens: torch.LongTensor = None,
                 target_token_ids: torch.Tensor = None):
        batch_size = source_tokens.size(0)
        source_max_length = source_tokens.size(1)

        tokens = source_tokens
        token_ids = source_token_ids.long()

        # Concat target tokens if exist
        if target_tokens is not None:
            tokens = torch.cat((tokens, target_tokens), 1)
            token_ids = torch.cat((token_ids, target_token_ids.long()), 1)

        is_unk = torch.eq(tokens, self._unk_index).long()
        # Create tensor with ids of unknown tokens only.
        # Those ids are batch-local.
        unk_only = token_ids * is_unk

        # Recalculate batch-local ids to range [1, count_of_unique_unk_tokens].
        # All known tokens have zero id.
        unk_token_nums = token_ids.new_zeros((batch_size, token_ids.size(1)))
        for i in range(batch_size):
            unique = torch.unique(unk_only[i, :], return_inverse=True, sorted=True)[1]
            unk_token_nums[i, :] = unique

        # Replace DEFAULT_OOV_TOKEN id with new batch-local ids starting from vocab_size
        # For example, if vocabulary size is 50000, the first unique unknown token will have 50000 index,
        # the second will have 50001 index and so on.
        tokens = tokens - tokens * is_unk + (self._vocab_size - 1) * is_unk + unk_token_nums

        modified_target_tokens = None
        modified_source_tokens = tokens
        if target_tokens is not None:
            # Remove target unknown tokens that do not exist in source tokens
            max_source_num = torch.max(tokens[:, :source_max_length], dim=1)[0]
            vocab_size = max_source_num.new_full((1,), self._vocab_size-1)
            max_source_num = torch.max(max_source_num, other=vocab_size).unsqueeze(1).expand((-1, tokens.size(1)))
            unk_target_tokens_mask = torch.gt(tokens, max_source_num).long()
            tokens = tokens - tokens * unk_target_tokens_mask + self._unk_index * unk_target_tokens_mask
            modified_target_tokens = tokens[:, source_max_length:]
            modified_source_tokens = tokens[:, :source_max_length]

        # Count unique unknown source tokens to create enough zeros for final distribution
        source_unk_count = torch.max(unk_token_nums[:, :source_max_length])
        extra_zeros = tokens.new_zeros((batch_size, source_unk_count), dtype=torch.float32)
        return extra_zeros, modified_source_tokens, modified_target_tokens

    def _encode(self, source_tokens: Dict[str, torch.LongTensor]) -> Dict[str, torch.Tensor]:
        # shape: (batch_size, max_input_sequence_length, encoder_input_dim)
        embedded_input = self._source_embedder.forward(source_tokens)
        # shape: (batch_size, max_input_sequence_length)
        source_mask = util.get_text_field_mask(source_tokens)
        # shape: (batch_size, max_input_sequence_length, encoder_output_dim)
        encoder_outputs = self._encoder.forward(embedded_input, source_mask)

        return {
                "source_mask": source_mask,
                "encoder_outputs": encoder_outputs,
        }

    def _init_decoder_state(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        batch_size = state["source_mask"].size(0)
        # shape: (batch_size, encoder_output_dim)
        final_encoder_output = util.get_final_encoder_states(
                state["encoder_outputs"],
                state["source_mask"],
                self._encoder.is_bidirectional())
        # Initialize the decoder hidden state with the final output of the encoder.
        # shape: (batch_size, decoder_output_dim)
        state["decoder_hidden"] = final_encoder_output

        encoder_outputs = state["encoder_outputs"]
        state["decoder_context"] = encoder_outputs.new_zeros(batch_size, self._decoder_output_dim)
        if self._embed_attn_to_output:
            state["attn_context"] = encoder_outputs.new_zeros(encoder_outputs.size(0), encoder_outputs.size(2))
        if self._use_coverage:
            state["coverage"] = encoder_outputs.new_zeros(batch_size, encoder_outputs.size(1))
        return state

    def _prepare_output_projections(self,
                                    last_predictions: torch.Tensor,
                                    state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # shape: (group_size, max_input_sequence_length, encoder_output_dim)
        encoder_outputs = state["encoder_outputs"]
        # shape: (group_size, max_input_sequence_length)
        source_mask = state["source_mask"]
        # shape: (group_size, decoder_output_dim)
        decoder_hidden = state["decoder_hidden"]
        # shape: (group_size, decoder_output_dim)
        decoder_context = state["decoder_context"]
        # shape: (group_size, decoder_output_dim)
        attn_context = state.get("attn_context", None)

        is_unk = (last_predictions >= self._vocab_size).long()
        last_predictions_fixed = last_predictions - last_predictions * is_unk + self._unk_index * is_unk
        embedded_input = self._target_embedder(last_predictions_fixed)

        coverage = state.get("coverage", None)

        def get_attention_context(decoder_hidden_inner):
            if coverage is None:
                attention_scores = self._attention(decoder_hidden_inner, encoder_outputs, source_mask)
            else:
                attention_scores = self._attention(decoder_hidden_inner, encoder_outputs, source_mask, coverage)
            attention_context = util.weighted_sum(encoder_outputs, attention_scores)
            return attention_scores, attention_context

        if not self._embed_attn_to_output:
            attn_scores, attn_context = get_attention_context(decoder_hidden)
            decoder_input = torch.cat((attn_context, embedded_input), -1)
            decoder_hidden, decoder_context = self._decoder_cell(decoder_input, (decoder_hidden, decoder_context))
            projection = self._hidden_projection_layer(decoder_hidden)
        else:
            decoder_input = torch.cat((attn_context, embedded_input), -1)
            decoder_hidden, decoder_context = self._decoder_cell(decoder_input, (decoder_hidden, decoder_context))
            attn_scores, attn_context = get_attention_context(decoder_hidden)
            projection = self._hidden_projection_layer(torch.cat((attn_context, decoder_hidden), -1))

        output_projections = self._output_projection_layer(projection)
        if self._use_coverage:
            state["coverage"] = coverage + attn_scores
        state["decoder_input"] = decoder_input
        state["decoder_hidden"] = decoder_hidden
        state["decoder_context"] = decoder_context
        state["attn_scores"] = attn_scores
        state["attn_context"] = attn_context

        return output_projections, state

    def _get_final_dist(self, state: Dict[str, torch.Tensor], output_projections):
        attn_dist = state["attn_scores"]
        tokens = state["tokens"]
        extra_zeros = state["extra_zeros"]
        attn_context = state["attn_context"]
        decoder_input = state["decoder_input"]
        decoder_hidden = state["decoder_hidden"]
        decoder_context = state["decoder_context"]

        decoder_state = torch.cat((decoder_hidden, decoder_context), 1)
        p_gen = self._p_gen_layer(torch.cat((attn_context, decoder_state, decoder_input), 1))
        p_gen = torch.sigmoid(p_gen)
        self._p_gen_sum += torch.mean(p_gen).item()
        self._p_gen_iterations += 1

        vocab_dist = F.softmax(output_projections, dim=-1)

        vocab_dist = vocab_dist * p_gen
        attn_dist = attn_dist * (1.0 - p_gen)
        if extra_zeros.size(1) != 0:
            vocab_dist = torch.cat((vocab_dist, extra_zeros), 1)
        final_dist = vocab_dist.scatter_add(1, tokens, attn_dist)
        normalization_factor = final_dist.sum(1, keepdim=True)
        final_dist = final_dist / normalization_factor

        return final_dist

    def _forward_loop(self,
                      state: Dict[str, torch.Tensor],
                      target_tokens: Dict[str, Dict[str, torch.LongTensor]] = None) -> Dict[str, torch.Tensor]:
        # shape: (batch_size, max_input_sequence_length)
        source_mask = state["source_mask"]
        batch_size = source_mask.size(0)

        num_decoding_steps = self._max_decoding_steps
        if target_tokens:
            # shape: (batch_size, max_target_sequence_length)
            targets = target_tokens["tokens"]["tokens"]
            _, target_sequence_length = targets.size()
            num_decoding_steps = target_sequence_length - 1

        if self._use_coverage:
            coverage_loss = source_mask.new_zeros(1, dtype=torch.float32)

        last_predictions = state["tokens"].new_full((batch_size,), fill_value=self._start_index)
        step_proba: List[torch.Tensor] = []
        step_predictions: List[torch.Tensor] = []
        for timestep in range(num_decoding_steps):
            if self.training and torch.rand(1).item() < self._scheduled_sampling_ratio:
                input_choices = last_predictions
            elif not target_tokens:
                input_choices = last_predictions
            else:
                input_choices = targets[:, timestep]

            if self._use_coverage:
                old_coverage = state["coverage"]

            output_projections, state = self._prepare_output_projections(input_choices, state)
            final_dist = self._get_final_dist(state, output_projections)
            step_proba.append(final_dist)
            last_predictions = torch.max(final_dist, 1)[1]
            step_predictions.append(last_predictions.unsqueeze(1))

            if self._use_coverage:
                step_coverage_loss = torch.sum(torch.min(state["attn_scores"], old_coverage), 1)
                coverage_loss = coverage_loss + step_coverage_loss

        # shape: (batch_size, num_decoding_steps)
        predictions = torch.cat(step_predictions, 1)

        output_dict = {"predictions": predictions}

        if target_tokens:
            # shape: (batch_size, num_decoding_steps, num_classes)
            num_classes = step_proba[0].size(1)
            proba = step_proba[0].new_zeros((batch_size, num_classes, len(step_proba)))
            for i, p in enumerate(step_proba):
                proba[:, :, i] = p

            loss = self._get_loss(proba, state["target_tokens"], self._eps)
            if self._use_coverage:
                coverage_loss = torch.mean(coverage_loss / num_decoding_steps)
                self._coverage_loss_sum += coverage_loss.item()
                self._coverage_iterations += 1
                modified_coverage_loss = relu(coverage_loss - self._coverage_shift) + self._coverage_shift - 1.0
                loss = loss + self._coverage_loss_weight * modified_coverage_loss
            output_dict["loss"] = loss

        return output_dict

    @staticmethod
    def _get_loss(proba: torch.LongTensor,
                  targets: torch.LongTensor,
                  eps: float) -> torch.Tensor:
        targets = targets[:, 1:]
        proba = torch.log(proba + eps)
        loss = torch.nn.NLLLoss(ignore_index=0)(proba, targets)
        return loss

    def _forward_beam_search(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        batch_size = state["tokens"].size()[0]
        start_predictions = state["tokens"].new_full((batch_size,), fill_value=self._start_index)

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
        log_probabilities = torch.log(final_dist + self._eps)
        return log_probabilities, state

    @overrides
    def make_output_human_readable(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        predicted_indices = output_dict["predictions"]
        if not isinstance(predicted_indices, np.ndarray):
            predicted_indices = predicted_indices.detach().cpu().numpy()
        all_predicted_tokens = []
        all_meta = output_dict["metadata"]
        all_source_to_target = output_dict["source_to_target"]
        for (indices, metadata), source_to_target in zip(zip(predicted_indices, all_meta), all_source_to_target):
            all_predicted_tokens.append(self._decode_sample(indices, metadata, source_to_target))
        output_dict["predicted_tokens"] = all_predicted_tokens
        return output_dict

    def _decode_sample(self, indices, metadata, source_to_target):
        all_predicted_tokens = []
        if len(indices.shape) == 1:
            indices = [indices]
        for sample_indices in indices:
            sample_indices = list(sample_indices)
            # Collect indices till the first end_symbol
            if self._end_index in sample_indices:
                sample_indices = sample_indices[:sample_indices.index(self._end_index)]
            # Get all unknown tokens from source
            original_source_tokens = metadata["source_tokens"]
            unk_tokens = list()
            for i, token_vocab_index in enumerate(source_to_target):
                if token_vocab_index != self._unk_index:
                    continue
                token = original_source_tokens[i]
                if token in unk_tokens:
                    continue
                unk_tokens.append(token)
            predicted_tokens = []
            for token_vocab_index in sample_indices:
                if token_vocab_index < self._vocab_size:
                    token = self.vocab.get_token_from_index(token_vocab_index, namespace=self._target_namespace)
                else:
                    unk_number = token_vocab_index - self._vocab_size
                    token = unk_tokens[unk_number]
                predicted_tokens.append(token)
            all_predicted_tokens.append(predicted_tokens)
        return all_predicted_tokens

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        if not self._use_coverage:
            return {}
        avg_coverage_loss = 0.0
        if self._coverage_iterations != 0:
            avg_coverage_loss = self._coverage_loss_sum / self._coverage_iterations
        avg_p_gen = self._p_gen_sum / self._p_gen_iterations if self._p_gen_iterations != 0 else 0.0
        metrics = {"coverage_loss": avg_coverage_loss, "p_gen": avg_p_gen}
        if reset:
            self._p_gen_sum = 0.0
            self._p_gen_iterations = 0
            self._coverage_loss_sum = 0.0
            self._coverage_iterations = 0
        return metrics
