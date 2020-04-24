from typing import Dict, List, Tuple, Optional
from overrides import overrides

import numpy
import torch
import torch.nn.functional as F
from torch.nn import Linear

from allennlp.common.checks import ConfigurationError
from allennlp.common.util import END_SYMBOL, START_SYMBOL
from allennlp.modules.seq2seq_decoders.seq_decoder import SeqDecoder
from allennlp.data import Vocabulary
from allennlp.modules import Embedding
from allennlp.modules.seq2seq_decoders.decoder_net import DecoderNet
from allennlp.nn import util
from allennlp.nn.beam_search import BeamSearch
from allennlp.training.metrics import Metric


@SeqDecoder.register("custom_auto_regressive_seq_decoder")
class CustomAutoRegressiveSeqDecoder(SeqDecoder):
    def __init__(
            self,
            vocab: Vocabulary,
            decoder_net: DecoderNet,
            max_decoding_steps: int,
            target_embedder: Embedding,
            target_namespace: str = "tokens",
            tie_output_embedding: bool = False,
            scheduled_sampling_ratio: float = 0,
            label_smoothing_ratio: Optional[float] = None,
            beam_size: int = 4,
            tensor_based_metric: Metric = None,
            token_based_metric: Metric = None,
    ) -> None:
        super().__init__(target_embedder)

        self._vocab = vocab

        self._decoder_net = decoder_net
        self._max_decoding_steps = max_decoding_steps
        self._target_namespace = target_namespace
        self._label_smoothing_ratio = label_smoothing_ratio

        self._start_index = self._vocab.get_token_index(START_SYMBOL, self._target_namespace)
        self._end_index = self._vocab.get_token_index(END_SYMBOL, self._target_namespace)
        self._beam_search = BeamSearch(self._end_index, max_steps=max_decoding_steps, beam_size=beam_size)

        target_vocab_size = self._vocab.get_vocab_size(self._target_namespace)

        if self.target_embedder.get_output_dim() != self._decoder_net.target_embedding_dim:
            raise ConfigurationError("Target Embedder output_dim doesn't match decoder module's input.")

        self._output_projection_layer = Linear(self._decoder_net.get_output_dim(), target_vocab_size)

        if tie_output_embedding:
            if self._output_projection_layer.weight.shape != self.target_embedder.weight.shape:
                raise ConfigurationError("Can't tie embeddings with output linear layer, due to shape mismatch")
            self._output_projection_layer.weight = self.target_embedder.weight

        self._tensor_based_metric = tensor_based_metric
        self._token_based_metric = token_based_metric
        self._scheduled_sampling_ratio = scheduled_sampling_ratio

    def _forward_beam_search(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        batch_size = state["source_mask"].size()[0]
        start_predictions = state["source_mask"].new_full((batch_size,), fill_value=self._start_index)

        # shape (all_top_k_predictions): (batch_size, beam_size, num_decoding_steps)
        # shape (log_probabilities): (batch_size, beam_size)
        all_top_k_predictions, log_probabilities = self._beam_search.search(start_predictions, state, self.take_step)

        output_dict = {
                "class_log_probabilities": log_probabilities,
                "predictions": all_top_k_predictions,
        }
        return output_dict

    def _forward_loss(self, state: Dict[str, torch.Tensor],
                      target_tokens: Dict[str, torch.LongTensor]) -> Dict[str, torch.Tensor]:
        # shape: (batch_size, max_input_sequence_length, encoder_output_dim)
        encoder_outputs = state["encoder_outputs"]

        # shape: (batch_size, max_input_sequence_length)
        source_mask = state["source_mask"]

        # shape: (batch_size, max_target_sequence_length)
        targets = target_tokens["tokens"]

        # Prepare embeddings for targets. They will be used as gold embeddings during decoder training
        # shape: (batch_size, max_target_sequence_length, embedding_dim)
        target_embedding = self.target_embedder(targets)

        # shape: (batch_size, max_target_batch_sequence_length)
        target_mask = util.get_text_field_mask(target_tokens)

        if self._scheduled_sampling_ratio == 0 and self._decoder_net.decodes_parallel:
            _, decoder_output = self._decoder_net(previous_state=state,
                                                  previous_steps_predictions=target_embedding[:, :-1, :],
                                                  encoder_outputs=encoder_outputs,
                                                  source_mask=source_mask,
                                                  previous_steps_mask=target_mask[:, :-1])

            # shape: (group_size, max_target_sequence_length, num_classes)
            logits = self._output_projection_layer(decoder_output)
        else:
            batch_size = source_mask.size()[0]
            _, target_sequence_length = targets.size()

            # The last input from the target is either padding or the end symbol.
            # Either way, we don't have to process it.
            num_decoding_steps = target_sequence_length - 1

            # Initialize target predictions with the start index.
            # shape: (batch_size,)
            last_predictions = source_mask.new_full((batch_size,), fill_value=self._start_index)

            # shape: (steps, batch_size, target_embedding_dim)
            steps_embeddings = torch.Tensor([])

            step_logits: List[torch.Tensor] = []

            for timestep in range(num_decoding_steps):
                if self.training and torch.rand(1).item() < self._scheduled_sampling_ratio:
                    # Use gold tokens at test time and at a rate of 1 - _scheduled_sampling_ratio
                    # during training.
                    # shape: (batch_size, steps, target_embedding_dim)
                    state['previous_steps_predictions'] = steps_embeddings

                    # shape: (batch_size, )
                    effective_last_prediction = last_predictions
                else:
                    # shape: (batch_size, )
                    effective_last_prediction = targets[:, timestep]

                    if timestep == 0:
                        state['previous_steps_predictions'] = torch.Tensor([])
                    else:
                        # shape: (batch_size, steps, target_embedding_dim)
                        state['previous_steps_predictions'] = target_embedding[:, :timestep]

                # shape: (batch_size, num_classes)
                output_projections, state = self._prepare_output_projections(effective_last_prediction, state)

                # list of tensors, shape: (batch_size, 1, num_classes)
                step_logits.append(output_projections.unsqueeze(1))

                # shape (predicted_classes): (batch_size,)
                _, predicted_classes = torch.max(output_projections, 1)

                # shape (predicted_classes): (batch_size,)
                last_predictions = predicted_classes

                # shape: (batch_size, 1, target_embedding_dim)
                last_predictions_embeddings = self.target_embedder(last_predictions).unsqueeze(1)

                # This step is required, since we want to keep up two different prediction history: gold and real
                if steps_embeddings.shape[-1] == 0: # pylint: disable=unsubscriptable-object
                    # There is no previous steps, except for start vectors in ``last_predictions``
                    # shape: (group_size, 1, target_embedding_dim)
                    steps_embeddings = last_predictions_embeddings
                else:
                    # shape: (group_size, steps_count, target_embedding_dim)
                    steps_embeddings = torch.cat([steps_embeddings, last_predictions_embeddings], 1)

            # shape: (batch_size, num_decoding_steps, num_classes)
            logits = torch.cat(step_logits, 1)

        # Compute loss.
        target_mask = util.get_text_field_mask(target_tokens)
        loss = self._get_loss(logits, targets, target_mask)

        # TODO: We will be using beam search to get predictions for validation, but if beam size in 1
        # we could consider taking the last_predictions here and building step_predictions
        # and use that instead of running beam search again, if performance in validation is taking a hit
        output_dict = {
                'loss': loss
        }

        return output_dict

    def _prepare_output_projections(
            self,
            last_predictions: torch.Tensor,
            state: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Decode current state and last prediction to produce produce projections
        into the target space, which can then be used to get probabilities of
        each target token for the next step.

        Inputs are the same as for `take_step()`.
        """
        # shape: (group_size, max_input_sequence_length, encoder_output_dim)
        encoder_outputs = state["encoder_outputs"]

        # shape: (group_size, max_input_sequence_length)
        source_mask = state["source_mask"]

        # shape: (group_size, steps_count, decoder_output_dim)
        previous_steps_predictions = state.get("previous_steps_predictions")

        # shape: (batch_size, 1, target_embedding_dim)
        last_predictions_embeddings = self.target_embedder(last_predictions).unsqueeze(1)

        if previous_steps_predictions is None or previous_steps_predictions.shape[-1] == 0:
            # There is no previous steps, except for start vectors in ``last_predictions``
            # shape: (group_size, 1, target_embedding_dim)
            previous_steps_predictions = last_predictions_embeddings
        else:
            # shape: (group_size, steps_count, target_embedding_dim)
            previous_steps_predictions = torch.cat([previous_steps_predictions, last_predictions_embeddings], 1)

        decoder_state, decoder_output = self._decoder_net(previous_state=state,
                                                          encoder_outputs=encoder_outputs,
                                                          source_mask=source_mask,
                                                          previous_steps_predictions=previous_steps_predictions)
        state["previous_steps_predictions"] = previous_steps_predictions

        # Update state with new decoder state, override previous state
        state.update(decoder_state)

        if self._decoder_net.decodes_parallel:
            decoder_output = decoder_output[:, -1, :]

        # shape: (group_size, num_classes)
        output_projections = self._output_projection_layer(decoder_output)

        return output_projections, state

    def _get_loss(self,
                  logits: torch.LongTensor,
                  targets: torch.LongTensor,
                  target_mask: torch.LongTensor) -> torch.Tensor:
        # shape: (batch_size, num_decoding_steps)
        relevant_targets = targets[:, 1:].contiguous()

        # shape: (batch_size, num_decoding_steps)
        relevant_mask = target_mask[:, 1:].contiguous()

        return util.sequence_cross_entropy_with_logits(logits,
                                                       relevant_targets,
                                                       relevant_mask,
                                                       label_smoothing=self._label_smoothing_ratio)

    def get_output_dim(self):
        return self._decoder_net.get_output_dim()

    def take_step(self,
                  last_predictions: torch.Tensor,
                  state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # shape: (group_size, num_classes)
        output_projections, state = self._prepare_output_projections(last_predictions, state)

        # shape: (group_size, num_classes)
        class_log_probabilities = F.log_softmax(output_projections, dim=-1)

        return class_log_probabilities, state

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        all_metrics: Dict[str, float] = {}
        if not self.training:
            if self._tensor_based_metric is not None:
                all_metrics.update(self._tensor_based_metric.get_metric(reset=reset))  # type: ignore
            if self._token_based_metric is not None:
                all_metrics.update(self._token_based_metric.get_metric(reset=reset))  # type: ignore
        return all_metrics

    @overrides
    def forward(self,
                encoder_out: Dict[str, torch.LongTensor],
                target_tokens: Dict[str, torch.LongTensor] = None) -> Dict[str, torch.Tensor]:
        state = encoder_out
        decoder_init_state = self._decoder_net.init_decoder_state(state)
        state.update(decoder_init_state)

        output_dict = self._forward_loss(state, target_tokens) if target_tokens else {}

        if not self.training:
            predictions = self._forward_beam_search(state)
            output_dict.update(predictions)

            if target_tokens:
                if self._tensor_based_metric is not None:
                    # shape: (batch_size, beam_size, max_sequence_length)
                    top_k_predictions = output_dict["predictions"]
                    # shape: (batch_size, max_predicted_sequence_length)
                    best_predictions = top_k_predictions[:, 0, :]
                    # shape: (batch_size, target_sequence_length)

                    self._tensor_based_metric(best_predictions, target_tokens["tokens"])  # type: ignore

                if self._token_based_metric is not None:
                    output_dict = self.decode(output_dict)
                    predicted_tokens = output_dict['predicted_tokens']

                    self._token_based_metric(predicted_tokens,  # type: ignore
                                             [y.text for y in target_tokens["tokens"][1:-1]])

        return output_dict

    @overrides
    def post_process(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        predicted_indices = output_dict["predictions"]
        if not isinstance(predicted_indices, numpy.ndarray):
            predicted_indices = predicted_indices.detach().cpu().numpy()
        all_predicted_tokens = []
        for indices in predicted_indices:
            # Beam search gives us the top k results for each source sentence in the batch
            # but we just want the single best.
            if len(indices.shape) > 1:
                indices = indices[0]
            indices = list(indices)
            # Collect indices till the first end_symbol
            if self._end_index in indices:
                indices = indices[:indices.index(self._end_index)]
            predicted_tokens = [self._vocab.get_token_from_index(x, namespace=self._target_namespace)
                                for x in indices]
            all_predicted_tokens.append(predicted_tokens)
        output_dict["predicted_tokens"] = all_predicted_tokens
        return output_dict