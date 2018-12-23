from typing import Dict, List, Tuple

import numpy
from overrides import overrides
import torch
import torch.nn.functional as F
from torch.nn.modules.linear import Linear
from torch.nn.modules.rnn import LSTMCell, LSTM
from torch.autograd import Variable

from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules import Attention, TextFieldEmbedder, Seq2SeqEncoder
from allennlp.models.model import Model
from allennlp.modules.token_embedders import Embedding
from allennlp.nn import util
from allennlp.nn.beam_search import BeamSearch
from allennlp.nn.util import get_lengths_from_binary_sequence_mask
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

seed = 1048596
numpy.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


class CustomAttention(torch.nn.Module):
    def __init__(self,
                 hidden_size: int):
        super(CustomAttention, self).__init__()

        self._hidden_size = hidden_size

        self.decoder_hidden_projection_layer = Linear(hidden_size * 2, hidden_size * 2)
        self.v = Linear(hidden_size * 2, 1, bias=False)

    def forward(self, decoder_state, encoder_outputs, encoder_feature, mask):
        batch_size, l, n = list(encoder_outputs.size())

        decoder_feature = self.decoder_hidden_projection_layer(decoder_state)
        decoder_feature = decoder_feature.unsqueeze(1).expand(batch_size, l, n).contiguous()
        decoder_feature = decoder_feature.view(-1, n)  # B * l x 2*hidden_dim
        encoder_feature = encoder_feature.contiguous().view(-1, n)

        features = encoder_feature + decoder_feature
        scores = self.v(F.tanh(features))
        scores = scores.view(-1, l)

        mask = mask.float()
        scores = F.softmax(scores, dim=1) * mask
        normalization_factor = scores.sum(1, keepdim=True)
        scores = scores / normalization_factor
        scores = scores.unsqueeze(1)  # B x 1 x l

        context = torch.bmm(scores, encoder_outputs)
        context = context.view(-1, self._hidden_size * 2)

        scores = scores.view(-1, l)  # B x l

        return context, scores


@Model.register("seq2seq")
class Seq2Seq(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 source_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 max_decoding_steps: int,
                 beam_size: int = None,
                 target_namespace: str = "tokens",
                 target_embedding_dim: int = None,
                 projection_dim: int = None,
                 tie_embeddings: bool = False) -> None:
        super(Seq2Seq, self).__init__(vocab)
        self._target_namespace = target_namespace
        self._tie_embeddings = tie_embeddings

        self._start_index = self.vocab.get_token_index(START_SYMBOL, self._target_namespace)
        self._end_index = self.vocab.get_token_index(END_SYMBOL, self._target_namespace)

        # Source embedder
        self._source_embedder = source_embedder
        assert "token_embedder_tokens" in dict(self._source_embedder.named_children())
        token_embedder = dict(self._source_embedder.named_children())["token_embedder_tokens"]

        # Encoder
        self._encoder = encoder
        self._encoder_output_dim = self._encoder.get_output_dim()
        self.reduce_h = Linear(self._encoder.get_output_dim(), self._encoder.get_output_dim() // 2)
        self.reduce_c = Linear(self._encoder.get_output_dim(), self._encoder.get_output_dim() // 2)
        self._feature_projection_layer = Linear(self._encoder.get_output_dim(),
            self._encoder.get_output_dim(), bias=False)

        # Target embedder
        num_classes = self.vocab.get_vocab_size(self._target_namespace)
        target_embedding_dim = target_embedding_dim or source_embedder.get_output_dim()
        self._target_embedder = Embedding(num_classes, target_embedding_dim)
        if self._tie_embeddings:
            assert "token_embedder_tokens" in dict(self._source_embedder.named_children())
            source_token_embedder = dict(self._source_embedder.named_children())["token_embedder_tokens"]
            self._target_embedder.weight = source_token_embedder.weight

        # Decoder
        self._decoder_input_dim = target_embedding_dim
        self._decoder_output_dim = self._encoder_output_dim // 2
        self._decoder_input_projection = Linear(self._decoder_output_dim * 2 + target_embedding_dim, self._decoder_input_dim)
        self._decoder_cell = LSTMCell(self._decoder_input_dim, self._decoder_output_dim)

        # Attention
        self._attention = CustomAttention(self._encoder_output_dim // 2)

        # Prediction
        self._projection_dim = projection_dim or self._source_embedder.get_output_dim()
        self._hidden_projection_layer = Linear(self._decoder_output_dim * 3, self._projection_dim)
        self._output_projection_layer = Linear(self._projection_dim, num_classes) 

        # Misc
        self._max_decoding_steps = max_decoding_steps
        self._beam_search = BeamSearch(self._end_index, max_steps=max_decoding_steps, beam_size=beam_size) if beam_size else None

    def take_step(self,
                  last_predictions: torch.Tensor,
                  state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # shape: (group_size, num_classes)
        output_projections, state = self._prepare_output_projections(last_predictions, state)

        # shape: (group_size, num_classes)
        class_log_probabilities = F.log_softmax(output_projections, dim=-1)

        return class_log_probabilities, state

    @overrides
    def forward(self,  # type: ignore
                source_tokens: Dict[str, torch.LongTensor],
                target_tokens: Dict[str, torch.LongTensor] = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        state = self._init_encoded_state(source_tokens)

        if target_tokens or not self._beam_search:
            return self._forward_loop(state, target_tokens=target_tokens)

        return self._forward_beam_search(state)

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        predicted_indices = output_dict["predictions"]
        if not isinstance(predicted_indices, numpy.ndarray):
            predicted_indices = predicted_indices.detach().cpu().numpy()
        all_predicted_tokens = []
        for indices in predicted_indices:
            if len(indices.shape) > 1:
                indices = indices[0]
            indices = list(indices)
            if self._end_index in indices:
                indices = indices[:indices.index(self._end_index)]
            predicted_tokens = [self.vocab.get_token_from_index(x, namespace=self._target_namespace)
                                for x in indices]
            all_predicted_tokens.append(predicted_tokens)
        output_dict["predicted_tokens"] = all_predicted_tokens
        return output_dict

    def _init_encoded_state(self, source_tokens: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # shape: (batch_size, max_input_sequence_length, encoder_input_dim)
        embedded_input = self._source_embedder(source_tokens)

        batch_size, _, _ = embedded_input.size()

        # shape: (batch_size, max_input_sequence_length)
        source_mask = util.get_text_field_mask(source_tokens)

        # shape: (batch_size, max_input_sequence_length, encoder_output_dim)
        encoder_outputs = self._encoder(embedded_input, source_mask)
        final_encoder_output = util.get_final_encoder_states(
            encoder_outputs, source_mask, _encoder.is_bidirectional())
        decoder_hidden = final_encoder_output
        decoder_hidden = F.relu(self.reduce_h(decoder_hidden))

        decoder_context = encoder_outputs.new_zeros(batch_size, self._decoder_output_dim)
        decoder_context = F.relu(self.reduce_c(decoder_context))

        encoder_outputs = encoder_outputs.contiguous()
        encoder_feature = self._feature_projection_layer(encoder_outputs)

        state = {
                "source_mask": source_mask,
                "encoder_outputs": encoder_outputs,
                "encoder_feature": encoder_feature,
                "decoder_hidden": decoder_hidden,
                "decoder_context": decoder_context
        }

        return state

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

        step_logits: List[torch.Tensor] = []
        step_probabilities: List[torch.Tensor] = []
        step_predictions: List[torch.Tensor] = []
        for timestep in range(num_decoding_steps):
            if not target_tokens:
                # shape: (batch_size,)
                input_choices = last_predictions
            else:
                # shape: (batch_size,)
                input_choices = targets[:, timestep]

            # shape: (batch_size, num_classes)
            output_projections, state = self._prepare_output_projections(input_choices, state)

            # list of tensors, shape: (batch_size, 1, num_classes)
            step_logits.append(output_projections.unsqueeze(1))

            # shape: (batch_size, num_classes)
            class_probabilities = F.softmax(output_projections, dim=-1)

            # list of tensors, shape: (batch_size, 1, num_classes)
            step_probabilities.append(class_probabilities.unsqueeze(1))

            # shape (predicted_classes): (batch_size,)
            _, predicted_classes = torch.max(class_probabilities, 1)

            # shape (predicted_classes): (batch_size,)
            last_predictions = predicted_classes

            # list of tensors, shape: (batch_size, 1)
            step_predictions.append(last_predictions.unsqueeze(1))

        # shape: (batch_size, num_decoding_steps, num_classes)
        logits = torch.cat(step_logits, 1)

        # shape: (batch_size, num_decoding_steps, num_classes)
        class_probabilities = torch.cat(step_probabilities, 1)

        # shape: (batch_size, num_decoding_steps)
        all_predictions = torch.cat(step_predictions, 1)

        output_dict = {
                "logits": logits,
                "class_probabilities": class_probabilities,
                "predictions": all_predictions,
        }

        # Compute loss.
        if target_tokens:
            target_mask = util.get_text_field_mask(target_tokens)
            loss = self._get_loss(logits, targets, target_mask)
            output_dict["loss"] = loss

        return output_dict

    def _forward_beam_search(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        batch_size = state["source_mask"].size(0)
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

        encoder_feature = state["encoder_feature"]
        # shape: (group_size, target_embedding_dim)
        embedded_input = self._target_embedder(last_predictions)

        if "attn_context" in state:
            context = state["attn_context"]
        else:
            context = decoder_context.new_zeros(batch_size, self._decoder_output_dim * 2)

        decoder_input = self._decoder_input_projection(torch.cat((context, embedded_input), 1))

        # shape (decoder_hidden): (batch_size, decoder_output_dim)
        # shape (decoder_context): (batch_size, decoder_output_dim)
        decoder_hidden, decoder_context = self._decoder_cell(decoder_input, (decoder_hidden, decoder_context))

        decoder_state = torch.cat((decoder_hidden.view(-1, self._decoder_output_dim),
                                   decoder_context.view(-1, self._decoder_output_dim)), 1)

        context, attn_scores = self._attention(decoder_state, encoder_outputs, encoder_feature, source_mask)
        output = torch.cat((decoder_hidden.view(-1, self._decoder_output_dim), context), 1)  # B x hidden_dim * 3
        output_projections = self._output_projection_layer(self._hidden_projection_layer(output))

        state["decoder_hidden"] = decoder_hidden
        state["decoder_context"] = decoder_context
        state["attn_context"] = context

        return output_projections, state

    @staticmethod
    def _get_loss(logits: torch.LongTensor,
                  targets: torch.LongTensor,
                  target_mask: torch.LongTensor) -> torch.Tensor:
        # shape: (batch_size, num_decoding_steps)
        relevant_targets = targets[:, 1:].contiguous()

        # shape: (batch_size, num_decoding_steps)
        relevant_mask = target_mask[:, 1:].contiguous()

        return util.sequence_cross_entropy_with_logits(logits, relevant_targets, relevant_mask)
