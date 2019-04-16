from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch.nn.modules.linear import Linear
from torch.nn.modules.rnn import LSTMCell

from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder
from allennlp.models.model import Model
from allennlp.modules.token_embedders import Embedding
from allennlp.nn.beam_search import BeamSearch
from allennlp.models.encoder_decoders.simple_seq2seq import SimpleSeq2Seq
from allennlp.nn import util

class CustomAttention(torch.nn.Module):
    def __init__(self,
                 hidden_size: int):
        super(CustomAttention, self).__init__()

        self._hidden_size = hidden_size

        self.decoder_hidden_projection_layer = Linear(hidden_size * 2, hidden_size, bias=False)
        self.encoder_outputs_projection_layer = Linear(hidden_size, hidden_size, bias=False)
        self.v = Linear(hidden_size, 1, bias=False)

    def forward(self, decoder_state, encoder_outputs, mask):
        batch_size, l, n = list(encoder_outputs.size())

        decoder_feature = self.decoder_hidden_projection_layer(decoder_state)
        decoder_feature = decoder_feature.unsqueeze(1).expand(batch_size, l, self._hidden_size).contiguous()
        decoder_feature = decoder_feature.view(-1, self._hidden_size)  # B * l x 2*hidden_dim

        encoder_feature = self.encoder_outputs_projection_layer(encoder_outputs.contiguous())
        encoder_feature = encoder_feature.contiguous().view(-1, self._hidden_size)

        features = encoder_feature + decoder_feature
        scores = self.v(torch.tanh(features))
        scores = scores.view(-1, l)

        mask = mask.float()
        scores = F.softmax(scores, dim=1) * mask
        normalization_factor = scores.sum(1, keepdim=True)
        scores = scores / normalization_factor
        context = util.weighted_sum(encoder_outputs, scores)
        
        return context, scores


@Model.register("seq2seq")
class Seq2Seq(SimpleSeq2Seq):
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
        if self._tie_embeddings:
            assert "token_embedder_tokens" in dict(self._source_embedder.named_children())
            source_token_embedder = dict(self._source_embedder.named_children())["token_embedder_tokens"]
            self._target_embedder.weight = source_token_embedder.weight

        # Decoder
        self._decoder_input_dim = target_embedding_dim
        self._decoder_input_projection = Linear(self._decoder_output_dim + target_embedding_dim,
                                                self._decoder_input_dim)
        self._decoder_cell = LSTMCell(self._decoder_input_dim, self._decoder_output_dim)

        # Attention
        self._attention = CustomAttention(self._decoder_output_dim)

        # Prediction
        self._projection_dim = projection_dim or self._source_embedder.get_output_dim()
        self._hidden_projection_layer = Linear(self._decoder_output_dim * 2, self._projection_dim)
        self._output_projection_layer = Linear(self._projection_dim, num_classes)

        # Misc
        self._max_decoding_steps = max_decoding_steps
        self._beam_search = BeamSearch(self._end_index, max_steps=max_decoding_steps, beam_size=beam_size) if beam_size else None

        self._bleu = False

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
        embedded_input = self._target_embedder.forward(last_predictions)

        if "attn_context" in state:
            context = state["attn_context"]
        else:
            context = decoder_context.new_zeros(batch_size, self._decoder_output_dim)

        decoder_input = self._decoder_input_projection(torch.cat((context, embedded_input), 1))

        # shape (decoder_hidden): (batch_size, decoder_output_dim)
        # shape (decoder_context): (batch_size, decoder_output_dim)
        decoder_hidden, decoder_context = self._decoder_cell(decoder_input, (decoder_hidden, decoder_context))

        decoder_state = torch.cat((decoder_hidden.view(-1, self._decoder_output_dim),
                                   decoder_context.view(-1, self._decoder_output_dim)), 1)

        context, attn_scores = self._attention.forward(decoder_state, encoder_outputs, source_mask)
        output = torch.cat((decoder_hidden.view(-1, self._decoder_output_dim), context), 1)  # B x hidden_dim
        output_projections = self._output_projection_layer(self._hidden_projection_layer(output))

        state["decoder_hidden"] = decoder_hidden
        state["decoder_context"] = decoder_context
        state["attn_context"] = context

        return output_projections, state

