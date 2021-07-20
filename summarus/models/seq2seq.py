from typing import Dict, Tuple

import torch
from torch.nn.modules.linear import Linear

from allennlp.data.vocabulary import Vocabulary
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder
from allennlp.models.model import Model
from allennlp.modules import Attention
from allennlp_models.generation.models.simple_seq2seq import SimpleSeq2Seq


@Model.register("seq2seq")
class Seq2Seq(SimpleSeq2Seq):
    def __init__(self,
                 vocab: Vocabulary,
                 source_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 max_decoding_steps: int,
                 attention: Attention = None,
                 beam_size: int = None,
                 target_namespace: str = "tokens",
                 target_embedding_dim: int = None,
                 scheduled_sampling_ratio: float = 0.,
                 use_projection: bool = False,
                 projection_dim: int = None,
                 tie_embeddings: bool = False) -> None:
        super(Seq2Seq, self).__init__(
            vocab=vocab,
            source_embedder=source_embedder,
            encoder=encoder,
            max_decoding_steps=max_decoding_steps,
            attention=attention,
            use_bleu=False,
            beam_size=beam_size,
            target_namespace=target_namespace,
            target_embedding_dim=target_embedding_dim,
            scheduled_sampling_ratio=scheduled_sampling_ratio
        )
        use_projection = use_projection or projection_dim is not None

        self._tie_embeddings = tie_embeddings

        if self._tie_embeddings:
            assert "token_embedder_tokens" in dict(self._source_embedder.named_children())
            source_token_embedder = dict(self._source_embedder.named_children())["token_embedder_tokens"]
            self._target_embedder.weight = source_token_embedder.weight

        num_classes = self.vocab.get_vocab_size(self._target_namespace)
        self._use_projection = use_projection
        if self._use_projection:
            self._projection_dim = projection_dim or self._source_embedder.get_output_dim()
            self._hidden_projection_layer = Linear(self._decoder_output_dim, self._projection_dim)
            self._output_projection_layer = Linear(self._projection_dim, num_classes)
        else:
            self._output_projection_layer = Linear(self._decoder_output_dim, num_classes)
        self._bleu = False

    def _prepare_output_projections(
        self,
        last_predictions: torch.Tensor,
        state: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:  # pylint: disable=line-too-long
        # shape: (group_size, max_input_sequence_length, encoder_output_dim)
        encoder_outputs = state["encoder_outputs"]
        # shape: (group_size, max_input_sequence_length)
        source_mask = state["source_mask"]
        # shape: (group_size, decoder_output_dim)
        decoder_hidden = state["decoder_hidden"]
        # shape: (group_size, decoder_output_dim)
        decoder_context = state["decoder_context"]
        # shape: (group_size, target_embedding_dim)
        embedded_input = self._target_embedder.forward(last_predictions)

        if self._attention:
            # shape: (group_size, encoder_output_dim)
            attended_input = self._prepare_attended_input(decoder_hidden, encoder_outputs, source_mask)
            # shape: (group_size, decoder_output_dim + target_embedding_dim)
            decoder_input = torch.cat((attended_input, embedded_input), -1)
        else:
            # shape: (group_size, target_embedding_dim)
            decoder_input = embedded_input

        # shape (decoder_hidden): (batch_size, decoder_output_dim)
        # shape (decoder_context): (batch_size, decoder_output_dim)
        decoder_hidden, decoder_context = self._decoder_cell(decoder_input, (decoder_hidden, decoder_context))

        state["decoder_hidden"] = decoder_hidden
        state["decoder_context"] = decoder_context

        # shape: (group_size, num_classes)
        if self._use_projection:
            output_projections = self._output_projection_layer(self._hidden_projection_layer(decoder_hidden))
        else:
            output_projections = self._output_projection_layer(decoder_hidden)

        return output_projections, state
