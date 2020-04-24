from typing import Dict

import torch
from torch.nn.modules.linear import Linear

from allennlp.data.vocabulary import Vocabulary
from allennlp.modules import TextFieldEmbedder, Seq2VecEncoder, Seq2SeqEncoder
from allennlp.models.model import Model
from allennlp.nn import util


@Model.register("summarunner")
class SummaRuNNer(Model):
    """
    Based on https://arxiv.org/pdf/1611.04230.pdf
    """
    def __init__(self,
                 vocab: Vocabulary,
                 source_embedder: TextFieldEmbedder,
                 sentence_encoder: Seq2VecEncoder,
                 sentence_accumulator: Seq2SeqEncoder) -> None:
        super(SummaRuNNer, self).__init__(vocab)
        self._source_embedder = source_embedder
        self._sentence_encoder = sentence_encoder
        self._sentence_accumulator = sentence_accumulator
        self._sentence_encoder_output_dim = self._sentence_encoder.get_output_dim()
        self._output_projection_layer = Linear(self._sentence_encoder_output_dim, 1)

    def forward(self,
                source_sentences: Dict[str, torch.Tensor],
                sentences_tags: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        tokens = source_sentences["tokens"]
        batch_size = tokens.size(0)
        sentences_count = tokens.size(1)
        max_sentence_length = tokens.size(2)
        tokens = tokens.reshape(batch_size * sentences_count, max_sentence_length)
        embedded_sentences = self._encode({"tokens": tokens})
        embedded_sentences = embedded_sentences.reshape(batch_size, sentences_count, -1)
        embedded_sentences = self._sentence_accumulator(embedded_sentences, mask=None)
        embedded_sentences = self._output_projection_layer(embedded_sentences).squeeze(2)

        predictions = embedded_sentences
        output_dict = {"predicted_tags": predictions}
        if sentences_tags is not None:
            loss = torch.nn.BCEWithLogitsLoss()(embedded_sentences, sentences_tags.float())
            output_dict["loss"] = loss
        return output_dict

    def _encode(self, source_tokens: Dict[str, torch.Tensor]) -> torch.Tensor:
        # shape: (batch_size, max_input_sequence_length, encoder_input_dim)
        embedded_input = self._source_embedder(source_tokens)
        # shape: (batch_size, max_input_sequence_length)
        source_mask = util.get_text_field_mask(source_tokens)
        # shape: (batch_size, max_input_sequence_length, encoder_output_dim)
        encoder_outputs = self._sentence_encoder(embedded_input, source_mask)
        return encoder_outputs
