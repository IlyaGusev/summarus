from typing import Dict

import torch
from torch.nn.modules import Linear, Embedding, Dropout
from torch.nn import Parameter

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
                 sentence_accumulator: Seq2SeqEncoder,
                 use_salience: bool,
                 use_pos_embedding: bool,
                 use_output_bias: bool,
                 use_novelty: bool,
                 dropout: float = 0.3,
                 pos_embedding_num: int = 50,
                 pos_embedding_size: int = 128) -> None:
        super(SummaRuNNer, self).__init__(vocab)

        self._source_embedder = source_embedder

        self._sentence_encoder = sentence_encoder
        self._se_output_dim = self._sentence_encoder.get_output_dim()

        self._sentence_accumulator = sentence_accumulator
        self._h_sentence_dim = self._sentence_accumulator.get_output_dim()

        self._dropout_layer = Dropout(dropout)

        self._content_projection_layer = Linear(self._h_sentence_dim, 1)

        self._use_salience = use_salience
        if use_salience:
            self._document_linear_layer = Linear(self._h_sentence_dim, self._h_sentence_dim, bias=True)
            self._salience_linear_layer = Linear(self._h_sentence_dim, self._h_sentence_dim, bias=False)

        self._use_pos_embedding = use_pos_embedding
        if use_pos_embedding:
            self._pos_embedding_num = pos_embedding_num
            self._pos_embedding_size = pos_embedding_size
            self._pos_embedding_layer = Embedding(pos_embedding_num, pos_embedding_size)
            self._pos_projection_layer = Linear(pos_embedding_size, 1)

        self._use_output_bias = use_output_bias
        if use_output_bias:
            self._output_bias = Parameter(torch.zeros(1).uniform_(-0.1, 0.1), requires_grad=True)

        self._use_novelty = use_novelty
        if use_novelty:
            self._novelty_linear_layer = Linear(self._h_sentence_dim, self._h_sentence_dim, bias=False)

    def forward(self,
                source_sentences: Dict[str, Dict[str, torch.Tensor]],
                sentences_tags: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        tokens = source_sentences["tokens"]["tokens"]
        batch_size = tokens.size(0)
        sentences_count = tokens.size(1)
        max_sentence_length = tokens.size(2)
        tokens = tokens.reshape(batch_size * sentences_count, max_sentence_length)

        sentences_embeddings = self._encode({"tokens": tokens})
        sentences_embeddings = sentences_embeddings.reshape(batch_size, sentences_count, -1)
        sentences_embeddings = self._dropout_layer(sentences_embeddings)

        h_sentences = self._sentence_accumulator(sentences_embeddings, mask=None)
        h_sentences = self._dropout_layer(h_sentences)

        output_dict = dict()
        content = self._content_projection_layer(h_sentences).squeeze(2)
        output_dict["content"] = content
        predictions = content

        if self._use_salience:
            document_embedding = self._document_linear_layer(torch.mean(h_sentences, dim=1))
            document_embedding = torch.tanh(document_embedding)
            salience_intermediate = self._salience_linear_layer(document_embedding).unsqueeze(2)
            salience = torch.bmm(h_sentences, salience_intermediate).squeeze(2)
            predictions = predictions + salience
            output_dict["salience"] = salience

        if self._use_pos_embedding:
            assert sentences_count <= self._pos_embedding_num
            position_ids = util.get_range_vector(sentences_count, tokens.device.index)
            position_ids = position_ids.unsqueeze(0).expand((batch_size, sentences_count))
            positional_embeddings = self._pos_embedding_layer(position_ids)
            positional_projection = self._pos_projection_layer(positional_embeddings).squeeze(2)
            predictions = predictions + positional_projection
            output_dict["pos"] = positional_projection

        if self._use_output_bias:
            predictions = predictions + self._output_bias

        if self._use_novelty:
            summary_representation = sentences_embeddings.new_zeros((batch_size, self._h_sentence_dim))
            novelty = content.new_zeros((batch_size, sentences_count))
            for sentence_num in range(sentences_count):
                novelty_intermediate = self._novelty_linear_layer(torch.tanh(summary_representation)).unsqueeze(2)
                sentence_num_state = h_sentences[:, sentence_num, :]
                novelty[:, sentence_num] = -torch.bmm(
                    sentence_num_state.unsqueeze(1),
                    novelty_intermediate
                ).squeeze(2).squeeze(1)
                predictions[:, sentence_num] += novelty[:, sentence_num]
                probabilities = torch.sigmoid(predictions[:, sentence_num])
                summary_representation += torch.mv(sentence_num_state.transpose(0, 1), probabilities)
            output_dict["novelty"] = novelty

        output_dict["predicted_tags"] = predictions
        if sentences_tags is not None:
            loss = torch.nn.BCEWithLogitsLoss()(predictions, sentences_tags.float())
            output_dict["loss"] = loss
        return output_dict

    def _encode(self, source_tokens: Dict[str, torch.Tensor]) -> torch.Tensor:
        # shape: (batch_size, max_input_sequence_length, encoder_input_dim)
        embedded_input = self._source_embedder({"tokens": source_tokens})
        # shape: (batch_size, max_input_sequence_length)
        source_mask = util.get_text_field_mask({"tokens": source_tokens})
        # shape: (batch_size, max_input_sequence_length, encoder_output_dim)
        encoder_outputs = self._sentence_encoder(embedded_input, source_mask)
        return encoder_outputs
