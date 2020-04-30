from typing import Dict, Optional

import torch
from overrides import overrides

from allennlp.common.checks import ConfigurationError
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder, Embedding
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.nn import util, InitializerApplicator, RegularizerApplicator
from allennlp_models.seq2seq.seq_decoder import SeqDecoder
from allennlp_models.seq2seq.composed_seq2seq_model import ComposedSeq2Seq


@Model.register("custom_composed_seq2seq")
class CustomComposedSeq2Seq(ComposedSeq2Seq):
    def __init__(self,
                 vocab: Vocabulary,
                 source_text_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 decoder: SeqDecoder,
                 tied_source_embedder_key: Optional[str] = None,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 **kwargs) -> None:

        super(CustomComposedSeq2Seq, self).__init__(vocab, source_text_embedder, encoder,
            decoder, tied_source_embedder_key, initializer, **kwargs)
