from typing import Dict, List, Tuple, Optional
from overrides import overrides

import numpy
import torch
import torch.nn.functional as F
from torch.nn import Linear

from allennlp.common.checks import ConfigurationError
from allennlp.common.util import END_SYMBOL, START_SYMBOL
from allennlp.data import Vocabulary
from allennlp.modules import Embedding
from allennlp.nn import util
from allennlp.nn.beam_search import BeamSearch
from allennlp.training.metrics import Metric
from allennlp_models.seq2seq.decoder_net import DecoderNet
from allennlp_models.seq2seq.seq_decoder import SeqDecoder
from allennlp_models.seq2seq.auto_regressive_seq_decoder import AutoRegressiveSeqDecoder


@SeqDecoder.register("custom_auto_regressive_seq_decoder")
class CustomAutoRegressiveSeqDecoder(AutoRegressiveSeqDecoder):
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
        super().__init__(vocab, decoder_net, max_decoding_steps, target_embedder,
            target_namespace, tie_output_embedding, scheduled_sampling_ratio,
            label_smoothing_ratio, beam_size, tensor_based_metric, token_based_metric)
