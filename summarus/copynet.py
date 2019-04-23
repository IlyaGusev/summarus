import logging
from typing import Dict, Any

import torch

from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Attention, TextFieldEmbedder, Seq2SeqEncoder
from allennlp.models.encoder_decoders.copynet_seq2seq import CopyNetSeq2Seq
from allennlp.training.metrics import Metric


logger = logging.getLogger(__name__)


@Model.register("custom_copynet_seq2seq")
class CustomCopyNetSeq2Seq(CopyNetSeq2Seq):
    def __init__(self,
                 vocab: Vocabulary,
                 source_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 attention: Attention,
                 beam_size: int,
                 max_decoding_steps: int,
                 target_embedding_dim: int = None,
                 copy_token: str = "@COPY@",
                 source_namespace: str = "tokens",
                 target_namespace: str = "target_tokens",
                 tensor_based_metric: Metric = None,
                 token_based_metric: Metric = None,
                 tie_embeddings: bool = False) -> None:
        target_embedding_dim = target_embedding_dim or source_embedder.get_output_dim()
        CopyNetSeq2Seq.__init__(
            self,
            vocab,
            source_embedder,
            encoder,
            attention,
            beam_size,
            max_decoding_steps,
            target_embedding_dim,
            copy_token,
            source_namespace,
            target_namespace,
            tensor_based_metric,
            token_based_metric
        )
        self._tie_embeddings = tie_embeddings

        if self._tie_embeddings:
            assert source_namespace == target_namespace
            assert "token_embedder_tokens" in dict(self._source_embedder.named_children())
            source_token_embedder = dict(self._source_embedder.named_children())["token_embedder_tokens"]
            self._target_embedder.weight = source_token_embedder.weight

        if tensor_based_metric is None:
            self._tensor_based_metric = None

    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        predicted_tokens = self._get_predicted_tokens(
            output_dict["predictions"],
            output_dict["metadata"],
            n_best=1
        )
        output_dict["predicted_tokens"] = predicted_tokens
        return output_dict
