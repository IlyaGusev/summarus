import torch
from torch.nn.modules.linear import Linear
from torch.nn import Parameter

from allennlp.modules.attention import Attention
from allennlp.nn.util import masked_softmax


@Attention.register("bahdanau")
class BahdanauAttention(Attention):
    def __init__(self, dim: int,
                 normalize: bool = True,
                 use_coverage: bool = False,
                 init_coverage_layer: bool = True,
                 use_attn_bias=False):
        super(BahdanauAttention, self).__init__(normalize)

        self._dim = dim
        self._use_coverage = use_coverage

        self._decoder_hidden_projection_layer = Linear(dim, dim, bias=False)
        self._encoder_outputs_projection_layer = Linear(dim, dim, bias=False)
        self._v = Linear(dim, 1, bias=False)
        self._use_attn_bias = use_attn_bias
        if use_attn_bias:
            self._bias = Parameter(torch.zeros(1), requires_grad=True)
        if use_coverage or init_coverage_layer:
            self._coverage_projection_layer = Linear(1, dim, bias=False)

    def forward(self,
                vector: torch.Tensor,
                matrix: torch.Tensor,
                matrix_mask: torch.Tensor = None,
                coverage: torch.Tensor = None) -> torch.Tensor:
        similarities = self._forward_internal(vector, matrix, coverage)
        if self._normalize:
            return masked_softmax(similarities, matrix_mask)
        else:
            return similarities

    def _forward_internal(self,
                          decoder_state: torch.Tensor,
                          encoder_outputs: torch.Tensor,
                          coverage: torch.Tensor = None):
        batch_size = encoder_outputs.size(0)
        source_length = encoder_outputs.size(1)

        encoder_feature = self._encoder_outputs_projection_layer(encoder_outputs)
        decoder_feature = self._decoder_hidden_projection_layer(decoder_state)
        decoder_feature = decoder_feature.unsqueeze(1).expand(batch_size, source_length, self._dim)

        features = encoder_feature + decoder_feature
        if self._use_coverage and coverage is not None:
            coverage_input = coverage.unsqueeze(2)
            coverage_feature = self._coverage_projection_layer(coverage_input)
            features = features + coverage_feature
        if self._use_attn_bias:
            features = features + self._bias
        scores = self._v(torch.tanh(features)).squeeze(2)
        return scores
