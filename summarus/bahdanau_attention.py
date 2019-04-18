import torch
import torch.nn.functional as F
from torch.nn.modules.linear import Linear

from allennlp.modules.attention import Attention


@Attention.register("bahdanau")
class BahdanauAttention(Attention):
    def __init__(self, dim: int, normalize: bool = True):
        super(BahdanauAttention, self).__init__(normalize)

        self._dim = dim

        self.decoder_hidden_projection_layer = Linear(dim, dim, bias=False)
        self.encoder_outputs_projection_layer = Linear(dim, dim, bias=False)
        self.v = Linear(dim, 1, bias=False)

    def _forward_internal(self, decoder_state, encoder_outputs):
        batch_size, l, n = list(encoder_outputs.size())

        decoder_feature = self.decoder_hidden_projection_layer(decoder_state)
        decoder_feature = decoder_feature.unsqueeze(1).expand(batch_size, l, self._dim).contiguous()
        decoder_feature = decoder_feature.view(-1, self._dim)  # B * l x hidden_dim

        encoder_feature = self.encoder_outputs_projection_layer(encoder_outputs.contiguous())
        encoder_feature = encoder_feature.contiguous().view(-1, self._dim)

        features = encoder_feature + decoder_feature
        scores = self.v(torch.tanh(features))
        scores = scores.view(-1, l)
        return scores

