from typing import Tuple, Dict, Optional
from overrides import overrides

import torch
from torch.autograd import Variable
from torch.nn.modules.transformer import TransformerDecoder, TransformerDecoderLayer, LayerNorm, Dropout, xavier_uniform_

# from allennlp.modules.seq2seq_encoders.bidirectional_language_model_transformer import PositionalEncoding
from allennlp.nn.util import add_positional_features
from allennlp.modules.seq2seq_decoders.decoder_net import DecoderNet


@DecoderNet.register("torch_transformer_decoder_net")
class TorchTransformerDecoderNet(DecoderNet):
    def __init__(self,
                 decoding_dim: int,
                 target_embedding_dim: int,
                 feedforward_hidden_dim: int,
                 num_layers: int,
                 num_attention_heads: int,
                 use_positional_encoding: bool = True,
                 positional_encoding_max_steps: int = 1000,
                 dropout_prob: float = 0.1) -> None:

        super().__init__(decoding_dim=decoding_dim,
                         target_embedding_dim=target_embedding_dim,
                         decodes_parallel=True)

        decoder_layer = TransformerDecoderLayer(decoding_dim, num_attention_heads, feedforward_hidden_dim, dropout_prob)
        decoder_norm = LayerNorm(decoding_dim)
        self._decoder = TransformerDecoder(decoder_layer, num_layers, decoder_norm)
        self._dropout = Dropout(dropout_prob)
        self._use_positional_encoding = use_positional_encoding
        self._reset_parameters()
        # self._positional_embedder = PositionalEncoding(decoding_dim, positional_encoding_max_steps) \
        #     if use_positional_encoding else None

    @overrides
    def init_decoder_state(self, encoder_out: Dict[str, torch.LongTensor]) -> Dict[str, torch.Tensor]:
        return {}

    @overrides
    def forward(self,
                previous_state: Dict[str, torch.Tensor],
                encoder_outputs: torch.Tensor,
                source_mask: torch.Tensor,
                previous_steps_predictions: torch.Tensor,
                previous_steps_mask: Optional[torch.Tensor] = None) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        seq_len = previous_steps_predictions.size(-2)
        future_mask = torch.triu(torch.ones(seq_len, seq_len, device=source_mask.device, dtype=torch.float)).transpose(0, 1)
        future_mask = future_mask.masked_fill(future_mask == 0, float('-inf')).masked_fill(future_mask == 1, float(0.0))
        future_mask = Variable(future_mask)

        if self._use_positional_encoding:
            previous_steps_predictions = add_positional_features(previous_steps_predictions)
        previous_steps_predictions = self._dropout(previous_steps_predictions)

        source_mask = ~(source_mask.bool())
        if previous_steps_mask is not None:
            previous_steps_mask = ~(previous_steps_mask.bool())
        previous_steps_predictions = previous_steps_predictions.permute(1, 0, 2)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        output = self._decoder(previous_steps_predictions, encoder_outputs,
                               tgt_mask=future_mask,
                               tgt_key_padding_mask=previous_steps_mask,
                               memory_key_padding_mask=source_mask)
        return {}, output.permute(1, 0, 2)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)
