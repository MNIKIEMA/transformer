import torch.nn as nn
from src.layers import Encoder, Decoder


class Transformer(nn.Module):
    def __init__(
        self,
        enc_vocab_size: int,
        dec_vocab_size: int,
        d_model: int,
        n_heads: int,
        d_ffn: int,
        num_layers: int,
        dropout: float = 0.1,
    ):
        super(Transformer, self).__init__()
        self.encoder = Encoder(
            vocab_size=enc_vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            d_ffn=d_ffn,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.decoder = Decoder(
            vocab_size=dec_vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            d_ffn=d_ffn,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.out_proj = nn.Linear(d_model, dec_vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        encoder_output = self.encoder(src, src_mask)
        print("encoder output shape", encoder_output.shape)
        decoder_output = self.decoder(tgt, encoder_output, src_mask, tgt_mask)
        return self.out_proj(decoder_output)
