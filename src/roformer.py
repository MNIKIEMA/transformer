from typing import Optional

import torch
import torch.nn as nn

from src.attention import RoMultiHeadAttention
from src.layers import FFN
from src.utils import compute_rope_params


class RoEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ffn: int, dropout: float = 0.1):
        super().__init__()
        self.mha = RoMultiHeadAttention(n_heads, d_model)
        self.ffn = FFN(d_model, d_ffn, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        _x = x
        x = self.mha(x, x, x, cos, sin, mask)
        x = self.dropout(x)
        x = self.norm1(x + _x)
        _x = x
        x = self.ffn(x)
        x = self.dropout(x)
        x = self.norm2(x + _x)
        return x


class RoDecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ffn: int, dropout: float = 0.1):
        super().__init__()
        self.mha1 = RoMultiHeadAttention(n_heads, d_model)
        self.mha2 = RoMultiHeadAttention(n_heads, d_model)
        self.ffn = FFN(d_model, d_ffn, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ):
        _x = x
        x = self.mha1(x, x, x, cos, sin, tgt_mask)
        x = self.dropout(x)
        x = self.norm1(x + _x)
        _x = x
        # cross-attention: apply_rope slices cos/sin by each tensor's seq_len
        x = self.mha2(x, encoder_output, encoder_output, cos, sin, src_mask)
        x = self.dropout(x)
        x = self.norm2(x + _x)
        _x = x
        x = self.ffn(x)
        x = self.dropout(x)
        x = self.norm3(x + _x)
        return x


class RoEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        d_ffn: int,
        num_layers: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.token_embd = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [RoEncoderLayer(d_model, n_heads, d_ffn, dropout) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        x = self.token_embd(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, cos, sin, mask)
        return x


class RoDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        d_ffn: int,
        num_layers: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.token_embd = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [RoDecoderLayer(d_model, n_heads, d_ffn, dropout) for _ in range(num_layers)]
        )

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ):
        x = self.token_embd(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, encoder_output, cos, sin, src_mask, tgt_mask)
        return x


class RoFormer(nn.Module):
    def __init__(
        self,
        enc_vocab_size: int,
        dec_vocab_size: int,
        d_model: int,
        n_heads: int,
        d_ffn: int,
        num_layers: int,
        dropout: float = 0.1,
        context_length: int = 5000,
        theta_base: int = 10_000,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.encoder = RoEncoder(enc_vocab_size, d_model, n_heads, d_ffn, num_layers, dropout)
        self.decoder = RoDecoder(dec_vocab_size, d_model, n_heads, d_ffn, num_layers, dropout)
        self.out_proj = nn.Linear(d_model, dec_vocab_size)

        # Precompute RoPE cos/sin for the maximum context length and register as buffers
        # so they move to the correct device with the model
        d_head = d_model // n_heads
        cos, sin = compute_rope_params(d_head, theta_base=theta_base, context_length=context_length)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ):
        encoder_output = self.encoder(src, self.cos, self.sin, src_mask)
        decoder_output = self.decoder(tgt, encoder_output, self.cos, self.sin, src_mask, tgt_mask)
        return self.out_proj(decoder_output)
