from typing import Optional
import math
import torch
import torch.nn as nn
from src.attention import MultiHeadAttention


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor):
        x = x + self.pe[:, : x.size(1)]  # type: ignore
        return x


class FFN(nn.Module):
    def __init__(self, d_model: int, d_ffn: int, dropout: float = 0.1):
        super(FFN, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(d_ffn, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor):
        return self.linear(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ffn: int, dropout: float = 0.1):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(n_heads, d_model)
        self.ffn = FFN(d_model, d_ffn, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        attn_output = self.mha(x, x, x, mask)
        attn_output = self.dropout(attn_output)
        x = self.norm1(x + attn_output)
        ffn_output = self.ffn(x)
        ffn_output = self.dropout(ffn_output)
        x = self.norm2(x + ffn_output)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ffn, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.mha1 = MultiHeadAttention(n_heads, d_model)
        self.mha2 = MultiHeadAttention(n_heads, d_model)
        self.ffn = FFN(d_model, d_ffn, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ):
        _x = x
        x = self.mha1(x, x, x, tgt_mask)
        x = self.dropout(x)
        x = self.norm1(x + _x)
        _x = x
        print("x shape", x.shape, "encoder_output shape", encoder_output.shape)
        x = self.mha2(x, encoder_output, encoder_output, src_mask)
        x = self.dropout(x)
        x = self.norm2(x + _x)
        _x = x
        x = self.ffn(x)
        x = self.dropout(x)
        x = self.norm3(x + _x)
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        d_ffn: int,
        num_layers: int,
        dropout: float = 0.1,
    ):
        super(Encoder, self).__init__()
        self.token_embd = nn.Embedding(vocab_size, d_model)
        self.embd = PositionalEncoding(d_model)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, d_ffn, dropout) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = self.token_embd(x)
        x = self.embd(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        d_ffn: int,
        num_layers: int,
        dropout: float = 0.1,
    ):
        super(Decoder, self).__init__()
        self.token_embd = nn.Embedding(vocab_size, d_model)
        self.embd = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, n_heads, d_ffn, dropout) for _ in range(num_layers)]
        )

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        x = self.token_embd(x)
        x = self.embd(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return x
