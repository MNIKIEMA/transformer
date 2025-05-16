import math
import torch
import torch.nn as nn
from typing import Optional


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads: int, d_model: int):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        B, T, D = query.size()

        # Linear projections
        print(f"query shape: {query.shape} key shape: {key.shape} value shape: {value.shape}")
        Q = (
            self.W_q(query).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        )  # [B, H, T, d_head]
        K = self.W_k(key).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        V = self.W_v(value).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn_weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, V)  # [B, H, T, d_head]
        context = context.transpose(1, 2).contiguous().view(B, T, D)  # [B, T, D]
        return self.out_proj(context)
