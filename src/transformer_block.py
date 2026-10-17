"""
One Transformer block (pre-norm).

    x = x + Attention(RMSNorm(x))      tokens exchange information
    x = x + MLP(RMSNorm(x))            the MLP changes each token alone

RMSNorm scales a copy of x. The original x goes on to the add (the residual).
So each block adds a change to x. It does not replace x.
The shape stays [batch, seq, 256] in each step.
"""

import torch
import torch.nn as nn

from src.attention import CausalSelfAttention
from src.mlp import SwiGLU
from src.rmsnorm import RMSNorm


class TransformerBlock(nn.Module):
    def __init__(self, embedding_dimension: int, num_heads: int, head_dimension: int,
                 feed_forward_dimension: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = RMSNorm(embedding_dimension)
        self.attention = CausalSelfAttention(embedding_dimension, num_heads, head_dimension, dropout)
        self.norm2 = RMSNorm(embedding_dimension)
        self.mlp = SwiGLU(embedding_dimension, feed_forward_dimension, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
