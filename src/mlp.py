"""
The SwiGLU feed-forward network (MLP). It works on each token alone.

    n [batch, seq, 256]
      -> gate = n W_gate,  up = n W_up          each [batch, seq, 1024]
      -> a = SiLU(gate) * up                    [batch, seq, 1024]   (position by position)
      -> out = a W_down                         [batch, seq, 256]

up: 1,024 candidate values (what to write).
gate: after SiLU, 1,024 dimmers (how much of each value goes through).
SiLU(x) = x * sigmoid(x). It has no parameters.

Attention moves information between tokens. The MLP does not: token i uses only its own n.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    def __init__(self, embedding_dimension: int, feed_forward_dimension: int, dropout: float = 0.0):
        super().__init__()
        # Three learned matrices, no bias. nn.Linear stores each one as [out, in]:
        # W_gate, W_up [1024, 256], W_down [256, 1024].
        self.W_gate = nn.Linear(embedding_dimension, feed_forward_dimension, bias=False)
        self.W_up = nn.Linear(embedding_dimension, feed_forward_dimension, bias=False)
        self.W_down = nn.Linear(feed_forward_dimension, embedding_dimension, bias=False)
        # Random start (mean 0, std 0.02), the same as in attention.
        for layer in (self.W_gate, self.W_up, self.W_down):
            nn.init.normal_(layer.weight, mean=0.0, std=0.02)
        self.dropout = nn.Dropout(dropout)   # on the output, during training only

    def forward(self, n: torch.Tensor) -> torch.Tensor:
        a = F.silu(self.W_gate(n)) * self.W_up(n)    # [batch, seq, 1024]
        return self.dropout(self.W_down(a))          # [batch, seq, 256]
