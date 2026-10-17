"""
RMSNorm: scale each token vector to a standard length.

    x [batch, seq, 256]  ->  x / rms(x) * gain  ->  [batch, seq, 256]
    rms(x) = sqrt(mean(x^2) + eps)

RMSNorm scales each token vector alone. The direction does not change.
After the scale, the mean of x^2 is 1, so the length is sqrt(256) = 16.

The model learns gain [256]. gain starts at 1, so at the start RMSNorm only scales.
eps = 1e-6 prevents a division by zero.
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # One learned factor for each dimension. Start: 1 (no change).
        self.gain = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)   # one number per token
        return x / rms * self.gain
