"""Sinusoidal positional encoding (paper §3.5).

Why fixed and not learned?
  1. Generalizes to sequence lengths longer than seen at train time.
  2. PE(pos+k) is a linear function of PE(pos), so relative positions are easy to
     recover via a (learnable) linear map inside attention.

Ported from the sibling attention-implementation repo, kept in its own file per the plan.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (L, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Non-persistent buffer: moves with .to(device), not saved in the state_dict.
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # (1, L, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        return x + self.pe[:, : x.size(1)]


if __name__ == "__main__":
    pe = SinusoidalPositionalEncoding(d_model=256, max_len=100)
    x = torch.zeros(2, 50, 256)
    out = pe(x)
    assert out.shape == x.shape, out.shape
    print("positional_encoding OK:", tuple(out.shape))
