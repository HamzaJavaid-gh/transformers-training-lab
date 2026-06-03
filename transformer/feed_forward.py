"""Position-wise feed-forward network (paper §3.3).

Attention only does a *linear* mix of values across positions — it cannot apply a
per-position non-linearity. The FFN (Linear -> ReLU -> Linear) is what gives each
position the capacity to compute something non-trivial. It's also ~2/3 of the model's
parameters at the paper's 4x d_ff/d_model ratio. Removing it is the `disable_ffn` ablation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer.config import ModelConfig


class PositionwiseFFN(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.disabled = cfg.disable_ffn
        self.linear1 = nn.Linear(cfg.d_model, cfg.d_ff)
        self.linear2 = nn.Linear(cfg.d_ff, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.disabled:
            return x
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


if __name__ == "__main__":
    cfg = ModelConfig(d_model=64, d_ff=256, dropout=0.0)
    ffn = PositionwiseFFN(cfg)
    x = torch.randn(2, 7, 64)
    out = ffn(x)
    assert out.shape == x.shape, out.shape
    print("feed_forward OK:", tuple(out.shape))
