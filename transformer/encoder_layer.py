"""One encoder layer: self-attention -> FFN, each wrapped in residual + LayerNorm.

Post-LN as in the original paper: x <- LayerNorm(x + Sublayer(x)). (Pre-norm is a later
experiment, not the baseline — see CLAUDE.md.)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from transformer.config import ModelConfig
from transformer.feed_forward import PositionwiseFFN
from transformer.multi_head_attention import MultiHeadAttention


class EncoderLayer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.self_attn = MultiHeadAttention(cfg)
        self.ffn = PositionwiseFFN(cfg)
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.dropout1 = nn.Dropout(cfg.dropout)
        self.dropout2 = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor | None) -> torch.Tensor:
        h = self.self_attn(x, x, x, mask=src_mask)
        x = self.norm1(x + self.dropout1(h))
        f = self.ffn(x)
        x = self.norm2(x + self.dropout2(f))
        return x


if __name__ == "__main__":
    cfg = ModelConfig(d_model=64, d_ff=256, n_heads=4, dropout=0.0)
    layer = EncoderLayer(cfg)
    x = torch.randn(2, 7, 64)
    out = layer(x, src_mask=None)
    assert out.shape == x.shape, out.shape
    print("encoder_layer OK:", tuple(out.shape))
