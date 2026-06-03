"""One decoder layer: masked self-attention -> cross-attention -> FFN.

Each sublayer is wrapped in residual + LayerNorm (post-LN).

  1. Masked self-attention: the decoder attends to its own outputs so far, with a
     causal mask so position t can't see t+1.. (the future). This mask is what makes
     training (teacher-forced) and inference (autoregressive) the SAME problem.
  2. Cross-attention: Q from the decoder, K & V from the encoder memory. This is the
     bridge from source to target. Disable it (no_cross_attn) and the decoder collapses
     to a target-only language model that ignores the input sentence.
  3. Position-wise FFN.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from transformer.config import ModelConfig
from transformer.feed_forward import PositionwiseFFN
from transformer.multi_head_attention import MultiHeadAttention


class DecoderLayer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.self_attn = MultiHeadAttention(cfg)
        self.cross_attn = None if cfg.no_cross_attn else MultiHeadAttention(cfg)
        self.ffn = PositionwiseFFN(cfg)
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.norm3 = nn.LayerNorm(cfg.d_model)
        self.dropout1 = nn.Dropout(cfg.dropout)
        self.dropout2 = nn.Dropout(cfg.dropout)
        self.dropout3 = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,                    # (B, T, d_model) decoder input
        memory: torch.Tensor,              # (B, S, d_model) encoder output
        tgt_mask: torch.Tensor | None,     # padding + causal, additive
        memory_mask: torch.Tensor | None,  # src padding, additive
    ) -> torch.Tensor:
        h = self.self_attn(x, x, x, mask=tgt_mask)
        x = self.norm1(x + self.dropout1(h))

        if self.cross_attn is not None:
            h = self.cross_attn(x, memory, memory, mask=memory_mask)
            x = self.norm2(x + self.dropout2(h))

        f = self.ffn(x)
        x = self.norm3(x + self.dropout3(f))
        return x


if __name__ == "__main__":
    from transformer.masks import build_tgt_mask

    cfg = ModelConfig(d_model=64, d_ff=256, n_heads=4, dropout=0.0)
    layer = DecoderLayer(cfg)
    x = torch.randn(2, 5, 64)
    memory = torch.randn(2, 7, 64)
    tgt_pad = torch.zeros(2, 5, dtype=torch.bool)
    out = layer(x, memory, tgt_mask=build_tgt_mask(tgt_pad), memory_mask=None)
    assert out.shape == x.shape, out.shape
    print("decoder_layer OK:", tuple(out.shape))
