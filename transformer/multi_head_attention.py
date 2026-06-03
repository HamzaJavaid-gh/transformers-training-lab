"""Scaled dot-product attention + multi-head wrapper (paper §3.2).

Ported from the sibling repo. Two additions for this project:
  - an optional `cache_attn` hook so notebooks can pull out attention weights for
    heatmaps WITHOUT paying any cost during training (default off);
  - cfg-driven `no_scaling` / `single_head` ablation support.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer.config import ModelConfig


def scaled_dot_product_attention(
    q: torch.Tensor,                       # (B, h, Tq, d_k)
    k: torch.Tensor,                       # (B, h, Tk, d_k)
    v: torch.Tensor,                       # (B, h, Tk, d_v)
    mask: torch.Tensor | None = None,      # (B, 1, Tq, Tk) additive, -inf at blocked positions
    scale: float | None = None,
    dropout: nn.Dropout | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V.

    The /sqrt(d_k) keeps dot products from growing with d_k, which would push softmax
    into low-gradient regions. Dropping it is the `no_scaling` ablation.
    """
    d_k = q.size(-1)
    if scale is None:
        scale = 1.0 / math.sqrt(d_k)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, h, Tq, Tk)
    if mask is not None:
        scores = scores + mask
    attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        attn = dropout(attn)
    out = torch.matmul(attn, v)  # (B, h, Tq, d_v)
    return out, attn


class MultiHeadAttention(nn.Module):
    """h parallel attention heads, each in a d_k-dim subspace, then concatenated.

    Multi-head lets each subspace track a different relation (adjacency, agreement,
    coreference, ...) — stronger than one wide attention of equal parameter count.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.h = cfg.effective_n_heads
        self.d_k = cfg.d_k
        self.d_model = cfg.d_model
        self.no_scaling = cfg.no_scaling

        self.W_q = nn.Linear(cfg.d_model, self.h * self.d_k)
        self.W_k = nn.Linear(cfg.d_model, self.h * self.d_k)
        self.W_v = nn.Linear(cfg.d_model, self.h * self.d_k)
        self.W_o = nn.Linear(self.h * self.d_k, cfg.d_model)
        self.attn_dropout = nn.Dropout(cfg.dropout)

        # Notebook hook: set .cache_attn = True to capture the last attention map.
        self.cache_attn = False
        self.last_attn: torch.Tensor | None = None

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        return x.view(B, T, self.h, self.d_k).transpose(1, 2)  # (B, h, T, d_k)

    def _merge(self, x: torch.Tensor) -> torch.Tensor:
        B, h, T, d_k = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, h * d_k)

    def forward(self, query_in, key_in, value_in, mask=None) -> torch.Tensor:
        # self-attention: query_in == key_in == value_in.
        # cross-attention: query_in from DECODER, key_in/value_in from ENCODER memory.
        q = self._split(self.W_q(query_in))
        k = self._split(self.W_k(key_in))
        v = self._split(self.W_v(value_in))

        scale = 1.0 if self.no_scaling else 1.0 / math.sqrt(self.d_k)
        out, attn = scaled_dot_product_attention(q, k, v, mask=mask, scale=scale, dropout=self.attn_dropout)
        if self.cache_attn:
            self.last_attn = attn.detach()  # (B, h, Tq, Tk)
        return self.W_o(self._merge(out))


if __name__ == "__main__":
    cfg = ModelConfig(d_model=64, n_heads=4, dropout=0.0)
    mha = MultiHeadAttention(cfg)
    x = torch.randn(2, 7, 64)
    out = mha(x, x, x)
    assert out.shape == x.shape, out.shape
    print("multi_head_attention OK:", tuple(out.shape))
