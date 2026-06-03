"""Additive attention masks.

We use the ADDITIVE convention everywhere: a mask is a float tensor added to the
attention scores before softmax — `0.0` where allowed, `-inf` where blocked. Shape is
`(B, 1, Tq, Tk)` so it broadcasts across heads (and across queries, for pad-only masks).

Inputs are BOOLEAN pad masks from the data layer (True = is-pad = blocked). Everything
here is built on the same device as its input, so it works transparently on CUDA/MPS/CPU.
"""

from __future__ import annotations

import torch


def additive_mask(boolean_blocked: torch.Tensor) -> torch.Tensor:
    """True-means-blocked boolean -> additive 0/-inf float on the same device."""
    return torch.zeros_like(boolean_blocked, dtype=torch.float).masked_fill_(boolean_blocked, float("-inf"))


def build_src_mask(src_pad: torch.Tensor) -> torch.Tensor:
    """src_pad: (B, S) bool -> (B, 1, 1, S). Same key positions blocked for every query/head."""
    return additive_mask(src_pad)[:, None, None, :]


def causal_mask(T: int, device: torch.device) -> torch.Tensor:
    """(1, 1, T, T) additive mask blocking position t from attending to positions > t."""
    blocked = torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)
    return additive_mask(blocked)[None, None, :, :]


def build_tgt_mask(tgt_pad: torch.Tensor, use_causal: bool = True) -> torch.Tensor:
    """tgt_pad: (B, T) bool -> (B, 1, T, T). Padding mask, optionally + causal mask."""
    B, T = tgt_pad.shape
    pad = additive_mask(tgt_pad)[:, None, None, :]   # (B, 1, 1, T)
    if not use_causal:
        return pad
    return pad + causal_mask(T, tgt_pad.device)      # broadcasts (B,1,1,T) + (1,1,T,T)
