"""Decoding: greedy and beam search.

Plain, readable loops (no KV cache — that's a deliberate Phase-later optimization).
The decoder re-runs over the full prefix each step, which is fine for Multi30k-length
sentences and keeps the logic transparent. Both functions are reusable by the BLEU
evaluation (Phase 6) and the future A100 inference.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def greedy_decode(model, src, src_pad, bos_id, eos_id, pad_id, max_len: int = 64) -> torch.Tensor:
    """Batched greedy decode: argmax at every step. Returns (B, L) token ids incl. <bos>."""
    model.eval()
    B, device = src.size(0), src.device
    memory = model.encode(src, src_pad)
    ys = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
    finished = torch.zeros(B, dtype=torch.bool, device=device)
    for _ in range(max_len - 1):
        decoded = model.decode(ys, memory, ys.eq(pad_id), src_pad)
        logits = model.output_proj(decoded[:, -1])           # (B, V)
        nxt = logits.argmax(dim=-1)
        nxt = torch.where(finished, torch.full_like(nxt, pad_id), nxt)
        ys = torch.cat([ys, nxt.unsqueeze(1)], dim=1)
        finished |= nxt.eq(eos_id)
        if finished.all():
            break
    return ys


def _length_penalty(length: int, alpha: float) -> float:
    # GNMT / Wu et al. length penalty used by the paper: ((5 + L) / (5 + 1))^alpha.
    return ((5 + length) / 6) ** alpha


@torch.no_grad()
def beam_search(model, src, src_pad, bos_id, eos_id, pad_id,
                beam_size: int = 4, alpha: float = 0.6, max_len: int = 64) -> list[int]:
    """Beam search for a SINGLE source (src shape (1, S)). Returns the best token id list.

    Scores are length-normalized so beam search doesn't unfairly prefer short outputs.
    """
    assert src.size(0) == 1, "beam_search decodes one sentence at a time"
    model.eval()
    device = src.device
    memory = model.encode(src, src_pad)  # (1, S, d)

    # Each beam: (cumulative_logprob, token_list, finished?)
    beams = [(0.0, [bos_id], False)]
    for _ in range(max_len - 1):
        if all(fin for _, _, fin in beams):
            break
        candidates: list[tuple[float, list[int], bool]] = []
        for score, toks, fin in beams:
            if fin:                       # keep finished hypotheses as-is
                candidates.append((score, toks, True))
                continue
            ys = torch.tensor([toks], device=device)
            decoded = model.decode(ys, memory, ys.eq(pad_id), src_pad)
            logp = F.log_softmax(model.output_proj(decoded[:, -1]), dim=-1)[0]  # (V,)
            topv, topi = logp.topk(beam_size)
            for v, i in zip(topv.tolist(), topi.tolist()):
                candidates.append((score + v, toks + [i], i == eos_id))
        # Keep the top `beam_size` by length-normalized score.
        candidates.sort(key=lambda c: c[0] / _length_penalty(len(c[1]), alpha), reverse=True)
        beams = candidates[:beam_size]

    best = max(beams, key=lambda c: c[0] / _length_penalty(len(c[1]), alpha))
    return best[1]
