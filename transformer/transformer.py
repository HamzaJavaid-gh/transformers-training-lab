"""The full Transformer: Encoder + Decoder + embeddings + output projection.

Composed from the per-file components built in Phase 2. Design decisions (all from the
paper / CLAUDE.md):
  - Embeddings scaled by sqrt(d_model) (§3.4).
  - Sinusoidal PE added to embeddings (unless the no_positional ablation is on).
  - Weight tying (§3.4): with a JOINT vocab we share ONE weight matrix across the
    source embedding, the target embedding, AND the pre-softmax projection.
  - Post-LN inside every layer; each stack ends with a final LayerNorm (the
    Annotated-Transformer convention — harmless under post-LN, and it normalizes the
    representation that feeds the output projection).
  - Masks are additive and built in transformer/masks.py.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from transformer.config import ModelConfig
from transformer.decoder_layer import DecoderLayer
from transformer.encoder_layer import EncoderLayer
from transformer.masks import build_src_mask, build_tgt_mask
from transformer.positional_encoding import SinusoidalPositionalEncoding


class Encoder(nn.Module):
    """N stacked encoder layers + a final LayerNorm."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.layers = nn.ModuleList(EncoderLayer(cfg) for _ in range(cfg.n_encoder_layers))
        self.norm = nn.LayerNorm(cfg.d_model)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor | None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)


class Decoder(nn.Module):
    """N stacked decoder layers + a final LayerNorm."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.layers = nn.ModuleList(DecoderLayer(cfg) for _ in range(cfg.n_decoder_layers))
        self.norm = nn.LayerNorm(cfg.d_model)

    def forward(self, x, memory, tgt_mask, memory_mask) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
        return self.norm(x)


class Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.tgt_embed = nn.Embedding(cfg.tgt_vocab_size, cfg.d_model, padding_idx=cfg.pad_id)
        joint = cfg.tie_embeddings and cfg.src_vocab_size == cfg.tgt_vocab_size
        # Joint vocab -> one shared embedding matrix for src and tgt (paper §3.4).
        self.src_embed = self.tgt_embed if joint else \
            nn.Embedding(cfg.src_vocab_size, cfg.d_model, padding_idx=cfg.pad_id)

        self.pos_enc = SinusoidalPositionalEncoding(cfg.d_model, cfg.max_seq_len)
        self.embed_dropout = nn.Dropout(cfg.dropout)

        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)

        self.output_proj = nn.Linear(cfg.d_model, cfg.tgt_vocab_size, bias=False)
        if cfg.tie_embeddings:
            # Tie the pre-softmax projection to the target embedding. Saves params and
            # gives the embedding direct gradient from the loss.
            self.output_proj.weight = self.tgt_embed.weight

        self._init_parameters()

    def _init_parameters(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def num_parameters(self) -> int:
        # Count unique tensors (tied weights share storage — don't double-count).
        seen, total = set(), 0
        for p in self.parameters():
            if id(p) in seen:
                continue
            seen.add(id(p))
            total += p.numel()
        return total

    # -- embedding -----------------------------------------------------------
    def _embed(self, tokens: torch.Tensor, embed: nn.Embedding) -> torch.Tensor:
        x = embed(tokens) * math.sqrt(self.cfg.d_model)  # §3.4 scaling
        if not self.cfg.no_positional:
            x = self.pos_enc(x)
        return self.embed_dropout(x)

    # -- forward pieces ------------------------------------------------------
    def encode(self, src: torch.Tensor, src_pad: torch.Tensor) -> torch.Tensor:
        return self.encoder(self._embed(src, self.src_embed), build_src_mask(src_pad))

    def decode(self, tgt_in, memory, tgt_pad, src_pad) -> torch.Tensor:
        tgt_mask = build_tgt_mask(tgt_pad, use_causal=not self.cfg.no_causal_mask)
        memory_mask = build_src_mask(src_pad)
        return self.decoder(self._embed(tgt_in, self.tgt_embed), memory, tgt_mask, memory_mask)

    def forward(self, src, tgt_in, src_pad, tgt_pad) -> torch.Tensor:
        memory = self.encode(src, src_pad)
        decoded = self.decode(tgt_in, memory, tgt_pad, src_pad)
        return self.output_proj(decoded)  # (B, T, tgt_vocab)

    # -- naive greedy decode (replaced by proper search in Phase 5) ----------
    @torch.no_grad()
    def generate(self, src, src_pad, bos_id, eos_id, pad_id, max_len=64) -> torch.Tensor:
        self.eval()
        B, device = src.size(0), src.device
        memory = self.encode(src, src_pad)
        ys = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        for _ in range(max_len - 1):
            decoded = self.decode(ys, memory, ys.eq(pad_id), src_pad)
            logits = self.output_proj(decoded[:, -1])  # (B, V)
            nxt = logits.argmax(dim=-1)
            nxt = torch.where(finished, torch.full_like(nxt, pad_id), nxt)
            ys = torch.cat([ys, nxt.unsqueeze(1)], dim=1)
            finished = finished | nxt.eq(eos_id)
            if finished.all():
                break
        return ys


def build_model(cfg: ModelConfig) -> Transformer:
    return Transformer(cfg)


if __name__ == "__main__":
    cfg = ModelConfig.smoke(src_vocab_size=1000, tgt_vocab_size=1000)
    model = Transformer(cfg)
    src = torch.randint(4, 1000, (2, 9))
    tgt_in = torch.randint(4, 1000, (2, 7))
    src_pad = torch.zeros(2, 9, dtype=torch.bool)
    tgt_pad = torch.zeros(2, 7, dtype=torch.bool)
    logits = model(src, tgt_in, src_pad, tgt_pad)
    assert logits.shape == (2, 7, 1000), logits.shape
    print(f"transformer OK: logits {tuple(logits.shape)}, params {model.num_parameters():,}")
