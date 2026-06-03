"""Training primitives + a reusable `fit()` loop.

The notebook (04_training.ipynb) uses the small primitives inline so every step is
visible; the future A100 script imports `fit()` for the real run. Everything is
device-agnostic and has an AMP hook (CUDA-only, default off — see CLAUDE.md: FP32 first).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer.config import TrainConfig


# ---------------------------------------------------------------------------
# Noam learning-rate schedule (paper §5.3)
# ---------------------------------------------------------------------------
def lr_at(step: int, d_model: int, warmup: int) -> float:
    """lr = d_model^-0.5 * min(step^-0.5, step * warmup^-1.5).

    Linear warmup for `warmup` steps (the `step * warmup^-1.5` branch), then inverse-
    sqrt decay (the `step^-0.5` branch). They cross exactly at step == warmup.
    """
    step = max(step, 1)
    return d_model ** -0.5 * min(step ** -0.5, step * warmup ** -1.5)


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for g in optimizer.param_groups:
        g["lr"] = lr


def make_optimizer(model: nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    # lr is overwritten every step by the Noam schedule, so the initial value is moot.
    return torch.optim.Adam(model.parameters(), lr=0.0, betas=cfg.adam_betas, eps=cfg.adam_eps)


# ---------------------------------------------------------------------------
# Label smoothing — hand-written to match torch's CrossEntropyLoss semantics
# ---------------------------------------------------------------------------
class LabelSmoothingLoss(nn.Module):
    """Smoothed cross-entropy, written out so the math is explicit.

    Target distribution: (1-eps) on the true class + eps/V spread over all V classes
    (this exactly matches `F.cross_entropy(..., label_smoothing=eps)`). Positions equal
    to `pad_id` are ignored. We verify the equivalence in the notebook, then prefer the
    built-in for speed.
    """

    def __init__(self, pad_id: int, smoothing: float = 0.1):
        super().__init__()
        self.pad_id = pad_id
        self.eps = smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # logits: (N, V), target: (N,)
        logp = F.log_softmax(logits, dim=-1)
        nll = -logp.gather(1, target.unsqueeze(1)).squeeze(1)  # -log p(true)
        smooth = -logp.mean(dim=-1)                            # mean over all classes
        loss = (1 - self.eps) * nll + self.eps * smooth
        mask = target != self.pad_id
        return loss[mask].mean()


def cross_entropy_loss(logits, tgt_out, pad_id, smoothing=0.1) -> torch.Tensor:
    """Built-in label-smoothed CE over the flattened batch (ignores pad)."""
    V = logits.size(-1)
    return F.cross_entropy(
        logits.reshape(-1, V), tgt_out.reshape(-1),
        ignore_index=pad_id, label_smoothing=smoothing,
    )


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
def grad_global_norm(model: nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.detach().norm().item() ** 2
    return total ** 0.5


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


# ---------------------------------------------------------------------------
# The reusable training loop
# ---------------------------------------------------------------------------
@dataclass
class History:
    step: list
    loss: list
    ppl: list
    lr: list
    grad_norm: list
    tok_per_sec: list


def fit(
    model: nn.Module,
    train_loader,
    train_cfg: TrainConfig,
    device: torch.device,
    max_steps: int,
    log_fn=print,
) -> History:
    """Teacher-forced training. Returns a History of logged points.

    Logs loss, perplexity, LR, grad norm, and tokens/sec every `train_cfg.log_every`
    steps. Clips grad norm to `train_cfg.grad_clip`. The AMP branch is CUDA-only.
    """
    model.to(device).train()
    d_model = model.cfg.d_model
    pad_id = model.cfg.pad_id
    opt = make_optimizer(model, train_cfg)
    # AMP is CUDA-only (A100). On MPS/CPU we MUST NOT touch autocast/GradScaler — even a
    # disabled CUDA scaler corrupts MPS numerics. So the default path is plain FP32.
    use_amp = train_cfg.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=True) if use_amp else None

    hist = History([], [], [], [], [], [])
    step = 0
    t0 = time.perf_counter()
    tokens = 0
    done = False
    while not done:
        for batch in train_loader:
            step += 1
            # NOTE: no non_blocking=True — it silently corrupts CPU->MPS transfers
            # (and only helps for pinned-memory -> CUDA copies, which we don't use).
            batch = {k: v.to(device) for k, v in batch.items()}
            lr = lr_at(step, d_model, train_cfg.warmup_steps)
            set_lr(opt, lr)
            opt.zero_grad(set_to_none=True)

            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(batch["src"], batch["tgt_in"], batch["src_pad"], batch["tgt_pad"])
                    loss = cross_entropy_loss(logits, batch["tgt_out"], pad_id, train_cfg.label_smoothing)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                gnorm = grad_global_norm(model)
                nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
                scaler.step(opt)
                scaler.update()
            else:
                logits = model(batch["src"], batch["tgt_in"], batch["src_pad"], batch["tgt_pad"])
                loss = cross_entropy_loss(logits, batch["tgt_out"], pad_id, train_cfg.label_smoothing)
                loss.backward()
                gnorm = grad_global_norm(model)
                nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
                opt.step()

            tokens += int((batch["tgt_out"] != pad_id).sum().item())
            if step % train_cfg.log_every == 0:
                _sync(device)
                dt = time.perf_counter() - t0
                tps = tokens / dt
                lval = loss.item()
                hist.step.append(step); hist.loss.append(lval)
                hist.ppl.append(math.exp(min(lval, 20)))
                hist.lr.append(lr); hist.grad_norm.append(gnorm); hist.tok_per_sec.append(tps)
                log_fn(f"step {step:6d} | loss {lval:6.3f} | ppl {math.exp(min(lval,20)):8.2f} "
                       f"| lr {lr:.2e} | gnorm {gnorm:6.2f} | {tps:8.0f} tok/s")
                t0 = time.perf_counter(); tokens = 0
            if step >= max_steps:
                done = True
                break
    return hist
