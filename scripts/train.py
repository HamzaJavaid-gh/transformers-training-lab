"""A100 training entrypoint for the Transformer.

Reuses the SAME components proven in the notebooks (transformer/*). This script adds the
production-loop concerns the notebooks skip: validation loss, periodic BLEU, checkpointing
(best + last), resume, and — so you can shut the GPU down and keep everything — full
artifact capture into runs/<run-name>/. Plain argparse + json + matplotlib, no W&B (CLAUDE.md).

Artifacts written per run (runs/<run-name>/):
  train.log         every stdout/stderr line (tee'd live — survives closing the terminal)
  metrics.jsonl     one JSON object per log/eval event (append-only, crash-safe)
  summary.json      full metric history + config (rewritten at each eval and at the end)
  samples.txt       sample val translations captured at each eval
  figures/          curves.png (loss/lr/grad-norm/BLEU) + attention_step*.png (evolving)

Examples
--------
  uv run python scripts/train.py --preset smoke --max-steps 200 --run-name smoke
  uv run python scripts/train.py --preset first-run --max-steps 100000 --batch-size 128 \
      --amp --run-name first_run
  uv run python scripts/train.py --preset first-run --resume checkpoints/first_run.last.pt
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: render figures to file, no display needed
import matplotlib.pyplot as plt  # noqa: E402

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

# Make `import transformer` work whether run from repo root or scripts/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tokenizers import Tokenizer  # noqa: E402

from transformer.config import ModelConfig, TrainConfig  # noqa: E402
from transformer.data import TranslationDataset, collate_fn, load_multi30k, make_dataloader  # noqa: E402
from transformer.evaluate import corpus_bleu, translate_corpus  # noqa: E402
from transformer.tokenizer import BOS_ID, EOS_ID, PAD_ID, train_joint_bpe  # noqa: E402
from transformer.train import (  # noqa: E402
    cross_entropy_loss, grad_global_norm, lr_at, make_optimizer, set_lr,
)
from transformer.transformer import Transformer  # noqa: E402


# ---------------------------------------------------------------------------
# Artifact capture helpers
# ---------------------------------------------------------------------------
class _Tee:
    """Write to several streams at once (console + logfile), flushing eagerly."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def append_jsonl(path: Path, obj: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(obj) + "\n")


def save_curves(fig_dir: Path, train_hist: dict, eval_hist: dict) -> None:
    fig, ax = plt.subplots(2, 2, figsize=(12, 7))
    if train_hist["step"]:
        ax[0, 0].plot(train_hist["step"], train_hist["loss"]); ax[0, 0].set_title("train loss")
        ax[0, 1].plot(train_hist["step"], train_hist["lr"]);   ax[0, 1].set_title("learning rate (Noam)")
        ax[1, 0].plot(train_hist["step"], train_hist["grad_norm"]); ax[1, 0].set_title("grad norm")
    if eval_hist["step"]:
        ax[1, 1].plot(eval_hist["step"], eval_hist["bleu"], marker="o"); ax[1, 1].set_title("val BLEU")
    for a in ax.flat:
        a.set_xlabel("step")
    fig.tight_layout()
    fig.savefig(fig_dir / "curves.png", dpi=110)
    plt.close(fig)


@torch.no_grad()
def save_attention(model, src_text, tokenizer, device, path: Path, max_len: int) -> None:
    """Greedy-decode one fixed example and save its cross-attention alignment heatmap."""
    model.eval()
    b = collate_fn([TranslationDataset([(src_text, "")], tokenizer, max_len)[0]])
    src, src_pad = b["src"].to(device), b["src_pad"].to(device)
    layer = model.decoder.layers[-1].cross_attn
    layer.cache_attn = True
    memory = model.encode(src, src_pad)
    ys = torch.tensor([[BOS_ID]], device=device)
    rows = []
    for _ in range(max_len - 1):
        decoded = model.decode(ys, memory, ys.eq(PAD_ID), src_pad)
        rows.append(layer.last_attn[0].mean(0)[-1].cpu())  # mean over heads, newest query -> (S,)
        nxt = model.output_proj(decoded[:, -1]).argmax(-1)
        ys = torch.cat([ys, nxt.unsqueeze(1)], dim=1)
        if nxt.item() == EOS_ID:
            break
    layer.cache_attn = False
    align = torch.stack(rows)
    src_toks = [tokenizer.id_to_token(i) for i in src[0].tolist()]
    gen_toks = [tokenizer.id_to_token(i) for i in ys[0, 1:].tolist()]
    fig, a = plt.subplots(figsize=(8, 5))
    im = a.imshow(align, cmap="viridis", aspect="auto")
    a.set_xticks(range(len(src_toks))); a.set_xticklabels(src_toks, rotation=90, fontsize=6)
    a.set_yticks(range(len(gen_toks))); a.set_yticklabels(gen_toks, fontsize=6)
    a.set_xlabel("source (EN)"); a.set_ylabel("generated (DE)")
    a.set_title(f"cross-attention\n{src_text}")
    fig.colorbar(im, ax=a, shrink=0.7)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
def get_device(name: str | None) -> torch.device:
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_config(preset: str, vocab: int, args) -> ModelConfig:
    overrides = dict(src_vocab_size=vocab, tgt_vocab_size=vocab, dropout=args.dropout,
                     max_seq_len=max(args.max_len + 8, 128))
    if preset == "smoke":
        return ModelConfig.smoke(**overrides)
    if preset == "base":
        return ModelConfig.base(**overrides)
    return ModelConfig(**overrides)  # "first-run" defaults


def move(batch, device):
    nb = device.type == "cuda"  # non_blocking is CUDA-only (corrupts MPS — see CLAUDE.md)
    return {k: v.to(device, non_blocking=nb) for k, v in batch.items()}


@torch.no_grad()
def eval_val_loss(model, loader, device, smoothing) -> float:
    model.eval()
    pad_id = model.cfg.pad_id
    total_loss, total_tok = 0.0, 0
    for batch in loader:
        batch = move(batch, device)
        logits = model(batch["src"], batch["tgt_in"], batch["src_pad"], batch["tgt_pad"])
        ntok = int((batch["tgt_out"] != pad_id).sum().item())
        loss = cross_entropy_loss(logits, batch["tgt_out"], pad_id, smoothing)
        total_loss += loss.item() * ntok
        total_tok += ntok
    return total_loss / max(total_tok, 1)


def save_ckpt(path, model, opt, step, best_val, tokenizer, train_cfg):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_config": model.cfg.__dict__,
        "optimizer_state_dict": opt.state_dict(),
        "step": step,
        "best_val_loss": best_val,
        "tokenizer": tokenizer.to_str(),
        "train_config": train_cfg.__dict__,
    }, path)


def main():
    p = argparse.ArgumentParser(description="Train the Transformer on Multi30k.")
    p.add_argument("--preset", choices=["smoke", "first-run", "base"], default="first-run")
    p.add_argument("--run-name", default="first_run")
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--max-steps", type=int, default=100_000)
    p.add_argument("--warmup-steps", type=int, default=4000)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--max-len", type=int, default=64)
    p.add_argument("--vocab-size", type=int, default=10_000)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=2000)
    p.add_argument("--bleu-sentences", type=int, default=200, help="val sentences for periodic BLEU")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--amp", action="store_true", help="mixed precision (CUDA only)")
    p.add_argument("--no-figures", action="store_true", help="skip PNG figure generation")
    p.add_argument("--device", default=None)
    p.add_argument("--resume", default=None, help="path to a .pt checkpoint to resume from")
    args = p.parse_args()

    # --- artifact capture: tee all output to runs/<run-name>/train.log ---
    run_dir = Path(args.runs_dir) / args.run_name
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    logfile = open(run_dir / "train.log", "a")
    sys.stdout = _Tee(sys.__stdout__, logfile)
    sys.stderr = _Tee(sys.__stderr__, logfile)

    device = get_device(args.device)
    torch.manual_seed(args.seed)
    print(f"\n===== run '{args.run_name}' | device {device} | preset {args.preset} =====")
    print(f"artifacts -> {run_dir}/  (train.log, metrics.jsonl, summary.json, samples.txt, figures/)")
    print("args:", json.dumps(vars(args)))

    # --- data + tokenizer ---
    data, online = load_multi30k()
    print(f"data: online={online} train={len(data['train'])} "
          f"val={len(data['validation'])} test={len(data['test'])}")

    start_step, best_val, resume_blob = 0, float("inf"), None
    if args.resume:
        resume_blob = torch.load(args.resume, map_location=device, weights_only=False)
        tokenizer = Tokenizer.from_str(resume_blob["tokenizer"])
        print(f"resuming from {args.resume} @ step {resume_blob['step']}")
    else:
        tokenizer = train_joint_bpe(data["train"], vocab_size=args.vocab_size)
    vocab = tokenizer.get_vocab_size()
    print(f"vocab: {vocab}")

    train_loader = make_dataloader(data["train"], tokenizer, batch_size=args.batch_size,
                                   shuffle=True, max_len=args.max_len,
                                   num_workers=args.num_workers,
                                   pin_memory=(device.type == "cuda"))
    val_loader = make_dataloader(data["validation"], tokenizer, batch_size=args.batch_size,
                                 shuffle=False, max_len=args.max_len)

    # --- model + optimizer ---
    cfg = build_config(args.preset, vocab, args)
    if resume_blob:
        cfg = ModelConfig(**resume_blob["model_config"])
    model = Transformer(cfg).to(device)
    opt = make_optimizer(model, TrainConfig(adam_betas=(0.9, 0.98), adam_eps=1e-9))
    if resume_blob:
        model.load_state_dict(resume_blob["model_state_dict"])
        opt.load_state_dict(resume_blob["optimizer_state_dict"])
        start_step = resume_blob["step"]
        best_val = resume_blob["best_val_loss"]
    print(f"model: {model.num_parameters():,} params | d_model={cfg.d_model} "
          f"h={cfg.n_heads} {cfg.n_encoder_layers}+{cfg.n_decoder_layers}")

    train_cfg = TrainConfig(batch_size=args.batch_size, max_steps=args.max_steps,
                            warmup_steps=args.warmup_steps, label_smoothing=args.label_smoothing,
                            grad_clip=args.grad_clip, log_every=args.log_every,
                            checkpoint_dir=args.checkpoint_dir, run_name=args.run_name, amp=args.amp)

    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=True) if use_amp else None
    if args.amp and not use_amp:
        print("warning: --amp requested but device is not CUDA; running FP32.")

    ckpt_best = Path(args.checkpoint_dir) / f"{args.run_name}.best.pt"
    ckpt_last = Path(args.checkpoint_dir) / f"{args.run_name}.last.pt"

    # histories for figures + summary.json
    train_hist = {"step": [], "loss": [], "ppl": [], "lr": [], "grad_norm": [], "tok_per_sec": []}
    eval_hist = {"step": [], "val_loss": [], "val_ppl": [], "bleu": []}
    # fixed examples so attention/sample figures are comparable across steps
    fixed_srcs = [s for s, _ in data["validation"][:5]]
    attn_example = fixed_srcs[0]

    def write_summary(final=False):
        summary = {
            "run_name": args.run_name, "preset": args.preset, "final": final,
            "params": model.num_parameters(), "config": cfg.__dict__, "args": vars(args),
            "best_val_loss": best_val, "last_step": step,
            "train_hist": train_hist, "eval_hist": eval_hist,
        }
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    def run_eval():
        nonlocal best_val
        vloss = eval_val_loss(model, val_loader, device, args.label_smoothing)
        bleu, _ = corpus_bleu(model, data["validation"][:args.bleu_sentences],
                              tokenizer, device, decode="greedy", max_len=args.max_len)
        eval_hist["step"].append(step); eval_hist["val_loss"].append(vloss)
        eval_hist["val_ppl"].append(math.exp(min(vloss, 20))); eval_hist["bleu"].append(bleu.score)

        # sample translations (greedy) on the fixed examples
        hyps = translate_corpus(model, fixed_srcs, tokenizer, device, decode="greedy", max_len=args.max_len)
        with open(run_dir / "samples.txt", "a") as f:
            f.write(f"\n===== step {step} | val_loss {vloss:.3f} | BLEU {bleu.score:.2f} =====\n")
            for (src, ref), hyp in zip(data["validation"][:5], hyps):
                f.write(f"EN : {src}\nref: {ref}\nhyp: {hyp}\n\n")

        tag = ""
        if vloss < best_val:
            best_val = vloss
            save_ckpt(ckpt_best, model, opt, step, best_val, tokenizer, train_cfg)
            tag = "  <- new best (saved)"
        save_ckpt(ckpt_last, model, opt, step, best_val, tokenizer, train_cfg)

        append_jsonl(metrics_path, {"type": "eval", "step": step, "val_loss": vloss,
                                    "val_ppl": math.exp(min(vloss, 20)), "bleu": bleu.score})
        if not args.no_figures:
            save_curves(fig_dir, train_hist, eval_hist)
            save_attention(model, attn_example, tokenizer, device,
                           fig_dir / f"attention_step{step:06d}.png", args.max_len)
        write_summary()
        print(f"  [eval @ {step}] val_loss {vloss:.3f} | val_ppl {math.exp(min(vloss,20)):.2f} "
              f"| BLEU {bleu.score:.2f}{tag}")
        model.train()

    # --- training loop ---
    model.train()
    pad_id = cfg.pad_id
    step = start_step
    t0, tokens = time.perf_counter(), 0
    done = False
    print(f"training to {args.max_steps} steps...\n")
    while not done:
        for batch in train_loader:
            step += 1
            batch = move(batch, device)
            set_lr(opt, lr_at(step, cfg.d_model, args.warmup_steps))
            opt.zero_grad(set_to_none=True)

            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(batch["src"], batch["tgt_in"], batch["src_pad"], batch["tgt_pad"])
                    loss = cross_entropy_loss(logits, batch["tgt_out"], pad_id, args.label_smoothing)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                gnorm = grad_global_norm(model)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(opt)
                scaler.update()
            else:
                logits = model(batch["src"], batch["tgt_in"], batch["src_pad"], batch["tgt_pad"])
                loss = cross_entropy_loss(logits, batch["tgt_out"], pad_id, args.label_smoothing)
                loss.backward()
                gnorm = grad_global_norm(model)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                opt.step()

            tokens += int((batch["tgt_out"] != pad_id).sum().item())

            if step % args.log_every == 0:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                dt = time.perf_counter() - t0
                lr = lr_at(step, cfg.d_model, args.warmup_steps)
                lval, tps = loss.item(), tokens / dt
                train_hist["step"].append(step); train_hist["loss"].append(lval)
                train_hist["ppl"].append(math.exp(min(lval, 20))); train_hist["lr"].append(lr)
                train_hist["grad_norm"].append(gnorm); train_hist["tok_per_sec"].append(tps)
                append_jsonl(metrics_path, {"type": "train", "step": step, "loss": lval,
                                            "ppl": math.exp(min(lval, 20)), "lr": lr,
                                            "grad_norm": gnorm, "tok_per_sec": tps})
                print(f"step {step:6d} | loss {lval:6.3f} | ppl {math.exp(min(lval,20)):8.2f} "
                      f"| lr {lr:.2e} | gnorm {gnorm:6.2f} | {tps:8.0f} tok/s")
                t0, tokens = time.perf_counter(), 0

            if step % args.eval_every == 0 or step >= args.max_steps:
                run_eval()

            if step >= args.max_steps:
                done = True
                break

    write_summary(final=True)
    print(f"\ndone. best val_loss {best_val:.3f}")
    print(f"checkpoints: {ckpt_best}  (best)  |  {ckpt_last}  (last)")
    print(f"artifacts:   {run_dir}/  (train.log, metrics.jsonl, summary.json, samples.txt, figures/)")
    logfile.close()


if __name__ == "__main__":
    main()
