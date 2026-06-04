"""Translate / evaluate from a trained checkpoint.

Examples
--------
  # translate sentences from the CLI
  uv run python scripts/translate.py --checkpoint checkpoints/first_run.best.pt \
      --beam 4 "A man is riding a bike." "Two dogs play in the snow."

  # translate from stdin (one sentence per line)
  echo "A child is playing." | uv run python scripts/translate.py -c checkpoints/first_run.best.pt

  # corpus BLEU on the full test set
  uv run python scripts/translate.py -c checkpoints/first_run.best.pt --bleu --beam 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tokenizers import Tokenizer  # noqa: E402

from transformer.config import ModelConfig  # noqa: E402
from transformer.data import load_multi30k  # noqa: E402
from transformer.evaluate import corpus_bleu, translate_corpus  # noqa: E402
from transformer.transformer import Transformer  # noqa: E402


def get_device(name):
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load(checkpoint, device):
    blob = torch.load(checkpoint, map_location=device, weights_only=False)
    tokenizer = Tokenizer.from_str(blob["tokenizer"])
    cfg = ModelConfig(**blob["model_config"])
    model = Transformer(cfg).to(device)
    model.load_state_dict(blob["model_state_dict"])
    model.eval()
    return model, tokenizer, blob.get("step")


def main():
    p = argparse.ArgumentParser(description="Translate or evaluate from a checkpoint.")
    p.add_argument("-c", "--checkpoint", required=True)
    p.add_argument("--beam", type=int, default=1, help="beam size (1 = greedy)")
    p.add_argument("--alpha", type=float, default=0.6, help="length penalty (beam only)")
    p.add_argument("--max-len", type=int, default=64)
    p.add_argument("--bleu", action="store_true", help="compute corpus BLEU on the test set")
    p.add_argument("--device", default=None)
    p.add_argument("sentences", nargs="*", help="sentences to translate (else read stdin)")
    args = p.parse_args()

    device = get_device(args.device)
    model, tokenizer, step = load(args.checkpoint, device)
    decode = "beam" if args.beam > 1 else "greedy"
    print(f"loaded {args.checkpoint} (step {step}) on {device} | decode={decode}"
          f"{f' beam={args.beam}' if decode == 'beam' else ''}", file=sys.stderr)

    if args.bleu:
        test = load_multi30k()[0]["test"]
        bleu, _ = corpus_bleu(model, test, tokenizer, device,
                              decode=decode, beam_size=args.beam, alpha=args.alpha, max_len=args.max_len)
        print(f"test BLEU ({len(test)} sentences): {bleu.score:.2f}")
        print("signature:", bleu)
        return

    sources = args.sentences or [ln.strip() for ln in sys.stdin if ln.strip()]
    if not sources:
        print("no input sentences", file=sys.stderr)
        return
    hyps = translate_corpus(model, sources, tokenizer, device,
                            decode=decode, beam_size=args.beam, alpha=args.alpha, max_len=args.max_len)
    for src, hyp in zip(sources, hyps):
        print(f"EN: {src}")
        print(f"DE: {hyp}\n")


if __name__ == "__main__":
    main()
