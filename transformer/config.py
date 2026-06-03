"""Model and training configuration.

`ModelConfig` holds architectural hyperparameters AND the ablation toggles (each
toggle disables one piece so we can watch it fail later, in the Phase-5 ablation
notebook). `TrainConfig` holds the optimization recipe from the paper.

Defaults are set to the **A100 first-run scale** (4+4 layers, d_model=256, h=8,
d_ff=1024, ~15-20M params). Notebooks that run on this Mac instantiate a *smaller*
config via `ModelConfig.smoke()` so a forward/overfit loop is seconds, not minutes.
"""

from __future__ import annotations

from dataclasses import dataclass

# Special-token ids — fixed by the tokenizer (see transformer/tokenizer.py).
PAD_ID, BOS_ID, EOS_ID, UNK_ID = 0, 1, 2, 3


@dataclass
class ModelConfig:
    # Vocab sizes are filled in from the tokenizer after it's built. With a JOINT
    # BPE vocab, src and tgt are equal (and weight tying becomes possible).
    src_vocab_size: int = 0
    tgt_vocab_size: int = 0

    # --- A100 "first run" scale (reduced from the paper's base for fast iteration) ---
    d_model: int = 256
    d_ff: int = 1024
    n_heads: int = 8
    n_encoder_layers: int = 4
    n_decoder_layers: int = 4
    dropout: float = 0.1

    # Max sequence length we precompute sinusoidal PE for (Multi30k 99th pct ~31).
    max_seq_len: int = 128

    pad_id: int = PAD_ID
    tie_embeddings: bool = True  # share tgt embedding with output projection (paper §3.4)

    # --- Ablation toggles (default off = the faithful baseline) ---
    disable_ffn: bool = False      # FFN block becomes identity
    no_scaling: bool = False       # drop the /sqrt(d_k) in attention
    no_causal_mask: bool = False   # decoder peeks at future tokens
    no_cross_attn: bool = False    # decoder ignores encoder memory
    no_positional: bool = False    # don't add positional encoding
    single_head: bool = False      # force h=1 at construction time

    @property
    def d_k(self) -> int:
        h = self.effective_n_heads
        assert self.d_model % h == 0, "d_model must be divisible by n_heads"
        return self.d_model // h

    @property
    def effective_n_heads(self) -> int:
        return 1 if self.single_head else self.n_heads

    @classmethod
    def smoke(cls, **overrides) -> "ModelConfig":
        """Tiny config for fast local notebook runs (seconds on MPS/CPU)."""
        base = dict(d_model=128, d_ff=512, n_heads=4,
                    n_encoder_layers=2, n_decoder_layers=2, max_seq_len=64)
        base.update(overrides)
        return cls(**base)

    @classmethod
    def base(cls, **overrides) -> "ModelConfig":
        """Full paper 'base' config (~65M params) — for the final A100 fidelity run."""
        base = dict(d_model=512, d_ff=2048, n_heads=8,
                    n_encoder_layers=6, n_decoder_layers=6, max_seq_len=128)
        base.update(overrides)
        return cls(**base)


@dataclass
class TrainConfig:
    batch_size: int = 128
    max_steps: int = 100_000
    warmup_steps: int = 4000          # Noam warmup (paper)
    label_smoothing: float = 0.1
    adam_betas: tuple[float, float] = (0.9, 0.98)
    adam_eps: float = 1e-9
    seed: int = 1337
    log_every: int = 50
    grad_clip: float = 1.0            # clip grad norm; watch the norm for explosions
    checkpoint_dir: str = "checkpoints"
    run_name: str = "baseline"
    amp: bool = False                 # mixed precision — OFF until the FP32 run works (A100 toggle)
