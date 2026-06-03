"""Joint BPE tokenizer for Multi30k (En+De share one vocab).

A joint vocabulary lets src/tgt share an embedding (and the output projection via
weight tying), and lets shared subwords — names, punctuation, cognates — reuse the
same ids. Special tokens are pinned to ids 0..3 so masking and loss-ignore are stable.

Proven interactively in notebooks/01_data_pipeline.ipynb, then lifted here.
"""

from __future__ import annotations

from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"
SPECIALS = [PAD, BOS, EOS, UNK]
PAD_ID, BOS_ID, EOS_ID, UNK_ID = 0, 1, 2, 3


def train_joint_bpe(pairs: list[tuple[str, str]], vocab_size: int = 10_000) -> Tokenizer:
    """Train a byte-level BPE on the concatenation of all src+tgt sentences.

    `pairs` is a list of (src, tgt) strings. `vocab_size` is auto-capped for tiny
    corpora so the trainer doesn't ask for more merges than exist.
    """
    corpus = [s for src, tgt in pairs for s in (src, tgt)]
    vocab_size = min(vocab_size, max(64, len(set(" ".join(corpus).split())) + len(SPECIALS)))
    tok = Tokenizer(models.BPE(unk_token=UNK))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=SPECIALS, show_progress=False)
    tok.train_from_iterator(corpus, trainer=trainer)
    assert [tok.token_to_id(s) for s in SPECIALS] == [0, 1, 2, 3], "special ids must be 0..3"
    return tok


def save(tok: Tokenizer, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(path))


def load(path: str | Path) -> Tokenizer:
    return Tokenizer.from_file(str(path))
