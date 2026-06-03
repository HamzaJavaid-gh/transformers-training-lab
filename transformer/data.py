"""Dataset, collate, and dataloaders for Multi30k.

Proven interactively in notebooks/01_data_pipeline.ipynb, then lifted here.

Masks: the collate returns BOOLEAN pad masks (True = is-pad = blocked). The model
(transformer/masks.py) converts those to the additive -inf form it needs. Keeping the
data layer boolean keeps it backend-agnostic and easy to visualize. The teacher-forcing
shift (tgt_in / tgt_out) also happens here.
"""

from __future__ import annotations

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from transformer.tokenizer import BOS_ID, EOS_ID, PAD_ID

# A tiny offline sample so the pipeline always runs without network.
TINY_FALLBACK = [
    ("A man in a blue shirt is standing on a ladder cleaning a window.",
     "Ein Mann in einem blauen Hemd steht auf einer Leiter und putzt ein Fenster."),
    ("A group of people are sitting outside a building.",
     "Eine Gruppe von Menschen sitzt vor einem Gebäude."),
    ("Two young men are smiling and laughing in a park.",
     "Zwei junge Männer lächeln und lachen in einem Park."),
    ("A little girl is running across a grassy field.",
     "Ein kleines Mädchen rennt über eine Wiese."),
    ("A dog jumps to catch a red ball in the air.",
     "Ein Hund springt, um einen roten Ball in der Luft zu fangen."),
    ("Three children are playing soccer on the beach.",
     "Drei Kinder spielen Fußball am Strand."),
]


def load_multi30k() -> tuple[dict[str, list[tuple[str, str]]], bool]:
    """Return (dict split->list[(en, de)], online?). Falls back to TINY_FALLBACK offline."""
    try:
        from datasets import load_dataset

        ds = load_dataset("bentrevett/multi30k")
        out = {s: list(zip(ds[s]["en"], ds[s]["de"]))
               for s in ("train", "validation", "test")}
        return out, True
    except Exception:
        data = TINY_FALLBACK * 8
        return {"train": data, "validation": TINY_FALLBACK[:4], "test": TINY_FALLBACK[:4]}, False


class TranslationDataset(Dataset):
    """Yields (src_ids, tgt_ids). Target is wrapped <bos> ... <eos>; src ends with <eos>."""

    def __init__(self, pairs, tokenizer, max_len: int = 64):
        self.pairs = pairs
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.pairs)

    def _encode(self, text: str, add_bos: bool) -> torch.Tensor:
        ids = self.tok.encode(text).ids[: self.max_len - 2]
        body = ([BOS_ID] if add_bos else []) + ids + [EOS_ID]
        return torch.tensor(body, dtype=torch.long)

    def __getitem__(self, i: int):
        en, de = self.pairs[i]
        return self._encode(en, add_bos=False), self._encode(de, add_bos=True)


def collate_fn(batch):
    """Pad to max-in-batch; build the teacher-forcing shift and boolean pad masks."""
    srcs, tgts = zip(*batch)
    src = pad_sequence(srcs, batch_first=True, padding_value=PAD_ID)   # (B, S)
    tgt = pad_sequence(tgts, batch_first=True, padding_value=PAD_ID)   # (B, T+1)
    tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]                          # position t predicts t+1
    return {
        "src": src,
        "tgt_in": tgt_in,
        "tgt_out": tgt_out,
        "src_pad": src.eq(PAD_ID),       # (B, S)  True = blocked
        "tgt_pad": tgt_in.eq(PAD_ID),    # (B, T)
    }


def make_dataloader(pairs, tokenizer, batch_size=128, shuffle=True, max_len=64, **kw) -> DataLoader:
    ds = TranslationDataset(pairs, tokenizer, max_len=max_len)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn, **kw)
