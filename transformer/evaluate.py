"""Corpus BLEU evaluation with sacrebleu.

Reusable for both the notebook and the A100 checkpoint. Greedy is batched; beam runs
one sentence at a time (it's the slow, careful path).
"""

from __future__ import annotations

import torch
from torch.nn.utils.rnn import pad_sequence

from transformer.data import TranslationDataset
from transformer.generate import beam_search, greedy_decode
from transformer.tokenizer import BOS_ID, EOS_ID, PAD_ID


def _clean(ids: list[int], tokenizer) -> str:
    ids = [i for i in ids if i not in (BOS_ID, EOS_ID, PAD_ID)]
    return tokenizer.decode(ids)


def _encode_sources(sources: list[str], tokenizer, max_len: int) -> list[torch.Tensor]:
    ds = TranslationDataset([(s, "") for s in sources], tokenizer, max_len)
    return [ds[i][0] for i in range(len(ds))]


@torch.no_grad()
def translate_corpus(
    model, sources, tokenizer, device,
    decode: str = "greedy", beam_size: int = 4, alpha: float = 0.6,
    max_len: int = 64, batch_size: int = 32,
) -> list[str]:
    """Translate a list of source strings. decode in {'greedy', 'beam'}."""
    model.eval()
    src_tensors = _encode_sources(sources, tokenizer, max_len)
    hyps: list[str] = []

    if decode == "greedy":
        for i in range(0, len(src_tensors), batch_size):
            chunk = src_tensors[i: i + batch_size]
            src = pad_sequence(chunk, batch_first=True, padding_value=PAD_ID).to(device)
            src_pad = src.eq(PAD_ID).to(device)
            out = greedy_decode(model, src, src_pad, BOS_ID, EOS_ID, PAD_ID, max_len)
            hyps += [_clean(row.tolist(), tokenizer) for row in out]
    elif decode == "beam":
        for t in src_tensors:
            src = t.unsqueeze(0).to(device)
            src_pad = src.eq(PAD_ID).to(device)
            ids = beam_search(model, src, src_pad, BOS_ID, EOS_ID, PAD_ID, beam_size, alpha, max_len)
            hyps.append(_clean(ids, tokenizer))
    else:
        raise ValueError(f"unknown decode mode: {decode!r}")
    return hyps


def corpus_bleu(model, pairs, tokenizer, device, **kw):
    """Return (sacrebleu BLEU object, hypotheses). `pairs` is a list of (src, ref)."""
    import sacrebleu

    sources = [s for s, _ in pairs]
    refs = [r for _, r in pairs]
    hyps = translate_corpus(model, sources, tokenizer, device, **kw)
    bleu = sacrebleu.corpus_bleu(hyps, [refs])
    return bleu, hyps
