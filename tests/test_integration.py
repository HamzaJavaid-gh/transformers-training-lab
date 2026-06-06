import torch
from transformer.config import ModelConfig
from transformer.train import LabelSmoothingLoss
from transformer.transformer import Transformer

VOCAB = 200


def test_full_forward_pass_and_overfit():
    torch.manual_seed(0)
    cfg = ModelConfig.smoke(src_vocab_size=VOCAB, tgt_vocab_size=VOCAB, dropout=0.0)
    model = Transformer(cfg).train()

    B, S, T = 2, 6, 5
    src     = torch.randint(4, VOCAB, (B, S))
    tgt_in  = torch.randint(4, VOCAB, (B, T))
    tgt_out = torch.randint(4, VOCAB, (B, T))
    src_pad = torch.zeros(B, S, dtype=torch.bool)
    tgt_pad = torch.zeros(B, T, dtype=torch.bool)

    # shape check
    logits = model(src, tgt_in, src_pad, tgt_pad)
    assert logits.shape == (B, T, VOCAB)

    # overfit check: loss should drop at least 5x in 100 steps
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    criterion = LabelSmoothingLoss(pad_id=cfg.pad_id, smoothing=0.1)

    initial_loss = best_loss = None
    for _ in range(100):
        optimizer.zero_grad()
        loss = criterion(model(src, tgt_in, src_pad, tgt_pad).reshape(-1, VOCAB), tgt_out.reshape(-1))
        loss.backward()
        optimizer.step()
        v = loss.item()
        if initial_loss is None:
            initial_loss = v
        if best_loss is None or v < best_loss:
            best_loss = v

    assert best_loss < initial_loss * 0.2, f"{initial_loss:.3f} -> {best_loss:.3f}"
