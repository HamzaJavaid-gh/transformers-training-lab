import math
import torch
from transformer.config import ModelConfig
from transformer.masks import build_src_mask, build_tgt_mask, causal_mask
from transformer.multi_head_attention import MultiHeadAttention
from transformer.positional_encoding import SinusoidalPositionalEncoding


def test_positional_encoding_shape_and_sin_cos():
    pe = SinusoidalPositionalEncoding(d_model=64, max_len=50)
    x = torch.zeros(2, 30, 64)
    assert pe(x).shape == x.shape
    buf = pe.pe[0]  # (max_len, d_model)
    # position 0: sin(0)=0 on even dims, cos(0)=1 on odd dims
    assert torch.allclose(buf[0, 0::2], torch.zeros(32), atol=1e-6)
    assert torch.allclose(buf[0, 1::2], torch.ones(32), atol=1e-6)


def test_multi_head_attention_output_shape():
    cfg = ModelConfig(src_vocab_size=100, tgt_vocab_size=100, d_model=64, n_heads=4, dropout=0.0)
    mha = MultiHeadAttention(cfg).eval()
    x = torch.randn(2, 7, 64)
    assert mha(x, x, x).shape == x.shape


def test_causal_mask_blocks_future():
    T = 6
    mask = causal_mask(T, device=torch.device("cpu"))
    assert mask.shape == (1, 1, T, T)
    for i in range(T):
        for j in range(T):
            expected = float("-inf") if j > i else 0.0
            assert mask[0, 0, i, j] == expected
