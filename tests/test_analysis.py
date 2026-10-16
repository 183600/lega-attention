"""分析收集器测试：等长样例批前向与逐样例结果一致，且记录按原始顺序展平。"""
import math

from lega.analysis import collect_records
from lega.data import CharTokenizer
from lega.model import GPT

_TEXT = ("The quick brown fox jumps over the lazy dog. "
         "A quiet river runs through the old town. 0123456789")


def _tok():
    return CharTokenizer(sorted(set(_TEXT)))


def _example(n, qs, vocab):
    ids = [1 + (i * 7 + 3) % (vocab - 1) for i in range(n)]
    return {"ids": ids, "query_span": (qs, n), "value_span": (n - 4, n - 3)}


def test_collect_records_batching_matches_per_example():
    import torch

    tok = _tok()
    torch.manual_seed(0)
    model = GPT(tok.n_vocab, block_size=64, d_model=32, n_layers=2, n_heads=2,
                variant="lega").eval()
    # 含重复长度与不同长度：覆盖分组堆叠与原始顺序恢复
    lengths = [(24, 20), (24, 19), (20, 16), (16, 12), (24, 18)]
    examples = [_example(n, qs, tok.n_vocab) for n, qs in lengths]
    task_of = ["niah_synth"] * len(examples)
    ctx_of = [64] * len(examples)
    r1 = collect_records(model, tok, "cpu", examples, task_of, ctx_of,
                         amp=False, batch_size=1)
    r8 = collect_records(model, tok, "cpu", examples, task_of, ctx_of,
                         amp=False, batch_size=8)
    assert len(r1) == len(r8) > 0
    for a, b in zip(r1, r8):
        assert a["example"] == b["example"]
        for key, va in a.items():
            vb = b[key]
            if isinstance(va, float):
                assert math.isclose(va, vb, rel_tol=1e-5, abs_tol=1e-6), (key, va, vb)
            else:
                assert va == vb, (key, va, vb)
    # 批处理后各样例记录仍按原始顺序连续排列
    assert [r["example"] for r in r1] == sorted(r["example"] for r in r1)
    print(f"  [ok] {len(r1)} 条记录逐位一致，样例顺序保持")
