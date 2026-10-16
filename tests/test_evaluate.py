"""语言建模坏案例测试：最差验证窗口转储（预测错误的样本要看得见）。"""
import numpy as np
import torch

from lega.data import CharTokenizer
from lega.evaluate import collect_worst_windows
from lega.model import GPT


def test_collect_worst_windows():
    """按末位下一 token 损失排序取最差窗口，字段齐全且降序。"""
    torch.manual_seed(0)
    tok = CharTokenizer(sorted("abcdefghij "))
    model = GPT(vocab_size=len(tok.itos), block_size=64, d_model=32, n_layers=1,
                n_heads=2, variant="lega")
    model.eval()
    # 合成验证流：64 token 窗口 × 32 个
    val = np.random.default_rng(0).integers(0, len(tok.itos), size=64 * 33)
    val = val.astype(np.uint16)
    rows = collect_worst_windows(model, val, ctx=64, device="cpu", tok=tok,
                                 n_worst=3, max_windows=32)
    assert len(rows) == 3
    assert all(set(r) >= {"window", "loss", "context_tail", "gold_next",
                          "pred_top5"} for r in rows)
    assert rows[0]["loss"] >= rows[1]["loss"] >= rows[2]["loss"]
    assert all(len(r["pred_top5"]) == 5 for r in rows)
    # top-5 概率归一（数值容差内）且首候选概率最大
    for r in rows:
        probs = [p for _, p in r["pred_top5"]]
        assert abs(sum(probs) - 1.0) < 0.05, probs
        assert probs == sorted(probs, reverse=True)
        assert r["context_tail"] and isinstance(r["gold_next"], str)
    print(f"  [ok] 最差窗口转储（top loss={rows[0]['loss']:.3f}，"
          f"gold={rows[0]['gold_next']!r}）")
