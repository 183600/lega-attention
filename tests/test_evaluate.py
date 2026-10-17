"""语言建模坏案例测试：最差验证窗口转储（预测错误的样本要看得见）。"""
import json
import os
import shutil
import tempfile

import numpy as np
import torch

from lega.data import CharTokenizer
from lega.evaluate import collect_worst_windows, load_train_throughput
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
    # top-5 为全词表 softmax 的截断：逐项概率降序、均在 (0,1]，总和不超过 1
    for r in rows:
        probs = [p for _, p in r["pred_top5"]]
        assert all(0.0 < p <= 1.0 for p in probs), probs
        assert sum(probs) <= 1.0 + 1e-9, probs
        assert probs == sorted(probs, reverse=True)
        assert r["context_tail"] and isinstance(r["gold_next"], str)
    print(f"  [ok] 最差窗口转储（top loss={rows[0]['loss']:.3f}，"
          f"gold={rows[0]['gold_next']!r}）")


def test_load_train_throughput_propagates_and_degrades_safely():
    """训练步时记录必须随被评测 checkpoint 传播；缺失/损坏时安全返回 (None, None)。"""
    tmp = tempfile.mkdtemp()
    try:
        assert load_train_throughput(tmp) == (None, None)
        with open(os.path.join(tmp, "eval.json"), "w", encoding="utf-8") as f:
            json.dump({"train_step_time_ms": 12.5,
                       "train_tokens_per_sec": 81920.0}, f)
        assert load_train_throughput(tmp) == (12.5, 81920.0)
        # 旧格式（无该字段）→ None 而非 KeyError
        with open(os.path.join(tmp, "eval.json"), "w", encoding="utf-8") as f:
            json.dump({"val_ppl": 10.0}, f)
        assert load_train_throughput(tmp) == (None, None)
        # 损坏 JSON 不阻塞评测
        with open(os.path.join(tmp, "eval.json"), "w", encoding="utf-8") as f:
            f.write("{broken")
        assert load_train_throughput(tmp) == (None, None)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] 训练步时记录随 checkpoint 传播（缺失/损坏安全降级）")


def test_collect_worst_windows_gold_in_context_uses_full_window():
    """gold_in_context 必须按完整窗口判定，而不是只看 160 字符的上下文尾部。"""
    torch.manual_seed(0)
    tok = CharTokenizer(sorted("abcdefghij "))
    model = GPT(vocab_size=len(tok.itos), block_size=64, d_model=32, n_layers=1,
                n_heads=2, variant="lega")
    model.eval()
    val = np.random.default_rng(0).integers(0, len(tok.itos), size=64 * 33)
    val = val.astype(np.uint16)
    rows = collect_worst_windows(model, val, ctx=64, device="cpu", tok=tok,
                                 n_worst=3, max_windows=32)
    for r in rows:
        w = int(r["window"])
        full = tok.decode(val[w * 64:(w + 1) * 64].tolist())
        gold = tok.decode([int(val[w * 64 + 64])])
        assert r["context_tail"] == full[-160:]
        assert r["gold_in_context"] == (bool(gold) and gold in full), r
    print("  [ok] 最差窗口 gold_in_context 按完整窗口判定")
