"""输出级干预测试：答案 NLL/margin 的逐头写回效用，而不是 hit 代理。"""
import torch

from lega.model import GPT
from lega.output_ablation import (head_output_intervention_scan, score_answer,
                                  score_answer_from_logits)


def _tiny():
    torch.manual_seed(0)
    return GPT(vocab_size=64, block_size=32, d_model=16, n_layers=2,
               n_heads=2, variant="lega", use_sdpa_standard=False).eval()


def test_main_rejects_fractional_answer_positions(monkeypatch):
    """JSON 里的 answer_start/end 与函数入口同口径，不能被 int() 静默截断。"""
    import json
    import os
    import tempfile

    import lega.model as model_mod
    from lega.output_ablation import main

    model = _tiny()
    monkeypatch.setattr(model_mod, "load_model",
                        lambda *a, **k: (model, {}))
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "idx.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"input_ids": [list(range(12))],
                       "answer_start": 1.5, "answer_end": 4}, f)
        try:
            main(["--ckpt", "unused", "--idx_json", path])
        except ValueError as e:
            assert "answer_start" in str(e), e
        else:
            raise AssertionError("小数 answer_start 必须被拒绝")


def test_score_answer_matches_logits_reuse():
    model = _tiny()
    idx = torch.randint(0, 64, (1, 12))
    a = score_answer(model, idx, 6, 10)
    out = model(idx)
    b = score_answer_from_logits(out["logits"], idx, 6, 10)
    assert a["n_answer_tokens"] == 4
    for key in ("nll", "prob", "margin"):
        assert abs(a[key] - b[key]) < 1e-6, key


def test_head_scan_reports_only_the_target_head_mass():
    """mean_abs_removed 必须是目标头通道的幅度，而不是所有头的均值。"""
    model = _tiny()
    idx = torch.randint(0, 64, (1, 16))
    start, end = 8, 13
    rep = head_output_intervention_scan(
        model, idx, start, end, layers=[0], heads=[0, 1], modes=("zero",))

    captured = {}

    def hook(module, args):
        captured["x"] = args[0].detach().clone()

    handle = model.blocks[0].attn.wo.register_forward_pre_hook(hook)
    try:
        with torch.no_grad():
            model(idx)
    finally:
        handle.remove()
    x = captured["x"]
    positions = list(range(start - 1, end - 1))
    dh = model.blocks[0].attn.d_head
    for row in rep["interventions"]:
        lo = row["head"] * dh
        expected = x[:, positions, lo:lo + dh].abs().mean().item()
        assert abs(row["mean_abs_removed"] - expected) < 1e-6, row
    # 两个头不可能恰好有相同的通道幅度；相同说明仍在拿全通道均值。
    vals = {r["head"]: r["mean_abs_removed"]
            for r in rep["interventions"]}
    assert vals[0] != vals[1], vals
    print("  [ok] 逐头干预的 mean_abs_removed 只统计目标头通道")


def test_head_output_scan_reports_paired_deltas_and_permutation():
    model = _tiny()
    idx = torch.randint(0, 64, (1, 16))
    rep = head_output_intervention_scan(
        model, idx, 8, 13, layers=[0, 1], heads=[0, 1],
        scope="answer", modes=("zero", "permute"))
    rows = rep["interventions"]
    assert len(rows) == 8
    base = rep["baseline"]
    assert base["n_answer_tokens"] == 5
    for row in rows:
        assert row["delta_nll"] == row["nll"] - base["nll"]
        assert row["n_positions"] == 5
        assert row["mean_abs_removed"] >= 0.0
    # 至少有一个头在答案位置的写回会影响答案 NLL；否则“逐头写回效用”
    # 在这个随机小模型上没有任何可测变化，测试也就失去意义。
    assert any(abs(r["delta_nll"]) > 1e-7 for r in rows)
    # 置换模式必须真有位置可换，且不应产生 NaN。
    for row in rows:
        assert row["mode"] in ("zero", "permute")
        assert torch.isfinite(torch.tensor(row["delta_nll"]))
