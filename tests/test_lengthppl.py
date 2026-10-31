"""自然文本 PPL 长度审计的纯统计、配对目标与 CLI 测试。"""
import json
import math
import os
import shutil
import sys
import tempfile

import numpy as np
import torch
import torch.nn.functional as F

from lega.lengthppl import (analyze_length_ppl_records,
                            collect_length_ppl_records,
                            compare_length_ppl_payloads,
                            compare_length_ppl_records)
from lega.model import GPT
from lega.pipeline.report.sections import _render_length_ppl
from lega.pipeline.report.summary import _length_ppl_interaction_rows
from lega.utils import set_seed


def _records(slope, lengths=(4, 8, 16), windows=3, base=2.0,
             target_ids=(7, 8)):
    recs = []
    for w in range(windows):
        for length in lengths:
            nll = base + w * 0.05 + slope * math.log2(length)
            recs.append({
                "window": w,
                "window_start": w * 16,
                "input_start": w * 16,
                "length": length,
                "query_positions": len(target_ids),
                "target_ids": list(target_ids),
                "nll": nll,
                "ppl": math.exp(nll),
            })
    return recs


def _payload(variant, slope, lengths=(4, 8, 16), windows=3):
    recs = _records(slope, lengths=lengths, windows=windows)
    result = analyze_length_ppl_records(
        recs, lengths, "suffix", n_bootstrap=0)
    return {
        "variant": variant,
        "lengths": list(lengths),
        "results": {"suffix": result},
    }


def test_suffix_targets_are_fixed_and_masked_loss_matches_manual_ce():
    set_seed(0)
    model = GPT(vocab_size=32, block_size=32, d_model=16, n_layers=1,
                n_heads=2, ffn_mult=2, dropout=0.0, attn_dropout=0.0,
                variant="standard")
    model.eval()
    val = np.arange(200, dtype=np.uint16) % 32
    starts = [0, 32, 64]
    lengths = [4, 8, 16]
    q = 2
    recs = collect_length_ppl_records(
        model, val, starts, lengths, q, "suffix", "cpu",
        amp=False, batch_windows=2)
    assert len(recs) == len(starts) * len(lengths)
    by_window = {}
    for rec in recs:
        by_window.setdefault(rec["window"], []).append(rec)
    for rows in by_window.values():
        first = rows[0]["target_ids"]
        assert all(r["target_ids"] == first for r in rows)
    # 手算一个窗口最后 q 个目标的 CE，必须与 masked loss 完全一致。
    rec = next(r for r in recs
               if r["window"] == 1 and r["length"] == 8)
    end = rec["window_start"] + max(lengths)
    ids = torch.from_numpy(
        val[end - rec["length"]:end].astype(np.int64))[None]
    targets = torch.from_numpy(
        val[end - q + 1:end + 1].astype(np.int64))[None]
    with torch.inference_mode():
        logits = model(ids)["logits"][0, -q:]
        manual = float(F.cross_entropy(logits, targets[0]).item())
    assert abs(rec["nll"] - manual) < 1e-6, (rec["nll"], manual)
    assert rec["target_ids"] == [int(x) for x in targets[0].tolist()]
    print("  [ok] suffix 固定目标、masked loss 与手算 CE 一致")


def test_prefix_targets_change_with_length():
    set_seed(0)
    model = GPT(vocab_size=32, block_size=32, d_model=16, n_layers=1,
                n_heads=2, ffn_mult=2, dropout=0.0, attn_dropout=0.0,
                variant="standard")
    model.eval()
    val = np.arange(200, dtype=np.uint16) % 32
    recs = collect_length_ppl_records(
        model, val, [0], [4, 8, 16], 2, "prefix", "cpu",
        amp=False, batch_windows=1)
    targets = [r["target_ids"] for r in recs]
    assert len({tuple(t) for t in targets}) > 1
    print("  [ok] prefix 目标随长度改变（只作敏感性分析）")


def test_analyze_recovers_known_slope_delta_and_context_gain():
    lengths = (4, 8, 16)
    recs = _records(0.1, lengths=lengths)
    out = analyze_length_ppl_records(
        recs, lengths, "suffix", n_bootstrap=0)
    assert out["available"] and out["n_windows"] == 3
    slope = out["nll_slope_per_log2"]
    assert abs(slope["estimate"] - 0.1) < 1e-12, slope
    assert slope["class"] == "no_ci"
    assert abs(out["context_gain_nats"]["estimate"] + 0.2) < 1e-12
    d4 = out["delta_vs_reference"]["4"]
    d8 = out["delta_vs_reference"]["8"]
    assert abs(d4["mean_delta_nll"] + 0.2) < 1e-12, d4
    assert abs(d8["mean_delta_nll"] + 0.1) < 1e-12, d8
    assert d4["direction"] == "shorter_context_lower_nll"
    assert out["target_pairing"]["fraction_consistent"] == 1.0
    assert out["per_length"]["16"]["ppl"] > out["per_length"]["4"]["ppl"]
    print("  [ok] 已知斜率/ΔNLL/context gain 精确恢复")


def test_analyze_marks_flat_slope_within_tolerance():
    recs = _records(0.0, lengths=(4, 8, 16))
    out = analyze_length_ppl_records(
        recs, (4, 8, 16), "suffix", tolerance=0.01, n_bootstrap=20)
    assert out["nll_slope_per_log2"]["class"] == "flat"
    assert out["nll_slope_per_log2"]["ci95"] is not None
    print("  [ok] 零斜率在容差内判为 flat")


def test_compare_recovers_length_interaction():
    lengths = (4, 8, 16)
    ref = _records(0.1, lengths=lengths)
    # candidate 在每个长度上都更差 0.2，且差距随长度缩小 0.1/log2。
    cand = []
    for rec in ref:
        row = dict(rec)
        row["nll"] = rec["nll"] + 0.2 - 0.1 * math.log2(rec["length"])
        cand.append(row)
    out = compare_length_ppl_records(
        cand, ref, lengths, "suffix", n_bootstrap=0)
    assert out["available"] and out["n_windows"] == 3
    slope = out["delta_slope_per_log2"]
    assert abs(slope["estimate"] + 0.1) < 1e-12, slope
    assert abs(out["per_length"]["4"]["mean_delta_nll"] - 0.0) < 1e-12
    assert abs(out["per_length"]["8"]["mean_delta_nll"] + 0.1) < 1e-12
    assert abs(out["per_length"]["16"]["mean_delta_nll"] + 0.2) < 1e-12
    assert out["target_pairing"]["fraction_consistent"] == 1.0
    print("  [ok] 候选-参考 ΔNLL 的长度交互斜率精确恢复")


def test_compare_payloads_reconstructs_per_window_curves():
    cand = _payload("lega", 0.0)
    ref = _payload("standard", 0.1)
    out = compare_length_ppl_payloads(
        cand, ref, "suffix", n_bootstrap=0)
    assert out["n_windows"] == 3
    assert out["candidate"] == "lega" and out["reference"] == "standard"
    assert abs(out["delta_slope_per_log2"]["estimate"] + 0.1) < 1e-12
    assert out["target_pairing"]["fraction_consistent"] == 1.0
    print("  [ok] payload 逐窗口曲线可重建并做交互比较")


def test_summary_interaction_rows_pair_seeds_and_report_missing():
    ppls = {
        "lega": {seed: _payload("lega", 0.0) for seed in (1, 2, 3)},
        "standard": {seed: _payload("standard", 0.1) for seed in (1, 2, 3)},
    }
    rows = _length_ppl_interaction_rows(ppls, [("lega", "standard")])
    assert len(rows) == 1
    row = rows[0]
    assert row["n_seeds"] == 3 and row["slope_median"] < 0
    assert abs(row["per_length_mean_delta"]["4"] + 0.2) < 1e-12
    assert row["target_pairing_fraction"] == 1.0
    assert _length_ppl_interaction_rows(ppls, [("lega", "missing")]) == []
    print("  [ok] 汇总交互行按种子配对，缺变体不伪造")


def test_render_length_ppl_reports_curves_and_interaction():
    payloads = {
        "lega": {7: _payload("lega", 0.0)},
        "standard": {7: _payload("standard", 0.1)},
    }
    interactions = [{
        "candidate": "lega", "reference": "standard", "n_seeds": 3,
        "lengths": [4, 8, 16],
        "slope_median": -0.1, "slope_p": 0.01,
        "per_length_mean_delta": {"4": 0.0, "8": -0.1, "16": -0.2},
    }]
    md = "\n".join(_render_length_ppl(
        payloads, requested=True, interactions=interactions))
    assert "自然文本 PPL 长度剂量-反应" in md
    assert "suffix" in md and "prefix" in md
    assert "候选优势随长度增加" in md
    assert "4–16" in md
    missing = "\n".join(_render_length_ppl({}, requested=True))
    assert "没有找到可用" in missing
    weak = dict(interactions[0], n_seeds=1)
    md_weak = "\n".join(_render_length_ppl(
        payloads, requested=True, interactions=[weak]))
    assert "种子不足/未定" in md_weak
    print("  [ok] 汇总页渲染长度曲线与跨变体交互（种子不足显式未定）")


def test_main_records_sys_argv(monkeypatch):
    import lega.lengthppl as lp

    tmp = tempfile.mkdtemp(prefix="lengthppl_argv_")
    try:
        def fake_run(*args, **kwargs):
            return {"variant": "lega", "lengths": [8, 16],
                    "window_starts": [],
                    "results": {
                        "suffix": {
                            "per_length": {"8": {"ppl": 2.0},
                                           "16": {"ppl": 1.9}},
                            "nll_slope_per_log2": {
                                "estimate": -0.1, "class": "decreasing"},
                        }}}

        monkeypatch.setattr(lp, "run_length_ppl", fake_run)
        out = os.path.join(tmp, "length_ppl.json")
        argv = ["-m lega.lengthppl", "--ckpt", os.path.join(tmp, "model.pt"),
                "--data_dir", tmp, "--lengths", "8,16", "--out", out]
        monkeypatch.setattr(sys, "argv", argv)
        assert lp.main(None) == 0
        with open(out, encoding="utf-8") as f:
            payload = json.load(f)
        assert payload["argv"] == argv
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
