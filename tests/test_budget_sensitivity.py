"""训练预算敏感性：对比/消融终点结论是否覆盖整个训练过程。"""
import json
import math
import os
import shutil
import tempfile

from lega.budget_sensitivity import (analyze_budget_sensitivity,
                                     curve_from_records, curves_from_runs,
                                     load_train_curve)
from tests.pipeline_helpers import _write_json
from run_all import build_summary


def _strict_json(obj):
    return json.dumps(obj, ensure_ascii=False, allow_nan=False)


def _curve(values):
    return {int(step): float(loss) for step, loss in values}


def _recs(values):
    return [{"step": step, "val_loss": loss} for step, loss in values]


def test_curve_from_records_filters_bad_and_collapses_duplicate_steps():
    recs = [
        {"step": 10, "val_loss": 2.5},
        {"step": "20", "val_loss": 2.4},          # 字符串 step：拒绝
        {"step": True, "val_loss": 2.3},          # bool：拒绝
        {"step": 30, "val_loss": float("nan")},   # 非有限：拒绝
        {"step": 40, "val_loss": "2.2"},          # JSON 数值字符串：接受
        {"step": 10, "val_loss": 2.5},            # 同值重复：合并
        {"step": 50, "val_loss": 2.1},
        {"step": 50, "val_loss": 2.0, "final": True},  # 终记录覆盖
        {"step": 0, "val_loss": 9.0},             # step 非正：拒绝
        {"step": 60, "val_loss": float("inf")},   # 非有限：拒绝
        "bad-record",
    ]
    assert curve_from_records(recs) == {10: 2.5, 40: 2.2, 50: 2.0}
    print("  [ok] 预算曲线：坏行/非规范 step 不进入轨迹，终记录幂等覆盖")


def test_stable_trajectory_reports_no_budget_sensitivity():
    curves = {
        "cand": {s: _curve([(100, 2.0), (200, 1.9), (300, 1.8)])
                 for s in (1, 2, 3)},
        "ref": {s: _curve([(100, 2.1), (200, 2.0), (300, 1.9)])
                for s in (1, 2, 3)},
    }
    out = analyze_budget_sensitivity(curves, [("cand", "ref")])
    p = out["pairs"][0]
    assert p["available"] and p["inferential"]
    assert p["n_seeds"] == 3 and p["n_steps"] == 3
    assert p["final_mean_diff"] < 0
    assert p["sign_agreement_with_final"] == 1.0
    assert p["n_sign_flips"] == 0
    assert p["trajectory_pattern"] == "stable"
    assert p["budget_sensitive"] is False
    assert p["stable_from_step"] == 100
    _strict_json(out)
    print("  [ok] 预算轨迹：方向一致 -> stable，且结果可严格 JSON 序列化")


def test_reversal_and_transient_patterns_are_distinguished():
    base = [2.0, 1.9, 1.8, 1.7]
    reversal_curves = {
        "cand": {s: _curve(zip((100, 200, 300, 400), base)) for s in (1, 2, 3)},
        "ref": {s: _curve(zip((100, 200, 300, 400),
                              [2.05, 2.0, 1.75, 1.6])) for s in (1, 2, 3)},
    }
    p = analyze_budget_sensitivity(
        reversal_curves, [("cand", "ref")])["pairs"][0]
    assert p["initial_sign"] == -1 and p["final_sign"] == 1
    assert p["sign_reversal"] is True
    assert p["budget_sensitive"] is True
    assert p["trajectory_pattern"] == "budget_sensitive"
    assert p["stable_from_step"] == 300
    assert sorted(p["significant_steps"]) == [100, 200, 300, 400]

    # 首末同向，但中段短暂翻向：transient 而非 reversal。
    transient = {
        "cand": {s: _curve([(100, 2.0), (200, 2.2), (300, 1.8)])
                 for s in (1, 2, 3)},
        "ref": {s: _curve([(100, 2.1), (200, 2.1), (300, 1.9)])
                for s in (1, 2, 3)},
    }
    p2 = analyze_budget_sensitivity(
        transient, [("cand", "ref")])["pairs"][0]
    assert p2["trajectory_pattern"] == "transient"
    assert p2["sign_reversal"] is False
    assert p2["n_sign_flips"] == 2
    print("  [ok] 预算轨迹：首末翻转/中段翻转分开报告，不混成同一模式")


def test_exact_step_intersection_does_not_interpolate_or_change_seed_set():
    curves = {
        "cand": {
            1: _curve([(100, 2.0), (200, 1.9), (300, 1.8), (400, 1.7)]),
            2: _curve([(100, 2.0), (200, 1.9), (300, 1.8)]),
            3: _curve([(100, 2.0), (200, 1.9), (300, 1.8)]),
        },
        "ref": {
            1: _curve([(100, 2.1), (200, 2.0), (250, 1.95), (300, 1.9)]),
            2: _curve([(100, 2.1), (200, 2.0), (300, 1.9)]),
            3: _curve([(100, 2.1), (200, 2.0), (300, 1.9)]),
        },
    }
    p = analyze_budget_sensitivity(curves, [("cand", "ref")])["pairs"][0]
    assert p["steps"] == [100, 200, 300]
    assert p["n_seeds"] == 3
    assert p["final_step"] == 300
    assert _strict_json(p)
    print("  [ok] 预算轨迹：精确 step 交集，缺失/额外评测步不插值")


def test_missing_curves_are_explicit_not_silently_empty():
    out = analyze_budget_sensitivity(
        {"cand": {1: _curve([(100, 2.0)])}},
        [("cand", "ref"), ("ref", "cand"), ("cand", "cand"),
         ("cand", "ref")])
    assert out["n_pairs"] == 2 and out["n_available"] == 0
    assert all(p["reason"] for p in out["pairs"])
    assert any(p["reference"] == "ref" for p in out["pairs"])
    assert all(p["available"] is False for p in out["pairs"])
    # 单点曲线可报告但 inferential=False，不生成显著性措辞。
    one = {"cand": {1: _curve([(100, 2.0)]), 2: _curve([(100, 2.0)])},
           "ref": {1: _curve([(100, 2.1)]), 2: _curve([(100, 2.1)])}}
    p = analyze_budget_sensitivity(one, [("cand", "ref")])["pairs"][0]
    assert p["available"] is True and p["inferential"] is False
    assert p["trajectory_pattern"] == "insufficient_steps"
    assert p["budget_sensitive"] is False
    for bad in (0.0, 1.0, float("nan"), True):
        try:
            analyze_budget_sensitivity({}, [], alpha=bad)
            raised = False
        except ValueError:
            raised = True
        assert raised, bad
    print("  [ok] 预算轨迹：缺曲线/单点/非法 alpha 显式降级，不伪造显著性")


def test_load_train_curve_tolerates_malformed_lines_and_numpy_bools():
    import numpy as np
    tmp = tempfile.mkdtemp()
    try:
        path = os.path.join(tmp, "train_log.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            f.write('{"step": 10, "val_loss": 2.5}\n')
            f.write('not json\n')
            f.write('{"step": 20, "val_loss": NaN}\n')
            f.write('{"step": 30, "val_loss": 2.3}\n')
        assert load_train_curve(path) == {10: 2.5, 30: 2.3}
        assert load_train_curve(os.path.join(tmp, "missing.jsonl")) == {}
        assert curve_from_records([
            {"step": 1, "val_loss": np.True_},
            {"step": 2, "val_loss": 2.0}]) == {2: 2.0}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] 预算曲线：损坏日志行安全跳过，np.bool_ 不伪装成数值")


def test_curves_from_runs_uses_checkpoint_directory_train_log():
    tmp = tempfile.mkdtemp()
    try:
        run_dir = os.path.join(tmp, "out", "lega", "seed1")
        os.makedirs(run_dir)
        with open(os.path.join(run_dir, "train_log.jsonl"), "w",
                  encoding="utf-8") as f:
            f.write('{"step": 10, "val_loss": 2.5}\n')
            f.write('{"step": 20, "val_loss": 2.0}\n')
        runs = {"lega": {1: {"eval": {
            "ckpt": os.path.join(run_dir, "model_best.pt")}}}}
        got = curves_from_runs(runs)
        assert got == {"lega": {1: {10: 2.5, 20: 2.0}}}
        assert curves_from_runs({"lega": {1: {"eval": {}}}}) == {}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] 预算曲线：由 eval.ckpt 同目录发现 train_log.jsonl")


def test_build_summary_persists_budget_sensitivity():
    tmp = tempfile.mkdtemp()
    try:
        seeds = [1, 2, 3]
        _write_json(os.path.join(tmp, "repro.json"),
                    {"variants": ["standard", "lega"], "seeds": seeds,
                     "smoke": False})
        for seed in seeds:
            for variant in ("standard", "lega"):
                run_dir = os.path.join(tmp, "out", variant, f"seed{seed}")
                os.makedirs(run_dir, exist_ok=True)
                ref = [2.10, 2.00, 1.90]
                cand = [2.00, 1.90, 2.05]
                values = ref if variant == "standard" else cand
                with open(os.path.join(run_dir, "train_log.jsonl"), "w",
                          encoding="utf-8") as f:
                    for step, loss in zip((100, 200, 300), values):
                        f.write(json.dumps({"step": step, "val_loss": loss})
                                + "\n")
                _write_json(os.path.join(tmp, f"eval_{variant}_seed{seed}.json"),
                            {"variant": variant, "seed": seed,
                             "val_ppl": math.exp(values[-1]),
                             "val_loss": values[-1], "n_params": 10,
                             "split": "test",
                             "ckpt": os.path.join(run_dir, "model_best.pt")})
        out = build_summary(tmp)
        md = out["markdown"]
        assert "训练预算敏感性" in md
        assert "预算敏感" in md
        payload = json.load(open(os.path.join(tmp, "significance.json"),
                                 encoding="utf-8"))
        budget = payload["budget_sensitivity"]
        assert budget["n_available"] >= 1
        pair = next(p for p in budget["pairs"]
                    if p["candidate"] == "lega"
                    and p["reference"] == "standard")
        assert pair["available"] is True
        assert pair["budget_sensitive"] is True
        assert pair["initial_mean_diff"] < 0 < pair["final_mean_diff"]
        _strict_json(payload)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] 汇总：预算轨迹进入 summary.md/significance.json 且可机读")


def test_build_summary_marks_missing_train_logs():
    tmp = tempfile.mkdtemp()
    try:
        _write_json(os.path.join(tmp, "repro.json"),
                    {"variants": ["standard", "lega"], "seeds": [1, 2, 3],
                     "smoke": False})
        for seed in (1, 2, 3):
            for variant, ppl in (("standard", 10.0), ("lega", 9.0)):
                _write_json(os.path.join(tmp, f"eval_{variant}_seed{seed}.json"),
                            {"variant": variant, "seed": seed,
                             "val_ppl": ppl, "val_loss": math.log(ppl),
                             "n_params": 10, "split": "test",
                             "ckpt": os.path.join(
                                 tmp, "missing", variant, f"seed{seed}",
                                 "model_best.pt")})
        md = build_summary(tmp)["markdown"]
        assert "无可用训练曲线" in md
        payload = json.load(open(os.path.join(tmp, "significance.json"),
                                 encoding="utf-8"))
        assert payload["budget_sensitivity"]["n_available"] == 0
        assert "不能据此宣称" in md
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] 汇总：缺 train_log 时不伪造预算轨迹，显式警告")
