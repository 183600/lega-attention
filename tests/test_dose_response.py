"""门控系数剂量-反应测试：只改目标参数、可恢复、曲线诊断口径正确。"""
import json
import math
import os
import tempfile

import numpy as np

import torch

from lega.dose_response import (apply_dose,
                                component_applicability,
                                dose_curve_diagnostics, parse_components,
                                parse_factors, run_dose_response)
from lega.ablate import restore_gate_params, snapshot_gate_params
from lega.model import GPT, save_checkpoint
from run_all import _dose_response_tests, build_summary


def _tiny(variant="lega"):
    return GPT(vocab_size=64, block_size=32, d_model=16, n_layers=2,
               n_heads=2, variant=variant, use_sdpa_standard=False)


def _points(loss_fn, factors=(0.0, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)):
    return [{"factor": f, "val_loss": loss_fn(f), "val_ppl": math.exp(loss_fn(f)),
             "window_losses": [loss_fn(f)] * 4}
            for f in factors]


def test_apply_dose_scales_only_target_and_restores_exactly():
    model = _tiny("lega")
    snap = snapshot_gate_params(model)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    rec = apply_dose(model, "A", 0.5, snap)
    assert rec["applied"] and rec["changed_tensors"] == 2
    for name, p in model.named_parameters():
        if name.endswith("gate_w"):
            assert torch.allclose(p, before[name] * 0.5)
        else:
            assert torch.equal(p, before[name])
    restore_gate_params(model, snap)
    for name, p in model.named_parameters():
        assert torch.equal(p, before[name])
    print("  [ok] 剂量干预只缩放目标系数，恢复后逐位一致")


def test_apply_dose_rejects_bad_factor_and_missing_component():
    model = _tiny("gate_only")
    snap = snapshot_gate_params(model)
    for poor in (True, False, "1_0", float("nan"), float("inf")):
        try:
            apply_dose(model, "A", poor, snap)
            raise AssertionError(f"{poor!r} 应显式拒绝")
        except ValueError:
            pass
    try:
        apply_dose(model, "D", 0.5, snap)
        raise AssertionError("未知组件应显式拒绝")
    except ValueError:
        pass
    applicable, reason, _caps = component_applicability(model, "B")
    assert not applicable and "conc" in reason
    rec = apply_dose(model, "B", 0.5, snap)
    assert rec["applied"] is False and rec["changed_tensors"] == 0
    print("  [ok] 非法剂量/缺失组件显式拒绝，不伪造清零")


def test_dose_curve_quadratic_diagnostics_recover_vertex_and_slope():
    diag = dose_curve_diagnostics(_points(lambda f: (f - 1.2) ** 2 + 2.0))
    assert diag["available"]
    assert abs(diag["quadratic_vertex"] - 1.2) < 1e-9
    assert diag["quadratic_vertex_in_grid"]
    assert abs(diag["curvature"] - 2.0) < 1e-9
    assert abs(diag["quadratic_r2"] - 1.0) < 1e-9
    assert abs(diag["local_slope"] + 0.4) < 1e-9
    assert diag["local_slope_side"] == "two_sided"
    assert abs(diag["best_factor"] - 1.25) < 1e-12
    assert diag["baseline_is_grid_minimum"] is False
    assert diag["monotonic"] is False
    assert diag["n_direction_changes"] == 1
    assert abs(diag["improvement_vs_baseline"] - 0.0375) < 1e-12
    assert diag["zero_delta"] is not None
    print("  [ok] 二次曲线顶点/局部斜率/网格最优/方向变化正确")


def test_dose_curve_linear_has_no_vertex_and_missing_baseline_is_explicit():
    diag = dose_curve_diagnostics(_points(lambda f: 3.0 + 0.2 * (f - 1.0)))
    assert diag["quadratic_vertex"] is None
    assert diag["quadratic_vertex_in_grid"] is False
    assert abs(diag["local_slope"] - 0.2) < 1e-12
    assert abs(diag["best_factor"]) < 1e-12
    assert diag["baseline_is_grid_minimum"] is False

    no_base = [p for p in _points(lambda f: 2.0 + f)
               if abs(p["factor"] - 1.0) > 1e-12]
    diag2 = dose_curve_diagnostics(no_base, baseline_loss=3.0)
    assert diag2["baseline_factor"] == 1.0
    assert abs(diag2["baseline_loss"] - 3.0) < 1e-12
    assert diag2["local_slope_side"] == "two_sided"

    upper_only = [p for p in no_base if p["factor"] > 1.0]
    diag3 = dose_curve_diagnostics(upper_only, baseline_loss=3.0)
    assert diag3["local_slope_side"] == "one_sided_upper"
    assert abs(diag3["local_slope"] - 1.0) < 1e-12
    assert dose_curve_diagnostics([])["available"] is False
    print("  [ok] 线性曲线不伪造顶点；缺基线时显式标注并允许外部基线")


def _dose_payload(seed, deltas, *, slope=-0.3, best_factor=1.0,
                  baseline_min=True):
    factors = sorted(deltas)
    points = []
    for f in factors:
        points.append({
            "factor": f,
            "applied": True,
            "no_op": bool(f == 0.0 and abs(deltas[f]) < 1e-15),
            "delta_loss_mean": deltas[f],
            "n_windows": 4,
            "window_indices": [0, 1, 2, 3],
            "val_loss": 2.0 + deltas[f],
        })
    return {
        "schema": 1,
        "seed": seed,
        "window_indices": [0, 1, 2, 3],
        "components": {
            "A": {
                "applicable": True,
                "points": points,
                "diagnostics": {
                    "local_slope": slope,
                    "best_factor": best_factor,
                    "baseline_is_grid_minimum": baseline_min,
                    "quadratic_vertex": 1.1,
                    "quadratic_vertex_in_grid": True,
                    "zero_delta": deltas.get(0.0),
                    "sign_flip_delta": None,
                },
            }
        },
    }


def test_cross_seed_dose_tests_use_paired_seed_deltas_and_component_families():
    deltas = [-0.10, -0.12, -0.08, -0.11, -0.09]
    doses = {"lega": {
        seed: _dose_payload(
            seed, {0.5: d, 1.0: 0.0, 1.25: -d},
            slope=-0.3, best_factor=1.0, baseline_min=True)
        for seed, d in enumerate(deltas)}}
    out = _dose_response_tests(doses)
    assert out["available"] and not out["excluded_variants"]
    factor_tests = [t for t in out["tests"]
                    if t["dose_family"] == "factor"]
    slope_tests = [t for t in out["tests"]
                   if t["dose_family"] == "local_slope"]
    assert len(factor_tests) == 2 and len(slope_tests) == 1
    assert {t["metric"] for t in factor_tests} == {"剂量反应ΔNLL[A]"}
    assert slope_tests[0]["metric"] == "剂量反应局部斜率[A]"
    assert slope_tests[0]["n"] == 5
    half = next(t for t in factor_tests if t["dose_factor"] == 0.5)
    assert abs(half["mean_diff"] + 0.1) < 1e-12
    assert half["p_adj"] is not None
    assert "显著更好" in half["verdict"]
    assert "局部斜率显著为负" in slope_tests[0]["verdict"]
    comp = next(c for c in out["components"] if c["component"] == "A")
    assert comp["available"] and comp["n_seeds"] == 5
    assert comp["local_slope_p_adj"] is not None
    assert comp["baseline_is_grid_minimum_rate"] == 1.0
    print("  [ok] 剂量曲线按种子配对、A/B/C 分族且判词不冒充确认性结论")


def test_cross_seed_dose_intersects_factors_and_flags_all_noop():
    doses = {}
    for seed in range(5):
        deltas = {0.0: 0.0, 0.5: -0.05, 1.0: 0.0}
        if seed != 0:
            deltas[1.25] = 0.2
        doses[seed] = _dose_payload(
            seed, deltas, slope=-0.1, best_factor=2.0, baseline_min=False)
    out = _dose_response_tests({"lega": doses})
    assert out["available"]
    factor_factors = {t["dose_factor"] for t in out["tests"]
                      if t["dose_family"] == "factor"}
    assert factor_factors == {0.0, 0.5}
    zero = next(t for t in out["tests"]
                if t["dose_family"] == "factor"
                and t["dose_factor"] == 0.0)
    assert zero["all_no_op"] and "no-op" in zero["verdict"]
    comp = next(c for c in out["components"] if c["component"] == "A")
    assert comp["common_factors"] == [0.0, 0.5, 1.0]
    print("  [ok] 剂量交集避免不同 λ 换种子；全 no-op 不伪装零效应")


def test_cross_seed_dose_with_only_baseline_is_explicitly_unavailable():
    payload = _dose_payload(0, {1.0: 0.0})
    out = _dose_response_tests({"lega": {0: payload}})
    assert out["available"] is False
    comp = out["components"][0]
    assert comp["available"] is False and "λ=1" in comp["reason"]
    print("  [ok] 只有基线剂量时明确不可用，不生成空检验")


def test_run_dose_response_end_to_end_tiny_checkpoint():
    model = _tiny("lega")
    with torch.no_grad():
        for block in model.blocks:
            block.attn.gate_a.fill_(0.2)
            block.attn.gate_b.fill_(-0.1)
            block.attn.gate_w.fill_(0.01)
    with tempfile.TemporaryDirectory() as d:
        ckpt = os.path.join(d, "model_best.pt")
        save_checkpoint(ckpt, model, extra={"seed": 1, "variant": "lega"})
        with open(os.path.join(d, "config_used.json"), "w",
                  encoding="utf-8") as f:
            json.dump({"train_ctx": 16, "eval_iters": 2, "amp": False,
                       "amp_dtype": None}, f)
        tokens = (np.arange(1024, dtype=np.uint16) % 64).astype(np.uint16)
        tokens.tofile(os.path.join(d, "val.bin"))
        for bad in (True, 1.5, -1, 2**32):
            try:
                run_dose_response(ckpt, d, eval_iters=2, ctx=16,
                                  device="cpu", seed=bad,
                                  factors=[0.0, 1.0])
                raise AssertionError(f"seed={bad!r} 应显式报错")
            except ValueError:
                pass
        payload = run_dose_response(
            ckpt, d, eval_iters=2, ctx=16, device="cpu", seed=1,
            factors=[0.0, 0.5, 1.0, 1.5], components=["A", "B", "C"])
        assert payload["variant"] == "lega"
        assert payload["split"] == "val"
        assert payload["eval_ctx"] == 16 and payload["eval_iters"] == 2
        assert payload["baseline"]["val_loss"] is not None
        for name in ("A", "B", "C"):
            comp = payload["components"][name]
            assert comp["applicable"] is True
            assert [p["factor"] for p in comp["points"]] == [0.0, 0.5, 1.0, 1.5]
            assert comp["points"][2]["no_op"] is True
            assert comp["points"][0]["no_op"] is False
            assert comp["diagnostics"]["available"] is True
            assert all(p.get("window_indices") == payload["window_indices"]
                       for p in comp["points"])
    print("  [ok] run_dose_response 端到端：tiny checkpoint + 合成 val.bin，"
          "λ=1 复用基线、逐窗口配对且不适用/非法种子显式处理")


def test_build_summary_wires_dose_response_into_markdown_and_json():
    with tempfile.TemporaryDirectory() as d:
        for seed in range(1, 6):
            payload = _dose_payload(
                seed, {0.5: -0.1, 1.0: 0.0, 1.25: 0.1},
                slope=-0.4, best_factor=1.0, baseline_min=True)
            payload.update({
                "variant": "lega",
                "split": "val",
                "eval_ctx": 16,
                "eval_iters": 2,
                "factors": [0.5, 1.0, 1.25],
                "baseline": {"val_loss": 2.0, "val_ppl": math.exp(2.0)},
            })
            with open(os.path.join(
                    d, f"dose_response_lega_seed{seed}.json"),
                    "w", encoding="utf-8") as f:
                json.dump(payload, f)
        out = build_summary(d)
        assert out["dose_response"]["available"]
        assert "门控系数剂量-反应" in out["markdown"]
        with open(os.path.join(d, "significance.json"), "r",
                  encoding="utf-8") as f:
            sig = json.load(f)
        assert sig["dose_response"]["available"]
        assert "winner's curse" in sig["dose_response_note"]
    print("  [ok] build_summary：剂量曲线进入 Markdown 与 significance.json")


def test_build_summary_only_reuses_doses_when_scope_is_true_bool():
    """repro.dose_response.executed 必须是真正的 true 才复用旧剂量曲线。

    否则上一轮 `dose_response_*.json` 会被当成本轮证据，让“本轮未开启
    剂量分析”的汇总仍带着剂量结论；同时 variants 白名单必须生效。
    """
    with tempfile.TemporaryDirectory() as d:
        for seed in range(1, 4):
            payload = _dose_payload(
                seed, {0.5: -0.1, 1.0: 0.0, 1.25: 0.1},
                slope=-0.4, best_factor=1.0, baseline_min=True)
            payload.update({
                "variant": "lega", "split": "val",
                "eval_ctx": 16, "eval_iters": 2,
                "factors": [0.5, 1.0, 1.25],
                "baseline": {"val_loss": 2.0, "val_ppl": math.exp(2.0)},
            })
            with open(os.path.join(
                    d, f"dose_response_lega_seed{seed}.json"),
                    "w", encoding="utf-8") as f:
                json.dump(payload, f)
        repro = os.path.join(d, "repro.json")
        with open(repro, "w", encoding="utf-8") as f:
            json.dump({"variants": ["lega"], "seeds": [1, 2, 3],
                       "smoke": "false",
                       "dose_response": {"executed": "false",
                                         "variants": ["lega"]}}, f)
        out = build_summary(d)
        assert out["dose_response"]["available"] is False
        with open(os.path.join(d, "significance.json"), "r",
                  encoding="utf-8") as f:
            sig = json.load(f)
        assert sig["dose_response"]["available"] is False
        assert "门控系数剂量-反应" not in out["markdown"]

        with open(repro, "w", encoding="utf-8") as f:
            json.dump({"variants": ["lega"], "seeds": [1, 2, 3],
                       "smoke": False,
                       "dose_response": {"executed": True,
                                         "variants": ["qgate"]}}, f)
        out = build_summary(d)
        assert out["dose_response"]["available"] is False

        with open(repro, "w", encoding="utf-8") as f:
            json.dump({"variants": ["lega"], "seeds": [1, 2, 3],
                       "smoke": False,
                       "dose_response": {"executed": True,
                                         "variants": ["lega"]}}, f)
        out = build_summary(d)
        assert out["dose_response"]["available"] is True
        assert "门控系数剂量-反应" in out["markdown"]
    print("  [ok] 剂量曲线只在 executed 为 true 且变体在范围内时复用")


def test_parse_helpers_are_strict_and_deduplicate():
    assert parse_factors("2,0,1,0.5,0.5") == [0.0, 0.5, 1.0, 2.0]
    assert parse_components("a,C,a") == ["A", "C"]
    for poor in ("", ",", "nan", "1_0", True):
        try:
            parse_factors(poor)
            raise AssertionError(f"因子 {poor!r} 应显式拒绝")
        except ValueError:
            pass
    for poor in ("", "D", "A,D"):
        try:
            parse_components(poor)
            raise AssertionError(f"组件 {poor!r} 应显式拒绝")
        except ValueError:
            pass
    print("  [ok] 因子/组件解析去重且拒绝非法输入")


if __name__ == "__main__":
    test_apply_dose_scales_only_target_and_restores_exactly()
    test_apply_dose_rejects_bad_factor_and_missing_component()
    test_dose_curve_quadratic_diagnostics_recover_vertex_and_slope()
    test_dose_curve_linear_has_no_vertex_and_missing_baseline_is_explicit()
    test_parse_helpers_are_strict_and_deduplicate()
    test_cross_seed_dose_tests_use_paired_seed_deltas_and_component_families()
    test_cross_seed_dose_intersects_factors_and_flags_all_noop()
    test_cross_seed_dose_with_only_baseline_is_explicitly_unavailable()
    test_run_dose_response_end_to_end_tiny_checkpoint()
    test_build_summary_wires_dose_response_into_markdown_and_json()
    test_build_summary_only_reuses_doses_when_scope_is_true_bool()
