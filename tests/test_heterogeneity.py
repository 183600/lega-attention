"""分层效应异质性（随机效应 meta 分析）的单元/集成测试。

覆盖：卡方尾概率、同质/异质合并、预测区间、坏层剔除、零层内方差、
种子矩阵 bootstrap 确定性，以及 RULER 任务/长度分层在流水线中的适配、
H4 挂载与汇总渲染。
"""
import math

from lega.heterogeneity import (attach_to_hypotheses, chi2_sf,
                                meta_analysis,
                                meta_analysis_from_seed_matrix)
from lega.pipeline import (_render_ruler_heterogeneity_section,
                           _ruler_heterogeneity_analysis)


def _close(a, b, tol=1e-9):
    return abs(float(a) - float(b)) <= tol


def test_chi2_sf_known_value_and_validation():
    assert _close(chi2_sf(3.841458820694124, 1), 0.05, 1e-9)
    assert _close(chi2_sf(0.0, 3), 1.0)
    assert chi2_sf(30.0, 2) < 1e-6
    for x, df in ((-1.0, 2), (1.0, 0), (math.inf, 2)):
        try:
            chi2_sf(x, df)
            raise AssertionError(f"非法 chi2_sf 参数应报错: {x}, {df}")
        except ValueError:
            pass
    print("  [ok] 卡方尾概率与非法参数")


def test_homogeneous_meta_analysis():
    res = meta_analysis([0.1, 0.1, 0.1], [0.02, 0.02, 0.02],
                        labels=["a", "b", "c"])
    assert res["status"] == "no_detected_heterogeneity", res
    assert _close(res["pooled_fixed"], 0.1)
    assert _close(res["pooled_random"], 0.1)
    assert _close(res["q"], 0.0)
    assert _close(res["i2"], 0.0) and _close(res["tau2"], 0.0)
    assert res["prediction_interval"] is not None
    assert _close(res["loo_pooled_random_range"][0], 0.1)
    assert res["loo_sign_stable"] is True
    print("  [ok] 同质层 -> I²=0、合并=共同效应、留一稳定")


def test_heterogeneous_meta_analysis_and_prediction_interval():
    res = meta_analysis([0.30, -0.28, 0.25], [0.02, 0.02, 0.03],
                        labels=["a", "b", "c"])
    assert res["status"] == "heterogeneous", res
    assert res["q_p"] < 0.001 and res["i2"] > 0.8, res
    assert res["tau2"] > 0 and res["tau"] > 0
    lo, hi = res["prediction_interval"]
    assert lo < res["pooled_random"] < hi
    assert res["loo_sign_stable"] is False
    assert res["loo_max_abs_shift_label"] in ("a", "b", "c")
    print("  [ok] 异质层 -> I²/tau² 上升、预测区间变宽、留一方向不稳定")


def test_invalid_layers_are_excluded_not_zero_filled():
    res = meta_analysis([0.1, None, "x", 0.2], [0.02, 0.02, 0.03, 0.0],
                        labels=["a", "b", "c", "d"])
    assert res["status"] == "insufficient", res
    assert res["k"] == 1 and len(res["excluded"]) == 3, res
    reasons = " ".join(row["reason"] for row in res["excluded"])
    assert "effect" in reasons and "se" in reasons
    try:
        meta_analysis([0.1, 0.2], [0.02])
        raise AssertionError("长度不一致应报错")
    except ValueError:
        pass
    try:
        meta_analysis([0.1, 0.2], [0.02, 0.03], labels=["a", "a"])
        raise AssertionError("重复层名应报错")
    except ValueError:
        pass
    print("  [ok] 坏层显式剔除并记账，不按 0 补齐")


def test_zero_within_variance_is_not_fabricated():
    res = meta_analysis_from_seed_matrix(
        [[0.1, 0.2], [0.1, 0.2], [0.1, 0.2]],
        labels=["a", "b"], n_bootstrap=50)
    assert res["status"] == "invalid", res
    assert res["pooled_random"] is None
    assert res["bootstrap"]["available"] is False
    print("  [ok] 层内方差全 0 时不伪造随机效应合并")


def test_seed_matrix_bootstrap_is_deterministic():
    rows = [[0.12, 0.02, -0.03],
            [-0.02, 0.04, -0.08],
            [0.10, -0.01, 0.02],
            [0.04, 0.03, -0.05],
            [0.08, 0.00, -0.02]]
    a = meta_analysis_from_seed_matrix(rows, labels=["a", "b", "c"],
                                       n_bootstrap=200, seed=7)
    b = meta_analysis_from_seed_matrix(rows, labels=["a", "b", "c"],
                                       n_bootstrap=200, seed=7)
    assert a["status"] == "heterogeneous", a
    assert a["bootstrap"] == b["bootstrap"]
    assert a["bootstrap"]["available"] is True
    assert a["bootstrap"]["n_valid_resamples"] > 0
    assert a["bootstrap"]["i2_ci95"] is not None
    assert a["bootstrap"]["prob_i2_ge_50pct"] is not None
    print("  [ok] 种子矩阵 bootstrap 固定种子可复现")


def _ruler(seed, effects, drop_task=None, noisy=True):
    results = {}
    for task, delta in effects.items():
        if task == drop_task:
            continue
        noise = (0.01 if seed % 2 else -0.01) if noisy else 0.0
        results[task] = {
            "256": {"em": 0.5 + delta + noise, "n": 10},
            "512": {"em": 0.5 + delta + 0.4 * noise, "n": 10},
        }
    return {"results": results, "ctx_lens": [256, 512],
            "n_samples": 10, "sample_seed": 1, "max_new": 8}


def _runs(task_effects=None):
    task_effects = task_effects or {
        "niah_synth": 0.10, "niah_corpus": 0.00, "niah_multikey": -0.10}
    runs = {"cand": {}, "ref": {}}
    for seed in (1, 2, 3, 4, 5):
        runs["cand"][seed] = {"ruler": _ruler(seed, task_effects)}
        runs["ref"][seed] = {"ruler": _ruler(seed, {k: 0.0
                                                    for k in task_effects},
                                            noisy=False)}
    return runs


def test_pipeline_ruler_task_heterogeneity_and_attachment():
    runs = _runs()
    analysis = _ruler_heterogeneity_analysis(
        runs, [("cand", "ref")], n_bootstrap=100)
    outcome = analysis["outcomes"][0]
    task = outcome["axes"]["task"]
    assert task["status"] == "heterogeneous", task
    assert task["k"] == 3 and task["n_seeds"] == 5
    assert task["i2_pct"] > 90
    assert abs(task["pooled_random"]) < 0.01
    assert task["loo_sign_stable"] is False
    context = outcome["axes"]["context"]
    assert context["status"] == "no_detected_heterogeneity", context

    hyp = [{"id": "H4", "candidate": "cand", "reference": "ref",
            "metric": "RULER平均", "status": "支持", "detail": "原判定"}]
    attach_to_hypotheses(hyp, analysis)
    assert hyp[0]["status"] == "支持"
    assert hyp[0]["detail"].startswith("原判定")
    assert "RULER 平均不能被解释为各层共同效应" in hyp[0]["detail"]
    assert hyp[0]["ruler_heterogeneity"]["task"]["i2"] == task["i2"]
    print("  [ok] 流水线识别任务异质性，挂到 H4 但不改判定")


def test_pipeline_excludes_protocol_mismatch_explicitly():
    runs = _runs()
    runs["ref"][5]["ruler"] = _ruler(5, {"niah_synth": 0.0,
                                         "niah_corpus": 0.0},
                                     drop_task="niah_multikey")
    analysis = _ruler_heterogeneity_analysis(
        runs, [("cand", "ref")], n_bootstrap=20)
    outcome = analysis["outcomes"][0]
    assert outcome["excluded"]["protocol_mismatch"] == 1, outcome
    assert outcome["n_seeds"] == 4, outcome
    task = outcome["axes"]["task"]
    assert task["excluded"]["protocol_mismatch"] == 1, task
    assert task["k"] == 3 and task["n_seeds"] == 4, task
    print("  [ok] 缺格/协议不一致的种子被显式剔除，不猜缺失值")


def test_render_section_reports_boundaries():
    analysis = _ruler_heterogeneity_analysis(
        _runs(), [("cand", "ref")], n_bootstrap=50)
    md = "\n".join(_render_ruler_heterogeneity_section(analysis))
    assert "RULER 任务/长度异质性" in md
    assert "I²" in md and "不等于各层同质" in md
    assert "不改变 H1–H7 判定" in md
    assert "cand vs ref" in md
    print("  [ok] 汇总渲染包含异质性、边界声明与对比行")
