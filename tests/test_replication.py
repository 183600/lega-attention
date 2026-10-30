"""新种子复现预测的单元/集成测试。

覆盖：t 预测区间比均值 CI 更宽、同号/反向概率、SESOI 概率、零方差退化、
n<3 与坏输入、留一方向一致率，以及配对检验/等价性输出中的集成字段。
"""
import math

from lega.equivalence import assess_paired_equivalence
from lega.replication import replication_forecast


def _close(a, b, tol=1e-9):
    return abs(float(a) - float(b)) <= tol


def test_prediction_interval_is_wider_than_mean_confidence_interval():
    vals = [1.0, 2.0, 3.0, 4.0, 5.0]
    r = replication_forecast(vals)
    assert r["n"] == 5 and r["df"] == 4
    lo, hi = r["prediction_interval"]
    assert _close((lo + hi) / 2.0, 3.0)
    # 单次复现要加上种子自身方差：PI 半宽 = t·sd·sqrt(1+1/n)
    # 均值 CI 半宽 = t·sd/sqrt(n)，前者严格更宽。
    mean_ci_half = 2.7764451051977987 * r["sd_diff"] / math.sqrt(5)
    assert hi - 3.0 > mean_ci_half * 1.5
    assert r["pi_excludes_zero"] is False
    assert 0.90 < r["prob_same_sign"] < 0.94
    assert r["status"] == "directionally_plausible"
    print("  [ok] 新种子预测区间比均值 CI 更宽，同号概率按 t 预测分布")


def test_sign_direction_and_probabilities_are_symmetric():
    pos = replication_forecast([1.0, 2.0, 3.0, 4.0, 5.0])
    neg = replication_forecast([-1.0, -2.0, -3.0, -4.0, -5.0])
    assert _close(pos["prob_same_sign"], neg["prob_same_sign"])
    assert _close(pos["prob_same_sign"], 1.0 - pos["prob_opposite_sign"])
    assert pos["expected_sign"] == "positive"
    assert neg["expected_sign"] == "negative"
    zero = replication_forecast([-2.0, -1.0, 0.0, 1.0, 2.0])
    assert zero["expected_sign"] == "zero"
    assert zero["prob_same_sign"] == 0.5
    assert zero["status"] == "undetermined"
    print("  [ok] 正/负方向对称，零均值同号概率=0.5")


def test_sesoi_probabilities_answer_different_questions():
    r = replication_forecast([1.0, 2.0, 3.0, 4.0, 5.0], margin=1.0)
    # 同号 ≠ 实际重要：同号概率高于“超过 SESOI”的概率。
    assert r["prob_same_sign"] > r["prob_meaningful_same_direction"]
    # 同向超过 SESOI、落在 ±SESOI、反向超过 SESOI 三者互斥且穷尽。
    total = (r["prob_meaningful_same_direction"] + r["prob_within_sesoi"]
             + r["prob_meaningful_opposite_direction"])
    assert _close(total, 1.0, 1e-9)
    for key in ("prob_meaningful_same_direction", "prob_within_sesoi",
                "prob_meaningful_opposite_direction", "prob_opposite_sign"):
        assert 0.0 <= r[key] <= 1.0, (key, r[key])
    # 反号概率包含“反向但仍在等价带内”的部分，故不小于反向实际重要概率。
    assert r["prob_opposite_sign"] >= r["prob_meaningful_opposite_direction"]
    print("  [ok] SESOI 同向重要/等价带/反向重要三区互补")


def test_zero_variance_is_flagged_not_treated_as_strong_evidence():
    r = replication_forecast([0.02] * 5)
    assert r["zero_variance"] is True
    assert r["prediction_interval"] == [0.02, 0.02]
    assert r["prob_same_sign"] == 1.0
    assert r["status"] == "degenerate_zero_variance"
    flat = replication_forecast([0.0] * 5)
    assert flat["zero_variance"] is True
    assert flat["prob_same_sign"] == 0.5
    assert flat["status"] == "degenerate_zero_variance"
    print("  [ok] 零方差单列为退化情形，不伪造区间宽度")


def test_insufficient_seeds_and_bad_inputs():
    for vals in ([], [1.0], [1.0, 2.0]):
        r = replication_forecast(vals)
        assert r["status"] == "insufficient_seeds", (vals, r)
        assert r["prob_same_sign"] is None
    # None/bool/非有限值整项剔除；剩余 n=3 才计算。
    r = replication_forecast([1.0, None, 2.0, True, float("nan"),
                              float("inf"), 3.0])
    assert r["n"] == 3 and r["status"] != "insufficient_seeds"
    # n 足够但方差不可表示：是 invalid，不是“种子不足”。
    huge = replication_forecast([1e308, -1e308, 1e308])
    assert huge["n"] == 3 and huge["status"] == "invalid", huge
    for bad_alpha in (0, 1, True, "0.05", float("nan")):
        try:
            replication_forecast([1.0, 2.0, 3.0], alpha=bad_alpha)
            raise AssertionError(f"非法 alpha 应被拒绝: {bad_alpha!r}")
        except ValueError:
            pass
    for bad_margin in (0, -1, True, "0.1", float("inf")):
        try:
            replication_forecast([1.0, 2.0, 3.0], margin=bad_margin)
            raise AssertionError(f"非法 margin 应被拒绝: {bad_margin!r}")
        except ValueError:
            pass
    print("  [ok] 种子不足/坏输入/非法 alpha/margin 显式处理")


def test_loo_sign_agreement_is_assumption_light_counterpart():
    assert replication_forecast([1.0, 2.0, 3.0, 4.0, 5.0])[
        "loo_sign_agreement"] == 1.0
    assert replication_forecast([-2.0, -1.0, 0.0, 1.0, 2.0])[
        "loo_sign_agreement"] < 0.5
    assert replication_forecast([1.0])["loo_sign_agreement"] is None
    print("  [ok] 留一方向一致率作为无分布假设的朴素对照")


def test_pipeline_pair_and_equivalence_carry_replication_fields():
    from run_all import _enrich_paired

    a = [98.0, 98.1, 97.9, 98.0, 98.0]
    b = [100.0] * 5
    res = {"n": 5, "mean_diff": -2.0, "std_diff": 0.1,
           "metric": "PPL", "lower_is_better": True}
    out = _enrich_paired(res, a, b)
    rep = out.get("replication")
    assert isinstance(rep, dict) and rep["n"] == 5
    assert rep["prob_same_sign"] > 0.99
    eq_rep = (out.get("equivalence") or {}).get("replication")
    assert isinstance(eq_rep, dict)
    # PPL 等价层在相对尺度上计算（−2%），SESOI=1%：预测新种子落在
    # ±1% 内的概率应低于“仍朝更好方向超过 SESOI”的概率。
    assert eq_rep["sesoi"] == 0.01
    assert (eq_rep["prob_meaningful_same_direction"]
            > eq_rep["prob_within_sesoi"])
    direct = assess_paired_equivalence(a, b, "PPL", lower_is_better=True)
    assert direct["replication"]["status"] == eq_rep["status"]
    print("  [ok] 配对检验/等价性输出集成新种子复现预测")


def test_replication_is_deterministic():
    vals = [0.01, 0.02, 0.015, 0.018, 0.012]
    assert replication_forecast(vals) == replication_forecast(vals)
    print("  [ok] 同一输入恒定复现")
