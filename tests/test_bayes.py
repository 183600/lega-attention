"""JZS 贝叶斯证据层的单元/集成测试。

覆盖：已知 BF 参照值、尺度/符号不变性、后验方向与 ROPE 三区互补、先验
尺度敏感性、零方差/种子不足/坏输入、稳健大数、SESOI 尺度接入、汇总渲染
与 significance.json 计数。
"""
import math

from lega.bayes import bayes_paired_evidence, evidence_label
from lega.equivalence import paired_metric_diffs


def _close(a, b, tol=1e-9):
    return abs(float(a) - float(b)) <= tol


def test_known_jzs_values_and_labels():
    # 参照值由高精度 (δ, logσ) 二重积分得到；网格误差应远小于 1%。
    big = bayes_paired_evidence([1.0, 2.0, 3.0, 4.0, 5.0])
    assert big["status"] == "ok" and big["n"] == 5
    assert _close(big["bf10"], 5.751914670076, tol=0.02), big
    assert big["label"] == "弱支持 H1", big
    assert big["log_bf10"] > math.log(3.0)
    null = bayes_paired_evidence([-0.1, 0.1, -0.05, 0.05, 0.0])
    assert _close(null["bf10"], 0.397362624480641, tol=1e-9), null
    assert null["label"] == "证据不足（不区分 H0/H1）", null
    assert evidence_label(0.01) == "极强支持 H0"
    assert evidence_label(0.05) == "中等支持 H0"
    assert evidence_label(5) == "弱支持 H1"
    assert evidence_label(50) == "强支持 H1"
    assert evidence_label(3.0) == "证据不足（不区分 H0/H1）"
    assert evidence_label(1.0 / 3.0) == "证据不足（不区分 H0/H1）"
    print("  [ok] JZS BF 参照值/证据标签")


def test_scale_and_sign_symmetry():
    base = [0.1, -0.2, 0.3, 0.05, 0.15]
    ref = bayes_paired_evidence(base, rope=0.2)
    for factor in (1e-6, 1e-3, 1e3, 1e6):
        scaled = bayes_paired_evidence([x * factor for x in base],
                                       rope=0.2 * factor)
        assert _close(scaled["log_bf10"], ref["log_bf10"], 1e-9), scaled
        assert _close(scaled["posterior_prob_positive"],
                      ref["posterior_prob_positive"], 1e-9)
    neg = bayes_paired_evidence([-x for x in base], rope=0.2)
    assert _close(neg["log_bf10"], ref["log_bf10"], 1e-9)
    assert _close(neg["posterior_prob_positive"],
                  ref["posterior_prob_negative"], 1e-9)
    assert _close(neg["posterior_median"], -ref["posterior_median"], 1e-9)
    print("  [ok] 尺度不变、符号对称")


def test_posterior_direction_and_rope_partition():
    r = bayes_paired_evidence([1.0, 2.0, 3.0, 4.0, 5.0], rope=1.0)
    assert 0.0 <= r["posterior_prob_positive"] <= 1.0
    assert _close(r["posterior_prob_positive"]
                  + r["posterior_prob_negative"], 1.0, 1e-9)
    total = (r["prob_in_rope"] + r["prob_meaningful_positive"]
             + r["prob_meaningful_negative"])
    assert _close(total, 1.0, 1e-9), r
    assert r["prob_meaningful_positive"] > 0.5
    assert r["posterior_ci95"][0] < r["posterior_median"] < r["posterior_ci95"][1]
    # ROPE 越宽，落在等价带内的后验质量不应下降。
    wide = bayes_paired_evidence([1.0, 2.0, 3.0, 4.0, 5.0], rope=3.0)
    assert wide["prob_in_rope"] > r["prob_in_rope"]
    print("  [ok] 后验方向/ROPE 三区互补")


def test_prior_scale_sensitivity_is_reported():
    r = bayes_paired_evidence([1.0, 2.0, 3.0, 4.0, 5.0])
    cells = r["sensitivity"]
    assert [c["factor"] for c in cells] == [0.5, 1.0, 2.0], cells
    assert all(c["bf10"] is not None for c in cells), cells
    assert r["sensitivity_verdict"] == "各先验尺度均支持 H1", r
    wider = bayes_paired_evidence([-0.1, 0.1, -0.05, 0.05, 0.0])
    assert "先验尺度" in wider["sensitivity_verdict"], wider
    print("  [ok] 0.5×/1×/2× 先验尺度敏感性并列披露")


def test_degenerate_insufficient_and_bad_inputs():
    zero = bayes_paired_evidence([1.0] * 5, rope=0.5)
    assert zero["status"] == "degenerate_zero_variance", zero
    assert zero["bf10"] is None and zero["label"] == "无法计算"
    assert zero["posterior_ci95"] == [1.0, 1.0]
    assert zero["posterior_prob_positive"] == 1.0
    assert zero["prob_meaningful_positive"] == 1.0
    flat = bayes_paired_evidence([0.0] * 5)
    assert flat["status"] == "degenerate_zero_variance"
    assert flat["posterior_prob_positive"] == 0.5
    one = bayes_paired_evidence([1.0])
    assert one["status"] == "insufficient_seeds" and one["bf10"] is None
    two = bayes_paired_evidence([1.0, 2.0])
    assert two["status"] == "insufficient_seeds"
    assert two["decision_eligible"] is False and two["bf10"] is not None
    # None/bool/非有限/字符串整项剔除。
    bad = bayes_paired_evidence([1.0, None, 2.0, True, float("nan"),
                                 float("inf"), "x", 3.0])
    assert bad["n"] == 3 and bad["status"] == "ok", bad
    huge = bayes_paired_evidence(
        [1e308, -1e308, 1e308, -1e308, 1e307], rope=1e306)
    assert huge["status"] == "ok" and huge["bf10"] is not None, huge
    assert huge["posterior_mean"] is None or math.isfinite(huge["posterior_mean"])
    for bad_scale in (True, 0, -1, float("nan"), float("inf"), "0.7", None):
        try:
            bayes_paired_evidence([1.0, 2.0, 3.0], prior_scale=bad_scale)
            raise AssertionError(f"非法 prior_scale 应被拒绝: {bad_scale!r}")
        except ValueError:
            pass
    for bad_rope in (True, 0, -1, float("nan"), float("inf"), "0.1"):
        try:
            bayes_paired_evidence([1.0, 2.0, 3.0], rope=bad_rope)
            raise AssertionError(f"非法 rope 应被拒绝: {bad_rope!r}")
        except ValueError:
            pass
    for bad_min in (True, 1, 2.5, "3"):
        try:
            bayes_paired_evidence([1.0, 2.0, 3.0], min_seeds=bad_min)
            raise AssertionError(f"非法 min_seeds 应被拒绝: {bad_min!r}")
        except ValueError:
            pass
    for bad_scales in ([], [True], ["0.5"], [0], [float("nan")]):
        try:
            bayes_paired_evidence([1.0, 2.0, 3.0],
                                  prior_scales=bad_scales)
            raise AssertionError(f"非法 prior_scales 应被拒绝: {bad_scales!r}")
        except ValueError:
            pass
    print("  [ok] 零方差/种子不足/坏输入/非法参数显式处理")


def test_pipeline_integration_uses_metric_scale_and_rope():
    from run_all import (_bayes_overview, _enrich_paired,
                         _render_bayes_section)

    a = [101.0, 102.0, 103.0, 104.0, 105.0]
    b = [100.0] * 5
    res = {"n": 5, "mean_diff": 3.0, "std_diff": 1.0,
           "metric": "PPL", "lower_is_better": True}
    out = _enrich_paired(res, a, b)
    bayes = out.get("bayes")
    eq = out.get("equivalence")
    assert isinstance(bayes, dict) and isinstance(eq, dict), out
    assert bayes["status"] == "ok" and bayes["bf10"] > 3.0, bayes
    assert _close(bayes["rope"], eq["margin"]), (bayes, eq)
    direct = bayes_paired_evidence(paired_metric_diffs(a, b, "PPL"),
                                   rope=eq["margin"])
    assert _close(bayes["bf10"], direct["bf10"]), (bayes, direct)
    assert paired_metric_diffs(a, b, "unknown") is None
    md = "\n".join(_render_bayes_section([out]))
    assert "贝叶斯证据" in md and "先验尺度" in md and "弱支持 H1" in md
    assert "P(ROPE)" in md and "补充性" in md
    overview = _bayes_overview([out])
    assert overview["n_rows"] == 1
    assert overview["counts"]["support_h1"] == 1, overview
    # n=10 的零效应数据应落到“弱支持 H0”，而不是被算作证据不足。
    null_a = [100.01, 99.99, 100.005, 99.995, 100.0] * 2
    null_b = [100.0] * 10
    null_raw = [x - y for x, y in zip(null_a, null_b)]
    null_res = {"n": 10, "mean_diff": sum(null_raw) / 10, "std_diff": 0.01,
                "metric": "PPL", "lower_is_better": True}
    null_out = _enrich_paired(null_res, null_a, null_b)
    assert null_out["bayes"]["bf10"] < 1.0 / 3.0, null_out["bayes"]
    null_overview = _bayes_overview([out, null_out])
    assert null_overview["counts"]["support_h1"] == 1
    assert null_overview["counts"]["support_h0"] == 1, null_overview
    # n<3 仍可算描述性 BF，但不计入支持/反对证据的计数。
    small = _enrich_paired(
        {"n": 2, "mean_diff": 1.5, "std_diff": 0.7, "metric": "PPL",
         "lower_is_better": True}, [101.0, 102.0], [100.0, 100.0])
    assert small["bayes"]["decision_eligible"] is False
    small_overview = _bayes_overview([small])
    assert small_overview["counts"]["insufficient"] == 1, small_overview
    print("  [ok] SESOI 尺度接入、ROPE 复用与汇总渲染")
