"""SESOI / TOST 等价性检验的单元测试。

覆盖：等价/非等价/边界、零方差、n<3、非法 margin/alpha、配对错位、
PPL 相对差与 log 比、RULER 绝对差、0.5×/1×/2× 敏感性、Holm 家族校正、
汇总渲染所需的 status/label，以及因果安慰剂解读。
"""
import json
import math
import os
import shutil
import tempfile

from lega.equivalence import (assess_paired_equivalence, interpret_equivalence,
                              interpolate_margin, metric_margin, t_critical,
                              tost_from_diffs, tost_paired)
from run_all import (_apply_holm, _causal_eq_note, _enrich_paired,
                     _equivalence_overview, _render_parameter_matched_control,
                     build_summary)


def test_tost_equivalent_and_non_equivalent():
    eq = tost_from_diffs([0.001, -0.002, 0.0005, 0.0015, -0.001], 0.01)
    assert eq["equivalent"] is True, eq
    assert eq["p_tost"] < 0.05, eq
    assert eq["ci90"][0] > -0.01 and eq["ci90"][1] < 0.01, eq
    neq = tost_from_diffs([0.020, 0.021, 0.019, 0.0205, 0.0195], 0.01)
    assert neq["equivalent"] is False, neq
    assert neq["p_tost"] > 0.05, neq
    assert neq["ci90"][0] > 0.01, neq
    print("  [ok] TOST 等价/非等价与 90% 区间")


def test_tost_zero_variance_and_boundary():
    zero = tost_from_diffs([0.0] * 5, 0.01)
    assert zero["equivalent"] is True and zero["p_tost"] == 0.0, zero
    inside = tost_from_diffs([0.005] * 5, 0.01)
    assert inside["equivalent"] is True, inside
    outside = tost_from_diffs([0.02] * 5, 0.01)
    assert outside["equivalent"] is False and outside["p_tost"] == 1.0, outside
    # 均值恰在边界上：保守地不判等价。
    on_boundary = tost_from_diffs([0.01] * 5, 0.01)
    assert on_boundary["equivalent"] is False, on_boundary
    # 对称方向同样处理。
    assert tost_from_diffs([-0.02] * 5, 0.01)["equivalent"] is False
    print("  [ok] TOST 零方差/边界保守处理")


def test_tost_rejects_bad_margin_alpha_and_pairs():
    for bad in (True, 0, -0.01, float("nan"), float("inf"), "0.01", None,
                10 ** 400):
        try:
            tost_from_diffs([0.001] * 5, bad)
            raise AssertionError(f"非法 margin 应被拒绝: {bad!r}")
        except ValueError:
            pass
    for bad in (True, 0.0, 0.5, 1.0, float("nan"), "0.05", None):
        try:
            tost_from_diffs([0.001] * 5, 0.01, alpha=bad)
            raise AssertionError(f"非法 alpha 应被拒绝: {bad!r}")
        except ValueError:
            pass
    try:
        tost_paired([1, 2], [1], 0.01)
        raise AssertionError("长度不一致应被拒绝")
    except ValueError:
        pass
    try:
        tost_paired([1, None], [1, 2], 0.01)
        raise AssertionError("None 位置不一致应被拒绝")
    except ValueError:
        pass
    # 非有限/bool 整对剔除，不伪造成观测。
    r = tost_paired([1.0, float("nan"), True, 3.0],
                    [1.0, 1.0, 1.0, 2.0], 0.5)
    assert r["n"] == 2, r
    print("  [ok] TOST 非法 margin/alpha/配对与坏观测剔除")


def test_tost_n_less_than_three_never_equivalent():
    r = tost_from_diffs([0.001, 0.001], 0.01)
    assert r["n"] == 2, r
    assert r["equivalent"] is False, r
    assert r["status"] == "insufficient_seeds", r
    empty = tost_from_diffs([], 0.01)
    assert empty["n"] == 0 and empty["equivalent"] is False
    assert empty["status"] == "insufficient_seeds"
    print("  [ok] n<3 不作等价性判定")


def test_metric_margins_and_transform_scales():
    assert metric_margin("unknown") is None
    assert metric_margin("PPL")["kind"] == "relative"
    assert metric_margin("PPL(log)")["kind"] == "log_ratio"
    assert metric_margin("RULER:niah_synth")["kind"] == "absolute"
    a = [100.8, 100.9, 100.7, 100.9, 100.8]
    b = [100.0] * 5
    rel = assess_paired_equivalence(a, b, "PPL", lower_is_better=True)
    assert rel["kind"] == "relative" and rel["equivalent"] is True, rel
    log = assess_paired_equivalence(a, b, "PPL(log)", lower_is_better=True)
    assert log["kind"] == "log_ratio" and log["equivalent"] is True, log
    # 参照 <=0 时相对差无定义：跳过而不是伪造巨大相对差。
    bad = assess_paired_equivalence([1.0, 2.0, 3.0], [0.0, 0.0, 0.0], "PPL")
    assert bad["n"] == 0 and bad["status"] == "insufficient_seeds", bad
    print("  [ok] 指标 SESOI 尺度与无效参照处理")


def test_sensitivity_grid_marks_only_wide_margin_equivalent():
    # 相对差恒为 +1.5%：0.5%/1% 不判等价，2% 判等价。
    a = [101.5] * 5
    b = [100.0] * 5
    res = assess_paired_equivalence(a, b, "PPL", lower_is_better=True)
    marks = [bool(c["equivalent"]) for c in res["sensitivity"]]
    assert marks == [False, False, True], res["sensitivity"]
    margins = [c["margin"] for c in res["sensitivity"]]
    assert abs(margins[0] - 0.005) < 1e-12
    assert abs(margins[1] - 0.01) < 1e-12
    assert abs(margins[2] - 0.02) < 1e-12
    # 候选明显更好（−2%）时：不等价（差异超过 1%），但非劣（没有更差）。
    better = assess_paired_equivalence([98.0, 98.1, 97.9, 98.0, 98.0],
                                       [100.0] * 5, "PPL",
                                       lower_is_better=True)
    assert better["equivalent"] is False and better["noninferior"] is True, better
    # log 尺度按几何比换算，而不是直接乘 log margin。
    log = assess_paired_equivalence(a, b, "PPL(log)", lower_is_better=True)
    lm = [c["margin"] for c in log["sensitivity"]]
    assert abs(lm[2] - math.log1p(0.02)) < 1e-12, lm
    print("  [ok] 0.5×/1×/2× 敏感性网格与 log 比换算")


def test_interpret_equivalence_categories():
    base = {"n": 5, "mean_diff": 0.001, "margin": 0.01,
            "p_tost": 0.01, "direction": "better", "beyond_margin": False,
            "noninferior": True}
    b = dict(base)
    b.update(interpret_equivalence(base, p_adj=0.9))
    assert b["status"] == "equivalent", b
    # 显著但幅度小于 SESOI：实际意义有限，而不是“重要收益”。
    small = {"n": 5, "mean_diff": 0.002, "margin": 0.01,
             "p_tost": 0.8, "direction": "better", "beyond_margin": False,
             "noninferior": True}
    s = interpret_equivalence(small, p_adj=0.01)
    assert s["status"] == "significant_below_sesoi", s
    # 显著且超过 SESOI：实际有意义。
    big = {"n": 5, "mean_diff": 0.03, "margin": 0.01,
           "p_tost": 0.9, "direction": "better", "beyond_margin": True,
           "noninferior": False}
    g = interpret_equivalence(big, p_adj=0.01)
    assert g["status"] == "significant_meaningful_better", g
    wrong = dict(big, direction="worse")
    w = interpret_equivalence(wrong, p_adj=0.01)
    assert w["status"] == "significant_meaningful_worse", w
    # 非劣：不满足等价（可能更好很多），但单侧支持“没有差超过 SESOI”。
    n = interpret_equivalence({"n": 5, "mean_diff": -0.02, "margin": 0.01,
                               "p_tost": 0.9, "direction": "better",
                               "beyond_margin": True, "noninferior": True},
                              p_adj=0.4)
    assert n["status"] == "noninferior", n
    # 未定不能被解释为无差异。
    u = interpret_equivalence({"n": 5, "mean_diff": 0.005, "margin": 0.01,
                               "p_tost": 0.3, "direction": "better",
                               "beyond_margin": False, "noninferior": False},
                              p_adj=0.4)
    assert u["status"] == "undetermined", u
    # 种子不足优先于任何 p 值。
    i = interpret_equivalence({"n": 2, "mean_diff": 0.0, "margin": 0.01,
                               "p_tost": 0.0, "direction": "same",
                               "beyond_margin": False, "noninferior": True},
                              p_adj=0.9)
    assert i["status"] == "insufficient_seeds", i
    # 合法但超出 float 上限的 p 值按缺失证据处理：不能让 OverflowError
    # 穿透判定入口，也不能把不可表示值当成显著。
    huge_adj = interpret_equivalence(
        {"n": 5, "p_tost": 0.9, "p_tost_adj": 10 ** 400}, p_adj=10 ** 400)
    assert huge_adj["status"] == "undetermined", huge_adj
    assert huge_adj["p_used"] == 0.9
    assert huge_adj["multiplicity_adjusted"] is False
    huge_raw = interpret_equivalence({"n": 5, "p_tost": 10 ** 400},
                                     p_adj=10 ** 400)
    assert huge_raw["status"] == "undetermined", huge_raw
    assert huge_raw["p_used"] is None
    # no-op 的零差不是等价证据。
    noop = interpret_equivalence({"n": 5, "mean_diff": 0.0, "margin": 0.01,
                                  "p_tost": 0.0, "direction": "same",
                                  "beyond_margin": False,
                                  "noninferior": True,
                                  "not_applicable": True,
                                  "not_applicable_reason": "全部 no-op"},
                                 p_adj=0.5)
    assert noop["status"] == "not_applicable", noop
    assert noop["equivalent"] is False, noop
    print("  [ok] 等价/非劣/显著但低于 SESOI/未定/种子不足分类")


def test_t_critical_known_values():
    assert abs(t_critical(4, 0.05) - 2.776445) < 5e-4, t_critical(4, 0.05)
    assert abs(t_critical(4, 0.10) - 2.131847) < 5e-4, t_critical(4, 0.10)
    assert t_critical(0, 0.05) is None
    assert t_critical(4, 0.0) is None
    assert t_critical(4, 1.0) is None
    # 合法但超出 float 上限的整数自由度不是可计算值：必须返回 None，
    # 不能把 OverflowError 抛给调用方。
    assert t_critical(10 ** 400, 0.05) is None
    # df 为最小次正规数时真实临界值远超搜索上限；不能把搜索边界当成结果。
    assert t_critical(5e-324, 0.05) is None
    print("  [ok] t 临界值已知数值")


def test_apply_holm_adjusts_tost_and_sensitivity_families():
    def mk(p_tost, n=5):
        eq = {"n": n, "p_tost": p_tost, "p_tost_adj": None,
              "mean_diff": 0.002, "margin": 0.01, "direction": "better",
              "beyond_margin": False, "noninferior": True,
              "multiplicity_adjusted": False,
              "sensitivity": [
                  {"n": n, "factor": f, "margin": 0.01 * f,
                   "p_tost": p_tost * f, "p_tost_adj": None,
                   "equivalent": False, "status": "undetermined"}
                  for f in (0.5, 1.0, 2.0)]}
        return {"candidate": f"c{p_tost}", "reference": "r", "metric": "PPL",
                "n": n, "p": 0.5, "mean_diff": 0.002, "std_diff": 0.001,
                "lower_is_better": True, "equivalence": eq}
    t1, t2 = mk(0.03), mk(0.04)
    tests = _apply_holm([t1, t2])
    # m=2 Holm step-down 的单调性：0.03→0.06、0.04→max(0.06,0.04)=0.06；
    # 两者都不再 <0.05，等价判定必须被校正后 p 收紧。
    assert tests[0]["equivalence"]["p_tost_adj"] == 0.06, tests[0]
    assert tests[1]["equivalence"]["p_tost_adj"] == 0.06, tests[1]
    assert tests[0]["equivalence"]["equivalent"] is False
    assert tests[1]["equivalence"]["equivalent"] is False
    # 敏感性网格每档也独立 Holm（0.015→0.03, 0.02→max(0.03,0.02)=0.03）。
    assert tests[0]["equivalence"]["sensitivity"][0]["p_tost_adj"] == 0.03
    assert tests[1]["equivalence"]["sensitivity"][0]["p_tost_adj"] == 0.03
    # n<3 不占家族大小，也不获得 p_tost_adj。
    small = mk(0.001, n=2)
    tiny = _apply_holm([mk(0.03), small])[1]
    assert tiny["equivalence"]["p_tost_adj"] is None, tiny
    assert tiny["equivalence"]["status"] == "insufficient_seeds", tiny
    print("  [ok] TOST 主族/敏感性网格 Holm 与 n<3 排除")


def test_parameter_matched_render_prefers_tost_equivalence():
    """TOST 等价时不能把“5 个种子方向一致”渲染成容量红利已被排除。"""
    t = {"candidate": "lega", "reference": "lega_placebo", "metric": "PPL",
         "n": 5, "mean_diff": -0.001, "std_diff": 0.0, "p_adj": 0.0,
         "equivalence": {"status": "equivalent"}}
    md = "\n".join(_render_parameter_matched_control(t))
    assert "SESOI 等价区" in md, md
    assert "证据特定性成立" not in md, md
    cap = {"candidate": "lega_placebo", "reference": "qgate", "metric": "PPL",
           "n": 5, "mean_diff": -0.001, "std_diff": 0.0, "p_adj": 0.0,
           "equivalence": {"status": "equivalent"}}
    md2 = "\n".join(_render_parameter_matched_control(t, capacity_test=cap))
    assert "噪声特征与 qgate 落在等价区" in md2, md2
    assert "单独即可降低 PPL" not in md2, md2
    print("  [ok] 参数匹配安慰剂：等价性优先于显著的微小方向差")


def test_holm_adjusted_tost_does_not_keep_raw_noninferior():
    """校正后 TOST p 不显著时，不能沿用校正前的 noninferior 原始标志。"""
    def mk(p):
        return {"candidate": "c", "reference": "r", "metric": "PPL",
                "n": 5, "p": 0.5, "mean_diff": 0.002, "std_diff": 0.001,
                "lower_is_better": True,
                "equivalence": {"n": 5, "p_tost": p, "p_tost_adj": None,
                                "mean_diff": 0.002, "margin": 0.01,
                                "direction": "worse", "beyond_margin": False,
                                "noninferior": True, "p_noninf_worse": p,
                                "multiplicity_adjusted": False,
                                "sensitivity": []}}
    out = _apply_holm([mk(0.03), mk(0.04)])
    for t in out:
        eq = t["equivalence"]
        assert eq["p_tost_adj"] == 0.06, eq
        assert eq["p_noninf_worse_adj"] == 0.06, eq
        assert eq["noninferior"] is False, eq
        assert eq["status"] != "noninferior", eq
    # 外部/手写结果若只带校正 p、没有可校正的单侧证据，也不能退回原始标志。
    stale = {"n": 5, "p_tost": 0.03, "p_tost_adj": 0.06,
             "mean_diff": 0.002, "margin": 0.01, "direction": "worse",
             "beyond_margin": False, "noninferior": True}
    r = interpret_equivalence(stale, p_adj=None)
    assert r["status"] != "noninferior", r
    print("  [ok] Holm 校正后不再沿用原始 noninferior 标志")


def test_enrich_paired_attaches_equivalence_and_renders_status():
    a = [99.8, 100.1, 99.9, 100.2, 100.0]
    b = [100.0] * 5
    res = {"candidate": "lega", "reference": "standard", "metric": "PPL",
           "lower_is_better": True, "n": 5, "mean_diff": -0.0002,
           "std_diff": 0.0002, "p": 0.1, "p_adj": None}
    out = _enrich_paired(res, a, b)
    eq = out["equivalence"]
    assert isinstance(eq, dict) and eq["status"] == "equivalent", eq
    assert eq["margin"] == 0.01 and eq["kind"] == "relative", eq
    # 未知指标不硬套 SESOI。
    unknown = {"candidate": "x", "reference": "y", "metric": "PPL交互",
               "lower_is_better": True, "n": 5, "mean_diff": 0.0,
               "std_diff": 0.1, "p": 0.5, "p_adj": None}
    assert _enrich_paired(unknown, [1] * 5, [1] * 5)["equivalence"] is None
    print("  [ok] _enrich_paired 接入等价性与未知指标跳过")


def test_equivalence_overview_and_causal_notes():
    a = [99.8, 100.1, 99.9, 100.2, 100.0]
    b = [100.0] * 5
    res = {"candidate": "lega", "reference": "standard", "metric": "PPL",
           "lower_is_better": True, "n": 5, "mean_diff": -0.0002,
           "std_diff": 0.0002, "p": 0.1, "p_adj": None}
    tests = _apply_holm([_enrich_paired(res, a, b)])
    ov = _equivalence_overview(tests)
    assert ov["counts"].get("equivalent") == 1, ov
    assert ov["margins"]["PPL"]["margin"] == 0.01, ov

    base = {"lesion": "head_permute", "n": 5, "n_no_op_seeds": 0,
            "equivalence": {"status": "equivalent", "n": 5,
                            "mean_diff": 0.001, "margin": 0.01,
                            "kind": "absolute", "display_scale": 1.0,
                            "display_unit": "nat"}}
    assert "容量" in _causal_eq_note(base)
    base["equivalence"]["status"] = "significant_meaningful_worse"
    assert "因果" in _causal_eq_note(base)
    base["n_no_op_seeds"] = 5
    assert "no-op" in _causal_eq_note(base)
    closed = {"lesion": "gate_closed", "n": 5, "n_no_op_seeds": 0,
              "equivalence": {"status": "equivalent", "n": 5}}
    assert "阳性对照失败" in _causal_eq_note(closed)
    print("  [ok] 等价性计数与因果安慰剂解读")


def test_build_summary_renders_equivalence_section():
    """汇总页必须把 TOST 等价性与“不显著”分开呈现。"""
    tmp = tempfile.mkdtemp()
    try:
        def write(variant, seed, ppl):
            with open(os.path.join(tmp, f"eval_{variant}_seed{seed}.json"),
                      "w", encoding="utf-8") as f:
                json.dump({"variant": variant, "seed": seed, "ppl": ppl,
                           "val_ppl": ppl, "val_loss": math.log(ppl),
                           "gate_mean": 0.5 if variant == "lega" else None,
                           "n_params": 100, "n_gate_extra": 0}, f)
        for seed in (1, 2, 3, 4, 5):
            write("lega", seed, 100.0)
            write("standard", seed, 100.2)
        out = build_summary(tmp)
        md = out["markdown"]
        assert "SESOI 与等价性/非劣性" in md, md
        eq_tests = [t for t in out["tests"] if t.get("metric") == "PPL"]
        assert eq_tests, out["tests"]
        eq = eq_tests[0]["equivalence"]
        assert eq["status"] == "equivalent", eq
        assert eq["p_tost_adj"] is not None and eq["equivalent"] is True, eq
        assert "等价（差异小于 SESOI" in md, md
        assert "未定" in md
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] 汇总页渲染 SESOI/TOST 段并与显著性分开")


def test_invalid_equivalence_status_survives_holm_and_render():
    """损坏/不可表示的等价性证据必须保留 invalid，不能被重算成种子不足。"""
    invalid = {"status": "invalid",
               "label": "无法计算等价性（ValueError: broken）",
               "equivalent": False, "multiplicity_adjusted": False,
               "p_used": None, "reason": "broken"}
    t = {"candidate": "lega", "reference": "standard", "metric": "PPL",
         "n": None, "p": None, "mean_diff": None, "std_diff": None,
         "lower_is_better": True, "equivalence": invalid}
    out = _apply_holm([t])[0]
    assert out["equivalence"]["status"] == "invalid", out["equivalence"]
    assert "broken" in out["equivalence"]["label"], out["equivalence"]
    # interpret 单独调用同样不能把 invalid 变成 insufficient_seeds。
    from lega.equivalence import interpret_equivalence
    r = interpret_equivalence(invalid, p_adj=0.9)
    assert r["status"] == "invalid" and "broken" in r["label"], r
    print("  [ok] invalid 等价性证据在 Holm/重算后仍保留")


def test_tost_extreme_finite_values_are_invalid_not_crash_or_nan():
    """有限但超出 float 表示范围的观测必须判 invalid，不能抛错或写 NaN/Inf。"""
    # 和超出 float 上限（sum -> inf，fsum 也溢出）。
    huge = tost_from_diffs([1e308] * 5, 0.01)
    assert huge["status"] == "invalid", huge
    assert huge["p_tost"] is None and huge["equivalent"] is False, huge
    # 均值可精确抵消，但方差超出 float 上限（朴素平方会 OverflowError）。
    spread = tost_from_diffs([1e308, -1e308] * 3, 0.01)
    assert spread["status"] == "invalid", spread
    assert spread["std_diff"] is None and spread["p_tost"] is None, spread
    # 极端但可表示的普通数值保持原判定，不能因硬化而误伤。
    normal = tost_from_diffs([0.001] * 5, 0.01)
    assert normal["equivalent"] is True and normal["std_diff"] == 0.0, normal
    print("  [ok] TOST 极端有限值安全降级为 invalid")


def test_tost_unrepresentable_integer_observations_are_skipped():
    """合法但超出 float 上限的整数观测不能以 OverflowError 穿透 TOST 入口。"""
    huge = [10 ** 400] * 5
    r = tost_from_diffs(huge, 0.01)
    assert r["n"] == 0 and r["status"] == "insufficient_seeds", r
    assert r["p_tost"] is None and r["equivalent"] is False, r

    paired = tost_paired(huge, [0.0] * 5, 0.01)
    assert paired["n"] == 0 and paired["n_pairs_valid"] == 0, paired
    # 同一组里可表示的观测仍照常参与，不能因一个坏值丢掉整对/整组。
    mixed = tost_paired([10 ** 400, 1.0], [0.0, 0.0], 0.5)
    assert mixed["n"] == 1 and mixed["n_pairs_valid"] == 1, mixed
    assert mixed["mean_diff"] == 1.0, mixed

    eq = assess_paired_equivalence(huge, [0.0] * 5, "PPL")
    assert eq is not None and eq["n"] == 0, eq
    assert eq["status"] == "insufficient_seeds", eq
    print("  [ok] TOST 不可表示的大整数观测按缺失剔除，不抛 OverflowError")


def test_log_ratio_margin_stays_finite_for_huge_log_margin():
    """log_ratio 的 expm1(margin) 可溢出，但换算后的对数 margin 仍可表示。"""
    huge = {"kind": "log_ratio", "margin": 1e308}
    assert interpolate_margin(huge, 0.0) == 0.0
    for factor in (0.5, 1.0, 2.0, 1e308):
        out = interpolate_margin(huge, factor)
        assert math.isfinite(out) and out >= 0.0, (factor, out)
    res = assess_paired_equivalence([1.0, 1.1, 1.2], [1.0, 1.0, 1.0],
                                    "PPL(log)", margin=1e308)
    assert all(math.isfinite(c["margin"]) for c in res["sensitivity"])
    print("  [ok] log_ratio 巨大 margin 在对数空间换算，不抛 OverflowError")


def test_interpolate_margin_rejects_unrepresentable_factor():
    """敏感性 margin 换算不能因超范围 factor 泄漏 OverflowError/inf。"""
    spec = {"kind": "absolute", "margin": 0.01}
    assert abs(interpolate_margin(spec, 2.0) - 0.02) < 1e-15
    log_spec = {"kind": "log_ratio", "margin": math.log1p(0.01),
                "relative_margin": 0.01}
    assert abs(interpolate_margin(log_spec, 0.5)
               - math.log1p(0.005)) < 1e-15
    for bad in (10 ** 400, -1.0, float("nan"), float("inf")):
        try:
            interpolate_margin(spec, bad)
            raised = False
        except ValueError:
            raised = True
        assert raised, bad
    try:
        interpolate_margin({"kind": "absolute", "margin": 1e308}, 1e308)
        raised = False
    except ValueError:
        raised = True
    assert raised
    print("  [ok] interpolate_margin 拒绝不可表示/非法 factor，不返回 inf")
