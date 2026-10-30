"""分析选择多重宇宙（specification curve）的单元/集成测试。

覆盖：方向取向、方向一致率/同向显著率、翻转与不显著规格、基准为零/规格
不足、坏输入，以及流水线对 PPL/RULER 口径的适配和预注册假设挂载。
"""
import math

from lega.multiverse import (attach_to_hypotheses, paired_specification,
                             specification_curve)
from lega.pipeline import _multiverse_analysis


def _close(a, b, tol=1e-9):
    return abs(float(a) - float(b)) <= tol


def _spec(spec_id, values_a, values_b, lower=True, dimension="scale",
          choice=None, reference=False, inferential=True):
    spec = paired_specification(
        spec_id, dimension, choice or spec_id, values_a, values_b,
        lower_is_better=lower, metric="M", inferential=inferential)
    spec["reference_spec"] = reference
    return spec


def test_orientation_and_consistency():
    ref = _spec("raw", [10, 11, 12], [12, 13, 14], lower=True, reference=True)
    same = _spec("log", [10, 11, 12], [12, 13, 14], lower=True)
    assert ref["valid"] and _close(ref["effect"], -2.0)
    assert _close(ref["oriented_effect"], 2.0), ref
    curve = specification_curve([ref, same], reference_id="raw")
    assert curve["status"] == "robust", curve
    assert curve["n_specs"] == 2 and curve["n_valid"] == 2
    assert _close(curve["direction_consistency"], 1.0)
    assert _close(curve["significance_consistency"], 1.0)
    assert curve["flipped_specs"] == [] and curve["nonsignificant_specs"] == []
    print("  [ok] 越小越好指标取向正确，全部同向显著 -> robust")


def test_sign_flip_is_fragile_and_dimension_is_located():
    # 算术均值候选更差，几何均值候选更好：尺度选择翻转结论。
    ref = _spec("raw", [1.0, 100.0, 100.0], [50.0, 50.0, 50.0],
                lower=True, dimension="metric_scale", reference=True)
    log = _spec("log", [math.log(v) for v in (1.0, 100.0, 100.0)],
                [math.log(50.0)] * 3, lower=True,
                dimension="metric_scale", choice="log")
    curve = specification_curve([ref, log], reference_id="raw")
    assert ref["effect"] > 0 and log["effect"] < 0, (ref, log)
    assert curve["status"] == "fragile", curve
    assert curve["n_opposite_direction"] == 1
    assert [s["id"] for s in curve["flipped_specs"]] == ["log"]
    assert curve["by_dimension"]["metric_scale"]["n_opposite"] == 1
    assert curve["by_dimension"]["metric_scale"]["direction_consistency"] == 0.5
    print("  [ok] 尺度翻转被判 fragile，并能定位到 metric_scale 维度")


def test_leave_one_out_nonsignificant_is_directionally_robust():
    ref = _spec("all", [1.0, 2.0, 3.0, 4.0], [0.0] * 4,
                lower=True, reference=True)
    loo = _spec("drop4", [1.0, 2.0, 3.0], [0.0] * 3,
                lower=True, dimension="seed_subset", choice="drop_seed=4")
    curve = specification_curve([ref, loo], reference_id="all")
    assert curve["status"] == "directionally_robust", curve
    assert _close(curve["direction_consistency"], 1.0)
    assert curve["n_significant_same"] == 1
    assert _close(curve["significance_consistency"], 0.5)
    assert [s["id"] for s in curve["nonsignificant_specs"]] == ["drop4"]
    print("  [ok] 方向一致但留一规格不显著 -> directionally_robust")


def test_zero_reference_and_insufficient_specs():
    zero = _spec("zero", [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], reference=True)
    other = _spec("other", [2.0, 2.0, 2.0], [1.0, 1.0, 1.0])
    curve = specification_curve([zero, other], reference_id="zero")
    assert curve["status"] == "undetermined_reference", curve
    assert curve["reference_direction"] == "zero"
    one = specification_curve([other], reference_id="other")
    assert one["status"] == "insufficient", one
    missing = specification_curve([other], reference_id="nope")
    assert missing["status"] == "insufficient", missing
    assert missing["reference_id"] is None
    print("  [ok] 基准为 0/规格不足/基准缺失均不作稳健性声明")


def test_invalid_specs_are_dropped_not_zero_filled():
    ref = _spec("raw", [1.0, 2.0, 3.0], [2.0, 3.0, 4.0], reference=True)
    short = _spec("short", [1.0, 2.0], [2.0, 3.0], choice="short")
    curve = specification_curve([ref, short], reference_id="raw")
    assert curve["n_specs"] == 2 and curve["n_valid"] == 1
    assert curve["n_dropped"] == 1
    assert curve["dropped_specs"][0]["id"] == "short"
    assert "min_seeds" in curve["dropped_specs"][0]["reason"]
    # 即使外部手写 valid=True，n 不足也不能进入一致率分母。
    forged = dict(short, id="forged", valid=True, oriented_effect=1.0)
    guarded = specification_curve([ref, forged], reference_id="raw")
    assert guarded["n_valid"] == 1 and guarded["n_dropped"] == 1, guarded
    print("  [ok] n 不足的规格被显式丢弃，不按 0 效应补齐")


def test_bad_inputs_are_rejected():
    for bad in (None, "", 3):
        try:
            paired_specification(bad, "d", "c", [1, 2, 3], [1, 2, 3], True)
            raise AssertionError(f"非法 spec_id 应被拒绝: {bad!r}")
        except ValueError:
            pass
    try:
        paired_specification("x", "d", "c", [1, 2, 3], [1, 2], True)
        raise AssertionError("长度不一致的配对应被拒绝")
    except ValueError:
        pass
    try:
        paired_specification("x", "d", "c", [1, 2, 3], [1, 2, 3], "yes")
        raise AssertionError("非 bool lower_is_better 应被拒绝")
    except ValueError:
        pass
    a = _spec("a", [1, 2, 3], [1, 2, 3])
    try:
        specification_curve([a, dict(a)], reference_id="a")
        raise AssertionError("重复规格 id 应被拒绝")
    except ValueError:
        pass
    print("  [ok] 非法 id/长度/bool/重复规格显式报错")


def _eval(ppl, split="test"):
    return {"ppl": ppl, "split": split, "eval_ctx": 128,
            "eval_iters": 8, "val_windows": 8,
            "val_tokens_evaluated": 1024}


def _ruler(macro):
    return {"results": {"niah_synth": {
        "256": {"em": macro, "n": 10}, "512": {"em": macro, "n": 10}}},
        "ctx_lens": [256, 512], "n_samples": 10, "sample_seed": 1,
        "max_new": 8}


def _runs(ppl_a, ppl_b, ruler_a=None, ruler_b=None,
          seeds=(1, 2, 3)):
    runs = {"lega": {}, "standard": {}}
    for i, seed in enumerate(seeds):
        runs["lega"][seed] = {"eval": _eval(ppl_a[i])}
        runs["standard"][seed] = {"eval": _eval(ppl_b[i])}
        if ruler_a is not None:
            runs["lega"][seed]["ruler"] = _ruler(ruler_a[i])
            runs["standard"][seed]["ruler"] = _ruler(ruler_b[i])
    return runs


def test_pipeline_multiverse_ppl_flip_and_hypothesis_attachment():
    runs = _runs([1.0, 100.0, 100.0], [50.0, 50.0, 50.0])
    multiverse = _multiverse_analysis(runs, [("lega", "standard")])
    ppl = next(o for o in multiverse["outcomes"] if o["metric"] == "PPL")
    assert ppl["status"] == "fragile", ppl
    assert any(s["choice"] == "log" for s in ppl["flipped_specs"])
    loo = [s for s in ppl["specs"]
           if s["id"].startswith("PPL:raw:drop_seed")]
    assert loo and all(not s["valid"] for s in loo), \
        "n=3 时留一规格应显式标为不可检验"
    assert multiverse["counts"].get("fragile") == 1

    hyp = [{"id": "H1", "candidate": "lega", "reference": "standard",
            "metric": "PPL", "status": "支持", "detail": "原判定"}]
    attach_to_hypotheses(hyp, multiverse)
    assert hyp[0]["multiverse_status"] == "fragile"
    assert "log" in hyp[0]["multiverse_flipped_choices"]
    assert "分析选择多重宇宙存在方向翻转规格" in hyp[0]["detail"]
    assert hyp[0]["status"] == "支持", "规格曲线不得改变预注册判定"
    print("  [ok] 流水线适配 PPL 尺度翻转，并挂到 H1 但不改判定")


def test_pipeline_multiverse_ruler_macro_micro_and_loo():
    runs = _runs([10.0, 11.0, 12.0], [12.0, 13.0, 14.0],
                 ruler_a=[0.5, 0.6, 0.7], ruler_b=[0.4, 0.5, 0.6])
    multiverse = _multiverse_analysis(runs, [("lega", "standard")])
    ruler = next(o for o in multiverse["outcomes"]
                 if o["metric"] == "RULER平均")
    assert ruler["reference_direction"] == "candidate_better", ruler
    assert any(s["choice"] == "micro" for s in ruler["specs"])
    # 三种子下留一规格 n=2，应显式不可检验而不是伪造成 0 效应。
    dropped = [s for s in ruler["specs"] if s["id"].startswith(
        "RULER:macro:drop_seed")]
    assert dropped and all(not s["valid"] for s in dropped), dropped
    print("  [ok] 流水线适配 RULER macro/micro，留一规格显式不可检验")
