"""缺失结果敏感性（完成者分析 vs 计划人群）的单元/流水线测试。

覆盖：计划/观测种子分类、两臂缺失率、临界点、观测范围边界、相对尺度
SESOI、坏输入不伪造稳健，以及 PPL/RULER 流水线适配与汇总渲染。
"""
import json
import math
import os
import shutil
import tempfile

from lega.missingness import aggregate_missingness, paired_missingness
from lega.pipeline import (_missingness_analysis,
                           _render_missingness_section)
from run_all import build_summary


def _close(a, b, tol=1e-9):
    return abs(float(a) - float(b)) <= tol


def test_complete_case_has_no_tipping_point():
    res = paired_missingness([1, 2, 3], {1: 10, 2: 11, 3: 12},
                             {1: 20, 2: 21, 3: 22})
    assert res["status"] == "complete", res
    assert res["n_planned"] == 3 and res["n_observed"] == 3
    assert res["n_missing"] == 0 and res["missing_fraction"] == 0.0
    assert res["observed_mean_diff"] == -10.0
    assert res["sign_robust"] is True and res["margin_robust"] is True
    assert res["missing_mean_for_zero"] is None
    assert res["full_mean_range"] == [-10.0, -10.0]
    print("  [ok] 无缺失：完成者分析=计划人群，临界点不适用")


def test_missing_candidate_is_robust_when_bounds_exclude_zero():
    res = paired_missingness(
        [1, 2, 3, 4], {1: 10, 2: 11, 3: 12},
        {1: 20, 2: 21, 3: 22, 4: 23},
        transform=lambda a, b: (a - b) / b if b else None, margin=0.01)
    assert res["missing_candidate"] == [4], res
    assert res["missing_reference"] == [] and res["missing_both"] == []
    assert _close(res["candidate_missing_fraction"], 0.25)
    assert _close(res["reference_missing_fraction"], 0.0)
    assert _close(res["differential_missingness"], 0.25)
    assert res["status"] == "robust_within_observed_range", res
    lo, hi = res["full_mean_range"]
    assert lo < res["observed_mean_diff"] < hi < 0.0
    assert res["missing_mean_for_zero"] > 0.0
    print("  [ok] 单臂缺失：两臂缺失率差与观测范围边界正确")


def test_missing_within_observed_range_can_flip_sign():
    # 完整配对差 [-0.1, .05, .05] 的均值为 0；缺失一个时，缺失值取
    # 观测范围内任一非零值都会让完整均值离开 0（范围跨过符号边界）。
    res = paired_missingness(
        [1, 2, 3, 4],
        {1: 0.9, 2: 1.05, 3: 1.05},
        {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0})
    assert res["n_observed"] == 3 and res["n_missing"] == 1
    assert res["status"] == "sensitive_within_observed_range", res
    assert res["sign_robust"] is False
    assert res["tipping_within_observed_range"] is True
    assert res["observed_min_diff"] <= res["missing_mean_for_zero"] \
        <= res["observed_max_diff"]
    assert res["full_mean_range"][0] < 0.0 < res["full_mean_range"][1]
    print("  [ok] 缺失值落在观测范围内即可翻转符号 -> 显式标敏感")


def test_margin_crossing_is_reported_separately():
    # 观测配对差 [-.1, .1, .12] 的均值为 +0.04；缺失一个时，缺失值
    # 取观测范围内的 -0.08 可把完整均值拉回 +1% SESOI 以内。
    res = paired_missingness(
        [1, 2, 3, 4], {1: 1.0, 2: 1.1, 3: 1.12},
        {1: 1.1, 2: 1.0, 3: 1.0, 4: 1.0}, margin=0.01)
    assert res["status"] == "sensitive_within_observed_range", res
    assert res["sign_robust"] is True
    assert res["margin_robust"] is False
    assert res["margin_tipping_within_observed_range"] is True
    print("  [ok] 符号稳健但实际意义阈值可被缺失值跨越")


def test_insufficient_observation_is_not_called_robust():
    res = paired_missingness([1, 2, 3], {1: 1}, {1: 2}, min_seeds=3)
    assert res["n_observed"] == 1
    assert res.get("insufficient_seeds") is True
    assert "完整配对只有 1 个" in res["note"]
    res0 = paired_missingness([1, 2], {1: None, 2: None},
                              {1: None, 2: None})
    assert res0["status"] == "no_observed_pairs", res0
    assert res0["sign_robust"] is None
    pi0 = res0["partial_identification"]
    assert pi0["mean_diff_bounds"] is None and pi0["sign_score_bounds"] is None
    resb = paired_missingness([1, 2], {1: None, 2: None},
                              {1: None, 2: None}, value_bounds=(0.0, 1.0))
    assert resb["partial_identification"]["mean_diff_bounds"] is None, resb
    print("  [ok] 观测不足/无完整配对：不作稳健判词")


def test_bad_values_and_empty_plan_are_explicit():
    res = paired_missingness([], {}, {})
    assert res["status"] == "insufficient", res
    res2 = paired_missingness(
        [1, 2, 3, 4],
        {1: True, 2: float("nan"), 3: "x", 4: 8},
        {1: 1, 2: 2, 3: 3, 4: 4})
    assert res2["n_observed"] == 1
    assert sorted(res2["missing_candidate"] + res2["missing_reference"]
                  + res2["missing_both"]) == [1, 2, 3]
    try:
        paired_missingness([True], {True: 1}, {True: 2})
        raise AssertionError("bool 种子应报错")
    except ValueError:
        pass
    print("  [ok] 坏值按缺失处理，空计划/bool 种子显式报错")


def test_planned_none_falls_back_to_observed_union():
    res = paired_missingness(None, {1: 1, 2: 2}, {1: 2, 2: 3})
    assert res["planned_seeds_source"] == "observed_union", res
    assert res["planned_seeds"] == [1, 2]
    assert res["status"] == "complete"
    print("  [ok] 未给计划种子时退化为两侧并集，并标明来源")


def test_manski_bounds_are_sharp_for_bounded_outcome():
    res = paired_missingness(
        [1, 2, 3, 4], {1: 1.0, 2: 0.0, 4: 0.0},
        {1: 0.0, 2: 1.0, 3: 0.5}, value_bounds=(0.0, 1.0))
    pi = res["partial_identification"]
    assert pi["value_bounds"] == [0.0, 1.0], pi
    lo, hi = pi["mean_diff_bounds"]
    assert _close(lo, -0.375) and _close(hi, 0.125), pi
    slo, shi = pi["sign_score_bounds"]
    assert _close(slo, -0.5) and _close(shi, 0.5), pi
    assert pi["mean_sign_identified"] is False
    assert pi["sign_score_identified"] is False
    assert pi["sign_identified"] is False
    assert pi["outcome_bounds_violations"] == []
    print("  [ok] 有界 outcome 的 Manski 均值/符号界为 sharp bounds")


def test_manski_bounds_identify_sign_and_margin():
    res = paired_missingness(
        [1, 2, 3], {1: 1.0, 2: 1.0}, {1: 0.0, 2: 0.0, 3: 0.0},
        value_bounds=(0.0, 1.0), margin=0.2)
    pi = res["partial_identification"]
    assert pi["sign_identified"] is True, pi
    assert pi["mean_sign_identified"] is True, pi
    assert pi["sign_identified_by"] == "mean_diff", pi
    assert pi["margin_status"] == "beyond_margin_positive", pi
    assert pi["practical_significance"] == "positive", pi
    assert pi["mean_diff_bounds"][0] > 0.2, pi
    print("  [ok] Manski 界可识别符号与实际意义方向")


def test_manski_without_outcome_bounds_uses_sign_score_only():
    res = paired_missingness(
        [1, 2, 3, 4], {1: 1.0, 2: 1.1, 3: 1.12},
        {1: 1.1, 2: 1.0, 3: 1.0, 4: 1.0},
        transform=lambda a, b: (a - b) / b if b else None, margin=0.01)
    pi = res["partial_identification"]
    assert pi["mean_diff_bounds"] is None, pi
    assert pi["sign_score_bounds"] is not None, pi
    assert pi["sign_identified"] is None or isinstance(
        pi["sign_identified"], bool), pi
    print("  [ok] 无界 outcome 只给符号得分 Manski 界")


def test_manski_rejects_bad_bounds_and_flags_out_of_range_values():
    try:
        paired_missingness([1], {1: 0.0}, {1: 1.0}, value_bounds=(1.0, 1.0))
        raise AssertionError("空 value_bounds 应报错")
    except ValueError:
        pass
    try:
        paired_missingness(
            [1, 2], {1: 1.0, 2: 1.0}, {1: 1.0, 2: 1.0},
            transform=lambda a, b: (a - b) / b, value_bounds=(0.0, 1.0))
        raise AssertionError("自定义尺度不应直接套用 outcome 范围")
    except ValueError:
        pass
    res = paired_missingness(
        [1, 2, 3], {1: 1.2, 2: 0.5}, {1: 0.0, 2: 0.5, 3: 0.0},
        value_bounds=(0.0, 1.0))
    pi = res["partial_identification"]
    assert pi["mean_diff_bounds"] is None, pi
    assert pi["outcome_bounds_violations"] == [1], pi
    assert pi["sign_score_bounds"] is not None, pi
    print("  [ok] 越界观测不伪造均值界，符号界仍保留")


def test_aggregate_missingness_does_not_zero_fill():
    entries = [
        {"status": "complete", "n_missing": 0},
        {"status": "sensitive_within_observed_range", "n_missing": 1,
         "partial_identification": {
             "mean_diff_bounds": None, "mean_sign_identified": None,
             "sign_score_identified": True}},
        {"status": "no_observed_pairs", "n_missing": 3,
         "partial_identification": {
             "mean_diff_bounds": [-1.0, 1.0],
             "mean_sign_identified": False,
             "sign_score_identified": False}},
        {"status": "invalid"},
    ]
    agg = aggregate_missingness(entries)
    assert agg["n_comparisons"] == 4
    assert agg["n_complete"] == 1
    assert agg["n_with_missing"] == 2
    assert agg["n_sensitive"] == 1
    assert agg["status_counts"]["invalid"] == 1
    assert agg["n_manski_mean_bounds"] == 1
    assert agg["n_manski_mean_sign_undetermined"] == 1
    assert agg["n_manski_sign_score_identified"] == 1
    assert agg["n_manski_sign_score_undetermined"] == 1
    print("  [ok] 汇总计数不把不可计算条目算成无缺失")


def _runs():
    def ev(ppl, ctx=256):
        return {"ppl": ppl, "split": "val", "eval_ctx": ctx,
                "eval_iters": 10, "val_windows": 10,
                "val_tokens_evaluated": ctx * 10}

    def ruler(avg):
        by_ctx = {"256": {"n": 20, "em": avg}}
        return {"ctx_lens": [256], "n_samples": 20, "sample_seed": 1234,
                "corpus_source": "local", "max_new": 12, "fwe_max_new": 40,
                "vt_chains": 4, "vt_depth": 4, "amp": True,
                "depth_stratified": True,
                "results": {"niah_synth": by_ctx}}

    return {
        "cand": {1: {"eval": ev(100), "ruler": ruler(0.5)},
                 2: {"eval": ev(110), "ruler": ruler(0.6)}},
        "ref": {1: {"eval": ev(120), "ruler": ruler(0.4)},
                2: {"eval": ev(130), "ruler": ruler(0.5)},
                3: {"eval": ev(140), "ruler": ruler(0.6)}},
    }


def test_pipeline_adapter_reports_missing_run_and_protocol_mismatch():
    runs = _runs()
    runs["ref"][2]["eval"]["eval_ctx"] = 512
    out = _missingness_analysis(runs, [("cand", "ref")], [1, 2, 3])
    ppl = [e for e in out["endpoints"] if e["metric"] == "PPL"][0]
    assert ppl["n_planned"] == 3 and ppl["n_observed"] == 1, ppl
    assert ppl["missing_candidate"] == [3], ppl
    assert ppl["protocol_mismatch"] == 1, ppl
    assert "缺失敏感性" in out["note"]
    assert out["planned_seeds"] == [1, 2, 3]
    print("  [ok] 流水线适配区分缺 run/缺值与协议不一致")


def test_pipeline_manski_bounds_for_ruler_and_ppl():
    out = _missingness_analysis(_runs(), [("cand", "ref")], [1, 2, 3])
    ppl = [e for e in out["endpoints"] if e["metric"] == "PPL"][0]
    ruler = [e for e in out["endpoints"] if e["metric"] == "RULER平均"][0]
    ppl_pi = ppl["partial_identification"]
    ruler_pi = ruler["partial_identification"]
    assert ppl_pi["mean_diff_bounds"] is None, ppl_pi
    assert ppl_pi["sign_score_bounds"] is not None, ppl_pi
    assert ruler_pi["value_bounds"] == [0.0, 1.0], ruler_pi
    assert ruler_pi["mean_diff_bounds"] is not None, ruler_pi
    assert ruler_pi["mean_sign_identified"] is False, ruler_pi
    assert ruler_pi["sign_score_identified"] is True, ruler_pi
    assert out["summary"]["n_manski_mean_bounds"] == 1, out["summary"]
    assert out["summary"]["n_manski_mean_sign_undetermined"] == 1, \
        out["summary"]
    print("  [ok] 流水线对 RULER 给均值界、对 PPL 只给符号界")


def test_render_section_includes_boundaries_and_warning():
    out = _missingness_analysis(_runs(), [("cand", "ref")], [1, 2, 3])
    md = "\n".join(_render_missingness_section(out))
    assert "缺失结果敏感性" in md
    assert "不是 MAR/MCAR 插补" in md
    assert "cand vs ref" in md and "PPL" in md
    assert "significance.json" in md and "missingness.endpoints" in md
    assert "Manski均值界" in md and "Manski符号界" in md
    assert "| 无界 |" in md
    assert "Manski均值方向未定" in md
    print("  [ok] 汇总渲染包含缺失模式、边界声明与机读字段指引")


def test_build_summary_wires_missingness_into_report_and_json():
    tmp = tempfile.mkdtemp()
    try:
        def write(variant, seed, ppl):
            path = os.path.join(tmp, f"eval_{variant}_seed{seed}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"variant": variant, "seed": seed, "ppl": ppl,
                           "val_ppl": ppl, "val_loss": math.log(ppl),
                           "gate_mean": 0.5, "n_params": 100,
                           "n_gate_extra": 0, "split": "val",
                           "eval_ctx": 256, "eval_iters": 10,
                           "val_windows": 10,
                           "val_tokens_evaluated": 2560}, f)
        for seed, ppl in ((1, 100.0), (2, 101.0)):
            write("lega", seed, ppl)
        for seed, ppl in ((1, 102.0), (2, 103.0), (3, 104.0)):
            write("standard", seed, ppl)
        with open(os.path.join(tmp, "repro.json"), "w",
                  encoding="utf-8") as f:
            json.dump({"variants": ["lega", "standard"],
                       "seeds": [1, 2, 3]}, f)
        out = build_summary(tmp)
        md = out["markdown"]
        assert "缺失结果敏感性" in md, md
        assert "缺失(候选/参照/双侧)" in md
        assert out["missingness"]["planned_seeds_source"] == "repro_scope"
        endpoints = out["missingness"]["endpoints"]
        assert len(endpoints) == 2, endpoints
        assert {(e["candidate"], e["reference"]) for e in endpoints} == {
            ("lega", "standard")}, endpoints
        ppl = [e for e in endpoints if e["metric"] == "PPL"][0]
        assert ppl["missing_candidate"] == [3], ppl
        assert ppl.get("insufficient_seeds") is True, ppl
        with open(os.path.join(tmp, "significance.json"), "r",
                  encoding="utf-8") as f:
            payload = json.load(f)
        assert payload["missingness"]["endpoints"][0]["metric"] == "PPL"
        assert "MAR/MCAR" in payload["missingness_note"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] build_summary 落盘并渲染缺失敏感性，计划缺失时标 repro_scope")
