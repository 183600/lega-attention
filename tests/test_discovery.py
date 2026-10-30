"""探索性种子级 FDR 控制（BH/BY）的单元测试。"""
import json
import math
import os
import shutil
import tempfile

from lega.discovery import (benjamini_hochberg, benjamini_yekutieli,
                            collect_seed_level_tests, exploratory_fdr)
from lega.pipeline.report.summary import build_summary


def test_benjamini_hochberg_hand_calculation():
    assert benjamini_hochberg([0.01, 0.04, 0.03]) == [0.03, 0.04, 0.04]
    q = benjamini_hochberg([0.01, 0.01, 0.04])
    assert abs(q[0] - 0.015) < 1e-12
    assert q[0] == q[1]
    assert abs(q[2] - 0.04) < 1e-12
    assert benjamini_hochberg([]) == []
    assert benjamini_hochberg([None, float("nan")]) == [None, None]
    print("  [ok] benjamini_hochberg（手算/并列 p/空输入）")


def test_benjamini_hochberg_invalid_p_do_not_enter_family():
    p = [None, float("nan"), True, "0.02", -0.1, 1.5, [0.1], 0.04]
    q = benjamini_hochberg(p)
    assert q[:7] == [None] * 7
    assert abs(q[7] - 0.04) < 1e-12
    assert benjamini_hochberg(iter([0.01, 0.04, 0.03])) == \
        [0.03, 0.04, 0.04]
    print("  [ok] benjamini_hochberg（非法 p 不占家族/生成器）")


def test_benjamini_yekutieli_sensitivity():
    bh = benjamini_hochberg([0.01, 0.04, 0.03])
    by = benjamini_yekutieli([0.01, 0.04, 0.03])
    harmonic = math.fsum(1.0 / i for i in range(1, 4))
    for a, b in zip(bh, by):
        assert abs(b - min(1.0, a * harmonic)) < 1e-12
    assert benjamini_yekutieli([None, None]) == [None, None]
    print("  [ok] benjamini_yekutieli（H_m 缩放/空输入）")


def test_exploratory_fdr_filters_and_annotations():
    t1 = {"candidate": "a", "reference": "b", "metric": "PPL",
          "p": 0.01, "n": 5, "mean_diff": -1.0, "fdr_source": "main"}
    t2 = {"candidate": "a", "reference": "b", "metric": "RULER",
          "p": 0.20, "n": 5, "mean_diff": 1.0, "fdr_source": "main"}
    t3 = {"candidate": "a", "reference": "b", "metric": "X",
          "p": 0.01, "n": 2}
    t4 = {"candidate": "a", "reference": "b", "metric": "Y",
          "p": None, "n": 5}
    out = exploratory_fdr([t1, t2, t3, t4])
    assert out["available"] is True
    assert out["family_size"] == 2
    assert out["n_tests_total"] == 4
    assert out["n_excluded"] == 2
    assert out["excluded_by_reason"] == {
        "insufficient_seeds": 1, "missing_or_invalid_p": 1}
    assert abs(t1["q_bh"] - 0.02) < 1e-12
    assert abs(t2["q_bh"] - 0.20) < 1e-12
    assert t1["fdr_significant"] is True
    assert t2["fdr_significant"] is False
    assert t3["q_bh"] is None and t4["q_bh"] is None
    assert t3["fdr_excluded_reason"] == "insufficient_seeds"
    assert t4["fdr_excluded_reason"] == "missing_or_invalid_p"
    assert out["n_discoveries"] == 1
    assert out["n_discoveries_by"] == 1
    assert abs(out["expected_false_discoveries_at_fdr"] - 0.05) < 1e-12
    assert out["p_threshold"] == 0.01
    assert abs(out["q_threshold"] - 0.02) < 1e-12
    assert out["n_by_source"] == {"main": 2}
    assert out["n_discoveries_by_source"] == {"main": 1}
    assert out["discoveries"][0]["source"] == "main"
    out2 = exploratory_fdr([t1, t1, t2])
    assert out2["family_size"] == 2
    empty = exploratory_fdr([])
    assert empty["available"] is False
    assert empty["family_size"] == 0
    print("  [ok] exploratory_fdr（过滤/注释/去重/空家族）")


def test_collect_seed_level_tests_sources():
    main = [
        {"metric": "PPL"},
        {"metric": "PPL(log)"},
        {"metric": "RULER平均", "ruler_task": "niah_synth"},
        {"metric": "PPL交互", "interaction": "A×B"},
        {"metric": "Shapley", "shapley_factor": "A"},
    ]
    causal = [
        {"metric": "causal"},
        {"metric": "matched", "matched_control": True},
    ]
    dose = {"tests": [{"metric": "dose"}]}
    length = {
        "ctx_tests": [{"metric": "ctx"}],
        "slope_tests": [{"metric": "slope"}],
        "span_tests": [{"metric": "span"}],
    }
    ngram = {"rows": [{"metric": "PPL vs ngram"}]}
    reparam = {"metrics": {"PPL": {
        "contrasts": [{"metric": "reparam"}],
        "shapley": [{"metric": "reparam shapley"}]}}}
    probe = {"tests": [{"metric": "probe"}]}
    out = collect_seed_level_tests(
        main_tests=main,
        causal_tests=causal,
        matched_ablation_contrasts=[{"metric": "raw-matched"}],
        dose_tests=dose,
        inference_shapley=[{"metric": "inference shapley"}],
        inference_interactions=[{"metric": "inference interaction"}],
        length_analysis=length,
        ngram_comparison=ngram,
        reparameterization=reparam,
        probe_comparison=probe,
    )
    sources = [t["fdr_source"] for t in out]
    assert sources == [
        "main", "ppl_log", "ruler_task", "factorial_interaction",
        "factorial_shapley", "causal_lesion", "causal_matched_lesion",
        "causal_matched_contrast", "dose_response", "inference_shapley",
        "inference_interaction", "ruler_length_ctx", "ruler_length_slope",
        "ruler_length_span", "ngram_baseline",
        "reparameterization_contrast", "reparameterization_shapley",
        "probe_baseline",
    ]
    assert collect_seed_level_tests() == []
    assert collect_seed_level_tests(main_tests="bad", dose_tests=[]) == []
    print("  [ok] collect_seed_level_tests（来源标注/坏输入）")


def test_build_summary_wires_exploratory_fdr_into_report_and_json():
    tmp = tempfile.mkdtemp()
    try:
        for seed, (std, gate) in enumerate(
                ((10.0, 9.0), (11.0, 10.0), (12.0, 11.0)), start=1):
            for variant, ppl in (("standard", std), ("gate_only", gate)):
                with open(os.path.join(
                        tmp, f"eval_{variant}_seed{seed}.json"),
                        "w", encoding="utf-8") as f:
                    json.dump({
                        "variant": variant, "seed": seed,
                        "val_ppl": ppl, "val_loss": math.log(ppl),
                        "gate_mean": 0.5, "n_params": 100,
                        "n_gate_extra": 0, "split": "val",
                    }, f, ensure_ascii=False)
        out = build_summary(tmp)
        fdr = out.get("exploratory_fdr")
        assert isinstance(fdr, dict) and fdr.get("available") is True
        assert fdr["family_size"] >= 1
        assert "探索性发现的 FDR 控制" in out["markdown"]
        with open(os.path.join(tmp, "significance.json"),
                  encoding="utf-8") as f:
            sig = json.load(f)
        assert sig["exploratory_fdr"]["family_size"] == fdr["family_size"]
        pair = next(
            t for t in sig["pairs"]
            if t.get("candidate") == "gate_only"
            and t.get("reference") == "standard"
            and t.get("metric") == "PPL")
        assert pair.get("q_bh") is not None
        assert pair.get("fdr_family_size") == fdr["family_size"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] build_summary（探索性 FDR 进入 Markdown 与 "
          "significance.json）")
