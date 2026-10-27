"""RULER/PPL 抽样配对、bootstrap 与推理期证据测试。"""
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from unittest import mock

from lega.data import CACHE_SCHEMA
from tests.pipeline_helpers import _mk, _write_json
from run_all import (ALPHA, AP_DIFF_METRIC, MIN_SEEDS, _ablation_coverage,
                      _ap_diff_tests, _apply_holm, _auroc_diff_tests, _cmd_env, _effect_status,
                      _enrich_paired, _eval_matches_window_spec,
                      _factorial_interaction_tests, _FACTORIAL_CONTRAST_CELLS,
                      _factorial_protocol_ok, _inference_shapley_tests,
                      INFERENCE_INTERACTION_METRIC, INFERENCE_SHAPLEY_METRIC,
                      PARAM_CONTROL_VARIANT,
                      PARAM_CONTROL_PAIR,
                      _int_or_zero, _lat_noise_pct,
                      _neural_eval_window_spec, _noise_floor_pct,
                      _noise_verdict_note, _loo_stability_str, _num_or_none,
                      _rate_or_none,
                      _seed_winloss_str, _paired_comparisons, _ppl_ci_rel_pct,
                      _log_ppl_of, _ppl_of, _probe_oof, _rel_pct,
                      _render_baseline_sections, _result_files,
                      _render_ruler_length,
                      _render_inference_shapley,
                      _render_parameter_matched_control,
                      _ruler_avg_micro, _ruler_avg_of, _ruler_cell_signature,
                      _ruler_depth_gap, _ruler_depth_means,
                      _ruler_fwe_recall, _ruler_min_cell_n, _ruler_protocol,
                      _ruler_task_means,
                      _render_registration_integrity, _resolve_jobs, _stat,
                      _seed_key, _strict_seed_int, _parse_seed_list,
                      apply_budget_gate, apply_integrity_gate, apply_scope_gate,
                      AUROC_DIFF_METRIC, AUROC_DIFF_REFERENCE,
                      build_comparison_pairs, build_summary,
                      _fmt_eq_margin, _fmt_eq_value,
                      _fmt_mean_std, _fmt_signed,
                      changed_gate_factors,
                      component_evidence, cross_variant_badcase_map,
                      COMPONENT_TESTS, GATE_FACTORS,
                      main as run_all_main,
                      data_protocol_preflight, data_report_binding_issue,
                      evaluate_registered_hypotheses,
                      load_registered_hypotheses, ngram_baseline_comparison,
                      parse_variants, registration_scope_check,
                      abort_ledger_on_failure, record_aborted_sanity,
                      mark_repro_failed, check_sanity_results,
                      _snapshot_existing_runs,
                      resolve_ngram_split, run_many,
                      single_factor_audit, single_factor_change,
                      SINGLE_FACTOR_CONTRASTS, SINGLE_FACTOR_PAIRS,
                      tradeoff_verdict, verify_controlled_setup)



def test_ruler_sample_paired_mcnemar_counts_and_exact_p():
    """固定样例上的配对：只消费 discordant 对，精确 McNemar 与计数一致。"""
    from run_all import (_mcnemar_exact_p, _ruler_sample_outcomes,
                         _ruler_sample_paired_tests)

    # 已知精确值：n=5 全偏向候选 -> 双侧 p=2/32=0.0625；n=6 -> 0.03125
    assert abs(_mcnemar_exact_p(5, 0) - 0.0625) < 1e-12
    assert abs(_mcnemar_exact_p(0, 6) - 0.03125) < 1e-12
    assert _mcnemar_exact_p(0, 0) == 1.0

    def cell(ems):
        return {"em": sum(ems) / len(ems), "n": len(ems),
                "samples": [{"sample_id": f"s{i}", "em": float(e)}
                            for i, e in enumerate(ems)]}

    def rul(samples, **kw):
        base = {
            "ctx_lens": [256], "n_samples": len(samples), "sample_seed": 1234,
            "corpus_source": "val", "max_new": 12, "fwe_max_new": 44,
            "vt_chains": 4, "vt_depth": 4,
            "results": {"niah": {"256": cell(samples)}},
        }
        base.update(kw)
        return {"ruler": base}

    runs = {
        "lega": {7: rul([1, 1, 1, 1, 0, 0, 0, 0]),
                 8: rul([1, 1, 1, 0, 0, 0, 0, 0])},
        "standard": {7: rul([0, 0, 0, 0, 0, 0, 0, 0]),
                     8: rul([0, 0, 0, 0, 0, 0, 0, 0])},
    }
    d = runs["lega"][7]["ruler"]
    assert _ruler_sample_outcomes(d)["s0"] == 1.0
    out = _ruler_sample_paired_tests(runs, [("lega", "standard")])
    assert len(out) == 1, out
    t = out[0]
    # seed 7: 4 个候选独对；seed 8: 3 个；合计 7 个 discordant，全偏向候选
    assert t["candidate_only"] == 7 and t["reference_only"] == 0, t
    assert t["n_samples"] == 16 and t["n_seeds"] == 2, t
    assert abs(t["mean_diff"] - 7 / 16) < 1e-12, t
    assert abs(t["p"] - 2 / (2 ** 7)) < 1e-12, t
    assert t["n_seed_wins"] == 2 and t["n_seed_losses"] == 0, t
    assert t["verdict"].startswith("样例级显著优于"), t

    # 旧产物没有逐样例字段：返回空结果而不是伪造 0 差异
    legacy = {"lega": {7: {"ruler": {"results": {"niah": {
        "256": {"em": 0.5, "n": 8}}}}}},
              "standard": {7: {"ruler": {"results": {"niah": {
                  "256": {"em": 0.5, "n": 8}}}}}}}
    assert _ruler_sample_paired_tests(legacy, [("lega", "standard")]) == []
    print("  [ok] RULER 样例级 McNemar：discordant 计数、精确 p、旧产物跳过")


def test_ruler_sample_cluster_test_avoids_seed_pseudo_replication():
    """同一样例在多个种子下重复出现时，池化 McNemar 会伪造独立性。"""
    from run_all import _ruler_sample_paired_tests

    def cell(ems):
        return {"em": sum(ems) / len(ems), "n": len(ems),
                "samples": [{"sample_id": f"s{i}", "em": float(e)}
                            for i, e in enumerate(ems)]}

    def rul(ems):
        return {"ruler": {
            "ctx_lens": [256], "n_samples": len(ems), "sample_seed": 1234,
            "corpus_source": "val", "max_new": 12, "fwe_max_new": 44,
            "vt_chains": 4, "vt_depth": 4,
            "results": {"niah": {"256": cell(ems)}}}}

    # 10 个训练种子反复评测同 2 个样例：池化后有 20 个“discordant 对”，
    # 但独立样例只有 2 个。聚类检验必须给出 n=2、p=0.5（不能显著）。
    runs = {
        "lega": {seed: rul([1, 1]) for seed in range(1, 11)},
        "standard": {seed: rul([0, 0]) for seed in range(1, 11)},
    }
    t = _ruler_sample_paired_tests(runs, [("lega", "standard")])[0]
    assert t["candidate_only"] == 20 and t["n_samples"] == 20, t
    assert t["p"] < 0.001, t
    assert t["cluster_n"] == 2, t
    assert abs(t["cluster_mean_diff"] - 1.0) < 1e-12, t
    assert t["cluster_p"] == 0.5, t
    assert t["cluster_verdict"].startswith("样例聚类后无显著差异"), t
    print("  [ok] RULER 样例聚类检验：同一样例跨种子重复不伪造独立性")


def test_ppl_window_paired_tests_counts_bootstrap_and_protocol():
    """PPL 窗口级配对：同窗口序号逐点相减，分层 bootstrap 与协议门槛。"""
    from run_all import _eval_window_losses, _ppl_window_paired_tests

    def ev(losses, idx, **kw):
        base = {"split": "test", "eval_ctx": 10, "eval_iters": len(idx),
                "val_windows": max(9, len(idx)),
                "val_tokens_evaluated": 10 * len(idx),
                "window_losses": losses, "window_indices": idx}
        base.update(kw)
        return {"eval": base}

    idx = list(range(0, 30, 3))  # 10 个窗口，越过最低窗口门槛
    runs = {
        "lega": {7: ev([1.0] * 10, idx), 8: ev([1.5] * 10, idx)},
        "standard": {7: ev([2.0] * 10, idx), 8: ev([2.0] * 10, idx)},
    }
    assert _eval_window_losses(runs["lega"][7]["eval"]) == \
        {i: 1.0 for i in idx}
    # 交叉熵不可能为负：坏记录整表按不可用处理，不能污染 avg NLL 差。
    assert _eval_window_losses({"window_losses": [1.0, -0.5],
                                "window_indices": [0, 1]}) is None
    # 窗口序号必须是唯一的非负下标；负值/重复都是损坏记录。
    assert _eval_window_losses({"window_losses": [1.0, 2.0],
                                "window_indices": [0, 0]}) is None
    assert _eval_window_losses({"window_losses": [1.0, 2.0],
                                "window_indices": [-1, 1]}) is None
    out = _ppl_window_paired_tests(runs, [("lega", "standard")],
                                   n_resamples=200)
    assert len(out) == 1, out
    t = out[0]
    assert t["n_seeds"] == 2 and t["n_windows_min"] == 10, t
    assert abs(t["mean_diff"] - (-0.75)) < 1e-12, t  # seed7 -1, seed8 -0.5
    assert t["ppl_ratio"] < 1.0, t
    assert t["mean_diff_ci95"][1] < 0, t
    assert t["block_mean_diff_ci95"][1] < 0, t
    assert t["block_length_min"] >= 1 and t["block_length_max"] >= 1, t
    assert "mean_lag1_autocorr" in t, t
    assert t["p_adj"] is not None and t["p_adj"] < 0.05, t
    assert t["verdict"].startswith("窗口级显著优于"), t

    # 单窗口差值本身有限但同号相加会溢出：per_seed 明细也不能写 inf。
    huge_idx = list(range(0, 30, 3))
    huge_runs = {
        "lega": {7: ev([1.5e308] * 10, huge_idx)},
        "standard": {7: ev([0.0] * 10, huge_idx)},
    }
    huge_t = _ppl_window_paired_tests(huge_runs, [("lega", "standard")],
                                      n_resamples=50)[0]
    assert huge_t["per_seed"][0]["mean_diff"] == 1.5e308, huge_t
    assert math.isfinite(huge_t["mean_diff"]), huge_t
    # 窗口序号不一致 = 不是同一批评测单元，整对剔除而不是按位置强配
    runs2 = {
        "lega": {7: ev([1.0, 1.0], [0, 4])},
        "standard": {7: ev([2.0, 2.0], [0, 5])},
    }
    assert _ppl_window_paired_tests(runs2, [("lega", "standard")]) == []
    # 协议不一致同样跳过
    runs3 = {
        "lega": {7: ev([1.0, 1.0], [0, 4])},
        "standard": {7: ev([2.0, 2.0], [0, 4], eval_ctx=20)},
    }
    assert _ppl_window_paired_tests(runs3, [("lega", "standard")]) == []
    # 一侧缺协议字段同样跳过，不能只比较交集后静默配对
    runs3b = {
        "lega": {7: ev([1.0, 1.0], [0, 4])},
        "standard": {7: ev([2.0, 2.0], [0, 4], val_windows=None)},
    }
    assert _ppl_window_paired_tests(runs3b, [("lega", "standard")]) == []

    # 1 个种子 × 3 个窗口：重采样自由度不足，只能报区间，不能宣称显著
    thin = {
        "lega": {7: ev([1.0, 1.0, 1.0], [0, 4, 8])},
        "standard": {7: ev([2.0, 2.0, 2.0], [0, 4, 8])},
    }
    tt = _ppl_window_paired_tests(thin, [("lega", "standard")],
                                  n_resamples=50)[0]
    assert tt["enough_windows"] is False
    assert tt["verdict"].startswith("证据不足"), tt
    assert "显著" not in tt["verdict"], tt
    print("  [ok] PPL 窗口级配对：逐窗口相减、分层 bootstrap、协议/序号/"
          "最小窗口门槛")


def test_build_summary_renders_both_eval_sampling_sections():
    """窗口级 bootstrap 与样例级 McNemar 都要进入汇总页/JSON，且与种子级并列。"""
    from run_all import build_summary

    tmp = tempfile.mkdtemp()
    try:
        def ev(v, losses, ppl):
            return {"variant": v, "seed": 1, "val_ppl": ppl, "val_loss": 1.0,
                    "split": "val", "eval_ctx": 10, "eval_iters": 3,
                    "eval_iters_requested": 3, "val_windows": 3,
                    "val_tokens_evaluated": 30, "n_params": 100,
                    "window_losses": losses, "window_indices": [0, 3, 6]}

        _write_json(os.path.join(tmp, "eval_standard.json"),
                    ev("standard", [2.0, 2.0, 2.0], 10.0))
        _write_json(os.path.join(tmp, "eval_lega.json"),
                    ev("lega", [1.0, 1.0, 1.0], 9.0))

        def rul(v, ems):
            cell = {"em": sum(ems) / len(ems), "n": len(ems),
                    "samples": [{"sample_id": f"s{i}", "em": float(e)}
                                for i, e in enumerate(ems)]}
            return {"variant": v, "seed": 1,
                    "ctx_lens": [256], "n_samples": len(ems),
                    "sample_seed": 7, "corpus_source": "val", "max_new": 12,
                    "fwe_max_new": 44, "vt_chains": 4, "vt_depth": 4,
                    "results": {"niah": {"256": cell}}}

        _write_json(os.path.join(tmp, "ruler_standard.json"),
                    rul("standard", [0, 0, 0, 0]))
        _write_json(os.path.join(tmp, "ruler_lega.json"),
                    rul("lega", [1, 1, 1, 1]))
        out = build_summary(tmp)
        md = out["markdown"]
        assert "### PPL 窗口级配对" in md and "PPL比" in md, md[-3000:]
        assert "### RULER 样例级配对" in md and "候选独对" in md, md[-3000:]
        sig = json.load(open(os.path.join(tmp, "significance.json"),
                             encoding="utf-8"))
        assert len(sig["ppl_window_pairs"]) == 1, sig.keys()
        assert len(sig["ruler_sample_pairs"]) == 1, sig.keys()
        assert len(out["ppl_window_tests"]) == 1
        assert len(out["ruler_sample_tests"]) == 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] build_summary 并列渲染窗口级/样例级评测抽样证据")


def test_ruler_macro_bootstrap_preserves_cell_weights_and_is_deterministic():
    """macro 的 bootstrap 必须在单元格内重采样：小格不能被大格淹没。"""
    from run_all import _ruler_macro_bootstrap_ci

    # 小格 2 个样例全 +1，大格 8 个样例全 0：macro = (1+0)/2 = 0.5。
    # 若错误地池化全部样例再重采样，会得到约 0.2，而不是恒定的 0.5。
    per_seed = [{("niah", "256"): [1.0, 1.0],
                 ("niah", "512"): [0.0] * 8}]
    ci = _ruler_macro_bootstrap_ci(per_seed, n_boot=200)
    assert ci is not None
    assert abs(ci[0] - 0.5) < 1e-12 and abs(ci[1] - 0.5) < 1e-12, ci
    assert ci == _ruler_macro_bootstrap_ci(per_seed, n_boot=200), "必须可复现"
    assert _ruler_macro_bootstrap_ci([]) is None
    # bootstrap 次数/随机种子是严格整数身份：0/负数/bool/小数不能经
    # int()/max(1,·) 静默变成另一次预算。
    for bad in (0, -1, True, 2.5):
        try:
            _ruler_macro_bootstrap_ci(per_seed, n_boot=bad)
            raised = False
        except ValueError:
            raised = True
        assert raised, bad
    try:
        _ruler_macro_bootstrap_ci(per_seed, rng_seed=-1)
        raised = False
    except ValueError:
        raised = True
    assert raised
    print("  [ok] RULER macro bootstrap：单元格等权、确定性、空输入安全")


def test_ruler_macro_absolute_ci_matches_task_equal_estimand():
    """绝对 macro 区间必须匹配任务等权点估计，而不是单元格等权。"""
    from run_all import _ruler_avg_of, _ruler_macro_absolute_ci

    def cell(ems, prefix):
        return {"em": sum(ems) / len(ems), "n": len(ems),
                "samples": [{"sample_id": f"{prefix}{i}", "em": float(e)}
                            for i, e in enumerate(ems)]}

    # 任务 a 只有 1 格全对；任务 b 有 2 格全错。任务等权 macro=0.5；
    # 若错误地按单元格等权会是 (1+0+0)/3≈0.33。
    payload = {"results": {
        "a": {"1": cell([1, 1], "a")},
        "b": {"1": cell([0] * 8, "b1"), "2": cell([0] * 8, "b2")}}}
    assert abs(_ruler_avg_of(payload) - 0.5) < 1e-12
    r = _ruler_macro_absolute_ci([payload, payload], n_boot=200)
    assert r is not None and r["n_seeds"] == 2 and r["n_cells"] == 6
    assert abs(r["ci95"][0] - 0.5) < 1e-12
    assert abs(r["ci95"][1] - 0.5) < 1e-12
    assert r == _ruler_macro_absolute_ci([payload, payload], n_boot=200)
    for bad in (0, -1, True, 2.5):
        try:
            _ruler_macro_absolute_ci([payload, payload], n_boot=bad)
            raised = False
        except ValueError:
            raised = True
        assert raised, bad
    # 旧产物无逐样例记录：不给区间，也不伪造 0 宽区间
    legacy = {"results": {"a": {"1": {"em": 0.5, "n": 2}}}}
    assert _ruler_macro_absolute_ci([legacy]) is None
    # 多种子混合：只要有一个 payload 缺逐样例记录或含空单元格，就不能
    # 丢掉该种子后仍声称区间覆盖全部观测种子。
    assert _ruler_macro_absolute_ci([payload, legacy]) is None
    empty_cell = {"results": {"a": {"1": cell([1, 1], "a"),
                                    "2": {"em": 0.5, "n": 0, "samples": []}}}}
    assert _ruler_macro_absolute_ci([payload, empty_cell]) is None
    print("  [ok] RULER 绝对 macro CI：任务等权、确定性、旧产物不伪造")


def test_ruler_sample_macro_micro_estimands_diverge():
    """样例等权 micro 与单元格等权 macro 可给出相反结论，必须同时报告。"""
    from run_all import _ruler_avg_of, _ruler_sample_paired_tests

    def cell(ems, prefix):
        return {"em": sum(ems) / len(ems), "n": len(ems),
                "samples": [{"sample_id": f"{prefix}{i}", "em": float(e)}
                            for i, e in enumerate(ems)]}

    small = cell([1, 1], "a")                  # 小格 +1，2 个样例
    big = cell([0] * 6, "b")                   # 大格 −1，6 个样例
    ref = cell([0, 0], "a"), cell([1] * 6, "b")

    def rul(v, cells):
        return {"ruler": {"variant": v, "seed": 1,
                          "ctx_lens": [256, 512], "n_samples": 8,
                          "sample_seed": 7, "corpus_source": "val",
                          "max_new": 12, "fwe_max_new": 44,
                          "vt_chains": 4, "vt_depth": 4,
                          "results": {"niah": {"256": cells[0],
                                               "512": cells[1]}}}}

    runs = {"lega": {1: rul("lega", (small, big))},
            "standard": {1: rul("standard", ref)}}
    t = _ruler_sample_paired_tests(runs, [("lega", "standard")])[0]
    # micro = (2 胜 − 6 负)/8 = −0.5；macro = (+1 −1)/2 = 0
    assert abs(t["mean_diff"] - (-0.5)) < 1e-12, t
    assert abs(t["macro_mean_diff"] - 0.0) < 1e-12, t
    assert t["n_cells"] == 2, t
    assert t["per_seed"][0]["n_cells"] == 2
    mci = t["macro_mean_diff_ci95"]
    assert mci is not None and mci[0] <= t["macro_mean_diff"] <= mci[1], t
    seed_macro = (_ruler_avg_of(runs["lega"][1]["ruler"])
                  - _ruler_avg_of(runs["standard"][1]["ruler"]))
    assert abs(seed_macro - t["macro_mean_diff"]) < 1e-12
    assert t["estimand_conflict"] is True
    assert "micro/macro 方向不一致" in t["verdict"], t
    print("  [ok] RULER 样例级效应量：micro 与 macro 口径分歧显式并列并告警")


def test_window_bootstrap_survives_huge_finite_window_diffs():
    """逐窗口差值本身有限但相加会溢出时，bootstrap 均值/区间仍需有限。"""
    from run_all import _window_bootstrap_stats
    huge = [[1e308, 1e308], [1e308, 1e308]]
    st = _window_bootstrap_stats(huge, n_resamples=50, seed=0)
    assert st is not None, st
    assert st["mean_diff"] == 1e308, st
    assert all(math.isfinite(x) for x in st["mean_diff_ci95"]), st
    assert all(math.isfinite(x) for x in st["block_mean_diff_ci95"]), st
    assert math.isfinite(st["p"]) and 0.0 <= st["p"] <= 1.0, st
    # 常数序列没有 lag-1 相关；巨大常数不能因溢出变成 NaN/警告。
    assert st["mean_lag1_autocorr"] is None, st
    # 正负巨大值抵消时均值应为 0，而不是 NaN。
    mixed = [[1e308, -1e308], [1e308, -1e308]]
    st2 = _window_bootstrap_stats(mixed, n_resamples=50, seed=0)
    assert st2 is not None and st2["mean_diff"] == 0.0, st2
    assert st2["p"] == 1.0, st2
    # 直接传入非有限窗口差时按缺失过滤；全部无效则返回 None。
    assert _window_bootstrap_stats([[float("nan")]]) is None
    # 重采样预算/种子必须是严格整数：0/负数会让空分位数崩溃，bool/小数
    # 不能经 int() 截断成另一个与声明不一致的预算。
    for bad in (0, -1, True, 2.5):
        try:
            _window_bootstrap_stats(huge, n_resamples=bad)
            raised = False
        except ValueError:
            raised = True
        assert raised, bad
    # 整数字符串与全项目 strict_int 口径一致，规范化为整数。
    assert _window_bootstrap_stats(huge, n_resamples="2") is not None
    try:
        _window_bootstrap_stats(huge, n_resamples=10, seed=-1)
        raised = False
    except ValueError:
        raised = True
    assert raised
    print("  [ok] 窗口分层 bootstrap：巨大有限差/正负抵消不溢出，"
          "非有限差过滤，重采样预算/种子严格校验")


def test_ruler_micro_average_survives_huge_finite_weights():
    """两个可表示但相加会溢出的样本计数仍应给出有限加权均值。"""
    both_one = {"results": {"niah": {
        "256": {"n": 1e308, "em": 1.0},
        "512": {"n": 1e308, "em": 1.0}}}}
    assert _ruler_avg_micro(both_one) == 1.0
    half = {"results": {"niah": {
        "256": {"n": 1e308, "em": 1.0},
        "512": {"n": 1e308, "em": 0.0}}}}
    assert abs(_ruler_avg_micro(half) - 0.5) < 1e-12
    # 极大权重与小权重混合时仍按 n 加权，不能被归一化改写成等权。
    mixed = {"results": {"niah": {
        "256": {"n": 1e308, "em": 1.0},
        "512": {"n": 1, "em": 0.0}}}}
    assert abs(_ruler_avg_micro(mixed) - 1.0) < 1e-12
    print("  [ok] RULER micro 平均：超大有限样本计数不再溢出成 NaN")


def test_log_ppl_ratio_unrepresentable_is_flagged_not_crash():
    """exp(log-PPL 差) 超出 float 上限时必须标缺失，不能中断汇总。"""
    with tempfile.TemporaryDirectory() as tmp:
        for seed in (1, 2, 3):
            for variant, ppl in (("lega", 1e300), ("standard", 1e-300)):
                _write_json(os.path.join(tmp, f"eval_{variant}_seed{seed}.json"), {
                    "variant": variant, "seed": seed, "val_ppl": ppl,
                    "val_loss": math.log(ppl), "gate_mean": 0.5,
                    "n_params": 100, "n_gate_extra": 0, "split": "val",
                    "eval_ctx": 16, "eval_iters": 4, "val_windows": 100,
                    "val_tokens_evaluated": 64})
        out = build_summary(tmp)
        log_tests = [t for t in out["tests"]
                     if t.get("metric") == "PPL(log)"]
        assert log_tests, out["tests"]
        t = log_tests[0]
        assert t["ppl_ratio"] is None, t
        assert t["ppl_ratio_ci95"] is None, t
        assert t["ppl_delta_pct"] is None, t
        assert t["ppl_ratio_overflow"] is True, t
        assert "PPL 比值尺度敏感性" in out["markdown"]
    print("  [ok] 汇总 log-PPL：不可表示的几何均值比标缺失而非崩溃")


def test_log_ppl_delta_pct_overflow_is_dropped():
    """exp(Δ) 可表示但 (ratio−1)×100 溢出时，派生百分比必须为 None。"""
    with tempfile.TemporaryDirectory() as tmp:
        for seed in (1, 2, 3):
            for variant, ppl in (("lega", 1e154), ("standard", 1e-154)):
                _write_json(os.path.join(tmp, f"eval_{variant}_seed{seed}.json"), {
                    "variant": variant, "seed": seed, "val_ppl": ppl,
                    "val_loss": math.log(ppl), "gate_mean": 0.5,
                    "n_params": 100, "n_gate_extra": 0, "split": "val",
                    "eval_ctx": 16, "eval_iters": 4, "val_windows": 100,
                    "val_tokens_evaluated": 64})
        out = build_summary(tmp)
        t = next(t for t in out["tests"]
                 if t.get("metric") == "PPL(log)")
        assert t["ppl_ratio"] is not None and math.isfinite(t["ppl_ratio"])
        assert t["ppl_delta_pct"] is None, t
    print("  [ok] 汇总 log-PPL：派生百分比换算溢出按缺失处理")


def test_factorial_protocol_handles_unhashable_split_value():
    """eval.json 的 split 被手改成数组/对象时，交互检验不能因 set() 崩溃。"""
    from run_all import _FACTORIAL_CONTRAST_CELLS
    maps = {
        cell: {1: {"eval": {"split": [], "eval_ctx": 16, "eval_iters": 4,
                            "val_windows": 100, "val_tokens_evaluated": 64}}}
        for cell in _FACTORIAL_CONTRAST_CELLS
    }
    # 全部相同但不可哈希：协议比较应给出布尔结果，而不是抛 TypeError。
    assert _factorial_protocol_ok(maps, 1, "PPL") is True
    # 与其他 split（字符串）不一致时必须判为不可配对。
    maps["ABC"][1]["eval"]["split"] = "val"
    assert _factorial_protocol_ok(maps, 1, "PPL") is False
    # 一格缺协议字段不是“共同字段一致”，不能继续做交互比较。
    maps["ABC"][1]["eval"]["split"] = []
    del maps["ABC"][1]["eval"]["eval_ctx"]
    assert _factorial_protocol_ok(maps, 1, "PPL") is False
    print("  [ok] 全因子协议核对：不可哈希 split/缺字段不再让汇总崩溃或误配")


def test_parameter_matched_control_and_inference_shapley_wiring():
    """参数匹配安慰剂与推理期 Shapley 必须真正进入汇总/显著性 JSON。

    仅有 lega_placebo 变体名不够：必须 (1) 生成 lega vs placebo 的配对检验；
    (2) 汇总页显式区分“证据特定性”与“容量/优化红利”；(3) 缺推理期全因子
    数据时如实提示，而不是伪造 8 格或把缺失当 0。
    """
    tmp = tempfile.mkdtemp()
    try:
        seeds = [1, 2, 3]
        base = {"standard": 10.0, "qgate": 9.8, "lega": 9.4,
                "lega_placebo": 9.45}
        for v, ppl in base.items():
            for s in seeds:
                _write_json(os.path.join(tmp, f"eval_{v}_seed{s}.json"), {
                    "variant": v, "seed": s, "val_ppl": ppl + 0.01 * s,
                    "val_loss": math.log(ppl), "n_params": 100,
                    "n_gate_extra": 0, "split": "val"})
        reg = {"hypotheses": []}
        intact = {"status": "intact", "confirmatory_valid": True,
                  "problems": [], "registry_sha256": "a" * 64,
                  "n_hypotheses": 0, "baseline_document_sha256": "b" * 64,
                  "chain_length": 0}
        with mock.patch("lega.pipeline.report.summary.load_registered_hypotheses",
                        return_value=reg), \
                mock.patch("lega.pipeline.report.summary.verify_registration",
                           return_value=intact):
            out = build_summary(tmp)
        md = out["markdown"]
        assert "参数匹配安慰剂对照" in md
        assert "lega_placebo" in md and "容量/优化红利" in md
        assert "推理期全因子 Shapley" in md
        assert "无法计算推理期 Shapley" in md  # 无 ablation 产物时如实提示
        sig = json.load(open(os.path.join(tmp, "significance.json"),
                             encoding="utf-8"))
        pmc = sig["parameter_matched_control"]
        assert pmc["available"] is True
        assert tuple(pmc["pair"]) == PARAM_CONTROL_PAIR
        assert pmc["test"] is not None
        assert pmc["test"]["capacity_control"] is True
        assert pmc["capacity_only_test"] is not None
        assert pmc["capacity_only_test"]["candidate"] == PARAM_CONTROL_VARIANT
        assert pmc["capacity_only_test"]["reference"] == "qgate"
        assert "容量本身" in md
        assert sig["inference_factorial_shapley"] == []
        assert any(t["candidate"] == "lega"
                   and t["reference"] == "lega_placebo"
                   and t["metric"] == "PPL" for t in sig["pairs"])
        pairs = build_comparison_pairs(
            {"lega": {1: {}}, "lega_placebo": {1: {}}, "standard": {1: {}}})
        assert ("lega", "lega_placebo") in pairs
        # 没运行 placebo 时不得声称“容量已排除”。
        assert "未运行 `lega_placebo`" in "\n".join(
            _render_parameter_matched_control(None))
        assert "无法计算推理期 Shapley" in "\n".join(
            _render_inference_shapley([]))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] 参数匹配安慰剂/推理期 Shapley 进入汇总与 significance.json")


def test_inference_factorial_interactions_enter_summary_and_significance():
    """推理期 2^3 交互必须与 Shapley 一起写入汇总页与 significance.json。

    只做 Shapley 会把交互平均分摊到因子上；同一批 8 格还必须给出
    A×B/A×C/B×C 的独立交互对比，且缺格时如实提示，不能静默省略。
    """
    from lega.ablate import INFERENCE_FACTORIAL_CELLS
    tmp = tempfile.mkdtemp()
    try:
        seeds = [1, 2, 3]
        _write_json(os.path.join(tmp, "repro.json"), {
            "variants": ["lega"], "seeds": seeds, "smoke": False,
            "ablation": {"executed": True, "variants": ["lega"]}})
        for s in seeds:
            _write_json(os.path.join(tmp, f"eval_lega_seed{s}.json"), {
                "variant": "lega", "seed": s, "split": "val",
                "val_ppl": 17.0, "val_loss": math.log(17.0),
                "n_params": 100, "n_gate_extra": 4})
            # 可加基线 + 纯 A×B 交互：AB/ABC 同时多 2，其余六格保持可加。
            vals = {"none": 10.0, "A": 11.0, "B": 12.0, "C": 14.0,
                    "AB": 15.0, "AC": 15.0, "BC": 16.0, "ABC": 19.0}
            vals = {c: v + 0.01 * s for c, v in vals.items()}
            recs = [{"lesion": lesion, "applied": True,
                     "window_indices": [0, 1, 2],
                     "val_ppl": float(vals[cell])}
                    for cell, lesion in INFERENCE_FACTORIAL_CELLS.items()
                    if lesion is not None]
            _write_json(os.path.join(tmp, f"ablation_lega_seed{s}.json"), {
                "variant": "lega", "seed": s,
                "window_indices": [0, 1, 2],
                "baseline": {"val_ppl": float(vals["ABC"])},
                "lesions": recs})
        out = build_summary(tmp)
        md = out["markdown"]
        assert "推理期全因子交互效应" in md
        assert "A×B" in md and "符号一致" in md
        assert len(out["inference_interactions"]) == 3
        sig = json.load(open(os.path.join(tmp, "significance.json"),
                             encoding="utf-8"))
        got = {t["interaction"]: t
               for t in sig["inference_factorial_interactions"]}
        assert set(got) == {"A×B", "A×C", "B×C"}
        assert abs(got["A×B"]["mean_diff"] - 2.0) < 1e-9, got["A×B"]
        assert abs(got["A×C"]["mean_diff"]) < 1e-9, got["A×C"]
        assert abs(got["B×C"]["mean_diff"]) < 1e-9, got["B×C"]
        assert all(t["metric"] == INFERENCE_INTERACTION_METRIC
                   for t in got.values())
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] 推理期交互进入汇总页与 significance.json，纯 A×B 归属正确")


def test_summary_rendering_tolerates_missing_effect_mean():
    """交互/Shapley/病灶表不能假设每个检验都有有限均值。

    极端有限观测可让交互对比的和溢出，_paired_diffs 会剔除不可表示的配对，
    于是 n=0、mean_diff=None；渲染层若直接 ``f"{mean:+.4f}"`` 会让整份
    汇总 TypeError。缺失均值必须显示 '-'，而不是让报告崩溃。
    """
    assert _fmt_signed(None) == "-"
    assert _fmt_signed(float("nan")) == "-"
    assert _fmt_signed(float("inf")) == "-"
    assert _fmt_signed(0.0) == "+0.0000"
    assert _fmt_mean_std({"mean_diff": None, "std_diff": None}) == "-"
    assert _fmt_mean_std({"mean_diff": 1.5, "std_diff": None}) == "+1.5000"
    with tempfile.TemporaryDirectory() as tmp:
        for v in ("standard", "lega"):
            _write_json(os.path.join(tmp, f"eval_{v}_seed1.json"), {
                "variant": v, "seed": 1, "split": "val",
                "val_ppl": 10.0, "val_loss": math.log(10.0),
                "n_params": 100, "n_gate_extra": 4,
            })
        bad_inter = [{
            "interaction": "A×B", "metric": "PPL交互",
            "mean_diff": None, "std_diff": None, "n": 0,
            "p_adj": None, "verdict": "无法检验",
        }]
        with mock.patch("lega.pipeline.report.summary._factorial_interaction_tests",
                        return_value=bad_inter), \
                mock.patch("lega.pipeline.report.summary._factorial_shapley_tests",
                           return_value=[]), \
                mock.patch("lega.pipeline.report.summary._ablation_lesion_records",
                           return_value=[]), \
                mock.patch("lega.pipeline.report.summary._inference_shapley_tests",
                           return_value=[]):
            out = build_summary(tmp)
        md = out["markdown"]
        assert "A×B" in md and "无法检验" in md
    print("  [ok] 汇总渲染：缺失/非有限均值显示 '-'，交互表不再 TypeError")


def test_equivalence_rendering_handles_scaled_overflow():
    """有限 TOST 值换算后仍可能溢出：报告不能出现 inf/nan。"""
    huge = 1e308
    for kind, eq in (
        ("relative", {"kind": "relative"}),
        ("log_ratio", {"kind": "log_ratio"}),
        ("scale", {"kind": "absolute", "display_scale": 100.0,
                   "display_unit": "pp"}),
    ):
        assert _fmt_eq_value(huge, eq) == "溢出", kind
        assert _fmt_eq_margin(dict(eq, margin=huge)) == "±溢出", kind
    assert _fmt_eq_value(float("nan"), {"kind": "relative"}) == "-"
    assert _fmt_eq_margin({"kind": "relative", "margin": float("inf")}) == "-"
    print("  [ok] 等价性渲染：×100/×scale 溢出显示“溢出”，不写出 inf")


def test_ruler_length_rendering_handles_scaled_overflow():
    """长度剂量-反应表：×100 溢出不能写出 inf/nan。"""
    huge = 1e308
    row = {
        "candidate": "cand", "reference": "ref", "n_seeds": 3,
        "contexts": [128, 256], "n_contexts": 2,
        "overall_mean_diff": huge,
        "per_context": [{
            "ctx": 128, "n": 3, "mean_diff": huge,
            "mean_diff_ci95": [-huge, huge],
            "p": 0.5, "p_adj": 0.5, "verdict": "未定",
            "candidate_mean": 0.5, "reference_mean": 0.4,
        }],
        "slope_per_doubling": {
            "mean_diff": huge, "verdict": "未定", "p_adj": 0.5, "n": 3,
        },
        "long_minus_short": {"mean_diff": huge},
        "direction_consistency": {"positive": 3, "negative": 0, "zero": 0},
    }
    md = "\n".join(_render_ruler_length({"pairs": [row]}))
    assert "inf" not in md and "nan" not in md
    assert "溢出" in md
    print("  [ok] RULER 长度渲染：×100 溢出显示“溢出”，不写出 inf")


def test_build_summary_renders_near_duplicate_audit():
    """近似重复的删除量/阈值/召回率必须进入 summary.md，而不是只躺在 JSON。"""
    tmp = tempfile.mkdtemp()
    try:
        data_report = {
            "protocol": {"split_first_then_preprocess": True},
            "duplicates": {"near_duplicate": {
                "shingle_size": 5,
                "threshold": 0.8,
                "lsh_recall_at_threshold": 0.9865,
                "train_lines_dropped": 12,
                "train_blocks_dropped_as_near_duplicates_of_eval": 3,
                "train_intra_split_near_duplicates_removed": 2,
                "val_test_near_duplicate_pairs": 1,
                "candidate_pairs": 40,
                "candidate_pairs_truncated": False,
                "oversized_groups_truncated": 0,
                "min_chars": 200,
                "train_chars_coverage_fraction": 0.62,
                "train_chars_skipped_short": 1200,
                "train_chars_skipped_few_shingles": 0,
                "train_chars_skipped_too_large": 800,
                "threshold_sensitivity_removal": {
                    "0.7": {"train_blocks_removed_total": 5,
                            "train_chars_removed": 5000,
                            "removal_fraction_chars": 0.05},
                    "0.8": {"train_blocks_removed_total": 3,
                            "train_chars_removed": 3000,
                            "removal_fraction_chars": 0.03},
                    "0.9": {"train_blocks_removed_total": 1,
                            "train_chars_removed": 1000,
                            "removal_fraction_chars": 0.01},
                },
            }, "post_cleanup": {
                "checked_pairs": [["train", "val"]],
                "train_eval_clean": True,
                "overlap_long_lines": {},
                "overlap_blocks": {},
            }},
            "accounting": {
                "train": {"tokens": 1000, "chars": 4000,
                          "utf8_bytes": 4000, "bytes_per_token": 4.0},
            },
        }
        out = build_summary(tmp, data_report=data_report)
        md = out["markdown"]
        assert "近似重复去污染" in md
        assert "98.650%" in md
        assert "只改 train" in md
        # 覆盖率与三档“实际删除量”必须进入汇总页，而不只是判定对数
        assert "近似重复扫描覆盖" in md and "62.0%" in md
        assert "阈值敏感性（实际删除量" in md
        assert "τ=0.7" in md and "τ=0.9" in md
        assert "清洗后复核" in md and "已无共享" in md
        assert "bytes/token" in md and "4.0000" in md
        # 旧/损坏产物里的阈值键可能是任意文本：渲染不能因此崩溃
        data_report["duplicates"]["near_duplicate"][
            "threshold_sensitivity_removal"] = {
                "bad-key": {"train_blocks_removed_total": 1,
                            "train_chars_removed": 10,
                            "removal_fraction_chars": 0.001}}
        md_bad = build_summary(tmp, data_report=data_report)["markdown"]
        assert "τ=bad-key" in md_bad
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] build_summary：近似重复审计进入汇总页")
