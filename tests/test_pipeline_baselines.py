"""基线合理性、n-gram 与预注册确认性判定测试。"""
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



def test_render_baseline_sections_reads_evidence_and_warns():
    """汇总页必须渲染简单基线 PPL 与基线调优证据；缺失时显式警告。"""
    tmp = tempfile.mkdtemp()
    try:
        # 缺证据：必须给出“未运行/未提供”的警告而不是静默
        empty = _render_baseline_sections(tmp)
        text = "\n".join(empty)
        assert "基线合理性" in text
        assert "未运行" in text
        # 提供证据：调优选择、最优/默认差距与 ngram PPL 都要出现
        _write_json(os.path.join(tmp, "baseline_tuning.json"), {
            "variant": "standard", "steps": 100, "n_trials": 3,
            "selected": {"lr": 0.0012, "weight_decay": 0.1,
                         "val_loss_best": 3.1},
            "default_trial": {"lr": 0.0006, "weight_decay": 0.1,
                              "val_loss_best": 3.2},
            "selected_minus_default_val_loss": -0.1,
            "default_is_optimal": False,
        })
        _write_json(os.path.join(tmp, "ngram_baseline.json"), {
            "split": "val", "train_tokens": 1000, "eval_tokens": 100,
            "vocab_size": 50, "ppl": {"unigram": 90.0, "bigram": 45.0},
        })
        _write_json(os.path.join(tmp, "repro.json"), {"smoke": False})
        text2 = "\n".join(_render_baseline_sections(tmp))
        assert "0.0012" in text2 and "3.1000" in text2
        assert "90.000" in text2 and "45.000" in text2
        assert "全流水线" in text2 and "受控变量" in text2
        # 已有工作代表未各自调优：汇总页必须限定“同超参受控对照”，
        # 不能把 qgate/null 的对比写成“优于各自最佳调优版本”。
        text_scope = "\n".join(_render_baseline_sections(
            tmp, variants=["standard", "qgate", "null", "lega"]))
        assert "调优范围" in text_scope and "同超参受控对照" in text_scope
        assert "`qgate`" in text_scope and "`null`" in text_scope
        # 只有 standard/lega 时不产生无关警告
        text_no_scope = "\n".join(_render_baseline_sections(
            tmp, variants=["standard", "lega"]))
        assert "调优范围" not in text_no_scope
        # 默认组合 trial 失败（default_is_optimal=None）：不得冒充“并非最优”
        _write_json(os.path.join(tmp, "baseline_tuning.json"), {
            "variant": "standard", "n_trials": 2, "selected": {
                "lr": 0.0012, "weight_decay": 0.1, "val_loss_best": 3.1},
            "default_trial": None, "default_is_optimal": None,
            "selected_minus_default_val_loss": None})
        text4 = "\n".join(_render_baseline_sections(tmp))
        assert "无法判定默认是否" in text4
        # 调优缺证据但 ngram 在：仍必须点名“未调优基线”的风险
        os.remove(os.path.join(tmp, "baseline_tuning.json"))
        text3 = "\n".join(_render_baseline_sections(tmp))
        assert "未提供基线调优证据" in text3
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] _render_baseline_sections（证据渲染/缺失警告）")


def test_probe_table_renders_both_metric_p_values():
    """判定可能由 AUROC 驱动时，表格不能只显示 PR-AUC 的 p 值造成自相矛盾。"""
    tmp = tempfile.mkdtemp()
    try:
        row = {"variant": "lega", "n_seeds": 5, "hit_base_rate": 0.10,
               "features": ["log_maxp", "log_n", "qterm"],
               "rule_auprc": 0.1, "probe_auprc": 0.30,
               "gate_auprc": 0.32, "probe_auroc": 0.80,
               "gate_auroc": 0.70, "probe_f1_fixed": 0.4,
               "probe_f1_tuned": 0.45, "delta_auprc": -0.02,
               "delta_auprc_ci95": [-0.05, 0.01],
               "delta_auroc": 0.10, "delta_auroc_ci95": [0.01, 0.19],
               "p_adj_auprc": 0.80, "p_adj_auroc": 0.01,
               "primary_detection_metric": "AUROC 与 PR-AUC 并报",
               "verdict": "门控显著优于最简监督线性基线"}
        text = "\n".join(_render_baseline_sections(
            tmp, probe_cmp={"rows": [row], "missing": []}))
        assert "p_adj(PR-AUC)" in text and "p_adj(AUROC)" in text, text
        assert "0.8000" in text and "0.0100" in text, text
        assert "ΔAUROC" in text and "+0.100" in text, text
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] 线性探针表同时展示 PR-AUC/AUROC 的 Δ 与 p_adj")


def test_render_baseline_selection_stability():
    """调优小节必须渲染逐种子支持、次优噪声差距与 LOO 复选，不能只报最优。"""
    tmp = tempfile.mkdtemp()
    try:
        _write_json(os.path.join(tmp, "baseline_tuning.json"), {
            "variant": "standard", "steps": 100, "n_trials": 3,
            "selection_seeds": [7, 8, 9],
            "selected": {"lr": 0.0012, "weight_decay": 0.1,
                         "val_loss_best": 3.1},
            "default_trial": {"lr": 0.0006, "weight_decay": 0.1,
                              "val_loss_best": 3.2},
            "selected_minus_default_val_loss": -0.1,
            "default_is_optimal": False,
            "selection_diagnostics": {
                "n_seeds": 3, "selected_support_count": 2,
                "selected_support_fraction": 2 / 3,
                "runner_up": {
                    "lr": 0.0006, "weight_decay": 0.1,
                    "val_loss_best_mean": 3.15,
                    "selected_minus_runner_up_mean": -0.05,
                    "paired_delta_mean": 0.02,
                    "paired_delta_std": 0.1,
                    "within_seed_noise": True},
                "leave_one_seed_out": {
                    "n_folds": 3, "agreement_count": 2,
                    "agreement_rate": 2 / 3,
                    "heldout_gain_vs_default_mean": -0.01,
                    "heldout_gain_vs_default_std": 0.02,
                    "wins_vs_default": 1, "ties_vs_default": 0,
                    "losses_vs_default": 2},
            }})
        text = "\n".join(_render_baseline_sections(tmp))
        assert "选择稳定性" in text and "2/3" in text
        assert "均值最优并非逐种子稳定最优" in text
        assert "最优−次优" in text and "一个种子噪声 SD 内" in text
        assert "留一种子复选" in text and "胜/平/负 1/0/2" in text
        assert "既翻转选择、又未在留出种子上优于默认" in text
        # 旧调优报告（多种子但无 selection_diagnostics）不得静默当作稳定
        doc = json.load(open(os.path.join(tmp, "baseline_tuning.json"),
                             encoding="utf-8"))
        doc.pop("selection_diagnostics")
        _write_json(os.path.join(tmp, "baseline_tuning.json"), doc)
        text_old = "\n".join(_render_baseline_sections(tmp))
        assert "缺少 `selection_diagnostics`" in text_old
        assert "重跑 `--tune_baseline`" in text_old
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] _render_baseline_sections：调优选择稳定性证据")


def test_resolve_ngram_split_auto_matches_neural_eval_rule():
    """auto 必须与 lega.evaluate 的 --split auto 同规则：有非空 test.bin → test。"""
    tmp = tempfile.mkdtemp()
    try:
        assert resolve_ngram_split(tmp, "auto") == "val"
        with open(os.path.join(tmp, "test.bin"), "wb") as f:
            f.write(b"")
        assert resolve_ngram_split(tmp, "auto") == "val"   # 空文件不算
        with open(os.path.join(tmp, "test.bin"), "wb") as f:
            f.write(b"\x01")
        assert resolve_ngram_split(tmp, "auto") == "test"
        assert resolve_ngram_split(tmp, "val") == "val"    # 显式选择透传
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] resolve_ngram_split（auto 与神经评测 split 规则一致）")


def test_ngram_baseline_comparison_same_split_stats_and_mismatch():
    """神经变体 vs 简单基线：同 split 才可比、单样本配对检验 + Holm 判定。"""
    tmp = tempfile.mkdtemp()
    try:
        _write_json(os.path.join(tmp, "ngram_baseline.json"), {
            "split": "test", "ppl": {"unigram": 200.0, "bigram": 100.0},
            "selected_alpha": {"unigram": 0.1, "bigram": 0.1}})
        runs = {}
        for v, base in (("standard", 30.0), ("lega", 20.0)):
            runs[v] = {s: {"eval": {"variant": v, "split": "test",
                                    "ppl": base + s}}
                       for s in (1, 2, 3)}
        out = ngram_baseline_comparison(runs, tmp)
        assert out["split"] == "test" and out["reference_model"] == "bigram"
        assert abs(out["ngram_ppl"] - 100.0) < 1e-12
        rows = {r["variant"]: r for r in out["rows"]}
        assert rows["lega"]["n"] == 3
        assert abs(rows["lega"]["ratio"] - 22.0 / 100.0) < 1e-12
        assert rows["lega"]["verdict"] == "显著优于统计基线"
        assert rows["lega"]["p_adj"] is not None and rows["lega"]["p_adj"] < ALPHA
        # 只有 n>=MIN_SEEDS 的行进入 Holm 家族：n=1 的行 p=None 不占家族大小。
        runs_family = {"standard": {1: {"eval": {"variant": "standard",
                                                 "split": "test", "ppl": 30.0}}},
                       "lega": {s: {"eval": {"variant": "lega",
                                             "split": "test",
                                             "ppl": 20.0 + s}}
                                for s in (1, 2, 3)}}
        out_family = ngram_baseline_comparison(runs_family, tmp)
        assert out_family["holm_family_size"] == 1, out_family
        assert out_family["n_rows"] == 2
        # 合法但接近 float 上限的 PPL：朴素均值/比值会溢出成 inf。
        _write_json(os.path.join(tmp, "ngram_baseline.json"), {
            "split": "test", "ppl": {"bigram": 1e308}, "selected_model": "bigram"})
        huge_runs = {"lega": {s: {"eval": {"variant": "lega", "split": "test",
                                          "ppl": 1e308}}
                              for s in (1, 2, 3)}}
        huge = ngram_baseline_comparison(huge_runs, tmp)
        hrow = huge["rows"][0]
        assert hrow["mean_ppl"] == 1e308, hrow
        assert hrow["ratio"] == 1.0, hrow
        assert math.isfinite(hrow["mean_ppl"]), hrow
        _write_json(os.path.join(tmp, "ngram_baseline.json"), {
            "split": "test", "ppl": {"unigram": 200.0, "bigram": 100.0},
            "selected_alpha": {"unigram": 0.1, "bigram": 0.1}})
        # 显著劣于统计基线要被点名（神经模型连 n-gram 都打不过）
        runs_bad = {"lega": {s: {"eval": {"variant": "lega", "split": "test",
                                          "ppl": 120.0 + s}}
                             for s in (1, 2, 3)}}
        out_bad = ngram_baseline_comparison(runs_bad, tmp)
        assert out_bad["rows"][0]["verdict"] == "显著劣于统计基线（严重）"
        # split 错位（val 评测 vs test 基线）不得混入比较，必须显式列出
        runs_mix = {"qgate": {1: {"eval": {"variant": "qgate", "split": "val",
                                            "ppl": 25.0}}}}
        out_mix = ngram_baseline_comparison(runs_mix, tmp)
        assert out_mix["rows"] == []
        assert out_mix["incomparable"][0]["variant"] == "qgate"
        # 部分种子 split 错位：可比种子照常进 rows，但错位种子必须显式列出，
        # 不能因为“还有可比种子”就静默丢弃
        runs_partial = {"qgate": {
            1: {"eval": {"variant": "qgate", "split": "val", "ppl": 25.0}},
            2: {"eval": {"variant": "qgate", "split": "test", "ppl": 24.0}},
            3: {"eval": {"variant": "qgate", "split": "test", "ppl": 26.0}}}}
        out_partial = ngram_baseline_comparison(runs_partial, tmp)
        assert out_partial["rows"][0]["n"] == 2
        assert out_partial["incomparable"] == [
            {"variant": "qgate", "eval_splits": ["val"], "excluded_seeds": [1]}]
        text_partial = "\n".join(
            _render_baseline_sections(tmp, ngram_cmp=out_partial))
        assert "未纳入比较" in text_partial and "[1]" in text_partial
        # 渲染：比较表与错位警告都必须出现在基线小节
        text = "\n".join(_render_baseline_sections(tmp, ngram_cmp=out))
        assert "显著优于统计基线" in text and "比值" in text
        text_mix = "\n".join(_render_baseline_sections(tmp, ngram_cmp=out_mix))
        assert "不一致" in text_mix and "未纳入比较" in text_mix
        # 文件缺失 → 空报告（调用方按“未运行”渲染）
        tmp2 = tempfile.mkdtemp()
        try:
            assert ngram_baseline_comparison(runs, tmp2) == {}
        finally:
            shutil.rmtree(tmp2, ignore_errors=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] ngram_baseline_comparison（同 split 配对、Holm、错位隔离）")


def test_ngram_comparison_includes_explicit_supplemental_and_control():
    """显式运行的补充/安慰剂对照必须有同 split 的统计基线检验。"""
    tmp = tempfile.mkdtemp()
    try:
        _write_json(os.path.join(tmp, "ngram_baseline.json"), {
            "split": "test", "ppl": {"bigram": 100.0},
            "selected_model": "bigram"})
        runs = {}
        for v, base in (("lega_raw", 60.0),
                        (PARAM_CONTROL_VARIANT, 80.0)):
            runs[v] = {s: {"eval": {"variant": v, "split": "test",
                                    "ppl": base + s}}
                       for s in (1, 2, 3)}
        out = ngram_baseline_comparison(runs, tmp)
        rows = {r["variant"]: r for r in out["rows"]}
        assert set(rows) == {"lega_raw", PARAM_CONTROL_VARIANT}, out
        assert rows["lega_raw"]["n"] == 3
        assert out["holm_family_size"] == 2, out
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] ngram_baseline_comparison（补充/安慰剂对照不被静默跳过）")


def test_ngram_comparison_rejects_baseline_from_other_data():
    """results/<tag> 复用时，旧 n-gram 基线的数据指纹必须与本次 run 一致。

    否则 --skip_ngram 会让上一份数据（甚至重建后的 test.bin）的 PPL 与当前
    神经模型 PPL 并排比较，制造与本次数据无关的“显著优于统计基线”。
    """
    tmp = tempfile.mkdtemp()
    try:
        _write_json(os.path.join(tmp, "repro.json"), {
            "data": {"digest": "new-data-digest"}})
        _write_json(os.path.join(tmp, "ngram_baseline.json"), {
            "split": "val", "data_fingerprint": "old-data-digest",
            "ppl": {"unigram": 100.0, "bigram": 80.0,
                    "interpolated_bigram": 70.0},
            "selected_model": "interpolated_bigram"})
        out = ngram_baseline_comparison({}, tmp)
        assert out.get("error"), out
        assert "数据指纹" in out["error"], out
        # 数据指纹一致时正常进入比较流程（无 runs 时返回空 rows 而非错误）
        _write_json(os.path.join(tmp, "ngram_baseline.json"), {
            "split": "val", "data_fingerprint": "new-data-digest",
            "ppl": {"unigram": 100.0, "bigram": 80.0,
                    "interpolated_bigram": 70.0},
            "selected_model": "interpolated_bigram"})
        ok = ngram_baseline_comparison({}, tmp)
        assert "error" not in ok, ok
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] ngram 基线数据指纹与本次 run 绑定")


def test_ngram_comparison_tolerates_malformed_repro_data():
    """repro.json 的 data 不是对象时，指纹核对应降级为不可核对而非崩溃。"""
    tmp = tempfile.mkdtemp()
    try:
        _write_json(os.path.join(tmp, "repro.json"), {"data": ["not", "dict"]})
        _write_json(os.path.join(tmp, "ngram_baseline.json"), {
            "split": "val", "data_fingerprint": "old-data-digest",
            "ppl": {"unigram": 100.0, "bigram": 80.0,
                    "interpolated_bigram": 70.0},
            "selected_model": "interpolated_bigram"})
        out = ngram_baseline_comparison({}, tmp)
        assert "error" not in out, out
        # 合法对象仍按原规则拒绝跨数据基线
        _write_json(os.path.join(tmp, "repro.json"),
                    {"data": {"digest": "new-data-digest"}})
        bad = ngram_baseline_comparison({}, tmp)
        assert "数据指纹" in bad.get("error", ""), bad
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] ngram 指纹核对容忍 repro.json 的畸形 data 字段")


def test_ngram_comparison_uses_val_selected_strongest_baseline():
    """简单基线参照必须取 val 选出的最强模型，而不是固定拿 bigram 当靶子。"""
    tmp = tempfile.mkdtemp()
    try:
        _write_json(os.path.join(tmp, "ngram_baseline.json"), {
            "split": "val",
            "ppl": {"unigram": 200.0, "bigram": 100.0,
                    "interpolated_bigram": 80.0},
            "selected_model": "interpolated_bigram",
            "selected_interpolated": {"lambda": 0.7, "alpha_unigram": 0.1,
                                      "alpha_bigram": 0.1,
                                      "val_mean_nll": 4.0},
            "selected_alpha": {"unigram": 0.1, "bigram": 0.1}})
        runs = {"lega": {s: {"eval": {"variant": "lega", "split": "val",
                                      "ppl": 70.0 + s}}
                         for s in (1, 2, 3)}}
        out = ngram_baseline_comparison(runs, tmp)
        assert out["reference_model"] == "interpolated_bigram"
        assert abs(out["ngram_ppl"] - 80.0) < 1e-12
        assert out["rows"][0]["verdict"] == "显著优于统计基线"
        # 历史产物没有 selected_model：按插值→bigram→unigram 的强度顺序回退
        _write_json(os.path.join(tmp, "ngram_baseline.json"), {
            "split": "val",
            "ppl": {"unigram": 200.0, "bigram": 100.0,
                    "interpolated_bigram": 80.0}})
        out_old = ngram_baseline_comparison(runs, tmp)
        assert out_old["reference_model"] == "interpolated_bigram"
        # 更老产物连插值都没有：仍回退 bigram
        _write_json(os.path.join(tmp, "ngram_baseline.json"), {
            "split": "val", "ppl": {"unigram": 200.0, "bigram": 100.0}})
        out_older = ngram_baseline_comparison(runs, tmp)
        assert out_older["reference_model"] == "bigram"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] ngram_baseline_comparison（val 选出的最强简单基线优先）")


def test_ngram_comparison_prefers_window_matched_reference():
    """同 split 之后还要同 token：窗口规格一致时用逐位置对齐参考并披露。"""
    tmp = tempfile.mkdtemp()
    try:
        spec = {"ctx": 8, "iters": 2, "n_windows": 5, "n_eval": 2,
                "n_tokens": 16}
        _write_json(os.path.join(tmp, "ngram_baseline.json"), {
            "split": "val",
            "ppl": {"unigram": 200.0, "bigram": 100.0},
            "ppl_window_matched": {"unigram": 180.0, "bigram": 90.0,
                                   "n_tokens": 16},
            "window_spec": spec,
            "selected_alpha": {"unigram": 0.1, "bigram": 0.1}})
        matched_eval = {"variant": "lega", "split": "val", "ppl": 20.0 + 0,
                        "eval_ctx": 8, "eval_iters": 2, "val_windows": 5}
        runs = {"lega": {s: {"eval": dict(matched_eval, ppl=20.0 + s)}
                         for s in (1, 2, 3)}}
        out = ngram_baseline_comparison(runs, tmp)
        assert out["reference_source"] == "window_matched"
        assert abs(out["ngram_ppl"] - 90.0) < 1e-12
        assert abs(out["ngram_ppl_full_split"] - 100.0) < 1e-12
        assert out["fallback_reason"] is None
        assert all(r["token_matched"] for r in out["rows"])
        text = "\n".join(_render_baseline_sections(tmp, ngram_cmp=out))
        assert "逐位置对齐" in text
        assert "在神经评测实际使用的窗口目标 token 上" in text
        # 有新产物时优先用“同一批窗口 token 上重新选参”的参考，而不是
        # 仍拿全 val 选出的模型/α；全 split 参考保持原样供绝对难度对照。
        _write_json(os.path.join(tmp, "ngram_baseline.json"), {
            "split": "val",
            "ppl": {"unigram": 200.0, "bigram": 100.0,
                    "interpolated_bigram": 90.0},
            "selected_model": "interpolated_bigram",
            "ppl_window_matched": {"unigram": 180.0, "bigram": 90.0,
                                   "interpolated_bigram": 90.0,
                                   "n_tokens": 16},
            "window_matched_selection": {
                "selected_model": "bigram",
                "selected_alpha": {"unigram": 0.1, "bigram": 0.1}},
            "ppl_window_matched_selected": {"unigram": 170.0,
                                            "bigram": 70.0,
                                            "n_tokens": 16},
            "window_spec": spec})
        out_win_sel = ngram_baseline_comparison(runs, tmp)
        assert out_win_sel["reference_source"] == "window_matched"
        assert out_win_sel["reference_model"] == "interpolated_bigram"
        assert out_win_sel["reference_model_window_matched"] == "bigram"
        assert abs(out_win_sel["ngram_ppl_window_matched"] - 70.0) < 1e-12
        assert abs(out_win_sel["ngram_ppl_full_split"] - 90.0) < 1e-12
        # selected 表存在但缺少 window_matched_selection 元数据时，比较层
        # 必须忽略它并回退 full 窗口表；渲染层也要引用同一张表，不能把
        # 未采用的 70.0 写成“逐位置对齐口径”的参考值。
        _write_json(os.path.join(tmp, "ngram_baseline.json"), {
            "split": "val",
            "ppl": {"unigram": 200.0, "bigram": 100.0,
                    "interpolated_bigram": 90.0},
            "selected_model": "interpolated_bigram",
            "ppl_window_matched": {"unigram": 180.0, "bigram": 90.0,
                                   "interpolated_bigram": 90.0,
                                   "n_tokens": 16},
            "ppl_window_matched_selected": {"unigram": 170.0,
                                            "bigram": 70.0,
                                            "n_tokens": 16},
            "window_spec": spec})
        out_no_meta = ngram_baseline_comparison(runs, tmp)
        assert abs(out_no_meta["ngram_ppl_window_matched"] - 90.0) < 1e-12
        assert out_no_meta["window_reference_values"] == {
            "unigram": 180.0, "bigram": 90.0,
            "interpolated_bigram": 90.0, "n_tokens": 16}
        text_no_meta = "\n".join(
            _render_baseline_sections(tmp, ngram_cmp=out_no_meta))
        assert "180.000" in text_no_meta and "90.000" in text_no_meta
        assert "170.000" not in text_no_meta and "70.000" not in text_no_meta
        _write_json(os.path.join(tmp, "ngram_baseline.json"), {
            "split": "val",
            "ppl": {"unigram": 200.0, "bigram": 100.0},
            "ppl_window_matched": {"unigram": 180.0, "bigram": 90.0,
                                   "n_tokens": 16},
            "window_spec": spec,
            "selected_alpha": {"unigram": 0.1, "bigram": 0.1}})
        # 任一 run 的窗口字段与规格不符 → 整体回退全 split 参考并显式说明
        runs_mis = {"lega": {s: {"eval": dict(matched_eval, ppl=20.0 + s,
                                              eval_iters=3)}
                             for s in (1, 2, 3)}}
        out_mis = ngram_baseline_comparison(runs_mis, tmp)
        assert out_mis["reference_source"] == "full_split"
        assert abs(out_mis["ngram_ppl"] - 100.0) < 1e-12
        assert "lega" in (out_mis["fallback_reason"] or "")
        assert not any(r["token_matched"] for r in out_mis["rows"])
        text_mis = "\n".join(_render_baseline_sections(tmp, ngram_cmp=out_mis))
        assert "未使用逐位置对齐参考" in text_mis
        # 回退时必须撤销“这些数字=神经评测实际使用的 token”的宣称
        assert "在神经评测实际使用的窗口目标 token 上" not in text_mis
        assert "回退全 split 参考" in text_mis
        # 历史产物缺窗口规格 → 同样回退且如实说明原因
        tmp2 = tempfile.mkdtemp()
        try:
            _write_json(os.path.join(tmp2, "ngram_baseline.json"), {
                "split": "val", "ppl": {"bigram": 100.0}})
            out_old = ngram_baseline_comparison(runs, tmp2)
            assert out_old["reference_source"] == "full_split"
            assert "window_spec" in (out_old["fallback_reason"] or "")
        finally:
            shutil.rmtree(tmp2, ignore_errors=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] ngram_baseline_comparison（窗口对齐参考优先、回退显式披露）")


def test_ngram_window_paired_test_uses_same_window_indices():
    """n-gram 对比必须能逐窗口配对；窗口序号不一致时显式跳过而不是强配。"""
    tmp = tempfile.mkdtemp()
    try:
        idx = list(range(0, 30, 3))  # 10 个窗口，越过最低窗口门槛
        spec = {"ctx": 8, "iters": 10, "n_windows": 20, "n_eval": 10,
                "n_tokens": 80}
        _write_json(os.path.join(tmp, "ngram_baseline.json"), {
            "split": "val", "ppl": {"bigram": 100.0},
            "selected_model": "bigram",
            "ppl_window_matched": {"bigram": 100.0, "n_tokens": 80},
            "window_losses_matched": [1.0] * 10,
            "window_indices_matched": idx,
            "window_losses_matched_model": "bigram",
            "window_spec": spec})
        base = {"variant": "lega", "split": "val", "ppl": 20.0,
                "eval_ctx": 8, "eval_iters": 10, "val_windows": 20,
                "val_tokens_evaluated": 80,
                "window_losses": [1.5] * 10, "window_indices": idx}
        runs = {"lega": {s: {"eval": dict(base, ppl=20.0 + s)}
                         for s in (1, 2, 3)}}
        out = ngram_baseline_comparison(runs, tmp)
        r = out["rows"][0]
        assert out["reference_source"] == "window_matched", out
        assert r["window_paired_n_seeds"] == 3, r
        assert abs(r["window_paired_mean_diff"] - 0.5) < 1e-12, r
        assert r["window_paired_block_ci95"] is not None, r
        assert r["window_paired_block_length_min"] >= 1, r
        assert r["window_paired_p_adj"] is not None, r
        assert r["window_paired_verdict"].startswith("窗口级显著劣于"), r
        text = "\n".join(_render_baseline_sections(tmp, ngram_cmp=out))
        assert "逐窗口配对（探索性" in text and "ΔNLL(神经−n-gram)" in text
        # 窗口序号不一致：整行逐窗口统计必须放弃，并写明原因
        bad = {s: {"eval": dict(base, ppl=20.0 + s, window_indices=[0, 4] * 5)}
               for s in (1, 2, 3)}
        out_bad = ngram_baseline_comparison({"lega": bad}, tmp)
        rb = out_bad["rows"][0]
        assert rb["window_paired_mean_diff"] is None, rb
        assert "窗口序号不一致" in (rb["window_paired_skip_reason"] or ""), rb
        assert "逐窗口配对（探索性" not in "\n".join(
            _render_baseline_sections(tmp, ngram_cmp=out_bad))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] n-gram 对比：逐窗口配对 + 窗口序号不一致显式降级")


def test_neural_eval_window_spec_matches_train_overrides():
    """n-gram 对齐用的窗口规格必须与训练/评测实际口径一致（含冒烟覆盖）。"""
    tiny = {"block_size": 256, "train_ctx": 256, "eval_iters": 20}
    assert _neural_eval_window_spec(tiny, False) == {"ctx": 256, "iters": 20}
    # 冒烟：train.py 的 SMOKE_OVERRIDES（train_ctx=128、eval_iters=2）必须生效
    assert _neural_eval_window_spec(tiny, True) == {"ctx": 128, "iters": 2}
    # train_ctx 被 block_size 截断；默认 eval_iters=50；非法规格返回 None
    assert _neural_eval_window_spec(
        {"block_size": 64, "train_ctx": 256}, False) == {"ctx": 64, "iters": 50}
    assert _neural_eval_window_spec({}, False) is None
    # bool/小数不得截断成“另一个窗口规格”
    for bad in ({"eval_iters": True}, {"eval_iters": 2.5},
                {"train_ctx": True}, {"block_size": 64.5}):
        assert _neural_eval_window_spec(bad, False) is None, bad
    assert _neural_eval_window_spec(
        {"block_size": 64, "train_ctx": 256.0, "eval_iters": 20.0},
        False) == {"ctx": 64, "iters": 20}
    print("  [ok] _neural_eval_window_spec（与 train/evaluate 口径一致）")


def test_main_rejects_empty_variants():
    """--variants 全为空项时必须在启动流水线前报错，不能产生空实验。"""
    tmp = tempfile.mkdtemp()
    try:
        with mock.patch.object(sys, "argv", [
                "run_all.py", "--variants", ", ,", "--out_root", tmp,
                "--results_dir", tmp]):
            try:
                run_all_main()
                raise AssertionError("空 --variants 应触发 argparse 错误")
            except SystemExit as e:
                assert e.code == 2, e.code
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] run_all：空 --variants 显式拒绝")


def test_registration_scope_check_and_gate():
    """登记表完整还不够：run 必须落在登记范围内，否则确认性判定降级。"""
    tmp = tempfile.mkdtemp()
    try:
        reg_cfg = os.path.join(tmp, "registered.json")
        act_cfg = os.path.join(tmp, "actual.json")
        base = {"d_model": 384, "n_layers": 6, "n_heads": 6, "ffn_mult": 4,
                "block_size": 1024, "steps": 8000, "train_ctx": 1024,
                "lr": 6e-4, "weight_decay": 0.1}
        _write_json(reg_cfg, base)
        # 调优注入的 lr/wd 改变不算越界（协议内），其它字段必须一致
        _write_json(act_cfg, dict(base, lr=3e-4))
        registered = {"scope": {"config": reg_cfg, "seeds": [1, 2, 3],
                                "data": "WikiText-103 (gpt2 BPE)"}}
        repro = {"config_file": act_cfg, "seeds": [1, 2, 4], "smoke": False,
                 "data": {"tokenizer": "gpt2"}}
        sc = registration_scope_check(registered, repro)
        assert sc["checked"] and sc["in_scope"] is True and not sc["problems"]
        assert any("4" in w for w in sc["warnings"])   # 未登记种子显式警告
        hyp = [{"id": "H1", "confirmatory_valid": True}]
        assert apply_scope_gate(hyp, sc)[0]["confirmatory_valid"] is True
        # 训练规模与登记配置不一致（如 tiny 配置冒充正式实验）→ 范围外
        _write_json(act_cfg, dict(base, steps=100))
        sc2 = registration_scope_check(registered, repro)
        assert sc2["in_scope"] is False and sc2["problems"]
        hyp2 = [{"id": "H1", "confirmatory_valid": True}]
        assert apply_scope_gate(hyp2, sc2)[0]["confirmatory_valid"] is False
        _write_json(act_cfg, base)
        # 伪数据（字符分词）冒充 gpt2 BPE → 范围外
        sc3 = registration_scope_check(
            registered, {"config_file": act_cfg, "seeds": [1],
                         "smoke": False, "data": {"tokenizer": "char"}})
        assert sc3["in_scope"] is False
        assert any("分词器" in p for p in sc3["problems"])
        # 冒烟运行 → 范围外
        sc4 = registration_scope_check(
            registered, {"config_file": act_cfg, "seeds": [1],
                         "smoke": True, "data": {"tokenizer": "gpt2"}})
        assert sc4["in_scope"] is False
        # 缺 repro → 不可确认（不能默认通过）→ 降级
        sc5 = registration_scope_check(registered, {})
        assert sc5["checked"] is False and sc5["in_scope"] is None
        hyp5 = [{"id": "H1", "confirmatory_valid": True}]
        assert apply_scope_gate(hyp5, sc5)[0]["confirmatory_valid"] is False
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] registration_scope_check/apply_scope_gate（范围外降级）")


def test_registration_scope_render_matches_gate_and_paths():
    """范围核对渲染必须与实际降级一致，且 config_file 路径解析正确。

    登记表未声明 scope 时 apply_scope_gate 明确不降级，渲染不得声称
    “按不可确认降级”；相对 config_file 必须按项目根解析，不能相对 cwd
    静默核对到另一个同名文件；非字符串 config_file 按不可读处理而非崩溃。
    """
    sc = registration_scope_check({}, {"seeds": [1]})
    hyp = [{"id": "H1", "confirmatory_valid": True}]
    apply_scope_gate(hyp, sc)
    assert hyp[0]["confirmatory_valid"] is True
    text = "\n".join(_render_registration_integrity(
        {"status": "intact", "confirmatory_valid": True}, sc))
    assert "不可确认" not in text and "降级" not in text
    assert "未声明 scope" in text
    # 声明了 scope 却核不动 → 实际降级，渲染必须同样声明降级
    sc2 = registration_scope_check({"scope": {"data": "WikiText-103 (gpt2 BPE)"}}, {})
    hyp2 = [{"id": "H1", "confirmatory_valid": True}]
    apply_scope_gate(hyp2, sc2)
    assert hyp2[0]["confirmatory_valid"] is False
    text2 = "\n".join(_render_registration_integrity(
        {"status": "intact", "confirmatory_valid": True}, sc2))
    assert "不可确认" in text2 and "降级" in text2
    # 损坏的 checked/confirmatory_valid 不能被渲染成“已核对/有效”。
    text_bad = "\n".join(_render_registration_integrity(
        {"status": "intact", "confirmatory_valid": "false"},
        {"scope": {"config": "x"}, "checked": "false", "problems": []}))
    assert "不可确认" in text_bad and "失效" in text_bad, text_bad
    tmp = tempfile.mkdtemp()
    try:
        reg = os.path.join(tmp, "reg.json")
        _write_json(reg, {"d_model": 8})
        _write_json(os.path.join(tmp, "actual.json"), {"d_model": 16})
        sc3 = registration_scope_check({"scope": {"config": reg}},
                                       {"config_file": "actual.json"}, root=tmp)
        assert sc3["problems"] and "d_model" in sc3["problems"][0]
        sc4 = registration_scope_check({"scope": {"config": reg}},
                                       {"config_file": {"x": 1}}, root=tmp)
        assert sc4["warnings"] and sc4["checked"] is False
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] 范围核对渲染/降级一致 + config_file 路径解析")


def test_registered_hypotheses_evaluation():
    """预注册假设判定：支持/拒绝/未定的边界与未登记文件时的提示。"""
    tests = [
        _mk("lega", "standard", "PPL", -0.5, 0.01, 5),
        _mk("lega_no_len", "lega", "PPL", 0.4, 0.02, 5),   # 去掉长度项显著变差
        _mk("qgate", "standard", "PPL", 0.1, 0.5, 5),      # 不显著
        _mk("x", "y", "PPL", 1.0, 0.01, 2),                # 种子不足
    ]
    # 让 p_adj 直接可判（单家族 m=4 Holm：0.01→0.04, 0.02→0.06, ...）
    _apply_holm(tests)
    reg = {"hypotheses": [
        {"id": "H1", "claim": "LEGA 优于 standard（PPL）", "metric": "PPL",
         "candidate": "lega", "reference": "standard",
         "direction": "candidate_better"},
        {"id": "H2", "claim": "长度校准有用（去一法）", "metric": "PPL",
         "candidate": "lega_no_len", "reference": "lega",
         "direction": "candidate_worse"},
        {"id": "H3", "claim": "LEGA 优于 qgate（PPL）", "metric": "PPL",
         "candidate": "lega", "reference": "qgate",   # 未运行 → 未检验
         "direction": "candidate_better"},
        {"id": "H4", "claim": "方向登记反了的假设", "metric": "PPL",
         "candidate": "lega", "reference": "standard",
         "direction": "candidate_worse"},             # 显著但方向相反 → 拒绝
        {"id": "H5", "claim": "种子不足的对比", "metric": "PPL",
         "candidate": "x", "reference": "y",
         "direction": "candidate_better"},
    ]}
    ev = {h["id"]: h for h in evaluate_registered_hypotheses(reg, tests)}
    assert ev["H1"]["status"] == "支持"
    assert ev["H2"]["status"] == "支持"
    assert ev["H3"]["status"] == "未检验"
    assert ev["H4"]["status"] == "拒绝"
    assert ev["H5"]["status"] == "未定" and "种子不足" in ev["H5"]["detail"]
    # 不显著 → 未定，且 detail 提示 MDE（有均值/方差可估时）
    reg2 = {"hypotheses": [
        {"id": "H6", "claim": "qgate 优于 standard", "metric": "PPL",
         "candidate": "qgate", "reference": "standard",
         "direction": "candidate_better"}]}
    ev2 = evaluate_registered_hypotheses(reg2, tests)
    assert ev2[0]["status"] == "未定" and "不显著" in ev2[0]["detail"]
    # 空注册文件：不判定
    assert evaluate_registered_hypotheses({}, tests) == []
    print("  [ok] evaluate_registered_hypotheses（支持/拒绝/未定/未检验边界）")


def test_global_fwer_gate_across_registered_hypotheses():
    """跨指标家族的确认族必须再做全局 Holm，不能各自 5% 就宣布支持。

    构造 6 条各自家族内 p_adj=0.02 的假设：家族内全部显著，但全局
    Holm（m=6）后 p_adj=0.12，全部应降级为“未定”；status_family 仍保留
    原判定供审计。2 条假设时全局 m=2 的 p_adj=0.04，仍可支持。
    """
    reg = {"hypotheses": []}
    tests = []
    for i in range(6):
        metric = f"M{i}"
        cand, ref = f"cand{i}", f"ref{i}"
        reg["hypotheses"].append({
            "id": f"H{i + 1}", "claim": f"claim {i}", "metric": metric,
            "candidate": cand, "reference": ref,
            "direction": "candidate_better"})
        tests.append(_mk(cand, ref, metric, -0.5, 0.02, 5))
    _apply_holm(tests)
    ev = evaluate_registered_hypotheses(reg, tests)
    assert len(ev) == 6, ev
    assert all(h["status_family"] == "支持" for h in ev), ev
    assert all(h["status"] == "未定" for h in ev), ev
    assert all(h["p_adj_confirmatory"] is not None
               and h["p_adj_confirmatory"] >= 0.05 for h in ev), ev
    assert all("全局 Holm" in h["detail"] for h in ev), ev

    reg2 = {"hypotheses": reg["hypotheses"][:2]}
    ev2 = evaluate_registered_hypotheses(reg2, tests)
    assert all(h["status"] == "支持" for h in ev2), ev2
    assert all(abs(h["p_adj_confirmatory"] - 0.04) < 1e-12
               for h in ev2), ev2
    print("  [ok] 全局确认族 Holm（跨指标家族 FWER；仅家族内显著会降级）")


def test_build_summary_renders_confirmatory_sections():
    """summary.md 必须包含注册假设小节与置换/CI/MDE 列，而非只留 JSON。"""
    tmp = tempfile.mkdtemp()
    try:
        seeds = [1, 2, 3]
        for v, ppl, al, aq in (
                ("standard", 10.0, None, None),
                ("lega", 9.4, 0.8, 0.7),
                ("lega_no_len", 9.9, 0.75, 0.7),
                ("qgate", 9.8, 0.72, 0.71)):
            for s in seeds:
                jitter = 0.02 * s
                _write_json(os.path.join(tmp, f"eval_{v}_seed{s}.json"), {
                    "variant": v, "seed": s, "val_ppl": ppl + jitter,
                    "val_loss": 2.3, "n_params": 100,
                    "n_gate_extra": 0, "split": "val"})
                if al is not None:
                    _write_json(os.path.join(tmp, f"analysis_{v}_seed{s}.json"), {
                        "variant": v, "seed": s, "auroc_logit": al + 0.01 * s,
                        "auroc_qterm": aq - 0.01 * s})
        reg = {"hypotheses": [
            {"id": "H1", "claim": "LEGA 优于 standard（PPL）", "metric": "PPL",
             "candidate": "lega", "reference": "standard",
             "direction": "candidate_better"}]}
        intact = {"status": "intact", "confirmatory_valid": True, "problems": [],
                  "registry_sha256": "a" * 64, "n_hypotheses": 1,
                  "baseline_document_sha256": "b" * 64, "chain_length": 0}
        with mock.patch("lega.pipeline.report.summary.load_registered_hypotheses",
                        return_value=reg), \
                mock.patch("lega.pipeline.report.summary.verify_registration",
                           return_value=intact):
            out = build_summary(tmp)
        md = out["markdown"]
        assert "注册假设判定" in md
        assert "H1" in md and "事后" in md
        # 配对检验表的新列
        assert "p_perm(Holm)" in md and "MDE" in md and "Δ 95% CI" in md
        assert "AUROC差" in md
        # 单因素审计必须进入汇总页与 significance.json
        assert "单因素消融审计" in md
        sig = json.load(open(os.path.join(tmp, "significance.json"),
                             encoding="utf-8"))
        assert sig["single_factor_audit"]["ok"] is True
        assert len(sig["single_factor_audit"]["contrasts"]) == 6
        assert any(t["metric"] == AUROC_DIFF_METRIC for t in sig["pairs"])
        assert sig["registered_hypotheses"], "注册假设判定必须进 significance.json"
        h1 = next(h for h in sig["registered_hypotheses"] if h["id"] == "H1")
        assert h1["status"] in ("支持", "未定", "拒绝")
        # 预注册完整性报告与完整性小节必须进入 JSON 与汇总页
        assert sig["registration_integrity"]["status"] in ("intact", "amended")
        assert sig["registration_integrity"]["confirmatory_valid"] is True
        assert "预注册完整性" in md and "防事后静默改写" in md
        assert all(h.get("confirmatory_valid") for h in
                   sig["registered_hypotheses"])
        # 每条 pair 检验都带稳健性字段
        for t in sig["pairs"]:
            assert "p_perm" in t and "mean_diff_ci95" in t and "mde" in t
            assert "median_diff" in t and "p_sign_adj" in t
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] build_summary（注册假设/置换/CI/MDE 进入汇总页与 JSON）")


def test_build_summary_downgrades_confirmatory_when_integrity_violated():
    """锁链校验失败时：确认性判定必须显式降级，不能假装边界仍成立。"""
    tmp = tempfile.mkdtemp()
    try:
        for s in (1, 2, 3):
            _write_json(os.path.join(tmp, f"eval_lega_seed{s}.json"), {
                "variant": "lega", "seed": s, "val_ppl": 9.0 + 0.01 * s,
                "val_loss": 2.2, "n_params": 100, "n_gate_extra": 0,
                "split": "val"})
            _write_json(os.path.join(tmp, f"eval_standard_seed{s}.json"), {
                "variant": "standard", "seed": s, "val_ppl": 10.0 + 0.01 * s,
                "val_loss": 2.3, "n_params": 100, "n_gate_extra": 0,
                "split": "val"})
        reg = {"hypotheses": [
            {"id": "H1", "claim": "LEGA 优于 standard（PPL）", "metric": "PPL",
             "candidate": "lega", "reference": "standard",
             "direction": "candidate_better"}]}
        violated = {"status": "violated", "confirmatory_valid": False,
                    "problems": ["已登记的假设 H1 科学内容被改写"],
                    "registry_sha256": "a" * 64, "n_hypotheses": 1,
                    "baseline_document_sha256": "b" * 64,
                    "chain_length": 0}
        with mock.patch("lega.pipeline.report.summary.load_registered_hypotheses",
                        return_value=reg), \
                mock.patch("lega.pipeline.report.summary.verify_registration",
                           return_value=violated):
            out = build_summary(tmp)
        md = out["markdown"]
        assert "已登记的假设 H1 科学内容被改写" in md
        assert "失效" in md and "降级为探索性" in md
        assert "⚠（无效）" in md
        sig = json.load(open(os.path.join(tmp, "significance.json"),
                             encoding="utf-8"))
        assert sig["registration_integrity"]["status"] == "violated"
        assert all(not h["confirmatory_valid"]
                   for h in sig["registered_hypotheses"])
        assert all("降级" in h["confirmatory_note"]
                   for h in sig["registered_hypotheses"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] build_summary（完整性违反 → 确认性判定显式降级）")


def test_build_summary_zero_failure_variant_blocks_consensus():
    """RULER 已评测且零错误的变体必须进入跨变体对照（n_failed=0）。"""
    tmp = tempfile.mkdtemp()
    try:
        case = {"sample_id": "niah_synth@256#0", "task": "niah_synth",
                "ctx": 256, "answer": "1234567", "answer_words": None,
                "pred": "7654321", "em": 0.0, "recall": None,
                "question": "What is the special magic number?",
                "needle_excerpt": "magic number is: 1234567.",
                "depth": 0.5, "meta": {}, "context_numbers": ["7654321"],
                "answer_in_prompt": True}
        for v, cases in (("standard", [case]), ("lega", [case]), ("qgate", [])):
            _write_json(os.path.join(tmp, f"ruler_{v}_seed1.json"),
                        {"variant": v, "seed": 1, "results": {},
                         "badcases": cases})
            _write_json(os.path.join(tmp, f"eval_{v}_seed1.json"), {
                "variant": v, "seed": 1, "val_ppl": 10.0, "val_loss": 2.3,
                "gate_mean": 0.5, "n_params": 100, "n_gate_extra": 0,
                "split": "test"})
        out = build_summary(tmp)
        cross = json.load(open(os.path.join(tmp, "badcase_cross.json"),
                               encoding="utf-8"))
        assert cross["ruler"]["per_variant"]["qgate"]["n_failed"] == 0
        assert cross["ruler"]["n_shared_all_failures"] == 0
        assert "全部通过" in out["markdown"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] build_summary：零错误变体计入跨变体对照")


def test_build_summary_tolerates_seedless_artifacts():
    """产物没有 seed 字段时按缺种子信息处理。

    _seed_key 对无名种子文件返回 None（全表以 None 表“无种子信息”）；ngram
    对比的 excluded_seeds/seeds 与线性探针逐种子表沿用同一口径。
    """
    tmp = tempfile.mkdtemp()
    try:
        _write_json(os.path.join(tmp, "eval_standard.json"),
                    {"variant": "standard", "val_ppl": 10.0, "val_loss": 2.3,
                     "n_params": 10, "split": "val"})
        # 无 seed 且 split 与 ngram 不一致 → 进入 excluded_seeds 分支
        _write_json(os.path.join(tmp, "eval_lega.json"),
                    {"variant": "lega", "val_ppl": 9.0, "val_loss": 2.2,
                     "n_params": 10, "split": "test"})
        _write_json(os.path.join(tmp, "ngram_baseline.json"),
                    {"ppl": {"unigram": 100.0, "bigram": 50.0},
                     "split": "val", "train_tokens": 100, "eval_tokens": 50,
                     "vocab_size": 20})
        _write_json(os.path.join(tmp, "analysis_lega.json"), {
            "variant": "lega",
            "linear_probe": {"primary": "conc_len_query",
                             "feature_sets": {"conc_len_query": {
                                 "oof": {"auprc": 0.5, "auroc": 0.6}}}}})
        build_summary(tmp)
        sig = json.load(open(os.path.join(tmp, "significance.json"),
                             encoding="utf-8"))
        excl = [x for x in sig["ngram_comparison"].get("incomparable", [])
                if x["variant"] == "lega"]
        assert excl and excl[0]["excluded_seeds"] == [None], sig
        probe = sig["linear_probe_comparison"]
        assert probe["rows"] and probe["rows"][0]["seeds"] == [None]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] build_summary：无 seed 产物按缺种子信息处理")
