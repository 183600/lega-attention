"""汇总报告、运行时指标与配对检验渲染测试。"""
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



def test_build_summary_worst_windows_and_cross_variant():
    """最差窗口归因与跨变体错误对照必须进入汇总页与 JSON，而不是只留在文件里。"""
    tmp = tempfile.mkdtemp()
    try:
        for v, sids in (("standard", ["niah_synth@256#0"]),
                        ("lega", ["niah_synth@256#0", "niah_synth@256#1"])):
            cases = [{"sample_id": sid, "task": "niah_synth", "ctx": 256,
                      "answer": "1234567", "answer_words": None,
                      "pred": "7654321", "em": 0.0, "recall": None,
                      "question": "What is the special magic number?",
                      "needle_excerpt": "magic number is: 1234567.",
                      "depth": 0.5, "meta": {}, "context_numbers": ["7654321"],
                      "answer_in_prompt": True} for sid in sids]
            _write_json(os.path.join(tmp, f"ruler_{v}_seed1.json"),
                        {"variant": v, "seed": 1, "results": {},
                         "badcases": cases})
            _write_json(os.path.join(tmp, f"eval_{v}_seed1.json"), {
                "variant": v, "seed": 1, "val_ppl": 10.0, "val_loss": 2.3,
                "gate_mean": 0.5, "n_params": 100, "n_gate_extra": 0,
                "split": "test",
                "worst_windows": [{"window": 1, "split": "test", "loss": 8.0,
                                   "context_tail": "sky is blue and",
                                   "gold_next": " green",
                                   "pred_top5": [[" blue", 0.3]]}]})
        out = build_summary(tmp)
        assert len(out["rows"]) == 2
        md = out["markdown"]
        assert "跨变体错误对照" in md and "方法特异性" in md
        assert "语言建模最差窗口" in md and "复读吸引子" in md
        cross_path = os.path.join(tmp, "badcase_cross.json")
        assert os.path.exists(cross_path)
        cross = json.load(open(cross_path, encoding="utf-8"))
        # lega 相对 standard 多错一题 = 方法特异性新增；standard 无错误清单时
        # 该样本才是共识失败（这里 standard 也错了 #0）
        assert cross["ruler"]["per_variant"]["lega"]["n_new_vs_reference"] == 1
        assert cross["ruler"]["n_shared_all_failures"] == 1
        # 两个变体的最差窗口是同一个 → 全体共识失败
        assert cross["lm_windows"]["n_shared_all_failures"] == 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] build_summary（最差窗口与跨变体错误对照进入汇总）")


def test_build_summary_worst_window_unknown_gold_context_not_false():
    """缺 gold_in_context 的旧窗口不能渲染成“gold 不在上下文中”：
    未知与明确的 False 必须区分，避免把未记录误读成可归因的错误。"""
    tmp = tempfile.mkdtemp()
    try:
        _write_json(os.path.join(tmp, "repro.json"),
                    {"variants": ["lega"], "seeds": [1], "smoke": False})
        _write_json(os.path.join(tmp, "eval_lega_seed1.json"), {
            "variant": "lega", "seed": 1, "val_ppl": 10.0, "val_loss": 2.3,
            "n_params": 10, "split": "test",
            "worst_windows": [{"window": 1, "split": "test", "loss": 8.0,
                               "gold_next": " green",
                               "pred_top5": [[" blue", 0.3]]}]})
        out = build_summary(tmp)
        md = out["markdown"]
        assert "gold 是否在上下文中未记录" in md, md
        assert "gold 不在上下文中" not in md, md
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] build_summary（未知 gold_in_context 不伪装成 False）")


def test_build_summary_worst_windows_falls_back_to_seed_with_data():
    """最小种子没转储最差窗口（未开 --worst_n）时，不得把其他种子的
    窗口一并静默丢弃。"""
    tmp = tempfile.mkdtemp()
    try:
        _write_json(os.path.join(tmp, "repro.json"),
                    {"variants": ["lega"], "seeds": [1, 2], "smoke": False})
        _write_json(os.path.join(tmp, "eval_lega_seed1.json"),
                    {"variant": "lega", "seed": 1, "val_ppl": 10.0,
                     "val_loss": 2.3, "n_params": 10, "split": "test"})
        _write_json(os.path.join(tmp, "eval_lega_seed2.json"), {
            "variant": "lega", "seed": 2, "val_ppl": 10.0, "val_loss": 2.3,
            "n_params": 10, "split": "test",
            "worst_windows": [{"window": 1, "split": "test", "loss": 8.0,
                               "context_tail": "sky is blue and",
                               "gold_next": " green",
                               "pred_top5": [[" blue", 0.3]]}]})
        out = build_summary(tmp)
        assert "最差窗口数" in out["markdown"]
        assert "8.000" in out["markdown"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] build_summary（最差窗口退化到有数据的种子）")


def test_build_summary_worst_windows_aggregate_seeds():
    """逐变体归因必须累计全部种子的最差窗口（同一窗口在多种子中错要重复
    计入以显示错误模式的跨种子稳定性）；若仍只取一个种子，这里会少 2 条。"""
    tmp = tempfile.mkdtemp()
    try:
        _write_json(os.path.join(tmp, "repro.json"),
                    {"variants": ["lega"], "seeds": [1, 2], "smoke": False})
        w1 = {"window": 1, "split": "test", "loss": 8.0,
              "context_tail": "sky is blue and", "gold_next": " green",
              "pred_top5": [[" blue", 0.3]]}          # 复读吸引子
        w2 = {"window": 2, "split": "test", "loss": 7.0,
              "context_tail": "the cat sat on", "gold_next": " the",
              "pred_top5": [[" mat", 0.4]]}           # 低置信
        _write_json(os.path.join(tmp, "eval_lega_seed1.json"), {
            "variant": "lega", "seed": 1, "val_ppl": 10.0, "val_loss": 2.3,
            "n_params": 10, "split": "test", "worst_windows": [w1]})
        _write_json(os.path.join(tmp, "eval_lega_seed2.json"), {
            "variant": "lega", "seed": 2, "val_ppl": 10.0, "val_loss": 2.3,
            "n_params": 10, "split": "test", "worst_windows": [w1, w2]})
        out = build_summary(tmp)
        md = out["markdown"]
        assert "| LEGA(ours) | 3 |" in md, md
        assert "复读吸引子（低置信下首选续写退化为抄上下文已有串）×2" in md, md
        assert "2 个种子" in md, md
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] build_summary（最差窗口跨种子累计、说明口径）")


def test_build_summary_prefers_current_naming_on_slot_collision():
    """同一 (变体, 种子) 同时存在新旧两种文件名时必须取本次命名约定的
    文件：否则在旧多种子目录上重跑单种子，陈旧的 eval_lega_seed<seed>.json
    会经 _seed_key 落到同一槽位并覆盖新写的 eval_lega.json。"""
    tmp = tempfile.mkdtemp()
    try:
        # 单种子运行（约定不带后缀）：新写的无后缀文件必须胜出
        _write_json(os.path.join(tmp, "repro.json"),
                    {"variants": ["lega"], "seeds": [1337], "smoke": False})
        _write_json(os.path.join(tmp, "eval_lega.json"),
                    {"variant": "lega", "seed": 1337, "val_ppl": 7.0,
                     "val_loss": 1.9, "n_params": 10})
        _write_json(os.path.join(tmp, "eval_lega_seed1337.json"),
                    {"variant": "lega", "seed": 1337, "val_ppl": 99.0,
                     "val_loss": 4.6, "n_params": 10})
        out = build_summary(tmp)
        assert [r["val_ppl"] for r in out["rows"]] == [7.0]
        # 多种子运行（约定带后缀）：上一轮单种子的无后缀残留不得覆盖
        os.remove(os.path.join(tmp, "eval_lega.json"))
        _write_json(os.path.join(tmp, "repro.json"),
                    {"variants": ["lega"], "seeds": [1337, 2], "smoke": False})
        _write_json(os.path.join(tmp, "eval_lega.json"),
                    {"variant": "lega", "seed": 1337, "val_ppl": 5.5,
                     "val_loss": 1.7, "n_params": 10})
        out2 = build_summary(tmp)
        assert [r["val_ppl"] for r in out2["rows"]] == [99.0]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] build_summary（同槽冲突按本次命名约定取数）")


def test_build_summary_missing_latency_entry():
    """latency.json 缺少某变体条目（--skip_latency 复用旧结果）时安全跳过该行。"""
    tmp = tempfile.mkdtemp()
    try:
        for v in ("standard", "lega"):
            _write_json(os.path.join(tmp, f"eval_{v}_seed1.json"), {
                "variant": v, "seed": 1, "val_ppl": 10.0, "val_loss": 2.3,
                "gate_mean": 0.5 if v == "lega" else None,
                "n_params": 100, "n_gate_extra": 0})
        # 只有 standard 有运行时基准；lega 无条目（值为 None）
        _write_json(os.path.join(tmp, "latency.json"), {"variants": {"standard": {
            "prefill_ms": {"256": {"1": 1.0}},
            "decode_ms_per_tok": {"256": 0.1},
            "efficiency": {
                "n_params": 100,
                "flops_per_tok": {"256": {"total_flops_per_tok": 1000}},
                "peak_mem_mb_prefill": {"256": 1.0}}}}})
        out = build_summary(tmp)
        assert len(out["rows"]) == 2
        assert "涨点-增幅权衡" in out["markdown"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] build_summary（latency 缺变体条目时安全跳过）")


def test_build_summary_reports_gate_overhead_per_head_and_flops():
    """门控开销必须逐头核算并对照额外 FLOPs，不能只给总参数差。"""
    tmp = tempfile.mkdtemp()
    try:
        _write_json(os.path.join(tmp, "eval_lega_seed1.json"), {
            "variant": "lega", "seed": 1, "val_ppl": 9.0, "val_loss": 2.0,
            "gate_mean": 0.5, "n_params": 10000, "n_gate_extra": 400,
            "n_gate_per_head": 100.0, "n_heads": 2, "n_layers": 2,
            "d_head": 16})
        _write_json(os.path.join(tmp, "latency.json"), {"variants": {"lega": {
            "prefill_ms": {"256": {"1": 1.0}},
            "decode_ms_per_tok": {"256": 0.1},
            "efficiency": {
                "n_params": 10000,
                "flops_per_tok": {"256": {
                    "total_flops_per_tok": 10000,
                    "gate_extra_flops_per_tok": 250}},
                "peak_mem_mb_prefill": {"256": 1.0}}}}})
        out = build_summary(tmp)
        row = next(r for r in out["rows"] if r["variant"] == "lega")
        assert row["n_gate_per_head"] == 100.0
        assert row["gate_param_pct"] == 4.0
        assert row["gate_flops_pct"] == 2.5
        md = out["markdown"]
        assert "门控开销核算" in md and "门控参数/头" in md
        assert "100" in md and "4.0000%" in md and "2.5000%" in md
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] build_summary（逐头门控参数 + 额外 FLOPs 占比披露）")


def test_build_summary_rejects_malformed_latency_values():
    """latency.json 里的字符串/布尔计时不能参与相对基线比较或崩溃汇总。"""
    tmp = tempfile.mkdtemp()
    try:
        for v in ("standard", "lega"):
            _write_json(os.path.join(tmp, f"eval_{v}_seed1.json"), {
                "variant": v, "seed": 1, "val_ppl": 10.0, "val_loss": 2.3,
                "gate_mean": 0.5 if v == "lega" else None,
                "n_params": 100, "n_gate_extra": 0})
        _write_json(os.path.join(tmp, "latency.json"), {"variants": {
            "standard": {"prefill_ms": {"256": {"1": "bad"}},
                         "decode_ms_per_tok": {"256": True}},
            "lega": {"prefill_ms": {"256": {"1": 2.0}},
                     "decode_ms_per_tok": {"256": 0.2}}}})
        out = build_summary(tmp)
        row = next(r for r in out["rows"] if r["variant"] == "lega")
        assert row.get("prefill_ms") is None, row
        assert row.get("prefill_rel_pct") is None, row
        assert row.get("decode_ms_per_tok") is None, row
        # 有限但比例不可表示的计时（baseline=1ms、候选 1e308ms）也不能
        # 变成 inf；同时 PPL 相对收益不可表示时 -None 不能抛 TypeError。
        _write_json(os.path.join(tmp, "latency.json"), {"variants": {
            "standard": {"prefill_ms": {"256": {"1": 1.0}},
                         "decode_ms_per_tok": {"256": 1.0}},
            "lega": {"prefill_ms": {"256": {"1": 1e308}},
                     "decode_ms_per_tok": {"256": 1e308}}}})
        out2 = build_summary(tmp)
        row2 = next(r for r in out2["rows"] if r["variant"] == "lega")
        assert row2.get("prefill_rel_pct") is None, row2
        assert row2.get("eff_prefill_delta_pct") is None, row2
        assert row2.get("eff_benefit_ppl_pct") is not None and math.isfinite(
            row2["eff_benefit_ppl_pct"]), row2
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] build_summary（畸形延迟值按缺失处理）")


def test_build_summary_surfaces_loss_spike_diagnosis():
    """瞬态损失尖峰（分段均值会掩盖）必须进入 summary.md 的显著提醒，
    不能只躺在 curve_diagnosis.json 里。"""
    tmp = tempfile.mkdtemp()
    try:
        for v in ("standard", "lega"):
            _write_json(os.path.join(tmp, f"eval_{v}_seed1.json"), {
                "variant": v, "seed": 1, "val_ppl": 10.0, "val_loss": 2.3,
                "gate_mean": None, "n_params": 100, "n_gate_extra": 0})
        base = {"status": "正常收敛", "evidence": [], "recommendations": ["x"],
                "nonfinite_values": 0, "malformed_records": 0,
                "has_grad_spike": False, "has_loss_spike": False,
                "train_loss_first": 4.0, "train_loss_last": 3.0,
                "val_loss_first": 4.1, "val_loss_best": 3.4,
                "val_best_step": 50, "val_loss_last": 3.4,
                "early_stop_gain_rel": 0.0}
        spike = dict(base, has_loss_spike=True,
                     recommendations=["损失尖峰（×9 @step 80）：先核对批次"])
        _write_json(os.path.join(tmp, "curve_diagnosis.json"),
                    {"standard": {"1": base}, "lega": {"1": spike}})
        out = build_summary(tmp)
        md = out["markdown"]
        assert "损失瞬态尖峰" in md, md[-2000:]
        assert "（损失尖峰）" in md
        assert "损失尖峰（×9" in md
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] build_summary（瞬态损失尖峰进入汇总页显著提醒）")


def test_build_summary_surfaces_curve_observability_warnings():
    """验证/日志点过少是配置级限制：必须进入 summary，而不是只留在训练
    启动日志里；否则汇总页的趋势/早停结论会被误读为可靠。"""
    tmp = tempfile.mkdtemp()
    try:
        for v in ("standard", "lega"):
            _write_json(os.path.join(tmp, f"eval_{v}_seed1.json"), {
                "variant": v, "seed": 1, "val_ppl": 10.0, "val_loss": 2.3,
                "gate_mean": None, "n_params": 100, "n_gate_extra": 0})
        base = {"status": "正常收敛", "evidence": [], "recommendations": ["x"],
                "nonfinite_values": 0, "malformed_records": 0,
                "has_grad_spike": False, "has_loss_spike": False,
                "train_loss_first": 4.0, "train_loss_last": 3.0,
                "val_loss_first": 4.1, "val_loss_best": 3.4,
                "val_best_step": 50, "val_loss_last": 3.4,
                "early_stop_gain_rel": 0.0,
                "curve_observability_warnings": [
                    "验证点预计仅 2 个（steps=20, eval_every=10）"]}
        _write_json(os.path.join(tmp, "curve_diagnosis.json"),
                    {"standard": {"1": base}, "lega": {"1": dict(base)}})
        md = build_summary(tmp)["markdown"]
        assert "曲线采样可观测性受限" in md, md[-2000:]
        assert "验证点预计仅 2 个" in md, md[-2000:]
        assert "eval_every" in md, md[-2000:]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] build_summary（曲线采样不足告警进入汇总页）")


def test_build_summary_gate_health_overflow_safe():
    """门控健康比例是有限但极大的外部值时，汇总不能写出 inf%。"""
    tmp = tempfile.mkdtemp()
    try:
        for v in ("standard", "lega"):
            _write_json(os.path.join(tmp, f"eval_{v}_seed1.json"), {
                "variant": v, "seed": 1, "val_ppl": 10.0, "val_loss": 2.3,
                "gate_mean": None, "n_params": 100, "n_gate_extra": 0})
        base = {"status": "正常收敛", "evidence": [], "recommendations": ["x"],
                "nonfinite_values": 0, "malformed_records": 0,
                "has_grad_spike": False, "has_loss_spike": False,
                "train_loss_first": 4.0, "train_loss_last": 3.0,
                "val_loss_first": 4.1, "val_loss_best": 3.4,
                "val_best_step": 50, "val_loss_last": 3.4,
                "early_stop_gain_rel": 0.0,
                "gate_health": {
                    "available": True, "status": "健康", "mean_last": 0.5,
                    "std_last": 0.1, "frac_low_last": 1e307,
                    "frac_high_last": 1e307, "layer_mean_last": [0.5],
                    "recommendations": ["x"]}}
        _write_json(os.path.join(tmp, "curve_diagnosis.json"),
                    {"standard": {"1": base}, "lega": {"1": dict(base)}})
        md = build_summary(tmp)["markdown"]
        assert "溢出" in md and "inf" not in md
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] build_summary：门控健康比例溢出安全降级")


def test_build_summary_early_stop_overflow_safe():
    """早停收益是有限但极大的外部值时，汇总不能写出 +inf%。"""
    tmp = tempfile.mkdtemp()
    try:
        for v in ("standard", "lega"):
            _write_json(os.path.join(tmp, f"eval_{v}_seed1.json"), {
                "variant": v, "seed": 1, "val_ppl": 10.0, "val_loss": 2.3,
                "gate_mean": None, "n_params": 100, "n_gate_extra": 0})
        base = {"status": "正常收敛", "evidence": [], "recommendations": ["x"],
                "nonfinite_values": 0, "malformed_records": 0,
                "has_grad_spike": False, "has_loss_spike": False,
                "train_loss_first": 4.0, "train_loss_last": 3.0,
                "val_loss_first": 4.1, "val_loss_best": 3.4,
                "val_best_step": 50, "val_loss_last": 3.4,
                "early_stop_gain_rel": 1e307}
        _write_json(os.path.join(tmp, "curve_diagnosis.json"),
                    {"standard": {"1": base}, "lega": {"1": dict(base)}})
        md = build_summary(tmp)["markdown"]
        assert "溢出" in md and "inf" not in md
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] build_summary：早停收益溢出安全降级")


def test_render_inference_shapley_overflow_safe():
    """推理期 Shapley 占比乘 100 溢出时不能渲染出 inf%。"""
    md = "\n".join(_render_inference_shapley([{
        "shapley_factor": "A", "shapley_factor_label": "A",
        "mean_diff": 1.0, "shapley_ppl_ratio": 1.0,
        "shapley_share": 1e307, "n": 3, "p_adj": 0.1,
        "training_phi_mean": 1.0, "training_inference_sign_agree": True,
        "verdict": "x"}]))
    assert "溢出" in md and "inf" not in md
    print("  [ok] 推理期 Shapley：占比溢出安全降级")


def test_latency_noise_helpers():
    """测量噪声读取与两段散布的方和根合成：缺失安全。"""
    entry = {"prefill_ms_spread_pct": {"256": {"1": 6.0}},
             "decode_ms_spread_pct": {"256": 4.0}}
    assert _lat_noise_pct(entry, "prefill", 256) == 6.0
    assert _lat_noise_pct(entry, "decode", 256) == 4.0
    assert _lat_noise_pct(entry, "prefill", 512) is None
    assert _lat_noise_pct({}, "decode", 256) is None
    assert _lat_noise_pct({"decode_ms_spread_pct": {"256": "bad"}},
                          "decode", 256) is None
    assert _noise_floor_pct(3.0, 4.0) == 5.0
    assert _noise_floor_pct(6.0, None) is None
    assert _noise_floor_pct(None, None) is None
    print("  [ok] 运行时测量噪声读取与合成（缺失安全）")


def test_tradeoff_ignores_deltas_within_measurement_noise():
    """运行时差异不超过重复测量噪声时按不可区分处理：不计入增幅、标 †。"""
    tmp = tempfile.mkdtemp()
    try:
        for v in ("standard", "lega"):
            _write_json(os.path.join(tmp, f"eval_{v}_seed1.json"), {
                "variant": v, "seed": 1, "val_ppl": 10.0, "val_loss": 2.3,
                "gate_mean": 0.5 if v == "lega" else None,
                "n_params": 100, "n_gate_extra": 0})

        def _lat(prefill, dec):
            return {"prefill_ms": {"256": {"1": prefill}},
                    "prefill_ms_spread_pct": {"256": {"1": 30.0}},
                    "decode_ms_per_tok": {"256": dec},
                    "decode_ms_spread_pct": {"256": 25.0},
                    "efficiency": {
                        "n_params": 100,
                        "flops_per_tok": {"256": {"total_flops_per_tok": 1000}},
                        "peak_mem_mb_prefill": {"256": 1.0}}}

        # lega 的 prefill/decode 各慢 5%，均远小于噪声方和根（~39%/35%）
        _write_json(os.path.join(tmp, "latency.json"), {"variants": {
            "standard": _lat(1.0, 0.1), "lega": _lat(1.05, 0.105)}})
        out = build_summary(tmp)
        row = next(r for r in out["rows"] if r["variant"] == "lega")
        assert row["eff_prefill_within_noise"] is True
        assert row["eff_decode_within_noise"] is True
        assert row["eff_prefill_noise_pct"] is not None
        # 其余指标一致：噪声内的运行时差归零后最大增幅为 0，而不是 5%
        assert row["eff_cost_max_pct"] == 0.0, row
        assert "†" in out["markdown"] and "噪声区间内" in out["markdown"]

        # 噪声很大但真实差异也大（+300%）时必须照常计入增幅
        _write_json(os.path.join(tmp, "latency.json"), {"variants": {
            "standard": _lat(1.0, 0.1), "lega": _lat(1.0, 0.4)}})
        out2 = build_summary(tmp)
        row2 = next(r for r in out2["rows"] if r["variant"] == "lega")
        assert row2["eff_decode_within_noise"] is False
        assert row2["eff_cost_max_pct"] == 300.0, row2

        # 旧格式无噪声字段：差异全部计入
        _write_json(os.path.join(tmp, "latency.json"), {"variants": {
            "standard": {"prefill_ms": {"256": {"1": 1.0}},
                         "decode_ms_per_tok": {"256": 0.1},
                         "efficiency": {
                             "n_params": 100,
                             "flops_per_tok": {"256": {"total_flops_per_tok": 1000}},
                             "peak_mem_mb_prefill": {"256": 1.0}}},
            "lega": _lat(1.05, 0.105)}})
        out3 = build_summary(tmp)
        row3 = next(r for r in out3["rows"] if r["variant"] == "lega")
        assert row3["eff_prefill_within_noise"] is False
        assert row3["eff_decode_within_noise"] is False
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] 涨点-增幅权衡：噪声内运行时差按不可区分处理（旧格式兼容）")


def test_tradeoff_includes_decode_latency():
    """逐 token 解码指标（decode ms/tok）必须进入涨点-增幅权衡的增幅与判定，不能只看 prefill。"""
    tmp = tempfile.mkdtemp()
    try:
        for v in ("standard", "lega"):
            _write_json(os.path.join(tmp, f"eval_{v}_seed1.json"), {
                "variant": v, "seed": 1, "val_ppl": 10.0, "val_loss": 2.3,
                "gate_mean": 0.5 if v == "lega" else None,
                "n_params": 100, "n_gate_extra": 0})

        def _lat(dec):
            return {"prefill_ms": {"256": {"1": 1.0}},
                    "decode_ms_per_tok": {"256": dec},
                    "efficiency": {
                        "n_params": 100,
                        "flops_per_tok": {"256": {"total_flops_per_tok": 1000}},
                        "peak_mem_mb_prefill": {"256": 1.0}}}

        _write_json(os.path.join(tmp, "latency.json"), {"variants": {
            "standard": _lat(0.1), "lega": _lat(0.4)}})
        out = build_summary(tmp)
        row = next(r for r in out["rows"] if r["variant"] == "lega")
        assert row["eff_decode_ms_per_tok"] == 0.4
        assert row["eff_decode_delta_pct"] == 300.0
        # 其余指标与基线一致时，最大增幅由 decode 项贡献
        assert row["eff_cost_max_pct"] == 300.0, row
        assert "Decode ms/tok" in out["markdown"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] 涨点-增幅权衡包含 decode 项")


def test_tradeoff_includes_kv_cache_memory():
    """KV cache 显存必须进入效率-精度权衡：长上下文内存开销不能只看参数/
    单次激活峰值；旧格式只有 per-token 时按 ctx 换算。"""
    tmp = tempfile.mkdtemp()
    try:
        for v in ("standard", "lega"):
            _write_json(os.path.join(tmp, f"eval_{v}_seed1.json"), {
                "variant": v, "seed": 1, "val_ppl": 10.0, "val_loss": 2.3,
                "gate_mean": 0.5 if v == "lega" else None,
                "n_params": 100, "n_gate_extra": 0})

        def _lat(kv_per_tok, kv_at_ctx=None):
            eff = {"n_params": 100,
                   "flops_per_tok": {"256": {"total_flops_per_tok": 1000}},
                   "peak_mem_mb_prefill": {"256": 1.0},
                   "kv_cache_mb_per_tok": kv_per_tok}
            if kv_at_ctx is not None:
                eff["kv_cache_mb_at_ctx"] = {"256": kv_at_ctx}
            return {"prefill_ms": {"256": {"1": 1.0}},
                    "decode_ms_per_tok": {"256": 0.1},
                    "efficiency": eff}

        _write_json(os.path.join(tmp, "latency.json"), {"variants": {
            "standard": _lat(0.001, kv_at_ctx=0.256),
            "lega": _lat(0.002, kv_at_ctx=0.512)}})
        out = build_summary(tmp)
        row = next(r for r in out["rows"] if r["variant"] == "lega")
        assert abs(row["eff_kv_cache_mb"] - 0.512) < 1e-9, row
        assert row["eff_kv_cache_delta_pct"] == 100.0, row
        assert row["eff_cost_max_pct"] == 100.0, row
        assert "KV cache" in out["markdown"] and "+100.0%" in out["markdown"]

        # 只有 per-token（旧 latency.json）时按公共 ctx 线性换算
        _write_json(os.path.join(tmp, "latency.json"), {"variants": {
            "standard": _lat(0.001), "lega": _lat(0.002)}})
        out2 = build_summary(tmp)
        row2 = next(r for r in out2["rows"] if r["variant"] == "lega")
        assert abs(row2["eff_kv_cache_mb"] - 0.512) < 1e-9, row2
        assert row2["eff_cost_max_pct"] == 100.0, row2
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] 涨点-增幅权衡包含 KV cache 显存（旧格式按 ctx 换算）")


def test_tradeoff_includes_train_step_time():
    """训练 ms/step 必须进入综合增幅与判定：其余指标不变时也要被计入。"""
    tmp = tempfile.mkdtemp()
    try:
        for v, tms in (("standard", 100.0), ("lega", 1000.0)):
            _write_json(os.path.join(tmp, f"eval_{v}_seed1.json"), {
                "variant": v, "seed": 1, "val_ppl": 10.0, "val_loss": 2.3,
                "gate_mean": 0.5 if v == "lega" else None,
                "n_params": 100, "n_gate_extra": 0,
                "train_step_time_ms": tms,
                "train_tokens_per_sec": 1000.0 / tms})

        def _lat():
            return {"prefill_ms": {"256": {"1": 1.0}},
                    "decode_ms_per_tok": {"256": 0.1},
                    "efficiency": {
                        "n_params": 100,
                        "flops_per_tok": {"256": {"total_flops_per_tok": 1000}},
                        "peak_mem_mb_prefill": {"256": 1.0}}}

        _write_json(os.path.join(tmp, "latency.json"), {"variants": {
            "standard": _lat(), "lega": _lat()}})
        out = build_summary(tmp)
        row = next(r for r in out["rows"] if r["variant"] == "lega")
        assert row["eff_train_step_ms"] == 1000.0, row
        assert row["eff_train_delta_pct"] == 900.0
        # 其余指标与基线一致时，最大增幅由训练 ms/step 贡献
        assert row["eff_cost_max_pct"] == 900.0, row
        assert "训练 ms/step" in out["markdown"]
        # 旧结果无该字段时优雅降级为不判定，不崩溃也不编造增幅
        _write_json(os.path.join(tmp, "eval_lega_seed1.json"), {
            "variant": "lega", "seed": 1, "val_ppl": 10.0, "val_loss": 2.3,
            "gate_mean": 0.5, "n_params": 100, "n_gate_extra": 0})
        out2 = build_summary(tmp)
        row2 = next(r for r in out2["rows"] if r["variant"] == "lega")
        assert row2["eff_train_step_ms"] is None
        assert row2["eff_train_delta_pct"] is None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] 涨点-增幅权衡包含训练 ms/step（旧结果缺失时降级）")


def test_tradeoff_includes_train_peak_memory():
    """训练进程峰值必须进入涨点-增幅权衡的增幅项，不能只看
    推理时序/参数量；CPU 训练缺该字段时安全降级。"""
    tmp = tempfile.mkdtemp()
    try:
        for v, mem in (("standard", 1000.0), ("lega", 3000.0)):
            _write_json(os.path.join(tmp, f"eval_{v}_seed1.json"), {
                "variant": v, "seed": 1, "val_ppl": 10.0, "val_loss": 2.3,
                "gate_mean": 0.5 if v == "lega" else None,
                "n_params": 100, "n_gate_extra": 0,
                "train_peak_mem_mb": mem})

        def _lat():
            return {"prefill_ms": {"256": {"1": 1.0}},
                    "decode_ms_per_tok": {"256": 0.1},
                    "efficiency": {
                        "n_params": 100,
                        "flops_per_tok": {"256": {"total_flops_per_tok": 1000}},
                        "peak_mem_mb_prefill": {"256": 1.0}}}

        _write_json(os.path.join(tmp, "latency.json"), {"variants": {
            "standard": _lat(), "lega": _lat()}})
        out = build_summary(tmp)
        row = next(r for r in out["rows"] if r["variant"] == "lega")
        assert row["eff_train_peak_mem_mb"] == 3000.0, row
        assert row["eff_train_mem_delta_pct"] == 200.0
        # 其余指标与基线一致时，最大增幅由训练进程峰值贡献
        assert row["eff_cost_max_pct"] == 200.0, row
        assert "训练峰值" in out["markdown"]
        # CPU 训练（无 CUDA 统计）：字段缺失时该项不参与判定
        _write_json(os.path.join(tmp, "eval_lega_seed1.json"), {
            "variant": "lega", "seed": 1, "val_ppl": 10.0, "val_loss": 2.3,
            "gate_mean": 0.5, "n_params": 100, "n_gate_extra": 0})
        out2 = build_summary(tmp)
        row2 = next(r for r in out2["rows"] if r["variant"] == "lega")
        assert row2["eff_train_peak_mem_mb"] is None
        assert row2["eff_train_mem_delta_pct"] is None
        assert row2["eff_cost_max_pct"] == 0.0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] 涨点-增幅权衡包含训练进程峰值（缺失时降级）")


def test_summary_tolerates_huge_int_train_stats():
    """eval.json 里超出 float 范围的合法整数不能中断汇总。"""
    huge = 10 ** 400
    tmp = tempfile.mkdtemp()
    try:
        _write_json(os.path.join(tmp, "eval_standard_seed1.json"), {
            "variant": "standard", "seed": 1, "val_ppl": 10.0,
            "val_loss": 2.3, "gate_mean": 0.5, "n_params": 100,
            "n_gate_extra": 0, "train_step_time_ms": huge,
            "train_peak_mem_mb": huge})
        out = build_summary(tmp)
        row = next(r for r in out["rows"] if r["variant"] == "standard")
        assert row.get("eff_train_step_ms") is None, row
        assert row.get("eff_train_peak_mem_mb") is None, row
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    # 多个种子都是合法但接近上限的有限计时：偶数个中位数的“先加后除”
    # 会溢出成 inf，汇总字段必须仍是有限值。
    tmp2 = tempfile.mkdtemp()
    try:
        for v in ("standard", "lega"):
            for seed in (1, 2):
                _write_json(os.path.join(tmp2, f"eval_{v}_seed{seed}.json"), {
                    "variant": v, "seed": seed, "val_ppl": 10.0,
                    "val_loss": 2.3, "gate_mean": 0.5, "n_params": 100,
                    "n_gate_extra": 0, "train_step_time_ms": 1e308,
                    "train_peak_mem_mb": 1e308, "split": "val"})
        _write_json(os.path.join(tmp2, "latency.json"), {"variants": {
            "standard": {"prefill_ms": {"256": {"1": 1.0}},
                         "decode_ms_per_tok": {"256": 1.0}},
            "lega": {"prefill_ms": {"256": {"1": 2.0}},
                     "decode_ms_per_tok": {"256": 2.0}}}})
        out2 = build_summary(tmp2)
        row2 = next(r for r in out2["rows"] if r["variant"] == "lega")
        assert row2["eff_train_step_ms"] == 1e308, row2
        assert row2["eff_train_peak_mem_mb"] == 1e308, row2
        assert math.isfinite(row2["eff_train_step_ms"]), row2
    finally:
        shutil.rmtree(tmp2, ignore_errors=True)
    print("  [ok] 训练 ms/step 与峰值内存的超大整数/上限值安全降级")


def test_summary_tolerates_huge_int_tuning_loo_counts():
    """调优报告的留一折计数字段是合法大整数时，比例渲染不能溢出。"""
    tmp = tempfile.mkdtemp()
    try:
        _write_json(os.path.join(tmp, "baseline_tuning.json"), {
            "variant": "standard", "steps": 10, "selection_seeds": [1, 2],
            "selected": {"lr": 1.0, "val_loss_best": 1.0},
            "default_trial": {"lr": 1.0, "val_loss_best": 1.0},
            "selection_diagnostics": {
                "n_seeds": 2, "selected_support_count": 1,
                "leave_one_seed_out": {"n_folds": 10 ** 400,
                                       "agreement_count": 1}}})
        lines = _render_baseline_sections(tmp, smoke=False)
        assert any("留一种子复选" in line for line in lines), lines
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] 调优报告的超大留一折计数安全降级")


def test_tradeoff_table_surfaces_ruler_benefit():
    """权衡表必须输出 RULER 收益列；PPL 持平而 RULER 显著提升时判定不能
    仍写“说服力有限”（长上下文检索是本项目预注册的核心主张）。"""
    tmp = tempfile.mkdtemp()
    try:
        seeds = [1, 2, 3]
        _write_json(os.path.join(tmp, "repro.json"),
                    {"variants": ["standard", "lega"], "seeds": seeds,
                     "smoke": False})
        lega_ppls = {1: 9.0, 2: 10.0, 3: 10.9}
        for s in seeds:
            for v in ("standard", "lega"):
                _write_json(os.path.join(tmp, f"eval_{v}_seed{s}.json"), {
                    "variant": v, "seed": s,
                    "val_ppl": 10.0 if v == "standard" else lega_ppls[s],
                    "val_loss": 2.3,
                    "gate_mean": 0.5 if v == "lega" else None,
                    "n_params": 100, "n_gate_extra": 0})
        for s in seeds:
            for v, em in (("standard", 0.40), ("lega", 0.43)):
                _write_json(os.path.join(tmp, f"ruler_{v}_seed{s}.json"), {
                    "variant": v, "seed": s,
                    "results": {"niah_synth": {"128": {"em": em}}}})

        def _lat():
            return {"prefill_ms": {"256": {"1": 1.0}},
                    "decode_ms_per_tok": {"256": 0.1},
                    "efficiency": {
                        "n_params": 100,
                        "flops_per_tok": {"256": {"total_flops_per_tok": 1000}},
                        "peak_mem_mb_prefill": {"256": 1.0}}}

        _write_json(os.path.join(tmp, "latency.json"), {"variants": {
            "standard": _lat(), "lega": _lat()}})
        out = build_summary(tmp)
        row = next(r for r in out["rows"] if r["variant"] == "lega")
        assert row["eff_benefit_ruler_pp"] == 3.0, row
        assert row["eff_ruler_p_adj"] is not None \
            and row["eff_ruler_p_adj"] < ALPHA, row
        assert row["eff_verdict"].startswith("可接受") \
            and "RULER" in row["eff_verdict"], row
        md = out["markdown"]
        assert "RULER Δ(pp)" in md and "+3.00 pp" in md, md[-3000:]
        assert "p_adj(RULER)" in md
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] 涨点-增幅权衡：RULER 收益列与判定依据（PPL 持平不误判）")


def test_audit_tolerates_early_stop_actual_steps():
    """启用早停时实际步数/tokens_seen 可以逐 run 不同（早停的定义），审计按
    计划步数核对并如实报告；未标记早停的步数差异仍是不可比错误。"""
    tmp = tempfile.mkdtemp()
    try:
        variants, seeds = ["standard", "lega"], [7]
        for v in variants:
            d = os.path.join(tmp, v)
            os.makedirs(d)
            _write_json(os.path.join(d, "config_used.json"), {
                "lr": 0.001, "steps": 10, "warmup": 2,
                "early_stop_patience": 2,
                "variant": v, "seed": 7, "device": "cpu", "tag": "t"})
        _write_json(os.path.join(tmp, "standard", "eval.json"), {
            "variant": "standard", "seed": 7, "data_fingerprint": "abc123",
            "code_commit": "deadbeef", "config_sha256": "shax",
            "steps": 10, "steps_planned": 10, "early_stopped": False,
            "tokens_seen": 1024})
        _write_json(os.path.join(tmp, "lega", "eval.json"), {
            "variant": "lega", "seed": 7, "data_fingerprint": "abc123",
            "code_commit": "deadbeef", "config_sha256": "shax",
            "steps": 6, "steps_planned": 10, "early_stopped": True,
            "tokens_seen": 614})
        audit = verify_controlled_setup(tmp, variants, seeds, 1,
                                        expected_data_digest="abc123")
        assert audit["ok"], audit
        assert len(audit["early_stop_runs"]) == 1, audit
        assert "lega/seed7" in audit["early_stop_runs"][0]
        assert audit["training_budget_comparable"] is False, audit

        # 计划步数不一致仍是错误（早停不能掩盖"计划就不同"）
        ev = json.load(open(os.path.join(tmp, "lega", "eval.json"),
                            encoding="utf-8"))
        ev["steps_planned"] = 12
        _write_json(os.path.join(tmp, "lega", "eval.json"), ev)
        try:
            verify_controlled_setup(tmp, variants, seeds, 1)
            raise AssertionError("计划步数不一致应触发 SystemExit")
        except SystemExit:
            pass

        # 未标记早停（early_stopped 缺失/False）的实际步数差异必须报错
        ev["steps_planned"] = 10
        ev["early_stopped"] = False
        _write_json(os.path.join(tmp, "lega", "eval.json"), ev)
        try:
            verify_controlled_setup(tmp, variants, seeds, 1)
            raise AssertionError("非早停的实际步数不一致应触发 SystemExit")
        except SystemExit:
            pass

        # 非早停时 tokens_seen 不一致同样报错
        ev = json.load(open(os.path.join(tmp, "lega", "eval.json"),
                            encoding="utf-8"))
        ev["steps"] = 10
        _write_json(os.path.join(tmp, "lega", "eval.json"), ev)
        try:
            verify_controlled_setup(tmp, variants, seeds, 1)
            raise AssertionError("tokens_seen 不一致应触发 SystemExit")
        except SystemExit:
            pass
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] 受控变量审计：早停按计划步数核对，非早停差异仍报错")


def test_factorial_interaction_contrast_math():
    """2^3 交互对比公式：A×B=2 时 A×C/B×C 应为 0，且标记为无方向检验。"""
    base = {"none": 0.0, "A": 1.0, "B": 2.0, "C": 4.0,
            "AB": 5.0, "AC": 5.0, "BC": 6.0, "ABC": 9.0}
    runs = {}
    for cell, variant in _FACTORIAL_CONTRAST_CELLS.items():
        runs[variant] = {s: {"eval": {"val_ppl": base[cell]}}
                         for s in (1, 2, 3)}
    out = _factorial_interaction_tests(runs)
    assert len(out) == 3, out
    ppl = {t["interaction"]: t for t in out if t["metric"] == "PPL交互"}
    assert len(ppl) == 3, out
    assert abs(ppl["A×B"]["mean_diff"] - 2.0) < 1e-9, ppl["A×B"]
    assert abs(ppl["A×C"]["mean_diff"]) < 1e-9, ppl["A×C"]
    assert abs(ppl["B×C"]["mean_diff"]) < 1e-9, ppl["B×C"]
    assert all(t.get("no_direction") for t in out), out
    incomplete = {k: v for k, v in runs.items() if k != "lega"}
    got = [t for t in _factorial_interaction_tests(incomplete)
           if t["metric"] == "PPL交互"]
    assert got == [], got
    print("  [ok] 全因子交互对比：标准 2^3 对比公式、无方向标记与不完整降级")


def test_factorial_interactions_exclude_protocol_mismatches():
    """交互对比用到全部 8 个格：任一格评测协议/单元不一致，该种子必须剔除；
    否则 split/上下文/RULER 题面差异会伪装成“交互效应”。"""
    base = {"none": 0.0, "A": 1.0, "B": 2.0, "C": 4.0,
            "AB": 5.0, "AC": 5.0, "BC": 6.0, "ABC": 9.0}
    runs = {}
    for cell, variant in _FACTORIAL_CONTRAST_CELLS.items():
        runs[variant] = {s: {"eval": {"val_ppl": base[cell],
                                      "split": "test", "eval_ctx": 512}}
                         for s in (1, 2, 3)}
    # 一个种子下某格的 split 不同：该种子的三条交互都不可用；另两个仍可检验
    runs["lega"][2]["eval"]["split"] = "val"
    out = [t for t in _factorial_interaction_tests(runs)
           if t["metric"] == "PPL交互"]
    assert len(out) == 3, out
    assert all(t["n"] == 2 for t in out), out
    assert all(t.get("excluded_protocol_mismatches") == 1 for t in out), out
    # 所有种子都混入另一协议：不产出交互检验，不能静默给出显著性
    for slot in runs["lega"].values():
        slot["eval"]["split"] = "val"
    assert [t for t in _factorial_interaction_tests(runs)
            if t["metric"] == "PPL交互"] == []
    # RULER 已评测单元集合不一致（缺格）同样不可配对
    for variant, slots in runs.items():
        for slot in slots.values():
            slot["eval"]["split"] = "test"
            slot["ruler"] = {"ctx_lens": [256], "n_samples": 4,
                             "results": {"niah_synth": {
                                 "256": {"em": 1.0, "n": 4}}}}
    for slot in runs["lega"].values():
        slot["ruler"]["results"] = {}
    out = _factorial_interaction_tests(runs)
    assert len([t for t in out if t["metric"] == "PPL交互"]) == 3, out
    assert [t for t in out if t["metric"] == "RULER平均交互"] == [], out
    # 恢复 RULER 单元覆盖后，仅让一个种子下某格 n 不同：该种子的交互剔除
    for variant, slots in runs.items():
        for slot in slots.values():
            slot["ruler"]["results"] = {"niah_synth": {
                "256": {"em": 1.0, "n": 4}}}
    runs["lega"][2]["ruler"]["results"]["niah_synth"]["256"]["n"] = 3
    out = [t for t in _factorial_interaction_tests(runs)
           if t["metric"] == "RULER平均交互"]
    assert len(out) == 3, out
    assert all(t["n"] == 2 for t in out), out
    assert all(t.get("excluded_protocol_mismatches") == 1 for t in out), out
    print("  [ok] 全因子交互剔除评测协议/单元/样本数不一致的种子")


def test_budget_gate_downgrades_confirmatory_on_early_stop():
    """早停导致实际训练量不等：确认性判定降级，已降级的保留原原因。"""
    hyp = [{"id": "H1", "status": "支持", "confirmatory_valid": True},
           {"id": "H2", "status": "未定", "confirmatory_valid": False,
            "confirmatory_note": "原有原因"}]
    out = apply_budget_gate(hyp, {
        "early_stop_runs": ["lega/seed7(step 6/10)"]})
    assert out[0]["confirmatory_valid"] is False, out
    assert "早停" in out[0]["confirmatory_note"], out
    assert out[1]["confirmatory_note"] == "原有原因", out
    unchanged = [{"id": "H1", "confirmatory_valid": True}]
    assert apply_budget_gate(unchanged, {"early_stop_runs": []})[0][
        "confirmatory_valid"] is True
    print("  [ok] 早停门：确认性判定降级为探索性，不覆盖既有降级原因")


def test_build_summary_marks_early_stop_run():
    """早停 run 必须在汇总页被显式标注（曲线按设计截断），且审计小节说明
    实际步数差异的核对口径。"""
    tmp = tempfile.mkdtemp()
    try:
        for v in ("standard", "lega"):
            _write_json(os.path.join(tmp, f"eval_{v}_seed1.json"), {
                "variant": v, "seed": 1, "val_ppl": 10.0, "val_loss": 2.3,
                "gate_mean": None, "n_params": 100, "n_gate_extra": 0})
        base = {"status": "过拟合趋势", "evidence": ["x"], "recommendations": ["y"],
                "nonfinite_values": 0, "malformed_records": 0,
                "has_grad_spike": False, "has_loss_spike": False,
                "train_loss_first": 4.0, "train_loss_last": 3.0,
                "val_loss_first": 4.1, "val_loss_best": 3.4,
                "val_best_step": 50, "val_loss_last": 3.4,
                "early_stop_gain_rel": 0.0, "early_stopped": False,
                "early_stop_step": None}
        stopped = dict(base, early_stopped=True, early_stop_step=500)
        _write_json(os.path.join(tmp, "curve_diagnosis.json"),
                    {"standard": {"1": base}, "lega": {"1": stopped}})
        audit = {"ok": True, "checked": 2, "missing_configs": [],
                 "mismatches": [], "meta_checked": 2, "missing_evals": [],
                 "meta_mismatches": [], "expected_data_digest": "abc123",
                 "exempt_keys": ["variant"], "early_stop_runs": ["lega/seed1(step 500/8000)"]}
        out = build_summary(tmp, audit=audit)
        md = out["markdown"]
        assert "（早停）" in md, md
        assert "1/1 种子需关注" in md, "需关注种子占比必须显式标出"
        assert "早停启用" in md, md
        assert "按计划步数核对" in md, md
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] 汇总页：早停标记 + 审计口径说明")


def test_paired_ppl_skips_seeds_with_mismatched_eval_split():
    """PPL 配对检验不能把 val 与 test 的种子配成一对：split 不同整对剔除。"""
    def ev(ppl, split):
        return {"eval": {"ppl": ppl, "split": split}}

    runs = {
        "lega": {7: ev(10.0, "val"), 8: ev(11.0, "test"),
                 9: ev(12.0, "val")},
        "standard": {7: ev(20.0, "val"), 8: ev(30.0, "val"),
                     9: ev(25.0, "test")},
    }
    out = _paired_comparisons(
        runs, [("lega", "standard")], "PPL",
        lambda slot: _ppl_of(slot.get("eval")), lower_better=True)
    assert len(out) == 1
    t = out[0]
    # 只有 seed7 的双方同为 val；seed8/9 的 split 错位被剔除
    assert t["n"] == 1
    assert abs(t["mean_diff"] - (10.0 - 20.0)) < 1e-12
    assert t.get("excluded_split_mismatches") == 2
    print("  [ok] PPL 配对：跨 eval split 的种子整对剔除")


def test_log_ppl_sensitivity_can_reverse_arithmetic_verdict():
    """PPL 是比值尺度：log-PPL 与算术 PPL 的均值结论可能方向相反。"""
    def ev(ppl, **kw):
        base = {"ppl": ppl, "split": "val", "eval_ctx": 512,
                "val_windows": 80}
        base.update(kw)
        return {"eval": base}

    # 算术差 +99, -100 → 均值 -0.5（候选更好）；
    # log 差 +4.605, -0.105 → 均值 +2.25（候选更差）。
    runs = {
        "lega": {1: ev(100.0), 2: ev(900.0)},
        "standard": {1: ev(1.0), 2: ev(1000.0)},
    }
    raw = _paired_comparisons(
        runs, [("lega", "standard")], "PPL",
        lambda slot: _ppl_of(slot.get("eval")), lower_better=True)[0]
    log = _paired_comparisons(
        runs, [("lega", "standard")], "PPL(log)",
        lambda slot: _log_ppl_of(slot.get("eval")), lower_better=True)[0]
    assert raw["mean_diff"] < 0 < log["mean_diff"], (raw, log)
    assert abs(raw["mean_diff"] - (-0.5)) < 1e-12, raw
    assert abs(log["mean_diff"] - (math.log(100.0) + math.log(0.9)) / 2) < 1e-12
    # log 指标仍走 PPL 的 split/ctx/窗口数协议门槛（不能跨评测协议配对）
    log2 = _paired_comparisons(
        {"lega": {1: ev(100.0, eval_ctx=256)},
         "standard": {1: ev(10.0)}},
        [("lega", "standard")], "PPL(log)",
        lambda slot: _log_ppl_of(slot.get("eval")), lower_better=True)
    assert log2 == [], log2
    assert _log_ppl_of({"ppl": 0}) is None
    assert _log_ppl_of({"ppl": -1}) is None
    # 负 PPL 不可能是真实观测，必须按缺失处理；0 是历史合成夹具的边界
    # 值，保留兼容；极小正值仍是合法观测。
    assert _ppl_of({"ppl": -1}) is None
    assert _ppl_of({"val_ppl": -1.0}) is None
    assert _ppl_of({"ppl": 0}) == 0.0
    assert _ppl_of({"ppl": 1e-300}) == 1e-300
    print("  [ok] log-PPL 敏感性：几何均值比可反转算术 PPL 的均值方向，协议门槛一致")


def test_paired_ppl_skips_eval_protocol_mismatch():
    """同 split 不同 ctx/窗口数仍是不同 token 子集，不能充当配对观测。"""
    def ev(ppl, ctx=512, windows=80):
        return {"eval": {"ppl": ppl, "split": "val", "eval_ctx": ctx,
                         "val_windows": windows}}

    runs = {
        "lega": {7: ev(10.0), 8: ev(11.0, ctx=256), 9: ev(12.0, windows=40)},
        "standard": {7: ev(20.0), 8: ev(30.0), 9: ev(25.0)},
    }
    out = _paired_comparisons(
        runs, [("lega", "standard")], "PPL",
        lambda slot: _ppl_of(slot.get("eval")), lower_better=True)
    assert len(out) == 1, out
    assert out[0]["n"] == 1, out
    assert out[0].get("excluded_protocol_mismatches") == 2, out
    # 一侧缺协议字段不是“共同字段一致”：无法证明用了同一批 token，整对剔除
    runs_missing = {
        "lega": {7: ev(10.0)},
        "standard": {7: {"eval": {"ppl": 20.0, "split": "val",
                                  "eval_ctx": 512}}},
    }
    out_missing = _paired_comparisons(
        runs_missing, [("lega", "standard")], "PPL",
        lambda slot: _ppl_of(slot.get("eval")), lower_better=True)
    assert out_missing == [], out_missing
    print("  [ok] PPL 配对：同 split 但 ctx/窗口数不一致或缺协议字段的种子整对剔除")


def test_build_summary_renders_log_ppl_sensitivity():
    """汇总页必须报告 log-PPL 几何均值比及其与原始 PPL 判定的一致性。"""
    tmp = tempfile.mkdtemp()
    try:
        pairs = [(100.0, 1.0), (900.0, 1000.0),
                 (100.0, 1.0), (900.0, 1000.0)]
        for seed, (lega_ppl, std_ppl) in enumerate(pairs, 1):
            _write_json(os.path.join(tmp, f"eval_lega_seed{seed}.json"), {
                "variant": "lega", "seed": seed, "val_ppl": lega_ppl,
                "val_loss": math.log(lega_ppl), "gate_mean": 0.5,
                "n_params": 100, "n_gate_extra": 0})
            _write_json(os.path.join(tmp, f"eval_standard_seed{seed}.json"), {
                "variant": "standard", "seed": seed, "val_ppl": std_ppl,
                "val_loss": math.log(std_ppl), "gate_mean": None,
                "n_params": 100, "n_gate_extra": 0})
        out = build_summary(tmp)
        md = out["markdown"]
        assert "PPL 比值尺度敏感性" in md, md
        assert "PPL几何比" in md, md
        log_tests = [t for t in out["tests"]
                     if t.get("metric") == "PPL(log)"]
        assert log_tests, out["tests"]
        t = log_tests[0]
        assert t.get("ppl_ratio") is not None
        assert abs(t["ppl_ratio"] - math.exp(t["mean_diff"])) < 1e-12
        assert t.get("ppl_ratio_ci95") is not None
        raw = [x for x in out["tests"] if x.get("metric") == "PPL"][0]
        assert "PPL(log)" not in md.split("## 关键对比")[1].split("### PPL")[0]
        assert raw["mean_diff"] < 0 < t["mean_diff"], (raw, t)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] 汇总页：log-PPL 几何均值比与原始 PPL 方向分歧显式呈现")


def test_paired_ruler_skips_protocol_and_cell_mismatch():
    """RULER 配对不能跨题面/抽样/生成预算或不同评测格集合比较 EM。"""
    def rul(em, **kw):
        base = {
            "ctx_lens": [256, 512], "n_samples": 30, "sample_seed": 1234,
            "corpus_source": "val", "max_new": 12, "fwe_max_new": 44,
            "vt_chains": 4, "vt_depth": 4,
            "results": {"niah": {"256": {"em": em, "n": 30},
                                 "512": {"em": em, "n": 30}}}}
        base.update(kw)
        return {"ruler": base}

    runs = {
        "lega": {7: rul(0.8), 8: rul(0.9, n_samples=20), 9: rul(0.7)},
        "standard": {7: rul(0.5), 8: rul(0.5), 9: rul(0.5, max_new=44)},
    }
    out = _paired_comparisons(
        runs, [("lega", "standard")], "RULER平均",
        lambda slot: _ruler_avg_of(slot.get("ruler")), lower_better=False)
    assert len(out) == 1, out
    t = out[0]
    assert t["n"] == 1, t
    assert abs(t["mean_diff"] - 0.3) < 1e-12, t
    assert t.get("excluded_protocol_mismatches") == 2, t
    # 只覆盖一个上下文格的评测同样不可与两格 macro 平均配对
    runs2 = {
        "lega": {7: rul(0.8)},
        "standard": {7: rul(0.5, results={
            "niah": {"256": {"em": 0.5, "n": 30}}})},
    }
    out2 = _paired_comparisons(
        runs2, [("lega", "standard")], "RULER平均",
        lambda slot: _ruler_avg_of(slot.get("ruler")), lower_better=False)
    assert out2 == [], out2
    # 同一评测格但样本数不同：聚合 EM 不是同一批观测，不能配对
    runs3 = {
        "lega": {7: rul(0.8)},
        "standard": {7: rul(0.5, results={
            "niah": {"256": {"em": 0.5, "n": 15},
                     "512": {"em": 0.5, "n": 30}}})},
    }
    out3 = _paired_comparisons(
        runs3, [("lega", "standard")], "RULER平均",
        lambda slot: _ruler_avg_of(slot.get("ruler")), lower_better=False)
    assert out3 == [], out3
    # 一侧缺抽样种子同样不可配对；不能只比较双方都有的字段
    runs4 = {
        "lega": {7: rul(0.8)},
        "standard": {7: rul(0.5, sample_seed=None)},
    }
    out4 = _paired_comparisons(
        runs4, [("lega", "standard")], "RULER平均",
        lambda slot: _ruler_avg_of(slot.get("ruler")), lower_better=False)
    assert out4 == [], out4
    print("  [ok] RULER 配对：协议/评测格/单元样本数不一致或缺协议字段的种子整对剔除")


def test_paired_ruler_per_task_skips_protocol_and_cell_mismatch():
    """逐任务（RULER:<task>）配对必须与综合平均走同一套可比性门槛。"""
    def rul(em, **kw):
        base = {
            "ctx_lens": [256, 512], "n_samples": 30, "sample_seed": 1234,
            "corpus_source": "val", "max_new": 12, "fwe_max_new": 44,
            "vt_chains": 4, "vt_depth": 4,
            "results": {"niah": {"256": {"em": em, "n": 30},
                                 "512": {"em": em, "n": 30}}}}
        base.update(kw)
        return {"ruler": base}

    runs = {
        "lega": {7: rul(0.8), 8: rul(0.9, n_samples=20)},
        "standard": {7: rul(0.5), 8: rul(0.5)},
    }
    out = _paired_comparisons(
        runs, [("lega", "standard")], "RULER:niah",
        lambda slot: _ruler_task_means(slot.get("ruler")).get("niah"),
        lower_better=False)
    assert len(out) == 1, out
    assert out[0]["n"] == 1, out
    assert out[0].get("excluded_protocol_mismatches") == 1, out
    # 同协议但评测格样本数不同：逐任务也不能配对
    runs2 = {
        "lega": {7: rul(0.8)},
        "standard": {7: rul(0.5, results={
            "niah": {"256": {"em": 0.5, "n": 15},
                     "512": {"em": 0.5, "n": 30}}})},
    }
    out2 = _paired_comparisons(
        runs2, [("lega", "standard")], "RULER:niah",
        lambda slot: _ruler_task_means(slot.get("ruler")).get("niah"),
        lower_better=False)
    assert out2 == [], out2
    print("  [ok] RULER 逐任务配对：协议/评测格不一致的种子整对剔除")


def test_build_summary_marks_clamped_ppl_and_tolerates_null_distribution():
    """PPL 截断必须在汇总表显式标注；分布漂移字段为 null（空 split）时不得崩溃。"""
    tmp = tempfile.mkdtemp()
    try:
        _write_json(os.path.join(tmp, "eval_lega_seed1.json"), {
            "variant": "lega", "seed": 1, "val_ppl": 4.85e8, "val_loss": 21.0,
            "val_ppl_clamped": True, "gate_mean": 0.5, "n_params": 100,
            "n_gate_extra": 0})
        _write_json(os.path.join(tmp, "eval_standard_seed1.json"), {
            "variant": "standard", "seed": 1, "val_ppl": 10.0, "val_loss": 2.3,
            "gate_mean": None, "n_params": 100, "n_gate_extra": 0})
        out = build_summary(tmp, data_report={
            "distribution_shift": {
                "val_vs_train": {"tv": None, "js_bits": None},
                "test_vs_train": {"tv": None, "js_bits": None,
                                  "eval_oov_token_rate": None}}})
        md = out["markdown"]
        assert "含截断值" in md
        assert "TV=-" in md and "JS=-" in md
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] build_summary（PPL 截断标注；分布漂移 null 安全）")


def test_build_summary_split_fractions_do_not_emit_inf():
    """有限但极大的划分比例/偏差乘 100 会溢出，汇总不能写出 inf%。"""
    tmp = tempfile.mkdtemp()
    try:
        out = build_summary(tmp, data_report={"split_protocol": {
            "actual_fractions": {"train": 1e307, "val": 0.1, "test": 0.1},
            "target_fractions": {"train": 0.8, "val": 0.1, "test": 0.1},
            "max_abs_fraction_deviation": 1e307}})
        md = out["markdown"]
        assert "溢出" in md and "inf" not in md
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] build_summary：划分比例溢出安全降级")


def test_build_summary_surfaces_group_crossing_split_protocol():
    """群组跨切点必须渲染进汇总页（元数据行 + 告警），不允许静默通过。"""
    tmp = tempfile.mkdtemp()
    try:
        _write_json(os.path.join(tmp, "eval_lega_seed1.json"), {
            "variant": "lega", "seed": 1, "val_ppl": 10.0, "val_loss": 2.3,
            "gate_mean": 0.5, "n_params": 100, "n_gate_extra": 0})
        out = build_summary(tmp, data_report={
            "split_protocol": {
                "mode": "temporal",
                "order": "temporal:train(earliest)->val->test(latest)",
                "fallback_char_cut": False, "shuffled": False,
                "group_unit": "heading_document", "n_headings": 1,
                "documents_cross_cut": True},
            "alerts": [{"level": "warning",
                        "code": "split_documents_cross_cut",
                        "message": "heading_document 群组跨 split"}]})
        md = out["markdown"]
        assert "群组单位=heading_document" in md
        assert "文档跨切点=True" in md
        assert "split_documents_cross_cut" in md
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] build_summary：群组跨 split 进入汇总页（元数据 + 告警）")


def test_paired_effect_below_seed_noise_is_explicitly_marked():
    """|Δ|<ΔSD 的边缘结果不能只写“不显著”，必须显式标注噪声内。"""
    diffs = [-0.5, 0.7, -0.3, 0.6, 0.4]
    ref = [11.0, 12.0, 11.0, 13.0, 9.0]
    cand = [r + d for r, d in zip(ref, diffs)]
    runs = {
        "lega": {s + 1: {"eval": {"ppl": v, "split": "val"}}
                 for s, v in enumerate(cand)},
        "standard": {s + 1: {"eval": {"ppl": v, "split": "val"}}
                     for s, v in enumerate(ref)},
    }
    out = _apply_holm(_paired_comparisons(
        runs, [("lega", "standard")], "PPL",
        lambda slot: _ppl_of(slot.get("eval")), lower_better=True))
    t = out[0]
    assert t["within_noise"] is True, t
    assert t["effect_to_noise"] < 1.0, t
    assert "噪声" in t["verdict"], t
    print("  [ok] 效应/噪声：|Δ|<ΔSD 的边缘结果显式标注")


def test_enrich_paired_adds_robustness_fields():
    """配对检验必须附带置换 p / bootstrap CI / MDE，三者口径一致可复核。"""
    a = [10.5, 10.6, 10.4, 10.7, 10.55]
    b = [10.0, 10.05, 10.02, 10.03, 10.04]
    res = _enrich_paired({"n": 5, "mean_diff": 0.5, "std_diff": 0.1}, a, b)
    assert res["p_perm"] is not None and 0.0 <= res["p_perm"] <= 1.0
    assert res["perm_exact"] is True          # n=5 <= 16 → 全枚举精确
    assert res["median_diff"] is not None and res["p_sign"] is not None
    assert (res["sign_n_pos"] + res["sign_n_neg"]
            + res["sign_n_zero"]) == 5
    lo, hi = res["mean_diff_ci95"]
    assert lo <= res["mean_diff"] <= hi
    assert res["mde"] > 0
    # 无可检效应（差恒 0）→ 置换 p=1，MDE 仍按方差给出
    res0 = _enrich_paired({"n": 3, "mean_diff": 0.0, "std_diff": 0.2},
                           [1, 2, 3], [1, 2, 3])
    assert res0["p_perm"] == 1.0 and res0["mde"] > 0
    assert res0["median_diff"] == 0.0 and res0["p_sign"] == 1.0
    # n=1 / 无均值：字段为 None 而不是崩溃
    res1 = _enrich_paired({"n": 0, "mean_diff": None, "std_diff": None}, [], [])
    assert res1["p_perm"] is None and res1["mean_diff_ci95"] is None
    assert res1["median_diff"] is None and res1["p_sign"] is None
    # bool 配对观测与检验主体同口径跳过：胜负计数不能多于实际 n。
    res2 = _enrich_paired({"n": 2, "mean_diff": 1.5, "std_diff": 0.5},
                          [True, 2.0, 3.0], [1.0, 1.0, 1.0])
    assert (res2["n_seed_wins"] + res2["n_seed_losses"]
            + res2["n_seed_ties"]) == 2
    # bootstrap CI 也必须用同一批有效配对差：[2-1, 3-1]，不能把
    # True-1 当作 0 混入。
    assert res2["mean_diff_ci95"] == (1.0, 2.0)
    # 均值有效但方差/MDE 无法解析（坏 JSON）时只置空相关字段，不能 TypeError。
    res3 = _enrich_paired({"n": 2, "mean_diff": 1.5, "std_diff": "bad"},
                          [1.5, 2.5], [1.0, 1.0])
    assert res3["within_noise"] is None and res3["below_mde"] is None
    print("  [ok] _enrich_paired（置换 p/CI/MDE 字段与口径；bool/坏方差安全）")


def test_enrich_paired_huge_finite_diffs_keep_loo_finite():
    """配对差本身有限但同号相加会溢出时，留一均值/方向仍必须可判定。"""
    from run_all import _enrich_paired
    a = [1e308] * 3
    b = [0.0] * 3
    res = {"n": 3, "mean_diff": 1e308, "std_diff": 0.0,
           "metric": "PPL", "lower_is_better": True, "p_adj": 0.01}
    out = _enrich_paired(res, a, b)
    assert out["loo_sign_stable"] is True, out
    assert math.isfinite(out["loo_min_abs_mean"]), out
    assert out["loo_min_abs_mean"] == 1e308, out
    # ���负巨大差混合、均值抵消为 0 时，方向不再声称稳定。
    c = [1e308, -1e308, 1e308]
    d = [0.0, 0.0, 0.0]
    out2 = _enrich_paired({"n": 3, "mean_diff": 1e308 / 3.0,
                           "std_diff": 1.0, "metric": "PPL",
                           "lower_is_better": True, "p_adj": 0.01}, c, d)
    assert math.isfinite(out2["loo_min_abs_mean"]), out2
    assert out2["loo_sign_stable"] is False, out2
    print("  [ok] 配对稳健性：巨大有限差值的留一均值不溢出")


def test_enrich_paired_leave_one_seed_out_stability():
    """n 很小时一个种子可能决定符号：必须给出逐种子胜负与留一方向稳定性。

    a-b = [+0.4, +0.4, +0.4, +0.4, -0.1]：均值 +0.3，但删掉第 5 个种子后
    仍为正；构造一个删掉任一种子即翻转的对比，检查 loo_sign_stable=False。
    """
    a = [1.0, 1.0, 1.0]
    b = [0.9, 0.9, -1.0]
    # 差 = [+0.1, +0.1, +2.0] 全为正 → 稳定
    res = _enrich_paired({"n": 3, "mean_diff": 0.733, "std_diff": 1.1}, a, b)
    assert res["n_seed_wins"] == 3 and res["n_seed_losses"] == 0, res
    assert res["n_seed_ties"] == 0
    assert res["loo_sign_stable"] is True, res
    # 差 = [+0.5, -0.45, -0.4]：整体均值 -0.117 < 0；删掉 -0.45 后剩余均值
    # (+0.5-0.4)/2 = +0.05 > 0 → 符号翻转，应判为不稳定；胜 1 负 2。
    res2 = _enrich_paired(
        {"n": 3, "mean_diff": -0.117, "std_diff": 0.5},
        [1.0, -0.45, -0.4], [0.5, 0.0, 0.0])
    assert res2["n_seed_wins"] == 1 and res2["n_seed_losses"] == 2, res2
    assert res2["loo_sign_stable"] is False, res2
    # 构造明确会翻转的例子：差 = [+1.0, -0.1, -0.1, -0.1]（n=4）
    res3 = _enrich_paired(
        {"n": 4, "mean_diff": 0.175, "std_diff": 0.5},
        [1.0, -0.1, -0.1, -0.1], [0.0, 0.0, 0.0, 0.0])
    # 整体均值 +0.175 > 0；删掉 +1.0 种子后均值 = -0.1 < 0 → 翻转
    assert res3["loo_sign_stable"] is False, res3
    # LOO 警告在 within_noise/below_mde 之后才生效（更严重的噪声警示优先）
    assert "留一" in _noise_verdict_note(
        {"loo_sign_stable": False})
    assert "留一" in _noise_verdict_note(res3) or res3["within_noise"]
    # 无信息（n<2）→ None，不崩溃
    res4 = _enrich_paired({"n": 1, "mean_diff": 0.1, "std_diff": None},
                          [1.0], [0.9])
    assert res4["n_seed_wins"] is None and res4["loo_sign_stable"] is None
    print("  [ok] _enrich_paired（逐种子胜负 / 留一方向稳定性）")


def test_apply_holm_permutation_agreement():
    """t 显著但置换不显著必须显式标 ⚠；两套 p 在同家族内分别 Holm 校正。"""
    # p_perm 缺失（旧格式 dict）：按同一口径处理，不崩不标
    tests = [_mk("lega", "standard", "PPL", -0.5, 0.01, 5)]
    _apply_holm(tests)
    assert "显著优于" in tests[0]["verdict"] and "⚠" not in tests[0]["verdict"]
    assert tests[0]["t_perm_agree"] is None
    # t 与置换一致显著：无警示，t_perm_agree=True
    t_ok = _mk("lega", "standard", "PPL", -0.5, 0.001, 5)
    t_ok["p_perm"] = 0.001
    _apply_holm([t_ok])
    assert t_ok["p_perm_adj"] == 0.001
    t_ok["p_sign"] = 0.01
    _apply_holm([t_ok])
    assert t_ok["p_sign_adj"] == 0.01
    assert t_ok["t_perm_agree"] is True and "⚠" not in t_ok["verdict"]
    # t 显著（p=0.02，家族 m=1）但置换不显著（p_perm=0.3）→ ⚠ 标注
    t_warn = _mk("lega", "standard", "PPL", -0.3, 0.02, 4)
    t_warn["p_perm"] = 0.3
    _apply_holm([t_warn])
    assert "显著优于" in t_warn["verdict"] and "置换检验不显著" in t_warn["verdict"]
    assert t_warn["t_perm_agree"] is False
    # t 不显著但置换显著 → 边缘证据标注
    t_edge = _mk("qgate", "standard", "PPL", -0.1, 0.09, 5)
    t_edge["p_perm"] = 0.01
    _apply_holm([t_edge])
    assert "不显著" in t_edge["verdict"] and "置换检验显著" in t_edge["verdict"]
    assert t_edge["t_perm_agree"] is False
    # 置换 p 也参与 Holm 家族校正：家族 m=2 时最小 p 乘 2
    fam = [_mk("a", "standard", "PPL", -0.5, 0.01, 5),
           _mk("b", "standard", "PPL", -0.4, 0.05, 5)]
    fam[0]["p_perm"] = 0.01
    fam[1]["p_perm"] = 0.05
    fam[0]["p_sign"] = 0.02
    fam[1]["p_sign"] = 0.04
    _apply_holm(fam)
    assert abs(fam[0]["p_perm_adj"] - 0.02) < 1e-12
    assert abs(fam[1]["p_perm_adj"] - 0.05) < 1e-12
    assert abs(fam[0]["p_sign_adj"] - 0.04) < 1e-12
    assert abs(fam[1]["p_sign_adj"] - 0.04) < 1e-12
    # 种子不足：不进入 agree 判定
    t_small = _mk("x", "y", "PPL", 1.0, 0.01, 2)
    t_small["p_perm"] = 0.01
    _apply_holm([t_small])
    assert t_small["t_perm_agree"] is None and "种子不足" in t_small["verdict"]
    print("  [ok] _apply_holm 置换检验一致性与 ⚠ 标注")


def test_auroc_diff_tests_family():
    """机制主张（AUROC logit > 仅查询）必须作为配对差正式检验，而非并排数字。"""
    # 构造 runs：lega 的 logit AUROC 每个种子都比 qterm 高（带种子间波动）；
    # qgate 的差值在 0 附近来回 → 应判不显著
    def _slot(al, aq):
        return {"analysis": {"auroc_logit": al, "auroc_qterm": aq}}
    runs = {
        "lega": {1: _slot(0.80, 0.70), 2: _slot(0.83, 0.72), 3: _slot(0.77, 0.68),
                 4: _slot(0.81, 0.70), 5: _slot(0.79, 0.685)},
        "qgate": {1: _slot(0.70, 0.71), 2: _slot(0.72, 0.71), 3: _slot(0.69, 0.70),
                  4: _slot(0.71, 0.72), 5: _slot(0.70, 0.69)},
        # 无分析数据的变体不产生检验
        "standard": {1: {}},
        # 单侧缺失的种子整对剔除（auroc_qterm=None 不算配对）
        "lega_no_len": {1: _slot(0.8, None), 2: _slot(None, None)},
        # AUROC 越界（坏/手改产物）按缺失处理，不能制造一条“显著”检验
        "lega_no_conc": {1: _slot(1.5, 0.5), 2: _slot(-0.1, 0.4)},
    }
    tests = _auroc_diff_tests(runs)
    by_v = {t["candidate"]: t for t in tests}
    assert set(by_v) == {"lega", "qgate"}, set(by_v)
    t = by_v["lega"]
    assert t["metric"] == AUROC_DIFF_METRIC
    assert t["reference"] == AUROC_DIFF_REFERENCE
    assert t["lower_is_better"] is False
    assert t["n"] == 5 and t["mean_diff"] > 0
    assert t["p"] < 0.001
    # 差值全为正但量级有波动：精确置换 p 恰为双侧下限 2/2^5=0.0625
    # （n=5 的粒度边界：置换检验到不了 α=0.05，见 _apply_holm 注释）
    assert t["perm_exact"] is True
    assert abs(t["p_perm"] - 2.0 / 32.0) < 1e-12
    # qgate 差值均值为 -0.002（噪声）→ t 检验不显著
    assert abs(by_v["qgate"]["mean_diff"] - (-0.002)) < 1e-12
    assert by_v["qgate"]["p"] > 0.05
    # Holm 校正 + 结论方向（AUROC 越大越好 → 差为正 = 显著优于）
    tests = _apply_holm(tests)
    assert "显著优于" in by_v["lega"]["verdict"]
    # t 显著但置换 p 下限 >α → 必须带 ⚠（显著性依赖正态假设）
    assert "⚠" in by_v["lega"]["verdict"] and by_v["lega"]["t_perm_agree"] is False
    assert "不显著" in by_v["qgate"]["verdict"]
    print("  [ok] _auroc_diff_tests（配对差检验/置换粒度边界/家族校正）")


def test_detection_diff_tests_include_supplemental_gated_variants():
    """补充门控对照有 analysis JSON，机制族不能按主矩阵 VARIANTS 静默漏检。"""
    def _slot(al, aq):
        return {"analysis": {"auroc_logit": al, "auroc_qterm": aq}}

    runs = {
        "lega": {s: _slot(0.80, 0.70) for s in (1, 2, 3)},
        "lega_raw": {s: _slot(0.78, 0.69) for s in (1, 2, 3)},
        "lega_detach_conc": {s: _slot(0.77, 0.68) for s in (1, 2, 3)},
        # 安慰剂不进入分析人群（GATED_FOR_ANALYSIS），即使手改产物也不混入。
        "lega_placebo": {s: _slot(0.90, 0.60) for s in (1, 2, 3)},
        "standard": {s: {} for s in (1, 2, 3)},
    }
    tests = _auroc_diff_tests(runs)
    assert {t["candidate"] for t in tests} == {
        "lega", "lega_raw", "lega_detach_conc"}, tests
    # 设计层按同一分析人群规划 AUROC/AP 家族大小；实际检验数必须与之一致。
    from lega.design.power import planned_family_sizes
    sizes = planned_family_sizes(
        ["lega", "lega_raw", "lega_detach_conc", "lega_placebo"],
        analysis_variants=["lega", "lega_raw", "lega_detach_conc"])
    assert sizes["auroc_family_size"] == len(tests), sizes
    print("  [ok] _auroc_diff_tests 纳入补充门控变体（与设计家族一致）")


def test_ap_diff_tests_family_and_registered_metric():
    """PR-AUC 机制族：不平衡主指标口径下的配对差检验（H7 的判定依据）。"""
    def _slot(al, aq):
        return {"analysis": {"auprc_logit": al, "auprc_qterm": aq}}
    runs = {
        "lega": {1: _slot(0.30, 0.10), 2: _slot(0.33, 0.11),
                 3: _slot(0.28, 0.09), 4: _slot(0.31, 0.12),
                 5: _slot(0.29, 0.10)},
        "qgate": {1: _slot(0.20, 0.21), 2: _slot(0.19, 0.20),
                  3: _slot(0.21, 0.20), 4: _slot(0.20, 0.20),
                  5: _slot(0.19, 0.21)},
    }
    tests = _ap_diff_tests(runs)
    by_v = {t["candidate"]: t for t in tests}
    assert set(by_v) == {"lega", "qgate"}
    t = by_v["lega"]
    assert t["metric"] == AP_DIFF_METRIC
    assert t["reference"] == AUROC_DIFF_REFERENCE
    assert t["lower_is_better"] is False
    assert t["n"] == 5 and t["mean_diff"] > 0.15
    tests = _apply_holm(tests)
    assert "显著优于" in by_v["lega"]["verdict"]
    assert "不显著" in by_v["qgate"]["verdict"]
    # H7 必须以该 metric 字符串登记（evaluate_registered_hypotheses 按
    # metric 精确匹配检验表，拼写不一致会让假设永远“未检验”）
    reg = {"hypotheses": [{
        "id": "H7",
        "claim": "机制主张（不平衡口径）：AP(logit) > AP(仅查询)",
        "metric": AP_DIFF_METRIC, "candidate": "lega",
        "reference": AUROC_DIFF_REFERENCE, "direction": "candidate_better"}]}
    ev = evaluate_registered_hypotheses(reg, tests)
    assert ev and ev[0]["status"] == "支持", ev
    print("  [ok] _ap_diff_tests（PR-AUC 配对差、Holm 校正、H7 判定）")
