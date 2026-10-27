"""RULER 分层/逐任务指标与健壮性测试。"""
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



def test_ngram_baseline_rejects_non_dict_json():
    """合法 JSON 但非对象（数组）的 ngram_baseline.json 视为不可用。"""
    tmp = tempfile.mkdtemp()
    try:
        _write_json(os.path.join(tmp, "ngram_baseline.json"), [1, 2, 3])
        r = ngram_baseline_comparison({}, tmp)
        assert r.get("error"), r
        out = build_summary(tmp)
        # 畸形文件按“未运行”处理
        assert "简单统计基线：未运行" in out["markdown"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] ngram_baseline.json 非对象：返回 error/跳过")


def test_ngram_nested_malformed_fields_do_not_crash_summary():
    """ngram_baseline.json 的嵌套字段是列表时按不可用/回退处理，不能崩溃。"""
    tmp = tempfile.mkdtemp()
    try:
        _write_json(os.path.join(tmp, "eval_lega_seed1.json"), {
            "variant": "lega", "seed": 1, "val_ppl": 20.0, "val_loss": 2.0,
            "split": "val"})
        _write_json(os.path.join(tmp, "ngram_baseline.json"), {
            "split": "val", "ppl": [1], "ppl_window_matched": [1],
            "window_spec": [1], "train_tokens": 100, "eval_tokens": 50,
            "vocab_size": 20})
        bad = ngram_baseline_comparison({}, tmp)
        assert bad.get("error") and "ppl" in bad["error"], bad
        out = build_summary(tmp)
        assert "简单基线对比不可用" in out["markdown"]
        # ppl 合法但窗口参考/规格是列表：必须回退全 split 并正常渲染
        _write_json(os.path.join(tmp, "ngram_baseline.json"), {
            "split": "val", "ppl": {"bigram": 100.0},
            "ppl_window_matched": [1], "window_spec": [1],
            "train_tokens": 100, "eval_tokens": 50, "vocab_size": 20})
        out2 = build_summary(tmp)
        assert "未使用逐位置对齐参考" in out2["markdown"]
        sig = json.load(open(os.path.join(tmp, "significance.json"),
                             encoding="utf-8"))
        cmp_doc = sig["ngram_comparison"]
        assert cmp_doc["reference_source"] == "full_split"
        assert "window_spec" in (cmp_doc["fallback_reason"] or "")
        # token 计数/规格计数非数值时只影响展示，不能让整份汇总崩溃
        for patch in ({"train_tokens": [1]}, {"eval_tokens": "x"},
                      {"ppl_window_matched": {"bigram": 90.0},
                       "window_spec": {"ctx": 8, "iters": 2, "n_eval": 2,
                                       "n_tokens": [1]}}):
            doc = {"split": "val", "ppl": {"bigram": 100.0}}
            doc.update(patch)
            _write_json(os.path.join(tmp, "ngram_baseline.json"), doc)
            build_summary(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] ngram_baseline.json 嵌套字段非对象：报错/回退而非崩溃")


def test_build_summary_tolerates_malformed_latency_nested_fields():
    """latency.json 的变体条目/prefill 单元是列表时按缺失跳过，不能崩溃。"""
    tmp = tempfile.mkdtemp()
    try:
        for v in ("standard", "lega"):
            _write_json(os.path.join(tmp, f"eval_{v}_seed1.json"), {
                "variant": v, "seed": 1, "val_ppl": 10.0, "val_loss": 2.3,
                "n_params": 100, "split": "val"})
        lat_path = os.path.join(tmp, "latency.json")
        _write_json(lat_path, {"variants": {
            "standard": [1], "lega": [1]}})
        build_summary(tmp)
        # 基线 standard_sdpa 自身嵌套字段畸形：prefill/efficiency/spread
        # 都不能在运行时指标与权衡小节的读取路径上崩溃。
        _write_json(lat_path, {"variants": {
            "standard": {"prefill_ms": {"256": {"1": 10.0}},
                         "decode_ms_per_tok": {"256": 1.0}},
            "lega": {"prefill_ms": {"256": {"1": 12.0}},
                     "decode_ms_per_tok": {"256": 1.2}},
            "standard_sdpa": {
                "prefill_ms": [1], "decode_ms_per_tok": [1],
                "efficiency": [1],
                "prefill_ms_spread_pct": {"256": [1]},
                "decode_ms_spread_pct": {"256": [1]}}}})
        out = build_summary(tmp)
        assert "涨点-增幅权衡" in out["markdown"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] latency.json 嵌套字段非对象：缺失跳过而非崩溃")


def test_summary_tolerates_infinite_ruler_n_and_numeric_garbage():
    """RULER n=Infinity 不得触发 OverflowError；非数值指标按缺失/跳过。"""
    assert _ruler_min_cell_n({"results": {"niah": {
        "256": {"n": float("inf"), "em": 1.0},
        "512": {"n": 3, "em": "bad"}}}}) == 3
    assert _ruler_min_cell_n({"results": {"niah": {
        "256": {"n": 2.5}, "512": {"n": 3}}}}) == 3
    # 负数/0 是损坏计数，不是“最小样本数”；忽略后才不会把报告写成 n=-3。
    assert _ruler_min_cell_n({"results": {"niah": {
        "256": {"n": -3}, "512": {"n": 4}, "1024": {"n": 0}}}}) == 4
    assert _ruler_min_cell_n({"results": {"niah": {
        "256": {"n": -3}}}}) is None
    assert _ruler_min_cell_n({"results": {"niah": {
        "256": {"n": True}, "512": {"n": "4"}}}}) == 4
    bad_n = {"results": {"niah": {
        "256": {"n": 1.5, "em": 0.0},
        "512": {"n": True, "em": 0.5},
        "1024": {"n": "2", "em": 1.0}}}}
    assert _ruler_avg_micro(bad_n) == 1.0
    # n 是合法大整数（超出 float 范围）时，v*n 会 OverflowError；坏计数按
    # 缺失跳过，其余格仍可估计，不能中断汇总。
    assert _ruler_avg_micro({"results": {"niah": {
        "256": {"n": 10 ** 400, "em": 1.0},
        "512": {"n": 2, "em": 1.0}}}}) == 1.0
    assert _ruler_avg_micro({"results": {"niah": {
        "256": {"n": 10 ** 400, "em": 1.0}}}}) is None
    assert _ruler_cell_signature(bad_n) == frozenset({
        ("niah", "256", None), ("niah", "512", None),
        ("niah", "1024", 2)})
    assert _ppl_of({"ppl": 10 ** 400}) is None
    assert _stat([10 ** 400, "bad", float("nan"), 2.0]) == {
        "mean": 2.0, "std": None, "n": 1}
    tmp = tempfile.mkdtemp()
    try:
        _write_json(os.path.join(tmp, "ruler_standard_seed1.json"), {
            "variant": "standard", "seed": 1, "split": "val",
            "results": {"niah_synth": {
                "256": {"n": float("inf"), "em": 1.0},
                "512": {"n": 3, "em": "bad"}}}})
        out = build_summary(tmp)
        assert "standard" in out["markdown"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] RULER n=Infinity/非数值指标：跳过而非崩溃")


def test_ruler_protocol_tracks_depth_stratification():
    """分层位置与随机位置不是同一批题面：配对检验必须拒绝跨协议比较。"""
    from run_all import _same_protocol_fields
    base = {
        "ctx_lens": [256], "n_samples": 3, "sample_seed": 1,
        "corpus_source": "val", "corpus_held_out": True,
        "max_new": 12, "fwe_max_new": 44, "vt_chains": 4,
        "vt_depth": 4, "amp": True,
    }
    stratified = _ruler_protocol(dict(base, depth_stratified=True))
    random_pos = _ruler_protocol(dict(base, depth_stratified=False))
    legacy = _ruler_protocol(base)
    assert stratified["depth_stratified"] is True
    assert random_pos["depth_stratified"] is False
    assert "depth_stratified" not in legacy
    assert _same_protocol_fields(stratified, dict(stratified))
    assert not _same_protocol_fields(stratified, random_pos)
    assert not _same_protocol_fields(stratified, legacy)
    print("  [ok] RULER 配对协议区分分层/随机位置")


def test_int_or_zero_rejects_bool_and_fractional_counts():
    """外部 JSON 的计数不能被 int() 把 True/5.9 伪装成 1/5。"""
    assert _int_or_zero(None) == 0 and _int_or_zero("5") == 5
    assert _int_or_zero(5.0) == 5
    for bad in (True, False, 5.5, -3, float("nan"), float("inf"),
                "bad", [5], {"n": 5}):
        assert _int_or_zero(bad) == 0, bad
    from run_all import _fmt4, _fmt_rate, _fmt_scalar_maybe
    assert _fmt_rate(True) == "-" and _fmt_rate(0.5) == "50.000%"
    assert _fmt4(True) == "-" and _fmt4(1.25) == "1.2500"
    assert _fmt_scalar_maybe(False) == "-" and _fmt_scalar_maybe("n/a") == "n/a"
    assert _fmt_scalar_maybe(2.5) == "2.500"
    print("  [ok] 计数/百分比/小数展示拒绝 bool 伪装数值")


def test_large_integer_identity_does_not_round_through_float():
    """>2**53 的整数/种子必须逐位比较，不能被 float 往返折叠成同一个值。"""
    from run_all import _integral_value, _same_protocol_value, _same_seed
    a, b = 2**53 + 1, 2**53
    assert _integral_value(a) == a
    assert not _same_seed(a, b)
    assert _same_seed(a, a) and _same_seed(str(a), a)
    assert _same_protocol_value(a, str(a))
    assert not _same_protocol_value(a, b)
    assert not _same_protocol_value(True, 1)
    # int("1_0") 会得到 10，但下划线写法不是规范整数串，不能与种子/协议值
    # 10 静默合并（strict_int 对同一输入显式报错）。
    assert _integral_value("1_0") is None
    assert not _same_seed("1_0", 10)
    assert not _same_protocol_value("1_0", 10)
    print("  [ok] 大整数身份不经 float 丢精度")


def test_summary_rejects_bool_as_numeric_observation():
    """外部 JSON 的 true/false 不能被当成 1.0/0.0 进入均值或配对检验。"""
    assert _num_or_none(True) is None and _num_or_none(False) is None
    assert _num_or_none("1_0") is None
    # AUROC/PR-AUC/F1/基率是 [0,1] 比例：越界值不能进入均值/配对检验。
    assert _rate_or_none(0.5) == 0.5 and _rate_or_none("0.25") == 0.25
    for bad in (1.5, -0.1, True, "1_0", float("nan"), float("inf"), None):
        assert _rate_or_none(bad) is None, bad
    assert _stat([True, False, "1_0", 2.0]) == {
        "mean": 2.0, "std": None, "n": 1}
    assert _ppl_of({"ppl": True}) is None
    assert _ppl_ci_rel_pct({"ppl": 1.0, "val_ppl_ci95": [True, 2.0]}) is None
    assert _ruler_task_means({"results": {"niah": {
        "256": {"em": True}}}}) == {}
    assert _ruler_avg_micro({"results": {"niah": {
        "256": {"em": True, "n": 1}}}}) is None
    # 汇总完整路径：bool PPL/窗口数/训练统计均按缺失降级，而不是 1.0。
    tmp = tempfile.mkdtemp()
    try:
        _write_json(os.path.join(tmp, "eval_standard_seed1.json"), {
            "variant": "standard", "seed": 1, "split": "val",
            "ppl": True, "val_ppl": False, "val_loss": True,
            "gate_mean": True, "epochs_seen": True, "val_windows": True,
            "train_step_time_ms": True, "train_peak_mem_mb": True})
        out = build_summary(tmp)
        row = next(r for r in out["rows"] if r["variant"] == "standard")
        assert row["val_ppl"] is None
        assert row["gate_mean"] is None
        assert row["epochs_seen"] is None
        assert row["eval_windows"] is None
        assert row.get("eff_train_step_ms") is None
        assert row.get("eff_train_peak_mem_mb") is None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] bool 不再伪装成数值观测（PPL/RULER/训练统计）")


def test_build_summary_tolerates_malformed_data_report_and_audit():
    """data_report/audit 嵌套字段畸形时按缺失渲染，不能毁掉整份汇总。"""
    tmp = tempfile.mkdtemp()
    try:
        for report in (
                {"protocol": [1]}, {"splits": [1]},
                {"tokenizer": {"unk_rate": [1]}},
                {"contamination": {"train_val": [1]}},
                {"distribution_shift": {"val_vs_train": [1]}},
                {"sanitization": [1]}, {"text_stats": [1]},
                {"alerts": [1]},
                {"test_access": [1]}, {"splits": {"val": [1]}}):
            out = build_summary(tmp, data_report=report)
            assert "数据协议与防泄漏自检" in out["markdown"]
        out = build_summary(tmp, data_report={
            "protocol": {"split_first_then_preprocess": True},
            "splits": {"val": {"tokens": "x"}},
            "sanitization": {"train": {"control_chars_deleted": "bad"}},
            "text_stats": {"train": [1]},
            "alerts": [1, {"level": "warning", "code": "c", "message": "m"}],
            "test_access": {"checkpoints_evaluated_multiple_times": 1}})
        assert "[c] m" in out["markdown"]
        for audit in ([1], "bad", {"ok": True, "checked": 1,
                                   "exempt_keys": [1],
                                   "early_stop_runs": [1]}):
            build_summary(tmp, audit=audit)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] data_report/audit 嵌套字段非对象：缺失降级而非崩溃")


def test_build_summary_tolerates_badcases_without_task():
    """旧/外部 RULER 错误清单可能缺少 task：汇总页用占位符渲染，不能崩溃。"""
    tmp = tempfile.mkdtemp()
    try:
        _write_json(os.path.join(tmp, "eval_lega.json"), {
            "variant": "lega", "seed": 1, "val_ppl": 10.0,
            "val_loss": 2.3, "n_params": 10, "split": "test"})
        _write_json(os.path.join(tmp, "ruler_lega.json"), {
            "variant": "lega", "seed": 1,
            "badcases": [None, {"answer_in_prompt": True, "pred": "7",
                                "answer": "1", "context_numbers": ["7"],
                                "question": "q"}]})
        out = build_summary(tmp)
        assert "坏案例分析" in out["markdown"]
        assert "-@ctxNone" in out["markdown"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] build_summary：缺 task 的错误样本降级为占位符")


def test_ruler_avg_macro_micro_defined_and_reported():
    """RULER 平均必须显式定义：macro（任务/格等权）与 micro（按 n 加权）并报。"""
    doc = {"results": {
        "niah_synth": {"8": {"em": 1.0, "n": 1},
                       "16": {"em": 0.0, "n": 9}},
        "vt": {"8": {"em": 0.0, "n": 1}, "16": {"em": 0.0, "n": 9}}}}
    # macro：任务内两格等权（0.5 与 0），再任务等权 → 0.25
    assert abs(_ruler_avg_of(doc) - 0.25) < 1e-12
    # micro：按 n 加权，只有 niah_synth@8 的 1 个样例命中 → 1/20
    assert abs(_ruler_avg_micro(doc) - 0.05) < 1e-12
    # fwe 主平均用严格集合 EM；recall 只作诊断单列，不进入平均。
    # 否则“把上下文词全部列出”的灌水策略可凭 recall 拿满分。
    fwe = {"results": {"fwe": {"8": {"em": 0.0, "recall": 0.75, "n": 4}}}}
    assert abs(_ruler_avg_of(fwe) - 0.0) < 1e-12
    assert abs(_ruler_avg_micro(fwe) - 0.0) < 1e-12
    assert abs(_ruler_fwe_recall(fwe) - 0.75) < 1e-12
    assert _ruler_fwe_recall({"results": {"vt": {"8": {"em": 1.0}}}}) is None
    # precision/f1 同为诊断口径，且对“灌水”敏感
    fwe2 = {"results": {"fwe": {"8": {"em": 0.0, "recall": 1.0,
                                      "precision": 0.5, "f1": 2 / 3, "n": 4}}}}
    assert abs(_ruler_task_means(fwe2, "precision").get("fwe") - 0.5) < 1e-12
    assert abs(_ruler_task_means(fwe2, "f1").get("fwe") - 2 / 3) < 1e-12
    # 旧产物无 precision 字段 → 缺失而非崩溃
    assert _ruler_task_means(fwe, "precision").get("fwe") is None
    # 缺字段/坏类型按缺失处理
    assert _ruler_avg_micro([1]) is None
    assert _ruler_avg_micro({"results": [1]}) is None
    assert _ruler_avg_micro({"results": {"vt": {"8": {"n": 0}}}}) is None
    # n=0 / 非法 n 的单元格没有有效观测，不能用一个空洞的 em=0 拉低 macro。
    zero_cell = {"results": {
        "niah_synth": {"8": {"em": 1.0, "n": 10},
                       "16": {"em": 0.0, "n": 0}}}}
    assert abs(_ruler_avg_of(zero_cell) - 1.0) < 1e-12
    assert abs(_ruler_avg_micro(zero_cell) - 1.0) < 1e-12
    bad_n_cell = {"results": {
        "niah_synth": {"8": {"em": 1.0, "n": 10},
                       "16": {"em": 0.0, "n": 1.5}}}}
    assert abs(_ruler_avg_of(bad_n_cell) - 1.0) < 1e-12
    # EM/F1 等是 [0,1] 比例：越界值不能参与 macro 或 micro 平均（否则汇总
    # 会出现 >100% 的“RULER 平均”并制造方法有效的假象）。
    bad = {"results": {"niah_synth": {"8": {"em": 1.5, "n": 10},
                                      "16": {"em": -1.0, "n": 10},
                                      "32": {"em": 0.5, "n": 10}}}}
    assert abs(_ruler_avg_of(bad) - 0.5) < 1e-12
    assert abs(_ruler_avg_micro(bad) - 0.5) < 1e-12
    assert _ruler_task_means({"results": {"vt": {"8": {"recall": 2.0}}}}) == {}
    print("  [ok] _ruler_avg_micro/_ruler_avg_of（macro/micro 定义与 fwe EM 口径）")


def test_ruler_depth_means_excludes_non_niah_and_missing():
    """位置分层只汇总 niah_*，格等权，旧产物缺 em_by_depth 时不伪造。

    总 EM 掩盖 lost-in-the-middle：early/late 的高分可与 middle 塌陷抵消。
    分层均值必须按 (任务, 上下文, 位置箱) 等权，不能让大 n 格支配；fwe/vt
    的 depth 语义不同（vt 是链深度），绝不能混入。
    """
    doc = {"results": {
        "niah_synth": {
            "8": {"em_by_depth": {
                "early": {"em": 1.0, "n": 1},
                "middle": {"em": 0.0, "n": 9}}},
            "16": {"em_by_depth": {
                "early": {"em": 0.0, "n": 1},
                "middle": {"em": 1.0, "n": 1},
                "late": {"em": 0.5, "n": 2}}}},
        "niah_multikey": {
            "8": {"em_by_depth": {
                "late": {"em": 0.0, "n": 2}}}},
        # 非 niah 任务即使带同名键也不得进入分层统计
        "vt": {"8": {"em_by_depth": {"early": {"em": 1.0, "n": 1}}}},
        "fwe": {"8": {"em_by_depth": {"middle": {"em": 1.0, "n": 1}}}},
    }}
    dm = _ruler_depth_means(doc)
    # early 两格 (1.0, 0.0) 等权；middle (0.0,1.0) 等权；
    # late (0.5, 0.0) 两格等权。
    assert abs(dm["early"] - 0.5) < 1e-12, dm
    assert abs(dm["middle"] - 0.5) < 1e-12, dm
    assert abs(dm["late"] - 0.25) < 1e-12, dm
    assert abs(_ruler_depth_gap(doc) - 0.25) < 1e-12
    # 旧产物 / 损坏字段 → 缺失而非伪造
    assert _ruler_depth_means({"results": {"niah_synth": {"8": {"em": 1.0}}}}) == {}
    assert _ruler_depth_means({"results": {"niah_synth": {"8": {
        "em_by_depth": {"early": {"em": "x", "n": 2}}}}}}) == {}
    assert _ruler_depth_means({"results": {"niah_synth": {"8": {
        "em_by_depth": {"early": {"em": 1.0, "n": 0}}}}}}) == {}
    assert _ruler_depth_gap({"results": {"niah_synth": {"8": {
        "em_by_depth": {"early": {"em": 1.0, "n": 1}}}}}}) is None
    print("  [ok] _ruler_depth_means（仅 niah、格等权、缺失不伪造）")


def test_build_summary_fwe_precision_f1_columns():
    """fwe 除严格 EM 外，必须把 precision/F1 诊断量渲染进汇总表。"""
    tmp = tempfile.mkdtemp()
    try:
        for s in (1, 2, 3):
            for v, prec in (("standard", 0.4), ("lega", 0.6)):
                _write_json(os.path.join(tmp, f"eval_{v}_seed{s}.json"), {
                    "variant": v, "seed": s, "val_ppl": 10.0, "val_loss": 2.3,
                    "n_params": 100, "n_gate_extra": 0, "split": "val"})
                _write_json(os.path.join(tmp, f"ruler_{v}_seed{s}.json"), {
                    "variant": v, "seed": s,
                    "results": {"fwe": {"256": {"em": 0.5, "recall": 0.8,
                                               "precision": prec,
                                               "f1": 0.5 + prec / 10,
                                               "n": 30}}}})
        out = build_summary(tmp)
        md = out["markdown"]
        assert "fwe prec↑" in md and "fwe F1↑" in md, md[-2000:]
        row = next(r for r in out["rows"] if r["variant"] == "lega")
        assert row.get("ruler_fwe_precision") == 0.6, row
        assert row.get("ruler_fwe_f1") is not None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] build_summary：fwe precision/F1 诊断列")




def test_build_summary_ruler_depth_stratification():
    """汇总必须单独显示 NIAH 位置分层，并对 lost-in-the-middle 给出提示。"""
    tmp = tempfile.mkdtemp()
    try:
        depths = {
            "standard": {"early": 0.9, "middle": 0.9, "late": 0.9},
            "lega": {"early": 0.9, "middle": 0.2, "late": 0.9},
        }
        for s in (1, 2, 3):
            for v, dm in depths.items():
                _write_json(os.path.join(tmp, f"eval_{v}_seed{s}.json"), {
                    "variant": v, "seed": s, "val_ppl": 10.0, "val_loss": 2.3,
                    "n_params": 100, "n_gate_extra": 0, "split": "val"})
                _write_json(os.path.join(tmp, f"ruler_{v}_seed{s}.json"), {
                    "variant": v, "seed": s,
                    "results": {"niah_synth": {"256": {
                        "em": sum(dm.values()) / 3.0, "n": 30,
                        "em_by_depth": {
                            b: {"em": val, "n": 10,
                                "em_ci95_wilson": [0.5, 0.99]}
                            for b, val in dm.items()}}}}})
        out = build_summary(tmp)
        md = out["markdown"]
        assert "RULER NIAH 位置分层" in md, md[-2000:]
        assert "early−late" in md
        assert "位置不均衡" in md and "middle 比两端较低者低 70.0 pp" in md
        row = next(r for r in out["rows"] if r["variant"] == "lega")
        assert abs(row["ruler_depth_middle"] - 0.2) < 1e-12, row
        assert abs(row["ruler_depth_early_minus_late"]) < 1e-12, row
        # 旧产物（无 em_by_depth）必须显式警告无法排除 lost-in-the-middle，
        # 而不是静默地少一张表。
        for s in (1, 2, 3):
            for v in depths:
                _write_json(os.path.join(tmp, f"ruler_{v}_seed{s}.json"), {
                    "variant": v, "seed": s,
                    "results": {"niah_synth": {"256": {"em": 0.5, "n": 30}}}})
        old = build_summary(tmp)
        assert "不含 needle 位置分层" in old["markdown"]
        assert "重跑 RULER" in old["markdown"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] build_summary：RULER 位置分层表与 middle 塌陷提示")


def test_build_summary_per_task_ruler_breakdown():
    """RULER 综合平均可能被方向相反的任务互相抵消，必须逐任务分解并单独成表。

    逐任务检验（metric 以 `RULER:` 开头）完整落入 significance.json 且带
    exploratory 标记；markdown 只在预注册 RULER 对比上渲染，且不进入主检验表。
    """
    tmp = tempfile.mkdtemp()
    try:
        seeds = [1, 2, 3]
        # 构造两个任务：niah_synth 上 lega 稳定更好，vt 上稳定更差，综合平均
        # 接近 0，验证“逐任务方向相反被综合平均掩盖”的可审计性。
        per_task = {
            "niah_synth": {"lega": 0.9, "standard": 0.5},
            "vt": {"lega": 0.1, "standard": 0.5},
        }
        for s in seeds:
            for v in ("lega", "standard"):
                _write_json(os.path.join(tmp, f"eval_{v}_seed{s}.json"), {
                    "variant": v, "seed": s, "val_ppl": 10.0 + 0.01 * s,
                    "val_loss": 2.3, "n_params": 100, "n_gate_extra": 0,
                    "split": "val"})
                _write_json(os.path.join(tmp, f"ruler_{v}_seed{s}.json"), {
                    "variant": v, "seed": s,
                    "ctx_lens": [256], "n_samples": 30, "sample_seed": 1234,
                    "corpus_source": "val", "max_new": 12, "fwe_max_new": 44,
                    "vt_chains": 4, "vt_depth": 4,
                    "results": {
                        t: {"256": {"em": vals[v], "n": 30}}
                        for t, vals in per_task.items()}})
        reg = {"hypotheses": [
            {"id": "H4", "claim": "LEGA 优于 standard（RULER）",
             "metric": "RULER平均", "candidate": "lega", "reference": "standard",
             "direction": "candidate_better"}]}
        intact = {"status": "intact", "confirmatory_valid": True, "problems": [],
                  "registry_sha256": "a" * 64, "n_hypotheses": 1,
                  "baseline_document_sha256": "b" * 64, "chain_length": 0}
        with mock.patch("lega.pipeline.report.summary.load_registered_hypotheses",
                        return_value=reg), \
                mock.patch("lega.pipeline.report.summary.verify_registration", return_value=intact):
            out = build_summary(tmp)
        sig = json.load(open(os.path.join(tmp, "significance.json"),
                             encoding="utf-8"))
        task_tests = [t for t in sig["pairs"] if t.get("metric", "")
                      .startswith("RULER:")]
        got = {(t["metric"], t["ruler_task"]) for t in task_tests}
        assert ("RULER:niah_synth", "niah_synth") in got, task_tests
        assert ("RULER:vt", "vt") in got, task_tests
        assert all(t.get("exploratory") for t in task_tests), task_tests
        niah = next(t for t in task_tests if t["ruler_task"] == "niah_synth")
        vt = next(t for t in task_tests if t["ruler_task"] == "vt")
        assert abs(niah["mean_diff"] - 0.4) < 1e-9, niah
        assert abs(vt["mean_diff"] + 0.4) < 1e-9, vt
        md = out["markdown"]
        assert "RULER 逐任务分解" in md
        assert "niah_synth" in md and "| vt |" in md
        # 逐任务行不得污染主检验表（表头 + 行都在同一小节内）
        assert "RULER:niah_synth" not in md.split("### RULER 逐任务分解")[0]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] RULER 逐任务分解：方向相反任务可审计，且不污染主表")


def test_summary_helpers_tolerate_nested_non_dict_values():
    """嵌套字段是数组/标量时按缺失处理，不能让整份汇总在这些 helper 上崩溃。"""
    assert _ruler_task_means([1]) == {}
    assert _ruler_task_means({"results": [1]}) == {}
    assert _probe_oof([1], "auprc") is None
    assert _probe_oof({"linear_probe": [1]}, "auprc") is None
    assert _ppl_of([1]) is None
    assert _ppl_ci_rel_pct([1]) is None
    assert _ppl_ci_rel_pct({
        "ppl": 1.0, "val_ppl_ci95": [10 ** 400, 1.0]}) is None
    assert _ppl_ci_rel_pct({
        "ppl": 1.0, "val_ppl_ci95": [5.0, 1.0]}) is None
    # 端点有限但差值/除以极小 PPL 后不可表示时同样按缺失处理，不能写 inf。
    assert _ppl_ci_rel_pct({
        "ppl": 1e-300, "val_ppl_ci95": [-1e308, 1e308]}) is None
    assert _ppl_ci_rel_pct({
        "ppl": 1e-320, "val_ppl_ci95": [0.0, 1e308]}) is None
    assert _neural_eval_window_spec({"train_ctx": float("inf")}, False) is None
    assert _neural_eval_window_spec({"eval_iters": "x"}, False) is None
    assert _neural_eval_window_spec([1], False) is None
    assert _eval_matches_window_spec(
        {"eval_ctx": float("inf")},
        {"ctx": 8, "n_eval": 2, "n_windows": 5}) is False
    spec = {"ctx": 8, "n_eval": 2, "n_windows": 5}
    assert _eval_matches_window_spec(
        {"eval_ctx": 8.5, "eval_iters": 2, "val_windows": 5}, spec) is False
    assert _eval_matches_window_spec(
        {"eval_ctx": True, "eval_iters": 2, "val_windows": 5}, spec) is False
    assert _eval_matches_window_spec(
        {"eval_ctx": "8", "eval_iters": "2", "val_windows": "5"}, spec) is True
    tmp = tempfile.mkdtemp()
    try:
        _write_json(os.path.join(tmp, "ngram_baseline.json"),
                    {"ppl": {"bigram": float("inf")}})
        assert ngram_baseline_comparison({}, tmp).get("error")
        _write_json(os.path.join(tmp, "ngram_baseline.json"), {
            "split": "val", "ppl": {"bigram": 100.0},
            "ppl_window_matched": {"bigram": float("inf")},
            "window_spec": {"ctx": 8, "iters": 2}})
        cmp_doc = ngram_baseline_comparison({}, tmp)
        assert cmp_doc["ngram_ppl_window_matched"] is None, cmp_doc
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] 汇总 helper：嵌套非对象/Inf 数值安全降级")


def test_build_summary_tolerates_malformed_eval_probe_repro_curve():
    """eval/analysis/repro/curve 的字段类型畸形时汇总仍能生成。"""
    tmp = tempfile.mkdtemp()
    try:
        _write_json(os.path.join(tmp, "eval_lega_seed1.json"), {
            "variant": "lega", "seed": 1, "val_ppl": 9.0, "val_loss": 2.2,
            "split": ["val"], "epochs_seen": "x",
            "val_tokens_evaluated": "x",
            "worst_windows": ["bad"], "worst_window_scope": {
                "screened": "inf", "total": "x"}})
        # 变体名是数组时既不能进 runs 字典，也不能被当作合法 row。
        _write_json(os.path.join(tmp, "eval_standard_seed1.json"), {
            "variant": ["standard"], "seed": 1, "val_ppl": 10.0})
        _write_json(os.path.join(tmp, "analysis_lega_seed1.json"), {
            "variant": "lega", "seed": 1, "auroc_logit": "x",
            "auroc_qterm": "x", "linear_probe": [1],
            "gate_f1": [1], "gate_f1_trivial": "x"})
        _write_json(os.path.join(tmp, "ngram_baseline.json"), {
            "split": "val", "ppl": {"bigram": 100.0}})
        _write_json(os.path.join(tmp, "repro.json"), {
            "variants": ["lega"], "seeds": [1], "git": [1],
            "data": {"files": ["x"]}})
        _write_json(os.path.join(tmp, "curve_diagnosis.json"), {
            "lega": {"1": {"status": "正常收敛",
                           "recommendations": {"a": 1}}}})
        out = build_summary(tmp)
        assert [r["variant"] for r in out["rows"]] == ["lega"]
        assert "训练曲线诊断" in out["markdown"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] build_summary：畸形 eval/analysis/repro/curve 不崩溃")


def test_build_summary_worst_window_scope_requires_valid_total():
    """缺失/非法 total 不能伪装成 100% 覆盖率，也不能显示 x/0 分母。"""
    tmp = tempfile.mkdtemp()
    try:
        def write(v, scope):
            _write_json(os.path.join(tmp, f"eval_{v}_seed1.json"), {
                "variant": v, "seed": 1, "val_ppl": 10.0, "val_loss": 2.3,
                "split": "val", "worst_window_scope": scope,
                "worst_windows": [{
                    "window": 1, "split": "val", "loss": 1.0,
                    "pred_top5": [["a", 0.9]], "gold_next": "b",
                    "context_tail": "x"}]})
        write("standard", {"screened": 30, "total": None})
        write("lega", {"screened": 30, "total": "x"})
        md = build_summary(tmp)["markdown"]
        assert "最低覆盖率" not in md, md
        assert "30/0" not in md, md
        # 合法 scope 仍应正常显示真实覆盖率，不能被上面的过滤连坐。
        write("lega", {"screened": 30, "total": 60})
        md2 = build_summary(tmp)["markdown"]
        assert "30/60，最低覆盖率 50%" in md2, md2
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] build_summary：缺失/非法 worst_window_scope.total 不伪造覆盖率")


def test_build_summary_split_note_matches_actual_eval_split():
    """非 val/test 或混合 split 不能被报告说成“在 val 集上评测”。"""
    tmp = tempfile.mkdtemp()
    try:
        _write_json(os.path.join(tmp, "eval_lega_seed1.json"), {
            "variant": "lega", "seed": 1, "val_ppl": 10.0, "val_loss": 2.3,
            "split": "train"})
        md = build_summary(tmp)["markdown"]
        assert "PPL 在 val 集上评测" not in md, md
        assert "PPL 在 train 集上评测" in md, md
        # 不同变体混用 val/test 时也不得只挑一边来宣传。
        _write_json(os.path.join(tmp, "eval_standard_seed1.json"), {
            "variant": "standard", "seed": 1, "val_ppl": 11.0,
            "val_loss": 2.4, "split": "test"})
        md2 = build_summary(tmp)["markdown"]
        assert "评测 split 缺失、非法或不一致" in md2, md2
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] build_summary：PPL split 标注与真实评测集一致")


def test_tradeoff_prefers_sdpa_runtime_baseline():
    """涨点-增幅的运行时要与主表同一基线（standard_sdpa 优先）。"""
    tmp = tempfile.mkdtemp()
    try:
        for v in ("standard", "standard_sdpa", "lega"):
            _write_json(os.path.join(tmp, f"eval_{v}_seed1.json"), {
                "variant": v, "seed": 1, "val_ppl": 10.0, "val_loss": 2.3,
                "gate_mean": 0.5 if v == "lega" else None,
                "n_params": 100, "n_gate_extra": 0})

        def _lat(prefill, dec):
            return {"prefill_ms": {"256": {"1": prefill}},
                    "decode_ms_per_tok": {"256": dec},
                    "efficiency": {
                        "n_params": 100,
                        "flops_per_tok": {"256": {"total_flops_per_tok": 1000}},
                        "peak_mem_mb_prefill": {"256": 1.0}}}

        _write_json(os.path.join(tmp, "latency.json"), {"variants": {
            "standard": _lat(10.0, 1.0),        # 显式路径（旧默认基线）
            "standard_sdpa": _lat(1.0, 0.1),    # 与候选同路径的可比基线
            "lega": _lat(1.1, 0.11)}})
        out = build_summary(tmp)
        row = next(r for r in out["rows"] if r["variant"] == "lega")
        assert row["prefill_rel_pct"] == 10.0, row
        assert row["eff_prefill_delta_pct"] == 10.0, row
        assert row["eff_cost_max_pct"] == 10.0, row
        assert "基线=standard_sdpa" in out["markdown"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] 涨点-增幅：运行时基线优先 standard_sdpa（与主表一致）")


def test_tradeoff_baseline_requires_data_at_common_ctx():
    """standard_sdpa 存在但缺公共 ctx 数据时必须退回有数据的 standard。

    common_ctx 只对运行变体取交集，standard_sdpa 可能没有该 ctx 的量测；
    此时若仍选它作基线，Prefill/Decode 增幅会整体缺失、低估开销。
    """
    tmp = tempfile.mkdtemp()
    try:
        for v in ("standard", "standard_sdpa", "lega"):
            _write_json(os.path.join(tmp, f"eval_{v}_seed1.json"), {
                "variant": v, "seed": 1, "val_ppl": 10.0, "val_loss": 2.3,
                "gate_mean": 0.5 if v == "lega" else None,
                "n_params": 100, "n_gate_extra": 0})

        def _lat(cells):
            return {"prefill_ms": {c: {"1": p} for c, (p, _d) in cells.items()},
                    "decode_ms_per_tok": {c: d for c, (_p, d) in cells.items()},
                    "efficiency": {
                        "n_params": 100,
                        "flops_per_tok": {"256": {"total_flops_per_tok": 1000}},
                        "peak_mem_mb_prefill": {"256": 1.0}}}

        _write_json(os.path.join(tmp, "latency.json"), {"variants": {
            "standard": _lat({"256": (10.0, 1.0), "512": (20.0, 2.0)}),
            "standard_sdpa": _lat({"256": (1.0, 0.1)}),
            "lega": _lat({"256": (1.1, 0.11), "512": (24.0, 2.4)})}})
        out = build_summary(tmp)
        row = next(r for r in out["rows"] if r["variant"] == "lega")
        assert row["latency_ctx"] == 512, row
        # 基线必须是 512 ctx 上有数据的 standard：24 vs 20 → +20%
        assert row["eff_prefill_delta_pct"] == 20.0, row
        assert "基线=standard（公共 ctx=512" in out["markdown"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] 涨点-增幅：基线必须在公共 ctx 上有量测数据")


def test_tradeoff_ignores_noncanonical_latency_ctx_keys():
    """latency 的 ctx key 必须与 str(ctx) 完全一致。

    int("4_0")==40、int("040")==40，但 _lat_metric_at 查的是 str(ctx)；
    若把畸形 key 纳入公共 ctx，可能选中一个根本没有量测数据的最大上下文。
    """
    tmp = tempfile.mkdtemp()
    try:
        for v in ("standard", "lega"):
            _write_json(os.path.join(tmp, f"eval_{v}_seed1.json"), {
                "variant": v, "seed": 1, "val_ppl": 10.0, "val_loss": 2.3})

        def _lat(cells):
            return {"prefill_ms": {c: {"1": p} for c, (p, _d) in cells.items()},
                    "decode_ms_per_tok": {c: d for c, (_p, d) in cells.items()}}

        _write_json(os.path.join(tmp, "latency.json"), {"variants": {
            "standard": _lat({"4_0": (4.0, 0.4), "32": (3.0, 0.3)}),
            "lega": _lat({"4_0": (4.4, 0.44), "32": (3.3, 0.33)})}})
        out = build_summary(tmp)
        row = next(r for r in out["rows"] if r["variant"] == "lega")
        assert row["latency_ctx"] == 32, row
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] 涨点-增幅：latency 公共 ctx 忽略下划线/非规范 key")


def test_scope_missing_config_or_data_is_not_confirmed():
    """缺 config/data 时范围核对必须“不可确认”，不能因种子匹配就通过。"""
    tmp = tempfile.mkdtemp()
    try:
        reg = os.path.join(tmp, "reg.json")
        actual = os.path.join(tmp, "actual.json")
        cfg = {"d_model": 384, "n_layers": 6, "n_heads": 6, "ffn_mult": 4,
               "block_size": 1024, "steps": 8000, "train_ctx": 1024}
        _write_json(reg, cfg)
        _write_json(actual, cfg)
        registered = {"scope": {"config": reg, "seeds": [1, 2, 3],
                                "data": "WikiText-103 (gpt2 BPE)"}}

        # config 与 data 都缺失：即使 seeds 匹配也不能通过
        sc = registration_scope_check(registered, {"seeds": [1]})
        assert sc["checked"] is False and sc["in_scope"] is None, sc
        assert sc["warnings"]
        hyp = [{"id": "H", "confirmatory_valid": True}]
        assert apply_scope_gate(hyp, sc)[0]["confirmatory_valid"] is False

        # 有 config、缺 data.tokenizer：仍不可确认
        sc2 = registration_scope_check(registered, {
            "config_file": actual, "seeds": [1], "data": {}})
        assert sc2["checked"] is False and sc2["in_scope"] is None, sc2

        # 三部分都可核对且匹配：通过；分词器匹配本身也算“已核对”
        sc3 = registration_scope_check(registered, {
            "config_file": actual, "seeds": [1],
            "data": {"tokenizer": "gpt2_bpe"}})
        assert sc3["checked"] is True and sc3["in_scope"] is True, sc3

        # scope 未声明 config/data、仅声明 seeds：种子核对通过即可判定
        sc4 = registration_scope_check({"scope": {"seeds": [1]}},
                                       {"seeds": [1]})
        assert sc4["checked"] is True and sc4["in_scope"] is True, sc4
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] 范围核对：缺 config/data 不可确认，匹配 tokenizer 计入核对")


def test_verify_controlled_setup_checks_nonearly_runs_with_early_peer():
    """同批中一个 run 早停，不能豁免其它 run 的实际步数/计划步数检查。"""
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
        common = {"data_fingerprint": "abc123", "code_commit": "deadbeef",
                  "config_sha256": "shax", "tokens_seen": 1024}
        _write_json(os.path.join(tmp, "standard", "eval.json"), dict(
            common, variant="standard", seed=7, steps=8,
            steps_planned=10, early_stopped=False))
        _write_json(os.path.join(tmp, "lega", "eval.json"), dict(
            common, variant="lega", seed=7, steps=6,
            steps_planned=10, early_stopped=True, tokens_seen=614))
        try:
            verify_controlled_setup(tmp, variants, seeds, 1,
                                    expected_data_digest="abc123")
            raise AssertionError("非早停 run 实际步数 != 计划步数应触发 SystemExit")
        except SystemExit:
            pass
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] 混合早停：非早停 run 仍核对实际/计划步数")


def test_summary_loads_tolerate_null_ppl_and_bad_json_seed():
    """缺失 ppl 回退 val_ppl；无法解析的 seed 按无种子信息处理。"""
    assert _ppl_of({"ppl": None, "val_ppl": 9.5}) == 9.5
    assert _ppl_of({"ppl": 8.0, "val_ppl": 9.5}) == 8.0
    assert _seed_key("eval_lega.json", {"seed": "not-an-int"}) is None
    assert _seed_key("eval_lega.json", {"seed": "7"}) == 7
    assert _seed_key("eval_lega.json", {"seed": 7.0}) == 7
    assert _seed_key("eval_lega.json", {"seed": 7.5}) is None
    assert _seed_key("eval_lega.json", {"seed": True}) is None
    assert _seed_key("eval_lega.json", {"seed": float("inf")}) is None
    tmp = tempfile.mkdtemp()
    try:
        _write_json(os.path.join(tmp, "eval_standard_seed1.json"), {
            "variant": "standard", "seed": 1, "val_ppl": 10.0,
            "val_loss": 2.3, "n_params": 10, "split": "val"})
        for name in ("repro.json", "latency.json", "baseline_tuning.json",
                     "curve_diagnosis.json", "sanity_standard.json"):
            _write_json(os.path.join(tmp, name), [1, 2, 3])
        with open(os.path.join(tmp, "eval_lega_seed1.json"), "w",
                  encoding="utf-8") as f:
            f.write("{ 坏 JSON")
        out = build_summary(tmp)
        assert "standard" in out["markdown"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] 汇总：ppl=null 回退、坏 JSON 文件跳过、坏 seed 不崩溃")
