"""数据协议/预检、任务并发与命令行环境变量测试。"""
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



def test_data_protocol_preflight_blocks_critical_violations():
    """已知群组/时序泄露的数据默认不得进入训练；显式覆盖才放行。"""
    with tempfile.TemporaryDirectory() as d:
        _write_json(os.path.join(d, "data_report.json"), {
            "alerts": [
                {"level": "warning", "code": "split_shuffled", "message": "打乱"},
                {"level": "warning", "code": "ngram_overlap_train_val",
                 "message": "重叠"},
            ]})
        try:
            data_protocol_preflight(d)
            raise AssertionError("严重协议违规应中止训练")
        except SystemExit as e:
            assert "数据协议前置检查失败" in str(e)
            assert "split_shuffled" in str(e)
        info = data_protocol_preflight(d, allow_override=True)
        assert info["override"] is True
        assert [a["code"] for a in info["critical_violations"]] == \
            ["split_shuffled", "ngram_overlap_train_val"]
    with tempfile.TemporaryDirectory() as d:
        info = data_protocol_preflight(d)
        assert info["report_present"] is False
        assert info["critical_violations"] == []
    print("  [ok] 数据协议闸门：严重违规默认中止，显式覆盖放行，报告缺失仅告警")


def test_data_report_protocol_declarations_block_training():
    """报告内的协议声明必须被独立核对，不能只依赖 alerts 列表。"""
    from lega.preflight import data_report_protocol_issues
    bad = {
        "protocol": {"split_first_then_preprocess": False,
                     "shuffle_at_split_time": True,
                     "test_sealed": False, "fit_on": "all"},
        "tokenizer": {"fit_on": "all"},
        "test_reads_at_prep": {"used_for_selection": True},
        "split_protocol": {"shuffled": True, "fallback_char_cut": True,
                           "documents_cross_cut": True,
                           "paragraphs_cross_cut": True},
    }
    issues = data_report_protocol_issues(bad)
    assert len(issues) >= 8, issues
    assert any("paragraphs_cross_cut" in msg for msg in issues), issues
    assert data_report_protocol_issues({"alerts": []}) == []
    safe = {"protocol": {"split_first_then_preprocess": True,
                         "shuffle_at_split_time": False,
                         "test_sealed": True, "fit_on": "train"},
            "tokenizer": {"fit_on": "train"},
            "test_reads_at_prep": {"used_for_selection": False},
            "split_protocol": {"shuffled": False, "fallback_char_cut": False,
                               "documents_cross_cut": False,
                               "paragraphs_cross_cut": False,
                               "verified_disjoint": True}}
    assert data_report_protocol_issues(safe) == []
    # 声明“已核对互斥”但报告里仍记录实际重叠/缺失：矛盾证据必须阻断，
    # 不能被 verified_disjoint=True 掩盖。
    contradictory = {"split_protocol": {
        "verified_disjoint": True,
        "overlap_long_lines": {"train_val": {"exact_lines": 1,
                                             "normalized_lines": 0}}}}
    issues = data_report_protocol_issues(contradictory)
    assert len(issues) == 1 and "矛盾" in issues[0], issues
    missing_true = {"split_protocol": {
        "verified_disjoint": True, "missing_files": ["test"]}}
    issues = data_report_protocol_issues(missing_true)
    assert len(issues) == 1 and "verified_disjoint=True" in issues[0], issues
    # 声明未验证却既不解释重叠也不解释缺失：不能静默放行。
    unexplained = {"split_protocol": {"verified_disjoint": False}}
    assert any("未记录" in msg
               for msg in data_report_protocol_issues(unexplained))
    # 交叉证据字段存在但类型损坏：不能当成缺失而放行 verified_disjoint。
    malformed_evidence = {"split_protocol": {
        "verified_disjoint": True,
        "overlap_long_lines": ["train_val"],
        "missing_files": "test"}}
    msgs = data_report_protocol_issues(malformed_evidence)
    assert any("overlap_long_lines" in m for m in msgs), msgs
    assert any("missing_files" in m for m in msgs), msgs
    # 清洗后复核失败必须独立于 alerts 被拦截；结构损坏不能当成缺失放行。
    residual = {"duplicates": {"post_cleanup": {
        "checked_pairs": [["train", "val"]], "train_eval_clean": False,
        "overlap_long_lines": {"train_val": {"exact_lines": 2}},
        "overlap_blocks": {"train_val": 1}}}}
    assert any("去污染未闭合" in m
               for m in data_report_protocol_issues(residual))
    malformed_post = {"duplicates": {"post_cleanup": {
        "checked_pairs": "train,val", "train_eval_clean": "no"}}}
    post_msgs = data_report_protocol_issues(malformed_post)
    assert any("checked_pairs" in m for m in post_msgs), post_msgs
    assert any("train_eval_clean" in m for m in post_msgs), post_msgs
    assert data_report_protocol_issues({"duplicates": {"post_cleanup": {
        "checked_pairs": [["train", "val"]], "train_eval_clean": True}}}) == []
    with tempfile.TemporaryDirectory() as d:
        _write_json(os.path.join(d, "data_report.json"),
                    {"alerts": [], "protocol": {"test_sealed": False}})
        try:
            data_protocol_preflight(d)
            raise AssertionError("报告声明 test_sealed=False 应阻断训练")
        except SystemExit as e:
            assert "data_report_protocol" in str(e)
        info = data_protocol_preflight(d, allow_override=True)
        assert any(a["code"] == "data_report_protocol"
                   for a in info["critical_violations"])
    print("  [ok] data_report 协议声明：不安全声明即使 alerts 为空也阻断训练")


def test_protocol_override_env_matches_cli_flag(monkeypatch):
    """run_all 自身的闸门必须和子进程一样识别环境变量放行。"""
    from lega.preflight import ALLOW_ENV, allow_protocol_violations
    monkeypatch.delenv(ALLOW_ENV, raising=False)
    assert not allow_protocol_violations(False)
    assert allow_protocol_violations(True)
    monkeypatch.setenv(ALLOW_ENV, "YES")
    assert allow_protocol_violations(False)
    monkeypatch.setenv(ALLOW_ENV, "0")
    assert not allow_protocol_violations(False)
    print("  [ok] 数据协议放行：CLI 与环境变量同口径")


def test_direct_train_entry_runs_data_protocol_preflight(monkeypatch):
    """直接 `python -m lega.train` 也必须被同一份防泄漏闸门拦截。"""
    from lega.train import main as train_main
    with tempfile.TemporaryDirectory() as d:
        _write_json(os.path.join(d, "data_report.json"), {
            "alerts": [{"level": "warning", "code": "split_shuffled",
                        "message": "打乱"}]})
        cfg = os.path.join(d, "cfg.json")
        _write_json(cfg, {"seed": 7})
        monkeypatch.setattr(sys, "argv", [
            "lega.train", "--config", cfg, "--variant", "standard",
            "--data_dir", d, "--out_dir", os.path.join(d, "out")])
        try:
            train_main()
            raise AssertionError("直接训练入口应执行数据协议闸门")
        except SystemExit as e:
            assert "数据协议前置检查失败" in str(e)
    print("  [ok] 直接训练入口执行数据协议闸门（严重告警默认中止）")


def test_data_report_binding_detects_tampering():
    with tempfile.TemporaryDirectory() as d:
        rp = os.path.join(d, "data_report.json")
        _write_json(rp, {"alerts": []})
        with open(rp, "rb") as f:
            digest = hashlib.sha256(f.read()).hexdigest()
        _write_json(os.path.join(d, "meta.json"), {"report_sha256": digest})
        assert data_report_binding_issue(d) is None
        _write_json(rp, {"alerts": [{"code": "split_shuffled"}]})
        issue = data_report_binding_issue(d)
        assert issue and "哈希不一致" in issue
        info = data_protocol_preflight(d, allow_override=True)
        assert info["binding_issue"] == issue and info["override"] is True
        assert any(a["code"] == "data_report_binding"
                   for a in info["critical_violations"])
        # 报告被删除而 meta 仍记录其哈希：缓存不完整，同样是硬阻断项。
        os.remove(rp)
        issue = data_report_binding_issue(d)
        assert issue and "缺失" in issue
        info = data_protocol_preflight(d, allow_override=True)
        assert any(a["code"] == "data_report_binding"
                   for a in info["critical_violations"])
    print("  [ok] data_report 与 meta 哈希绑定：篡改/缺失/过期报告被阻断")


def test_data_protocol_preflight_detects_stale_source_hash():
    """源 txt 改动后直接训练必须被阻断，不能静默沿用旧 bin/report。"""
    from lega.preflight import data_source_hash_issue
    with tempfile.TemporaryDirectory() as d:
        txt = os.path.join(d, "train.txt")
        with open(txt, "w", encoding="utf-8") as f:
            f.write("original training text\n")
        with open(txt, "rb") as f:
            digest = hashlib.sha256(f.read()).hexdigest()
        _write_json(os.path.join(d, "meta.json"), {
            "cache_schema": CACHE_SCHEMA, "fit_on": "train",
            "source_sha256": {"train": digest}})
        assert data_source_hash_issue(d) is None
        with open(txt, "w", encoding="utf-8") as f:
            f.write("changed training text\n")
        issue = data_source_hash_issue(d)
        assert issue and "train.txt" in issue and "不一致" in issue
        try:
            data_protocol_preflight(d)
            raise AssertionError("源文本改动后应阻断训练")
        except SystemExit as e:
            assert "data_source_hash" in str(e)
        info = data_protocol_preflight(d, allow_override=True)
        assert info["source_hash_issue"] == issue
        assert any(a["code"] == "data_source_hash"
                   for a in info["critical_violations"])
        # 缓存没记录 test，但目录里新增了 test.txt：同样是未绑定源文本。
        with open(txt, "rb") as f:
            changed = hashlib.sha256(f.read()).hexdigest()
        _write_json(os.path.join(d, "meta.json"), {
            "cache_schema": CACHE_SCHEMA, "fit_on": "train",
            "source_sha256": {"train": changed}})
        with open(os.path.join(d, "test.txt"), "w", encoding="utf-8") as f:
            f.write("new test text\n")
        issue = data_source_hash_issue(d)
        assert issue and "test.txt" in issue
    print("  [ok] 源文本哈希：txt 改动/新增 test 而未重建 bin 时阻断训练")


def test_data_protocol_preflight_detects_replaced_bin():
    """同尺寸替换 bin 必须在训练前被内容哈希阻断（不只看 token 数量）。"""
    from lega.preflight import data_bin_hash_issue
    with tempfile.TemporaryDirectory() as d:
        txt = os.path.join(d, "train.txt")
        with open(txt, "w", encoding="utf-8") as f:
            f.write("x" * 100 + "\n")
        with open(txt, "rb") as f:
            txt_digest = hashlib.sha256(f.read()).hexdigest()
        train_bin = os.path.join(d, "train.bin")
        with open(train_bin, "wb") as f:
            f.write(b"\x01\x00" * 50)
        with open(train_bin, "rb") as f:
            digest = hashlib.sha256(f.read()).hexdigest()
        meta = os.path.join(d, "meta.json")
        _write_json(meta, {
            "cache_schema": CACHE_SCHEMA, "fit_on": "train",
            "source_sha256": {"train": txt_digest},
            "bin_sha256": {"train": digest}})
        assert data_bin_hash_issue(d) is None
        with open(train_bin, "r+b") as f:
            f.write(b"\x02\x00")
        issue = data_bin_hash_issue(d)
        assert issue and "train.bin" in issue and "不一致" in issue
        try:
            data_protocol_preflight(d)
            raise AssertionError("被替换的 bin 应阻断训练")
        except SystemExit as e:
            assert "data_bin_hash" in str(e)
        info = data_protocol_preflight(d, allow_override=True)
        assert info["bin_hash_issue"] == issue
        assert any(a["code"] == "data_bin_hash"
                   for a in info["critical_violations"])
        # 有源文本却完全没有 bin 哈希记录：不可判定，同样阻断
        _write_json(meta, {
            "cache_schema": CACHE_SCHEMA, "fit_on": "train",
            "source_sha256": {"train": txt_digest}})
        issue = data_bin_hash_issue(d)
        assert issue and "未记录" in issue
    print("  [ok] bin 内容哈希：同尺寸替换/未记录时阻断训练")


def test_data_protocol_preflight_blocks_stale_cache_schema():
    """旧版清洗/去污染管线生成的 bin 不得静默进入训练。"""
    with tempfile.TemporaryDirectory() as d:
        meta = os.path.join(d, "meta.json")
        _write_json(meta, {"cache_schema": CACHE_SCHEMA - 1})
        try:
            data_protocol_preflight(d)
            raise AssertionError("旧缓存协议应中止训练")
        except SystemExit as e:
            assert "data_cache_schema" in str(e)
            assert "重建" in str(e)
        info = data_protocol_preflight(d, allow_override=True)
        assert any(a["code"] == "data_cache_schema"
                   for a in info["critical_violations"])
        assert info["cache_schema_issue"]
        assert "不一致" in info["cache_schema_issue"]
        _write_json(meta, {"cache_schema": str(CACHE_SCHEMA),
                           "fit_on": "train"})
        info = data_protocol_preflight(d)
        assert info["cache_schema_issue"] is None
        assert info["critical_violations"] == []
        # 小数 schema 不能被 int() 截断成当前版本而静默放行。
        _write_json(meta, {"cache_schema": CACHE_SCHEMA + 0.9,
                           "fit_on": "train"})
        info = data_protocol_preflight(d, allow_override=True)
        assert info["cache_schema_issue"]
        assert "cache_schema=-1" in info["cache_schema_issue"]
        # schema 正确但词表在全量数据上拟合：仍必须拦截。
        _write_json(meta, {"cache_schema": CACHE_SCHEMA, "fit_on": "all"})
        try:
            data_protocol_preflight(d)
            raise AssertionError("fit_on=all 的缓存不应进入训练")
        except SystemExit as e:
            assert "fit_on" in str(e)
        info = data_protocol_preflight(d, allow_override=True)
        assert "fit_on" in info["cache_schema_issue"]
    with tempfile.TemporaryDirectory() as d:
        _write_json(os.path.join(d, "meta.json"), {"cache_schema": "bad"})
        info = data_protocol_preflight(d, allow_override=True)
        issue = info["cache_schema_issue"]
        assert issue and "cache_schema=-1" in issue
    print("  [ok] 数据缓存协议版本：旧版/损坏 schema 默认阻断，重建后放行")


def test_resolve_jobs_defaults():
    class _A:
        jobs = None
        wandb = False
        device = "cpu"

    # 小核（<=4）默认 cpu+1；大核维持 4 并发上限（见 _resolve_jobs）
    cpu = os.cpu_count() or 1
    assert _resolve_jobs(_A()) == (cpu + 1 if cpu <= 4 else min(4, cpu))
    _A.wandb = True
    assert _resolve_jobs(_A()) == 1
    _A.wandb = False
    _A.device = "cuda"
    assert _resolve_jobs(_A()) == 1
    _A.device = "cpu"
    _A.jobs = 7
    assert _resolve_jobs(_A()) == 7
    print("  [ok] _resolve_jobs（显式 jobs / CUDA / wandb 默认值）")


def test_parse_variants_dedups_preserving_order():
    assert parse_variants("lega") == ["lega"]
    assert parse_variants(" lega , standard ,lega,, leafless ") == \
        ["lega", "standard", "leafless"]
    # 重复变体在并发阶段会生成重复命令并写同一输出路径，必须按首次出现去重
    assert parse_variants("standard,standard,standard") == ["standard"]
    assert parse_variants(", ,") == []
    print("  [ok] parse_variants：保序去重、忽略空白项")


def test_run_many_sequential_parallel_and_failure():
    ok = [sys.executable, "-c", "pass"]
    run_many([ok], 1)
    run_many([ok, ok], 2)
    bad = [sys.executable, "-c", "import sys; sys.exit(3)"]
    for cmds, jobs in (([ok, bad, ok], 2), ([bad], 1)):
        try:
            run_many(cmds, jobs)
            raise AssertionError("非零退出码应触发 CalledProcessError")
        except subprocess.CalledProcessError as e:
            assert e.returncode == 3
    print("  [ok] run_many（顺序/并发/失败传播）")


def test_cmd_env_sets_pythonhashseed_per_command_seed():
    """每条命令的 PYTHONHASHSEED 必须等于它自己的 --seed，而不是父进程基准值。"""
    e = _cmd_env([sys.executable, "-m", "lega.train", "--seed", "2024"])
    assert e["PYTHONHASHSEED"] == "2024"
    assert e["CUBLAS_WORKSPACE_CONFIG"] in (":4096:8", ":16:8")
    e_eq = _cmd_env([sys.executable, "-m", "lega.train", "--seed=2025"])
    assert e_eq["PYTHONHASHSEED"] == "2025"
    base = dict(os.environ)
    base["PYTHONHASHSEED"] = "1337"
    assert _cmd_env([sys.executable, "x.py"], base)["PYTHONHASHSEED"] == "1337"
    for bad in ("notint", "1.5", "-1", "True"):
        try:
            _cmd_env([sys.executable, "x.py", "--seed", bad], base)
        except ValueError:
            pass
        else:
            raise AssertionError(f"非法 --seed 未被拒绝: {bad!r}")
    try:
        _cmd_env([sys.executable, "x.py", "--seed"], base)
    except ValueError:
        pass
    else:
        raise AssertionError("缺失 --seed 取值未被拒绝")
    # 继承的 CUBLAS 值无效时必须改写为受支持配置；有效值保留
    bad_cublas = dict(base, CUBLAS_WORKSPACE_CONFIG=":bad:")
    assert _cmd_env([sys.executable, "x.py"], bad_cublas)[
        "CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    good_cublas = dict(base, CUBLAS_WORKSPACE_CONFIG=":16:8")
    assert _cmd_env([sys.executable, "x.py"], good_cublas)[
        "CUBLAS_WORKSPACE_CONFIG"] == ":16:8"
    # 真实子进程验证：退出码非 0 即断言失败
    code = ("import os,sys;"
            "assert os.environ['PYTHONHASHSEED']==sys.argv[1], "
            "os.environ['PYTHONHASHSEED']")
    run_many([[sys.executable, "-c", code, "42", "--seed=42"]], 1)
    print("  [ok] _cmd_env/run_many 逐命令对齐 PYTHONHASHSEED 与 --seed")
