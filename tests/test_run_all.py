"""对比/消融协议测试：Holm 校正结论、组件有效性判定与受控变量审计。

覆盖四条关键科研口径：
  * 多重比较校正后结论才有效（避免 8 变体两两对比下的假阳性）；
  * “各组件均有用”必须有显著统计证据，组合跑全本身不算证据；
  * 小样本下 t 检验结论必须与置换检验互证，冲突要显式暴露而不是静默采信；
  * 受控变量审计除超参数外还核对数据指纹/代码版本/训练步数。
"""
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
from run_all import (ALPHA, AP_DIFF_METRIC, MIN_SEEDS, _ablation_coverage,
                      _ap_diff_tests, _apply_holm, _auroc_diff_tests, _cmd_env, _effect_status,
                      _enrich_paired, _eval_matches_window_spec,
                      _factorial_interaction_tests, _FACTORIAL_CONTRAST_CELLS,
                      _factorial_protocol_ok, _inference_shapley_tests,
                      INFERENCE_SHAPLEY_METRIC, PARAM_CONTROL_VARIANT,
                      PARAM_CONTROL_PAIR,
                      _int_or_zero, _lat_noise_pct,
                      _neural_eval_window_spec, _noise_floor_pct,
                      _noise_verdict_note, _loo_stability_str, _num_or_none,
                      _rate_or_none,
                      _seed_winloss_str, _paired_comparisons, _ppl_ci_rel_pct,
                      _log_ppl_of, _ppl_of, _probe_oof, _rel_pct,
                      _render_baseline_sections, _result_files,
                      _render_inference_shapley,
                      _render_parameter_matched_control,
                      _ruler_avg_micro, _ruler_avg_of, _ruler_cell_signature,
                      _ruler_depth_gap, _ruler_depth_means,
                      _ruler_fwe_recall, _ruler_min_cell_n,
                      _ruler_task_means,
                      _render_registration_integrity, _resolve_jobs, _stat,
                      _seed_key, _strict_seed_int, _parse_seed_list,
                      apply_budget_gate, apply_integrity_gate, apply_scope_gate,
                      AUROC_DIFF_METRIC, AUROC_DIFF_REFERENCE,
                      build_comparison_pairs, build_summary,
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


def _mk(cand, ref, metric, mean_diff, p, n=5, lower=True):
    return {"candidate": cand, "reference": ref, "metric": metric,
            "mean_diff": mean_diff, "std_diff": 0.1, "p": p, "p_adj": None,
            "n": n, "lower_is_better": lower, "d_z": 1.0}


def _write_json(path, obj):
    # 测试 fixture 默认模拟“配置显式统一走显式注意力路径”；需要测试缺键/
    # 默认值行为的用例直接写文件，避免被这里的默认值掩盖。
    if (os.path.basename(path) == "config_used.json"
            and isinstance(obj, dict) and "use_sdpa_standard" not in obj):
        obj = dict(obj, use_sdpa_standard=False)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)


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


def test_apply_holm_verdicts():
    tests = [
        _mk("lega", "standard", "PPL", -0.5, 0.01, 5),
        _mk("qgate", "standard", "PPL", -0.01, 0.2, 5),
        _mk("lega_no_query", "lega", "PPL", 0.3, 0.04, 5),
        _mk("lega_len_only", "standard", "PPL", -0.2, 0.05, 2),
        _mk("lega", "standard", "RULER平均", 0.1, 0.03, 5, lower=False),
    ]
    _apply_holm(tests)
    # PPL 家族 m=4：升序 p=[0.01,0.04,0.05,0.2] → adj=[0.04,0.12,0.12,0.2]
    assert abs(tests[0]["p_adj"] - 0.04) < 1e-12
    assert "显著优于" in tests[0]["verdict"]
    assert abs(tests[2]["p_adj"] - 0.12) < 1e-12
    assert "不显著" in tests[2]["verdict"]
    assert "种子不足" in tests[3]["verdict"]
    # RULER 是独立家族（m=1）：校正不改变 0.03<alpha，且方向按 lower=False 判定
    assert abs(tests[4]["p_adj"] - 0.03) < 1e-12
    assert "显著优于" in tests[4]["verdict"]
    # 无法检验（n<2 → p=None）不参与家族、不给结论
    t = _mk("x", "y", "PPL", 1.0, None, n=1)
    _apply_holm([t])
    assert t["p_adj"] is None and "无法检验" in t["verdict"]
    assert ALPHA == 0.05 and MIN_SEEDS == 3
    print("  [ok] _apply_holm（逐指标家族校正/显著性结论/种子不足）")


def test_component_evidence_requires_significance():
    tests = [
        # A：去一法 Full−A 显著变差（PPL 变大）→ 有证据；全因子未检出
        _mk("lega_no_query", "lega", "PPL", 0.4, 0.005),
        _mk("qgate", "gate_only", "PPL", -0.01, 0.7),
        # B：两个方向都不显著 → 无证据
        _mk("lega_no_conc", "lega", "PPL", 0.05, 0.5),
        _mk("lega_conc_only", "gate_only", "PPL", -0.02, 0.6),
        # C：PPL 上去掉反而更好（显著），但 C 的主指标是 RULER；RULER 未检验
        _mk("lega_no_len", "lega", "PPL", -0.1, 0.001),
        _mk("lega_len_only", "gate_only", "PPL", 0.03, 0.4),
    ]
    _apply_holm(tests)
    comp = {c["component"]: c for c in component_evidence(tests)}
    assert comp["A 查询项"]["supported"] == ["PPL"]
    assert comp["A 查询项"]["verdict"].startswith("有证据支持")
    assert comp["A 查询项"]["primary_metric"] == "PPL"
    assert comp["B 集中度证据"]["verdict"] == "证据不足"
    # C 的主指标是 RULER；PPL 上的（相反）结果只能作探索性补充
    assert comp["C 长度校准"]["verdict"] == "证据不足"
    assert comp["C 长度校准"]["primary_metric"] == "RULER平均"
    assert "方向相反" in comp["C 长度校准"]["loo"]["PPL"]["reason"]
    assert comp["C 长度校准"]["secondary"]["PPL"]["exploratory"] is True
    print("  [ok] component_evidence（主指标判定/去一法/全因子/方向门槛）")


def test_component_evidence_requires_valid_seed_count():
    """缺 n / n 非法时不能默认“种子充足”而给出组件有用证据。

    汇总文件可能被旧版本或手工编辑成缺 n；阈值判定必须先证明 n>=MIN_SEEDS。
    """
    t = _mk("lega_no_query", "lega", "PPL", 0.4, 0.005)
    t.pop("n")
    t["p_adj"] = 0.005
    st = _effect_status(t, removing=True)
    assert not st["supported"] and "种子" in st["reason"], st
    for bad in (True, 2.5, "bad"):
        t2 = _mk("lega_no_query", "lega", "PPL", 0.4, 0.005)
        t2["n"] = bad
        t2["p_adj"] = 0.005
        st2 = _effect_status(t2, removing=True)
        assert not st2["supported"] and "种子" in st2["reason"], st2
    print("  [ok] 组件证据拒绝缺 n/非法 n 的“显著”检验")


def test_component_evidence_flags_contradiction():
    """一套消融显著支持、另一套显著反向时必须判冲突，不能用 OR 静默取胜。"""
    tests = [
        _mk("lega_no_query", "lega", "PPL", 0.4, 0.005),   # 去 A 变差：支持
        _mk("qgate", "gate_only", "PPL", 0.3, 0.004),      # 加 A 反而变差：反向
    ]
    _apply_holm(tests)
    a = next(c for c in component_evidence(tests) if c["component"] == "A 查询项")
    assert a["supported"] == []
    assert a["verdict"].startswith("证据冲突")
    # 只有一套显著反向、另一套不显著时仍必须判冲突，不能降格成“证据不足”
    one_sided = [
        _mk("lega_no_query", "lega", "PPL", 0.4, 0.5),      # 去 A 不显著
        _mk("qgate", "gate_only", "PPL", 0.3, 0.005),       # 加 A 显著反向
    ]
    _apply_holm(one_sided)
    a2 = next(c for c in component_evidence(one_sided)
              if c["component"] == "A 查询项")
    assert a2["verdict"].startswith("证据冲突"), a2
    print("  [ok] component_evidence（任一套显著反向 -> 证据冲突，另一套不显著亦然）")


def test_tradeoff_verdict_rejects_bad_deals():
    """涨点-增幅权衡判定：不显著涨点说服力有限；大增幅小收益被拒绝。"""
    # 不显著 + 增幅 >10% → 不可行（微小且不显著的涨点 + 大增幅）
    v = tradeoff_verdict(benefit_pct=0.3, p_adj=0.4, cost_pct=30.0)
    assert v.startswith("不可行"), v
    # 不显著 + 增幅小 → 说服力有限
    v = tradeoff_verdict(benefit_pct=0.3, p_adj=0.4, cost_pct=2.0)
    assert v.startswith("说服力有限"), v
    # 显著但增幅 >5×收益 → 收益-增幅比不利
    v = tradeoff_verdict(benefit_pct=0.5, p_adj=0.01, cost_pct=900.0)
    assert "收益-增幅比不利" in v or "需审视" in v, v
    # 显著 + 增幅温和（<5×收益）→ 可接受
    v = tradeoff_verdict(benefit_pct=5.0, p_adj=0.01, cost_pct=10.0)
    assert v == "可接受", v
    # 显著但增幅 >50% → 需审视
    v = tradeoff_verdict(benefit_pct=5.0, p_adj=0.01, cost_pct=60.0)
    assert v.startswith("需审视"), v
    # 数据缺失 → 不下结论
    assert tradeoff_verdict(None, 0.01, 10.0) == "数据不足"
    assert tradeoff_verdict(1.0, 0.01, None) == "数据不足"
    assert tradeoff_verdict(None, None, 10.0,
                            ruler_benefit_pp=None, ruler_p_adj=None) == "数据不足"
    # PPL 显著变差不是“收益”：绝不能得到“可接受”
    v = tradeoff_verdict(benefit_pct=-5.0, p_adj=0.01, cost_pct=1.0)
    assert v.startswith("不可行") and "可接受" not in v, v
    v = tradeoff_verdict(benefit_pct=-0.3, p_adj=0.4, cost_pct=2.0)
    assert v.startswith("说服力有限"), v
    # 种子数不足与全表口径一致：p_adj 再小也不判定
    v = tradeoff_verdict(benefit_pct=5.0, p_adj=0.001, cost_pct=1.0, n_seeds=2)
    assert v.startswith("种子不足"), v
    assert tradeoff_verdict(5.0, 0.01, 1.0, n_seeds=MIN_SEEDS) == "可接受"
    for bad in (True, 3.9, "bad", -2):
        v = tradeoff_verdict(5.0, 0.01, 1.0, n_seeds=bad)
        assert v.startswith("种子"), (bad, v)
    v = tradeoff_verdict(5.0, 0.01, 1.0, n_seeds=5,
                         ruler_benefit_pp=1.0, ruler_p_adj=0.01,
                         ruler_n_seeds=2.9)
    assert v.startswith("种子") and "RULER" in v, v
    # 外部 JSON 的 bool/非数值不能伪装成 p=0 或抛 TypeError。
    assert tradeoff_verdict(5.0, False, 1.0, n_seeds=5).startswith(
        "说服力有限")
    assert tradeoff_verdict(True, 0.01, 1.0, n_seeds=5) == "数据不足"
    assert tradeoff_verdict(5.0, "bad", 1.0, n_seeds=5).startswith(
        "说服力有限")
    assert tradeoff_verdict(5.0, 0.01, "bad", n_seeds=5) == "数据不足"
    print("  [ok] tradeoff_verdict（不显著/大增幅/收益增幅比/负收益/种子不足）")


def test_tradeoff_verdict_covers_ruler_primary_metric():
    """权衡判定必须覆盖 RULER 主指标：只看 PPL 会误判长上下文涨点，
    也会漏掉 RULER 显著退步。"""
    # PPL 持平（不显著）+ RULER 显著涨点 + 增幅温和 → 有涨点且注明依据
    v = tradeoff_verdict(benefit_pct=0.1, p_adj=0.6, cost_pct=8.0,
                         n_seeds=5, ruler_benefit_pp=1.5, ruler_p_adj=0.01,
                         ruler_n_seeds=5)
    assert v.startswith("可接受") and "RULER" in v, v
    # RULER 单独显著改善 + 增幅超过 10×收益（RULER 独立阈值）→ 比值不利
    v = tradeoff_verdict(benefit_pct=0.1, p_adj=0.6, cost_pct=30.0,
                         n_seeds=5, ruler_benefit_pp=1.0, ruler_p_adj=0.01,
                         ruler_n_seeds=5)
    assert "收益-增幅比不利" in v and "RULER" in v, v
    # RULER 显著改善但 PPL 显著退步 → 权衡，而不是“不可行”
    v = tradeoff_verdict(benefit_pct=-3.0, p_adj=0.01, cost_pct=10.0,
                         n_seeds=5, ruler_benefit_pp=2.0, ruler_p_adj=0.01,
                         ruler_n_seeds=5)
    assert v.startswith("权衡") and "PPL" in v and "RULER" in v \
        and "不可行" not in v, v
    # PPL 显著改善但 RULER 显著退步 → 权衡（核心检索目标受损必须显式）
    v = tradeoff_verdict(benefit_pct=4.0, p_adj=0.01, cost_pct=5.0,
                         n_seeds=5, ruler_benefit_pp=-2.0, ruler_p_adj=0.01,
                         ruler_n_seeds=5)
    assert v.startswith("权衡") and "RULER" in v and "受损" in v, v
    # RULER 显著退步且无 PPL 涨点 → 不可行（不能因 PPL 不动就静默放过）
    v = tradeoff_verdict(benefit_pct=0.0, p_adj=0.6, cost_pct=5.0,
                         n_seeds=5, ruler_benefit_pp=-1.5, ruler_p_adj=0.01,
                         ruler_n_seeds=5)
    assert v.startswith("不可行") and "RULER" in v, v
    # 两个主指标都显著改善 → 依据两者；RULER 侧种子不足同样不能判定
    v = tradeoff_verdict(benefit_pct=3.0, p_adj=0.01, cost_pct=8.0,
                         n_seeds=5, ruler_benefit_pp=1.0, ruler_p_adj=0.01,
                         ruler_n_seeds=5)
    assert v.startswith("可接受") and "PPL+RULER" in v, v
    v = tradeoff_verdict(benefit_pct=3.0, p_adj=0.01, cost_pct=8.0,
                         n_seeds=5, ruler_benefit_pp=1.0, ruler_p_adj=0.01,
                         ruler_n_seeds=2)
    assert v.startswith("种子不足") and "RULER" in v, v
    print("  [ok] tradeoff_verdict（RULER 涨点/退步/权衡/独立阈值）")


def test_load_registered_hypotheses_tolerates_non_dict_json():
    """顶层为 list/null/str 的合法 JSON 也必须按“空 dict”降级而不是崩。"""
    tmp = tempfile.mkdtemp()
    try:
        p = os.path.join(tmp, "hypotheses.json")
        for bad in ([{"id": "H1"}], None, "text"):
            with open(p, "w", encoding="utf-8") as f:
                json.dump(bad, f, ensure_ascii=False)
            assert load_registered_hypotheses(p) == {}
        _write_json(p, {"hypotheses": [{"id": "H1"}]})
        assert load_registered_hypotheses(p)["hypotheses"][0]["id"] == "H1"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] load_registered_hypotheses：非 dict JSON 安全降级")


def test_build_summary_audit_zero_checked_is_warning_not_failure():
    """ok=True 但 checked=0 表示“没有可核对的 run”，不能渲染成 ❌ 未通过。"""
    tmp = tempfile.mkdtemp()
    try:
        out = build_summary(tmp, audit={"ok": True, "checked": 0,
                                        "mismatches": []})
        md = out["markdown"]
        assert "未检查受控变量" in md
        assert "❌ 未通过" not in md
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] build_summary：checked=0 的审计显示为未检查而非未通过")


def test_cross_variant_badcase_map_min_seed_even_when_empty():
    """最小种子零错误也注册为 []（参与共识判定）；
    最小种子无可用的 RULER 错误清单时退化到最小的有清单的种子；未评测不进表。"""
    cases99 = [{"sample_id": "x"}]
    # 最小种子（7）零错误，较大种子（99）有错 → 仍取最小种子的 []
    runs = {"lega": {7: {"ruler": {"badcases": []}},
                     99: {"ruler": {"badcases": cases99}}}}
    assert cross_variant_badcase_map(runs, ["lega"]) == {"lega": []}
    # 最小种子没有 RULER 结果 → 退化到最小有结果的种子
    runs2 = {"lega": {1: {}, 5: {"ruler": {"badcases": cases99}}}}
    assert cross_variant_badcase_map(runs2, ["lega"]) == {"lega": cases99}
    # 最小种子的 ruler 产物缺 badcases 键/非列表 → 不是“零错误”，
    # 退化到下一个有合法清单的种子；全无合法清单则排除该变体。
    runs3 = {"lega": {1: {"ruler": {"results": {}}},
                      2: {"ruler": {"badcases": "bad"}},
                      5: {"ruler": {"badcases": cases99}}}}
    assert cross_variant_badcase_map(runs3, ["lega"]) == {"lega": cases99}
    runs4 = {"lega": {1: {"ruler": {"results": {}}},
                      2: {"ruler": {"badcases": "bad"}}}}
    assert cross_variant_badcase_map(runs4, ["lega"]) == {}
    # 完全未跑 RULER → 不注册（无法参与对照）
    assert cross_variant_badcase_map({"lega": {1: {}}}, ["lega"]) == {}
    print("  [ok] cross_variant_badcase_map（最小种子优先，零错误显式注册）")


def test_build_summary_scopes_discovery_to_current_repro_run():
    """results/<tag> 跨多次运行复用时，汇总只纳入 repro.json 记录的本次
    eval/sanity（不同实验的观测不相配成对）。"""
    tmp = tempfile.mkdtemp()
    try:
        _write_json(os.path.join(tmp, "repro.json"),
                    {"variants": ["lega"], "seeds": [1], "smoke": False})
        _write_json(os.path.join(tmp, "eval_lega.json"),
                    {"variant": "lega", "seed": 1, "val_ppl": 9.0,
                     "val_loss": 2.0, "n_params": 10})
        _write_json(os.path.join(tmp, "sanity_lega.json"),
                    {"variant": "lega", "passed": True,
                     "initial": {"loss": 2.5},
                     "final": {"loss": 1.0, "token_acc": 0.4}})
        # 上一轮遗留：旧变体、旧种子、未通过的旧 sanity
        _write_json(os.path.join(tmp, "eval_standard.json"),
                    {"variant": "standard", "seed": 1, "val_ppl": 10.0,
                     "val_loss": 2.3, "n_params": 10})
        _write_json(os.path.join(tmp, "eval_lega_seed99.json"),
                    {"variant": "lega", "seed": 99, "val_ppl": 1.0,
                     "val_loss": 0.1, "n_params": 10})
        _write_json(os.path.join(tmp, "sanity_standard.json"),
                    {"variant": "standard", "passed": False,
                     "initial": {"loss": 9.9},
                     "final": {"loss": 9.9, "token_acc": 0.01}})
        out = build_summary(tmp)
        assert [r["variant"] for r in out["rows"]] == ["lega"]
        md = out["markdown"]
        assert "❌ 未通过" not in md           # 旧 sanity 不得混入
        assert "10.0000" not in md            # 旧 standard 的 PPL 不得混入
        assert "0.1000" not in md             # 旧种子的 val_loss 不得混入
        sig = json.load(open(os.path.join(tmp, "significance.json"),
                             encoding="utf-8"))
        assert sig["pairs"] == []
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] build_summary（repro.json 限定本次运行的变体/种子发现范围）")


def test_rel_pct_and_eff_cells():
    """相对变化计算：缺失与除零安全返回 None。"""
    assert _rel_pct(100.0, 110.0) == 10.0
    assert _rel_pct(50.0, 45.0) == -10.0
    assert _rel_pct(None, 10.0) is None
    assert _rel_pct(10.0, None) is None
    assert _rel_pct(0.0, 10.0) is None
    # 两端有限但百分比不可表示时同样是缺失，不能返回 ±inf 参与 max()。
    assert _rel_pct(1.0, 1e308) is None
    assert _rel_pct(1e-308, 1.0) is None
    # _eff_cell 读取 latency 条目的结构与运行时字段
    entry = {"n_params": 100, "efficiency": {
        "n_params": 100, "params_mb": 0.4,
        "flops_per_tok": {"256": {"total_flops_per_tok": 12345}},
        "peak_mem_mb_prefill": {"256": 3.5}}}
    from run_all import _eff_cell
    assert _eff_cell(entry, "params", 256) == 100
    assert _eff_cell(entry, "flops", 256) == 12345
    assert _eff_cell(entry, "mem", 256) == 3.5
    assert _eff_cell(None, "flops", 256) is None
    assert _eff_cell({"efficiency": {}}, "flops", 256) is None
    from run_all import _noise_floor_pct
    assert _noise_floor_pct(3.0, 4.0) == 5.0
    assert _noise_floor_pct(1e308, 1e308) is None
    assert _noise_floor_pct(-1.0, 2.0) is None
    print("  [ok] _rel_pct/_eff_cell（缺失安全）")


def test_result_files_empty_scope_does_not_fall_back_to_all():
    """repro.seeds=[] 表示本次 run 无种子，不能扫描目录里的旧结果。"""
    tmp = tempfile.mkdtemp()
    try:
        for name in ("eval_standard.json", "eval_lega_seed1.json"):
            _write_json(os.path.join(tmp, name), {"variant": "x"})
        assert _result_files(tmp, "eval", None) == sorted(
            os.path.join(tmp, n) for n in
            ("eval_standard.json", "eval_lega_seed1.json"))
        assert _result_files(tmp, "eval", []) == []
        assert len(_result_files(tmp, "eval", [1])) == 2
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] _result_files：空 seeds 范围不回退全量旧结果")


def test_effect_status_directions():
    # RULER 越大越好：全因子“加入变好”必须 mean_diff>0
    good = _mk("qgate", "standard", "RULER平均", 0.05, 0.01, lower=False)
    good["p_adj"] = 0.01
    bad = _mk("qgate", "standard", "RULER平均", -0.05, 0.01, lower=False)
    bad["p_adj"] = 0.01
    assert _effect_status(good, removing=False)["supported"]
    assert not _effect_status(bad, removing=False)["supported"]
    # 种子不足直接判无证据，即使 p 值很小
    tiny = _mk("qgate", "standard", "PPL", -0.5, 0.001, n=2)
    tiny["p_adj"] = 0.001
    assert not _effect_status(tiny, removing=False)["supported"]
    print("  [ok] _effect_status（方向、种子门槛）")


def test_effect_status_rejects_significant_but_fragile_effects():
    """显著但 |Δ|<ΔSD 或留一翻转不能算组件有用，更不能进入确认性证据。

    用户口径：0.2% 的提升若落在 0.5% 的种子方差内，即使 p<0.05 也没有
    统计意义。这里显式构造“显著但噪声内/留一翻转”的检验，确认判定被降级。
    """
    noisy = _mk("qgate", "gate_only", "PPL", -0.02, 0.005)
    noisy["p_adj"] = 0.01
    noisy["within_noise"] = True
    noisy["loo_sign_stable"] = True
    st = _effect_status(noisy, removing=False)
    assert not st["supported"] and st["fragile"] and not st["opposite"], st
    assert "噪声" in st["reason"], st
    fragile_loo = _mk("qgate", "gate_only", "PPL", -0.4, 0.005)
    fragile_loo["p_adj"] = 0.01
    fragile_loo["within_noise"] = False
    fragile_loo["loo_sign_stable"] = False
    st2 = _effect_status(fragile_loo, removing=False)
    assert not st2["supported"] and st2["fragile"], st2
    assert "留一" in st2["reason"], st2
    # 两套消融都显著但都不稳健：组件结论必须是“证据不稳健”，不能是“有用”
    tests = [
        _mk("lega_no_query", "lega", "PPL", 0.02, 0.005),
        _mk("qgate", "gate_only", "PPL", -0.02, 0.005),
    ]
    for t in tests:
        t.update({"within_noise": True, "loo_sign_stable": False})
    _apply_holm(tests)
    a = next(c for c in component_evidence(tests) if c["component"] == "A 查询项")
    assert a["supported"] == [] and a["fragile"] is True, a
    assert a["confirmatory_supported"] is False, a
    assert a["evidence_status"] == "exploratory", a
    assert a["verdict"].startswith("证据不稳健"), a
    print("  [ok] 组件证据拒绝“显著但噪声内/留一翻转”的脆弱收益")


def test_verify_controlled_setup_consistency():
    tmp = tempfile.mkdtemp()
    try:
        variants, seeds = ["standard", "lega"], [7]
        for v in variants:
            d = os.path.join(tmp, v)
            os.makedirs(d)
            _write_json(os.path.join(d, "config_used.json"), {
                "lr": 0.001, "steps": 10, "warmup": 2,
                "variant": v, "seed": 7, "device": "cpu", "tag": "t"})
            _write_json(os.path.join(d, "eval.json"), {
                "variant": v, "seed": 7, "data_fingerprint": "abc123",
                "code_commit": "deadbeef", "config_sha256": "shax",
                "steps": 10, "tokens_seen": 1024})
        audit = verify_controlled_setup(tmp, variants, seeds, 1,
                                        expected_data_digest="abc123")
        assert audit["ok"] and audit["checked"] == 2
        assert audit["meta_checked"] == 2 and not audit["meta_mismatches"]

        # 篡改数据指纹：同一 config 跑了不同数据不可比，必须报错
        ev_path = os.path.join(tmp, "lega", "eval.json")
        ev = json.load(open(ev_path, encoding="utf-8"))
        ev["data_fingerprint"] = "different"
        _write_json(ev_path, ev)
        try:
            verify_controlled_setup(tmp, variants, seeds, 1,
                                    expected_data_digest="abc123")
            raise AssertionError("数据指纹不一致应触发 SystemExit")
        except SystemExit:
            pass

        # 训练步数（steps）不一致同样必须报错
        ev["data_fingerprint"] = "abc123"
        ev["steps"] = 12
        _write_json(ev_path, ev)
        try:
            verify_controlled_setup(tmp, variants, seeds, 1,
                                    expected_data_digest="abc123")
            raise AssertionError("训练步数不一致应触发 SystemExit")
        except SystemExit:
            pass

        # 超参数（config_used.json）不一致必须报错
        ev["steps"] = 10
        _write_json(ev_path, ev)
        cfg_path = os.path.join(tmp, "lega", "config_used.json")
        cfg = json.load(open(cfg_path, encoding="utf-8"))
        cfg["lr"] = 0.002
        _write_json(cfg_path, cfg)
        try:
            verify_controlled_setup(tmp, variants, seeds, 1,
                                    expected_data_digest="abc123")
            raise AssertionError("学习率不一致应触发 SystemExit")
        except SystemExit:
            pass

        # 显式 null seed：审计应如实报告“不一致”
        cfg["lr"] = 0.001
        cfg["seed"] = None
        _write_json(cfg_path, cfg)
        try:
            verify_controlled_setup(tmp, variants, seeds, 1,
                                    expected_data_digest="abc123")
            raise AssertionError("config seed=null 应触发 SystemExit")
        except SystemExit:
            pass
        cfg["seed"] = 7
        _write_json(cfg_path, cfg)
        ev["seed"] = None
        _write_json(ev_path, ev)
        try:
            verify_controlled_setup(tmp, variants, seeds, 1,
                                    expected_data_digest="abc123")
            raise AssertionError("eval seed=null 应触发 SystemExit")
        except SystemExit:
            pass
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    # 缺少 eval.json 时只警告不阻塞（--skip_train 复用旧产物的场景）
    tmp2 = tempfile.mkdtemp()
    try:
        d = os.path.join(tmp2, "standard")
        os.makedirs(d)
        _write_json(os.path.join(d, "config_used.json"),
                    {"lr": 0.001, "variant": "standard", "seed": 1,
                     "device": "cpu", "tag": "t"})
        audit = verify_controlled_setup(tmp2, ["standard"], [1], 1)
        assert audit["ok"] and audit["meta_checked"] == 0
        assert audit["missing_evals"] == ["standard/seed1"]
    finally:
        shutil.rmtree(tmp2, ignore_errors=True)
    print("  [ok] verify_controlled_setup（数据/代码/步数/超参数双层审计）")


def test_verify_controlled_setup_checks_actual_data_order():
    """同 seed 下各变体必须消费同一训练批次序列，而不只是 config/seed 相同。"""
    with tempfile.TemporaryDirectory() as tmp:
        variants, seeds = ["standard", "lega"], [7]
        base = {"lr": 0.001, "steps": 10, "variant": None, "seed": 7,
                "device": "cpu", "tag": "t"}
        meta = {"data_fingerprint": "abc123", "code_commit": "deadbeef",
                "config_sha256": "shax", "steps": 10, "tokens_seen": 1024}
        for v in variants:
            d = os.path.join(tmp, v)
            os.makedirs(d)
            _write_json(os.path.join(d, "config_used.json"),
                        dict(base, variant=v))
            _write_json(os.path.join(d, "eval.json"), dict(
                meta, variant=v, seed=7, data_order_sha256="order-A"))
        audit = verify_controlled_setup(tmp, variants, seeds, 1,
                                        expected_data_digest="abc123")
        assert audit["ok"] and audit["data_order_checked"] == 1, audit
        assert not audit["data_order_missing"], audit
        # 同 seed 但实际采样序列不同：即使 config/seed 完全一致也必须失败
        ev = json.load(open(os.path.join(tmp, "lega", "eval.json"),
                            encoding="utf-8"))
        ev["data_order_sha256"] = "order-B"
        _write_json(os.path.join(tmp, "lega", "eval.json"), ev)
        try:
            verify_controlled_setup(tmp, variants, seeds, 1,
                                    expected_data_digest="abc123")
            raise AssertionError("数据顺序指纹不一致应触发 SystemExit")
        except SystemExit:
            pass
        # 旧产物缺少指纹：只警告，不伪造一致
        ev.pop("data_order_sha256")
        _write_json(os.path.join(tmp, "lega", "eval.json"), ev)
        audit = verify_controlled_setup(tmp, variants, seeds, 1,
                                        expected_data_digest="abc123")
        assert audit["ok"] and audit["data_order_missing"] == ["lega/seed7"]
    print("  [ok] 受控变量审计：同 seed 实际训练数据顺序指纹必须一致")


def test_verify_controlled_setup_rejects_mixed_attention_backend():
    """standard/legA 开了 SDPA 时实现路径不同；正式对比必须统一为显式路径。"""
    for sdpa, should_fail in ((True, True), (False, False)):
        tmp = tempfile.mkdtemp()
        try:
            variants, seeds = ["standard", "lega"], [7]
            for v in variants:
                d = os.path.join(tmp, v)
                os.makedirs(d)
                _write_json(os.path.join(d, "config_used.json"), {
                    "lr": 0.001, "steps": 10, "warmup": 2,
                    "use_sdpa_standard": sdpa,
                    "variant": v, "seed": 7, "device": "cpu", "tag": "t"})
                _write_json(os.path.join(d, "eval.json"), {
                    "variant": v, "seed": 7, "data_fingerprint": "abc123",
                    "code_commit": "deadbeef", "config_sha256": "shax",
                    "steps": 10, "tokens_seen": 1024})
            if should_fail:
                try:
                    verify_controlled_setup(tmp, variants, seeds, 1,
                                            expected_data_digest="abc123")
                    raise AssertionError("混合 SDPA/显式路径应触发 SystemExit")
                except SystemExit:
                    pass
            else:
                audit = verify_controlled_setup(
                    tmp, variants, seeds, 1, expected_data_digest="abc123")
                assert audit["ok"]
                assert audit["attention_backends"] == {
                    "standard": "explicit", "lega": "explicit"}
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    # lega_placebo 也注册了集中度项（a·c 的槽位保留、证据为噪声），实际
    # 走显式注意力；审计必须把它当成 explicit，否则会漏报 standard(SDPA)
    # 与 placebo(显式) 的实现路径混用。
    tmp = tempfile.mkdtemp()
    try:
        variants, seeds = ["standard", "lega_placebo"], [7]
        for v in variants:
            d = os.path.join(tmp, v)
            os.makedirs(d)
            _write_json(os.path.join(d, "config_used.json"), {
                "lr": 0.001, "steps": 10, "warmup": 2,
                "use_sdpa_standard": True,
                "variant": v, "seed": 7, "device": "cpu", "tag": "t"})
            _write_json(os.path.join(d, "eval.json"), {
                "variant": v, "seed": 7, "data_fingerprint": "abc123",
                "code_commit": "deadbeef", "config_sha256": "shax",
                "steps": 10, "tokens_seen": 1024})
        try:
            verify_controlled_setup(tmp, variants, seeds, 1,
                                    expected_data_digest="abc123")
        except SystemExit:
            pass
        else:
            raise AssertionError("lega_placebo 与 standard 的注意力路径不同，"
                                 "审计必须拒绝")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] verify_controlled_setup（混合注意力实现路径被拒绝）")


def test_verify_controlled_setup_reports_malformed_attention_switch():
    """config_used 的 use_sdpa_standard 非法时，审计应报告问题而非崩溃。"""
    tmp = tempfile.mkdtemp()
    try:
        for v in ("standard", "lega"):
            d = os.path.join(tmp, v)
            os.makedirs(d)
            _write_json(os.path.join(d, "config_used.json"), {
                "lr": 0.001, "steps": 10, "use_sdpa_standard": "maybe",
                "variant": v, "seed": 7, "device": "cpu", "tag": "t"})
            _write_json(os.path.join(d, "eval.json"), {
                "variant": v, "seed": 7, "data_fingerprint": "abc123",
                "code_commit": "deadbeef", "config_sha256": "shax",
                "steps": 10, "tokens_seen": 1024})
        try:
            verify_controlled_setup(tmp, ["standard", "lega"], [7], 1,
                                    expected_data_digest="abc123")
            raise AssertionError("非法 use_sdpa_standard 应触发 SystemExit")
        except SystemExit:
            pass
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] 受控变量审计：非法注意力开关报告问题而非崩溃")


def test_verify_controlled_setup_infers_missing_attention_switch():
    """缺省 use_sdpa_standard 时 train.py 用默认 True，审计不能静默跳过。

    config_used.json 没写该键时实际仍走 SDPA；standard/qgate 与必须显式
    softmax 的 lega 因而实现路径不同。旧实现只在“键存在”时才比较，导致
    一份完全省略该开关的配置绕过混合后端检查。
    """
    def fixture(variants):
        tmp = tempfile.mkdtemp()
        for v in variants:
            d = os.path.join(tmp, v)
            os.makedirs(d)
            # 直接写文件：不经测试 helper 的默认注入，模拟真实旧配置缺键。
            with open(os.path.join(d, "config_used.json"), "w",
                      encoding="utf-8") as f:
                json.dump({"lr": 0.001, "steps": 10, "variant": v,
                           "seed": 7, "device": "cpu", "tag": "t"}, f)
            _write_json(os.path.join(d, "eval.json"), {
                "variant": v, "seed": 7, "data_fingerprint": "abc123",
                "code_commit": "deadbeef", "config_sha256": "shax",
                "steps": 10, "tokens_seen": 1024})
        return tmp
    tmp = fixture(["standard", "lega"])
    try:
        try:
            verify_controlled_setup(tmp, ["standard", "lega"], [7], 1,
                                    expected_data_digest="abc123")
            raise AssertionError("缺键默认 True 导致的混合后端应触发 SystemExit")
        except SystemExit:
            pass
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    # 两个都缺键但都含集中度项：默认 True 不影响它们（仍需显式 softmax），
    # 实际后端一致为 explicit；审计通过并显式记录“推断自默认值”。
    tmp = fixture(["lega", "lega_no_len"])
    try:
        audit = verify_controlled_setup(
            tmp, ["lega", "lega_no_len"], [7], 1,
            expected_data_digest="abc123")
        assert audit["ok"] and not audit["mixed_attention"], audit
        assert audit["attention_key_seen"] is False
        assert set(audit["attention_backends"].values()) == {"explicit"}
        assert len(audit["attention_missing"]) == 2
        assert len(audit["attention_inferred"]) == 2
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] 缺省注意力开关按默认值推断，混合后端不再漏检")


def test_verify_controlled_setup_rejects_mixed_device_types():
    """不同变体跑在 CPU/CUDA 上时 PPL 差异无法归因于方法，必须拒绝。"""
    def fixture(devices):
        tmp = tempfile.mkdtemp()
        for v, dev in zip(("standard", "lega"), devices):
            d = os.path.join(tmp, v)
            os.makedirs(d)
            _write_json(os.path.join(d, "config_used.json"), {
                "lr": 0.001, "steps": 10, "variant": v, "seed": 7,
                "device": dev, "tag": "t"})
            _write_json(os.path.join(d, "eval.json"), {
                "variant": v, "seed": 7, "data_fingerprint": "abc123",
                "code_commit": "deadbeef", "config_sha256": "shax",
                "steps": 10, "tokens_seen": 1024})
        return tmp
    tmp = fixture(("cpu", "cuda:0"))
    try:
        try:
            verify_controlled_setup(tmp, ["standard", "lega"], [7], 1,
                                    expected_data_digest="abc123")
            raise AssertionError("混合 CPU/CUDA 应触发 SystemExit")
        except SystemExit:
            pass
        # 同类型不同卡不算混杂（仍会记录原始 device），审计通过
        tmp2 = fixture(("cuda:0", "cuda:1"))
        try:
            audit = verify_controlled_setup(
                tmp2, ["standard", "lega"], [7], 1,
                expected_data_digest="abc123")
            assert audit["ok"] and not audit["mixed_device"], audit
            assert set(audit["device_types"].values()) == {"cuda"}, audit
        finally:
            shutil.rmtree(tmp2, ignore_errors=True)
        # 旧产物缺 device 只警告，不伪造“一致”
        tmp3 = fixture(("cpu", "cpu"))
        evpath = os.path.join(tmp3, "lega", "config_used.json")
        cfg = json.load(open(evpath, encoding="utf-8"))
        cfg.pop("device")
        _write_json(evpath, cfg)
        try:
            audit = verify_controlled_setup(
                tmp3, ["standard", "lega"], [7], 1,
                expected_data_digest="abc123")
            assert audit["ok"] and audit["device_missing"] == ["lega/seed7"], audit
        finally:
            shutil.rmtree(tmp3, ignore_errors=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] 受控变量审计（混合设备类型被拒绝，缺 device 只告警）")


def test_stat_counts_finite_only():
    """_stat 的 n 与 mean/std 口径一致：NaN/Inf 不参与也不虚增样本量。"""
    st = _stat([1.0, float("nan"), 3.0, None])
    assert st["n"] == 2 and st["mean"] == 2.0
    assert abs(st["std"] - 2 ** 0.5) < 1e-12
    assert _stat([float("nan"), float("inf"), None]) is None
    assert _stat([2.5]) == {"mean": 2.5, "std": None, "n": 1}
    print("  [ok] _stat（NaN 不计入 n，与 mean/std 口径一致）")


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
    print("  [ok] PPL 配对：同 split 但 ctx/窗口数不一致的种子整对剔除")


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
    print("  [ok] RULER 配对：协议/评测格/单元样本数不一致的种子整对剔除")


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
        with mock.patch("run_all.load_registered_hypotheses",
                        return_value=reg), \
                mock.patch("run_all.verify_registration",
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
        with mock.patch("run_all.load_registered_hypotheses",
                        return_value=reg), \
                mock.patch("run_all.verify_registration",
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
        with mock.patch("run_all.load_registered_hypotheses",
                        return_value=reg), \
                mock.patch("run_all.verify_registration", return_value=intact):
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


def test_record_aborted_sanity_writes_reason_and_failed_checks():
    """sanity 闸门失败：台账必须有 aborted 记录、失败原因与未通过项。"""
    from lega.experiment import read_records
    tmp = tempfile.mkdtemp()
    try:
        results_dir = os.path.join(tmp, "results")
        os.makedirs(results_dir)
        _write_json(os.path.join(results_dir, "sanity_lega.json"), {
            "passed": False, "variant": "lega", "seed": 7,
            "checks": {"loss_below_threshold": False,
                       "next_token_shift_consistent": True},
            "initial": {"loss": 4.2}, "final": {"loss": 3.9,
                                                "token_acc": 0.1}})
        ledger = os.path.join(tmp, "experiments.jsonl")
        common = dict(
            run_id="rid1", tag="t", config_path="configs/tiny.json",
            config_sha256="a" * 64, config={"lr": 1e-3}, variants=["lega"],
            seeds=[7], smoke=False, argv=["run_all.py"],
            git={"commit": "c" * 40, "dirty": False}, data_fingerprint="fp",
            runtime={"torch": "x"}, out_root="out/t", results_dir=results_dir,
            time_start="2026-01-01 00:00:00")
        failed = record_aborted_sanity(ledger, common, "lega", 1, results_dir)
        assert failed["passed"] is False
        assert failed["failed_checks"] == ["loss_below_threshold"]
        recs = read_records(ledger)
        assert len(recs) == 1
        rec = recs[0]
        assert rec["status"] == "aborted"
        assert rec["extra"]["error"] == "sanity_gate_failed"
        assert rec["extra"]["variant"] == "lega" and rec["extra"]["returncode"] == 1
        assert rec["sanity"]["verdict"] == {"lega": "fail"}
        assert rec["sanity"]["failed"]["final_loss"] == 3.9
        assert rec["config"] == {"lr": 1e-3}
        # 结果文件缺失时同样留痕（failed.present=False），不崩
        os.remove(os.path.join(results_dir, "sanity_lega.json"))
        ledger2 = os.path.join(tmp, "exp2.jsonl")
        failed2 = record_aborted_sanity(ledger2, common, "lega", 2, results_dir)
        assert failed2["present"] is False
        assert read_records(ledger2)[0]["status"] == "aborted"
        # 文件看起来 PASS 但属于别的变体/种子/上一轮：aborted 证据必须
        # 写入失败原因并显式把摘要 passed 改为 False，避免自相矛盾。
        _write_json(os.path.join(results_dir, "sanity_lega.json"), {
            "passed": True, "variant": "lega", "seed": 7,
            "checks": {"a": True}})
        ledger3 = os.path.join(tmp, "exp3.jsonl")
        failed3 = record_aborted_sanity(
            ledger3, common, "lega", 0, results_dir, reason="stale_result")
        assert failed3["gate_failure_reason"] == "stale_result"
        assert failed3["passed"] is False
        rec3 = read_records(ledger3)[0]
        assert rec3["sanity"]["reason"] == "stale_result"
        assert rec3["sanity"]["failed"]["passed"] is False
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] 台账 aborted 记录：失败原因 + 未通过项 + 缺失结果容错")


def test_append_aborted_record_covers_run_chain_failures():
    """train/eval/ruler/analysis 任一阶段失败也要落 aborted（含阶段与变体）。"""
    from lega.experiment import read_records
    from run_all import append_aborted_record
    tmp = tempfile.mkdtemp()
    try:
        ledger = os.path.join(tmp, "experiments.jsonl")
        common = dict(
            run_id="rid2", tag="t", config_path="configs/tiny.json",
            config_sha256="b" * 64, config={"lr": 1e-3}, variants=["lega"],
            seeds=[7], smoke=False, argv=["run_all.py"], git={},
            data_fingerprint="fp", runtime={}, out_root="out/t",
            results_dir="results/t", time_start="2026-01-01 00:00:00")
        append_aborted_record(ledger, common, error="run_chain_failed",
                              stage="train/eval/ruler/analysis",
                              variant="lega", returncode=1)
        rec = read_records(ledger)[0]
        assert rec["status"] == "aborted"
        assert rec["extra"] == {"error": "run_chain_failed",
                                "stage": "train/eval/ruler/analysis",
                                "variant": "lega", "returncode": 1}
        assert rec["time_end"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] run 链失败落 aborted（阶段/变体/退出码）")


def test_aborted_record_marks_repro_failed_without_losing_start_facts():
    """失败轮次的 repro.json 必须标成 aborted，不能看起来仍在运行。"""
    from run_all import append_aborted_record
    tmp = tempfile.mkdtemp()
    try:
        results_dir = os.path.join(tmp, "results")
        os.makedirs(results_dir)
        repro_path = os.path.join(results_dir, "repro.json")
        start = {"status": None, "seed_audit": {"7": {"is_plan": True}},
                 "git": {"commit": "a" * 40}, "tag": "t"}
        with open(repro_path, "w", encoding="utf-8") as f:
            json.dump(start, f)
        ledger = os.path.join(tmp, "experiments.jsonl")
        common = dict(
            run_id="rid_fail", tag="t", config_path="configs/tiny.json",
            config_sha256="b" * 64, config={"lr": 1e-3}, variants=["lega"],
            seeds=[7], smoke=False, argv=["run_all.py"], git={},
            data_fingerprint="fp", runtime={}, out_root="out/t",
            results_dir=results_dir, time_start="2026-01-01 00:00:00")
        append_aborted_record(ledger, common, error="sanity_gate_failed",
                              stage="sanity", variant="lega", returncode=2,
                              sanity={"executed": True,
                                      "verdict": {"lega": "fail"}})
        with open(repro_path, "r", encoding="utf-8") as f:
            repro = json.load(f)
        assert repro["status"] == "aborted"
        assert repro["failure"]["error"] == "sanity_gate_failed"
        assert repro["failure"]["stage"] == "sanity"
        assert repro["failure"]["variant"] == "lega"
        assert repro["failure"]["returncode"] == 2
        assert repro["tracking_end"]["stage"] == "sanity"
        # 启动期事实（seed_audit/git/tag）不能被失败终态覆盖
        assert repro["seed_audit"] == start["seed_audit"]
        assert repro["git"] == start["git"] and repro["tag"] == "t"
        # 缺失/畸形 repro 不阻塞台账，也不抛异常
        assert mark_repro_failed(os.path.join(tmp, "missing"),
                                 error="x", stage="y") is False
        with open(repro_path, "w", encoding="utf-8") as f:
            f.write("{bad")
        assert mark_repro_failed(results_dir, error="x", stage="y") is False
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] repro.json 失败终态：保留启动事实 + 缺失/畸形容错")


def test_snapshot_existing_runs_preserves_failed_round_evidence():
    """失败轮次不能只剩 aborted：已产生的 run 证据要进台账快照。"""
    from lega.experiment import read_records
    from run_all import append_aborted_record
    tmp = tempfile.mkdtemp()
    try:
        out_root = os.path.join(tmp, "out")
        vdir = os.path.join(out_root, "lega")
        os.makedirs(vdir)
        with open(os.path.join(vdir, "run_manifest.json"), "w",
                  encoding="utf-8") as f:
            json.dump({"tag": "t", "variant": "lega", "seed": 7,
                       "seed_report": {"seed": 7, "seed_verified": True},
                       "git": {"commit": "a" * 40}, "data": {"digest": "d" * 16}}, f)
        with open(os.path.join(vdir, "config_used.json"), "w",
                  encoding="utf-8") as f:
            json.dump({"lr": 1e-3}, f)
        with open(os.path.join(vdir, "eval.json"), "w", encoding="utf-8") as f:
            json.dump({"variant": "lega", "seed": 7, "val_ppl": 12.0}, f)
        with open(os.path.join(vdir, "train_log.jsonl"), "w",
                  encoding="utf-8") as f:
            f.write(json.dumps({"step": 1, "loss": 3.0}) + "\n")
        start = time.time()
        common = dict(
            run_id="rid_partial", tag="t", config_path="configs/tiny.json",
            config_sha256="a" * 64, config={"lr": 1e-3},
            out_root=out_root, variants=["lega"], seeds=[7],
            smoke=False, argv=["run_all.py"], git={},
            data_fingerprint="fp", runtime={},
            results_dir=os.path.join(tmp, "results"),
            time_start=time.strftime("%Y-%m-%d %H:%M:%S",
                                     time.localtime(start)))
        runs = _snapshot_existing_runs(common)
        assert len(runs) == 1, runs
        assert runs[0]["variant"] == "lega" and runs[0]["seed"] == 7
        assert runs[0]["provenance"]["seed_report"]["seed"] == 7
        assert runs[0]["result"]["val_ppl"] == 12.0
        assert "run_manifest.json" in runs[0]["files"]
        assert "model.pt" not in runs[0]["files"]   # 未产生就不伪造
        ledger = os.path.join(tmp, "experiments.jsonl")
        append_aborted_record(ledger, common, error="run_chain_failed",
                              stage="eval", variant="lega", returncode=1,
                              runs=runs)
        rec = read_records(ledger)[0]
        assert rec["status"] == "aborted" and rec["runs"] == runs
        # 上一轮残留（manifest mtime 早于本轮开始）不得当成本轮证据；
        # 只早 0.5 秒同样必须排除，不能有“2 秒宽限”把上一轮末尾收进来。
        old = start - 3600
        mpath = os.path.join(vdir, "run_manifest.json")
        os.utime(mpath, (old, old))
        assert _snapshot_existing_runs(common) == []
        floor = time.mktime(time.strptime(common["time_start"],
                                         "%Y-%m-%d %H:%M:%S"))
        os.utime(mpath, (floor - 0.5, floor - 0.5))
        assert _snapshot_existing_runs(common) == []
        # 开始时刻不可解析时无法证明目录属于本轮：保守返回空，不能全收。
        os.utime(mpath, (time.time() + 1, time.time() + 1))
        assert _snapshot_existing_runs(dict(common, time_start="bad")) == []
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] 失败轮次快照已产生的 run 证据；旧目录不冒充本轮")


def test_check_sanity_results_reads_back_verdict_not_exit_code():
    """退出码 0 不能代替证据：逐个复核 sanity 结果文件。"""
    tmp = tempfile.mkdtemp()
    try:
        def _write(name, obj):
            with open(os.path.join(tmp, name), "w", encoding="utf-8") as f:
                json.dump(obj, f)

        checks = {"loss_below_threshold": True,
                  "next_token_shift_consistent": True}
        _write("sanity_good.json", {"passed": True, "variant": "good",
                                    "checks": checks})
        _write("sanity_bad.json", {"passed": False, "variant": "bad",
                                   "checks": dict(checks,
                                                  loss_below_threshold=False)})
        # passed=true 但 checks 为空：sanity_summary 判 malformed/不通过
        _write("sanity_empty.json", {"passed": True, "variant": "empty",
                                     "checks": {}})
        _write("sanity_wrongvariant.json",
               {"passed": True, "variant": "standard", "checks": checks})
        _write("sanity_wrongseed.json",
               {"passed": True, "variant": "wrongseed", "seed": 99,
                "checks": checks})
        with open(os.path.join(tmp, "sanity_unparseable.json"), "w",
                  encoding="utf-8") as f:
            f.write("{bad")
        verdict, failures = check_sanity_results(
            ["good", "bad", "empty", "unparseable", "missing",
             "wrongvariant"], tmp)
        assert verdict == {"good": "pass"}, verdict
        by_variant = {x["variant"]: x for x in failures}
        assert set(by_variant) == {"bad", "empty", "unparseable", "missing",
                                   "wrongvariant"}
        assert by_variant["wrongvariant"]["reason"] == "variant_mismatch"
        # seed 只在显式期望时核对；不传 expected 时 wrongseed 正常通过
        verdict2, failures2 = check_sanity_results(
            ["good", "wrongseed"], tmp, expected_seed=7)
        assert verdict2 == {"good": "pass"}, verdict2
        assert failures2[0]["variant"] == "wrongseed"
        assert failures2[0]["reason"] == "seed_mismatch"
        assert by_variant["bad"]["reason"] == "not_passed"
        assert by_variant["empty"]["reason"] == "not_passed"
        assert by_variant["missing"]["reason"] == "missing_or_unparseable"
        # 文件存在但 JSON 不可解析：present=true 但 passed=None，复核同样阻断
        assert by_variant["unparseable"]["reason"] == "not_passed"
        assert by_variant["bad"]["path"].endswith("sanity_bad.json")
        # 本轮 sanity 没有真正写回：旧文件即使 PASS 也不能当作本次证据。
        import time
        good_path = os.path.join(tmp, "sanity_good.json")
        old_ts = time.time() - 3600
        os.utime(good_path, (old_ts, old_ts))
        v2, f2 = check_sanity_results(["good"], tmp, not_before=time.time())
        assert v2 == {} and len(f2) == 1
        assert f2[0]["reason"] == "stale_result"
        # mtime 新鲜时同一文件正常通过
        v3, f3 = check_sanity_results(["good"], tmp, not_before=0.0)
        assert v3 == {"good": "pass"} and not f3
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] sanity 结果读回复核：passed=false/空 checks/缺失/陈旧均阻断")


def test_abort_ledger_context_records_postrun_stage_failure():
    """训练链之后任何阶段失败都必须把 started 收成 aborted（含当前阶段）。"""
    from lega.experiment import read_records
    tmp = tempfile.mkdtemp()
    try:
        ledger = os.path.join(tmp, "experiments.jsonl")
        common = dict(
            run_id="rid3", tag="t", config_path="configs/tiny.json",
            config_sha256="c" * 64, config={"lr": 1e-3}, variants=["lega"],
            seeds=[7], smoke=False, argv=["run_all.py"], git={},
            data_fingerprint="fp", runtime={}, out_root="out/t",
            results_dir="results/t", time_start="2026-01-01 00:00:00")
        stage = {"name": "latency"}
        try:
            with abort_ledger_on_failure(ledger, common, lambda: stage["name"]):
                stage["name"] = "plots"
                raise subprocess.CalledProcessError(3, ["plots-cmd"])
        except subprocess.CalledProcessError as e:
            assert e.returncode == 3
        else:
            raise AssertionError("原异常必须继续抛出")
        rec = read_records(ledger)[0]
        assert rec["status"] == "aborted"
        assert rec["extra"] == {"error": "postrun_stage_failed",
                                "stage": "plots", "returncode": 3}
        # 正常退出不得写 aborted；记录阶段快照而不是事后猜测
        ledger2 = os.path.join(tmp, "exp2.jsonl")
        stage["name"] = "summary"
        with abort_ledger_on_failure(ledger2, common, lambda: stage["name"]):
            pass
        assert read_records(ledger2) == []
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] postrun 阶段失败自动补 aborted（阶段/退出码）")


def test_summary_tolerates_malformed_sanity_and_baseline_tuning():
    """sanity/tuning 的嵌套字段被手改/旧格式污染时，汇总不能崩。"""
    tmp = tempfile.mkdtemp()
    try:
        _write_json(os.path.join(tmp, "eval_standard.json"), {
            "variant": "standard", "seed": 1, "val_ppl": 10.0,
            "val_loss": 2.3, "n_params": 10, "split": "val"})
        _write_json(os.path.join(tmp, "sanity_standard.json"), {
            "variant": [1, 2], "passed": True, "initial": [1, 2],
            "final": {"loss": "bad", "token_acc": [1, 2]}})
        _write_json(os.path.join(tmp, "baseline_tuning.json"), {
            "grid": [{"lr": [1, 2], "weight_decay": {"x": 1}}],
            "trials": 5, "selected": {"lr": [1]},
            "default_trial": [], "selection_seeds": {"x": 1}})
        out = build_summary(tmp)
        assert "standard" in out["markdown"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] 汇总：sanity/tuning 嵌套字段畸形安全跳过")


def test_summary_ignores_sanity_for_other_seed():
    """同 tag 换种子重跑时，旧种子的 sanity PASS 不能冒充本轮证据。

    run_all 只用首个种子跑 sanity；results/<tag> 被复用且 --skip_sanity 时，
    旧 sanity_*.json 仍留在目录里。汇总只按变体过滤会把它渲染成本轮闸门，
    让“本轮未跑 sanity”看上去有 PASS 证据。
    """
    tmp = tempfile.mkdtemp()
    try:
        _write_json(os.path.join(tmp, "repro.json"), {
            "variants": ["standard", "lega"], "seeds": [7]})
        _write_json(os.path.join(tmp, "sanity_standard.json"), {
            "variant": "standard", "seed": 7, "passed": True,
            "initial": {"loss": 4.0}, "final": {"loss": 0.1, "token_acc": 1.0}})
        _write_json(os.path.join(tmp, "sanity_lega.json"), {
            "variant": "lega", "seed": 1, "passed": True,
            "initial": {"loss": 4.0}, "final": {"loss": 9.9, "token_acc": 0.0}})
        out = build_summary(tmp)
        md = out["markdown"]
        assert "0.1000" in md, md
        # 旧种子（1）的 sanity 必须被剔除，不能显示其最终 loss/准确率
        assert "9.9000" not in md, md
        assert "1.0000" in md  # 当前种子 sanity 的 token 准确率
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] 汇总：按首个种子剔除旧 sanity 证据")


def test_registered_hypothesis_rejects_missing_or_invalid_n():
    """n 缺失/非法时不能仅凭 p_adj 显著就判“支持”。"""
    tests = [
        {"candidate": "lega", "reference": "standard", "metric": "PPL",
         "mean_diff": -0.5, "lower_is_better": True, "p": 0.01,
         "p_adj": 0.01},
        {"candidate": "lega2", "reference": "standard", "metric": "PPL",
         "mean_diff": -0.5, "lower_is_better": True, "p": 0.01,
         "p_adj": 0.01, "n": "bad"},
        {"candidate": "lega3", "reference": "standard", "metric": "PPL",
         "mean_diff": -0.5, "lower_is_better": True, "p": 0.01,
         "p_adj": 0.01, "n": 2},
    ]
    reg = {"hypotheses": [
        {"id": "H1", "metric": "PPL", "candidate": "lega",
         "reference": "standard", "direction": "candidate_better"},
        {"id": "H2", "metric": "PPL", "candidate": "lega2",
         "reference": "standard", "direction": "candidate_better"},
        {"id": "H3", "metric": "PPL", "candidate": "lega3",
         "reference": "standard", "direction": "candidate_better"},
    ]}
    ev = {h["id"]: h for h in evaluate_registered_hypotheses(reg, tests)}
    assert ev["H1"]["status"] == "未定" and "样本量" in ev["H1"]["detail"]
    assert ev["H2"]["status"] == "未定" and "样本量" in ev["H2"]["detail"]
    assert ev["H3"]["status"] == "未定" and "种子不足" in ev["H3"]["detail"]
    print("  [ok] 预注册判定拒绝缺 n/非法 n 的“显著”证据")


def test_registration_paths_tolerate_malformed_hypotheses_and_status():
    """登记表 hypotheses 非列表/条目非对象、完整性 status 非字符串均不崩。"""
    assert evaluate_registered_hypotheses({"hypotheses": 5}, [1, None]) == []
    out = evaluate_registered_hypotheses(
        {"hypotheses": [1, {"id": "H1", "metric": "PPL",
                            "candidate": "lega", "reference": "standard",
                            "direction": "candidate_better"}]}, [1])
    assert len(out) == 1 and out[0]["status"] == "未检验"
    assert apply_integrity_gate({"x": 1}, {"confirmatory_valid": True}) == {"x": 1}
    hyp = [{"confirmatory_valid": True}]
    apply_integrity_gate(hyp, {"confirmatory_valid": True})
    # 损坏的“有效性”字段不能经 truthiness 授予确认性资格。
    for bogus in ("false", 1, "yes", [1], {"ok": True}, None):
        hp = [{"confirmatory_valid": True}]
        apply_integrity_gate(hp, {"confirmatory_valid": bogus})
        assert hp[0]["confirmatory_valid"] is False, bogus
    apply_scope_gate(hyp, {"scope": {"config": "x"}, "checked": False,
                           "problems": [1]})
    assert hyp[0]["confirmatory_valid"] is False
    # checked 字段损坏时不能经 truthiness 当成“已核对在范围内”。
    for bogus in ("false", 1, "yes", [1], {"ok": True}, None):
        hp = [{"confirmatory_valid": True}]
        apply_scope_gate(hp, {"scope": {"config": "x"},
                              "checked": bogus, "problems": []})
        assert hp[0]["confirmatory_valid"] is False, bogus
    md = _render_registration_integrity(
        {"status": [1, 2], "problems": [1], "warnings": 5},
        {"scope": 5, "checked": False, "problems": "x"})
    assert any("预注册完整性" in line for line in md)
    assert cross_variant_badcase_map({"lega": {1: {"ruler": 5}}},
                                     ["lega"]) == {}
    assert cross_variant_badcase_map({"lega": [1]}, ["lega"]) == {}
    print("  [ok] 预注册路径：畸形 hypotheses/status/slot 安全降级")


def test_factorial_none_cell_is_within_gate_control():
    """2^3 的 none 格必须是 gate_only：standard 无门会混入额外因子。"""
    from run_all import CONTROL_CELLS, FACTORIAL_CELLS, LOO_CELLS
    cells = dict(FACTORIAL_CELLS)
    assert cells["无 A/B/C（有门）"] == "gate_only", cells
    variants = [v for _, v in FACTORIAL_CELLS]
    assert "standard" not in variants, variants
    assert len(set(variants)) == 8 and set(variants) == {
        "gate_only", "qgate", "lega_conc_only", "lega_len_only",
        "lega_no_len", "lega_no_conc", "lega_no_query", "lega"}, variants
    assert [v for _, v in CONTROL_CELLS] == ["standard"]
    assert [v for _, v in LOO_CELLS] == [
        "lega", "lega_no_query", "lega_no_conc", "lega_no_len"]
    print("  [ok] 全因子 none=gate_only，standard 仅作范围外无门基线")


def test_ablation_coverage_requires_min_seeds_per_cell():
    """消融矩阵“覆盖完整”必须每格≥MIN_SEEDS，单种子冒烟格不能算覆盖。"""
    cells = (("A", "a"), ("B", "b"), ("C", "c"))
    rows = [{"variant": "a", "n_seeds": 3},
            {"variant": "b", "n_seeds": 1}]
    done, missing, under = _ablation_coverage(cells, {"a", "b"}, 
                                               {r["variant"]: r for r in rows})
    assert done == [("A", "a")], done
    assert missing == [("B", "b"), ("C", "c")], missing
    assert under == [("B", "b", 1)], under
    # 无 n_seeds 的旧 row 不得默认成“种子充足”
    done, missing, under = _ablation_coverage(
        cells, {"a"}, {"a": {"variant": "a"}})
    assert not done and under == [("A", "a", 0)], (done, missing, under)
    # bool/小数/负数不是有效种子数，不能截断成“覆盖完整”
    for bad in (True, 3.9, -3, "bad"):
        done, missing, under = _ablation_coverage(
            cells, {"a"}, {"a": {"variant": "a", "n_seeds": bad}})
        assert not done and under == [("A", "a", 0)], (bad, done, missing, under)
    done, _, _ = _ablation_coverage(
        cells, {"a"}, {"a": {"variant": "a", "n_seeds": "3"}})
    assert done == [("A", "a")], done
    print("  [ok] 消融矩阵覆盖按每格种子数判定，单种子格明确标为不足")


def test_single_factor_audit_and_pair_construction():
    """单因素消融必须只改一个 A/B/C 因子；B-only/C-only 不能与 Full 配对。"""
    audit = single_factor_audit()
    assert audit["ok"] and not audit["problems"], audit
    assert len(audit["contrasts"]) == 6
    assert {c["factor"] for c in audit["contrasts"]} == {"A", "B", "C"}
    assert {c["direction"] for c in audit["contrasts"]} == {"remove", "add"}
    for c in audit["contrasts"]:
        assert c["changed_factors"] == [c["factor"]], c
    assert len(SINGLE_FACTOR_PAIRS) == 6
    assert len(set(SINGLE_FACTOR_PAIRS)) == 6

    # 去一法：Full−X vs Full 只改变 X
    assert single_factor_change("lega_no_query", "lega") == "A"
    assert single_factor_change("lega_no_conc", "lega") == "B"
    assert single_factor_change("lega_no_len", "lega") == "C"
    # 全因子增量：X-only vs gate_only 只加入 X
    assert single_factor_change("qgate", "gate_only") == "A"
    assert single_factor_change("lega_conc_only", "gate_only") == "B"
    assert single_factor_change("lega_len_only", "gate_only") == "C"
    # B-only/C-only vs Full 同时改变两个因子，不能进单因素对比表
    assert changed_gate_factors("lega_conc_only", "lega") == {"A", "C"}
    assert changed_gate_factors("lega_len_only", "lega") == {"A", "B"}
    assert single_factor_change("lega_conc_only", "lega") is None
    assert single_factor_change("lega_len_only", "lega") is None
    # vs standard / LEGA vs qgate 是整体方法对比，不是门控内单因素
    assert single_factor_change("lega", "standard") is None
    assert single_factor_change("lega", "qgate") is None

    # 组件有效性判定实际消费的每个检验对都必须是已审计的单因素对比
    expected_factor = {"A 查询项": "A", "B 集中度证据": "B",
                       "C 长度校准": "C"}
    assert {name for name, *_ in COMPONENT_TESTS} == set(expected_factor)
    for name, loo_pair, add_pair, _primary in COMPONENT_TESTS:
        factor = expected_factor[name]
        assert loo_pair in SINGLE_FACTOR_PAIRS, (name, loo_pair)
        assert add_pair in SINGLE_FACTOR_PAIRS, (name, add_pair)
        assert single_factor_change(*loo_pair) == factor, (name, loo_pair)
        assert single_factor_change(*add_pair) == factor, (name, add_pair)

    runs = {"standard": {1: {}}, "gate_only": {1: {}},
            "qgate": {1: {}}, "lega_conc_only": {1: {}},
            "lega_len_only": {1: {}}, "lega": {1: {}},
            "lega_no_len": {1: {}}, "lega_no_conc": {1: {}},
            "lega_no_query": {1: {}}}
    pairs = build_comparison_pairs(runs, {"hypotheses": []})
    for c, r in SINGLE_FACTOR_PAIRS:
        assert (c, r) in pairs, (c, r)
    assert ("lega_conc_only", "lega") not in pairs
    assert ("lega_len_only", "lega") not in pairs
    # 非 vs standard 的对比要么是单因素消融，要么是显式登记的整体假设
    for c, r in pairs:
        if r != "standard":
            assert (c, r) in SINGLE_FACTOR_PAIRS, (c, r)
    reg = {"hypotheses": [{"metric": "PPL", "candidate": "lega",
                           "reference": "qgate"}]}
    pairs = build_comparison_pairs(runs, reg)
    assert ("lega", "qgate") in pairs
    assert single_factor_change("lega", "qgate") is None
    print("  [ok] 单因素审计：6 组对比只改一个因子，多因子对比不混入")


def test_single_factor_audit_requires_complete_and_audited_contrasts():
    """审计表少一条/重复一条、或组件检验绕过审计表，都必须显式失败。"""
    with mock.patch("run_all.SINGLE_FACTOR_CONTRASTS",
                    SINGLE_FACTOR_CONTRASTS[:-1]):
        audit = single_factor_audit()
        assert not audit["ok"], audit
        assert any("缺少单因素对比" in p for p in audit["problems"]), audit
    with mock.patch("run_all.SINGLE_FACTOR_CONTRASTS",
                    SINGLE_FACTOR_CONTRASTS + (SINGLE_FACTOR_CONTRASTS[0],)):
        audit = single_factor_audit()
        assert not audit["ok"], audit
        assert any("重复的单因素对比" in p for p in audit["problems"]), audit
    bad_component = COMPONENT_TESTS + (
        ("D 额外组件", ("lega_conc_only", "lega"),
         ("lega_len_only", "lega"), "PPL"),)
    with mock.patch("run_all.COMPONENT_TESTS", bad_component):
        audit = single_factor_audit()
        assert not audit["ok"], audit
        assert any("未审计的单因素对比" in p
                   for p in audit["problems"]), audit
    print("  [ok] 单因素审计拒绝缺失/重复对比与绕过审计表的组件检验")


def test_single_factor_audit_rejects_broken_design_metadata():
    """2^3 设计元数据、方向与组件检验本身损坏时，审计也必须显式失败。"""
    # 8 个变体名都在，但两个变体拥有相同因子集合：某个因子格实际不存在。
    bad_factors = dict(GATE_FACTORS)
    bad_factors["qgate"] = frozenset({"A", "B"})
    with mock.patch("run_all.GATE_FACTORS", bad_factors):
        audit = single_factor_audit()
        assert not audit["ok"], audit
        assert any("完整 2^3" in p for p in audit["problems"]), audit

    # 把“去掉 A”写成“加入 A”：方向是实验定义的一部分，不能靠展示字段糊弄。
    swapped = (("A", "add", "lega_no_query", "lega"),) + tuple(
        c for c in SINGLE_FACTOR_CONTRASTS if c != SINGLE_FACTOR_CONTRASTS[0])
    with mock.patch("run_all.SINGLE_FACTOR_CONTRASTS", swapped):
        audit = single_factor_audit()
        assert not audit["ok"], audit
        assert any("方向" in p for p in audit["problems"]), audit

    # 组件检验缺一项/重复一项/名称无法映射/主指标非法，都不能静默通过。
    with mock.patch("run_all.COMPONENT_TESTS", COMPONENT_TESTS[:2]):
        audit = single_factor_audit()
        assert not audit["ok"], audit
        assert any("缺少组件检验" in p for p in audit["problems"]), audit
    with mock.patch("run_all.COMPONENT_TESTS", COMPONENT_TESTS + (COMPONENT_TESTS[0],)):
        audit = single_factor_audit()
        assert not audit["ok"], audit
        assert any("重复的检验定义" in p for p in audit["problems"]), audit
    bad_name = COMPONENT_TESTS + (
        ("D 额外组件", ("lega_no_query", "lega"),
         ("qgate", "gate_only"), "PPL"),)
    with mock.patch("run_all.COMPONENT_TESTS", bad_name):
        audit = single_factor_audit()
        assert not audit["ok"], audit
        assert any("无法映射到 A/B/C" in p for p in audit["problems"]), audit
    bad_metric = (COMPONENT_TESTS[0][:3] + ("loss",),) + COMPONENT_TESTS[1:]
    with mock.patch("run_all.COMPONENT_TESTS", bad_metric):
        audit = single_factor_audit()
        assert not audit["ok"], audit
        assert any("不是已登记指标" in p for p in audit["problems"]), audit
    # 组件把去一法/全因子两套对比对调：因子虽然都是 A，方向仍然错误。
    swapped_component = (("A 查询项", ("qgate", "gate_only"),
                          ("lega_no_query", "lega"), "PPL"),) + COMPONENT_TESTS[1:]
    with mock.patch("run_all.COMPONENT_TESTS", swapped_component):
        audit = single_factor_audit()
        assert not audit["ok"], audit
        assert any("方向为 add" in p or "方向为 remove" in p
                   for p in audit["problems"]), audit

    # 外部损坏键不能让“是否单因素”的查询崩溃，只能返回 None/不可归因。
    assert changed_gate_factors([], "lega") is None
    assert changed_gate_factors("lega", {"bad": "key"}) is None
    unknown_factor = dict(GATE_FACTORS, mystery=frozenset({"D"}))
    with mock.patch("run_all.GATE_FACTORS", unknown_factor):
        assert changed_gate_factors("mystery", "gate_only") is None
        assert single_factor_change("mystery", "gate_only") is None
    print("  [ok] 单因素审计拒绝不完整 2^3、错误方向与损坏的组件检验")


def test_paired_comparisons_marks_contrast_type():
    """配对结果必须机器可读地标出 single_factor/overall，避免事后误归因。"""
    runs = {}
    for v, base in (("standard", 12.0), ("gate_only", 11.0),
                    ("qgate", 10.0), ("lega_conc_only", 9.0),
                    ("lega_len_only", 9.5), ("lega", 8.0),
                    ("lega_no_len", 8.5), ("lega_no_conc", 8.2),
                    ("lega_no_query", 8.8)):
        runs[v] = {s: {"eval": {"val_ppl": base + 0.01 * s, "split": "val"}}
                   for s in (1, 2, 3)}
    pairs = list(SINGLE_FACTOR_PAIRS) + [
        ("lega", "standard"), ("lega", "qgate"),
        ("lega_conc_only", "lega"),
        # 方向写反的单因子对没有登记在 SINGLE_FACTOR_PAIRS 中，不能
        # 仅凭“对称差=1”就获得可归因标记。
        ("lega", "lega_no_query")]
    out = _paired_comparisons(
        runs, pairs, "PPL", lambda slot: _ppl_of(slot.get("eval")),
        lower_better=True)
    by = {(t["candidate"], t["reference"]): t for t in out}
    for c, r in SINGLE_FACTOR_PAIRS:
        t = by[(c, r)]
        assert t["contrast_type"] == "single_factor", (c, r, t)
        assert len(t["changed_factors"]) == 1, t
    assert by[("lega", "standard")]["contrast_type"] == "overall"
    assert by[("lega", "qgate")]["contrast_type"] == "overall"
    assert by[("lega_conc_only", "lega")]["contrast_type"] == "overall"
    assert by[("lega", "lega_no_query")]["contrast_type"] == "overall"
    print("  [ok] 配对检验结果带 single_factor/overall 机器可读标记")


def test_build_summary_excludes_multifactor_ablation_pairs():
    """汇总不得再把 B-only/C-only 与 Full 的差异当成单组件消融。"""
    tmp = tempfile.mkdtemp()
    try:
        bases = {"standard": 11.0, "gate_only": 10.5, "qgate": 10.2,
                 "lega_conc_only": 10.0, "lega_len_only": 10.3,
                 "lega": 9.5, "lega_no_len": 9.7, "lega_no_conc": 9.6,
                 "lega_no_query": 9.9}
        for v, base in bases.items():
            for s in (1, 2, 3):
                _write_json(os.path.join(tmp, f"eval_{v}_seed{s}.json"), {
                    "variant": v, "seed": s, "split": "val",
                    "val_ppl": base + 0.01 * s, "val_loss": 2.3,
                    "n_params": 100, "n_gate_extra": 0})
        out = build_summary(tmp)
        sig = json.load(open(os.path.join(tmp, "significance.json"),
                             encoding="utf-8"))
        pairs = {(t["candidate"], t["reference"]) for t in sig["pairs"]}
        assert ("lega_conc_only", "lega") not in pairs
        assert ("lega_len_only", "lega") not in pairs
        for pair in SINGLE_FACTOR_PAIRS:
            assert pair in pairs, pair
        assert sig["single_factor_audit"]["ok"] is True
        assert "单因素消融审计" in out["markdown"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] 汇总排除 B-only/C-only vs Full 的多因子“消融”对比")


def test_build_summary_only_reuses_ablation_when_scope_is_true_bool():
    """repro.ablation.executed 必须是真正的 true 才复用旧病灶结果。

    "false"/1 经 truthiness 会把上一轮 ablation_*.json 冒充成本轮证据，
    让“本轮 --skip_ablation”的汇总仍带着推理期因果结论。
    """
    tmp = tempfile.mkdtemp()
    try:
        _write_json(os.path.join(tmp, "repro.json"), {
            "variants": ["lega"], "seeds": [1], "smoke": "false",
            "ablation": {"executed": "false", "variants": ["lega"]}})
        _write_json(os.path.join(tmp, "ablation_lega.json"), {
            "variant": "lega", "seed": 1,
            "lesions": [{"lesion": "no_query", "label": "去查询项",
                         "applicable": True, "applied": True,
                         "delta_loss_mean": 0.1, "n_windows": 10}]})
        build_summary(tmp)
        sig = json.load(open(os.path.join(tmp, "significance.json"),
                             encoding="utf-8"))
        assert sig["causal_ablation"] == [], sig["causal_ablation"]
        # 显式 executed=true 时才消费同一份文件（n=1 仍会如实报告）。
        _write_json(os.path.join(tmp, "repro.json"), {
            "variants": ["lega"], "seeds": [1], "smoke": False,
            "ablation": {"executed": True, "variants": ["lega"]}})
        build_summary(tmp)
        sig = json.load(open(os.path.join(tmp, "significance.json"),
                             encoding="utf-8"))
        assert len(sig["causal_ablation"]) == 1, sig["causal_ablation"]
        assert sig["causal_ablation"][0]["lesion"] == "no_query"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] 消融证据只在 executed 为真正 true 时复用，"
          "malformed repro 不会冒充本轮结论")


def test_verify_controlled_setup_rejects_eval_protocol_mismatch():
    """同一训练 config 但独立评测用了不同窗口/上下文，PPL 不可配对。"""
    with tempfile.TemporaryDirectory() as tmp:
        variants, seeds = ["standard", "lega"], [7]
        base = {"split": "test", "eval_ctx": 512, "eval_iters": 50,
                "eval_iters_requested": 50, "val_windows": 80,
                "val_tokens_evaluated": 25600,
                "selection_set_is_eval_set": False}
        for v in variants:
            d = os.path.join(tmp, v)
            os.makedirs(d)
            _write_json(os.path.join(d, "config_used.json"), {
                "lr": 0.001, "steps": 10, "variant": v, "seed": 7,
                "device": "cpu", "tag": "t"})
            _write_json(os.path.join(d, "eval.json"), {
                "variant": v, "seed": 7, "data_fingerprint": "abc123",
                "code_commit": "deadbeef", "config_sha256": "shax",
                "steps": 10, "tokens_seen": 1024, **base})
        audit = verify_controlled_setup(tmp, variants, seeds, 1,
                                        expected_data_digest="abc123")
        assert audit["ok"] and audit["eval_protocol_checked"] == 2, audit
        for key, bad in (("eval_ctx", 256), ("eval_iters", 25),
                         ("val_windows", 40), ("split", "val")):
            p = os.path.join(tmp, "lega", "eval.json")
            ev = json.load(open(p, encoding="utf-8"))
            ev[key] = bad
            _write_json(p, ev)
            try:
                verify_controlled_setup(tmp, variants, seeds, 1,
                                        expected_data_digest="abc123")
                raise AssertionError(f"{key} 不一致应触发 SystemExit")
            except SystemExit:
                pass
            ev[key] = base[key]
            _write_json(p, ev)
        # 旧产物完全缺评测协议字段：不伪造一致，但也保持兼容（显式记录缺失）
        for key in ("split", "eval_ctx", "eval_iters", "eval_iters_requested",
                    "val_windows", "val_tokens_evaluated",
                    "selection_set_is_eval_set"):
            ev.pop(key, None)
        _write_json(os.path.join(tmp, "lega", "eval.json"), ev)
        audit = verify_controlled_setup(tmp, variants, seeds, 1,
                                        expected_data_digest="abc123")
        assert audit["ok"], audit
        assert "lega/seed7" in audit["eval_protocol_missing"], audit
    print("  [ok] 受控变量审计：PPL 评测 split/ctx/iters/windows 不一致即拒绝")


def test_verify_controlled_setup_normalizes_numeric_strings():
    """旧 eval.json 把协议数量写成字符串时，语义相同的数值不应被判不一致。"""
    with tempfile.TemporaryDirectory() as tmp:
        variants, seeds = ["standard", "lega"], [7]
        base = {"split": "test", "eval_ctx": 512, "eval_iters": 50,
                "eval_iters_requested": 50, "val_windows": 80,
                "val_tokens_evaluated": 25600,
                "selection_set_is_eval_set": False}
        for v in variants:
            d = os.path.join(tmp, v)
            os.makedirs(d)
            _write_json(os.path.join(d, "config_used.json"), {
                "lr": 0.001, "steps": 10, "variant": v, "seed": 7,
                "device": "cpu", "tag": "t"})
            _write_json(os.path.join(d, "eval.json"), {
                "variant": v, "seed": 7, "data_fingerprint": "abc123",
                "code_commit": "deadbeef", "config_sha256": "shax",
                "steps": 10, "tokens_seen": 1024, **base})
        ev = json.load(open(os.path.join(tmp, "lega", "eval.json"),
                            encoding="utf-8"))
        for key in ("eval_ctx", "eval_iters", "eval_iters_requested",
                    "val_windows", "val_tokens_evaluated"):
            ev[key] = str(ev[key])
        _write_json(os.path.join(tmp, "lega", "eval.json"), ev)
        audit = verify_controlled_setup(tmp, variants, seeds, 1,
                                        expected_data_digest="abc123")
        assert audit["ok"] and audit["eval_protocol_checked"] == 2, audit
    print("  [ok] 受控变量审计：数值字符串与数值按相同语义比较")


def test_verify_controlled_setup_rejects_fractional_steps():
    """实际步数/评测窗口被小数污染时不能被 int() 截断后冒充一致。"""
    with tempfile.TemporaryDirectory() as tmp:
        variants, seeds = ["standard", "lega"], [7]
        base = {"split": "test", "eval_ctx": 512, "eval_iters": 50,
                "eval_iters_requested": 50, "val_windows": 80,
                "val_tokens_evaluated": 25600,
                "selection_set_is_eval_set": False}
        for v in variants:
            d = os.path.join(tmp, v)
            os.makedirs(d)
            _write_json(os.path.join(d, "config_used.json"), {
                "lr": 0.001, "steps": 10, "variant": v, "seed": 7,
                "device": "cpu", "tag": "t"})
            _write_json(os.path.join(d, "eval.json"), {
                "variant": v, "seed": 7, "data_fingerprint": "abc123",
                "code_commit": "deadbeef", "config_sha256": "shax",
                "steps": 10, "tokens_seen": 1024, **base})
        for v, actual in (("standard", 10), ("lega", 10.5)):
            p = os.path.join(tmp, v, "eval.json")
            ev = json.load(open(p, encoding="utf-8"))
            ev["steps_planned"] = 10.5
            ev["steps"] = actual
            _write_json(p, ev)
        try:
            verify_controlled_setup(tmp, variants, seeds, 1,
                                    expected_data_digest="abc123")
            raise AssertionError("小数实际步数不应被截断为一致")
        except SystemExit:
            pass
        # 双方都记录相同的小数步数同样不能算“同一预算”：步数必须可解析为整数
        for v in variants:
            p = os.path.join(tmp, v, "eval.json")
            ev = json.load(open(p, encoding="utf-8"))
            ev["steps_planned"] = 10.5
            ev["steps"] = 10.5
            _write_json(p, ev)
        try:
            verify_controlled_setup(tmp, variants, seeds, 1,
                                    expected_data_digest="abc123")
            raise AssertionError("小数步数不能被当作合法训练预算")
        except SystemExit:
            pass
        # JSON true 不是步数：bool 会被 float() 当成 1，必须显式拒绝
        for v in variants:
            p = os.path.join(tmp, v, "eval.json")
            ev = json.load(open(p, encoding="utf-8"))
            ev["steps_planned"] = True
            ev["steps"] = True
            ev["tokens_seen"] = True
            _write_json(p, ev)
        try:
            verify_controlled_setup(tmp, variants, seeds, 1,
                                    expected_data_digest="abc123")
            raise AssertionError("布尔步数/tokens_seen 不能被当作训练量")
        except SystemExit:
            pass
        # "1_0" 经 int()/float() 都会变成 10；下划线写法不是训练量。
        for v in variants:
            p = os.path.join(tmp, v, "eval.json")
            ev = json.load(open(p, encoding="utf-8"))
            ev["steps_planned"] = "1_0"
            ev["steps"] = "1_0"
            ev["tokens_seen"] = "1_0"
            _write_json(p, ev)
        try:
            verify_controlled_setup(tmp, variants, seeds, 1,
                                    expected_data_digest="abc123")
            raise AssertionError("下划线步数/tokens_seen 不能被当作训练量")
        except SystemExit:
            pass
    print("  [ok] 受控变量审计：小数/布尔/下划线步数不会被截断为整数")


def test_verify_controlled_setup_rejects_fractional_seeds():
    """config/eval 的小数种子不能被 int() 截断后冒充预期种子。"""
    with tempfile.TemporaryDirectory() as tmp:
        variants, seeds = ["standard", "lega"], [7]
        base = {"split": "test", "eval_ctx": 512, "eval_iters": 50,
                "eval_iters_requested": 50, "val_windows": 80,
                "val_tokens_evaluated": 25600,
                "selection_set_is_eval_set": False}
        for v in variants:
            d = os.path.join(tmp, v)
            os.makedirs(d)
            _write_json(os.path.join(d, "config_used.json"), {
                "lr": 0.001, "steps": 10, "variant": v, "seed": 7,
                "device": "cpu", "tag": "t"})
            _write_json(os.path.join(d, "eval.json"), {
                "variant": v, "seed": 7, "data_fingerprint": "abc123",
                "code_commit": "deadbeef", "config_sha256": "shax",
                "steps": 10, "tokens_seen": 1024, **base})
        cfg_path = os.path.join(tmp, "lega", "config_used.json")
        cfg = json.load(open(cfg_path, encoding="utf-8"))
        cfg["seed"] = 7.5
        _write_json(cfg_path, cfg)
        try:
            verify_controlled_setup(tmp, variants, seeds, 1,
                                    expected_data_digest="abc123")
            raise AssertionError("config 小数种子不应通过审计")
        except SystemExit:
            pass
        cfg["seed"] = 7
        _write_json(cfg_path, cfg)
        ev_path = os.path.join(tmp, "lega", "eval.json")
        ev = json.load(open(ev_path, encoding="utf-8"))
        ev["seed"] = 7.5
        _write_json(ev_path, ev)
        try:
            verify_controlled_setup(tmp, variants, seeds, 1,
                                    expected_data_digest="abc123")
            raise AssertionError("eval 小数种子不应通过审计")
        except SystemExit:
            pass
    print("  [ok] 受控变量审计：小数种子不会被截断为预期种子")


def test_component_evidence_marks_early_stop_budget():
    """早停预算不等时，JSON 证据必须显式降为探索性而非只说 supported。"""
    tests = [
        _mk("lega_no_query", "lega", "PPL", 0.4, 0.005),
        _mk("qgate", "gate_only", "PPL", -0.01, 0.7),
    ]
    _apply_holm(tests)
    audit = {"early_stop_runs": ["lega/seed7(step 6/10)"]}
    a = next(c for c in component_evidence(tests, audit=audit)
             if c["component"] == "A 查询项")
    assert a["supported"] == ["PPL"]                 # 统计显著性仍保留
    assert a["training_budget_comparable"] is False
    assert a["confirmatory_supported"] is False
    assert a["evidence_status"] == "exploratory"
    assert "早停" in a["budget_note"] and a["early_stop_runs"]
    ok = next(c for c in component_evidence(tests)
              if c["component"] == "A 查询项")
    assert ok["training_budget_comparable"] is True
    assert ok["confirmatory_supported"] is True
    assert ok["evidence_status"] == "confirmatory"
    print("  [ok] component_evidence：早停预算不等时机器可读降级为探索性")


def test_seed_parsing_rejects_bool_and_fractional_truncation():
    """--seeds/配置 seeds 不能被 int() 截断：True → 1、1.5 → 1 会改身份。

    种子身份错位后结果目录名、台账与播种的 RNG 相互矛盾；更糟的是
    ``dict.fromkeys`` 会把 1.5 与 1 去重，导致"N 个种子"的证据凭空缩水。
    bool 在 dict 里与 1 同键，同样不能被当作种子 1。
    """
    assert _strict_seed_int(7) == 7
    assert _strict_seed_int(7.0) == 7
    assert _strict_seed_int("1337") == 1337
    assert _strict_seed_int(" 42 ") == 42
    for bad in (True, False, 1.5, -0.5, -1, 2 ** 32, float("nan"),
                float("inf"), "1.5", "abc", "", [7], {"seed": 1}, None):
        try:
            _strict_seed_int(bad)
            raised = False
        except ValueError:
            raised = True
        assert raised, bad
    assert _parse_seed_list("1,2, 3") == [1, 2, 3]
    assert _parse_seed_list("") == []
    # 字符串 "2.0" 不是整数身份（与 tuning.parse_seeds 同口径拒绝）
    for bad in ("1,2.0", "1,1.5", "1,true", "1,foo"):
        try:
            _parse_seed_list(bad)
            raised = False
        except ValueError:
            raised = True
        assert raised, bad
    print("  [ok] run_all 种子解析拒绝 bool/小数截断（含 --tune_seeds 同口径）")


def test_controlled_setup_selection_frac_rejects_bad_planned_steps():
    """checkpoint 选型比例的分母必须是可解释为整数的计划步数。

    steps_planned=true 时 True/1==1 会让比例虚高；10.5 会让比例用错误预算
    做除法。两者都应记为 None（缺比例），而字符串 "10"/10.0 仍可用。
    """
    with tempfile.TemporaryDirectory() as tmp:
        variants, seeds = ["standard"], [7]
        base = {"split": "test", "eval_ctx": 512, "eval_iters": 50,
                "eval_iters_requested": 50, "val_windows": 80,
                "val_tokens_evaluated": 25600,
                "selection_set_is_eval_set": False}
        d = os.path.join(tmp, "standard")
        os.makedirs(d)
        _write_json(os.path.join(d, "config_used.json"), {
            "lr": 0.001, "steps": 10, "variant": "standard", "seed": 7,
            "device": "cpu", "tag": "t"})
        _write_json(os.path.join(d, "eval.json"), {
            "variant": "standard", "seed": 7, "data_fingerprint": "abc123",
            "code_commit": "deadbeef", "config_sha256": "shax",
            "steps": 10, "tokens_seen": 1024, "ckpt_step": 5, **base})
        for bad, expect_frac in ((True, None), (10.5, None), ("10", 0.5),
                                 (10.0, 0.5)):
            ev = json.load(open(os.path.join(d, "eval.json"), encoding="utf-8"))
            ev["steps_planned"] = bad
            ev["steps"] = 10            # 实际步数保持正常，聚焦选型比例解析
            ev["tokens_seen"] = 1024
            # 计划步数非法（bool/小数）时全局一致性审计会先失败；此时只断言
            # “不会算出比例”，比例字段本身在成功路径上再验证。
            if expect_frac is None:
                ev_bad = dict(ev, steps=bad)   # 实际步数也与计划同值（非法）
                _write_json(os.path.join(d, "eval.json"), ev_bad)
                try:
                    rep = verify_controlled_setup(tmp, variants, seeds, 1,
                                                  expected_data_digest="abc123")
                except SystemExit:
                    continue
                # 未触发全局失败时，选型比例仍必须是 None（分母不可整数化）
                assert rep["selection_step_points"][0][3] is None, bad
                continue
            _write_json(os.path.join(d, "eval.json"), ev)
            rep = verify_controlled_setup(tmp, variants, seeds, 1,
                                          expected_data_digest="abc123")
            point = rep["selection_step_points"][0]
            assert point[3] == expect_frac, (bad, point)
        # 合法大整数：ckpt_step/计划步数可精确解析，但比例转 float 会溢出；
        # 审计必须继续运行并把比例记为缺失，而不是崩溃。
        ev = json.load(open(os.path.join(d, "eval.json"), encoding="utf-8"))
        ev.update({"steps_planned": 10, "steps": 10, "tokens_seen": 1024,
                   "ckpt_step": 10 ** 400})
        _write_json(os.path.join(d, "eval.json"), ev)
        rep = verify_controlled_setup(tmp, variants, seeds, 1,
                                      expected_data_digest="abc123")
        assert rep["ok"] and rep["selection_step_points"][0][3] is None, rep
        # steps 与计划步数的数值/字符串表示必须按精确整数比较，不能先转
        # float（10**400 会溢出）后误判为不一致。
        huge = 10 ** 400
        ev.update({"steps_planned": huge, "steps": str(huge),
                   "tokens_seen": huge, "ckpt_step": huge})
        _write_json(os.path.join(d, "eval.json"), ev)
        rep = verify_controlled_setup(tmp, variants, seeds, 1,
                                      expected_data_digest="abc123")
        assert rep["ok"], rep
    print("  [ok] 受控变量审计：选型比例分母拒绝 bool/小数计划步数；"
          "超大整数精确比较且比例溢出安全降级")


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
    print("  [ok] 窗口分层 bootstrap：巨大有限差/正负抵消不溢出，非有限差过滤")


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
    print("  [ok] 全因子协议核对：不可哈希 split 不再让汇总崩溃")


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
        with mock.patch("run_all.load_registered_hypotheses",
                        return_value=reg), \
                mock.patch("run_all.verify_registration",
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
        with mock.patch("run_all._factorial_interaction_tests",
                        return_value=bad_inter), \
                mock.patch("run_all._factorial_shapley_tests",
                           return_value=[]), \
                mock.patch("run_all._ablation_lesion_records",
                           return_value=[]), \
                mock.patch("run_all._inference_shapley_tests",
                           return_value=[]):
            out = build_summary(tmp)
        md = out["markdown"]
        assert "A×B" in md and "无法检验" in md
    print("  [ok] 汇总渲染：缺失/非有限均值显示 '-'，交互表不再 TypeError")
