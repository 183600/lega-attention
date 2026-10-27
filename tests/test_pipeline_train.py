"""训练前 sanity、aborted 台账与失败恢复测试。"""
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
