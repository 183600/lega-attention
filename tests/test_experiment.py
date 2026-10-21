"""实验台账测试：追加/读取、started→completed 生命周期、产物哈希、run 快照。"""
import json
import os
import shutil
import tempfile

from lega.experiment import (append_record, build_experiment_record,
                             collect_artifacts, ledger_path, new_run_id,
                             read_records, snapshot_train_run, verify_ledger)
from lega.repro import sha256_file


def test_run_id_unique_readable_and_no_scientific_rng():
    a = new_run_id("tiny", "deadbeef" * 8)
    b = new_run_id("tiny", "deadbeef" * 8)
    assert a != b                      # 同一秒内连续调用也必须唯一
    assert "tiny" in a and "deadbeef"[:8] in a
    # 固定时钟下批量生成也必须全部唯一。
    import lega.experiment as exp
    real_time_ns = exp.time.time_ns
    try:
        exp.time.time_ns = lambda: 1_700_000_000_123_456_789
        ids = [exp.new_run_id("tiny", "deadbeef" * 8) for _ in range(50)]
    finally:
        exp.time.time_ns = real_time_ns
    assert len(set(ids)) == len(ids)
    print(f"  [ok] run_id 唯一且可读: {a}")


def test_ledger_append_read_lifecycle_and_tolerates_bad_lines():
    tmp = tempfile.mkdtemp()
    try:
        path = os.path.join(tmp, "experiments.jsonl")
        assert read_records(path) == []          # 不存在 → 空列表
        common = dict(tag="t", config_path="configs/tiny.json",
                      config_sha256="a" * 64, variants=["lega", "standard"],
                      seeds=[1, 2], smoke=False, argv=["run_all.py"],
                      git={"commit": "c" * 40, "dirty": False},
                      data_fingerprint="fp123", runtime={"torch": "x"},
                      out_root="out/t", results_dir="results/t",
                      time_start="2026-01-01 00:00:00")
        rid = new_run_id("t", common["config_sha256"])
        append_record(path, build_experiment_record(
            run_id=rid, status="started", **common))
        append_record(path, build_experiment_record(
            run_id=rid, status="completed", time_end="2026-01-01 01:00:00",
            sanity={"executed": True, "verdict": {"lega": "pass"}},
            runs=[{"variant": "lega", "seed": 1}],
            artifacts={"summary.md": {"sha256": "h", "size": 3}},
            **common))
        with open(path, "a", encoding="utf-8") as f:
            f.write("这不是 JSON\n")              # 畸形行不得让读取失败
        recs = read_records(path)
        assert len(recs) == 2
        assert recs[0]["status"] == "started" and recs[1]["status"] == "completed"
        assert recs[1]["seeds"] == [1, 2]
        assert recs[1]["sanity"]["verdict"] == {"lega": "pass"}
        assert recs[1]["artifacts"]["summary.md"]["sha256"] == "h"
        json.dumps(recs)                          # 记录必须整体可序列化
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] 台账 started/completed 追加读取 + 畸形行容错")


def test_ledger_path_default_and_override():
    assert ledger_path().endswith(os.path.join("results", "experiments.jsonl"))
    assert ledger_path("/x/y") == os.path.join("/x/y", "experiments.jsonl")
    print("  [ok] ledger_path 默认/自定义根目录")


def test_collect_artifacts_hashes_and_skips_missing():
    tmp = tempfile.mkdtemp()
    try:
        p1 = os.path.join(tmp, "a.json")
        with open(p1, "w", encoding="utf-8") as f:
            f.write("{}")
        arts = collect_artifacts([p1, os.path.join(tmp, "missing.bin")],
                                 rel_to=tmp)
        assert list(arts) == ["a.json"]
        assert arts["a.json"]["size"] == 2 and len(arts["a.json"]["sha256"]) == 64
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] collect_artifacts 哈希/大小/缺失跳过/相对路径")


def test_snapshot_train_run_reads_eval_and_hashes_logs():
    tmp = tempfile.mkdtemp()
    try:
        out = os.path.join(tmp, "lega")
        os.makedirs(out)
        with open(os.path.join(out, "eval.json"), "w", encoding="utf-8") as f:
            json.dump({"variant": "lega", "val_ppl": 12.5, "steps": 100,
                       "tokens_seen": 4096, "code_commit": "c" * 40,
                       "data_fingerprint": "fp", "config_sha256": "sha",
                       "sanity_passed": True}, f)
        with open(os.path.join(out, "train_log.jsonl"), "w", encoding="utf-8") as f:
            f.write('{"step": 1, "loss": 1.0}\n')
        with open(os.path.join(out, "model_best.pt"), "wb") as f:
            f.write(b"fake-ckpt")               # 被评测产物也必须被哈希绑定
        snap = snapshot_train_run("lega", 7, out)
        assert snap["variant"] == "lega" and snap["seed"] == 7
        assert snap["result"]["val_ppl"] == 12.5
        assert snap["result"]["sanity_passed"] is True
        assert set(snap["files"]) == {"eval.json", "train_log.jsonl",
                                      "model_best.pt"}
        assert len(snap["files"]["train_log.jsonl"]["sha256"]) == 64
        assert len(snap["files"]["model_best.pt"]["sha256"]) == 64
        real = snap["files"]["model_best.pt"]["sha256"]
        # 独立评测 eval_*.json 的已知哈希：传入即复用，不重算
        snap2 = snapshot_train_run(
            "lega", 7, out, known_ckpt_sha256="f" * 64,
            known_ckpt_path=os.path.join(out, "model_best.pt"))
        assert snap2["files"]["model_best.pt"]["sha256"] == "f" * 64
        # 已知哈希对应的路径不是本 run 的 model_best.pt：回退重算
        snap3 = snapshot_train_run(
            "lega", 7, out, known_ckpt_sha256="f" * 64,
            known_ckpt_path=os.path.join(tmp, "other.pt"))
        assert snap3["files"]["model_best.pt"]["sha256"] == real
        # out_dir/eval.json 自带 ckpt/ckpt_sha256 时同样复用
        with open(os.path.join(out, "eval.json"), "w", encoding="utf-8") as f:
            json.dump({"ckpt": os.path.join(out, "model_best.pt"),
                       "ckpt_sha256": "e" * 64}, f)
        snap4 = snapshot_train_run("lega", 7, out)
        assert snap4["files"]["model_best.pt"]["sha256"] == "e" * 64
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] snapshot_train_run 最终指标 + 曲线/配置/checkpoint 哈希")


def test_snapshot_train_run_embeds_curve_for_history_after_overwrite():
    """台账必须内嵌训练曲线：同 tag 重跑覆盖 train_log.jsonl 后历史仍可还原。"""
    tmp = tempfile.mkdtemp()
    try:
        out = os.path.join(tmp, "lega")
        os.makedirs(out)
        rows = [
            {"step": 1, "loss": 4.0, "ppl": 54.6, "time": "t1",
             "step_time_ms": 12.5},
            {"step": 10, "loss": 3.0, "ppl": 20.1, "grad_norm": 1.2},
            {"step": 10, "val_loss": 2.8, "val_ppl": 16.4},
            {"step": 20, "loss": 2.5, "final": True},
        ]
        with open(os.path.join(out, "train_log.jsonl"), "w",
                  encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
            f.write("not json\n")
        snap = snapshot_train_run("lega", 7, out)
        curve = snap["train_curve"]
        assert curve["n_records"] == 4 and not curve["downsampled"]
        assert [r["step"] for r in curve["records"]] == [1, 10, 10, 20]
        assert curve["records"][-1]["final"] is True
        # 覆盖日志后，台账快照仍保留旧曲线内容
        with open(os.path.join(out, "train_log.jsonl"), "w",
                  encoding="utf-8") as f:
            f.write(json.dumps({"step": 999, "loss": 0.0}) + "\n")
        assert snapshot_train_run("lega", 7, out)["train_curve"] != curve
        json.dumps(snap)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] 台账内嵌训练曲线（覆盖日志后历史仍可还原）")


def test_snapshot_train_run_degrades_on_non_dict_eval():
    """eval.json 顶层为数组/标量时不能崩溃，其余产物哈希仍应正常记录。"""
    tmp = tempfile.mkdtemp()
    try:
        out = os.path.join(tmp, "lega")
        os.makedirs(out)
        with open(os.path.join(out, "model_best.pt"), "wb") as f:
            f.write(b"fake-ckpt")
        for bad in ([1, 2], "text", 3, None):
            with open(os.path.join(out, "eval.json"), "w", encoding="utf-8") as f:
                json.dump(bad, f)
            snap = snapshot_train_run("lega", 7, out)
            assert "result" not in snap
            assert "model_best.pt" in snap["files"]
        # 非字符串 ckpt 字段同样按不可用候选跳过，而不是 TypeError
        with open(os.path.join(out, "eval.json"), "w", encoding="utf-8") as f:
            json.dump({"ckpt": 123, "ckpt_sha256": "e" * 64}, f)
        snap = snapshot_train_run("lega", 7, out)
        assert len(snap["files"]["model_best.pt"]["sha256"]) == 64
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] snapshot_train_run：非对象 eval.json/非法 ckpt 字段安全降级")


def test_experiment_record_embeds_resolved_hyperparameters():
    """台账必须自含超参数：config 文件被改写/删除后仍可还原实际配置。"""
    rec = build_experiment_record(
        run_id="r1", status="started", tag="t", config_path="configs/tiny.json",
        config_sha256="a" * 64, variants=["lega"], seeds=[1], smoke=False,
        argv=["run_all.py"], git={"commit": "c" * 40, "dirty": False},
        data_fingerprint="fp", runtime={"torch": "x"}, out_root="out/t",
        results_dir="results/t", time_start="2026-01-01 00:00:00",
        config={"lr": 3e-4, "steps": 100, "seeds": [1]})
    assert rec["config"] == {"lr": 3e-4, "steps": 100, "seeds": [1]}
    # 未提供 config 的旧调用方保持兼容（字段为 None）
    rec2 = build_experiment_record(
        run_id="r2", status="started", tag="t", config_path="c.json",
        config_sha256="b" * 64, variants=["lega"], seeds=[1], smoke=False,
        argv=["run_all.py"], git={}, data_fingerprint=None, runtime={},
        out_root="out/t", results_dir="results/t", time_start="t0")
    assert rec2["config"] is None
    json.dumps(rec)
    print("  [ok] 台账内嵌解析后的超参数（config 自含）")


def test_experiment_record_rejects_fractional_and_bool_seeds():
    """seeds 是实验身份：bool/小数不得被 int() 静默折叠成 1。"""
    common = dict(
        run_id="r", status="started", tag="t", config_path="c.json",
        config_sha256="a" * 64, variants=["lega"], smoke=False,
        argv=["run_all.py"], git={}, data_fingerprint=None, runtime={},
        out_root="out/t", results_dir="results/t", time_start="t0")
    for bad in (True, False, 1.5, float("nan"), "1_0", None):
        try:
            build_experiment_record(seeds=[bad], **common)
            raise AssertionError(f"非法 seed 应报错: {bad!r}")
        except ValueError:
            pass
    rec = build_experiment_record(seeds=[1, 2.0, "3"], **common)
    assert rec["seeds"] == [1, 2, 3], rec["seeds"]
    print("  [ok] 台账 seeds 拒绝 bool/小数，整数值浮点/字符串规范化")


def test_snapshot_train_run_embeds_config_used():
    """run 快照内嵌 config_used.json 内容：单 run 超参数可从台账直接还原。"""
    tmp = tempfile.mkdtemp()
    try:
        out = os.path.join(tmp, "lega")
        os.makedirs(out)
        with open(os.path.join(out, "config_used.json"), "w",
                  encoding="utf-8") as f:
            json.dump({"lr": 1e-3, "seed": 9, "variant": "lega",
                       "train_ctx": 64}, f)
        snap = snapshot_train_run("lega", 9, out)
        assert snap["config_used"]["lr"] == 1e-3
        assert snap["config_used"]["seed"] == 9
        # config_used.json 缺失/畸形时静默降级，不影响其余哈希
        os.remove(os.path.join(out, "config_used.json"))
        assert "config_used" not in snapshot_train_run("lega", 9, out)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] run 快照内嵌 config_used.json（含缺失降级）")


def test_snapshot_embeds_provenance_and_flags_eval_identity_mismatch():
    """台账快照必须自含 seed_report/环境/sanity；旧 eval 不得冒充本 run。"""
    tmp = tempfile.mkdtemp()
    try:
        out = os.path.join(tmp, "lega")
        os.makedirs(out)
        with open(os.path.join(out, "eval.json"), "w", encoding="utf-8") as f:
            json.dump({"variant": "standard", "seed": 999, "val_ppl": 1.0}, f)
        with open(os.path.join(out, "run_manifest.json"), "w",
                  encoding="utf-8") as f:
            json.dump({"config_sha256": "c" * 64,
                       "git": {"commit": "a" * 40, "dirty": False},
                       "worktree": {"available": True, "diff_sha256": None},
                       "data": {"digest": "d" * 16},
                       "seed_report": {"seed": 7,
                                       "rng_fingerprints": {"torch_cpu": "x"}},
                       "runtime": {"torch": "test"},
                       "sanity": {"passed": True, "sha256": "s" * 64}}, f)
        snap = snapshot_train_run("lega", 7, out)
        prov = snap["provenance"]
        assert prov["seed_report"]["seed"] == 7
        assert prov["runtime"]["torch"] == "test"
        assert prov["sanity"]["passed"] is True
        assert prov["data_fingerprint"] == "d" * 16
        assert prov["worktree"]["diff_sha256"] is None
        assert snap["eval_identity_mismatches"]
        assert any("variant" in x for x in snap["eval_identity_mismatches"])
        assert any("seed" in x for x in snap["eval_identity_mismatches"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] 台账快照内嵌逐 run 溯源 + 标记 eval 身份不匹配")


def test_snapshot_flags_fractional_eval_seed():
    """eval.json 的小数种子不能被 int() 截断后冒充本 run 的种子。"""
    tmp = tempfile.mkdtemp()
    try:
        out = os.path.join(tmp, "lega")
        os.makedirs(out)
        with open(os.path.join(out, "eval.json"), "w", encoding="utf-8") as f:
            json.dump({"variant": "lega", "seed": 7.5, "val_ppl": 1.0}, f)
        snap = snapshot_train_run("lega", 7, out)
        assert snap["eval_identity_mismatches"]
        assert any("seed" in x for x in snap["eval_identity_mismatches"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] 台账快照拒绝小数 eval 种子")


def test_verify_ledger_completed_coverage_and_config_identity():
    """completed 必须覆盖 variants×seeds；同 run_id 的配置哈希不得改写。"""
    tmp = tempfile.mkdtemp()
    try:
        path = os.path.join(tmp, "experiments.jsonl")
        common = {"run_id": "bad", "config_sha256": "a" * 64,
                  "variants": ["lega", "standard"], "seeds": [1, 2]}
        append_record(path, dict(common, status="started"))
        append_record(path, dict(
            common, status="completed",
            runs=[{"variant": "lega", "seed": 1},
                  {"variant": "lega", "seed": 1},   # 重复计权
                  {"variant": "other", "seed": 9},  # 不属于本轮
                  "not-a-run"]))                    # 畸形条目
        # 同一 run_id 的 completed 记录改了配置哈希
        append_record(path, {"run_id": "rewrite", "status": "started",
                             "config_sha256": "a" * 64,
                             "variants": ["lega"], "seeds": [1]})
        append_record(path, {"run_id": "rewrite", "status": "completed",
                             "config_sha256": "b" * 64,
                             "variants": ["lega"], "seeds": [1],
                             "runs": [{"variant": "lega", "seed": 1}]})
        # 覆盖完整且无重复的应通过
        append_record(path, {"run_id": "ok", "status": "started",
                             "config_sha256": "c" * 64,
                             "variants": ["lega"], "seeds": [1, 2]})
        append_record(path, {"run_id": "ok", "status": "completed",
                             "config_sha256": "c" * 64,
                             "variants": ["lega"], "seeds": [1, 2],
                             "runs": [{"variant": "lega", "seed": 1},
                                      {"variant": "lega", "seed": 2}]})
        rep = verify_ledger(path)
        by_id = {i.get("run_id"): i for i in rep["issues"] if i.get("run_id")}
        for issue in ("runs_coverage_missing", "runs_coverage_unexpected",
                      "runs_duplicate", "runs_malformed"):
            assert issue in by_id["bad"]["issues"], issue
        assert "config_sha256_rewritten" in by_id["rewrite"]["issues"]
        assert "ok" not in by_id
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] verify_ledger：completed 覆盖完整性 + 配置哈希改写")


def test_verify_ledger_rejects_scope_rewrite_and_duplicate_declaration():
    """完成记录不能改写 started 的 variants×seeds，重复声明也不能静默去重。"""
    tmp = tempfile.mkdtemp()
    try:
        path = os.path.join(tmp, "experiments.jsonl")
        append_record(path, {"run_id": "shrink", "status": "started",
                             "variants": ["lega"], "seeds": [1, 2]})
        append_record(path, {"run_id": "shrink", "status": "completed",
                             "variants": ["lega"], "seeds": [1],
                             "runs": [{"variant": "lega", "seed": 1}]})
        append_record(path, {"run_id": "dup-seed", "status": "started",
                             "variants": ["lega"], "seeds": [1, 1]})
        append_record(path, {"run_id": "dup-seed", "status": "completed",
                             "variants": ["lega"], "seeds": [1, 1],
                             "runs": [{"variant": "lega", "seed": 1}]})
        append_record(path, {"run_id": "dup-var", "status": "started",
                             "variants": ["lega", "lega"], "seeds": [1]})
        append_record(path, {"run_id": "dup-var", "status": "completed",
                             "variants": ["lega", "lega"], "seeds": [1],
                             "runs": [{"variant": "lega", "seed": 1}]})
        rep = verify_ledger(path)
        by_id = {i.get("run_id"): i for i in rep["issues"] if i.get("run_id")}
        assert "scope_rewritten" in by_id["shrink"]["issues"]
        assert "declared_seeds_duplicate" in by_id["dup-seed"]["issues"]
        assert "declared_variants_duplicate" in by_id["dup-var"]["issues"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] verify_ledger：scope 改写/重复声明可发现")


def test_verify_ledger_rejects_malformed_run_coverage_fields():
    """声明了 variants/seeds 却缺 runs（或 runs 非列表）时必须报畸形，不能静默跳过。"""
    tmp = tempfile.mkdtemp()
    try:
        path = os.path.join(tmp, "experiments.jsonl")
        for i, runs in enumerate(("absent", "not-a-list", None)):
            common = {"run_id": f"r{i}", "status": "started",
                      "variants": ["lega"], "seeds": [7]}
            append_record(path, common)
            done = {"run_id": f"r{i}", "status": "completed",
                    "variants": ["lega"], "seeds": [7]}
            if runs != "absent":
                done["runs"] = runs
            append_record(path, done)
        # 旧台账既无 variants/seeds 也无 runs：保留生命周期核验兼容性
        append_record(path, {"run_id": "legacy", "status": "started"})
        append_record(path, {"run_id": "legacy", "status": "completed"})
        rep = verify_ledger(path)
        by_id = {i.get("run_id"): i for i in rep["issues"] if i.get("run_id")}
        for i in range(3):
            assert "runs_malformed" in by_id[f"r{i}"]["issues"], by_id
        assert "legacy" not in by_id
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] verify_ledger：runs 缺失/非列表显式报畸形")


def test_verify_ledger_lifecycle_and_artifact_hashes():
    tmp = tempfile.mkdtemp()
    try:
        path = os.path.join(tmp, "experiments.jsonl")
        art = os.path.join(tmp, "summary.md")
        with open(art, "w", encoding="utf-8") as f:
            f.write("ok")
        good = {"run_id": "r1", "status": "started"}
        done = {"run_id": "r1", "status": "completed",
                "artifacts": {"summary.md": {
                    "sha256": sha256_file(art), "size": 2}}}
        append_record(path, good)
        append_record(path, done)
        append_record(path, {"run_id": "r2", "status": "started"})
        append_record(path, {"run_id": "r3", "status": "completed"})
        append_record(path, {"run_id": "r3", "status": "aborted"})
        rep = verify_ledger(path, check_artifacts=True, root=tmp)
        assert rep["n_records"] == 5 and rep["n_runs"] == 3
        assert rep["ok"] is False
        issues = {i.get("run_id"): i for i in rep["issues"] if i.get("run_id")}
        assert "dangling_started" in issues["r2"]["issues"]
        assert "no_started_record" in issues["r3"]["issues"]
        assert "multiple_terminal_records" in issues["r3"]["issues"]
        # 产物被改写：completed 记录必须报哈希不符
        with open(art, "a", encoding="utf-8") as f:
            f.write("tampered")
        rep2 = verify_ledger(path, check_artifacts=True, root=tmp)
        assert any("sha256_mismatch" in str(i) for i in rep2["issues"])
        # 正常 started→completed 链 + 正确哈希应通过
        path2 = os.path.join(tmp, "ok.jsonl")
        append_record(path2, {"run_id": "ok", "status": "started"})
        append_record(path2, {"run_id": "ok", "status": "completed"})
        assert verify_ledger(path2)["ok"] is True
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] verify_ledger：悬挂/重复终态/无 started/产物哈希可发现")


def test_verify_ledger_tolerates_nonfinite_seed_values():
    """seeds/run.seed 为 JSON Infinity 时必须报畸形而不是让核验崩溃。"""
    tmp = tempfile.mkdtemp()
    try:
        path = os.path.join(tmp, "experiments.jsonl")
        append_record(path, {"run_id": "bad-expected", "status": "started",
                             "variants": ["lega"], "seeds": [1, float("inf")]})
        append_record(path, {"run_id": "bad-expected", "status": "completed",
                             "variants": ["lega"], "seeds": [1, float("inf")],
                             "runs": [{"variant": "lega", "seed": 1}]})
        append_record(path, {"run_id": "bad-run", "status": "started",
                             "variants": ["lega"], "seeds": [1]})
        append_record(path, {"run_id": "bad-run", "status": "completed",
                             "variants": ["lega"], "seeds": [1],
                             "runs": [{"variant": "lega", "seed": float("inf")}]})
        rep = verify_ledger(path)
        by_id = {i.get("run_id"): i for i in rep["issues"] if i.get("run_id")}
        assert "runs_expected_unparseable" in by_id["bad-expected"]["issues"]
        assert "runs_malformed" in by_id["bad-run"]["issues"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] verify_ledger：非有限种子显式报畸形，不崩溃")

def test_verify_ledger_rejects_fractional_seed_values():
    """小数种子不能被 int() 截断后冒充整数种子参与覆盖核对。"""
    tmp = tempfile.mkdtemp()
    try:
        path = os.path.join(tmp, "experiments.jsonl")
        append_record(path, {"run_id": "bad-expected", "status": "started",
                             "variants": ["lega"], "seeds": [1, 1.5]})
        append_record(path, {"run_id": "bad-expected", "status": "completed",
                             "variants": ["lega"], "seeds": [1, 1.5],
                             "runs": [{"variant": "lega", "seed": 1}]})
        append_record(path, {"run_id": "bad-run", "status": "started",
                             "variants": ["lega"], "seeds": [1]})
        append_record(path, {"run_id": "bad-run", "status": "completed",
                             "variants": ["lega"], "seeds": [1],
                             "runs": [{"variant": "lega", "seed": 1.5}]})
        append_record(path, {"run_id": "ok", "status": "started",
                             "variants": ["lega"], "seeds": [1, 2]})
        append_record(path, {"run_id": "ok", "status": "completed",
                             "variants": ["lega"], "seeds": [1, 2],
                             "runs": [{"variant": "lega", "seed": 1},
                                      {"variant": "lega", "seed": 2}]})
        rep = verify_ledger(path)
        by_id = {i.get("run_id"): i for i in rep["issues"] if i.get("run_id")}
        assert "runs_expected_unparseable" in by_id["bad-expected"]["issues"]
        assert "runs_malformed" in by_id["bad-run"]["issues"]
        assert "ok" not in by_id, by_id
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] verify_ledger：小数种子显式报畸形，不截断为整数")
