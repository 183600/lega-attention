"""实验台账测试：追加/读取、started→completed 生命周期、产物哈希、run 快照。"""
import json
import os
import shutil
import tempfile

from lega.experiment import (append_record, build_experiment_record,
                             collect_artifacts, ledger_path, new_run_id,
                             read_records, snapshot_train_run)


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
                       "data_fingerprint": "fp", "config_sha256": "sha"}, f)
        with open(os.path.join(out, "train_log.jsonl"), "w", encoding="utf-8") as f:
            f.write('{"step": 1, "loss": 1.0}\n')
        snap = snapshot_train_run("lega", 7, out)
        assert snap["variant"] == "lega" and snap["seed"] == 7
        assert snap["result"]["val_ppl"] == 12.5
        assert set(snap["files"]) == {"eval.json", "train_log.jsonl"}
        assert len(snap["files"]["train_log.jsonl"]["sha256"]) == 64
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] snapshot_train_run 最终指标 + 曲线/配置哈希")
