"""Hash-chained ledgers: deletion/reordering/content edits must be detectable."""

import json
import os
import shutil
import tempfile

from lega.ledger import GENESIS_SHA256, chain_record, record_sha256, verify_chain
# Reproducibility: Set random seeds
import random
import numpy as np
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
try:
    import torch
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
except ImportError:
    pass


def _write(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")


def _read(path):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def test_chain_links_records_to_genesis_and_previous():
    a = chain_record({"i": 1}, None)
    b = chain_record({"i": 2}, a)
    assert a["prev_record_sha256"] == GENESIS_SHA256
    assert b["prev_record_sha256"] == record_sha256(a)
    assert a["record_sha256"] == record_sha256(a)
    rep = verify_chain([a, b])
    assert rep["ok"] and rep["fully_chained"] and rep["legacy_records"] == 0
    assert rep["head_sha256"] == record_sha256(b)
    assert rep["anchored"] is False
    print("  [ok] 链：首条接 genesis，后续接前一条摘要")


def test_content_edit_breaks_chain():
    recs = []
    prev = None
    for i in range(3):
        prev = chain_record({"i": i, "v": "x"}, prev)
        recs.append(prev)
    recs[1] = dict(recs[1], v="tampered")
    rep = verify_chain(recs)
    assert rep["ok"] is False
    issues = {x["issue"] for x in rep["issues"]}
    assert "self_mismatch" in issues and "prev_mismatch" in issues, issues
    print("  [ok] 链：改写中间记录内容可发现（self/prev 双重失配）")


def test_middle_record_deletion_breaks_chain():
    recs, prev = [], None
    for i in range(3):
        prev = chain_record({"i": i}, prev)
        recs.append(prev)
    rep = verify_chain([recs[0], recs[2]])
    assert rep["ok"] is False
    assert any(x["issue"] == "prev_mismatch" for x in rep["issues"])
    print("  [ok] 链：删除中间记录留下断裂链接")


def test_reordering_breaks_chain():
    recs, prev = [], None
    for i in range(3):
        prev = chain_record({"i": i}, prev)
        recs.append(prev)
    rep = verify_chain([recs[0], recs[2], recs[1]])
    assert rep["ok"] is False
    print("  [ok] 链：重排记录可发现")


def test_legacy_records_are_reported_not_silently_verified():
    legacy = {"i": 0}
    first = chain_record({"i": 1}, legacy)
    rep = verify_chain([legacy, first])
    assert rep["ok"] is True
    assert rep["legacy_records"] == 1 and rep["fully_chained"] is False
    # 链开始后不允许再出现无链字段的记录（可能是被替换的旧格式记录）
    rep2 = verify_chain([first, {"i": 2}])
    assert rep2["ok"] is False
    assert any(x["issue"] == "missing_chain_fields" for x in rep2["issues"])
    print("  [ok] 链：旧记录显式计为 legacy，链开始后缺失链字段报错")


def test_test_access_ledger_is_chained_and_tampering_is_visible():
    from lega.datacheck import (
        TEST_ACCESS_LOG,
        append_test_access,
        read_test_access_log,
        summarize_test_access,
    )

    d = tempfile.mkdtemp()
    try:
        append_test_access(d, {"ckpt_sha256": "a", "test_data_fingerprint": "v1"})
        append_test_access(d, {"ckpt_sha256": "b", "test_data_fingerprint": "v1"})
        records = read_test_access_log(d)
        assert all("record_sha256" in r for r in records)
        rep = summarize_test_access(records)
        assert rep["ledger_integrity"]["ok"] is True
        assert rep["ledger_integrity"]["fully_chained"] is True
        records[0]["test_data_fingerprint"] = "forged"
        _write(os.path.join(d, TEST_ACCESS_LOG), records)
        rep = summarize_test_access(read_test_access_log(d))
        assert rep["ledger_integrity"]["ok"] is False
        assert rep["ledger_integrity"]["issues"]
    finally:
        shutil.rmtree(d, ignore_errors=True)
    print("  [ok] 测试集访问账本：链校验通过，手改记录可见")


def test_experiment_ledger_is_chained_and_verify_reports_breaks():
    from lega.experiment import append_record, read_records, verify_ledger

    tmp = tempfile.mkdtemp()
    try:
        path = os.path.join(tmp, "experiments.jsonl")
        append_record(path, {"run_id": "r1", "status": "started"})
        append_record(path, {"run_id": "r1", "status": "completed"})
        rep = verify_ledger(path)
        assert rep["chain"]["ok"] is True
        assert rep["chain"]["fully_chained"] is True
        records = read_records(path)
        records[0]["run_id"] = "forged"
        _write(path, records)
        rep = verify_ledger(path)
        assert rep["ok"] is False
        assert rep["chain"]["ok"] is False
        assert any(i.get("issue") == "ledger_chain_broken" for i in rep["issues"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] 实验台账：verify_ledger 报告链断裂")


def test_broken_test_access_chain_downgrades_confirmatory_claims():
    from lega.design.preparation import test_seal_precheck
    from lega.testreuse import apply_test_reuse_gate, test_reuse_report

    broken = {
        "looks_by_test_data_fingerprint": {"v1": ["run-1"]},
        "unattributed_records_by_test_data_fingerprint": {},
        "ledger_integrity": {
            "ok": False,
            "issues": [{"index": 0, "issue": "prev_mismatch"}],
        },
    }
    rep = test_reuse_report(broken, "v1", planned_looks=1)
    assert rep["unverifiable"] is True
    assert rep["ledger_integrity_ok"] is False
    assert "SHA-256 链" in rep["reason"]
    hyp = [{"metric": "PPL", "confirmatory_valid": True}]
    out = apply_test_reuse_gate(hyp, rep, ["PPL"])
    assert out[0]["confirmatory_valid"] is False
    assert out[0]["test_ledger_integrity_ok"] is False
    seal = test_seal_precheck(
        {"summary": broken, "test_data_fingerprint": "v1"}, planned_looks=1
    )
    assert seal["can_start_confirmatory"] is False
    # 旧账本（有 legacy 记录但链本身一致）不触发降级。
    legacy = dict(
        broken,
        ledger_integrity={
            "ok": True,
            "legacy_records": 2,
            "fully_chained": False,
            "issues": [],
        },
    )
    rep2 = test_reuse_report(legacy, "v1", planned_looks=1)
    assert rep2["unverifiable"] is False
    print("  [ok] 访问账本链断裂：确认性判定与开跑闸门同步降级")


def test_new_records_chain_from_legacy_tail_without_rewriting_history():
    from lega.datacheck import (
        TEST_ACCESS_LOG,
        append_test_access,
        read_test_access_log,
        summarize_test_access,
    )

    d = tempfile.mkdtemp()
    try:
        legacy = {
            "ckpt_sha256": "old",
            "test_data_fingerprint": "v1",
            "time": "2026-01-01 00:00:00",
        }
        _write(os.path.join(d, TEST_ACCESS_LOG), [legacy])
        append_test_access(d, {"ckpt_sha256": "new", "test_data_fingerprint": "v1"})
        records = read_test_access_log(d)
        assert records[0] == legacy  # 旧记录原样保留
        assert records[1]["prev_record_sha256"] == record_sha256(legacy)
        rep = summarize_test_access(records)
        assert rep["ledger_integrity"]["ok"] is True
        assert rep["ledger_integrity"]["legacy_records"] == 1
        assert rep["ledger_integrity"]["fully_chained"] is False
    finally:
        shutil.rmtree(d, ignore_errors=True)
    print("  [ok] 新记录可从旧账本尾部接链，历史不被改写")
