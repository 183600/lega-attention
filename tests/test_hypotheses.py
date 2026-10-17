"""预注册完整性测试：锁基线、修订链、静默改写/删除/未留痕新增的检测边界。

重点不是“哈希函数能算对”，而是把科研协议变成可执行断言：
  * 登记表被删 → 必须被检出（并有锁文件留证）；
  * 既有假设被静默改写/删除 → violation，confirmatory_valid=False；
  * 新增假设未留痕 → violation（防止看到结果后补登记）；
  * 合法修订（只新增 + --amend 留痕）→ amended 且仍可确认。
"""
import contextlib
import io
import json
import os
import shutil
import tempfile

from lega.hypotheses import (HYPOTHESIS_FIELDS, append_amendment, fingerprint,
                             hypothesis_digest, init_lock, load_registration,
                             main, transition_problems, validate_registration,
                             verify_registration)


def _doc(claims):
    return {"hypotheses": [
        {"id": hid, "claim": c, "metric": "PPL", "candidate": "lega",
         "reference": "standard", "direction": "candidate_better"}
        for hid, c in claims]}


def _write(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _pair(tmp):
    return (os.path.join(tmp, "reg.json"), os.path.join(tmp, "reg.lock.json"))


def test_fingerprint_science_only_and_deterministic():
    """逐假设摘要只看科学字段；辅助注释字段不改变摘要（文档哈希会变）。"""
    h = {"id": "H1", "claim": "c", "metric": "PPL", "candidate": "a",
         "reference": "b", "direction": "candidate_better"}
    h2 = dict(h, note="补充解释，不影响声明", rationale="x")
    assert hypothesis_digest(h) == hypothesis_digest(h2)
    h3 = dict(h, candidate="c")           # 变体对调 → 摘要必变
    assert hypothesis_digest(h) != hypothesis_digest(h3)
    # 键序不同、辅助字段不同的等值文档 → 文档哈希相同
    d1 = {"hypotheses": [h], "extra": 1}
    d2 = {"extra": 1, "hypotheses": [dict(h)]}
    assert fingerprint(d1)["document_sha256"] == fingerprint(d2)["document_sha256"]
    assert fingerprint(d1)["hypotheses"] == {"H1": hypothesis_digest(h)}
    print("  [ok] hypothesis_digest/fingerprint（科学内容摘要、键序无关）")


def test_validate_registration_schema():
    assert validate_registration(_doc([("H1", "c")])) == []
    errors = validate_registration({"hypotheses": []})
    assert errors and "非空" in errors[0]
    bad = {"hypotheses": [
        {"id": "H1", "claim": "c", "metric": "PPL", "candidate": "a",
         "reference": "b", "direction": "candidate_better"},
        {"id": "H1", "claim": "c2", "metric": "PPL", "candidate": "a",
         "reference": "b", "direction": "candidate_better"},
        {"id": "H3", "claim": "c3", "metric": "PPL", "candidate": "a",
         "reference": "b", "direction": "bogus"},
        {"id": "H4", "claim": "c4", "candidate": "a",
         "reference": "b", "direction": "candidate_better"},
    ]}
    errs = validate_registration(bad)
    assert any("重复" in e for e in errs)
    assert any("direction" in e for e in errs)
    assert any("缺少字段 metric" in e for e in errs)  # H4 无 metric
    print("  [ok] validate_registration（重复 id/非法 direction/缺字段）")


def test_verify_intact_then_silent_tamper_detected():
    tmp = tempfile.mkdtemp()
    try:
        reg, lock = _pair(tmp)
        _write(reg, _doc([("H1", "c1"), ("H2", "c2")]))
        init_lock(registry_path=reg, lock_path=lock)
        r = verify_registration(registry_path=reg, lock_path=lock)
        assert r["status"] == "intact" and r["confirmatory_valid"]
        assert r["n_hypotheses"] == 2 and r["chain_length"] == 0

        # 静默改写既有假设的 claim → violated（科学内容摘要变化）
        d = load_registration(reg)
        d["hypotheses"][0]["claim"] = "被偷偷改过的声明"
        _write(reg, d)
        r = verify_registration(registry_path=reg, lock_path=lock)
        assert r["status"] == "violated" and not r["confirmatory_valid"]
        assert any("被改写" in p for p in r["problems"])

        # 静默删除一条 → violated
        d = load_registration(reg)
        d["hypotheses"] = d["hypotheses"][1:]
        _write(reg, d)
        r = verify_registration(registry_path=reg, lock_path=lock)
        assert r["status"] == "violated"
        assert any("被删除" in p for p in r["problems"])

        # 静默新增（未留痕）→ violated（防“看到结果后补预注册”）
        d = load_registration(reg)
        d["hypotheses"].append({"id": "H9", "claim": "新增", "metric": "PPL",
                                "candidate": "x", "reference": "y",
                                "direction": "candidate_better"})
        _write(reg, d)
        r = verify_registration(registry_path=reg, lock_path=lock)
        assert r["status"] == "violated"
        assert any("未在修订链留痕" in p for p in r["problems"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] verify_registration（改写/删除/未留痕新增 → violated）")


def test_amendment_appends_only_and_rejects_forbidden_edits():
    tmp = tempfile.mkdtemp()
    try:
        reg, lock = _pair(tmp)
        _write(reg, _doc([("H1", "c1")]))
        init_lock(registry_path=reg, lock_path=lock)

        # 修订必须给理由
        try:
            append_amendment(registry_path=reg, lock_path=lock)
            raise AssertionError("缺少 --reason 应报错")
        except ValueError as e:
            assert "reason" in str(e)

        # 无变化 → 无可留痕
        try:
            append_amendment(registry_path=reg, lock_path=lock, reason="noop")
            raise AssertionError("无变化应报错")
        except ValueError as e:
            assert "没有可留痕" in str(e)

        # 改写既有假设 → 拒绝（必须由人显式处理，这正是修改变贵的地方）
        d = load_registration(reg)
        d["hypotheses"][0]["claim"] = "改过的"
        _write(reg, d)
        try:
            append_amendment(registry_path=reg, lock_path=lock, reason="改")
            raise AssertionError("改写既有假设应被拒绝")
        except ValueError as e:
            assert "append-only" in str(e)

        # 恢复原状后合法追加 H2 → amended，仍可确认
        d = load_registration(reg)
        d["hypotheses"][0]["claim"] = "c1"
        d["hypotheses"].append({"id": "H2", "claim": "c2", "metric": "PPL",
                                "candidate": "a", "reference": "b",
                                "direction": "candidate_worse"})
        _write(reg, d)
        link = append_amendment(registry_path=reg, lock_path=lock,
                                reason="追加 H2（正式实验前）")
        assert link["before"]["n_hypotheses"] == 1
        assert link["after"]["n_hypotheses"] == 2
        r = verify_registration(registry_path=reg, lock_path=lock)
        assert r["status"] == "amended" and r["confirmatory_valid"]
        # 追加后再静默删掉 H2 → violated（链头含 H2）
        d = load_registration(reg)
        d["hypotheses"] = d["hypotheses"][:1]
        _write(reg, d)
        r = verify_registration(registry_path=reg, lock_path=lock)
        assert r["status"] == "violated"

        # 链本身被破坏（起点与上一环不一致）→ 既判 violated，也拒绝继续追加
        lockobj = json.load(open(lock, encoding="utf-8"))
        lockobj["chain"][0]["before"] = dict(lockobj["chain"][0]["before"])
        lockobj["chain"][0]["before"]["hypotheses"] = {"HX": "deadbeef"}
        _write(lock, lockobj)
        r = verify_registration(registry_path=reg, lock_path=lock)
        assert r["status"] == "violated"
        assert any("链断裂" in p for p in r["problems"])
        _write(reg, _doc([("H1", "c1")]))
        try:
            append_amendment(registry_path=reg, lock_path=lock, reason="x")
            raise AssertionError("链已损坏时不得追加修订")
        except ValueError as e:
            assert "已损坏" in str(e)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] append_amendment（只增不改、无理由/无变化/改写均拒绝）")


def test_verify_missing_files_and_malformed_lock():
    tmp = tempfile.mkdtemp()
    try:
        reg, lock = _pair(tmp)
        # 两者皆缺
        r = verify_registration(registry_path=reg, lock_path=lock)
        assert r["status"] == "missing_registry" and not r["confirmatory_valid"]
        # 有登记表无锁
        _write(reg, _doc([("H1", "c1")]))
        r = verify_registration(registry_path=reg, lock_path=lock)
        assert r["status"] == "missing_lock" and not r["confirmatory_valid"]
        # 锁基线在、登记表被删 → missing_registry 且提示锁记录条数
        init_lock(registry_path=reg, lock_path=lock)
        os.remove(reg)
        r = verify_registration(registry_path=reg, lock_path=lock)
        assert r["status"] == "missing_registry"
        assert any("1" in p and "已登记假设" in p for p in r["problems"])
        # 锁结构损坏 → violated
        _write(reg, _doc([("H1", "c1")]))
        _write(lock, {"baseline": {"hypotheses": "坏掉"}})
        r = verify_registration(registry_path=reg, lock_path=lock)
        assert r["status"] == "violated"
        # 登记表结构非法（重复 id）→ violated 且列出 schema 错误
        _write(reg, _doc([("H1", "a"), ("H1", "b")]))
        r = verify_registration(registry_path=reg, lock_path=lock)
        assert r["status"] == "violated" and r["schema_errors"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] verify_registration（缺文件/坏锁/非法登记表的状态边界）")


def test_transition_problems_lists_all_edit_kinds():
    before = fingerprint(_doc([("H1", "c1"), ("H2", "c2")]))
    after = fingerprint(_doc([("H1", "改写"), ("H3", "新增")]))
    problems = transition_problems(before, after)
    assert any("H1" in p and "被改写" in p for p in problems)
    assert any("H2" in p and "被删除" in p for p in problems)
    assert not any("H3" in p for p in problems)  # 新增由调用方判定是否留痕
    print("  [ok] transition_problems（改写/删除列举、新增不算问题）")


def test_cli_verify_exit_codes():
    tmp = tempfile.mkdtemp()
    try:
        reg, lock = _pair(tmp)
        _write(reg, _doc([("H1", "c1")]))
        init_lock(registry_path=reg, lock_path=lock)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = main(["--verify", "--registry", reg, "--lock", lock])
        assert rc == 0 and "intact" in out.getvalue()
        d = load_registration(reg)
        d["hypotheses"][0]["claim"] = "改了"
        _write(reg, d)
        with contextlib.redirect_stdout(io.StringIO()):
            rc = main(["--verify", "--registry", reg, "--lock", lock])
        assert rc == 1
        # --init 不覆盖已有锁
        with contextlib.redirect_stdout(io.StringIO()):
            rc = main(["--init", "--registry", reg, "--lock", lock])
        assert rc == 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] CLI --verify/--init 退出码（可确认=0，violated/拒绝=1）")


def test_hypothesis_fields_cover_analysis_keys():
    """摘要字段必须覆盖判定所需字段：漏一个就会让改写该字段逃过检测。"""
    assert set(HYPOTHESIS_FIELDS) >= {"id", "claim", "metric", "candidate",
                                      "reference", "direction"}
    print("  [ok] HYPOTHESIS_FIELDS 覆盖判定字段")


def test_verify_registration_rejects_non_list_chain():
    """锁文件的 chain 被替换为非列表（字符串/数字/对象）时必须判 violated：
    否则篡改审计链会被静默当作“无修订”而仍确认有效（int 还会直接崩）。"""
    tmp = tempfile.mkdtemp()
    try:
        reg, lock = _pair(tmp)
        _write(reg, _doc([("H1", "c1")]))
        init_lock(registry_path=reg, lock_path=lock)
        for bad in ("corrupted-string", 42, {"0": {}}):
            lk = json.load(open(lock, encoding="utf-8"))
            lk["chain"] = bad
            _write(lock, lk)
            r = verify_registration(registry_path=reg, lock_path=lock)
            assert r["status"] == "violated", bad
            assert not r["confirmatory_valid"], bad
            assert any("chain" in p for p in r["problems"]), bad
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] verify_registration：非列表 chain 按损坏处理（不再静默通过）")
