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

from lega.hypotheses import (HYPOTHESIS_FIELDS, LOCK_PATH, REGISTRY_PATH,
                             append_amendment, fingerprint,
                             hypothesis_digest, init_lock, load_registration,
                             main, transition_problems, upgrade_lock,
                             validate_registration, verify_registration)


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
        {"id": ["H5"], "claim": "c5", "metric": "PPL", "candidate": "a",
         "reference": "b", "direction": "candidate_better"},
    ]}
    errs = validate_registration(bad)
    assert any("重复" in e for e in errs)
    assert any("direction" in e for e in errs)
    assert any("缺少字段 metric" in e for e in errs)  # H4 无 metric
    # 不可哈希 id（数组）必须显式报 schema 错误，而不是在去重时 TypeError
    assert any("非空字符串" in e for e in errs)
    print("  [ok] validate_registration（重复 id/非法 direction/缺字段/坏 id）")


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


def test_document_sha_change_warns_without_voiding_confirmatory():
    """注释类字段可改（只 warning）；判定参数 decision_rule/scope 被改写
    必须判 violated——否则改范围就能把越界 run 洗成 confirmatory。"""
    tmp = tempfile.mkdtemp()
    try:
        reg, lock = _pair(tmp)
        _write(reg, _doc([("H1", "c1")]))
        init_lock(registry_path=reg, lock_path=lock)
        d = load_registration(reg)
        d["decision_rule"] = {"alpha": 0.5}   # 改写判定参数 → 必须否决
        _write(reg, d)
        r = verify_registration(registry_path=reg, lock_path=lock)
        assert r["status"] == "violated" and not r["confirmatory_valid"]
        assert r["locked_document_matches"] is False
        assert any("decision_rule" in p for p in r["problems"])
        # 追加 note（明确允许的可读性字段）只警告不判违反
        d2 = load_registration(reg)
        d2.pop("decision_rule")
        d2["hypotheses"][0]["note"] = "补充阅读说明"
        _write(reg, d2)
        r2 = verify_registration(registry_path=reg, lock_path=lock)
        assert r2["confirmatory_valid"] and r2["warnings"]
        assert r2["document_sha256_matches_chain_head"] is False
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] verify_registration：注释警告、判定参数改写 → violated")


def test_scope_tamper_rejected_and_legacy_lock_upgrade():
    """旧锁未锁定 scope 时给 warning；升级后改写 scope → violated，
    且不能用 --amend 把改写过的范围“补手续”。"""
    tmp = tempfile.mkdtemp()
    try:
        reg, lock = _pair(tmp)
        _write(reg, _doc([("H1", "c1")]))
        init_lock(registry_path=reg, lock_path=lock)
        # 模拟旧锁：删掉 locked_document（旧版锁只有 baseline+chain）
        lk = json.load(open(lock, encoding="utf-8"))
        lk.pop("locked_document")
        if isinstance(lk.get("baseline"), dict):
            lk["baseline"].pop("locked_document_sha256", None)
        _write(lock, lk)
        r = verify_registration(registry_path=reg, lock_path=lock)
        # 没有任何 scope/decision_rule 锁定摘要：在显式升级前不能给
        # confirmatory 结论，否则旧锁可被静默改写范围。
        assert r["status"] == "violated" and not r["confirmatory_valid"]
        assert not r["locked_document_matches"]
        assert any("upgrade-lock" in p for p in r["problems"]), r
        # 显式升级后锁定当前 scope/decision_rule
        up = upgrade_lock(registry_path=reg, lock_path=lock,
                          reason="补锁判定参数", at="2026-01-01 00:00:00")
        assert up["upgraded"] is True
        assert verify_registration(registry_path=reg, lock_path=lock)[
            "confirmatory_valid"]
        # 静默改写 scope → violated
        d = load_registration(reg)
        d["scope"] = {"config": "configs/tiny.json", "seeds": [1]}
        _write(reg, d)
        r = verify_registration(registry_path=reg, lock_path=lock)
        assert r["status"] == "violated" and not r["confirmatory_valid"]
        assert any("scope/decision_rule" in p for p in r["problems"])
        # 追加假设也不能绕过范围锁
        d["hypotheses"].append({"id": "H2", "claim": "c2", "metric": "PPL",
                                "candidate": "a", "reference": "b",
                                "direction": "candidate_worse"})
        _write(reg, d)
        try:
            append_amendment(registry_path=reg, lock_path=lock, reason="补 H2")
            raise AssertionError("改写范围后不得追加修订")
        except ValueError as e:
            assert "判定参数" in str(e)
        # 已有的不同摘要不能被“升级”覆盖（先恢复假设集合，隔离范围改写）
        d2 = _doc([("H1", "c1")])
        d2["scope"] = {"config": "configs/small.json"}
        _write(reg, d2)
        try:
            upgrade_lock(registry_path=reg, lock_path=lock, reason="洗白")
            raise AssertionError("不同判定参数不得被升级覆盖")
        except ValueError as e:
            assert "判定参数" in str(e) or "另一组" in str(e)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] scope/decision_rule 独立锁定、旧锁升级与防洗白")


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


def test_corrupt_json_is_violation_not_missing_file():
    """文件存在但 JSON 损坏必须判 violated；不能洗成“从未登记/没有锁”。"""
    tmp = tempfile.mkdtemp()
    try:
        reg, lock = _pair(tmp)
        _write(reg, _doc([("H1", "c1")]))
        init_lock(registry_path=reg, lock_path=lock)
        with open(lock, "w", encoding="utf-8") as f:
            f.write("{truncated")
        r = verify_registration(registry_path=reg, lock_path=lock)
        assert r["status"] == "violated", r
        assert not r["confirmatory_valid"]
        assert any("不可解析" in p for p in r["problems"])
        os.remove(lock)
        init_lock(registry_path=reg, lock_path=lock)
        with open(reg, "w", encoding="utf-8") as f:
            f.write("not json")
        r = verify_registration(registry_path=reg, lock_path=lock)
        assert r["status"] == "violated", r
        assert not r["confirmatory_valid"]
        assert any("不可解析" in p for p in r["problems"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] 损坏的锁/登记表 → violated（不冒充 missing）")


def test_init_lock_creates_missing_parent_directory():
    """自定义 --lock 指向尚不存在的目录时也必须能首次登记（原子写入先建目录）。"""
    tmp = tempfile.mkdtemp()
    try:
        reg = os.path.join(tmp, "reg.json")
        lock = os.path.join(tmp, "nested", "deeper", "reg.lock.json")
        _write(reg, _doc([("H1", "c1")]))
        init_lock(registry_path=reg, lock_path=lock)
        assert os.path.exists(lock)
        r = verify_registration(registry_path=reg, lock_path=lock)
        assert r["status"] == "intact" and r["confirmatory_valid"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] init_lock（锁文件父目录不存在时自动创建）")


def test_malformed_lock_baseline_and_unhashable_ids_do_not_crash():
    """锁基线/登记表被改成异常结构时必须走 violated/schema_errors 报告，
    而不是让验证器 AttributeError/TypeError 崩溃（它是审计的最后一环）。"""
    tmp = tempfile.mkdtemp()
    try:
        reg, lock = _pair(tmp)
        _write(reg, _doc([("H1", "c1")]))
        init_lock(registry_path=reg, lock_path=lock)
        for bad in ("corrupted", 42, ["x"], None):
            lk = json.load(open(lock, encoding="utf-8"))
            lk["baseline"] = bad
            _write(lock, lk)
            r = verify_registration(registry_path=reg, lock_path=lock)
            assert r["status"] == "violated", bad
            assert not r["confirmatory_valid"], bad
        # 登记表被删且锁基线坏掉：missing_registry 报告不得崩溃
        os.remove(reg)
        r = verify_registration(registry_path=reg, lock_path=lock)
        assert r["status"] == "missing_registry"
        # 不可哈希 id（数组）→ violated + schema_errors，而非去重时崩溃
        _write(reg, {"hypotheses": [
            {"id": ["H1"], "claim": "c", "metric": "PPL", "candidate": "a",
             "reference": "b", "direction": "candidate_better"}]})
        r = verify_registration(registry_path=reg, lock_path=lock)
        assert r["status"] == "violated" and r["schema_errors"]
        # fingerprint 对不可哈希 id 也要能算摘要（报告落盘路径）
        fp = fingerprint({"hypotheses": [
            {"id": ["H1"], "claim": "c", "metric": "PPL", "candidate": "a",
             "reference": "b", "direction": "candidate_better"}]})
        assert fp["n_hypotheses"] == 1 and len(fp["hypotheses"]) == 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] 坏锁基线/不可哈希 id：违规报告而非崩溃")


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


def test_missing_registry_with_malformed_baseline_does_not_crash():
    """登记表被删且 lock.baseline.hypotheses 是标量时，报告不能 TypeError。"""
    tmp = tempfile.mkdtemp()
    try:
        reg, lock = _pair(tmp)
        _write(reg, _doc([("H1", "c1")]))
        init_lock(registry_path=reg, lock_path=lock)
        lk = json.load(open(lock, encoding="utf-8"))
        lk["baseline"]["hypotheses"] = 42
        _write(lock, lk)
        os.remove(reg)
        r = verify_registration(registry_path=reg, lock_path=lock)
        assert r["status"] == "missing_registry"
        assert any("登记表文件不存在" in p for p in r["problems"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] 缺登记表 + 畸形基线：missing_registry 报告不崩溃")


def test_amend_with_null_chain_treated_as_empty():
    """早期锁文件的 chain=null 在验证时兼容为无修订，amend 也必须容忍。"""
    tmp = tempfile.mkdtemp()
    try:
        reg, lock = _pair(tmp)
        _write(reg, _doc([("H1", "c1")]))
        init_lock(registry_path=reg, lock_path=lock)
        lk = json.load(open(lock, encoding="utf-8"))
        lk["chain"] = None
        _write(lock, lk)
        assert verify_registration(registry_path=reg, lock_path=lock)[
            "status"] == "intact"
        _write(reg, _doc([("H1", "c1"), ("H2", "c2")]))
        link = append_amendment(registry_path=reg, lock_path=lock,
                                reason="追加 H2")
        assert link["before"]["n_hypotheses"] == 1
        r = verify_registration(registry_path=reg, lock_path=lock)
        assert r["status"] == "amended" and r["confirmatory_valid"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] chain=null 的锁文件可正常追加修订")


def test_verify_registration_rejects_non_list_chain():
    """锁文件的 chain 被替换为非列表（字符串/数字/对象）时必须判 violated：
    否则篡改审计链会被静默当作“无修订”而仍确认有效。"""
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
    print("  [ok] verify_registration：非列表 chain 按损坏处理（静默通过一律判违反）")


def test_repo_registration_present_and_confirmatory():
    """仓库自带的预注册登记表必须存在、可确认、且不在会被清理的 docs/ 下。

    本测试把“登记表与锁必须在 preregistration/ 且完整”变成不可绕过的
    仓库断言。
    """
    assert os.path.exists(REGISTRY_PATH), (
        f"预注册登记表缺失：{REGISTRY_PATH}；禁止删除，修正走 --amend 留痕")
    assert os.path.exists(LOCK_PATH), (
        f"预注册锁文件缺失：{LOCK_PATH}；没有锁链就无法区分“未登记”与“被静默改写”")
    rel = os.path.relpath(REGISTRY_PATH, os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    assert rel == os.path.join("preregistration", "hypotheses.json"), rel
    assert not rel.startswith("docs" + os.sep), (
        "登记表不能放在 docs/：流水线清理会删除该目录下的非白名单文件")
    report = verify_registration()
    assert report["confirmatory_valid"], (
        f"仓库预注册完整性校验失败：{report['status']} {report['problems']}")
    doc = load_registration()
    ids = [h["id"] for h in doc["hypotheses"]]
    assert "H7" in ids, "H7（AP差机制对照）必须已登记"
    h7 = next(h for h in doc["hypotheses"] if h["id"] == "H7")
    assert h7["metric"] == "AP差(logit−仅查询)"
    assert h7["candidate"] == "lega" and h7["direction"] == "candidate_better"
    print("  [ok] 仓库预注册（preregistration/ 存在、confirmatory、含 H7）")
