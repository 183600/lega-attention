"""预注册假设的完整性与防篡改（pre-registration integrity）。

科研方法上的问题：预注册（在正式实验前把方向性主张写死）只有在“事后无法
静默改写/删除”时才有意义。单独一份普通 JSON 在被改动或删除后，run_all
只能看到“文件不在”，无法区分“从未登记”与“登记后被删/改”，等于确认性
（confirmatory）与探索性（exploratory）的边界失效。

本模块把登记表变成可验证的追加式文档：

  * 登记表与锁文件存放在 ``preregistration/``（**故意不放在 docs/**：
    流水线清理只保留 docs/design_notes.md）；
  * 锁文件 ``preregistration/hypotheses.lock.json`` 记录登记的**基线**
    （整文档 SHA-256 + 逐假设科学内容摘要）与**修订链**：每次修正/追加
    假设都要在链上留痕（改动前/后两份状态 + 理由）；
  * 逐假设摘要只覆盖科学内容字段（id/claim/metric/candidate/reference/
    direction）：改写 claim、对调变体、删条目都会使摘要变化；只加注释类
    辅助字段不影响摘要；
  * 验证器沿链重放，要求既有假设的摘要**只增不改不删**，且当前登记表
    的假设集合必须与链头完全一致——新增假设若未留痕同样判违反（否则可以
    “看到结果后再补一条预注册”）；
  * 验证状态：``intact`` / ``amended`` / ``violated`` / ``missing_lock`` /
    ``missing_registry``。只有前两者 ``confirmatory_valid=True``；
    run_all 在 violated/missing 时把全部确认性判定显式降级为探索性。

CLI（修正登记表的唯一正确入口）::

    python -m lega.hypotheses --verify          # 退出码 0=可确认 1=不可确认
    python -m lega.hypotheses --init            # 首次登记：生成锁基线
    python -m lega.hypotheses --amend --reason "追加 H7：…"

局限（如实声明）：本机制防止的是**静默/意外改写**——任何绕过锁链的改动
都会在校验中显形或被 git diff 暴露；对同时改写登记表与锁文件、且同时
改写 git 历史的攻击者无法在仓库内部防御，这正是需要实验台账与 git 历史
共同审计的原因。
"""
import argparse
import hashlib
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 登记表与锁文件位于 preregistration/：CI 清理只保留 docs/design_notes.md。
REGISTRY_PATH = os.path.join(ROOT, "preregistration", "hypotheses.json")
LOCK_PATH = os.path.join(ROOT, "preregistration", "hypotheses.lock.json")

# 科学内容字段：只有这些字段参与逐假设摘要；说明/注释类字段可自由补充。
HYPOTHESIS_FIELDS = ("id", "claim", "metric", "candidate", "reference",
                     "direction")
VALID_DIRECTIONS = ("candidate_better", "candidate_worse")

VALID_STATUSES = ("intact", "amended", "violated", "missing_lock",
                  "missing_registry")


def _canonical(obj) -> str:
    """确定性 JSON 序列化：键排序、无多余空白，保证跨进程同内容同哈希。"""
    return json.dumps(obj, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"))


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def hypothesis_digest(h: dict) -> str:
    """单条假设科学内容的摘要（16 位十六进制）。

    只取 HYPOTHESIS_FIELDS：给条目补 ``note``/``rationale`` 一类辅助字段
    不算修改假设，改写任一声明字段则摘要必变。
    """
    core = {k: (h or {}).get(k) for k in HYPOTHESIS_FIELDS}
    return _sha256_text(_canonical(core))[:16]


def _hypothesis_key(hid) -> str:
    """摘要表的键：字符串 id 原样，其他 JSON 值规范化——非哈希类型（如
    数组）若直接作键会在指纹阶段崩溃，篡改过的登记表应走 schema_errors
    被显式拒绝，而不是让验证器抛 AttributeError/TypeError。"""
    return hid if isinstance(hid, str) else _canonical(hid)


def fingerprint(doc) -> dict:
    """登记文本的指纹：整文档 SHA-256 + 逐假设摘要表 + 条数与顺序。"""
    doc = doc if isinstance(doc, dict) else {}
    hs = doc.get("hypotheses")
    hs = hs if isinstance(hs, list) else []
    ids = [h.get("id") for h in hs if isinstance(h, dict)]
    return {
        "document_sha256": _sha256_text(_canonical(doc)),
        "n_hypotheses": len(hs),
        "hypothesis_ids": ids,
        "hypotheses": {_hypothesis_key(h.get("id")): hypothesis_digest(h)
                       for h in hs if isinstance(h, dict) and h.get("id")},
    }


def validate_registration(doc) -> list:
    """登记表结构校验；返回错误列表（空列表=通过）。

    校验是“可验证预注册”的前提：缺字段/重复 id/非法 direction 的条目
    无法被检验协议可靠判定，必须在汇总前显式拒绝而不是静默跳过。
    """
    errors = []
    if not isinstance(doc, dict):
        return ["登记表顶层不是 JSON 对象"]
    hs = doc.get("hypotheses")
    if not isinstance(hs, list) or not hs:
        return ["登记表缺少非空 hypotheses 列表"]
    seen = set()
    for i, h in enumerate(hs):
        where = f"hypotheses[{i}]"
        if not isinstance(h, dict):
            errors.append(f"{where} 不是 JSON 对象")
            continue
        for f in HYPOTHESIS_FIELDS:
            if not h.get(f):
                errors.append(f"{where} 缺少字段 {f}")
        if h.get("direction") not in VALID_DIRECTIONS:
            errors.append(f"{where} direction={h.get('direction')!r} 非法"
                          f"（应为 {'/'.join(VALID_DIRECTIONS)}）")
        hid = h.get("id")
        if not isinstance(hid, str) or not hid:
            # id 必须是可哈希的非空字符串；否则摘要表/去重都无法可靠工作
            errors.append(f"{where} id 缺失或不是非空字符串：{hid!r}")
        elif hid in seen:
            errors.append(f"{where} 重复的假设 id：{hid}")
        else:
            seen.add(hid)
    return errors


def load_registration(path: str = None):
    """容错读取登记表；文件缺失/不可解析返回 None（不抛错）。"""
    path = path or REGISTRY_PATH
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            doc = json.load(f)
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, dict) else None


def load_lock(path: str = None):
    """容错读取锁文件；缺失/不可解析返回 None。"""
    path = path or LOCK_PATH
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            lock = json.load(f)
    except (OSError, ValueError):
        return None
    return lock if isinstance(lock, dict) else None


def _state_ok(state) -> bool:
    return (isinstance(state, dict)
            and isinstance(state.get("hypotheses"), dict)
            and isinstance(state.get("document_sha256"), str))


def _lock_baseline(lock) -> dict:
    """锁文件 baseline 字段；结构异常（非 dict）时退化为空 dict，
    保证验证器对损坏的锁文件给出问题报告而不是属性错误。"""
    b = lock.get("baseline") if isinstance(lock, dict) else None
    return b if isinstance(b, dict) else {}


def transition_problems(before: dict, after: dict) -> list:
    """一次状态转移的 append-only 检查：既有假设只许原样保留。

    返回问题列表；“新增假设”不算问题（由调用方按是否留痕进一步判定）。
    """
    problems = []
    b = before.get("hypotheses") if isinstance(before, dict) else None
    a = after.get("hypotheses") if isinstance(after, dict) else None
    if not isinstance(b, dict) or not isinstance(a, dict):
        return ["状态缺少 hypotheses 摘要表"]
    for hid, digest in b.items():
        if hid not in a:
            problems.append(f"已登记的假设 {hid} 被删除")
        elif a[hid] != digest:
            problems.append(
                f"已登记的假设 {hid} 科学内容被改写"
                f"（摘要 {digest} → {a[hid]}）")
    return problems


def _replay_chain(lock: dict):
    """沿修订链重放，返回 (链头状态, 问题列表)。

    链上每一环都必须：结构完整、起点等于上一环终点、转移满足
    append-only（既有假设摘要不变、不删）。
    """
    problems = []
    head = lock.get("baseline")
    if not _state_ok(head):
        return None, ["锁文件 baseline 结构不完整（缺 document_sha256/"
                      "hypotheses）"]
    chain = lock.get("chain")
    if chain is None:
        chain = []  # 兼容早期无 chain 字段的锁格式
    elif not isinstance(chain, list):
        # 链结构被破坏（如被替换成字符串/数字）时不能静默当作"无修订"：
        # 那会让篡改过的锁文件仍被判 intact/amended，防篡改边界失效。
        return None, [f"锁文件 chain 结构非法（{type(chain).__name__}），"
                      "修订链无法重放——按损坏处理"]
    for i, link in enumerate(chain):
        if (not isinstance(link, dict) or not _state_ok(link.get("before"))
                or not _state_ok(link.get("after"))):
            problems.append(f"修订链[{i}] 结构不完整")
            continue
        if link["before"]["hypotheses"] != head["hypotheses"]:
            problems.append(f"修订链[{i}] 的起点与上一环终点不一致"
                            "（链断裂或被改写）")
        problems += [f"修订链[{i}]：{p}" for p in
                     transition_problems(head, link["after"])]
        head = link["after"]
    return head, problems


def verify_registration(registry_path: str = None,
                        lock_path: str = None) -> dict:
    """验证登记表相对锁文件与修订链的完整性。

    返回报告（可直接进 significance.json）：
      status             intact/amended/violated/missing_lock/missing_registry
      confirmatory_valid True 仅当 status ∈ {intact, amended}
      problems           违反 append-only 的具体原因列表（会作废 confirmatory）
      warnings           不否决 confirmatory 但需人工核对的提示（如整文档
                         SHA 变化可来自 note/decision_rule/scope 等锁链未覆盖
                         字段——只加注释类字段是允许的，改写判定参数则不行，
                         锁链无法区分，必须看 git diff）
      schema_errors      登记表结构错误
      ...                SHA-256、条数、链长、假设摘要等审计字段
    """
    registry_path = registry_path or REGISTRY_PATH
    lock_path = lock_path or LOCK_PATH
    def _rel(p):
        try:
            return os.path.relpath(p, ROOT) if os.path.abspath(p).startswith(
                ROOT) else p
        except ValueError:  # 不同盘符（Windows）时退化为原路径
            return p

    report = {
        "status": None,
        "confirmatory_valid": False,
        "registry": _rel(registry_path) if os.path.exists(registry_path) else None,
        "lock": _rel(lock_path) if os.path.exists(lock_path) else None,
        "registry_sha256": None,
        "n_hypotheses": None,
        "baseline_document_sha256": None,
        "chain_length": None,
        "document_sha256_matches_chain_head": None,
        "problems": [],
        "warnings": [],
        "schema_errors": [],
    }
    doc = load_registration(registry_path)
    lock = load_lock(lock_path)

    if doc is None:
        report["status"] = "missing_registry"
        report["problems"].append(
            "登记表文件不存在或不可解析" +
            (f"（锁文件基线记录 {len(_lock_baseline(lock).get('hypotheses') or {})} "
             "条已登记假设，删除登记表即失去预注册边界）"
             if isinstance(lock, dict) else "（且无锁文件）"))
        return report

    fp = fingerprint(doc)
    report.update({"registry_sha256": fp["document_sha256"],
                   "n_hypotheses": fp["n_hypotheses"],
                   "hypothesis_ids": fp["hypothesis_ids"],
                   "hypothesis_digests": fp["hypotheses"]})
    report["schema_errors"] = validate_registration(doc)
    if report["schema_errors"]:
        report["status"] = "violated"
        report["problems"] = list(report["schema_errors"])
        return report

    if lock is None:
        report["status"] = "missing_lock"
        report["problems"].append(
            "缺少锁文件 preregistration/hypotheses.lock.json：无法验证"
            "登记表自登记以来未被改写；请用 --init（首次登记）恢复基线")
        return report

    report["baseline_document_sha256"] = _lock_baseline(lock).get(
        "document_sha256")
    raw_chain = lock.get("chain")
    report["chain_length"] = (len(raw_chain) if isinstance(raw_chain, list)
                              else None)
    head, problems = _replay_chain(lock)
    if head is None:
        report["status"] = "violated"
        report["problems"] = problems
        return report

    # 当前状态必须与链头语义一致（既有假设原样保留）。
    problems += transition_problems(head, fp)
    if fp["hypotheses"] != head["hypotheses"]:
        added = sorted(set(fp["hypotheses"]) - set(head["hypotheses"]))
        if added:
            problems.append(
                "新增了未在修订链留痕的假设：" + ", ".join(added) +
                "（登记后新增假设必须经 --amend 留痕，否则等于事后补登记）")
    report["document_sha256_matches_chain_head"] = bool(
        fp["document_sha256"] == head.get("document_sha256"))
    if problems:
        report["status"] = "violated"
        report["problems"] = problems
        return report

    if not report["document_sha256_matches_chain_head"]:
        # 科学内容字段未变但整文档哈希变了：合法来源是给假设追加 note 等
        # 可读性字段（允许，不影响判定）；非法来源是改写 decision_rule /
        # scope 等锁链未摘要的判定参数——锁链无法区分，故不否决
        # confirmatory，但必须显式提示人工核对 git diff。
        report["warnings"].append(
            "整文档 SHA-256 与链头不一致（假设科学内容未变）：可能是追加了 "
            "note 等注释类字段（允许）；但也可能是改写了 decision_rule/"
            "scope 等判定参数（不允许、锁链不覆盖）。请核对 git diff，"
            "必要时用 --amend 留痕。")
    report["status"] = "amended" if report["chain_length"] else "intact"
    report["confirmatory_valid"] = True
    return report


def _write_json(obj, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def init_lock(registry_path: str = None, lock_path: str = None,
              reason: str = None, at: str = None) -> dict:
    """首次登记：用当前登记表生成锁基线（已存在锁文件时拒绝覆盖）。"""
    registry_path = registry_path or REGISTRY_PATH
    lock_path = lock_path or LOCK_PATH
    if os.path.exists(lock_path):
        raise ValueError(f"锁文件已存在：{lock_path}；--init 只用于首次登记，"
                         "修正请用 --amend 留痕")
    doc = load_registration(registry_path)
    if doc is None:
        raise ValueError(f"登记表不存在或不可解析：{registry_path}")
    errors = validate_registration(doc)
    if errors:
        raise ValueError("登记表结构错误，拒绝生成锁基线：" + "；".join(errors))
    lock = {
        "algo": "sha256",
        "baseline": fingerprint(doc),
        "created_at": at or time.strftime("%Y-%m-%d %H:%M:%S"),
        "reason": reason or "首次登记",
        "chain": [],
    }
    _write_json(lock, lock_path)
    return lock


def append_amendment(registry_path: str = None, lock_path: str = None,
                     reason: str = None, at: str = None) -> dict:
    """把登记表的当前状态作为一次修订追加到链上（append-only 校验）。

    前提：登记表已被人工/脚本编辑完毕；本函数用**链头**作为 before，
    校验“既有假设只增不改不删”，只允许新增假设。任何改写/删除会被拒绝，
    必须由人显式处理（这正是想让修改变得昂贵的地方）。
    """
    if not reason:
        raise ValueError("修订必须给出 --reason（说明动机与时间点）")
    registry_path = registry_path or REGISTRY_PATH
    lock_path = lock_path or LOCK_PATH
    lock = load_lock(lock_path)
    if lock is None:
        raise ValueError(f"锁文件不存在或不可解析：{lock_path}；首次登记请用 --init")
    doc = load_registration(registry_path)
    if doc is None:
        raise ValueError(f"登记表不存在或不可解析：{registry_path}")
    errors = validate_registration(doc)
    if errors:
        raise ValueError("登记表结构错误，拒绝修订：" + "；".join(errors))
    head, problems = _replay_chain(lock)
    if head is None or problems:
        raise ValueError("锁文件已损坏（链无法重放）：" + "；".join(problems))
    fp = fingerprint(doc)
    problems += transition_problems(head, fp)
    if fp["hypotheses"] == head["hypotheses"]:
        raise ValueError("假设集合与链头完全一致，没有可留痕的修订")
    if problems:
        raise ValueError("修订违反 append-only（既有假设被改写/删除）："
                         + "；".join(problems))
    link = {"at": at or time.strftime("%Y-%m-%d %H:%M:%S"),
            "reason": reason, "before": head, "after": fp}
    lock.setdefault("chain", []).append(link)
    _write_json(lock, lock_path)
    return link


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="预注册假设登记表的完整性与防篡改工具")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--verify", action="store_true", help="验证登记表完整性")
    g.add_argument("--init", action="store_true",
                   help="首次登记：生成锁基线（锁已存在时拒绝）")
    g.add_argument("--amend", action="store_true",
                   help="把当前登记表作为一次修订留痕（只允许新增假设）")
    ap.add_argument("--reason", default=None, help="修订理由（--amend 必填）")
    ap.add_argument("--registry", default=None)
    ap.add_argument("--lock", default=None)
    ap.add_argument("--json", action="store_true", help="以 JSON 输出结果")
    args = ap.parse_args(argv)
    kw = {"registry_path": args.registry, "lock_path": args.lock}

    if args.verify:
        report = verify_registration(**kw)
        if args.json:
            print(json.dumps(report, ensure_ascii=False, indent=2))
        else:
            icon = {"intact": "✅", "amended": "✅", "violated": "⛔",
                    "missing_lock": "⚠", "missing_registry": "⛔"}.get(
                        report["status"], "?")
            print(f"{icon} 预注册完整性：{report['status']}"
                  f"（假设 {report.get('n_hypotheses')} 条，"
                  f"修订链 {report.get('chain_length')} 环，"
                  f"confirmatory_valid={report['confirmatory_valid']}）")
            for p in report["problems"]:
                print(f"  - {p}")
            for w in report.get("warnings") or []:
                print(f"  ⚠ {w}")
        return 0 if report["confirmatory_valid"] else 1

    try:
        if args.init:
            lock = init_lock(reason=args.reason, **kw)
            print(f"✅ 已生成锁基线：{args.lock or LOCK_PATH}"
                  f"（{lock['baseline']['n_hypotheses']} 条假设）")
        else:
            link = append_amendment(reason=args.reason, **kw)
            print(f"✅ 已追加修订（{link['at']}）：{link['reason']}"
                  f"（假设 {link['before']['n_hypotheses']} → "
                  f"{link['after']['n_hypotheses']} 条）")
    except ValueError as e:
        print(f"⛔ {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
