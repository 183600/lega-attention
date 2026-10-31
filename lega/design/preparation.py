"""开跑前准备审计：数据身份与测试集尘封核查。

设计计划只有绑定到**实际输入数据**并在任何测试集消费之前落盘，才能作为
先验证据。本模块只读数据目录的哈希与访问账本，不评测模型、不训练、也不
读取 test 的 token 内容：

* ``capture_preparation``：在计划生成前抓取数据指纹与 test 访问账本摘要；
* ``test_seal_precheck``：按计划的 look 预算核对“开跑前是否已经用完/无法
  核对测试集访问”，避免先看 test 再补设计；
* ``data_identity_check`` / ``test_seal_check``：把结果转成设计层可渲染的
  检查项，字段与 ``design_plan.json`` 一起落盘、进入 plan digest。
"""
from __future__ import annotations

import os

DEFAULT_PLANNED_LOOKS = 1


def _as_dict(value):
    return value if isinstance(value, dict) else {}


def _positive_int(value, default=DEFAULT_PLANNED_LOOKS):
    if isinstance(value, bool) or not isinstance(value, int):
        return default
    return value if value >= 1 else default


def _text(value):
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


def data_identity(fingerprint):
    """把 ``lega.repro.data_fingerprint`` 结果压成计划可绑定的稳定摘要。

    不包含 mtime/ctime 与缓存路径：同一份字节复制到另一台机器后应得到同一
    计划摘要；逐文件 SHA-256 仍保留，便于冲突时定位是哪份数据变了。
    """
    fingerprint = _as_dict(fingerprint)
    digest = _text(fingerprint.get("digest"))
    entries = fingerprint.get("files")
    if not isinstance(entries, list):
        entries = fingerprint.get("entries")
    files = []
    if isinstance(entries, list):
        for entry in entries:
            entry = _as_dict(entry)
            name = _text(entry.get("name"))
            sha = _text(entry.get("sha256"))
            if name is None or sha is None:
                continue
            size = entry.get("size")
            files.append({
                "name": name,
                "size": (size if isinstance(size, int)
                         and not isinstance(size, bool) and size >= 0
                         else None),
                "sha256": sha,
            })
    files.sort(key=lambda item: item["name"])
    test_bin_sha256 = next(
        (item["sha256"] for item in files if item["name"] == "test.bin"),
        None)
    return {
        "available": bool(digest),
        "algo": fingerprint.get("algo"),
        "digest": digest,
        "file_count": len(files),
        "files": files,
        "test_bin_sha256": test_bin_sha256,
    }


def capture_preparation(data_dir: str, data_fingerprint_value=None) -> dict:
    """抓取计划前的数据身份与 test 访问账本；失败时显式标记不可用。"""
    identity = data_identity(data_fingerprint_value)
    if not identity.get("available") and data_dir:
        try:
            from lega.repro import data_fingerprint as _data_fingerprint

            identity = data_identity(_data_fingerprint(data_dir))
        except (OSError, ValueError, TypeError) as exc:
            identity = {
                "available": False, "algo": None, "digest": None,
                "file_count": 0, "files": [], "test_bin_sha256": None,
                "error": f"{type(exc).__name__}: {exc}",
            }

    test_access = {
        "ledger_present": False,
        "records_total": 0,
        "summary": {},
        "test_data_fingerprint": None,
    }
    if data_dir:
        try:
            from lega.datacheck import (TEST_ACCESS_LOG,
                                        read_test_access_log,
                                        summarize_test_access)

            ledger_path = os.path.join(data_dir, TEST_ACCESS_LOG)
            records = read_test_access_log(data_dir)
            test_access.update({
                "ledger_present": os.path.exists(ledger_path),
                "records_total": len(records),
                "summary": summarize_test_access(records),
            })
        except (OSError, ValueError, TypeError) as exc:
            test_access["error"] = f"{type(exc).__name__}: {exc}"
        try:
            from lega.repro import test_data_fingerprint

            test_access["test_data_fingerprint"] = test_data_fingerprint(
                data_dir)
        except (OSError, ValueError, TypeError) as exc:
            test_access["fingerprint_error"] = (
                f"{type(exc).__name__}: {exc}")
    return {"data_identity": identity, "test_access": test_access}


def test_seal_precheck(raw_test_access, planned_looks=DEFAULT_PLANNED_LOOKS,
                       current_run_id=None) -> dict:
    """按预注册 look 预算核对开跑前的 test 访问。

    区分“本次 run 已经计数”与“此前其他独立 run 已占用预算”：计划应在
    新 run 消费 test 之前生成，因此 ``prior_looks`` 才是可用的独立 look 数。
    旧账本缺少 ``pipeline_run_id`` 时不能把“无法核对”当成“没访问过”。
    """
    from lega.testreuse import test_reuse_report

    raw = _as_dict(raw_test_access)
    planned = _positive_int(planned_looks)
    summary = _as_dict(raw.get("summary"))
    fingerprint = _text(raw.get("test_data_fingerprint"))
    report = test_reuse_report(summary, fingerprint, current_run_id, planned)
    observed = report.get("observed_looks")
    observed = observed if isinstance(observed, int) and not isinstance(
        observed, bool) else 0
    current_counted = report.get("current_run_in_ledger") is True
    prior_looks = max(0, observed - (1 if current_counted else 0))
    unattributed = report.get("unattributed_records")
    unattributed = (unattributed if isinstance(unattributed, int)
                    and not isinstance(unattributed, bool) else 0)
    exhausted = prior_looks >= planned
    can_start = (report.get("checked") is True and fingerprint is not None
                 and not exhausted and unattributed == 0)
    if not raw:
        reason = "未提供数据目录/访问账本，无法核对测试集尘封"
    elif fingerprint is None:
        reason = "当前数据没有 test.bin 指纹：本轮不消费测试集"
    elif report.get("checked") is not True:
        reason = str(report.get("reason") or "访问账本不可核对")
    elif exhausted:
        reason = (f"开跑前同一测试版本已有 {prior_looks} 个独立 look，"
                  f"达到/超过预算 {planned}；本轮不能再产生确认性 test 证据")
    elif unattributed:
        reason = (f"当前测试版本有 {unattributed} 条旧账本记录缺少 "
                  "pipeline_run_id，无法确认独立 look 次数")
    else:
        reason = (f"开跑前已有 {prior_looks} 个独立 look；预算 {planned}，"
                  "本轮在预算内")
    out = dict(report)
    out.update({
        "ledger_present": raw.get("ledger_present") is True,
        "records_total": raw.get("records_total", 0),
        "planned_looks": planned,
        "prior_looks": prior_looks,
        "current_run_already_counted": current_counted,
        "remaining_before_this_run": max(0, planned - prior_looks),
        "budget_exhausted_before_run": exhausted,
        "can_start_confirmatory": bool(can_start),
        "applicable": fingerprint is not None,
        "reason": reason,
    })
    return out


def data_identity_check(identity, test_access=None):
    """返回 (status, detail)：数据指纹与 test.bin 哈希的一致性核查。"""
    identity = _as_dict(identity)
    test_access = _as_dict(test_access)
    if not identity.get("available"):
        return "warn", (
            "设计计划未取得数据指纹（"
            + str(identity.get("error") or "数据文件缺失/不可读")
            + "）：计划无法绑定到具体数据字节，数据替换后不能靠计划摘要发现")
    detail = (f"数据指纹 digest={identity.get('digest')}，覆盖 "
              f"{identity.get('file_count', 0)} 个文件；逐文件 SHA-256 已写入"
              " design_plan.json 并参与 plan_digest")
    planned_test = _text(identity.get("test_bin_sha256"))
    ledger_test = _text(test_access.get("test_data_fingerprint"))
    if planned_test is None:
        detail += "；当前数据无 test.bin"
    elif ledger_test is not None and planned_test != ledger_test:
        return "fail", (
            "test.bin 哈希在数据指纹与访问账本之间不一致："
            f"{planned_test[:16]} != {ledger_test[:16]}；无法把 test 访问"
            "绑定到同一数据版本")
    elif ledger_test is not None:
        detail += "；test.bin 哈希与访问账本一致"
    return "pass", detail


def test_seal_check(report):
    """返回 (status, detail)：把开跑前 test 访问预检转成设计检查。"""
    report = _as_dict(report)
    if not report:
        return "info", ("未执行开跑前 test 访问核查（未提供数据目录）；"
                        "这不等于测试集未被访问过")
    if report.get("applicable") is not True:
        return "info", str(report.get("reason") or "当前数据无 test.bin")
    if report.get("checked") is not True:
        return "warn", str(report.get("reason") or "访问账本不可核对")
    if report.get("budget_exhausted_before_run") is True:
        return "fail", str(report.get("reason"))
    if (report.get("unattributed_records") or 0) > 0:
        return "warn", str(report.get("reason"))
    return "pass", str(report.get("reason"))
