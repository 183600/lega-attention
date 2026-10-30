"""测试集重复使用审计：按独立流水线运行核对确认性 look 预算。"""
from __future__ import annotations

from numbers import Integral

DEFAULT_TEST_LOOK_BUDGET = 1
DEFAULT_ALPHA = 0.05


def _positive_int(value, default):
    if isinstance(value, bool) or not isinstance(value, Integral):
        return default
    value = int(value)
    return value if value >= 1 else default


def _nonnegative_int(value):
    if isinstance(value, bool) or not isinstance(value, Integral):
        return 0
    return max(0, int(value))


def _text(value):
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


def repro_test_data_fingerprint(repro):
    data = repro.get("data") if isinstance(repro, dict) else None
    files = data.get("files") if isinstance(data, dict) else None
    if not isinstance(files, list):
        return None
    for entry in files:
        if not isinstance(entry, dict) or entry.get("name") != "test.bin":
            continue
        sha = _text(entry.get("sha256"))
        if sha is not None:
            return sha
    return None


def test_dependent_metrics(runs):
    metrics = set()
    if not isinstance(runs, dict):
        return metrics
    for by_seed in runs.values():
        if not isinstance(by_seed, dict):
            continue
        for slot in by_seed.values():
            if not isinstance(slot, dict):
                continue
            ev = slot.get("eval")
            if isinstance(ev, dict) and ev.get("split") == "test":
                metrics.add("PPL")
            ruler = slot.get("ruler")
            if (isinstance(ruler, dict)
                    and ruler.get("corpus_source") == "test"):
                metrics.add("RULER")
    return metrics


def test_reuse_report(test_access, current_test_fingerprint,
                      current_run_id=None,
                      planned_looks=DEFAULT_TEST_LOOK_BUDGET,
                      alpha=DEFAULT_ALPHA):
    planned = _positive_int(planned_looks, DEFAULT_TEST_LOOK_BUDGET)
    report = {
        "checked": False,
        "planned_looks": planned,
        "observed_looks": 0,
        "current_test_data_fingerprint": None,
        "current_run_id": _text(current_run_id),
        "current_run_in_ledger": None,
        "run_ids": [],
        "unattributed_records": 0,
        "exceeded": False,
        "unverifiable": False,
        "look_adjusted_alpha": None,
        "reason": None,
    }
    if not isinstance(test_access, dict):
        report["reason"] = "缺少测试集访问摘要，无法核对独立 look 次数"
        return report
    looks = test_access.get("looks_by_test_data_fingerprint")
    if not isinstance(looks, dict):
        report["reason"] = (
            "旧账本摘要未按 pipeline_run_id 分组，无法核对独立 look 次数")
        return report
    fingerprint = _text(current_test_fingerprint)
    report["current_test_data_fingerprint"] = fingerprint
    report["checked"] = True
    if fingerprint is None:
        report["reason"] = "当前运行没有 test.bin 指纹（未消费测试集）"
        return report
    raw_ids = looks.get(fingerprint)
    run_ids = sorted({run_id for run_id in (
        _text(value) for value in (raw_ids if isinstance(raw_ids, list) else []))
        if run_id is not None})
    unattributed_map = test_access.get(
        "unattributed_records_by_test_data_fingerprint")
    unattributed = (_nonnegative_int(unattributed_map.get(fingerprint))
                    if isinstance(unattributed_map, dict) else 0)
    observed = len(run_ids)
    current = _text(current_run_id)
    report.update({
        "observed_looks": observed,
        "run_ids": run_ids,
        "unattributed_records": unattributed,
        "current_run_in_ledger": (None if current is None
                                  else current in run_ids),
        "exceeded": observed > planned,
        "unverifiable": bool(unattributed),
    })
    if isinstance(alpha, (int, float)) and not isinstance(alpha, bool) \
            and 0.0 < float(alpha) < 1.0:
        report["look_adjusted_alpha"] = float(alpha) / planned
    if report["exceeded"]:
        report["reason"] = (
            f"当前测试版本已被 {observed} 个独立流水线运行访问，"
            f"超过预注册预算 {planned}")
    elif unattributed:
        report["reason"] = (
            f"当前测试版本有 {unattributed} 条旧记录缺少 pipeline_run_id，"
            "无法确认独立 look 次数")
    else:
        report["reason"] = (
            f"当前测试版本有 {observed} 个独立流水线运行，"
            f"预算 {planned}")
    return report


def apply_test_reuse_gate(hyp_eval, reuse_report, dependent_metrics,
                          alpha=DEFAULT_ALPHA):
    items = hyp_eval if isinstance(hyp_eval, list) else []
    reuse = reuse_report if isinstance(reuse_report, dict) else {}
    if reuse.get("checked") is not True:
        return items
    deps = {str(metric) for metric in (dependent_metrics or [])
            if isinstance(metric, str) and metric}
    planned = _positive_int(reuse.get("planned_looks"),
                            DEFAULT_TEST_LOOK_BUDGET)
    exceeded = reuse.get("exceeded") is True
    adjusted_alpha = reuse.get("look_adjusted_alpha")
    if isinstance(adjusted_alpha, bool) or not isinstance(
            adjusted_alpha, (int, float)):
        adjusted_alpha = (float(alpha) / planned
                          if isinstance(alpha, (int, float))
                          and not isinstance(alpha, bool)
                          and 0.0 < float(alpha) < 1.0 else None)
    observed = reuse.get("observed_looks", 0)
    unattributed = reuse.get("unattributed_records", 0)
    for h in items:
        if not isinstance(h, dict):
            continue
        metric = h.get("metric")
        if not isinstance(metric, str) or not any(
                metric.startswith(dep) for dep in deps):
            continue
        if h.get("confirmatory_valid") is False:
            continue
        if exceeded:
            reason = (
                f"测试集重复使用超过预注册 look 预算（当前版本 "
                f"{observed} 个独立流水线运行 > {planned}）："
                "依赖测试集的确认性判定降级为探索性")
            if unattributed:
                reason += f"；另有 {unattributed} 条旧记录缺少 pipeline_run_id"
            h["confirmatory_valid"] = False
            h["confirmatory_note"] = reason
            h["test_reuse_exceeded"] = True
            h["test_reuse_observed_looks"] = observed
            h["test_reuse_planned_looks"] = planned
            continue
        p = h.get("p_adj_confirmatory")
        if (planned > 1 and adjusted_alpha is not None
                and isinstance(p, (int, float)) and not isinstance(p, bool)
                and 0.0 <= float(p) <= 1.0
                and float(p) > adjusted_alpha):
            h["confirmatory_valid"] = False
            h["confirmatory_note"] = (
                f"预注册 {planned} 次测试集 look：全局校正后 p={float(p):.6g} "
                f"未低于按 look 数保守校正的阈值 "
                f"alpha/planned={adjusted_alpha:.6g}；"
                "依赖测试集的确认性判定降级为探索性")
            h["test_look_adjusted_alpha"] = adjusted_alpha
            h["test_reuse_observed_looks"] = observed
            h["test_reuse_planned_looks"] = planned
            continue
        if planned > 1:
            h["test_look_adjusted_alpha"] = adjusted_alpha
            h["test_reuse_observed_looks"] = observed
            h["test_reuse_planned_looks"] = planned
    return items
