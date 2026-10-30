"""计划历史、分析人群、设计符合性与渲染/CLI。"""
from __future__ import annotations
from ._shared import _install as _shared_install

_shared_install(globals())

def plan_digest(plan: dict) -> str:
    """计划内容的 SHA-256（排除 created_at/plan_digest 等非科学字段）。"""
    if not isinstance(plan, dict):
        raise ValueError("plan 必须是 dict")
    payload = {k: v for k, v in plan.items()
               if k not in ("created_at", "plan_digest")}
    # 配置文件的绝对路径是机器相关的运行位置，不是科学内容；占位文件系统
    # 位置差异不应让同一设计产生不同摘要（config_sha256 已绑定内容）。
    if isinstance(payload.get("inputs"), dict):
        payload["inputs"] = {k: v for k, v in payload["inputs"].items()
                             if k != "config_file"}
    # 预注册配置文件的位置同样是运行位置；是否一致由 config_diffs 表达。
    if isinstance(payload.get("scope_readiness"), dict):
        payload["scope_readiness"] = {
            k: v for k, v in payload["scope_readiness"].items()
            if k not in ("declared_config", "config_file")}
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _history_path(path: str) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(path)),
                        "design_plan_history.jsonl")


def load_design_plan_history(path: str) -> list:
    """读取设计计划历史（``path`` 可以是计划文件或所在目录）。

    每次 ``save_design_plan`` 追加一行（相同 digest 连续写入去重）。历史文件
    让“同一结果目录被重跑/改种子”时，旧计划不会随 ``design_plan.json`` 被
    静默覆盖；损坏行跳过，不影响当前计划。
    """
    if isinstance(path, (str, os.PathLike)):
        p = str(path)
        # 传目录时读目录下的历史；传计划文件时读同目录的历史
        # （不能把 plan JSON 本身当 JSONL 逐行解析）。
        p = (os.path.join(p, "design_plan_history.jsonl")
             if os.path.isdir(p) else _history_path(p))
    else:
        return []
    out = []
    try:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if isinstance(row, dict):
                    out.append(row)
    except OSError:
        return []
    return out


def save_design_plan(plan: dict, path: str) -> None:
    """原子写入计划，并追加到同目录的 ``design_plan_history.jsonl``。

    历史追加在计划写成功之后；连续写入同一 digest 时去重，避免每次
    ``--skip_train`` 汇总都制造重复记录。
    """
    from lega.utils import save_json
    save_json(plan, path)
    hist = _history_path(path)
    entry = {
        "created_at": plan.get("created_at"),
        "plan_digest": plan.get("plan_digest"),
        "status": plan.get("status"),
        "seeds": ((plan.get("inputs") or {}).get("seeds")
                  if isinstance(plan.get("inputs"), dict) else None),
        "variants": ((plan.get("inputs") or {}).get("variants")
                     if isinstance(plan.get("inputs"), dict) else None),
        "config_sha256": ((plan.get("inputs") or {}).get("config_sha256")
                          if isinstance(plan.get("inputs"), dict) else None),
    }
    try:
        prior = load_design_plan_history(hist)
        if prior and prior[-1].get("plan_digest") == entry["plan_digest"]:
            return
        os.makedirs(os.path.dirname(hist) or ".", exist_ok=True)
        with open(hist, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False,
                              separators=(",", ":")) + "\n")
    except OSError:
        # 历史写入失败不应毁掉主计划；主计划已原子落盘。
        pass


def load_design_plan(results_dir: str) -> dict:
    """读取结果目录的 design_plan.json；缺失/损坏/非对象返回 {}。"""
    if not isinstance(results_dir, (str, os.PathLike)):
        return {}
    path = os.path.join(results_dir, "design_plan.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            doc = json.load(f)
    except (OSError, ValueError):
        return {}
    return doc if isinstance(doc, dict) else {}


# ------------------------------------------------------- 分析人群 / 设计符合性
def _cell_set(variants, seeds):
    return [(v, s) for v in variants for s in seeds]


def analysis_population(plan: dict, runs: dict) -> dict:
    """把实际发现的 run 与计划内 (变体, 种子) 单元逐一比对。

    ``runs`` 为 ``{variant: {seed: {"eval":..., "ruler":..., "analysis":...}}}``
    （build_summary 的发现结构）。只按“是否真的产出对应产物”判断，不按
    结果好坏/是否显著筛选。
    """
    plan = _as_dict(plan)
    inputs = _as_dict(plan.get("inputs"))
    variants = [str(v) for v in _as_list(inputs.get("variants"))]
    analysis_variants = inputs.get("analysis_variants")
    analysis_variants = (set(str(v) for v in analysis_variants)
                         if isinstance(analysis_variants, list) else set(variants))
    seeds = [s for s in _as_list(inputs.get("seeds"))
             if isinstance(s, int) and not isinstance(s, bool)]
    runs = _as_dict(runs)
    expected = _cell_set(variants, seeds)
    missing_eval, missing_ruler, missing_analysis = [], [], []
    observed_eval = []
    by_variant = {}
    for v in variants:
        vm = _as_dict(runs.get(v))
        row = {"planned_seeds": list(seeds), "observed_seeds": [],
               "missing_eval": [], "missing_ruler": [],
               "missing_analysis": []}
        for s in seeds:
            slot = _as_dict(vm.get(s))
            label = f"{v}/seed{s}"
            has_eval = isinstance(slot.get("eval"), dict)
            has_ruler = isinstance(slot.get("ruler"), dict)
            has_analysis = isinstance(slot.get("analysis"), dict)
            if has_eval:
                observed_eval.append((v, s))
                row["observed_seeds"].append(s)
            else:
                missing_eval.append(label)
                row["missing_eval"].append(s)
            if not has_ruler:
                missing_ruler.append(label)
                row["missing_ruler"].append(s)
            if v in analysis_variants and not has_analysis:
                missing_analysis.append(label)
                row["missing_analysis"].append(s)
        by_variant[v] = row
    extra_variants = sorted(str(v) for v in runs
                            if str(v) not in set(variants))
    observed_seed_set = set()
    for v in variants:
        observed_seed_set.update(_as_dict(runs.get(v)).keys())
    extra_seeds = sorted((str(s) for s in observed_seed_set
                          if s not in set(seeds)))
    return {
        "expected_cells": len(expected),
        "observed_eval_cells": len(observed_eval),
        "missing_eval": missing_eval,
        "missing_ruler": missing_ruler,
        "missing_analysis": missing_analysis,
        "complete_eval": not missing_eval,
        "complete_ruler": not missing_ruler,
        "complete_analysis": not missing_analysis,
        "extra_variants": extra_variants,
        "extra_seeds": extra_seeds,
        "by_variant": by_variant,
    }


def _stage_enabled(stages, key: str) -> bool:
    if not isinstance(stages, dict):
        return True
    value = stages.get(key)
    # 未记录阶段（旧 repro）按启用处理；显式 false 才是关闭。
    return value is not False


def _analysis_code_audit(planned: dict, current: dict) -> dict:
    """比较计划落盘时与汇总时的分析代码/配置树身份。"""
    planned = _as_dict(planned)
    current = _as_dict(current)
    out = {
        "checked": False, "matches": None, "reason": None,
        "planned_digest": planned.get("digest"),
        "current_digest": current.get("digest"),
        "changed_files": [], "added_files": [], "removed_files": [],
        "n_changed": 0, "n_added": 0, "n_removed": 0,
    }
    planned_digest = planned.get("digest")
    if not isinstance(planned_digest, str) or not planned_digest:
        out["reason"] = "legacy_plan_without_code_fingerprint"
        return out
    current_digest = current.get("digest")
    if not isinstance(current_digest, str) or not current_digest:
        out["reason"] = "current_code_fingerprint_unavailable"
        out["error"] = current.get("error")
        return out
    out["checked"] = True
    if planned_digest == current_digest:
        out["matches"] = True
        out["reason"] = "unchanged"
        return out
    out["matches"] = False
    out["reason"] = "digest_changed"
    planned_map = {}
    for e in _as_list(planned.get("entries")):
        if isinstance(e, dict) and isinstance(e.get("path"), str):
            planned_map[e["path"]] = e.get("sha256")
    current_map = {}
    for e in _as_list(current.get("entries")):
        if isinstance(e, dict) and isinstance(e.get("path"), str):
            current_map[e["path"]] = e.get("sha256")
    changed = sorted(p for p in set(planned_map) & set(current_map)
                     if planned_map[p] != current_map[p])
    added = sorted(set(current_map) - set(planned_map))
    removed = sorted(set(planned_map) - set(current_map))
    out.update({
        "changed_files": changed[:20], "added_files": added[:20],
        "removed_files": removed[:20],
        "n_changed": len(changed), "n_added": len(added),
        "n_removed": len(removed),
    })
    return out


def _provenance_drift_audit(repro: dict) -> dict:
    """读取实验期间代码/数据身份漂移；缺失时明确标为未核对。"""
    repro = _as_dict(repro)
    tracking = _as_dict(repro.get("tracking_end"))
    drift = _as_dict(tracking.get("drift"))
    out = {"checked": False, "changed": None, "reason": None,
           "mismatches": [], "unknown": []}
    if not drift:
        if repro.get("status") == "completed_with_drift":
            out["changed"] = True
            out["reason"] = "status_completed_with_drift"
        else:
            out["reason"] = "tracking_end_missing"
        return out
    changed = drift.get("changed")
    out["mismatches"] = [m for m in _as_list(drift.get("mismatches"))
                         if isinstance(m, dict)]
    out["unknown"] = _as_text_list(drift.get("unknown"))
    if changed is True:
        out.update({"checked": True, "changed": True,
                    "reason": "drift_detected"})
    elif changed is False:
        out.update({"checked": True, "changed": False,
                    "reason": ("no_drift_some_unknown" if out["unknown"]
                               else "no_drift")})
    else:
        out["reason"] = "drift_flag_missing"
    return out


def design_conformance(plan: dict, repro: dict, runs: dict,
                       audit: dict = None, stages: dict = None) -> dict:
    """核对“先验设计计划”与本次实际 run 的一致性。

    返回 ``checked``（计划是否存在且与本次 repro 匹配）、``in_plan``、
    ``problems``/``warnings`` 与 ``population``。``checked=False`` 时不据此
    降级（旧产物/手工汇总保持兼容），但汇总页会显式告警。
    """
    plan = _as_dict(plan)
    repro = _as_dict(repro)
    out = {"checked": False, "in_plan": None, "reason": None,
           "problems": [], "warnings": [], "population": None,
           "plan_digest": plan.get("plan_digest"),
           "plan_status": plan.get("status")}
    if not plan:
        out["reason"] = "missing_plan"
        out["warnings"].append(
            "结果目录缺少 design_plan.json（旧产物或未走统一入口）："
            "无法核对先验设计/分析人群，不据此降级确认性判定")
        return out
    if plan.get("schema_version") != SCHEMA_VERSION:
        out["reason"] = "unsupported_schema"
        out["warnings"].append(
            f"design_plan.json schema={plan.get('schema_version')!r} 不受支持，"
            "忽略该计划")
        return out

    inputs = _as_dict(plan.get("inputs"))
    plan_variants = [str(v) for v in _as_list(inputs.get("variants"))]
    plan_seeds = [s for s in _as_list(inputs.get("seeds"))
                  if isinstance(s, int) and not isinstance(s, bool)]
    repro_variants = [str(v) for v in _as_list(repro.get("variants"))]
    repro_seeds = [s for s in _as_list(repro.get("seeds"))
                   if isinstance(s, int) and not isinstance(s, bool)]
    if repro.get("smoke") is True:
        out["reason"] = "smoke_run"
        out["warnings"].append(
            "本次为冒烟运行：设计计划只检查流程，不用于确认性设计符合性判定")
        return out
    if set(plan_variants) != set(repro_variants):
        out["reason"] = "plan_repro_variant_mismatch"
        out["warnings"].append(
            "design_plan.json 的变体集与 repro.json 不一致（计划可能是上一轮"
            "遗留）：不据此核对/降级")
        return out
    if set(plan_seeds) != set(repro_seeds):
        out["reason"] = "plan_repro_seed_mismatch"
        out["warnings"].append(
            "design_plan.json 的种子集与 repro.json 不一致（计划可能是上一轮"
            "遗留）：不据此核对/降级")
        return out
    plan_sha = inputs.get("config_sha256")
    repro_sha = repro.get("config_sha256")
    if (isinstance(plan_sha, str) and isinstance(repro_sha, str)
            and plan_sha and repro_sha and plan_sha != repro_sha):
        out["reason"] = "plan_repro_config_mismatch"
        out["warnings"].append(
            "design_plan.json 的配置 SHA-256 与 repro.json 不一致：计划可能"
            "对应旧配置，不据此核对/降级")
        return out

    # 启动时 repro.json 已记录计划摘要；汇总时磁盘上的 design_plan.json 若
    # 与之不同，说明计划在结果产生后被改写。此时后续代码冻结即使“通过”
    # 也只是对照了改写后的计划，不能作为先验设计证据。
    run_plan_digest = _as_dict(repro.get("design")).get("plan_digest")
    file_plan_digest = plan.get("plan_digest")
    digest_checked = (
        isinstance(run_plan_digest, str) and bool(run_plan_digest)
        and isinstance(file_plan_digest, str) and bool(file_plan_digest))
    out["plan_integrity"] = {
        "checked": digest_checked,
        "matches": (run_plan_digest == file_plan_digest
                    if digest_checked else None),
        "run_plan_digest": run_plan_digest,
        "file_plan_digest": file_plan_digest,
    }
    out["plan_integrity_gate"] = (
        digest_checked and run_plan_digest != file_plan_digest)

    pop = analysis_population(plan, runs)
    out["population"] = pop
    out["checked"] = True
    if out["plan_integrity_gate"]:
        out["problems"].append(
            "design_plan.json 的 plan_digest 与 repro.json 记录的启动时摘要"
            "不一致：计划可能在结果产生后被改写，分析人群/设计证据不可信；"
            "确认性判定降级为探索性")
    if not pop["complete_eval"]:
        out["problems"].append(
            f"计划内 eval 单元缺失 {len(pop['missing_eval'])} 个："
            + ", ".join(pop["missing_eval"][:8])
            + (" ..." if len(pop["missing_eval"]) > 8 else ""))
    if _stage_enabled(stages, "ruler") and not pop["complete_ruler"]:
        out["problems"].append(
            f"计划内 RULER 单元缺失 {len(pop['missing_ruler'])} 个："
            + ", ".join(pop["missing_ruler"][:8])
            + (" ..." if len(pop["missing_ruler"]) > 8 else ""))
    if _stage_enabled(stages, "analysis") and not pop["complete_analysis"]:
        out["problems"].append(
            f"计划内机制分析单元缺失 {len(pop['missing_analysis'])} 个："
            + ", ".join(pop["missing_analysis"][:8])
            + (" ..." if len(pop["missing_analysis"]) > 8 else ""))
    if pop["extra_variants"]:
        out["warnings"].append(
            f"实际产物含计划外变体 {pop['extra_variants']}（不进入本计划的"
            "分析人群，也不据此提升证据）")
    if pop["extra_seeds"]:
        out["warnings"].append(
            f"实际产物含计划外种子 {pop['extra_seeds']}（不进入本计划的"
            "分析人群）")
    audit = _as_dict(audit)
    audit_missing = _as_text_list(audit.get("missing_evals"))
    if audit_missing and not out["problems"]:
        # 产物检查与受控变量审计互相印证；审计缺失但 runs 里存在（例如
        # out/ 被清理）时也要留告警。
        out["warnings"].append(
            f"受控变量审计报告缺少 {len(audit_missing)} 个 eval.json："
            + ", ".join(audit_missing[:6]))
    # 计划本身的检查失败也要进入符合性结论：少跑预注册种子时，
    # (变体, 种子) 人口可能“看起来完整”（因为计划就是用少了种子的清单
    # 生成的），但设计已不在登记范围内。seed_scope / hypothesis_variants
    # 是全局设计缺陷，由 apply_design_gate 对所有相关假设降级；声明功效
    # 不足只作告警，不直接推翻统计判定。
    plan_checks = [c for c in _as_list(plan.get("checks"))
                   if isinstance(c, dict)]
    failures = [c for c in plan_checks if c.get("status") == "fail"]
    out["plan_failures"] = [
        {"id": c.get("id"), "detail": c.get("detail")} for c in failures]
    endpoint_failures = {}
    for c in failures:
        endpoint = PLAN_FAILURE_ENDPOINTS.get(c.get("id"))
        if endpoint:
            endpoint_failures.setdefault(endpoint, []).append(
                {"id": c.get("id"), "detail": c.get("detail")})
    out["endpoint_plan_failures"] = endpoint_failures
    global_ids = GLOBAL_PLAN_FAILURE_IDS
    out["global_plan_gate"] = any(
        c.get("id") in global_ids for c in failures)
    if out["global_plan_gate"]:
        out["problems"].append(
            "先验设计检查失败（全局）："
            + "；".join(str(c.get("detail")) for c in failures
                       if c.get("id") in global_ids))
    for endpoint, items in sorted(endpoint_failures.items()):
        out["problems"].append(
            f"先验设计检查失败（{endpoint} 端点）："
            + "；".join(str(i.get("detail")) for i in items))
        out["warnings"].append(
            f"{endpoint} 端点相关确认性判定将降级；其他端点不受此特定"
            "检查影响")

    # --- 代码冻结：计划落盘后分析代码/配置不得再变。旧计划没有指纹字段时
    # 保持兼容（不降级），但必须显式标记“未核对”，不能用未记录冒充未变化。
    planned_code = _as_dict(inputs.get("analysis_code"))
    current_code = (_analysis_code_fingerprint()
                    if planned_code.get("digest") else {})
    code_audit = _analysis_code_audit(planned_code, current_code)
    out["analysis_code"] = code_audit
    out["code_freeze_gate"] = code_audit.get("matches") is False
    if out["code_freeze_gate"]:
        changed = code_audit.get("changed_files") or []
        extra = (f"；变更文件示例：{', '.join(changed[:5])}"
                 if changed else "")
        out["problems"].append(
            "分析代码/配置树在计划落盘后发生变化（代码冻结失败）："
            f"{str(code_audit.get('planned_digest'))[:12]} -> "
            f"{str(code_audit.get('current_digest'))[:12]}"
            f"（变更 {code_audit.get('n_changed', 0)}、新增 "
            f"{code_audit.get('n_added', 0)}、删除 "
            f"{code_audit.get('n_removed', 0)} 个文件）{extra}；"
            "确认性判定降级为探索性")
    elif code_audit.get("checked") is not True:
        out["warnings"].append(
            "分析代码冻结未核对（"
            + str(code_audit.get("reason") or "unknown")
            + "）：计划未记录指纹或当前指纹不可用；不据此降级")

    # --- 实验期间漂移：开始时/结束时的 git/工作区/源码树/数据不一致时，
    # 结果不能声称对应单一版本，确认性判定同样降级。
    drift_audit = _provenance_drift_audit(repro)
    out["provenance_drift"] = drift_audit
    out["provenance_drift_gate"] = drift_audit.get("changed") is True
    if out["provenance_drift_gate"]:
        fields = []
        for m in drift_audit.get("mismatches", []):
            fields.append(str(m.get("field")))
        detail = "；".join(f for f in fields if f) or "unknown"
        out["problems"].append(
            "实验期间代码/数据身份发生变化（漂移字段："
            + detail + "）：该轮结果不能声称全程对应单一版本，"
            "确认性判定降级为探索性")
    elif drift_audit.get("checked") is not True:
        out["warnings"].append(
            "实验期间代码/数据漂移未核对（"
            + str(drift_audit.get("reason") or "unknown")
            + "）：仅表示缺少可核对记录，不等于未发生变化")

    out["in_plan"] = not out["problems"]
    return out


def _as_text_list(value):
    return [str(x) for x in value] if isinstance(value, list) else []


def _metric_endpoint(metric: str) -> str:
    if not isinstance(metric, str):
        return "eval"
    if metric.startswith("RULER"):
        return "ruler"
    if metric.startswith("AUROC") or metric.startswith("AP差") or \
            metric.startswith("PR-AUC"):
        return "analysis"
    return "eval"


def _affected_variants(hyp: dict, plan_variants) -> set:
    """假设涉及的训练变体；分析内对照（如“仅查询项基线”）不算。"""
    out = set()
    planned = set(plan_variants)
    for key in ("candidate", "reference"):
        name = hyp.get(key)
        if isinstance(name, str) and name in planned:
            out.add(name)
    return out


def apply_design_gate(hyp_eval, conformance: dict, plan: dict = None):
    """计划内单元缺失时，受影响假设的确认性判定自动降级。

    只降级“缺失单元涉及该假设候选/参照变体或该指标端点”的条目；未受影响
    的假设保留原判定。计划缺失/与本次运行不匹配时不做降级（向后兼容），
    但写入 ``design_population_ok=None`` 让读者知道人群未被核对。

    与 scope/integrity/budget 门同义：设计上不可核对的“支持/拒绝”不能
    冒充预注册证据；降级只改变证据等级，不改变统计量本身。
    """
    if not isinstance(hyp_eval, list):
        return hyp_eval
    conf = _as_dict(conformance)
    plan = _as_dict(plan) if plan is None else _as_dict(plan)
    pop = _as_dict(conf.get("population"))
    inputs = _as_dict(plan.get("inputs"))
    plan_variants = [str(v) for v in _as_list(inputs.get("variants"))]
    checked = conf.get("checked") is True
    global_reasons = []
    if checked and conf.get("global_plan_gate") is True:
        global_reasons.append(
            "先验设计检查失败："
            + "；".join(str(f.get("detail")) for f in
                       _as_list(conf.get("plan_failures"))
                       if isinstance(f, dict)
                       and f.get("id") in GLOBAL_PLAN_FAILURE_IDS))
    if checked and conf.get("plan_integrity_gate") is True:
        global_reasons.append(
            "design_plan.json 与启动时 repro.json 记录的计划摘要不一致"
            "（计划在结果产生后被改写）；该确认性判定降级为探索性")
    if checked and conf.get("code_freeze_gate") is True:
        code = _as_dict(conf.get("analysis_code"))
        global_reasons.append(
            "分析代码/配置树在计划落盘后发生变化（代码冻结失败："
            f"变更 {code.get('n_changed', 0)}、新增 "
            f"{code.get('n_added', 0)}、删除 "
            f"{code.get('n_removed', 0)} 个文件）；"
            "该确认性判定降级为探索性")
    if checked and conf.get("provenance_drift_gate") is True:
        prov = _as_dict(conf.get("provenance_drift"))
        fields = [str(m.get("field")) for m in
                  _as_list(prov.get("mismatches"))
                  if isinstance(m, dict) and m.get("field")]
        global_reasons.append(
            "实验期间代码/数据身份发生变化（漂移字段："
            + ("、".join(fields) if fields else "unknown")
            + "）；该轮结果不能声称全程对应单一版本，"
            "该确认性判定降级为探索性")
    global_fail = bool(global_reasons)
    global_reason = "；".join(global_reasons)
    for h in hyp_eval:
        if not isinstance(h, dict):
            continue
        if not checked:
            h["design_population_ok"] = None
            h["design_note"] = (
                "设计计划缺失或与本次运行不匹配，无法核对分析人群；"
                "未据此降级，但该判定不在先验设计证据之内")
            continue
        if global_fail:
            h["design_population_ok"] = False
            h["design_note"] = global_reason
            if h.get("confirmatory_valid") is not False:
                h["confirmatory_valid"] = False
                h["confirmatory_note"] = global_reason
            continue
        endpoint = _metric_endpoint(h.get("metric"))
        endpoint_fail_items = _as_dict(
            conf.get("endpoint_plan_failures")).get(endpoint)
        if isinstance(endpoint_fail_items, list) and endpoint_fail_items:
            reason = (
                f"先验设计检查失败（{endpoint} 端点）："
                + "；".join(str(i.get("detail")) for i in endpoint_fail_items
                           if isinstance(i, dict))
                + "；该确认性判定降级为探索性")
            h["design_population_ok"] = False
            h["design_note"] = reason
            if h.get("confirmatory_valid") is not False:
                h["confirmatory_valid"] = False
                h["confirmatory_note"] = reason
            continue
        missing = _as_text_list(pop.get({
            "eval": "missing_eval", "ruler": "missing_ruler",
            "analysis": "missing_analysis"}[endpoint]))
        affected = _affected_variants(h, plan_variants)
        missing_variants = set()
        for cell in missing:
            v = cell.split("/seed", 1)[0]
            if v in affected:
                missing_variants.add(v)
        if not missing_variants:
            h["design_population_ok"] = True
            h["design_note"] = (
                f"计划内 {endpoint} 单元完整（{len(affected)} 个相关变体）")
            continue
        h["design_population_ok"] = False
        reason = (
            "分析人群不完整：计划内单元缺失涉及 "
            + ", ".join(sorted(missing_variants))
            + f" 的 {endpoint} 产物；不得只保留跑成功的种子，"
            "该确认性判定降级为探索性")
        h["design_note"] = reason
        if h.get("confirmatory_valid") is False:
            continue  # 已因更早的完整性/范围/预算门降级，保留原原因
        h["confirmatory_valid"] = False
        h["confirmatory_note"] = reason
    return hyp_eval


# --------------------------------------------------------------- 实测灵敏度
def observed_design_sensitivity(tests) -> list:
    """用本次实测配对差 SD 反推每个对比的 MDE，并与 SESOI 比较（描述性）。

    注意这不是“观测功效”（observe power）——它不把 p 值换个说法，而是回答
    “以本次实际方差，这个设计能分辨多小的效应”。与 SESOI 同尺度：
    ``equivalence`` 已把 PPL 换成相对差、RULER/AUROC 换成绝对差。
    """
    rows = []
    for t in (tests if isinstance(tests, list) else []):
        if not isinstance(t, dict):
            continue
        eq = t.get("equivalence")
        if not isinstance(eq, dict) or eq.get("status") == "invalid":
            continue
        sd = eq.get("std_diff")
        n = eq.get("n")
        margin = eq.get("margin")
        try:
            sd_f = float(sd)
            margin_f = float(margin)
        except (TypeError, ValueError, OverflowError):
            continue
        if not isinstance(n, int) or isinstance(n, bool) or n < 2:
            continue
        if not (math.isfinite(sd_f) and sd_f >= 0.0
                and math.isfinite(margin_f) and margin_f > 0.0):
            continue
        mde = (0.0 if sd_f == 0.0
               else min_detectable_effect(sd_f, n, DEFAULT_ALPHA,
                                         DEFAULT_POWER))
        try:
            pw = paired_t_power(margin_f, sd_f, n, DEFAULT_ALPHA) \
                if sd_f > 0.0 else 1.0
        except ValueError:
            pw = None
        rows.append({
            "candidate": t.get("candidate"),
            "reference": t.get("reference"),
            "metric": t.get("metric"),
            "n": n,
            "std_diff_sesoi_units": sd_f,
            "sesoi_margin": margin_f,
            "observed_mde_sesoi_units": mde,
            "power_for_sesoi": pw,
            "adequate_for_sesoi": (None if mde is None
                                   else bool(mde <= margin_f)),
        })
    return rows


# ------------------------------------------------------------------- 渲染
def _count(value):
    return f"{value:,}" if isinstance(value, int) and not isinstance(
        value, bool) else "-"


def _fmt(x, nd=4, pct=False, signed=False):
    if x is None:
        return "-"
    try:
        f = float(x)
    except (TypeError, ValueError, OverflowError):
        return "-"
    if not math.isfinite(f):
        return "-"
    if pct:
        scaled = f * 100.0
        # 有限输入乘 100 仍可能溢出（如 1e308），不能写出 inf%。
        if not math.isfinite(scaled):
            return "溢出"
        return f"{scaled:+.{nd}f}%" if signed else f"{scaled:.{nd}f}%"
    return f"{f:+.{nd}f}" if signed else f"{f:.{nd}f}"


def render_design_section(plan: dict, conformance: dict = None,
                          observed: list = None) -> list:
    """把先验计划/设计符合性/实测灵敏度渲染成 summary.md 小节。"""
    md = ["", "## 实验设计与分析人群（先验计划 vs 实际）", ""]
    plan = _as_dict(plan)
    conf = _as_dict(conformance)
    if not plan:
        md.append("- ⚠ 未找到 `design_plan.json`（旧产物或未走统一入口）："
                  "本次无法核对先验设计、分析人群与缺失单元；"
                  "这不等于设计没有问题，只说明没有可核对证据。")
        return md
    inputs = _as_dict(plan.get("inputs"))
    status = plan.get("status")
    icon = {"ok": "✅", "warning": "⚠", "fail": "❌"}.get(status, "•")
    md.append(
        f"- {icon} 计划状态：`{status}`；变体 {len(_as_list(inputs.get('variants')))} "
        f"× 种子 {len(_as_list(inputs.get('seeds')))}；"
        f"alpha={_fmt(inputs.get('alpha'), 3)}、"
        f"power={_fmt(inputs.get('power'), 3)}；"
        f"SHA-256 `{str(plan.get('plan_digest'))[:12]}`（训练前落盘）")
    md.append("- 先验计划只使用配置与预注册范围，不读取任何结果/测试集；"
              "种子数或 RULER 样本量不足时不会在事后被解释成“无效应”。")
    mp = _as_dict(plan.get("multiplicity_plan"))
    if mp:
        sizes = _as_dict(mp.get("family_sizes"))
        md.append(
            f"- 确认性多重比较：`{mp.get('mode')}`；PPL 家族 m="
            f"{sizes.get('PPL')}、RULER 家族 m={sizes.get('RULER')}、"
            f"全局确认族 k={sizes.get('global_confirmatory')}；"
            f"PPL/RULER/AUROC/AP 有效阈值 α_eff="
            f"{_fmt(_as_dict(mp.get('alpha_effective')).get('PPL'), 4)}/"
            f"{_fmt(_as_dict(mp.get('alpha_effective')).get('RULER'), 4)}/"
            f"{_fmt(_as_dict(mp.get('alpha_effective')).get('AUROC'), 4)}/"
            f"{_fmt(_as_dict(mp.get('alpha_effective')).get('AP'), 4)}。"
            "先验功效同时给出未校正与确认性校正两栏；正式判定以后者为准。")
    ro = _as_dict(plan.get("run_order"))
    if ro:
        mode = ro.get("mode")
        extra = ""
        if mode == "randomized_blocked":
            extra = (f"；order_seed={ro.get('order_seed')}"
                     f"（{ro.get('seed_source')}）")
        md.append(
            f"- run 顺序：`{mode}`，{ro.get('n_blocks')} 个种子区块 × "
            f"{len(_as_list(inputs.get('variants')))} 个变体 = "
            f"{ro.get('n_cells')} 个 run{extra}。"
            + ("种子为区组、区块内随机化，顺序在跑前写死；"
               if mode == "randomized_blocked"
               else "该顺序未随机化，变体可能与时间/硬件漂移混淆；"))

    checks = [c for c in _as_list(plan.get("checks")) if isinstance(c, dict)]
    if checks:
        md += ["", "| 设计检查 | 状态 | 说明 |", "|---|---|---|"]
        for c in checks:
            st = {"pass": "✅ pass", "warn": "⚠ warn",
                  "fail": "❌ fail", "info": "ℹ info"}.get(
                      c.get("status"), str(c.get("status")))
            detail = str(c.get("detail", "")).replace("|", "\\|")
            md.append(f"| {c.get('title')} | {st} | {detail} |")

    db = _as_dict(plan.get("data_budget"))
    if db:
        md += ["", "**开跑前数据/评测预算**", ""]
        if not db.get("report_present"):
            md.append(
                "- ⚠ 没有 `data_report.json`：计划无法核对训练 token 触碰"
                "次数与 val/test 窗口；这不改变统计量，但属于准备不完整。")
        else:
            epochs = db.get("token_touches_per_train_token")
            md.append(
                f"- train tokens={_count(db.get('train_tokens'))}；"
                f"计划每种子 token 触碰="
                f"{_count(db.get('planned_tokens_seen_per_seed'))}"
                + (f"（≈{_fmt(epochs, 2)}×train；预期唯一 token 覆盖≈"
                   f"{_fmt(db.get('expected_unique_token_fraction'), 1, pct=True)}）"
                   if epochs is not None else "")
                + "。>1× 表示语料被重复使用，必须在报告中披露。")
            md.append(
                f"- eval_ctx={_count(db.get('eval_ctx'))}；val 可用窗口="
                f"{_count(db.get('val_windows_available'))} / 计划 "
                f"eval_iters={_count(db.get('eval_iters_planned'))}；"
                f"test 可用窗口={_count(db.get('test_windows_available'))}"
                "（test 只在最终评估访问）。")
        sr = _as_dict(plan.get("scope_readiness"))
        if sr:
            diffs = _as_text_list(sr.get("config_diffs"))
            scope_check = next(
                (c for c in checks if c.get("id") == "scope_config"), {})
            if scope_check.get("status") == "pass" and not diffs:
                config_note = "配置规模与预注册一致"
            elif diffs:
                config_note = "配置不一致：" + "；".join(diffs)
            else:
                config_note = (
                    "配置规模未核对（"
                    + str(scope_check.get("status") or "unknown") + "）")
            md.append(
                "- 预注册范围核对：" + config_note
                + f"；data_report 分词器={sr.get('actual_tokenizer') or '-'}。")

    ppl = _as_dict(_as_dict(plan.get("endpoints")).get("ppl"))
    if ppl:
        md += ["", f"**PPL 先验灵敏度**（SESOI={_fmt(ppl.get('sesoi_relative'), 2, pct=True)}；"
                   f"计划 {ppl.get('planned_seeds')} 个种子；"
                   f"未校正 80% 功效下可容忍的配对差 SD≈"
                   f"{_fmt(ppl.get('max_paired_sd_for_sesoi'), 3, pct=True)}；"
                   f"确认性校正后≈"
                   f"{_fmt(ppl.get('max_paired_sd_for_sesoi_confirmatory'), 3, pct=True)}）", "",
               "| 配对差SD(相对) | MDE 未校正 | MDE 确认性 | 功效 未校正 | 功效 确认性 | 所需种子 未校正 | 所需种子 确认性 |",
               "|---|---|---|---|---|---|---|"]
        for row in _as_list(ppl.get("sensitivity")):
            if not isinstance(row, dict):
                continue
            md.append("| " + " | ".join([
                _fmt(row.get("paired_sd_relative"), 3, pct=True),
                _fmt(row.get("mde_relative"), 3, pct=True),
                _fmt(row.get("mde_relative_confirmatory"), 3, pct=True),
                _fmt(row.get("power_for_sesoi"), 3),
                _fmt(row.get("power_for_sesoi_confirmatory"), 3),
                _fmt(row.get("required_seeds_for_sesoi"), 0),
                _fmt(row.get("required_seeds_for_sesoi_confirmatory"), 0),
            ]) + " |")

    ruler = _as_dict(_as_dict(plan.get("endpoints")).get("ruler"))
    if ruler:
        md += ["", f"**RULER 评测样本级先验灵敏度（配对 McNemar）**"
                   f"（SESOI={_fmt(ruler.get('sesoi_pp'), 2, pct=True)}；"
                   f"每格 n={ruler.get('samples_per_cell')}；"
                   f"格子数≈{ruler.get('cells')}；"
                   f"未校正 80% 功效下可容忍的不一致率≈"
                   f"{_fmt(ruler.get('max_discordance_for_sesoi'), 4, pct=True)}；"
                   f"确认性校正后≈"
                   f"{_fmt(ruler.get('max_discordance_for_sesoi_confirmatory'), 4, pct=True)}）", "",
               "| 不一致率 | 每格MDE 未校正 | 每格MDE 确认性 | 功效 未校正 | 功效 确认性 | 所需每格样本 未校正 | 所需每格样本 确认性 |",
               "|---|---|---|---|---|---|---|"]
        for row in _as_list(ruler.get("sensitivity")):
            if not isinstance(row, dict):
                continue
            md.append("| " + " | ".join([
                _fmt(row.get("discordance"), 2, pct=True),
                _fmt(row.get("mde_pp"), 2, pct=True),
                _fmt(row.get("mde_pp_confirmatory"), 2, pct=True),
                _fmt(row.get("power_for_sesoi"), 3),
                _fmt(row.get("power_for_sesoi_confirmatory"), 3),
                _fmt(row.get("required_samples_per_cell_for_sesoi"), 0),
                _fmt(row.get("required_samples_per_cell_for_sesoi_confirmatory"), 0),
            ]) + " |")
        md.append("")
        best_case_note = ""
        if ruler.get("mde_pp_pooled_best_case_at_reference") is not None:
            best_case_note = (
                f"即使把 {ruler.get('cells')} 个格子当作完全独立的 "
                f"{ruler.get('pooled_samples_best_case')} 个样例"
                f"（最乐观上界），参考不一致率下的 MDE 仍≈"
                f"{_fmt(ruler.get('mde_pp_pooled_best_case_at_reference'), 2, pct=True)}")
            if ruler.get("mde_pp_pooled_best_case_min_over_grid") is not None:
                best_case_note += (
                    f"、网格内最乐观仍≈"
                    f"{_fmt(ruler.get('mde_pp_pooled_best_case_min_over_grid'), 2, pct=True)}")
            best_case_note += "。"
        md.append(
            "注：RULER 的配对 McNemar 只使用不一致对，推断单位是评测样例；"
            "它描述评测抽样精度，不是确认性 H4 的检验单位。每格样本数远小于"
            " SESOI 所需时，“等价/非劣”只能作为描述性证据。"
            + best_case_note
            + "要把它升级为确认性主张，须先增加每格样本数并重新登记。")

        seed = _as_dict(ruler.get("seed_level"))
        if seed:
            md += ["", "**RULER 种子级先验灵敏度（确认性 H4 的实际检验单位，"
                       "配对 t）**（SESOI=" + _fmt(seed.get("sesoi_pp"), 2, pct=True)
                       + "；"
                       f"计划 {seed.get('planned_seeds')} 个种子；"
                       f"未校正 80% 功效下可容忍的跨种子配对差 SD≈"
                       f"{_fmt(seed.get('max_paired_sd_for_sesoi_pp'), 3, pct=True)}；"
                       f"确认性校正后≈"
                       f"{_fmt(seed.get('max_paired_sd_for_sesoi_pp_confirmatory'), 3, pct=True)}）", "",
                   "| 跨种子配对差SD | MDE 未校正 | MDE 确认性 | 功效 未校正 | 功效 确认性 | 所需种子 未校正 | 所需种子 确认性 |",
                   "|---|---|---|---|---|---|---|"]
            for row in _as_list(seed.get("sensitivity")):
                if not isinstance(row, dict):
                    continue
                md.append("| " + " | ".join([
                    _fmt(row.get("paired_sd_pp"), 3, pct=True),
                    _fmt(row.get("mde_pp"), 3, pct=True),
                    _fmt(row.get("mde_pp_confirmatory"), 3, pct=True),
                    _fmt(row.get("power_for_sesoi"), 3),
                    _fmt(row.get("power_for_sesoi_confirmatory"), 3),
                    _fmt(row.get("required_seeds_for_sesoi"), 0),
                    _fmt(row.get("required_seeds_for_sesoi_confirmatory"), 0),
                ]) + " |")
            md.append("")
            md.append(
                "注：本表的 SD 是换训练种子时 macro RULER EM 配对差的标准差"
                "（绝对 EM 尺度），与上一张逐格 McNemar 表回答不同问题；"
                "增加每格样例数不会降低种子级方差，反之亦然。")

    endpoints_doc = _as_dict(plan.get("endpoints"))
    mech_rows = []
    for key, title in (("auroc", "AUROC"), ("ap", "AP")):
        ep = _as_dict(endpoints_doc.get(key))
        if ep:
            mech_rows.append((key, title, ep))
    if mech_rows:
        md += ["", "**机制端点先验灵敏度**（跨种子配对差，绝对尺度；"
                   "单 run 有限样例误差另由聚类 bootstrap 报告）", "",
               "| 端点 | SESOI | 配对差SD | MDE 未校正 | MDE 确认性 | "
               "功效 未校正 | 功效 确认性 | 所需种子 未校正 | 所需种子 确认性 |",
               "|---|---|---|---|---|---|---|---|---|"]
        for _key, title, ep in mech_rows:
            for row in _as_list(ep.get("sensitivity")):
                if not isinstance(row, dict):
                    continue
                md.append("| " + " | ".join([
                    title,
                    _fmt(ep.get("sesoi_absolute"), 3),
                    _fmt(row.get("paired_sd_absolute"), 4),
                    _fmt(row.get("mde_absolute"), 4),
                    _fmt(row.get("mde_absolute_confirmatory"), 4),
                    _fmt(row.get("power_for_sesoi"), 3),
                    _fmt(row.get("power_for_sesoi_confirmatory"), 3),
                    _fmt(row.get("required_seeds_for_sesoi"), 0),
                    _fmt(row.get("required_seeds_for_sesoi_confirmatory"), 0),
                ]) + " |")
        md.append("")
        md.append(
            "注：本表的 SD 是换训练种子时配对差的标准差，不包含单次评测"
            "有限样例的抽样误差；后者回答“换题面/样例波动多少”，"
            "两者不能互相替代。")

    # ---- 设计符合性/分析人群
    md += ["", "**设计符合性与分析人群**", ""]
    if not conf:
        md.append("- ⚠ 未生成设计符合性报告。")
    elif conf.get("checked") is not True:
        reason = conf.get("reason")
        md.append(f"- ⚠ 未核对（reason=`{reason}`）："
                  + "；".join(_as_text_list(conf.get("warnings"))[:3]))
    else:
        pop = _as_dict(conf.get("population"))
        md.append(
            f"- 计划内单元格 {pop.get('expected_cells')} 个；"
            f"实际有 eval 产物 {pop.get('observed_eval_cells')} 个；"
            f"完整={pop.get('complete_eval')}")
        integrity = _as_dict(conf.get("plan_integrity"))
        if integrity.get("matches") is False:
            md.append(
                "- ❌ 计划完整性失败：`design_plan.json` 的 plan_digest 与"
                "启动时 `repro.json` 记录不一致，计划可能在结果产生后被"
                "改写；确认性判定已降级。")
        code = _as_dict(conf.get("analysis_code"))
        if code:
            if code.get("matches") is True:
                md.append(
                    "- ✅ 分析代码冻结：自计划落盘后未变化（digest "
                    f"`{str(code.get('current_digest'))[:12]}`）。")
            elif code.get("matches") is False:
                files = _as_text_list(code.get("changed_files"))
                md.append(
                    "- ❌ 分析代码冻结失败：计划落盘后代码/配置已变化（"
                    f"变更 {code.get('n_changed', 0)}、新增 "
                    f"{code.get('n_added', 0)}、删除 "
                    f"{code.get('n_removed', 0)} 个文件"
                    + (f"；示例 {', '.join(files[:5])}" if files else "")
                    + "）；确认性判定已降级。")
            else:
                md.append(
                    "- ⚠ 分析代码冻结未核对（"
                    + str(code.get("reason") or "unknown")
                    + "）：不能把“未记录”当成“未变化”。")
        prov = _as_dict(conf.get("provenance_drift"))
        if prov:
            if prov.get("changed") is True:
                fields = [str(m.get("field")) for m in
                          _as_list(prov.get("mismatches"))
                          if isinstance(m, dict) and m.get("field")]
                md.append(
                    "- ❌ 实验期间代码/数据身份漂移（"
                    + ("、".join(fields) if fields else "unknown")
                    + "）：结果不能声称全程对应单一版本，确认性判定已降级。")
            elif prov.get("checked") is True:
                unknown = _as_text_list(prov.get("unknown"))
                if unknown:
                    md.append(
                        "- ⚠ 实验期间代码/数据身份：已核对字段未漂移，但 "
                        + "、".join(unknown)
                        + " 未核对；不能把部分核对当成全部一致。")
                else:
                    md.append("- ✅ 实验期间代码/数据身份未漂移。")
            else:
                md.append(
                    "- ⚠ 实验期间代码/数据漂移未核对（"
                    + str(prov.get("reason") or "unknown")
                    + "）：缺少可核对记录不等于未发生变化。")
        if conf.get("problems"):
            for p in _as_text_list(conf.get("problems"))[:5]:
                md.append(f"- ❌ {p}")
            md.append(
                "- 结论：受影响假设的确认性判定已自动降级为探索性；"
                "不得只保留跑成功的种子/变体来宣称预注册结果。"
                "若缺失机制与结果相关（例如发散只发生在某变体），"
                "可用样本估计本身也有偏。")
        else:
            md.append("- ✅ 计划内单元齐全：确认性判定基于全部计划种子。")
        for w in _as_text_list(conf.get("warnings"))[:3]:
            md.append(f"- ⚠ {w}")
        by_variant = _as_dict(pop.get("by_variant"))
        missing_rows = [v for v, row in by_variant.items()
                        if isinstance(row, dict) and row.get("missing_eval")]
        if missing_rows:
            md.append("- 逐变体缺失种子："
                      + "；".join(
                          f"{v}: {row.get('missing_eval')}"
                          for v, row in by_variant.items()
                          if isinstance(row, dict) and row.get("missing_eval")))

    # ---- 实测灵敏度（描述性）
    observed = [r for r in (observed or []) if isinstance(r, dict)]
    if observed:
        md += ["", "**本次实测方差下的设计灵敏度（未校正口径，描述性，不作显著性判定）**", "",
               "| 对比 | 指标 | n | 配对差SD | SESOI | 实测MDE | 对SESOI功效 | 是否够用 |",
               "|---|---|---|---|---|---|---|---|"]
        for r in observed[:20]:
            md.append("| " + " | ".join([
                f"{r.get('candidate')} vs {r.get('reference')}",
                str(r.get("metric")), str(r.get("n")),
                _fmt(r.get("std_diff_sesoi_units"), 4),
                _fmt(r.get("sesoi_margin"), 4),
                _fmt(r.get("observed_mde_sesoi_units"), 4),
                _fmt(r.get("power_for_sesoi"), 3),
                ("✅" if r.get("adequate_for_sesoi") is True
                 else "⚠" if r.get("adequate_for_sesoi") is False else "-"),
            ]) + " |")
        md.append("")
        md.append(
            "注：实测 MDE 由本次观测到的配对差 SD 反推，回答“这次设计实际"
            "能分辨多小的效应”；MDE>SESOI 的对比即使显著，也不能证明达到"
            "实际重要差异。它不是把 p 值改写成“观测功效”。")
    return md


# ----------------------------------------------------------------------- CLI
def _parse_seeds(spec):
    if spec is None or spec == "":
        return []
    out = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        # 不能先用 int(part, 0)：int("1_0", 0) == 10、int("0x10", 0) == 16，
        # 会让拼写/进制不同的种子静默变成另一个训练身份。直接用全项目的
        # 严格种子入口，与 run_all/tuning 的 --seeds 口径一致。
        from lega.utils import seed_int
        out.append(seed_int(part, "seed"))
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="生成训练前设计计划（先验功效/分析人群/设计符合性）")
    p.add_argument("--config", required=True)
    p.add_argument("--seeds", default=None,
                   help="逗号分隔种子；缺省读配置 seeds（再缺省 [seed]）")
    p.add_argument("--variants", default=None,
                   help="逗号分隔变体；缺省全部可运行变体")
    p.add_argument("--out", default="design_plan.json")
    p.add_argument("--registered", default=None,
                   help="预注册表路径（默认 preregistration/hypotheses.json）")
    p.add_argument("--ruler_n_samples", type=int, default=None)
    p.add_argument("--data_dir", default=None,
                   help="数据目录（读取 data_report.json 做 token/窗口预算检查）")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    try:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except (OSError, ValueError) as e:
        print(f"[design] 读取配置失败：{e}", file=sys.stderr)
        return 2
    if not isinstance(cfg, dict):
        print("[design] 配置顶层必须是对象", file=sys.stderr)
        return 2

    if args.variants:
        variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    else:
        variants = sorted(_known_variant_names())
    if not variants:
        print("[design] 没有可用变体（请显式 --variants）", file=sys.stderr)
        return 2

    if args.seeds:
        seeds = _parse_seeds(args.seeds)
    else:
        raw = cfg.get("seeds")
        if raw is None:
            raw = [cfg.get("seed", 1337)]
        seeds = [_seed_int(s, "seed") for s in raw]

    registered = None
    reg_path = args.registered
    if reg_path is None:
        reg_path = os.path.join(
            _DESIGN_ROOT, "preregistration", "hypotheses.json")
    try:
        with open(reg_path, "r", encoding="utf-8") as f:
            registered = json.load(f)
    except (OSError, ValueError):
        registered = None

    data_report = None
    if args.data_dir:
        try:
            from lega.preflight import load_data_report
            data_report = load_data_report(args.data_dir)
        except (ImportError, OSError, ValueError):
            data_report = None
        if data_report is None:
            print("[design] 警告：--data_dir 下缺少可解析的 data_report.json，"
                  "跳过 token/窗口预算检查", file=sys.stderr)

    try:
        plan = build_design_plan(
            cfg, seeds, variants, registered,
            config_file=args.config,
            ruler_n_samples=args.ruler_n_samples, smoke=args.smoke,
            data_report=data_report)
    except ValueError as e:
        print(f"[design] 计划生成失败：{e}", file=sys.stderr)
        return 2
    save_design_plan(plan, args.out)
    if not args.quiet:
        print(f"[design] 计划已写入 {args.out}（status={plan['status']}，"
              f"digest={plan['plan_digest'][:12]}）")
        for c in plan["checks"]:
            if c["status"] in ("warn", "fail"):
                print(f"[design] {c['status']}: {c['title']} — {c['detail']}")
    # 设计缺陷不在此处硬失败（可用 --smoke/探索性运行），但退出码区分
    # fail，便于脚本在正式实验前做闸门。
    return 2 if plan["status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
