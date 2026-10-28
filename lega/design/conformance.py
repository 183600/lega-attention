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

    pop = analysis_population(plan, runs)
    out["population"] = pop
    out["checked"] = True
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
    global_ids = {"seed_scope", "hypothesis_variants"}
    out["global_plan_gate"] = any(
        c.get("id") in global_ids for c in failures)
    if out["global_plan_gate"]:
        out["problems"].append(
            "先验设计检查失败（全局）："
            + "；".join(str(c.get("detail")) for c in failures
                       if c.get("id") in global_ids))
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
    global_fail = (checked and conf.get("global_plan_gate") is True)
    global_reason = (
        "先验设计检查失败："
        + "；".join(str(f.get("detail")) for f in
                   _as_list(conf.get("plan_failures"))
                   if isinstance(f, dict)
                   and f.get("id") in ("seed_scope", "hypothesis_variants"))
        if global_fail else "")
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
        md += ["", f"**RULER 先验灵敏度**（SESOI={_fmt(ruler.get('sesoi_pp'), 2, pct=True)}；"
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
            "注：RULER 的配对 McNemar 只使用不一致对；每格样本数远小于"
            " SESOI 所需时，“等价/非劣”只能作为描述性证据。"
            + best_case_note
            + "要把它升级为确认性主张，须先增加每格样本数并重新登记。")

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
        out.append(_seed_int(int(part, 0), "seed"))
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
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "preregistration", "hypotheses.json")
    try:
        with open(reg_path, "r", encoding="utf-8") as f:
            registered = json.load(f)
    except (OSError, ValueError):
        registered = None

    try:
        plan = build_design_plan(
            cfg, seeds, variants, registered,
            config_file=args.config,
            ruler_n_samples=args.ruler_n_samples, smoke=args.smoke)
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
