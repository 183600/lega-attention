"""算力预算预检与**结果无关**的取舍阶梯。

为什么属于实验设计
------------------
本项目的正式实验规模是"10 个变体 × 5 个种子 × 8000 步"，算力申请里写的是
100~300 GPU·小时。这类估算如果只停留在文档里，真正的风险是：等算力账单出来
时，研究者才开始决定"跑哪几个臂、跑几个种子"——而那时**已经能看到部分结果
了**，取舍就变成结果依赖的决策，确认性结论会被"我们只跑了看起来好的那几组"
污染。

本模块把这件事前移成机器可读的闸门：

* **估算**：用试点/既有 run 的实测单步耗时（``train_step_time_ms``）或配置里
  声明的 ``design.expected_step_time_ms``，估计计划网格的总 GPU·小时；
* **判定**：与配置声明的 ``design.compute_budget_gpu_hours`` 比较，给出
  ``pass``/``warn``/``fail``；
* **取舍阶梯**：给出**机械的、结果无关的**削减方案，且明确排除"删掉确认性
  对照"这一类动作——那属于修改预注册，必须走修订链留痕，不能由算力预算
  自动决定。

阶梯的优先级是固定的（写死在代码里，不随结果变化）：

1. 先去掉**非确认性**的可选臂（安慰剂、容量匹配、原始参数化等探索性对照）；
2. 再按计划里的种子顺序**保留前缀**、减少种子数（下限是显著性判定所需的
   最小种子数 3；再少就只能描述，不能判定）。

任何削减都会写进设计计划并参与 ``plan_digest``：预算不足这件事本身必须在
开跑前可见，而不是在结果出来后才被解释成"我们没时间跑完"。
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from numbers import Real

from ._shared import _install as _shared_install

_shared_install(globals())


BUDGET_SCHEMA_VERSION = 1

BUDGET_KIND = "compute_budget_plan"

#: 评测（RULER/机制分析/坏案例）相对训练的耗时比例的默认值。RULER 是逐题
#: 贪心解码，样本数 × 上下文档位 × 任务数决定了它远不止一次前向；1/4 是
#: 保守的中位估计，实际值应由试点产物校正。
DEFAULT_EVAL_RATIO = 0.25

#: 估算不确定度带宽：实测单步耗时本身有波动，预算判定按 ±30% 取上界。
DEFAULT_ESTIMATE_BAND = 0.3

#: 显著性判定所需的最小种子数（少于该值只能描述）。
MIN_ANALYZABLE_SEEDS = 3

#: 可选（非确认性）臂的固定削减优先级。越靠前越先被舍弃；顺序写死，
#: 不随结果或运行顺序变化。
OPTIONAL_ARM_PRIORITY = (
    "lega_placebo",
    "lega_null_query",
    "lega_null_conc",
    "lega_null_len",
    "lega_raw",
    "lega_raw_no_len",
    "lega_raw_conc_only",
    "lega_raw_no_query",
)


def _budget_finite(value):
    if value is None or isinstance(value, bool):
        return None
    if not isinstance(value, Real):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _budget_positive(value, name: str):
    out = _budget_finite(value)
    if out is None or out <= 0.0:
        raise ValueError(f"{name} 必须是正数，收到 {value!r}")
    return out


def _budget_int(value, name: str, minimum: int = 1):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} 必须是整数，收到 {value!r}")
    if value < minimum:
        raise ValueError(f"{name} 必须 >= {minimum}，收到 {value!r}")
    return value


def measured_step_time_ms(results_dir: str):
    """从已有产物里取实测单步耗时（中位数），单位毫秒。

    只读 ``eval_*.json`` 的 ``train_step_time_ms`` 字段：它由训练进程写入，
    是"这一步真的花了多久"的实测值，比任何 FLOPs 估算都可靠。找不到时返回
    ``None``（调用方必须如实标注"无可用实测"，不得用 0 代替）。
    """
    if not results_dir or not os.path.isdir(str(results_dir)):
        return None
    values = []
    for path in sorted(glob.glob(os.path.join(str(results_dir), "eval_*.json"))):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict):
            continue
        value = _budget_finite(data.get("train_step_time_ms"))
        if value is not None and value > 0.0:
            values.append(value)
    if not values:
        return None
    values.sort()
    mid = len(values) // 2
    if len(values) % 2 == 1:
        return float(values[mid])
    return float((values[mid - 1] + values[mid]) / 2.0)


def _budget_config_steps(config: dict):
    cfg = config if isinstance(config, dict) else {}
    steps = _budget_finite(cfg.get("steps"))
    if steps is None or steps <= 0:
        return None
    return int(steps)


def estimate_run_hours(
    config: dict,
    *,
    step_time_ms=None,
    eval_ratio: float = DEFAULT_EVAL_RATIO,
    band: float = DEFAULT_ESTIMATE_BAND,
):
    """单个 (变体, 种子) run 的 GPU·小时估计。

    ``step_time_ms`` 为实测（或声明）的单步耗时；``eval_ratio`` 是评测相对
    训练的耗时比例。返回里同时给出点估计与上界（点估计 ×(1+band)），预算
    判定用上界，避免"刚好卡在预算线上"的乐观结论。
    """
    steps = _budget_config_steps(config)
    step_ms = _budget_finite(step_time_ms)
    if steps is None or step_ms is None or step_ms <= 0.0:
        return {
            "available": False,
            "steps": steps,
            "step_time_ms": step_ms,
            "eval_ratio": float(eval_ratio),
            "hours_per_run": None,
            "hours_per_run_upper": None,
            "reason": (
                "缺少训练步数或单步耗时，无法估算"
                if steps is not None
                else "配置缺少 steps，无法估算"
            ),
        }
    train_hours = steps * step_ms / 1000.0 / 3600.0
    ratio = float(eval_ratio)
    total = train_hours * (1.0 + max(0.0, ratio))
    band_value = max(0.0, float(band))
    return {
        "available": True,
        "steps": steps,
        "step_time_ms": float(step_ms),
        "eval_ratio": ratio,
        "hours_per_run": float(total),
        "hours_per_run_upper": float(total * (1.0 + band_value)),
        "band": band_value,
        "reason": None,
    }


def confirmatory_variants(registered: dict, planned):
    """预注册对照涉及的变体（确认性臂）。"""
    registered = registered if isinstance(registered, dict) else {}
    hyps = registered.get("hypotheses")
    out = set()
    for hyp in hyps if isinstance(hyps, (list, tuple)) else []:
        if not isinstance(hyp, dict):
            continue
        for key in ("candidate", "reference"):
            name = hyp.get(key)
            if isinstance(name, str) and name in set(planned):
                out.add(name)
    return out


def _budget_optional_order(variants):
    """可选臂的削减顺序：固定优先级在前，其余按名称稳定排序。"""
    variants = list(variants)
    priority = [v for v in OPTIONAL_ARM_PRIORITY if v in variants]
    rest = sorted(v for v in variants if v not in priority)
    return priority + rest


def build_budget_plan(
    config: dict,
    variants,
    seeds,
    *,
    registered: dict = None,
    budget_gpu_hours=None,
    step_time_ms=None,
    estimate_source: str = None,
    eval_ratio: float = DEFAULT_EVAL_RATIO,
    band: float = DEFAULT_ESTIMATE_BAND,
    policy: str = "warn",
    min_seeds: int = MIN_ANALYZABLE_SEEDS,
) -> dict:
    """生成算力预算计划与取舍阶梯。

    ``policy`` 为 ``warn``（默认，超预算只告警并给出阶梯）或 ``fail``
    （超预算即设计失败，用于"预算就是硬约束"的场景）。
    """
    if policy not in ("warn", "fail"):
        raise ValueError(f"policy 必须是 warn/fail，收到 {policy!r}")
    variants = [str(v) for v in variants]
    seeds = [int(s) for s in seeds]
    if not variants or not seeds:
        raise ValueError("variants/seeds 不能为空")

    budget = _budget_finite(budget_gpu_hours)
    if budget is not None and budget <= 0.0:
        raise ValueError("budget_gpu_hours 必须是正数")

    run = estimate_run_hours(
        config, step_time_ms=step_time_ms, eval_ratio=eval_ratio, band=band
    )
    confirmatory = sorted(confirmatory_variants(registered, variants))
    optional = [v for v in variants if v not in set(confirmatory)]
    optional_ordered = _budget_optional_order(optional)

    n_cells = len(variants) * len(seeds)
    estimate = None
    estimate_upper = None
    if run["available"]:
        estimate = run["hours_per_run"] * n_cells
        estimate_upper = run["hours_per_run_upper"] * n_cells

    ladder = []
    if estimate_upper is not None:
        keep_variants = list(variants)
        keep_seeds = list(seeds)
        dropped_variants = []
        dropped_seeds = []
        if optional_ordered:
            keep_variants = [v for v in variants if v not in set(optional_ordered)]
            dropped_variants = list(optional_ordered)
            ladder.append(
                {
                    "step": 1,
                    "action": "drop_optional_arms",
                    "dropped_variants": list(dropped_variants),
                    "dropped_seeds": [],
                    "kept_cells": len(keep_variants) * len(keep_seeds),
                    "estimated_gpu_hours": run["hours_per_run_upper"]
                    * len(keep_variants)
                    * len(keep_seeds),
                    "note": (
                        "去掉非确认性的可选臂（固定优先级）；确认性对照一个"
                        "都不动"
                    ),
                }
            )
        while len(keep_seeds) > int(min_seeds):
            keep_seeds = keep_seeds[:-1]
            dropped_seeds.append(seeds[len(keep_seeds)])
            ladder.append(
                {
                    "step": len(ladder) + 1,
                    "action": "reduce_seeds",
                    "dropped_variants": [],
                    "dropped_seeds": list(dropped_seeds),
                    "kept_cells": len(keep_variants) * len(keep_seeds),
                    "estimated_gpu_hours": run["hours_per_run_upper"]
                    * len(keep_variants)
                    * len(keep_seeds),
                    "note": (
                        f"按计划种子顺序保留前缀，减少到 {len(keep_seeds)} 个种子"
                        f"（显著性判定下限 {int(min_seeds)}）"
                    ),
                }
            )

    recommendation = None
    if estimate_upper is not None and budget is not None:
        if estimate_upper <= budget:
            recommendation = {
                "step": 0,
                "action": "keep_full_plan",
                "dropped_variants": [],
                "dropped_seeds": [],
                "kept_cells": n_cells,
                "estimated_gpu_hours": estimate_upper,
                "note": "完整计划已在预算内",
            }
        else:
            for rung in ladder:
                if rung["estimated_gpu_hours"] <= budget:
                    recommendation = rung
                    break

    if budget is None:
        status = "info"
    elif estimate_upper is None:
        status = "warn"
    elif estimate_upper <= budget:
        status = "pass"
    elif policy == "fail":
        status = "fail"
    else:
        status = "warn"

    plan = {
        "schema_version": BUDGET_SCHEMA_VERSION,
        "kind": BUDGET_KIND,
        "status": status,
        "budget_gpu_hours": budget,
        "policy": policy,
        "estimated_gpu_hours": estimate,
        "estimated_gpu_hours_upper": estimate_upper,
        "estimate_source": estimate_source,
        "hours_per_run": run.get("hours_per_run"),
        "hours_per_run_upper": run.get("hours_per_run_upper"),
        "step_time_ms": run.get("step_time_ms"),
        "eval_ratio": run.get("eval_ratio"),
        "n_cells": n_cells,
        "n_variants": len(variants),
        "n_seeds": len(seeds),
        "variants": list(variants),
        "seeds": list(seeds),
        "confirmatory_variants": confirmatory,
        "optional_variants": optional_ordered,
        "ladder": ladder,
        "recommendation": recommendation,
        "note": (
            "取舍阶梯只做机械削减：去掉非确认性可选臂、按计划顺序减少种子；"
            "它不会删除任何确认性对照——那属于修改预注册，必须走修订链。"
            "预算判定用估算上界（点估计 ×(1+band)）。"
        ),
    }
    if estimate_upper is None:
        plan["reason"] = run.get("reason")
    return plan


def budget_check(plan: dict, *, policy: str = "warn") -> dict:
    """把预算计划翻译成设计检查条目。"""
    plan = plan if isinstance(plan, dict) else {}
    status = plan.get("status")
    budget = plan.get("budget_gpu_hours")
    estimate = plan.get("estimated_gpu_hours_upper")
    if budget is None:
        return {
            "id": "compute_budget",
            "title": "算力预算预检",
            "status": "info",
            "detail": (
                "未声明 design.compute_budget_gpu_hours：不做预算判定；"
                "如需硬约束请显式声明预算与 design.compute_budget_policy"
            ),
        }
    if estimate is None:
        return {
            "id": "compute_budget",
            "title": "算力预算预检",
            "status": "warn",
            "detail": (
                f"声明预算 {float(budget):.1f} GPU·小时，但缺少实测单步耗时"
                "（或 design.expected_step_time_ms），无法估算总开销："
                "预算是否够用未知，不能默认够用"
            ),
        }
    rec = plan.get("recommendation")
    if status == "pass":
        return {
            "id": "compute_budget",
            "title": "算力预算预检",
            "status": "pass",
            "detail": (
                f"估算上界 {float(estimate):.1f} GPU·小时 ≤ 预算 "
                f"{float(budget):.1f} GPU·小时（{plan.get('n_cells')} 个 run 单元）"
            ),
        }
    detail = (
        f"估算上界 {float(estimate):.1f} GPU·小时 > 预算 {float(budget):.1f} "
        f"GPU·小时（{plan.get('n_cells')} 个 run 单元）"
    )
    if isinstance(rec, dict):
        detail += (
            f"；按预注册的取舍阶梯建议：{rec.get('note')}（保留 "
            f"{rec.get('kept_cells')} 个单元，估算 "
            f"{float(rec.get('estimated_gpu_hours') or 0.0):.1f} GPU·小时）"
        )
    else:
        detail += (
            "；阶梯内没有能压进预算的机械方案：需要研究者决定是追加算力、"
            "修改预算，还是走修订链调整预注册范围"
        )
    if status == "fail":
        detail += "；design.compute_budget_policy=fail，本轮按设计失败处理"
    else:
        detail += "；design.compute_budget_policy=warn（默认），超预算不阻断运行"
    return {
        "id": "compute_budget",
        "title": "算力预算预检",
        "status": status,
        "detail": detail,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="生成算力预算计划与取舍阶梯")
    p.add_argument("--config", required=True)
    p.add_argument("--variants", default=None, help="逗号分隔变体；缺省用配置外全部")
    p.add_argument("--seeds", default=None, help="逗号分隔种子；缺省读配置")
    p.add_argument("--registered", default=None, help="预注册登记表路径")
    p.add_argument("--results_dir", default=None, help="用于取实测单步耗时的结果目录")
    p.add_argument("--budget_gpu_hours", type=float, default=None)
    p.add_argument("--eval_ratio", type=float, default=DEFAULT_EVAL_RATIO)
    p.add_argument("--out", default="compute_budget.json")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    try:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except (OSError, ValueError) as e:
        print(f"[budget] 读取配置失败：{e}", file=sys.stderr)
        return 2
    if not isinstance(cfg, dict):
        print("[budget] 配置顶层必须是对象", file=sys.stderr)
        return 2

    design_cfg = cfg.get("design") if isinstance(cfg.get("design"), dict) else {}
    variants = None
    if args.variants:
        variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    if variants is None:
        try:
            from lega.attention import RUNNABLE_VARIANTS

            variants = list(RUNNABLE_VARIANTS)
        except (ImportError, AttributeError):
            variants = []
    if not variants:
        print("[budget] 无法确定变体集合，请显式 --variants", file=sys.stderr)
        return 2

    seeds = None
    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip().isdigit()]
    if seeds is None:
        raw = cfg.get("seeds")
        if raw is None:
            raw = [cfg.get("seed", 1337)]
        if isinstance(raw, (list, tuple)):
            seeds = [int(s) for s in raw]
    if not seeds:
        print("[budget] 无法确定种子集合，请显式 --seeds", file=sys.stderr)
        return 2

    registered = None
    reg_path = args.registered
    if reg_path is None:
        reg_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "preregistration",
            "hypotheses.json",
        )
    try:
        with open(reg_path, "r", encoding="utf-8") as f:
            registered = json.load(f)
    except (OSError, ValueError):
        registered = None

    step_time_ms = measured_step_time_ms(args.results_dir) if args.results_dir else None
    source = "measured" if step_time_ms is not None else None
    if step_time_ms is None:
        step_time_ms = design_cfg.get("expected_step_time_ms")
        if step_time_ms is not None:
            source = "declared"
    budget = args.budget_gpu_hours
    if budget is None:
        budget = design_cfg.get("compute_budget_gpu_hours")
    policy = design_cfg.get("compute_budget_policy", "warn")

    try:
        plan = build_budget_plan(
            cfg,
            variants,
            seeds,
            registered=registered,
            budget_gpu_hours=budget,
            step_time_ms=step_time_ms,
            estimate_source=source,
            eval_ratio=args.eval_ratio,
            policy=policy,
        )
    except ValueError as e:
        print(f"[budget] 生成失败：{e}", file=sys.stderr)
        return 2

    try:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(plan, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.write("\n")
    except OSError as e:
        print(f"[budget] 写入失败：{e}", file=sys.stderr)
        return 2

    if not args.quiet:
        print(
            f"[budget] status={plan['status']} 估算上界="
            f"{plan.get('estimated_gpu_hours_upper')} GPU·小时 预算="
            f"{plan.get('budget_gpu_hours')} -> {args.out}"
        )
        for rung in plan.get("ladder", []):
            print(
                f"[budget] 阶梯 {rung['step']}: {rung['action']} "
                f"保留 {rung['kept_cells']} 单元 ≈ "
                f"{float(rung['estimated_gpu_hours']):.1f} GPU·小时"
            )
    return 2 if plan.get("status") == "fail" else 0


if __name__ == "__main__":
    from lega.bootstrap import ensure_hash_seed

    ensure_hash_seed()
    raise SystemExit(main())
