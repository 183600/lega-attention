"""训练预算敏感性：对比/消融结论是否只在终点成立。

训练期消融（以及一切变体对比）通常只比较**最终 checkpoint**。但最终值
本身是有限训练预算下的一个观测：某个组件可能只是让优化早期更快，收益在
训练后段被其他通路追上；也可能在早期有害、后期才转为有益。只看终点会把
“暂时领先”写成“方法更好”，或把“早期有用、后期消失”当成稳定结论。

本模块只消费训练日志里**已经记录**的验证损失（`train_log.jsonl` 的
`val_loss`），按同一种子、同一评测步对齐候选与参照，给出逐预算的配对差
轨迹。它不重新训练、不重新选 checkpoint，也不改变终点检验：

* 对齐使用精确 step 交集，不插值。不同评测计划/不同步数的 run 不会被
  强行配成同一预算；
* 每个 step 的均值和检验都只使用两侧都存在的种子；轨迹取所有可用种子的
  共同 step，保证各点种子组成不随预算变化；
* 逐 step 的 p 值仅用于描述轨迹（各 step 高度相关，不能当作独立检验），
  因此不进入任何确认性判定，也不做“挑一个显著 step”的事后选择；
* 轨迹是 **val** 损失（checkpoint 选择集），不是 held-out test 终点；
  `train_log.jsonl` 缺失或 checkpoint 路径不可达时显式标注不可用。

判读口径：

* ``stable``：各评测步的平均差符号一致；
* ``transient``：符号在中间翻转但首末方向相同；
* ``reversal_nonsignificant``：首末方向相反但没有任何相反方向的显著步；
* ``budget_sensitive``：首末方向相反，且正负两个方向各至少有一个 step
  达到 p<alpha。这是描述性别名，不是确认性结论（未做跨 step 多重校正，
  且各 step 相关）。
"""
from __future__ import annotations

import json
import math
import os
from numbers import Integral
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np

from .utils import bootstrap_mean_ci, min_detectable_effect, paired_t_test

BUDGET_METRIC = "PPL曲线(log val loss)"
MIN_SEEDS = 3
ALPHA = 0.05
_BOOTSTRAP_RESAMPLES = 2000


def _finite(value):
    """有限实数；bool/下划线数字串/NaN/Inf/坏类型 -> None。"""
    if isinstance(value, (bool, np.bool_)):
        return None
    if isinstance(value, str) and "_" in value:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return out if math.isfinite(out) else None


def _strict_positive_int(value):
    """正整数 step；bool/小数/字符串一律拒绝（不静默截断）。"""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        return None
    value = int(value)
    return value if value > 0 else None


def curve_from_records(records) -> Dict[int, float]:
    """训练日志记录 -> ``{step: val_loss}``，坏行跳过、重复 step 取最后一条。

    只保留同时具备正数 ``step`` 与有限 ``val_loss`` 的记录；训练日志没有
    ``val_loss`` 的训练点自然不进入曲线。重复 step 通常是同一评测在收尾时
    又写了一次最终记录：若数值相同直接合并；若不同则采信后写入的记录
    （`final=True` 记录在日志末尾），并在值相同时保持幂等。
    """
    if not isinstance(records, (list, tuple)):
        return {}
    out: Dict[int, float] = {}
    for rec in records:
        if not isinstance(rec, Mapping):
            continue
        step = _strict_positive_int(rec.get("step"))
        loss = _finite(rec.get("val_loss"))
        if step is None or loss is None:
            continue
        out[step] = loss
    return out


def load_train_curve(path: str) -> Dict[int, float]:
    """读取 ``train_log.jsonl`` 的 val_loss 曲线；文件缺失/坏行安全降级。"""
    if not isinstance(path, str) or not path or not os.path.exists(path):
        return {}
    records = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except (TypeError, ValueError):
                    continue
                if isinstance(rec, Mapping):
                    records.append(rec)
    except OSError:
        return {}
    return curve_from_records(records)


def curves_from_runs(runs) -> Dict[str, Dict]:
    """从汇总层的 ``runs[variant][seed]`` 槽位发现训练曲线。

    eval/ruler/analysis 槽位不直接携带曲线；eval 记录的 ``ckpt`` 指向该 run
    的 checkpoint，因此同目录的 ``train_log.jsonl`` 就是产生该 checkpoint
    的训练曲线。路径缺失/旧产物不可达时该 (变体, 种子) 直接跳过。
    """
    out: Dict[str, Dict] = {}
    if not isinstance(runs, Mapping):
        return out
    for variant, by_seed in runs.items():
        if not isinstance(variant, str) or not isinstance(by_seed, Mapping):
            continue
        for seed, slots in by_seed.items():
            if not isinstance(slots, Mapping):
                continue
            ev = slots.get("eval")
            if not isinstance(ev, Mapping):
                continue
            ckpt = ev.get("ckpt")
            if not isinstance(ckpt, str) or not ckpt:
                continue
            curve = load_train_curve(
                os.path.join(os.path.dirname(ckpt), "train_log.jsonl"))
            if curve:
                out.setdefault(variant, {})[seed] = curve
    return out


def _ordered_keys(mapping):
    try:
        return sorted(mapping)
    except TypeError:
        return sorted(mapping, key=lambda s: (type(s).__name__, str(s)))


def _sign(value: Optional[float]) -> int:
    if value is None:
        return 0
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def _as_curve(value) -> Dict[int, float]:
    if isinstance(value, (list, tuple)):
        return curve_from_records(value)
    if not isinstance(value, Mapping):
        return {}
    out = {}
    for step, loss in value.items():
        s = _strict_positive_int(step)
        f = _finite(loss)
        if s is not None and f is not None:
            out[s] = f
    return out


def _step_record(step: int, diffs: List[float], alpha: float) -> Dict:
    """单个评测步的跨种子配对差摘要（无独立显著性声明）。"""
    res = paired_t_test(diffs, [0.0] * len(diffs))
    ci = bootstrap_mean_ci(diffs, n_resamples=_BOOTSTRAP_RESAMPLES, ci=1.0 - alpha)
    mean_diff = res.get("mean_diff")
    sign = _sign(mean_diff)
    return {
        "step": int(step),
        "n_seeds": int(res.get("n") or 0),
        "mean_diff": mean_diff,
        "std_diff": res.get("std_diff"),
        "ci95": list(ci) if ci is not None else None,
        "p": res.get("p"),
        "mde": min_detectable_effect(res.get("std_diff"), res.get("n")),
        "sign": sign,
        "exploratory": True,
    }


def _pair_analysis(curves, candidate: str, reference: str,
                   alpha: float, min_seeds: int) -> Dict:
    base = {
        "candidate": candidate,
        "reference": reference,
        "metric": BUDGET_METRIC,
        "lower_is_better": True,
        "available": False,
        "reason": "",
        "exploratory": True,
    }
    c_map = curves.get(candidate) if isinstance(curves, Mapping) else None
    r_map = curves.get(reference) if isinstance(curves, Mapping) else None
    if not isinstance(c_map, Mapping) or not c_map:
        base["reason"] = "candidate 缺少可解析训练曲线"
        return base
    if not isinstance(r_map, Mapping) or not r_map:
        base["reason"] = "reference 缺少可解析训练曲线"
        return base
    common_seeds = [s for s in _ordered_keys(c_map) if s in r_map]
    if not common_seeds:
        base["reason"] = "两侧没有共同种子"
        return base
    c_curves = {s: _as_curve(c_map.get(s)) for s in common_seeds}
    r_curves = {s: _as_curve(r_map.get(s)) for s in common_seeds}
    common_seeds = [s for s in common_seeds
                    if c_curves[s] and r_curves[s]]
    if not common_seeds:
        base["reason"] = "共同种子的训练曲线为空"
        return base
    # 取所有共同种子的共同 step：不让“某些 step 只有部分种子”改变轨迹的
    # 种子组成，也不跨评测计划插值。
    common_steps = None
    for s in common_seeds:
        steps = set(c_curves[s]) & set(r_curves[s])
        common_steps = steps if common_steps is None else (common_steps & steps)
    steps = sorted(common_steps or [])
    base.update({
        "available": bool(steps),
        "reason": "" if steps else "两侧没有共同评测步",
        "n_seeds": len(common_seeds),
        "seeds": [str(s) for s in common_seeds],
        "steps": steps,
        "n_steps": len(steps),
    })
    if not steps:
        return base

    per_step = []
    for step in steps:
        diffs = [c_curves[s][step] - r_curves[s][step]
                 for s in common_seeds]
        rec = _step_record(step, diffs, alpha)
        per_step.append(rec)

    final = per_step[-1]
    initial = per_step[0]
    means = [r.get("mean_diff") for r in per_step
             if r.get("mean_diff") is not None]
    signs = [r.get("sign") for r in per_step if r.get("sign") is not None]
    nonzero = [s for s in signs if s != 0]
    n_flips = sum(1 for a, b in zip(nonzero, nonzero[1:]) if a != b)
    final_sign = _sign(final.get("mean_diff"))
    initial_sign = _sign(initial.get("mean_diff"))
    agreeing = (sum(1 for s in nonzero if s == final_sign) / len(nonzero)
                if nonzero and final_sign != 0 else None)
    significant = [r for r in per_step
                   if r.get("p") is not None and r["p"] < alpha
                   and (r.get("mean_diff") or 0.0) != 0.0]
    significant_signs = {_sign(r.get("mean_diff")) for r in significant}
    opposite_significant = (-1 in significant_signs
                            and 1 in significant_signs)
    sign_reversal = (initial_sign != 0 and final_sign != 0
                     and initial_sign != final_sign)
    crosses_zero = bool(means) and min(means) < 0.0 < max(means)
    inferential = len(common_seeds) >= min_seeds and len(steps) >= 2
    if len(steps) < 2:
        pattern = "insufficient_steps"
    elif not nonzero:
        pattern = "all_zero"
    elif sign_reversal and opposite_significant and inferential:
        pattern = "budget_sensitive"
    elif sign_reversal:
        pattern = "reversal_nonsignificant"
    elif n_flips:
        pattern = "transient"
    else:
        pattern = "stable"

    # 最终方向首次稳定下来的 step：之后所有非零符号都与终点一致。
    stable_from_step = None
    if final_sign != 0:
        idx = 0
        for i in range(len(per_step) - 1, -1, -1):
            s = per_step[i].get("sign")
            if s != 0 and s != final_sign:
                idx = i + 1
                break
        stable_from_step = next(
            (r["step"] for r in per_step[idx:] if r.get("sign") == final_sign),
            steps[0])
    significant_steps = [r["step"] for r in significant]
    mean_range = [min(means), max(means)] if means else None
    base.update({
        "per_step": per_step,
        "initial_step": initial["step"],
        "initial_mean_diff": initial.get("mean_diff"),
        "final_step": final["step"],
        "final_mean_diff": final.get("mean_diff"),
        "final_ci95": final.get("ci95"),
        "final_p": final.get("p"),
        "final_mde": final.get("mde"),
        "mean_diff_range": mean_range,
        "sign_agreement_with_final": agreeing,
        "n_sign_flips": n_flips,
        "initial_sign": initial_sign,
        "final_sign": final_sign,
        "sign_reversal": bool(sign_reversal),
        "crosses_zero": bool(crosses_zero),
        "significant_steps": significant_steps,
        "opposite_significant_steps": bool(opposite_significant),
        "budget_sensitive": bool(
            sign_reversal and opposite_significant and inferential),
        "stable_from_step": stable_from_step,
        "trajectory_pattern": pattern,
        "inferential": bool(inferential),
        "note": ("轨迹只作描述；逐 step p 值高度相关且未做跨 step 校正，"
                 "不用于挑选有利 checkpoint，也不改变预注册终点判定。"),
    })
    return base


def analyze_budget_sensitivity(curves, pairs: Iterable[Tuple[str, str]],
                               alpha: float = ALPHA,
                               min_seeds: int = MIN_SEEDS) -> Dict:
    """对一组对比逐预算做同种子配对差轨迹。

    ``curves`` 为 ``{variant: {seed: {step: val_loss}}}``（也接受训练日志
    记录列表）；``pairs`` 为 ``(candidate, reference)``。返回可 JSON 序列化
    的 ``pairs`` 列表与汇总计数；缺失曲线显式写 ``available=false``，不会
    用全量 split/其他 step 伪造轨迹。
    """
    alpha_f = _finite(alpha)
    if alpha_f is None or not 0.0 < alpha_f < 1.0:
        raise ValueError(f"alpha 必须在 (0,1)，收到 {alpha!r}")
    if (isinstance(min_seeds, bool) or not isinstance(min_seeds, Integral)
            or int(min_seeds) < 2):
        raise ValueError(f"min_seeds 必须 >= 2，收到 {min_seeds!r}")
    min_seeds = int(min_seeds)
    curves = curves if isinstance(curves, Mapping) else {}
    results = []
    seen = set()
    for pair in (pairs or []):
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        candidate, reference = pair
        if not isinstance(candidate, str) or not isinstance(reference, str):
            continue
        if candidate == reference or (candidate, reference) in seen:
            continue
        seen.add((candidate, reference))
        results.append(_pair_analysis(curves, candidate, reference,
                                      alpha_f, min_seeds))
    available = [r for r in results if r.get("available")]
    sensitive = [r for r in available if r.get("budget_sensitive")]
    return {
        "schema": 1,
        "analysis": "training_budget_sensitivity",
        "metric": BUDGET_METRIC,
        "lower_is_better": True,
        "alpha": alpha_f,
        "min_seeds": min_seeds,
        "n_curves": sum(len(v) for v in curves.values()
                        if isinstance(v, Mapping)),
        "n_pairs": len(results),
        "n_available": len(available),
        "n_budget_sensitive": len(sensitive),
        "pairs": results,
        "note": ("训练日志 val_loss 上的同种子、同 step 配对差轨迹；"
                 "用于检查对比/消融结论是否只在终点成立。val 同时是 "
                 "checkpoint 选择集，且逐 step p 未做跨 step 校正，"
                 "故全部为探索性证据；不重新选 checkpoint，"
                 "不替代 held-out test 上的终点检验。"),
    }
