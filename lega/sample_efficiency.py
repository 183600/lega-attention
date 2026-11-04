"""样本效率（steps-to-target）：到达参照终点水平需要多少训练步。

终点对比（PPL/RULER 配对检验）回答"训练完谁更好"；训练预算敏感性
（``lega.budget_sensitivity``）回答"终点结论是否在整个训练过程成立"。
但两者都不能区分两种性质不同的收益：

* **终点更好**：候选的渐近水平低于参照（同样训练够久也更好）；
* **仅收敛更快**：候选更早到达参照的最终水平，但渐近水平相同——
  这类收益在等步数对比中表现为"暂时领先"，延长训练后消失。

本模块用 time-to-result 口径补上这个维度：以**参照在该种子下的终点
val_loss 为目标**，计算两侧曲线首次降到目标及以下的评测步
``steps_to_target``。参照按构造必在其终点步到达目标；候选更早到达
（比值 <1）说明收敛更快，从未到达说明其终点尚未追平参照。

判读口径与边界：

* 目标是 **val** 损失（checkpoint 选择集），不是 held-out test；结论只
  关于训练动力学，不改变任何终点/预注册判定，全部为探索性证据；
* 逐种子独立计算，再跨种子聚合：log2(步数比) 的配对 t 检验（H0：比值
  =1）+ Holm 校正（本分析内部一个小家族）+ 符号检验作稳健性对照；
* ``train_log.jsonl`` 只在 ``eval_every`` 间隔上有观测：步数比精确到
  评测粒度，"快 1.4 倍"与"快 1.6 倍"可能在同一档观测上不可区分；
* 首次穿越可能是噪声性瞬态（此后又升回目标以上）：如实标注
  ``transient``，不把它伪装成稳定收敛；
* 参照在首个评测步即达目标时比值退化（目标太容易），显式标记
  ``degenerate`` 并从检验中剔除该种子；
* 候选在自身预算内从未到达目标：不计入比值（不插值、不外推），但
  计入 ``n_not_reached`` 并在结论中披露——这是"更慢"的最强形式；
* 曲线缺失/无共同种子时显式 ``available=false``，不伪造对比。
"""

from __future__ import annotations

import math
from numbers import Integral
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np

from .budget_sensitivity import (
    _as_curve,
    _finite,
    _ordered_keys,
    _strict_positive_int,
)
from .utils import holm_bonferroni, paired_t_test

SAMPLE_EFFICIENCY_METRIC = "样本效率(log2步数比)"
MIN_SEEDS = 3
ALPHA = 0.05


def steps_to_target(
    curve: Mapping, target: float, lower_is_better: bool = True
) -> Optional[Dict]:
    """曲线中首次到达目标的评测步；未到达返回 None。

    ``curve`` 为 ``{step: val_loss}``。返回 ``{"step": int,
    "transient": bool}``：``transient=True`` 表示该步之后（截至曲线末尾）
    损失又回到目标之外——首次穿越是瞬态而非稳定收敛。目标必须有限；
    曲线为空/无有效点时返回 None。
    """
    tgt = _finite(target)
    if tgt is None or not isinstance(curve, Mapping) or not curve:
        return None
    points = sorted(
        (step, loss)
        for step, loss in (
            (s, _finite(v)) for s, v in curve.items()
        )
        if _ordered_key(step) is not None and loss is not None
    )
    if not points:
        return None
    first = None
    for step, loss in points:
        reached = loss <= tgt if lower_is_better else loss >= tgt
        if reached:
            first = step
            break
    if first is None:
        return None
    if lower_is_better:
        after = [loss for step, loss in points if _ordered_key(step) > first]
        transient = bool(after) and max(after) > tgt
    else:
        after = [loss for step, loss in points if _ordered_key(step) > first]
        transient = bool(after) and min(after) < tgt
    return {"step": first, "transient": transient}


def _ordered_key(step) -> Optional[int]:
    """评测步 → 正整数；bool/小数/非正数一律拒绝（与预算轨迹同口径）。"""
    return _strict_positive_int(step)


def _pair_analysis(
    curves, candidate: str, reference: str, alpha: float, min_seeds: int
) -> Dict:
    base = {
        "candidate": candidate,
        "reference": reference,
        "metric": SAMPLE_EFFICIENCY_METRIC,
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
    common_seeds = [s for s in common_seeds if c_curves[s] and r_curves[s]]
    if not common_seeds:
        base["reason"] = "共同种子的训练曲线为空"
        return base

    per_seed = []
    log_ratios = []
    n_not_reached = 0
    n_transient = 0
    n_degenerate = 0
    reached_earlier = 0
    n_comparable = 0
    for s in common_seeds:
        r_curve = r_curves[s]
        # 目标 = 参照在该种子下的终点 val_loss：两侧用同一目标值，
        # "快多少"因此是同一水平线上的步数比较，而不是各自终点的比较。
        target = r_curve[max(r_curve, key=_ordered_key)]
        c_hit = steps_to_target(c_curves[s], target)
        r_hit = steps_to_target(r_curve, target)
        rec = {"seed": str(s), "target": target}
        if r_hit is None:
            # 参照终点值按构造必可达；None 只能来自坏曲线/坏目标。
            rec.update(
                {
                    "steps_reference": None,
                    "steps_candidate": c_hit["step"] if c_hit else None,
                    "log2_ratio": None,
                    "degenerate": False,
                    "transient_candidate": bool(c_hit and c_hit["transient"]),
                }
            )
            per_seed.append(rec)
            continue
        rec["steps_reference"] = r_hit["step"]
        rec["degenerate"] = bool(
            r_hit["step"] == min(r_curve, key=_ordered_key)
        )
        rec["transient_candidate"] = bool(c_hit and c_hit["transient"])
        if c_hit is None:
            rec["steps_candidate"] = None
            rec["log2_ratio"] = None
            n_not_reached += 1
            per_seed.append(rec)
            continue
        rec["steps_candidate"] = c_hit["step"]
        if c_hit["transient"]:
            n_transient += 1
        if rec["degenerate"]:
            n_degenerate += 1
            per_seed.append(rec)
            continue
        n_comparable += 1
        if c_hit["step"] < r_hit["step"]:
            reached_earlier += 1
        ratio = c_hit["step"] / r_hit["step"]
        rec["log2_ratio"] = math.log2(ratio) if ratio > 0 else None
        if rec["log2_ratio"] is not None and math.isfinite(rec["log2_ratio"]):
            log_ratios.append(rec["log2_ratio"])
        per_seed.append(rec)

    base.update(
        {
            "available": True,
            "reason": "",
            "n_seeds": len(common_seeds),
            "seeds": [str(s) for s in common_seeds],
            "per_seed": per_seed,
            "target_rule": "reference 逐种子终点 val_loss（两侧同一目标）",
            "n_comparable": n_comparable,
            "n_not_reached": n_not_reached,
            "n_transient": n_transient,
            "n_degenerate_reference_first_step": n_degenerate,
            "frac_reached_earlier": (
                reached_earlier / n_comparable if n_comparable else None
            ),
        }
    )
    if n_comparable == 0:
        base.update(
            {
                "median_ratio": None,
                "mean_log2_ratio": None,
                "p": None,
                "p_adj": None,
                "inferential": False,
                "verdict": (
                    "无可比较种子（候选全部未到达参照终点，或参照在首个评测步"
                    "即达目标）；不能给出收敛速度结论"
                ),
            }
        )
        return base
    ratios = [2.0**v for v in log_ratios if v is not None]
    base["median_ratio"] = float(np.median(ratios)) if ratios else None
    base["mean_log2_ratio"] = (
        float(np.mean(log_ratios)) if log_ratios else None
    )
    if len(log_ratios) < 2 or all(
        abs(v - log_ratios[0]) < 1e-15 for v in log_ratios
    ):
        # 零方差/单观测：t 检验无定义；只能描述，不能报 p。
        base["p"] = None
        base["p_adj"] = None
        base["inferential"] = False
        if len(log_ratios) < 2:
            base["verdict"] = (
                f"可比较种子不足（n={len(log_ratios)}），仅描述性"
                if log_ratios
                else "无可计算步数比"
            )
        else:
            base["verdict"] = (
                f"步数比跨种子无方差（median_ratio={base['median_ratio']:.3f}）；"
                "仅描述性"
            )
        return base
    res = paired_t_test(log_ratios, [0.0] * len(log_ratios))
    base["p"] = res.get("p")
    base["mean_log2_ratio"] = res.get("mean_diff")
    base["mean_diff"] = res.get("mean_diff")
    base["std_log2_ratio"] = res.get("std_diff")
    base["n"] = res.get("n")
    base["inferential"] = bool(
        res.get("n") is not None and res["n"] >= min_seeds
    )
    if not base["inferential"]:
        base["verdict"] = f"种子不足（n={res.get('n')}<{min_seeds}），仅描述性"
    elif base["p"] is not None and base["p"] < alpha:
        mean_lr = base.get("mean_log2_ratio") or 0.0
        base["verdict"] = (
            "显著更快到达参照终点（Holm 校正前）"
            if mean_lr < 0
            else "显著更慢到达参照终点（Holm 校正前）"
        )
    else:
        base["verdict"] = "未检出收敛速度差异（Holm 校正前）"
    if n_not_reached:
        base["verdict"] += f"；⚠ {n_not_reached} 个种子未在预算内到达参照终点"
    if n_transient:
        base["verdict"] += f"；{n_transient} 个种子的首次穿越为瞬态"
    return base


def analyze_sample_efficiency(
    curves,
    pairs: Iterable[Tuple[str, str]],
    alpha: float = ALPHA,
    min_seeds: int = MIN_SEEDS,
) -> Dict:
    """对一组对比做同种子 steps-to-target（样本效率）分析。

    ``curves`` 为 ``{variant: {seed: {step: val_loss}}}``（与
    ``budget_sensitivity`` 同一输入，可直接复用 ``curves_from_runs`` 的
    结果）；``pairs`` 为 ``(candidate, reference)``。返回可 JSON 序列化的
    ``pairs`` 列表；缺失曲线显式 ``available=false``。p 值在本分析内部
    做同指标 Holm 校正；全部为探索性证据，不改变终点/预注册判定。
    """
    alpha_f = _finite(alpha)
    if alpha_f is None or not 0.0 < alpha_f < 1.0:
        raise ValueError(f"alpha 必须在 (0,1)，收到 {alpha!r}")
    if (
        isinstance(min_seeds, bool)
        or not isinstance(min_seeds, Integral)
        or int(min_seeds) < 2
    ):
        raise ValueError(f"min_seeds 必须 >= 2，收到 {min_seeds!r}")
    min_seeds = int(min_seeds)
    curves = curves if isinstance(curves, Mapping) else {}
    results = []
    seen = set()
    for pair in pairs or []:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        candidate, reference = pair
        if not isinstance(candidate, str) or not isinstance(reference, str):
            continue
        if candidate == reference or (candidate, reference) in seen:
            continue
        seen.add((candidate, reference))
        results.append(
            _pair_analysis(curves, candidate, reference, alpha_f, min_seeds)
        )
    # 本分析内部的小 Holm 家族：同一 metric、全部可检验的配对。
    testable = [p for p in results if p.get("p") is not None]
    pads = holm_bonferroni([p.get("p") for p in testable])
    for p, padj in zip(testable, pads):
        p["p_adj"] = padj
        if p.get("inferential") and padj is not None:
            mean_lr = p.get("mean_log2_ratio") or 0.0
            if padj < alpha_f:
                p["verdict"] = (
                    "显著更快到达参照终点（Holm 校正后）"
                    if mean_lr < 0
                    else "显著更慢到达参照终点（Holm 校正后）"
                )
            else:
                p["verdict"] = "未检出收敛速度差异（Holm 校正后）"
            if p.get("n_not_reached"):
                p["verdict"] += (
                    f"；⚠ {p['n_not_reached']} 个种子未在预算内到达参照终点"
                )
            if p.get("n_transient"):
                p["verdict"] += f"；{p['n_transient']} 个种子的首次穿越为瞬态"
    available = [p for p in results if p.get("available")]
    return {
        "schema": 1,
        "analysis": "sample_efficiency",
        "metric": SAMPLE_EFFICIENCY_METRIC,
        "lower_is_better": True,
        "alpha": alpha_f,
        "min_seeds": min_seeds,
        "n_curves": sum(len(v) for v in curves.values() if isinstance(v, Mapping)),
        "n_pairs": len(results),
        "n_available": len(available),
        "n_significant": sum(
            1
            for p in testable
            if p.get("p_adj") is not None and p["p_adj"] < alpha_f
        ),
        "pairs": results,
        "note": (
            "样本效率（steps-to-target，探索性）：以参照逐种子终点 val_loss "
            "为目标，比较两侧首次到达目标的评测步。log2(步数比) 跨种子配对 "
            "t + 本分析内 Holm 校正；比值精确到 eval_every 评测粒度，"
            "首次穿越可能是瞬态（已标注）。目标是 val（选型集）而非 "
            "held-out test；不改变终点/预注册判定，也不能替代终点对比——"
            "收敛更快与终点更好是两种不同的收益。"
        ),
    }
