"""跨文档 vs 文档内窗口的效应分层（把“拼接伪影”从机制收益里分离并检验交互）。

背景
====
本项目在**顺序拼接**的 token 流上滑窗评测：同一批评测窗口里，一部分完全
落在单篇文档内，另一部分跨过文档边界。跨文档窗口正是数据拼接本身注入的
“无关上下文”，而 LEGA 的机制主张是“找不到可靠信息时少写无关内容”。只报
整份 PPL 的候选−参照差无法回答一个关键问题：

    候选的收益是否只出现在跨文档窗口（拼接伪影），还是文档内窗口同样
    成立（更广的语言建模改善）？

``lega.evaluate`` 为每个评测窗口落盘“是否跨文档边界”标志
（``window_crossing_documents``，与 ``window_losses``/``window_indices``
同序同长）。本模块据此做**同一批窗口内的分层**：

* 跨文档窗口：候选−参照的平均 ΔNLL；
* 文档内窗口：候选−参照的平均 ΔNLL；
* 交互 = 跨文档 ΔNLL − 文档内 ΔNLL。

每个种子得到一个交互值，跨种子做同种子配对 t 检验 + 独立 Holm；交互显著
为负说明收益显著集中在跨文档窗口，显著为正说明候选在跨文档窗口相对更差，
不显著只能写“未检出分层差异”。这与 ``doc_aligned_eval``（另一套“文档内”
窗口网格）互补：后者是**不同窗口集合**的绝对值，本层是**同一窗口集合**上
的分层，因此可以直接做交互。

边界
====
* 两层窗口共享同一批种子、同一个 checkpoint，不是独立研究；p 值只描述
  “这两类窗口上的效应差”，不构成新的因果识别；
* 文档内窗口依赖文档边界切分协议，文档太短时可能没有文档内窗口，此时该
  对比显式不可用，不按 0 补齐，也不把缺失伪造为“无分层差异”；
* 该分析为**探索性**、未预注册，不进入任何确认性判定；只有交互检验进入
  探索性 FDR 家族，它也不能替代逐窗口配对的整体稳健性分析。
"""

from __future__ import annotations

import math
from numbers import Integral, Real
from typing import List, Mapping, Optional, Tuple

import numpy as np

from .utils import holm_bonferroni, paired_t_test
# NOTE: Module-level seed setting removed to prevent silent RNG state
# reset on import. Use the centralized seed function at the experiment
# entry point instead (e.g., repro.set_all_seeds / seed_everything).

#: 分层交互检验的指标名：跨文档窗口的 ΔNLL 减去文档内窗口的 ΔNLL。
DOC_STRATA_INTERACTION_METRIC = "跨文档−文档内ΔNLL交互"

#: 两个分层各自的 ΔNLL 指标名（描述性，不进入 FDR 主族）。
DOC_STRATA_CROSS_METRIC = "跨文档窗口ΔNLL"

DOC_STRATA_ALIGNED_METRIC = "文档内窗口ΔNLL"

DOC_STRATA_NOTE = (
    "同一批评测窗口按“是否跨文档边界”分层：交互=跨文档ΔNLL−文档内ΔNLL。"
    "交互显著为负=收益集中在跨文档窗口（需警惕拼接伪影/无关上下文），"
    "显著为正=候选在跨文档窗口相对更差，不显著=未检出分层差异。"
    "探索性证据，不改变预注册判定；两层共享同一批种子与 checkpoint，"
    "不是独立研究。"
)


def _finite(value) -> Optional[float]:
    """JSON 数值 → 有限 float；bool/非数值/NaN/Inf 一律 None。"""
    if value is None or isinstance(value, bool):
        return None
    if not isinstance(value, Real):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return out if math.isfinite(out) else None


def _int_key(value) -> Optional[int]:
    """窗口序号/起点 → 非负整数；bool/小数/字符串/负数 None。"""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, Integral):
        out = int(value)
    elif isinstance(value, float) and math.isfinite(value) and value.is_integer():
        out = int(value)
    else:
        return None
    return out if out >= 0 else None


def _as_dict(value) -> dict:
    return value if isinstance(value, dict) else {}


def _seed_map(runs: Mapping, variant: str) -> dict:
    """取某变体的 {seed: slot}；有带种子的结果时忽略无种子文件。"""
    m = runs.get(variant) if isinstance(runs, Mapping) else None
    m = m if isinstance(m, Mapping) else {}
    tagged = {s: slot for s, slot in m.items() if s is not None}
    return tagged if tagged else dict(m)


def _window_records(payload) -> Optional[List[dict]]:
    """eval payload → [{key, loss, cross}]；缺字段/不一致返回 None。

    只接受逐窗口损失、窗口键与跨文档标志三者一一对应的记录。任一缺失、
    长度不等、含非法值都返回 None，由调用方按“不可用”处理，绝不静默
    按位置强配或补齐。
    """
    d = _as_dict(payload)
    losses = d.get("window_losses")
    if not isinstance(losses, list) or not losses:
        return None
    keys_raw = None
    if isinstance(d.get("window_indices"), list) and len(d["window_indices"]) == len(
        losses
    ):
        keys_raw = d["window_indices"]
    elif isinstance(d.get("window_starts"), list) and len(d["window_starts"]) == len(
        losses
    ):
        keys_raw = d["window_starts"]
    if keys_raw is None:
        return None
    cross = d.get("window_crossing_documents")
    if not isinstance(cross, list) or len(cross) != len(losses):
        return None
    records = []
    seen = set()
    for loss, raw_key, flag in zip(losses, keys_raw, cross):
        value = _finite(loss)
        key = _int_key(raw_key)
        if value is None or value < 0.0 or key is None or key in seen:
            return None
        if not isinstance(flag, bool):
            return None
        seen.add(key)
        records.append({"key": key, "loss": value, "cross": flag})
    return records


def _protocol_pair(ea: dict, eb: dict) -> Optional[Tuple[dict, dict]]:
    """两侧评测协议可比时返回 (records_a, records_b)，否则 None。"""
    ra = _window_records(ea)
    rb = _window_records(eb)
    if ra is None or rb is None:
        return None
    # 同一批评测窗口：split / eval_ctx 一致，且窗口键集合完全相同；否则
    # 不是同一窗口网格上的分层，强行配对会引入协议差异。
    for key in ("split", "eval_ctx"):
        if ea.get(key, "val" if key == "split" else None) != eb.get(
            key, "val" if key == "split" else None
        ):
            return None
    by_a = {r["key"]: r for r in ra}
    by_b = {r["key"]: r for r in rb}
    if set(by_a) != set(by_b):
        return None
    # 同一 key 的跨文档标志必须一致（同一数据与 ctx 下应当如此）；不一致
    # 说明窗口定义漂移，整对拒绝而不是挑一侧采信。
    for key, rec in by_a.items():
        if rec["cross"] != by_b[key]["cross"]:
            return None
    return ra, rb


def _seed_strata(ea: dict, eb: dict) -> Optional[dict]:
    """单个种子的分层配对差；协议不可比/缺文档内或跨文档窗口返回 None。"""
    paired = _protocol_pair(ea, eb)
    if paired is None:
        return None
    ra, rb = paired
    by_b = {r["key"]: r["loss"] for r in rb}
    cross_diffs, aligned_diffs = [], []
    for rec in ra:
        diff = rec["loss"] - by_b[rec["key"]]
        if rec["cross"]:
            cross_diffs.append(diff)
        else:
            aligned_diffs.append(diff)
    if not cross_diffs or not aligned_diffs:
        return None
    cross_mean = float(np.mean(cross_diffs))
    aligned_mean = float(np.mean(aligned_diffs))
    return {
        "n_cross": len(cross_diffs),
        "n_aligned": len(aligned_diffs),
        "cross_mean_diff": cross_mean,
        "aligned_mean_diff": aligned_mean,
        "interaction": cross_mean - aligned_mean,
    }


def _test_block(values, metric, lower_is_better=True) -> dict:
    """对一组同种子配对标量做单样本配对 t 检验（H0：均值=0）。"""
    vals = [v for v in values if _finite(v) is not None]
    if not vals:
        return {"n": 0, "mean_diff": None, "std_diff": None, "p": None}
    res = paired_t_test(vals, [0.0] * len(vals))
    t = res.get("t")
    d_z = res.get("d_z")
    return {
        "metric": metric,
        "lower_is_better": bool(lower_is_better),
        "n": res.get("n"),
        "mean_diff": res.get("mean_diff"),
        "std_diff": res.get("std_diff"),
        # t/df 在零方差时是 ±inf，写进 JSON 会破坏严格 JSON 解析；如实转为
        # null（p 仍由 paired_t_test 给出：零方差且均值非 0 → p=0）。
        "t": t if t is not None and math.isfinite(t) else None,
        "p": res.get("p"),
        "d_z": d_z if d_z is not None and math.isfinite(d_z) else None,
        "n_seeds": len(vals),
    }


def _pair_result(
    candidate: str, reference: str, runs: Mapping
) -> dict:
    base = {
        "candidate": candidate,
        "reference": reference,
        "metric": DOC_STRATA_INTERACTION_METRIC,
        "lower_is_better": True,
        "exploratory": True,
        "available": False,
        "reason": "",
    }
    c_map = _seed_map(runs, candidate)
    r_map = _seed_map(runs, reference)
    if not c_map:
        base["reason"] = "候选变体缺少评测产物"
        return base
    if not r_map:
        base["reason"] = "参照变体缺少评测产物"
        return base
    common = sorted(
        (s for s in c_map if s in r_map), key=lambda s: (s is None, str(s))
    )
    if not common:
        base["reason"] = "两侧没有共同种子"
        return base
    per_seed = []
    skipped = 0
    for s in common:
        sa = _as_dict(c_map.get(s))
        sb = _as_dict(r_map.get(s))
        strata = _seed_strata(_as_dict(sa.get("eval")), _as_dict(sb.get("eval")))
        if strata is None:
            skipped += 1
            continue
        strata["seed"] = s if isinstance(s, (int, str)) and not isinstance(s, bool) else None
        per_seed.append(strata)
    if not per_seed:
        base["reason"] = (
            "没有可用种子：缺 window_crossing_documents（旧产物/无文档侧车）、"
            "两侧窗口不一致，或某一层没有窗口"
        )
        if skipped:
            base["excluded_seeds"] = skipped
        return base
    cross_vals = [r["cross_mean_diff"] for r in per_seed]
    aligned_vals = [r["aligned_mean_diff"] for r in per_seed]
    inter_vals = [r["interaction"] for r in per_seed]
    cross_block = _test_block(cross_vals, DOC_STRATA_CROSS_METRIC)
    aligned_block = _test_block(aligned_vals, DOC_STRATA_ALIGNED_METRIC)
    inter_block = _test_block(inter_vals, DOC_STRATA_INTERACTION_METRIC)
    base.update(
        {
            "available": True,
            "reason": "",
            "n": len(per_seed),
            "n_seeds": len(per_seed),
            "n_cross_min": min(r["n_cross"] for r in per_seed),
            "n_cross_max": max(r["n_cross"] for r in per_seed),
            "n_aligned_min": min(r["n_aligned"] for r in per_seed),
            "n_aligned_max": max(r["n_aligned"] for r in per_seed),
            "per_seed": per_seed,
            "cross": cross_block,
            "aligned": aligned_block,
            "interaction": inter_block,
        }
    )
    if skipped:
        base["excluded_seeds"] = skipped
    return base


def _verdict(inter: dict, p_adj, alpha: float, min_seeds: int) -> str:
    mean_diff = _finite(inter.get("mean_diff"))
    n = inter.get("n")
    if mean_diff is None or not isinstance(n, int) or n < 1:
        return "无法检验"
    if n < min_seeds:
        return f"证据不足（种子 n={n}<{min_seeds}），仅报点估计"
    if p_adj is None:
        return "无法检验（方差退化或指标缺失）"
    if p_adj < alpha:
        if mean_diff < 0:
            return "分层显著：收益集中在跨文档窗口（交互为负，需警惕拼接伪影）"
        return "分层显著：候选在跨文档窗口相对更差（交互为正）"
    return "未检出跨文档/文档内差异（Holm 校正后不显著）"


def analyze_doc_strata(
    runs: Mapping,
    pairs,
    alpha: float = 0.05,
    min_seeds: int = 3,
) -> dict:
    """跨文档 vs 文档内窗口的分层分解与交互检验（探索性）。

    ``runs`` 为 ``{variant: {seed: {"eval": eval_payload, ...}}}``；``pairs``
    为 ``(candidate, reference)`` 列表。返回每对的分层均值、交互检验与一个
    可直接进入探索性 FDR 家族的 ``interaction_tests`` 列表。缺失/旧产物
    显式 ``available=False``，不伪造结论。
    """
    runs = runs if isinstance(runs, Mapping) else {}
    pairs = pairs if isinstance(pairs, (list, tuple)) else []
    out_pairs = []
    for pair in pairs:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        cand, ref = pair
        if not isinstance(cand, str) or not isinstance(ref, str):
            continue
        out_pairs.append(_pair_result(cand, ref, runs))

    available = [p for p in out_pairs if p.get("available")]
    # 交互检验单独一族 Holm（每个 pair 一条）——它才是“收益是否分层”的
    # 主检验，也是进入探索性 FDR 的检验。
    pads = holm_bonferroni(
        [(p.get("interaction") or {}).get("p") for p in available]
    )
    interaction_tests = []
    for pair, p_adj in zip(available, pads):
        inter = pair.get("interaction") or {}
        pair["p_adj"] = p_adj
        verdict = _verdict(inter, p_adj, float(alpha), int(min_seeds))
        inter["p_adj"] = p_adj
        inter["verdict"] = verdict
        pair["verdict"] = verdict
        interaction_tests.append(
            {
                "candidate": pair.get("candidate"),
                "reference": pair.get("reference"),
                "metric": DOC_STRATA_INTERACTION_METRIC,
                "lower_is_better": True,
                "n": inter.get("n"),
                "mean_diff": inter.get("mean_diff"),
                "std_diff": inter.get("std_diff"),
                "p": inter.get("p"),
                "p_adj": p_adj,
                "within_noise": None,
                "below_mde": None,
                "verdict": verdict,
                "exploratory": True,
            }
        )
    return {
        "available": bool(available),
        "metric": DOC_STRATA_INTERACTION_METRIC,
        "alpha": float(alpha),
        "min_seeds": int(min_seeds),
        "n_pairs": len(out_pairs),
        "n_available": len(available),
        "pairs": out_pairs,
        "interaction_tests": interaction_tests,
        "note": DOC_STRATA_NOTE,
    }
