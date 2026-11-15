"""试点（pilot）阶段的方差锚定与确认性可行性判定。


为什么要有这一步
----------------
本项目的事前功效分析（:func:`lega.design.build_design_plan`）建立在一组
**假设方差**上：``design.ppl_paired_sd_rel_grid`` 等网格回答的是"如果跨种子
配对差的噪声是这个量级，计划里的 5 个种子够不够检出预注册的 SESOI"。它无法
回答"在我这套数据/模型/训练预算下，噪声到底有多大"，于是功效结论始终是
"如果…就…"的假设句。

试点（pilot）阶段补的正是这一环：用**与确认性种子互不相交**的试点种子跑
一遍最小可行流程，估计跨种子配对差的标准差，把该锚点写进训练前的设计计划，
使事前功效从"假设网格"升级为"试点锚点 + 假设网格"两套口径并列。

三条硬规则（防止试点变成偷看）
--------------------------------
1. **试点只能读 val**：test 上的任何读数都会把确认性终点变成"已经看过一次
   的结果"。本模块拒绝 split=test 的 PPL 产物，并把被拒条目计数留痕。
2. **试点种子与确认性种子必须不相交**：否则方差锚定与确认性推断共用同一
   批训练随机性，锚点不再独立，"用试点定噪声、用确认性下结论"就退化成同一
   批数据既当先验又当证据。
3. **试点只估计噪声，不下方向结论**：输出里没有"哪个方法更好"的字段；锚点
   只提供噪声水平与功效可行性，效应方向留给确认性阶段。

保守口径
--------
试点样本量很小（通常 2~4 个种子），点估计的 SD 本身带有很大不确定性，且
**倾向于低估**真实噪声。因此可行性判定不使用 SD 点估计，而使用其**单侧
置信上限**（卡方口径，:func:`pilot_sd_upper`），并额外给出配对差 bootstrap
的 SD 区间与 MAD 稳健 SD 供人工核对。判定结论只有三档：

* ``proceed``：按试点锚定的噪声，计划种子数在 SESOI 上已达目标功效；
* ``increase_seeds``：需要更多种子（给出所需种子数），在合理上限内可达；
* ``infeasible``：即使把种子数提到上限也达不到目标功效——此时问题不在
  样本量，而在"效应与噪声的比例"，应重新考虑 SESOI、模型规模或数据口径。

锚点产物
--------
``pilot_variance.json``（由 ``python -m lega.design.pilot`` 生成）带自身
SHA-256 摘要；设计计划把锚点内容与摘要一起写入 ``design_plan.json``，参与
``plan_digest``，因此"先跑正式实验、再回头把锚点换成对自己有利的噪声估计"
会被摘要发现。
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import random
import statistics
import sys
from numbers import Real

from ._shared import _install as _shared_install
from .power import (
    min_detectable_effect,
    paired_t_power,
    required_pairs,
)

_shared_install(globals())


PILOT_SCHEMA_VERSION = 1

PILOT_KIND = "pilot_variance_anchor"

#: 判定"加种子是否值得"时允许的最大确认性种子数。超过这个数量，成本
#: 与批次时长通常已不可接受，应回到效应/噪声本身找原因。
DEFAULT_MAX_FEASIBLE_SEEDS = 30

#: SD 单侧置信上限的置信水平（0.90 → 用 10% 下侧分位数）。
SD_UPPER_CONFIDENCE = 0.90

#: 配对差 SD 的 bootstrap 重采样次数。
DEFAULT_BOOTSTRAP = 2000

#: 试点至少需要的配对数：2 个种子只能给出极不稳定的一个自由度。
MIN_PILOT_PAIRS = 2

#: 建议的试点配对数：低于该值时 SD 锚点的警告升级。
RECOMMENDED_PILOT_PAIRS = 3

#: 各端点的试点口径。``kind`` 决定配对差的计算方式，``sesoi`` 与
#: lega.equivalence.metric_margin 的登记口径一致（相对 PPL ±1%、
#: RULER EM ±1.00 pp、AUROC/PR-AUC 差 ±0.010）。
METRIC_SPECS = {
    "PPL": {
        "artifact": "eval",
        "kind": "relative",
        "sesoi": 0.01,
        "label": "相对 PPL ±1%",
        "within_variant": False,
    },
    "RULER平均": {
        "artifact": "ruler",
        "kind": "absolute",
        "sesoi": 0.01,
        "label": "RULER 平均 EM ±1.00 pp",
        "within_variant": False,
    },
    "AUROC差(logit−仅查询)": {
        "artifact": "analysis",
        "kind": "absolute",
        "sesoi": 0.01,
        "label": "AUROC 差 ±0.010",
        "within_variant": True,
    },
    "AP差(logit−仅查询)": {
        "artifact": "analysis",
        "kind": "absolute",
        "sesoi": 0.01,
        "label": "PR-AUC 差 ±0.010",
        "within_variant": True,
    },
}

#: 可行性判定的档位（由好到差）。
FEASIBILITY_VERDICTS = ("proceed", "increase_seeds", "infeasible", "no_anchor")


def _pilot_finite(value):
    """有限浮点；bool/非数值/NaN/Inf 一律按缺失处理。"""
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


def _pilot_normal_quantile(p: float) -> float:
    """标准正态分位数（不引入 scipy 依赖）。"""
    from statistics import NormalDist

    return float(NormalDist().inv_cdf(p))


def _pilot_chi2_quantile(df: int, p: float) -> float:
    """卡方分位数（Wilson–Hilferty 近似）。

    只用于 SD 的单侧置信上限，属于保守近似：df 越小近似越粗，但方向上
    始终把置信上限推高（宁可低估功效，也不高估）。
    """
    if df <= 0:
        return None
    if not 0.0 < p < 1.0:
        return None
    z = _pilot_normal_quantile(p)
    a = 2.0 / (9.0 * df)
    base = 1.0 - a + z * math.sqrt(a)
    if base <= 0.0:
        return None
    return df * base**3


def pilot_sd_upper(sd, n: int, confidence: float = SD_UPPER_CONFIDENCE):
    """配对差标准差的单侧置信上限（卡方口径）。

    ``sigma_upper = sd * sqrt(max((n-1, 0.0)) / chi2_{1-confidence, n-1})``。试点样本
    极小时该上限非常保守（例如 n=3 时约为点估计的 2~3 倍），这正是想要的
    方向：功效规划不能建立在"试点噪声恰好很小"的乐观假设上。
    """
    sd_val = _pilot_finite(sd)
    if sd_val is None or sd_val <= 0.0:
        return None
    if not isinstance(n, int) or isinstance(n, bool) or n < 2:
        return None
    lower = _pilot_chi2_quantile(n - 1, 1.0 - float(confidence))
    if lower is None or lower <= 0.0:
        return None
    out = sd_val * math.sqrt((n - 1) / lower)
    return out if math.isfinite(out) else None


def _pilot_robust_sd(values):
    """MAD 稳健 SD（1.4826 × median(|x − median|)）。"""
    if len(values) < 2:
        return None
    med = statistics.median(values)
    mad = statistics.median([abs(v - med) for v in values])
    out = 1.4826 * mad
    return out if out > 0.0 else None


def _pilot_bootstrap_sd_ci(values, n_boot: int, seed: int):
    """配对差 SD 的 bootstrap 区间（重采样配对差本身）。"""
    n = len(values)
    if n < 2 or n_boot <= 0:
        return None
    rng = random.Random(seed)
    sds = []
    for _ in range(int(n_boot)):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        try:
            sds.append(statistics.stdev(sample))
        except statistics.StatisticsError:
            continue
    if len(sds) < 2:
        return None
    sds.sort()
    lo = sds[max(0, int(0.025 * len(sds)) - 0)]
    hi = sds[min(len(sds) - 1, int(0.975 * len(sds)))]
    return [float(lo), float(hi)]


def pilot_paired_stats(
    diffs,
    *,
    bootstrap: int = DEFAULT_BOOTSTRAP,
    seed: int = 0,
    confidence: float = SD_UPPER_CONFIDENCE,
):
    """配对差的集中趋势与噪声估计。

    返回 ``mean_diff``（方向信息，仅供人工核对，不进入功效假设）、
    ``sd``（样本标准差）、``sd_upper``（单侧置信上限，功效判定用）、
    ``sd_robust``（MAD 稳健 SD）与 ``sd_ci95``（bootstrap 区间）。
    """
    vals = []
    for d in diffs if isinstance(diffs, (list, tuple)) else []:
        f = _pilot_finite(d)
        if f is not None:
            vals.append(float(f))
    out = {
        "n": len(vals),
        "mean_diff": None,
        "sd": None,
        "sd_upper": None,
        "sd_robust": None,
        "sd_ci95": None,
        "diffs": vals,
    }
    if not vals:
        return out
    out["mean_diff"] = float(statistics.fmean(vals))
    if len(vals) < 2:
        return out
    sd = float(statistics.stdev(vals))
    out["sd"] = sd
    out["sd_upper"] = pilot_sd_upper(sd, len(vals), confidence)
    out["sd_robust"] = _pilot_robust_sd(vals)
    out["sd_ci95"] = _pilot_bootstrap_sd_ci(vals, bootstrap, seed)
    return out


def registered_pilot_contrasts(registered: dict) -> list:
    """从预注册登记表提取需要方差锚定的对照。

    只取本模块支持的端点（PPL / RULER平均 / AUROC差 / AP差）；机制类端点
    的"参照"是同一变体内的"仅查询项"基线，因此按 ``within_variant`` 标记，
    配对差在变体内计算（每个种子一个观测），不跨变体配对。
    """
    registered = registered if isinstance(registered, dict) else {}
    hyps = registered.get("hypotheses")
    out = []
    seen = set()
    for hyp in hyps if isinstance(hyps, (list, tuple)) else []:
        if not isinstance(hyp, dict):
            continue
        metric = hyp.get("metric")
        spec = METRIC_SPECS.get(metric)
        if spec is None:
            continue
        cid = str(hyp.get("id") or f"{hyp.get('candidate')}__vs__{hyp.get('reference')}")
        if cid in seen:
            continue
        seen.add(cid)
        out.append(
            {
                "id": cid,
                "metric": metric,
                "candidate": hyp.get("candidate"),
                "reference": hyp.get("reference"),
                "direction": hyp.get("direction"),
                "kind": spec["kind"],
                "sesoi": spec["sesoi"],
                "label": spec["label"],
                "within_variant": bool(spec["within_variant"]),
            }
        )
    return out


def _pilot_read_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _pilot_seed_of(path: str, payload: dict):
    """产物里的种子：优先 payload，其次文件名里的 ``_seed<N>`` 约定。"""
    raw = payload.get("seed")
    value = _pilot_finite(raw)
    if value is not None and float(value).is_integer():
        return int(value)
    stem = os.path.splitext(os.path.basename(path))[0]
    parts = stem.split("_seed")
    if len(parts) >= 2:
        tail = parts[-1].split("_")[0]
        if tail.isdigit():
            return int(tail)
    return None


def _pilot_ruler_macro_em(payload: dict):
    """RULER 平均 EM（macro）：任务等权、任务内上下文档位等权。

    口径与汇总层 ``_ruler_avg_of`` 一致（严格集合 EM，fwe 的 recall 不进
    平均）；口径不一致会把"噪声估计"建立在另一个量上。
    """
    results = payload.get("results")
    if not isinstance(results, dict):
        return None
    task_means = []
    for task, by_ctx in results.items():
        if not isinstance(task, str) or task.startswith("_"):
            continue
        if not isinstance(by_ctx, dict) or not by_ctx:
            continue
        vals = []
        for cell in by_ctx.values():
            if not isinstance(cell, dict):
                continue
            if "n" in cell and cell.get("n") is not None:
                n = _pilot_finite(cell.get("n"))
                if n is None or n <= 0:
                    continue
            em = _pilot_finite(cell.get("em"))
            if em is not None and 0.0 <= em <= 1.0:
                vals.append(em)
        if vals:
            task_means.append(float(statistics.fmean(vals)))
    if not task_means:
        return None
    return float(statistics.fmean(task_means))


def _pilot_metric_value(payload: dict, metric: str):
    """从单条产物里取该端点的观测值；返回 ``(value, reason)``。"""
    spec = METRIC_SPECS.get(metric)
    if spec is None:
        return None, "端点口径未登记"
    if metric == "PPL":
        split = str(payload.get("split") or "val")
        if split == "test":
            return None, "试点只允许 val 评测（test 读数会把确认性终点变成已看过）"
        value = _pilot_finite(payload.get("ppl"))
        if value is None:
            value = _pilot_finite(payload.get("val_ppl"))
        if value is None:
            return None, "产物缺少可用的 PPL"
        if value < 0:
            return None, "PPL 为负，产物损坏"
        return value, None
    if metric == "RULER平均":
        value = _pilot_ruler_macro_em(payload)
        if value is None:
            return None, "产物缺少可用的 RULER 平均 EM"
        return value, None
    if metric == "AUROC差(logit−仅查询)":
        a = _pilot_finite(payload.get("auroc_logit"))
        b = _pilot_finite(payload.get("auroc_qterm"))
        if a is None or b is None:
            return None, "产物缺少 auroc_logit/auroc_qterm"
        return a - b, None
    if metric == "AP差(logit−仅查询)":
        a = _pilot_finite(payload.get("auprc_logit"))
        b = _pilot_finite(payload.get("auprc_qterm"))
        if a is None or b is None:
            return None, "产物缺少 auprc_logit/auprc_qterm"
        return a - b, None
    return None, "端点口径未登记"


def _pilot_collect_series(results_dir: str, metric: str, seeds=None):
    """扫描结果目录里与该端点有关的产物，返回 ``{variant: {seed: value}}``。

    同时返回被拒条目（test split、坏产物、种子不在试点集合内），供锚点
    留痕：拒绝原因必须可见，否则"少算了几条"与"协议不允许算"会混为一谈。
    """
    spec = METRIC_SPECS.get(metric)
    series = {}
    rejected = []
    if spec is None:
        return series, rejected
    prefix = spec["artifact"]
    paths = sorted(glob.glob(os.path.join(results_dir, f"{prefix}_*.json")))
    for path in paths:
        payload = _pilot_read_json(path)
        if payload is None:
            rejected.append({"path": os.path.basename(path), "reason": "产物不可读"})
            continue
        variant = payload.get("variant")
        if not isinstance(variant, str) or not variant:
            rejected.append({"path": os.path.basename(path), "reason": "产物缺少变体名"})
            continue
        seed = _pilot_seed_of(path, payload)
        if seed is None:
            rejected.append({"path": os.path.basename(path), "reason": "产物缺少种子信息"})
            continue
        if seeds is not None and seed not in seeds:
            rejected.append(
                {"path": os.path.basename(path), "reason": "种子不在试点集合内"}
            )
            continue
        value, reason = _pilot_metric_value(payload, metric)
        if value is None:
            rejected.append({"path": os.path.basename(path), "reason": reason})
            continue
        series.setdefault(variant, {})[seed] = value
    return series, rejected


def _pilot_relative_diff(cand, ref):
    if ref is None or ref <= 0.0:
        return None
    return (cand - ref) / ref


def _pilot_contrast_diffs(contrast: dict, series: dict):
    """按对照口径取配对差（候选 − 参照，方向与登记一致）。"""
    metric = contrast["metric"]
    kind = contrast["kind"]
    candidate = contrast.get("candidate")
    reference = contrast.get("reference")
    if contrast.get("within_variant"):
        values = series.get(candidate) or {}
        return [(seed, val) for seed, val in sorted(values.items())]
    cand_values = series.get(candidate) or {}
    ref_values = series.get(reference) or {}
    out = []
    for seed in sorted(set(cand_values) & set(ref_values)):
        cand = cand_values[seed]
        ref = ref_values[seed]
        if kind == "relative":
            diff = _pilot_relative_diff(cand, ref)
        else:
            diff = cand - ref
        if diff is not None:
            out.append((seed, diff))
    return out


def pilot_anchor_digest(anchor: dict) -> str:
    """锚点内容摘要（排除时间戳等非科学字段）。"""
    payload = {
        k: v
        for k, v in anchor.items()
        if k not in ("created_at", "anchor_digest", "source")
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def build_pilot_variance(
    results_dir: str,
    *,
    registered: dict = None,
    pilot_seeds=None,
    confirmatory_seeds=None,
    contrasts=None,
    bootstrap: int = DEFAULT_BOOTSTRAP,
    seed: int = 0,
    require_disjoint: bool = True,
) -> dict:
    """从试点结果目录估计配对差噪声，产出方差锚点。

    ``pilot_seeds`` 为试点使用的种子集合；``confirmatory_seeds`` 为确认性
    种子集合。两者有交集时（``require_disjoint=True``）锚点判 ``fail``：
    锚点与确认性推断共用同一批训练随机性时，"用试点定噪声"不再是独立信息。
    """
    problems = []
    warnings = []
    results_dir = os.path.abspath(str(results_dir))
    if not os.path.isdir(results_dir):
        anchor = {
            "schema_version": PILOT_SCHEMA_VERSION,
            "kind": PILOT_KIND,
            "created_at": _pilot_now(),
            "status": "fail",
            "problems": [f"试点结果目录不存在：{results_dir}"],
            "warnings": [],
            "source": {"results_dir": results_dir},
            "contrasts": {},
        }
        anchor["anchor_digest"] = pilot_anchor_digest(anchor)
        return anchor

    pilot_seed_list = None
    if pilot_seeds is not None:
        pilot_seed_list = sorted({int(s) for s in pilot_seeds})
    confirmatory_seed_list = (
        sorted({int(s) for s in confirmatory_seeds}) if confirmatory_seeds else None
    )
    overlap = []
    if pilot_seed_list is not None and confirmatory_seed_list:
        overlap = sorted(set(pilot_seed_list) & set(confirmatory_seed_list))
        if overlap and require_disjoint:
            problems.append(
                f"试点种子与确认性种子重叠 {overlap}：方差锚点不再独立于"
                "确认性推断，锚点不可用"
            )
        elif overlap:
            warnings.append(
                f"试点种子与确认性种子重叠 {overlap}：锚点独立性受损，"
                "结论按探索性对待"
            )
    if pilot_seed_list is None:
        warnings.append(
            "未声明试点种子集合：无法核对试点与确认性种子的隔离，"
            "锚点仅作参考"
        )

    contrast_list = contrasts
    if contrast_list is None:
        contrast_list = registered_pilot_contrasts(registered)
    if not contrast_list:
        problems.append("没有可用的对照（登记表为空或端点口径未登记）")

    series_cache = {}
    contrasts_out = {}
    for contrast in contrast_list:
        metric = contrast["metric"]
        if metric not in series_cache:
            series_cache[metric] = _pilot_collect_series(
                results_dir, metric, pilot_seed_list
            )
        series, rejected = series_cache[metric]
        pairs = _pilot_contrast_diffs(contrast, series)
        diffs = [d for _seed, d in pairs]
        stats = pilot_paired_stats(diffs, bootstrap=bootstrap, seed=seed)
        entry = {
            "id": contrast["id"],
            "metric": metric,
            "candidate": contrast.get("candidate"),
            "reference": contrast.get("reference"),
            "kind": contrast["kind"],
            "sesoi": contrast["sesoi"],
            "label": contrast.get("label"),
            "within_variant": bool(contrast.get("within_variant")),
            "seeds": [s for s, _d in pairs],
            "n_pairs": len(pairs),
            "n_rejected_artifacts": len(rejected),
            "rejected_examples": rejected[:5],
            "stats": stats,
            "sd": stats.get("sd"),
            "sd_upper": stats.get("sd_upper"),
            "mean_diff": stats.get("mean_diff"),
        }
        if stats["n"] < MIN_PILOT_PAIRS:
            entry["usable"] = False
            entry["note"] = (
                f"可用配对数 {stats['n']} < {MIN_PILOT_PAIRS}：无法估计噪声，"
                "该端点没有方差锚点"
            )
            warnings.append(
                f"{contrast['id']}（{metric}）可用配对数只有 {stats['n']}："
                "噪声锚点不可用"
            )
        elif not stats.get("sd"):
            entry["usable"] = False
            entry["note"] = (
                "配对差在试点里没有任何波动（SD=0）：通常意味着试点设置退化或"
                "读数被复制，不能据此宣称功效充足；该端点按无锚点处理"
            )
            warnings.append(
                f"{contrast['id']}（{metric}）试点配对差无波动：锚点不可用"
            )
        else:
            entry["usable"] = True
            if stats["n"] < RECOMMENDED_PILOT_PAIRS:
                entry["note"] = (
                    f"试点配对数 {stats['n']} 少于建议的 {RECOMMENDED_PILOT_PAIRS}："
                    "SD 置信上限非常保守，可行性判定偏保守"
                )
                warnings.append(
                    f"{contrast['id']}（{metric}）试点配对数只有 {stats['n']}："
                    "锚点偏保守"
                )
            else:
                entry["note"] = None
        contrasts_out[contrast["id"]] = entry

    usable = [c for c in contrasts_out.values() if c.get("usable")]
    if not usable and not problems:
        problems.append("所有端点都没有足够的试点配对，无法给出方差锚点")

    status = "fail" if problems else ("warn" if warnings else "ok")
    anchor = {
        "schema_version": PILOT_SCHEMA_VERSION,
        "kind": PILOT_KIND,
        "created_at": _pilot_now(),
        "status": status,
        "problems": problems,
        "warnings": warnings,
        "source": {
            "results_dir": results_dir,
            "pilot_seeds": pilot_seed_list,
            "confirmatory_seeds": confirmatory_seed_list,
            "seed_overlap": overlap,
            "bootstrap": int(bootstrap),
            "bootstrap_seed": int(seed),
        },
        "contrasts": contrasts_out,
    }
    anchor["anchor_digest"] = pilot_anchor_digest(anchor)
    return anchor


def _pilot_now() -> str:
    import time

    return time.strftime("%Y-%m-%d %H:%M:%S")


def pilot_feasibility(
    anchor: dict,
    *,
    n_seeds: int,
    alpha: float = 0.05,
    power: float = 0.8,
    max_seeds: int = DEFAULT_MAX_FEASIBLE_SEEDS,
    alpha_by_metric: dict = None,
):
    """用锚点噪声判断计划种子数是否够检出登记的 SESOI。

    判定只用**噪声**：SESOI 视为"真实差异恰好等于最小重要差异"这一最不利
    情形；锚点里的 ``mean_diff``（试点观测到的方向）不参与功效假设，否则
    试点会变成"先看一眼效果再决定跑多少种子"。

    同时给出两套口径：

    * 保守口径（``verdict``）：用 SD 的单侧置信上限 ``sd_upper``。试点样本
      极少时它可能比点估计大数倍，判定因此偏保守——这是想要的方向。
    * 点估计口径（``verdict_point``）：用 SD 点估计，给出"如果试点噪声
      恰好就是真实噪声"的情形。

    两个口径都摆在读者面前，研究者才能判断"要加种子"还是"锚点本身太粗"。
    """
    anchor = anchor if isinstance(anchor, dict) else {}
    contrasts = anchor.get("contrasts")
    rows = []
    alpha_by_metric = alpha_by_metric if isinstance(alpha_by_metric, dict) else {}

    def _verdict_of(margin, sd, n, alpha_eff):
        mde = min_detectable_effect(sd, n, alpha_eff, float(power))
        pw = paired_t_power(margin, sd, n, alpha_eff)
        req = required_pairs(
            margin,
            sd,
            alpha_eff,
            float(power),
            max_n=max(int(max_seeds) * 10, 1000),
        )
        if pw is not None and pw >= float(power):
            verdict = "proceed"
        elif req is not None and req <= int(max_seeds):
            verdict = "increase_seeds"
        else:
            verdict = "infeasible"
        return mde, pw, req, verdict

    for cid, entry in (contrasts if isinstance(contrasts, dict) else {}).items():
        if not isinstance(entry, dict):
            continue
        metric = entry.get("metric")
        margin = _pilot_finite(entry.get("sesoi"))
        sd_point = _pilot_finite(entry.get("sd"))
        sd_upper = _pilot_finite(entry.get("sd_upper"))
        sd_used = sd_upper if sd_upper is not None else sd_point
        alpha_eff = _pilot_finite(alpha_by_metric.get(metric))
        if alpha_eff is None:
            alpha_eff = float(alpha)
        n_pairs = entry.get("n_pairs")
        row = {
            "id": cid,
            "metric": metric,
            "sesoi": margin,
            "sd_point": sd_point,
            "sd_upper": sd_upper,
            "sd_used": sd_used,
            "sd_source": "sd_upper" if sd_upper is not None else "sd",
            "alpha_effective": alpha_eff,
            "n_seeds_planned": int(n_seeds),
            "n_pilot_pairs": n_pairs,
            "mde_at_planned": None,
            "power_at_sesoi": None,
            "required_seeds": None,
            "verdict": "no_anchor",
            "mde_at_planned_point": None,
            "power_at_sesoi_point": None,
            "required_seeds_point": None,
            "verdict_point": "no_anchor",
        }
        if margin is None or margin <= 0.0 or sd_used is None or sd_used <= 0.0:
            row["note"] = "缺少可用的方差锚点或 SESOI，无法判定"
            rows.append(row)
            continue
        (
            row["mde_at_planned"],
            row["power_at_sesoi"],
            row["required_seeds"],
            row["verdict"],
        ) = _verdict_of(margin, sd_used, int(n_seeds), alpha_eff)
        if sd_point is not None and sd_point > 0.0:
            (
                row["mde_at_planned_point"],
                row["power_at_sesoi_point"],
                row["required_seeds_point"],
                row["verdict_point"],
            ) = _verdict_of(margin, sd_point, int(n_seeds), alpha_eff)
        notes = []
        if isinstance(n_pairs, int) and n_pairs < RECOMMENDED_PILOT_PAIRS:
            notes.append(
                f"试点配对数只有 {n_pairs}，SD 置信上限非常保守，"
                "保守口径的判定会偏悲观"
            )
        if row["verdict"] != row["verdict_point"]:
            notes.append(
                f"点估计口径的判定是 {row['verdict_point']}，保守口径是 "
                f"{row['verdict']}：两者不一致说明锚点自身的不确定性就是"
                "结论的主要来源，应当增加试点种子而不是直接下结论"
            )
        row["note"] = "；".join(notes) if notes else None
        rows.append(row)

    verdict_rank = {"proceed": 0, "increase_seeds": 1, "infeasible": 2}
    worst = None
    worst_point = None
    n_anchored = 0
    for row in rows:
        rank = verdict_rank.get(row.get("verdict"))
        if rank is None:
            continue
        n_anchored += 1
        if worst is None or rank > verdict_rank[worst]:
            worst = row.get("verdict")
        rank_point = verdict_rank.get(row.get("verdict_point"))
        if rank_point is None:
            continue
        if worst_point is None or rank_point > verdict_rank[worst_point]:
            worst_point = row.get("verdict_point")
    if worst is None:
        worst = "no_anchor"
    if worst_point is None:
        worst_point = "no_anchor"
    return {
        "n_seeds_planned": int(n_seeds),
        "alpha": float(alpha),
        "power": float(power),
        "max_feasible_seeds": int(max_seeds),
        "verdict": worst,
        "verdict_point": worst_point,
        "n_endpoints_with_anchor": n_anchored,
        "n_endpoints_without_anchor": len(rows) - n_anchored,
        "rows": rows,
        "note": (
            "判定基于试点锚定的噪声与登记 SESOI；不依赖试点观测到的效应"
            "方向。verdict 用 SD 单侧置信上限（保守），verdict_point 用 SD "
            "点估计。若 verdict=infeasible，问题在效应/噪声比例而非样本量。"
            "没有锚点的端点不参与整体判定，但它们的先验功效仍只有假设噪声口径。"
        ),
    }


def load_pilot_anchor(path: str):
    """读取锚点文件；缺失/损坏返回 ``None``（调用方按"没有锚点"处理）。"""
    if not path:
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    if data.get("kind") != PILOT_KIND:
        return None
    return data


def _pilot_cli_contrasts(registered: dict):
    return registered_pilot_contrasts(registered)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="从试点结果估计跨种子配对差噪声，产出方差锚点"
    )
    p.add_argument("--results_dir", required=True, help="试点结果目录")
    p.add_argument("--config", default=None, help="配置文件（读确认性/试点种子）")
    p.add_argument(
        "--registered",
        default=None,
        help="预注册登记表路径（默认 preregistration/hypotheses.json）",
    )
    p.add_argument("--pilot_seeds", default=None, help="逗号分隔的试点种子")
    p.add_argument("--confirmatory_seeds", default=None, help="逗号分隔的确认性种子")
    p.add_argument("--out", default="pilot_variance.json")
    p.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    cfg = {}
    if args.config:
        try:
            with open(args.config, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except (OSError, ValueError) as e:
            print(f"[pilot] 读取配置失败：{e}", file=sys.stderr)
            return 2
        if not isinstance(cfg, dict):
            cfg = {}

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

    design_cfg = cfg.get("design") if isinstance(cfg.get("design"), dict) else {}

    def _seed_list(text):
        if text is None:
            return None
        out = []
        for part in str(text).split(","):
            part = part.strip()
            if not part:
                continue
            if not part.isdigit():
                print(f"[pilot] 非法种子 {part!r}", file=sys.stderr)
                return None
            out.append(int(part))
        return out or None

    pilot_seeds = _seed_list(args.pilot_seeds)
    if pilot_seeds is None:
        pilot_seeds = design_cfg.get("pilot_seeds")
    confirmatory_seeds = _seed_list(args.confirmatory_seeds)
    if confirmatory_seeds is None:
        raw = cfg.get("seeds")
        if raw is None:
            raw = [cfg.get("seed", 1337)]
        if isinstance(raw, (list, tuple)):
            confirmatory_seeds = [int(s) for s in raw if isinstance(s, int)]

    anchor = build_pilot_variance(
        args.results_dir,
        registered=registered,
        pilot_seeds=pilot_seeds,
        confirmatory_seeds=confirmatory_seeds,
        bootstrap=args.bootstrap,
        seed=args.seed,
    )
    try:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(anchor, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.write("\n")
    except OSError as e:
        print(f"[pilot] 写入失败：{e}", file=sys.stderr)
        return 2
    if not args.quiet:
        print(
            f"[pilot] 锚点 status={anchor['status']} "
            f"digest={str(anchor.get('anchor_digest'))[:12]} -> {args.out}"
        )
        for cid, entry in anchor.get("contrasts", {}).items():
            stats = entry.get("stats") or {}
            print(
                f"[pilot] {cid}（{entry.get('metric')}）：n={stats.get('n')} "
                f"sd={stats.get('sd')} sd_upper={stats.get('sd_upper')}"
            )
        for problem in anchor.get("problems", []):
            print(f"[pilot] fail: {problem}")
        for warning in anchor.get("warnings", []):
            print(f"[pilot] warn: {warning}")
    return 2 if anchor.get("status") == "fail" else 0


if __name__ == "__main__":
    from lega.bootstrap import ensure_hash_seed

    ensure_hash_seed()
    raise SystemExit(main())
