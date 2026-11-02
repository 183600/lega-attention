"""多方案排名稳定性与选择校正（selection-adjusted）推断。

现有配对检验逐一回答“候选 vs 某个参照”，但没有回答多方案结果分析里的
两个不同问题：

1. **选择偏差 / winner's curse**：同时训练 10 个变体时，报告“最优变体”
   本身是从多个相关、带噪的方案中挑出的最大值。即使所有方案在总体中
   完全相同，有限种子下也几乎总有一个“最好”的；只把它的均值与第二名
   做普通两两比较，会低估“这次挑的就是最大值”带来的乐观偏差。
2. **排名稳定性**：换一批训练种子，名次是否还一样？若第一名只在少数
   种子上领先，或与第二名大量并列，就不应声称存在单一最优方案。

本模块因此把**每个种子内的方案排序**作为分析单位（秩统计量对单调变换
和离群值稳健，且 PPL 这类比值尺度不需要先取 log）：

* 逐种子名次（1 = 最好，并列取平均名次）；
* 平均名次、名次分布、逐种子“最优”次数；
* 对种子做 bootstrap（重采样种子后重新选一次最优），给出每个方案的
  成为最优概率与平均名次的区间——描述性，不假设正态；
* 在“方案标签在每个种子内可交换”的全局零假设下做置换检验。统计量是
  平均名次相对均匀名次的最大改善（以及整体名次跨度），从而把“从 V 个
  方案里挑最好”的选择效应放进零分布，而不是拿被挑中的方案与固定参照
  做普通配对检验。

边界（必须如实声明）
====================
* 这是**探索性/描述性**结果分析，未写入预注册锁链，不改变任何 Holm/
  FWER、TOST、Bayes 或预注册的支持/拒绝/未定判定。
* 置换检验的零假设是“方案标签在种子内可交换”（强零假设）。变体之间
  共享种子、数据与评测协议，因此按种子内打乱标签保留了种子效应与方案
  间的相关结构；但若不同方案的缺失机制不同、或训练预算/协议不可比，
  可交换性不成立，p 值只作线索。
* bootstrap 只重采样训练种子，不重采样评测窗口/样例；它回答“换种子后
  排名是否稳定”，不覆盖评测抽样不确定性（后者见窗口/样例级 bootstrap）。
* p 值是 Monte-Carlo 置换 p（含观测样本的 +1 校正），分辨率约为
  1/(n_permutations+1)；n<3 个共同种子或方案数<2 时不下判定、不伪造 p。
* 名次之间不是等距量表，平均名次差不等于性能差；排名分析不能替代效应
  量、SESOI/等价性、MDE 或任何成对检验。所有方案在两两比较中都不显著
  时，仍可能因“相对排序一致”而出现很小的置换 p——反之亦然，必须并读。
"""

from __future__ import annotations

import math
import random
from numbers import Real
from typing import Dict, List, Mapping, Sequence

DEFAULT_ALPHA = 0.05
MIN_SEEDS = 3
DEFAULT_BOOTSTRAP = 2000
DEFAULT_PERMUTATIONS = 20000

STATUS_CN = {
    "ok": "排名可分析",
    "insufficient_seeds": "共同种子不足，无法评估排名稳定性/选择偏差",
    "insufficient_variants": "方案数不足 2，无法排名",
    "no_common_seeds": "方案之间没有共同种子，无法逐种子配对排名",
    "invalid": "输入无效，无法排名",
}


def _finite(value):
    """有限实数观测；bool/字符串/非有限/不可表示值按缺失处理。"""
    if value is None or isinstance(value, bool):
        return None
    if type(value).__name__ == "bool_":
        return None
    if isinstance(value, str) or not isinstance(value, Real):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return out if math.isfinite(out) else None


def _strict_int(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} 必须是整数，收到 {value!r}")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} 必须 >= {minimum}，收到 {value}")
    return value


def _strict_alpha(value):
    out = _finite(value)
    if out is None or not 0.0 < out < 1.0:
        raise ValueError(f"alpha 必须在 (0, 1) 内，收到 {value!r}")
    return out


def _seed_sort_key(seed):
    return (type(seed).__name__, str(seed))


def _normalise(scores, variants=None, lower_is_better=False):
    """把 {variant: {seed: value}} 规范成共同的 (variants, seeds, matrix)。

    matrix 按种子排列，每个元素是该种子下各方案“越大越好”的定向分数。
    只保留所有方案都存在的种子，并在返回值里显式给出被丢弃的种子——不能
    让某个方案多一个种子就与其他方案错位配对。
    """
    if not isinstance(scores, Mapping):
        raise ValueError(
            f"scores 必须是 variant->seed->value 的映射，收到 "
            f"{type(scores).__name__}"
        )
    if not isinstance(lower_is_better, bool):
        raise ValueError(f"lower_is_better 必须是 bool，收到 {lower_is_better!r}")
    if variants is None:
        variants = list(scores.keys())
    else:
        try:
            variants = list(variants)
        except TypeError as exc:
            raise ValueError("variants 必须是可迭代的方案名序列") from exc
    cleaned: List[str] = []
    seen = set()
    for variant in variants:
        if not isinstance(variant, str) or not variant.strip():
            raise ValueError(f"方案名必须是非空字符串，收到 {variant!r}")
        if variant in seen:
            raise ValueError(f"方案名重复：{variant!r}")
        seen.add(variant)
        cleaned.append(variant)
    variants = cleaned

    per_variant: Dict[str, Dict[object, float]] = {}
    union_seeds = set()
    for variant in variants:
        raw = scores.get(variant)
        if not isinstance(raw, Mapping):
            per_variant[variant] = {}
            continue
        values: Dict[object, float] = {}
        for seed, value in raw.items():
            num = _finite(value)
            if num is None:
                continue
            values[seed] = -num if lower_is_better else num
        per_variant[variant] = values
        union_seeds.update(values.keys())

    common = None
    for variant in variants:
        keys = set(per_variant[variant].keys())
        common = keys if common is None else (common & keys)
    common = common or set()
    seeds = sorted(common, key=_seed_sort_key)
    dropped = sorted(union_seeds - common, key=_seed_sort_key)
    matrix = [[per_variant[variant][seed] for variant in variants] for seed in seeds]
    return variants, seeds, matrix, dropped


def _row_ranks(values: Sequence[float]) -> List[float]:
    """一行定向分数 -> 名次（越大名次越靠前；并列取平均名次）。"""
    ranks = []
    for value in values:
        better = sum(1 for other in values if other > value)
        equal = sum(1 for other in values if other == value)
        ranks.append(1.0 + better + (equal - 1) / 2.0)
    return ranks


def _mean(values):
    """有限值的均值；正常路径保持朴素求和，溢出时按尺度缩放后还原。"""
    values = list(values)
    if not values:
        return None
    total = sum(values)
    if math.isfinite(total):
        return total / len(values)
    scale = max(abs(value) for value in values)
    if scale == 0.0:
        return 0.0
    if not math.isfinite(scale):
        return None
    scaled = sum(value / scale for value in values) / len(values) * scale
    return scaled if math.isfinite(scaled) else None


def _median(values):
    if not values:
        return None
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return float(ordered[mid])
    return (float(ordered[mid - 1]) + float(ordered[mid])) / 2.0


def _percentile(values, q):
    """线性插值百分位（q 在 [0,1]）；空输入返回 None。"""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = q * (len(ordered) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(ordered[lo])
    frac = pos - lo
    return float(ordered[lo] * (1.0 - frac) + ordered[hi] * frac)


def _base_payload(status, variants, seeds, dropped, lower_is_better, min_seeds):
    return {
        "status": status,
        "status_cn": STATUS_CN.get(status, status),
        "available": status == "ok",
        "min_seeds": min_seeds,
        "n_variants": len(variants),
        "n_seeds": len(seeds),
        "seeds": list(seeds),
        "dropped_seeds": list(dropped),
        "dropped_seed_count": len(dropped),
        "lower_is_better": bool(lower_is_better),
        "direction": "rank 1 = best (higher oriented score)",
    }


def ranking_table(scores, variants=None, lower_is_better=False, min_seeds=MIN_SEEDS):
    """逐种子名次与汇总排名（描述性）。

    返回 ``rows``（每方案的平均/中位名次、名次分布、逐种子最优次数与
    平均定向分数）、``rank_matrix``（逐种子名次与从好到差的顺序）以及
    ``leader``/``leaders``。``available=False`` 时仍尽可能保留描述性
    名次，但不给出“最优方案”的推断性措辞。
    """
    min_seeds = _strict_int(min_seeds, "min_seeds", minimum=2)
    variants, seeds, matrix, dropped = _normalise(scores, variants, lower_is_better)
    if len(variants) < 2:
        return {
            **_base_payload(
                "insufficient_variants",
                variants,
                seeds,
                dropped,
                lower_is_better,
                min_seeds,
            ),
            "rows": [],
            "rank_matrix": [],
            "leader": None,
            "leaders": [],
            "leader_unique": None,
        }
    if not seeds:
        return {
            **_base_payload(
                "no_common_seeds", variants, seeds, dropped, lower_is_better, min_seeds
            ),
            "rows": [],
            "rank_matrix": [],
            "leader": None,
            "leaders": [],
            "leader_unique": None,
        }

    rank_rows = [_row_ranks(row) for row in matrix]
    per_variant_ranks: Dict[str, List[float]] = {v: [] for v in variants}
    per_variant_scores: Dict[str, List[float]] = {v: [] for v in variants}
    rank_matrix = []
    for seed, row, ranks in zip(seeds, matrix, rank_rows):
        order = [
            variant
            for _, variant in sorted(
                zip(row, variants), key=lambda pair: (-pair[0], pair[1])
            )
        ]
        rank_matrix.append(
            {
                "seed": seed,
                "ranks": dict(zip(variants, ranks)),
                "order": order,
            }
        )
        for variant, value, rank in zip(variants, row, ranks):
            per_variant_scores[variant].append(value)
            per_variant_ranks[variant].append(rank)

    n_seeds = len(seeds)
    n_variants = len(variants)
    rows = []
    for variant in variants:
        ranks = per_variant_ranks[variant]
        counts = {}
        for rank in ranks:
            key = int(rank) if float(rank).is_integer() else rank
            counts[key] = counts.get(key, 0) + 1
        best_count = sum(1 for rank in ranks if rank <= 1.0)
        worst_count = sum(1 for rank in ranks if rank >= n_variants)
        rows.append(
            {
                "variant": variant,
                "n": len(ranks),
                "mean_rank": _mean(ranks),
                "median_rank": _median(ranks),
                "min_rank": min(ranks),
                "max_rank": max(ranks),
                "rank_counts": counts,
                "best_seed_count": best_count,
                "best_seed_rate": best_count / n_seeds,
                "worst_seed_count": worst_count,
                "mean_oriented_score": _mean(per_variant_scores[variant]),
            }
        )
    rows.sort(key=lambda row: (row["mean_rank"], row["variant"]))
    best_mean = rows[0]["mean_rank"]
    leaders = [
        row["variant"] for row in rows if abs(row["mean_rank"] - best_mean) <= 1e-12
    ]
    status = "ok" if n_seeds >= min_seeds else "insufficient_seeds"
    return {
        **_base_payload(status, variants, seeds, dropped, lower_is_better, min_seeds),
        "rows": rows,
        "rank_matrix": rank_matrix,
        "leader": leaders[0] if len(leaders) == 1 else None,
        "leaders": leaders,
        "leader_unique": len(leaders) == 1,
        "note": (
            "名次对单调变换稳健且并列取平均名次；平均名次差不是"
            "性能差，必须与成对效应量/SESOI 并读。"
        ),
    }


def bootstrap_rank_stability(
    scores,
    variants=None,
    lower_is_better=False,
    min_seeds=MIN_SEEDS,
    n_bootstrap=DEFAULT_BOOTSTRAP,
    seed=0,
):
    """对训练种子做 bootstrap：换一批种子后排名/最优方案是否稳定。

    每次有放回地重采样 ``n_seeds`` 个种子，按平均定向分数重新排名，统计
    每个方案成为最优的概率（并列最优按 1/k 分摊）与平均名次的 95% 区间。
    这是**描述性**不确定性：只覆盖训练种子抽样，不覆盖评测窗口/样例。
    """
    min_seeds = _strict_int(min_seeds, "min_seeds", minimum=2)
    n_bootstrap = _strict_int(n_bootstrap, "n_bootstrap")
    seed = _strict_int(seed, "seed", minimum=0)
    variants, seeds, matrix, dropped = _normalise(scores, variants, lower_is_better)
    payload = _base_payload("ok", variants, seeds, dropped, lower_is_better, min_seeds)
    payload.update(
        {
            "n_bootstrap": n_bootstrap,
            "bootstrap_seed": seed,
            "rows": [],
            "leader": None,
            "leaders": [],
        }
    )
    if len(variants) < 2:
        payload["status"] = "insufficient_variants"
        payload["status_cn"] = STATUS_CN[payload["status"]]
        payload["available"] = False
        return payload
    if len(seeds) < min_seeds:
        payload["status"] = "insufficient_seeds"
        payload["status_cn"] = STATUS_CN[payload["status"]]
        payload["available"] = False
        return payload

    rng = random.Random(seed)
    n_seeds = len(seeds)
    top_share = [0.0] * len(variants)
    rank_samples: Dict[str, List[float]] = {v: [] for v in variants}
    for _ in range(n_bootstrap):
        idx = [rng.randrange(n_seeds) for _ in range(n_seeds)]
        means = [_mean([matrix[i][j] for i in idx]) for j in range(len(variants))]
        ranks = _row_ranks(means)
        best = max(means)
        tied = [j for j, value in enumerate(means) if value == best]
        share = 1.0 / len(tied)
        for j in tied:
            top_share[j] += share
        for j, variant in enumerate(variants):
            rank_samples[variant].append(ranks[j])

    rows = []
    for j, variant in enumerate(variants):
        ranks = rank_samples[variant]
        rows.append(
            {
                "variant": variant,
                "top_probability": top_share[j] / n_bootstrap,
                "mean_rank_bootstrap": _mean(ranks),
                "mean_rank_ci95": [
                    _percentile(ranks, 0.025),
                    _percentile(ranks, 0.975),
                ],
            }
        )
    rows.sort(
        key=lambda row: (
            -row["top_probability"],
            row["mean_rank_bootstrap"],
            row["variant"],
        )
    )
    leaders = [
        row["variant"]
        for row in rows
        if abs(row["top_probability"] - rows[0]["top_probability"]) <= 1e-12
    ]
    payload.update(
        {
            "rows": rows,
            "leader": leaders[0] if len(leaders) == 1 else None,
            "leaders": leaders,
            "note": (
                "bootstrap 只重采样训练种子，不覆盖评测抽样；概率是"
                "描述性稳定度，不是“方案为真的概率”，也不改变预注册判定。"
            ),
        }
    )
    return payload


def permutation_selection_test(
    scores,
    variants=None,
    lower_is_better=False,
    min_seeds=MIN_SEEDS,
    n_permutations=DEFAULT_PERMUTATIONS,
    seed=0,
    alpha=DEFAULT_ALPHA,
):
    """“方案标签在种子内可交换”的全局零假设置换检验。

    统计量：``max_j(mean_j − grand_mean)``（最优方案相对总体平均的改善）
    与 ``max_j mean_j − min_j mean_j``（整体名次跨度）。在每个种子内独立
    打乱方案标签，保留种子效应与方案间的相关结构，得到“从 V 个方案里
    挑最好”的零分布。p 值为 Monte-Carlo 置换 p（含观测样本 +1 校正）。

    该检验回答“是否存在任何方案差异 / 观察到的领先是否大于纯选择效应”，
    不指定哪个方案更好；事后点名最优方案仍需成对检验与效应量。
    """
    min_seeds = _strict_int(min_seeds, "min_seeds", minimum=2)
    n_permutations = _strict_int(n_permutations, "n_permutations")
    seed = _strict_int(seed, "seed", minimum=0)
    alpha = _strict_alpha(alpha)
    variants, seeds, matrix, dropped = _normalise(scores, variants, lower_is_better)
    payload = _base_payload("ok", variants, seeds, dropped, lower_is_better, min_seeds)
    payload.update(
        {
            "n_permutations": n_permutations,
            "permutation_seed": seed,
            "alpha": alpha,
            "exact": False,
            "p_max": None,
            "p_range": None,
            "significant_max": None,
            "significant_range": None,
            "winner": None,
            "runner_up": None,
            "lead_over_runner_up": None,
            "observed_max_centered": None,
            "expected_null_max_centered": None,
            "selection_inflation": None,
        }
    )
    if len(variants) < 2:
        payload["status"] = "insufficient_variants"
        payload["status_cn"] = STATUS_CN[payload["status"]]
        payload["available"] = False
        return payload
    if len(seeds) < min_seeds:
        payload["status"] = "insufficient_seeds"
        payload["status_cn"] = STATUS_CN[payload["status"]]
        payload["available"] = False
        return payload

    n_variants = len(variants)
    means = [_mean([row[j] for row in matrix]) for j in range(n_variants)]
    grand = _mean(means)
    centered = [value - grand for value in means]
    ordered = sorted(zip(centered, variants), key=lambda p: (-p[0], p[1]))
    obs_max = ordered[0][0]
    obs_range = max(means) - min(means)
    winner = ordered[0][1]
    runner_up = ordered[1][1] if n_variants > 1 else None
    lead = (
        (means[variants.index(winner)] - means[variants.index(runner_up)])
        if runner_up
        else None
    )

    rng = random.Random(seed)
    ge_max = 0
    ge_range = 0
    null_max_total = 0.0
    tol = 1e-12
    row_indices = list(range(n_variants))
    for _ in range(n_permutations):
        # 每个种子内独立打乱方案标签，再按列（方案）求跨种子均值。
        # 注意不能把“种子行的均值”当成方案均值：那样会把种子水平差异
        # 误当成方案差异，也会让完全相同的方案产生虚假的 max 统计量。
        columns = [[] for _ in range(n_variants)]
        for row in matrix:
            order = list(row_indices)
            rng.shuffle(order)
            for j, source in enumerate(order):
                columns[j].append(row[source])
        perm_means = [_mean(column) for column in columns]
        perm_grand = _mean(perm_means)
        perm_max = max(value - perm_grand for value in perm_means)
        perm_range = max(perm_means) - min(perm_means)
        null_max_total += perm_max
        if perm_max >= obs_max - tol:
            ge_max += 1
        if perm_range >= obs_range - tol:
            ge_range += 1

    p_max = (1 + ge_max) / (n_permutations + 1)
    p_range = (1 + ge_range) / (n_permutations + 1)
    expected_null_max = null_max_total / n_permutations
    payload.update(
        {
            "p_max": p_max,
            "p_range": p_range,
            "significant_max": bool(p_max < alpha),
            "significant_range": bool(p_range < alpha),
            "winner": winner,
            "runner_up": runner_up,
            "lead_over_runner_up": lead,
            "observed_max_centered": obs_max,
            "expected_null_max_centered": expected_null_max,
            "selection_inflation": obs_max - expected_null_max,
            "note": (
                "p_max 检验“是否存在方案差异/领先是否超过纯选择最大值”，"
                "不指定哪个方案更好；Monte-Carlo p 分辨率约 "
                f"1/{n_permutations + 1}。方案标签交换性不成立（缺失机制/"
                "预算/协议不同）时 p 只作线索，不改变预注册判定。"
            ),
        }
    )
    return payload


def analyze_ranking(
    scores,
    variants=None,
    lower_is_better=False,
    min_seeds=MIN_SEEDS,
    n_bootstrap=DEFAULT_BOOTSTRAP,
    n_permutations=DEFAULT_PERMUTATIONS,
    seed=0,
    alpha=DEFAULT_ALPHA,
) -> dict:
    """把名次表、种子 bootstrap 与选择校正置换检验合并成一个结果层。

    供流水线对每个指标（PPL、RULER 平均等）直接消费；``seed`` 同时驱动
    bootstrap 与置换检验，保证同一输入恒定复现。任何子层不可用都保留
    明确的 ``status``/``reason``，不伪造 0 效应。
    """
    try:
        table = ranking_table(
            scores,
            variants=variants,
            lower_is_better=lower_is_better,
            min_seeds=min_seeds,
        )
    except (TypeError, ValueError) as exc:
        return {
            "status": "invalid",
            "status_cn": STATUS_CN["invalid"],
            "available": False,
            "reason": str(exc),
            "table": None,
            "bootstrap": None,
            "selection_test": None,
        }
    out = {
        "status": table.get("status"),
        "status_cn": table.get("status_cn"),
        "available": bool(table.get("available")),
        "method": (
            "逐种子秩统计 + 种子 bootstrap + 种子内标签置换的"
            "选择校正检验（探索性、描述性）"
        ),
        "table": table,
        "bootstrap": None,
        "selection_test": None,
    }
    if not table.get("available"):
        out["reason"] = table.get("status_cn")
        return out
    out["bootstrap"] = bootstrap_rank_stability(
        scores,
        variants=variants,
        lower_is_better=lower_is_better,
        min_seeds=min_seeds,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    out["selection_test"] = permutation_selection_test(
        scores,
        variants=variants,
        lower_is_better=lower_is_better,
        min_seeds=min_seeds,
        n_permutations=n_permutations,
        seed=seed,
        alpha=alpha,
    )
    return out
