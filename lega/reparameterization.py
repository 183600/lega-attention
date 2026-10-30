"""参数化稳健性：归一化集中度 c 与原始 log max_p 的对照分析。

LEGA 的 B 因子在默认实现里使用 ``c = log(n * max_j p_j)``，即
``a*c = a*log max_p + a*log n``；它同时携带集中度与长度信息。显式的
C 因子 ``b*log n`` 因此并不与 B 因子里的长度分量正交。仅凭
``lega_no_len`` 与 ``lega`` 的差异不能把长度效应单独归给 C，也不能把
``lega_no_conc`` 的差异全部解释为“集中度证据”的贡献。

本模块把同一组 A/B/C 因子放到两种参数化上：

* 归一化：``a*log(n*max_p) + b*log n``（现有主矩阵）；
* 原始：``a*log max_p + b*log n``（补充矩阵）。

两种参数化都能表示同一个线性门控函数族。比较它们的去一法/单因子边
以及 2^3 Shapley 分解，回答的是“因子归因是否依赖参数化”，而不是
“哪种参数化更好”。差异-of-differences 检验按同一种子配对：

    D_i = effect_raw,i - effect_normalized,i

其中 effect 是“候选相对参照”的定向改善（PPL 用 -log PPL，RULER 用
原始 EM），正值表示原始参数化下该因子的改善更大。对 D_i 做单样本
配对 t 检验；n<3 时只报告描述统计，不作显著性判定。

这是探索性分析：它没有预注册，不替代预注册的单因素确认检验。
"""
import math
from typing import Mapping

from .factorial import CELL_ORDER, decomposition, oriented_value
from .utils import (holm_bonferroni, mean_std, min_detectable_effect,
                    paired_t_test)

# 门控内 2^3 的归一化参数化单元格（与主矩阵一致）。
NORMALIZED_CELLS = {
    "none": "gate_only",
    "A": "qgate",
    "B": "lega_conc_only",
    "C": "lega_len_only",
    "AB": "lega_no_len",
    "AC": "lega_no_conc",
    "BC": "lega_no_query",
    "ABC": "lega",
}

# 原始参数化：B 只乘 log max_p，不再并入 log n。none/A/C 与归一化网格
# 共用同一变体，因为只有 B 的表示改变；AB/AC/BC/ABC 需要对应的原始版。
RAW_CELLS = {
    "none": "gate_only",
    "A": "qgate",
    "B": "lega_raw_conc_only",
    "C": "lega_len_only",
    "AB": "lega_raw_no_len",
    "AC": "lega_no_conc",
    "BC": "lega_raw_no_query",
    "ABC": "lega_raw",
}

RAW_FACTORIAL_VARIANTS = tuple(
    dict.fromkeys(RAW_CELLS[cell] for cell in CELL_ORDER))

# 只属于原始网格、归一化网格没有的变体。判断“是否运行了 raw 对照”必须
# 看这些变体；none/A/C 等共享格不能作为原始对照已运行的证据。
RAW_SPECIFIC_VARIANTS = tuple(
    v for v in RAW_FACTORIAL_VARIANTS
    if v not in set(NORMALIZED_CELLS.values()))

FACTOR_LABELS = {"A": "A 查询项", "B": "B 集中度证据", "C": "C 长度校准"}

MIN_SEEDS = 3

ALPHA = 0.05

# (因子, 边类型, 归一化(候选, 参照), 原始(候选, 参照), 说明)。
# Full−B 的归一化候选 lega_no_conc 与原始候选相同、参照不同：这正好把
# “B 里是否含 a*log n”这一差异隔离出来。B 的增量边同时给出更干净的
# “只加入 B”对照（gate_only 为共同基准）。
CONTRASTS = (
    ("A", "remove", ("lega_no_query", "lega"),
     ("lega_raw_no_query", "lega_raw"),
     "Full−A：归一化版同时去掉 a*log n，原始版只去掉 a*log max_p。"),
    ("B", "remove", ("lega_no_conc", "lega"),
     ("lega_no_conc", "lega_raw"),
     "Full−B：归一化版去掉 a*(log max_p+log n)，原始版只去掉 a*log max_p。"),
    ("C", "remove", ("lega_no_len", "lega"),
     ("lega_raw_no_len", "lega_raw"),
     "Full−C：两版都只去掉 b*log n。"),
    ("B", "add", ("lega_conc_only", "gate_only"),
     ("lega_raw_conc_only", "gate_only"),
     "仅加入 B：归一化版为 a*(log max_p+log n)，原始版为 a*log max_p。"),
)


def _finite(value):
    if isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return out if math.isfinite(out) else None


def _ordered_seeds(mapping):
    try:
        return sorted(mapping)
    except TypeError:
        return sorted(mapping, key=lambda s: (type(s).__name__, str(s)))


def contrast_effects(values: Mapping, metric: str, candidate: str,
                     reference: str) -> dict:
    """逐种子计算候选−参照的定向效应（正值=改善，PPL 已转 −log）。"""
    out = {}
    if not isinstance(values, Mapping):
        return out
    for seed, row in values.items():
        if not isinstance(row, Mapping):
            continue
        if candidate not in row or reference not in row:
            continue
        try:
            vc = oriented_value(metric, row[candidate])
            vr = oriented_value(metric, row[reference])
        except (TypeError, ValueError):
            continue
        effect = vc - vr
        if math.isfinite(effect):
            out[seed] = effect
    return out


def _sign(value):
    if value is None:
        return 0
    return 1 if value > 0 else (-1 if value < 0 else 0)


def _effect_summary(effects):
    vals = [effects[s] for s in _ordered_seeds(effects)]
    mean, std = mean_std(vals)
    return {"n": len(vals), "mean": mean, "std": std}


def _parameterization_verdict(res, normalized_mean, raw_mean,
                              alpha=ALPHA, min_seeds=MIN_SEEDS):
    n = res.get("n") or 0
    if n < min_seeds:
        return (f"样本不足（n<{min_seeds}）：仅描述，"
                "不作参数化稳健性判定")
    p_adj = res.get("p_adj")
    if p_adj is None:
        return "无法检验（效应差方差不可估计）"
    sign_norm = _sign(normalized_mean)
    sign_raw = _sign(raw_mean)
    flipped = sign_norm != 0 and sign_raw != 0 and sign_norm != sign_raw
    if p_adj < alpha:
        if flipped:
            return "参数化依赖：效应方向翻转"
        return "参数化依赖：效应大小随参数化显著变化"
    if flipped:
        return "方向不一致但未达显著（不得据此宣称稳健）"
    mde = _finite(res.get("mde"))
    observed = _finite(res.get("mean_diff"))
    if mde is not None and observed is not None and abs(observed) < mde:
        return "未检出参数化差异（观察差<MDE，功效不足；不证明等价）"
    return "未检出参数化差异（不证明等价）"


def _assign_holm(tests, alpha=ALPHA, min_seeds=MIN_SEEDS):
    pads = holm_bonferroni([
        (t.get("p") if (t.get("n") or 0) >= min_seeds else None)
        for t in tests])
    for test, padj in zip(tests, pads):
        test["p_adj"] = padj
        norm_mean = test.get("normalized_effect_mean")
        raw_mean = test.get("raw_effect_mean")
        if norm_mean is None:
            norm_mean = test.get("normalized_phi_mean")
        if raw_mean is None:
            raw_mean = test.get("raw_phi_mean")
        test["verdict"] = _parameterization_verdict(
            test, norm_mean, raw_mean, alpha=alpha, min_seeds=min_seeds)
    return tests


def contrast_parameterization_tests(values: Mapping, metric: str,
                                    min_seeds: int = MIN_SEEDS,
                                    alpha: float = ALPHA) -> list:
    """逐条单因子边比较“原始参数化−归一化参数化”的效应。"""
    tests = []
    for factor, direction, norm, raw, note in CONTRASTS:
        norm_eff = contrast_effects(values, metric, norm[0], norm[1])
        raw_eff = contrast_effects(values, metric, raw[0], raw[1])
        seeds = [s for s in _ordered_seeds(norm_eff) if s in raw_eff]
        diffs = [raw_eff[s] - norm_eff[s] for s in seeds]
        res = paired_t_test(diffs, [0.0] * len(diffs))
        norm_sum = _effect_summary({s: norm_eff[s] for s in seeds})
        raw_sum = _effect_summary({s: raw_eff[s] for s in seeds})
        signs = [(_sign(norm_eff[s]), _sign(raw_eff[s])) for s in seeds]
        agree = [a == b for a, b in signs if a != 0 and b != 0]
        test = {
            "candidate": f"{factor}（{direction}）",
            "reference": "原始参数化−归一化参数化",
            "metric": f"{metric}参数化差异",
            "base_metric": metric,
            "factor": factor,
            "factor_label": FACTOR_LABELS[factor],
            "direction": direction,
            "normalized_candidate": norm[0],
            "normalized_reference": norm[1],
            "raw_candidate": raw[0],
            "raw_reference": raw[1],
            "normalized_effect_mean": norm_sum["mean"],
            "normalized_effect_std": norm_sum["std"],
            "raw_effect_mean": raw_sum["mean"],
            "raw_effect_std": raw_sum["std"],
            "seed_sign_agreement": (
                float(sum(agree) / len(agree)) if agree else None),
            "sign_agree": (
                None if (norm_sum["mean"] is None or raw_sum["mean"] is None
                         or _sign(norm_sum["mean"]) == 0
                         or _sign(raw_sum["mean"]) == 0)
                else bool(_sign(norm_sum["mean"]) == _sign(raw_sum["mean"]))),
            "normalized_effect_by_seed": {
                str(s): norm_eff[s] for s in seeds},
            "raw_effect_by_seed": {str(s): raw_eff[s] for s in seeds},
            "difference_by_seed": {str(s): d for s, d in zip(seeds, diffs)},
            "mde": min_detectable_effect(res.get("std_diff"), res.get("n")),
            "no_direction": True,
            "lower_is_better": None,
            "exploratory": True,
            "note": note,
        }
        test.update(res)
        if metric == "PPL":
            for key, effect in (("normalized", norm_sum["mean"]),
                                ("raw", raw_sum["mean"])):
                test[f"{key}_ppl_ratio"] = (
                    math.exp(max(-20.0, min(20.0, -float(effect))))
                    if effect is not None else None)
        elif metric == "RULER平均":
            for key, effect in (("normalized", norm_sum["mean"]),
                                ("raw", raw_sum["mean"])):
                test[f"{key}_effect_pp"] = (
                    float(effect) * 100.0 if effect is not None else None)
        tests.append(test)
    return _assign_holm(tests, alpha=alpha, min_seeds=min_seeds)


def shapley_effects(values: Mapping, cells: Mapping, metric: str) -> dict:
    """逐种子计算指定参数化 2^3 网格的 Shapley 分解。"""
    out = {}
    if not isinstance(values, Mapping):
        return out
    for seed, row in values.items():
        if not isinstance(row, Mapping):
            continue
        raw = {}
        ok = True
        for cell in CELL_ORDER:
            variant = cells.get(cell)
            if variant is None or variant not in row:
                ok = False
                break
            raw[cell] = row[variant]
        if not ok:
            continue
        try:
            oriented = {cell: oriented_value(metric, raw[cell])
                        for cell in CELL_ORDER}
        except (TypeError, ValueError):
            continue
        out[seed] = decomposition(oriented)
    return out


def shapley_parameterization_tests(values: Mapping, metric: str,
                                   min_seeds: int = MIN_SEEDS,
                                   alpha: float = ALPHA) -> list:
    """比较两种参数化下 A/B/C 的 Shapley 贡献（同种子配对）。"""
    norm = shapley_effects(values, NORMALIZED_CELLS, metric)
    raw = shapley_effects(values, RAW_CELLS, metric)
    seeds = [s for s in _ordered_seeds(norm) if s in raw]
    if not seeds:
        return []
    tests = []
    for factor in ("A", "B", "C"):
        diffs = [raw[s]["phi"][factor] - norm[s]["phi"][factor]
                 for s in seeds]
        res = paired_t_test(diffs, [0.0] * len(diffs))
        norm_phi = [norm[s]["phi"][factor] for s in seeds]
        raw_phi = [raw[s]["phi"][factor] for s in seeds]
        norm_mean, norm_std = mean_std(norm_phi)
        raw_mean, raw_std = mean_std(raw_phi)
        test = {
            "candidate": f"{factor} 的 Shapley 贡献",
            "reference": "原始参数化−归一化参数化",
            "metric": f"{metric}Shapley参数化差异",
            "base_metric": metric,
            "factor": factor,
            "factor_label": FACTOR_LABELS[factor],
            "normalized_phi_mean": norm_mean,
            "normalized_phi_std": norm_std,
            "raw_phi_mean": raw_mean,
            "raw_phi_std": raw_std,
            "sign_agree": (
                None if (norm_mean is None or raw_mean is None
                         or _sign(norm_mean) == 0 or _sign(raw_mean) == 0)
                else bool(_sign(norm_mean) == _sign(raw_mean))),
            "normalized_phi_by_seed": {
                str(s): norm[s]["phi"][factor] for s in seeds},
            "raw_phi_by_seed": {
                str(s): raw[s]["phi"][factor] for s in seeds},
            "difference_by_seed": {
                str(s): d for s, d in zip(seeds, diffs)},
            "normalized_total_effect_mean": mean_std(
                [norm[s]["total_effect"] for s in seeds])[0],
            "raw_total_effect_mean": mean_std(
                [raw[s]["total_effect"] for s in seeds])[0],
            "normalized_efficiency_gap_max": max(
                abs(norm[s]["efficiency_gap"]) for s in seeds),
            "raw_efficiency_gap_max": max(
                abs(raw[s]["efficiency_gap"]) for s in seeds),
            "mde": min_detectable_effect(res.get("std_diff"), res.get("n")),
            "no_direction": True,
            "lower_is_better": None,
            "exploratory": True,
        }
        test.update(res)
        if metric == "PPL":
            for key, effect in (("normalized", norm_mean),
                                ("raw", raw_mean)):
                test[f"{key}_ppl_ratio"] = (
                    math.exp(max(-20.0, min(20.0, -float(effect))))
                    if effect is not None else None)
        tests.append(test)
    tests = _assign_holm(tests, alpha=alpha, min_seeds=min_seeds)
    for test in tests:
        test["shapley_factor"] = test["factor"]
        test["shapley_factor_label"] = FACTOR_LABELS[test["factor"]]
    return tests


def analyze(values: Mapping, metric: str, min_seeds: int = MIN_SEEDS,
            alpha: float = ALPHA) -> dict:
    """对一种指标执行参数化稳健性分析并给出汇总判定。"""
    contrasts = contrast_parameterization_tests(
        values, metric, min_seeds=min_seeds, alpha=alpha)
    shapley = shapley_parameterization_tests(
        values, metric, min_seeds=min_seeds, alpha=alpha)
    claims = contrasts + shapley
    eligible = [t for t in claims
                if (t.get("n") or 0) >= min_seeds
                and t.get("p_adj") is not None]
    n_sig = sum(1 for t in eligible if t["p_adj"] < alpha)
    flips = [t["candidate"] for t in eligible
             if t.get("sign_agree") is False]
    if not claims or not eligible:
        status = "insufficient"
        verdict = ("缺少可配对且 n>=" + str(min_seeds)
                   + " 的原始/归一化观测，无法评估参数化稳健性。"
                   if claims else
                   "缺少可配对的原始/归一化网格，无法评估参数化稳健性。")
    elif n_sig or flips:
        status = "parameterization_dependent"
        detail = []
        if n_sig:
            detail.append(f"{n_sig} 条差异检验显著")
        if flips:
            detail.append("方向不一致：" + "、".join(flips))
        verdict = ("参数化稳健性未建立：" + "；".join(detail)
                   + "。不能把归一化 c 的因子效应直接解释为纯集中度。")
    else:
        status = "no_difference_detected"
        verdict = ("未检出参数化差异；这是“未拒绝稳健”而非“证明等价”，"
                   "需结合各检验的 MDE 与种子数解读。")
    return {
        "metric": metric,
        "contrasts": contrasts,
        "shapley": shapley,
        "summary": {
            "status": status,
            "verdict": verdict,
            "n_contrasts": len(contrasts),
            "n_shapley": len(shapley),
            "n_significant": n_sig,
            "sign_flips": flips,
            "min_seeds": min_seeds,
            "alpha": alpha,
        },
        "exploratory": True,
    }
