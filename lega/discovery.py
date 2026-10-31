"""探索性种子级发现的 FDR 控制（Benjamini–Hochberg / Benjamini–Yekutieli）。

为什么需要这一层
================
项目已经对每个指标家族单独做 Holm–Bonferroni 校正，控制该家族内的
family-wise error rate（FWER）。但探索性检验横跨 PPL、RULER、因果病灶、
剂量-反应、推理期 Shapley/交互、长度分层、n-gram 基线与参数化稳健性等
多个家族；“每个家族各自 5%”并不控制“全部探索性发现中假发现的比例”。
当汇总页同时展示几十个未预注册对比时，读者需要一个跨家族的探索性错误率
尺度，而不是把每个家族内的 p_adj 当成互不相干的结论。

本模块把**非预注册的种子级探索性检验**合并为一个明确的发现族，计算
Benjamini–Hochberg（BH）q 值。BH 在独立或正相关（PRDS）下控制
false discovery rate（FDR）：E[V/R | R>0]·P(R>0) ≤ α。共享种子、共享
checkpoint 与共享评测样本会引入依赖，PRDS 未必严格成立，因此同时报告
Benjamini–Yekutieli（BY）q 值作为任意依赖下的保守敏感性。

家族定义
========
纳入（当相应分析实际运行且产生 p 值时）：

* 主汇总表中所有**未登记为预注册假设**的种子级配对检验（含整体对比、
  单因素对比、训练期交互、训练期 Shapley、PPL(log)、RULER 逐任务）；
* 推理期因果病灶与活动匹配病灶（每个 (变体, 病灶) 跨种子检验）；
* 原始病灶 − 匹配病灶的跨种子对照；
* 连续剂量-反应的跨种子检验；
* 推理期 Shapley 与推理期交互；
* RULER 长度分档、长度斜率与首末落差；
* n-gram 基线的逐变体单样本配对检验；
* 参数化稳健性对照/Shapley 与线性探针基线（若运行）。

排除：预注册确认性检验（由 Holm/FWER 与预注册门单独处理）、样例级
McNemar、窗口级 PPL、RULER 分层 meta 分析、分析选择多重宇宙、缺失结果
敏感性、逐 step 训练预算轨迹。这些层的推断单位不是“训练种子”，或本身
是敏感性/描述性分析，混入会改变家族含义。

边界
====
* q 值是 BH/BY 校正后的显著性尺度，**不是“该假设为真的概率”**，也不是
  效应量；q<α 只说明该发现进入 FDR 控制下的发现集合。
* 本层不改变任何 Holm/FWER 判定、预注册 支持/拒绝/未定、MDE、TOST 或
  确认性结论；探索性发现仍须在新数据/新种子上预注册后复现。
* 家族由流水线的分析层固定，不由观察到的 p 值挑选；但“哪些层实际运行”
  会随 CLI 变化，汇总页必须如实报告家族大小与各来源计数。
* n<3 或 p 值缺失/非法的检验不进入家族，也不被算作“不显著”；它们与
  主协议一致，只作描述性证据。
"""
import math
from numbers import Integral, Real

DEFAULT_ALPHA = 0.05
DEFAULT_MIN_SEEDS = 3

SOURCE_CN = {
    "main": "主检验（未登记）",
    "ppl_log": "PPL(log) 尺度",
    "ruler_task": "RULER 逐任务",
    "factorial_interaction": "训练期交互",
    "factorial_shapley": "训练期 Shapley",
    "causal_lesion": "推理期因果病灶",
    "causal_matched_lesion": "活动匹配病灶",
    "causal_matched_contrast": "原始−匹配病灶",
    "capacity_decomposition": "训练期容量匹配证据消融",
    "dose_response": "剂量-反应",
    "inference_shapley": "推理期 Shapley",
    "inference_interaction": "推理期交互",
    "ruler_length_ctx": "RULER 长度分档",
    "ruler_length_slope": "RULER 长度斜率",
    "ruler_length_span": "RULER 首末落差",
    "ngram_baseline": "n-gram 基线",
    "reparameterization_contrast": "参数化对照",
    "reparameterization_shapley": "参数化 Shapley",
    "probe_baseline": "线性探针基线",
    "unknown": "未标注来源",
}


def source_label(source):
    """把机器可读来源名转成汇总页中文标签。"""
    return SOURCE_CN.get(source, str(source) if source is not None else "-")


def _finite_p(value):
    """p 值 → float；None/NaN/Inf/bool/越界/非数值一律按缺失处理。"""
    if value is None or isinstance(value, bool):
        return None
    if type(value).__name__ == "bool_":
        return None
    if not isinstance(value, Real):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return out if math.isfinite(out) and 0.0 <= out <= 1.0 else None


def _valid_n(value):
    """有效种子数 → 非负整数；bool/小数/非数值/负数按缺失处理。"""
    if value is None or isinstance(value, bool):
        return None
    if type(value).__name__ == "bool_":
        return None
    if isinstance(value, Integral):
        out = int(value)
    elif isinstance(value, float) and math.isfinite(value) \
            and value.is_integer():
        out = int(value)
    else:
        return None
    return out if out >= 0 else None


def _alpha_value(alpha):
    """FDR 水平必须是 (0,1) 内的有限实数。"""
    if isinstance(alpha, bool) or not isinstance(alpha, Real):
        raise ValueError(f"alpha 必须是 (0,1) 内的有限实数，收到 {alpha!r}")
    out = float(alpha)
    if not math.isfinite(out) or not 0.0 < out < 1.0:
        raise ValueError(f"alpha 必须是 (0,1) 内的有限实数，收到 {alpha!r}")
    return out


def _min_seeds_value(min_seeds):
    """最小种子数必须是 >=1 的整数。"""
    if isinstance(min_seeds, bool) or not isinstance(min_seeds, int):
        raise ValueError(f"min_seeds 必须是 >=1 的整数，收到 {min_seeds!r}")
    if min_seeds < 1:
        raise ValueError(f"min_seeds 必须是 >=1 的整数，收到 {min_seeds!r}")
    return min_seeds


def benjamini_hochberg(pvals):
    """Benjamini–Hochberg step-up 校正的 q 值（同序返回，缺失保持 None）。

    对有效 p 升序排列，第 i 个（1-based）的 q 为
    ``min_{j>=i} m * p_(j) / j``，并保证 q 单调不减且封顶 1。
    从最大秩向前取 running minimum 自然给出该定义；相同 p 值会得到相同 q。
    None/NaN/Inf/bool/越界/非数值不参与家族大小，也不影响其他 p 的校正。
    """
    pvals = list(pvals)
    norm = [_finite_p(p) for p in pvals]
    idx = [i for i, p in enumerate(norm) if p is not None]
    out = [None] * len(pvals)
    m = len(idx)
    if m == 0:
        return out
    order = sorted(idx, key=lambda i: (norm[i], i))
    q_sorted = [None] * m
    running = 1.0
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        value = norm[i] * m / rank
        running = min(running, value)
        q_sorted[rank - 1] = min(1.0, max(norm[i], running))
    for rank, i in enumerate(order):
        out[i] = q_sorted[rank]
    return out


def benjamini_yekutieli(pvals):
    """Benjamini–Yekutieli 校正 q 值（任意依赖下控制 FDR 的保守版本）。

    BY 的调整 p 等于 BH q 值乘以调和数 ``H_m = sum_{i=1..m} 1/i`` 后封顶 1。
    它不要求独立/PRDS，但在 m 较大时明显更保守；本模块只把它作为共享种子/
    模型依赖下的敏感性对照，不替代 BH 主结果。
    """
    bh = benjamini_hochberg(pvals)
    m = sum(1 for q in bh if q is not None)
    if m == 0:
        return bh
    harmonic = math.fsum(1.0 / i for i in range(1, m + 1))
    return [None if q is None else min(1.0, q * harmonic) for q in bh]


def _text(value):
    """JSON 标签的安全文本形式；None 保持 None。"""
    if value is None or isinstance(value, str):
        return value
    return str(value)


def _compact_test(t):
    """把一条检验压缩为 JSON/Markdown 可消费的发现行。"""
    candidate = t.get("candidate")
    if candidate is None:
        candidate = t.get("variant")
    detail = {}
    for key in ("lesion", "ruler_ctx", "dose_factor", "interaction",
                "shapley_factor", "variant"):
        value = t.get(key)
        if value is not None:
            detail[key] = _text(value)
    return {
        "source": t.get("fdr_source"),
        "candidate": _text(candidate),
        "reference": _text(t.get("reference")),
        "metric": _text(t.get("metric")),
        "n": t.get("n"),
        "p": t.get("p"),
        "q_bh": t.get("q_bh"),
        "q_by": t.get("q_by"),
        "mean_diff": t.get("mean_diff"),
        "within_noise": t.get("within_noise"),
        "below_mde": t.get("below_mde"),
        "loo_sign_stable": t.get("loo_sign_stable"),
        "verdict": _text(t.get("verdict")),
        "fdr_excluded_reason": t.get("fdr_excluded_reason"),
        "detail": detail or None,
    }


def exploratory_fdr(tests, alpha=DEFAULT_ALPHA, min_seeds=DEFAULT_MIN_SEEDS):
    """对一组种子级探索性检验计算 BH/BY q 值并返回发现摘要。

    输入为检验 dict 列表，每条至少需要 ``p`` 与 ``n``；函数会就地写入
    ``q_bh``/``q_by``/``fdr_significant``/``fdr_by_significant``/
    ``fdr_family_size``/``fdr_source``/``fdr_excluded_reason``，并返回
    可机读摘要（家族大小、各来源计数、发现列表、排除原因等）。

    家族只包含 n>=min_seeds 且 p 可解析的检验；n 不足或 p 缺失的检验
    显式标为排除，不占家族大小，也不被当成“不显著”证据。
    """
    alpha = _alpha_value(alpha)
    min_seeds = _min_seeds_value(min_seeds)
    tests = tests if isinstance(tests, list) else []
    items, seen = [], set()
    for t in tests:
        if not isinstance(t, dict) or id(t) in seen:
            continue
        seen.add(id(t))
        items.append(t)

    included, excluded = [], []
    for t in items:
        p = _finite_p(t.get("p"))
        n = _valid_n(t.get("n"))
        reason = None
        if p is None:
            reason = "missing_or_invalid_p"
        elif n is None:
            reason = "missing_or_invalid_n"
        elif n < min_seeds:
            reason = "insufficient_seeds"
        t["fdr_family"] = "exploratory_seed_level"
        t["fdr_source"] = t.get("fdr_source") or "unknown"
        if reason is not None:
            t["q_bh"] = None
            t["q_by"] = None
            t["fdr_significant"] = False
            t["fdr_by_significant"] = False
            t["fdr_family_size"] = None
            t["fdr_excluded_reason"] = reason
            excluded.append(_compact_test(t))
        else:
            t["fdr_excluded_reason"] = None
            included.append(t)

    m = len(included)
    q_bh = benjamini_hochberg([t.get("p") for t in included])
    harmonic = (math.fsum(1.0 / i for i in range(1, m + 1))
                if m else None)
    for t, q in zip(included, q_bh):
        t["q_bh"] = q
        t["q_by"] = (None if q is None or harmonic is None
                     else min(1.0, q * harmonic))
        t["fdr_family_size"] = m
        t["fdr_significant"] = bool(q is not None and q < alpha)
        t["fdr_by_significant"] = bool(
            t.get("q_by") is not None and t["q_by"] < alpha)
    for t in items:
        if t.get("fdr_family_size") is None:
            t["fdr_family_size"] = m

    discoveries = [_compact_test(t) for t in included
                   if t.get("fdr_significant")]
    discoveries.sort(key=lambda r: (r.get("q_bh"), r.get("p")))
    by_discoveries = [_compact_test(t) for t in included
                      if t.get("fdr_by_significant")]
    by_discoveries.sort(key=lambda r: (r.get("q_by"), r.get("p")))

    n_by_source, n_discoveries_by_source = {}, {}
    for t in included:
        source = t.get("fdr_source") or "unknown"
        n_by_source[source] = n_by_source.get(source, 0) + 1
        if t.get("fdr_significant"):
            n_discoveries_by_source[source] = (
                n_discoveries_by_source.get(source, 0) + 1)
    excluded_by_reason = {}
    for row in excluded:
        reason = row.get("fdr_excluded_reason") or "unknown"
        excluded_by_reason[reason] = excluded_by_reason.get(reason, 0) + 1

    n_discoveries = len(discoveries)
    p_threshold = (max(r["p"] for r in discoveries)
                   if discoveries else None)
    q_threshold = (max(r["q_bh"] for r in discoveries)
                   if discoveries else None)
    return {
        "available": bool(m),
        "method": ("Benjamini–Hochberg step-up FDR; "
                   "Benjamini–Yekutieli as dependence-robust sensitivity"),
        "alpha": alpha,
        "min_seeds": min_seeds,
        "family": "exploratory_seed_level",
        "family_definition": (
            "所有非预注册的种子级探索性检验：主表未登记项、因果病灶/匹配"
            "对照、剂量-反应、推理期 Shapley/交互、RULER 长度、n-gram "
            "基线、参数化稳健性与线性探针基线（实际运行时）"),
        "family_size": m,
        "n_tests_total": len(items),
        "n_included": m,
        "n_excluded": len(excluded),
        "excluded_by_reason": excluded_by_reason,
        "harmonic_factor": harmonic,
        "n_discoveries": n_discoveries,
        "n_discoveries_by": len(by_discoveries),
        "expected_false_discoveries_at_fdr": alpha * n_discoveries,
        "expected_false_discoveries_note": (
            "启发式折算 α×R：BH 保证 E[V/R]≤α，不保证单次实现的假发现数 "
            "V≤αR；该字段只用于量级沟通，不是逐次上界。"),
        "p_threshold": p_threshold,
        "q_threshold": q_threshold,
        "n_by_source": n_by_source,
        "n_discoveries_by_source": n_discoveries_by_source,
        "discoveries": discoveries,
        "by_discoveries": by_discoveries,
        "excluded": excluded,
        "note": (
            "BH 在独立/PRDS 下控制 FDR；共享种子/模型/评测样本带来依赖，"
            "故并列 BY q 作为任意依赖下的保守敏感性。q 不是假设为真的概率，"
            "不替代 Holm/FWER、预注册判定、MDE 或 TOST，也不把探索性发现"
            "升级为确认性结论。n<3 或 p 缺失的检验不进入家族，也不等于"
            "“不显著”。"),
    }


def _main_source(t):
    """主检验的子来源标签（便于汇总页分来源计数）。"""
    if t.get("shapley_factor"):
        return "factorial_shapley"
    if t.get("interaction"):
        return "factorial_interaction"
    if t.get("ruler_task"):
        return "ruler_task"
    if t.get("metric") == "PPL(log)":
        return "ppl_log"
    return "main"


def collect_seed_level_tests(main_tests=None, causal_tests=None,
                             matched_ablation_contrasts=None,
                             dose_tests=None, inference_shapley=None,
                             inference_interactions=None,
                             capacity_decomposition=None,
                             length_analysis=None, ngram_comparison=None,
                             reparameterization=None, probe_comparison=None):
    """合并各分析层的种子级探索性检验，并写入 ``fdr_source``。

    只做结构收集，不判断 p/n 是否可检验；过滤与 q 值由
    :func:`exploratory_fdr` 完成。输入结构缺失或为坏类型时静默跳过，
    不伪造空检验。
    """
    out = []

    def add(items, source, source_fn=None):
        for t in (items if isinstance(items, list) else []):
            if not isinstance(t, dict):
                continue
            t["fdr_source"] = (source_fn(t) if source_fn is not None
                               else source)
            out.append(t)

    add(main_tests, "main", _main_source)
    add(causal_tests, "causal_lesion",
        lambda t: ("causal_matched_lesion"
                   if t.get("matched_control") else "causal_lesion"))
    add(matched_ablation_contrasts, "causal_matched_contrast")
    if isinstance(dose_tests, dict):
        add(dose_tests.get("tests"), "dose_response")
    add(inference_shapley, "inference_shapley")
    add(inference_interactions, "inference_interaction")
    if isinstance(capacity_decomposition, dict):
        add(capacity_decomposition.get("tests"), "capacity_decomposition")
    if isinstance(length_analysis, dict):
        add(length_analysis.get("ctx_tests"), "ruler_length_ctx")
        add(length_analysis.get("slope_tests"), "ruler_length_slope")
        add(length_analysis.get("span_tests"), "ruler_length_span")
    if isinstance(ngram_comparison, dict):
        add(ngram_comparison.get("rows"), "ngram_baseline")
    if isinstance(reparameterization, dict):
        metrics = reparameterization.get("metrics")
        if isinstance(metrics, dict):
            for block in metrics.values():
                if not isinstance(block, dict):
                    continue
                add(block.get("contrasts"), "reparameterization_contrast")
                add(block.get("shapley"), "reparameterization_shapley")
    if isinstance(probe_comparison, dict):
        add(probe_comparison.get("tests"), "probe_baseline")
    return out
