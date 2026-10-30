"""配对差的贝叶斯证据：小样本下“支持 H1 / 支持 H0 / 证据不足”的补充层。

为什么需要这一层
================
项目的主判定是频率学派的同种子配对 t 检验 + Holm 校正，显著性只回答
“能否拒绝 Δ=0”；``p>0.05`` 既可能是真实无效应，也可能只是种子少、功效低。
SESOI/TOST 回答“差异是否小于实际重要阈值”，复现预测回答“换个种子方向是否
稳定”，但都没有直接给出“当前数据相对支持哪个假设、证据有多强”。

本模块对配对差 ``d_i = 候选_i − 参照_i`` 拟合标准的一样本 JZS 模型：

    d_i ~ Normal(μ, σ²),   δ = μ/σ ~ Cauchy(0, r),   p(σ) ∝ 1/σ

其中 ``r`` 是标准化效应先验尺度（默认 ``1/√2``，Rouder et al. 2009 的
常用默认值）。在网格上对 ``(δ, σ)`` 做数值积分，输出：

* ``bf10`` / ``bf01``：H1（μ≠0）对 H0（μ=0）的贝叶斯因子及其倒数；
  ``bf10=1`` 表示数据不区分两个假设，不是“无效应”；
* ``posterior_prob_positive`` / ``posterior_prob_negative``：后验方向概率；
* ``posterior_median`` / ``posterior_mean`` / ``posterior_ci95``：μ 的后验
  位置与 95% 可信区间（单位与输入配对差一致）；
* 给定 ROPE（实际等价带）时，``prob_in_rope``、``prob_meaningful_positive``
  与 ``prob_meaningful_negative``：后验落在“等价带 / 同向超过 ROPE /
  反向超过 ROPE”三个互斥区域的质量；
* ``sensitivity``：先验尺度取 0.5×/1×/2× 时的 BF 与证据标签，防止只报
  一个对主张有利的先验。

边界必须如实声明
================
* 这是**补充性/描述性**证据，不改变预注册的 支持/拒绝/未定 判定，也不能
  替代 t 检验、置换检验、TOST 或 MDE；
* JZS 仍假设配对差独立且近似正态；3~5 个种子无法检验该假设。先验尺度是
  分析前选择，看到结果后再换先验属于事后选择；本模块固定报告敏感性网格，
  结论随先验翻转时必须并列披露，不能只挑一个 BF；
* BF10 不是“H1 为真的概率”，也不是 H0 的后验概率；它只比较两个模型在
  当前数据下的边际似然；
* ROPE 后验质量不是 TOST：TOST 用抽样分布控制错误率，这里只是后验质量；
  两者可以并读，不能互相替代；
* 零方差（所有配对差相同）是模型退化情形：H1 可用 σ→0 精确拟合，BF 无
  定义，本模块显式标为 ``degenerate_zero_variance``，不伪造证据；
* n<``min_seeds``（默认 3）时仍给出描述性后验，但 ``decision_eligible``
  为假，不能据此写“数据支持 H1/H0”。
"""
from __future__ import annotations

import math
from numbers import Integral, Real

import numpy as np

__all__ = ["bayes_paired_evidence", "evidence_label", "DEFAULT_PRIOR_SCALE"]

DEFAULT_PRIOR_SCALE = 1.0 / math.sqrt(2.0)
DEFAULT_PRIOR_FACTORS = (0.5, 1.0, 2.0)
DEFAULT_MIN_SEEDS = 3
_DELTA_GRID = 400
_SIGMA_GRID = 241
_SIGMA_SPAN = 12.0


def _finite(value):
    """有限实数观测；bool/字符串/非有限/不可表示值 -> None。"""
    if value is None or isinstance(value, (bool, np.bool_)):
        return None
    if isinstance(value, (str, bytes)) or not isinstance(value, Real):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return out if math.isfinite(out) else None


def _strict_positive(value, name):
    out = _finite(value)
    if out is None or out <= 0.0:
        raise ValueError(f"{name} 必须是有限正数，收到 {value!r}")
    return out


def _strict_int(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} 必须是整数，收到 {value!r}")
    value = int(value)
    if value < 2:
        raise ValueError(f"{name} 必须 >= 2，收到 {value}")
    return value


def _pair_values(diffs):
    """配对差 -> 有限 ndarray；坏值整项剔除，保持输入顺序。"""
    if isinstance(diffs, (str, bytes)) or diffs is None:
        return np.empty(0, dtype=float)
    try:
        raw = list(diffs)
    except TypeError:
        return np.empty(0, dtype=float)
    out = []
    for value in raw:
        f = _finite(value)
        if f is not None:
            out.append(f)
    return np.asarray(out, dtype=float)


def _logsumexp(values):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("-inf")
    peak = float(np.max(arr))
    if not math.isfinite(peak):
        return peak
    return float(peak + math.log(float(np.sum(np.exp(arr - peak)))))


def _safe_exp(log_value):
    if log_value is None or not math.isfinite(log_value):
        return None
    if log_value > 700.0:
        return None
    if log_value < -700.0:
        return 0.0
    out = math.exp(log_value)
    return out if math.isfinite(out) else None


def _mean_sd(values):
    """稳健均值/样本标准差；先按最大绝对值缩放，避免 1e308 级输入溢出。"""
    arr = np.asarray(values, dtype=float)
    if arr.size < 2:
        return None
    peak = float(np.max(np.abs(arr)))
    if not math.isfinite(peak) or peak == 0.0:
        return (0.0, 0.0) if peak == 0.0 else None
    scaled = arr / peak
    mean = float(np.mean(scaled)) * peak
    sd = float(np.std(scaled, ddof=1)) * peak
    if not math.isfinite(mean) or not math.isfinite(sd):
        return None
    return mean, sd


def _posterior_quantiles(values, weights, levels):
    order = np.argsort(values)
    v = np.asarray(values, dtype=float)[order]
    w = np.asarray(weights, dtype=float)[order]
    total = float(np.sum(w))
    if not math.isfinite(total) or total <= 0.0:
        return [None for _ in levels]
    cdf = np.cumsum(w) / total
    out = []
    for level in levels:
        idx = int(np.searchsorted(cdf, level, side="left"))
        idx = min(max(idx, 0), len(v) - 1)
        out.append(float(v[idx]))
    return out


def _grid(diffs, prior_scale, rope, n_delta, n_sigma):
    """在标准化数据上做网格积分；返回 BF、后验与诊断量。"""
    n = int(diffs.size)
    moments = _mean_sd(diffs)
    if moments is None:
        return None
    mean, sd = moments
    scale = max(abs(mean), sd)
    if not math.isfinite(scale) or scale <= 0.0:
        return None
    z = diffs / scale
    sum_z = float(np.sum(z))
    sd_z = float(np.std(z, ddof=1))
    if not math.isfinite(sd_z) or sd_z <= 0.0:
        return None
    ss0 = float(np.sum(z * z))

    log_sd = math.log(sd_z)
    log_sigma = np.linspace(log_sd - _SIGMA_SPAN, log_sd + _SIGMA_SPAN,
                            n_sigma)
    sigma = np.exp(log_sigma)
    loglik0 = (-0.5 * n * math.log(2.0 * math.pi) - n * log_sigma
               - ss0 / (2.0 * sigma ** 2))

    u = (np.arange(n_delta) + 0.5) / n_delta
    delta = prior_scale * np.tan(math.pi * (u - 0.5))
    mu_z = np.outer(delta, sigma)
    sse = ss0 - 2.0 * mu_z * sum_z + n * mu_z ** 2
    sse = np.maximum(sse, 0.0)
    loglik1 = (-0.5 * n * math.log(2.0 * math.pi) - n * log_sigma[None, :]
               - sse / (2.0 * sigma[None, :] ** 2))

    log_l0 = _logsumexp(loglik0) - math.log(n_sigma)
    log_l1 = (_logsumexp(loglik1.ravel())
              - math.log(n_sigma * n_delta))
    log_bf10 = log_l1 - log_l0

    peak = float(np.max(loglik1))
    weights = np.exp(loglik1 - peak)
    total = float(np.sum(weights))
    if not math.isfinite(total) or total <= 0.0:
        return None
    weights /= total
    wflat = weights.ravel()
    mflat = mu_z.ravel()
    # 数值网格可能恰有一个 δ=0 的节点（偶数点数通常不会）；连续后验在
    # 0 处没有原子，把它对称分给两侧，保证方向概率互补且不伪造偏向。
    zero_mass = float(np.sum(wflat[mflat == 0.0]))
    prob_positive = float(np.sum(wflat[mflat > 0.0])) + 0.5 * zero_mass
    prob_negative = float(np.sum(wflat[mflat < 0.0])) + 0.5 * zero_mass
    quantiles = _posterior_quantiles(
        mflat, wflat, (0.025, 0.5, 0.975))
    mean_z = float(np.sum(wflat * mflat))

    def _raw(value_z):
        if value_z is None or not math.isfinite(value_z):
            return None
        value = value_z * scale
        return float(value) if math.isfinite(value) else None

    out = {
        "log_bf10": float(log_bf10),
        "bf10": _safe_exp(log_bf10),
        "bf01": _safe_exp(-log_bf10),
        "posterior_mean": _raw(mean_z),
        "posterior_median": _raw(quantiles[1]),
        "posterior_ci95": [_raw(quantiles[0]), _raw(quantiles[2])],
        "posterior_prob_positive": prob_positive,
        "posterior_prob_negative": prob_negative,
        "standardized_effect": (mean / sd if sd > 0.0 else None),
    }
    if rope is not None:
        # 在标准化尺度上比较，避免 rope/μ 的原始值相乘溢出。
        rope_z = rope / scale
        inside = float(np.sum(wflat[np.abs(mflat) < rope_z]))
        out.update({
            "rope": float(rope),
            "prob_in_rope": inside,
            "prob_meaningful_positive": float(
                np.sum(wflat[mflat >= rope_z])),
            "prob_meaningful_negative": float(
                np.sum(wflat[mflat <= -rope_z])),
        })
    return out


def evidence_label(bf10):
    """Jeffreys 证据标签（双侧）。

    阈值只用于可读分类，不改变任何统计判定；``bf10`` 为 None（上溢/不可
    表示）时按 ``log_bf10`` 另行处理，不在这里猜测。
    """
    value = _finite(bf10)
    if value is None or value <= 0.0:
        return "无法计算"
    if value >= 100.0:
        return "极强支持 H1"
    if value >= 30.0:
        return "强支持 H1"
    if value >= 10.0:
        return "中等支持 H1"
    if value > 3.0:
        return "弱支持 H1"
    if value >= 1.0 / 3.0:
        return "证据不足（不区分 H0/H1）"
    if value > 1.0 / 10.0:
        return "弱支持 H0"
    if value > 1.0 / 30.0:
        return "中等支持 H0"
    if value > 1.0 / 100.0:
        return "强支持 H0"
    return "极强支持 H0"


def _label_from_log(log_bf10):
    if log_bf10 is None or not math.isfinite(log_bf10):
        return "无法计算"
    if log_bf10 >= math.log(100.0):
        return "极强支持 H1"
    if log_bf10 >= math.log(30.0):
        return "强支持 H1"
    if log_bf10 >= math.log(10.0):
        return "中等支持 H1"
    if log_bf10 > math.log(3.0):
        return "弱支持 H1"
    if log_bf10 >= -math.log(3.0):
        return "证据不足（不区分 H0/H1）"
    if log_bf10 > -math.log(10.0):
        return "弱支持 H0"
    if log_bf10 > -math.log(30.0):
        return "中等支持 H0"
    if log_bf10 > -math.log(100.0):
        return "强支持 H0"
    return "极强支持 H0"


def _prior_sensitivity(diffs, prior_scale, factors, rope):
    rows = []
    for factor in factors:
        scale = prior_scale * factor
        result = _grid(diffs, scale, rope, _DELTA_GRID, _SIGMA_GRID)
        if result is None:
            rows.append({"factor": float(factor), "prior_scale": float(scale),
                         "log_bf10": None, "bf10": None,
                         "label": "无法计算"})
            continue
        log_bf10 = result["log_bf10"]
        rows.append({
            "factor": float(factor),
            "prior_scale": float(scale),
            "log_bf10": log_bf10,
            "bf10": result["bf10"],
            "label": _label_from_log(log_bf10),
        })
    return rows


def _sensitivity_verdict(rows):
    labels = [r.get("label") for r in rows
              if r.get("label") not in (None, "无法计算")]
    if not labels:
        return "无法计算"
    if all(lbl.endswith("H1") for lbl in labels):
        return "各先验尺度均支持 H1"
    if all(lbl.endswith("H0") for lbl in labels):
        return "各先验尺度均支持 H0"
    if all(lbl == "证据不足（不区分 H0/H1）" for lbl in labels):
        return "各先验尺度均证据不足"
    if any(lbl == "证据不足（不区分 H0/H1）" for lbl in labels):
        return "先验尺度可改变证据强度"
    return "先验尺度可改变证据方向"


def bayes_paired_evidence(diffs, rope=None, prior_scale=DEFAULT_PRIOR_SCALE,
                          prior_scales=None, min_seeds=DEFAULT_MIN_SEEDS,
                          n_delta=_DELTA_GRID, n_sigma=_SIGMA_GRID):
    """配对差的 JZS 贝叶斯因子与后验（补充性证据）。

    参数
    ----
    diffs : 序列
        配对差（候选−参照）。None/bool/非有限值整项剔除。
    rope : float, optional
        实际等价带半宽，单位与 ``diffs`` 一致；给出时报告后验落在
        ``(−rope, rope)`` 的质量。不改变 BF。
    prior_scale : float
        标准化效应 δ 的 Cauchy 先验尺度（默认 1/√2）。必须在看到数据前
        固定；本模块另报 0.5×/1×/2× 敏感性。
    prior_scales : 序列, optional
        自定义敏感性因子（相对于 ``prior_scale``，默认 0.5/1/2）。
    min_seeds : int
        判定资格下限（默认 3）；n 低于该值时只给描述性后验。
    n_delta, n_sigma : int
        网格点数（测试/极端输入可调；默认值已足够稳定）。

    返回 dict；无法计算时返回带 ``status``/``reason`` 的结构，不抛异常
    （非法参数仍抛 ValueError，便于调用方发现配置错误）。
    """
    prior_scale = _strict_positive(prior_scale, "prior_scale")
    if rope is not None:
        rope = _strict_positive(rope, "rope")
    min_seeds = _strict_int(min_seeds, "min_seeds")
    n_delta = _strict_int(n_delta, "n_delta")
    n_sigma = _strict_int(n_sigma, "n_sigma")
    if prior_scales is None:
        factors = list(DEFAULT_PRIOR_FACTORS)
    else:
        if isinstance(prior_scales, (str, bytes)):
            raise ValueError("prior_scales 必须是正数序列")
        try:
            raw_factors = list(prior_scales)
        except TypeError as exc:
            raise ValueError(
                f"prior_scales 必须是正数序列，收到 {prior_scales!r}") from exc
        factors = []
        for factor in raw_factors:
            if (isinstance(factor, (bool, np.bool_))
                    or isinstance(factor, (str, bytes))
                    or not isinstance(factor, Real)):
                raise ValueError(
                    f"prior_scales 因子必须是有限正数，收到 {factor!r}")
            factors.append(_strict_positive(factor, "prior_scales 因子"))
        if not factors:
            raise ValueError("prior_scales 不能为空")

    values = _pair_values(diffs)
    n = int(values.size)
    base = {
        "n": n,
        "prior_scale": float(prior_scale),
        "min_seeds": int(min_seeds),
        "method": ("one-sample JZS Bayes factor and posterior on paired "
                   "differences (Cauchy prior on standardized effect, "
                   "Jeffreys prior on scale)"),
        "note": ("补充性证据：不改变预注册判定，也不替代 t/置换/TOST/MDE；"
                 "BF10=1 表示不区分 H0/H1，不是无效应。"),
        "decision_eligible": bool(n >= min_seeds),
    }
    if n < 2:
        base.update({
            "status": "insufficient_seeds",
            "reason": "配对差少于 2 个，无法估计尺度",
            "bf10": None, "bf01": None, "log_bf10": None,
            "label": "无法计算",
            "posterior_mean": None, "posterior_median": None,
            "posterior_ci95": None,
            "posterior_prob_positive": None,
            "posterior_prob_negative": None,
            "standardized_effect": None,
            "sensitivity": [], "sensitivity_verdict": "无法计算",
        })
        return base

    moments = _mean_sd(values)
    if moments is None:
        base.update({
            "status": "invalid",
            "reason": "配对差的均值/标准差超出 float 表示范围",
            "bf10": None, "bf01": None, "log_bf10": None,
            "label": "无法计算",
            "posterior_mean": None, "posterior_median": None,
            "posterior_ci95": None,
            "posterior_prob_positive": None,
            "posterior_prob_negative": None,
            "standardized_effect": None,
            "sensitivity": [], "sensitivity_verdict": "无法计算",
        })
        return base
    mean, sd = moments
    if sd == 0.0:
        prob = 1.0 if mean > 0.0 else (0.0 if mean < 0.0 else 0.5)
        base.update({
            "status": "degenerate_zero_variance",
            "reason": ("所有配对差相同：H1 可用 σ→0 精确拟合，BF 无定义；"
                       "不能当作强证据"),
            "bf10": None, "bf01": None, "log_bf10": None,
            "label": "无法计算",
            "posterior_mean": mean, "posterior_median": mean,
            "posterior_ci95": [mean, mean],
            "posterior_prob_positive": prob,
            "posterior_prob_negative": 1.0 - prob,
            "standardized_effect": None,
            "sensitivity": [], "sensitivity_verdict": "无法计算",
        })
        if rope is not None:
            base.update({
                "rope": float(rope),
                "prob_in_rope": float(abs(mean) < rope),
                "prob_meaningful_positive": float(mean >= rope),
                "prob_meaningful_negative": float(mean <= -rope),
            })
        return base

    result = _grid(values, prior_scale, rope, n_delta, n_sigma)
    if result is None:
        base.update({
            "status": "invalid",
            "reason": "后验网格不可归一化（输入过大/不可表示）",
            "bf10": None, "bf01": None, "log_bf10": None,
            "label": "无法计算",
            "posterior_mean": None, "posterior_median": None,
            "posterior_ci95": None,
            "posterior_prob_positive": None,
            "posterior_prob_negative": None,
            "standardized_effect": None,
            "sensitivity": [], "sensitivity_verdict": "无法计算",
        })
        return base

    sensitivity = _prior_sensitivity(
        values, prior_scale, factors, rope)
    result.update({
        "status": "ok" if n >= min_seeds else "insufficient_seeds",
        "reason": (None if n >= min_seeds
                   else f"n={n}<{min_seeds}，只作描述性后验"),
        "label": _label_from_log(result["log_bf10"]),
        "sensitivity": sensitivity,
        "sensitivity_verdict": _sensitivity_verdict(sensitivity),
    })
    base.update(result)
    return base
