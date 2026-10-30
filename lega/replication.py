"""新种子复现预测：显著结果在下一次独立训练种子上能否重现。

为什么需要这一层
================
项目的主判定是跨种子配对 t 检验 + Holm 校正。它回答的是“在当前这组种子上，
平均配对差能否拒绝 0”，而不是“再抽一个新种子，候选是否仍会赢”。二者在小
样本下差别很大：

* 均值的置信区间随 ``1/sqrt(n)`` 收缩，描述的是**均值**的不确定性；
* 单个新种子的预测区间还要加上种子自身方差，宽度按
  ``sqrt(1 + 1/n)`` 放大，n=3~5 时明显更宽；
* 把 95% CI 当作“下一个种子有 95% 概率落在其中”会系统性过度自信。

本模块因此为每组配对差 ``d_i = 候选_i − 参照_i`` 给出：

* ``prediction_interval``：同一配置/数据/评测协议下，换一个训练种子的
  t 预测区间 ``mean ± t_{n−1,1−α/2} · sd · sqrt(1+1/n)``；
* ``prob_same_sign``：在该预测分布下，新种子与已观测均值同号的概率
  ``F_t(|mean| / (sd·sqrt(1+1/n)); n−1)``；
* 若给定 SESOI，还给出新种子落在“实际重要方向”与“±SESOI 等价带”内的
  预测概率；
* ``loo_sign_agreement``：不含分布假设的留一种子方向一致率，作为对 t
  预测的朴素对照。

边界
====
* 这是**描述性/探索性**预测，不是新的确认性检验，也不改变预注册判定；
* 模型假设种子效应近似可交换、独立且正态。3~5 个种子无法检验该假设；
  离散/重尾或系统性种子×数据交互会让概率失真；
* 只回答“同配置、同数据、同评测协议下换训练种子”，不外推到新数据、
  新超参数、新硬件或更长上下文；
* n<3 与项目其余协议一致，不作判定；零方差（所有种子配对差完全相同）
  是模型退化情形，单独标记，不能当作强复现证据。
"""
import math
from numbers import Real

from .utils import mean_std, student_t_two_sided_p

DEFAULT_ALPHA = 0.05
MIN_SEEDS = 3

_STATUS_CN = {
    "insufficient_seeds": "种子不足（n<3），不作复现预测",
    "invalid": "无法计算（均值/方差不可表示或输入无效）",
    "degenerate_zero_variance": "零方差退化（所有配对差相同），预测不可当强证据",
    "undetermined": "方向未定（新种子同号概率 <0.8）",
    "directionally_plausible": "方向可复现（新种子同号概率 ≥0.8）",
    "directionally_consistent": "方向稳定可复现（新种子同号概率 ≥0.95）",
}


def _finite_observation(value):
    """有限实数观测；bool/字符串/非有限/不可表示值按缺失处理。"""
    if value is None or isinstance(value, bool):
        return None
    if type(value).__name__ == "bool_":
        return None
    if isinstance(value, str) or not isinstance(value, Real):
        return None
    try:
        f = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return f if math.isfinite(f) else None


def _strict_int(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} 必须是整数，收到 {value!r}")
    return value


def _strict_alpha(value) -> float:
    f = _finite_observation(value)
    if f is None or not 0.0 < f < 1.0:
        raise ValueError(f"alpha 必须在 (0, 1) 内，收到 {value!r}")
    return f


def _strict_margin(value) -> float:
    f = _finite_observation(value)
    if f is None or f <= 0.0:
        raise ValueError(f"SESOI margin 必须 > 0，收到 {value!r}")
    return f


def _t_cdf(x: float, df: float):
    """Student-t 分布 CDF；复用双侧 p 值实现，避免引入 scipy。"""
    if math.isnan(x) or not math.isfinite(df) or df <= 0.0:
        return None
    if x == float("inf"):
        return 1.0
    if x == float("-inf"):
        return 0.0
    if not math.isfinite(x):
        return None
    p2 = student_t_two_sided_p(x, df)
    if p2 is None or not math.isfinite(p2):
        return None
    return 1.0 - p2 / 2.0 if x >= 0.0 else p2 / 2.0


def _t_critical(df: float, alpha: float):
    """Student-t 双侧临界值（二分求根）；无法可靠表示时返回 None。"""
    if not math.isfinite(df) or df <= 0.0 or not 0.0 < alpha < 1.0:
        return None
    lo, hi = 0.0, 1.0
    while student_t_two_sided_p(hi, df) > alpha:
        if hi >= 1e12:
            return None
        hi = min(hi * 2.0, 1e12)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if student_t_two_sided_p(mid, df) > alpha:
            lo = mid
        else:
            hi = mid
        if hi - lo <= 1e-12 * max(1.0, hi):
            break
    root = 0.5 * (lo + hi)
    if abs(student_t_two_sided_p(root, df) - alpha) > 1e-6:
        return None
    return root


def _clamp01(value):
    f = _finite_observation(value)
    if f is None:
        return None
    return min(1.0, max(0.0, f))


def _loo_sign_agreement(diffs):
    """留一种子方向一致率：删掉 i 后均值符号与 d_i 符号一致的比例。

    零差或留一均值为 0 时按 0.5 计入（无方向信息）。该量不使用正态假设，
    只作 t 预测的朴素对照，不是独立的复现概率估计。
    """
    n = len(diffs)
    if n < 2:
        return None
    score = 0.0
    for i, d in enumerate(diffs):
        rest = diffs[:i] + diffs[i + 1:]
        rest_mean, _ = mean_std(rest)
        if rest_mean is None or rest_mean == 0.0 or d == 0.0:
            score += 0.5
        elif (rest_mean > 0.0) == (d > 0.0):
            score += 1.0
    return score / float(n)


def _empty_result(n: int, alpha: float, min_seeds: int, margin):
    return {
        "method": "t 预测区间 + 正态种子模型下的同号概率（描述性）",
        "n": int(n), "df": None, "alpha": alpha, "level": 1.0 - alpha,
        "min_seeds": min_seeds,
        "mean_diff": None, "sd_diff": None, "se_mean": None,
        "prediction_sd": None, "prediction_interval": None,
        "pi_excludes_zero": None, "expected_sign": None,
        "prob_same_sign": None, "prob_opposite_sign": None,
        "sesoi": margin, "prob_meaningful_same_direction": None,
        "prob_meaningful_opposite_direction": None,
        "prob_within_sesoi": None,
        "loo_sign_agreement": None,
        "zero_variance": None,
        "status": "invalid",
        "label": _STATUS_CN["invalid"],
        "note": ("预测的是“同配置/同数据/同评测协议下换一个训练种子”，"
                 "不外推到新数据、新超参数或新硬件；描述性证据，不改变"
                 "预注册判定。"),
    }


def replication_forecast(values, margin=None, alpha: float = DEFAULT_ALPHA,
                         min_seeds: int = MIN_SEEDS) -> dict:
    """对一组（已按指标尺度变换的）配对差做新种子复现预测。

    ``values`` 为 ``d_i = 候选_i − 参照_i``；调用方负责先把 PPL 转成相对差/
    log 比等与分析尺度一致的形式。PPL 的原始单位、相对单位与 log 单位会给出
    不同的 SESOI 概率，不能混用。

    返回字段见模块文档；所有概率均为模型预测值，不是频率学置信度。输入中的
    None/bool/非有限值整项剔除；有效 n<min_seeds 或统计量不可表示时返回
    ``status=insufficient_seeds``/``invalid``，绝不伪造成 0 或 1。
    """
    alpha = _strict_alpha(alpha)
    min_seeds = _strict_int(min_seeds, "min_seeds")
    if min_seeds < 1:
        raise ValueError(f"min_seeds 必须 >= 1，收到 {min_seeds!r}")
    margin = None if margin is None else _strict_margin(margin)

    vals = []
    for v in (values if isinstance(values, (list, tuple)) else list(values)):
        f = _finite_observation(v)
        if f is not None:
            vals.append(f)
    n = len(vals)
    out = _empty_result(n, alpha, min_seeds, margin)
    if n == 0:
        out["status"] = "insufficient_seeds"
        out["label"] = _STATUS_CN["insufficient_seeds"]
        return out

    mean, sd = mean_std(vals)
    out["mean_diff"] = mean
    out["sd_diff"] = sd
    out["loo_sign_agreement"] = _loo_sign_agreement(vals)
    if mean is None:
        return out
    out["expected_sign"] = ("positive" if mean > 0.0 else
                            "negative" if mean < 0.0 else "zero")
    if n < min_seeds:
        out["status"] = "insufficient_seeds"
        out["label"] = _STATUS_CN["insufficient_seeds"]
        return out
    if sd is None:
        # n 足够大但样本标准差不可表示（极端有限值相减溢出）：与“种子
        # 不足”区分，不能把数值失败伪装成需要补种子。
        return out

    df = n - 1
    out["df"] = df
    out["se_mean"] = sd / math.sqrt(n) if sd > 0.0 else 0.0
    out["zero_variance"] = bool(sd == 0.0)

    if sd == 0.0:
        out["prediction_sd"] = 0.0
        out["prediction_interval"] = [mean, mean]
        out["pi_excludes_zero"] = bool(mean != 0.0)
        out["prob_same_sign"] = 1.0 if mean != 0.0 else 0.5
        out["prob_opposite_sign"] = 1.0 - out["prob_same_sign"]
        out["status"] = "degenerate_zero_variance"
        out["label"] = _STATUS_CN["degenerate_zero_variance"]
    else:
        try:
            pred_sd = sd * math.sqrt(1.0 + 1.0 / n)
        except (OverflowError, ValueError):
            pred_sd = None
        out["prediction_sd"] = pred_sd
        tcrit = _t_critical(df, alpha)
        if pred_sd is None or not math.isfinite(pred_sd) or tcrit is None:
            return out
        half = tcrit * pred_sd
        lo, hi = mean - half, mean + half
        if math.isfinite(lo) and math.isfinite(hi):
            out["prediction_interval"] = [lo, hi]
            out["pi_excludes_zero"] = bool(lo > 0.0 or hi < 0.0)
        z = abs(mean) / pred_sd if pred_sd > 0.0 else None
        if z is None:
            return out
        prob = _t_cdf(z, df)
        if prob is None:
            return out
        out["prob_same_sign"] = _clamp01(prob)
        out["prob_opposite_sign"] = _clamp01(1.0 - prob)
        if (out["pi_excludes_zero"] or out["prob_same_sign"] >= 0.95):
            out["status"] = "directionally_consistent"
        elif out["prob_same_sign"] >= 0.8:
            out["status"] = "directionally_plausible"
        else:
            out["status"] = "undetermined"
        out["label"] = _STATUS_CN[out["status"]]
        if out["prediction_interval"] is None:
            out["status"] = "invalid"
            out["label"] = _STATUS_CN["invalid"]

    if margin is not None:
        pred_sd = out.get("prediction_sd")
        if pred_sd == 0.0:
            within = 1.0 if abs(mean) < margin else 0.0
            same = (mean >= margin if mean > 0.0 else
                    mean <= -margin if mean < 0.0 else 0.0)
            out["prob_meaningful_same_direction"] = 1.0 if same else 0.0
            out["prob_within_sesoi"] = within
            out["prob_meaningful_opposite_direction"] = 1.0 - within - (
                1.0 if same else 0.0)
        elif pred_sd is not None and math.isfinite(pred_sd) and pred_sd > 0.0:
            c_hi = _t_cdf((margin - mean) / pred_sd, df)
            c_lo = _t_cdf((-margin - mean) / pred_sd, df)
            if c_hi is not None and c_lo is not None:
                out["prob_within_sesoi"] = _clamp01(c_hi - c_lo)
                if mean >= 0.0:
                    out["prob_meaningful_same_direction"] = _clamp01(1.0 - c_hi)
                    out["prob_meaningful_opposite_direction"] = _clamp01(c_lo)
                else:
                    out["prob_meaningful_same_direction"] = _clamp01(c_lo)
                    out["prob_meaningful_opposite_direction"] = _clamp01(
                        1.0 - c_hi)
    return out
