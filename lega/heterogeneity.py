"""分层效应异质性：RULER macro 平均是否代表各任务/长度的共同效应。

只报 ``RULER平均`` 时，一个显著（或不显著）的 macro 配对差可能由两种完全
不同的数据结构产生：所有任务/长度都同向小幅变化，或某些层大涨、另一些层
反向变化后相互抵消。前者支持“平均效应”的读法，后者说明平均不是任何一个
层的共同效应。本模块用**随机效应 meta 分析**把这种异质性显式化：

* 固定效应合并 ``pooled_fixed`` 与随机效应合并 ``pooled_random``；
* Cochran's Q、I²、tau² 描述层间效应方差；
* 随机效应 95% 区间与（k>=3 时的）95% 预测区间：前者是合并均值的精度，
  后者回答“换一个同质层，效应可能落在哪里”；
* 逐个删除一个层后的合并效应，检查结论是否由单个层驱动；
* 按**种子**重采样的 bootstrap 区间，保留同一种子内各层的相关性。

边界
====
* 这里的“层”是 RULER 任务或上下文长度档位。它们**不是独立研究**：同一种子
  同时贡献所有层，上下文层还嵌套在同一批评测样例里。因此 Q 的卡方尾概率
  和 I² 只作描述性异质性指标，不能当作独立研究的正式检验；bootstrap 只
  覆盖训练种子相关性，不覆盖层间共享评测样例。
* 任务层与长度层回答不同问题，不能把两套层混进同一个 meta 分析重复计数。
* 本模块为探索性结果分析，未预注册，不改变 H1–H7 的确认性判定，也不
  替代逐层检验、SESOI/等价性或 MDE。
* “未检出异质性”不等于各层效应相同；I² 高也不表示某个层显著。最终仍须
  回到逐层点估计/区间解释“效应来自哪里”。

本模块只依赖 Python 标准库与 NumPy；输入检查对 bool/字符串/NaN/Inf 严格，
坏层显式剔除并记账，不按 0 效应补齐。
"""
import math
from numbers import Real

import numpy as np

from .utils import student_t_two_sided_p

DEFAULT_ALPHA = 0.05
MIN_STRATA = 2
DEFAULT_BOOTSTRAP = 2000

_STATUS_CN = {
    "insufficient": "层数不足，无法评估异质性",
    "invalid": "输入无效，无法评估异质性",
    "degenerate_zero_variance": "层内方差全为 0 且效应不全相同，异质性无法用随机效应模型表示",
    "no_detected_heterogeneity": "未检出异质性（不等于各层同质）",
    "moderate_heterogeneity": "检出异质性（I²<50%）",
    "heterogeneous": "检出显著异质性（I²≥50%）",
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


def _strict_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} 必须是整数，收到 {value!r}")
    return int(value)


def _strict_alpha(value):
    out = _finite(value)
    if out is None or not 0.0 < out < 1.0:
        raise ValueError(f"alpha 必须在 (0, 1) 内，收到 {value!r}")
    return out


def _safe_float(value):
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return out if math.isfinite(out) else None


def _gammaincc(a, x):
    """正则化上不完全伽马函数 Q(a, x)（Numerical Recipes 级数/连分式）。"""
    if not (math.isfinite(a) and math.isfinite(x)) or a <= 0.0 or x < 0.0:
        raise ValueError("gammaincc 需要 a>0 且 x>=0 的有限值")
    if x == 0.0:
        return 1.0
    log_prefactor = a * math.log(x) - x - math.lgamma(a)
    if x < a + 1.0:
        ap = a
        term = 1.0 / a
        total = term
        for _ in range(1000):
            ap += 1.0
            term *= x / ap
            total += term
            if abs(term) <= abs(total) * 1e-15:
                break
        p = total * math.exp(log_prefactor)
        return min(1.0, max(0.0, 1.0 - p))
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b if abs(b) > tiny else 1.0 / tiny
    h = d
    for i in range(1, 1000):
        an = -float(i) * (float(i) - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) <= 1e-15:
            break
    q = math.exp(log_prefactor) * h
    return min(1.0, max(0.0, q))


def chi2_sf(x, df):
    """卡方分布上尾概率 P(X >= x)；用于 Cochran's Q 的描述性 p 值。"""
    x = _finite(x)
    df = _finite(df)
    if x is None or df is None or df <= 0.0 or x < 0.0:
        raise ValueError(f"chi2_sf 需要 x>=0 且 df>0，收到 x={x!r}, df={df!r}")
    return _gammaincc(df / 2.0, x / 2.0)


def _t_critical(df, alpha):
    """双侧 t 临界值；df<=0 时返回 None。"""
    if df is None or df <= 0.0:
        return None
    target = alpha
    lo, hi = 0.0, 1.0
    for _ in range(60):
        if student_t_two_sided_p(hi, df) <= target:
            break
        hi *= 2.0
        if hi > 1e9:
            return None
    for _ in range(100):
        mid = (lo + hi) / 2.0
        if student_t_two_sided_p(mid, df) > target:
            lo = mid
        else:
            hi = mid
    out = (lo + hi) / 2.0
    return out if math.isfinite(out) else None


def _mean(values):
    return float(math.fsum(values) / len(values))


def _sample_sd(values):
    if len(values) < 2:
        return None
    mean = _mean(values)
    var = math.fsum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(var) if var >= 0.0 and math.isfinite(var) else None


def _sign(value):
    if value is None:
        return 0
    if value > 0.0:
        return 1
    if value < 0.0:
        return -1
    return 0


def _empty_result(k, labels, alpha, min_k, excluded, reason, status="insufficient"):
    return {
        "status": status,
        "label": _STATUS_CN.get(status, status),
        "reason": reason,
        "k": k,
        "df": max(k - 1, 0),
        "alpha": alpha,
        "min_k": min_k,
        "labels": list(labels),
        "per_stratum": [],
        "excluded": excluded,
        "pooled_fixed": None,
        "se_fixed": None,
        "ci95_fixed": None,
        "pooled_random": None,
        "se_random": None,
        "ci95_random": None,
        "prediction_interval": None,
        "q": None,
        "q_p": None,
        "i2": None,
        "i2_pct": None,
        "h2": None,
        "tau2": None,
        "tau": None,
        "leave_one_out": [],
        "loo_pooled_random_range": None,
        "loo_i2_range": None,
        "loo_sign_stable": None,
        "loo_max_abs_shift_label": None,
        "method": "random-effects meta-analysis (DerSimonian-Laird)",
        "note": None,
    }


def meta_analysis(effects, ses, labels=None, alpha=DEFAULT_ALPHA,
                  min_k=MIN_STRATA, leave_one_out=True):
    """对分层效应做随机效应 meta 分析，并给出异质性与留一诊断。

    ``effects[i]`` 是第 i 层的效应（候选−参照，统一方向由调用方负责），
    ``ses[i]`` 是同一层效应估计的标准误。两者必须等长；缺失/非有限效应或
    非正标准误的层会被显式剔除并记入 ``excluded``，不按 0 补齐。

    ``min_k`` 是允许计算合并效应的最少层数（默认 2，便于两层探索性展示；
    预测区间仍要求 k>=3）。``leave_one_out=False`` 跳过留一层诊断，供
    bootstrap 等不需要该字段的重复调用使用。返回结构为 JSON 安全的基本
    类型；所有推断仅作描述性异质性诊断。
    """
    alpha = _strict_alpha(alpha)
    min_k = _strict_int(min_k, "min_k")
    if min_k < 2:
        raise ValueError(f"min_k 必须 >= 2，收到 {min_k}")
    if not isinstance(leave_one_out, bool):
        raise ValueError(
            f"leave_one_out 必须是 bool，收到 {leave_one_out!r}")
    try:
        effect_values = list(effects)
        se_values = list(ses)
    except TypeError as exc:
        raise ValueError("effects/ses 必须可迭代") from exc
    if len(effect_values) != len(se_values):
        raise ValueError(
            f"effects/ses 长度不一致：{len(effect_values)} != {len(se_values)}")
    if labels is None:
        label_values = [f"stratum_{i + 1}" for i in range(len(effect_values))]
    else:
        try:
            label_values = list(labels)
        except TypeError as exc:
            raise ValueError("labels 必须可迭代") from exc
        if len(label_values) != len(effect_values):
            raise ValueError(
                f"labels 长度不一致：{len(label_values)} != {len(effect_values)}")
    norm_labels = []
    for i, label in enumerate(label_values):
        text = str(label) if label is not None else f"stratum_{i + 1}"
        if not text.strip():
            text = f"stratum_{i + 1}"
        norm_labels.append(text)
    if len(set(norm_labels)) != len(norm_labels):
        raise ValueError("labels 必须互不重复")

    kept_effects, kept_ses, kept_labels = [], [], []
    excluded = []
    for idx, (effect, se, label) in enumerate(
            zip(effect_values, se_values, norm_labels)):
        fe, fs = _finite(effect), _finite(se)
        if fe is None:
            excluded.append({"index": idx, "label": label,
                             "reason": "effect 非有限或不可表示"})
            continue
        if fs is None or fs <= 0.0:
            excluded.append({"index": idx, "label": label,
                             "reason": "se 必须为有限正数"})
            continue
        kept_effects.append(fe)
        kept_ses.append(fs)
        kept_labels.append(label)

    k = len(kept_effects)
    if k == 0:
        return _empty_result(k, norm_labels, alpha, min_k, excluded,
                             "没有同时具备有限效应与正标准误的层", "invalid")
    if k < min_k:
        return _empty_result(
            k, kept_labels, alpha, min_k, excluded,
            f"有效层数 k={k} < min_k={min_k}，不计算合并效应")

    weights = [1.0 / (se * se) for se in kept_ses]
    sw = math.fsum(weights)
    if not math.isfinite(sw) or sw <= 0.0:
        return _empty_result(k, kept_labels, alpha, min_k, excluded,
                             "固定效应权重不可表示", "invalid")
    weighted = [w * y for w, y in zip(weights, kept_effects)]
    pooled_fixed = math.fsum(weighted) / sw
    se_fixed = math.sqrt(1.0 / sw)
    q = math.fsum(w * (y - pooled_fixed) ** 2
                  for w, y in zip(weights, kept_effects))
    df = k - 1
    q_p = chi2_sf(q, df) if math.isfinite(q) else None
    i2 = (max(0.0, 1.0 - df / q) if q > 0.0 and math.isfinite(q) else 0.0)
    h2 = (q / df if df > 0 and math.isfinite(q) else None)
    c = sw - math.fsum(w * w for w in weights) / sw
    tau2 = (max(0.0, (q - df) / c) if c > 0.0 and math.isfinite(q) else 0.0)
    if not math.isfinite(tau2):
        tau2 = 0.0

    random_weights = [1.0 / (se * se + tau2) for se in kept_ses]
    sw_random = math.fsum(random_weights)
    if not math.isfinite(sw_random) or sw_random <= 0.0:
        return _empty_result(k, kept_labels, alpha, min_k, excluded,
                             "随机效应权重不可表示", "invalid")
    pooled_random = math.fsum(
        w * y for w, y in zip(random_weights, kept_effects)) / sw_random
    se_random = math.sqrt(1.0 / sw_random)
    t_crit_mean = _t_critical(df, alpha)

    def _ci(mean, se):
        if mean is None or se is None or t_crit_mean is None:
            return None
        lo, hi = mean - t_crit_mean * se, mean + t_crit_mean * se
        return [lo, hi] if math.isfinite(lo) and math.isfinite(hi) else None

    prediction_interval = None
    prediction_df = k - 2
    t_crit_pred = _t_critical(prediction_df, alpha) if prediction_df > 0 else None
    if t_crit_pred is not None:
        half = t_crit_pred * math.sqrt(tau2 + se_random * se_random)
        lo, hi = pooled_random - half, pooled_random + half
        if math.isfinite(lo) and math.isfinite(hi):
            prediction_interval = [lo, hi]

    if q_p is not None and q_p < alpha:
        status = ("heterogeneous" if i2 >= 0.5
                  else "moderate_heterogeneity")
    else:
        status = "no_detected_heterogeneity"

    per_stratum = []
    for label, effect, se, w, rw in zip(
            kept_labels, kept_effects, kept_ses, weights, random_weights):
        per_stratum.append({
            "label": label,
            "effect": effect,
            "se": se,
            "weight_fixed": w,
            "weight_random": rw,
        })

    loo_rows = []
    if leave_one_out and k > 2:
        for j, label in enumerate(kept_labels):
            sub_effects = [y for i, y in enumerate(kept_effects) if i != j]
            sub_ses = [s for i, s in enumerate(kept_ses) if i != j]
            sub = meta_analysis(sub_effects, sub_ses,
                                labels=[x for i, x in enumerate(kept_labels)
                                        if i != j],
                                alpha=alpha, min_k=2, leave_one_out=False)
            estimate = _safe_float(sub.get("pooled_random"))
            i2_sub = _safe_float(sub.get("i2"))
            loo_rows.append({
                "label": label,
                "pooled_random": estimate,
                "i2": i2_sub,
                "shift": (estimate - pooled_random
                          if estimate is not None else None),
            })
    loo_estimates = [row["pooled_random"] for row in loo_rows
                     if row["pooled_random"] is not None]
    loo_i2 = [row["i2"] for row in loo_rows
              if row["i2"] is not None]
    loo_sign_stable = None
    if loo_estimates and abs(pooled_random) > 1e-12:
        sign = _sign(pooled_random)
        loo_sign_stable = all(_sign(v) == sign for v in loo_estimates)
    shift_row = None
    for row in loo_rows:
        shift = row.get("shift")
        if shift is not None and (shift_row is None
                                  or abs(shift) > abs(shift_row["shift"])):
            shift_row = row

    note = ("随机效应 meta 分析（DerSimonian-Laird）。层间共享种子且层内共享"
            "评测样例，Q 的卡方 p 值与 I² 仅作描述性异质性指标；"
            "预测区间回答“换一个同质层效应可能落在哪里”，不是新层效应的"
            "置信区间。未检出异质性不等于各层效应相同。")
    return {
        "status": status,
        "label": _STATUS_CN.get(status, status),
        "reason": None,
        "k": k,
        "df": df,
        "alpha": alpha,
        "min_k": min_k,
        "labels": kept_labels,
        "per_stratum": per_stratum,
        "excluded": excluded,
        "pooled_fixed": pooled_fixed,
        "se_fixed": se_fixed,
        "ci95_fixed": _ci(pooled_fixed, se_fixed),
        "pooled_random": pooled_random,
        "se_random": se_random,
        "ci95_random": _ci(pooled_random, se_random),
        "prediction_interval": prediction_interval,
        "q": q,
        "q_p": q_p,
        "i2": i2,
        "i2_pct": i2 * 100.0,
        "h2": h2,
        "tau2": tau2,
        "tau": math.sqrt(tau2) if math.isfinite(tau2) else None,
        "leave_one_out": loo_rows,
        "loo_pooled_random_range": ((min(loo_estimates), max(loo_estimates))
                                    if loo_estimates else None),
        "loo_i2_range": ((min(loo_i2), max(loo_i2)) if loo_i2 else None),
        "loo_sign_stable": loo_sign_stable,
        "loo_max_abs_shift_label": (shift_row["label"] if shift_row else None),
        "method": "random-effects meta-analysis (DerSimonian-Laird)",
        "note": note,
    }


def _percentile_ci(values, alpha):
    if not values:
        return None
    arr = np.asarray(values, dtype=float)
    lo = float(np.percentile(arr, 100.0 * alpha / 2.0))
    hi = float(np.percentile(arr, 100.0 * (1.0 - alpha / 2.0)))
    return [lo, hi] if math.isfinite(lo) and math.isfinite(hi) else None


def meta_analysis_from_seed_matrix(values, labels=None, alpha=DEFAULT_ALPHA,
                                   min_k=MIN_STRATA,
                                   n_bootstrap=DEFAULT_BOOTSTRAP,
                                   seed=0):
    """从“种子 × 层”效应矩阵计算 meta 分析，并按种子 bootstrap。

    ``values`` 每行是一个种子、每列是一个层的效应（候选−参照）。矩阵必须
    完整且有限：行列缺失会让“同一层”的种子集合不同，跨层可比性无法保证；
    调用方应先对齐/剔除。每层的标准误取跨种子样本标准差 / sqrt(n_seeds)。
    bootstrap 重采样**整行种子**，保留同一种子内各层相关性；它不覆盖层间
    共享评测样例，仍只是探索性不确定性刻画。
    """
    alpha = _strict_alpha(alpha)
    min_k = _strict_int(min_k, "min_k")
    n_bootstrap = _strict_int(n_bootstrap, "n_bootstrap")
    if min_k < 2:
        raise ValueError(f"min_k 必须 >= 2，收到 {min_k}")
    if n_bootstrap < 0:
        raise ValueError(f"n_bootstrap 必须 >= 0，收到 {n_bootstrap}")
    seed = _strict_int(seed, "seed")
    try:
        rows = [list(row) for row in values]
    except TypeError as exc:
        raise ValueError("values 必须是二维可迭代对象") from exc
    if not rows:
        raise ValueError("values 不能为空")
    width = len(rows[0])
    if width < 1:
        raise ValueError("values 每行至少需要 1 列")
    if any(len(row) != width for row in rows):
        raise ValueError("values 必须是矩形矩阵")
    if labels is None:
        labels = [f"stratum_{i + 1}" for i in range(width)]
    else:
        labels = list(labels)
        if len(labels) != width:
            raise ValueError(
                f"labels 长度 {len(labels)} != 列数 {width}")
    matrix = np.empty((len(rows), width), dtype=float)
    for i, row in enumerate(rows):
        for j, value in enumerate(row):
            finite = _finite(value)
            if finite is None:
                raise ValueError(
                    f"values[{i}][{j}] 不是有限实数：{value!r}；"
                    "调用方必须先对齐/剔除缺失层")
            matrix[i, j] = finite

    n_seeds = matrix.shape[0]
    effects, ses = [], []
    for j in range(width):
        col = matrix[:, j]
        mean = float(np.mean(col))
        effects.append(mean)
        sd = float(np.std(col, ddof=1)) if n_seeds >= 2 else None
        tol = 1e-12 * max(1.0, abs(mean))
        ses.append((sd / math.sqrt(n_seeds)
                    if sd is not None and sd > tol else 0.0))
    if n_seeds < 2:
        res = _empty_result(width, labels, alpha, min_k, [],
                            f"种子数 n={n_seeds} < 2，无法估计层内标准误")
    else:
        res = meta_analysis(effects, ses, labels=labels, alpha=alpha,
                            min_k=min_k)
    res["n_seeds"] = n_seeds
    res["seed_effects"] = matrix.tolist()
    res["bootstrap"] = {
        "available": False,
        "n_resamples": n_bootstrap,
        "n_valid_resamples": 0,
        "seed": seed,
        "reason": None,
    }
    if (n_bootstrap == 0 or n_seeds < 3
            or res.get("status") in ("insufficient", "invalid")):
        res["bootstrap"]["reason"] = (
            "n_bootstrap=0" if n_bootstrap == 0
            else "种子数 < 3，重采样自由度不足"
            if n_seeds < 3 else "meta 分析不可用")
        return res
    included = [s["label"] for s in res.get("per_stratum", [])]
    if len(included) < min_k:
        res["bootstrap"]["reason"] = "可用层数不足"
        return res
    included_idx = [res["labels"].index(label) for label in included]

    rng = np.random.default_rng(seed)
    i2_values, tau2_values, pooled_values, qp_values = [], [], [], []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n_seeds, size=n_seeds)
        sub = matrix[idx, :]
        sub_effects, sub_ses, sub_labels = [], [], []
        for j in included_idx:
            col = sub[:, j]
            mean = float(np.mean(col))
            sd = float(np.std(col, ddof=1))
            tol = 1e-12 * max(1.0, abs(mean))
            if not math.isfinite(sd) or sd <= tol:
                continue
            se = sd / math.sqrt(n_seeds)
            if not math.isfinite(se) or se <= 0.0:
                continue
            sub_effects.append(mean)
            sub_ses.append(se)
            sub_labels.append(res["labels"][j])
        if len(sub_effects) < min_k:
            continue
        sub_res = meta_analysis(sub_effects, sub_ses, labels=sub_labels,
                                alpha=alpha, min_k=min_k,
                                leave_one_out=False)
        if sub_res.get("status") in ("insufficient", "invalid"):
            continue
        i2 = _safe_float(sub_res.get("i2"))
        tau2 = _safe_float(sub_res.get("tau2"))
        pooled = _safe_float(sub_res.get("pooled_random"))
        qp = _safe_float(sub_res.get("q_p"))
        if i2 is not None:
            i2_values.append(i2)
        if tau2 is not None:
            tau2_values.append(tau2)
        if pooled is not None:
            pooled_values.append(pooled)
        if qp is not None:
            qp_values.append(qp)

    n_valid = len(i2_values)
    boot = res["bootstrap"]
    boot.update({
        "available": bool(n_valid),
        "n_valid_resamples": n_valid,
        "reason": None if n_valid else "所有重采样都因层内方差为 0 被跳过",
        "i2_ci95": _percentile_ci(i2_values, alpha),
        "tau2_ci95": _percentile_ci(tau2_values, alpha),
        "pooled_random_ci95": _percentile_ci(pooled_values, alpha),
        "q_p_ci95": _percentile_ci(qp_values, alpha),
        "prob_i2_ge_50pct": (float(np.mean(np.asarray(i2_values) >= 0.5))
                             if i2_values else None),
    })
    full_pooled = _safe_float(res.get("pooled_random"))
    if full_pooled is not None and abs(full_pooled) > 1e-12 and pooled_values:
        sign = _sign(full_pooled)
        boot["prob_pooled_random_same_sign"] = float(
            np.mean([_sign(v) == sign for v in pooled_values]))
    else:
        boot["prob_pooled_random_same_sign"] = None
    return res


def attach_to_hypotheses(hypotheses, analysis, metric="RULER平均"):
    """把 RULER 分层异质性摘要挂到同候选/参照的登记假设上，但不改判定。

    只在假设的 ``metric`` 与给定指标一致且存在同 (candidate, reference)
    的异质性结果时附加 ``ruler_heterogeneity``。若任务或长度层检出异质性，
    在 ``detail`` 追加警示：macro 平均不能写成“各层共同效应”，必须并列
    逐层结果。假设的 ``status``/``p_adj`` 等确认性字段保持不变。
    """
    if not isinstance(hypotheses, list) or not isinstance(analysis, dict):
        return hypotheses
    outcomes = [o for o in (analysis.get("outcomes") or [])
                if isinstance(o, dict)]
    for hyp in hypotheses:
        if not isinstance(hyp, dict) or hyp.get("metric") != metric:
            continue
        outcome = next(
            (o for o in outcomes
             if o.get("candidate") == hyp.get("candidate")
             and o.get("reference") == hyp.get("reference")), None)
        if outcome is None:
            continue
        axes = outcome.get("axes") if isinstance(outcome.get("axes"), dict) else {}
        attached = {}
        warnings = []
        for axis in ("task", "context"):
            row = axes.get(axis)
            if not isinstance(row, dict):
                continue
            attached[axis] = {
                "status": row.get("status"),
                "label": row.get("label"),
                "i2": row.get("i2"),
                "i2_pct": row.get("i2_pct"),
                "q_p": row.get("q_p"),
                "tau2": row.get("tau2"),
                "pooled_random": row.get("pooled_random"),
                "ci95_random": row.get("ci95_random"),
                "prediction_interval": row.get("prediction_interval"),
                "n_strata": row.get("k"),
                "n_seeds": row.get("n_seeds"),
                "bootstrap": row.get("bootstrap"),
            }
            if row.get("status") in ("heterogeneous",
                                     "moderate_heterogeneity"):
                axis_cn = "任务" if axis == "task" else "上下文长度"
                i2_pct = _safe_float(row.get("i2_pct"))
                q_p = _safe_float(row.get("q_p"))
                warnings.append(
                    f"{axis_cn}层异质性（I²="
                    f"{f'{i2_pct:.1f}%' if i2_pct is not None else 'NA'}，"
                    f"Q p={f'{q_p:.4f}' if q_p is not None else 'NA'}）")
        if attached:
            hyp["ruler_heterogeneity"] = attached
        if warnings:
            detail = hyp.get("detail") or ""
            hyp["detail"] = (
                detail + "；⚠ RULER 平均不能被解释为各层共同效应："
                + "；".join(warnings)
                + "。须并列逐任务/逐长度结果，不能用 macro 平均掩盖方向"
                  "相反的层")
    return hypotheses
