"""缺失 run 的敏感性分析：完成者分析会不会因缺失单元而改变结论。

为什么需要这一层
================
主检验只使用候选/参照都成功产出的种子（available-case / complete-case）。
如果缺失与结果相关——例如更大的变体更容易 NaN、更难的种子更容易被中止——
那么“跑成功的这几次恰好更好/更差”本身就会制造方向。设计符合性门只保证
不把缺失单元冒充成全量确认性证据，并不量化“缺失多少、要极端到什么程度
才能翻转结论”。本模块因此为每个配对比较报告：

* **缺失模式**：计划种子中双方齐全、仅候选缺失、仅参照缺失、双方缺失的
  数量与具体种子；缺失率与两臂缺失率之差（differential attrition）；
* **临界点（tipping point）**：全部缺失配对差取同一个值 δ 时，δ 需要多大
  才能把完整案例均值推到 0、+SESOI 或 −SESOI；δ 是否落在观测到的配对差
  范围内；
* **观测范围边界**：把缺失配对差全部设为观测到的最小/最大配对差时，完整
  均值的可能范围；若该范围跨过 0 或实际重要阈值，说明结论对缺失机制敏感；
* **Manski 最坏情形界（部分识别）**：当 outcome 有已知有限范围 ``[L, U]``
  时，对每个缺失配对利用已知一侧构造差值区间（候选缺/参照 r：
  ``[L-r, U-r]``；参照缺/候选 c：``[c-U, c-L]``；双缺：
  ``[L-U, U-L]``），在计划种子数上取平均得到均值差的 sharp bounds。
  对所有指标还在缺失配对处把符号得分（+1/0/−1）取 ``[-1, +1]``，得到
  符号得分界。界跨 0 时判“最坏情形方向未定”，这不等于无效应。

边界
====
* 这不是多重插补，也不假设缺失随机（MAR/MCAR）。观测到的配对差范围只是
  一个**透明的敏感性尺度**，不是缺失值的概率区间。它本身仍不是严格最坏
  情形界；真正的 Manski 界只在给定 outcome 范围时计算，并可能很宽。
* “临界点在观测范围外”只说明翻转需要比任何已观测种子更极端的缺失结果，
  不证明缺失随机、也不证明结论稳健；样本很小时观测范围本身就很窄。
* Manski 界只假设 outcome 范围与计划种子，不假设缺失机制；PPL 无界，因此
  不给均值差界，只给符号得分界。界未定 ≠ 无效应，界很宽也不等于结论无效。
* 缺失敏感性不进入 Holm/FWER，不改变预注册的 支持/拒绝/未定；它补充的是
  “可用样本结论在何种缺失机制下会失效”，与设计降级、MDE、TOST 并列阅读。
* 计划种子为空/观测对为空/数值不可表示时显式返回 ``insufficient`` 或
  ``invalid``，不把“无法计算”写成“没有缺失”或“稳健”；0 个完整观测时
  也不报告只由 outcome 范围决定的空 Manski 界。
"""
import math
from numbers import Real

from .utils import mean_std

MIN_SEEDS = 3

_STATUS_CN = {
    "complete": "无缺失（完成者分析等于计划配对人群）",
    "no_observed_pairs": "没有可用的完整配对观测，无法做缺失敏感性分析",
    "insufficient": "计划种子不足，无法判断缺失模式",
    "invalid": "输入不可表示，缺失敏感性无法计算",
    "robust_within_observed_range": (
        "在观测到的配对差范围内补缺失，不会翻转符号/实际意义判定"),
    "sensitive_within_observed_range": (
        "缺失配对差只要落在观测范围内，就可能翻转符号/实际意义判定"),
}

_LABEL_CN = {
    "complete": "无缺失",
    "no_observed_pairs": "无完整配对",
    "insufficient": "种子不足",
    "invalid": "无法计算",
    "robust_within_observed_range": "观测范围内稳健",
    "sensitive_within_observed_range": "缺失敏感",
}

_METHOD = (
    "complete-case tipping-point / observed-range / Manski-bound sensitivity "
    "analysis (缺失值统一取同一配对差 δ；Manski 界只假设 outcome 范围与"
    "计划种子)")


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


def _seed(value):
    """计划种子必须是精确整数；bool/小数/字符串一律拒绝。"""
    if isinstance(value, bool) or type(value).__name__ == "bool_":
        raise ValueError(f"seed 必须是整数，收到 {value!r}")
    if isinstance(value, int):
        return int(value)
    # numpy 整数：不在标准库类型里显式依赖 numpy，按 __index__ 判断。
    try:
        if hasattr(value, "__index__") and not isinstance(value, float):
            return int(value.__index__())
    except (TypeError, ValueError, OverflowError):
        pass
    raise ValueError(f"seed 必须是整数，收到 {value!r}")


def _strict_int(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} 必须是整数，收到 {value!r}")
    if value < minimum:
        raise ValueError(f"{name} 必须 >= {minimum}，收到 {value!r}")
    return int(value)


def _strict_margin(value):
    out = _finite(value)
    if out is None or out <= 0.0:
        raise ValueError(f"SESOI margin 必须 > 0，收到 {value!r}")
    return out


def _strict_value_bounds(value):
    """outcome 的有限取值区间；区间必须非空且端点有限。"""
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"value_bounds 必须是 [lo, hi]，收到 {value!r}")
    lo, hi = _finite(value[0]), _finite(value[1])
    if lo is None or hi is None or lo >= hi:
        raise ValueError(f"value_bounds 必须满足 lo < hi 且端点有限，收到 {value!r}")
    try:
        width = hi - lo
    except (TypeError, ValueError, OverflowError):
        width = None
    if width is None or not math.isfinite(width):
        raise ValueError(f"value_bounds 宽度不可表示，收到 {value!r}")
    return (float(lo), float(hi))


def _div(num, den):
    """有限除法；溢出/除以 0 返回 None。"""
    try:
        out = num / den
    except (TypeError, ValueError, ZeroDivisionError, OverflowError):
        return None
    return out if math.isfinite(out) else None


def _convex(n, d, m, x):
    """``(n*d + m*x) / (n+m)`` 的尺度稳定写法。"""
    total = n + m
    if total <= 0:
        return None
    try:
        out = (n / total) * d + (m / total) * x
    except (TypeError, ValueError, ZeroDivisionError, OverflowError):
        return None
    return out if math.isfinite(out) else None


def _missing_value_for(threshold, n_observed, observed_mean, n_missing):
    """使完整均值等于 ``threshold`` 的缺失配对差（所有缺失取同一值）。

    ``threshold`` 为 0 时即符号翻转临界点；为 ±SESOI 时为实际意义临界点。
    """
    if n_missing <= 0 or observed_mean is None:
        return None
    try:
        ratio = n_observed / n_missing
        out = threshold + ratio * (threshold - observed_mean)
    except (TypeError, ValueError, ZeroDivisionError, OverflowError):
        return None
    return out if math.isfinite(out) else None


def _normalise_plan(planned_seeds, candidate_values, reference_values):
    if planned_seeds is None:
        keys = set()
        for mapping in (candidate_values, reference_values):
            if isinstance(mapping, dict):
                keys.update(mapping.keys())
        plan = sorted(keys)
        source = "observed_union"
    else:
        if not isinstance(planned_seeds, (list, tuple, set, frozenset)):
            raise ValueError("planned_seeds 必须是种子列表或 None")
        plan = [_seed(s) for s in planned_seeds]
        source = "declared"
    unique = []
    seen = set()
    for s in plan:
        if s not in seen:
            seen.add(s)
            unique.append(s)
    return unique, source


def _values_for(mapping, seeds):
    out = {}
    if not isinstance(mapping, dict):
        return out
    for s in seeds:
        out[s] = _finite(mapping.get(s))
    return out


def _partial_identification(plan, cand, ref, observed, *, value_bounds, margin):
    """在只给定 outcome 范围与计划种子的条件下计算 sharp Manski 界。

    均值差界只在 outcome 有界、至少有一个完整观测且所有已观测值都在界内
    时计算；符号得分界对任意尺度都可用。缺失配对的符号可以是 +1/0/−1，
    因此符号界在缺失数上线性变宽；这不是概率模型，也不假设 MAR/MCAR。
    """
    n = len(plan)
    observed_map = {s: d for s, d in observed}
    missing_n = n - len(observed_map)
    violations = set()
    lo_b = hi_b = None
    if value_bounds is not None:
        lo_b, hi_b = value_bounds

    sum_lo = 0.0
    sum_hi = 0.0
    for s in plan:
        cv, rv = cand.get(s), ref.get(s)
        if value_bounds is not None:
            for value in (cv, rv):
                if value is not None and not (lo_b <= value <= hi_b):
                    violations.add(s)
        if s in observed_map:
            diff = observed_map[s]
            sum_lo += diff
            sum_hi += diff
            continue
        if value_bounds is None:
            continue
        if cv is not None and rv is not None:
            sum_lo += lo_b - hi_b
            sum_hi += hi_b - lo_b
        elif cv is not None:
            sum_lo += cv - hi_b
            sum_hi += cv - lo_b
        elif rv is not None:
            sum_lo += lo_b - rv
            sum_hi += hi_b - rv
        else:
            sum_lo += lo_b - hi_b
            sum_hi += hi_b - lo_b

    mean_lo = mean_hi = None
    if (value_bounds is not None and n > 0 and observed_map
            and not violations):
        lo_candidate = min(sum_lo / n, sum_hi / n)
        hi_candidate = max(sum_lo / n, sum_hi / n)
        if math.isfinite(lo_candidate) and math.isfinite(hi_candidate):
            mean_lo, mean_hi = lo_candidate, hi_candidate

    signs = [1 if d > 0.0 else -1 if d < 0.0 else 0
             for d in observed_map.values()]
    sign_sum = sum(signs)
    sign_lo = (_div(float(sign_sum - missing_n), float(n))
               if n and observed_map else None)
    sign_hi = (_div(float(sign_sum + missing_n), float(n))
               if n and observed_map else None)
    if sign_lo is not None and sign_hi is not None and sign_lo > sign_hi:
        sign_lo, sign_hi = sign_hi, sign_lo

    mean_sign_identified = None
    if mean_lo is not None:
        mean_sign_identified = bool(mean_lo > 0.0 or mean_hi < 0.0)
    sign_score_identified = None
    if sign_lo is not None:
        sign_score_identified = bool(sign_lo > 0.0 or sign_hi < 0.0)

    sign_identified = None
    sign_identified_by = None
    if mean_sign_identified is True:
        sign_identified = True
        sign_identified_by = "mean_diff"
    elif sign_score_identified is True:
        sign_identified = True
        sign_identified_by = "sign_score"
    elif (mean_sign_identified is False
          or sign_score_identified is False):
        sign_identified = False

    within_margin = None
    practical_significance = None
    margin_status = None
    if margin is not None and mean_lo is not None:
        within_margin = bool(mean_lo > -margin and mean_hi < margin)
        if mean_lo > margin:
            practical_significance = "positive"
            margin_status = "beyond_margin_positive"
        elif mean_hi < -margin:
            practical_significance = "negative"
            margin_status = "beyond_margin_negative"
        elif within_margin:
            margin_status = "within_margin"
        else:
            margin_status = "undetermined"

    if not observed_map:
        if value_bounds is None:
            note = ("没有完整配对观测；无界 outcome 下无法计算符号得分界，"
                    "不能用 0 个观测做方向判断。")
        else:
            note = ("没有完整配对观测；Manski 界只剩由 outcome 范围决定的"
                    "空界，未作为分析结果报告，不能用 0 个观测做方向判断。")
    elif value_bounds is None:
        note = ("outcome 无界：不计算均值差最坏情形界；符号得分界只假设缺失"
                "配对方向未知，仍不假设缺失随机。")
    elif violations:
        note = ("已观测 outcome 超出声明范围，均值差 Manski 界不可信，故不"
                "计算；符号得分界仍可用。")
    else:
        note = (f"outcome∈[{lo_b:g}, {hi_b:g}] 下的 sharp Manski 界：只假设"
                "取值范围与计划种子，不假设缺失随机；界未定≠无效应。")

    return {
        "method": ("Manski sharp partial-identification bounds under known "
                   "outcome range and unrestricted missingness"),
        "value_bounds": ([lo_b, hi_b] if value_bounds is not None else None),
        "outcome_bounds_violations": sorted(violations),
        "mean_diff_bounds": ([mean_lo, mean_hi] if mean_lo is not None else None),
        "sign_score_bounds": ([sign_lo, sign_hi]
                              if sign_lo is not None else None),
        "mean_sign_identified": mean_sign_identified,
        "sign_score_identified": sign_score_identified,
        "sign_identified": sign_identified,
        "sign_identified_by": sign_identified_by,
        "within_margin": within_margin,
        "practical_significance": practical_significance,
        "margin_status": margin_status,
        "n_missing_used": int(missing_n),
        "note": note,
    }


def paired_missingness(planned_seeds, candidate_values, reference_values,
                       *, transform=None, margin=None,
                       min_seeds: int = MIN_SEEDS,
                       value_bounds=None) -> dict:
    """对一组计划种子做缺失模式 + 临界点 + 观测范围敏感性分析。

    ``candidate_values`` / ``reference_values`` 为 ``{seed: value}``；缺失、
    bool、字符串、NaN/Inf 以及无法表示的值统一按缺失处理。``planned_seeds``
    为 ``None`` 时退化为两侧观测种子的并集（此时只能发现“单臂缺失”，无法
    发现计划了但两侧都没跑的种子）。

    ``transform(candidate, reference)`` 把原始观测变成分析尺度上的配对差
    （默认 ``candidate - reference``）；PPL 可传相对差以与 1% SESOI 对齐。
    ``value_bounds=(lo, hi)`` 声明 outcome 的已知有限范围（例如 RULER EM
    的 ``(0, 1)``），用于计算均值差的 sharp Manski 最坏情形界；它只适用于
    默认 identity transform（自定义尺度需先明确该尺度上的差值范围）。
    无界 outcome 只能计算符号得分界。返回 dict 字段见模块文档；``status``
    不表示统计显著性，只描述缺失模式下结论的脆弱性。
    """
    min_seeds = _strict_int(min_seeds, "min_seeds", minimum=1)
    margin = None if margin is None else _strict_margin(margin)
    value_bounds = _strict_value_bounds(value_bounds)
    identity_transform = transform is None
    if value_bounds is not None and not identity_transform:
        raise ValueError(
            "value_bounds 只适用于默认 identity transform；自定义尺度需先"
            "明确该尺度上的差值范围")
    if transform is None:
        def transform(a, b):
            return a - b
    if not callable(transform):
        raise ValueError("transform 必须可调用或为 None")

    plan, source = _normalise_plan(planned_seeds,
                                   candidate_values, reference_values)
    cand = _values_for(candidate_values, plan)
    ref = _values_for(reference_values, plan)

    observed = []
    missing_both, missing_candidate, missing_reference = [], [], []
    invalid_diff = []
    for s in plan:
        cv, rv = cand.get(s), ref.get(s)
        if cv is None and rv is None:
            missing_both.append(s)
            continue
        if cv is None:
            missing_candidate.append(s)
            continue
        if rv is None:
            missing_reference.append(s)
            continue
        try:
            diff = _finite(transform(cv, rv))
        except (TypeError, ValueError, ZeroDivisionError, OverflowError):
            diff = None
        if diff is None:
            invalid_diff.append(s)
            continue
        observed.append((s, diff))

    n_planned = len(plan)
    n_observed = len(observed)
    n_missing = n_planned - n_observed
    missing_fraction = _div(float(n_missing), float(n_planned)) \
        if n_planned else None
    candidate_missing_fraction = (
        _div(float(len(missing_candidate) + len(missing_both)), float(n_planned))
        if n_planned else None)
    reference_missing_fraction = (
        _div(float(len(missing_reference) + len(missing_both)), float(n_planned))
        if n_planned else None)
    differential = None
    if (candidate_missing_fraction is not None
            and reference_missing_fraction is not None):
        differential = candidate_missing_fraction - reference_missing_fraction

    diffs = [d for _, d in observed]
    mean, sd = mean_std(diffs)
    result = {
        "method": _METHOD,
        "planned_seeds_source": source,
        "planned_seeds": list(plan),
        "n_planned": int(n_planned),
        "n_observed": int(n_observed),
        "n_missing": int(n_missing),
        "missing_fraction": missing_fraction,
        "candidate_missing_fraction": candidate_missing_fraction,
        "reference_missing_fraction": reference_missing_fraction,
        "differential_missingness": differential,
        "observed_seeds": [s for s, _ in observed],
        "missing_both": missing_both,
        "missing_candidate": missing_candidate,
        "missing_reference": missing_reference,
        "invalid_pair": invalid_diff,
        "observed_mean_diff": mean,
        "observed_sd_diff": sd,
        "observed_min_diff": (min(diffs) if diffs else None),
        "observed_max_diff": (max(diffs) if diffs else None),
        "min_seeds": min_seeds,
        "margin": margin,
    }
    result["partial_identification"] = _partial_identification(
        plan, cand, ref, observed, value_bounds=value_bounds, margin=margin)

    if n_planned == 0:
        result.update({
            "status": "insufficient",
            "label": _LABEL_CN["insufficient"],
            "sign_robust": None,
            "margin_robust": None,
            "full_mean_range": None,
            "missing_mean_for_zero": None,
            "missing_mean_for_plus_margin": None,
            "missing_mean_for_minus_margin": None,
            "tipping_within_observed_range": None,
            "note": "计划种子为空；无法判断缺失模式，不能写成“没有缺失”。",
        })
        return result
    if n_observed == 0:
        result.update({
            "status": "no_observed_pairs",
            "label": _LABEL_CN["no_observed_pairs"],
            "sign_robust": None,
            "margin_robust": None,
            "full_mean_range": None,
            "missing_mean_for_zero": None,
            "missing_mean_for_plus_margin": None,
            "missing_mean_for_minus_margin": None,
            "tipping_within_observed_range": None,
            "note": ("全部计划配对都缺失或不可用：没有完成者估计，"
                     "不能用 0 个观测做任何方向判断。"),
        })
        return result
    if mean is None:
        result.update({
            "status": "invalid",
            "label": _LABEL_CN["invalid"],
            "sign_robust": None,
            "margin_robust": None,
            "full_mean_range": None,
            "missing_mean_for_zero": None,
            "missing_mean_for_plus_margin": None,
            "missing_mean_for_minus_margin": None,
            "tipping_within_observed_range": None,
            "note": "完成者均值不可表示；不能伪造敏感性结论。",
        })
        return result
    if n_missing == 0:
        result.update({
            "status": "complete",
            "label": _LABEL_CN["complete"],
            "sign_robust": True,
            "margin_robust": True,
            "full_mean_range": [mean, mean],
            "missing_mean_for_zero": None,
            "missing_mean_for_plus_margin": None,
            "missing_mean_for_minus_margin": None,
            "tipping_within_observed_range": False,
            "note": "计划配对全部可用；临界点/边界分析不适用。",
        })
        return result
    if n_observed < min_seeds:
        # 仍给出描述性边界，但显式标注种子不足；不做“稳健”判词。
        result["insufficient_seeds"] = True

    lo_diff, hi_diff = min(diffs), max(diffs)
    range_lo = _convex(n_observed, mean, n_missing, lo_diff)
    range_hi = _convex(n_observed, mean, n_missing, hi_diff)
    if range_lo is None or range_hi is None:
        result.update({
            "status": "invalid",
            "label": _LABEL_CN["invalid"],
            "sign_robust": None,
            "margin_robust": None,
            "full_mean_range": None,
            "missing_mean_for_zero": None,
            "missing_mean_for_plus_margin": None,
            "missing_mean_for_minus_margin": None,
            "tipping_within_observed_range": None,
            "note": "观测范围边界不可表示；不能伪造敏感性结论。",
        })
        return result

    full_lo, full_hi = sorted((range_lo, range_hi))
    sign_robust = not (full_lo <= 0.0 <= full_hi)
    missing_for_zero = _missing_value_for(0.0, n_observed, mean, n_missing)
    missing_for_plus = (None if margin is None else
                        _missing_value_for(margin, n_observed, mean, n_missing))
    missing_for_minus = (None if margin is None else
                         _missing_value_for(-margin, n_observed, mean,
                                            n_missing))

    def _inside(value):
        if value is None:
            return None
        return bool(lo_diff <= value <= hi_diff)

    margin_robust = None
    if margin is not None:
        crosses_plus = bool(full_lo < margin < full_hi)
        crosses_minus = bool(full_lo < -margin < full_hi)
        # 边界恰好落在阈值上不算“跨过”（浮点下应保留明确的一侧）。
        margin_robust = not (crosses_plus or crosses_minus)
    else:
        crosses_plus = crosses_minus = None

    tipping_within = _inside(missing_for_zero)
    margin_tipping_within = None
    if margin is not None:
        margin_tipping_within = bool(
            (crosses_plus or crosses_minus)
            and (_inside(missing_for_plus) or _inside(missing_for_minus)))

    sensitive = (not sign_robust) or (
        margin is not None and margin_robust is False)
    status = ("sensitive_within_observed_range" if sensitive
              else "robust_within_observed_range")
    if result.get("insufficient_seeds"):
        note = (f"完整配对只有 {n_observed} 个（< {min_seeds}）：以下临界点/"
                "边界仅作描述，不足以判定稳健或敏感。")
    elif sensitive:
        note = ("缺失配对差只要落在已观测范围内就可能改变符号/实际意义"
                "判定；必须报告缺失机制，不能只用完成者样本宣称稳健。")
    else:
        note = ("在“缺失配对差落在已观测范围”这一透明敏感性尺度下，"
                "符号/实际意义判定不变；这不能证明缺失随机，也不能排除"
                "更极端的缺失值。")
    result.update({
        "status": status,
        "label": _LABEL_CN[status],
        "sign_robust": bool(sign_robust),
        "margin_robust": margin_robust,
        "full_mean_range": [float(full_lo), float(full_hi)],
        "missing_mean_for_zero": missing_for_zero,
        "missing_mean_for_plus_margin": missing_for_plus,
        "missing_mean_for_minus_margin": missing_for_minus,
        "tipping_within_observed_range": tipping_within,
        "margin_tipping_within_observed_range": margin_tipping_within,
        "note": note,
    })
    return result


def aggregate_missingness(entries) -> dict:
    """汇总多个 ``paired_missingness`` 结果，不把不可计算的条目按 0 补齐。"""
    rows = [e for e in (entries or []) if isinstance(e, dict)]
    counts = {}
    for e in rows:
        status = e.get("status")
        key = status if isinstance(status, str) else "invalid"
        counts[key] = counts.get(key, 0) + 1
    with_missing = sum(1 for e in rows
                       if isinstance(e.get("n_missing"), int)
                       and e.get("n_missing") > 0)
    sensitive = sum(1 for e in rows
                    if e.get("status") == "sensitive_within_observed_range")
    complete = sum(1 for e in rows if e.get("status") == "complete")
    manski_mean = manski_mean_sign = manski_mean_undetermined = 0
    manski_sign = manski_undetermined = 0
    for e in rows:
        pi = e.get("partial_identification")
        if not isinstance(pi, dict):
            continue
        has_missing = (isinstance(e.get("n_missing"), int)
                       and e.get("n_missing") > 0)
        if not has_missing:
            continue
        if pi.get("mean_diff_bounds") is not None:
            manski_mean += 1
        if pi.get("mean_sign_identified") is True:
            manski_mean_sign += 1
        elif pi.get("mean_sign_identified") is False:
            manski_mean_undetermined += 1
        if pi.get("sign_score_identified") is True:
            manski_sign += 1
        elif pi.get("sign_score_identified") is False:
            manski_undetermined += 1
    return {
        "n_comparisons": len(rows),
        "n_complete": complete,
        "n_with_missing": with_missing,
        "n_sensitive": sensitive,
        "n_manski_mean_bounds": manski_mean,
        "n_manski_mean_sign_identified": manski_mean_sign,
        "n_manski_mean_sign_undetermined": manski_mean_undetermined,
        "n_manski_sign_score_identified": manski_sign,
        "n_manski_sign_score_undetermined": manski_undetermined,
        "status_counts": counts,
        "note": ("计数只覆盖已计算的指标×对比；状态缺失/无法计算的条目不"
                 "计入“无缺失”，也不按稳健处理。Manski 计数只统计实际"
                 "计算出的最坏情形界，不把“无法计算”当成已识别。"),
    }
