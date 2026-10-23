"""SESOI、等价性（TOST）与非劣性检验。

为什么需要这一层
================
项目的每个配对比较都做优效性检验（H0: Δ=0）。但“没有检出显著差异”并不
等于“没有差异”，更不等于“差异小到没有实际意义”。MDE 只回答“当前种子数/
方差下能检测多大的效应”，无法回答“数据是否支持差异小于某个阈值”。本模块
用双单侧 t 检验（TOST）补上后者：

* **SESOI**（smallest effect size of interest，最小重要差异）是分析前声明的
  实际重要阈值，不是从当前数据估计出来的；
* **等价** = 100(1−2α)% 置信区间完全落在 (−SESOI, +SESOI) 内，即可以在
  α 水平拒绝“差异至少和 SESOI 一样大”；
* **非劣** = 单侧检验支持“候选没有比参照差超过 SESOI”；
* **统计显著但幅度低于 SESOI** = 有可重复的差异，但实际意义有限；
* **未定** = 区间既不能排除 0，也不能排除 SESOI 以外的差异——功效不足或
  方差过大，不能写成“无差异”。

口径
====
* PPL 主指标：逐种子相对差 ``(candidate − reference) / reference``，阈值
  1%（相对 PPL）；
* PPL(log)：逐种子 ``log(PPL_candidate) − log(PPL_reference)``，阈值
  ``log(1.01)``（几何均值比 1.01）；
* RULER/EM：绝对差，阈值 1 个百分点；
* AUROC / PR-AUC 差：绝对差，阈值 0.01；
* 因果病灶 ΔNLL：绝对差，阈值 0.01 nat（约 1% PPL）；
* Shapley：PPL 用 −log PPL 单位，阈值 0.01；RULER 用 EM，阈值 0.01。

这些阈值是**协议约定**而非数据估计；汇总同时报 0.5×/1×/2× 的敏感性网格，
结论若依赖网格点必须并列披露。等价性检验不改变预注册的优效性判定，只补充
“实际意义/无差异证据”的解释；n<3 与项目其余统计协议一致，不作等价性判定。

TOST 的 p 值同样是一个可做多重比较校正的检验族：``run_all`` 在同一指标家族
内对 TOST 原始 p 做 Holm 校正，等价性结论基于校正后的 ``p_tost_adj``。种子
只有 3~5 个时 t 分布假设仍是大样本近似，等价区间应结合 MDE、置换检验与
逐种子方向一起读，不能单独当成“证明无效”的强证据。
"""
import math
from numbers import Real

from .utils import student_t_two_sided_p

# 与 run_all 的统计协议保持一致；此模块不反向 import run_all，避免循环依赖。
DEFAULT_ALPHA = 0.05
MIN_SEEDS = 3
SENSITIVITY_FACTORS = (0.5, 1.0, 2.0)


def _strict_number(value, name: str) -> float:
    """有限实数；bool/字符串/NaN/Inf 显式拒绝（与项目证据解析同口径）。"""
    if isinstance(value, bool):
        raise ValueError(f"{name} 必须是有限实数，收到 bool {value!r}")
    if isinstance(value, str) or not isinstance(value, Real):
        raise ValueError(f"{name} 必须是有限实数，收到 {value!r}")
    try:
        f = float(value)
    except (TypeError, ValueError, OverflowError) as e:
        # 合法但超出 float 上限的整数（如 10**400）不是“有限实数”意义上的
        # 可计算值；统一按 ValueError 拒绝，避免裸 OverflowError 穿透。
        raise ValueError(f"{name} 必须是有限实数，收到 {value!r}") from e
    if not math.isfinite(f):
        raise ValueError(f"{name} 必须是有限实数，收到 {value!r}")
    return f


def _finite_p(value):
    """p 值 → float；bool/非数值/非有限/超出 float 上限的值按缺失处理。

    报告可能被手工编辑或从外部 JSON 重建，p 值字段可能是合法但巨大的整数
    （如 10**400）。``float()``/``math.isfinite()`` 会抛 OverflowError，不能
    让它穿透判定入口；与 ``_observation`` 同口径视为不可用证据。
    """
    if value is None or isinstance(value, bool):
        return None
    if type(value).__name__ == "bool_":
        return None
    if not isinstance(value, Real):
        return None
    try:
        f = float(value)
    except (OverflowError, TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _observation(value):
    """单条观测：None/非数值/非有限/bool/不可表示值返回 None（整对剔除）。

    合法但超出 float 表示范围的整数（如 10**400）不是可计算的有限观测：
    ``float()`` 会抛 OverflowError，若让它穿透，TOST 入口会以未捕获异常
    失败，而不是像其它坏观测一样被剔除。与 ``_strict_number`` 同口径按
    缺失处理。
    """
    if value is None or isinstance(value, bool):
        return None
    # numpy.bool_ 不是 numbers.Real，但某些版本会伪装成整数；按类型名挡住。
    if type(value).__name__ == "bool_":
        return None
    if not isinstance(value, Real):
        return None
    try:
        f = float(value)
    except (OverflowError, TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def metric_margin(metric):
    """返回指标的默认 SESOI 规格（副本）；未知指标返回 None。

    字段：
    * ``kind``：``relative``（相对差）、``log_ratio``（对数比差）、
      ``absolute``（原尺度绝对差）；
    * ``margin``：TOST 阈值，单位与 kind 一致；
    * ``relative_margin``：仅 log_ratio 使用，等价于几何均值比 −1；
    * ``display_scale`` / ``display_unit``：汇总渲染用的换算。
    """
    specs = {
        "PPL": {"kind": "relative", "margin": 0.01,
                "label": "相对 PPL ±1%",
                "display_scale": 100.0, "display_unit": "%"},
        "PPL(log)": {"kind": "log_ratio", "margin": math.log1p(0.01),
                     "relative_margin": 0.01,
                     "label": "PPL 几何均值比 ±1%",
                     "display_scale": 100.0, "display_unit": "%"},
        "PPL vs ngram": {"kind": "relative", "margin": 0.01,
                         "label": "相对 PPL ±1%（vs 统计基线）",
                         "display_scale": 100.0, "display_unit": "%"},
        "PPL窗口配对": {"kind": "absolute", "margin": 0.01,
                       "label": "逐窗口 NLL ±0.01 nat",
                       "display_scale": 1.0, "display_unit": "nat"},
        "RULER平均": {"kind": "absolute", "margin": 0.01,
                     "label": "RULER EM ±1.00 pp",
                     "display_scale": 100.0, "display_unit": "pp"},
        "RULER样例配对": {"kind": "absolute", "margin": 0.01,
                        "label": "RULER 样例配对差 ±1.00 pp",
                        "display_scale": 100.0, "display_unit": "pp"},
        "AUROC差(logit−仅查询)": {
            "kind": "absolute", "margin": 0.01,
            "label": "AUROC 差 ±0.010",
            "display_scale": 1.0, "display_unit": ""},
        "AP差(logit−仅查询)": {
            "kind": "absolute", "margin": 0.01,
            "label": "PR-AUC 差 ±0.010",
            "display_scale": 1.0, "display_unit": ""},
        "PR-AUC差(探针−门控)": {
            "kind": "absolute", "margin": 0.01,
            "label": "PR-AUC 差 ±0.010",
            "display_scale": 1.0, "display_unit": ""},
        "AUROC差(探针−门控)": {
            "kind": "absolute", "margin": 0.01,
            "label": "AUROC 差 ±0.010",
            "display_scale": 1.0, "display_unit": ""},
        "因果消融ΔNLL(lesion−normal)": {
            "kind": "absolute", "margin": 0.01,
            "label": "ΔNLL ±0.01 nat",
            "display_scale": 1.0, "display_unit": "nat"},
        "Shapley(PPL)": {"kind": "absolute", "margin": 0.01,
                        "label": "Shapley φ ±0.01（−log PPL 单位）",
                        "display_scale": 1.0, "display_unit": ""},
        "Shapley(RULER平均)": {"kind": "absolute", "margin": 0.01,
                              "label": "Shapley φ ±1.00 pp",
                              "display_scale": 100.0, "display_unit": "pp"},
    }
    if not isinstance(metric, str):
        return None
    spec = specs.get(metric)
    if spec is None and metric.startswith("RULER:"):
        spec = {"kind": "absolute", "margin": 0.01,
                "label": "RULER 任务 EM ±1.00 pp",
                "display_scale": 100.0, "display_unit": "pp"}
    return dict(spec) if isinstance(spec, dict) else None


def _validate_margin(margin, kind: str) -> float:
    margin = _strict_number(margin, "SESOI margin")
    if margin <= 0.0:
        raise ValueError(f"SESOI margin 必须 > 0，收到 {margin!r}")
    if kind == "log_ratio" and margin >= 1.0:
        # log margin 通常很小；>=1 表示比值超过 e 倍，仍允许但显式说明不常用。
        pass
    return margin


def _validate_alpha(alpha) -> float:
    alpha = _strict_number(alpha, "alpha")
    if not 0.0 < alpha < 0.5:
        # TOST 的 1−2α 区间必须在 (0,1) 内；α>=0.5 时“90% 区间”不成立。
        raise ValueError(f"alpha 必须在 (0, 0.5) 内，收到 {alpha!r}")
    return alpha


def _invalid_result(result: dict, reason: str) -> dict:
    """把无法可靠计算的 TOST 结果标为 invalid，而不是抛错或写入 NaN/Inf。

    输入观测本身都通过了有限性检查，但有限值的和/方差仍可能超出 float
    可表示范围（如 ±1e308 交替）。这种数据无法给出有意义的 t 区间，必须
    与“种子不足”“未定”区分开；下游 ``interpret_equivalence`` 会保留 invalid
    状态，不能把它改写成“没有效应”。
    """
    result.update({
        "status": "invalid",
        "equivalent": False,
        "noninferior": False,
        "p_low": None,
        "p_high": None,
        "p_h0_mu_le_neg_margin": None,
        "p_h0_mu_ge_pos_margin": None,
        "p_tost": None,
        "ci90": None,
        "ci95": None,
        "note": f"无法可靠计算 TOST：{reason}（按 invalid 处理，不代表等价/无差异）",
    })
    return result


def _paired_diffs(a, b, kind: str):
    """按位置配对并应用尺度变换；坏观测整对剔除。

    与 ``lega.utils._paired_diffs`` 的严格口径一致：长度必须一致，None 必须
    成对出现；非有限/非数值/bool 整对剔除。相对差需要参照 >0，对数比需要
    两侧 >0，否则该对无法定义、跳过（不把 0/负 PPL 伪造成巨大相对差）。
    """
    a_list, b_list = list(a), list(b)
    if len(a_list) != len(b_list):
        raise ValueError(
            f"配对样本长度不一致: {len(a_list)} vs {len(b_list)}")
    diffs = []
    for x, y in zip(a_list, b_list):
        if (x is None) != (y is None):
            raise ValueError("配对样本 None 位置不一致，无法按位置一一对应")
        xf, yf = _observation(x), _observation(y)
        if xf is None or yf is None:
            continue
        if kind == "relative":
            if yf <= 0.0:
                continue
            d = (xf - yf) / yf
        elif kind == "log_ratio":
            if xf <= 0.0 or yf <= 0.0:
                continue
            d = math.log(xf) - math.log(yf)
        else:
            d = xf - yf
        if math.isfinite(d):
            diffs.append(d)
    return diffs


def t_critical(df, two_sided_alpha: float = 0.05):
    """Student-t 双侧临界值（无 scipy，二分求根）；无效 df/alpha 返回 None。"""
    try:
        df = _strict_number(df, "df")
        alpha = _strict_number(two_sided_alpha, "two_sided_alpha")
    except ValueError:
        return None
    if df <= 0.0 or not 0.0 < alpha < 1.0:
        return None
    lo, hi = 0.0, 1.0
    # 先指数扩张上界，避免极小 alpha/极大 df 时 1e3 不够。
    while (student_t_two_sided_p(hi, df) > alpha and hi < 1e12):
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if student_t_two_sided_p(mid, df) > alpha:
            lo = mid
        else:
            hi = mid
        if hi - lo <= 1e-12 * max(1.0, hi):
            break
    return 0.5 * (lo + hi)


def _one_sided_upper(t, df):
    """P(T_df >= t)。"""
    p2 = student_t_two_sided_p(t, df)
    if p2 is None or math.isnan(p2):
        return float("nan")
    return p2 / 2.0 if t >= 0 else 1.0 - p2 / 2.0


def _one_sided_lower(t, df):
    """P(T_df <= t)。"""
    return 1.0 - _one_sided_upper(t, df)


def tost_from_diffs(diffs, margin: float, alpha: float = DEFAULT_ALPHA,
                    min_seeds: int = MIN_SEEDS) -> dict:
    """对已变换的配对差做 TOST；返回完整统计量，不依赖原始 a/b。

    H0_low: μ <= −margin；H0_high: μ >= +margin。TOST p = max(p_low, p_high)。
    等价判定要求 ``p_tost < alpha`` 且 ``n >= min_seeds``（项目小样本协议）。
    """
    margin = _validate_margin(margin, "absolute")
    alpha = _validate_alpha(alpha)
    if isinstance(min_seeds, bool) or not isinstance(min_seeds, int):
        raise ValueError(f"min_seeds 必须是整数，收到 {min_seeds!r}")
    if min_seeds < 1:
        raise ValueError(f"min_seeds 必须 >= 1，收到 {min_seeds!r}")
    vals = []
    for v in (diffs if isinstance(diffs, (list, tuple)) else list(diffs)):
        f = _observation(v)
        if f is not None:
            vals.append(f)
    n = len(vals)
    result = {
        "method": "TOST（配对 t，双单侧）",
        "n": n, "df": None, "mean_diff": None, "std_diff": None, "se": None,
        "margin": margin, "alpha": alpha, "min_seeds": min_seeds,
        "t_low": None, "t_high": None,
        "p_low": None, "p_high": None,
        "p_h0_mu_le_neg_margin": None, "p_h0_mu_ge_pos_margin": None,
        "p_tost": None,
        "ci90": None, "ci95": None,
        "equivalent": False, "noninferior": False,
        "p_noninf_worse": None, "significant": None,
        "status": "insufficient_seeds" if n < min_seeds else "undetermined",
        "note": ("margin 为分析前声明的 SESOI；等价=差异小于 SESOI，"
                 "不是“效应恰为 0”；n<3 不作判定。"),
    }
    if n == 0:
        result["note"] = "无有效配对观测，无法做等价性检验。"
        return result
    total = sum(vals)
    if not math.isfinite(total):
        # 有限观测的和也可能超出 float 上限（如 5×1e308）。fsum 对可抵消
        # 的巨大正负值给出精确结果；仍不可表示时按 invalid 处理。
        try:
            total = math.fsum(vals)
        except (OverflowError, ValueError):
            return _invalid_result(result, "有效观测之和超出 float 表示范围")
    mean = total / n
    if not math.isfinite(mean):
        return _invalid_result(result, "均值超出 float 表示范围")
    result["mean_diff"] = float(mean)
    if n < 2:
        return result
    try:
        variance = sum((v - mean) ** 2 for v in vals) / (n - 1)
    except (OverflowError, ZeroDivisionError):
        variance = float("inf")
    if not math.isfinite(variance):
        # 朴素平方可能中间溢出，而真实方差仍在 float 范围内：先按最大
        # 偏差缩放，再还原。仍不可表示时如实标为 invalid（不能写 NaN/Inf）。
        scale = max(abs(v - mean) for v in vals)
        if scale == 0.0:
            variance = 0.0
        else:
            try:
                normalized = sum(((v - mean) / scale) ** 2 for v in vals)
                variance = scale * scale * normalized / (n - 1)
            except (OverflowError, ZeroDivisionError):
                return _invalid_result(result, "方差超出 float 表示范围")
            if not math.isfinite(variance):
                return _invalid_result(result, "方差超出 float 表示范围")
    sd = math.sqrt(max(0.0, variance))
    if not math.isfinite(sd):
        return _invalid_result(result, "标准差超出 float 表示范围")
    df = n - 1
    result.update({"df": df, "std_diff": float(sd)})
    if sd == 0.0:
        se = 0.0
        p_low = 0.0 if mean > -margin else 1.0
        p_high = 0.0 if mean < margin else 1.0
        # 均值恰好落在边界上时，单侧 H0 不被拒绝：保守取 p=1。
        if mean <= -margin:
            p_low = 1.0
        if mean >= margin:
            p_high = 1.0
        result.update({"se": 0.0, "t_low": None, "t_high": None,
                       "p_low": p_low, "p_high": p_high,
                       "p_tost": max(p_low, p_high),
                       "ci90": [float(mean), float(mean)],
                       "ci95": [float(mean), float(mean)]})
    else:
        se = sd / math.sqrt(n)
        t_low = (mean + margin) / se
        t_high = (mean - margin) / se
        p_low = _one_sided_upper(t_low, df)   # H0: μ <= −margin
        p_high = _one_sided_lower(t_high, df)  # H0: μ >= +margin
        crit90 = t_critical(df, 2.0 * alpha)
        crit95 = t_critical(df, alpha)
        result.update({
            "se": float(se), "t_low": float(t_low), "t_high": float(t_high),
            "p_low": float(p_low), "p_high": float(p_high),
            "p_tost": float(max(p_low, p_high)),
            "ci90": ([float(mean - crit90 * se), float(mean + crit90 * se)]
                     if crit90 is not None else None),
            "ci95": ([float(mean - crit95 * se), float(mean + crit95 * se)]
                     if crit95 is not None else None),
        })
    p_tost = result.get("p_tost")
    if p_tost is None or not math.isfinite(float(p_tost)):
        return _invalid_result(result, "TOST p 值非有限")
    for key in ("ci90", "ci95"):
        ci = result.get(key)
        if ci is not None and not all(math.isfinite(float(x)) for x in ci):
            return _invalid_result(result, f"{key} 超出 float 表示范围")
    result["equivalent"] = bool(
        n >= min_seeds and p_tost is not None and p_tost < alpha)
    result["significant"] = bool(
        n >= min_seeds and result.get("ci95") is not None
        and (result["ci95"][0] > 0.0 or result["ci95"][1] < 0.0))
    result["status"] = ("equivalent" if result["equivalent"]
                        else "insufficient_seeds" if n < min_seeds
                        else "not_equivalent")
    # 显式别名，避免把 p_low/p_high 误读成“下/上尾”而不是两个单侧 H0。
    result["p_h0_mu_le_neg_margin"] = result.get("p_low")
    result["p_h0_mu_ge_pos_margin"] = result.get("p_high")
    return result


def tost_paired(a, b, margin: float, alpha: float = DEFAULT_ALPHA,
                kind: str = "absolute", min_seeds: int = MIN_SEEDS) -> dict:
    """对配对样本 a−b 做 TOST（kind=relative/log_ratio/absolute）。"""
    if kind not in ("absolute", "relative", "log_ratio"):
        raise ValueError(f"未知的 TOST 尺度 kind={kind!r}")
    margin = _validate_margin(margin, kind)
    diffs = _paired_diffs(a, b, kind)
    res = tost_from_diffs(diffs, margin, alpha=alpha, min_seeds=min_seeds)
    res.update({"kind": kind, "n_pairs_valid": len(diffs)})
    return res


def _margin_for_factor(spec: dict, factor: float) -> float:
    """按敏感性因子换算 margin；log_ratio 按几何比换算而不是直接乘 log。

    因子/基础 margin 必须是能够表示的有限数：合法但超出 float 上限的整数
    factor 若直接参与乘法会抛 OverflowError（或得到 inf），不能穿透到
    敏感性网格/汇总。无法表示时显式 ValueError，而不是伪造一个无穷阈值。
    """
    if not isinstance(spec, dict):
        raise ValueError(f"margin 规格必须是对象，收到 {spec!r}")
    factor = _strict_number(factor, "margin factor")
    if factor < 0:
        raise ValueError(f"margin factor 必须 >= 0，收到 {factor!r}")
    kind = spec.get("kind")
    base = _strict_number(spec.get("margin"), "margin")
    if base <= 0:
        raise ValueError(f"margin 必须 > 0，收到 {base!r}")
    if kind == "log_ratio":
        rel = spec.get("relative_margin")
        rel = math.expm1(base) if rel is None else _strict_number(
            rel, "relative_margin")
        if rel < 0:
            raise ValueError(f"relative_margin 必须 >= 0，收到 {rel!r}")
        try:
            out = math.log1p(rel * factor)
        except (OverflowError, ValueError):
            raise ValueError(
                f"log-ratio margin 不可表示（relative_margin={rel!r}, "
                f"factor={factor!r}）") from None
    else:
        try:
            out = base * factor
        except OverflowError:
            raise ValueError(
                f"margin 不可表示（margin={base!r}, factor={factor!r}）"
            ) from None
    if not math.isfinite(out):
        raise ValueError(
            f"margin 不可表示（margin={base!r}, factor={factor!r}）")
    return float(out)


def assess_paired_equivalence(a, b, metric, lower_is_better: bool = False,
                              alpha: float = DEFAULT_ALPHA,
                              margin=None,
                              min_seeds: int = MIN_SEEDS,
                              sensitivity: bool = True):
    """对一组配对观测做“SESOI 解释层”，返回 None 表示该指标未定义 SESOI。

    返回 dict 既包含 TOST 原始统计量，也包含方向/是否超过 SESOI/非劣性与
    0.5×/1×/2× 敏感性网格。最终 status 由 ``interpret_equivalence`` 在
    Holm 校正后写入（调用方负责）。
    """
    spec = metric_margin(metric)
    if spec is None:
        return None
    alpha = _validate_alpha(alpha)
    kind = spec["kind"]
    if margin is None:
        margin = spec["margin"]
    margin = _validate_margin(margin, kind)
    diffs = _paired_diffs(a, b, kind)
    out = tost_from_diffs(diffs, margin, alpha=alpha, min_seeds=min_seeds)
    out.update({
        "metric": metric, "kind": kind, "margin": margin,
        "margin_label": spec.get("label", ""),
        "display_scale": spec.get("display_scale", 1.0),
        "display_unit": spec.get("display_unit", ""),
        "lower_is_better": bool(lower_is_better),
        "n_pairs_valid": len(diffs),
        "multiplicity_adjusted": False,
    })
    mean = out.get("mean_diff")
    if mean is not None:
        if mean == 0.0:
            direction = "same"
        else:
            better = (mean < 0.0) if lower_is_better else (mean > 0.0)
            direction = "better" if better else "worse"
        out.update({
            "direction": direction,
            "beyond_margin": bool(abs(mean) >= margin),
            "practically_meaningful": bool(
                direction != "same" and abs(mean) >= margin),
        })
    else:
        out.update({"direction": None, "beyond_margin": None,
                    "practically_meaningful": None})
    if out.get("status") == "invalid":
        # 数值不可表示（而非种子不足/未定）：保留 invalid 与原因，不能继续
        # 走下面的 status 分支把它改写成 insufficient_seeds/not_equivalent。
        out.update({"p_noninf_worse": None, "noninferior": False,
                    "sensitivity": []})
        return out
    # 非劣性：只检验“劣化不超过 SESOI”这一侧（方向随 lower_is_better 翻转）。
    p_low, p_high = out.get("p_low"), out.get("p_high")
    if lower_is_better:
        p_noninf = p_high  # H0: μ >= +margin（候选更差超过 SESOI）
    else:
        p_noninf = p_low   # H0: μ <= −margin（候选更差超过 SESOI）
    out["p_noninf_worse"] = p_noninf
    out["noninferior"] = bool(
        out.get("n", 0) >= min_seeds
        and p_noninf is not None and p_noninf < alpha)
    if out["equivalent"]:
        out["status"] = "equivalent"
    elif out["noninferior"]:
        out["status"] = "noninferior"
    elif out.get("n", 0) < min_seeds:
        out["status"] = "insufficient_seeds"
    else:
        out["status"] = "undetermined"

    if sensitivity:
        grid = []
        spec_for_grid = dict(spec, margin=margin)
        if kind == "log_ratio":
            # 自定义 margin 时也要按当前 margin 重新换算几何比，不能沿用
            # spec 里的默认 1%。
            spec_for_grid["relative_margin"] = math.expm1(margin)
        for factor in SENSITIVITY_FACTORS:
            m2 = _margin_for_factor(spec_for_grid, factor)
            cell = tost_from_diffs(diffs, m2, alpha=alpha,
                                   min_seeds=min_seeds)
            cell.update({"factor": factor, "margin": m2,
                         "p_tost_adj": None})
            grid.append(cell)
        out["sensitivity"] = grid
    else:
        out["sensitivity"] = []
    return out


_STATUS_LABELS = {
    "equivalent": "等价（差异小于 SESOI；不等于效应为 0）",
    "noninferior": "非劣（劣化未超过 SESOI）",
    "significant_meaningful_better": "显著且实际有意义（优于，超过 SESOI）",
    "significant_meaningful_worse": "显著且实际有意义（劣于，超过 SESOI）",
    "significant_below_sesoi": "统计显著但幅度在 SESOI 内（实际意义有限）",
    "undetermined": "未定（无法排除超过 SESOI 的差异）",
    "insufficient_seeds": "种子不足（n<3），不作等价性判定",
    "not_applicable": "不适用（该指标未定义 SESOI）",
    "invalid": "无法计算（输入损坏或尺度未定义）",
}

# 判定优先级：等价/非劣之外的“统计显著”只在优效性检验通过后成立。
_MEANINGFUL = {
    ("better", True): "significant_meaningful_better",
    ("worse", True): "significant_meaningful_worse",
    ("better", False): "significant_below_sesoi",
    ("worse", False): "significant_below_sesoi",
}


def interpret_equivalence(eq, p_adj=None, alpha: float = DEFAULT_ALPHA,
                          min_seeds: int = MIN_SEEDS) -> dict:
    """把 TOST 统计量与优效性 p_adj 合成一个可渲染的实际意义判定。

    优先级：种子不足 → 等价 → 优效性显著（再分是否超过 SESOI）→ 非劣 →
    未定。等价性用 ``p_tost_adj``（若已做 Holm 校正）否则用 ``p_tost``；
    ``multiplicity_adjusted`` 如实标明是否经过家族校正。该判定不改变预注册
    假设的 支持/拒绝/未定 状态，只回答“差异是否有实际意义”。
    """
    if not isinstance(eq, dict):
        return {"status": "invalid", "label": _STATUS_LABELS["invalid"],
                "equivalent": False, "multiplicity_adjusted": False,
                "p_used": None}
    if eq.get("not_applicable") is True:
        # 典型情形：因果病灶全部种子 no-op（干预没有真正改变前向）。
        # 此时零差不是“等价证据”，必须单列为不适用。
        return {"status": "not_applicable",
                "label": (eq.get("not_applicable_reason")
                          or _STATUS_LABELS["not_applicable"]),
                "equivalent": False, "multiplicity_adjusted": False,
                "p_used": None}
    if eq.get("status") == "invalid":
        # 输入损坏或统计量不可表示：必须保留 invalid，不能在重算时退化成
        # “种子不足/未定”（那会让损坏证据看起来只是样本少）。
        return {"status": "invalid",
                "label": (eq.get("label") or _STATUS_LABELS["invalid"]),
                "equivalent": False,
                "multiplicity_adjusted": bool(
                    eq.get("multiplicity_adjusted") is True),
                "p_used": None}
    alpha = _validate_alpha(alpha)
    n = eq.get("n")
    n_ok = (isinstance(n, int) and not isinstance(n, bool)
            and n >= min_seeds)
    p_tost_adj = _finite_p(eq.get("p_tost_adj"))
    p_tost_raw = _finite_p(eq.get("p_tost"))
    adjusted = p_tost_adj is not None
    p_used = p_tost_adj if adjusted else p_tost_raw
    status = "undetermined"
    if not n_ok:
        status = "insufficient_seeds"
    elif p_used is not None and p_used < alpha:
        status = "equivalent"
    else:
        pa = _finite_p(p_adj)
        if pa is not None and pa < alpha:
            direction = eq.get("direction")
            beyond = eq.get("beyond_margin")
            status = _MEANINGFUL.get((direction, bool(beyond)),
                                     "significant_below_sesoi")
        elif (eq.get("noninferior") is True and n_ok
              # 校正后 p 不显著时不能退回原始 noninferior 标志：那会出
              # 现“multiplicity_adjusted=True、p_used>alpha、却判非劣”的
              # 自相矛盾结论。只有明确记录了已校正的非劣性证据才采信。
              and (not adjusted
                   or eq.get("noninferior_adjusted") is True)):
            status = "noninferior"
    label = _STATUS_LABELS.get(status, status)
    if status == "significant_below_sesoi" and eq.get("direction") == "worse":
        label += "；方向：劣于"
    elif status == "significant_below_sesoi" and eq.get("direction") == "better":
        label += "；方向：优于"
    return {
        "status": status,
        "label": label,
        "equivalent": status == "equivalent",
        "multiplicity_adjusted": bool(adjusted),
        "p_used": p_used,
    }


def interpolate_margin(spec: dict, factor: float) -> float:
    """公开的敏感性 margin 换算（供测试/报告复用）。"""
    return _margin_for_factor(spec, factor)
