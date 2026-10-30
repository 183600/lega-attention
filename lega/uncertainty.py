"""结果分析中的不确定性分解：种子随机性 vs 评测单元抽样。

同一批评测窗口/样例在多个训练种子下重复出现时，配对差矩阵天然是
``seed × unit`` 的二维结构。只看跨种子 Mean ± Std 会把“换一批评测窗口/
样例”的不确定性漏掉；只看窗口/样例级 bootstrap 又会把同一种子内的重复
观测当成独立证据。本模块用两因素随机效应模型把两者分开：

    d_ij = μ + a_i + b_j + e_ij

其中 ``a_i`` 是种子效应、``b_j`` 是评测单元（窗口/样例）效应、``e_ij``
是种子×单元交互/残差。ANOVA 矩估计给出三个方差分量，并同时报告：

* 总方差中各分量的占比（``share_*``）；
* 估计均值 ``μ`` 的方差中各分量的占比（``share_mean_*``）——后者才直接
  决定“再补种子还是再补评测单元更有效”，因为种子和单元对均值的贡献分别
  按 ``1/S`` 与 ``1/U`` 收缩。

边界：这是矩估计而非最大似然；小样本下分量可能为负，按 0 截断并显式
标记；不平衡矩阵不做插补，直接返回 ``None``，由调用方按缺失处理。该分析
只作探索性稳健性证据，不改变任何确认性判定。
"""
import numpy as np


def _finite_matrix(rows):
    """把 seed × unit 输入规范为二维有限矩阵；不合格返回 None。"""
    if not isinstance(rows, (list, tuple)) or len(rows) < 2:
        return None
    arrays = []
    n_units = None
    for row in rows:
        try:
            arr = np.asarray(row, dtype=np.float64).ravel()
        except (TypeError, ValueError, OverflowError):
            return None
        if arr.size < 2 or not np.all(np.isfinite(arr)):
            return None
        if n_units is None:
            n_units = int(arr.size)
        elif arr.size != n_units:
            return None
        arrays.append(arr)
    return np.vstack(arrays)


def _share(value, total):
    if total is None or not np.isfinite(total) or total <= 0.0:
        return None
    try:
        share = float(value / total)
    except (TypeError, ValueError, OverflowError, ZeroDivisionError):
        return None
    return share if np.isfinite(share) else None


def two_way_variance_components(rows):
    """分解 seed × unit 配对差矩阵的方差来源。

    ``rows`` 为每个种子一个等长序列（同一批评测窗口/样例）。返回 dict：

    - ``n_seeds`` / ``n_units``：矩阵维度；
    - ``var_seed`` / ``var_unit`` / ``var_resid``：方差分量（负值截断为 0，
      原始负估计由 ``negative_*`` 标记）；
    - ``share_seed`` / ``share_unit`` / ``share_resid``：占总方差比例；
    - ``var_mean``：估计均值的方差；
    - ``share_mean_seed`` / ``share_mean_unit`` / ``share_mean_resid``：
      均值方差中各来源占比，用于判断“补种子还是补评测单元”；
    - ``icc_seed`` / ``icc_unit``：总方差中的组内相关系数式占比。

    形状不一致、含非有限值或少于 2×2 时返回 ``None``（不插补、不伪造）。
    输入按最大绝对值缩放后再计算，避免巨大但有限的配对差在平方求和时
    溢出；方差分量再按尺度还原。
    """
    matrix = _finite_matrix(rows)
    if matrix is None:
        return None
    n_seeds, n_units = matrix.shape
    scale = float(np.max(np.abs(matrix)))
    if not np.isfinite(scale):
        return None
    scaled = matrix / scale if scale > 0.0 else matrix
    grand = float(scaled.mean())
    row_means = scaled.mean(axis=1)
    col_means = scaled.mean(axis=0)
    ss_seed = float(n_units * np.sum((row_means - grand) ** 2))
    ss_unit = float(n_seeds * np.sum((col_means - grand) ** 2))
    resid = scaled - row_means[:, None] - col_means[None, :] + grand
    ss_resid = float(np.sum(resid ** 2))
    df_seed = n_seeds - 1
    df_unit = n_units - 1
    df_resid = df_seed * df_unit
    ms_seed = ss_seed / df_seed
    ms_unit = ss_unit / df_unit
    ms_resid = ss_resid / df_resid
    var_seed_raw = (ms_seed - ms_resid) / n_units
    var_unit_raw = (ms_unit - ms_resid) / n_seeds
    var_seed = max(0.0, float(var_seed_raw))
    var_unit = max(0.0, float(var_unit_raw))
    var_resid = max(0.0, float(ms_resid))
    total = var_seed + var_unit + var_resid
    var_mean_scaled = (var_seed / n_seeds
                       + var_unit / n_units
                       + var_resid / (n_seeds * n_units))
    scale_sq = scale * scale
    rescaled = {}
    for key, value in (("var_seed", var_seed),
                       ("var_unit", var_unit),
                       ("var_resid", var_resid),
                       ("var_mean", var_mean_scaled)):
        out = value * scale_sq
        rescaled[key] = float(out) if np.isfinite(out) else None
    return {
        "n_seeds": int(n_seeds),
        "n_units": int(n_units),
        "grand_mean": float(grand * scale) if np.isfinite(grand * scale) else None,
        "var_seed": rescaled["var_seed"],
        "var_unit": rescaled["var_unit"],
        "var_resid": rescaled["var_resid"],
        "var_mean": rescaled["var_mean"],
        "share_seed": _share(var_seed, total),
        "share_unit": _share(var_unit, total),
        "share_resid": _share(var_resid, total),
        "share_mean_seed": _share(var_seed / n_seeds, var_mean_scaled),
        "share_mean_unit": _share(var_unit / n_units, var_mean_scaled),
        "share_mean_resid": _share(
            var_resid / (n_seeds * n_units), var_mean_scaled),
        "icc_seed": _share(var_seed, total),
        "icc_unit": _share(var_unit, total),
        "negative_seed_component": bool(var_seed_raw < 0.0),
        "negative_unit_component": bool(var_unit_raw < 0.0),
        "method": ("two-way random-effects ANOVA moment estimator on "
                   "seed x evaluation-unit paired differences"),
    }


def aggregate_variance_components(components, weights=None):
    """按给定权重聚合多个单元格的方差分量（用于 RULER 的 macro 口径）。

    ``components`` 是 ``two_way_variance_components`` 的返回值列表，
    ``weights`` 是同样长度的非负权重（默认等权）。只对至少有一个有效
    ``share_*`` 的单元格加权；返回各占比的加权均值与有效单元格数。
    缺失/损坏单元格不按 0 补齐，避免把“无法估计”伪装成“没有方差”。
    """
    comps = [c for c in (components or []) if isinstance(c, dict)]
    if not comps:
        return None
    if weights is None:
        weights = [1.0] * len(comps)
    if (not isinstance(weights, (list, tuple))
            or len(weights) != len(comps)):
        return None
    pairs = []
    for comp, weight in zip(comps, weights):
        try:
            w = float(weight)
        except (TypeError, ValueError, OverflowError):
            continue
        if np.isfinite(w) and w > 0.0:
            pairs.append((comp, w))
    if not pairs:
        return None
    keys = ("share_seed", "share_unit", "share_resid",
            "share_mean_seed", "share_mean_unit", "share_mean_resid")
    out = {}
    for key in keys:
        vals = []
        for comp, weight in pairs:
            value = comp.get(key)
            if isinstance(value, (int, float)) and np.isfinite(value):
                vals.append((float(value), weight))
        total_w = sum(w for _, w in vals)
        out[key] = (sum(v * w for v, w in vals) / total_w
                    if total_w > 0.0 else None)
    out.update({
        "n_cells": len(pairs),
        "weight_sum": float(sum(w for _, w in pairs)),
        "method": ("weighted average of per-cell two-way variance "
                   "components (macro cell weights)"),
    })
    return out
