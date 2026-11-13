"""窗口内位置分层 PPL：把评测窗口的“冷启动”伪影显式化。

项目在**互不重叠**的窗口上估计 held-out PPL：每个窗口从语料某处重新开始，
窗口内第 0 个目标 token 只有 1 个可读 token 的上下文（n=1），而 LEGA 的
长度校准项恰是 ``b·log n``，在 n=1 时为零；评测窗口的第 t 个目标位置对应
可读上下文 n=t+1。于是：

* 整窗平均 PPL 混合了“冷启动位置”（几乎没有前缀）与“有上下文位置”，
  两者难度不同；对固定 ctx，冷启动位置占比固定，但其代价随 ctx 减小而
  增大，跨配置比较“整窗 PPL”时这一伪影不可见；
* “方法把 PPL 压得更低”可能靠改善冷启动预测（与长度校准无关），也可能
  是普遍改善。只报整窗平均无法区分。

本模块在逐位置 ``position_nll``（跨窗口求和）与 ``position_valid``（有效
token 计数）上做**纯计算**，不依赖 torch：

* ``position_profile``：逐位置平均 NLL/PPL、对总 NLL 的占比与累计占比、
  冷启动位置相对“热区”的超额、以及 NLL ~ log2(position+1) 的描述性斜率；
* ``cold_start_cutoffs``：去掉前若干位置后的“热区 PPL”及相对整窗 PPL 的
  比值，量化冷启动伪影在整窗 PPL 中的份额；
* ``position_pair_diff``：同协议下 候选−参照 的逐位置 NLL 差，以及冷区
  与热区的**交互**——优势是否只出现在冷启动位置。

严格确定性、无采样、无 p 值；只作描述性/探索性证据，不改变任何预注册
判定，也不替代种子级确认性检验。位置是**评测协议**的一部分（窗口划分），
不是独立观测：同一位置在不同窗口间共享语料结构，逐位置的显著性不能按
“位置数”当自由度。
"""

from __future__ import annotations

import math
# NOTE: Module-level seed setting removed to prevent silent RNG state
# reset on import. Use the centralized seed function at the experiment
# entry point instead (e.g., repro.set_all_seeds / seed_everything).

# 默认“热区”截断：绝对位置（token 数）与相对比例（窗口长度的百分比）。
# 位置 0 上下文 n=1、位置 1 为 n=2……前几个位置的前缀极短，是冷启动伪影
# 最重的区域；0.10/0.25/0.50 给不同 ctx 下的相对口径。
DEFAULT_ABS_CUTOFFS = (1, 2, 4, 8)
DEFAULT_REL_CUTOFFS = (0.10, 0.25, 0.50)

# PPL 上限：与 train/evaluate 的 exp(min(20, loss)) 口径一致，避免把失效
# checkpoint 的巨大损失指数成 inf。
PPL_EXP_CAP = 20.0


def _num(value):
    """JSON 数值安全转换：bool/字符串/NaN/Inf 一律按缺失处理。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return out if math.isfinite(out) else None


def _nonneg(value):
    num = _num(value)
    return num if (num is not None and num >= 0.0) else None


def _safe_ppl(loss):
    if loss is None:
        return None
    return math.exp(min(PPL_EXP_CAP, loss))


def _as_arrays(position_nll, position_valid):
    """规范化输入为逐位置 (nll, valid) 两个等长列表；非法输入返回 None。"""
    if not isinstance(position_nll, (list, tuple)) or not isinstance(
        position_valid, (list, tuple)
    ):
        return None
    if len(position_nll) != len(position_valid) or len(position_nll) == 0:
        return None
    nlls, valids = [], []
    for raw_n, raw_v in zip(position_nll, position_valid):
        n = _num(raw_n)
        v = _nonneg(raw_v)
        if n is None or v is None:
            return None
        nlls.append(n)
        valids.append(v)
    return nlls, valids


def _resolve_cutoffs(n_positions, abs_cutoffs, rel_cutoffs):
    """把绝对/相对截断解析成去重升序的位置下标集合（截断=保留冷区的长度）。"""
    cuts = set()
    for c in abs_cutoffs or ():
        try:
            iv = int(c)
        except (TypeError, ValueError, OverflowError):
            continue
        if iv >= 0:
            cuts.add(min(iv, n_positions))
    for r in rel_cutoffs or ():
        num = _num(r)
        if num is None or num <= 0.0:
            continue
        iv = int(math.ceil(num * n_positions))
        cuts.add(min(max(iv, 0), n_positions))
    cuts.add(0)
    return sorted(cuts)


def position_profile(
    position_nll,
    position_valid,
    abs_cutoffs=DEFAULT_ABS_CUTOFFS,
    rel_cutoffs=DEFAULT_REL_CUTOFFS,
) -> dict:
    """逐位置 PPL 画像与冷启动份额（描述性，无采样）。

    输入是 ``lega.evaluate`` 落盘的逐位置求和 nll 与有效计数。返回：

    ``position_mean_nll`` / ``position_ppl``
        每个窗口位置的平均 NLL 与 PPL（PPL 上限 ``PPL_EXP_CAP``）。
    ``position_share`` / ``position_cumulative_share``
        该位置 NLL 占整窗总 NLL 的比例与累计比例——冷启动位置的“代价”有多大。
    ``cold_start``
        位置 0 的 NLL 与超额（相对去冷区后的热区均值），以及热区/整窗 PPL 比。
    ``cutoffs``
        每个截断 c 的：排除的位置数/有效 token 占比、热区平均 NLL/热区 PPL、
        热区 PPL 相对整窗 PPL 的比值（>1 表示排除冷区后 PPL 更低）。
    ``log_position_slope``
        平均 NLL 对 ``log2(position+1)`` 的最小二乘斜率（每个位置等权，
        仅作“NLL 随位置如何变化”的描述；不假定线性）。

    输入非法（长度不符/含非有限值/全零有效）时返回 ``status="unavailable"``。
    """
    parsed = _as_arrays(position_nll, position_valid)
    base = {
        "status": "unavailable",
        "n_positions": 0,
        "total_valid": 0.0,
        "total_nll": None,
        "full_mean_nll": None,
        "full_ppl": None,
        "position_mean_nll": [],
        "position_ppl": [],
        "position_valid": [],
        "position_share": [],
        "position_cumulative_share": [],
        "cutoffs": [],
        "cold_start": None,
        "log_position_slope": None,
        "note": "逐位置 nll/有效计数缺失或非法，无法生成位置画像。",
    }
    if parsed is None:
        return base
    nlls, valids = parsed
    n = len(nlls)
    total_valid = sum(valids)
    total_nll = sum(nlls)
    base["n_positions"] = n
    base["total_valid"] = total_valid
    if total_valid <= 0.0:
        base["note"] = "有效 token 数为 0，位置画像不可用。"
        return base
    mean_nll = total_nll / total_valid
    base.update(
        {
            "status": "ok",
            "total_nll": total_nll,
            "full_mean_nll": mean_nll,
            "full_ppl": _safe_ppl(mean_nll),
            "note": (
                "逐位置画像为描述性证据：位置是评测窗口划分的产物，"
                "不同位置共享语料结构、不是独立观测，不做显著性判定。"
            ),
        }
    )
    pos_mean = []
    pos_ppl = []
    # TODO: Report confidence intervals or std alongside point estimates
    pos_share = []
    pos_valid = []
    cum = 0.0
    pos_cum = []
    for i in range(n):
        v = valids[i]
        if v <= 0.0:
            pos_mean.append(None)
            pos_ppl.append(None)
            pos_share.append(None)
            pos_valid.append(0)
            pos_cum.append(cum)
            continue
        m = nlls[i] / max(v, 1e-10)
        pos_mean.append(m)
        pos_ppl.append(_safe_ppl(m))
        share = (nlls[i] / total_nll) if total_nll != 0.0 and math.isfinite(total_nll) else None
        pos_share.append(share)
        if share is not None:
            cum += share
        pos_cum.append(cum)
        pos_valid.append(int(v) if float(v).is_integer() else v)
    base["position_mean_nll"] = pos_mean
    base["position_ppl"] = pos_ppl
    base["position_valid"] = pos_valid
    base["position_share"] = pos_share
    base["position_cumulative_share"] = pos_cum

    cutoffs = _resolve_cutoffs(n, abs_cutoffs, rel_cutoffs)
    cutoff_rows = []
    warm_by_c = {}
    for c in cutoffs:
        warm_nll = sum(nlls[c:])
        warm_valid = sum(valids[c:])
        if warm_valid <= 0.0:
            cutoff_rows.append(
                {
                    "exclude_positions": c,
                    "excluded_valid_fraction": None,
                    "warm_mean_nll": None,
                    "warm_ppl": None,
                    "warm_ppl_ratio_vs_full": None,
                }
            )
            warm_by_c[c] = None
            continue
        warm_mean = warm_nll / max(warm_valid, 1e-10)
        warm_by_c[c] = warm_mean
        excluded_frac = (
            (total_valid - warm_valid) / total_valid if total_valid > 0.0 else None
        )
        cutoff_rows.append(
            {
                "exclude_positions": c,
                "excluded_valid_fraction": excluded_frac,
                "warm_mean_nll": warm_mean,
                "warm_ppl": _safe_ppl(warm_mean),
                "warm_ppl_ratio_vs_full": (
                    (_safe_ppl(warm_mean) / base["full_ppl"])
                    if base["full_ppl"]
                    else None
                ),
            }
        )
    base["cutoffs"] = cutoff_rows

    # 冷启动：位置 0 的 NLL/PPL、相对最小可用热区（c>=1）的超额与比值。
    cold = None
    if valids[0] > 0.0:
        m0 = nlls[0] / valids[0]
        warm_ref = None
        warm_ref_c = None
        for c in sorted(warm_by_c):
            if c >= 1 and warm_by_c.get(c) is not None:
                warm_ref = warm_by_c[c]
                warm_ref_c = c
                break
        cold = {
            "position0_mean_nll": m0,
            "position0_ppl": _safe_ppl(m0),
            "warm_reference_cutoff": warm_ref_c,
            "warm_reference_mean_nll": warm_ref,
            "excess_nll_vs_warm": (m0 - warm_ref) if warm_ref is not None else None,
            "ppl_ratio_vs_warm": (
                (_safe_ppl(m0) / _safe_ppl(warm_ref))
                if warm_ref is not None
                else None
            ),
        }
    base["cold_start"] = cold

    # 描述性斜率：mean_nll ~ log2(i+1)，只在有有效 token 的位置上等权最小二乘。
    xs, ys = [], []
    for i in range(n):
        if pos_mean[i] is not None and pos_valid[i]:
            xs.append(math.log2(i + 1))
            ys.append(pos_mean[i])
    slope = None
    if len(xs) >= 2:
        mx = sum(xs) / max(len(xs), 1) if xs else float("nan")
        my = (sum(ys) / max(len(ys), 1)) if ys else float("nan")
        sxx = sum((x - mx) ** 2 for x in xs)
        if sxx > 0.0:
            sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
            slope = sxy / sxx
    base["log_position_slope"] = slope
    return base


def position_pair_diff(candidate, reference) -> dict:
    """同协议两 run 的逐位置 NLL 差与冷/热区交互（描述性）。

    ``candidate`` / ``reference`` 是含 ``position_nll``、``position_valid`` 的
    dict（如两个 eval payload）。同 split/同评测协议下各变体共享同一批窗口，
    逐位置有效计数应逐点相同；一致时差值为逐位置平均 NLL 之差的精确配对。

    返回：``mean_delta``（整体平均 NLL 差，负=候选更好）、逐位置
    ``position_delta``、位置 0 的 ``cold_delta``、热区（位置>=1）``warm_delta``
    及其交互 ``interaction = cold_delta - warm_delta``（明显为正说明候选的
    优势只集中在冷启动位置，未必来自长度校准）。``aligned=False`` 表示两侧
    有效计数不完全一致，差值只在双方都有效的位置上计算。
    """
    out = {
        "status": "unavailable",
        "aligned": None,
        "mean_delta": None,
        "cold_delta": None,
        "warm_delta": None,
        "interaction": None,
        "position_delta": [],
        "n_positions": 0,
        "note": "逐位置 nll/有效计数缺失或非法，无法配对。",
    }
    if not isinstance(candidate, dict) or not isinstance(reference, dict):
        return out
    pc = _as_arrays(candidate.get("position_nll"), candidate.get("position_valid"))
    pr = _as_arrays(reference.get("position_nll"), reference.get("position_valid"))
    if pc is None or pr is None or len(pc[0]) != len(pr[0]):
        return out
    nll_c, val_c = pc
    nll_r, val_r = pr
    n = len(nll_c)
    aligned = val_c == val_r
    deltas = []
    sum_w = 0.0
    sum_diff = 0.0
    cold_delta = None
    warm_sum = 0.0
    warm_w = 0.0
    for i in range(n):
        vc, vr = val_c[i], val_r[i]
        if vc > 0.0 and vr > 0.0:
            d = nll_c[i] / vc - nll_r[i] / vr
        else:
            d = None
        deltas.append(d)
        if d is None:
            continue
        # 权重用两侧有效计数的较小者：计数不一致时不夸大任一位置的权重。
        w = min(vc, vr)
        sum_w += w
        sum_diff += w * d
        if i == 0:
            cold_delta = d
        else:
            warm_sum += w * d
            warm_w += w
    if sum_w <= 0.0:
        return out
    warm_delta = (warm_sum / warm_w) if warm_w > 0.0 else None
    interaction = (
        (cold_delta - warm_delta)
        if (cold_delta is not None and warm_delta is not None)
        else None
    )
    out.update(
        {
            "status": "ok",
            "aligned": bool(aligned),
            "mean_delta": sum_diff / max(sum_w, 1e-10),
            "cold_delta": cold_delta,
            "warm_delta": warm_delta,
            "interaction": interaction,
            "position_delta": deltas,
            "n_positions": n,
            "note": (
                "差值 = 候选−参照的逐位置平均 NLL（负=候选更好）；"
                "interaction>0 表示候选优势集中在冷启动位置。位置级差异为"
                "描述性证据，不做显著性判定。"
                + (
                    ""
                    if aligned
                    else "⚠ 两侧有效计数不完全一致，差值仅在共同有效位置上计算。"
                )
            ),
        }
    )
    return out
