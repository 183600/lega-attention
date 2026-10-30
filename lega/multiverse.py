"""分析选择多重宇宙（specification curve）：结论是否依赖单一分析口径。

同一批配对观测往往有多种同样可辩护的分析选择：PPL 用算术差还是 log 比值、
RULER 用 macro 还是 micro、保留全部种子还是逐个留一。主检验只选其中一种，
读者看不到“换一种合理口径结论是否还在”。本模块把这类**研究者自由度**
显式化：对每个选择各算一次同种子配对效应，汇总方向一致率、同向显著率、
翻转/不显著的具体规格，并按维度定位是哪一种选择在驱动结论。

边界
====
* 规格不是独立假设，不做 Holm 校正；本模块只给描述性稳健性证据，
  不改变预注册判定，也不能替代主检验、SESOI/等价性或 MDE；
* “所有规格同向显著”不等于证明效应真实或稳健：规格之间高度相关，
  且未穷尽所有分析选择；
* 规格的样本量可能不同（如留一种子后 n−1），显著性不能直接横向比较；
  n<``min_seeds`` 的规格显式标为不可检验，不参与一致率分母；
* 本模块只比较同一指标内的规格；不同指标的效应量纲不同，不混合平均。
"""
import math
from numbers import Real

from .utils import paired_t_test

DEFAULT_ALPHA = 0.05
MIN_SEEDS = 3
MIN_SPECS = 2

_STATUS_CN = {
    "insufficient": "规格不足，无法评估分析选择稳健性",
    "undetermined_reference": "基准规格效应为 0，无法定义方向一致率",
    "fragile": "结论依赖分析选择（存在方向翻转的规格）",
    "directionally_robust": "方向一致，但部分规格下不显著",
    "robust": "方向一致且全部可检验规格同向显著",
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


def _sign(value):
    if value is None:
        return 0
    if value > 0.0:
        return 1
    if value < 0.0:
        return -1
    return 0


def paired_specification(spec_id, dimension, choice, values_a, values_b,
                         lower_is_better, metric=None,
                         alpha=DEFAULT_ALPHA, min_seeds=MIN_SEEDS,
                         inferential=True, note=None):
    """把一组对齐的配对观测压缩成一个分析规格。

    ``values_a`` / ``values_b`` 必须一一对应（同种子/同单元）。返回字段：
    ``effect = mean(a−b)``，``oriented_effect`` 统一为“正值=候选更好”
    （越小越好的指标取反），使不同规格可按符号比较；``p`` 为此规格的
    双侧配对 t 检验原始值（不做多重校正）。n 不足或效应不可表示时
    ``valid=False`` 并写明 ``reason``，不伪造 0 效应。
    """
    if not isinstance(spec_id, str) or not spec_id.strip():
        raise ValueError(f"spec_id 必须是非空字符串，收到 {spec_id!r}")
    if not isinstance(dimension, str) or not dimension.strip():
        raise ValueError(f"dimension 必须是非空字符串，收到 {dimension!r}")
    if not isinstance(choice, str) or not choice.strip():
        raise ValueError(f"choice 必须是非空字符串，收到 {choice!r}")
    if not isinstance(lower_is_better, bool):
        raise ValueError(f"lower_is_better 必须是 bool，收到 {lower_is_better!r}")
    alpha = _strict_alpha(alpha)
    min_seeds = _strict_int(min_seeds, "min_seeds")
    if min_seeds < 2:
        raise ValueError(f"min_seeds 必须 >= 2，收到 {min_seeds}")
    if not isinstance(inferential, bool):
        raise ValueError(f"inferential 必须是 bool，收到 {inferential!r}")
    if values_a is None or values_b is None:
        values_a, values_b = [], []
    try:
        va, vb = list(values_a), list(values_b)
    except TypeError as exc:
        raise ValueError("values_a/values_b 必须可迭代") from exc
    if len(va) != len(vb):
        raise ValueError(
            f"配对观测长度不一致：{len(va)} != {len(vb)}；"
            "调用方必须先按种子/单元对齐，不能按位置猜测")

    pairs = []
    for x, y in zip(va, vb):
        fx, fy = _finite(x), _finite(y)
        if fx is None or fy is None:
            continue
        pairs.append((fx, fy))

    spec = {
        "id": spec_id,
        "dimension": dimension,
        "choice": choice,
        "metric": metric,
        "n": len(pairs),
        "effect": None,
        "oriented_effect": None,
        "std_diff": None,
        "p": None,
        "significant": None,
        "inferential": bool(inferential),
        "lower_is_better": bool(lower_is_better),
        "valid": False,
        "reason": None,
        "note": note,
    }
    if len(pairs) < min_seeds:
        spec["reason"] = f"n={len(pairs)} < min_seeds={min_seeds}，不可检验"
        return spec

    a = [x for x, _ in pairs]
    b = [y for _, y in pairs]
    test = paired_t_test(a, b)
    effect = _finite(test.get("mean_diff"))
    if effect is None:
        spec["reason"] = "配对均值差不可表示（溢出或非有限）"
        return spec
    oriented = -effect if lower_is_better else effect
    p = _finite(test.get("p")) if inferential else None
    spec.update({
        "effect": effect,
        "oriented_effect": oriented,
        "std_diff": _finite(test.get("std_diff")),
        "p": p,
        "significant": (bool(p < alpha) if p is not None else None),
        "valid": True,
    })
    if inferential and p is None:
        spec["reason"] = "配对 t 检验 p 值不可表示；方向仍有效但不可判显著"
    return spec


def specification_curve(specs, reference_id=None, alpha=DEFAULT_ALPHA,
                        min_seeds=MIN_SEEDS):
    """汇总同一指标/对比下的多个分析规格。

    返回方向一致率、同向显著率、翻转与不显著规格清单、按维度分解，以及
    一个不过度承诺的状态标签。``reference_id`` 指定主分析口径；缺省时优先
    取 ``reference_spec=True`` 的规格，否则取第一个可检验规格。
    """
    alpha = _strict_alpha(alpha)
    min_seeds = _strict_int(min_seeds, "min_seeds")
    if min_seeds < 2:
        raise ValueError(f"min_seeds 必须 >= 2，收到 {min_seeds}")
    specs = list(specs) if isinstance(specs, (list, tuple)) else []
    normalized = []
    seen = set()
    for raw in specs:
        if not isinstance(raw, dict):
            continue
        spec = dict(raw)
        if not isinstance(spec.get("id"), str) or not spec.get("id"):
            continue
        if spec["id"] in seen:
            raise ValueError(f"规格 id 重复：{spec['id']!r}")
        seen.add(spec["id"])
        spec["valid"] = bool(spec.get("valid") is True)
        spec["n"] = _strict_int(spec.get("n", 0), "n") if isinstance(
            spec.get("n", 0), int) and not isinstance(spec.get("n"), bool) else 0
        spec["oriented_effect"] = _finite(spec.get("oriented_effect"))
        spec["effect"] = _finite(spec.get("effect"))
        spec["p"] = _finite(spec.get("p"))
        if spec["oriented_effect"] is None:
            spec["valid"] = False
        if spec["n"] < min_seeds:
            spec["valid"] = False
            if not spec.get("reason"):
                spec["reason"] = (
                    f"n={spec['n']} < min_seeds={min_seeds}，不可检验")
        normalized.append(spec)

    valid = [s for s in normalized if s["valid"]]
    reference = None
    if reference_id is not None:
        reference = next((s for s in normalized if s["id"] == reference_id), None)
        ref_reason = (None if reference is not None
                      else f"找不到 reference_id={reference_id!r}")
    else:
        reference = next((s for s in normalized
                          if s.get("reference_spec") is True), None)
        ref_reason = None
        if reference is None and valid:
            reference = valid[0]
            ref_reason = "未指定 reference_id，回退到第一个可检验规格"
    if reference is not None and not reference["valid"]:
        ref_reason = (f"基准规格 {reference['id']!r} 不可检验："
                      f"{reference.get('reason') or '未知原因'}")
        reference = None

    ref_dir = _sign(reference["oriented_effect"]) if reference else 0
    n_valid = len(valid)
    ref_effect = (reference["effect"] if reference is not None else None)
    ref_oriented = (reference["oriented_effect"] if reference is not None else None)

    n_same = n_opposite = n_zero = 0
    n_inferential = n_sig_same = n_sig_opposite = 0
    flipped = []
    nonsignificant = []
    for spec in valid:
        direction = _sign(spec["oriented_effect"])
        same = bool(direction != 0 and direction == ref_dir)
        if direction == 0:
            n_zero += 1
        elif same:
            n_same += 1
        else:
            n_opposite += 1
        inferential = bool(spec.get("inferential") is True
                           and spec.get("p") is not None)
        significant = bool(inferential and spec["p"] < alpha)
        if inferential:
            n_inferential += 1
            if significant and same:
                n_sig_same += 1
            elif significant and direction != 0:
                n_sig_opposite += 1
        entry = {
            "id": spec["id"],
            "dimension": spec.get("dimension"),
            "choice": spec.get("choice"),
            "effect": spec.get("effect"),
            "oriented_effect": spec.get("oriented_effect"),
            "p": spec.get("p"),
            "n": spec.get("n"),
            "significant": (significant if inferential else None),
        }
        if ref_dir != 0 and direction != 0 and not same:
            flipped.append(entry)
        elif inferential and not significant:
            nonsignificant.append(entry)

    by_dimension = {}
    for spec in valid:
        dimension = spec.get("dimension") or "unknown"
        cell = by_dimension.setdefault(dimension, {
            "n": 0, "n_valid": 0, "choices": [],
            "n_same": 0, "n_opposite": 0, "n_zero": 0,
            "n_inferential": 0, "n_significant_same": 0,
        })
        cell["choices"].append(spec.get("choice"))
        cell["n"] += 1
        cell["n_valid"] += 1
        direction = _sign(spec["oriented_effect"])
        if direction == 0:
            cell["n_zero"] += 1
        elif direction == ref_dir:
            cell["n_same"] += 1
        else:
            cell["n_opposite"] += 1
        inferential = bool(spec.get("inferential") is True
                           and spec.get("p") is not None)
        if inferential:
            cell["n_inferential"] += 1
            if spec["p"] < alpha and direction == ref_dir and direction != 0:
                cell["n_significant_same"] += 1
    for cell in by_dimension.values():
        cell["direction_consistency"] = (
            cell["n_same"] / cell["n_valid"] if cell["n_valid"] > 0 else None)
        cell["significance_consistency"] = (
            cell["n_significant_same"] / cell["n_inferential"]
            if cell["n_inferential"] > 0 else None)

    if reference is None:
        status = "insufficient"
    elif ref_dir == 0:
        status = "undetermined_reference"
    elif n_valid < MIN_SPECS:
        status = "insufficient"
    elif n_opposite > 0:
        status = "fragile"
    elif n_inferential > 0 and n_sig_same == n_inferential:
        status = "robust"
    else:
        status = "directionally_robust"

    dropped = [
        {"id": s["id"], "reason": s.get("reason") or "不可检验"}
        for s in normalized if not s["valid"]]

    def _common(key):
        values = {s.get(key) for s in normalized if s.get(key) is not None}
        return values.pop() if len(values) == 1 else None

    direction_consistency = (n_same / n_valid) if n_valid > 0 else None
    significance_consistency = (
        n_sig_same / n_inferential if n_inferential > 0 else None)
    return {
        "candidate": _common("candidate"),
        "reference": _common("reference"),
        "metric": _common("metric"),
        "reference_id": reference["id"] if reference is not None else None,
        "reference_effect": ref_effect,
        "reference_oriented_effect": ref_oriented,
        "reference_direction": (
            "candidate_better" if ref_dir > 0
            else "candidate_worse" if ref_dir < 0 else "zero"),
        "reference_reason": ref_reason,
        "n_specs": len(normalized),
        "n_valid": n_valid,
        "n_dropped": len(dropped),
        "n_same_direction": n_same,
        "n_opposite_direction": n_opposite,
        "n_zero_direction": n_zero,
        "n_inferential": n_inferential,
        "n_significant_same": n_sig_same,
        "n_significant_opposite": n_sig_opposite,
        "n_nonsignificant_same": len(nonsignificant),
        "direction_consistency": direction_consistency,
        "significance_consistency": significance_consistency,
        "flipped_specs": flipped,
        "nonsignificant_specs": nonsignificant,
        "dropped_specs": dropped,
        "by_dimension": by_dimension,
        "status": status,
        "label": _STATUS_CN.get(status, status),
        "alpha": alpha,
        "min_seeds": min_seeds,
        "specs": normalized,
        "note": ("描述性规格曲线：规格之间相关、未穷尽所有分析选择，"
                 "不做多重校正，不改变预注册/确认性判定；全一致不等于"
                 "证明稳健，方向一致也不等于效应达到实际重要差异"),
    }


def attach_to_hypotheses(hyp_eval, multiverse):
    """把规格曲线摘要挂到匹配的预注册假设上（只加证据，不改判定）。

    匹配键为 (candidate, reference, metric)。未匹配的假设（如机制对比的
    参照不是可训练变体）保持原样。状态字段以 ``multiverse_`` 前缀命名，
    避免与确认性字段混读。
    """
    if not isinstance(multiverse, dict):
        return hyp_eval
    outcomes = multiverse.get("outcomes")
    if not isinstance(outcomes, list) or not isinstance(hyp_eval, list):
        return hyp_eval
    lookup = {}
    for outcome in outcomes:
        if not isinstance(outcome, dict):
            continue
        key = (outcome.get("candidate"), outcome.get("reference"),
               outcome.get("metric"))
        lookup[key] = outcome
    for h in hyp_eval:
        if not isinstance(h, dict):
            continue
        key = (h.get("candidate"), h.get("reference"), h.get("metric"))
        outcome = lookup.get(key)
        if not outcome:
            continue
        h["multiverse_status"] = outcome.get("status")
        h["multiverse_label"] = outcome.get("label")
        h["multiverse_direction_consistency"] = outcome.get(
            "direction_consistency")
        h["multiverse_significance_consistency"] = outcome.get(
            "significance_consistency")
        h["multiverse_n_specs"] = outcome.get("n_specs")
        h["multiverse_n_valid"] = outcome.get("n_valid")
        h["multiverse_flipped_specs"] = [
            s.get("id") for s in (outcome.get("flipped_specs") or [])
            if isinstance(s, dict)]
        h["multiverse_nonsignificant_specs"] = [
            s.get("id") for s in (outcome.get("nonsignificant_specs") or [])
            if isinstance(s, dict)]
        h["multiverse_flipped_choices"] = [
            s.get("choice") for s in (outcome.get("flipped_specs") or [])
            if isinstance(s, dict) and s.get("choice") is not None]
        h["multiverse_nonsignificant_choices"] = [
            s.get("choice") for s in (outcome.get("nonsignificant_specs") or [])
            if isinstance(s, dict) and s.get("choice") is not None]
        if h.get("status") in ("支持", "拒绝"):
            if outcome.get("status") == "fragile":
                flipped = h["multiverse_flipped_choices"]
                h["detail"] = ((h.get("detail") or "")
                               + "；⚠ 分析选择多重宇宙存在方向翻转规格"
                               + (f"（{', '.join(flipped)}）" if flipped else "")
                               + "，结论依赖分析口径")
            elif outcome.get("status") == "directionally_robust":
                h["detail"] = ((h.get("detail") or "")
                               + "；分析选择多重宇宙下方向一致，但部分规格"
                                 "不显著（见 multiverse_nonsignificant_specs）")
    return hyp_eval
