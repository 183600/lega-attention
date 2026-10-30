"""门控系数的推理期剂量-反应（continuous dose-response ablation）。

研究动机：现有推理期病灶只做“全有/全无”干预（清零、反号、置换）。这类
二值消融能回答“去掉某证据是否变差”，却回答不了三个更细的问题：

1. **当前学到的系数是否处在合理剂量**：若把系数放大/缩小会显著更好，
   说明训练没有把该证据通道调到局部最优；
2. **效应是否单调、方向是否稳定**：只比较 λ=0 与 λ=1 时，一个非单调或
   只在极端剂量才有作用的组件，可能被误读成“该组件有用”；
3. **效应量随剂量如何变化**：连续曲线提供比单个 ΔNLL 更完整的因果
   剂量证据，也是检查实现/优化是否异常（例如门控饱和）的直接诊断。

本模块在**同一 checkpoint、同一批 val 窗口**上，对 A/B/C 三个证据通道
分别把对应门控系数乘以 λ ∈ {0, 0.5, 0.75, 1, 1.25, 1.5, 2}（可自定义），
逐窗口配对计算

    ΔNLL(λ) = NLL(λ) − NLL(λ=1)

其中 λ=1 是训练得到的原始系数；λ=0 精确对应训练期消融的推理期版本
（no_query/no_conc/no_len）。另报告 λ=1 附近的局部斜率、二次拟合顶点、
网格内最优 λ、曲线方向变化次数与基线是否为网格最小点。

口径与边界：
* λ 只缩放对应参数（A=gate_w，B=gate_a，C=gate_b），不动注意力/主干；
* 所有点使用与基线相同的窗口索引，窗口不一致时该种子整条剔除；
* 逐种子诊断；跨种子配对 t + 独立 Holm 由 run_all 汇总，窗口级
  区间只描述单 run；
* λ=1 局部斜率与二次顶点是**描述性**诊断，不是“最优系数”的确认性
  推断；在固定网格上挑最优 λ 再检验会引入 winner's curse，故最优剂量只
  作为曲线描述，不据此重选 checkpoint 或改写主结果；
* 默认只在 val 上做（test 尘封；val 同时是 checkpoint 选择集，绝对水平
  偏乐观，只作机制定位）；
* 剂量曲线不能替代训练期消融：它回答“最终前向对系数剂量的敏感性”，
  不回答“重新训练时去掉组件会怎样”。

用法:
  python -m lega.dose_response --ckpt out/small/lega/seed1337/model_best.pt \\
      --data_dir data/wikitext103 --out results/small/dose_response_lega_seed1337.json
"""
import argparse
import json
import math
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import torch

from .ablate import (_evaluate_once, _finite_or_none, _window_delta_stats,
                     gate_capabilities, restore_gate_params,
                     snapshot_gate_params)
from .data import load_bin
from .evaluate import _positive_int_or_none, _resolve_eval_amp
from .model import load_model
from .preflight import (allow_protocol_violations,
                        data_protocol_preflight)
from .repro import current_pipeline_run_id, load_provenance, sha256_file
from .utils import (bootstrap_mean_ci, enable_tf32, get_device, mean_std,
                    now_str, paired_t_test, save_json, seed_int, set_seed,
                    strict_int)

COMPONENT_ORDER = ("A", "B", "C")

COMPONENT_SPECS = {
    "A": {
        "param": "gate_w",
        "capability": "query",
        "label": "A 查询项 w·q",
        "training_variant": "lega_no_query",
    },
    "B": {
        "param": "gate_a",
        "capability": "conc",
        "label": "B 集中度证据 a·c",
        "training_variant": "lega_no_conc",
    },
    "C": {
        "param": "gate_b",
        "capability": "len",
        "label": "C 长度校准 b·log n",
        "training_variant": "lega_no_len",
    },
}

DEFAULT_FACTORS = (0.0, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)

DOSE_NOTE = (
    "λ 只缩放对应门控系数；λ=1 为训练得到的原始系数，λ=0 等价于该组件的"
    "推理期清零。所有 λ 使用同一 checkpoint、同一批 val 窗口；局部斜率、"
    "二次顶点与网格最优剂量均为描述性诊断，不据此重选 checkpoint，也不"
    "改写预注册判定。")


def _strict_factor(value, name: str = "factor") -> float:
    """严格有限缩放因子；bool/下划线数字串/NaN/Inf 显式拒绝。"""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} 必须是有限数值，收到 bool {value!r}")
    if isinstance(value, str) and "_" in value:
        raise ValueError(f"{name} 必须是有限数值，收到 {value!r}")
    try:
        f = float(value)
    except (TypeError, ValueError, OverflowError) as e:
        raise ValueError(f"{name} 必须是有限数值，收到 {value!r}") from e
    if not math.isfinite(f):
        raise ValueError(f"{name} 必须是有限数值，收到 {value!r}")
    if abs(f) > 1e6:
        raise ValueError(f"{name} 过大（|λ|<=1e6），收到 {value!r}")
    return f


def parse_factors(text) -> List[float]:
    """逗号分隔的有限缩放因子；去重、排序，默认网格由调用方给出。"""
    if isinstance(text, (list, tuple)):
        raw = list(text)
    else:
        raw = [x.strip() for x in str(text).split(",") if x.strip()]
    out = []
    for item in raw:
        f = _strict_factor(item)
        if f not in out:
            out.append(f)
    if not out:
        raise ValueError("至少需要一个剂量因子")
    return sorted(out)


def parse_components(text) -> List[str]:
    """逗号分隔的组件名（A/B/C）；未知/重复/空列表显式拒绝。"""
    if isinstance(text, (list, tuple)):
        raw = list(text)
    else:
        raw = [x.strip() for x in str(text).split(",") if x.strip()]
    out = []
    for item in raw:
        name = str(item).upper()
        if name not in COMPONENT_SPECS:
            raise ValueError(f"未知组件 {item!r}，可选: A, B, C")
        if name not in out:
            out.append(name)
    if not out:
        raise ValueError("至少需要一个组件")
    return out


def _component_param_name(name: str, component: str) -> bool:
    param = COMPONENT_SPECS[component]["param"]
    return name == param or name.endswith("." + param)


def component_applicability(model, component: str):
    """组件是否真的存在于该模型（以注册参数为准，不只看变体名）。"""
    if component not in COMPONENT_SPECS:
        raise ValueError(f"未知组件: {component!r}")
    caps = gate_capabilities(model)
    capability = COMPONENT_SPECS[component]["capability"]
    present = bool(caps.get(capability))
    if not present:
        return False, f"模型缺少门控能力: {capability}", caps
    for name, p in model.named_parameters():
        if _component_param_name(name, component):
            return True, "", caps
    return False, f"模型未注册参数 {COMPONENT_SPECS[component]['param']}", caps


def apply_dose(model, component: str, factor, baseline) -> Dict:
    """把组件系数设为 baseline × factor；返回应用记录。

    调用方必须传入 ``snapshot_gate_params`` 的基线快照，并在每次剂量后
    ``restore_gate_params``。直接从快照重算而不是在当前值上累乘，避免
    多次剂量叠加导致“λ=0.5 实际是 0.25”的静默错误。
    """
    factor = _strict_factor(factor)
    if component not in COMPONENT_SPECS:
        raise ValueError(f"未知组件: {component!r}")
    if not isinstance(baseline, dict) or not baseline:
        raise ValueError("剂量干预需要非空的门控参数基线快照")
    names = []
    for name, _ in model.named_parameters():
        if _component_param_name(name, component):
            names.append(name)
    if not names:
        return {"applied": False,
                "reason": f"模型未注册组件 {component} 的参数",
                "changed_tensors": 0, "max_abs_delta": 0.0, "no_op": False}
    changed = 0
    max_delta = 0.0
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name not in names:
                continue
            if name not in baseline:
                raise ValueError(f"基线快照缺少参数 {name}")
            base = baseline[name]
            if not torch.is_tensor(base):
                raise ValueError(f"基线快照 {name} 不是张量")
            if tuple(base.shape) != tuple(p.shape):
                raise ValueError(
                    f"基线快照 {name} 形状不一致："
                    f"{tuple(base.shape)} vs {tuple(p.shape)}")
            old = p.detach().clone()
            p.copy_(base.to(device=p.device, dtype=p.dtype) * factor)
            delta = (p.detach() - old).abs()
            max_delta = max(max_delta,
                            float(delta.max().item()) if delta.numel() else 0.0)
            changed += 1
    return {"applied": True, "reason": "", "factor": float(factor),
            "changed_tensors": int(changed),
            "max_abs_delta": float(max_delta),
            "no_op": bool(max_delta <= 1e-12)}


def _finite_point_pairs(points):
    out = []
    for p in points if isinstance(points, list) else []:
        if not isinstance(p, dict):
            continue
        f = _finite_or_none(p.get("factor"))
        loss = _finite_or_none(p.get("val_loss"))
        if f is None or loss is None:
            continue
        out.append((float(f), float(loss), p))
    out.sort(key=lambda x: x[0])
    return out


def _local_slope(pairs, factor: float = 1.0, base_loss=None):
    """λ=1 附近的局部斜率（NLL 对 λ 的导数，越小越好）。

    优先用紧邻 λ=1 的左右两点做中心差分；只有单侧时退回单侧差分并显式
    标注。缺 λ=1 点但调用方提供 baseline_loss 时，用外部基线构造单侧
    差分，避免把“没有基线点”误判成“无法计算斜率”。返回
    (slope, side, lower, upper, window_slope_se, window_ci)。
    """
    lower = [x for x in pairs if x[0] < factor]
    upper = [x for x in pairs if x[0] > factor]
    base = next((x for x in pairs if abs(x[0] - factor) <= 1e-12), None)
    if base is None:
        bl = _finite_or_none(base_loss)
        if bl is not None:
            base = (float(factor), float(bl), {})
    if not lower and not upper:
        return None, "none", None, None, None, None
    if lower and upper:
        lo = max(lower, key=lambda x: x[0])
        hi = min(upper, key=lambda x: x[0])
        denom = hi[0] - lo[0]
        if denom <= 0:
            return None, "invalid", None, None, None, None
        slope = (hi[1] - lo[1]) / denom
        side = "two_sided"
        # 逐窗口斜率只作描述：相邻窗口可能相关，不能当独立重复。
        w_lo = lo[2].get("window_losses")
        w_hi = hi[2].get("window_losses")
        se = ci = None
        if (isinstance(w_lo, list) and isinstance(w_hi, list)
                and len(w_lo) == len(w_hi) and len(w_lo) >= 2):
            diffs = []
            for a, b in zip(w_hi, w_lo):
                fa, fb = _finite_or_none(a), _finite_or_none(b)
                if fa is None or fb is None:
                    continue
                diffs.append((fa - fb) / denom)
            if len(diffs) >= 2:
                mean, std = mean_std(diffs)
                if mean is not None and std is not None:
                    se = float(std / math.sqrt(len(diffs)))
                    if not math.isfinite(se):
                        se = None
                ci = bootstrap_mean_ci(diffs)
        return slope, side, lo[0], hi[0], se, ci
    if upper:
        hi = min(upper, key=lambda x: x[0])
        if base is None:
            return None, "invalid", None, None, None, None
        denom = hi[0] - factor
        if denom <= 0:
            return None, "invalid", None, None, None, None
        return ((hi[1] - base[1]) / denom, "one_sided_upper", factor,
                hi[0], None, None)
    lo = max(lower, key=lambda x: x[0])
    if base is None:
        return None, "invalid", None, None, None, None
    denom = factor - lo[0]
    if denom <= 0:
        return None, "invalid", None, None, None, None
    return ((base[1] - lo[1]) / denom, "one_sided_lower", lo[0], factor,
            None, None)


def _quadratic_fit(pairs):
    """最小二乘拟合 NLL ≈ a·λ² + b·λ + c；返回曲率/顶点/R²。"""
    if len(pairs) < 3:
        return {"curvature": None, "vertex": None, "vertex_in_grid": False,
                "r2": None, "n_points": len(pairs)}
    x = np.asarray([p[0] for p in pairs], dtype=float)
    y = np.asarray([p[1] for p in pairs], dtype=float)
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        return {"curvature": None, "vertex": None, "vertex_in_grid": False,
                "r2": None, "n_points": len(pairs)}
    x_range = float(x.max() - x.min())
    if x_range <= 0:
        return {"curvature": None, "vertex": None, "vertex_in_grid": False,
                "r2": None, "n_points": len(pairs)}
    design = np.vstack([x * x, x, np.ones_like(x)]).T
    try:
        coef, _residuals, _rank, _sv = np.linalg.lstsq(design, y, rcond=None)
    except np.linalg.LinAlgError:
        return {"curvature": None, "vertex": None, "vertex_in_grid": False,
                "r2": None, "n_points": len(pairs)}
    a, b, c = (float(v) for v in coef)
    if not (math.isfinite(a) and math.isfinite(b) and math.isfinite(c)):
        return {"curvature": None, "vertex": None, "vertex_in_grid": False,
                "r2": None, "n_points": len(pairs)}
    pred = design @ coef
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = (1.0 - ss_res / ss_tot) if ss_tot > 0 else None
    vertex = None
    # 曲率必须显著大于 0 才有“内部极小值”；线性拟合的 a≈0 时顶点无意义。
    if a > 1e-12:
        vertex = -b / (2.0 * a)
        if not math.isfinite(vertex):
            vertex = None
    return {
        "curvature": 2.0 * a,
        "vertex": vertex,
        "vertex_in_grid": bool(
            vertex is not None and x.min() - 1e-12 <= vertex <= x.max() + 1e-12),
        "r2": r2,
        "n_points": len(pairs),
    }


def dose_curve_diagnostics(points, baseline_loss=None) -> Dict:
    """由逐剂量点计算曲线诊断（全部为确定性描述统计）。

    points 每项至少含 ``factor``、``val_loss``；可选 ``window_losses`` 用于
    局部斜率的窗口级（描述性）区间。缺 λ=1 点且未给 baseline_loss 时，
    以离 1 最近的损失作为基线并在结果中标注。
    """
    pairs = _finite_point_pairs(points)
    if not pairs:
        return {"available": False, "reason": "没有有效的剂量点"}
    base_factor = 1.0
    base_point = next((x for x in pairs if abs(x[0] - base_factor) <= 1e-12),
                      None)
    base_loss = _finite_or_none(baseline_loss)
    if base_loss is None:
        if base_point is not None:
            base_loss = base_point[1]
        else:
            closest = min(pairs, key=lambda x: abs(x[0] - base_factor))
            base_factor = closest[0]
            base_loss = closest[1]
    base_loss = float(base_loss)
    factors = [p[0] for p in pairs]
    losses = [p[1] for p in pairs]
    deltas = [p[1] - base_loss for p in pairs]
    slope, side, lower, upper, slope_se, slope_ci = _local_slope(
        pairs, base_factor, base_loss)
    quad = _quadratic_fit(pairs)
    best = min(pairs, key=lambda x: (x[1], abs(x[0] - base_factor), x[0]))
    improvements = [base_loss - v for v in losses]
    loss_diffs = [losses[i + 1] - losses[i] for i in range(len(losses) - 1)]
    nonzero = [d for d in loss_diffs if abs(d) > 1e-12]
    direction_changes = sum(
        1 for a, b in zip(nonzero, nonzero[1:]) if (a > 0) != (b > 0))
    zero_delta = next((d for f, d in zip(factors, deltas)
                       if abs(f) <= 1e-12), None)
    flip_delta = next((d for f, d in zip(factors, deltas)
                       if abs(f + 1.0) <= 1e-12), None)
    base_ppl = None
    best_ppl = None
    base_info = next((p[2] for p in pairs
                      if abs(p[0] - base_factor) <= 1e-12), None)
    best_info = best[2]
    if isinstance(base_info, dict):
        base_ppl = _finite_or_none(base_info.get("val_ppl"))
    if isinstance(best_info, dict):
        best_ppl = _finite_or_none(best_info.get("val_ppl"))
    best_ppl_ratio = (best_ppl / base_ppl
                      if base_ppl and best_ppl and base_ppl > 0
                      and best_ppl > 0 else None)
    return {
        "available": True,
        "baseline_factor": float(base_factor),
        "baseline_loss": float(base_loss),
        "factors": factors,
        "losses": losses,
        "deltas": deltas,
        "local_slope": slope,
        "local_slope_side": side,
        "local_slope_bracket": [lower, upper],
        "local_slope_window_se": slope_se,
        "local_slope_window_ci95": slope_ci,
        "curvature": quad.get("curvature"),
        "quadratic_vertex": quad.get("vertex"),
        "quadratic_vertex_in_grid": bool(quad.get("vertex_in_grid")),
        "quadratic_r2": quad.get("r2"),
        "quadratic_n_points": quad.get("n_points"),
        "best_factor": float(best[0]),
        "best_loss": float(best[1]),
        "baseline_is_grid_minimum": bool(abs(best[0] - base_factor) <= 1e-12),
        "improvement_vs_baseline": float(base_loss - best[1]),
        "improvement_vs_baseline_pct": (
            100.0 * (base_loss - best[1]) / base_loss
            if base_loss != 0 else None),
        "best_ppl_ratio_vs_baseline": best_ppl_ratio,
        "n_direction_changes": int(direction_changes),
        "monotonic": direction_changes == 0,
        "zero_delta": zero_delta,
        "sign_flip_delta": flip_delta,
        "note": ("局部斜率=NLL 对 λ 的导数（<0 表示增大系数会降低 NLL，"
                 "当前系数可能偏小；>0 表示可能偏大）；二次顶点仅在曲率>0 "
                 "时有定义；网格最优 λ 与改善量是描述性统计，固定网格上"
                 "挑最优会引入 winner's curse，不据此重选 checkpoint 或"
                 "改写预注册判定。窗口级斜率区间只描述单 run，相邻窗口"
                 "可能相关，跨种子检验由 run_all 汇总。"),
    }


def _metric_subset(metrics: Dict) -> Dict:
    """只保留剂量曲线消费的指标，避免每个点复制整份评测负载。"""
    keys = ("val_ppl", "val_loss", "val_ppl_clamped", "gate_mean",
            "gate_std", "gate_frac_low", "gate_frac_high",
            "gate_layer_mean", "val_loss_se", "val_loss_ci95",
            "val_ppl_ci95", "val_loss_min", "val_loss_max")
    return {k: metrics.get(k) for k in keys}


def run_dose_response(ckpt: str, data_dir: str, *,
                      eval_iters: Optional[int] = None,
                      ctx: Optional[int] = None,
                      device: str = "auto",
                      seed: Optional[int] = None,
                      factors=None,
                      components=None,
                      allow_data_protocol_violations: bool = False) -> Dict:
    """在 val split 上对一个 checkpoint 执行门控剂量-反应扫描。"""
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"checkpoint 不存在: {ckpt}")
    factors = parse_factors(DEFAULT_FACTORS if factors is None else factors)
    components = parse_components(COMPONENT_ORDER if components is None
                                  else components)
    preflight = data_protocol_preflight(
        data_dir, allow_override=allow_protocol_violations(
            allow_data_protocol_violations))
    if not preflight.get("report_present"):
        print("[dose] 警告：数据目录缺少 data_report.json，"
              "无法做评测前防泄漏闸门（旧数据或未走统一入口）")
    elif preflight.get("critical_violations"):
        print("[dose] ⚠ 数据协议自检发现严重问题但已显式放行；"
              "本分析不得用于正式结论")

    device = get_device(device)
    enable_tf32(device)
    model, ck = load_model(ckpt, device=device)
    caps = gate_capabilities(model)
    if not caps["gated"]:
        raise ValueError(
            f"变体 {model.variant!r} 没有输出门，无法做门控剂量-反应；"
            "该分析只适用于 gate_only/qgate/lega 及其消融")

    ckpt_seed = ck.get("seed")
    if seed is None:
        if ckpt_seed is not None:
            try:
                seed = seed_int(ckpt_seed, "checkpoint seed")
            except (TypeError, ValueError):
                seed = None
    else:
        seed = seed_int(seed, "seed")
    set_seed(0 if seed is None else seed)

    ckpt_dir = os.path.dirname(os.path.abspath(ckpt))
    cfg_used = {}
    cfg_path = os.path.join(ckpt_dir, "config_used.json")
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            cfg_used = loaded if isinstance(loaded, dict) else {}
        except (OSError, ValueError):
            cfg_used = {}
    if ctx is None:
        train_ctx = _positive_int_or_none(cfg_used.get("train_ctx"))
        ctx = min(model.block_size, train_ctx) if train_ctx else model.block_size
    ctx = strict_int(ctx, "ctx")
    if ctx <= 0 or ctx > model.block_size:
        raise ValueError(f"ctx 必须在 [1, {model.block_size}]，收到 {ctx}")
    if eval_iters is None:
        eval_iters = _positive_int_or_none(cfg_used.get("eval_iters")) or 50
    eval_iters = strict_int(eval_iters, "eval_iters")
    if eval_iters <= 0:
        raise ValueError(f"eval_iters 必须 >= 1，收到 {eval_iters}")
    amp = _resolve_eval_amp(cfg_used, device)

    val = load_bin(data_dir, "val", allow_test=False)
    base = _evaluate_once(model, val, ctx, device, eval_iters, amp)
    if base.get("window_losses") is None or base.get("window_indices") is None:
        raise RuntimeError("基线评测未返回逐窗口损失，无法做剂量配对分析")
    if _finite_or_none(base.get("val_loss")) is None:
        raise RuntimeError("基线 val_loss 非有限，无法作为剂量曲线的 δ=0 参照")
    base_windows = list(base["window_losses"])
    base_indices = list(base["window_indices"])

    snapshot = snapshot_gate_params(model)
    out_components = {}
    try:
        for component in components:
            spec = COMPONENT_SPECS[component]
            applicable, reason, _caps = component_applicability(
                model, component)
            rec = {
                "component": component,
                "label": spec["label"],
                "parameter": spec["param"],
                "training_variant": spec["training_variant"],
                "applicable": bool(applicable),
                "reason": reason,
                "points": [],
            }
            if not applicable:
                out_components[component] = rec
                continue
            for factor in factors:
                restore_gate_params(model, snapshot)
                point = {"factor": float(factor)}
                if abs(factor - 1.0) <= 1e-12:
                    point.update({
                        "applied": True,
                        "no_op": True,
                        "reason": "λ=1 为基线，直接复用基线评测",
                    })
                    point.update(_metric_subset(base))
                    point.update({
                        "window_losses": list(base_windows),
                        "window_indices": list(base_indices),
                        "delta_loss_mean": 0.0,
                        "ppl_ratio": 1.0,
                        "ppl_ratio_clamped": False,
                        "n_windows": len(base_windows),
                    })
                    rec["points"].append(point)
                    continue
                try:
                    applied = apply_dose(model, component, factor, snapshot)
                    if not applied.get("applied"):
                        point.update({
                            "applied": False,
                            "no_op": False,
                            "reason": applied.get("reason", "未施加"),
                        })
                        rec["points"].append(point)
                        continue
                    point.update(applied)
                    evaluated = _evaluate_once(
                        model, val, ctx, device, eval_iters, amp)
                finally:
                    restore_gate_params(model, snapshot)
                if evaluated.get("window_indices") != base_indices:
                    point.update({
                        "applied": False,
                        "reason": "剂量点评测窗口与基线不一致，拒绝配对",
                    })
                    rec["points"].append(point)
                    continue
                point.update(_metric_subset(evaluated))
                point.update({
                    "window_losses": list(evaluated.get("window_losses") or []),
                    "window_indices": list(
                        evaluated.get("window_indices") or []),
                })
                point.update(_window_delta_stats(
                    base_windows, evaluated.get("window_losses") or []))
                delta = point.get("delta_loss_mean")
                point["ppl_ratio"] = (
                    math.exp(min(20.0, max(-20.0, float(delta))))
                    if delta is not None else None)
                point["ppl_ratio_clamped"] = bool(
                    delta is not None and not (-20.0 <= float(delta) <= 20.0))
                rec["points"].append(point)
            rec["diagnostics"] = dose_curve_diagnostics(
                rec["points"], baseline_loss=base.get("val_loss"))
            out_components[component] = rec
    finally:
        restore_gate_params(model, snapshot)

    return {
        "schema": 1,
        "pipeline_run_id": current_pipeline_run_id(),
        "variant": model.variant,
        "ckpt": os.path.abspath(ckpt),
        "ckpt_sha256": sha256_file(ckpt),
        "seed": (None if seed is None else int(seed)),
        "split": "val",
        "eval_ctx": int(ctx),
        "eval_iters": int(eval_iters),
        "val_windows": base.get("val_windows"),
        "val_tokens_evaluated": base.get("val_tokens_evaluated"),
        "window_indices": base_indices,
        "amp": bool(amp),
        "factors": factors,
        "components": out_components,
        "gate_capabilities": caps,
        "baseline": _metric_subset(base),
        "baseline_window_losses": base_windows,
        "protocol": {
            "split": "val",
            "same_windows": True,
            "note": DOSE_NOTE,
        },
        "provenance": load_provenance(ckpt_dir, data_dir, device=device),
        "generated_at": now_str(),
        "argv": list(sys.argv),
    }


def main():
    p = argparse.ArgumentParser(
        description="门控系数的推理期剂量-反应扫描（同一 checkpoint 的连续消融）")
    p.add_argument("--ckpt", required=True,
                   help="model_best.pt / model.pt（必须是带输出门的变体）")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out", required=True, help="输出 JSON 路径")
    p.add_argument("--eval_iters", default=None,
                   help="评测窗口数；默认继承 checkpoint 的 config_used.json"
                        "（缺失时 50，与 lega.evaluate 一致）")
    p.add_argument("--ctx", default=None,
                   help="评测上下文；默认继承训练配置的 train_ctx")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", default=None,
                   help="覆盖 checkpoint 记录的种子（只影响确定性播种，"
                        "不改变权重）")
    p.add_argument("--factors", default=",".join(
        f"{x:g}" for x in DEFAULT_FACTORS),
        help="逗号分隔的缩放因子网格（默认 0,0.5,0.75,1,1.25,1.5,2）")
    p.add_argument("--components", default="A,B,C",
                   help="要扫描的组件（默认 A,B,C）")
    p.add_argument("--allow_data_protocol_violations", action="store_true",
                   help="即使 data_report 有严重泄露告警也继续（仅复现历史；"
                        "结果不得用于正式结论）")
    args = p.parse_args()
    seed = None
    if args.seed is not None:
        try:
            seed = strict_int(args.seed, "--seed")
        except (TypeError, ValueError) as e:
            p.error(str(e))
    try:
        factors = parse_factors(args.factors)
        components = parse_components(args.components)
    except ValueError as e:
        p.error(str(e))
    try:
        payload = run_dose_response(
            args.ckpt, args.data_dir, eval_iters=args.eval_iters, ctx=args.ctx,
            device=args.device, seed=seed, factors=factors,
            components=components,
            allow_data_protocol_violations=args.allow_data_protocol_violations)
    except (ValueError, FileNotFoundError) as e:
        p.error(str(e))
    save_json(payload, args.out)
    base = payload["baseline"]
    print(f"[dose] {payload['variant']} seed={payload['seed']} "
          f"val_loss={base.get('val_loss')} val_ppl={base.get('val_ppl')}")
    for name, rec in payload["components"].items():
        if not rec.get("applicable"):
            print(f"  - {name}: 不适用（{rec.get('reason')}）")
            continue
        diag = rec.get("diagnostics") or {}
        print(f"  - {name}: local_slope={diag.get('local_slope')} "
              f"vertex={diag.get('quadratic_vertex')} "
              f"best_factor={diag.get('best_factor')}")
    print(f"[dose] -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
