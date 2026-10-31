"""自然文本上的 PPL 长度剂量-反应审计。

``lega.lengthcalib`` 只审计注意力统计量 ``c = log(n * max_j p_j)`` 随长度的
斜率；它不回答“同一批 query token 的下一 token 损失是否随上下文长度变化”。
本模块在真实 **val** 文本上构造嵌套配对窗口，在固定 query token 上直接测量
下一 token NLL/PPL 随长度的变化::

    suffix: 固定窗口末尾与最后 q 个目标 token，只增加前缀（主分析）
    prefix: 固定窗口起点，目标 token 随长度改变（敏感性分析）

``suffix`` 模式下每个窗口在各长度上的目标 token 完全相同，因此
``NLL(L) - NLL(L_ref)`` 是逐窗口配对差；对 ``log2(L)`` 的斜率描述“长度每翻
一倍，同一批 query 的 NLL 变化多少 nat”。这是描述性/探索性审计：它不消融
门控参数，也不替代 RULER 长度分档或推理期因果干预。checkpoint 在 val 上
选择，因此该曲线只描述长度形状，不是独立 test 集 PPL 证据。

用法::

    python -m lega.lengthppl --ckpt out/small/lega/model_best.pt \
        --data_dir data/wikitext103 --out results/small/length_ppl_lega.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Iterable, List, Optional

import numpy as np
import torch

from .data import load_bin
from .lengthcalib import (DEFAULT_GROWTH_MODES, DEFAULT_LENGTHS,
                          calibration_window_starts, parse_growth_modes,
                          parse_lengths)
from .model import load_model
from .repro import current_pipeline_run_id, load_provenance, sha256_file
from .utils import (amp_autocast, config_bool, enable_tf32, get_device,
                    save_json, set_seed, strict_int)

__all__ = [
    "DEFAULT_LENGTHS",
    "DEFAULT_GROWTH_MODES",
    "analyze_length_ppl_records",
    "collect_length_ppl_records",
    "compare_length_ppl_records",
    "compare_length_ppl_payloads",
    "run_length_ppl",
    "main",
]

SCHEMA = 1
DEFAULT_TOLERANCE = 0.01
_IGNORE_INDEX = -100


def _finite_float(value):
    if isinstance(value, (bool, np.bool_)):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return out if math.isfinite(out) else None


def _integer(value, name: str, minimum: int = None):
    try:
        out = strict_int(value, name)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(str(exc)) from exc
    if minimum is not None and out < minimum:
        raise ValueError(f"{name} 必须 >= {minimum}，收到 {out}")
    return out


def _ppl_from_nll(nll):
    nll = _finite_float(nll)
    if nll is None or nll > 700.0:
        return None
    return float(math.exp(nll))


def _input_ids(val, start: int, length: int, growth: str,
               window_length: int) -> np.ndarray:
    if growth == "suffix":
        end = int(start) + int(window_length)
        return np.asarray(val[end - int(length):end], dtype=np.int64)
    return np.asarray(val[int(start):int(start) + int(length)], dtype=np.int64)


def _input_start(start: int, length: int, growth: str,
                 window_length: int) -> int:
    if growth == "suffix":
        return int(start) + int(window_length) - int(length)
    return int(start)


def _query_targets(val, start: int, length: int, growth: str,
                   window_length: int, query_positions: int) -> np.ndarray:
    if growth == "suffix":
        end = int(start) + int(window_length)
        return np.asarray(val[end - int(query_positions) + 1:end + 1],
                          dtype=np.int64)
    end = int(start) + int(length)
    return np.asarray(val[end - int(query_positions) + 1:end + 1],
                      dtype=np.int64)


def collect_length_ppl_records(model, val, window_starts: List[int],
                               lengths: List[int], query_positions: int,
                               growth: str, device: str,
                               amp: bool = True,
                               batch_windows: int = 8) -> List[dict]:
    """逐 (window, length) 收集固定 query 位置的下一 token NLL。

    输入窗口长度为 ``length``；目标张量只在最后 ``query_positions`` 个位置
    有效，其余位置为 ``-100``。因此模型返回的 ``loss_per_window`` 是每个
    窗口“最后 q 个目标 token 的平均 NLL”，而不是整窗平均——这样 suffix
    模式在不同长度间比较的是完全相同的目标 token。
    """
    growth = parse_growth_modes(growth)[0]
    lengths = parse_lengths(lengths)
    query_positions = _integer(query_positions, "query_positions", 1)
    if query_positions > min(lengths):
        raise ValueError(
            f"query_positions={query_positions} 超过最短长度 {min(lengths)}")
    batch_windows = _integer(batch_windows, "batch_windows", 1)
    max_length = max(lengths)
    starts = [_integer(s, "window_start", 0) for s in window_starts]
    if not starts:
        raise ValueError("window_starts 不能为空")
    for start in starts:
        if start + max_length >= len(val):
            raise ValueError(
                f"窗口起点 {start} + 最长长度 {max_length} 需要至少一个"
                f"目标 token，但 val 长度只有 {len(val)}")
    records: List[dict] = []
    amp_ctx = amp_autocast(device, bool(amp))
    with torch.inference_mode():
        for length in lengths:
            for b0 in range(0, len(starts), batch_windows):
                batch_starts = starts[b0:b0 + batch_windows]
                ids_np = np.stack([
                    _input_ids(val, start, length, growth, max_length)
                    for start in batch_starts])
                targets_np = np.full_like(ids_np, _IGNORE_INDEX, dtype=np.int64)
                target_ids = [
                    _query_targets(val, start, length, growth, max_length,
                                   query_positions)
                    for start in batch_starts]
                for b in range(targets_np.shape[0]):
                    targets_np[b, -query_positions:] = target_ids[b]
                ids = torch.from_numpy(ids_np)
                targets = torch.from_numpy(targets_np)
                if device.startswith("cuda"):
                    ids = ids.to(device)
                    targets = targets.to(device)
                with amp_ctx:
                    out = model(ids, targets=targets,
                                return_window_losses=True)
                losses = out.get("loss_per_window")
                if losses is None:
                    raise RuntimeError(
                        "模型没有返回 loss_per_window；无法做自然文本 PPL "
                        "长度审计")
                losses = losses.float().cpu().numpy()
                for b, start in enumerate(batch_starts):
                    nll = _finite_float(losses[b])
                    if nll is None:
                        raise RuntimeError(
                            f"长度 PPL 出现非有限损失: window_start={start}, "
                            f"length={length}, growth={growth}")
                    records.append({
                        "window": b0 + b,
                        "window_start": int(start),
                        "input_start": _input_start(
                            start, length, growth, max_length),
                        "length": int(length),
                        "query_positions": int(query_positions),
                        "target_ids": [int(x) for x in target_ids[b]],
                        "nll": float(nll),
                        "ppl": _ppl_from_nll(nll),
                    })
    return records


def _records_by_window(records: List[dict], lengths: List[int]):
    grouped = {}
    for rec in records:
        if not isinstance(rec, dict):
            continue
        nll = _finite_float(rec.get("nll"))
        length = _finite_float(rec.get("length"))
        if nll is None or length is None or int(length) not in lengths:
            continue
        try:
            window = _integer(rec.get("window"), "window", 0)
        except ValueError:
            continue
        key = (window, int(length))
        if key in grouped:
            grouped[key] = None
        else:
            grouped[key] = rec
    windows = sorted({w for (w, _) in grouped})
    out = []
    for window in windows:
        rows = [grouped.get((window, length)) for length in lengths]
        if any(row is None for row in rows):
            continue
        if any(_finite_float(row.get("nll")) is None for row in rows):
            continue
        out.append({
            "window": window,
            "window_start": _integer(rows[0].get("window_start"),
                                     "window_start", 0),
            "lengths": np.asarray(lengths, dtype=np.float64),
            "nll": np.asarray([float(row["nll"]) for row in rows],
                              dtype=np.float64),
            "target_ids": [list(row.get("target_ids") or []) for row in rows],
        })
    return out


def _slope_from_groups(groups: List[dict], value_key: str = "nll"):
    xs = []
    ys = []
    for group in groups:
        lengths = np.asarray(group["lengths"], dtype=np.float64)
        values = np.asarray(group[value_key], dtype=np.float64)
        if lengths.size < 2 or values.size != lengths.size:
            continue
        if not np.all(np.isfinite(values)):
            continue
        x = np.log2(lengths)
        xm = float(x.mean())
        ym = float(values.mean())
        xs.append(x - xm)
        ys.append(values - ym)
    if not xs:
        return None
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    sxx = float(np.dot(x, x))
    if not math.isfinite(sxx) or sxx <= 0.0:
        return None
    slope = float(np.dot(x, y) / sxx)
    return slope if math.isfinite(slope) else None


def _bootstrap_groups(groups: List[dict], fn, n_bootstrap: int, seed: int):
    if not groups:
        return None
    observed = fn(groups)
    if observed is None:
        return None
    if n_bootstrap <= 0 or len(groups) < 2:
        return {
            "estimate": float(observed),
            "ci95": None,
            "p_boot": None,
            "n_valid": 0,
            "n_requested": int(n_bootstrap),
        }
    rng = np.random.default_rng(seed)
    boots = []
    n = len(groups)
    for _ in range(int(n_bootstrap)):
        picks = rng.integers(0, n, size=n)
        sample = [groups[int(i)] for i in picks]
        value = fn(sample)
        if value is not None and math.isfinite(float(value)):
            boots.append(float(value))
    if not boots:
        return {
            "estimate": float(observed),
            "ci95": None,
            "p_boot": None,
            "n_valid": 0,
            "n_requested": int(n_bootstrap),
        }
    arr = np.asarray(boots, dtype=np.float64)
    lo, hi = np.quantile(arr, [0.025, 0.975])
    centered = arr - float(observed)
    if np.allclose(arr, arr[0], rtol=0.0, atol=1e-12):
        p = None
    else:
        p = float((np.sum(np.abs(centered) >= abs(float(observed))) + 1.0)
                  / (arr.size + 1.0))
    return {
        "estimate": float(observed),
        "ci95": [float(lo), float(hi)],
        "p_boot": p,
        "n_valid": int(arr.size),
        "n_requested": int(n_bootstrap),
    }


def _mean(values):
    arr = np.asarray([v for v in values if _finite_float(v) is not None],
                     dtype=np.float64)
    if arr.size == 0:
        return None
    return float(arr.mean())


def _slope_class(slope, ci, tolerance: float):
    if slope is None:
        return "undefined"
    if ci is None:
        return "no_ci"
    lo, hi = ci
    if lo > tolerance:
        return "increasing"
    if hi < -tolerance:
        return "decreasing"
    if lo >= -tolerance and hi <= tolerance:
        return "flat"
    return "unresolved"


def _target_pairing(groups: List[dict], growth: str):
    if not groups:
        return {"available": False, "n_windows": 0}
    if growth == "suffix":
        checks = []
        for group in groups:
            if not group.get("target_ids"):
                continue
            first = group["target_ids"][0]
            if not first:
                continue
            checks.append(all(ids == first for ids in group["target_ids"]))
        if not checks:
            return {
                "available": False,
                "mode": "same_targets_across_lengths",
                "n_windows": len(groups),
                "note": "产物缺少目标 token 记录，无法核验配对目标",
            }
        return {
            "available": True,
            "mode": "same_targets_across_lengths",
            "n_windows": len(groups),
            "n_checked": len(checks),
            "n_consistent": int(sum(checks)),
            "fraction_consistent": float(sum(checks) / len(checks)),
        }
    return {
        "available": True,
        "mode": "targets_change_with_length",
        "n_windows": len(groups),
        "note": "prefix 模式的 query token 随长度改变，不是配对目标；只作敏感性分析",
    }


def analyze_length_ppl_records(records: List[dict], lengths: Iterable[int],
                               growth: str = "suffix", *,
                               reference_length: Optional[int] = None,
                               tolerance: float = DEFAULT_TOLERANCE,
                               n_bootstrap: int = 400,
                               seed: int = 0) -> dict:
    """分析单个 checkpoint 的自然文本 PPL 长度曲线。"""
    lengths = parse_lengths(lengths)
    growth = parse_growth_modes(growth)[0]
    tolerance = _finite_float(tolerance)
    if tolerance is None or tolerance < 0.0:
        raise ValueError(f"tolerance 必须是非负有限数值，收到 {tolerance!r}")
    n_bootstrap = _integer(n_bootstrap, "n_bootstrap", 0)
    seed = _integer(seed, "seed", 0)
    if reference_length is None:
        reference_length = max(lengths)
    reference_length = _integer(reference_length, "reference_length", 2)
    if reference_length not in lengths:
        raise ValueError(
            f"reference_length={reference_length} 不在 lengths={lengths} 中")
    records = [r for r in records if isinstance(r, dict)]
    groups = _records_by_window(records, lengths)
    result = {
        "available": bool(groups),
        "growth": growth,
        "lengths": lengths,
        "reference_length": int(reference_length),
        "n_records": len(records),
        "n_windows": len(groups),
        "tolerance": float(tolerance),
        "n_bootstrap": int(n_bootstrap),
        "seed": int(seed),
        "target_pairing": _target_pairing(groups, growth),
        "per_length": {},
        "delta_vs_reference": {},
        "nll_slope_per_log2": {
            "estimate": None, "ci95": None, "p_boot": None,
            "class": "undefined",
        },
        "context_gain_nats": {
            "estimate": None, "ci95": None, "p_boot": None,
        },
        "per_window": [],
    }
    if not groups:
        return result
    ref_idx = lengths.index(reference_length)
    for idx, length in enumerate(lengths):
        values = np.asarray([g["nll"][idx] for g in groups],
                            dtype=np.float64)
        mean_nll = float(values.mean())
        sd = float(values.std(ddof=1)) if values.size > 1 else 0.0
        se = (sd / math.sqrt(values.size)) if values.size > 1 else None
        result["per_length"][str(length)] = {
            "n_windows": int(values.size),
            "mean_nll": mean_nll,
            "sd_nll": sd,
            "se_nll": se,
            "median_nll": float(np.median(values)),
            "ppl": _ppl_from_nll(mean_nll),
        }
    for idx, length in enumerate(lengths):
        if length == reference_length:
            continue
        deltas = [g["nll"][idx] - g["nll"][ref_idx] for g in groups]
        boot = _bootstrap_groups(
            [{"delta": float(d)} for d in deltas],
            lambda gs: _mean([g["delta"] for g in gs]),
            n_bootstrap, seed + 1009 * int(length))
        mean_delta = None if boot is None else boot["estimate"]
        result["delta_vs_reference"][str(length)] = {
            "n_windows": len(deltas),
            "mean_delta_nll": mean_delta,
            "median_delta_nll": float(np.median(deltas)),
            "ppl_ratio": (None if mean_delta is None
                          else _ppl_from_nll(mean_delta)),
            "win_rate_shorter_better": float(
                np.mean(np.asarray(deltas) < 0.0)),
            "ci95": None if boot is None else boot["ci95"],
            "p_boot": None if boot is None else boot["p_boot"],
            "direction": (
                "shorter_context_lower_nll" if mean_delta is not None
                and mean_delta < 0 else
                "longer_context_lower_nll" if mean_delta is not None
                and mean_delta > 0 else "no_difference"),
        }
    slope_boot = _bootstrap_groups(
        groups, lambda gs: _slope_from_groups(gs, "nll"),
        n_bootstrap, seed + 2003)
    result["nll_slope_per_log2"] = {
        "estimate": None if slope_boot is None else slope_boot["estimate"],
        "ci95": None if slope_boot is None else slope_boot["ci95"],
        "p_boot": None if slope_boot is None else slope_boot["p_boot"],
        "class": _slope_class(
            None if slope_boot is None else slope_boot["estimate"],
            None if slope_boot is None else slope_boot["ci95"],
            tolerance),
        "n_windows": len(groups),
        "n_obs": int(sum(g["lengths"].size for g in groups)),
    }
    gains = [g["nll"][0] - g["nll"][-1] for g in groups]
    gain_boot = _bootstrap_groups(
        [{"gain": float(x)} for x in gains],
        lambda gs: _mean([g["gain"] for g in gs]),
        n_bootstrap, seed + 3001)
    result["context_gain_nats"] = {
        "estimate": None if gain_boot is None else gain_boot["estimate"],
        "ci95": None if gain_boot is None else gain_boot["ci95"],
        "p_boot": None if gain_boot is None else gain_boot["p_boot"],
        "note": ("shortest_minus_longest NLL；正值表示更长上下文降低损失，"
                 "负值表示更长上下文反而更差"),
    }
    for group in groups:
        target_consistent = None
        if growth == "suffix":
            first = group["target_ids"][0]
            target_consistent = all(ids == first
                                    for ids in group["target_ids"])
        result["per_window"].append({
            "window": int(group["window"]),
            "window_start": int(group["window_start"]),
            "nll_by_length": {
                str(int(length)): float(value)
                for length, value in zip(lengths, group["nll"])},
            "slope_per_log2": _slope_from_groups([group], "nll"),
            "delta_vs_reference": {
                str(int(length)): float(group["nll"][idx] - group["nll"][ref_idx])
                for idx, length in enumerate(lengths)
                if length != reference_length},
            "target_ids_consistent": target_consistent,
            "target_ids_by_length": {
                str(int(length)): [int(x) for x in ids]
                for length, ids in zip(lengths, group["target_ids"])},
        })
    return result


def compare_length_ppl_records(candidate_records: List[dict],
                               reference_records: List[dict],
                               lengths: Iterable[int],
                               growth: str = "suffix", *,
                               candidate: str = None,
                               reference: str = None,
                               tolerance: float = DEFAULT_TOLERANCE,
                               n_bootstrap: int = 400,
                               seed: int = 0) -> dict:
    """配对比较两个 checkpoint 的 PPL 长度曲线（长度交互）。

    对每个共同窗口和长度计算 ``NLL_candidate - NLL_reference``，再对
    ``log2(L)`` 回归。斜率显著非 0 表示两变体的差距随上下文长度变化；
    这是探索性的交互证据，不是因果长度效应。
    """
    lengths = parse_lengths(lengths)
    growth = parse_growth_modes(growth)[0]
    tolerance = _finite_float(tolerance)
    if tolerance is None or tolerance < 0.0:
        raise ValueError(f"tolerance 必须是非负有限数值，收到 {tolerance!r}")
    n_bootstrap = _integer(n_bootstrap, "n_bootstrap", 0)
    seed = _integer(seed, "seed", 0)
    cand_groups = {g["window"]: g for g in
                   _records_by_window(candidate_records, lengths)}
    ref_groups = {g["window"]: g for g in
                  _records_by_window(reference_records, lengths)}
    windows = sorted(set(cand_groups) & set(ref_groups))
    groups = []
    target_checks = []
    for window in windows:
        cg = cand_groups[window]
        rg = ref_groups[window]
        delta = cg["nll"] - rg["nll"]
        if (cg.get("target_ids") and rg.get("target_ids")
                and any(cg["target_ids"]) and any(rg["target_ids"])):
            target_checks.append(cg["target_ids"] == rg["target_ids"])
        groups.append({
            "window": window,
            "window_start": cg["window_start"],
            "lengths": np.asarray(lengths, dtype=np.float64),
            "delta": np.asarray(delta, dtype=np.float64),
        })
    result = {
        "available": bool(groups),
        "growth": growth,
        "lengths": lengths,
        "candidate": candidate,
        "reference": reference,
        "n_windows": len(groups),
        "tolerance": float(tolerance),
        "n_bootstrap": int(n_bootstrap),
        "seed": int(seed),
        "target_pairing": {
            "available": bool(target_checks),
            "n_windows": len(groups),
            "n_checked": len(target_checks),
            "n_consistent": int(sum(target_checks)),
            "fraction_consistent": (
                float(sum(target_checks) / len(target_checks))
                if target_checks else None),
            "note": (None if target_checks else
                     "产物缺少目标 token 记录，无法核验配对目标"),
        },
        "per_length": {},
        "delta_slope_per_log2": {
            "estimate": None, "ci95": None, "p_boot": None,
            "class": "undefined",
        },
    }
    if not groups:
        return result
    for idx, length in enumerate(lengths):
        values = np.asarray([g["delta"][idx] for g in groups],
                            dtype=np.float64)
        boot = _bootstrap_groups(
            [{"delta": float(v)} for v in values],
            lambda gs: _mean([g["delta"] for g in gs]),
            n_bootstrap, seed + 4001 * int(length))
        mean_delta = None if boot is None else boot["estimate"]
        result["per_length"][str(length)] = {
            "n_windows": int(values.size),
            "mean_delta_nll": mean_delta,
            "median_delta_nll": float(np.median(values)),
            "ppl_ratio": (None if mean_delta is None
                          else _ppl_from_nll(mean_delta)),
            "win_rate_candidate_better": float(
                np.mean(values < 0.0)),
            "ci95": None if boot is None else boot["ci95"],
            "p_boot": None if boot is None else boot["p_boot"],
        }
    slope_boot = _bootstrap_groups(
        groups, lambda gs: _slope_from_groups(gs, "delta"),
        n_bootstrap, seed + 5003)
    result["delta_slope_per_log2"] = {
        "estimate": None if slope_boot is None else slope_boot["estimate"],
        "ci95": None if slope_boot is None else slope_boot["ci95"],
        "p_boot": None if slope_boot is None else slope_boot["p_boot"],
        "class": _slope_class(
            None if slope_boot is None else slope_boot["estimate"],
            None if slope_boot is None else slope_boot["ci95"],
            tolerance),
        "n_windows": len(groups),
        "n_obs": int(sum(g["lengths"].size for g in groups)),
        "note": ("ΔNLL 对 log2(L) 的斜率；负值表示候选相对参考的优势"
                 "随长度增加，正值表示优势随长度缩小/劣势扩大"),
    }
    return result


def _payload_groups(payload: dict, growth: str, lengths: List[int]):
    results = payload.get("results") if isinstance(payload, dict) else None
    block = results.get(growth) if isinstance(results, dict) else None
    if not isinstance(block, dict):
        return []
    rows = block.get("per_window")
    if not isinstance(rows, list):
        return []
    groups = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        by_length = row.get("nll_by_length")
        if not isinstance(by_length, dict):
            continue
        try:
            values = [float(by_length[str(length)]) for length in lengths]
        except (KeyError, TypeError, ValueError):
            continue
        if not all(math.isfinite(v) for v in values):
            continue
        by_targets = row.get("target_ids_by_length") or {}
        if not isinstance(by_targets, dict):
            by_targets = {}
        target_ids = [list(by_targets.get(str(length)) or [])
                      for length in lengths]
        try:
            window = _integer(row.get("window"), "window", 0)
            window_start = _integer(row.get("window_start"),
                                    "window_start", 0)
        except ValueError:
            continue
        groups.append({
            "window": window,
            "window_start": window_start,
            "lengths": np.asarray(lengths, dtype=np.float64),
            "nll": np.asarray(values, dtype=np.float64),
            "target_ids": target_ids,
        })
    return groups


def compare_length_ppl_payloads(candidate_payload: dict,
                                reference_payload: dict,
                                growth: str = "suffix", *,
                                tolerance: float = DEFAULT_TOLERANCE,
                                n_bootstrap: int = 400,
                                seed: int = 0) -> dict:
    """从两个 ``run_length_ppl`` payload 的逐窗口曲线做配对交互比较。"""
    if not isinstance(candidate_payload, dict) or not isinstance(
            reference_payload, dict):
        raise ValueError("payload 必须是对象")
    cand_lengths = candidate_payload.get("lengths")
    ref_lengths = reference_payload.get("lengths")
    if not isinstance(cand_lengths, list) or not isinstance(ref_lengths, list):
        raise ValueError("payload 缺少 lengths")
    lengths = parse_lengths(cand_lengths)
    if lengths != parse_lengths(ref_lengths):
        raise ValueError(
            f"两个 payload 的 lengths 不一致: {lengths} vs {ref_lengths}")
    cand = _payload_groups(candidate_payload, growth, lengths)
    ref = _payload_groups(reference_payload, growth, lengths)
    return compare_length_ppl_records(
        [r for g in cand for r in _group_to_records(g)],
        [r for g in ref for r in _group_to_records(g)],
        lengths, growth,
        candidate=candidate_payload.get("variant"),
        reference=reference_payload.get("variant"),
        tolerance=tolerance, n_bootstrap=n_bootstrap, seed=seed)


def _group_to_records(group: dict):
    out = []
    for idx, length in enumerate(group["lengths"]):
        out.append({
            "window": int(group["window"]),
            "window_start": int(group["window_start"]),
            "length": int(length),
            "nll": float(group["nll"][idx]),
            "target_ids": (group["target_ids"][idx]
                           if idx < len(group["target_ids"]) else []),
        })
    return out


def _load_config(path: str = None, ckpt_dir: str = None) -> dict:
    candidates = []
    if path:
        candidates.append(path)
    if ckpt_dir:
        candidates.append(os.path.join(ckpt_dir, "config_used.json"))
    for candidate in candidates:
        if not candidate or not os.path.exists(candidate):
            continue
        try:
            with open(candidate, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except (OSError, ValueError):
            continue
        if isinstance(cfg, dict):
            return cfg
    return {}


def _config_default(cfg: dict, key: str, default):
    block = cfg.get("length_ppl")
    block = block if isinstance(block, dict) else {}
    return block.get(key, default)


def run_length_ppl(ckpt: str, data_dir: str, *, lengths=None,
                   n_windows: int = 8, query_positions: int = 4,
                   growth_modes=None, n_bootstrap: int = 400,
                   tolerance: float = DEFAULT_TOLERANCE, seed=None,
                   device: str = "auto", amp=None,
                   config: str = None) -> dict:
    ckpt_dir = os.path.dirname(os.path.abspath(ckpt))
    cfg = _load_config(config, ckpt_dir)
    lp_cfg = cfg.get("length_ppl")
    lp_cfg = lp_cfg if isinstance(lp_cfg, dict) else {}
    if lengths is None:
        lengths = lp_cfg.get("lengths")
    if lengths is None:
        lengths = (cfg.get("length_calibration") or {}).get("lengths")
    if lengths is None:
        lengths = (cfg.get("length_null") or {}).get("lengths")
    if lengths is None:
        lengths = list(DEFAULT_LENGTHS)
    lengths = parse_lengths(lengths)
    if growth_modes is None:
        growth_modes = lp_cfg.get("growth_modes", DEFAULT_GROWTH_MODES)
    growth_modes = parse_growth_modes(growth_modes)
    n_windows = _integer(n_windows, "n_windows", 1)
    query_positions = _integer(query_positions, "query_positions", 1)
    n_bootstrap = _integer(n_bootstrap, "n_bootstrap", 0)
    tolerance = _finite_float(tolerance)
    if tolerance is None or tolerance < 0.0:
        raise ValueError(f"tolerance 必须是非负有限数值，收到 {tolerance!r}")

    device = get_device(device)
    enable_tf32(device)
    model, ck = load_model(ckpt, device=device)
    max_length = max(lengths)
    if max_length > model.block_size:
        raise ValueError(
            f"最长长度 {max_length} 超过模型 block_size={model.block_size}")
    if query_positions > min(lengths):
        raise ValueError(
            f"query_positions={query_positions} 超过最短长度 {min(lengths)}")
    if seed is None:
        raw_seed = ck.get("seed")
        seed = 0 if raw_seed is None else _integer(raw_seed, "checkpoint seed", 0)
    seed = _integer(seed, "seed", 0)
    set_seed(seed)
    if amp is None:
        amp = bool(config_bool(cfg.get("amp", False), "amp"))
    val = load_bin(data_dir, "val")
    starts = calibration_window_starts(len(val), max_length, n_windows)
    results = {}
    for growth in growth_modes:
        records = collect_length_ppl_records(
            model, val, starts, lengths, query_positions, growth,
            device=device, amp=bool(amp))
        results[growth] = analyze_length_ppl_records(
            records, lengths, growth, tolerance=tolerance,
            n_bootstrap=n_bootstrap, seed=seed)
        results[growth]["n_records"] = len(records)
    return {
        "schema": SCHEMA,
        "pipeline_run_id": current_pipeline_run_id(),
        "variant": ck.get("variant", model.variant),
        "ckpt": os.path.abspath(ckpt),
        "ckpt_sha256": sha256_file(ckpt),
        "seed": seed,
        "split": "val",
        "device": device,
        "amp": bool(amp),
        "model_block_size": int(model.block_size),
        "lengths": lengths,
        "n_windows_requested": n_windows,
        "window_starts": starts,
        "query_positions": query_positions,
        "growth_modes": growth_modes,
        "tolerance": tolerance,
        "n_bootstrap": n_bootstrap,
        "results": results,
        "protocol": {
            "split": "val",
            "targets": (
                "每个输入窗口只在最后 query_positions 个位置计算下一 token "
                "NLL；其余目标位置为 ignore_index"),
            "suffix_growth": (
                "固定窗口末尾与最后 q 个目标 token，只增加前缀；同一窗口"
                "跨长度的目标 token 完全相同，用于配对比较"),
            "prefix_growth": (
                "固定窗口起点，目标 token 随长度改变；只作敏感性分析"),
            "bootstrap": (
                "以窗口为聚类单元；区间/p 值为描述性不确定性，"
                "不是确认性长度效应检验"),
            "note": (
                "观测性审计；前向路径沿用 checkpoint 的 use_sdpa_standard，"
                "不消融门控参数，不替代 RULER 长度分档、长度零假设或"
                "推理期因果干预。只读 val，不访问 test。"),
        },
        "provenance": load_provenance(ckpt_dir, data_dir, device=device),
        "argv": [],
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="自然文本 PPL 长度剂量-反应审计")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--config", default=None,
                   help="可选配置；默认读取 checkpoint 目录的 config_used.json")
    p.add_argument("--lengths", default=None,
                   help="逗号分隔的长度；默认读 length_ppl.lengths，"
                        "再回退到 length_calibration.lengths / length_null.lengths")
    p.add_argument("--n_windows", type=int, default=None)
    p.add_argument("--query_positions", type=int, default=None)
    p.add_argument("--growth_modes", default=None)
    p.add_argument("--bootstrap", type=int, default=None,
                   help="窗口聚类 bootstrap 重采样次数（0 关闭）")
    p.add_argument("--tolerance", type=float, default=None,
                   help="判定 NLL 斜率绝对值的容差（默认 0.01 nat/log2）")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--amp", dest="amp", action="store_true", default=None)
    p.add_argument("--no_amp", dest="amp", action="store_false")
    args = p.parse_args(argv)
    ckpt_dir = os.path.dirname(os.path.abspath(args.ckpt))
    cfg = _load_config(args.config, ckpt_dir)
    lengths = args.lengths
    if lengths is None:
        lengths = _config_default(cfg, "lengths", None)
    if lengths is None:
        lengths = (cfg.get("length_calibration") or {}).get("lengths")
    if lengths is None:
        lengths = (cfg.get("length_null") or {}).get("lengths")
    if lengths is None:
        lengths = list(DEFAULT_LENGTHS)
    try:
        lengths = parse_lengths(lengths)
        n_windows = (args.n_windows if args.n_windows is not None
                     else _config_default(cfg, "n_windows", 8))
        query_positions = (args.query_positions
                           if args.query_positions is not None
                           else _config_default(cfg, "query_positions", 4))
        n_bootstrap = (args.bootstrap if args.bootstrap is not None
                       else _config_default(cfg, "bootstrap", 400))
        tolerance = (args.tolerance if args.tolerance is not None
                     else _config_default(cfg, "tolerance",
                                          DEFAULT_TOLERANCE))
        growth_modes = (args.growth_modes if args.growth_modes is not None
                        else _config_default(cfg, "growth_modes",
                                             DEFAULT_GROWTH_MODES))
        payload = run_length_ppl(
            args.ckpt, args.data_dir, lengths=lengths, n_windows=n_windows,
            query_positions=query_positions, growth_modes=growth_modes,
            n_bootstrap=n_bootstrap, tolerance=tolerance, seed=args.seed,
            device=args.device, amp=args.amp, config=args.config)
        payload["argv"] = list(sys.argv if argv is None else argv)
    except (ValueError, FileNotFoundError, PermissionError) as e:
        p.error(str(e))
    if args.out:
        save_json(payload, args.out)
        print(f"[lengthppl] -> {args.out}")
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"[lengthppl] {payload['variant']} lengths={payload['lengths']} "
          f"windows={len(payload['window_starts'])}")
    for growth, result in payload["results"].items():
        slope = result["nll_slope_per_log2"]
        per_length = result["per_length"]
        first = per_length.get(str(payload["lengths"][0]), {}).get("ppl")
        last = per_length.get(str(payload["lengths"][-1]), {}).get("ppl")
        print(f"  - {growth}: nll_slope={slope.get('estimate')} "
              f"class={slope.get('class')} ppl[{payload['lengths'][0]}]="
              f"{first} ppl[{payload['lengths'][-1]}]={last}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
