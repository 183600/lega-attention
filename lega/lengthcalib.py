"""训练后注意力的经验长度校准审计。

``lega.length_null`` 只诊断合成零假设分布；它不能回答训练模型的实际注意力
是否已经长度校准。本模块在真实 val 文本上构造跨长度的配对窗口，逐层逐头
测量::

    c = log(n) + log(max_j p_j)

随 ``log2(n)`` 的斜率。``c`` 的斜率接近 0 表示最大注意力概率按 1/n 收缩，
斜率接近 1 表示 ``c`` 仍随长度增长。suffix 增长固定窗口末尾（query token
与后缀内容不变，只增加前缀），并显式加入 ``(window, query_offset)`` 固定
效应；prefix 增长作为敏感性分析。

用法::

    python -m lega.lengthcalib --ckpt out/small/lega/model_best.pt \
        --data_dir data/wikitext103 --out results/small/length_calibration_lega.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Iterable, List

import numpy as np
import torch

from .data import load_bin
from .model import load_model
from .repro import (current_pipeline_run_id, load_provenance, sha256_file)
from .train import _val_eval_indices
from .utils import (amp_autocast, config_bool, enable_tf32, get_device,
                    save_json, set_seed, strict_int)

__all__ = [
    "DEFAULT_GROWTH_MODES",
    "DEFAULT_LENGTHS",
    "analyze_calibration_records",
    "calibration_window_starts",
    "collect_calibration_records",
    "parse_growth_modes",
    "parse_lengths",
    "run_length_calibration",
    "main",
]

DEFAULT_LENGTHS = (32, 64, 128, 256, 512, 1024)
DEFAULT_GROWTH_MODES = ("suffix", "prefix")
GROWTH_MODES = ("suffix", "prefix")
METRIC_KEYS = ("c", "log_max_p", "gate_logit", "g")


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


def parse_lengths(value: Iterable[int]) -> List[int]:
    if isinstance(value, str):
        raw = [x.strip() for x in value.split(",") if x.strip()]
        lengths = [_integer(x, "lengths") for x in raw]
    elif isinstance(value, (list, tuple)):
        lengths = [_integer(x, "lengths") for x in value]
    else:
        raise ValueError(f"lengths 需要整数列表或逗号分隔字符串，收到 {value!r}")
    if len(lengths) < 2:
        raise ValueError("lengths 至少需要两个不同长度")
    if len(set(lengths)) != len(lengths):
        raise ValueError(f"lengths 不能重复，收到 {lengths}")
    for length in lengths:
        if length < 2:
            raise ValueError(f"lengths 中每个长度必须 >= 2，收到 {length}")
    return sorted(lengths)


def parse_growth_modes(value) -> List[str]:
    if isinstance(value, str):
        modes = [x.strip() for x in value.split(",") if x.strip()]
    elif isinstance(value, (list, tuple)):
        modes = [str(x).strip() for x in value]
    else:
        raise ValueError(f"growth_modes 需要字符串列表，收到 {value!r}")
    if not modes:
        raise ValueError("growth_modes 不能为空")
    invalid = [m for m in modes if m not in GROWTH_MODES]
    if invalid:
        raise ValueError(
            f"未知 growth_mode: {', '.join(invalid)}；可选: {', '.join(GROWTH_MODES)}")
    if len(set(modes)) != len(modes):
        raise ValueError(f"growth_modes 不能重复，收到 {modes}")
    return modes


def calibration_window_starts(val_len: int, max_length: int,
                              n_windows: int) -> List[int]:
    """与训练评测同口径的等间隔非重叠窗口起点（token 下标）。"""
    indices, _ = _val_eval_indices(
        val_len, max_length, _integer(n_windows, "n_windows", 1))
    return [int(i) * int(max_length) for i in indices]


def _window_ids(val, start: int, length: int, growth: str,
                window_length: int) -> np.ndarray:
    if growth == "suffix":
        end = start + int(window_length)
        return np.asarray(val[end - length:end], dtype=np.uint16)
    return np.asarray(val[start:start + length], dtype=np.uint16)


def _ids_start(start: int, length: int, growth: str,
               window_length: int) -> int:
    if growth == "suffix":
        return start + int(window_length) - int(length)
    return int(start)


def collect_calibration_records(model, val, window_starts: List[int],
                                lengths: List[int], query_positions: int,
                                growth: str, device: str,
                                amp: bool = True,
                                batch_windows: int = 8) -> List[dict]:
    """逐 (window, length, query_offset, layer, head) 收集注意力统计。"""
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
        if start + max_length > len(val):
            raise ValueError(
                f"窗口起点 {start} + 最长长度 {max_length} 超出 val 长度 {len(val)}")
    records: List[dict] = []
    amp_ctx = amp_autocast(device, bool(amp))
    for length in lengths:
        for b0 in range(0, len(starts), batch_windows):
            batch_starts = starts[b0:b0 + batch_windows]
            ids_np = np.stack([
                _window_ids(val, start, length, growth, max_length)
                for start in batch_starts]).astype(np.int64)
            ids = torch.from_numpy(ids_np)
            if device.startswith("cuda"):
                ids = ids.to(device)
            with torch.no_grad(), amp_ctx:
                out = model(ids, need_stats=True, last_logits_only=True)
            stats = out.get("gate_stats") or []
            if not stats:
                raise ValueError("模型没有返回 gate_stats；无法做长度校准审计")
            q = query_positions
            for layer_stats in stats:
                layer = _integer(layer_stats.get("layer"), "layer", 0)
                max_p = layer_stats["max_p"][:, :, -q:].float().cpu().numpy()
                log_n = layer_stats["log_n"][-q:].float().cpu().numpy()
                gate_g = layer_stats.get("g")
                gate_logit = layer_stats.get("logit")
                if gate_g is not None:
                    gate_g = gate_g[:, :, -q:].float().cpu().numpy()
                if gate_logit is not None:
                    gate_logit = gate_logit[:, :, -q:].float().cpu().numpy()
                for b, start in enumerate(batch_starts):
                    window = b0 + b
                    for head in range(max_p.shape[1]):
                        for offset in range(q):
                            mp = _finite_float(max_p[b, head, offset])
                            ln = _finite_float(log_n[offset])
                            if mp is None or ln is None or mp <= 0.0:
                                continue
                            rec = {
                                "window": window,
                                "window_start": _ids_start(
                                    start, length, growth, max_length),
                                "length": int(length),
                                "query_offset": int(offset),
                                "layer": int(layer),
                                "head": int(head),
                                "max_p": mp,
                                "log_max_p": float(math.log(mp)),
                                "log_n": ln,
                                "c": float(ln + math.log(mp)),
                            }
                            if gate_g is not None:
                                g = _finite_float(gate_g[b, head, offset])
                                if g is not None:
                                    rec["g"] = g
                            if gate_logit is not None:
                                logit = _finite_float(
                                    gate_logit[b, head, offset])
                                if logit is not None:
                                    rec["gate_logit"] = logit
                            records.append(rec)
    return records


def _metric_groups(records: List[dict], key: str) -> List[dict]:
    grouped = {}
    for rec in records:
        if not isinstance(rec, dict):
            continue
        value = _finite_float(rec.get(key))
        length = _finite_float(rec.get("length"))
        if value is None or length is None or length < 2:
            continue
        length = int(length)
        try:
            window = _integer(rec.get("window"), "window", 0)
            offset = _integer(rec.get("query_offset"), "query_offset", 0)
        except ValueError:
            continue
        g = grouped.setdefault((window, offset), {})
        if length in g:
            g[length] = None
        else:
            g[length] = value
    out = []
    for (window, offset), values in grouped.items():
        pairs = [(n, v) for n, v in values.items() if v is not None]
        if len(pairs) < 2:
            continue
        out.append({
            "window": window,
            "query_offset": offset,
            "lengths": np.asarray([p[0] for p in pairs], dtype=np.float64),
            "values": np.asarray([p[1] for p in pairs], dtype=np.float64),
        })
    return out


def _slope_from_groups(groups: List[dict]):
    if not groups:
        return None
    xs = []
    ys = []
    for group in groups:
        lengths = np.asarray(group["lengths"], dtype=np.float64)
        values = np.asarray(group["values"], dtype=np.float64)
        if lengths.size < 2 or values.size != lengths.size:
            continue
        x = np.log2(lengths)
        y = values
        xm = float(x.mean())
        ym = float(y.mean())
        xs.append(x - xm)
        ys.append(y - ym)
    if not xs:
        return None
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    sxx = float(np.dot(x, x))
    if not math.isfinite(sxx) or sxx <= 0.0:
        return None
    slope = float(np.dot(x, y) / sxx)
    if not math.isfinite(slope):
        return None
    return slope


def _slope_from_points(lengths, values):
    xs = np.log2(np.asarray(lengths, dtype=np.float64))
    ys = np.asarray(values, dtype=np.float64)
    if xs.size < 2 or ys.size != xs.size:
        return None
    xm = float(xs.mean())
    ym = float(ys.mean())
    xc = xs - xm
    sxx = float(np.dot(xc, xc))
    if not math.isfinite(sxx) or sxx <= 0.0:
        return None
    slope = float(np.dot(xc, ys - ym) / sxx)
    return slope if math.isfinite(slope) else None


def _bootstrap_slope(groups: List[dict], n_bootstrap: int, seed: int):
    if not groups or n_bootstrap <= 0:
        return None
    by_window = {}
    for group in groups:
        by_window.setdefault(group["window"], []).append(group)
    windows = sorted(by_window)
    if len(windows) < 2:
        return None
    rng = np.random.default_rng(seed)
    slopes = []
    for _ in range(n_bootstrap):
        picks = rng.integers(0, len(windows), size=len(windows))
        sample = []
        for pick in picks:
            sample.extend(by_window[windows[int(pick)]])
        slope = _slope_from_groups(sample)
        if slope is not None:
            slopes.append(slope)
    if not slopes:
        return None
    arr = np.asarray(slopes, dtype=np.float64)
    lo, hi = np.quantile(arr, [0.025, 0.975])
    neg = float(np.mean(arr <= 0.0))
    pos = float(np.mean(arr >= 0.0))
    p = min(1.0, 2.0 * min((neg * arr.size + 1.0) / (arr.size + 1.0),
                           (pos * arr.size + 1.0) / (arr.size + 1.0)))
    return {
        "ci95": [float(lo), float(hi)],
        "p_boot": float(p),
        "n_valid": int(arr.size),
        "n_requested": int(n_bootstrap),
    }


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


def _summarize_metric(per_head: List[dict], key: str, tolerance: float):
    slopes = [p[key + "_slope"] for p in per_head
              if _finite_float(p.get(key + "_slope")) is not None]
    if not slopes:
        return {
            "available": False,
            "n_heads": 0,
            "tolerance": tolerance,
        }
    arr = np.asarray(slopes, dtype=np.float64)
    classes = [p.get(key + "_class") for p in per_head]
    return {
        "available": True,
        "n_heads": int(arr.size),
        "median": float(np.median(arr)),
        "mean": float(arr.mean()),
        "q25": float(np.quantile(arr, 0.25)),
        "q75": float(np.quantile(arr, 0.75)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "n_increasing": int(sum(c == "increasing" for c in classes)),
        "n_decreasing": int(sum(c == "decreasing" for c in classes)),
        "n_flat": int(sum(c == "flat" for c in classes)),
        "n_unresolved": int(sum(c == "unresolved" for c in classes)),
        "n_undefined": int(sum(c == "undefined" for c in classes)),
        "n_no_ci": int(sum(c == "no_ci" for c in classes)),
        "fraction_flat": float(sum(c == "flat" for c in classes) / len(per_head)),
        "tolerance": tolerance,
    }


def _per_length_summary(records: List[dict], lengths: List[int]):
    out = {}
    for length in lengths:
        rows = [r for r in records if r.get("length") == length]
        row = {"n": len(rows)}
        for key in ("c", "max_p", "log_max_p", "log_n", "gate_logit", "g"):
            vals = [v for v in (_finite_float(r.get(key)) for r in rows)
                    if v is not None]
            if vals:
                arr = np.asarray(vals, dtype=np.float64)
                row[key] = {
                    "mean": float(arr.mean()),
                    "sd": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
                    "n": int(arr.size),
                }
        out[str(length)] = row
    return out


def analyze_calibration_records(records: List[dict], lengths: Iterable[int],
                                growth: str = "suffix", *,
                                tolerance: float = 0.05,
                                n_bootstrap: int = 400,
                                seed: int = 0) -> dict:
    lengths = parse_lengths(lengths)
    growth = parse_growth_modes(growth)[0]
    tolerance = _finite_float(tolerance)
    if tolerance is None or tolerance < 0.0:
        raise ValueError(f"tolerance 必须是非负有限数值，收到 {tolerance!r}")
    n_bootstrap = _integer(n_bootstrap, "n_bootstrap", 0)
    seed = _integer(seed, "seed", 0)
    records = [r for r in records if isinstance(r, dict)]
    heads = sorted({(r.get("layer"), r.get("head")) for r in records
                    if isinstance(r.get("layer"), int)
                    and isinstance(r.get("head"), int)})
    per_head = []
    for layer, head in heads:
        rows = [r for r in records
                if r.get("layer") == layer and r.get("head") == head]
        row = {"layer": layer, "head": head}
        for key in METRIC_KEYS:
            groups = _metric_groups(rows, key)
            slope = _slope_from_groups(groups)
            boot = _bootstrap_slope(
                groups, n_bootstrap,
                seed + 1000003 * (int(layer) * 1000 + int(head))
                + sum(ord(ch) for ch in key))
            row[key + "_slope"] = slope
            row[key + "_ci95"] = None if boot is None else boot["ci95"]
            row[key + "_p_boot"] = None if boot is None else boot["p_boot"]
            row[key + "_class"] = _slope_class(
                slope, None if boot is None else boot["ci95"], tolerance)
            row[key + "_n_groups"] = len(groups)
            row[key + "_n_obs"] = sum(int(g["lengths"].size) for g in groups)
        windows = {g["window"] for g in _metric_groups(rows, "c")}
        row["n_windows"] = len(windows)
        per_head.append(row)
    summary = {key: _summarize_metric(per_head, key, tolerance)
               for key in METRIC_KEYS}
    per_length = _per_length_summary(records, lengths)
    log_n_means = []
    for length in lengths:
        row = per_length.get(str(length), {}).get("log_n")
        if not isinstance(row, dict):
            log_n_means = []
            break
        log_n_means.append(row.get("mean"))
    summary["log_n"] = {
        "available": len(log_n_means) == len(lengths),
        "sanity_slope_from_all": (
            _slope_from_points(lengths, log_n_means)
            if len(log_n_means) == len(lengths) else None),
        "n_records": len(records),
    }
    return {
        "growth": growth,
        "lengths": lengths,
        "tolerance": tolerance,
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "fixed_effects": "window + query_offset",
        "per_length": per_length,
        "summary": summary,
        "per_head": per_head,
    }


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


def run_length_calibration(ckpt: str, data_dir: str, *, lengths=None,
                           n_windows: int = 8, query_positions: int = 4,
                           growth_modes=None, n_bootstrap: int = 400,
                           tolerance: float = 0.05, seed=None,
                           device: str = "auto", amp=None,
                           config: str = None) -> dict:
    ckpt_dir = os.path.dirname(os.path.abspath(ckpt))
    cfg = _load_config(config, ckpt_dir)
    lc_cfg = cfg.get("length_calibration")
    lc_cfg = lc_cfg if isinstance(lc_cfg, dict) else {}
    if lengths is None:
        lengths = lc_cfg.get("lengths")
    if lengths is None:
        lengths = (cfg.get("length_null") or {}).get("lengths")
    if lengths is None:
        lengths = list(DEFAULT_LENGTHS)
    lengths = parse_lengths(lengths)
    if growth_modes is None:
        growth_modes = lc_cfg.get("growth_modes", DEFAULT_GROWTH_MODES)
    growth_modes = parse_growth_modes(growth_modes)
    n_windows = _integer(n_windows, "n_windows", 1)
    query_positions = _integer(query_positions, "query_positions", 1)
    n_bootstrap = _integer(n_bootstrap, "n_bootstrap", 0)
    tolerance = _finite_float(tolerance)
    if tolerance is None or tolerance < 0.0:
        raise ValueError(f"tolerance 必须是非负有限数值，收到 {tolerance!r}")

    device = get_device(device)
    enable_tf32(device)
    model, ck = load_model(ckpt, device=device, use_sdpa_standard=False)
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
        records = collect_calibration_records(
            model, val, starts, lengths, query_positions, growth,
            device=device, amp=bool(amp))
        results[growth] = analyze_calibration_records(
            records, lengths, growth, tolerance=tolerance,
            n_bootstrap=n_bootstrap, seed=seed)
        results[growth]["n_records"] = len(records)
    return {
        "schema": 1,
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
            "fixed_effects": "window + query_offset",
            "suffix_growth": (
                "固定窗口末尾与 query token，只增加前缀长度；用于控制位置"
                "与 query 内容混杂"),
            "prefix_growth": (
                "固定窗口起点，只增加后缀长度；query token 随长度改变，"
                "作为敏感性分析"),
            "note": (
                "观测性审计；不消融门控参数，不替代 RULER 分档或因果干预。"
                "bootstrap 以窗口为聚类单元，区间为描述性不确定性。"),
        },
        "provenance": load_provenance(ckpt_dir, data_dir, device=device),
        "argv": [],
    }


def _config_default(cfg: dict, key: str, default):
    block = cfg.get("length_calibration")
    block = block if isinstance(block, dict) else {}
    return block.get(key, default)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="训练后注意力的经验长度校准审计")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--config", default=None,
                   help="可选配置；默认读取 checkpoint 目录的 config_used.json")
    p.add_argument("--lengths", default=None,
                   help="逗号分隔的长度；默认读 length_calibration.lengths，"
                        "再回退到 length_null.lengths")
    p.add_argument("--n_windows", type=int, default=None)
    p.add_argument("--query_positions", type=int, default=None)
    p.add_argument("--growth_modes", default=None)
    p.add_argument("--bootstrap", type=int, default=None,
                   help="窗口聚类 bootstrap 重采样次数（0 关闭）")
    p.add_argument("--tolerance", type=float, default=None,
                   help="判定 slope 为 flat 的绝对阈值（默认 0.05）")
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
                     else _config_default(cfg, "tolerance", 0.05))
        growth_modes = (args.growth_modes if args.growth_modes is not None
                        else _config_default(cfg, "growth_modes",
                                             DEFAULT_GROWTH_MODES))
        payload = run_length_calibration(
            args.ckpt, args.data_dir, lengths=lengths, n_windows=n_windows,
            query_positions=query_positions, growth_modes=growth_modes,
            n_bootstrap=n_bootstrap, tolerance=tolerance, seed=args.seed,
            device=args.device, amp=args.amp, config=args.config)
        payload["argv"] = list(sys.argv if argv is None else argv)
    except (ValueError, FileNotFoundError, PermissionError) as e:
        p.error(str(e))
    if args.out:
        save_json(payload, args.out)
        print(f"[lengthcalib] -> {args.out}")
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"[lengthcalib] {payload['variant']} lengths={payload['lengths']} "
          f"windows={len(payload['window_starts'])}")
    for growth, result in payload["results"].items():
        c_sum = result["summary"]["c"]
        print(f"  - {growth}: c_slope_median="
              f"{c_sum.get('median')} flat_heads="
              f"{c_sum.get('n_flat')}/{c_sum.get('n_heads')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
