"""嵌套上下文配对 NIAH：把“上下文长度”与“证据位置/距离”分开。

每个 ``pair_id`` 在所有上下文长度和 growth 模式下使用同一 key/value 与同一
填充流。needle 周围的局部上下文固定，只在远左/远右扩展：

* ``suffix``：固定前缀，向右扩展后缀；证据绝对位置固定、到问题的距离增长；
* ``prefix``：固定后缀，向左扩展前缀；证据位置增长、到问题的距离固定；
* ``split``：两侧各扩展一半；位置与距离同时增长。

这样长度效应不再是“随机位置 + 不同样本”的观测相关，而是同一证据在配对
上下文中的条件比较。分析只做描述性斜率与配对 McNemar，不改写预注册的
确认性判定。
"""
import hashlib
import json
import math
import os
import random
from typing import Dict, List, Optional

import torch

from .ruler import (ContextTooSmall, FILLER, KEY_WORDS, NEEDLE_TMPL,
                    QUESTION_TMPL, _cached_encode, _needle_excerpt, _rand_digit,
                    _stable_hash, _token_span_for_text, _tokenizer_roundtrip,
                    record_corpus_test_access, score_prediction,
                    standalone_numeric_tokens)
from .utils import (amp_autocast, amp_dtype_for, config_bool, holm_bonferroni,
                    mcnemar_exact_p, save_json, strict_int, wilson_score_ci)

GROWTH_MODES = ("suffix", "prefix", "split")
DEFAULT_GROWTH_MODES = ("suffix", "prefix")
CORE_WINDOW = 16


def _normalize_ctx_lens(ctx_lens) -> List[int]:
    if isinstance(ctx_lens, (str, bytes)):
        raise ValueError(
            f"ctx_lens 必须是整数序列，收到字符串 {ctx_lens!r}")
    vals = [strict_int(c, "ctx_lens") for c in ctx_lens]
    if len(vals) < 2:
        raise ValueError(f"嵌套设计至少需要两个上下文长度，收到 {vals}")
    if any(c < 1 for c in vals):
        raise ValueError(f"ctx_lens 必须都是 >= 1 的整数，收到 {vals}")
    if len(set(vals)) != len(vals):
        raise ValueError(f"ctx_lens 不能重复（重复档位会破坏配对设计），收到 {vals}")
    return sorted(vals)


def _normalize_growth_modes(growth_modes) -> tuple:
    if isinstance(growth_modes, (str, bytes)):
        raise ValueError(
            f"growth_modes 必须是序列，收到字符串 {growth_modes!r}")
    vals = tuple(growth_modes)
    if not vals:
        raise ValueError("growth_modes 不能为空")
    unknown = sorted(set(vals) - set(GROWTH_MODES))
    if unknown:
        raise ValueError(
            f"未知 growth 模式 {unknown}；可选 {', '.join(GROWTH_MODES)}")
    if len(set(vals)) != len(vals):
        raise ValueError(f"growth_modes 不能重复，收到 {vals}")
    return vals


def _hash_ids(ids: List[int]) -> str:
    raw = json.dumps([int(i) for i in ids], separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _make_filler(tok, length: int, seed: int, mode: str,
                 corpus_sents: Optional[List[str]]) -> List[int]:
    if length < 1:
        raise ValueError(f"填充流长度必须 >= 1，收到 {length}")
    rng = random.Random(int(seed))
    if mode == "corpus":
        if not corpus_sents:
            raise ContextTooSmall("niah_corpus 需要语料句子，但 corpus_sents 为空")
        out: List[int] = []
        while len(out) < length:
            ids = _cached_encode(tok, " " + rng.choice(corpus_sents).strip())
            if not ids:
                raise ContextTooSmall("干扰语料被分词为空，无法构造嵌套上下文")
            out.extend(ids)
        return out[:length]
    if mode != "synth":
        raise ValueError(f"未知 mode {mode!r}；可选 'synth'/'corpus'")
    unit = _cached_encode(tok, " " + FILLER)
    if not unit:
        raise ContextTooSmall("FILLER 被分词为空，无法构造嵌套上下文")
    return (unit * (length // len(unit) + 1))[:length]


def build_nested_niah(tok, ctx_lens, growth: str, *, key: Optional[str] = None,
                      value: Optional[str] = None, filler_seed: int = 0,
                      filler_ids: Optional[List[int]] = None,
                      pair_id: Optional[str] = None, mode: str = "synth",
                      corpus_sents: Optional[List[str]] = None,
                      base_ctx: Optional[int] = None) -> List[Dict]:
    """构造同一证据、同一填充流、不同上下文的配对 NIAH 样例。"""
    if growth not in GROWTH_MODES:
        raise ValueError(
            f"growth 必须是 {GROWTH_MODES} 之一，收到 {growth!r}")
    ctxs = _normalize_ctx_lens(ctx_lens)
    filler_seed = strict_int(filler_seed, "filler_seed")
    if mode not in ("synth", "corpus"):
        raise ValueError(f"mode 必须是 'synth'/'corpus'，收到 {mode!r}")
    base = ctxs[0] if base_ctx is None else strict_int(base_ctx, "base_ctx")
    if base != ctxs[0]:
        raise ValueError(
            f"base_ctx 必须等于最短上下文 {ctxs[0]}，收到 {base}")

    core_rng = random.Random(_stable_hash(f"nested-core:{filler_seed}"))
    if key is None:
        key = core_rng.choice(KEY_WORDS)
    if value is None:
        value = _rand_digit(core_rng, 7, 7)
    key, value = str(key), str(value)
    if not key or "\n" in key:
        raise ValueError(f"key 必须是非空单行字符串，收到 {key!r}")
    if not value.isdigit():
        raise ValueError(f"value 必须是数字串（NIAH 使用数值 EM），收到 {value!r}")
    if pair_id is None:
        pair_id = f"niah_nested:{filler_seed}:{key}:{value}"
    pair_id = str(pair_id)

    if not _tokenizer_roundtrip(tok, value):
        raise ContextTooSmall(
            "tokenizer 无法无损表示 niah 数值答案，无法定位 gold 跨度；"
            f"value={value!r}")
    needle_ids = tok.encode("\n" + NEEDLE_TMPL.format(key=key, value=value))
    question_ids = _cached_encode(tok, QUESTION_TMPL.format(key=key))
    if not needle_ids or not question_ids:
        raise ContextTooSmall("needle/question 被分词为空，无法构造样例")
    try:
        vs, ve = _token_span_for_text(tok, needle_ids, value)
    except AssertionError as e:
        raise ContextTooSmall(str(e)) from e

    base_room = base - len(needle_ids) - len(question_ids)
    if base_room < 8:
        raise ContextTooSmall(
            f"嵌套 NIAH 需要最短 ctx >= {len(needle_ids)+len(question_ids)+8}，"
            f"收到 {base}")
    base_left = max(1, base_room // 4)
    base_right = base_room - base_left
    if base_right < 1:
        raise ContextTooSmall(f"最短 ctx 的填充空间不足，room={base_room}")
    max_room = ctxs[-1] - len(needle_ids) - len(question_ids)
    max_extra = max_room - base_room
    filler_len = base_room + 2 * max_extra
    if filler_ids is None:
        filler_ids = _make_filler(
            tok, filler_len,
            _stable_hash(f"nested-filler:{filler_seed}"), mode, corpus_sents)
    else:
        filler_ids = [strict_int(x, "filler_ids") for x in filler_ids]
        if len(filler_ids) < filler_len:
            raise ValueError(
                f"filler_ids 长度 {len(filler_ids)} < 设计所需 {filler_len}；"
                "不能截断/重采样，否则跨长度填充流不再相同")
    filler_ids = filler_ids[:filler_len]
    filler_hash = _hash_ids(filler_ids)
    pivot = base_left + max_extra
    window = max(1, min(CORE_WINDOW, base_left, base_right))

    out: List[Dict] = []
    for ctx in ctxs:
        room = ctx - len(needle_ids) - len(question_ids)
        extra = room - base_room
        if extra < 0:
            raise ContextTooSmall(f"ctx={ctx} 不足以容纳 needle+question")
        if growth == "suffix":
            left_extra, right_extra = 0, extra
        elif growth == "prefix":
            left_extra, right_extra = extra, 0
        else:
            left_extra = extra // 2
            right_extra = extra - left_extra
        start = pivot - base_left - left_extra
        end = pivot + base_right + right_extra
        if start < 0 or end > len(filler_ids):
            raise AssertionError(
                f"内部填充切片越界: start={start}, end={end}, "
                f"len={len(filler_ids)}, ctx={ctx}, growth={growth}")
        prefix = filler_ids[start:pivot]
        suffix = filler_ids[pivot:end]
        ids = prefix + needle_ids + suffix + question_ids
        if len(ids) != ctx:
            raise AssertionError(
                f"嵌套样例长度 {len(ids)} != 请求 ctx={ctx}："
                "精确长度是配对设计的前提")
        core_ids = prefix[-window:] + needle_ids + suffix[:window]
        meta = {
            "key": key,
            "value": value,
            "pair_id": pair_id,
            "growth": growth,
            "nominal_ctx": ctx,
            "actual_tokens": len(ids),
            "n_tokens": len(ids),
            "base_ctx": base,
            "needle_pos": len(prefix),
            "evidence_position_fraction": len(prefix) / max(
                1, len(ids) - len(question_ids)),
            "evidence_distance_tokens": len(suffix),
            "distance_to_query_tokens": len(suffix),
            "prefix_tokens": len(prefix),
            "suffix_tokens": len(suffix),
            "depth": len(prefix) / max(1, len(ids) - len(question_ids)),
            "filler_stream_sha256": filler_hash,
            "core_context_sha256": _hash_ids(core_ids),
            "core_window": window,
        }
        out.append({
            "task": "niah_nested",
            "source_task": "niah_corpus" if mode == "corpus" else "niah_synth",
            "ids": ids,
            "answer": value,
            "answer_words": None,
            "needle_span": (len(prefix), len(prefix) + len(needle_ids)),
            "value_span": (len(prefix) + vs, len(prefix) + ve),
            "query_span": (len(prefix) + len(needle_ids) + len(suffix), len(ids)),
            "sample_id": f"{pair_id}@{growth}@{ctx}",
            "instance_id": f"{pair_id}@{ctx}",
            "pair_id": pair_id,
            "growth": growth,
            "meta": meta,
        })
    return out


def build_nested_sample(tok, ctx_lens, *, seed: int = 1234,
                        sample_index: int = 0,
                        growth_modes=DEFAULT_GROWTH_MODES,
                        mode: str = "synth",
                        corpus_sents: Optional[List[str]] = None,
                        base_ctx: Optional[int] = None) -> Dict:
    """为同一 pair 构造全部 growth 模式；任一模式失败则整个 pair 不可用。"""
    ctxs = _normalize_ctx_lens(ctx_lens)
    seed = strict_int(seed, "seed")
    sample_index = strict_int(sample_index, "sample_index")
    growth_modes = _normalize_growth_modes(growth_modes)
    core_rng = random.Random(
        _stable_hash(f"nested-core:{seed}:{sample_index}"))
    key = core_rng.choice(KEY_WORDS)
    value = _rand_digit(core_rng, 7, 7)
    filler_seed = _stable_hash(f"nested-filler-sample:{seed}:{sample_index}")
    pair_id = f"niah_nested:{seed}:{sample_index}"
    built: Dict[str, List[Dict]] = {}
    skipped: List[str] = []
    for growth in growth_modes:
        try:
            built[growth] = build_nested_niah(
                tok, ctxs, growth, key=key, value=value,
                filler_seed=filler_seed, pair_id=pair_id, mode=mode,
                corpus_sents=corpus_sents, base_ctx=base_ctx)
        except ContextTooSmall as e:
            skipped.append(f"{growth}@{pair_id}: {e}")
    return {"pair_id": pair_id, "key": key, "value": value,
            "growth": built, "skipped": skipped,
            "ok": not skipped and len(built) == len(growth_modes)}


def _outcome_records(cell: Dict, where: str) -> Optional[Dict[str, Dict]]:
    outcomes = cell.get("outcomes")
    if outcomes is None:
        return None
    if not isinstance(outcomes, list):
        raise ValueError(f"{where}.outcomes 必须是列表")
    out: Dict[str, Dict] = {}
    for i, rec in enumerate(outcomes):
        if not isinstance(rec, dict):
            raise ValueError(f"{where}.outcomes[{i}] 必须是对象")
        pid = rec.get("pair_id")
        if not isinstance(pid, str) or not pid:
            raise ValueError(f"{where}.outcomes[{i}].pair_id 必须是非空字符串")
        if pid in out:
            raise ValueError(
                f"{where}.outcomes 中 pair_id={pid!r} 重复；配对键必须唯一")
        em = rec.get("em")
        if em not in (0, 0.0, 1, 1.0):
            raise ValueError(
                f"{where}.outcomes[{i}].em 必须是 0/1，收到 {em!r}")
        out[pid] = rec
    n = cell.get("n")
    if n is not None and strict_int(n, f"{where}.n") != len(out):
        raise ValueError(
            f"{where}.n={n} 与 outcomes 长度 {len(out)} 不一致")
    if cell.get("em") is not None and out:
        mean = sum(int(r["em"]) for r in out.values()) / len(out)
        if abs(float(cell["em"]) - mean) > 1e-9:
            raise ValueError(
                f"{where}.em={cell['em']} 与 outcomes 均值 {mean} 不一致")
    return out


def _slope(xs: List[float], ys: List[float]) -> Optional[float]:
    n = len(xs)
    if n < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    if den <= 0.0:
        return None
    return sum((xs[i] - mx) * (ys[i] - my) for i in range(n)) / den


def _as_results(payload) -> tuple:
    if not isinstance(payload, dict):
        raise ValueError("嵌套结果必须是 JSON 对象")
    if "results" in payload:
        return payload["results"], payload.get("design") or {}
    return payload, {}


def analyze_nested(payload, base_ctx: Optional[int] = None) -> Dict:
    """对配对嵌套结果做逐档描述、同一 pair 的转移与设计自检。"""
    results, design = _as_results(payload)
    if not isinstance(results, dict) or not results:
        raise ValueError("嵌套 results 不能为空")
    growth_modes = _normalize_growth_modes(tuple(results))
    all_ctx = set()
    normalized = {}
    for growth in growth_modes:
        cells = results[growth]
        if not isinstance(cells, dict) or not cells:
            raise ValueError(f"growth={growth} 的 results 不能为空")
        by_ctx = {}
        for raw_ctx, cell in cells.items():
            ctx = strict_int(raw_ctx, f"{growth}.ctx")
            if ctx in by_ctx:
                raise ValueError(
                    f"growth={growth} 中 ctx={ctx} 以不同键重复出现")
            by_ctx[ctx] = cell
            all_ctx.add(ctx)
        normalized[growth] = dict(sorted(by_ctx.items()))
    base = (strict_int(base_ctx, "base_ctx") if base_ctx is not None
            else design.get("base_ctx"))
    base = (strict_int(base, "base_ctx") if base is not None else min(all_ctx))
    if base not in all_ctx:
        raise ValueError(f"base_ctx={base} 不在任何 growth 的档位中")

    out_growth: Dict[str, Dict] = {}
    checks = {
        "lengths_exact": None,
        "core_context_consistent": True,
        "pairing_complete": True,
        "suffix_position_fixed": None,
        "prefix_distance_fixed": None,
        "suffix_distance_varies": None,
        "prefix_position_varies": None,
    }
    any_outcomes = False
    length_checked = False
    for growth in growth_modes:
        cells = normalized[growth]
        ctxs = list(cells)
        growth_base = base if base in cells else ctxs[0]
        records = {ctx: _outcome_records(cells[ctx], f"results.{growth}.{ctx}")
                   for ctx in ctxs}
        if any(rec is not None for rec in records.values()):
            any_outcomes = True
        else:
            checks["pairing_complete"] = False
            checks["core_context_consistent"] = False

        stats = {}
        for ctx in ctxs:
            cell = cells[ctx]
            recs = records[ctx]
            if recs is not None:
                n = len(recs)
                k = sum(int(r["em"]) for r in recs.values())
                em = k / n if n else None
                ci = wilson_score_ci(k, n) if n else None
            else:
                n = (strict_int(cell["n"], f"results.{growth}.{ctx}.n")
                     if cell.get("n") is not None else 0)
                em = (float(cell["em"]) if cell.get("em") is not None
                      else None)
                ci = cell.get("em_ci95_wilson")
            rec = {"ctx": ctx, "n": n, "em": em,
                   "em_ci95_wilson": list(ci) if ci is not None else None,
                   "outcomes_present": recs is not None}
            if recs is not None:
                rec["n_correct"] = int(sum(int(r["em"])
                                           for r in recs.values()))
            stats[str(ctx)] = rec

            if cell.get("actual_tokens") is not None:
                length_checked = True
                if checks["lengths_exact"] is None:
                    checks["lengths_exact"] = True
                if strict_int(cell["actual_tokens"],
                              f"results.{growth}.{ctx}.actual_tokens") != ctx:
                    checks["lengths_exact"] = False
            if recs is not None:
                for r in recs.values():
                    if r.get("actual_tokens") is not None:
                        length_checked = True
                        if checks["lengths_exact"] is None:
                            checks["lengths_exact"] = True
                        if strict_int(r["actual_tokens"],
                                      "outcome.actual_tokens") != ctx:
                            checks["lengths_exact"] = False

        all_pids = [set(recs) for recs in records.values()
                    if recs is not None]
        complete_pids = set.intersection(*all_pids) if all_pids else set()
        if not all_pids or len(all_pids) != len(ctxs):
            checks["pairing_complete"] = False
        if all_pids and complete_pids != set.union(*all_pids):
            checks["pairing_complete"] = False
        for pid in complete_pids:
            pair_recs = [recs[pid] for recs in records.values()
                         if recs is not None]
            cores = [r.get("core_context_sha256") for r in pair_recs]
            key_values = [(r.get("key"), r.get("value")) for r in pair_recs]
            if (any(c is None for c in cores) or len(set(cores)) != 1
                    or any(k is None or v is None
                           for k, v in key_values)
                    or len(set(key_values)) != 1):
                checks["core_context_consistent"] = False

        transitions = []
        if records[growth_base] is not None:
            bmap = records[growth_base]
            for ctx in ctxs:
                if ctx == growth_base or records[ctx] is None:
                    continue
                cmap = records[ctx]
                common = sorted(set(bmap) & set(cmap))
                if not common:
                    transitions.append({
                        "ctx": ctx, "n_pairs": 0,
                        "reason": "没有共同 pair_id",
                    })
                    continue
                base_only = sum(1 for p in common
                                if int(bmap[p]["em"]) == 1
                                and int(cmap[p]["em"]) == 0)
                ctx_only = sum(1 for p in common
                               if int(bmap[p]["em"]) == 0
                               and int(cmap[p]["em"]) == 1)
                both = sum(1 for p in common
                           if int(bmap[p]["em"]) == 1
                           and int(cmap[p]["em"]) == 1)
                neither = sum(1 for p in common
                              if int(bmap[p]["em"]) == 0
                              and int(cmap[p]["em"]) == 0)
                b_em = sum(int(bmap[p]["em"]) for p in common) / len(common)
                c_em = sum(int(cmap[p]["em"]) for p in common) / len(common)
                transitions.append({
                    "ctx": ctx,
                    "base_ctx": growth_base,
                    "n_pairs": len(common),
                    "base_em": b_em,
                    "ctx_em": c_em,
                    "em_diff": c_em - b_em,
                    "both_correct": both,
                    "base_only": base_only,
                    "ctx_only": ctx_only,
                    "neither": neither,
                    "mcnemar_p": mcnemar_exact_p(ctx_only, base_only),
                })

        xs = [math.log2(c) for c in ctxs
              if stats[str(c)]["em"] is not None]
        ys = [stats[str(c)]["em"] for c in ctxs
              if stats[str(c)]["em"] is not None]
        slope = _slope(xs, ys)
        positions = [r.get("needle_pos") for recs in records.values()
                     if recs is not None for r in recs.values()
                     if r.get("needle_pos") is not None]
        distances = [r.get("evidence_distance_tokens")
                     for recs in records.values() if recs is not None
                     for r in recs.values()
                     if r.get("evidence_distance_tokens") is not None]
        if positions:
            if growth == "suffix":
                checks["suffix_position_fixed"] = len(set(positions)) == 1
            if growth == "prefix":
                checks["prefix_position_varies"] = len(set(positions)) > 1
        if distances:
            if growth == "prefix":
                checks["prefix_distance_fixed"] = len(set(distances)) == 1
            if growth == "suffix":
                checks["suffix_distance_varies"] = len(set(distances)) > 1

        out_growth[growth] = {
            "base_ctx": growth_base,
            "ctx_lens": ctxs,
            "cells": stats,
            "paired_transitions": transitions,
            "em_slope_per_log2_ctx": slope,
            "em_slope_pp_per_doubling": (slope * 100.0
                                         if slope is not None else None),
            "streams": {
                "filler_stream_sha256": sorted({
                    c.get("filler_stream_sha256") for c in cells.values()
                    if c.get("filler_stream_sha256")}),
                "core_context_sha256": sorted({
                    r.get("core_context_sha256")
                    for recs in records.values() if recs is not None
                    for r in recs.values()
                    if r.get("core_context_sha256")}),
            },
        }
    if not any_outcomes:
        checks["pairing_complete"] = False
        checks["core_context_consistent"] = False
    if not length_checked:
        checks["lengths_exact"] = None
    transitions = [t for g in out_growth.values()
                   for t in g["paired_transitions"]
                   if t.get("mcnemar_p") is not None]
    adjusted = holm_bonferroni([t["mcnemar_p"] for t in transitions])
    for rec, p_adj in zip(transitions, adjusted):
        rec["p_holm"] = p_adj
    return {
        "task": "niah_nested_analysis",
        "scope": "exploratory_paired_design",
        "base_ctx": base,
        "growth": out_growth,
        "checks": checks,
        "holm_family": "nested_within_length",
        "notes": [
            "同一 pair 在所有档位共享 key/value 与填充流；后缀/前缀模式的"
            "局部上下文固定，长度变化由远端扩展产生。",
            "paired_transitions 是同一批评测题面上的配对 McNemar；"
            "p_holm 对全部档位检验做 Holm 校正，仍不跨模型、不替代预注册"
            "的确认性检验。",
            "斜率只汇总本 payload 的档位点估计，n<3 或无 outcomes 时为 null。",
        ],
    }


def _normalize_cells(cells: Dict, where: str) -> Dict[int, Dict]:
    if not isinstance(cells, dict) or not cells:
        raise ValueError(f"{where} 不能为空")
    out = {}
    for raw_ctx, cell in cells.items():
        ctx = strict_int(raw_ctx, f"{where}.ctx")
        if ctx in out:
            raise ValueError(f"{where} 中 ctx={ctx} 以不同键重复出现")
        out[ctx] = cell
    return dict(sorted(out.items()))


def compare_nested(candidate, reference) -> Dict:
    """按 pair_id 对齐两个嵌套 payload，做逐档配对 McNemar + Holm。"""
    c_res, c_design = _as_results(candidate)
    r_res, r_design = _as_results(reference)
    if not isinstance(c_res, dict) or not isinstance(r_res, dict):
        raise ValueError("candidate/reference results 必须是对象")
    common_growth = sorted(set(c_res) & set(r_res))
    if not common_growth:
        raise ValueError("candidate/reference 没有共同 growth 模式")
    for field in ("mode", "base_ctx"):
        cv, rv = c_design.get(field), r_design.get(field)
        if cv is not None and rv is not None and cv != rv:
            raise ValueError(
                f"candidate/reference design.{field} 不一致: {cv!r} vs {rv!r}")

    tests: List[Dict] = []
    for growth in common_growth:
        c_cells = _normalize_cells(c_res[growth], f"candidate.{growth}")
        r_cells = _normalize_cells(r_res[growth], f"reference.{growth}")
        for ctx in sorted(set(c_cells) & set(r_cells)):
            crecs = _outcome_records(
                c_cells[ctx], f"candidate.{growth}.{ctx}")
            rrecs = _outcome_records(
                r_cells[ctx], f"reference.{growth}.{ctx}")
            if crecs is None or rrecs is None:
                continue
            common = sorted(set(crecs) & set(rrecs))
            if not common:
                continue
            for pid in common:
                for field in ("core_context_sha256", "key", "value",
                              "needle_pos", "evidence_distance_tokens"):
                    cv = crecs[pid].get(field)
                    rv = rrecs[pid].get(field)
                    if cv is not None and rv is not None and cv != rv:
                        raise ValueError(
                            f"{growth}@{ctx} pair={pid} 的 {field} 不一致；"
                            "两侧不是同一批配对题面，不能做配对 McNemar")
            c_only = sum(1 for p in common
                         if int(crecs[p]["em"]) == 1
                         and int(rrecs[p]["em"]) == 0)
            r_only = sum(1 for p in common
                         if int(crecs[p]["em"]) == 0
                         and int(rrecs[p]["em"]) == 1)
            both = sum(1 for p in common
                       if int(crecs[p]["em"]) == 1
                       and int(rrecs[p]["em"]) == 1)
            neither = sum(1 for p in common
                          if int(crecs[p]["em"]) == 0
                          and int(rrecs[p]["em"]) == 0)
            c_em = sum(int(crecs[p]["em"]) for p in common) / len(common)
            r_em = sum(int(rrecs[p]["em"]) for p in common) / len(common)
            tests.append({
                "growth": growth,
                "ctx": ctx,
                "n_pairs": len(common),
                "candidate_em": c_em,
                "reference_em": r_em,
                "em_diff": c_em - r_em,
                "both_correct": both,
                "candidate_only": c_only,
                "reference_only": r_only,
                "neither": neither,
                "mcnemar_p": mcnemar_exact_p(c_only, r_only),
            })
    pvals = [t["mcnemar_p"] for t in tests]
    adjusted = holm_bonferroni(pvals)
    for rec, p_adj in zip(tests, adjusted):
        rec["p_holm"] = p_adj

    slopes = {}
    for growth in common_growth:
        c_cells = _normalize_cells(c_res[growth], f"candidate.{growth}")
        r_cells = _normalize_cells(r_res[growth], f"reference.{growth}")
        common_ctx = sorted(set(c_cells) & set(r_cells))
        if len(common_ctx) < 2:
            continue
        c_em = []
        r_em = []
        for ctx in common_ctx:
            c_cell, r_cell = c_cells[ctx], r_cells[ctx]
            c_em.append(float(c_cell["em"])
                        if c_cell.get("em") is not None else None)
            r_em.append(float(r_cell["em"])
                        if r_cell.get("em") is not None else None)
        if all(v is not None for v in c_em + r_em):
            xs = [math.log2(c) for c in common_ctx]
            cs = _slope(xs, c_em)
            rs = _slope(xs, r_em)
            slopes[growth] = {
                "common_ctx": common_ctx,
                "candidate_slope": cs,
                "reference_slope": rs,
                "slope_diff": (cs - rs if cs is not None and rs is not None
                               else None),
            }
    return {
        "task": "niah_nested_comparison",
        "scope": "exploratory_paired_design",
        "growth": common_growth,
        "tests": tests,
        "n_tests": len(tests),
        "holm_family": "nested_context",
        "slopes": slopes,
        "notes": [
            "只比较两侧 pair_id、key/value、needle 位置与距离都一致的档位。",
            "p_holm 对该 payload 的全部 growth×ctx 检验做 Holm 校正，"
            "仍属探索性，不替代预注册确认性检验。",
        ],
    }


def _device_ids(rows: List[List[int]], device: str) -> torch.Tensor:
    out = torch.tensor(rows, dtype=torch.long)
    return out if device == "cpu" else out.to(device)


@torch.inference_mode()
def run_nested_niah(model, tok, device: str, ctx_lens, n_samples: int = 30,
                    seed: int = 1234, max_new: int = 12,
                    growth_modes=DEFAULT_GROWTH_MODES,
                    mode: str = "synth",
                    corpus_sents: Optional[List[str]] = None,
                    base_ctx: Optional[int] = None, gen_batch: int = 8,
                    amp: bool = True, verbose: bool = True,
                    collect_badcases: bool = True) -> Dict:
    """在单个 checkpoint 上运行嵌套 NIAH，返回逐档 EM 与逐 pair 结果。"""
    ctxs = _normalize_ctx_lens(ctx_lens)
    n_samples = strict_int(n_samples, "n_samples")
    seed = strict_int(seed, "seed")
    max_new = strict_int(max_new, "max_new")
    gen_batch = strict_int(gen_batch, "gen_batch")
    growth_modes = _normalize_growth_modes(growth_modes)
    if mode not in ("synth", "corpus"):
        raise ValueError(f"mode 必须是 'synth'/'corpus'，收到 {mode!r}")
    if n_samples < 1:
        raise ValueError(f"n_samples 必须 >= 1，收到 {n_samples}")
    if max_new < 1:
        raise ValueError(f"max_new 必须 >= 1，收到 {max_new}")
    if gen_batch < 1:
        raise ValueError(f"gen_batch 必须 >= 1，收到 {gen_batch}")
    limit = model.block_size - max_new - 1
    if limit < 1:
        raise ValueError(
            f"max_new={max_new} 过大：block_size={model.block_size} "
            "无法为 prompt 留出位置")
    too_long = [c for c in ctxs if c > limit]
    if too_long:
        raise ValueError(
            f"ctx_lens={too_long} 超过 block_size={model.block_size} - "
            f"max_new={max_new} - 1 的可容纳上限 {limit}；"
            "嵌套设计要求精确长度，不能静默截断")

    samples = []
    skipped: List[str] = []
    for i in range(n_samples):
        try:
            sample = build_nested_sample(
                tok, ctxs, seed=seed, sample_index=i,
                growth_modes=growth_modes, mode=mode,
                corpus_sents=corpus_sents, base_ctx=base_ctx)
        except ContextTooSmall as e:
            skipped.append(f"sample {i}: {e}")
            continue
        if not sample["ok"]:
            skipped.extend(sample["skipped"])
            continue
        samples.append(sample)

    use_cuda = device.startswith("cuda")
    amp_dt = amp_dtype_for(device, bool(amp))
    amp_ctx = amp_autocast(device, amp_dt is not None)
    results: Dict[str, Dict] = {g: {} for g in growth_modes}
    badcases: List[Dict] = []
    for growth in growth_modes:
        if not samples:
            break
        for ctx in ctxs:
            examples = []
            for sample in samples:
                ex = next((e for e in sample["growth"][growth]
                           if e["meta"]["nominal_ctx"] == ctx), None)
                if ex is None:
                    raise AssertionError(
                        f"pair={sample['pair_id']} growth={growth} 缺少 ctx={ctx}")
                examples.append(ex)
            outcomes: List[Dict] = []
            for s0 in range(0, len(examples), gen_batch):
                part = examples[s0:s0 + gen_batch]
                length = len(part[0]["ids"])
                for ex in part:
                    if len(ex["ids"]) != length:
                        raise AssertionError("同格样例长度不一致，无法批量解码")
                ids = _device_ids([e["ids"] for e in part], device)
                with amp_ctx:
                    out_ids = model.generate(
                        ids, max_new_tokens=max_new)
                out_l = out_ids[:, length:].tolist()
                for row, ex in enumerate(part):
                    pred_text = tok.decode(out_l[row])
                    sc = score_prediction(
                        "niah_corpus" if mode == "corpus" else "niah_synth",
                        ex["answer"], pred_text)
                    meta = ex["meta"]
                    outcomes.append({
                        "pair_id": ex["pair_id"],
                        "sample_id": ex["sample_id"],
                        "em": float(sc["em"]),
                        "key": meta["key"],
                        "value": meta["value"],
                        "actual_tokens": meta["actual_tokens"],
                        "needle_pos": meta["needle_pos"],
                        "evidence_distance_tokens":
                            meta["evidence_distance_tokens"],
                        "depth": meta["depth"],
                        "core_context_sha256": meta["core_context_sha256"],
                    })
                    if collect_badcases and sc["em"] < 1.0:
                        prompt_text = tok.decode(ex["ids"])
                        badcases.append({
                            "sample_id": ex["sample_id"],
                            "pair_id": ex["pair_id"],
                            "growth": growth,
                            "ctx": ctx,
                            "task": "niah_nested",
                            "answer": ex["answer"],
                            "pred": pred_text.strip(),
                            "key": meta["key"],
                            "needle_excerpt": _needle_excerpt(tok, ex),
                            "answer_in_prompt": ex["answer"] in prompt_text,
                            "context_numbers": standalone_numeric_tokens(
                                prompt_text),
                        })
            n = len(outcomes)
            k = sum(o["em"] for o in outcomes)
            em = (k / n) if n else None
            ci = wilson_score_ci(int(round(k)), n) if n else None
            first = examples[0]["meta"] if examples else {}
            results[growth][str(ctx)] = {
                "n": n,
                "em": em,
                "em_ci95_wilson": list(ci) if ci is not None else None,
                "actual_tokens": ctx,
                "needle_pos": first.get("needle_pos"),
                "evidence_distance_tokens": first.get(
                    "evidence_distance_tokens"),
                "filler_stream_sha256": first.get("filler_stream_sha256"),
                "outcomes": outcomes,
            }
            if verbose:
                print(
                    f"[nested] {growth}@{ctx}: em={em:.3f} (n={n})"
                    if em is not None else
                    f"[nested] {growth}@{ctx}: 无有效样例")
    design = {
        "ctx_lens": ctxs,
        "base_ctx": ctxs[0] if base_ctx is None else strict_int(
            base_ctx, "base_ctx"),
        "growth_modes": list(growth_modes),
        "n_samples": n_samples,
        "seed": seed,
        "max_new": max_new,
        "mode": mode,
        "gen_batch": gen_batch,
        "amp": bool(amp_dt is not None),
        "paired_across_lengths": True,
        "pairing_key": "pair_id",
        "core_window": CORE_WINDOW,
        "prompt_layout": "prefix_filler + needle + suffix_filler + question",
        "growth_semantics": {
            "suffix": "fixed_prefix_grow_suffix",
            "prefix": "grow_prefix_fixed_suffix",
            "split": "grow_both_sides",
        },
    }
    return {"task": "niah_nested", "design": design, "results": results,
            "_skipped": skipped, "_badcases": badcases}


def main() -> None:
    import argparse

    from .data import load_tokenizer
    from .model import load_model
    from .preflight import (allow_protocol_violations,
                            data_protocol_preflight)
    from .repro import load_provenance, sha256_file
    from .ruler import load_corpus_sents_with_source
    from .utils import enable_tf32, get_device, set_seed

    p = argparse.ArgumentParser(
        description="嵌套上下文配对 NIAH：同一证据、同一填充流、精确长度")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--ctx_lens", default=None, help="逗号分隔；至少两档")
    p.add_argument("--n_samples", type=int, default=None)
    p.add_argument("--growth_modes", default=",".join(DEFAULT_GROWTH_MODES))
    p.add_argument("--base_ctx", type=int, default=None,
                   help="基线档位；必须等于最短 ctx，默认最短档")
    p.add_argument("--mode", default="synth", choices=["synth", "corpus"])
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--max_new", type=int, default=None)
    p.add_argument("--gen_batch", type=int, default=None)
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--allow_test_corpus", action="store_true",
                   help="允许 niah_corpus 干扰语句取自 test.txt（写访问账本）")
    p.add_argument("--allow_data_protocol_violations", action="store_true",
                   help="即使 data_report 标记泄露也继续（结果不得用于正式结论）")
    args = p.parse_args()

    ctx_lens = None
    if args.ctx_lens:
        ctx_lens = [strict_int(x, "--ctx_lens")
                    for x in args.ctx_lens.split(",")]
    growth_modes = _normalize_growth_modes(
        tuple(x.strip() for x in args.growth_modes.split(",") if x.strip()))

    data_preflight = data_protocol_preflight(
        args.data_dir,
        allow_override=allow_protocol_violations(
            args.allow_data_protocol_violations))
    if not data_preflight["report_present"]:
        print("[nested] 警告：数据目录缺少 data_report.json，"
              "无法执行评测前防泄漏闸门")
    elif data_preflight["critical_violations"]:
        print(f"[nested] ⚠ 数据协议自检发现 "
              f"{len(data_preflight['critical_violations'])} 个严重问题，"
              "但已显式放行（结果不得用于正式结论）")

    device = get_device(args.device)
    enable_tf32(device)
    model, ck = load_model(args.ckpt, device=device)
    ckpt_sha256 = sha256_file(args.ckpt)
    tok = load_tokenizer(args.data_dir)
    seed_raw = ck.get("seed")
    seed = (None if seed_raw is None
            else strict_int(seed_raw, "checkpoint seed"))
    set_seed(0 if seed is None else seed)

    cfg_path = os.path.join(os.path.dirname(os.path.abspath(args.ckpt)),
                            "config_used.json")
    cfg = {}
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, encoding="utf-8") as f:
                cfg = json.load(f)
        except (OSError, ValueError):
            cfg = {}
        if not isinstance(cfg, dict):
            cfg = {}
    rl = cfg.get("ruler")
    rl = rl if isinstance(rl, dict) else {}

    def _cfg_int(key, default):
        if key not in rl:
            return default
        return strict_int(rl[key], f"ruler.{key}")

    if ctx_lens is None:
        raw = rl.get("ctx_lens")
        if not isinstance(raw, (list, tuple)):
            raise ValueError(
                "未提供 --ctx_lens 且 checkpoint 配置缺少 ruler.ctx_lens 列表")
        ctx_lens = [strict_int(x, "ruler.ctx_lens") for x in raw]
    n_samples = (args.n_samples if args.n_samples is not None
                 else _cfg_int("n_samples", 30))
    sample_seed = (args.seed if args.seed is not None
                   else _cfg_int("seed", 1234))
    max_new = (args.max_new if args.max_new is not None
               else _cfg_int("max_new", 12))
    gen_batch = (args.gen_batch if args.gen_batch is not None
                 else _cfg_int("gen_batch", 8))
    amp = config_bool(cfg.get("amp", True), "amp") and not args.no_amp

    corpus_sents, corpus_source = (None, None)
    if args.mode == "corpus":
        corpus_sents, corpus_source = load_corpus_sents_with_source(
            args.data_dir, allow_test=bool(args.allow_test_corpus))
        if corpus_source == "test":
            if record_corpus_test_access(
                    args.data_dir, corpus_source, ckpt_sha256=ckpt_sha256,
                    variant=ck.get("variant"), seed=seed, stage="nested"):
                print("[nested] ⚠ 干扰语料取自 test.txt（尘封 split）："
                      "已写入 test 访问账本")
            else:
                print("[nested] ⚠ 干扰语料取自 test.txt 且访问账本写入失败")
        elif corpus_source == "train":
            print("[nested] ⚠ 干扰语料回退到 train.txt：评测上下文与训练语料"
                  "重叠，分数不可与 held-out 运行直接比较")
        elif corpus_source is None:
            print("[nested] ⚠ 未找到可用干扰语料；niah_corpus 将无有效样例")
        else:
            print(f"[nested] niah_corpus 干扰语料来源: {corpus_source}.txt")

    result = run_nested_niah(
        model, tok, device, ctx_lens=ctx_lens, n_samples=n_samples,
        seed=sample_seed, max_new=max_new, growth_modes=growth_modes,
        mode=args.mode, corpus_sents=corpus_sents, base_ctx=args.base_ctx,
        gen_batch=gen_batch, amp=amp)
    badcases = result.pop("_badcases")
    skipped = result.pop("_skipped")
    if any(result["results"].get(g) for g in result["results"]):
        analysis = analyze_nested(result)
    else:
        analysis = {
            "task": "niah_nested_analysis",
            "scope": "exploratory_paired_design",
            "checks": {},
            "error": "no_valid_samples",
            "skipped": skipped,
        }
    payload = {
        "variant": ck.get("variant", model.variant),
        "seed": seed,
        "sample_seed": sample_seed,
        "task": "niah_nested",
        "design": result["design"],
        "results": result["results"],
        "analysis": analysis,
        "badcases": badcases,
        "skipped": skipped,
        "max_new": max_new,
        "gen_batch": int(gen_batch),
        "amp": bool(amp),
        "corpus_source": corpus_source,
        "corpus_held_out": corpus_source in ("val", "test"),
        "ckpt": os.path.abspath(args.ckpt),
        "ckpt_sha256": ckpt_sha256,
        "data_protocol_preflight": data_preflight,
    }
    payload.update(load_provenance(
        os.path.dirname(os.path.abspath(args.ckpt)), args.data_dir,
        device=device))
    save_json(payload, args.out)
    print(f"[nested] 完成 -> {args.out}")
    checks = analysis["checks"]
    print(f"[nested] 设计自检: {checks}")
    if badcases:
        print(f"[nested] 预测错误样本 {len(badcases)} 条已随结果落盘")


if __name__ == "__main__":
    main()
