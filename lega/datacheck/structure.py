"""文档边界结构与跨文档窗口审计。

数据处理有一个隐含前提：train/val/test 的 token 流由若干“独立文本单元”
（WikiText 的文章，或无标题语料的空行段落）顺序拼接而成，而训练/评测窗口
按固定长度在这条拼接流上滑动。窗口因此可能同时覆盖两个文档——模型被要求
“接着上一篇文章写下一篇”，评测 PPL 里也混入了这类跨文档跳转的损失。

这与本项目的构念直接相关：LEGA 主张在注意力头找不到可靠信息时减少残差流
里的无关内容。若语料主要由跨文档窗口构成，“无关内容”就被数据拼接本身大量
注入，门控收益可能来自拼接伪影而不是机制本身。因此本模块提供可复核的证据：

* ``split_documents``：把文本切成连续的文档段（标题优先，否则空行段落）；
* ``document_stats``：文档长度分布；
* ``window_boundary_stats``：统计跨文档窗口比例（显式窗口起点，或训练采样
  器“均匀随机起点”口径下的精确期望）；
* ``doc_aligned_windows``：挑选完全落在单个文档内的窗口（含标签 token 也
  在同一文档内），用于“文档内 PPL”这一独立口径。

所有函数只返回 JSON 原生类型，可直接写入 data_report/eval.json。
"""

from __future__ import annotations

import bisect
import statistics

from ._shared import _install as _shared_install

_shared_install(globals())

# 文档切分协议标识：写进 meta/data_report，作为缓存身份的一部分。
DOC_STRUCTURE_PROTOCOL = "heading_or_paragraph_v1"

# 文档切分口径（group_unit）：
#   heading_document   —— WikiText 风格标题行开启新文档；
#   blank_line_paragraph —— 无标题语料按空行分隔的段落块；
#   single_segment     —— 未检测到任何文档结构（整段视为一个单元）。
GROUP_UNIT_HEADING = "heading_document"
GROUP_UNIT_PARAGRAPH = "blank_line_paragraph"
GROUP_UNIT_SINGLE = "single_segment"

# data_report 里用于展示“跨文档窗口比例随上下文长度变化”的标准档位。
# 这张表只依赖文档长度分布，与本次实验的 block_size 无关；本次实验实际
# ctx 下的数值由 eval.json / train 侧记录给出。
REPORT_CTX_GRID = (128, 256, 512, 1024, 2048, 4096)


def _line_spans(text: str):
    """逐行返回 (起始偏移, 结束偏移)；与 text.split("\\n") 的序列一致。"""
    n = len(text)
    pos = 0
    while True:
        nl = text.find("\n", pos)
        if nl < 0:
            yield pos, n
            return
        yield pos, nl
        pos = nl + 1


def split_documents(text: str) -> dict:
    """把文本切成连续的文档段（字符偏移，半开区间）。

    规则（确定性、可复核）：

    * 文本含 WikiText 风格标题行（``= 标题 =``）时，标题行开启新文档；
      第一个标题之前的内容单独成段（前言）。
    * 无标题时按空行分隔的段落块切分；每段包含其后的空行，因此各段连续
      覆盖全文——拼接各段文本与原文逐字节相同。

    返回 {"segments": [(start, end), ...], "group_unit": str, "n_documents": int}。
    """
    if not isinstance(text, str) or not text:
        return {"segments": [], "group_unit": GROUP_UNIT_SINGLE, "n_documents": 0}
    heading_starts = []
    for start, end in _line_spans(text):
        line = text[start:end].strip()
        if line and _HEADING_RE.match(line):
            heading_starts.append(start)
    if heading_starts:
        starts = [0]
        for s in heading_starts:
            if s > starts[-1]:
                starts.append(s)
        unit = GROUP_UNIT_HEADING
    else:
        para_starts = []
        for start, end in _line_spans(text):
            if text[start:end].strip():
                para_starts.append(start)
        if not para_starts:
            return {
                "segments": [(0, len(text))],
                "group_unit": GROUP_UNIT_SINGLE,
                "n_documents": 1,
            }
        # 首段从 0 开始（含前导空行），避免产生只含空白的“伪文档”。
        starts = [0]
        for s in para_starts[1:]:
            if s > starts[-1]:
                starts.append(s)
        unit = GROUP_UNIT_PARAGRAPH if len(starts) > 1 else GROUP_UNIT_SINGLE
    bounds = starts + [len(text)]
    segments = [(bounds[i], bounds[i + 1]) for i in range(len(starts))]
    segments = [(s, e) for s, e in segments if e > s]
    return {
        "segments": segments,
        "group_unit": unit,
        "n_documents": len(segments),
    }


def _distribution(values) -> dict:
    """长度分布（min/median/mean/p90/max）；空序列返回 None 值而不是 0。"""
    vals = [int(v) for v in values]
    if not vals:
        return {
            "min": None,
            "median": None,
            "mean": None,
            "p90": None,
            "max": None,
            "sum": 0,
            "count": 0,
        }
    ordered = sorted(vals)
    n = len(ordered)
    p90_idx = min(n - 1, int(-(-9 * n // 10)) - 1)
    return {
        "min": int(ordered[0]),
        "median": float(statistics.median(ordered)),
        "mean": float(sum(ordered) / n),
        "p90": int(ordered[p90_idx]),
        "max": int(ordered[-1]),
        "sum": int(sum(ordered)),
        "count": int(n),
    }


def document_stats(text: str, segments, group_unit: str = None) -> dict:
    """文档长度分布（字符数与行数）。"""
    segments = list(segments or [])
    chars = [int(e) - int(s) for s, e in segments]
    lines = []
    for s, e in segments:
        s, e = int(s), int(e)
        if e <= s:
            lines.append(0)
            continue
        n_newline = text.count("\n", s, e)
        tail_partial = 0 if text[e - 1] == "\n" else 1
        lines.append(n_newline + tail_partial)
    return {
        "n_documents": len(segments),
        "group_unit": group_unit or GROUP_UNIT_SINGLE,
        "chars_total": len(text),
        "chars_per_document": _distribution(chars),
        "lines_per_document": _distribution(lines),
        "empty_documents": int(sum(1 for c in chars if c <= 0)),
    }


def token_boundary_positions(doc_token_counts) -> list:
    """文档 token 计数 → 内部文档边界位置（新文档首个 token 的下标）。

    长度为 0 的文档不产生边界（其位置与相邻文档重合），因此边界严格递增。
    """
    pos = 0
    bounds = []
    counts = [int(c) for c in (doc_token_counts or [])]
    for c in counts[:-1]:
        pos += c
        if pos > 0 and (not bounds or pos > bounds[-1]):
            bounds.append(pos)
    return bounds


def _merge_intervals(intervals):
    """合并区间并返回总覆盖长度（相邻/重叠区间只算一次）。"""
    if not intervals:
        return 0
    merged = []
    for lo, hi in sorted(intervals):
        if hi < lo:
            continue
        if merged and lo <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    return sum(hi - lo + 1 for lo, hi in merged)


def window_boundary_stats(boundaries, n_tokens: int, ctx: int, starts=None) -> dict:
    """跨文档窗口统计。

    窗口输入是 token ``[s, s+ctx-1]``、标签是 ``[s+1, s+ctx]``：只要存在
    文档边界 b 落在 ``[s+1, s+ctx]``，该窗口就跨越了文档边界——包括“标签
    正好是下一文档第一个 token”的情形（模型在预测新文档的开头）。

    ``starts=None`` 时按训练采样器的口径计算期望比例：起点在
    ``[0, n_tokens-ctx-1]`` 上均匀随机，跨文档起点集合是各边界对应区间的
    并集，因此该比例是精确期望而非近似。
    """
    ctx = strict_int(ctx, "ctx")
    if ctx < 1:
        raise ValueError(f"ctx 必须 >= 1，收到 {ctx!r}")
    n_tokens = strict_int(n_tokens, "n_tokens")
    if n_tokens < 0:
        raise ValueError(f"n_tokens 必须 >= 0，收到 {n_tokens!r}")
    bounds = sorted({int(b) for b in (boundaries or []) if 0 < int(b) < n_tokens})
    base = {
        "available": True,
        "ctx": int(ctx),
        "n_tokens": int(n_tokens),
        "n_boundaries": len(bounds),
    }
    if starts is None:
        n_starts = n_tokens - ctx
        if n_starts <= 0:
            return {
                **base,
                "mode": "uniform_random_expectation",
                "windows_total": 0,
                "windows_crossing_documents": 0,
                "crossing_window_fraction": None,
                "reason": f"token 数 {n_tokens} 不足以组成 ctx={ctx} 的窗口",
            }
        intervals = []
        for b in bounds:
            lo = max(0, b - ctx)
            hi = min(b - 1, n_starts - 1)
            if hi >= lo:
                intervals.append((lo, hi))
        crossing = _merge_intervals(intervals)
        return {
            **base,
            "mode": "uniform_random_expectation",
            "windows_total": int(n_starts),
            "windows_crossing_documents": int(crossing),
            "crossing_window_fraction": float(crossing / n_starts),
        }
    starts = sorted({int(s) for s in starts})
    crossing = 0
    for s in starts:
        idx = bisect.bisect_right(bounds, s)
        if idx < len(bounds) and bounds[idx] <= s + ctx:
            crossing += 1
    total = len(starts)
    return {
        **base,
        "mode": "explicit_starts",
        "windows_total": int(total),
        "windows_crossing_documents": int(crossing),
        "crossing_window_fraction": float(crossing / total) if total else None,
    }


def grid_window_starts(n_tokens: int, ctx: int) -> list:
    """评测协议的非重叠窗口网格（offset=0），与训练内评测口径一致。"""
    ctx = strict_int(ctx, "ctx")
    if ctx < 1:
        raise ValueError(f"ctx 必须 >= 1，收到 {ctx!r}")
    n_tokens = strict_int(n_tokens, "n_tokens")
    n_windows = (n_tokens - 1) // ctx if n_tokens > 0 else 0
    return [k * ctx for k in range(max(0, n_windows))]


def doc_aligned_windows(doc_token_counts, ctx: int) -> dict:
    """挑选完全落在单个文档内的窗口起点。

    窗口输入 ``[s, s+ctx-1]``、标签 ``[s+1, s+ctx]``：标签也必须在同一文档
    内，因此文档至少需要 ``ctx+1`` 个 token。返回的 starts 是 token 下标，
    按文档顺序排列。
    """
    ctx = strict_int(ctx, "ctx")
    if ctx < 1:
        raise ValueError(f"ctx 必须 >= 1，收到 {ctx!r}")
    counts = [int(c) for c in (doc_token_counts or [])]
    starts = []
    pos = 0
    eligible_docs = 0
    tokens_in_eligible = 0
    for c in counts:
        if c > 0 and c >= ctx + 1:
            eligible_docs += 1
            tokens_in_eligible += c
            n_windows = (c - 1) // ctx
            for k in range(n_windows):
                starts.append(pos + k * ctx)
        pos += c
    total_tokens = pos
    tokens_covered = len(starts) * ctx
    return {
        "ctx": int(ctx),
        "starts": starts,
        "n_windows": len(starts),
        "n_documents_total": len(counts),
        "n_documents_eligible": eligible_docs,
        "tokens_total": int(total_tokens),
        "tokens_in_eligible_documents": int(tokens_in_eligible),
        "tokens_covered": int(tokens_covered),
        "token_coverage_fraction": (
            float(tokens_covered / total_tokens) if total_tokens else None
        ),
    }


def subsample_starts(starts, limit: int) -> list:
    """从起点序列里等间隔取不超过 limit 个（含首尾），口径与评测抽样一致。"""
    limit = strict_int(limit, "limit")
    if limit < 1:
        raise ValueError(f"limit 必须 >= 1，收到 {limit!r}")
    values = sorted({int(s) for s in (starts or [])})
    if len(values) <= limit:
        return values
    if limit == 1:
        return [values[len(values) // 2]]
    idx = [k * (len(values) - 1) // (limit - 1) for k in range(limit)]
    return [values[i] for i in idx]
