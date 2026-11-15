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
* ``group_disjointness_audit``：群组级 split 互斥性审计——把“按文档
  划分”的**声明**升级为可复核结论（整篇文档/段落重复、行被截断三类）；
* ``window_boundary_stats``：统计跨文档窗口比例（显式窗口起点，或训练采样
  器“均匀随机起点”口径下的精确期望）；
* ``window_crossing_flags``：逐窗口给出“是否跨文档边界”的标志（与窗口
  列表一一对应），供“跨文档 vs 文档内”的效应分层使用；
* ``doc_aligned_windows``：挑选完全落在单个文档内的窗口（含标签 token 也
  在同一文档内），用于“文档内 PPL”这一独立口径；
* ``doc_aligned_sampling_plan``：构造“训练窗口不跨文档”的采样计划（文档
  权重 ∝ 文档 token 数，与默认拼接流采样保持同一份文档混合比例），把
  “跨文档跳转”从不可控的拼接伪影变成**受控因子**。

除 ``doc_aligned_sampling_plan``（内部采样结构，含 numpy 数组）之外，所有
函数只返回 JSON 原生类型，可直接写入 data_report/eval.json。
"""

from __future__ import annotations

import bisect
import hashlib
import statistics
import unicodedata

import numpy as np

from ._shared import _install as _shared_install
# Reproducibility: Set random seeds
import random
import numpy as np
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
try:
    import torch
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
except ImportError:
    pass

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
        "mean": float(sum(ordered) / max(n, 1)),
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


# ---------------------------------------------------------------- 群组级互斥性

# 群组互斥性审计协议标识：写进 data_report.split_protocol，作为“划分是否
# 真的按文档/段落切开”的可复核证据的版本号。
GROUP_DISJOINTNESS_PROTOCOL = "group_disjointness_v1"

# 行前缀探针长度（字符）：用于发现“同一行在两个 split 里被截断”的情形。
# 逐行逐字节比较看不到共享（两侧行内容不同），但只要截断点在该长度之后，
# 两侧“≤该长度且按词边界回退”的前缀必然相同——这是判定“上游切分真的落在
# 文档边界上”的关键证据：文档被从中间切开时，行内容不再相等。
# 取 80 字符而不是更短：短前缀在真实语料里极易偶然相同（固定句式、
# 模板化文本），会把告警变成噪声；80 字符仍覆盖绝大多数自然语言行。
LINE_PREFIX_PROBE_CHARS = 80


def normalize_group_text(text: str) -> str:
    """群组/段落比较键：NFKC + casefold + 空白折叠。

    与近似重复筛查（``text._near_dup_normalize``）同一口径：只做确定性的
    等价归一化，不做同义词/词干替换，避免把语义不同的文本判成同一群组。
    """
    if not isinstance(text, str) or not text:
        return ""
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def group_digest(normalized: str) -> str:
    """归一化文本的短摘要（16 hex）。只作比较键，不承担完整性证明。"""
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def _word_boundary_prefix(norm: str, limit: int) -> str:
    """``norm`` 中长度不超过 ``limit`` 的最长词边界前缀。

    词边界即空格位置：归一化后词与词之间恰好一个空格。截断到某个词之后
    的行与完整行的“≤limit 词边界前缀”相同，因此该前缀可识别“同一行的
    同一前缀在两个 split 里同时出现”（整行被复制）。找不到词边界（首个词
    就超过 limit）时返回空串。

    注意：它**不能**识别“文档被从行中间切开”的情形——切开后的前半段与
    后半段是同一行的不同片段，前缀摘要并不相同（前半段的前缀属于原行，
    后半段的前缀属于原行的后半部分）。行中切开的证据由两处给出：本地
    划分路径在切点处的精确自检（``split_protocol.lines_cross_cut``），
    以及本模块的边界片段诊断（``boundary_continuity``）。
    """
    if not norm:
        return ""
    if len(norm) <= limit:
        return norm
    cut = norm.rfind(" ", 0, limit + 1)
    if cut <= 0:
        return ""
    return norm[:cut]


def _line_prefix_digests(text: str, prefix_chars: int):
    """逐行的词边界前缀摘要集合（发现被整行复制的共享行）。

    只索引长度不小于 ``prefix_chars`` 的行：更短的行在真实语料里极易
    偶然相同（标题、固定用语），纳入只会制造噪声。返回集合里的每个摘要
    都对应“至少 ``prefix_chars`` 个字符的共同前缀”。
    """
    out = set()
    if not text:
        return out
    for start, end in _line_spans(text):
        raw = text[start:end]
        if not raw.strip():
            continue
        norm = normalize_group_text(raw)
        if len(norm) < prefix_chars:
            continue
        pref = _word_boundary_prefix(norm, prefix_chars)
        if pref:
            out.add(group_digest(pref))
    return out


def _first_word(text: str) -> str:
    """首个空白分隔词（跳过开头的空行）。"""
    for line in (text or "").splitlines():
        stripped = line.strip()
        if stripped:
            parts = stripped.split()
            return parts[0] if parts else ""
    return ""


def _last_word(text: str) -> str:
    """末个空白分隔词（跳过结尾的空行）。"""
    lines = [ln for ln in (text or "").splitlines() if ln.strip()]
    if not lines:
        return ""
    parts = lines[-1].split()
    return parts[-1] if parts else ""


def boundary_continuity(texts: dict) -> dict:
    """相邻 split 的边界片段诊断（时间序 train→val→test）。

    文档被从**行中间**切开时，群组比较与逐行比较都看不出来（两侧的行内容
    不同），但切点两侧会留下半个词。这里把切点两侧的词片段落盘：

    * ``tail_word`` / ``head_word``：切点两侧的词片段；
    * ``mid_word_cut_suspected``：切点疑似落在词中间——两侧的词拼在一起
      既不等于前者的完整词、也不等于后者的完整词，且“两词相邻”在前一个
      split 末尾附近没有出现过。

    这是**启发式诊断**而非判定：没有原始语料时无法区分“真的切开”与
    “原文本来就以半词开头”。本地划分路径另有精确自检
    （``split_protocol.lines_cross_cut``）。
    """
    texts = texts or {}
    order = [s for s in ("train", "val", "test") if s in texts]
    if len(order) < 2:
        order = sorted(texts)
    pairs = []
    for i, a in enumerate(order[:-1]):
        b = order[i + 1]
        ta = texts.get(a) or ""
        tb = texts.get(b) or ""
        tail_word = _last_word(ta)
        head_word = _first_word(tb)
        joined = f"{tail_word} {head_word}"
        nearby = False
        if tail_word and head_word:
            window = " ".join(ta.split()[-2000:])
            nearby = joined in window
        suspected = bool(
            tail_word
            and head_word
            and tail_word != head_word
            and not nearby
            and ta.rstrip(" \t").endswith(tail_word[-1])
            and tb.lstrip(" \t").startswith(head_word[0])
        )
        pairs.append(
            {
                "pair": _pair_key(a, b),
                "tail_word": tail_word,
                "head_word": head_word,
                "pair_appears_adjacent_nearby": bool(nearby),
                "mid_word_cut_suspected": bool(suspected),
            }
        )
    return {
        "order": "->".join(order),
        "pairs": pairs,
        "n_suspected": int(sum(1 for p in pairs if p["mid_word_cut_suspected"])),
        "note": (
            "切点两侧的词片段诊断：疑似在词中间切开时给出样例供人工复核；"
            "没有原始语料时无法完全排除原文本身以半词开头的情况"
        ),
    }


def _paragraph_spans(text: str, start: int, end: int):
    """群组内部的空行分段（字符偏移，半开区间）。

    段落是“同一文档被切到两个 split”时最细的可核对单位：一个段落整体
    出现在两个 split 里，说明上游切分没有落在文档边界上。
    """
    spans = []
    cur = int(start)
    pos = int(start)
    limit = int(end)
    while pos < limit:
        nl = text.find("\n", pos, limit)
        line_end = limit if nl < 0 else nl
        if not text[pos:line_end].strip():
            if cur < pos:
                spans.append((cur, pos))
            cur = line_end + 1 if nl >= 0 else limit
        pos = line_end + 1 if nl >= 0 else limit
    if cur < limit:
        spans.append((cur, limit))
    return spans


def _pair_key(a: str, b: str) -> str:
    return f"{a}|{b}"


def group_disjointness_audit(
    texts: dict,
    min_chars: int = 40,
    prefix_chars: int = LINE_PREFIX_PROBE_CHARS,
) -> dict:
    """群组级 split 互斥性审计。

    “官方 split 按文档划分”“时序切分吸附到文档边界”这类说法此前只是
    **声明**：逐行比较只能发现两侧行内容完全相同的情形，文档若被从中间
    切开（行被截断），两侧行不再相等，声明与实际内容就脱节了。本审计把
    声明升级为可复核的结论，分三层：

    * **群组级**：整篇文档（标题模式）或整段（空行段落模式）的内容在两个
      split 里同时出现——同一份文档被复制进两个 split；
    * **段落级**：文档内部的某个段落同时出现在两个 split——文档被从段落
      边界切开；
    * **行级**：同一行（词边界前缀相同）在两个 split 里同时出现——整行被
      复制。该层可能有偶然命中（固定用语、模板句），因此只作告警证据并
      给出样例，不单独作为硬失败依据。

    另外给出 ``boundary_continuity``：相邻 split 的切点两侧词片段，用于
    人工复核“文档被从行中间切开”的情形。

    返回值只含 JSON 原生类型，可直接写进 data_report。
    """
    min_chars = int(min_chars)
    prefix_chars = int(prefix_chars)
    if min_chars < 1:
        raise ValueError(f"min_chars 必须 >= 1，收到 {min_chars!r}")
    if prefix_chars < 1:
        raise ValueError(f"prefix_chars 必须 >= 1，收到 {prefix_chars!r}")
    splits = [s for s in sorted(texts or {}) if isinstance((texts or {}).get(s), str)]
    per_split = {}
    group_index = {}
    paragraph_index = {}
    prefix_index = {}
    for split in splits:
        text = texts.get(split) or ""
        info = split_documents(text)
        segments = info["segments"]
        group_unit = info["group_unit"]
        n_groups = len(segments)
        n_groups_eligible = 0
        n_paragraphs_eligible = 0
        group_hashes = set()
        paragraph_hashes = set()
        for gi, (gs, ge) in enumerate(segments):
            norm = normalize_group_text(text[gs:ge])
            if len(norm) >= min_chars:
                n_groups_eligible += 1
                digest = group_digest(norm)
                group_hashes.add(digest)
                where = group_index.setdefault(digest, {})
                where[split] = where.get(split, 0) + 1
            for ps, pe in _paragraph_spans(text, gs, ge):
                pnorm = normalize_group_text(text[ps:pe])
                if len(pnorm) >= min_chars:
                    n_paragraphs_eligible += 1
                    digest = group_digest(pnorm)
                    paragraph_hashes.add(digest)
                    where = paragraph_index.setdefault(digest, {})
                    where[split] = where.get(split, 0) + 1
        prefixes = _line_prefix_digests(text, prefix_chars)
        for digest in prefixes:
            where = prefix_index.setdefault(digest, {})
            where[split] = where.get(split, 0) + 1
        per_split[split] = {
            "group_unit": group_unit,
            "n_groups": int(n_groups),
            "n_groups_eligible": int(n_groups_eligible),
            "n_paragraphs_eligible": int(n_paragraphs_eligible),
            "n_line_prefixes": int(len(prefixes)),
        }
    pairs = {}
    for i, a in enumerate(splits):
        for b in splits[i + 1 :]:
            shared_groups = sorted(
                d for d, where in group_index.items() if a in where and b in where
            )
            shared_paragraphs = sorted(
                d for d, where in paragraph_index.items() if a in where and b in where
            )
            shared_prefixes = sorted(
                d for d, where in prefix_index.items() if a in where and b in where
            )
            pairs[_pair_key(a, b)] = {
                "shared_groups": int(len(shared_groups)),
                "shared_groups_examples": shared_groups[:5],
                "shared_paragraphs": int(len(shared_paragraphs)),
                "shared_paragraphs_examples": shared_paragraphs[:5],
                "shared_line_prefixes": int(len(shared_prefixes)),
                "shared_line_prefixes_examples": shared_prefixes[:5],
            }
    groups_disjoint = all(p["shared_groups"] == 0 for p in pairs.values())
    paragraphs_disjoint = all(p["shared_paragraphs"] == 0 for p in pairs.values())
    prefixes_disjoint = all(p["shared_line_prefixes"] == 0 for p in pairs.values())
    problems = []
    if not groups_disjoint:
        problems.append("shared_groups")
    if not paragraphs_disjoint:
        problems.append("shared_paragraphs")
    if not prefixes_disjoint:
        problems.append("shared_line_prefixes")
    return {
        "protocol": GROUP_DISJOINTNESS_PROTOCOL,
        "min_chars": int(min_chars),
        "prefix_chars": int(prefix_chars),
        "splits": per_split,
        "pairs": pairs,
        "groups_disjoint": bool(groups_disjoint),
        "paragraphs_disjoint": bool(paragraphs_disjoint),
        "line_prefixes_disjoint": bool(prefixes_disjoint),
        "problems": problems,
        "verdict": "disjoint" if not problems else "shared:" + ",".join(problems),
        "boundary_continuity": boundary_continuity(texts),
        "note": (
            "群组级/段落级共享是文档被复制或被从文档边界切开的直接证据；"
            "行级共享提示整行被复制，可能含偶然命中，只作告警并给出样例；"
            "boundary_continuity 给出切点两侧词片段供人工复核行中切开的情形"
        ),
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
            "crossing_window_fraction": float(crossing / max(n_starts, 1)),
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
        "crossing_window_fraction": float(crossing / max(total, 1)) if total else None,
    }


def window_crossing_flags(boundaries, n_tokens: int, ctx: int, starts) -> list:
    """逐窗口返回是否跨文档边界（与 ``starts`` 同序、同长）。

    与 :func:`window_boundary_stats` 的 ``explicit_starts`` 口径一致：窗口
    输入是 token ``[s, s+ctx-1]``、标签是 ``[s+1, s+ctx]``；只要存在文档
    边界 ``b`` 落在 ``[s+1, s+ctx]``，该窗口就跨越了文档边界。

    区别在于这里**不去重也不排序**：输出与调用方给出的窗口列表
    （``window_losses``/``window_indices`` 的顺序）一一对应，因此可以把
    跨文档标志直接绑定到逐窗口损失，做“跨文档 vs 文档内”的分层分解。
    起点越界（``s<0`` 或 ``s+ctx >= n_tokens``，标签 token 不存在）直接
    报错，不静默产生错误的边界判断。
    """
    ctx = strict_int(ctx, "ctx")
    if ctx < 1:
        raise ValueError(f"ctx 必须 >= 1，收到 {ctx!r}")
    n_tokens = strict_int(n_tokens, "n_tokens")
    if n_tokens < 0:
        raise ValueError(f"n_tokens 必须 >= 0，收到 {n_tokens!r}")
    bounds = sorted({int(b) for b in (boundaries or []) if 0 < int(b) < n_tokens})
    flags = []
    max_start = n_tokens - ctx - 1
    for raw in starts if starts is not None else []:
        s = strict_int(raw, "start")
        if s < 0 or s > max_start:
            raise ValueError(
                f"窗口起点必须落在 [0, {max_start}] 内（n_tokens={n_tokens}, "
                f"ctx={ctx}），收到 {s}"
            )
        idx = bisect.bisect_right(bounds, s)
        flags.append(bool(idx < len(bounds) and bounds[idx] <= s + ctx))
    return flags


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
            float(tokens_covered / max(total_tokens, 1)) if total_tokens else None
        ),
    }


def doc_aligned_coverage(doc_token_counts, ctx: int) -> dict:
    """文档内窗口覆盖量（与 :func:`doc_aligned_windows` 同口径，不物化起点）。

    报告只需要“有多少个窗口完全落在单个文档内、覆盖多少 token”，不需要
    起点列表本身：大语料在 ctx=128 时起点可达百万级，写进 JSON 报告既
    臃肿又没有消费方。
    """
    ctx = strict_int(ctx, "ctx")
    if ctx < 1:
        raise ValueError(f"ctx 必须 >= 1，收到 {ctx!r}")
    counts = [int(c) for c in (doc_token_counts or [])]
    n_windows = 0
    eligible_docs = 0
    tokens_in_eligible = 0
    total_tokens = 0
    for c in counts:
        total_tokens += c
        if c > 0 and c >= ctx + 1:
            eligible_docs += 1
            tokens_in_eligible += c
            n_windows += (c - 1) // ctx
    tokens_covered = n_windows * ctx
    return {
        "ctx": int(ctx),
        "n_windows": int(n_windows),
        "n_documents_total": len(counts),
        "n_documents_eligible": int(eligible_docs),
        "tokens_total": int(total_tokens),
        "tokens_in_eligible_documents": int(tokens_in_eligible),
        "tokens_covered": int(tokens_covered),
        "token_coverage_fraction": (
            float(tokens_covered / max(total_tokens, 1)) if total_tokens else None
        ),
    }


def doc_aligned_sampling_plan(doc_token_counts, ctx: int) -> dict:
    """构造“训练窗口不跨文档”的采样计划（几何 + 覆盖率核算）。

    与 :func:`doc_aligned_windows` 的分工：后者产出**评测**用的不重叠窗口
    起点（每篇文档内按 ctx 步长铺开），本函数产出**训练**用的采样计划——
    训练窗口允许重叠，起点在合法区间内连续分布。

    采样口径（必须写进报告，否则“只禁止跨文档”这句话无法复核）：

    * 文档权重 ∝ 该文档的 token 数：与默认采样器（起点在拼接流上均匀随机）
      保持**同一份文档混合比例**，两者唯一的差别是窗口不再跨边界；
    * 文档内起点在 ``[doc_start, doc_start + n_tokens - ctx - 1]`` 上均匀
      （标签 token 也必须落在同一文档内，因此文档至少需要 ``ctx + 1`` 个
      token）；
    * 短于 ``ctx + 1`` 的文档无法承载任何合法窗口，直接不参与采样；它们的
      数量与 token 占比必须显式报告——这是该受控因子的代价，而不是可以
      忽略的实现细节。

    返回值含 numpy 数组（供采样器直接使用），**不是 JSON 安全**的；需要落盘
    的核算请用 :func:`doc_aligned_sampling_report`。
    """
    ctx = strict_int(ctx, "ctx")
    if ctx < 1:
        raise ValueError(f"ctx 必须 >= 1，收到 {ctx!r}")
    counts = []
    for c in (doc_token_counts or []):
        n = strict_int(c, "doc_token_counts")
        if n < 0:
            raise ValueError(f"文档 token 数不能为负，收到 {c!r}")
        counts.append(n)
    doc_starts = []
    doc_tokens = []
    doc_valid_starts = []
    total_tokens = 0
    dropped_docs = 0
    dropped_tokens = 0
    eligible_tokens = 0
    for c in counts:
        start = total_tokens
        total_tokens += c
        if c >= ctx + 1:
            doc_starts.append(start)
            doc_tokens.append(c)
            # 合法起点数 = 文档 token 数 - ctx（起点 + ctx 必须仍在文档内）
            doc_valid_starts.append(c - ctx)
            eligible_tokens += c
        else:
            dropped_docs += 1
            dropped_tokens += c
    starts_arr = np.asarray(doc_starts, dtype=np.int64)
    tokens_arr = np.asarray(doc_tokens, dtype=np.int64)
    valid_arr = np.asarray(doc_valid_starts, dtype=np.int64)
    cum_arr = np.cumsum(tokens_arr) if tokens_arr.size else np.zeros(0, np.int64)
    return {
        "ctx": int(ctx),
        "n_documents_total": len(counts),
        "n_documents_eligible": int(starts_arr.size),
        "n_documents_dropped": int(dropped_docs),
        "tokens_total": int(total_tokens),
        "tokens_in_eligible_documents": int(eligible_tokens),
        "tokens_in_dropped_documents": int(dropped_tokens),
        "token_coverage_fraction": (
            float(eligible_tokens / max(total_tokens, 1)) if total_tokens else None
        ),
        "n_valid_starts": int(valid_arr.sum()) if valid_arr.size else 0,
        "doc_starts": starts_arr,
        "doc_tokens": tokens_arr,
        "doc_valid_starts": valid_arr,
        "cum_tokens": cum_arr,
    }


def doc_aligned_sampling_report(plan: dict) -> dict:
    """把采样计划投影成 JSON 安全的核算字段（可写入报告/manifest）。"""
    plan = plan if isinstance(plan, dict) else {}
    keys = (
        "ctx",
        "n_documents_total",
        "n_documents_eligible",
        "n_documents_dropped",
        "tokens_total",
        "tokens_in_eligible_documents",
        "tokens_in_dropped_documents",
        "token_coverage_fraction",
        "n_valid_starts",
    )
    out = {k: plan.get(k) for k in keys}
    out["protocol"] = "doc_aligned_train_sampler_v1"
    out["note"] = (
        "训练窗口（含标签 token）严格落在单个文档内；文档权重 ∝ 文档 token "
        "数，与默认拼接流均匀采样保持同一份文档混合比例，唯一差别是禁止"
        "跨文档窗口。被丢弃的短文档（token 数 < ctx+1）无法承载合法窗口，"
        "其占比即本受控因子的代价。"
    )
    return out


def document_structure_report(
    text: str,
    segments,
    group_unit: str,
    doc_token_counts,
    ctx_grid=REPORT_CTX_GRID,
) -> dict:
    """单个 split 的文档结构审计（长度分布 + 跨文档窗口比例 + 文档内覆盖）。

    ``cross_document_windows`` 用训练采样器的口径（起点在 token 流上均匀
    随机）给出**期望**跨文档比例；``doc_aligned_windows`` 给出“完全落在
    单个文档内”的窗口数与 token 覆盖量。两者一起回答：这份语料的拼接结构
    在多大程度上把“跨文档跳转”注入了训练与评测窗口。
    """
    stats = document_stats(text, segments, group_unit)
    counts = [int(c) for c in (doc_token_counts or [])]
    n_tokens = int(sum(counts))
    boundaries = token_boundary_positions(counts)
    cross = {}
    aligned = {}
    for ctx in ctx_grid:
        ctx = int(ctx)
        key = str(ctx)
        cross[key] = window_boundary_stats(boundaries, n_tokens, ctx, starts=None)
        aligned[key] = doc_aligned_coverage(counts, ctx)
    return {
        "group_unit": group_unit,
        "n_documents": int(stats["n_documents"]),
        "chars_total": int(stats["chars_total"]),
        "tokens_total": n_tokens,
        "chars_per_document": stats["chars_per_document"],
        "lines_per_document": stats["lines_per_document"],
        "tokens_per_document": _distribution(counts),
        "n_boundaries": int(len(boundaries)),
        "cross_document_windows": cross,
        "doc_aligned_windows": aligned,
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
    idx = [k * (len(values) - 1) // max(limit - 1, 1) for k in range(limit)]
    return [values[i] for i in idx]
