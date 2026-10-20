"""数据卫生与防泄漏检查：划分协议、清洗、去污染、漂移筛查与测试集访问账本。

实验协议（本模块是实现而不是口号）:
  * 先划分 train/val/test，再做任何预处理；字符级词表只在 train 上拟合，
    val/test 只做 transform，未见字符统一映射到 <unk>（见 data.py）；
  * 连续文本（--local_txt）按时间先后切分：最早 80% 训练、其后 10% 验证、
    最新 10% 测试；切点对齐文档/段落/行边界，同一文档绝不跨 split；
  * 重复样本：与 val/test 精确重复的长行从 train 移除（去污染），val/test
    保持尘封不动；另用采样 64-gram 哈希筛查跨 split 的 token 级重叠；
  * token 流硬校验：整数 dtype、有限值、id ∈ [0, vocab)、uint16 可容纳；
  * 分布漂移：train 与 val/test 的一元 token 分布 TV/JS 散度与 OOV
    （默认全量精确计数，显式 max_samples 时才采样）；
  * 文本清洗：U+FFFD 替换字符、NUL 与非法控制符（标签噪声来源）；
  * 测试集访问账本：每次 test 评测都追加记录到 data_dir/.test_access_log.jsonl，
    使“反复在测试集上评测/调参”的行为可审计。

所有函数只返回 JSON 原生类型（int/float/str/bool/None），可直接写入报告。
"""
import hashlib
import json
import os
import re
import time
import unicodedata

import numpy as np

try:
    import fcntl
except ImportError:  # Windows 等无 flock 平台：退化为原行为（单进程安全）
    fcntl = None

UNK_TOKEN = "<unk>"
TEST_ACCESS_LOG = ".test_access_log.jsonl"

# 非法控制符（保留 \t \n \r）；U+FFFD 替换为空格（长度不变，不跨词粘连）
_SANITIZE_TABLE = {i: None for i in range(32) if i not in (9, 10, 13)}
_SANITIZE_TABLE[0x7f] = None
_SANITIZE_TABLE[ord("\ufffd")] = " "

# 行级标签噪声（语言建模标签即下一 token，来自同一份清洗前文本）：
# 含 U+FFFD/非法控制符的行比例反映“标签质量”。
# 只匹配坏字符本身：单字符类扫描是线性的；若用 `[^\n]*坏字符[^\n]*` 按行
# 匹配，正则引擎在无坏字符的行上会从每个起点回溯整行（O(行长²/行)），
# 长行大语料下不可取。行号由坏字符位置之间的换行增量换算
# （见 sanitize_text），整体单趟线性。
_BAD_CHAR_RE = re.compile(r"[\ufffd\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# WikiText 风格的一级/多级标题行：作为“文档边界”的首选切点。
# 兼容 `= Title =`、`== Section ==`、`=== Sub ===` 及 WikiText-103 原始
# 语料的分隔写法 `= = Section = =`、`= = = Sub = = =`。
_HEADING_RE = re.compile(
    r"(?m)^[ \t]*(?:=[ \t]*)+[^=\s\r\n][^\r\n]*?(?:[ \t]*=)+[ \t]*\r?$")


# ---------------------------------------------------------------- 文本清洗

def sanitize_text(text: str):
    """清除会污染标签的坏字符；返回 (clean_text, stats)。

    只做逐字符的确定性替换/删除，不依赖任何跨样本统计量，因此可以在
    划分之后对每个 split 独立执行，不产生泄露。

    stats 额外给出行级标签噪声估计（bad_char_line_rate）：坏字符不是均匀
    分布的，逐行比例比总量更能反映“多少标签受影响”。
    """
    stats = {
        "replacement_char_replaced_with_space": int(text.count("\ufffd")),
        "nul_deleted": int(text.count("\x00")),
    }
    clean = text.translate(_SANITIZE_TABLE)
    # translate 只删除 NUL/控制符（U+FFFD 是 1:1 替换），长度差即删除数
    stats["control_chars_deleted"] = int(len(text) - len(clean))
    n_lines = int(text.count("\n")) + 1
    stats["lines"] = n_lines
    # 字符计数为 0 时对应扫描必然无命中（U+FFFD 计数/删除计数分别覆盖两类
    # 坏字符），直接记为 0。一趟 _BAD_CHAR_RE 扫描即可同时得到三类行计数：
    # 命中行数即 lines_with_bad_chars，同一行出现多次只计一次；逐字符按
    # 类型分别累计 lines_with_replacement_char / lines_with_control_chars。
    # 行号 = 该位置之前已跨过的换行数（search_from 单调前移，累计线性）。
    has_repl = stats["replacement_char_replaced_with_space"] > 0
    has_ctrl = stats["control_chars_deleted"] > 0
    n_repl_lines = 0
    n_ctrl_lines = 0
    n_bad_lines = 0
    if has_repl or has_ctrl:
        line_no = 0
        search_from = 0
        last_bad = last_repl = last_ctrl = -1
        for m in _BAD_CHAR_RE.finditer(text):
            pos = m.start()
            line_no += text.count("\n", search_from, pos)
            search_from = pos
            if line_no != last_bad:
                n_bad_lines += 1
                last_bad = line_no
            if has_repl and text[pos] == "\ufffd":
                if line_no != last_repl:
                    n_repl_lines += 1
                    last_repl = line_no
            elif has_ctrl and line_no != last_ctrl:
                n_ctrl_lines += 1
                last_ctrl = line_no
    stats["lines_with_replacement_char"] = n_repl_lines
    stats["lines_with_control_chars"] = n_ctrl_lines
    stats["lines_with_bad_chars"] = n_bad_lines
    stats["bad_char_line_rate"] = float(n_bad_lines / max(1, n_lines))
    return clean, stats


# ---------------------------------------------------------------- 重复/去污染

def _normalize_line(s: str) -> str:
    """行内 NFKC + 空白折叠后的比较键。

    用于捕获“仅缩进/空格差异”或 Unicode 兼容字符变体（全角 Ａ vs A、
    表意空格 U+3000 vs 普通空格）的复制：精确字符串匹配会漏掉这类近似
    重复，而它们同样是跨 split 的样本级泄露。只做确定性的 NFKC 与空白
    归一化，不做大小写/标点改写（避免把语义不同的英文句子误判为重复）。
    """
    s = unicodedata.normalize("NFKC", s)
    return " ".join(s.split())


def _iter_lines(text: str):
    """逐行迭代，与 text.split("\\n") 的长度/内容序列一致。"""
    start = 0
    while True:
        i = text.find("\n", start)
        if i < 0:
            yield text[start:]
            return
        yield text[start:i]
        start = i + 1


def decontaminate_texts(texts: dict, min_chars: int = 40):
    """清理 train 侧重复样本，并诊断 eval 两侧（val/test）的重复。

    三类重复都处理/统计（最小改动原则：只改 train，val/test 保持尘封）：
      * train 与 val/test 的跨 split 精确重复；
      * train 与 val/test 的跨 split 空白归一化重复；
      * train **内部**重复：保留首次出现，删除后续精确/空白归一化重复
        （重复训练样本会放大其权重并增加记忆，属于数据清洗而非评测改写）。
    行内容先 strip；短行（< min_chars，如 WikiText 的 "= References =" 标题）
    不计入去污染，避免大面积误删；但仍会统计其重复数量供审查。
    返回 (cleaned_texts, stats)。

    val/test 内容绝不被改写；val 与 test 之间的重复另行统计
    （val_test_overlap_*）并由报告告警——两侧样本重叠会让“用 val 选模型、
    在 test 上最终评估”不再独立。

    比较集合使用 128-bit 摘要（16 字节碰撞概率可忽略），
    train 输出按删除切点拼接未删除片段。
    """
    min_chars = int(min_chars)
    # 顶层非对象或缺少 train（旧格式/手改缓存）时按空文本处理，避免一个
    # 损坏的数据缓存让 prepare_data 直接 KeyError/AttributeError。
    texts = texts if isinstance(texts, dict) else {}

    def digest(s: str) -> bytes:
        return hashlib.blake2b(s.encode("utf-8"), digest_size=16).digest()

    def scan(t):
        if not isinstance(t, str):
            t = ""
        # 单次扫描：每行只 strip/normalize 一次。长行（>= min_chars）用于
        # 去污染；非空行的重复都计数——短行不参与去污染，但重复数量仍统计
        # 供审查（见 docstring）。纯空白行不计，避免换行聚集把重复数虚增。
        # 集合使用 128-bit 摘要（16 字节碰撞概率可忽略）。
        n_long, n_valid = 0, 0
        seen_long, seen_long_norm = set(), set()
        seen_valid, seen_valid_norm = set(), set()
        for ln in _iter_lines(t or ""):
            s = ln.strip()
            if not s:
                continue
            n_valid += 1
            sn = _normalize_line(s)
            d = digest(s)
            dn = d if sn == s else digest(sn)
            seen_valid.add(d)
            seen_valid_norm.add(dn)
            if len(s) >= min_chars:
                n_long += 1
                seen_long.add(d)
                seen_long_norm.add(dn)
        return {"n_long": n_long, "long": seen_long,
                "long_norm": seen_long_norm, "n_valid": n_valid,
                "n_uniq": len(seen_valid), "n_uniq_norm": len(seen_valid_norm)}

    eval_lines, eval_norm_lines = set(), set()
    eval_counts = {}
    split_lines = {}
    for split in ("val", "test"):
        st = scan(texts.get(split))
        eval_lines.update(st["long"])
        eval_norm_lines.update(st["long_norm"])
        eval_counts[split] = (st["n_long"], st["n_valid"], st["n_uniq"],
                              st["n_uniq_norm"])
        split_lines[split] = (st["long"], st["long_norm"])
    val_long, val_long_norm = split_lines["val"]
    test_long, test_long_norm = split_lines["test"]
    # eval 两侧的重复：不修改任何一侧（test 必须尘封），但必须显式报告。
    val_test_overlap = len(val_long & test_long)
    val_test_overlap_norm = len(val_long_norm & test_long_norm)

    train_text = texts.get("train")
    if not isinstance(train_text, str):
        train_text = ""
    n_train, n_valid_train = 0, 0
    # 仅用于计数 len() 的集合存摘要。
    seen_valid_train, seen_valid_train_norm = set(), set()
    # train 内部去重：长行摘要集合（首次出现保留）。
    seen_train_long, seen_train_long_norm = set(), set()
    dropped_exact, dropped_norm = 0, 0
    dropped_train_exact, dropped_train_norm = 0, 0
    # 流式扫描 train：被删行（连同其后换行）不进入任何片段；
    # 未删除片段在最后一次性拼接。
    parts, seg_start, any_drop = [], 0, False
    tail_dropped = False
    pos, n = 0, len(train_text)
    while True:
        nl = train_text.find("\n", pos)
        line_end = n if nl < 0 else nl
        s = train_text[pos:line_end].strip()
        d = dn = None
        if s:
            n_valid_train += 1
            sn = _normalize_line(s)
            d = digest(s)
            dn = d if sn == s else digest(sn)
            seen_valid_train.add(d)
            seen_valid_train_norm.add(dn)
        drop = False
        if s and len(s) >= min_chars:
            n_train += 1
            if d in eval_lines:
                drop, dropped_exact = True, dropped_exact + 1
            elif dn in eval_norm_lines:
                drop, dropped_norm = True, dropped_norm + 1
            elif d in seen_train_long:
                drop, dropped_train_exact = True, dropped_train_exact + 1
            elif dn in seen_train_long_norm:
                drop, dropped_train_norm = True, dropped_train_norm + 1
            # 无论是否删除都登记，保证后续同内容行被识别为重复。
            seen_train_long.add(d)
            seen_train_long_norm.add(dn)
        if drop:
            any_drop = True
            if pos > seg_start:
                parts.append(train_text[seg_start:pos])
            seg_start = line_end + 1 if nl >= 0 else n
            tail_dropped = nl < 0
        if nl < 0:
            break
        pos = nl + 1

    out = dict(texts)
    if any_drop:
        if seg_start < n:
            parts.append(train_text[seg_start:])
        merged = "".join(parts)
        # 原行为是"保留行以 \n 连接"：末行被删时不应留下其前导换行
        # （文本本身以 \n 结尾时末元素是空行，永远不会被删）。
        if tail_dropped and merged.endswith("\n"):
            merged = merged[:-1]
        out["train"] = merged
    else:
        out["train"] = train_text
    stats = {"min_chars": min_chars,
             "normalization": "collapse_whitespace",
             "unicode_normalization": "NFKC",
             "train_lines_dropped_as_duplicates_of_eval": int(dropped_exact),
             "train_lines_dropped_as_normalized_duplicates_of_eval":
                 int(dropped_norm),
             "train_intra_split_duplicate_lines_removed":
                 int(dropped_train_exact),
             "train_intra_split_normalized_duplicate_lines_removed":
                 int(dropped_train_norm),
             "val_test_overlap_long_lines": int(val_test_overlap),
             "val_test_overlap_normalized_long_lines":
                 int(val_test_overlap_norm),
             "val_test_overlap_rate": float(
                 val_test_overlap / max(1, len(val_long)))}
    per_split = {"train": (n_train, n_valid_train, len(seen_valid_train),
                           len(seen_valid_train_norm)),
                 "val": eval_counts["val"], "test": eval_counts["test"]}
    for split, (n_long, n_valid, n_uniq, n_uniq_norm) in per_split.items():
        stats[f"{split}_long_lines"] = int(n_long)
        stats[f"{split}_intra_split_duplicate_lines"] = int(
            n_valid - n_uniq)
        stats[f"{split}_intra_split_normalized_duplicate_lines"] = int(
            n_valid - n_uniq_norm)
    return out, stats


def _iter_blocks(text: str, min_lines: int = 2):
    """迭代多行块：返回 (起始偏移, 结束偏移, 归一化摘要, 内容行数)。

    块由空行分隔；标题行也会开启新块（WikiText 的一篇文章/小节），避免
    无空行语料被当成一个巨型块。块只包含非空行，摘要对每行做空白归一化后
    以换行连接。结束偏移取“下一个块首行”的起点，因此删除范围会一并带走
    块后的空行；块位于文件末尾时结束偏移为文本长度。
    """
    if not isinstance(text, str) or not text:
        return
    n = len(text)
    start = None
    digest = None
    count = 0
    blank_since_content = False
    pos = 0
    while True:
        nl = text.find("\n", pos)
        line_end = n if nl < 0 else nl
        s = text[pos:line_end].strip()
        if s:
            is_heading = bool(_HEADING_RE.match(s))
            if count and (blank_since_content or is_heading):
                if count >= int(min_lines):
                    yield start, pos, digest.digest(), count
                start, digest, count = None, None, 0
            if count == 0:
                start = pos
                digest = hashlib.blake2b(digest_size=16)
            norm = _normalize_line(s)
            digest.update(norm.encode("utf-8"))
            digest.update(b"\n")
            count += 1
            blank_since_content = False
        elif count:
            blank_since_content = True
        if nl < 0:
            break
        pos = nl + 1
    if count >= int(min_lines):
        yield start, n, digest.digest(), count


def _remove_ranges(text: str, ranges) -> str:
    """按字符区间删除片段；相邻/重叠区间先合并，未命中的原文保持逐字节不变。"""
    merged = []
    for start, end in sorted((int(a), int(b)) for a, b in ranges):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    if not merged:
        return text
    parts, prev = [], 0
    for start, end in merged:
        if start > prev:
            parts.append(text[prev:start])
        prev = max(prev, end)
    if prev < len(text):
        parts.append(text[prev:])
    return "".join(parts)


def decontaminate_documents(texts: dict, min_lines: int = 2):
    """块级去污染：只从 train 删除与 val/test 或 train 前文重复的多行块。

    行级去污染（decontaminate_texts）只处理长度 >= min_chars 的单行，无法
    发现“由多行短行组成的整段/整文档”跨 split 复制。这里按空行/标题切块，
    对块内每行做空白归一化后哈希；仅处理至少 min_lines 行的块，单行块继续
    交给行级逻辑，避免重复统计。

    只改写 train：val/test 保持逐字节原样（test 尘封）；val 与 test 之间的
    块重复只统计并告警。train 内部重复块保留首次出现。
    返回 (cleaned_texts, stats)。
    """
    texts = texts if isinstance(texts, dict) else {}

    def scan(t):
        hashes, n_blocks, n_lines = set(), 0, 0
        for _, _, digest, count in _iter_blocks(t, min_lines):
            hashes.add(digest)
            n_blocks += 1
            n_lines += count
        return hashes, n_blocks, n_lines

    eval_scan = {split: scan(texts.get(split)) for split in ("val", "test")}
    eval_hashes = set().union(*(st[0] for st in eval_scan.values()))
    train_text = texts.get("train")
    if not isinstance(train_text, str):
        train_text = ""

    seen, ranges = set(), []
    n_blocks, n_lines, dropped_lines = 0, 0, 0
    n_cross, n_intra = 0, 0
    for start, end, digest, count in _iter_blocks(train_text, min_lines):
        n_blocks += 1
        n_lines += count
        if digest in eval_hashes:
            n_cross += 1
            ranges.append((start, end))
            dropped_lines += count
        elif digest in seen:
            n_intra += 1
            ranges.append((start, end))
            dropped_lines += count
        else:
            seen.add(digest)

    dropped_chars = sum(b - a for a, b in ranges)
    out = dict(texts)
    current = _remove_ranges(train_text, ranges)
    # 删除一个块可能把两侧剩余行/标题重新拼成一个块，而它恰好又与
    # eval 或 train 前文重复（典型：删掉中间小节标题后，前后短行合并成
    # val 的整段）。只扫一遍会把这种“新生成”的副本留在 train 里，函数
    # 也不再幂等；因此反复扫描删除后的文本，直到没有可删块。
    while current != train_text:
        pass_hashes, pass_ranges = set(), []
        for start, end, digest, count in _iter_blocks(current, min_lines):
            if digest in eval_hashes:
                n_cross += 1
                pass_ranges.append((start, end))
                dropped_lines += count
            elif digest in pass_hashes:
                n_intra += 1
                pass_ranges.append((start, end))
                dropped_lines += count
            else:
                pass_hashes.add(digest)
        nxt = _remove_ranges(current, pass_ranges)
        if nxt == current:
            break
        dropped_chars += sum(b - a for a, b in pass_ranges)
        current = nxt
    out["train"] = current
    val_hashes = eval_scan["val"][0]
    test_hashes = eval_scan["test"][0]
    val_test_overlap = len(val_hashes & test_hashes)
    stats = {
        "min_lines": int(min_lines),
        "normalization": "collapse_whitespace",
        "unicode_normalization": "NFKC",
        "train_blocks_scanned": int(n_blocks),
        "train_block_lines_scanned": int(n_lines),
        "train_blocks_dropped_as_duplicates_of_eval": int(n_cross),
        "train_intra_split_duplicate_blocks_removed": int(n_intra),
        "train_lines_dropped": int(dropped_lines),
        "train_chars_dropped": int(dropped_chars),
        "val_blocks_scanned": int(eval_scan["val"][1]),
        "test_blocks_scanned": int(eval_scan["test"][1]),
        "val_test_overlap_blocks": int(val_test_overlap),
        "val_test_overlap_rate": float(
            val_test_overlap / max(1, len(val_hashes))),
    }
    return out, stats


def _iter_line_lengths(text: str):
    """逐行字符长度，与 text.split("\\n") 的长度序列一致，但不构造行字符串列表。"""
    start = 0
    while True:
        i = text.find("\n", start)
        if i < 0:
            yield len(text) - start
            return
        yield i - start
        start = i + 1


def text_length_stats(text: str, huge_line_chars: int = 100000) -> dict:
    """逐行长度分布：用于发现异常值（超长行会主导训练窗口）。

    只统计长度，不做删改；极端行的处置留给数据源（本流程不把长行当噪声删除，
    以免引入未记录的人为分布改动）。
    """
    if not text:
        return {"lines": 0, "chars": 0}
    lens = np.fromiter(_iter_line_lengths(text), dtype=np.int64,
                       count=text.count("\n") + 1)
    return {
        "lines": int(lens.size),
        "chars": int(len(text)),
        "line_chars_p50": float(np.percentile(lens, 50)),
        "line_chars_p99": float(np.percentile(lens, 99)),
        "line_chars_max": int(lens.max()),
        "lines_gt_huge_threshold": int((lens > int(huge_line_chars)).sum()),
        "huge_line_threshold_chars": int(huge_line_chars),
    }


# ---------------------------------------------------------------- token 流校验

def check_vocab_capacity(vocab_size: int) -> None:
    """token id 以 uint16 存储，词表必须满足 max_id = vocab_size-1 <= 65535。"""
    if int(vocab_size) < 1 or int(vocab_size) > 65536:
        raise ValueError(
            f"vocab_size={vocab_size} 无法用 uint16 存储（要求 1..65536）")


def token_stream_stats(ids, vocab_size: int, split: str = "train") -> dict:
    """token 流硬校验：整数、有限、id ∈ [0, vocab)；非法数据直接报错而非静默。

    NaN/Inf 对整数流天然不存在，但对浮点输入（未来若改用连续特征）必须显式
    拒绝——NaN 混入会经 embedding 传播为非法索引或静默取 0。
    """
    arr = np.asarray(ids)
    if arr.size == 0:
        raise ValueError(f"{split} token 流为空，无法训练/评测")
    n_nan = n_inf = 0
    if arr.dtype.kind == "f":
        n_nan = int(np.isnan(arr).sum())
        n_inf = int(np.isinf(arr).sum())
        if n_nan or n_inf:
            raise ValueError(f"{split} token 流含 NaN={n_nan}/Inf={n_inf}")
        # 非整数浮点 id（如 1.5、-0.5）一旦 astype 截断会静默改变 token，
        # 必须在范围检查前显式拒绝（-0.5 截断为 0 会绕过越界检查）。
        arr_r = np.rint(arr)
        if not np.array_equal(arr, arr_r):
            bad = arr[arr != arr_r].ravel()[:5]
            raise ValueError(
                f"{split} token 流含非整数 id，例如 {bad.tolist()}；"
                "浮点流必须是整数值")
        arr_i = arr_r.astype(np.int64)
    elif arr.dtype.kind in "iu":
        # 整数流的 min/max 在任何整数 dtype 上都是精确运算，无需升位。
        arr_i = arr
    else:
        raise ValueError(f"{split} token 流 dtype 非法: {arr.dtype}")
    vmin, vmax = int(arr_i.min()), int(arr_i.max())
    if vmin < 0 or vmax >= int(vocab_size) or vmax > 65535:
        raise ValueError(
            f"{split} token id 越界: [{vmin}, {vmax}] 不在 [0, {vocab_size}) "
            "或超出 uint16 可存储范围（<= 65535），数据/词表不一致")
    # 至此 vmax <= 65535 由本函数保证，写入 uint16 无损
    return {"split": str(split), "tokens": int(arr.size), "dtype": str(arr.dtype),
            "min_id": vmin, "max_id": vmax, "nan": n_nan, "inf": n_inf}


def unknown_char_rate(tok, text: str):
    """字符级分词器在给定文本上的 OOV 字符比例；BPE 等字节级分词返回 None。"""
    stoi = getattr(tok, "stoi", None)
    if stoi is None or not text:
        return None
    # 多字符特殊 token（如 <unk>）不是可编码字符，跳过
    table = {ord(ch): None for ch in stoi if len(ch) == 1}
    remaining = text.translate(table)
    return float(len(remaining) / max(1, len(text)))


# ---------------------------------------------------------------- 泄漏筛查

def _window_hashes(ids, n: int, n_samples: int, seed: int):
    """采样 n 长 token 窗口的哈希集合（有放回采样，重复可忽略）。"""
    ids = np.asarray(ids)
    L = int(ids.size)
    if L < n:
        return set()
    total = L - n + 1
    rng = np.random.default_rng(seed)
    m = min(int(n_samples), total)
    starts = (rng.integers(0, total, size=m) if m < total
              else np.arange(total))
    windows = np.lib.stride_tricks.sliding_window_view(ids, n)
    picked = np.ascontiguousarray(windows[np.asarray(starts, dtype=np.int64)])
    if picked.dtype != np.uint16:
        picked = picked.astype(np.uint16)
    out = set()
    for row in picked:
        out.add(hashlib.blake2b(row.tobytes(), digest_size=16).digest())
    return out


def ngram_overlap(train_ids, eval_ids, n: int = 64, n_samples: int = 20000,
                  seed: int = 0) -> dict:
    """采样长 n-gram 哈希筛查 eval 窗口在 train 中出现的比例（泄漏下界指标）。

    两侧都采样，因此结果是“采样到的最小重叠”，不是精确上界；>0 即说明存在
    跨 split 的逐 token 复制，应人工核查。n=64 足够长，自然语言巧合重叠可忽略。
    """
    train_h = _window_hashes(train_ids, n, n_samples, seed)
    eval_h = _window_hashes(eval_ids, n, n_samples, seed + 1)
    if not eval_h:
        return {"n": int(n), "eval_windows_sampled": 0, "overlap_rate": None}
    hits = len(train_h & eval_h)
    return {"n": int(n),
            "train_windows_sampled": int(len(train_h)),
            "eval_windows_sampled": int(len(eval_h)),
            "windows_found_in_train_sample": int(hits),
            "overlap_rate": float(hits / len(eval_h)),
            "screening": True}


def distribution_shift(train_ids, eval_ids, vocab_size: int,
                       max_samples: int = None, seed: int = 0) -> dict:
    """train 与 eval 的一元 token 分布漂移（默认全量精确计数）。

    返回 TV 距离、JS 散度（bit）、eval 中 train 未出现的 token 占比（OOV 率）
    以及频率差最大的若干 token id。OOV 必须基于完整 train 词表判断：若对
    train 采样后用“采样计数为 0”判 OOV，会把真实存在但未被采样的 token
    误报为 OOV。因此默认用分块 `bincount` 做全量计数；只有显式传入
    max_samples（且小于流长度）时才退化为有放回采样，并在结果中标记
    count_mode="sampled" 供审计。
    """
    try:
        limit = int(max_samples) if max_samples is not None else None
    except (TypeError, ValueError, OverflowError):
        limit = None
    if limit is not None and limit <= 0:
        limit = None

    def counts(ids, s):
        arr = np.asarray(ids)
        L = int(arr.size)
        if L == 0:
            return None, False
        sampled = limit is not None and limit < L
        if sampled:
            rng = np.random.default_rng(s)
            arr = arr[rng.integers(0, L, size=limit)]
            L = int(arr.size)
        out = np.zeros(int(vocab_size), dtype=np.int64)
        # 分块计数：内存只与 chunk 和词表大小有关，不随语料线性增长。
        chunk = 5_000_000
        for i in range(0, L, chunk):
            out += np.bincount(arr[i:i + chunk], minlength=int(vocab_size))
        return out.astype(np.float64), sampled

    c_train, s_train = counts(train_ids, seed)
    c_eval, s_eval = counts(eval_ids, seed + 1)
    if c_train is None or c_eval is None:
        return {"tv": None, "js_bits": None, "eval_oov_token_rate": None,
                "top_divergent_token_ids": [], "count_mode": "empty"}
    p = c_train / max(1.0, c_train.sum())
    q = c_eval / max(1.0, c_eval.sum())
    eps = 1e-12
    tv = 0.5 * float(np.abs(p - q).sum())
    pp, qq = p + eps, q + eps
    m = 0.5 * (pp + qq)

    def _kl(a, b):
        return float(np.sum(a * np.log2(a / b)))

    js = 0.5 * _kl(pp, m) + 0.5 * _kl(qq, m)
    oov = float(c_eval[c_train == 0].sum() / max(1.0, c_eval.sum()))
    order = np.argsort(-np.abs(p - q))[:10]
    exact = not (s_train or s_eval)
    return {
        "tv": tv, "js_bits": js, "eval_oov_token_rate": oov,
        "eval_oov_token_count": int(c_eval[c_train == 0].sum()),
        "top_divergent_token_ids": [int(i) for i in order],
        "count_mode": "exact" if exact else "sampled",
        "tokens_train": int(c_train.sum()),
        "tokens_eval": int(c_eval.sum()),
        # 兼容旧报告字段：精确模式下即全量 token 数。
        "sampled_tokens_train": int(c_train.sum()),
        "sampled_tokens_eval": int(c_eval.sum()),
    }


# ---------------------------------------------------------------- 时序划分

_NEWLINE_RE = re.compile("\n")
# 空行：两个换行之间只允许空白（兼容 CRLF 与“有空格的行”）。
_BLANK_LINE_RE = re.compile(r"(?<=\n)[ \t\r]*\n")


def _boundary_candidates(text: str, lo: int, hi: int, kind: str):
    # 换行/空行候选用正则扫描；标题候选额外校验真实行首（切片首位的
    # re.MULTILINE '^' 对任意切片起点都会匹配，可能是行中假命中）。
    seg = text[lo:hi]
    if kind == "heading":
        positions = []
        for m in _HEADING_RE.finditer(seg):
            if m.start() == 0 and lo > 0 and text[lo - 1] != "\n":
                # 切片起点落在行中间：串首的 '^' 不是真实行首，剔除该假标题
                continue
            positions.append(lo + m.start())
        return positions
    if kind == "blank":
        # 候选点是第二个换行之后（即下一个内容行起点）；CRLF/含空白的空行
        # 需要用到 m.end() 而不是 m.start()+1。
        return [lo + m.end() for m in _BLANK_LINE_RE.finditer(seg)]
    return [lo + m.start() + 1 for m in _NEWLINE_RE.finditer(seg)]


def _pick_boundary(text: str, target: int, window: int):
    """在 target 附近挑一个尽量靠近、且尽量落在文档边界（标题 > 空行 > 换行）的切点。

    候选打分 = |距离| - slack*window，slack 越大越优先：标题 0.75、空行 0.35、
    换行 0。确定性选择，无随机性。
    """
    lo = max(0, target - window)
    hi = min(len(text), target + window)
    best = None
    for kind, slack in (("heading", 0.75), ("blank", 0.35), ("newline", 0.0)):
        for pos in _boundary_candidates(text, lo, hi, kind):
            if pos <= 0 or pos >= len(text):
                continue
            score = (abs(pos - target) - slack * window, pos, kind)
            if best is None or score[0] < best[0] or (
                    score[0] == best[0] and score[1] < best[1]):
                best = score
    if best is None:
        return None, None
    return int(best[1]), best[2]


def _heading_positions(text: str):
    """所有有效标题行的行首偏移（含位置 0 的标题：它计数，但不作切点）。

    位置 0 的标题必须计入 n_headings（它是真实存在的文档边界，漏计会造成
    “6 个文档只报 5 个标题”的审计错误）；是否可选为切点由调用方保证——
    切点候选一律要求 pos>0，否则 train 会为空。
    """
    return [m.start() for m in _HEADING_RE.finditer(text)]


def _nearest_heading_in_range(headings, target: int, lo: int, hi: int):
    """[lo, hi] 内离 target 最近（并列取靠前）的标题行首；无则 None。"""
    cands = [h for h in headings if lo <= h <= hi]
    if not cands:
        return None
    return min(cands, key=lambda h: (abs(h - target), h))


def temporal_split_text(text: str, train_frac: float = 0.8,
                        val_frac: float = 0.1, window_frac: float = 0.02,
                        min_gap: int = 1):
    """连续文本的时序划分：train=最早、val=其后、test=最新（绝不随机打乱）。

    切点优先落在文档边界（WikiText 标题行 > 空行段落 > 换行），搜索窗口为
    len(text)*window_frac。**只要文本含标题行，就优先在全文范围内选最近的
    标题作为切点**（哪怕超出窗口）——标题定义的“文档”是群组单位，同一文档
    绝不跨 split；仅当有效范围内没有任何标题时才退回窗口内的段落/换行切分，
    此时按段落/行语义仍是完整单位。找不到任何换行时退化为字符切分
    （info 中 fallback_char_cut=True 提示该文件无法做文档级划分）。

    info["documents_cross_cut"]（标题文档是否跨切点）是显式自检：为 True 时
    调用方应告警——只有“标题间距大于划分区间”这类无法两全的语料才会出现。
    返回 (train_text, val_text, test_text, info)。
    """
    n = len(text)
    if n < 3:
        raise ValueError("文本太短（<3 字符），无法做 train/val/test 时序划分")
    t1 = max(1, int(n * float(train_frac)))
    t2 = max(t1 + 1, int(n * (float(train_frac) + float(val_frac))))
    t2 = min(t2, n - 1)
    window = max(256, int(n * float(window_frac)))
    heads = _heading_positions(text)
    p1, k1 = _pick_boundary(text, t1, window)
    # 标题优先：窗口内没有“可用”标题时在全文中找最近的（范围保证 p1 < t2，
    # val 非空）。窗口内命中的标题若 >= t2 同样不可用，必须换正文范围内的标题。
    if heads and (p1 is None or k1 != "heading" or p1 >= t2):
        h = _nearest_heading_in_range(heads, t1, 1, t2 - 1)
        if h is not None:
            p1, k1 = h, "heading"
    p2, k2 = _pick_boundary(text, t2, window)
    # 同理：窗口内命中的标题若离 p1 太近（<= p1+min_gap，val 会被压空/切序
    # 倒置），也要换到 [p1+min_gap, n-1] 里最近的标题。
    if heads and (p2 is None or k2 != "heading"
                  or p2 <= (p1 or 0) + min_gap):
        h = _nearest_heading_in_range(heads, t2, (p1 or 0) + min_gap, n - 1)
        if h is not None:
            p2, k2 = h, "heading"
    fallback = False
    # p1 必须严格小于 t2：否则 val=[p1,p2) 会被 p2≈t2 压空并触发“空 split”。
    # 短文本里唯一边界可能落在 t2 之后（如仅有的换行恰在结尾附近），
    # 此时退化为按比例字符切分，并保证 p1 至少留 1 个字符给 train、1 个给 val。
    if p1 is None or p1 >= t2:
        p1, k1, fallback = min(t1, t2 - 1), "char", True
    if p2 is None or p2 <= p1 + min_gap:
        p2, k2, fallback = max(p1 + min_gap, t2), "char", True
    if p2 >= n:
        p2 = n
    train_text, val_text, test_text = text[:p1], text[p1:p2], text[p2:]
    if not (train_text and val_text and test_text):
        raise ValueError("时序划分产生了空 split，请检查文本长度与划分比例")
    heading_set = set(heads)
    documents_cross = bool(heads) and (p1 not in heading_set
                                       or p2 not in heading_set)
    info = {
        "order": "temporal:train(earliest)->val->test(latest)",
        "cut_offsets": [int(p1), int(p2)], "cut_kinds": [str(k1), str(k2)],
        "window_chars": int(window), "fallback_char_cut": bool(fallback),
        "shuffled": False,
        # 群组自检：标题定义文档；为 True 表示至少一个文档跨了切点
        "n_headings": int(len(heads)),
        "documents_cross_cut": bool(documents_cross),
        "group_unit": ("heading_document" if heads else "line_segment"),
        "lengths_chars": {"train": len(train_text), "val": len(val_text),
                          "test": len(test_text)},
    }
    return train_text, val_text, test_text, info


# ---------------------------------------------------------------- 测试集访问账本

def read_test_access_log(data_dir: str) -> list:
    path = os.path.join(data_dir, TEST_ACCESS_LOG)
    if not os.path.exists(path):
        return []
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            # 合法 JSON 但非对象的行（null/数组/标量）按坏行跳过；
            # summarize_test_access 需要按 dict 取字段。
            if isinstance(rec, dict):
                records.append(rec)
    return records


def append_test_access(data_dir: str, record: dict) -> dict:
    """追加一条 test 访问记录并返回审计摘要（原子写入 + 进程间文件锁）。

    读-改-写必须跨进程互斥：run_all 的评测阶段会并发评测多个变体，无锁时
    两个进程各自读旧账本再覆盖，会互相吞掉记录。锁加在独立的 .lock 文件上
    （对账本文件本身加锁不可行：os.replace 会更换 inode，后到者锁的是旧文件）；
    无 fcntl 的平台退化为原行为，不阻塞评测。
    """
    path = os.path.join(data_dir, TEST_ACCESS_LOG)
    lock_fp = None
    if fcntl is not None:
        lock_fp = open(path + ".lock", "a")
        fcntl.flock(lock_fp.fileno(), fcntl.LOCK_EX)
    try:
        records = read_test_access_log(data_dir)
        entry = dict(record)
        entry.setdefault("time", time.strftime("%Y-%m-%d %H:%M:%S"))
        records.append(entry)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        os.replace(tmp, path)
        return summarize_test_access(records)
    finally:
        if lock_fp is not None:
            fcntl.flock(lock_fp.fileno(), fcntl.LOCK_UN)
            lock_fp.close()


def summarize_test_access(records) -> dict:
    """测试集访问审计：评测次数、不同 checkpoint 数、被重复评测的 checkpoint。"""
    records = [r for r in (records or []) if isinstance(r, dict)]
    ckpts = [str(r.get("ckpt_sha256") or r.get("ckpt") or "")
             for r in records]
    counts = {}
    for c in ckpts:
        if c:
            counts[c] = counts.get(c, 0) + 1
    repeated = [c[:16] for c, n in counts.items() if n > 1]
    return {
        "evaluations": int(len(records)),
        "distinct_checkpoints": int(len(counts)),
        "checkpoints_evaluated_multiple_times": repeated,
        "first_time": records[0].get("time") if records else None,
        "last_time": records[-1].get("time") if records else None,
    }


# ---------------------------------------------------------------- 报告组装

# 自检告警阈值：不是统计检验的显著性水平，而是“必须人工复核”的操作触发线，
# 与开放集/结构不同分布下的经验值一致（可通过 extra_report 覆盖记录）。
ALERT_THRESHOLDS = {
    "ngram_overlap_rate": 0.0,       # 任何跨 split 长 n-gram 复制都不允许
    "distribution_tv": 0.1,          # 一元 TV 距离 > 0.1 视为明显漂移
    "distribution_js_bits": 0.1,     # JS 散度（bit）
    "eval_oov_token_rate": 0.02,     # eval 中 train 未见 token 质量占比
    "eval_unk_char_rate": 0.005,     # val/test 字符级 OOV 率
    "bad_char_line_rate": 0.01,      # 含坏字符的行占比（标签噪声）
}


def leakage_alerts(tokenizer: dict, sanitization: dict, duplicates: dict,
                   contamination: dict, distribution: dict,
                   text_stats: dict = None, thresholds: dict = None) -> list:
    """把防泄漏/清洗诊断转成显式告警（供打印与汇总页展示）。

    返回 [{"level": "warning"|"info", "code": ..., "message": ...}]。
    没有告警时返回空列表——"无告警"与"没有自检"是两回事，后者由报告缺失表示。
    """
    def _num(v):
        try:
            f = float(v)
        except (TypeError, ValueError, OverflowError):
            return None
        return f if np.isfinite(f) else None

    def _count(v):
        try:
            return int(v or 0)
        except (TypeError, ValueError, OverflowError):
            return 0

    th = dict(ALERT_THRESHOLDS)
    if isinstance(thresholds, dict):
        for k, v in thresholds.items():
            if k in th:
                f = _num(v)
                if f is not None:
                    th[k] = f
    alerts = []

    def add(level, code, message):
        alerts.append({"level": level, "code": code, "message": message})

    contamination = contamination if isinstance(contamination, dict) else {}
    for key, label in (("train_val", "train/val"), ("train_test", "train/test"),
                       ("val_test", "val/test")):
        c = contamination.get(key)
        c = c if isinstance(c, dict) else {}
        rate = _num(c.get("overlap_rate"))
        if rate is not None and rate > th["ngram_overlap_rate"]:
            add("warning", f"ngram_overlap_{key}",
                f"{label} 采样 {c.get('n', '?')}-gram 重叠率 {rate:.4%}"
                "（>0 表示疑似跨 split 语料复制，须人工核查）")
    distribution = distribution if isinstance(distribution, dict) else {}
    for key, label in (("val_vs_train", "val−train"),
                       ("test_vs_train", "test−train"),
                       ("test_vs_val", "test−val")):
        d = distribution.get(key)
        d = d if isinstance(d, dict) else {}
        tv, js = _num(d.get("tv")), _num(d.get("js_bits"))
        if tv is not None and tv > th["distribution_tv"]:
            add("warning", f"distribution_tv_{key}",
                f"{label} 一元 token TV 距离 {tv:.4f} > "
                f"{th['distribution_tv']}（分布漂移明显，结论外推需谨慎）")
        if js is not None and js > th["distribution_js_bits"]:
            add("warning", f"distribution_js_{key}",
                f"{label} JS 散度 {js:.4f} bit > {th['distribution_js_bits']}")
        oov = _num(d.get("eval_oov_token_rate"))
        if oov is not None and oov > th["eval_oov_token_rate"]:
            add("warning", f"eval_oov_{key}",
                f"{label} 方向 OOV token 质量占比 {oov:.4%} > "
                f"{th['eval_oov_token_rate']}")
    tokenizer = tokenizer if isinstance(tokenizer, dict) else {}
    unk_rate = tokenizer.get("unk_rate")
    for split, rate in (unk_rate if isinstance(unk_rate, dict) else {}).items():
        rate = _num(rate)
        if rate is not None and rate > th["eval_unk_char_rate"]:
            add("warning", f"unk_char_rate_{split}",
                f"{split} 字符级 OOV 率 {rate:.4%} > "
                f"{th['eval_unk_char_rate']}（该 split 有较多字符映射 <unk>）")
    sanitization = sanitization if isinstance(sanitization, dict) else {}
    for split, st in sanitization.items():
        st = st if isinstance(st, dict) else {}
        rate = _num(st.get("bad_char_line_rate"))
        if rate is not None and rate > th["bad_char_line_rate"]:
            add("warning", f"bad_char_lines_{split}",
                f"{split} 含坏字符行占比 {rate:.4%} > "
                f"{th['bad_char_line_rate']}（标签噪声偏高，建议回查语料）")
    dup = duplicates if isinstance(duplicates, dict) else {}
    n_exact = _count(dup.get("train_lines_dropped_as_duplicates_of_eval"))
    n_norm = _count(
        dup.get("train_lines_dropped_as_normalized_duplicates_of_eval"))
    if n_exact or n_norm:
        add("info", "decontamination_applied",
            f"已从 train 移除与 val/test 重复的长行：精确 {n_exact} 行、"
            f"空白归一化后额外 {n_norm} 行（val/test 未改写）")
    n_train_dup = _count(
        dup.get("train_intra_split_duplicate_lines_removed"))
    n_train_dup_norm = _count(
        dup.get("train_intra_split_normalized_duplicate_lines_removed"))
    if n_train_dup or n_train_dup_norm:
        add("info", "train_intra_duplicates_removed",
            f"已从 train 移除内部重复长行：精确 {n_train_dup} 行、"
            f"空白归一化后额外 {n_train_dup_norm} 行（保留首次出现）")
    doc = dup.get("document_level")
    doc = doc if isinstance(doc, dict) else {}
    n_doc_cross = _count(
        doc.get("train_blocks_dropped_as_duplicates_of_eval"))
    n_doc_intra = _count(
        doc.get("train_intra_split_duplicate_blocks_removed"))
    if n_doc_cross or n_doc_intra:
        add("info", "document_decontamination_applied",
            f"已从 train 移除多行块：与 val/test 重复 {n_doc_cross} 个、"
            f"train 内部重复 {n_doc_intra} 个"
            f"（共 {_count(doc.get('train_lines_dropped'))} 行；"
            "单行块交给行级去污染，val/test 未改写）")
    n_doc_eval_dup = _count(doc.get("val_test_overlap_blocks"))
    if n_doc_eval_dup:
        add("warning", "eval_splits_overlap_documents",
            f"val 与 test 有 {n_doc_eval_dup} 个多行块重复：两侧均保持尘封"
            "未改写，但 val 选型会部分预演 test，最终 test 指标的独立性下降，"
            "须核查/重建划分")
    n_eval_dup = _count(dup.get("val_test_overlap_long_lines"))
    n_eval_dup_norm = _count(
        dup.get("val_test_overlap_normalized_long_lines"))
    if n_eval_dup or n_eval_dup_norm:
        add("warning", "eval_splits_overlap",
            f"val 与 test 有长行重复：精确 {n_eval_dup} 条、空白归一化后共 "
            f"{n_eval_dup_norm} 条（含精确重复）：两侧均保持尘封未改写，但 "
            "val 选型会部分预演 test，最终 test 指标的独立性下降，"
            "须核查/重建划分")
    for split in ("val", "test"):
        n_dup = _count(dup.get(f"{split}_intra_split_duplicate_lines"))
        n_dup_norm = _count(
            dup.get(f"{split}_intra_split_normalized_duplicate_lines"))
        if n_dup or n_dup_norm:
            # 统计键覆盖所有非空行（含短行），且归一化计数包含精确重复；
            # 只读精确计数会漏掉“仅空白/NFKC 差异”的重复，文案也不能写成
            # “长行”（与 decontaminate_texts 的口径不一致）。
            if n_dup_norm:
                detail = (f"空白归一化后 {n_dup_norm} 条（其中精确 {n_dup} 条）")
            else:
                detail = f"精确 {n_dup} 条"
            add("info", f"{split}_intra_duplicates",
                f"{split} 内部有重复行：{detail}（未改写，避免破坏尘封；"
                "重复会高估评测有效样本量，解读置信区间时须注意）")
    text_stats = text_stats if isinstance(text_stats, dict) else {}
    for split, st in text_stats.items():
        st = st if isinstance(st, dict) else {}
        n_huge = _count(st.get("lines_gt_huge_threshold"))
        if n_huge:
            add("info", f"huge_lines_{split}",
                f"{split} 有 {n_huge} 行超过 "
                f"{st.get('huge_line_threshold_chars')} 字符"
                "（异常值已计数未删除，训练窗口可能被其主导）")
    return alerts


# 会让 train/val/test 独立性失效的协议违规：run_all 在训练前据此硬失败
# （可显式覆盖），避免“报告里有警告但全量训练照跑”。
CRITICAL_SPLIT_ALERT_CODES = (
    "split_shuffled", "split_char_cut", "split_documents_cross_cut")


def critical_split_alerts(data_report: dict) -> list:
    """从 data_report 中筛出必须在训练前处理的划分协议违规。"""
    if not isinstance(data_report, dict):
        return []
    alerts = data_report.get("alerts")
    if not isinstance(alerts, list):
        return []
    return [a for a in alerts
            if isinstance(a, dict)
            and a.get("code") in CRITICAL_SPLIT_ALERT_CODES]


def split_protocol_alerts(split_protocol: dict) -> list:
    """把上游划分协议的群组/时序风险转成显式告警（元数据不能“静默通过”）。

    split_protocol 来自 prepare_data 的 temporal_split_text info 或官方 split
    的静态描述。group_unit 定义的群组（如 heading_document）一旦跨 split，
    train/val/test 就不再独立——必须与 ngram 重叠同级进入 alerts，
    供汇总页与复核者看到，而不是只躺在报告元数据里。
    """
    alerts = []
    p = split_protocol if isinstance(split_protocol, dict) else {}
    if not p:
        return alerts

    def add(level, code, message):
        alerts.append({"level": level, "code": code, "message": message})

    if p.get("shuffled"):
        add("warning", "split_shuffled",
            "划分协议声明 shuffled=True：时序先后/群组完整性不可保证，须人工核查")
    if p.get("fallback_char_cut"):
        add("warning", "split_char_cut",
            "无可用换行或文档边界，退化为字符切分——同一文档可能跨 split"
            "（群组泄露），须人工核查")
    if p.get("documents_cross_cut"):
        add("warning", "split_documents_cross_cut",
            f"切点未落在文档边界：{p.get('group_unit', 'group')} 群组跨 split"
            f"（标题 {p.get('n_headings', '?')} 个、切点类型 "
            f"{p.get('cut_kinds', '?')}），违反“群组整体落在同一侧”，"
            "须调整划分比例或更换可整体划分的语料")
    return alerts


def build_data_report(tokenizer: dict, splits: dict, sanitization: dict,
                      duplicates: dict, contamination: dict,
                      distribution: dict, text_stats: dict = None,
                      extra: dict = None,
                      alert_thresholds: dict = None) -> dict:
    """汇总为 data_report.json（只含 JSON 原生类型）。"""
    alert_thresholds = (alert_thresholds
                        if isinstance(alert_thresholds, dict) else {})
    alerts = leakage_alerts(tokenizer, sanitization, duplicates,
                            contamination, distribution, text_stats,
                            thresholds=alert_thresholds)
    report = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "protocol": {
            "split_first_then_preprocess": True,
            "fit_on": "train_only",
            "test_sealed": True,
            "shuffle_at_split_time": False,
            "batch_shuffle_during_training_allowed": True,
        },
        "tokenizer": tokenizer,
        "splits": splits,
        "sanitization": sanitization,
        "text_stats": text_stats or {},
        "duplicates": duplicates,
        "contamination": contamination,
        "distribution_shift": distribution,
        "alert_thresholds": dict(ALERT_THRESHOLDS, **alert_thresholds),
        "alerts": alerts,
        "notes": [
            "所有拟合类统计（字符词表等）只在 train 上完成，val/test 只做 transform；",
            "去污染只改写 train：先删除与 val/test 重复的多行块及 train 内部"
            "重复块（单行块交给行级逻辑），再移除与 val/test 重复的长行并删除"
            "train 内部重复长行（保留首次出现）；val/test 不被改写；",
            "val 与 test 之间的重复/64-gram 重叠单独统计并告警：两侧均不改写，"
            "但重叠会削弱 final test 相对 val 选择集的独立性；",
            "64-gram 重叠为采样筛查下界，>0 时应人工核查对应语料；",
            "一元分布漂移默认全量精确计数（显式 max_samples 时才采样），"
            "用于提示 split 间分布差异而非统计检验；",
            "alerts 同时包含协议违规（群组/时序，见 split_protocol）与阈值触发的"
            "复核提示（见 alert_thresholds），不是显著性检验。",
        ],
    }
    if isinstance(extra, dict):
        report.update(extra)
    # 协议违规（群组跨 split / 字符切分 / 打乱）置顶，确保汇总页的告警截断
    # 不会把更严重的问题挤出视野。
    protocol_alerts = split_protocol_alerts(report.get("split_protocol"))
    if protocol_alerts:
        report["alerts"] = protocol_alerts + report["alerts"]
    return report
