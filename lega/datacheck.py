"""数据卫生与防泄漏检查：划分协议、清洗、去污染、漂移筛查与测试集访问账本。

实验协议（本模块是实现而不是口号）:
  * 先划分 train/val/test，再做任何预处理；字符级词表只在 train 上拟合，
    val/test 只做 transform，未见字符统一映射到 <unk>（见 data.py）；
  * 连续文本（--local_txt）按时间先后切分：最早 80% 训练、其后 10% 验证、
    最新 10% 测试；切点对齐文档/段落/行边界，同一文档绝不跨 split；
  * 重复样本：与 val/test 精确重复的长行从 train 移除（去污染），val/test
    保持尘封不动；另用采样 64-gram 哈希筛查跨 split 的 token 级重叠；
  * token 流硬校验：整数 dtype、有限值、id ∈ [0, vocab)、uint16 可容纳；
  * 分布漂移：train 与 val/test 的（采样）一元 token 分布 TV/JS 散度与 OOV；
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
    r"(?m)^[ \t]*(?:=[ \t]*)+[^=\s\n][^\n]*?(?:[ \t]*=)+[ \t]*$")


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
    """行内空白折叠（含制表符/连续空格）后的比较键。

    用于捕获“仅缩进/空格差异”的复制：精确字符串匹配会漏掉这类近似重复，
    而它们同样是跨 split 的样本级泄露。只做确定性的空白归一化，不做
    大小写/标点改写（避免把语义不同的英文句子误判为重复）。
    """
    return " ".join(s.split())


def decontaminate_texts(texts: dict, min_chars: int = 40):
    """跨 split 重复的长行只从 train 移除；val/test 保持原样（尘封）。

    两类重复都处理（最小改动原则，宁删 train 不碰 eval）：
      * 精确重复（strip 后逐字符相同）；
      * 空白归一化重复（连续空白折叠后相同，捕获仅缩进/空格差异的复制）。
    行内容先 strip；短行（< min_chars，如 WikiText 的 "= References =" 标题）
    不计入去污染，避免大面积误删；但仍会统计其重复数量供审查。
    返回 (cleaned_texts, stats)。
    """
    min_chars = int(min_chars)

    def digest(s: str) -> bytes:
        return hashlib.blake2b(s.encode("utf-8"), digest_size=16).digest()

    def scan(t):
        # 单次扫描：每行只 strip/normalize 一次。长行（>= min_chars）用于
        # 去污染；非空行的重复都计数——短行不参与去污染，但重复数量仍统计
        # 供审查（见 docstring）。纯空白行不计，避免换行聚集把重复数虚增。
        # 仅用于 len() 的计数集合存 128-bit 摘要而非整行字符串
        # （16 字节碰撞概率可忽略）。
        n_long, n_valid = 0, 0
        seen_long, seen_long_norm = set(), set()
        seen_valid, seen_valid_norm = set(), set()
        for ln in (t or "").split("\n"):
            s = ln.strip()
            if not s:
                continue
            n_valid += 1
            sn = _normalize_line(s)
            d = digest(s)
            seen_valid.add(d)
            # 行本身已是归一化形态时两个摘要相同，直接复用第一次哈希。
            seen_valid_norm.add(d if sn == s else digest(sn))
            if len(s) >= min_chars:
                n_long += 1
                seen_long.add(s)
                seen_long_norm.add(sn)
        return (n_long, seen_long, seen_long_norm, n_valid,
                len(seen_valid), len(seen_valid_norm))

    eval_lines, eval_norm_lines = set(), set()
    eval_counts = {}
    for split in ("val", "test"):
        (n_long, seen_long, seen_long_norm, n_valid, n_uniq,
         n_uniq_norm) = scan(texts.get(split))
        eval_lines.update(seen_long)
        eval_norm_lines.update(seen_long_norm)
        eval_counts[split] = (n_long, n_valid, n_uniq, n_uniq_norm)

    kept, dropped_exact, dropped_norm = [], 0, 0
    n_train, n_valid_train = 0, 0
    # 仅用于计数 len() 的集合存摘要；归一化每行只算一次。
    seen_valid_train, seen_valid_train_norm = set(), set()
    for ln in texts["train"].split("\n"):
        s = ln.strip()
        sn = None
        if s:
            n_valid_train += 1
            sn = _normalize_line(s)
            d = digest(s)
            seen_valid_train.add(d)
            seen_valid_train_norm.add(d if sn == s else digest(sn))
        if len(s) >= min_chars:
            n_train += 1
            if s in eval_lines:
                dropped_exact += 1
                continue
            if sn in eval_norm_lines:
                dropped_norm += 1
                continue
        kept.append(ln)

    out = dict(texts)
    out["train"] = "\n".join(kept)
    stats = {"min_chars": min_chars,
             "normalization": "collapse_whitespace",
             "train_lines_dropped_as_duplicates_of_eval": int(dropped_exact),
             "train_lines_dropped_as_normalized_duplicates_of_eval":
                 int(dropped_norm)}
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
    L = len(ids)
    if L < n:
        return set()
    total = L - n + 1
    rng = np.random.default_rng(seed)
    m = min(int(n_samples), total)
    starts = (rng.integers(0, total, size=m) if m < total
              else np.arange(total))
    out = set()
    for s in starts:
        win = np.asarray(ids[int(s):int(s) + n], dtype=np.uint16)
        out.add(hashlib.blake2b(win.tobytes(), digest_size=16).digest())
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
                       max_samples: int = 2_000_000, seed: int = 0) -> dict:
    """train 与 eval 的一元 token 分布漂移（采样估计）。

    返回 TV 距离、JS 散度（bit）、eval 中 train 未出现的 token 占比（OOV 率）
    以及频率差最大的若干 token id。稀疏 token 的排序仅作审查提示。
    """
    def counts(ids, s):
        L = len(ids)
        if L == 0:
            return None
        rng = np.random.default_rng(s)
        n = min(int(max_samples), L)
        idx = (rng.integers(0, L, size=n) if n < L else np.arange(L))
        arr = np.asarray(ids)
        return np.bincount(arr[idx], minlength=vocab_size).astype(np.float64)

    c_train = counts(train_ids, seed)
    c_eval = counts(eval_ids, seed + 1)
    if c_train is None or c_eval is None:
        return {"tv": None, "js_bits": None, "eval_oov_token_rate": None,
                "top_divergent_token_ids": []}
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
    return {
        "tv": tv, "js_bits": js, "eval_oov_token_rate": oov,
        "eval_oov_token_count": int(c_eval[c_train == 0].sum()),
        "top_divergent_token_ids": [int(i) for i in order],
        "sampled_tokens_train": int(c_train.sum()),
        "sampled_tokens_eval": int(c_eval.sum()),
    }


# ---------------------------------------------------------------- 时序划分

_NEWLINE_RE = re.compile("\n")
_BLANK_LINE_RE = re.compile(r"(?<=\n)\n")


def _boundary_candidates(seg: str, base: int, kind: str):
    # 换行/空行候选用正则扫描。
    if kind == "heading":
        return [base + m.start() for m in _HEADING_RE.finditer(seg)
                if base + m.start() > 0]
    if kind == "blank":
        return [base + m.start() + 1 for m in _BLANK_LINE_RE.finditer(seg)]
    return [base + m.start() + 1 for m in _NEWLINE_RE.finditer(seg)]


def _pick_boundary(text: str, target: int, window: int):
    """在 target 附近挑一个尽量靠近、且尽量落在文档边界（标题 > 空行 > 换行）的切点。

    候选打分 = |距离| - slack*window，slack 越大越优先：标题 0.75、空行 0.35、
    换行 0。确定性选择，无随机性。
    """
    lo = max(0, target - window)
    hi = min(len(text), target + window)
    seg = text[lo:hi]
    best = None
    for kind, slack in (("heading", 0.75), ("blank", 0.35), ("newline", 0.0)):
        for pos in _boundary_candidates(seg, lo, kind):
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
    # 标题优先：窗口内没有标题时在全文中找最近的（范围保证 p1 < t2，val 非空）
    if heads and (p1 is None or k1 != "heading"):
        h = _nearest_heading_in_range(heads, t1, 1, t2 - 1)
        if h is not None:
            p1, k1 = h, "heading"
    p2, k2 = _pick_boundary(text, t2, window)
    if heads and (p2 is None or k2 != "heading"):
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
                records.append(json.loads(line))
            except ValueError:
                continue
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
    th = dict(ALERT_THRESHOLDS)
    if thresholds:
        th.update(thresholds)
    alerts = []

    def add(level, code, message):
        alerts.append({"level": level, "code": code, "message": message})

    for key, label in (("train_val", "train/val"), ("train_test", "train/test")):
        c = (contamination or {}).get(key) or {}
        rate = c.get("overlap_rate")
        if rate is not None and rate > th["ngram_overlap_rate"]:
            add("warning", f"ngram_overlap_{key}",
                f"{label} 采样 {c.get('n', '?')}-gram 重叠率 {rate:.4%}"
                "（>0 表示疑似跨 split 语料复制，须人工核查）")
    for key, label in (("val_vs_train", "val−train"),
                       ("test_vs_train", "test−train")):
        d = (distribution or {}).get(key) or {}
        tv, js = d.get("tv"), d.get("js_bits")
        if tv is not None and tv > th["distribution_tv"]:
            add("warning", f"distribution_tv_{key}",
                f"{label} 一元 token TV 距离 {tv:.4f} > "
                f"{th['distribution_tv']}（分布漂移明显，结论外推需谨慎）")
        if js is not None and js > th["distribution_js_bits"]:
            add("warning", f"distribution_js_{key}",
                f"{label} JS 散度 {js:.4f} bit > {th['distribution_js_bits']}")
        oov = d.get("eval_oov_token_rate")
        if oov is not None and oov > th["eval_oov_token_rate"]:
            add("warning", f"eval_oov_{key}",
                f"{label} 方向 OOV token 质量占比 {oov:.4%} > "
                f"{th['eval_oov_token_rate']}")
    for split, rate in ((tokenizer or {}).get("unk_rate") or {}).items():
        if rate is not None and rate > th["eval_unk_char_rate"]:
            add("warning", f"unk_char_rate_{split}",
                f"{split} 字符级 OOV 率 {rate:.4%} > "
                f"{th['eval_unk_char_rate']}（该 split 有较多字符映射 <unk>）")
    for split, st in (sanitization or {}).items():
        rate = (st or {}).get("bad_char_line_rate")
        if rate is not None and rate > th["bad_char_line_rate"]:
            add("warning", f"bad_char_lines_{split}",
                f"{split} 含坏字符行占比 {rate:.4%} > "
                f"{th['bad_char_line_rate']}（标签噪声偏高，建议回查语料）")
    dup = duplicates or {}
    n_exact = int(dup.get("train_lines_dropped_as_duplicates_of_eval") or 0)
    n_norm = int(
        dup.get("train_lines_dropped_as_normalized_duplicates_of_eval") or 0)
    if n_exact or n_norm:
        add("info", "decontamination_applied",
            f"已从 train 移除与 val/test 重复的长行：精确 {n_exact} 行、"
            f"空白归一化后额外 {n_norm} 行（val/test 未改写）")
    for split, st in (text_stats or {}).items():
        n_huge = int((st or {}).get("lines_gt_huge_threshold") or 0)
        if n_huge:
            add("info", f"huge_lines_{split}",
                f"{split} 有 {n_huge} 行超过 "
                f"{(st or {}).get('huge_line_threshold_chars')} 字符"
                "（异常值已计数未删除，训练窗口可能被其主导）")
    return alerts


def build_data_report(tokenizer: dict, splits: dict, sanitization: dict,
                      duplicates: dict, contamination: dict,
                      distribution: dict, text_stats: dict = None,
                      extra: dict = None,
                      alert_thresholds: dict = None) -> dict:
    """汇总为 data_report.json（只含 JSON 原生类型）。"""
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
        "alert_thresholds": dict(ALERT_THRESHOLDS, **(alert_thresholds or {})),
        "alerts": alerts,
        "notes": [
            "所有拟合类统计（字符词表等）只在 train 上完成，val/test 只做 transform；",
            "去污染只从 train 移除与 val/test 重复（精确/空白归一化）的长行，"
            "val/test 不被改写；",
            "64-gram 重叠为采样筛查下界，>0 时应人工核查对应语料；",
            "一元分布漂移为采样估计，用于提示 train/test 分布差异而非统计检验；",
            "alerts 为阈值触发的复核提示（见 alert_thresholds），不是显著性检验。",
        ],
    }
    if extra:
        report.update(extra)
    return report
