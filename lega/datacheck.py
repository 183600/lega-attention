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
    """
    stats = {
        "replacement_char_replaced_with_space": int(text.count("\ufffd")),
        "nul_deleted": int(text.count("\x00")),
    }
    clean = text.translate(_SANITIZE_TABLE)
    # translate 只删除 NUL/控制符（U+FFFD 是 1:1 替换），长度差即删除数
    stats["control_chars_deleted"] = int(len(text) - len(clean))
    return clean, stats


# ---------------------------------------------------------------- 重复/去污染

def decontaminate_texts(texts: dict, min_chars: int = 40):
    """跨 split 精确重复的长行只从 train 移除；val/test 保持原样（尘封）。

    行内容先 strip；短行（< min_chars，如 WikiText 的 "= References =" 标题）
    不计入去污染，避免大面积误删；但仍会统计其重复数量供审查。
    返回 (cleaned_texts, stats)。
    """
    min_chars = int(min_chars)

    def scan_long(t):
        # 单次扫描：每行只 strip 一次，同时得到长行数与去重后的长行集合
        n_long, seen = 0, set()
        for ln in (t or "").split("\n"):
            s = ln.strip()
            if len(s) >= min_chars:
                n_long += 1
                seen.add(s)
        return n_long, seen

    eval_lines = set()
    eval_counts = {}
    for split in ("val", "test"):
        n_long, seen = scan_long(texts.get(split))
        eval_lines.update(seen)
        eval_counts[split] = (n_long, len(seen))

    kept, dropped = [], 0
    n_train, seen_train = 0, set()
    for ln in texts["train"].split("\n"):
        s = ln.strip()
        if len(s) >= min_chars:
            n_train += 1
            seen_train.add(s)
            if s in eval_lines:
                dropped += 1
                continue
        kept.append(ln)

    out = dict(texts)
    out["train"] = "\n".join(kept)
    stats = {"min_chars": min_chars,
             "train_lines_dropped_as_duplicates_of_eval": int(dropped)}
    for split in ("train", "val", "test"):
        n_long, n_uniq = ((n_train, len(seen_train)) if split == "train"
                          else eval_counts[split])
        stats[f"{split}_long_lines"] = int(n_long)
        stats[f"{split}_intra_split_duplicate_lines"] = int(n_long - n_uniq)
    return out, stats


def text_length_stats(text: str, huge_line_chars: int = 100000) -> dict:
    """逐行长度分布：用于发现异常值（超长行会主导训练窗口）。

    只统计长度，不做删改；极端行的处置留给数据源（本流程不把长行当噪声删除，
    以免引入未记录的人为分布改动）。
    """
    if not text:
        return {"lines": 0, "chars": 0}
    lens = np.fromiter((len(ln) for ln in text.split("\n")), dtype=np.int64,
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
    if vmin < 0 or vmax >= int(vocab_size):
        raise ValueError(
            f"{split} token id 越界: [{vmin}, {vmax}] 不在 [0, {vocab_size})，"
            "数据/词表不一致")
    # 重新扫一遍确认索引在写入 uint16 前无损（vmax 已保证 <= 65535）
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

def _boundary_candidates(seg: str, base: int, kind: str):
    if kind == "heading":
        return [base + m.start() for m in _HEADING_RE.finditer(seg)
                if base + m.start() > 0]
    if kind == "blank":
        return [base + i + 1 for i in range(1, len(seg))
                if seg[i] == "\n" and seg[i - 1] == "\n"]
    return [base + i + 1 for i in range(len(seg)) if seg[i] == "\n"]


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


def temporal_split_text(text: str, train_frac: float = 0.8,
                        val_frac: float = 0.1, window_frac: float = 0.02,
                        min_gap: int = 1):
    """连续文本的时序划分：train=最早、val=其后、test=最新（绝不随机打乱）。

    切点优先落在文档边界（WikiText 标题行 > 空行段落 > 换行），搜索窗口为
    len(text)*window_frac，保证同一段/文档不跨 split。找不到任何换行时退化为
    字符切分（info 中 fallback_char_cut=True 提示该文件无法做文档级划分）。
    返回 (train_text, val_text, test_text, info)。
    """
    n = len(text)
    if n < 3:
        raise ValueError("文本太短（<3 字符），无法做 train/val/test 时序划分")
    t1 = max(1, int(n * float(train_frac)))
    t2 = max(t1 + 1, int(n * (float(train_frac) + float(val_frac))))
    t2 = min(t2, n - 1)
    window = max(256, int(n * float(window_frac)))
    p1, k1 = _pick_boundary(text, t1, window)
    p2, k2 = _pick_boundary(text, t2, window)
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
    info = {
        "order": "temporal:train(earliest)->val->test(latest)",
        "cut_offsets": [int(p1), int(p2)], "cut_kinds": [str(k1), str(k2)],
        "window_chars": int(window), "fallback_char_cut": bool(fallback),
        "shuffled": False,
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

def build_data_report(tokenizer: dict, splits: dict, sanitization: dict,
                      duplicates: dict, contamination: dict,
                      distribution: dict, text_stats: dict = None,
                      extra: dict = None) -> dict:
    """汇总为 data_report.json（只含 JSON 原生类型）。"""
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
        "notes": [
            "所有拟合类统计（字符词表等）只在 train 上完成，val/test 只做 transform；",
            "去污染只从 train 移除与 val/test 精确重复的长行，val/test 不被改写；",
            "64-gram 重叠为采样筛查下界，>0 时应人工核查对应语料；",
            "一元分布漂移为采样估计，用于提示 train/test 分布差异而非统计检验。",
        ],
    }
    if extra:
        report.update(extra)
    return report
