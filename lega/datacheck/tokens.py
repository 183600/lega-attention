"""长度统计、token 流校验与泄漏筛查。"""
from ._shared import _install as _shared_install

_shared_install(globals())

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


def _utf8_byte_length(text: str, chunk_chars: int = 1 << 20) -> int:
    """UTF-8 字节数（按字符边界分块编码，避免对超大语料整串复制）。

    ``len(text.encode("utf-8"))`` 会同时持有原文与编码副本；WikiText-103
    训练集数百 MB，准备阶段再复制一份不必要。按 Python 字符边界切片不会
    拆开多字节字符，因此分块编码的字节数与整串编码完全一致。
    """
    if not text:
        return 0
    if len(text) <= chunk_chars:
        return len(text.encode("utf-8"))
    total = 0
    for i in range(0, len(text), chunk_chars):
        total += len(text[i:i + chunk_chars].encode("utf-8"))
    return int(total)


def text_length_stats(text: str, huge_line_chars: int = 100000) -> dict:
    """逐行长度分布：用于发现异常值（超长行会主导训练窗口）。

    只统计长度，不做删改；极端行的处置留给数据源（本流程不把长行当噪声删除，
    以免引入未记录的人为分布改动）。

    huge_line_chars 是判定身份：bool/小数/字符串经 int() 截断会改变阈值
    （True→1 让所有非空行都算“超长”，40.5→40 改变审计口径），必须先严格
    整数化；负数没有可行语义，也显式拒绝。
    """
    huge_line_chars = strict_int(huge_line_chars, "huge_line_chars")
    if huge_line_chars < 0:
        raise ValueError(f"huge_line_chars 必须 >= 0，收到 {huge_line_chars!r}")
    if not text:
        return {"lines": 0, "chars": 0, "utf8_bytes": 0}
    lens = np.fromiter(_iter_line_lengths(text), dtype=np.int64,
                       count=text.count("\n") + 1)
    return {
        "lines": int(lens.size),
        "chars": int(len(text)),
        # 字节级核算与分词器无关：PPL 是 token 级指标，字符级兜底与 BPE
        # 的 token 数不同，只有 bytes/token 与 UTF-8 字节数能把“分母”
        # 写清楚，便于判断不同分词器/不同语料规模的结果是否可比。
        "utf8_bytes": int(_utf8_byte_length(text)),
        "line_chars_p50": float(np.percentile(lens, 50)),
        "line_chars_p99": float(np.percentile(lens, 99)),
        "line_chars_max": int(lens.max()),
        "lines_gt_huge_threshold": int((lens > int(huge_line_chars)).sum()),
        "huge_line_threshold_chars": int(huge_line_chars),
    }


# ---------------------------------------------------------------- token 流校验

def check_vocab_capacity(vocab_size: int) -> None:
    """token id 以 uint16 存储，词表必须满足 max_id = vocab_size-1 <= 65535。"""
    # 词表大小是整数身份：bool（True→1）、小数（4.9→4）与下划线数字串
    # 都不能被 int() 静默改写成另一个容量后通过 uint16 闸门。
    vocab_size = strict_int(vocab_size, "vocab_size")
    if vocab_size < 1 or vocab_size > 65536:
        raise ValueError(
            f"vocab_size={vocab_size} 无法用 uint16 存储（要求 1..65536）")


def token_stream_stats(ids, vocab_size: int, split: str = "train") -> dict:
    """token 流硬校验：整数、有限、id ∈ [0, vocab)；非法数据直接报错而非静默。

    NaN/Inf 对整数流天然不存在，但对浮点输入（未来若改用连续特征）必须显式
    拒绝——NaN 混入会经 embedding 传播为非法索引或静默取 0。
    """
    # 词表大小是协议身份：bool/小数/下划线数字串不能经 int() 变成另一个
    # 容量，否则越界检查会按与声明不同的词表计算。
    vocab_size = strict_int(vocab_size, "vocab_size")
    if vocab_size < 1:
        raise ValueError(f"vocab_size 必须是正整数，收到 {vocab_size}")
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
    # 至此 vmax <= 65535 由本函数保证，写入 uint16 无损。
    # 词表覆盖/集中度是“数据退化”的早期信号：重复串、模板化或分词失败
    # 会让少数 token 占据绝大多数位置。它们在 token 流层面确定性可算，
    # 与模型无关，随 data_report 落盘，便于跨数据版本比较。
    counts = np.bincount(arr_i, minlength=0)
    total = int(counts.sum())
    observed = counts[counts > 0]
    probs = observed.astype(np.float64) / float(total)
    entropy_bits = float(-np.sum(probs * np.log2(probs))) if probs.size else 0.0
    distinct = int(observed.size)
    singleton_mass = float(counts[counts == 1].sum() / total) if total else 0.0
    top_share = float(observed.max() / total) if total and observed.size else 0.0
    return {"split": str(split), "tokens": int(arr.size), "dtype": str(arr.dtype),
            "min_id": vmin, "max_id": vmax, "nan": n_nan, "inf": n_inf,
            "distinct_ids": distinct,
            "type_token_ratio": float(distinct / total) if total else 0.0,
            "token_entropy_bits": entropy_bits,
            "most_common_token_share": top_share,
            "singleton_token_mass": singleton_mass,
            "vocab_coverage": (float(distinct / int(vocab_size))
                               if int(vocab_size) > 0 else 0.0)}


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

def _window_hashes(ids, n: int, n_samples: int, seed: int, ignore_ids=None):
    """采样 n 长 token 窗口的哈希集合（有放回采样，重复可忽略）。

    id 必须是 [0, 65536) 内的整数（项目 token 流统一 uint16）。旧实现直接
    `astype(np.uint16)`：70000 会环绕成 4464，与合法 token 共享哈希，既可能
    把真实泄漏漏报，也可能把无关窗口误报为重叠。脏数据必须显式拒绝。

    ignore_ids 用于排除含指定 id 的窗口（字符级分词器的 <unk>）：不同未见
    字符都会映射到同一个 unk，若不排除，val/test 的 OOV 长串会被误报成
    跨 split 复制并触发 hard fail。
    """
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)):
        raise ValueError(f"n 必须是正整数，收到 {n!r}")
    n = int(n)
    if n <= 0:
        raise ValueError(f"n 必须是正整数，收到 {n!r}")
    if isinstance(n_samples, bool) or not isinstance(n_samples, (int, np.integer)):
        raise ValueError(f"n_samples 必须是正整数，收到 {n_samples!r}")
    n_samples = int(n_samples)
    if n_samples <= 0:
        raise ValueError(f"n_samples 必须是正整数，收到 {n_samples!r}")
    ids = np.asarray(ids)
    if ids.size:
        if ids.dtype.kind not in "iu":
            raise ValueError(
                f"token id 必须是整数流，收到 dtype={ids.dtype}")
        lo, hi = int(ids.min()), int(ids.max())
        if lo < 0 or hi > 65535:
            raise ValueError(
                f"token id 越界: [{lo}, {hi}] 不在 uint16 token 范围 "
                "[0, 65536)，拒绝静默环绕")
    L = int(ids.size)
    if L < n:
        return set()
    total = L - n + 1
    rng = np.random.default_rng(seed)
    m = min(n_samples, total)
    starts = (rng.integers(0, total, size=m) if m < total
              else np.arange(total))
    windows = np.lib.stride_tricks.sliding_window_view(ids, n)
    picked = np.ascontiguousarray(windows[np.asarray(starts, dtype=np.int64)])
    picked = picked.astype(np.uint16, copy=False)
    if ignore_ids is not None:
        # ignore_ids 必须是可表示的真实 token id：bool/小数/越界值经 int()
        # 或 uint16 转换会变成另一个合法 id，静默排除错误的窗口。漏排除
        # <unk> 会把 OOV 长串误报成泄漏，误排除普通 token 则可能漏掉真实
        # 的跨 split 复制；两者都会污染硬闸门结论。
        try:
            raw_ignore = list(ignore_ids)
        except TypeError as e:
            raise ValueError(
                f"ignore_ids 必须是可迭代的 token id 集合，收到 {ignore_ids!r}"
            ) from e
        ignore_set = set()
        for value in raw_ignore:
            try:
                iv = strict_int(value, "ignore_ids")
            except ValueError as e:
                raise ValueError(
                    f"ignore_ids 中包含非法 token id: {value!r}") from e
            if iv < 0 or iv > 65535:
                raise ValueError(
                    f"ignore_ids 中 token id 越界: {iv} 不在 [0, 65536)")
            ignore_set.add(iv)
        ignore = np.fromiter(ignore_set, dtype=np.uint16)
        if ignore.size:
            picked = picked[~np.any(np.isin(picked, ignore), axis=1)]
    out = set()
    for row in picked:
        out.add(hashlib.blake2b(row.tobytes(), digest_size=16).digest())
    return out


def ngram_overlap(train_ids, eval_ids, n: int = 64, n_samples: int = 100000,
                  seed: int = 0, ignore_ids=None) -> dict:
    """采样长 n-gram 哈希筛查 eval 窗口在 train 中出现的比例（泄漏下界指标）。

    两侧都采样，因此结果是“采样到的最小重叠”，不是精确上界；>0 即说明存在
    跨 split 的逐 token 复制，应人工核查。n=64 足够长，自然语言巧合重叠可忽略。
    n_samples 默认 10 万：该指标是训练前硬闸门（CRITICAL_SPLIT_ALERT_CODES），
    样本太少时会漏掉占比很低但仍属违规的复制段；实测 10 万窗口约 0.1s/侧，
    代价可忽略，覆盖率是 2 万时的 5 倍。
    ignore_ids 见 _window_hashes：字符级 <unk> 不应作为复制证据。
    """
    train_h = _window_hashes(train_ids, n, n_samples, seed,
                             ignore_ids=ignore_ids)
    eval_h = _window_hashes(eval_ids, n, n_samples, seed + 1,
                            ignore_ids=ignore_ids)
    if not eval_h:
        return {"n": int(n),
                "train_windows_sampled": int(len(train_h)),
                "eval_windows_sampled": 0,
                "windows_found_in_train_sample": 0,
                "overlap_rate": None,
                "screening": True}
    hits = len(train_h & eval_h)
    return {"n": int(n),
            "train_windows_sampled": int(len(train_h)),
            "eval_windows_sampled": int(len(eval_h)),
            "windows_found_in_train_sample": int(hits),
            "overlap_rate": float(hits / len(eval_h)),
            "screening": True}


# 短 n-gram 筛查的默认长度：64-gram（ngram_overlap）是“必须拦截”的强证据，
# 但短于 64 token 的复制（整句/段落级复用，尤其模板化文本）会完全漏检。
# 短 n 更易被自然语言常见搭配命中，因此只做**告警**、不参与硬闸门。


_EXACT_HASH_BASE1 = 1099511628211
_EXACT_HASH_BASE2 = 0x9E3779B185EBCA87
_EXACT_HASH_MASK = (1 << 64) - 1


def _odd_mod_inverse(base: int) -> int:
    if base % 2 == 0:
        raise ValueError(f"滚动哈希基数必须是奇数，收到 {base}")
    inv = 1
    for _ in range(6):
        inv = (inv * (2 - base * inv)) & _EXACT_HASH_MASK
    return inv


def _rolling_window_hashes(ids, n: int, base: int) -> np.ndarray:
    values = np.asarray(ids, dtype=np.uint64)
    length = int(values.size)
    count = length - n + 1
    if count <= 0:
        return np.zeros(0, dtype=np.uint64)
    inv_base = np.uint64(_odd_mod_inverse(base))
    base = np.uint64(base)
    inv_pows = np.empty(length, dtype=np.uint64)
    inv_pows[0] = 1
    if length > 1:
        inv_pows[1:] = np.multiply.accumulate(
            np.full(length - 1, inv_base, dtype=np.uint64))
    weighted = np.empty(length + 1, dtype=np.uint64)
    weighted[0] = 0
    weighted[1:] = np.add.accumulate(values * inv_pows)
    base_pows = np.empty(length + 1, dtype=np.uint64)
    base_pows[0] = 1
    base_pows[1:] = np.multiply.accumulate(
        np.full(length, base, dtype=np.uint64))
    prefix = np.empty(length + 1, dtype=np.uint64)
    prefix[0] = 0
    prefix[1:] = base_pows[:length] * weighted[1:]
    return prefix[n:] - base_pows[n] * prefix[:count]


def _pair_window_hashes(ids, n: int) -> tuple:
    return (_rolling_window_hashes(ids, n, _EXACT_HASH_BASE1),
            _rolling_window_hashes(ids, n, _EXACT_HASH_BASE2))


def ngram_overlap_exact(reference_ids, eval_ids, n: int = 64,
                        max_eval_windows: int = 2_000_000,
                        ignore_ids=None, chunk_tokens: int = 1_000_000,
                        reference_split: str = "train") -> dict:
    """全量扫描 reference split 的 n-gram，精确判定 eval 窗口是否出现在其中。

    与 ``ngram_overlap`` 的双侧采样不同，本函数对 reference 的所有 n-gram 建
    双 64-bit 滚动指纹，并对候选命中再做逐字节复核；因此结果不是采样下界。
    eval 窗口默认全查，只有超过 ``max_eval_windows`` 时才等间隔抽样并在
    ``screening=True`` 中显式说明，避免把抽样结果误读为全量结论。

    ``reference_split`` 只用于报告标注，说明本次全量基准是 train 还是 val
    （val/test 互查时基准是 val）。``ignore_ids`` 与 ``_window_hashes`` 同义：
    含这些 id 的 eval 窗口不计入分母（字符级 <unk> 的 OOV 长串不是复制证据）。
    返回字段全是 JSON 原生类型，可直接写入 data_report。
    """
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)):
        raise ValueError(f"n 必须是正整数，收到 {n!r}")
    n = int(n)
    if n <= 0:
        raise ValueError(f"n 必须是正整数，收到 {n!r}")
    for name, value in (("max_eval_windows", max_eval_windows),
                        ("chunk_tokens", chunk_tokens)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} 必须是正整数，收到 {value!r}")
        if int(value) <= 0:
            raise ValueError(f"{name} 必须是正整数，收到 {value!r}")
    max_eval_windows = int(max_eval_windows)
    chunk_tokens = int(chunk_tokens)
    if reference_split not in ("train", "val", "test"):
        raise ValueError(
            f"reference_split 必须是 train/val/test，收到 {reference_split!r}")

    def _as_token_array(ids, label):
        arr = np.asarray(ids)
        if arr.ndim != 1:
            raise ValueError(f"{label} 必须是一维 token 流")
        if arr.size:
            if arr.dtype.kind not in "iu":
                raise ValueError(
                    f"{label} token id 必须是整数流，收到 dtype={arr.dtype}")
            lo, hi = int(arr.min()), int(arr.max())
            if lo < 0 or hi > 65535:
                raise ValueError(
                    f"{label} token id 越界: [{lo}, {hi}] 不在 uint16 token "
                    "范围 [0, 65536)，拒绝静默环绕")
        return arr

    reference_ids = _as_token_array(reference_ids, reference_split)
    eval_ids = _as_token_array(eval_ids, "eval")
    ignore_set = set()
    if ignore_ids is not None:
        try:
            raw_ignore = list(ignore_ids)
        except TypeError as e:
            raise ValueError(
                f"ignore_ids 必须是可迭代的 token id 集合，收到 {ignore_ids!r}"
            ) from e
        for value in raw_ignore:
            try:
                iv = strict_int(value, "ignore_ids")
            except ValueError as e:
                raise ValueError(
                    f"ignore_ids 中包含非法 token id: {value!r}") from e
            if iv < 0 or iv > 65535:
                raise ValueError(
                    f"ignore_ids 中 token id 越界: {iv} 不在 [0, 65536)")
            ignore_set.add(iv)

    reference_total = max(0, int(reference_ids.size) - n + 1)
    eval_total = max(0, int(eval_ids.size) - n + 1)
    eval_ignored = 0
    eval_available = eval_total
    if ignore_set and eval_total:
        bad = np.isin(eval_ids, np.fromiter(ignore_set, dtype=np.uint64))
        csum = np.empty(int(eval_ids.size) + 1, dtype=np.int64)
        csum[0] = 0
        np.cumsum(bad, dtype=np.int64, out=csum[1:])
        valid = (csum[n:] - csum[:-n]) == 0
        eval_available = int(valid.sum())
        eval_ignored = eval_total - eval_available
        starts = np.flatnonzero(valid)
    else:
        starts = np.arange(eval_total, dtype=np.int64)

    sampled = eval_available > max_eval_windows
    if sampled:
        picked = np.linspace(0, eval_available - 1,
                             num=max_eval_windows).astype(np.int64)
        starts = starts[picked]

    checked = int(starts.size)
    hits = np.zeros(checked, dtype=bool)
    if checked and reference_total:
        all_h1, all_h2 = _pair_window_hashes(eval_ids, n)
        eval_h1, eval_h2 = all_h1[starts], all_h2[starts]
        order = np.argsort(eval_h1, kind="mergesort")
        sorted_h1 = eval_h1[order]
        sorted_h2 = eval_h2[order]
        resolved = set()
        for begin in range(0, int(reference_ids.size), chunk_tokens):
            end = min(int(reference_ids.size), begin + chunk_tokens + n - 1)
            if end - begin < n:
                break
            block = reference_ids[begin:end]
            block_h1, block_h2 = _pair_window_hashes(block, n)
            candidate = np.isin(block_h1, sorted_h1)
            if not candidate.any():
                continue
            positions = np.flatnonzero(candidate)
            pairs = np.stack((block_h1[positions], block_h2[positions]),
                             axis=1)
            unique_pairs, first = np.unique(
                pairs, axis=0, return_index=True)
            for row, first_idx in zip(unique_pairs, first):
                key = (int(row[0]), int(row[1]))
                if key in resolved:
                    continue
                resolved.add(key)
                lo = int(np.searchsorted(sorted_h1, row[0], side="left"))
                hi = int(np.searchsorted(sorted_h1, row[0], side="right"))
                if lo >= hi:
                    continue
                matched = lo + np.flatnonzero(sorted_h2[lo:hi] == row[1])
                if not matched.size:
                    continue
                local = int(positions[int(first_idx)])
                window = block[local:local + n]
                for sorted_pos in matched:
                    original = int(order[int(sorted_pos)])
                    start = int(starts[original])
                    if np.array_equal(eval_ids[start:start + n], window):
                        hits[original] = True

    found = int(hits.sum())
    return {
        "n": int(n),
        "reference_split": reference_split,
        "reference_windows_scanned": int(reference_total),
        "eval_windows_total": int(eval_total),
        "eval_windows_excluded_ignore": int(eval_ignored),
        "eval_windows_available": int(eval_available),
        "eval_windows_checked": checked,
        "windows_found_in_reference": found,
        "overlap_rate": (float(found / checked) if checked else None),
        "exact_scan": True,
        "eval_sampling": "evenly_spaced" if sampled else "all",
        "screening": bool(sampled),
        "hash": "rolling_poly64x2+exact_bytes",
    }


def short_ngram_overlap(train_ids, eval_ids, n: int = SHORT_NGRAM_N,
                        n_samples: int = 100000, seed: int = 0,
                        ignore_ids=None) -> dict:
    """短 n-gram 重叠诊断（默认 n=32，告警用）。与 ngram_overlap 同口径。"""
    res = ngram_overlap(train_ids, eval_ids, n=n, n_samples=n_samples, seed=seed,
                        ignore_ids=ignore_ids)
    res["diagnostic_only"] = True
    return res


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
    # 词表大小是协议身份：bool/小数/下划线数字串不能被 int() 截断成另一个
    # 计数区间宽度；负值/零也无法定义 token 分布。token 流以 uint16 存储，
    # 超出 65536 的词表还会让本函数的密集直方图按未声明的容量分配内存
    # （10**12 直接尝试 TB 级分配），必须与 prepare_bins 同一闸门拒绝。
    vocab_size = strict_int(vocab_size, "vocab_size")
    if vocab_size < 1:
        raise ValueError(f"vocab_size 必须是正整数，收到 {vocab_size}")
    check_vocab_capacity(vocab_size)
    if max_samples is None:
        limit = None
    else:
        limit = strict_int(max_samples, "max_samples")
        if limit <= 0:
            raise ValueError(
                f"max_samples 必须是正整数或 None，收到 {max_samples!r}")

    def counts(ids, s):
        arr = np.asarray(ids)
        L = int(arr.size)
        if L == 0:
            return None, False
        # token id 是协议身份：越界 id 会让 minlength 直方图变长，两个 split
        # 长度不一致时在比较处崩溃；若两边恰好同样越界则会被当成合法 token
        # 静默参与 TV/JS。负 id 甚至在 bincount 内抛出难懂的下层错误。
        if arr.dtype.kind == "f":
            if int(np.isnan(arr).sum()) or int(np.isinf(arr).sum()):
                raise ValueError(f"{s} token 流含 NaN/Inf，无法统计分布")
            arr_r = np.rint(arr)
            if not np.array_equal(arr, arr_r):
                raise ValueError(f"{s} token 流含非整数 id，无法统计分布")
            arr = arr_r.astype(np.int64)
        elif arr.dtype.kind not in "iu":
            raise ValueError(f"{s} token 流 dtype 非法: {arr.dtype}")
        lo, hi = int(arr.min()), int(arr.max())
        if lo < 0 or hi >= int(vocab_size):
            raise ValueError(
                f"{s} token id 越界: [{lo}, {hi}] 不在 "
                f"[0, {vocab_size})")
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

# 空行：两个换行之间只允许空白（兼容 CRLF 与“有空格的行”）。

# 时序划分的“实际比例偏离目标”告警线（绝对百分点）。切点对齐文档/段落
# 边界可能把 80/10/10 拉成 70/12/18；这不是错误，但必须显式报告，否则
# “train 占 80%”只是目标值，读者会误以为实际语料也按此规模训练。


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


def _paragraph_positions(text: str):
    """所有空行段落的下一行起点；空行是段落边界，段落是群组单位。"""
    return [m.end() for m in _BLANK_LINE_RE.finditer(text)]


def _nearest_paragraph_in_range(paragraphs, target: int, lo: int, hi: int):
    """[lo, hi] 内离 target 最近（并列取靠前）的空行段落边界；无则 None。"""
    cands = [p for p in paragraphs if lo <= p <= hi]
    if not cands:
        return None
    return min(cands, key=lambda p: (abs(p - target), p))


def temporal_split_text(text: str, train_frac: float = 0.8,
                        val_frac: float = 0.1, window_frac: float = 0.02,
                        min_gap: int = 1):
    """连续文本的时序划分：train=最早、val=其后、test=最新（绝不随机打乱）。

    切点优先落在文档边界（WikiText 标题行 > 空行段落 > 换行），搜索窗口为
    len(text)*window_frac。**只要文本含标题行，就优先在全文范围内选最近的
    标题作为切点**（哪怕超出窗口）——标题定义的“文档”是群组单位，同一文档
    绝不跨 split。无标题但有空行段落时同理：**优先在全文范围内选最近的空行
    段落边界**，段落是群组单位，同一段落绝不跨 split，info 中
    paragraphs_cross_cut=True 表示无法做到（只有“段落长度大于划分区间”这类
    无法两全的语料才会出现）。仅当既无标题也无空行段落时，才退回窗口内的
    换行切分（此时行是语义上的完整单位），找不到任何换行时退化为字符切分
    （info 中 fallback_char_cut=True 提示该文件无法做文档级划分）。

    info["documents_cross_cut"] / info["paragraphs_cross_cut"] 是显式自检：
    为 True 时调用方应告警或拒绝。返回 (train_text, val_text, test_text, info)。
    """
    n = len(text)
    if n < 3:
        raise ValueError("文本太短（<3 字符），无法做 train/val/test 时序划分")
    # 比例必须能保证三个 split 都非空：负值/零/和 >=1/NaN/Inf 都会被
    # max(1,...) 与 min(n-1,...) 静默夹成看似合法的错误切分。
    ratios = {}
    for name, value in (("train_frac", train_frac), ("val_frac", val_frac),
                        ("window_frac", window_frac)):
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{name} 必须是数值比例，收到 bool {value!r}")
        if isinstance(value, str) and "_" in value:
            raise ValueError(f"{name} 必须是数值比例，收到 {value!r}")
        try:
            ratios[name] = float(value)
        except (TypeError, ValueError, OverflowError) as e:
            raise ValueError(f"{name} 必须是数值比例，收到 {value!r}") from e
    if not 0 < ratios["train_frac"] < 1:
        raise ValueError(f"train_frac 必须在 (0,1) 内，收到 {train_frac!r}")
    if not 0 < ratios["val_frac"] < 1:
        raise ValueError(f"val_frac 必须在 (0,1) 内，收到 {val_frac!r}")
    if ratios["train_frac"] + ratios["val_frac"] >= 1:
        raise ValueError("train_frac + val_frac 必须 < 1，否则 test 为空")
    if not 0 < ratios["window_frac"] <= 1:
        raise ValueError(f"window_frac 必须在 (0,1] 内，收到 {window_frac!r}")
    train_frac, val_frac = ratios["train_frac"], ratios["val_frac"]
    window_frac = ratios["window_frac"]
    # min_gap 参与切点算术与文本切片：bool/小数会把它变成 1 或浮点索引
    # （text[p1:p2] 直接 TypeError），负数则允许 val 被压到 0/负长度边界。
    min_gap = strict_int(min_gap, "min_gap")
    if min_gap < 0:
        raise ValueError(f"min_gap 必须 >= 0，收到 {min_gap!r}")
    t1 = max(1, int(n * train_frac))
    t2 = max(t1 + 1, int(n * (train_frac + val_frac)))
    t2 = min(t2, n - 1)
    window = max(256, int(n * window_frac))
    heads = _heading_positions(text)
    paragraphs = [] if heads else _paragraph_positions(text)
    p1, k1 = _pick_boundary(text, t1, window)
    # 标题优先：窗口内没有“可用”标题时在全文中找最近的（范围保证 p1 < t2，
    # val 非空）。窗口内命中的标题若 >= t2 同样不可用，必须换正文范围内的标题。
    if heads and (p1 is None or k1 != "heading" or p1 >= t2):
        h = _nearest_heading_in_range(heads, t1, 1, t2 - 1)
        if h is not None:
            p1, k1 = h, "heading"
    elif paragraphs and (p1 is None or k1 != "blank" or p1 >= t2):
        h = _nearest_paragraph_in_range(paragraphs, t1, 1, t2 - 1)
        if h is not None:
            p1, k1 = h, "blank"
    p2, k2 = _pick_boundary(text, t2, window)
    # 同理：窗口内命中的标题若离 p1 太近（<= p1+min_gap，val 会被压空/切序
    # 倒置），也要换到 [p1+min_gap, n-1] 里最近的标题。
    if heads and (p2 is None or k2 != "heading"
                  or p2 <= (p1 or 0) + min_gap):
        h = _nearest_heading_in_range(heads, t2, (p1 or 0) + min_gap, n - 1)
        if h is not None:
            p2, k2 = h, "heading"
    elif paragraphs and (p2 is None or k2 != "blank"
                         or p2 <= (p1 or 0) + min_gap):
        h = _nearest_paragraph_in_range(paragraphs, t2,
                                        (p1 or 0) + min_gap, n - 1)
        if h is not None:
            p2, k2 = h, "blank"
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
    paragraph_set = set(paragraphs)
    documents_cross = bool(heads) and (p1 not in heading_set
                                       or p2 not in heading_set)
    paragraphs_cross = bool(paragraphs) and (
        p1 not in paragraph_set or p2 not in paragraph_set)
    if heads:
        group_unit = "heading_document"
    elif paragraphs:
        group_unit = "blank_line_paragraph"
    else:
        group_unit = "line_segment"
    # 目标比例 vs 实际比例：切点吸附到文档/段落边界后，实际训练规模可能
    # 偏离协议声明。偏差本身不是泄露，但会让“80/10/10”的规模描述失真，
    # 也可能导致 val/test 过小、检验功效不足，因此逐项落盘并设告警线。
    target_fractions = {
        "train": float(train_frac),
        "val": float(val_frac),
        "test": float(1.0 - train_frac - val_frac),
    }
    actual_fractions = {
        "train": float(len(train_text) / n),
        "val": float(len(val_text) / n),
        "test": float(len(test_text) / n),
    }
    fraction_deviation = {
        k: float(actual_fractions[k] - target_fractions[k])
        for k in target_fractions}
    info = {
        "order": "temporal:train(earliest)->val->test(latest)",
        "cut_offsets": [int(p1), int(p2)], "cut_kinds": [str(k1), str(k2)],
        "target_offsets": [int(t1), int(t2)],
        "cut_deviation_chars": [int(p1 - t1), int(p2 - t2)],
        "window_chars": int(window), "fallback_char_cut": bool(fallback),
        "shuffled": False,
        "target_fractions": target_fractions,
        "actual_fractions": actual_fractions,
        "fraction_deviation": fraction_deviation,
        "max_abs_fraction_deviation": float(
            max(abs(v) for v in fraction_deviation.values())),
        "fraction_tolerance": float(SPLIT_FRACTION_TOLERANCE),
        # 群组自检：标题定义文档、空行定义段落；为 True 表示群组跨了切点
        "n_headings": int(len(heads)),
        "n_paragraph_boundaries": int(len(paragraphs)),
        "documents_cross_cut": bool(documents_cross),
        "paragraphs_cross_cut": bool(paragraphs_cross),
        "group_unit": group_unit,
        "lengths_chars": {"train": len(train_text), "val": len(val_text),
                          "test": len(test_text)},
    }
    return train_text, val_text, test_text, info


