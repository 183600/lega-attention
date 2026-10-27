"""文本清洗、精确/近似去污染。"""
from ._shared import _install as _shared_install

_shared_install(globals())

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


def _line_digest(s: str) -> bytes:
    """128-bit 行摘要（重复检测用；16 字节碰撞概率可忽略）。"""
    return hashlib.blake2b(s.encode("utf-8"), digest_size=16).digest()


def decontaminate_texts(texts: dict, min_chars: int = 40):
    """清理 train 侧重复样本，并诊断 eval 两侧（val/test）的重复。

    三类重复都处理/统计（最小改动原则：只改 train，val/test 保持尘封）：
      * train 与 val/test 的跨 split 精确重复；
      * train 与 val/test 的跨 split 空白归一化重复；
      * train **内部**重复：保留首次出现，删除后续精确/空白归一化重复
        （重复训练样本会放大其权重并增加记忆，属于数据清洗而非评测改写）。
    行内容先 strip；短行（< min_chars，如 WikiText 的 "= References =" 标题）
    不计入去污染，避免大面积误删；但短行的跨 split 复制会单独统计并告警
    （*_overlap_short_lines 与归一化计数），不会成为“既不删除也不报告”的
    盲区。返回 (cleaned_texts, stats)。

    val/test 内容绝不被改写；val 与 test 之间的重复另行统计
    （val_test_overlap_*）并由报告告警——两侧样本重叠会让“用 val 选模型、
    在 test 上最终评估”不再独立。

    比较集合使用 128-bit 摘要（16 字节碰撞概率可忽略），
    train 输出按删除切点拼接未删除片段。
    """
    min_chars = strict_int(min_chars, "min_chars")
    if min_chars < 0:
        raise ValueError(f"min_chars 必须 >= 0，收到 {min_chars!r}")
    # 顶层非对象或缺少 train（旧格式/手改缓存）时按空文本处理，避免一个
    # 损坏的数据缓存让 prepare_data 直接 KeyError/AttributeError。
    texts = texts if isinstance(texts, dict) else {}

    digest = _line_digest

    def scan(t):
        if not isinstance(t, str):
            t = ""
        # 单次扫描：每行只 strip/normalize 一次。长行（>= min_chars）用于
        # 去污染；短行单独收集用于跨 split 重叠诊断（不自动删除）；非空行
        # 的重复都计数——短行不参与去污染，但重复数量仍统计供审查。
        n_long, n_short, n_valid = 0, 0, 0
        seen_long, seen_long_norm = set(), set()
        seen_short, seen_short_norm = set(), set()
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
            # 长短行按归一化后的内容长度分类：同一内容在不同 split 里
            # 只因空白/NFKC 差异导致原始长度跨越阈值时，两侧必须落进同一
            # 集合，否则重复既不会被去污染，也不会出现在短行诊断里。
            if len(sn) >= min_chars:
                n_long += 1
                seen_long.add(d)
                seen_long_norm.add(dn)
            else:
                n_short += 1
                seen_short.add(d)
                seen_short_norm.add(dn)
        return {"n_long": n_long, "n_short": n_short, "long": seen_long,
                "long_norm": seen_long_norm, "short": seen_short,
                "short_norm": seen_short_norm, "n_valid": n_valid,
                "n_uniq": len(seen_valid), "n_uniq_norm": len(seen_valid_norm)}

    eval_lines, eval_norm_lines = set(), set()
    eval_counts, eval_short_counts = {}, {}
    split_lines, split_short = {}, {}
    for split in ("val", "test"):
        st = scan(texts.get(split))
        eval_lines.update(st["long"])
        eval_norm_lines.update(st["long_norm"])
        eval_counts[split] = (st["n_long"], st["n_valid"], st["n_uniq"],
                              st["n_uniq_norm"])
        eval_short_counts[split] = st["n_short"]
        split_lines[split] = (st["long"], st["long_norm"])
        split_short[split] = (st["short"], st["short_norm"])
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
    # 短行（< min_chars）不参与去污染，但跨 split 复制必须能被看见：
    # 单独收集摘要，用于诊断计数与告警。
    seen_short_train, seen_short_train_norm = set(), set()
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
            if len(sn) < min_chars:
                seen_short_train.add(d)
                seen_short_train_norm.add(dn)
        drop = False
        if s and len(sn) >= min_chars:
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
    short_pairs = {
        "train_val": (seen_short_train, seen_short_train_norm,
                      split_short["val"][0], split_short["val"][1],
                      eval_short_counts["val"]),
        "train_test": (seen_short_train, seen_short_train_norm,
                       split_short["test"][0], split_short["test"][1],
                       eval_short_counts["test"]),
        "val_test": (split_short["val"][0], split_short["val"][1],
                     split_short["test"][0], split_short["test"][1],
                     eval_short_counts["test"]),
    }
    for key, (a, an, b, bn, denom) in short_pairs.items():
        exact = len(a & b)
        stats[f"{key}_overlap_short_lines"] = int(exact)
        stats[f"{key}_overlap_normalized_short_lines"] = int(len(an & bn))
        stats[f"{key}_short_overlap_rate"] = float(exact / max(1, denom))
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
    # 块定义（至少几行才算“多行块”）必须是真实整数：bool/小数若被
    # _iter_blocks 里的 int() 静默截断，会把单行块当成多行块删除，
    # 去污染统计与 val/test 保持原样的承诺随之失真。
    min_lines = strict_int(min_lines, "min_lines")
    if min_lines < 1:
        raise ValueError(f"min_lines 必须 >= 1，收到 {min_lines}")
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


# ---------------------------------------------------------------- 近似重复
#
# 精确行/块去重只能抓住逐字节（或仅空白/NFKC 差异）的复制。真实语料里更常见
# 的是“改了几个词/删了一段”的近似复制：它同样会让 train 记住 val/test 的
# 内容，也会在 train 内部放大同一段文本的权重。这里用字符 5-gram 的
# MinHash + LSH 做近似重复筛查，再对候选对精确计算 Jaccard/containment。
#
# 口径：
#   * 只从 train 删除；val/test 逐字节不动（与精确去污染一致）；
#   * 跨 split 用 max(Jaccard, containment)：val 的一小段被整段抄进更长的
#     train 文档时，Jaccard 会被长度稀释，containment 才能发现；
#   * train 内部只用 Jaccard，避免“短段落被长文档包含”导致过度删除；
#   * 阈值是协议参数而不是从数据估计出来的；报告同时给出 0.70/阈值/0.90
#     三档敏感性，读者能看出删除量对阈值的依赖。
# 近似重复扫描的“可审计覆盖率”下限：低于该比例时，报告不能暗示全语料都已
# 做过近重复筛查。短块（标题/固定用语）与超大块被跳过是协议选择，但跳过了
# 多少字符必须显式可见；覆盖率不足只告警、不自动改阈值，避免按结果挑参数。
# 包含关系（containment）候选的确定性锚点数。MinHash/LSH 估计的是
# Jaccard：短 val 段被整段抄进长 train 文档时 Jaccard 很低，band 不会碰撞，
# 但 containment=1。这里从每个块的 shingle 集合中均匀取少量真实 shingle
# 作为锚点；扫描另一侧时统计命中的锚点数，达到阈值比例即进入候选，再由
# 精确 Jaccard/containment 裁决。候选阶段只需高召回，精确阶段保证不误删。



def _near_dup_normalize(text: str) -> str:
    """近似重复比较键：NFKC + casefold + 空白折叠。

    只做确定性的等价归一化，不做同义词/词干替换：那会把语义不同的文本
    判成重复。保留标点与词序，使 5-gram 仍能区分“同一段但改了关键否定词”。
    """
    s = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(s.split())


def _mix64(x):
    """SplitMix64 的向量化混合；输入/输出均按 uint64 环绕。"""
    x = x + _U64_GOLDEN
    x = (x ^ (x >> _U64(30))) * _U64_MIX1
    x = (x ^ (x >> _U64(27))) * _U64_MIX2
    return x ^ (x >> _U64(31))


def _near_dup_shingles(norm: str, n: int = NEAR_DUP_SHINGLE):
    """归一化文本的 UTF-8 字节 n-gram 哈希（排序去重后的 uint64 数组）。

    用字节而不是字符：多语言语料无需分词器，CJK/英文/符号走同一条确定性
    路径；NFKC 已把兼容字符折叠，字节边界不会改变同一文本的哈希口径。
    """
    b = np.frombuffer(norm.encode("utf-8"), dtype=np.uint8).astype(np.uint64)
    if b.size < n:
        return np.zeros(0, dtype=np.uint64)
    with np.errstate(over="ignore"):
        h = b[: b.size - n + 1].copy()
        for k in range(1, n):
            h *= _U64_BASE
            h += b[k: b.size - n + 1 + k]
        h = _mix64(h)
    return np.unique(h)


def _near_dup_signature(hashes, n_hashes: int = NEAR_DUP_HASHES):
    """MinHash 签名：每个 hash 函数取 shingle 哈希的最小值。

    18 个独立混合种子、6 band × 3 row：J=0.80 时至少一个 band 相同的概率为
    1-(1-0.80³)^6≈98.7%，漏检只发生在低相似度一侧，且会在报告里给出
    该理论召回率，而不是把 LSH 当作精确算法。
    """
    sig = np.empty(int(n_hashes), dtype=np.uint64)
    for i in range(int(n_hashes)):
        seed = _U64(((i + 1) * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF)
        sig[i] = _mix64(hashes ^ seed).min()
    return sig


def _near_dup_band_key(signature, band: int):
    """一个 LSH band 的桶键：必须使用该 band 的全部 ``NEAR_DUP_ROWS`` 行。

    LSH 的候选召回率公式 ``1-(1-J^rows)^bands`` 与报告字段都按 rows 行计算；
    桶键若少用一行，实际行为就退化为更小的 rows（召回更高、候选更多，但
    与声明的签名参数/理论召回率不一致）。这里集中构造键，避免扫描循环与
    报告口径再次漂移。
    """
    i0 = int(band) * NEAR_DUP_ROWS
    return tuple(int(signature[i0 + j]) for j in range(NEAR_DUP_ROWS))


def _near_dup_anchor_ids(hashes) -> np.ndarray:
    """从去重 shingle 集合中均匀取少量真实 shingle 作为包含候选锚点。"""
    n = int(hashes.size)
    if n <= 0:
        return np.zeros(0, dtype=np.uint64)
    k = min(int(NEAR_DUP_ANCHORS), n)
    if k == 1:
        idx = np.zeros(1, dtype=np.int64)
    else:
        idx = np.unique(np.round(np.linspace(0, n - 1, k)).astype(np.int64))
    return np.asarray(hashes[idx], dtype=np.uint64)


def _near_dup_build_anchor_index(metas, threshold: float):
    """由块元数据里的 ``_anchors`` 建倒排索引，返回 (index, required)。

    锚点是块内真实存在的 shingle 哈希，因此“块 A 被块 B 完整包含”时，
    A 的每个锚点都会出现在 B 的 shingle 集合里；扫描 B 的 shingle 即可
    发现候选。required 是每个块需要命中的锚点数（阈值的宽松下界），只用于
    候选阶段；最终是否删除仍由精确 Jaccard/containment 决定。
    """
    by_key = {}
    required = {}
    for idx, m in enumerate(metas):
        anchors = m.pop("_anchors", None)
        if anchors is None or int(anchors.size) == 0:
            continue
        # 阈值的 0.5 倍：候选阶段偏向高召回（近阈值副本可能只命中部分
        # 锚点），避免把“近似重复”漏成“没有候选”。精确阶段仍按原阈值。
        req = max(1, int(math.floor(NEAR_DUP_ANCHORS * float(threshold)
                                      * 0.5)))
        req = min(req, int(anchors.size))
        required[idx] = req
        for h in anchors.tolist():
            by_key.setdefault(int(h), []).append(idx)
    keys = np.asarray(sorted(by_key), dtype=np.uint64)
    ids_by_key = [tuple(by_key[int(k)]) for k in keys.tolist()]
    return (keys, ids_by_key), required


def _near_dup_anchor_hits(hashes, anchor_index) -> dict:
    """返回 {块索引: 命中锚点数}；hashes 为该块去重后的 shingle 集合。"""
    keys, ids_by_key = anchor_index
    if hashes.size == 0 or keys.size == 0:
        return {}
    pos = np.searchsorted(keys, hashes)
    pos_clip = np.clip(pos, 0, keys.size - 1)
    hit = keys[pos_clip] == hashes
    if not bool(hit.any()):
        return {}
    counts = {}
    for j in pos_clip[hit].tolist():
        for idx in ids_by_key[j]:
            counts[idx] = counts.get(idx, 0) + 1
    return counts


def _near_dup_similarity(a, b):
    """返回 (Jaccard, containment)；两者都基于去重后的 shingle 集合。"""
    if a.size == 0 or b.size == 0:
        return 0.0, 0.0
    inter = int(np.intersect1d(a, b, assume_unique=True).size)
    union = int(a.size) + int(b.size) - inter
    jac = float(inter / union) if union > 0 else 0.0
    denom = min(int(a.size), int(b.size))
    con = float(inter / denom) if denom > 0 else 0.0
    return jac, con


def _near_dup_collect(text: str, split: str, min_chars: int,
                      min_shingles: int, max_chars: int,
                      collect_anchors: bool = False,
                      anchor_index=None):
    """把文本切成段落块并计算签名；返回 (metadata, signatures, counts)。

    counts 同时记录块数与字符数：只报“扫描了 N 个块”无法说明这些块覆盖了
    语料的多少字符；短块/超大块被跳过的影响必须用字符覆盖率量化。
    """
    metas, sigs = [], []
    counts = {"scanned": 0, "short": 0, "few_shingles": 0, "too_large": 0,
              "chars_scanned": 0, "chars_short": 0,
              "chars_few_shingles": 0, "chars_too_large": 0,
              "chars_eligible": 0}
    for start, end, _digest, n_lines in _iter_blocks(text, min_lines=1):
        block_chars = int(end - start)
        counts["scanned"] += 1
        counts["chars_scanned"] += block_chars
        if end - start > int(max_chars):
            counts["too_large"] += 1
            counts["chars_too_large"] += block_chars
            continue
        norm = _near_dup_normalize(text[start:end])
        if len(norm) < int(min_chars):
            counts["short"] += 1
            counts["chars_short"] += block_chars
            continue
        hashes = _near_dup_shingles(norm)
        if hashes.size < int(min_shingles):
            counts["few_shingles"] += 1
            counts["chars_few_shingles"] += block_chars
            continue
        meta = {"split": str(split), "start": int(start),
                "end": int(end), "lines": int(n_lines)}
        counts["chars_eligible"] += block_chars
        if collect_anchors:
            meta["_anchors"] = _near_dup_anchor_ids(hashes)
        if anchor_index is not None:
            hits = _near_dup_anchor_hits(hashes, anchor_index)
            if hits:
                meta["_anchor_hits"] = hits
        metas.append(meta)
        sigs.append(_near_dup_signature(hashes))
    return metas, sigs, counts


def _near_dup_pair_type(split_a: str, split_b: str) -> str:
    """把无序 split 对映射到报告字段名（train/val/test 为封闭枚举）。

    按协议顺序而不是字典序排序：字典序会把 val/test 写成 ``test_val``，
    与报告里固定的 ``val_test`` 字段/告警查表不一致。
    """
    order = {"train": 0, "val": 1, "test": 2}
    a, b = sorted((str(split_a), str(split_b)),
                  key=lambda x: (order.get(x, 99), x))
    return f"{a}_{b}"


def decontaminate_near_duplicates(texts: dict,
                                  threshold: float = NEAR_DUP_THRESHOLD,
                                  min_chars: int = NEAR_DUP_MIN_CHARS,
                                  min_shingles: int = NEAR_DUP_MIN_SHINGLES):
    """近似重复去污染：只改写 train，val/test 逐字节保持原样。

    返回 (cleaned_texts, stats)。统计写入 data_report.json 的
    ``duplicates.near_duplicate``，包括候选/核实对数、删除行数、三档阈值
    敏感性、LSH 理论召回率与截断标记。任何截断（超大块、超大候选簇、
    候选总数上限）都显式记录，避免把“没扫到”写成“没有近似重复”。
    """
    if isinstance(threshold, bool):
        raise ValueError(f"threshold 必须是 (0,1] 内的数值，收到 {threshold!r}")
    try:
        threshold = float(threshold)
    except (TypeError, ValueError, OverflowError) as e:
        raise ValueError(
            f"threshold 必须是 (0,1] 内的数值，收到 {threshold!r}") from e
    if not math.isfinite(threshold) or not 0.0 < threshold <= 1.0:
        raise ValueError(f"threshold 必须是 (0,1] 内的数值，收到 {threshold!r}")
    min_chars = strict_int(min_chars, "min_chars")
    min_shingles = strict_int(min_shingles, "min_shingles")
    if min_chars < 1:
        raise ValueError(f"min_chars 必须 >= 1，收到 {min_chars}")
    if min_shingles < 1:
        raise ValueError(f"min_shingles 必须 >= 1，收到 {min_shingles}")

    texts = texts if isinstance(texts, dict) else {}
    out = dict(texts)
    train_text = texts.get("train")
    if not isinstance(train_text, str):
        train_text = ""
        if "train" in out:
            # 与 decontaminate_documents 一致：损坏的 train 字段按空文本处理，
            # 不把非字符串原样传给下游编码器。
            out["train"] = ""

    eval_metas, eval_sigs = [], []
    eval_counts = {}
    for split in ("val", "test"):
        t = texts.get(split)
        if not isinstance(t, str) or not t:
            eval_counts[split] = {"scanned": 0, "short": 0,
                                  "few_shingles": 0, "too_large": 0,
                                  "eligible": 0}
            continue
        metas, sigs, counts = _near_dup_collect(
            t, split, min_chars, min_shingles, NEAR_DUP_MAX_BLOCK_CHARS,
            collect_anchors=True)
        eval_counts[split] = dict(counts, eligible=len(metas))
        eval_metas.extend(metas)
        eval_sigs.extend(sigs)

    # 包含关系候选：MinHash 只估计 Jaccard。先用 eval 块的锚点建倒排索引，
    # 扫描 train 块可发现“短 eval 段被长 train 文档整体包含”（Jaccard 低、
    # containment 高）的副本；train 锚点索引稍后反向扫描 eval 块。
    eval_anchor_index, eval_anchor_required = _near_dup_build_anchor_index(
        eval_metas, threshold)
    train_metas, train_sigs, train_counts = _near_dup_collect(
        train_text, "train", min_chars, min_shingles, NEAR_DUP_MAX_BLOCK_CHARS,
        collect_anchors=True, anchor_index=eval_anchor_index)
    train_anchor_index, train_anchor_required = _near_dup_build_anchor_index(
        train_metas, threshold)
    n_eval = len(eval_metas)
    n_train = len(train_metas)

    # 以 4 位小数的显示键去重：threshold 恰好/极接近 0.70 或 0.90 时，
    # 若按键分组后仍对多个浮点值循环，会把同一候选对重复计数。
    sens_values = {}
    for t in (0.70, float(threshold), 0.90):
        sens_values[str(round(float(t), 4))] = float(t)
    sensitivity = sorted(sens_values.values())
    sens_keys = {v: k for k, v in sens_values.items()}
    pair_types = ("train_val", "train_test", "train_train", "val_test")
    sens_counts = {sens_keys[t]: {k: 0 for k in pair_types}
                   for t in sensitivity}
    # 每个敏感性阈值下“实际会删掉哪些块”。只报候选/判定对数无法回答
    # “阈值从 0.7 调到 0.9，训练集少多少字符”，也无法与 removal_fraction
    # 对照；这里逐阈值维护删除集合，最后统一换算为块/行/字符量。
    sens_removal = {
        sens_keys[t]: {"cross": set(), "intra": set()}
        for t in sensitivity}
    sens_removal_stats = {
        sens_keys[t]: {
            "train_blocks_removed_cross_split": 0,
            "train_blocks_removed_intra_split": 0,
            "train_blocks_removed_total": 0,
            "train_lines_removed": 0,
            "train_chars_removed": 0,
            "removal_fraction_chars": 0.0,
        } for t in sensitivity}
    max_score = 0.0
    top_examples = []
    candidate_pairs = 0
    containment_candidate_pairs = 0
    truncated_groups = 0
    truncated_candidates = False
    prefilter_skipped = 0
    verified_pairs = 0

    # 只要有至少两个块可比就运行：train 为空时仍需做 val-test 诊断；
    # train 只有 1 块且无 eval 时没有任何 pair，可跳过。
    if (n_train + n_eval) >= 2 and (n_train > 0 or n_eval > 1):
        eval_arr = (np.asarray(eval_sigs, dtype=np.uint64)
                    if eval_sigs else np.zeros(
                        (0, NEAR_DUP_HASHES), dtype=np.uint64))
        train_arr = (np.asarray(train_sigs, dtype=np.uint64)
                     if train_sigs else np.zeros(
                         (0, NEAR_DUP_HASHES), dtype=np.uint64))
        sigs = np.vstack([eval_arr, train_arr])
        metas = eval_metas + train_metas
        n_all = len(metas)
        # LSH：逐 band 建桶再丢弃，内存只与一个 band 的块数有关。
        candidates = set()
        for band in range(NEAR_DUP_BANDS):
            buckets = {}
            for i in range(n_all):
                key = _near_dup_band_key(sigs[i], band)
                buckets.setdefault(key, []).append(i)
            for ids in buckets.values():
                if len(ids) < 2:
                    continue
                if len(ids) > NEAR_DUP_MAX_GROUP:
                    truncated_groups += 1
                    ids = ids[:NEAR_DUP_MAX_GROUP]
                for x in range(len(ids)):
                    for y in range(x + 1, len(ids)):
                        a, b = ids[x], ids[y]
                        # val-val / test-test 的内部近似重复与跨 split 去污染
                        # 无关，保留它们只会放大候选集；val-test 仍需保留诊断。
                        if (metas[a]["split"] == metas[b]["split"]
                                and metas[a]["split"] != "train"):
                            continue
                        pair = (a, b) if a < b else (b, a)
                        if pair in candidates:
                            continue
                        candidates.add(pair)
                        if len(candidates) >= NEAR_DUP_MAX_CANDIDATES:
                            truncated_candidates = True
                            break
                    if truncated_candidates:
                        break
                if truncated_candidates:
                    break
            if truncated_candidates:
                break
        # 只对候选对做精确相似度；缓存 shingle 集合，超过上限时清空重算。
        cache = {}

        def _hashes(idx):
            arr = cache.get(idx)
            if arr is None:
                m = metas[idx]
                norm = _near_dup_normalize(
                    texts[m["split"]][m["start"]:m["end"]])
                arr = _near_dup_shingles(norm)
                if len(cache) >= NEAR_DUP_CACHE_BLOCKS:
                    cache.clear()
                cache[idx] = arr
            return arr

        # 包含关系候选（Jaccard-LSH 无法覆盖）：train 块里命中 eval 锚点，
        # 或 eval 块里命中 train/另一 eval 块锚点。只进入候选集，是否删除
        # 仍由下方精确 max(Jaccard, containment) 按协议阈值决定。
        anchor_candidates = set()

        def _add_anchor_candidate(a, b):
            nonlocal truncated_candidates
            if a == b:
                return
            pair = (a, b) if a < b else (b, a)
            anchor_candidates.add(pair)
            if pair in candidates:
                return
            if len(candidates) >= NEAR_DUP_MAX_CANDIDATES:
                truncated_candidates = True
                return
            candidates.add(pair)

        # train 块命中 eval 锚点：覆盖“短 eval 段被长 train 文档包含”。
        for ti, m in enumerate(train_metas):
            hits = m.pop("_anchor_hits", None)
            if not hits or truncated_candidates:
                continue
            for ei, count in hits.items():
                if count >= eval_anchor_required.get(ei, 1):
                    _add_anchor_candidate(ei, n_eval + ti)
        # eval 块命中 train 锚点（短 train 段被长 eval 文档包含）以及
        # eval 内部锚点（val-test containment 诊断）。train 为空时后者
        # 仍必须运行，不能把 val/test 诊断错误地绑定到 train 是否存在。
        if not truncated_candidates and (n_train or n_eval > 1):
            for ei, m in enumerate(eval_metas):
                if truncated_candidates:
                    break
                hits = _near_dup_anchor_hits(_hashes(ei), train_anchor_index)
                for ti, count in hits.items():
                    if count >= train_anchor_required.get(ti, 1):
                        _add_anchor_candidate(ei, n_eval + ti)
                if n_eval > 1:
                    hits = _near_dup_anchor_hits(_hashes(ei), eval_anchor_index)
                    for ej, count in hits.items():
                        if ej == ei:
                            continue
                        if count < eval_anchor_required.get(ej, 1):
                            continue
                        if (metas[ei]["split"] == metas[ej]["split"]
                                and metas[ei]["split"] != "train"):
                            continue
                        _add_anchor_candidate(ei, ej)
        containment_candidate_pairs = len(anchor_candidates)
        candidate_pairs = len(candidates)

        remove_cross, remove_intra = set(), set()
        # 签名预筛：18 个 MinHash 的相等比例是 Jaccard/containment 的无偏
        # 估计（SE≈0.1）。远低于阈值的候选对直接跳过，避免在模板化/重复
        # 语料上把预算耗在精确哈希上。地板按**敏感性网格的最低阈值**而不是
        # 当前协议阈值计算：否则把阈值设得较高时，低阈值那一档的“若放宽会
        # 删多少”会因预筛漏掉候选而被低估。跳过的对数单列在报告里。
        prefilter_floor = float(min(sensitivity)) * 0.6
        for a, b in candidates:
            ma, mb = metas[a], metas[b]
            # 锚点候选由 containment 召回：MinHash 相等比例估计的是
            # Jaccard，对“短段被长文包含”会远低于阈值，不能拿它预筛。
            if (a, b) not in anchor_candidates:
                est = float(np.mean(sigs[a] == sigs[b]))
                if est < prefilter_floor:
                    prefilter_skipped += 1
                    continue
            jac, con = _near_dup_similarity(_hashes(a), _hashes(b))
            if ma["split"] == "train" and mb["split"] == "train":
                score = jac
            else:
                score = max(jac, con)
            if score > max_score:
                max_score = score
            ptype = _near_dup_pair_type(ma["split"], mb["split"])
            # 先确定该对“若达阈值会删哪一侧”，再按每个敏感性阈值登记。
            # 跨 split 只删 train 侧；train 内部只删后出现的副本。
            train_idx = later = None
            if ptype in ("train_val", "train_test"):
                train_idx = a if ma["split"] == "train" else b
            elif ptype == "train_train":
                later = a if (ma["start"], ma["end"]) > (
                    mb["start"], mb["end"]) else b
            for t in sensitivity:
                if score >= t:
                    key = sens_keys[t]
                    sens_counts[key][ptype] += 1
                    if train_idx is not None:
                        sens_removal[key]["cross"].add(train_idx)
                    elif later is not None:
                        sens_removal[key]["intra"].add(later)
            if score >= threshold:
                verified_pairs += 1
                if train_idx is not None:
                    remove_cross.add(train_idx)
                elif later is not None:
                    remove_intra.add(later)
                if len(top_examples) < 10 or score > top_examples[-1]["score"]:
                    top_examples.append({
                        "pair": ptype, "jaccard": float(jac),
                        "containment": float(con), "score": float(score),
                        "chars_a": int(ma["end"] - ma["start"]),
                        "chars_b": int(mb["end"] - mb["start"])})
                    top_examples.sort(key=lambda r: -r["score"])
                    del top_examples[10:]
        remove_all = remove_cross | remove_intra
        if remove_all:
            ranges = [(metas[i]["start"], metas[i]["end"])
                      for i in sorted(remove_all)]
            out["train"] = _remove_ranges(train_text, ranges)
        dropped_lines = sum(int(metas[i]["lines"]) for i in remove_all)
        dropped_chars = sum(int(metas[i]["end"] - metas[i]["start"])
                            for i in remove_all)
        for t in sensitivity:
            key = sens_keys[t]
            cross = sens_removal[key]["cross"]
            intra = sens_removal[key]["intra"] - cross
            removed = cross | intra
            removed_chars = sum(
                int(metas[i]["end"] - metas[i]["start"]) for i in removed)
            sens_removal_stats[key] = {
                "train_blocks_removed_cross_split": int(len(cross)),
                "train_blocks_removed_intra_split": int(len(intra)),
                "train_blocks_removed_total": int(len(removed)),
                "train_lines_removed": int(sum(
                    int(metas[i]["lines"]) for i in removed)),
                "train_chars_removed": int(removed_chars),
                "removal_fraction_chars": float(
                    removed_chars / max(1, len(train_text))),
            }
    else:
        remove_cross, remove_intra = set(), set()
        dropped_lines = 0
        dropped_chars = 0

    def _eval_count(split, key):
        return int(eval_counts.get(split, {}).get(key, 0))

    train_chars_total = int(len(train_text))
    train_chars_eligible = int(train_counts["chars_eligible"])
    train_chars_scanned = int(train_counts["chars_scanned"])
    stats = {
        "method": "minhash_lsh_exact_verify",
        "similarity": ("max(jaccard, containment) for cross-split; "
                       "jaccard for train-internal"),
        "shingle_size": int(NEAR_DUP_SHINGLE),
        "shingle_unit": "utf-8 byte n-grams after NFKC+casefold+whitespace",
        "threshold": float(threshold),
        "min_chars": int(min_chars),
        "min_shingles": int(min_shingles),
        "max_block_chars": int(NEAR_DUP_MAX_BLOCK_CHARS),
        "signature_hashes": int(NEAR_DUP_HASHES),
        "lsh_bands": int(NEAR_DUP_BANDS),
        "lsh_rows": int(NEAR_DUP_ROWS),
        "lsh_recall_at_threshold": float(
            1.0 - (1.0 - threshold ** NEAR_DUP_ROWS) ** NEAR_DUP_BANDS),
        "threshold_sensitivity": sens_counts,
        # 与 threshold_sensitivity（判定对数）互补：实际删除的块/行/字符量。
        # 阈值升高会同时减少“判为近重复的对”和“被删掉的训练数据”，两者
        # 不能互相替代；只报对数会让读者误以为删除量也随之线性变化。
        "threshold_sensitivity_removal": sens_removal_stats,
        "train_blocks_scanned": int(train_counts["scanned"]),
        "train_blocks_eligible": int(n_train),
        "train_blocks_skipped_short": int(train_counts["short"]),
        "train_blocks_skipped_few_shingles": int(train_counts["few_shingles"]),
        "train_blocks_skipped_too_large": int(train_counts["too_large"]),
        # 覆盖率：短块/超大块会被跳过，因此“没有近重复”只在已扫描的
        # 字符范围内成立。把分母写清楚，避免把未扫描部分当成已验证。
        "train_chars_scanned": train_chars_scanned,
        "train_chars_skipped_short": int(train_counts["chars_short"]),
        "train_chars_skipped_few_shingles": int(
            train_counts["chars_few_shingles"]),
        "train_chars_skipped_too_large": int(train_counts["chars_too_large"]),
        "train_chars_coverage_fraction": float(
            train_chars_eligible / max(1, train_chars_total)),
        "train_scanned_block_char_coverage_fraction": float(
            train_chars_eligible / max(1, train_chars_scanned)),
        "near_duplicate_min_coverage": float(NEAR_DUP_MIN_COVERAGE),
        "val_blocks_scanned": _eval_count("val", "scanned"),
        "val_blocks_eligible": _eval_count("val", "eligible"),
        "val_blocks_skipped_short": _eval_count("val", "short"),
        "val_blocks_skipped_few_shingles": _eval_count("val", "few_shingles"),
        "val_blocks_skipped_too_large": _eval_count("val", "too_large"),
        "test_blocks_scanned": _eval_count("test", "scanned"),
        "test_blocks_eligible": _eval_count("test", "eligible"),
        "test_blocks_skipped_short": _eval_count("test", "short"),
        "test_blocks_skipped_few_shingles": _eval_count("test", "few_shingles"),
        "test_blocks_skipped_too_large": _eval_count("test", "too_large"),
        "candidate_pairs": int(candidate_pairs),
        "containment_candidate_pairs": int(containment_candidate_pairs),
        "signature_prefilter_floor": float(min(sensitivity) * 0.6),
        "signature_prefilter_skipped": int(prefilter_skipped),
        "candidate_pairs_truncated": bool(truncated_candidates),
        "oversized_groups_truncated": int(truncated_groups),
        "verified_pairs_at_threshold": int(verified_pairs),
        "max_similarity": float(max_score),
        "max_similarity_scope": ("LSH and containment-anchor candidates "
                                 "passing exact verification"),
        "top_similarities": top_examples,
        "train_blocks_dropped_as_near_duplicates_of_eval": int(
            len(remove_cross)),
        "train_intra_split_near_duplicates_removed": int(
            len(remove_intra - remove_cross)),
        "train_blocks_dropped_total": int(len(remove_cross | remove_intra)),
        "train_lines_dropped": int(dropped_lines),
        "train_chars_dropped": int(dropped_chars),
        "train_chars_eligible": train_chars_eligible,
        "train_chars_total": train_chars_total,
        "removal_fraction_chars": float(
            dropped_chars / max(1, train_chars_total)),
        "val_test_near_duplicate_pairs": int(
            sens_counts[sens_keys[threshold]]["val_test"]),
        "val_test_near_duplicate_pairs_by_threshold": {
            sens_keys[t]: int(sens_counts[sens_keys[t]]["val_test"])
            for t in sensitivity},
    }
    return out, stats


