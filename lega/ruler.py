"""简化版 RULER 长上下文检索评测（5 任务），全部在 token 层面构造、保留 gold 跨度。

任务（对应 RULER 命名）:
  niah_synth    niah_single_1   合成重复句干扰下的单键检索
  niah_corpus   niah_single_2   自然文本干扰下的单键检索
  niah_multikey niah_multikey_1 多键干扰检索（同类干扰项中找目标键）
  fwe           fwe             高频词提取（区分高频/低频词）
  vt            vt              变量追踪（多链取值）

设计要点:
  * prompt 由“干扰文本 ids + needle ids + 问题 ids”直接拼接而成，
    needle_span 保留整条证据句；analysis.py 用 value_span（答案数值 token）
    判定是否真正命中答案证据；
  * 生成采用贪心解码 + KV 缓存，与训练相同的因果语言模型接口；
  * 指标: niah/multikey/vt 用数值命中 EM；fwe 用 top-10 词召回率 recall 与严格 EM。
"""
import math
import random
import re
import zlib
from collections import Counter
from typing import Dict, List, Optional

import numpy as np
import torch

from .utils import (amp_autocast, amp_dtype_for, config_bool, save_json,
                    strict_int, wilson_score_ci)
from .repro import sha256_file, test_data_fingerprint

KEY_WORDS = ["lyre", "orchid", "wolf", "raven", "ember", "cinder", "maple", "slate",
             "quartz", "falcon", "harbor", "meadow", "thistle", "cobalt", "juniper",
             "basalt", "willow", "garnet", "lantern", "marrow", "pebble", "saffron",
             "tundra", "vessel", "walnut", "zephyr", "gambit", "hollow", "ivory", "jasper"]
FILLER = ("The grass is green. The sky is blue. The sun is yellow. "
          "Here we go. There and back again.")
# 句末保留一个空格：needle 可被插在任意 token 边界，若不补空格会出现
# "...7587205.g" 这类数值与后续单词粘连，既破坏可读性，也让“完整数值
# token”解析无法区分句号与缺损小数（7587205.x）。
NEEDLE_TMPL = "One of the special magic numbers for {key} is: {value}. "
QUESTION_TMPL = ("\nWhat is the special magic number for {key} mentioned in the provided text?"
                 "\nAnswer: The special magic number for {key} mentioned in the provided text is")
TASKS = ("niah_synth", "niah_corpus", "niah_multikey", "fwe", "vt")
# NIAH 的 needle 相对位置分箱：早期/中部/末期。长上下文模型常见
# “lost in the middle”，只报任务平均 EM 会让早/晚期的高分抵消中部塌陷；
# 因此每个 niah 单元格同时落盘分层 EM 与 Wilson CI（样本量按箱计）。
DEPTH_BINS = ("early", "middle", "late")
DEPTH_EDGES = (0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0)


def depth_bin(value) -> Optional[str]:
    """needle 位置比例（0~1）→ early/middle/late；非法输入返回 None。

    边界按半开区间 [0,1/3)、[1/3,2/3)、[2/3,1]；位置恰好为 1 的末期样本
    必须落入 late，不能因浮点边界被丢弃。
    """
    if value is None or isinstance(value, (bool, np.bool_, str, bytes)):
        return None
    try:
        d = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(d) or d < 0.0 or d > 1.0:
        return None
    if d < DEPTH_EDGES[1]:
        return "early"
    if d < DEPTH_EDGES[2]:
        return "middle"
    return "late"


def _int_or_none(value):
    """整数/整数值浮点/整数字符串 -> int；bool/小数/非有限/溢出 -> None。

    使用 strict_int 而不是 int()：后者会把 np.float32(4.5) 截成 4、
    把 np.bool_(True) 当成 1，绕过 build_vt 的链数/深度校验。
    """
    try:
        return strict_int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _stable_hash(s: str) -> int:
    """跨进程稳定的字符串哈希（内置 hash() 对字符串带随机盐，不可用于实验播种）。"""
    return zlib.crc32(s.encode("utf-8"))


def _cached_encode(tok, text: str) -> List[int]:
    """按分词器实例缓存稳定文本的编码结果。

    缓存挂在 tokenizer 实例上（id 不回收到不同对象），只对固定模板的短串使用。
    """
    cache = getattr(tok, "_lega_enc_cache", None)
    if cache is None:
        cache = {}
        tok._lega_enc_cache = cache
    ids = cache.get(text)
    if ids is None:
        ids = tok.encode(text)
        cache[text] = ids
    return ids


def _tokenizer_roundtrip(tok, text: str) -> bool:
    """分词器能否无损表示 ``text``（不含 <unk> 等替换）。

    字符级分词器只在 train 上拟合词表，train 没有数字时所有数值答案都会
    编码成同一个 ``<unk>``：此时 RULER 的 gold 跨度根本无法定位，构造器
    必须在编码前拒绝该样例（由调用方按 ContextTooSmall 记录 skipped），
    而不是在 ``_token_span_for_text`` 里抛 AssertionError。
    """
    try:
        return tok.decode(tok.encode(text)) == text
    except Exception:
        return False


class ContextTooSmall(Exception):
    """上下文长度空间不足以容纳 needle + 问题（该样例跳过）。"""


def _rand_digit(rng: random.Random, lo_digits: int, hi_digits: int) -> str:
    lo = 10 ** (lo_digits - 1)
    hi = 10 ** hi_digits - 1
    return str(rng.randint(lo, hi))


def _depth_position_range(room: int, target: str) -> tuple:
    """目标深度箱允许的整数插入位置 [lo, hi]（pos/room 必须仍在箱内）。"""
    idx = DEPTH_BINS.index(target)
    lower = DEPTH_EDGES[idx]
    upper = DEPTH_EDGES[idx + 1]
    max_pos = max(1, room - 1)
    lo = max(1, int(0.1 * room), int(math.ceil(lower * room)))
    hi = max_pos
    if idx + 1 < len(DEPTH_BINS):
        # early/middle 的上界是开区间；取最后一个整数 pos 使 pos/room < upper。
        hi = min(hi, int(math.ceil(upper * room)) - 1)
    if lo > hi:
        raise ContextTooSmall(
            f"目标深度箱 {target} 在 room={room} 时无可用插入位置")
    return lo, hi


def _free_position_segments(lo: int, hi: int, blocked) -> List[tuple]:
    """[lo,hi] 内不被干扰行内部占用的整数区间；端点可插入，行内不可。"""
    segments: List[tuple] = []
    cur = lo
    for a, b in sorted((int(a), int(b)) for a, b in blocked):
        if b <= cur:
            continue
        if a >= hi:
            break
        # 严格位于 (a,b) 的整数才会截断干扰行；a/b 本身是安全边界。
        f_lo, f_hi = max(lo, a + 1), min(hi, b - 1)
        if f_lo > f_hi:
            continue
        if f_lo > cur:
            segments.append((cur, f_lo - 1))
        cur = max(cur, f_hi + 1)
        if cur > hi:
            break
    if cur <= hi:
        segments.append((cur, hi))
    return segments


def _choose_depth_position(rng: random.Random, room: int, target: str,
                           blocked=()) -> int:
    """在目标深度箱内均匀选取不被干扰行截断的插入位置。

    直接对允许区间加权抽样，而不是先随机再接受/拒绝：multikey 的干扰行
    可能占据箱内较大比例，拒绝采样会让实际位置分布偏向空隙少的一侧。
    """
    lo, hi = _depth_position_range(room, target)
    segments = _free_position_segments(lo, hi, blocked)
    if not segments:
        # 极端情况下整箱都被干扰行覆盖：退到箱子附近最近的合法边界，
        # 保持行完整优先于深度分箱（调用方会记录实际 depth，便于审计）。
        all_segments = _free_position_segments(1, max(1, room - 1), blocked)
        if not all_segments:
            return max(1, room - 1)
        below = [seg for seg in all_segments if seg[1] < lo]
        above = [seg for seg in all_segments if seg[0] > hi]
        if below:
            return below[-1][1]
        if above:
            return above[0][0]
        return all_segments[0][0]
    total = sum(e - s + 1 for s, e in segments)
    pick = rng.randint(0, total - 1)
    for s, e in segments:
        width = e - s + 1
        if pick < width:
            return s + pick
        pick -= width
    return segments[-1][1]


def _token_span_for_text(tok, ids: List[int], text: str) -> tuple:
    """定位 text 在已编码 token 序列中的精确子跨度（跨 BPE 边界稳健）。

    token 边界不一定与字符串边界一致（例如 GPT-2 BPE 把空格并入数字 token），
    因此不能只独立编码 text 后做子序列搜索。这里在最长几十个 token 的
    needle/定义行内穷举连续窗口，并校验窗口解码文本精确等于目标字符串。
    """
    n = len(ids)
    # GPT-2 BPE 的 decode 是逐 token 字节拼接，因此对可逐 token 解码的内容，
    # 窗口解码等价于预计算逐 token 字符串后的 "".join。逐 token 解码失败
    # （理论上的多字节 UTF-8 拆分）或未命中时回退到逐窗口 decode，保证语义不变。
    try:
        pieces = [tok.decode(ids[i:i + 1]) for i in range(n)]
    except Exception:
        pieces = None
    if pieces is not None:
        for width in range(1, n + 1):
            for start in range(0, n - width + 1):
                if "".join(pieces[start:start + width]).strip() == text:
                    return start, start + width
    for width in range(1, n + 1):
        for start in range(0, n - width + 1):
            if tok.decode(ids[start:start + width]).strip() == text:
                return start, start + width
    raise AssertionError(f"无法在 token 序列中定位答案文本 {text!r}")


# ---------------------------------------------------------------- NIAH 家族

def build_niah(tok, ctx: int, rng: random.Random, mode: str = "synth",
               multikey: bool = False, corpus_sents: Optional[List[str]] = None,
               target_depth: Optional[str] = None) -> Dict:
    if target_depth is not None and target_depth not in DEPTH_BINS:
        raise ValueError(
            f"target_depth 必须是 {DEPTH_BINS} 之一或 None，收到 {target_depth!r}")
    key = rng.choice(KEY_WORDS)
    value = _rand_digit(rng, 7, 7)
    # 字符级词表只在 train 上拟合：train 没有数字时数值答案会被映射成
    # <unk>，gold 跨度无法定位。此时按可跳过的构造失败处理，而不是把
    # AssertionError 抛到评测主循环。键词/模板的 OOV 不会阻断数值跨度定位，
    # 保持旧行为（否则普通英文 char 词表只要缺一个字母就会跳过全部 NIAH）。
    if not _tokenizer_roundtrip(tok, value):
        raise ContextTooSmall(
            "tokenizer 无法无损表示 niah 的数值答案（字符级词表缺少数字；"
            f"value={value!r}）")
    needle_ids = tok.encode("\n" + NEEDLE_TMPL.format(key=key, value=value))
    # 问题模板只随 key 变化（30 个键），跨样例缓存编码结果
    question_ids = _cached_encode(tok, QUESTION_TMPL.format(key=key))
    room = ctx - len(needle_ids) - len(question_ids)
    if room < 16:
        raise ContextTooSmall(f"niah 需要 ctx>=~{len(needle_ids)+len(question_ids)+16}, 当前 {ctx}")

    # 干扰文本（长度恰好 room）
    if mode == "corpus" and not corpus_sents:
        # 语料缺失时不能用合成 FILLER 冒充 corpus（否则 niah_corpus 分数失真，
        # 却仍以 niah_corpus 名义写入结果），直接按不可构造样例跳过。
        raise ContextTooSmall("niah_corpus 需要语料句子，但 corpus_sents 为空（val.txt 缺失？）")
    if mode == "corpus":
        # 语料句子的编码结果按分词器缓存
        filler: List[int] = []
        while len(filler) < room:
            filler.extend(_cached_encode(tok, " " + rng.choice(corpus_sents).strip()))
        filler = filler[:room]
    else:
        unit = _cached_encode(tok, " " + FILLER)
        filler = (unit * (room // len(unit) + 1))[:room]

    if multikey:
        # 在干扰文本中散布同类干扰键值对（键不同、值不同）
        others = [k for k in KEY_WORDS if k != key]
        n_distract = max(4, room // 48)
        # BPE 下不同干扰键的编码长度可能相差多个 token；按最大真实键长度估算，
        # 避免用较短的假键高估可放置行数，导致尾部干扰行被截成半句。
        distract_len = getattr(tok, "_lega_distract_len", None)
        if distract_len is None:
            distract_len = 1 + max(
                len(tok.encode(NEEDLE_TMPL.format(key=k, value="0000000")))
                for k in KEY_WORDS)
            tok._lega_distract_len = distract_len
        n_distract = min(n_distract, max(1, room // distract_len))
        # 升序插入并跟踪累积位移 shift：保证任何干扰行的结束位置都不越过
        # 最终截断点 room，否则 filler[:room] 会把尾部干扰行截成半句。
        slots = sorted(rng.sample(range(2, room), min(n_distract, room - 2)))
        shift = 0
        distractor_spans = []
        for sp in slots:
            k2 = rng.choice(others)
            # 干扰值必须区别于真值，否则看错行的模型会因干扰值碰巧等于
            # needle 值而获得 EM=1，虚高 multikey 指标
            v2 = _rand_digit(rng, 7, 7)
            while v2 == value:
                v2 = _rand_digit(rng, 7, 7)
            line = tok.encode("\n" + NEEDLE_TMPL.format(key=k2, value=v2))
            if sp + shift + len(line) > room:
                continue  # 该行会越过截断点被截成半句，跳过
            # 原地切片插入：与列表拼接结果完全一致。
            filler[sp + shift:sp + shift] = line
            distractor_spans.append((sp + shift, sp + shift + len(line)))
            shift += len(line)
        filler = filler[:room]

    # needle 插入（保证完整放入）
    if target_depth is not None:
        # 分层抽样：在目标位置箱内抽样，并避开干扰行内部，保证实际
        # depth_bin 与目标箱一致、early/middle/late 的样本量不随随机种子漂移。
        pos = _choose_depth_position(
            rng, room, target_depth,
            blocked=distractor_spans if multikey else ())
    else:
        max_pos = max(1, room - 1)
        lo_pos = max(1, int(0.1 * room))
        pos = rng.randint(lo_pos, max_pos) if max_pos > lo_pos else max_pos
        if multikey:
            # needle 也不能把干扰行从中间截断：否则干扰值会裂成两个数字
            # （与上方“干扰行不得越过截断点”是同一条完整性约定）。就近吸附到
            # 行边界，不额外消耗随机源，pos 的分布基本不变。
            for a, b in distractor_spans:
                if a < pos < b:
                    pos = a if pos - a <= b - pos else b
                    break
    hay = filler[:pos] + needle_ids + filler[pos:]
    ids = hay + question_ids
    assert len(ids) <= ctx, f"样例长度 {len(ids)} 超过 ctx={ctx}"
    # 机制分析的 gold 证据应是答案数值本身；整句模板里的停用词/键词不能让
    # “盯住模板但没检索到数值”的查询位被误判为 hit。
    vs, ve = _token_span_for_text(tok, needle_ids, value)
    return {
        "task": "niah_multikey" if multikey else ("niah_corpus" if mode == "corpus" else "niah_synth"),
        "ids": ids,
        "answer": value,
        "answer_words": None,
        # 兼容字段：整条 needle 句；机制 hit 判定使用下方精确的 value_span。
        "needle_span": (pos, pos + len(needle_ids)),
        "value_span": (pos + vs, pos + ve),
        "query_span": (len(hay), len(ids)),
        "meta": {"key": key, "value": value, "depth": pos / max(1, room),
                 "depth_target": target_depth, "n_tokens": len(ids)},
    }


# ---------------------------------------------------------------- FWE

_SYL = ["ka", "ro", "mi", "ta", "lu", "sen", "dor", "vil", "pen", "tor",
        "nal", "bes", "qui", "mon", "fel", "gar", "hil", "jas", "ker", "lum"]


def _pseudo_words(rng: random.Random, n: int) -> List[str]:
    words = set()
    while len(words) < n:
        words.add("".join(rng.choice(_SYL) for _ in range(rng.randint(2, 3))))
    return sorted(words)


FWE_QUESTION = ("\nWhat are the 10 most common words in the above list?"
                "\nAnswer: The 10 most common words in the above list are")


def build_fwe(tok, ctx: int, rng: random.Random) -> Dict:
    q_ids = _cached_encode(tok, FWE_QUESTION)
    # 字符级词表可能缺少某些字母：OOV 伪词会被编码成 <unk>，既把正文
    # 变成重复的 unk 串，也会让“从 decode 文本统计 gold”选出 unk/残词。
    # 只保留能被分词器无损表示的伪词；不足时按可跳过样例处理。
    candidates = _pseudo_words(rng, 240)
    words = [w for w in candidates
             if _tokenizer_roundtrip(tok, " " + w)]
    if len(words) < 20:
        raise ContextTooSmall(
            "tokenizer 无法无损表示足够的 fwe 伪词（字符级词表缺少字母）")
    words = words[:120]
    freq = set(rng.sample(words, 10))
    # 目标正文 token 长度空间；高频词出现 k 次、低频词 2 次
    room = ctx - len(q_ids)
    if room < 40:
        raise ContextTooSmall(f"fwe 需要 ctx>=~{len(q_ids)+40}, 当前 {ctx}")
    # 按词均 token 数规划词频结构（高频词 k 次、低频词 2 次），
    # 保证正文能完整放入 token 长度空间，而不是事后硬截断破坏词频分布。
    avg_len = max(1.0, len(tok.encode(" " + " ".join(words))) / len(words))
    k = int(room / avg_len - 2 * len(words)) // 10
    if k < 3:
        # k 必须 > 2（否则高频/低频无法区分）；规划放不下时固定 k=3，
        # 交由下方循环删除低频词压缩正文。
        k = 3
    while True:
        tokens: List[str] = []
        for w in words:
            tokens.extend([w] * (k if w in freq else 2))
        rng.shuffle(tokens)
        lines = []
        for i in range(0, len(tokens), 12):
            lines.append(" ".join(tokens[i:i + 12]))
        body_ids = tok.encode("\n".join(lines))
        if len(body_ids) <= room:
            break
        # 估算偏保守时的兜底：优先删掉部分低频词（不影响 top-10 结构）；
        # 低频词删完仍超长度空间则降低 k。
        removable = [w for w in words if w not in freq]
        if removable:
            words.remove(removable[0])
        elif k > 3:
            k -= 1
        else:
            raise ContextTooSmall(f"fwe 长度空间不足: room={room}, 需要更大 ctx")
    ids = body_ids + q_ids
    # gold 必须由“计划写入正文的词”统计，而不是 decode(encoded_ids)：
    # 字符级 <unk> 会把不同 OOV 词折叠成 unk，decode 后的词频已不是
    # 题目真正要求的高频词。
    cnt = Counter(tokens)
    top10 = [w for w, _ in cnt.most_common(10)]
    if len(top10) < 10:
        raise ContextTooSmall(
            f"fwe 正文只有 {len(top10)} 个可区分的词，无法构造 top-10")
    return {
        "task": "fwe",
        "ids": ids,
        "answer": " ".join(sorted(top10)),
        "answer_words": top10,
        "needle_span": None,
        "query_span": (len(body_ids), len(ids)),
        "meta": {"n_tokens": len(ids), "k_high": k},
    }


# ---------------------------------------------------------------- VT

VT_QUESTION = ("\nGiven the variable definitions above, what is the value of VAR {L}{d}?"
               "\nAnswer: The value of VAR {L}{d} is")


def build_vt(tok, ctx: int, rng: random.Random, n_chains: int = 4, depth: int = 4) -> Dict:
    n_chains_raw, depth_raw = n_chains, depth
    n_chains = _int_or_none(n_chains)
    if n_chains is None or not 1 <= n_chains <= 6:
        raise ValueError(f"n_chains 必须是 1-6 的整数，收到 {n_chains_raw!r}")
    depth = _int_or_none(depth)
    if depth is None or depth < 1:
        raise ValueError(
            f"depth 必须是 >= 1 的整数（问题模板会引用 VAR L1），收到 {depth_raw!r}")
    letters = ["A", "B", "C", "D", "E", "F"][:n_chains]
    values = {L: _rand_digit(rng, 5, 5) for L in letters}
    # 数值答案必须能被分词器无损表示（字符级词表可能没有数字）；否则
    # gold 跨度会定位失败。键/问题模板的字符缺失同样会让题目不可读。
    if any(not _tokenizer_roundtrip(tok, v) for v in values.values()):
        raise ContextTooSmall(
            "tokenizer 无法无损表示 vt 的数值答案（字符级词表缺少数字）")
    lines = []
    for L in letters:
        lines.append(f"VAR {L}1 = {values[L]}.")
        for j in range(2, depth + 1):
            lines.append(f"VAR {L}{j} = {L}{j-1}.")
    rng.shuffle(lines)

    qL = rng.choice(letters)
    question_ids = _cached_encode(tok, VT_QUESTION.format(L=qL, d=depth))
    room = ctx - len(question_ids)
    if room < 16:
        raise ContextTooSmall(f"vt 需要 ctx>=~{len(question_ids)+16}, 当前 {ctx}")

    parts: List[List[int]] = []
    line_spans: List[tuple] = []    # (line_text, start, end)，供定位真正的 gold 证据行
    cur = 0
    for ln in lines:
        ids_ln = tok.encode(ln + "\n")
        line_spans.append((ln, cur, cur + len(ids_ln)))
        parts.append(ids_ln)
        cur += len(ids_ln)
    body_ids = [i for p in parts for i in p]
    if len(body_ids) > room:
        raise ContextTooSmall(f"vt 正文 {len(body_ids)} 超过长度空间 {room}，可减小 vt_chains/vt_depth")
    ids = body_ids + question_ids
    # gold 跨度应锁定到“实际包含目标数值”的定义行（VAR {L}1 = value）。
    target_prefix = f"VAR {qL}1 ="
    _ln, s, e = next((ln, sp, ep) for ln, sp, ep in line_spans if ln.startswith(target_prefix))
    vs, ve = _token_span_for_text(tok, ids[s:e], values[qL])
    return {
        "task": "vt",
        "ids": ids,
        "answer": values[qL],
        "answer_words": None,
        "needle_span": (s, e),
        "value_span": (s + vs, s + ve),
        "query_span": (len(body_ids), len(ids)),
        "meta": {"chain": qL, "depth": depth, "n_chains": n_chains, "n_tokens": len(ids)},
    }


# ---------------------------------------------------------------- 打分

# 完整数值 token：可选的符号/小数/科学计数法都先整体匹配，再与 gold 做
# 字符串全等比较。只截取 \d{4,} 会把 -7021853、7021853.9、7021853abc 都
# 误判为命中 7021853。
_NUM_RE = re.compile(
    r"(?<![\w.+\-*/%\u2212])[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?"
    # 数字后若紧跟 "." 再接单词/数字，说明这是残缺的数值 token（.9abc/.x），
    # 不能回退成只匹配点号前的整数；单独句号（"7021853."）仍允许。
    # 运算符/比值/范围后缀（7021853/2、+1、:123 等）说明这是表达式而非
    # 独立答案，不能把其中的前缀数字当成最终回答。
    r"(?![\w]|\.\w|\s*[+\-*/%\u2212]\s*\d|\s*:\s*\d)")
_WORD_RE = re.compile(r"[a-z]+")

# 表达式运算符：出现在候选数字左侧（允许空格）时，该数字是运算数而非独立答案。
_NUM_EXPR_OPS = "+-*/%\u2212"


def standalone_numeric_tokens(text: str) -> List[str]:
    """提取“独立答案”位置的数值 token，排除表达式/比值中的运算数。

    `_NUM_RE` 的后缀前瞻会拒绝 `7021853/2` 里的前缀数字，但正则引擎随后仍可
    从 `/` 之后匹配到 `2`，把表达式的右操作数误当成模型答案。本函数逐个
    检查候选左侧上下文：
      * 左侧（跳过空格）是算术运算符 -> 该 token 是运算数，跳过；
      * 左侧是冒号且冒号前是数字/右括号 -> 比值/范围（123:456），跳过；
      * 冒号前是普通文本（answer: 123 / answer:123）-> 标签，保留。
    这样 `7021853 / 2`、`3*4`、`123:456` 都不会被计为独立回答，而
    `answer: 2` 仍能正常解析。返回顺序与原文本一致。
    """
    if not isinstance(text, str) or not text:
        return []
    out = []
    for m in _NUM_RE.finditer(text):
        j = m.start() - 1
        while j >= 0 and text[j].isspace():
            j -= 1
        if j >= 0:
            ch = text[j]
            if ch in _NUM_EXPR_OPS:
                continue
            if ch == ":":
                k = j - 1
                while k >= 0 and text[k].isspace():
                    k -= 1
                if k >= 0 and (text[k].isdigit() or text[k] in ".)]"):
                    continue
        out.append(m.group(0))
    return out


def score_prediction(task: str, answer: str, pred_text: str,
                     answer_words: Optional[List[str]] = None) -> Dict:
    if task in ("niah_synth", "niah_corpus", "niah_multikey", "vt"):
        # EM 只承认“第一个数值回答”：若模型先答错、随后在同一输出里再蒙中正确答案，
        # 记为命中会系统性地高估检索能力。
        nums = standalone_numeric_tokens(pred_text)
        return {"em": float(bool(nums) and nums[0] == answer), "recall": None,
                "precision": None, "f1": None}
    if task == "fwe":
        pred = set(_WORD_RE.findall(pred_text.lower()))
        top = set(answer_words or [])
        recall = len(pred & top) / max(1, len(top))
        # 严格 EM：预测词集合必须与 top-10 完全相等（多答无关词同样算错），
        # 否则模型“背出 top-10 再灌水”也会被记为命中。
        em = float(bool(top) and pred == top)
        # precision/F1 为诊断量：recall 不惩罚“把上下文所有词都列出来”的
        # 灌水策略，precision 才有惩罚（多余预测压低 precision）。二者不进
        # 主平均（主指标仍是严格集合 EM），只用于区分“没找到”与“找对但多答”。
        inter = len(pred & top)
        precision = inter / len(pred) if pred else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)
        return {"em": em, "recall": recall, "precision": precision, "f1": f1}
    raise ValueError(f"未知任务 {task}")


def _needle_excerpt(tok, ex: Dict, pad: int = 40, max_chars: int = 300) -> Optional[str]:
    """gold 证据附近的摘录（token 跨度两侧各 pad 个 token），供人工复核。"""
    span = ex.get("value_span") or ex.get("needle_span")
    if not span:
        return None
    s, e = span
    seg = tok.decode(ex["ids"][max(0, s - pad):min(len(ex["ids"]), e + pad)])
    return seg if len(seg) <= max_chars else seg[:max_chars] + "…"


# ---------------------------------------------------------------- 运行评测

@torch.inference_mode()
def run_ruler(model, tok, device: str, ctx_lens: List[int], n_samples: int = 30,
              seed: int = 1234, max_new: int = 12, fwe_max_new: int = 44,
              vt_chains: int = 4, vt_depth: int = 4,
              corpus_sents: Optional[List[str]] = None,
              tasks: tuple = TASKS, verbose: bool = True,
              amp: bool = True, gen_batch: int = 8,
              collect_badcases: bool = True,
              depth_stratified: bool = True) -> Dict:
    """返回 {task: {ctx: {"em":, "em_ci95_wilson":, "recall":, "n"}}}、"_skipped" 与 "_badcases"。

    CUDA 上默认启用 bf16 autocast，与 train/evaluate 的评测精度口径一致。
    同一任务、同一上下文档下 prompt 等长的样例合并为一次批量贪婪解码：
    因果注意力与贪婪解码都逐行独立，逐样例评分口径不变。
    每格 em 附带 Wilson 95% 置信区间：n_samples 较小时点估计的抽样误差
    不可忽略，跨变体比较时应同时看区间是否重叠（区间重叠不代表差异不
    显著，正式判定仍以同种子配对检验为准）。

    collect_badcases=True 时，预测错误的样本连同问题、gold 摘录、上下文数值表
    一起落入 "_badcases"：错误样本是改进思路的最重要来源，不能只留聚合分数。

    depth_stratified=True（默认）时，niah_* 每格按 i 轮转分配目标位置箱，
    并在箱内抽样：n_samples 个样例的 early/middle/late 目标数至多差 1，
    避免纯随机位置让小样本格的中部箱为空或明显偏少，同时保持跨变体同题。
    """
    # 入口严格规范化：bool/小数在此显式拒绝。int() 截断会把 n_samples=1.5
    # 当成 1、把 ctx_lens=[True] 当成 1（样例全过短被跳过、结果为空却看似
    # 成功），并让落盘的协议字段与调用方声明的配置不一致。
    # 字符串会被逐字符迭代成 [2, 5, 6] 这类无意义上下文长度，必须显式拒绝。
    if isinstance(ctx_lens, (str, bytes)):
        raise ValueError(
            f"ctx_lens 必须是整数序列，收到字符串 {ctx_lens!r}（应传 [256, 512]）")
    ctx_lens = [strict_int(c, "ctx_lens") for c in ctx_lens]
    n_samples = strict_int(n_samples, "n_samples")
    seed = strict_int(seed, "seed")
    max_new = strict_int(max_new, "max_new")
    fwe_max_new = strict_int(fwe_max_new, "fwe_max_new")
    vt_chains = strict_int(vt_chains, "vt_chains")
    vt_depth = strict_int(vt_depth, "vt_depth")
    gen_batch = strict_int(gen_batch, "gen_batch")
    depth_stratified = config_bool(depth_stratified, "depth_stratified")
    if n_samples < 1:
        raise ValueError(f"n_samples 必须 >= 1，收到 {n_samples}")
    if not ctx_lens:
        raise ValueError("ctx_lens 不能为空")
    if any(c < 1 for c in ctx_lens):
        raise ValueError(f"ctx_lens 必须都是 >= 1 的整数，收到 {ctx_lens}")
    for name, value in (("max_new", max_new), ("fwe_max_new", fwe_max_new),
                        ("vt_chains", vt_chains), ("vt_depth", vt_depth),
                        ("gen_batch", gen_batch)):
        if value < 1:
            raise ValueError(f"{name} 必须 >= 1，收到 {value}")
    # 生成预算必须先于网格循环校验：若 max_gen 用满 block_size，room_ctx
    # 会 <=0，所有样例都以 ContextTooSmall 被跳过，函数却返回“成功”的空
    # 结果，看起来像评测完成而不是配置错误。
    for task in tasks:
        max_gen = fwe_max_new if task == "fwe" else max_new
        if model.block_size - max_gen - 1 < 1:
            raise ValueError(
                f"{task} 的生成预算 max_gen={max_gen} 过大："
                f"block_size={model.block_size} 需为 prompt 留出至少 1 个位置")
    results: Dict[str, Dict] = {}
    skipped: List[str] = []
    badcases: List[Dict] = []
    use_cuda = device.startswith("cuda")
    amp_dt = amp_dtype_for(device, bool(amp))
    amp_ctx = amp_autocast(device, amp_dt is not None)
    gen_batch = max(1, int(gen_batch))
    # 静态 KV 缓冲在单个网格内跨样例复用：按本网格历史最大 (B, total) 扩容，
    # 之后按所需窗口返回切片视图；网格开始时重置（见下）。
    attn0 = model.blocks[0].attn
    kv_B = 0
    kv_cap = 0
    kv_bufs: Optional[list] = None
    kv_dtype = amp_dt or attn0.wk.weight.dtype

    def _get_kv(B: int, total: int):
        nonlocal kv_B, kv_cap, kv_bufs
        if kv_B < B or kv_cap < total:
            kv_B, kv_cap = max(kv_B, B), max(kv_cap, total)
            kv_bufs = [
                (torch.empty(kv_B, attn0.n_heads, kv_cap, attn0.d_head,
                             dtype=kv_dtype, device=device),
                 torch.empty(kv_B, attn0.n_heads, kv_cap, attn0.d_head,
                             dtype=kv_dtype, device=device))
                for _ in model.blocks]
        return [(k[:B, :, :total], v[:B, :, :total]) for k, v in kv_bufs]

    # prompt（单样例或等长批量）的 H2D 走常驻 pinned 缓冲 + non_blocking
    # 异步拷贝：按总元素数扩容，行主序写入 pinned 的 NumPy 视图。
    pin_cap = 0
    pin_buf: Optional[torch.Tensor] = None
    pin_np = None          # pinned 缓冲的 NumPy 视图（与 pin_buf 共享存储）
    pin_ev: Optional[torch.cuda.Event] = None

    def _ids_to_device(rows: List[List[int]]) -> torch.Tensor:
        nonlocal pin_cap, pin_buf, pin_np, pin_ev
        R, L = len(rows), len(rows[0])
        n = R * L
        if not use_cuda:
            # 等长行整表转 tensor
            out = torch.from_numpy(np.asarray(rows, dtype=np.int64))
            return out if device == "cpu" else out.to(device)
        if pin_ev is not None:
            # 覆盖或替换常驻 pinned 缓冲前，必须等待上一轮 non_blocking H2D
            # 拷贝把旧缓冲读完：否则主机写入与在途 DMA 构成数据竞争；扩容时
            # 旧缓冲还可能被 pinned 分配器回收给新缓冲导致数据被踩。
            pin_ev.synchronize()
        if pin_cap < n:
            pin_buf = torch.empty(n, dtype=torch.long, pin_memory=True)
            # pin_buf.numpy() 与 pin_buf 共享存储；之后 pin_buf 永不重新分配
            # （容量不足时整体重建），视图生命周期安全。
            pin_np = pin_buf.numpy()
            pin_cap = n
            pin_ev = torch.cuda.Event()
        # 等长行按 (R, L) 写入 pinned 缓冲的 NumPy 视图（切片连续可 reshape）。
        pin_np[:n].reshape(R, L)[:] = rows
        out = pin_buf[:n].view(R, L).to(device, non_blocking=True)
        pin_ev.record()
        return out
    for task in tasks:
        results[task] = {}
        seen_rooms: set = set()
        for ctx in ctx_lens:
            # 每个 (task, ctx) 网格开始时重置 KV 缓冲。
            kv_B = 0
            kv_cap = 0
            kv_bufs = None
            # prompt 需为生成预留空间：prompt + max_new ≤ block_size，否则解码时会
            # 越过 block_size/RoPE 最大位置而报错
            max_gen = fwe_max_new if task == "fwe" else max_new
            room_ctx = min(ctx, model.block_size - max_gen - 1)
            if room_ctx in seen_rooms:
                # 多个 ctx 被 clamp 到同一 room_ctx：样本种子与构造完全相同，
                # 再跑一遍只会得到重复结果并以同键覆盖已有记录，直接跳过
                if verbose:
                    print(f"[ruler] {task}@ctx{ctx}: 与已处理的 ctx 同为 room_ctx={room_ctx}，跳过")
                continue
            seen_rooms.add(room_ctx)
            # 先按原种子顺序构造全部样例（变体间配对比较依赖同一批样本），
            # 再按 prompt 等长分组做批量贪婪解码。
            samples: List[Dict] = []
            # 起始箱只由 (task, room_ctx) 决定：n_samples=1/2 时也不会永远
            # 只抽 early；完整网格上 three bins 的目标数仍至多差 1。
            depth_start = (_stable_hash(f"{task}:{room_ctx}") % len(DEPTH_BINS)
                           if depth_stratified and task.startswith("niah") else 0)
            for i in range(n_samples):
                # 种子混入 room_ctx：同一 (task, i) 的样例在不同上下文长度下
                # 若共用同一 rng 构造序列前缀，needle 的 key/value 会跨长度相关，
                # 长度扫掠曲线各点误差不再独立。
                rng = random.Random(seed * 100003 + _stable_hash(task) % 99991
                                    + room_ctx * 31 + i)
                target_depth = (DEPTH_BINS[(depth_start + i) % len(DEPTH_BINS)]
                                if depth_stratified and task.startswith("niah")
                                else None)
                try:
                    if task == "niah_synth":
                        ex = build_niah(tok, room_ctx, rng, mode="synth",
                                        target_depth=target_depth)
                    elif task == "niah_corpus":
                        ex = build_niah(tok, room_ctx, rng, mode="corpus",
                                        corpus_sents=corpus_sents,
                                        target_depth=target_depth)
                    elif task == "niah_multikey":
                        ex = build_niah(tok, room_ctx, rng, mode="synth",
                                        multikey=True, target_depth=target_depth)
                    elif task == "fwe":
                        ex = build_fwe(tok, room_ctx, rng)
                    elif task == "vt":
                        ex = build_vt(tok, room_ctx, rng, n_chains=vt_chains, depth=vt_depth)
                    else:
                        raise ValueError(task)
                except ContextTooSmall as e:
                    # 样例长度因样例而异（fwe/vt）：跳过该样例而非终止整个任务，
                    # 避免单个超长样例让不同模型的有效样本量 n 悄悄不同
                    skipped.append(f"{task}@ctx{ctx}: {e}")
                    continue
                # 稳定标识：各变体用同一 (task, ctx, i) 构造同一批样本，
                # 跨变体错误对照靠它把错误的个案一一对齐（而不是靠文本猜测）
                ex["sample_id"] = f"{task}@{room_ctx}#{i}"
                samples.append(ex)
            if not samples:
                continue
            by_len: Dict[int, List[Dict]] = {}
            for ex in samples:
                by_len.setdefault(len(ex["ids"]), []).append(ex)
            ems, recalls, n = [], [], 0
            precisions, f1s = [], []
            # 逐样例结果随单元格落盘：变体间用同一 sample_seed 构造同一批
            # 题面，聚合 EM 的跨种子比较只反映训练随机性；样例级配对
            # （McNemar）才能回答“在固定的这批评测样例上，候选是否更常
            # 答对”，同时避免把 n=20~30 的评测抽样误差当成模型差异。
            outcomes: List[Dict] = []
            depth_ems: Dict[str, List[float]] = {}
            for length in sorted(by_len):
                grp = by_len[length]
                total = min(length + max_gen, model.block_size)
                for s0 in range(0, len(grp), gen_batch):
                    part = grp[s0:s0 + gen_batch]
                    # 等长 prompt 堆成 (B, L)（单样例即 B=1）统一走 pinned
                    # 缓冲异步 H2D；批量贪婪解码逐行独立，每行结果与逐样例
                    # 生成一致。
                    with amp_ctx:
                        ids = _ids_to_device([e["ids"] for e in part])
                        out_ids = model.generate(
                            ids, max_new_tokens=max_gen,
                            kv_cache=_get_kv(len(part), total))
                    # 只把生成段转 Python 列表（提示前缀不参与解码）
                    out_ids_l = out_ids[:, length:].tolist()
                    for r, ex in enumerate(part):
                        pred_text = tok.decode(out_ids_l[r])
                        sc = score_prediction(task, ex["answer"], pred_text,
                                              ex["answer_words"])
                        ems.append(sc["em"])
                        rec = {"sample_id": ex.get("sample_id"),
                               "em": float(sc["em"])}
                        if task.startswith("niah"):
                            # 目标箱与实际箱分开落盘：极端构造下插入位置可能
                            # 被干扰行边界挤到相邻箱，审计时不能把“计划分层”
                            # 当成“实际分层”。
                            rec["depth_target"] = target_depth
                            # 位置分层只在 NIAH 有意义（vt 的 meta.depth 是
                            # 链深度而非 token 位置，不能混用）。
                            dval = ex.get("meta", {}).get("depth")
                            db = depth_bin(dval)
                            rec["depth"] = (float(dval) if db is not None
                                            else None)
                            rec["depth_bin"] = db
                            if db is not None:
                                depth_ems.setdefault(db, []).append(
                                    float(sc["em"]))
                        outcomes.append(rec)
                        if sc["recall"] is not None:
                            recalls.append(sc["recall"])
                        if sc.get("precision") is not None:
                            precisions.append(sc["precision"])
                        if sc.get("f1") is not None:
                            f1s.append(sc["f1"])
                        n += 1
                        if collect_badcases and sc["em"] < 1.0:
                            prompt_text = tok.decode(ex["ids"])
                            prompt_nums = set(standalone_numeric_tokens(
                                prompt_text))
                            qs, qe = ex["query_span"]
                            badcases.append({
                                "sample_id": ex.get("sample_id"),
                                "task": task, "ctx": int(room_ctx),
                                "answer": ex["answer"],
                                "answer_words": ex["answer_words"],
                                "pred": pred_text.strip(),
                                "em": sc["em"], "recall": sc["recall"],
                                "precision": sc.get("precision"),
                                "f1": sc.get("f1"),
                                "question": tok.decode(ex["ids"][qs:qe]),
                                "needle_excerpt": _needle_excerpt(tok, ex),
                                # 坏案例的 depth 字段统一语义为“needle 位置比例”
                                # （0~1）；vt 的 meta.depth 是链深度（整数），
                                # 不能按位置解释，仅 niah 家族记录（vt 链深度
                                # 仍在 meta 中可见）。
                                "depth": (ex.get("meta", {}).get("depth")
                                          if task.startswith("niah") else None),
                                "meta": {k: v for k, v in ex.get("meta", {}).items()
                                         if k != "n_tokens"},
                                # 上下文中除 gold 外的全部数值（区分"误抄干扰值"
                                # 与"幻觉数字"）；gold 是否真的在 prompt 里
                                #（标注/构造自检：不在即为构造错误）
                                "context_numbers": sorted(prompt_nums - {ex["answer"]}),
                                "answer_in_prompt": (ex["answer"] in prompt_nums
                                                     if task != "fwe" else None),
                            })
            if n:
                # 以实际使用的上下文长度记录：若 ctx 超出 block_size 长度空间，
                # 样本是在缩短后的 room_ctx 上构建的，记入 ctx 会虚增长上下文分数
                results[task][str(room_ctx)] = {
                    "em": float(sum(ems) / n),
                    # EM 是 0/1 二值指标：n<=30 的任务档位下点估计的抽样误差
                    # 相当可观（1/n 即 ~3%），Wilson 95% 区间把这份不确定性
                    # 显式随结果落盘（小样本下比 Wald 区间可靠，端点不越界）。
                    "em_ci95_wilson": list(wilson_score_ci(int(round(sum(ems))), n)),
                    "recall": (float(sum(recalls) / len(recalls)) if recalls else None),
                    "precision": (float(sum(precisions) / len(precisions))
                                  if precisions else None),
                    "f1": (float(sum(f1s) / len(f1s)) if f1s else None),
                    "n": n,
                    # 逐样例 EM（sample_id 跨变体稳定对齐）：供下游做
                    # 样例级配对检验，见 run_all 的 RULER 样例配对小节。
                    "samples": outcomes,
                }
                if task.startswith("niah"):
                    target_counts = {
                        b: sum(1 for s in outcomes
                               if s.get("depth_target") == b)
                        for b in DEPTH_BINS}
                    results[task][str(room_ctx)]["depth_stratified"] = bool(
                        depth_stratified)
                    results[task][str(room_ctx)]["depth_target_counts"] = (
                        target_counts)
                    results[task][str(room_ctx)]["depth_assignment_matches_target"] = (
                        all(s.get("depth_bin") == s.get("depth_target")
                            for s in outcomes))
                    # n>=3 才能覆盖三箱；目标数至多差 1 才算按计划均衡。
                    results[task][str(room_ctx)]["depth_target_balanced"] = (
                        (max(target_counts.values())
                         - min(target_counts.values()) <= 1
                         and all(v > 0 for v in target_counts.values()))
                        if depth_stratified and n >= len(DEPTH_BINS) else None)
                    # 位置分层：早期高分与中期塌陷会在总 EM 里互相抵消。
                    # 每箱独立 Wilson CI（n 为箱内样本数），避免拿总 n
                    # 冒充分层精度；空箱不写入，下游按缺失处理。
                    results[task][str(room_ctx)]["em_by_depth"] = {
                        b: {
                            "n": len(depth_ems[b]),
                            "em": float(sum(depth_ems[b]) / len(depth_ems[b])),
                            "em_ci95_wilson": list(wilson_score_ci(
                                int(round(sum(depth_ems[b]))),
                                len(depth_ems[b]))),
                        }
                        for b in DEPTH_BINS if depth_ems.get(b)
                    }
                if verbose:
                    r = results[task][str(room_ctx)]
                    extra = (f" recall={r['recall']:.3f}"
                             f" prec={r['precision']:.3f}"
                             f" f1={r['f1']:.3f}"
                             if r["recall"] is not None else "")
                    print(f"[ruler] {task}@ctx{room_ctx}: em={r['em']:.3f}{extra} (n={n})")
    if skipped:
        results["_skipped"] = skipped
        if verbose:
            for s in skipped[:10]:
                print(f"[ruler] 跳过: {s}")
    if badcases:
        results["_badcases"] = badcases
    return results


def load_corpus_sents_with_source(data_dir: str, max_chars: int = 300000,
                                  splits=("val", "test", "train"),
                                  allow_test: bool = False):
    """抽取 niah_corpus 的干扰语句，返回 (sents, 实际使用的 split)。

    评测上下文绝不能来自训练语料：若干扰句取自 train.txt，模型在“自然文本
    干扰”任务上评的是见过的文本，长上下文检索会被系统性高估，机制分析
    （misleading_rate 等）也建立在记忆而非泛化上。因此按 held-out 优先级
    选择：val → test → train；仅当没有任何 held-out 文本时才回退 train
    （调用方须显式告警，并把来源写入结果供审计）。

    但 test 是尘封 split：本函数直接读裸 .txt，绕过 lega.data.load_bin 的
    allow_test 闸门，若默认就把 test 当候选，val 缺失时每次 ruler/analysis
    都会无账本地读测试集（并把 test 文本喂进评测 prompt）。因此默认
    allow_test=False 时 test 从候选中剔除（退化为 val → train）；只有显式
    allow_test=True 才纳入 test，且调用方必须写访问账本。
    """
    import os
    for split in splits:
        # split 是文件名片段，不能是任意路径："./test" 或绝对路径会绕过
        # 下面的“test 只有 allow_test=True 才可读”闸门（os.path.join 对绝对
        # 路径还会丢弃 data_dir）。只允许数据协议定义的三个 split。
        if not isinstance(split, str) or split not in ("train", "val", "test"):
            raise ValueError(
                f"splits 只能包含 'train'/'val'/'test'，收到 {split!r}")
        # 只接受真正的 bool True：非布尔真值（如 "false"）不能打开 test。
        if split == "test" and allow_test is not True:
            continue
        path = os.path.join(data_dir, f"{split}.txt")
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read(max_chars)
        except OSError:
            continue
        sents = [s.strip() for s in
                 re.split(r"(?<=[.!?])\s+", text.replace("\n", " "))]
        sents = [s for s in sents if 20 <= len(s) <= 200]
        if sents:
            return sents, split
    return [], None


def load_corpus_sents(data_dir: str, max_chars: int = 300000) -> List[str]:
    """兼容旧调用：只返回语句列表（来源见 load_corpus_sents_with_source）。

    兼容入口同样不放行 test（尘封 split），见 load_corpus_sents_with_source。
    """
    return load_corpus_sents_with_source(data_dir, max_chars=max_chars)[0]


def record_corpus_test_access(data_dir: str, source, ckpt_sha256: str = None,
                              variant=None, seed=None, stage: str = "ruler") -> bool:
    """干扰语料取自 test 时补写 test 访问账本（best-effort，返回是否写入成功）。

    ruler/analysis 直接读裸 test.txt 属于账本旁路；显式放行后必须留下记录，
    否则“测试集被哪些阶段读过”无法审计。账本不可写（只读目录等）时不阻塞
    评测，但调用方须打印告警。
    """
    if source != "test":
        return False
    import os
    from .datacheck import append_test_access
    source_path = os.path.join(data_dir, "test.txt")
    try:
        append_test_access(data_dir, {
            "reason": f"{stage}_niah_corpus_distractors",
            "ckpt_sha256": ckpt_sha256,
            "variant": variant,
            "seed": seed,
            # 账本统一按 test.bin 版本分组；同时记录实际读取的裸 txt 哈希，
            # 防止 txt 与 bin 不同步时审计链只覆盖派生文件。
            "test_data_fingerprint": test_data_fingerprint(data_dir),
            "corpus_source_sha256": (sha256_file(source_path)
                                     if os.path.exists(source_path) else None),
        })
        return True
    except OSError:
        return False


def main():
    import argparse
    from .model import load_model
    from .data import load_tokenizer
    from .preflight import (allow_protocol_violations,
                            data_protocol_preflight)
    from .repro import load_provenance
    from .utils import get_device, enable_tf32, set_seed, strict_int

    p = argparse.ArgumentParser(description="RULER 5 任务评测")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--ctx_lens", default=None, help="逗号分隔，默认用训练配置")
    p.add_argument("--n_samples", type=int, default=None)
    p.add_argument("--gen_batch", type=int, default=None,
                   help="等长 prompt 批量贪婪解码的最大批大小")
    p.add_argument("--no_amp", action="store_true", help="禁用 bf16 autocast（CUDA）")
    p.add_argument("--device", default="auto")
    p.add_argument("--allow_test_corpus", action="store_true",
                   help="允许 niah_corpus 干扰语句取自 test.txt（尘封 split 默认"
                        "禁止；放行时写入 test 访问账本，见 load_corpus_sents_with_source）")
    p.add_argument("--allow_data_protocol_violations", action="store_true",
                   help="即使 data_report 标记泄露也继续评测（仅用于复现历史"
                        "实验；结果不得用于正式结论）")
    args = p.parse_args()

    data_preflight = data_protocol_preflight(
        args.data_dir,
        allow_override=allow_protocol_violations(
            args.allow_data_protocol_violations))
    if not data_preflight["report_present"]:
        print("[ruler] 警告：数据目录缺少 data_report.json，"
              "无法执行评测前防泄漏闸门（旧数据或未走统一入口）")
    elif data_preflight["critical_violations"]:
        print(f"[ruler] ⚠ 数据协议自检发现 "
              f"{len(data_preflight['critical_violations'])} 个严重问题，"
              "但已显式放行（结果不得用于正式结论）")

    device = get_device(args.device)
    enable_tf32(device)
    model, ck = load_model(args.ckpt, device=device)
    # 被评测权重的精确内容指纹：RULER 结果必须与产生它的 checkpoint 一一
    # 绑定，而不是只信目录名；同一路径被新权重覆盖后可立即发现。
    ckpt_sha256 = sha256_file(args.ckpt)
    tok = load_tokenizer(args.data_dir)
    seed_raw = ck.get("seed")
    # checkpoint 元数据的种子同样不能被 int() 截断（1.5 → 1、True → 1）：
    # 截断后评测播种的种子与记录的种子不是同一个，身份审计形同虚设。
    # 输出也使用规范化后的值（5.0 -> 5），保证结果与真正播种一致。
    seed = (None if seed_raw is None
            else strict_int(seed_raw, "checkpoint seed"))
    set_seed(0 if seed is None else seed)

    import json
    import os
    cfg_path = os.path.join(os.path.dirname(os.path.abspath(args.ckpt)), "config_used.json")
    cfg = {}
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, encoding="utf-8") as f:
                cfg = json.load(f)
        except (OSError, ValueError):
            cfg = {}
        # 合法 JSON 但非对象（数组/标量）的配置按空配置处理，走默认值
        # 而不是在 .get 上崩溃。
        if not isinstance(cfg, dict):
            cfg = {}
    rl = cfg.get("ruler")
    rl = rl if isinstance(rl, dict) else {}

    def _cfg_int(key, default):
        """ruler 配置整数：缺失用默认值，非法值显式拒绝。

        这里不能用 int()：4.5 → 4、True → 1 的截断会发生在 build_vt 的
        链数/深度守卫**之前**，把非法配置悄悄洗成合法值；样本数/上下文
        被截断或变成 1 还会让“配对可比的样例集”与配置声明不一致。
        """
        if key not in rl:
            return default
        return strict_int(rl[key], f"ruler.{key}")

    if args.ctx_lens:
        ctx_lens = [strict_int(x, "--ctx_lens")
                    for x in args.ctx_lens.split(",")]
    else:
        raw_ctx_lens = rl.get("ctx_lens", [min(256, model.block_size)])
        if not isinstance(raw_ctx_lens, (list, tuple)):
            raise ValueError(
                f"ruler.ctx_lens 需要整数列表，收到 {raw_ctx_lens!r}")
        ctx_lens = [strict_int(x, "ruler.ctx_lens") for x in raw_ctx_lens]
    n_samples = (args.n_samples if args.n_samples is not None
                 else _cfg_int("n_samples", 30))
    gen_batch = (args.gen_batch if args.gen_batch is not None
                 else _cfg_int("gen_batch", 8))
    sample_seed = _cfg_int("seed", 1234)
    max_new = _cfg_int("max_new", 12)
    fwe_max_new = _cfg_int("fwe_max_new", 44)
    vt_chains = _cfg_int("vt_chains", 4)
    vt_depth = _cfg_int("vt_depth", 4)
    # 与训练内评测口径一致：尊重训练配置的 amp 设置（默认开启）
    amp = config_bool(cfg.get("amp", True), "amp") and not args.no_amp
    # NIAH 位置分层是生成协议的一部分：关闭时落盘 False，旧产物缺字段，
    # 下游协议核对因此不会把“随机位置”与“分层位置”的 EM 直接配对。
    depth_stratified = config_bool(
        rl.get("depth_stratified", True), "ruler.depth_stratified")

    corpus_sents, corpus_source = load_corpus_sents_with_source(
        args.data_dir, allow_test=bool(args.allow_test_corpus))
    if corpus_source == "test":
        # test 作为干扰语料是账本旁路的显式放行：必须留痕，否则测试集被
        # prompt 级读取完全不可审计（load_bin 的闸门管不到裸 .txt 读取）。
        if record_corpus_test_access(args.data_dir, corpus_source,
                                     ckpt_sha256=ckpt_sha256,
                                     variant=ck.get("variant"),
                                     seed=seed, stage="ruler"):
            print("[ruler] ⚠ niah_corpus 干扰语料取自 test.txt（尘封 split）："
                  "已写入 test 访问账本；test 文本进入评测 prompt，"
                  "与后续 test PPL 评测共用同一批文本，需在报告中披露")
        else:
            print("[ruler] ⚠ niah_corpus 干扰语料取自 test.txt 且访问账本写入"
                  "失败：本次读取未被记录，测试集审计存在缺口")
    elif corpus_source == "train":
        print("[ruler] ⚠ niah_corpus 干扰语料回退到 train.txt（缺少 val/test 文本）："
              "评测上下文与训练语料重叠，长上下文检索分数偏高、机制分析建立在"
              "见过的文本上，不能与 held-out 语料的运行直接比较")
    elif corpus_source is None:
        print("[ruler] ⚠ 未找到可用干扰语料（val/test/train 文本均缺失）："
              "niah_corpus 将被跳过")
    else:
        print(f"[ruler] niah_corpus 干扰语料来源: {corpus_source}.txt（held-out）")
    results = run_ruler(model, tok, device, ctx_lens=ctx_lens, n_samples=n_samples,
                        seed=sample_seed,
                        max_new=max_new,
                        fwe_max_new=fwe_max_new,
                        vt_chains=vt_chains,
                        vt_depth=vt_depth,
                        corpus_sents=corpus_sents,
                        amp=amp, gen_batch=gen_batch,
                        depth_stratified=depth_stratified)
    # 坏案例是 "_" 前缀的辅助键：不混入 results，避免下游聚合把它当任务
    badcases = results.pop("_badcases", [])
    payload = {"variant": ck.get("variant", model.variant), "seed": seed,
               "ctx_lens": ctx_lens, "n_samples": n_samples,
               "sample_seed": sample_seed, "results": results,
               "badcases": badcases,
               # 生成口径随结果落盘：不同 max_new/解码预算或任务构造参数下的
               # EM 不是同一评测协议，下游不得直接配对（见 run_all 的协议核对）。
               "max_new": max_new,
               "fwe_max_new": fwe_max_new,
               "vt_chains": vt_chains,
               "vt_depth": vt_depth,
               "gen_batch": int(gen_batch),
               "amp": bool(amp),
               "depth_stratified": bool(depth_stratified),
               # 干扰语料来源可审计：train 回退会让检索分数不可比
               "corpus_source": corpus_source,
               "corpus_held_out": corpus_source in ("val", "test"),
               # 被评测 checkpoint 的路径与内容指纹：结果与权重一一绑定
               "ckpt": os.path.abspath(args.ckpt),
               "ckpt_sha256": ckpt_sha256,
               # 评测前的数据协议闸门结论随结果落盘，防止泄露数据的结果
               # 被当成 held-out 证据引用。
               "data_protocol_preflight": data_preflight}
    payload.update(load_provenance(
        os.path.dirname(os.path.abspath(args.ckpt)), args.data_dir,
        device=device))
    save_json(payload, args.out)
    print(f"[ruler] 完成 -> {args.out}")
    if badcases:
        print(f"[ruler] 预测错误样本 {len(badcases)} 条已随结果落盘"
              "（payload['badcases']），可用 python -m lega.badcase 生成错误归因报告")


if __name__ == "__main__":
    main()
