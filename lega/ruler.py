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
import random
import re
import zlib
from collections import Counter
from typing import Dict, List, Optional

import torch

from .utils import save_json

KEY_WORDS = ["lyre", "orchid", "wolf", "raven", "ember", "cinder", "maple", "slate",
             "quartz", "falcon", "harbor", "meadow", "thistle", "cobalt", "juniper",
             "basalt", "willow", "garnet", "lantern", "marrow", "pebble", "saffron",
             "tundra", "vessel", "walnut", "zephyr", "gambit", "hollow", "ivory", "jasper"]
FILLER = ("The grass is green. The sky is blue. The sun is yellow. "
          "Here we go. There and back again.")
NEEDLE_TMPL = "One of the special magic numbers for {key} is: {value}."
QUESTION_TMPL = ("\nWhat is the special magic number for {key} mentioned in the provided text?"
                 "\nAnswer: The special magic number for {key} mentioned in the provided text is")
TASKS = ("niah_synth", "niah_corpus", "niah_multikey", "fwe", "vt")


def _stable_hash(s: str) -> int:
    """跨进程稳定的字符串哈希（内置 hash() 对字符串带随机盐，不可用于实验播种）。"""
    return zlib.crc32(s.encode("utf-8"))


def _cached_encode(tok, text: str) -> List[int]:
    """按分词器实例缓存稳定文本的编码结果。

    样例构造（尤其 corpus 干扰文本与合成 FILLER）会在 n_samples × ctx_lens
    个样例间反复编码同一批字符串；缓存挂在 tokenizer 实例上（id 不回收到
    不同对象），只对固定模板的短串使用。
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


class ContextTooSmall(Exception):
    """上下文长度空间不足以容纳 needle + 问题（该样例跳过）。"""


def _rand_digit(rng: random.Random, lo_digits: int, hi_digits: int) -> str:
    lo = 10 ** (lo_digits - 1)
    hi = 10 ** hi_digits - 1
    return str(rng.randint(lo, hi))


def _token_span_for_text(tok, ids: List[int], text: str) -> tuple:
    """定位 text 在已编码 token 序列中的精确子跨度（跨 BPE 边界稳健）。

    token 边界不一定与字符串边界一致（例如 GPT-2 BPE 把空格并入数字 token），
    因此不能只独立编码 text 后做子序列搜索。这里在最长几十个 token 的
    needle/定义行内穷举连续窗口，并校验窗口解码文本精确等于目标字符串。
    """
    n = len(ids)
    # GPT-2 BPE 的 decode 是逐 token 字节拼接，因此对可逐 token 解码的内容，
    # 窗口解码等价于预计算逐 token 字符串后的 "".join——把 O(n²) 次 decode
    # 降为 O(n) 次。逐 token 解码失败（理论上的多字节 UTF-8 拆分）或未命中时
    # 回退到原始的逐窗口 decode，保证语义不变。
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
               multikey: bool = False, corpus_sents: Optional[List[str]] = None) -> Dict:
    key = rng.choice(KEY_WORDS)
    value = _rand_digit(rng, 7, 7)
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
        # 语料句子的编码结果按分词器缓存：同一句子会在多样例/多 ctx 下被反复抽中
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
        # BPE 下不同干扰键的编码长度可能相差多个 token；按最大真实键成本估算，
        # 避免用较短的假键高估可放置行数，导致尾部干扰行被截成半句。
        distract_cost = getattr(tok, "_lega_distract_cost", None)
        if distract_cost is None:
            distract_cost = 1 + max(
                len(tok.encode(NEEDLE_TMPL.format(key=k, value="0000000")))
                for k in KEY_WORDS)
            tok._lega_distract_cost = distract_cost
        n_distract = min(n_distract, max(1, room // distract_cost))
        # 升序插入并跟踪累积位移 shift：保证任何干扰行的结束位置都不越过
        # 最终截断点 room，否则 filler[:room] 会把尾部干扰行截成半句。
        slots = sorted(rng.sample(range(2, room), min(n_distract, room - 2)))
        shift = 0
        for sp in slots:
            k2 = rng.choice(others)
            v2 = _rand_digit(rng, 7, 7)
            line = tok.encode("\n" + NEEDLE_TMPL.format(key=k2, value=v2))
            if sp + shift + len(line) > room:
                continue  # 该行会越过截断点被截成半句，跳过
            filler = filler[:sp + shift] + line + filler[sp + shift:]
            shift += len(line)
        filler = filler[:room]

    # needle 插入（保证完整放入）
    max_pos = max(1, room - 1)
    lo_pos = max(1, int(0.1 * room))
    pos = rng.randint(lo_pos, max_pos) if max_pos > lo_pos else max_pos
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
                 "n_tokens": len(ids)},
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
    words = _pseudo_words(rng, 120)
    freq = set(rng.sample(words, 10))
    # 目标正文 token 长度空间；高频词出现 k 次、低频词 2 次
    room = ctx - len(q_ids)
    if room < 40:
        raise ContextTooSmall(f"fwe 需要 ctx>=~{len(q_ids)+40}, 当前 {ctx}")
    # 根据实测的词均 token 数规划词频结构（高频词 k 次、低频词 2 次），
    # 保证正文能完整放入 token 长度空间，而不是事后硬截断破坏词频分布。
    avg_cost = max(1.0, len(tok.encode(" " + " ".join(words))) / len(words))
    k = int(room / avg_cost - 2 * len(words)) // 10
    if k < 3:  # k 必须 > 2，否则高频词与低频词无法区分
        raise ContextTooSmall(f"fwe 长度空间不足: room={room}, 需要更大 ctx")
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
    # 以“实际进入正文”的词频确定 top-10（裁剪后词频会变）
    body_text = tok.decode(body_ids)
    cnt = Counter(re.findall(r"[a-z]+", body_text))
    top10 = [w for w, _ in cnt.most_common(10)]
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
    if not 1 <= n_chains <= 6:
        raise ValueError(f"n_chains 必须在 1-6 之间，收到 {n_chains}")
    letters = ["A", "B", "C", "D", "E", "F"][:n_chains]
    values = {L: _rand_digit(rng, 5, 5) for L in letters}
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
    # 原先用同一条链所有行的 min/max 包围盒：行被打散后会包住别的链的整行，
    # 使 analysis 里的 hit 指标把“注意力落到无关行”误判为找对。
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

_NUM_RE = re.compile(r"\d{4,}")
_WORD_RE = re.compile(r"[a-z]+")


def score_prediction(task: str, answer: str, pred_text: str,
                     answer_words: Optional[List[str]] = None) -> Dict:
    if task in ("niah_synth", "niah_corpus", "niah_multikey", "vt"):
        # EM 只承认“第一个数值回答”：若模型先答错、随后在同一输出里再蒙中正确答案，
        # 记为命中会系统性地高估检索能力。
        nums = _NUM_RE.findall(pred_text)
        return {"em": float(bool(nums) and nums[0] == answer), "recall": None}
    if task == "fwe":
        pred = set(_WORD_RE.findall(pred_text.lower()))
        top = set(answer_words or [])
        recall = len(pred & top) / max(1, len(top))
        # 严格 EM：预测词集合必须与 top-10 完全相等（多答无关词同样算错），
        # 否则模型“背出 top-10 再灌水”也会被记为命中。
        return {"em": float(bool(top) and pred == top), "recall": recall}
    raise ValueError(f"未知任务 {task}")


# ---------------------------------------------------------------- 运行评测

@torch.inference_mode()
def run_ruler(model, tok, device: str, ctx_lens: List[int], n_samples: int = 30,
              seed: int = 1234, max_new: int = 12, fwe_max_new: int = 44,
              vt_chains: int = 4, vt_depth: int = 4,
              corpus_sents: Optional[List[str]] = None,
              tasks: tuple = TASKS, verbose: bool = True) -> Dict:
    """返回 {task: {ctx: {"em":, "recall":, "n":}}} 及跳过说明。"""
    results: Dict[str, Dict] = {}
    skipped: List[str] = []
    for task in tasks:
        results[task] = {}
        seen_rooms: set = set()
        for ctx in ctx_lens:
            ems, recalls, n = [], [], 0
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
            for i in range(n_samples):
                rng = random.Random(seed * 100003 + _stable_hash(task) % 99991 + i)
                try:
                    if task == "niah_synth":
                        ex = build_niah(tok, room_ctx, rng, mode="synth")
                    elif task == "niah_corpus":
                        ex = build_niah(tok, room_ctx, rng, mode="corpus", corpus_sents=corpus_sents)
                    elif task == "niah_multikey":
                        ex = build_niah(tok, room_ctx, rng, mode="synth", multikey=True)
                    elif task == "fwe":
                        ex = build_fwe(tok, room_ctx, rng)
                    elif task == "vt":
                        ex = build_vt(tok, room_ctx, rng, n_chains=vt_chains, depth=vt_depth)
                    else:
                        raise ValueError(task)
                except ContextTooSmall as e:
                    skipped.append(f"{task}@ctx{ctx}: {e}")
                    break
                ids = torch.tensor([ex["ids"]], dtype=torch.long, device=device)
                out_ids = model.generate(ids, max_new_tokens=fwe_max_new if task == "fwe" else max_new)
                pred_text = tok.decode(out_ids[0, len(ex["ids"]):].tolist())
                sc = score_prediction(task, ex["answer"], pred_text, ex["answer_words"])
                ems.append(sc["em"])
                if sc["recall"] is not None:
                    recalls.append(sc["recall"])
                n += 1
            if n:
                # 以实际使用的上下文长度记录：若 ctx 超出 block_size 长度空间，
                # 样本是在缩短后的 room_ctx 上构建的，记入 ctx 会虚增长上下文分数
                results[task][str(room_ctx)] = {
                    "em": float(sum(ems) / n),
                    "recall": (float(sum(recalls) / len(recalls)) if recalls else None),
                    "n": n,
                }
                if verbose:
                    r = results[task][str(room_ctx)]
                    extra = f" recall={r['recall']:.3f}" if r["recall"] is not None else ""
                    print(f"[ruler] {task}@ctx{room_ctx}: em={r['em']:.3f}{extra} (n={n})")
    if skipped:
        results["_skipped"] = skipped
        if verbose:
            for s in skipped[:10]:
                print(f"[ruler] 跳过: {s}")
    return results


def load_corpus_sents(data_dir: str, max_chars: int = 300000) -> List[str]:
    """从 val.txt 抽自然句，作为 niah_corpus 的干扰语料。"""
    import os
    path = os.path.join(data_dir, "val.txt")
    if not os.path.exists(path):
        return []
    text = open(path, "r", encoding="utf-8").read(max_chars)
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.replace("\n", " "))]
    return [s for s in sents if 20 <= len(s) <= 200]


def main():
    import argparse
    from .model import load_model
    from .data import load_tokenizer
    from .utils import get_device

    p = argparse.ArgumentParser(description="RULER 5 任务评测")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--ctx_lens", default=None, help="逗号分隔，默认用训练配置")
    p.add_argument("--n_samples", type=int, default=None)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    device = get_device(args.device)
    model, ck = load_model(args.ckpt, device=device)
    tok = load_tokenizer(args.data_dir)

    import json
    import os
    cfg_path = os.path.join(os.path.dirname(os.path.abspath(args.ckpt)), "config_used.json")
    cfg = json.load(open(cfg_path, encoding="utf-8")) if os.path.exists(cfg_path) else {}
    rl = cfg.get("ruler", {})
    ctx_lens = ([int(x) for x in args.ctx_lens.split(",")] if args.ctx_lens
                else rl.get("ctx_lens", [min(256, model.block_size)]))
    n_samples = args.n_samples or int(rl.get("n_samples", 30))

    results = run_ruler(model, tok, device, ctx_lens=ctx_lens, n_samples=n_samples,
                        seed=int(rl.get("seed", 1234)),
                        max_new=int(rl.get("max_new", 12)),
                        fwe_max_new=int(rl.get("fwe_max_new", 44)),
                        vt_chains=int(rl.get("vt_chains", 4)),
                        vt_depth=int(rl.get("vt_depth", 4)),
                        corpus_sents=load_corpus_sents(args.data_dir))
    payload = {"variant": ck.get("variant", model.variant), "ctx_lens": ctx_lens,
               "n_samples": n_samples, "results": results}
    save_json(payload, args.out)
    print(f"[ruler] 完成 -> {args.out}")


if __name__ == "__main__":
    main()
