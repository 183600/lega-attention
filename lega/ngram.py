"""简单统计基线：一元/二元/插值二元平滑 n-gram 的 held-out PPL。

科研方法问题：没有简单基线时，“神经模型 PPL=xx”无法回答“数据本身
有多难、复杂架构是否真的比无参数的统计模型更好”；而一个故意没调好
的弱基线（如只扫两档 α 的纯 bigram）会把“复杂模型更好”变成廉价结论。
本模块提供零训练、选参只在验证集的 n-gram 基线：

  * 一元：P(w) = (c(w) + α) / (N + α·V)，即加 α 平滑
  * 二元：P(w_t | w_{t-1}) = (c(w_{t-1}, w_t) + α) / (c(w_{t-1}) + α·V)
  * 插值二元：P = λ·P_bigram + (1−λ)·P_unigram；λ=1 时退化为二元模型，
    因此“最强简单基线”永不低于纯二元（选择面包含它）

α 与 λ 只在内层**验证集**上按 mean NLL 选优（与神经模型的 checkpoint
选择同口径），并联合选择、最终报告 val 上最强的简单模型；选择记录完整
落盘，供审阅者核对“没有偷偷用测试集调参”。测试集 split 需显式
`--allow_test` 并写入数据目录的访问账本（与 lega.evaluate 同一账本），
保持尘封协议一致。

约定：训练统计只来自 `train.bin`；评测 split 内容绝不参与拟合或选参
（split="val" 时 α 选择与报告同集，属轻微乐观，已在输出中显式注明）。
"""
import argparse
import json
import math
import os

import numpy as np

from .data import load_bin, load_meta
from .datacheck import append_test_access
from .preflight import (allow_protocol_violations,
                        data_protocol_preflight)
from .repro import test_data_fingerprint
from .utils import strict_int

# α 候选：跨 3 个数量级，覆盖稀疏/密集两种极端；只数 3 档，避免把
# “选平滑系数”变成又一次大规模调参。
ALPHA_GRID = (0.01, 0.1, 1.0)
# 插值 bigram 的权重网格：P = λ·P_addα(w|prev) + (1−λ)·P_addα(w)。
# λ 只在 val 上选；与 add-α 的 α 联合扫描，避免“平滑弱所以基线弱”。
INTERP_LAMBDA_GRID = (0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 1.0)


# 训练流统计的分块粒度（token 数）：每块内升位并做块内 np.unique。
_PAIR_CHUNK_TOKENS = 4_000_000


def _pair_keys(ids: np.ndarray, vocab_size: int) -> np.ndarray:
    """token 序列 → 相邻对的唯一整数键 prev*V + cur（int64 防溢出）。

    ids 已是 int64 时不做复制；乘积结果缓冲原地累加后继 token。
    """
    ids = np.asarray(ids, dtype=np.int64)
    keys = ids[:-1] * int(vocab_size)
    keys += ids[1:]
    return keys


def _merge_two_counted(a, b):
    """归并两个按 key 升序去重的 (keys, counts) 分块，相同 key 计数相加。"""
    ka, ca = a
    kb, cb = b
    if len(ka) == 0:
        return b
    if len(kb) == 0:
        return a
    keys = np.concatenate((ka, kb))
    counts = np.concatenate((ca, cb))
    order = np.argsort(keys, kind="stable")
    keys = keys[order]
    counts = counts[order]
    first = np.empty(keys.shape[0], dtype=bool)
    first[0] = True
    np.not_equal(keys[1:], keys[:-1], out=first[1:])
    idx = np.flatnonzero(first)
    return keys[idx], np.add.reduceat(counts, idx)


def _merge_counted(pieces):
    """多路两两归并各块内已排序去重的 (keys, counts)，得到全局有序唯一频次。"""
    while len(pieces) > 1:
        merged = []
        for i in range(0, len(pieces), 2):
            merged.append(_merge_two_counted(pieces[i], pieces[i + 1])
                          if i + 1 < len(pieces) else pieces[i])
        pieces = merged
    return pieces[0]


def _pair_counts_chunked(ids: np.ndarray, vocab_size: int):
    """分块统计相邻对频次；返回按 key 升序去重的 (pair_keys, pair_counts)。

    每块（可配 chunk_tokens）尾部多取一个 token，使块内相邻对覆盖
    [start, end-1) 且块间不重不漏；块内 np.unique 后归并有序块结果。
    """
    n = int(len(ids))
    if n < 2:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    vocab_size = int(vocab_size)
    chunk_tokens = int(_PAIR_CHUNK_TOKENS)
    pieces = []
    start = 0
    while start < n - 1:
        end = min(n, start + chunk_tokens + 1)
        chunk = np.asarray(ids[start:end], dtype=np.int64)
        keys = chunk[:-1] * vocab_size
        keys += chunk[1:]
        uk, uc = np.unique(keys, return_counts=True)
        if len(uk):
            pieces.append((uk, uc))
        start = end - 1
    if not pieces:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return _merge_counted(pieces)


def fit_counts(ids: np.ndarray, vocab_size: int):
    """在训练流上统计一元/二元频次；返回 (uni_counts, pair_keys, pair_counts)。

    一元分块 bincount；二元的 V² 稀疏不可枚举，按块生成相邻对、
    块内 np.unique 后归并。
    """
    ids = np.asarray(ids)
    vocab_size = int(vocab_size)
    uni = np.zeros(max(1, vocab_size), dtype=np.float64)
    n = int(len(ids))
    for start in range(0, n, _PAIR_CHUNK_TOKENS):
        chunk = np.asarray(ids[start:start + _PAIR_CHUNK_TOKENS],
                           dtype=np.int64)
        uni += np.bincount(chunk, minlength=vocab_size)
    pair_keys, pair_counts = _pair_counts_chunked(ids, vocab_size)
    return uni, pair_keys, pair_counts.astype(np.float64)


def _bigram_counts_for(pair_keys: np.ndarray, pair_counts: np.ndarray,
                       keys: np.ndarray) -> np.ndarray:
    """查 eval 二元组的训练频次：排序键上二分，未出现即 0。"""
    if len(pair_keys) == 0 or len(keys) == 0:
        return np.zeros(len(keys), dtype=np.float64)
    pos = np.searchsorted(pair_keys, keys)
    pos_clip = np.clip(pos, 0, len(pair_keys) - 1)
    hit = pair_keys[pos_clip] == keys
    return np.where(hit, pair_counts[pos_clip], 0.0)


def _context_counts(pair_keys: np.ndarray, pair_counts: np.ndarray,
                    vocab_size: int) -> np.ndarray:
    """训练流中每个 token 作为“前文”出现的次数（即其后确实有 token 的次数）。

    不能用 ``uni[prev]``：一元频次包含训练流最后一个 token，但它没有后继，
    不可能是任何二元事件的前文。把它计入分母会让 add-α 二元分布在每个
    prev 上凭空多出一个（几乎不可能的）后继位置，系统性低估 PPL。
    """
    vocab_size = int(vocab_size)
    if len(pair_keys) == 0:
        return np.zeros(max(1, vocab_size), dtype=np.float64)
    prev = np.asarray(pair_keys, dtype=np.int64) // vocab_size
    return np.bincount(prev, weights=pair_counts,
                       minlength=max(1, vocab_size)).astype(np.float64)


def _validate_alpha(alpha) -> float:
    """平滑系数必须是有限非负数；NaN/Inf/负值会把 PPL 变成非法数值。"""
    try:
        alpha = float(alpha)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"平滑系数 α 非数值: {alpha!r}") from None
    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError(f"平滑系数 α 必须是有限非负数: {alpha!r}")
    return alpha


def _validate_lambda(lam) -> float:
    """插值权重必须在 [0, 1] 内，否则组合出来的“概率”不再合法。"""
    try:
        lam = float(lam)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"插值权重 λ 非数值: {lam!r}") from None
    if not math.isfinite(lam) or not 0.0 <= lam <= 1.0:
        raise ValueError(f"插值权重 λ 必须在 [0, 1]: {lam!r}")
    return lam


def _smoothed_probs(counts, denominator) -> np.ndarray:
    """加 α 概率；α=0 且上下文频次为 0 时概率记为 0 而不是 0/0=NaN。"""
    numerator = np.asarray(counts, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    out = np.zeros(np.broadcast(numerator, denominator).shape,
                   dtype=np.float64)
    np.divide(numerator, denominator, out=out, where=denominator > 0)
    # 0 概率的 -log 是 Inf，会让报告出现 NaN/Inf；与插值路径一致地
    # 用 1e-300 下限保留“几乎不可能”的排序，同时输出有限 NLL。
    return np.clip(out, 1e-300, None)


def ppl_unigram(eval_ids: np.ndarray, uni: np.ndarray, alpha: float) -> dict:
    """加 α 平滑一元模型的 per-token 平均负对数似然与 PPL。"""
    alpha = _validate_alpha(alpha)
    vocab = int(len(uni))
    n = int(len(eval_ids))
    if n == 0:
        return {"ppl": None, "mean_nll": None, "n_tokens": 0}
    total = float(uni.sum())
    # uint16 流可直接做花式索引
    c = uni[np.asarray(eval_ids)]
    probs = _smoothed_probs(c + alpha, total + alpha * vocab)
    mean_nll = float(-np.log(probs).mean())
    return {"ppl": float(math.exp(min(20.0, mean_nll))),
            "mean_nll": mean_nll, "n_tokens": n,
            "ppl_clamped": bool(mean_nll > 20.0)}


def _bigram_terms(eval_ids: np.ndarray, uni: np.ndarray,
                  pair_keys: np.ndarray, pair_counts: np.ndarray,
                  vocab_size: int):
    """评测流的逐 token 二元项：返回 (相邻对训练频次, 前文频次, 计分 token 数)。"""
    if len(eval_ids) < 2:
        return None, None, 0
    ids = np.asarray(eval_ids, dtype=np.int64)
    keys = _pair_keys(ids, vocab_size)
    prev = ids[:-1]
    c_pair = _bigram_counts_for(pair_keys, pair_counts, keys)
    c_prev = _context_counts(pair_keys, pair_counts, vocab_size)[prev]
    return c_pair, c_prev, int(len(keys))


def _bigram_ppl(c_pair: np.ndarray, c_prev: np.ndarray, n_tokens: int,
                alpha: float, vocab_size: int) -> dict:
    """由预计算的逐 token 频次给出加 α 平滑二元模型的 PPL。"""
    alpha = _validate_alpha(alpha)
    if n_tokens <= 0:
        return {"ppl": None, "mean_nll": None, "n_tokens": 0}
    probs = _smoothed_probs(c_pair + alpha,
                            c_prev + alpha * int(vocab_size))
    mean_nll = float(-np.log(probs).mean())
    return {"ppl": float(math.exp(min(20.0, mean_nll))),
            "mean_nll": mean_nll, "n_tokens": n_tokens,
            "ppl_clamped": bool(mean_nll > 20.0)}


def ppl_bigram(eval_ids: np.ndarray, uni: np.ndarray, pair_keys: np.ndarray,
               pair_counts: np.ndarray, alpha: float, vocab_size: int) -> dict:
    """加 α 平滑二元模型；第一个 token 无前文，从第二个 token 起计分。"""
    c_pair, c_prev, n_tokens = _bigram_terms(eval_ids, uni, pair_keys,
                                             pair_counts, vocab_size)
    return _bigram_ppl(c_pair, c_prev, n_tokens, alpha, vocab_size)


def _interp_ppl_from_counts(cur_ids: np.ndarray, uni: np.ndarray,
                            c_pair: np.ndarray, c_prev: np.ndarray,
                            n_tokens: int, alpha_uni: float, alpha_bi: float,
                            lam: float, vocab_size: int) -> dict:
    """由“当前 token + 预计算上下文频次”给出插值 bigram 的 PPL。"""
    alpha_uni = _validate_alpha(alpha_uni)
    alpha_bi = _validate_alpha(alpha_bi)
    lam = _validate_lambda(lam)
    if n_tokens <= 0:
        return {"ppl": None, "mean_nll": None, "n_tokens": 0}
    cur = np.asarray(cur_ids, dtype=np.int64)
    vocab_size = int(vocab_size)
    total = float(uni.sum())
    p_uni = _smoothed_probs(uni[cur] + alpha_uni,
                            total + alpha_uni * vocab_size)
    p_bi = _smoothed_probs(c_pair + alpha_bi,
                           c_prev + alpha_bi * vocab_size)
    p = float(lam) * p_bi + (1.0 - float(lam)) * p_uni
    mean_nll = float(-np.log(np.clip(p, 1e-300, None)).mean())
    return {"ppl": float(math.exp(min(20.0, mean_nll))),
            "mean_nll": mean_nll, "n_tokens": int(n_tokens),
            "ppl_clamped": bool(mean_nll > 20.0)}


def _interp_ppl_from_terms(eval_ids: np.ndarray, uni: np.ndarray,
                           c_pair: np.ndarray, c_prev: np.ndarray,
                           n_tokens: int, alpha_uni: float, alpha_bi: float,
                           lam: float, vocab_size: int) -> dict:
    """由预计算频次给出插值 bigram 的 PPL（λ 在 val 上选）。"""
    # 评测 token 从第二个开始计分；与 _bigram_terms 的 c_pair/c_prev 对齐。
    ids = np.asarray(eval_ids, dtype=np.int64)
    cur = ids[1:1 + int(n_tokens)]
    return _interp_ppl_from_counts(cur, uni, c_pair, c_prev, n_tokens,
                                   alpha_uni, alpha_bi, lam, vocab_size)


def ppl_interpolated_bigram(eval_ids: np.ndarray, uni: np.ndarray,
                            pair_keys: np.ndarray, pair_counts: np.ndarray,
                            alpha_uni: float, alpha_bi: float, lam: float,
                            vocab_size: int) -> dict:
    """插值 bigram：λ·P_addα(w|prev) + (1−λ)·P_addα(w)。

    相比纯 add-α bigram，未出现过的上下文不再只回退到近似均匀分布，
    而是回退到一元分布；这是仍属“简单统计基线”的更强版本。
    """
    c_pair, c_prev, n_tokens = _bigram_terms(eval_ids, uni, pair_keys,
                                             pair_counts, vocab_size)
    return _interp_ppl_from_terms(eval_ids, uni, c_pair, c_prev, n_tokens,
                                  alpha_uni, alpha_bi, lam, vocab_size)


def window_positions(n_tokens: int, ctx: int, iters: int,
                     return_starts: bool = False):
    """神经评测窗口实际打分的 token 位置（与 train._val_eval_indices 同规则）。

    神经 PPL 只在 `eval_iters` 个等间隔、不重复的窗口上评测：窗口序号按
    `_val_eval_indices` 生成（含首尾、iters 超窗口数时取全部），每个窗口
    消费 x=ids[s·ctx:(s+1)·ctx]、y=ids[s·ctx+1:(s+1)·ctx+1]，因此被打分的
    token 位置是每个窗口的 s·ctx+1 .. s·ctx+ctx（0-based）。n-gram 基线若在
    整个 split 上打分，两侧就是不同 token 集合上的 PPL（“同 split”不等于
    “同 token”），比较存在未知方向的偏移；本函数给出逐位置对齐的评分集合。

    规则必须与 `lega.train._val_eval_indices` 完全一致（tests 中有逐例等价
    断言）。窗口规格非法或评测流过短时返回 None（调用方按缺失处理）。
    """
    try:
        n_tokens = strict_int(n_tokens, "n_tokens")
        ctx = strict_int(ctx, "ctx")
        iters = strict_int(iters, "iters")
    except (TypeError, ValueError):
        return None
    if ctx < 1 or iters < 1:
        return None
    n_windows = (n_tokens - 1) // ctx
    if n_windows <= 0:
        return None
    n_eval = min(iters, n_windows)
    if n_eval == 1:
        starts = np.array([(n_windows - 1) // 2], dtype=np.int64)
    else:
        starts = (np.arange(n_eval, dtype=np.int64) * (n_windows - 1)
                  // (n_eval - 1))
    offsets = np.arange(1, ctx + 1, dtype=np.int64)
    pos = (starts[:, None] * ctx + offsets[None, :]).reshape(-1)
    # return_starts 供逐窗口对齐：窗口序号是 s·ctx（与神经评测的
    # window_indices 完全一致），pos 只是这些窗口内被计分的 token 位置。
    return (pos, starts) if return_starts else pos


def _validate_vocab_and_ids(meta: dict, arrays) -> int:
    """校验 meta.vocab_size 与各 split 的 token id，避免用错误词表算基线。

    meta 是磁盘上的外部输入：``int("4.5")`` 会抛错但 ``int(3.7)``/``int(True)``
    会静默截断；更危险的是 bin 中的 id 超出 vocab_size 时，二元键
    ``prev * V + cur`` 会发生不同 token 对撞，context 频次与分母也按错误 V
    计算，最终 PPL 看似正常却对应另一个模型。这里在拟合前显式拒绝。
    """
    vocab_size = strict_int(meta.get("vocab_size"), "meta.vocab_size")
    if vocab_size <= 0:
        raise ValueError(
            f"meta.vocab_size 必须为正整数，收到 {vocab_size}")
    for name, ids in arrays:
        n = int(len(ids))
        if n == 0:
            continue
        lo, hi = int(np.min(ids)), int(np.max(ids))
        if lo < 0 or hi >= vocab_size:
            raise ValueError(
                f"{name} 含非法 token id（范围 {lo}..{hi}，"
                f"meta.vocab_size={vocab_size}，合法范围 "
                f"0..{vocab_size - 1}）：meta 与 bin 不一致，"
                "拒绝用错误词表计算 n-gram 基线")
    return vocab_size


def _set_ngram_window_losses(report: dict, probs, starts, n_eval: int,
                             ctx: int, order: str) -> None:
    """把选定简单模型的逐 token NLL 聚成逐窗口平均损失（与神经口径一致）。

    神经 `window_losses` 是每个窗口内目标 token 的平均交叉熵，窗口序号是
    窗口起点 s·ctx。这里必须用同样的 reshape/mean，才能在下游逐窗口配对。
    """
    if starts is None or n_eval <= 0:
        return
    nll = -np.log(np.clip(np.asarray(probs, dtype=np.float64),
                          1e-300, None))
    if nll.size != int(n_eval) * int(ctx):
        return
    losses = nll.reshape(int(n_eval), int(ctx)).mean(axis=1)
    report["window_losses_matched"] = [float(v) for v in losses]
    report["window_indices_matched"] = [int(v) for v in starts[:int(n_eval)]]
    report["window_losses_matched_model"] = order


def evaluate(data_dir: str, split: str = "val",
             alphas=ALPHA_GRID, allow_test: bool = False,
             window_spec: dict = None) -> dict:
    """拟合并评测 n-gram 基线，返回完整报告 dict（不写文件）。

    选参协议：α 与插值权重 λ 只在 **val** 上按 mean NLL 联合选优
    （val 不存在或过短时报错），`selected_model` 取 val 上最强的简单
    模型（unigram/bigram/interpolated_bigram）；随后在 `split` 上按
    选定参数报告；split == "val" 时选参与报告同集，报告中以
    `alpha_selected_on`、`selected_model_selected_on` 与
    `selection_eval_same_split` 显式披露。

    window_spec（可选 {"ctx": int, "iters": int}）给出神经评测实际使用的
    窗口规格：此时额外计算 `ppl_window_matched`——只在 `window_positions`
    生成的、与神经评测逐位置一致的 token 上按同一选定 α 打分，供汇总页
    做真正的同 token 对比；窗口规格无效时以 `window_spec_error` 显式标注。
    """
    meta = load_meta(data_dir)
    # 规范化候选网格：调用方可能传生成器；后面要在 uni/bi/插值三段重复扫描。
    # 非法 α（NaN/Inf/负值）必须显式拒绝，不能静默参与选参并写出非法 JSON。
    alphas = tuple(_validate_alpha(a) for a in alphas)
    if not alphas:
        raise ValueError("平滑系数候选网格不能为空")
    # 训练流保持 uint16 memmap：fit_counts 内部分块升位/分块计数。
    train_ids = load_bin(data_dir, "train")
    test_access = None
    if split == "test":
        # 必须显式传 bool True：字符串 "false" 等真值不能开锁。
        if allow_test is not True:
            raise PermissionError(
                "test 集已尘封：评测 n-gram 基线需显式 allow_test=True"
                "（不接受非布尔真值），并会写入访问账本")
        eval_ids = np.asarray(load_bin(data_dir, "test", allow_test=True),
                              dtype=np.int64)
        # 访问账本必须在选参/拟合/读 val 之前落盘：否则 val 缺失、拟合失败
        # 或账本写入异常会让 test 已被消费却没有访问记录，尘封协议失效。
        test_access = append_test_access(data_dir, {
            "split": "test", "kind": "ngram_baseline",
            "ckpt": "ngram-baseline",
            "test_data_fingerprint": test_data_fingerprint(data_dir),
            "note": "简单统计基线评测（α 在 val 上选择）"})
    else:
        eval_ids = np.asarray(load_bin(data_dir, split), dtype=np.int64)
    # 选参流固定为 val：即使调用方要求报告 train（诊断用），α/λ 也不能
    # 在报告集上选择，否则 alpha_selected_on 的声明与真实协议不符。
    sel_ids = (eval_ids if split == "val"
               else np.asarray(load_bin(data_dir, "val"), dtype=np.int64))
    vocab_size = _validate_vocab_and_ids(
        meta, (("train", train_ids), (split, eval_ids), ("val", sel_ids)))
    uni, pair_keys, pair_counts = fit_counts(train_ids, vocab_size)
    # 选参与报告各自评测流的逐 token 二元项：α 只影响频次到 PPL 的换算，
    # 不改变二元组频次本身；split == "val" 时两个流是同一数组，可共用。
    sel_terms = _bigram_terms(sel_ids, uni, pair_keys, pair_counts, vocab_size)
    eval_terms = (sel_terms if eval_ids is sel_ids
                  else _bigram_terms(eval_ids, uni, pair_keys, pair_counts,
                                     vocab_size))
    # α 选择：val 上按 mean_nll（等价于 PPL 排序，避免 exp 截断干扰选择）
    best = {"unigram": None, "bigram": None}
    for order in ("unigram", "bigram"):
        best_nll = None
        for a in alphas:
            if order == "unigram":
                m = ppl_unigram(sel_ids, uni, a)
            else:
                m = _bigram_ppl(*sel_terms, a, vocab_size)
            if m["mean_nll"] is None:
                continue
            if best_nll is None or m["mean_nll"] < best_nll:
                best_nll = m["mean_nll"]
                best[order] = float(a)
    # 插值 bigram：P = λ·P_addα(w|prev) + (1−λ)·P_addα(w)。λ 与两个 α
    # 在 val 上联合选择（仍按 mean NLL）；这保证“神经 vs 简单基线”的
    # 对照不会输给一个故意没调好的纯 bigram。ppl_by_lambda 记录每个 λ
    # 的最优 α 组合，供审阅选择面是否平坦。
    interp = None
    interp_by_lambda = {}
    if sel_terms[2] > 0:
        for lam in INTERP_LAMBDA_GRID:
            best_lam = None
            for au in alphas:
                for ab in alphas:
                    m = _interp_ppl_from_terms(sel_ids, uni, *sel_terms, au, ab,
                                               lam, vocab_size)
                    if m["mean_nll"] is None:
                        continue
                    if (best_lam is None
                            or m["mean_nll"] < best_lam["val_mean_nll"]):
                        best_lam = {"lambda": float(lam),
                                    "alpha_unigram": float(au),
                                    "alpha_bigram": float(ab),
                                    "val_mean_nll": m["mean_nll"]}
            if best_lam is not None:
                interp_by_lambda[str(lam)] = best_lam
                if (interp is None
                        or best_lam["val_mean_nll"] < interp["val_mean_nll"]):
                    interp = best_lam
    # 简单基线内部也按 val 选“最强模型”，而不是固定拿 bigram 当靶子。
    val_mean_nll = {}
    if best["unigram"] is not None:
        val_mean_nll["unigram"] = ppl_unigram(
            sel_ids, uni, best["unigram"])["mean_nll"]
    if best["bigram"] is not None:
        val_mean_nll["bigram"] = _bigram_ppl(
            *sel_terms, best["bigram"], vocab_size)["mean_nll"]
    if interp is not None:
        val_mean_nll["interpolated_bigram"] = interp["val_mean_nll"]
    usable = {k: v for k, v in val_mean_nll.items() if v is not None}
    selected_model = min(usable, key=usable.get) if usable else None
    report = {
        "baseline": ("smoothed n-gram (add-α unigram/bigram + val-tuned "
                     "interpolation), no gradient training"),
        "split": split,
        "allow_test": bool(allow_test),
        "tokenizer": meta.get("tokenizer"),
        "vocab_size": vocab_size,
        "train_tokens": int(len(train_ids)),
        "eval_tokens": int(len(eval_ids)),
        "alpha_grid": [float(a) for a in alphas],
        "interp_lambda_grid": [float(x) for x in INTERP_LAMBDA_GRID],
        "selected_alpha": best,
        "selected_interpolated": interp,
        "interpolated_by_lambda": interp_by_lambda,
        "val_mean_nll": val_mean_nll,
        "selected_model": selected_model,
        "alpha_selected_on": "val (mean NLL，与 model_best 选择同口径)",
        "selected_model_selected_on": "val (mean NLL，与 model_best 选择同口径)",
        "selection_eval_same_split": bool(split == "val"),
        "ppl": {},
        "ppl_by_alpha": {},
    }
    if split == "test":
        report["test_access"] = test_access
    for order in ("unigram", "bigram", "interpolated_bigram"):
        if order == "unigram":
            a = best["unigram"]
            m = ppl_unigram(eval_ids, uni, a) if a is not None else None
        elif order == "bigram":
            a = best["bigram"]
            m = (_bigram_ppl(*eval_terms, a, vocab_size)
                 if a is not None else None)
        else:
            m = (_interp_ppl_from_terms(
                eval_ids, uni, *eval_terms, interp["alpha_unigram"],
                interp["alpha_bigram"], interp["lambda"], vocab_size)
                if interp is not None else None)
        if m is None:
            report["ppl"][order] = None
            continue
        report["ppl"][order] = m["ppl"]
        # test 上只消费一次：仅报告 val 选定的参数，不额外扫网格
        # （避免"一次评测"被多次读数的灰色空间）。
        if split != "test" and order in ("unigram", "bigram"):
            by_alpha = {}
            for a2 in alphas:
                if order == "unigram":
                    m2 = ppl_unigram(eval_ids, uni, a2)
                else:
                    m2 = _bigram_ppl(*eval_terms, a2, vocab_size)
                by_alpha[str(a2)] = m2["ppl"]
            report["ppl_by_alpha"][order] = by_alpha
    # 窗口对齐口径：只在神经评测窗口的目标 token 上打分（见 window_positions）。
    # 与全 split 口径使用同一选定 α（test 上不额外扫网格，仍是单次消费）。
    if window_spec:
        try:
            if not isinstance(window_spec, dict):
                raise TypeError
            # 用严格整数解析：int(128.5)=128、int(True)=1 会改变“同窗口”
            # 的对齐口径——window_positions 与 train._val_eval_indices 的
            # 逐位置一致依赖 ctx/iters 是精确整数。bool/小数视为非法规格。
            spec_ctx = (None if window_spec.get("ctx") is None
                        else strict_int(window_spec.get("ctx") or 0,
                                        "window_spec.ctx"))
            spec_iters = (None if window_spec.get("iters") is None
                          else strict_int(window_spec.get("iters") or 0,
                                          "window_spec.iters"))
        except (ValueError, TypeError):
            spec_ctx = spec_iters = None
        pos = (window_positions(len(eval_ids), spec_ctx, spec_iters)
               if spec_ctx is not None and spec_iters is not None else None)
        if pos is not None and spec_ctx is not None and spec_iters is not None:
            pos, starts = window_positions(
                len(eval_ids), spec_ctx, spec_iters, return_starts=True)
        else:
            starts = None
        if pos is None:
            report["window_spec"] = None
            report["window_spec_error"] = (
                f"窗口规格无效或评测流过短（ctx={spec_ctx}, iters={spec_iters}, "
                f"eval_tokens={len(eval_ids)}），无法逐位置对齐")
        else:
            n_windows = (len(eval_ids) - 1) // spec_ctx
            n_eval = min(spec_iters, n_windows)
            wm = {
                "n_tokens": int(pos.size),
                "positions": "每个窗口 s·ctx+1 .. s·ctx+ctx（0-based），"
                             "与 get_val_windows 的目标 token 集合逐位置一致",
                "unigram": None, "bigram": None,
                "interpolated_bigram": None,
            }
            pos = np.asarray(pos, dtype=np.int64)
            ctx_counts = _context_counts(pair_keys, pair_counts, vocab_size)
            for order in ("unigram", "bigram", "interpolated_bigram"):
                if order in ("unigram", "bigram"):
                    a = best[order]
                    if a is None:
                        continue
                    if order == "unigram":
                        m = ppl_unigram(eval_ids[pos], uni, a)
                        if order == selected_model:
                            cur = np.asarray(eval_ids[pos], dtype=np.int64)
                            probs = _smoothed_probs(
                                uni[cur] + a,
                                float(uni.sum()) + a * vocab_size)
                            _set_ngram_window_losses(
                                report, probs, starts, n_eval, spec_ctx, order)
                    else:
                        prev = np.asarray(eval_ids[pos - 1], dtype=np.int64)
                        keys = prev * vocab_size
                        keys += np.asarray(eval_ids[pos], dtype=np.int64)
                        c_pair = _bigram_counts_for(pair_keys, pair_counts,
                                                    keys)
                        m = _bigram_ppl(c_pair, ctx_counts[prev],
                                        int(pos.size), a, vocab_size)
                        if order == selected_model:
                            probs = _smoothed_probs(
                                c_pair + a,
                                ctx_counts[prev] + a * vocab_size)
                            _set_ngram_window_losses(
                                report, probs, starts, n_eval, spec_ctx, order)
                else:
                    if interp is None:
                        continue
                    prev = np.asarray(eval_ids[pos - 1], dtype=np.int64)
                    keys = prev * vocab_size
                    keys += np.asarray(eval_ids[pos], dtype=np.int64)
                    c_pair = _bigram_counts_for(pair_keys, pair_counts, keys)
                    m = _interp_ppl_from_counts(
                        eval_ids[pos], uni, c_pair, ctx_counts[prev],
                        int(pos.size), interp["alpha_unigram"],
                        interp["alpha_bigram"], interp["lambda"], vocab_size)
                    if order == selected_model:
                        cur = np.asarray(eval_ids[pos], dtype=np.int64)
                        total = float(uni.sum())
                        p_uni = _smoothed_probs(
                            uni[cur] + interp["alpha_unigram"],
                            total + interp["alpha_unigram"] * vocab_size)
                        p_bi = _smoothed_probs(
                            c_pair + interp["alpha_bigram"],
                            ctx_counts[prev]
                            + interp["alpha_bigram"] * vocab_size)
                        probs = (interp["lambda"] * p_bi
                                 + (1.0 - interp["lambda"]) * p_uni)
                        _set_ngram_window_losses(
                            report, probs, starts, n_eval, spec_ctx, order)
                wm[order] = m["ppl"]
            report["ppl_window_matched"] = wm
            report["window_spec"] = {
                "ctx": spec_ctx, "iters": spec_iters,
                "n_windows": int(n_windows), "n_eval": int(n_eval),
                "n_tokens": int(pos.size),
            }
    return report


def main():
    p = argparse.ArgumentParser(
        description="简单统计基线：一元/二元/插值二元平滑 n-gram PPL")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--out", default=None, help="JSON 输出路径（默认打印）")
    p.add_argument("--alphas", default=None,
                   help="逗号分隔的 α 网格，默认 0.01,0.1,1.0")
    p.add_argument("--allow_test", action="store_true",
                   help="允许在 test 上评测（写访问账本）")
    p.add_argument("--window_ctx", type=int, default=None,
                   help="神经评测的窗口长度（与 --window_iters 一起给出时，"
                        "额外报告与神经评测逐位置对齐的 PPL）")
    p.add_argument("--window_iters", type=int, default=None,
                   help="神经评测的窗口数（与 --window_ctx 成对使用）")
    p.add_argument("--allow_data_protocol_violations", action="store_true",
                   help="即使 data_report 标记泄露也继续评测（仅用于复现历史"
                        "实验；结果不得用于正式结论）")
    args = p.parse_args()
    if (args.window_ctx is None) != (args.window_iters is None):
        p.error("--window_ctx 与 --window_iters 必须成对给出")
    data_preflight = data_protocol_preflight(
        args.data_dir,
        allow_override=allow_protocol_violations(
            args.allow_data_protocol_violations))
    if not data_preflight["report_present"]:
        print("[ngram] 警告：数据目录缺少 data_report.json，"
              "无法执行评测前防泄漏闸门（旧数据或未走统一入口）")
    elif data_preflight["critical_violations"]:
        print(f"[ngram] ⚠ 数据协议自检发现 "
              f"{len(data_preflight['critical_violations'])} 个严重问题，"
              "但已显式放行（结果不得用于正式结论）")
    window_spec = ({"ctx": args.window_ctx, "iters": args.window_iters}
                   if args.window_ctx is not None else None)
    alphas = (tuple(float(x) for x in args.alphas.split(","))
              if args.alphas else ALPHA_GRID)
    report = evaluate(args.data_dir, split=args.split, alphas=alphas,
                      allow_test=args.allow_test, window_spec=window_spec)
    report["data_protocol_preflight"] = data_preflight
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    print(text)
    print(f"[ngram] {args.split} PPL: unigram={report['ppl'].get('unigram')} "
          f"bigram={report['ppl'].get('bigram')} "
          f"interp={report['ppl'].get('interpolated_bigram')} "
          f"(selected={report.get('selected_model')}, "
          f"λ={((report.get('selected_interpolated') or {}).get('lambda'))})")


if __name__ == "__main__":
    main()
