"""简单统计基线：一元/二元平滑 n-gram 的 held-out PPL。

科研方法问题：没有简单基线时，“神经模型 PPL=xx”无法回答“数据本身
有多难、复杂架构是否真的比无参数的统计模型更好”。本模块提供一个
零训练、零调参空间（除平滑系数在验证集上选优）的 n-gram 基线：

  * 一元：P(w) = (c(w) + α) / (N + α·V)，即加 α 平滑
  * 二元：P(w_t | w_{t-1}) = (c(w_{t-1}, w_t) + α) / (c(w_{t-1}) + α·V)

平滑系数 α 只在**验证集**上选优（与神经模型的 checkpoint 选择同口径），
再在指定 held-out split 上报告 PPL；选择记录完整落盘，供审阅者核对
“没有偷偷用测试集调参”。测试集 split 需显式 `--allow_test` 并写入
数据目录的访问账本（与 lega.evaluate 同一账本），保持尘封协议一致。

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

# α 候选：跨 3 个数量级，覆盖稀疏/密集两种极端；只数 3 档，避免把
# “选平滑系数”变成又一次大规模调参。
ALPHA_GRID = (0.01, 0.1, 1.0)


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


def ppl_unigram(eval_ids: np.ndarray, uni: np.ndarray, alpha: float) -> dict:
    """加 α 平滑一元模型的 per-token 平均负对数似然与 PPL。"""
    vocab = int(len(uni))
    n = int(len(eval_ids))
    if n == 0:
        return {"ppl": None, "mean_nll": None, "n_tokens": 0}
    total = float(uni.sum())
    # uint16 流可直接做花式索引
    c = uni[np.asarray(eval_ids)]
    probs = (c + alpha) / (total + alpha * vocab)
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
    c_prev = uni[prev]
    return c_pair, c_prev, int(len(keys))


def _bigram_ppl(c_pair: np.ndarray, c_prev: np.ndarray, n_tokens: int,
                alpha: float, vocab_size: int) -> dict:
    """由预计算的逐 token 频次给出加 α 平滑二元模型的 PPL。"""
    if n_tokens <= 0:
        return {"ppl": None, "mean_nll": None, "n_tokens": 0}
    probs = (c_pair + alpha) / (c_prev + alpha * int(vocab_size))
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


def window_positions(n_tokens: int, ctx: int, iters: int):
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
    n_tokens, ctx, iters = int(n_tokens), int(ctx), int(iters)
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
    return (starts[:, None] * ctx + offsets[None, :]).reshape(-1)


def evaluate(data_dir: str, split: str = "val",
             alphas=ALPHA_GRID, allow_test: bool = False,
             window_spec: dict = None) -> dict:
    """拟合并评测 n-gram 基线，返回完整报告 dict（不写文件）。

    选参协议：α 在 **val** 上选优（val 不存在或过短时报错），随后在
    `split` 上按选定 α 报告；split == "val" 时选参与报告同集，报告中
    以 `alpha_selected_on` 与 `selection_eval_same_split` 显式披露。

    window_spec（可选 {"ctx": int, "iters": int}）给出神经评测实际使用的
    窗口规格：此时额外计算 `ppl_window_matched`——只在 `window_positions`
    生成的、与神经评测逐位置一致的 token 上按同一选定 α 打分，供汇总页
    做真正的同 token 对比；窗口规格无效时以 `window_spec_error` 显式标注。
    """
    meta = load_meta(data_dir)
    vocab_size = int(meta["vocab_size"])
    # 训练流保持 uint16 memmap：fit_counts 内部分块升位/分块计数。
    train_ids = load_bin(data_dir, "train")
    if split == "test":
        if not allow_test:
            raise PermissionError(
                "test 集已尘封：评测 n-gram 基线需显式 allow_test=True，"
                "并会写入访问账本")
        eval_ids = np.asarray(load_bin(data_dir, "test", allow_test=True),
                              dtype=np.int64)
        sel_ids = np.asarray(load_bin(data_dir, "val"), dtype=np.int64)
    else:
        eval_ids = np.asarray(load_bin(data_dir, split), dtype=np.int64)
        sel_ids = eval_ids
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
    report = {
        "baseline": "smoothed n-gram (add-α), no gradient training",
        "split": split,
        "allow_test": bool(allow_test),
        "tokenizer": meta.get("tokenizer"),
        "vocab_size": vocab_size,
        "train_tokens": int(len(train_ids)),
        "eval_tokens": int(len(eval_ids)),
        "alpha_grid": [float(a) for a in alphas],
        "selected_alpha": best,
        "alpha_selected_on": "val (mean NLL，与 model_best 选择同口径)",
        "selection_eval_same_split": bool(split == "val"),
        "ppl": {},
        "ppl_by_alpha": {},
    }
    if split == "test":
        report["test_access"] = append_test_access(data_dir, {
            "split": "test", "kind": "ngram_baseline",
            "ckpt": "ngram-baseline",
            "note": "简单统计基线评测（α 在 val 上选择）"})
    for order in ("unigram", "bigram"):
        a = best[order]
        if a is None:
            report["ppl"][order] = None
            continue
        if order == "unigram":
            m = ppl_unigram(eval_ids, uni, a)
        else:
            m = _bigram_ppl(*eval_terms, a, vocab_size)
        report["ppl"][order] = m["ppl"]
        # test 上只消费一次：仅报告 val 选定的 α，不额外扫网格
        # （避免"一次评测"被多次读数的灰色空间）。
        if split != "test":
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
        spec_ctx = int(window_spec.get("ctx", 0) or 0)
        spec_iters = int(window_spec.get("iters", 0) or 0)
        pos = window_positions(len(eval_ids), spec_ctx, spec_iters)
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
            }
            pos = np.asarray(pos, dtype=np.int64)
            for order in ("unigram", "bigram"):
                a = best[order]
                if a is None:
                    continue
                if order == "unigram":
                    m = ppl_unigram(eval_ids[pos], uni, a)
                else:
                    prev = np.asarray(eval_ids[pos - 1], dtype=np.int64)
                    keys = prev * vocab_size
                    keys += np.asarray(eval_ids[pos], dtype=np.int64)
                    c_pair = _bigram_counts_for(pair_keys, pair_counts, keys)
                    m = _bigram_ppl(c_pair, uni[prev], int(pos.size), a,
                                    vocab_size)
                wm[order] = m["ppl"]
            report["ppl_window_matched"] = wm
            report["window_spec"] = {
                "ctx": spec_ctx, "iters": spec_iters,
                "n_windows": int(n_windows), "n_eval": int(n_eval),
                "n_tokens": int(pos.size),
            }
    return report


def main():
    p = argparse.ArgumentParser(description="简单统计基线：平滑 n-gram PPL")
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
    args = p.parse_args()
    if (args.window_ctx is None) != (args.window_iters is None):
        p.error("--window_ctx 与 --window_iters 必须成对给出")
    window_spec = ({"ctx": args.window_ctx, "iters": args.window_iters}
                   if args.window_ctx is not None else None)
    alphas = (tuple(float(x) for x in args.alphas.split(","))
              if args.alphas else ALPHA_GRID)
    report = evaluate(args.data_dir, split=args.split, alphas=alphas,
                      allow_test=args.allow_test, window_spec=window_spec)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    print(text)
    print(f"[ngram] {args.split} PPL: unigram={report['ppl'].get('unigram')} "
          f"bigram={report['ppl'].get('bigram')}")


if __name__ == "__main__":
    main()
