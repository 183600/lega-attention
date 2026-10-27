"""集中度统计 c 的长度零假设诊断。

LEGA 文档里“均匀注意力时 c=0”只说明**均匀分布这一特例**与长度无关。
一般情况下::

    c = log(n) + log(max_j p_j)

若 max_j p_j 不按 1/n 收缩，c 仍会随 n 增长。因此“c 已经消除长度混淆”
必须用零假设分布来检验，而不是只看均匀分布单元测试。本模块提供与模型
无关的合成诊断：

* ``random``   : 每个位置独立 N(0, score_sd²) 打分（无有效信息）；
* ``repeated`` : 把同一组基础打分复制若干次（重复上下文）；
* ``uniform``  : 所有打分相同（均匀注意力，理论上 c=0）。

对每档长度重复 ``n_trials`` 次，报告 c 的均值/标准差/分位数，以及
``c ~ log2(n)`` 的斜率。斜率显著偏离 0 就说明 c 仍携带长度信息，不能
把 c 直接解释为“已长度归一化的证据强度”。

用法::

    python -m lega.length_null --lengths 32,64,128,256,512 --out results/length_null.json
"""
from __future__ import annotations

import argparse
import json
import math
from numbers import Integral, Real
from typing import Iterable, List

import numpy as np

__all__ = ["simulate_concentration", "length_null_report", "main"]


def _positive_lengths(lengths: Iterable[int]) -> List[int]:
    out = []
    for value in lengths:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
            raise ValueError(f"长度必须是正整数，收到 {value!r}")
        value = int(value)
        if value < 2:
            raise ValueError(f"长度必须 >= 2（否则 softmax 只有一项），收到 {value}")
        out.append(value)
    if not out:
        raise ValueError("lengths 不能为空")
    return out


def _positive_float(value, name: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} 必须是数值，收到 {value!r}")
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} 必须是数值，收到 {value!r}") from exc
    if not math.isfinite(out) or out < 0.0 or (out == 0.0 and not allow_zero):
        raise ValueError(f"{name} 必须是{'非负' if allow_zero else '正'}有限数值，"
                         f"收到 {value!r}")
    return out


def simulate_concentration(length: int, *, score_sd: float = 1.0,
                           mode: str = "random", n_trials: int = 2000,
                           seed: int = 0) -> np.ndarray:
    """返回 ``n_trials`` 个 c = log(n) + log(max softmax p) 样本。"""
    length = _positive_lengths([length])[0]
    score_sd = _positive_float(score_sd, "score_sd", allow_zero=True)
    n_trials = int(_positive_lengths([max(2, int(n_trials))])[0])
    if mode not in ("random", "repeated", "uniform"):
        raise ValueError(f"mode 必须是 random/repeated/uniform，收到 {mode!r}")
    rng = np.random.default_rng(int(seed))
    if mode == "uniform":
        scores = np.zeros((n_trials, length), dtype=np.float64)
    elif mode == "repeated":
        # 生成一个固定基础向量并复制到 length：重复上下文的打分完全重复。
        base_len = max(1, min(length, 16))
        base = rng.normal(0.0, score_sd, size=(n_trials, base_len))
        reps = int(math.ceil(length / base_len))
        scores = np.tile(base, (1, reps))[:, :length]
    else:
        scores = rng.normal(0.0, score_sd, size=(n_trials, length))
    scores -= scores.max(axis=1, keepdims=True)
    p = np.exp(scores)
    p /= p.sum(axis=1, keepdims=True)
    max_p = p.max(axis=1)
    return np.log(float(length)) + np.log(np.maximum(max_p, 1e-300))


def _summary(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    q = np.quantile(values, [0.05, 0.5, 0.95])
    return {"mean": float(values.mean()), "std": float(values.std(ddof=1))
            if values.size > 1 else 0.0,
            "q05": float(q[0]), "median": float(q[1]), "q95": float(q[2]),
            "n": int(values.size)}


def _slope(lengths: List[int], means: List[float]) -> float | None:
    if len(lengths) < 2 or any(m is None for m in means):
        return None
    xs = np.log2(np.asarray(lengths, dtype=np.float64))
    ys = np.asarray(means, dtype=np.float64)
    xm, ym = xs.mean(), ys.mean()
    sxx = float(((xs - xm) ** 2).sum())
    if sxx <= 0.0:
        return None
    return float(((xs - xm) * (ys - ym)).sum() / sxx)


def length_null_report(lengths: Iterable[int], *, score_sd: float = 1.0,
                       modes=("random", "repeated", "uniform"),
                       n_trials: int = 2000, seed: int = 0) -> dict:
    """对每档长度/每种零假设生成 c 分布，并给出随 log2(n) 的斜率。"""
    lengths = _positive_lengths(lengths)
    modes = tuple(modes)
    out = {"lengths": lengths, "score_sd": float(score_sd),
           "n_trials": int(n_trials), "seed": int(seed),
           "modes": {}}
    for mode in modes:
        per = {}
        means = []
        for i, n in enumerate(lengths):
            vals = simulate_concentration(
                n, score_sd=score_sd, mode=mode, n_trials=n_trials,
                seed=int(seed) + 1000003 * i)
            per[str(n)] = _summary(vals)
            means.append(per[str(n)]["mean"])
        slope = _slope(lengths, means)
        out["modes"][mode] = {
            "per_length": per,
            "slope_c_per_log2_n": slope,
            "length_independent": (
                None if slope is None
                else bool(abs(slope) <= 0.05)),
            "note": (
                "slope=0 是“c 与长度无关”的必要证据；random 模式不仅凭"
                "均匀特例就宣称长度解耦。"),
        }
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="c 的长度零假设诊断")
    p.add_argument("--lengths", default="32,64,128,256,512,1024",
                   help="逗号分隔的正整数长度")
    p.add_argument("--score_sd", type=float, default=1.0,
                   help="随机打分标准差（0 表示均匀注意力）")
    p.add_argument("--modes", default="random,repeated,uniform")
    p.add_argument("--n_trials", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)
    lengths = [int(x) for x in args.lengths.split(",") if x.strip()]
    modes = [x.strip() for x in args.modes.split(",") if x.strip()]
    report = length_null_report(lengths, score_sd=args.score_sd,
                                modes=modes, n_trials=args.n_trials,
                                seed=args.seed)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"[length_null] -> {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
