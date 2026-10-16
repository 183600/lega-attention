"""通用工具测试：AUROC（并列取平均秩）与 Pearson 的边界行为。

AUROC 是机制分析的核心指标（门控分数区分 hit/miss），这里用逐对比较的
定义式独立复算，专门覆盖并列分数的平均秩路径。
"""
import math

import numpy as np

from lega.utils import auroc, pearson


def _brute_auroc(scores, labels):
    """定义式 AUROC：逐对比较，pos>neg 记 1、并列记 0.5、否则记 0。"""
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return float("nan")
    tot = 0.0
    for p in pos:
        for n in neg:
            tot += 1.0 if p > n else (0.5 if p == n else 0.0)
    return tot / (len(pos) * len(neg))


def test_auroc_matches_bruteforce():
    rng = np.random.default_rng(0)
    checked = 0
    for trial in range(300):
        n = int(rng.integers(4, 40))
        if trial % 2:
            # 小整数分数：制造大量并列（覆盖平均秩路径）
            scores = rng.integers(0, 4, size=n).astype(np.float64)
        else:
            scores = rng.normal(size=n)
        labels = (rng.random(n) < 0.5).astype(int)
        if labels.sum() in (0, n):
            continue
        got = auroc(scores, labels)
        ref = _brute_auroc(scores.tolist(), labels.tolist())
        assert abs(got - ref) < 1e-12, f"trial={trial}: auroc={got} != brute={ref}"
        checked += 1
    assert checked > 200
    print(f"  [ok] auroc 与定义式一致（{checked} 个随机用例，含并列）")


def test_auroc_ties_and_permutation():
    scores = np.array([1.0, 2.0, 2.0, 2.0, 3.0, 0.0])
    labels = np.array([0, 1, 0, 1, 0, 1])
    ref = _brute_auroc(scores.tolist(), labels.tolist())
    perm = np.random.default_rng(1).permutation(len(scores))
    got = auroc(scores[perm], labels[perm])
    assert abs(got - ref) < 1e-12, f"排列不变性失败: {got} != {ref}"

    # 全部并列：平均秩下 AUROC 必须恰为 0.5
    flat = auroc(np.ones(6), np.array([1, 0, 1, 0, 1, 0]))
    assert abs(flat - 0.5) < 1e-12, f"全并列应为 0.5，得到 {flat}"

    # 完全可分与完全反序
    sep = auroc(np.array([3.0, 4.0, 5.0, 0.0, 1.0, 2.0]),
                np.array([1, 1, 1, 0, 0, 0]))
    assert abs(sep - 1.0) < 1e-12, f"完全可分应为 1.0，得到 {sep}"
    rev = auroc(np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
                np.array([1, 1, 1, 0, 0, 0]))
    assert abs(rev - 0.0) < 1e-12, f"完全反序应为 0.0，得到 {rev}"
    print("  [ok] auroc 并列/排列/极端情形")


def test_auroc_single_class_nan():
    assert math.isnan(auroc(np.array([1.0, 2.0]), np.array([1, 1])))
    assert math.isnan(auroc(np.array([1.0, 2.0]), np.array([0, 0])))
    print("  [ok] 单一类别返回 nan（调用方负责转 None）")


def test_pearson_guards():
    x = np.arange(10, dtype=np.float64)
    assert abs(pearson(x, 2 * x + 1) - 1.0) < 1e-12
    assert abs(pearson(x, -x) + 1.0) < 1e-12
    # 常量输入/样本不足/非有限 → None（避免 NaN 字面量写入 JSON）
    assert pearson(x, np.ones_like(x)) is None
    assert pearson(x[:2], x[:2]) is None
    # NaN 成对剔除
    xn, yn = x.copy(), x.copy()
    xn[0] = np.nan
    yn[1] = np.nan
    assert abs(pearson(xn, yn) - 1.0) < 1e-9
    print("  [ok] pearson 边界（常量/样本不足/NaN 成对剔除）")
