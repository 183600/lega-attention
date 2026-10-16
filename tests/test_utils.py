"""通用工具测试：AUROC（并列取平均秩）、Pearson 与统计检验的边界行为。

AUROC 是机制分析的核心指标（门控分数区分 hit/miss），这里用逐对比较的
定义式独立复算，专门覆盖并列分数的平均秩路径。paired_t_test 是“多种子
Mean ± Std + 显著性”协议的核心，用 Student-t 已知临界值核对 p 值精度。
"""
import math

import numpy as np

from lega.utils import (auroc, holm_bonferroni, mean_std, paired_t_test,
                        pearson, student_t_two_sided_p)


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


def test_mean_std():
    # 样本标准差（ddof=1）：np.std([1,2,3,4], ddof=1) = sqrt(5/3)
    m, s = mean_std([1, 2, 3, 4])
    assert abs(m - 2.5) < 1e-12
    assert abs(s - math.sqrt(5.0 / 3.0)) < 1e-12
    # n=1：std 不可估计（不是 0），调用方据此标注 n=1
    m1, s1 = mean_std([3.5])
    assert m1 == 3.5 and s1 is None
    # 空/全 None/非有限值
    assert mean_std([]) == (None, None)
    assert mean_std([None, None]) == (None, None)
    m2, _ = mean_std([1.0, float("nan"), 3.0])
    assert m2 == 2.0
    print("  [ok] mean_std（ddof=1、n=1、空输入、NaN 剔除）")


def test_paired_t_matches_t_table():
    # Student-t 临界值表（双侧）：df=4, t=2.776445 -> p=0.05；df=3, t=3.182446 -> p=0.05
    assert abs(student_t_two_sided_p(2.776445, 4) - 0.05) < 1e-6
    assert abs(student_t_two_sided_p(3.182446, 3) - 0.05) < 1e-6
    assert abs(student_t_two_sided_p(0.0, 5) - 1.0) < 1e-12
    # 配对检验：diffs=[1,2,3,4] -> t=3.872983, df=3, p≈0.030466
    r = paired_t_test([1, 2, 3, 4], [0, 0, 0, 0])
    assert r["n"] == 4 and r["df"] == 3
    assert abs(r["mean_diff"] - 2.5) < 1e-12
    assert abs(r["t"] - 3.872983346207417) < 1e-9
    assert abs(r["p"] - 0.030466291662170977) < 1e-6
    # 交换顺序只翻符号，p 不变
    r2 = paired_t_test([0, 0, 0, 0], [1, 2, 3, 4])
    assert abs(r2["p"] - r["p"]) < 1e-12
    assert abs(r2["mean_diff"] + r["mean_diff"]) < 1e-12
    print("  [ok] paired_t_test 与 Student-t 临界值表一致")


def test_paired_t_edge_cases():
    # 完全相同的两组 -> t=0, p=1（不显著）
    r = paired_t_test([5.0, 5.0, 5.0], [5.0, 5.0, 5.0])
    assert r["t"] == 0.0 and r["p"] == 1.0
    # 常数差且方差为 0：均值非 0 记 p=0，均值 0 记 p=1
    r = paired_t_test([6.0, 6.0, 6.0], [5.0, 5.0, 5.0])
    assert r["p"] == 0.0 and math.isinf(r["t"])
    # n=1：无法检验，p 为 None
    r = paired_t_test([2.0], [1.0])
    assert r["n"] == 1 and r["p"] is None and r["t"] is None
    # 长度不一致必须报错，避免静默错配种子
    try:
        paired_t_test([1.0, 2.0], [1.0])
        raise AssertionError("长度不一致应抛 ValueError")
    except ValueError:
        pass
    # 显著 vs 不显著：大效应小方差 vs 小效应大方差
    sig = paired_t_test([10.5 + i * 0.01 for i in range(5)],
                        [10.0 + i * 0.01 for i in range(5)])
    assert sig["p"] < 0.01
    noise = paired_t_test([1.0, -1.0, 1.0, -1.0, 1.0],
                          [0.9, -0.9, 0.9, -0.9, 0.9])
    assert noise["p"] > 0.05
    print("  [ok] paired_t_test 边界（全同/零方差/n=1/长度错配/显著与不显著）")


def test_paired_t_effect_size():
    # d_z = mean_diff / std_diff（配对 Cohen's d）；diffs=[1,2,3,4]
    r = paired_t_test([1, 2, 3, 4], [0, 0, 0, 0])
    d_ref = 2.5 / math.sqrt(5.0 / 3.0)
    assert abs(r["d_z"] - d_ref) < 1e-9
    # 零方差/n<2 时效应量不可估计
    assert paired_t_test([6.0, 6.0], [5.0, 5.0])["d_z"] is None
    assert paired_t_test([2.0], [1.0])["d_z"] is None
    print("  [ok] paired_t_test 配对效应量 d_z")


def test_paired_t_nonfinite_and_pairing():
    """非有限观测与 None 必须按位置成对处理，避免错配种子或污染统计。"""
    # NaN/Inf 观测的配对整体剔除（同 mean_std 的 NaN 口径）
    r = paired_t_test([1.0, 2.0, float("nan"), 4.0], [0.0, 0.0, 0.0, 0.0])
    assert r["n"] == 3 and abs(r["mean_diff"] - 7.0 / 3.0) < 1e-12
    r = paired_t_test([float("inf"), 2.0], [0.0, 1.0])
    assert r["n"] == 1 and r["p"] is None
    # 全为非有限 -> 无样本可检验
    r = paired_t_test([float("nan")], [float("inf")])
    assert r["n"] == 0 and r["p"] is None and r["mean_diff"] is None
    # None 成对出现：整对剔除；位置错开说明种子集合没对齐，必须报错
    r = paired_t_test([1.0, None, 3.0], [0.0, None, 2.0])
    assert r["n"] == 2 and abs(r["mean_diff"] - 1.0) < 1e-12
    r = paired_t_test([None, None], [None, None])
    assert r["n"] == 0 and r["p"] is None
    try:
        paired_t_test([1.0, None], [None, 2.0])
        raise AssertionError("None 位置不一致应抛 ValueError")
    except ValueError:
        pass
    # NaN 的 t 不能伪装成 p=0（最显著）
    assert math.isnan(student_t_two_sided_p(float("nan"), 3))
    assert student_t_two_sided_p(float("inf"), 3) == 0.0
    print("  [ok] paired_t_test 非有限剔除/None 成对校验；NaN 不伪装显著")


def test_holm_bonferroni():
    # 手算：m=4, p=[0.01,0.02,0.03,0.04]，升序各乘以 4,3,2,1 后取前缀最大值
    adj = holm_bonferroni([0.01, 0.02, 0.03, 0.04])
    assert adj == [0.04, 0.06, 0.06, 0.06], adj
    # None 不占家族大小：m=2, [0.01, None, 0.04] -> [0.02, None, 0.04]
    adj = holm_bonferroni([0.01, None, 0.04])
    assert adj[1] is None
    assert abs(adj[0] - 0.02) < 1e-12 and abs(adj[2] - 0.04) < 1e-12
    # 单调不减、不小于原始 p、封顶 1
    ps = [0.9, 0.001, 0.5, 0.04, None, 0.2]
    adj = holm_bonferroni(ps)
    assert all(a >= p and a <= 1.0 for a, p in zip(adj, ps) if p is not None)
    order = sorted((i for i in range(len(ps)) if ps[i] is not None),
                   key=lambda i: ps[i])
    vals = [adj[i] for i in order]
    assert all(b + 1e-12 >= a for a, b in zip(vals, vals[1:])), vals
    # 全 None / 空输入
    assert holm_bonferroni([]) == []
    assert holm_bonferroni([None, None]) == [None, None]
    print("  [ok] holm_bonferroni（手算示例/None 不占家族/单调与封顶）")
