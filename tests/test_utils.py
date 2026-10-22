"""通用工具测试：AUROC/PR-AUC/F1（不平衡指标口径）、Pearson 与统计检验。

AUROC 是机制分析的核心指标（门控分数区分 hit/miss），这里用逐对比较的
定义式独立复算，专门覆盖并列分数的平均秩路径。PR-AUC（average precision）
是类别不平衡时的主指标，用逐步积分的定义式独立复算；F1 必须同时给出
macro/micro/weighted 口径，用不平衡算例核对三者的差异。paired_t_test 是
“多种子 Mean ± Std + 显著性”协议的核心，用 Student-t 已知临界值核对 p
值精度；sign-flip 置换检验用独立枚举复算核对精确 p 值。
"""
import json
import math
import os
import shutil
import tempfile

import numpy as np

from lega.utils import (auroc, average_precision, bootstrap_mean_ci,
                        calibration_metrics, config_bool, config_finite_float,
                        config_probability, f1_scores, holm_bonferroni, mean_std,
                        min_detectable_effect, paired_sign_flip_test,
                        paired_sign_test, paired_t_test, pearson,
                        roc_pr_aucs, save_json,
                        student_t_two_sided_p, t_critical_95, wilson_score_ci)


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


def test_auroc_filters_nonfinite_like_average_precision():
    """非有限分数先剔除（与 AP 同口径）：NaN 不参与排序，也不虚高 AUROC。"""
    got = auroc(np.array([np.nan, 3.0, 2.0, 1.0]), np.array([1, 1, 0, 0]))
    assert abs(got - 1.0) < 1e-12, got
    got = auroc(np.array([np.inf, 3.0, 2.0, 0.0]), np.array([1, 1, 0, 0]))
    assert abs(got - 1.0) < 1e-12, got
    # 剔除后只剩单一类别 → 仍返回 NaN
    assert math.isnan(auroc(np.array([np.nan, 1.0]), np.array([1, 1])))
    print("  [ok] auroc：非有限分数剔除（与 average_precision 同口径）")


def test_score_label_length_mismatch_is_rejected():
    """分数与标签必须一一对应：长度不同不能静默截断或多出标签。

    有限值掩码只按 scores 的形状取 labels；若 labels 更长，多出的标签会
    被无声丢弃，AUROC/AP 就在错误的样本集上计算。三个入口统一显式报错。
    """
    for fn in (auroc, average_precision, roc_pr_aucs):
        for scores, labels in (([0.1, 0.9], [1]),
                               ([0.1], [1, 0]),
                               (np.array([[0.1, 0.9]]), np.array([1]))):
            try:
                fn(scores, labels)
            except ValueError:
                pass
            else:
                raise AssertionError(
                    f"{fn.__name__} 未拒绝长度不一致: {scores!r} / {labels!r}")
    # 同尺寸多维输入按元素对齐（与 pearson/calibration 的扁平口径一致）
    got = auroc(np.array([[0.1, 0.2], [0.8, 0.9]]),
                np.array([[0, 0], [1, 1]]))
    assert abs(got - 1.0) < 1e-12, got
    print("  [ok] auroc/average_precision/roc_pr_aucs 拒绝长度不一致")


def test_save_json_writes_strict_json_atomically():
    """NaN/Inf 写成 null（合法 JSON），临时文件不残留。"""
    tmp = tempfile.mkdtemp()
    try:
        p = os.path.join(tmp, "out.json")
        save_json({"loss": float("nan"),
                   "x": [float("inf"), {"y": float("-inf")}],
                   "np": {"nan": np.float32("nan"), "int": np.int64(3),
                          "bool": np.bool_(True),
                          "arr": np.array([1.0, np.inf], dtype=np.float32)},
                   "keep": 0.5}, p)
        txt = open(p, encoding="utf-8").read()
        assert "NaN" not in txt and "Infinity" not in txt
        strict = json.loads(
            txt, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))
        assert strict == {
            "loss": None, "x": [None, {"y": None}],
            "np": {"nan": None, "int": 3, "bool": True, "arr": [1.0, None]},
            "keep": 0.5}
        assert not [f for f in os.listdir(tmp) if ".tmp" in f]
        # 序列化中途失败也不能留下半截临时文件，否则读者可能误把
        # “上次失败写入的残留”当成有效报告。
        bad = os.path.join(tmp, "bad.json")
        try:
            save_json({"bad": object()}, bad)
            raise AssertionError("不可序列化对象应报错")
        except TypeError:
            pass
        assert not os.path.exists(bad)
        assert not [f for f in os.listdir(tmp) if ".tmp" in f]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] save_json：非有限浮点转 null、原子写、失败清理临时文件")


def _brute_ap(scores, labels):
    """定义式 PR-AUC：按降序唯一阈值逐步积分（并列同组），独立于实现。"""
    pairs = sorted(zip(scores, labels), key=lambda t: -t[0])
    n_pos = sum(1 for _, y in pairs if y == 1)
    if n_pos == 0 or n_pos == len(pairs):
        return float("nan")
    ap, prev_r = 0.0, 0.0
    for i in range(len(pairs)):
        if i + 1 < len(pairs) and pairs[i + 1][0] == pairs[i][0]:
            continue  # 只在并列组末尾结算，保证同分整体进阈值
        tp = sum(1 for _, y in pairs[:i + 1] if y == 1)
        prec = tp / (i + 1)
        rec = tp / n_pos
        ap += (rec - prev_r) * prec
        prev_r = rec
    return ap


def test_average_precision_matches_bruteforce():
    """PR-AUC 与逐步积分定义式一致（含大量并列分数）。"""
    rng = np.random.default_rng(3)
    checked = 0
    for trial in range(300):
        n = int(rng.integers(4, 40))
        if trial % 2:
            scores = rng.integers(0, 4, size=n).astype(np.float64)
        else:
            scores = rng.normal(size=n)
        # 低正类率以贴近机制分析的真实不平衡场景
        labels = (rng.random(n) < 0.3).astype(int)
        if labels.sum() in (0, n):
            continue
        got = average_precision(scores, labels)
        ref = _brute_ap(scores.tolist(), labels.tolist())
        assert abs(got - ref) < 1e-12, f"trial={trial}: ap={got} != brute={ref}"
        checked += 1
    assert checked > 200
    print(f"  [ok] average_precision 与定义式一致（{checked} 个用例，含并列）")


def test_average_precision_properties():
    """随机基线=正类率、完全可分=1、反序≤正类率、单一类别 NaN。"""
    # 全部并列：一次召回全正类，precision=正类率 → AP 必须恰为正类率
    scores = np.ones(10)
    labels = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    assert abs(average_precision(scores, labels) - 0.2) < 1e-12
    # 完全可分 → 1.0
    sep = np.array([5.0, 4.0, 3.0, 2.0, 1.0, 0.0])
    y_sep = np.array([1, 1, 1, 0, 0, 0])
    assert abs(average_precision(sep, y_sep) - 1.0) < 1e-12
    # 完全反序（全负在前）：AP 应低于正类率
    rev = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    assert average_precision(rev, y_sep) < 0.5
    # 单一类别/空输入 → NaN（调用方转 None）
    assert math.isnan(average_precision(np.array([1.0, 2.0]), np.array([1, 1])))
    assert math.isnan(average_precision(np.array([]), np.array([])))
    # 非有限分数剔除而非污染
    nan_scores = np.array([np.nan, 3.0, 2.0, 1.0])
    assert abs(average_precision(nan_scores, np.array([1, 1, 0, 0])) - 1.0) < 1e-12
    print("  [ok] average_precision（随机基线/可分/反序/NaN 剔除）")


def test_roc_pr_aucs_bitwise_equals_separate_metrics():
    """组合入口必须是两单指标函数的逐位等价替身。

    覆盖大量并列、非有限分数、单一类别、全并列等边界；任一元素不满足
    逐位相等（或同为 NaN）即失败。
    """
    rng = np.random.default_rng(11)
    cases = []
    for trial in range(200):
        n = int(rng.integers(2, 60))
        if trial % 3 == 0:
            scores = rng.integers(0, 4, size=n).astype(np.float64)
        elif trial % 3 == 1:
            scores = np.round(rng.normal(size=n), 1)
        else:
            scores = rng.normal(size=n)
        labels = (rng.random(n) < 0.35).astype(np.float64)
        cases.append((scores, labels))
    cases += [
        (np.array([1.0, 2.0]), np.array([1.0, 1.0])),          # 单一类别
        (np.array([1.0, 2.0]), np.array([0.0, 0.0])),
        (np.ones(6), np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])),
        (np.array([np.nan, 3.0, 2.0, 1.0, np.inf]),
         np.array([1.0, 1.0, 0.0, 0.0, 1.0])),
        (np.array([-0.0, 0.0, 1.0]), np.array([0.0, 1.0, 1.0])),
        (np.array([]), np.array([])),
        (np.array([1.0, 2.0, 3.0]), np.array([1.0, 1.0, 2.0])),  # 非二值退化为原函数
    ]
    checked = 0
    for scores, labels in cases:
        ref_a, ref_p = auroc(scores, labels), average_precision(scores, labels)
        got_a, got_p = roc_pr_aucs(scores, labels)
        ok_a = (math.isnan(ref_a) and math.isnan(got_a)) or ref_a == got_a
        ok_p = (math.isnan(ref_p) and math.isnan(got_p)) or ref_p == got_p
        assert ok_a and ok_p, (f"scores={scores} labels={labels}: "
                               f"ref=({ref_a}, {ref_p}) got=({got_a}, {got_p})")
        checked += 1
    assert checked == len(cases)
    print(f"  [ok] roc_pr_aucs 与 auroc/average_precision 逐位一致（{checked} 组）")


def test_f1_scores_three_averages_and_imbalance():
    """不平衡算例下 macro/micro/weighted 的差异必须如实给出且可复算。"""
    # 10 个样本、3 正 7 负，全预测为正：class1 P=0.3,R=1,f1=6/13；class0 全 0
    y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    y_pred = np.ones(10, dtype=int)
    out = f1_scores(y_true, y_pred)
    f1_pos = 2 * 0.3 * 1.0 / 1.3
    assert abs(out["per_class"][1]["f1"] - f1_pos) < 1e-12
    assert abs(out["per_class"][0]["f1"] - 0.0) < 1e-12
    assert abs(out["macro"] - (f1_pos + 0.0) / 2) < 1e-12
    # 多分类 micro 汇总全部类的 TP/FP/FN，等于逐样本准确率 3/10；
    # weighted 按支持度加权（多数类全错 → 被拉低）
    assert abs(out["micro"] - 0.3) < 1e-12
    assert abs(out["weighted"] - f1_pos * 3 / 10) < 1e-12
    # 三者显著分离：只报“F1”不足以复现结论
    assert out["weighted"] < out["macro"] < out["micro"]
    # 多分类：labels 限定只统计子集，未出现的类 support=0、f1=0；
    # 长度不一致必须报错而不是静默对齐
    yt = np.array([0, 1, 2, 2])
    yp = np.array([0, 2, 2, 1])
    out2 = f1_scores(yt, yp, labels=[0, 1, 2])
    assert out2["n"] == 4 and set(out2["per_class"]) == {0, 1, 2}
    assert out2["per_class"][0]["f1"] == 1.0
    assert out2["per_class"][1]["f1"] == 0.0
    out3 = f1_scores(yt, yp, labels=[0, 1])
    assert set(out3["per_class"]) == {0, 1}
    try:
        f1_scores([1, 0], [1])
        raise AssertionError("长度不一致应抛 ValueError")
    except ValueError:
        pass
    # 类别是整数身份：bool/小数/非数值不能被截断成另一个合法类别
    for bad in (True, 1.5, float("nan"), "bad"):
        for call in (lambda: f1_scores([bad, 0], [0, 1]),
                     lambda: f1_scores([0, 1], [bad, 0]),
                     lambda: f1_scores([0, 1], [0, 1], labels=[0, bad])):
            try:
                call()
            except ValueError:
                pass
            else:
                raise AssertionError(f"非法类别未被拒绝: {bad!r}")
    assert f1_scores([1.0, 0.0], [1.0, 1.0])["n"] == 2
    # 重复 labels 会让 per_class 覆盖、macro/weighted 重复计权
    for dup in ([0, 0], [1, 0, 1]):
        try:
            f1_scores([0, 1], [0, 1], labels=dup)
            raise AssertionError(f"重复 labels 应抛 ValueError: {dup!r}")
        except ValueError:
            pass
    print("  [ok] f1_scores（macro/micro/weighted 分离、多分类、长度校验/"
          "非法类别与重复 labels 拒绝）")


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
    # bool 不是连续观测：True/False 不能被当作 1.0/0.0 算相关系数
    assert pearson([True, False, True, False, True], x[:5].tolist()) is None
    assert pearson(np.array([True, False, True, False]), x[:4]) is None
    try:
        pearson([1.0, 2.0], [1.0])
    except ValueError:
        pass
    else:
        raise AssertionError("pearson 长度不一致未被拒绝")
    print("  [ok] pearson 边界（常量/样本不足/NaN 成对剔除/bool 与长度校验）")


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


def test_bool_not_counted_as_numeric_observation():
    """bool 不是统计观测：True/False 不能被当成 1.0/0.0 参与均值或配对差。"""
    assert mean_std([True, np.bool_(False), 2.0]) == (2.0, None)
    assert mean_std([True, False]) == (None, None)
    assert bootstrap_mean_ci([True, False, 5.0]) is None
    r = paired_t_test([True, 2.0, 3.0], [1.0, 1.0, 1.0])
    assert r["n"] == 2 and abs(r["mean_diff"] - 1.5) < 1e-12
    print("  [ok] bool 不计入均值/bootstrap/配对检验")


def test_config_bool_validation():
    """配置开关不能把任意非空字符串/非零数值静默当成 True。"""
    for value, expected in ((True, True), (False, False),
                            (np.bool_(True), True), (np.bool_(False), False),
                            (1, True), (0, False), (1.0, True), (0.0, False),
                            (np.int64(1), True), ("true", True), ("FALSE", False),
                            (" 0 ", False)):
        assert config_bool(value, "x") is expected, (value, expected)
    for bad in ("yes", "no", "on", "", 2, -1, 0.5, float("nan"),
                float("inf"), None, []):
        try:
            config_bool(bad, "x")
            raise AssertionError(f"config_bool({bad!r}) 应被拒绝")
        except ValueError as exc:
            assert "x" in str(exc)
    print("  [ok] config_bool 只接受明确布尔/0/1，拒绝任意真值转换")


def test_config_float_and_probability_validation():
    """模型配置浮点不能把 bool/NaN/Inf 静默转成数值，概率还须落在 [0, 1]。"""
    for bad in (True, False, np.bool_(True), float("nan"), float("inf"),
                float("-inf"), None, "abc", []):
        for fn in (config_finite_float, config_probability):
            try:
                fn(bad, "x")
                raise AssertionError(f"{fn.__name__}({bad!r}) 应被拒绝")
            except ValueError as exc:
                assert "x" in str(exc)
    for bad in (-0.01, 1.01):
        try:
            config_probability(bad, "p")
            raise AssertionError(f"config_probability({bad!r}) 应被拒绝")
        except ValueError as exc:
            assert "p" in str(exc)
    assert config_finite_float("2.5", "x") == 2.5
    assert config_finite_float(2, "x") == 2.0
    assert config_probability(0, "p") == 0.0
    assert config_probability(1, "p") == 1.0
    assert config_probability("0.25", "p") == 0.25
    print("  [ok] config_finite_float/config_probability 拒绝 bool/NaN/Inf/越界")


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


def test_t_critical_95_matches_known_table():
    """小样本 PPL 区间必须用 t 临界值，而不是固定 1.96。"""
    assert abs(t_critical_95(1) - 12.7062) < 1e-3
    assert abs(t_critical_95(2) - 4.30265) < 1e-3
    assert abs(t_critical_95(9) - 2.26216) < 1e-3
    assert abs(t_critical_95(19) - 2.09302) < 1e-3
    assert abs(t_critical_95(30) - 2.04227) < 1e-3
    assert abs(t_critical_95(300) - 1.95996) < 1e-4
    # 非法自由度安全返回 None，不抛异常
    assert t_critical_95(0) is None and t_critical_95(None) is None
    assert t_critical_95(float("inf")) is None
    assert t_critical_95(True) is None and t_critical_95(np.bool_(True)) is None
    print("  [ok] t_critical_95 与 Student-t 临界值表一致")


def test_paired_t_edge_cases():
    # 完全相同的两组 -> t=0, p=1（不显著）
    r = paired_t_test([5.0, 5.0, 5.0], [5.0, 5.0, 5.0])
    assert r["t"] == 0.0 and r["p"] == 1.0
    # 常数差且方差为 0：均值非 0 记 p=0，均值 0 记 p=1
    r = paired_t_test([6.0, 6.0, 6.0], [5.0, 5.0, 5.0])
    assert r["p"] == 0.0 and math.isinf(r["t"]) and r["t"] > 0
    # 零方差且均值为负：t 必须带负号（方向不能被说反）
    r = paired_t_test([5.0, 5.0, 5.0], [6.0, 6.0, 6.0])
    assert r["p"] == 0.0 and math.isinf(r["t"]) and r["t"] < 0
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
    # NaN 与 None 同义：不占家族大小，也不得污染其他 p 的校正值
    adj = holm_bonferroni([float("nan"), 0.01, 0.02, 0.5])
    assert adj[0] is None, adj
    assert abs(adj[1] - 0.03) < 1e-12 and abs(adj[2] - 0.04) < 1e-12
    assert abs(adj[3] - 0.5) < 1e-12
    assert holm_bonferroni([float("nan"), float("nan")]) == [None, None]
    # bool/字符串/容器/越界 p 均按缺失处理，不占家族大小也不崩溃
    adj = holm_bonferroni([True, "0.02", -0.1, 1.5, [0.1], 0.04])
    assert adj[0] is None and adj[2] is None, adj
    assert adj[3] is None and adj[4] is None, adj
    assert abs(adj[1] - 0.04) < 1e-12 and abs(adj[5] - 0.04) < 1e-12, adj
    assert holm_bonferroni([False, np.bool_(True)]) == [None, None]
    print("  [ok] holm_bonferroni（手算/None 与 NaN 不占家族/非法 p 按缺失）")


def _brute_signflip_p(diffs):
    """定义式精确 p：枚举 2^n 个符号翻转，数 |mean| >= |观测均值| 的比例。"""
    n = len(diffs)
    obs = abs(sum(diffs) / n)
    cnt = 0
    for mask in range(2 ** n):
        flipped = [(d if (mask >> i) & 1 == 0 else -d)
                   for i, d in enumerate(diffs)]
        if abs(sum(flipped) / n) >= obs - 1e-15:
            cnt += 1
    return cnt / float(2 ** n)


def test_paired_sign_flip_exact_p_values():
    """n<=16 全枚举：p_perm 必须与定义式独立复算逐位一致。"""
    cases = [
        [5.0, 4.0, 6.0, 7.0, 5.0],            # 同号大效应
        [1.0, -1.0, 1.0, -1.0, 1.0],          # 均值近 0 的噪声
        [0.3, -0.1, 0.2, 0.25, -0.05, 0.4],   # 小效应
        [2.0, 1.0, 3.0],                      # n=3
    ]
    for diffs in cases:
        r = paired_sign_flip_test(diffs, [0.0] * len(diffs))
        assert r["exact"] is True
        ref = _brute_signflip_p(diffs)
        assert abs(r["p_perm"] - ref) < 1e-12, (diffs, r["p_perm"], ref)
    # 同号大效应：只有全正/全负两个极端达到观测均值 → p = 2/2^n
    r = paired_sign_flip_test([5.0, 4.0, 6.0, 7.0, 5.0], [0.0] * 5)
    assert abs(r["p_perm"] - 2.0 / 32.0) < 1e-12
    # 全零差：p=1（无可检效应）
    r = paired_sign_flip_test([1.0, 1.0], [1.0, 1.0])
    assert r["p_perm"] == 1.0
    print("  [ok] paired_sign_flip_test 精确 p 与独立枚举一致（n=3~6）")


def test_paired_sign_test_exact_and_zeros():
    """符号检验：只数方向、零差剔除，精确双侧 p 与二项定义一致。"""
    r = paired_sign_test([1.0, 2.0, 3.0, 4.0, 5.0], [0.0] * 5)
    assert r["median_diff"] == 3.0
    assert (r["n_pos"], r["n_neg"], r["n_zero"]) == (5, 0, 0)
    assert abs(r["p_sign"] - 2.0 / 32.0) < 1e-12
    # 零差不提供方向信息：3 正 + 1 零 → nz=3，p=2/8=0.25
    r = paired_sign_test([1.0, 2.0, 3.0, 0.0], [0.0] * 4)
    assert r["n_zero"] == 1 and r["p_sign"] == 0.25
    # 方向各半 → p=1（双侧精确值上限截断）
    r = paired_sign_test([1.0, -1.0, 2.0, -2.0, 0.0], [0.0] * 5)
    assert r["n_pos"] == 2 and r["n_neg"] == 2 and r["p_sign"] == 1.0
    assert r["median_diff"] == 0.0
    # 全零差 → p=1；n=0 字段为 None 而不是崩溃
    r = paired_sign_test([1.0, 1.0], [1.0, 1.0])
    assert r["p_sign"] == 1.0 and r["n_zero"] == 2
    r = paired_sign_test([], [])
    assert r["n"] == 0 and r["p_sign"] is None and r["median_diff"] is None
    # 与 t 检验共用配对口径：None 必须成对、bool/非有限值跳过
    assert paired_sign_test([1.0, None, 3.0], [0.0, None, 2.0])["n"] == 2
    try:
        paired_sign_test([1.0, None], [None, 2.0])
        raise AssertionError("None 位置不一致应抛 ValueError")
    except ValueError:
        pass
    r = paired_sign_test([1.0, float("nan"), True], [0.0, 1.0, 1.0])
    assert r["n"] == 1 and r["p_sign"] == 1.0
    print("  [ok] paired_sign_test（精确符号 p、零差剔除、边界）")


def test_paired_sign_flip_mc_and_guards():
    """n>16 走固定种子 Monte-Carlo；边界与 paired_t_test 同口径。"""
    rng = np.random.default_rng(7)
    a = list(rng.normal(10.0, 0.05, 20))
    b = list(rng.normal(10.0, 0.05, 20))
    r = paired_sign_flip_test(a, b)
    assert r["exact"] is False and r["n_resamples"] >= 1000
    # 确定性：同一输入两次调用结果一致
    assert r["p_perm"] == paired_sign_flip_test(a, b)["p_perm"]
    # 无效应数据 p 应偏大、强效应 p 应很小（Monte-Carlo 口径下方向正确）
    assert paired_sign_flip_test(a, b)["p_perm"] > 0.01
    strong_a = [10.0 + i * 0.2 for i in range(20)]
    strong_b = [10.0 + i * 0.01 for i in range(20)]
    assert paired_sign_flip_test(strong_a, strong_b)["p_perm"] < 0.001
    # 边界：n<2 不检验；None/长度错配与 t 检验同口径
    r = paired_sign_flip_test([2.0], [1.0])
    assert r["p_perm"] is None and r["n"] == 1
    r = paired_sign_flip_test([1.0, None, 3.0], [0.0, None, 2.0])
    assert r["n"] == 2
    try:
        paired_sign_flip_test([1.0, None], [None, 2.0])
        raise AssertionError("None 位置不一致应抛 ValueError")
    except ValueError:
        pass
    r = paired_sign_flip_test([1.0, float("nan")], [0.0, 1.0])
    assert r["n"] == 1 and r["p_perm"] is None
    # 重采样次数/种子是计数身份：bool/小数/非正数不能截断后继续
    for bad in (True, 0, -1, 1.5, "bad"):
        try:
            paired_sign_flip_test([1.0, 2.0], [0.0, 0.0], n_resamples=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"非法 n_resamples 未被拒绝: {bad!r}")
    for bad in (True, -1, 1.5, "bad"):
        try:
            paired_sign_flip_test([1.0, 2.0], [0.0, 0.0], seed=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"非法 seed 未被拒绝: {bad!r}")
    assert paired_sign_flip_test([1.0, 2.0], [0.0, 0.0],
                                 n_resamples="1000", seed="3")["p_perm"] is not None
    print("  [ok] paired_sign_flip_test Monte-Carlo/确定性/边界口径/"
          "非法重采样与种子拒绝")


def test_bootstrap_mean_ci():
    # 常规数据：区间覆盖样本均值、宽度合理、确定性复现
    vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    ci1 = bootstrap_mean_ci(vals, n_resamples=2000, seed=3)
    ci2 = bootstrap_mean_ci(vals, n_resamples=2000, seed=3)
    assert ci1 == ci2
    lo, hi = ci1
    assert lo < 4.5 < hi and hi - lo < 4.0
    # NaN/None 剔除后仍可用
    ci = bootstrap_mean_ci([1.0, float("nan"), 3.0, None, 5.0])
    assert ci is not None and ci[0] < 3.0 < ci[1]
    # n<2 与全 NaN：无区间可言
    assert bootstrap_mean_ci([2.5]) is None
    assert bootstrap_mean_ci([float("nan"), None]) is None
    # 退化样本（全同值）：区间坍缩到点
    assert bootstrap_mean_ci([5.0, 5.0, 5.0]) == (5.0, 5.0)
    for bad in (True, 0, -1, 1.5, "bad"):
        try:
            bootstrap_mean_ci(vals, n_resamples=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"非法 n_resamples 未被拒绝: {bad!r}")
    for bad in (True, 0, 1, 1.5, -0.1, float("nan"), "bad"):
        try:
            bootstrap_mean_ci(vals, n_resamples=10, ci=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"非法 ci 未被拒绝: {bad!r}")
    for bad in (True, -1, 1.5, "bad"):
        try:
            bootstrap_mean_ci(vals, n_resamples=10, seed=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"非法 seed 未被拒绝: {bad!r}")
    assert bootstrap_mean_ci(vals, n_resamples="200", ci="0.9", seed="3")
    print("  [ok] bootstrap_mean_ci（覆盖均值/确定性/NaN 剔除/退化样本/"
          "非法参数拒绝）")


def test_min_detectable_effect():
    # MDE 随种子数增加而减小、随方差增大而增大
    m5 = min_detectable_effect(0.1, 5)
    m10 = min_detectable_effect(0.1, 10)
    assert m5 > m10 > 0
    m_big = min_detectable_effect(1.0, 5)
    assert m_big > m5
    # 正态近似口径：z_{0.975}+z_{0.8} = 1.959964 + 0.841621 ≈ 2.801586
    ref = 2.801586 * 0.1 / math.sqrt(5)
    assert abs(m5 - ref) < 1e-6
    # 不可估计的输入
    assert min_detectable_effect(None, 5) is None
    assert min_detectable_effect(0.0, 5) is None
    assert min_detectable_effect(0.1, 1) is None
    # bool/非有限 std 不是效应量观测；n/α/power 不能截断或越界
    assert min_detectable_effect(True, 5) is None
    assert min_detectable_effect(float("nan"), 5) is None
    for bad in (True, 5.5, "bad"):
        try:
            min_detectable_effect(0.1, bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"非法 n 未被拒绝: {bad!r}")
    for name in ("alpha", "power"):
        for bad in (True, 0, 1, -0.1, 1.5, float("nan"), "bad"):
            try:
                min_detectable_effect(0.1, 5, **{name: bad})
            except ValueError:
                pass
            else:
                raise AssertionError(f"非法 {name} 未被拒绝: {bad!r}")
    assert min_detectable_effect("0.1", "5", alpha="0.05", power="0.8") > 0
    print("  [ok] min_detectable_effect（随 n 递减/口径常数/边界/"
          "非法参数拒绝）")


def test_wilson_score_ci():
    # 手算核对：k=8, n=10, z=1.959964 → (0.4906, 0.9437)
    lo, hi = wilson_score_ci(8, 10)
    assert abs(lo - 0.4906) < 1e-3 and abs(hi - 0.9437) < 1e-3
    # 点估计必须落在区间内
    assert lo <= 0.8 <= hi
    # 端点闭式且不越界：k=0 → 下界恰为 0；k=n → 上界恰为 1（无浮点残差）
    lo0, hi0 = wilson_score_ci(0, 10)
    assert lo0 == 0.0 and 0.0 < hi0 < 0.5
    lo1, hi1 = wilson_score_ci(10, 10)
    assert lo1 > 0.5 and hi1 == 1.0
    # n 增大区间收窄
    w5 = wilson_score_ci(4, 5)
    w50 = wilson_score_ci(40, 50)
    assert (w50[1] - w50[0]) < (w5[1] - w5[0])
    # 边界：n<=0 / None
    assert wilson_score_ci(3, 0) is None
    assert wilson_score_ci(None, 10) is None
    # k 越界被钳制而不是产生非法区间
    lo, hi = wilson_score_ci(12, 10)
    assert 0.0 <= lo <= hi <= 1.0
    # n/k 是计数身份，ci 是 (0,1) 内概率：bool/小数/非法置信水平拒绝
    for bad in (True, 1.5, "bad"):
        try:
            wilson_score_ci(bad, 10)
        except ValueError:
            pass
        else:
            raise AssertionError(f"非法 k 未被拒绝: {bad!r}")
        try:
            wilson_score_ci(3, bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"非法 n 未被拒绝: {bad!r}")
    for bad in (True, 0, 1, -0.1, 1.5, float("nan"), "bad"):
        try:
            wilson_score_ci(3, 10, ci=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"非法 ci 未被拒绝: {bad!r}")
    assert wilson_score_ci("8", "10", ci="0.95")
    print("  [ok] wilson_score_ci（手算值/端点钳制/区间收窄/非法参数拒绝）")


def test_calibration_metrics_brier_ece_and_baseline():
    """校准：完美概率 Brier=0、常数基率 Brier 为无信息基线、ECE 识别错校准。"""
    y = [0.0] * 80 + [1.0] * 20
    perfect = list(y)
    c = calibration_metrics(perfect, y, n_bins=10)
    assert c["brier"] == 0.0 and c["ece"] == 0.0 and c["mce"] == 0.0
    assert c["brier_skill"] == 1.0
    assert c["ece_adaptive"] == 0.0 and c["mce_adaptive"] == 0.0
    # Murphy 分解是恒等式（标准三项 + 箱内项），且各分量有限
    assert abs(c["brier_reliability"] - c["brier_resolution"]
               + c["brier_uncertainty"] + c["brier_within_score"]
               - 2.0 * c["brier_within_cov"] - c["brier"]) < 1e-12
    assert c["brier_reliability"] == 0.0
    assert abs(c["brier_uncertainty"] - 0.2 * 0.8) < 1e-12
    # 常数基率预测：Brier 恰为 p(1-p)，skill≈0
    const = calibration_metrics([0.2] * len(y), y, n_bins=10)
    assert abs(const["brier"] - 0.2 * 0.8) < 1e-12
    assert abs(const["brier_skill"]) < 1e-12
    # 排序正确但概率整体过高：Brier 比完美差，ECE>0
    over = [0.9 if v == 1.0 else 0.6 for v in y]
    c2 = calibration_metrics(over, y, n_bins=10)
    assert c2["brier"] > 0.0 and c2["ece"] > 0.2
    assert c2["ece_adaptive"] > 0.2 and c2["mce_adaptive"] >= c2["ece_adaptive"]
    # 分箱计数/边界：10 箱内计数总和等于样本数
    assert sum(b["n"] for b in c2["bins"]) == len(y)
    assert c2["n"] == len(y) and c2["n_pos"] == 20
    assert abs(c2["brier_reliability"] - c2["brier_resolution"]
               + c2["brier_uncertainty"] + c2["brier_within_score"]
               - 2.0 * c2["brier_within_cov"] - c2["brier"]) < 1e-12
    # 等频分箱：n_bins=4 时每组样本数相差不超过 1（200/4 恰为 50）
    c4 = calibration_metrics(over, y, n_bins=4)
    assert c4["ece_adaptive"] is not None and c4["mce_adaptive"] >= 0.0
    # 单类/空输入：Brier 可算，ECE 不可解释地返回 None
    one = calibration_metrics([0.7, 0.8], [1.0, 1.0])
    assert one["brier"] is not None and one["ece"] is None
    assert one["ece_adaptive"] is None and one["brier_reliability"] is None
    empty = calibration_metrics([], [])
    assert empty["n"] == 0 and empty["brier"] is None
    assert empty["ece_adaptive"] is None and empty["mce_adaptive"] is None
    # n_bins 是计数身份：bool/小数/非数值不能被 int() 截断或让 range() 崩溃
    for bad in (True, 10.5, "bad", None):
        try:
            calibration_metrics([0.5, 0.6], [0.0, 1.0], n_bins=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"非法 n_bins 未被拒绝: {bad!r}")
    # bool 分数/标签不是概率观测：True 不能冒充 1.0 的正类或完美分数
    mixed = calibration_metrics([True, 0.25, 0.75], [True, 0.0, 1.0])
    assert mixed["n"] == 2 and mixed["n_pos"] == 1
    assert abs(mixed["base_rate"] - 0.5) < 1e-12
    no_lab = calibration_metrics([0.5, 0.6], [True, False])
    assert no_lab["n"] == 0 and no_lab["brier"] is None
    # 长度不一致必须显式拒绝，不能按短边静默截断配对
    try:
        calibration_metrics([0.1, 0.2], [0.0])
    except ValueError:
        pass
    else:
        raise AssertionError("scores/labels 长度不一致未被拒绝")
    nstr = calibration_metrics([0.2] * 10, [0.0] * 8 + [1.0] * 2,
                               n_bins="10")
    assert nstr["n_bins"] == 10 and len(nstr["bins"]) == 10
    print("  [ok] calibration_metrics（Brier/ECE/常数基线/边界/"
          "非法 n_bins 与 bool 观测）")


def test_calibration_metrics_brier_decomposition_within_bin_terms():
    """连续分数在等频箱内不恒定：标准三项不足，箱内项必须补上恒等式。"""
    c = calibration_metrics([0.1, 0.2, 0.8, 0.9], [0, 1, 0, 1], n_bins=2)
    # 两组箱内分数均不恒定；三项分解会低估/高估原始 Brier，必须带箱内项
    assert abs(c["brier_within_score"] - 0.0025) < 1e-15
    assert abs(c["brier_within_cov"] - 0.025) < 1e-15
    assert abs(c["brier_binning"] - (-0.0475)) < 1e-15
    assert abs(c["brier_reliability"] - c["brier_resolution"]
               + c["brier_uncertainty"] - c["brier"]) > 1e-6
    assert abs(c["brier_reliability"] - c["brier_resolution"]
               + c["brier_uncertainty"] + c["brier_within_score"]
               - 2.0 * c["brier_within_cov"] - c["brier"]) < 1e-15
    print("  [ok] Brier 分解补箱内方差/协方差项，连续分数下恒等")
