"""线性探针简单基线测试：分组 CV 防泄漏、λ 调优、不平衡指标口径与门控对照。

科研方法口径：机制主张需要对照“有标签监督的最简线性模型”，否则高
AUROC/PR-AUC 可能只说明廉价特征本身线性可分。本测试覆盖：
  * 分组 CV 不能让同一样例的记录跨折（否则 OOF 指标因“背题”虚高）；
  * 标准化与拟合只用训练折，λ 按 OOF log-loss 从网格选择且可复现；
  * F1 三口径（macro/micro/weighted）与不平衡主指标口径必须落盘；
  * 门控 vs 探针的配对检验方向语义：Δ=探针−门控，>0 表示简单基线更好。
"""
import math
import random
import shutil
import tempfile

from lega.probe import (DECISION_THRESHOLD, FEATURE_SETS, _group_folds,
                        fit_logreg, linear_probe, predict_logreg)
from run_all import (linear_probe_baseline_comparison, _render_baseline_sections)


def _synth_records(n_examples=18, per_example=4, signal=True, qterm=True,
                   seed=0):
    """合成机制记录：example 级难度 + 行级可学习信号（默认强相关）。"""
    rng = random.Random(seed)
    records = []
    for ex in range(n_examples):
        p_hit = 0.25 if (ex % 3) else 0.65
        for j in range(per_example):
            hit = 1.0 if rng.random() < p_hit else 0.0
            base = (0.35 + 0.45 * hit) if signal else 0.5
            rec = {
                "example": ex, "layer": 0, "head": ex % 2, "query_pos": j,
                "task": "niah_synth", "ctx": 256, "hit": hit,
                "max_p": min(0.99, base + rng.random() * 0.1),
                "log_n": 5.0 + rng.random(),
                "g": 0.7 * hit + 0.2, "logit": 2.0 * hit - 1.0,
            }
            if qterm:
                rec["qterm"] = (hit - 0.5) + rng.gauss(0, 0.4)
            records.append(rec)
    return records


def test_fit_logreg_separable_and_regularization():
    """可分数数据 → 概率接近 0/1；大 λ 收缩权重；单类训练折退化为先验。"""
    X = [[-2.0], [-1.0], [1.0], [2.0]]
    y = [0.0, 0.0, 1.0, 1.0]
    m = fit_logreg(X, y, l2=0.0)
    p = predict_logreg(m, X)
    assert p[0] < 0.05 and p[-1] > 0.95
    m_reg = fit_logreg(X, y, l2=10.0)
    assert abs(m_reg["coef"][0]) < abs(m["coef"][0])
    m_one = fit_logreg(X, [1.0, 1.0, 1.0, 1.0], l2=0.0)
    p_one = predict_logreg(m_one, [[-2.0], [2.0]])
    assert p_one[0] == 1.0 and p_one[1] == 1.0
    print("  [ok] fit_logreg（可分/正则收缩/单类退化）")


def test_linear_probe_grouped_cv_prevents_leakage():
    """分组 CV 的硬约束：同一样例的记录必须整体落在同一折。

    若按行随机切分，同一样例的其余记录会进入训练折，把该样例的难度
    “背”给留出折；这里直接检查折分配不变量（每个样例的折号唯一），
    并用“样例级特征不可泛化”的数据确认 OOF 成绩回落到随机水平（~0.5）。
    对照组：特征在行级与 hit 强相关时 AUROC 应显著 >0.5。
    """
    groups = [ex for ex in range(20) for _ in range(4)]
    folds = _group_folds(groups, 5, seed=3)
    for ex in range(20):
        idx = [i for i, g in enumerate(groups) if g == ex]
        assert len(set(folds[i] for i in idx)) == 1, f"样例 {ex} 跨折"
    leaky = []
    for ex in range(20):
        hit = 1.0 if ex % 2 else 0.0
        for j in range(4):
            leaky.append({"example": ex, "hit": hit,
                          "max_p": 0.3 + 0.02 * ex, "log_n": 4.0 + 0.01 * ex})
    rep = linear_probe(leaky, folds=5, seed=3)
    oof = rep["feature_sets"][rep["primary"]]["oof"]
    assert oof["auroc"] is None or abs(oof["auroc"] - 0.5) < 0.2, oof
    rep2 = linear_probe(_synth_records(seed=1), folds=5, seed=3)
    oof2 = rep2["feature_sets"][rep2["primary"]]["oof"]
    assert oof2["auroc"] > 0.8, oof2
    print(f"  [ok] 分组 CV（样例不跨折；无信号 AUROC={oof['auroc']}，"
          f"有信号 AUROC={oof2['auroc']:.3f}）")


def test_linear_probe_feature_sets_l2_selection_and_determinism():
    """qterm 齐备时主特征集含查询项；λ 从网格按 OOF log-loss 选择且可复现。"""
    recs = _synth_records(qterm=True)
    r1 = linear_probe(recs, folds=4, l2_grid=[0.0, 0.1, 1.0], seed=5)
    assert r1["primary"] == "conc_len_query"
    assert list(r1["feature_sets"]) == ["conc_len", "conc_len_query"]
    fs = r1["feature_sets"]["conc_len_query"]
    assert fs["features"] == ["log_maxp", "log_n", "qterm"]
    assert fs["selected_l2"] in [0.0, 0.1, 1.0]
    assert len(fs["per_l2"]) == 3
    # 主特征集：λ 选择必须对应 per_l2 中最小 log-loss
    loss_min = min(x["logloss"] for x in fs["per_l2"])
    assert math.isclose(
        min(x["logloss"] for x in fs["per_l2"]), loss_min)
    sel = next(x for x in fs["per_l2"] if x["l2"] == fs["selected_l2"])
    assert sel["logloss"] <= loss_min + 1e-12
    r2 = linear_probe(recs, folds=4, l2_grid=[0.0, 0.1, 1.0], seed=5)
    assert r1 == r2, "同一输入必须逐位可复现"
    # 无 qterm（如 standard 的记录）→ 主特征集退到廉价统计量
    recs_noq = [{k: v for k, v in r.items() if k != "qterm"}
                for r in recs]
    r3 = linear_probe(recs_noq, folds=4, seed=5)
    assert r3["primary"] == "conc_len"
    assert list(r3["feature_sets"]) == ["conc_len"]
    print("  [ok] 特征集选择/λ 网格选择/确定性")


def test_linear_probe_imbalance_metrics_and_f1_flavors():
    """F1 三口径与不平衡主指标口径必须随报告落盘；max_p=0 不产生 -inf。"""
    recs = _synth_records(n_examples=12, per_example=4, seed=2)
    # 制造极度不平衡：把绝大多数标签改为 miss，保留少量 hit
    for i, r in enumerate(recs):
        r["hit"] = 1.0 if i % 200 == 0 else 0.0
    recs.append({"example": 99, "hit": 1.0, "max_p": 0.0, "log_n": 5.0,
                 "qterm": 0.1})
    rep = linear_probe(recs, folds=3, seed=0)
    assert rep["n_valid"] == len(recs)
    assert rep["base_rate"] is not None and rep["base_rate"] < 0.05
    assert rep["primary_detection_metric"] == "PR-AUC"
    assert rep["imbalance_level"] in ("severe(<5%)", "extreme(≤1:1000)")
    fs = rep["feature_sets"][rep["primary"]]
    f1 = fs["oof"]["f1"]
    assert f1 is not None and f1["threshold"] == DECISION_THRESHOLD
    for key in ("macro", "micro", "weighted", "per_class"):
        assert key in f1
    # 极低正类率下 PR-AUC 必须有值且以正类率为随机基线参照
    assert fs["oof"]["auprc"] is not None
    print(f"  [ok] 不平衡口径（base_rate={rep['base_rate']:.4f}，"
          f"主指标={rep['primary_detection_metric']}，F1 三口径齐备）")


def test_linear_probe_edge_cases():
    """空输入/单类/样例不足：不抛错、显式给出不可估计原因。"""
    empty = linear_probe([])
    assert empty["n_valid"] == 0 and "未训练" in empty["note"]
    one_class = linear_probe([dict(r, hit=1.0)
                              for r in _synth_records(n_examples=4)])
    assert one_class["n_valid"] > 0
    assert "不可估计" in one_class["note"]
    few = _synth_records(n_examples=1, per_example=6, seed=1)
    few[0]["hit"], few[1]["hit"] = 0.0, 1.0
    rep = linear_probe(few, folds=5, seed=0)
    assert "不可估计" in rep["note"]
    print("  [ok] 边界情形（空/单类/样例不足）")


def _analysis(seed, probe_auprc, gate_auprc, probe_auroc, gate_auroc,
              base_rate=0.03):
    return {
        "variant": "lega", "seed": seed, "hit_base_rate": base_rate,
        "auprc_logit": gate_auprc, "auroc_logit": gate_auroc,
        "auprc_maxp": 0.05, "auroc_maxp": 0.6,
        "linear_probe": {
            "folds": 5, "primary": "conc_len",
            "primary_detection_metric": "PR-AUC",
            "imbalance_level": "severe(<5%)",
            "feature_sets": {"conc_len": {"features": ["log_maxp", "log_n"],
                                          "selected_l2": 0.1,
                                          "oof": {"auprc": probe_auprc,
                                                  "auroc": probe_auroc}}},
        },
    }


def test_linear_probe_comparison_direction_and_missing():
    """Δ=探针−门控：探针显著更好时必须点名“门控判别优越性不成立”。"""
    runs = {"lega": {s: {"analysis": _analysis(s, 0.60 + 0.005 * s,
                                               0.20 + 0.002 * s,
                                               0.90 + 0.002 * s,
                                               0.70 + 0.002 * s)}
                     for s in (1, 2, 3, 4, 5)}}
    out = linear_probe_baseline_comparison(runs)
    assert out["rows"] and out["rows"][0]["variant"] == "lega"
    row = out["rows"][0]
    assert row["delta_auprc"] > 0.3
    assert row["p_adj_auprc"] < 0.05
    assert "简单监督基线更好" in row["verdict"]
    # 反向：门控 logit 全面更好 → 不得出现“优越性不成立”
    runs2 = {"lega": {s: {"analysis": _analysis(s, 0.10, 0.50 + 0.01 * s,
                                                0.55, 0.95)}
                      for s in (1, 2, 3, 4, 5)}}
    out2 = linear_probe_baseline_comparison(runs2)
    assert out2["rows"][0]["delta_auprc"] < 0
    assert "门控显著优于" in out2["rows"][0]["verdict"]
    # 旧产物缺 linear_probe：必须显式列出，不能静默当作“无对照”
    runs3 = {"lega": {1: {"analysis": {"variant": "lega", "hit_base_rate": 0.1}}}}
    out3 = linear_probe_baseline_comparison(runs3)
    assert out3["rows"] == []
    assert out3["missing"] and out3["missing"][0]["variant"] == "lega"
    print("  [ok] linear_probe 对照方向/缺失显式化")


def test_render_baseline_sections_shows_probe_table_and_missing():
    """汇总页必须渲染探针表与缺失警告（与门控 logit 并排）。"""
    runs = {"lega": {s: {"analysis": _analysis(s, 0.55, 0.30 + 0.01 * s,
                                               0.88, 0.75)}
                     for s in (1, 2, 3, 4, 5)}}
    probe_cmp = linear_probe_baseline_comparison(runs)
    tmp = tempfile.mkdtemp()
    try:
        text = "\n".join(_render_baseline_sections(tmp, probe_cmp=probe_cmp))
        assert "简单学习基线" in text and "PR-AUC探针(OOF)" in text
        assert "ΔPR-AUC(探针−门控)" in text
        assert "命中基线率" in text
        # 缺证据时必须点名，而不是让读者以为已有对照
        missing = linear_probe_baseline_comparison(
            {"lega": {1: {"analysis": {"variant": "lega",
                                       "hit_base_rate": 0.1}}}})
        text2 = "\n".join(_render_baseline_sections(tmp, probe_cmp=missing))
        assert "线性探针简单基线缺失" in text2
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] 基线小节渲染探针表/缺失警告")


def test_probe_feature_sets_cover_gate_linear_terms():
    """full 特征集必须包含门控 logit 的线性项（log max_p、log n、qterm）。"""
    names = {k for k, _, fs in FEATURE_SETS
             for k in fs}
    for key, _, feats in FEATURE_SETS:
        if key == "conc_len_query":
            assert set(feats) == {"log_maxp", "log_n", "qterm"}
    assert "log_maxp" in names and "log_n" in names
    print("  [ok] 特征集与门控线性项覆盖一致")
