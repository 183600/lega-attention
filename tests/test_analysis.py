"""分析收集器测试：等长样例批前向与逐样例结果一致，且记录按原始顺序展平。

另覆盖机制分析的稳健性协议：τ 敏感性扫描（结论不能挂在单一阈值上）与
按样例聚类的 bootstrap AUROC 区间（同一样例的记录不是独立观测）。
"""
import math

from lega.analysis import (_bounded_float, _float_grid, _int_at_least,
                           auroc_cluster_bootstrap, collect_records,
                           resolve_analysis_ctx, summarize, tau_sensitivity)
from lega.data import CACHE_SCHEMA, CharTokenizer
from lega.model import GPT

_TEXT = ("The quick brown fox jumps over the lazy dog. "
         "A quiet river runs through the old town. 0123456789")


def _tok():
    return CharTokenizer(sorted(set(_TEXT)))


def _example(n, qs, vocab):
    ids = [1 + (i * 7 + 3) % (vocab - 1) for i in range(n)]
    return {"ids": ids, "query_span": (qs, n), "value_span": (n - 4, n - 3)}


def test_collect_records_batching_matches_per_example():
    import torch

    tok = _tok()
    torch.manual_seed(0)
    model = GPT(tok.n_vocab, block_size=64, d_model=32, n_layers=2, n_heads=2,
                variant="lega").eval()
    # 含重复长度与不同长度：覆盖分组堆叠与原始顺序恢复
    lengths = [(24, 20), (24, 19), (20, 16), (16, 12), (24, 18)]
    examples = [_example(n, qs, tok.n_vocab) for n, qs in lengths]
    task_of = ["niah_synth"] * len(examples)
    ctx_of = [64] * len(examples)
    r1 = collect_records(model, tok, "cpu", examples, task_of, ctx_of,
                         amp=False, batch_size=1)
    r8 = collect_records(model, tok, "cpu", examples, task_of, ctx_of,
                         amp=False, batch_size=8)
    assert len(r1) == len(r8) > 0
    for a, b in zip(r1, r8):
        assert a["example"] == b["example"]
        for key, va in a.items():
            vb = b[key]
            if isinstance(va, float):
                assert math.isclose(va, vb, rel_tol=1e-5, abs_tol=1e-6), (key, va, vb)
            else:
                assert va == vb, (key, va, vb)
    # 批处理后各样例记录仍按原始顺序连续排列
    assert [r["example"] for r in r1] == sorted(r["example"] for r in r1)
    print(f"  [ok] {len(r1)} 条记录逐位一致，样例顺序保持")


def test_collect_records_c_keeps_length_normalization_for_ablations():
    """集中度 c 在所有含 a·c 的变体中都必须是 log(n·max_p)，不能随 b·log n 消融。"""
    import torch

    tok = _tok()
    examples = [_example(24, 20, tok.n_vocab)]
    for variant in ("lega", "lega_no_len", "lega_conc_only"):
        torch.manual_seed(0)
        model = GPT(tok.n_vocab, block_size=64, d_model=32, n_layers=1,
                    n_heads=2, variant=variant).eval()
        recs = collect_records(model, tok, "cpu", examples, ["niah_synth"],
                               [64], amp=False, batch_size=1)
        assert recs, variant
        for r in recs:
            expected = r["log_n"] + math.log(max(r["max_p"], 1e-30))
            assert math.isclose(r["c"], expected, rel_tol=1e-9, abs_tol=1e-9), (
                variant, r["c"], expected)
    print("  [ok] 消融变体的 c 仍保留 log n 归一化")


def _mk_records():
    """合成记录：两个样例，hit 与分数强相关（AUROC 应接近 1）。

    每条记录只需机制统计消费的字段（example/hit/max_p/logit/qterm/c）。
    """
    records = []
    for ex in (0, 1, 2, 3, 4):
        n_pos, n_neg = 4, 4
        for i in range(n_pos):
            records.append({"example": ex, "hit": 1.0, "max_p": 0.6,
                            "logit": 2.0 + i * 0.1, "qterm": 0.5,
                            "c": 1.5, "g": 0.9, "log_n": 4.0,
                            "task": "niah_synth", "ctx": 256,
                            "layer": 0, "head": i % 2, "query_pos": i})
        for i in range(n_neg):
            records.append({"example": ex, "hit": 0.0, "max_p": 0.2,
                            "logit": -2.0 - i * 0.1, "qterm": -0.5,
                            "c": -1.5, "g": 0.2, "log_n": 4.0,
                            "task": "niah_synth", "ctx": 256,
                            "layer": 0, "head": i % 2, "query_pos": i})
    return records


def test_analysis_numeric_args_reject_invalid_values():
    """τ/网格/计数不能被 bool、NaN、负值或显式 0 静默改写。"""
    for bad in (True, "x", "1_0", float("nan"), float("inf")):
        try:
            _bounded_float(bad, "tau", minimum=0.0, maximum=1.0)
        except ValueError:
            pass
        else:
            raise AssertionError(f"非法 tau 未被拒绝: {bad!r}")
    for bad in (-0.1, 1.1):
        try:
            _bounded_float(bad, "tau", minimum=0.0, maximum=1.0)
        except ValueError:
            pass
        else:
            raise AssertionError(f"越界 tau 未被拒绝: {bad!r}")
    assert _bounded_float("0.5", "tau", minimum=0.0, maximum=1.0) == 0.5
    assert _float_grid("0.3, 0.5,", "tau_grid", 0.0, 1.0) == [0.3, 0.5]
    assert _float_grid([], "tau_grid", 0.0, 1.0) == []
    assert _float_grid(["0.3", "", None, "  ", 0.5], "tau_grid",
                       0.0, 1.0) == [0.3, 0.5]
    for bad in ("0.5,-1", "0.5,nan", True):
        try:
            _float_grid(bad, "probe_l2", minimum=0.0)
        except ValueError:
            pass
        else:
            raise AssertionError(f"非法网格未被拒绝: {bad!r}")
    # 显式 n_examples=0 不能回退成默认 30；bool/小数同样拒绝
    for bad in (0, -1, True, 1.5):
        try:
            _int_at_least(bad, "n_examples", 1)
        except ValueError:
            pass
        else:
            raise AssertionError(f"非法计数未被拒绝: {bad!r}")
    assert _int_at_least(2.0, "n_examples", 1) == 2
    assert _int_at_least(0, "bootstrap_resamples", 0) == 0
    print("  [ok] 分析数值参数：拒绝 bool/NaN/越界/显式 0 回退")


def test_analysis_ctx_resolution_rejects_empty_room():
    """ctx/max_new 非法或生成预算占满 block_size 时显式报错。

    旧行为会把 ctx 夹成 <=0，所有样例被 ContextTooSmall 跳过，最终落盘一份
    n_records=0 的“成功”空分析，配置错误被伪装成正常结果。
    """
    assert resolve_analysis_ctx(256, 512, 12) == 243
    assert resolve_analysis_ctx(256, 64, 12) == 64
    for args in ((256, 0, 12), (256, -5, 12), (256, 64, 0),
                 (256, 64, -1), (10, 64, 12), (256, 64, 300),
                 (True, 64, 12), (256, 1.5, 12)):
        try:
            resolve_analysis_ctx(*args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"非法分析 ctx 参数未被拒绝: {args!r}")
    print("  [ok] 分析 ctx 预留生成位置并拒绝空 room/非法参数")


def test_tau_sensitivity_sweep():
    """τ 扫描：各阈值下的条件统计方向正确，主 τ 的值与 summarize 一致。"""
    records = _mk_records()
    taus = [0.3, 0.5, 0.7]
    ts = tau_sensitivity(records, taus)
    assert ts["taus"] == taus
    # 集中率随 τ 单调不增
    cr = ts["conc_rate"]
    assert all(cr[i] >= cr[i + 1] for i in range(len(cr) - 1)), cr
    # 合成数据：max_p=0.9（hit）/0.2（miss）→ τ=0.3/0.5 时集中即 hit
    assert abs(ts["p_hit_given_conc"][0] - 1.0) < 1e-12   # τ=0.3
    assert abs(ts["p_hit_given_conc"][1] - 1.0) < 1e-12   # τ=0.5
    # τ=0.7 时无人"集中"→ 条件统计不可算（None）
    assert ts["p_hit_given_conc"][2] is None
    assert ts["misleading_rate"][2] == 0.0                # P(集中∧看错)=0
    # 与 summarize 主阈值口径逐字段一致
    s = summarize(records, "lega", 0.5)
    for key in ("conc_rate", "p_hit_given_conc", "p_hit_given_notconc",
                "misleading_rate", "misleading_given_conc"):
        assert s[key] == ts[key][1], key
    # hit=None（无 gold 跨度）的记录不进统计但不崩溃
    records.append({"example": 9, "hit": None, "max_p": 0.9, "logit": 1.0,
                    "qterm": 0.0, "c": 0.5, "g": 0.8, "log_n": 4.0})
    ts2 = tau_sensitivity(records, [0.5])
    assert ts2["conc_rate"][0] > 0 and ts2["p_hit_given_conc"][0] is not None
    print("  [ok] tau_sensitivity（单调性/与 summarize 口径一致/None 容错）")


def test_auroc_cluster_bootstrap_ci():
    """样例聚类 bootstrap：AUROC 区间覆盖点估计，logit−qterm 差区间为正。"""
    records = _mk_records()
    out = auroc_cluster_bootstrap(records, n_resamples=200, seed=0)
    assert out["n_examples"] == 5 and out["n_resamples"] == 200
    ci_logit = out["ci95"]["logit"]
    ci_qterm = out["ci95"]["qterm"]
    ci_diff = out["ci95"]["logit_minus_qterm"]
    assert ci_logit and ci_qterm and ci_diff
    # 可分合成数据：AUROC=1，区间应覆盖且偏大
    assert ci_logit[0] <= 1.0 <= ci_logit[1] and ci_logit[0] > 0.6
    assert ci_qterm[0] <= 1.0 <= ci_qterm[1]
    # logit 与 qterm 均完全可分 → 差恒 0，区间应包含 0
    assert ci_diff[0] <= 0.0 <= ci_diff[1]
    # 确定性：同种子重跑逐位一致
    out2 = auroc_cluster_bootstrap(records, n_resamples=200, seed=0)
    assert out2 == out
    # PR-AUC 族：合成数据对 logit 与 qterm 均完全可分 → AP 均为 1，差恒 0
    ap_logit = out["ci95"]["ap_logit"]
    ap_diff = out["ci95"]["ap_logit_minus_qterm"]
    assert ap_logit and abs(ap_logit[0] - 1.0) < 1e-9 and abs(ap_logit[1] - 1.0) < 1e-9
    assert (out["ci95"]["ap_qterm"] is not None)
    assert ap_diff and abs(ap_diff[0]) < 1e-9
    # qterm 无信号（hit/miss 分数全部并列）→ AUROC=0.5，logit 区间下界仍高于它
    weak = [dict(r) for r in records]
    for r in weak:
        r["qterm"] = 0.5
    out3 = auroc_cluster_bootstrap(weak, n_resamples=200, seed=0)
    assert abs(out3["ci95"]["qterm"][0] - 0.5) < 1e-9
    assert out3["ci95"]["logit"][0] > out3["ci95"]["qterm"][0]
    # qterm 全并列 → PR-AUC 随机基线（正类率 0.5），AP 差恒为 0.5
    assert abs(out3["ci95"]["ap_qterm"][0] - 0.5) < 1e-9
    assert abs(out3["ci95"]["ap_logit_minus_qterm"][0] - 0.5) < 1e-9
    # 样例过少（<2 个聚类）或无有效分数：安全返回空区间
    few = auroc_cluster_bootstrap(records[:8], n_resamples=50)
    assert few["ci95"] == {}
    nokey = auroc_cluster_bootstrap(
        [{"example": i, "hit": 1.0, "max_p": 0.9} for i in range(4)],
        n_resamples=50)
    assert nokey["ci95"] == {}
    # 重采样次数是计数身份：bool/小数/非数值不能被 int() 截断或崩溃
    for bad in (True, 1.5, "bad", -1):
        try:
            auroc_cluster_bootstrap(records, n_resamples=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"非法 n_resamples 未被拒绝: {bad!r}")
    assert auroc_cluster_bootstrap(records, n_resamples="50")["n_resamples"] == 50
    # 置信水平必须是 (0,1) 内的有限数值；ci=True 会把区间伪装成 100% 却
    # 仍以 ci95 为键，ci>=1 会让分位数越界。
    for bad in (True, 0, 1.0, 1.5, -0.1, float("nan"), float("inf"), "bad"):
        try:
            auroc_cluster_bootstrap(records, n_resamples=10, ci=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"非法 ci 未被拒绝: {bad!r}")
    ok = auroc_cluster_bootstrap(records, n_resamples=200, ci="0.9")
    assert ok["ci_level"] == 0.9 and ok["ci95"]
    print("  [ok] auroc_cluster_bootstrap（区间覆盖/确定性/聚类不足容错/AP 族/"
          "非法重采样与置信水平拒绝）")


def test_analysis_tolerates_malformed_external_records():
    """外部/截断记录缺 example/max_p/log_n/layer/head 时跳过而非崩溃。"""
    records = _mk_records() + [
        "junk",
        # 缺 log_n/layer/head：仍可进入汇总，但不能伪造逐头统计
        {"example": 90, "hit": 1.0, "max_p": 0.9, "logit": 2.0,
         "qterm": 0.5, "c": 1.5, "g": 0.8},
        # 类型错误的 hit/max_p/layer/head：按缺失处理
        {"example": 91, "hit": "bad", "max_p": "bad", "log_n": None,
         "layer": "bad", "head": []},
        {"example": 92, "hit": 0.0, "max_p": 0.1, "logit": 0.0,
         "qterm": 0.0, "c": 0.0, "g": 0.2, "log_n": 4.0,
         "layer": 0, "head": 0},
        # 缺 example：bootstrap 无法聚类，跳过
        {"hit": 1.0, "max_p": 0.9, "logit": 1.0, "qterm": 0.5,
         "c": 1.2, "g": 0.9, "log_n": 4.0, "layer": 0, "head": 0},
    ]
    from lega.analysis import _finite_float
    assert _finite_float("1_0") is None
    s = summarize(records, "unknown_variant", 0.5)
    assert s["n_records"] == len(records)
    assert s["variant_cn"] == "unknown_variant"
    assert {(p["layer"], p["head"]) for p in s["per_head"]} == {
        (0, 0), (0, 1)}
    ts = tau_sensitivity(records, [0.3, 0.5])
    assert len(ts["conc_rate"]) == 2
    out = auroc_cluster_bootstrap(records, n_resamples=50, seed=0)
    assert out["n_examples"] == 7, out
    # max_p 全缺失不能伪装成“从未集中”
    missing_mp = summarize([{"example": 0, "hit": 1.0, "logit": 1.0,
                             "qterm": 0.5, "c": 1.2, "g": 0.9,
                             "log_n": 4.0, "layer": 0, "head": 0}],
                           "lega", 0.5)
    assert missing_mp["conc_rate"] is None
    assert missing_mp["misleading_rate"] is None
    print("  [ok] 机制分析：畸形外部记录按缺失跳过，不触发 KeyError")


def test_summarize_rejects_non_binary_hit_and_missing_maxp():
    """hit 非 0/1 不能计入标签；逐头缺 max_p 时均值为 None 而非 NaN。"""
    recs = [
        {"example": 0, "hit": 1.5, "max_p": 0.9, "g": 0.8,
         "logit": 1.0, "qterm": 0.0, "c": 0.0, "log_n": 4.0,
         "layer": 0, "head": 0},
        {"example": 1, "hit": 0.0, "max_p": 0.1, "g": 0.2,
         "logit": -1.0, "qterm": 0.0, "c": 0.0, "log_n": 4.0,
         "layer": 0, "head": 0},
        {"example": 2, "hit": 1.0, "g": 0.7, "logit": 2.0,
         "qterm": 0.0, "c": 0.0, "log_n": 4.0, "layer": 0, "head": 1},
    ]
    s = summarize(recs, "lega", 0.5)
    # 1.5 既不是正类也不是负类，不能进入 n_hit/n_miss 或基率
    assert s["n_hit"] == 1 and s["n_miss"] == 1
    assert abs(s["hit_base_rate"] - 0.5) < 1e-12
    assert s["hit_rate"] == 0.5
    # head 1 只有一条记录且缺 max_p：均值未知，不能写成 NaN/null
    h1 = [p for p in s["per_head"] if p["head"] == 1][0]
    assert h1["mean_max_p"] is None
    # τ 扫描与 summarize 必须同口径：1.5 不是正类，不能让
    # p_hit_given_conc / misleading_rate 越界。
    ts = tau_sensitivity(recs, [0.5])
    assert ts["p_hit_given_conc"][0] is None, ts   # 唯一的集中样本 hit 非法
    assert ts["misleading_rate"][0] == 0.0, ts
    assert ts["p_hit_given_notconc"][0] == 0.0, ts  # 未集中样本 hit=0.0
    # 超范围 layer（int64 转换会饱和合并不同头）按畸形记录跳过
    huge = [dict(r, layer=10 ** 19) for r in recs]
    assert summarize(huge, "lega", 0.5)["per_head"] == []
    print("  [ok] summarize：非二值 hit 被剔除、缺 max_p 不产生 NaN、超范围层头跳过")


def test_out_of_range_probabilities_are_missing_not_evidence():
    """max_p/g 越界不可能来自 softmax/sigmoid，不能伪造集中率或完美校准。"""
    recs = [
        {"example": 0, "hit": 1.0, "max_p": 2.0, "g": 2.0,
         "logit": 2.0, "qterm": 0.5, "c": 1.0, "log_n": 4.0,
         "layer": 0, "head": 0},
        {"example": 1, "hit": 0.0, "max_p": -1.0, "g": -1.0,
         "logit": -2.0, "qterm": -0.5, "c": -1.0, "log_n": 4.0,
         "layer": 0, "head": 0},
    ]
    s = summarize(recs, "lega", 0.5)
    assert s["hit_rate"] == 0.5
    assert s["conc_rate"] is None, s
    assert s["p_hit_given_conc"] is None and s["p_hit_given_notconc"] is None
    assert s["misleading_rate"] is None
    assert s["calibration_g"] is None and s["brier_g"] is None
    assert s["mean_g_hit"] is None and s["mean_g_miss"] is None
    ts = tau_sensitivity(recs, [0.3, 0.5])
    assert ts["conc_rate"] == [None, None], ts
    assert ts["p_hit_given_conc"] == [None, None], ts
    print("  [ok] 越界 max_p/g 按缺失处理，不伪造集中与校准证据")


def test_bootstrap_filters_missing_scores_per_resample():
    """部分分数缺失不应让整个 bootstrap 重采样作废。"""
    recs = []
    for ex in range(12):
        for k in range(20):
            recs.append({
                "example": ex, "hit": float(k < 10),
                "logit": None if k == 0 else (2.0 if k < 10 else -1.0),
                "qterm": 0.5 if k < 10 else -0.5, "c": 0.0})
    point = summarize(recs, "lega", 0.5)["auroc_logit"]
    assert point == 1.0
    out = auroc_cluster_bootstrap(recs, n_resamples=200, seed=0)
    ci = out["ci95"]["logit"]
    assert ci is not None and ci[0] <= point <= ci[1], ci
    assert out["n_valid_resamples"] > 0
    print("  [ok] 聚类 bootstrap：按记录过滤缺失分数，而非丢弃整个重采样")


def test_summarize_reports_imbalance_aware_metrics():
    """summarize 必须报告正类率、不平衡等级、主指标与三口径门控 F1。"""
    # 19 正 381 负 → 正类率 4.75%（<5%）触发 PR-AUC 主指标
    records = []
    for i in range(400):
        hit = 1.0 if i < 19 else 0.0
        records.append({
            "example": i // 8, "hit": hit, "max_p": 0.8 if hit else 0.1,
            "logit": 2.0 if hit else -1.0, "qterm": 0.5,
            "c": 1.0, "g": 0.9 if hit else 0.4,
            "log_n": 4.0, "task": "niah_synth", "ctx": 256,
            "layer": 0, "head": i % 2, "query_pos": i % 4})
    s = summarize(records, "lega", 0.5)
    assert s["n_hit"] == 19 and s["n_miss"] == 381
    assert abs(s["hit_base_rate"] - 19 / 400) < 1e-12
    assert s["primary_detection_metric"] == "PR-AUC"
    assert s["imbalance_level"].startswith("severe")
    assert s["auprc_logit"] is not None and s["auprc_logit"] > 0.9
    # 全并列的 qterm：PR-AUC 应等于正类率（随机基线）
    assert abs(s["auprc_qterm"] - s["hit_base_rate"]) < 1e-9
    # 门控 F1：三口径齐全、宏平均对稀有类更严格（< micro）
    gf = s["gate_f1"]
    assert gf and gf["threshold"] == 0.5
    for k in ("macro", "micro", "weighted"):
        assert 0.0 <= gf[k] <= 1.0
    assert gf["macro"] <= gf["micro"] + 1e-12
    assert set(gf["per_class"]) == {"0", "1"}
    # 固定规则“稻草人”对照：恒写/恒不写 F1 必须并报（不平衡下学习 F1
    # 只有高于多数类固定规则才说明门控真有决策价值）
    triv = s["gate_f1_trivial"]
    # 概率校准：Brier/ECE 与常数基率 Brier 必须落盘；排序好不等于校准好
    cal = s["calibration_g"]
    assert cal and cal["n"] == 400 and cal["n_pos"] == 19
    assert s["brier_g"] is not None and s["ece_g"] is not None
    assert s["brier_g_constant"] is not None
    assert len(cal["bins"]) == 10
    assert triv and set(triv) >= {"always_write", "never_write"}
    p_base = 19 / 400
    # 恒写规则的 micro F1 在二分类下等于正类率（= 全判正的准确率）
    assert abs(triv["always_write"]["micro"] - p_base) < 1e-9
    f1_pos = 2 * p_base / (1 + p_base)
    assert abs(triv["always_write"]["macro"] - f1_pos / 2) < 1e-9
    assert triv["never_write"]["macro"] is not None
    # 学习门控（此例完美）必须显著优于恒写基线
    assert gf["macro"] > triv["always_write"]["macro"]
    # 不平衡温和时主指标改为并报（避免阈值语义被误解为“禁止 AUROC”）
    mild = [dict(r) for r in records]
    for i, r in enumerate(mild):
        r["hit"] = 1.0 if i < 100 else 0.0
    s2 = summarize(mild, "lega", 0.5)
    assert s2["primary_detection_metric"] == "AUROC 与 PR-AUC 并报"
    # 无 hit 标签的记录不产生 F1（None），不崩溃
    s3 = summarize([{"example": 0, "hit": None, "max_p": 0.9, "logit": 1.0,
                     "qterm": 0.0, "c": 0.5, "g": 0.8, "log_n": 4.0,
                     "layer": 0, "head": 0, "query_pos": 0}],
                   "lega", 0.5)
    assert s3["gate_f1"] is None
    assert s3["gate_f1_trivial"] is None   # 无标签时不报稻草人对照
    # hit==1 切片 g 全缺：mean_g_hit 必须是 None；整数组守卫会把这里
    # 误判为“有非 NaN”并对空切片 nanmean，返回 nan 且发 RuntimeWarning
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        s4 = summarize([
            {"example": 0, "hit": 1.0, "max_p": 0.9, "logit": 1.0,
             "qterm": 0.0, "c": 0.5, "g": None, "log_n": 4.0, "layer": 0,
             "head": 0, "query_pos": 0},
            {"example": 1, "hit": 0.0, "max_p": 0.1, "logit": 0.0,
             "qterm": 0.0, "c": 0.5, "g": 0.4, "log_n": 4.0, "layer": 0,
             "head": 0, "query_pos": 1},
        ], "lega", 0.5)
    assert s4["mean_g_hit"] is None and s4["mean_g_miss"] is not None
    print("  [ok] summarize（正类率/主指标/PR-AUC/三口径门控 F1/全缺 g 切片）")


def test_analysis_payload_binds_checkpoint_hash(monkeypatch):
    """机制分析结果必须绑定被分析 checkpoint 的路径与内容指纹。"""
    import json
    import os
    import shutil
    import sys
    import tempfile

    import numpy as np
    import torch

    from lega.model import GPT, save_checkpoint
    from lega.repro import sha256_file

    d = tempfile.mkdtemp()
    try:
        data = os.path.join(d, "data")
        os.makedirs(data)
        text = "The quick brown fox jumps over the lazy dog. 0123456789"
        itos = sorted(set(text))
        with open(os.path.join(data, "meta.json"), "w", encoding="utf-8") as f:
            json.dump({"tokenizer": "char", "itos": itos, "char_unk": False,
                       "vocab_size": len(itos), "cache_schema": CACHE_SCHEMA,
                       "fit_on": "train"}, f)
        arr = np.array([itos.index(c) for c in text], dtype=np.uint16)
        arr.tofile(os.path.join(data, "train.bin"))
        arr.tofile(os.path.join(data, "val.bin"))
        ckpt = os.path.join(d, "model.pt")
        torch.manual_seed(0)
        model = GPT(vocab_size=len(itos), block_size=128, d_model=8,
                    n_layers=1, n_heads=1, ffn_mult=2, variant="lega",
                    dropout=0.0, attn_dropout=0.0, gate_bias_init=2.0)
        save_checkpoint(ckpt, model, extra={"variant": "lega", "seed": 5})
        out_dir = os.path.join(d, "results")
        os.makedirs(out_dir)
        monkeypatch.setattr(sys, "argv", [
            "analysis", "--ckpt", ckpt, "--data_dir", data,
            "--out_dir", out_dir, "--ctx", "64", "--n_examples", "1",
            "--device", "cpu"])
        from lega.analysis import main as analysis_main
        analysis_main()
        with open(os.path.join(out_dir, "analysis_lega.json"),
                  encoding="utf-8") as f:
            payload = json.load(f)
        assert payload["ckpt"] == os.path.abspath(ckpt)
        assert payload["ckpt_sha256"] == sha256_file(ckpt)
    finally:
        shutil.rmtree(d, ignore_errors=True)
    print("  [ok] analysis payload 绑定 checkpoint 路径与内容哈希")
