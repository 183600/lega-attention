"""分析收集器测试：等长样例批前向与逐样例结果一致，且记录按原始顺序展平。

另覆盖机制分析的稳健性协议：τ 敏感性扫描（结论不能挂在单一阈值上）与
按样例聚类的 bootstrap AUROC 区间（同一样例的记录不是独立观测）。
"""
import math

from lega.analysis import (auroc_cluster_bootstrap, collect_records,
                           summarize, tau_sensitivity)
from lega.data import CharTokenizer
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
    # qterm 无信号（hit/miss 分数全部并列）→ AUROC=0.5，logit 区间下界仍高于它
    weak = [dict(r) for r in records]
    for r in weak:
        r["qterm"] = 0.5
    out3 = auroc_cluster_bootstrap(weak, n_resamples=200, seed=0)
    assert abs(out3["ci95"]["qterm"][0] - 0.5) < 1e-9
    assert out3["ci95"]["logit"][0] > out3["ci95"]["qterm"][0]
    # 样例过少（<2 个聚类）或无有效分数：安全返回空区间
    few = auroc_cluster_bootstrap(records[:8], n_resamples=50)
    assert few["ci95"] == {}
    nokey = auroc_cluster_bootstrap(
        [{"example": i, "hit": 1.0, "max_p": 0.9} for i in range(4)],
        n_resamples=50)
    assert nokey["ci95"] == {}
    print("  [ok] auroc_cluster_bootstrap（区间覆盖/确定性/聚类不足容错）")
