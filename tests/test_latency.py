"""评测流水线完整性：基准模块可运行，且已接入配置、run_all 汇总与绘图链路。"""
import json
import os
import sys
import tempfile

import torch

from lega.latency import (bench_decode, bench_decode_stats, bench_prefill,
                          bench_prefill_stats, efficiency_metrics,
                          estimate_flops_per_token, timing_stats)
from lega.model import GPT
from lega.plots import fig_latency  # noqa: F401  （确保绘图接线存在）

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_timing_stats_reports_spread():
    """逐次计时 → 中位数/p10/p90/相对散布；单调序列的统计值精确可算。"""
    st = timing_stats([1.0, 2.0, 3.0, 4.0, 5.0])
    assert st["median_ms"] == 3.0 and st["n"] == 5
    assert st["p10_ms"] == 1.4 and st["p90_ms"] == 4.6
    # spread=(4.6-1.4)/3=106.67%
    assert abs(st["spread_pct"] - 106.667) < 0.01, st
    # 恒定序列散布为 0；空输入安全返回 None
    assert timing_stats([2.0, 2.0])["spread_pct"] == 0.0
    empty = timing_stats([])
    assert empty["median_ms"] is None and empty["spread_pct"] is None
    # bool/负数/非有限/非数值不是计时观测，不能伪装成 1ms 或污染散布
    noisy = timing_stats([True, False, -1.0, float("nan"), float("inf"),
                          "bad", "1_0", None, 2.0, "4.0"])
    assert noisy["n"] == 2 and noisy["median_ms"] == 3.0, noisy
    print("  [ok] timing_stats（中位数/p10/p90/散布；非法计时按缺失）")


def test_latency_bench_runs_on_cpu():
    """prefill/decode 基准可实际运行，且 decode 的 prefill 长度按 block_size 钳制。"""
    torch.manual_seed(0)
    model = GPT(vocab_size=64, block_size=64, d_model=64, n_layers=2, n_heads=2,
                ffn_mult=4, variant="lega")
    model.eval()
    ms = bench_prefill(model, "cpu", ctx=32, batch_size=1, iters=1, warmup=1, vocab=64)
    assert ms > 0, f"prefill 数值异常: {ms}"
    dms, pf_len = bench_decode(model, "cpu", ctx=32, decode_tokens=8, iters=1,
                               warmup=1, vocab=64)
    assert dms > 0 and pf_len == 32, f"decode 基准异常: dms={dms}, pf_len={pf_len}"
    print(f"  [ok] prefill={ms:.2f}ms decode={dms:.3f}ms/tok pf={pf_len}")


def test_latency_stats_carry_measurement_noise():
    """带统计的基准必须给出逐次计时散布与 decode 的 prefill 基线噪声。"""
    torch.manual_seed(0)
    model = GPT(vocab_size=64, block_size=64, d_model=32, n_layers=2, n_heads=2,
                ffn_mult=4, variant="lega")
    model.eval()
    st = bench_prefill_stats(model, "cpu", ctx=16, batch_size=1, iters=3,
                             warmup=1, vocab=64)
    assert st["n"] == 3 and st["median_ms"] > 0
    assert st["p10_ms"] <= st["median_ms"] <= st["p90_ms"]
    assert st["spread_pct"] is not None and st["spread_pct"] >= 0
    dst, pf_len = bench_decode_stats(model, "cpu", ctx=16, decode_tokens=4,
                                     iters=3, warmup=1, vocab=64)
    assert pf_len == 16 and dst["median_ms"] > 0
    assert dst["prefill_baseline_median_ms"] > 0
    assert dst["prefill_baseline_spread_pct"] is not None
    # timing_stats 剔除负样本：总数/负样本/有效样本必须同时可见，
    # 不能把 n（有效计时数）误读成实际重复次数。
    assert dst["n_samples"] >= dst["n"] > 0, dst
    assert 0 <= dst["n_negative_samples"] <= dst["n_nonpositive_samples"]
    assert dst["n_nonpositive_samples"] <= dst["n_samples"]
    # 兼容接口返回值与统计口径一致
    assert bench_prefill(model, "cpu", ctx=16, batch_size=1, iters=2, warmup=1,
                         vocab=64) > 0
    print("  [ok] bench_*_stats（逐次计时散布 + prefill 基线噪声）")


def test_flops_estimate_matches_formula():
    """FLOPs/token 解析估计必须与公式一致，且门控项精确可算。"""
    torch.manual_seed(0)
    V, L, d, H, ctx = 64, 2, 64, 2, 32
    model = GPT(vocab_size=V, block_size=64, d_model=d, n_layers=L, n_heads=H,
                ffn_mult=4, variant="lega")
    f = estimate_flops_per_token(model, ctx)
    # matmul MACs：每层 q/k/v/o 各 d² + FFN 两个 4d²，×层数；lm_head d·V 计一次
    per_layer = 3 * d * d + d * d + 4 * d * d + 4 * d * d
    expect_mm = 2 * (per_layer * L + d * V)
    assert f["matmul_flops_per_tok"] == expect_mm, f
    # 注意力：每层 4·kv_len·d
    assert f["attn_flops_per_tok"] == 4 * ctx * d * L
    # 门控额外：每层 2d（w·q）+ ctx·H（逐头 max_j p_j）
    assert f["gate_extra_flops_per_tok"] == (2 * d + ctx * H) * L
    assert f["total_flops_per_tok"] == sum(
        f[k] for k in ("matmul_flops_per_tok", "attn_flops_per_tok",
                       "gate_extra_flops_per_tok"))
    # standard 无门控项
    f_std = estimate_flops_per_token(
        GPT(vocab_size=V, block_size=64, d_model=d, n_layers=L, n_heads=H,
            variant="standard"), ctx)
    assert f_std["gate_extra_flops_per_tok"] == 0
    # 同结构下 lega 与 standard 的 matmul/attn 完全一致（差异只来自门控）
    assert f_std["matmul_flops_per_tok"] == f["matmul_flops_per_tok"]
    assert f_std["attn_flops_per_tok"] == f["attn_flops_per_tok"]
    print(f"  [ok] FLOPs 估计（matmul={expect_mm:,}，gate 项精确可算）")


def test_flops_gate_terms_match_variant():
    """门控项必须按变体实际启用的公式项计价（qgate/null/仅长度/仅集中度）。"""
    torch.manual_seed(0)
    V, L, d, H, ctx = 64, 2, 64, 2, 32

    def gate(variant):
        m = GPT(vocab_size=V, block_size=64, d_model=d, n_layers=L, n_heads=H,
                ffn_mult=4, variant=variant)
        return estimate_flops_per_token(m, ctx)["gate_extra_flops_per_tok"]

    wq = 2 * d * L        # w·q 项
    scan = ctx * H * L    # max_j p_j 扫描
    assert gate("qgate") == wq, "qgate 无集中度项，不应计 max 扫描"
    assert gate("null") == 4 * d * L, "null 只有解析合并 null 槽的 QK+AV"
    assert gate("lega_len_only") == 0, "仅长度门控无 w·q/扫描项"
    assert gate("lega_no_query") == scan
    assert gate("lega_conc_only") == scan
    assert gate("lega") == wq + scan
    assert gate("standard") == 0
    print(f"  [ok] 门控 FLOPs 按变体计价（w·q={wq}，扫描={scan}）")


def test_efficiency_metrics_on_cpu():
    """efficiency_metrics：参数量/KV 缓存/FLOPs 齐全；CPU 无 CUDA 分配指标（记 None）。"""
    torch.manual_seed(0)
    model = GPT(vocab_size=64, block_size=64, d_model=32, n_layers=2, n_heads=2,
                variant="lega")
    model.eval()
    eff = efficiency_metrics(model, "cpu", [32, 64])
    assert eff["n_params"] == sum(p.numel() for p in model.parameters())
    assert eff["params_mb"] > 0
    # KV 缓存：2(K和V)×层数×头数×d_head×每元素字节（接口按 MB 保留 6 位小数）
    kv_bytes_expect = 2 * 2 * 2 * 16 * model.blocks[0].attn.wk.weight.element_size()
    assert abs(eff["kv_cache_mb_per_tok"] - kv_bytes_expect / 2**20) < 1e-6
    assert set(eff["flops_per_tok"]) == {"32", "64"}
    assert all(v["total_flops_per_tok"] > 0 for v in eff["flops_per_tok"].values())
    assert all(v is None for v in eff["peak_mem_mb_prefill"].values()), \
        "CPU 环境不应伪造 CUDA 分配数值"
    print(f"  [ok] efficiency_metrics（params={eff['n_params']:,}，"
          f"KV={eff['kv_cache_mb_per_tok']*2**20:.0f}B/tok）")


def test_latency_wired_into_configs_and_runner():
    """基准配置、run_all 接线与汇总/绘图链路均存在。"""
    for name in ("tiny.json", "small.json"):
        with open(os.path.join(ROOT, "configs", name), encoding="utf-8") as f:
            cfg = json.load(f)
        assert "latency" in cfg, f"configs/{name} 缺少 latency 配置块"
    with open(os.path.join(ROOT, "run_all.py"), encoding="utf-8") as f:
        src = f.read()
    assert "lega.latency" in src, "run_all.py 未接入 lega.latency"
    assert '"--seed", str(seeds[0])' in src, "run_all.py 未把首个种子传给 latency"
    assert "tradeoff_verdict" in src, "run_all.py 未接入涨点-增幅权衡判定"
    print("  [ok] 结构与运行时基准已接入 configs/run_all/plots")


def test_latency_main_records_seed_provenance():
    """latency 的随机权重/输入必须使用显式 --seed，并把实际种子写入 meta。"""
    from lega.latency import main
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "meta.json"), "w", encoding="utf-8") as f:
            json.dump({"vocab_size": 64}, f)
        out = os.path.join(d, "latency.json")
        argv = [sys.argv[0], "--config", os.path.join(ROOT, "configs", "tiny.json"),
                "--data_dir", d, "--out", out, "--variants", "standard",
                "--seed", "2024", "--smoke", "--device", "cpu"]
        old = sys.argv
        sys.argv = argv
        try:
            main()
        finally:
            sys.argv = old
        with open(out, encoding="utf-8") as f:
            payload = json.load(f)
        assert payload["meta"]["seed"] == 2024
        entry = payload["variants"]["standard"]
        assert entry["weights"] == "random" and entry["ckpt_seed"] is None
        assert entry["ckpt"] is None and entry["ckpt_sha256"] is None
    print("  [ok] latency --seed 进入 set_seed 并落盘 meta/ckpt_seed")


def test_latency_config_ints_reject_fractional_and_bool():
    """latency 配置整数不得被 int() 截断（True → 1、256.5 → 256）。

    落盘 meta 记录的是解析后的值：截断会把“配置声明的基准条件”与“实际
    测量条件”悄悄对齐成同一个错值，读者无从发现 iters/ctx 与配置不一致。
    """
    from lega.latency import _cfg_int, _cfg_int_list
    assert _cfg_int({}, "iters", 20) == 20
    assert _cfg_int({"iters": 7.0}, "iters", 20) == 7
    for bad in (True, 7.5, float("nan"), float("inf"), "abc", [7], None):
        try:
            _cfg_int({"iters": bad}, "iters", 20)
            raised = False
        except ValueError:
            raised = True
        assert raised, bad
    assert _cfg_int_list({}, "ctx_lens", [256]) == [256]
    assert _cfg_int_list({"ctx_lens": [256.0, "512"]}, "ctx_lens", []) == [256, 512]
    for bad in ([True], [256.5], 256, None, "256,512", [[]]):
        try:
            _cfg_int_list({"ctx_lens": bad}, "ctx_lens", [1])
            raised = False
        except ValueError:
            raised = True
        assert raised, bad
    print("  [ok] latency 配置整数/整数列表拒绝 bool/小数截断")
