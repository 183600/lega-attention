"""评测流水线完整性：延迟/效率基准模块可运行，且已接入配置、run_all 汇总与绘图链路。"""
import json
import os

import torch

from lega.latency import bench_decode, bench_prefill, efficiency_metrics, \
    estimate_flops_per_token
from lega.model import GPT
from lega.plots import fig_latency  # noqa: F401  （确保绘图接线存在）

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_latency_bench_runs_on_cpu():
    """prefill/decode 基准可实际运行，且 decode 的 prefill 长度按 block_size 钳制。"""
    torch.manual_seed(0)
    model = GPT(vocab_size=64, block_size=64, d_model=64, n_layers=2, n_heads=2,
                ffn_mult=4, variant="lega")
    model.eval()
    ms = bench_prefill(model, "cpu", ctx=32, batch_size=1, iters=1, warmup=1, vocab=64)
    assert ms > 0, f"prefill 耗时异常: {ms}"
    dms, pf_len = bench_decode(model, "cpu", ctx=32, decode_tokens=8, iters=1,
                               warmup=1, vocab=64)
    assert dms > 0 and pf_len == 32, f"decode 基准异常: dms={dms}, pf_len={pf_len}"
    print(f"  [ok] prefill={ms:.2f}ms decode={dms:.3f}ms/tok pf={pf_len}")


def test_flops_estimate_matches_formula():
    """FLOPs/token 解析估计必须与公式一致，且门控开销精确可算。"""
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
    # standard 无门控开销
    f_std = estimate_flops_per_token(
        GPT(vocab_size=V, block_size=64, d_model=d, n_layers=L, n_heads=H,
            variant="standard"), ctx)
    assert f_std["gate_extra_flops_per_tok"] == 0
    # 同结构下 lega 与 standard 的 matmul/attn 完全一致（差异只来自门控）
    assert f_std["matmul_flops_per_tok"] == f["matmul_flops_per_tok"]
    assert f_std["attn_flops_per_tok"] == f["attn_flops_per_tok"]
    print(f"  [ok] FLOPs 估计（matmul={expect_mm:,}，gate 开销精确可算）")


def test_efficiency_metrics_on_cpu():
    """efficiency_metrics：参数量/KV 缓存/FLOPs 齐全；CPU 无峰值显存（记 None）。"""
    torch.manual_seed(0)
    model = GPT(vocab_size=64, block_size=64, d_model=32, n_layers=2, n_heads=2,
                variant="lega")
    model.eval()
    eff = efficiency_metrics(model, "cpu", [32, 64])
    assert eff["n_params"] == sum(p.numel() for p in model.parameters())
    assert eff["params_mb"] > 0
    # KV 缓存：2(K和V)×层数×头数×d_head×每元素字节
    kv_expect = 2 * 2 * 2 * 16 * model.blocks[0].wk.weight.element_size()
    assert abs(eff["kv_cache_mb_per_tok"] * 2**20 - kv_expect) < 1e-6
    assert set(eff["flops_per_tok"]) == {"32", "64"}
    assert all(v["total_flops_per_tok"] > 0 for v in eff["flops_per_tok"].values())
    assert all(v is None for v in eff["peak_mem_mb_prefill"].values()), \
        "CPU 环境不应伪造峰值显存数值"
    print(f"  [ok] efficiency_metrics（params={eff['n_params']:,}，"
          f"KV={eff['kv_cache_mb_per_tok']*2**20:.0f}B/tok）")


def test_latency_wired_into_configs_and_runner():
    """延迟配置、run_all 接线与汇总/绘图链路均存在。"""
    for name in ("tiny.json", "small.json"):
        with open(os.path.join(ROOT, "configs", name), encoding="utf-8") as f:
            cfg = json.load(f)
        assert "latency" in cfg, f"configs/{name} 缺少 latency 配置块"
    with open(os.path.join(ROOT, "run_all.py"), encoding="utf-8") as f:
        src = f.read()
    assert "lega.latency" in src, "run_all.py 未接入 lega.latency"
    assert "tradeoff_verdict" in src, "run_all.py 未接入效率-精度权衡判定"
    print("  [ok] latency/efficiency 已接入 configs/run_all/plots")
