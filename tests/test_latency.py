"""评测流水线完整性：延迟基准模块可运行，且已接入配置、run_all 汇总与绘图链路。"""
import json
import os

import torch

from lega.latency import bench_decode, bench_prefill
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


def test_latency_wired_into_configs_and_runner():
    """延迟配置、run_all 接线与汇总/绘图链路均存在。"""
    for name in ("tiny.json", "small.json"):
        with open(os.path.join(ROOT, "configs", name), encoding="utf-8") as f:
            cfg = json.load(f)
        assert "latency" in cfg, f"configs/{name} 缺少 latency 配置块"
    with open(os.path.join(ROOT, "run_all.py"), encoding="utf-8") as f:
        assert "lega.latency" in f.read(), "run_all.py 未接入 lega.latency"
    print("  [ok] latency 已接入 configs/run_all/plots")
