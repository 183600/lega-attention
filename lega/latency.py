"""推理基准：预填充（prefill）与带 KV 缓存的逐 token 解码（decode）。

对比各变体在相同配置下两种前向模式的相对差异。
CUDA 上默认与 train/evaluate/ruler/analysis 一样启用 bf16 autocast
（--no_amp 回退 fp32/TF32）。
用法:
  python -m lega.latency --config configs/small.json --data_dir data/wikitext103 \
      --ckpt_dir out/small --out results/small/latency.json
"""
import argparse
import contextlib
import glob
import json
import os
import time
from typing import Dict

import torch

from .attention import VARIANTS, _LEGA_FAMILY, _USE_QUERY
from .data import load_meta
from .model import GPT, load_model, count_params
from .repro import data_fingerprint, git_info, sha256_file
from .utils import enable_tf32, get_device, set_seed, save_json


def _sync(device):
    # 必须同步模型所在设备：torch.cuda.synchronize() 默认只同步当前设备，
    # 在 cuda:N（N≠当前设备）上会量到异步发射耗时而不是真实计算耗时。
    if device is not None and str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


# ---------------------------------------------------------------- 效率指标

def estimate_flops_per_token(model, kv_len: int) -> dict:
    """前向 FLOPs/token 的解析估计（增量对比比绝对值更重要）。

    约定（与常见 2*N 口径一致，乘法+加法各计 1 FLOP）：
      * matmul：逐层 qkv/wo/ffn 权重 MACs×2 + lm_head×2（权重共享只计一次）；
      * attn：每个 query token 对 kv_len 个 key 的 QK 与 AV 各一次 → 4·kv_len·d/层；
      * gate：按变体实际启用的公式项计价——w·q（含查询项变体，2d/层）、
        对每头注意力列求 max_j p_j（含集中度项变体，≈kv_len·H/层）、
        null 槽额外一个 key 的 QK+AV（仅 null，4d/层）；standard 无门控项。
    """
    d = model.d_model
    n_layers = len(model.blocks)
    blk = model.blocks[0]
    # 每层 matmul MACs：q/k/v/o 四个 d×d 投影 + FFN 两个 d×(ffn_mult·d)
    per_layer_macs = sum(lin.weight.numel()
                         for lin in (blk.attn.wq, blk.attn.wk, blk.attn.wv,
                                     blk.attn.wo, blk.w1, blk.w2))
    matmul_macs = per_layer_macs * n_layers \
        + model.lm_head.weight.numel()   # 与 tok_emb 共享，只计一次
    attn_flops = 4 * int(kv_len) * d * n_layers
    variant = model.variant
    gate_flops = 0
    if variant in _USE_QUERY:
        gate_flops += 2 * d * n_layers                 # w·q: 每层 2·d
    if variant in _LEGA_FAMILY:
        # max_j p_j: 每头扫描 kv_len 个概率 -> 每层 kv_len·n_heads
        gate_flops += int(kv_len) * model.n_heads * n_layers
    if variant == "null":
        # 解析合并的 null 槽：每层多一个 key 的 QK 与 AV -> 4·d
        gate_flops += 4 * d * n_layers
    return {
        "matmul_flops_per_tok": int(2 * matmul_macs),
        "attn_flops_per_tok": int(attn_flops),
        "gate_extra_flops_per_tok": int(gate_flops),
        "total_flops_per_tok": int(2 * matmul_macs + attn_flops + gate_flops),
        "kv_len": int(kv_len),
        "note": "解析估计：matmul 2·MACs + 注意力 4·kv_len·d/层 + 门控项；"
                "变体间相对差比绝对值更可靠",
    }


def efficiency_metrics(model, device: str, ctxs, kv_dtype=None) -> dict:
    """参数量/KV 缓存/FLOPs 等结构指标：效率-精度权衡的输入。"""
    params_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    attn0 = model.blocks[0].attn
    kv_dtype = kv_dtype or attn0.wk.weight.dtype
    kv_bytes_per_tok = (2 * len(model.blocks) * attn0.n_heads * attn0.d_head
                        * torch.empty(0, dtype=kv_dtype).element_size())
    eff = {
        "n_params": count_params(model),
        "params_mb": round(params_bytes / 2**20, 3),
        "kv_cache_mb_per_tok": round(kv_bytes_per_tok / 2**20, 6),
        "flops_per_tok": {},
        "peak_mem_mb_prefill": {},
    }
    # 测量口径与真实推理一致：无梯度 + 与基准相同的 AMP dtype。
    amp = (device.startswith("cuda")
           and kv_dtype in (torch.bfloat16, torch.float16))
    for ctx in ctxs:
        if ctx > model.block_size:
            continue
        eff["flops_per_tok"][str(ctx)] = estimate_flops_per_token(model, ctx)
        # 单次 prefill 前向（batch=1）内实测；CPU 无 CUDA 分配语义，记 None
        if device.startswith("cuda"):
            x = torch.randint(0, model.vocab_size, (1, ctx), device=device)
            ctxmgr = (torch.autocast("cuda", dtype=kv_dtype) if amp
                      else contextlib.nullcontext())
            with torch.inference_mode(), ctxmgr:
                model(x, last_logits_only=True)  # 预热以稳定计时
                torch.cuda.reset_peak_memory_stats(device)
                model(x, last_logits_only=True)
            torch.cuda.synchronize(device)
            # 峰值内存统计也必须绑定模型所在设备（默认是 current device）。
            eff["peak_mem_mb_prefill"][str(ctx)] = round(
                torch.cuda.max_memory_allocated(device) / 2**20, 3)
            del x
        else:
            eff["peak_mem_mb_prefill"][str(ctx)] = None
    return eff


@torch.inference_mode()
def bench_prefill(model, device, ctx: int, batch_size: int, iters: int, warmup: int,
                  vocab: int) -> float:
    x = torch.randint(0, vocab, (batch_size, ctx), device=device)
    # 测的是生成式推理的 prefill：只需要最后一个位置的 logits。
    # 计算全词表 (B, T, V) logits 会掩盖注意力变体之间的真实差异。
    for _ in range(warmup):
        model(x, last_logits_only=True)
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        model(x, last_logits_only=True)
    _sync(device)
    return (time.perf_counter() - t0) / iters * 1000.0  # ms


@torch.inference_mode()
def bench_decode(model, device, ctx: int, decode_tokens: int, iters: int, warmup: int,
                 vocab: int, kv_dtype=None):
    """返回 (每 token 解码毫秒, 实际预填充长度)。

    逐 token 基线由本函数按实际 pf_len 带缓存复测，不依赖外部
    prefill 测量值。
    """
    # 预填充长度留出生成余量，避免超过 RoPE 最大位置。
    # 最大位置索引为 pf_len + decode_tokens - 2（第 1 步是 prefill，其后
    # decode_tokens - 1 步各推进 1 个位置），因此 pf_len 最大可至
    # block_size - decode_tokens + 1。
    room = model.block_size - decode_tokens + 1
    if room < 1:
        raise ValueError(
            f"block_size={model.block_size} 不足以解码 {decode_tokens} 个 token")
    pf_len = max(1, min(ctx, room))
    x = torch.randint(0, vocab, (1, pf_len), device=device)

    # 静态 KV 缓存：每层一次性预分配 (1, H, pf_len+decode_tokens-1, Dh)，
    # 逐 token 就地写入；与 generate 同源。缓存 dtype 与计算精度一致。
    attn0 = model.blocks[0].attn
    kvdtype = attn0.wk.weight.dtype if kv_dtype is None else kv_dtype
    total = pf_len + decode_tokens - 1
    kv_cache = [
        (torch.empty(1, attn0.n_heads, total, attn0.d_head,
                     dtype=kvdtype, device=device),
         torch.empty(1, attn0.n_heads, total, attn0.d_head,
                     dtype=kvdtype, device=device))
        for _ in model.blocks]

    def run_once():
        # 第 1 步：整个提示词的 prefill（cache_pos=0）
        out = model(x, kv_cache=kv_cache, cache_pos=0, last_logits_only=True)
        cur = out["logits"][:, -1].argmax(dim=-1, keepdim=True)
        # 后续 decode_tokens - 1 步：单 token 前向，cache_pos 为绝对写入位置
        for i in range(1, decode_tokens):
            out = model(cur, kv_cache=kv_cache, cache_pos=pf_len + i - 1,
                        last_logits_only=True)
            cur = out["logits"][:, -1].argmax(dim=-1, keepdim=True)

    for _ in range(warmup):
        run_once()
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        run_once()
    _sync(device)
    total = time.perf_counter() - t0
    # 每次 run_once 内含一次 pf_len 的 prefill；pf_len 可能与 ctx 不同
    # （pf_len = min(ctx, block_size - decode_tokens + 1)）。外部基线仅在
    # batch=1 且 pf_len == ctx 时可复用；其余情况均按实际 pf_len 重新测。
    # 基线必须与 run_once 内的 prefill 同口径（同样写入 KV 缓存），否则基线
    # 偏小、decode_total 被系统性高估。bench_prefill 传入的 prefill_ms 不含缓存
    # 写入，不可直接复用，这里始终按实际 pf_len 带缓存复测。
    for _ in range(warmup):  # 与 bench_prefill 保持一致的预热次数
        model(x, kv_cache=kv_cache, cache_pos=0, last_logits_only=True)
    _sync(device)
    tp = time.perf_counter()
    for _ in range(iters):
        model(x, kv_cache=kv_cache, cache_pos=0, last_logits_only=True)
    _sync(device)
    prefill_t = (time.perf_counter() - tp) / iters
    # run_once 内第 1 步是 prefill 前向（past is None 处理整条 x），其后
    # decode_tokens - 1 步才是真正的单 token decode 前向。
    if decode_tokens < 2:
        raise RuntimeError(
            f"decode_tokens={decode_tokens} 过小：第 1 步是 prefill 前向，"
            "至少需要 2 才能包含 1 步纯 decode")
    decode_total = total - prefill_t * iters
    if decode_total <= 0:
        raise RuntimeError(
            "decode 总时长不大于 prefill 基线，无法可靠扣除；请增加 iters/decode_tokens "
            "或确认 prefill 基线对应 batch=1 且长度正确")
    return decode_total / (iters * (decode_tokens - 1)) * 1000.0, pf_len  # ms/tok


def main():
    p = argparse.ArgumentParser(description="推理基准")
    p.add_argument("--config", required=True)
    p.add_argument("--data_dir", required=True, help="读取 meta.json 获取词表大小")
    p.add_argument("--ckpt_dir", default=None,
                   help="若存在 <ckpt_dir>/<variant>/model_best.pt|model.pt 则加载权重")
    p.add_argument("--out", required=True)
    p.add_argument("--variants", default="standard,standard_sdpa,qgate,null,lega")
    p.add_argument("--device", default="auto")
    p.add_argument("--no_amp", action="store_true", help="禁用 bf16 autocast（CUDA）")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    meta = load_meta(args.data_dir)
    device = get_device(args.device)
    # 与 train/evaluate/ruler/analysis 的推理口径一致：CUDA 上启用 bf16
    # autocast（--no_amp 可回退 fp32/TF32 以便与旧基准对照）；TF32 仅在
    # 回退的 fp32 路径上生效（enable_tf32 不影响 bf16/AMP 路径）。
    use_amp = (not args.no_amp) and device.startswith("cuda")
    amp_ctx = (torch.autocast("cuda", dtype=torch.bfloat16)
               if use_amp else contextlib.nullcontext())
    kv_dtype = torch.bfloat16 if use_amp else None
    enable_tf32(device)
    lcfg = dict(cfg.get("latency", {}))

    if args.smoke:
        lcfg = {"ctx_lens": [128, 256], "batch_sizes": [1],
                "decode_tokens": 16, "iters": 2, "warmup": 1}
    ctx_lens = [int(c) for c in lcfg.get("ctx_lens", [256, 512, 1024])]
    batch_sizes = [int(b) for b in lcfg.get("batch_sizes", [1, 8])]
    decode_tokens = int(lcfg.get("decode_tokens", 64))
    if decode_tokens < 2:
        raise ValueError(
            f"latency.decode_tokens={decode_tokens} 过小：第 1 步是 prefill 前向，"
            "至少需要 2 才能包含 1 步纯 decode")
    iters = int(lcfg.get("iters", 20))
    warmup = int(lcfg.get("warmup", 5))
    if device == "cpu":
        iters, warmup = min(iters, 3), min(warmup, 1)

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    set_seed(0)
    results: Dict = {}
    for v in variants:
        if v == "standard_sdpa":
            vname, use_sdpa = "standard", True
        elif v == "standard":
            vname, use_sdpa = "standard", False
        else:
            vname, use_sdpa = v, True
        assert vname in VARIANTS, f"未知变体 {v}"
        # 与 evaluate/ruler/analysis 口径一致：优先 val 上选出的 best-checkpoint。
        # 兼容两种布局：单种子 out/<tag>/<variant>/，多种子 out/<tag>/<variant>/seed<seed>/；
        # 多种子时按最小种子号的 checkpoint 测量（该指标由架构决定，与种子无关）。
        ckpt_path = None
        if args.ckpt_dir:
            base = os.path.join(args.ckpt_dir, vname)
            seed_dirs = sorted(
                (p for p in glob.glob(os.path.join(base, "seed*"))
                 if os.path.isdir(p)
                 and os.path.basename(p)[4:].isdigit()),
                key=lambda p: int(os.path.basename(p)[4:]))
            for d in [base] + seed_dirs:
                best = os.path.join(d, "model_best.pt")
                ckpt_path = (best if os.path.exists(best)
                             else os.path.join(d, "model.pt"))
                if os.path.exists(ckpt_path):
                    break
                ckpt_path = None
        if ckpt_path and os.path.exists(ckpt_path):
            model, _ = load_model(ckpt_path, device=device, use_sdpa_standard=use_sdpa)
            src = "ckpt"
        else:
            model_cfg = dict(
                vocab_size=int(meta["vocab_size"]), block_size=int(cfg["block_size"]),
                d_model=int(cfg["d_model"]), n_layers=int(cfg["n_layers"]),
                n_heads=int(cfg["n_heads"]), ffn_mult=int(cfg.get("ffn_mult", 4)),
                dropout=0.0, attn_dropout=0.0, variant=vname,
                gate_bias_init=float(cfg.get("gate_bias_init", 2.0)),
                use_sdpa_standard=use_sdpa)
            model = GPT(**model_cfg).to(device)
            src = "random"
        model.eval()
        entry = {"label": v, "variant": vname, "use_sdpa": use_sdpa, "weights": src,
                 "n_params": count_params(model), "prefill_ms": {},
                 "decode_ms_per_tok": {}, "decode_prefill_ctx": {}}
        # 效率侧指标：参数量/KV 缓存/FLOPs —— 效率-精度权衡的输入。
        entry["efficiency"] = efficiency_metrics(model, device, ctx_lens,
                                                 kv_dtype=kv_dtype)
        for ctx in ctx_lens:
            if ctx > model.block_size:
                # 与 decode 路径的钳制行为一致：超出 block_size 的 ctx 直接跳过，
                # 避免整个基准因单个配置而崩溃。
                print(f"[latency] {v}@ctx{ctx}: 跳过，超过 block_size={model.block_size}")
                continue
            entry["prefill_ms"][str(ctx)] = {}
            for bs in batch_sizes:
                with amp_ctx:
                    ms = bench_prefill(model, device, ctx, bs, iters, warmup,
                                       model.vocab_size)
                entry["prefill_ms"][str(ctx)][str(bs)] = round(ms, 4)
            # 只有 bs=1 的 prefill 才能作为逐 token decode 的扣除基线；
            # 若 batch_sizes 不含 1，则由 bench_decode 按实际 pf_len 重新测量。
            if not batch_sizes:
                raise ValueError("latency.batch_sizes 不能为空")
            with amp_ctx:
                dms, pf_len = bench_decode(model, device, ctx, decode_tokens, iters,
                                           warmup, model.vocab_size, kv_dtype=kv_dtype)
            entry["decode_ms_per_tok"][str(ctx)] = round(dms, 4)
            # decode 因生成余量被钳制时必须如实记录实际 prompt 长度，不能把
            # 较短 KV 前缀的数值标注成原 ctx 的数值。
            entry["decode_prefill_ctx"][str(ctx)] = pf_len
            bs1 = entry["prefill_ms"][str(ctx)]
            bs_key = "1" if "1" in bs1 else next(iter(bs1))
            suffix = "" if pf_len == ctx else f"（decode 实际 prefill={pf_len}）"
            print(f"[latency] {v}@ctx{ctx}: prefill(bs={bs_key})="
                  f"{bs1[bs_key]:.2f}ms "
                  f"decode={dms:.3f}ms/tok{suffix}")
        results[v] = entry

    git = git_info()
    try:
        data_fp = data_fingerprint(args.data_dir)["digest"]
    except FileNotFoundError:
        data_fp = None
    payload = {
        "meta": {"device": device, "ctx_lens": ctx_lens, "batch_sizes": batch_sizes,
                 "decode_tokens": decode_tokens, "iters": iters, "warmup": warmup,
                 "amp": bool(use_amp), "torch": torch.__version__,
                 "code_commit": git["commit"], "code_dirty": git["dirty"],
                 "data_fingerprint": data_fp,
                 "config_sha256": sha256_file(args.config)},
        "variants": results,
    }
    save_json(payload, args.out)
    print(f"[latency] 完成 -> {args.out}")


if __name__ == "__main__":
    main()
