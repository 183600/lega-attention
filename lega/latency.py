"""推理延迟基准：预填充（prefill）与带 KV 缓存的逐 token 解码（decode）。

对比各变体的推理耗时，回答“实际推理延迟”这一评测维度。
用法:
  python -m lega.latency --config configs/small.json --data_dir data/wikitext103 \
      --ckpt_dir out/small --out results/small/latency.json
"""
import argparse
import json
import os
import time
from typing import Dict

import torch

from .attention import VARIANTS
from .data import load_meta
from .model import GPT, load_model, count_params
from .utils import enable_tf32, get_device, set_seed, save_json


def _sync(device):
    if device.startswith("cuda"):
        torch.cuda.synchronize()


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
                 vocab: int):
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
    # 逐 token 就地写入；与 generate 同源。
    attn0 = model.blocks[0].attn
    kvdtype = attn0.wk.weight.dtype
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
    p = argparse.ArgumentParser(description="推理延迟基准")
    p.add_argument("--config", required=True)
    p.add_argument("--data_dir", required=True, help="读取 meta.json 获取词表大小")
    p.add_argument("--ckpt_dir", default=None,
                   help="若存在 <ckpt_dir>/<variant>/model_best.pt|model.pt 则加载权重")
    p.add_argument("--out", required=True)
    p.add_argument("--variants", default="standard,standard_sdpa,qgate,null,lega")
    p.add_argument("--device", default="auto")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    meta = load_meta(args.data_dir)
    device = get_device(args.device)
    # 与 train/evaluate/ruler/analysis 的推理口径一致：CUDA 上启用 TF32，
    # 否则基准测的是与流水线其它环节不同的精度设置。
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
        # 与 evaluate/ruler/analysis 口径一致：优先 val 上选出的 best-checkpoint
        ckpt_path = None
        if args.ckpt_dir:
            best = os.path.join(args.ckpt_dir, vname, "model_best.pt")
            ckpt_path = (best if os.path.exists(best)
                         else os.path.join(args.ckpt_dir, vname, "model.pt"))
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
        for ctx in ctx_lens:
            if ctx > model.block_size:
                # 与 decode 路径的钳制行为一致：超出 block_size 的 ctx 直接跳过，
                # 避免整个基准因单个配置而崩溃。
                print(f"[latency] {v}@ctx{ctx}: 跳过，超过 block_size={model.block_size}")
                continue
            entry["prefill_ms"][str(ctx)] = {}
            for bs in batch_sizes:
                ms = bench_prefill(model, device, ctx, bs, iters, warmup, model.vocab_size)
                entry["prefill_ms"][str(ctx)][str(bs)] = round(ms, 4)
            # 只有 bs=1 的 prefill 才能作为逐 token decode 的扣除基线；
            # 若 batch_sizes 不含 1，则由 bench_decode 按实际 pf_len 重新测量。
            if not batch_sizes:
                raise ValueError("latency.batch_sizes 不能为空")
            dms, pf_len = bench_decode(model, device, ctx, decode_tokens, iters, warmup,
                                       model.vocab_size)
            entry["decode_ms_per_tok"][str(ctx)] = round(dms, 4)
            # decode 因生成余量被钳制时必须如实记录实际 prompt 长度，不能把
            # 较短 KV 前缀的耗时标注成原 ctx 的耗时。
            entry["decode_prefill_ctx"][str(ctx)] = pf_len
            bs1 = entry["prefill_ms"][str(ctx)]
            bs_key = "1" if "1" in bs1 else next(iter(bs1))
            suffix = "" if pf_len == ctx else f"（decode 实际 prefill={pf_len}）"
            print(f"[latency] {v}@ctx{ctx}: prefill(bs={bs_key})="
                  f"{bs1[bs_key]:.2f}ms "
                  f"decode={dms:.3f}ms/tok{suffix}")
        results[v] = entry

    payload = {
        "meta": {"device": device, "ctx_lens": ctx_lens, "batch_sizes": batch_sizes,
                 "decode_tokens": decode_tokens, "iters": iters, "warmup": warmup,
                 "torch": torch.__version__},
        "variants": results,
    }
    save_json(payload, args.out)
    print(f"[latency] 完成 -> {args.out}")


if __name__ == "__main__":
    main()
