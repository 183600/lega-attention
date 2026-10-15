"""实际推理延迟基准：prefill（只算最后位置 logits）与带 KV 缓存的逐 token decode。

对 ckpt_dir 下每个变体的检查点分别测量：
  * prefill_ms         长度为 prefill_len 的提示一次前向（last_logits_only）耗时；
  * decode_ms_per_token  预填后逐 token 贪心解码（静态 KV 缓存）的每 token 耗时。
standard 变体额外测一条 standard_sdpa：与显式路径数值一致（有单测），
仅对比 SDPA 快速路径与显式路径的延迟，保证公平对比两者都测。

用法:
  python -m lega.latency --config configs/small.json --data_dir data/wikitext103 \
      --ckpt_dir out/small --out results/small/latency.json
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from .data import load_bin
from .model import load_model
from .utils import enable_tf32, get_device, save_json


def _sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def _timed(fn, warmup: int, reps: int, device: str) -> float:
    """返回 reps 次调用的平均毫秒数（先 warmup 次预热）。"""
    for _ in range(warmup):
        fn()
    _sync(device)
    tot = 0.0
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        _sync(device)
        tot += (time.perf_counter() - t0) * 1000.0
    return tot / max(1, reps)


def _alloc_kv(model, batch: int, total: int, device: str):
    attn0 = model.blocks[0].attn
    kvdtype = attn0.wk.weight.dtype
    return [
        (torch.empty(batch, attn0.n_heads, total, attn0.d_head,
                     dtype=kvdtype, device=device),
         torch.empty(batch, attn0.n_heads, total, attn0.d_head,
                     dtype=kvdtype, device=device))
        for _ in model.blocks]


def measure(model, prompt: torch.Tensor, prefill_len: int, decode_len: int,
            warmup: int, reps: int, device: str) -> dict:
    """prefill 与逐 token decode 的延迟（毫秒）。decode 逐 token 计时，含 KV 缓存写入。"""
    model.eval()
    idx = prompt[:, :prefill_len]
    total = prefill_len + decode_len
    with torch.inference_mode():
        # prefill：一次前向，只取最后位置 logits（生成场景口径）
        def do_prefill():
            model(idx, last_logits_only=True)
        prefill_ms = _timed(do_prefill, warmup, reps, device)

        # decode：先在静态缓存上预填（不计时），由预填 logits 得到位置 prefill_len
        # 的解码 token，再逐 token 前向并写缓存；缓存位置与 model.generate 的
        # 逐 token 口径一致（0..prefill_len-1 已由预填写入）。
        def decode_once():
            kv_cache = _alloc_kv(model, idx.size(0), total, device)
            pre = model(idx, kv_cache=kv_cache, cache_pos=0, last_logits_only=True)
            cur = pre["logits"][:, -1].argmax(dim=-1, keepdim=True)
            _sync(device)
            t0 = time.perf_counter()
            for i in range(decode_len):
                out = model(cur, kv_cache=kv_cache, cache_pos=prefill_len + i,
                            last_logits_only=True)
                cur = out["logits"][:, -1].argmax(dim=-1, keepdim=True)
            _sync(device)
            return (time.perf_counter() - t0) * 1000.0
        # 预热同样走完整 decode（分配器/内核预热）
        for _ in range(warmup):
            decode_once()
        dec_ms = sum(decode_once() for _ in range(reps)) / max(1, reps)
    return {
        "prefill_ms": prefill_ms,
        "decode_ms_per_token": dec_ms / max(1, decode_len),
        "prefill_len": prefill_len,
        "decode_len": decode_len,
    }


def main():
    p = argparse.ArgumentParser(description="prefill / decode 延迟基准")
    p.add_argument("--config", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--ckpt_dir", required=True,
                   help="按 <ckpt_dir>/<variant>/model_best.pt|model.pt 找检查点")
    p.add_argument("--out", required=True)
    p.add_argument("--prefill_len", type=int, default=None)
    p.add_argument("--decode_len", type=int, default=None)
    p.add_argument("--warmup", type=int, default=None)
    p.add_argument("--reps", type=int, default=None)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    lat = cfg.get("latency", {})
    device = get_device(args.device)
    enable_tf32(device)

    # 提示取自 val.bin：长度基准对提示内容不敏感，任意真实 token 序列即可
    val = load_bin(args.data_dir, "val")
    block_size = int(cfg["block_size"])
    decode_len = int(args.decode_len or lat.get("decode_len", 64))
    prefill_len = int(args.prefill_len
                      or min(int(lat.get("prefill_len", 512)), block_size))
    # 生成上限受 block_size（RoPE 位置表）约束，与 model.forward 的断言一致
    prefill_len = min(prefill_len, block_size - decode_len, len(val) - 1)
    assert prefill_len > 0, "block_size 不足以容纳 prefill_len + decode_len"
    warmup = int(args.warmup if args.warmup is not None else lat.get("warmup", 2))
    reps = int(args.reps if args.reps is not None else lat.get("reps", 5))
    prompt = torch.from_numpy(
        np.asarray(val[:prefill_len], dtype=np.int64)).unsqueeze(0).to(device)

    results = {}
    skipped = []
    for entry in sorted(os.listdir(args.ckpt_dir)):
        vdir = os.path.join(args.ckpt_dir, entry)
        if not os.path.isdir(vdir):
            continue
        ckpt = (os.path.join(vdir, "model_best.pt")
                if os.path.exists(os.path.join(vdir, "model_best.pt"))
                else os.path.join(vdir, "model.pt"))
        if not os.path.exists(ckpt):
            skipped.append(entry)
            continue
        if entry == "standard":
            # 显式路径与 SDPA 快速路径都测：两者数值一致（有单测），只比延迟
            for tag, use_sdpa in (("standard", False), ("standard_sdpa", True)):
                model, _ = load_model(ckpt, device=device,
                                      use_sdpa_standard=use_sdpa)
                results[tag] = measure(model, prompt, prefill_len, decode_len,
                                       warmup, reps, device)
                results[tag]["ckpt"] = ckpt
                del model
        else:
            model, ck = load_model(ckpt, device=device)
            results[ck.get("variant", entry)] = measure(
                model, prompt, prefill_len, decode_len, warmup, reps, device)
            results[ck.get("variant", entry)]["ckpt"] = ckpt
            del model

    payload = {
        "device": device,
        "config": args.config,
        "ckpt_dir": args.ckpt_dir,
        "prefill_len": prefill_len,
        "decode_len": decode_len,
        "warmup": warmup,
        "reps": reps,
        "amp": False,   # 固定 fp32 计时，避免与训练 amp 设置耦合
        "results": results,
        "skipped": skipped,
    }
    save_json(payload, args.out)
    for tag, m in results.items():
        print(f"[latency] {tag}: prefill={m['prefill_ms']:.2f}ms "
              f"decode={m['decode_ms_per_token']:.2f}ms/token")
    print(f"[latency] -> {args.out}")


if __name__ == "__main__":
    main()
