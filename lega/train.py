"""训练脚本（全部变体共用，保证相同参数量级与训练步数）。

用法:
  python -m lega.train --config configs/tiny.json --variant lega \
      --data_dir data/dummy --out_dir out/tiny/lega [--smoke] [--wandb]
"""
import argparse
import contextlib
import json
import math
import os
import time

# 须在任何 CUDA 初始化之前生效。
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch

from .attention import VARIANTS, VARIANT_CN
from .data import load_bin, load_meta, get_train_batch, get_val_batch, get_val_windows
from .model import GPT, configure_optimizer, count_params, gate_extra_params, save_checkpoint
from .utils import set_seed, now_str, save_json

SMOKE_OVERRIDES = {
    "steps": 20, "eval_every": 10, "eval_iters": 2, "log_every": 5,
    "warmup": 5, "batch_size": 8, "train_ctx": 128,
}


def _val_eval_indices(val_len: int, ctx: int, iters: int):
    """均匀覆盖整个验证集的不重复窗口序号。

    旧实现从第 0 个窗口顺序取 idx，且 eval_iters 超过窗口数时循环复用，
    会得到偏向文本开头、覆盖重复的 PPL/gate 统计。这里改为：
      * iters > 可用窗口数：评估全部窗口（每个只算一次）；
      * iters <= 可用窗口数：在整个验证集上等间隔抽样（含首尾窗口）。
    """
    iters = int(iters)
    if iters < 1:
        raise ValueError(f"eval_iters 必须 >= 1，收到 {iters}")
    n_windows = (int(val_len) - 1) // int(ctx)
    if n_windows <= 0:
        raise ValueError(f"验证集只有 {val_len} tokens，不足以组成 ctx={ctx} 的窗口")
    n_eval = min(iters, n_windows)
    if n_eval == 1:
        # 单窗估计取中间窗口，避免天然偏向验证集开头
        indices = np.array([(n_windows - 1) // 2], dtype=np.int64)
    else:
        indices = (np.arange(n_eval, dtype=np.int64)
                   * (n_windows - 1) // (n_eval - 1))
    return indices, n_windows


def evaluate_val(model, val, ctx, device, iters, amp=False, batch_windows: int = None):
    """在不重复、覆盖全验证集的窗口上同时计算 val_loss 与门控均值。

    多个窗口拼成一个批量做前向：每个窗口长度相同，token 加权后的均值
    loss/门控统计与逐窗口完全一致。
    batch_windows 默认按每批约 16384 tokens 估算，并随验证集窗口数收窄。
    """
    was_training = model.training
    model.eval()
    collect_gate = model.variant not in ("standard", "null")
    indices, n_windows = _val_eval_indices(len(val), ctx, iters)
    if batch_windows is None:
        batch_windows = max(1, min(16384 // max(1, ctx), len(indices)))
    batch_windows = int(batch_windows)
    # 设备端累积、循环结束后一次性 .item()：避免每个窗口批一次 GPU 同步
    loss_acc = None
    gate_acc = None
    weights = []
    gate_weights = []
    amp_ctx = (torch.autocast("cuda", dtype=torch.bfloat16)
               if (amp and device.startswith("cuda")) else contextlib.nullcontext())
    # inference_mode：纯前向评测不生成 autograd 元数据（版本计数/保存变量）；
    # 返回的统计只经 .item() 消费。
    with torch.inference_mode():
        for s in range(0, len(indices), batch_windows):
            chunk = indices[s:s + batch_windows]
            x, y = get_val_windows(val, ctx, chunk, device)
            with amp_ctx:
                # 验证评测只消费门控均值 g，使用轻量统计
                out = model(x, targets=y, need_stats=collect_gate, stats_argmax=False)
            nb = len(chunk)     # 窗口等长，按窗口数加权即 token 加权
            lt = out["loss"].detach().float() * nb
            loss_acc = lt if loss_acc is None else loss_acc + lt
            weights.append(nb)
            if collect_gate and out["gate_stats"]:
                gs = [s_["g"].detach().float().mean()
                      for s_ in out["gate_stats"] if s_.get("g") is not None]
                if gs:
                    gt = torch.stack(gs).mean() * nb
                    gate_acc = gt if gate_acc is None else gate_acc + gt
                    gate_weights.append(nb)
    w = np.asarray(weights, dtype=np.float64)
    # 必须保持为 Python float：numpy 标量会进 checkpoint extra，
    # 使 torch.load(weights_only=True) 反序列化失败且不合法
    val_loss = float(float(loss_acc.item()) / w.sum())
    gate_mean = (float(gate_acc.item()) / float(np.sum(gate_weights))
                 if gate_weights else None)
    if was_training:
        model.train()
    return {
        "val_loss": val_loss,
        "gate_mean": gate_mean,
        "eval_iters_requested": int(iters),
        "eval_iters": int(len(indices)),
        "val_windows": int(n_windows),
        "val_tokens_evaluated": int(len(indices) * ctx),
    }


def estimate_val_loss(model, val, ctx, device, iters, amp=False):
    """兼容旧调用：只返回 val_loss。"""
    return evaluate_val(model, val, ctx, device, iters, amp=amp)["val_loss"]


def mean_gate_value(model, val, ctx, device, iters=1, amp=False):
    """兼容旧调用的门控均值；与 evaluate_val 使用同一组验证窗口与精度。"""
    return evaluate_val(model, val, ctx, device, iters, amp=amp)["gate_mean"]


def main():
    p = argparse.ArgumentParser(description="LEGA 训练")
    p.add_argument("--config", required=True, help="实验配置 JSON")
    p.add_argument("--variant", required=True, choices=VARIANTS)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out_dir", default=None, help="默认 out/<config名>/<variant>")
    p.add_argument("--smoke", action="store_true", help="冒烟模式（极少步数，验证流程）")
    p.add_argument("--wandb", action="store_true", help="可选 W&B 日志（需先 wandb login）")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--steps", type=int, default=None, help="覆盖配置中的训练步数")
    args = p.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if args.smoke:
        for k, v in SMOKE_OVERRIDES.items():
            cfg[k] = min(v, cfg.get(k, v)) if k in ("batch_size",) else v
    if args.steps is not None:
        cfg["steps"] = args.steps
    seed = args.seed if args.seed is not None else cfg.get("seed", 1337)
    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")

    tag = os.path.splitext(os.path.basename(args.config))[0] + ("_smoke" if args.smoke else "")
    out_dir = args.out_dir or os.path.join("out", tag, args.variant)
    os.makedirs(out_dir, exist_ok=True)
    cfg_used = dict(cfg)
    cfg_used.update({"variant": args.variant, "seed": seed, "device": device, "tag": tag})
    with open(os.path.join(out_dir, "config_used.json"), "w", encoding="utf-8") as f:
        json.dump(cfg_used, f, ensure_ascii=False, indent=2)

    set_seed(seed)
    meta = load_meta(args.data_dir)
    train = load_bin(args.data_dir, "train")
    val = load_bin(args.data_dir, "val")

    block_size = int(cfg["block_size"])
    train_ctx = min(int(cfg.get("train_ctx", block_size)), block_size)

    model_cfg = dict(
        vocab_size=int(meta["vocab_size"]), block_size=block_size,
        d_model=int(cfg["d_model"]), n_layers=int(cfg["n_layers"]),
        n_heads=int(cfg["n_heads"]), ffn_mult=int(cfg.get("ffn_mult", 4)),
        dropout=float(cfg.get("dropout", 0.0)),
        attn_dropout=float(cfg.get("attn_dropout", 0.0)),
        variant=args.variant,
        gate_bias_init=float(cfg.get("gate_bias_init", 2.0)),
        use_sdpa_standard=bool(cfg.get("use_sdpa_standard", True)),
    )
    if device.startswith("cuda"):
        # 对精度不敏感的 fp32 累乘走 TF32（Ampere+）；
        # 门控路径本身在 AMP 下已是 bf16，不受影响
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    model = GPT(**model_cfg).to(device)
    if cfg.get("compile", False) and device.startswith("cuda"):
        try:
            model = torch.compile(model)
            print("[train] torch.compile 已启用")
        except Exception as e:
            print(f"[train] torch.compile 失败（回退 eager）: {e}")
    n_params = count_params(model)
    n_extra = gate_extra_params(model)
    print(f"[train] variant={args.variant}({VARIANT_CN[args.variant]}) device={device} "
          f"params={n_params:,} (门控额外 {n_extra:,})")
    print(f"[train] steps={cfg['steps']} batch={cfg['batch_size']} ctx={train_ctx} "
          f"tokens/step={cfg['batch_size'] * train_ctx:,} seed={seed}")

    opt = configure_optimizer(
        model, lr=float(cfg["lr"]), weight_decay=float(cfg.get("weight_decay", 0.1)),
        fused=(device.startswith("cuda")))
    base_lrs = [g["lr"] for g in opt.param_groups]
    # 梯度裁剪的参数列表与优化器同口径，只构建一次；
    # 训练热循环里不再每步重新遍历生成器
    clip_params = [p for p in model.parameters() if p.requires_grad]

    steps = int(cfg["steps"])
    warmup = int(cfg.get("warmup", 100))
    grad_accum = int(cfg.get("grad_accum", 1))
    log_every = int(cfg.get("log_every", 10))
    eval_every = int(cfg.get("eval_every", 500))
    eval_iters = int(cfg.get("eval_iters", 50))
    grad_clip = float(cfg.get("grad_clip", 1.0))
    use_amp = bool(cfg.get("amp", True)) and device.startswith("cuda")
    amp_ctx = (torch.autocast("cuda", dtype=torch.bfloat16)
               if use_amp else contextlib.nullcontext())

    def lr_factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(1, warmup)
        prog = min(1.0, (step - warmup) / max(1, steps - 1 - warmup))
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * prog))

    wb = None
    if args.wandb:
        try:
            import wandb
            wb = wandb.init(project="lega-attention", name=f"{tag}-{args.variant}",
                            config=cfg_used, reinit=True)
            print("[train] WandB 已启动")
        except Exception as e:
            print(f"[train] WandB 启动失败（忽略，继续本地训练）: {e}")
            wb = None

    log_path = os.path.join(out_dir, "train_log.jsonl")
    # 同一 out_dir 重跑代表一次新实验：覆盖旧日志，避免 step 轴混入上一轮数据。
    log_f = open(log_path, "w", encoding="utf-8")
    rng = np.random.default_rng(seed)
    model.train()
    opt.zero_grad(set_to_none=True)
    t0 = time.time()
    tokens_seen = 0
    best_val = float("inf")

    def do_eval(step):
        nonlocal best_val
        metrics = evaluate_val(model, val, train_ctx, device, eval_iters, amp=use_amp)
        vl = metrics["val_loss"]
        gmean = metrics["gate_mean"]
        ppl = math.exp(min(20, vl))
        rec = {"step": step, **metrics, "val_ppl": ppl,
               "val_ppl_clamped": bool(vl > 20), "time": now_str()}
        log_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        log_f.flush()
        mark = ""
        if vl < best_val:
            best_val = vl
            mark = " (best)"
        print(f"[train] step {step}: val_ppl={ppl:.3f}"
              + (f" gate_mean={gmean:.3f}" if gmean is not None else "") + mark)
        if wb:
            wb.log({"val_ppl": ppl, "val_loss": vl,
                    **({"gate_mean": gmean} if gmean is not None else {})},
                   step=step)
        return vl

    for step in range(steps):
        lr_f = lr_factor(step)
        for g, base in zip(opt.param_groups, base_lrs):
            g["lr"] = base * lr_f

        # 在设备上累积 loss，避免每个 micro-step 一次 .item() 引发的 GPU 同步
        loss_acc_dev = torch.zeros((), device=device)
        for _ in range(grad_accum):
            x, y = get_train_batch(train, train_ctx, int(cfg["batch_size"]), device, rng)
            with amp_ctx:
                out = model(x, targets=y)
                loss = out["loss"] / grad_accum
            loss.backward()
            loss_acc_dev.add_(out["loss"].detach(), alpha=1.0 / grad_accum)
            # backward 已消费完计算图；解除对 out（含逐层 past_kv）与 loss 的引用
            del out, loss
        torch.nn.utils.clip_grad_norm_(clip_params, grad_clip)
        opt.step()
        opt.zero_grad(set_to_none=True)
        tokens_seen += int(cfg["batch_size"]) * train_ctx * grad_accum

        if (step + 1) % log_every == 0 or step == 0:
            loss_acc = float(loss_acc_dev.item())   # 仅在记录日志时同步
            dt = time.time() - t0
            rec = {"step": step + 1, "loss": loss_acc, "ppl": math.exp(min(20, loss_acc)),
                   "lr": base_lrs[0] * lr_f, "tok_per_s": tokens_seen / max(dt, 1e-6),
                   "time": now_str()}
            log_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            log_f.flush()
            print(f"[train] step {step+1}/{steps} loss={loss_acc:.4f} "
                  f"ppl={math.exp(min(20, loss_acc)):.3f} lr={base_lrs[0]*lr_f:.2e} "
                  f"tok/s={tokens_seen/max(dt,1e-6):,.0f}")
            if wb:
                wb.log({"loss": loss_acc, "lr": base_lrs[0] * lr_f,
                        "tok_per_s": tokens_seen / max(dt, 1e-6)}, step=step + 1)
        if (step + 1) % eval_every == 0:
            do_eval(step + 1)

    final_metrics = evaluate_val(model, val, train_ctx, device, eval_iters, amp=use_amp)
    final_val_loss = final_metrics["val_loss"]
    final_ppl = math.exp(min(20, final_val_loss))
    gmean = final_metrics["gate_mean"]
    save_checkpoint(os.path.join(out_dir, "model.pt"), model,
                    extra={"variant": args.variant, "step": steps, "seed": seed,
                           "val_ppl": final_ppl, "val_loss": final_val_loss,
                           "val_ppl_clamped": bool(final_val_loss > 20),
                           "tokens_seen": tokens_seen,
                           "n_params": n_params, "n_gate_extra": n_extra})
    save_json({"variant": args.variant, "val_ppl": final_ppl, "val_loss": final_val_loss,
               "val_ppl_clamped": bool(final_val_loss > 20),
               "gate_mean": gmean, **{k: final_metrics[k] for k in
                                      ("eval_iters_requested", "eval_iters",
                                       "val_windows", "val_tokens_evaluated")},
               "n_params": n_params, "n_gate_extra": n_extra,
               "tokens_seen": tokens_seen, "steps": steps, "seed": seed,
               "wall_time_s": time.time() - t0},
              os.path.join(out_dir, "eval.json"))
    log_f.write(json.dumps({"step": steps, **final_metrics,
                            "val_ppl": final_ppl,
                            "val_ppl_clamped": bool(final_val_loss > 20),
                            "final": True},
                           ensure_ascii=False) + "\n")
    log_f.close()
    print(f"[train] 完成: val_ppl={final_ppl:.3f} 用时 {time.time()-t0:.1f}s -> {out_dir}")
    if wb:
        wb.finish()


if __name__ == "__main__":
    main()
