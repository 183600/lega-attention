"""过拟合小批量 sanity check —— 全量训练前的第一道代码正确性闸门。

原理：抽取一个固定小批次（默认 batch_size 条、ctx 个 token），关闭全部
正则化与数据扰动后反复优化同一批数据。若前向/反向/损失/标签映射任一环节
有严重 bug（梯度断流、loss 计算错误、label 错位等），训练损失不可能降到
接近 0、下一 token 准确率也不可能接近 100%；反之，能过拟合只能说明
“这条最小学习链路是通的”。

通过是**必要不充分**条件：数据增强、特征对齐、标签映射等 bug 仍可能
漏网（本检查不含增强与真实数据顺序），需配合中间结果检查（首步 loss ≈
log(V)、梯度范数 > 0、逐层输出统计等），并保留本 JSON 作为实验记录。

用法:
  python -m lega.sanity --config configs/tiny.json --data_dir data/dummy \
      --variant lega --out results/tiny_sanity/lega.json
退出码：0=通过，1=未通过（run_all 会用其阻断后续全量训练）。
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

from .attention import VARIANTS, VARIANT_CN
from .data import load_bin, load_meta, get_train_batch
from .model import GPT, count_params
from .repro import build_run_manifest
from .utils import save_json, set_seed

# 冒烟模式：更小批次/更少步数，保持“能快速过拟合”的性质
SMOKE_OVERRIDES = {"steps": 60, "batch_size": 4, "ctx": 64, "log_every": 10}

DEFAULTS = {"steps": 300, "batch_size": 8, "ctx": 128, "lr": 3e-3,
            "warmup": 10, "grad_clip": 1.0, "log_every": 25,
            "loss_threshold": 0.05, "acc_threshold": 0.99}


@torch.no_grad()
def batch_metrics(model, x: torch.Tensor, y: torch.Tensor,
                  max_chunk_tokens: int = 8192) -> dict:
    """固定批的 loss / 下一 token 准确率 / 整条序列全对率（fp32、逐块计算）。

    直接消费 logits 计算指标（而非模型的训练路径），因此同时校验了
    lm_head 输出与标签的对齐关系。
    """
    was_training = model.training
    model.eval()
    B, T = x.shape
    V = model.vocab_size
    rows = max(1, int(max_chunk_tokens) // max(1, T))
    total_loss = 0.0
    n_tokens = 0
    correct = 0
    exact = 0
    for s in range(0, B, rows):
        xb, yb = x[s:s + rows], y[s:s + rows]
        logits = model(xb)["logits"]                    # (b, T, V) fp32
        lf = logits.reshape(-1, V)
        tgt = yb.reshape(-1)
        total_loss += float(F.cross_entropy(lf, tgt, reduction="sum"))
        pred = lf.argmax(dim=-1)
        correct += int((pred == tgt).sum())
        n_tokens += int(tgt.numel())
        exact += int((pred.reshape(xb.shape[0], T) == yb).all(dim=1).sum())
    if was_training:
        model.train()
    return {
        "loss": total_loss / max(1, n_tokens),
        "token_acc": correct / max(1, n_tokens),
        "exact_match": exact / max(1, B),
        "n_tokens": n_tokens,
    }


def run_overfit_check(model: GPT, x: torch.Tensor, y: torch.Tensor, *,
                      steps: int, lr: float, device: str, seed: int,
                      warmup: int = 10, grad_clip: float = 1.0,
                      log_every: int = 25,
                      loss_threshold: float = 0.05,
                      acc_threshold: float = 0.99) -> dict:
    """在同一个固定批上训练 model 并返回可审计的检查结果。

    调用方必须已保证 model 的正则化已关闭（dropout=0/attn_dropout=0，
    优化器 weight_decay=0）；本函数只用固定批、无任何数据增强。
    """
    set_seed(seed)
    # 关闭 TF32：sanity 追求数值上确定地“压到 0”，与训练速度无关
    prev_tf32_mm = torch.backends.cuda.matmul.allow_tf32
    prev_tf32_cudnn = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        model.train()
        initial = batch_metrics(model, x, y)
        opt = torch.optim.AdamW(model.parameters(), lr=float(lr),
                                weight_decay=0.0, betas=(0.9, 0.95))
        history = []
        first_grad_norm = None
        nan_seen = False
        loss_no_grad = False
        for step in range(int(steps)):
            cur_lr = float(lr) * (step + 1) / max(1, warmup) \
                if step < warmup else float(lr)
            for g in opt.param_groups:
                g["lr"] = cur_lr
            opt.zero_grad(set_to_none=True)
            loss = model(x, targets=y)["loss"]
            if not torch.isfinite(loss):
                nan_seen = True
                break
            # 梯度断流的典型表现：loss 不挂在 autograd 图上（如误用 detach、
            # 冻结了全部参数）。直接等价于“梯度没有反向传播”，记为失败而非崩溃。
            if not loss.requires_grad:
                loss_no_grad = True
                break
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], grad_clip)
            if first_grad_norm is None:
                first_grad_norm = float(gn)
            opt.step()
            if (step + 1) % int(log_every) == 0 or step == 0:
                history.append({"step": step + 1, "loss": float(loss.detach())})
        final = batch_metrics(model, x, y)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_tf32_mm
        torch.backends.cudnn.allow_tf32 = prev_tf32_cudnn
        model.eval()

    checks = {
        "loss_below_threshold": final["loss"] <= float(loss_threshold),
        "token_acc_above_threshold": final["token_acc"] >= float(acc_threshold),
        "loss_decreased": final["loss"] < initial["loss"],
        "finite": (not nan_seen) and math.isfinite(final["loss"]),
        "loss_requires_grad": not loss_no_grad,
        "gradient_nonzero_first_step": (first_grad_norm is not None
                                        and first_grad_norm > 0.0),
    }
    passed = all(checks.values())
    return {
        "check": "overfit_small_batch",
        "passed": bool(passed),
        "checks": checks,
        "thresholds": {"loss": float(loss_threshold),
                       "token_acc": float(acc_threshold)},
        "initial": initial,
        "final": final,
        "log_vocab": math.log(model.vocab_size),
        "first_grad_norm": first_grad_norm,
        "steps": int(steps),
        "lr": float(lr),
        "history": history,
        "caveat": ("通过为必要非充分条件：数据增强/特征对齐/标签映射等 bug "
                   "仍可能漏网；请配合中间结果检查（首步 loss≈log(V)、"
                   "梯度范数>0、逐层统计）并使用真实数据小规模预跑。"),
    }


def main():
    p = argparse.ArgumentParser(description="过拟合小批量 sanity check")
    p.add_argument("--config", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--variant", required=True, choices=VARIANTS)
    p.add_argument("--out", default=None, help="结果 JSON 输出路径")
    p.add_argument("--device", default="auto")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--ctx", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--loss_threshold", type=float, default=None)
    p.add_argument("--acc_threshold", type=float, default=None)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    opt = dict(DEFAULTS)
    opt["loss_threshold"] = float(cfg.get("sanity", {}).get(
        "loss_threshold", DEFAULTS["loss_threshold"]))
    opt["acc_threshold"] = float(cfg.get("sanity", {}).get(
        "acc_threshold", DEFAULTS["acc_threshold"]))
    if args.smoke:
        opt.update(SMOKE_OVERRIDES)
    for name in ("steps", "batch_size", "ctx", "lr", "seed",
                 "loss_threshold", "acc_threshold"):
        val = getattr(args, name)
        if val is not None:
            opt[name] = val
    seed = int(opt.get("seed") if opt.get("seed") is not None
               else cfg.get("seed", 1337))
    device = (args.device if args.device != "auto"
              else ("cuda" if torch.cuda.is_available() else "cpu"))
    set_seed(seed)

    meta = load_meta(args.data_dir)
    train = load_bin(args.data_dir, "train")
    block_size = int(cfg["block_size"])
    train_ctx = min(int(cfg.get("train_ctx", block_size)), block_size)
    ctx = min(int(opt["ctx"]), train_ctx)
    batch_size = int(opt["batch_size"])

    # 显式关闭正则化：dropout/attn_dropout=0；优化器 weight_decay=0（见上）
    model = GPT(
        vocab_size=int(meta["vocab_size"]), block_size=block_size,
        d_model=int(cfg["d_model"]), n_layers=int(cfg["n_layers"]),
        n_heads=int(cfg["n_heads"]), ffn_mult=int(cfg.get("ffn_mult", 4)),
        dropout=0.0, attn_dropout=0.0, variant=args.variant,
        gate_bias_init=float(cfg.get("gate_bias_init", 2.0)),
        use_sdpa_standard=bool(cfg.get("use_sdpa_standard", True)),
    ).to(device)
    print(f"[sanity] variant={args.variant}({VARIANT_CN[args.variant]}) "
          f"device={device} params={count_params(model):,}")
    # 固定批：整轮训练复用同一批数据（无增强、无重采样）
    rng = np.random.default_rng(seed)
    x, y = get_train_batch(train, ctx, batch_size, device, rng)
    print(f"[sanity] 固定批: {batch_size} 条 x {ctx} tokens，"
          f"steps={opt['steps']} lr={opt['lr']} seed={seed}")

    res = run_overfit_check(
        model, x, y, steps=int(opt["steps"]), lr=float(opt["lr"]),
        device=device, seed=seed, warmup=int(opt["warmup"]),
        grad_clip=float(opt["grad_clip"]), log_every=int(opt["log_every"]),
        loss_threshold=float(opt["loss_threshold"]),
        acc_threshold=float(opt["acc_threshold"]))
    res.update({
        "variant": args.variant,
        "variant_cn": VARIANT_CN[args.variant],
        "seed": seed,
        "batch_size": batch_size,
        "ctx": ctx,
        "device": device,
        "config_file": os.path.abspath(args.config),
        "data_dir": os.path.abspath(args.data_dir),
        "manifest": build_run_manifest(
            tag="sanity", variant=args.variant, seed=seed,
            config_path=args.config, data_dir=args.data_dir, device=device),
    })
    last = res["history"][-1]["loss"] if res["history"] else float("nan")
    print(f"[sanity] {'PASS' if res['passed'] else 'FAIL'}: "
          f"loss {res['initial']['loss']:.4f} -> {res['final']['loss']:.4f} "
          f"(曲线末值 {last:.4f}, logV={res['log_vocab']:.3f}), "
          f"token_acc={res['final']['token_acc']:.4f}, "
          f"exact={res['final']['exact_match']:.3f}")
    if not res["passed"]:
        failed = [k for k, ok in res["checks"].items() if not ok]
        print(f"[sanity] 未通过项: {', '.join(failed)}；"
              "若无法过拟合小批量，说明梯度/损失/标签存在严重 bug，"
              "请先修复再开全量训练。")
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        save_json(res, args.out)
        print(f"[sanity] 结果 -> {args.out}")
    sys.exit(0 if res["passed"] else 1)


if __name__ == "__main__":
    main()
