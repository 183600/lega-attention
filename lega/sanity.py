"""过拟合小批量 sanity check —— 全量训练前的第一道代码正确性闸门。

原理：抽取一个固定小批次（默认 batch_size 条、ctx 个 token），关闭全部
正则化与数据扰动后反复优化同一批数据。若前向/反向/损失/标签映射任一环节
存在严重问题（梯度断流、loss 计算错误、label 错位等），训练损失不可能降到
接近 0、下一 token 准确率也不可能接近 100%；反之，能过拟合只能说明
“这条最小学习链路是通的”。

通过是**必要不充分**条件：数据增强、特征对齐、标签映射等问题仍可能
漏网（本检查不含增强与真实数据顺序），需配合中间结果检查（首步 loss ≈
log(V)、梯度范数 > 0、**逐参数梯度覆盖**等），并保留本 JSON 作为实验记录。
逐参数梯度覆盖能定位“总梯度范数正常、但某个子模块（如门控）没接进计算
图”的断流问题：loss 可反传不代表每个参数都能收到梯度。
此外本检查还做两条中间结果自动核对：
  * 训练路径 loss（lm_head 分块 + 交叉熵）必须与“直接对完整 logits 算
    F.cross_entropy”一致——分块/normalization/ignore_index 写错时，
    loss 仍可反传、仍可能过拟合，但两条路径的数值会对不上；
  * 固定批必须满足 y = x 右移一位——标签映射在消费点再次硬校验。

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

# 训练路径 loss 与直接 logits CE 的一致性容差：实测同一权重/同一批数据下
# 两条路径差异 ≤1e-7（相对），1e-3 只用于吸收分块/归约顺序的浮点噪声，
# 真正的计算错误（错误 chunk 边界、错误归一化/ignore 处理）远大于此。
LOSS_CONSISTENCY_REL_TOL = 1e-3
LOSS_CONSISTENCY_ABS_TOL = 1e-3

# 冒烟模式：更小批次/更少步数，保持“能快速过拟合”的性质
SMOKE_OVERRIDES = {"steps": 60, "batch_size": 4, "ctx": 64, "log_every": 10}

DEFAULTS = {"steps": 300, "batch_size": 8, "ctx": 128, "lr": 3e-3,
            "warmup": 10, "grad_clip": 1.0, "log_every": 25,
            "loss_threshold": 0.05, "acc_threshold": 0.99}


@torch.no_grad()
def batch_metrics(model, x: torch.Tensor, y: torch.Tensor,
                  max_chunk_tokens: int = 8192,
                  max_chunk_elems: int = 16_777_216) -> dict:
    """固定批的 loss / 下一 token 准确率 / 整条序列全对率（fp32、逐块计算）。

    直接消费 logits 计算指标（而非模型的训练路径），因此同时校验了
    lm_head 输出与标签的对齐关系。

    内存约束（显存/内存峰值与词表规模解耦）：
      * max_chunk_tokens 限制单次前向的 token 数（rows*T）；
      * max_chunk_elems 限制单次 logits 张量 (rows*T*V) 与交叉熵内部
        log_softmax 缓冲 (ce_tokens*V) 的元素量——大词表下 F.cross_entropy
        会在整块 (rows*T, V) 上再分配一份同尺寸缓冲，必须按元素量二次分块，
        否则词表一大峰值内存就会数倍于 logits 本身。
        （单条序列自身的 logits 无法再拆，rows 至少为 1。）
    """
    was_training = model.training
    model.eval()
    B, T = x.shape
    V = model.vocab_size
    budget = max(1, int(max_chunk_elems))
    rows = max(1, min(int(max_chunk_tokens) // max(1, T),
                      budget // max(1, T * V)))
    # 交叉熵/argmax 的逐 token 块：与训练头 _ChunkedLinearCE 的分块预算同口径
    ce_tokens = max(256, min(4096, budget // max(1, V)))
    total_loss = 0.0
    n_tokens = 0
    correct = 0
    exact = 0
    for s in range(0, B, rows):
        xb, yb = x[s:s + rows], y[s:s + rows]
        logits = model(xb)["logits"]                    # (b, T, V) fp32
        lf = logits.reshape(-1, V)
        tgt = yb.reshape(-1)
        pred = torch.empty(lf.size(0), dtype=torch.int64, device=lf.device)
        for t0 in range(0, lf.size(0), ce_tokens):
            lc = lf[t0:t0 + ce_tokens]
            tc = tgt[t0:t0 + ce_tokens]
            total_loss += float(F.cross_entropy(lc, tc, reduction="sum"))
            pred[t0:t0 + ce_tokens] = lc.argmax(dim=-1)
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


def gradient_coverage(model) -> dict:
    """逐参数梯度覆盖检查（在第一次 backward 之后调用）。

    只看总梯度范数会漏掉“某个子模块没接进计算图”的问题：主体梯度正常时
    总范数依然 >0（如门控参数被 detach/漏传），只有逐参数检查才能发现。
    分类：
      * missing_grad —— grad is None：该参数与 loss 之间图断流（严重故障）；
      * nonfinite_grad —— grad 含 NaN/Inf：数值故障，继续训练只会污染权重；
      * zero_grad —— grad 全 0：给出列表供人工判断（首步恰为 0 属于可疑但
        不一定错误，例如无输入信号的偏置），不单独判失败。
    """
    missing, nonfinite, zero = [], [], []
    n_trainable = 0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        n_trainable += 1
        g = p.grad
        if g is None:
            missing.append(name)
        elif not bool(torch.isfinite(g).all()):
            nonfinite.append(name)
        elif float(g.detach().abs().max()) == 0.0:
            zero.append(name)
    return {"n_trainable": n_trainable, "missing_grad": missing,
            "nonfinite_grad": nonfinite, "zero_grad": zero}


def losses_consistent(a, b, rel_tol: float = LOSS_CONSISTENCY_REL_TOL,
                      abs_tol: float = LOSS_CONSISTENCY_ABS_TOL) -> bool:
    """训练路径 loss 与直接 logits 交叉熵的数值一致性判定（None/非有限即失败）。"""
    if a is None or b is None:
        return False
    fa, fb = float(a), float(b)
    return (math.isfinite(fa) and math.isfinite(fb)
            and math.isclose(fa, fb, rel_tol=float(rel_tol),
                             abs_tol=float(abs_tol)))


def run_overfit_check(model: GPT, x: torch.Tensor, y: torch.Tensor, *,
                      steps: int, lr: float, device: str, seed: int,
                      warmup: int = 10, grad_clip: float = 1.0,
                      log_every: int = 25,
                      loss_threshold: float = 0.05,
                      acc_threshold: float = 0.99) -> dict:
    """在同一个固定批上训练 model 并返回可审计的检查结果。

    调用方必须已保证 model 的正则化已关闭（dropout=0/attn_dropout=0，
    优化器 weight_decay=0）；本函数只用固定批、无任何数据增强。

    除了“能否过拟合”，本函数还自动核对两条中间结果：
      * 首步训练路径 loss（模型内部 lm_head 分块 + 交叉熵）与 batch_metrics
        直接对完整 logits 算 F.cross_entropy 的结果一致——loss 计算错误
        （分块边界/归一化/ignore_index）未必妨碍过拟合，但数值会对不上；
      * 标签映射 y == x 右移一位——在数据消费点再次硬校验。
    """
    set_seed(seed)
    # 标签映射硬校验：固定批必须在消费点满足 next-token 移位关系
    shift_ok = (x.dim() == 2 and x.shape == y.shape
                and bool(torch.equal(x[:, 1:], y[:, :-1])))
    # 关闭 TF32：sanity 追求数值上确定地“压到 0”
    prev_tf32_mm = torch.backends.cuda.matmul.allow_tf32
    prev_tf32_cudnn = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    grad_cov = None
    try:
        model.train()
        initial = batch_metrics(model, x, y)
        opt = torch.optim.AdamW(model.parameters(), lr=float(lr),
                                weight_decay=0.0, betas=(0.9, 0.95))
        history = []
        first_grad_norm = None
        first_train_loss = None
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
            # 首步训练路径 loss：与 initial（直接 logits CE）同为“优化前的
            # 同一模型、同一批数据”，用于核对两条损失计算路径是否一致。
            if step == 0:
                first_train_loss = float(loss.detach())
            # 梯度断流的典型表现：loss 不挂在 autograd 图上（如误用 detach、
            # 冻结了全部参数）。直接等价于“梯度没有反向传播”，记为失败而非崩溃。
            if not loss.requires_grad:
                loss_no_grad = True
                break
            loss.backward()
            # 首步逐参数覆盖：总范数正常但个别子模块没拿到梯度（图断流）时，
            # 这里能定位到具体参数名（见 gradient_coverage 文档）。
            if step == 0:
                grad_cov = gradient_coverage(model)
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
        "all_params_get_grads": (grad_cov is not None
                                 and not grad_cov["missing_grad"]),
        "grads_finite": (grad_cov is not None
                         and not grad_cov["nonfinite_grad"]),
        # 中间结果核对：训练路径 loss 与直接 logits CE 数值一致（损失计算正确性）
        "train_loss_matches_direct_logits": losses_consistent(
            first_train_loss, initial["loss"]),
        # 中间结果核对：消费点的标签映射是 next-token 移位（而非错位/换序）
        "next_token_shift_consistent": bool(shift_ok),
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
        "first_train_loss": first_train_loss,
        "loss_consistency": {
            "train_path": first_train_loss,
            "direct_logits": initial["loss"],
            "rel_tol": LOSS_CONSISTENCY_REL_TOL,
            "abs_tol": LOSS_CONSISTENCY_ABS_TOL,
        },
        "grad_coverage": grad_cov,
        "steps": int(steps),
        "lr": float(lr),
        "history": history,
        "caveat": ("通过为必要非充分条件：数据增强/特征对齐/标签映射等问题 "
                   "仍可能漏网；本检查已自动核对训练路径 loss 与直接 "
                   "logits CE（首步）以及 y=x 右移一位，其余中间结果"
                   "（首步 loss≈log(V)、梯度范数>0、逐参数梯度覆盖、逐层"
                   "统计）请结合本 JSON 人工复核，并使用真实数据小规模预跑。"),
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
              "若无法过拟合小批量，说明梯度/损失/标签存在严重问题，"
              "请先排查再开全量训练。")
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        save_json(res, args.out)
        print(f"[sanity] 结果 -> {args.out}")
    sys.exit(0 if res["passed"] else 1)


if __name__ == "__main__":
    main()
