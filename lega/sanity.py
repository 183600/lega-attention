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
图”的断流问题：loss 可反传不代表每个参数都能收到梯度；首步全零梯度
（例如子模块被 detach 后又乘 0）同样视为断流，而不是只统计不判定。
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
from .preflight import (allow_protocol_violations,
                         data_protocol_preflight)
from .repro import build_run_manifest
from .utils import config_bool, save_json, set_seed, strict_int

# 训练路径 loss 与直接 logits CE 的一致性容差：1e-3 只用于吸收分块/
# 归约顺序的浮点噪声，真正的计算错误（错误 chunk 边界、错误归一化/
# ignore 处理）远大于此。
LOSS_CONSISTENCY_REL_TOL = 1e-3
LOSS_CONSISTENCY_ABS_TOL = 1e-3

# 冒烟模式：更小批次/更少步数，保持“能快速过拟合”的性质
SMOKE_OVERRIDES = {"steps": 60, "batch_size": 4, "ctx": 64, "log_every": 10}

# 随机初始化下交叉熵应接近 log(V)：偏差过大意味着 logits 尺度/初始化/标签
# 分布异常（这是排查标签错位与初始化错误的第一中间量）。容差取 1.0 nat 的
# 宽松值，只拦“数量级都不对”的问题，避免误伤正常随机初始化。
DEFAULT_INITIAL_LOSS_TOL = 1.0

DEFAULTS = {"steps": 300, "batch_size": 8, "ctx": 128, "lr": 3e-3,
            "warmup": 10, "grad_clip": 1.0, "log_every": 25,
            "loss_threshold": 0.05, "acc_threshold": 0.99,
            "initial_loss_tol": DEFAULT_INITIAL_LOSS_TOL}


@torch.inference_mode()
def batch_metrics(model, x: torch.Tensor, y: torch.Tensor,
                  max_chunk_tokens: int = 8192,
                  max_chunk_elems: int = 16_777_216) -> dict:
    """固定批的 loss / 下一 token 准确率 / 整条序列全对率（fp32、逐块计算）。

    直接消费 logits 计算指标（而非模型的训练路径），因此同时校验了
    lm_head 输出与标签的对齐关系。

    inference_mode：纯指标计算，返回的只有 Python 标量、无张量逃逸。

    计算逐块进行：max_chunk_tokens 限制单次前向的 token 数（rows*T），
    max_chunk_elems 限制单块 logits 与交叉熵逐 token 块的元素量
    （单条序列自身的 logits 无法再拆，rows 至少为 1）。
    """
    was_training = model.training
    model.eval()
    try:
        B, T = x.shape
        V = model.vocab_size
        max_elems = max(1, int(max_chunk_elems))
        rows = max(1, min(int(max_chunk_tokens) // max(1, T),
                          max_elems // max(1, T * V)))
        # 交叉熵/argmax 的逐 token 块：与训练头 _ChunkedLinearCE 的分块一致
        ce_tokens = max(256, min(4096, max_elems // max(1, V)))
        total_loss = 0.0
        n_tokens = 0
        correct = 0
        exact = 0
        for s in range(0, B, rows):
            xb, yb = x[s:s + rows], y[s:s + rows]
            logits = model(xb)["logits"]                # (b, T, V) fp32
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
        return {
            "loss": total_loss / max(1, n_tokens),
            "token_acc": correct / max(1, n_tokens),
            "exact_match": exact / max(1, B),
            "n_tokens": n_tokens,
        }
    finally:
        # 前向/损失计算抛异常时也必须恢复调用方模式：否则失败一次 sanity
        # 就会把训练模型静默留在 eval 模式，后续 dropout/正则化全部失效。
        if was_training:
            model.train()


def active_regularization(model) -> list:
    """列出模型里仍生效的 dropout 类正则化（sanity 要求全部关闭）。

    只关优化器 weight_decay 不够：dropout>0 时每次前向都注入随机噪声，固定
    批也无法压到接近 0 损失——“连小批量都过拟合不了”会被误判为梯度/损失
    有错误。这里自动检查 Dropout 模块与注意力 attn_dropout，作为显式失败项。
    """
    active = []
    for name, m in model.named_modules():
        p = getattr(m, "p", None)
        if isinstance(m, torch.nn.Dropout) and p and float(p) > 0:
            active.append(f"{name or type(m).__name__}: Dropout(p={float(p):g})")
        ad = getattr(m, "attn_dropout", None)
        if ad is not None and float(ad) > 0:
            active.append(f"{name or type(m).__name__}: attn_dropout={float(ad):g}")
    return active


def gradient_coverage(model) -> dict:
    """逐参数梯度覆盖检查（在第一次 backward 之后调用）。

    只看总梯度范数会漏掉“某个子模块没接进计算图”的问题：主体梯度正常时
    总范数依然 >0（如门控参数被 detach/漏传），只有逐参数检查才能发现。
    分类：
      * missing_grad —— grad is None：该参数与 loss 之间图断流（严重问题）；
      * nonfinite_grad —— grad 含 NaN/Inf：数值问题，继续训练只会污染权重；
      * zero_grad —— grad 全 0：疑似的图断流（如子模块被 detach 后又乘 0）。
        overfit 检查将其作为独立失败项；若确有“首步本应为 0”的参数，应在
        报告里人工复核，而不是让整条链路静默通过。
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


def _finite_float_arg(value, name: str) -> float:
    """有限浮点配置；bool/NaN/Inf/非数值显式拒绝（True 不能冒充 1.0）。"""
    if isinstance(value, bool):
        raise ValueError(f"{name} 必须是有限数值，收到 bool {value!r}")
    try:
        f = float(value)
    except (TypeError, ValueError, OverflowError) as e:
        raise ValueError(f"{name} 必须是有限数值，收到 {value!r}") from e
    if not math.isfinite(f):
        raise ValueError(f"{name} 必须是有限数值，收到 {value!r}")
    return f


def run_overfit_check(model: GPT, x: torch.Tensor, y: torch.Tensor, *,
                      steps: int, lr: float, device: str, seed: int,
                      warmup: int = 10, grad_clip: float = 1.0,
                      log_every: int = 25,
                      loss_threshold: float = 0.05,
                      acc_threshold: float = 0.99,
                      initial_loss_tol: float = DEFAULT_INITIAL_LOSS_TOL) -> dict:
    """在同一个固定批上训练 model 并返回可审计的检查结果。

    调用方必须已保证 model 的正则化已关闭（dropout=0/attn_dropout=0，
    优化器 weight_decay=0）；本函数只用固定批、无任何数据增强。

    除了“能否过拟合”，本函数还自动核对三条中间结果：
      * 首步训练路径 loss（模型内部 lm_head 分块 + 交叉熵）与 batch_metrics
        直接对完整 logits 算 F.cross_entropy 的结果一致——loss 计算错误
        （分块边界/归一化/ignore_index）未必妨碍过拟合，但数值会对不上；
      * 初始 loss ≈ log(V)：随机初始化 + 未见数据下交叉熵应接近均匀分布
        的对数值，偏差超过 initial_loss_tol（默认 1.0 nat，只拦数量级错误；
        负值表示显式禁用该检查）说明初始化/logits 尺度/标签分布存在严重问题；
      * 标签映射 y == x 右移一位——在数据消费点再次硬校验。
      * 正则化已关闭（dropout/attn_dropout 全部为 0）——开着 dropout 的
        “过拟合失败”是配置错误而非代码错误，自动检查避免误判。
    """
    # log_every=0 会在训练中途 ``% 0`` 崩溃，bool/小数则会被 int() 截断成
    # 与调用方声明不同的日志频率；在建模/开跑前统一拒绝。
    log_every = strict_int(log_every, "log_every")
    if log_every <= 0:
        raise ValueError(f"log_every 必须是正整数，收到 {log_every!r}")
    steps = strict_int(steps, "steps")
    if steps < 0:
        raise ValueError(f"steps 必须是非负整数，收到 {steps!r}")
    warmup = strict_int(warmup, "warmup")
    if warmup < 0:
        raise ValueError(f"warmup 必须是非负整数，收到 {warmup!r}")
    lr = _finite_float_arg(lr, "lr")
    if lr < 0:
        raise ValueError(f"lr 必须 >= 0，收到 {lr!r}")
    grad_clip = _finite_float_arg(grad_clip, "grad_clip")
    if grad_clip < 0:
        raise ValueError(f"grad_clip 必须 >= 0，收到 {grad_clip!r}")
    # 阈值/容差也必须拒绝 bool/NaN/Inf：True 会被 float() 当成 1.0，
    # 从而伪造一个看似合法的过拟合/准确率门槛。
    loss_threshold = _finite_float_arg(loss_threshold, "loss_threshold")
    acc_threshold = _finite_float_arg(acc_threshold, "acc_threshold")
    initial_loss_tol = _finite_float_arg(initial_loss_tol, "initial_loss_tol")
    set_seed(seed)
    # 标签映射硬校验：固定批必须在消费点满足 next-token 移位关系
    shift_ok = (x.dim() == 2 and x.shape == y.shape
                and bool(torch.equal(x[:, 1:], y[:, :-1])))
    active_reg = active_regularization(model)
    # 关闭 TF32：sanity 追求数值上确定地“压到 0”
    prev_tf32_mm = torch.backends.cuda.matmul.allow_tf32
    prev_tf32_cudnn = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    grad_cov = None
    try:
        model.train()
        initial = batch_metrics(model, x, y)
        fused = str(device).startswith("cuda")
        opt = torch.optim.AdamW(model.parameters(), lr=float(lr),
                                weight_decay=0.0, betas=(0.9, 0.95),
                                fused=fused, foreach=not fused)
        history = []
        first_grad_norm = None
        first_train_loss = None
        nan_seen = False
        loss_no_grad = False
        trainable_params = [p for p in model.parameters() if p.requires_grad]
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
            gn = torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)
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

    log_vocab = math.log(model.vocab_size)
    initial_loss_gap = float(initial["loss"]) - log_vocab
    # initial_loss_tol < 0 表示显式禁用该检查（仍记录偏差供人工复核）
    initial_loss_ok = (float(initial_loss_tol) < 0) or (
        math.isfinite(initial_loss_gap)
        and abs(initial_loss_gap) <= float(initial_loss_tol))
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
        # 全零梯度同样说明该参数与 loss 之间没有有效信号（例如子模块被
        # detach 后再乘 0）：总范数可能仍然正常，只有逐参数检查能发现。
        "no_zero_grads_first_step": (grad_cov is not None
                                     and not grad_cov["zero_grad"]),
        "grads_finite": (grad_cov is not None
                         and not grad_cov["nonfinite_grad"]),
        # 中间结果核对：训练路径 loss 与直接 logits CE 数值一致（损失计算正确性）
        "train_loss_matches_direct_logits": losses_consistent(
            first_train_loss, initial["loss"]),
        # 中间结果核对：消费点的标签映射是 next-token 移位（而非错位/换序）
        "next_token_shift_consistent": bool(shift_ok),
        # 中间结果核对：初始 loss 接近 log(V)（初始化/logits 尺度/标签分布正常）
        "initial_loss_near_log_vocab": initial_loss_ok,
        # 配置前提：正则化必须关闭，否则“过拟合失败”无法归因于代码错误
        "regularization_disabled": not active_reg,
    }
    passed = all(checks.values())
    return {
        "check": "overfit_small_batch",
        "passed": bool(passed),
        "checks": checks,
        "thresholds": {"loss": float(loss_threshold),
                       "token_acc": float(acc_threshold),
                       "initial_loss_tol": float(initial_loss_tol)},
        "initial": initial,
        "final": final,
        "log_vocab": log_vocab,
        "initial_loss_minus_log_vocab": initial_loss_gap,
        "active_regularization": active_reg,
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
                   "logits CE（首步）、y=x 右移一位与初始 loss≈log(V)，"
                   "其余中间结果（梯度范数>0、逐参数梯度覆盖含全零梯度、"
                   "逐层统计）请结合本 JSON 人工复核，并使用真实数据小规模"
                   "预跑。"),
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
    p.add_argument("--initial_loss_tol", type=float, default=None,
                   help="初始 loss 与 log(V) 的允许偏差（nat，默认 1.0）")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--allow_data_protocol_violations", action="store_true",
                   help="即使 data_report 标记泄露也继续 sanity（仅用于复现"
                        "历史实验；结果不得用于正式结论）")
    args = p.parse_args()

    data_preflight = data_protocol_preflight(
        args.data_dir,
        allow_override=allow_protocol_violations(
            args.allow_data_protocol_violations))
    if not data_preflight["report_present"]:
        print("[sanity] 警告：数据目录缺少 data_report.json，"
              "无法执行训练前防泄漏闸门（旧数据或未走统一入口）")
    elif data_preflight["critical_violations"]:
        print(f"[sanity] ⚠ 数据协议自检发现 "
              f"{len(data_preflight['critical_violations'])} 个严重问题，"
              "但已显式放行（结果不得用于正式结论）")

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    opt = dict(DEFAULTS)
    sanity_cfg = cfg.get("sanity") or {}
    opt["loss_threshold"] = _finite_float_arg(
        sanity_cfg.get("loss_threshold", DEFAULTS["loss_threshold"]),
        "sanity.loss_threshold")
    opt["acc_threshold"] = _finite_float_arg(
        sanity_cfg.get("acc_threshold", DEFAULTS["acc_threshold"]),
        "sanity.acc_threshold")
    opt["initial_loss_tol"] = _finite_float_arg(
        sanity_cfg.get("initial_loss_tol", DEFAULTS["initial_loss_tol"]),
        "sanity.initial_loss_tol")
    if args.smoke:
        opt.update(SMOKE_OVERRIDES)
    for name in ("steps", "batch_size", "ctx", "lr", "seed",
                 "loss_threshold", "acc_threshold", "initial_loss_tol"):
        val = getattr(args, name)
        if val is not None:
            opt[name] = val
    # 种子必须是精确整数："seed": true/1.5 若被 int() 截断成 1，sanity 会用
    # 与配置声明不同的种子跑固定批过拟合，而结果 JSON 记录的是截断值。
    seed = strict_int(
        opt.get("seed") if opt.get("seed") is not None
        else cfg.get("seed", 1337), "seed")
    device = (args.device if args.device != "auto"
              else ("cuda" if torch.cuda.is_available() else "cpu"))
    set_seed(seed)

    def _cfg_positive_int(key, default=None):
        value = cfg.get(key, default)
        value = strict_int(value, f"config.{key}")
        if value <= 0:
            raise ValueError(f"config.{key} 必须是正整数，收到 {value}")
        return value

    meta = load_meta(args.data_dir)
    train = load_bin(args.data_dir, "train")
    block_size = _cfg_positive_int("block_size")
    train_ctx = min(_cfg_positive_int("train_ctx", block_size), block_size)
    ctx = min(strict_int(opt["ctx"], "ctx"), train_ctx)
    if ctx <= 0:
        raise ValueError(f"ctx 必须是正整数，收到 {ctx}")
    batch_size = strict_int(opt["batch_size"], "batch_size")
    if batch_size <= 0:
        raise ValueError(f"batch_size 必须是正整数，收到 {batch_size}")
    vocab_size = strict_int(meta.get("vocab_size"), "meta.vocab_size")
    if vocab_size <= 0:
        raise ValueError(f"meta.vocab_size 必须是正整数，收到 {vocab_size}")

    # 显式关闭正则化：dropout/attn_dropout=0；优化器 weight_decay=0（见上）
    model = GPT(
        vocab_size=vocab_size, block_size=block_size,
        d_model=_cfg_positive_int("d_model"),
        n_layers=_cfg_positive_int("n_layers"),
        n_heads=_cfg_positive_int("n_heads"),
        ffn_mult=_cfg_positive_int("ffn_mult", 4),
        dropout=0.0, attn_dropout=0.0, variant=args.variant,
        gate_bias_init=float(cfg.get("gate_bias_init", 2.0)),
        use_sdpa_standard=config_bool(
            cfg.get("use_sdpa_standard", True), "use_sdpa_standard"),
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
        acc_threshold=float(opt["acc_threshold"]),
        initial_loss_tol=float(opt["initial_loss_tol"]))
    res.update({
        "variant": args.variant,
        "variant_cn": VARIANT_CN[args.variant],
        "seed": seed,
        "batch_size": batch_size,
        "ctx": ctx,
        "device": device,
        "config_file": os.path.abspath(args.config),
        "data_dir": os.path.abspath(args.data_dir),
        "data_protocol_preflight": data_preflight,
        "manifest": build_run_manifest(
            tag="sanity", variant=args.variant, seed=seed,
            config_path=args.config, data_dir=args.data_dir, device=device),
    })
    last = res["history"][-1]["loss"] if res["history"] else float("nan")
    print(f"[sanity] {'PASS' if res['passed'] else 'FAIL'}: "
          f"loss {res['initial']['loss']:.4f} -> {res['final']['loss']:.4f} "
          f"(曲线末值 {last:.4f}, logV={res['log_vocab']:.3f}, "
          f"初始偏差 {res['initial_loss_minus_log_vocab']:+.3f}/"
          f"±{res['thresholds']['initial_loss_tol']:.2f}), "
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
