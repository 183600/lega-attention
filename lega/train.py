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
import statistics
import time

# 须在任何 CUDA 初始化之前生效。
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch

from .attention import VARIANTS, VARIANT_CN
from .data import load_bin, load_meta, get_train_batch, get_val_batch, get_val_windows
from .model import GPT, configure_optimizer, count_params, gate_extra_params, save_checkpoint
from .repro import build_run_manifest
from .utils import set_seed, now_str, save_json, enable_tf32


# ---------------------------------------------------------------- NaN 排查路径

def _nonfinite_stats(obj) -> dict:
    """递归统计张量结构里的 NaN/Inf 元素数（tuple/list/dict；非张量忽略）。

    模块钩子的输入/输出可能是张量、张量元组/列表或 dict（如 attention 返回
    past_kv），只对浮点/复数张量统计——整数索引张量不存在 NaN。
    """
    stats = {"nan": 0, "inf": 0, "elems": 0, "nonfinite": 0}

    def visit(o):
        if torch.is_tensor(o):
            if not (torch.is_floating_point(o) or torch.is_complex(o)):
                return
            stats["elems"] += int(o.numel())
            stats["nan"] += int(torch.isnan(o).sum())
            stats["inf"] += int(torch.isinf(o).sum())
        elif isinstance(o, (tuple, list)):
            for t in o:
                visit(t)
        elif isinstance(o, dict):
            for t in o.values():
                visit(t)

    visit(obj)
    stats["nonfinite"] = stats["nan"] + stats["inf"]
    return stats


def _numerical_hint(module_name: str) -> str:
    """按模块在计算图中的位置给出 log(0)/除零/softmax 溢出的排查指向。"""
    name = module_name or ""
    if name.endswith(".attn"):
        return ("注意力核心：Q/K/V 投影走函数式 F.linear（不经过子模块钩子，"
                "溯源到本模块即可能来自投影权重）；复核 max_p→0 时的 log(0)"
                "（已 clamp_min(1e-30)）、score 缩放、除零与 softmax 溢出")
    if "gate_" in name:
        return "门控参数：复核集中度项 c 与长度项 log n 的输入范围"
    if ".ln" in name or name == "ln_f" or name.endswith("ln1") or name.endswith("ln2"):
        return "LayerNorm/归一化：检查其输入是否已被上游污染（方差/半径除法）"
    if name == "lm_head" or name in ("wte", "wpe"):
        return "输出/嵌入路径：检查 logits 范围与权重是否已被污染"
    return "通用算子：按该模块在计算图中的位置定位 log/除零/溢出路径"


def audit_activations(model, pending_batches, device: str,
                      amp_dtype=None) -> dict:
    """用前向钩子定位首个“输入有限、输出非有限”的模块（NaN 排查第 2 步的主动检查）。

    仅靠非有限参数列表只能发现“哪些权重坏了”，无法区分 NaN 是数据/权重传播
    而来，还是某个算子在**有限输入**上产生（log(0)、除零、softmax 溢出、
    fp16/bf16 上溢）。本函数对失败批次做一次带钩子的前向，按实际执行顺序记录
    每个模块输出/输入的 NaN/Inf 情况：

      * origin          第一个输入有限但输出非有限的模块 —— 最可疑的数值故障点，
                        附该模块位置的排查指向（softmax/log/归一化/投影）；
      * first_nonfinite 第一个输出非有限的模块：若与 origin 不同，说明 NaN 来自
                        更早的算子（如 attention 内部的函数式 softmax/log）；
      * records         非有限激活沿计算图的传播链（按执行顺序，截断到上限）。

    前向只消费 x 不消费 y（标签越界不会妨碍激活审计），异常被捕获并写入
    forward_error 而不是让排查本身崩溃。
    """
    batch = next((b for b in (pending_batches or []) if b is not None), None)
    if batch is None:
        return {"checked": False, "n_modules_checked": 0, "records": [],
                "origin": None, "first_nonfinite": None, "forward_error": None,
                "findings": ["无可用批次（NaN 出现在记录成功批次之前），跳过激活定位"]}
    x = batch[0]
    records, handles = [], []
    cap = 60

    def hook(name):
        def _h(module, inputs, output):
            out_st = _nonfinite_stats(output)
            if not out_st["nonfinite"]:
                return
            if len(records) >= cap:
                return
            in_st = _nonfinite_stats(inputs)
            records.append({
                "seq": len(records), "module": name,
                "output_nan": out_st["nan"], "output_inf": out_st["inf"],
                "input_nonfinite": in_st["nonfinite"],
            })
        return _h

    for name, mod in model.named_modules():
        if name:
            handles.append(mod.register_forward_hook(hook(name)))
    was_training = model.training
    forward_error = None
    try:
        model.eval()
        use_cuda_amp = amp_dtype is not None and str(device).startswith("cuda")
        amp_ctx = (torch.autocast("cuda", dtype=amp_dtype) if use_cuda_amp
                   else contextlib.nullcontext())
        with torch.no_grad(), amp_ctx:
            # last_logits_only：只对最后位置过 lm_head；注意力/门控（NaN 最
            # 常见的产生位置）仍对整段前缀执行，逐模块钩子照常触发。
            model(x, last_logits_only=True)
    except Exception as e:  # 排查过程本身绝不能因前向失败而中断
        forward_error = f"{type(e).__name__}: {e}"
    finally:
        for h in handles:
            h.remove()
        model.train(was_training)

    origin = next((r for r in records if r["input_nonfinite"] == 0), None)
    if origin is not None:
        origin = dict(origin, hint=_numerical_hint(origin["module"]))
    first = records[0] if records else None
    findings = []
    if forward_error:
        findings.append(f"激活审计前向失败（{forward_error}），改用参数/数据检查定位")
    if origin is not None:
        findings.append(
            f"首个非有限激活源: {origin['module']}（输入有限，输出 NaN="
            f"{origin['output_nan']}/Inf={origin['output_inf']}）→ {origin['hint']}")
    elif first is not None:
        findings.append(
            f"第一个非有限模块 {first['module']} 的输入已含非有限值"
            f"（NaN/Inf={first['input_nonfinite']}）：NaN 来自更早的函数式算子或"
            "输入数据，按模块名检查该模块内部的 log/除零/softmax 路径")
    elif not forward_error:
        findings.append(
            "所有模块输出均有限——NaN 产生在损失计算/评测聚合环节"
            "（如分块交叉熵的目标越界），不是前向算子溢出")
    return {
        # checked = 审计前向成功执行；records/origin 为空表示前向算子全部有限
        "checked": forward_error is None,
        "n_modules_checked": len(handles),
        "records": records[:cap],
        "truncated": len(records) >= cap,
        "origin": origin,
        "first_nonfinite": first,
        "forward_error": forward_error,
        "findings": findings,
    }


def run_nan_checklist(model, pending_batches, device: str, step: int, where: str,
                      loss_value, lr: float, grad_norm_history, amp_dtype,
                      vocab_size: int, consumed_batches=None) -> dict:
    """训练出现非有限值时的系统化排查：数据 → 代码 → 混合精度 → 梯度/学习率。

    排查顺序固定（先便宜后昂贵、先外部后内部），每步产出可核对的证据，
    避免"看到 NaN 就乱调参"。返回可直接写盘的报告 dict。

    consumed_batches：产出该损失的**失败步批次**（训练循环已预取并覆盖了
    它们，调用方必须显式传入）。数据检查与激活定位都优先检查这些批次——
    只看待训练批次会把"罪魁数据"漏掉。
    """
    report = {
        "time": now_str(), "step": int(step), "where": str(where),
        "loss_value": (None if loss_value is None else float(loss_value)),
        "device": str(device), "lr_at_failure": float(lr),
        "amp_dtype": (str(amp_dtype) if amp_dtype else None),
    }

    # ---- 1. 数据：输入流是否含非法值（NaN/Inf/越界 id）；先查失败步批次
    labeled = [(f"失败步批{i}", b) for i, b in enumerate(consumed_batches or [])
               if b is not None] \
        + [(f"待训练批{i}", b) for i, b in enumerate(pending_batches or [])
           if b is not None]
    data_findings = []
    n_checked = 0
    for label, batch in labeled:
        x, y = batch
        for name, t in (("x", x), ("y", y)):
            n_checked += 1
            if not torch.is_floating_point(t) and not torch.is_complex(t):
                lo, hi = int(t.min()), int(t.max())
                bad = (lo < 0) or (hi >= vocab_size)
                data_findings.append(
                    f"{label} {name}: 整数 id 范围 [{lo}, {hi}]，"
                    + ("**越界 vocab**" if bad else "合法"))
            else:
                n_nan = int(torch.isnan(t).sum())
                n_inf = int(torch.isinf(t).sum())
                data_findings.append(
                    f"{label} {name}: 浮点输入 NaN={n_nan}/Inf={n_inf}"
                    + ("（**数据侧混入非法值**）" if n_nan or n_inf else ""))
    report["checklist_data"] = {
        "findings": data_findings,
        "n_tensors_checked": n_checked,
        "n_consumed_batches": sum(1 for b in (consumed_batches or [])
                                  if b is not None),
        "note": ("失败步批=产出该损失的批次（优先检查）；token 流为 uint16 整数且 "
                 "prepare_bins 入库时已拒绝 NaN/Inf/越界 id（见 data_report.json "
                 "的 token 流硬校验）；浮点输入出现 NaN/Inf 才是数据侧问题"),
    }

    # ---- 2. 代码：定位已含 NaN/Inf 的参数张量（把锅从"玄学"收敛到具体层），
    #      再用前向钩子定位首个“输入有限、输出非有限”的算子（主动检查
    #      log(0)/除零/softmax 溢出，而不是只打印已知防护清单）
    bad_params = []
    for n, p in model.named_parameters():
        if not torch.isfinite(p).all():
            n_nan = int(torch.isnan(p).sum())
            n_inf = int(torch.isinf(p).sum())
            bad_params.append(f"{n} (NaN={n_nan}, Inf={n_inf}, shape={list(p.shape)})")
        if len(bad_params) >= 8:
            break
    audit_batches = [b for b in (consumed_batches or []) if b is not None] \
        + [b for b in (pending_batches or []) if b is not None]
    activation_audit = audit_activations(model, audit_batches, device,
                                         amp_dtype=amp_dtype)
    report["checklist_params"] = {
        "nonfinite_tensors": bad_params,
        "note": ("非有限值出现在哪些层决定了排查方向：门控/输出侧 → 检查 log(0)、"
                 "除零与 softmax 数值溢出路径；注意力核心 → 检查 score 缩放；"
                 "本仓库已知防护：门控 c 项 log(max_p) 有 clamp_min(1e-30)，"
                 "交叉熵经 log_softmax（数值稳定），除法处均有 clamp_min 兜底"),
    }
    report["checklist_activations"] = activation_audit

    # ---- 3. 混合精度：当前 autocast 精度与下溢/溢出风险
    if amp_dtype is None:
        prec_note = "未启用 autocast（fp32/TF32 路径）：可排除混合精度下溢因素"
    elif "bfloat16" in str(amp_dtype):
        prec_note = ("bf16 autocast：动态范围与 fp32 相同，下溢风险低；"
                     "极大数据仍可能溢出，可对可疑层加 .float() 升精度复测")
    else:
        prec_note = ("fp16 autocast：梯度/激活下溢是常见 NaN 来源，"
                     "应使用 GradScaler 损失缩放，或改用 bf16 复测")
    report["checklist_precision"] = {"amp_dtype": (str(amp_dtype) if amp_dtype
                                                   else None), "note": prec_note}

    # ---- 4. 梯度爆炸 / 学习率：pre-clip 梯度范数历史是否出现尖峰/Inf
    hist = [g for g in (grad_norm_history or []) if g is not None]
    opt_findings = []
    # 梯度范数触发的排查（where=grad_norm）时，触发值本身就是非有限的当前范数；
    # 历史只保留有限值（避免 Infinity 写进 JSON），因此不能只靠 hist[-1] 判定，
    # 必须用触发值显式报告"已爆炸"（否则专家信号会退化为"未见尖峰"）。
    trigger_nonfinite = (where == "grad_norm" and loss_value is not None
                         and not math.isfinite(float(loss_value)))
    if trigger_nonfinite:
        opt_findings.append(
            f"本次 pre-clip 梯度范数 {float(loss_value):.3e} 非有限 → 梯度已爆炸")
    if hist:
        finite = [g for g in hist if math.isfinite(g)]
        last = hist[-1]
        if finite and len(finite) >= 3:
            med = statistics.median(finite[:-1]) or 1e-12
            if last > 10 * med:
                spike_note = f"（尖峰 ×{last / med:.0f}，**疑似梯度爆炸**）"
            elif trigger_nonfinite:
                spike_note = "（历史尾部未见尖峰，但本次已非有限）"
            else:
                spike_note = "（未见爆炸性尖峰）"
            opt_findings.append(
                f"最近 pre-clip 梯度范数 {last:.3e}，此前中位数 {med:.3e}"
                + spike_note)
        if not math.isfinite(last):
            opt_findings.append("最近一次 pre-clip 梯度范数非有限 → 梯度已爆炸")
    else:
        opt_findings.append("无梯度范数历史（NaN 出现过早，尚未记录到 log 点）")
    opt_findings.append(
        f"失败时学习率 {lr:.2e}（warmup 后峰值 {lr:.2e} 量级；"
        "过大学习率会在 warmup 结束前后率先炸出 NaN）")
    report["checklist_optimization"] = {"findings": opt_findings,
                                        "grad_norm_history_tail": hist[-20:]}

    # ---- 建议动作：按固定排查顺序给出
    actions = [
        "1) 数据：核对 data_report.json 的 token 流硬校验与上方批次范围检查；"
        "浮点输入出现 NaN/Inf 才需回查数据管道",
        "2) 代码：先看 checklist_activations 的 origin（首个“输入有限、输出非有限”"
        "的模块）再对照非有限参数列表，复核该层 log(0)/除零/softmax 溢出路径"
        "（本仓库门控 c 项已 clamp_min(1e-30)）",
        f"3) 精度：{prec_note}",
        "4) 优化：降低学习率（如减半）、延长 warmup、收紧 grad_clip（如 0.5），"
        "并观察梯度范数是否再出尖峰",
    ]
    report["suggested_actions"] = actions
    return report


def _abort_on_nan(report: dict, out_dir: str) -> None:
    """落盘 NaN 排查报告并中止训练：NaN 之后的步数只会污染曲线。"""
    path = os.path.join(out_dir, "nan_report.json")
    save_json(report, path)
    print(f"[train] ❌ 检测到非有限损失（step {report['step']}，{report['where']}），"
          f"排查报告 -> {path}")
    for key, title in (("checklist_data", "数据"), ("checklist_params", "参数定位"),
                       ("checklist_activations", "激活定位"),
                       ("checklist_precision", "精度"), ("checklist_optimization", "梯度/学习率")):
        sec = report.get(key) or {}
        lines = sec.get("findings") or sec.get("nonfinite_tensors") or []
        print(f"[train] 排查-{title}: " + ("；".join(lines[:3]) if lines
                                            else sec.get("note", "无异常")))
    print("[train] 建议动作（按顺序）:")
    for a in report["suggested_actions"]:
        print(f"[train]   {a}")
    raise SystemExit(
        f"训练在 step {report['step']} 出现 {report['where']} 非有限值，已中止；"
        f"按 {path} 中的排查路径定位后重跑")

SMOKE_OVERRIDES = {
    "steps": 20, "eval_every": 10, "eval_iters": 2, "log_every": 5,
    "warmup": 5, "batch_size": 8, "train_ctx": 128,
}


def _val_eval_indices(val_len: int, ctx: int, iters: int):
    """均匀覆盖整个验证集的不重复窗口序号。

    规则：
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
    # 设备端累积，循环结束后统一读取。
    # 窗口等长，权重即窗口数，用 Python int 就地累计。
    loss_acc = torch.zeros((), device=device)
    gate_acc = torch.zeros((), device=device)
    total_w = 0
    gate_w_total = 0
    amp_ctx = (torch.autocast("cuda", dtype=torch.bfloat16)
               if (amp and device.startswith("cuda")) else contextlib.nullcontext())
    # inference_mode：纯前向评测；返回的统计只经 .item() 消费。
    with torch.inference_mode():
        # 预取流水线：在当前批前向之前发起下一窗口批的 memmap 收集与 H2D
        # 提交；_stage_to_device 的乒乓暂存（双槽 + event 保护）保证两批
        # 在途互不干扰。
        n_batches = (len(indices) + batch_windows - 1) // batch_windows
        xy = get_val_windows(val, ctx, indices[:batch_windows], device)
        for bi, s in enumerate(range(0, len(indices), batch_windows)):
            chunk = indices[s:s + batch_windows]
            x, y = xy
            if bi + 1 < n_batches:
                xy = get_val_windows(val, ctx,
                                     indices[s + batch_windows:s + 2 * batch_windows],
                                     device)
            with amp_ctx:
                # 验证评测只消费门控均值 g，使用轻量统计
                out = model(x, targets=y, need_stats=collect_gate, stats_argmax=False)
            nb = len(chunk)     # 窗口等长，按窗口数加权即 token 加权
            # add_(alpha=nb) 直接按批大小加权累加（loss 前向/门控闭式求和
            # 本就以 fp32 返回）。
            loss_acc.add_(out["loss"].detach(), alpha=float(nb))
            total_w += nb
            if collect_gate and out["gate_stats"]:
                # 在设备端先 stack 成 (n_layers,) 再求均值
                gs = [s_["g"].detach().float().mean()
                      for s_ in out["gate_stats"] if s_.get("g") is not None]
                if gs:
                    gate_acc.add_(torch.stack(gs).mean(), alpha=float(nb))
                    gate_w_total += nb
    # 必须保持为 Python float：numpy 标量会进 checkpoint extra，
    # 使 torch.load(weights_only=True) 反序列化失败且不合法
    val_loss = float(float(loss_acc.item()) / total_w)
    gate_mean = (float(gate_acc.item()) / gate_w_total
                 if gate_w_total else None)
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

    set_seed(seed)
    meta = load_meta(args.data_dir)
    train = load_bin(args.data_dir, "train")
    val = load_bin(args.data_dir, "val")

    # 实验溯源：代码版本 + 数据指纹 + 运行环境 + 实际命令行，训练前落盘，
    # 与 config_used.json（超参数）和 train_log.jsonl（训练曲线）共同构成
    # 可复现、可追溯的完整记录。
    manifest = build_run_manifest(
        tag=tag, variant=args.variant, seed=seed, config_path=args.config,
        data_dir=args.data_dir, device=device)
    save_json(manifest, os.path.join(out_dir, "run_manifest.json"))
    provenance = {
        "code_commit": manifest["git"]["commit"],
        "code_dirty": manifest["git"]["dirty"],
        "data_fingerprint": manifest["data"]["digest"],
        "config_sha256": manifest["config_sha256"],
    }

    block_size = int(cfg["block_size"])
    train_ctx = min(int(cfg.get("train_ctx", block_size)), block_size)
    # 记录实际生效的上下文长度（经过 min 截断），保证下游评测复现同一口径
    cfg_used["train_ctx"] = train_ctx
    with open(os.path.join(out_dir, "config_used.json"), "w", encoding="utf-8") as f:
        json.dump(cfg_used, f, ensure_ascii=False, indent=2)

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
    # 按设备启用 TF32（Ampere+）；门控路径在 AMP 下已是 bf16
    enable_tf32(device)
    model = GPT(**model_cfg).to(device)
    if cfg.get("compile", False) and device.startswith("cuda"):
        try:
            model = torch.compile(model)
            print("[train] torch.compile 已启用")
        except Exception as e:
            print(f"[train] torch.compile 失败（回退 eager）: {e}")
    # 模型构造会按变体消耗不同数量的全局 RNG（如 null 的 k_null 初始化），
    # 若不重置，训练期任何随机性（dropout/注意力 dropout 等）在变体间不可比。
    # 各变体共享的基础权重初始化已完成，这里重放同一 seed，保证同一 seed 下
    # 所有变体的训练随机流完全一致（数据顺序由独立的 default_rng(seed) 决定）。
    set_seed(seed)
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
    # 梯度裁剪的参数列表与优化器同口径
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
    # 流水线预取的首批：训练循环内每个 micro-step 再预取下一批
    batches = [get_train_batch(train, train_ctx, int(cfg["batch_size"]), device, rng)
               for _ in range(grad_accum)]
    tokens_seen = 0
    best_val = float("inf")
    last_eval_step = None
    last_eval_metrics = None
    # loss 设备端累加器（每步 zero_()）
    loss_acc_dev = torch.zeros((), device=device)
    # pre-clip 梯度范数：设备标量只保留引用，log 点才 .item()；
    # 历史尾部供 NaN 排查判断"梯度爆炸 vs 学习率过大"
    gn_dev_recent = []
    gn_hist = []
    # 最近一步实际消费的 micro-batch：预取会覆盖 batches，NaN 排查若只看待训练
    # 批次会漏掉"罪魁数据"，这里保留引用供 _nan_guard 优先检查
    last_batches = []

    def _nan_guard(where: str, value, step: int):
        """log/eval 点检查损失与梯度范数的有限性，异常即触发系统排查。"""
        if value is not None and not math.isfinite(float(value)):
            report = run_nan_checklist(
                model, batches, device, step, where, value,
                lr=base_lrs[0] * lr_factor(max(0, step - 1)),
                grad_norm_history=gn_hist, amp_dtype=(torch.bfloat16 if use_amp else None),
                vocab_size=int(meta["vocab_size"]),
                consumed_batches=last_batches)
            _abort_on_nan(report, out_dir)

    def do_eval(step):
        nonlocal best_val, last_eval_step, last_eval_metrics
        metrics = evaluate_val(model, val, train_ctx, device, eval_iters, amp=use_amp)
        last_eval_step, last_eval_metrics = step, metrics
        vl = metrics["val_loss"]
        _nan_guard("val_loss", vl, step)
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
            # 同步保存最优 checkpoint，避免下游评测只能用到最终步模型
            save_checkpoint(os.path.join(out_dir, "model_best.pt"), model,
                            extra={"variant": args.variant, "step": step,
                                   "seed": seed, "val_loss": vl,
                                   "val_ppl": math.exp(min(20, vl)),
                                   "val_ppl_clamped": bool(vl > 20),
                                   **provenance})
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

        # 在设备上累积 loss；快照本步将消费的批次（下一步预取会覆盖 batches）
        last_batches = [b for b in batches if b is not None]
        loss_acc_dev.zero_()
        for mi in range(grad_accum):
            x, y = batches[mi]
            # 预取下一 micro-batch：在当前步前向/反向之前发起 memmap 收集与
            # pinned H2D 提交（_stage_to_device 乒乓暂存双槽 + event 保护，
            # 两批在途互不干扰）。rng 抽取顺序与逐次采样完全一致，确定性不变。
            if step + 1 < steps or mi + 1 < grad_accum:
                batches[mi] = get_train_batch(train, train_ctx,
                                              int(cfg["batch_size"]), device, rng)
            else:
                batches[mi] = None
            with amp_ctx:
                out = model(x, targets=y)
                # grad_accum==1（默认配置）时除法结果就是 loss 本身
                loss = out["loss"] if grad_accum == 1 else out["loss"] / grad_accum
            loss.backward()
            loss_acc_dev.add_(out["loss"].detach(), alpha=1.0 / grad_accum)
            # backward 已消费完计算图；解除对 out（含逐层 past_kv）与 loss 的引用
            del out, loss
        # pre-clip 梯度范数：clip_grad_norm_ 返回裁剪前的总范数，保留设备标量
        gn_dev = torch.nn.utils.clip_grad_norm_(clip_params, grad_clip)
        gn_dev_recent.append(gn_dev)
        if len(gn_dev_recent) > log_every + grad_accum + 8:
            gn_dev_recent.pop(0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        tokens_seen += int(cfg["batch_size"]) * train_ctx * grad_accum

        if (step + 1) % log_every == 0 or step == 0:
            loss_acc = float(loss_acc_dev.item())
            gn_last = float(gn_dev_recent[-1].item()) if gn_dev_recent else None
            if gn_last is not None and math.isfinite(gn_last):
                gn_hist.append(gn_last)
                if len(gn_hist) > 200:
                    gn_hist.pop(0)
            _nan_guard("train_loss", loss_acc, step + 1)
            _nan_guard("grad_norm", gn_last, step + 1)
            rec = {"step": step + 1, "loss": loss_acc, "ppl": math.exp(min(20, loss_acc)),
                   "lr": base_lrs[0] * lr_f, "time": now_str()}
            if gn_last is not None:
                rec["grad_norm"] = gn_last
            log_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            log_f.flush()
            print(f"[train] step {step+1}/{steps} loss={loss_acc:.4f} "
                  f"ppl={math.exp(min(20, loss_acc)):.3f} lr={base_lrs[0]*lr_f:.2e}"
                  + (f" gnorm={gn_last:.2f}" if gn_last is not None else ""))
            if wb:
                wb.log({"loss": loss_acc, "lr": base_lrs[0] * lr_f,
                        **({"grad_norm": gn_last} if gn_last is not None else {})},
                       step=step + 1)
        if (step + 1) % eval_every == 0:
            do_eval(step + 1)

    if last_eval_step == steps:
        final_metrics = last_eval_metrics
    else:
        final_metrics = evaluate_val(model, val, train_ctx, device, eval_iters, amp=use_amp)
        _nan_guard("val_loss(final)", final_metrics["val_loss"], steps)
        # 最终步评测必须同样参与 best-checkpoint 选择，与训练循环内口径一致：
        # 否则 steps 非 eval_every 整数倍（或 steps < eval_every）时，更优的最终
        # 状态不会进 model_best.pt，下游 run_all 会静默回退到最终步模型，却仍按
        # “val 上选出的最优 checkpoint”口径报告。
        if final_metrics["val_loss"] < best_val:
            best_val = final_metrics["val_loss"]
            save_checkpoint(os.path.join(out_dir, "model_best.pt"), model,
                            extra={"variant": args.variant, "step": steps,
                                   "seed": seed, "val_loss": best_val,
                                   "val_ppl": math.exp(min(20, best_val)),
                                   "val_ppl_clamped": bool(best_val > 20),
                                   **provenance})
    final_val_loss = final_metrics["val_loss"]
    final_ppl = math.exp(min(20, final_val_loss))
    gmean = final_metrics["gate_mean"]
    save_checkpoint(os.path.join(out_dir, "model.pt"), model,
                    extra={"variant": args.variant, "step": steps, "seed": seed,
                           "val_ppl": final_ppl, "val_loss": final_val_loss,
                           "val_ppl_clamped": bool(final_val_loss > 20),
                           "tokens_seen": tokens_seen,
                           "n_params": n_params, "n_gate_extra": n_extra,
                           **provenance})
    save_json({"variant": args.variant, "val_ppl": final_ppl, "val_loss": final_val_loss,
               "val_ppl_clamped": bool(final_val_loss > 20),
               "gate_mean": gmean, **{k: final_metrics[k] for k in
                                      ("eval_iters_requested", "eval_iters",
                                       "val_windows", "val_tokens_evaluated")},
               "n_params": n_params, "n_gate_extra": n_extra,
               "tokens_seen": tokens_seen, "steps": steps, "seed": seed,
               **provenance},
              os.path.join(out_dir, "eval.json"))
    log_f.write(json.dumps({"step": steps, **final_metrics,
                            "val_ppl": final_ppl,
                            "val_ppl_clamped": bool(final_val_loss > 20),
                            "final": True},
                           ensure_ascii=False) + "\n")
    log_f.close()
    print(f"[train] 完成: val_ppl={final_ppl:.3f} -> {out_dir}")
    if wb:
        wb.finish()


if __name__ == "__main__":
    main()
