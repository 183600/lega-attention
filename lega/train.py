"""训练脚本（全部变体共用，保证相同参数量级与训练步数）。

用法:
  python -m lega.train --config configs/tiny.json --variant lega \
      --data_dir data/dummy --out_dir out/tiny/lega [--smoke] [--wandb]
"""
import argparse
import hashlib
import json
import math
import os
import statistics
import time
from typing import Optional

# 须在任何 CUDA 初始化之前生效。
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch

from .attention import VARIANTS, VARIANT_CN
from .curves import (GATE_SAT_HIGH, GATE_SAT_LOW, LIVE_WARN_EVALS,
                     curve_observability_warnings, live_overfit_hint)
from .data import load_bin, load_meta, get_train_batch, get_val_batch, get_val_windows
from .model import (GPT, configure_optimizer, count_params,
                    gate_extra_params, gate_extra_params_per_head,
                    save_checkpoint)
from .preflight import (allow_protocol_violations,  # noqa: E402
                        data_protocol_preflight)
from .repro import (build_run_manifest, sanity_identity_check, sanity_summary,
                    sha256_file, write_worktree_patch)
from .utils import (amp_autocast, amp_dtype_for, config_bool, data_rng,
                    enable_tf32, now_str, save_json, set_seed,
                    state_dict_fingerprint, strict_int,
                    t_critical_95)


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


def _hooked_forward(model, x, device: str, amp_dtype, cap: int = 60):
    """带前向钩子跑一次 x，返回 (非有限激活记录, 前向错误串)。

    记录按实际执行顺序，元素为模块名 + 输入/输出 NaN/Inf 计数；只保留
    输出非有限且未超过 cap 的模块。异常被捕获返回而不抛出——排查路径
    本身绝不能因前向失败而中断。
    """
    records, handles = [], []

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
        amp_ctx = amp_autocast(device, amp_dtype is not None)
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
    return records, forward_error


def audit_activations(model, pending_batches, device: str,
                      amp_dtype=None) -> dict:
    """用前向钩子定位首个“输入有限、输出非有限”的模块（NaN 排查第 2 步的主动检查）。

    仅靠非有限参数列表只能发现“哪些权重坏了”，无法区分 NaN 是数据/权重传播
    而来，还是某个算子在**有限输入**上产生（log(0)、除零、softmax 溢出、
    fp16/bf16 上溢）。本函数对失败批次做一次带钩子的前向，按实际执行顺序记录
    每个模块输出/输入的 NaN/Inf 情况：

      * origin          第一个输入有限但输出非有限的模块，附该模块位置的
                        排查指向（softmax/log/归一化/投影）；
      * first_nonfinite 第一个输出非有限的模块：若与 origin 不同，说明 NaN 来自
                        更早的算子（如 attention 内部的函数式 softmax/log）；
      * records         非有限激活沿计算图的传播链（按执行顺序，截断到上限）。
      * precision_recheck  混合精度路径出现非有限时，对同一批次做 fp32 复测：
                        若 fp32 全部有限 → 数值问题来自混合精度（下溢/上溢），
                        否则排除精度因素、按数据/代码/优化排查。

    前向只消费 x 不消费 y（标签越界不会妨碍激活审计），异常被捕获并写入
    forward_error 而不是让排查本身崩溃。

    候选批次（NaN 所在 log 区间内每一步消费的批次）会逐批检查：NaN 现场
    可能落在区间中的任意一步，只看第一批会漏掉真正的诱发批次。找到首个
    出现非有限激活的批次即停止；所有候选都有限时才用首批结果下
    "损失/聚合环节"的结论。
    """
    candidates = [b for b in (pending_batches or []) if b is not None]
    if not candidates:
        return {"checked": False, "n_modules_checked": 0, "records": [],
                "origin": None, "first_nonfinite": None, "forward_error": None,
                "precision_recheck": None, "n_batches_checked": 0,
                "checked_batch_index": None,
                "findings": ["无可用批次（NaN 出现在记录成功批次之前），跳过激活定位"]}
    cap = 60
    records, forward_error = [], None
    x = candidates[0][0]
    n_batches_checked = 0
    checked_index = 0
    fallback = None   # 首个"无非有限记录"的结果（优先无前向错误者）
    first_error = None
    for i, batch in enumerate(candidates):
        recs, err = _hooked_forward(model, batch[0], device, amp_dtype,
                                    cap=cap)
        n_batches_checked += 1
        if err is not None and first_error is None:
            first_error = err
        if recs:
            records, forward_error = recs, err
            x, checked_index = batch[0], i
            break
        if fallback is None or (fallback[1] is not None and err is None):
            fallback = (recs, err, batch[0], i)
    else:
        records, forward_error, x, checked_index = fallback
    if forward_error is None and first_error is not None:
        # 任一候选前向失败都意味着审计未完整覆盖：不能据此断言“全部有限”
        forward_error = first_error

    origin = next((r for r in records if r["input_nonfinite"] == 0), None)
    if origin is not None:
        origin = dict(origin, hint=_numerical_hint(origin["module"]))
    first = records[0] if records else None
    findings = []
    if n_batches_checked > 1:
        if records:
            findings.append(
                f"激活定位逐批检查了 {n_batches_checked} 个候选批次"
                f"（命中第 {checked_index + 1} 个）：NaN 现场在区间中段时"
                "只用单批会检查错误的输入")
        else:
            findings.append(
                f"激活定位逐批检查了 {n_batches_checked} 个候选批次，"
                "均未发现非有限激活")
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
            f"所有候选批次的模块输出均有限（已检查 {n_batches_checked} 批）——"
            "NaN 产生在损失计算/评测聚合环节（如分块交叉熵的目标越界），"
            "不是前向算子溢出")

    # 混合精度隔离复测：amp 路径出现非有限时，用同一批次在 fp32 下重跑。
    # 这是"先查数据/代码、再查混合精度"顺序里的主动证据，而不是只报 dtype。
    # CPU bf16 与 fp32 的动态范围相同，不存在 fp16 式下溢，且 CPU 上 fp32
    # 复测无法隔离硬件内核差异；该复测只在 CUDA AMP 路径触发。
    amp_on = amp_dtype is not None and str(device).startswith("cuda")
    precision_recheck = None
    if amp_on and (records or forward_error):
        fp_records, fp_err = _hooked_forward(model, x, device, None, cap=cap)
        if fp_err is not None:
            precision_recheck = {
                "ran": True, "fp32_nonfinite_records": None,
                "fp32_origin": None, "fp32_forward_error": fp_err,
                "verdict": ("fp32 复测未能完成（前向失败），无法隔离混合精度因素；"
                            "先按数据/代码路径排查")}
        else:
            fp_origin = next((r for r in fp_records if r["input_nonfinite"] == 0),
                             None)
            if fp_records:
                verdict = (f"fp32 复测同样出现 {len(fp_records)} 个非有限激活 → "
                           "不是混合精度下溢/上溢，按数据/代码/优化路径排查")
            else:
                verdict = (
                    f"{str(amp_dtype).replace('torch.', '')} 路径非有限而同一批次 "
                    "fp32 复测全部有限 → 混合精度数值问题：fp16 应加 GradScaler "
                    "损失缩放或改用 bf16，并对可疑层（见 origin）升精度复测")
            precision_recheck = {
                "ran": True, "fp32_nonfinite_records": len(fp_records),
                "fp32_origin": (dict(fp_origin, hint=_numerical_hint(
                    fp_origin["module"])) if fp_origin else None),
                "fp32_forward_error": None, "verdict": verdict}
        findings.append(precision_recheck["verdict"])
    return {
        # checked = 审计前向成功执行；records/origin 为空表示前向算子全部有限
        "checked": forward_error is None,
        "n_modules_checked": len(list(model.named_modules())) - 1,
        "n_batches_checked": n_batches_checked,
        "checked_batch_index": checked_index,
        "records": records[:cap],
        "truncated": len(records) >= cap,
        "origin": origin,
        "first_nonfinite": first,
        "forward_error": forward_error,
        "precision_recheck": precision_recheck,
        "findings": findings,
    }


def run_nan_checklist(model, pending_batches, device: str, step: int, where: str,
                      loss_value, lr: float, grad_norm_history, amp_dtype,
                      vocab_size: int, consumed_batches=None,
                      consumed_step_labels=None, peak_lr=None,
                      first_bad_step=None, interval=None,
                      optimizer=None) -> dict:
    """训练出现非有限值时的系统化排查：数据 → 代码 → 混合精度 → 梯度/学习率。

    排查顺序固定（先便宜后昂贵、先外部后内部），每步产出可核对的证据，
    避免"看到 NaN 就乱调参"。返回可直接写盘的报告 dict。

    consumed_batches：失败所在 log 区间内每一步消费的候选批次（训练循环已
    预取并覆盖了 batches，调用方必须显式传入；顺序=区间内步序）。数据检查
    与激活定位都遍历这些批次——NaN 可能在区间中段首次出现，只用"最后一步
    批次"会把"罪魁数据"漏掉。consumed_step_labels 为与之一一对应的日志步号
    （1-based），用于把每条数据检查定位到具体步；缺省时保留旧标签。
    first_bad_step / interval：设备端标记的区间内首个非有限损失步与区间范围；
    检测点（step）可能晚于首个失败步，报告必须同时给出两者。
    optimizer：可选。Adam 的 exp_avg/exp_avg_sq 可能先于参数溢出；若只重置
    参数而不清优化器状态，下一步会把 NaN 写回权重。调用方在失败步传真实
    优化器，报告会列出非有限状态张量及所属参数。
    """
    report = {
        "time": now_str(), "step": int(step), "where": str(where),
        "loss_value": (None if loss_value is None else float(loss_value)),
        "device": str(device), "lr_at_failure": float(lr),
        "amp_dtype": (str(amp_dtype) if amp_dtype else None),
        # NaN 在 log 点才被检测到：step 是检测点，first_bad_step 才是区间内
        # 首个出现非有限损失的步（两者可能相差最多 log_every-1 步）。
        "first_bad_step": (int(first_bad_step) if first_bad_step is not None
                           else None),
        "interval": ([int(interval[0]), int(interval[1])] if interval else None),
    }

    # ---- 1. 数据：输入流是否含非法值（NaN/Inf/越界 id）；先查失败区间候选批
    labels = list(consumed_step_labels or [])
    labeled = []
    for i, b in enumerate(consumed_batches or []):
        if b is None:
            continue
        lab = labels[i] if i < len(labels) else None
        labeled.append((f"失败区间批{i}@step{lab}" if lab is not None
                        else f"失败步批{i}", b, lab))
    labeled += [(f"待训练批{i}", b, None)
                for i, b in enumerate(pending_batches or []) if b is not None]
    data_findings = []
    bad_data = []
    n_checked = 0
    for label, batch, batch_step in labeled:
        x, y = batch
        for name, t in (("x", x), ("y", y)):
            n_checked += 1
            if not torch.is_floating_point(t) and not torch.is_complex(t):
                lo, hi = int(t.min()), int(t.max())
                bad = (lo < 0) or (hi >= vocab_size)
                data_findings.append(
                    f"{label} {name}: 整数 id 范围 [{lo}, {hi}]，"
                    + ("**越界 vocab**" if bad else "合法"))
                if bad:
                    bad_data.append({"batch": label, "batch_step": batch_step,
                                     "tensor": name, "kind": "id_out_of_range",
                                     "min_id": lo, "max_id": hi})
            else:
                n_nan = int(torch.isnan(t).sum())
                n_inf = int(torch.isinf(t).sum())
                data_findings.append(
                    f"{label} {name}: 浮点输入 NaN={n_nan}/Inf={n_inf}"
                    + ("（**数据侧混入非法值**）" if n_nan or n_inf else ""))
                if n_nan or n_inf:
                    bad_data.append({"batch": label, "batch_step": batch_step,
                                     "tensor": name, "kind": "nonfinite_input",
                                     "nan": n_nan, "inf": n_inf})
    bad_steps = sorted({d["batch_step"] for d in bad_data
                        if d.get("batch_step") is not None})
    report["checklist_data"] = {
        "findings": data_findings,
        "n_tensors_checked": n_checked,
        "n_consumed_batches": sum(1 for b in (consumed_batches or [])
                                  if b is not None),
        "first_bad_step": report["first_bad_step"],
        "bad_batches": bad_data,
        "first_bad_data_step": (bad_steps[0] if bad_steps else None),
        "n_bad_batches": len({d["batch"] for d in bad_data}),
        "note": ("失败区间批=检测点所在 log 区间内每一步消费的候选批次"
                 "（优先检查；NaN 可能早在区间中段出现，预取已覆盖 batches）；"
                 + (f"本区间首个非有限损失步={int(first_bad_step)}，"
                    "检测点 step 不等于首个失败步；"
                    if first_bad_step is not None else "")
                 + "token 流为 uint16 整数且 prepare_bins 入库时已拒绝 "
                   "NaN/Inf/越界 id（见 data_report.json 的 token 流硬校验）；"
                   "浮点输入出现 NaN/Inf 才是数据侧问题"),
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
    # 参数梯度：反向已经溢出的直接证据（训练循环在失败步保留梯度到排查后
    # 才清零）。它比“参数已经变 NaN”更早、更能定位到具体层。
    bad_grads = []
    for n, p in model.named_parameters():
        if p.grad is None or not torch.isfinite(p.grad).all():
            if p.grad is not None:
                bad_grads.append(
                    f"{n} (NaN={int(torch.isnan(p.grad).sum())}, "
                    f"Inf={int(torch.isinf(p.grad).sum())}, "
                    f"shape={list(p.grad.shape)})")
        if len(bad_grads) >= 8:
            break
    # 优化器状态：Adam 动量/二阶矩是非有限值时，即使当前参数仍有限，
    # 下一步更新也会把状态里的溢出传播回权重（只重置参数无效）。
    bad_opt_state = []
    n_state_tensors = 0
    if optimizer is not None and hasattr(optimizer, "state"):
        name_by_id = {id(p): n for n, p in model.named_parameters()}
        try:
            state_items = list(optimizer.state.items())
        except (AttributeError, TypeError):
            state_items = []
        for p, st in state_items:
            if not isinstance(st, dict):
                continue
            for k, v in st.items():
                if not torch.is_tensor(v) or not (
                        torch.is_floating_point(v) or torch.is_complex(v)):
                    continue
                n_state_tensors += 1
                if torch.isfinite(v).all():
                    continue
                if len(bad_opt_state) < 8:
                    bad_opt_state.append(
                        f"{name_by_id.get(id(p), 'param')}.{k} "
                        f"(NaN={int(torch.isnan(v).sum())}, "
                        f"Inf={int(torch.isinf(v).sum())}, "
                        f"shape={list(v.shape)})")
    report["checklist_params"] = {
        "nonfinite_tensors": bad_params,
        "nonfinite_grads": bad_grads,
        "note": ("非有限值出现在哪些层决定了排查方向：门控/输出侧 → 检查 log(0)、"
                 "除零与 softmax 数值溢出路径；注意力核心 → 检查 score 缩放；"
                 "梯度非有限说明反向已溢出（常先于参数变 NaN）；"
                 "本仓库已知防护：门控 c 项 log(max_p) 有 clamp_min(1e-30)，"
                 "交叉熵经 log_softmax（数值稳定），除法处均有 clamp_min 兜底"),
    }
    report["checklist_activations"] = activation_audit

    # ---- 3. 混合精度：当前 autocast 精度与下溢/溢出风险；
    #     若激活审计做了 fp32 隔离复测，把结论并入本步（有证据而非只有推断）
    if amp_dtype is None:
        prec_note = "未启用 autocast（fp32/TF32 路径）：可排除混合精度下溢因素"
    elif "bfloat16" in str(amp_dtype):
        prec_note = ("bf16 autocast：动态范围与 fp32 相同，下溢风险低；"
                     "极大数据仍可能溢出，可对可疑层加 .float() 升精度复测")
    else:
        prec_note = ("fp16 autocast：梯度/激活下溢是常见 NaN 来源，"
                     "应使用 GradScaler 损失缩放，或改用 bf16 复测")
    recheck = activation_audit.get("precision_recheck")
    if recheck and recheck.get("ran"):
        prec_note += "；fp32 隔离复测：" + recheck["verdict"]
    report["checklist_precision"] = {"amp_dtype": (str(amp_dtype) if amp_dtype
                                                   else None), "note": prec_note,
                                     "recheck": recheck}

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
    hist_max = hist_med = hist_spike_ratio = None
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
        if len(finite) >= 3:
            # 整个有限历史里的最坏峰值：尖峰可能发生在更早的 log 点、
            # 当前已恢复；只看最后一点会漏掉“爆炸后恢复”的脆弱训练。
            hist_max = float(max(finite))
            hist_med = float(statistics.median(finite))
            if hist_med > 0:
                hist_spike_ratio = hist_max / hist_med
            if hist_spike_ratio is not None and hist_spike_ratio >= 10:
                opt_findings.append(
                    f"历史 pre-clip 梯度范数峰值 {hist_max:.3e} 为中位数 "
                    f"{hist_med:.3e} 的 {hist_spike_ratio:.0f} 倍"
                    "（历史上出现过爆炸性尖峰；即使当前已恢复也要检查学习率/"
                    "grad_clip 与对应步批次）")
        if not math.isfinite(last):
            opt_findings.append("最近一次 pre-clip 梯度范数非有限 → 梯度已爆炸")
    else:
        opt_findings.append("无梯度范数历史（NaN 出现过早，尚未记录到 log 点）")
    if bad_grads:
        opt_findings.append(
            f"参数梯度出现 {len(bad_grads)} 个非有限张量（{bad_grads[0]}）："
            "反向已溢出/爆炸，下一轮更新会把 NaN 写入参数")
    if bad_opt_state:
        opt_findings.append(
            f"优化器状态出现 {len(bad_opt_state)} 个非有限张量（{bad_opt_state[0]}）："
            "Adam 动量/二阶矩已溢出，仅重置参数无效，需恢复 checkpoint 或重建"
            "优化器状态")
    elif optimizer is not None and n_state_tensors == 0:
        opt_findings.append("优化器尚无状态张量（首步即失败或优化器未 step）")
    # peak_lr 为调度峰值（warmup 后的 base lr）：失败步可能仍在 warmup 中，
    # 实际 lr 与峰值不同，必须分别显示才能判断“是否已到峰值附近才炸”
    opt_findings.append(
        f"失败时学习率 {lr:.2e}"
        + (f"（调度峰值 {peak_lr:.2e}）" if peak_lr is not None else "")
        + "：过大学习率会在 warmup 结束前后率先炸出 NaN")
    report["checklist_optimization"] = {
        "findings": opt_findings,
        "grad_norm_history_tail": hist[-20:],
        "grad_norm_history_max": hist_max,
        "grad_norm_history_median": hist_med,
        "grad_norm_history_spike_ratio": hist_spike_ratio,
        "nonfinite_grads": bad_grads,
        "nonfinite_optimizer_state": bad_opt_state,
        "n_optimizer_state_tensors_checked": n_state_tensors,
    }

    # ---- 建议动作：按固定排查顺序给出
    actions = [
        "1) 数据：核对 data_report.json 的 token 流硬校验与上方批次范围检查；"
        "浮点输入出现 NaN/Inf 才需回查数据管道",
        "2) 代码：先看 checklist_activations 的 origin（首个“输入有限、输出非有限”"
        "的模块）再对照非有限参数列表，复核该层 log(0)/除零/softmax 溢出路径"
        "（本仓库门控 c 项已 clamp_min(1e-30)）",
        f"3) 精度：{prec_note}",
        "4) 优化：先看 checklist_optimization 的梯度/优化器状态是否非有限"
        "（Adam 动量溢出需恢复 checkpoint 或重建状态，不能只重置参数），"
        "再降低学习率（如减半）、延长 warmup、收紧 grad_clip（如 0.5），"
        "并观察梯度范数是否再出尖峰",
    ]
    report["suggested_actions"] = actions
    # 固定排查顺序显式落盘：消费方不应依赖 JSON 对象键的插入顺序来恢复
    # “数据→代码→混合精度→优化”的因果链。
    report["checklist_order"] = [
        "checklist_data", "checklist_params", "checklist_activations",
        "checklist_precision", "checklist_optimization",
    ]
    return report


def _abort_on_nan(report: dict, out_dir: str) -> None:
    """落盘 NaN 排查报告并中止训练：NaN 之后的步数只会污染曲线。"""
    path = os.path.join(out_dir, "nan_report.json")
    save_json(report, path)
    where_bad = ""
    if report.get("first_bad_step") is not None:
        where_bad = f"，本区间首个非有限损失步 {report['first_bad_step']}"
        if report.get("interval"):
            where_bad += (f"（区间 {report['interval'][0]}~"
                          f"{report['interval'][1]}；检测点晚于失败步，排查输入"
                          "已覆盖整个区间）")
    print(f"[train] ❌ 检测到非有限损失（step {report['step']}，{report['where']}"
          f"{where_bad}），排查报告 -> {path}")
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
    iters = strict_int(iters, "eval_iters")
    if iters < 1:
        raise ValueError(f"eval_iters 必须 >= 1，收到 {iters}")
    val_len = strict_int(val_len, "val_len")
    ctx = strict_int(ctx, "ctx")
    if ctx < 1:
        raise ValueError(f"ctx 必须 >= 1，收到 {ctx}")
    n_windows = (val_len - 1) // ctx
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


def val_audit_batch(val, ctx: int, iters: int, device: str, fallback=None):
    """val_loss 非有限时，取一个真实验证窗口批次作为数据/激活排查的输入。

    验证窗口才是产生该损失的输入现场；训练批次只应在取窗失败时回退
    （如验证集过短），排查报告中由调用方如实呈现。返回批次列表
    [(x, y)]，形状约定与 run_nan_checklist 的 consumed_batches 一致。
    """
    try:
        indices, _ = _val_eval_indices(len(val), ctx, iters)
        return [get_val_windows(val, ctx, indices[:1], device)]
    except (ValueError, OSError, RuntimeError):
        return [b for b in (fallback or []) if b is not None]


def evaluate_val(model, val, ctx, device, iters, amp=False, batch_windows: int = None,
                 collect_window_losses: bool = False):
    """在不重复、覆盖全验证集的窗口上同时计算 val_loss 与门控均值。

    多个窗口拼成一个批量做前向：每个窗口长度相同，token 加权后的均值
    loss/门控统计与逐窗口完全一致。
    batch_windows 默认按每批约 16384 tokens 估算，并随验证集窗口数收窄。
    collect_window_losses=True 时额外返回逐窗口损失与窗口序号（仅独立
    评测入口使用）：供下游按同一批窗口做配对分析，量化“有限窗口抽样”
    对变体差异的影响。训练内评测不需要，避免把几十个浮点写进每次
    train_log。
    """
    was_training = model.training
    model.eval()
    try:
        collect_gate = model.variant not in ("standard", "null")
        indices, n_windows = _val_eval_indices(len(val), ctx, iters)
        if batch_windows is None:
            batch_windows = max(1, min(16384 // max(1, ctx), len(indices)))
        else:
            batch_windows = strict_int(batch_windows, "batch_windows")
            if batch_windows <= 0:
                raise ValueError(
                    f"batch_windows 必须是正整数，收到 {batch_windows}")
        # 设备端累积，循环结束后统一读取。
        # 窗口等长，权重即窗口数，用 Python int 就地累计。
        loss_acc = torch.zeros((), device=device)
        gate_acc = torch.zeros((), device=device)
        gate_sq_acc = torch.zeros((), device=device)
        gate_low_acc = torch.zeros((), device=device)
        gate_high_acc = torch.zeros((), device=device)
        gate_layer_acc = []
        total_w = 0
        gate_w_total = 0
        window_losses = []
        amp_ctx = amp_autocast(device, bool(amp))
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
                    # 验证评测只消费门控均值 g，使用轻量统计；额外取每窗口损失
                    # 用于估计“有限个验证窗口”带来的抽样误差。
                    out = model(x, targets=y, need_stats=collect_gate,
                                stats_argmax=False, return_window_losses=True)
                nb = len(chunk)     # 窗口等长，按窗口数加权即 token 加权
                # add_(alpha=nb) 直接按批大小加权累加（loss 前向/门控闭式求和
                # 本就以 fp32 返回）。
                loss_acc.add_(out["loss"].detach(), alpha=float(nb))
                total_w += nb
                wl = out["loss_per_window"].detach().float().cpu().numpy()
                window_losses.extend(float(v) for v in wl.reshape(-1))
                if collect_gate and out["gate_stats"]:
                    # 在设备端先 stack 成 (n_layers,) 再求均值；同时保留二阶矩、
                    # 低/高饱和比例与逐层均值，供训练曲线诊断门控是否塌缩/饱和。
                    means, sqs, lows, highs = [], [], [], []
                    for li, s_ in enumerate(out["gate_stats"]):
                        g = s_.get("g")
                        if g is None:
                            continue
                        gf = g.detach().float()
                        means.append(gf.mean())
                        sqs.append((gf * gf).mean())
                        lows.append((gf < GATE_SAT_LOW).float().mean())
                        highs.append((gf > GATE_SAT_HIGH).float().mean())
                        if li >= len(gate_layer_acc):
                            gate_layer_acc.append(torch.zeros((), device=device))
                        gate_layer_acc[li].add_(means[-1], alpha=float(nb))
                    if means:
                        gate_acc.add_(torch.stack(means).mean(), alpha=float(nb))
                        gate_sq_acc.add_(torch.stack(sqs).mean(), alpha=float(nb))
                        gate_low_acc.add_(torch.stack(lows).mean(), alpha=float(nb))
                        gate_high_acc.add_(torch.stack(highs).mean(), alpha=float(nb))
                        gate_w_total += nb
        # 必须保持为 Python float：numpy 标量会进 checkpoint extra，
        # 使 torch.load(weights_only=True) 反序列化失败且不合法
        val_loss = float(float(loss_acc.item()) / total_w)
        gate_mean = (float(gate_acc.item()) / gate_w_total
                     if gate_w_total else None)
        gate_std = None
        gate_frac_low = None
        gate_frac_high = None
        gate_layer_mean = None
        if gate_w_total:
            mean_sq = float(gate_sq_acc.item()) / gate_w_total
            var = max(0.0, mean_sq - gate_mean * gate_mean)
            gate_std = math.sqrt(var)
            gate_frac_low = float(gate_low_acc.item()) / gate_w_total
            gate_frac_high = float(gate_high_acc.item()) / gate_w_total
            gate_layer_mean = [float(x.item()) / gate_w_total
                               for x in gate_layer_acc]
        # 验证集只评测 eval_iters 个等间隔窗口：点估计本身带抽样误差。这里把
        # 窗口作为抽样单元，报告窗口间标准误与 t 近似 95% CI（小样本比正态
        # 更宽）；CI 不包含窗口间空间相关，也不替代跨随机种子的方差（后者在
        # run_all 汇总）。
        val_loss_se = None
        val_loss_ci95 = None
        val_ppl_ci95 = None
        val_loss_min = None
        val_loss_max = None
        losses = np.asarray(window_losses, dtype=np.float64)
        if losses.size and np.all(np.isfinite(losses)):
            val_loss_min = float(losses.min())
            val_loss_max = float(losses.max())
            if losses.size >= 2:
                se = float(losses.std(ddof=1) / math.sqrt(losses.size))
                # 小样本用 t 临界值：eval_iters=20 时 t≈2.09，比正态 1.96 宽
                # 约 7%；窗口数少时不该假装有正态大样本的精度。
                tcrit = t_critical_95(losses.size - 1)
                if math.isfinite(se) and tcrit is not None:
                    lo, hi = val_loss - tcrit * se, val_loss + tcrit * se
                    val_loss_se = se
                    val_loss_ci95 = [lo, hi]
                    val_ppl_ci95 = [math.exp(min(20.0, lo)),
                                    math.exp(min(20.0, hi))]
        out = {
            "val_loss": val_loss,
            "val_loss_se": val_loss_se,
            "val_loss_ci95": val_loss_ci95,
            "val_ppl_ci95": val_ppl_ci95,
            "val_loss_min": val_loss_min,
            "val_loss_max": val_loss_max,
            "gate_mean": gate_mean,
            "gate_std": gate_std,
            "gate_frac_low": gate_frac_low,
            "gate_frac_high": gate_frac_high,
            "gate_layer_mean": gate_layer_mean,
            "eval_iters_requested": int(iters),
            "eval_iters": int(len(indices)),
            "val_windows": int(n_windows),
            "val_tokens_evaluated": int(len(indices) * ctx),
        }
        if collect_window_losses:
            out["window_losses"] = [float(v) for v in window_losses]
            out["window_indices"] = [int(i) for i in indices]
        return out
    finally:
        model.train(was_training)


def estimate_val_loss(model, val, ctx, device, iters, amp=False):
    """兼容旧调用：只返回 val_loss。"""
    return evaluate_val(model, val, ctx, device, iters, amp=amp)["val_loss"]


def mean_gate_value(model, val, ctx, device, iters=1, amp=False):
    """兼容旧调用的门控均值；与 evaluate_val 使用同一组验证窗口与精度。"""
    return evaluate_val(model, val, ctx, device, iters, amp=amp)["gate_mean"]


# ---------------------------------------------------------------- 早停（可选）

def _config_int(value, name: str, minimum: int) -> int:
    """配置整数：拒绝 bool/小数/非有限值，并检查配置语义下界。

    训练循环里有 ``% log_every``、``/ grad_accum`` 等直接运算：0 会在训练
    中途才炸，bool/1.5 则会被 int() 静默截断并与 config_used/manifest 记录
    的声明值错位。这里在建模前统一失败，避免“日志已落盘、训练已开跑”后
    才发现配置不可执行。
    """
    value = strict_int(value, f"config.{name}")
    if value < int(minimum):
        raise ValueError(f"config.{name} 必须 >= {minimum}，收到 {value}")
    return value


def _meta_vocab_size(meta: dict) -> int:
    """meta.vocab_size 必须是精确正整数。

    meta.json 是外部输入：``int(True)``/``int(1.5)`` 会静默变成 1，
    ``int(-3)`` 会构造出负词表并在训练中途才以越界错误失败。模型词表一旦
    与数据元数据错位，checkpoint 和评测口径都不可比，因此在建模前显式拒绝。
    """
    vocab_size = strict_int(meta.get("vocab_size"), "meta.vocab_size")
    if vocab_size <= 0:
        raise ValueError(
            f"meta.vocab_size 必须是正整数，收到 {vocab_size}")
    return vocab_size


def _config_nonneg_float(value, name: str) -> float:
    """训练数值超参数必须是有限非负数，拒绝 bool/NaN/Inf/负值。

    ``float(True)=1.0`` 会让布尔值冒充真实超参数；负的 lr/weight_decay 会
    朝错误方向更新，负的 grad_clip 会让 ``clip_grad_norm_`` 把梯度整体反号。
    这些配置错误必须在开训/写任何 run 产物之前失败。
    """
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"config.{name} 必须是数值，收到 bool {value!r}")
    if isinstance(value, str) and "_" in value:
        raise ValueError(f"config.{name} 必须是数值，收到 {value!r}")
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as e:
        raise ValueError(
            f"config.{name} 必须是数值，收到 {value!r}") from e
    if not math.isfinite(value) or value < 0:
        raise ValueError(
            f"config.{name} 必须是有限非负数，收到 {value!r}")
    return value


def _config_finite_float(value, name: str) -> float:
    """训练配置中的普通浮点超参数必须是有限数，拒绝 bool/NaN/Inf。

    ``float(True)=1.0`` 会让布尔值冒充真实超参数；NaN/Inf 一旦进入模型
    初始化或优化器，会在训练若干步后才污染参数，无法从 run 产物区分。
    """
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"config.{name} 必须是数值，收到 bool {value!r}")
    if isinstance(value, str) and "_" in value:
        raise ValueError(f"config.{name} 必须是数值，收到 {value!r}")
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as e:
        raise ValueError(
            f"config.{name} 必须是数值，收到 {value!r}") from e
    if not math.isfinite(value):
        raise ValueError(
            f"config.{name} 必须是有限数值，收到 {value!r}")
    return value


def _config_probability(value, name: str) -> float:
    """dropout 类概率必须落在 [0, 1] 且为有限数（拒绝 bool）。

    NaN 与 0 比较恒为 False，会静默走 ``nn.Identity`` 分支把 dropout
    关掉；bool True 会变成 p=1 把激活全部置零。两者都会让配置声明的
    正则化强度与实际训练行为不一致，必须在构造模型前显式拒绝。
    """
    value = _config_finite_float(value, name)
    if not 0.0 <= value <= 1.0:
        raise ValueError(
            f"config.{name} 必须在 [0, 1] 内，收到 {value!r}")
    return value


def _config_min_delta(value) -> float:
    """早停相对改善阈值：必须是 [0, 1) 内有限浮点，拒绝 bool/NaN/Inf。

    early_stop_update 用 ``vl < best * (1 - min_delta)`` 判定实质改善：
    min_delta<0 会把正常改善判成未改善，>=1 则任何非负损失都不可能
    “改善”（best*(1-1)<=0），耐心计数将无限累加。NaN 还会绕过比较，
    因此必须在开训前显式拒绝，而不是交给 max()/float() 静默处理。
    """
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"config.early_stop_min_delta 必须是数值，收到 bool {value!r}")
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as e:
        raise ValueError(
            f"config.early_stop_min_delta 必须是数值，收到 {value!r}") from e
    if not math.isfinite(value) or not 0.0 <= value < 1.0:
        raise ValueError(
            f"config.early_stop_min_delta 必须在 [0, 1) 内且有限，收到 {value!r}")
    return value


def _val_ci_rel(vl: float, val_ci95) -> Optional[float]:
    """当前验证损失 95% CI 的相对半宽；缺失/非法区间返回 None。

    早停与曲线诊断必须用同一噪声口径：有限验证窗口的抽样误差不是改善。
    区间必须有限、有序且 vl>0，否则不参与阈值（旧日志/坏区间保持原行为）。
    """
    if not isinstance(val_ci95, (list, tuple)) or len(val_ci95) != 2:
        return None
    try:
        lo, hi = float(val_ci95[0]), float(val_ci95[1])
    except (TypeError, ValueError, OverflowError):
        return None
    if (not math.isfinite(lo) or not math.isfinite(hi)
            or hi < lo or not math.isfinite(vl) or vl <= 0):
        return None
    return (hi - lo) / 2.0 / vl


def early_stop_update(vl: float, best_val: float, evals_since_improve: int,
                      min_delta: float = 0.0, patience: int = 0,
                      val_ci95=None):
    """一次验证评测后更新早停状态，返回 (best_val, evals_since_improve, stop)。

    口径（与曲线诊断的"过拟合→早停"建议衔接，默认关闭、显式开启才生效）：
      * 改善 = 验证损失相对当前最优有实质下降（vl < best·(1−min_delta)），
        相对阈值不随任务损失尺度漂移；min_delta=0 退化为严格更优；
      * val_ci95 可用时，实质改善还必须超过当前验证损失的 95% CI 相对半宽
        （有效阈值 max(min_delta, ci_rel)）：有限窗口的抽样噪声里的“改善”
        不算改善，否则每隔几次评测的小幅噪声下降会不断重置耐心、让早停
        永远无法触发；也不应把噪声下降当成模型真的变好；
      * best_val 按严格最小维护（checkpoint 选择口径不变），因此
        min_delta>0 或噪声内的“小幅改善”仍会保存 checkpoint，但不重置耐心；
      * 连续 patience 次评测无实质改善 → stop=True；patience<=0 时功能关闭，
        计数仍如实维护（供日志显示"距最优已多少次评测未改善"）；
      * 非有限 vl 由 NaN 防护路径负责中止，这里不消费也不误判改善。
    """
    if not math.isfinite(float(vl)):
        return best_val, evals_since_improve, False
    if best_val == float("inf"):
        return float(vl), 0, False
    noise_rel = _val_ci_rel(float(vl), val_ci95)
    threshold = max(0.0, float(min_delta), noise_rel or 0.0)
    improved = float(vl) < best_val * (1.0 - threshold)
    new_best = float(vl) if float(vl) < best_val else best_val
    if improved:
        return new_best, 0, False
    count = int(evals_since_improve) + 1
    # patience<=0 表示关闭：负值曾因 truthy 而立即触发退出，必须显式排除。
    return new_best, count, bool(int(patience) > 0 and count >= int(patience))


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
    p.add_argument("--early_stop_patience", type=int, default=None,
                   help="验证损失连续 N 次评测无实质改善则早停（0=关闭，"
                        "默认读配置 early_stop_patience；启用后各 run 实际步数"
                        "可能不同，run_all 审计按计划步数核对）")
    p.add_argument("--sanity_file", default=None,
                   help="开训前小批量过拟合 sanity 结果 JSON 路径；提供时把"
                        "“本 run 通过代码正确性闸门”的摘要与哈希写入 "
                        "run_manifest.json，使 checkpoint 可自证（run_all "
                        "默认透传）")
    p.add_argument("--allow_data_protocol_violations", action="store_true",
                   help="即使 data_report 检测到打乱/字符切分/文档跨 split/"
                        "跨 split n-gram 重叠也继续训练（仅用于复现历史实验；"
                        "run_all 通过同名参数与环境变量向子进程透传）")
    args = p.parse_args()

    # 数据协议闸门必须在任何训练计算之前执行：直接 `python -m lega.train`
    # 与 run_all/调优 trial 共用同一份防泄漏结论，避免“报告有告警但训练照跑”。
    allow_override = allow_protocol_violations(
        args.allow_data_protocol_violations)
    data_preflight = data_protocol_preflight(
        args.data_dir, allow_override=allow_override)
    if not data_preflight["report_present"]:
        print("[train] 警告：数据目录缺少 data_report.json，"
              "无法执行训练前防泄漏闸门（旧数据或未走统一入口）")
    elif data_preflight["critical_violations"]:
        detail = "；".join(
            f"[{a.get('code')}] {a.get('message')}"
            for a in data_preflight["critical_violations"][:5])
        print(f"[train] ⚠ 数据协议自检发现 "
              f"{len(data_preflight['critical_violations'])} 个严重问题，"
              f"但已显式放行（结果不得用于正式结论）：{detail}")

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise ValueError(
            f"配置文件顶层必须是 JSON 对象，收到 {type(cfg).__name__}: "
            f"{args.config}")
    if args.smoke:
        for k, v in SMOKE_OVERRIDES.items():
            cfg[k] = min(v, cfg.get(k, v)) if k in ("batch_size",) else v
    if args.steps is not None:
        cfg["steps"] = args.steps
    if args.early_stop_patience is not None:
        cfg["early_stop_patience"] = args.early_stop_patience
    patience_raw = cfg.get("early_stop_patience", 0)
    early_stop_patience = _config_int(
        0 if patience_raw is None else patience_raw, "early_stop_patience", 0)
    min_delta_raw = cfg.get("early_stop_min_delta", 0.0)
    early_stop_min_delta = _config_min_delta(
        0.0 if min_delta_raw is None else min_delta_raw)
    # 数值超参数在开训前统一校验：负 lr/weight_decay 会朝反方向优化，
    # 负 grad_clip 会翻转梯度符号，NaN/Inf 会直接污染参数；这些错误不能
    # 等到写出 manifest/config_used 或训练若干步后才暴露。
    lr = _config_nonneg_float(cfg["lr"], "lr")
    weight_decay = _config_nonneg_float(
        cfg.get("weight_decay", 0.1), "weight_decay")
    grad_clip = _config_nonneg_float(cfg.get("grad_clip", 1.0), "grad_clip")
    dropout = _config_probability(cfg.get("dropout", 0.0), "dropout")
    attn_dropout = _config_probability(
        cfg.get("attn_dropout", 0.0), "attn_dropout")
    gate_bias_init = _config_finite_float(
        cfg.get("gate_bias_init", 2.0), "gate_bias_init")
    # 种子必须是精确整数：配置里的 "seed": true → 1 或 1.5 → 1 会让本次
    # 训练以与声明不同的种子播种，而 checkpoint/manifest/曲线记的是截断值，
    # 事后无法区分。int(True)==1、int(1.5)==1 都不可接受。
    seed = strict_int(
        args.seed if args.seed is not None else cfg.get("seed", 1337), "seed")
    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")

    # 实际生效精度必须在写 config_used/manifest 之前确定：只记录 "amp": true
    # 无法区分 bf16 autocast 与不可用时的 fp32 回退，事后会把两种数值路径
    # 当成同一实验对比。
    amp_enabled = config_bool(cfg.get("amp", True), "amp")
    amp_dt = amp_dtype_for(device, amp_enabled)

    tag = os.path.splitext(os.path.basename(args.config))[0] + ("_smoke" if args.smoke else "")
    out_dir = args.out_dir or os.path.join("out", tag, args.variant)
    os.makedirs(out_dir, exist_ok=True)
    cfg_used = dict(cfg)
    cfg_used.update({"variant": args.variant, "seed": seed, "device": device,
                     "tag": tag,
                     # 显式写入实际生效的早停口径：CLI 覆盖也必须进入受控变量审计
                     "early_stop_patience": early_stop_patience,
                     "early_stop_min_delta": early_stop_min_delta,
                     # 记录规范化后的模型正则化超参数：bool/字符串输入
                     # 不能只在内存里被转换，而让 config_used 复述原始值。
                     "dropout": dropout,
                     "attn_dropout": attn_dropout,
                     "gate_bias_init": gate_bias_init,
                     # 实际生效精度（None=未启用 autocast，回退 fp32）
                     "amp": amp_enabled,
                     "amp_dtype": (str(amp_dt).replace("torch.", "")
                                   if amp_dt is not None else None),
                     # 数据协议是否被显式放行：正式结果必须能区分
                     "data_protocol_override": bool(allow_override)})

    set_seed(seed)
    meta = load_meta(args.data_dir)
    vocab_size = _meta_vocab_size(meta)
    train = load_bin(args.data_dir, "train")
    val = load_bin(args.data_dir, "val")

    # 按设备启用 TF32（Ampere+）；门控路径在 AMP 下已是 bf16。
    # 必须在 build_run_manifest 之前：manifest 记录的是训练实际生效的精度开关，
    # 否则 CUDA 上会记下 tf32_matmul=False 的“训练前默认值”，环境溯源失真。
    enable_tf32(device)

    # 实验溯源：代码版本 + 数据指纹 + 运行环境 + 实际命令行，训练前落盘，
    # 与 config_used.json（超参数）和 train_log.jsonl（训练曲线）共同构成
    # 可复现、可追溯的完整记录。
    sanity_info = sanity_summary(args.sanity_file) if args.sanity_file else None
    if sanity_info and not sanity_info.get("present"):
        print(f"[train] ⚠ --sanity_file 指定的结果文件不存在：{args.sanity_file}"
              "（manifest 按 present=false 记录，不冒充通过）")
    manifest = build_run_manifest(
        tag=tag, variant=args.variant, seed=seed, config_path=args.config,
        data_dir=args.data_dir, device=device, sanity=sanity_info)
    # 精度与 tf32 是影响数值可比性的受控变量，且只有 enable_tf32 之后才是
    # 训练实际取值；紧随 build_run_manifest 补入，保证随后写出的 manifest/
    # checkpoint/台账携带的是生效值而不是配置默认值。
    manifest["precision"] = {
        "amp_enabled": amp_enabled,
        "amp_dtype": (str(amp_dt).replace("torch.", "")
                      if amp_dt is not None else None),
        "autocast": amp_dt is not None,
        # TF32 仅 CUDA 生效；CPU 上记录布尔默认值会被误读成已启用。
        "tf32_matmul": (bool(torch.backends.cuda.matmul.allow_tf32)
                        if str(device).startswith("cuda") else None),
        "tf32_cudnn": (bool(torch.backends.cudnn.allow_tf32)
                       if str(device).startswith("cuda") else None),
    }
    identity = None
    if sanity_info and sanity_info.get("present"):
        # 闸门证据必须和本 run 是同一份代码/配置/数据；旧配置或旧数据的
        # PASS 不能被当作本 run 的通过证据（seed 不同是设计行为，不判失败）。
        identity = sanity_identity_check(sanity_info, manifest)
        sanity_info["identity"] = identity
        manifest["sanity"] = sanity_info
        if identity["hard_mismatches"]:
            print("[train] ⚠ sanity 证据身份不一致（"
                  + ", ".join(identity["hard_mismatches"])
                  + "）：该 PASS 不属于本 run，manifest/eval 按未验证记录")
        elif identity["unknown_fields"]:
            print("[train] ⚠ sanity 证据缺少身份字段（"
                  + ", ".join(identity["unknown_fields"])
                  + "）：旧文件无法核对，PASS 原样保留但不标记为已验证")
    # dirty 工作区：把 diff 补丁随 run 落盘，否则仅凭 commit 无法复原代码
    worktree = manifest.get("worktree") or {}
    if worktree.get("dirty") and worktree.get("diff_bytes"):
        patch = write_worktree_patch(os.path.join(out_dir, "worktree.patch"))
        if patch.get("patch_file"):
            patch["patch_file"] = os.path.basename(patch["patch_file"])
            if patch.get("patch_sha256") != worktree.get("diff_sha256"):
                patch["differs_from_manifest"] = True
            manifest["worktree_patch"] = patch
    save_json(manifest, os.path.join(out_dir, "run_manifest.json"))
    # 本 run 的 eval.json 也继承闸门结论；硬身份不一致时不得冒充通过
    sanity_passed = None
    if (sanity_info and sanity_info.get("present")
            and not (identity and identity["hard_mismatches"])):
        sanity_passed = sanity_info.get("passed")
    provenance = {
        "code_commit": manifest["git"]["commit"],
        "code_dirty": manifest["git"]["dirty"],
        "data_fingerprint": manifest["data"]["digest"],
        "config_sha256": manifest["config_sha256"],
        # checkpoint/eval 自含种子生效证据与实际精度路径：即使 manifest 被
        # 移走，单看权重文件也能核对播种与 bf16/fp32 口径。
        "seed_report": manifest.get("seed_report"),
        "training_precision": manifest["precision"],
        "sanity_passed": sanity_passed,
        # 通过值是否由与本 run 同代码/配置/数据的证据支持（旧文件无法核对
        # 时为 None，而不是假称已验证）
        "sanity_verified": (identity.get("verified")
                            if isinstance(identity, dict) else None),
    }

    block_size = _config_int(cfg["block_size"], "block_size", 1)
    train_ctx = min(_config_int(cfg.get("train_ctx", block_size),
                                "train_ctx", 1), block_size)
    d_model = _config_int(cfg["d_model"], "d_model", 1)
    n_layers = _config_int(cfg["n_layers"], "n_layers", 1)
    n_heads = _config_int(cfg["n_heads"], "n_heads", 1)
    ffn_mult = _config_int(cfg.get("ffn_mult", 4), "ffn_mult", 1)
    steps = _config_int(cfg["steps"], "steps", 1)
    warmup = _config_int(cfg.get("warmup", 100), "warmup", 0)
    grad_accum = _config_int(cfg.get("grad_accum", 1), "grad_accum", 1)
    log_every = _config_int(cfg.get("log_every", 10), "log_every", 1)
    eval_every = _config_int(cfg.get("eval_every", 500), "eval_every", 1)
    eval_iters = _config_int(cfg.get("eval_iters", 50), "eval_iters", 1)
    batch_size = _config_int(cfg["batch_size"], "batch_size", 1)
    # 记录实际生效的数值（含 train_ctx 的 min 截断与字符串/整数值浮点规范化），
    # 保证下游评测复现同一口径，而不是复述未经执行的原始配置。
    cfg_used.update({
        "block_size": block_size, "train_ctx": train_ctx,
        "d_model": d_model, "n_layers": n_layers, "n_heads": n_heads,
        "ffn_mult": ffn_mult, "steps": steps, "warmup": warmup,
        "grad_accum": grad_accum, "log_every": log_every,
        "eval_every": eval_every, "eval_iters": eval_iters,
        "batch_size": batch_size,
    })
    config_used_path = os.path.join(out_dir, "config_used.json")
    with open(config_used_path, "w", encoding="utf-8") as f:
        json.dump(cfg_used, f, ensure_ascii=False, indent=2)
    config_used_sha256 = sha256_file(config_used_path)
    manifest["config_used"] = cfg_used
    manifest["config_used_sha256"] = config_used_sha256
    save_json(manifest, os.path.join(out_dir, "run_manifest.json"))
    provenance["config_used_sha256"] = config_used_sha256
    provenance["resolved_config"] = cfg_used

    model_cfg = dict(
        vocab_size=vocab_size, block_size=block_size,
        d_model=d_model, n_layers=n_layers,
        n_heads=n_heads, ffn_mult=ffn_mult,
        dropout=dropout,
        attn_dropout=attn_dropout,
        variant=args.variant,
        gate_bias_init=gate_bias_init,
        use_sdpa_standard=config_bool(
            cfg.get("use_sdpa_standard", True), "use_sdpa_standard"),
    )
    model = GPT(**model_cfg)
    init_fp = state_dict_fingerprint(model.state_dict())
    manifest["init_state_fingerprint"] = init_fp
    save_json(manifest, os.path.join(out_dir, "run_manifest.json"))
    provenance["init_state_fingerprint"] = init_fp
    model = model.to(device)
    if config_bool(cfg.get("compile", False), "compile") and device.startswith("cuda"):
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
    n_per_head = gate_extra_params_per_head(model)
    print(f"[train] variant={args.variant}({VARIANT_CN[args.variant]}) device={device} "
          f"params={n_params:,} (门控额外 {n_extra:,}"
          + (f"，逐头 {n_per_head:g}" if n_per_head is not None else "")
          + ")")
    print(f"[train] steps={steps} batch={batch_size} ctx={train_ctx} "
          f"tokens/step={batch_size * train_ctx:,} seed={seed}")

    opt = configure_optimizer(
        model, lr=lr, weight_decay=weight_decay,
        fused=(device.startswith("cuda")))
    base_lrs = [g["lr"] for g in opt.param_groups]
    # 梯度裁剪的参数列表与优化器同口径
    clip_params = [p for p in model.parameters() if p.requires_grad]

    amp_ctx = amp_autocast(device, amp_dt is not None)
    # 曲线点数在训练开始前就固定了：采样过稀时过拟合/早停/尖峰诊断会静默
    # 失效，必须在启动日志里明确告警（不阻断，冒烟训练可有意稀疏）。
    curve_warnings = curve_observability_warnings(steps, eval_every, log_every)
    for _w in curve_warnings:
        print(f"[train] ⚠ 曲线可观测性: {_w}")

    def lr_factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(1, warmup)
        prog = min(1.0, (step - warmup) / max(1, steps - 1 - warmup))
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * prog))

    wb = None
    if args.wandb:
        try:
            import wandb
            # 名字含 seed：多种子实验在 W&B 上不互相覆盖；config 同时带
            # 代码版本/数据指纹，远端记录也能回答“这条曲线对应哪份代码/数据”
            wb = wandb.init(project="lega-attention",
                            name=f"{tag}-{args.variant}-seed{seed}",
                            config={**cfg_used, **provenance}, reinit=True)
            print("[train] WandB 已启动")
        except Exception as e:
            print(f"[train] WandB 启动失败（忽略，继续本地训练）: {e}")
            wb = None

    log_path = os.path.join(out_dir, "train_log.jsonl")
    # 同一 out_dir 重跑代表一次新实验：覆盖旧日志，避免 step 轴混入上一轮数据。
    log_f = open(log_path, "w", encoding="utf-8")
    rng = data_rng(seed)
    model.train()
    opt.zero_grad(set_to_none=True)
    # 流水线预取的首批：训练循环内每个 micro-step 再预取下一批
    # 实际采样顺序指纹：与 config/seed 审计互补，证明各变体消费了同一批
    # 训练窗口（同 seed 但采样代码路径/RNG 被提前消耗时 config 审计看不出）。
    data_order_hasher = hashlib.sha256()
    batches = [get_train_batch(train, train_ctx, batch_size, device, rng,
                               order_hasher=data_order_hasher)
               for _ in range(grad_accum)]
    tokens_seen = 0
    best_val = float("inf")
    last_eval_step = None
    last_eval_metrics = None
    # 早停状态：patience=0（默认）时功能关闭，但计数仍维护并写入日志；
    # stop_step 非 None 表示本次 run 被早停终止（曲线按设计截断）。
    evals_since_improve = 0
    stop_step = None
    # loss 设备端累加器（每步 zero_()）
    loss_acc_dev = torch.zeros((), device=device)
    # pre-clip 梯度范数：设备标量只保留引用，log 点才 .item()；
    # 历史尾部供 NaN 排查判断"梯度爆炸 vs 学习率过大"
    gn_dev_recent = []
    gn_hist = []
    # log 区间峰值：只记录区间最后一步会漏掉区间中段的梯度爆炸/单步损失
    # 尖峰。用设备标量累积，log 点才 .item()，不引入每步同步。
    gn_max_dev = torch.full((), -1.0, device=device)
    gn_max_step_dev = torch.full((), -1.0, device=device)
    gn_nonfinite_dev = torch.zeros((), dtype=torch.bool, device=device)
    loss_max_dev = torch.full((), -1.0, device=device)
    loss_max_step_dev = torch.full((), -1.0, device=device)
    # 最近训练日志损失：验证停步时用来区分过拟合（训练仍在降）与
    # 欠拟合/优化不足（训练也停滞或上升），见 curves.live_overfit_hint。
    train_loss_hist = []
    # NaN 现场按 log 区间保留：NaN 在 log 点才被检测到（区间累计损失），真正
    # 诱发的批次可能在本区间任意一步，而预取早已把 batches 覆盖成后续数据。
    # interval_batches 保存每一步消费的 micro-batch 引用（≤ log_every 步，
    # log 点清空）；interval_first_bad_dev 在设备端标记区间内首个非有限损失步
    # （0-based，-1=无；不每步同步），_nan_guard 触发时把整个区间的候选批次
    # （带步标签）交给排查，而不是用"最后一步批次"冒充失败步批。
    interval_batches = []   # [(0-based step, [consumed micro-batch, ...])]
    interval_first_bad_dev = torch.full((), -1.0, device=device)
    # 当前训练步的 0 维设备标量：预分配后每步 fill_。
    step_dev = torch.zeros((), device=device)
    # 步时计时：按 log 区间累计墙钟并扣除评测时间（.item() 已同步设备，
    # 区间边界无需额外 synchronize）。pending_eval_s 供 do_eval 累加。
    step_times_ms: list = []
    interval_t0 = time.perf_counter()
    interval_steps = 0
    pending_eval_s = 0.0

    def _interval_candidates():
        """本 log 区间内每一步消费的训练批次及对应日志步号（1-based）。"""
        consumed, labels = [], []
        for s0, bs in interval_batches:
            for b in bs:
                consumed.append(b)
                labels.append(s0 + 1)
        return consumed, labels

    def _nan_guard(where: str, value, step: int, make_consumed=None,
                   first_bad_step=None, interval=None):
        """log/eval 点检查损失与梯度范数的有限性，异常即触发系统排查。

        make_consumed：按需构造"失败输入批次"的函数（val_loss 时应给出真实
        验证窗口）；缺省用本 log 区间内全部候选训练批次（含失败步）。
        first_bad_step/interval：设备端标记的首个非有限损失步与区间范围，
        让报告能区分"检测点"与"失败点"。
        """
        if value is not None and not math.isfinite(float(value)):
            labels = None
            if make_consumed is not None:
                consumed = [b for b in make_consumed() if b is not None]
            else:
                consumed, labels = _interval_candidates()
            report = run_nan_checklist(
                model, batches, device, step, where, value,
                lr=base_lrs[0] * lr_factor(max(0, step - 1)),
                grad_norm_history=gn_hist, amp_dtype=amp_dt,
                vocab_size=vocab_size,
                consumed_batches=consumed, consumed_step_labels=labels,
                peak_lr=base_lrs[0], first_bad_step=first_bad_step,
                interval=interval, optimizer=opt)
            _abort_on_nan(report, out_dir)

    def do_eval(step):
        nonlocal best_val, last_eval_step, last_eval_metrics, pending_eval_s, \
            evals_since_improve, stop_step
        t_eval = time.perf_counter()
        metrics = evaluate_val(model, val, train_ctx, device, eval_iters, amp=amp_dt is not None)
        pending_eval_s += time.perf_counter() - t_eval
        last_eval_step, last_eval_metrics = step, metrics
        vl = metrics["val_loss"]
        # val 损失非有限时用真实验证窗口做数据/激活排查（本区间训练批次只是回退）
        _nan_guard("val_loss", vl, step,
                   make_consumed=lambda: val_audit_batch(
                       val, train_ctx, eval_iters, device,
                       fallback=_interval_candidates()[0]))
        gmean = metrics["gate_mean"]
        ppl = math.exp(min(20, vl))
        prev_best = best_val
        best_val, evals_since_improve, should_stop = early_stop_update(
            vl, best_val, evals_since_improve, early_stop_min_delta,
            early_stop_patience, val_ci95=metrics.get("val_loss_ci95"))
        gap_s = None
        if math.isfinite(prev_best) and prev_best > 0:
            gap = (vl - prev_best) / prev_best
            if math.isfinite(gap):
                gap_s = gap
        rec = {"step": step, **metrics, "val_ppl": ppl,
               "val_ppl_clamped": bool(vl > 20), "time": now_str(),
               "evals_since_best": evals_since_improve,
               # 当前 val 相对历史最优的差距：曲线诊断的原材料，训练中途
               # 的预警也与它同口径（见 curves.live_overfit_hint）
               "val_vs_best_rel": gap_s}
        # 只在每个"未改善连击"的第 LIVE_WARN_EVALS 次提示一次（改善后计数
        # 归零，下轮连击会再次提示），避免逐评测刷屏
        hint = (live_overfit_hint(evals_since_improve, prev_best, vl,
                                  early_stop_patience,
                                  train_recent=train_loss_hist[-5:],
                                  val_ci95=metrics.get("val_loss_ci95"))
                if evals_since_improve == LIVE_WARN_EVALS else None)
        if hint:
            rec["overfit_hint"] = hint
        if should_stop:
            rec["early_stop_trigger"] = True
        log_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        log_f.flush()
        mark = ""
        if vl < prev_best:
            mark = " (best)"
            # 同步保存最优 checkpoint，避免下游评测只能用到最终步模型
            save_checkpoint(os.path.join(out_dir, "model_best.pt"), model,
                            extra={"variant": args.variant, "step": step,
                                   "steps_planned": steps,
                                   "seed": seed, "val_loss": vl,
                                   "val_ppl": math.exp(min(20, vl)),
                                   "val_ppl_clamped": bool(vl > 20),
                                   **provenance})
        if should_stop:
            stop_step = step
            mark += (f" [early-stop: {evals_since_improve} 次评测无实质改善；"
                     f"best 在更早的 eval，checkpoint 用 model_best.pt]")
        print(f"[train] step {step}: val_ppl={ppl:.3f}"
              + (f" gate_mean={gmean:.3f}" if gmean is not None else "") + mark)
        if hint:
            print(f"[train] ⚠ {hint}")
        if wb:
            wb.log({"val_ppl": ppl, "val_loss": vl,
                    **({"gate_mean": gmean} if gmean is not None else {})},
                   step=step)
        return vl

    for step in range(steps):
        # 梯度保留到本步排查完成后再清（下一步开始时清）：失败现场的
        # p.grad 是反向溢出的直接证据，提前 zero_grad 会让 NaN 排查只能看到
        # 已被优化器写坏的参数。首步在循环前已 zero_grad。
        if step > 0:
            opt.zero_grad(set_to_none=True)
        interval_steps += 1
        lr_f = lr_factor(step)
        for g, base in zip(opt.param_groups, base_lrs):
            g["lr"] = base * lr_f

        # 在设备上累积 loss；按 log 区间快照本步将消费的批次（下一步预取会
        # 覆盖 batches，快照保留引用供 NaN 排查定位"哪一步的数据出问题"）
        interval_batches.append((step, [b for b in batches if b is not None]))
        loss_acc_dev.zero_()
        step_dev.fill_(step)
        for mi in range(grad_accum):
            x, y = batches[mi]
            # 预取下一 micro-batch：在当前步前向/反向之前发起 memmap 收集与
            # pinned H2D 提交（_stage_to_device 乒乓暂存双槽 + event 保护，
            # 两批在途互不干扰）。rng 抽取顺序与逐次采样完全一致，确定性不变。
            if step + 1 < steps or mi + 1 < grad_accum:
                batches[mi] = get_train_batch(
                    train, train_ctx, batch_size, device, rng,
                    order_hasher=data_order_hasher)
            else:
                batches[mi] = None
            with amp_ctx:
                out = model(x, targets=y)
                # grad_accum==1（默认配置）时除法结果就是 loss 本身
                loss = out["loss"] if grad_accum == 1 else out["loss"] / grad_accum
            step_loss = out["loss"].detach()
            # 设备端标记区间内首个非有限损失步（0-based），log 点统一读取。
            bad_now = ~torch.isfinite(step_loss)
            interval_first_bad_dev = torch.where(
                (interval_first_bad_dev < 0) & bad_now,
                step_dev, interval_first_bad_dev)
            # 区间内单步损失峰值：区间均值会把“一步异常”稀释到阈值以下，
            # 峰值序列供 curves 的瞬态尖峰诊断单独检查。
            finite_step_loss = torch.isfinite(step_loss)
            new_loss_max = finite_step_loss & (
                (loss_max_dev < 0) | (step_loss > loss_max_dev))
            loss_max_dev = torch.where(new_loss_max, step_loss, loss_max_dev)
            loss_max_step_dev = torch.where(new_loss_max, step_dev,
                                            loss_max_step_dev)
            loss.backward()
            loss_acc_dev.add_(step_loss, alpha=1.0 / grad_accum)
            # backward 已消费完计算图；解除对 out（含逐层 past_kv）与 loss 的引用
            del out, loss
        # pre-clip 梯度范数：clip_grad_norm_ 返回裁剪前的总范数，保留设备标量
        gn_dev = torch.nn.utils.clip_grad_norm_(clip_params, grad_clip)
        gn_dev_recent.append(gn_dev)
        if len(gn_dev_recent) > log_every + grad_accum + 8:
            gn_dev_recent.pop(0)
        # pre-clip 梯度范数区间峰值（含非有限值）：log 点最后一步之外的一次
        # 梯度爆炸同样是 NaN 之前的信号，不能被区间采样漏掉。
        gn_finite = torch.isfinite(gn_dev)
        new_gn_max = gn_finite & ((gn_max_dev < 0) | (gn_dev > gn_max_dev))
        gn_max_dev = torch.where(new_gn_max, gn_dev, gn_max_dev)
        gn_max_step_dev = torch.where(new_gn_max, step_dev, gn_max_step_dev)
        gn_nonfinite_dev = gn_nonfinite_dev | (~gn_finite)
        opt.step()
        tokens_seen += batch_size * train_ctx * grad_accum

        if (step + 1) % log_every == 0 or step == 0:
            loss_acc = float(loss_acc_dev.item())
            # 设备端标记的区间内首个非有限损失步（0-based）→ 1-based 日志步
            first_bad_local = int(interval_first_bad_dev.item())
            first_bad_step = (first_bad_local + 1 if first_bad_local >= 0
                              else None)
            interval = ((interval_batches[0][0] + 1, step + 1)
                        if interval_batches else None)
            gn_last = float(gn_dev_recent[-1].item()) if gn_dev_recent else None
            gn_max = (float(gn_max_dev.item())
                      if bool((gn_max_dev >= 0).item()) else None)
            gn_nonfinite = bool(gn_nonfinite_dev.item())
            gn_max_step = (int(gn_max_step_dev.item()) + 1
                           if gn_max_step_dev >= 0 else None)
            loss_max = (float(loss_max_dev.item())
                        if bool((loss_max_dev >= 0).item()) else None)
            loss_max_step = (int(loss_max_step_dev.item()) + 1
                             if loss_max_step_dev >= 0 else None)
            # .item() 已同步本区间全部前向/反向/优化器内核，
            # 此处墙钟差为区间步时（扣除评测）
            t_now = time.perf_counter()
            dt = t_now - interval_t0 - pending_eval_s
            step_ms = (dt / interval_steps * 1000.0
                       if interval_steps > 0 and dt > 0 else None)
            if step_ms is not None:
                step_times_ms.append(step_ms)
            interval_t0 = t_now
            interval_steps = 0
            pending_eval_s = 0.0
            # 历史保留区间峰值（有限时）：NaN 排查看“区间内最坏梯度”而不是
            # 恰好落在 log 边界的最后一步。
            gn_hist_val = (gn_max if (gn_max is not None
                                      and math.isfinite(gn_max)) else gn_last)
            if gn_hist_val is not None and math.isfinite(gn_hist_val):
                gn_hist.append(gn_hist_val)
                if len(gn_hist) > 200:
                    gn_hist.pop(0)
            _nan_guard("train_loss", loss_acc, step + 1,
                       first_bad_step=first_bad_step, interval=interval)
            # 区间内任一步非有限时即使最后一步已恢复也要中止（梯度已爆炸）。
            if gn_nonfinite or (gn_last is not None
                                and not math.isfinite(gn_last)):
                gn_guard = (gn_max if (gn_max is not None
                                       and not math.isfinite(gn_max))
                            else float("inf"))
            else:
                gn_guard = gn_max if gn_max is not None else gn_last
            _nan_guard("grad_norm", gn_guard, step + 1,
                       first_bad_step=first_bad_step, interval=interval)
            # 本区间现场已检查完毕（异常路径已在中止前写出报告）：清空，
            # 避免跨区间串批、控制引用数量 ≤ log_every 步
            interval_batches.clear()
            interval_first_bad_dev = torch.full_like(interval_first_bad_dev, -1.0)
            train_loss_hist.append(loss_acc)
            if len(train_loss_hist) > 20:
                train_loss_hist.pop(0)
            rec = {"step": step + 1, "loss": loss_acc, "ppl": math.exp(min(20, loss_acc)),
                   "lr": base_lrs[0] * lr_f, "time": now_str()}
            if gn_last is not None:
                rec["grad_norm"] = gn_last
            if gn_max is not None:
                rec["grad_norm_max"] = gn_max
                rec["grad_norm_max_step"] = gn_max_step
            if loss_max is not None:
                rec["loss_max"] = loss_max
                rec["loss_max_step"] = loss_max_step
            if step_ms is not None:
                rec["step_time_ms"] = round(step_ms, 3)
            log_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            log_f.flush()
            print(f"[train] step {step+1}/{steps} loss={loss_acc:.4f} "
                  f"ppl={math.exp(min(20, loss_acc)):.3f} lr={base_lrs[0]*lr_f:.2e}"
                  + (f" gnorm={gn_last:.2f}" if gn_last is not None else ""))
            if wb:
                wb.log({"loss": loss_acc, "lr": base_lrs[0] * lr_f,
                        **({"grad_norm": gn_last} if gn_last is not None else {}),
                        **({"grad_norm_max": gn_max}
                           if gn_max is not None else {}),
                        **({"loss_max": loss_max}
                           if loss_max is not None else {})},
                       step=step + 1)
            # 区间统计已落盘，清空峰值以统计下一个 log 区间。
            gn_max_dev.fill_(-1.0)
            gn_max_step_dev.fill_(-1.0)
            gn_nonfinite_dev.fill_(False)
            loss_max_dev.fill_(-1.0)
            loss_max_step_dev.fill_(-1.0)
        if (step + 1) % eval_every == 0:
            do_eval(step + 1)
            # 早停：验证连续 patience 次评测无实质改善即终止（下游评测
            # 本就用 val 选出的 model_best.pt）
            if stop_step is not None:
                break

    # stop_step==steps 说明触发点恰在最后一次评测：其实已跑满计划步数，
    # 不算"提前终止"，避免把完整训练的 run 标成早停。
    stopped_early = stop_step is not None and stop_step < steps
    final_step = stop_step if stopped_early else steps
    if last_eval_step == final_step:
        final_metrics = last_eval_metrics
    else:
        final_metrics = evaluate_val(model, val, train_ctx, device, eval_iters, amp=amp_dt is not None)
        _nan_guard("val_loss(final)", final_metrics["val_loss"], steps,
                   make_consumed=lambda: val_audit_batch(
                       val, train_ctx, eval_iters, device,
                       fallback=_interval_candidates()[0]))
        # 最终步评测必须同样参与 best-checkpoint 选择，与训练循环内口径一致：
        # 否则 steps 非 eval_every 整数倍（或 steps < eval_every）时，更优的最终
        # 状态不会进 model_best.pt，下游 run_all 会静默回退到最终步模型，却仍按
        # “val 上选出的最优 checkpoint”口径报告。
        improved_for_ckpt = final_metrics["val_loss"] < best_val
        # 最终评测也要更新耐心计数（改善归零、未改善累加），否则最后一条
        # 日志会出现 val_vs_best_rel=0 却 evals_since_best 仍为旧值的矛盾。
        # 必须用本次评测前的历史最优做更新：若先写 best_val 再调用，刷新最优
        # 的最终评测会被误判为“未改善”而把计数加一。patience 传 0：训练
        # 此刻已结束，早停状态只用于如实记录。
        best_val, evals_since_improve, _ = early_stop_update(
            final_metrics["val_loss"], best_val, evals_since_improve,
            early_stop_min_delta, 0,
            val_ci95=final_metrics.get("val_loss_ci95"))
        if improved_for_ckpt:
            save_checkpoint(os.path.join(out_dir, "model_best.pt"), model,
                            extra={"variant": args.variant, "step": steps,
                                   "steps_planned": steps,
                                   "seed": seed,
                                   "val_loss": final_metrics["val_loss"],
                                   "val_ppl": math.exp(
                                       min(20, final_metrics["val_loss"])),
                                   "val_ppl_clamped": bool(
                                       final_metrics["val_loss"] > 20),
                                   **provenance})
    final_val_loss = final_metrics["val_loss"]
    final_ppl = math.exp(min(20, final_val_loss))
    gmean = final_metrics["gate_mean"]
    # 训练数据覆盖率：tokens_seen 是“见过的 token 总数”，除以训练集大小
    # 得到约等价 epoch 数（随机窗口采样，不是严格的一遍遍历）。它回答的是
    # “曲线下降是否只是还没看完数据”，与 eval_iters 的验证覆盖率互补。
    train_tokens = int(len(train))
    epochs_seen = (tokens_seen / train_tokens) if train_tokens else None
    data_order_sha256 = data_order_hasher.hexdigest()
    # 步时汇总：区间步时的中位数，抗首区间冷启动与偶发卡顿。
    tokens_per_step = batch_size * train_ctx * grad_accum
    train_step_ms = (statistics.median(step_times_ms)
                     if step_times_ms else None)
    train_tokens_per_sec = (tokens_per_step * 1000.0 / train_step_ms
                            if train_step_ms else None)
    # 训练进程级高水位统计（CUDA 设备；其他设备为 None）。
    train_peak_mem_mb = None
    if device.startswith("cuda") and torch.cuda.is_available():
        try:
            train_peak_mem_mb = round(
                torch.cuda.max_memory_allocated() / (1024.0 ** 2), 1)
        except (RuntimeError, ValueError):
            train_peak_mem_mb = None
    save_checkpoint(os.path.join(out_dir, "model.pt"), model,
                    extra={"variant": args.variant, "step": final_step,
                           "steps_planned": steps,
                           "early_stopped": stopped_early,
                           "seed": seed,
                           "val_ppl": final_ppl, "val_loss": final_val_loss,
                           "val_ppl_clamped": bool(final_val_loss > 20),
                           "tokens_seen": tokens_seen,
                           "train_tokens": train_tokens,
                           "epochs_seen": (round(epochs_seen, 4)
                                           if epochs_seen is not None else None),
                           "data_order_sha256": data_order_sha256,
                           "n_params": n_params, "n_gate_extra": n_extra,
                           **provenance})
    save_json({"variant": args.variant, "val_ppl": final_ppl, "val_loss": final_val_loss,
               "val_ppl_clamped": bool(final_val_loss > 20),
               "gate_mean": gmean,
               "gate_std": final_metrics.get("gate_std"),
               "gate_frac_low": final_metrics.get("gate_frac_low"),
               "gate_frac_high": final_metrics.get("gate_frac_high"),
               "gate_layer_mean": final_metrics.get("gate_layer_mean"),
               **{k: final_metrics[k] for k in
                                      ("eval_iters_requested", "eval_iters",
                                       "val_windows", "val_tokens_evaluated",
                                       "val_loss_se", "val_loss_ci95",
                                       "val_ppl_ci95", "val_loss_min",
                                       "val_loss_max")},
               "n_params": n_params, "n_gate_extra": n_extra,
               "train_step_time_ms": (round(train_step_ms, 3)
                                      if train_step_ms else None),
               "train_tokens_per_sec": (round(train_tokens_per_sec, 1)
                                        if train_tokens_per_sec else None),
               "train_peak_mem_mb": train_peak_mem_mb,
               "tokens_seen": tokens_seen,
               "train_tokens": train_tokens,
               "epochs_seen": (round(epochs_seen, 4)
                               if epochs_seen is not None else None),
               "data_order_sha256": data_order_sha256,
               # steps=实际运行步数（早停时小于计划）；steps_planned=计划步数。
               # 受控变量审计按 steps_planned 核对，实际步数差异由 early_stopped
               # 如实暴露，而不是静默伪装成跑满了计划步数。
               "steps": final_step, "steps_planned": steps,
               "early_stopped": stopped_early,
               "early_stop_patience": early_stop_patience,
               "early_stop_min_delta": early_stop_min_delta,
               "curve_observability_warnings": curve_warnings,
               "seed": seed,
               **provenance},
              os.path.join(out_dir, "eval.json"))
    final_gap = None
    if (final_val_loss is not None and math.isfinite(float(final_val_loss))
            and math.isfinite(best_val) and best_val > 0):
        g = (float(final_val_loss) - best_val) / best_val
        final_gap = g if math.isfinite(g) else None
    final_rec = {"step": final_step, **final_metrics,
                 "val_ppl": final_ppl,
                 "val_ppl_clamped": bool(final_val_loss > 20),
                 "final": True, "evals_since_best": evals_since_improve,
                 "val_vs_best_rel": final_gap}
    if stopped_early:
        final_rec["early_stop"] = True
    log_f.write(json.dumps(final_rec, ensure_ascii=False) + "\n")
    log_f.close()
    if stopped_early:
        print(f"[train] 早停于 step {final_step}/{steps}（连续 "
              f"{evals_since_improve} 次评测无实质改善）；"
              f"下游使用 model_best.pt")
    print(f"[train] 完成: val_ppl={final_ppl:.3f} -> {out_dir}")
    if wb:
        wb.finish()


if __name__ == "__main__":
    main()
