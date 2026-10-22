"""训练后因果病灶消融（post-training causal lesion study）。

科研动机：训练期消融（全因子 + 去一法）回答“**重新训练**时去掉某个组件
会怎样”，但它改变的是优化轨迹，不能回答“训练好的模型在推理时是否真的
**使用**该组件”。两者可以不一致：某个组件在训练期有用，可能只是因为
它改变了早期优化/正则化，而最终前向已经把它学到接近 0；反之，推理期
清零一个很小的系数也可能没有可测影响。把“训练期消融”当成“机制证据”
会混淆这两种因果问题。

本模块在**同一个 checkpoint**、**同一批验证窗口**上做权重病灶干预，
逐窗口配对比较干预前后的 NLL：

  组件病灶（与训练期 A/B/C 对应的推理期版本）
    no_query   : gate_w := 0                 （去 A：查询项）
    no_conc    : gate_a := 0                 （去 B：集中度项）
    no_len     : gate_b := 0                 （去 C：长度项）
    no_evidence: gate_w, gate_a, gate_b := 0 （A+B+C 全去，只留学习到的 d）
    query_only : gate_a, gate_b := 0          （仅 A：w·q + d）
    conc_only  : gate_w, gate_b := 0          （仅 B：a·c + d）
    len_only   : gate_w, gate_a := 0          （仅 C：b·log n + d）

  上述 8 个格子（含正常基线 ABC）构成**同一 checkpoint 内的 2^3 全因子
  干预设计**：训练期全因子消融回答“重新训练时去掉组件会怎样”，这里则
  直接干预训练好的门控，回答“最终前向的组件结构与交互是否一致”，并可在
  同一 checkpoint 上做推理期 Shapley 分解。

  门控强度/信息病灶
    gate_open        : 所有证据系数 := 0，d := +30（g→1，等价于不门控）
    gate_closed      : 所有证据系数 := 0，d := −30（g→0，注意力输出被关闭）
    gate_constant_mean: 证据系数 := 0，d := logit(观测平均 g)
                        （保留平均写入强度，去掉逐位置/逐查询变化）
    head_permute     : 每层把门参数在头之间置换（容量与参数分布完全不变，
                        破坏“头—证据”对应关系；天然安慰剂对照）
    sign_flip        : 证据系数整体反号（保留幅度，反转证据使用方向）

所有干预都只改门控参数，不改注意力/主干；评测沿用 `train.evaluate_val`
的同一窗口规格与精度路径，因此干预前后的窗口损失逐窗口配对。默认只在
**val** split 上做：test 已尘封，且本分析是探索性机制检查，不能为了多一个
数字反复访问 test。

输出 `ablation_<variant>[_seed<seed>].json`，包含基线/每个病灶的 val_loss、
PPL、门控均值/标准差/饱和比例、逐窗口 ΔNLL、窗口级配对 t/符号检验，以及
溯源字段。跨种子的正式配对检验由 `run_all` 汇总（窗口级 p 仅描述性：
相邻窗口可能相关，不能当作独立样本的确认性证据）。

用法:
  python -m lega.ablate --ckpt out/small/lega/seed1337/model_best.pt \\
      --data_dir data/wikitext103 --out results/small/ablation_lega_seed1337.json
"""
import argparse
import json
import math
import os
import sys
from typing import Dict, List, Mapping, Optional

import numpy as np
import torch

from .data import get_val_windows, load_bin
from .evaluate import (_positive_int_or_none, _resolve_eval_amp,
                       json_safe_eval_metrics)
from .model import load_model
from .preflight import (allow_protocol_violations,
                        data_protocol_preflight)
from .repro import load_provenance, sha256_file
from .train import _val_eval_indices, evaluate_val
from .utils import (amp_autocast, bootstrap_mean_ci, enable_tf32,
                    get_device, now_str, paired_sign_test, paired_t_test,
                    save_json, seed_int, set_seed, strict_int)

# 病灶定义（单一事实来源，run_all 汇总复用标签/目标/顺序）。
# requires：需要的门控能力（query/conc/len/gated），任一缺失则该病灶不适用；
# expected：若目标机制在推理期确实被使用，ΔNLL 的预期符号（"worse"=变差）。
LESION_SPECS = (
    {
        "id": "no_query", "label": "去查询项 A（gate_w=0）",
        "target": "A 查询项 w·q", "kind": "zero",
        "params": ("gate_w",), "requires": ("query",),
        "expected": "worse",
        "note": "推理期等价于 Full−A；与训练期去一法 lega_no_query 对照",
    },
    {
        "id": "no_conc", "label": "去集中度项 B（gate_a=0）",
        "target": "B 集中度证据 a·c", "kind": "zero",
        "params": ("gate_a",), "requires": ("conc",),
        "expected": "worse",
        "note": "推理期等价于 Full−B；与训练期去一法 lega_no_conc 对照",
    },
    {
        "id": "no_len", "label": "去长度项 C（gate_b=0）",
        "target": "C 长度校准 b·log n", "kind": "zero",
        "params": ("gate_b",), "requires": ("len",),
        "expected": "worse",
        "note": "推理期等价于 Full−C：只去 b·log n，c 中的 log n 仍随 "
                "a·c 保留（与训练期单因素 C 的定义一致，不能解读为"
                "“完全移除长度依赖”）",
    },
    {
        "id": "no_evidence", "label": "去全部证据项 A+B+C（只留 d）",
        "target": "全部证据输入", "kind": "zero_evidence",
        "params": ("gate_w", "gate_a", "gate_b"),
        "requires": ("any_evidence",),
        "expected": "worse",
        "note": "推理期退化为 gate_only：保留学习到的均值门偏置，去掉全部"
                "查询/注意力/长度依赖",
    },
    {
        "id": "gate_open", "label": "门恒开（g→1）",
        "target": "门控写回本身", "kind": "constant",
        "params": (), "requires": ("gated",),
        "expected": "worse",
        "note": "若显著变差，说明门在抑制某些位置的写回；若不变，门控"
                "接近恒等，方法退化为标准注意力",
    },
    {
        "id": "gate_closed", "label": "门恒关（g→0）",
        "target": "门控写回本身", "kind": "closed",
        "params": (), "requires": ("gated",),
        "expected": "worse",
        "note": "阳性对照：关闭注意力输出应显著变差；若不差，说明主干"
                "完全不依赖该层注意力，评测/实现需排查",
    },
    {
        "id": "gate_constant_mean", "label": "门恒为观测均值",
        "target": "逐位置门控变化", "kind": "constant_mean",
        "params": (), "requires": ("gated",),
        "expected": "worse",
        "note": "保留每个头/层的观测平均写入强度，去掉逐 query/位置的 g "
                "变化（头间均值差异保留）：检验收益是否来自“按位置调节”"
                "而非整体缩放",
    },
    {
        "id": "head_permute", "label": "门参数在头间置换（安慰剂）",
        "target": "头—证据对应关系", "kind": "head_permute",
        "params": (), "requires": ("gated", "multi_head"),
        "expected": "worse",
        "note": "参数个数、数值分布、门控结构完全不变，只打乱头与证据的"
                "对应；容量解释无法解释显著变差",
    },
    {
        "id": "sign_flip", "label": "证据系数反号",
        "target": "证据使用方向", "kind": "sign_flip",
        "params": ("gate_w", "gate_a", "gate_b"),
        "requires": ("any_evidence",),
        "expected": "worse",
        "note": "保留幅度与容量、反转证据方向；若不变，门控对证据方向不敏感",
    },
    # 以下三个“只保留单因子”格用于同一 checkpoint 的 2^3 全因子/Shapley；
    # 它们相对 Full 同时去掉另外两个因子，故标记 factorial_cell，不进入
    # 单病灶 ΔNLL 因果表（见 run_all._ablation_lesion_records）。
    {
        "id": "query_only", "label": "仅查询项 A（gate_a=gate_b=0）",
        "target": "A 查询项 w·q", "kind": "zero", "factorial_cell": True,
        "params": ("gate_a", "gate_b"), "requires": ("query",),
        "expected": "worse",
        "note": "推理期全因子 2^3 的 A 格：保留 w·q + d，清零 B/C；"
                "与训练期 qgate 对照（若模型还含 B/C，二者不再等价）",
    },
    {
        "id": "conc_only", "label": "仅集中度项 B（gate_w=gate_b=0）",
        "target": "B 集中度证据 a·c", "kind": "zero", "factorial_cell": True,
        "params": ("gate_w", "gate_b"), "requires": ("conc",),
        "expected": "worse",
        "note": "推理期全因子 2^3 的 B 格：保留 a·c + d，清零 A/C；"
                "与训练期 lega_conc_only 对照",
    },
    {
        "id": "len_only", "label": "仅长度项 C（gate_w=gate_a=0）",
        "target": "C 长度校准 b·log n", "kind": "zero", "factorial_cell": True,
        "params": ("gate_w", "gate_a"), "requires": ("len",),
        "expected": "worse",
        "note": "推理期全因子 2^3 的 C 格：保留 b·log n + d，清零 A/B；"
                "与训练期 lega_len_only 对照",
    },
)
LESION_BY_ID = {spec["id"]: spec for spec in LESION_SPECS}
LESION_ORDER = tuple(spec["id"] for spec in LESION_SPECS)

# 同一 checkpoint 内的 2^3 全因子干预设计：cell -> 病灶 id。
# "ABC" 是正常基线（不施加病灶），None 表示调用方读取 baseline 指标。
# 映射与 lega.factorial.CELL_ORDER 一致，供推理期 Shapley 分解复用。
INFERENCE_FACTORIAL_CELLS = {
    "none": "no_evidence",   # 只留 d
    "A": "query_only",       # w·q + d
    "B": "conc_only",        # a·c + d
    "C": "len_only",         # b·log n + d
    "AB": "no_len",          # Full−C
    "AC": "no_conc",         # Full−B
    "BC": "no_query",        # Full−A
    "ABC": None,             # 正常前向（基线）
}
INFERENCE_FACTORIAL_ORDER = tuple(INFERENCE_FACTORIAL_CELLS)



def _finite_or_none(value):
    """严格有限浮点；bool/下划线数字串/NaN/Inf/坏类型返回 None。

    门控干预的参数可能来自外部 JSON（逐头均值等）。与项目其他证据解析入口
    同口径：``float("1_0") == 10.0``、``np.bool_(True)`` 经 float() 变成 1.0
    都会把损坏/伪造的值伪装成合法门控强度。
    """
    if isinstance(value, (bool, np.bool_)):
        return None
    if isinstance(value, str) and "_" in value:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return f if math.isfinite(f) else None


def gate_modules(model) -> List:
    """返回模型中带输出门的注意力模块；无门变体返回空列表。"""
    out = []
    for block in getattr(model, "blocks", []):
        attn = getattr(block, "attn", None)
        if attn is not None and hasattr(attn, "gate_d"):
            out.append(attn)
    return out


def gate_capabilities(model) -> Dict:
    """探测模型实际注册了哪些门控项（以参数是否存在为准）。

    不能只看 `variant` 名字：旧 checkpoint/手工构造模型可能名实不符，
    病灶适用性必须由真实参数决定，否则会把“不存在的项”当成“清零成功”。
    """
    mods = gate_modules(model)
    caps = {
        "gated": bool(mods),
        "n_layers": len(mods),
        "n_heads": int(getattr(mods[0], "n_heads", 0)) if mods else 0,
        "query": bool(mods) and all(hasattr(m, "gate_w") for m in mods),
        "conc": bool(mods) and all(hasattr(m, "gate_a") for m in mods),
        "len": bool(mods) and all(hasattr(m, "gate_b") for m in mods),
    }
    caps["any_evidence"] = caps["query"] or caps["conc"] or caps["len"]
    caps["multi_head"] = caps["n_heads"] > 1
    return caps


def lesion_applicability(model, lesion_id: str):
    """返回 (applicable, reason, required_capabilities)。"""
    spec = LESION_BY_ID.get(lesion_id)
    if spec is None:
        raise ValueError(f"未知病灶: {lesion_id!r}")
    caps = gate_capabilities(model)
    missing = [r for r in spec["requires"] if not caps.get(r)]
    if missing:
        return False, f"模型缺少门控能力: {', '.join(missing)}", spec["requires"]
    return True, "", spec["requires"]


def snapshot_gate_params(model) -> Dict[str, torch.Tensor]:
    """克隆全部门控参数（按完整参数名），供病灶后逐位恢复。"""
    snap = {}
    for name, p in model.named_parameters():
        if ".gate_" in name or name.endswith(("gate_w", "gate_a",
                                              "gate_b", "gate_d")):
            snap[name] = p.detach().clone()
    return snap


def restore_gate_params(model, snapshot: Mapping[str, torch.Tensor]) -> None:
    """把门控参数恢复到快照；缺失/形状不符显式报错（不能静默留下病灶）。

    形状不符时 ``copy_`` 会按广播规则写入而不报错，可能只改一部分门控值；
    恢复失败必须显式暴露，不能把被污染的模型继续用于后续病灶评测。
    """
    if not snapshot:
        raise ValueError("门控参数快照为空：该模型没有可恢复的门控参数")
    seen = set()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name in snapshot:
                value = snapshot[name]
                if not torch.is_tensor(value):
                    raise ValueError(
                        f"恢复门控参数失败：快照 {name} 不是张量"
                        f"（{type(value).__name__}）")
                if tuple(value.shape) != tuple(p.shape):
                    raise ValueError(
                        f"恢复门控参数失败：{name} 形状不一致 "
                        f"（快照 {tuple(value.shape)} vs 模型 {tuple(p.shape)}）")
                p.copy_(value)
                seen.add(name)
    missing = sorted(set(snapshot) - seen)
    if missing:
        raise RuntimeError(f"恢复门控参数失败，以下参数不存在: {missing}")


def _layer_permutation(n_heads: int, seed: int, layer: int):
    """确定性的非恒等头置换（恒等置换会让“安慰剂”变成空操作）。

    置换种子必须是合法的非负整数：``int(1.5)`` 会截断成 1，``int(True)``
    会变成 1，使两次声明不同的置换实际相同，安慰剂对照的可复现性失真。
    """
    n_heads = strict_int(n_heads, "n_heads")
    if n_heads < 1:
        raise ValueError(f"n_heads 必须 >= 1，收到 {n_heads}")
    seed = strict_int(seed, "perm_seed")
    if seed < 0:
        raise ValueError(f"perm_seed 必须 >= 0，收到 {seed}")
    layer = strict_int(layer, "layer")
    rng = np.random.default_rng(seed + 1000003 * layer)
    perm = rng.permutation(n_heads)
    if n_heads > 1 and np.array_equal(perm, np.arange(n_heads)):
        perm = np.roll(np.arange(n_heads), 1)
    return perm


def apply_lesion(model, lesion_id: str, *, gate_mean: Optional[float] = None,
                 gate_head_mean: Optional[List[List[float]]] = None,
                 perm_seed: int = 0) -> Dict:
    """就地施加指定病灶；返回应用记录（是否生效、改变了多少）。

    调用方负责先 `snapshot_gate_params` 并在评测后 `restore_gate_params`。
    本函数不捕获异常：中途失败时参数可能已部分修改，调用方仍应恢复快照。
    """
    spec = LESION_BY_ID.get(lesion_id)
    if spec is None:
        raise ValueError(f"未知病灶: {lesion_id!r}")
    applicable, reason, _requires = lesion_applicability(model, lesion_id)
    if not applicable:
        return {"applied": False, "reason": reason,
                "changed_tensors": 0, "max_abs_delta": 0.0, "no_op": False}
    mods = gate_modules(model)
    kind = spec["kind"]
    changed = 0
    max_delta = 0.0

    def _record(old, new):
        nonlocal changed, max_delta
        changed += 1
        if old is not None:
            delta = (new - old).abs()
            value = float(delta.max().item()) if delta.numel() else 0.0
            max_delta = max(max_delta, value)

    with torch.no_grad():
        if kind in ("zero", "zero_evidence"):
            names = (spec["params"] if kind == "zero"
                     else ("gate_w", "gate_a", "gate_b"))
            for m in mods:
                for name in names:
                    p = getattr(m, name, None)
                    if p is None:
                        continue
                    old = p.detach().clone()
                    p.zero_()
                    _record(old, p.detach())
        elif kind in ("constant", "closed", "constant_mean"):
            head_d = None
            if kind == "constant_mean":
                # 优先用逐层/逐头的观测均值：这样只抹平“逐位置/逐查询”的
                # 变化，保留各头自己的平均写入强度；若调用方只给全局均值，
                # 则退化为全局常数门（同时抹平头间差异，需在报告中说明）。
                if gate_head_mean is not None:
                    if (not isinstance(gate_head_mean, (list, tuple))
                            or len(gate_head_mean) != len(mods)):
                        return {"applied": False,
                                "reason": "逐头平均门控值与模型层数不匹配",
                                "changed_tensors": 0,
                                "max_abs_delta": 0.0, "no_op": False}
                    head_d = []
                    for m, vals in zip(mods, gate_head_mean):
                        if (not isinstance(vals, (list, tuple))
                                or len(vals) != int(m.n_heads)):
                            return {"applied": False,
                                    "reason": "逐头平均门控值与头数不匹配",
                                    "changed_tensors": 0,
                                    "max_abs_delta": 0.0, "no_op": False}
                        row = []
                        for g in vals:
                            gf = _finite_or_none(g)
                            if gf is None or not 0.0 < gf < 1.0:
                                return {"applied": False,
                                        "reason": (
                                            f"逐头平均门控值不可用于恒定门"
                                            f"病灶: {g!r}（需 0<g<1）"),
                                        "changed_tensors": 0,
                                        "max_abs_delta": 0.0, "no_op": False}
                            row.append(math.log(gf / (1.0 - gf)))
                        head_d.append(row)
                else:
                    g = _finite_or_none(gate_mean)
                    if g is None or not 0.0 < g < 1.0:
                        return {"applied": False,
                                "reason": (f"观测平均门控值不可用于恒定门病灶: "
                                           f"{gate_mean!r}（需 0<g<1）"),
                                "changed_tensors": 0, "max_abs_delta": 0.0,
                                "no_op": False}
                    d_value = math.log(g / (1.0 - g))
            elif kind == "constant":
                d_value = 30.0
            else:
                d_value = -30.0
            for li, m in enumerate(mods):
                for name in ("gate_w", "gate_a", "gate_b"):
                    p = getattr(m, name, None)
                    if p is None:
                        continue
                    old = p.detach().clone()
                    p.zero_()
                    _record(old, p.detach())
                old = m.gate_d.detach().clone()
                if head_d is not None:
                    m.gate_d.copy_(torch.as_tensor(
                        head_d[li], dtype=m.gate_d.dtype,
                        device=m.gate_d.device))
                else:
                    m.gate_d.fill_(d_value)
                _record(old, m.gate_d.detach())
        elif kind == "head_permute":
            for li, m in enumerate(mods):
                H = int(m.n_heads)
                perm = _layer_permutation(H, perm_seed, li)
                idx = torch.as_tensor(perm, dtype=torch.long,
                                      device=m.gate_d.device)
                for name in ("gate_w", "gate_a", "gate_b", "gate_d"):
                    p = getattr(m, name, None)
                    if p is None:
                        continue
                    old = p.detach().clone()
                    p.copy_(p.detach()[idx])
                    _record(old, p.detach())
        elif kind == "sign_flip":
            for m in mods:
                for name in ("gate_w", "gate_a", "gate_b"):
                    p = getattr(m, name, None)
                    if p is None:
                        continue
                    old = p.detach().clone()
                    p.neg_()
                    _record(old, p.detach())
        else:  # pragma: no cover - 防御分支
            raise ValueError(f"未知病灶类型: {kind!r}")

    return {"applied": True, "reason": "",
            "changed_tensors": int(changed),
            "max_abs_delta": float(max_delta),
            "no_op": bool(max_delta == 0.0)}


def gate_head_means(model, val, ctx: int, device: str, eval_iters: int,
                    amp: bool = False):
    """逐层、逐头的观测平均门控值（与 `evaluate_val` 同一批窗口/精度）。

    用于 `gate_constant_mean`：把每个头的门固定为它自己的平均 g，只去掉
    逐位置/逐查询变化，保留头间与层间的平均写入强度差异。返回
    `[[head0_mean, ...], ...]`（层顺序与 `model.blocks` 一致）；某层没有
    门控统计时该层为 None（调用方会退化为全局均值或拒绝）。
    """
    ctx = strict_int(ctx, "ctx")
    eval_iters = strict_int(eval_iters, "eval_iters")
    indices, _n_windows = _val_eval_indices(len(val), ctx, eval_iters)
    batch_windows = max(1, min(16384 // max(1, ctx), len(indices)))
    sums = None
    counts = None
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for s in range(0, len(indices), batch_windows):
                chunk = indices[s:s + batch_windows]
                x, _y = get_val_windows(val, int(ctx), chunk, device)
                with amp_autocast(device, bool(amp)):
                    out = model(x, need_stats=True, stats_argmax=False)
                stats = out.get("gate_stats") or []
                if sums is None:
                    sums = [None] * len(stats)
                    counts = [0] * len(stats)
                for li, st in enumerate(stats):
                    g = st.get("g") if isinstance(st, dict) else None
                    if g is None:
                        continue
                    gf = g.detach().double()
                    if gf.dim() >= 3:
                        mean_h = gf.mean(dim=(0, 2))
                        weight = int(gf.shape[0])
                    elif gf.dim() == 2:
                        mean_h = gf.mean(dim=1)
                        weight = 1
                    else:
                        continue
                    if sums[li] is None:
                        sums[li] = mean_h * weight
                    else:
                        sums[li] = sums[li] + mean_h * weight
                    counts[li] += weight
    finally:
        model.train(was_training)
    if not sums:
        return None
    out = []
    for li in range(len(sums)):
        if sums[li] is None or counts[li] <= 0:
            out.append(None)
        else:
            out.append([float(v) for v in (sums[li] / counts[li]).tolist()])
    return out


def _window_delta_stats(base_losses, lesion_losses) -> Dict:
    """同一批窗口的 ΔNLL = lesion − base 的配对描述统计。

    窗口级 t/符号检验只作描述：PPL 窗口按时间顺序排列、相邻窗口可能相关，
    i.i.d. 假设会低估不确定性。跨种子的确认性口径在 run_all 汇总（每个
    种子先聚合成一个 Δ，再对种子做配对检验）。

    非有限/非数值/下划线数字串的窗口 loss 按缺失剔除，并在 ``n_invalid_windows``
    如实计数：把 NaN 传给 paired_t_test 会让 n 与实际 ``n_windows`` 不一致，
    均值也会变成 NaN，下游 PPL 比/显著性由此失真。
    """
    if len(base_losses) != len(lesion_losses):
        return {"n_windows": 0, "error": "窗口数不一致",
                "delta_losses": []}
    deltas = []
    invalid = 0
    for a, b in zip(lesion_losses, base_losses):
        fa, fb = _finite_or_none(a), _finite_or_none(b)
        if fa is None or fb is None:
            invalid += 1
            continue
        delta = fa - fb
        if not math.isfinite(delta):
            invalid += 1
            continue
        deltas.append(delta)
    if not deltas:
        return {"n_windows": 0, "error": "无有效窗口", "delta_losses": [],
                "n_invalid_windows": invalid}
    paired = paired_t_test(deltas, [0.0] * len(deltas))
    sign = paired_sign_test(deltas, [0.0] * len(deltas))
    ci = bootstrap_mean_ci(deltas)
    # 与 paired_t_test 共用同一套缩放后的稳定均值/标准差：deltas 都是有限值，
    # 但 [1e308, 1e308] 的朴素 np.mean/np.std 会在求和/平方时溢出成 inf，
    # 让点估计、PPL 比与 JSON 结果出现非有限值。paired 的 n 已剔除缺失窗口。
    mean = paired.get("mean_diff")
    se = None
    std = paired.get("std_diff")
    if std is not None and paired.get("n", 0) >= 2:
        se = float(std / math.sqrt(paired["n"]))
        if not math.isfinite(se):
            se = None
    out = {
        "n_windows": len(deltas),
        "delta_loss_mean": mean,
        "delta_loss_se": se,
        "delta_loss_ci95": list(ci) if ci else None,
        "p_window_t": paired.get("p"),
        "t_window": paired.get("t"),
        "p_window_sign": sign.get("p_sign"),
        "window_n_pos": sign.get("n_pos"),
        "window_n_neg": sign.get("n_neg"),
        "delta_losses": deltas,
    }
    if invalid:
        out["n_invalid_windows"] = invalid
    return out


def _evaluate_once(model, val, ctx, device, eval_iters, amp) -> Dict:
    metrics = evaluate_val(model, val, ctx, device, eval_iters, amp=amp,
                           collect_window_losses=True)
    norm = json_safe_eval_metrics(metrics)
    out = dict(norm)
    for key in ("val_loss_se", "val_loss_ci95", "val_ppl_ci95",
                "val_loss_min", "val_loss_max", "gate_mean", "gate_std",
                "gate_frac_low", "gate_frac_high", "gate_layer_mean",
                "eval_iters_requested", "eval_iters", "val_windows",
                "val_tokens_evaluated", "window_losses", "window_indices"):
        out[key] = metrics.get(key)
    return out


def run_ablation(ckpt: str, data_dir: str, *, eval_iters: Optional[int] = None,
                 ctx: Optional[int] = None, device: str = "auto",
                 seed: Optional[int] = None, perm_seed: int = 0,
                 allow_data_protocol_violations: bool = False) -> Dict:
    """在 val split 上对一个 checkpoint 执行全部适用病灶并返回 payload。"""
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"checkpoint 不存在: {ckpt}")
    preflight = data_protocol_preflight(
        data_dir, allow_override=allow_protocol_violations(
            allow_data_protocol_violations))
    if not preflight.get("report_present"):
        print("[ablate] 警告：数据目录缺少 data_report.json，"
              "无法做评测前防泄漏闸门（旧数据或未走统一入口）")
    elif preflight.get("critical_violations"):
        print("[ablate] ⚠ 数据协议自检发现严重问题但已显式放行；"
              "本分析不得用于正式结论")

    device = get_device(device)
    enable_tf32(device)
    model, ck = load_model(ckpt, device=device)
    caps = gate_capabilities(model)
    if not caps["gated"]:
        raise ValueError(
            f"变体 {model.variant!r} 没有输出门，无法做门控病灶消融；"
            "该分析只适用于 gate_only/qgate/lega 及其消融")

    ckpt_seed = ck.get("seed")
    if seed is None:
        if ckpt_seed is not None:
            try:
                seed = seed_int(ckpt_seed, "checkpoint seed")
            except (TypeError, ValueError):
                seed = None
    else:
        # 显式传入的 seed 同样走全项目统一的严格种子口径：int(True)==1、
        # int(1.5)==1 会让实际播种/写入结果的种子与调用方声明不一致。
        seed = seed_int(seed, "seed")
    perm_seed = strict_int(perm_seed, "perm_seed")
    if perm_seed < 0:
        raise ValueError(f"perm_seed 必须 >= 0，收到 {perm_seed}")
    set_seed(0 if seed is None else seed)

    # 与 lega.evaluate 相同的评测上下文/窗口数/AMP 口径：病灶前后的差异
    # 必须只来自门控参数，不能混入评测协议变化。
    ckpt_dir = os.path.dirname(os.path.abspath(ckpt))
    cfg_used = {}
    cfg_path = os.path.join(ckpt_dir, "config_used.json")
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            cfg_used = loaded if isinstance(loaded, dict) else {}
        except (OSError, ValueError):
            cfg_used = {}
    if ctx is None:
        train_ctx = _positive_int_or_none(cfg_used.get("train_ctx"))
        ctx = min(model.block_size, train_ctx) if train_ctx else model.block_size
    ctx = strict_int(ctx, "ctx")
    if ctx <= 0 or ctx > model.block_size:
        raise ValueError(f"ctx 必须在 [1, {model.block_size}]，收到 {ctx}")
    if eval_iters is None:
        eval_iters = _positive_int_or_none(cfg_used.get("eval_iters")) or 50
    eval_iters = strict_int(eval_iters, "eval_iters")
    if eval_iters <= 0:
        raise ValueError(f"eval_iters 必须 >= 1，收到 {eval_iters}")
    amp = _resolve_eval_amp(cfg_used, device)

    val = load_bin(data_dir, "val", allow_test=False)
    base = _evaluate_once(model, val, ctx, device, eval_iters, amp)
    if base.get("window_losses") is None or base.get("window_indices") is None:
        raise RuntimeError("基线评测未返回逐窗口损失，无法做配对病灶分析")
    base_windows = list(base["window_losses"])
    gate_mean = base.get("gate_mean")

    snapshot = snapshot_gate_params(model)
    head_mean_cache = None
    lesions = []
    for spec in LESION_SPECS:
        lid = spec["id"]
        applicable, reason, requires = lesion_applicability(model, lid)
        rec = {
            "lesion": lid, "label": spec["label"], "target": spec["target"],
            "expected": spec["expected"], "note": spec["note"],
            "requires": list(requires),
        }
        if not applicable:
            rec.update({"applicable": False, "reason": reason,
                        "applied": False})
            lesions.append(rec)
            continue
        try:
            if lid == "gate_constant_mean" and head_mean_cache is None:
                # 只在需要时多跑一遍窗口前向；拿不到逐头统计时回退全局均值。
                head_mean_cache = gate_head_means(
                    model, val, ctx, device, eval_iters, amp=amp)
            applied = apply_lesion(model, lid, gate_mean=gate_mean,
                                   gate_head_mean=head_mean_cache,
                                   perm_seed=perm_seed)
        except BaseException:
            # 病灶可能已部分写入：先恢复快照再抛出，避免污染后续分析。
            restore_gate_params(model, snapshot)
            raise
        rec.update(applied)
        if not applied.get("applied"):
            rec["applicable"] = False
            rec["reason"] = applied.get("reason", "未施加")
            lesions.append(rec)
            continue
        try:
            lesion_eval = _evaluate_once(model, val, ctx, device, eval_iters, amp)
        finally:
            # 无论评测是否成功，都必须恢复参数，避免后续病灶叠加。
            restore_gate_params(model, snapshot)
        if lesion_eval.get("window_indices") != base.get("window_indices"):
            rec.update({"applicable": False,
                        "reason": "病灶评测窗口与基线不一致，拒绝配对"})
            lesions.append(rec)
            continue
        rec.update(lesion_eval)
        rec.update(_window_delta_stats(base_windows,
                                       lesion_eval.get("window_losses") or []))
        delta = rec.get("delta_loss_mean")
        rec["ppl_ratio"] = (math.exp(min(20.0, max(-20.0, float(delta))))
                            if delta is not None else None)
        rec["ppl_ratio_clamped"] = bool(
            delta is not None and not (-20.0 <= float(delta) <= 20.0))
        rec["delta_ppl_pct"] = ((rec["ppl_ratio"] - 1.0) * 100.0
                                if rec["ppl_ratio"] is not None else None)
        # 门控干预是否真的改变了前向：直接看门控分布，而不是相信参数写入。
        # 恒定均值病灶保留均值、只去掉逐位置变化，因此必须同时比较标准差，
        # 否则会把它误判成“没有改变前向”。
        base_mean = _finite_or_none(gate_mean)
        base_std = _finite_or_none(base.get("gate_std"))
        les_mean = _finite_or_none(rec.get("gate_mean"))
        les_std = _finite_or_none(rec.get("gate_std"))
        rec["gate_mean_changed"] = bool(
            base_mean is not None and les_mean is not None
            and abs(les_mean - base_mean) > 1e-6)
        rec["gate_std_changed"] = bool(
            base_std is not None and les_std is not None
            and abs(les_std - base_std) > 1e-6)
        rec["gate_changed"] = bool(
            rec["gate_mean_changed"] or rec["gate_std_changed"])
        lesions.append(rec)
    restore_gate_params(model, snapshot)

    payload = {
        "schema": 1,
        "variant": model.variant,
        "ckpt": os.path.abspath(ckpt),
        "ckpt_sha256": sha256_file(ckpt),
        "seed": (None if seed is None else int(seed)),
        "split": "val",
        "eval_ctx": int(ctx),
        "eval_iters": int(eval_iters),
        "val_windows": base.get("val_windows"),
        "val_tokens_evaluated": base.get("val_tokens_evaluated"),
        "window_indices": base.get("window_indices"),
        "amp": bool(amp),
        "gate_capabilities": caps,
        "baseline": {k: base.get(k) for k in (
            "val_ppl", "val_loss", "val_ppl_clamped", "gate_mean",
            "gate_std", "gate_frac_low", "gate_frac_high",
            "gate_layer_mean", "val_loss_se", "val_loss_ci95",
            "val_ppl_ci95", "val_loss_min", "val_loss_max")},
        "baseline_window_losses": base_windows,
        "lesions": lesions,
        "protocol": {
            "split": "val",
            "same_windows": True,
            "note": ("同一 checkpoint、同一批验证窗口上的权重病灶；"
                     "query_only/conc_only/len_only 是 2^3 的单因子格，"
                     "与 no_* 及基线共同用于推理期 Shapley，不作单病灶"
                     "因果归因；窗口级 p 仅描述性（相邻窗口可能相关），"
                     "跨种子配对检验由 run_all 汇总"),
        },
        "provenance": load_provenance(ckpt_dir, data_dir),
        "generated_at": now_str(),
        "argv": list(sys.argv),
    }
    return payload


def main():
    p = argparse.ArgumentParser(
        description="训练后因果病灶消融（同一 checkpoint 的推理期权重干预）")
    p.add_argument("--ckpt", required=True,
                   help="model_best.pt / model.pt（必须是带输出门的变体）")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out", required=True, help="输出 JSON 路径")
    p.add_argument("--eval_iters", default=None,
                   help="评测窗口数；默认继承 checkpoint 的 config_used.json"
                        "（缺失时 50，与 lega.evaluate 一致）")
    p.add_argument("--ctx", default=None,
                   help="评测上下文；默认继承训练配置的 train_ctx")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", default=None,
                   help="覆盖 checkpoint 记录的种子（只影响确定性置换/播种，"
                        "不改变权重）")
    p.add_argument("--perm_seed", default="0",
                   help="头置换病灶的确定性随机种子（默认 0；严格整数）")
    p.add_argument("--allow_data_protocol_violations", action="store_true",
                   help="即使 data_report 有严重泄露告警也继续（仅复现历史；"
                        "结果不得用于正式结论）")
    args = p.parse_args()
    seed = None
    if args.seed is not None:
        try:
            seed = strict_int(args.seed, "--seed")
        except (TypeError, ValueError) as e:
            p.error(str(e))
    try:
        perm_seed = strict_int(args.perm_seed, "--perm_seed")
    except (TypeError, ValueError) as e:
        p.error(str(e))
    if perm_seed < 0:
        p.error(f"--perm_seed 必须 >= 0，收到 {perm_seed}")
    try:
        payload = run_ablation(
            args.ckpt, args.data_dir, eval_iters=args.eval_iters, ctx=args.ctx,
            device=args.device, seed=seed, perm_seed=perm_seed,
            allow_data_protocol_violations=args.allow_data_protocol_violations)
    except (ValueError, FileNotFoundError) as e:
        p.error(str(e))
    save_json(payload, args.out)
    base = payload["baseline"]
    print(f"[ablate] {payload['variant']} seed={payload['seed']} "
          f"val_loss={base.get('val_loss')} val_ppl={base.get('val_ppl')} "
          f"gate_mean={base.get('gate_mean')}")
    for rec in payload["lesions"]:
        if rec.get("applicable") is False and rec.get("applied") is False:
            print(f"  - {rec['lesion']}: 不适用（{rec.get('reason')}）")
            continue
        print(f"  - {rec['lesion']}: ΔNLL={rec.get('delta_loss_mean')} "
              f"PPL比={rec.get('ppl_ratio')} "
              f"p_window={rec.get('p_window_t')}"
              + (" [no-op：该组件系数已为 0]" if rec.get("no_op") else ""))
    print(f"[ablate] -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
