"""训练 run 的顺序分配（随机化区组设计）。

为什么顺序属于“实验设计”而不是工程细节
======================================
同一批变体在同一台机器上顺序训练时，时间不是中性的：GPU 温度/降频、文件
缓存、驱动状态、后台任务都可能随轮次漂移。若按 ``for variant: for seed``
执行，某个变体永远排在前/后，方法效应就与时间效应（以及任何单调漂移）
完全混淆。正确的做法是把**种子当作区组**：每个种子区块内包含全部变体，
区块内顺序由跑前写死的随机种子打乱。这样：

* 每个变体在每个时间区块内各出现一次，时间漂移不再系统性偏向某个变体；
* 顺序随机化使“某个变体恰好总在最佳/最差时段”不再是固定安排；
* 随机种子与最终顺序写入 ``design_plan.json``，可复算、可审计，不能看到
  结果后再挑一个“有利顺序”。

本模块只做确定性的顺序分配，不训练、不读数据、不访问 test。历史结果目录
里的旧计划若没有 ``run_order``，仍按旧的变体主序执行；新计划默认使用
随机化区组设计，并在设计检查中显式标注旧顺序的混淆风险。
"""
from __future__ import annotations

import hashlib
import json
import operator

from ._shared import _install as _shared_install

_shared_install(globals())

RUN_ORDER_MODES = ("randomized_blocked", "seed_major", "variant_major")

DEFAULT_RUN_ORDER = "randomized_blocked"

# 顺序随机化的盐值固定写死在代码里：它只用于把“同一设计”映射到可复算的
# 顺序，不应由结果或运行环境改变。显式配置 design.run_order_seed 可覆盖。
RUN_ORDER_SALT = "lega-run-order-v1"


def _strict_int(value, name: str, minimum: int = 0) -> int:
    """严格整数：bool/小数/字符串拒绝；与设计层其它入口同口径。"""
    if isinstance(value, bool):
        raise ValueError(f"{name} 必须是整数，收到 bool {value!r}")
    try:
        out = operator.index(value)
    except TypeError:
        raise ValueError(f"{name} 必须是整数，收到 {value!r}") from None
    if out < minimum:
        raise ValueError(f"{name} 必须 >= {minimum}，收到 {out}")
    return out


def _canonical(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"))


def _derived_order_seed(variants, seeds, mode: str) -> int:
    """由计划输入派生顺序种子，避免依赖机器/时间/结果。

    同一组变体、种子与模式必须得到同一顺序；改变其中任一项都会得到不同的
    顺序种子。显式 ``design.run_order_seed`` 优先于该派生值。
    """
    payload = {"salt": RUN_ORDER_SALT, "mode": mode,
               "variants": list(variants), "seeds": list(seeds)}
    digest = hashlib.sha256(_canonical(payload).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def _hash_stream(seed: int, counter: int) -> int:
    """确定性 64 位随机流：SHA-256(seed, counter)，跨 Python 版本稳定。"""
    raw = hashlib.sha256(
        f"{RUN_ORDER_SALT}:{int(seed)}:{int(counter)}".encode("utf-8")).digest()
    return int.from_bytes(raw[:8], "big")


def _shuffle(items, seed: int):
    """用 SHA-256 计数器流做 Fisher-Yates，避免依赖 random 模块实现版本。"""
    out = list(items)
    counter = 0
    for i in range(len(out) - 1, 0, -1):
        j = _hash_stream(seed, counter) % (i + 1)
        counter += 1
        out[i], out[j] = out[j], out[i]
    return out


def build_run_order(variants, seeds, *, mode: str = DEFAULT_RUN_ORDER,
                    order_seed=None) -> dict:
    """生成可复算的 run 顺序计划。

    返回字段：

    * ``mode``：``randomized_blocked``（默认）、``seed_major`` 或
      ``variant_major``（历史行为）；
    * ``order_seed`` / ``seed_source``：实际种子及其来源（配置或派生）；
    * ``blocks``：按种子分块后的变体顺序；
    * ``cells``：逐 run 的 ``position/block/variant/seed``，供流水线直接
      消费与审计。
    """
    if not isinstance(mode, str) or mode not in RUN_ORDER_MODES:
        raise ValueError(
            f"design.run_order 必须是 {RUN_ORDER_MODES} 之一，收到 {mode!r}")
    if not isinstance(variants, (list, tuple)) or not variants:
        raise ValueError("variants 必须是非空列表")
    if not all(isinstance(v, str) and v for v in variants):
        raise ValueError("variants 必须是非空字符串列表")
    if not isinstance(seeds, (list, tuple)) or not seeds:
        raise ValueError("seeds 必须是非空列表")
    variants = list(variants)
    seeds = [_strict_int(s, "seed", minimum=0) for s in seeds]
    if len(set(variants)) != len(variants):
        raise ValueError("variants 不能重复")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds 不能重复")
    if any(s > MAX_SEED for s in seeds):
        raise ValueError(f"seed 必须在 0..{MAX_SEED} 内")

    if order_seed is None:
        order_seed = _derived_order_seed(variants, seeds, mode)
        seed_source = "derived_from_plan_inputs"
    else:
        order_seed = _strict_int(order_seed, "design.run_order_seed",
                                 minimum=0)
        if order_seed > MAX_SEED:
            raise ValueError(
                f"design.run_order_seed 必须在 0..{MAX_SEED} 内，"
                f"收到 {order_seed}")
        seed_source = "configured"

    cells = []
    if mode == "variant_major":
        for variant in variants:
            for seed in seeds:
                cells.append({"position": len(cells) + 1,
                              "block": seed, "variant": variant,
                              "seed": seed})
    elif mode == "seed_major":
        for seed in seeds:
            order = list(variants)
            for variant in order:
                cells.append({"position": len(cells) + 1,
                              "block": seed, "variant": variant,
                              "seed": seed})
    else:  # randomized_blocked
        for block_index, seed in enumerate(seeds):
            # 每个种子区块使用独立的流，避免区块顺序影响下一种子的洗牌。
            block_seed = _hash_stream(order_seed, 1_000_000 + block_index)
            order = _shuffle(variants, block_seed)
            for variant in order:
                cells.append({"position": len(cells) + 1,
                              "block": seed, "variant": variant,
                              "seed": seed})

    # blocks 必须与 cells 同源：统一按种子列出该区块实际执行的变体顺序。
    # variant_major 不是按种子连续执行，但仍可按种子过滤得到每块内的顺序；
    # 不能为它返回 n_cells 个单 run 条目，否则 n_blocks 与 blocks 长度冲突。
    blocks = [{"seed": seed,
               "variants": [c["variant"] for c in cells
                            if c["seed"] == seed]}
              for seed in seeds]

    expected = {(v, s) for v in variants for s in seeds}
    observed = [(c["variant"], c["seed"]) for c in cells]
    if len(observed) != len(expected) or set(observed) != expected:
        raise ValueError("run 顺序没有恰好覆盖全部 (变体, 种子) 单元")
    if len({c["position"] for c in cells}) != len(cells):
        raise ValueError("run 顺序的 position 必须唯一")
    return {
        "mode": mode,
        "order_seed": int(order_seed),
        "seed_source": seed_source,
        "n_blocks": len(seeds),
        "n_cells": len(cells),
        "blocks": blocks,
        "cells": cells,
        "note": (
            "种子为区组，区块内变体顺序在跑前由 order_seed 确定；顺序不读取"
            "任何结果。randomized_blocked 用于防止时间/硬件漂移与变体混淆。"),
    }


def validate_run_order(order: dict, variants, seeds) -> list:
    """核对计划顺序是否覆盖全部单元且满足所选模式；返回问题列表。"""
    problems = []
    if not isinstance(order, dict):
        return ["run_order 不是对象"]
    mode = order.get("mode")
    if mode not in RUN_ORDER_MODES:
        problems.append(f"run_order.mode 非法：{mode!r}")
    variants = list(variants or [])
    seeds_ok = True
    try:
        seeds = [_strict_int(s, "seed", minimum=0) for s in (seeds or [])]
    except ValueError as e:
        problems.append(f"计划种子非法：{e}")
        seeds = []
        seeds_ok = False
    order_seed = order.get("order_seed")
    if (not isinstance(order_seed, int) or isinstance(order_seed, bool)
            or not 0 <= order_seed <= MAX_SEED):
        problems.append(
            f"run_order.order_seed 必须是 0..{MAX_SEED} 的整数")
    seed_source = order.get("seed_source")
    if seed_source not in ("configured", "derived_from_plan_inputs"):
        problems.append(f"run_order.seed_source 非法：{seed_source!r}")
    elif (seed_source == "derived_from_plan_inputs"
          and isinstance(order_seed, int)
          and not isinstance(order_seed, bool)
          and mode in RUN_ORDER_MODES):
        if order_seed != _derived_order_seed(variants, seeds, mode):
            problems.append(
                "run_order.order_seed 与计划输入/模式派生的种子不一致")
    cells = order.get("cells")
    if not isinstance(cells, list) or not cells:
        return problems + ["run_order.cells 必须是非空列表"]
    expected = {(v, s) for v in variants for s in seeds}
    seen = []
    positions = []
    for i, c in enumerate(cells):
        if not isinstance(c, dict):
            problems.append(f"run_order.cells[{i}] 不是对象")
            continue
        v, s = c.get("variant"), c.get("seed")
        if not isinstance(v, str) or not isinstance(s, int) or isinstance(s, bool):
            problems.append(f"run_order.cells[{i}] 的 variant/seed 类型非法")
            continue
        seen.append((v, s))
        pos = c.get("position")
        if not isinstance(pos, int) or isinstance(pos, bool):
            problems.append(f"run_order.cells[{i}].position 不是整数")
        else:
            positions.append(pos)
        block = c.get("block")
        if not isinstance(block, int) or isinstance(block, bool) or block != s:
            problems.append(
                f"run_order.cells[{i}].block 必须与 seed 一致")
    if len(seen) != len(expected) or set(seen) != expected:
        problems.append("run_order 没有恰好覆盖全部 (变体, 种子) 单元")
    if positions and positions != list(range(1, len(cells) + 1)):
        problems.append(
            "run_order.position 必须按 cells 列表顺序从 1 连续递增")
    n_cells = order.get("n_cells")
    if (not isinstance(n_cells, int) or isinstance(n_cells, bool)
            or n_cells != len(cells)):
        problems.append("run_order.n_cells 必须等于 cells 数量")
    n_blocks = order.get("n_blocks")
    if (not isinstance(n_blocks, int) or isinstance(n_blocks, bool)
            or n_blocks != len(seeds)):
        problems.append("run_order.n_blocks 必须等于计划种子数")
    if mode in ("randomized_blocked", "seed_major"):
        # 区块模式不仅要求每个 seed 覆盖全部变体，还要求同一 seed 的
        # cells 在运行顺序中连续。否则把 variant_major 的计划仅改 mode
        # 就能伪装成区组设计：每个 seed 仍覆盖全部变体，但时间漂移与
        # 变体主序完全混淆。
        seen_seeds = set()
        current_seed = object()
        for _, seed in seen:
            if seed == current_seed:
                continue
            if seed in seen_seeds:
                problems.append(
                    f"{mode} 的种子区块必须连续：seed {seed} 被拆成多个区块")
                break
            seen_seeds.add(seed)
            current_seed = seed
    if mode == "randomized_blocked":
        if len(seeds) != len(set(seeds)):
            problems.append("randomized_blocked 的种子区块不能重复")
        for seed in seeds:
            block = [(v, s) for v, s in seen if s == seed]
            if sorted(v for v, _ in block) != sorted(variants):
                problems.append(
                    f"种子区块 {seed} 没有恰好包含每个变体一次")
    if mode == "variant_major":
        expected_order = [(v, s) for v in variants for s in seeds]
        if seen != expected_order:
            problems.append("variant_major 顺序必须与旧实现一致")
    if mode == "seed_major":
        for seed in seeds:
            block = [(v, s) for v, s in seen if s == seed]
            if [v for v, _ in block] != variants:
                problems.append(f"种子区块 {seed} 的变体顺序与输入不一致")
    blocks = order.get("blocks")
    if seeds_ok:
        if not isinstance(blocks, list) or len(blocks) != len(seeds):
            problems.append("run_order.blocks 必须按计划种子逐块列出")
        elif len(seen) == len(cells):
            for i, seed in enumerate(seeds):
                b = blocks[i]
                if not isinstance(b, dict):
                    problems.append(f"run_order.blocks[{i}] 不是对象")
                    continue
                if (not isinstance(b.get("seed"), int)
                        or isinstance(b.get("seed"), bool)
                        or b.get("seed") != seed):
                    problems.append(
                        f"run_order.blocks[{i}].seed 必须与计划种子一致")
                expected_variants = [v for v, s in seen if s == seed]
                if b.get("variants") != expected_variants:
                    problems.append(
                        f"run_order.blocks[{i}].variants 必须与 cells 中"
                        "该种子区块的实际顺序一致")
    return problems
