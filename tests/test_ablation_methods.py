"""对比/消融方法测试：Shapley 分解、推理期病灶消融与跨种子汇总。

覆盖三条新增科研口径：
  * 全因子 Shapley 值的数学性质（可加、纯交互、效率性、方向转换）；
  * 推理期权重病灶只改门控参数、可逐位恢复，且不适用/空操作被如实标注；
  * 病灶结果按“先种子聚合、再跨种子配对”推断，不把窗口当独立样本。
"""
import json
import math
import os
import tempfile

import numpy as np
import torch

from lega.ablate import (INFERENCE_FACTORIAL_CELLS, LESION_BY_ID,
                         LESION_ORDER, apply_lesion,
                         gate_capabilities, gate_head_means,
                         lesion_applicability, restore_gate_params,
                         run_ablation, snapshot_gate_params,
                         _finite_or_none, _layer_permutation,
                         _window_delta_stats)
from lega.factorial import (CELL_ORDER, decomposition, oriented_value,
                            ppl_ratio_from_phi, shapley_values, shares)
from lega.model import GPT, save_checkpoint
from run_all import (_ablation_lesion_records, _factorial_shapley_tests,
                     _inference_interaction_tests, _inference_shapley_tests,
                     _render_inference_interactions,
                     CAUSAL_ABLATION_METRIC, INFERENCE_INTERACTION_METRIC,
                     INFERENCE_SHAPLEY_METRIC, SHAPLEY_PPL_METRIC)


def _tiny(variant):
    return GPT(vocab_size=64, block_size=32, d_model=16, n_layers=2,
               n_heads=2, variant=variant, use_sdpa_standard=False)


def _ppl_cell(ppl):
    return {"eval": {"split": "val", "val_ppl": ppl, "val_loss": math.log(ppl)}}


def test_shapley_additive_equals_component_weights():
    # v(S) = Σ_{i∈S} w_i 时 Shapley 值应精确等于 w_i。
    w = {"A": 1.0, "B": 2.0, "C": 4.0}
    values = {}
    for cell in CELL_ORDER:
        fs = set() if cell == "none" else set(cell)
        values[cell] = sum(w[f] for f in fs)
    phi = shapley_values(values)
    assert abs(phi["A"] - 1.0) < 1e-12
    assert abs(phi["B"] - 2.0) < 1e-12
    assert abs(phi["C"] - 4.0) < 1e-12
    dec = decomposition(values)
    assert abs(dec["total_effect"] - 7.0) < 1e-12
    assert abs(dec["efficiency_gap"]) < 1e-12
    print("  [ok] Shapley：可加价值函数下 φ=各因子权重，效率性成立")


def test_shapley_pure_interaction_is_shared_not_zero():
    # 只有 A、B 同时存在才有收益：v(AB)=v(ABC)=1，其余为 0。
    # 两因子纯交互的 Shapley 应各分 1/2，C 为 0——这正是“单因素边
    # 在某个参照格下会得到 0、另一个参照格得到 1”的非可加例子。
    values = {c: 0.0 for c in CELL_ORDER}
    values["AB"] = 1.0
    values["ABC"] = 1.0
    phi = shapley_values(values)
    assert abs(phi["A"] - 0.5) < 1e-12
    assert abs(phi["B"] - 0.5) < 1e-12
    assert abs(phi["C"]) < 1e-12
    print("  [ok] Shapley：纯 A×B 交互被两因子均分，C 的零玩家 φ=0")


def test_shapley_requires_all_cells_and_finite_values():
    values = {c: 1.0 for c in CELL_ORDER if c != "BC"}
    try:
        shapley_values(values)
        raise AssertionError("缺格应显式报错，而不是按 0 补齐")
    except ValueError:
        pass
    bad = {c: 1.0 for c in CELL_ORDER}
    bad["AC"] = float("nan")
    try:
        shapley_values(bad)
        raise AssertionError("NaN 应显式报错")
    except ValueError:
        pass
    # 下划线数字串与 np.bool_ 不是规范观测：float("1_0")==10、bool→1
    # 会把损坏的外部证据伪装成另一个真实单元格值。
    for poor in ("1_0", np.bool_(True), True):
        bad = {c: 1.0 for c in CELL_ORDER}
        bad["AC"] = poor
        try:
            shapley_values(bad)
            raise AssertionError(f"{poor!r} 应显式报错")
        except ValueError:
            pass
        try:
            oriented_value("PPL", poor)
            raise AssertionError(f"PPL 原始值 {poor!r} 应显式报错")
        except ValueError:
            pass
    print("  [ok] Shapley：缺格/非有限值/下划线数字串/bool 显式拒绝，"
          "不伪造单元格")


def test_oriented_value_direction_and_ppl_ratio():
    assert abs(oriented_value("PPL", 10.0) + math.log(10.0)) < 1e-12
    assert abs(oriented_value("RULER平均", 0.25) - 0.25) < 1e-12
    assert abs(ppl_ratio_from_phi(0.0) - 1.0) < 1e-12
    assert ppl_ratio_from_phi(1.0) < 1.0  # φ>0 -> 降低 PPL
    try:
        oriented_value("PPL", 0.0)
        raise AssertionError("PPL=0 不能做 log 分解")
    except ValueError:
        pass
    print("  [ok] Shapley：PPL 转 −log（φ>0 降低 PPL），非法值显式拒绝")


def test_factorial_shapley_tests_additive_runs():
    # 构造 5 个种子的可加 PPL 结果：A/B/C 的 −log PPL 贡献为 1/2/4。
    variants = {"none": "gate_only", "A": "qgate", "B": "lega_conc_only",
                "C": "lega_len_only", "AB": "lega_no_len",
                "AC": "lega_no_conc", "BC": "lega_no_query", "ABC": "lega"}
    factors = {"none": "", "A": "A", "B": "B", "C": "C", "AB": "AB",
               "AC": "AC", "BC": "BC", "ABC": "ABC"}
    weights = {"A": 1.0, "B": 2.0, "C": 4.0}
    runs = {}
    for cell, v in variants.items():
        runs[v] = {}
        gain = sum(weights[f] for f in factors[cell])
        for seed in (1, 2, 3, 4, 5):
            runs[v][seed] = _ppl_cell(math.exp(-gain))
    tests = _factorial_shapley_tests(runs)
    assert len(tests) == 3
    got = {t["shapley_factor"]: t for t in tests}
    assert abs(got["A"]["mean_diff"] - 1.0) < 1e-9
    assert abs(got["B"]["mean_diff"] - 2.0) < 1e-9
    assert abs(got["C"]["mean_diff"] - 4.0) < 1e-9
    assert all(t["metric"] == SHAPLEY_PPL_METRIC for t in tests)
    assert abs(got["A"]["shapley_share"] - 1.0 / 7.0) < 1e-9
    assert got["A"]["shapley_ppl_ratio"] < 1.0
    assert got["A"]["n"] == 5
    assert got["A"]["p_adj"] is not None
    print("  [ok] Shapley 汇总：逐种子 φ、占比、PPL 比与 Holm 校正")


def test_lesion_capabilities_and_applicability():
    standard = _tiny("standard")
    gate_only = _tiny("gate_only")
    lega = _tiny("lega")
    assert gate_capabilities(standard)["gated"] is False
    caps = gate_capabilities(lega)
    assert caps["query"] and caps["conc"] and caps["len"]
    assert caps["multi_head"] and caps["n_heads"] == 2
    assert lesion_applicability(lega, "no_query")[0] is True
    assert lesion_applicability(gate_only, "no_query")[0] is False
    assert lesion_applicability(gate_only, "gate_open")[0] is True
    print("  [ok] 病灶适用性由真实注册参数探测，不靠变体名猜测")


def test_apply_and_restore_component_lesions():
    model = _tiny("lega")
    # 让各头参数不同，便于检查 head_permute 与恢复。
    with torch.no_grad():
        for li, block in enumerate(model.blocks):
            a = block.attn
            a.gate_w.copy_(torch.arange(a.gate_w.numel()).reshape(a.gate_w.shape)
                           + 1.0 + li)
            a.gate_a.copy_(torch.tensor([0.5, -0.25]) + li)
            a.gate_b.copy_(torch.tensor([0.1, 0.2]) + li)
            a.gate_d.copy_(torch.tensor([1.5, 2.5]) + li)
    snap = snapshot_gate_params(model)
    before = {n: p.clone() for n, p in model.named_parameters() if n in snap}

    for lid in ("no_query", "no_conc", "no_len", "no_evidence", "sign_flip"):
        spec = LESION_BY_ID[lid]
        rec = apply_lesion(model, lid, gate_mean=0.8)
        assert rec["applied"] is True and rec["changed_tensors"] > 0
        for block in model.blocks:
            a = block.attn
            if lid == "no_query":
                assert torch.count_nonzero(a.gate_w) == 0
            if lid in ("no_conc", "no_evidence"):
                assert torch.count_nonzero(a.gate_a) == 0
            if lid in ("no_len", "no_evidence"):
                assert torch.count_nonzero(a.gate_b) == 0
        restore_gate_params(model, snap)
        for n, p in model.named_parameters():
            if n in before:
                assert torch.equal(p, before[n]), f"恢复失败: {n}"

    # 恢复后正负号翻转必须与原文一致（检查 sign_flip 不污染下一次病灶）。
    for lid in LESION_ORDER:
        if not lesion_applicability(model, lid)[0]:
            continue
        apply_lesion(model, lid, gate_mean=0.8)
        restore_gate_params(model, snap)
    for n, p in model.named_parameters():
        if n in before:
            assert torch.equal(p, before[n]), f"最终恢复失败: {n}"
    print("  [ok] 组件病灶清零/反号后逐位恢复；重复病灶不叠加")


def test_gate_strength_and_head_permute_lesions():
    model = _tiny("lega")
    with torch.no_grad():
        for li, block in enumerate(model.blocks):
            a = block.attn
            a.gate_w.copy_(torch.arange(a.gate_w.numel(),
                                        dtype=torch.float32).reshape(a.gate_w.shape))
            a.gate_a.copy_(torch.tensor([0.5, -0.25]))
            a.gate_b.copy_(torch.tensor([0.1, 0.2]))
            a.gate_d.copy_(torch.tensor([1.5, 2.5]))
    snap = snapshot_gate_params(model)

    rec = apply_lesion(model, "gate_open", gate_mean=0.8)
    assert rec["applied"] and all(float(m.gate_d[0].detach()) == 30.0
                                  for m in [b.attn for b in model.blocks])
    restore_gate_params(model, snap)
    rec = apply_lesion(model, "gate_closed", gate_mean=0.8)
    assert rec["applied"] and all(float(m.gate_d[0].detach()) == -30.0
                                  for m in [b.attn for b in model.blocks])
    restore_gate_params(model, snap)

    rec = apply_lesion(model, "gate_constant_mean", gate_mean=0.8)
    expected = math.log(0.8 / 0.2)
    assert rec["applied"]
    for m in [b.attn for b in model.blocks]:
        assert abs(float(m.gate_d[0].detach()) - expected) < 1e-6
    restore_gate_params(model, snap)

    # head_permute 必须保持每个参数的多重集合（容量/分布不变），但改变头分配。
    before_a = model.blocks[0].attn.gate_a.clone()
    rec = apply_lesion(model, "head_permute", perm_seed=0)
    assert rec["applied"] is True
    after_a = model.blocks[0].attn.gate_a
    assert torch.allclose(torch.sort(before_a).values, torch.sort(after_a).values)
    assert not torch.equal(before_a, after_a)
    assert rec["max_abs_delta"] > 0
    restore_gate_params(model, snap)

    # 全零参数时 head_permute 是诚实的 no-op（不是失败）。
    with torch.no_grad():
        for m in [b.attn for b in model.blocks]:
            m.gate_w.zero_(); m.gate_a.zero_(); m.gate_b.zero_()
            m.gate_d.fill_(2.0)
    rec = apply_lesion(model, "head_permute", perm_seed=0)
    assert rec["applied"] is True and rec["no_op"] is True
    print("  [ok] 门强度/恒定门/头置换病灶；全同头时置换如实报告 no-op")


def test_constant_mean_rejects_invalid_gate_mean():
    model = _tiny("lega")
    snap = snapshot_gate_params(model)
    rec = apply_lesion(model, "gate_constant_mean", gate_mean=1.0)
    assert rec["applied"] is False and "0<g<1" in rec["reason"]
    restore_gate_params(model, snap)
    print("  [ok] 恒定门病灶拒绝越界均值，不静默夹断")



def test_gate_head_means_and_per_head_constant_lesion():
    model = _tiny("lega")
    val = (np.arange(4096, dtype=np.int64) % 64)
    means = gate_head_means(model, val, ctx=32, device="cpu", eval_iters=2)
    assert len(means) == 2
    assert all(m is not None and len(m) == 2 for m in means)
    assert all(0.0 < g < 1.0 for row in means for g in row)

    snap = snapshot_gate_params(model)
    rec = apply_lesion(model, "gate_constant_mean", gate_head_mean=means)
    assert rec["applied"] is True
    for li, m in enumerate(model.blocks):
        for h in range(2):
            expected = math.log(means[li][h] / (1.0 - means[li][h]))
            assert abs(float(m.attn.gate_d[h].detach()) - expected) < 1e-5
            assert torch.count_nonzero(m.attn.gate_w) == 0
            assert torch.count_nonzero(m.attn.gate_a) == 0
            assert torch.count_nonzero(m.attn.gate_b) == 0
    restore_gate_params(model, snap)
    # 形状不匹配必须显式拒绝，而不是静默写一半。
    bad = apply_lesion(model, "gate_constant_mean",
                       gate_head_mean=[[0.5]])
    assert bad["applied"] is False
    print("  [ok] 逐头恒定门保留各头均值、抹平位置变化；形状错误显式拒绝")


def test_run_ablation_end_to_end_tiny_checkpoint():
    model = _tiny("lega")
    with torch.no_grad():
        for block in model.blocks:
            block.attn.gate_a.fill_(0.2)
            block.attn.gate_b.fill_(-0.1)
            block.attn.gate_w.fill_(0.01)
    with tempfile.TemporaryDirectory() as d:
        ckpt = os.path.join(d, "model_best.pt")
        save_checkpoint(ckpt, model, extra={"seed": 1, "variant": "lega"})
        with open(os.path.join(d, "config_used.json"), "w",
                  encoding="utf-8") as f:
            json.dump({"train_ctx": 16, "eval_iters": 2, "amp": False,
                       "amp_dtype": None}, f)
        tokens = (np.arange(1024, dtype=np.uint16) % 64).astype(np.uint16)
        tokens.tofile(os.path.join(d, "val.bin"))
        # 显式 seed / perm_seed 不能经 int() 截断：True→1、1.5→1 会让落盘
        # seed 与实际播种不一致，甚至两次声明不同置换实际相同。
        for bad in (True, 1.5, -1, 2**32):
            try:
                run_ablation(ckpt, d, eval_iters=2, ctx=16,
                             device="cpu", seed=bad)
                raise AssertionError(f"seed={bad!r} 应显式报错")
            except ValueError:
                pass
        try:
            run_ablation(ckpt, d, eval_iters=2, ctx=16, device="cpu",
                         seed=1, perm_seed=True)
            raise AssertionError("perm_seed=True 应显式报错")
        except ValueError:
            pass
        payload = run_ablation(ckpt, d, eval_iters=2, ctx=16,
                               device="cpu", seed=1)
        assert payload["variant"] == "lega"
        assert payload["split"] == "val"
        assert payload["eval_ctx"] == 16 and payload["eval_iters"] == 2
        assert len(payload["lesions"]) == len(LESION_ORDER)
        assert payload["baseline"]["val_loss"] is not None
        ids = {r["lesion"]: r for r in payload["lesions"]}
        assert ids["gate_constant_mean"]["applied"] is True
        assert ids["gate_constant_mean"]["gate_std_changed"] is True
        assert ids["no_query"]["applied"] is True
        # 同一个 payload 内每个病灶的窗口必须与基线一致。
        assert all(r.get("n_windows") == ids["no_query"]["n_windows"]
                   for r in payload["lesions"] if r.get("applied"))
    print("  [ok] run_ablation 端到端：tiny checkpoint + 合成 val.bin，"
          "逐窗口配对与逐头恒定门均生效")

def test_window_delta_stats_pairs_by_window():
    base = [1.0, 2.0, 3.0, 4.0]
    lesion = [1.5, 2.5, 3.5, 4.5]
    st = _window_delta_stats(base, lesion)
    assert st["n_windows"] == 4
    assert abs(st["delta_loss_mean"] - 0.5) < 1e-12
    assert st["delta_losses"] == [0.5, 0.5, 0.5, 0.5]
    assert st["p_window_t"] is not None
    # 坏窗口 loss 不能污染均值/有效样本量：NaN、bool、下划线数字串按缺失剔除，
    # 并如实报告被剔除的窗口数。
    st_bad = _window_delta_stats(
        [1.0, float("nan"), 3.0, 4.0, True, "1_0"],
        [2.0, 2.0, 4.0, float("inf"), 5.0, 6.0])
    assert st_bad["n_windows"] == 2, st_bad
    assert st_bad["n_invalid_windows"] == 4, st_bad
    assert st_bad["delta_losses"] == [1.0, 1.0], st_bad
    assert math.isfinite(st_bad["delta_loss_mean"]), st_bad
    assert _window_delta_stats([float("nan")], [1.0])["n_windows"] == 0
    # 合法但接近 float 上限的窗口差：朴素 np.mean/np.std 会溢出成 inf；
    # 点估计/标准误必须仍是有限值（不能把坏 JSON 值写进 ablation 结果）。
    huge = _window_delta_stats([1e308, 1e308], [0.0, 0.0])
    assert huge["n_windows"] == 2, huge
    assert huge["delta_loss_mean"] == -1e308, huge
    assert huge["delta_loss_se"] == 0.0, huge
    print("  [ok] 病灶窗口差按位置配对，Δ=lesion−normal；"
          "坏窗口 loss 不污染均值与配对样本量；巨大有限差值不溢出")


def test_ablate_numeric_and_seed_guards():
    """新模块的严格数值/种子口径与全项目一致。"""
    for poor in ("1_0", True, np.bool_(True), float("nan"),
                 float("inf"), None, "bad"):
        assert _finite_or_none(poor) is None, poor
    assert _finite_or_none("0.25") == 0.25
    for poor in (True, 1.5, -1, "1_0"):
        try:
            _layer_permutation(2, poor, 0)
            raise AssertionError(f"perm_seed={poor!r} 应显式报错")
        except ValueError:
            pass
    perm = _layer_permutation(4, 7, 0)
    assert sorted(perm.tolist()) == [0, 1, 2, 3]
    assert not np.array_equal(perm, np.arange(4))
    # 逐头恒定门均值同样拒绝坏类型/下划线数字串，不静默写成合法门控。
    model = _tiny("lega")
    snap = snapshot_gate_params(model)
    bad = apply_lesion(model, "gate_constant_mean",
                       gate_head_mean=[["0.2_5", 0.5], [0.5, 0.5]])
    assert bad["applied"] is False, bad
    restore_gate_params(model, snap)
    # 快照形状不符时 copy_ 会按广播静默写入：必须显式拒绝，不能留下半恢复
    # 的门控参数。
    bad_snap = dict(snap)
    bad_snap[next(iter(bad_snap))] = torch.zeros(1)
    try:
        restore_gate_params(model, bad_snap)
        raise AssertionError("形状不符的快照应显式报错")
    except (ValueError, RuntimeError):
        pass
    restore_gate_params(model, snap)
    print("  [ok] 病灶消融：坏数值/截断种子显式拒绝，头置换仍确定性非恒等；"
          "快照形状不符不静默广播")



def _payload(variant, seed, lesion_deltas, no_op=False):
    lesions = []
    for lid, delta in lesion_deltas.items():
        lesions.append({
            "lesion": lid, "applicable": True, "applied": True,
            "no_op": bool(no_op), "delta_loss_mean": delta,
            "n_windows": 10,
        })
    return {"variant": variant, "seed": seed, "lesions": lesions}


def test_causal_ablation_uses_seed_level_paired_inference():
    deltas = [0.09, 0.10, 0.11, 0.10, 0.10]
    ablations = {"lega": {s + 1: _payload("lega", s + 1,
                                          {"no_query": d})
                          for s, d in enumerate(deltas)}}
    tests = _ablation_lesion_records(ablations)
    assert len(tests) == 1
    t = tests[0]
    assert t["metric"] == CAUSAL_ABLATION_METRIC
    assert t["n"] == 5
    assert abs(t["mean_diff"] - 0.10) < 1e-12
    assert t["mean_ppl_ratio"] > 1.0
    assert t["causal_contribution"] is True
    assert "推理期该机制有因果贡献" in t["causal_verdict"]
    print("  [ok] 因果消融：先按种子聚合 ΔNLL，再做跨种子配对+Holm")


def test_causal_ablation_marks_noop_and_skips_not_applicable():
    # 全部种子 no-op：系数已为 0，不能把“无差异”当成“无因果作用”的证据。
    ablations = {"lega": {s: _payload("lega", s, {"no_conc": 0.0}, no_op=True)
                          for s in (1, 2, 3)}}
    tests = _ablation_lesion_records(ablations)
    t = tests[0]
    assert t["n_no_op_seeds"] == 3
    assert t["causal_contribution"] is False
    assert "no-op" in t["causal_verdict"]
    # 不适用记录不进入检验。
    ablations2 = {"lega": {1: {"variant": "lega", "seed": 1, "lesions": [
        {"lesion": "no_len", "applicable": False, "reason": "缺少 gate_b"}]}}}
    assert _ablation_lesion_records(ablations2) == []
    # 未真正施加（applied=false 或缺少 applied）的坏记录即使带着 ΔNLL，
    # 也不能进入跨种子检验，否则会把“没做干预”伪装成效应。
    for rec in ({"lesion": "no_query", "applicable": True,
                 "applied": False, "delta_loss_mean": 0.5, "n_windows": 10},
                {"lesion": "no_query", "applicable": True,
                 "delta_loss_mean": 0.5, "n_windows": 10}):
        bad = {"lega": {s: {"variant": "lega", "seed": s, "lesions": [dict(rec)]}
                        for s in (1, 2, 3)}}
        assert _ablation_lesion_records(bad) == [], rec
    print("  [ok] 因果消融：no-op 与不适用/未施加记录如实标注，"
          "不伪造 0 效应")


def test_inference_factorial_cells_complete_2x3_and_zero_only_targets():
    """推理期补齐 2^3：单因子格只清零目标项，不动其它门控参数。

    训练期全因子改变优化轨迹；同一 checkpoint 内的 2^3 干预才能回答
    “最终前向是否真的使用该组件/交互”。这里核对每个格子的清零模式，
    防止把 B-only 误写成“去掉 A”、或把 no_len 误当成“完全移除长度依赖”。
    """
    m = _tiny("lega")
    with torch.no_grad():
        for block in m.blocks:
            block.attn.gate_w.normal_(0, 0.05)
            block.attn.gate_a.normal_()
            block.attn.gate_b.normal_()
            block.attn.gate_d.normal_()
    snap = snapshot_gate_params(m)
    expected_zero = {
        "no_evidence": {"gate_w", "gate_a", "gate_b"},
        "query_only": {"gate_a", "gate_b"},
        "conc_only": {"gate_w", "gate_b"},
        "len_only": {"gate_w", "gate_a"},
        "no_len": {"gate_b"},
        "no_conc": {"gate_a"},
        "no_query": {"gate_w"},
    }
    assert set(INFERENCE_FACTORIAL_CELLS) == set(CELL_ORDER)
    assert LESION_BY_ID["query_only"].get("factorial_cell") is True
    assert LESION_BY_ID["conc_only"].get("factorial_cell") is True
    assert LESION_BY_ID["len_only"].get("factorial_cell") is True
    for cell, lesion_id in INFERENCE_FACTORIAL_CELLS.items():
        if lesion_id is None:
            assert cell == "ABC"
            continue
        rec = apply_lesion(m, lesion_id, perm_seed=0)
        assert rec["applied"] is True, (cell, lesion_id, rec)
        for li, block in enumerate(m.blocks):
            attn = block.attn
            for name in ("gate_w", "gate_a", "gate_b"):
                p = getattr(attn, name)
                key = f"blocks.{li}.attn.{name}"
                if name in expected_zero[lesion_id]:
                    assert torch.count_nonzero(p) == 0, (cell, name)
                else:
                    assert torch.equal(p, snap[key]), (cell, name)
            # d 永远保留（none 格只留学习到的偏置）。
            assert torch.equal(attn.gate_d, snap[f"blocks.{li}.attn.gate_d"])
        restore_gate_params(m, snap)
    print("  [ok] 推理期 2^3 单因子格：只清零目标项，d 始终保留")


def _inference_payload(seed, const=3.0, weights=None):
    """按可加价值函数 v=const+Σw_i 生成 8 格 PPL，Shapley 应精确等于 w_i。"""
    weights = dict(weights or {"A": 0.2, "B": 0.3, "C": 0.4})
    # ABC 基线必须包含全部因子；否则“总效应”会被错误地定义成负值。
    base_ppl = math.exp(-(const + sum(weights.values())))
    recs = []
    for cell, lesion_id in INFERENCE_FACTORIAL_CELLS.items():
        if lesion_id is None:
            continue
        factors = set() if cell == "none" else set(cell)
        v = const + sum(weights[f] for f in factors)
        ppl = math.exp(-v)
        recs.append({"lesion": lesion_id, "applied": True,
                     "window_indices": [0, 1, 2], "val_ppl": ppl})
    return {"variant": "lega", "seed": seed, "window_indices": [0, 1, 2],
            "baseline": {"val_ppl": base_ppl}, "lesions": recs}


def test_inference_shapley_recovers_additive_components_across_seeds():
    weights = {"A": 0.2, "B": 0.3, "C": 0.4}
    abl = {"lega": {s: _inference_payload(s, 3.0 + 0.01 * s, weights)
                    for s in (1, 2, 3)}}
    out = _inference_shapley_tests(abl)
    assert len(out) == 3, out
    by = {t["shapley_factor"]: t for t in out}
    for f, w in weights.items():
        t = by[f]
        assert t["metric"] == INFERENCE_SHAPLEY_METRIC
        assert abs(t["mean_diff"] - w) < 1e-9, (f, t["mean_diff"])
        assert abs(t["shapley_ppl_ratio"] - math.exp(-w)) < 1e-9
        assert t["n"] == 3
        assert t["inference_time"] is True
        assert t["shapley_efficiency_gap_max"] < 1e-9

    # 缺格/窗口不一致时必须整条跳过，而不是把缺失格当 0。
    broken = {"lega": {"1": _inference_payload(1)}}
    broken["lega"]["1"]["lesions"] = broken["lega"]["1"]["lesions"][:-1]
    assert _inference_shapley_tests(broken) == []
    mism = {"lega": {"1": _inference_payload(1)}}
    mism["lega"]["1"]["lesions"][0]["window_indices"] = [0, 1]
    assert _inference_shapley_tests(mism) == []
    # 单因子格相对 Full 同时去掉另外两个因子，不能混进单病灶因果表。
    causal_payload = _inference_payload(1)
    for rec in causal_payload["lesions"]:
        rec["delta_loss_mean"] = 0.1
    causal = _ablation_lesion_records({"lega": {"1": causal_payload}})
    assert causal, causal
    assert not any(t.get("lesion") in {"query_only", "conc_only", "len_only"}
                   for t in causal), causal
    print("  [ok] 推理期 Shapley：可加价值下精确还原 φ，缺格/错窗跳过；"
          "单因子格不混入单病灶因果表")


def _inference_ppl_payload(seed, values):
    """用原始 PPL 单元格构造推理期 2^3 payload（交互检验与 Shapley 同结构）。"""
    recs = []
    for cell, lesion_id in INFERENCE_FACTORIAL_CELLS.items():
        if lesion_id is None:
            continue
        recs.append({"lesion": lesion_id, "applied": True,
                     "window_indices": [0, 1, 2],
                     "val_ppl": float(values[cell])})
    return {"variant": "lega", "seed": seed, "window_indices": [0, 1, 2],
            "baseline": {"val_ppl": float(values["ABC"])}, "lesions": recs}


def test_inference_interactions_match_training_scale_and_skip_bad_seeds():
    """推理期交互：可加 PPL 下 A×B/A×C/B×C 为 0；纯 A×B 交互只落在 A×B。

    这里刻意用原始 PPL（与训练期 `PPL交互` 同尺度），避免把 Shapley 的
    −log 口径直接搬来后与训练期符号不可比。缺格/错窗/非 lega 必须整条
    剔除，不能按 0 补齐伪造“可加性成立”。
    """
    additive = {"none": 10.0, "A": 11.0, "B": 12.0, "C": 14.0,
                "AB": 13.0, "AC": 15.0, "BC": 16.0, "ABC": 17.0}
    abl = {"lega": {s: _inference_ppl_payload(s, additive)
                    for s in (1, 2, 3)}}
    out = _inference_interaction_tests(abl)
    assert len(out) == 3, out
    by = {t["interaction"]: t for t in out}
    assert all(t["metric"] == INFERENCE_INTERACTION_METRIC for t in out)
    assert all(abs(t["mean_diff"]) < 1e-9 for t in out), by
    assert all(t["inference_time"] is True and t["no_direction"] is True
               for t in out)

    pure = dict(additive)
    pure["AB"] += 2.0
    pure["ABC"] += 2.0
    abl = {"lega": {s: _inference_ppl_payload(s, pure) for s in (1, 2, 3)}}
    by = {t["interaction"]: t for t in _inference_interaction_tests(abl)}
    assert abs(by["A×B"]["mean_diff"] - 2.0) < 1e-9, by["A×B"]
    assert abs(by["A×C"]["mean_diff"]) < 1e-9, by["A×C"]
    assert abs(by["B×C"]["mean_diff"]) < 1e-9, by["B×C"]

    huge = {"none": 1.7e308, "A": 1e-308, "B": 1e-308, "C": 1e-308,
            "AB": 1.7e308, "AC": 1e-308, "BC": 1e-308, "ABC": 1e-308}
    abl = {"lega": {s: _inference_ppl_payload(s, huge) for s in (1, 2, 3)}}
    by = {t["interaction"]: t for t in _inference_interaction_tests(abl)}
    assert abs(by["A×B"]["mean_diff"] - 1.7e308) < 1e299, by["A×B"]
    assert by["A×B"]["excluded_protocol_mismatches"] == 0, by["A×B"]
    assert abs(by["A×C"]["mean_diff"]) < 1e-9, by["A×C"]
    assert abs(by["B×C"]["mean_diff"]) < 1e-9, by["B×C"]

    # 交互值本身超出 float 表示范围（八格仍有限）：该种子不能悄悄消失，
    # 必须计入排除数；其余可表示种子照常进入检验。
    unrepresentable = {"none": 1.7e308, "A": 1e-308, "B": 1e-308,
                       "C": 1.7e308, "AB": 1.7e308, "AC": 1e-308,
                       "BC": 1e-308, "ABC": 1.7e308}
    abl = {"lega": {1: _inference_ppl_payload(1, additive),
                    2: _inference_ppl_payload(2, unrepresentable)}}
    by = {t["interaction"]: t for t in _inference_interaction_tests(abl)}
    assert by["A×B"]["n_protocol_seeds"] == 1, by["A×B"]
    assert by["A×B"]["excluded_protocol_mismatches"] == 1, by["A×B"]

    # 一个种子缺格：该种子整条剔除，n=2 且如实计入排除数。
    broken = {"lega": {s: _inference_ppl_payload(s, additive)
                       for s in (1, 2, 3)}}
    broken["lega"][3]["lesions"] = broken["lega"][3]["lesions"][:-1]
    out = _inference_interaction_tests(broken)
    assert out and all(t["n"] == 2 for t in out), out
    assert all(t["excluded_protocol_mismatches"] == 1 for t in out), out
    # 窗口不一致同缺格；非 lega 的 2^3 本来不完整，不消费也不计入排除。
    bad_win = {"lega": {s: _inference_ppl_payload(s, additive)
                        for s in (1, 2, 3)}}
    bad_win["lega"][1]["lesions"][0]["window_indices"] = [0, 1]
    assert _inference_interaction_tests(bad_win)[0]["n"] == 2
    assert _inference_interaction_tests(
        {"qgate": {s: _inference_ppl_payload(s, additive)
                   for s in (1, 2, 3)}}) == []
    print("  [ok] 推理期交互：同训练期 PPL 尺度、纯交互归属正确，"
          "缺格/错窗整条剔除")


def test_inference_interaction_renderer_flags_sign_disagreement():
    """汇总页必须把训练期/推理期交互符号不一致显式标 ⚠，而不是只报一期。"""
    infer = [{"interaction": "A×B", "mean_diff": -1.5, "std_diff": 0.2,
              "n": 3, "p_adj": 0.01, "verdict": "显著偏离可加性",
              "training_interaction_mean": 2.0,
              "training_inference_sign_agree": False}]
    md = "\n".join(_render_inference_interactions(infer, []))
    assert "A×B" in md and "否 ⚠" in md and "-1.5000" in md, md
    absent = "\n".join(_render_inference_interactions([], []))
    assert "无法计算推理期交互" in absent
    print("  [ok] 推理期交互汇总：训练/推理符号冲突显式标注，缺证据不静默")


def test_inference_interaction_renderer_tolerates_unhashable_labels():
    """外部/旧产物的交互标签可能是 list/dict；不能让 dict 查找崩溃整页汇总。"""
    infer = [{"interaction": ["A", "B"], "mean_diff": 1.0, "n": 3}]
    train = [{"interaction": ["A", "B"], "metric": "PPL交互",
              "mean_diff": 2.0}]
    md = "\n".join(_render_inference_interactions(infer, train))
    assert "推理期全因子交互效应" in md and "1.0000" in md, md
    # 不可哈希标签无法与训练期结果可靠对齐，按缺失处理而不是硬塞进 dict。
    assert "2.0000" not in md, md
    print("  [ok] 推理期交互汇总：不可哈希标签按缺失处理，不崩溃")


def test_inference_shapley_only_uses_full_lega_checkpoint():
    """推理期 2^3 只在 Full 模型上定义；qgate/placebo 的缺项格不能混入。

    训练期 Shapley 由 lega 的 8 个全因子格定义；qgate 只有 A、placebo 的
    B/C 是噪声，对它们施加同一套病灶会产生大量 no-op 格，且汇总页没有
    变体列，混入后会渲染成无法区分的重复因子行。
    """
    payload = _inference_payload(1)
    full = _inference_shapley_tests({"lega": {1: payload}})
    assert len(full) == 3, full
    assert _inference_shapley_tests({"qgate": {1: payload}}) == []
    assert _inference_shapley_tests({"lega_placebo": {1: payload}}) == []
    print("  [ok] 推理期 Shapley 只使用 Full(=lega) checkpoint，"
          "缺项变体不产生重复因子行")


def test_shapley_shares_never_emit_nonfinite_ratios():
    """有限 φ/total 相除溢出时，占比不可表示，不能返回 inf 污染 JSON。"""
    assert shares({"A": 1.0, "B": 3.0}, 4.0) == {"A": 0.25, "B": 0.75}
    assert shares({"A": -1.0}, 0.0) == {}
    # 极大 φ / 极小 total：真实占比超出 float 范围，按无法表示处理。
    assert shares({"A": 1e308, "B": 1e308}, 1e-11) == {}
    assert shares({"A": 1e308}, 1e308) == {"A": 1.0}
    print("  [ok] Shapley 占比：极小 total/溢出时返回空表，不写出 inf")


def test_shapley_rescales_opposite_huge_cell_values():
    """有限单元格之差溢出后，Shapley 仍须给出可表示的有限结果。"""
    a = 1e308
    values = {"none": 0.0, "A": 0.0, "B": -a, "AB": a,
              "C": a, "AC": -a, "BC": 0.0, "ABC": 0.0}
    phi = shapley_values(values)
    assert all(abs(v) < 1e-12 for v in phi.values()), phi
    dec = decomposition(values)
    assert dec["total_effect"] == 0.0
    assert abs(dec["efficiency_gap"]) < 1e-12
    # 真实总效应（2e308）不可表示：显式拒绝，不能把 inf 写进报告；
    # 单因子 φ 仍可表示时照常返回。
    bad = {c: 0.0 for c in CELL_ORDER}
    bad["ABC"], bad["none"] = 1e308, -1e308
    phi_bad = shapley_values(bad)
    assert all(math.isfinite(v) for v in phi_bad.values()), phi_bad
    try:
        decomposition(bad)
        raised = False
    except ValueError:
        raised = True
    assert raised
    print("  [ok] Shapley：反向极值先缩放再差分，总效应不可表示时显式拒绝")
