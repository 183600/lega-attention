"""原始集中度参数化与参数化稳健性分析的单元测试。"""
import json
import math
import os
import tempfile

import torch

from lega.attention import (GATE_FACTORS, RUNNABLE_VARIANTS, SUPPLEMENTAL_VARIANTS,
                            VARIANT_CN, CausalSelfAttention)
from lega.factorial import CELL_ORDER
from lega.model import expected_gate_extras, gate_extra_params
from lega.reparameterization import (CONTRASTS, NORMALIZED_CELLS, RAW_CELLS,
                                     RAW_FACTORIAL_VARIANTS, analyze,
                                     contrast_effects,
                                     contrast_parameterization_tests)
from run_all import (_render_reparameterization, _reparameterization_analysis,
                      build_summary)
from tests.pipeline_helpers import _write_json


def _ppl_cell(ppl, seed=1):
    return {"eval": {"variant": "x", "seed": seed, "split": "val",
                     "val_ppl": float(ppl)}}


def _full_grid(seed, raw_b_ppl=7.0):
    """同一组 A/B/C 因子的归一化/原始 PPL 网格。"""
    norm = {"none": 10.0, "A": 9.0, "B": 8.0, "C": 6.0,
            "AB": 7.0, "AC": 5.0, "BC": 4.0, "ABC": 3.0}
    raw = dict(norm)
    raw["B"] = raw_b_ppl
    values = {}
    for cell in CELL_ORDER:
        values[NORMALIZED_CELLS[cell]] = norm[cell]
    for cell in CELL_ORDER:
        values[RAW_CELLS[cell]] = raw[cell]
    return {seed: values}


def test_raw_factorial_design_is_complete_and_named():
    """原始参数化必须补齐与归一化网格同构的 8 格。"""
    assert tuple(NORMALIZED_CELLS) == CELL_ORDER
    assert tuple(RAW_CELLS) == CELL_ORDER
    assert len(set(RAW_CELLS.values())) == 8
    assert len(set(NORMALIZED_CELLS.values())) == 8
    assert set(RAW_FACTORIAL_VARIANTS) == set(RAW_CELLS.values())
    assert set(RAW_FACTORIAL_VARIANTS) <= set(RUNNABLE_VARIANTS)
    for variant in ("lega_raw_conc_only", "lega_raw_no_query"):
        assert variant in SUPPLEMENTAL_VARIANTS
        assert variant not in GATE_FACTORS
        assert variant in VARIANT_CN
    assert RAW_CELLS["none"] == NORMALIZED_CELLS["none"] == "gate_only"
    assert RAW_CELLS["A"] == NORMALIZED_CELLS["A"] == "qgate"
    assert RAW_CELLS["C"] == NORMALIZED_CELLS["C"] == "lega_len_only"
    assert RAW_CELLS["AC"] == NORMALIZED_CELLS["AC"] == "lega_no_conc"
    from lega.attention import VARIANTS
    assert set(RAW_FACTORIAL_VARIANTS) - set(VARIANTS) == {
        "lega_raw", "lega_raw_no_len", "lega_raw_conc_only",
        "lega_raw_no_query"}
    print("  [ok] 原始参数化 8 格完整，且与归一化网格共享 none/A/C 结构")


def test_raw_only_variants_forward_and_param_budget():
    """补齐的 raw 变体可训练、参数预算与公式一致。"""
    for variant in ("lega_raw_conc_only", "lega_raw_no_query"):
        torch.manual_seed(0)
        m = CausalSelfAttention(32, 2, variant=variant,
                                use_sdpa_standard=False)
        assert gate_extra_params(m) == expected_gate_extras(
            variant, 16, 2, 1), variant
        x = torch.randn(2, 6, 32)
        out, _, stats = m(x, rope=None, need_stats=True)
        assert torch.isfinite(out).all(), variant
        assert stats["max_p"] is not None, variant
        (out * out).mean().backward()
        for name, p in m.named_parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), (variant, name)
    print("  [ok] raw_conc_only/raw_no_query 前向、反向与参数预算")


def test_raw_gate_formula_does_not_include_log_n_for_b_alone():
    """原始 B 只乘 log max_p；B+C 格才显式加入 b*log n。"""
    max_p = torch.full((1, 1, 1), 0.25)
    log_n = torch.full((1, 1, 1), math.log(8.0))
    for variant, expected in (
            ("lega_raw_conc_only", 0.5 * math.log(0.25)),
            ("lega_raw_no_query",
             0.5 * math.log(0.25) - 0.25 * math.log(8.0))):
        m = CausalSelfAttention(16, 1, variant=variant,
                                use_sdpa_standard=False)
        with torch.no_grad():
            m.gate_a.fill_(0.5)
            if hasattr(m, "gate_b"):
                m.gate_b.fill_(-0.25)
            m.gate_d.zero_()
        _, logit, _ = m._gate_logit(None, max_p, log_n, (1, 1, 1),
                                    keep_terms=True)
        assert abs(logit.item() - expected) < 1e-6, (variant, logit.item())
    print("  [ok] 原始 B 不含 log n；B+C 格才含显式长度项")


def test_contrast_difference_detects_parameterization_shift():
    """B 的增量效应在原始参数化下更大时，D 检验必须检出。"""
    values = {}
    for seed in (1, 2, 3, 4, 5):
        values.update(_full_grid(seed))
    tests = contrast_parameterization_tests(values, "PPL")
    got = {(t["factor"], t["direction"]): t for t in tests}
    b_add = got[("B", "add")]
    expected = math.log(10.0 / 7.0) - math.log(10.0 / 8.0)
    assert abs(b_add["mean_diff"] - expected) < 1e-9, b_add
    assert b_add["n"] == 5
    assert b_add["p_adj"] is not None and b_add["p_adj"] < 0.05
    assert "参数化依赖" in b_add["verdict"], b_add
    # 原始/归一化 full 与 no_conc 都没变，B 的去一法边不应出现差异。
    assert abs(got[("B", "remove")]["mean_diff"]) < 1e-12
    print("  [ok] 参数化差异检验检出 B 的长度归一化分量")


def test_identical_grids_give_zero_parameterization_difference():
    """两套网格完全等价时，所有 Shapley/单因子差异都应为 0。"""
    values = {}
    for seed in (1, 2, 3, 4):
        values.update(_full_grid(seed, raw_b_ppl=8.0))
    result = analyze(values, "PPL")
    assert result["summary"]["status"] == "no_difference_detected", result
    assert result["summary"]["n_significant"] == 0
    for test in result["contrasts"] + result["shapley"]:
        assert abs(test["mean_diff"]) < 1e-12, test
        assert test["p_adj"] == 1.0, test
    assert all(t["factor"] for t in result["shapley"])
    print("  [ok] 等价网格：差异为 0 且不误报参数化依赖")


def test_insufficient_seeds_are_not_called_robust():
    """n<3 时不能把“未检出”写成稳健。"""
    values = {}
    for seed in (1, 2):
        values.update(_full_grid(seed, raw_b_ppl=7.0))
    result = analyze(values, "PPL")
    assert result["summary"]["status"] == "insufficient", result["summary"]
    assert "无法评估" in result["summary"]["verdict"]
    print("  [ok] 参数化稳健性：n<3 显式降级为证据不足")


def _pipeline_runs():
    runs = {}
    for seed in (1, 2, 3):
        for variant, ppl in _full_grid(seed, raw_b_ppl=7.0)[seed].items():
            runs.setdefault(variant, {})[seed] = _ppl_cell(ppl, seed)
    return runs


def test_pipeline_reparameterization_requires_both_complete_grids():
    """流水线只在两套网格完整且同种子可比时给出分析。"""
    runs = _pipeline_runs()
    out = _reparameterization_analysis(runs)
    assert out["available"] is True, out
    assert out["missing_variants"] == []
    assert out["not_run"] is False
    ppl = out["metrics"]["PPL"]
    assert ppl["n_protocol_seeds"] == 3
    assert ppl["summary"]["status"] == "parameterization_dependent"
    assert ppl["summary"]["n_significant"] >= 1

    partial = {v: m for v, m in runs.items()
               if v not in ("lega_raw", "lega_raw_no_query")}
    out = _reparameterization_analysis(partial)
    assert out["available"] is False
    assert out["not_run"] is False
    assert "lega_raw" in out["missing_variants"]
    assert "lega_raw_no_query" in out["missing_variants"]
    md = "\n".join(_render_reparameterization(out))
    assert "参数化稳健性分析不可用" in md
    print("  [ok] 流水线：完整双网格才分析，缺格给出可机读原因")


def test_pipeline_reparameterization_not_run_is_silent():
    """默认主矩阵未跑 raw 对照时，汇总不新增噪声小节。"""
    runs = _pipeline_runs()
    runs = {v: m for v, m in runs.items() if v not in RAW_FACTORIAL_VARIANTS}
    out = _reparameterization_analysis(runs)
    assert out["not_run"] is True
    assert out["available"] is False
    assert _render_reparameterization(out) == []
    print("  [ok] 默认矩阵未运行 raw 对照时，参数化小节保持静默")


def test_build_summary_wires_reparameterization_into_report_and_json():
    """完整双网格时汇总页/significance.json 都消费参数化稳健性结果。"""
    with tempfile.TemporaryDirectory() as tmp:
        for seed in (1, 2, 3):
            for variant, ppl in _full_grid(seed, raw_b_ppl=7.0)[seed].items():
                _write_json(os.path.join(
                    tmp, f"eval_{variant}_seed{seed}.json"), {
                        "variant": variant, "seed": seed, "split": "val",
                        "val_ppl": ppl, "val_loss": math.log(ppl),
                        "n_params": 100, "n_gate_extra": 4})
        out = build_summary(tmp)
        assert out["reparameterization"]["available"] is True
        with open(os.path.join(tmp, "significance.json"),
                  "r", encoding="utf-8") as f:
            sig = json.load(f)
        assert sig["reparameterization"]["available"] is True
        assert "参数化稳健性（原始 log max_p vs 归一化 c；探索性）" in \
            out["markdown"]
        assert "B（add）" in out["markdown"]
    print("  [ok] build_summary 将参数化稳健性写入报告与 significance.json")


def test_build_summary_omits_reparameterization_section_when_not_run():
    """默认矩阵未跑 raw 对照时，summary 不出现空的参数化小节。"""
    with tempfile.TemporaryDirectory() as tmp:
        for seed in (1, 2, 3):
            for variant, ppl in (("standard", 10.0), ("lega", 8.0),
                                 ("gate_only", 9.0), ("qgate", 8.5)):
                _write_json(os.path.join(
                    tmp, f"eval_{variant}_seed{seed}.json"), {
                        "variant": variant, "seed": seed, "split": "val",
                        "val_ppl": ppl, "val_loss": math.log(ppl),
                        "n_params": 100, "n_gate_extra": 4})
        out = build_summary(tmp)
        assert out["reparameterization"]["not_run"] is True
        assert "参数化稳健性（原始" not in out["markdown"]
    print("  [ok] 未运行 raw 对照时汇总保持静默")
