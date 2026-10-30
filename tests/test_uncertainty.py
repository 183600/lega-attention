"""种子 × 评测单元不确定性分解与双层 bootstrap 的回归测试。"""
import os
import shutil
import tempfile

import numpy as np

from lega.uncertainty import (aggregate_variance_components,
                              two_way_variance_components)
from tests.pipeline_helpers import _write_json


def _close(a, b, tol=1e-10):
    return abs(float(a) - float(b)) <= tol


def test_two_way_variance_components_recovers_additive_components():
    seed_effect = np.asarray([-1.5, -0.5, 0.5, 1.5])
    unit_effect = np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0])
    matrix = seed_effect[:, None] + unit_effect[None, :]
    comp = two_way_variance_components(matrix.tolist())
    assert comp is not None
    assert comp["n_seeds"] == 4 and comp["n_units"] == 5
    assert _close(comp["var_seed"], np.var(seed_effect, ddof=1))
    assert _close(comp["var_unit"], np.var(unit_effect, ddof=1))
    assert _close(comp["var_resid"], 0.0)
    assert _close(comp["share_seed"] + comp["share_unit"]
                  + comp["share_resid"], 1.0)
    assert _close(comp["share_mean_seed"] + comp["share_mean_unit"]
                  + comp["share_mean_resid"], 1.0)
    assert comp["negative_seed_component"] is False
    assert comp["negative_unit_component"] is False
    print("  [ok] 两因素方差分解：可加模型精确还原种子/单元分量")


def test_two_way_variance_components_truncates_negative_estimates():
    # 拉丁方：每行/每列均值相同，种子与单元分量均为负，残差为正。
    rows = [[0.0, 1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0, 0.0],
            [2.0, 3.0, 0.0, 1.0],
            [3.0, 0.0, 1.0, 2.0]]
    comp = two_way_variance_components(rows)
    assert comp is not None
    assert comp["var_seed"] == 0.0 and comp["var_unit"] == 0.0
    assert comp["negative_seed_component"] is True
    assert comp["negative_unit_component"] is True
    assert comp["share_seed"] == 0.0 and comp["share_unit"] == 0.0
    assert comp["share_resid"] == 1.0
    assert comp["var_resid"] > 0.0
    print("  [ok] 两因素方差分解：负分量截断为 0 并显式标记")


def test_two_way_variance_components_rejects_invalid_matrices():
    for bad in (None, [], [[1.0, 2.0]], [[1.0, 2.0], [3.0]],
                [[1.0, float("nan")], [2.0, 3.0]],
                [[1.0, float("inf")], [2.0, 3.0]],
                [[1.0, "x"], [2.0, 3.0]], "abc"):
        assert two_way_variance_components(bad) is None, bad
    print("  [ok] 两因素方差分解：空/畸形/非有限输入不伪造结果")


def test_aggregate_variance_components_uses_cell_weights():
    a = {"share_seed": 1.0, "share_unit": 0.0, "share_resid": 0.0,
         "share_mean_seed": 0.8, "share_mean_unit": 0.2,
         "share_mean_resid": 0.0}
    b = {"share_seed": 0.0, "share_unit": 0.5, "share_resid": 0.5,
         "share_mean_seed": 0.0, "share_mean_unit": 1.0,
         "share_mean_resid": 0.0}
    agg = aggregate_variance_components([a, b], [1.0, 3.0])
    assert agg is not None and agg["n_cells"] == 2
    assert _close(agg["share_seed"], 0.25)
    assert _close(agg["share_unit"], 0.375)
    assert _close(agg["share_resid"], 0.375)
    assert _close(agg["share_mean_seed"], 0.2)
    assert _close(agg["share_mean_unit"], 0.8)
    assert aggregate_variance_components([], [1.0]) is None
    assert aggregate_variance_components([a, b], [1.0]) is None
    assert aggregate_variance_components([a, b], [0.0, 0.0]) is None
    print("  [ok] 方差分量聚合：按单元格权重加权、缺失权重不补齐")


def test_ruler_macro_hierarchical_bootstrap_deterministic_and_structure_aware():
    from run_all import _ruler_macro_hierarchical_bootstrap

    per_seed = [{("a", "1"): [1.0, 1.0], ("a", "2"): [1.0, 1.0]}
                for _ in range(3)]
    r = _ruler_macro_hierarchical_bootstrap(per_seed, n_boot=100)
    assert r is not None
    assert r["ci95"] == [1.0, 1.0]
    assert r["observed"] == 1.0
    assert r["n_seeds"] == 3 and r["n_cells"] == 6 and r["n_samples"] == 12
    assert r == _ruler_macro_hierarchical_bootstrap(per_seed, n_boot=100)
    assert 0.0 <= r["p"] <= 1.0 and r["same_sign"] == 1.0
    # 不同种子单元格结构不一致时不能强行按任务平均。
    broken = [per_seed[0], per_seed[1],
              {("a", "1"): [1.0, 1.0], ("b", "2"): [1.0, 1.0]}]
    assert _ruler_macro_hierarchical_bootstrap(broken, n_boot=50) is None
    for bad in (0, -1, True, 2.5):
        try:
            _ruler_macro_hierarchical_bootstrap(per_seed, n_boot=bad)
            raised = False
        except ValueError:
            raised = True
        assert raised, bad
    print("  [ok] RULER 双层 bootstrap：确定性、结构校验、严格预算")


def test_ruler_macro_variance_components_uses_common_aligned_cells():
    from run_all import _ruler_macro_variance_components

    ids = {("a", "1"): ("s0", "s1", "s2", "s3"),
           ("a", "2"): ("t0", "t1", "t2", "t3")}
    diffs = [
        {("a", "1"): [1.0, 0.0, 1.0, 0.0],
         ("a", "2"): [1.0, 1.0, 0.0, 0.0]},
        {("a", "1"): [0.0, 1.0, 0.0, 1.0],
         ("a", "2"): [0.0, 1.0, 1.0, 0.0]},
        {("a", "1"): [1.0, 1.0, 0.0, 0.0],
         ("a", "2"): [1.0, 0.0, 1.0, 0.0]},
    ]
    comp = _ruler_macro_variance_components(diffs, [ids] * 3)
    assert comp is not None
    assert comp["n_cells"] == 2 and comp["n_cells_skipped"] == 0
    assert comp["n_seeds"] == 3 and comp["n_cells_common"] == 2
    # 样例 id 未对齐的单元格不能进入方差分解。
    bad_ids = [dict(ids), dict(ids), dict(ids)]
    bad_ids[2][("a", "1")] = ("s0", "s1", "s2", "x")
    bad = _ruler_macro_variance_components(diffs, bad_ids)
    assert bad is not None and bad["n_cells"] == 1
    assert bad["n_cells_skipped"] == 1
    print("  [ok] RULER 方差分量：仅用跨种子共有且 sample_id 对齐的单元格")


def test_ruler_sample_paired_tests_exposes_hierarchical_and_variance():
    from run_all import _ruler_sample_paired_tests

    def cell(ems, prefix):
        return {"em": sum(ems) / len(ems), "n": len(ems),
                "samples": [{"sample_id": f"{prefix}{i}", "em": float(e)}
                            for i, e in enumerate(ems)]}

    def rul(v, cells):
        return {"ruler": {
            "variant": v, "seed": 1, "ctx_lens": [256, 512],
            "n_samples": 8, "sample_seed": 7, "corpus_source": "val",
            "max_new": 12, "fwe_max_new": 44, "vt_chains": 4,
            "vt_depth": 4, "results": {
                "niah": {"256": cells[0], "512": cells[1]}}}}

    cand = {
        1: rul("lega", (cell([1, 1, 0, 0], "a"), cell([1, 0, 0, 0], "b"))),
        2: rul("lega", (cell([0, 1, 0, 0], "a"), cell([1, 1, 0, 0], "b"))),
        3: rul("lega", (cell([1, 0, 1, 0], "a"), cell([0, 0, 0, 1], "b"))),
    }
    ref = {s: rul("standard", (cell([0] * 4, "a"), cell([0] * 4, "b")))
           for s in (1, 2, 3)}
    t = _ruler_sample_paired_tests(
        {"lega": cand, "standard": ref}, [("lega", "standard")])[0]
    assert t["macro_mean_diff_ci95_hier"] is not None, t
    assert t["macro_p_hier"] is not None
    assert t["macro_p_hier_adj"] is not None
    assert t["macro_hier_verdict"]
    assert t["macro_ci95_hier_method"]
    comp = t["variance_components"]
    assert comp is not None and comp["n_seeds"] == 3
    assert comp["n_cells"] == 2 and comp["n_cells_skipped"] == 0
    assert comp["share_mean_seed"] is not None
    print("  [ok] RULER 样例检验：双层 CI 与方差分量进入结果")


def test_ppl_window_paired_tests_exposes_variance_components():
    from run_all import _ppl_window_paired_tests

    def ev(losses, idx):
        return {"eval": {"split": "test", "eval_ctx": 10,
                         "eval_iters": len(idx), "val_windows": len(idx),
                         "val_tokens_evaluated": 10 * len(idx),
                         "window_losses": losses, "window_indices": idx}}

    idx = [0, 3, 6, 9, 12]
    runs = {
        "lega": {
            7: ev([1.0, 1.2, 0.9, 1.1, 1.0], idx),
            8: ev([1.1, 1.0, 1.2, 0.8, 1.0], idx),
            9: ev([0.9, 1.1, 1.0, 1.2, 1.0], idx),
        },
        "standard": {s: ev([2.0] * 5, idx) for s in (7, 8, 9)},
    }
    t = _ppl_window_paired_tests(runs, [("lega", "standard")],
                                 n_resamples=100)[0]
    assert t["variance_components_aligned"] is True
    comp = t["variance_components"]
    assert comp is not None and comp["n_seeds"] == 3
    assert comp["n_units"] == 5
    assert comp["share_mean_unit"] is not None
    # 不同种子窗口序号不一致时不做方差分解（避免按位置伪配对）。
    alt_idx = [0, 3, 6, 9, 13]
    mis = {
        "lega": {7: ev([1.0] * 5, idx),
                 8: ev([1.0] * 5, alt_idx),
                 9: ev([1.0] * 5, idx)},
        "standard": {7: ev([2.0] * 5, idx),
                     8: ev([2.0] * 5, alt_idx),
                     9: ev([2.0] * 5, idx)},
    }
    tm = _ppl_window_paired_tests(mis, [("lega", "standard")],
                                  n_resamples=100)[0]
    assert tm["variance_components_aligned"] is False
    assert tm["variance_components"] is None
    print("  [ok] PPL 窗口检验：方差分量进入结果且要求窗口序号对齐")


def test_build_summary_renders_uncertainty_decomposition():
    from run_all import build_summary

    tmp = tempfile.mkdtemp()
    try:
        def ev(v, seed, losses, ppl):
            return {"variant": v, "seed": seed, "val_ppl": ppl,
                    "val_loss": 1.0, "split": "val", "eval_ctx": 10,
                    "eval_iters": 5, "eval_iters_requested": 5,
                    "val_windows": 5, "val_tokens_evaluated": 50,
                    "n_params": 100, "window_losses": losses,
                    "window_indices": [0, 3, 6, 9, 12]}

        def rul(v, seed, cells):
            return {"variant": v, "seed": seed, "ctx_lens": [256, 512],
                    "n_samples": 8, "sample_seed": 7, "corpus_source": "val",
                    "max_new": 12, "fwe_max_new": 44, "vt_chains": 4,
                    "vt_depth": 4, "results": {"niah": cells}}

        def cell(ems, prefix):
            return {"em": sum(ems) / len(ems), "n": len(ems),
                    "samples": [{"sample_id": f"{prefix}{i}", "em": float(e)}
                                for i, e in enumerate(ems)]}

        for s in (1, 2, 3):
            _write_json(os.path.join(tmp, f"eval_standard_seed{s}.json"),
                        ev("standard", s, [2.0] * 5, 10.0))
            _write_json(os.path.join(tmp, f"eval_lega_seed{s}.json"),
                        ev("lega", s, [1.0 + 0.1 * s] * 5, 9.0))
            ref_cells = {"256": cell([0] * 4, "a"),
                         "512": cell([0] * 4, "b")}
            cand_cells = {
                "256": cell([1, 1, 0, 0] if s == 1 else
                            ([0, 1, 0, 0] if s == 2 else [1, 0, 1, 0]), "a"),
                "512": cell([1, 0, 0, 0] if s == 1 else
                            ([1, 1, 0, 0] if s == 2 else [0, 0, 0, 1]), "b"),
            }
            _write_json(os.path.join(tmp, f"ruler_standard_seed{s}.json"),
                        rul("standard", s, ref_cells))
            _write_json(os.path.join(tmp, f"ruler_lega_seed{s}.json"),
                        rul("lega", s, cand_cells))
        out = build_summary(tmp)
        md = out["markdown"]
        assert "### 不确定性来源分解" in md, md[-5000:]
        assert "均值方差:种子" in md and "优先补" in md, md[-5000:]
        assert len(out["ruler_sample_tests"]) == 1
        assert len(out["ppl_window_tests"]) == 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] build_summary：不确定性来源表进入 Markdown")
