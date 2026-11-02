"""多方案排名稳定性与选择校正检验。"""
import json

from lega.ranking import (analyze_ranking, bootstrap_rank_stability,
                          permutation_selection_test, ranking_table)


def _scores(values, lower_is_better=False):
    """values: {variant: [per-seed values]} -> 统一种子 0..n-1 的映射。"""
    return {variant: {i: value for i, value in enumerate(values[variant])}
            for variant in values}


def test_ranking_table_orders_and_counts():
    scores = _scores({
        "A": [10.0, 11.0, 12.0, 13.0],
        "B": [8.0, 8.5, 9.0, 9.5],
        "C": [5.0, 5.5, 6.0, 6.5],
    })
    out = ranking_table(scores)
    assert out["available"] and out["status"] == "ok"
    assert out["leader"] == "A" and out["leaders"] == ["A"]
    rows = {row["variant"]: row for row in out["rows"]}
    assert rows["A"]["mean_rank"] == 1.0
    assert rows["B"]["mean_rank"] == 2.0
    assert rows["C"]["mean_rank"] == 3.0
    assert rows["A"]["best_seed_count"] == 4
    assert rows["C"]["worst_seed_count"] == 4
    assert out["rank_matrix"][0]["order"] == ["A", "B", "C"]
    print("  [ok] 稳定序：平均名次/最优次数/逐种子顺序")


def test_ranking_table_lower_is_better_and_ties():
    # PPL 越低越好：A 最低 -> 名次 1；第二种子 B 与 A 并列 -> 平均名次 1.5。
    scores = _scores({
        "A": [1.0, 2.0, 1.0],
        "B": [2.0, 2.0, 3.0],
        "C": [3.0, 4.0, 4.0],
    })
    out = ranking_table(scores, lower_is_better=True)
    rows = {row["variant"]: row for row in out["rows"]}
    assert abs(rows["A"]["mean_rank"] - 7.0 / 6.0) < 1e-12
    assert abs(rows["B"]["mean_rank"] - 11.0 / 6.0) < 1e-12
    assert rows["C"]["mean_rank"] == 3.0
    # 并列最优：种子 0 和 2 的 A 单独最优，种子 1 A/B 并列最优。
    assert rows["A"]["best_seed_count"] == 2
    assert rows["B"]["best_seed_count"] == 0
    assert out["rank_matrix"][1]["ranks"]["B"] == 1.5
    print("  [ok] lower_is_better 定向正确，并列名次取平均值")


def test_ranking_drops_unaligned_and_invalid_observations():
    scores = {
        "A": {0: 1.0, 1: 2.0, 2: float("nan"), 3: 4.0},
        "B": {0: 2.0, 1: 3.0, 3: 5.0, 4: 99.0},
        "C": {0: 3.0, 1: 4.0, 3: 6.0, 2: True},
    }
    out = ranking_table(scores)
    # 只有种子 0/1/3 在三个方案中都有有限值；2 因 NaN/bool 被丢弃，4 只在 B。
    assert out["seeds"] == [0, 1, 3]
    # 种子 2 的 A 为 NaN、C 为 bool，B 又缺失：该种子没有任何可用观测，
    # 不进入并集；种子 4 只有 B，属于“未对齐”而显式丢弃。
    assert set(out["dropped_seeds"]) == {4}
    assert out["n_seeds"] == 3 and out["available"]
    assert [row["variant"] for row in out["rows"]] == ["C", "B", "A"]
    print("  [ok] 只保留共同、有限的种子，坏值/未对齐种子显式记账")


def test_insufficient_seeds_do_not_claim_significance():
    scores = _scores({"A": [3.0, 3.0], "B": [1.0, 1.0]})
    table = ranking_table(scores)
    assert not table["available"]
    assert table["status"] == "insufficient_seeds"
    boot = bootstrap_rank_stability(scores, n_bootstrap=50)
    assert not boot["available"] and boot["rows"] == []
    perm = permutation_selection_test(scores, n_permutations=100)
    assert not perm["available"]
    assert perm["p_max"] is None and perm["significant_max"] is None
    out = analyze_ranking(scores, n_bootstrap=50, n_permutations=100)
    assert not out["available"]
    assert out["bootstrap"] is None and out["selection_test"] is None
    assert "种子不足" in out["reason"]
    print("  [ok] n<3 只报不可用，不伪造 p 值/显著性")


def test_bootstrap_is_deterministic_and_probabilities_normalized():
    scores = _scores({
        "A": [10.0, 10.2, 9.8, 10.1, 9.9],
        "B": [9.0, 9.1, 8.9, 9.2, 8.8],
        "C": [8.0, 7.9, 8.1, 8.0, 8.2],
    })
    first = bootstrap_rank_stability(scores, n_bootstrap=500, seed=7)
    second = bootstrap_rank_stability(scores, n_bootstrap=500, seed=7)
    assert first == second
    assert first["available"]
    total = sum(row["top_probability"] for row in first["rows"])
    assert abs(total - 1.0) < 1e-9
    rows = {row["variant"]: row for row in first["rows"]}
    assert rows["A"]["top_probability"] > 0.9
    assert rows["A"]["mean_rank_ci95"][0] == 1.0
    print("  [ok] bootstrap 固定种子可复现，top 概率归一，主导方案稳定")


def test_permutation_selection_test_detects_clear_separation():
    scores = _scores({
        "A": [30.0, 31.0, 29.0, 30.5, 30.2, 30.1, 30.3],
        "B": [20.0, 21.0, 19.0, 20.5, 20.2, 20.1, 20.3],
        "C": [10.0, 11.0, 9.0, 10.5, 10.2, 10.1, 10.3],
    })
    out = permutation_selection_test(scores, n_permutations=2000, seed=3)
    assert out["available"]
    assert out["p_max"] is not None and out["p_max"] < 0.01
    assert out["significant_max"] is True
    assert out["winner"] == "A" and out["runner_up"] == "B"
    assert out["observed_max_centered"] > out["expected_null_max_centered"]
    print("  [ok] 明确分离时选择校正置换检验给出小 p")


def test_permutation_null_does_not_invent_effect():
    # 三个方案在每个种子上完全相同：没有可检测的方案差异。
    scores = _scores({
        "A": [1.0, 2.0, 3.0],
        "B": [1.0, 2.0, 3.0],
        "C": [1.0, 2.0, 3.0],
    })
    out = permutation_selection_test(scores, n_permutations=500, seed=1)
    assert out["available"]
    assert out["p_max"] == 1.0 and out["p_range"] == 1.0
    assert out["significant_max"] is False
    assert out["selection_inflation"] == 0.0
    table = ranking_table(scores)
    assert table["leaders"] == ["A", "B", "C"]
    assert table["leader"] is None and table["leader_unique"] is False
    print("  [ok] 完全相同的方案：p=1、并列领先，不制造效应")


def test_selection_test_controls_seed_level_selection():
    # 每个种子内名次结构一致（A>B>C），但种子间整体水平漂移。
    # 普通“挑最好”会乐观；种子内标签置换仍能识别稳定的方案差异。
    scores = _scores({
        "A": [11.0, 21.0, 31.0, 41.0, 51.0, 61.0],
        "B": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        "C": [9.0, 19.0, 29.0, 39.0, 49.0, 59.0],
    })
    out = permutation_selection_test(scores, n_permutations=1000, seed=0)
    assert out["available"] and out["winner"] == "A"
    # 名次完全一致时，任何种子内置换都改变不了 A 的领先。
    assert out["p_max"] < 0.01
    print("  [ok] 保留种子效应（水平漂移）后仍识别出一致的名次差异")


def test_analyze_ranking_is_json_safe_and_handles_bad_input():
    scores = _scores({"A": [3.0, 3.1, 3.2], "B": [2.0, 2.1, 2.2]})
    out = analyze_ranking(scores, n_bootstrap=200, n_permutations=200,
                          seed=2)
    assert out["available"]
    json.dumps(out, allow_nan=False)
    bad = analyze_ranking([("not", "a", "mapping")])
    assert bad["status"] == "invalid" and not bad["available"]
    for func, kwargs in ((ranking_table, {"min_seeds": 1}),
                         (bootstrap_rank_stability, {"n_bootstrap": 0}),
                         (permutation_selection_test, {"alpha": 1.0})):
        try:
            func({"A": {0: 1.0}}, **kwargs)
        except ValueError:
            continue
        raise AssertionError(f"{func.__name__} 应拒绝非法参数 {kwargs!r}")
    print("  [ok] 结果 JSON 可序列化；坏输入/非法参数显式降级或报错")


def test_pipeline_ranking_analysis_and_render():
    """流水线层：从 runs 提取 PPL/RULER，渲染不崩溃且无可避免的缺失。"""
    from lega.pipeline.eval.comparisons import (_ranking_analysis,
                                                _render_ranking_section)
    runs = {}
    for variant, base in (("standard", 10.0), ("qgate", 9.5),
                          ("lega", 9.0)):
        runs[variant] = {}
        for seed in (1, 2, 3):
            runs[variant][seed] = {
                "eval": {"ppl": base + 0.1 * seed},
                "ruler": {"mean_em": 0.5 + 0.01 * seed},
            }
    out = _ranking_analysis(runs)
    assert out["metrics"]["PPL"]["available"]
    assert out["metrics"]["PPL"]["table"]["leader"] == "lega"
    md = "\n".join(_render_ranking_section(out))
    assert "多方案排名与选择校正" in md
    assert "p_max" in md
    assert "lega" in md
    empty = _ranking_analysis({})
    assert not empty["metrics"]["PPL"]["available"]
    assert "少于两个" in empty["metrics"]["PPL"]["reason"]
    print("  [ok] 流水线排名层从 runs 提取指标、渲染，并对空输入降级")
