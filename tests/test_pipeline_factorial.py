"""全因子/单因素消融、预注册与协议核对测试。"""
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from unittest import mock

from lega.data import CACHE_SCHEMA
from tests.pipeline_helpers import _mk, _write_json
from run_all import (ALPHA, AP_DIFF_METRIC, MIN_SEEDS, _ablation_coverage,
                      _ap_diff_tests, _apply_holm, _auroc_diff_tests, _cmd_env, _effect_status,
                      _enrich_paired, _eval_matches_window_spec,
                      _factorial_interaction_tests, _FACTORIAL_CONTRAST_CELLS,
                      _factorial_protocol_ok, _inference_shapley_tests,
                      INFERENCE_INTERACTION_METRIC, INFERENCE_SHAPLEY_METRIC,
                      PARAM_CONTROL_VARIANT,
                      PARAM_CONTROL_PAIR,
                      _int_or_zero, _lat_noise_pct,
                      _neural_eval_window_spec, _noise_floor_pct,
                      _noise_verdict_note, _loo_stability_str, _num_or_none,
                      _rate_or_none,
                      _seed_winloss_str, _paired_comparisons, _ppl_ci_rel_pct,
                      _log_ppl_of, _ppl_of, _probe_oof, _rel_pct,
                      _render_baseline_sections, _result_files,
                      _render_ruler_length,
                      _render_inference_shapley,
                      _render_parameter_matched_control,
                      _ruler_avg_micro, _ruler_avg_of, _ruler_cell_signature,
                      _ruler_depth_gap, _ruler_depth_means,
                      _ruler_fwe_recall, _ruler_min_cell_n, _ruler_protocol,
                      _ruler_task_means,
                      _render_registration_integrity, _resolve_jobs, _stat,
                      _seed_key, _strict_seed_int, _parse_seed_list,
                      apply_budget_gate, apply_integrity_gate, apply_scope_gate,
                      AUROC_DIFF_METRIC, AUROC_DIFF_REFERENCE,
                      build_comparison_pairs, build_summary,
                      _fmt_eq_margin, _fmt_eq_value,
                      _fmt_mean_std, _fmt_signed,
                      changed_gate_factors,
                      component_evidence, cross_variant_badcase_map,
                      COMPONENT_TESTS, GATE_FACTORS,
                      main as run_all_main,
                      data_protocol_preflight, data_report_binding_issue,
                      evaluate_registered_hypotheses,
                      load_registered_hypotheses, ngram_baseline_comparison,
                      parse_variants, registration_scope_check,
                      abort_ledger_on_failure, record_aborted_sanity,
                      mark_repro_failed, check_sanity_results,
                      _snapshot_existing_runs,
                      resolve_ngram_split, run_many,
                      single_factor_audit, single_factor_change,
                      SINGLE_FACTOR_CONTRASTS, SINGLE_FACTOR_PAIRS,
                      tradeoff_verdict, verify_controlled_setup)



def test_registered_hypothesis_rejects_missing_or_invalid_n():
    """n 缺失/非法时不能仅凭 p_adj 显著就判“支持”。"""
    tests = [
        {"candidate": "lega", "reference": "standard", "metric": "PPL",
         "mean_diff": -0.5, "lower_is_better": True, "p": 0.01,
         "p_adj": 0.01},
        {"candidate": "lega2", "reference": "standard", "metric": "PPL",
         "mean_diff": -0.5, "lower_is_better": True, "p": 0.01,
         "p_adj": 0.01, "n": "bad"},
        {"candidate": "lega3", "reference": "standard", "metric": "PPL",
         "mean_diff": -0.5, "lower_is_better": True, "p": 0.01,
         "p_adj": 0.01, "n": 2},
    ]
    reg = {"hypotheses": [
        {"id": "H1", "metric": "PPL", "candidate": "lega",
         "reference": "standard", "direction": "candidate_better"},
        {"id": "H2", "metric": "PPL", "candidate": "lega2",
         "reference": "standard", "direction": "candidate_better"},
        {"id": "H3", "metric": "PPL", "candidate": "lega3",
         "reference": "standard", "direction": "candidate_better"},
    ]}
    ev = {h["id"]: h for h in evaluate_registered_hypotheses(reg, tests)}
    assert ev["H1"]["status"] == "未定" and "样本量" in ev["H1"]["detail"]
    assert ev["H2"]["status"] == "未定" and "样本量" in ev["H2"]["detail"]
    assert ev["H3"]["status"] == "未定" and "种子不足" in ev["H3"]["detail"]
    print("  [ok] 预注册判定拒绝缺 n/非法 n 的“显著”证据")


def test_registration_paths_tolerate_malformed_hypotheses_and_status():
    """登记表 hypotheses 非列表/条目非对象、完整性 status 非字符串均不崩。"""
    assert evaluate_registered_hypotheses({"hypotheses": 5}, [1, None]) == []
    out = evaluate_registered_hypotheses(
        {"hypotheses": [1, {"id": "H1", "metric": "PPL",
                            "candidate": "lega", "reference": "standard",
                            "direction": "candidate_better"}]}, [1])
    assert len(out) == 1 and out[0]["status"] == "未检验"
    assert apply_integrity_gate({"x": 1}, {"confirmatory_valid": True}) == {"x": 1}
    hyp = [{"confirmatory_valid": True}]
    apply_integrity_gate(hyp, {"confirmatory_valid": True})
    # 损坏的“有效性”字段不能经 truthiness 授予确认性资格。
    for bogus in ("false", 1, "yes", [1], {"ok": True}, None):
        hp = [{"confirmatory_valid": True}]
        apply_integrity_gate(hp, {"confirmatory_valid": bogus})
        assert hp[0]["confirmatory_valid"] is False, bogus
    apply_scope_gate(hyp, {"scope": {"config": "x"}, "checked": False,
                           "problems": [1]})
    assert hyp[0]["confirmatory_valid"] is False
    # checked 字段损坏时不能经 truthiness 当成“已核对在范围内”。
    for bogus in ("false", 1, "yes", [1], {"ok": True}, None):
        hp = [{"confirmatory_valid": True}]
        apply_scope_gate(hp, {"scope": {"config": "x"},
                              "checked": bogus, "problems": []})
        assert hp[0]["confirmatory_valid"] is False, bogus
    md = _render_registration_integrity(
        {"status": [1, 2], "problems": [1], "warnings": 5},
        {"scope": 5, "checked": False, "problems": "x"})
    assert any("预注册完整性" in line for line in md)
    assert cross_variant_badcase_map({"lega": {1: {"ruler": 5}}},
                                     ["lega"]) == {}
    assert cross_variant_badcase_map({"lega": [1]}, ["lega"]) == {}
    print("  [ok] 预注册路径：畸形 hypotheses/status/slot 安全降级")


def test_factorial_none_cell_is_within_gate_control():
    """2^3 的 none 格必须是 gate_only：standard 无门会混入额外因子。"""
    from run_all import CONTROL_CELLS, FACTORIAL_CELLS, LOO_CELLS
    cells = dict(FACTORIAL_CELLS)
    assert cells["无 A/B/C（有门）"] == "gate_only", cells
    variants = [v for _, v in FACTORIAL_CELLS]
    assert "standard" not in variants, variants
    assert len(set(variants)) == 8 and set(variants) == {
        "gate_only", "qgate", "lega_conc_only", "lega_len_only",
        "lega_no_len", "lega_no_conc", "lega_no_query", "lega"}, variants
    assert [v for _, v in CONTROL_CELLS] == ["standard"]
    assert [v for _, v in LOO_CELLS] == [
        "lega", "lega_no_query", "lega_no_conc", "lega_no_len"]
    print("  [ok] 全因子 none=gate_only，standard 仅作范围外无门基线")


def test_ablation_coverage_requires_min_seeds_per_cell():
    """消融矩阵“覆盖完整”必须每格≥MIN_SEEDS，单种子冒烟格不能算覆盖。"""
    cells = (("A", "a"), ("B", "b"), ("C", "c"))
    rows = [{"variant": "a", "n_seeds": 3},
            {"variant": "b", "n_seeds": 1}]
    done, missing, under = _ablation_coverage(cells, {"a", "b"}, 
                                               {r["variant"]: r for r in rows})
    assert done == [("A", "a")], done
    assert missing == [("B", "b"), ("C", "c")], missing
    assert under == [("B", "b", 1)], under
    # 无 n_seeds 的旧 row 不得默认成“种子充足”
    done, missing, under = _ablation_coverage(
        cells, {"a"}, {"a": {"variant": "a"}})
    assert not done and under == [("A", "a", 0)], (done, missing, under)
    # bool/小数/负数不是有效种子数，不能截断成“覆盖完整”
    for bad in (True, 3.9, -3, "bad"):
        done, missing, under = _ablation_coverage(
            cells, {"a"}, {"a": {"variant": "a", "n_seeds": bad}})
        assert not done and under == [("A", "a", 0)], (bad, done, missing, under)
    done, _, _ = _ablation_coverage(
        cells, {"a"}, {"a": {"variant": "a", "n_seeds": "3"}})
    assert done == [("A", "a")], done
    print("  [ok] 消融矩阵覆盖按每格种子数判定，单种子格明确标为不足")


def test_single_factor_audit_and_pair_construction():
    """单因素消融必须只改一个 A/B/C 因子；B-only/C-only 不能与 Full 配对。"""
    audit = single_factor_audit()
    assert audit["ok"] and not audit["problems"], audit
    assert len(audit["contrasts"]) == 6
    assert {c["factor"] for c in audit["contrasts"]} == {"A", "B", "C"}
    assert {c["direction"] for c in audit["contrasts"]} == {"remove", "add"}
    for c in audit["contrasts"]:
        assert c["changed_factors"] == [c["factor"]], c
    assert len(SINGLE_FACTOR_PAIRS) == 6
    assert len(set(SINGLE_FACTOR_PAIRS)) == 6

    # 去一法：Full−X vs Full 只改变 X
    assert single_factor_change("lega_no_query", "lega") == "A"
    assert single_factor_change("lega_no_conc", "lega") == "B"
    assert single_factor_change("lega_no_len", "lega") == "C"
    # 全因子增量：X-only vs gate_only 只加入 X
    assert single_factor_change("qgate", "gate_only") == "A"
    assert single_factor_change("lega_conc_only", "gate_only") == "B"
    assert single_factor_change("lega_len_only", "gate_only") == "C"
    # B-only/C-only vs Full 同时改变两个因子，不能进单因素对比表
    assert changed_gate_factors("lega_conc_only", "lega") == {"A", "C"}
    assert changed_gate_factors("lega_len_only", "lega") == {"A", "B"}
    assert single_factor_change("lega_conc_only", "lega") is None
    assert single_factor_change("lega_len_only", "lega") is None
    # vs standard / LEGA vs qgate 是整体方法对比，不是门控内单因素
    assert single_factor_change("lega", "standard") is None
    assert single_factor_change("lega", "qgate") is None

    # 组件有效性判定实际消费的每个检验对都必须是已审计的单因素对比
    expected_factor = {"A 查询项": "A", "B 集中度证据": "B",
                       "C 长度校准": "C"}
    assert {name for name, *_ in COMPONENT_TESTS} == set(expected_factor)
    for name, loo_pair, add_pair, _primary in COMPONENT_TESTS:
        factor = expected_factor[name]
        assert loo_pair in SINGLE_FACTOR_PAIRS, (name, loo_pair)
        assert add_pair in SINGLE_FACTOR_PAIRS, (name, add_pair)
        assert single_factor_change(*loo_pair) == factor, (name, loo_pair)
        assert single_factor_change(*add_pair) == factor, (name, add_pair)

    runs = {"standard": {1: {}}, "gate_only": {1: {}},
            "qgate": {1: {}}, "lega_conc_only": {1: {}},
            "lega_len_only": {1: {}}, "lega": {1: {}},
            "lega_no_len": {1: {}}, "lega_no_conc": {1: {}},
            "lega_no_query": {1: {}}}
    pairs = build_comparison_pairs(runs, {"hypotheses": []})
    for c, r in SINGLE_FACTOR_PAIRS:
        assert (c, r) in pairs, (c, r)
    assert ("lega_conc_only", "lega") not in pairs
    assert ("lega_len_only", "lega") not in pairs
    # 非 vs standard 的对比要么是单因素消融，要么是显式登记的整体假设
    for c, r in pairs:
        if r != "standard":
            assert (c, r) in SINGLE_FACTOR_PAIRS, (c, r)
    reg = {"hypotheses": [{"metric": "PPL", "candidate": "lega",
                           "reference": "qgate"}]}
    pairs = build_comparison_pairs(runs, reg)
    assert ("lega", "qgate") in pairs
    assert single_factor_change("lega", "qgate") is None
    print("  [ok] 单因素审计：6 组对比只改一个因子，多因子对比不混入")


def test_single_factor_audit_rejects_alt_conc_metadata_drift():
    """替代集中度变体必须保持 B-only，且不能混入主 2^3 因子表。"""
    from lega.attention import (ALT_CONC_STATS, ALT_CONC_VARIANTS,
                                RAW_GATE_FACTORS, _USE_CONC)
    audit = single_factor_audit()
    assert audit["ok"] and not audit["problems"], audit
    rows = audit["alt_conc_contrasts"]
    assert {r["variant"] for r in rows} == set(ALT_CONC_VARIANTS)
    for row in rows:
        assert row["mode"] == ALT_CONC_STATS[row["variant"]]
        assert row["replaced_factor"] == "B" and row["reference"] == "lega"
        assert row["variant"] not in GATE_FACTORS
        assert row["variant"] not in RAW_GATE_FACTORS

    with mock.patch("lega.pipeline.common.ALT_CONC_STATS",
                    {"lega_l2": "l2"}):
        audit = single_factor_audit()
        assert not audit["ok"], audit
        assert any("ALT_CONC_STATS" in p for p in audit["problems"]), audit
    with mock.patch("lega.pipeline.common.ALT_CONC_STATS",
                    dict(ALT_CONC_STATS, lega_entropy="max")):
        audit = single_factor_audit()
        assert not audit["ok"], audit
        assert any("未实现" in p for p in audit["problems"]), audit
    bad_factors = dict(GATE_FACTORS)
    bad_factors["lega_l2"] = frozenset({"A", "B", "C"})
    with mock.patch("lega.pipeline.common.GATE_FACTORS", bad_factors):
        audit = single_factor_audit()
        assert not audit["ok"], audit
        assert any("混入了主" in p for p in audit["problems"]), audit
    with mock.patch("lega.pipeline.common._USE_CONC",
                    tuple(v for v in _USE_CONC if v != "lega_entropy")):
        audit = single_factor_audit()
        assert not audit["ok"], audit
        assert any("未保持 lega 的 A/B/C" in p
                   for p in audit["problems"]), audit
    print("  [ok] 替代集中度审计：B-only 元数据与主因子表隔离")


def test_single_factor_audit_requires_complete_and_audited_contrasts():
    """审计表少一条/重复一条、或组件检验绕过审计表，都必须显式失败。"""
    with mock.patch("lega.pipeline.common.SINGLE_FACTOR_CONTRASTS",
                    SINGLE_FACTOR_CONTRASTS[:-1]):
        audit = single_factor_audit()
        assert not audit["ok"], audit
        assert any("缺少单因素对比" in p for p in audit["problems"]), audit
    with mock.patch("lega.pipeline.common.SINGLE_FACTOR_CONTRASTS",
                    SINGLE_FACTOR_CONTRASTS + (SINGLE_FACTOR_CONTRASTS[0],)):
        audit = single_factor_audit()
        assert not audit["ok"], audit
        assert any("重复的单因素对比" in p for p in audit["problems"]), audit
    bad_component = COMPONENT_TESTS + (
        ("D 额外组件", ("lega_conc_only", "lega"),
         ("lega_len_only", "lega"), "PPL"),)
    with mock.patch("lega.pipeline.common.COMPONENT_TESTS", bad_component):
        audit = single_factor_audit()
        assert not audit["ok"], audit
        assert any("未审计的单因素对比" in p
                   for p in audit["problems"]), audit
    print("  [ok] 单因素审计拒绝缺失/重复对比与绕过审计表的组件检验")


def test_single_factor_audit_rejects_broken_pair_table():
    """汇总实际消费的 SINGLE_FACTOR_PAIRS 也必须与声明边逐项一致。"""
    audit = single_factor_audit()
    assert audit["ok"] and len(audit["pairs"]) == 6, audit

    with mock.patch("lega.pipeline.common.SINGLE_FACTOR_PAIRS",
                    SINGLE_FACTOR_PAIRS[:-1]):
        audit = single_factor_audit()
        assert not audit["ok"], audit
        assert any("SINGLE_FACTOR_PAIRS 缺少已声明对比" in p
                   for p in audit["problems"]), audit

    extra = (("lega_conc_only", "lega"),
             ("lega_len_only", "lega"))
    with mock.patch("lega.pipeline.common.SINGLE_FACTOR_PAIRS",
                    SINGLE_FACTOR_PAIRS + extra):
        audit = single_factor_audit()
        assert not audit["ok"], audit
        assert any("实际不改变恰好一个" in p
                   for p in audit["problems"]), audit

    rest = tuple(p for p in SINGLE_FACTOR_PAIRS
                 if p != ("lega_no_query", "lega"))
    with mock.patch("lega.pipeline.common.SINGLE_FACTOR_PAIRS", rest):
        audit = single_factor_audit()
        assert not audit["ok"], audit
        assert any("不在 SINGLE_FACTOR_PAIRS 中" in p
                   for p in audit["problems"]), audit

    with mock.patch("lega.pipeline.common.SINGLE_FACTOR_PAIRS",
                    SINGLE_FACTOR_PAIRS + (SINGLE_FACTOR_PAIRS[0],)):
        audit = single_factor_audit()
        assert not audit["ok"], audit
        assert any("重复的单因素对比对" in p
                   for p in audit["problems"]), audit
    print("  [ok] 单因素配对表拒绝漏项/多因子/组件缺项/重复项")


def test_single_factor_audit_rejects_broken_design_metadata():
    """2^3 设计元数据、方向与组件检验本身损坏时，审计也必须显式失败。"""
    # 8 个变体名都在，但两个变体拥有相同因子集合：某个因子格实际不存在。
    bad_factors = dict(GATE_FACTORS)
    bad_factors["qgate"] = frozenset({"A", "B"})
    with mock.patch("lega.pipeline.common.GATE_FACTORS", bad_factors):
        audit = single_factor_audit()
        assert not audit["ok"], audit
        assert any("完整 2^3" in p for p in audit["problems"]), audit

    # 把“去掉 A”写成“加入 A”：方向是实验定义的一部分，不能靠展示字段糊弄。
    swapped = (("A", "add", "lega_no_query", "lega"),) + tuple(
        c for c in SINGLE_FACTOR_CONTRASTS if c != SINGLE_FACTOR_CONTRASTS[0])
    with mock.patch("lega.pipeline.common.SINGLE_FACTOR_CONTRASTS", swapped):
        audit = single_factor_audit()
        assert not audit["ok"], audit
        assert any("方向" in p for p in audit["problems"]), audit

    # 组件检验缺一项/重复一项/名称无法映射/主指标非法，都不能静默通过。
    with mock.patch("lega.pipeline.common.COMPONENT_TESTS", COMPONENT_TESTS[:2]):
        audit = single_factor_audit()
        assert not audit["ok"], audit
        assert any("缺少组件检验" in p for p in audit["problems"]), audit
    with mock.patch("lega.pipeline.common.COMPONENT_TESTS", COMPONENT_TESTS + (COMPONENT_TESTS[0],)):
        audit = single_factor_audit()
        assert not audit["ok"], audit
        assert any("重复的检验定义" in p for p in audit["problems"]), audit
    bad_name = COMPONENT_TESTS + (
        ("D 额外组件", ("lega_no_query", "lega"),
         ("qgate", "gate_only"), "PPL"),)
    with mock.patch("lega.pipeline.common.COMPONENT_TESTS", bad_name):
        audit = single_factor_audit()
        assert not audit["ok"], audit
        assert any("无法映射到 A/B/C" in p for p in audit["problems"]), audit
    bad_metric = (COMPONENT_TESTS[0][:3] + ("loss",),) + COMPONENT_TESTS[1:]
    with mock.patch("lega.pipeline.common.COMPONENT_TESTS", bad_metric):
        audit = single_factor_audit()
        assert not audit["ok"], audit
        assert any("不是已登记指标" in p for p in audit["problems"]), audit
    # 组件把去一法/全因子两套对比对调：因子虽然都是 A，方向仍然错误。
    swapped_component = (("A 查询项", ("qgate", "gate_only"),
                          ("lega_no_query", "lega"), "PPL"),) + COMPONENT_TESTS[1:]
    with mock.patch("lega.pipeline.common.COMPONENT_TESTS", swapped_component):
        audit = single_factor_audit()
        assert not audit["ok"], audit
        assert any("方向为 add" in p or "方向为 remove" in p
                   for p in audit["problems"]), audit

    # 外部损坏键不能让“是否单因素”的查询崩溃，只能返回 None/不可归因。
    assert changed_gate_factors([], "lega") is None
    assert changed_gate_factors("lega", {"bad": "key"}) is None
    unknown_factor = dict(GATE_FACTORS, mystery=frozenset({"D"}))
    with mock.patch("lega.pipeline.common.GATE_FACTORS", unknown_factor):
        assert changed_gate_factors("mystery", "gate_only") is None
        assert single_factor_change("mystery", "gate_only") is None
    print("  [ok] 单因素审计拒绝不完整 2^3、错误方向与损坏的组件检验")


def test_paired_comparisons_marks_contrast_type():
    """配对结果必须机器可读地标出 single_factor/overall，避免事后误归因。"""
    runs = {}
    for v, base in (("standard", 12.0), ("gate_only", 11.0),
                    ("qgate", 10.0), ("lega_conc_only", 9.0),
                    ("lega_len_only", 9.5), ("lega", 8.0),
                    ("lega_no_len", 8.5), ("lega_no_conc", 8.2),
                    ("lega_no_query", 8.8)):
        runs[v] = {s: {"eval": {"val_ppl": base + 0.01 * s, "split": "val"}}
                   for s in (1, 2, 3)}
    pairs = list(SINGLE_FACTOR_PAIRS) + [
        ("lega", "standard"), ("lega", "qgate"),
        ("lega_conc_only", "lega"),
        # 方向写反的单因子对没有登记在 SINGLE_FACTOR_PAIRS 中，不能
        # 仅凭“对称差=1”就获得可归因标记。
        ("lega", "lega_no_query")]
    out = _paired_comparisons(
        runs, pairs, "PPL", lambda slot: _ppl_of(slot.get("eval")),
        lower_better=True)
    by = {(t["candidate"], t["reference"]): t for t in out}
    for c, r in SINGLE_FACTOR_PAIRS:
        t = by[(c, r)]
        assert t["contrast_type"] == "single_factor", (c, r, t)
        assert len(t["changed_factors"]) == 1, t
    assert by[("lega", "standard")]["contrast_type"] == "overall"
    assert by[("lega", "qgate")]["contrast_type"] == "overall"
    assert by[("lega_conc_only", "lega")]["contrast_type"] == "overall"
    assert by[("lega", "lega_no_query")]["contrast_type"] == "overall"
    print("  [ok] 配对检验结果带 single_factor/overall 机器可读标记")


def test_build_summary_excludes_multifactor_ablation_pairs():
    """汇总不得再把 B-only/C-only 与 Full 的差异当成单组件消融。"""
    tmp = tempfile.mkdtemp()
    try:
        bases = {"standard": 11.0, "gate_only": 10.5, "qgate": 10.2,
                 "lega_conc_only": 10.0, "lega_len_only": 10.3,
                 "lega": 9.5, "lega_no_len": 9.7, "lega_no_conc": 9.6,
                 "lega_no_query": 9.9}
        for v, base in bases.items():
            for s in (1, 2, 3):
                _write_json(os.path.join(tmp, f"eval_{v}_seed{s}.json"), {
                    "variant": v, "seed": s, "split": "val",
                    "val_ppl": base + 0.01 * s, "val_loss": 2.3,
                    "n_params": 100, "n_gate_extra": 0})
        out = build_summary(tmp)
        sig = json.load(open(os.path.join(tmp, "significance.json"),
                             encoding="utf-8"))
        pairs = {(t["candidate"], t["reference"]) for t in sig["pairs"]}
        assert ("lega_conc_only", "lega") not in pairs
        assert ("lega_len_only", "lega") not in pairs
        for pair in SINGLE_FACTOR_PAIRS:
            assert pair in pairs, pair
        assert sig["single_factor_audit"]["ok"] is True
        assert "单因素消融审计" in out["markdown"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] 汇总排除 B-only/C-only vs Full 的多因子“消融”对比")


def test_build_summary_only_reuses_ablation_when_scope_is_true_bool():
    """repro.ablation.executed 必须是真正的 true 才复用旧病灶结果。

    "false"/1 经 truthiness 会把上一轮 ablation_*.json 冒充成本轮证据，
    让“本轮 --skip_ablation”的汇总仍带着推理期因果结论。
    """
    tmp = tempfile.mkdtemp()
    try:
        _write_json(os.path.join(tmp, "repro.json"), {
            "variants": ["lega"], "seeds": [1], "smoke": "false",
            "ablation": {"executed": "false", "variants": ["lega"]}})
        _write_json(os.path.join(tmp, "ablation_lega.json"), {
            "variant": "lega", "seed": 1,
            "lesions": [{"lesion": "no_query", "label": "去查询项",
                         "applicable": True, "applied": True,
                         "delta_loss_mean": 0.1, "n_windows": 10}]})
        build_summary(tmp)
        sig = json.load(open(os.path.join(tmp, "significance.json"),
                             encoding="utf-8"))
        assert sig["causal_ablation"] == [], sig["causal_ablation"]
        # 显式 executed=true 时才消费同一份文件（n=1 仍会如实报告）。
        _write_json(os.path.join(tmp, "repro.json"), {
            "variants": ["lega"], "seeds": [1], "smoke": False,
            "ablation": {"executed": True, "variants": ["lega"]}})
        build_summary(tmp)
        sig = json.load(open(os.path.join(tmp, "significance.json"),
                             encoding="utf-8"))
        assert len(sig["causal_ablation"]) == 1, sig["causal_ablation"]
        assert sig["causal_ablation"][0]["lesion"] == "no_query"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] 消融证据只在 executed 为真正 true 时复用，"
          "malformed repro 不会冒充本轮结论")


def test_verify_controlled_setup_rejects_eval_protocol_mismatch():
    """同一训练 config 但独立评测用了不同窗口/上下文，PPL 不可配对。"""
    with tempfile.TemporaryDirectory() as tmp:
        variants, seeds = ["standard", "lega"], [7]
        base = {"split": "test", "eval_ctx": 512, "eval_iters": 50,
                "eval_iters_requested": 50, "val_windows": 80,
                "val_tokens_evaluated": 25600,
                "selection_set_is_eval_set": False}
        for v in variants:
            d = os.path.join(tmp, v)
            os.makedirs(d)
            _write_json(os.path.join(d, "config_used.json"), {
                "lr": 0.001, "steps": 10, "variant": v, "seed": 7,
                "device": "cpu", "tag": "t"})
            _write_json(os.path.join(d, "eval.json"), {
                "variant": v, "seed": 7, "data_fingerprint": "abc123",
                "code_commit": "deadbeef", "config_sha256": "shax",
                "steps": 10, "tokens_seen": 1024, **base})
        audit = verify_controlled_setup(tmp, variants, seeds, 1,
                                        expected_data_digest="abc123")
        assert audit["ok"] and audit["eval_protocol_checked"] == 2, audit
        for key, bad in (("eval_ctx", 256), ("eval_iters", 25),
                         ("val_windows", 40), ("split", "val")):
            p = os.path.join(tmp, "lega", "eval.json")
            ev = json.load(open(p, encoding="utf-8"))
            ev[key] = bad
            _write_json(p, ev)
            try:
                verify_controlled_setup(tmp, variants, seeds, 1,
                                        expected_data_digest="abc123")
                raise AssertionError(f"{key} 不一致应触发 SystemExit")
            except SystemExit:
                pass
            ev[key] = base[key]
            _write_json(p, ev)
        # 旧产物完全缺评测协议字段：不伪造一致，但也保持兼容（显式记录缺失）
        for key in ("split", "eval_ctx", "eval_iters", "eval_iters_requested",
                    "val_windows", "val_tokens_evaluated",
                    "selection_set_is_eval_set"):
            ev.pop(key, None)
        _write_json(os.path.join(tmp, "lega", "eval.json"), ev)
        audit = verify_controlled_setup(tmp, variants, seeds, 1,
                                        expected_data_digest="abc123")
        assert audit["ok"], audit
        assert "lega/seed7" in audit["eval_protocol_missing"], audit
    print("  [ok] 受控变量审计：PPL 评测 split/ctx/iters/windows 不一致即拒绝")


def test_verify_controlled_setup_normalizes_numeric_strings():
    """旧 eval.json 把协议数量写成字符串时，语义相同的数值不应被判不一致。"""
    with tempfile.TemporaryDirectory() as tmp:
        variants, seeds = ["standard", "lega"], [7]
        base = {"split": "test", "eval_ctx": 512, "eval_iters": 50,
                "eval_iters_requested": 50, "val_windows": 80,
                "val_tokens_evaluated": 25600,
                "selection_set_is_eval_set": False}
        for v in variants:
            d = os.path.join(tmp, v)
            os.makedirs(d)
            _write_json(os.path.join(d, "config_used.json"), {
                "lr": 0.001, "steps": 10, "variant": v, "seed": 7,
                "device": "cpu", "tag": "t"})
            _write_json(os.path.join(d, "eval.json"), {
                "variant": v, "seed": 7, "data_fingerprint": "abc123",
                "code_commit": "deadbeef", "config_sha256": "shax",
                "steps": 10, "tokens_seen": 1024, **base})
        ev = json.load(open(os.path.join(tmp, "lega", "eval.json"),
                            encoding="utf-8"))
        for key in ("eval_ctx", "eval_iters", "eval_iters_requested",
                    "val_windows", "val_tokens_evaluated"):
            ev[key] = str(ev[key])
        _write_json(os.path.join(tmp, "lega", "eval.json"), ev)
        audit = verify_controlled_setup(tmp, variants, seeds, 1,
                                        expected_data_digest="abc123")
        assert audit["ok"] and audit["eval_protocol_checked"] == 2, audit
    print("  [ok] 受控变量审计：数值字符串与数值按相同语义比较")


def test_verify_controlled_setup_rejects_fractional_steps():
    """实际步数/评测窗口被小数污染时不能被 int() 截断后冒充一致。"""
    with tempfile.TemporaryDirectory() as tmp:
        variants, seeds = ["standard", "lega"], [7]
        base = {"split": "test", "eval_ctx": 512, "eval_iters": 50,
                "eval_iters_requested": 50, "val_windows": 80,
                "val_tokens_evaluated": 25600,
                "selection_set_is_eval_set": False}
        for v in variants:
            d = os.path.join(tmp, v)
            os.makedirs(d)
            _write_json(os.path.join(d, "config_used.json"), {
                "lr": 0.001, "steps": 10, "variant": v, "seed": 7,
                "device": "cpu", "tag": "t"})
            _write_json(os.path.join(d, "eval.json"), {
                "variant": v, "seed": 7, "data_fingerprint": "abc123",
                "code_commit": "deadbeef", "config_sha256": "shax",
                "steps": 10, "tokens_seen": 1024, **base})
        for v, actual in (("standard", 10), ("lega", 10.5)):
            p = os.path.join(tmp, v, "eval.json")
            ev = json.load(open(p, encoding="utf-8"))
            ev["steps_planned"] = 10.5
            ev["steps"] = actual
            _write_json(p, ev)
        try:
            verify_controlled_setup(tmp, variants, seeds, 1,
                                    expected_data_digest="abc123")
            raise AssertionError("小数实际步数不应被截断为一致")
        except SystemExit:
            pass
        # 双方都记录相同的小数步数同样不能算“同一预算”：步数必须可解析为整数
        for v in variants:
            p = os.path.join(tmp, v, "eval.json")
            ev = json.load(open(p, encoding="utf-8"))
            ev["steps_planned"] = 10.5
            ev["steps"] = 10.5
            _write_json(p, ev)
        try:
            verify_controlled_setup(tmp, variants, seeds, 1,
                                    expected_data_digest="abc123")
            raise AssertionError("小数步数不能被当作合法训练预算")
        except SystemExit:
            pass
        # JSON true 不是步数：bool 会被 float() 当成 1，必须显式拒绝
        for v in variants:
            p = os.path.join(tmp, v, "eval.json")
            ev = json.load(open(p, encoding="utf-8"))
            ev["steps_planned"] = True
            ev["steps"] = True
            ev["tokens_seen"] = True
            _write_json(p, ev)
        try:
            verify_controlled_setup(tmp, variants, seeds, 1,
                                    expected_data_digest="abc123")
            raise AssertionError("布尔步数/tokens_seen 不能被当作训练量")
        except SystemExit:
            pass
        # "1_0" 经 int()/float() 都会变成 10；下划线写法不是训练量。
        for v in variants:
            p = os.path.join(tmp, v, "eval.json")
            ev = json.load(open(p, encoding="utf-8"))
            ev["steps_planned"] = "1_0"
            ev["steps"] = "1_0"
            ev["tokens_seen"] = "1_0"
            _write_json(p, ev)
        try:
            verify_controlled_setup(tmp, variants, seeds, 1,
                                    expected_data_digest="abc123")
            raise AssertionError("下划线步数/tokens_seen 不能被当作训练量")
        except SystemExit:
            pass
    print("  [ok] 受控变量审计：小数/布尔/下划线步数不会被截断为整数")


def test_verify_controlled_setup_rejects_fractional_seeds():
    """config/eval 的小数种子不能被 int() 截断后冒充预期种子。"""
    with tempfile.TemporaryDirectory() as tmp:
        variants, seeds = ["standard", "lega"], [7]
        base = {"split": "test", "eval_ctx": 512, "eval_iters": 50,
                "eval_iters_requested": 50, "val_windows": 80,
                "val_tokens_evaluated": 25600,
                "selection_set_is_eval_set": False}
        for v in variants:
            d = os.path.join(tmp, v)
            os.makedirs(d)
            _write_json(os.path.join(d, "config_used.json"), {
                "lr": 0.001, "steps": 10, "variant": v, "seed": 7,
                "device": "cpu", "tag": "t"})
            _write_json(os.path.join(d, "eval.json"), {
                "variant": v, "seed": 7, "data_fingerprint": "abc123",
                "code_commit": "deadbeef", "config_sha256": "shax",
                "steps": 10, "tokens_seen": 1024, **base})
        cfg_path = os.path.join(tmp, "lega", "config_used.json")
        cfg = json.load(open(cfg_path, encoding="utf-8"))
        cfg["seed"] = 7.5
        _write_json(cfg_path, cfg)
        try:
            verify_controlled_setup(tmp, variants, seeds, 1,
                                    expected_data_digest="abc123")
            raise AssertionError("config 小数种子不应通过审计")
        except SystemExit:
            pass
        cfg["seed"] = 7
        _write_json(cfg_path, cfg)
        ev_path = os.path.join(tmp, "lega", "eval.json")
        ev = json.load(open(ev_path, encoding="utf-8"))
        ev["seed"] = 7.5
        _write_json(ev_path, ev)
        try:
            verify_controlled_setup(tmp, variants, seeds, 1,
                                    expected_data_digest="abc123")
            raise AssertionError("eval 小数种子不应通过审计")
        except SystemExit:
            pass
    print("  [ok] 受控变量审计：小数种子不会被截断为预期种子")


def test_component_evidence_marks_early_stop_budget():
    """早停预算不等时，JSON 证据必须显式降为探索性而非只说 supported。"""
    tests = [
        _mk("lega_no_query", "lega", "PPL", 0.4, 0.005),
        _mk("qgate", "gate_only", "PPL", -0.01, 0.7),
    ]
    _apply_holm(tests)
    audit = {"early_stop_runs": ["lega/seed7(step 6/10)"]}
    a = next(c for c in component_evidence(tests, audit=audit)
             if c["component"] == "A 查询项")
    assert a["supported"] == ["PPL"]                 # 统计显著性仍保留
    assert a["training_budget_comparable"] is False
    assert a["confirmatory_supported"] is False
    assert a["evidence_status"] == "exploratory"
    assert "早停" in a["budget_note"] and a["early_stop_runs"]
    ok = next(c for c in component_evidence(tests)
              if c["component"] == "A 查询项")
    assert ok["training_budget_comparable"] is True
    assert ok["confirmatory_supported"] is True
    assert ok["evidence_status"] == "confirmatory"
    print("  [ok] component_evidence：早停预算不等时机器可读降级为探索性")


def test_seed_parsing_rejects_bool_and_fractional_truncation():
    """--seeds/配置 seeds 不能被 int() 截断：True → 1、1.5 → 1 会改身份。

    种子身份错位后结果目录名、台账与播种的 RNG 相互矛盾；更糟的是
    ``dict.fromkeys`` 会把 1.5 与 1 去重，导致"N 个种子"的证据凭空缩水。
    bool 在 dict 里与 1 同键，同样不能被当作种子 1。
    """
    assert _strict_seed_int(7) == 7
    assert _strict_seed_int(7.0) == 7
    assert _strict_seed_int("1337") == 1337
    assert _strict_seed_int(" 42 ") == 42
    for bad in (True, False, 1.5, -0.5, -1, 2 ** 32, float("nan"),
                float("inf"), "1.5", "abc", "", [7], {"seed": 1}, None):
        try:
            _strict_seed_int(bad)
            raised = False
        except ValueError:
            raised = True
        assert raised, bad
    assert _parse_seed_list("1,2, 3") == [1, 2, 3]
    assert _parse_seed_list("") == []
    # 字符串 "2.0" 不是整数身份（与 tuning.parse_seeds 同口径拒绝）
    for bad in ("1,2.0", "1,1.5", "1,true", "1,foo"):
        try:
            _parse_seed_list(bad)
            raised = False
        except ValueError:
            raised = True
        assert raised, bad
    print("  [ok] run_all 种子解析拒绝 bool/小数截断（含 --tune_seeds 同口径）")


def test_controlled_setup_selection_frac_rejects_bad_planned_steps():
    """checkpoint 选型比例的分母必须是可解释为整数的计划步数。

    steps_planned=true 时 True/1==1 会让比例虚高；10.5 会让比例用错误预算
    做除法。两者都应记为 None（缺比例），而字符串 "10"/10.0 仍可用。
    """
    with tempfile.TemporaryDirectory() as tmp:
        variants, seeds = ["standard"], [7]
        base = {"split": "test", "eval_ctx": 512, "eval_iters": 50,
                "eval_iters_requested": 50, "val_windows": 80,
                "val_tokens_evaluated": 25600,
                "selection_set_is_eval_set": False}
        d = os.path.join(tmp, "standard")
        os.makedirs(d)
        _write_json(os.path.join(d, "config_used.json"), {
            "lr": 0.001, "steps": 10, "variant": "standard", "seed": 7,
            "device": "cpu", "tag": "t"})
        _write_json(os.path.join(d, "eval.json"), {
            "variant": "standard", "seed": 7, "data_fingerprint": "abc123",
            "code_commit": "deadbeef", "config_sha256": "shax",
            "steps": 10, "tokens_seen": 1024, "ckpt_step": 5, **base})
        for bad, expect_frac in ((True, None), (10.5, None), ("10", 0.5),
                                 (10.0, 0.5)):
            ev = json.load(open(os.path.join(d, "eval.json"), encoding="utf-8"))
            ev["steps_planned"] = bad
            ev["steps"] = 10            # 实际步数保持正常，聚焦选型比例解析
            ev["tokens_seen"] = 1024
            # 计划步数非法（bool/小数）时全局一致性审计会先失败；此时只断言
            # “不会算出比例”，比例字段本身在成功路径上再验证。
            if expect_frac is None:
                ev_bad = dict(ev, steps=bad)   # 实际步数也与计划同值（非法）
                _write_json(os.path.join(d, "eval.json"), ev_bad)
                try:
                    rep = verify_controlled_setup(tmp, variants, seeds, 1,
                                                  expected_data_digest="abc123")
                except SystemExit:
                    continue
                # 未触发全局失败时，选型比例仍必须是 None（分母不可整数化）
                assert rep["selection_step_points"][0][3] is None, bad
                continue
            _write_json(os.path.join(d, "eval.json"), ev)
            rep = verify_controlled_setup(tmp, variants, seeds, 1,
                                          expected_data_digest="abc123")
            point = rep["selection_step_points"][0]
            assert point[3] == expect_frac, (bad, point)
        # 合法大整数：ckpt_step/计划步数可精确解析，但比例转 float 会溢出；
        # 审计必须继续运行并把比例记为缺失，而不是崩溃。
        ev = json.load(open(os.path.join(d, "eval.json"), encoding="utf-8"))
        ev.update({"steps_planned": 10, "steps": 10, "tokens_seen": 1024,
                   "ckpt_step": 10 ** 400})
        _write_json(os.path.join(d, "eval.json"), ev)
        rep = verify_controlled_setup(tmp, variants, seeds, 1,
                                      expected_data_digest="abc123")
        assert rep["ok"] and rep["selection_step_points"][0][3] is None, rep
        # steps 与计划步数的数值/字符串表示必须按精确整数比较，不能先转
        # float（10**400 会溢出）后误判为不一致。
        huge = 10 ** 400
        ev.update({"steps_planned": huge, "steps": str(huge),
                   "tokens_seen": huge, "ckpt_step": huge})
        _write_json(os.path.join(d, "eval.json"), ev)
        rep = verify_controlled_setup(tmp, variants, seeds, 1,
                                      expected_data_digest="abc123")
        assert rep["ok"], rep
    print("  [ok] 受控变量审计：选型比例分母拒绝 bool/小数计划步数；"
          "超大整数精确比较且比例溢出安全降级")


def test_capacity_decomposition_uses_matched_null_arm():
    """容量分解：Full/Null/No 三臂同 seed，info+capacity=total。"""
    from run_all import (_capacity_decomposition,
                        _render_capacity_decomposition)
    runs = {}
    for variant, ppls in {
        "lega": [10.0, 11.0, 12.0],
        "lega_null_query": [12.0, 14.0, 15.0],
        "lega_no_query": [15.0, 18.0, 19.0],
    }.items():
        runs[variant] = {
            seed + 1: {"eval": {"split": "val", "val_ppl": value}}
            for seed, value in enumerate(ppls)
        }
    out = _capacity_decomposition(runs)
    assert out["requested"] is True and out["available"] is True
    rows = out["metrics"]["PPL"]
    row = next(r for r in rows if r["component"] == "A")
    assert row["status"] == "estimated" and row["n"] == 3
    assert abs(row["info_mean"] - 2.6666666667) < 1e-6
    assert abs(row["capacity_mean"] - 3.6666666667) < 1e-6
    assert abs(row["total_mean"] - 6.3333333334) < 1e-6
    assert row["identity_residual_max_abs"] == 0.0
    assert row["info_p_adj"] is not None
    assert row["capacity_p_adj"] is not None
    assert row["total_p_adj"] is not None
    # B/C 只缺对照时必须显式标记，而不是返回空表或伪造估计。
    for component in ("B", "C"):
        rec = next(r for r in rows if r["component"] == component)
        assert rec["status"] == "missing_variants"
        assert rec["missing_variants"]
    md = _render_capacity_decomposition(out)
    assert any("训练期逐组件容量匹配" in line for line in md)
    assert any("信息效应" in line and "容量效应" in line for line in md)
    print("  [ok] 容量分解使用同 seed 三臂并保留缺失状态")
