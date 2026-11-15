"""受控单维度对照的纪律测试：只改一项的设计维度必须真的只改一项。

受控单维度对照（替代集中度统计量 / 反传路径 / 证据标量共享粒度 /
空槽写入）回答的是“只改这一项会怎样”。它们与 A/B/C 组件归因不是同一
个问题——两侧的 A/B/C 因子集合本来就必须完全相同——因此不能套用因子
差判定，也不能被笼统标成“整体方法对比”。本文件覆盖三层：

  * **登记层**：每组对照声明了它只改哪个设计维度；
  * **实现层**：实测两侧的维度向量确实只差这一个维度，且门控因子集合
    相同（否则它已经变成多维度对照或因子消融）；
  * **结果层**：配对结果带 ``contrast_type="single_dimension"`` 标记，
    独立 Holm 家族，且永远不能被当作 A/B/C 组件证据。
"""

from unittest import mock

from run_all import (
    CONTROLLED_DIMENSION_CONTRASTS,
    CONTROLLED_DIMENSION_KEYS,
    _audit_controlled_single_dimension,
    _component_test_status,
    _paired_comparisons,
    _ppl_of,
    _render_controlled_dimension,
    controlled_dimension_analysis,
    controlled_dimension_pairs,
    controlled_dimension_tests,
    single_factor_audit,
)


def _runs(bases, seeds=(1, 2, 3)):
    """构造最小 run 集合：{变体: {种子: {"eval": {...}}}}。"""
    out = {}
    for variant, base in bases.items():
        out[variant] = {
            s: {"eval": {"val_ppl": base + 0.01 * s, "split": "val"}} for s in seeds
        }
    return out


def test_controlled_dimension_registry_declares_one_dimension_each():
    """登记表里的每一组都必须声明一个合法维度，且两侧不同。"""
    assert CONTROLLED_DIMENSION_CONTRASTS, "受控单维度对照登记表为空"
    seen = set()
    for entry in CONTROLLED_DIMENSION_CONTRASTS:
        cand = entry["candidate"]
        ref = entry["reference"]
        assert cand != ref, entry
        assert entry["dimension"] in CONTROLLED_DIMENSION_KEYS, entry
        assert (cand, ref) not in seen, f"重复的受控对照：{cand} vs {ref}"
        seen.add((cand, ref))
    print("  [ok] 受控单维度对照登记表结构合法")


def test_controlled_dimension_audit_accepts_registered_contrasts():
    """实现层核对：每组对照实测只差一个设计维度，且门控因子集合相同。"""
    audit = _audit_controlled_single_dimension()
    assert audit["ok"] is True, audit["problems"]
    assert audit["contrasts"], "审计没有产出任何受控对照行"
    for row in audit["contrasts"]:
        assert row["ok"] is True, row
        assert row["changed_dimensions"] == [row["dimension"]], row
        assert row["gate_factors_candidate"] == row["gate_factors_reference"], row
    print("  [ok] 受控单维度对照实现层核对通过")


def test_controlled_dimension_audit_detects_multi_dimension_drift():
    """实现悄悄多改一个维度时，审计必须显式失败。"""
    real = {
        "lega": {
            "variant": "lega",
            "gate_factors": ["A", "B", "C"],
            "conc_stat": "max",
            "conc_detach": False,
            "gate_shared_scalars": False,
            "null_value_learnable": None,
            "error": None,
        },
    }

    def fake_probe(variant, **kwargs):
        row = dict(
            real.get(
                variant,
                {
                    "variant": variant,
                    "gate_factors": ["A", "B", "C"],
                    "conc_stat": "max",
                    "conc_detach": False,
                    "gate_shared_scalars": False,
                    "null_value_learnable": None,
                    "error": None,
                },
            )
        )
        if variant == "lega_l2":
            row["conc_stat"] = "l2"
            # 漂移：除了换统计量，还顺手 detach 了集中度的反传路径。
            row["conc_detach"] = True
        return row

    with mock.patch(
        "lega.pipeline.common.probe_controlled_dimensions", side_effect=fake_probe
    ):
        audit = _audit_controlled_single_dimension()
    assert audit["ok"] is False, "多维度漂移没有被审计挡住"
    assert any("lega_l2" in p for p in audit["problems"]), audit["problems"]
    row = next(r for r in audit["contrasts"] if r["candidate"] == "lega_l2")
    assert row["ok"] is False, row
    assert row["changed_dimensions"] == ["conc_stat", "conc_detach"], row
    print("  [ok] 受控对照的多维度漂移被审计显式拒绝")


def test_controlled_dimension_audit_detects_gate_factor_drift():
    """两侧的门控因子集合不同就不再是“单维度对照”，必须显式失败。"""
    base = {
        "variant": "x",
        "gate_factors": ["A", "B", "C"],
        "conc_stat": "max",
        "conc_detach": False,
        "gate_shared_scalars": False,
        "null_value_learnable": None,
        "error": None,
    }

    def fake_probe(variant, **kwargs):
        row = dict(base)
        row["variant"] = variant
        if variant == "lega_l2":
            row["conc_stat"] = "l2"
            # 漂移：顺带关掉了 C 因子，这条对照已经变成因子消融。
            row["gate_factors"] = ["A", "B"]
        return row

    with mock.patch(
        "lega.pipeline.common.probe_controlled_dimensions", side_effect=fake_probe
    ):
        audit = _audit_controlled_single_dimension()
    assert audit["ok"] is False
    assert any("因子集合" in p for p in audit["problems"]), audit["problems"]
    print("  [ok] 受控对照里的因子改动被审计显式拒绝")


def test_single_factor_audit_exposes_controlled_dimension_section():
    """单因素审计的输出里必须带上受控单维度对照这一段。"""
    audit = single_factor_audit()
    assert audit["ok"] is True, audit["problems"]
    section = audit.get("controlled_single_dimension")
    assert isinstance(section, dict), audit.keys()
    assert section["ok"] is True, section["problems"]
    assert section["contrasts"], "受控单维度对照段没有任何行"
    print("  [ok] 单因素审计输出含受控单维度对照段")


def test_paired_comparisons_labels_controlled_contrasts_as_single_dimension():
    """受控对照必须单独标注，不能被笼统写成 overall，也不能冒充单因素。"""
    runs = _runs({"lega": 8.0, "lega_l2": 8.2, "lega_shared": 8.4, "standard": 11.0})
    pairs = [("lega_l2", "lega"), ("lega_shared", "lega"), ("lega", "standard")]
    out = _paired_comparisons(
        runs, pairs, "PPL", lambda slot: _ppl_of(slot.get("eval")), lower_better=True
    )
    by = {(t["candidate"], t["reference"]): t for t in out}
    for pair in (("lega_l2", "lega"), ("lega_shared", "lega")):
        t = by[pair]
        assert t["contrast_type"] == "single_dimension", t
        assert t["changed_dimensions"], t
        assert t["changed_factors"] == [], t
    # 网格外对比仍是整体方法对比。
    assert by[("lega", "standard")]["contrast_type"] == "overall"
    print("  [ok] 受控单维度对照在配对结果里带 single_dimension 标记")


def test_controlled_dimension_tests_form_an_independent_family():
    """受控对照走独立 Holm 家族，且不得进入组件证据。"""
    runs = _runs({"lega": 8.0, "lega_l2": 8.4})
    assert controlled_dimension_pairs(runs) == [("lega_l2", "lega")]
    tests = controlled_dimension_tests(runs)
    assert tests, "没有产出任何受控对照检验"
    for t in tests:
        assert t["exploratory"] is True, t
        assert t["controlled_dimension"] is True, t
        assert t["contrast_type"] == "single_dimension", t
        assert t.get("p_adj") is not None, t
    # 受控对照永远不能被当作 A/B/C 组件证据。
    status = _component_test_status(tests[0], removing=True)
    assert status["supported"] is False, status
    assert status.get("multi_factor_contrast") is True, status
    print("  [ok] 受控单维度对照独立成族且被排除在组件证据之外")


def test_controlled_dimension_analysis_reports_missing_runs():
    """两侧都没跑时必须说“证据缺失”，不能伪造“无影响”。"""
    analysis = controlled_dimension_analysis({})
    assert analysis["summary"]["n_ran"] == 0, analysis["summary"]
    verdict = analysis["summary"]["verdict"]
    assert "未运行" in verdict or "没有" in verdict, verdict
    md = "\n".join(_render_controlled_dimension(analysis))
    assert "受控单维度对照" in md, md
    assert "未运行" in md or "没有" in md, md
    print("  [ok] 受控单维度对照缺失时如实报告证据不足")


def test_report_renders_controlled_dimension_table():
    """汇总页必须渲染出对照、维度与实现层核对结论。"""
    runs = _runs({"lega": 8.0, "lega_l2": 8.4, "null": 9.0, "null_zero": 9.2})
    analysis = controlled_dimension_analysis(runs)
    md = "\n".join(_render_controlled_dimension(analysis))
    assert "受控单维度对照" in md, md
    assert "lega_l2" in md and "null_zero" in md, md
    assert "实现层核对通过" in md, md
    print("  [ok] 汇总页渲染受控单维度对照小节")
