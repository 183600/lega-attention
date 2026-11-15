"""统计校正、组件证据与受控变量审计测试。"""

import json
import os
import shutil
import tempfile
from unittest import mock

from lega.ablate import (
    LESION_ORDER as ABLATION_LESION_ORDER,
    probe_lesion_channels,
)
from tests.pipeline_helpers import _mk, _write_json
from run_all import (
    ALPHA,
    MIN_SEEDS,
    _apply_holm,
    _effect_status,
    _noise_floor_pct,
    _rel_pct,
    _result_files,
    _stat,
    build_comparison_pairs,
    build_summary,
    component_evidence,
    cross_variant_badcase_map,
    load_registered_hypotheses,
    single_factor_audit,
    tradeoff_verdict,
    verify_controlled_setup,
)


def test_apply_holm_verdicts():
    tests = [
        _mk("lega", "standard", "PPL", -0.5, 0.01, 5),
        _mk("qgate", "standard", "PPL", -0.01, 0.2, 5),
        _mk("lega_no_query", "lega", "PPL", 0.3, 0.04, 5),
        _mk("lega_len_only", "standard", "PPL", -0.2, 0.05, 2),
        _mk("lega", "standard", "RULER平均", 0.1, 0.03, 5, lower=False),
    ]
    _apply_holm(tests)
    # PPL 家族 m=4：升序 p=[0.01,0.04,0.05,0.2] → adj=[0.04,0.12,0.12,0.2]
    assert abs(tests[0]["p_adj"] - 0.04) < 1e-12
    assert "显著优于" in tests[0]["verdict"]
    assert abs(tests[2]["p_adj"] - 0.12) < 1e-12
    assert "不显著" in tests[2]["verdict"]
    assert "种子不足" in tests[3]["verdict"]
    # RULER 是独立家族（m=1）：校正不改变 0.03<alpha，且方向按 lower=False 判定
    assert abs(tests[4]["p_adj"] - 0.03) < 1e-12
    assert "显著优于" in tests[4]["verdict"]
    # 无法检验（n<2 → p=None）不参与家族、不给结论
    t = _mk("x", "y", "PPL", 1.0, None, n=1)
    _apply_holm([t])
    assert t["p_adj"] is None and "无法检验" in t["verdict"]
    assert ALPHA == 0.05 and MIN_SEEDS == 3
    print("  [ok] _apply_holm（逐指标家族校正/显著性结论/种子不足）")


def test_component_evidence_requires_significance():
    tests = [
        # A：去一法 Full−A 显著变差（PPL 变大）→ 有证据；全因子未检出
        _mk("lega_no_query", "lega", "PPL", 0.4, 0.005),
        _mk("qgate", "gate_only", "PPL", -0.01, 0.7),
        # B：两个方向都不显著 → 无证据
        _mk("lega_no_conc", "lega", "PPL", 0.05, 0.5),
        _mk("lega_conc_only", "gate_only", "PPL", -0.02, 0.6),
        # C：PPL 上去掉反而更好（显著），但 C 的主指标是 RULER；RULER 未检验
        _mk("lega_no_len", "lega", "PPL", -0.1, 0.001),
        _mk("lega_len_only", "gate_only", "PPL", 0.03, 0.4),
    ]
    _apply_holm(tests)
    comp = {c["component"]: c for c in component_evidence(tests)}
    assert comp["A 查询项"]["supported"] == ["PPL"]
    assert comp["A 查询项"]["verdict"].startswith("有证据支持")
    assert comp["A 查询项"]["primary_metric"] == "PPL"
    assert comp["B 集中度证据"]["verdict"] == "证据不足"
    # C 的主指标是 RULER；PPL 上的（相反）结果只能作探索性补充
    assert comp["C 长度校准"]["verdict"] == "证据不足"
    assert comp["C 长度校准"]["primary_metric"] == "RULER平均"
    assert "方向相反" in comp["C 长度校准"]["loo"]["PPL"]["reason"]
    assert comp["C 长度校准"]["secondary"]["PPL"]["exploratory"] is True
    print("  [ok] component_evidence（主指标判定/去一法/全因子/方向门槛）")


def test_component_evidence_requires_valid_seed_count():
    """缺 n / max(n, 1) 非法时不能默认“种子充足”而给出组件有用证据。

    汇总文件可能被旧版本或手工编辑成缺 n；阈值判定必须先证明 n>=MIN_SEEDS。
    """
    t = _mk("lega_no_query", "lega", "PPL", 0.4, 0.005)
    t.pop("n")
    t["p_adj"] = 0.005
    st = _effect_status(t, removing=True)
    assert not st["supported"] and "种子" in st["reason"], st
    for bad in (True, 2.5, "bad"):
        t2 = _mk("lega_no_query", "lega", "PPL", 0.4, 0.005)
        t2["n"] = bad
        t2["p_adj"] = 0.005
        st2 = _effect_status(t2, removing=True)
        assert not st2["supported"] and "种子" in st2["reason"], st2
    print("  [ok] 组件证据拒绝缺 n/非法 n 的“显著”检验")


def test_component_evidence_flags_contradiction():
    """一套消融显著支持、另一套显著反向时必须判冲突，不能用 OR 静默取胜。"""
    tests = [
        _mk("lega_no_query", "lega", "PPL", 0.4, 0.005),  # 去 A 变差：支持
        _mk("qgate", "gate_only", "PPL", 0.3, 0.004),  # 加 A 反而变差：反向
    ]
    _apply_holm(tests)
    a = next(c for c in component_evidence(tests) if c["component"] == "A 查询项")
    assert a["supported"] == []
    assert a["verdict"].startswith("证据冲突")
    # 只有一套显著反向、另一套不显著时仍必须判冲突，不能降格成“证据不足”
    one_sided = [
        _mk("lega_no_query", "lega", "PPL", 0.4, 0.5),  # 去 A 不显著
        _mk("qgate", "gate_only", "PPL", 0.3, 0.005),  # 加 A 显著反向
    ]
    _apply_holm(one_sided)
    a2 = next(c for c in component_evidence(one_sided) if c["component"] == "A 查询项")
    assert a2["verdict"].startswith("证据冲突"), a2
    print("  [ok] component_evidence（任一套显著反向 -> 证据冲突，另一套不显著亦然）")


def test_component_evidence_rejects_non_single_factor_tests():
    """组件判定只能使用被标记为单因素的检验。

    ``_paired_comparisons`` 只给已审计的 6 条边打 ``contrast_type="single_factor"``；
    其它对比（vs standard、LEGA vs qgate、B-only vs Full 这类同时改两个
    因子的边）标为 ``overall``。若组件判定误用了这类显著结果，"组件有用"
    就建立在多因子差异上——单因素边界必须在使用点再拦一次。
    """
    tests = [
        _mk("lega_no_query", "lega", "PPL", 0.4, 0.005),
        _mk("qgate", "gate_only", "PPL", -0.01, 0.7),
    ]
    for t in tests:
        t["contrast_type"] = "overall"
    _apply_holm(tests)
    comp = {c["component"]: c for c in component_evidence(tests)}
    a = comp["A 查询项"]
    assert a["supported"] == [], a
    assert a["verdict"] == "证据不足", a
    assert "不是单因素" in a["loo"]["PPL"]["reason"], a
    # 标成 single_factor 的同一条对比才能进入判定
    tests[0]["contrast_type"] = "single_factor"
    comp = {c["component"]: c for c in component_evidence(tests)}
    assert comp["A 查询项"]["supported"] == ["PPL"], comp["A 查询项"]
    print("  [ok] 组件判定拒绝非单因素对比（overall 不得支撑组件归因）")


def test_tradeoff_verdict_rejects_bad_deals():
    """涨点-增幅权衡判定：不显著涨点说服力有限；大增幅小收益被拒绝。"""
    # 不显著 + 增幅 >10% → 不可行（微小且不显著的涨点 + 大增幅）
    v = tradeoff_verdict(benefit_pct=0.3, p_adj=0.4, cost_pct=30.0)
    assert v.startswith("不可行"), v
    # 不显著 + 增幅小 → 说服力有限
    v = tradeoff_verdict(benefit_pct=0.3, p_adj=0.4, cost_pct=2.0)
    assert v.startswith("说服力有限"), v
    # 显著但增幅 >5×收益 → 收益-增幅比不利
    v = tradeoff_verdict(benefit_pct=0.5, p_adj=0.01, cost_pct=900.0)
    assert "收益-增幅比不利" in v or "需审视" in v, v
    # 显著 + 增幅温和（<5×收益）→ 可接受
    v = tradeoff_verdict(benefit_pct=5.0, p_adj=0.01, cost_pct=10.0)
    assert v == "可接受", v
    # 显著但增幅 >50% → 需审视
    v = tradeoff_verdict(benefit_pct=5.0, p_adj=0.01, cost_pct=60.0)
    assert v.startswith("需审视"), v
    # 数据缺失 → 不下结论
    assert tradeoff_verdict(None, 0.01, 10.0) == "数据不足"
    assert tradeoff_verdict(1.0, 0.01, None) == "数据不足"
    assert (
        tradeoff_verdict(None, None, 10.0, ruler_benefit_pp=None, ruler_p_adj=None)
        == "数据不足"
    )
    # PPL 显著变差不是“收益”：绝不能得到“可接受”
    v = tradeoff_verdict(benefit_pct=-5.0, p_adj=0.01, cost_pct=1.0)
    assert v.startswith("不可行") and "可接受" not in v, v
    v = tradeoff_verdict(benefit_pct=-0.3, p_adj=0.4, cost_pct=2.0)
    assert v.startswith("说服力有限"), v
    # 种子数不足与全表口径一致：p_adj 再小也不判定
    v = tradeoff_verdict(benefit_pct=5.0, p_adj=0.001, cost_pct=1.0, n_seeds=2)
    assert v.startswith("种子不足"), v
    assert tradeoff_verdict(5.0, 0.01, 1.0, n_seeds=MIN_SEEDS) == "可接受"
    for bad in (True, 3.9, "bad", -2):
        v = tradeoff_verdict(5.0, 0.01, 1.0, n_seeds=bad)
        assert v.startswith("种子"), (bad, v)
    v = tradeoff_verdict(
        5.0,
        0.01,
        1.0,
        n_seeds=5,
        ruler_benefit_pp=1.0,
        ruler_p_adj=0.01,
        ruler_n_seeds=2.9,
    )
    assert v.startswith("种子") and "RULER" in v, v
    # 外部 JSON 的 bool/非数值不能伪装成 p=0 或抛 TypeError。
    assert tradeoff_verdict(5.0, False, 1.0, n_seeds=5).startswith("说服力有限")
    assert tradeoff_verdict(True, 0.01, 1.0, n_seeds=5) == "数据不足"
    assert tradeoff_verdict(5.0, "bad", 1.0, n_seeds=5).startswith("说服力有限")
    assert tradeoff_verdict(5.0, 0.01, "bad", n_seeds=5) == "数据不足"
    print("  [ok] tradeoff_verdict（不显著/大增幅/收益增幅比/负收益/种子不足）")


def test_tradeoff_verdict_covers_ruler_primary_metric():
    """权衡判定必须覆盖 RULER 主指标：只看 PPL 会误判长上下文涨点，
    也会漏掉 RULER 显著退步。"""
    # PPL 持平（不显著）+ RULER 显著涨点 + 增幅温和 → 有涨点且注明依据
    v = tradeoff_verdict(
        benefit_pct=0.1,
        p_adj=0.6,
        cost_pct=8.0,
        n_seeds=5,
        ruler_benefit_pp=1.5,
        ruler_p_adj=0.01,
        ruler_n_seeds=5,
    )
    assert v.startswith("可接受") and "RULER" in v, v
    # RULER 单独显著改善 + 增幅超过 10×收益（RULER 独立阈值）→ 比值不利
    v = tradeoff_verdict(
        benefit_pct=0.1,
        p_adj=0.6,
        cost_pct=30.0,
        n_seeds=5,
        ruler_benefit_pp=1.0,
        ruler_p_adj=0.01,
        ruler_n_seeds=5,
    )
    assert "收益-增幅比不利" in v and "RULER" in v, v
    # RULER 显著改善但 PPL 显著退步 → 权衡，而不是“不可行”
    v = tradeoff_verdict(
        benefit_pct=-3.0,
        p_adj=0.01,
        cost_pct=10.0,
        n_seeds=5,
        ruler_benefit_pp=2.0,
        ruler_p_adj=0.01,
        ruler_n_seeds=5,
    )
    assert v.startswith("权衡") and "PPL" in v and "RULER" in v and "不可行" not in v, v
    # PPL 显著改善但 RULER 显著退步 → 权衡（核心检索目标受损必须显式）
    v = tradeoff_verdict(
        benefit_pct=4.0,
        p_adj=0.01,
        cost_pct=5.0,
        n_seeds=5,
        ruler_benefit_pp=-2.0,
        ruler_p_adj=0.01,
        ruler_n_seeds=5,
    )
    assert v.startswith("权衡") and "RULER" in v and "受损" in v, v
    # RULER 显著退步且无 PPL 涨点 → 不可行（不能因 PPL 不动就静默放过）
    v = tradeoff_verdict(
        benefit_pct=0.0,
        p_adj=0.6,
        cost_pct=5.0,
        n_seeds=5,
        ruler_benefit_pp=-1.5,
        ruler_p_adj=0.01,
        ruler_n_seeds=5,
    )
    assert v.startswith("不可行") and "RULER" in v, v
    # 两个主指标都显著改善 → 依据两者；RULER 侧种子不足同样不能判定
    v = tradeoff_verdict(
        benefit_pct=3.0,
        p_adj=0.01,
        cost_pct=8.0,
        n_seeds=5,
        ruler_benefit_pp=1.0,
        ruler_p_adj=0.01,
        ruler_n_seeds=5,
    )
    assert v.startswith("可接受") and "PPL+RULER" in v, v
    v = tradeoff_verdict(
        benefit_pct=3.0,
        p_adj=0.01,
        cost_pct=8.0,
        n_seeds=5,
        ruler_benefit_pp=1.0,
        ruler_p_adj=0.01,
        ruler_n_seeds=2,
    )
    assert v.startswith("种子不足") and "RULER" in v, v
    print("  [ok] tradeoff_verdict（RULER 涨点/退步/权衡/独立阈值）")


def test_load_registered_hypotheses_tolerates_non_dict_json():
    """顶层为 list/null/str 的合法 JSON 也必须按“空 dict”降级而不是崩。"""
    tmp = tempfile.mkdtemp()
    try:
        p = os.path.join(tmp, "hypotheses.json")
        for bad in ([{"id": "H1"}], None, "text"):
            with open(p, "w", encoding="utf-8") as f:
                json.dump(bad, f, ensure_ascii=False)
            assert load_registered_hypotheses(p) == {}
        _write_json(p, {"hypotheses": [{"id": "H1"}]})
        assert load_registered_hypotheses(p)["hypotheses"][0]["id"] == "H1"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] load_registered_hypotheses：非 dict JSON 安全降级")


def test_build_summary_audit_zero_checked_is_warning_not_failure():
    """ok=True 但 checked=0 表示“没有可核对的 run”，不能渲染成 ❌ 未通过。"""
    tmp = tempfile.mkdtemp()
    try:
        out = build_summary(tmp, audit={"ok": True, "checked": 0, "mismatches": []})
        md = out["markdown"]
        assert "未检查受控变量" in md
        assert "❌ 未通过" not in md
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] build_summary：checked=0 的审计显示为未检查而非未通过")


def test_cross_variant_badcase_map_min_seed_even_when_empty():
    """最小种子零错误也注册为 []（参与共识判定）；
    最小种子无可用的 RULER 错误清单时退化到最小的有清单的种子；未评测不进表。"""
    cases99 = [{"sample_id": "x"}]
    # 最小种子（7）零错误，较大种子（99）有错 → 仍取最小种子的 []
    runs = {
        "lega": {7: {"ruler": {"badcases": []}}, 99: {"ruler": {"badcases": cases99}}}
    }
    assert cross_variant_badcase_map(runs, ["lega"]) == {"lega": []}
    # 最小种子没有 RULER 结果 → 退化到最小有结果的种子
    runs2 = {"lega": {1: {}, 5: {"ruler": {"badcases": cases99}}}}
    assert cross_variant_badcase_map(runs2, ["lega"]) == {"lega": cases99}
    # 最小种子的 ruler 产物缺 badcases 键/非列表 → 不是“零错误”，
    # 退化到下一个有合法清单的种子；全无合法清单则排除该变体。
    runs3 = {
        "lega": {
            1: {"ruler": {"results": {}}},
            2: {"ruler": {"badcases": "bad"}},
            5: {"ruler": {"badcases": cases99}},
        }
    }
    assert cross_variant_badcase_map(runs3, ["lega"]) == {"lega": cases99}
    runs4 = {"lega": {1: {"ruler": {"results": {}}}, 2: {"ruler": {"badcases": "bad"}}}}
    assert cross_variant_badcase_map(runs4, ["lega"]) == {}
    # 完全未跑 RULER → 不注册（无法参与对照）
    assert cross_variant_badcase_map({"lega": {1: {}}}, ["lega"]) == {}
    print("  [ok] cross_variant_badcase_map（最小种子优先，零错误显式注册）")


def test_build_summary_scopes_discovery_to_current_repro_run():
    """results/<tag> 跨多次运行复用时，汇总只纳入 repro.json 记录的本次
    eval/sanity（不同实验的观测不相配成对）。"""
    tmp = tempfile.mkdtemp()
    try:
        _write_json(
            os.path.join(tmp, "repro.json"),
            {"variants": ["lega"], "seeds": [1], "smoke": False},
        )
        _write_json(
            os.path.join(tmp, "eval_lega.json"),
            {
                "variant": "lega",
                "seed": 1,
                "val_ppl": 9.0,
                "val_loss": 2.0,
                "n_params": 10,
            },
        )
        _write_json(
            os.path.join(tmp, "sanity_lega.json"),
            {
                "variant": "lega",
                "passed": True,
                "initial": {"loss": 2.5},
                "final": {"loss": 1.0, "token_acc": 0.4},
            },
        )
        # 上一轮遗留：旧变体、旧种子、未通过的旧 sanity
        _write_json(
            os.path.join(tmp, "eval_standard.json"),
            {
                "variant": "standard",
                "seed": 1,
                "val_ppl": 10.0,
                "val_loss": 2.3,
                "n_params": 10,
            },
        )
        _write_json(
            os.path.join(tmp, "eval_lega_seed99.json"),
            {
                "variant": "lega",
                "seed": 99,
                "val_ppl": 1.0,
                "val_loss": 0.1,
                "n_params": 10,
            },
        )
        _write_json(
            os.path.join(tmp, "sanity_standard.json"),
            {
                "variant": "standard",
                "passed": False,
                "initial": {"loss": 9.9},
                "final": {"loss": 9.9, "token_acc": 0.01},
            },
        )
        out = build_summary(tmp)
        assert [r["variant"] for r in out["rows"]] == ["lega"]
        md = out["markdown"]
        assert "❌ 未通过" not in md  # 旧 sanity 不得混入
        assert "10.0000" not in md  # 旧 standard 的 PPL 不得混入
        assert "0.1000" not in md  # 旧种子的 val_loss 不得混入
        sig = json.load(open(os.path.join(tmp, "significance.json"), encoding="utf-8"))
        assert sig["pairs"] == []
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] build_summary（repro.json 限定本次运行的变体/种子发现范围）")


def test_build_summary_excludes_artifacts_from_other_pipeline_runs():
    """同一 results 目录重跑时，旧 run_id 或旧数据指纹的产物不得混入汇总。"""
    tmp = tempfile.mkdtemp()
    try:
        repro = {
            "variants": ["lega"],
            "seeds": [1, 2],
            "smoke": False,
            "run_id": "run-new",
            "data": {"digest": "data-new"},
        }
        _write_json(os.path.join(tmp, "repro.json"), repro)
        current = {
            "variant": "lega",
            "seed": 1,
            "val_ppl": 9.8765,
            "val_loss": 2.0,
            "n_params": 10,
            "pipeline_run_id": "run-new",
            "data_fingerprint": "data-new",
        }
        _write_json(os.path.join(tmp, "eval_lega_seed1.json"), current)
        stale_run = {
            "variant": "lega",
            "seed": 2,
            "val_ppl": 1.2345,
            "val_loss": 0.1,
            "n_params": 10,
            "pipeline_run_id": "run-old",
            "data_fingerprint": "data-new",
        }
        _write_json(os.path.join(tmp, "eval_lega_seed2.json"), stale_run)
        # 旧流水线还留下一个缺少 run_id 的同 (变体, 种子) 产物：无法证明
        # 属于本轮，必须排除而不是默认信任。
        _write_json(
            os.path.join(tmp, "ruler_lega_seed2.json"),
            {"variant": "lega", "seed": 2, "n": 1, "data_fingerprint": "data-new"},
        )
        out = build_summary(tmp)
        assert "1.2345" not in out["markdown"]
        row = out["rows"][0]
        assert row["variant"] == "lega"
        assert row.get("n_seeds") == 1, row

        # run_id 相同但数据指纹不同：仍是另一份数据上的观测。
        stolen = dict(current)
        stolen.update(seed=2, val_ppl=5.4321, data_fingerprint="data-old")
        _write_json(os.path.join(tmp, "eval_lega_seed2.json"), stolen)
        out = build_summary(tmp)
        assert "5.4321" not in out["markdown"]
        assert out["rows"][0].get("n_seeds") == 1, out["rows"][0]

        # 本轮完整产物（run_id 与数据指纹都匹配）重新纳入，n 回到 2。
        fresh = dict(current)
        fresh.update(seed=2, val_ppl=8.0)
        _write_json(os.path.join(tmp, "eval_lega_seed2.json"), fresh)
        _write_json(
            os.path.join(tmp, "ruler_lega_seed2.json"),
            {
                "variant": "lega",
                "seed": 2,
                "n": 1,
                "pipeline_run_id": "run-new",
                "data_fingerprint": "data-new",
            },
        )
        out = build_summary(tmp)
        assert out["rows"][0].get("n_seeds") == 2, out["rows"][0]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] build_summary 按 run_id/数据指纹排除旧流水线产物")


def test_build_summary_warns_on_eval_runtime_drift():
    """评测环境与训练清单不一致时，汇总页必须显式警告而不是只写进 JSON。

    只记录全局 repro.json 的 runtime 不够：results 目录可能被另一环境单独
    补跑（--skip_train / 手工 evaluate），汇总读者看到的仍是同一份表格。
    """
    tmp = tempfile.mkdtemp()
    try:
        _write_json(
            os.path.join(tmp, "repro.json"),
            {
                "variants": ["lega"],
                "seeds": [1],
                "smoke": False,
                "runtime": {"python": "3.12", "torch": "2.14"},
            },
        )
        rec = {
            "variant": "lega",
            "seed": 1,
            "val_ppl": 9.0,
            "val_loss": 2.0,
            "n_params": 10,
        }
        _write_json(os.path.join(tmp, "eval_lega.json"), rec)
        md = build_summary(tmp)["markdown"]
        assert "评测环境" not in md  # 旧产物没有该字段：不伪造警告

        _write_json(
            os.path.join(tmp, "eval_lega.json"),
            dict(rec, runtime_matches_training=True, runtime_mismatches=[]),
        )
        md = build_summary(tmp)["markdown"]
        assert "评测环境" not in md

        _write_json(
            os.path.join(tmp, "eval_lega.json"),
            dict(
                rec,
                runtime_matches_training=False,
                runtime_mismatches=[
                    {"field": "torch", "train": "2.14", "eval": "2.15"}
                ],
            ),
        )
        md = build_summary(tmp)["markdown"]
        assert "评测环境" in md and "torch" in md and "lega/seed1" in md

        _write_json(
            os.path.join(tmp, "eval_lega.json"),
            dict(
                rec,
                source_tree_matches_training=False,
                train_source_tree_digest="aaa",
                eval_source_tree_digest="bbb",
            ),
        )
        md = build_summary(tmp)["markdown"]
        assert "源码/配置树" in md and "lega/seed1" in md

        _write_json(
            os.path.join(tmp, "repro.json"),
            {
                "variants": ["lega"],
                "seeds": [1],
                "smoke": True,
                "runtime": {"python": "3.12", "torch": "2.14"},
                "environment_gate": {
                    "status": "warn",
                    "message": "冒烟环境与 requirements-lock.txt 不一致",
                },
            },
        )
        md = build_summary(tmp)["markdown"]
        assert "环境锁闸门" in md and "requirements-lock.txt" in md
        assert "不得声称在锁定环境中产生" in md
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] build_summary 显式警告 eval/train 运行环境/源码树漂移")


def test_rel_pct_and_eff_cells():
    """相对变化计算：缺失与除零安全返回 None。"""
    assert _rel_pct(100.0, 110.0) == 10.0
    assert _rel_pct(50.0, 45.0) == -10.0
    assert _rel_pct(None, 10.0) is None
    assert _rel_pct(10.0, None) is None
    assert _rel_pct(0.0, 10.0) is None
    # 两端有限但百分比不可表示时同样是缺失，不能返回 ±inf 参与 max()。
    assert _rel_pct(1.0, 1e308) is None
    assert _rel_pct(1e-308, 1.0) is None
    # _eff_cell 读取 latency 条目的结构与运行时字段
    entry = {
        "n_params": 100,
        "efficiency": {
            "n_params": 100,
            "params_mb": 0.4,
            "flops_per_tok": {"256": {"total_flops_per_tok": 12345}},
            "peak_mem_mb_prefill": {"256": 3.5},
        },
    }
    from run_all import _eff_cell

    assert _eff_cell(entry, "params", 256) == 100
    assert _eff_cell(entry, "flops", 256) == 12345
    assert _eff_cell(entry, "mem", 256) == 3.5
    assert _eff_cell(None, "flops", 256) is None
    assert _eff_cell({"efficiency": {}}, "flops", 256) is None
    # 旧格式只有 per-token KV 时按 ctx 线性换算；超大/溢出 ctx 必须按缺失
    # 处理，而不是把 OverflowError 抛给汇总或写出 inf。
    old_kv = {"efficiency": {"kv_cache_mb_per_tok": 2.0}}
    assert _eff_cell(old_kv, "kv_mem", 256) == 512.0
    assert _eff_cell(old_kv, "kv_mem", 10**400) is None
    huge_kv = {"efficiency": {"kv_cache_mb_per_tok": 1e308}}
    assert _eff_cell(huge_kv, "kv_mem", 1e308) is None
    assert _noise_floor_pct(3.0, 4.0) == 5.0
    assert _noise_floor_pct(1e308, 1e308) is None
    assert _noise_floor_pct(-1.0, 2.0) is None
    # 外部坏值不能参与 `< 0` 比较（TypeError），统一按缺失处理。
    assert _noise_floor_pct(True, 2.0) is None
    assert _noise_floor_pct("bad", 2.0) is None
    assert _noise_floor_pct(float("nan"), 2.0) is None
    print("  [ok] _rel_pct/_eff_cell（缺失安全）")


def test_result_files_empty_scope_does_not_fall_back_to_all():
    """repro.seeds=[] 表示本次 run 无种子，不能扫描目录里的旧结果。"""
    tmp = tempfile.mkdtemp()
    try:
        for name in ("eval_standard.json", "eval_lega_seed1.json"):
            _write_json(os.path.join(tmp, name), {"variant": "x"})
        assert _result_files(tmp, "eval", None) == sorted(
            os.path.join(tmp, n) for n in ("eval_standard.json", "eval_lega_seed1.json")
        )
        assert _result_files(tmp, "eval", []) == []
        assert len(_result_files(tmp, "eval", [1])) == 2
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] _result_files：空 seeds 范围不回退全量旧结果")


def test_effect_status_directions():
    # RULER 越大越好：全因子“加入变好”必须 mean_diff>0
    good = _mk("qgate", "standard", "RULER平均", 0.05, 0.01, lower=False)
    good["p_adj"] = 0.01
    bad = _mk("qgate", "standard", "RULER平均", -0.05, 0.01, lower=False)
    bad["p_adj"] = 0.01
    assert _effect_status(good, removing=False)["supported"]
    assert not _effect_status(bad, removing=False)["supported"]
    # 种子不足直接判无证据，即使 p 值很小
    tiny = _mk("qgate", "standard", "PPL", -0.5, 0.001, n=2)
    tiny["p_adj"] = 0.001
    assert not _effect_status(tiny, removing=False)["supported"]
    print("  [ok] _effect_status（方向、种子门槛）")


def test_effect_status_requires_explicit_lower_is_better():
    """字符串 "false" 不能经 bool() 变成“越低越好”，否则方向判定反转。"""
    t = _mk("qgate", "standard", "RULER平均", -0.05, 0.01, lower="false")
    t["p_adj"] = 0.01
    st = _effect_status(t, removing=False)
    assert not st["supported"] and st["opposite"], st
    hm = _mk("qgate", "standard", "PPL", -0.05, 0.01, lower="false")
    _apply_holm([hm])
    assert "显著劣于" in hm["verdict"], hm
    print("  [ok] lower_is_better 只认显式布尔（方向判定不反转）")


def test_effect_status_rejects_significant_but_fragile_effects():
    """显著但 |Δ|<ΔSD 或留一翻转不能算组件有用，更不能进入确认性证据。

    用户口径：0.2% 的提升若落在 0.5% 的种子方差内，即使 p<0.05 也没有
    # NOTE: Consider if 0.05 threshold is appropriate for this context
    统计意义。这里显式构造“显著但噪声内/留一翻转”的检验，确认判定被降级。
    """
    noisy = _mk("qgate", "gate_only", "PPL", -0.02, 0.005)
    noisy["p_adj"] = 0.01
    noisy["within_noise"] = True
    noisy["loo_sign_stable"] = True
    st = _effect_status(noisy, removing=False)
    assert not st["supported"] and st["fragile"] and not st["opposite"], st
    assert "噪声" in st["reason"], st
    fragile_loo = _mk("qgate", "gate_only", "PPL", -0.4, 0.005)
    fragile_loo["p_adj"] = 0.01
    fragile_loo["within_noise"] = False
    fragile_loo["loo_sign_stable"] = False
    st2 = _effect_status(fragile_loo, removing=False)
    assert not st2["supported"] and st2["fragile"], st2
    assert "留一" in st2["reason"], st2
    # 两套消融都显著但都不稳健：组件结论必须是“证据不稳健”，不能是“有用”
    tests = [
        _mk("lega_no_query", "lega", "PPL", 0.02, 0.005),
        _mk("qgate", "gate_only", "PPL", -0.02, 0.005),
    ]
    for t in tests:
        t.update({"within_noise": True, "loo_sign_stable": False})
    _apply_holm(tests)
    a = next(c for c in component_evidence(tests) if c["component"] == "A 查询项")
    assert a["supported"] == [] and a["fragile"] is True, a
    assert a["confirmatory_supported"] is False, a
    assert a["evidence_status"] == "exploratory", a
    assert a["verdict"].startswith("证据不稳健"), a
    print("  [ok] 组件证据拒绝“显著但噪声内/留一翻转”的脆弱收益")


def test_verify_controlled_setup_consistency():
    tmp = tempfile.mkdtemp()
    try:
        variants, seeds = ["standard", "lega"], [7]
        for v in variants:
            d = os.path.join(tmp, v)
            os.makedirs(d)
            _write_json(
                os.path.join(d, "config_used.json"),
                {
                    "lr": 0.001,
                    "steps": 10,
                    "warmup": 2,
                    "variant": v,
                    "seed": 7,
                    "device": "cpu",
                    "tag": "t",
                },
            )
            _write_json(
                os.path.join(d, "eval.json"),
                {
                    "variant": v,
                    "seed": 7,
                    "data_fingerprint": "abc123",
                    "code_commit": "deadbeef",
                    "config_sha256": "shax",
                    "steps": 10,
                    "tokens_seen": 1024,
                },
            )
        audit = verify_controlled_setup(
            tmp, variants, seeds, 1, expected_data_digest="abc123"
        )
        assert audit["ok"] and audit["checked"] == 2
        assert audit["meta_checked"] == 2 and not audit["meta_mismatches"]

        # 篡改数据指纹：同一 config 跑了不同数据不可比，必须报错
        ev_path = os.path.join(tmp, "lega", "eval.json")
        ev = json.load(open(ev_path, encoding="utf-8"))
        ev["data_fingerprint"] = "different"
        _write_json(ev_path, ev)
        try:
            verify_controlled_setup(
                tmp, variants, seeds, 1, expected_data_digest="abc123"
            )
            raise AssertionError("数据指纹不一致应触发 SystemExit")
        except SystemExit:
            pass

        # 训练步数（steps）不一致同样必须报错
        ev["data_fingerprint"] = "abc123"
        ev["steps"] = 12
        _write_json(ev_path, ev)
        try:
            verify_controlled_setup(
                tmp, variants, seeds, 1, expected_data_digest="abc123"
            )
            raise AssertionError("训练步数不一致应触发 SystemExit")
        except SystemExit:
            pass

        # 超参数（config_used.json）不一致必须报错
        ev["steps"] = 10
        _write_json(ev_path, ev)
        cfg_path = os.path.join(tmp, "lega", "config_used.json")
        cfg = json.load(open(cfg_path, encoding="utf-8"))
        cfg["lr"] = 0.002
        _write_json(cfg_path, cfg)
        try:
            verify_controlled_setup(
                tmp, variants, seeds, 1, expected_data_digest="abc123"
            )
            raise AssertionError("学习率不一致应触发 SystemExit")
        except SystemExit:
            pass

        # 显式 null seed：审计应如实报告“不一致”
        cfg["lr"] = 0.001
        cfg["seed"] = None
        _write_json(cfg_path, cfg)
        try:
            verify_controlled_setup(
                tmp, variants, seeds, 1, expected_data_digest="abc123"
            )
            raise AssertionError("config seed=null 应触发 SystemExit")
        except SystemExit:
            pass
        cfg["seed"] = 7
        _write_json(cfg_path, cfg)
        ev["seed"] = None
        _write_json(ev_path, ev)
        try:
            verify_controlled_setup(
                tmp, variants, seeds, 1, expected_data_digest="abc123"
            )
            raise AssertionError("eval seed=null 应触发 SystemExit")
        except SystemExit:
            pass
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    # 缺少 eval.json 时只警告不阻塞（--skip_train 复用旧产物的场景）
    tmp2 = tempfile.mkdtemp()
    try:
        d = os.path.join(tmp2, "standard")
        os.makedirs(d)
        _write_json(
            os.path.join(d, "config_used.json"),
            {
                "lr": 0.001,
                "variant": "standard",
                "seed": 1,
                "device": "cpu",
                "tag": "t",
            },
        )
        audit = verify_controlled_setup(tmp2, ["standard"], [1], 1)
        assert audit["ok"] and audit["meta_checked"] == 0
        assert audit["missing_evals"] == ["standard/seed1"]
    finally:
        shutil.rmtree(tmp2, ignore_errors=True)
    print("  [ok] verify_controlled_setup（数据/代码/步数/超参数双层审计）")


def test_verify_controlled_setup_checks_actual_data_order():
    """同 seed 下各变体必须消费同一训练批次序列，而不只是 config/seed 相同。"""
    with tempfile.TemporaryDirectory() as tmp:
        variants, seeds = ["standard", "lega"], [7]
        base = {
            "lr": 0.001,
            "steps": 10,
            "variant": None,
            "seed": 7,
            "device": "cpu",
            "tag": "t",
        }
        meta = {
            "data_fingerprint": "abc123",
            "code_commit": "deadbeef",
            "config_sha256": "shax",
            "steps": 10,
            "tokens_seen": 1024,
        }
        for v in variants:
            d = os.path.join(tmp, v)
            os.makedirs(d)
            _write_json(os.path.join(d, "config_used.json"), dict(base, variant=v))
            _write_json(
                os.path.join(d, "eval.json"),
                dict(meta, variant=v, seed=7, data_order_sha256="order-A"),
            )
        audit = verify_controlled_setup(
            tmp, variants, seeds, 1, expected_data_digest="abc123"
        )
        assert audit["ok"] and audit["data_order_checked"] == 1, audit
        assert not audit["data_order_missing"], audit
        # 同 seed 但实际采样序列不同：即使 config/seed 完全一致也必须失败
        ev = json.load(open(os.path.join(tmp, "lega", "eval.json"), encoding="utf-8"))
        ev["data_order_sha256"] = "order-B"
        _write_json(os.path.join(tmp, "lega", "eval.json"), ev)
        try:
            verify_controlled_setup(
                tmp, variants, seeds, 1, expected_data_digest="abc123"
            )
            raise AssertionError("数据顺序指纹不一致应触发 SystemExit")
        except SystemExit:
            pass
        # 旧产物缺少指纹：只警告，不伪造一致
        ev.pop("data_order_sha256")
        _write_json(os.path.join(tmp, "lega", "eval.json"), ev)
        audit = verify_controlled_setup(
            tmp, variants, seeds, 1, expected_data_digest="abc123"
        )
        assert audit["ok"] and audit["data_order_missing"] == ["lega/seed7"]
    print("  [ok] 受控变量审计：同 seed 实际训练数据顺序指纹必须一致")


def test_verify_controlled_setup_rejects_mixed_attention_backend():
    """standard/legA 开了 SDPA 时实现路径不同；正式对比必须统一为显式路径。"""
    for sdpa, should_fail in ((True, True), (False, False)):
        tmp = tempfile.mkdtemp()
        try:
            variants, seeds = ["standard", "lega"], [7]
            for v in variants:
                d = os.path.join(tmp, v)
                os.makedirs(d)
                _write_json(
                    os.path.join(d, "config_used.json"),
                    {
                        "lr": 0.001,
                        "steps": 10,
                        "warmup": 2,
                        "use_sdpa_standard": sdpa,
                        "variant": v,
                        "seed": 7,
                        "device": "cpu",
                        "tag": "t",
                    },
                )
                _write_json(
                    os.path.join(d, "eval.json"),
                    {
                        "variant": v,
                        "seed": 7,
                        "data_fingerprint": "abc123",
                        "code_commit": "deadbeef",
                        "config_sha256": "shax",
                        "steps": 10,
                        "tokens_seen": 1024,
                    },
                )
            if should_fail:
                try:
                    verify_controlled_setup(
                        tmp, variants, seeds, 1, expected_data_digest="abc123"
                    )
                    raise AssertionError("混合 SDPA/显式路径应触发 SystemExit")
                except SystemExit:
                    pass
            else:
                audit = verify_controlled_setup(
                    tmp, variants, seeds, 1, expected_data_digest="abc123"
                )
                assert audit["ok"]
                assert audit["attention_backends"] == {
                    "standard": "explicit",
                    "lega": "explicit",
                }
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    # lega_placebo 也注册了集中度项（a·c 的槽位保留、证据为噪声），实际
    # 走显式注意力；审计必须把它当成 explicit，否则会漏报 standard(SDPA)
    # 与 placebo(显式) 的实现路径混用。
    tmp = tempfile.mkdtemp()
    try:
        variants, seeds = ["standard", "lega_placebo"], [7]
        for v in variants:
            d = os.path.join(tmp, v)
            os.makedirs(d)
            _write_json(
                os.path.join(d, "config_used.json"),
                {
                    "lr": 0.001,
                    "steps": 10,
                    "warmup": 2,
                    "use_sdpa_standard": True,
                    "variant": v,
                    "seed": 7,
                    "device": "cpu",
                    "tag": "t",
                },
            )
            _write_json(
                os.path.join(d, "eval.json"),
                {
                    "variant": v,
                    "seed": 7,
                    "data_fingerprint": "abc123",
                    "code_commit": "deadbeef",
                    "config_sha256": "shax",
                    "steps": 10,
                    "tokens_seen": 1024,
                },
            )
        try:
            verify_controlled_setup(
                tmp, variants, seeds, 1, expected_data_digest="abc123"
            )
        except SystemExit:
            pass
        else:
            raise AssertionError(
                "lega_placebo 与 standard 的注意力路径不同，" "审计必须拒绝"
            )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    # lega_raw 不并入 log n，但仍消费原始 log max_p，必须走显式 softmax；
    # 与 standard(SDPA) 混跑同样不能被审计放行。
    tmp = tempfile.mkdtemp()
    try:
        variants, seeds = ["standard", "lega_raw"], [7]
        for v in variants:
            d = os.path.join(tmp, v)
            os.makedirs(d)
            _write_json(
                os.path.join(d, "config_used.json"),
                {
                    "lr": 0.001,
                    "steps": 10,
                    "warmup": 2,
                    "use_sdpa_standard": True,
                    "variant": v,
                    "seed": 7,
                    "device": "cpu",
                    "tag": "t",
                },
            )
            _write_json(
                os.path.join(d, "eval.json"),
                {
                    "variant": v,
                    "seed": 7,
                    "data_fingerprint": "abc123",
                    "code_commit": "deadbeef",
                    "config_sha256": "shax",
                    "steps": 10,
                    "tokens_seen": 1024,
                },
            )
        try:
            verify_controlled_setup(
                tmp, variants, seeds, 1, expected_data_digest="abc123"
            )
        except SystemExit:
            pass
        else:
            raise AssertionError(
                "lega_raw 与 standard 的注意力路径不同，" "审计必须拒绝"
            )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] verify_controlled_setup（混合注意力实现路径被拒绝）")


def test_supplemental_variants_enter_summary_comparison_pairs():
    """显式运行的补充对照必须有相对 standard 的配对检验。

    补充变体不属于默认矩阵，历史上若只把它们写进汇总行、却不进入
    build_comparison_pairs，读者会看到 PPL 数值却没有任何显著性检验；
    设计层的 planned_comparison_pairs 又把它们算进 Holm 家族，两侧
    家族大小不一致。
    """
    from lega.attention import RUNNABLE_VARIANTS, SUPPLEMENTAL_VARIANTS

    runs = {v: {1: {}} for v in RUNNABLE_VARIANTS}
    pairs = build_comparison_pairs(runs, {"hypotheses": []})
    for v in SUPPLEMENTAL_VARIANTS:
        assert (v, "standard") in pairs, v
    # 预注册对比若点名补充/对照变体，也必须真正进入配对表并被检验。
    reg = {
        "hypotheses": [{"metric": "PPL", "candidate": "lega_raw", "reference": "lega"}]
    }
    assert ("lega_raw", "lega") in build_comparison_pairs(runs, reg)
    # 未运行的补充变体不得凭空产生对比。
    only_standard = build_comparison_pairs({"standard": {1: {}}}, {"hypotheses": []})
    for v in SUPPLEMENTAL_VARIANTS:
        assert (v, "standard") not in only_standard, v
    print("  [ok] 补充对照进入汇总配对，未运行时不伪造对比")


def test_verify_controlled_setup_reports_malformed_attention_switch():
    """config_used 的 use_sdpa_standard 非法时，审计应报告问题而非崩溃。"""
    tmp = tempfile.mkdtemp()
    try:
        for v in ("standard", "lega"):
            d = os.path.join(tmp, v)
            os.makedirs(d)
            _write_json(
                os.path.join(d, "config_used.json"),
                {
                    "lr": 0.001,
                    "steps": 10,
                    "use_sdpa_standard": "maybe",
                    "variant": v,
                    "seed": 7,
                    "device": "cpu",
                    "tag": "t",
                },
            )
            _write_json(
                os.path.join(d, "eval.json"),
                {
                    "variant": v,
                    "seed": 7,
                    "data_fingerprint": "abc123",
                    "code_commit": "deadbeef",
                    "config_sha256": "shax",
                    "steps": 10,
                    "tokens_seen": 1024,
                },
            )
        try:
            verify_controlled_setup(
                tmp, ["standard", "lega"], [7], 1, expected_data_digest="abc123"
            )
            raise AssertionError("非法 use_sdpa_standard 应触发 SystemExit")
        except SystemExit:
            pass
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] 受控变量审计：非法注意力开关报告问题而非崩溃")


def test_verify_controlled_setup_infers_missing_attention_switch():
    """缺省 use_sdpa_standard 时 train.py 用默认 True，审计不能静默跳过。

    config_used.json 没写该键时实际仍走 SDPA；standard/qgate 与必须显式
    softmax 的 lega 因而实现路径不同。旧实现只在“键存在”时才比较，导致
    一份完全省略该开关的配置绕过混合后端检查。
    """

    def fixture(variants):
        tmp = tempfile.mkdtemp()
        for v in variants:
            d = os.path.join(tmp, v)
            os.makedirs(d)
            # 直接写文件：不经测试 helper 的默认注入，模拟真实旧配置缺键。
            with open(os.path.join(d, "config_used.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "lr": 0.001,
                        "steps": 10,
                        "variant": v,
                        "seed": 7,
                        "device": "cpu",
                        "tag": "t",
                    },
                    f,
                )
            _write_json(
                os.path.join(d, "eval.json"),
                {
                    "variant": v,
                    "seed": 7,
                    "data_fingerprint": "abc123",
                    "code_commit": "deadbeef",
                    "config_sha256": "shax",
                    "steps": 10,
                    "tokens_seen": 1024,
                },
            )
        return tmp

    tmp = fixture(["standard", "lega"])
    try:
        try:
            verify_controlled_setup(
                tmp, ["standard", "lega"], [7], 1, expected_data_digest="abc123"
            )
            raise AssertionError("缺键默认 True 导致的混合后端应触发 SystemExit")
        except SystemExit:
            pass
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    # 两个都缺键但都含集中度项：默认 True 不影响它们（仍需显式 softmax），
    # 实际后端一致为 explicit；审计通过并显式记录“推断自默认值”。
    tmp = fixture(["lega", "lega_no_len"])
    try:
        audit = verify_controlled_setup(
            tmp, ["lega", "lega_no_len"], [7], 1, expected_data_digest="abc123"
        )
        assert audit["ok"] and not audit["mixed_attention"], audit
        assert audit["attention_key_seen"] is False
        assert set(audit["attention_backends"].values()) == {"explicit"}
        assert len(audit["attention_missing"]) == 2
        assert len(audit["attention_inferred"]) == 2
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] 缺省注意力开关按默认值推断，混合后端不再漏检")


def test_verify_controlled_setup_rejects_mixed_device_types():
    """不同变体跑在 CPU/CUDA 上时 PPL 差异无法归因于方法，必须拒绝。"""

    def fixture(devices):
        tmp = tempfile.mkdtemp()
        for v, dev in zip(("standard", "lega"), devices):
            d = os.path.join(tmp, v)
            os.makedirs(d)
            _write_json(
                os.path.join(d, "config_used.json"),
                {
                    "lr": 0.001,
                    "steps": 10,
                    "variant": v,
                    "seed": 7,
                    "device": dev,
                    "tag": "t",
                },
            )
            _write_json(
                os.path.join(d, "eval.json"),
                {
                    "variant": v,
                    "seed": 7,
                    "data_fingerprint": "abc123",
                    "code_commit": "deadbeef",
                    "config_sha256": "shax",
                    "steps": 10,
                    "tokens_seen": 1024,
                },
            )
        return tmp

    tmp = fixture(("cpu", "cuda:0"))
    try:
        try:
            verify_controlled_setup(
                tmp, ["standard", "lega"], [7], 1, expected_data_digest="abc123"
            )
            raise AssertionError("混合 CPU/CUDA 应触发 SystemExit")
        except SystemExit:
            pass
        # 同类型不同卡不算混杂（仍会记录原始 device），审计通过
        tmp2 = fixture(("cuda:0", "cuda:1"))
        try:
            audit = verify_controlled_setup(
                tmp2, ["standard", "lega"], [7], 1, expected_data_digest="abc123"
            )
            assert audit["ok"] and not audit["mixed_device"], audit
            assert set(audit["device_types"].values()) == {"cuda"}, audit
        finally:
            shutil.rmtree(tmp2, ignore_errors=True)
        # 旧产物缺 device 只警告，不伪造“一致”
        tmp3 = fixture(("cpu", "cpu"))
        evpath = os.path.join(tmp3, "lega", "config_used.json")
        cfg = json.load(open(evpath, encoding="utf-8"))
        cfg.pop("device")
        _write_json(evpath, cfg)
        try:
            audit = verify_controlled_setup(
                tmp3, ["standard", "lega"], [7], 1, expected_data_digest="abc123"
            )
            assert audit["ok"] and audit["device_missing"] == ["lega/seed7"], audit
        finally:
            shutil.rmtree(tmp3, ignore_errors=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] 受控变量审计（混合设备类型被拒绝，缺 device 只告警）")


def test_stat_counts_finite_only():
    """_stat 的 n 与 mean/std 口径一致：NaN/Inf 不参与也不虚增样本量。"""
    st = _stat([1.0, float("nan"), 3.0, None])
    assert st["n"] == 2 and st["mean"] == 2.0
    assert abs(st["std"] - 2**0.5) < 1e-12
    assert _stat([float("nan"), float("inf"), None]) is None
    assert _stat([2.5]) == {"mean": 2.5, "std": None, "n": 1}
    print("  [ok] _stat（NaN 不计入 n，与 mean/std 口径一致）")


def test_single_factor_audit_covers_implementation_factors():
    """实现层因子核对：扰动门控参数真正生效的因子集必须等于声明集合。

    ``GATE_FACTORS``/``RAW_GATE_FACTORS`` 从实现的 ``_USE_*`` 开关派生，
    所以“改前向忘了改开关”会让元数据与实现一起失真：某个变体名义上只有
    A+B，前向却仍乘 ``b·log n``，单因素对比在数据上悄悄变成多因子对比。
    这里核对 (1) 每个带门变体都出现且与声明一致，(2) 探针漂移会被判
    ``ok=False`` 并给出可读原因，(3) 缺参数/置错通道/漏登记同样显形。
    """
    import lega.attention as attention_mod

    audit = single_factor_audit()
    assert audit["ok"] is True, audit["problems"]
    rows = {r["variant"]: r for r in audit["implementation_factors"]}
    assert len(rows) == len(attention_mod._GATED), rows
    for row in audit["implementation_factors"]:
        assert row["ok"] is True, row
        assert row["effective"] == row["declared"], row
    assert rows["lega"]["effective"] == ["A", "B", "C"]
    assert rows["lega_conc_only"]["effective"] == ["B"]
    # 容量匹配对照：参数全保留，只有自己那条证据通道失效
    for component, variant in (
        ("A", "lega_null_query"),
        ("B", "lega_null_conc"),
        ("C", "lega_null_len"),
    ):
        row = rows[variant]
        assert set(row["params"]) == {"gate_w", "gate_a", "gate_b"}, row
        assert row["effective"] == sorted(set("ABC") - {component}), row
        assert row["null_component"] == component, row

    real_probe = attention_mod.probe_gate_factors

    def drifted(variant, **kw):
        probe = real_probe(variant, **kw)
        if variant == "lega_conc_only":
            probe["effective"] = ["B", "C"]
        return probe

    attention_mod.probe_gate_factors = drifted
    try:
        drifted_audit = single_factor_audit()
    finally:
        attention_mod.probe_gate_factors = real_probe
    assert drifted_audit["ok"] is False
    assert any("实现层生效因子" in p for p in drifted_audit["problems"]), drifted_audit[
        "problems"
    ]

    def broken(variant, **kw):
        return {
            "variant": variant,
            "gated": True,
            "params": ["gate_w"],
            "effective": ["A"],
            "bias_effective": False,
            "error": None,
            "null_component": None,
            "use_raw_conc": False,
            "conc_mode": "max",
        }

    attention_mod.probe_gate_factors = broken
    try:
        unregistered = single_factor_audit()
    finally:
        attention_mod.probe_gate_factors = real_probe
    assert unregistered["ok"] is False
    assert any(
        "置零组件" in p or "参数" in p or "生效因子" in p
        for p in unregistered["problems"]
    ), unregistered["problems"]
    print("  [ok] 单因素审计覆盖实现层因子（声明漂移/参数缺失/漏登记均显形）")


def test_single_factor_audit_covers_registered_ablation_hypotheses():
    """登记表里自称"核心消融"的假设必须真的只改变一个门控因子。

    H2–H4 是对外声明的单组件确认性主张。若登记表被改成多因子对比
    （候选换成 B-only 与 Full 比），或优效性 direction 与该边的
    add/remove 语义写反，汇总仍会输出一个看似单组件的 支持/拒绝。
    整体方法对比（vs standard、vs qgate）与机制单样本族必须被归为
    overall/multi_factor 而不误报。
    """
    import copy
    import json
    import lega.hypotheses as hypotheses_mod

    audit = single_factor_audit()
    assert audit["ok"] is True, audit["problems"]
    rows = {
        r["id"]: r
        for r in audit["registered_single_factor"]
        if isinstance(r, dict) and r.get("id")
    }
    assert {"H1", "H2", "H3", "H4", "H5", "H6", "H7"} <= set(rows), rows
    for hyp_id, factor in (("H2", "A"), ("H3", "B"), ("H4", "C")):
        assert rows[hyp_id]["claims_single_factor"] is True, rows[hyp_id]
        assert rows[hyp_id]["contrast_type"] == "single_factor", rows[hyp_id]
        assert rows[hyp_id]["factor"] == factor, rows[hyp_id]
        assert rows[hyp_id]["direction_mismatch"] is False, rows[hyp_id]
    # 整体/多因子对比不得被当成单组件消融，也不得被误报为问题
    assert rows["H1"]["contrast_type"] == "overall"
    assert rows["H5"]["contrast_type"] == "multi_factor"
    assert rows["H5"]["changed_factors"] == ["B", "C"], rows["H5"]
    assert rows["H6"]["contrast_type"] == "overall"
    assert not any(
        "H1" in p or "H5" in p or "H6" in p for p in audit["problems"]
    ), audit["problems"]

    registry = json.load(open(hypotheses_mod.REGISTRY_PATH, encoding="utf-8"))
    drifted = copy.deepcopy(registry)
    by_id = {h["id"]: h for h in drifted["hypotheses"]}
    by_id["H3"]["candidate"] = "lega_conc_only"  # 变成同时改 A+C
    by_id["H4"]["direction"] = "candidate_better"  # 与 remove 边写反
    real_loader = hypotheses_mod.load_registration
    hypotheses_mod.load_registration = lambda path=None: drifted
    try:
        bad = single_factor_audit()
    finally:
        hypotheses_mod.load_registration = real_loader
    assert bad["ok"] is False
    joined = "；".join(bad["problems"])
    assert "H3" in joined and "多个因子" in joined, joined
    assert "H4" in joined and "direction" in joined, joined
    print("  [ok] 单因素审计覆盖登记表消融假设（多因子/方向写反均显形）")


def test_single_factor_audit_covers_lesion_implementation():
    """推理期病灶必须在实现层只动它声明的证据通道。

    训练期变体已有实现层探针（``probe_gate_factors``）；推理期病灶此前只有
    元数据核对，元数据与实现一起漂移时（例如把 ``len_permute`` 写成同时
    错位 ``c`` 里的 log n），单因素推理期实验会悄悄变成多因素实验而审计
    全绿。这里核对 (1) 每个病灶都实测且与声明一致，(2) 实测漂移会被判
    ``ok=False`` 并给出可读原因，(3) 探针失败同样显形而不静默跳过。
    """
    audit = single_factor_audit()
    assert audit["ok"] is True, audit["problems"]
    rows = {r["lesion"]: r for r in audit["inference_lesion_implementation"]}
    assert set(rows) == set(ABLATION_LESION_ORDER), rows
    for row in audit["inference_lesion_implementation"]:
        assert row["ok"] is True, row
        assert row["affected"] == row["expected"], row
        assert row["applied"] is True, row
    assert rows["no_query"]["expected"] == ["A"]
    assert rows["no_conc_matched"]["expected"] == ["B"]
    assert rows["len_permute"]["expected"] == ["C"]
    assert rows["query_only"]["expected"] == ["B", "C"]

    def drifted(lesion_id, **kwargs):
        probe = {
            "lesion": lesion_id,
            "variant": "lega",
            "applied": True,
            "reason": "",
            "affected": ["A", "B"],
            "expected": ["A"],
            "no_op": False,
            "error": None,
        }
        if lesion_id != "no_query":
            real = probe_lesion_channels(lesion_id)
            probe["affected"] = real["affected"]
            probe["expected"] = real["expected"]
        return probe

    with mock.patch("lega.ablate.probe_lesion_channels", drifted):
        bad = single_factor_audit()
    assert bad["ok"] is False
    assert any(
        "no_query" in p and "实现层实际影响通道" in p for p in bad["problems"]
    ), bad["problems"]

    def broken(lesion_id, **kwargs):
        return {
            "lesion": lesion_id,
            "affected": [],
            "expected": [],
            "applied": False,
            "error": "RuntimeError: 探针不可用",
        }

    with mock.patch("lega.ablate.probe_lesion_channels", broken):
        failed = single_factor_audit()
    assert failed["ok"] is False
    assert any(
        "实现层核对失败" in p for p in failed["problems"]
    ), failed["problems"]
    print("  [ok] 单因素审计覆盖推理期病灶实现层（漂移/探针失败均显形）")


def test_registered_multi_factor_contrast_is_blocked_without_wording_hint():
    """登记成确认性主张的网格内多因子对比必须被拦截（不看 claim 措辞）。

    B-only（`lega_conc_only`）与 Full 同时改变 A+C 两个因子：它既不是
    整体方法对比，也不能当单组件证据。历史实现只在 claim 里出现
    “核心消融/去一法/单组件/单因素”等措辞时才报错——换个措辞就能把
    多因子对比登记成确认性主张，并拿到一个看起来像单组件结论的
    “支持/拒绝”。这里核对：

    1. 审计按因子差本身（而非中文措辞）显式失败并点名该对比；
    2. 该对比不进入汇总配对表（否则会进入 Holm 家族并被当成证据）；
    3. 对应假设判为“不可判定”，确认性资格被撤销；
    4. 显式批准的整体对比（LEGA vs qgate，预注册 H5）不受影响。
    """
    import copy
    import json
    import lega.hypotheses as hypotheses_mod
    from lega.attention import RUNNABLE_VARIANTS
    from run_all import (
        apply_single_factor_gate,
        evaluate_registered_hypotheses,
    )

    registry = json.load(open(hypotheses_mod.REGISTRY_PATH, encoding="utf-8"))
    drifted = copy.deepcopy(registry)
    by_id = {h["id"]: h for h in drifted["hypotheses"]}
    # 措辞刻意不含“核心消融/去一法/单组件/单因素”等标记词：拦截必须
    # 来自因子差本身。
    by_id["H3"]["candidate"] = "lega_conc_only"
    by_id["H3"]["claim"] = "LEGA 与仅集中度门控变体的 PPL 差异"
    real_loader = hypotheses_mod.load_registration
    hypotheses_mod.load_registration = lambda path=None: drifted
    try:
        audit = single_factor_audit()
    finally:
        hypotheses_mod.load_registration = real_loader

    assert audit["ok"] is False, audit["problems"]
    assert any(
        "H3" in p and "多因子" in p for p in audit["problems"]
    ), audit["problems"]
    rows = {
        r["id"]: r for r in audit["registered_single_factor"] if r.get("id")
    }
    assert rows["H3"]["contrast_type"] == "multi_factor_forbidden", rows["H3"]
    assert rows["H3"]["forbidden"] is True, rows["H3"]
    assert rows["H3"]["changed_factors"] == ["A", "C"], rows["H3"]
    forbidden = audit["forbidden_multi_factor_contrasts"]
    assert any(r.get("id") == "H3" for r in forbidden), forbidden

    # 配对表不得包含该对比：多因子差异不能进入同指标 Holm 家族。
    runs = {v: {1: {}} for v in RUNNABLE_VARIANTS}
    pairs = build_comparison_pairs(runs, drifted)
    assert ("lega_conc_only", "lega") not in pairs, pairs
    assert ("lega_conc_only", "lega") not in build_comparison_pairs(
        runs, {"hypotheses": []}
    ), pairs
    # 显式批准的整体对比（LEGA vs qgate）仍然允许进入配对表。
    assert ("lega", "qgate") in pairs, pairs
    assert rows["H5"]["contrast_type"] == "multi_factor", rows["H5"]
    assert rows["H5"]["forbidden"] is False, rows["H5"]

    # 判定层：没有审计时这条主张会拿到“支持/拒绝”；有审计时必须
    # 变成“不可判定”，并撤销确认性资格。
    tests = [
        {**_mk("lega_conc_only", "lega", "PPL", -0.5, 0.001, 5), "p_adj": 0.01},
        {**_mk("lega", "standard", "PPL", -0.4, 0.002, 5), "p_adj": 0.01},
    ]
    naive = {h["id"]: h for h in evaluate_registered_hypotheses(drifted, tests)}
    assert naive["H3"]["status"] in ("支持", "拒绝"), naive["H3"]
    ev = evaluate_registered_hypotheses(drifted, tests, audit=audit)
    by_hyp = {h["id"]: h for h in ev}
    assert by_hyp["H3"]["status"] == "不可判定", by_hyp["H3"]
    assert "不是单因素" in by_hyp["H3"]["detail"], by_hyp["H3"]
    gated = {h["id"]: h for h in apply_single_factor_gate(ev, audit)}
    assert gated["H3"]["confirmatory_valid"] is False, gated["H3"]
    assert gated["H3"]["confirmatory_note"], gated["H3"]
    print("  [ok] 登记为确认性主张的网格内多因子对比被拦截（不看措辞）")
