"""对比/消融协议测试：Holm 校正结论、组件有效性判定与受控变量审计。

覆盖四条关键科研口径：
  * 多重比较校正后结论才有效（避免 8 变体两两对比下的假阳性）；
  * “各组件均有用”必须有显著统计证据，组合跑全本身不算证据；
  * 小样本下 t 检验结论必须与置换检验互证，冲突要显式暴露而不是静默采信；
  * 受控变量审计除超参数外还核对数据指纹/代码版本/训练步数。
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from unittest import mock

from lega.data import CACHE_SCHEMA
from run_all import (ALPHA, AP_DIFF_METRIC, MIN_SEEDS, _ap_diff_tests,
                      _apply_holm, _auroc_diff_tests, _cmd_env, _effect_status,
                      _enrich_paired, _eval_matches_window_spec,
                      _factorial_interaction_tests, _FACTORIAL_CONTRAST_CELLS,
                      _lat_noise_pct,
                      _neural_eval_window_spec, _noise_floor_pct,
                      _paired_comparisons, _ppl_ci_rel_pct, _ppl_of,
                      _probe_oof, _rel_pct,
                      _render_baseline_sections, _result_files,
                      _ruler_avg_micro, _ruler_avg_of, _ruler_min_cell_n,
                      _ruler_task_means,
                      _render_registration_integrity, _resolve_jobs, _stat,
                      _seed_key,
                      apply_budget_gate, apply_integrity_gate, apply_scope_gate,
                      AUROC_DIFF_METRIC, AUROC_DIFF_REFERENCE,
                      build_summary,
                      component_evidence, cross_variant_badcase_map,
                      data_protocol_preflight, data_report_binding_issue,
                      evaluate_registered_hypotheses,
                      load_registered_hypotheses, ngram_baseline_comparison,
                      parse_variants, registration_scope_check,
                      record_aborted_sanity,
                      resolve_ngram_split, run_many,
                      tradeoff_verdict, verify_controlled_setup)


def _mk(cand, ref, metric, mean_diff, p, n=5, lower=True):
    return {"candidate": cand, "reference": ref, "metric": metric,
            "mean_diff": mean_diff, "std_diff": 0.1, "p": p, "p_adj": None,
            "n": n, "lower_is_better": lower, "d_z": 1.0}


def _write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)


def test_data_protocol_preflight_blocks_critical_violations():
    """已知群组/时序泄露的数据默认不得进入训练；显式覆盖才放行。"""
    with tempfile.TemporaryDirectory() as d:
        _write_json(os.path.join(d, "data_report.json"), {
            "alerts": [
                {"level": "warning", "code": "split_shuffled", "message": "打乱"},
                {"level": "warning", "code": "ngram_overlap_train_val",
                 "message": "重叠"},
            ]})
        try:
            data_protocol_preflight(d)
            raise AssertionError("严重协议违规应中止训练")
        except SystemExit as e:
            assert "数据协议前置检查失败" in str(e)
            assert "split_shuffled" in str(e)
        info = data_protocol_preflight(d, allow_override=True)
        assert info["override"] is True
        assert [a["code"] for a in info["critical_violations"]] == \
            ["split_shuffled"]
    with tempfile.TemporaryDirectory() as d:
        info = data_protocol_preflight(d)
        assert info["report_present"] is False
        assert info["critical_violations"] == []
    print("  [ok] 数据协议闸门：严重违规默认中止，显式覆盖放行，报告缺失仅告警")


def test_data_report_binding_detects_tampering():
    with tempfile.TemporaryDirectory() as d:
        rp = os.path.join(d, "data_report.json")
        _write_json(rp, {"alerts": []})
        with open(rp, "rb") as f:
            digest = hashlib.sha256(f.read()).hexdigest()
        _write_json(os.path.join(d, "meta.json"), {"report_sha256": digest})
        assert data_report_binding_issue(d) is None
        _write_json(rp, {"alerts": [{"code": "split_shuffled"}]})
        issue = data_report_binding_issue(d)
        assert issue and "哈希不一致" in issue
        info = data_protocol_preflight(d, allow_override=True)
        assert info["binding_issue"] == issue and info["override"] is True
        assert any(a["code"] == "data_report_binding"
                   for a in info["critical_violations"])
    print("  [ok] data_report 与 meta 哈希绑定：篡改/过期报告被阻断")


def test_data_protocol_preflight_blocks_stale_cache_schema():
    """旧版清洗/去污染管线生成的 bin 不得静默进入训练。"""
    with tempfile.TemporaryDirectory() as d:
        meta = os.path.join(d, "meta.json")
        _write_json(meta, {"cache_schema": CACHE_SCHEMA - 1})
        try:
            data_protocol_preflight(d)
            raise AssertionError("旧缓存协议应中止训练")
        except SystemExit as e:
            assert "data_cache_schema" in str(e)
            assert "重建" in str(e)
        info = data_protocol_preflight(d, allow_override=True)
        assert any(a["code"] == "data_cache_schema"
                   for a in info["critical_violations"])
        assert info["cache_schema_issue"]
        assert "不一致" in info["cache_schema_issue"]
        _write_json(meta, {"cache_schema": str(CACHE_SCHEMA)})
        info = data_protocol_preflight(d)
        assert info["cache_schema_issue"] is None
        assert info["critical_violations"] == []
    with tempfile.TemporaryDirectory() as d:
        _write_json(os.path.join(d, "meta.json"), {"cache_schema": "bad"})
        info = data_protocol_preflight(d, allow_override=True)
        issue = info["cache_schema_issue"]
        assert issue and "cache_schema=-1" in issue
    print("  [ok] 数据缓存协议版本：旧版/损坏 schema 默认阻断，重建后放行")


def test_resolve_jobs_defaults():
    class _A:
        jobs = None
        wandb = False
        device = "cpu"

    # 小核（<=4）默认 cpu+1；大核维持 4 并发上限（见 _resolve_jobs）
    cpu = os.cpu_count() or 1
    assert _resolve_jobs(_A()) == (cpu + 1 if cpu <= 4 else min(4, cpu))
    _A.wandb = True
    assert _resolve_jobs(_A()) == 1
    _A.wandb = False
    _A.device = "cuda"
    assert _resolve_jobs(_A()) == 1
    _A.device = "cpu"
    _A.jobs = 7
    assert _resolve_jobs(_A()) == 7
    print("  [ok] _resolve_jobs（显式 jobs / CUDA / wandb 默认值）")


def test_parse_variants_dedups_preserving_order():
    assert parse_variants("lega") == ["lega"]
    assert parse_variants(" lega , standard ,lega,, leafless ") == \
        ["lega", "standard", "leafless"]
    # 重复变体在并发阶段会生成重复命令并写同一输出路径，必须按首次出现去重
    assert parse_variants("standard,standard,standard") == ["standard"]
    assert parse_variants(", ,") == []
    print("  [ok] parse_variants：保序去重、忽略空白项")


def test_run_many_sequential_parallel_and_failure():
    ok = [sys.executable, "-c", "pass"]
    run_many([ok], 1)
    run_many([ok, ok], 2)
    bad = [sys.executable, "-c", "import sys; sys.exit(3)"]
    for cmds, jobs in (([ok, bad, ok], 2), ([bad], 1)):
        try:
            run_many(cmds, jobs)
            raise AssertionError("非零退出码应触发 CalledProcessError")
        except subprocess.CalledProcessError as e:
            assert e.returncode == 3
    print("  [ok] run_many（顺序/并发/失败传播）")


def test_cmd_env_sets_pythonhashseed_per_command_seed():
    """每条命令的 PYTHONHASHSEED 必须等于它自己的 --seed，而不是父进程基准值。"""
    e = _cmd_env([sys.executable, "-m", "lega.train", "--seed", "2024"])
    assert e["PYTHONHASHSEED"] == "2024"
    assert e["CUBLAS_WORKSPACE_CONFIG"]  # CUDA 初始化前必须存在
    base = dict(os.environ)
    base["PYTHONHASHSEED"] = "1337"
    assert _cmd_env([sys.executable, "x.py"], base)["PYTHONHASHSEED"] == "1337"
    assert _cmd_env([sys.executable, "x.py", "--seed", "notint"], base
                    )["PYTHONHASHSEED"] == "1337"
    # 真实子进程验证：退出码非 0 即断言失败
    code = ("import os,sys;"
            "assert os.environ['PYTHONHASHSEED']==sys.argv[1], "
            "os.environ['PYTHONHASHSEED']")
    run_many([[sys.executable, "-c", code, "42", "--seed", "42"]], 1)
    print("  [ok] _cmd_env/run_many 逐命令对齐 PYTHONHASHSEED 与 --seed")


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


def test_component_evidence_flags_contradiction():
    """一套消融显著支持、另一套显著反向时必须判冲突，不能用 OR 静默取胜。"""
    tests = [
        _mk("lega_no_query", "lega", "PPL", 0.4, 0.005),   # 去 A 变差：支持
        _mk("qgate", "gate_only", "PPL", 0.3, 0.004),      # 加 A 反而变差：反向
    ]
    _apply_holm(tests)
    a = next(c for c in component_evidence(tests) if c["component"] == "A 查询项")
    assert a["supported"] == []
    assert a["verdict"].startswith("证据冲突")
    # 只有一套显著反向、另一套不显著时仍必须判冲突，不能降格成“证据不足”
    one_sided = [
        _mk("lega_no_query", "lega", "PPL", 0.4, 0.5),      # 去 A 不显著
        _mk("qgate", "gate_only", "PPL", 0.3, 0.005),       # 加 A 显著反向
    ]
    _apply_holm(one_sided)
    a2 = next(c for c in component_evidence(one_sided)
              if c["component"] == "A 查询项")
    assert a2["verdict"].startswith("证据冲突"), a2
    print("  [ok] component_evidence（任一套显著反向 -> 证据冲突，另一套不显著亦然）")


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
    assert tradeoff_verdict(None, None, 10.0,
                            ruler_benefit_pp=None, ruler_p_adj=None) == "数据不足"
    # PPL 显著变差不是“收益”：绝不能得到“可接受”
    v = tradeoff_verdict(benefit_pct=-5.0, p_adj=0.01, cost_pct=1.0)
    assert v.startswith("不可行") and "可接受" not in v, v
    v = tradeoff_verdict(benefit_pct=-0.3, p_adj=0.4, cost_pct=2.0)
    assert v.startswith("说服力有限"), v
    # 种子数不足与全表口径一致：p_adj 再小也不判定
    v = tradeoff_verdict(benefit_pct=5.0, p_adj=0.001, cost_pct=1.0, n_seeds=2)
    assert v.startswith("种子不足"), v
    assert tradeoff_verdict(5.0, 0.01, 1.0, n_seeds=MIN_SEEDS) == "可接受"
    print("  [ok] tradeoff_verdict（不显著/大增幅/收益增幅比/负收益/种子不足）")


def test_tradeoff_verdict_covers_ruler_primary_metric():
    """权衡判定必须覆盖 RULER 主指标：只看 PPL 会误判长上下文涨点，
    也会漏掉 RULER 显著退步。"""
    # PPL 持平（不显著）+ RULER 显著涨点 + 增幅温和 → 有涨点且注明依据
    v = tradeoff_verdict(benefit_pct=0.1, p_adj=0.6, cost_pct=8.0,
                         n_seeds=5, ruler_benefit_pp=1.5, ruler_p_adj=0.01,
                         ruler_n_seeds=5)
    assert v.startswith("可接受") and "RULER" in v, v
    # RULER 单独显著改善 + 增幅超过 10×收益（RULER 独立阈值）→ 比值不利
    v = tradeoff_verdict(benefit_pct=0.1, p_adj=0.6, cost_pct=30.0,
                         n_seeds=5, ruler_benefit_pp=1.0, ruler_p_adj=0.01,
                         ruler_n_seeds=5)
    assert "收益-增幅比不利" in v and "RULER" in v, v
    # RULER 显著改善但 PPL 显著退步 → 权衡，而不是“不可行”
    v = tradeoff_verdict(benefit_pct=-3.0, p_adj=0.01, cost_pct=10.0,
                         n_seeds=5, ruler_benefit_pp=2.0, ruler_p_adj=0.01,
                         ruler_n_seeds=5)
    assert v.startswith("权衡") and "PPL" in v and "RULER" in v \
        and "不可行" not in v, v
    # PPL 显著改善但 RULER 显著退步 → 权衡（核心检索目标受损必须显式）
    v = tradeoff_verdict(benefit_pct=4.0, p_adj=0.01, cost_pct=5.0,
                         n_seeds=5, ruler_benefit_pp=-2.0, ruler_p_adj=0.01,
                         ruler_n_seeds=5)
    assert v.startswith("权衡") and "RULER" in v and "受损" in v, v
    # RULER 显著退步且无 PPL 涨点 → 不可行（不能因 PPL 不动就静默放过）
    v = tradeoff_verdict(benefit_pct=0.0, p_adj=0.6, cost_pct=5.0,
                         n_seeds=5, ruler_benefit_pp=-1.5, ruler_p_adj=0.01,
                         ruler_n_seeds=5)
    assert v.startswith("不可行") and "RULER" in v, v
    # 两个主指标都显著改善 → 依据两者；RULER 侧种子不足同样不能判定
    v = tradeoff_verdict(benefit_pct=3.0, p_adj=0.01, cost_pct=8.0,
                         n_seeds=5, ruler_benefit_pp=1.0, ruler_p_adj=0.01,
                         ruler_n_seeds=5)
    assert v.startswith("可接受") and "PPL+RULER" in v, v
    v = tradeoff_verdict(benefit_pct=3.0, p_adj=0.01, cost_pct=8.0,
                         n_seeds=5, ruler_benefit_pp=1.0, ruler_p_adj=0.01,
                         ruler_n_seeds=2)
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
        out = build_summary(tmp, audit={"ok": True, "checked": 0,
                                        "mismatches": []})
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
    runs = {"lega": {7: {"ruler": {"badcases": []}},
                     99: {"ruler": {"badcases": cases99}}}}
    assert cross_variant_badcase_map(runs, ["lega"]) == {"lega": []}
    # 最小种子没有 RULER 结果 → 退化到最小有结果的种子
    runs2 = {"lega": {1: {}, 5: {"ruler": {"badcases": cases99}}}}
    assert cross_variant_badcase_map(runs2, ["lega"]) == {"lega": cases99}
    # 最小种子的 ruler 产物缺 badcases 键/非列表 → 不是“零错误”，
    # 退化到下一个有合法清单的种子；全无合法清单则排除该变体。
    runs3 = {"lega": {1: {"ruler": {"results": {}}},
                      2: {"ruler": {"badcases": "bad"}},
                      5: {"ruler": {"badcases": cases99}}}}
    assert cross_variant_badcase_map(runs3, ["lega"]) == {"lega": cases99}
    runs4 = {"lega": {1: {"ruler": {"results": {}}},
                      2: {"ruler": {"badcases": "bad"}}}}
    assert cross_variant_badcase_map(runs4, ["lega"]) == {}
    # 完全未跑 RULER → 不注册（无法参与对照）
    assert cross_variant_badcase_map({"lega": {1: {}}}, ["lega"]) == {}
    print("  [ok] cross_variant_badcase_map（最小种子优先，零错误显式注册）")


def test_build_summary_scopes_discovery_to_current_repro_run():
    """results/<tag> 跨多次运行复用时，汇总只纳入 repro.json 记录的本次
    eval/sanity（不同实验的观测不相配成对）。"""
    tmp = tempfile.mkdtemp()
    try:
        _write_json(os.path.join(tmp, "repro.json"),
                    {"variants": ["lega"], "seeds": [1], "smoke": False})
        _write_json(os.path.join(tmp, "eval_lega.json"),
                    {"variant": "lega", "seed": 1, "val_ppl": 9.0,
                     "val_loss": 2.0, "n_params": 10})
        _write_json(os.path.join(tmp, "sanity_lega.json"),
                    {"variant": "lega", "passed": True,
                     "initial": {"loss": 2.5},
                     "final": {"loss": 1.0, "token_acc": 0.4}})
        # 上一轮遗留：旧变体、旧种子、未通过的旧 sanity
        _write_json(os.path.join(tmp, "eval_standard.json"),
                    {"variant": "standard", "seed": 1, "val_ppl": 10.0,
                     "val_loss": 2.3, "n_params": 10})
        _write_json(os.path.join(tmp, "eval_lega_seed99.json"),
                    {"variant": "lega", "seed": 99, "val_ppl": 1.0,
                     "val_loss": 0.1, "n_params": 10})
        _write_json(os.path.join(tmp, "sanity_standard.json"),
                    {"variant": "standard", "passed": False,
                     "initial": {"loss": 9.9},
                     "final": {"loss": 9.9, "token_acc": 0.01}})
        out = build_summary(tmp)
        assert [r["variant"] for r in out["rows"]] == ["lega"]
        md = out["markdown"]
        assert "❌ 未通过" not in md           # 旧 sanity 不得混入
        assert "10.0000" not in md            # 旧 standard 的 PPL 不得混入
        assert "0.1000" not in md             # 旧种子的 val_loss 不得混入
        sig = json.load(open(os.path.join(tmp, "significance.json"),
                             encoding="utf-8"))
        assert sig["pairs"] == []
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] build_summary（repro.json 限定本次运行的变体/种子发现范围）")


def test_rel_pct_and_eff_cells():
    """相对变化计算：缺失与除零安全返回 None。"""
    assert _rel_pct(100.0, 110.0) == 10.0
    assert _rel_pct(50.0, 45.0) == -10.0
    assert _rel_pct(None, 10.0) is None
    assert _rel_pct(10.0, None) is None
    assert _rel_pct(0.0, 10.0) is None
    # _eff_cell 读取 latency 条目的结构与运行时字段
    entry = {"n_params": 100, "efficiency": {
        "n_params": 100, "params_mb": 0.4,
        "flops_per_tok": {"256": {"total_flops_per_tok": 12345}},
        "peak_mem_mb_prefill": {"256": 3.5}}}
    from run_all import _eff_cell
    assert _eff_cell(entry, "params", 256) == 100
    assert _eff_cell(entry, "flops", 256) == 12345
    assert _eff_cell(entry, "mem", 256) == 3.5
    assert _eff_cell(None, "flops", 256) is None
    assert _eff_cell({"efficiency": {}}, "flops", 256) is None
    print("  [ok] _rel_pct/_eff_cell（缺失安全）")


def test_result_files_empty_scope_does_not_fall_back_to_all():
    """repro.seeds=[] 表示本次 run 无种子，不能扫描目录里的旧结果。"""
    tmp = tempfile.mkdtemp()
    try:
        for name in ("eval_standard.json", "eval_lega_seed1.json"):
            _write_json(os.path.join(tmp, name), {"variant": "x"})
        assert _result_files(tmp, "eval", None) == sorted(
            os.path.join(tmp, n) for n in
            ("eval_standard.json", "eval_lega_seed1.json"))
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


def test_verify_controlled_setup_consistency():
    tmp = tempfile.mkdtemp()
    try:
        variants, seeds = ["standard", "lega"], [7]
        for v in variants:
            d = os.path.join(tmp, v)
            os.makedirs(d)
            _write_json(os.path.join(d, "config_used.json"), {
                "lr": 0.001, "steps": 10, "warmup": 2,
                "variant": v, "seed": 7, "device": "cpu", "tag": "t"})
            _write_json(os.path.join(d, "eval.json"), {
                "variant": v, "seed": 7, "data_fingerprint": "abc123",
                "code_commit": "deadbeef", "config_sha256": "shax",
                "steps": 10, "tokens_seen": 1024})
        audit = verify_controlled_setup(tmp, variants, seeds, 1,
                                        expected_data_digest="abc123")
        assert audit["ok"] and audit["checked"] == 2
        assert audit["meta_checked"] == 2 and not audit["meta_mismatches"]

        # 篡改数据指纹：同一 config 跑了不同数据不可比，必须报错
        ev_path = os.path.join(tmp, "lega", "eval.json")
        ev = json.load(open(ev_path, encoding="utf-8"))
        ev["data_fingerprint"] = "different"
        _write_json(ev_path, ev)
        try:
            verify_controlled_setup(tmp, variants, seeds, 1,
                                    expected_data_digest="abc123")
            raise AssertionError("数据指纹不一致应触发 SystemExit")
        except SystemExit:
            pass

        # 训练步数（steps）不一致同样必须报错
        ev["data_fingerprint"] = "abc123"
        ev["steps"] = 12
        _write_json(ev_path, ev)
        try:
            verify_controlled_setup(tmp, variants, seeds, 1,
                                    expected_data_digest="abc123")
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
            verify_controlled_setup(tmp, variants, seeds, 1,
                                    expected_data_digest="abc123")
            raise AssertionError("学习率不一致应触发 SystemExit")
        except SystemExit:
            pass

        # 显式 null seed：审计应如实报告“不一致”
        cfg["lr"] = 0.001
        cfg["seed"] = None
        _write_json(cfg_path, cfg)
        try:
            verify_controlled_setup(tmp, variants, seeds, 1,
                                    expected_data_digest="abc123")
            raise AssertionError("config seed=null 应触发 SystemExit")
        except SystemExit:
            pass
        cfg["seed"] = 7
        _write_json(cfg_path, cfg)
        ev["seed"] = None
        _write_json(ev_path, ev)
        try:
            verify_controlled_setup(tmp, variants, seeds, 1,
                                    expected_data_digest="abc123")
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
        _write_json(os.path.join(d, "config_used.json"),
                    {"lr": 0.001, "variant": "standard", "seed": 1,
                     "device": "cpu", "tag": "t"})
        audit = verify_controlled_setup(tmp2, ["standard"], [1], 1)
        assert audit["ok"] and audit["meta_checked"] == 0
        assert audit["missing_evals"] == ["standard/seed1"]
    finally:
        shutil.rmtree(tmp2, ignore_errors=True)
    print("  [ok] verify_controlled_setup（数据/代码/步数/超参数双层审计）")


def test_verify_controlled_setup_rejects_mixed_attention_backend():
    """standard/legA 开了 SDPA 时实现路径不同；正式对比必须统一为显式路径。"""
    for sdpa, should_fail in ((True, True), (False, False)):
        tmp = tempfile.mkdtemp()
        try:
            variants, seeds = ["standard", "lega"], [7]
            for v in variants:
                d = os.path.join(tmp, v)
                os.makedirs(d)
                _write_json(os.path.join(d, "config_used.json"), {
                    "lr": 0.001, "steps": 10, "warmup": 2,
                    "use_sdpa_standard": sdpa,
                    "variant": v, "seed": 7, "device": "cpu", "tag": "t"})
                _write_json(os.path.join(d, "eval.json"), {
                    "variant": v, "seed": 7, "data_fingerprint": "abc123",
                    "code_commit": "deadbeef", "config_sha256": "shax",
                    "steps": 10, "tokens_seen": 1024})
            if should_fail:
                try:
                    verify_controlled_setup(tmp, variants, seeds, 1,
                                            expected_data_digest="abc123")
                    raise AssertionError("混合 SDPA/显式路径应触发 SystemExit")
                except SystemExit:
                    pass
            else:
                audit = verify_controlled_setup(
                    tmp, variants, seeds, 1, expected_data_digest="abc123")
                assert audit["ok"]
                assert audit["attention_backends"] == {
                    "standard": "explicit", "lega": "explicit"}
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] verify_controlled_setup（混合注意力实现路径被拒绝）")


def test_stat_counts_finite_only():
    """_stat 的 n 与 mean/std 口径一致：NaN/Inf 不参与也不虚增样本量。"""
    st = _stat([1.0, float("nan"), 3.0, None])
    assert st["n"] == 2 and st["mean"] == 2.0
    assert abs(st["std"] - 2 ** 0.5) < 1e-12
    assert _stat([float("nan"), float("inf"), None]) is None
    assert _stat([2.5]) == {"mean": 2.5, "std": None, "n": 1}
    print("  [ok] _stat（NaN 不计入 n，与 mean/std 口径一致）")


def test_build_summary_worst_windows_and_cross_variant():
    """最差窗口归因与跨变体错误对照必须进入汇总页与 JSON，而不是只留在文件里。"""
    tmp = tempfile.mkdtemp()
    try:
        for v, sids in (("standard", ["niah_synth@256#0"]),
                        ("lega", ["niah_synth@256#0", "niah_synth@256#1"])):
            cases = [{"sample_id": sid, "task": "niah_synth", "ctx": 256,
                      "answer": "1234567", "answer_words": None,
                      "pred": "7654321", "em": 0.0, "recall": None,
                      "question": "What is the special magic number?",
                      "needle_excerpt": "magic number is: 1234567.",
                      "depth": 0.5, "meta": {}, "context_numbers": ["7654321"],
                      "answer_in_prompt": True} for sid in sids]
            _write_json(os.path.join(tmp, f"ruler_{v}_seed1.json"),
                        {"variant": v, "seed": 1, "results": {},
                         "badcases": cases})
            _write_json(os.path.join(tmp, f"eval_{v}_seed1.json"), {
                "variant": v, "seed": 1, "val_ppl": 10.0, "val_loss": 2.3,
                "gate_mean": 0.5, "n_params": 100, "n_gate_extra": 0,
                "split": "test",
                "worst_windows": [{"window": 1, "split": "test", "loss": 8.0,
                                   "context_tail": "sky is blue and",
                                   "gold_next": " green",
                                   "pred_top5": [[" blue", 0.3]]}]})
        out = build_summary(tmp)
        assert len(out["rows"]) == 2
        md = out["markdown"]
        assert "跨变体错误对照" in md and "方法特异性" in md
        assert "语言建模最差窗口" in md and "复读吸引子" in md
        cross_path = os.path.join(tmp, "badcase_cross.json")
        assert os.path.exists(cross_path)
        cross = json.load(open(cross_path, encoding="utf-8"))
        # lega 相对 standard 多错一题 = 方法特异性新增；standard 无错误清单时
        # 该样本才是共识失败（这里 standard 也错了 #0）
        assert cross["ruler"]["per_variant"]["lega"]["n_new_vs_reference"] == 1
        assert cross["ruler"]["n_shared_all_failures"] == 1
        # 两个变体的最差窗口是同一个 → 全体共识失败
        assert cross["lm_windows"]["n_shared_all_failures"] == 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] build_summary（最差窗口与跨变体错误对照进入汇总）")


def test_build_summary_worst_windows_falls_back_to_seed_with_data():
    """最小种子没转储最差窗口（未开 --worst_n）时，不得把其他种子的
    窗口一并静默丢弃。"""
    tmp = tempfile.mkdtemp()
    try:
        _write_json(os.path.join(tmp, "repro.json"),
                    {"variants": ["lega"], "seeds": [1, 2], "smoke": False})
        _write_json(os.path.join(tmp, "eval_lega_seed1.json"),
                    {"variant": "lega", "seed": 1, "val_ppl": 10.0,
                     "val_loss": 2.3, "n_params": 10, "split": "test"})
        _write_json(os.path.join(tmp, "eval_lega_seed2.json"), {
            "variant": "lega", "seed": 2, "val_ppl": 10.0, "val_loss": 2.3,
            "n_params": 10, "split": "test",
            "worst_windows": [{"window": 1, "split": "test", "loss": 8.0,
                               "context_tail": "sky is blue and",
                               "gold_next": " green",
                               "pred_top5": [[" blue", 0.3]]}]})
        out = build_summary(tmp)
        assert "最差窗口数" in out["markdown"]
        assert "8.000" in out["markdown"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] build_summary（最差窗口退化到有数据的种子）")


def test_build_summary_worst_windows_aggregate_seeds():
    """逐变体归因必须累计全部种子的最差窗口（同一窗口在多种子中错要重复
    计入以显示错误模式的跨种子稳定性）；若仍只取一个种子，这里会少 2 条。"""
    tmp = tempfile.mkdtemp()
    try:
        _write_json(os.path.join(tmp, "repro.json"),
                    {"variants": ["lega"], "seeds": [1, 2], "smoke": False})
        w1 = {"window": 1, "split": "test", "loss": 8.0,
              "context_tail": "sky is blue and", "gold_next": " green",
              "pred_top5": [[" blue", 0.3]]}          # 复读吸引子
        w2 = {"window": 2, "split": "test", "loss": 7.0,
              "context_tail": "the cat sat on", "gold_next": " the",
              "pred_top5": [[" mat", 0.4]]}           # 低置信
        _write_json(os.path.join(tmp, "eval_lega_seed1.json"), {
            "variant": "lega", "seed": 1, "val_ppl": 10.0, "val_loss": 2.3,
            "n_params": 10, "split": "test", "worst_windows": [w1]})
        _write_json(os.path.join(tmp, "eval_lega_seed2.json"), {
            "variant": "lega", "seed": 2, "val_ppl": 10.0, "val_loss": 2.3,
            "n_params": 10, "split": "test", "worst_windows": [w1, w2]})
        out = build_summary(tmp)
        md = out["markdown"]
        assert "| LEGA(ours) | 3 |" in md, md
        assert "复读吸引子（低置信下首选续写退化为抄上下文已有串）×2" in md, md
        assert "2 个种子" in md, md
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] build_summary（最差窗口跨种子累计、说明口径）")


def test_build_summary_prefers_current_naming_on_slot_collision():
    """同一 (变体, 种子) 同时存在新旧两种文件名时必须取本次命名约定的
    文件：否则在旧多种子目录上重跑单种子，陈旧的 eval_lega_seed<seed>.json
    会经 _seed_key 落到同一槽位并覆盖新写的 eval_lega.json。"""
    tmp = tempfile.mkdtemp()
    try:
        # 单种子运行（约定不带后缀）：新写的无后缀文件必须胜出
        _write_json(os.path.join(tmp, "repro.json"),
                    {"variants": ["lega"], "seeds": [1337], "smoke": False})
        _write_json(os.path.join(tmp, "eval_lega.json"),
                    {"variant": "lega", "seed": 1337, "val_ppl": 7.0,
                     "val_loss": 1.9, "n_params": 10})
        _write_json(os.path.join(tmp, "eval_lega_seed1337.json"),
                    {"variant": "lega", "seed": 1337, "val_ppl": 99.0,
                     "val_loss": 4.6, "n_params": 10})
        out = build_summary(tmp)
        assert [r["val_ppl"] for r in out["rows"]] == [7.0]
        # 多种子运行（约定带后缀）：上一轮单种子的无后缀残留不得覆盖
        os.remove(os.path.join(tmp, "eval_lega.json"))
        _write_json(os.path.join(tmp, "repro.json"),
                    {"variants": ["lega"], "seeds": [1337, 2], "smoke": False})
        _write_json(os.path.join(tmp, "eval_lega.json"),
                    {"variant": "lega", "seed": 1337, "val_ppl": 5.5,
                     "val_loss": 1.7, "n_params": 10})
        out2 = build_summary(tmp)
        assert [r["val_ppl"] for r in out2["rows"]] == [99.0]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] build_summary（同槽冲突按本次命名约定取数）")


def test_build_summary_missing_latency_entry():
    """latency.json 缺少某变体条目（--skip_latency 复用旧结果）时安全跳过该行。"""
    tmp = tempfile.mkdtemp()
    try:
        for v in ("standard", "lega"):
            _write_json(os.path.join(tmp, f"eval_{v}_seed1.json"), {
                "variant": v, "seed": 1, "val_ppl": 10.0, "val_loss": 2.3,
                "gate_mean": 0.5 if v == "lega" else None,
                "n_params": 100, "n_gate_extra": 0})
        # 只有 standard 有运行时基准；lega 无条目（值为 None）
        _write_json(os.path.join(tmp, "latency.json"), {"variants": {"standard": {
            "prefill_ms": {"256": {"1": 1.0}},
            "decode_ms_per_tok": {"256": 0.1},
            "efficiency": {
                "n_params": 100,
                "flops_per_tok": {"256": {"total_flops_per_tok": 1000}},
                "peak_mem_mb_prefill": {"256": 1.0}}}}})
        out = build_summary(tmp)
        assert len(out["rows"]) == 2
        assert "涨点-增幅权衡" in out["markdown"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] build_summary（latency 缺变体条目时安全跳过）")


def test_build_summary_surfaces_loss_spike_diagnosis():
    """瞬态损失尖峰（分段均值会掩盖）必须进入 summary.md 的显著提醒，
    不能只躺在 curve_diagnosis.json 里。"""
    tmp = tempfile.mkdtemp()
    try:
        for v in ("standard", "lega"):
            _write_json(os.path.join(tmp, f"eval_{v}_seed1.json"), {
                "variant": v, "seed": 1, "val_ppl": 10.0, "val_loss": 2.3,
                "gate_mean": None, "n_params": 100, "n_gate_extra": 0})
        base = {"status": "正常收敛", "evidence": [], "recommendations": ["x"],
                "nonfinite_values": 0, "malformed_records": 0,
                "has_grad_spike": False, "has_loss_spike": False,
                "train_loss_first": 4.0, "train_loss_last": 3.0,
                "val_loss_first": 4.1, "val_loss_best": 3.4,
                "val_best_step": 50, "val_loss_last": 3.4,
                "early_stop_gain_rel": 0.0}
        spike = dict(base, has_loss_spike=True,
                     recommendations=["损失尖峰（×9 @step 80）：先核对批次"])
        _write_json(os.path.join(tmp, "curve_diagnosis.json"),
                    {"standard": {"1": base}, "lega": {"1": spike}})
        out = build_summary(tmp)
        md = out["markdown"]
        assert "损失瞬态尖峰" in md, md[-2000:]
        assert "（损失尖峰）" in md
        assert "损失尖峰（×9" in md
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] build_summary（瞬态损失尖峰进入汇总页显著提醒）")


def test_build_summary_surfaces_curve_observability_warnings():
    """验证/日志点过少是配置级限制：必须进入 summary，而不是只留在训练
    启动日志里；否则汇总页的趋势/早停结论会被误读为可靠。"""
    tmp = tempfile.mkdtemp()
    try:
        for v in ("standard", "lega"):
            _write_json(os.path.join(tmp, f"eval_{v}_seed1.json"), {
                "variant": v, "seed": 1, "val_ppl": 10.0, "val_loss": 2.3,
                "gate_mean": None, "n_params": 100, "n_gate_extra": 0})
        base = {"status": "正常收敛", "evidence": [], "recommendations": ["x"],
                "nonfinite_values": 0, "malformed_records": 0,
                "has_grad_spike": False, "has_loss_spike": False,
                "train_loss_first": 4.0, "train_loss_last": 3.0,
                "val_loss_first": 4.1, "val_loss_best": 3.4,
                "val_best_step": 50, "val_loss_last": 3.4,
                "early_stop_gain_rel": 0.0,
                "curve_observability_warnings": [
                    "验证点预计仅 2 个（steps=20, eval_every=10）"]}
        _write_json(os.path.join(tmp, "curve_diagnosis.json"),
                    {"standard": {"1": base}, "lega": {"1": dict(base)}})
        md = build_summary(tmp)["markdown"]
        assert "曲线采样可观测性受限" in md, md[-2000:]
        assert "验证点预计仅 2 个" in md, md[-2000:]
        assert "eval_every" in md, md[-2000:]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] build_summary（曲线采样不足告警进入汇总页）")


def test_latency_noise_helpers():
    """测量噪声读取与两段散布的方和根合成：缺失安全。"""
    entry = {"prefill_ms_spread_pct": {"256": {"1": 6.0}},
             "decode_ms_spread_pct": {"256": 4.0}}
    assert _lat_noise_pct(entry, "prefill", 256) == 6.0
    assert _lat_noise_pct(entry, "decode", 256) == 4.0
    assert _lat_noise_pct(entry, "prefill", 512) is None
    assert _lat_noise_pct({}, "decode", 256) is None
    assert _lat_noise_pct({"decode_ms_spread_pct": {"256": "bad"}},
                          "decode", 256) is None
    assert _noise_floor_pct(3.0, 4.0) == 5.0
    assert _noise_floor_pct(6.0, None) is None
    assert _noise_floor_pct(None, None) is None
    print("  [ok] 运行时测量噪声读取与合成（缺失安全）")


def test_tradeoff_ignores_deltas_within_measurement_noise():
    """运行时差异不超过重复测量噪声时按不可区分处理：不计入增幅、标 †。"""
    tmp = tempfile.mkdtemp()
    try:
        for v in ("standard", "lega"):
            _write_json(os.path.join(tmp, f"eval_{v}_seed1.json"), {
                "variant": v, "seed": 1, "val_ppl": 10.0, "val_loss": 2.3,
                "gate_mean": 0.5 if v == "lega" else None,
                "n_params": 100, "n_gate_extra": 0})

        def _lat(prefill, dec):
            return {"prefill_ms": {"256": {"1": prefill}},
                    "prefill_ms_spread_pct": {"256": {"1": 30.0}},
                    "decode_ms_per_tok": {"256": dec},
                    "decode_ms_spread_pct": {"256": 25.0},
                    "efficiency": {
                        "n_params": 100,
                        "flops_per_tok": {"256": {"total_flops_per_tok": 1000}},
                        "peak_mem_mb_prefill": {"256": 1.0}}}

        # lega 的 prefill/decode 各慢 5%，均远小于噪声方和根（~39%/35%）
        _write_json(os.path.join(tmp, "latency.json"), {"variants": {
            "standard": _lat(1.0, 0.1), "lega": _lat(1.05, 0.105)}})
        out = build_summary(tmp)
        row = next(r for r in out["rows"] if r["variant"] == "lega")
        assert row["eff_prefill_within_noise"] is True
        assert row["eff_decode_within_noise"] is True
        assert row["eff_prefill_noise_pct"] is not None
        # 其余指标一致：噪声内的运行时差归零后最大增幅为 0，而不是 5%
        assert row["eff_cost_max_pct"] == 0.0, row
        assert "†" in out["markdown"] and "噪声区间内" in out["markdown"]

        # 噪声很大但真实差异也大（+300%）时必须照常计入增幅
        _write_json(os.path.join(tmp, "latency.json"), {"variants": {
            "standard": _lat(1.0, 0.1), "lega": _lat(1.0, 0.4)}})
        out2 = build_summary(tmp)
        row2 = next(r for r in out2["rows"] if r["variant"] == "lega")
        assert row2["eff_decode_within_noise"] is False
        assert row2["eff_cost_max_pct"] == 300.0, row2

        # 旧格式无噪声字段：差异全部计入
        _write_json(os.path.join(tmp, "latency.json"), {"variants": {
            "standard": {"prefill_ms": {"256": {"1": 1.0}},
                         "decode_ms_per_tok": {"256": 0.1},
                         "efficiency": {
                             "n_params": 100,
                             "flops_per_tok": {"256": {"total_flops_per_tok": 1000}},
                             "peak_mem_mb_prefill": {"256": 1.0}}},
            "lega": _lat(1.05, 0.105)}})
        out3 = build_summary(tmp)
        row3 = next(r for r in out3["rows"] if r["variant"] == "lega")
        assert row3["eff_prefill_within_noise"] is False
        assert row3["eff_decode_within_noise"] is False
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] 涨点-增幅权衡：噪声内运行时差按不可区分处理（旧格式兼容）")


def test_tradeoff_includes_decode_latency():
    """逐 token 解码指标（decode ms/tok）必须进入涨点-增幅权衡的增幅与判定，不能只看 prefill。"""
    tmp = tempfile.mkdtemp()
    try:
        for v in ("standard", "lega"):
            _write_json(os.path.join(tmp, f"eval_{v}_seed1.json"), {
                "variant": v, "seed": 1, "val_ppl": 10.0, "val_loss": 2.3,
                "gate_mean": 0.5 if v == "lega" else None,
                "n_params": 100, "n_gate_extra": 0})

        def _lat(dec):
            return {"prefill_ms": {"256": {"1": 1.0}},
                    "decode_ms_per_tok": {"256": dec},
                    "efficiency": {
                        "n_params": 100,
                        "flops_per_tok": {"256": {"total_flops_per_tok": 1000}},
                        "peak_mem_mb_prefill": {"256": 1.0}}}

        _write_json(os.path.join(tmp, "latency.json"), {"variants": {
            "standard": _lat(0.1), "lega": _lat(0.4)}})
        out = build_summary(tmp)
        row = next(r for r in out["rows"] if r["variant"] == "lega")
        assert row["eff_decode_ms_per_tok"] == 0.4
        assert row["eff_decode_delta_pct"] == 300.0
        # 其余指标与基线一致时，最大增幅由 decode 项贡献
        assert row["eff_cost_max_pct"] == 300.0, row
        assert "Decode ms/tok" in out["markdown"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] 涨点-增幅权衡包含 decode 项")


def test_tradeoff_includes_kv_cache_memory():
    """KV cache 显存必须进入效率-精度权衡：长上下文内存开销不能只看参数/
    单次激活峰值；旧格式只有 per-token 时按 ctx 换算。"""
    tmp = tempfile.mkdtemp()
    try:
        for v in ("standard", "lega"):
            _write_json(os.path.join(tmp, f"eval_{v}_seed1.json"), {
                "variant": v, "seed": 1, "val_ppl": 10.0, "val_loss": 2.3,
                "gate_mean": 0.5 if v == "lega" else None,
                "n_params": 100, "n_gate_extra": 0})

        def _lat(kv_per_tok, kv_at_ctx=None):
            eff = {"n_params": 100,
                   "flops_per_tok": {"256": {"total_flops_per_tok": 1000}},
                   "peak_mem_mb_prefill": {"256": 1.0},
                   "kv_cache_mb_per_tok": kv_per_tok}
            if kv_at_ctx is not None:
                eff["kv_cache_mb_at_ctx"] = {"256": kv_at_ctx}
            return {"prefill_ms": {"256": {"1": 1.0}},
                    "decode_ms_per_tok": {"256": 0.1},
                    "efficiency": eff}

        _write_json(os.path.join(tmp, "latency.json"), {"variants": {
            "standard": _lat(0.001, kv_at_ctx=0.256),
            "lega": _lat(0.002, kv_at_ctx=0.512)}})
        out = build_summary(tmp)
        row = next(r for r in out["rows"] if r["variant"] == "lega")
        assert abs(row["eff_kv_cache_mb"] - 0.512) < 1e-9, row
        assert row["eff_kv_cache_delta_pct"] == 100.0, row
        assert row["eff_cost_max_pct"] == 100.0, row
        assert "KV cache" in out["markdown"] and "+100.0%" in out["markdown"]

        # 只有 per-token（旧 latency.json）时按公共 ctx 线性换算
        _write_json(os.path.join(tmp, "latency.json"), {"variants": {
            "standard": _lat(0.001), "lega": _lat(0.002)}})
        out2 = build_summary(tmp)
        row2 = next(r for r in out2["rows"] if r["variant"] == "lega")
        assert abs(row2["eff_kv_cache_mb"] - 0.512) < 1e-9, row2
        assert row2["eff_cost_max_pct"] == 100.0, row2
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] 涨点-增幅权衡包含 KV cache 显存（旧格式按 ctx 换算）")


def test_tradeoff_includes_train_step_time():
    """训练 ms/step 必须进入综合增幅与判定：其余指标不变时也要被计入。"""
    tmp = tempfile.mkdtemp()
    try:
        for v, tms in (("standard", 100.0), ("lega", 1000.0)):
            _write_json(os.path.join(tmp, f"eval_{v}_seed1.json"), {
                "variant": v, "seed": 1, "val_ppl": 10.0, "val_loss": 2.3,
                "gate_mean": 0.5 if v == "lega" else None,
                "n_params": 100, "n_gate_extra": 0,
                "train_step_time_ms": tms,
                "train_tokens_per_sec": 1000.0 / tms})

        def _lat():
            return {"prefill_ms": {"256": {"1": 1.0}},
                    "decode_ms_per_tok": {"256": 0.1},
                    "efficiency": {
                        "n_params": 100,
                        "flops_per_tok": {"256": {"total_flops_per_tok": 1000}},
                        "peak_mem_mb_prefill": {"256": 1.0}}}

        _write_json(os.path.join(tmp, "latency.json"), {"variants": {
            "standard": _lat(), "lega": _lat()}})
        out = build_summary(tmp)
        row = next(r for r in out["rows"] if r["variant"] == "lega")
        assert row["eff_train_step_ms"] == 1000.0, row
        assert row["eff_train_delta_pct"] == 900.0
        # 其余指标与基线一致时，最大增幅由训练 ms/step 贡献
        assert row["eff_cost_max_pct"] == 900.0, row
        assert "训练 ms/step" in out["markdown"]
        # 旧结果无该字段时优雅降级为不判定，不崩溃也不编造增幅
        _write_json(os.path.join(tmp, "eval_lega_seed1.json"), {
            "variant": "lega", "seed": 1, "val_ppl": 10.0, "val_loss": 2.3,
            "gate_mean": 0.5, "n_params": 100, "n_gate_extra": 0})
        out2 = build_summary(tmp)
        row2 = next(r for r in out2["rows"] if r["variant"] == "lega")
        assert row2["eff_train_step_ms"] is None
        assert row2["eff_train_delta_pct"] is None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] 涨点-增幅权衡包含训练 ms/step（旧结果缺失时降级）")


def test_tradeoff_includes_train_peak_memory():
    """训练进程峰值必须进入涨点-增幅权衡的增幅项，不能只看
    推理时序/参数量；CPU 训练缺该字段时安全降级。"""
    tmp = tempfile.mkdtemp()
    try:
        for v, mem in (("standard", 1000.0), ("lega", 3000.0)):
            _write_json(os.path.join(tmp, f"eval_{v}_seed1.json"), {
                "variant": v, "seed": 1, "val_ppl": 10.0, "val_loss": 2.3,
                "gate_mean": 0.5 if v == "lega" else None,
                "n_params": 100, "n_gate_extra": 0,
                "train_peak_mem_mb": mem})

        def _lat():
            return {"prefill_ms": {"256": {"1": 1.0}},
                    "decode_ms_per_tok": {"256": 0.1},
                    "efficiency": {
                        "n_params": 100,
                        "flops_per_tok": {"256": {"total_flops_per_tok": 1000}},
                        "peak_mem_mb_prefill": {"256": 1.0}}}

        _write_json(os.path.join(tmp, "latency.json"), {"variants": {
            "standard": _lat(), "lega": _lat()}})
        out = build_summary(tmp)
        row = next(r for r in out["rows"] if r["variant"] == "lega")
        assert row["eff_train_peak_mem_mb"] == 3000.0, row
        assert row["eff_train_mem_delta_pct"] == 200.0
        # 其余指标与基线一致时，最大增幅由训练进程峰值贡献
        assert row["eff_cost_max_pct"] == 200.0, row
        assert "训练峰值" in out["markdown"]
        # CPU 训练（无 CUDA 统计）：字段缺失时该项不参与判定
        _write_json(os.path.join(tmp, "eval_lega_seed1.json"), {
            "variant": "lega", "seed": 1, "val_ppl": 10.0, "val_loss": 2.3,
            "gate_mean": 0.5, "n_params": 100, "n_gate_extra": 0})
        out2 = build_summary(tmp)
        row2 = next(r for r in out2["rows"] if r["variant"] == "lega")
        assert row2["eff_train_peak_mem_mb"] is None
        assert row2["eff_train_mem_delta_pct"] is None
        assert row2["eff_cost_max_pct"] == 0.0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] 涨点-增幅权衡包含训练进程峰值（缺失时降级）")


def test_summary_tolerates_huge_int_train_stats():
    """eval.json 里超出 float 范围的合法整数不能中断汇总。"""
    huge = 10 ** 400
    tmp = tempfile.mkdtemp()
    try:
        _write_json(os.path.join(tmp, "eval_standard_seed1.json"), {
            "variant": "standard", "seed": 1, "val_ppl": 10.0,
            "val_loss": 2.3, "gate_mean": 0.5, "n_params": 100,
            "n_gate_extra": 0, "train_step_time_ms": huge,
            "train_peak_mem_mb": huge})
        out = build_summary(tmp)
        row = next(r for r in out["rows"] if r["variant"] == "standard")
        assert row.get("eff_train_step_ms") is None, row
        assert row.get("eff_train_peak_mem_mb") is None, row
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] 训练 ms/step 与峰值内存的超大整数安全降级")


def test_tradeoff_table_surfaces_ruler_benefit():
    """权衡表必须输出 RULER 收益列；PPL 持平而 RULER 显著提升时判定不能
    仍写“说服力有限”（长上下文检索是本项目预注册的核心主张）。"""
    tmp = tempfile.mkdtemp()
    try:
        seeds = [1, 2, 3]
        _write_json(os.path.join(tmp, "repro.json"),
                    {"variants": ["standard", "lega"], "seeds": seeds,
                     "smoke": False})
        lega_ppls = {1: 9.0, 2: 10.0, 3: 10.9}
        for s in seeds:
            for v in ("standard", "lega"):
                _write_json(os.path.join(tmp, f"eval_{v}_seed{s}.json"), {
                    "variant": v, "seed": s,
                    "val_ppl": 10.0 if v == "standard" else lega_ppls[s],
                    "val_loss": 2.3,
                    "gate_mean": 0.5 if v == "lega" else None,
                    "n_params": 100, "n_gate_extra": 0})
        for s in seeds:
            for v, em in (("standard", 0.40), ("lega", 0.43)):
                _write_json(os.path.join(tmp, f"ruler_{v}_seed{s}.json"), {
                    "variant": v, "seed": s,
                    "results": {"niah_synth": {"128": {"em": em}}}})

        def _lat():
            return {"prefill_ms": {"256": {"1": 1.0}},
                    "decode_ms_per_tok": {"256": 0.1},
                    "efficiency": {
                        "n_params": 100,
                        "flops_per_tok": {"256": {"total_flops_per_tok": 1000}},
                        "peak_mem_mb_prefill": {"256": 1.0}}}

        _write_json(os.path.join(tmp, "latency.json"), {"variants": {
            "standard": _lat(), "lega": _lat()}})
        out = build_summary(tmp)
        row = next(r for r in out["rows"] if r["variant"] == "lega")
        assert row["eff_benefit_ruler_pp"] == 3.0, row
        assert row["eff_ruler_p_adj"] is not None \
            and row["eff_ruler_p_adj"] < ALPHA, row
        assert row["eff_verdict"].startswith("可接受") \
            and "RULER" in row["eff_verdict"], row
        md = out["markdown"]
        assert "RULER Δ(pp)" in md and "+3.00 pp" in md, md[-3000:]
        assert "p_adj(RULER)" in md
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] 涨点-增幅权衡：RULER 收益列与判定依据（PPL 持平不误判）")


def test_audit_tolerates_early_stop_actual_steps():
    """启用早停时实际步数/tokens_seen 可以逐 run 不同（早停的定义），审计按
    计划步数核对并如实报告；未标记早停的步数差异仍是不可比错误。"""
    tmp = tempfile.mkdtemp()
    try:
        variants, seeds = ["standard", "lega"], [7]
        for v in variants:
            d = os.path.join(tmp, v)
            os.makedirs(d)
            _write_json(os.path.join(d, "config_used.json"), {
                "lr": 0.001, "steps": 10, "warmup": 2,
                "early_stop_patience": 2,
                "variant": v, "seed": 7, "device": "cpu", "tag": "t"})
        _write_json(os.path.join(tmp, "standard", "eval.json"), {
            "variant": "standard", "seed": 7, "data_fingerprint": "abc123",
            "code_commit": "deadbeef", "config_sha256": "shax",
            "steps": 10, "steps_planned": 10, "early_stopped": False,
            "tokens_seen": 1024})
        _write_json(os.path.join(tmp, "lega", "eval.json"), {
            "variant": "lega", "seed": 7, "data_fingerprint": "abc123",
            "code_commit": "deadbeef", "config_sha256": "shax",
            "steps": 6, "steps_planned": 10, "early_stopped": True,
            "tokens_seen": 614})
        audit = verify_controlled_setup(tmp, variants, seeds, 1,
                                        expected_data_digest="abc123")
        assert audit["ok"], audit
        assert len(audit["early_stop_runs"]) == 1, audit
        assert "lega/seed7" in audit["early_stop_runs"][0]
        assert audit["training_budget_comparable"] is False, audit

        # 计划步数不一致仍是错误（早停不能掩盖"计划就不同"）
        ev = json.load(open(os.path.join(tmp, "lega", "eval.json"),
                            encoding="utf-8"))
        ev["steps_planned"] = 12
        _write_json(os.path.join(tmp, "lega", "eval.json"), ev)
        try:
            verify_controlled_setup(tmp, variants, seeds, 1)
            raise AssertionError("计划步数不一致应触发 SystemExit")
        except SystemExit:
            pass

        # 未标记早停（early_stopped 缺失/False）的实际步数差异必须报错
        ev["steps_planned"] = 10
        ev["early_stopped"] = False
        _write_json(os.path.join(tmp, "lega", "eval.json"), ev)
        try:
            verify_controlled_setup(tmp, variants, seeds, 1)
            raise AssertionError("非早停的实际步数不一致应触发 SystemExit")
        except SystemExit:
            pass

        # 非早停时 tokens_seen 不一致同样报错
        ev = json.load(open(os.path.join(tmp, "lega", "eval.json"),
                            encoding="utf-8"))
        ev["steps"] = 10
        _write_json(os.path.join(tmp, "lega", "eval.json"), ev)
        try:
            verify_controlled_setup(tmp, variants, seeds, 1)
            raise AssertionError("tokens_seen 不一致应触发 SystemExit")
        except SystemExit:
            pass
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] 受控变量审计：早停按计划步数核对，非早停差异仍报错")


def test_factorial_interaction_contrast_math():
    """2^3 交互对比公式：A×B=2 时 A×C/B×C 应为 0，且标记为无方向检验。"""
    base = {"none": 0.0, "A": 1.0, "B": 2.0, "C": 4.0,
            "AB": 5.0, "AC": 5.0, "BC": 6.0, "ABC": 9.0}
    runs = {}
    for cell, variant in _FACTORIAL_CONTRAST_CELLS.items():
        runs[variant] = {s: {"eval": {"val_ppl": base[cell]}}
                         for s in (1, 2, 3)}
    out = _factorial_interaction_tests(runs)
    assert len(out) == 3, out
    ppl = {t["interaction"]: t for t in out if t["metric"] == "PPL交互"}
    assert len(ppl) == 3, out
    assert abs(ppl["A×B"]["mean_diff"] - 2.0) < 1e-9, ppl["A×B"]
    assert abs(ppl["A×C"]["mean_diff"]) < 1e-9, ppl["A×C"]
    assert abs(ppl["B×C"]["mean_diff"]) < 1e-9, ppl["B×C"]
    assert all(t.get("no_direction") for t in out), out
    incomplete = {k: v for k, v in runs.items() if k != "lega"}
    got = [t for t in _factorial_interaction_tests(incomplete)
           if t["metric"] == "PPL交互"]
    assert got == [], got
    print("  [ok] 全因子交互对比：标准 2^3 对比公式、无方向标记与不完整降级")


def test_factorial_interactions_exclude_protocol_mismatches():
    """交互对比用到全部 8 个格：任一格评测协议/单元不一致，该种子必须剔除；
    否则 split/上下文/RULER 题面差异会伪装成“交互效应”。"""
    base = {"none": 0.0, "A": 1.0, "B": 2.0, "C": 4.0,
            "AB": 5.0, "AC": 5.0, "BC": 6.0, "ABC": 9.0}
    runs = {}
    for cell, variant in _FACTORIAL_CONTRAST_CELLS.items():
        runs[variant] = {s: {"eval": {"val_ppl": base[cell],
                                      "split": "test", "eval_ctx": 512}}
                         for s in (1, 2, 3)}
    # 一个种子下某格的 split 不同：该种子的三条交互都不可用；另两个仍可检验
    runs["lega"][2]["eval"]["split"] = "val"
    out = [t for t in _factorial_interaction_tests(runs)
           if t["metric"] == "PPL交互"]
    assert len(out) == 3, out
    assert all(t["n"] == 2 for t in out), out
    assert all(t.get("excluded_protocol_mismatches") == 1 for t in out), out
    # 所有种子都混入另一协议：不产出交互检验，不能静默给出显著性
    for slot in runs["lega"].values():
        slot["eval"]["split"] = "val"
    assert [t for t in _factorial_interaction_tests(runs)
            if t["metric"] == "PPL交互"] == []
    # RULER 已评测单元集合不一致（缺格）同样不可配对
    for variant, slots in runs.items():
        for slot in slots.values():
            slot["eval"]["split"] = "test"
            slot["ruler"] = {"ctx_lens": [256], "n_samples": 4,
                             "results": {"niah_synth": {
                                 "256": {"em": 1.0, "n": 4}}}}
    for slot in runs["lega"].values():
        slot["ruler"]["results"] = {}
    out = _factorial_interaction_tests(runs)
    assert len([t for t in out if t["metric"] == "PPL交互"]) == 3, out
    assert [t for t in out if t["metric"] == "RULER平均交互"] == [], out
    # 恢复 RULER 单元覆盖后，仅让一个种子下某格 n 不同：该种子的交互剔除
    for variant, slots in runs.items():
        for slot in slots.values():
            slot["ruler"]["results"] = {"niah_synth": {
                "256": {"em": 1.0, "n": 4}}}
    runs["lega"][2]["ruler"]["results"]["niah_synth"]["256"]["n"] = 3
    out = [t for t in _factorial_interaction_tests(runs)
           if t["metric"] == "RULER平均交互"]
    assert len(out) == 3, out
    assert all(t["n"] == 2 for t in out), out
    assert all(t.get("excluded_protocol_mismatches") == 1 for t in out), out
    print("  [ok] 全因子交互剔除评测协议/单元/样本数不一致的种子")


def test_budget_gate_downgrades_confirmatory_on_early_stop():
    """早停导致实际训练量不等：确认性判定降级，已降级的保留原原因。"""
    hyp = [{"id": "H1", "status": "支持", "confirmatory_valid": True},
           {"id": "H2", "status": "未定", "confirmatory_valid": False,
            "confirmatory_note": "原有原因"}]
    out = apply_budget_gate(hyp, {
        "early_stop_runs": ["lega/seed7(step 6/10)"]})
    assert out[0]["confirmatory_valid"] is False, out
    assert "早停" in out[0]["confirmatory_note"], out
    assert out[1]["confirmatory_note"] == "原有原因", out
    unchanged = [{"id": "H1", "confirmatory_valid": True}]
    assert apply_budget_gate(unchanged, {"early_stop_runs": []})[0][
        "confirmatory_valid"] is True
    print("  [ok] 早停门：确认性判定降级为探索性，不覆盖既有降级原因")


def test_build_summary_marks_early_stop_run():
    """早停 run 必须在汇总页被显式标注（曲线按设计截断），且审计小节说明
    实际步数差异的核对口径。"""
    tmp = tempfile.mkdtemp()
    try:
        for v in ("standard", "lega"):
            _write_json(os.path.join(tmp, f"eval_{v}_seed1.json"), {
                "variant": v, "seed": 1, "val_ppl": 10.0, "val_loss": 2.3,
                "gate_mean": None, "n_params": 100, "n_gate_extra": 0})
        base = {"status": "过拟合趋势", "evidence": ["x"], "recommendations": ["y"],
                "nonfinite_values": 0, "malformed_records": 0,
                "has_grad_spike": False, "has_loss_spike": False,
                "train_loss_first": 4.0, "train_loss_last": 3.0,
                "val_loss_first": 4.1, "val_loss_best": 3.4,
                "val_best_step": 50, "val_loss_last": 3.4,
                "early_stop_gain_rel": 0.0, "early_stopped": False,
                "early_stop_step": None}
        stopped = dict(base, early_stopped=True, early_stop_step=500)
        _write_json(os.path.join(tmp, "curve_diagnosis.json"),
                    {"standard": {"1": base}, "lega": {"1": stopped}})
        audit = {"ok": True, "checked": 2, "missing_configs": [],
                 "mismatches": [], "meta_checked": 2, "missing_evals": [],
                 "meta_mismatches": [], "expected_data_digest": "abc123",
                 "exempt_keys": ["variant"], "early_stop_runs": ["lega/seed1(step 500/8000)"]}
        out = build_summary(tmp, audit=audit)
        md = out["markdown"]
        assert "（早停）" in md, md
        assert "1/1 种子需关注" in md, "需关注种子占比必须显式标出"
        assert "早停启用" in md, md
        assert "按计划步数核对" in md, md
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] 汇总页：早停标记 + 审计口径说明")


def test_paired_ppl_skips_seeds_with_mismatched_eval_split():
    """PPL 配对检验不能把 val 与 test 的种子配成一对：split 不同整对剔除。"""
    def ev(ppl, split):
        return {"eval": {"ppl": ppl, "split": split}}

    runs = {
        "lega": {7: ev(10.0, "val"), 8: ev(11.0, "test"),
                 9: ev(12.0, "val")},
        "standard": {7: ev(20.0, "val"), 8: ev(30.0, "val"),
                     9: ev(25.0, "test")},
    }
    out = _paired_comparisons(
        runs, [("lega", "standard")], "PPL",
        lambda slot: _ppl_of(slot.get("eval")), lower_better=True)
    assert len(out) == 1
    t = out[0]
    # 只有 seed7 的双方同为 val；seed8/9 的 split 错位被剔除
    assert t["n"] == 1
    assert abs(t["mean_diff"] - (10.0 - 20.0)) < 1e-12
    assert t.get("excluded_split_mismatches") == 2
    print("  [ok] PPL 配对：跨 eval split 的种子整对剔除")


def test_paired_ppl_skips_eval_protocol_mismatch():
    """同 split 不同 ctx/窗口数仍是不同 token 子集，不能充当配对观测。"""
    def ev(ppl, ctx=512, windows=80):
        return {"eval": {"ppl": ppl, "split": "val", "eval_ctx": ctx,
                         "val_windows": windows}}

    runs = {
        "lega": {7: ev(10.0), 8: ev(11.0, ctx=256), 9: ev(12.0, windows=40)},
        "standard": {7: ev(20.0), 8: ev(30.0), 9: ev(25.0)},
    }
    out = _paired_comparisons(
        runs, [("lega", "standard")], "PPL",
        lambda slot: _ppl_of(slot.get("eval")), lower_better=True)
    assert len(out) == 1, out
    assert out[0]["n"] == 1, out
    assert out[0].get("excluded_protocol_mismatches") == 2, out
    print("  [ok] PPL 配对：同 split 但 ctx/窗口数不一致的种子整对剔除")


def test_paired_ruler_skips_protocol_and_cell_mismatch():
    """RULER 配对不能跨题面/抽样/生成预算或不同评测格集合比较 EM。"""
    def rul(em, **kw):
        base = {
            "ctx_lens": [256, 512], "n_samples": 30, "sample_seed": 1234,
            "corpus_source": "val", "max_new": 12, "fwe_max_new": 44,
            "vt_chains": 4, "vt_depth": 4,
            "results": {"niah": {"256": {"em": em, "n": 30},
                                 "512": {"em": em, "n": 30}}}}
        base.update(kw)
        return {"ruler": base}

    runs = {
        "lega": {7: rul(0.8), 8: rul(0.9, n_samples=20), 9: rul(0.7)},
        "standard": {7: rul(0.5), 8: rul(0.5), 9: rul(0.5, max_new=44)},
    }
    out = _paired_comparisons(
        runs, [("lega", "standard")], "RULER平均",
        lambda slot: _ruler_avg_of(slot.get("ruler")), lower_better=False)
    assert len(out) == 1, out
    t = out[0]
    assert t["n"] == 1, t
    assert abs(t["mean_diff"] - 0.3) < 1e-12, t
    assert t.get("excluded_protocol_mismatches") == 2, t
    # 只覆盖一个上下文格的评测同样不可与两格 macro 平均配对
    runs2 = {
        "lega": {7: rul(0.8)},
        "standard": {7: rul(0.5, results={
            "niah": {"256": {"em": 0.5, "n": 30}}})},
    }
    out2 = _paired_comparisons(
        runs2, [("lega", "standard")], "RULER平均",
        lambda slot: _ruler_avg_of(slot.get("ruler")), lower_better=False)
    assert out2 == [], out2
    # 同一评测格但样本数不同：聚合 EM 不是同一批观测，不能配对
    runs3 = {
        "lega": {7: rul(0.8)},
        "standard": {7: rul(0.5, results={
            "niah": {"256": {"em": 0.5, "n": 15},
                     "512": {"em": 0.5, "n": 30}}})},
    }
    out3 = _paired_comparisons(
        runs3, [("lega", "standard")], "RULER平均",
        lambda slot: _ruler_avg_of(slot.get("ruler")), lower_better=False)
    assert out3 == [], out3
    print("  [ok] RULER 配对：协议/评测格/单元样本数不一致的种子整对剔除")


def test_build_summary_marks_clamped_ppl_and_tolerates_null_distribution():
    """PPL 截断必须在汇总表显式标注；分布漂移字段为 null（空 split）时不得崩溃。"""
    tmp = tempfile.mkdtemp()
    try:
        _write_json(os.path.join(tmp, "eval_lega_seed1.json"), {
            "variant": "lega", "seed": 1, "val_ppl": 4.85e8, "val_loss": 21.0,
            "val_ppl_clamped": True, "gate_mean": 0.5, "n_params": 100,
            "n_gate_extra": 0})
        _write_json(os.path.join(tmp, "eval_standard_seed1.json"), {
            "variant": "standard", "seed": 1, "val_ppl": 10.0, "val_loss": 2.3,
            "gate_mean": None, "n_params": 100, "n_gate_extra": 0})
        out = build_summary(tmp, data_report={
            "distribution_shift": {
                "val_vs_train": {"tv": None, "js_bits": None},
                "test_vs_train": {"tv": None, "js_bits": None,
                                  "eval_oov_token_rate": None}}})
        md = out["markdown"]
        assert "含截断值" in md
        assert "TV=-" in md and "JS=-" in md
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] build_summary（PPL 截断标注；分布漂移 null 安全）")


def test_build_summary_surfaces_group_crossing_split_protocol():
    """群组跨切点必须渲染进汇总页（元数据行 + 告警），不允许静默通过。"""
    tmp = tempfile.mkdtemp()
    try:
        _write_json(os.path.join(tmp, "eval_lega_seed1.json"), {
            "variant": "lega", "seed": 1, "val_ppl": 10.0, "val_loss": 2.3,
            "gate_mean": 0.5, "n_params": 100, "n_gate_extra": 0})
        out = build_summary(tmp, data_report={
            "split_protocol": {
                "mode": "temporal",
                "order": "temporal:train(earliest)->val->test(latest)",
                "fallback_char_cut": False, "shuffled": False,
                "group_unit": "heading_document", "n_headings": 1,
                "documents_cross_cut": True},
            "alerts": [{"level": "warning",
                        "code": "split_documents_cross_cut",
                        "message": "heading_document 群组跨 split"}]})
        md = out["markdown"]
        assert "群组单位=heading_document" in md
        assert "文档跨切点=True" in md
        assert "split_documents_cross_cut" in md
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] build_summary：群组跨 split 进入汇总页（元数据 + 告警）")


def test_paired_effect_below_seed_noise_is_explicitly_marked():
    """|Δ|<ΔSD 的边缘结果不能只写“不显著”，必须显式标注噪声内。"""
    diffs = [-0.5, 0.7, -0.3, 0.6, 0.4]
    ref = [11.0, 12.0, 11.0, 13.0, 9.0]
    cand = [r + d for r, d in zip(ref, diffs)]
    runs = {
        "lega": {s + 1: {"eval": {"ppl": v, "split": "val"}}
                 for s, v in enumerate(cand)},
        "standard": {s + 1: {"eval": {"ppl": v, "split": "val"}}
                     for s, v in enumerate(ref)},
    }
    out = _apply_holm(_paired_comparisons(
        runs, [("lega", "standard")], "PPL",
        lambda slot: _ppl_of(slot.get("eval")), lower_better=True))
    t = out[0]
    assert t["within_noise"] is True, t
    assert t["effect_to_noise"] < 1.0, t
    assert "噪声" in t["verdict"], t
    print("  [ok] 效应/噪声：|Δ|<ΔSD 的边缘结果显式标注")


def test_enrich_paired_adds_robustness_fields():
    """配对检验必须附带置换 p / bootstrap CI / MDE，三者口径一致可复核。"""
    a = [10.5, 10.6, 10.4, 10.7, 10.55]
    b = [10.0, 10.05, 10.02, 10.03, 10.04]
    res = _enrich_paired({"n": 5, "mean_diff": 0.5, "std_diff": 0.1}, a, b)
    assert res["p_perm"] is not None and 0.0 <= res["p_perm"] <= 1.0
    assert res["perm_exact"] is True          # n=5 <= 16 → 全枚举精确
    lo, hi = res["mean_diff_ci95"]
    assert lo <= res["mean_diff"] <= hi
    assert res["mde"] > 0
    # 无可检效应（差恒 0）→ 置换 p=1，MDE 仍按方差给出
    res0 = _enrich_paired({"n": 3, "mean_diff": 0.0, "std_diff": 0.2},
                           [1, 2, 3], [1, 2, 3])
    assert res0["p_perm"] == 1.0 and res0["mde"] > 0
    # n=1 / 无均值：字段为 None 而不是崩溃
    res1 = _enrich_paired({"n": 0, "mean_diff": None, "std_diff": None}, [], [])
    assert res1["p_perm"] is None and res1["mean_diff_ci95"] is None
    print("  [ok] _enrich_paired（置换 p/CI/MDE 字段与口径）")


def test_apply_holm_permutation_agreement():
    """t 显著但置换不显著必须显式标 ⚠；两套 p 在同家族内分别 Holm 校正。"""
    # p_perm 缺失（旧格式 dict）：按同一口径处理，不崩不标
    tests = [_mk("lega", "standard", "PPL", -0.5, 0.01, 5)]
    _apply_holm(tests)
    assert "显著优于" in tests[0]["verdict"] and "⚠" not in tests[0]["verdict"]
    assert tests[0]["t_perm_agree"] is None
    # t 与置换一致显著：无警示，t_perm_agree=True
    t_ok = _mk("lega", "standard", "PPL", -0.5, 0.001, 5)
    t_ok["p_perm"] = 0.001
    _apply_holm([t_ok])
    assert t_ok["p_perm_adj"] == 0.001
    assert t_ok["t_perm_agree"] is True and "⚠" not in t_ok["verdict"]
    # t 显著（p=0.02，家族 m=1）但置换不显著（p_perm=0.3）→ ⚠ 标注
    t_warn = _mk("lega", "standard", "PPL", -0.3, 0.02, 4)
    t_warn["p_perm"] = 0.3
    _apply_holm([t_warn])
    assert "显著优于" in t_warn["verdict"] and "置换检验不显著" in t_warn["verdict"]
    assert t_warn["t_perm_agree"] is False
    # t 不显著但置换显著 → 边缘证据标注
    t_edge = _mk("qgate", "standard", "PPL", -0.1, 0.09, 5)
    t_edge["p_perm"] = 0.01
    _apply_holm([t_edge])
    assert "不显著" in t_edge["verdict"] and "置换检验显著" in t_edge["verdict"]
    assert t_edge["t_perm_agree"] is False
    # 置换 p 也参与 Holm 家族校正：家族 m=2 时最小 p 乘 2
    fam = [_mk("a", "standard", "PPL", -0.5, 0.01, 5),
           _mk("b", "standard", "PPL", -0.4, 0.05, 5)]
    fam[0]["p_perm"] = 0.01
    fam[1]["p_perm"] = 0.05
    _apply_holm(fam)
    assert abs(fam[0]["p_perm_adj"] - 0.02) < 1e-12
    assert abs(fam[1]["p_perm_adj"] - 0.05) < 1e-12
    # 种子不足：不进入 agree 判定
    t_small = _mk("x", "y", "PPL", 1.0, 0.01, 2)
    t_small["p_perm"] = 0.01
    _apply_holm([t_small])
    assert t_small["t_perm_agree"] is None and "种子不足" in t_small["verdict"]
    print("  [ok] _apply_holm 置换检验一致性与 ⚠ 标注")


def test_auroc_diff_tests_family():
    """机制主张（AUROC logit > 仅查询）必须作为配对差正式检验，而非并排数字。"""
    # 构造 runs：lega 的 logit AUROC 每个种子都比 qterm 高（带种子间波动）；
    # qgate 的差值在 0 附近来回 → 应判不显著
    def _slot(al, aq):
        return {"analysis": {"auroc_logit": al, "auroc_qterm": aq}}
    runs = {
        "lega": {1: _slot(0.80, 0.70), 2: _slot(0.83, 0.72), 3: _slot(0.77, 0.68),
                 4: _slot(0.81, 0.70), 5: _slot(0.79, 0.685)},
        "qgate": {1: _slot(0.70, 0.71), 2: _slot(0.72, 0.71), 3: _slot(0.69, 0.70),
                  4: _slot(0.71, 0.72), 5: _slot(0.70, 0.69)},
        # 无分析数据的变体不产生检验
        "standard": {1: {}},
        # 单侧缺失的种子整对剔除（auroc_qterm=None 不算配对）
        "lega_no_len": {1: _slot(0.8, None), 2: _slot(None, None)},
    }
    tests = _auroc_diff_tests(runs)
    by_v = {t["candidate"]: t for t in tests}
    assert set(by_v) == {"lega", "qgate"}, set(by_v)
    t = by_v["lega"]
    assert t["metric"] == AUROC_DIFF_METRIC
    assert t["reference"] == AUROC_DIFF_REFERENCE
    assert t["lower_is_better"] is False
    assert t["n"] == 5 and t["mean_diff"] > 0
    assert t["p"] < 0.001
    # 差值全为正但量级有波动：精确置换 p 恰为双侧下限 2/2^5=0.0625
    # （n=5 的粒度边界：置换检验到不了 α=0.05，见 _apply_holm 注释）
    assert t["perm_exact"] is True
    assert abs(t["p_perm"] - 2.0 / 32.0) < 1e-12
    # qgate 差值均值为 -0.002（噪声）→ t 检验不显著
    assert abs(by_v["qgate"]["mean_diff"] - (-0.002)) < 1e-12
    assert by_v["qgate"]["p"] > 0.05
    # Holm 校正 + 结论方向（AUROC 越大越好 → 差为正 = 显著优于）
    tests = _apply_holm(tests)
    assert "显著优于" in by_v["lega"]["verdict"]
    # t 显著但置换 p 下限 >α → 必须带 ⚠（显著性依赖正态假设）
    assert "⚠" in by_v["lega"]["verdict"] and by_v["lega"]["t_perm_agree"] is False
    assert "不显著" in by_v["qgate"]["verdict"]
    print("  [ok] _auroc_diff_tests（配对差检验/置换粒度边界/家族校正）")


def test_ap_diff_tests_family_and_registered_metric():
    """PR-AUC 机制族：不平衡主指标口径下的配对差检验（H7 的判定依据）。"""
    def _slot(al, aq):
        return {"analysis": {"auprc_logit": al, "auprc_qterm": aq}}
    runs = {
        "lega": {1: _slot(0.30, 0.10), 2: _slot(0.33, 0.11),
                 3: _slot(0.28, 0.09), 4: _slot(0.31, 0.12),
                 5: _slot(0.29, 0.10)},
        "qgate": {1: _slot(0.20, 0.21), 2: _slot(0.19, 0.20),
                  3: _slot(0.21, 0.20), 4: _slot(0.20, 0.20),
                  5: _slot(0.19, 0.21)},
    }
    tests = _ap_diff_tests(runs)
    by_v = {t["candidate"]: t for t in tests}
    assert set(by_v) == {"lega", "qgate"}
    t = by_v["lega"]
    assert t["metric"] == AP_DIFF_METRIC
    assert t["reference"] == AUROC_DIFF_REFERENCE
    assert t["lower_is_better"] is False
    assert t["n"] == 5 and t["mean_diff"] > 0.15
    tests = _apply_holm(tests)
    assert "显著优于" in by_v["lega"]["verdict"]
    assert "不显著" in by_v["qgate"]["verdict"]
    # H7 必须以该 metric 字符串登记（evaluate_registered_hypotheses 按
    # metric 精确匹配检验表，拼写不一致会让假设永远“未检验”）
    reg = {"hypotheses": [{
        "id": "H7",
        "claim": "机制主张（不平衡口径）：AP(logit) > AP(仅查询)",
        "metric": AP_DIFF_METRIC, "candidate": "lega",
        "reference": AUROC_DIFF_REFERENCE, "direction": "candidate_better"}]}
    ev = evaluate_registered_hypotheses(reg, tests)
    assert ev and ev[0]["status"] == "支持", ev
    print("  [ok] _ap_diff_tests（PR-AUC 配对差、Holm 校正、H7 判定）")


def test_render_baseline_sections_reads_evidence_and_warns():
    """汇总页必须渲染简单基线 PPL 与基线调优证据；缺失时显式警告。"""
    tmp = tempfile.mkdtemp()
    try:
        # 缺证据：必须给出“未运行/未提供”的警告而不是静默
        empty = _render_baseline_sections(tmp)
        text = "\n".join(empty)
        assert "基线合理性" in text
        assert "未运行" in text
        # 提供证据：调优选择、最优/默认差距与 ngram PPL 都要出现
        _write_json(os.path.join(tmp, "baseline_tuning.json"), {
            "variant": "standard", "steps": 100, "n_trials": 3,
            "selected": {"lr": 0.0012, "weight_decay": 0.1,
                         "val_loss_best": 3.1},
            "default_trial": {"lr": 0.0006, "weight_decay": 0.1,
                              "val_loss_best": 3.2},
            "selected_minus_default_val_loss": -0.1,
            "default_is_optimal": False,
        })
        _write_json(os.path.join(tmp, "ngram_baseline.json"), {
            "split": "val", "train_tokens": 1000, "eval_tokens": 100,
            "vocab_size": 50, "ppl": {"unigram": 90.0, "bigram": 45.0},
        })
        _write_json(os.path.join(tmp, "repro.json"), {"smoke": False})
        text2 = "\n".join(_render_baseline_sections(tmp))
        assert "0.0012" in text2 and "3.1000" in text2
        assert "90.000" in text2 and "45.000" in text2
        assert "全流水线" in text2 and "受控变量" in text2
        # 默认组合 trial 失败（default_is_optimal=None）：不得冒充“并非最优”
        _write_json(os.path.join(tmp, "baseline_tuning.json"), {
            "variant": "standard", "n_trials": 2, "selected": {
                "lr": 0.0012, "weight_decay": 0.1, "val_loss_best": 3.1},
            "default_trial": None, "default_is_optimal": None,
            "selected_minus_default_val_loss": None})
        text4 = "\n".join(_render_baseline_sections(tmp))
        assert "无法判定默认是否" in text4
        # 调优缺证据但 ngram 在：仍必须点名“未调优基线”的风险
        os.remove(os.path.join(tmp, "baseline_tuning.json"))
        text3 = "\n".join(_render_baseline_sections(tmp))
        assert "未提供基线调优证据" in text3
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] _render_baseline_sections（证据渲染/缺失警告）")


def test_resolve_ngram_split_auto_matches_neural_eval_rule():
    """auto 必须与 lega.evaluate 的 --split auto 同规则：有非空 test.bin → test。"""
    tmp = tempfile.mkdtemp()
    try:
        assert resolve_ngram_split(tmp, "auto") == "val"
        with open(os.path.join(tmp, "test.bin"), "wb") as f:
            f.write(b"")
        assert resolve_ngram_split(tmp, "auto") == "val"   # 空文件不算
        with open(os.path.join(tmp, "test.bin"), "wb") as f:
            f.write(b"\x01")
        assert resolve_ngram_split(tmp, "auto") == "test"
        assert resolve_ngram_split(tmp, "val") == "val"    # 显式选择透传
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] resolve_ngram_split（auto 与神经评测 split 规则一致）")


def test_ngram_baseline_comparison_same_split_stats_and_mismatch():
    """神经变体 vs 简单基线：同 split 才可比、单样本配对检验 + Holm 判定。"""
    tmp = tempfile.mkdtemp()
    try:
        _write_json(os.path.join(tmp, "ngram_baseline.json"), {
            "split": "test", "ppl": {"unigram": 200.0, "bigram": 100.0},
            "selected_alpha": {"unigram": 0.1, "bigram": 0.1}})
        runs = {}
        for v, base in (("standard", 30.0), ("lega", 20.0)):
            runs[v] = {s: {"eval": {"variant": v, "split": "test",
                                    "ppl": base + s}}
                       for s in (1, 2, 3)}
        out = ngram_baseline_comparison(runs, tmp)
        assert out["split"] == "test" and out["reference_model"] == "bigram"
        assert abs(out["ngram_ppl"] - 100.0) < 1e-12
        rows = {r["variant"]: r for r in out["rows"]}
        assert rows["lega"]["n"] == 3
        assert abs(rows["lega"]["ratio"] - 22.0 / 100.0) < 1e-12
        assert rows["lega"]["verdict"] == "显著优于统计基线"
        assert rows["lega"]["p_adj"] is not None and rows["lega"]["p_adj"] < ALPHA
        # 显著劣于统计基线要被点名（神经模型连 n-gram 都打不过）
        runs_bad = {"lega": {s: {"eval": {"variant": "lega", "split": "test",
                                          "ppl": 120.0 + s}}
                             for s in (1, 2, 3)}}
        out_bad = ngram_baseline_comparison(runs_bad, tmp)
        assert out_bad["rows"][0]["verdict"] == "显著劣于统计基线（严重）"
        # split 错位（val 评测 vs test 基线）不得混入比较，必须显式列出
        runs_mix = {"qgate": {1: {"eval": {"variant": "qgate", "split": "val",
                                            "ppl": 25.0}}}}
        out_mix = ngram_baseline_comparison(runs_mix, tmp)
        assert out_mix["rows"] == []
        assert out_mix["incomparable"][0]["variant"] == "qgate"
        # 部分种子 split 错位：可比种子照常进 rows，但错位种子必须显式列出，
        # 不能因为“还有可比种子”就静默丢弃
        runs_partial = {"qgate": {
            1: {"eval": {"variant": "qgate", "split": "val", "ppl": 25.0}},
            2: {"eval": {"variant": "qgate", "split": "test", "ppl": 24.0}},
            3: {"eval": {"variant": "qgate", "split": "test", "ppl": 26.0}}}}
        out_partial = ngram_baseline_comparison(runs_partial, tmp)
        assert out_partial["rows"][0]["n"] == 2
        assert out_partial["incomparable"] == [
            {"variant": "qgate", "eval_splits": ["val"], "excluded_seeds": [1]}]
        text_partial = "\n".join(
            _render_baseline_sections(tmp, ngram_cmp=out_partial))
        assert "未纳入比较" in text_partial and "[1]" in text_partial
        # 渲染：比较表与错位警告都必须出现在基线小节
        text = "\n".join(_render_baseline_sections(tmp, ngram_cmp=out))
        assert "显著优于统计基线" in text and "比值" in text
        text_mix = "\n".join(_render_baseline_sections(tmp, ngram_cmp=out_mix))
        assert "不一致" in text_mix and "未纳入比较" in text_mix
        # 文件缺失 → 空报告（调用方按“未运行”渲染）
        tmp2 = tempfile.mkdtemp()
        try:
            assert ngram_baseline_comparison(runs, tmp2) == {}
        finally:
            shutil.rmtree(tmp2, ignore_errors=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] ngram_baseline_comparison（同 split 配对、Holm、错位隔离）")


def test_ngram_comparison_uses_val_selected_strongest_baseline():
    """简单基线参照必须取 val 选出的最强模型，而不是固定拿 bigram 当靶子。"""
    tmp = tempfile.mkdtemp()
    try:
        _write_json(os.path.join(tmp, "ngram_baseline.json"), {
            "split": "val",
            "ppl": {"unigram": 200.0, "bigram": 100.0,
                    "interpolated_bigram": 80.0},
            "selected_model": "interpolated_bigram",
            "selected_interpolated": {"lambda": 0.7, "alpha_unigram": 0.1,
                                      "alpha_bigram": 0.1,
                                      "val_mean_nll": 4.0},
            "selected_alpha": {"unigram": 0.1, "bigram": 0.1}})
        runs = {"lega": {s: {"eval": {"variant": "lega", "split": "val",
                                      "ppl": 70.0 + s}}
                         for s in (1, 2, 3)}}
        out = ngram_baseline_comparison(runs, tmp)
        assert out["reference_model"] == "interpolated_bigram"
        assert abs(out["ngram_ppl"] - 80.0) < 1e-12
        assert out["rows"][0]["verdict"] == "显著优于统计基线"
        # 历史产物没有 selected_model：按插值→bigram→unigram 的强度顺序回退
        _write_json(os.path.join(tmp, "ngram_baseline.json"), {
            "split": "val",
            "ppl": {"unigram": 200.0, "bigram": 100.0,
                    "interpolated_bigram": 80.0}})
        out_old = ngram_baseline_comparison(runs, tmp)
        assert out_old["reference_model"] == "interpolated_bigram"
        # 更老产物连插值都没有：仍回退 bigram
        _write_json(os.path.join(tmp, "ngram_baseline.json"), {
            "split": "val", "ppl": {"unigram": 200.0, "bigram": 100.0}})
        out_older = ngram_baseline_comparison(runs, tmp)
        assert out_older["reference_model"] == "bigram"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] ngram_baseline_comparison（val 选出的最强简单基线优先）")


def test_ngram_comparison_prefers_window_matched_reference():
    """同 split 之后还要同 token：窗口规格一致时用逐位置对齐参考并披露。"""
    tmp = tempfile.mkdtemp()
    try:
        spec = {"ctx": 8, "iters": 2, "n_windows": 5, "n_eval": 2,
                "n_tokens": 16}
        _write_json(os.path.join(tmp, "ngram_baseline.json"), {
            "split": "val",
            "ppl": {"unigram": 200.0, "bigram": 100.0},
            "ppl_window_matched": {"unigram": 180.0, "bigram": 90.0,
                                   "n_tokens": 16},
            "window_spec": spec,
            "selected_alpha": {"unigram": 0.1, "bigram": 0.1}})
        matched_eval = {"variant": "lega", "split": "val", "ppl": 20.0 + 0,
                        "eval_ctx": 8, "eval_iters": 2, "val_windows": 5}
        runs = {"lega": {s: {"eval": dict(matched_eval, ppl=20.0 + s)}
                         for s in (1, 2, 3)}}
        out = ngram_baseline_comparison(runs, tmp)
        assert out["reference_source"] == "window_matched"
        assert abs(out["ngram_ppl"] - 90.0) < 1e-12
        assert abs(out["ngram_ppl_full_split"] - 100.0) < 1e-12
        assert out["fallback_reason"] is None
        assert all(r["token_matched"] for r in out["rows"])
        text = "\n".join(_render_baseline_sections(tmp, ngram_cmp=out))
        assert "逐位置对齐" in text
        assert "在神经评测实际使用的窗口目标 token 上" in text
        # 任一 run 的窗口字段与规格不符 → 整体回退全 split 参考并显式说明
        runs_mis = {"lega": {s: {"eval": dict(matched_eval, ppl=20.0 + s,
                                              eval_iters=3)}
                             for s in (1, 2, 3)}}
        out_mis = ngram_baseline_comparison(runs_mis, tmp)
        assert out_mis["reference_source"] == "full_split"
        assert abs(out_mis["ngram_ppl"] - 100.0) < 1e-12
        assert "lega" in (out_mis["fallback_reason"] or "")
        assert not any(r["token_matched"] for r in out_mis["rows"])
        text_mis = "\n".join(_render_baseline_sections(tmp, ngram_cmp=out_mis))
        assert "未使用逐位置对齐参考" in text_mis
        # 回退时必须撤销“这些数字=神经评测实际使用的 token”的宣称
        assert "在神经评测实际使用的窗口目标 token 上" not in text_mis
        assert "回退全 split 参考" in text_mis
        # 历史产物缺窗口规格 → 同样回退且如实说明原因
        tmp2 = tempfile.mkdtemp()
        try:
            _write_json(os.path.join(tmp2, "ngram_baseline.json"), {
                "split": "val", "ppl": {"bigram": 100.0}})
            out_old = ngram_baseline_comparison(runs, tmp2)
            assert out_old["reference_source"] == "full_split"
            assert "window_spec" in (out_old["fallback_reason"] or "")
        finally:
            shutil.rmtree(tmp2, ignore_errors=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] ngram_baseline_comparison（窗口对齐参考优先、回退显式披露）")


def test_neural_eval_window_spec_matches_train_overrides():
    """n-gram 对齐用的窗口规格必须与训练/评测实际口径一致（含冒烟覆盖）。"""
    tiny = {"block_size": 256, "train_ctx": 256, "eval_iters": 20}
    assert _neural_eval_window_spec(tiny, False) == {"ctx": 256, "iters": 20}
    # 冒烟：train.py 的 SMOKE_OVERRIDES（train_ctx=128、eval_iters=2）必须生效
    assert _neural_eval_window_spec(tiny, True) == {"ctx": 128, "iters": 2}
    # train_ctx 被 block_size 截断；默认 eval_iters=50；非法规格返回 None
    assert _neural_eval_window_spec(
        {"block_size": 64, "train_ctx": 256}, False) == {"ctx": 64, "iters": 50}
    assert _neural_eval_window_spec({}, False) is None
    print("  [ok] _neural_eval_window_spec（与 train/evaluate 口径一致）")


def test_registration_scope_check_and_gate():
    """登记表完整还不够：run 必须落在登记范围内，否则确认性判定降级。"""
    tmp = tempfile.mkdtemp()
    try:
        reg_cfg = os.path.join(tmp, "registered.json")
        act_cfg = os.path.join(tmp, "actual.json")
        base = {"d_model": 384, "n_layers": 6, "n_heads": 6, "ffn_mult": 4,
                "block_size": 1024, "steps": 8000, "train_ctx": 1024,
                "lr": 6e-4, "weight_decay": 0.1}
        _write_json(reg_cfg, base)
        # 调优注入的 lr/wd 改变不算越界（协议内），其它字段必须一致
        _write_json(act_cfg, dict(base, lr=3e-4))
        registered = {"scope": {"config": reg_cfg, "seeds": [1, 2, 3],
                                "data": "WikiText-103 (gpt2 BPE)"}}
        repro = {"config_file": act_cfg, "seeds": [1, 2, 4], "smoke": False,
                 "data": {"tokenizer": "gpt2"}}
        sc = registration_scope_check(registered, repro)
        assert sc["checked"] and sc["in_scope"] is True and not sc["problems"]
        assert any("4" in w for w in sc["warnings"])   # 未登记种子显式警告
        hyp = [{"id": "H1", "confirmatory_valid": True}]
        assert apply_scope_gate(hyp, sc)[0]["confirmatory_valid"] is True
        # 训练规模与登记配置不一致（如 tiny 配置冒充正式实验）→ 范围外
        _write_json(act_cfg, dict(base, steps=100))
        sc2 = registration_scope_check(registered, repro)
        assert sc2["in_scope"] is False and sc2["problems"]
        hyp2 = [{"id": "H1", "confirmatory_valid": True}]
        assert apply_scope_gate(hyp2, sc2)[0]["confirmatory_valid"] is False
        _write_json(act_cfg, base)
        # 伪数据（字符分词）冒充 gpt2 BPE → 范围外
        sc3 = registration_scope_check(
            registered, {"config_file": act_cfg, "seeds": [1],
                         "smoke": False, "data": {"tokenizer": "char"}})
        assert sc3["in_scope"] is False
        assert any("分词器" in p for p in sc3["problems"])
        # 冒烟运行 → 范围外
        sc4 = registration_scope_check(
            registered, {"config_file": act_cfg, "seeds": [1],
                         "smoke": True, "data": {"tokenizer": "gpt2"}})
        assert sc4["in_scope"] is False
        # 缺 repro → 不可确认（不能默认通过）→ 降级
        sc5 = registration_scope_check(registered, {})
        assert sc5["checked"] is False and sc5["in_scope"] is None
        hyp5 = [{"id": "H1", "confirmatory_valid": True}]
        assert apply_scope_gate(hyp5, sc5)[0]["confirmatory_valid"] is False
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] registration_scope_check/apply_scope_gate（范围外降级）")


def test_registration_scope_render_matches_gate_and_paths():
    """范围核对渲染必须与实际降级一致，且 config_file 路径解析正确。

    登记表未声明 scope 时 apply_scope_gate 明确不降级，渲染不得声称
    “按不可确认降级”；相对 config_file 必须按项目根解析，不能相对 cwd
    静默核对到另一个同名文件；非字符串 config_file 按不可读处理而非崩溃。
    """
    sc = registration_scope_check({}, {"seeds": [1]})
    hyp = [{"id": "H1", "confirmatory_valid": True}]
    apply_scope_gate(hyp, sc)
    assert hyp[0]["confirmatory_valid"] is True
    text = "\n".join(_render_registration_integrity(
        {"status": "intact", "confirmatory_valid": True}, sc))
    assert "不可确认" not in text and "降级" not in text
    assert "未声明 scope" in text
    # 声明了 scope 却核不动 → 实际降级，渲染必须同样声明降级
    sc2 = registration_scope_check({"scope": {"data": "WikiText-103 (gpt2 BPE)"}}, {})
    hyp2 = [{"id": "H1", "confirmatory_valid": True}]
    apply_scope_gate(hyp2, sc2)
    assert hyp2[0]["confirmatory_valid"] is False
    text2 = "\n".join(_render_registration_integrity(
        {"status": "intact", "confirmatory_valid": True}, sc2))
    assert "不可确认" in text2 and "降级" in text2
    tmp = tempfile.mkdtemp()
    try:
        reg = os.path.join(tmp, "reg.json")
        _write_json(reg, {"d_model": 8})
        _write_json(os.path.join(tmp, "actual.json"), {"d_model": 16})
        sc3 = registration_scope_check({"scope": {"config": reg}},
                                       {"config_file": "actual.json"}, root=tmp)
        assert sc3["problems"] and "d_model" in sc3["problems"][0]
        sc4 = registration_scope_check({"scope": {"config": reg}},
                                       {"config_file": {"x": 1}}, root=tmp)
        assert sc4["warnings"] and sc4["checked"] is False
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] 范围核对渲染/降级一致 + config_file 路径解析")


def test_registered_hypotheses_evaluation():
    """预注册假设判定：支持/拒绝/未定的边界与未登记文件时的提示。"""
    tests = [
        _mk("lega", "standard", "PPL", -0.5, 0.01, 5),
        _mk("lega_no_len", "lega", "PPL", 0.4, 0.02, 5),   # 去掉长度项显著变差
        _mk("qgate", "standard", "PPL", 0.1, 0.5, 5),      # 不显著
        _mk("x", "y", "PPL", 1.0, 0.01, 2),                # 种子不足
    ]
    # 让 p_adj 直接可判（单家族 m=4 Holm：0.01→0.04, 0.02→0.06, ...）
    _apply_holm(tests)
    reg = {"hypotheses": [
        {"id": "H1", "claim": "LEGA 优于 standard（PPL）", "metric": "PPL",
         "candidate": "lega", "reference": "standard",
         "direction": "candidate_better"},
        {"id": "H2", "claim": "长度校准有用（去一法）", "metric": "PPL",
         "candidate": "lega_no_len", "reference": "lega",
         "direction": "candidate_worse"},
        {"id": "H3", "claim": "LEGA 优于 qgate（PPL）", "metric": "PPL",
         "candidate": "lega", "reference": "qgate",   # 未运行 → 未检验
         "direction": "candidate_better"},
        {"id": "H4", "claim": "方向登记反了的假设", "metric": "PPL",
         "candidate": "lega", "reference": "standard",
         "direction": "candidate_worse"},             # 显著但方向相反 → 拒绝
        {"id": "H5", "claim": "种子不足的对比", "metric": "PPL",
         "candidate": "x", "reference": "y",
         "direction": "candidate_better"},
    ]}
    ev = {h["id"]: h for h in evaluate_registered_hypotheses(reg, tests)}
    assert ev["H1"]["status"] == "支持"
    assert ev["H2"]["status"] == "支持"
    assert ev["H3"]["status"] == "未检验"
    assert ev["H4"]["status"] == "拒绝"
    assert ev["H5"]["status"] == "未定" and "种子不足" in ev["H5"]["detail"]
    # 不显著 → 未定，且 detail 提示 MDE（有均值/方差可估时）
    reg2 = {"hypotheses": [
        {"id": "H6", "claim": "qgate 优于 standard", "metric": "PPL",
         "candidate": "qgate", "reference": "standard",
         "direction": "candidate_better"}]}
    ev2 = evaluate_registered_hypotheses(reg2, tests)
    assert ev2[0]["status"] == "未定" and "不显著" in ev2[0]["detail"]
    # 空注册文件：不判定
    assert evaluate_registered_hypotheses({}, tests) == []
    print("  [ok] evaluate_registered_hypotheses（支持/拒绝/未定/未检验边界）")


def test_build_summary_renders_confirmatory_sections():
    """summary.md 必须包含注册假设小节与置换/CI/MDE 列，而非只留 JSON。"""
    tmp = tempfile.mkdtemp()
    try:
        seeds = [1, 2, 3]
        for v, ppl, al, aq in (
                ("standard", 10.0, None, None),
                ("lega", 9.4, 0.8, 0.7),
                ("lega_no_len", 9.9, 0.75, 0.7),
                ("qgate", 9.8, 0.72, 0.71)):
            for s in seeds:
                jitter = 0.02 * s
                _write_json(os.path.join(tmp, f"eval_{v}_seed{s}.json"), {
                    "variant": v, "seed": s, "val_ppl": ppl + jitter,
                    "val_loss": 2.3, "n_params": 100,
                    "n_gate_extra": 0, "split": "val"})
                if al is not None:
                    _write_json(os.path.join(tmp, f"analysis_{v}_seed{s}.json"), {
                        "variant": v, "seed": s, "auroc_logit": al + 0.01 * s,
                        "auroc_qterm": aq - 0.01 * s})
        reg = {"hypotheses": [
            {"id": "H1", "claim": "LEGA 优于 standard（PPL）", "metric": "PPL",
             "candidate": "lega", "reference": "standard",
             "direction": "candidate_better"}]}
        intact = {"status": "intact", "confirmatory_valid": True, "problems": [],
                  "registry_sha256": "a" * 64, "n_hypotheses": 1,
                  "baseline_document_sha256": "b" * 64, "chain_length": 0}
        with mock.patch("run_all.load_registered_hypotheses",
                        return_value=reg), \
                mock.patch("run_all.verify_registration",
                           return_value=intact):
            out = build_summary(tmp)
        md = out["markdown"]
        assert "注册假设判定" in md
        assert "H1" in md and "事后" in md
        # 配对检验表的新列
        assert "p_perm(Holm)" in md and "MDE" in md and "Δ 95% CI" in md
        assert "AUROC差" in md
        sig = json.load(open(os.path.join(tmp, "significance.json"),
                             encoding="utf-8"))
        assert any(t["metric"] == AUROC_DIFF_METRIC for t in sig["pairs"])
        assert sig["registered_hypotheses"], "注册假设判定必须进 significance.json"
        h1 = next(h for h in sig["registered_hypotheses"] if h["id"] == "H1")
        assert h1["status"] in ("支持", "未定", "拒绝")
        # 预注册完整性报告与完整性小节必须进入 JSON 与汇总页
        assert sig["registration_integrity"]["status"] in ("intact", "amended")
        assert sig["registration_integrity"]["confirmatory_valid"] is True
        assert "预注册完整性" in md and "防事后静默改写" in md
        assert all(h.get("confirmatory_valid") for h in
                   sig["registered_hypotheses"])
        # 每条 pair 检验都带稳健性字段
        for t in sig["pairs"]:
            assert "p_perm" in t and "mean_diff_ci95" in t and "mde" in t
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] build_summary（注册假设/置换/CI/MDE 进入汇总页与 JSON）")


def test_build_summary_downgrades_confirmatory_when_integrity_violated():
    """锁链校验失败时：确认性判定必须显式降级，不能假装边界仍成立。"""
    tmp = tempfile.mkdtemp()
    try:
        for s in (1, 2, 3):
            _write_json(os.path.join(tmp, f"eval_lega_seed{s}.json"), {
                "variant": "lega", "seed": s, "val_ppl": 9.0 + 0.01 * s,
                "val_loss": 2.2, "n_params": 100, "n_gate_extra": 0,
                "split": "val"})
            _write_json(os.path.join(tmp, f"eval_standard_seed{s}.json"), {
                "variant": "standard", "seed": s, "val_ppl": 10.0 + 0.01 * s,
                "val_loss": 2.3, "n_params": 100, "n_gate_extra": 0,
                "split": "val"})
        reg = {"hypotheses": [
            {"id": "H1", "claim": "LEGA 优于 standard（PPL）", "metric": "PPL",
             "candidate": "lega", "reference": "standard",
             "direction": "candidate_better"}]}
        violated = {"status": "violated", "confirmatory_valid": False,
                    "problems": ["已登记的假设 H1 科学内容被改写"],
                    "registry_sha256": "a" * 64, "n_hypotheses": 1,
                    "baseline_document_sha256": "b" * 64,
                    "chain_length": 0}
        with mock.patch("run_all.load_registered_hypotheses",
                        return_value=reg), \
                mock.patch("run_all.verify_registration",
                           return_value=violated):
            out = build_summary(tmp)
        md = out["markdown"]
        assert "已登记的假设 H1 科学内容被改写" in md
        assert "失效" in md and "降级为探索性" in md
        assert "⚠（无效）" in md
        sig = json.load(open(os.path.join(tmp, "significance.json"),
                             encoding="utf-8"))
        assert sig["registration_integrity"]["status"] == "violated"
        assert all(not h["confirmatory_valid"]
                   for h in sig["registered_hypotheses"])
        assert all("降级" in h["confirmatory_note"]
                   for h in sig["registered_hypotheses"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] build_summary（完整性违反 → 确认性判定显式降级）")


def test_build_summary_zero_failure_variant_blocks_consensus():
    """RULER 已评测且零错误的变体必须进入跨变体对照（n_failed=0）。"""
    tmp = tempfile.mkdtemp()
    try:
        case = {"sample_id": "niah_synth@256#0", "task": "niah_synth",
                "ctx": 256, "answer": "1234567", "answer_words": None,
                "pred": "7654321", "em": 0.0, "recall": None,
                "question": "What is the special magic number?",
                "needle_excerpt": "magic number is: 1234567.",
                "depth": 0.5, "meta": {}, "context_numbers": ["7654321"],
                "answer_in_prompt": True}
        for v, cases in (("standard", [case]), ("lega", [case]), ("qgate", [])):
            _write_json(os.path.join(tmp, f"ruler_{v}_seed1.json"),
                        {"variant": v, "seed": 1, "results": {},
                         "badcases": cases})
            _write_json(os.path.join(tmp, f"eval_{v}_seed1.json"), {
                "variant": v, "seed": 1, "val_ppl": 10.0, "val_loss": 2.3,
                "gate_mean": 0.5, "n_params": 100, "n_gate_extra": 0,
                "split": "test"})
        out = build_summary(tmp)
        cross = json.load(open(os.path.join(tmp, "badcase_cross.json"),
                               encoding="utf-8"))
        assert cross["ruler"]["per_variant"]["qgate"]["n_failed"] == 0
        assert cross["ruler"]["n_shared_all_failures"] == 0
        assert "全部通过" in out["markdown"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] build_summary：零错误变体计入跨变体对照")


def test_build_summary_tolerates_seedless_artifacts():
    """产物没有 seed 字段时按缺种子信息处理。

    _seed_key 对无名种子文件返回 None（全表以 None 表“无种子信息”）；ngram
    对比的 excluded_seeds/seeds 与线性探针逐种子表沿用同一口径。
    """
    tmp = tempfile.mkdtemp()
    try:
        _write_json(os.path.join(tmp, "eval_standard.json"),
                    {"variant": "standard", "val_ppl": 10.0, "val_loss": 2.3,
                     "n_params": 10, "split": "val"})
        # 无 seed 且 split 与 ngram 不一致 → 进入 excluded_seeds 分支
        _write_json(os.path.join(tmp, "eval_lega.json"),
                    {"variant": "lega", "val_ppl": 9.0, "val_loss": 2.2,
                     "n_params": 10, "split": "test"})
        _write_json(os.path.join(tmp, "ngram_baseline.json"),
                    {"ppl": {"unigram": 100.0, "bigram": 50.0},
                     "split": "val", "train_tokens": 100, "eval_tokens": 50,
                     "vocab_size": 20})
        _write_json(os.path.join(tmp, "analysis_lega.json"), {
            "variant": "lega",
            "linear_probe": {"primary": "conc_len_query",
                             "feature_sets": {"conc_len_query": {
                                 "oof": {"auprc": 0.5, "auroc": 0.6}}}}})
        build_summary(tmp)
        sig = json.load(open(os.path.join(tmp, "significance.json"),
                             encoding="utf-8"))
        excl = [x for x in sig["ngram_comparison"].get("incomparable", [])
                if x["variant"] == "lega"]
        assert excl and excl[0]["excluded_seeds"] == [None], sig
        probe = sig["linear_probe_comparison"]
        assert probe["rows"] and probe["rows"][0]["seeds"] == [None]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] build_summary：无 seed 产物按缺种子信息处理")


def test_ngram_baseline_rejects_non_dict_json():
    """合法 JSON 但非对象（数组）的 ngram_baseline.json 视为不可用。"""
    tmp = tempfile.mkdtemp()
    try:
        _write_json(os.path.join(tmp, "ngram_baseline.json"), [1, 2, 3])
        r = ngram_baseline_comparison({}, tmp)
        assert r.get("error"), r
        out = build_summary(tmp)
        # 畸形文件按“未运行”处理
        assert "简单统计基线：未运行" in out["markdown"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] ngram_baseline.json 非对象：返回 error/跳过")


def test_ngram_nested_malformed_fields_do_not_crash_summary():
    """ngram_baseline.json 的嵌套字段是列表时按不可用/回退处理，不能崩溃。"""
    tmp = tempfile.mkdtemp()
    try:
        _write_json(os.path.join(tmp, "eval_lega_seed1.json"), {
            "variant": "lega", "seed": 1, "val_ppl": 20.0, "val_loss": 2.0,
            "split": "val"})
        _write_json(os.path.join(tmp, "ngram_baseline.json"), {
            "split": "val", "ppl": [1], "ppl_window_matched": [1],
            "window_spec": [1], "train_tokens": 100, "eval_tokens": 50,
            "vocab_size": 20})
        bad = ngram_baseline_comparison({}, tmp)
        assert bad.get("error") and "ppl" in bad["error"], bad
        out = build_summary(tmp)
        assert "简单基线对比不可用" in out["markdown"]
        # ppl 合法但窗口参考/规格是列表：必须回退全 split 并正常渲染
        _write_json(os.path.join(tmp, "ngram_baseline.json"), {
            "split": "val", "ppl": {"bigram": 100.0},
            "ppl_window_matched": [1], "window_spec": [1],
            "train_tokens": 100, "eval_tokens": 50, "vocab_size": 20})
        out2 = build_summary(tmp)
        assert "未使用逐位置对齐参考" in out2["markdown"]
        sig = json.load(open(os.path.join(tmp, "significance.json"),
                             encoding="utf-8"))
        cmp_doc = sig["ngram_comparison"]
        assert cmp_doc["reference_source"] == "full_split"
        assert "window_spec" in (cmp_doc["fallback_reason"] or "")
        # token 计数/规格计数非数值时只影响展示，不能让整份汇总崩溃
        for patch in ({"train_tokens": [1]}, {"eval_tokens": "x"},
                      {"ppl_window_matched": {"bigram": 90.0},
                       "window_spec": {"ctx": 8, "iters": 2, "n_eval": 2,
                                       "n_tokens": [1]}}):
            doc = {"split": "val", "ppl": {"bigram": 100.0}}
            doc.update(patch)
            _write_json(os.path.join(tmp, "ngram_baseline.json"), doc)
            build_summary(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] ngram_baseline.json 嵌套字段非对象：报错/回退而非崩溃")


def test_build_summary_tolerates_malformed_latency_nested_fields():
    """latency.json 的变体条目/prefill 单元是列表时按缺失跳过，不能崩溃。"""
    tmp = tempfile.mkdtemp()
    try:
        for v in ("standard", "lega"):
            _write_json(os.path.join(tmp, f"eval_{v}_seed1.json"), {
                "variant": v, "seed": 1, "val_ppl": 10.0, "val_loss": 2.3,
                "n_params": 100, "split": "val"})
        lat_path = os.path.join(tmp, "latency.json")
        _write_json(lat_path, {"variants": {
            "standard": [1], "lega": [1]}})
        build_summary(tmp)
        # 基线 standard_sdpa 自身嵌套字段畸形：prefill/efficiency/spread
        # 都不能在运行时指标与权衡小节的读取路径上崩溃。
        _write_json(lat_path, {"variants": {
            "standard": {"prefill_ms": {"256": {"1": 10.0}},
                         "decode_ms_per_tok": {"256": 1.0}},
            "lega": {"prefill_ms": {"256": {"1": 12.0}},
                     "decode_ms_per_tok": {"256": 1.2}},
            "standard_sdpa": {
                "prefill_ms": [1], "decode_ms_per_tok": [1],
                "efficiency": [1],
                "prefill_ms_spread_pct": {"256": [1]},
                "decode_ms_spread_pct": {"256": [1]}}}})
        out = build_summary(tmp)
        assert "涨点-增幅权衡" in out["markdown"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] latency.json 嵌套字段非对象：缺失跳过而非崩溃")


def test_summary_tolerates_infinite_ruler_n_and_numeric_garbage():
    """RULER n=Infinity 不得触发 OverflowError；非数值指标按缺失/跳过。"""
    assert _ruler_min_cell_n({"results": {"niah": {
        "256": {"n": float("inf"), "em": 1.0},
        "512": {"n": 3, "em": "bad"}}}}) == 3
    assert _ppl_of({"ppl": 10 ** 400}) is None
    assert _stat([10 ** 400, "bad", float("nan"), 2.0]) == {
        "mean": 2.0, "std": None, "n": 1}
    tmp = tempfile.mkdtemp()
    try:
        _write_json(os.path.join(tmp, "ruler_standard_seed1.json"), {
            "variant": "standard", "seed": 1, "split": "val",
            "results": {"niah_synth": {
                "256": {"n": float("inf"), "em": 1.0},
                "512": {"n": 3, "em": "bad"}}}})
        out = build_summary(tmp)
        assert "standard" in out["markdown"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] RULER n=Infinity/非数值指标：跳过而非崩溃")


def test_build_summary_tolerates_malformed_data_report_and_audit():
    """data_report/audit 嵌套字段畸形时按缺失渲染，不能毁掉整份汇总。"""
    tmp = tempfile.mkdtemp()
    try:
        for report in (
                {"protocol": [1]}, {"splits": [1]},
                {"tokenizer": {"unk_rate": [1]}},
                {"contamination": {"train_val": [1]}},
                {"distribution_shift": {"val_vs_train": [1]}},
                {"sanitization": [1]}, {"text_stats": [1]},
                {"alerts": [1]},
                {"test_access": [1]}, {"splits": {"val": [1]}}):
            out = build_summary(tmp, data_report=report)
            assert "数据协议与防泄漏自检" in out["markdown"]
        out = build_summary(tmp, data_report={
            "protocol": {"split_first_then_preprocess": True},
            "splits": {"val": {"tokens": "x"}},
            "sanitization": {"train": {"control_chars_deleted": "bad"}},
            "text_stats": {"train": [1]},
            "alerts": [1, {"level": "warning", "code": "c", "message": "m"}],
            "test_access": {"checkpoints_evaluated_multiple_times": 1}})
        assert "[c] m" in out["markdown"]
        for audit in ([1], "bad", {"ok": True, "checked": 1,
                                   "exempt_keys": [1],
                                   "early_stop_runs": [1]}):
            build_summary(tmp, audit=audit)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] data_report/audit 嵌套字段非对象：缺失降级而非崩溃")


def test_ruler_avg_macro_micro_defined_and_reported():
    """RULER 平均必须显式定义：macro（任务/格等权）与 micro（按 n 加权）并报。"""
    doc = {"results": {
        "niah_synth": {"8": {"em": 1.0, "n": 1},
                       "16": {"em": 0.0, "n": 9}},
        "vt": {"8": {"em": 0.0, "n": 1}, "16": {"em": 0.0, "n": 9}}}}
    # macro：任务内两格等权（0.5 与 0），再任务等权 → 0.25
    assert abs(_ruler_avg_of(doc) - 0.25) < 1e-12
    # micro：按 n 加权，只有 niah_synth@8 的 1 个样例命中 → 1/20
    assert abs(_ruler_avg_micro(doc) - 0.05) < 1e-12
    # fwe 用 recall，其余用 EM
    fwe = {"results": {"fwe": {"8": {"em": 0.0, "recall": 0.75, "n": 4}}}}
    assert abs(_ruler_avg_of(fwe) - 0.75) < 1e-12
    assert abs(_ruler_avg_micro(fwe) - 0.75) < 1e-12
    # 缺字段/坏类型按缺失处理
    assert _ruler_avg_micro([1]) is None
    assert _ruler_avg_micro({"results": [1]}) is None
    assert _ruler_avg_micro({"results": {"vt": {"8": {"n": 0}}}}) is None
    print("  [ok] _ruler_avg_micro/_ruler_avg_of（macro/micro 定义与 fwe 口径）")


def test_summary_helpers_tolerate_nested_non_dict_values():
    """嵌套字段是数组/标量时按缺失处理，不能让整份汇总在这些 helper 上崩溃。"""
    assert _ruler_task_means([1]) == {}
    assert _ruler_task_means({"results": [1]}) == {}
    assert _probe_oof([1], "auprc") is None
    assert _probe_oof({"linear_probe": [1]}, "auprc") is None
    assert _ppl_of([1]) is None
    assert _ppl_ci_rel_pct([1]) is None
    assert _ppl_ci_rel_pct({
        "ppl": 1.0, "val_ppl_ci95": [10 ** 400, 1.0]}) is None
    assert _ppl_ci_rel_pct({
        "ppl": 1.0, "val_ppl_ci95": [5.0, 1.0]}) is None
    assert _neural_eval_window_spec({"train_ctx": float("inf")}, False) is None
    assert _neural_eval_window_spec({"eval_iters": "x"}, False) is None
    assert _neural_eval_window_spec([1], False) is None
    assert _eval_matches_window_spec(
        {"eval_ctx": float("inf")},
        {"ctx": 8, "n_eval": 2, "n_windows": 5}) is False
    spec = {"ctx": 8, "n_eval": 2, "n_windows": 5}
    assert _eval_matches_window_spec(
        {"eval_ctx": 8.5, "eval_iters": 2, "val_windows": 5}, spec) is False
    assert _eval_matches_window_spec(
        {"eval_ctx": True, "eval_iters": 2, "val_windows": 5}, spec) is False
    assert _eval_matches_window_spec(
        {"eval_ctx": "8", "eval_iters": "2", "val_windows": "5"}, spec) is True
    tmp = tempfile.mkdtemp()
    try:
        _write_json(os.path.join(tmp, "ngram_baseline.json"),
                    {"ppl": {"bigram": float("inf")}})
        assert ngram_baseline_comparison({}, tmp).get("error")
        _write_json(os.path.join(tmp, "ngram_baseline.json"), {
            "split": "val", "ppl": {"bigram": 100.0},
            "ppl_window_matched": {"bigram": float("inf")},
            "window_spec": {"ctx": 8, "iters": 2}})
        cmp_doc = ngram_baseline_comparison({}, tmp)
        assert cmp_doc["ngram_ppl_window_matched"] is None, cmp_doc
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] 汇总 helper：嵌套非对象/Inf 数值安全降级")


def test_build_summary_tolerates_malformed_eval_probe_repro_curve():
    """eval/analysis/repro/curve 的字段类型畸形时汇总仍能生成。"""
    tmp = tempfile.mkdtemp()
    try:
        _write_json(os.path.join(tmp, "eval_lega_seed1.json"), {
            "variant": "lega", "seed": 1, "val_ppl": 9.0, "val_loss": 2.2,
            "split": ["val"], "epochs_seen": "x",
            "val_tokens_evaluated": "x",
            "worst_windows": ["bad"], "worst_window_scope": {
                "screened": "inf", "total": "x"}})
        # 变体名是数组时既不能进 runs 字典，也不能被当作合法 row。
        _write_json(os.path.join(tmp, "eval_standard_seed1.json"), {
            "variant": ["standard"], "seed": 1, "val_ppl": 10.0})
        _write_json(os.path.join(tmp, "analysis_lega_seed1.json"), {
            "variant": "lega", "seed": 1, "auroc_logit": "x",
            "auroc_qterm": "x", "linear_probe": [1],
            "gate_f1": [1], "gate_f1_trivial": "x"})
        _write_json(os.path.join(tmp, "ngram_baseline.json"), {
            "split": "val", "ppl": {"bigram": 100.0}})
        _write_json(os.path.join(tmp, "repro.json"), {
            "variants": ["lega"], "seeds": [1], "git": [1],
            "data": {"files": ["x"]}})
        _write_json(os.path.join(tmp, "curve_diagnosis.json"), {
            "lega": {"1": {"status": "正常收敛",
                           "recommendations": {"a": 1}}}})
        out = build_summary(tmp)
        assert [r["variant"] for r in out["rows"]] == ["lega"]
        assert "训练曲线诊断" in out["markdown"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] build_summary：畸形 eval/analysis/repro/curve 不崩溃")


def test_tradeoff_prefers_sdpa_runtime_baseline():
    """涨点-增幅的运行时要与主表同一基线（standard_sdpa 优先）。"""
    tmp = tempfile.mkdtemp()
    try:
        for v in ("standard", "standard_sdpa", "lega"):
            _write_json(os.path.join(tmp, f"eval_{v}_seed1.json"), {
                "variant": v, "seed": 1, "val_ppl": 10.0, "val_loss": 2.3,
                "gate_mean": 0.5 if v == "lega" else None,
                "n_params": 100, "n_gate_extra": 0})

        def _lat(prefill, dec):
            return {"prefill_ms": {"256": {"1": prefill}},
                    "decode_ms_per_tok": {"256": dec},
                    "efficiency": {
                        "n_params": 100,
                        "flops_per_tok": {"256": {"total_flops_per_tok": 1000}},
                        "peak_mem_mb_prefill": {"256": 1.0}}}

        _write_json(os.path.join(tmp, "latency.json"), {"variants": {
            "standard": _lat(10.0, 1.0),        # 显式路径（旧默认基线）
            "standard_sdpa": _lat(1.0, 0.1),    # 与候选同路径的可比基线
            "lega": _lat(1.1, 0.11)}})
        out = build_summary(tmp)
        row = next(r for r in out["rows"] if r["variant"] == "lega")
        assert row["prefill_rel_pct"] == 10.0, row
        assert row["eff_prefill_delta_pct"] == 10.0, row
        assert row["eff_cost_max_pct"] == 10.0, row
        assert "基线=standard_sdpa" in out["markdown"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] 涨点-增幅：运行时基线优先 standard_sdpa（与主表一致）")


def test_scope_missing_config_or_data_is_not_confirmed():
    """缺 config/data 时范围核对必须“不可确认”，不能因种子匹配就通过。"""
    tmp = tempfile.mkdtemp()
    try:
        reg = os.path.join(tmp, "reg.json")
        actual = os.path.join(tmp, "actual.json")
        cfg = {"d_model": 384, "n_layers": 6, "n_heads": 6, "ffn_mult": 4,
               "block_size": 1024, "steps": 8000, "train_ctx": 1024}
        _write_json(reg, cfg)
        _write_json(actual, cfg)
        registered = {"scope": {"config": reg, "seeds": [1, 2, 3],
                                "data": "WikiText-103 (gpt2 BPE)"}}

        # config 与 data 都缺失：即使 seeds 匹配也不能通过
        sc = registration_scope_check(registered, {"seeds": [1]})
        assert sc["checked"] is False and sc["in_scope"] is None, sc
        assert sc["warnings"]
        hyp = [{"id": "H", "confirmatory_valid": True}]
        assert apply_scope_gate(hyp, sc)[0]["confirmatory_valid"] is False

        # 有 config、缺 data.tokenizer：仍不可确认
        sc2 = registration_scope_check(registered, {
            "config_file": actual, "seeds": [1], "data": {}})
        assert sc2["checked"] is False and sc2["in_scope"] is None, sc2

        # 三部分都可核对且匹配：通过；分词器匹配本身也算“已核对”
        sc3 = registration_scope_check(registered, {
            "config_file": actual, "seeds": [1],
            "data": {"tokenizer": "gpt2_bpe"}})
        assert sc3["checked"] is True and sc3["in_scope"] is True, sc3

        # scope 未声明 config/data、仅声明 seeds：种子核对通过即可判定
        sc4 = registration_scope_check({"scope": {"seeds": [1]}},
                                       {"seeds": [1]})
        assert sc4["checked"] is True and sc4["in_scope"] is True, sc4
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] 范围核对：缺 config/data 不可确认，匹配 tokenizer 计入核对")


def test_verify_controlled_setup_checks_nonearly_runs_with_early_peer():
    """同批中一个 run 早停，不能豁免其它 run 的实际步数/计划步数检查。"""
    tmp = tempfile.mkdtemp()
    try:
        variants, seeds = ["standard", "lega"], [7]
        for v in variants:
            d = os.path.join(tmp, v)
            os.makedirs(d)
            _write_json(os.path.join(d, "config_used.json"), {
                "lr": 0.001, "steps": 10, "warmup": 2,
                "early_stop_patience": 2,
                "variant": v, "seed": 7, "device": "cpu", "tag": "t"})
        common = {"data_fingerprint": "abc123", "code_commit": "deadbeef",
                  "config_sha256": "shax", "tokens_seen": 1024}
        _write_json(os.path.join(tmp, "standard", "eval.json"), dict(
            common, variant="standard", seed=7, steps=8,
            steps_planned=10, early_stopped=False))
        _write_json(os.path.join(tmp, "lega", "eval.json"), dict(
            common, variant="lega", seed=7, steps=6,
            steps_planned=10, early_stopped=True, tokens_seen=614))
        try:
            verify_controlled_setup(tmp, variants, seeds, 1,
                                    expected_data_digest="abc123")
            raise AssertionError("非早停 run 实际步数 != 计划步数应触发 SystemExit")
        except SystemExit:
            pass
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] 混合早停：非早停 run 仍核对实际/计划步数")


def test_summary_loads_tolerate_null_ppl_and_bad_json_seed():
    """缺失 ppl 回退 val_ppl；无法解析的 seed 按无种子信息处理。"""
    assert _ppl_of({"ppl": None, "val_ppl": 9.5}) == 9.5
    assert _ppl_of({"ppl": 8.0, "val_ppl": 9.5}) == 8.0
    assert _seed_key("eval_lega.json", {"seed": "not-an-int"}) is None
    tmp = tempfile.mkdtemp()
    try:
        _write_json(os.path.join(tmp, "eval_standard_seed1.json"), {
            "variant": "standard", "seed": 1, "val_ppl": 10.0,
            "val_loss": 2.3, "n_params": 10, "split": "val"})
        for name in ("repro.json", "latency.json", "baseline_tuning.json",
                     "curve_diagnosis.json", "sanity_standard.json"):
            _write_json(os.path.join(tmp, name), [1, 2, 3])
        with open(os.path.join(tmp, "eval_lega_seed1.json"), "w",
                  encoding="utf-8") as f:
            f.write("{ 坏 JSON")
        out = build_summary(tmp)
        assert "standard" in out["markdown"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] 汇总：ppl=null 回退、坏 JSON 文件跳过、坏 seed 不崩溃")


def test_record_aborted_sanity_writes_reason_and_failed_checks():
    """sanity 闸门失败：台账必须有 aborted 记录、失败原因与未通过项。"""
    from lega.experiment import read_records
    tmp = tempfile.mkdtemp()
    try:
        results_dir = os.path.join(tmp, "results")
        os.makedirs(results_dir)
        _write_json(os.path.join(results_dir, "sanity_lega.json"), {
            "passed": False, "variant": "lega", "seed": 7,
            "checks": {"loss_below_threshold": False,
                       "next_token_shift_consistent": True},
            "initial": {"loss": 4.2}, "final": {"loss": 3.9,
                                                "token_acc": 0.1}})
        ledger = os.path.join(tmp, "experiments.jsonl")
        common = dict(
            run_id="rid1", tag="t", config_path="configs/tiny.json",
            config_sha256="a" * 64, config={"lr": 1e-3}, variants=["lega"],
            seeds=[7], smoke=False, argv=["run_all.py"],
            git={"commit": "c" * 40, "dirty": False}, data_fingerprint="fp",
            runtime={"torch": "x"}, out_root="out/t", results_dir=results_dir,
            time_start="2026-01-01 00:00:00")
        failed = record_aborted_sanity(ledger, common, "lega", 1, results_dir)
        assert failed["passed"] is False
        assert failed["failed_checks"] == ["loss_below_threshold"]
        recs = read_records(ledger)
        assert len(recs) == 1
        rec = recs[0]
        assert rec["status"] == "aborted"
        assert rec["extra"]["error"] == "sanity_gate_failed"
        assert rec["extra"]["variant"] == "lega" and rec["extra"]["returncode"] == 1
        assert rec["sanity"]["verdict"] == {"lega": "fail"}
        assert rec["sanity"]["failed"]["final_loss"] == 3.9
        assert rec["config"] == {"lr": 1e-3}
        # 结果文件缺失时同样留痕（failed.present=False），不崩
        os.remove(os.path.join(results_dir, "sanity_lega.json"))
        ledger2 = os.path.join(tmp, "exp2.jsonl")
        failed2 = record_aborted_sanity(ledger2, common, "lega", 2, results_dir)
        assert failed2["present"] is False
        assert read_records(ledger2)[0]["status"] == "aborted"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] 台账 aborted 记录：失败原因 + 未通过项 + 缺失结果容错")


def test_append_aborted_record_covers_run_chain_failures():
    """train/eval/ruler/analysis 任一阶段失败也要落 aborted（含阶段与变体）。"""
    from lega.experiment import read_records
    from run_all import append_aborted_record
    tmp = tempfile.mkdtemp()
    try:
        ledger = os.path.join(tmp, "experiments.jsonl")
        common = dict(
            run_id="rid2", tag="t", config_path="configs/tiny.json",
            config_sha256="b" * 64, config={"lr": 1e-3}, variants=["lega"],
            seeds=[7], smoke=False, argv=["run_all.py"], git={},
            data_fingerprint="fp", runtime={}, out_root="out/t",
            results_dir="results/t", time_start="2026-01-01 00:00:00")
        append_aborted_record(ledger, common, error="run_chain_failed",
                              stage="train/eval/ruler/analysis",
                              variant="lega", returncode=1)
        rec = read_records(ledger)[0]
        assert rec["status"] == "aborted"
        assert rec["extra"] == {"error": "run_chain_failed",
                                "stage": "train/eval/ruler/analysis",
                                "variant": "lega", "returncode": 1}
        assert rec["time_end"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] run 链失败落 aborted（阶段/变体/退出码）")


def test_summary_tolerates_malformed_sanity_and_baseline_tuning():
    """sanity/tuning 的嵌套字段被手改/旧格式污染时，汇总不能崩。"""
    tmp = tempfile.mkdtemp()
    try:
        _write_json(os.path.join(tmp, "eval_standard.json"), {
            "variant": "standard", "seed": 1, "val_ppl": 10.0,
            "val_loss": 2.3, "n_params": 10, "split": "val"})
        _write_json(os.path.join(tmp, "sanity_standard.json"), {
            "variant": [1, 2], "passed": True, "initial": [1, 2],
            "final": {"loss": "bad", "token_acc": [1, 2]}})
        _write_json(os.path.join(tmp, "baseline_tuning.json"), {
            "grid": [{"lr": [1, 2], "weight_decay": {"x": 1}}],
            "trials": 5, "selected": {"lr": [1]},
            "default_trial": [], "selection_seeds": {"x": 1}})
        out = build_summary(tmp)
        assert "standard" in out["markdown"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] 汇总：sanity/tuning 嵌套字段畸形安全跳过")


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
    apply_scope_gate(hyp, {"scope": {"config": "x"}, "checked": False,
                           "problems": [1]})
    assert hyp[0]["confirmatory_valid"] is False
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
    print("  [ok] 受控变量审计：小数步数不会被截断为整数")


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
