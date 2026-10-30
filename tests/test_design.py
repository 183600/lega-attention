"""实验设计与准备层测试：先验功效、设计计划、分析人群与确认性降级。

覆盖：
  * 配对 t / McNemar 的功效与所需样本数互为逆运算，且拒绝非法输入；
  * 计划在训练前写死种子/变体/SESOI 灵敏度，缺登记种子时 fail；
  * 确认性功效按 Holm 家族/全局族的最终 α_eff 规划，并与 run_all 配对一致；
  * 计划内 (变体, 种子) 单元缺失时，只降级受影响假设的确认性判定；
  * 旧产物没有 design_plan.json 时不误降级（向后兼容）。
"""
import json
import math
import os
import shutil
import statistics
import tempfile

from lega.design import (_fmt, analysis_population, apply_design_gate,
                         build_design_plan, build_run_order,
                         design_conformance,
                         load_design_plan, main as design_main,
                         mcnemar_mde, mcnemar_power,
                         mcnemar_required_samples,
                         max_discordance_for_effect, max_sd_for_effect,
                         min_detectable_effect, multiplicity_alpha,
                         observed_design_sensitivity, paired_t_power,
                         planned_comparison_pairs, planned_family_sizes,
                         plan_digest, render_design_section,
                         required_pairs, save_design_plan,
                         validate_run_order)


def test_design_cli_seed_parser_is_strict():
    """--seeds 的 "1_0"/"0x10" 不能经 int(..., 0) 静默变成别的种子。"""
    from lega.design import _parse_seeds
    assert _parse_seeds("1, 2") == [1, 2]
    assert _parse_seeds(" 1 , ,2 ") == [1, 2]
    for bad in ("1_0", "0x10", "0b10", "1.0", "True", "-1"):
        try:
            _parse_seeds(bad)
            raise AssertionError(f"--seeds={bad!r} 应显式报错")
        except ValueError:
            pass
    print("  [ok] design --seeds 与全项目严格种子口径一致")


def _write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)


def _plan(seeds=(1, 2, 3, 4, 5), variants=("lega", "standard", "qgate"),
          registered=None, **kw):
    cfg = {"ruler": {"ctx_lens": [256, 512], "n_samples": 30},
           "design": {"alpha": 0.05, "power": 0.8}}
    if registered is None:
        registered = {"hypotheses": [
            {"id": "H1", "metric": "PPL", "candidate": "lega",
             "reference": "standard", "direction": "candidate_better"},
            {"id": "H5", "metric": "PPL", "candidate": "lega",
             "reference": "qgate", "direction": "candidate_better"},
        ]}
    return build_design_plan(cfg, list(seeds), list(variants), registered,
                             known_variants={"lega", "standard", "qgate"},
                             **kw)


def test_paired_power_and_required_seeds():
    """配对 t：功效随 n 单调增，required_pairs 与 power 互为逆运算。"""
    for n in (3, 5, 8, 12):
        assert paired_t_power(0.01, 0.01, n) > paired_t_power(
            0.01, 0.01, n - 1), n
        # 双侧检验只看效应幅度，负方向必须与正方向同功效。
        assert paired_t_power(-0.01, 0.01, n) == paired_t_power(
            0.01, 0.01, n), n
    n = required_pairs(0.01, 0.01, alpha=0.05, power=0.8)
    assert n == 8, n
    assert paired_t_power(0.01, 0.01, n) >= 0.8
    assert paired_t_power(0.01, 0.01, n - 1) < 0.8
    # MDE 与功效是同一公式的两种写法。
    mde = min_detectable_effect(0.01, 5)
    assert abs(paired_t_power(mde, 0.01, 5) - 0.8) < 1e-5
    # 80% 功效下可容忍的最大 SD：把 SD 设为该值，功效恰为 0.8。
    sd_max = max_sd_for_effect(0.01, 5)
    assert abs(paired_t_power(0.01, sd_max, 5) - 0.8) < 1e-5
    # 与全项目使用的 MDE 口径逐位一致，避免两套公式漂移。
    from lega.utils import min_detectable_effect as utils_mde
    for sd in (0.002, 0.01, 0.05):
        for nn in (3, 5, 10):
            assert abs(min_detectable_effect(sd, nn)
                       - utils_mde(sd, nn)) < 1e-15
    # 非法输入显式拒绝，不静默返回“看起来显著”的值。
    for args in (("x", 0.01, 5), (0.01, 0, 5), (0.01, 0.01, True),
                 (0.01, 0.01, 1)):
        try:
            paired_t_power(*args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"应拒绝非法输入 {args}")
    print("  [ok] 配对 t 功效/所需种子数与非法输入")


def test_mcnemar_power_and_required_samples():
    """McNemar：小样本精确功效、大样本近似、所需样本数互为逆运算。"""
    small = mcnemar_power(30, 0.10, 0.01)
    assert small["method"] == "exact"
    # n=30 时期望不一致对只有 3 个，精确检验对 1pp 几乎无功效。
    assert small["power"] < 0.05, small
    big = mcnemar_power(8000, 0.10, 0.01)
    assert big["method"] == "normal_approx"
    assert big["power"] >= 0.8, big
    need = mcnemar_required_samples(0.01, 0.10)
    assert need is not None and 7000 <= need <= 9000, need
    assert mcnemar_power(need, 0.10, 0.01)["power"] >= 0.8
    assert mcnemar_power(need - 1, 0.10, 0.01)["power"] < 0.8
    # 正态近似起点可能高于精确所需样本数；max_n 恰好覆盖精确解时不能
    # 因近似起点超限而误报“不可达”。
    assert mcnemar_required_samples(
        0.02, 0.02, alpha=0.1, power=0.9, max_n=398) == 398
    assert mcnemar_required_samples(
        0.02, 0.02, alpha=0.1, power=0.9, max_n=397) is None
    # MDE 与所需样本数同源：n=30、π=0.1 时约 16pp。
    mde = mcnemar_mde(30, 0.10)
    assert 0.15 < mde < 0.17, mde
    # 效应越大功效越高。
    assert mcnemar_power(200, 0.2, 0.10)["power"] > \
        mcnemar_power(200, 0.2, 0.01)["power"]
    # 边际差大于不一致率在概率上不可能。
    try:
        mcnemar_power(30, 0.05, 0.10)
    except ValueError:
        pass
    else:
        raise AssertionError("|effect|>discordance 必须拒绝")
    # 可容忍不一致率与 n 成正比。
    assert max_discordance_for_effect(0.01, 60) > \
        max_discordance_for_effect(0.01, 30)
    print("  [ok] McNemar 精确/近似功效与所需每格样本数")


def test_build_design_plan_flags_missing_registered_seeds():
    """少跑预注册种子：计划必须 fail 并在 checks 中写明。"""
    reg = {"scope": {"seeds": [1, 2, 3, 4, 5]},
           "hypotheses": [
               {"id": "H1", "metric": "PPL", "candidate": "lega",
                "reference": "standard", "direction": "candidate_better"}]}
    plan = _plan(seeds=(1, 2, 3), registered=reg)
    assert plan["status"] == "fail", plan["status"]
    by_id = {c["id"]: c for c in plan["checks"]}
    assert by_id["seed_scope"]["status"] == "fail"
    assert "4" in by_id["seed_scope"]["detail"]
    assert by_id["hypothesis_variants"]["status"] == "pass"
    # 灵敏度表必须存在，且 PPL 的 1% SESOI 与 n=3 的断点可读。
    ppl = plan["endpoints"]["ppl"]
    assert ppl["planned_seeds"] == 3
    assert len(ppl["sensitivity"]) >= 3
    assert ppl["max_paired_sd_for_sesoi"] is not None
    ruler = plan["endpoints"]["ruler"]
    assert ruler["samples_per_cell"] == 30
    assert ruler["cells"] == 2 * 5
    assert any(r["discordance"] == 0.1 for r in ruler["sensitivity"])
    assert plan["plan_digest"] == plan_digest(plan)
    # 摘要里必须包含“分析人群/排除规则”的先验声明与登记假设快照。
    assert plan["analysis_protocol"]["exclusions"]
    assert plan["registered_plan"][0]["id"] == "H1"
    assert plan["registered_plan"][0]["metric"] == "PPL"
    print("  [ok] 设计计划：缺失预注册种子判 fail，灵敏度表完整")


def test_confirmatory_power_uses_planned_holm_burden():
    """先验功效必须按最终 Holm 阈值规划，不能高估确认性灵敏度。"""
    reg = {"hypotheses": [
        {"id": "H1", "metric": "PPL", "candidate": "lega",
         "reference": "standard"},
        {"id": "H5", "metric": "PPL", "candidate": "lega",
         "reference": "qgate"}]}
    plan = _plan(registered=reg)
    mp = plan["multiplicity_plan"]
    assert mp["family_sizes"]["PPL"] == 3, mp
    assert mp["family_sizes"]["global_confirmatory"] == 2, mp
    alpha_eff = mp["alpha_effective"]["PPL"]
    assert abs(alpha_eff - 0.05 / 3) < 1e-15, alpha_eff
    ppl = plan["endpoints"]["ppl"]
    row = next(r for r in ppl["sensitivity"]
               if abs(r["paired_sd_relative"] - 0.01) < 1e-12)
    assert row["power_for_sesoi_confirmatory"] < row["power_for_sesoi"]
    assert (row["mde_relative_confirmatory"]
            > row["mde_relative"])
    assert (row["required_seeds_for_sesoi_confirmatory"]
            > row["required_seeds_for_sesoi"])
    assert (ppl["max_paired_sd_for_sesoi_confirmatory"]
            < ppl["max_paired_sd_for_sesoi"])
    assert any(c["id"] == "ppl_confirmatory_reference"
               and c["status"] == "warn" for c in plan["checks"])

    raw = _plan(registered=reg, multiplicity="none")
    assert abs(raw["multiplicity_plan"]["alpha_effective"]["PPL"]
               - 0.05) < 1e-15
    raw_row = next(r for r in raw["endpoints"]["ppl"]["sensitivity"]
                   if abs(r["paired_sd_relative"] - 0.01) < 1e-12)
    assert (abs(raw_row["power_for_sesoi_confirmatory"]
                - raw_row["power_for_sesoi"]) < 1e-15)

    # 正式矩阵：9 个变体 vs standard + 6 个单因素 + 1 个预注册整体对比
    # = 16 个 PPL/RULER 检验；7 条预注册假设构成全局确认族。
    variants = ["standard", "gate_only", "qgate", "null", "lega",
                "lega_no_len", "lega_no_conc", "lega_no_query",
                "lega_conc_only", "lega_len_only"]
    analysis = [v for v in variants if v not in ("standard", "gate_only")]
    reg7 = {"hypotheses": [
        {"id": f"H{i}", "metric": "PPL", "candidate": "lega",
         "reference": "standard"} for i in range(1, 7)]
        + [{"id": "H7", "metric": "PPL", "candidate": "lega",
            "reference": "qgate"}]}
    sizes = planned_family_sizes(variants, reg7,
                                 analysis_variants=analysis)
    assert sizes["ppl_family_size"] == 16, sizes
    assert sizes["global_family_size"] == 7, sizes
    assert sizes["auroc_family_size"] == 8, sizes
    plan7 = build_design_plan(
        {"ruler": {"ctx_lens": [256, 512], "n_samples": 30}},
        [1, 2, 3, 4, 5], variants, reg7,
        known_variants=set(variants), analysis_variants=analysis)
    assert abs(plan7["multiplicity_plan"]["alpha_effective"]["PPL"]
               - 0.05 / 16) < 1e-15
    assert abs(plan7["multiplicity_plan"]["alpha_effective"]["AUROC"]
               - 0.05 / 8) < 1e-15
    # 设计层与汇总层的配对集合必须逐项一致，防止任一侧新增对比后
    # 设计功效仍按旧家族大小计算。
    import lega.design as design_mod
    import run_all
    assert planned_comparison_pairs(variants, reg7) == \
        run_all.build_comparison_pairs({v: {} for v in variants}, reg7)
    # 显式加入补充对照时，设计层的家族大小同样必须与汇总层一致。
    from lega.attention import SUPPLEMENTAL_VARIANTS
    supp = variants + list(SUPPLEMENTAL_VARIANTS)
    assert planned_comparison_pairs(supp, reg7) == \
        run_all.build_comparison_pairs({v: {} for v in supp}, reg7)
    assert (set(design_mod._DEFAULT_GATED_FOR_ANALYSIS)
            == set(run_all.GATED_FOR_ANALYSIS))

    try:
        _plan(registered=reg, multiplicity="bogus")
    except ValueError:
        pass
    else:
        raise AssertionError("未知 multiplicity 必须拒绝")
    text = "\n".join(render_design_section(plan))
    assert "确认性" in text and "α_eff" in text
    print("  [ok] 确认性功效按 Holm 家族负担规划，并与 run_all 配对一致")


def test_build_design_plan_smoke_downgrades_fail():
    """冒烟运行的 fail 降为 warn：流程检查不应冒充正式设计失败。"""
    reg = {"scope": {"seeds": [1, 2, 3, 4, 5]},
           "hypotheses": [
               {"id": "H1", "metric": "PPL", "candidate": "lega",
                "reference": "standard"}]}
    plan = _plan(seeds=(1,), registered=reg, smoke=True,
                 ruler_n_samples=2)
    assert plan["status"] == "warning", plan["status"]
    seed = {c["id"]: c for c in plan["checks"]}["seed_scope"]
    assert seed["status"] == "warn"
    assert "冒烟" in seed["detail"]
    print("  [ok] 设计计划：冒烟把 fail 降为 warn 并留痕")


def test_analysis_population_and_conformance():
    """计划内单元缺失必须显式进入 population/problems。"""
    plan = _plan()
    runs = {
        "lega": {1: {"eval": {}, "ruler": {}, "analysis": {}},
                 2: {"eval": {}, "ruler": {}, "analysis": {}},
                 3: {"eval": {}}},                     # 缺 ruler/analysis
        "standard": {1: {"eval": {}, "ruler": {}, "analysis": {}},
                     2: {"eval": {}, "ruler": {}, "analysis": {}},
                     3: {"eval": {}, "ruler": {}, "analysis": {}}},
        "qgate": {1: {"eval": {}, "ruler": {}, "analysis": {}},
                  2: {"eval": {}, "ruler": {}, "analysis": {}}},  # 缺 seed3
    }
    pop = analysis_population(plan, runs)
    assert pop["expected_cells"] == 15
    assert pop["observed_eval_cells"] == 8
    assert "qgate/seed3" in pop["missing_eval"]
    assert "lega/seed3" in pop["missing_ruler"]
    assert pop["complete_eval"] is False
    repro = {"variants": plan["inputs"]["variants"],
             "seeds": plan["inputs"]["seeds"], "smoke": False,
             "config_sha256": plan["inputs"]["config_sha256"]}
    conf = design_conformance(plan, repro, runs,
                              stages={"ruler": True, "analysis": True})
    assert conf["checked"] is True
    assert conf["in_plan"] is False
    assert any("qgate/seed3" in p for p in conf["problems"])
    # repro 与计划不一致（上一轮遗留计划）时不据此降级。
    conf2 = design_conformance(plan, dict(repro, seeds=[1, 2]),
                               runs)
    assert conf2["checked"] is False
    assert conf2["reason"] == "plan_repro_seed_mismatch"
    # 完全没有计划（旧产物）：checked=False，不误判。
    conf3 = design_conformance({}, repro, runs)
    assert conf3["checked"] is False
    assert conf3["reason"] == "missing_plan"
    assert load_design_plan(tempfile.mkdtemp()) == {}
    print("  [ok] 分析人群/设计符合性：缺失单元与计划错配均显形")


def test_apply_design_gate_targets_only_affected_hypotheses():
    """缺 qgate 只降级 H5/含 qgate 的假设；H1、H6 不受影响。"""
    plan = _plan(seeds=(1, 2, 3))
    runs = {
        "lega": {s: {"eval": {}, "ruler": {}, "analysis": {}}
                 for s in (1, 2, 3)},
        "standard": {s: {"eval": {}, "ruler": {}, "analysis": {}}
                     for s in (1, 2, 3)},
        "qgate": {1: {"eval": {}, "ruler": {}, "analysis": {}},
                  2: {"eval": {}, "ruler": {}, "analysis": {}}},
    }
    repro = {"variants": plan["inputs"]["variants"],
             "seeds": plan["inputs"]["seeds"], "smoke": False}
    conf = design_conformance(plan, repro, runs)
    hyps = [
        {"id": "H1", "metric": "PPL", "candidate": "lega",
         "reference": "standard", "confirmatory_valid": True},
        {"id": "H5", "metric": "PPL", "candidate": "lega",
         "reference": "qgate", "confirmatory_valid": True},
        {"id": "H4", "metric": "RULER平均", "candidate": "lega",
         "reference": "standard", "confirmatory_valid": True},
        {"id": "H6", "metric": "AUROC差(logit−仅查询)", "candidate": "lega",
         "reference": "仅查询项基线", "confirmatory_valid": True},
    ]
    out = apply_design_gate(hyps, conf, plan)
    by = {h["id"]: h for h in out}
    assert by["H1"]["confirmatory_valid"] is True
    assert by["H1"]["design_population_ok"] is True
    assert by["H5"]["confirmatory_valid"] is False
    assert by["H5"]["design_population_ok"] is False
    assert "分析人群不完整" in by["H5"]["confirmatory_note"]
    assert by["H4"]["confirmatory_valid"] is True  # RULER 单元齐全
    assert by["H6"]["confirmatory_valid"] is True  # analysis 单元齐全
    # 已因其他门降级的条目保留原原因，不被覆盖。
    hyps2 = [{"id": "H5", "metric": "PPL", "candidate": "lega",
              "reference": "qgate", "confirmatory_valid": False,
              "confirmatory_note": "预注册完整性失败"}]
    out2 = apply_design_gate(hyps2, conf, plan)
    assert out2[0]["confirmatory_note"] == "预注册完整性失败"
    # 计划不可核对时不降级。
    hyps3 = [{"id": "H1", "metric": "PPL", "candidate": "lega",
              "reference": "standard", "confirmatory_valid": True}]
    out3 = apply_design_gate(hyps3, {"checked": False}, plan)
    assert out3[0]["confirmatory_valid"] is True
    assert out3[0]["design_population_ok"] is None
    print("  [ok] 设计门：只降级受影响假设，保留更早的降级原因")


def test_observed_design_sensitivity():
    """实测 SD 反推 MDE：与 SESOI 同尺度比较，不伪装成观测功效。"""
    tests = [
        {"candidate": "lega", "reference": "standard", "metric": "PPL",
         "equivalence": {"status": "not_equivalent", "std_diff": 0.01,
                         "n": 5, "margin": 0.01}},
        {"candidate": "lega", "reference": "qgate", "metric": "PPL",
         "equivalence": {"status": "not_equivalent", "std_diff": 0.002,
                         "n": 5, "margin": 0.01}},
        {"candidate": "bad", "reference": "x", "metric": "PPL",
         "equivalence": {"status": "invalid", "std_diff": 0.1,
                         "n": 5, "margin": 0.01}},
    ]
    rows = observed_design_sensitivity(tests)
    assert len(rows) == 2
    r0, r1 = rows
    assert r0["adequate_for_sesoi"] is False
    assert r0["observed_mde_sesoi_units"] > r0["sesoi_margin"]
    assert r1["adequate_for_sesoi"] is True
    assert r1["power_for_sesoi"] > 0.99
    print("  [ok] 实测灵敏度：MDE 与 SESOI 比较、跳过 invalid")


def test_render_design_section_with_and_without_plan():
    """汇总页必须把计划、检查、灵敏度与人群偏差渲染出来。"""
    plan = _plan(seeds=(1, 2, 3))
    runs = {
        "lega": {s: {"eval": {}, "ruler": {}, "analysis": {}}
                 for s in (1, 2)},
        "standard": {s: {"eval": {}, "ruler": {}, "analysis": {}}
                     for s in (1, 2, 3)},
        "qgate": {s: {"eval": {}, "ruler": {}, "analysis": {}}
                  for s in (1, 2, 3)},
    }
    repro = {"variants": plan["inputs"]["variants"],
             "seeds": plan["inputs"]["seeds"], "smoke": False}
    conf = design_conformance(plan, repro, runs)
    md = "\n".join(render_design_section(
        plan, conf, observed=observed_design_sensitivity([])))
    assert "实验设计与分析人群" in md
    assert "PPL 先验灵敏度" in md and "RULER 先验灵敏度" in md
    assert "设计符合性与分析人群" in md
    assert "lega/seed3" in md
    assert "确认性判定已自动降级" in md
    assert "配对差SD" in md
    md2 = "\n".join(render_design_section({}, None))
    assert "未找到 `design_plan.json`" in md2
    print("  [ok] 汇总页：计划/检查/灵敏度/人群偏差与无计划告警")


def test_fmt_percent_handles_scaled_overflow():
    """有限值乘 100 溢出时不能渲染 inf%，也保留正常百分比输出。"""
    assert _fmt(1e307, 2, pct=True) == "溢出"
    assert _fmt(1e307, 2, pct=True, signed=True) == "溢出"
    assert _fmt(0.0123, 2, pct=True) == "1.23%"
    assert _fmt(0.0123, 2, pct=True, signed=True) == "+1.23%"
    print("  [ok] 表格百分比防溢出")


def test_design_cli_writes_plan_and_exit_code():
    """CLI：写出计划；有 fail 时退出码 2，警告时 0。"""
    tmp = tempfile.mkdtemp()
    try:
        cfg = {"seeds": [1, 2, 3], "seed": 1,
               "ruler": {"ctx_lens": [256], "n_samples": 30}}
        cfg_path = os.path.join(tmp, "cfg.json")
        _write_json(cfg_path, cfg)
        reg = {"hypotheses": [
            {"id": "H1", "metric": "PPL", "candidate": "lega",
             "reference": "standard"}]}
        reg_path = os.path.join(tmp, "reg.json")
        _write_json(reg_path, reg)
        out = os.path.join(tmp, "plan.json")
        data_dir = os.path.join(tmp, "data")
        os.makedirs(data_dir)
        _write_json(os.path.join(data_dir, "data_report.json"), {
            "accounting": {"train": {"tokens": 100000},
                           "val": {"tokens": 100000},
                           "test": {"tokens": 100000}},
            "tokenizer": {"kind": "char"}})
        rc = design_main(["--config", cfg_path, "--variants", "lega,standard",
                          "--registered", reg_path, "--out", out,
                          "--data_dir", data_dir, "--quiet"])
        assert rc == 0, rc
        plan = json.load(open(out, encoding="utf-8"))
        assert plan["inputs"]["seeds"] == [1, 2, 3]
        assert plan["status"] in ("ok", "warning")
        assert plan["data_budget"]["report_present"] is True
        # 少跑预注册种子 → fail → 退出码 2。
        _write_json(reg_path, {"scope": {"seeds": [1, 2, 3, 4, 5]},
                               "hypotheses": reg["hypotheses"]})
        rc2 = design_main(["--config", cfg_path, "--variants",
                           "lega,standard", "--registered", reg_path,
                           "--out", out, "--quiet"])
        assert rc2 == 2, rc2
        plan2 = json.load(open(out, encoding="utf-8"))
        assert plan2["status"] == "fail"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] design CLI：写计划、fail 退出码 2、重写原子落盘")


def test_design_cli_loads_default_preregistration():
    """不传 --registered 时必须从仓库根目录读取预注册表，而非 lega/ 下。"""
    tmp = tempfile.mkdtemp()
    try:
        cfg = {"seeds": [1337, 2024, 42, 7, 2025],
               "ruler": {"ctx_lens": [256, 512], "n_samples": 30}}
        cfg_path = os.path.join(tmp, "cfg.json")
        _write_json(cfg_path, cfg)
        out = os.path.join(tmp, "plan.json")
        design_main(["--config", cfg_path, "--variants", "lega,standard",
                     "--out", out, "--quiet"])
        plan = json.load(open(out, encoding="utf-8"))
        assert plan["inputs"]["registered_hypotheses"], plan["inputs"]
        assert any(c.get("id") == "scope_config"
                   for c in plan["checks"] if isinstance(c, dict))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] design CLI：默认预注册表路径正确")


def test_plan_digest_tracks_scientific_content():
    """计划摘要对种子/变体敏感，对生成时间不敏感。"""
    p1 = _plan()
    p2 = _plan()
    assert p1["plan_digest"] == p2["plan_digest"]
    p3 = _plan(seeds=(1, 2, 3, 4, 5, 6))
    assert p3["plan_digest"] != p1["plan_digest"]
    p4 = _plan(variants=("lega", "standard"))
    assert p4["plan_digest"] != p1["plan_digest"]
    p5 = _plan(data_report={"accounting": {"train": {"tokens": 1000}}})
    assert p5["plan_digest"] != p1["plan_digest"]
    print("  [ok] 计划摘要：同一设计稳定、内容/数据预算变化即变")


def test_design_plan_records_test_look_budget():
    """测试集 look 预算进入计划；默认 1，非法值显式 fail 而不是静默采用。"""
    plan = _plan()
    assert plan["test_look_budget"] == 1
    check = {c["id"]: c for c in plan["checks"]}["test_look_budget"]
    assert check["status"] == "info"

    custom = build_design_plan(
        {"ruler": {"ctx_lens": [256], "n_samples": 30},
         "design": {"test_look_budget": 2}},
        [1, 2, 3], ["lega", "standard"], None,
        known_variants={"lega", "standard"})
    assert custom["test_look_budget"] == 2
    check = {c["id"]: c for c in custom["checks"]}["test_look_budget"]
    assert check["status"] == "pass" and "2" in check["detail"]

    bad = build_design_plan(
        {"ruler": {"ctx_lens": [256], "n_samples": 30},
         "design": {"test_look_budget": 0}},
        [1, 2, 3], ["lega", "standard"], None,
        known_variants={"lega", "standard"})
    assert bad["test_look_budget"] == 1
    check = {c["id"]: c for c in bad["checks"]}["test_look_budget"]
    assert check["status"] == "fail"
    print("  [ok] 设计计划：test look 预算默认/显式/非法值均显形")


def test_design_plan_accepts_seed_zero_and_caps_seed_range():
    """种子 0 与 utils.seed_int 同口径合法；负数/超出 2^32-1 必须拒绝。"""
    plan = _plan(seeds=(0, 1, 2))
    assert plan["inputs"]["seeds"] == [0, 1, 2]
    for bad in (-1, 2 ** 32, 10 ** 400):
        try:
            _plan(seeds=(0, bad))
        except ValueError:
            pass
        else:
            raise AssertionError(f"应拒绝 seed={bad!r}")
    print("  [ok] 设计计划：seed 0 合法，越界种子显式拒绝")


def test_design_plan_rejects_malformed_inputs():
    """非法种子/变体/配置必须在训练前显式失败。"""
    for seeds, variants in (([], ["lega"]), ([1, 1], ["lega"]),
                            ([1], []), ([1], ["lega", "lega"])):
        try:
            build_design_plan({}, seeds, variants, None,
                              known_variants={"lega"})
        except ValueError:
            pass
        else:
            raise AssertionError(f"应拒绝 seeds={seeds}, variants={variants}")
    # 配置里的设计网格非法时不能静默跳过检查。
    try:
        build_design_plan({"design": {"ppl_paired_sd_rel_grid": [0.1, -1]}},
                          [1, 2, 3], ["lega"], None,
                          known_variants={"lega"})
    except ValueError:
        pass
    else:
        raise AssertionError("负的设计网格必须拒绝")
    # 网格项必须逐项严格校验，不能把 bool/字符串/NaN/超大整数隐式转成 float。
    for bad_grid in ([True], ["0.1"], [float("nan")], [10 ** 400]):
        try:
            build_design_plan(
                {"design": {"ppl_paired_sd_rel_grid": bad_grid}},
                [1, 2, 3], ["lega"], None, known_variants={"lega"})
        except ValueError:
            pass
        else:
            raise AssertionError(f"非法设计网格必须拒绝：{bad_grid!r}")
    print("  [ok] 设计计划：非法输入显式失败")


def test_build_summary_downgrades_confirmatory_on_missing_planned_seed():
    """端到端：汇总读入先验计划，缺计划内种子的 H1 降级为探索性。"""
    from unittest import mock
    import run_all

    tmp = tempfile.mkdtemp()
    try:
        seeds = [1, 2, 3]
        variants = ["lega", "standard"]
        reg = {"hypotheses": [
            {"id": "H1", "claim": "LEGA 优于 standard", "metric": "PPL",
             "candidate": "lega", "reference": "standard",
             "direction": "candidate_better"}]}
        plan = build_design_plan(
            {"ruler": {"ctx_lens": [256], "n_samples": 30}},
            seeds, variants, reg, known_variants=set(variants))
        save_design_plan(plan, os.path.join(tmp, "design_plan.json"))
        _write_json(os.path.join(tmp, "repro.json"), {
            "variants": variants, "seeds": seeds, "smoke": False,
            "config_sha256": plan["inputs"]["config_sha256"],
            "design": {"stages": {"ruler": False, "analysis": False}},
        })
        for s in seeds:
            _write_json(os.path.join(tmp, f"eval_standard_seed{s}.json"), {
                "variant": "standard", "seed": s, "val_ppl": 10.0 + 0.1 * s,
                "val_loss": 2.3, "n_params": 100, "split": "val"})
        for s in (1, 2):  # seed3 缺失
            _write_json(os.path.join(tmp, f"eval_lega_seed{s}.json"), {
                "variant": "lega", "seed": s, "val_ppl": 9.0 + 0.1 * s,
                "val_loss": 2.2, "n_params": 100, "split": "val"})
        intact = {"status": "intact", "confirmatory_valid": True,
                  "problems": [], "registry_sha256": "a" * 64,
                  "n_hypotheses": 1, "baseline_document_sha256": "b" * 64,
                  "chain_length": 0}
        with mock.patch("lega.pipeline.report.summary.load_registered_hypotheses",
                        return_value=reg), \
                mock.patch("lega.pipeline.report.summary.verify_registration",
                           return_value=intact):
            out = run_all.build_summary(tmp)
        md = out["markdown"]
        assert "实验设计与分析人群" in md
        assert "lega/seed3" in md
        sig = json.load(open(os.path.join(tmp, "significance.json"),
                             encoding="utf-8"))
        h1 = next(h for h in sig["registered_hypotheses"] if h["id"] == "H1")
        assert h1["design_population_ok"] is False
        assert h1["confirmatory_valid"] is False
        assert "分析人群不完整" in h1["confirmatory_note"]
        conf = json.load(open(os.path.join(tmp, "design_conformance.json"),
                              encoding="utf-8"))
        assert conf["checked"] is True and conf["in_plan"] is False
        assert "lega/seed3" in " ".join(conf["problems"])
        # 计划/实测灵敏度也必须进入 significance.json，供程序扫描。
        assert sig["design_plan"]["plan_digest"] == plan["plan_digest"]
        assert isinstance(sig["observed_design_sensitivity"], list)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] 端到端：缺计划内种子 → H1 确认性降级 + 产物落盘")


def test_missing_registered_seeds_trigger_global_design_gate():
    """计划只用 3 个登记种子时，即使 3×变体人口齐全也必须全局降级。"""
    reg = {"scope": {"seeds": [1, 2, 3, 4, 5]},
           "hypotheses": [
               {"id": "H1", "metric": "PPL", "candidate": "lega",
                "reference": "standard"}]}
    plan = build_design_plan(
        {"ruler": {"ctx_lens": [256], "n_samples": 30}},
        [1, 2, 3], ["lega", "standard"], reg,
        known_variants={"lega", "standard"})
    assert plan["status"] == "fail"
    runs = {v: {s: {"eval": {}, "ruler": {}, "analysis": {}}
                for s in (1, 2, 3)} for v in ("lega", "standard")}
    repro = {"variants": plan["inputs"]["variants"],
             "seeds": plan["inputs"]["seeds"], "smoke": False}
    conf = design_conformance(plan, repro, runs)
    assert conf["checked"] is True
    assert conf["global_plan_gate"] is True
    assert conf["population"]["complete_eval"] is True  # 3×2 都齐全
    hyps = [{"id": "H1", "metric": "PPL", "candidate": "lega",
             "reference": "standard", "confirmatory_valid": True}]
    out = apply_design_gate(hyps, conf, plan)
    assert out[0]["confirmatory_valid"] is False
    assert out[0]["design_population_ok"] is False
    assert "预注册种子" in out[0]["confirmatory_note"]
    print("  [ok] 设计门：少跑预注册种子触发全局降级（人口齐全也不例外）")


def test_plan_digest_ignores_machine_specific_config_path():
    """同一设计在不同工作目录/路径下摘要一致；内容变化仍会变。"""
    cfg = {"ruler": {"ctx_lens": [256], "n_samples": 30}}
    reg = {"hypotheses": [{"id": "H1", "metric": "PPL",
                           "candidate": "lega", "reference": "standard"}]}
    a = build_design_plan(cfg, [1, 2, 3], ["lega", "standard"], reg,
                          config_file="/mnt/a/configs/small.json",
                          known_variants={"lega", "standard"})
    b = build_design_plan(cfg, [1, 2, 3], ["lega", "standard"], reg,
                          config_file="/home/other/small.json",
                          known_variants={"lega", "standard"})
    assert a["plan_digest"] == b["plan_digest"]
    c = build_design_plan(cfg, [1, 2, 3], ["lega", "standard"], reg,
                          config_sha256="f" * 64,
                          known_variants={"lega", "standard"})
    assert c["plan_digest"] != a["plan_digest"]
    print("  [ok] 计划摘要：忽略机器路径、对配置内容敏感")


def test_design_plan_history_preserves_redesign_events():
    """计划历史：同 digest 去重、不同 digest 追加、损坏行跳过。"""
    from lega.design import load_design_plan_history
    tmp = tempfile.mkdtemp()
    try:
        out = os.path.join(tmp, "design_plan.json")
        p1 = _plan()
        p2 = _plan()
        save_design_plan(p1, out)
        save_design_plan(p2, out)  # 同 digest：不重复追加
        hist = load_design_plan_history(tmp)
        assert len(hist) == 1, hist
        p3 = _plan(seeds=(1, 2, 3, 4, 5, 6))
        save_design_plan(p3, out)
        hist = load_design_plan_history(out)
        assert len(hist) == 2
        assert hist[-1]["plan_digest"] == p3["plan_digest"]
        with open(os.path.join(tmp, "design_plan_history.jsonl"),
                  "a", encoding="utf-8") as f:
            f.write("not-json\n")
        assert len(load_design_plan_history(tmp)) == 2
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] 计划历史：去重、追加、损坏行容错")


def test_design_plan_checks_context_span_for_length_claims():
    """只有一档/跨度过窄的 ctx 不能支撑长度校准主张，计划必须告警。"""
    def check(ctx):
        p = build_design_plan(
            {"ruler": {"ctx_lens": ctx, "n_samples": 30}},
            [1, 2, 3], ["lega"], None, known_variants={"lega"})
        return {c["id"]: c for c in p["checks"]}["ruler_context_span"]

    assert check([256])["status"] == "warn"
    assert check([512, 600])["status"] == "warn"
    assert check([256, 512, 1024])["status"] == "pass"
    print("  [ok] 设计计划：上下文档位跨度不足时告警")


def test_design_scope_config_keys_match_pipeline():
    """设计层与流水线的预注册配置规模字段不能各自漂移。"""
    from lega.design import SCOPE_CONFIG_KEYS
    from lega.pipeline._shared import _SCOPE_CONFIG_KEYS
    assert tuple(SCOPE_CONFIG_KEYS) == tuple(_SCOPE_CONFIG_KEYS)
    print("  [ok] 设计/流水线 scope.config 字段一致")


def test_design_plan_validates_ruler_context_and_model_capacity():
    """RULER 上下文必须是正整数，且不能超过 block_size/训练上下文。"""
    def checks(ctx, **extra):
        cfg = {"ruler": {"ctx_lens": ctx, "n_samples": 30}, **extra}
        plan = build_design_plan(
            cfg, [1, 2, 3], ["lega"], None, known_variants={"lega"})
        return {c["id"]: c for c in plan["checks"]}

    ok = checks([256, 512], block_size=512, train_ctx=512)
    assert ok["ruler_context_values"]["status"] == "pass"
    assert ok["model_context"]["status"] == "pass"
    assert ok["ruler_context_model"]["status"] == "pass"
    assert checks([256, 512.5], block_size=512)["ruler_context_values"][
        "status"] == "fail"
    assert checks([True, 256], block_size=512)["ruler_context_values"][
        "status"] == "fail"
    assert checks([256, 1024], block_size=512, train_ctx=512)[
        "ruler_context_model"]["status"] == "fail"
    assert checks([256, 512], block_size=512, train_ctx=256)[
        "ruler_context_model"]["status"] == "warn"
    assert checks([256, 512], block_size=256, train_ctx=512)[
        "model_context"]["status"] == "fail"
    print("  [ok] 设计计划：RULER 上下文整数/模型容量检查")


def test_design_plan_accepts_length_null_defaults_and_string_lists():
    """length_null 缺省长度/模式由运行时生成，不能在设计层误报 fail。"""
    cfg = {"block_size": 128, "train_ctx": 128,
           "ruler": {"ctx_lens": [64, 128], "n_samples": 30},
           "length_null": {"enabled": True}}
    plan = build_design_plan(
        cfg, [1, 2, 3], ["lega"], None, known_variants={"lega"})
    by = {c["id"]: c for c in plan["checks"]}
    assert by["length_null_coverage"]["status"] == "info"
    cfg["length_null"] = {"enabled": True, "lengths": "64, 128",
                          "modes": "random,repeated"}
    plan2 = build_design_plan(
        cfg, [1, 2, 3], ["lega"], None, known_variants={"lega"})
    by2 = {c["id"]: c for c in plan2["checks"]}
    assert by2["length_null_coverage"]["status"] == "pass"
    print("  [ok] length_null：默认/字符串长度与模式不被误判")


def test_design_plan_checks_data_budget_and_registered_scope():
    """开跑前用 data_report 核对 token/窗口，并核对预注册配置/分词器。"""
    tmp = tempfile.mkdtemp()
    try:
        declared = os.path.join(tmp, "registered.json")
        actual = os.path.join(tmp, "actual.json")
        base = {"d_model": 64, "n_layers": 2, "n_heads": 2,
                "ffn_mult": 2, "block_size": 128, "train_ctx": 128,
                "steps": 10}
        _write_json(declared, base)
        _write_json(actual, base)
        cfg = dict(base, batch_size=2, grad_accum=1, eval_iters=5,
                   ruler={"ctx_lens": [64, 128], "n_samples": 30},
                   length_null={"enabled": True, "lengths": [64, 128],
                                "modes": ["random"]})
        reg = {"scope": {"config": declared, "data": "GPT-2 BPE"},
               "hypotheses": []}
        data = {
            "accounting": {"train": {"tokens": 1000},
                           "val": {"tokens": 100000},
                           "test": {"tokens": 100000}},
            "tokenizer": {"kind": "gpt2"},
        }
        plan = build_design_plan(
            cfg, [1, 2, 3], ["lega"], reg, config_file=actual,
            data_report=data, known_variants={"lega"})
        by = {c["id"]: c for c in plan["checks"]}
        assert by["scope_config"]["status"] == "pass"
        assert by["scope_data"]["status"] == "pass"
        assert by["data_budget"]["status"] == "info"
        assert by["eval_budget"]["status"] == "pass"
        assert by["length_null_coverage"]["status"] == "pass"
        budget = plan["data_budget"]
        assert budget["planned_tokens_seen_per_seed"] == 2560
        assert budget["token_touches_per_train_token"] == 2.56
        assert budget["val_windows_available"] == (100000 - 1) // 128

        bad_cfg = os.path.join(tmp, "bad.json")
        _write_json(bad_cfg, dict(base, d_model=32))
        bad = build_design_plan(
            cfg, [1, 2, 3], ["lega"], reg, config_file=bad_cfg,
            data_report=data, known_variants={"lega"})
        bad_by = {c["id"]: c for c in bad["checks"]}
        assert bad_by["scope_config"]["status"] == "fail"
        assert "d_model" in bad_by["scope_config"]["detail"]

        char_data = {
            "accounting": data["accounting"],
            "tokenizer": {"kind": "char"},
        }
        char = build_design_plan(
            cfg, [1, 2, 3], ["lega"], reg, config_file=actual,
            data_report=char_data, known_variants={"lega"})
        char_by = {c["id"]: c for c in char["checks"]}
        assert char_by["scope_data"]["status"] == "fail"

        tiny_val = {
            "accounting": {"train": {"tokens": 1000},
                           "val": {"tokens": 100},
                           "test": {"tokens": 100000}},
            "tokenizer": {"kind": "gpt2"},
        }
        small = build_design_plan(
            cfg, [1, 2, 3], ["lega"], reg, config_file=actual,
            data_report=tiny_val, known_variants={"lega"})
        small_by = {c["id"]: c for c in small["checks"]}
        assert small_by["eval_budget"]["status"] == "fail"

        empty_train = {
            "accounting": {"train": {"tokens": 0},
                           "val": {"tokens": 100000},
                           "test": {"tokens": 100000}},
            "tokenizer": {"kind": "gpt2"},
        }
        empty = build_design_plan(
            cfg, [1, 2, 3], ["lega"], reg, config_file=actual,
            data_report=empty_train, known_variants={"lega"})
        empty_by = {c["id"]: c for c in empty["checks"]}
        assert empty_by["data_budget"]["status"] == "fail"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] 设计计划：数据/评测预算与预注册 scope 检查")


def test_design_gate_downgrades_only_affected_endpoint_on_capacity_failure():
    """RULER 上下文超容量只降级 RULER 假设，不连带 PPL 对比。"""
    cfg = {"ruler": {"ctx_lens": [256, 512], "n_samples": 30},
           "block_size": 256, "train_ctx": 256}
    reg = {"hypotheses": [
        {"id": "H1", "metric": "PPL", "candidate": "lega",
         "reference": "standard", "direction": "candidate_better"},
        {"id": "H4", "metric": "RULER平均", "candidate": "lega",
         "reference": "standard", "direction": "candidate_better"},
    ]}
    plan = build_design_plan(
        cfg, [1, 2, 3], ["lega", "standard"], reg,
        known_variants={"lega", "standard"})
    by = {c["id"]: c for c in plan["checks"]}
    assert by["ruler_context_model"]["status"] == "fail"
    runs = {
        v: {s: {"eval": {}, "ruler": {}, "analysis": {}}
            for s in (1, 2, 3)}
        for v in ("lega", "standard")
    }
    repro = {"variants": plan["inputs"]["variants"],
             "seeds": plan["inputs"]["seeds"], "smoke": False}
    conf = design_conformance(plan, repro, runs)
    assert conf["global_plan_gate"] is False
    assert conf["endpoint_plan_failures"]["ruler"][0]["id"] == \
        "ruler_context_model"
    hyps = [
        {"id": "H1", "metric": "PPL", "candidate": "lega",
         "reference": "standard", "confirmatory_valid": True},
        {"id": "H4", "metric": "RULER平均", "candidate": "lega",
         "reference": "standard", "confirmatory_valid": True},
    ]
    out = apply_design_gate(hyps, conf, plan)
    out_by = {h["id"]: h for h in out}
    assert out_by["H1"]["confirmatory_valid"] is True
    assert out_by["H1"]["design_population_ok"] is True
    assert out_by["H4"]["confirmatory_valid"] is False
    assert "ruler" in out_by["H4"]["confirmatory_note"]
    print("  [ok] 设计门：端点级失败只降级对应端点")


def test_analysis_population_ignores_variants_without_analysis_support():
    """standard 不产出机制分析：不得因此把它算成“分析人群缺失”。"""
    reg = {"hypotheses": [{"id": "H1", "metric": "PPL",
                           "candidate": "lega", "reference": "standard"}]}
    plan = build_design_plan(
        {"ruler": {"ctx_lens": [256], "n_samples": 30}},
        [1, 2, 3], ["lega", "standard"], reg,
        known_variants={"lega", "standard"},
        analysis_variants=["lega"])
    runs = {
        "lega": {s: {"eval": {}, "ruler": {}, "analysis": {}}
                 for s in (1, 2, 3)},
        "standard": {s: {"eval": {}, "ruler": {}}
                     for s in (1, 2, 3)},
    }
    pop = analysis_population(plan, runs)
    assert pop["complete_eval"] is True
    assert pop["complete_ruler"] is True
    assert pop["complete_analysis"] is True
    assert pop["missing_analysis"] == []
    repro = {"variants": plan["inputs"]["variants"],
             "seeds": plan["inputs"]["seeds"], "smoke": False}
    conf = design_conformance(plan, repro, runs)
    assert conf["checked"] is True and conf["in_plan"] is True
    print("  [ok] 分析人群：不为无分析能力的标准变体伪造缺失")


def test_design_power_helpers_handle_nonpositive_z_sum():
    """目标功效低到 z 和 <= 0 时，反推量不能除零或返回负值。"""
    alpha = 0.05
    z_alpha = statistics.NormalDist().inv_cdf(1.0 - alpha / 2.0)
    at_alpha = statistics.NormalDist().cdf(-z_alpha)
    for power in (at_alpha, at_alpha / 2.0):
        assert min_detectable_effect(1.0, 10, alpha=alpha,
                                     power=power) == 0.0
        assert max_sd_for_effect(1.0, 10, alpha=alpha, power=power) is None
        assert required_pairs(1.0, 1.0, alpha=alpha, power=power) == 2
        assert required_pairs(1e-300, 1e300, alpha=alpha, power=power) == 2
        assert mcnemar_mde(10, 0.1, alpha=alpha, power=power) == 0.0
        assert max_discordance_for_effect(
            1.0, 10, alpha=alpha, power=power) == 1.0
        from lega.utils import min_detectable_effect as utils_mde
        assert utils_mde(1.0, 10, alpha=alpha, power=power) == 0.0
    print("  [ok] 设计功效函数：z 和 <= 0 时不除零/不返回负值")


def test_design_power_helpers_tolerate_counts_beyond_float_range():
    """合法的大整数样本数不能让功效/MDE 计算抛 OverflowError。"""
    huge = 10 ** 400  # √n = 1e200 仍可表示
    assert paired_t_power(0.01, 0.01, huge) == 1.0
    assert abs(paired_t_power(0.0, 0.01, huge) - 0.05) < 1e-12
    mde = min_detectable_effect(0.01, huge)
    assert mde is not None and 0.0 < mde < 1e-190, mde
    mde_huge_sd = min_detectable_effect(1e308, 4)
    assert mde_huge_sd is not None and 1e308 < mde_huge_sd < 2e308, mde_huge_sd
    mde_mixed = min_detectable_effect(1e308, huge)
    assert mde_mixed is not None and 1e108 < mde_mixed < 1e109, mde_mixed
    sd_max = max_sd_for_effect(0.1, huge)
    assert sd_max is not None and sd_max > 1e190, sd_max
    mc = mcnemar_power(huge, 0.1, 0.01)
    assert mc["method"] == "normal_approx" and mc["power"] == 1.0
    assert mc["expected_discordant"] is None
    assert mcnemar_power(huge, 0.1, 0.0)["power"] < 0.06
    mde_bin = mcnemar_mde(huge, 0.1)
    assert mde_bin is not None and 0.0 < mde_bin < 1e-190, mde_bin
    assert max_discordance_for_effect(0.1, huge) == 1.0
    # 极大效应/极大计数不能让平方或乘法中间结果溢出。
    assert max_discordance_for_effect(1e200, 2) == 1.0
    # 极小但合法的效应平方后会下溢为 0；所需样本数不可达，不能因
    # effect**2 == 0 抛 ZeroDivisionError。
    assert mcnemar_required_samples(5e-324, 0.1) is None
    assert required_pairs(5e-324, 1.0) is None
    assert required_pairs(1e308, 1e308) == 8
    sd_max_huge = max_sd_for_effect(1e308, 4)
    assert sd_max_huge is not None and 5e307 < sd_max_huge < 1e308, sd_max_huge

    # alpha 小到 1-alpha/2 在 float 中舍入为 1 时，正态临界值不能抛错；
    # 与 utils.min_detectable_effect 保持同一口径。
    from lega.utils import min_detectable_effect as utils_mde
    for tiny_alpha in (1e-16, 1e-300, 5e-324):
        assert math.isfinite(paired_t_power(1.0, 1.0, 10, alpha=tiny_alpha))
        assert math.isfinite(min_detectable_effect(
            1.0, 10, alpha=tiny_alpha))
        assert math.isfinite(max_sd_for_effect(1.0, 10, alpha=tiny_alpha))
        assert required_pairs(1.0, 1.0, alpha=tiny_alpha) is not None
        assert math.isfinite(mcnemar_power(
            1000, 0.1, 0.01, alpha=tiny_alpha)["power"])
        assert math.isfinite(mcnemar_mde(1000, 0.1, alpha=tiny_alpha))
        assert abs(min_detectable_effect(1.0, 10, alpha=tiny_alpha)
                   - utils_mde(1.0, 10, alpha=tiny_alpha)) < 1e-15
    huge_burden = multiplicity_alpha(0.05, 10 ** 400, 1)
    assert huge_burden["representable"] is False
    assert huge_burden["alpha_effective"] == 0.05

    astronomical = 10 ** 1000  # √n 本身已超出 float 上限
    assert paired_t_power(0.01, 0.01, astronomical) == 1.0
    assert abs(paired_t_power(0.0, 0.01, astronomical) - 0.05) < 1e-12
    assert min_detectable_effect(0.01, astronomical) is None
    assert max_sd_for_effect(0.1, astronomical) is None
    assert mcnemar_power(astronomical, 0.1, 0.01)["power"] == 1.0
    assert mcnemar_mde(astronomical, 0.1) is None
    assert max_discordance_for_effect(0.1, astronomical) == 1.0
    print("  [ok] 设计功效函数：超 float 范围的合法大 n 不抛错/不产出 inf")


def test_noncentral_t_power_is_more_conservative_for_small_seeds():
    """3~5 个种子时，非中心 t 仿真必须低于正态近似（后者高估功效）。"""
    from lega.design import paired_t_power_nct
    for n in (3, 5):
        normal = paired_t_power(0.01, 0.01, n)
        out = paired_t_power_nct(0.01, 0.01, n, n_sim=20000, seed=3)
        assert out["method"] == "noncentral_t_simulation"
        assert 0.0 <= out["power"] <= 1.0
        assert out["power"] < normal, (n, out["power"], normal)
        neg = paired_t_power_nct(-0.01, 0.01, n, n_sim=20000, seed=3)
        assert neg["power"] == out["power"]
        assert neg["ncp"] == out["ncp"]
    zero = paired_t_power_nct(0.0, 0.01, 5, n_sim=5000, seed=1)
    assert abs(zero["power"] - 0.05) < 1e-12


def test_run_order_is_randomized_blocked_and_reproducible():
    """run 顺序默认按种子区块随机化：覆盖完整、可复算、旧顺序显式告警。"""
    p1 = _plan()
    ro = p1["run_order"]
    assert ro["mode"] == "randomized_blocked"
    assert ro["n_cells"] == 15 and ro["n_blocks"] == 5
    assert ro["seed_source"] == "derived_from_plan_inputs"
    assert validate_run_order(ro, p1["inputs"]["variants"],
                              p1["inputs"]["seeds"]) == []
    for seed in p1["inputs"]["seeds"]:
        block = [c["variant"] for c in ro["cells"] if c["seed"] == seed]
        assert sorted(block) == sorted(p1["inputs"]["variants"]), block
    assert [c["position"] for c in ro["cells"]] == list(range(1, 16))
    # 同一设计重复生成必须得到同一顺序与摘要（不能依赖时间/机器）。
    p2 = _plan()
    assert p2["run_order"]["cells"] == ro["cells"]
    assert p2["plan_digest"] == p1["plan_digest"]

    cfg = {"ruler": {"ctx_lens": [256, 512], "n_samples": 30},
           "design": {"alpha": 0.05, "power": 0.8,
                      "run_order_seed": 12345}}
    p3 = build_design_plan(cfg, [1, 2, 3], ["lega", "standard", "qgate"],
                           None, known_variants={"lega", "standard", "qgate"})
    assert p3["run_order"]["seed_source"] == "configured"
    assert p3["run_order"]["order_seed"] == 12345
    assert validate_run_order(p3["run_order"],
                              ["lega", "standard", "qgate"], [1, 2, 3]) == []

    cfg["design"]["run_order"] = "variant_major"
    p4 = build_design_plan(cfg, [1, 2, 3], ["lega", "standard", "qgate"],
                           None, known_variants={"lega", "standard", "qgate"})
    check = {c["id"]: c for c in p4["checks"]}["run_order"]
    assert check["status"] == "warn" and "混淆" in check["detail"]
    assert [(c["variant"], c["seed"]) for c in p4["run_order"]["cells"]] == \
        [(v, s) for v in ["lega", "standard", "qgate"] for s in (1, 2, 3)]

    cfg["design"]["run_order"] = "bogus"
    try:
        build_design_plan(cfg, [1, 2, 3], ["lega"], None,
                          known_variants={"lega"})
    except ValueError:
        pass
    else:
        raise AssertionError("未知 run_order 模式必须拒绝")

    assert build_run_order(["a", "b"], [1, 2])["mode"] == "randomized_blocked"
    print("  [ok] run 顺序：随机化区组、覆盖完整、可复算、旧顺序告警")


def test_validate_run_order_rejects_noncontiguous_seed_blocks():
    """区块模式必须按 seed 连续；仅改 mode 的 variant_major 计划应被拒绝。"""
    variants, seeds = ["a", "b"], [1, 2]
    sm = build_run_order(variants, seeds, mode="seed_major")
    assert validate_run_order(sm, variants, seeds) == []
    vm = build_run_order(variants, seeds, mode="variant_major")
    for bogus_mode in ("randomized_blocked", "seed_major"):
        tampered = dict(vm, mode=bogus_mode)
        problems = validate_run_order(tampered, variants, seeds)
        assert any("种子区块必须连续" in p for p in problems), problems
    # 顺序正确但仅把 seed_major 改标为 randomized_blocked 无法从字段上
    # 区分随机化与固定顺序；这里只保证真正的变体主序会被拒绝。
    print("  [ok] run 顺序：种子区块不连续的计划不能伪装成区组设计")


def test_validate_run_order_rejects_stale_positions_and_metadata():
    """position/block/计数必须与实际 cells 一致，不能只做表面排列检查。"""
    variants, seeds = ["a", "b"], [1, 2]
    ro = build_run_order(variants, seeds, mode="randomized_blocked")
    assert validate_run_order(ro, variants, seeds) == []
    vm = build_run_order(variants, seeds, mode="variant_major")
    assert len(vm["blocks"]) == vm["n_blocks"]
    assert [b["variants"] for b in vm["blocks"]] == [variants, variants]
    for key, value in (("order_seed", True), ("order_seed", -1),
                       ("order_seed", ro["order_seed"] + 1),
                       ("order_seed", None)):
        problems = validate_run_order(dict(ro, **{key: value}),
                                      variants, seeds)
        assert any("order_seed" in p for p in problems), (key, value, problems)
    for value in ("manual", None, 1):
        problems = validate_run_order(dict(ro, seed_source=value),
                                      variants, seeds)
        assert any("seed_source" in p for p in problems), (value, problems)
    bad_blocks = [dict(b) for b in ro["blocks"]]
    bad_blocks[0]["variants"] = list(reversed(bad_blocks[0]["variants"]))
    problems = validate_run_order(dict(ro, blocks=bad_blocks),
                                  variants, seeds)
    assert any("blocks[0].variants" in p for p in problems), problems
    problems = validate_run_order(dict(ro, blocks=[]), variants, seeds)
    assert any("run_order.blocks" in p for p in problems), problems
    cells = [dict(c) for c in ro["cells"]]
    first = [i for i, c in enumerate(cells) if c["seed"] == 1]
    i, j = first[0], first[1]
    cells[i], cells[j] = cells[j], cells[i]
    problems = validate_run_order(dict(ro, cells=cells), variants, seeds)
    assert any("position 必须按 cells 列表顺序" in p for p in problems), problems
    for key, value in (("n_cells", len(cells) + 1),
                       ("n_blocks", len(seeds) + 1)):
        problems = validate_run_order(dict(ro, **{key: value}),
                                      variants, seeds)
        assert any(key in p for p in problems), (key, problems)
    bad_block = [dict(c) for c in ro["cells"]]
    bad_block[0]["block"] = 999
    problems = validate_run_order(dict(ro, cells=bad_block), variants, seeds)
    assert any("block 必须与 seed 一致" in p for p in problems), problems
    print("  [ok] run 顺序：position/block/计数与实际 cells 不一致会被拒绝")


def test_mechanism_endpoint_sensitivity_and_declared_power():
    """H6/H7 机制端点也必须先验规划，不能只规划 PPL/RULER。"""
    reg = {"hypotheses": [
        {"id": "H6", "metric": "AUROC差(logit−仅查询)", "candidate": "lega",
         "reference": "仅查询项基线", "direction": "candidate_better"},
        {"id": "H7", "metric": "AP差(logit−仅查询)", "candidate": "lega",
         "reference": "仅查询项基线", "direction": "candidate_better"}]}
    p = _plan(registered=reg)
    for key, metric in (("auroc", "AUROC差(logit−仅查询)"),
                        ("ap", "AP差(logit−仅查询)")):
        ep = p["endpoints"][key]
        assert ep["metric"] == metric
        assert ep["sesoi_absolute"] == 0.01
        row = next(r for r in ep["sensitivity"]
                   if abs(r["paired_sd_absolute"] - 0.01) < 1e-12)
        assert row["mde_absolute_confirmatory"] > row["mde_absolute"]
        assert (row["power_for_sesoi_confirmatory"]
                < row["power_for_sesoi"])
        assert row["required_seeds_for_sesoi_confirmatory"] >= \
            row["required_seeds_for_sesoi"]
    assert any(c["id"] == "mechanism_endpoint_variants"
               and c["status"] == "pass" for c in p["checks"])

    cfg = {"ruler": {"ctx_lens": [256, 512], "n_samples": 30},
           "design": {"alpha": 0.05, "power": 0.8,
                      "auroc_paired_sd": 0.1},
           "analysis": {"n_examples": 40}}
    cfg["design"]["mechanism_sesoi"] = 0.02
    custom = build_design_plan(cfg, [1, 2, 3], ["lega", "standard"], reg,
                               known_variants={"lega", "standard"})
    assert custom["endpoints"]["auroc"]["sesoi_absolute"] == 0.02
    assert custom["endpoints"]["ap"]["sesoi_absolute"] == 0.02
    cfg["design"].pop("mechanism_sesoi")
    bad = build_design_plan(cfg, [1, 2, 3], ["lega", "standard"], reg,
                            known_variants={"lega", "standard"})
    check = {c["id"]: c for c in bad["checks"]}["auroc_declared_power"]
    assert check["status"] == "fail" and "功效不足" in check["detail"]

    # 有机制假设但候选不在可分析子集：必须 fail，而不是等跑完才发现。
    no_analysis = build_design_plan(
        {"ruler": {"ctx_lens": [256, 512], "n_samples": 30}},
        [1, 2, 3], ["lega", "standard"], reg,
        known_variants={"lega", "standard"}, analysis_variants=["standard"])
    check = {c["id"]: c for c in no_analysis["checks"]}[
        "mechanism_endpoint_variants"]
    assert check["status"] == "fail" and "lega" in check["detail"]
    print("  [ok] 机制端点：确认性灵敏度、声明功效不足/分析人群缺失均显形")


def test_render_design_section_includes_run_order_and_mechanism():
    """汇总页必须渲染 run 顺序与机制端点灵敏度。"""
    md = "\n".join(render_design_section(_plan()))
    assert "run 顺序" in md and "randomized_blocked" in md
    assert "机制端点先验灵敏度" in md
    assert "AUROC" in md and "AP" in md
    print("  [ok] 汇总页：run 顺序与机制端点灵敏度")
