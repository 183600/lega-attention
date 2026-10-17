"""对比/消融协议测试：Holm 校正结论、组件有效性判定与受控变量审计。

覆盖四条关键科研口径：
  * 多重比较校正后结论才有效（避免 8 变体两两对比下的假阳性）；
  * “各组件均有用”必须有显著统计证据，组合跑全本身不算证据；
  * 小样本下 t 检验结论必须与置换检验互证，冲突要显式暴露而不是静默采信；
  * 受控变量审计除超参数外还核对数据指纹/代码版本/训练步数。
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from unittest import mock

from run_all import (ALPHA, MIN_SEEDS, _apply_holm, _auroc_diff_tests,
                      _cmd_env, _effect_status, _enrich_paired, _rel_pct,
                      _resolve_jobs, _stat, AUROC_DIFF_METRIC,
                      AUROC_DIFF_REFERENCE, build_summary, component_evidence,
                      evaluate_registered_hypotheses, load_registered_hypotheses,
                      run_many, tradeoff_verdict, verify_controlled_setup)


def _mk(cand, ref, metric, mean_diff, p, n=5, lower=True):
    return {"candidate": cand, "reference": ref, "metric": metric,
            "mean_diff": mean_diff, "std_diff": 0.1, "p": p, "p_adj": None,
            "n": n, "lower_is_better": lower, "d_z": 1.0}


def _write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)


def test_resolve_jobs_defaults():
    class _A:
        jobs = None
        wandb = False
        device = "cpu"

    assert _resolve_jobs(_A()) == min(4, os.cpu_count() or 1)
    _A.wandb = True
    assert _resolve_jobs(_A()) == 1
    _A.wandb = False
    _A.device = "cuda"
    assert _resolve_jobs(_A()) == 1
    _A.device = "cpu"
    _A.jobs = 7
    assert _resolve_jobs(_A()) == 7
    print("  [ok] _resolve_jobs（显式 jobs / CUDA / wandb 默认值）")


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
        # A：去一法 Full−A 显著变差（PPL 变大）→ 有证据
        _mk("lega_no_query", "lega", "PPL", 0.4, 0.005),
        _mk("qgate", "standard", "PPL", -0.01, 0.7),
        # B：两个方向都不显著 → 无证据
        _mk("lega_no_conc", "lega", "PPL", 0.05, 0.5),
        _mk("lega_conc_only", "standard", "PPL", -0.02, 0.6),
        # C：方向相反（去掉反而更好），即使显著也不算“该组件有用”
        _mk("lega_no_len", "lega", "PPL", -0.1, 0.001),
        _mk("lega_len_only", "standard", "PPL", 0.03, 0.4),
    ]
    _apply_holm(tests)
    comp = {c["component"]: c for c in component_evidence(tests)}
    assert comp["A 查询项"]["supported"] == ["PPL"]
    assert comp["A 查询项"]["verdict"] == "有证据支持"
    assert comp["B 集中度证据"]["verdict"] == "证据不足"
    # C 的去一法方向相反：不构成有用证据
    assert comp["C 长度校准"]["verdict"] == "证据不足"
    assert "方向相反" in comp["C 长度校准"]["loo"]["PPL"]["reason"]
    print("  [ok] component_evidence（去一法/全因子证据、方向与显著性门槛）")


def test_tradeoff_verdict_rejects_bad_deals():
    """效率-精度权衡判定：不显著涨点说服力有限；大增幅小收益被拒绝。"""
    # 不显著 + 增幅 >10% → 不可行（微小且不显著的涨点 + 大增幅）
    v = tradeoff_verdict(benefit_pct=0.3, p_adj=0.4, cost_pct=30.0)
    assert v.startswith("不可行"), v
    # 不显著 + 增幅小 → 说服力有限
    v = tradeoff_verdict(benefit_pct=0.3, p_adj=0.4, cost_pct=2.0)
    assert v.startswith("说服力有限"), v
    # 显著但增幅 >5×收益 → 收益-增幅比不利（0.5% 涨点 vs 10 倍计算不可行）
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


def test_rel_pct_and_eff_cells():
    """相对变化计算：缺失与除零安全返回 None。"""
    assert _rel_pct(100.0, 110.0) == 10.0
    assert _rel_pct(50.0, 45.0) == -10.0
    assert _rel_pct(None, 10.0) is None
    assert _rel_pct(10.0, None) is None
    assert _rel_pct(0.0, 10.0) is None
    # _eff_cell 读取 latency 条目的效率字段
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
        assert "效率-精度权衡" in out["markdown"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] build_summary（latency 缺变体条目时安全跳过）")


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
    # p_perm 缺失（旧格式 dict）：行为与原先完全一致，不崩不标
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
        with mock.patch("run_all.load_registered_hypotheses",
                        return_value=reg):
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
        # 每条 pair 检验都带稳健性字段
        for t in sig["pairs"]:
            assert "p_perm" in t and "mean_diff_ci95" in t and "mde" in t
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] build_summary（注册假设/置换/CI/MDE 进入汇总页与 JSON）")
