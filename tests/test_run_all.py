"""对比/消融协议测试：Holm 校正结论、组件有效性判定与受控变量审计。

覆盖三条关键科研口径：
  * 多重比较校正后结论才有效（避免 8 变体两两对比下的假阳性）；
  * “各组件均有用”必须有显著统计证据，组合跑全本身不算证据；
  * 受控变量审计除超参数外还核对数据指纹/代码版本/训练步数。
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

from run_all import (ALPHA, MIN_SEEDS, _apply_holm, _cmd_env, _effect_status,
                     _rel_pct, _resolve_jobs, _stat, build_summary,
                     component_evidence, run_many, tradeoff_verdict,
                     verify_controlled_setup)


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
    print("  [ok] tradeoff_verdict（不显著/大增幅/收益增幅比判定）")


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
