"""早停（可选）测试：更新口径、端到端训练行为、与曲线诊断/受控变量审计的衔接。

背景：曲线诊断会把"训练降而验证停滞/回升"判为过拟合并建议早停。本文件
覆盖：
  * early_stop_update 的判定口径（严格最小 checkpoint + 相对 min_delta 耐心）；
  * 默认关闭（patience=0）时训练跑满计划步数；
  * 开启后端到端：train_log 写入 early_stop 标记、eval.json 区分
    steps（实际）/steps_planned（计划）、下游仍使用 model_best.pt；
  * curve_diagnosis 把早停识别为"设计内截断"而非数据不足/训练中断。
"""
import json
import os
import sys
import tempfile

from lega.curves import diagnose_run_dir, diagnose_curve, load_train_log
from lega.data import prepare_bins
from lega.train import _config_int, _config_min_delta, early_stop_update


def test_early_stop_update_semantics():
    # 首次评测（best=inf）必然建立基线，不触发停止
    best, cnt, stop = early_stop_update(3.0, float("inf"), 0, 0.0, 2)
    assert best == 3.0 and cnt == 0 and not stop
    # 严格更优 → 重置计数
    best, cnt, stop = early_stop_update(2.9, 3.0, 1, 0.0, 2)
    assert best == 2.9 and cnt == 0 and not stop
    # 持平不算改善：计数累加，达到 patience 即停
    best, cnt, stop = early_stop_update(2.9, 2.9, 0, 0.0, 2)
    assert best == 2.9 and cnt == 1 and not stop
    best, cnt, stop = early_stop_update(2.9, 2.9, 1, 0.0, 2)
    assert cnt == 2 and stop
    # patience=0（默认关闭）：计数仍维护，但永不触发停止
    best, cnt, stop = early_stop_update(5.0, 2.9, 7, 0.0, 0)
    assert best == 2.9 and cnt == 8 and not stop
    # 负 patience 同样是关闭（不能被 truthy 误判为立即停止）
    best, cnt, stop = early_stop_update(5.0, 2.9, 7, 0.0, -1)
    assert best == 2.9 and cnt == 8 and not stop
    # min_delta：幅度小于阈值不算"实质改善"（不重置计数），但严格更小仍进 best
    best, cnt, stop = early_stop_update(2.89, 2.9, 0, 0.05, 2)
    assert best == 2.89 and cnt == 1 and not stop
    # 超过阈值的改善正常重置
    best, cnt, stop = early_stop_update(2.7, 2.9, 1, 0.05, 2)
    assert best == 2.7 and cnt == 0 and not stop
    # 非有限值由 NaN 防护路径中止，这里不消费、也不误判改善
    best, cnt, stop = early_stop_update(float("nan"), 2.9, 1, 0.0, 1)
    assert best == 2.9 and cnt == 1 and not stop
    print("  [ok] early_stop_update（严格最优 + min_delta + patience 口径）")


def test_config_int_rejects_bool_fraction_and_below_min():
    """训练配置整数不能被 int() 截断；0 间隔必须在使用前失败。"""
    for bad in (True, 1.5, "1.5", float("nan"), float("inf"), 0, -1):
        try:
            _config_int(bad, "log_every", 1)
        except ValueError:
            pass
        else:
            raise AssertionError(f"非法配置未被拒绝: {bad!r}")
    assert _config_int(2.0, "log_every", 1) == 2
    assert _config_int("2", "log_every", 1) == 2
    assert _config_int(0, "warmup", 0) == 0
    print("  [ok] 训练配置整数：拒绝 bool/小数/非有限值/0 间隔")


def test_config_min_delta_rejects_nonfinite_and_out_of_range():
    """min_delta 越界会让改善阈值失真（>=1 永不改善、<0 永不重置）。"""
    for bad in (True, False, -0.1, 1.0, 2, float("nan"),
                float("inf"), "nan", None, object()):
        try:
            _config_min_delta(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"非法 min_delta 未被拒绝: {bad!r}")
    assert _config_min_delta(0) == 0.0
    assert _config_min_delta(0.05) == 0.05
    assert _config_min_delta("0.1") == 0.1
    print("  [ok] early_stop_min_delta：拒绝 bool/非有限/越界值")


def _write_data(d: str) -> None:
    """构造足够长的小语料（字符级），保证 train/val 都能组出 ctx=32 的窗口。"""
    base = "the quick brown fox jumps over the lazy dog. "
    with open(os.path.join(d, "train.txt"), "w", encoding="utf-8") as f:
        f.write(base * 60)
    with open(os.path.join(d, "val.txt"), "w", encoding="utf-8") as f:
        f.write(base * 20)


def _run_train(cfg: dict, data_dir: str, out_dir: str, extra_argv=()):
    import lega.train as train_mod
    cfg_path = os.path.join(data_dir, "cfg.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f)
    argv = ["lega.train", "--config", cfg_path, "--variant", "standard",
            "--data_dir", data_dir, "--out_dir", out_dir,
            "--device", "cpu", "--seed", "1"] + list(extra_argv)
    old = sys.argv
    sys.argv = argv
    try:
        train_mod.main()
    finally:
        sys.argv = old


def _cfg():
    return {
        "d_model": 32, "n_layers": 2, "n_heads": 2, "ffn_mult": 4,
        "block_size": 32, "dropout": 0.0, "attn_dropout": 0.0,
        "gate_bias_init": 2.0, "use_sdpa_standard": True,
        "batch_size": 2, "grad_accum": 1, "steps": 6, "lr": 0.0,
        "warmup": 1, "weight_decay": 0.0, "grad_clip": 1.0, "train_ctx": 32,
        "amp": False, "seed": 1, "log_every": 2, "eval_every": 2,
        "eval_iters": 2,
    }


def test_train_rejects_zero_log_every_before_training():
    """log_every=0 必须在开训前报错，而不是训练中途 ZeroDivisionError。"""
    with tempfile.TemporaryDirectory() as tmp:
        data_dir = os.path.join(tmp, "data")
        os.makedirs(data_dir)
        _write_data(data_dir)
        prepare_bins(data_dir, kind="char", force=True)
        cfg = _cfg()
        cfg["log_every"] = 0
        out_dir = os.path.join(tmp, "run")
        try:
            _run_train(cfg, data_dir, out_dir)
        except ValueError as e:
            assert "log_every" in str(e), e
        else:
            raise AssertionError("log_every=0 未被拒绝")
        assert not os.path.exists(os.path.join(out_dir, "train_log.jsonl"))
        assert not os.path.exists(os.path.join(out_dir, "eval.json"))
    print("  [ok] log_every=0 在开训前显式拒绝")


def test_train_early_stop_end_to_end():
    """lr=0 时验证损失恒定：patience=1 必然在第二次评测处早停。

    同时验证：eval.json 区分实际/计划步数、train_log 带 early_stop 标记、
    CPU 下训练进程统计字段为 None（CUDA 才有），以及 CLI 覆盖配置生效。
    """
    with tempfile.TemporaryDirectory() as tmp:
        data_dir = os.path.join(tmp, "data")
        os.makedirs(data_dir)
        _write_data(data_dir)
        prepare_bins(data_dir, kind="char", force=True)
        cfg = _cfg()
        cfg["early_stop_patience"] = 0     # 配置关闭；CLI 覆盖为 1
        out_dir = os.path.join(tmp, "run1")
        _run_train(cfg, data_dir, out_dir,
                   extra_argv=["--early_stop_patience", "1"])
        ev = json.load(open(os.path.join(out_dir, "eval.json"),
                            encoding="utf-8"))
        assert ev["early_stopped"] is True, ev
        assert ev["early_stop_patience"] == 1
        assert ev["steps"] == 4 and ev["steps_planned"] == 6, ev
        assert ev["tokens_seen"] == 4 * 2 * 32
        assert ev["train_peak_mem_mb"] is None, "CPU 训练没有 CUDA 进程统计"
        cfg_used = json.load(open(os.path.join(out_dir, "config_used.json"),
                                  encoding="utf-8"))
        assert cfg_used["early_stop_patience"] == 1, "CLI 覆盖必须进入受控变量记录"
        assert os.path.exists(os.path.join(out_dir, "model_best.pt"))
        recs = load_train_log(os.path.join(out_dir, "train_log.jsonl"))
        assert any(r.get("early_stop") for r in recs), "train_log 必须留下早停标记"
        diag = diagnose_run_dir(out_dir)
        assert diag["early_stopped"] is True and diag["early_stop_step"] == 4
        assert any("早停" in e for e in diag["evidence"])
        # 默认关闭（patience=0）：训满计划步数且不写早停标记
        out_dir2 = os.path.join(tmp, "run2")
        cfg2 = _cfg()
        _run_train(cfg2, data_dir, out_dir2)
        ev2 = json.load(open(os.path.join(out_dir2, "eval.json"),
                             encoding="utf-8"))
        assert ev2["early_stopped"] is False
        assert ev2["steps"] == 6 and ev2["steps_planned"] == 6
        recs2 = load_train_log(os.path.join(out_dir2, "train_log.jsonl"))
        assert not any(r.get("early_stop") for r in recs2)
        # 每个评测点都记录相对历史最优的差距（训练中途预警的原材料）
        eval_recs = [r for r in recs2 if "val_loss" in r]
        assert eval_recs and all("val_vs_best_rel" in r for r in eval_recs)
        # steps 非 eval_every 整数倍：最终补评测也要更新耐心计数，否则
        # 最后一条日志的 evals_since_best 与 val_vs_best_rel 自相矛盾。
        out_dir3 = os.path.join(tmp, "run3")
        cfg3 = _cfg()
        cfg3["steps"] = 5
        _run_train(cfg3, data_dir, out_dir3)
        recs3 = load_train_log(os.path.join(out_dir3, "train_log.jsonl"))
        final3 = [r for r in recs3 if r.get("final")]
        assert len(final3) == 1, final3
        prev_evals = [r for r in recs3
                      if "val_loss" in r and not r.get("final")]
        assert prev_evals
        prev_best = min(r["val_loss"] for r in prev_evals)
        expected_cnt = (0 if final3[0]["val_loss"] < prev_best
                        else prev_evals[-1]["evals_since_best"] + 1)
        assert final3[0]["evals_since_best"] == expected_cnt, final3[0]
        assert abs(final3[0]["val_vs_best_rel"]) < 1e-12, final3[0]
    print("  [ok] train 端到端早停（CLI 覆盖 + 标记落盘 + 默认关闭兼容）")


def test_final_eval_improvement_resets_patience_counter(monkeypatch):
    """最终补评测若刷新历史最优，evals_since_best 必须归零。

    回归：曾先把 best_val 写成本次结果，再交给 early_stop_update，导致
    “最终评测改善”被误判为未改善，日志同时出现 val_vs_best_rel=0 与
    evals_since_best=1 的矛盾。
    """
    with tempfile.TemporaryDirectory() as tmp:
        data_dir = os.path.join(tmp, "data")
        os.makedirs(data_dir)
        _write_data(data_dir)
        prepare_bins(data_dir, kind="char", force=True)
        cfg = _cfg()
        cfg.update({"steps": 3, "eval_every": 2, "log_every": 3,
                    "eval_iters": 1, "early_stop_patience": 0})
        out_dir = os.path.join(tmp, "run")
        import lega.train as train_mod
        losses = iter([3.0, 2.0])

        def fake_evaluate_val(*args, **kwargs):
            vl = next(losses)
            return {
                "val_loss": vl, "val_loss_se": None,
                "val_loss_ci95": None, "val_ppl_ci95": None,
                "val_loss_min": vl, "val_loss_max": vl,
                "gate_mean": None, "eval_iters_requested": 1,
                "eval_iters": 1, "val_windows": 1,
                "val_tokens_evaluated": 32,
            }

        monkeypatch.setattr(train_mod, "evaluate_val", fake_evaluate_val)
        _run_train(cfg, data_dir, out_dir)
        recs = load_train_log(os.path.join(out_dir, "train_log.jsonl"))
        final = [r for r in recs if r.get("final")]
        assert len(final) == 1, final
        assert final[0]["val_loss"] == 2.0
        assert final[0]["evals_since_best"] == 0, final[0]
        assert abs(final[0]["val_vs_best_rel"]) < 1e-12, final[0]
    print("  [ok] 最终补评测刷新最优时耐心计数归零")


def test_train_live_overfit_hint_end_to_end():
    """早停默认关闭时，验证连续多次无改善必须在训练日志里留下过拟合/平台
    预警——只看最终数值会漏掉过程信号。"""
    with tempfile.TemporaryDirectory() as tmp:
        data_dir = os.path.join(tmp, "data")
        os.makedirs(data_dir)
        _write_data(data_dir)
        prepare_bins(data_dir, kind="char", force=True)
        cfg = _cfg()
        cfg["steps"] = 10        # eval_every=2 → 5 次评测，足够累积 3 次无改善
        cfg["eval_every"] = 2
        out_dir = os.path.join(tmp, "run")
        _run_train(cfg, data_dir, out_dir)
        recs = load_train_log(os.path.join(out_dir, "train_log.jsonl"))
        hints = [r for r in recs if r.get("overfit_hint")]
        assert hints, "lr=0 时验证损失不变，应触发过拟合/平台预警"
        assert all(r["evals_since_best"] >= 3 for r in hints), hints
        assert "平台" in hints[0]["overfit_hint"]
    print("  [ok] train 中途过拟合/平台预警（连续无改善写日志）")


def test_curve_marks_early_stop_not_data_insufficient():
    """早停截断 ≠ 数据不足：少量评测点也须如实说明截断原因，且不得再建议
    "增加训练步数/评测密度"或"延长训练"（与早停决策自相矛盾）。"""
    recs = [{"step": 10, "loss": 3.0}, {"step": 20, "loss": 2.8},
            {"step": 30, "loss": 2.6}, {"step": 40, "loss": 2.5}]
    recs += [{"step": 50, "val_loss": 3.0, "val_ppl": 20.1}]
    recs.append({"step": 50, "early_stop": True})
    d = diagnose_curve(recs)
    assert d["status"] == "数据不足", d
    assert d["early_stopped"] is True and d["early_stop_step"] == 50
    assert not any("eval_every" in r or "训练步数" in r
                   for r in d["recommendations"]), d["recommendations"]
    assert any("早停" in r for r in d["recommendations"])
    assert not any("延长训练" in r and "不要" not in r
                   for r in d["recommendations"])
    # 正常曲线的结论不受影响（无标记 → early_stopped=False）
    d2 = diagnose_curve([{"step": 10, "loss": 4.2}, {"step": 20, "loss": 3.5},
                         {"step": 30, "loss": 2.9},
                         {"step": 50, "val_loss": 4.0},
                         {"step": 100, "val_loss": 3.2}])
    assert d2["early_stopped"] is False and d2["early_stop_step"] is None
    print("  [ok] 曲线诊断：早停标注为设计内截断（不误判数据不足/建议续训）")
