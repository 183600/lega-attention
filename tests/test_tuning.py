"""基线调优协议测试：网格构造、选择口径、默认近优判定与失败处理。

科研口径：调优只能看验证集；默认组合必须参与比较；“默认近优”判定要有
显式容差；trial 失败必须如实记录而不是静默当作某个损失值。
"""
import json
import os
import shutil
import tempfile
from unittest import mock

from lega.tuning import (DEFAULT_LR_MULTS, _read_best_val_loss, _trial_name,
                         build_grid, parse_floats, run_trial, tune)


def test_build_grid_default_first_and_dedup():
    grid = build_grid(6e-4, 0.1, lr_mults=[0.5, 1.0, 2.0])
    assert grid[0] == (6e-4, 0.1), grid
    assert len(grid) == 3 and len(set(grid)) == 3
    # 乘子重复/含 1：不得产生重复 trial；网格先默认后其余
    grid2 = build_grid(6e-4, 0.1, lr_mults=[1.0, 1.0, 2.0])
    assert grid2[0] == (6e-4, 0.1)
    assert grid2[1] == (1.2e-3, 0.1)
    assert len(grid2) == 2
    # wd_list 不含默认值时自动补默认并排最前
    grid3 = build_grid(6e-4, 0.1, lr_mults=[1.0], wd_list=[0.01])
    assert grid3[0] == (6e-4, 0.1)
    assert set(w for _, w in grid3) == {0.1, 0.01}
    assert parse_floats("0.5, 1,2") == [0.5, 1.0, 2.0]
    print("  [ok] build_grid/parse_floats（默认在首、去重、wd 补默认）")


def test_build_grid_default_first_when_mult_missing():
    """乘子不含 1.0（CLI 允许）时默认组合仍须排首位，而不是崩溃。"""
    grid = build_grid(6e-4, 0.1, lr_mults=[0.5, 2.0])
    assert grid[0] == (6e-4, 0.1)
    assert len(grid) == 3 and (3e-4, 0.1) in grid and (1.2e-3, 0.1) in grid
    print("  [ok] build_grid（乘子不含 1.0 时默认组合仍补首位）")


def test_run_trial_injects_lr_and_weight_decay():
    """trial 的 lr/weight_decay 必须真实注入训练命令（trial 专属配置），
    否则整个网格退化为同一组默认超参数重复训练。"""
    tmp = tempfile.mkdtemp()
    try:
        cfg_path = os.path.join(tmp, "config.json")
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump({"lr": 6e-4, "weight_decay": 0.1, "steps": 10,
                       "seed": 7, "d_model": 8}, f)
        out_dir = os.path.join(tmp, "trial")
        seen = {}

        class _Proc:
            returncode = 0
            stdout = ""
            stderr = ""

        def fake_run(cmd, capture_output=True, text=True, timeout=None):
            seen["cmd"] = cmd
            tcfg_path = cmd[cmd.index("--config") + 1]
            with open(tcfg_path, encoding="utf-8") as f:
                seen["cfg"] = json.load(f)
            with open(os.path.join(out_dir, "train_log.jsonl"), "w",
                      encoding="utf-8") as f:
                f.write(json.dumps({"step": 10, "val_loss": 1.0}) + "\n")
            return _Proc()

        with mock.patch("lega.tuning.subprocess.run", side_effect=fake_run):
            rec = run_trial(cfg_path, tmp, "standard", 7, 3e-4, 0.01, 10,
                            out_dir)
        assert "error" not in rec and rec["val_loss_best"] == 1.0
        assert seen["cfg"]["lr"] == 3e-4
        assert seen["cfg"]["weight_decay"] == 0.01
        assert seen["cfg"]["d_model"] == 8          # 其余字段原样继承
        assert os.path.exists(os.path.join(out_dir, "config_trial.json"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] run_trial（lr/weight_decay 经 trial 配置注入训练命令）")


def test_trial_name_filesystem_safe():
    name = _trial_name(6e-4, 0.1)
    assert "/" not in name and "." not in name
    assert _trial_name(1.2e-3, 0.01) != name
    print("  [ok] _trial_name（文件名安全、可区分）")


def test_read_best_val_loss_prefers_log_min():
    tmp = tempfile.mkdtemp()
    try:
        tdir = os.path.join(tmp, "trial")
        os.makedirs(tdir)
        with open(os.path.join(tdir, "train_log.jsonl"), "w",
                  encoding="utf-8") as f:
            f.write(json.dumps({"step": 10, "val_loss": 1.0}) + "\n")
            f.write(json.dumps({"step": 20, "val_loss": 0.9}) + "\n")
            f.write(json.dumps({"step": 30, "val_loss": 0.95,
                                "final": True}) + "\n")
            f.write("不是 JSON\n")
        with open(os.path.join(tdir, "eval.json"), "w", encoding="utf-8") as f:
            json.dump({"val_loss": 0.95}, f)
        best, step, final, src = _read_best_val_loss(tdir)
        assert best == 0.9 and step == 20 and final == 0.95
        assert src == "train_log.jsonl(min)"
        # 只有 eval.json：回退最终值并标注来源
        tdir2 = os.path.join(tmp, "only_eval")
        os.makedirs(tdir2)
        with open(os.path.join(tdir2, "eval.json"), "w", encoding="utf-8") as f:
            json.dump({"val_loss": 1.1}, f)
        best2, step2, final2, src2 = _read_best_val_loss(tdir2)
        assert best2 == 1.1 and step2 is None and final2 == 1.1
        assert "eval.json" in src2
        # 什么都没有：全 None
        tdir3 = os.path.join(tmp, "empty")
        os.makedirs(tdir3)
        assert _read_best_val_loss(tdir3) == (None, None, None, None)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] _read_best_val_loss（日志最小值优先、回退与来源标注）")


def _fake_trial(val_loss_map):
    def fake(config, data_dir, variant, seed, lr, wd, steps, out_dir,
             device="auto", timeout=None):
        return {"lr": lr, "weight_decay": wd, "seed": seed, "steps": steps,
                "out_dir": out_dir, "duration_s": 0.1, "returncode": 0,
                "val_loss_best": val_loss_map[(lr, wd)],
                "best_step": steps, "val_loss_final": val_loss_map[(lr, wd)],
                "val_loss_source": "train_log.jsonl(min)"}
    return fake


def test_tune_selects_grid_best_and_flags_default_not_optimal():
    tmp = tempfile.mkdtemp()
    try:
        cfg_path = os.path.join(tmp, "config.json")
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump({"lr": 6e-4, "weight_decay": 0.1, "steps": 10,
                       "seed": 7}, f)
        out = os.path.join(tmp, "tuning")
        loss = {(6e-4, 0.1): 1.0, (3e-4, 0.1): 1.1, (1.2e-3, 0.1): 0.9}
        with mock.patch("lega.tuning.run_trial",
                        side_effect=_fake_trial(loss)):
            report = tune(cfg_path, tmp, out, lr_mults=[0.5, 1.0, 2.0])
        assert report["n_trials"] == 3 and report["n_failed"] == 0
        assert report["selected"]["lr"] == 1.2e-3
        assert report["default_is_optimal"] is False
        assert abs(report["selected_minus_default_val_loss"] - (-0.1)) < 1e-12
        assert report["test_split_used"] is False
        assert report["seed"] == 7 and report["steps"] == 10
        assert os.path.exists(os.path.join(out, "baseline_tuning.json"))
        # 默认即最优（或在容差内）→ default_is_optimal True
        loss2 = {(6e-4, 0.1): 0.9, (3e-4, 0.1): 1.1, (1.2e-3, 0.1): 1.0}
        with mock.patch("lega.tuning.run_trial",
                        side_effect=_fake_trial(loss2)):
            report2 = tune(cfg_path, tmp, out, lr_mults=[0.5, 1.0, 2.0])
        assert report2["default_is_optimal"] is True
        # 全部失败 → 必须报错，而不是给一个虚假的 selected
        def all_fail(*a, **k):
            return {"lr": 0.0, "weight_decay": 0.0, "error": "boom"}
        with mock.patch("lega.tuning.run_trial", side_effect=all_fail):
            try:
                tune(cfg_path, tmp, out, lr_mults=[1.0])
                raise AssertionError("全部 trial 失败应抛 RuntimeError")
            except RuntimeError:
                pass
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] tune（网格最优选择/默认近优判定/全失败报错）")


def test_tune_dedupes_and_default_mult_present():
    assert 1.0 in DEFAULT_LR_MULTS
    print("  [ok] DEFAULT_LR_MULTS 含 1.0（默认组合必在网格内）")
