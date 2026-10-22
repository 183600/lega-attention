"""基线调优协议测试：网格构造、选择口径、默认近优判定与失败处理。

科研口径：调优只能看验证集；默认组合必须参与比较；“默认近优”判定要有
显式容差；trial 失败必须如实记录而不是静默当作某个损失值。
"""
import json
import math
import os
import shutil
import tempfile
from unittest import mock

from lega.tuning import (DEFAULT_LR_MULTS, _grid_stats, _read_best_val_loss,
                         _selection_diagnostics, _trial_identity_summary,
                         _trial_name, build_grid, parse_floats, run_trial,
                         trial_env, tune)


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
    # 非有限值/非数值/下划线数字串不能进入网格（NaN lr 会让所有 trial 失效
    # 且污染报告；float("1_0")=10.0 会静默放大学习率）。
    for bad in ("nan", "inf", "-inf", "0.5,nan", "oops", "1_0", "0.5,1_0"):
        try:
            parse_floats(bad)
            raised = False
        except ValueError:
            raised = True
        assert raised, bad
    print("  [ok] build_grid/parse_floats（默认在首、去重、wd 补默认、拒绝非有限）")


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

        def fake_run(cmd, capture_output=True, text=True, timeout=None,
                     env=None):
            seen["cmd"] = cmd
            seen["env"] = env
            tcfg_path = cmd[cmd.index("--config") + 1]
            with open(tcfg_path, encoding="utf-8") as f:
                seen["cfg"] = json.load(f)
            with open(os.path.join(out_dir, "train_log.jsonl"), "w",
                      encoding="utf-8") as f:
                f.write(json.dumps({"step": 10, "val_loss": 1.0}) + "\n")
            with open(os.path.join(out_dir, "run_manifest.json"), "w",
                      encoding="utf-8") as f:
                json.dump({"seed": 7,
                           "seed_report": {"seed_verified": True},
                           "git": {"commit": "c" * 40, "dirty": False},
                           "worktree": {"diff_sha256": "d" * 64},
                           "data": {"digest": "f" * 64},
                           "config_sha256": "e" * 64}, f)
            return _Proc()

        with mock.patch("lega.tuning.subprocess.run", side_effect=fake_run):
            rec = run_trial(cfg_path, tmp, "standard", 7, 3e-4, 0.01, 10,
                            out_dir)
        assert "error" not in rec and rec["val_loss_best"] == 1.0
        assert seen["cfg"]["lr"] == 3e-4
        assert seen["cfg"]["weight_decay"] == 0.01
        assert seen["cfg"]["d_model"] == 8          # 其余字段原样继承
        # each trial's Python hash salt must match its own seed (not inherit
        # the parent/first-seed value); CUBLAS config before CUDA init
        assert seen["env"]["PYTHONHASHSEED"] == "7"
        assert seen["env"]["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
        assert rec["pythonhashseed"] == "7"
        assert os.path.exists(os.path.join(out_dir, "config_trial.json"))
        # 子进程自己写的 manifest 才是种子/版本确实生效的证据
        assert rec["manifest"]["present"] is True
        assert rec["manifest"]["seed_verified"] is True
        assert rec["manifest"]["data_fingerprint"] == "f" * 64
        assert rec["seed_identity_ok"] is True
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] run_trial（lr/weight_decay 经 trial 配置注入训练命令）")


def test_trial_name_filesystem_safe():
    name = _trial_name(6e-4, 0.1)
    assert "/" not in name and "." not in name
    assert _trial_name(1.2e-3, 0.01) != name
    # .6g 会把这两个不同的 lr 都写成 1，导致 trial 目录互相覆盖
    assert _trial_name(1.0000004, 0.1) != _trial_name(1.0000005, 0.1)
    assert _trial_name(-1e-5, 0.1) != _trial_name(1e-5, 0.1)
    print("  [ok] _trial_name（文件名安全、浮点值可区分）")


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


def test_tune_embeds_provenance_and_checks_trial_manifests():
    """报告必须记录代码/数据身份，并核对子进程 manifest 是否一致。"""
    tmp = tempfile.mkdtemp()
    try:
        cfg_path = os.path.join(tmp, "config.json")
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump({"lr": 6e-4, "weight_decay": 0.1, "steps": 10,
                       "seed": 7}, f)
        out = os.path.join(tmp, "tuning")

        def fake(config, data_dir, variant, seed, lr, wd, steps, out_dir,
                 **kwargs):
            digest = "a" * 64 if lr == 3e-4 else "b" * 64
            return {"lr": lr, "weight_decay": wd, "seed": seed,
                    "steps": steps, "out_dir": out_dir, "returncode": 0,
                    "val_loss_best": 1.0 if lr == 3e-4 else 1.1,
                    "val_loss_final": 1.0, "best_step": steps,
                    "val_loss_source": "train_log.jsonl(min)",
                    "manifest": {"present": True, "seed": seed,
                                 "seed_verified": True,
                                 "git_commit": "c" * 40,
                                 "data_fingerprint": digest}}

        with mock.patch("lega.tuning.run_trial", side_effect=fake):
            report = tune(cfg_path, tmp, out, lr_mults=[0.5, 2.0])
        for key in ("code_commit", "code_dirty", "worktree_diff_sha256",
                    "data_fingerprint", "config_sha256", "runtime"):
            assert key in report, key
        ident = report["trial_identity"]
        assert ident["checked"] is True and ident["consistent"] is False
        assert ident["data_fingerprints"] == ["a" * 64, "b" * 64]
        assert report["test_split_used"] is False
        # 全部 trial 缺 manifest → 未核对，不得冒充通过
        with mock.patch("lega.tuning.run_trial", side_effect=_fake_trial(
                {(6e-4, 0.1): 1.0, (3e-4, 0.1): 1.1, (1.2e-3, 0.1): 1.0})):
            report2 = tune(cfg_path, tmp, out, lr_mults=[1.0])
        assert report2["trial_identity"]["checked"] is False
        assert report2["trial_identity"]["consistent"] is None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] tune 报告内嵌 provenance 并核对 trial manifest 一致性")


def test_trial_identity_requires_manifest_for_every_success_and_valid_seed():
    """部分成功 trial 缺 manifest、种子缺失/畸形或代码数据未记录时不得判一致。"""
    def trial(seed=7, present=True, manifest_seed=7, verified=True,
              data_fp="d" * 64, commit="c" * 40):
        rec = {"lr": 6e-4, "weight_decay": 0.1, "seed": seed,
               "val_loss_best": 1.0}
        if present:
            rec["manifest"] = {"present": True, "seed": manifest_seed,
                               "seed_verified": verified,
                               "data_fingerprint": data_fp,
                               "git_commit": commit}
        else:
            rec["manifest"] = {"present": False}
        return rec

    failed = {"lr": 3e-4, "weight_decay": 0.1, "seed": 7, "error": "boom"}
    ident = _trial_identity_summary(
        [trial(), trial(seed=8, present=False), failed])
    assert ident["checked"] is True
    assert ident["n_successful"] == 2
    assert ident["n_with_manifest"] == 1
    assert ident["n_missing_manifest"] == 1
    assert ident["consistent"] is False

    for bad in (trial(manifest_seed=None), trial(manifest_seed="bad"),
                trial(manifest_seed=7.5), trial(manifest_seed=True),
                trial(manifest_seed=float("nan")),
                trial(data_fp=None), trial(commit=None),
                trial(verified=False)):
        one = _trial_identity_summary([bad])
        assert one["checked"] is True and one["consistent"] is False, one
    integral = _trial_identity_summary([trial(manifest_seed=7.0)])
    assert integral["consistent"] is True, integral
    print("  [ok] trial identity（成功 trial 全部可核对才判一致）")


def test_tuning_numeric_args_reject_bool_and_nonfinite():
    """网格/容差/子进程参数不能被 bool 或非有限值静默改写。"""
    import tempfile
    for bad in (True, float("nan"), float("inf"), "bad"):
        for kwargs in ({"lr0": bad}, {"wd0": bad},
                       {"lr_mults": [1.0, bad]}, {"wd_list": [bad]}):
            args = {"lr0": 6e-4, "wd0": 0.1}
            args.update(kwargs)
            try:
                build_grid(**args)
            except ValueError:
                pass
            else:
                raise AssertionError(f"非法网格参数未被拒绝: {kwargs}")
    with tempfile.TemporaryDirectory() as d:
        for kwargs in ({"seed": True}, {"seed": -1},
                       {"steps": 10.5}, {"steps": 0},
                       {"lr": True}, {"lr": float("nan")},
                       {"wd": True}, {"timeout": True},
                       {"timeout": -1}):
            args = dict(config=os.path.join(d, "missing.json"),
                        data_dir=d, variant="lega", seed=7, lr=6e-4,
                        wd=0.1, steps=10, out_dir=os.path.join(d, "out"))
            args.update(kwargs)
            try:
                run_trial(**args)
            except ValueError:
                pass
            else:
                raise AssertionError(f"非法 trial 参数未被拒绝: {kwargs}")
        cfg_path = os.path.join(d, "cfg.json")
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump({"lr": 6e-4, "weight_decay": 0.1, "steps": 10,
                       "seed": 7}, f)
        for bad in (True, float("nan"), -1.0):
            try:
                tune(cfg_path, d, os.path.join(d, "tune"), tolerance=bad)
            except ValueError:
                pass
            else:
                raise AssertionError(f"非法 tolerance 未被拒绝: {bad!r}")
    try:
        trial_env(True)
    except ValueError:
        pass
    else:
        raise AssertionError("bool seed 未被 trial_env 拒绝")
    print("  [ok] 调优数值参数拒绝 bool/NaN/Inf/负值")


def test_parse_seeds_rejects_non_integer():
    """--seeds 是整数列表：浮点/下划线/负数必须报错，不能静默改写种子身份。"""
    from lega.tuning import parse_seeds
    assert parse_seeds("1, 2,3") == [1, 2, 3]
    assert parse_seeds("") is None
    assert parse_seeds(" 4 ") == [4]
    for bad in ("1.5", "1_0", "-1", "4294967296", "true", "0x10"):
        try:
            parse_seeds(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"非法种子未被拒绝: {bad!r}")
    print("  [ok] parse_seeds：整数校验（拒绝浮点/下划线/负数种子）")


def test_tune_multiseed_selects_by_mean_not_lucky_seed():
    """多种子选择：按平均 val_loss 选配置，不被单个“幸运种子”左右。"""
    tmp = tempfile.mkdtemp()
    try:
        cfg_path = os.path.join(tmp, "config.json")
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump({"lr": 6e-4, "weight_decay": 0.1, "steps": 10,
                       "seed": 7}, f)
        out = os.path.join(tmp, "tuning")
        # 默认配置 3 个种子都稳（1.0）；备选配置均值更差，
        # 但种子 7 上恰好更低（0.9）——单种子选择会误选它。
        vals = {(6e-4, 0.1, 7): 1.0, (6e-4, 0.1, 8): 1.0, (6e-4, 0.1, 9): 1.0,
                (3e-4, 0.1, 7): 0.9, (3e-4, 0.1, 8): 1.2, (3e-4, 0.1, 9): 1.2}

        def fake(config, data_dir, variant, seed, lr, wd, steps, out_dir,
                 device="auto", timeout=None):
            v = vals[(lr, wd, seed)]
            return {"lr": lr, "weight_decay": wd, "seed": seed,
                    "steps": steps, "out_dir": out_dir, "returncode": 0,
                    "val_loss_best": v, "best_step": steps,
                    "val_loss_final": v, "val_loss_source": "train_log.jsonl(min)"}

        with mock.patch("lega.tuning.run_trial", side_effect=fake):
            report = tune(cfg_path, tmp, out, lr_mults=[0.5, 1.0],
                          seeds=[7, 8, 9])
        assert report["selection_seeds"] == [7, 8, 9]
        assert report["single_seed_selection"] is False
        assert report["n_trials"] == 2 and report["n_runs"] == 6
        assert report["selected"]["lr"] == 6e-4        # 按均值选，而非幸运种子
        assert abs(report["selected"]["val_loss_best_mean"] - 1.0) < 1e-12
        assert report["selected"]["val_loss_best_std"] == 0.0
        assert report["default_is_optimal"] is True
        by_lr = {g["lr"]: g for g in report["grid"]}
        assert abs(by_lr[3e-4]["val_loss_best_mean"]
                   - (0.9 + 1.2 + 1.2) / 3) < 1e-12
        assert by_lr[3e-4]["val_loss_best_std"] > 0
        assert set(by_lr[3e-4]["val_loss_best_per_seed"]) == {"7", "8", "9"}
        # 稳定性报告随 tune() 落盘：默认配置均值最优，但只在 2/3 种子上
        # 实际赢；次优与其均值差 0.1，小于逐种子配对差 1 个 SD（噪声内）。
        diag = report["selection_diagnostics"]
        assert diag["selected_support_count"] == 2, diag
        assert diag["selection_flip"] is True
        assert diag["runner_up"]["lr"] == 3e-4
        assert diag["runner_up"]["within_seed_noise"] is True
        assert diag["leave_one_seed_out"]["agreement_rate"] == 1.0
        # 单种子（默认）路径保持旧口径且显式标注“选择只用 1 个种子”
        with mock.patch("lega.tuning.run_trial",
                        side_effect=_fake_trial({(6e-4, 0.1): 1.0,
                                                 (3e-4, 0.1): 0.9})):
            report_s = tune(cfg_path, tmp, out, lr_mults=[0.5, 1.0])
        assert report_s["single_seed_selection"] is True
        assert report_s["selection_seeds"] == [7]
        assert report_s["selected"]["lr"] == 3e-4       # 单种子仍按点估计选
        assert report_s["selection_diagnostics"]["n_seeds"] == 1
        assert report_s["selection_diagnostics"]["leave_one_seed_out"] is None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] tune（多种子按均值选择、幸运种子不翻盘、单种子显式标注）")


def test_selection_diagnostics_reports_flip_noise_and_loo():
    """选择稳定性必须显式量化：均值最优可能只赢 1/3 种子、差距在噪声内。"""
    def g(lr, vals, default=False):
        mean = sum(vals) / len(vals)
        return {"lr": lr, "weight_decay": 0.1, "n_seeds": 3,
                "seeds": [7, 8, 9], "complete": True,
                "val_loss_best_mean": mean, "val_loss_best_std": None,
                "val_loss_best_per_seed": {str(seed): v
                                           for seed, v in zip((7, 8, 9), vals)},
                "default": default}

    default = g(1.0, [0.9, 0.9, 1.4], default=True)
    selected = g(2.0, [1.0, 1.0, 1.1])
    diag = _selection_diagnostics([default, selected], selected, default)
    assert diag["n_seeds"] == 3
    assert diag["selected_support_count"] == 1
    assert abs(diag["selected_support_fraction"] - 1 / 3) < 1e-12
    assert diag["selection_flip"] is True
    assert {w["lr"] for w in diag["per_seed_winners"]} == {1.0, 2.0}
    ru = diag["runner_up"]
    assert ru["lr"] == 1.0
    assert abs(ru["selected_minus_runner_up_mean"] + (1 / 30)) < 1e-12
    assert ru["within_seed_noise"] is True
    assert abs(ru["paired_delta_mean"] - 1 / 30) < 1e-12
    loo = diag["leave_one_seed_out"]
    assert loo["n_folds"] == 3 and loo["agreement_count"] == 2
    assert abs(loo["agreement_rate"] - 2 / 3) < 1e-12
    assert loo["wins_vs_default"] == 0
    assert loo["ties_vs_default"] == 1
    assert loo["losses_vs_default"] == 2
    assert abs(loo["heldout_gain_vs_default_mean"] + 1 / 15) < 1e-12
    # 单种子 / 缺默认：不得伪造稳定性或 LOO
    one = g(1.0, [1.0])
    one["seeds"] = [7]
    one["val_loss_best_per_seed"] = {"7": 1.0}
    d1 = _selection_diagnostics([one], one, one)
    assert d1["n_seeds"] == 1 and d1["leave_one_seed_out"] is None
    assert d1["selected_support_fraction"] == 1.0
    assert _selection_diagnostics([], {}, None)["n_seeds"] == 0
    print("  [ok] _selection_diagnostics（支持率、噪声差距、LOO 复选）")


def test_grid_stats_duplicate_success_is_not_counted_as_failure():
    """同一配置/种子重复成功的 trial 只去重，不得被算成失败。"""
    trials = [
        {"lr": 1.0, "weight_decay": 0.1, "seed": 7, "val_loss_best": 1.0},
        {"lr": 1.0, "weight_decay": 0.1, "seed": 7, "val_loss_best": 1.1},
        {"lr": 1.0, "weight_decay": 0.1, "seed": 8, "error": "boom"},
    ]
    g = _grid_stats([(1.0, 0.1)], trials, [7, 8])[0]
    assert g["n_seeds"] == 1 and g["missing_seeds"] == [8]
    assert g["n_failed"] == 1, g
    assert g["complete"] is False
    # 种子/损失值是身份与观测：True→1、1.5→1 的截断会把不同 trial 折叠成
    # 同一颗种子；bool 损失也不是数值。
    bad = [
        {"lr": 1.0, "weight_decay": 0.1, "seed": True, "val_loss_best": 1.0},
        {"lr": 1.0, "weight_decay": 0.1, "seed": 1.5, "val_loss_best": 1.0},
        {"lr": 1.0, "weight_decay": 0.1, "seed": 7, "val_loss_best": True},
    ]
    g = _grid_stats([(1.0, 0.1)], bad, [7])[0]
    assert g["n_seeds"] == 0 and g["n_failed"] == 3, g
    assert g["missing_seeds"] == [7] and g["complete"] is False
    try:
        _grid_stats([(1.0, 0.1)], trials, [True])
        raise AssertionError("选择种子 True 应显式报错")
    except ValueError:
        pass
    print("  [ok] _grid_stats：重复成功去重、失败计数不虚增；"
          "坏种子/坏损失不截断合并")


def test_tune_excludes_partial_seed_coverage_and_dedupes_seeds():
    """部分种子成功的配置不能靠“留下的幸运种子”赢过完整覆盖的默认配置。"""
    tmp = tempfile.mkdtemp()
    try:
        cfg_path = os.path.join(tmp, "config.json")
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump({"lr": 6e-4, "weight_decay": 0.1, "steps": 2,
                       "seed": 7}, f)
        out = os.path.join(tmp, "tuning")
        calls = []

        def fake(config, data_dir, variant, seed, lr, wd, steps, out_dir,
                 device="auto", timeout=None):
            calls.append((lr, seed, out_dir))
            base = {"lr": lr, "weight_decay": wd, "seed": seed, "steps": steps,
                    "out_dir": out_dir, "returncode": 0}
            if lr == 3e-4 and seed != 7:
                base["error"] = "boom"
                return base
            base.update({"val_loss_best": 0.5 if lr == 3e-4 else 1.0,
                         "best_step": steps, "val_loss_final": 1.0,
                         "val_loss_source": "train_log.jsonl(min)"})
            return base

        with mock.patch("lega.tuning.run_trial", side_effect=fake):
            report = tune(cfg_path, tmp, out, lr_mults=[0.5, 1.0],
                          seeds=[7, 7, 8, 9])
        # 重复种子去重后每配置只跑 3 次，不再伪造“4 种子”证据
        assert report["selection_seeds"] == [7, 8, 9]
        assert report["selection_seed_count"] == 3
        assert len(calls) == 6
        assert all("_seed7" in c[2] for c in calls if c[1] == 7)
        by_lr = {g["lr"]: g for g in report["grid"]}
        assert by_lr[3e-4]["complete"] is False
        assert by_lr[3e-4]["missing_seeds"] == [8, 9]
        assert by_lr[3e-4]["n_seeds"] == 1
        assert by_lr[6e-4]["complete"] is True
        assert report["selected"]["lr"] == 6e-4
        assert report["default_is_optimal"] is True
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] tune：剔除部分种子覆盖配置并去重选择种子")


def test_tune_dedupes_and_default_mult_present():
    assert 1.0 in DEFAULT_LR_MULTS
    print("  [ok] DEFAULT_LR_MULTS 含 1.0（默认组合必在网格内）")


def test_read_best_val_loss_skips_non_dict_and_bad_eval():
    """日志非对象行/非数值 val_loss、eval 顶层非对象均不能崩溃。"""
    tmp = tempfile.mkdtemp()
    try:
        tdir = os.path.join(tmp, "trial")
        os.makedirs(tdir)
        with open(os.path.join(tdir, "train_log.jsonl"), "w",
                  encoding="utf-8") as f:
            f.write("null\n[1]\n")
            f.write(json.dumps({"step": 1, "val_loss": "bad"}) + "\n")
            f.write(json.dumps({"step": 2, "val_loss": 1.2}) + "\n")
        with open(os.path.join(tdir, "eval.json"), "w",
                  encoding="utf-8") as f:
            json.dump([1, 2, 3], f)
        assert _read_best_val_loss(tdir) == (1.2, 2, None,
                                             "train_log.jsonl(min)")
        tdir2 = os.path.join(tmp, "only_bad_eval")
        os.makedirs(tdir2)
        with open(os.path.join(tdir2, "eval.json"), "w",
                  encoding="utf-8") as f:
            json.dump("oops", f)
        assert _read_best_val_loss(tdir2) == (None, None, None, None)
        # bool / 下划线数字串 / 非有限值不是合法 val_loss：float(True)=1.0、
        # float("1_0")=10.0 若被接受，网格会按伪造的损失选超参数。
        tdir3 = os.path.join(tmp, "bad_values")
        os.makedirs(tdir3)
        with open(os.path.join(tdir3, "train_log.jsonl"), "w",
                  encoding="utf-8") as f:
            for bad in (True, "1_0", "nan", "inf"):
                f.write(json.dumps({"step": 1, "val_loss": bad}) + "\n")
            f.write(json.dumps({"step": 2, "val_loss": 0.5}) + "\n")
        with open(os.path.join(tdir3, "eval.json"), "w",
                  encoding="utf-8") as f:
            json.dump({"val_loss": "1_0"}, f)
        assert _read_best_val_loss(tdir3) == (0.5, 2, None,
                                              "train_log.jsonl(min)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] 调优日志/eval 非对象或坏值：跳过而不崩溃")


def test_trial_env_aligns_hashseed_per_trial():
    """逐 trial 的 PYTHONHASHSEED 必须等于该 trial 自己的 seed。"""
    from lega.tuning import trial_env
    e = trial_env(123)
    assert e["PYTHONHASHSEED"] == "123"
    assert e["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    # 不修改调用方传入的环境对象
    base = {"PYTHONHASHSEED": "999"}
    e2 = trial_env(7, base)
    assert e2["PYTHONHASHSEED"] == "7" and base["PYTHONHASHSEED"] == "999"
    print("  [ok] trial_env：PYTHONHASHSEED 对齐 trial seed")


def test_tuning_manifest_identity_rejects_numpy_scalars():
    """manifest 种子核对不能把 np.bool_/np.float32(7.5) 截断成 1/7。"""
    import numpy as np
    from lega.tuning import _as_int, _finite_number
    for bad in (np.bool_(True), np.float32(7.5), np.float64(7.5)):
        assert _as_int(bad) is None, bad
    assert _as_int(np.float64(7.0)) == 7
    # 整数值的 NumPy 浮点标量语义明确，应与 Python 7.0 一样接受；小数仍拒绝。
    assert _as_int(np.float32(7.0)) == 7
    for bad in (np.bool_(True), np.bool_(False), "1_0"):
        try:
            _finite_number(bad, "x")
            raise AssertionError(f"np.bool_ 应被拒绝: {bad!r}")
        except ValueError:
            pass
    print("  [ok] 调优身份核对拒绝 NumPy bool/小数标量")


def test_trial_identity_tolerates_malformed_manifest_fields():
    """manifest 的 data/git 身份为列表等非字符串时，报告不能整体崩溃。"""
    from lega.tuning import _trial_identity_summary

    def _trial(lr, data, commit):
        return {"lr": lr, "weight_decay": 0.1, "val_loss_best": 1.0,
                "seed": 1,
                "manifest": {"present": True, "seed": 1,
                             "data_fingerprint": data,
                             "git_commit": commit,
                             "seed_verified": True}}

    good = _trial_identity_summary([
        _trial(0.1, "d" * 16, "c" * 40),
        _trial(0.2, "d" * 16, "c" * 40)])
    assert good["consistent"] is True and good["malformed_data_fingerprint"] is False
    bad = _trial_identity_summary([
        _trial(0.1, ["d"], "c" * 40),
        _trial(0.2, "d" * 16, ["c"])])
    assert bad["consistent"] is False
    assert bad["malformed_data_fingerprint"] is True
    assert bad["malformed_git_commit"] is True
    # 合法字符串仍保留供审计；畸形值只把 consistent 置 False。
    assert bad["data_fingerprints"] == ["d" * 16]
    assert bad["git_commits"] == ["c" * 40]


def test_build_grid_rejects_unrepresentable_lr_product():
    """有限 lr0/mult 相乘溢出时不能生成 inf 的 trial 配置。"""
    from lega.tuning import build_grid
    try:
        build_grid(1e308, 0.1, lr_mults=(2.0,))
        raised = False
    except ValueError:
        raised = True
    assert raised
    g = build_grid(1e-3, 0.1, lr_mults=(0.5, 2.0))
    assert g[0] == (1e-3, 0.1)
    assert all(math.isfinite(x) and math.isfinite(w) for x, w in g)
    print("  [ok] 调优网格拒绝溢出为 inf 的 lr 组合")
