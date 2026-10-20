"""基线超参数调优：基线（standard）在验证集上的 lr/权重衰减网格。

科研方法问题：如果新方法只与“随手一个学习率”的基线比较，涨点可能全部
来自超参数红利，而非所验证的模块。本模块对**基线变体**（默认
`standard`）做一个小网格搜索：

  * 网格 = lr × weight_decay 的笛卡尔积（默认 lr 乘子 0.5/1/2、
    wd 取配置默认值），默认组合必须包含在内并作为第一个 trial；
  * 选择只用**验证集** val_loss（与训练中 model_best.pt 的选择口径一致：
    取 train_log.jsonl 中所有评测点的最小值），测试集从不加载
    （lega.data.load_bin 对 test 默认拒绝）；
  * 输出 `default_is_optimal` 判定：默认超参数与网格最优的 val_loss 差
    小于相对容差（默认 0.5%）即视为“接近最优”，否则流水线应改用最优
    组合统一训练所有变体（受控变量不变，仅整体平移）。

每个 trial 以子进程调用 `python -m lega.train` 真实训练，产出目录内
`eval.json` / `train_log.jsonl` / `run_manifest.json` 全量保留，便于审计。
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time

DEFAULT_LR_MULTS = (0.5, 1.0, 2.0)
DEFAULT_TOLERANCE = 0.005


def parse_floats(s):
    return [float(x) for x in str(s).split(",") if str(x).strip() != ""]


def parse_seeds(s, parser=None):
    """逗号分隔的整数种子列表；"1.5" 这类非整数被拒绝而不是静默截断成 1。"""
    out = []
    for x in str(s).split(","):
        x = x.strip()
        if not x:
            continue
        try:
            out.append(int(x))
        except ValueError:
            msg = f"--seeds 需要整数种子，收到 {x!r}"
            if parser is not None:
                parser.error(msg)
            raise ValueError(msg)
    return out or None


def trial_env(seed: int, env=None) -> dict:
    """trial 子进程的环境：PYTHONHASHSEED 对齐该 trial 自己的 seed。

    PYTHONHASHSEED 必须在解释器启动前设置才影响该进程；若让 trial 继承父
    进程的哈希盐，所有 trial 会共享父进程/首个种子的哈希随机化，同 seed
    重跑也不再稳定。CUBLAS_WORKSPACE_CONFIG 须在任何 CUDA 初始化前生效。
    """
    e = dict(env if env is not None else os.environ)
    e["PYTHONHASHSEED"] = str(int(seed))
    e.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    return e


def build_grid(lr0: float, wd0: float, lr_mults=DEFAULT_LR_MULTS,
               wd_list=None):
    """生成网格；默认组合 (lr0, wd0) 确保存在且排在首位。"""
    wds = [wd0] if wd_list is None else list(wd_list)
    if wd0 not in wds:
        wds = [wd0] + wds
    grid = []
    for wd in wds:
        for m in lr_mults:
            lr = lr0 * float(m)
            pair = (lr, wd)
            if pair not in grid:
                grid.append(pair)
    default = (lr0, wd0)
    # 乘子不含 1.0 时默认组合不在笛卡尔积里：显式补在首位，而不是让
    # list.remove 抛 ValueError（CLI 允许自定义 --lr_mults）。
    grid = [p for p in grid if p != default]
    return [default] + grid


def _trial_name(lr: float, wd: float) -> str:
    def fmt(x):
        s = f"{x:.6g}"
        return s.replace("+", "").replace("-", "m").replace(".", "p")
    return f"trial_lr{fmt(lr)}_wd{fmt(wd)}"


def _read_best_val_loss(trial_dir: str):
    """trial 的最优 val_loss：优先 train_log.jsonl 全评测点最小值。

    返回 (val_loss_best, best_step, val_loss_final, source)；train_log 缺失
    时回退 eval.json 的最终值（并标注来源，便于审计差异）。
    """
    log_path = os.path.join(trial_dir, "train_log.jsonl")
    best, best_step = None, None
    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                # 合法 JSON 但非对象的行（null/数组）按坏行跳过。
                if not isinstance(rec, dict):
                    continue
                vl = rec.get("val_loss")
                if vl is None:
                    continue
                try:
                    vl = float(vl)
                except (TypeError, ValueError, OverflowError):
                    continue
                if not math.isfinite(vl):
                    continue
                if best is None or vl < best:
                    best, best_step = vl, rec.get("step")
    final = None
    ev_path = os.path.join(trial_dir, "eval.json")
    if os.path.exists(ev_path):
        try:
            with open(ev_path, "r", encoding="utf-8") as f:
                ev = json.load(f)
        except (OSError, ValueError):
            ev = None
        if isinstance(ev, dict) and ev.get("val_loss") is not None:
            try:
                val = float(ev["val_loss"])
            except (TypeError, ValueError, OverflowError):
                val = None
            if val is not None and math.isfinite(val):
                final = val
    if best is None:
        return final, None, final, "eval.json(final)" if final is not None else None
    return best, best_step, final, "train_log.jsonl(min)"


def _manifest_identity(out_dir: str) -> dict:
    """从 trial 的 run_manifest.json 提取可核对的种子/代码/数据身份。

    trial 在独立子进程中训练；若只看父进程传入的 --seed，无法证明子进程
    真的固定了该种子、用了同一份数据。manifest 是子进程自己写的证据，
    这里只提取核对所需字段，缺失/畸形时显式 present=False。
    """
    path = os.path.join(out_dir, "run_manifest.json")
    out = {"present": False, "seed": None, "seed_verified": None,
           "config_sha256": None, "git_commit": None, "git_dirty": None,
           "worktree_diff_sha256": None, "data_fingerprint": None}
    if not os.path.exists(path):
        return out
    try:
        with open(path, "r", encoding="utf-8") as f:
            m = json.load(f)
    except (OSError, ValueError):
        return out
    if not isinstance(m, dict):
        return out
    out["present"] = True
    out["seed"] = m.get("seed")
    out["config_sha256"] = m.get("config_sha256")
    git = m.get("git")
    git = git if isinstance(git, dict) else {}
    out["git_commit"] = git.get("commit")
    out["git_dirty"] = git.get("dirty")
    worktree = m.get("worktree")
    worktree = worktree if isinstance(worktree, dict) else {}
    out["worktree_diff_sha256"] = worktree.get("diff_sha256")
    data = m.get("data")
    data = data if isinstance(data, dict) else {}
    out["data_fingerprint"] = data.get("digest")
    seed_report = m.get("seed_report")
    seed_report = seed_report if isinstance(seed_report, dict) else {}
    out["seed_verified"] = seed_report.get("seed_verified")
    return out


def _trial_identity_summary(trials) -> dict:
    """汇总成功 trial 的 manifest 身份，检查是否用了同一份代码/数据。"""
    manifests = [t.get("manifest") for t in trials
                 if isinstance(t.get("manifest"), dict)
                 and t["manifest"].get("present")]
    datas = sorted({m.get("data_fingerprint") for m in manifests
                    if m.get("data_fingerprint") is not None})
    commits = sorted({m.get("git_commit") for m in manifests
                      if m.get("git_commit") is not None})
    seed_mismatches = []
    for t in trials:
        m = t.get("manifest")
        if not isinstance(m, dict) or not m.get("present"):
            continue
        if m.get("seed") is not None and int(m["seed"]) != int(t.get("seed", -1)):
            seed_mismatches.append({"lr": t.get("lr"),
                                    "wd": t.get("weight_decay"),
                                    "requested_seed": t.get("seed"),
                                    "manifest_seed": m.get("seed")})
    checked = bool(manifests)
    consistent = None
    if checked:
        consistent = (len(datas) <= 1 and len(commits) <= 1
                      and not seed_mismatches
                      and all(m.get("seed_verified") is True
                              for m in manifests))
    return {
        "checked": checked,
        "n_with_manifest": len(manifests),
        "data_fingerprints": datas,
        "git_commits": commits,
        "seed_mismatches": seed_mismatches,
        "consistent": consistent,
        "note": ("consistent=True 要求全部成功 trial 的 manifest 都证明种子"
                 "已生效，且代码 commit/数据指纹一致；缺失 manifest 时为 "
                 "None（未核对），不冒充通过。"),
    }


def run_trial(config: str, data_dir: str, variant: str, seed: int, lr: float,
              wd: float, steps: int, out_dir: str, device: str = "auto",
              timeout=None) -> dict:
    """以子进程真实训练一个 trial；失败不抛异常，返回带 error 的记录。

    trial 的 lr/weight_decay 通过写入 trial 专属配置注入训练命令（其余字段
    原样继承）：`lega.train` 只从 --config 读取超参数，若不注入，网格中的每个
    trial 实际都用同一组默认超参数训练，调优结论完全失真。
    """
    os.makedirs(out_dir, exist_ok=True)
    with open(config, "r", encoding="utf-8") as f:
        trial_cfg = json.load(f)
    trial_cfg["lr"] = float(lr)
    trial_cfg["weight_decay"] = float(wd)
    trial_config = os.path.join(out_dir, "config_trial.json")
    with open(trial_config, "w", encoding="utf-8") as f:
        json.dump(trial_cfg, f, ensure_ascii=False, indent=2)
    cmd = [sys.executable, "-m", "lega.train",
           "--config", trial_config, "--variant", variant,
           "--data_dir", data_dir, "--out_dir", out_dir,
           "--seed", str(int(seed)), "--steps", str(int(steps)),
           "--device", device]
    t0 = time.perf_counter()
    run_env = trial_env(seed)
    rec = {"lr": float(lr), "weight_decay": float(wd), "seed": int(seed),
           "steps": int(steps), "out_dir": out_dir,
           # 逐 trial 的确定性环境也可审计（PYTHONHASHSEED 必须与 seed 对齐）
           "pythonhashseed": run_env["PYTHONHASHSEED"],
           "cublas_workspace_config": run_env.get("CUBLAS_WORKSPACE_CONFIG")}
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout, env=run_env)
    except subprocess.TimeoutExpired:
        rec.update({"returncode": None, "error": f"timeout({timeout}s)",
                    "duration_s": round(time.perf_counter() - t0, 1)})
        return rec
    rec["duration_s"] = round(time.perf_counter() - t0, 1)
    rec["returncode"] = int(proc.returncode)
    if proc.returncode != 0:
        rec["error"] = (proc.stderr or proc.stdout or "")[-800:]
        return rec
    best, best_step, final, source = _read_best_val_loss(out_dir)
    if best is None:
        rec["error"] = "训练成功但未找到 val_loss（train_log.jsonl/eval.json 缺失）"
        return rec
    manifest = _manifest_identity(out_dir)
    rec.update({
        "val_loss_best": best, "best_step": best_step,
        "val_loss_final": final,
        "val_ppl_best": math.exp(min(20.0, best)),
        "val_loss_source": source,
        "manifest": manifest,
        # 父进程请求的种子必须与子进程 manifest 一致且确认生效
        "seed_identity_ok": bool(
            manifest["present"] and manifest.get("seed_verified") is True
            and manifest.get("seed") is not None
            and int(manifest["seed"]) == int(seed)),
    })
    return rec


def _grid_stats(grid, trials, seeds):
    """把逐 (配置, 种子) trial 聚合为逐配置统计（选择只看验证集）。

    单种子选择容易被 val_loss 的种子噪声左右（winner's curse）：网格里
    “最优配置”换一个种子重跑就可能翻转。多种子时按平均 val_loss 选择，
    并给出跨种子标准差，让“最优−默认”的差距可对照噪声解读。
    """
    seed_list = [int(s) for s in seeds]
    out = []
    for lr, wd in grid:
        ts = [t for t in trials if t["lr"] == lr and t["weight_decay"] == wd]
        # 按种子去重：同一配置/种子重复跑的记录不能把均值与 n_seeds 撑大。
        per_seed = {}
        for t in ts:
            if t.get("val_loss_best") is None or t.get("seed") is None:
                continue
            per_seed[str(int(t["seed"]))] = t["val_loss_best"]
        vals = list(per_seed.values())
        mean = (sum(vals) / len(vals)) if vals else None
        std = None
        if len(vals) >= 2:
            var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
            std = math.sqrt(var)
        missing = [s for s in seed_list if str(s) not in per_seed]
        n_failed = sum(1 for t in ts
                       if t.get("val_loss_best") is None
                       or t.get("seed") is None)
        out.append({
            "lr": lr, "weight_decay": wd,
            "n_seeds": len(per_seed), "n_failed": n_failed,
            "seeds": seed_list, "missing_seeds": missing,
            # 只有覆盖全部选择种子的配置才与其他配置可比：部分种子成功的
            # 配置可能只是“留下了幸运种子”，不能参与 min(mean) 选择。
            "complete": not missing,
            "val_loss_best": mean,            # 兼容名：多种子时为均值
            "val_loss_best_mean": mean,
            "val_loss_best_std": std,
            "val_loss_best_per_seed": per_seed,
        })
    return out


def tune(config: str, data_dir: str, out_dir: str,
         variant: str = "standard", seed=None, steps=None,
         lr_mults=DEFAULT_LR_MULTS, wd_list=None, device: str = "auto",
         tolerance: float = DEFAULT_TOLERANCE, timeout=None,
         seeds=None) -> dict:
    """执行基线网格调优并落盘 `baseline_tuning.json`。

    选择口径：seeds=None 时只用配置种子（或显式 seed）；传 seeds=[...]
    时每个网格配置在每个种子上各训练一次，按**平均 val_loss** 选择。
    报告里 `single_seed_selection` 显式标注“只用 1 个种子选择”——此时
    网格最优可能只是该种子的幸运值，不能据此宣称基线已充分调优。
    """
    with open(config, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    lr0 = float(cfg["lr"])
    wd0 = float(cfg.get("weight_decay", 0.1))
    if seeds is None:
        seeds = [int(cfg.get("seed", 1337) if seed is None else seed)]
    else:
        seeds = [int(s) for s in seeds] or [int(cfg.get("seed", 1337))]
    # 重复种子（如 --seeds 7,7）否则会重复训练同一目录并伪造“多种子证据”。
    seeds = list(dict.fromkeys(seeds))
    seed = seeds[0]  # 兼容字段：报告里的主（第一个）选择种子
    steps = int(cfg["steps"] if steps is None else steps)
    grid = build_grid(lr0, wd0, lr_mults=lr_mults, wd_list=wd_list)
    trials = []
    for lr, wd in grid:
        for s in seeds:
            tdir = os.path.join(out_dir, _trial_name(lr, wd)
                                + (f"_seed{s}" if len(seeds) > 1 else ""))
            print(f"[tuning] trial lr={lr:.3g} wd={wd:.3g} seed={s} -> {tdir}")
            rec = run_trial(config, data_dir, variant, s, lr, wd, steps, tdir,
                            device=device, timeout=timeout)
            status = (f"val_loss={rec['val_loss_best']:.4f}"
                      if rec.get("val_loss_best") is not None
                      else f"FAILED: {rec.get('error', '')[:120]}")
            print(f"[tuning]   {status}")
            trials.append(rec)
    grid_stats = _grid_stats(grid, trials, seeds)
    ok = [g for g in grid_stats if g["val_loss_best_mean"] is not None]
    if not ok:
        raise RuntimeError("全部 trial 失败，无法完成基线调优；"
                           "请检查训练日志（各 trial 目录 stderr 已记录）")
    complete = [g for g in ok if g.get("complete")]
    if not complete:
        raise RuntimeError(
            "没有任何配置在所有选择种子上完成评测，无法按均值比较；"
            "请检查失败 trial 的日志或减少 seeds 后重跑")
    selected = min(complete, key=lambda g: g["val_loss_best_mean"])
    default = next((g for g in grid_stats
                    if g["lr"] == lr0 and g["weight_decay"] == wd0), None)
    gap = None
    default_is_optimal = None
    if (default is not None and default["val_loss_best_mean"] is not None
            and default.get("complete")):
        # gap = 最优 − 默认：默认“近优”要求默认不比最优差出容差以上，
        # 即 gap ≥ −tol·|最优|（gap 为负表示最优显著优于默认 → 不近优）。
        gap = selected["val_loss_best_mean"] - default["val_loss_best_mean"]
        same_as_default = (selected["lr"] == default["lr"]
                           and selected["weight_decay"] == default["weight_decay"])
        default_is_optimal = bool(
            same_as_default
            or gap >= -tolerance * abs(selected["val_loss_best_mean"]))
    # 调优报告必须能回答“在哪个代码/数据版本上选出的超参数”。父进程
    # 记录自身身份，逐 trial 则核对子进程 manifest，避免 trial 实际用了
    # 另一份数据/代码（或种子根本没生效）却仍被当作同一实验。
    from .repro import (data_fingerprint, git_info, runtime_info,
                        sha256_file, worktree_state)
    try:
        data_digest = data_fingerprint(data_dir).get("digest")
    except FileNotFoundError:
        data_digest = None
    git = git_info()
    worktree = worktree_state()
    trial_identity = _trial_identity_summary(trials)
    if trial_identity["consistent"] is False:
        print("[tuning] ⚠ 成功 trial 的 manifest 身份不一致："
              f"{trial_identity}；不得把网格结果当作单一版本调优证据")
    report = {
        "variant": variant,
        "config": os.path.abspath(config),
        "data_dir": os.path.abspath(data_dir),
        "seed": seed,
        "selection_seeds": [int(s) for s in seeds],
        "selection_seed_count": len(seeds),
        "single_seed_selection": len(seeds) == 1,
        "steps": steps,
        "lr_grid": [t["lr"] for t in trials],
        "weight_decay_grid": [t["weight_decay"] for t in trials],
        "n_trials": len(grid),
        "n_runs": len(trials),
        "n_failed": sum(1 for t in trials if t.get("val_loss_best") is None),
        "selection_metric": "val_loss（train_log.jsonl 全评测点最小值，"
                            "与 model_best.pt 选择口径一致）"
                            + ("；跨种子取均值" if len(seeds) > 1 else ""),
        "selection_aggregation": ("单种子（选择可能不可复现，见 "
                                  "single_seed_selection）" if len(seeds) == 1
                                  else f"{len(seeds)} 个种子的平均 val_loss"),
        "tolerance": float(tolerance),
        "selected": selected,
        "default_trial": default,
        "selected_minus_default_val_loss": gap,
        "default_is_optimal": default_is_optimal,
        "test_split_used": False,
        # 调优结论的版本身份：代码 commit/工作区、数据指纹、配置内容、
        # 运行环境，以及逐 trial manifest 的一致性核对结果。
        "code_commit": git.get("commit"),
        "code_dirty": git.get("dirty"),
        "worktree_diff_sha256": worktree.get("diff_sha256"),
        "data_fingerprint": data_digest,
        "config_sha256": sha256_file(config),
        "runtime": runtime_info(),
        "trial_identity": trial_identity,
        "grid": grid_stats,
        "trials": trials,
    }
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "baseline_tuning.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[tuning] 最优 lr={selected['lr']:.3g} "
          f"wd={selected['weight_decay']:.3g} "
          f"val_loss={selected['val_loss_best']:.4f}"
          + ("（单种子选择，结论可能受种子噪声影响）"
             if len(seeds) == 1 else f"（{len(seeds)} 种子均值）")
          + f" -> {out_path}")
    return report


def main():
    p = argparse.ArgumentParser(description="基线（standard）lr/权重衰减网格调优")
    p.add_argument("--config", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out_dir", required=True,
                   help="trial 目录与 baseline_tuning.json 的输出根目录")
    p.add_argument("--variant", default="standard")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--seeds", default=None,
                   type=lambda s: parse_seeds(s, p),
                   help="逗号分隔的选择种子列表（整数）：每个网格配置在每个种子"
                        "上各训一次，按平均 val_loss 选择（默认只用 --seed/配置"
                        "种子；单种子选择可能被噪声左右，正式实验建议 ≥3 个）")
    p.add_argument("--steps", type=int, default=None,
                   help="每个 trial 的训练步数（默认用配置值；可先用较小步数）")
    p.add_argument("--lr_mults", default=",".join(str(m) for m in DEFAULT_LR_MULTS))
    p.add_argument("--wd_list", default=None, help="逗号分隔；默认仅配置的 wd")
    p.add_argument("--device", default="auto")
    p.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE,
                   help="default_is_optimal 的相对容差（默认 0.005）")
    p.add_argument("--timeout", type=float, default=None,
                   help="单个 trial 超时秒数")
    args = p.parse_args()
    tune(args.config, args.data_dir, args.out_dir, variant=args.variant,
         seed=args.seed, steps=args.steps,
         lr_mults=parse_floats(args.lr_mults),
         wd_list=(parse_floats(args.wd_list) if args.wd_list else None),
         device=args.device, tolerance=args.tolerance, timeout=args.timeout,
         seeds=args.seeds)


if __name__ == "__main__":
    main()
