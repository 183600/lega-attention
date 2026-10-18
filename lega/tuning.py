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
                vl = rec.get("val_loss")
                if vl is None:
                    continue
                if best is None or float(vl) < best:
                    best, best_step = float(vl), rec.get("step")
    final = None
    ev_path = os.path.join(trial_dir, "eval.json")
    if os.path.exists(ev_path):
        with open(ev_path, "r", encoding="utf-8") as f:
            ev = json.load(f) or {}
        if ev.get("val_loss") is not None:
            final = float(ev["val_loss"])
    if best is None:
        return final, None, final, "eval.json(final)" if final is not None else None
    return best, best_step, final, "train_log.jsonl(min)"


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
    rec = {"lr": float(lr), "weight_decay": float(wd), "seed": int(seed),
           "steps": int(steps), "out_dir": out_dir}
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout)
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
    rec.update({
        "val_loss_best": best, "best_step": best_step,
        "val_loss_final": final,
        "val_ppl_best": math.exp(min(20.0, best)),
        "val_loss_source": source,
    })
    return rec


def _grid_stats(grid, trials, seeds):
    """把逐 (配置, 种子) trial 聚合为逐配置统计（选择只看验证集）。

    单种子选择容易被 val_loss 的种子噪声左右（winner's curse）：网格里
    “最优配置”换一个种子重跑就可能翻转。多种子时按平均 val_loss 选择，
    并给出跨种子标准差，让“最优−默认”的差距可对照噪声解读。
    """
    out = []
    for lr, wd in grid:
        ts = [t for t in trials if t["lr"] == lr and t["weight_decay"] == wd]
        per_seed = {str(t.get("seed")): t.get("val_loss_best") for t in ts}
        vals = [t["val_loss_best"] for t in ts
                if t.get("val_loss_best") is not None]
        mean = (sum(vals) / len(vals)) if vals else None
        std = None
        if len(vals) >= 2:
            var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
            std = math.sqrt(var)
        out.append({
            "lr": lr, "weight_decay": wd,
            "n_seeds": len(vals), "n_failed": len(ts) - len(vals),
            "seeds": [int(s) for s in seeds],
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
    selected = min(ok, key=lambda g: g["val_loss_best_mean"])
    default = next((g for g in grid_stats
                    if g["lr"] == lr0 and g["weight_decay"] == wd0), None)
    gap = None
    default_is_optimal = None
    if default is not None and default["val_loss_best_mean"] is not None:
        # gap = 最优 − 默认：默认“近优”要求默认不比最优差出容差以上，
        # 即 gap ≥ −tol·|最优|（gap 为负表示最优显著优于默认 → 不近优）。
        gap = selected["val_loss_best_mean"] - default["val_loss_best_mean"]
        same_as_default = (selected["lr"] == default["lr"]
                           and selected["weight_decay"] == default["weight_decay"])
        default_is_optimal = bool(
            same_as_default
            or gap >= -tolerance * abs(selected["val_loss_best_mean"]))
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
