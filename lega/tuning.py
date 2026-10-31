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

import numpy as np

from .repro import current_pipeline_run_id
from .utils import (ensure_cublas_workspace_config, load_config_json,
                    mean_std, save_json, seed_int, strict_int)

DEFAULT_LR_MULTS = (0.5, 1.0, 2.0)
def _finite_number(value, name: str) -> float:
    """有限浮点超参数/容差；bool/NaN/Inf/非数值显式拒绝。"""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} 必须是有限数值，收到 bool {value!r}")
    if isinstance(value, str) and "_" in value:
        raise ValueError(f"{name} 必须是有限数值，收到 {value!r}")
    try:
        f = float(value)
    except (TypeError, ValueError, OverflowError) as e:
        raise ValueError(f"{name} 必须是有限数值，收到 {value!r}") from e
    if not math.isfinite(f):
        raise ValueError(f"{name} 必须是有限数值，收到 {value!r}")
    return f


DEFAULT_TOLERANCE = 0.005


def parse_floats(s):
    """逗号分隔的有限浮点数；NaN/Inf/非数值显式拒绝而不是进入训练网格。"""
    out = []
    for x in str(s).split(","):
        x = x.strip()
        if not x:
            continue
        # float("1_0")==10.0：下划线数字串不是规范数值，不能静默变成另一个
        # 学习率/WD 乘数（网格报告会与命令行声明不一致）。
        if "_" in x:
            raise ValueError(f"需要逗号分隔的数值，收到 {x!r}")
        try:
            v = float(x)
        except ValueError as e:
            raise ValueError(f"需要逗号分隔的数值，收到 {x!r}") from e
        if not math.isfinite(v):
            raise ValueError(f"数值必须有限（拒绝 NaN/Inf），收到 {x!r}")
        out.append(v)
    return out


def parse_seeds(s, parser=None):
    """逗号分隔的合法种子列表；非整数/越界显式拒绝而不是截断或晚失败。

    与 run_all 的种子解析同口径：``int("1_0")`` 会得到 10、``int(True)`` 会
    得到 1，都会让声明的种子与实际训练身份错位；负种子/超过 NumPy 与
    PYTHONHASHSEED 共同上限的种子则会在 trial 启动后才失败，必须在这里
    就报错。
    """
    out = []
    for x in str(s).split(","):
        x = x.strip()
        if not x:
            continue
        try:
            seed = seed_int(x, "种子")
        except ValueError as e:
            msg = str(e)
            if parser is not None:
                parser.error(msg)
            raise ValueError(msg) from e
        out.append(seed)
    return out or None


def trial_env(seed: int, env=None) -> dict:
    """trial 子进程的环境：PYTHONHASHSEED 对齐该 trial 自己的 seed。

    PYTHONHASHSEED 必须在解释器启动前设置才影响该进程；若让 trial 继承父
    进程的哈希盐，所有 trial 会共享父进程/首个种子的哈希随机化，同 seed
    重跑也不再稳定。CUBLAS_WORKSPACE_CONFIG 须在任何 CUDA 初始化前生效。
    """
    seed = seed_int(seed, "seed")
    e = dict(env if env is not None else os.environ)
    e["PYTHONHASHSEED"] = str(seed)
    # 非法/缺失的 CUBLAS 工作区配置必须在任何 CUDA 初始化前修复，不能
    # 用 setdefault 把父进程继承来的非法值原样传进 trial 子进程。
    ensure_cublas_workspace_config(e)
    return e


def build_grid(lr0: float, wd0: float, lr_mults=DEFAULT_LR_MULTS,
               wd_list=None):
    """生成网格；默认组合 (lr0, wd0) 确保存在且排在首位。"""
    lr0 = _finite_number(lr0, "lr0")
    wd0 = _finite_number(wd0, "wd0")
    if lr0 < 0 or wd0 < 0:
        raise ValueError(f"lr0/wd0 必须 >= 0，收到 {lr0!r}/{wd0!r}")
    lr_mults = [_finite_number(m, "lr_mults") for m in lr_mults]
    if any(m < 0 for m in lr_mults):
        raise ValueError(f"lr_mults 必须 >= 0，收到 {lr_mults!r}")
    wds = [wd0] if wd_list is None else [
        _finite_number(w, "wd_list") for w in wd_list]
    if any(w < 0 for w in wds):
        raise ValueError(f"wd_list 必须 >= 0，收到 {wds!r}")
    if wd0 not in wds:
        wds = [wd0] + wds
    grid = []
    for wd in wds:
        for m in lr_mults:
            lr = lr0 * m
            # 两个有限数的乘积仍可能溢出成 inf（如 1e308 × 2）；这种
            # trial 配置无法写入合法 JSON/无法训练，必须在建网格时拒绝。
            if not math.isfinite(lr) or not math.isfinite(wd):
                raise ValueError(
                    f"网格组合不可表示：lr0={lr0!r} × {m!r} = {lr!r}, "
                    f"wd={wd!r}")
            pair = (lr, wd)
            if pair not in grid:
                grid.append(pair)
    default = (lr0, wd0)
    # 乘子不含 1.0 时默认组合不在笛卡尔积里：显式补在首位，而不是让
    # list.remove 抛 ValueError（CLI 允许自定义 --lr_mults）。
    grid = [p for p in grid if p != default]
    return [default] + grid


def _trial_name(lr: float, wd: float) -> str:
    """trial 目录名：浮点的最短可往返表示，保证不同 lr/wd 不会同名。

    旧实现用 ``.6g`` 只有 6 位有效数字：1.0000004 与 1.0000005 会得到同一个
    目录名，后一个 trial 的 config/log/checkpoint 静默覆盖前一个，网格结果
    被悄悄丢掉。``repr(float)`` 对有限浮点是最短且可往返的表示，不同值必有
    不同字符串；替换仅用于文件名安全（不产生 ``.``/``-``/``+``）。
    """
    def fmt(x):
        s = repr(float(x))
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
                    # 与 train_log 的其他消费方同口径：bool/下划线数字串不是
                    # 规范观测，不能被 float() 当成 1.0 或 10.0 参与网格选择。
                    vl = _finite_number(vl, "val_loss")
                except ValueError:
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
                final = _finite_number(ev["val_loss"], "val_loss")
            except ValueError:
                final = None
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


def _as_int(value):
    """严格整数化：bool/非有限浮点/小数一律视为不可核对。

    manifest 是子进程自证身份的证据；若把 7.5 截断成 7 或把 True 当成 1，
    伪造的种子也能与请求种子“一致”，身份核对就失去意义。复用 strict_int
    也覆盖 np.bool_/np.float32 等会被 int() 静默截断的 NumPy 标量。
    """
    try:
        return strict_int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _trial_identity_summary(trials) -> dict:
    """汇总成功 trial 的 manifest 身份，检查是否用了同一份代码/数据。"""
    successful = [t for t in trials if t.get("val_loss_best") is not None]
    manifests = []
    for t in successful:
        m = t.get("manifest")
        if isinstance(m, dict) and m.get("present"):
            manifests.append(m)
    # manifest 字段是外部 JSON：列表/对象不可哈希，直接放进 set 会让整份
    # 调优报告在全部 trial 跑完后崩溃。只接受字符串身份；非字符串记为
    # 不可核对（consistent=False），而不是丢弃后当成“一致”。
    raw_datas = [m.get("data_fingerprint") for m in manifests
                 if m.get("data_fingerprint") is not None]
    raw_commits = [m.get("git_commit") for m in manifests
                   if m.get("git_commit") is not None]
    datas = sorted({v for v in raw_datas if isinstance(v, str)})
    commits = sorted({v for v in raw_commits if isinstance(v, str)})
    malformed_data = any(not isinstance(v, str) for v in raw_datas)
    malformed_commit = any(not isinstance(v, str) for v in raw_commits)
    seed_mismatches = []
    for t in successful:
        m = t.get("manifest")
        if not isinstance(m, dict) or not m.get("present"):
            continue
        requested = _as_int(t.get("seed"))
        actual = _as_int(m.get("seed"))
        if actual is None or requested is None or actual != requested:
            seed_mismatches.append({"lr": t.get("lr"),
                                    "wd": t.get("weight_decay"),
                                    "requested_seed": t.get("seed"),
                                    "manifest_seed": m.get("seed")})
    n_missing = len(successful) - len(manifests)
    checked = bool(manifests)
    consistent = None
    if checked:
        consistent = (n_missing == 0
                      and not malformed_data and not malformed_commit
                      and len(datas) <= 1 and len(commits) <= 1
                      and all(m.get("data_fingerprint") is not None
                              for m in manifests)
                      and all(m.get("git_commit") is not None
                              for m in manifests)
                      and not seed_mismatches
                      and all(m.get("seed_verified") is True
                              and _as_int(m.get("seed")) is not None
                              for m in manifests))
    return {
        "checked": checked,
        "n_successful": len(successful),
        "n_with_manifest": len(manifests),
        "n_missing_manifest": n_missing,
        "data_fingerprints": datas,
        "git_commits": commits,
        "malformed_data_fingerprint": malformed_data,
        "malformed_git_commit": malformed_commit,
        "seed_mismatches": seed_mismatches,
        "consistent": consistent,
        "note": ("consistent=True 要求全部成功 trial 的 manifest 都证明种子"
                 "已生效，且代码 commit/数据指纹一致；全部缺失 manifest 时为 "
                 "None（未核对），部分缺失或字段不可核对时为 False，"
                 "不冒充通过。"),
    }


def _load_config_object(path: str) -> dict:
    """读取配置：拒绝重复键与非对象顶层，数组/标量无法按字段继承。"""
    return load_config_json(path)


def run_trial(config: str, data_dir: str, variant: str, seed: int, lr: float,
              wd: float, steps: int, out_dir: str, device: str = "auto",
              timeout=None) -> dict:
    """以子进程真实训练一个 trial；失败不抛异常，返回带 error 的记录。

    trial 的 lr/weight_decay 通过写入 trial 专属配置注入训练命令（其余字段
    原样继承）：`lega.train` 只从 --config 读取超参数，若不注入，网格中的每个
    trial 实际都用同一组默认超参数训练，调优结论完全失真。
    """
    seed = seed_int(seed, "seed")
    steps = strict_int(steps, "steps")
    if steps <= 0:
        raise ValueError(f"steps 必须 >= 1，收到 {steps!r}")
    lr = _finite_number(lr, "lr")
    wd = _finite_number(wd, "wd")
    if lr < 0 or wd < 0:
        raise ValueError(f"lr/wd 必须 >= 0，收到 {lr!r}/{wd!r}")
    if timeout is not None:
        timeout = _finite_number(timeout, "timeout")
        if timeout <= 0:
            raise ValueError(f"timeout 必须 > 0，收到 {timeout!r}")
    os.makedirs(out_dir, exist_ok=True)
    trial_cfg = _load_config_object(config)
    trial_cfg["lr"] = lr
    trial_cfg["weight_decay"] = wd
    trial_config = os.path.join(out_dir, "config_trial.json")
    with open(trial_config, "w", encoding="utf-8") as f:
        json.dump(trial_cfg, f, ensure_ascii=False, indent=2)
    cmd = [sys.executable, "-m", "lega.train",
           "--config", trial_config, "--variant", variant,
           "--data_dir", data_dir, "--out_dir", out_dir,
           "--seed", str(seed), "--steps", str(steps),
           "--device", device]
    t0 = time.perf_counter()
    run_env = trial_env(seed)
    rec = {"lr": lr, "weight_decay": wd, "seed": seed,
           "steps": steps, "out_dir": out_dir,
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
            and _as_int(manifest.get("seed")) == int(seed)),
    })
    return rec


def _grid_stats(grid, trials, seeds):
    """把逐 (配置, 种子) trial 聚合为逐配置统计（选择只看验证集）。

    单种子选择容易被 val_loss 的种子噪声左右（winner's curse）：网格里
    “最优配置”换一个种子重跑就可能翻转。多种子时按平均 val_loss 选择，
    并给出跨种子标准差，让“最优−默认”的差距可对照噪声解读。
    """
    seed_list = []
    for s in seeds:
        sid = _as_int(s)
        if sid is None:
            raise ValueError(f"seeds 必须是整数种子，收到 {s!r}")
        seed_list.append(sid)
    out = []
    for lr, wd in grid:
        ts = [t for t in trials
              if isinstance(t, dict)
              and t.get("lr") == lr and t.get("weight_decay") == wd]

        def _best(t):
            raw = t.get("val_loss_best")
            if raw is None:
                return None
            try:
                return _finite_number(raw, "val_loss_best")
            except ValueError:
                return None

        # 按种子去重：同一配置/种子重复跑的记录不能把均值与 n_seeds 撑大。
        # 种子身份必须严格整数化：int(True)==1、int(1.5)==1 会把两次不同
        # 声明折叠成同一颗种子，均值与 missing 判定随之失真。
        per_seed = {}
        for t in ts:
            sid = _as_int(t.get("seed"))
            best = _best(t)
            if sid is None or best is None:
                continue
            per_seed[str(sid)] = best
        vals = list(per_seed.values())
        # 合法但接近 float 上限的 val_loss 在朴素 sum/平方差中会溢出成
        # inf/nan，使 baseline_tuning.json 成为非法 JSON（Infinity）。
        # 用缩放后的稳定均值/标准差，保持数值尺度不变。
        mean, std = mean_std(vals) if vals else (None, None)
        missing = [s for s in seed_list if str(s) not in per_seed]
        n_failed = sum(1 for t in ts
                       if _as_int(t.get("seed")) is None
                       or _best(t) is None)
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


def _selection_diagnostics(grid_stats, selected, default) -> dict:
    """网格选择的稳定性证据：逐种子支持、次优差距、留一种子复选。

    只看“均值最优”仍可能把种子噪声当成超参数信号：最优配置也许在多数
    种子上都不是第一，或与次优的差距小于逐种子配对波动。这里给出
      1. 每个选择种子的实际赢家与最优配置的支持率；
      2. 最优−次优的均值差，以及该差的逐种子配对标准差（差距在噪声内
         时不得把“最优”当作稳定结论）；
      3. 留一种子复选：在其余种子上重新选配置，再在被留出的种子上比较
         它与默认配置的 held-out val_loss（全部同批 trial，未用测试集）。
    这些是描述性稳定性证据，不是新的显著性检验；LOO 折之间也不独立。
    """
    complete = [
        g for g in (grid_stats or [])
        if isinstance(g, dict) and g.get("complete")
        and g.get("val_loss_best_mean") is not None
        and isinstance(g.get("val_loss_best_per_seed"), dict)
    ]
    out = {
        "n_configs": len(complete),
        "n_seeds": 0,
        "seeds": [],
        "selected_support_count": None,
        "selected_support_fraction": None,
        "per_seed_winners": [],
        "runner_up": None,
        "selection_flip": None,
        "leave_one_seed_out": None,
        "note": ("逐种子赢家/支持率与最优−次优差距都是描述性稳定性证据；"
                 "LOO 折之间不独立，不得把 agreement_rate 当作显著性检验。"),
    }
    if not complete or not isinstance(selected, dict):
        return out
    # 跨配置的共同种子；顺序沿用报告里的 selection_seeds。
    seed_order = []
    for g in complete:
        for s in g.get("seeds") or []:
            if str(s) not in seed_order:
                seed_order.append(str(s))
    common = [s for s in seed_order
              if all(s in g["val_loss_best_per_seed"] for g in complete)]
    out["n_seeds"] = len(common)
    out["seeds"] = [int(s) for s in common if str(s).lstrip("-").isdigit()] or common
    if not common:
        return out

    def _cfg(g):
        return {"lr": g.get("lr"), "weight_decay": g.get("weight_decay")}

    def _same(a, b):
        return (isinstance(a, dict) and isinstance(b, dict)
                and a.get("lr") == b.get("lr")
                and a.get("weight_decay") == b.get("weight_decay"))

    sel_stat = next((g for g in complete if _same(g, selected)), None)
    if sel_stat is None:
        return out
    winners = []
    support = 0
    for s in common:
        vals = [(g["val_loss_best_per_seed"][s], idx, g)
                for idx, g in enumerate(complete)]
        best_val = min(v[0] for v in vals)
        # 并列最优按网格顺序取第一个，但支持率把并列者都算作支持。
        tied = [v for v in vals if abs(v[0] - best_val) <= 1e-12]
        win = min(tied, key=lambda v: v[1])[2]
        if any(_same(g, selected) for _, _, g in tied):
            support += 1
        winners.append({"seed": int(s) if str(s).lstrip("-").isdigit() else s,
                        "lr": win.get("lr"),
                        "weight_decay": win.get("weight_decay"),
                        "val_loss": float(best_val),
                        "tie": len(tied) > 1})
    out["per_seed_winners"] = winners
    out["selected_support_count"] = support
    out["selected_support_fraction"] = support / len(common)
    out["selection_flip"] = support < len(common)

    others = [g for g in complete if g is not sel_stat]
    if others:
        runner = min(others, key=lambda g: g["val_loss_best_mean"])
        # 字段名是 selected − runner-up：selected 是均值最小者，故该值应
        # 为负（越小越好）；逐种子 paired_delta 则按“次优−最优”定义。
        margin = (sel_stat["val_loss_best_mean"]
                  - runner["val_loss_best_mean"])
        deltas = [runner["val_loss_best_per_seed"][s]
                  - sel_stat["val_loss_best_per_seed"][s] for s in common]
        # 与网格均值同一套缩放口径：接近 float 上限的 val_loss 差值
        # 在朴素 sum/平方中会溢出为 inf，使诊断字段成为非法 JSON。
        dmean, dstd = mean_std(deltas)
        dsem = (dstd / math.sqrt(len(deltas))
                if dstd is not None else None)
        out["runner_up"] = {
            **_cfg(runner),
            "val_loss_best_mean": runner["val_loss_best_mean"],
            "selected_minus_runner_up_mean": margin,
            "paired_delta_mean": dmean,
            "paired_delta_std": dstd,
            "paired_delta_sem": dsem,
            # 均值差不超过逐种子配对差的 1 个标准差：差距与种子噪声同量级。
            "within_seed_noise": (None if dstd is None
                                  else abs(margin) <= dstd),
        }

    if len(common) >= 3 and len(complete) >= 2:
        agreement = 0
        gains = []
        per_fold = []
        for held in common:
            train = [s for s in common if s != held]
            loo_stat = min(
                complete,
                key=lambda g: mean_std(
                    [g["val_loss_best_per_seed"][s] for s in train])[0])
            agree = _same(loo_stat, sel_stat)
            agreement += int(agree)
            rec = {
                "held_out_seed": (int(held) if str(held).lstrip("-").isdigit()
                                  else held),
                "selected_lr": loo_stat.get("lr"),
                "selected_weight_decay": loo_stat.get("weight_decay"),
                "agrees_with_full_selection": agree,
                "heldout_val_loss": loo_stat["val_loss_best_per_seed"][held],
            }
            if isinstance(default, dict) and _same(default, sel_stat):
                default_loss = default["val_loss_best_per_seed"].get(held)
            else:
                default_loss = (default.get("val_loss_best_per_seed", {}).get(held)
                                if isinstance(default, dict) else None)
            if default_loss is not None:
                gain = default_loss - rec["heldout_val_loss"]
                rec["heldout_gain_vs_default"] = gain
                gains.append(gain)
            per_fold.append(rec)
        loo = {
            "n_folds": len(common),
            "agreement_count": agreement,
            "agreement_rate": agreement / len(common),
            "per_fold": per_fold,
            "heldout_gain_vs_default_mean": None,
            "heldout_gain_vs_default_std": None,
            "wins_vs_default": None,
            "ties_vs_default": None,
            "losses_vs_default": None,
        }
        if gains:
            gm, gstd = mean_std(gains)
            loo.update({
                "heldout_gain_vs_default_mean": gm,
                "heldout_gain_vs_default_std": gstd,
                "wins_vs_default": sum(1 for g in gains if g > 1e-12),
                "ties_vs_default": sum(1 for g in gains if abs(g) <= 1e-12),
                "losses_vs_default": sum(1 for g in gains if g < -1e-12),
            })
        out["leave_one_seed_out"] = loo
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
    cfg = _load_config_object(config)
    lr0 = _finite_number(cfg.get("lr"), "config.lr")
    wd0 = _finite_number(cfg.get("weight_decay", 0.1), "config.weight_decay")
    if lr0 < 0 or wd0 < 0:
        raise ValueError(f"config.lr/weight_decay 必须 >= 0，收到 {lr0!r}/{wd0!r}")
    tolerance = _finite_number(tolerance, "tolerance")
    if tolerance < 0:
        raise ValueError(f"tolerance 必须 >= 0，收到 {tolerance!r}")
    if seeds is None:
        seeds = [seed_int(
            cfg.get("seed", 1337) if seed is None else seed, "seed")]
    else:
        seeds = [seed_int(s, "seeds") for s in seeds] or [
            seed_int(cfg.get("seed", 1337), "seed")]
    # 重复种子（如 --seeds 7,7）否则会重复训练同一目录并伪造“多种子证据”。
    seeds = list(dict.fromkeys(seeds))
    seed = seeds[0]  # 兼容字段：报告里的主（第一个）选择种子
    # steps 同样必须是精确整数：int(10.5)=10 会让调优预算与配置声明不一致
    steps = strict_int(cfg["steps"] if steps is None else steps, "steps")
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
        "pipeline_run_id": current_pipeline_run_id(),
        "code_commit": git.get("commit"),
        "code_dirty": git.get("dirty"),
        "worktree_diff_sha256": worktree.get("diff_sha256"),
        "data_fingerprint": data_digest,
        "config_sha256": sha256_file(config),
        "runtime": runtime_info(),
        "trial_identity": trial_identity,
        # 选择稳定性：多种子均值最优不等于稳定最优；逐种子支持率、
        # 最优−次优差距与留一种子复选必须随报告落盘。
        "selection_diagnostics": _selection_diagnostics(
            grid_stats, selected, default),
        "grid": grid_stats,
        "trials": trials,
    }
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "baseline_tuning.json")
    save_json(report, out_path)
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
