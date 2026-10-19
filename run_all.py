#!/usr/bin/env python3
"""一键实验流水线：训练全部变体 → PPL → RULER → 机制分析 → 图表 → 汇总表。

变体（相同参数量级与训练步数）:
  standard | qgate | null | lega | lega_no_len | lega_no_conc | lega_no_query
  | lega_conc_only | lega_len_only

对比与消融协议（科研统计口径）:
  * 受控变量：所有变体 × 种子使用同一 config 训练（同一优化器/学习率策略/
    训练步数/数据顺序），训练后自动核对各 run 的 config_used.json 与
    eval.json 溯源（数据指纹/代码 commit/实际训练步数），任何与“除变体外
    全部一致”相悖的差异都会直接报错，避免把调参收益或数据差异误归因给方法；
  * 多种子：关键指标跨随机种子报告 Mean ± Std（种子一致 → 变体间配对比较），
    配对 t 检验的同指标家族做 Holm–Bonferroni 校正以控制 FWER，并报告
    Cohen's d_z 效应量；默认种子数见 configs（tiny 3 / small 5），可用
    --seeds 覆盖；
  * 小样本稳健性：种子只有 3~5 个时 t 检验的正态假设不可靠，每组配对同时
    报告符号翻转置换检验（分布无关）p_perm（同家族独立 Holm 校正）、配对差
    bootstrap 95% CI 与最小可检测效应 MDE——t 与置换结论冲突显式标 ⚠，
    “不显著”先对照 MDE 区分“无效应”与“功效不足”；
  * 机制主张检验：auroc_logit − auroc_qterm 的同种子配对差作为独立指标
    家族正式检验（与逐 run 样例聚类 bootstrap 区间互为印证）；
  * 预注册：preregistration/hypotheses.json 在正式实验前登记方向性主张，
    汇总时逐条判定 支持/拒绝/未定；未登记的对比一律为事后（exploratory）
    分析，只能作为线索（登记表刻意不放 docs/：该目录会被流水线自动清理）；
  * 消融完整性：全因子（无/A/B/C/AB/AC/BC/ABC）与去一法（Full/Full-A/
    Full-B/Full-C）两套组合全部实现；汇总表显式列出消融矩阵，并基于
    Holm 校正后的显著证据逐一判定 A/B/C 组件是否“有用”——组合齐全但证据
    不足时不会下“各组件均有用”的结论。

用法:
  # 冒烟测试（tiny 配置 + 伪数据，仅验证全流程可跑通；默认只跑 1 个种子）
  python run_all.py --config configs/tiny.json --data_dir data/dummy --smoke

  # 正式实验（small 配置 + WikiText-103，默认跑配置中的全部种子）
  python scripts/prepare_data.py --out_dir data/wikitext103
  python run_all.py --config configs/small.json --data_dir data/wikitext103

  # 自定义种子列表（例如 3 个种子）
  python run_all.py --config configs/small.json --data_dir data/wikitext103 \
      --seeds 1337,2024,42
"""
import argparse
import csv
import glob
import json
import math
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from lega.attention import VARIANTS, VARIANT_CN  # noqa: E402
from lega.badcase import (CATEGORY_CN as BADCASE_CN,  # noqa: E402
                          WINDOW_CATEGORY_CN, analyze_badcases,
                          analyze_worst_windows, cross_variant_analysis,
                          process_eval_json, process_ruler_json)
from lega.curves import (LOSS_SPIKE_FACTOR, STATUS_ORDER,  # noqa: E402
                         diagnose_run_dir, worst_status)
from lega.datacheck import read_test_access_log, summarize_test_access  # noqa: E402
from lega.experiment import (append_record, build_experiment_record,  # noqa: E402
                             collect_artifacts, ledger_path, new_run_id,
                             snapshot_train_run)
from lega.hypotheses import (REGISTRY_PATH as HYPOTHESES_PATH,  # noqa: E402
                             fingerprint as registration_fingerprint,
                             load_registration, verify_registration)
from lega.ngram import evaluate as ngram_evaluate  # noqa: E402
from lega.tuning import DEFAULT_TOLERANCE as TUNE_TOLERANCE  # noqa: E402
from lega.tuning import parse_floats, tune as tune_baseline  # noqa: E402
from lega.repro import data_fingerprint, git_info, runtime_info, sha256_file  # noqa: E402
from lega.utils import (bootstrap_mean_ci, get_device, holm_bonferroni,
                        mean_std, min_detectable_effect, paired_sign_flip_test,
                        paired_t_test, save_json)  # noqa: E402

ALL_VARIANTS = list(VARIANTS)
# 机制分析不依赖输出门也可计算“集中但看错率/P(hit|集中）”：
# null 变体虽无 g/logit，仍应纳入；standard 无需诊断，保持跳过。
GATED_FOR_ANALYSIS = ("lega", "qgate", "null", "lega_no_len", "lega_no_conc",
                      "lega_no_query", "lega_conc_only", "lega_len_only")

# 全因子消融矩阵（A=查询项 w·q, B=集中度证据 log max_j p_j, C=长度校准 log n）。
# standard 表示无门控“无”格；三组件两两与全部组合均对应一个独立变体。
FACTORIAL_CELLS = (
    ("无（无门）", "standard"),
    ("A 查询", "qgate"),
    ("B 集中度", "lega_conc_only"),
    ("C 长度", "lega_len_only"),
    ("A+B", "lega_no_len"),
    ("A+C", "lega_no_conc"),
    ("B+C", "lega_no_query"),
    ("A+B+C", "lega"),
)
# 去一法（leave-one-out）：Full 及各去掉一个组件的结果。
LOO_CELLS = (
    ("Full", "lega"),
    ("Full−A（去查询）", "lega_no_query"),
    ("Full−B（去集中度）", "lega_no_conc"),
    ("Full−C（去长度）", "lega_no_len"),
)

# 受控变量核对时允许逐 run 不同的字段（方法与复现标识，不属于受控变量）
_AUDIT_EXEMPT_KEYS = {"variant", "seed", "device", "tag"}

# 显著性协议：同一指标家族内 Holm 校正后 p<ALPHA 才判定显著；
# 少于 MIN_SEEDS 个种子时只报描述统计，不作显著性/组件有效性判定。
ALPHA = 0.05
MIN_SEEDS = 3

_SEED_RE = re.compile(r"_seed(\d+)$")


def _cmd_env(cmd, env=None):
    """为单条子进程命令构造环境：PYTHONHASHSEED 与该命令 --seed 对齐。

    PYTHONHASHSEED 必须在解释器启动前设置才生效，父进程统一设一个值会让
    所有 (变体, 种子) 子进程共享第一个种子的哈希盐；这里逐命令解析
    --seed，保证每个 run 的 Python 哈希随机化也由它自己的种子唯一确定
    （没有 --seed 的命令如 latency/plots 沿用基准值）。
    """
    e = dict(env if env is not None else os.environ)
    if "--seed" in cmd:
        i = cmd.index("--seed")
        if i + 1 < len(cmd):
            try:
                e["PYTHONHASHSEED"] = str(int(cmd[i + 1]))
            except (TypeError, ValueError):
                pass
    e.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    return e


def run(cmd, env=None):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT, env=_cmd_env(cmd, env))


def _resolve_jobs(args) -> int:
    """并发度：显式 --jobs 优先；未指定时 CPU 取 min(4, 核数)，CUDA/--wandb 取 1。

    小核机器（<=4 核）默认取 核数+1：_jobs_env 已把每个子进程压到单 BLAS
    线程，总线程数不随 worker 数增加；大核机器维持 4 并发。显式 --jobs
    完全以用户值为准，不受此默认影响。
    """
    if args.jobs is not None:
        return max(1, int(args.jobs))
    if args.wandb:
        return 1
    if get_device(args.device).startswith("cuda"):
        return 1
    cpu = os.cpu_count() or 1
    return min(4, cpu) if cpu > 4 else cpu + 1


def parse_variants(spec: str) -> list:
    """解析 --variants 逗号列表，保留顺序并按首次出现去重（与 --seeds 同口径）。

    重复变体会让每个阶段生成重复命令（sanity/训练/评测/机制分析同一输出路径），
    并发执行时互相覆盖结果文件——去重后行为与显式去重的种子列表一致。
    """
    return list(dict.fromkeys(v.strip() for v in spec.split(",") if v.strip()))


def _jobs_env(jobs: int):
    """并发时按 jobs 均分每个子进程的 BLAS 线程配额；jobs<=1 返回 None。"""
    if jobs <= 1:
        return None
    env = os.environ.copy()
    n = max(1, (os.cpu_count() or 1) // jobs)
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS"):
        env[k] = str(n)
    return env


def run_many(cmds, jobs: int, env=None) -> None:
    """顺序或并发执行多条独立子进程命令。

    并发只改变多个独立 run 之间的调度，不改变单个 run 的行为；每条命令的
    PYTHONHASHSEED 按自身 --seed 设置（见 _cmd_env）。失败时等待全部命令
    结束后抛出首个（按提交顺序）非零退出码的 CalledProcessError。
    """
    jobs = max(1, int(jobs))
    if jobs == 1 or len(cmds) <= 1:
        for cmd in cmds:
            subprocess.run(cmd, check=True, cwd=ROOT, env=_cmd_env(cmd, env))
        return
    rcs = [None] * len(cmds)

    def _one(i, cmd):
        rcs[i] = subprocess.run(cmd, cwd=ROOT, env=_cmd_env(cmd, env)
                                ).returncode

    with ThreadPoolExecutor(max_workers=min(jobs, len(cmds))) as ex:
        list(ex.map(lambda t: _one(*t), enumerate(cmds)))
    for cmd, rc in zip(cmds, rcs):
        if rc != 0:
            raise subprocess.CalledProcessError(rc, cmd)


def _repo_path(path: str) -> str:
    """相对路径统一解释为仓库根目录；父进程与子进程必须看到同一位置。"""
    return path if os.path.isabs(path) else os.path.abspath(os.path.join(ROOT, path))


def _seed_tag(seed: int, n_seeds: int) -> str:
    """多种子运行时给结果文件名加 _seed<seed> 后缀；单种子保持旧文件名。"""
    return f"_seed{seed}" if n_seeds > 1 else ""


def _variant_dir(out_root: str, variant: str, seed: int, n_seeds: int) -> str:
    """多种子运行时每个 (变体, 种子) 独立子目录；单种子保持旧布局。"""
    if n_seeds > 1:
        return os.path.join(out_root, variant, f"seed{seed}")
    return os.path.join(out_root, variant)


def verify_controlled_setup(out_root: str, variants, seeds, n_seeds: int,
                            expected_data_digest: str = None) -> dict:
    """核对所有变体 × 种子确实除待验证模块外使用完全相同的训练条件。

    对比实验的公平性前提：优化器/LR 策略/步数/数据顺序等受控变量必须一致，
    且各 run 必须消费同一份数据、同一代码版本、同样的训练步数。该检查把
    “公平对比”从口头约定变成可执行断言，分两层：
     1. config_used.json：除 variant/seed/device/tag 外的字段逐项一致；
     2. eval.json 溯源：数据指纹、代码提交、配置 SHA-256、计划训练步数
        （steps_planned，旧格式回退 steps）跨 run 一致，并与本次流水线的
        数据指纹比对，避免"同一 config 跑了不同数据/代码版本"被静默当成可比。
        实际步数/tokens_seen 默认也必须逐项一致；启用早停（early_stopped）
        时各 run 可以停在不同的步数——这是早停的定义而非条件失控，此时按
        计划步数核对，实际步数逐 run 如实报告、不再要求相同。
     任一层不一致都会报错并列出差异。
    """
    ref = None
    ref_name = None
    mismatches = []
    missing = []
    checked = 0
    meta_keys = ("data_fingerprint", "code_commit", "config_sha256",
                 "steps_planned")
    meta_ref = None
    meta_ref_name = None
    meta_mismatches = []
    meta_missing = []
    meta_checked = 0
    early_stop_runs = []
    actual_meta = []
    for v in variants:
        for seed in seeds:
            vdir = _variant_dir(out_root, v, seed, n_seeds)
            cfg_path = os.path.join(vdir, "config_used.json")
            if not os.path.exists(cfg_path):
                missing.append(f"{v}/seed{seed}")
            else:
                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                if int(cfg.get("seed", seed)) != int(seed):
                    mismatches.append(f"{v}/seed{seed}: config seed={cfg.get('seed')}")
                ctrl = {k: val for k, val in cfg.items()
                        if k not in _AUDIT_EXEMPT_KEYS}
                if ref is None:
                    ref, ref_name = ctrl, f"{v}/seed{seed}"
                elif ctrl != ref:
                    keys = sorted(set(ctrl) | set(ref))
                    diffs = [f"{k}: {ref.get(k)!r} vs {ctrl.get(k)!r}"
                             for k in keys if ref.get(k) != ctrl.get(k)]
                    mismatches.append(f"{v}/seed{seed} 相对 {ref_name} 的差异: "
                                      + "; ".join(diffs[:8]))
                checked += 1
            ev_path = os.path.join(vdir, "eval.json")
            if not os.path.exists(ev_path):
                meta_missing.append(f"{v}/seed{seed}")
                continue
            with open(ev_path, "r", encoding="utf-8") as f:
                ev = json.load(f)
            meta = {k: ev.get(k) for k in meta_keys}
            if meta["steps_planned"] is None:
                meta["steps_planned"] = ev.get("steps")
            actual_meta.append((f"{v}/seed{seed}", ev.get("steps"),
                                ev.get("tokens_seen")))
            if ev.get("early_stopped"):
                early_stop_runs.append(f"{v}/seed{seed}(step {ev.get('steps')}"
                                       f"/{meta['steps_planned']})")
            here = []
            if int(ev.get("seed", seed)) != int(seed):
                here.append(f"eval seed={ev.get('seed')}")
            if (expected_data_digest is not None
                    and meta["data_fingerprint"] != expected_data_digest):
                here.append(f"data_fingerprint={meta['data_fingerprint']!r} "
                            f"!= 本次流水线数据 {expected_data_digest!r}")
            if meta_ref is None:
                meta_ref, meta_ref_name = meta, f"{v}/seed{seed}"
            else:
                diffs = [f"{k}: {meta_ref.get(k)!r} vs {meta.get(k)!r}"
                         for k in meta_keys if meta_ref.get(k) != meta.get(k)]
                if diffs:
                    here.append(f"相对 {meta_ref_name} 的差异: " + "; ".join(diffs))
            if here:
                meta_mismatches.append(f"{v}/seed{seed}: " + "; ".join(here))
            meta_checked += 1
    # 实际步数/tokens_seen：未启用早停时必须严格一致；启用早停时逐 run 可以
    # 不同（已在 early_stop_runs 如实列出），不再作为可比性失败。
    if not early_stop_runs and actual_meta:
        a_ref_name, a_ref_steps, a_ref_tokens = actual_meta[0]
        for name, steps_run, tokens_run in actual_meta[1:]:
            diffs = []
            if a_ref_steps != steps_run:
                diffs.append(f"steps: {a_ref_steps!r} vs {steps_run!r}")
            if a_ref_tokens != tokens_run:
                diffs.append(f"tokens_seen: {a_ref_tokens!r} vs {tokens_run!r}")
            if diffs:
                meta_mismatches.append(f"{name} 相对 {a_ref_name} 的差异: "
                                       + "; ".join(diffs))
    if missing:
        print(f"[audit] 警告：缺少 {len(missing)} 个 run 的 config_used.json，"
              f"无法核对受控变量: {', '.join(missing[:6])}"
              + (" ..." if len(missing) > 6 else ""))
    if meta_missing:
        print(f"[audit] 警告：缺少 {len(meta_missing)} 个 run 的 eval.json，"
              f"无法核对数据指纹/代码版本/训练步数: {', '.join(meta_missing[:6])}"
              + (" ..." if len(meta_missing) > 6 else ""))
    if mismatches or meta_mismatches:
        msg = ("受控变量自检失败：对比实验除待验证模块外必须完全一致，"
               "否则观测差异不能归因于方法本身。\n  - "
               + "\n  - ".join(mismatches + meta_mismatches))
        print(f"[audit] {msg}")
        raise SystemExit(2)
    if checked:
        print(f"[audit] 受控变量自检通过：{checked} 个 run 的优化器/LR/步数等配置一致")
    if meta_checked:
        print(f"[audit] 训练溯源自检通过：{meta_checked} 个 run 的数据指纹/"
              f"代码版本/计划训练步数一致"
              + (f"（数据指纹 {expected_data_digest}）"
                 if expected_data_digest else ""))
    if early_stop_runs:
        print(f"[audit] 早停启用：{len(early_stop_runs)} 个 run 未跑满计划步数"
              f"（{', '.join(early_stop_runs[:6])}"
              + (" ..." if len(early_stop_runs) > 6 else "")
              + "）；实际步数差异已如实记录，按计划步数核对可比性")
    return {"ok": not (mismatches or meta_mismatches), "checked": checked,
            "missing_configs": missing, "mismatches": mismatches,
            "meta_checked": meta_checked, "missing_evals": meta_missing,
            "meta_mismatches": meta_mismatches,
            "early_stop_runs": early_stop_runs,
            "expected_data_digest": expected_data_digest,
            "exempt_keys": sorted(_AUDIT_EXEMPT_KEYS)}


# ---------------------------------------------------------------- 汇总统计

def _seed_key(path: str, data: dict):
    m = _SEED_RE.search(os.path.splitext(os.path.basename(path))[0])
    if m:
        return int(m.group(1))
    s = data.get("seed")
    return int(s) if s is not None else None


def _result_files(results_dir: str, prefix: str, scope_seeds) -> list:
    """按“命名约定优先”的顺序返回 <prefix>_*.json。

    多种子运行的结果带 _seed<seed> 后缀、单种子不带。两种命名可能同时
    存在（如在旧多种子目录上重跑单种子），且经 _seed_key 会落到同一
    (变体, 种子) 槽位；调用方按序赋值、后写覆盖，因此必须让与本次
    约定一致的文件排在最后，否则陈旧的旧命名文件覆盖新结果。
    """
    fps = glob.glob(os.path.join(results_dir, f"{prefix}_*.json"))
    if not scope_seeds:
        return sorted(fps)
    expected_tagged = len(scope_seeds) > 1

    def convention_match(fp: str) -> int:
        stem = os.path.splitext(os.path.basename(fp))[0]
        tagged = _SEED_RE.search(stem) is not None
        return int(tagged == expected_tagged)

    return sorted(fps, key=lambda fp: (convention_match(fp), fp))


def _sorted_seeds(mapping) -> list:
    return sorted(mapping.keys(),
                  key=lambda s: (s is None, s if s is not None else -1))


def _variant_seed_map(runs: dict, variant: str) -> dict:
    """取某变体的 {seed: slot}；已有带种子的结果时忽略无种子的文件，
    同一次实验不重复计为额外种子。"""
    m = (runs or {}).get(variant) or {}
    tagged = {s: slot for s, slot in m.items() if s is not None}
    return tagged if tagged else m


def cross_variant_badcase_map(runs: dict, variants) -> dict:
    """跨变体对齐用：每个变体取“最小种子”的 RULER 错误清单。

    最小种子没有 RULER 结果时（例如只跑了其他种子）退化到最小的有结果的
    种子；已评测且零错误的变体记为 [] 并参与对照。未跑过 RULER 的变体
    不进结果（无法参与对照）。
    """
    out = {}
    for v in variants:
        m = _variant_seed_map(runs, v)
        for s in _sorted_seeds(m):
            r = m[s].get("ruler")
            if r is not None:
                out[v] = r.get("badcases") or []
                break
    return out


def _stat(values) -> dict:
    """跨种子聚合：{mean, std, n}；无有效值时 None。

    n 只计有限值，与 mean_std 的口径一致：NaN/Inf 不参与统计也不虚增
    样本量（否则会显示“n=5”但均值实际只来自 4 个种子）。
    """
    vals = [v for v in values if v is not None]
    m, s = mean_std(vals)
    if m is None:
        return None
    return {"mean": m, "std": s,
            "n": sum(1 for v in vals if math.isfinite(float(v)))}


def _ppl_of(d):
    if not d:
        return None
    v = d.get("ppl", d.get("val_ppl"))
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _ruler_task_means(d):
    """单次 RULER 结果：各任务先在上下文档位求平均（fwe 用 recall，其余 EM）。"""
    out = {}
    for task, by_ctx in ((d or {}).get("results") or {}).items():
        if task.startswith("_") or not isinstance(by_ctx, dict) or not by_ctx:
            continue
        metric = "recall" if task == "fwe" else "em"
        vals = [m[metric] for m in by_ctx.values()
                if isinstance(m, dict) and m.get(metric) is not None]
        if vals:
            out[task] = float(sum(vals) / len(vals))
    return out


def _ruler_avg_of(d):
    tm = _ruler_task_means(d)
    return float(sum(tm.values()) / len(tm)) if tm else None


def _stat_series(seed_map, payload_key, extractor):
    vals = []
    for s in _sorted_seeds(seed_map):
        d = seed_map[s].get(payload_key)
        if d:
            vals.append(extractor(d))
    return _stat(vals)


def _fmt_stat(st, nd=3) -> str:
    if not st or st.get("mean") is None:
        return "-"
    mean = st["mean"]
    if st.get("n", 1) < 2 or st.get("std") is None:
        return f"{mean:.{nd}f} (n=1)"
    return f"{mean:.{nd}f} ± {st['std']:.{nd}f} (n={st['n']})"


def _enrich_paired(res: dict, a, b) -> dict:
    """为配对检验补充稳健性证据：置换检验 p、配对差 bootstrap CI 与 MDE。

    种子数只有 3~5 时 t 检验的正态假设并不可靠：置换检验给出分布无关的
    对照 p 值（两套结论冲突时必须显式标注，而不是静默采信 t 检验）；
    bootstrap CI 展示效应量的不确定范围；MDE 回答"以当前种子方差，
    多大的效应才检测得到"——区分"没有效应"与"功效不足"。
    均为确定性计算（固定随机种子），同一输入恒定复现。
    """
    if res.get("n", 0) >= 1 and res.get("mean_diff") is not None:
        perm = paired_sign_flip_test(a, b)
        res["p_perm"] = perm.get("p_perm")
        res["perm_exact"] = perm.get("exact")
        res["mean_diff_ci95"] = bootstrap_mean_ci(
            [x - y for x, y in zip(a, b)])
        res["mde"] = min_detectable_effect(res.get("std_diff"), res.get("n"))
    else:
        res["p_perm"] = res["perm_exact"] = None
        res["mean_diff_ci95"] = res["mde"] = None
    return res


def _paired_comparisons(runs, pairs, metric_name, extractor, lower_better) -> list:
    """对每个 (候选, 参照) 在共同种子上做配对 t 检验（原始 p，未校正）。"""
    out = []
    for cand, ref in pairs:
        if cand not in runs or ref not in runs:
            continue
        cm = _variant_seed_map(runs, cand)
        rm = _variant_seed_map(runs, ref)
        seeds = [s for s in _sorted_seeds(cm) if s in rm]
        a, b = [], []
        for s in seeds:
            va, vb = extractor(cm[s]), extractor(rm[s])
            if va is None or vb is None:
                continue
            a.append(va)
            b.append(vb)
        if not a:
            continue
        res = paired_t_test(a, b)
        res.update({"candidate": cand, "reference": ref, "metric": metric_name,
                    "lower_is_better": bool(lower_better)})
        out.append(_enrich_paired(res, a, b))
    return out


def _apply_holm(tests, min_seeds=MIN_SEEDS, alpha=ALPHA) -> list:
    """对同一指标家族内的全部配对检验做 Holm–Bonferroni 校正并给出结论。

    汇总同时检验几十组对比；不校正时 FWER 随检验数上升，名义 p<0.05 极易由
    运气产生。结论一律基于同一指标家族内校正后的 p_adj；原始 p 仍保留在
    结果 JSON 中供查阅。

    置换检验的 p_perm 在同一指标家族内单独做 Holm 校正（它与 t 检验是两套
    独立的检验族，合并会双重惩罚）：校正后两套结论是否一致被显式记录为
    t_perm_agree；t 显著而置换不显著（小样本下 t 的正态假设失效的典型症状）
    时在结论里追加警示，而不是让读者只看到一个 p 值。
    """
    by_metric = {}
    for t in tests:
        by_metric.setdefault(t.get("metric"), []).append(t)
    for group in by_metric.values():
        pads = holm_bonferroni([t.get("p") for t in group])
        ppads = holm_bonferroni([t.get("p_perm") for t in group])
        for t, padj, ppadj in zip(group, pads, ppads):
            t["p_adj"] = padj
            t["p_perm_adj"] = ppadj
            if t.get("p") is None:
                t["verdict"] = f"无法检验（n={t['n']} 或方差退化），不作判定"
            elif t["n"] < min_seeds:
                t["verdict"] = f"种子不足（n={t['n']}<{min_seeds}），无法判定"
            elif t.get("p_adj") is not None and t["p_adj"] < alpha:
                better = ((t["mean_diff"] < 0) if t["lower_is_better"]
                          else (t["mean_diff"] > 0))
                t["verdict"] = ("显著优于（Holm 校正后）" if better
                                else "显著劣于（Holm 校正后）")
                if ppadj is not None and ppadj >= alpha:
                    t["verdict"] += "；⚠ 置换检验不显著（见注）"
            else:
                t["verdict"] = ("不显著（Holm 校正后 p="
                                f"{t['p_adj']:.4f}）"
                                if t.get("p_adj") is not None else "不显著")
                if ppadj is not None and ppadj < alpha:
                    t["verdict"] += "；置换检验显著（边缘证据，建议补种子确认）"
            t["t_perm_agree"] = (
                None if (t.get("p_adj") is None or ppadj is None
                        or t["n"] < min_seeds)
                else (t["p_adj"] < alpha) == (ppadj < alpha))
    return tests


# 组件有效性判定所用的检验对：组件 -> (去一法检验, 全因子检验)。
# 去一法: Full−X vs Full；全因子: 仅 X vs 无门控（standard）。
COMPONENT_TESTS = (
    ("A 查询项", ("lega_no_query", "lega"), ("qgate", "standard")),
    ("B 集中度证据", ("lega_no_conc", "lega"), ("lega_conc_only", "standard")),
    ("C 长度校准", ("lega_no_len", "lega"), ("lega_len_only", "standard")),
)
_EVIDENCE_METRICS = ("PPL", "RULER平均")


def _find_test(tests, pair, metric):
    for t in tests:
        if ((t.get("candidate"), t.get("reference")) == pair
                and t.get("metric") == metric):
            return t
    return None


def _effect_status(t, removing: bool) -> dict:
    """单条配对检验 -> “组件有用”的证据状态（须方向正确且 Holm 校正后显著）。

    removing=True（去一法）：候选=去掉组件的 Full−X，显著变差才算证据；
    removing=False（全因子）：候选=仅含该组件的 X，显著变好才算证据。
    """
    if t is None:
        return {"supported": False, "reason": "无对应检验（变体未运行）",
                "p_adj": None}
    if t.get("n") is not None and t["n"] < MIN_SEEDS:
        return {"supported": False, "reason": f"种子不足（n={t['n']}）",
                "p_adj": t.get("p_adj")}
    if (t.get("mean_diff") is None or t.get("p_adj") is None
            or t.get("p_adj") >= ALPHA):
        return {"supported": False, "reason": "Holm 校正后不显著",
                "p_adj": t.get("p_adj")}
    lb = bool(t.get("lower_is_better"))
    if removing:
        evidence = (t["mean_diff"] > 0) if lb else (t["mean_diff"] < 0)
    else:
        evidence = (t["mean_diff"] < 0) if lb else (t["mean_diff"] > 0)
    return {"supported": bool(evidence),
            "reason": "显著" if evidence else "方向相反（去掉反而更好/加入反而更差）",
            "p_adj": t.get("p_adj")}


def component_evidence(tests) -> list:
    """由配对检验结果汇总各组件（A/B/C）是否“有用”的统计证据。

    组件有用的判定必须有可检验的显著证据，而不是组合是否跑全：
    去一法（Full−X 显著变差）或全因子（仅 X 显著优于无门控）任一方向成立
    才算“有证据支持”；所有方向均不显著/种子不足则记“证据不足”。调用方
    据此决定能否下“各组件均有用”的结论。
    """
    out = []
    for name, loo_pair, add_pair in COMPONENT_TESTS:
        item = {"component": name, "loo_pair": list(loo_pair),
                "add_pair": list(add_pair), "loo": {}, "add": {},
                "supported": []}
        best_detail = {}
        for metric in _EVIDENCE_METRICS:
            loo = _effect_status(_find_test(tests, loo_pair, metric), True)
            add = _effect_status(_find_test(tests, add_pair, metric), False)
            item["loo"][metric] = loo
            item["add"][metric] = add
            if loo["supported"] or add["supported"]:
                item["supported"].append(metric)
                best_detail[metric] = {
                    "direction": "loo" if loo["supported"] else "add",
                    "p_adj": (loo if loo["supported"] else add)["p_adj"]}
        item["evidence"] = best_detail
        item["verdict"] = "有证据支持" if item["supported"] else "证据不足"
        out.append(item)
    return out


# 机制主张的检验族：门控完整 logit 与仅查询项的 AUROC / PR-AUC 之差
# （越大越好）。两族必须分别检验：hit 判别通常类别极不平衡，AUROC 会因
# 负样本基数巨大而乐观，PR-AUC 是此时更贴近研究目标的口径。
AUROC_DIFF_METRIC = "AUROC差(logit−仅查询)"
AP_DIFF_METRIC = "AP差(logit−仅查询)"
AUROC_DIFF_REFERENCE = "仅查询项基线"


def _detection_diff_tests(runs, key_cand: str, metric: str) -> list:
    """同一 run 内 key_cand(logit) − key_cand(qterm) 的跨种子配对检验。

    “完整门控 logit 比仅查询项更能判断该不该写入”是本方法的核心机制
    主张，但把两个并排的数放在表里不构成证据：它们来自同一批记录、同一
    批样例，必须作为配对差值检验。每个 (变体, 种子) run 的差值当作一次
    观测（H0：差为 0），单一侧缺失的种子整对剔除，n 如实反映有效配对数。
    全家族与 PPL/RULER 一样走 Holm 校正。逐 run 的样例聚类 bootstrap
    区间（analysis JSON 的 auroc_cluster_bootstrap）与本跨种子检验互为
    印证：前者回答“换样例波动多少”，后者回答“换训练种子方向是否稳定”。
    """
    out = []
    for v in sorted(runs):
        if v not in VARIANTS:
            continue
        m = _variant_seed_map(runs, v)
        seeds = _sorted_seeds(m)
        if not seeds:
            continue
        a, b = [], []
        for s in seeds:
            an = m[s].get("analysis") or {}
            al, aq = an.get(key_cand + "_logit"), an.get(key_cand + "_qterm")
            if al is None or aq is None:
                continue
            a.append(al)
            b.append(aq)
        if not a:
            continue
        res = paired_t_test(a, b)
        res.update({"candidate": v, "reference": AUROC_DIFF_REFERENCE,
                    "metric": metric, "lower_is_better": False})
        out.append(_enrich_paired(res, a, b))
    return out


def _auroc_diff_tests(runs) -> list:
    """AUROC(logit)−AUROC(qterm) 配对差检验（H6 的判定依据）。"""
    return _detection_diff_tests(runs, "auroc", AUROC_DIFF_METRIC)


def _ap_diff_tests(runs) -> list:
    """PR-AUC(logit)−PR-AUC(qterm) 配对差检验（不平衡下的主指标口径，H7）。"""
    return _detection_diff_tests(runs, "auprc", AP_DIFF_METRIC)


# 注册假设（confirmatory）：在正式实验前登记的方向性主张；未登记的对比
# 一律视为事后（exploratory）分析，只能作为提出新假设的线索。
# 登记表与防篡改锁链的实现见 lega/hypotheses.py（锁基线 + 修订链）。


def load_registered_hypotheses(path: str = HYPOTHESES_PATH) -> dict:
    """读取预注册假设文件；缺失/损坏时返回空 dict（汇总页显式提示）。"""
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
    except (OSError, ValueError) as e:
        print(f"[summary] 警告：注册假设文件不可读（{e}），跳过判定")
        return {}
    if not isinstance(d, dict):
        print("[summary] 警告：注册假设文件顶层不是 JSON 对象，跳过判定")
        return {}
    if not isinstance(d.get("hypotheses"), list) or not d["hypotheses"]:
        print("[summary] 警告：注册假设文件无 hypotheses 条目，跳过判定")
        return {}
    return d


def evaluate_registered_hypotheses(registered: dict, tests: list,
                                    alpha: float = ALPHA) -> list:
    """把预注册假设逐条对照 Holm 校正后的配对检验，判定 支持/拒绝/未定。

    “拒绝”只在方向相反且校正后显著时给出（假设被证伪）；
    “未定”包括种子不足、对比未运行、校正后不显著——不显著绝不等于
    “假设成立”，也不等于“无效应”，配合 MDE 才能判断是功效不足还是
    真的没有效应。
    """
    if not registered:
        return []
    out = []
    for h in registered.get("hypotheses", []):
        t = _find_test(tests, (h.get("candidate"), h.get("reference")),
                       h.get("metric"))
        item = {"id": h.get("id"), "claim": h.get("claim"),
                "metric": h.get("metric"),
                "candidate": h.get("candidate"), "reference": h.get("reference"),
                "direction": h.get("direction"),
                "p_adj": None, "p_perm_adj": None, "mean_diff": None,
                "n": None, "mde": None, "status": None, "detail": None}
        if t is None:
            item["status"] = "未检验"
            item["detail"] = "对应的对比/指标没有检验结果（run 缺失或未评测）"
        elif t.get("n") is not None and t["n"] < MIN_SEEDS:
            item["status"] = "未定"
            item["detail"] = f"种子不足（n={t['n']}<{MIN_SEEDS}）"
        elif t.get("p_adj") is None:
            item["status"] = "未定"
            item["detail"] = "无法检验（方差退化或指标缺失）"
        elif t["p_adj"] >= alpha:
            item["status"] = "未定"
            item["detail"] = "Holm 校正后不显著"
            if t.get("mde") is not None and t.get("mean_diff") is not None:
                item["detail"] += (f"（观测差 {t['mean_diff']:+.4f}，"
                                   f"当前种子数可检测的最小效应 "
                                   f"{t['mde']:+.4f}）")
        else:
            better = (t["mean_diff"] < 0 if t["lower_is_better"]
                      else t["mean_diff"] > 0)
            supported = (better if h.get("direction") == "candidate_better"
                         else not better)
            item["status"] = "支持" if supported else "拒绝"
            item["detail"] = ("方向与登记一致且校正后显著" if supported
                              else "校正后显著但方向与登记相反")
            if t.get("p_perm_adj") is not None and t["p_perm_adj"] >= alpha:
                item["detail"] += "；⚠ 置换检验不显著，小样本下需谨慎"
        if t is not None:
            item.update({"p_adj": t.get("p_adj"),
                         "p_perm_adj": t.get("p_perm_adj"),
                         "mean_diff": t.get("mean_diff"),
                         "n": t.get("n"), "mde": t.get("mde")})
        out.append(item)
    return out


def apply_integrity_gate(hyp_eval: list, integrity: dict) -> list:
    """预注册完整性失败时给全部确认性判定加盖“无效”戳。

    预注册的全部意义在于事后不能静默改写；一旦 lega/hypotheses.py 的
    锁链校验发现登记表被删/改/无留痕新增，这些判定就不再具有 confirmatory
    地位——逐条标注 valid 标志与原因，汇总页显式降级为探索性，
    而不是假装边界仍然成立。
    """
    valid = bool((integrity or {}).get("confirmatory_valid"))
    reason = ("登记表完整性校验通过" if valid else
              "预注册完整性校验未通过（登记表被删/改/无留痕新增，"
              "或锁文件缺失）：判定降级为探索性，不得作为 confirmatory 结论")
    for h in hyp_eval or []:
        h["confirmatory_valid"] = valid
        h["confirmatory_note"] = reason
    return hyp_eval


# 预注册 scope 核对的配置字段：模型/训练规模（结论可迁移性的前提）。
# lr/weight_decay 刻意排除：按本项目协议，基线调优（--tune_baseline）会在
# 验证集上选定统一超参并注入全部变体，属预注册流程的一部分而非范围外改动。
_SCOPE_CONFIG_KEYS = ("d_model", "n_layers", "n_heads", "ffn_mult",
                      "block_size", "steps", "train_ctx")


def registration_scope_check(registered: dict, repro: dict,
                             root: str = ROOT) -> dict:
    """核对“本次 run 是否落在预注册 scope 内”（confirmatory 资格的前提）。

    预注册假设只在登记的实验范围内具有确认性地位：用 tiny 配置、冒烟
    步数、字符伪数据或其它规模跑出的“支持/拒绝”不能冒充预注册结论。
      * smoke 运行 = 明确的范围外；
      * 配置的模型/训练规模字段逐项比对（lr/wd 除外，见上）；
      * scope.data 声明 gpt2 BPE 时核对实际数据分词器（伪数据/换词表会
        改变目标语言建模任务本身，必须显形）；
      * 额外（未登记）种子只列为 warning：种子可交换、不直接推翻判定，
        但偏离登记计划必须显式可见，由读者知情。
    返回 {checked, in_scope, problems, warnings, scope}；无法核对（缺
    repro/config 文件）时 checked=False、in_scope=None，由
    apply_scope_gate 按“不可确认”处理，而不是默认通过。
    """
    scope = (registered or {}).get("scope") or {}
    out = {"scope": scope, "checked": False, "in_scope": None,
           "problems": [], "warnings": []}
    if not scope:
        out["warnings"].append("登记表未提供 scope 字段，无法核对实验范围")
        return out
    if not repro:
        out["warnings"].append("缺少 repro.json，无法核对实验范围")
        return out

    def _load_json(p):
        if not isinstance(p, (str, os.PathLike)):
            return None
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            return None

    checked_any = False
    if repro.get("smoke"):
        out["problems"].append("冒烟运行（--smoke）不在预注册范围（正式配置/全部种子）内")
        checked_any = True

    want_rel = str(scope.get("config") or "")
    want_path = (want_rel if os.path.isabs(want_rel)
                 else os.path.join(root, want_rel))
    got_rel = repro.get("config_file")
    # 与 scope.config 对称：相对路径按项目根解析（否则会相对 cwd 去读另一个
    # 同名文件，悄悄核对错配置）
    got_path = (os.path.join(root, got_rel)
                if isinstance(got_rel, str) and got_rel
                and not os.path.isabs(got_rel) else got_rel)
    want_cfg, got_cfg = (_load_json(want_path) if want_path else None,
                         _load_json(got_path) if got_path else None)
    if want_cfg is None:
        out["warnings"].append(
            f"登记 scope.config 不可读（{want_rel or '-'}），跳过配置规模核对")
    elif got_cfg is None:
        out["warnings"].append(f"本次配置不可读（{got_path or '-'}），跳过配置规模核对")
    else:
        diffs = [f"{k}: 登记={want_cfg.get(k)} vs 实际={got_cfg.get(k)}"
                 for k in _SCOPE_CONFIG_KEYS
                 if k in want_cfg and want_cfg.get(k) != got_cfg.get(k)]
        if diffs:
            out["problems"].append(
                "模型/训练规模与预注册配置不一致（" + "；".join(diffs) + "）")
        checked_any = True

    scope_data = str(scope.get("data") or "")
    actual_tok = ((repro.get("data") or {}).get("tokenizer")
                  if isinstance(repro.get("data"), dict) else None)
    if "gpt2" in scope_data.lower() and actual_tok is not None \
            and "gpt2" not in str(actual_tok).lower():
        out["problems"].append(
            f"数据分词器与预注册不一致（登记含 gpt2 BPE，实际 {actual_tok}）")
        checked_any = True

    want_seeds = scope.get("seeds")
    got_seeds = repro.get("seeds")
    if isinstance(want_seeds, list) and isinstance(got_seeds, list):
        extra = [s for s in got_seeds if s not in want_seeds]
        if extra:
            out["warnings"].append(
                f"本次含未登记种子 {extra}：判定基于登记计划外的抽样，证据强度打折")
        checked_any = True
    out["checked"] = bool(checked_any)
    if out["problems"]:
        out["in_scope"] = False
    elif out["checked"]:
        out["in_scope"] = True
    return out


def apply_scope_gate(hyp_eval: list, scope_check: dict) -> list:
    """预注册范围核对失败/无法核对时，确认性判定显式降级。

    与完整性门（apply_integrity_gate）同义：登记表没被篡改，但本次 run
    根本不在登记范围内（如 tiny 配置、冒烟步数、伪数据、无法核对），
    此时“支持/拒绝”同样不作数——降级为探索性线索，避免范围外结果
    冒充预注册结论。
    """
    scope_check = scope_check or {}
    if not scope_check.get("scope"):
        # 登记表未声明 scope（历史/精简登记表）：无从核对，仅由汇总页告警，
        # 不因“没有可核对的字段”本身降级已通过的完整性判定。
        return hyp_eval
    problems = scope_check.get("problems") or []
    if not problems and scope_check.get("checked"):
        return hyp_eval
    reason = ("本次运行超出预注册范围：" + "；".join(problems)
              if problems else
              "无法核对本次运行是否落在预注册范围（缺少 repro/config），"
              "确认性判定不可确认")
    for h in hyp_eval or []:
        if h.get("confirmatory_valid") is False:
            continue  # 已因完整性失败降级，保留原原因
        h["confirmatory_valid"] = False
        h["confirmatory_note"] = reason
    return hyp_eval


def _render_registration_integrity(integrity: dict,
                                   scope_check: dict = None) -> list:
    """预注册完整性小节：删除/静默改写必须在校验中显形。"""
    integrity = integrity or {}
    st = integrity.get("status")
    icon = {"intact": "✅", "amended": "✅", "violated": "⛔",
            "missing_lock": "⚠", "missing_registry": "⛔"}.get(st, "?")
    desc = {
        "intact": "与锁基线一致（登记后未发生任何修订）",
        "amended": "与修订链一致（既有假设未被改写/删除，"
                   f"链上共 {integrity.get('chain_length')} 环留痕修订）",
        "violated": "完整性校验失败：登记表被删除/改写，或新增假设未留痕",
        "missing_lock": "缺少锁文件，无法验证登记表未被静默改写",
        "missing_registry": "登记表文件缺失，预注册边界不存在",
    }.get(st, "无法判定")
    lines = ["### 预注册完整性（防事后静默改写）", "",
             f"{icon} 状态：`{st}` —— {desc}。"]
    sha = integrity.get("registry_sha256")
    if sha:
        lines.append(f"- 登记表 SHA-256：`{sha[:16]}…`（假设 "
                     f"{integrity.get('n_hypotheses')} 条"
                     + (f"；锁基线 `{str(integrity.get('baseline_document_sha256'))[:16]}…`"
                        if integrity.get("baseline_document_sha256") else "")
                     + "）")
    else:
        lines.append("- 登记表 SHA-256：不可用（文件缺失或不可解析）")
    for p in (integrity.get("problems") or []):
        lines.append(f"- ⛔ {p}")
    for w in (integrity.get("warnings") or []):
        lines.append(f"- ⚠ {w}")
    # 预注册范围核对：完整性通过不等于本轮 run 落在登记范围内
    # （tiny 配置/冒烟步数/伪数据跑出的“支持”同样不能冒充确认性结论）。
    sc = scope_check or {}
    scope = sc.get("scope") or {}
    scope_failed = False
    if scope or sc:
        # 只有登记表声明了 scope 时才谈得上“范围核对失败”
        scope_failed = bool(scope) and (
            bool(sc.get("problems")) or not sc.get("checked"))
        lines.append("")
        lines.append("#### 预注册范围核对（确认性资格的前提）")
        if sc.get("problems"):
            lines.append("- ⛔ 本次运行超出预注册范围：")
            for p in sc["problems"]:
                lines.append(f"  - {p}")
        elif sc.get("checked"):
            cfg_note = (f"模型/训练规模与 `{scope.get('config')}` 一致"
                        if scope.get("config") else "配置已核对")
            tok_note = ("；数据分词器与 scope.data 一致"
                        if "gpt2" in str(scope.get("data") or "").lower()
                        else "")
            lines.append(f"- ✅ 本次 run 在预注册范围内：{cfg_note}{tok_note}。")
        else:
            # 无 scope 的登记表不降级（与 apply_scope_gate 的早退一致），
            # 只有“声明了 scope 却核不动”才是不可确认
            lines.append("- ⚠ 无法核对预注册范围（缺少 repro/config 信息）："
                         + ("确认性判定按“不可确认”降级。" if scope else
                            "登记表未声明 scope，完整性判定不受影响。"))
        for w in sc.get("warnings") or []:
            lines.append(f"- ⚠ {w}")
    if integrity.get("confirmatory_valid") and not scope_failed:
        lines.append("- 修订规则：既有假设只增不改不删；新增假设必须经 "
                     "`python -m lega.hypotheses --amend --reason ...` 留痕，"
                     "否则判违反。")
    else:
        lines.append("- ⚠ **本页全部“注册假设判定”因完整性/范围核对未通过而"
                     "失效（降级为探索性分析）**：排除问题后重跑汇总。")
    lines.append("")
    return lines


def _fmt_rate(x) -> str:
    if x is None:
        return "-"
    try:
        return f"{float(x):.3%}"
    except (TypeError, ValueError):
        return "-"


def _fmt4(x) -> str:
    """四位小数；None 显式显示 '-'（如空 split 的分布漂移统计为 null）。"""
    if x is None:
        return "-"
    try:
        return f"{float(x):.4f}"
    except (TypeError, ValueError):
        return "-"


def _has_test_split(data_dir: str) -> bool:
    """数据目录是否有非空 test.bin（与 lega.evaluate 的 --split auto 同规则）。"""
    p = os.path.join(data_dir, "test.bin")
    return os.path.exists(p) and os.path.getsize(p) > 0


def resolve_ngram_split(data_dir: str, requested: str = "auto") -> str:
    """把 --ngram_split 的 auto 解析为与神经模型评测完全一致的 split。

    口径不一致的陷阱：神经模型的 PPL 默认在 held-out test 上评测
    （存在 test.bin 时），而 n-gram 基线若固定在 val 上算，汇总页给出的
    “神经模型 vs 统计基线”就是 test-vs-val 的错位比较（数据集切片不同、
    难度不同，绝对差值不可解释）。auto 与 lega.evaluate 的 --split auto
    使用同一条判定规则：test.bin 存在且非空 → test，否则回退 val。
    """
    if requested == "auto":
        return "test" if _has_test_split(data_dir) else "val"
    return requested


def ngram_baseline_comparison(runs: dict, results_dir: str) -> dict:
    """神经变体 vs 简单统计基线（同 split）的统计对比。

    科研方法问题：“n-gram PPL=xx” 与 “神经 PPL=yy” 并排放着不构成
    “复杂模型比简单基线好”的证据——必须同 split、必须对照方差。n-gram
    对给定 split 是确定值（无种子方差），因此每个神经变体的逐种子 PPL
    与同一常数配对 = 单样本检验；只在两侧 split 一致的种子上比较，
    split 不一致的 run 显式列出且不混入（避免 test-vs-val 错位）。
    多个变体同时对比时沿用 Holm–Bonferroni 校正。该对比不属于预注册
    假设，标为补充性/探索性证据：用于回答“数据绝对难度有多高、神经
    模型是否真的优于无梯度基线”，而不是替代变体间的受控对比。
    """
    path = os.path.join(results_dir, "ngram_baseline.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            ng = json.load(f) or {}
    except (OSError, ValueError):
        return {}
    ppl_map = ng.get("ppl") or {}
    ref_name = next((k for k in ("bigram", "unigram")
                     if ppl_map.get(k) is not None), None)
    if ref_name is None:
        return {"error": "ngram_baseline.json 没有可用 PPL"}
    try:
        ref_ppl = float(ppl_map[ref_name])
    except (TypeError, ValueError):
        return {"error": "ngram PPL 非数值"}
    ng_split = ng.get("split")
    rows, incomparable = [], []
    for v in sorted(runs):
        if v not in VARIANTS:
            continue
        m = _variant_seed_map(runs, v)
        vals, seeds_used, eval_splits, excl_seeds = [], [], set(), []
        for s in _sorted_seeds(m):
            d = m[s].get("eval")
            if not d:
                continue
            ev_split = d.get("split", "val")
            if ev_split != ng_split:
                eval_splits.add(ev_split)
                excl_seeds.append(s)
                continue
            p = _ppl_of(d)
            if p is None or not math.isfinite(p):
                continue
            vals.append(p)
            seeds_used.append(s)
        if eval_splits:
            # 只要存在 split 错位的种子就必须显式列出（哪怕同变体另有可比
            # 种子已被纳入）：静默丢弃部分种子同样会让读者高估证据完整性
            incomparable.append({"variant": v,
                                 "eval_splits": sorted(eval_splits),
                                 "excluded_seeds": [int(s) for s in excl_seeds]})
        if not vals:
            continue
        res = paired_t_test(vals, [ref_ppl] * len(vals))
        res.update({"candidate": v, "reference": ref_name,
                    "metric": "PPL vs ngram", "lower_is_better": True})
        res = _enrich_paired(res, vals, [ref_ppl] * len(vals))
        mean_ppl = float(sum(vals) / len(vals))
        rows.append({
            "variant": v, "n": len(vals), "seeds": [int(s) for s in seeds_used],
            "mean_ppl": mean_ppl,
            "std_ppl": mean_std(vals)[1],
            "ratio": mean_ppl / ref_ppl if ref_ppl > 0 else None,
            "reduction_pct": (1.0 - mean_ppl / ref_ppl) * 100.0
            if ref_ppl > 0 else None,
            "p": res.get("p"), "p_adj": None,
            "mean_diff": res.get("mean_diff"),
            "mean_diff_ci95": res.get("mean_diff_ci95"),
            "p_perm": res.get("p_perm"), "mde": res.get("mde"),
            "verdict": None,
        })
    if not rows:
        return {"split": ng_split, "reference_model": ref_name,
                "ngram_ppl": ref_ppl, "rows": [],
                "incomparable": incomparable,
                "note": "没有与 n-gram 同 split 的神经评测结果可比"}
    pads = holm_bonferroni([r.get("p") for r in rows])
    for r, padj in zip(rows, pads):
        r["p_adj"] = padj
        if r["n"] < MIN_SEEDS:
            r["verdict"] = f"种子不足（n={r['n']}<{MIN_SEEDS}）"
        elif padj is None:
            r["verdict"] = "无法检验"
        elif padj < ALPHA and r["mean_ppl"] < ref_ppl:
            r["verdict"] = "显著优于统计基线"
        elif padj < ALPHA:
            r["verdict"] = "显著劣于统计基线（严重）"
        else:
            r["verdict"] = "无显著差异（对照 MDE）"
    return {"split": ng_split, "reference_model": ref_name,
            "ngram_ppl": ref_ppl, "rows": rows, "incomparable": incomparable,
            "holm_family_size": len(rows),
            "note": ("n-gram 无种子方差：逐种子 PPL 与同一常数配对 = 单样本检验；"
                     "本对比为补充性/探索性证据，不替代变体间受控对比")}


def _probe_oof(analysis: dict, key: str):
    """从机制分析 JSON 取线性探针主特征集的 out-of-fold 指标。"""
    lp = (analysis or {}).get("linear_probe") or {}
    fs = (lp.get("feature_sets") or {}).get(lp.get("primary") or "") or {}
    return (fs.get("oof") or {}).get(key)


# 机制判别的简单学习基线对照（补充性/探索性）：门控 logit 是语言建模的
# 副产品、训练时从未见过 hit 标签；线性探针在同一批记录上用 hit 标签监督
# 训练（按样例分组 CV、λ 按 OOF log-loss 调优，见 lega/probe.py）。
# Δ = 探针 − 门控：>0 表示简单监督基线更好（门控的“判别优越性”不成立），
# ≈0 表示门控与最简线性模型打平，<0 才是门控在判别上显著更好。
LINEAR_PROBE_AP_METRIC = "PR-AUC差(探针−门控)"
LINEAR_PROBE_AUROC_METRIC = "AUROC差(探针−门控)"


def linear_probe_baseline_comparison(runs: dict) -> dict:
    """门控 logit vs 线性探针简单监督基线的逐种子配对对照。

    没有这个对照时，“门控 logit 的 AUROC/PR-AUC 高”可能只说明廉价特征
    （log max_p、log n、qterm）本身线性可分，复杂门控结构的贡献无从谈起。
    探针是**有标签监督**的简单基线：探针超过门控属预期，门控必须在
    主指标上不被探针显著超越，才有资格谈“更能判断该不该写入”。
    单统计量规则（max_p）一并列出，作为最低档次的规则系统对照。
    """
    rows, tests_ap, tests_aud, missing = [], [], [], []
    for v in sorted(runs):
        if v not in VARIANTS:
            continue
        m = _variant_seed_map(runs, v)
        per_seed, has_analysis = [], False
        for s in _sorted_seeds(m):
            an = m[s].get("analysis") or {}
            if an:
                has_analysis = True
            lp = an.get("linear_probe") or {}
            fs = (lp.get("feature_sets") or {}).get(lp.get("primary") or "") or {}
            oof = fs.get("oof") or {}
            if oof.get("auprc") is None or oof.get("auroc") is None:
                continue
            per_seed.append({
                "seed": int(s),
                "probe_auprc": float(oof["auprc"]),
                "probe_auroc": float(oof["auroc"]),
                "gate_auprc": an.get("auprc_logit"),
                "gate_auroc": an.get("auroc_logit"),
                "rule_auprc": an.get("auprc_maxp"),
                "rule_auroc": an.get("auroc_maxp"),
                "base_rate": an.get("hit_base_rate"),
                "probe_l2": fs.get("selected_l2"),
                "probe_features": fs.get("features"),
                "folds": lp.get("folds"),
                "primary_detection_metric": lp.get("primary_detection_metric"),
                "imbalance_level": lp.get("imbalance_level"),
            })
        if not per_seed:
            if has_analysis:
                missing.append({"variant": v,
                                "reason": "analysis JSON 缺少 linear_probe"
                                          "（旧产物或 --no_probe）"})
            continue

        def _ms(key):
            vals = [r[key] for r in per_seed if r.get(key) is not None]
            return mean_std(vals) if vals else (None, None)

        probe_ap_m, probe_ap_s = _ms("probe_auprc")
        gate_ap_m, gate_ap_s = _ms("gate_auprc")
        probe_auc_m, probe_auc_s = _ms("probe_auroc")
        gate_auc_m, gate_auc_s = _ms("gate_auroc")
        rule_ap_m, _ = _ms("rule_auprc")
        rule_auc_m, _ = _ms("rule_auroc")
        base_m, _ = _ms("base_rate")

        def _test(key_p, key_g, metric):
            a, b = [], []
            for r in per_seed:
                if r.get(key_p) is None or r.get(key_g) is None:
                    continue
                a.append(r[key_p])
                b.append(r[key_g])
            if not a:
                return None
            res = paired_t_test(a, b)
            res.update({"candidate": v, "reference": "linear_probe",
                        "metric": metric, "lower_is_better": False})
            return _enrich_paired(res, a, b)

        t_ap = _test("probe_auprc", "gate_auprc", LINEAR_PROBE_AP_METRIC)
        t_aud = _test("probe_auroc", "gate_auroc", LINEAR_PROBE_AUROC_METRIC)
        if t_ap:
            tests_ap.append(t_ap)
        if t_aud:
            tests_aud.append(t_aud)
        d_ap = [r["probe_auprc"] - r["gate_auprc"] for r in per_seed
                if r.get("gate_auprc") is not None]
        d_auc = [r["probe_auroc"] - r["gate_auroc"] for r in per_seed
                 if r.get("gate_auroc") is not None]
        primary = next((r.get("primary_detection_metric") for r in per_seed
                        if r.get("primary_detection_metric")), None)
        rows.append({
            "variant": v, "n_seeds": len(per_seed),
            "seeds": [r["seed"] for r in per_seed],
            "imbalance_level": next((r.get("imbalance_level") for r in per_seed
                                     if r.get("imbalance_level")), None),
            "hit_base_rate": base_m,
            "primary_detection_metric": primary,
            "features": next((r.get("probe_features") for r in per_seed
                              if r.get("probe_features")), None),
            "probe_l2": next((r.get("probe_l2") for r in per_seed
                              if r.get("probe_l2") is not None), None),
            "folds": next((r.get("folds") for r in per_seed
                           if r.get("folds") is not None), None),
            "rule_auprc": rule_ap_m, "probe_auprc": probe_ap_m,
            "probe_auprc_std": probe_ap_s,
            "gate_auprc": gate_ap_m, "gate_auprc_std": gate_ap_s,
            "rule_auroc": rule_auc_m, "probe_auroc": probe_auc_m,
            "probe_auroc_std": probe_auc_s,
            "gate_auroc": gate_auc_m, "gate_auroc_std": gate_auc_s,
            "delta_auprc": (sum(d_ap) / len(d_ap)) if d_ap else None,
            "delta_auroc": (sum(d_auc) / len(d_auc)) if d_auc else None,
        })
    tests = _apply_holm(tests_ap + tests_aud)
    for t in tests:
        # _apply_holm 的结论文案面向“候选优于参照”，此处 Δ=探针−门控，
        # 语义相反：Δ>0 是简单监督基线更好，必须改写以免方向误导。
        n = t.get("n") or 0
        if t.get("p") is None:
            t["verdict"] = "无法检验（样本或方差退化），不作判定"
        elif n < MIN_SEEDS:
            t["verdict"] = f"种子不足（n={n}<{MIN_SEEDS}），不作判定"
        elif t.get("p_adj") is not None and t["p_adj"] < ALPHA:
            t["verdict"] = ("⚠ 简单监督基线更好：门控的判别优越性不成立"
                            if t["mean_diff"] > 0 else
                            "门控显著优于最简监督线性基线")
        else:
            t["verdict"] = ("无显著差异（对照 MDE）——打平不等于门控结构"
                            "无用，但也不构成“判别优越性”的证据")
    # 行内证据在 Holm 校正与自定义 verdict 之后回填（校正对象是 tests 里的
    # 同一批 dict；若在循环里先取值会拿到校正前的 None）。
    by_key = {(t.get("candidate"), t.get("metric")): t for t in tests}
    for r in rows:
        t_ap = by_key.get((r["variant"], LINEAR_PROBE_AP_METRIC))
        t_aud = by_key.get((r["variant"], LINEAR_PROBE_AUROC_METRIC))
        r["p_adj_auprc"] = (t_ap or {}).get("p_adj")
        r["p_adj_auroc"] = (t_aud or {}).get("p_adj")
        r["delta_auprc_ci95"] = (t_ap or {}).get("mean_diff_ci95")
        r["delta_auroc_ci95"] = (t_aud or {}).get("mean_diff_ci95")
        r["verdict"] = ((t_ap if r.get("primary_detection_metric") == "PR-AUC"
                         else t_aud) or {}).get("verdict")
    return {
        "metrics": {"ap": LINEAR_PROBE_AP_METRIC,
                    "auroc": LINEAR_PROBE_AUROC_METRIC},
        "rows": rows, "tests": tests, "missing": missing,
        "note": ("线性探针=有标签监督的简单基线（同批记录、按样例分组 CV、"
                 "λ 按 OOF log-loss 网格调优；OOF 指标防“背题”泄漏）；"
                 "门控 logit 为语言建模副产品、训练未见 hit 标签。"
                 "Δ=探针−门控：>0 属预期（判别能力本可由线性模型从廉价"
                 "统计量学得），门控必须在主指标上不被探针显著超越才有"
                 "资格谈“结构优越性”。单统计量规则（max_p）为最低档次"
                 "对照。本对比为补充性/探索性证据，不替代预注册的确认性"
                 "检验。"),
    }


def _render_baseline_sections(results_dir: str, smoke=None,
                              ngram_cmp: dict = None,
                              probe_cmp: dict = None) -> list:
    """基线合理性小节：简单统计基线 + 基线超参数调优证据。

    科研方法问题：如果复杂方法只与“随手设一个学习率”的弱基线比较，
    涨点可能全部来自超参数红利。本小节渲染两类证据：
      1. 简单基线（一元/二元平滑 n-gram）在同一 held-out split 上的 PPL，
         给出绝对难度参照，并（ngram_cmp 提供时）逐变体做同 split 的
         单样本配对检验，回答“神经模型是否显著优于无梯度基线”；
      2. 基线（standard）在验证集上的学习率/权重衰减网格调优结果与
         “默认是否已接近最优”判定。
    缺少调优证据时显式警告“未调优基线”，而不是让读者默认基线已调好。
    smoke 可由调用方传入；为 None 时自行读取 repro.json。
    """
    if smoke is None:
        smoke = False
        rfp = os.path.join(results_dir, "repro.json")
        if os.path.exists(rfp):
            try:
                with open(rfp, "r", encoding="utf-8") as f:
                    smoke = bool((json.load(f) or {}).get("smoke"))
            except (OSError, ValueError):
                smoke = False
    lines = ["", "## 基线合理性（简单基线 + 调优证据）", ""]
    ngram = None
    nfp = os.path.join(results_dir, "ngram_baseline.json")
    if os.path.exists(nfp):
        try:
            with open(nfp, "r", encoding="utf-8") as f:
                ngram = json.load(f)
        except (OSError, ValueError):
            ngram = None
    if ngram:
        uni = ((ngram.get("ppl") or {}).get("unigram"))
        bi = ((ngram.get("ppl") or {}).get("bigram"))
        split = ngram.get("split")
        lines.append(
            f"- 简单统计基线（train 拟合、{split} 评测，无梯度、无调参空间）："
            f"一元平滑 PPL={_fmt_scalar_maybe(uni)}、"
            f"二元平滑 PPL={_fmt_scalar_maybe(bi)}"
            f"（训练 token {int(ngram.get('train_tokens') or 0):,}、"
            f"评测 token {int(ngram.get('eval_tokens') or 0):,}、"
            f"词表 {ngram.get('vocab_size')}）。主科学对比仍是神经变体之间；"
            "该基线只回答“绝对难度有多高、神经模型是否显著优于统计基线”。")
    else:
        lines.append("- 简单统计基线：未运行（`python -m lega.ngram`，"
                     "结果写入 `results/<tag>/ngram_baseline.json`）。")
    # 逐变体 vs 简单基线的同 split 配对对比：只并排报两个数不构成
    # “复杂模型优于简单基线”的证据，必须给出比值、校正后 p 与判定。
    cmp = ngram_cmp or {}
    if cmp.get("rows"):
        ref = cmp.get("reference_model")
        lines += [
            "",
            f"- 神经变体 vs 简单统计基线（{ref}，split={cmp.get('split')}；"
            "n-gram 无种子方差 → 逐种子 PPL 与同一常数配对做单样本检验，"
            f"Holm 校正（家族 {cmp.get('holm_family_size', len(cmp['rows']))} "
            "个变体）；属补充性/探索性对比）：",
            "",
            "| 变体 | n | 神经 PPL(mean) | 统计基线 PPL | 比值 | 相对降低 | p_adj | 判定 |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for r in cmp["rows"]:
            ratio = (_fmt4(r.get("ratio"))
                     if r.get("ratio") is not None else "-")
            red = (f"{r['reduction_pct']:.1f}%"
                   if r.get("reduction_pct") is not None else "-")
            padj = (_fmt4(r.get("p_adj"))
                    if r.get("p_adj") is not None else "-")
            lines.append(
                f"| {r.get('variant')} | {r.get('n')} | "
                f"{_fmt_scalar_maybe(r.get('mean_ppl'))} | "
                f"{_fmt_scalar_maybe(cmp.get('ngram_ppl'))} | {ratio} | {red} "
                f"| {padj} | {r.get('verdict')} |")
        lines += ["",
                  "  注：比值 <1 且判定“显著优于统计基线”才说明神经模型真的"
                  "比无梯度基线好；本对比未预注册，只是补充证据，"
                  "不替代变体间受控配对检验。"]
    elif cmp.get("error"):
        lines.append(f"- ⚠ 简单基线对比不可用：{cmp['error']}")
    elif cmp.get("note") and not cmp.get("rows"):
        lines.append(f"- ⚠ {cmp['note']}")
    for item in cmp.get("incomparable") or []:
        excl = item.get("excluded_seeds") or []
        lines.append(
            f"- ⚠ 变体 `{item.get('variant')}` 的评测 split="
            f"{item.get('eval_splits')} 与 n-gram split={cmp.get('split')} "
            "不一致"
            + (f"的种子 {excl} " if excl else "：")
            + "未纳入比较（不同切片的 PPL 不可比；用 "
            "`--ngram_split auto` 或对齐后再比较）。")
    tune = None
    tfp = os.path.join(results_dir, "baseline_tuning.json")
    if os.path.exists(tfp):
        try:
            with open(tfp, "r", encoding="utf-8") as f:
                tune = json.load(f)
        except (OSError, ValueError):
            tune = None
    if tune:
        best = tune.get("selected") or {}
        dflt = tune.get("default_trial") or {}
        gap = tune.get("selected_minus_default_val_loss")
        sel_seeds = tune.get("selection_seeds") or [tune.get("seed")]
        grid = tune.get("grid") or []
        n_lr = len({c.get("lr") for c in grid}) if grid else "-"
        n_wd = len({c.get("weight_decay") for c in grid}) if grid else "-"
        std = best.get("val_loss_best_std")
        lines.append(
            f"- 基线调优（变体 `{tune.get('variant', 'standard')}`，"
            f"网格 {n_lr} 档 lr × {n_wd} 档 wd = "
            f"{tune.get('n_trials', len(grid) or len(tune.get('trials') or []))} "
            f"组配置、选择用 {len(sel_seeds)} 个种子、trial 步数 "
            f"{tune.get('steps', '-')}（与正式训练同步数与否以此为准）、"
            f"验证集选择、测试集未使用）：最优 lr={best.get('lr')}"
            + (f"、weight_decay={best.get('weight_decay')}"
               if best.get("weight_decay") is not None else "")
            + f"，最优 val_loss={_fmt4(best.get('val_loss_best', best.get('val_loss')))}"
            + (f" ± {_fmt4(std)}（跨选择种子）" if std is not None else "")
            + f"；默认 lr={dflt.get('lr')} 的 val_loss="
            f"{_fmt4(dflt.get('val_loss_best', dflt.get('val_loss')))}"
            f"（最优−默认={_fmt4(gap)}）。")
        if len(sel_seeds) < 2:
            lines.append("- ⚠ 网格选择只用 1 个种子：val_loss 的种子噪声可能让"
                         "“最优配置”在下一次抽样中翻转（winner's curse）；"
                         "多组接近时应改用 `--tune_seeds`（多带几个种子按均值"
                         "选择）后重跑，不要据此宣称基线已充分调优。")
        if tune.get("default_is_optimal") is True:
            lines.append("- ✅ 默认超参数即为网格最优/在容差内：基线的对比不受"
                         "“选了不利学习率”的质疑。全流水线所有变体使用同一"
                         "（已调优的）配置，受控变量保持一致。")
        elif tune.get("default_is_optimal") is False:
            lines.append("- 默认超参数并非网格最优：全流水线已改用上述最优"
                         "配置训练**所有**变体（受控变量一致），因此与基线的"
                         "对比不是“调好的新方法 vs 没调过的基线”。")
        else:
            lines.append("- ⚠ 默认组合的 trial 失败或缺失，无法判定默认是否"
                         "近优；请核对 `baseline_tuning.json` 的 trials 记录。")
    else:
        lines.append("- 基线调优：未运行。"
                     + ("（冒烟模式不要求。）" if smoke else
                        "⚠ 未提供基线调优证据：若基线（standard）没有经过"
                        "至少一档学习率/权重衰减的网格验证，方法涨点可能只是"
                        "超参数红利。请用 `run_all.py --tune_baseline` 或在"
                        "`results/<tag>/baseline_tuning.json` 提供证据。"))
    # 机制判别的“最简监督线性模型”对照：没有它，“门控 logit 判别力高”
    # 无法排除“廉价特征本身线性可分”。探针 OOF 有标签监督、门控无标签；
    # Δ=探针−门控，>0 属预期，门控不被显著超越才谈得上结构优越性。
    probe = probe_cmp or {}
    if probe.get("rows"):
        lines += [
            "",
            "- 机制判别的简单学习基线（线性探针，OOF；与门控 logit 逐种子"
            "配对、Holm 校正；属补充性/探索性对比）：",
            "",
            "| 变体 | n | 命中基线率 | 特征集 | PR-AUC规则(max_p) | "
            "PR-AUC探针(OOF) | PR-AUC门控(logit) | AUROC探针(OOF) | "
            "AUROC门控(logit) | ΔPR-AUC(探针−门控) | p_adj | 判定 |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for r in probe["rows"]:
            feat = r.get("features")
            feat_s = "+".join(str(x) for x in feat) if feat else "-"
            delta = r.get("delta_auprc")
            ci = r.get("delta_auprc_ci95")
            delta_s = ("-" if delta is None else
                       f"{delta:+.3f}"
                       + (f" [{(ci or [None, None])[0]:+.3f}, "
                          f"{(ci or [None, None])[1]:+.3f}]"
                          if ci else ""))
            lines.append(
                f"| {r.get('variant')} | {r.get('n_seeds')} | "
                f"{_fmt4(r.get('hit_base_rate'))} | {feat_s} | "
                f"{_fmt4(r.get('rule_auprc'))} | "
                f"{_fmt4(r.get('probe_auprc'))} | "
                f"{_fmt4(r.get('gate_auprc'))} | "
                f"{_fmt4(r.get('probe_auroc'))} | "
                f"{_fmt4(r.get('gate_auroc'))} | {delta_s} | "
                f"{_fmt4(r.get('p_adj_auprc'))} | "
                f"{r.get('verdict') or '-'} |")
        lines += ["", f"  {probe.get('note', '')}"]
        lines.append(
            f"  注：命中基线率=hit 正类率，也是 PR-AUC 的随机猜基线；"
            f"主指标={next((r.get('primary_detection_metric') for r in probe['rows'] if r.get('primary_detection_metric')), 'n/a')}"
            f"（正类率 <5% 时 PR-AUC 为主，AUROC 因负类基数大而乐观）；"
            f"探针 λ 逐变体按 OOF log-loss 网格选择（见 analysis_*.json "
            f"的 linear_probe.per_l2）；PR-AUC 探针/OOF 与门控在同一样本上"
            f"计算，Δ 的 CI 与 p_adj 见 significance.json。")
    if probe.get("missing"):
        names = "、".join(f"{x.get('variant')}（{x.get('reason')}）"
                          for x in probe["missing"])
        lines.append(f"- ⚠ 线性探针简单基线缺失：{names}；"
                     "机制主张缺少“有标签监督的最简线性模型”对照，"
                     "不能用“门控 logit 判别力高”替代。")
    lines.append("")
    return lines


def _fmt_scalar_maybe(x) -> str:
    if x is None:
        return "-"
    try:
        return f"{float(x):.3f}"
    except (TypeError, ValueError):
        return str(x)


def _dedupe_advice(items, limit=6):
    """按 (轴, 建议) 去重改进方向并截断（不同变体的同一模式只留一条）。"""
    out, seen = [], set()
    for label, r in items:
        key = (r.get("axis"), r.get("suggestion"))
        if key in seen:
            continue
        seen.add(key)
        out.append((label, r))
        if len(out) >= limit:
            break
    return out


def _render_advice(md, advice, title, max_items=6):
    """渲染“坏案例驱动的改进方向”小节（证据 → 建议 → 下一步实验）。"""
    advice = _dedupe_advice(advice, limit=max_items)
    if not advice:
        return
    md += ["", title, ""]
    for label, r in advice:
        md.append(f"- **[{r.get('axis')}·{r.get('priority')}]**（{label}）"
                  f"{r.get('trigger')}")
        md.append(f"  - 建议：{r.get('suggestion')}")
        md.append(f"  - 下一步：{r.get('next_experiment')}")


# ---------------------------------------------------------------- 曲线诊断与权衡

def diagnose_training_curves(chain, data_dir: str, results_dir: str) -> dict:
    """对每个 (变体, 种子) run 做训练曲线诊断并落盘汇总。

    只看最终数值会掩盖训练动态：训练降/验证升=过拟合，两者都不降=欠拟合或
    数据管道问题，NaN=训练失效。诊断结果进入 summary，另存
    curve_diagnosis.json 供逐 run 复核。
    """
    log_vocab = None
    meta_path = os.path.join(data_dir, "meta.json")
    if os.path.exists(meta_path):
        try:
            import math as _math
            with open(meta_path, "r", encoding="utf-8") as f:
                log_vocab = _math.log(int(json.load(f)["vocab_size"]))
        except (OSError, ValueError, KeyError, TypeError):
            log_vocab = None
    diag = {}
    for v, seed, vdir, _suffix in chain:
        d = diagnose_run_dir(vdir, log_vocab=log_vocab)
        diag.setdefault(v, {})[str(seed)] = d
    if diag:
        save_json(diag, os.path.join(results_dir, "curve_diagnosis.json"))
        n_bad = sum(1 for vs in diag.values() for d in vs.values()
                    if d["status"] not in ("正常收敛", "数据不足"))
        print(f"[curves] 训练曲线诊断完成：{n_bad} 个 run 需要关注"
              f" -> {os.path.join(results_dir, 'curve_diagnosis.json')}")
    return diag


def _eff_cell(entry, key, ctx):
    """latency.json 变体条目里的结构与运行时字段（flops 取 total，mem 取量测值）。"""
    if not entry:
        return None
    eff = entry.get("efficiency") or {}
    if key == "flops":
        cell = (eff.get("flops_per_tok") or {}).get(str(ctx))
        return cell.get("total_flops_per_tok") if cell else None
    if key == "mem":
        return (eff.get("peak_mem_mb_prefill") or {}).get(str(ctx))
    if key == "params":
        return eff.get("n_params") or entry.get("n_params")
    if key == "params_mb":
        return eff.get("params_mb")
    return None


def _rel_pct(base, cand):
    """cand 相对 base 的百分比变化；任一缺失或 base=0 时返回 None。"""
    if base is None or cand is None:
        return None
    if base == 0:
        return None
    return (cand - base) / abs(base) * 100.0


def _lat_noise_pct(entry, metric, ctx, bs="1"):
    """latency.json 中某运行时指标的测量相对散布（spread_pct）；缺失返回 None。

    spread_pct=(p90-p10)/median，由 lega.latency 逐次计时给出。旧格式
    latency.json 没有该字段时返回 None（按无噪声信息处理）。
    """
    if not entry:
        return None
    key = ("prefill_ms_spread_pct" if metric == "prefill"
           else "decode_ms_spread_pct")
    cell = (entry.get(key) or {}).get(str(ctx))
    if metric == "prefill":
        cell = (cell or {}).get(str(bs))
    if cell is None:
        return None
    try:
        return float(cell)
    except (TypeError, ValueError):
        return None


def _noise_floor_pct(base_noise, cand_noise):
    """两端的测量散布按方和根合成（比值差的方差近似为相对方差之和）。

    只有两端都有散布信息时才给出噪声下限：一侧缺失时无法判断"差值是否
    小于测量噪声"，按未过滤的原口径处理，避免用单侧散布低估/高估噪声。
    """
    if base_noise is None or cand_noise is None:
        return None
    return math.sqrt(base_noise * base_noise + cand_noise * cand_noise)


# RULER 收益以 EM 绝对百分点计，与 PPL 的相对百分比单位不同：收支比规则
# 不能混算，RULER 单独设阈值（EM 提升通常比 PPL 百分比更难获得，倍数放宽
# 到 10×；下限 0.5 pp 避免收益趋近 0 时比值判定失效）。
RULER_COST_RATIO = 10.0
RULER_RATIO_FLOOR_PP = 0.5


def tradeoff_verdict(benefit_pct, p_adj, cost_pct, alpha=ALPHA,
                     n_seeds=None, ruler_benefit_pp=None, ruler_p_adj=None,
                     ruler_n_seeds=None) -> str:
    """涨点-增幅权衡判定：不看增幅的涨点没有意义；收益必须覆盖两个主指标。

    两个主指标：PPL（benefit_pct=相对基线的百分比改善，正=更好）与 RULER
    平均 EM（ruler_benefit_pp=相对基线的绝对百分点改善，正=更好）。本项目
    预注册主张横跨两者（H1 PPL、H4 长上下文检索），只看 PPL 会把“PPL 持平
    但 RULER 显著提升”的真涨点误判为无效，也会漏掉“PPL 涨、RULER 显著
    退步”的严重信号。

    规则（显式、可审计）：
      * 两个主指标都无数据或增幅缺失 → 数据不足；
      * 参与判定的指标种子不足（给定 n<MIN_SEEDS）→ 无法判定，与全表口径一致；
      * 某一主指标显著劣化：若另一主指标同时显著改善 → 权衡（必须显式
        说明取舍）；否则 → 不可行；
      * 任一主指标显著改善 → 有涨点：综合增幅（各指标相对基线的最大增幅）
        >50% → 需审视；按显著者中最不利的收益套用收支比规则（PPL 5×，
        RULER 单独 10× 且下限 0.5 pp，单位不同不混算）→ 收益-增幅比不利；
        否则 → 可接受，并注明依据哪个指标；
      * 有正向涨点但不显著 → 说服力有限（增幅>10% 判不可行）。
    微小涨点伴随巨大增幅的改进在这里会被明确拒绝。
    """
    if (benefit_pct is None and ruler_benefit_pp is None) or cost_pct is None:
        return "数据不足"
    lacks = []
    if benefit_pct is not None and n_seeds is not None and n_seeds < MIN_SEEDS:
        lacks.append(f"PPL n={n_seeds}")
    if (ruler_benefit_pp is not None and ruler_n_seeds is not None
            and ruler_n_seeds < MIN_SEEDS):
        lacks.append(f"RULER n={ruler_n_seeds}")
    if lacks:
        return f"种子不足（{'、'.join(lacks)}<{MIN_SEEDS}），无法判定"
    ppl_sig = p_adj is not None and p_adj < alpha
    rul_sig = ruler_p_adj is not None and ruler_p_adj < alpha
    ppl_up = benefit_pct is not None and benefit_pct > 0 and ppl_sig
    rul_up = (ruler_benefit_pp is not None and ruler_benefit_pp > 0
              and rul_sig)
    ppl_down = benefit_pct is not None and benefit_pct < 0 and ppl_sig
    rul_down = (ruler_benefit_pp is not None and ruler_benefit_pp < 0
                and rul_sig)
    if ppl_down and rul_up:
        return (f"权衡：PPL 显著退步 {benefit_pct:.1f}% 但 RULER 显著提升 "
                f"+{ruler_benefit_pp:.2f} pp（增幅 {cost_pct:.0f}%，"
                "检索目标改善，须说明 PPL 取舍）")
    if rul_down and ppl_up:
        return (f"权衡：PPL 显著改善 {benefit_pct:.1f}% 但 RULER 显著退步 "
                f"{ruler_benefit_pp:.2f} pp（增幅 {cost_pct:.0f}%，"
                "核心检索目标受损，须说明取舍）")
    if ppl_down:
        extra = (f"，且 RULER 显著退步 {ruler_benefit_pp:+.2f} pp"
                 if rul_down else "")
        return (f"不可行：PPL 显著差于基线（{benefit_pct:.1f}%）"
                f"{extra}")
    if rul_down:
        return f"不可行：RULER 显著差于基线（{ruler_benefit_pp:+.2f} pp）"
    if ppl_up or rul_up:
        if cost_pct > 50:
            return f"需审视：显著但增幅 {cost_pct:.0f}%"
        if ppl_up and cost_pct > 5 * max(benefit_pct, 0.5):
            return (f"收益-增幅比不利：PPL 收益 {benefit_pct:.1f}% vs "
                    f"增幅 {cost_pct:.0f}%")
        if rul_up and cost_pct > RULER_COST_RATIO * max(ruler_benefit_pp,
                                                        RULER_RATIO_FLOOR_PP):
            return (f"收益-增幅比不利：RULER 收益 +{ruler_benefit_pp:.2f} pp vs "
                    f"增幅 {cost_pct:.0f}%")
        basis = "PPL" if ppl_up else "RULER"
        if ppl_up and rul_up:
            basis = "PPL+RULER"
        return "可接受" if basis == "PPL" else f"可接受（依据 {basis} 显著涨点）"
    any_pos = ((benefit_pct is not None and benefit_pct > 0)
               or (ruler_benefit_pp is not None and ruler_benefit_pp > 0))
    if any_pos:
        if cost_pct > 10:
            return f"不可行：涨点不显著且增幅 {cost_pct:.0f}%"
        return "说服力有限：涨点未通过显著性检验"
    if cost_pct > 10:
        return f"不可行：无涨点且增幅 {cost_pct:.0f}%"
    return "说服力有限：无涨点且差异不显著"


def build_summary(results_dir: str, audit: dict = None,
                  data_report: dict = None, preloaded: dict = None) -> dict:
    """汇总 PPL / RULER / 参数量等指标为 Markdown + CSV + JSON。

    多种子口径：每个变体的每个指标先按种子聚合为 Mean ± Std；变体间比较使用
    同一批种子上的配对 t 检验（同种子共享初始化与数据顺序，配对消除公共方差）。
    消融矩阵显式列出全因子 8 个组合的覆盖情况，缺失组合会被标注。
    data_report（可选）来自数据目录的防泄漏自检报告，汇总为独立小节。
    preloaded（可选）为 {results 下结果文件路径: 已解析 JSON} 缓存：坏案例
    阶段已解析过的 eval/ruler 文件在此共用同一份结果。
    """

    def load(p):
        if preloaded and p in preloaded:
            return preloaded[p]
        if not os.path.exists(p):
            return None
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)

    # --- “本次 run”范围限定：results/<tag> 会被同一 tag 的多次运行复用
    # （改种子/变体集、--skip_* 等），聚合按 repro.json 记录的本次
    # variants/seeds 过滤，不同实验的观测不混入同一均值/配对检验；
    # 没有 repro.json（直接调用/手工构建目录）时保持全量发现。
    repro = load(os.path.join(results_dir, "repro.json")) or {}
    scope_variants = repro.get("variants") if isinstance(
        repro.get("variants"), list) else None
    scope_seeds = repro.get("seeds") if isinstance(
        repro.get("seeds"), list) else None

    def in_run_scope(fp, d):
        if not isinstance(d, dict) or d.get("variant") is None:
            return False
        if scope_variants is not None and d["variant"] not in scope_variants:
            return False
        if scope_seeds is not None and _seed_key(fp, d) not in scope_seeds:
            return False
        return True

    # --- 发现所有 run：（变体, 种子） -> {eval, ruler, analysis}
    runs = {}

    def slot(v, s):
        return runs.setdefault(v, {}).setdefault(
            s, {"eval": None, "ruler": None, "analysis": None})

    for fp in _result_files(results_dir, "eval", scope_seeds):
        d = load(fp)
        if in_run_scope(fp, d):
            slot(d["variant"], _seed_key(fp, d))["eval"] = d
    for fp in _result_files(results_dir, "ruler", scope_seeds):
        d = load(fp)
        if in_run_scope(fp, d):
            slot(d["variant"], _seed_key(fp, d))["ruler"] = d
    for fp in _result_files(results_dir, "analysis", scope_seeds):
        if "records" in os.path.basename(fp):
            continue
        d = load(fp)
        if in_run_scope(fp, d):
            slot(d["variant"], _seed_key(fp, d))["analysis"] = d

    variants = sorted(v for v in runs if v in VARIANTS)
    eval_splits = set()
    for v in variants:
        for s in _variant_seed_map(runs, v).values():
            if s.get("eval"):
                eval_splits.add(s["eval"].get("split", "val"))

    # 汇总运行时指标时必须使用所有变体共有的同一上下文档，不允许各取自己的最大 ctx。
    lat = load(os.path.join(results_dir, "latency.json")) or {}
    lat_entries = {v: (lat.get("variants") or {}).get(v) for v in variants}

    def _prefill_at(e, ctx):
        if not e or ctx is None:
            return None
        return (e.get("prefill_ms") or {}).get(str(ctx), {}).get("1")

    common_ctxs = None
    for v in variants:
        e = lat_entries.get(v)
        if not e:
            continue
        usable = {int(c) for c, by_bs in (e.get("prefill_ms") or {}).items()
                  if isinstance(by_bs, dict) and by_bs.get("1") is not None
                  and (e.get("decode_ms_per_tok") or {}).get(c) is not None}
        common_ctxs = usable if common_ctxs is None else (common_ctxs & usable)
    common_ctx = max(common_ctxs) if common_ctxs else None

    def _clock_extractor(key):
        return lambda d: (d or {}).get(key)

    metric_specs = [
        ("val_ppl", lambda m: _stat_series(m, "eval", _ppl_of)),
        ("gate_mean", lambda m: _stat_series(m, "eval", _clock_extractor("gate_mean"))),
        ("val_loss", lambda m: _stat_series(m, "eval", _clock_extractor("val_loss"))),
        ("ruler_avg", lambda m: _stat_series(m, "ruler", _ruler_avg_of)),
    ]
    ruler_tasks = ("niah_synth", "niah_corpus", "niah_multikey", "fwe", "vt")
    for task in ruler_tasks:
        metric_specs.append(
            (f"ruler_{task}",
             (lambda t: lambda m: _stat_series(
                 m, "ruler", lambda d: _ruler_task_means(d).get(t)))(task)))
    for key in ("auroc_logit", "auroc_qterm", "misleading_rate",
                "misleading_given_conc", "p_hit_given_conc",
                "auprc_logit", "auprc_qterm", "hit_base_rate"):
        metric_specs.append(
            (key, (lambda k: lambda m: _stat_series(
                m, "analysis", _clock_extractor(k)))(key)))
    # 门控决策点 F1（g>0.5）：显式只取 macro 进汇总表（不平衡下对稀有类
    # 最诚实的口径）；micro/weighted 与逐类明细在 analysis_*.json 中
    # 的 gate_f1 字段并列完整保留，避免“只写 F1 不注明平均口径”。
    metric_specs.append(
        ("gate_f1_macro", lambda m: _stat_series(
            m, "analysis",
            lambda d: (d.get("gate_f1") or {}).get("macro"))))
    # 恒写（全判 hit）规则的 macro F1：门控决策的“稻草人”对照。
    # 不平衡下门控 F1 若不明显高于该固定规则，就没有决策价值。
    metric_specs.append(
        ("gate_f1_macro_always_write", lambda m: _stat_series(
            m, "analysis",
            lambda d: (((d.get("gate_f1_trivial") or {})
                        .get("always_write") or {}).get("macro")))))
    # 线性探针（有标签监督的简单基线）OOF 指标：进入 summary.json/csv，
    # 供“门控 vs 最简线性模型”的补充性对照审计（渲染见基线合理性小节）。
    metric_specs.append(
        ("probe_auprc", lambda m: _stat_series(
            m, "analysis", lambda d: _probe_oof(d, "auprc"))))
    metric_specs.append(
        ("probe_auroc", lambda m: _stat_series(
            m, "analysis", lambda d: _probe_oof(d, "auroc"))))

    rows = []
    for v in variants:
        m = _variant_seed_map(runs, v)
        row = {"variant": v, "variant_cn": VARIANT_CN.get(v, v),
               "n_seeds": len(m)}
        for key, fn in metric_specs:
            st = fn(m)
            if st:
                row[key] = st["mean"]
                row[f"{key}_std"] = st["std"]
                row[f"{key}_n"] = st["n"]
            else:
                row[key] = row[f"{key}_std"] = row[f"{key}_n"] = None
        ev_example = next((s["eval"] for s in m.values() if s.get("eval")), None)
        row["params"] = (ev_example or {}).get("n_params")
        row["gate_extra"] = (ev_example or {}).get("n_gate_extra", 0)
        row["val_ppl_clamped"] = any(
            (s.get("eval") or {}).get("val_ppl_clamped") for s in m.values())
        # 运行时指标（取全部变体的最大公共 ctx、batch=1）；不受种子影响，测一次
        e = lat_entries.get(v)
        if e and common_ctx is not None:
            ckey = str(common_ctx)
            row["prefill_ms"] = (e.get("prefill_ms") or {}).get(ckey, {}).get("1")
            row["decode_ms_per_tok"] = (e.get("decode_ms_per_tok") or {}).get(ckey)
            row["decode_prefill_ctx"] = (e.get("decode_prefill_ctx") or {}).get(ckey)
            row["latency_ctx"] = common_ctx
        rows.append(row)

    # 运行时基线：优先取 standard_sdpa，缺失时退回显式 standard。
    lat_all = lat.get("variants") or {}
    base_ms, base_name = None, None
    for cand in ("standard_sdpa", "standard"):
        if _prefill_at(lat_all.get(cand), common_ctx) is not None:
            base_ms, base_name = _prefill_at(lat_all.get(cand), common_ctx), cand
            break
    for r in rows:
        if (base_ms and base_ms > 0 and r.get("prefill_ms") is not None
                and r["prefill_ms"] > 0
                and r["variant"] not in ("standard", "standard_sdpa")):
            r["prefill_rel_pct"] = round(
                (r["prefill_ms"] - base_ms) / base_ms * 100, 2)

    # --- 配对显著性检验（同种子）：PPL 与 RULER 平均
    pairs = []
    for cand in ALL_VARIANTS:
        if cand != "standard":
            pairs.append((cand, "standard"))
    leafless = ("lega_no_len", "lega_no_conc", "lega_no_query",
                "lega_conc_only", "lega_len_only")
    pairs += [(c, "lega") for c in leafless]
    # 预注册假设涉及的变体间对照必须可估计：登记的 (候选, 参照) 若不在
    # 常规矩阵里（如 lega vs qgate 的“vs 已有工作”主张），在此补入，
    # 保证每条 confirmatory 主张都有对应检验，而不是“登记了却没检验”。
    registered = load_registered_hypotheses()
    for h in (registered.get("hypotheses") or []):
        c, r = h.get("candidate"), h.get("reference")
        if h.get("metric") in (AUROC_DIFF_METRIC, AP_DIFF_METRIC):
            continue  # 单样本机制族，由 _auroc_diff_tests/_ap_diff_tests 覆盖
        if c in VARIANTS and r in VARIANTS and (c, r) not in pairs:
            pairs.append((c, r))
    pairs = list(dict.fromkeys(pairs))
    tests = []
    tests += _paired_comparisons(
        runs, pairs, "PPL", lambda slot: _ppl_of(slot.get("eval")),
        lower_better=True)
    tests += _paired_comparisons(
        runs, pairs, "RULER平均", lambda slot: _ruler_avg_of(slot.get("ruler")),
        lower_better=False)
    # 机制主张的正式检验族：AUROC(logit) − AUROC(qterm) 同种子配对差，
    # 以及不平衡口径下的 PR-AUC 配对差（两族分别 Holm 校正）
    tests += _auroc_diff_tests(runs)
    tests += _ap_diff_tests(runs)
    tests = _apply_holm(tests)
    comp = component_evidence(tests)
    # 预注册假设判定：confirmatory（登记主张）与 exploratory（其余对比）。
    # 完整性锁链把“事后静默改写登记表”变成可检测事件：校验未通过时
    # 全部确认性判定被显式降级（apply_integrity_gate），不会假装边界仍成立。
    integrity = verify_registration()
    # 范围门：登记表完整还不够——本次 run 必须真的落在登记的实验范围内
    # （config 规模/数据/种子计划），否则同样不能给 confirmatory 判定。
    scope_check = registration_scope_check(registered, repro)
    hyp_eval = apply_scope_gate(
        apply_integrity_gate(
            evaluate_registered_hypotheses(registered, tests), integrity),
        scope_check)

    # --- Markdown
    cols = [("variant_cn", "变体"), ("n_seeds", "种子数"), ("params", "参数量"),
            ("gate_extra", "门控额外参数"), ("val_ppl", "PPL↓"),
            ("ruler_avg", "RULER平均↑"),
            ("ruler_niah_synth", "niah_synth↑"),
            ("ruler_niah_corpus", "niah_corpus↑"),
            ("ruler_niah_multikey", "multikey↑"), ("ruler_fwe", "fwe↑"),
            ("ruler_vt", "vt↑"), ("prefill_ms", "Prefill ms↓"),
            ("prefill_rel_pct", "相对基线%"),
            ("decode_ms_per_tok", "Decode ms/tok↓"),
            ("gate_mean", "门控均值"), ("misleading_rate", "集中但看错率"),
            ("misleading_given_conc", "集中后看错率"),
            ("hit_base_rate", "命中基线率"),
            ("auroc_logit", "AUROC(logit)"), ("auroc_qterm", "AUROC(仅查询)"),
            ("auprc_logit", "PR-AUC(logit)↑"), ("auprc_qterm", "PR-AUC(仅查询)↑"),
            ("gate_f1_macro", "门控F1(macro)↑"),
            ("gate_f1_macro_always_write", "恒写基线F1(macro)")]

    def fmt_stat_cell(r, key):
        st = {"mean": r.get(key), "std": r.get(f"{key}_std"),
              "n": r.get(f"{key}_n")}
        cell = _fmt_stat(st)
        # PPL 被钳制在 exp(20) 的 run（发散）必须显式标注：不能与正常 PPL
        # 在同一列表头中静默比较（行级 val_ppl_clamped 由评测写入）。
        if key == "val_ppl" and r.get("val_ppl_clamped"):
            cell = f"{cell} ⚠含截断值"
        return cell

    def _fmt_scalar(x):
        if x is None:
            return "-"
        if isinstance(x, float):
            return f"{x:.3f}" if abs(x) < 10 else f"{x:,.0f}"
        return str(x)

    md = ["# LEGA 实验汇总", "",
          f"> 自动生成于 {time.strftime('%Y-%m-%d %H:%M:%S')}；"
          f"PPL/RULER 为跨随机种子的 Mean ± Std（变体间同种子配对，"
          f"显著性经 Holm–Bonferroni 校正），运行时指标在相同配置下测得。", ""]
    if rows:
        stat_keys = {"val_ppl", "ruler_avg", "gate_mean", "misleading_rate",
                     "misleading_given_conc", "auroc_logit", "auroc_qterm",
                     "auprc_logit", "auprc_qterm", "hit_base_rate",
                     "gate_f1_macro", "gate_f1_macro_always_write"} \
            | {f"ruler_{t}" for t in ruler_tasks}
        header = "| " + " | ".join(c for _, c in cols) + " |"
        sep = "|" + "---|" * len(cols)
        md += [header, sep]
        for r in rows:
            cells = []
            for k, _ in cols:
                if k in stat_keys:
                    cells.append(fmt_stat_cell(r, k))
                else:
                    cells.append(_fmt_scalar(r.get(k)))
            md.append("| " + " | ".join(cells) + " |")
        split_note = ("PPL 在 held-out test 集上评测（checkpoint 选择在 val 上完成）"
                      if eval_splits == {"test"} else
                      "⚠ PPL 在 val 集上评测，该集同时用于 best-checkpoint 选择，"
                      "数值偏乐观；请用 prepare_data.py 生成 held-out test 后重跑评测")
        min_n = min((r.get("n_seeds") or 1) for r in rows)
        seed_note = (f"关键指标至少 {min_n} 个随机种子；变体间比较为同种子配对。"
                     if min_n >= 3 else
                     f"⚠ 当前最少只有 {min_n} 个随机种子（<3），"
                     "无法可靠评估方差，结论只能是探索性的；"
                     "正式实验请用 ≥3 个种子（configs 默认已达要求）。")
        cols_notes = [
            "注: 统计单元格为 Mean ± Std (n=种子数)；n=1 时 Std 不可估计。",
            "注: niah/fwe/vt 为各上下文档位平均后的跨种子均值，"
            "每格样本数见 ruler_*.json（各变体同 seed 使用同一批样本）；",
            "“集中但看错率”= P(max_p>τ 且注意力落点不在答案数值跨度 value_span 内)，"
            "见机制分析。",
            "注: PPL 单元格标注 ⚠含截断值时，该变体至少有 1 个种子验证损失 >20，"
            "PPL 已按 exp(20) 截断；发散的 run 不能与正常 PPL 直接比较。",
            "注（类别不平衡与指标口径）：“命中基线率”= 机制分析中 hit 正类率，"
            "即 PR-AUC 的随机猜基线；正类率 <5% 时 AUROC 会因负样本基数巨大、"
            "FPR 增长缓慢而过于乐观，结论应以 PR-AUC 为主（极度不平衡 <0.1% "
            "时更是如此），两者并报供对照。F1 一律为 **macro**（各类等权，对"
            "稀有类最诚实）；micro/weighted 与逐类明细见 analysis_*.json 的 "
            "`gate_f1`（门控 g>0.5 决策点）。门控 F1 必须对照"
            "“恒写基线F1(macro)”（恒判写入的固定规则）：不高于这个无信息"
            "基线时，门控的决策价值不成立（`gate_f1_trivial` 另含恒不写"
            "规则）。RULER 的 EM 是生成式准确率、无决策阈值，不受此口径"
            "影响；fwe 另报召回率。",
            seed_note, split_note, ""]
        if common_ctx is not None:
            cols_notes.insert(1, f"“相对基线%”以 {base_name or 'standard'} 的 prefill 为基线"
                                 f"（公共 ctx={common_ctx}、batch=1，见 latency.json）；")
        md += [""] + cols_notes
    else:
        md.append("（暂无结果，先运行训练评测）")

    # --- 基线合理性（简单统计基线 + 基线调优证据 + 逐变体 vs 统计基线
    #     + 机制判别的线性探针简单监督基线）
    ngram_cmp = ngram_baseline_comparison(runs, results_dir)
    probe_cmp = linear_probe_baseline_comparison(runs)
    md += _render_baseline_sections(results_dir,
                                    smoke=bool((repro or {}).get("smoke")),
                                    ngram_cmp=ngram_cmp,
                                    probe_cmp=probe_cmp)

    # --- 受控变量自检
    md += ["", "## 对比实验受控变量自检", ""]
    if audit:
        if audit.get("ok"):
            if audit.get("checked"):
                md.append(f"- ✅ 通过：{audit['checked']} 个 run 的优化器、学习率、"
                          f"训练步数、数据顺序等受控设置逐项一致；逐 run 仅 "
                          f"`{'`, `'.join(audit['exempt_keys'])}` 不同。")
            else:
                md.append("- ⚠ 未检查受控变量：没有 run 可用于核对"
                          "（全部缺少 config_used.json），不能据此声称一致性。")
            if audit.get("meta_checked"):
                digest = audit.get("expected_data_digest")
                stopped = audit.get("early_stop_runs") or []
                md.append(f"- ✅ 训练溯源一致：{audit['meta_checked']} 个 run 的"
                          f"数据指纹、代码提交、配置 SHA-256、计划训练步数"
                          f"逐项一致"
                          + ("；实际步数差异为早停所致（见下条）"
                             if stopped else "")
                          + (f"（数据指纹 `{digest}`，并与本次流水线数据比对）"
                             if digest else "。"))
            if audit.get("early_stop_runs"):
                md.append(
                    f"- ⏹ 早停启用：{len(audit['early_stop_runs'])} 个 run 未跑满"
                    f"计划步数（{', '.join(audit['early_stop_runs'][:8])}"
                    + (" …" if len(audit["early_stop_runs"]) > 8 else "")
                    + "）：实际步数可因验证曲线不同而不同（早停的定义），"
                    "受控变量按计划步数核对；逐 run 实际步数与 tokens_seen "
                    "见各 eval.json。")
        else:
            md.append(f"- ❌ 未通过：{audit.get('mismatches') or audit}")
        if audit.get("missing_configs"):
            md.append(f"- ⚠ 缺少配置文件的 run：{', '.join(audit['missing_configs'][:8])}"
                      + (" …" if len(audit["missing_configs"]) > 8 else ""))
        if audit.get("missing_evals"):
            md.append(f"- ⚠ 缺少 eval.json、无法核对数据/代码/训练步数的 run："
                      f"{', '.join(audit['missing_evals'][:8])}"
                      + (" …" if len(audit["missing_evals"]) > 8 else ""))
    else:
        md.append("- ⚠ 未执行（直接调用 build_summary 或缺少 out 目录）。")

    # --- 数据协议与防泄漏自检（由 lega.data.prepare_bins 生成 data_report.json）
    md += ["", "## 数据协议与防泄漏自检", ""]
    if data_report:
        proto = data_report.get("protocol") or {}
        md.append(
            f"- 协议：先划分后预处理={proto.get('split_first_then_preprocess')}、"
            f"拟合统计只来自 {proto.get('fit_on')}、"
            f"划分时打乱={proto.get('shuffle_at_split_time')}；"
            "训练阶段 batch 级随机采样（shuffle）不构成泄露。")
        sp = data_report.get("split_protocol") or {}
        if sp:
            md.append(f"- 划分方式：{sp.get('mode', '-')}；"
                      f"顺序 {sp.get('order', '-')}；"
                      f"文档边界回退={sp.get('fallback_char_cut', '-')}"
                      + ("（⚠ 字符切分可能让同一文档跨 split）"
                         if sp.get("fallback_char_cut") else ""))
            group = sp.get("group_unit")
            cross = sp.get("documents_cross_cut")
            if group or cross is not None:
                md.append(f"- 群组完整性：群组单位={group or '-'}；"
                          f"标题数={sp.get('n_headings', '-')}；"
                          f"文档跨切点={cross if cross is not None else '-'}"
                          + ("（⚠ 群组跨 split，见告警 "
                             "split_documents_cross_cut）" if cross else ""))
        tok = data_report.get("tokenizer") or {}
        unk = tok.get("unk_rate") or {}
        unk_s = "、".join(f"{k}={_fmt_rate(v)}" for k, v in unk.items())
        md.append(f"- 分词器：{tok.get('kind')}（vocab={tok.get('vocab_size')}，"
                  f"拟合集={tok.get('fit_on')}）；OOV 字符率：{unk_s or 'N/A'}"
                  "（未知字符映射 <unk>，val/test 不参与词表拟合）")
        splits = data_report.get("splits") or {}
        md.append("- token 流硬校验：" + "；".join(
            f"{k} {int(v.get('tokens', 0)):,} tokens"
            f"（id∈[{v.get('min_id')},{v.get('max_id')}]，"
            f"NaN={v.get('nan')}，Inf={v.get('inf')}）"
            for k, v in splits.items()))
        dup = data_report.get("duplicates") or {}
        cont = data_report.get("contamination") or {}
        md.append(
            f"- 去污染：与 val/test 精确重复的训练长行移除 "
            f"{dup.get('train_lines_dropped_as_duplicates_of_eval', '-')} 行；"
            f"采样 {cont.get('train_val', {}).get('n', '-')}-gram 重叠率 "
            f"val={_fmt_rate((cont.get('train_val') or {}).get('overlap_rate'))}、"
            f"test={_fmt_rate((cont.get('train_test') or {}).get('overlap_rate'))}"
            "（采样筛查下界，>0 需人工核查）")
        dist = data_report.get("distribution_shift") or {}
        dv = dist.get("val_vs_train") or {}
        dt = dist.get("test_vs_train") or {}
        md.append(
            f"- 分布漂移（一元 token，采样估计）：val−train TV={_fmt4(dv.get('tv'))}、"
            f"JS={_fmt4(dv.get('js_bits'))} bit；test−train TV={_fmt4(dt.get('tv'))}、"
            f"JS={_fmt4(dt.get('js_bits'))} bit；test OOV token 率="
            f"{_fmt_rate(dt.get('eval_oov_token_rate'))}")
        san = data_report.get("sanitization") or {}
        noise = sum(int((san.get(s) or {}).get("replacement_char_replaced_with_space", 0))
                    for s in san)
        ctrl = sum(int((san.get(s) or {}).get("control_chars_deleted", 0))
                   for s in san)
        md.append(f"- 文本清洗（标签噪声）：U+FFFD 替换 {noise:,} 处、"
                  f"非法控制符删除 {ctrl:,} 处；清洗为逐字符确定性规则，"
                  "不依赖跨样本统计量。")
        ts = data_report.get("text_stats") or {}
        if ts:
            md.append("- 行长度异常检查：" + "；".join(
                f"{k} 行数={int(v.get('lines') or 0):,}、"
                f"最长行={int(v.get('line_chars_max') or 0):,} 字符、"
                f"超长行(>10万字符)={int(v.get('lines_gt_huge_threshold') or 0)}"
                for k, v in ts.items()))
        alerts = data_report.get("alerts") or []
        if alerts:
            n_warn = sum(1 for a in alerts if a.get("level") == "warning")
            shown = "；".join(f"[{a.get('code')}] {a.get('message')}"
                              for a in alerts[:5])
            md.append(f"- ⚠ 自检告警（{n_warn} warning / 共 {len(alerts)} 条）："
                      + shown + (" …" if len(alerts) > 5 else "")
                      + "（阈值见 data_report.json 的 alert_thresholds）")
        else:
            md.append("- 自检告警：无（全部指标在 alert_thresholds 阈值内；"
                      "阈值是复核触发线而非显著性检验）")
        ta = data_report.get("test_access") or {}
        if ta:
            warn = ("" if not ta.get("checkpoints_evaluated_multiple_times")
                    else "；⚠ 存在被重复评测的 checkpoint："
                         + "、".join(ta["checkpoints_evaluated_multiple_times"][:4]))
            md.append(f"- 测试集尘封账本：累计 test 评测 {ta.get('evaluations', 0)} 次、"
                      f"不同 checkpoint {ta.get('distinct_checkpoints', 0)} 个"
                      f"{warn}。测试集只用于最终评估；据此调参即视为泄露。")
        else:
            md.append("- 测试集尘封账本：暂无 test 评测记录。")
    else:
        md.append("- ⚠ 未找到 `data_report.json`（旧数据或未走统一数据入口）；"
                  "请用 `scripts/prepare_data.py` / `scripts/make_dummy_data.py` "
                  "重建数据以生成防泄漏自检。")

    # --- 可复现性与 sanity check（代码版本 / 数据指纹 / 过拟合小批量）
    md += ["", "## 可复现性与 sanity check", ""]
    if repro:
        git = repro.get("git") or {}
        data = repro.get("data") or {}
        rt = repro.get("runtime") or {}
        commit = git.get("commit") or "unknown"
        dirty = git.get("dirty")
        md.append(f"- 代码版本：`{str(commit)[:12]}`"
                  + (f"（分支 {git.get('branch')}）" if git.get("branch") else "")
                  + ("；⚠ 工作区 dirty，结果对应未提交状态" if dirty else
                     ("" if dirty is None else "；工作区 clean")))
        files = data.get("files") or []
        md.append(f"- 数据指纹：`{data.get('digest')}`"
                  f"（tokenizer={data.get('tokenizer')}，"
                  + "、".join(f"{e['name']} sha256:{e['sha256'][:12]}…" for e in files)
                  + "）")
        md.append(f"- 配置 SHA-256：`{str(repro.get('config_sha256'))[:16]}…`；"
                  f"种子：{repro.get('seeds')}")
        md.append(f"- 运行环境：python={rt.get('python')} torch={rt.get('torch')} "
                  f"device={rt.get('device')} gpu={rt.get('gpu')} cuda={rt.get('cuda')}")
        md.append("- 完整记录：`repro.json`（全局）、各 run 目录 `run_manifest.json`"
                  "（代码/数据/环境/命令行）、`config_used.json`（超参数）、"
                  "`train_log.jsonl`（训练曲线）。")
    else:
        md.append("- ⚠ 未找到 `repro.json`（直接调用 build_summary？）；"
                  "无法核对代码版本与数据指纹。")
    sanity_files = sorted(glob.glob(os.path.join(results_dir, "sanity_*.json")))
    # 与 runs 同一“本次 run”口径（sanity 无种子维度，只按变体过滤）。
    if scope_variants is not None:
        sanity_files = [fp for fp in sanity_files
                        if (load(fp) or {}).get("variant") in scope_variants]
    if sanity_files:
        md.append("")
        md.append("| 变体 | sanity | 初始 loss | 最终 loss | token 准确率 |")
        md.append("|---|---|---|---|---|")
        for fp in sanity_files:
            d = load(fp)
            if not d:
                continue
            ini = (d.get("initial") or {}).get("loss")
            fin = (d.get("final") or {}).get("loss")
            acc = (d.get("final") or {}).get("token_acc")
            md.append(f"| {VARIANT_CN.get(d.get('variant'), d.get('variant'))} | "
                      f"{'✅ 通过' if d.get('passed') else '❌ 未通过'} | "
                      f"{'-' if ini is None else f'{ini:.4f}'} | "
                      f"{'-' if fin is None else f'{fin:.4f}'} | "
                      f"{'-' if acc is None else f'{acc:.4f}'} |")
        md.append("")
        md.append("注: sanity check 只验证固定小批量的前向/反向/损失/标签链路能被"
                  "优化到接近零损失（必要非充分）；数据增强/特征对齐/标签映射等"
                  "问题仍可能漏网，需配合中间结果检查。")
    else:
        md.append("- ⚠ 未找到 sanity 结果（本次可能跳过了 sanity check）。")
    if os.path.exists(os.path.join(results_dir, "fig_train_curves.png")):
        md.append("- 训练曲线图：`fig_train_curves.png`（各变体训练 loss 与 val PPL，"
                  "多种子取均值）。")

    # --- 训练曲线诊断：结果分析不能只看最终数值，曲线本身先要过一遍体检
    md += ["", "## 训练曲线诊断（过拟合/欠拟合/发散/NaN）", ""]
    curve = load(os.path.join(results_dir, "curve_diagnosis.json"))
    if curve:
        md.append("诊断口径：训练降而验证升=过拟合；训练停滞（不再实质下降）而"
                  "验证回升同样判过拟合（平台期泛化先恶化，不按欠拟合处理）；"
                  "训练仍在实质下降而验证无实质变化=过拟合趋势（验证不再随训练"
                  "改善的早期信号，按正则化/早停处理，不据此续训）；"
                  "两者都不降=欠拟合/学习率不当/数据管道问题；出现 NaN=训练失效；"
                  "瞬态损失尖峰（分段均值会掩盖）与损坏日志行单独报告。"
                  "逐 run 明细见 `curve_diagnosis.json`。")
        md.append("")
        md.append("| 变体 | 状态（最差种子） | 训练损失 首→末 | 验证损失 首→最优→末 "
                  "| 早停空间 | 关键建议 |")
        md.append("|---|---|---|---|---|---|")
        any_nan = False
        any_grad_spike = False
        any_loss_spike = False
        any_malformed = False
        for v in variants:
            by_seed = curve.get(v) or {}
            if not by_seed:
                continue
            diags = list(by_seed.values())
            status = worst_status([d["status"] for d in diags])
            worst = next(d for d in diags if d["status"] == status)
            any_nan = any_nan or any(d.get("nonfinite_values") for d in diags)
            spike = any(d.get("has_grad_spike") for d in diags)
            any_grad_spike = any_grad_spike or spike
            lspike = any(d.get("has_loss_spike") for d in diags)
            any_loss_spike = any_loss_spike or lspike
            any_malformed = any_malformed or any(d.get("malformed_records")
                                                 for d in diags)
            es = [d.get("early_stop_gain_rel") for d in diags
                  if d.get("early_stop_gain_rel") is not None]

            def _f4(x):
                return "-" if x is None else f"{x:.4f}"

            train_s = (f"{_f4(worst['train_loss_first'])}→"
                       f"{_f4(worst['train_loss_last'])}")
            val_s = (f"{_f4(worst['val_loss_first'])}→"
                     f"{_f4(worst['val_loss_best'])}@s{worst['val_best_step']}"
                     f"→{_f4(worst['val_loss_last'])}")
            es_s = ("-" if not es else
                    f"{sum(es) / len(es):+.1%}"
                    + ("（最终劣于最优，已用 best checkpoint）"
                       if sum(es) / len(es) > 0.02 else ""))
            rec = worst["recommendations"][:1]
            es_flagged = any(d.get("early_stopped") for d in diags)
            # 需关注种子的占比：只看"最差种子"分不清系统性异常与单个种子偶发，
            # 1/5 与 5/5 需要完全不同的处置
            n_attention = sum(1 for d in diags
                              if d.get("status") not in ("正常收敛", "数据不足"))
            attention_s = (f"（{n_attention}/{len(diags)} 种子需关注）"
                           if n_attention else "")
            status_s = status + attention_s \
                + ("　（梯度尖峰）" if spike else "") \
                + ("　（损失尖峰）" if lspike else "") \
                + ("　（早停）" if es_flagged else "")
            md.append(f"| {VARIANT_CN.get(v, v)} | {status_s} | {train_s} "
                      f"| {val_s} | {es_s} | {rec[0] if rec else '-'} |")
        md.append("")
        if any_nan:
            md.append("⚠ 存在出现非有限损失的 run：训练已失效，"
                      "对应 run 目录的 `nan_report.json` 给出了"
                      "数据→代码→混合精度→梯度/学习率的系统排查路径"
                      "（含检测点前的整个 log 区间候选批次与区间内首个非有限"
                      "损失步 `first_bad_step`，排查的是失败现场而非最后一步"
                      "批次），处理前该变体的一切指标都不可信。")
        else:
            md.append("注：全部 run 无 NaN/Inf；有异常状态的变体请按建议处理"
                      "（过拟合/过拟合趋势→正则化/增数据/早停，欠拟合→查学习率"
                      "与数据管道）后重跑，再进入对比结论。")
        if any_grad_spike:
            md.append("")
            md.append("⚠ 检测到 pre-clip 梯度范数非有限或爆炸性尖峰"
                      "（最大/中位数 ≥10×）的 run：这是 NaN/发散的早期信号，"
                      "即使损失曲线看似正常也应先降低学习率/收紧 grad_clip "
                      "后重跑，细节见 `curve_diagnosis.json` 的 grad_norm 字段。")
        if any_loss_spike:
            md.append("")
            md.append(f"⚠ 检测到训练损失瞬态尖峰（某点 > 邻近中位数 "
                      f"{LOSS_SPIKE_FACTOR:.0f}×，分段均值会掩盖）：先核对该"
                      "步批次数据与梯度范数，再决定收紧 grad_clip/降低学习率；"
                      "细节见 `curve_diagnosis.json` 的 loss_spikes 字段。")
        if any_malformed:
            md.append("")
            md.append("⚠ 部分 train_log.jsonl 记录无法解析（已跳过并计数）："
                      "日志可能被中断或损坏，曲线结论只基于其余记录点，"
                      "复核前不要把它当作完整训练历史。")
        any_early = any(d.get("early_stopped")
                        for vs in curve.values() for d in vs.values())
        if any_early:
            md.append("")
            md.append("⏹ 上表带「早停」标记的变体触发了早停（验证连续 patience "
                      "次无实质改善即终止，曲线按设计截断，区别于日志缺失/中断）："
                      "下游评测使用的仍是 val 选出的 `model_best.pt`；若其最优"
                      "仍出现在训练早期，应先处理过拟合/学习率问题，而不是"
                      "统一延长所有变体的训练步数。")
    else:
        md.append("- ⚠ 未找到 `curve_diagnosis.json`：只看最终数值会掩盖训练动态，"
                  "请先运行训练（或 `python -m lega.curves --run_dir <run 目录>`）。")

    # --- 坏案例分析：预测错误的样本逐一归因（改进思路的最重要来源）
    md += ["", "## 坏案例分析（预测错误的样本与归因）", ""]
    bad_rows, bad_example_pool = [], []
    # 跨变体对照要求各变体使用同一批评测样本：取每个变体最小种子的错误
    # 清单（RULER 样本由配置里的评测种子构造，与训练种子无关，可逐条对齐）
    per_variant_badcases = cross_variant_badcase_map(runs, variants)
    for v in variants:
        m = _variant_seed_map(runs, v)
        cases = []
        for s in m.values():
            r = s.get("ruler")
            if r:
                cases += r.get("badcases") or []
        if not cases:
            continue
        ana = analyze_badcases(cases, variant=v)
        cats = sorted(ana["category_counts"].items(), key=lambda kv: -kv[1])
        cat_s = "；".join(f"{BADCASE_CN.get(k, k)}×{n}" for k, n in cats[:3])
        bad_rows.append((v, ana, cat_s))
        bad_example_pool += [(v, c) for c in ana["cases"]]
    if bad_rows:
        md.append("错误样本不是污点而是信息：逐条区分**标注错误**（gold 不在 "
                  "prompt 中→构造错误）、**难度边界样本**（needle 极深）与"
                  "**模型盲区**（误抄干扰值/幻觉数字/无数值回答）。"
                  "逐案例详情见 `badcases_*.md`。")
        md.append("")
        n_seeds_max = max((len(_variant_seed_map(runs, v)) for v in variants),
                          default=0)
        if n_seeds_max > 1:
            md.append(f"注：RULER 样本集跨种子相同（由评测构造种子决定），下表"
                      f"错误样本数为最多 **{n_seeds_max} 个种子的累计频次**"
                      "（同一样本在 k 个种子下错就计 k 次），用于衡量错误的"
                      "稳定性；唯一样本级的对照见下节「跨变体错误对照」"
                      "（取最小种子）。")
            md.append("")
        md.append("| 变体 | 错误样本数（跨种子累计） | 主要归因 | niah 深位(>0.75)失败占比 "
                  "| 标注/构造错误 |")
        md.append("|---|---|---|---|---|")
        for v, ana, cat_s in bad_rows:
            dfr = ana.get("deep_position_fail_ratio")
            md.append(f"| {VARIANT_CN.get(v, v)} | {ana['n_badcases']} | {cat_s} "
                      f"| {'-' if dfr is None else f'{dfr:.0%}'} "
                      f"| {ana['label_errors'] or 0} |")
        md.append("")
        # 摘 3 条代表性个案放进汇总页：错误必须"看得见"，不能只留在文件里
        shown, seen_cat = 0, set()
        for v, c in bad_example_pool:
            if shown >= 3 or c["category"] in seen_cat:
                continue
            seen_cat.add(c["category"])
            shown += 1
            q = (c.get("question") or "").strip().replace("\n", " ")[:120]
            pred = (c.get("pred") or "").strip().replace("\n", " ")[:80]
            md += [f"- **{VARIANT_CN.get(v, v)} · {c['task']}@ctx{c.get('ctx')}** "
                   f"—— {BADCASE_CN.get(c['category'], c['category'])}",
                   f"  - 问题：{q}…",
                   f"  - 输出：`{pred}` vs gold `{c.get('answer')}`；"
                   f"{c.get('explanation', '')}"]
        md.append("")
        n_label = sum(a["label_errors"] for _, a, _ in bad_rows)
        if n_label:
            md.append(f"⚠ 共 {n_label} 条 gold 不在 prompt 中的样本：这是"
                      "构造/标注错误，不是模型失败，须先处理评测再下结论。")
        # 错误归因的终点是改进方向：把主导错误模式转成下一步可执行的实验，
        # 而不是停在分类表（坏案例是提出新思路的最重要来源）。
        _render_advice(md, [(v, r) for v, a, _ in bad_rows
                            for r in (a.get("recommendations") or [])],
                       "### 坏案例驱动的改进方向（RULER）")
    else:
        md.append("- 暂无错误样本记录（RULER 未运行或全部命中）。"
                  "全部命中时也建议人工抽查若干正确样本确认没有'碰巧命中'。")

    # --- 跨变体错误对照：共识失败（数据/难度）vs 方法特异性盲区
    bad_cross = None
    if len(per_variant_badcases) >= 2:
        bad_cross = cross_variant_analysis(per_variant_badcases,
                                           reference="standard")
        md += ["", "### 跨变体错误对照（共识失败 vs 方法特异性）", ""]
        n_shared = bad_cross["n_shared_all_failures"]
        md.append(
            f"各变体取最小种子的错误清单（样本集相同，可逐条对齐；RULER 已"
            f"评测但零错误的变体按“全部通过”计入，不会被误算作共识失败）：共 "
            f"{bad_cross['n_unique_failed_items']} 个不同的失败样本，"
            f"{n_shared} 个为全体（已评测）变体共识失败——共识失败（尤其当 "
            "standard 也错）更可能是数据/构造/内在难度问题，不应算作某一"
            "方法的问题；相对 standard 的新增错误才是该方法的特异性盲区，"
            "优先逐条分析。")
        md += ["", "| 变体 | 错误样本数 | 全体共识失败 | 相对standard新增 "
                   "| 相对standard改对 |", "|---|---|---|---|---|"]
        for v in bad_cross["variants"]:
            pv = (bad_cross["per_variant"] or {}).get(v) or {}

            def _cnt(x):
                return "-" if x is None else str(x)

            md.append(f"| {VARIANT_CN.get(v, v)} | {_cnt(pv.get('n_failed'))} "
                      f"| {_cnt(pv.get('n_shared_all'))} "
                      f"| {_cnt(pv.get('n_new_vs_reference'))} "
                      f"| {_cnt(pv.get('n_fixed_vs_reference'))} |")
        # 展示至多 2 条特异性新增错误（方法盲区的最直接线索）
        shown = 0
        for v in bad_cross["variants"]:
            if v == "standard" or shown >= 2:
                continue
            for c in (bad_cross["per_variant"][v].get("new_vs_reference") or []):
                if shown >= 2:
                    break
                shown += 1
                ctx_s = f"@ctx{c['ctx']}" if c.get("ctx") is not None else ""
                md.append(
                    f"- **{VARIANT_CN.get(v, v)} · {c.get('task')}{ctx_s} "
                    f"特异性新增错误**：问题「{str(c.get('excerpt') or '')[-100:]}」；"
                    f"输出 `{c.get('pred')}` vs gold `{c.get('answer')}`")
        if n_shared and bad_cross["shared_failures"]:
            c0 = bad_cross["shared_failures"][0]
            ctx_s = f"@ctx{c0['ctx']}" if c0.get("ctx") is not None else ""
            md.append(f"- 共识失败示例：{c0.get('task')}{ctx_s}，问题"
                      f"「{str(c0.get('excerpt') or '')[-100:]}」；gold "
                      f"`{c0.get('answer')}`（所有变体都错，先检查样本本身）")
        md.append("")
        md.append("注: 跨变体对照只使用各变体最小种子的错误清单以保证样本集一致，"
                  "多种子时不同种子的失败集合变化属正常波动，不在此表统计。")
        _render_advice(md, [("跨变体", r) for r in
                            (bad_cross.get("recommendations") or [])],
                       "### 坏案例驱动的改进方向（跨变体）", max_items=4)
    elif per_variant_badcases:
        md += ["", "（仅一个变体有错误样本，无法做跨变体对照）"]

    # --- 语言建模最差窗口（PPL 坏案例）：PPL 只反映平均，错误必须看得见
    md += ["", "## 语言建模最差窗口（预测错误的样本）", ""]
    per_variant_windows = {}
    for v in variants:
        m = _variant_seed_map(runs, v)
        # 取第一个真正转储了最差窗口的种子：最小种子的评测可能没开
        # --worst_n，若只看它会把其他种子的窗口一并静默丢弃。
        ev0, wins = {}, []
        for s in _sorted_seeds(m):
            cand = m[s].get("eval") or {}
            cand_wins = cand.get("worst_windows") or []
            if cand_wins:
                ev0, wins = cand, cand_wins
                break
        if not wins:
            continue
        # 评测窗口集合由 (val 长度, ctx, eval_iters) 决定，与训练种子无关：
        # 给每个窗口写入稳定 key，供跨变体对齐
        keyed = []
        for w in wins:
            w = dict(w)
            w.setdefault("split", ev0.get("split"))
            w["sample_id"] = f"lm:{w.get('split') or '?'}#w{w.get('window')}"
            keyed.append(w)
        per_variant_windows[v] = keyed
    win_cross = None
    if per_variant_windows:
        md.append("PPL 是所有窗口的平均，看不出「模型究竟错了哪句」。下表按窗口"
                  "末位交叉熵挑出各变体最差的一次 next-token 预测并归因："
                  "高置信错答=top1 概率≥0.5 却错（系统性盲区，含高置信复读）；"
                  "复读吸引子=低置信下首选续写退化为抄上下文已有串；低置信="
                  "模型在多个候选间平均（样本本身多解/噪声，或训练不足）。"
                  "逐案例详情见 `worstcases_*.md`。")
        md += ["", "| 变体 | 最差窗口数 | 主要归因 | 平均 top-1 概率 "
                   "| gold 在上下文中 | gold 进 top-5 | 平均 loss |",
               "|---|---|---|---|---|---|---|"]
        window_pool, seen_cat, window_advice = [], set(), []
        for v, wins in per_variant_windows.items():
            ana = analyze_worst_windows(wins, variant=v)
            window_advice += [(v, r) for r in (ana.get("recommendations") or [])]
            cats = sorted(ana["category_counts"].items(), key=lambda kv: -kv[1])
            cat_s = "；".join(f"{WINDOW_CATEGORY_CN.get(k, k)}×{n}"
                              for k, n in cats[:3])
            mt1 = ("-" if ana["mean_top1_prob"] is None
                   else f"{ana['mean_top1_prob']:.3f}")
            gic = ("-" if ana["gold_in_context_rate"] is None
                   else f"{ana['gold_in_context_rate']:.0%}")
            g5 = ("-" if ana["gold_in_top5_rate"] is None
                  else f"{ana['gold_in_top5_rate']:.0%}")
            ml = ("-" if ana["mean_loss"] is None else f"{ana['mean_loss']:.3f}")
            md.append(f"| {VARIANT_CN.get(v, v)} | {ana['n_windows']} | {cat_s} "
                      f"| {mt1} | {gic} | {g5} | {ml} |")
            for c in ana["cases"]:
                if c["category"] not in seen_cat:
                    seen_cat.add(c["category"])
                    window_pool.append((v, c))
        # 摘 3 条不同类别的个案进汇总页
        for v, c in window_pool[:3]:
            tail = str(c.get("context_tail") or "").replace("\n", " ")[-100:]
            p1 = c.get("top1_prob")
            p1_s = f"（p={p1:.3f}）" if p1 is not None else ""
            tag = ("gold 在上下文中" if c.get("gold_in_context")
                   else "gold 不在上下文中")
            md += [f"- **{VARIANT_CN.get(v, v)} · 窗口 #{c.get('window')}"
                   + (f"（{c['split']}）" if c.get("split") else "")
                   + f"** —— {WINDOW_CATEGORY_CN.get(c['category'], c['category'])}",
                   f"  - 上下文尾部：…{tail}",
                   f"  - 模型 top-1：`{c.get('top1')}`{p1_s}；真实续写："
                   f"`{c.get('gold_next')}`（{tag}）"]
        if len(per_variant_windows) >= 2:
            win_cross = cross_variant_analysis(per_variant_windows,
                                               reference="standard")
            n_shared_w = win_cross["n_shared_all_failures"]
            md.append("")
            md.append(
                f"跨变体看：{win_cross['n_unique_failed_items']} 个不同的最差窗口，"
                f"其中 {n_shared_w} 个为全体变体共识失败（更可能是数据的局部"
                "不可预测/噪声，而非某一模型的问题）；"
                + (f"⚠ 参照 standard 无最差窗口记录，新增/改对计数不可得。"
                   if not win_cross.get("reference_available") else
                   "相对 standard 的新增窗口才是该方法引入的预测退化。"))
            window_advice += [("跨变体", r) for r in
                              (win_cross.get("recommendations") or [])]
        _render_advice(md, window_advice,
                       "### 坏案例驱动的改进方向（最差窗口）", max_items=4)
    else:
        md.append("- 暂无最差窗口记录（评测未含 --worst_n 或评测被跳过）。"
                  "PPL 平均分无法替代对错误样本的逐条检查。")

    # --- 消融矩阵（全因子 + 去一法）
    present = set(runs.keys())
    md += ["", "## 消融矩阵（全因子 + 去一法）", ""]
    factorial_done = [(name, v) for name, v in FACTORIAL_CELLS if v in present]
    factorial_missing = [(name, v) for name, v in FACTORIAL_CELLS if v not in present]
    loo_missing = [(name, v) for name, v in LOO_CELLS if v not in present]
    rows_by_var = {r["variant"]: r for r in rows}
    md.append("| 组合 | 变体 | PPL↓ | RULER平均↑ |")
    md.append("|---|---|---|---|")
    for name, v in FACTORIAL_CELLS:
        r = rows_by_var.get(v)
        if r:
            md.append(f"| {name} | {VARIANT_CN.get(v, v)} | "
                      f"{fmt_stat_cell(r, 'val_ppl')} | "
                      f"{fmt_stat_cell(r, 'ruler_avg')} |")
        else:
            md.append(f"| {name} | {v}（未运行） | - | - |")
    if not factorial_missing:
        md.append("")
        md.append("全因子组合覆盖 8/8、去一法覆盖完整；但组合覆盖只是必要条件，"
                  "组件是否有用须以下方统计证据为准。")
    elif not loo_missing:
        md.append("")
        md.append("⚠ 已覆盖去一法（Full/Full−A/Full−B/Full−C），但全因子缺失组合："
                  + "、".join(f"{n}({v})" for n, v in factorial_missing)
                  + "；涉及这些组合的单独组件结论不完整。")
    else:
        md.append("")
        md.append("⚠ **消融不完整**：全因子缺失 "
                  + "、".join(f"{n}({v})" for n, v in factorial_missing)
                  + "；去一法缺失 "
                  + "、".join(f"{n}({v})" for n, v in loo_missing)
                  + "。仅凭部分组合无法支撑“每个组件都有用”，请补齐后重跑。")

    # --- 组件有效性判定：只有 Holm 校正后的显著证据才能支撑“组件有用”
    md += ["", "### 组件有效性判定（同种子配对 t 检验，Holm 校正后 p<0.05 才算证据）", ""]
    md.append("| 组件 | 去一法 Full−X vs Full | 全因子 X vs 无门控 | 判定 |")
    md.append("|---|---|---|---|")

    def _fmt_evidence(st):
        p = st.get("p_adj")
        ps = f"p_adj={p:.4f}" if p is not None else "p_adj=-"
        return f"{'✅' if st['supported'] else '—'} {ps}（{st['reason']}）"

    for c in comp:
        loo_cells = "；".join(f"{m}: {_fmt_evidence(c['loo'][m])}"
                             for m in _EVIDENCE_METRICS)
        add_cells = "；".join(f"{m}: {_fmt_evidence(c['add'][m])}"
                             for m in _EVIDENCE_METRICS)
        md.append(f"| {c['component']} | {loo_cells} | {add_cells} | "
                  f"{c['verdict']} |")
    supported = [c["component"] for c in comp if c["supported"]]
    unsupported = [c["component"] for c in comp if not c["supported"]]
    if not comp:
        md.append("")
        md.append("⚠ 无法判定组件有效性（无配对检验结果）。")
    elif not unsupported and not factorial_missing:
        md.append("")
        md.append("✅ 统计证据支持“各组件均有用”：A/B/C 三个组件均至少在一个指标上"
                  "有 Holm 校正后的显著证据，且全因子消融 8/8 完整；"
                  "去一法与全因子两套消融可交叉印证。")
    else:
        md.append("")
        if unsupported:
            md.append("⚠ **不能下“各组件均有用”的结论**："
                      + "、".join(unsupported)
                      + " 缺少 Holm 校正后的显著证据（幅度不足、方向相反或"
                      "种子数不够）；组合齐全并不等于组件有效。")
        if factorial_missing:
            md.append("⚠ 全因子缺失组合："
                      + "、".join(f"{n}({v})" for n, v in factorial_missing)
                      + "，组件单独贡献的覆盖不完整。")
    md.append("")
    md.append("注: 去一法证据=去掉该组件的 Full−X 显著差于 Full（去掉有害）；"
              "全因子证据=仅含该组件的 X 显著优于无门控 standard（只加它也有收益）。"
              "判定要求方向正确且 Holm 校正后 p<0.05、种子数≥3；"
              "两套消融结论不一致时应以“证据不足”处理并补实验。")

    # --- 配对检验表
    md += ["", "## 关键对比（同种子配对，t 检验 + 置换检验双重印证，Holm 校正）", ""]
    if tests:
        md.append("| 对比 | 指标 | Δ均值 ± Δ标准差 | Δ 95% CI | d_z | n | p | "
                  "p_adj(Holm) | p_perm(Holm) | MDE | 结论 |")
        md.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for t in tests:
            if t.get("mean_diff") is None:
                continue
            std = t.get("std_diff")
            delta = (f"{t['mean_diff']:+.4f}"
                     + (f" ± {std:.4f}" if std is not None else ""))
            ci = t.get("mean_diff_ci95")
            ci_s = (f"[{ci[0]:+.4f}, {ci[1]:+.4f}]"
                    if ci else "-")
            p = f"{t['p']:.4f}" if t.get("p") is not None else "-"
            padj = (f"{t['p_adj']:.4f}" if t.get("p_adj") is not None else "-")
            ppadj = (f"{t['p_perm_adj']:.4f}"
                     if t.get("p_perm_adj") is not None else "-")
            dz = f"{t['d_z']:+.2f}" if t.get("d_z") is not None else "-"
            mde = (f"{t['mde']:+.4f}" if t.get("mde") is not None else "-")
            md.append(f"| {t['candidate']} vs {t['reference']} | {t['metric']} | "
                      f"{delta} | {ci_s} | {dz} | {t['n']} | {p} | {padj} | "
                      f"{ppadj} | {mde} | {t.get('verdict', '-')} |")
        md.append("")
        md.append("注: Δ = 候选 − 参照；PPL 越小越好、RULER/AUROC差 越大越好；"
                  "d_z = Δ均值/Δ标准差为配对 Cohen's d 效应量。"
                  f"p 为双侧配对 t 检验原始值，p_adj 为同一指标家族内 "
                  f"Holm–Bonferroni 校正值（控制多重比较的 FWER），"
                  f"结论一律基于 p_adj<{ALPHA} 且 n≥{MIN_SEEDS}；"
                  f"n<3 只报描述统计、不判定显著性。种子不足时请先补齐种子再下结论。")
        md.append("注: p_perm 为符号翻转置换检验（分布无关；n≤16 全枚举精确）"
                  "在同一指标家族内独立 Holm 校正后的值——种子只有 3~5 个时"
                  "t 检验的正态假设不可靠，置换检验不支持的“显著”会标注"
                  "⚠。**粒度边界**：置换双侧 p 值的理论下限是 2/2^n——"
                  "n=3 时 0.25、n=5 时 0.0625、n=6 才降至 0.03125：种子数 <6 "
                  "时“置换不显著”是粒度所限而非 t 检验一定虚警，此时显著性"
                  "完全依赖 t 的正态假设；n≥6 后两套检验不一致才是真实的"
                  "稳健性警报。Δ 95% CI 为配对差的 bootstrap 区间；MDE = 当前"
                  "种子数与方差下 80% 功效可检测的最小效应：观测差远小于 "
                  "MDE 的“不显著”是功效不足，不是无效应。")
    else:
        md.append("（无可配对的对比；请对多个变体运行同一组种子）")

    # --- 注册假设判定：confirmatory 与 exploratory 的边界必须是显式的
    md += ["", "## 注册假设判定（confirmatory；其余对比均为事后探索）", ""]
    md += _render_registration_integrity(integrity, scope_check=scope_check)
    if hyp_eval:
        n_sup = sum(1 for h in hyp_eval if h["status"] == "支持")
        n_ref = sum(1 for h in hyp_eval if h["status"] == "拒绝")
        n_und = sum(1 for h in hyp_eval if h["status"] == "未定")
        valid = all(h.get("confirmatory_valid") for h in hyp_eval)
        md.append(f"预注册文件 `preregistration/hypotheses.json`（登记于正式实验前；"
                  f"不放在 docs/ 以免被流水线自动清理删除）："
                  f"{len(hyp_eval)} 条方向性主张，判定 支持 {n_sup}、拒绝 {n_ref}、"
                  f"未定 {n_und}。只有“支持”能写进结论；“拒绝”= 方向相反且校正后"
                  "显著（假设被证伪）；“未定”不得解释为任何方向的证据。")
        if not valid:
            md.append("")
            md.append("> ⛔ 预注册完整性校验未通过：下表的“支持/拒绝”不作数，"
                      "已按探索性线索处理（见上方完整性小节）。")
        md.append("")
        md.append("| ID | 主张 | Δ均值 | n | p_adj | p_perm | MDE | 判定 |")
        md.append("|---|---|---|---|---|---|---|---|")
        for h in hyp_eval:
            padj = (f"{h['p_adj']:.4f}" if h.get("p_adj") is not None else "-")
            pp = (f"{h['p_perm_adj']:.4f}"
                  if h.get("p_perm_adj") is not None else "-")
            delta = (f"{h['mean_diff']:+.4f}"
                     if h.get("mean_diff") is not None else "-")
            mde = f"{h['mde']:+.4f}" if h.get("mde") is not None else "-"
            mark = {"支持": "✅", "拒绝": "❌"}.get(h["status"], "—")
            if not h.get("confirmatory_valid"):
                mark = "⚠（无效）"
            md.append(f"| {h['id']} | {h['claim']} | {delta} | "
                      f"{h.get('n') if h.get('n') is not None else '-'} | "
                      f"{padj} | {pp} | {mde} | {mark} {h['status']}"
                      f"（{h['detail']}） |")
        md.append("")
        md.append("注: 未登记在本文件里的对比（含本汇总表中的其余行）一律属"
                  "事后（exploratory）分析：p 值只受本表家族内校正保护，"
                  "其“显著”只能作为提出新假设的线索；要升级为结论须先登记"
                  "再在新的种子/数据上检验。")
    else:
        md.append("- ⚠ 未找到可用的注册假设文件（`preregistration/hypotheses.json`）。"
                  "本页全部对比均为事后分析：没有预注册边界，任何“显著”都只能"
                  "按探索性线索对待。请登记方向性主张后重跑正式实验。")

    # --- 涨点-增幅权衡：评价改进不能只看指标涨点
    md += ["", "## 涨点-增幅权衡（涨点必须对照增幅）", ""]
    if common_ctx is not None and lat_entries.get("standard"):
        base_e = lat_entries["standard"]
        base_prefill = (base_e.get("prefill_ms") or {}).get(str(common_ctx), {}).get("1")
        base_decode = (base_e.get("decode_ms_per_tok") or {}).get(str(common_ctx))
        base_params = _eff_cell(base_e, "params", common_ctx)
        base_flops = _eff_cell(base_e, "flops", common_ctx)
        md.append(f"基线=standard（公共 ctx={common_ctx}、batch=1）。增幅="
                  "Prefill ms/Decode ms(tok)/训练 ms(step)/训练进程峰值/参数量/"
                  "FLOPs(token)/实测分配量 相对基线增幅的最大值（Prefill/Decode "
                  "的差异若不超过两端重复测量的噪声散布则按不可区分处理，标 † "
                  "且不计入增幅）。收益覆盖两个主指标：PPL 相对改善（%）与 "
                  "RULER 平均 EM 改善（绝对百分点 pp）——预注册主张横跨两者"
                  "（H1 PPL、H4 长上下文检索），只看 PPL 会误判长上下文涨点，"
                  "也会漏掉 RULER 显著退步。判定规则：两个主指标都无数据或"
                  "增幅缺失→无法判定；任一参与判定的指标种子不足（n<3）→无法"
                  "判定；某一主指标显著差于基线（Holm 校正后 p<α）→不可行，"
                  "但另一主指标同时显著改善时判“权衡”并显式说明取舍；任一主"
                  "指标显著改善即有涨点（按显著者中最不利的收益套用收支比："
                  "PPL >5×收益、RULER 单独 >10×收益且下限 0.5 pp，单位不同不"
                  "混算），涨点不显著→说服力有限（增幅>10% 判不可行）；显著但"
                  "增幅>50% 或超过上述收支比 →收益-增幅比不利/需审视。微小改善"
                  "伴随巨大增幅的方案会被明确拒绝。")

        def _train_step_ms(v):
            """该变体各种子训练步时的中位数（毫秒/步；旧 eval.json 无此字段→None）。

            同一 run_all 内所有变体在同一台机器、同一批次下训练，该步时是
            同口径的直接可比时序（评测与 checkpoint 写入已从区间计时中
            扣除）；中位数避免首步冷启动与偶发卡顿污染。
            """
            vals = []
            for s in _variant_seed_map(runs, v).values():
                x = (s.get("eval") or {}).get("train_step_time_ms")
                if x is None:
                    continue
                try:
                    vals.append(float(x))
                except (TypeError, ValueError):
                    continue
            if not vals:
                return None
            vals.sort()
            n = len(vals)
            return vals[n // 2] if n % 2 else 0.5 * (vals[n // 2 - 1] + vals[n // 2])

        def _train_peak_mem_mb(v):
            """该变体各种子训练进程峰值的跨种子中位数（MB；旧 eval.json 无→None）。

            与推理单次前向的实测值分开报告：两者口径不同（进程级高水位
            含优化器状态与激活 vs prefill 单次分配）。
            """
            vals = []
            for s in _variant_seed_map(runs, v).values():
                x = (s.get("eval") or {}).get("train_peak_mem_mb")
                if x is None:
                    continue
                try:
                    vals.append(float(x))
                except (TypeError, ValueError):
                    continue
            if not vals:
                return None
            vals.sort()
            n = len(vals)
            return vals[n // 2] if n % 2 else 0.5 * (vals[n // 2 - 1] + vals[n // 2])

        md.append("")
        md.append("| 变体 | PPL 相对基线 | p_adj(PPL) | RULER Δ(pp) | p_adj(RULER) "
                  "| Prefill ms | Decode ms/tok "
                  "| 训练 ms/step | 训练峰值 | 参数量 | FLOPs/token | 实测分配 | 判定 |")
        md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        ppl_test = {t["candidate"]: t for t in tests
                    if t.get("reference") == "standard" and t.get("metric") == "PPL"
                    and t.get("mean_diff") is not None}
        ruler_test = {t["candidate"]: t for t in tests
                      if t.get("reference") == "standard"
                      and t.get("metric") == "RULER平均"
                      and t.get("mean_diff") is not None}
        base_train_ms = _train_step_ms("standard")
        base_train_mem = _train_peak_mem_mb("standard")
        n_metric_within_noise = 0
        for r in rows:
            v = r["variant"]
            if v == "standard" or not lat_entries.get(v):
                continue
            e = lat_entries[v]
            ppl = r.get("val_ppl")
            std_row = next((x for x in rows if x["variant"] == "standard"), None)
            std_ppl = (std_row or {}).get("val_ppl")
            benefit = (-_rel_pct(std_ppl, ppl)) if (ppl is not None and std_ppl) else None
            t = ppl_test.get(v)
            p_adj = t.get("p_adj") if t else None
            n_ppl = t.get("n") if t else None
            # RULER 平均 EM 的配对差是比例（0~1），换算为绝对百分点后进入
            # 权衡判定：与 PPL 的相对百分比单位不同，绝不混算收益比。
            rt = ruler_test.get(v)
            ruler_benefit_pp = (rt["mean_diff"] * 100.0
                                if rt and rt.get("mean_diff") is not None
                                else None)
            ruler_p_adj = rt.get("p_adj") if rt else None
            n_ruler = rt.get("n") if rt else None
            pre_delta = _rel_pct(base_prefill,
                                 (e.get("prefill_ms") or {}).get(str(common_ctx), {}).get("1"))
            dec_v = (e.get("decode_ms_per_tok") or {}).get(str(common_ctx))
            dec_delta = _rel_pct(base_decode, dec_v)
            tr_v = _train_step_ms(v)
            tr_delta = _rel_pct(base_train_ms, tr_v)
            memtr_v = _train_peak_mem_mb(v)
            memtr_delta = _rel_pct(base_train_mem, memtr_v)
            par_delta = _rel_pct(base_params, _eff_cell(e, "params", common_ctx))
            flo_delta = _rel_pct(base_flops, _eff_cell(e, "flops", common_ctx))
            mem_v = _eff_cell(e, "mem", common_ctx)
            mem_b = _eff_cell(base_e, "mem", common_ctx)
            mem_delta = _rel_pct(mem_b, mem_v) if (mem_v is not None and mem_b) else None
            # 运行时差异的测量噪声：|Δ| 不超过两端散布的方和根时不可区分，
            # 按 0 计入增幅（原始值仍保留在 JSON，避免把噪声当真实差异）。
            pre_noise = _noise_floor_pct(
                _lat_noise_pct(base_e, "prefill", common_ctx),
                _lat_noise_pct(e, "prefill", common_ctx))
            dec_noise = _noise_floor_pct(
                _lat_noise_pct(base_e, "decode", common_ctx),
                _lat_noise_pct(e, "decode", common_ctx))
            pre_within = (pre_delta is not None and pre_noise is not None
                          and abs(pre_delta) <= pre_noise)
            dec_within = (dec_delta is not None and dec_noise is not None
                          and abs(dec_delta) <= dec_noise)
            n_metric_within_noise += int(pre_within) + int(dec_within)
            deltas = [d for d in ((0.0 if pre_within else pre_delta),
                                  (0.0 if dec_within else dec_delta),
                                  tr_delta, memtr_delta, par_delta, flo_delta,
                                  mem_delta)
                      if d is not None]
            cost = max(deltas) if deltas else None
            verdict = tradeoff_verdict(benefit, p_adj, cost, n_seeds=n_ppl,
                                       ruler_benefit_pp=ruler_benefit_pp,
                                       ruler_p_adj=ruler_p_adj,
                                       ruler_n_seeds=n_ruler)
            r["eff_cost_max_pct"] = round(cost, 2) if cost is not None else None
            r["eff_flops_per_tok"] = _eff_cell(e, "flops", common_ctx)
            r["eff_prefill_delta_pct"] = (round(pre_delta, 2)
                                          if pre_delta is not None else None)
            r["eff_prefill_noise_pct"] = (round(pre_noise, 2)
                                          if pre_noise is not None else None)
            r["eff_prefill_within_noise"] = pre_within
            r["eff_decode_ms_per_tok"] = dec_v
            r["eff_decode_delta_pct"] = (round(dec_delta, 2)
                                         if dec_delta is not None else None)
            r["eff_decode_noise_pct"] = (round(dec_noise, 2)
                                         if dec_noise is not None else None)
            r["eff_decode_within_noise"] = dec_within
            r["eff_train_step_ms"] = tr_v
            r["eff_train_delta_pct"] = (round(tr_delta, 2)
                                        if tr_delta is not None else None)
            r["eff_train_peak_mem_mb"] = memtr_v
            r["eff_train_mem_delta_pct"] = (round(memtr_delta, 2)
                                            if memtr_delta is not None else None)
            r["eff_peak_mem_mb"] = mem_v
            r["eff_benefit_ppl_pct"] = (round(benefit, 2)
                                        if benefit is not None else None)
            r["eff_ruler_p_adj"] = ruler_p_adj
            r["eff_benefit_ruler_pp"] = (round(ruler_benefit_pp, 3)
                                         if ruler_benefit_pp is not None else None)
            r["eff_verdict"] = verdict

            def _d(x, plus=True, within=False):
                if x is None:
                    return "-"
                s = f"{x:+.1f}%" if plus else f"{x:.1f}%"
                return s + "†" if within else s

            def _dpp(x):
                return "-" if x is None else f"{x:+.2f} pp"

            md.append(f"| {VARIANT_CN.get(v, v)} | {_d(benefit)} "
                      f"| {'-' if p_adj is None else f'{p_adj:.4f}'} "
                      f"| {_dpp(ruler_benefit_pp)} "
                      f"| {'-' if ruler_p_adj is None else f'{ruler_p_adj:.4f}'} "
                      f"| {_d(pre_delta, within=pre_within)} "
                      f"| {_d(dec_delta, within=dec_within)} | {_d(tr_delta)} "
                      f"| {_d(memtr_delta)} "
                      f"| {_d(par_delta)} "
                      f"| {_d(flo_delta)} "
                      f"| {'-' if mem_v is None else f'{mem_v:.0f} MB'} "
                      f"| {verdict} |")
        md.append("")
        if n_metric_within_noise:
            md.append(f"† {n_metric_within_noise} 个运行时涨幅落在重复测量的"
                      "噪声区间内（[p10,p90] 相对散布的方和根），按不可区分"
                      "处理、不计入增幅；原始差值仍在 summary.json 中。")
        md.append("注：RULER Δ(pp) 为候选相对 standard 的平均 EM 配对差"
                  "（绝对百分点，越大越好；样本集跨种子相同、同种子配对），"
                  "p_adj(RULER) 取自 RULER 指标家族的 Holm 校正配对检验；"
                  "判定在 PPL 与 RULER 任一显著涨点成立时注明依据"
                  "（可接受（依据 …）），显著退步会单独给出权衡/不可行结论，"
                  "不会因另一个指标涨而被静默忽略。"
                  "FLOPs/token 为解析估计（matmul 2·MACs + 注意力 4·kv_len·d/层 "
                  "+ 门控项，见 latency.json 的 efficiency 字段）；实测分配"
                  "为 batch=1 prefill 实测（CUDA），CPU 环境不适用。"
                  "Decode ms/tok 为带 KV 缓存的逐 token 解码时序，"
                  "训练 ms/step 为纯训练步（区间墙钟已扣除评测与 "
                  "checkpoint 写入）跨种子中位数（见 eval_*.json 的 "
                  "train_step_time_ms），训练峰值为该变体训练进程的进程级"
                  "高水位（含模型/优化器/激活/评测窗口，CUDA，跨种子中位数；"
                  "见 eval_*.json 的 train_peak_mem_mb；CPU 训练为 '-'），"
                  "均为相对基线的百分比变化。"
                  "Prefill/Decode 为逐次计时的中位数，重复测量的相对散布见 "
                  "latency.json 的 *_spread_pct；运行时间指标受硬件与同机负载"
                  "影响，跨机器对比无效。")
    else:
        md.append("- ⚠ 缺少运行时/结构基准（latency.json 或 standard 基线），"
                  "无法给出权衡判定；没有增幅对照的涨点不能作为结论。")

    os.makedirs(results_dir, exist_ok=True)
    md_text = "\n".join(md)
    with open(os.path.join(results_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write(md_text)

    # CSV（统计列以并列的 mean/std/n 展开，便于程序消费）
    if rows:
        all_keys = sorted({k for r in rows for k in r})
        with open(os.path.join(results_dir, "summary.csv"), "w", newline="",
                  encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=all_keys)
            w.writeheader()
            w.writerows(rows)
    with open(os.path.join(results_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    with open(os.path.join(results_dir, "significance.json"), "w",
              encoding="utf-8") as f:
        json.dump({"pairs": tests,
                   "note": (f"同种子配对 t 检验；结论基于同一指标家族内 "
                            f"Holm–Bonferroni 校正后的 p_adj（alpha={ALPHA}），"
                            f"n<{MIN_SEEDS} 不判定显著性；"
                            "d_z 为配对 Cohen's d 效应量。p_perm/p_perm_adj 为"
                            "符号翻转置换检验（分布无关，n≤16 精确）的原始/家族内"
                            "Holm 校正值，t 与置换结论冲突时见 t_perm_agree 与"
                            "verdict 中的 ⚠ 标注；mean_diff_ci95 为配对差 "
                            "bootstrap 95% 区间；mde 为当前种子数/方差下 80% "
                            "功效的最小可检测效应（不显著时先对照 MDE 再下结论）"),
                   "component_evidence": comp,
                   "registered_hypotheses": hyp_eval,
                   "registered_hypotheses_source": (
                       "preregistration/hypotheses.json" if hyp_eval else None),
                   "registration_integrity": integrity,
                   "registration_scope": scope_check,
                   "ngram_comparison": ngram_cmp,
                   "linear_probe_comparison": probe_cmp,
                   "auroc_diff_note": (
                       f"{AUROC_DIFF_METRIC} 家族为机制主张的正式检验：每个 "
                       "(变体, 种子) run 的 auroc_logit − auroc_qterm 配对差，"
                       "与逐 run 样例聚类 bootstrap 区间"
                       "（analysis_*.json 的 auroc_cluster_bootstrap）互为印证")},
                  f, ensure_ascii=False, indent=2)
    cross_out = {}
    if bad_cross:
        cross_out["ruler"] = bad_cross
    if win_cross:
        cross_out["lm_windows"] = win_cross
    if cross_out:
        save_json(cross_out, os.path.join(results_dir, "badcase_cross.json"))
    print(f"[summary] 汇总 {len(rows)} 个变体 -> {results_dir}/summary.md "
          f"（配对检验 {len(tests)} 组）")
    return {"rows": rows, "markdown": md_text, "tests": tests}


def main():
    p = argparse.ArgumentParser(description="一键实验流水线（多种子 + 受控对比 + 完整消融）")
    p.add_argument("--config", default="configs/tiny.json")
    p.add_argument("--data_dir", default="data/dummy")
    p.add_argument("--out_root", default="out")
    p.add_argument("--results_dir", default=None, help="默认 results/<tag>")
    p.add_argument("--variants", default=",".join(ALL_VARIANTS))
    p.add_argument("--seeds", default=None,
                   help="逗号分隔的随机种子；默认取配置 seeds（冒烟模式仅取首个）")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--skip_train", action="store_true")
    p.add_argument("--skip_sanity", action="store_true",
                   help="跳过过拟合小批量 sanity check（默认每变体训练前必跑，"
                        "失败即中止全量训练）")
    p.add_argument("--sanity_steps", type=int, default=None,
                   help="sanity check 的优化步数（默认 300，冒烟 60）")
    p.add_argument("--early_stop_patience", type=int, default=None,
                   help="验证损失连续 N 次评测无实质改善则早停（透传 lega.train；"
                        "默认读配置 early_stop_patience，缺省关闭）。启用后各 run "
                        "实际步数可不同，受控变量审计按计划步数核对并在汇总页"
                        "如实标记")
    p.add_argument("--skip_ruler", action="store_true")
    p.add_argument("--skip_latency", action="store_true")
    p.add_argument("--skip_analysis", action="store_true")
    p.add_argument("--skip_plots", action="store_true")
    p.add_argument("--tune_baseline", action="store_true",
                   help="训练前对基线（standard）做 lr/权重衰减网格调优（仅验证集）；"
                        "选定组合注入配置并统一用于全部变体，证据写入 "
                        "results/<tag>/baseline_tuning.json")
    p.add_argument("--tune_steps", type=int, default=None,
                   help="调优 trial 的训练步数（默认与正式训练相同；显式缩小步数时"
                        "汇总页会如实记录，不能当作与正式同步数的调优）")
    p.add_argument("--tune_lr_mults", default="0.5,1,2",
                   help="调优网格的 lr 乘子（逗号分隔，默认 0.5,1,2）")
    p.add_argument("--tune_wd_list", default=None,
                   help="调优网格的 weight_decay 列表（逗号分隔；默认仅配置值）")
    p.add_argument("--tune_seeds", default=None,
                   help="调优选择用的种子（逗号分隔；默认只用首个正式种子）。"
                        "单种子选择可能被 val_loss 噪声左右（winner's curse），"
                        "正式实验建议给 ≥3 个（网格配置数 × 种子数 次训练）")
    p.add_argument("--skip_ngram", action="store_true",
                   help="跳过简单统计基线（一元/二元平滑 n-gram PPL）")
    p.add_argument("--ngram_split", default="auto",
                   choices=["auto", "val", "test"],
                   help="n-gram 基线评测 split；auto（默认）与 lega.evaluate 的 "
                        "--split auto 同规则：有非空 test.bin 用 held-out test，"
                        "否则回退 val——保证与神经模型 PPL 同 split 可比；"
                        "test 会写访问账本")
    p.add_argument("--jobs", type=int, default=None,
                   help="sanity/训练/机制分析阶段的并发子进程数；默认 CPU 上 "
                        "min(4, 核数)（<=4 核时为核数+1），CUDA 或 --wandb 下为 1")
    args = p.parse_args()

    args.config = _repo_path(args.config)
    args.data_dir = _repo_path(args.data_dir)
    py = sys.executable
    tag = os.path.splitext(os.path.basename(args.config))[0] + ("_smoke" if args.smoke else "")
    out_root = os.path.join(_repo_path(args.out_root), tag)
    results_dir = _repo_path(args.results_dir) if args.results_dir else os.path.join(ROOT, "results", tag)
    os.makedirs(results_dir, exist_ok=True)
    variants = parse_variants(args.variants)
    invalid = sorted(set(variants) - set(ALL_VARIANTS))
    if invalid:
        p.error(f"未知变体: {', '.join(invalid)}；可选: {', '.join(ALL_VARIANTS)}")

    if not os.path.exists(args.config):
        p.error(f"配置文件不存在: {args.config}")
    with open(args.config, "r", encoding="utf-8") as cf:
        cfg = json.load(cf)

    # 多种子协议：显式 --seeds 优先；否则用配置 seeds；冒烟模式只跑首个种子。
    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    else:
        cfg_seeds = cfg.get("seeds") or [cfg.get("seed", 1337)]
        seeds = [int(cfg_seeds[0])] if args.smoke else [int(s) for s in cfg_seeds]
    seeds = list(dict.fromkeys(seeds))
    if not seeds:
        p.error("种子列表为空")
    n_seeds = len(seeds)
    print(f"[run_all] 变体={variants}")
    print(f"[run_all] 种子={seeds}（{n_seeds} 个）tag={tag}")
    jobs = _resolve_jobs(args)
    if jobs > 1:
        print(f"[run_all] jobs={jobs}（sanity/训练/评测/机制分析并发；--jobs 可覆盖）")

    # 子进程随机性/确定性环境：PYTHONHASHSEED 必须在解释器启动前设置；
    # 这里给出基准值（无 --seed 的命令用），带 --seed 的命令由 _cmd_env 逐条
    # 覆盖为自身种子。CUBLAS 工作区配置须在任何 CUDA 初始化前生效。
    os.environ.setdefault("PYTHONHASHSEED", str(seeds[0]))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    # 基线调优（可选）：先调 standard 的 lr/权重衰减（只用验证集，test 不加载），
    # 把最优组合注入配置并统一用于所有变体——保证对比是“调过的基线 vs 新方法”，
    # 而不是把超参数红利误归因给模块。证据落盘 results/<tag>/baseline_tuning.json。
    if args.tune_baseline and args.skip_train:
        print("[tuning] --skip_train：跳过基线调优（没有要训练的新 run）")
    elif args.tune_baseline:
        tune_root = os.path.join(out_root, "_baseline_tuning")
        tune_seeds = ([int(s.strip()) for s in args.tune_seeds.split(",")
                       if s.strip()] if args.tune_seeds else None)
        tune_report = tune_baseline(
            args.config, args.data_dir, tune_root, variant="standard",
            seed=seeds[0], seeds=tune_seeds, steps=args.tune_steps,
            lr_mults=parse_floats(args.tune_lr_mults),
            wd_list=(parse_floats(args.tune_wd_list)
                     if args.tune_wd_list else None),
            device=args.device, tolerance=TUNE_TOLERANCE)
        save_json(tune_report,
                  os.path.join(results_dir, "baseline_tuning.json"))
        best = tune_report["selected"]
        cfg["lr"] = float(best["lr"])
        cfg["weight_decay"] = float(best["weight_decay"])
        tuned_path = os.path.join(results_dir, "config_tuned.json")
        save_json(cfg, tuned_path)
        args.config = tuned_path
        print(f"[tuning] 全部变体改用调优后配置: lr={cfg['lr']:.3g} "
              f"wd={cfg['weight_decay']:.3g} -> {tuned_path}")

    # 简单统计基线：一元/二元平滑 n-gram 的 held-out PPL，给出“数据绝对难度”
    # 参照（零训练零参数；平滑系数只在 val 上选）。split 默认 auto：与神经
    # 模型的评测 split 完全一致（test.bin 存在且非空 → held-out test），
    # 否则汇总页的“神经 vs 统计基线”会变成 test-vs-val 的错位比较。
    if not args.skip_ngram:
        ngram_split = resolve_ngram_split(args.data_dir, args.ngram_split)
        try:
            ng = ngram_evaluate(args.data_dir, split=ngram_split,
                                allow_test=(ngram_split == "test"))
            save_json(ng, os.path.join(results_dir, "ngram_baseline.json"))
            print(f"[ngram] 简单基线 {ngram_split} PPL: "
                  f"unigram={ng['ppl'].get('unigram')} "
                  f"bigram={ng['ppl'].get('bigram')} "
                  f"(α={ng['selected_alpha']})")
        except (OSError, ValueError) as e:
            print(f"[run_all] ⚠ n-gram 基线跳过：{e}")

    # 全局溯源清单：本次流水线的代码版本、数据指纹、运行环境与命令。
    git = git_info()
    config_sha256 = sha256_file(args.config)
    # 预注册完整性：把登记表指纹与锁链校验结论写进 repro 与实验台账，
    # 使“本轮实验对应哪份登记表、是否发生静默改写”可事后审计。
    reg_doc = load_registration()
    reg_integrity = verify_registration()
    reg_info = {
        "path": os.path.relpath(HYPOTHESES_PATH, ROOT),
        "present": reg_doc is not None,
        "fingerprint": (registration_fingerprint(reg_doc)
                        if reg_doc is not None else None),
        "integrity": reg_integrity,
    }
    if not reg_integrity.get("confirmatory_valid"):
        print(f"[repro] ⚠ 预注册完整性未通过（status="
              f"{reg_integrity.get('status')}）：本轮全部对比按探索性对待")
    else:
        print(f"[repro] 预注册完整性 {reg_integrity.get('status')}"
              f"（假设 {reg_integrity.get('n_hypotheses')} 条，"
              f"修订链 {reg_integrity.get('chain_length')} 环）")
    try:
        data_fp = data_fingerprint(args.data_dir)
        data_digest = data_fp["digest"]
    except FileNotFoundError as e:
        # 数据 bin 尚未生成（如 --skip_train 复用历史产物）时不阻塞流水线，
        # 但在计数指纹处显式标注缺失，避免汇总页假装数据版本已知。
        data_fp, data_digest = None, None
        print(f"[repro] 警告：无法计算数据指纹（{e}）")
    repro = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tag": tag,
        "argv": sys.argv,
        "config_file": args.config,
        "config_sha256": config_sha256,
        "seeds": seeds,
        "variants": variants,
        "smoke": bool(args.smoke),
        "git": git,
        "data": data_fp,
        "runtime": runtime_info(args.device),
        "registration": reg_info,
    }
    save_json(repro, os.path.join(results_dir, "repro.json"))
    print(f"[repro] code={git.get('commit_short')} dirty={git.get('dirty')} "
          f"data={data_digest} -> {os.path.join(results_dir, 'repro.json')}")

    # 实验台账：本轮 started 先落盘（崩溃轮次也留痕），结束时追加同一 run_id
    # 的 completed 记录（含最终指标与产物哈希），跨轮次可追溯，见 lega/experiment.py。
    ledger = ledger_path()
    run_id = new_run_id(tag, config_sha256)
    ledger_common = dict(
        run_id=run_id, tag=tag, config_path=args.config,
        config_sha256=config_sha256, variants=variants, seeds=seeds,
        smoke=bool(args.smoke), argv=sys.argv, git=git,
        data_fingerprint=data_digest, runtime=repro["runtime"],
        out_root=out_root, results_dir=results_dir,
        time_start=repro["time"], registration=reg_info)
    append_record(ledger, build_experiment_record(status="started",
                                                  **ledger_common))
    print(f"[repro] 实验台账 run_id={run_id} -> {ledger}")

    # 冒烟：少量样例，ctx 取配置 block_size（离线字符分词下 needle+问题
    # 模板约 211 token，128 会让 RULER 全部任务与机制分析被跳过）
    smoke_ctx = min(256, int(cfg.get("block_size", 256))) if args.smoke else 0

    # 过拟合小批量 sanity check：全量训练前的代码正确性闸门（梯度/损失/标签）。
    # 每个变体跑一次（用首个种子），失败立即中止——连小批量都无法过拟合时不应
    # 继续全量训练。通过不代表数据增强/对齐/映射等问题不存在。
    sanity = {}
    if args.skip_sanity or args.skip_train:
        print("[run_all] sanity check 已跳过（--skip_sanity/--skip_train）")
    else:
        cmds = []
        for v in variants:
            out_json = os.path.join(results_dir, f"sanity_{v}.json")
            cmd = [py, "-m", "lega.sanity", "--config", args.config,
                   "--data_dir", args.data_dir, "--variant", v,
                   "--seed", str(seeds[0]), "--out", out_json,
                   "--device", args.device]
            if args.sanity_steps is not None:
                cmd += ["--steps", str(args.sanity_steps)]
            if args.smoke:
                cmd += ["--smoke"]
            print("+", " ".join(cmd), flush=True)
            cmds.append(cmd)
        try:
            run_many(cmds, jobs, _jobs_env(jobs))
        except subprocess.CalledProcessError as e:
            v = variants[cmds.index(e.cmd)]
            raise SystemExit(
                f"sanity check 失败（变体 {v}，退出码 {e.returncode}）：无法过拟合小批量，"
                "说明梯度反传/损失计算/标签对齐存在严重问题；请先排查再开全量"
                f"训练。结果见 {os.path.join(results_dir, f'sanity_{v}.json')}。")
        for v in variants:
            sanity[v] = "pass"
        print(f"[run_all] sanity check 全部通过: {', '.join(sanity.keys())}")

    # 每个 (变体, 种子) 是一个独立 run：训练 → 独立评测 → RULER → 机制分析。
    # 四个阶段由 lega.postrun 在同一进程内顺序执行：单 run 的行为与逐条独立
    # 进程完全相同（各阶段 main() 自行 set_seed/加载模型/写自己的产物，评测/
    # RULER/分析只依赖本 run 的 checkpoint）。evaluate 追加 test 访问账本，
    # append_test_access 已用进程间文件锁保证不丢记录。
    chain = []
    chain_cmds = []
    for v in variants:
        for seed in seeds:
            vdir = _variant_dir(out_root, v, seed, n_seeds)
            suffix = _seed_tag(seed, n_seeds)
            chain.append((v, seed, vdir, suffix))
            cmd = [py, "-m", "lega.postrun",
                   "--config", args.config, "--variant", v, "--seed", str(seed),
                   "--data_dir", args.data_dir, "--out_dir", vdir,
                   "--results_dir", results_dir, "--suffix", suffix,
                   "--device", args.device]
            if args.skip_train:
                cmd += ["--skip_train"]
            if args.wandb:
                cmd += ["--wandb"]
            if args.early_stop_patience is not None:
                cmd += ["--early_stop_patience", str(args.early_stop_patience)]
            if args.smoke:
                cmd += ["--smoke", "--ruler_ctx_lens", str(smoke_ctx),
                        "--ruler_n_samples", "2",
                        "--analysis_n_examples", "2",
                        "--analysis_ctx", str(smoke_ctx)]
            if args.skip_ruler:
                cmd += ["--skip_ruler"]
            if args.skip_analysis or v not in GATED_FOR_ANALYSIS:
                cmd += ["--skip_analysis"]
            print("+", " ".join(cmd), flush=True)
            chain_cmds.append(cmd)
    if chain_cmds:
        try:
            run_many(chain_cmds, jobs, _jobs_env(jobs))
        except subprocess.CalledProcessError as e:
            # 失败可能出现在链的任一阶段（postrun stderr 已标明阶段名与等价
            # 命令）；训练中止最常见的原因是 NaN 防护触发（train.py 会先落盘
            # nan_report.json 再退出），把排查入口直接指给用户
            v0 = variants[chain_cmds.index(e.cmd) // max(1, len(seeds))]
            raise SystemExit(
                f"run 链失败（变体 {v0}，退出码 {e.returncode}）：训练/评测/"
                "机制分析任一阶段失败即中止；若为训练 NaN 中止，对应 run 目录"
                "下已生成 nan_report.json（数据→代码→混合精度→梯度/学习率的"
                "系统排查路径；报告含检测点前的整个 log 区间候选批次与区间内"
                "首个非有限损失步 first_bad_step，数据/激活排查针对失败现场而"
                f"非最后一步批次），排查后重跑；命令: {' '.join(e.cmd)}")

    # 坏案例报告：把每个变体预测错误的样本逐一归因成 Markdown；
    # 错误样本是改进思路的最重要来源，不能只留聚合分数。
    # 解析结果进 preloaded 缓存：build_summary 汇总时使用同一份 payload。
    preloaded = {}
    if not args.skip_ruler:
        for fp in sorted(glob.glob(os.path.join(results_dir, "ruler_*.json"))):
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                ana = process_ruler_json(fp, payload=payload)
            except (OSError, ValueError, KeyError) as e:
                print(f"[badcase] 跳过 {fp}: {e}")
                continue
            preloaded[fp] = payload
            if ana["n_badcases"]:
                top = ", ".join(f"{BADCASE_CN.get(k, k)}×{n}"
                                for k, n in sorted(ana["category_counts"].items(),
                                                   key=lambda kv: -kv[1])[:3])
                print(f"[badcase] {os.path.basename(fp)}: "
                      f"{ana['n_badcases']} 条错误样本 —— {top}"
                      f" -> badcases_*.md")

    # 语言建模坏案例：把 PPL 最差窗口（next-token 预测错误）逐一归因成
    # Markdown，与 RULER 坏案例同等对待——PPL 平均分看不出模型错在哪句话
    for fp in sorted(glob.glob(os.path.join(results_dir, "eval_*.json"))):
        try:
            with open(fp, "r", encoding="utf-8") as f:
                payload = json.load(f)
            ana = process_eval_json(fp, payload=payload)
        except (OSError, ValueError, KeyError) as e:
            print(f"[badcase] 跳过 {fp}: {e}")
            continue
        preloaded[fp] = payload
        if not ana["n_windows"]:
            continue
        top = ", ".join(f"{WINDOW_CATEGORY_CN.get(k, k)}×{n}"
                        for k, n in sorted(ana["category_counts"].items(),
                                           key=lambda kv: -kv[1])[:3])
        print(f"[badcase] {os.path.basename(fp)}: 最差窗口 {ana['n_windows']} 个"
              f" —— {top} -> worstcases_*.md")

    if not args.skip_latency:
        # 显式给出基准的变体清单：lega.latency 的默认值不含消融变体，
        # 不传会让汇总表的消融变体对应列恒为 "-"。多种子时按各变体首个
        # 种子的 checkpoint 测量（该指标由架构决定，与训练种子无关）。
        lat_variants = ",".join(dict.fromkeys(["standard", "standard_sdpa"] + variants))
        run([py, "-m", "lega.latency", "--config", args.config,
             "--data_dir", args.data_dir, "--ckpt_dir", out_root,
             "--out", os.path.join(results_dir, "latency.json"),
             "--variants", lat_variants,
             "--device", args.device] + (["--smoke"] if args.smoke else []))

    if not args.skip_plots:
        run([py, "-m", "lega.plots", "--results_dir", results_dir,
             "--train_root", out_root])

    # 训练曲线诊断：曲线体检（过拟合/欠拟合/发散/NaN）在汇总前完成，
    # 结果落盘 curve_diagnosis.json 并进入 summary.md
    diagnose_training_curves(chain, args.data_dir, results_dir)

    # 数据协议防泄漏报告（prepare_bins 生成）：在所有评测结束后读取，连同实时
    # 测试集访问账本复制进 results，供汇总与审计（早读会漏掉本轮 test 评测）。
    data_report = None
    dr_path = os.path.join(args.data_dir, "data_report.json")
    if os.path.exists(dr_path):
        with open(dr_path, "r", encoding="utf-8") as f:
            data_report = json.load(f)
        data_report["test_access"] = summarize_test_access(
            read_test_access_log(args.data_dir))
        save_json(data_report, os.path.join(results_dir, "data_report.json"))
        print(f"[repro] 防泄漏报告 -> {os.path.join(results_dir, 'data_report.json')}")
    else:
        print("[repro] 警告：数据目录缺少 data_report.json（旧数据或未走统一入口），"
              "汇总将标注防泄漏自检缺失")

    audit = verify_controlled_setup(out_root, variants, seeds, n_seeds,
                                    expected_data_digest=data_digest)
    build_summary(results_dir, audit=audit, data_report=data_report,
                  preloaded=preloaded)

    # 收尾台账：最终指标（逐 run 的 eval.json）+ 全部结果文件的 SHA-256。
    # 曲线/结果文件后续被覆盖或手改，都能通过哈希与当时记录比对发现。
    # evaluate 阶段已对本 run 的 model_best.pt 做过全量哈希（写在
    # results_dir/eval_<variant><suffix>.json 的 ckpt/ckpt_sha256）；
    # 路径匹配时台账使用该值，否则（未评测/旧评测文件）重算。
    known_ckpt = {}
    for v, seed, _vdir, suffix in chain:
        try:
            with open(os.path.join(results_dir, f"eval_{v}{suffix}.json"),
                      "r", encoding="utf-8") as f:
                rec = json.load(f)
        except (OSError, ValueError):
            rec = None
        if isinstance(rec, dict) and rec.get("ckpt_sha256") and rec.get("ckpt"):
            known_ckpt[(v, seed)] = (str(rec["ckpt_sha256"]), str(rec["ckpt"]))
    run_snapshots = []
    for v, seed, vdir, _suffix in chain:
        sha, ckpt = known_ckpt.get((v, seed), (None, None))
        run_snapshots.append(snapshot_train_run(
            v, seed, vdir, known_ckpt_sha256=sha, known_ckpt_path=ckpt))
    result_files = [p for p in glob.glob(os.path.join(results_dir, "*"))
                    if os.path.isfile(p)]
    append_record(ledger, build_experiment_record(
        status="completed", time_end=time.strftime("%Y-%m-%d %H:%M:%S"),
        sanity={"executed": bool(sanity), "verdict": sanity},
        runs=run_snapshots,
        artifacts=collect_artifacts(result_files, rel_to=ROOT),
        **ledger_common))
    print(f"[repro] 实验台账已记录 completed（run_id={run_id}）-> {ledger}")
    print(f"\n[run_all] 全部完成；结果目录: {results_dir}")


if __name__ == "__main__":
    main()
