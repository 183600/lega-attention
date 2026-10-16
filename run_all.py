#!/usr/bin/env python3
"""一键实验流水线：训练全部变体 → PPL → RULER → 延迟 → 机制分析 → 图表 → 汇总表。

变体（相同参数量级与训练步数）:
  standard | qgate | null | lega | lega_no_len | lega_no_conc | lega_no_query
  | lega_conc_only | lega_len_only

对比与消融协议（科研统计口径）:
  * 受控变量：所有变体 × 种子使用同一 config 训练（同一优化器/学习率策略/
    训练步数/数据顺序），训练后自动核对各 run 的 config_used.json，任何与
    “除变体外全部一致”相悖的差异都会直接报错，避免把调参收益误归因给方法；
  * 多种子：关键指标跨随机种子报告 Mean ± Std（种子一致 → 变体间配对比较），
    并用配对 t 检验给出显著性；默认种子数见 configs（tiny 3 / small 5），
    可用 --seeds 覆盖；
  * 消融完整性：全因子（无/A/B/C/AB/AC/BC/ABC）与去一法（Full/Full-A/
    Full-B/Full-C）两套组合全部实现；汇总表显式列出消融矩阵，缺失组合会
    标注“消融不完整”，不允许部分组合支撑全部组件有效的结论。

用法:
  # 冒烟测试（tiny 配置 + 伪数据，几分钟内跑通全流程；默认只跑 1 个种子）
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
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from lega.attention import VARIANTS, VARIANT_CN  # noqa: E402
from lega.utils import mean_std, paired_t_test  # noqa: E402

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

_SEED_RE = re.compile(r"_seed(\d+)$")


def run(cmd, env=None):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT, env=env)


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


def verify_controlled_setup(out_root: str, variants, seeds, n_seeds: int) -> dict:
    """核对所有变体 × 种子确实除待验证模块外使用完全相同的训练配置。

    对比实验的公平性前提：优化器/LR 策略/步数/数据顺序等受控变量必须一致。
    该检查把“公平对比”从口头约定变成可执行断言：任何 config_used.json 中
    除 variant/seed/device/tag 外的字段不一致都会报错并列出差异。
    """
    ref = None
    ref_name = None
    mismatches = []
    missing = []
    checked = 0
    for v in variants:
        for seed in seeds:
            cfg_path = os.path.join(_variant_dir(out_root, v, seed, n_seeds),
                                    "config_used.json")
            if not os.path.exists(cfg_path):
                missing.append(f"{v}/seed{seed}")
                continue
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
    if missing:
        print(f"[audit] 警告：缺少 {len(missing)} 个 run 的 config_used.json，"
              f"无法核对受控变量: {', '.join(missing[:6])}"
              + (" ..." if len(missing) > 6 else ""))
    if mismatches:
        msg = ("受控变量自检失败：对比实验除待验证模块外必须完全一致，"
               "否则观测差异不能归因于方法本身。\n  - " + "\n  - ".join(mismatches))
        print(f"[audit] {msg}")
        raise SystemExit(2)
    if checked:
        print(f"[audit] 受控变量自检通过：{checked} 个 run 的优化器/LR/步数等配置一致")
    return {"ok": not mismatches, "checked": checked,
            "missing_configs": missing, "mismatches": mismatches,
            "exempt_keys": sorted(_AUDIT_EXEMPT_KEYS)}


# ---------------------------------------------------------------- 汇总统计

def _seed_key(path: str, data: dict):
    m = _SEED_RE.search(os.path.splitext(os.path.basename(path))[0])
    if m:
        return int(m.group(1))
    s = data.get("seed")
    return int(s) if s is not None else None


def _sorted_seeds(mapping) -> list:
    return sorted(mapping.keys(),
                  key=lambda s: (s is None, s if s is not None else -1))


def _variant_seed_map(runs: dict, variant: str) -> dict:
    """取某变体的 {seed: slot}；若已有带种子的结果，忽略无种子的旧格式文件，
    避免把同一次实验重复计为一个额外种子。"""
    m = (runs or {}).get(variant) or {}
    tagged = {s: slot for s, slot in m.items() if s is not None}
    return tagged if tagged else m


def _stat(values) -> dict:
    """跨种子聚合：{mean, std, n}；无有效值时 None。"""
    vals = [v for v in values if v is not None]
    m, s = mean_std(vals)
    if m is None:
        return None
    return {"mean": m, "std": s, "n": len(vals)}


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
        if task == "_skipped" or not by_ctx:
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


def _paired_comparisons(runs, pairs, metric_name, extractor, lower_better,
                        min_seeds=3) -> list:
    """对每个 (候选, 参照) 在共同种子上做配对 t 检验。"""
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
        if res.get("p") is None:
            res["verdict"] = f"种子不足（n={res['n']}<{min_seeds}），无法判定"
        elif res["n"] < min_seeds:
            res["verdict"] = f"种子不足（n={res['n']}<{min_seeds}），无法判定"
        elif res["p"] < 0.05:
            better = (res["mean_diff"] < 0) if lower_better else (res["mean_diff"] > 0)
            res["verdict"] = "显著优于" if better else "显著劣于"
        else:
            res["verdict"] = "不显著"
        out.append(res)
    return out


def build_summary(results_dir: str, audit: dict = None) -> dict:
    """汇总 PPL / RULER / 延迟 / 参数量为 Markdown + CSV + JSON。

    多种子口径：每个变体的每个指标先按种子聚合为 Mean ± Std；变体间比较使用
    同一批种子上的配对 t 检验（同种子共享初始化与数据顺序，配对消除公共方差）。
    消融矩阵显式列出全因子 8 个组合的覆盖情况，缺失组合会被标注。
    """

    def load(p):
        if not os.path.exists(p):
            return None
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)

    # --- 发现所有 run：（变体, 种子） -> {eval, ruler, analysis}
    runs = {}

    def slot(v, s):
        return runs.setdefault(v, {}).setdefault(
            s, {"eval": None, "ruler": None, "analysis": None})

    for fp in sorted(glob.glob(os.path.join(results_dir, "eval_*.json"))):
        d = load(fp)
        if d and d.get("variant"):
            slot(d["variant"], _seed_key(fp, d))["eval"] = d
    for fp in sorted(glob.glob(os.path.join(results_dir, "ruler_*.json"))):
        d = load(fp)
        if d and d.get("variant"):
            slot(d["variant"], _seed_key(fp, d))["ruler"] = d
    for fp in sorted(glob.glob(os.path.join(results_dir, "analysis_*.json"))):
        if "records" in os.path.basename(fp):
            continue
        d = load(fp)
        if d and d.get("variant"):
            slot(d["variant"], _seed_key(fp, d))["analysis"] = d

    variants = sorted(v for v in runs if v in VARIANTS)
    eval_splits = set()
    for v in variants:
        for s in _variant_seed_map(runs, v).values():
            if s.get("eval"):
                eval_splits.add(s["eval"].get("split", "val"))

    # 汇总延迟时必须使用所有变体共有的同一上下文档，不允许各取自己的最大 ctx。
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
                "misleading_given_conc", "p_hit_given_conc"):
        metric_specs.append(
            (key, (lambda k: lambda m: _stat_series(
                m, "analysis", _clock_extractor(k)))(key)))

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
        # 延迟（取全部变体的最大公共 ctx、batch=1）；延迟不受种子影响，测一次
        e = lat_entries.get(v)
        if e and common_ctx is not None:
            ckey = str(common_ctx)
            row["prefill_ms"] = (e.get("prefill_ms") or {}).get(ckey, {}).get("1")
            row["decode_ms_per_tok"] = (e.get("decode_ms_per_tok") or {}).get(ckey)
            row["decode_prefill_ctx"] = (e.get("decode_prefill_ctx") or {}).get(ckey)
            row["latency_ctx"] = common_ctx
        rows.append(row)

    # 延迟基线：优先取 standard_sdpa，缺失时退回显式 standard。
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
    pairs = list(dict.fromkeys(pairs))
    tests = []
    tests += _paired_comparisons(
        runs, pairs, "PPL", lambda slot: _ppl_of(slot.get("eval")),
        lower_better=True)
    tests += _paired_comparisons(
        runs, pairs, "RULER平均", lambda slot: _ruler_avg_of(slot.get("ruler")),
        lower_better=False)

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
            ("auroc_logit", "AUROC(logit)"), ("auroc_qterm", "AUROC(仅查询)")]

    def fmt_stat_cell(r, key):
        st = {"mean": r.get(key), "std": r.get(f"{key}_std"),
              "n": r.get(f"{key}_n")}
        return _fmt_stat(st)

    def _fmt_scalar(x):
        if x is None:
            return "-"
        if isinstance(x, float):
            return f"{x:.3f}" if abs(x) < 10 else f"{x:,.0f}"
        return str(x)

    md = ["# LEGA 实验汇总", "",
          f"> 自动生成于 {time.strftime('%Y-%m-%d %H:%M:%S')}；"
          f"PPL/RULER 为跨随机种子的 Mean ± Std（变体间同种子配对），"
          f"延迟在相同配置下测得。", ""]
    if rows:
        stat_keys = {"val_ppl", "ruler_avg", "gate_mean", "misleading_rate",
                     "misleading_given_conc", "auroc_logit", "auroc_qterm"} \
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
            "见机制分析。", seed_note, split_note, ""]
        if common_ctx is not None:
            cols_notes.insert(1, f"“相对基线%”以 {base_name or 'standard'} 的 prefill 为基线"
                                 f"（公共 ctx={common_ctx}、batch=1，见 latency.json）；")
        md += [""] + cols_notes
    else:
        md.append("（暂无结果，先运行训练评测）")

    # --- 受控变量自检
    md += ["", "## 对比实验受控变量自检", ""]
    if audit:
        if audit.get("ok") and audit.get("checked"):
            md.append(f"- ✅ 通过：{audit['checked']} 个 run 的优化器、学习率、"
                      f"训练步数、数据顺序等受控设置逐项一致；逐 run 仅 "
                      f"`{'`, `'.join(audit['exempt_keys'])}` 不同。")
        else:
            md.append(f"- ❌ 未通过：{audit.get('mismatches') or audit}")
        if audit.get("missing_configs"):
            md.append(f"- ⚠ 缺少配置文件的 run：{', '.join(audit['missing_configs'][:8])}"
                      + (" …" if len(audit["missing_configs"]) > 8 else ""))
    else:
        md.append("- ⚠ 未执行（直接调用 build_summary 或缺少 out 目录）。")

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
        md.append("✅ 全因子消融完整（8/8 组合全部报告），可支撑“各组件均有用”的结论；"
                  "去一法结果两两一致即可交叉印证。")
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

    # --- 配对检验表
    md += ["", "## 关键对比（同种子配对 t 检验）", ""]
    if tests:
        md.append("| 对比 | 指标 | Δ均值 ± Δ标准差 | n | p 值 | 结论 |")
        md.append("|---|---|---|---|---|---|")
        for t in tests:
            if t.get("mean_diff") is None:
                continue
            std = t.get("std_diff")
            delta = (f"{t['mean_diff']:+.4f}"
                     + (f" ± {std:.4f}" if std is not None else ""))
            p = f"{t['p']:.4f}" if t.get("p") is not None else "-"
            md.append(f"| {t['candidate']} vs {t['reference']} | {t['metric']} | "
                      f"{delta} | {t['n']} | {p} | {t.get('verdict', '-')} |")
        md.append("")
        md.append("注: Δ = 候选 − 参照；PPL 越小越好、RULER 越大越好。"
                  "p 为双侧配对 t 检验，未做多重比较校正，正负号以 Δ 为准；"
                  "n<3 不判定显著性。种子不足时请先补齐种子再下结论。")
    else:
        md.append("（无可配对的对比；请对多个变体运行同一组种子）")

    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))

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
        json.dump({"pairs": tests, "note": "同种子配对 t 检验；未做多重比较校正"},
                  f, ensure_ascii=False, indent=2)
    print(f"[summary] 汇总 {len(rows)} 个变体 -> {results_dir}/summary.md "
          f"（配对检验 {len(tests)} 组）")
    return {"rows": rows, "markdown": "\n".join(md), "tests": tests}


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
    p.add_argument("--skip_ruler", action="store_true")
    p.add_argument("--skip_latency", action="store_true")
    p.add_argument("--skip_analysis", action="store_true")
    p.add_argument("--skip_plots", action="store_true")
    args = p.parse_args()

    args.config = _repo_path(args.config)
    args.data_dir = _repo_path(args.data_dir)
    py = sys.executable
    tag = os.path.splitext(os.path.basename(args.config))[0] + ("_smoke" if args.smoke else "")
    out_root = os.path.join(_repo_path(args.out_root), tag)
    results_dir = _repo_path(args.results_dir) if args.results_dir else os.path.join(ROOT, "results", tag)
    os.makedirs(results_dir, exist_ok=True)
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
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

    ruler_over = []
    analysis_over = []
    if args.smoke:
        # 冒烟：少量样例，ctx 取配置 block_size（离线字符分词下 needle+问题
        # 模板约 211 token，128 会让 RULER 全部任务与机制分析被跳过）
        smoke_ctx = min(256, int(cfg.get("block_size", 256)))
        ruler_over = ["--n_samples", "2", "--ctx_lens", str(smoke_ctx)]
        analysis_over = ["--n_examples", "2", "--ctx", str(smoke_ctx)]

    for v in variants:
        for seed in seeds:
            vdir = _variant_dir(out_root, v, seed, n_seeds)
            suffix = _seed_tag(seed, n_seeds)
            if not args.skip_train:
                run([py, "-m", "lega.train", "--config", args.config,
                     "--variant", v, "--data_dir", args.data_dir,
                     "--out_dir", vdir, "--seed", str(seed),
                     "--device", args.device] + (["--smoke"] if args.smoke else [])
                    + (["--wandb"] if args.wandb else []))
            # 优先评测验证集选出的最优 checkpoint（与 model selection 口径一致），
            # 仅在不存在（如 --skip_train 复用旧产物）时回退到最终步模型。
            best = os.path.join(vdir, "model_best.pt")
            ckpt = best if os.path.exists(best) else os.path.join(vdir, "model.pt")
            if not os.path.exists(ckpt):
                print(f"[run_all] 跳过 {v}/seed{seed}：无 checkpoint")
                continue
            run([py, "-m", "lega.evaluate", "--ckpt", ckpt,
                 "--data_dir", args.data_dir,
                 "--out", os.path.join(results_dir, f"eval_{v}{suffix}.json"),
                 "--device", args.device])
            if not args.skip_ruler:
                run([py, "-m", "lega.ruler", "--ckpt", ckpt,
                     "--data_dir", args.data_dir,
                     "--out", os.path.join(results_dir, f"ruler_{v}{suffix}.json"),
                     "--device", args.device] + ruler_over)

    if not args.skip_latency:
        # 显式给出延迟基准的变体清单：lega.latency 的默认值不含消融变体，
        # 不传会让汇总表的消融变体延迟列恒为 "-"。多种子时延迟按各变体首个
        # 种子的 checkpoint 测量（延迟由架构决定，与训练种子无关）。
        lat_variants = ",".join(dict.fromkeys(["standard", "standard_sdpa"] + variants))
        run([py, "-m", "lega.latency", "--config", args.config,
             "--data_dir", args.data_dir, "--ckpt_dir", out_root,
             "--out", os.path.join(results_dir, "latency.json"),
             "--variants", lat_variants,
             "--device", args.device] + (["--smoke"] if args.smoke else []))

    if not args.skip_analysis:
        for v in variants:
            if v not in GATED_FOR_ANALYSIS:
                continue
            for seed in seeds:
                vdir = _variant_dir(out_root, v, seed, n_seeds)
                best = os.path.join(vdir, "model_best.pt")
                ckpt = best if os.path.exists(best) else os.path.join(vdir, "model.pt")
                if not os.path.exists(ckpt):
                    continue
                run([py, "-m", "lega.analysis", "--ckpt", ckpt,
                     "--data_dir", args.data_dir,
                     "--out_dir", results_dir, "--device", args.device,
                     "--suffix", _seed_tag(seed, n_seeds)] + analysis_over)

    if not args.skip_plots:
        run([py, "-m", "lega.plots", "--results_dir", results_dir])

    audit = verify_controlled_setup(out_root, variants, seeds, n_seeds)
    build_summary(results_dir, audit=audit)
    print(f"\n[run_all] 全部完成；结果目录: {results_dir}")


if __name__ == "__main__":
    main()
