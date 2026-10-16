#!/usr/bin/env python3
"""一键实验流水线：训练全部变体 → PPL → RULER → 延迟 → 机制分析 → 图表 → 汇总表。

变体（相同参数量级与训练步数）:
  standard | qgate | null | lega | lega_no_len | lega_no_conc | lega_no_query

用法:
  # 冒烟测试（tiny 配置 + 伪数据，几分钟内跑通全流程）
  python run_all.py --config configs/tiny.json --data_dir data/dummy --smoke

  # 正式实验（small 配置 + WikiText-103）
  python scripts/prepare_data.py --out_dir data/wikitext103
  python run_all.py --config configs/small.json --data_dir data/wikitext103
"""
import argparse
import csv
import glob
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from lega.attention import VARIANTS, VARIANT_CN  # noqa: E402

ALL_VARIANTS = list(VARIANTS)
# 机制分析不依赖输出门也可计算“集中但看错率/P(hit|集中）”：
# null 变体虽无 g/logit，仍应纳入；standard 无需诊断，保持跳过。
GATED_FOR_ANALYSIS = ("lega", "qgate", "null", "lega_no_len", "lega_no_conc", "lega_no_query")


def run(cmd, env=None):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT, env=env)


def _repo_path(path: str) -> str:
    """相对路径统一解释为仓库根目录；父进程与子进程必须看到同一位置。"""
    return path if os.path.isabs(path) else os.path.abspath(os.path.join(ROOT, path))


def build_summary(results_dir: str) -> dict:
    """汇总 PPL / RULER / 延迟 / 参数量为 Markdown + CSV + JSON。"""

    def load(p):
        if not os.path.exists(p):
            return None
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)

    variants = []
    eval_splits = set()
    for fp in sorted(glob.glob(os.path.join(results_dir, "eval_*.json"))):
        d = load(fp)
        if d:
            variants.append(d["variant"])
            eval_splits.add(d.get("split", "val"))

    # 汇总延迟时必须使用所有变体共有的同一上下文档，不允许各取自己的最大 ctx。
    lat = load(os.path.join(results_dir, "latency.json")) or {}
    lat_entries = {v: (lat.get("variants") or {}).get(v) for v in variants}
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

    rows = []
    for v in variants:
        ev = load(os.path.join(results_dir, f"eval_{v}.json")) or {}
        ru = load(os.path.join(results_dir, f"ruler_{v}.json")) or {}
        an = load(os.path.join(results_dir, f"analysis_{v}.json")) or {}
        row = {
            "variant": v,
            "variant_cn": VARIANT_CN.get(v, v),
            "params": ev.get("n_params"),
            "gate_extra": ev.get("n_gate_extra", 0),
            # PPL 被钳制在 exp(20) 时必须显式标注：发散的 run 不能与正常 PPL
            # 在同一列表头中静默比较。
            "val_ppl": (">4.85e8 (已截断)" if ev.get("val_ppl_clamped")
                        else ev.get("val_ppl")),
            "gate_mean": ev.get("gate_mean"),
        }
        # RULER 各任务先取所有上下文档平均
        ems = []
        for task, by_ctx in (ru.get("results") or {}).items():
            if task == "_skipped" or not by_ctx:
                continue
            # fwe 的任务主指标是 top-10 recall；严格 EM 对长词表提取任务过严，
            # 不能和直接检索任务的数值 EM 混算。
            metric = "recall" if task == "fwe" else "em"
            vals = [m[metric] for m in by_ctx.values() if m.get(metric) is not None]
            if vals:
                row[f"ruler_{task}"] = sum(vals) / len(vals)
                ems.append(row[f"ruler_{task}"])
        # 先在每个任务内对上下文档求平均，再对任务求宏平均：避免某任务因
        # block_size/生成余量钳制少跑一档而被隐式降权。
        row["ruler_avg"] = sum(ems) / len(ems) if ems else None
        # 延迟（取全部变体的最大公共 ctx、batch=1）
        e = lat_entries.get(v)
        if e and common_ctx is not None:
            ckey = str(common_ctx)
            row["prefill_ms"] = (e.get("prefill_ms") or {}).get(ckey, {}).get("1")
            row["decode_ms_per_tok"] = (e.get("decode_ms_per_tok") or {}).get(ckey)
            row["decode_prefill_ctx"] = (e.get("decode_prefill_ctx") or {}).get(ckey)
            row["latency_ctx"] = common_ctx
        # 机制分析关键指标
        for k, key in [("auroc_logit", "auroc_logit"), ("auroc_qterm", "auroc_qterm"),
                       ("misleading_rate", "misleading_rate"),
                       ("misleading_given_conc", "misleading_given_conc"),
                       ("p_hit_given_conc", "p_hit_given_conc")]:
            row[k] = an.get(key)
        rows.append(row)

    # 延迟基线：优先取 standard_sdpa，缺失时退回显式 standard。
    # standard_sdpa 不单独训练/评测、不会出现在 eval_*.json 里，
    # 因此基线必须从 latency.json 的条目直取而非从 rows 里找。
    lat_all = lat.get("variants") or {}

    def _prefill_at_common(variant_name):
        e = lat_all.get(variant_name)
        if not e or common_ctx is None:
            return None
        return (e.get("prefill_ms") or {}).get(str(common_ctx), {}).get("1")

    base_ms, base_name = None, None
    for cand in ("standard_sdpa", "standard"):
        e = lat_all.get(cand)
        if e and _prefill_at_common(cand) is not None:
            base_ms, base_name = _prefill_at_common(cand), cand
            break
    for r in rows:
        if base_ms is not None and base_ms > 0 \
                and r.get("prefill_ms") is not None and r["prefill_ms"] > 0 \
                and r["variant"] not in ("standard", "standard_sdpa"):
            r["prefill_rel_pct"] = round(
                (r["prefill_ms"] - base_ms) / base_ms * 100, 2)

    # Markdown
    cols = [("variant_cn", "变体"), ("params", "参数量"), ("gate_extra", "门控额外参数"),
            ("val_ppl", "PPL↓"), ("ruler_avg", "RULER平均↑"),
            ("ruler_niah_synth", "niah_synth↑"), ("ruler_niah_corpus", "niah_corpus↑"),
            ("ruler_niah_multikey", "multikey↑"), ("ruler_fwe", "fwe↑"), ("ruler_vt", "vt↑"),
            ("prefill_ms", "Prefill ms↓"), ("prefill_rel_pct", "相对基线%"),
            ("decode_ms_per_tok", "Decode ms/tok↓"), ("gate_mean", "门控均值"),
            ("misleading_rate", "集中但看错率"), ("misleading_given_conc", "集中后看错率"),
            ("auroc_logit", "AUROC(logit)"),
            ("auroc_qterm", "AUROC(仅查询)")]

    def fmt(x):
        if x is None:
            return "-"
        if isinstance(x, float):
            return f"{x:.3f}" if abs(x) < 10 else f"{x:,.0f}"
        return str(x)

    md = ["# LEGA 实验汇总", "",
          f"> 自动生成于 {time.strftime('%Y-%m-%d %H:%M:%S')}；PPL/RULER/延迟均在相同配置与步数下测得。",
          ""]
    if rows:
        header = "| " + " | ".join(c for _, c in cols) + " |"
        sep = "|" + "---|" * len(cols)
        md += [header, sep]
        for r in rows:
            md.append("| " + " | ".join(fmt(r.get(k)) for k, _ in cols) + " |")
        split_note = ("PPL 在 held-out test 集上评测（checkpoint 选择在 val 上完成）"
                      if eval_splits == {"test"} else
                      "⚠ PPL 在 val 集上评测，该集同时用于 best-checkpoint 选择，"
                      "数值偏乐观；请用 prepare_data.py 生成 held-out test 后重跑评测")
        cols_notes = ["注: niah/fwe/vt 列为各上下文档的 EM/召回平均，每格样本数 n 见 ruler_*.json"
                      "（各变体使用相同 seed 的同一批样本，差异为配对比较）；",
                      "“集中但看错率”= P(max_p>τ 且注意力落点不在答案数值跨度 value_span 内)，见机制分析。",
                      split_note, ""]
        if common_ctx is not None:
            cols_notes.insert(1, f"“相对基线%”以 {base_name or 'standard'} 的 prefill 为基线"
                                 f"（公共 ctx={common_ctx}、batch=1，见 latency.json）；")
        md += [""] + cols_notes
    else:
        md.append("（暂无结果，先运行训练评测）")
    with open(os.path.join(results_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    # CSV
    if rows:
        all_keys = sorted({k for r in rows for k in r})
        with open(os.path.join(results_dir, "summary.csv"), "w", newline="",
                  encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=all_keys)
            w.writeheader()
            w.writerows(rows)
    with open(os.path.join(results_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"[summary] 汇总 {len(rows)} 个变体 -> {results_dir}/summary.md")
    return {"rows": rows, "markdown": "\n".join(md)}


def main():
    p = argparse.ArgumentParser(description="一键实验流水线")
    p.add_argument("--config", default="configs/tiny.json")
    p.add_argument("--data_dir", default="data/dummy")
    p.add_argument("--out_root", default="out")
    p.add_argument("--results_dir", default=None, help="默认 results/<tag>")
    p.add_argument("--variants", default=",".join(ALL_VARIANTS))
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

    ruler_over = []
    analysis_over = []
    if args.smoke:
        # 冒烟：少量样例，ctx 取配置 block_size（离线字符分词下 needle+问题
        # 模板约 211 token，128 会让 RULER 全部任务与机制分析被跳过）
        with open(args.config, "r", encoding="utf-8") as cf:
            smoke_ctx = min(256, int(json.load(cf).get("block_size", 256)))
        ruler_over = ["--n_samples", "2", "--ctx_lens", str(smoke_ctx)]
        analysis_over = ["--n_examples", "2", "--ctx", str(smoke_ctx)]

    for v in variants:
        if not args.skip_train:
            run([py, "-m", "lega.train", "--config", args.config, "--variant", v,
                 "--data_dir", args.data_dir, "--out_dir", os.path.join(out_root, v),
                 "--device", args.device] + (["--smoke"] if args.smoke else [])
                + (["--wandb"] if args.wandb else []))
        # 优先评测验证集选出的最优 checkpoint（与 model selection 口径一致），
        # 仅在不存在（如 --skip_train 复用旧产物）时回退到最终步模型。
        # 必须在训练之后再选择：全新训练前 best 尚不存在，提前选择会静默把
        # 下游 PPL/RULER 评测锁定到最终步 model.pt，绕过 val 上的模型选择。
        best = os.path.join(out_root, v, "model_best.pt")
        ckpt = best if os.path.exists(best) else os.path.join(out_root, v, "model.pt")
        if os.path.exists(ckpt):
            run([py, "-m", "lega.evaluate", "--ckpt", ckpt, "--data_dir", args.data_dir,
                 "--out", os.path.join(results_dir, f"eval_{v}.json"),
                 "--device", args.device])
            if not args.skip_ruler:
                run([py, "-m", "lega.ruler", "--ckpt", ckpt, "--data_dir", args.data_dir,
                     "--out", os.path.join(results_dir, f"ruler_{v}.json"),
                     "--device", args.device] + ruler_over)

    if not args.skip_latency:
        # 显式给出延迟基准的变体清单：lega.latency 的默认值不含消融变体，
        # 不传会让汇总表的 lega_no_len/no_conc/no_query 延迟列恒为 "-"。
        lat_variants = ",".join(dict.fromkeys(["standard", "standard_sdpa"] + variants))
        run([py, "-m", "lega.latency", "--config", args.config, "--data_dir", args.data_dir,
             "--ckpt_dir", out_root, "--out", os.path.join(results_dir, "latency.json"),
             "--variants", lat_variants,
             "--device", args.device] + (["--smoke"] if args.smoke else []))

    if not args.skip_analysis:
        for v in variants:
            if v not in GATED_FOR_ANALYSIS:
                continue
            best = os.path.join(out_root, v, "model_best.pt")
            ckpt = best if os.path.exists(best) else os.path.join(out_root, v, "model.pt")
            if os.path.exists(ckpt):
                run([py, "-m", "lega.analysis", "--ckpt", ckpt, "--data_dir", args.data_dir,
                     "--out_dir", results_dir, "--device", args.device] + analysis_over)

    if not args.skip_plots:
        run([py, "-m", "lega.plots", "--results_dir", results_dir])

    build_summary(results_dir)
    print(f"\n[run_all] 全部完成；结果目录: {results_dir}")


if __name__ == "__main__":
    main()
