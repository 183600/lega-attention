"""结果可视化：读取 results/<tag>/ 下的 JSON/CSV，输出中文标注 PNG 图表。

自动检测存在的结果文件，缺失的图跳过并提示（不阻塞流水线）。
用法:
  python -m lega.plots --results_dir results/tiny_smoke
"""
import argparse
import glob
import json
import os
import re

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.font_manager as fm
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    # 画图依赖缺失时不应炸掉整条流水线（design: 缺失的图跳过并提示）
    plt = fm = None
    _HAS_MPL = False
import numpy as np

from .attention import VARIANT_CN

# 中文字体：自动发现 Noto Sans SC / 霞鹜文楷 / 文泉驿等 CJK 字体；找不到则退回默认
import glob as _glob

_CJK_CANDIDATES = (
    _glob.glob("/usr/share/fonts/**/NotoSansSC*.ttf", recursive=True)
    + _glob.glob("/usr/share/fonts/**/NotoSansCJK*.ttc", recursive=True)
    + _glob.glob("/usr/share/fonts/**/LXGWWenKai*.ttf", recursive=True)
    + _glob.glob("/usr/share/fonts/**/Sarasa*SC*.ttf", recursive=True)
    + _glob.glob("/usr/share/fonts/**/wqy*.tt[cf]", recursive=True)
)
_CJK_FAMILY = None
if _HAS_MPL:
    for fp in _CJK_CANDIDATES:
        try:
            fm.fontManager.addfont(fp)
            _CJK_FAMILY = fm.FontProperties(fname=fp).get_name()
            break
        except Exception:
            continue
    if _CJK_FAMILY:
        plt.rcParams["font.sans-serif"] = [_CJK_FAMILY, "DejaVu Sans"]
    else:
        # 找不到 CJK 字体时图表照常生成，但中文标题/坐标轴会渲染为方框；
        # 与“缺失的图跳过并提示”的设计一致，这里显式提示一次，避免静默产出乱码图。
        print("[plots] 未找到 CJK 字体（NotoSansSC/WQY 等），图表中文将显示为方框；"
              "如需中文图表请安装 fonts-noto-cjk")
    plt.rcParams["axes.unicode_minus"] = False

COLORS = {
    "standard": "#7f8c8d", "standard_sdpa": "#bdc3c7", "qgate": "#e67e22",
    "null": "#8e44ad", "lega": "#c0392b", "lega_no_len": "#e74c3c",
    "lega_no_conc": "#16a085", "lega_no_query": "#2980b9",
    "lega_conc_only": "#27ae60", "lega_len_only": "#8e6e00",
}


def _label(v):
    return VARIANT_CN.get(v, v)


def _load(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


_SEED_SUFFIX_RE = re.compile(r"_seed(\d+)$")


def _prefer_seeded(paths):
    """存在 _seed<s> 结果文件时忽略无种子的旧文件：否则同一次实验会被多算一次。"""
    seeded = [p for p in paths
              if _SEED_SUFFIX_RE.search(os.path.splitext(os.path.basename(p))[0])]
    return seeded if seeded else paths


def _group_by_variant(pattern):
    """按变体分组 glob 到的结果文件（自动合并多种子 _seed<seed> 文件）。

    种子识别与 run_all 的汇总口径一致：文件名 _seed<s> 或 JSON 内的 seed 字段；
    若某变体已有带种子的结果，则忽略完全无种子信息（None）的旧文件，避免同一次
    实验被重复计为一个额外种子。
    """
    groups = {}
    for fp in sorted(glob.glob(pattern)):
        d = _load(fp)
        if not (d and d.get("variant")):
            continue
        m = _SEED_SUFFIX_RE.search(os.path.splitext(os.path.basename(fp))[0])
        seed = int(m.group(1)) if m else d.get("seed")
        groups.setdefault(d["variant"], []).append((seed, d))
    return {v: ([d for s, d in items if s is not None]
                if any(s is not None for s, _ in items)
                else [d for _, d in items])
            for v, items in groups.items()}


def fig_ppl(rdir):
    groups = _group_by_variant(os.path.join(rdir, "eval_*.json"))
    if not groups:
        return False
    rows = []
    splits = set()
    for v, ds in groups.items():
        # 多种子：每变体取 PPL 均值，误差棒为样本标准差；单种子不画误差棒。
        ppls = [d.get("ppl", d.get("val_ppl")) for d in ds]
        ppls = [p for p in ppls if p is not None]
        if not ppls:
            continue
        mean = float(np.mean(ppls))
        std = float(np.std(ppls, ddof=1)) if len(ppls) > 1 else 0.0
        rows.append((v, mean, std, any(d.get("val_ppl_clamped", False) for d in ds)))
        splits.add(ds[0].get("split", "val"))
    if not rows:
        return False
    rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(7, 4.2), constrained_layout=True)
    names = [_label(v) for v, _, _, _ in rows]
    vals = [p for _, p, _, _ in rows]
    errs = [s for _, _, s, _ in rows]
    # 发散运行的 PPL 被截断到 exp(20)，以斜线纹理区分，避免被当作正常数值解读
    bars = ax.bar(names, vals, yerr=errs if any(errs) else None, capsize=3,
                  color=[COLORS.get(v, "#34495e") for v, _, _, _ in rows])
    for bar, (_, _, _, clamped) in zip(bars, rows):
        if clamped:
            bar.set_hatch("//")
    ax.bar_label(bars, labels=[("≥ %.2e (发散截断)" % p) if c
                               else ("%.2f±%.2f" % (p, s) if s else "%.2f" % p)
                               for (_, p, s, c) in rows], fontsize=9)
    if splits == {"test"}:
        set_name = "测试集"
    elif len(splits) == 1:
        set_name = "验证集"
    else:
        set_name = "各评测集"   # 混合 split 时不冒充单一集合
    ax.set_ylabel(f"{set_name}困惑度 (PPL, 越低越好)")
    ax.set_title("语言建模困惑度对比（相同配置与步数；误差棒=跨 seed 标准差）")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(os.path.join(rdir, "fig_ppl.png"), dpi=150)
    plt.close(fig)
    return True


def fig_ruler(rdir):
    groups = _group_by_variant(os.path.join(rdir, "ruler_*.json"))
    if not groups:
        return False
    # 多种子：同一 (task, ctx) 的指标跨 seed 先求均值再画，避免只显示最后一个种子
    data = {}
    for v, ds in groups.items():
        agg = {}
        for d in ds:
            for task, by_ctx in (d.get("results") or {}).items():
                if task == "_skipped" or not isinstance(by_ctx, dict):
                    continue
                for ctx, m in by_ctx.items():
                    if not isinstance(m, dict):
                        continue
                    cell = agg.setdefault(task, {}).setdefault(
                        ctx, {"em": [], "recall": []})
                    if m.get("em") is not None:
                        cell["em"].append(m["em"])
                    if m.get("recall") is not None:
                        cell["recall"].append(m["recall"])
        data[v] = {
            task: {ctx: {
                "em": (float(np.mean(cell["em"])) if cell["em"] else None),
                "recall": (float(np.mean(cell["recall"]))
                           if cell["recall"] else None)}
                for ctx, cell in by_ctx.items()}
            for task, by_ctx in agg.items()}
    tasks = ["niah_synth", "niah_corpus", "niah_multikey", "fwe", "vt"]
    avail = [t for t in tasks if any(t in r and r[t] for r in data.values())]
    if not avail:
        return False
    fig, axes = plt.subplots(1, len(avail), figsize=(4.2 * len(avail), 4.0),
                             constrained_layout=True, squeeze=False)
    axes = axes[0]
    for ax, task in zip(axes, avail):
        for v, r in sorted(data.items()):
            if task not in r or not r[task]:
                continue
            items = sorted(((int(k), m) for k, m in r[task].items()))
            xs = [x for x, _ in items]
            # fwe 主指标是 top-10 recall；其余检索任务为 EM。
            metric = "recall" if task == "fwe" else "em"
            ys = [m[metric] * 100 for _, m in items]
            ax.plot(xs, ys, marker="o", label=_label(v), color=COLORS.get(v))
        ax.set_title(task)
        ax.set_xlabel("上下文长度 (tokens)")
        ax.set_ylabel("Top-10 Recall (%)" if task == "fwe" else "EM (%)")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, bbox_to_anchor=(0.02, 0.98), loc="upper left")
    fig.suptitle("RULER 长上下文检索表现（贪心解码）", y=1.04)
    fig.savefig(os.path.join(rdir, "fig_ruler_em.png"), dpi=150)
    plt.close(fig)
    return True


def fig_latency(rdir):
    d = _load(os.path.join(rdir, "latency.json"))
    if not d:
        return False
    variants = d["variants"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    prefill_bs = None
    for v, e in variants.items():
        ctxs = sorted(int(c) for c in e["prefill_ms"].keys())
        if not ctxs:
            continue
        bs_key = "1" if "1" in e["prefill_ms"][str(ctxs[0])] else str(list(e["prefill_ms"][str(ctxs[0])].keys())[0])
        prefill_bs = prefill_bs or bs_key
        ax1.plot(ctxs, [e["prefill_ms"][str(c)][bs_key] for c in ctxs],
                 marker="o", label=_label(v) + ("(SDPA)" if e.get("use_sdpa") and v == "standard_sdpa" else ""),
                 color=COLORS.get(v))
    ax1.set_xlabel("上下文长度 (tokens)")
    ax1.set_ylabel(f"预填充耗时 (ms, batch={prefill_bs})")
    ax1.set_title("Prefill 延迟")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8)
    for v, e in variants.items():
        ctx_keys = [str(c) for c in sorted(int(c) for c in e["decode_ms_per_tok"].keys())]
        if not ctx_keys:
            continue
        actual = e.get("decode_prefill_ctx") or {}
        xs = [int(actual.get(c, c)) for c in ctx_keys]
        ax2.plot(xs, [e["decode_ms_per_tok"][c] for c in ctx_keys],
                 marker="s", label=_label(v), color=COLORS.get(v))
    ax2.set_xlabel("decode 前的实际 KV 上下文长度 (tokens)")
    ax2.set_ylabel("每 token 解码耗时 (ms)")
    ax2.set_title("Decode 延迟（KV 缓存，batch=1）")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8)
    fig.suptitle("实际推理延迟对比", y=1.02)
    fig.savefig(os.path.join(rdir, "fig_latency.png"), dpi=150)
    plt.close(fig)
    return True


def _read_records(rdir, variant):
    import csv
    # 兼容单种子 analysis_records_<v>.csv 与多种子 analysis_records_<v>_seed<s>.csv，
    # 多种子记录合并后画图（分布图上的种子差异会自然体现为更宽的分布）。
    paths = _prefer_seeded(
        [os.path.join(rdir, f"analysis_records_{variant}.csv")]
        + sorted(glob.glob(os.path.join(
            rdir, f"analysis_records_{variant}_seed*.csv"))))
    rows = []
    for path in paths:
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows.append({k: (float(v) if v not in ("", "None", "nan") else None)
                             for k, v in r.items() if k not in ("task",)})
    return rows


def fig_gate_hist(rdir, variant="lega"):
    rows = _read_records(rdir, variant)
    if not rows:
        return False
    g_hit = [r["g"] for r in rows if r.get("g") is not None and r.get("hit") == 1.0]
    g_miss = [r["g"] for r in rows if r.get("g") is not None and r.get("hit") == 0.0]
    if not g_hit and not g_miss:
        return False
    fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    bins = np.linspace(0, 1, 41)
    if g_miss:
        ax.hist(g_miss, bins=bins, alpha=0.6, label=f"注意力没找对 (n={len(g_miss)})",
                color="#7f8c8d", density=True)
    if g_hit:
        ax.hist(g_hit, bins=bins, alpha=0.6, label=f"注意力找对了 (n={len(g_hit)})",
                color="#c0392b", density=True)
    ax.set_xlabel(f"门控值 g（{VARIANT_CN.get(variant, variant)}）")
    ax.set_ylabel("密度")
    ax.set_title("门控值分布：找对 vs 没找对（问题区查询位）")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(os.path.join(rdir, "fig_gate_hist.png"), dpi=150)
    plt.close(fig)
    return True


def fig_calibration(rdir, variant="lega"):
    rows = [r for r in _read_records(rdir, variant) if r.get("g") is not None and r.get("c") is not None]
    if len(rows) < 20:
        return False
    c = np.array([r["c"] for r in rows])
    g = np.array([r["g"] for r in rows])
    ln = np.array([r["log_n"] for r in rows])
    med = np.median(ln)
    fig, ax = plt.subplots(figsize=(6.4, 4.4), constrained_layout=True)
    for mask, name, color in ((ln <= med, f"短上下文 log n≤{med:.1f}", "#2980b9"),
                              (ln > med, f"长上下文 log n>{med:.1f}", "#c0392b")):
        if mask.sum() < 5:
            continue
        cc, gg = c[mask], g[mask]
        qs = np.quantile(cc, np.linspace(0, 1, 11))
        mids, means = [], []
        for qi, (a, b) in enumerate(zip(qs[:-1], qs[1:])):
            if qi == len(qs) - 2:
                m = (cc >= a) & (cc <= b)
            else:
                m = (cc >= a) & (cc < b)
            if m.sum() >= 2:
                mids.append(float(cc[m].mean()))
                means.append(float(gg[m].mean()))
        ax.plot(mids, means, marker="o", label=name, color=color)
    ax.set_xlabel("证据集中度 c = log(n · max_j p_j)")
    ax.set_ylabel("平均门控值 g")
    ax.set_title("门控校准曲线：c 越大（证据越集中）门开得越大吗？")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(os.path.join(rdir, "fig_calibration.png"), dpi=150)
    plt.close(fig)
    return True


def fig_misleading(rdir):
    """各变体：集中时找对率 vs 集中但看错率 —— “集中≠找对”的直接证据。"""
    keys = ("p_hit_given_conc", "p_hit_given_notconc", "misleading_rate")
    groups = {}
    files = [fp for fp in sorted(glob.glob(os.path.join(rdir, "analysis_*.json")))
             if "records" not in os.path.basename(fp)]
    for fp in _prefer_seeded(files):
        d = _load(fp)
        if d and d.get("n_records"):
            groups.setdefault(d["variant"], []).append(d)
    rows = []
    for v, ds in groups.items():
        row = {"variant": v}
        for k in keys:
            vals = [d[k] for d in ds if d.get(k) is not None]
            row[k] = float(np.mean(vals)) if vals else None
        rows.append(row)
    if not rows:
        return False
    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    x = np.arange(len(rows))
    w = 0.27
    vals = {
        "P(hit|集中)": [r.get("p_hit_given_conc") for r in rows],
        "P(hit|不集中)": [r.get("p_hit_given_notconc") for r in rows],
        "集中但看错率": [r.get("misleading_rate") for r in rows],
    }
    for i, (name, vs) in enumerate(vals.items()):
        # None 表示该指标无数据（如该变体没有集中查询位），画成 0 会造成虚假结论，
        # 用 nan 让 matplotlib 留空
        ax.bar(x + (i - 1) * w, [np.nan if v is None else v for v in vs], w, label=name,
               color=["#c0392b", "#2980b9", "#e67e22"][i])
    ax.set_xticks(x)
    ax.set_xticklabels([_label(r["variant"]) for r in rows], rotation=15, fontsize=9)
    ax.set_ylabel("比率")
    ax.set_title("注意力很集中，就代表找对信息了吗？（问题区查询位统计）")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(os.path.join(rdir, "fig_misleading.png"), dpi=150)
    plt.close(fig)
    return True


def fig_auroc_heads(rdir, variant="lega"):
    files = _prefer_seeded(
        [os.path.join(rdir, f"analysis_{variant}.json")]
        + sorted(glob.glob(os.path.join(
            rdir, f"analysis_{variant}_seed*.json"))))
    # 多种子：逐 (层, 头) 的 AUROC 跨 seed 求均值后再画
    agg = {}
    for fp in files:
        d = _load(fp)
        if not d:
            continue
        for h in d.get("per_head") or []:
            key = (h["layer"], h["head"])
            cell = agg.setdefault(key, {"qterm": [], "logit": []})
            if h.get("auroc_qterm") is not None:
                cell["qterm"].append(h["auroc_qterm"])
            if h.get("auroc_logit") is not None:
                cell["logit"].append(h["auroc_logit"])
    ph = [{"layer": k[0], "head": k[1],
           "auroc_qterm": float(np.mean(v["qterm"])),
           "auroc_logit": float(np.mean(v["logit"]))}
          for k, v in sorted(agg.items()) if v["qterm"] and v["logit"]]
    if not ph:
        return False
    fig, ax = plt.subplots(figsize=(5.6, 5.0), constrained_layout=True)
    xs = [h["auroc_qterm"] for h in ph]
    ys = [h["auroc_logit"] for h in ph]
    ax.scatter(xs, ys, c=[COLORS.get(variant, "#c0392b")], s=42, alpha=0.85)
    lim = [max(0.35, min(xs + ys) - 0.05), min(1.0, max(xs + ys) + 0.05)]
    ax.plot(lim, lim, "--", color="#95a5a6", lw=1, label="对角线：证据统计无增量")
    ax.set_xlabel("仅查询项 w·q 的 AUROC")
    ax.set_ylabel("完整门控 logit 的 AUROC")
    ax.set_title("逐头对比：加入 (c, log n) 后能否更好区分“找对/没找对”")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.savefig(os.path.join(rdir, "fig_auroc_heads.png"), dpi=150)
    plt.close(fig)
    return True


def main():
    p = argparse.ArgumentParser(description="结果画图")
    p.add_argument("--results_dir", required=True)
    args = p.parse_args()
    rdir = args.results_dir
    os.makedirs(rdir, exist_ok=True)

    if not _HAS_MPL:
        print("[plots] 未安装 matplotlib，跳过全部图表生成（不阻塞流水线）")
        return

    made = []
    for name, fn, kw in [
        ("fig_ppl.png", fig_ppl, ()),
        ("fig_ruler_em.png", fig_ruler, ()),
        ("fig_latency.png", fig_latency, ()),
        ("fig_gate_hist.png", fig_gate_hist, ("lega",)),
        ("fig_calibration.png", fig_calibration, ("lega",)),
        ("fig_misleading.png", fig_misleading, ()),
        ("fig_auroc_heads.png", fig_auroc_heads, ("lega",)),
    ]:
        try:
            ok = fn(rdir, *kw)
        except Exception as e:
            print(f"[plots] {name} 生成失败: {e}")
            continue
        if ok:
            made.append(name)
            print(f"[plots] 已生成 {name}")
        else:
            print(f"[plots] {name} 跳过（结果文件不足）")
    print(f"[plots] 完成：{len(made)} 张图 -> {rdir}")


if __name__ == "__main__":
    main()
