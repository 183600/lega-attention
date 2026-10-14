"""结果可视化：读取 results/<tag>/ 下的 JSON/CSV，输出中文标注 PNG 图表。

自动检测存在的结果文件，缺失的图跳过并提示（不阻塞流水线）。
用法:
  python -m lega.plots --results_dir results/tiny_smoke
"""
import argparse
import glob
import json
import os

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
}


def _label(v):
    return VARIANT_CN.get(v, v)


def _load(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def fig_ppl(rdir):
    rows = []
    for fp in sorted(glob.glob(os.path.join(rdir, "eval_*.json"))):
        d = _load(fp)
        if d:
            rows.append((d["variant"], d["val_ppl"], bool(d.get("val_ppl_clamped", False))))
    if not rows:
        return False
    rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(7, 4.2), constrained_layout=True)
    names = [_label(v) for v, _, _ in rows]
    vals = [p for _, p, _ in rows]
    # 发散运行的 PPL 被截断到 exp(20)，以斜线纹理区分，避免被当作正常数值解读
    bars = ax.bar(names, vals,
                  color=[COLORS.get(v, "#34495e") for v, _, _ in rows])
    for bar, (_, _, clamped) in zip(bars, rows):
        if clamped:
            bar.set_hatch("//")
    ax.bar_label(bars, labels=[("≥ %.2e (发散截断)" % p) if c else "%.2f" % p
                               for (_, p, c) in rows], fontsize=9)
    ax.set_ylabel("验证集困惑度 (PPL, 越低越好)")
    ax.set_title("语言建模困惑度对比（相同参数量级与训练步数）")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(os.path.join(rdir, "fig_ppl.png"), dpi=150)
    plt.close(fig)
    return True


def fig_ruler(rdir):
    files = sorted(glob.glob(os.path.join(rdir, "ruler_*.json")))
    if not files:
        return False
    data = {}
    for fp in files:
        d = _load(fp)
        if d:
            data[d["variant"]] = d["results"]
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


def _read_records(rdir, variant):
    import csv
    path = os.path.join(rdir, f"analysis_records_{variant}.csv")
    if not os.path.exists(path):
        return []
    rows = []
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
    rows = []
    for fp in sorted(glob.glob(os.path.join(rdir, "analysis_*.json"))):
        base = os.path.basename(fp)
        if "records" in base:
            continue
        d = _load(fp)
        if d and d.get("n_records"):
            rows.append(d)
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
    d = _load(os.path.join(rdir, f"analysis_{variant}.json"))
    if not d or not d.get("per_head"):
        return False
    ph = [h for h in d["per_head"]
          if h.get("auroc_qterm") is not None and h.get("auroc_logit") is not None]
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
    ax.set_title(f"逐头对比：加入 (c, log n) 后能否更好区分“找对/没找对”")
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
