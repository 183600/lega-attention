"""结果可视化：读取 results/<tag>/ 下的 JSON/CSV，输出中文标注 PNG 图表。

自动检测存在的结果文件，缺失的图跳过并提示（不阻塞流水线）。
用法:
  python -m lega.plots --results_dir results/tiny_smoke
"""
import argparse
import glob
import json
import math
import os
import re
from collections import OrderedDict

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


def _finite_float(v):
    """JSON 数值安全转换：非数值/非有限值返回 None（坏点跳过而非崩溃）。"""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError, OverflowError):
        return None
    return f if np.isfinite(f) else None


def _load(path):
    """读取 JSON；缺失/不可读/损坏一律返回 None（坏文件不毁整张图）。"""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


_SEED_SUFFIX_RE = re.compile(r"_seed(\d+)$")


def _is_tagged(path):
    return bool(_SEED_SUFFIX_RE.search(os.path.splitext(os.path.basename(path))[0]))


def _expected_tagged(rdir):
    """本次 run 的命名约定：repro.json 的 seeds 多于一个 → 结果带 _seed 后缀。

    未知（无/坏 repro.json）时返回 None，调用方退回“_seed 优先”的旧行为。
    """
    d = _load(os.path.join(rdir, "repro.json")) if rdir else None
    seeds = d.get("seeds") if isinstance(d, dict) else None
    if isinstance(seeds, list):
        # 显式 seeds=[] 表示本次 run 没有种子，不是“未知”；不能退回
        # 扫描旧命名布局。
        return len(seeds) > 1
    return None


def _run_scope(rdir):
    """repro.json 记录的本次 run 变体/种子范围；未知时对应项为 None。

    仅按“带不带 _seed 后缀”筛选还不够：同一 tag 下换过具体种子（如
    seeds 7,8 → 7,9）重跑时，旧 seed8 与本次一样带后缀，会被混入均值。
    必须再按 repro.json 的 seeds/variants 白名单过滤。
    """
    d = _load(os.path.join(rdir, "repro.json")) if rdir else None
    if not isinstance(d, dict):
        return None, None
    variants = d.get("variants")
    variant_scope = None
    if isinstance(variants, list):
        variant_scope = {v for v in variants if isinstance(v, str)}
    seeds = d.get("seeds")
    seed_scope = None
    if isinstance(seeds, list):
        parsed = set()
        for x in seeds:
            # 种子身份必须是整数；1.5 截断成 1 会把不同 run 混为一谈。
            f = _finite_float(x)
            if f is None or f != math.floor(f):
                continue
            parsed.add(int(f))
        # 空列表/全部非法同样是“显式无可用种子”，不能当作未知而放开
        # 白名单（否则旧结果会混入）。
        seed_scope = parsed
    return variant_scope, seed_scope


def _path_seed(path):
    m = _SEED_SUFFIX_RE.search(os.path.splitext(os.path.basename(path))[0])
    return int(m.group(1)) if m else None


def _seed_in_scope(seed, scope):
    if scope is None:
        return True
    if not scope:
        # 本次 run 明确没有任何种子：所有产物都不属于它。
        return False
    if seed is None:
        # 有种子白名单时，缺种子信息的文件仍按旧行为保留（如手工目录），
        # 但显式空白名单已在上面排除。
        return True
    f = _finite_float(seed)
    if f is None or f != math.floor(f):
        # 非整数种子（含外部坏 JSON）不能与整数白名单中的某项相等。
        return False
    return int(f) in scope


def _prefer_seeded(paths, rdir=None):
    """只保留与本次 run 命名约定一致的结果文件：否则同一次实验会被多算一次。

    多种子运行的产物带 _seed<s> 后缀、单种子不带；在旧目录上以不同种子数重跑
    时两种命名的文件可能同时存在，谁属于“本次 run”必须按 repro.json 判定，
    不能永远偏爱 _seed 文件（那会把旧多种子结果混进单种子报告）。无 repro
    信息时退回“_seed 优先”的旧行为。
    """
    expected = _expected_tagged(rdir) if rdir else None
    _, seed_scope = _run_scope(rdir) if rdir else (None, None)
    if seed_scope is not None:
        # 先剔除不属于本次种子白名单的过期产物（含同后缀的旧种子）。
        # 过滤后为空时宁可不出图，也不回退去画别的 run 的结果。
        paths = [p for p in paths
                 if _seed_in_scope(_path_seed(p), seed_scope)]
        if not paths:
            return []
    if expected is not None:
        keep = [p for p in paths
                if _is_tagged(p) == expected and os.path.exists(p)]
        if keep:
            return keep
    seeded = [p for p in paths if _is_tagged(p) and os.path.exists(p)]
    if seeded:
        return seeded
    existing = [p for p in paths if os.path.exists(p)]
    return existing or list(paths)


def _group_items(items, rdir=None):
    """items: [(path, data)] → {variant: [data]}，同一 (变体, 种子) 只留一个。

    种子识别与 run_all 的汇总口径一致（文件名 _seed<s> 或 JSON 的 seed 字段）；
    裸命名文件与 _seed<s> 文件可能指向同一种子（JSON 内有 seed），两者都留会
    把同一次实验算两次并把过期结果混进均值。冲突时与本次 run 命名约定
    （repro.json）一致者胜出；无 repro 信息时 _seed 命名优先（旧行为）。
    完全无种子信息的文件在该变体已有种子结果时忽略（与旧行为一致）。
    """
    expected = _expected_tagged(rdir) if rdir else None
    variant_scope, seed_scope = _run_scope(rdir) if rdir else (None, None)
    groups = {}
    for fp, d in items:
        # 合法 JSON 但非对象（如误放入的数组）或变体名非法（数组/对象既不是
        # 合法标识也不能作为字典键）的文件按无效结果跳过，不让一个坏文件
        # 毁掉整张图（与 curves/badcase 的容错口径一致）。
        if not (isinstance(d, dict) and isinstance(d.get("variant"), str)
                and d["variant"]):
            continue
        if variant_scope is not None and d["variant"] not in variant_scope:
            continue
        tagged = _is_tagged(fp)
        m = _SEED_SUFFIX_RE.search(os.path.splitext(os.path.basename(fp))[0])
        seed = int(m.group(1)) if m else d.get("seed")
        try:
            hash(seed)
        except TypeError:
            # 外部/旧格式 JSON 的 seed 可能是数组/对象：按缺种子信息处理。
            seed = None
        if not _seed_in_scope(seed, seed_scope):
            continue
        groups.setdefault(d["variant"], []).append((seed, d, tagged))
    out = {}
    for v, items_v in groups.items():
        if expected is not None:
            # 与 _prefer_seeded 同规则按命名约定筛掉另一族过期文件，但必须
            # 逐变体进行：不能因为某个变体有本次约定的文件，就把其它变体
            # 仅有的另一族结果整体丢弃（那会让这些变体从图上消失）。
            keep_v = [it for it in items_v if it[2] == expected]
            if keep_v:
                items_v = keep_v
        seeded = [it for it in items_v if it[0] is not None]
        if not seeded:
            out[v] = [d for _, d, _ in items_v]
            continue
        slots = {}
        for seed, d, tagged in seeded:
            cur = slots.get(seed)
            if cur is None:
                slots[seed] = (tagged, d)
                continue
            cur_tagged = cur[0]
            if expected is not None:
                win = (tagged == expected) and (cur_tagged != expected)
            else:
                win = tagged and not cur_tagged
            if win:
                slots[seed] = (tagged, d)
        out[v] = [d for _, d in slots.values()]
    return out


def _group_by_variant(pattern):
    """按变体分组 glob 到的结果文件（自动合并多种子 _seed<seed> 文件）。"""
    rdir = os.path.dirname(pattern)
    items = []
    for fp in sorted(glob.glob(pattern)):
        d = _load(fp)
        if d is not None:
            items.append((fp, d))
    return _group_items(items, rdir)


def _ppl_split_label(splits, invalid_split=False):
    """单一评测集的显示名；未知 split 不得冒充验证集。"""
    if invalid_split or len(splits) != 1:
        return "各评测集"
    split = next(iter(splits))
    if split == "test":
        return "测试集"
    if split == "val":
        return "验证集"
    return f"{split} 集" if isinstance(split, str) and split else "未知评测集"


def fig_ppl(rdir):
    groups = _group_by_variant(os.path.join(rdir, "eval_*.json"))
    if not groups:
        return False
    rows = []
    splits = set()
    invalid_split = False
    for v, ds in groups.items():
        # 多种子：每变体取 PPL 均值，误差棒为样本标准差；单种子不画误差棒。
        # ppl 为 null 时回退 val_ppl（与 run_all._ppl_of 同口径），坏值跳过。
        ppls = []
        for d in ds:
            p = d.get("ppl")
            if p is None:
                p = d.get("val_ppl")
            p = _finite_float(p)
            if p is not None:
                ppls.append(p)
        if not ppls:
            continue
        mean = float(np.mean(ppls))
        std = float(np.std(ppls, ddof=1)) if len(ppls) > 1 else 0.0
        rows.append((v, mean, std, any(d.get("val_ppl_clamped", False) for d in ds)))
        # 收集全部种子的 split（多种子均在同一 split 上评测；这里避免只看
        # 首个文件时漏掉其它种子的信息，标签才不会误导）
        for d in ds:
            try:
                splits.add(d.get("split", "val"))
            except TypeError:
                invalid_split = True
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
    set_name = _ppl_split_label(splits, invalid_split)
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
            results = d.get("results")
            if not isinstance(results, dict):
                continue
            for task, by_ctx in results.items():
                if task == "_skipped" or not isinstance(by_ctx, dict):
                    continue
                for ctx, m in by_ctx.items():
                    if not isinstance(m, dict):
                        continue
                    cell = agg.setdefault(task, {}).setdefault(
                        ctx, {"em": [], "recall": []})
                    em = _finite_float(m.get("em"))
                    recall = _finite_float(m.get("recall"))
                    if em is not None:
                        cell["em"].append(em)
                    if recall is not None:
                        cell["recall"].append(recall)
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
            items = []
            for k, m in r[task].items():
                try:
                    ctx = int(k)
                except (TypeError, ValueError, OverflowError):
                    continue
                items.append((ctx, m))
            items.sort(key=lambda it: it[0])
            # fwe 主指标是 top-10 recall；其余检索任务为 EM。
            metric = "recall" if task == "fwe" else "em"
            points = [(x, m[metric]) for x, m in items
                      if _finite_float(m.get(metric)) is not None]
            if not points:
                continue
            xs = [x for x, _ in points]
            ys = [y * 100 for _, y in points]
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
    if not isinstance(d, dict):
        return False
    variants = d.get("variants")
    if not isinstance(variants, dict):
        return False
    variant_scope, _ = _run_scope(rdir)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    # 每个变体先收集可用的 batch 档位。Prefill 对比必须用同一 batch：
    # 不能拿第一个变体的档位（如 "1"）去查另一个只测过 "8" 的变体，
    # 否则后者整条线静默消失，图上只剩一个变体却仍标着 batch=1。
    series = []
    for v, e in variants.items():
        if variant_scope is not None and v not in variant_scope:
            continue
        if not isinstance(e, dict):
            continue
        prefill = e.get("prefill_ms")
        if not isinstance(prefill, dict):
            continue
        by_ctx = {}
        for c, cell in prefill.items():
            if not isinstance(cell, dict):
                continue
            try:
                by_ctx[int(c)] = cell
            except (TypeError, ValueError, OverflowError):
                continue
        if by_ctx:
            series.append((v, e, by_ctx))
    batch_sets = [{str(k) for cell in by_ctx.values() for k in cell}
                  for _, _, by_ctx in series]
    common = set.intersection(*batch_sets) if batch_sets else set()

    def _batch_key(keys):
        if not keys:
            return None
        if "1" in keys:
            return "1"
        return min(keys, key=lambda k: (not k.isdigit(), int(k)
                                        if k.isdigit() else 0, k))

    common_key = _batch_key(common)
    mixed_batches = common_key is None
    used_keys = set()
    for v, e, by_ctx in series:
        key = common_key if common_key is not None else _batch_key(
            {str(k) for cell in by_ctx.values() for k in cell})
        if key is None:
            continue
        used_keys.add(key)
        xs, ys = [], []
        for c in sorted(by_ctx):
            val = _finite_float(by_ctx[c].get(key))
            if val is not None:
                xs.append(c)
                ys.append(val)
        if not xs:
            continue
        ax1.plot(xs, ys, marker="o",
                 label=_label(v) + ("(SDPA)" if e.get("use_sdpa")
                                    and v == "standard_sdpa" else ""),
                 color=COLORS.get(v))
    ax1.set_xlabel("上下文长度 (tokens)")
    if mixed_batches:
        # 没有共同 batch 时说明各变体测的 workload 不同，不能假装可横向比较。
        batch_label = ("各变体首个可用档位 "
                       f"({', '.join(sorted(used_keys))})——不可横向比较")
    else:
        batch_label = str(common_key)
    ax1.set_ylabel(f"Prefill (ms, batch={batch_label})")
    ax1.set_title("Prefill")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8)
    for v, e in variants.items():
        if variant_scope is not None and v not in variant_scope:
            continue
        if not isinstance(e, dict):
            continue
        decode = e.get("decode_ms_per_tok")
        if not isinstance(decode, dict):
            continue
        actual = e.get("decode_prefill_ctx")
        actual = actual if isinstance(actual, dict) else {}
        points = []
        for c, val in decode.items():
            dec = _finite_float(val)
            if dec is None:
                continue
            try:
                ctx = int(c)
            except (TypeError, ValueError, OverflowError):
                continue
            actual_ctx = _finite_float(actual.get(c))
            x = int(actual_ctx) if actual_ctx is not None else ctx
            points.append((x, dec))
        if not points:
            continue
        points.sort(key=lambda it: it[0])
        xs = [x for x, _ in points]
        ys = [y for _, y in points]
        ax2.plot(xs, ys, marker="s", label=_label(v), color=COLORS.get(v))
    ax2.set_xlabel("decode 前的实际 KV 上下文长度 (tokens)")
    ax2.set_ylabel("Decode (ms/tok)")
    ax2.set_title("Decode（KV 缓存，batch=1）")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8)
    fig.suptitle("推理基准对比", y=1.02)
    fig.savefig(os.path.join(rdir, "fig_latency.png"), dpi=150)
    plt.close(fig)
    return True


_RECORDS_CACHE: OrderedDict = OrderedDict()
_RECORDS_CACHE_MAX = 8


def _read_records(rdir, variant):
    """读入逐记录 CSV，返回列名 -> np.float64 数组（缺失值为 NaN）。

    同一批文件被多张图消费，按 (路径, mtime, size) 缓存，文件变化自动
    失效；缓存保留多个 key（LRU 上限 8）。
    """
    import csv
    # 兼容单种子 analysis_records_<v>.csv 与多种子 analysis_records_<v>_seed<s>.csv，
    # 多种子记录合并后画图（分布图上的种子差异会自然体现为更宽的分布）。
    paths = _prefer_seeded(
        [os.path.join(rdir, f"analysis_records_{variant}.csv")]
        + sorted(glob.glob(os.path.join(
            rdir, f"analysis_records_{variant}_seed*.csv"))), rdir=rdir)
    key = (variant, tuple(
        (p, os.path.getmtime(p), os.path.getsize(p))
        for p in paths if os.path.exists(p)))
    cached = _RECORDS_CACHE.get(key)
    if cached is not None:
        _RECORDS_CACHE.move_to_end(key)
        return cached
    names = []
    per_file = []   # [(file_cols, n_rows), ...]，保持文件顺序
    for path in paths:
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if not header:
                continue
            idx = {name: i for i, name in enumerate(header) if name != "task"}
            for name in idx:
                if name not in names:
                    names.append(name)
            file_cols = {name: [] for name in idx}
            n_rows = 0
            for parts in reader:
                n_rows += 1
                for name, i in idx.items():
                    v = parts[i] if i < len(parts) else ""
                    if v in ("", "None", "nan"):
                        file_cols[name].append(np.nan)
                        continue
                    try:
                        file_cols[name].append(float(v))
                    except (TypeError, ValueError, OverflowError):
                        file_cols[name].append(np.nan)
        per_file.append((file_cols, n_rows))
    out = {}
    # 合并多种子文件时表头可能因代码版本不同而缺列。每个文件在行空间里占
    # 自己的 [offset, offset+n_rows) 区间，缺列时必须按该文件的真实行数在此
    # 区间补 NaN；统一在列尾补会让后续文件的行整体前移，列与记录错位
    # （消费方用 isfinite 掩码按位置配对，错位会静默产生错误图表）。
    for name in names:
        vals = []
        for file_cols, n_rows in per_file:
            col = file_cols.get(name)
            vals.extend(col if col is not None else [np.nan] * n_rows)
        out[name] = np.asarray(vals, dtype=np.float64)
    _RECORDS_CACHE[key] = out
    _RECORDS_CACHE.move_to_end(key)
    while len(_RECORDS_CACHE) > _RECORDS_CACHE_MAX:
        _RECORDS_CACHE.popitem(last=False)
    return out


def fig_gate_hist(rdir, variant="lega"):
    cols = _read_records(rdir, variant)
    g, hit = cols.get("g"), cols.get("hit")
    if g is None or hit is None or g.size == 0:
        return False
    ok = np.isfinite(g)
    g_hit = g[ok & (hit == 1.0)].tolist()
    g_miss = g[ok & (hit == 0.0)].tolist()
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
    cols = _read_records(rdir, variant)
    c, g, ln = cols.get("c"), cols.get("g"), cols.get("log_n")
    if c is None or g is None or ln is None:
        return False
    m = np.isfinite(c) & np.isfinite(g) & np.isfinite(ln)
    if int(m.sum()) < 20:
        return False
    c, g, ln = c[m], g[m], ln[m]
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
    files = [(fp, _load(fp))
             for fp in sorted(glob.glob(os.path.join(rdir, "analysis_*.json")))
             if "records" not in os.path.basename(fp)]
    # 种子择优必须按变体分别进行（同 _group_by_variant）：全局 _prefer_seeded
    # 会因一个变体有 _seed 文件而把其它变体仅有的无种子结果整体丢弃；同一
    # (变体, 种子) 的裸/带后缀重复文件也在这里按命名约定去重。
    grouped = _group_items([(fp, d) for fp, d in files
                            if isinstance(d, dict) and d.get("n_records")], rdir)
    rows = []
    for v, ds in grouped.items():
        row = {"variant": v}
        for k in keys:
            vals = [x for x in (_finite_float(d.get(k)) for d in ds)
                    if x is not None]
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
            rdir, f"analysis_{variant}_seed*.json"))), rdir=rdir)
    # 多种子：逐 (层, 头) 的 AUROC 跨 seed 求均值后再画
    agg = {}
    for fp in files:
        d = _load(fp)
        if not isinstance(d, dict):
            continue
        heads = d.get("per_head")
        for h in (heads if isinstance(heads, list) else []):
            if not isinstance(h, dict):
                continue
            key = (h.get("layer"), h.get("head"))
            try:
                hash(key)
            except TypeError:
                continue
            cell = agg.setdefault(key, {"qterm": [], "logit": []})
            qterm = _finite_float(h.get("auroc_qterm"))
            logit = _finite_float(h.get("auroc_logit"))
            if qterm is not None:
                cell["qterm"].append(qterm)
            if logit is not None:
                cell["logit"].append(logit)
    def _head_key(k):
        try:
            return (0, float(k[0]), float(k[1]))
        except (TypeError, ValueError, OverflowError):
            return (1, str(k[0]), str(k[1]))

    ph = [{"layer": k[0], "head": k[1],
           "auroc_qterm": float(np.mean(v["qterm"])),
           "auroc_logit": float(np.mean(v["logit"]))}
          for k, v in sorted(agg.items(), key=lambda kv: _head_key(kv[0]))
          if v["qterm"] and v["logit"]]
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


def _load_train_logs(train_root, rdir=None):
    """扫描 out/<tag>/<variant>[/seed<s>]/train_log.jsonl，按变体归组。

    多种子拆分为多份记录；同一步的曲线在画图时取跨种子均值。
    rdir 给定时（repro.json 所在的结果目录）：同一变体下两种命名布局
    并存说明目录里混有不同种子数的旧 run（以不同 --seeds 在同一 tag 上
    重跑），按本次命名约定逐变体择优，避免把旧布局日志与本次日志混在
    一起求均值；某变体只有另一族文件时仍保留（不因其它变体而丢弃）。
    """
    series = {}
    expected = _expected_tagged(rdir) if rdir else None
    variant_scope, seed_scope = _run_scope(rdir) if rdir else (None, None)
    paths = (sorted(glob.glob(os.path.join(train_root, "*", "train_log.jsonl")))
             + sorted(glob.glob(os.path.join(
                 train_root, "*", "seed*", "train_log.jsonl"))))
    for path in paths:
        parts = os.path.normpath(path).split(os.sep)
        # 目录结构 .../<variant>/[seed<s>/]train_log.jsonl
        tagged = parts[-2].startswith("seed")
        variant = parts[-3] if tagged else parts[-2]
        if variant_scope is not None and variant not in variant_scope:
            continue
        if tagged:
            m = re.search(r"seed(\d+)$", parts[-2])
            seed = int(m.group(1)) if m else None
            if not _seed_in_scope(seed, seed_scope):
                continue
        elif not _seed_in_scope(None, seed_scope):
            # 显式空种子范围时，无 _seed 后缀的旧日志同样不属于本次 run。
            continue
        recs = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except ValueError:
                        continue
                    # 合法 JSON 但非对象（null/数组/标量）按损坏行跳过：
                    # 后续按 dict 取字段，不能让它毁掉整张训练曲线图。
                    if isinstance(rec, dict):
                        recs.append(rec)
        except OSError:
            continue
        if recs:
            series.setdefault(variant, []).append((tagged, recs))
    out = {}
    for v, items in series.items():
        if expected is not None:
            keep = [it for it in items if it[0] == expected]
            if keep:
                items = keep
        out[v] = [recs for _, recs in items]
    return out


def _step_or_none(v):
    """训练日志 step → 整数；bool/非有限浮点/小数/不可解析值返回 None。

    小数 step 若被 int() 截断，1.5 与 1.0 会合并到同一 x 坐标，多 run
    均值因而失真；整数值浮点（2.0）仍是合法 step，与 curves 同口径。
    """
    if isinstance(v, bool):
        return None
    if isinstance(v, float):
        if not math.isfinite(v) or not v.is_integer():
            return None
        v = int(v)
    else:
        try:
            v = int(v)
        except (TypeError, ValueError, OverflowError):
            return None
    try:
        if not math.isfinite(float(v)):
            return None
    except (OverflowError, ValueError):
        return None
    return v


def _mean_by_step(runs, key):
    """把多份 run 的记录按 step 对齐求均值（缺该 key 的记录跳过）。"""
    acc = {}
    for recs in runs:
        for r in recs:
            if not isinstance(r, dict):
                continue
            if r.get(key) is None or r.get("step") is None:
                continue
            step = _step_or_none(r["step"])
            if step is None:
                continue
            try:
                val = float(r[key])
            except (TypeError, ValueError, OverflowError):
                continue
            if not np.isfinite(val):
                continue
            acc.setdefault(step, []).append(val)
    xs = sorted(acc)
    return xs, [float(np.mean(acc[x])) for x in xs]


def fig_train_curves(rdir, train_root):
    """训练 loss、val PPL 与训练/验证损失对照三视图。

    同轴对照图让"训练降而验证升/停滞"的过拟合形态一眼可见——只看某一条
    曲线或只看最终数值都可能漏掉训练动态。
    """
    if not train_root or not os.path.isdir(train_root):
        return False
    series = _load_train_logs(train_root, rdir)
    if not series:
        return False
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.5, 4.2),
                                        constrained_layout=True)
    for i, v in enumerate(sorted(series)):
        color = COLORS.get(v, "#34495e")
        lw = 2.0 if v == "lega" else 1.2
        xs, ys = _mean_by_step(series[v], "loss")
        if xs:
            ax1.plot(xs, ys, label=_label(v), color=color, lw=lw, alpha=0.9)
        xs2, ys2 = _mean_by_step(series[v], "val_ppl")
        if xs2:
            ax2.plot(xs2, ys2, label=_label(v), color=color, marker=".",
                     lw=lw, alpha=0.9)
        # 同一坐标下的训练/验证损失：实线=训练，虚线=验证；两线间距
        # 即为泛化差距，扩大或验证线走平/上翘即过拟合信号。
        # 图例标签只由首个变体挂一次（线色区分变体），避免无标签图例告警。
        if xs:
            ax3.plot(xs, ys, color=color, lw=lw, alpha=0.9,
                     label=("训练 loss" if i == 0 else None))
        xv, yv = _mean_by_step(series[v], "val_loss")
        if xv:
            ax3.plot(xv, yv, color=color, lw=lw, ls="--", alpha=0.75,
                     label=("验证 loss" if i == 0 else None))
    ax1.set_xlabel("训练步数")
    ax1.set_ylabel("训练 loss")
    ax1.set_yscale("log")
    ax1.set_title("训练损失曲线（多种子均值）")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=7)
    ax2.set_xlabel("训练步数")
    ax2.set_ylabel("验证 PPL")
    ax2.set_yscale("log")
    ax2.set_title("验证困惑度曲线（多种子均值）")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=7)
    ax3.set_xlabel("训练步数")
    ax3.set_ylabel("loss")
    ax3.set_yscale("log")
    ax3.set_title("训练(实线)/验证(虚线)损失对照")
    ax3.grid(alpha=0.3)
    ax3.legend(fontsize=7)
    fig.suptitle(f"训练过程留档（源目录 {train_root}）", y=1.03)
    fig.savefig(os.path.join(rdir, "fig_train_curves.png"), dpi=150)
    plt.close(fig)
    return True


def main():
    p = argparse.ArgumentParser(description="结果画图")
    p.add_argument("--results_dir", required=True)
    p.add_argument("--train_root", default=None,
                   help="训练输出根目录（out/<tag>），用于绘制训练曲线")
    args = p.parse_args()
    rdir = args.results_dir
    os.makedirs(rdir, exist_ok=True)

    if not _HAS_MPL:
        print("[plots] 未安装 matplotlib，跳过全部图表生成（不阻塞流水线）")
        return

    targets = [
        ("fig_ppl.png", fig_ppl, ()),
        ("fig_ruler_em.png", fig_ruler, ()),
        ("fig_latency.png", fig_latency, ()),
        ("fig_gate_hist.png", fig_gate_hist, ("lega",)),
        ("fig_calibration.png", fig_calibration, ("lega",)),
        ("fig_misleading.png", fig_misleading, ()),
        ("fig_auroc_heads.png", fig_auroc_heads, ("lega",)),
    ]
    if args.train_root:
        targets.append(("fig_train_curves.png", fig_train_curves, (args.train_root,)))

    made = []
    for name, fn, kw in targets:
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
