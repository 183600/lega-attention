"""配对检验、格式化、等价性与组件证据。"""
from .._shared import _install as _shared_install

_shared_install(globals())

def _stat_series(seed_map, payload_key, extractor):
    vals = []
    seed_map = seed_map if isinstance(seed_map, dict) else {}
    for s in _sorted_seeds(seed_map):
        slot = seed_map.get(s)
        d = slot.get(payload_key) if isinstance(slot, dict) else None
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
    多大的效应才检测得到"——区分"没有效应"与"功效不足"；新种子复现
    预测（replication_forecast）回答"再换一个训练种子，方向是否仍一致"，
    避免把均值 CI 误读成单次复现区间。均为确定性计算（固定随机种子），
    同一输入恒定复现。
    """
    # 配对差统一按 _paired_diffs 口径构造：bool/None/非有限值不参与，
    # bootstrap CI 与置换/符号检验使用同一批观测。
    diffs = []
    for x, y in zip(a, b):
        fx, fy = _finite_number(x), _finite_number(y)
        if fx is None or fy is None:
            continue
        d = fx - fy
        if math.isfinite(d):
            diffs.append(d)
    if res.get("n", 0) >= 1 and res.get("mean_diff") is not None:
        perm = paired_sign_flip_test(a, b)
        res["p_perm"] = perm.get("p_perm")
        res["perm_exact"] = perm.get("exact")
        sign = paired_sign_test(a, b)
        res["median_diff"] = sign.get("median_diff")
        res["p_sign"] = sign.get("p_sign")
        res["sign_n_pos"] = sign.get("n_pos")
        res["sign_n_neg"] = sign.get("n_neg")
        res["sign_n_zero"] = sign.get("n_zero")
        res["mean_diff_ci95"] = bootstrap_mean_ci(diffs)
        res["mde"] = min_detectable_effect(res.get("std_diff"), res.get("n"))
    else:
        res["p_perm"] = res["perm_exact"] = None
        res["median_diff"] = res["p_sign"] = None
        res["sign_n_pos"] = res["sign_n_neg"] = res["sign_n_zero"] = None
        res["mean_diff_ci95"] = res["mde"] = None

    # 效应 vs 噪声：同一对比里“显著”与“幅度有意义”是两件事。配对差
    # 的标准差 sd_diff 是种子/评测波动的直接尺度；|Δ|<sd_diff 表示单看
    # 任何一个种子，符号都可能翻转，这类边缘结果必须显式标注。
    diff = res.get("mean_diff")
    sd = res.get("std_diff")
    res["effect_to_noise"] = res["within_noise"] = res["below_mde"] = None
    if diff is not None and sd is not None:
        try:
            diff_f, sd_f = float(diff), float(sd)
        except (TypeError, ValueError, OverflowError):
            diff_f = sd_f = None
        if (diff_f is not None and sd_f is not None
                and math.isfinite(diff_f) and math.isfinite(sd_f)):
            res["within_noise"] = bool(sd_f > 0 and abs(diff_f) < sd_f)
            res["effect_to_noise"] = (abs(diff_f) / sd_f
                                      if sd_f > 0 else None)
            if (res["effect_to_noise"] is not None
                    and not math.isfinite(res["effect_to_noise"])):
                res["effect_to_noise"] = None
    if diff is not None and res.get("mde") is not None:
        try:
            diff_f, mde_f = abs(float(diff)), float(res["mde"])
        except (TypeError, ValueError, OverflowError):
            diff_f = mde_f = None
        if (diff_f is not None and mde_f is not None
                and math.isfinite(diff_f) and math.isfinite(mde_f)):
            res["below_mde"] = bool(abs(diff_f) < abs(mde_f))

    # 留一稳健性：n=5 时一个“幸运/倒霉种子”就可能决定均值符号与显著性。
    # 逐种子胜负计数与 leave-one-seed-out 均值方向给出两条不含正态假设的
    # 稳定性证据：若删掉任一单个种子后均值符号翻转，或逐种子胜负并非
    # 一致，则报道的“显著优于”对个体种子高度敏感，必须显式标注。
    res["n_seed_wins"] = res["n_seed_losses"] = res["n_seed_ties"] = None
    res["loo_sign_stable"] = res["loo_min_abs_mean"] = None
    if len(diffs) >= 2:
        wins = sum(1 for d in diffs if d > 0)
        losses = sum(1 for d in diffs if d < 0)
        res["n_seed_wins"], res["n_seed_losses"] = wins, losses
        res["n_seed_ties"] = len(diffs) - wins - losses
        # 与 _stable_mean_std 同口径：朴素 sum 在多个 1e308 同号配对差上会
        # 溢出成 inf，使“删一种子后的方向”失去意义。mean_std 先缩放再还原。
        loo_means = []
        for i in range(len(diffs)):
            rest = diffs[:i] + diffs[i + 1:]
            rest_mean, _ = mean_std(rest)
            if rest_mean is not None:
                loo_means.append(rest_mean)
        overall_mean, _ = mean_std(diffs)
        if overall_mean is not None and loo_means:
            sign = 1 if overall_mean > 0 else -1
            res["loo_sign_stable"] = bool(
                all((m > 0) == (sign > 0) for m in loo_means))
            res["loo_min_abs_mean"] = float(min(abs(m) for m in loo_means))

    # 新种子复现预测：均值 CI 不是单次复现区间。这里按原始指标尺度给出
    # “同配置/同数据下再抽一个训练种子”的预测区间与同号概率；指标尺度
    # （相对 PPL/log 比等）上的 SESOI 概率另由 equivalence.replication 给出。
    res["replication"] = None
    try:
        res["replication"] = replication_forecast(diffs)
    except (TypeError, ValueError, OverflowError) as exc:
        res["replication"] = {
            "status": "invalid", "label": f"无法计算复现预测（{exc}）",
            "note": "输入不可表示；按缺失处理，不伪造概率。"}

    # SESOI 解释层（TOST）：显著性检验不能证明“无差异”。这里为每个有
    # 明确定义 SESOI 的指标附加“差异是否小于实际重要阈值”的证据；
    # 最终 status 会在 _apply_holm 里用同一指标家族内 Holm 校正后的
    # TOST p 重新判定（此处先用原始 p 给独立调用方一个可用值）。
    res["equivalence"] = None
    try:
        eq = assess_paired_equivalence(
            a, b, res.get("metric"),
            lower_is_better=(res.get("lower_is_better") is True),
            alpha=ALPHA, min_seeds=MIN_SEEDS)
    except (TypeError, ValueError, OverflowError) as exc:
        res["equivalence"] = {
            "status": "invalid",
            "label": f"无法计算等价性（{type(exc).__name__}: {exc}）",
            "equivalent": False, "multiplicity_adjusted": False,
            "p_used": None, "reason": str(exc)}
    else:
        if eq is not None:
            eq.update(interpret_equivalence(
                eq, res.get("p_adj"), alpha=ALPHA, min_seeds=MIN_SEEDS))
            res["equivalence"] = eq
    return res


def _fmt_signed(value, spec: str = "+.4f") -> str:
    """有限数值按带符号格式渲染；None/NaN/Inf 等缺失值显示 '-'。

    汇总表直接对 t['mean_diff'] 做格式化的前提是统计检验一定给出了有限
    均值；n=0、方差退化或输入不可表示时该字段为 None。缺少这个保护会让
    一个坏结果文件把整份汇总变成 TypeError。
    """
    f = _finite_number(value)
    return "-" if f is None else format(f, spec)


def _fmt_mean_std(t, key: str = "mean_diff", spec: str = "+.4f") -> str:
    """配对检验的“均值 ± 标准差”渲染；任一缺失时只渲染可用部分。"""
    t = t if isinstance(t, dict) else {}
    mean = _finite_number(t.get(key))
    if mean is None:
        return "-"
    std = _finite_number(t.get("std_diff"))
    return format(mean, spec) + (f" ± {std:.4f}" if std is not None else "")


def _fmt_pct1(value) -> str:
    """比例按百分数渲染；缺失显示 '-'，乘 100 溢出时显示“溢出”。"""
    f = _finite_number(value)
    if f is None:
        return "-"
    scaled = f * 100.0
    return f"{scaled:.1f}%" if math.isfinite(scaled) else "溢出"


def _fmt_prob(value) -> str:
    """预测概率按整数百分比渲染；缺失/损坏显示 '-'。"""
    f = _finite_number(value)
    if f is None or not 0.0 <= f <= 1.0:
        return "-"
    return f"{f:.0%}"


def _seed_winloss_str(t: dict) -> str:
    """逐种子配对差符号计数渲染为“胜/负/平”（无信息时返回 '-'）。"""
    if not isinstance(t, dict) or t.get("n_seed_wins") is None:
        return "-"
    return (f"{t['n_seed_wins']}/{t.get('n_seed_losses', 0)}/"
            f"{t.get('n_seed_ties', 0)}")


def _loo_stability_str(t: dict) -> str:
    """留一方向稳定性渲染：✅ 稳定 / ❌ 删任一种子即翻转 / '-' 无信息。"""
    if not isinstance(t, dict) or t.get("loo_sign_stable") is None:
        return "-"
    return "✅" if t["loo_sign_stable"] else "❌"


def _fmt_eq_value(value, eq) -> str:
    """把 TOST 尺度上的数值换算为报告单位（相对差/对数比/绝对差）。"""
    f = _finite_number(value)
    if f is None or not isinstance(eq, dict):
        return "-"
    kind = eq.get("kind")
    if kind == "relative":
        v = f * 100.0
        return f"{v:+.2f}%" if math.isfinite(v) else "溢出"
    if kind == "log_ratio":
        try:
            v = math.expm1(f) * 100.0
        except OverflowError:
            return "溢出"
        return f"{v:+.2f}%" if math.isfinite(v) else "溢出"
    scale = _finite_number(eq.get("display_scale"))
    scale = 1.0 if scale is None or scale == 0 else scale
    unit = eq.get("display_unit") or ""
    v = f * scale
    if not math.isfinite(v):
        # 有限输入 × 有限 scale 仍可能溢出（如 1e308 × 100）。
        return "溢出"
    if unit == "pp":
        return f"{v:+.2f} pp"
    if unit == "nat":
        return f"{v:+.4f} nat"
    if unit:
        return f"{v:+.4f} {unit}"
    return f"{v:+.4f}"


def _fmt_eq_margin(eq) -> str:
    """渲染 SESOI 阈值（带 ± 与单位）。"""
    if not isinstance(eq, dict):
        return "-"
    m = _finite_number(eq.get("margin"))
    if m is None:
        return "-"
    if eq.get("kind") == "relative":
        v = m * 100.0
        return f"±{v:.2f}%" if math.isfinite(v) else "±溢出"
    if eq.get("kind") == "log_ratio":
        try:
            v = math.expm1(m) * 100.0
        except OverflowError:
            return "±溢出"
        return f"±{v:.2f}%" if math.isfinite(v) else "±溢出"
    scale = _finite_number(eq.get("display_scale"))
    scale = 1.0 if scale is None or scale == 0 else scale
    unit = eq.get("display_unit") or ""
    v = m * scale
    if not math.isfinite(v):
        # 有限 margin × 有限 scale 仍可能溢出。
        return "±溢出"
    if unit == "pp":
        return f"±{v:.2f} pp"
    if unit == "nat":
        return f"±{v:.4f} nat"
    if unit:
        return f"±{v:.4f} {unit}"
    return f"±{v:.4f}"


def _eq_p_str(eq) -> str:
    """TOST p：优先 Holm 校正值；未校正时用 † 标注。"""
    if not isinstance(eq, dict):
        return "-"
    p = _finite_number(eq.get("p_tost_adj"))
    suffix = ""
    if p is None:
        p = _finite_number(eq.get("p_tost"))
        suffix = "†"
    if p is None:
        return "-"
    return f"{p:.4f}{suffix}"


def _eq_sensitivity_str(eq) -> str:
    """0.5×/1×/2× 网格的等价性标记（每格独立 Holm 校正）。"""
    if not isinstance(eq, dict):
        return "-"
    sens = eq.get("sensitivity")
    if not isinstance(sens, list) or not sens:
        return "-"
    parts = []
    for cell in sens:
        if not isinstance(cell, dict):
            continue
        factor = _finite_number(cell.get("factor"))
        n = cell.get("n")
        n_ok = (isinstance(n, int) and not isinstance(n, bool)
                and n >= MIN_SEEDS)
        if cell.get("status") == "invalid" or not n_ok:
            mark = "–"
        else:
            mark = "✅" if cell.get("equivalent") is True else "❌"
        f_s = f"{factor:g}×" if factor is not None else "?"
        parts.append(f"{f_s}{mark}")
    return " ".join(parts) if parts else "-"


# 主表之外单独做 SESOI 解释的核心指标；交互/Shapley/逐任务等探索性指标
# 若也定义 SESOI 会淹没主线，保留在 significance.json 而不进汇总主段。
# 这里用与 AUROC_DIFF_METRIC/AP_DIFF_METRIC 相同的字面量，避免常量定义
# 顺序造成的导入期 NameError（函数体在调用时才解析该元组）。


def _equivalence_rows(tests) -> list:
    rows = []
    for t in (tests if isinstance(tests, list) else []):
        if not isinstance(t, dict):
            continue
        if t.get("interaction") or t.get("ruler_task") or t.get("shapley_factor"):
            continue
        eq = t.get("equivalence")
        if not isinstance(eq, dict):
            continue
        if t.get("metric") not in _EQ_CORE_METRICS:
            continue
        if t.get("mean_diff") is None:
            continue
        rows.append(t)
    order = {m: i for i, m in enumerate(_EQ_CORE_METRICS)}
    rows.sort(key=lambda t: (order.get(t.get("metric"), 99),
                             str(t.get("candidate")),
                             str(t.get("reference"))))
    return rows


def _equivalence_overview(tests) -> dict:
    """significance.json 用的等价性计数（只统计核心指标，与汇总表一致）。"""
    rows = _equivalence_rows(tests)
    counts = {}
    for t in rows:
        status = (t.get("equivalence") or {}).get("status")
        counts[status] = counts.get(status, 0) + 1
    margins = {}
    for t in rows:
        eq = t.get("equivalence") or {}
        metric = t.get("metric")
        if metric not in margins and metric is not None:
            margins[metric] = {
                "margin": eq.get("margin"), "kind": eq.get("kind"),
                "label": eq.get("margin_label")}
    replication_counts = {}
    for t in rows:
        rep = (t.get("equivalence") or {}).get("replication") or {}
        status = rep.get("status") if isinstance(rep, dict) else None
        if status is not None:
            replication_counts[status] = replication_counts.get(status, 0) + 1
    return {
        "method": ("双单侧配对 t 检验（TOST）；等价=100(1−2α)% 区间完全落在 "
                   "±SESOI 内；TOST p 在同一指标家族内 Holm 校正"),
        "alpha": ALPHA,
        "n_rows": len(rows),
        "counts": counts,
        "margins": margins,
        "replication": {
            "method": ("新种子 t 预测区间与同号概率（描述性；"
                       "mean ± t·sd·sqrt(1+1/n)，种子独立可交换正态假设）"),
            "counts": replication_counts,
            "directionally_consistent_threshold": 0.95,
            "directionally_plausible_threshold": 0.80,
            "note": ("预测的是同配置/同数据/同评测协议下换一个训练种子；"
                     "不是对均值 CI 的重复，也不外推到新数据/新超参数；"
                     "不改变确认性判定。"),
        },
        "sensitivity_factors": [0.5, 1.0, 2.0],
        "note": ("SESOI 是分析前声明的协议阈值，不由当前数据估计；等价只说明"
                 "差异小于 SESOI，不等于效应为 0；未定不能写成无差异；"
                 "该解释层不改变预注册假设的 支持/拒绝/未定 判定。"),
    }


def _render_equivalence_section(tests) -> list:
    rows = _equivalence_rows(tests)
    md = ["", "### SESOI 与等价性/非劣性（TOST；补充性解释）", ""]
    md.append("显著性检验只能回答“能否拒绝 Δ=0”，不能证明“没有差异”。"
              "本表在预先声明的 SESOI（最小重要差异）上做双单侧 t 检验"
              "（TOST）：**只有 TOST 校正后显著且 90% 区间完全落在 ±SESOI "
              "内，才可写“实际等价”**；区间跨过 0 或 SESOI 边界时只能写"
              "“未定”，不能写成“无差异”。")
    if not rows:
        md.append("")
        md.append("（没有定义了 SESOI 的核心对比；或旧产物缺少 equivalence 字段。）")
        return md
    md.append("")
    md.append("| 对比 | 指标 | n | Δ | SESOI | 90% CI | p_TOST(Holm) | "
              "实际意义判定 | 0.5×/1×/2× |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for t in rows:
        eq = t.get("equivalence") or {}
        ci = eq.get("ci90")
        ci_s = ("-"
                if not (isinstance(ci, (list, tuple)) and len(ci) == 2)
                else f"[{_fmt_eq_value(ci[0], eq)}, {_fmt_eq_value(ci[1], eq)}]")
        n = eq.get("n")
        md.append(
            f"| {t.get('candidate')} vs {t.get('reference')} | "
            f"{t.get('metric')} | {n if n is not None else '-'} | "
            f"{_fmt_eq_value(eq.get('mean_diff'), eq)} | "
            f"{_fmt_eq_margin(eq)} | {ci_s} | {_eq_p_str(eq)} | "
            f"{eq.get('label', '-')} | {_eq_sensitivity_str(eq)} |")
    counts = {}
    for t in rows:
        status = (t.get("equivalence") or {}).get("status")
        counts[status] = counts.get(status, 0) + 1
    md.append("")
    md.append(
        f"本次核心对比 {len(rows)} 组：等价 "
        f"{counts.get('equivalent', 0)}、非劣 "
        f"{counts.get('noninferior', 0)}、显著但低于 SESOI "
        f"{counts.get('significant_below_sesoi', 0)}、显著且实际有意义 "
        f"{counts.get('significant_meaningful_better', 0) + counts.get('significant_meaningful_worse', 0)}、"
        f"未定 {counts.get('undetermined', 0)}、种子不足 "
        f"{counts.get('insufficient_seeds', 0)}、无效 "
        f"{counts.get('invalid', 0)}。")

    rep_rows = []
    for t in rows:
        rep = (t.get("equivalence") or {}).get("replication")
        if isinstance(rep, dict):
            rep_rows.append((t, (t.get("equivalence") or {}), rep))
    if rep_rows:
        md += ["", "#### 新种子复现预测（同配置/同数据换训练种子；描述性）", ""]
        md.append("均值的置信区间描述的是**均值**的不确定性；下面预测的是"
                  "**单个新训练种子**的配对差。预测区间按 "
                  "`mean ± t·sd·sqrt(1+1/n)`，同号概率是“种子效应独立可交换"
                  "正态”模型下的预测概率，不是 p 值、不是复现成功率保证。"
                  "n<3 不作概率判定；零方差（所有种子配对差完全相同）单列，"
                  "不能当作强复现证据。该表不改变任何预注册判定。")
        md += ["", "| 对比 | 指标 | n | 新种子预测区间 | 同号概率 | "
               "实际重要方向概率 | ±SESOI内概率 | 留一方向一致率 | 状态 |",
               "|---|---|---|---|---|---|---|---|---|"]
        for t, eq, rep in rep_rows:
            pi = rep.get("prediction_interval")
            pi_s = ("-"
                    if not (isinstance(pi, (list, tuple)) and len(pi) == 2)
                    else f"[{_fmt_eq_value(pi[0], eq)}, "
                         f"{_fmt_eq_value(pi[1], eq)}]")
            md.append(
                f"| {t.get('candidate')} vs {t.get('reference')} | "
                f"{t.get('metric')} | {rep.get('n', '-')} | {pi_s} | "
                f"{_fmt_prob(rep.get('prob_same_sign'))} | "
                f"{_fmt_prob(rep.get('prob_meaningful_same_direction'))} | "
                f"{_fmt_prob(rep.get('prob_within_sesoi'))} | "
                f"{_fmt_prob(rep.get('loo_sign_agreement'))} | "
                f"{rep.get('label') or rep.get('status') or '-'} |")
        md.append("")
        md.append("注: “同号概率”指新种子与本次观测均值同号（候选仍更好或仍"
                  "更差，取决于 Δ 的符号）；“实际重要方向概率”要求新种子不仅"
                  "同号而且超过 SESOI；“±SESOI内概率”是预测新种子落在等价带"
                  "内的概率——三者回答不同问题，不能只挑一个报告。"
                  "留一方向一致率不使用正态假设，只作朴素对照；"
                  "3~5 个种子无法检验正态/可交换假设，重尾或种子×数据交互"
                  "会使预测概率失真。")
    md.append("")
    md.append("注: 默认 SESOI 为 PPL 相对差 1%（log-PPL 为几何均值比 1%）、"
              "RULER EM 1.00 pp、AUROC/PR-AUC 0.010、因果 ΔNLL 0.01 nat、"
              "Shapley 0.01（PPL 为 −log 单位，RULER 为 1.00 pp）。这些是"
              "**协议约定**而非从当前数据估计；0.5×/1×/2× 网格各自在同一"
              "指标家族内 Holm 校正，结论若依赖网格点必须并列披露。"
              "p_TOST 后的 † 表示该对比未进入家族校正（通常 n<3 或补充性"
              "基线），只能作描述；等价/非劣不改变预注册的优效性判定，"
              "也不能把“未定”解释成“无差异”。")
    return md


# 因果病灶中“零假设本身有科学含义”的关键对照：安慰剂、阳性对照、
# 门恒开/恒关、证据反号、去全部证据。其余病灶仍以“是否有因果贡献”为主，
# 不强行套等价性解读。


def _render_multiverse_section(multiverse) -> list:
    """渲染分析选择多重宇宙（规格曲线）表；只展示可机读的描述性证据。"""
    md = ["", "### 分析选择多重宇宙（specification curve；探索性稳健性）", ""]
    if not isinstance(multiverse, dict):
        md.append("（本次汇总未计算规格曲线。）")
        return md
    outcomes = [o for o in (multiverse.get("outcomes") or [])
                if isinstance(o, dict)]
    if not outcomes:
        md.append("（没有可枚举分析规格的配对；旧产物或种子/协议不足。）")
        return md
    md.append("同一对比换一种同样可辩护的分析口径，方向/显著性是否还在？"
              "下表并列 PPL 原始差 vs log 比值、RULER macro vs micro、"
              "全种子 vs 逐个留一。规格之间高度相关且未穷尽所有选择，"
              "**不是独立假设、不做多重校正**；本表只用于披露分析选择敏感性，"
              "不改变预注册判定，也不能用“全部规格显著”替代效应量/SESOI。")
    md += ["", "| 对比 | 指标 | 规格(有效/总) | 基准Δ | 方向一致率 | "
           "同向显著率 | 方向翻转规格 | 同向但不显著规格 | 状态 |",
           "|---|---|---|---|---|---|---|---|---|"]
    status_order = {"fragile": 0, "undetermined_reference": 1,
                    "insufficient": 2, "directionally_robust": 3,
                    "robust": 4}
    ordered = sorted(
        outcomes,
        key=lambda o: (status_order.get(o.get("status"), 9),
                       str(o.get("candidate")), str(o.get("reference")),
                       str(o.get("metric"))))
    for outcome in ordered:
        flipped = [s.get("choice") for s in (outcome.get("flipped_specs") or [])
                   if isinstance(s, dict) and s.get("choice") is not None]
        nonsig = [s.get("choice")
                  for s in (outcome.get("nonsignificant_specs") or [])
                  if isinstance(s, dict) and s.get("choice") is not None]
        dc = outcome.get("direction_consistency")
        sc = outcome.get("significance_consistency")
        md.append(
            f"| {outcome.get('candidate')} vs {outcome.get('reference')} | "
            f"{outcome.get('metric')} | "
            f"{outcome.get('n_valid')}/{outcome.get('n_specs')} | "
            f"{_fmt_signed(outcome.get('reference_effect'))} | "
            f"{f'{dc:.0%}' if isinstance(dc, (int, float)) else '-'} | "
            f"{f'{sc:.0%}' if isinstance(sc, (int, float)) else '-'} | "
            f"{', '.join(flipped) if flipped else '-'} | "
            f"{', '.join(nonsig) if nonsig else '-'} | "
            f"{outcome.get('label') or outcome.get('status') or '-'} |")
    counts = multiverse.get("counts") or {}
    md.append("")
    md.append(
        "状态计数：方向翻转/依赖分析选择 "
        f"{counts.get('fragile', 0)}、方向一致但部分不显著 "
        f"{counts.get('directionally_robust', 0)}、全部可检验规格同向显著 "
        f"{counts.get('robust', 0)}、规格不足 "
        f"{counts.get('insufficient', 0)}、基准为 0 "
        f"{counts.get('undetermined_reference', 0)}。")
    md.append("注: 方向一致率 = 与基准规格同号的规格占比；同向显著率 = "
              "在可检验规格中“同向且 p<α”的占比。留一规格的自由度更低"
              "（n 减 1），其“不显著”是功效下降而非反向证据；规格数少时"
              "不要解读一致率的小数点。完整逐规格效应/p 值见 "
              "`significance.json` 的 `multiverse.outcomes`。")
    return md


def _causal_eq_note(t) -> str:
    """把因果病灶的 TOST 结果翻译成“该对照实际说明了什么”。"""
    if not isinstance(t, dict):
        return "-"
    eq = t.get("equivalence")
    if not isinstance(eq, dict):
        return "-"
    n = _integral_value(t.get("n"))
    no_op = _integral_value(t.get("n_no_op_seeds"))
    if (n is not None and n > 0 and no_op == n):
        return ("全部种子 no-op：训练后该组件系数已为 0，病灶没有真正改变"
                "前向；不作等价性判定（“无差异”不是“无效应”）")
    status = eq.get("status")
    lesion = t.get("lesion")
    if status == "equivalent":
        if lesion == "head_permute":
            return ("安慰剂差异在 SESOI 内：与“容量/参数分布即可解释”一致；"
                    "但不能证明效应恰为 0，也不能据此宣称头—证据对应有因果作用")
        if lesion in ("conc_permute", "query_permute", "len_permute"):
            return ("分布匹配安慰剂差异在 SESOI 内：与“该位置证据的逐点"
                    "对应关系可被其他位置替换”一致；但只能说明未检出可检测"
                    "作用，不能证明位置对应恰无贡献")
        if lesion == "gate_closed":
            return ("⚠ 阳性对照失败：关闭注意力输出仍与正常等价（差异小于 "
                    "SESOI），应先排查实现/评测，而不是解释方法")
        if lesion == "gate_open":
            return ("门控接近恒等：差异小于 SESOI，方法在该 checkpoint "
                    "退化为标准注意力")
        if lesion == "sign_flip":
            return ("门控对证据方向不敏感：反号后的差异小于 SESOI")
        if lesion == "no_evidence":
            return ("去掉全部证据项后的差异小于 SESOI：推理期证据输入的"
                    "可检测贡献有限")
        return "病灶差异小于 SESOI：推理期该机制的可检测贡献有限"
    if status == "significant_meaningful_worse":
        if lesion == "head_permute":
            return ("安慰剂显著变差且超过 SESOI：头—证据对应有因果作用，"
                    "容量/分布解释不足以说明收益")
        if lesion in ("conc_permute", "query_permute", "len_permute"):
            return ("分布匹配安慰剂显著变差且超过 SESOI：逐点位置对应有"
                    "可检测的因果作用；逐值分布/容量不能解释该差异")
        if lesion == "gate_closed":
            return "阳性对照通过：关闭注意力输出显著变差且超过 SESOI"
        if lesion == "gate_open":
            return "门控抑制写回：恒开显著变差且超过 SESOI"
        if lesion == "sign_flip":
            return "反号显著变差且超过 SESOI：门控确实在使用证据方向"
        return "病灶显著变差且超过 SESOI：推理期该机制有实际重要贡献"
    if status == "significant_meaningful_better":
        return ("病灶显著改善且超过 SESOI：该机制在推理期反向有害/冗余，"
                "不能写成“机制有用”")
    if status == "significant_below_sesoi":
        return ("统计显著但幅度低于 SESOI：有可重复差异，实际意义有限")
    if status == "noninferior":
        return "非劣：未检出超过 SESOI 的劣化"
    if status == "undetermined":
        return ("未定：置信区间未排除 SESOI，不能宣称安慰剂无效应或该机制"
                "无贡献；需补种子/降方差")
    if status == "insufficient_seeds":
        return "种子不足：不作等价性判定"
    return eq.get("label") or "-"


def _render_matched_ablation_contrasts(contrasts) -> list:
    """渲染“原始组件病灶 − 活动匹配病灶”的跨种子对比表。

    这张表不是“信息效应 + 写回效应”的精确分解：匹配只控制逐头平均 g，
    不控制方差/饱和比例/跨层联合分布。差值显著为正只说明原始 ΔNLL 中有
    一部分与平均门控开度变化共变；信息特异效应必须另看匹配病灶相对正常
    基线的检验。
    """
    if not contrasts:
        return []
    md = ["", "#### 消融混杂控制：活动匹配病灶（原始 ΔNLL − 匹配 ΔNLL）", ""]
    md.append("直接清零证据系数会同时改变“使用什么信息”和“门平均开多大”。"
              "匹配病灶在去掉同一组件后逐头平移 d，使平均 g 与基线一致；"
              "下表检验原始−匹配的跨种子配对差（>0 表示原始病灶中有一部分"
              "可由平均写回强度变化解释）。该检验与原始病灶/匹配病灶各自"
              "相对基线的检验分属不同 Holm 家族，不能互相替代：信息特异"
              "效应看匹配病灶相对基线的独立检验，写回强度混杂看本表。")
    md += ["", "| 变体 | 组件 | 原始−匹配 ΔNLL均值 ± SD | n种子 | "
           "逐种子胜负 | p_adj(Holm) | 判定 |",
           "|---|---|---|---|---|---|---|"]
    for t in contrasts:
        padj = (f"{t['p_adj']:.4f}"
                if t.get("p_adj") is not None else "-")
        md.append(
            f"| {t['variant']} | {t.get('matched_factor', '-')} "
            f"({t.get('matched_of', '-')} − {t['lesion']}) | "
            f"{_fmt_mean_std(t)} | {t['n']} | {_seed_winloss_str(t)} | "
            f"{padj} | {t.get('matched_verdict', '-')} |")
    md.append("")
    md.append("注: 匹配只保持每个头/层的平均门控，不保持门控方差、饱和比例"
              "或跨层联合分布；偏移又在同一批 val 窗口上估计（每头/层仅一个"
              "标量），属于事后均值匹配而非留出样本上的确认性控制。因此"
              "“原始−匹配”显著只定位一个混杂来源，不是把原始效应精确分解为"
              "“信息效应 + 写回效应”或形式化中介分解。")
    return md


def _render_causal_equivalence_section(causal_tests) -> list:
    rows = [t for t in (causal_tests if isinstance(causal_tests, list) else [])
            if isinstance(t, dict)
            and t.get("lesion") in _CAUSAL_EQ_LESIONS
            and isinstance(t.get("equivalence"), dict)
            and t.get("mean_diff") is not None]
    md = ["", "#### 关键病灶的 SESOI 解读（安慰剂/阳性对照不能只看 p>0.05）", ""]
    if not rows:
        md.append("（缺少可解释的因果病灶等价性结果。）")
        return md
    md.append("| 变体 | 病灶 | ΔNLL | SESOI | p_TOST | 解读 |")
    md.append("|---|---|---|---|---|---|")
    for t in rows:
        eq = t["equivalence"]
        p_s = "-" if eq.get("not_applicable") is True else _eq_p_str(eq)
        md.append(
            f"| {t.get('variant')} | {t.get('lesion_label', t.get('lesion'))} | "
            f"{_fmt_eq_value(eq.get('mean_diff'), eq)} | "
            f"{_fmt_eq_margin(eq)} | {p_s} | {_causal_eq_note(t)} |")
    md.append("")
    md.append("注: 病灶 ΔNLL 的 SESOI 为 0.01 nat（约 1% PPL），是协议约定。"
              "`head_permute` 是推理期容量/分布安慰剂；"
              "`conc_permute/query_permute/len_permute` 是 B/A/C 位置证据的"
              "分布匹配安慰剂：**等价**支持“逐值分布/容量或位置可替换即可"
              "解释”，**显著变差且超过 SESOI** 才支持逐点对应有因果作用；"
              "`gate_closed` 是阳性对照，若等价则说明实验灵敏度不足，必须先"
              "排查实现/评测。`no_op` 行不参与等价性判定。")
    return md


def _causal_equivalence_overview(causal_tests) -> dict:
    """因果病灶的 SESOI 计数摘要（significance.json）。"""
    rows = [t for t in (causal_tests if isinstance(causal_tests, list) else [])
            if isinstance(t, dict)
            and t.get("lesion") in _CAUSAL_EQ_LESIONS
            and isinstance(t.get("equivalence"), dict)]
    counts = {}
    for t in rows:
        status = (t.get("equivalence") or {}).get("status")
        counts[status] = counts.get(status, 0) + 1
    return {
        "method": "TOST on ΔNLL vs 0，SESOI=0.01 nat；同一指标家族 Holm",
        "n_rows": len(rows), "counts": counts,
        "lesions": list(_CAUSAL_EQ_LESIONS),
        "note": ("no_op 不参与等价性判定；安慰剂等价只支持“容量/分布可解释”，"
                 "不等于效应恰为 0；gate_closed 等价是阳性对照失败信号。"),
    }


def _render_parameter_matched_control(t, capacity_test=None, ran=False) -> list:
    """渲染训练期参数匹配安慰剂对照（lega vs 同参数/同 FLOPs 的噪声证据门）。"""
    md = ["", "### 参数匹配安慰剂对照（训练期：容量/优化红利）", ""]
    if not isinstance(t, dict):
        if ran:
            md.append("⚠ 本轮运行了 `lega_placebo`，但没有与 `lega` 可配对的"
                      "共同种子/一致评测协议，无法形成容量对照；请检查 "
                      "seed 与 eval 协议，不能据此声称容量已排除。")
        else:
            md.append("⚠ 本轮未运行 `lega_placebo`。LEGA vs standard 的差异仍混有"
                      "“多出的门控参数/一次门控运算”这一容量解释；需要排除时应显式"
                      "加 `--param_control`（或在 `--variants` 中点名 "
                      "`lega_placebo`）后重跑。")
        return md
    mean = _finite_number(t.get("mean_diff"))
    std = _finite_number(t.get("std_diff"))
    ci = t.get("mean_diff_ci95")
    ci_s = (f"[{ci[0]:+.4f}, {ci[1]:+.4f}]"
            if isinstance(ci, (list, tuple)) and len(ci) == 2
            and all(_finite_number(x) is not None for x in ci) else "-")
    p_adj = _finite_number(t.get("p_adj"))
    eq = t.get("equivalence") if isinstance(t.get("equivalence"), dict) else {}
    eq_status = eq.get("status", "-")
    # 等价性结论优先于优效性 p：TOST 已判定差异落在 SESOI 内时，即使
    # 5 个种子方向一致、p_adj<alpha，也不能宣称“容量红利不能解释”——
    # 那会与同表“PPL SESOI 等价性=equivalent”自相矛盾。
    if eq_status == "equivalent":
        reading = ("落入 SESOI 等价区：无法区分 LEGA 与噪声证据对照，"
                   "不能把收益归因于 B/C 证据（容量/优化红利未被排除）")
    elif mean is not None and p_adj is not None and p_adj < ALPHA and mean < 0:
        if t.get("within_noise") or t.get("loo_sign_stable") is False:
            reading = "方向有利但效应落在种子噪声内/留一翻转：只能算线索"
        else:
            reading = ("证据特定性成立：同参数/同 FLOPs 下，真实 B/C 证据优于"
                       "噪声证据；容量/优化红利不能单独解释")
    elif mean is not None and mean > 0 and p_adj is not None and p_adj < ALPHA:
        reading = "噪声证据反而更好：B/C 证据项可能有害或训练不稳定"
    else:
        reading = "未定：方向/显著性不足，须对照 MDE 与 TOST 解读，不能写成“无差异”"
    md.append("| 对比 | ΔPPL均值 ± ΔSD | Δ 95% CI | n | 胜/负/平 | p_adj(Holm) | "
              "PPL SESOI 等价性 | 解读 |")
    md.append("|---|---|---|---|---|---|---|---|")
    delta = ("-" if mean is None else
             f"{mean:+.4f}" + (f" ± {std:.4f}" if std is not None else ""))
    md.append(f"| lega vs lega_placebo | {delta} | {ci_s} | {t.get('n', '-')} | "
              f"{_seed_winloss_str(t)} | "
              f"{'-' if p_adj is None else f'{p_adj:.4f}'} | {eq_status} | "
              f"{reading} |")
    if isinstance(capacity_test, dict):
        m2 = _finite_number(capacity_test.get("mean_diff"))
        s2 = _finite_number(capacity_test.get("std_diff"))
        p2 = _finite_number(capacity_test.get("p_adj"))
        ci2 = capacity_test.get("mean_diff_ci95")
        ci2_s = (f"[{ci2[0]:+.4f}, {ci2[1]:+.4f}]"
                 if isinstance(ci2, (list, tuple)) and len(ci2) == 2
                 and all(_finite_number(x) is not None for x in ci2) else "-")
        eq2 = (capacity_test.get("equivalence")
               if isinstance(capacity_test.get("equivalence"), dict) else {})
        if eq2.get("status") == "equivalent":
            read2 = "噪声特征与 qgate 落在等价区：额外容量本身无实质影响"
        elif m2 is not None and p2 is not None and p2 < ALPHA and m2 < 0:
            read2 = ("额外容量/优化自由度单独即可降低 PPL：lega vs placebo 的"
                     "差异必须在该容量红利背景下解读")
        elif m2 is not None and p2 is not None and p2 < ALPHA and m2 > 0:
            read2 = ("噪声特征显著差于 qgate：对照可能“有害”，"
                     "lega > placebo 不能单独证明 B/C 证据有用")
        else:
            read2 = "未定：容量本身的影响方向/显著性不足"
        d2 = ("-" if m2 is None else
              f"{m2:+.4f}" + (f" ± {s2:.4f}" if s2 is not None else ""))
        md.append(f"| lega_placebo vs qgate（容量本身） | {d2} | {ci2_s} | "
                  f"{capacity_test.get('n', '-')} | "
                  f"{_seed_winloss_str(capacity_test)} | "
                  f"{'-' if p2 is None else f'{p2:.4f}'} | "
                  f"{eq2.get('status', '-')} | {read2} |")
    else:
        md.append("| lega_placebo vs qgate（容量本身） | - | - | - | - | - | - | "
                  "未运行/缺少 qgate，无法判断噪声特征本身是中性还是有害 |")
    md.append("")
    md.append("注: `lega_placebo` 的门控形状/参数个数/门控 FLOPs 与 `lega` 完全相同"
              "（w·q + a·u + b·v + d），但 u/v 是逐位置/逐头的确定性伪随机特征，"
              "不含任何查询/注意力/长度信息。二者唯一差别是 B/C 证据是否有信息，"
              "因此该对比把“多几个门控参数的容量/优化红利”从 B/C 证据效应中分离"
              "出来。它是探索性对照（未预注册），不能替代 H2–H5 的确认性检验；"
              "ΔPPL<0 表示 LEGA 更低。")
    return md


def _render_inference_shapley(inference_tests) -> list:
    """渲染同一 checkpoint 内的 2^3 干预 Shapley（与训练期 Shapley 并列）。"""
    rows = [t for t in (inference_tests if isinstance(inference_tests, list)
                        else []) if isinstance(t, dict)]
    md = ["", "### 推理期全因子 Shapley（同一 checkpoint 内干预；探索性）", ""]
    if not rows:
        md.append("⚠ 无法计算推理期 Shapley：需要 `lega` 的 8 个推理期格子"
                  "（正常基线、no_evidence 以及 query_only/conc_only/len_only/"
                  "no_query/no_conc/no_len）全部施加成功、且与基线共享同一批 "
                  "val 窗口。缺格或旧产物缺 `window_indices` 时如实跳过，"
                  "不按 0 补齐。")
        return md
    md.append("训练期全因子消融改变优化轨迹；本表在同一 checkpoint 上只清零门控"
              "项，直接对**训练好的前向**做 2^3 干预并计算 Shapley 值 "
              "`φ_A+φ_B+φ_C=ABC−none`。φ>0 表示该因子在最终前向中降低 PPL，"
              "`exp(−φ)` 为 PPL 几何均值比。与训练期 φ 符号不一致时，说明"
              "“重新训练后的组件贡献结构”没有在最终前向复现，应并列披露。")
    md += ["", "| 因子 | φ(推理期) | PPL几何比 | 占总效应 | n种子 | p_adj(Holm) | "
           "φ(训练期) | 符号一致 | 判定 |",
           "|---|---|---|---|---|---|---|---|---|"]
    for t in rows:
        phi = _finite_number(t.get("mean_diff"))
        ratio = _finite_number(t.get("shapley_ppl_ratio"))
        share = _finite_number(t.get("shapley_share"))
        tr = _finite_number(t.get("training_phi_mean"))
        agree = t.get("training_inference_sign_agree")
        p_adj = _finite_number(t.get("p_adj"))
        p_adj_s = "-" if p_adj is None else f"{p_adj:.4f}"
        md.append(
            f"| {t.get('shapley_factor_label', t.get('shapley_factor'))} | "
            f"{'-' if phi is None else f'{phi:+.4f}'} | "
            f"{'-' if ratio is None else f'{ratio:.4f}'} | "
            f"{_fmt_pct1(share)} | "
            f"{t.get('n', '-')} | "
            f"{p_adj_s} | "
            f"{'-' if tr is None else f'{tr:+.4f}'} | "
            f"{'-' if agree is None else ('是' if agree else '否 ⚠')} | "
            f"{t.get('verdict', '-')} |")
    md.append("")
    md.append("注: 推理期 φ 由 `query_only/conc_only/len_only`（单因子格）、"
              "`no_query/no_conc/no_len`（两因子格）与 `no_evidence`（无因子格）"
              "同基线共同分解；这些格子只改门控参数、不重新训练，因此是"
              "“最终前向是否使用该组件”的机制证据，不是“重新训练后是否有用”的"
              "训练期证据。推理期与训练期 Shapley 属探索性分析，未预注册。")
    return md


def _render_inference_interactions(inference_tests, training_tests) -> list:
    """渲染同一 checkpoint 内的 2^3 交互（与训练期交互并列）。"""
    rows = [t for t in (inference_tests if isinstance(inference_tests, list)
                        else []) if isinstance(t, dict)]
    md = ["", "### 推理期全因子交互效应（同一 checkpoint 内干预；探索性）", ""]
    if not rows:
        md.append("⚠ 无法计算推理期交互：需要 `lega` 的 8 个推理期格子全部"
                  "施加成功、且与基线共享同一批 val 窗口；缺格或旧产物缺 "
                  "`window_indices` 时如实跳过，不按 0 补齐。")
        return md
    md.append("训练期交互回答“重新训练时组件收益能否按单组件加总”；本表在同一"
              "checkpoint 上对 2^3 格子做同样的 A×B/A×C/B×C 对比，回答"
              "“训练好的前向里交互结构是否仍然存在”。数值为原始 PPL 的"
              "标准交互对比（与训练期 `PPL交互` 同尺度），交互=0 表示可加；"
              "显著非零只说明不能按单组件加总，不套用优/劣方向。")
    train = {}
    for t in (training_tests if isinstance(training_tests, list) else []):
        key = _test_label_key(t.get("interaction")) if isinstance(t, dict) else None
        if key is not None and t.get("metric") == "PPL交互":
            train[key] = t
    md += ["", "| 交互 | ΔPPL均值 ± ΔSD | n种子 | p_adj(Holm) | "
           "训练期ΔPPL | 符号一致 | 判定 |", "|---|---|---|---|---|---|---|"]
    for t in rows:
        tr = _finite_number(t.get("training_interaction_mean"))
        if tr is None:
            key = _test_label_key(t.get("interaction"))
            tr = _finite_number((train.get(key) or {}).get("mean_diff")
                                if key is not None else None)
        ph = _finite_number(t.get("mean_diff"))
        agree = t.get("training_inference_sign_agree")
        if agree is None and tr is not None and ph is not None:
            agree = bool((tr > 0) == (ph > 0) or (tr == 0 and ph == 0))
        p_adj = _finite_number(t.get("p_adj"))
        md.append(
            f"| {t.get('interaction', '-')} | {_fmt_mean_std(t)} | "
            f"{t.get('n', '-')} | "
            f"{'-' if p_adj is None else f'{p_adj:.4f}'} | "
            f"{'-' if tr is None else f'{tr:+.4f}'} | "
            f"{'-' if agree is None else ('是' if agree else '否 ⚠')} | "
            f"{t.get('verdict', '-')} |")
    md.append("")
    md.append("注: 推理期交互与训练期交互都属探索性、未预注册；二者符号不一致"
              "说明“重新训练后是否可加”没有在最终前向复现，应并列披露而不是"
              "只报有利的一期。n<3 或 Holm 校正后不显著只表示未检出，"
              "不等于交互不存在。")
    return md


def _render_ruler_length(analysis) -> list:
    """渲染 RULER 上下文长度剂量-反应（逐 ctx + 斜率 + 首末落差）。"""
    analysis = analysis if isinstance(analysis, dict) else {}
    raw_pairs = analysis.get("pairs")
    rows = ([p for p in raw_pairs if isinstance(p, dict)]
            if isinstance(raw_pairs, list) else [])
    md = ["", "### RULER 上下文长度剂量-反应（探索性；检验“长度校准”是否兑现）", ""]
    if not rows:
        md.append("⚠ 无法做上下文长度分层：缺少同一批种子的 RULER 产物，或"
                  "payload 没有按 ctx 分档的 `results`。只报跨长度平均会把"
                  "“短上下文涨、长上下文跌”与“所有长度小涨”混成同一个数，"
                  "因此不能在没有本表时宣称“长上下文有效”。")
        return md
    md.append("本方法的核心主张是随上下文长度校准门控，因此结果分析必须回答"
              "“优势是否随长度变化”，而不只是报一个跨长度平均。下表把每个"
              "种子的 RULER macro EM 先按**请求的 `ctx_lens` 档位**分档（任务"
              "等权；不同任务的 room_ctx 可能因生成预算被 clamp 到不同值，"
              "按请求档位对齐以免把任务组成变化当成新长度档），再对同一档做"
              "同种子配对差 `Δ=候选−参照`：逐 ctx 表回答哪一档驱动结论；斜率"
              "为逐种子 `ΔEM~log2(ctx)` 的 OLS 系数（长度每翻一倍的 EM 变化，"
              "pp/2×）；长−短为最长与最短档的落差，不假定线性。三者是三个"
              "独立 Holm 家族，全部为探索性、未预注册，不改变 H1–H7 的确认性"
              "判定。")

    def _pp(v):
        # 配对行来自外部/旧版 significance.json：损坏的字符串/NaN/Inf 不能
        # 让整份汇总页在格式化时崩溃，按缺失显示 "-"。
        f = _finite_number(v)
        if f is None:
            return "-"
        scaled = f * 100.0
        # 有限输入乘 100 仍可能溢出（如 1e308），不能写出 +inf。
        return f"{scaled:+.2f}" if math.isfinite(scaled) else "溢出"

    def _padj(t):
        p = _finite_number((t or {}).get("p_adj"))
        return "-" if p is None else f"{p:.4f}"

    def _n_contexts(p):
        n = _integral_value(p.get("n_contexts"))
        return n if n is not None and n >= 0 else 0

    multi = [p for p in rows if _n_contexts(p) >= LENGTH_MIN_CONTEXTS]
    single = [p for p in rows if _n_contexts(p) < LENGTH_MIN_CONTEXTS]
    if single:
        md += ["", f"⚠ {len(single)} 组对比只有 1 个公共请求档位，无法检验"
                   "长度趋势（只能报该档的描述统计）；如需回答长度主张，"
                   "`ruler.ctx_lens` 至少要覆盖 2 档、建议 3 档以上。"]
    if multi:
        md += ["", "| 对比 | 种子 | 请求ctx | 总体Δ(pp) | 最长ctx Δ(pp) | "
                   "长−短(pp) | 斜率(pp/2×) | 方向 正/负/零 | 长度趋势判定 |",
               "|---|---|---|---|---|---|---|---|---|"]
        n_reversal = n_trend = n_unbalanced = n_masking = n_loo = 0
        for p in multi:
            slope = (p.get("slope_per_doubling")
                     if isinstance(p.get("slope_per_doubling"), dict)
                     else {})
            span = (p.get("long_minus_short")
                    if isinstance(p.get("long_minus_short"), dict) else {})
            per_ctx = ([c for c in p.get("per_context")
                        if isinstance(c, dict)]
                       if isinstance(p.get("per_context"), list) else [])
            longest = (per_ctx[-1].get("mean_diff") if per_ctx else None)
            dc = (p.get("direction_consistency")
                  if isinstance(p.get("direction_consistency"), dict) else {})
            flags = []
            if p.get("overall_vs_longest_sign_flip"):
                flags.append("⚠ 总平均与最长ctx反向")
            if p.get("overall_vs_slope_sign_flip"):
                flags.append("⚠ 总平均与斜率反向")
            if p.get("overall_vs_span_sign_flip"):
                flags.append("⚠ 总平均与长−短反向")
            if p.get("long_context_masking"):
                flags.append("⚠ 总平均在SESOI内但最长ctx超出SESOI（平均掩盖长度效应）")
            if p.get("slope_loo_sign_stable") is False:
                flags.append("⚠ 删掉任一单个ctx后斜率符号翻转（趋势依赖个别档位）")
            if p.get("balanced_contexts") is False:
                flags.append("⚠ 各档任务集合不等权（分层均值≠总体分解）")
            n_reversal += int(bool(p.get("overall_vs_longest_sign_flip")))
            n_masking += int(bool(p.get("long_context_masking")))
            n_loo += int(p.get("slope_loo_sign_stable") is False)
            n_unbalanced += int(p.get("balanced_contexts") is False)
            slope_verdict = slope.get("verdict") or "-"
            # 不能按 “显著” 子串计数：``"不显著（Holm ...）"`` 也包含该词。
            # 只把校正后 p<α 的斜率检验算作显著趋势。
            slope_padj = _finite_number(slope.get("p_adj"))
            slope_n = _integral_value(slope.get("n"))
            if (slope_padj is not None and slope_n is not None
                    and slope_n >= MIN_SEEDS and slope_padj < ALPHA):
                n_trend += 1
            verdict = slope_verdict
            if flags:
                verdict += "；" + "；".join(flags)
            contexts = (p.get("contexts")
                        if isinstance(p.get("contexts"), (list, tuple))
                        else [])
            md.append(
                f"| {p.get('candidate')} vs {p.get('reference')} | "
                f"{p.get('n_seeds', '-')} | "
                f"{', '.join(str(c) for c in contexts) if contexts else '-'} | "
                f"{_pp(p.get('overall_mean_diff'))} | {_pp(longest)} | "
                f"{_pp(span.get('mean_diff'))} | "
                f"{_pp(slope.get('mean_diff'))} | "
                f"{dc.get('positive', 0)}/{dc.get('negative', 0)}/"
                f"{dc.get('zero', 0)} | {verdict} |")
        md += ["", f"本次 {len(multi)} 组可分解对比：{n_trend} 组斜率经 Holm 校正"
                   f"后显著，{n_reversal} 组出现“总平均与最长上下文方向相反”，"
                   f"{n_masking} 组“总平均在 ±1pp SESOI 内、最长档却超出”，"
                   f"{n_loo} 组斜率在删掉任一单个 ctx 后符号翻转，"
                   f"{n_unbalanced} 组各档任务集合不等权。方向相反、平均掩盖或"
                   "留一翻转时不能只报总体 RULER 平均；不等权时分层均值只是"
                   "描述，不是总体均值的严格分解。"]

    md += ["", "**逐上下文配对明细（Δ=候选−参照，pp）**", "",
           "| 对比 | ctx | n种子 | 候选EM | 参照EM | Δ(pp) | 95% CI(pp) | "
           "p | p_adj(Holm) | 结论 |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for p in rows:
        per_ctx = ([c for c in p.get("per_context")
                    if isinstance(c, dict)]
                   if isinstance(p.get("per_context"), list) else [])
        for c in per_ctx:
            ci = c.get("mean_diff_ci95")
            if not (isinstance(ci, (list, tuple)) and len(ci) == 2
                    and all(_finite_number(x) is not None for x in ci)):
                ci_s = "-"
            else:
                lo, hi = ci[0] * 100.0, ci[1] * 100.0
                ci_s = (f"[{lo:+.2f}, {hi:+.2f}]"
                        if math.isfinite(lo) and math.isfinite(hi)
                        else "溢出")
            p_raw = _finite_number(c.get("p"))
            md.append(
                f"| {p.get('candidate')} vs {p.get('reference')} | "
                f"{c.get('ctx')} | {c.get('n', '-')} | "
                f"{_fmt4(c.get('candidate_mean'))} | "
                f"{_fmt4(c.get('reference_mean'))} | "
                f"{_pp(c.get('mean_diff'))} | {ci_s} | "
                f"{'-' if p_raw is None else f'{p_raw:.4f}'} | "
                f"{_padj(c)} | {c.get('verdict') or '-'} |")
    md += ["", "注: 逐 ctx 检验只覆盖训练种子随机性；每个格子只有 20~30 个评测"
              "样例，评测抽样误差另见 `ruler_*.json` 的逐格 Wilson CI。斜率/长−短"
              "的替代解释是“长度效应集中在中段”：斜率显著而首末落差≈0、或二者"
              "方向不一致时，不能写成“上下文越长越好/越差”。所有对比的同一"
              f"检验族共用一次 Holm 校正（本次 m={len(analysis.get('ctx_tests') or [])}"
              f"/{len(analysis.get('slope_tests') or [])}"
              f"/{len(analysis.get('span_tests') or [])}，分别对应逐 ctx/斜率/长−短；"
              "保守）；`n<3` 只报描述统计。未检出趋势只说明当前种子数/档位数"
              "下无法检出，不等于“无长度效应”，需对照 JSON 的 `mde`、逐档 "
              "CI 与种子胜负后再解读。完整逐种子"
              "斜率、逐 ctx 差与检验 p 值见 `significance.json` 的 "
              "`ruler_length_pairs` / `ruler_length_tests`。"]
    return md


def _verdict_head(value) -> str:
    """取结论首词（去掉括号解释），用于跨尺度/口径一致性比较。"""
    return str(value if value is not None else "-").split("（")[0]


def _noise_verdict_note(t: dict) -> str:
    """把效应/噪声关系压成一句可渲染的警示（无信息时返回空串）。"""
    if not isinstance(t, dict):
        return ""
    if t.get("within_noise"):
        return "；⚠ |Δ|<ΔSD，效应落在种子/评测噪声内"
    if t.get("below_mde"):
        return "；效应低于当前种子数可检测的 MDE"
    if t.get("loo_sign_stable") is False:
        return "；⚠ 留一法方向翻转：删掉任一单个种子即改变均值符号，结论依赖个别种子"
    return ""


def _paired_observations(runs, cand, ref, metric_name, extractor):
    """提取同种子、同评测协议的配对观测 ``[(seed, a, b)]``。

    PPL 的 eval.json 可能分别落在 val/test（旧产物或手工重评）上：跨 split
    的“配对差”混入了评测集难度差异，不是同条件对照。RULER 的题面/抽样/
    生成预算/已评测格任一不同，EM 也不是同一批样例上的配对观测。此类种子
    整对剔除，并返回逐类剔除计数，供调用方显式报告而不是静默丢样本。
    """
    if cand not in runs or ref not in runs:
        return [], 0, 0, 0
    cm = _variant_seed_map(runs, cand)
    rm = _variant_seed_map(runs, ref)
    seeds = [s for s in _sorted_seeds(cm) if s in rm]
    observations = []
    split_skipped = protocol_skipped = cell_skipped = 0
    for s in seeds:
        sa, sb = cm.get(s), rm.get(s)
        if not (isinstance(sa, dict) and isinstance(sb, dict)):
            continue
        if metric_name.startswith("PPL"):
            ea = sa.get("eval") if isinstance(sa.get("eval"), dict) else {}
            eb = sb.get("eval") if isinstance(sb.get("eval"), dict) else {}
            if ea.get("split", "val") != eb.get("split", "val"):
                split_skipped += 1
                continue
            pa, pb = _ppl_protocol(ea), _ppl_protocol(eb)
            if not _same_protocol_fields(pa, pb):
                protocol_skipped += 1
                continue
        elif metric_name.startswith("RULER"):
            ra = sa.get("ruler") if isinstance(sa.get("ruler"), dict) else {}
            rb = sb.get("ruler") if isinstance(sb.get("ruler"), dict) else {}
            # 同 seed 只保证训练随机性相同；RULER 题面/抽样/生成预算与
            # 干扰语料来源任一不同，EM 就不是同一批样例上的配对观测。
            # 逐任务分解（RULER:<task>）、综合平均与 micro 口径必须走同一套
            # 可比性门槛，否则会从不一致的样本数/评测格上得出假结论。
            pa, pb = _ruler_protocol(ra), _ruler_protocol(rb)
            if not _same_protocol_fields(pa, pb):
                protocol_skipped += 1
                continue
            ca = _ruler_cell_signature(ra)
            cb = _ruler_cell_signature(rb)
            if ca is not None and cb is not None and ca != cb:
                cell_skipped += 1
                continue
        va, vb = extractor(sa), extractor(sb)
        if va is None or vb is None:
            continue
        observations.append((s, va, vb))
    return observations, split_skipped, protocol_skipped, cell_skipped


def _paired_comparisons(runs, pairs, metric_name, extractor, lower_better) -> list:
    """对每个 (候选, 参照) 在共同种子上做配对 t 检验（原始 p，未校正）。

    配对观测由 :func:`_paired_observations` 统一提取；跨 split/协议/格
    不一致的种子整对剔除，并在结果里记录剔除了多少对。
    """
    out = []
    for cand, ref in pairs:
        observations, split_skipped, protocol_skipped, cell_skipped = (
            _paired_observations(runs, cand, ref, metric_name, extractor))
        if not observations:
            continue
        a = [x for _, x, _ in observations]
        b = [y for _, _, y in observations]
        res = paired_t_test(a, b)
        change = single_factor_change(cand, ref)
        # 只有审计表登记的六条边才可标成 single_factor。仅凭“对称差=1”会把
        # 未登记的单因子对（例如方向写反、或临时拼出的对比）也标成可归因
        # 证据，绕过 SINGLE_FACTOR_CONTRASTS 这个唯一事实来源。
        audited = (cand, ref) in SINGLE_FACTOR_PAIRS
        is_single = bool(audited and change)
        res.update({"candidate": cand, "reference": ref, "metric": metric_name,
                    "lower_is_better": bool(lower_better),
                    # 单因素标记进入 significance.json：只有单因素对比可用于
                    # 组件归因；vs standard / LEGA vs qgate 等整体对比即使显著
                    # 也不能把差异归给某一个 A/B/C 因子。
                    "contrast_type": ("single_factor" if is_single else "overall"),
                    "changed_factors": ([change] if is_single else [])})
        if split_skipped:
            res["excluded_split_mismatches"] = split_skipped
        if protocol_skipped:
            res["excluded_protocol_mismatches"] = protocol_skipped
        if cell_skipped:
            res["excluded_cell_mismatches"] = cell_skipped
        out.append(_enrich_paired(res, a, b))
    return out


def _multiverse_spec(spec_id, cand, ref, metric, dimension, choice, observations,
                     lower_better, reference=False, note=None):
    """构造一个规格并附上对比身份；保持每项规格的配对种子顺序。"""
    spec = paired_specification(
        spec_id, dimension, choice,
        [x for _, x, _ in observations], [y for _, _, y in observations],
        lower_is_better=lower_better, metric=metric, alpha=ALPHA,
        min_seeds=MIN_SEEDS, note=note)
    spec.update({"candidate": cand, "reference": ref,
                 "reference_spec": bool(reference)})
    return spec


def _multiverse_analysis(runs, pairs, alpha=ALPHA, min_seeds=MIN_SEEDS):
    """对每个对比/指标枚举可辩护的分析选择并汇总规格曲线。

    当前纳入的维度：
    * ``metric_scale``：PPL 原始配对差 vs log-PPL（几何均值比）；
    * ``seed_subset``：全部种子 vs 逐个留一（检验结论是否由单一种子决定）；
    * ``aggregation``：RULER macro（预注册口径）vs micro（样例加权）。

    这些规格高度相关，因此**不做多重校正**，也不改变任何确认性判定；
    只回答“换一个同样合理的分析口径，方向/显著性是否还在”。
    """
    outcomes = []
    for cand, ref in pairs:
        ppl_obs, *_ = _paired_observations(
            runs, cand, ref, "PPL", lambda slot: _ppl_of(slot.get("eval")))
        if ppl_obs:
            specs = [_multiverse_spec(
                "PPL:raw:all", cand, ref, "PPL", "metric_scale", "raw",
                ppl_obs, True, reference=True,
                note="预注册主口径：原始 PPL 配对差")]
            log_obs = [(s, math.log(a), math.log(b))
                       for s, a, b in ppl_obs if a > 0 and b > 0]
            specs.append(_multiverse_spec(
                "PPL:log:all", cand, ref, "PPL", "metric_scale", "log",
                log_obs, True,
                note="log-PPL 配对差（几何均值比尺度）"))
            for seed, _, _ in ppl_obs:
                specs.append(_multiverse_spec(
                    f"PPL:raw:drop_seed{seed}", cand, ref, "PPL",
                    "seed_subset", f"drop_seed={seed}",
                    [row for row in ppl_obs if row[0] != seed], True,
                    note=f"留一稳健性：删除种子 {seed}"))
            outcomes.append(specification_curve(
                specs, reference_id="PPL:raw:all", alpha=alpha,
                min_seeds=min_seeds))

        ruler_obs, *_ = _paired_observations(
            runs, cand, ref, "RULER平均",
            lambda slot: _ruler_avg_of(slot.get("ruler")))
        if ruler_obs:
            specs = [_multiverse_spec(
                "RULER:macro:all", cand, ref, "RULER平均", "aggregation",
                "macro", ruler_obs, False, reference=True,
                note="预注册口径：任务等权、任务内上下文档位等权")]
            micro_obs, *_ = _paired_observations(
                runs, cand, ref, "RULER(micro)",
                lambda slot: _ruler_avg_micro(slot.get("ruler")))
            specs.append(_multiverse_spec(
                "RULER:micro:all", cand, ref, "RULER平均", "aggregation",
                "micro", micro_obs, False,
                note="样例加权口径（按每格 n 加权）"))
            for seed, _, _ in ruler_obs:
                specs.append(_multiverse_spec(
                    f"RULER:macro:drop_seed{seed}", cand, ref, "RULER平均",
                    "seed_subset", f"drop_seed={seed}",
                    [row for row in ruler_obs if row[0] != seed], False,
                    note=f"留一稳健性：删除种子 {seed}"))
            outcomes.append(specification_curve(
                specs, reference_id="RULER:macro:all", alpha=alpha,
                min_seeds=min_seeds))

    counts = {}
    for outcome in outcomes:
        status = outcome.get("status")
        counts[status] = counts.get(status, 0) + 1
    return {
        "method": ("分析选择多重宇宙/specification curve：同一对比在不同"
                   "可辩护分析口径下重算配对效应，报告方向一致率、同向"
                   "显著率与翻转规格；描述性，不做多重校正"),
        "alpha": alpha,
        "min_seeds": min_seeds,
        "counts": counts,
        "outcomes": outcomes,
        "note": ("规格之间相关且未穷尽所有分析选择；“全部规格同向显著”"
                 "不等于证明稳健或达到实际重要差异；方向翻转/不显著只"
                 "要求同时披露，不改变预注册判定；PPL log 规格与 RULER "
                 "micro 规格是尺度/聚合选择，不能据此挑选有利口径。"),
    }


def _adjust_equivalence_family(group, min_seeds=MIN_SEEDS,
                               alpha=ALPHA) -> None:
    """在同一指标家族内对 TOST p 做 Holm 校正（独立于优效性检验族）。

    为什么单独校正：等价性假设（|Δ|<SESOI）与优效性假设（Δ=0）是两类不同
    的推断；把两者混进同一 Holm 家族会双重惩罚。这里只对“有 SESOI 定义且
    n>=min_seeds”的对比计数，n 不足的对比不占家族大小（p 置 None，但仍保留
    原始 TOST p 供描述）。敏感性网格的每个尺度也各自在同一家族内校正，
    否则 0.5×/1×/2× 的“等价/不等价”会受未校正的网格搜索影响。
    """
    eq_tests = [t for t in group if isinstance(t.get("equivalence"), dict)]
    if not eq_tests:
        return

    def _n_ok(obj):
        if not isinstance(obj, dict) or obj.get("not_applicable") is True:
            return False
        n = obj.get("n")
        return (isinstance(n, int) and not isinstance(n, bool)
                and n >= min_seeds)

    pads = holm_bonferroni([
        (t["equivalence"].get("p_tost") if _n_ok(t["equivalence"]) else None)
        for t in eq_tests])
    # 非劣性是单侧假设，必须与 TOST 分开校正：把 TOST 的校正 p 拿来当作
    # 非劣性证据会既漏报（单侧 p 更小）又谎报（校正后不显著仍沿用原始
    # noninferior 标志）。这里对同一家族内的单侧 p 独立做 Holm。
    noninf_pads = holm_bonferroni([
        (t["equivalence"].get("p_noninf_worse")
         if _n_ok(t["equivalence"]) else None)
        for t in eq_tests])
    for t, padj, noninf_adj in zip(eq_tests, pads, noninf_pads):
        eq = t["equivalence"]
        eq["p_tost_adj"] = padj
        eq["p_noninf_worse_adj"] = noninf_adj
        eq["noninferior_adjusted"] = True
        eq["noninferior"] = bool(
            _n_ok(eq) and noninf_adj is not None and noninf_adj < alpha)
        eq["multiplicity_adjusted"] = padj is not None

    n_grid = 0
    for t in eq_tests:
        sens = t["equivalence"].get("sensitivity")
        if isinstance(sens, list):
            n_grid = max(n_grid, len(sens))
    for j in range(n_grid):
        cells = []
        for t in eq_tests:
            sens = t["equivalence"].get("sensitivity")
            if (isinstance(sens, list) and len(sens) > j
                    and isinstance(sens[j], dict)):
                cells.append(sens[j])
        if not cells:
            continue
        adjs = holm_bonferroni([
            (c.get("p_tost") if _n_ok(c) else None) for c in cells])
        for c, padj in zip(cells, adjs):
            c["p_tost_adj"] = padj
            c["multiplicity_adjusted"] = padj is not None
            if c.get("status") == "invalid":
                # 不可表示/损坏的敏感性档（如 2× 绝对 margin 溢出）没有
                # 可检验的 p；不能把它改写成“不等价”，否则会把“无法计算”
                # 冒充成反对等价的证据。
                c["equivalent"] = False
                c["status"] = "invalid"
                continue
            c["equivalent"] = bool(
                _n_ok(c) and padj is not None and padj < alpha)
            c["status"] = ("equivalent" if c["equivalent"]
                           else "insufficient_seeds" if not _n_ok(c)
                           else "not_equivalent")


def _apply_holm(tests, min_seeds=MIN_SEEDS, alpha=ALPHA) -> list:
    """对同一指标家族内的全部配对检验做 Holm–Bonferroni 校正并给出结论。

    汇总同时检验几十组对比；不校正时 FWER 随检验数上升，名义 p<0.05 极易由
    运气产生。结论一律基于同一指标家族内校正后的 p_adj；原始 p 仍保留在
    结果 JSON 中供查阅。

    置换检验的 p_perm 与符号检验的 p_sign 各自在同一指标家族内单独做 Holm
    校正（它们与 t 检验是独立的检验族，合并会双重惩罚）：t 与置换结论是否
    一致被显式记录为 t_perm_agree；t 显著而置换不显著（小样本下 t 的正态
    假设失效的典型症状）时在结论里追加警示，而不是让读者只看到一个 p 值。
    符号检验只作稳健性对照，不参与确认性判定（nz<6 时其双侧精确 p 的
    下限 2/2^nz>0.05，不显著可能只是粒度所限）。
    """
    tests = [t for t in (tests if isinstance(tests, list) else [])
             if isinstance(t, dict)]
    by_metric = {}
    for t in tests:
        by_metric.setdefault(t.get("metric"), []).append(t)
    for group in by_metric.values():
        _adjust_equivalence_family(group, min_seeds=min_seeds, alpha=alpha)
        pads = holm_bonferroni([t.get("p") for t in group])
        ppads = holm_bonferroni([t.get("p_perm") for t in group])
        psads = holm_bonferroni([t.get("p_sign") for t in group])
        for t, padj, ppadj, psadj in zip(group, pads, ppads, psads):
            t["p_adj"] = padj
            t["p_perm_adj"] = ppadj
            t["p_sign_adj"] = psadj
            n = t.get("n")
            n_valid = (isinstance(n, int) and not isinstance(n, bool)
                       and n >= 0)
            p_norm = _finite_number(t.get("p"))
            p_valid = (p_norm is not None
                       and 0.0 <= p_norm <= 1.0)
            if t.get("p") is None:
                t["verdict"] = f"无法检验（n={n} 或方差退化），不作判定"
            elif not p_valid:
                # holm_bonferroni 会把 NaN/Inf/bool/越界/不可比较的 p 归一化
                # 为 None；此时不能顺着“p_adj=None”落到“不显著”，否则损坏
                # 的 p 值会被报告成“没有效应”。
                t["verdict"] = (
                    f"无法检验（p 值非法或不可解析 p={t.get('p')!r}），"
                    "不作判定")
            elif not n_valid:
                # 汇总文件可能来自旧版本/手工编辑：缺 n 不能让整个报告
                # KeyError 崩溃，也不能默认满足最小种子数而给出显著性结论。
                t["verdict"] = f"无法检验（样本量缺失或非法 n={n!r}），不作判定"
            elif n < min_seeds:
                t["verdict"] = f"种子不足（n={n}<{min_seeds}），无法判定"
            elif t.get("no_direction"):
                if t.get("p_adj") is not None and t["p_adj"] < alpha:
                    t["verdict"] = ("显著偏离可加性（Holm 校正后；"
                                    "方向见 Δ 符号）")
                    if ppadj is not None and ppadj >= alpha:
                        t["verdict"] += "；⚠ 置换检验不显著（见注）"
                else:
                    t["verdict"] = ("未检出显著交互（Holm 校正后 p="
                                    f"{t['p_adj']:.4f}）"
                                    if t.get("p_adj") is not None
                                    else "未检出显著交互")
            elif t.get("p_adj") is not None and t["p_adj"] < alpha:
                better = ((t["mean_diff"] < 0)
                          if t.get("lower_is_better") is True
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
                        or not n_valid or n < min_seeds)
                else (t["p_adj"] < alpha) == (ppadj < alpha))
            note = _noise_verdict_note(t)
            if note and isinstance(t.get("verdict"), str):
                t["verdict"] += note
            # 等价性 status 依赖优效性 p_adj：必须在 p_adj 写入后重算。
            eq = t.get("equivalence")
            if isinstance(eq, dict):
                eq.update(interpret_equivalence(
                    eq, t.get("p_adj"), alpha=alpha, min_seeds=min_seeds))
    return tests


# 组件有效性判定所用的检验对：组件 -> (去一法检验, 全因子检验, 主指标)。
# 去一法: Full−X vs Full；全因子: 仅 X vs gate_only（已有门但无 A/B/C）。
# 主指标按 preregistration/hypotheses.json 的登记口径固定：A/B 在语言建模
# PPL，C 在长上下文 RULER。另一指标只作探索性补充，不能用来“换个指标再
# 判一次”以规避多重比较（两族分别 Holm 后取 OR 会放大假阳性）。


def _find_test(tests, pair, metric):
    for t in (tests if isinstance(tests, list) else []):
        if (isinstance(t, dict)
                and (t.get("candidate"), t.get("reference")) == pair
                and t.get("metric") == metric):
            return t
    return None


def _effect_status(t, removing: bool) -> dict:
    """单条配对检验 -> “组件有用”的证据状态（须方向正确、显著且稳健）。

    removing=True（去一法）：候选=去掉组件的 Full−X，显著变差才算证据；
    removing=False（全因子）：候选=仅含该组件的 X，显著变好才算证据。
    opposite=True 表示检验显著但方向相反——这是“去掉反而更好/加入反而
    更差”的反证，不能被同一组件的另一套消融静默抵消。

    显著不等于稳健：若 |Δ|<ΔSD（效应落在逐种子配对差噪声内）或删掉任一
    单个种子就翻转均值符号，即使 Holm 校正后显著也标 fragile，不算
    “组件有用”的证据，更不能进入 confirmatory_supported。
    """
    if t is None:
        return {"supported": False, "opposite": False, "fragile": False,
                "reason": "无对应检验（变体未运行）", "p_adj": None}
    n = _integral_value(t.get("n"))
    if n is None:
        return {"supported": False, "opposite": False, "fragile": False,
                "reason": f"种子数缺失或非法（n={t.get('n')!r}）",
                "p_adj": t.get("p_adj")}
    if n < MIN_SEEDS:
        return {"supported": False, "opposite": False, "fragile": False,
                "reason": f"种子不足（n={n}）", "p_adj": t.get("p_adj")}
    if (t.get("mean_diff") is None or t.get("p_adj") is None
            or t.get("p_adj") >= ALPHA):
        reason = "Holm 校正后不显著"
        if t.get("within_noise"):
            reason += "（效应在种子噪声内）"
        elif t.get("below_mde"):
            reason += "（效应低于 MDE）"
        return {"supported": False, "opposite": False, "fragile": False,
                "reason": reason, "p_adj": t.get("p_adj")}
    lb = t.get("lower_is_better") is True
    if removing:
        evidence = (t["mean_diff"] > 0) if lb else (t["mean_diff"] < 0)
    else:
        evidence = (t["mean_diff"] < 0) if lb else (t["mean_diff"] > 0)
    if not evidence:
        return {"supported": False, "opposite": True, "fragile": False,
                "reason": "方向相反（去掉反而更好/加入反而更差）",
                "p_adj": t.get("p_adj")}
    fragile_reasons = []
    if t.get("within_noise"):
        fragile_reasons.append("|Δ|<ΔSD（种子噪声内）")
    if t.get("loo_sign_stable") is False:
        fragile_reasons.append("留一方向翻转")
    if fragile_reasons:
        return {"supported": False, "opposite": False, "fragile": True,
                "reason": "显著但不稳健（" + "；".join(fragile_reasons) + "）",
                "p_adj": t.get("p_adj")}
    return {"supported": True, "opposite": False, "fragile": False,
            "reason": "显著且稳健", "p_adj": t.get("p_adj")}


def _combine_component_evidence(loo: dict, add: dict) -> dict:
    """合并去一法与全因子两套消融证据，显式处理方向冲突与稳健性。

    规则（与汇总页注释一致）：
      * 两套均稳健支持 -> 有证据支持（一致）；
      * 仅一套稳健支持、另一套不显著 -> 有证据支持（另一套未检出，可能功效不足）；
      * 仅一套稳健支持、另一套显著但不稳健 -> 仍有证据，但标 fragile，
        不能作为确认性结论（confirmatory_supported=False）；
      * 任一套显著反向 -> 证据冲突（即使另一套支持也不能宣称有用）；
      * 都无显著证据；或仅有显著但不稳健的证据 -> 证据不足/不稳健。
    """
    # 任一预注册主指标上的显著反向本身就是反证；另一套“未检出”不能把
    # 它稀释成“证据不足”。只有两套都不显著/都不支持时才是证据不足。
    if loo.get("opposite") or add.get("opposite"):
        conflict = True
    else:
        conflict = False
    if conflict:
        return {"has_evidence": False, "consistent": False, "fragile": False,
                "verdict": "证据冲突（一套消融显著反向）"}
    if loo.get("supported") and add.get("supported"):
        return {"has_evidence": True, "consistent": True, "fragile": False,
                "verdict": "有证据支持（两套消融一致且稳健）"}
    if loo.get("supported"):
        if add.get("fragile"):
            return {"has_evidence": True, "consistent": False,
                    "fragile": True,
                    "verdict": "有证据支持（仅去一法稳健；全因子显著但"
                               "落在噪声内/留一翻转）"}
        return {"has_evidence": True, "consistent": False, "fragile": False,
                "verdict": "有证据支持（仅去一法；全因子未检出）"}
    if add.get("supported"):
        if loo.get("fragile"):
            return {"has_evidence": True, "consistent": False,
                    "fragile": True,
                    "verdict": "有证据支持（仅全因子稳健；去一法显著但"
                               "落在噪声内/留一翻转）"}
        return {"has_evidence": True, "consistent": False, "fragile": False,
                "verdict": "有证据支持（仅全因子；去一法未检出）"}
    if loo.get("fragile") or add.get("fragile"):
        return {"has_evidence": False, "consistent": False, "fragile": True,
                "verdict": "证据不稳健（显著但落在种子噪声内/留一方向翻转）"}
    return {"has_evidence": False, "consistent": False, "fragile": False,
            "verdict": "证据不足"}


def component_evidence(tests, audit: dict = None) -> list:
    """由配对检验结果汇总各组件（A/B/C）是否“有用”的统计证据。

    统计口径：
      * 判定只用预注册主指标（A/B=PPL，C=RULER），另一指标作为探索性
        补充展示——两族分别 Holm 后再取 OR 等于给同一主张两次显著性机会；
      * 去一法（Full−X 显著变差）与全因子（X-only 显著优于 gate_only）
        两套证据合并且检测方向冲突；gate_only 只含门控偏置 d、不含
        A/B/C，避免把“输出门本身”算作组件收益；
      * 任一套显著反向时判“证据冲突”，不再宣称该组件有用；
      * 显著但 |Δ|<ΔSD 或留一方向翻转的证据标为 fragile：最多只能算
        探索性线索，不能升级为“组件有用”的确认性结论。
      * 早停导致实际训练量不等时，统计检验本身仍可计算，但“组件有用”
        不再具备确认性：每条证据同时带 training_budget_comparable /
        confirmatory_supported / evidence_status，供机器消费时不会把
        探索性结果误读成确认结论。
    """
    audit = _as_dict(audit)
    stopped = _as_text_list(audit.get("early_stop_runs"))
    budget_ok = not stopped
    out = []
    for name, loo_pair, add_pair, primary in COMPONENT_TESTS:
        item = {"component": name, "loo_pair": list(loo_pair),
                "add_pair": list(add_pair), "primary_metric": primary,
                "loo": {}, "add": {}, "supported": [],
                "training_budget_comparable": budget_ok,
                "fragile": False,
                "confirmatory_supported": False,
                "evidence_status": "none"}
        if stopped:
            item["budget_note"] = (
                "早停使实际训练量不等（" + "；".join(stopped[:4])
                + (" ..." if len(stopped) > 4 else "")
                + "）：组件证据仅供探索，不能作为确认性结论")
            item["early_stop_runs"] = stopped
        for metric in _EVIDENCE_METRICS:
            item["loo"][metric] = _effect_status(
                _find_test(tests, loo_pair, metric), True)
            item["add"][metric] = _effect_status(
                _find_test(tests, add_pair, metric), False)
        loo_p = item["loo"][primary]
        add_p = item["add"][primary]
        combined = _combine_component_evidence(loo_p, add_p)
        if combined["has_evidence"]:
            item["supported"].append(primary)
        item.update(combined)
        item["confirmatory_supported"] = bool(
            combined["has_evidence"] and budget_ok
            and not combined.get("fragile"))
        if item["confirmatory_supported"]:
            item["evidence_status"] = "confirmatory"
        elif combined["has_evidence"] or combined.get("fragile") or not budget_ok:
            item["evidence_status"] = "exploratory"
        item["evidence"] = ({primary: {
            "direction": ("loo" if loo_p["supported"] else "add"),
            "p_adj": (loo_p if loo_p["supported"] else add_p).get("p_adj")}}
            if combined["has_evidence"] else {})
        item["secondary"] = {
            metric: {"exploratory": True,
                     "loo_supported": item["loo"][metric]["supported"],
                     "add_supported": item["add"][metric]["supported"]}
            for metric in _EVIDENCE_METRICS if metric != primary}
        out.append(item)
    return out
