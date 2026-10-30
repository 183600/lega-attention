"""汇总报告渲染（第三部分：检验表与 JSON 落盘）。"""
import io

from .._shared import _install as _shared_install

_shared_install(globals())


def _variance_share_str(comp, key):
    """把方差占比格式化为百分比；缺失/非有限值显示为 '-'。"""
    value = comp.get(key) if isinstance(comp, dict) else None
    if not isinstance(value, (int, float)) or not np.isfinite(value):
        return "-"
    return f"{value * 100:.1f}%"


def _variance_advice(comp):
    """按均值方差占比给出“补种子/补评测单元”的探索性建议。"""
    seed_share = comp.get("share_mean_seed") if isinstance(comp, dict) else None
    unit_share = comp.get("share_mean_unit") if isinstance(comp, dict) else None
    if (not isinstance(seed_share, (int, float))
            or not isinstance(unit_share, (int, float))
            or not np.isfinite(seed_share) or not np.isfinite(unit_share)):
        return "-"
    if seed_share >= 0.65:
        return "优先补种子"
    if unit_share >= 0.65:
        return "优先补评测单元"
    return "两者都需增加"


def _summary_part_c(**kw):
    if "_lat_metric_at" in kw:
        _lat_metric_at = kw["_lat_metric_at"]
    if "_prefill_at" in kw:
        _prefill_at = kw["_prefill_at"]
    if "audit" in kw:
        audit = kw["audit"]
    if "bad_cross" in kw:
        bad_cross = kw["bad_cross"]
    budget_sensitivity = kw.get("budget_sensitivity")
    if "capacity_only_test" in kw:
        capacity_only_test = kw["capacity_only_test"]
    if "causal_tests" in kw:
        causal_tests = kw["causal_tests"]
    matched_ablation_contrasts = (kw.get("matched_ablation_contrasts") or [])
    dose_response = kw.get("dose_tests")
    if "ci" in kw:
        ci = kw["ci"]
    if "ci_s" in kw:
        ci_s = kw["ci_s"]
    if "common_ctx" in kw:
        common_ctx = kw["common_ctx"]
    if "comp" in kw:
        comp = kw["comp"]
    if "d" in kw:
        d = kw["d"]
    if "delta" in kw:
        delta = kw["delta"]
    if "design_conf" in kw:
        design_conf = kw["design_conf"]
    if "design_plan" in kw:
        design_plan = kw["design_plan"]
    test_reuse = kw.get("test_reuse")
    if "e" in kw:
        e = kw["e"]
    if "extra" in kw:
        extra = kw["extra"]
    if "hyp_eval" in kw:
        hyp_eval = kw["hyp_eval"]
    if "inference_interactions" in kw:
        inference_interactions = kw["inference_interactions"]
    if "inference_shapley" in kw:
        inference_shapley = kw["inference_shapley"]
    if "integrity" in kw:
        integrity = kw["integrity"]
    if "k" in kw:
        k = kw["k"]
    if "lat_entries" in kw:
        lat_entries = kw["lat_entries"]
    if "lat_variants" in kw:
        lat_variants = kw["lat_variants"]
    if "length_analysis" in kw:
        length_analysis = kw["length_analysis"]
    if "log_tests" in kw:
        log_tests = kw["log_tests"]
    if "loo" in kw:
        loo = kw["loo"]
    if "md" in kw:
        md = kw["md"]
    if "ngram_cmp" in kw:
        ngram_cmp = kw["ngram_cmp"]
    if "p_adj" in kw:
        p_adj = kw["p_adj"]
    if "padj" in kw:
        padj = kw["padj"]
    if "param_control_test" in kw:
        param_control_test = kw["param_control_test"]
    if "ppl_window_tests" in kw:
        ppl_window_tests = kw["ppl_window_tests"]
    if "probe_cmp" in kw:
        probe_cmp = kw["probe_cmp"]
    if "r" in kw:
        r = kw["r"]
    if "ratio" in kw:
        ratio = kw["ratio"]
    if "registered" in kw:
        registered = kw["registered"]
    multiverse = kw.get("multiverse")
    reparameterization = kw.get("reparameterization")
    if "results_dir" in kw:
        results_dir = kw["results_dir"]
    if "rows" in kw:
        rows = kw["rows"]
    if "rt" in kw:
        rt = kw["rt"]
    if "ruler_sample_tests" in kw:
        ruler_sample_tests = kw["ruler_sample_tests"]
    if "runs" in kw:
        runs = kw["runs"]
    if "s" in kw:
        s = kw["s"]
    if "scope_check" in kw:
        scope_check = kw["scope_check"]
    if "sf_audit" in kw:
        sf_audit = kw["sf_audit"]
    if "shapley_tests" in kw:
        shapley_tests = kw["shapley_tests"]
    if "shown" in kw:
        shown = kw["shown"]
    if "t" in kw:
        t = kw["t"]
    if "tests" in kw:
        tests = kw["tests"]
    if "v" in kw:
        v = kw["v"]
    if "verdict" in kw:
        verdict = kw["verdict"]
    if "w" in kw:
        w = kw["w"]
    if "win_cross" in kw:
        win_cross = kw["win_cross"]
    if "wl" in kw:
        wl = kw["wl"]

    md += ["", "## 关键对比（同种子配对，t 检验 + 置换检验双重印证，Holm 校正）", ""]
    if tests:
        md.append("| 对比 | 类型 | 指标 | Δ均值 ± Δ标准差 | Δ中位数 | Δ 95% CI | d_z | "
                  "n | 胜/负/平 | 留一稳定 | p | p_adj(Holm) | "
                  "p_perm(Holm) | p_sign(Holm) | MDE | 结论 |")
        md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for t in tests:
            if t.get("interaction"):
                continue  # 交互效应单独成表，避免与主效应表重复
            if t.get("ruler_task"):
                continue  # 逐任务分解单独成表，避免主表被 5 个任务淹没
            if t.get("metric") == "PPL(log)":
                continue  # 比值尺度敏感性单独成表，避免与原始 PPL 重复
            if t.get("shapley_factor"):
                continue  # Shapley 分解单独成表，避免与单因素边重复
            if t.get("mean_diff") is None:
                continue
            if t.get("contrast_type") == "single_factor":
                typ = ("单因素 " + "/".join(t.get("changed_factors") or []))
            elif t.get("metric") in (AUROC_DIFF_METRIC, AP_DIFF_METRIC):
                typ = "同 run 机制对比"
            else:
                typ = "整体/多因素"
            std = t.get("std_diff")
            delta = (f"{t['mean_diff']:+.4f}"
                     + (f" ± {std:.4f}" if std is not None else ""))
            med = (f"{t['median_diff']:+.4f}"
                   if t.get("median_diff") is not None else "-")
            ci = t.get("mean_diff_ci95")
            ci_s = (f"[{ci[0]:+.4f}, {ci[1]:+.4f}]"
                    if ci else "-")
            p = f"{t['p']:.4f}" if t.get("p") is not None else "-"
            padj = (f"{t['p_adj']:.4f}" if t.get("p_adj") is not None else "-")
            ppadj = (f"{t['p_perm_adj']:.4f}"
                     if t.get("p_perm_adj") is not None else "-")
            psadj = (f"{t['p_sign_adj']:.4f}"
                     if t.get("p_sign_adj") is not None else "-")
            dz = f"{t['d_z']:+.2f}" if t.get("d_z") is not None else "-"
            mde = (f"{t['mde']:+.4f}" if t.get("mde") is not None else "-")
            wl = _seed_winloss_str(t)
            loo = _loo_stability_str(t)
            md.append(f"| {t['candidate']} vs {t['reference']} | {typ} "
                      f"| {t['metric']} | "
                      f"{delta} | {med} | {ci_s} | {dz} | {t['n']} | {wl} | "
                      f"{loo} | {p} | {padj} | {ppadj} | {psadj} | {mde} | "
                      f"{t.get('verdict', '-')} |")
        md.append("")
        md.append("注: **单因素** = 候选与参照只差一个门控因子（去一法 Full−X "
                  "vs Full，或全因子增量 X-only vs gate_only），只有这类对比"
                  "可用于组件归因；**整体/多因素**（各变体 vs standard、"
                  "LEGA vs qgate 等）回答总体性能，不能把差异归给单个组件。")
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
                  "稳健性警报。Δ中位数为逐种子配对差中位数（对离群种子不敏感）；"
                  "p_sign 为忽略零差后的精确符号检验，在同一指标家族内独立"
                  "Holm 校正，仅作稳健性对照、不参与确认性判定：有效非零"
                  "配对数 nz<6 时其双侧精确 p 的下限 2/2^nz>0.05，"
                  "“不显著”可能只是粒度所限。Δ 95% CI 为配对差的 bootstrap 区间；MDE = 当前"
                  "种子数与方差下 80% 功效可检测的最小效应：观测差远小于 "
                  "MDE 的“不显著”是功效不足，不是无效应。效应/噪声比 = "
                  "|Δ|/ΔSD（ΔSD 为逐种子配对差的标准差）：|Δ|<ΔSD 的结果"
                  "在结论中标注为“种子/评测噪声内”的边缘证据，即使 p_adj "
                  "显著也不能当作稳健收益。")
        md.append("注: “胜/负/平”= 逐种子配对差为正/负/零的种子计数（不含任何"
                  "分布假设）；“留一稳定”= 删掉任一单个种子后均值符号不变，"
                  "❌ 表示结论由个别种子决定，即使 p_adj 显著也不稳健。")
    else:
        md.append("（无可配对的对比；请对多个变体运行同一组种子）")

    # --- SESOI/TOST：把“不显著”与“实际等价”分开。
    md += _render_equivalence_section(tests)

    # --- PPL 比值尺度敏感性（探索性）：log-PPL 几何均值比。
    log_tests = [t for t in tests
                 if t.get("metric") == "PPL(log)"
                 and t.get("mean_diff") is not None]
    if log_tests:
        md += ["", "### PPL 比值尺度敏感性（log-PPL 几何均值比；探索性）", "",
               "PPL 是比值尺度，算术均值差会被高 PPL 种子/变体主导；log-PPL "
               "配对差 Δ=mean(log PPL候选−log PPL参照) 的指数 exp(Δ) 即几何"
               "均值比（<1 表示候选更好），区间由配对差 bootstrap CI 取指数"
               "得到。该表只检验结论是否依赖尺度；预注册主判定仍是原始 PPL，"
               "不能在两种尺度中挑对主张有利的口径。", "",
               "| 对比 | n | ΔlogPPL | PPL几何比 | 比值95% CI | p_adj(Holm) | "
               "SESOI等价性(±1%) | 原始PPL判定 | 尺度结论 |",
               "|---|---|---|---|---|---|---|---|---|"]
        n_mismatch = 0
        for t in log_tests:
            raw = _find_test(tests, (t.get("candidate"), t.get("reference")),
                             "PPL")
            raw_verdict = (_verdict_head(raw.get("verdict"))
                           if isinstance(raw, dict) else "未运行")
            ci = t.get("ppl_ratio_ci95")
            ci_s = (f"[{ci[0]:.4f}, {ci[1]:.4f}]" if ci else "-")
            ratio = (f"{t['ppl_ratio']:.4f}"
                     if t.get("ppl_ratio") is not None else "-")
            padj = (f"{t['p_adj']:.4f}" if t.get("p_adj") is not None else "-")
            log_verdict = _verdict_head(t.get("verdict"))
            mismatch = (raw_verdict != log_verdict
                        and raw_verdict not in ("未运行", "未定")
                        and log_verdict != "未定")
            n_mismatch += int(mismatch)
            mark = ("⚠ 尺度结论不一致" if mismatch else log_verdict)
            eq = t.get("equivalence") if isinstance(t.get("equivalence"), dict) else {}
            eq_p = _eq_p_str(eq) if eq else "-"
            eq_s = ("-" if not eq else
                    f"{eq.get('label') or '-'}"
                    + ("" if eq_p == "-" else f"（p={eq_p}）"))
            md.append(
                f"| {t['candidate']} vs {t['reference']} | {t['n']} | "
                f"{_fmt_signed(t.get('mean_diff'))} | {ratio} | {ci_s} | {padj} | "
                f"{eq_s} | {raw_verdict} | {mark} |")
        md += ["", "注: 几何均值比只改变“平均”的尺度，不改变配对结构；"
               f"原始 PPL 与 log-PPL 的结论{('有' if n_mismatch else '没有')}"
               f"不一致（{n_mismatch} 组），不一致时必须同时报告并解释为"
               "尺度敏感性，不能静默选择其中一个。逐种子 log 差与几何比见 "
               "`significance.json` 的 `ppl_log_pairs`。"]

    # --- 分析选择多重宇宙（探索性）：结论是否依赖原始/log、macro/micro、
    #     全种子/留一这些同样可辩护的口径。
    md += _render_multiverse_section(multiverse)

    # --- PPL 窗口级配对（探索性）：有限窗口抽样的评估不确定性。
    #     与上面的种子级 t 检验回答不同问题，不能互相替代。
    md += ["", "### PPL 窗口级配对（评估抽样不确定性；探索性稳健性）", "",
           "PPL 只在 `eval_iters` 个等间隔窗口上估计，种子级检验只看每个种子的"
           "聚合 PPL。各变体共享同一批窗口序号，以下逐窗口配对 `loss_candidate−"
           "loss_reference`，并用“先重采样种子、再重采样窗口”的分层 bootstrap "
           "给出平均 NLL 差的区间；窗口间空间相关被忽略，故只作探索性证据。"
           "`PPL比`=exp(ΔNLL)，<1 表示候选更好。"
           f"低于 {MIN_PPL_WINDOW_SEEDS} 个种子或每种子 {MIN_PPL_WINDOWS} "
           "个窗口时只报点估计/区间，不生成“显著”措辞（重采样自由度不足）。", ""]
    if ppl_window_tests:
        md += ["| 对比 | 种子 | 窗口数(最小–最大) | ΔNLL均值 | 分层bootstrap 95% CI | "
               "块bootstrap 95% CI | PPL比 | p_adj(Holm) | 种子级判定（供对照） | 结论 |",
               "|---|---|---|---|---|---|---|---|---|---|"]
        for t in ppl_window_tests:
            seed_t = _find_test(tests, (t.get("candidate"), t.get("reference")),
                                "PPL")
            seed_verdict = (seed_t.get("verdict") if isinstance(seed_t, dict)
                            else "未运行")
            ci = t.get("mean_diff_ci95") or [None, None]
            ci_s = (f"[{ci[0]:+.4f}, {ci[1]:+.4f}]"
                    if ci[0] is not None and ci[1] is not None else "-")
            bci = t.get("block_mean_diff_ci95") or [None, None]
            bci_s = (f"[{bci[0]:+.4f}, {bci[1]:+.4f}]"
                     if bci[0] is not None and bci[1] is not None else "-")
            ratio = (f"{t['ppl_ratio']:.4f}"
                     if t.get("ppl_ratio") is not None else "-")
            padj = (f"{t['p_adj']:.4f}"
                    if t.get("p_adj") is not None else "-")
            md.append(
                f"| {t['candidate']} vs {t['reference']} | {t['n_seeds']} | "
                f"{t['n_windows_min']}–{t['n_windows_max']} | "
                f"{_fmt_signed(t.get('mean_diff'))} | {ci_s} | {bci_s} | "
                f"{ratio} | {padj} | "
                f"{str(seed_verdict).split('（')[0]} | {t.get('verdict', '-')} |")
        md += ["", "注: ΔNLL 为负表示候选的窗口平均交叉熵更低（更好）。"
               "分层 bootstrap 的 p 值由中心化重采样分布计算，与种子级 t "
               "检验不是同一个零假设；块bootstrap 用移动块（块长 n^(1/3)）"
               "保留相邻窗口相关，与 i.i.d. 区间并报，以更宽者解读稳健性。"
               "两者冲突时说明“换训练种子”和“换评测窗口”给出不同信息，"
               "均不足以单独升级为确认性结论。逐种子窗口胜负明细见 "
               "`significance.json` 的 `ppl_window_pairs`。"]

        def _ci_str(ci):
            if (not isinstance(ci, (list, tuple)) or len(ci) != 2):
                return "-"
            lo, hi = _fmt_signed(ci[0]), _fmt_signed(ci[1])
            return f"[{lo}, {hi}]" if lo != "-" and hi != "-" else "-"

        robust_rows = [
            t for t in ppl_window_tests
            if isinstance(t.get("robustness"), dict)
            and t.get("median_diff") is not None]
        if robust_rows:
            md += ["", "**窗口级稳健性：均值是否由少数窗口驱动**", "",
                   "平均 NLL 改善可能只来自少数困难窗口。下表并列逐种子中位数、"
                   "10% 截尾均值与改善窗口占比（ΔNLL<0 的窗口比例），区间均由"
                   "“先重采样种子、再重采样窗口”的分层 bootstrap 得到；移动块"
                   "版本见 `significance.json` 的 `block_*` 字段。这些是描述性"
                   "稳健性标记：均值显著但中位数/截尾均值方向不一致或多数窗口"
                   "未改善时，只能写“均值受少数窗口驱动”，不能写成普遍改善。", "",
                   "| 对比 | ΔNLL均值 | ΔNLL中位数 [95%CI] | "
                   "10%截尾均值 [95%CI] | 改善窗口占比 [95%CI] | 稳健性结论 |",
                   "|---|---|---|---|---|---|"]
            for t in robust_rows:
                rob = t.get("robustness") or {}
                if rob.get("outlier_sensitive"):
                    verdict = "⚠ 均值可能由少数窗口驱动"
                elif rob.get("broad_based"):
                    verdict = "均值/中位数/截尾同向，改善覆盖多数窗口"
                elif rob.get("direction_consistent"):
                    verdict = "方向一致，但中位数区间含 0 或多数窗口未同向"
                else:
                    verdict = "均值与稳健统计方向不一致"
                md.append(
                    f"| {t['candidate']} vs {t['reference']} | "
                    f"{_fmt_signed(t.get('mean_diff'))} | "
                    f"{_fmt_signed(t.get('median_diff'))} "
                    f"{_ci_str(t.get('median_diff_ci95'))} | "
                    f"{_fmt_signed(t.get('trimmed_mean_diff_10pct'))} "
                    f"{_ci_str(t.get('trimmed_mean_diff_10pct_ci95'))} | "
                    f"{_fmt_pct1(t.get('window_win_rate'))} "
                    f"{_ci_str(t.get('window_win_rate_ci95'))} | {verdict} |")
            md += ["", "注: 改善窗口占比为逐种子比例的平均；平局窗口计入分母、"
                   "不计为改善。中位数/截尾均值对极端窗口不敏感，与均值并报"
                   "用于识别“少数大赢窗口拉高平均”的情形；它们不构成新的"
                   "显著性检验，也不改变上表的 Holm 判定。"]
    else:
        md.append("（缺少逐窗口损失（旧产物）或没有同 split/同协议的配对 run；"
                  "重新运行独立评测即可启用。）")

    # --- RULER 逐任务分解（探索性）：只渲染预注册 RULER 假设涉及的对比；
    #     全部逐任务检验仍完整保存在 significance.json（tests[*].ruler_task）。
    task_rows = [t for t in tests if t.get("ruler_task")]
    if task_rows:
        reg_ruler_pairs = {
            (h.get("candidate"), h.get("reference"))
            for h in (_as_dict(registered).get("hypotheses") or [])
            if isinstance(h, dict) and h.get("metric") == "RULER平均"}
        if not reg_ruler_pairs:
            reg_ruler_pairs = {("lega", "standard")}
        shown = [t for t in task_rows
                 if (t.get("candidate"), t.get("reference"))
                 in reg_ruler_pairs]
        md += ["", "### RULER 逐任务分解（探索性；综合平均的辅助诊断）", "",
               "综合平均可能把不同任务的方向相反效应相互抵消，因此对预注册"
               "RULER 对比逐任务各做一次同种子配对检验；每个任务是一个独立"
               "Holm 家族（探索性，不替代综合平均的确认性判定）。", ""]
        if shown:
            md += ["| 对比 | 任务 | Δ均值 ± Δ标准差 | 胜/负/平 | 留一稳定 | n | "
                   "p_adj(Holm) | 结论 |",
                   "|---|---|---|---|---|---|---|---|"]
            for t in shown:
                delta = _fmt_mean_std(t)
                padj = (f"{t['p_adj']:.4f}"
                        if t.get("p_adj") is not None else "-")
                wl = _seed_winloss_str(t)
                loo = _loo_stability_str(t)
                md.append(f"| {t['candidate']} vs {t['reference']} | "
                          f"{t['ruler_task']} | {delta} | {wl} | {loo} | "
                          f"{t['n']} | {padj} | {t.get('verdict', '-')} |")
        else:
            md.append("（预注册 RULER 对比未运行，无法分解）")
        md += ["", "注: 全部逐任务检验（含未渲染的对比）见 "
               "`significance.json` 中 metric 以 `RULER:` 开头的条目；这些"
               "检验未预注册方向，只作机制/定位线索，需在新数据上确认。"
               "“胜/负/平”为逐种子配对差符号计数；“留一稳定”= 删掉任一单个"
               "种子后均值符号不变，❌ 表示结论由个别种子决定。"]

    # --- RULER 样例级配对（探索性稳健性证据）：固定题面上的 McNemar。
    #     跨种子 t 检验只看每个种子的聚合 EM，n=20~30 的评测抽样误差
    #     没有进入检验；同一 sample_id 配对能利用“候选对/参照错”的方向
    #     信息，回答不同的统计问题。它不是确认性检验，也不替代种子级判定。
    md += ["", "### RULER 样例级配对（McNemar 精确检验；探索性稳健性）", "",
           "跨种子聚合 EM 检验回答“换训练种子是否稳定”；每格只有 20~30 个"
           "样例，以下用变体间共享的 `sample_id` 做逐样例配对，回答“在这批"
           "固定题面上候选是否更常答对”。只统计结果不同的样对（候选独对 vs "
           "参照独对），相同结果不提供方向信息；多种子按 discordant 计数合并，"
           "独立 Holm 家族。", ""]
    if ruler_sample_tests:
        md += ["| 对比 | 配对种子 | 配对样例 | 候选独对/参照独对 | "
               "ΔEM micro(候选−参照) | ΔEM macro(任务等权) | "
               "ΔEM macro 95% CI | ΔEM macro 双层95% CI | 种子胜/负/平 | "
               "p_adj(Holm) | 种子级判定（供对照） | 结论 |",
               "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for t in ruler_sample_tests:
            seed_t = _find_test(tests, (t.get("candidate"), t.get("reference")),
                                "RULER平均")
            seed_verdict = (seed_t.get("verdict") if isinstance(seed_t, dict)
                            else "未运行")
            delta = (f"{t['mean_diff']:+.4f}"
                     if t.get("mean_diff") is not None else "-")
            macro_delta = (f"{t['macro_mean_diff']:+.4f}"
                           if t.get("macro_mean_diff") is not None else "-")
            mci = t.get("macro_mean_diff_ci95")
            mci_s = (f"[{mci[0]:+.4f}, {mci[1]:+.4f}]" if mci else "-")
            hci = t.get("macro_mean_diff_ci95_hier")
            hci_s = (f"[{hci[0]:+.4f}, {hci[1]:+.4f}]" if hci else "-")
            padj = (f"{t['p_adj']:.4f}"
                    if t.get("p_adj") is not None else "-")
            md.append(
                f"| {t['candidate']} vs {t['reference']} | {t['n_seeds']} | "
                f"{t['n_samples']} | {t['candidate_only']}/{t['reference_only']} | "
                f"{delta} | {macro_delta} | {mci_s} | {hci_s} | "
                f"{t['n_seed_wins']}/{t['n_seed_losses']}/{t['n_seed_ties']} | "
                f"{padj} | {str(seed_verdict).split('（')[0]} | "
                f"{t.get('verdict', '-')} |")
        md += ["", "注: McNemar 只使用 discordant 对，n_samples 是参与配对的总样例"
               "数而非独立证据数；多个训练种子重复使用同一批 sample_id，合并"
               "检验只作探索性证据。`ΔEM micro` 样例等权、与 McNemar 池化计数"
               "对应；`ΔEM macro` 任务等权、任务内上下文档位等权，与预注册 "
               "`RULER平均` 及上表种子级"
               "判定同一估计量。`ΔEM macro 95% CI` 固定已观测种子，在每个"
               "(任务, 上下文) 单元格内重采样配对样例，再任务内、任务间等权平均，只覆盖"
               "评测抽样不确定性；种子数很少时该区间可能很窄，不能替代种子级"
               "检验。`ΔEM macro 双层95% CI` 同时重采样训练种子与配对样例，"
               "覆盖“换种子且换题面”的合并不确定性（独立 Holm 家族，仍为"
               "探索性）。两者方向不一致时，结论依赖“平均值”定义，不能"
               "只挑对主张有利的口径。若样例级显著而种子级不显著（或反之），"
               "说明两个不确定性来源给出不同信息：前者多为评测样本量/题目"
               "难度，后者为训练随机性，均不能单独升级为确认性结论。"
               "逐种子明细见 `significance.json` 的 `ruler_sample_pairs`。"]
        cluster_rows = [t for t in ruler_sample_tests
                        if t.get("cluster_p") is not None]
        if cluster_rows:
            md += ["", "**样例聚类稳健性（每个 sample_id 先跨种子平均；"
                   "避免同一样例被多个种子重复计数）**", "",
                   "| 对比 | 独立样例 | 聚类ΔEM | 95% CI | p_adj(Holm) | 判定 |",
                   "|---|---|---|---|---|---|"]
            for t in cluster_rows:
                ci = t.get("cluster_mean_diff_ci95") or [None, None]
                ci_s = (f"[{ci[0]:+.4f}, {ci[1]:+.4f}]"
                        if ci[0] is not None and ci[1] is not None else "-")
                padj = (f"{t['cluster_p_adj']:.4f}"
                        if t.get("cluster_p_adj") is not None else "-")
                md.append(
                    f"| {t['candidate']} vs {t['reference']} | "
                    f"{t.get('cluster_n')} | "
                    f"{_fmt_signed(t.get('cluster_mean_diff'))} | "
                    f"{ci_s} | {padj} | {t.get('cluster_verdict', '-')} |")
            md += ["", "注: 聚类检验先把每个 sample_id 在多个训练种子下的配对差"
                   "取平均，再以样例为独立单位做符号翻转（n≤16 精确，否则固定"
                   "种子 Monte-Carlo）。它只覆盖样例等权 micro 口径，不校正"
                   "macro；与池化 McNemar 的差异本身说明“重复种子被当成独立"
                   "证据”的程度，两者应并列解读。"]
    else:
        md.append("（缺少逐样例 EM 或没有协议一致的配对 run；旧产物需重跑 "
                  "RULER 才能启用该分析。）")

    # --- 不确定性来源分解：种子随机性 vs 评测单元抽样。
    variance_rows = []
    for v_label, v_items in (("PPL窗口", ppl_window_tests or []),
                             ("RULER macro", ruler_sample_tests or [])):
        for v_test in v_items:
            v_comp = v_test.get("variance_components")
            if isinstance(v_comp, dict):
                variance_rows.append((v_label, v_test, v_comp))
    if variance_rows:
        md += ["", "### 不确定性来源分解（种子 vs 评测单元；探索性）", "",
               "把每个 (种子, 评测单元) 单元格内的配对差视为二维矩阵，用两因素"
               "随机效应 ANOVA 矩估计分解种子/单元/残差方差。`总方差`占比描述"
               "单次观测的波动；`均值方差`占比已按 1/S、1/U 收缩，才是“补种子"
               "还是补评测单元”的直接依据。负方差分量按 0 截断（小样本常见），"
               "不平衡/缺格不插补，仅用所有种子共有的单元格。", "",
               "| 分析 | 对比 | 种子 | 评测单元/格 | 总方差:种子 | "
               "总方差:单元 | 总方差:残差 | 均值方差:种子 | "
               "均值方差:单元 | 建议 |",
               "|---|---|---|---|---|---|---|---|---|---|"]
        for v_label, v_test, v_comp in variance_rows:
            v_units = v_comp.get("n_units")
            if v_units is None:
                v_units = v_comp.get("n_cells")
            md.append(
                f"| {v_label} | {v_test.get('candidate')} vs "
                f"{v_test.get('reference')} | "
                f"{v_comp.get('n_seeds', '-')} | "
                f"{v_units if v_units is not None else '-'} | "
                f"{_variance_share_str(v_comp, 'share_seed')} | "
                f"{_variance_share_str(v_comp, 'share_unit')} | "
                f"{_variance_share_str(v_comp, 'share_resid')} | "
                f"{_variance_share_str(v_comp, 'share_mean_seed')} | "
                f"{_variance_share_str(v_comp, 'share_mean_unit')} | "
                f"{_variance_advice(v_comp)} |")
        md += ["", "注: 建议阈值（种子/单元均值方差占比 ≥65%）只是资源分配"
               "的探索性提示，不是确认性判定；小种子数下 ANOVA 分量本身不"
               "稳定，应与上方 t 检验、置换检验和双层 bootstrap 区间并读。"
               "PPL 窗口相邻相关会违反单元独立假设，分量只作近似，区间"
               "稳健性仍以移动块 bootstrap 为准。RULER 行显示共有单元格数，"
               "只聚合所有种子共有的 (任务, 上下文) 单元格；被跳过的单元格"
               "数见 `significance.json` 的 "
               "`variance_components.n_cells_skipped`。"]

    # --- 注册假设判定：confirmatory 与 exploratory 的边界必须是显式的
    md += ["", "## 注册假设判定（confirmatory；其余对比均为事后探索）", ""]
    md += _render_registration_integrity(integrity, scope_check=scope_check)
    if hyp_eval:
        n_sup = sum(1 for h in hyp_eval if h["status"] == "支持")
        n_ref = sum(1 for h in hyp_eval if h["status"] == "拒绝")
        n_und = sum(1 for h in hyp_eval if h["status"] == "未定")
        n_untested = sum(1 for h in hyp_eval if h["status"] == "未检验")
        counts = f"支持 {n_sup}、拒绝 {n_ref}、未定 {n_und}"
        if n_untested:
            # “未检验”（对比未运行/指标对不上）与“未定”是不同状态，
            # 必须显式计数，否则四项之和小于登记条数、读者会误以为漏算。
            counts += f"、未检验 {n_untested}"
        valid = all(h.get("confirmatory_valid") for h in hyp_eval)
        md.append(f"预注册文件 `preregistration/hypotheses.json`（登记于正式实验前；"
                  f"不放在 docs/ 以免被流水线自动清理删除）："
                  f"{len(hyp_eval)} 条方向性主张，判定 {counts}。"
                  "只有“支持”能写进结论；“拒绝”= 方向相反且校正后"
                  "显著（假设被证伪）；“未定/未检验”不得解释为任何方向的证据。")
        if not valid:
            md.append("")
            md.append("> ⛔ 预注册完整性校验未通过：下表的“支持/拒绝”不作数，"
                      "已按探索性线索处理（见上方完整性小节）。")
        md.append("")
        md.append("| ID | 主张 | Δ均值 | n | p_adj(家族) | p_adj(全体确认族) | "
                  "p_perm | MDE | 判定 |")
        md.append("|---|---|---|---|---|---|---|---|---|")
        n_global_downgraded = 0
        for h in hyp_eval:
            padj = (f"{h['p_adj']:.4f}" if h.get("p_adj") is not None else "-")
            pg = (f"{h['p_adj_confirmatory']:.4f}"
                  if h.get("p_adj_confirmatory") is not None else "-")
            if (h.get("status_family") in ("支持", "拒绝")
                    and h.get("status") == "未定"
                    and h.get("p_adj_confirmatory") is not None):
                n_global_downgraded += 1
            pp = (f"{h['p_perm_adj']:.4f}"
                  if h.get("p_perm_adj") is not None else "-")
            delta = _fmt_signed(h.get("mean_diff"))
            mde = f"{h['mde']:+.4f}" if h.get("mde") is not None else "-"
            mark = {"支持": "✅", "拒绝": "❌"}.get(h["status"], "—")
            if not h.get("confirmatory_valid"):
                mark = "⚠（无效）"
            md.append(f"| {h['id']} | {h['claim']} | {delta} | "
                      f"{h.get('n') if h.get('n') is not None else '-'} | "
                      f"{padj} | {pg} | {pp} | {mde} | {mark} {h['status']}"
                      f"（{h['detail']}） |")
        md.append("")
        if n_global_downgraded:
            md.append(f"注: {n_global_downgraded} 条主张在指标家族内显著，但"
                      "跨全部预注册假设的全局 Holm 校正后不显著，已按确认族"
                      "整体 FWER 降级为“未定”；status_family 仍保留原家族判定"
                      "供审计。")
        md.append("注: p_adj(家族) 为同一指标家族内 Holm；p_adj(全体确认族) "
                  "为全部可检验预注册假设（n≥"
                  f"{MIN_SEEDS}）组成的确认族 Holm，控制“任意一条确认性结论"
                  "为假阳性”的整体 FWER。判定要求两者均 < "
                  f"{ALPHA}：仅家族内显著不足以升级为确认性结论。"
                  "未登记在本文件里的对比（含本汇总表中的其余行）一律属"
                  "事后（exploratory）分析：p 值只受本表家族内校正保护，"
                  "其“显著”只能作为提出新假设的线索；要升级为结论须先登记"
                  "再在新的种子/数据上检验。")
    else:
        md.append("- ⚠ 未找到可用的注册假设文件（`preregistration/hypotheses.json`）。"
                  "本页全部对比均为事后分析：没有预注册边界，任何“显著”都只能"
                  "按探索性线索对待。请登记方向性主张后重跑正式实验。")

    # --- 门控开销：逐头参数与额外计算必须如实核算，不能只报“涨点”。
    # 参数是实际注册参数（gate_extra_params），FLOPs 是 latency.json 的解析
    # 估计（见 estimate_flops_per_token）；两者都取全部变体公共 ctx，避免
    # 拿不同上下文的 FLOPs 比较。
    md += ["", "## 门控开销核算（逐头参数与额外计算）", ""]
    md.append("门控/空注意力的开销按**逐头参数量**与**解析 FLOPs**分列，"
              "与收益（PPL/RULER）分开报告：参数量取自实际注册参数，"
              "FLOPs 为 batch=1 prefill 的解析估计（matmul 2·MACs + "
              "注意力 4·kv_len·d/层 + 门控项），不是实测延迟。")
    md.append("")
    md.append("| 变体 | 门控参数/头 | 门控额外参数 | 占总参数 | "
              "门控 FLOPs/token | 占总 FLOPs | 说明 |")
    md.append("|---|---|---|---|---|---|---|")
    for r in rows:
        v = r.get("variant")
        per_head = _num_or_none(r.get("n_gate_per_head"))
        extra = _num_or_none(r.get("gate_extra"))
        params = _num_or_none(r.get("params"))
        # 参数/FLOPs 来自外部结果文件：负参数、0/极小分母或负开销都会把
        # “占比”算成 inf/-inf。占比只在两端都是非负有限且分母 >0 时定义，
        # 结果非有限则按缺失处理，不能写进汇总表/JSON。
        param_pct = None
        if (extra is not None and extra >= 0.0
                and params is not None and params > 0.0):
            param_pct = extra / params * 100.0
            if not math.isfinite(param_pct):
                param_pct = None
        e = lat_entries.get(v)
        gate_flops = (_eff_cell(e, "gate_flops", common_ctx)
                      if common_ctx is not None else None)
        total_flops = (_eff_cell(e, "flops", common_ctx)
                       if common_ctx is not None else None)
        flop_pct = None
        if (gate_flops is not None and gate_flops >= 0.0
                and total_flops is not None and total_flops > 0.0):
            flop_pct = gate_flops / total_flops * 100.0
            if not math.isfinite(flop_pct):
                flop_pct = None
        r["gate_param_pct"] = (round(param_pct, 4)
                               if param_pct is not None else None)
        r["gate_flops_per_tok"] = gate_flops
        r["gate_flops_pct"] = (round(flop_pct, 4)
                               if flop_pct is not None else None)
        note = ("无门控（外部基线）" if v in ("standard", "standard_sdpa")
                else "逐头 w/a/b/d 或 null 槽" if (extra or 0) > 0 else "-")
        md.append(
            f"| {VARIANT_CN.get(v, v)} "
            f"| {'-' if per_head is None else f'{per_head:g}'} "
            f"| {'-' if extra is None else f'{extra:,.0f}'} "
            f"| {'-' if param_pct is None else f'{param_pct:.4f}%'} "
            f"| {'-' if gate_flops is None else f'{gate_flops:,.0f}'} "
            f"| {'-' if flop_pct is None else f'{flop_pct:.4f}%'} "
            f"| {note} |")
    md.append("")
    md.append("逐头参数 = 总额外参数 / (层数 × 头数)，来自实际注册的 "
              "gate_w/gate_a/gate_b/gate_d 或 k_null/v_null；LEGA 每头 "
              "d_head+3，qgate 每头 d_head+1，null 每头 2·d_head，"
              "gate_only 每头 1。**收益必须对照这两列开销解读**："
              "“门控参数占比很小”不等于“额外计算可忽略”——集中度项要扫描 "
              "每个头的 max_j p_j，其 FLOPs 随上下文长度线性增长。"
              + ("当前公共 ctx=" + str(common_ctx)
                 + "；不同 ctx 的数值见 latency.json。" if common_ctx
                 else "当前缺少公共 ctx 的 latency 数据，FLOPs 列不可得。"))

    # --- 涨点-增幅权衡：评价改进不能只看指标涨点
    md += ["", "## 涨点-增幅权衡（涨点必须对照增幅）", ""]
    # 运行时基线与上表 prefill_rel_pct 同一策略：优先 standard_sdpa（与全部
    # 候选的 SDPA 推理路径可比），缺失时退回显式 standard。
    # 训练步时与进程峰值两列仍取 standard 训练产物（只有它有训练记录）。
    # 注：standard_sdpa 不在 VARIANTS 里，只存在于 latency.json 的变体表。
    # 必须与上表 prefill_rel_pct 的选取口径一致：不仅要存在该变体，还要在
    # 公共 ctx 上真有 prefill 数据，否则会选中一个在该 ctx 无数据的基线，
    # 令 Prefill/Decode 增幅整体缺失、低估开销并可能翻转权衡判定。
    lat_base_name = next((n for n in ("standard_sdpa", "standard")
                          if _prefill_at(lat_variants.get(n),
                                         common_ctx) is not None), None)
    if common_ctx is not None and lat_base_name:
        base_e = lat_variants[lat_base_name]
        base_prefill = _prefill_at(base_e, common_ctx)
        base_decode = _lat_metric_at(base_e, "decode_ms_per_tok", common_ctx)
        base_params = _eff_cell(base_e, "params", common_ctx)
        base_flops = _eff_cell(base_e, "flops", common_ctx)
        base_kv = _eff_cell(base_e, "kv_mem", common_ctx)
        md.append(f"基线={lat_base_name}（公共 ctx={common_ctx}、batch=1）。增幅="
                  "Prefill ms/Decode ms(tok)/训练 ms(step)/训练进程峰值/参数量/"
                  "FLOPs(token)/KV cache/量测分配量 相对基线增幅的最大值（Prefill/Decode "
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
                x = _finite_number((s.get("eval") or {}).get(
                    "train_step_time_ms"))
                if x is not None and x >= 0:
                    vals.append(x)
            if not vals:
                return None
            vals.sort()
            n = len(vals)
            if n % 2:
                return vals[n // 2]
            # 合法但接近 float 上限的计时/内存中位数：先加后除会溢出。
            return vals[n // 2 - 1] / 2.0 + vals[n // 2] / 2.0

        def _train_peak_mem_mb(v):
            """该变体各种子训练进程峰值的跨种子中位数（MB；旧 eval.json 无→None）。

            与推理单次前向的分配值分开报告：两者口径不同（进程级高水位
            含优化器状态与激活 vs prefill 单次分配）。
            """
            vals = []
            for s in _variant_seed_map(runs, v).values():
                x = _finite_number((s.get("eval") or {}).get(
                    "train_peak_mem_mb"))
                if x is not None and x >= 0:
                    vals.append(x)
            if not vals:
                return None
            vals.sort()
            n = len(vals)
            if n % 2:
                return vals[n // 2]
            # 合法但接近 float 上限的计时/内存中位数：先加后除会溢出。
            return vals[n // 2 - 1] / 2.0 + vals[n // 2] / 2.0

        md.append("")
        md.append("| 变体 | PPL 相对基线 | p_adj(PPL) | RULER Δ(pp) | p_adj(RULER) "
                  "| Prefill ms | Decode ms/tok "
                  "| 训练 ms/step | 训练峰值 | 参数量 | FLOPs/token | KV cache "
                  "| 量测分配 | 判定 |")
        md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
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
            # 基线行（standard）不参与涨点-增幅判定；standard_sdpa 不在
            # VARIANTS 里，根本不会出现在 rows 中。
            if v == "standard" or not lat_entries.get(v):
                continue
            e = lat_entries[v]
            ppl = r.get("val_ppl")
            std_row = next((x for x in rows if x["variant"] == "standard"), None)
            std_ppl = (std_row or {}).get("val_ppl")
            ppl_rel = _rel_pct(std_ppl, ppl) if (ppl is not None and std_ppl) else None
            benefit = -ppl_rel if ppl_rel is not None else None
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
            pre_delta = _rel_pct(base_prefill, _prefill_at(e, common_ctx))
            dec_v = _lat_metric_at(e, "decode_ms_per_tok", common_ctx)
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
            kv_v = _eff_cell(e, "kv_mem", common_ctx)
            kv_delta = _rel_pct(base_kv, kv_v)
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
                                  kv_delta, mem_delta)
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
            r["eff_kv_cache_mb"] = kv_v
            r["eff_kv_cache_delta_pct"] = (round(kv_delta, 2)
                                           if kv_delta is not None else None)
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
                      f"| {_d(kv_delta)} "
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
                  "+ 门控项，见 latency.json 的 efficiency 字段）；KV cache 为公共 "
                  "ctx 下按推理 dtype 计算的 K/V 缓存总量（随 ctx 线性增长，见 "
                  "efficiency.kv_cache_mb_at_ctx），与参数量/激活峰值分开报告；分配量为"
                  "batch=1 prefill 在 CUDA 上的记录，CPU 环境不适用。"
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
    save_text(md_text, os.path.join(results_dir, "summary.md"))

    # CSV（统计列以并列的 mean/std/n 展开，便于程序消费）
    if rows:
        all_keys = sorted({k for r in rows for k in r})
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=all_keys)
        w.writeheader()
        w.writerows(rows)
        save_text(buf.getvalue(), os.path.join(results_dir, "summary.csv"))
    save_json(rows, os.path.join(results_dir, "summary.json"))
    # 设计符合性单独落盘：即使 summary.md 被删改，计划/实际人群的偏差仍可
    # 机读；significance.json 内同时保留一份。
    save_json({"plan_digest": design_plan.get("plan_digest"),
               "plan_status": design_plan.get("status"),
               **design_conf},
              os.path.join(results_dir, "design_conformance.json"))
    significance_payload = {"pairs": tests,
                   "single_factor_audit": sf_audit,
                   "note": (f"同种子配对 t 检验；结论基于同一指标家族内 "
                            f"Holm–Bonferroni 校正后的 p_adj（alpha={ALPHA}），"
                            f"n<{MIN_SEEDS} 不判定显著性；"
                            "d_z 为配对 Cohen's d 效应量。p_perm/p_perm_adj 为"
                            "符号翻转置换检验（分布无关，n≤16 精确）的原始/家族内"
                            "Holm 校正值，t 与置换结论冲突时见 t_perm_agree 与"
                            "verdict 中的 ⚠ 标注；mean_diff_ci95 为配对差 "
                            "bootstrap 95% 区间；mde 为当前种子数/方差下 80% "
                            "功效的最小可检测效应（不显著时先对照 MDE 再下结论）"),
                   "equivalence": _equivalence_overview(tests),
                   "multiverse": multiverse,
                   "multiverse_note": (
                       "分析选择多重宇宙（探索性）：对每个对比重算原始/log "
                       "PPL、macro/micro RULER、全种子/逐个留一等可辩护口径，"
                       "报告方向一致率、同向显著率与翻转规格。规格相关且未"
                       "穷尽，不做多重校正，不改变预注册判定；翻转/不显著"
                       "只要求并列披露，不能据此挑选有利口径。"),
                   "component_evidence": comp,
                   "training_budget": {
                       "comparable": not _as_text_list(
                           (audit or {}).get("early_stop_runs")),
                       "early_stop_runs": _as_text_list(
                           (audit or {}).get("early_stop_runs"))},
                   "registered_hypotheses": hyp_eval,
                   "registered_hypotheses_source": (
                       "preregistration/hypotheses.json" if hyp_eval else None),
                   "confirmatory_family": {
                       "method": ("Holm–Bonferroni over all tested registered "
                                  "hypotheses (n>=MIN_SEEDS), using raw p; "
                                  "family p_adj retained separately"),
                       "alpha": ALPHA,
                       "n_tested": sum(
                           1 for h in hyp_eval
                           if h.get("p_adj_confirmatory") is not None),
                       "n_global_downgraded": sum(
                           1 for h in hyp_eval
                           if h.get("status_family") in ("支持", "拒绝")
                           and h.get("status") == "未定"
                           and h.get("p_adj_confirmatory") is not None)},
                   "registration_integrity": integrity,
                   "registration_scope": scope_check,
                   "design_plan": design_plan,
                   "design_conformance": design_conf,
                   "test_reuse": test_reuse,
                   "observed_design_sensitivity": (
                       observed_design_sensitivity(tests)),
                   "ngram_comparison": ngram_cmp,
                   "linear_probe_comparison": probe_cmp,
                   "ruler_sample_pairs": ruler_sample_tests,
                   "ruler_length_pairs": length_analysis.get("pairs", []),
                   "ruler_length_tests": {
                       "per_context": length_analysis.get("ctx_tests", []),
                       "slope": length_analysis.get("slope_tests", []),
                       "long_minus_short": length_analysis.get(
                           "span_tests", [])},
                   "ruler_length_note": length_analysis.get("note"),
                   "ppl_window_pairs": ppl_window_tests,
                   "ppl_log_pairs": [
                       t for t in tests if t.get("metric") == "PPL(log)"],
                   "factorial_interactions": [
                       t for t in tests if t.get("interaction")],
                   "factorial_shapley": shapley_tests,
                   "reparameterization": reparameterization,
                   "reparameterization_note": (
                       "参数化稳健性（探索性）：归一化 c=log(n*max_p) 与原始 "
                       "log max_p 两套完整 2^3 网格上的单因子边/Shapley 对照。"
                       "逐种子 D=raw−normalized 做配对 t 检验并 Holm 校正；"
                       "D 显著或方向不一致表示因子归因依赖参数化，不能把 "
                       "归一化 c 的效应直接解释为纯集中度；不显著不证明等价。"
                       "未运行 raw 对照时 not_run=true，不得据此宣称稳健。"),
                   "factorial_shapley_note": (
                       "全因子 2^3 的 Shapley 效应分解（探索性）：PPL 在 "
                       "−log(PPL) 上分解（φ>0 降低 PPL，exp(−φ)=PPL 几何"
                       "均值比），RULER 用原始 EM；φ_A+φ_B+φ_C 等于 "
                       "Full−none 的总效应（效率性），交互效应被平均分摊到"
                       "参与因子；不替代预注册的单因素确认检验"),
                   "budget_sensitivity": budget_sensitivity,
                   "budget_sensitivity_note": (
                       "训练预算敏感性（探索性）：由 train_log.jsonl 的 "
                       "val_loss 生成同种子、同 step 的候选−参照轨迹；"
                       "逐 step p 未做跨 step 校正且各 step 相关，"
                       "budget_sensitive 只提示方向在训练预算间翻转，"
                       "不用于重新选 checkpoint，也不改变预注册终点判定。"
                       "缺 train_log.jsonl/共同 step 时 available=false。"),
                   "causal_ablation": causal_tests,
                   "causal_ablation_equivalence": _causal_equivalence_overview(
                       causal_tests),
                   "matched_ablation_contrasts": (
                       matched_ablation_contrasts),
                   "matched_ablation_contrasts_note": (
                       "活动匹配对照（独立 Holm 家族）：原始组件病灶减去"
                       "逐层逐头匹配平均门控后的病灶；差值为正说明原始 ΔNLL "
                       "中有一部分可由平均门控开度变化解释。只匹配平均 g"
                       "（一阶矩），不匹配方差/饱和比例，不能把差值当成精确"
                       "的“信息效应+写回效应”分解；信息特异效应看匹配病灶"
                       "相对基线的独立检验"),
                   "parameter_matched_control": {
                       "variant": PARAM_CONTROL_VARIANT,
                       "pair": list(PARAM_CONTROL_PAIR),
                       "available": bool(param_control_test),
                       "test": param_control_test,
                       "capacity_only_test": capacity_only_test,
                       "note": ("训练期参数匹配安慰剂：lega_placebo 与 lega 的"
                                "门控参数形状/个数/门控 FLOPs 完全相同，但把 "
                                "B/C 证据替换为确定性伪随机特征；lega vs "
                                "lega_placebo 的 PPL 配对差用于排除“多几个门控"
                                "参数的容量/优化红利”。探索性、未预注册；"
                                "未运行该变体时 available=false，不得默认容量"
                                "已排除。")},
                   "inference_factorial_shapley": inference_shapley,
                   "inference_factorial_shapley_note": (
                       "同一 checkpoint 内的 2^3 门控干预（正常基线、"
                       "query_only/conc_only/len_only、no_query/no_conc/no_len、"
                       "no_evidence）上的 Shapley 分解（探索性）：φ>0 表示该"
                       "因子在训练好的前向里降低 PPL；与训练期 Shapley 并列，"
                       "用于区分“重新训练后的组件贡献”与“最终前向实际使用”。"),
                   "inference_factorial_interactions": inference_interactions,
                   "inference_factorial_interactions_note": (
                       "同一 checkpoint 内的 2^3 干预上的 A×B/A×C/B×C 交互"
                       "检验（探索性、独立 Holm 家族）：与训练期 `PPL交互` "
                       "同尺度并列，检查“重新训练后的交互结构能否在最终前向"
                       "复现”；交互=0 表示可加，显著非零不等于某个组件有用。"),
                   "dose_response": dose_response,
                   "dose_response_note": (
                       "门控系数剂量-反应（lega.dose_response，val split，"
                       "探索性）：同一 checkpoint/同一批窗口上把 A(gate_w)、"
                       "B(gate_a)、C(gate_b) 系数乘以 λ，逐窗口配对 "
                       "ΔNLL(λ)=NLL(λ)−NLL(1)；λ=1 为原始系数，λ=0 对应"
                       "对应组件的推理期清零。A/B/C 与局部斜率各自按 metric "
                       "分 Holm 家族；局部斜率、二次顶点与网格最优 λ 均为"
                       "描述性诊断，固定网格挑最优会引入 winner's curse，"
                       "不据此重选 checkpoint 或改写预注册判定；"
                       "no_op/全 no-op 不作等价性判定"),
                   "causal_ablation_note": (
                       "推理期因果病灶消融（lega.ablate，val split）：同一 "
                       "checkpoint/同一批窗口上只改门控参数，逐窗口配对 ΔNLL；"
                       "no_query/no_conc/no_len/no_evidence 是训练期去一法的"
                       "推理期对应，no_*_matched 是逐层逐头匹配平均门控的"
                       "活动匹配对照（独立 Holm 家族）；"
                       "query_only/conc_only/len_only 补齐同一 "
                       "checkpoint 的 2^3 单因子格；head_permute 是容量/分布"
                       "不变的安慰剂对照，conc_permute/query_permute/"
                       "len_permute 是 B/A/C 位置证据的分布匹配安慰剂；"
                       "gate_open/closed/constant_mean 检验写回强度与逐位置"
                       "变化；跨种子 p 经 Holm 校正，窗口级 p "
                       "仅描述性；equivalence 字段给出 ΔNLL 相对 0.01 nat "
                       "SESOI 的 TOST 结果，no_op 不作等价性判定"),
                   "factorial_interactions_note": (
                       "全因子 2^3 的 A×B/A×C/B×C 交互对比（探索性，独立 "
                       "Holm 家族，基准 gate_only）：交互=0 表示组件效应可加；"
                       "显著非零说明同时加入两个组件时收益不能按单加总"),
                   "auroc_diff_note": (
                       f"{AUROC_DIFF_METRIC} 家族为机制主张的正式检验：每个 "
                       "(变体, 种子) run 的 auroc_logit − auroc_qterm 配对差，"
                       "与逐 run 样例聚类 bootstrap 区间"
                       "（analysis_*.json 的 auroc_cluster_bootstrap）互为印证")}
    save_json(significance_payload,
              os.path.join(results_dir, "significance.json"))
    cross_out = {}
    if bad_cross:
        cross_out["ruler"] = bad_cross
    if win_cross:
        cross_out["lm_windows"] = win_cross
    if cross_out:
        save_json(cross_out, os.path.join(results_dir, "badcase_cross.json"))
    print(f"[summary] 汇总 {len(rows)} 个变体 -> {results_dir}/summary.md "
          f"（配对检验 {len(tests)} 组）")

    _assigned = {
        "_d",
        "_dpp",
        "_train_peak_mem_mb",
        "_train_step_ms",
        "all_keys",
        "base_decode",
        "base_e",
        "base_flops",
        "base_kv",
        "base_params",
        "base_prefill",
        "base_train_mem",
        "base_train_ms",
        "bci",
        "bci_s",
        "benefit",
        "budget_sensitivity",
        "ci",
        "ci_s",
        "cluster_rows",
        "cost",
        "counts",
        "cross_out",
        "dec_delta",
        "dec_noise",
        "dec_v",
        "dec_within",
        "delta",
        "deltas",
        "dz",
        "e",
        "eq",
        "eq_p",
        "eq_s",
        "extra",
        "f",
        "flo_delta",
        "flop_pct",
        "gate_flops",
        "h",
        "kv_delta",
        "kv_v",
        "lat_base_name",
        "log_tests",
        "log_verdict",
        "loo",
        "macro_delta",
        "mark",
        "mci",
        "mci_s",
        "md",
        "md_text",
        "mde",
        "med",
        "mem_b",
        "mem_delta",
        "mem_v",
        "memtr_delta",
        "memtr_v",
        "mismatch",
        "n_global_downgraded",
        "n_metric_within_noise",
        "n_mismatch",
        "n_ppl",
        "n_ref",
        "n_ruler",
        "n_sup",
        "n_und",
        "n_untested",
        "note",
        "p",
        "p_adj",
        "padj",
        "par_delta",
        "param_pct",
        "params",
        "per_head",
        "pg",
        "pp",
        "ppadj",
        "ppl",
        "ppl_rel",
        "ppl_test",
        "pre_delta",
        "pre_noise",
        "pre_within",
        "psadj",
        "r",
        "ratio",
        "raw",
        "raw_verdict",
        "reg_ruler_pairs",
        "reparameterization",
        "rt",
        "ruler_benefit_pp",
        "ruler_p_adj",
        "ruler_test",
        "seed_t",
        "seed_verdict",
        "shown",
        "std",
        "std_ppl",
        "std_row",
        "t",
        "task_rows",
        "total_flops",
        "tr_delta",
        "tr_v",
        "typ",
        "v",
        "valid",
        "verdict",
        "w",
        "wl",
    }
    return {k: v for k, v in locals().items() if k in _assigned}
