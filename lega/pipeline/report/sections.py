"""汇总报告渲染（第二部分：曲线、坏案例、消融）。"""
from .._shared import _install as _shared_install

_shared_install(globals())


def _fmt_pct(value, spec: str = ".0%") -> str:
    """百分比缺失显示 '-'；有限值乘 100 溢出时显示“溢出”。"""
    f = _finite_number(value)
    if f is None:
        return "-"
    return "溢出" if not math.isfinite(f * 100.0) else format(f, spec)


def _fmt_pct0(value) -> str:
    return _fmt_pct(value)


_BUDGET_PATTERN_CN = {
    "stable": "稳定",
    "transient": "中途翻转",
    "reversal_nonsignificant": "首末翻转（未显著）",
    "budget_sensitive": "预算敏感",
    "insufficient_steps": "共同步数不足",
    "all_zero": "全零差",
}


def _render_budget_sensitivity(budget):
    """渲染训练预算轨迹；只作描述，不改变终点/确认性判定。"""
    out = ["", "### 训练预算敏感性（探索性：结论是否只在终点成立）", ""]
    if not isinstance(budget, dict):
        out.append("⚠ 未生成训练预算敏感性结果（汇总阶段缺少曲线）。")
        return out
    pairs = [p for p in (budget.get("pairs") or []) if isinstance(p, dict)]
    available = [p for p in pairs if p.get("available")]
    if not available:
        out.append("⚠ 无可用训练曲线（`train_log.jsonl` 缺失、checkpoint "
                   "路径不可达，或两侧没有共同评测步）；本轮不评估对比/消融"
                   "结论是否随训练预算变化，不能据此宣称“终点结论稳定”。")
        return out
    out.append("只使用训练日志中已有的 **val** 损失，按同种子、同评测步对齐；"
               "不插值、不重新选 checkpoint、不改变终点检验。val 同时是选型集，"
               "逐 step p 值高度相关且未做跨 step 校正，故本表全部为探索性证据，"
               "不能替代 held-out test 上的终点结论。")
    out.append("")
    out.append("| 对比 | 评测步（首→末） | n种子 | 终点Δ(logPPL) | 符号一致率 | "
               "符号翻转 | 反向显著步 | 轨迹模式 |")
    out.append("|---|---|---|---|---|---|---|---|")
    def _order(p):
        return (not bool(p.get("budget_sensitive")),
                -(p.get("n_sign_flips") or 0),
                str(p.get("candidate")), str(p.get("reference")))
    for p in sorted(available, key=_order)[:14]:
        delta = _fmt_signed(p.get("final_mean_diff"))
        agree = _fmt_pct(p.get("sign_agreement_with_final"), ".0%")
        flips = p.get("n_sign_flips")
        flips_s = "-" if flips is None else str(flips)
        if p.get("budget_sensitive"):
            reverse = "⚠ 是"
        elif p.get("sign_reversal"):
            reverse = "方向翻转"
        else:
            reverse = "否"
        pattern = _BUDGET_PATTERN_CN.get(
            p.get("trajectory_pattern"), str(p.get("trajectory_pattern") or "-"))
        out.append(
            f"| {p.get('candidate')} vs {p.get('reference')} "
            f"| {p.get('initial_step')}→{p.get('final_step')} "
            f"| {p.get('n_seeds')} | {delta} | {agree} | {flips_s} "
            f"| {reverse} | {pattern} |")
    if len(available) > 14:
        out.append("")
        out.append(f"（仅列前 14 组；其余 {len(available) - 14} 组的逐 step "
                   "明细见 `significance.json` 的 `budget_sensitivity`。）")
    out.append("")
    out.append("注: Δ(logPPL)=候选−参照（<0 表示候选在该预算更好）；"
               "“反向显著步”指至少两个评测步以相反符号达到未校正 p<α，"
               "只提示轨迹不稳定，不是确认性结论。模式 `稳定` 仅表示方向未翻转，"
               "不代表每个预算都统计显著；`中途翻转` 与 `首末翻转` 才是"
               "“不能用终点代表整个训练过程”的直接信号。")
    return out


def _summary_part_b(**kw):
    if "audit" in kw:
        audit = kw["audit"]
    if "c" in kw:
        c = kw["c"]
    if "cand" in kw:
        cand = kw["cand"]
    if "capacity_only_test" in kw:
        capacity_only_test = kw["capacity_only_test"]
    if "causal_tests" in kw:
        causal_tests = kw["causal_tests"]
    matched_ablation_contrasts = (kw.get("matched_ablation_contrasts") or [])
    budget_sensitivity = kw.get("budget_sensitivity")
    if "ci" in kw:
        ci = kw["ci"]
    if "comp" in kw:
        comp = kw["comp"]
    if "d" in kw:
        d = kw["d"]
    if "e" in kw:
        e = kw["e"]
    if "eps" in kw:
        eps = kw["eps"]
    if "fmt_stat_cell" in kw:
        fmt_stat_cell = kw["fmt_stat_cell"]
    if "fp" in kw:
        fp = kw["fp"]
    if "inference_interactions" in kw:
        inference_interactions = kw["inference_interactions"]
    if "inference_shapley" in kw:
        inference_shapley = kw["inference_shapley"]
    if "k" in kw:
        k = kw["k"]
    if "load" in kw:
        load = kw["load"]
    if "m" in kw:
        m = kw["m"]
    if "md" in kw:
        md = kw["md"]
    if "name" in kw:
        name = kw["name"]
    if "param_control_test" in kw:
        param_control_test = kw["param_control_test"]
    if "r" in kw:
        r = kw["r"]
    if "repro" in kw:
        repro = kw["repro"]
    reparameterization = kw.get("reparameterization")
    if "results_dir" in kw:
        results_dir = kw["results_dir"]
    if "rows" in kw:
        rows = kw["rows"]
    if "runs" in kw:
        runs = kw["runs"]
    if "s" in kw:
        s = kw["s"]
    if "scope_seeds" in kw:
        scope_seeds = kw["scope_seeds"]
    if "scope_variants" in kw:
        scope_variants = kw["scope_variants"]
    if "sf_audit" in kw:
        sf_audit = kw["sf_audit"]
    if "shapley_tests" in kw:
        shapley_tests = kw["shapley_tests"]
    if "shown" in kw:
        shown = kw["shown"]
    if "st" in kw:
        st = kw["st"]
    if "t" in kw:
        t = kw["t"]
    if "tests" in kw:
        tests = kw["tests"]
    if "v" in kw:
        v = kw["v"]
    if "variants" in kw:
        variants = kw["variants"]

    md += ["", "## 可复现性与 sanity check", ""]
    if repro:
        git = _as_dict(repro.get("git"))
        data = _as_dict(repro.get("data"))
        rt = _as_dict(repro.get("runtime"))
        commit = git.get("commit") or "unknown"
        dirty = git.get("dirty")
        md.append(f"- 代码版本：`{str(commit)[:12]}`"
                  + (f"（分支 {git.get('branch')}）" if git.get("branch") else "")
                  + ("；⚠ 工作区 dirty，结果对应未提交状态" if dirty else
                     ("" if dirty is None else "；工作区 clean")))
        files = data.get("files")
        files = [e for e in files if isinstance(e, dict)] \
            if isinstance(files, list) else []
        md.append(f"- 数据指纹：`{data.get('digest')}`"
                  f"（tokenizer={data.get('tokenizer')}，"
                  + "、".join(f"{e.get('name')} "
                              f"sha256:{str(e.get('sha256'))[:12]}…"
                              for e in files)
                  + "）")
        md.append(f"- 配置 SHA-256：`{str(repro.get('config_sha256'))[:16]}…`；"
                  f"种子：{repro.get('seeds')}")
        src = repro.get("source_tree")
        if isinstance(src, dict) and isinstance(src.get("digest"), str):
            md.append(f"- 源码/配置树指纹：`{src['digest']}`"
                      f"（{src.get('files')} 个文件，独立于 git；"
                      "无 git 的源码包也可复算）")
        md.append(f"- 运行环境：python={rt.get('python')} torch={rt.get('torch')} "
                  f"device={rt.get('device')} gpu={rt.get('gpu')} cuda={rt.get('cuda')}")
        # 逐 run 核对评测环境与训练清单：pipeline 内的 eval 通常同环境，但
        # results 目录可能被另一环境单独补跑（--skip_train/手工 evaluate），
        # 仅看全局 repro.json 的 pipeline runtime 无法发现这种错配。
        runtime_drift_rows = []
        for _v, _seed_map in runs.items():
            for _seed, _slots in _seed_map.items():
                _ev = (_slots or {}).get("eval")
                if not isinstance(_ev, dict) or \
                        _ev.get("runtime_matches_training") is not False:
                    continue
                _fields = ",".join(
                    str(m.get("field"))
                    for m in (_ev.get("runtime_mismatches") or [])
                    if isinstance(m, dict) and m.get("field")) or "unknown"
                runtime_drift_rows.append(f"{_v}/seed{_seed}（{_fields}）")
        if runtime_drift_rows:
            md.append("- ⚠ 以下 run 的**评测环境**与训练 `run_manifest.json` "
                      "记录不一致："
                      + "、".join(runtime_drift_rows)
                      + "；同一 checkpoint 在不同 Python/PyTorch/NumPy/CUDA/"
                      "cuDNN/GPU 上可能因浮点内核差异得到不同数值，相关指标不得"
                      "声称与训练环境完全相同（逐项见 `eval_*.json` 的 "
                      "`train_runtime`/`eval_runtime`）。")
        source_drift_rows = []
        for _v, _seed_map in runs.items():
            for _seed, _slots in _seed_map.items():
                _ev = (_slots or {}).get("eval")
                if (not isinstance(_ev, dict)
                        or _ev.get("source_tree_matches_training") is not False):
                    continue
                source_drift_rows.append(f"{_v}/seed{_seed}")
        if source_drift_rows:
            md.append("- ⚠ 以下 run 的**评测源码/配置树**与训练 "
                      "`run_manifest.json` 记录不一致："
                      + "、".join(source_drift_rows)
                      + "；评测可能运行在已修改的代码/配置上，相关指标不得"
                      "声称与训练使用同一版本（逐项见 `eval_*.json` 的 "
                      "`train_source_tree_digest`/`eval_source_tree_digest`）。")
        md.append("- 完整记录：`repro.json`（全局）、各 run 目录 `run_manifest.json`"
                  "（代码/数据/环境/命令行）、`config_used.json`（超参数）、"
                  "`train_log.jsonl`（训练曲线）。")
    else:
        md.append("- ⚠ 未找到 `repro.json`（直接调用 build_summary？）；"
                  "无法核对代码版本与数据指纹。")
    sanity_files = sorted(glob.glob(os.path.join(results_dir, "sanity_*.json")))

    def _load_dict(p):
        d = load(p)
        return d if isinstance(d, dict) else None

    # 与 runs 同一“本次 run”口径：sanity 虽无“每变体多档种子”，但
    # run_all 只用首个种子跑 sanity；同一 results 目录被换种子重跑（尤其
    # --skip_sanity）时，旧种子的 PASS 文件不能在本轮汇总里冒充本次证据。
    expected_sanity_seed = (_integral_value(scope_seeds[0])
                            if scope_seeds else None)
    if scope_variants is not None:
        sanity_files = [fp for fp in sanity_files
                        if (_load_dict(fp) or {}).get("variant") in scope_variants]
    if expected_sanity_seed is not None:
        kept = []
        for fp in sanity_files:
            d = _load_dict(fp) or {}
            d_seed = _integral_value(d.get("seed"))
            # 旧 sanity 文件没有 seed 字段时保留（无法核对，不能伪装成
            # “已属于本次 run”）；明确记录为其他 seed 的文件必须剔除。
            if d_seed is not None and d_seed != expected_sanity_seed:
                continue
            kept.append(fp)
        sanity_files = kept
    if sanity_files:
        md.append("")
        md.append("| 变体 | sanity | 初始 loss | 最终 loss | token 准确率 |")
        md.append("|---|---|---|---|---|")
        for fp in sanity_files:
            d = _load_dict(fp)
            if not d:
                continue
            ini = _num_or_none(_as_dict(d.get("initial")).get("loss"))
            fin = _num_or_none(_as_dict(d.get("final")).get("loss"))
            acc = _num_or_none(_as_dict(d.get("final")).get("token_acc"))
            sv = d.get("variant")
            sv = VARIANT_CN.get(sv, sv) if isinstance(sv, str) else "?"
            md.append(f"| {sv} | "
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
    # 合法 JSON 但非对象（数组/标量）的诊断文件按“无诊断”处理。
    if not isinstance(curve, dict):
        curve = None
    if curve:
        md.append("诊断口径：训练降而验证升=过拟合；训练停滞（不再实质下降）而"
                  "验证回升同样判过拟合（平台期泛化先恶化，不按欠拟合处理）；"
                  "训练仍在实质下降而验证无实质变化=过拟合趋势（验证不再随训练"
                  "改善的早期信号，按正则化/早停处理，不据此续训）；"
                  "两者都不降=欠拟合/学习率不当/数据管道问题；"
                  "验证损失显著低于训练损失（倒挂）=泄漏/划分错配/评估口径"
                  "不一致，不能当“验证更好”；best 与次优差距落在评估噪声内="
                  "checkpoint 选型不可靠（winner's curse）；曲线虽在降但"
                  "训练覆盖 <1 epoch=训练不足（还没学完，不能当收敛）；"
                  "出现 NaN=训练失效；"
                  "瞬态损失尖峰（分段均值会掩盖；另查区间单步峰值 loss_max）"
                  "与损坏日志行单独报告；全程首/末之外还比较末段（后 1/3 vs "
                  "中 1/3），晚期平台/回升不被早段大幅改善掩盖。"
                  "逐 run 明细见 `curve_diagnosis.json`。")
        md.append("")
        md.append("| 变体 | 状态（最差种子） | 训练覆盖(epoch) | 训练损失 首→末 "
                  "| 验证损失 首→最优→末 | 早停空间 | 关键建议 |")
        md.append("|---|---|---|---|---|---|---|")
        any_nan = False
        any_grad_spike = False
        any_loss_spike = False
        any_malformed = False
        any_inversion = False
        any_selection_noise = False
        observability = set()
        for v in variants:
            by_seed = curve.get(v)
            if not isinstance(by_seed, dict):
                continue
            diags = [d for d in by_seed.values()
                     if isinstance(d, dict) and isinstance(d.get("status"), str)]
            if not diags:
                continue
            status = worst_status([d["status"] for d in diags])
            worst = next(d for d in diags if d["status"] == status)
            any_nan = any_nan or any(d.get("nonfinite_values") for d in diags)
            spike = any(d.get("has_grad_spike") for d in diags)
            any_grad_spike = any_grad_spike or spike
            lspike = any(d.get("has_loss_spike") for d in diags)
            any_loss_spike = any_loss_spike or lspike
            any_malformed = any_malformed or any(d.get("malformed_records")
                                                 for d in diags)
            any_inversion = any_inversion or any(d.get("val_below_train")
                                                 for d in diags)
            any_selection_noise = any_selection_noise or any(
                d.get("best_checkpoint_within_noise") for d in diags)
            for d in diags:
                for w in (d.get("curve_observability_warnings") or []):
                    if isinstance(w, str) and w.strip():
                        observability.add(w.strip())
            es = [x for x in (_num_or_none(d.get("early_stop_gain_rel"))
                              for d in diags) if x is not None]

            def _f4(x):
                x = _num_or_none(x)
                return "-" if x is None else f"{x:.4f}"

            train_s = (f"{_f4(worst.get('train_loss_first'))}→"
                       f"{_f4(worst.get('train_loss_last'))}")
            val_s = (f"{_f4(worst.get('val_loss_first'))}→"
                     f"{_f4(worst.get('val_loss_best'))}"
                     f"@s{worst.get('val_best_step')}"
                     f"→{_f4(worst.get('val_loss_last'))}")
            es_avg = (sum(es) / len(es)) if es else None
            es_s = ("-" if es_avg is None else
                    _fmt_pct(es_avg, "+.1%")
                    + ("（最终劣于最优，已用 best checkpoint）"
                       if es_avg > 0.02 else ""))
            eps = [x for x in (_num_or_none(d.get("epochs_seen"))
                               for d in diags) if x is not None]
            ep_min = min(eps) if eps else None
            ep_s = ("-" if ep_min is None else
                    f"{ep_min:.3f}" + (" ⚠<1" if ep_min < 1.0 else ""))
            rec = worst.get("recommendations")
            rec = rec[:1] if isinstance(rec, list) else []
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
            md.append(f"| {VARIANT_CN.get(v, v)} | {status_s} | {ep_s} "
                      f"| {train_s} | {val_s} | {es_s} "
                      f"| {rec[0] if rec else '-'} |")
        md.append("")
        if observability:
            shown = sorted(observability)
            more = len(shown) - 2
            md.append("⚠ 曲线采样可观测性受限（配置级，所有种子同因）："
                      + "；".join(shown[:2])
                      + (f"；另有 {more} 条同类告警" if more > 0 else "")
                      + "。上述趋势/早停结论的可靠性受此限制，正式实验请"
                      "按告警调整 `eval_every`/`log_every` 后重跑。")
            md.append("")
        if any_nan:
            md.append("⚠ 存在出现非有限损失的 run：训练已失效，"
                      "对应 run 目录的 `nan_report.json` 给出了"
                      "数据→代码→混合精度→梯度/学习率的系统排查路径"
                      "（含检测点前的整个 log 区间候选批次与区间内首个非有限"
                      "损失步 `first_bad_step`，排查的是失败现场而非最后一步"
                      "批次），处理前该变体的一切指标都不可信。")
        else:
            md.append("注：全部 run 的损失曲线无 NaN/Inf（梯度非有限另见下）；"
                      "有异常状态的变体请按建议处理"
                      "（过拟合/过拟合趋势→正则化/增数据/早停，欠拟合→查学习率"
                      "与数据管道，训练不足→延长训练/增大数据覆盖）后重跑，"
                      "再进入对比结论。")
        if any_inversion:
            md.append("")
            md.append("⚠ 存在验证损失显著低于训练损失的 run（与泛化差距方向"
                      "相反）：先按 datacheck 审计跨 split 重复/n-gram 重叠与"
                      "划分边界，再核对 train/val 的 dropout 状态、精度与损失"
                      "统计口径；排除泄漏/错配前，该 run 的验证指标不得用于"
                      "方法比较。")
        if any_selection_noise:
            md.append("")
            md.append("⚠ 存在 best checkpoint 与次优验证损失差距落在该点 95% CI "
                      "内的 run：最优与次优无法区分（winner's curse），"
                      "不能只报告 best 数字；应同时报告 final/多种子均值，"
                      "并增大 eval_iters 或验证集后再据此选型。")
        if any_grad_spike:
            md.append("")
            md.append("⚠ 检测到 pre-clip 梯度范数非有限或爆炸性尖峰"
                      "（区间峰值/中位数 ≥10×）的 run：这是 NaN/发散的早期信号，"
                      "即使损失曲线看似正常也应先降低学习率/收紧 grad_clip "
                      "后重跑；发生步与逐点清单见 `curve_diagnosis.json` 的 "
                      "`grad_norm_max*`/`grad_spikes` 字段。")
        if any_loss_spike:
            md.append("")
            md.append(f"⚠ 检测到训练损失瞬态尖峰（某点 > 邻近中位数 "
                      f"{LOSS_SPIKE_FACTOR:.0f}×，分段均值会掩盖；"
                      "区间单步峰值 loss_max 另查）：先核对该"
                      "步批次数据与梯度范数，再决定收紧 grad_clip/降低学习率；"
                      "细节见 `curve_diagnosis.json` 的 "
                      "`loss_spikes`/`loss_peak_spikes` 字段。")
        if any_malformed:
            md.append("")
            md.append("⚠ 部分 train_log.jsonl 记录无法解析（已跳过并计数）："
                      "日志可能被中断或损坏，曲线结论只基于其余记录点，"
                      "复核前不要把它当作完整训练历史。")
        any_early = any(
            d.get("early_stopped")
            for vs in curve.values() if isinstance(vs, dict)
            for d in vs.values() if isinstance(d, dict))
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

    # --- 门控健康：机制是否真的在起作用，不能只看最终 PPL
    md += ["", "## 门控健康（机制是否真的在起作用）", ""]
    gate_order = {"健康": 0, "不适用（无门控统计）": 0,
                  "门控近似常数（无样本级区分）": 2,
                  "门控剧烈漂移": 3,
                  "门控塌缩（几乎不写入）": 4,
                  "门控饱和（退化为标准注意力）": 4}
    gate_rows = []
    if isinstance(curve, dict):
        for v in variants:
            by_seed = curve.get(v)
            if not isinstance(by_seed, dict):
                continue
            hs = [d.get("gate_health") for d in by_seed.values()
                  if isinstance(d, dict)
                  and isinstance(d.get("gate_health"), dict)
                  and d["gate_health"].get("available")]
            if not hs:
                continue
            worst_g = max(hs, key=lambda h: gate_order.get(h.get("status"), -1))
            status_g = worst_g.get("status") or "-"
            m_last = _num_or_none(worst_g.get("mean_last"))
            s_last = _num_or_none(worst_g.get("std_last"))
            lo = _num_or_none(worst_g.get("frac_low_last"))
            hi = _num_or_none(worst_g.get("frac_high_last"))
            layers = [x for x in (_num_or_none(y) for y in
                                  (worst_g.get("layer_mean_last") or []))
                      if x is not None]
            layer_s = ("-" if not layers else
                       (f"{min(layers):.2f}" if len(layers) == 1
                        else f"{min(layers):.2f}~{max(layers):.2f}"))
            n_bad_gate = sum(1 for h in hs
                             if h.get("status") not in ("健康",))
            status_s = status_g + (f"（{n_bad_gate}/{len(hs)} 种子需关注）"
                                   if n_bad_gate else "")
            rec = (worst_g.get("recommendations") or [""])[0]
            gate_rows.append(
                f"| {VARIANT_CN.get(v, v)} | {status_s} | "
                f"{'-' if m_last is None else f'{m_last:.3f}'} | "
                f"{'-' if s_last is None else f'{s_last:.4f}'} | "
                f"{_fmt_pct0(lo)} | {_fmt_pct0(hi)} | {layer_s} | {rec} |")
    if gate_rows:
        md.append("门控健康检查的是方法机制本身：门控塌缩/饱和/常数化时，"
                  "PPL 的涨跌不能归因于查询、集中度或长度校准项。逐 run 数值"
                  "见 `curve_diagnosis.json` 的 `gate_health`。")
        md.append("")
        md.append("| 变体 | 门控健康（最差种子） | 末点均值 | std | 低饱和比例 "
                  "| 高饱和比例 | 逐层均值范围 | 关键建议 |")
        md.append("|---|---|---|---|---|---|---|---|")
        md += gate_rows
        bad_gate = [r for r in gate_rows if "健康" not in r.split("|")[2]]
        if bad_gate:
            md.append("")
            md.append("⚠ 存在门控塌缩/饱和/常数化/漂移：对应的 PPL/RULER 差异"
                      "可能来自“门关闭/门透明”而不是所声称的机制；先按建议"
                      "修复门控再解释结果，必要时改报“机制未生效”。")
    else:
        md.append("- 暂无门控统计（旧日志或未评测门控变体）；门控均值不能"
                  "替代分布检查，建议用当前版本重跑训练内评测。")

    # --- 坏案例分析：预测错误的样本逐一归因（改进思路的最重要来源）
    md += ["", "## 坏案例分析（预测错误的样本与归因）", ""]
    bad_rows, bad_example_pool = [], []
    # 跨变体对照要求各变体使用同一批评测样本：取每个变体最小种子的错误
    # 清单（RULER 样本由配置里的评测种子构造，与训练种子无关，可逐条对齐）
    per_variant_badcases = cross_variant_badcase_map(runs, variants)
    corpus_sources = set()
    for v in variants:
        for s in _variant_seed_map(runs, v).values():
            r = s.get("ruler") or {}
            if r.get("corpus_source"):
                corpus_sources.add(str(r["corpus_source"]))
    for v in variants:
        m = _variant_seed_map(runs, v)
        cases = []
        for s in m.values():
            r = s.get("ruler")
            if r:
                badcases = r.get("badcases")
                if isinstance(badcases, list):
                    cases += badcases
        if not cases:
            continue
        ana = analyze_badcases(cases, variant=v)
        cats = sorted(ana["category_counts"].items(), key=lambda kv: -kv[1])
        cat_s = "；".join(f"{BADCASE_CN.get(k, k)}×{n}" for k, n in cats[:3])
        bad_rows.append((v, ana, cat_s))
        bad_example_pool += [(v, c) for c in ana["cases"]]
    if corpus_sources:
        src_s = "、".join(f"{x}.txt" for x in sorted(corpus_sources))
        if "train" in corpus_sources:
            md.append(f"⚠ RULER niah_corpus 干扰语料来自 **{src_s}**："
                      "训练文本进入评测上下文会让长上下文检索偏容易，"
                      "该任务的绝对分数与 held-out 语料运行不可比（见 ruler payload "
                      "的 `corpus_source`）。")
        else:
            md.append(f"RULER niah_corpus 干扰语料来源：{src_s}（held-out，"
                      "未使用训练文本）。")
        md.append("")
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
                      f"| {_fmt_pct0(dfr)} "
                      f"| {ana['label_errors'] or 0} |")
        md.append("")
        # 摘 3 条代表性个案放进汇总页：错误必须"看得见"，不能只留在文件里
        shown, seen_cat = 0, set()
        for v, c in bad_example_pool:
            if shown >= 3 or c["category"] in seen_cat:
                continue
            seen_cat.add(c["category"])
            shown += 1
            q = _as_text(c.get("question")).strip().replace("\n", " ")[:120]
            pred = _as_text(c.get("pred")).strip().replace("\n", " ")[:80]
            md += [f"- **{VARIANT_CN.get(v, v)} · {c.get('task') or '-'}"
                   f"@ctx{c.get('ctx')}** "
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
        shared_ci = bad_cross.get("shared_all_ci95")
        shared_ci_s = ""
        if isinstance(shared_ci, (list, tuple)) and len(shared_ci) == 2:
            shared_ci_s = (f"（Wilson 95% CI {_fmt_pct0(shared_ci[0])}–"
                           f"{_fmt_pct0(shared_ci[1])}）")
        md.append(
            f"各变体取最小种子的错误清单（样本集相同，可逐条对齐；RULER 已"
            f"评测但零错误的变体按“全部通过”计入，不会被误算作共识失败）：共 "
            f"{bad_cross['n_unique_failed_items']} 个不同的失败样本，"
            f"{n_shared} 个为全体（已评测）变体共识失败{shared_ci_s}——"
            "共识失败（尤其当 "
            "standard 也错）更可能是数据/构造/内在难度问题，不应算作某一"
            "方法的问题；相对 standard 的新增错误才是该方法的特异性盲区，"
            "优先逐条分析。")
        md += ["", "| 变体 | 错误样本数 | 全体共识失败 | 相对standard新增 "
                   "| 相对standard改对 | 配对p (Holm) |",
                   "|---|---|---|---|---|---|"]
        for v in bad_cross["variants"]:
            pv = (bad_cross["per_variant"] or {}).get(v) or {}

            def _cnt(x):
                return "-" if x is None else str(x)

            p_adj = pv.get("mcnemar_p_holm")
            p_s = (f"{p_adj:.3g}" if isinstance(p_adj, (int, float))
                   else "-")
            md.append(f"| {VARIANT_CN.get(v, v)} | {_cnt(pv.get('n_failed'))} "
                      f"| {_cnt(pv.get('n_shared_all'))} "
                      f"| {_cnt(pv.get('n_new_vs_reference'))} "
                      f"| {_cnt(pv.get('n_fixed_vs_reference'))} "
                      f"| {p_s} |")
        md.append(
            "注: 新增/改对是同一批样本上的配对计数，p 为精确 McNemar 检验"
            "（多候选 Holm 校正）；p≥0.05 时只能当方向性线索，不能据此"
            "断言方法优劣。")
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
    # 逐变体归因用全部种子的窗口累计（错误模式必须跨种子稳定）；跨变体
    # 对齐仍只用最小种子的同一批窗口（样本集必须一致）。
    per_variant_windows = {}
    per_variant_windows_first = {}
    window_seed_counts = {}
    for v in variants:
        m = _variant_seed_map(runs, v)
        all_keyed, first_keyed = [], None
        for s in _sorted_seeds(m):
            cand = m[s].get("eval") or {}
            wins = cand.get("worst_windows")
            if not isinstance(wins, list) or not wins:
                continue
            # 评测窗口集合由 (val/test 长度, ctx, eval_iters) 决定，与训练
            # 种子无关：给每个窗口写入稳定 key，供跨变体对齐。
            keyed = []
            for w in wins:
                if not isinstance(w, dict):
                    continue
                w = dict(w)
                w.setdefault("split", cand.get("split"))
                w["sample_id"] = f"lm:{w.get('split') or '?'}#w{w.get('window')}"
                keyed.append(w)
            all_keyed.extend(keyed)
            window_seed_counts[v] = window_seed_counts.get(v, 0) + 1
            if first_keyed is None:
                first_keyed = keyed
        if all_keyed:
            per_variant_windows[v] = all_keyed
            per_variant_windows_first[v] = first_keyed
    win_cross = None
    if per_variant_windows:
        md.append("PPL 是所有窗口的平均，看不出「模型究竟错了哪句」。下表按窗口"
                  "末位交叉熵挑出各变体最差的一次 next-token 预测并归因："
                  "高置信错答=top1 概率≥0.5 却错（系统性盲区，含高置信复读）；"
                  "复读吸引子=低置信下首选续写退化为抄上下文已有串；低置信="
                  "模型在多个候选间平均（样本本身多解/噪声，或训练不足）。"
                  "逐案例详情见 `worstcases_*.md`。")
        n_win_seeds = max(window_seed_counts.values()) if window_seed_counts else 0
        if n_win_seeds > 1:
            md.append("")
            md.append(f"注：下表按最多 {n_win_seeds} 个种子累计最差窗口——同一窗口"
                      "在 k 个种子中都差就计 k 次，用于判断错误模式是否跨种子"
                      "稳定；跨变体对照仍取各变体最小种子的同一批窗口。")
        # 最差窗口的“分母”必须可见：只报 top-10 而不报候选/总窗口数，会把
        # 候选子集里的最差误读成全量最差，也会让不同覆盖率的变体不可比。
        scopes = set()
        for v in variants:
            for s in _sorted_seeds(_variant_seed_map(runs, v)):
                cand = (_variant_seed_map(runs, v)[s].get("eval") or {})
                sc = cand.get("worst_window_scope")
                screened = (_integral_value(sc.get("screened"))
                            if isinstance(sc, dict) else None)
                total = (_integral_value(sc.get("total"))
                         if isinstance(sc, dict) else None)
                # 分母缺失/非法时不能写成 0 再按 100% 覆盖率展示：那会把
                # “未知筛查范围”伪装成“已覆盖全量窗口”，并输出 30/0。
                if (screened is not None and total is not None
                        and total > 0 and 0 <= screened <= total):
                    scopes.add((screened, total))
        if scopes:
            md.append("")
            lo_cov = min(scr / tot for scr, tot in scopes)
            shown = "；".join(f"{scr}/{tot}" for scr, tot in sorted(scopes)[:3])
            extra = " 等" if len(scopes) > 3 else ""
            md.append(f"注：最差窗口在等间隔候选窗口子集中排序（本次 "
                      f"{shown}{extra}，最低覆盖率 {_fmt_pct0(lo_cov)}），不是全量窗口"
                      "的严格最差；各变体独立选各自的候选窗口，跨变体的归因"
                      "占比是描述性对照，不是同一批样本的配对比较。")
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
            gic = _fmt_pct0(ana["gold_in_context_rate"])
            g5 = _fmt_pct0(ana["gold_in_top5_rate"])
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
            if c.get("gold_in_context") is True:
                tag = "gold 在上下文中"
            elif c.get("gold_in_context") is False:
                tag = "gold 不在上下文中"
            else:
                tag = "gold 是否在上下文中未记录"
            md += [f"- **{VARIANT_CN.get(v, v)} · 窗口 #{c.get('window')}"
                   + (f"（{c['split']}）" if c.get("split") else "")
                   + f"** —— {WINDOW_CATEGORY_CN.get(c['category'], c['category'])}",
                   f"  - 上下文尾部：…{tail}",
                   f"  - 模型 top-1：`{c.get('top1')}`{p1_s}；真实续写："
                   f"`{c.get('gold_next')}`（{tag}）"]
        if len(per_variant_windows_first) >= 2:
            win_cross = cross_variant_analysis(per_variant_windows_first,
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
        md.append("- 暂无最差窗口记录（评测未含 --worst_n、评测被跳过，或 test "
                  "split 默认尘封最差窗口转储——需人工迭代请在 val 上做，"
                  "确需 test 错误清单须显式 --worst_on_test 并在报告中披露）。"
                  "PPL 平均分无法替代对错误样本的逐条检查。")

    # --- 消融矩阵（全因子 + 去一法）
    present = set(runs.keys())
    md += ["", "## 消融矩阵（全因子 + 去一法）", ""]
    rows_by_var = {r["variant"]: r for r in rows}
    factorial_done, factorial_missing, factorial_under = _ablation_coverage(
        FACTORIAL_CELLS, present, rows_by_var)
    loo_done, loo_missing, loo_under = _ablation_coverage(
        LOO_CELLS, present, rows_by_var)
    control_done, control_missing, control_under = _ablation_coverage(
        CONTROL_CELLS, present, rows_by_var)
    underpowered_cells = factorial_under + loo_under + control_under

    def _cell_label(name, v):
        r = rows_by_var.get(v)
        if not r:
            return name, f"{v}（未运行）"
        n = _integral_value(r.get("n_seeds"))
        if n is None or n < 0:
            n = 0
        if n < MIN_SEEDS:
            return name, f"{v}（种子不足 n={n}<{MIN_SEEDS}）"
        return name, f"{VARIANT_CN.get(v, v)}（n={n}）"

    md.append("| 组合 | 变体 | PPL↓ | RULER平均↑ |")
    md.append("|---|---|---|---|")
    for name, v in FACTORIAL_CELLS:
        label, cell = _cell_label(name, v)
        r = rows_by_var.get(v)
        if r:
            md.append(f"| {label} | {cell} | "
                      f"{fmt_stat_cell(r, 'val_ppl')} | "
                      f"{fmt_stat_cell(r, 'ruler_avg')} |")
        else:
            md.append(f"| {label} | {cell} | - | - |")
    for name, v in CONTROL_CELLS:
        label, cell = _cell_label(name, v)
        r = rows_by_var.get(v)
        if r:
            md.append(f"| {label} | {cell} | "
                      f"{fmt_stat_cell(r, 'val_ppl')} | "
                      f"{fmt_stat_cell(r, 'ruler_avg')} |")
        else:
            md.append(f"| {label} | {cell} | - | - |")
    if (not factorial_missing and not loo_missing and not control_missing
            and not underpowered_cells):
        md.append("")
        md.append("全因子（门控内 2^3，none=`gate_only`）覆盖 8/8、去一法覆盖完整、"
                  "无门控 `standard` 外部基线已运行，且每格种子数≥"
                  f"{MIN_SEEDS}；但组合覆盖只是必要条件，"
                  "组件是否有用须以下方统计证据为准。`gate_only` 只含可学习门偏置 d，"
                  "是“有门但无 A/B/C”的对照：standard vs gate_only 度量门本身，"
                  "X-only vs gate_only 才是组件 X 的受控效应。")
    elif not factorial_missing and not loo_missing:
        md.append("")
        md.append("全因子（门控内 2^3）8/8、去一法完整；⚠ 缺少无门控 `standard` "
                  "外部基线：H1（LEGA vs 标准注意力）无法估计。组件 A/B/C 的判定"
                  "不依赖 standard，仍以 `gate_only` 为受控对照。")
    elif not factorial_missing:
        md.append("")
        md.append("⚠ 全因子（门控内 2^3）覆盖 8/8，但去一法缺失组合："
                  + "、".join(f"{n}({v})" for n, v in loo_missing)
                  + "；Full−X 的组件结论不完整。")
    elif not loo_missing:
        md.append("")
        md.append("⚠ 已覆盖去一法（Full/Full−A/Full−B/Full−C），但全因子缺失/种子不足："
                  + "、".join(f"{n}({v})" for n, v in factorial_missing)
                  + "；涉及这些组合的单独组件结论不完整。")
    else:
        md.append("")
        md.append("⚠ **消融不完整**：全因子缺失/种子不足 "
                  + "、".join(f"{n}({v})" for n, v in factorial_missing)
                  + "；去一法缺失/种子不足 "
                  + "、".join(f"{n}({v})" for n, v in loo_missing)
                  + ("；无门控 standard 外部基线缺失/种子不足 "
                     + "、".join(v for _, v in control_missing)
                     if control_missing else "")
                  + "。仅凭部分组合无法支撑“每个组件都有用”，请补齐后重跑。")
    if underpowered_cells:
        md.append("")
        md.append("⚠ **消融格种子数不足**（"
                  + f"要求每格≥{MIN_SEEDS} 个种子，单种子无法做配对检验）："
                  + "、".join(f"{n} n={k}" for n, _v, k in underpowered_cells[:8])
                  + (" ..." if len(underpowered_cells) > 8 else "")
                  + "；这些格不参与覆盖完整性宣称。")

    # --- 单因素消融审计：需要单因素的实验必须每次只改一个因子。
    md += ["", "### 单因素消融审计（每次只改变一个因子）", ""]
    if sf_audit["ok"]:
        md.append("✅ 组件归因所用的 6 组消融对比均只改变一个门控因子"
                  "（去一法 Full−X vs Full，全因子增量 X-only vs `gate_only`）；"
                  "不存在“B-only/C-only vs Full”这类同时改两个因子却被当作"
                  "单组件证据的对比。")
    else:
        md.append("⚠ **单因素消融审计未通过**："
                  + "；".join(sf_audit["problems"])
                  + "。这些对比不能用于组件归因，需先修正变体/配对定义。")
    md.append("")
    md.append("| 对比 | 方向 | 改变的因子 | 实际因子差 |")
    md.append("|---|---|---|---|")
    direction_cn = {"remove": "去一法：去掉该因子",
                    "add": "全因子增量：只加该因子"}
    for c in sf_audit["contrasts"]:
        md.append(f"| {c['candidate']} vs {c['reference']} "
                  f"| {direction_cn.get(c['direction'], c['direction'])} "
                  f"| {c['factor']} "
                  f"| {', '.join(c['changed_factors'])} |")
    md.append("")
    md.append("注: 全因子 2^3 本身是多因子设计（用于 A×B/A×C/B×C 交互检验，"
              "见下表，属探索性）；组件主效应只从上述单因素边估计。"
              "各变体 vs `standard` 与 LEGA vs `qgate` 是整体方法对比，"
              "可同时含多个因子，不得把差异归因给某一个 A/B/C 组件。"
              "单因素审计结果同时写入 `significance.json` 的 "
              "`single_factor_audit`。")

    md += _render_budget_sensitivity(budget_sensitivity)

    # --- 组件有效性判定：只有 Holm 校正后的显著证据才能支撑“组件有用”
    md += ["", "### 组件有效性判定（同种子配对 t 检验，Holm 校正后 p<0.05 才算证据）", ""]
    md.append("判定只用预注册主指标（A/B=PPL，C=RULER）；另一指标仅作探索性"
              "补充，不参与判定，避免两个指标族分别校正后取 OR 放大假阳性。")
    md.append("")
    md.append("| 组件 | 主指标 | 去一法 Full−X vs Full | 全因子 X vs gate_only | 判定 |")
    md.append("|---|---|---|---|---|")

    def _fmt_evidence(st):
        p = st.get("p_adj")
        ps = f"p_adj={p:.4f}" if p is not None else "p_adj=-"
        return f"{'✅' if st['supported'] else '—'} {ps}（{st['reason']}）"

    for c in comp:
        loo_cells = "；".join(f"{m}: {_fmt_evidence(c['loo'][m])}"
                             for m in _EVIDENCE_METRICS)
        add_cells = "；".join(f"{m}: {_fmt_evidence(c['add'][m])}"
                             for m in _EVIDENCE_METRICS)
        verdict = c["verdict"]
        if not c.get("training_budget_comparable", True):
            verdict += "（探索性：训练预算不等）"
        md.append(f"| {c['component']} | {c['primary_metric']} | {loo_cells} | "
                  f"{add_cells} | {verdict} |")
    supported = [c["component"] for c in comp if c["supported"]]
    unsupported = [c["component"] for c in comp if not c["supported"]]
    fragile = [c["component"] for c in comp if c.get("fragile")]
    inconsistent = [c["component"] for c in comp if not c.get("consistent")]
    sf_ok = bool(sf_audit["ok"])
    if not comp:
        md.append("")
        md.append("⚠ 无法判定组件有效性（无配对检验结果）。")
    elif (not unsupported and not fragile and not factorial_missing
          and not loo_missing and not inconsistent and sf_ok):
        md.append("")
        md.append("✅ 统计证据支持“各组件均有用”：A/B/C 三个组件均在预注册主指标上"
                  "有 Holm 校正后的显著且稳健证据，去一法与全因子两套消融一致，"
                  "且门控内全因子 8/8、去一法完整。")
    elif (not unsupported and not fragile and not factorial_missing
          and not loo_missing and sf_ok):
        md.append("")
        md.append("✅ 统计证据支持各组件有用，但仅部分组件同时得到两套消融支持："
                  + "、".join(inconsistent)
                  + "；另一套未检出可能只是功效不足，报告时不得写成"
                  "“两套消融交叉印证”。")
    elif (not unsupported and not factorial_missing and not loo_missing
          and fragile and sf_ok):
        md.append("")
        md.append("⚠ **不能下“各组件均有用”的稳健结论**："
                  + "、".join(fragile)
                  + " 的证据虽通过 Holm 校正，但 |Δ|<ΔSD（种子噪声内）或"
                  "留一方向翻转；这类显著结果只能作为探索性线索，"
                  "需增加种子/降低方差后重跑才能确认。")
    else:
        md.append("")
        if not sf_ok:
            md.append("⚠ **不能下组件有效性结论**：单因素消融审计未通过，"
                      "上述对比不满足“每次只改一个因子”，不能用于组件归因。")
        if unsupported:
            md.append("⚠ **不能下“各组件均有用”的结论**："
                      + "、".join(unsupported)
                      + " 缺少主指标上 Holm 校正后的显著且稳健证据（幅度不足、"
                      "方向相反、噪声内、证据冲突或种子数不够）；"
                      "组合齐全并不等于组件有效。")
        if fragile:
            md.append("⚠ 不稳健组件：" + "、".join(fragile)
                      + "（显著但效应落在种子噪声内或留一翻转）。")
        if factorial_missing:
            md.append("⚠ 全因子缺失组合："
                      + "、".join(f"{n}({v})" for n, v in factorial_missing)
                      + "，组件单独贡献的覆盖不完整。")
        if loo_missing:
            md.append("⚠ 去一法缺失："
                      + "、".join(f"{n}({v})" for n, v in loo_missing)
                      + "，Full−X 的组件结论不完整。")
    if control_missing:
        md.append("")
        md.append("⚠ 缺少无门控 `standard` 外部基线：H1（LEGA vs 标准注意力）"
                  "无法估计；这不影响门控内 A/B/C 的受控判定。")
    stopped_comp = _as_text_list((audit or {}).get("early_stop_runs"))
    if stopped_comp:
        md.append("")
        md.append(f"⚠ 早停导致实际训练量不等（{len(stopped_comp)} 个 run）："
                  "下方组件证据仍按配对检验数值报告，但属于探索性分析；"
                  "确认性结论需在相同实际步数下重跑。")
    md.append("")
    md.append("注: 去一法证据=去掉该组件的 Full−X 显著差于 Full（去掉有害）；"
              "全因子证据=仅含该组件的 X 显著优于 gate_only（只加它、且已有"
              "门偏置时仍有收益）。判定要求方向正确、Holm 校正后 p<0.05、"
              "种子数≥3，且效应不落在 |Δ|<ΔSD 的种子噪声内、留一法不翻转；"
              "任一套显著反向记“证据冲突”，以证据不足处理。")

    # --- 全因子交互效应（探索性）：主效应有用 ≠ 可加。A 的收益可能在已有
    #     B 时缩小（次可加）或放大（协同）；以 gate_only 为无组件基准，
    #     逐种子算 8 个 gated 单元格的 2^3 交互对比并检验。
    inter = [t for t in tests if t.get("interaction")]
    md += ["", "### 门控内 2^3 交互效应（探索性，独立 Holm 家族）", "",
           "以 `gate_only`（有门但无 A/B/C）为无组件基准，比较 8 个 gated "
           "单元格；交互=0 表示两个组件的效应可加，显著非零说明同时加入"
           "两者时收益不能按单组件加总。该检验未预注册方向，只报是否偏离"
           "可加性与 Δ 符号（PPL 越小越好、RULER 越大越好，符号解释随指标而变）。",
           "", "| 交互 | 指标 | Δ均值 ± Δ标准差 | n | p_adj(Holm) | "
           "结论 |", "|---|---|---|---|---|---|"]
    if inter:
        for t in inter:
            delta = _fmt_mean_std(t)
            padj = (f"{t['p_adj']:.4f}" if t.get("p_adj") is not None else "-")
            md.append(f"| {t['interaction']} | {t['metric']} | {delta} | "
                      f"{t['n']} | {padj} | {t.get('verdict', '-')} |")
    else:
        md.append("（缺少完整 8 格或全部种子不足，无法计算交互对比）")
    md.append("")
    md.append("注: 交互对比以 gate_only 为无组件基准，对第三个因子的两个水平"
              "取平均，逐种子单样本配对检验；n<3 或 Holm 校正后不显著只表示"
              "未检出，不等于交互不存在。")

    # --- 全因子 Shapley 分解（探索性）：单因素边依赖参照格，交互非零时
    #     不同参照格给出不同的“X 的贡献”；Shapley 对全部加入顺序取平均，
    #     且效率性保证 φ_A+φ_B+φ_C 恰等于 Full−none 的总效应。
    md += ["", "### 全因子 Shapley 效应分解（探索性，交互感知）", ""]
    md.append("单因素消融边回答“固定其他因子时删/加 X 会怎样”，但参照格不同"
              "时结论可能不同；Shapley 值把 Full−none 的总效应按所有加入顺序"
              "平均分摊给 A/B/C，天然把 A×B/A×C/B×C 交互平均进参与因子。"
              "PPL 口径在 −log(PPL) 上分解（比值尺度可加），φ>0 表示该因子"
              "降低 PPL，`exp(−φ)` 即其 PPL 几何均值比；RULER 用原始 EM。"
              "占比只在每个种子的总效应均为正时报告，否则留空、不强行归一化。"
              "该分解未预注册，属探索性证据，不替代预注册的单因素确认检验。")
    if shapley_tests:
        md += ["", "| 因子 | 指标 | φ均值 ± SD | 95% CI | 占总效应 | "
               "PPL几何比 | n | p_adj(Holm) | 结论 |",
               "|---|---|---|---|---|---|---|---|---|"]
        for t in shapley_tests:
            factor = t.get("shapley_factor_label") or t.get("shapley_factor")
            delta = _fmt_mean_std(t)
            ci = t.get("mean_diff_ci95")
            ci_s = (f"[{ci[0]:+.4f}, {ci[1]:+.4f}]" if ci else "-")
            share = _fmt_pct(t.get("shapley_share"), ".1%")
            ratio = (f"{t['shapley_ppl_ratio']:.4f}"
                     if t.get("shapley_ppl_ratio") is not None else "-")
            padj = (f"{t['p_adj']:.4f}"
                    if t.get("p_adj") is not None else "-")
            md.append(f"| {factor} | {t.get('shapley_metric')} | {delta} | "
                      f"{ci_s} | {share} | {ratio} | {t['n']} | {padj} | "
                      f"{t.get('verdict', '-')} |")
        md.append("")
        md.append("注: φ 的符号方向=“越大越好”（PPL 已转为 −log）；"
                  "“占总效应”为逐种子 Shapley 份额的均值；总效应为 Full−none，"
                  "效率性残差见 `significance.json` 的 "
                  "`shapley_efficiency_gap_max`（浮点误差量级）。"
                  "Shapley 显著不等于单因素边也显著：若两者方向冲突，"
                  "说明贡献依赖其他因子的水平（交互），必须在报告中并列披露。")
    else:
        md.append("")
        md.append("⚠ 无法计算 Shapley 分解：门控内 8 格未全部运行、种子不足，"
                  "或各格评测协议不一致；此时只能参考上面的单因素边与交互表。")

    md += _render_reparameterization(reparameterization)

    # --- 推理期因果病灶消融（探索性）：训练期消融回答“重新训练时去掉
    #     组件会怎样”，本表回答“训练好的模型推理时是否真的使用该组件”。
    md += ["", "### 推理期因果消融（权重病灶，val split；探索性）", ""]
    md.append("训练期消融改变优化轨迹，不能证明最终前向仍在**使用**某组件；"
              "本表在同一 checkpoint、同一批 val 窗口上只改门控参数，"
              "逐窗口配对比较 ΔNLL=病灶−正常（>0 表示该机制在推理期有用）。"
              "`no_query/no_conc/no_len/no_evidence` 是训练期去一法的推理期"
              "对应；`gate_open/closed/constant_mean` 检验门控写回强度与逐位置"
              "变化；`head_permute` 在参数个数、分布、结构完全不变的前提下打乱"
              "头—证据对应，是容量解释的安慰剂对照；"
              "`conc_permute/query_permute/len_permute` 分别把 B/A/C 位置证据"
              "沿序列维循环移位，保持逐值分布、均值/方差不变而只破坏位置对应；"
              "`sign_flip` 反转证据方向。"
              "默认只在 val 上做（test 尘封；val 同时是 checkpoint 选择集，"
              "故绝对水平偏乐观，本表只作机制定位）。跨种子 p 经 Holm 校正；"
              "窗口级 p 仅描述单 run，不作为确认性证据。")
    if causal_tests:
        md += ["", "| 变体 | 病灶 | 目标机制 | ΔNLL均值 ± SD | PPL比 | n种子 | "
               "逐种子胜负 | 留一 | p_adj(Holm) | 判定 |",
               "|---|---|---|---|---|---|---|---|---|---|"]
        for t in causal_tests:
            delta = _fmt_mean_std(t)
            ratio = (f"{t['mean_ppl_ratio']:.4f}"
                     + (" ⚠截断" if t.get("ppl_ratio_clamped") else "")
                     if t.get("mean_ppl_ratio") is not None else "-")
            wl = _seed_winloss_str(t)
            loo = _loo_stability_str(t)
            padj = (f"{t['p_adj']:.4f}"
                    if t.get("p_adj") is not None else "-")
            md.append(f"| {t['variant']} | {t['lesion_label']} | "
                      f"{t.get('target', '-')} | {delta} | {ratio} | {t['n']} | "
                      f"{wl} | {loo} | {padj} | {t.get('causal_verdict', '-')} |")
        md.append("")
        md.append("注: `PPL比=exp(ΔNLL均值)`（>1 表示病灶更差）；ΔNLL 与 PPL 比"
                  "只在同一变体/同一 checkpoint 内解释，不跨变体比较绝对值。"
                  "`no_op` 表示该组件训练后系数已为 0（推理期未使用），"
                  "此时“无差异”不是“无效应”的证据；`gate_closed` 是阳性对照，"
                  "若关闭注意力输出仍无差异，应先排查实现/评测而非解释方法。"
                  "训练期消融显著而推理期病灶无差异，说明组件收益可能来自训练"
                  "期的优化/正则化而非推理期必需；推理期显著而训练期不显著，"
                  "则提示训练期消融被再训练补偿。两种证据并列，不互相替代。"
                  "`no_len` 只去 b·log n；c 中的 log n 仍随 a·c 保留，"
                  "因此它检验的是“额外长度系数 C”，不是“完全移除长度依赖”。"
                  "`no_*_matched` 是活动匹配对照：去掉同一组件的同时逐层逐头"
                  "平移 d，使平均 g 与基线相同；它只匹配平均门控（一阶矩），"
                  "不匹配方差/饱和比例，故不能把匹配版写成“机制完全匹配”。")
    else:
        md.append("")
        md.append("⚠ 未发现推理期因果消融结果（`--skip_ablation`、"
                  "`--ablate_variants` 为空，或变体不含输出门）；"
                  "组件结论目前只有训练期消融证据，不能据此声称“推理期使用了"
                  "该组件”。")

    # --- 活动匹配对照：原始病灶 − 匹配病灶。若差值显著为正，说明原始
    #     ΔNLL 中有一部分只是“门平均开度变了”，不能全部归因于信息内容；
    #     若差值不显著，也不能反过来说原始效应全部来自信息内容（功效问题）。
    if matched_ablation_contrasts:
        md += _render_matched_ablation_contrasts(matched_ablation_contrasts)
    elif causal_tests:
        md += ["", "⚠ 未发现可用的活动匹配对照（记录缺失、未成功施加或无法"
               "配对）；原始组件 ΔNLL 仍可能混入平均门控开度变化，不能单独"
               "归因给信息内容。"]

    # --- 关键病灶的等价性解读（尤其安慰剂/阳性对照）：p>0.05 不能直接
    #     写成“安慰剂无效应”，必须看 TOST 是否把差异限制在 SESOI 内。
    md += _render_causal_equivalence_section(causal_tests)

    # --- 训练期参数匹配安慰剂（容量/优化红利的对照）与同一 checkpoint 内的
    #     2^3 推理期 Shapley（与训练期 Shapley 并列）。
    md += _render_parameter_matched_control(
        param_control_test, capacity_only_test,
        ran=PARAM_CONTROL_VARIANT in runs)
    md += _render_inference_shapley(inference_shapley)
    md += _render_inference_interactions(inference_interactions, tests)

    _assigned = {
        "_cell_label",
        "_cnt",
        "_ev",
        "_f4",
        "_fields",
        "_fmt_evidence",
        "_load_dict",
        "_seed",
        "_seed_map",
        "_slots",
        "_v",
        "acc",
        "add_cells",
        "all_keyed",
        "ana",
        "any_early",
        "any_grad_spike",
        "any_inversion",
        "any_loss_spike",
        "any_malformed",
        "any_nan",
        "any_selection_noise",
        "attention_s",
        "bad_cross",
        "bad_example_pool",
        "bad_gate",
        "bad_rows",
        "badcases",
        "by_seed",
        "c",
        "c0",
        "cand",
        "cases",
        "cat_s",
        "cats",
        "cell",
        "ci",
        "ci_s",
        "commit",
        "control_done",
        "control_missing",
        "control_under",
        "corpus_sources",
        "ctx_s",
        "curve",
        "d",
        "d_seed",
        "data",
        "delta",
        "dfr",
        "diags",
        "direction_cn",
        "dirty",
        "ep_min",
        "ep_s",
        "eps",
        "es",
        "es_flagged",
        "es_s",
        "expected_sanity_seed",
        "extra",
        "factor",
        "factorial_done",
        "factorial_missing",
        "factorial_under",
        "files",
        "fin",
        "first_keyed",
        "fp",
        "fragile",
        "g5",
        "gate_order",
        "gate_rows",
        "gic",
        "git",
        "hi",
        "hs",
        "inconsistent",
        "ini",
        "inter",
        "kept",
        "keyed",
        "label",
        "layer_s",
        "layers",
        "lo",
        "lo_cov",
        "loo",
        "loo_cells",
        "loo_done",
        "loo_missing",
        "loo_under",
        "lspike",
        "m",
        "m_last",
        "md",
        "ml",
        "more",
        "mt1",
        "n_attention",
        "n_bad_gate",
        "n_label",
        "n_seeds_max",
        "n_shared",
        "n_shared_w",
        "n_win_seeds",
        "name",
        "observability",
        "p1",
        "p1_s",
        "p_adj",
        "p_s",
        "padj",
        "per_variant_badcases",
        "per_variant_windows",
        "per_variant_windows_first",
        "pred",
        "present",
        "pv",
        "q",
        "r",
        "ratio",
        "rec",
        "rows_by_var",
        "rt",
        "runtime_drift_rows",
        "s",
        "s_last",
        "sanity_files",
        "sc",
        "scopes",
        "screened",
        "seen_cat",
        "sf_ok",
        "share",
        "shared_ci",
        "shared_ci_s",
        "shown",
        "spike",
        "src_s",
        "status",
        "status_g",
        "status_s",
        "stopped_comp",
        "supported",
        "sv",
        "t",
        "tag",
        "tail",
        "total",
        "train_s",
        "underpowered_cells",
        "unsupported",
        "v",
        "val_s",
        "verdict",
        "w",
        "win_cross",
        "window_advice",
        "window_pool",
        "window_seed_counts",
        "wins",
        "wl",
        "worst",
        "worst_g",
    }
    return {k: v for k, v in locals().items() if k in _assigned}


def _render_reparameterization(reparam):
    """渲染归一化 c vs 原始 log max_p 的参数化稳健性（探索性）。"""
    if not isinstance(reparam, dict) or not reparam.get("available"):
        if not isinstance(reparam, dict) or reparam.get("not_run"):
            return []
        md = ["", "### 参数化稳健性（原始 log max_p vs 归一化 c；探索性）", ""]
        reason = reparam.get("reason") or "原始/归一化网格不完整"
        md.append("⚠ 参数化稳健性分析不可用：" + str(reason))
        missing = reparam.get("missing_variants") or []
        if missing:
            md.append("")
            md.append("缺少变体：" + "、".join(str(v) for v in missing))
        return md

    def _num(value, spec="+.4f"):
        f = _finite_number(value)
        return format(f, spec) if f is not None else "-"

    def _ci(test):
        ci = test.get("mean_diff_ci95")
        return (f"[{ci[0]:+.4f}, {ci[1]:+.4f}]"
                if ci else "-")

    md = ["", "### 参数化稳健性（原始 log max_p vs 归一化 c；探索性）", ""]
    md.append(
        "归一化 c=log(n·max_p) 与原始 log max_p 属于同一线性门控函数族；"
        "前者把 a·log n 并入 B，导致 B 与显式长度项 C 不正交。本表按同种子"
        "配对比较两种参数化下的单因子边与 Shapley 贡献：D=原始−归一化，"
        "PPL 的效应已定向为 −log PPL（正值=候选降低 PPL 更多），RULER 为"
        "原始 EM 差（正值=原始参数化改善更大）。D 显著非零表示因子归因"
        "依赖参数化，不能把归一化 c 的效应直接解释为纯集中度；D 不显著"
        "只表示未拒绝稳健，不证明等价，须对照 MDE。")
    metrics = reparam.get("metrics") if isinstance(
        reparam.get("metrics"), dict) else {}
    for metric in ("PPL", "RULER平均"):
        result = metrics.get(metric)
        if not isinstance(result, dict):
            continue
        contrasts = [t for t in (result.get("contrasts") or [])
                     if isinstance(t, dict)]
        shapley = [t for t in (result.get("shapley") or [])
                   if isinstance(t, dict)]
        if not contrasts and not shapley:
            continue
        summary = result.get("summary") if isinstance(
            result.get("summary"), dict) else {}
        md += ["", f"**{metric}**（n≈{result.get('n_protocol_seeds', '-')}；"
              f"{summary.get('verdict', '-')}）", ""]
        if contrasts:
            md += ["| 因子边 | 归一化效应 | 原始效应 | 原始−归一化 [95% CI] | "
                   "n | 种子符号一致 | p_adj(Holm) | 判定 |",
                   "|---|---|---|---|---|---|---|---|"]
            for t in contrasts:
                agree = t.get("seed_sign_agreement")
                agree_s = f"{agree:.0%}" if isinstance(agree, (int, float)) else "-"
                md.append(
                    f"| {t.get('factor', '-')}（{t.get('direction', '-')}） "
                    f"| {_num(t.get('normalized_effect_mean'))} "
                    f"| {_num(t.get('raw_effect_mean'))} "
                    f"| {_num(t.get('mean_diff'))} {_ci(t)} "
                    f"| {t.get('n', '-')} | {agree_s} "
                    f"| {_num(t.get('p_adj'), '.4f')} "
                    f"| {t.get('verdict', '-')} |")
        if shapley:
            md += ["", "| Shapley 因子 | 归一化 φ | 原始 φ | 原始−归一化 [95% CI] | "
                   "n | p_adj(Holm) | 判定 |", "|---|---|---|---|---|---|---|"]
            for t in shapley:
                md.append(
                    f"| {t.get('factor_label') or t.get('factor', '-')} "
                    f"| {_num(t.get('normalized_phi_mean'))} "
                    f"| {_num(t.get('raw_phi_mean'))} "
                    f"| {_num(t.get('mean_diff'))} {_ci(t)} "
                    f"| {t.get('n', '-')} "
                    f"| {_num(t.get('p_adj'), '.4f')} "
                    f"| {t.get('verdict', '-')} |")
            md.append("")
            md.append("注：φ 的符号仍为“越大越好”；效率性残差分别见 "
                      "`normalized_efficiency_gap_max` 与 "
                      "`raw_efficiency_gap_max`。")
    return md


def _render_length_null(report, stale: bool = False) -> list:
    """渲染 c~log2(n) 零假设诊断（README 要求的长度主张必要证据）。"""
    md = ["", "## 长度零假设诊断（c ~ log2 n）", ""]
    if not isinstance(report, dict):
        md.append("- ⚠ 未找到 `length_null.json`：任何“长度解耦/长度校准”"
                  "正式主张都缺少 README 要求的 c~log2(n) 斜率证据；"
                  "不能用均匀注意力特例单元测试替代。请重跑统一流水线生成诊断。")
        return md
    if stale:
        md.append("- ⚠ `length_null.json` 不属于本轮 `run_id`（旧结果或手工"
                  "复制）：不得作为本轮长度主张的证据。")
    lengths = report.get("lengths")
    if not isinstance(lengths, list) or not lengths:
        md.append("- ⚠ 诊断缺少有效长度清单，无法复核斜率。")
        return md
    try:
        score_sd = float(report.get("score_sd"))
    except (TypeError, ValueError, OverflowError):
        score_sd = None
    md.append(
        f"- 长度档：{', '.join(str(x) for x in lengths)}；"
        f"每档重复 {report.get('n_trials')} 次；"
        f"打分标准差={'-' if score_sd is None else f'{score_sd:.3g}'}；"
        f"随机种子={report.get('seed')}。")
    modes = report.get("modes")
    if not isinstance(modes, dict) or not modes:
        md.append("- ⚠ 诊断缺少 modes 结果。")
        return md
    md += ["", "| 零假设模式 | c 均值范围 | c~log2(n) 斜率 | "
               "是否长度无关(|slope|≤0.05) |", "|---|---|---|---|"]
    slopes = {}
    for mode in ("uniform", "repeated", "random"):
        data = modes.get(mode)
        if not isinstance(data, dict):
            continue
        per = data.get("per_length")
        means = []
        if isinstance(per, dict):
            for n in lengths:
                row = per.get(str(n))
                if isinstance(row, dict):
                    try:
                        means.append(float(row.get("mean")))
                    except (TypeError, ValueError, OverflowError):
                        continue
        slope = data.get("slope_c_per_log2_n")
        try:
            slope = float(slope)
        except (TypeError, ValueError, OverflowError):
            slope = None
        slopes[mode] = slope
        span = ("-" if not means else f"{min(means):+.4f} … {max(means):+.4f}")
        indep = data.get("length_independent")
        md.append(f"| {mode} | {span} | "
                  f"{'-' if slope is None else f'{slope:+.4f}'} | "
                  f"{'-' if indep is None else ('是' if indep else '否')} |")
    md.append("")
    random_slope = slopes.get("random")
    if isinstance(random_slope, float) and abs(random_slope) > 0.05:
        md.append(
            f"- ⚠ 随机上下文下 c 仍随长度增长（slope={random_slope:+.4f}）："
            "`c=log(n·max_j p_j)` 只在均匀分布特例下与长度无关，不能把 c "
            "单独解释为“已去除长度混淆的充分证据”。正式结论必须同时报告"
            "该斜率，并与 `b·log n`、RULER 分档结果合读。")
    else:
        md.append("- 随机零假设下 slope 未超过 0.05 阈值；仍需按上表报告，"
                  "不能只引用均匀特例。")
    anomalies = [m for m in ("uniform", "repeated")
                 if isinstance(slopes.get(m), float)
                 and abs(slopes[m]) > 0.05]
    if anomalies:
        md.append("- ⚠ 零斜率对照偏离 0：" + "、".join(anomalies)
                  + "；这通常意味着诊断实现/样本量异常，先排查再解读 random 斜率。")
    md.append("- 边界：本表是对**合成零假设分布**的诊断，不是对训练模型实际"
              "注意力分布的测量；它约束“c 本身已长度解耦”的措辞，不替代模型"
              "级 c 分布与长度分层 RULER 证据。")
    return md
