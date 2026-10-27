"""简单统计/线性探针基线与基线合理性渲染。"""
from .._shared import _install as _shared_install

_shared_install(globals())

def _as_dict(v) -> dict:
    """外部/旧格式 JSON 的嵌套对象：非 dict 统一按空对象处理。"""
    return v if isinstance(v, dict) else {}


def _as_dict_list(v) -> list:
    """只保留列表中的对象元素，供外部 JSON 的告警/访问记录使用。"""
    return [x for x in v if isinstance(x, dict)] if isinstance(v, list) else []


def _as_text_list(v) -> list:
    return [str(x) for x in v] if isinstance(v, list) else []


def _as_text(v) -> str:
    """外部 JSON 的文本字段：非字符串按可读文本降级（缺失为空串）。"""
    return v if isinstance(v, str) else ("" if v is None else str(v))


def _int_or_zero(v) -> int:
    """外部/旧版 JSON 的计数 → 非负整数；bool/小数/负数/坏类型按 0 处理。"""
    iv = _integral_value(v)
    return iv if iv is not None and iv >= 0 else 0


def _fmt_rate(x) -> str:
    f = _finite_number(x)
    return "-" if f is None else f"{f:.3%}"


def _fmt4(x) -> str:
    """四位小数；None/非数值显式显示 '-'（如空 split 的分布漂移统计为 null）。"""
    f = _finite_number(x)
    return "-" if f is None else f"{f:.4f}"


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


def _neural_eval_window_spec(cfg: dict, smoke: bool):
    """推导神经评测实际使用的窗口规格 (ctx, iters)，供 n-gram 逐位置对齐。

    口径必须与 train.py/evaluate.py 完全一致：ctx = min(block_size, train_ctx)
    （冒烟模式先应用 train.py 的 SMOKE_OVERRIDES），iters = eval_iters
    （缺省 50）。规格非法时返回 None（不阻塞基线评测）。
    """
    from lega.train import SMOKE_OVERRIDES
    if not isinstance(cfg, dict):
        return None
    try:
        # 严格整数：int(True)/int(2.5) 会把“同窗口”对齐的 ctx/iters 静默改值，
        # 使 n-gram 与神经评测实际上在不同 token 集合上比较。
        ctx = strict_int(cfg.get("train_ctx") or cfg.get("block_size") or 0,
                         "train_ctx")
        iters = strict_int(cfg.get("eval_iters", 50), "eval_iters")
        if smoke:
            if "train_ctx" in SMOKE_OVERRIDES:
                ctx = strict_int(SMOKE_OVERRIDES["train_ctx"], "smoke.train_ctx")
            if "eval_iters" in SMOKE_OVERRIDES:
                iters = strict_int(SMOKE_OVERRIDES["eval_iters"],
                                   "smoke.eval_iters")
        block = strict_int(cfg.get("block_size") or 0, "block_size")
    except ValueError:
        return None
    if block > 0:
        ctx = min(ctx, block)
    if ctx < 1 or iters < 1:
        return None
    return {"ctx": ctx, "iters": iters}


def _eval_matches_window_spec(ev: dict, wspec: dict) -> bool:
    """run 的 eval JSON 是否与 n-gram 窗口规格逐字段一致。

    字段口径：eval_ctx=实际评测上下文、eval_iters=实际窗口数（已按可用
    窗口数截断）、val_windows=评测 split 的可用窗口数。任一项缺失/不同即
    视为未对齐（历史产物、ctx/iters 被覆盖等情况），回退全 split 参考。
    """
    if not isinstance(ev, dict) or not isinstance(wspec, dict) or not wspec:
        return False
    return (_same_integral_value(ev.get("eval_ctx"), wspec.get("ctx"))
            and _same_integral_value(ev.get("eval_iters"),
                                     wspec.get("n_eval"))
            and _same_integral_value(ev.get("val_windows"),
                                     wspec.get("n_windows")))


def ngram_baseline_comparison(runs: dict, results_dir: str) -> dict:
    """神经变体 vs 简单统计基线（同 split、能校验时同 token）的统计对比。

    科研方法问题：“n-gram PPL=xx” 与 “神经 PPL=yy” 并排放着不构成
    “复杂模型比简单基线好”的证据——必须同 split、必须对照方差，而且
    两边必须打在**同一批 token** 上：神经 PPL 只在 eval_iters 个等间隔
    窗口的目标 token 上计算，n-gram 若在整个 split 上打分就是不同 token
    集合上的 PPL（同 split 不等于同 token）。n-gram 对给定 split 是确定值
    （无种子方差），因此每个神经变体的逐种子 PPL 与同一常数配对 = 单样本
    检验；只在两侧 split 一致的种子上比较，split 不一致的 run 显式列出且
    不混入（避免 test-vs-val 错位）。参考值的优先级：全部纳入比较的 run
    的 eval 字段都与 n-gram 的窗口规格一致时，优先用
    `ppl_window_matched_selected`（在同一批窗口 token 上重新选参），旧
    产物才用 `ppl_window_matched`（全 val 选参）；任一不一致（历史产物/
    规格缺省）则整体回退全 split 参考并在结果里显式标注 `reference_source`
    ——同一 Holm 家族必须共用同一个参考值。多个变体同时对比时沿用
    Holm–Bonferroni 校正。该对比不属于预注册假设，标为补充性/探索性证据。
    """
    path = os.path.join(results_dir, "ngram_baseline.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            ng = json.load(f)
    except (OSError, ValueError):
        return {}
    # 合法 JSON 但非对象（如数组）按不可用处理。
    if not isinstance(ng, dict):
        return {"error": "ngram_baseline.json 不是 JSON 对象"}
    # 旧基线必须绑定到本轮的数据身份：results/<tag> 复用且 --skip_ngram 时
    # 目录里的 n-gram PPL 可能来自另一份数据（甚至重建后的 test.bin），
    # 与当前神经 PPL 并排比较会制造“显著优于统计基线”的假结论。新产物
    # 带 data_fingerprint；旧产物缺该字段时保持兼容（无法核对），但只要有
    # 指纹且与 repro.json 不一致就直接标记不可用。
    repro_path = os.path.join(results_dir, "repro.json")
    if os.path.exists(repro_path):
        try:
            with open(repro_path, "r", encoding="utf-8") as f:
                repro_doc = json.load(f)
        except (OSError, ValueError):
            repro_doc = None
        # repro.json 是外部/旧产物：data 合法 JSON 但不是对象（数组/标量）
        # 时按“无法核对指纹”处理，不能在整个汇总页上抛 AttributeError。
        expected_fp = (_as_dict(repro_doc.get("data")).get("digest")
                       if isinstance(repro_doc, dict) else None)
        actual_fp = ng.get("data_fingerprint")
        if (isinstance(expected_fp, str) and expected_fp
                and isinstance(actual_fp, str) and actual_fp
                and actual_fp != expected_fp):
            return {"error": "ngram_baseline.json 的数据指纹与本次 run 的 "
                             "repro.json 不一致：该基线来自另一份数据，"
                             "不能与当前神经 PPL 直接比较；请重跑 n-gram 基线",
                    "split": ng.get("split"),
                    "ngram_data_fingerprint": actual_fp,
                    "run_data_fingerprint": expected_fp}
    ppl_map = ng.get("ppl")
    if ppl_map is not None and not isinstance(ppl_map, dict):
        return {"error": "ngram_baseline.json 的 ppl 不是 JSON 对象"}
    ppl_map = ppl_map or {}
    # 优先用 n-gram 侧按 val 选出的“最强简单基线”，避免固定拿较弱
    # bigram 当靶子；历史产物没有 selected_model 时再按插值→bigram→unigram
    # 的强度顺序回退。
    ref_name = None
    sel = ng.get("selected_model")
    if isinstance(sel, str) and ppl_map.get(sel) is not None:
        ref_name = sel
    if ref_name is None:
        ref_name = next((k for k in ("interpolated_bigram", "bigram",
                                     "unigram")
                         if ppl_map.get(k) is not None), None)
    if ref_name is None:
        return {"error": "ngram_baseline.json 没有可用 PPL"}
    ref_ppl_full = _finite_number(ppl_map[ref_name])
    if ref_ppl_full is None:
        return {"error": "ngram PPL 非数值"}
    if ref_ppl_full <= 0:
        return {"error": "ngram PPL 非有限值或非正数"}
    wspec = ng.get("window_spec")
    wspec = wspec if isinstance(wspec, dict) else {}
    # 窗口参考优先使用“在同一批窗口 token 上重新选参”的结果：神经模型
    # 的 checkpoint 由 eval_iters 个窗口上的 val loss 选出，n-gram 若仍用
    # 完整 val 选出的 α/λ，选参与比较的 token 集就不同。新产物会落盘
    # `window_matched_selection` + `ppl_window_matched_selected`；旧产物
    # 缺少这些字段时回退到全 val 选参的 `ppl_window_matched`（保持兼容）。
    wm_sel = ng.get("window_matched_selection")
    wm_sel = wm_sel if isinstance(wm_sel, dict) else {}
    wm_map_full = ng.get("ppl_window_matched")
    wm_map_full = wm_map_full if isinstance(wm_map_full, dict) else {}
    wm_map_sel = ng.get("ppl_window_matched_selected")
    wm_map_sel = wm_map_sel if isinstance(wm_map_sel, dict) else {}
    wm_model_sel = wm_sel.get("selected_model")
    wm_ref_name = ref_name
    wm_ref = None
    wm_ref_map = wm_map_full
    if (isinstance(wm_model_sel, str)
            and _finite_number(wm_map_sel.get(wm_model_sel)) is not None):
        wm_ref_name = wm_model_sel
        wm_ref = _finite_number(wm_map_sel.get(wm_model_sel))
        wm_ref_map = wm_map_sel
        ng_win_field = "window_losses_matched_selected"
        ng_idx_field = "window_indices_matched_selected"
        ng_win_model = ng.get("window_losses_matched_selected_model")
    else:
        wm_ref = _finite_number(wm_map_full.get(ref_name))
        wm_ref_map = wm_map_full
        ng_win_field = "window_losses_matched"
        ng_idx_field = "window_indices_matched"
        ng_win_model = ng.get("window_losses_matched_model")
    # 选定窗口参考仍不可用/非正时，回到全 val 选参的历史口径。
    if wm_ref is None or wm_ref <= 0:
        wm_ref_name = ref_name
        wm_ref = _finite_number(wm_map_full.get(ref_name))
        wm_ref_map = wm_map_full
        ng_win_field = "window_losses_matched"
        ng_idx_field = "window_indices_matched"
        ng_win_model = ng.get("window_losses_matched_model")
    # n-gram 逐窗口损失（选定模型）用于真正的窗口级配对，而不是只把
    # 神经逐种子 PPL 与一个常数比较。缺字段/模型不匹配时显式降级。
    ng_win = _eval_window_losses({
        "window_losses": ng.get(ng_win_field),
        "window_indices": ng.get(ng_idx_field)})
    if (ng_win is not None and ng_win_model is not None
            and ng_win_model != wm_ref_name):
        ng_win = None
        ng_win_model = None
    ng_split = ng.get("split")
    # 第一遍：收集逐种子 PPL 与该 run 是否与窗口规格逐字段一致。
    staged, incomparable = [], []
    runs = runs if isinstance(runs, dict) else {}
    for v in sorted(runs):
        if v not in VARIANTS:
            continue
        m = _variant_seed_map(runs, v)
        vals, seeds_used, eval_splits, excl_seeds = [], [], set(), []
        matched = []
        win_maps = []
        for s in _sorted_seeds(m):
            slot = m.get(s)
            d = slot.get("eval") if isinstance(slot, dict) else None
            if not d:
                continue
            ev_split = d.get("split", "val")
            if ev_split != ng_split:
                try:
                    eval_splits.add(ev_split)
                except TypeError:
                    eval_splits.add(str(ev_split))
                excl_seeds.append(s)
                continue
            p = _ppl_of(d)
            if p is None or not math.isfinite(p):
                continue
            vals.append(p)
            seeds_used.append(s)
            matched.append(_eval_matches_window_spec(d, wspec))
            win_maps.append(_eval_window_losses(d))
        if eval_splits:
            # 只要存在 split 错位的种子就必须显式列出（哪怕同变体另有可比
            # 种子已被纳入）：静默丢弃部分种子同样会让读者高估证据完整性
            incomparable.append({"variant": v,
                                 "eval_splits": sorted(eval_splits, key=str),
                                 "excluded_seeds": [
                                     _integral_value(s)
                                     for s in excl_seeds]})
        if vals:
            staged.append({"variant": v, "vals": vals,
                           "seeds": [_integral_value(s)
                                     for s in seeds_used],
                           "matched": matched, "win_maps": win_maps})
    # 参考值决策（第一步）：全部纳入比较的 run 都与窗口规格一致，且
    # window-matched PPL 可用 → 用逐位置对齐参考；否则整体回退全 split。
    # 同一 Holm 家族必须共用同一参考值，不能逐行混用。
    if wm_ref is not None and wm_ref <= 0:
        wm_ref = None
    all_matched = bool(staged) and all(all(r["matched"]) for r in staged)
    use_window = bool(wspec) and wm_ref is not None and all_matched
    ref_ppl = wm_ref if use_window else ref_ppl_full
    ref_source = "window_matched" if use_window else "full_split"
    rows = []
    for st in staged:
        v, vals = st["variant"], st["vals"]
        # 窗口级配对：同一批窗口序号上 d_w = loss_neural − loss_ngram。
        # 仅当全体 run 都采用逐位置对齐参考、n-gram 有选定模型的逐窗口
        # 损失且窗口序号完全一致时才计算；否则明确置 None，不按位置强配。
        win_diffs = []
        win_skip_reason = None
        if not use_window:
            win_skip_reason = "参考已回退全 split，无逐窗口对齐"
        elif ng_win is None:
            win_skip_reason = "n-gram 缺少选定模型的逐窗口损失"
        else:
            for wmap in st.get("win_maps") or []:
                if not wmap or set(wmap) != set(ng_win):
                    win_skip_reason = "神经与 n-gram 的窗口序号不一致"
                    win_diffs = []
                    break
                keys = sorted(wmap)
                arr = np.asarray([wmap[k] - ng_win[k] for k in keys],
                                 dtype=np.float64)
                if arr.size == 0 or not np.all(np.isfinite(arr)):
                    win_skip_reason = "逐窗口损失非有限值"
                    win_diffs = []
                    break
                win_diffs.append(arr)
        win_stats = (_window_bootstrap_stats(win_diffs)
                     if win_diffs else None)
        res = paired_t_test(vals, [ref_ppl] * len(vals))
        res.update({"candidate": v, "reference": ref_name,
                    "metric": "PPL vs ngram", "lower_is_better": True})
        res = _enrich_paired(res, vals, [ref_ppl] * len(vals))
        # 与其它汇总统计同口径：朴素 sum 会在多个 1e308 的合法 PPL 上溢出。
        mean_ppl, std_ppl = mean_std(vals)
        ratio = (mean_ppl / ref_ppl
                 if mean_ppl is not None and ref_ppl > 0 else None)
        if ratio is not None and not math.isfinite(ratio):
            ratio = None
        reduction_pct = ((1.0 - ratio) * 100.0
                         if ratio is not None else None)
        if reduction_pct is not None and not math.isfinite(reduction_pct):
            reduction_pct = None
        rows.append({
            "variant": v, "n": len(vals), "seeds": st["seeds"],
            "mean_ppl": mean_ppl,
            "std_ppl": std_ppl,
            "ratio": ratio,
            "reduction_pct": reduction_pct,
            "p": res.get("p"), "p_adj": None,
            "mean_diff": res.get("mean_diff"),
            "mean_diff_ci95": res.get("mean_diff_ci95"),
            "p_perm": res.get("p_perm"), "mde": res.get("mde"),
            "token_matched": bool(all(st["matched"])) and use_window,
            "window_paired_n_seeds": (win_stats["n_seeds"]
                                      if win_stats else None),
            "window_paired_mean_diff": (win_stats["mean_diff"]
                                        if win_stats else None),
            "window_paired_ci95": (win_stats["mean_diff_ci95"]
                                   if win_stats else None),
            "window_paired_block_ci95": (
                win_stats.get("block_mean_diff_ci95") if win_stats else None),
            "window_paired_block_p": (
                win_stats.get("block_p") if win_stats else None),
            "window_paired_block_length_min": (
                win_stats.get("block_length_min") if win_stats else None),
            "window_paired_block_length_max": (
                win_stats.get("block_length_max") if win_stats else None),
            "window_paired_mean_lag1_autocorr": (
                win_stats.get("mean_lag1_autocorr") if win_stats else None),
            "window_paired_ppl_ratio": (
                float(math.exp(min(20.0, win_stats["mean_diff"])))
                if win_stats else None),
            "window_paired_p": (win_stats["p"] if win_stats else None),
            "window_paired_p_adj": None,
            "window_paired_n_windows_min": (
                win_stats["n_windows_min"] if win_stats else None),
            "window_paired_n_windows_max": (
                win_stats["n_windows_max"] if win_stats else None),
            "window_paired_enough": (
                bool(win_stats and win_stats["n_seeds"] >= MIN_PPL_WINDOW_SEEDS
                     and win_stats["n_windows_min"] >= MIN_PPL_WINDOWS)
                if win_stats else None),
            "window_paired_skip_reason": win_skip_reason,
            "window_paired_verdict": None,
            "verdict": None,
        })
    # 回退原因：把“为什么没用窗口对齐参考”显式落盘，避免读者默认已对齐。
    mismatch_variants = [st["variant"] for st in staged
                         if not all(st["matched"])]
    if use_window:
        fallback_reason = None
    elif not wspec:
        fallback_reason = "ngram_baseline.json 缺少 window_spec（历史产物）"
    elif wm_ref is None:
        fallback_reason = "window-matched PPL 缺失或非数值"
    else:
        fallback_reason = ("存在与窗口规格不一致的 run（eval_ctx/eval_iters/"
                           f"val_windows 任一字段不符）：{mismatch_variants}")
    if not rows:
        return {"split": ng_split, "reference_model": ref_name,
                "ngram_ppl": ref_ppl, "rows": [],
                "reference_source": ref_source,
                "ngram_ppl_full_split": ref_ppl_full,
                "ngram_ppl_window_matched": wm_ref,
                "reference_model_window_matched": wm_ref_name,
                "window_reference_values": wm_ref_map,
                "window_matched_selection": wm_sel or None,
                "window_spec": wspec or None,
                "window_paired_reference_model": ng_win_model,
                "window_paired_available": bool(ng_win is not None),
                "fallback_reason": fallback_reason,
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
    # 窗口级配对也是独立检验族：Holm 校正只作用于有逐窗口数据的变体。
    win_rows = [r for r in rows if r.get("window_paired_p") is not None]
    win_pads = holm_bonferroni([r.get("window_paired_p") for r in win_rows])
    for r, padj in zip(win_rows, win_pads):
        r["window_paired_p_adj"] = padj
        if r.get("window_paired_enough") is False:
            r["window_paired_verdict"] = (
                "证据不足（种子或窗口数低于门槛），仅报点估计/区间")
        elif padj is None:
            r["window_paired_verdict"] = "无法检验"
        elif padj < ALPHA and r["window_paired_mean_diff"] < 0:
            r["window_paired_verdict"] = "窗口级显著优于统计基线（Holm 校正后）"
        elif padj < ALPHA:
            r["window_paired_verdict"] = "窗口级显著劣于统计基线（Holm 校正后）"
        else:
            r["window_paired_verdict"] = "窗口级无显著差异（Holm 校正后）"
    note = ("n-gram 无种子方差：逐种子 PPL 与同一常数配对 = 单样本检验；"
            "本对比为补充性/探索性证据，不替代变体间受控对比")
    if use_window:
        note = ("参考值取自与神经评测逐位置对齐的窗口目标 token"
                f"（窗口规格 ctx={wspec.get('ctx')}, iters={wspec.get('iters')}, "
                f"共 {wspec.get('n_tokens')} 个评分 token）；" + note)
    return {"split": ng_split, "reference_model": ref_name,
            "ngram_ppl": ref_ppl, "rows": rows, "incomparable": incomparable,
            "reference_source": ref_source,
            "ngram_ppl_full_split": ref_ppl_full,
            "ngram_ppl_window_matched": wm_ref,
            "reference_model_window_matched": wm_ref_name,
            "window_reference_values": wm_ref_map,
            "window_matched_selection": wm_sel or None,
            "window_spec": wspec or None,
            "window_paired_reference_model": ng_win_model,
            "window_paired_available": bool(ng_win is not None),
            "fallback_reason": fallback_reason,
            # Holm 家族只包含实际参与检验（p 非 None）的行；n<MIN_SEEDS
            # 的变体不占家族大小，渲染时不能把总行数说成校正家族大小。
            "holm_family_size": sum(1 for r in rows
                                    if r.get("p") is not None),
            "n_rows": len(rows),
            "note": note}


def _probe_oof(analysis: dict, key: str, subkey: str = None):
    """从机制分析 JSON 取线性探针主特征集的 out-of-fold 指标。

    subkey 用于嵌套指标（如 oof.f1.macro / oof.f1_tuned.macro）。
    """
    lp = _as_dict(_as_dict(analysis).get("linear_probe"))
    primary = lp.get("primary")
    fs = _as_dict(_as_dict(lp.get("feature_sets")).get(
        primary if isinstance(primary, str) else ""))
    val = _as_dict(fs.get("oof")).get(key)
    if subkey is not None:
        return _as_dict(val).get(subkey)
    return val


# 机制判别的简单学习基线对照（补充性/探索性）：门控 logit 是语言建模的
# 副产品、训练时从未见过 hit 标签；线性探针在同一批记录上用 hit 标签监督
# 训练（嵌套、按样例分组并按标签分层的 CV；λ/阈值只在内层 OOF 选，
# 且 λ 按与主检验对齐的 PR-AUC/AUROC，见 lega/probe.py）。
# Δ = 探针 − 门控：>0 表示简单监督基线更好（门控的“判别优越性”不成立），
# ≈0 表示门控与最简线性模型打平，<0 才是门控在判别上显著更好。


def linear_probe_baseline_comparison(runs: dict) -> dict:
    """门控 logit vs 线性探针简单监督基线的逐种子配对对照。

    没有这个对照时，“门控 logit 的 AUROC/PR-AUC 高”可能只说明廉价特征
    （log max_p、log n、qterm）本身线性可分，复杂门控结构的贡献无从谈起。
    探针是**有标签监督**的简单基线：探针超过门控属预期，门控必须在
    主指标上不被探针显著超越，才有资格谈“更能判断该不该写入”。
    单统计量规则（max_p）一并列出，作为最低档次的规则系统对照。
    """
    rows, tests_ap, tests_aud, missing = [], [], [], []
    runs = runs if isinstance(runs, dict) else {}
    for v in sorted(runs):
        if v not in VARIANTS:
            continue
        m = _variant_seed_map(runs, v)
        per_seed, has_analysis = [], False
        for s in _sorted_seeds(m):
            slot = m.get(s)
            an = slot.get("analysis") if isinstance(slot, dict) else None
            if not isinstance(an, dict):
                continue
            has_analysis = True
            lp = _as_dict(an.get("linear_probe"))
            primary = lp.get("primary")
            fs = _as_dict(_as_dict(lp.get("feature_sets")).get(
                primary if isinstance(primary, str) else ""))
            oof = _as_dict(fs.get("oof"))
            probe_auprc = _rate_or_none(oof.get("auprc"))
            probe_auroc = _rate_or_none(oof.get("auroc"))
            if probe_auprc is None or probe_auroc is None:
                continue
            probe_f1_fixed = _rate_or_none(
                _as_dict(oof.get("f1")).get("macro"))
            probe_f1_tuned = _rate_or_none(
                _as_dict(oof.get("f1_tuned")).get("macro"))
            per_seed.append({
                # 无 seed 字段（_seed_key 返回 None）时按缺种子信息处理。
                # 非整数身份（bool/小数）同样记 None，绝不截断成别的种子号。
                "seed": _integral_value(s),
                "probe_auprc": probe_auprc,
                "probe_auroc": probe_auroc,
                "probe_f1_fixed": probe_f1_fixed,
                "probe_f1_tuned": probe_f1_tuned,
                "gate_auprc": _rate_or_none(an.get("auprc_logit")),
                "gate_auroc": _rate_or_none(an.get("auroc_logit")),
                "rule_auprc": _rate_or_none(an.get("auprc_maxp")),
                "rule_auroc": _rate_or_none(an.get("auroc_maxp")),
                "base_rate": _rate_or_none(an.get("hit_base_rate")),
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
        probe_f1_fixed_m, _ = _ms("probe_f1_fixed")
        probe_f1_tuned_m, _ = _ms("probe_f1_tuned")
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
            "probe_f1_fixed": probe_f1_fixed_m,
            "probe_f1_tuned": probe_f1_tuned_m,
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
        primary_detection = r.get("primary_detection_metric")
        if primary_detection == "PR-AUC":
            r["verdict"] = (t_ap or {}).get("verdict")
        elif primary_detection == "AUROC":
            r["verdict"] = (t_aud or {}).get("verdict")
        else:
            # “AUROC 与 PR-AUC 并报”（以及旧产物缺失主指标）：不能默认只
            # 看 AUROC。两个检验可能一个显著、另一个不显著，单取任一个都
            # 会隐藏另一指标上的反证；并列渲染两者的判定。
            parts = []
            if t_ap is not None:
                parts.append("PR-AUC：" + str(t_ap.get("verdict") or "-"))
            if t_aud is not None:
                parts.append("AUROC：" + str(t_aud.get("verdict") or "-"))
            r["verdict"] = "；".join(parts) if parts else "-"
    return {
        "metrics": {"ap": LINEAR_PROBE_AP_METRIC,
                    "auroc": LINEAR_PROBE_AUROC_METRIC},
        "rows": rows, "tests": tests, "missing": missing,
        "note": ("线性探针=有标签监督的简单基线（同批记录、嵌套分组 CV、"
                 "按 hit 标签分层；λ 在内层 OOF 按主指标选、阈值在内层 OOF "
                 "按 macro-F1 选，最终 OOF 不参与选参）；"
                 "门控 logit 为语言建模副产品、训练未见 hit 标签。"
                 "Δ=探针−门控：>0 属预期（判别能力本可由线性模型从廉价"
                 "统计量学得），门控必须在主指标上不被探针显著超越才有"
                 "资格谈“结构优越性”。单统计量规则（max_p）为最低档次"
                 "对照。本对比为补充性/探索性证据，不替代预注册的确认性"
                 "检验。"),
    }


def _render_baseline_sections(results_dir: str, smoke=None,
                              ngram_cmp: dict = None,
                              probe_cmp: dict = None,
                              variants=None) -> list:
    """基线合理性小节：简单统计基线 + 基线超参数调优证据。

    科研方法问题：如果复杂方法只与“随手设一个学习率”的弱基线比较，
    涨点可能全部来自超参数红利。本小节渲染两类证据：
      1. 简单基线（一元/二元/插值二元平滑 n-gram）在同一 held-out split 上的 PPL，
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
                    doc = json.load(f)
                # 合法 JSON 但非对象（数组/标量）按缺 repro 处理；smoke 只接受
                # 真正的 bool true，字符串 "false" 不能伪装成冒烟运行。
                smoke = (doc.get("smoke") is True
                         if isinstance(doc, dict) else False)
            except (OSError, ValueError):
                smoke = False
    lines = ["", "## 基线合理性（简单基线 + 调优证据）", ""]
    ngram = None
    nfp = os.path.join(results_dir, "ngram_baseline.json")
    if os.path.exists(nfp):
        try:
            with open(nfp, "r", encoding="utf-8") as f:
                ngram = json.load(f)
            if not isinstance(ngram, dict):
                ngram = None
        except (OSError, ValueError):
            ngram = None
    if ngram:
        ppl_map = ngram.get("ppl")
        ppl_map = ppl_map if isinstance(ppl_map, dict) else {}
        uni = ppl_map.get("unigram")
        bi = ppl_map.get("bigram")
        interp = ppl_map.get("interpolated_bigram")
        split = ngram.get("split")
        lines.append(
            f"- 简单统计基线（train 拟合、{split} 评测；无梯度；α/λ 只在 "
            f"val 上按 mean NLL 选，仍属零训练基线）："
            f"一元平滑 PPL={_fmt_scalar_maybe(uni)}、"
            f"二元平滑 PPL={_fmt_scalar_maybe(bi)}、"
            f"插值二元 PPL={_fmt_scalar_maybe(interp)}、"
            f"val 选出的最强简单模型={ngram.get('selected_model') or '-'}"
            f"（训练 token {_int_or_zero(ngram.get('train_tokens')):,}、"
            f"评测 token {_int_or_zero(ngram.get('eval_tokens')):,}、"
            f"词表 {ngram.get('vocab_size')}）。主科学对比仍是神经变体之间；"
            "该基线只回答“绝对难度有多高、神经模型是否显著优于统计基线”。")
        # 新产物优先展示“在同一批窗口 token 上重新选参”的参考；旧产物
        # 只有全 val 选参的 ppl_window_matched，保持兼容。
        wm_sel = ngram.get("ppl_window_matched_selected")
        wm_sel = wm_sel if isinstance(wm_sel, dict) else {}
        # 比较表实际采用哪张窗口参考值表由 ngram_baseline_comparison 决定
        # （selected 表存在但缺少 selection 元数据时会被忽略）。渲染必须
        # 复用同一选择，否则同一份报告里表格与说明会引用不同基线。
        cmp_window_values = ((ngram_cmp or {}).get("window_reference_values")
                             if isinstance(ngram_cmp, dict) else None)
        if isinstance(cmp_window_values, dict):
            wm = cmp_window_values
        else:
            wm = wm_sel or ngram.get("ppl_window_matched")
        wm = wm if isinstance(wm, dict) else {}
        wsel = ngram.get("window_matched_selection")
        wsel = wsel if isinstance(wsel, dict) else {}
        wspec = ngram.get("window_spec")
        wspec = wspec if isinstance(wspec, dict) else {}
        # 只有比较表确实采用了逐位置对齐参考（全部 run 的窗口规格一致）时，
        # 这些数字才真的打在“神经评测实际使用的窗口 token”上；一旦回退到全
        # split 参考（历史产物/规格不符），同样的数字只是按声明规格算出的
        # 参考值，不能宣称与神经评测同 token——必须显式区分，否则基线证据
        # 被高估。
        token_matched = ((ngram_cmp or {}).get("reference_source")
                         == "window_matched")
        if wm and wspec and token_matched:
            lines.append(
                f"- 逐位置对齐口径（同 split 还不够，两边必须打在同一批 token "
                f"上）：在神经评测实际使用的窗口目标 token 上"
                f"（ctx={wspec.get('ctx')}、iters={wspec.get('iters')}、"
                f"n_eval={wspec.get('n_eval')}、"
                f"共 {_int_or_zero(wspec.get('n_tokens')):,} 个评分 token），"
                f"一元 PPL={_fmt_scalar_maybe(wm.get('unigram'))}、"
                f"二元 PPL={_fmt_scalar_maybe(wm.get('bigram'))}、"
                f"插值二元 PPL={_fmt_scalar_maybe(wm.get('interpolated_bigram'))}"
                f"（{('α/λ 在同一批窗口 token 上重新选择，selected=' + str(wsel.get('selected_model'))) if wm_sel and wsel else '与全 split 口径同一选定 α/λ'}）。")
        elif wm and wspec:
            lines.append(
                f"- 窗口规格参考值（按 n-gram 侧声明规格 ctx={wspec.get('ctx')}、"
                f"iters={wspec.get('iters')}、n_eval={wspec.get('n_eval')}、"
                f"共 {_int_or_zero(wspec.get('n_tokens')):,} 个评分 token）："
                f"一元 PPL={_fmt_scalar_maybe(wm.get('unigram'))}、"
                f"二元 PPL={_fmt_scalar_maybe(wm.get('bigram'))}、"
                f"插值二元 PPL={_fmt_scalar_maybe(wm.get('interpolated_bigram'))}"
                f"（{('α/λ 在同一批窗口 token 上重新选择，selected=' + str(wsel.get('selected_model'))) if wm_sel and wsel else '与全 split 口径同一选定 α/λ'}）。⚠ 本次神经评测未全部满足该"
                "窗口规格，实际比较已回退全 split 参考（见下方比较表），"
                "这些数字不代表神经评测实际使用的 token 集合。")
        elif ngram.get("window_spec_error"):
            lines.append(f"- ⚠ n-gram 窗口对齐不可用："
                         f"{ngram['window_spec_error']}")
    else:
        lines.append("- 简单统计基线：未运行（`python -m lega.ngram`，"
                     "结果写入 `results/<tag>/ngram_baseline.json`）。")
    # 逐变体 vs 简单基线的同 split 配对对比：只并排报两个数不构成
    # “复杂模型优于简单基线”的证据，必须给出比值、校正后 p 与判定。
    cmp = ngram_cmp or {}
    if cmp.get("rows"):
        ref = cmp.get("reference_model")
        matched = cmp.get("reference_source") == "window_matched"
        basis = ("参考值与神经评测逐位置对齐（同一批窗口目标 token）"
                 if matched else
                 "参考值为全 split PPL（未逐位置对齐，见下方说明）")
        lines += [
            "",
            f"- 神经变体 vs 简单统计基线（{ref}，split={cmp.get('split')}；"
            f"{basis}；n-gram 无种子方差 → 逐种子 PPL 与同一常数配对做单样本"
            f"检验，Holm 校正（家族 "
            f"{cmp.get('holm_family_size', len(cmp['rows']))} 个变体）；"
            "属补充性/探索性对比）：",
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
                  "比无梯度基线好；两边必须同 split 且尽量同 token——神经 "
                  "PPL 只在 eval_iters 个窗口的目标 token 上计算，若本表未用"
                  "窗口对齐参考（reference_source=full_split），说明存在"
                  "与窗口规格不一致的 run，比值含 token 集合差异带来的偏移。"
                  "本对比未预注册，只是补充证据，不替代变体间受控配对检验。"]
        if not matched and cmp.get("fallback_reason"):
            lines.append(f"  ⚠ 未使用逐位置对齐参考：{cmp['fallback_reason']}"
                         "（重跑 n-gram 基线或对齐 eval 规格后即可对齐）。")
        win_rows = [r for r in cmp["rows"]
                    if r.get("window_paired_mean_diff") is not None]
        if win_rows:
            lines += [
                "",
                "  **逐窗口配对（探索性；比“与常数做单样本检验”多利用窗口"
                "抽样信息）**",
                "",
                "| 变体 | 种子 | 窗口数 | ΔNLL(神经−n-gram) | 95% CI | "
                "块bootstrap 95% CI | PPL比 | p_adj(Holm) | 判定 |",
                "|---|---|---|---|---|---|---|---|---|",
            ]
            for r in win_rows:
                ci = r.get("window_paired_ci95") or [None, None]
                ci_s = (f"[{ci[0]:+.4f}, {ci[1]:+.4f}]"
                        if ci[0] is not None and ci[1] is not None else "-")
                bci = r.get("window_paired_block_ci95") or [None, None]
                bci_s = (f"[{bci[0]:+.4f}, {bci[1]:+.4f}]"
                         if bci[0] is not None and bci[1] is not None else "-")
                ratio = (f"{r['window_paired_ppl_ratio']:.4f}"
                         if r.get("window_paired_ppl_ratio") is not None else "-")
                padj = (_fmt4(r.get("window_paired_p_adj"))
                        if r.get("window_paired_p_adj") is not None else "-")
                lines.append(
                    f"| {r.get('variant')} | {r.get('window_paired_n_seeds')} | "
                    f"{r.get('window_paired_n_windows_min')}–"
                    f"{r.get('window_paired_n_windows_max')} | "
                    f"{_fmt4(r.get('window_paired_mean_diff'))} | {ci_s} | "
                    f"{bci_s} | {ratio} | {padj} | "
                    f"{r.get('window_paired_verdict')} |")
            lines += ["", "  注: ΔNLL<0 表示神经模型在相同窗口上更好；分层 "
                      "bootstrap 同时重采样训练种子与评测窗口，窗口独立"
                      "重采样忽略空间相关；块bootstrap 用移动块（块长 "
                      "n^(1/3)）保留相邻窗口相关，两者并报，以更宽区间"
                      "解读稳健性，故只作探索性证据。该检验要求"
                      "神经与 n-gram 使用同一批窗口序号；不满足时主表仍"
                      "保留，但逐窗口行不生成。"]
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
        if not isinstance(tune, dict):
            tune = None
    if tune:
        best = tune.get("selected")
        best = best if isinstance(best, dict) else {}
        dflt = tune.get("default_trial")
        dflt = dflt if isinstance(dflt, dict) else {}
        gap = tune.get("selected_minus_default_val_loss")
        sel_seeds = tune.get("selection_seeds") or [tune.get("seed")]
        sel_seeds = sel_seeds if isinstance(sel_seeds, list) else []
        grid = tune.get("grid") or []
        grid = [c for c in grid if isinstance(c, dict)] if isinstance(grid, list) else []
        def _n_distinct(key):
            vals = set()
            for c in grid:
                v = c.get(key)
                try:
                    hash(v)
                except TypeError:
                    continue
                vals.add(v)
            return len(vals) if grid else "-"

        n_lr, n_wd = _n_distinct("lr"), _n_distinct("weight_decay")
        trials = tune.get("trials")
        n_trials = tune.get("n_trials")
        if n_trials is None:
            n_trials = (len(grid) or (len(trials)
                                      if isinstance(trials, list) else "-"))
        std = best.get("val_loss_best_std")
        lines.append(
            f"- 基线调优（变体 `{tune.get('variant', 'standard')}`，"
            f"网格 {n_lr} 档 lr × {n_wd} 档 wd = "
            f"{n_trials} "
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
        diag = tune.get("selection_diagnostics")
        if isinstance(diag, dict) and _int_or_zero(diag.get("n_seeds")) >= 2:
            n_sel = _int_or_zero(diag.get("n_seeds"))
            support = _int_or_zero(diag.get("selected_support_count"))
            frac = _num_or_none(diag.get("selected_support_fraction"))
            if frac is not None:
                lines.append(
                    f"- 选择稳定性：最优配置在 {support}/{n_sel} 个选择种子上"
                    f"实际取得最低 val_loss（支持率 {frac:.0%}）；逐种子赢家与"
                    "完整配对值见 `baseline_tuning.json` 的 "
                    "`selection_diagnostics`。")
                if support < n_sel:
                    lines.append(
                        "- ⚠ 均值最优并非逐种子稳定最优：至少一个种子上另一"
                        "配置更好，说明排序受种子噪声影响；差距接近噪声时"
                        "不能把该配置当作“已验证的最佳超参数”。")
            ru = diag.get("runner_up")
            if isinstance(ru, dict):
                margin = _num_or_none(ru.get("selected_minus_runner_up_mean"))
                dstd = _num_or_none(ru.get("paired_delta_std"))
                dmean = _num_or_none(ru.get("paired_delta_mean"))
                noise = ru.get("within_seed_noise")
                lines.append(
                    "- 最优−次优：次优 lr="
                    f"{ru.get('lr')}"
                    + (f"、weight_decay={ru.get('weight_decay')}"
                       if ru.get("weight_decay") is not None else "")
                    + f"，均值差 {_fmt4(margin)}；逐种子（次优−最优）配对差 "
                    f"mean={_fmt4(dmean)}、SD={_fmt4(dstd)}"
                    + ("——差距在一个种子噪声 SD 内（不稳定）"
                       if noise is True else
                       "——差距大于一个种子噪声 SD"
                       if noise is False else ""))
                if noise is True:
                    lines.append(
                        "- ⚠ 最优与次优的差距未超过逐种子配对波动：更换种子"
                        "即可能翻转选择；此时应优先看两者在正式种子上的"
                        "下游指标，而不是把“网格最优”当成真实优势。")
            loo = diag.get("leave_one_seed_out")
            if isinstance(loo, dict):
                n_folds = _int_or_zero(loo.get("n_folds"))
                agree = _int_or_zero(loo.get("agreement_count"))
                gain = _num_or_none(loo.get("heldout_gain_vs_default_mean"))
                gstd = _num_or_none(loo.get("heldout_gain_vs_default_std"))
                w, t, l = (loo.get("wins_vs_default"),
                           loo.get("ties_vs_default"),
                           loo.get("losses_vs_default"))
                agree_txt = ""
                if n_folds:
                    try:
                        agree_txt = f"（{agree / n_folds:.0%}）"
                    except (OverflowError, ZeroDivisionError):
                        # 外部 JSON 的合法大整数可精确解析，但转 float 做
                        # 比例会溢出；比例按缺失渲染，不能让汇总崩溃。
                        agree_txt = ""
                lines.append(
                    f"- 留一种子复选：{n_folds} 折中 {agree} 折重新选出的配置"
                    "与全量选择一致"
                    + agree_txt
                    + "；在被留出的种子上，LOO 选择相对默认配置的 held-out "
                    f"val_loss 平均增益 {_fmt4(gain)}"
                    + (f" ± {_fmt4(gstd)}" if gstd is not None else "")
                    + (f"（胜/平/负 {w}/{t}/{l}；正=优于默认）"
                       if w is not None else "")
                    + "。LOO 折不独立，仅作稳定性描述。")
                if (n_folds and agree < n_folds
                        and gain is not None and gain <= 0):
                    lines.append(
                        "- ⚠ 留一复选既翻转选择、又未在留出种子上优于默认："
                        "网格调优的泛化收益缺乏稳定性证据，正式结论应以"
                        "新种子复验为准。")
        elif len(sel_seeds) >= 2:
            lines.append(
                "- ⚠ 调优报告缺少 `selection_diagnostics`（旧格式）：无法核对"
                "逐种子支持率、最优−次优噪声差距与留一种子复选；请重跑 "
                "`--tune_baseline` 后再据此宣称基线已稳定调优。")
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
        # 调优只覆盖 tune.variant（默认 standard）；qgate/null 是“已有工作
        # 代表”，若也直接用同一配置，就不能把对比写成“打败了各自最佳调优
        # 的已有方法”。这是设计上的受控对照，但结论范围必须写清楚。
        tuned_variant = tune.get("variant", "standard")
        prior = [v for v in ("qgate", "null")
                 if isinstance(variants, (list, tuple, set)) and v in variants]
        untuned = [v for v in prior if v != tuned_variant]
        if untuned:
            lines.append(
                f"- ⚠ 调优范围：`baseline_tuning.json` 只调优了 "
                f"`{tuned_variant}`；已有工作代表 {', '.join('`'+v+'`' for v in untuned)} "
                "直接继承同一组 lr/weight_decay，未各自做网格调优。因此"
                "“优于已有工作”只能理解为**同超参受控对照**，不是“优于"
                "各自最佳调优版本”；若差距接近 MDE/种子噪声，不能据此宣称"
                "方法优越。需要强结论时应分别调优各基线（并同时调优本方"
                "方法）后在新种子上复验。")
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
            "AUROC门控(logit) | F1探针(macro,0.5) | F1探针(macro,调优) | "
            "ΔPR-AUC(探针−门控) | p_adj(PR-AUC) | ΔAUROC(探针−门控) | "
            "p_adj(AUROC) | 判定 |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]

        def _delta_ci(r, key):
            delta = r.get(key)
            ci = r.get(key + "_ci95")
            if delta is None:
                return "-"
            text = f"{delta:+.3f}"
            if (isinstance(ci, (list, tuple)) and len(ci) == 2
                    and all(_finite_number(x) is not None for x in ci)):
                text += f" [{_finite_number(ci[0]):+.3f}, "
                text += f"{_finite_number(ci[1]):+.3f}]"
            return text

        for r in probe["rows"]:
            feat = r.get("features")
            feat_s = "+".join(str(x) for x in feat) if feat else "-"
            lines.append(
                f"| {r.get('variant')} | {r.get('n_seeds')} | "
                f"{_fmt4(r.get('hit_base_rate'))} | {feat_s} | "
                f"{_fmt4(r.get('rule_auprc'))} | "
                f"{_fmt4(r.get('probe_auprc'))} | "
                f"{_fmt4(r.get('gate_auprc'))} | "
                f"{_fmt4(r.get('probe_auroc'))} | "
                f"{_fmt4(r.get('gate_auroc'))} | "
                f"{_fmt4(r.get('probe_f1_fixed'))} | "
                f"{_fmt4(r.get('probe_f1_tuned'))} | "
                f"{_delta_ci(r, 'delta_auprc')} | "
                f"{_fmt4(r.get('p_adj_auprc'))} | "
                f"{_delta_ci(r, 'delta_auroc')} | "
                f"{_fmt4(r.get('p_adj_auroc'))} | "
                f"{r.get('verdict') or '-'} |")
        lines += ["", f"  {probe.get('note', '')}"]
        lines.append(
            f"  注：命中基线率=hit 正类率，也是 PR-AUC 的随机猜基线；"
            f"主指标={next((r.get('primary_detection_metric') for r in probe['rows'] if r.get('primary_detection_metric')), 'n/a')}"
            f"（正类率 <5% 时 PR-AUC 为主，AUROC 因负类基数大而乐观）；"
            f"探针 λ 用嵌套分组 CV、在内层 OOF 上按主指标（PR-AUC 或 AUROC）"
            f"选择，阈值在内层 OOF 上按 macro-F1 选择（见 analysis_*.json 的 "
            f"linear_probe.per_fold / per_l2）；固定 0.5 阈值 F1 仅作与门控 "
            f"g>0.5 的同口径对照，调优阈值 F1 才代表简单基线被充分调参后的"
            f"决策能力。PR-AUC 探针/OOF 与门控在同一样本上计算，Δ 的 CI 与 "
            f"p_adj 见 significance.json。")
    if probe.get("missing"):
        names = "、".join(f"{x.get('variant')}（{x.get('reason')}）"
                          for x in probe["missing"])
        lines.append(f"- ⚠ 线性探针简单基线缺失：{names}；"
                     "机制主张缺少“有标签监督的最简线性模型”对照，"
                     "不能用“门控 logit 判别力高”替代。")
    lines.append("")
    return lines


def _fmt_scalar_maybe(x) -> str:
    f = _finite_number(x)
    if f is not None:
        return f"{f:.3f}"
    # 旧产物里可能有 "n/a" 这类说明性字符串，原样保留；bool/容器等
    # 非数值类型不能显示成 1.000/"[...]" 伪观测。
    return x if isinstance(x, str) else "-"


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
