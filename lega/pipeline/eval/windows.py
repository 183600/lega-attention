"""PPL 窗口级配对与 bootstrap。"""
from .._shared import _install as _shared_install

_shared_install(globals())

def _eval_window_losses(d):
    """eval payload → {window_index: loss}；缺失/损坏/重复序号返回 None。

    只接受逐窗口损失与窗口序号一一对应且序号唯一的记录。旧产物没有这些
    字段时返回 None，下游跳过窗口级分析，不影响既有 PPL 配对检验。
    交叉熵损失不可能为负：负值来自坏记录/错误归一化，若参与逐窗口配对会
    污染 avg NLL 差、bootstrap 区间与 PPL 比，必须整表按不可用处理。
    """
    if not isinstance(d, dict):
        return None
    losses = d.get("window_losses")
    indices = d.get("window_indices")
    if (not isinstance(losses, list) or not isinstance(indices, list)
            or not losses or len(losses) != len(indices)):
        return None
    out = {}
    for loss, idx in zip(losses, indices):
        v = _num_or_none(loss)
        i = _integral_value(idx)
        # 窗口序号是 s·ctx 的非负下标；负数/重复都不是同一批评测单元。
        if v is None or v < 0.0 or i is None or i < 0 or i in out:
            return None
        out[i] = v
    return out


def _trimmed_mean(values, fraction: float = 0.10):
    """按比例双侧截尾的均值；样本过少时不截尾（退化为普通均值）。"""
    arr = np.asarray(values, dtype=np.float64).ravel()
    if arr.size == 0:
        return None
    k = int(math.floor(float(arr.size) * float(fraction)))
    if k <= 0:
        return float(arr.mean())
    if 2 * k >= arr.size:
        return float(np.median(arr))
    arr = np.sort(arr)
    return float(arr[k:arr.size - k].mean())


def _window_bootstrap_stats(seed_diffs, n_resamples: int = 2000,
                            seed: int = 0) -> dict:
    """逐窗口配对差的分层 bootstrap（先重采样种子、再重采样窗口）。

    输入 seed_diffs 是每个种子的窗口差数组；返回平均差、百分位区间、
    中心化零分布 p 值（P(|mean*−obs| ≥ |obs|)）与同号重采样比例。
    固定 seed 保证可复现。

    除均值外，同时报告对少数极端窗口不敏感的中位数、10% 截尾均值、
    改善/恶化/持平窗口占比与 ΔNLL 分位数，并给它们同样的分层 bootstrap
    区间。平均 NLL 改善可能只来自极少数困难窗口；只看均值会把“少数窗口
    的大幅改善”误读成“普遍改善”。这些稳健统计不构成新的确认性检验，
    只用于判断均值结论是否被少数评测单元主导。

    同时给出 i.i.d. 窗口重采样与移动块（moving-block）重采样：窗口按时间
    顺序排列时，相邻窗口可能受同一文本片段/上下文漂移影响，把它们当成
    独立观测会低估区间。块长取各序列 ``round(n^(1/3))``（n<4 时退化为 1，
    即与 i.i.d. 相同），块起点在循环序列上均匀抽取；同时报告 lag-1
    自相关均值，便于判断两种区间为何不同。块 bootstrap 仍是近似，只作
    探索性稳健性证据。
    """
    # 重采样次数/种子是循环预算与随机源身份：0/负数会让分位数计算在空数组
    # 上崩溃，bool/小数经 int() 截断后会与调用方声明的次数不一致。
    n_resamples = strict_int(n_resamples, "n_resamples")
    if n_resamples < 1:
        raise ValueError(f"n_resamples 必须 >= 1，收到 {n_resamples!r}")
    seed = strict_int(seed, "seed")
    if seed < 0:
        raise ValueError(f"seed 必须 >= 0，收到 {seed!r}")
    arrays = []
    for d in (seed_diffs or []):
        arr = np.asarray(d, dtype=np.float64).ravel()
        # 窗口差应由调用方保证有限；这里仍独立过滤，避免直接调用方把
        # NaN/Inf 送进均值后让整份探索性证据变成 NaN。
        arr = arr[np.isfinite(arr)]
        if arr.size:
            arrays.append(arr)
    if not arrays:
        return None
    n_seeds = len(arrays)
    # 先按全局最大绝对值缩放再重采样：合法但接近 float 上限的窗口差
    # （如 ±1e308）在朴素求和/平方时会溢出成 inf，使均值、区间和 p 值
    # 全部失真。缩放后各值落在 [-1,1]，统计量按尺度等变还原。
    scale = float(max(np.max(np.abs(d)) for d in arrays))
    if scale > 0.0:
        arrays_scaled = [d / scale for d in arrays]
    else:
        arrays_scaled = [np.zeros_like(d) for d in arrays]
    obs_scaled = float(np.mean([d.mean() for d in arrays_scaled]))
    obs = float(obs_scaled * scale)
    if not math.isfinite(obs):
        return None

    quantile_levels = (0.10, 0.25, 0.50, 0.75, 0.90)
    obs_median_scaled = float(np.mean(
        [float(np.median(d)) for d in arrays_scaled]))
    obs_trimmed_scaled = float(np.mean(
        [_trimmed_mean(d, 0.10) for d in arrays_scaled]))
    obs_win_rate = float(np.mean(
        [float(np.mean(d < 0.0)) for d in arrays_scaled]))
    obs_loss_rate = float(np.mean(
        [float(np.mean(d > 0.0)) for d in arrays_scaled]))
    obs_tie_rate = float(np.mean(
        [float(np.mean(d == 0.0)) for d in arrays_scaled]))
    obs_quantiles_scaled = {
        f"p{int(round(q * 100))}": float(np.mean(
            [float(np.quantile(d, q)) for d in arrays_scaled]))
        for q in quantile_levels}

    rng = np.random.default_rng(seed)
    reps_scaled = np.empty(int(n_resamples), dtype=np.float64)
    reps_median_scaled = np.empty(int(n_resamples), dtype=np.float64)
    reps_trimmed_scaled = np.empty(int(n_resamples), dtype=np.float64)
    reps_win_rate = np.empty(int(n_resamples), dtype=np.float64)
    for r in range(int(n_resamples)):
        sidx = rng.integers(0, n_seeds, n_seeds)
        means, medians, trimmed, wins = [], [], [], []
        for si in sidx:
            d = arrays_scaled[int(si)]
            w = rng.integers(0, d.size, d.size)
            sample = d[w]
            means.append(float(sample.mean()))
            medians.append(float(np.median(sample)))
            trimmed.append(_trimmed_mean(sample, 0.10))
            wins.append(float(np.mean(sample < 0.0)))
        reps_scaled[r] = float(np.mean(means))
        reps_median_scaled[r] = float(np.mean(medians))
        reps_trimmed_scaled[r] = float(np.mean(trimmed))
        reps_win_rate[r] = float(np.mean(wins))
    reps = reps_scaled * scale
    if not np.all(np.isfinite(reps)):
        return None
    alpha = 0.025

    def _scaled_interval(reps_scaled_values):
        lo_s, hi_s = np.quantile(reps_scaled_values, [alpha, 1.0 - alpha])
        lo, hi = float(lo_s * scale), float(hi_s * scale)
        return [lo, hi] if math.isfinite(lo) and math.isfinite(hi) else None

    def _rate_interval(reps_values):
        lo, hi = np.quantile(reps_values, [alpha, 1.0 - alpha])
        lo, hi = float(lo), float(hi)
        return [lo, hi] if math.isfinite(lo) and math.isfinite(hi) else None

    lo, hi = _scaled_interval(reps_scaled)
    if lo is None:
        return None
    ge = int((np.abs(reps_scaled - obs_scaled) >= abs(obs_scaled)).sum())
    p_boot = (((ge + 1) / (int(n_resamples) + 1))
              if obs_scaled != 0 else 1.0)
    same_sign = (float(np.mean((reps_scaled > 0) == (obs_scaled > 0)))
                 if obs_scaled != 0 else 0.0)

    # 移动块 bootstrap：块内保留窗口顺序/相邻相关，块间独立。块长用
    # n^(1/3) 的经验规则；n<4 时块长 1（等价于 i.i.d. 重采样）。
    def _block_len(n):
        return max(1, int(round(int(n) ** (1.0 / 3.0)))) if n >= 4 else 1
    block_lengths = [_block_len(d.size) for d in arrays_scaled]
    block_reps_scaled = np.empty(int(n_resamples), dtype=np.float64)
    block_reps_median_scaled = np.empty(int(n_resamples), dtype=np.float64)
    block_reps_trimmed_scaled = np.empty(int(n_resamples), dtype=np.float64)
    block_reps_win_rate = np.empty(int(n_resamples), dtype=np.float64)
    for r in range(int(n_resamples)):
        sidx = rng.integers(0, n_seeds, n_seeds)
        means, medians, trimmed, wins = [], [], [], []
        for si in sidx:
            d = arrays_scaled[int(si)]
            bl = block_lengths[int(si)]
            if bl <= 1:
                idx = rng.integers(0, d.size, d.size)
            else:
                n_blocks = (d.size + bl - 1) // bl
                starts = rng.integers(0, d.size, n_blocks)
                idx = ((starts[:, None] + np.arange(bl)[None, :]) % d.size)
                idx = idx.reshape(-1)[:d.size]
            sample = d[idx]
            means.append(float(sample.mean()))
            medians.append(float(np.median(sample)))
            trimmed.append(_trimmed_mean(sample, 0.10))
            wins.append(float(np.mean(sample < 0.0)))
        block_reps_scaled[r] = float(np.mean(means))
        block_reps_median_scaled[r] = float(np.mean(medians))
        block_reps_trimmed_scaled[r] = float(np.mean(trimmed))
        block_reps_win_rate[r] = float(np.mean(wins))
    block_reps = block_reps_scaled * scale
    if not np.all(np.isfinite(block_reps)):
        return None
    b_lo_s, b_hi_s = np.quantile(block_reps_scaled, [alpha, 1.0 - alpha])
    b_lo, b_hi = float(b_lo_s * scale), float(b_hi_s * scale)
    if not (math.isfinite(b_lo) and math.isfinite(b_hi)):
        return None
    b_ge = int((np.abs(block_reps_scaled - obs_scaled) >= abs(obs_scaled)).sum())
    p_block = (((b_ge + 1) / (int(n_resamples) + 1))
               if obs_scaled != 0 else 1.0)
    b_same_sign = (float(np.mean((block_reps_scaled > 0) == (obs_scaled > 0)))
                   if obs_scaled != 0 else 0.0)
    lag1 = []
    for d in arrays:
        if d.size >= 3:
            x, y = d[:-1], d[1:]
            # 相关系数对尺度不变；先按本条序列的最大绝对值缩放，避免
            # 1e308 量级的窗口差在 np.std/corrcoef 内平方求和溢出成 inf/inf
            # （只会产生 NaN 与运行时警告，lag-1 信息也被静默丢弃）。
            scale_d = float(np.max(np.abs(d)))
            if scale_d > 0.0:
                x, y = x / scale_d, y / scale_d
            if np.std(x) > 0 and np.std(y) > 0:
                v = float(np.corrcoef(x, y)[0, 1])
                if math.isfinite(v):
                    lag1.append(v)
    return {
        "mean_diff": obs, "mean_diff_ci95": [lo, hi],
        "p": p_boot, "bootstrap_same_sign": same_sign,
        "median_diff": float(obs_median_scaled * scale),
        "median_diff_ci95": _scaled_interval(reps_median_scaled),
        "block_median_diff_ci95": _scaled_interval(block_reps_median_scaled),
        "trimmed_mean_diff_10pct": float(obs_trimmed_scaled * scale),
        "trimmed_mean_diff_10pct_ci95": _scaled_interval(reps_trimmed_scaled),
        "block_trimmed_mean_diff_10pct_ci95": _scaled_interval(
            block_reps_trimmed_scaled),
        "window_win_rate": obs_win_rate,
        "window_loss_rate": obs_loss_rate,
        "window_tie_rate": obs_tie_rate,
        "window_win_rate_ci95": _rate_interval(reps_win_rate),
        "block_window_win_rate_ci95": _rate_interval(block_reps_win_rate),
        "diff_quantiles": {
            key: float(value * scale)
            for key, value in obs_quantiles_scaled.items()},
        "block_mean_diff_ci95": [b_lo, b_hi],
        "block_p": p_block, "block_bootstrap_same_sign": b_same_sign,
        "block_length_min": int(min(block_lengths)),
        "block_length_max": int(max(block_lengths)),
        "mean_lag1_autocorr": (float(np.mean(lag1)) if lag1 else None),
        "n_resamples": int(n_resamples),
        "n_seeds": n_seeds,
        "n_windows_min": int(min(d.size for d in arrays)),
        "n_windows_max": int(max(d.size for d in arrays))}


def _ppl_window_paired_tests(runs, pairs, n_resamples: int = 2000,
                             seed: int = 0) -> list:
    """逐窗口配对 + 种子/窗口分层 bootstrap（探索性评估不确定性证据）。

    变体的跨种子 t 检验把“每个种子的 PPL”当作一次观测，只反映训练随机性；
    而 PPL 是在 eval_iters 个等间隔窗口上估的，有限窗口抽样本身也有误差。
    同一 split/同评测协议下各变体共享同一批窗口序号，因此可逐窗口配对：
    d_w = loss_candidate(w) − loss_reference(w)。

    这里对 (种子, 窗口) 做分层 bootstrap：先重采样训练种子，再在每个被选中
    的种子里重采样窗口，得到“换种子 + 换一批评测窗口”下平均 NLL 差的
    不确定性区间。窗口间存在空间相关，按窗口独立重采样的区间是近似值；
    因此只作探索性稳健性证据，不替代确认性种子级检验。
    """
    out = []
    runs = runs if isinstance(runs, dict) else {}
    for cand, ref in (pairs if isinstance(pairs, list) else []):
        if cand not in runs or ref not in runs:
            continue
        cm = _variant_seed_map(runs, cand)
        rm = _variant_seed_map(runs, ref)
        seeds = [s for s in _sorted_seeds(cm) if s in rm]
        seed_diffs = []
        per_seed = []
        skipped = 0
        window_counts = set()
        window_keys_ref = None
        window_keys_aligned = True
        for s in seeds:
            sa = cm.get(s) if isinstance(cm.get(s), dict) else {}
            sb = rm.get(s) if isinstance(rm.get(s), dict) else {}
            ea = sa.get("eval") if isinstance(sa.get("eval"), dict) else None
            eb = sb.get("eval") if isinstance(sb.get("eval"), dict) else None
            if ea is None or eb is None:
                skipped += 1
                continue
            if ea.get("split", "val") != eb.get("split", "val"):
                skipped += 1
                continue
            pa, pb = _ppl_protocol(ea), _ppl_protocol(eb)
            if not _same_protocol_fields(pa, pb):
                skipped += 1
                continue
            wa = _eval_window_losses(ea)
            wb = _eval_window_losses(eb)
            if wa is None or wb is None or set(wa) != set(wb):
                skipped += 1
                continue
            keys = sorted(wa)
            if window_keys_ref is None:
                window_keys_ref = keys
            elif keys != window_keys_ref:
                window_keys_aligned = False
            diffs = np.asarray([wa[k] - wb[k] for k in keys], dtype=np.float64)
            if diffs.size == 0 or not np.all(np.isfinite(diffs)):
                skipped += 1
                continue
            seed_diffs.append(diffs)
            window_counts.add(int(diffs.size))
            # diffs 逐元素有限，但同号巨大值相加会溢出；与分层 bootstrap
            # 同口径用缩放后的稳定均值，避免 per_seed 明细出现 inf。
            seed_mean, _ = mean_std(diffs)
            per_seed.append({
                "seed": _integral_value(s),
                "n_windows": int(diffs.size),
                "mean_diff": seed_mean,
                "window_wins": int((diffs < 0).sum()),
                "window_losses": int((diffs > 0).sum()),
                "window_ties": int((diffs == 0).sum()),
            })
        if not seed_diffs:
            continue
        n_seeds = len(seed_diffs)
        stats = _window_bootstrap_stats(seed_diffs, n_resamples, seed)
        if stats is None:
            continue
        obs = stats["mean_diff"]
        variance = None
        if window_keys_aligned and n_seeds >= 2 and min(window_counts) >= 2:
            variance = two_way_variance_components(seed_diffs)
        enough = (n_seeds >= MIN_PPL_WINDOW_SEEDS
                  and min(window_counts) >= MIN_PPL_WINDOWS)

        def _sign(value):
            if value is None:
                return None
            return 1 if value > 0.0 else (-1 if value < 0.0 else 0)

        def _excludes_zero(ci):
            return bool(ci and ci[0] is not None and ci[1] is not None
                        and ((ci[0] > 0.0 and ci[1] > 0.0)
                             or (ci[0] < 0.0 and ci[1] < 0.0)))

        mean_sign = _sign(obs)
        median_sign = _sign(stats.get("median_diff"))
        trimmed_sign = _sign(stats.get("trimmed_mean_diff_10pct"))
        nonzero_signs = {s for s in (mean_sign, median_sign, trimmed_sign)
                         if s is not None and s != 0}
        direction_consistent = len(nonzero_signs) <= 1
        majority = None
        if obs is not None and obs != 0.0:
            majority = bool(
                (stats.get("window_win_rate", 0.0) > 0.5) if obs < 0.0
                else (stats.get("window_loss_rate", 0.0) > 0.5))
        mean_ci_sig = _excludes_zero(stats.get("mean_diff_ci95"))
        median_ci_sig = _excludes_zero(stats.get("median_diff_ci95"))
        outlier_sensitive = bool(
            enough and mean_ci_sig
            and (not direction_consistent
                 or majority is False
                 or not median_ci_sig))
        broad_based = bool(
            enough and mean_ci_sig and median_ci_sig
            and direction_consistent and majority is True)
        window_wins_total = int(sum((d < 0.0).sum() for d in seed_diffs))
        window_losses_total = int(sum((d > 0.0).sum() for d in seed_diffs))
        window_ties_total = int(sum((d == 0.0).sum() for d in seed_diffs))
        res = {
            "candidate": cand, "reference": ref,
            "metric": "PPL窗口配对", "lower_is_better": True,
            "window_level": True, "exploratory": True,
            "n": n_seeds, "n_seeds": n_seeds,
            "n_windows_min": min(window_counts),
            "n_windows_max": max(window_counts),
            "n_window_observations": int(sum(d.size for d in seed_diffs)),
            "variance_components": variance,
            "variance_components_aligned": bool(window_keys_aligned),
            "enough_windows": bool(enough),
            "min_window_seeds": MIN_PPL_WINDOW_SEEDS,
            "min_windows_per_seed": MIN_PPL_WINDOWS,
            "mean_diff": obs,
            "mean_diff_ci95": stats["mean_diff_ci95"],
            "median_diff": stats.get("median_diff"),
            "median_diff_ci95": stats.get("median_diff_ci95"),
            "block_median_diff_ci95": stats.get(
                "block_median_diff_ci95"),
            "trimmed_mean_diff_10pct": stats.get(
                "trimmed_mean_diff_10pct"),
            "trimmed_mean_diff_10pct_ci95": stats.get(
                "trimmed_mean_diff_10pct_ci95"),
            "block_trimmed_mean_diff_10pct_ci95": stats.get(
                "block_trimmed_mean_diff_10pct_ci95"),
            "window_win_rate": stats.get("window_win_rate"),
            "window_loss_rate": stats.get("window_loss_rate"),
            "window_tie_rate": stats.get("window_tie_rate"),
            "window_win_rate_ci95": stats.get("window_win_rate_ci95"),
            "block_window_win_rate_ci95": stats.get(
                "block_window_win_rate_ci95"),
            "window_wins_total": window_wins_total,
            "window_losses_total": window_losses_total,
            "window_ties_total": window_ties_total,
            "diff_quantiles": stats.get("diff_quantiles"),
            "robustness": {
                "mean_ci_excludes_zero": bool(mean_ci_sig),
                "median_ci_excludes_zero": bool(median_ci_sig),
                "mean_median_same_sign": (
                    None if mean_sign is None or median_sign is None
                    else bool(mean_sign == median_sign)),
                "mean_trimmed_same_sign": (
                    None if mean_sign is None or trimmed_sign is None
                    else bool(mean_sign == trimmed_sign)),
                "direction_consistent": bool(direction_consistent),
                "majority_windows_in_mean_direction": majority,
                "outlier_sensitive": outlier_sensitive,
                "broad_based": broad_based,
                "note": ("均值、中位数、10% 截尾均值与逐窗口胜负用于"
                         "区分“普遍改善”与“少数窗口驱动”。"
                         "outlier_sensitive=True 表示均值方向与稳健统计"
                         "不一致、或多数窗口未朝均值方向改善；"
                         "broad_based=True 表示均值/中位数/截尾均值同向、"
                         "中位数区间不含 0 且多数窗口同向。"
                         "两者均为描述性标记，不改变显著性判定。")},
            "block_mean_diff_ci95": stats.get("block_mean_diff_ci95"),
            "block_p": stats.get("block_p"),
            "block_bootstrap_same_sign": stats.get("block_bootstrap_same_sign"),
            "block_length_min": stats.get("block_length_min"),
            "block_length_max": stats.get("block_length_max"),
            "mean_lag1_autocorr": stats.get("mean_lag1_autocorr"),
            "ppl_ratio": float(math.exp(min(20.0, obs))),
            "p": stats["p"], "p_adj": None,
            "bootstrap_same_sign": stats["bootstrap_same_sign"],
            "n_resamples": int(n_resamples),
            "per_seed": per_seed,
            "verdict": None,
            "note": ("逐窗口 NLL 差（候选−参照），负值=候选更好；"
                     "分层 bootstrap 同时重采样种子与窗口；i.i.d. 口径忽略"
                     "窗口相关，block_* 为移动块重采样（块长 n^(1/3)）的"
                     "稳健性对照；variance_components 把方差分解为种子/"
                     "窗口/残差三部分。以上均为探索性证据"),
        }
        if skipped:
            res["excluded_runs"] = skipped
        out.append(res)
    pads = holm_bonferroni([t.get("p") for t in out])
    for t, padj in zip(out, pads):
        t["p_adj"] = padj
        ci = t.get("mean_diff_ci95")
        ci_s = (f"，95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}]" if ci else "")
        if t.get("mean_diff") is None or padj is None:
            t["verdict"] = "无法检验"
        elif not t.get("enough_windows"):
            t["verdict"] = (f"证据不足（种子 {t['n_seeds']}，每种子窗口 "
                            f"{t['n_windows_min']}–{t['n_windows_max']}；"
                            f"最低 {MIN_PPL_WINDOW_SEEDS} 种子×"
                            f"{MIN_PPL_WINDOWS} 窗口），仅报点估计{ci_s}")
        elif padj < ALPHA:
            t["verdict"] = ("窗口级显著优于（Holm 校正后）"
                            if t["mean_diff"] < 0 else
                            "窗口级显著劣于（Holm 校正后）")
        else:
            t["verdict"] = (f"窗口级无显著差异（Holm 校正后 p={padj:.4f}"
                            f"{ci_s}）")
        if (t.get("robustness") or {}).get("outlier_sensitive"):
            t["verdict"] += ("；⚠ 窗口级稳健性：均值改善与中位数/截尾均值"
                             "方向不一致或多数窗口未同向，可能由少数窗口驱动")
    return out
