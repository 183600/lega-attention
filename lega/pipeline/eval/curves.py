"""训练曲线诊断、效率指标与涨点-增幅权衡。"""
from .._shared import _install as _shared_install

_shared_install(globals())

def diagnose_training_curves(chain, data_dir: str, results_dir: str) -> dict:
    """对每个 (变体, 种子) run 做训练曲线诊断并落盘汇总。

    只看最终数值会掩盖训练动态：训练降/验证升=过拟合，两者都不降=欠拟合或
    数据管道问题，NaN=训练失效。诊断结果进入 summary，另存
    curve_diagnosis.json 供逐 run 复核。
    """
    log_vocab = None
    meta_path = os.path.join(data_dir, "meta.json")
    if os.path.exists(meta_path):
        try:
            import math as _math
            with open(meta_path, "r", encoding="utf-8") as f:
                vocab = _integral_value(json.load(f)["vocab_size"])
            if vocab is None or vocab <= 0:
                raise ValueError(f"meta.vocab_size 非法: {vocab!r}")
            log_vocab = _math.log(vocab)
        except (OSError, ValueError, KeyError, TypeError):
            log_vocab = None
    diag = {}
    for v, seed, vdir, _suffix in chain:
        d = diagnose_run_dir(vdir, log_vocab=log_vocab)
        diag.setdefault(v, {})[str(seed)] = d
    if diag:
        save_json(diag, os.path.join(results_dir, "curve_diagnosis.json"))
        n_bad = sum(1 for vs in diag.values() for d in vs.values()
                    if d["status"] not in ("正常收敛", "数据不足"))
        print(f"[curves] 训练曲线诊断完成：{n_bad} 个 run 需要关注"
              f" -> {os.path.join(results_dir, 'curve_diagnosis.json')}")
    return diag


def _num_or_none(v):
    """JSON 数值安全转换：缺失/bool/非数值/非有限值统一返回 None。"""
    return _finite_number(v)


def _eff_cell(entry, key, ctx):
    """latency.json 变体条目里的结构与运行时字段（flops 取 total，mem 取量测值）。"""
    if not isinstance(entry, dict):
        return None
    eff = entry.get("efficiency")
    eff = eff if isinstance(eff, dict) else {}
    if key in ("flops", "total_flops", "gate_flops"):
        by_ctx = eff.get("flops_per_tok")
        if not isinstance(by_ctx, dict):
            return None
        cell = by_ctx.get(str(ctx))
        if not isinstance(cell, dict):
            return None
        field = ("gate_extra_flops_per_tok" if key == "gate_flops"
                 else "total_flops_per_tok")
        return _num_or_none(cell.get(field))
    if key == "mem":
        by_ctx = eff.get("peak_mem_mb_prefill")
        return (_num_or_none(by_ctx.get(str(ctx)))
                if isinstance(by_ctx, dict) else None)
    if key == "params":
        return _num_or_none(eff.get("n_params") or entry.get("n_params"))
    if key == "params_mb":
        return _num_or_none(eff.get("params_mb"))
    if key == "kv_mem":
        # 优先取 latency.json 已按 ctx 落盘的值；旧格式只有 per-token 时
        # 按 ctx 线性换算（KV cache 大小随上下文长度线性增长）。
        by_ctx = eff.get("kv_cache_mb_at_ctx")
        if isinstance(by_ctx, dict):
            v = _num_or_none(by_ctx.get(str(ctx)))
            if v is not None:
                return v
        per_tok = _num_or_none(eff.get("kv_cache_mb_per_tok"))
        if per_tok is not None:
            # ctx 可能来自外部 JSON，是合法但超出 float 上限的大整数；换算
            # 结果也可能溢出成 inf。两者都不能进入汇总/判定，按缺失处理。
            try:
                out = per_tok * float(ctx)
            except (TypeError, ValueError, OverflowError):
                return None
            return out if math.isfinite(out) else None
    return None


def _rel_pct(base, cand):
    """cand 相对 base 的百分比变化；任一缺失/base=0/结果不可表示时返回 None。

    两端都是有限值不代表百分比可表示：base 极小或 cand−base 溢出时会得到
    ±inf，写入汇总行后 JSON 只能把它变成 null，而判定逻辑仍可能按 inf 参与
    max()/比较。这里统一在入口按缺失处理。
    """
    base, cand = _num_or_none(base), _num_or_none(cand)
    if base is None or cand is None or base == 0:
        return None
    try:
        out = (cand - base) / abs(base) * 100.0
    except (OverflowError, ZeroDivisionError):
        return None
    return out if math.isfinite(out) else None


def _lat_noise_pct(entry, metric, ctx, bs="1"):
    """latency.json 中某运行时指标的测量相对散布（spread_pct）；缺失返回 None。

    spread_pct=(p90-p10)/median，由 lega.latency 逐次计时给出。旧格式
    latency.json 没有该字段时返回 None（按无噪声信息处理）。
    """
    if not isinstance(entry, dict):
        return None
    key = ("prefill_ms_spread_pct" if metric == "prefill"
           else "decode_ms_spread_pct")
    by_ctx = entry.get(key)
    if not isinstance(by_ctx, dict):
        return None
    cell = by_ctx.get(str(ctx))
    if metric == "prefill":
        cell = cell.get(str(bs)) if isinstance(cell, dict) else None
    return _num_or_none(cell)


def _noise_floor_pct(base_noise, cand_noise):
    """两端的测量散布按方和根合成（比值差的方差近似为相对方差之和）。

    只有两端都有散布信息时才给出噪声下限：一侧缺失时无法判断"差值是否
    小于测量噪声"，按未过滤的原口径处理，避免用单侧散布低估/高估噪声。
    外部 JSON 里的 bool/字符串/NaN/Inf 先统一按缺失处理，不能直接参与
    ``< 0`` 比较而抛 TypeError。
    """
    base_noise = _num_or_none(base_noise)
    cand_noise = _num_or_none(cand_noise)
    if (base_noise is None or cand_noise is None
            or base_noise < 0.0 or cand_noise < 0.0):
        return None
    try:
        floor = math.sqrt(base_noise * base_noise + cand_noise * cand_noise)
    except (OverflowError, ValueError):
        return None
    return floor if math.isfinite(floor) and floor >= 0.0 else None


# RULER 收益以 EM 绝对百分点计，与 PPL 的相对百分比单位不同：收支比规则
# 不能混算，RULER 单独设阈值（EM 提升通常比 PPL 百分比更难获得，倍数放宽
# 到 10×；下限 0.5 pp 避免收益趋近 0 时比值判定失效）。


def tradeoff_verdict(benefit_pct, p_adj, cost_pct, alpha=ALPHA,
                     n_seeds=None, ruler_benefit_pp=None, ruler_p_adj=None,
                     ruler_n_seeds=None) -> str:
    """涨点-增幅权衡判定：不看增幅的涨点没有意义；收益必须覆盖两个主指标。

    两个主指标：PPL（benefit_pct=相对基线的百分比改善，正=更好）与 RULER
    平均 EM（ruler_benefit_pp=相对基线的绝对百分点改善，正=更好）。本项目
    预注册主张横跨两者（H1 PPL、H4 长上下文检索），只看 PPL 会把“PPL 持平
    但 RULER 显著提升”的真涨点误判为无效，也会漏掉“PPL 涨、RULER 显著
    退步”的严重信号。

    规则（显式、可审计）：
      * 两个主指标都无数据或增幅缺失 → 数据不足；
      * 参与判定的指标种子不足（给定 n<MIN_SEEDS）→ 无法判定，与全表口径一致；
      * 某一主指标显著劣化：若另一主指标同时显著改善 → 权衡（必须显式
        说明取舍）；否则 → 不可行；
      * 任一主指标显著改善 → 有涨点：综合增幅（各指标相对基线的最大增幅）
        >50% → 需审视；按显著者中最不利的收益套用收支比规则（PPL 5×，
        RULER 单独 10× 且下限 0.5 pp，单位不同不混算）→ 收益-增幅比不利；
        否则 → 可接受，并注明依据哪个指标；
      * 有正向涨点但不显著 → 说服力有限（增幅>10% 判不可行）。
    微小涨点伴随巨大增幅的改进在这里会被明确拒绝。
    """
    # 外部/旧汇总可能把数值写成 bool/字符串/NaN；先在入口统一成有限数值或
    # None，避免 False 被 `False < alpha` 当成显著 p=0，也避免字符串比较
    # 直接抛 TypeError。
    benefit_pct = _num_or_none(benefit_pct)
    cost_pct = _num_or_none(cost_pct)
    ruler_benefit_pp = _num_or_none(ruler_benefit_pp)
    p_adj = _num_or_none(p_adj)
    ruler_p_adj = _num_or_none(ruler_p_adj)
    if (benefit_pct is None and ruler_benefit_pp is None) or cost_pct is None:
        return "数据不足"
    raw_n, raw_ruler_n = n_seeds, ruler_n_seeds
    n_seeds = _integral_value(n_seeds)
    ruler_n_seeds = _integral_value(ruler_n_seeds)
    bad_n = []
    if n_seeds is None and raw_n is not None:
        bad_n.append(f"PPL n={raw_n!r}")
    if ruler_n_seeds is None and raw_ruler_n is not None:
        bad_n.append(f"RULER n={raw_ruler_n!r}")
    if bad_n:
        return f"种子数非法（{'、'.join(bad_n)}），无法判定"
    lacks = []
    if benefit_pct is not None and n_seeds is not None and n_seeds < MIN_SEEDS:
        lacks.append(f"PPL n={n_seeds}")
    if (ruler_benefit_pp is not None and ruler_n_seeds is not None
            and ruler_n_seeds < MIN_SEEDS):
        lacks.append(f"RULER n={ruler_n_seeds}")
    if lacks:
        return f"种子不足（{'、'.join(lacks)}<{MIN_SEEDS}），无法判定"
    ppl_sig = p_adj is not None and p_adj < alpha
    rul_sig = ruler_p_adj is not None and ruler_p_adj < alpha
    ppl_up = benefit_pct is not None and benefit_pct > 0 and ppl_sig
    rul_up = (ruler_benefit_pp is not None and ruler_benefit_pp > 0
              and rul_sig)
    ppl_down = benefit_pct is not None and benefit_pct < 0 and ppl_sig
    rul_down = (ruler_benefit_pp is not None and ruler_benefit_pp < 0
                and rul_sig)
    if ppl_down and rul_up:
        return (f"权衡：PPL 显著退步 {benefit_pct:.1f}% 但 RULER 显著提升 "
                f"+{ruler_benefit_pp:.2f} pp（增幅 {cost_pct:.0f}%，"
                "检索目标改善，须说明 PPL 取舍）")
    if rul_down and ppl_up:
        return (f"权衡：PPL 显著改善 {benefit_pct:.1f}% 但 RULER 显著退步 "
                f"{ruler_benefit_pp:.2f} pp（增幅 {cost_pct:.0f}%，"
                "核心检索目标受损，须说明取舍）")
    if ppl_down:
        extra = (f"，且 RULER 显著退步 {ruler_benefit_pp:+.2f} pp"
                 if rul_down else "")
        return (f"不可行：PPL 显著差于基线（{benefit_pct:.1f}%）"
                f"{extra}")
    if rul_down:
        return f"不可行：RULER 显著差于基线（{ruler_benefit_pp:+.2f} pp）"
    if ppl_up or rul_up:
        if cost_pct > 50:
            return f"需审视：显著但增幅 {cost_pct:.0f}%"
        if ppl_up and cost_pct > 5 * max(benefit_pct, 0.5):
            return (f"收益-增幅比不利：PPL 收益 {benefit_pct:.1f}% vs "
                    f"增幅 {cost_pct:.0f}%")
        if rul_up and cost_pct > RULER_COST_RATIO * max(ruler_benefit_pp,
                                                        RULER_RATIO_FLOOR_PP):
            return (f"收益-增幅比不利：RULER 收益 +{ruler_benefit_pp:.2f} pp vs "
                    f"增幅 {cost_pct:.0f}%")
        basis = "PPL" if ppl_up else "RULER"
        if ppl_up and rul_up:
            basis = "PPL+RULER"
        return "可接受" if basis == "PPL" else f"可接受（依据 {basis} 显著涨点）"
    any_pos = ((benefit_pct is not None and benefit_pct > 0)
               or (ruler_benefit_pp is not None and ruler_benefit_pp > 0))
    if any_pos:
        if cost_pct > 10:
            return f"不可行：涨点不显著且增幅 {cost_pct:.0f}%"
        return "说服力有限：涨点未通过显著性检验"
    if cost_pct > 10:
        return f"不可行：无涨点且增幅 {cost_pct:.0f}%"
    return "说服力有限：无涨点且差异不显著"
