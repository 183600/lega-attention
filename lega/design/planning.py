"""设计计划与多重比较规划。"""
from __future__ import annotations
from ._shared import _install as _shared_install

_shared_install(globals())

def build_design_plan(config: dict, seeds, variants, registered: dict = None,
                      *, config_file: str = None, config_sha256: str = None,
                      alpha: float = None, power: float = None,
                      ruler_n_samples: int = None, smoke: bool = False,
                      known_variants=None, analysis_variants=None,
                      multiplicity: str = None) -> dict:
    """根据配置/种子计划/预注册表生成训练前的设计计划。

    不读数据、不训练、不访问 test；只做输入解析与解析式功效计算，因此可以
    在正式训练之前（基线调优等准备步骤之后）生成并落盘。``registered`` 为登记表 dict（可空）；
    ``known_variants`` 用于把假设里的候选/参照映射到变体，缺省时惰性导入。

    返回 dict；``status`` 为 ok/warning/fail。``fail`` 不在此处中止实验，
    由调用方决定是否继续，但会进入汇总页与确认性降级证据。
    """
    config = _as_dict(config)
    if not isinstance(seeds, (list, tuple)) or not seeds:
        raise ValueError("seeds 必须是非空列表")
    seeds = [_seed_int(s, "seed") for s in seeds]
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds 不能重复")
    if not isinstance(variants, (list, tuple)) or not variants:
        raise ValueError("variants 必须是非空列表")
    variants = [str(v) for v in variants]
    if len(set(variants)) != len(variants):
        raise ValueError("variants 不能重复")

    design_cfg = _as_dict(config.get("design"))
    alpha = (_open_unit(alpha, "alpha") if alpha is not None
             else _open_unit(design_cfg.get("alpha", DEFAULT_ALPHA), "alpha"))
    power = (_open_unit(power, "power") if power is not None
             else _open_unit(design_cfg.get("power", DEFAULT_POWER), "power"))
    ppl_grid = design_cfg.get("ppl_paired_sd_rel_grid", list(PPL_SD_GRID))
    ruler_grid = design_cfg.get("ruler_discordance_grid",
                                list(RULER_DISCORDANCE_GRID))
    try:
        ppl_grid = sorted({_finite_float(x, "ppl_paired_sd_rel_grid",
                                         positive=True)
                           for x in _finite_sequence(ppl_grid,
                                                     "ppl_paired_sd_rel_grid")})
        ruler_grid = sorted({
            _finite_float(x, "ruler_discordance_grid", positive=True,
                          upper=1.0)
            for x in _finite_sequence(ruler_grid,
                                      "ruler_discordance_grid")})
    except ValueError as e:
        raise ValueError(f"设计网格非法：{e}") from None
    if not ppl_grid or not ruler_grid:
        raise ValueError("设计网格不能为空")

    ruler_cfg = _as_dict(config.get("ruler"))
    if ruler_n_samples is None:
        ruler_n_samples = ruler_cfg.get("n_samples", 30)
    ruler_n_samples = _strict_int(ruler_n_samples, "ruler n_samples",
                                 minimum=1)
    ctx_lens = ruler_cfg.get("ctx_lens")
    if isinstance(ctx_lens, (list, tuple)) and ctx_lens:
        n_contexts = len(set(str(c) for c in ctx_lens))
    else:
        n_contexts = 1

    registered = _as_dict(registered)
    hyps = [h for h in _as_list(registered.get("hypotheses"))
            if isinstance(h, dict)]
    scope = _as_dict(registered.get("scope"))
    scope_seeds = scope.get("seeds")
    scope_seeds = (scope_seeds if isinstance(scope_seeds, list) else None)
    scope_seeds_int = None
    if scope_seeds is not None:
        try:
            scope_seeds_int = [_seed_int(s, "预注册 scope seed")
                               for s in scope_seeds]
        except ValueError:
            scope_seeds_int = None

    known = (set(known_variants) if known_variants is not None
             else _known_variant_names())
    planned = set(variants)
    analysis_explicit = analysis_variants is not None
    if analysis_variants is None:
        # 默认只把真正会产出机制分析的门控变体列入分析人群；standard
        # 无输出门、不写 analysis_*.json，不能因此被判成“分析缺失”。
        analysis_variants = [v for v in variants
                             if v in _DEFAULT_GATED_FOR_ANALYSIS]
    if not isinstance(analysis_variants, (list, tuple)):
        raise ValueError("analysis_variants 必须是列表")
    analysis_variants = [str(v) for v in analysis_variants]
    unknown_analysis = sorted(set(analysis_variants) - planned)
    if unknown_analysis:
        raise ValueError(
            f"analysis_variants 含计划外变体：{unknown_analysis}")
    n_seeds = len(seeds)

    mode = (multiplicity if multiplicity is not None
            else design_cfg.get("multiplicity", DEFAULT_MULTIPLICITY))
    if not isinstance(mode, str) or mode not in MULTIPLICITY_MODES:
        raise ValueError(
            f"design.multiplicity 必须是 {MULTIPLICITY_MODES} 之一，"
            f"收到 {mode!r}")
    family_sizes = planned_family_sizes(
        variants, registered,
        analysis_variants if analysis_explicit else None)
    ppl_multiplicity = multiplicity_alpha(
        alpha, family_sizes["ppl_family_size"],
        family_sizes["global_family_size"], mode)
    ruler_multiplicity = multiplicity_alpha(
        alpha, family_sizes["ruler_family_size"],
        family_sizes["global_family_size"], mode)
    auroc_multiplicity = multiplicity_alpha(
        alpha, family_sizes["auroc_family_size"],
        family_sizes["global_family_size"], mode)
    ap_multiplicity = multiplicity_alpha(
        alpha, family_sizes["ap_family_size"],
        family_sizes["global_family_size"], mode)
    ppl_alpha = ppl_multiplicity["alpha_effective"]
    ruler_alpha = ruler_multiplicity["alpha_effective"]
    auroc_alpha = auroc_multiplicity["alpha_effective"]
    ap_alpha = ap_multiplicity["alpha_effective"]

    # ---- 检查 1：最小种子数（低于 3 不判定显著性，属协议内但只能探索）
    if n_seeds >= MIN_SEEDS:
        checks = [_check("seed_minimum", "种子数下限", "pass",
                         f"计划 {n_seeds} 个种子（>= {MIN_SEEDS}）")]
    else:
        checks = [_check(
            "seed_minimum", "种子数下限", "warn",
            f"计划只有 {n_seeds} 个种子（< {MIN_SEEDS}）：只能描述，"
            "不能做显著性/组件有效性判定")]

    # ---- 检查 2：与预注册种子计划一致（少跑不等于“范围外”，但确认性
    # 证据不再覆盖登记的全部种子；必须显式降级/告警）
    if scope_seeds_int is None:
        checks.append(_check(
            "seed_scope", "预注册种子计划", "warn",
            "登记表未提供可读的 scope.seeds，无法核对计划种子是否完整；"
            "确认性结论仍受范围门约束"))
    else:
        missing_scope = [s for s in scope_seeds_int if s not in seeds]
        extra_scope = [s for s in seeds if s not in scope_seeds_int]
        if missing_scope:
            checks.append(_check(
                "seed_scope", "预注册种子计划", "fail",
                f"计划缺少预注册种子 {missing_scope}：即使跑完，"
                "登记范围内仍有未观测种子；确认性判定应降级为探索性"))
        elif extra_scope:
            checks.append(_check(
                "seed_scope", "预注册种子计划", "warn",
                f"计划包含未登记种子 {extra_scope}：结论含登记计划外的抽样，"
                "证据强度打折"))
        else:
            checks.append(_check(
                "seed_scope", "预注册种子计划", "pass",
                f"计划种子与登记表一致（{seeds}）"))

    # ---- 检查 3：登记假设的候选/参照变体都必须在计划矩阵内
    missing_hyps = []
    unmapped = []
    for h in hyps:
        hid = h.get("id", "?")
        cand = h.get("candidate")
        ref = h.get("reference")
        for role, name in (("候选", cand), ("参照", ref)):
            if not isinstance(name, str) or not name:
                missing_hyps.append(f"{hid} 的{role}未命名")
                continue
            if name in known and name not in planned:
                missing_hyps.append(f"{hid} 的{role} {name} 不在计划变体中")
            elif name not in known and name not in planned:
                # “仅查询项基线”等分析内对照不是训练变体：只要求候选可映射。
                if role == "参照":
                    unmapped.append(f"{hid} 的参照 {name} 为分析内对照（无对应变体）")
                else:
                    missing_hyps.append(
                        f"{hid} 的候选 {name} 既不是已知变体也不在计划矩阵中")
    if not hyps:
        checks.append(_check("hypothesis_variants", "预注册假设覆盖", "warn",
                             "登记表无假设可核对"))
    elif missing_hyps:
        checks.append(_check(
            "hypothesis_variants", "预注册假设覆盖", "fail",
            "计划矩阵无法检验全部登记假设：" + "；".join(missing_hyps)))
    else:
        detail = f"{len(hyps)} 条登记假设的候选/训练变体均在计划矩阵中"
        if unmapped:
            detail += "；" + "；".join(sorted(set(unmapped))[:3])
        checks.append(_check("hypothesis_variants", "预注册假设覆盖",
                             "pass", detail))

    # ---- 检查 4：先验功效必须按最终确认阈值规划，而非名义 α
    mult_detail = (
        f"模式={mode}；PPL/RULER 计划家族 m="
        f"{family_sizes['ppl_family_size']}，全局确认族 k="
        f"{family_sizes['global_family_size']}；"
        f"PPL α_eff={ppl_alpha:.6g}、RULER α_eff={ruler_alpha:.6g}、"
        f"AUROC α_eff={auroc_alpha:.6g}、AP α_eff={ap_alpha:.6g}"
        f"（名义 α={alpha:.6g}）。按 Holm 最坏情形 α/max(m,k) 规划，"
        "避免用未校正阈值高估确认性功效")
    if not (ppl_multiplicity["representable"]
            and ruler_multiplicity["representable"]):
        mult_detail += "；警告：家族负担过大导致 α/m 下溢，校正阈值不可表示"
        mult_status = "warn"
    else:
        mult_status = "info"
    checks.append(_check("multiplicity_plan", "确认性多重比较规划",
                         mult_status, mult_detail))

    # ---- PPL 先验灵敏度：对合理 SD 网格给 MDE/功效/所需种子数
    ppl_margin = 0.01  # 与 lega.equivalence.metric_margin("PPL") 一致
    try:
        from lega.equivalence import metric_margin
        spec = metric_margin("PPL")
        if isinstance(spec, dict) and spec.get("margin") is not None:
            ppl_margin = _finite_float(spec.get("margin"), "PPL SESOI",
                                       positive=True)
    except Exception:  # pragma: no cover - 保持模块可独立使用
        pass
    ppl_rows = []
    can_power = n_seeds >= 2
    for sd in ppl_grid:
        if can_power:
            mde = min_detectable_effect(sd, n_seeds, alpha, power)
            pw = paired_t_power(ppl_margin, sd, n_seeds, alpha)
            mde_confirm = min_detectable_effect(
                sd, n_seeds, ppl_alpha, power)
            pw_confirm = paired_t_power(ppl_margin, sd, n_seeds, ppl_alpha)
            pw_nct = (paired_t_power_nct(
                ppl_margin, sd, n_seeds, alpha, n_sim=20000, seed=0)["power"]
                if 2 <= n_seeds <= 10 else None)
        else:
            mde = pw = mde_confirm = pw_confirm = pw_nct = None
        ppl_rows.append({
            "paired_sd_relative": sd,
            "mde_relative": mde,
            "power_for_sesoi": pw,
            "power_for_sesoi_nct": pw_nct,
            "required_seeds_for_sesoi": required_pairs(
                ppl_margin, sd, alpha, power),
            "mde_relative_confirmatory": mde_confirm,
            "power_for_sesoi_confirmatory": pw_confirm,
            "required_seeds_for_sesoi_confirmatory": required_pairs(
                ppl_margin, sd, ppl_alpha, power),
        })
    ppl_break_even = (max_sd_for_effect(ppl_margin, n_seeds, alpha, power)
                      if can_power else None)
    ppl_break_even_confirm = (
        max_sd_for_effect(ppl_margin, n_seeds, ppl_alpha, power)
        if can_power else None)
    declared_ppl_sd = None
    declared_disc = None
    if "ppl_paired_sd_rel" in design_cfg:
        declared_ppl_sd = design_cfg.get("ppl_paired_sd_rel")
        declared_ppl_sd = _finite_float(declared_ppl_sd,
                                        "design.ppl_paired_sd_rel",
                                        positive=True)
        need = required_pairs(ppl_margin, declared_ppl_sd, alpha, power)
        pw = (paired_t_power(ppl_margin, declared_ppl_sd, n_seeds, alpha)
              if can_power else None)
        ok = (can_power and need is not None and need <= n_seeds)
        checks.append(_check(
            "ppl_declared_power", "PPL 计划功效（声明假设）",
            "pass" if ok else "fail",
            f"声明配对差 SD={declared_ppl_sd:.4g}；{n_seeds} 种子对 "
            f"{ppl_margin:.2%} SESOI 的功效≈"
            + (f"{pw:.3f}" if pw is not None else "无法计算（n<2）")
            + "，"
            + (f"需 {need} 个种子（当前 {n_seeds}）" if need is not None
               else "所需种子数超出可计算范围")
            + ("" if ok else "：设计对 SESOI 功效不足")))
    else:
        checks.append(_check(
            "ppl_declared_power", "PPL 计划功效（声明假设）", "info",
            "配置未声明 design.ppl_paired_sd_rel：不把任一 SD 假设当作已知，"
            "仅报告灵敏度曲线（见 ppl.sensitivity）"))

    if declared_ppl_sd is not None:
        need_c = required_pairs(ppl_margin, declared_ppl_sd,
                                ppl_alpha, power)
        pw_c = (paired_t_power(ppl_margin, declared_ppl_sd,
                              n_seeds, ppl_alpha)
                if can_power else None)
        ok_c = (can_power and need_c is not None and need_c <= n_seeds)
        checks.append(_check(
            "ppl_confirmatory_power", "PPL 确认性计划功效（多重比较校正）",
            "pass" if ok_c else "fail",
            f"声明配对差 SD={declared_ppl_sd:.4g}；按 {mode} 的 "
            f"α_eff={ppl_alpha:.6g}（家族 m="
            f"{family_sizes['ppl_family_size']}，全局 k="
            f"{family_sizes['global_family_size']}），{n_seeds} 种子对 "
            f"{ppl_margin:.2%} SESOI 的功效≈"
            + (f"{pw_c:.3f}" if pw_c is not None else "无法计算（n<2）")
            + "，"
            + (f"需 {need_c} 个种子（当前 {n_seeds}）"
               if need_c is not None else "所需种子数超出可计算范围")
            + ("" if ok_c else "：通过多重比较后对 SESOI 功效不足，"
               "当前设计不能确认该效应")))
    else:
        checks.append(_check(
            "ppl_confirmatory_power", "PPL 确认性计划功效（多重比较校正）",
            "info",
            "未声明 design.ppl_paired_sd_rel；见 sensitivity 中的 "
            "power_for_sesoi_confirmatory"))

    # 参考 SD 不是数据估计，只用于展示“常见噪声水平下”的设计灵敏度。
    ref_sd = DEFAULT_PPL_SD_REF
    ref_pw = (paired_t_power(ppl_margin, ref_sd, n_seeds, alpha)
              if can_power else None)
    ref_pw_confirm = (paired_t_power(ppl_margin, ref_sd, n_seeds, ppl_alpha)
                      if can_power else None)
    ref_need = required_pairs(ppl_margin, ref_sd, alpha, power)
    ref_need_confirm = required_pairs(ppl_margin, ref_sd, ppl_alpha, power)
    ref_detail = (
        f"若配对差 SD={ref_sd:.1%}（参考值），{n_seeds} 种子对 "
        f"{ppl_margin:.1%} SESOI 的未校正功效≈"
        + (f"{ref_pw:.3f}" if ref_pw is not None else "无法计算（n<2）")
        + (f"；按确认性 α_eff={ppl_alpha:.6g} 仅≈{ref_pw_confirm:.3f}"
           if ref_pw_confirm is not None else "")
        + (f"，需 {ref_need} 个种子（校正后 {ref_need_confirm}）"
           if ref_need is not None and ref_need_confirm is not None
           else ""))
    if ppl_break_even is not None:
        ref_detail += f"；未校正 80% 功效下最大 SD≈{ppl_break_even:.3%}"
    if ppl_break_even_confirm is not None:
        ref_detail += f"（确认性 {ppl_break_even_confirm:.3%}）"
    checks.append(_check("ppl_reference_sensitivity",
                         "PPL 参考灵敏度（非数据估计）", "info",
                         ref_detail))
    if (ref_pw_confirm is not None and ref_pw_confirm < power):
        checks.append(_check(
            "ppl_confirmatory_reference", "PPL 确认性参考功效", "warn",
            f"参考 SD={ref_sd:.1%} 时，{n_seeds} 种子对 "
            f"{ppl_margin:.1%} SESOI 的确认性功效仅≈{ref_pw_confirm:.3f}"
            f"（目标 {power:.0%}；α_eff={ppl_alpha:.6g}，"
            f"需约 {ref_need_confirm} 个种子）。{n_seeds} 个种子只能"
            "可靠检出远大于 1% 的 PPL 差异；1% SESOI 结论应标为"
            "描述性/探索性"))

    # ---- 上下文档位设计：至少两档且跨度足够，才谈得上“长度校准”
    ctx_values = []
    if isinstance(ctx_lens, (list, tuple)):
        for c in ctx_lens:
            try:
                v = _finite_float(c, "ctx_lens", positive=True)
            except ValueError:
                continue
            ctx_values.append(v)
    if len(ctx_values) < 2:
        checks.append(_check(
            "ruler_context_span", "RULER 上下文档位", "warn",
            f"计划只有 {len(ctx_values)} 个有效上下文档位：无法检验长度"
            "剂量-反应/长度校准主张，跨长度平均也不能替代分档结果"))
    else:
        ratio = max(ctx_values) / min(ctx_values)
        if ratio < 2.0:
            checks.append(_check(
                "ruler_context_span", "RULER 上下文档位", "warn",
                f"上下文档位跨度仅 {min(ctx_values):g}–{max(ctx_values):g}"
                f"（×{ratio:.2f}）：长度效应可能被限制在过窄范围，"
                "只能作该范围内的结论"))
        else:
            checks.append(_check(
                "ruler_context_span", "RULER 上下文档位", "pass",
                f"上下文档位 {min(ctx_values):g}–{max(ctx_values):g}"
                f"（×{ratio:.2f}，{len(ctx_values)} 档）：可检验长度趋势"))

    # ---- RULER 先验灵敏度：每格样本数决定 McNemar 能检出多大的边际差
    ruler_margin = 0.01  # 与 RULER平均 SESOI 一致
    try:
        from lega.equivalence import metric_margin
        spec = metric_margin("RULER平均")
        if isinstance(spec, dict) and spec.get("margin") is not None:
            ruler_margin = _finite_float(spec.get("margin"), "RULER SESOI",
                                         positive=True)
    except Exception:  # pragma: no cover
        pass
    ruler_rows = []
    for disc in ruler_grid:
        try:
            cell = mcnemar_power(ruler_n_samples, disc, ruler_margin, alpha)
            mde = mcnemar_mde(ruler_n_samples, disc, alpha, power)
            need = mcnemar_required_samples(ruler_margin, disc, alpha, power)
            cell_confirm = mcnemar_power(
                ruler_n_samples, disc, ruler_margin, ruler_alpha)
            mde_confirm = mcnemar_mde(
                ruler_n_samples, disc, ruler_alpha, power)
            need_confirm = mcnemar_required_samples(
                ruler_margin, disc, ruler_alpha, power)
        except ValueError:
            continue
        ruler_rows.append({
            "discordance": disc,
            "power_for_sesoi": cell["power"],
            "mde_pp": mde,
            "required_samples_per_cell_for_sesoi": need,
            "power_for_sesoi_confirmatory": cell_confirm["power"],
            "mde_pp_confirmatory": mde_confirm,
            "required_samples_per_cell_for_sesoi_confirmatory": need_confirm,
        })
    ruler_break_even = max_discordance_for_effect(
        ruler_margin, ruler_n_samples, alpha, power)
    ruler_break_even_confirm = max_discordance_for_effect(
        ruler_margin, ruler_n_samples, ruler_alpha, power)
    declared_disc = design_cfg.get("ruler_discordance")
    if declared_disc is not None:
        declared_disc = _finite_float(declared_disc,
                                      "design.ruler_discordance",
                                      positive=True, upper=1.0)
        try:
            need = mcnemar_required_samples(ruler_margin, declared_disc,
                                            alpha, power)
            pw = mcnemar_power(ruler_n_samples, declared_disc,
                               ruler_margin, alpha)["power"]
        except ValueError as e:
            checks.append(_check(
                "ruler_declared_power", "RULER 计划功效（声明假设）",
                "fail", f"声明的不一致率无法计算：{e}"))
        else:
            ok = need is not None and need <= ruler_n_samples
            checks.append(_check(
                "ruler_declared_power", "RULER 计划功效（声明假设）",
                "pass" if ok else "warn",
                f"声明不一致率={declared_disc:.3g}；每格 "
                f"n={ruler_n_samples} 对 {ruler_margin:.2%} SESOI 的功效"
                f"≈{pw:.3f}"
                + (f"，需每格 {need} 个样例" if need is not None else "")
                + ("" if ok else "：当前 RULER 样本量不足以确认该 SESOI，"
                   "等价/优效判定只能作描述性证据")))
    else:
        checks.append(_check(
            "ruler_declared_power", "RULER 计划功效（声明假设）", "info",
            "配置未声明 design.ruler_discordance：仅报告灵敏度曲线"))

    if declared_disc is not None:
        try:
            need_c = mcnemar_required_samples(
                ruler_margin, declared_disc, ruler_alpha, power)
            pw_c = mcnemar_power(ruler_n_samples, declared_disc,
                                 ruler_margin, ruler_alpha)["power"]
        except ValueError as e:
            checks.append(_check(
                "ruler_confirmatory_power",
                "RULER 确认性计划功效（多重比较校正）", "fail",
                f"声明的 RULER 不一致率在确认性阈值下无法计算：{e}"))
        else:
            ok_c = need_c is not None and need_c <= ruler_n_samples
            checks.append(_check(
                "ruler_confirmatory_power",
                "RULER 确认性计划功效（多重比较校正）",
                "pass" if ok_c else "warn",
                f"声明不一致率={declared_disc:.3g}；按 {mode} 的 "
                f"α_eff={ruler_alpha:.6g}（家族 m="
                f"{family_sizes['ruler_family_size']}，全局 k="
                f"{family_sizes['global_family_size']}），每格 "
                f"n={ruler_n_samples} 对 {ruler_margin:.2%} SESOI 的功效"
                f"≈{pw_c:.3f}"
                + (f"，需每格 {need_c} 个样例" if need_c is not None else "")
                + ("" if ok_c else "：通过多重比较后样本量不足，"
                   "等价/优效判定只能作描述性证据")))
    else:
        checks.append(_check(
            "ruler_confirmatory_power",
            "RULER 确认性计划功效（多重比较校正）", "info",
            "未声明 design.ruler_discordance；见 sensitivity 中的 "
            "power_for_sesoi_confirmatory"))

    ref_disc = DEFAULT_RULER_DISCORDANCE_REF
    try:
        ref_cell = mcnemar_power(ruler_n_samples, ref_disc,
                                 ruler_margin, alpha)
        ref_need = mcnemar_required_samples(ruler_margin, ref_disc,
                                            alpha, power)
        ref_mde = mcnemar_mde(ruler_n_samples, ref_disc, alpha, power)
        ref_cell_confirm = mcnemar_power(
            ruler_n_samples, ref_disc, ruler_margin, ruler_alpha)
        ref_need_confirm = mcnemar_required_samples(
            ruler_margin, ref_disc, ruler_alpha, power)
        ref_mde_confirm = mcnemar_mde(
            ruler_n_samples, ref_disc, ruler_alpha, power)
    except ValueError:
        ref_cell, ref_need, ref_mde = {"power": None}, None, None
        ref_cell_confirm, ref_need_confirm, ref_mde_confirm = (
            {"power": None}, None, None)
    cells = n_contexts * _TASKS_FOR_RULER
    pooled_n = ruler_n_samples * cells
    try:
        pooled_mde = mcnemar_mde(pooled_n, ref_disc, alpha, power)
    except ValueError:
        pooled_mde = None
    pooled_grid = []
    for disc in ruler_grid:
        try:
            pooled_grid.append(mcnemar_mde(pooled_n, disc, alpha, power))
        except ValueError:
            continue
    pooled_min = min((v for v in pooled_grid if v is not None),
                     default=None)
    checks.append(_check(
        "ruler_reference_sensitivity", "RULER 参考灵敏度（非数据估计）",
        "info",
        f"若不一致率={ref_disc:.0%}（参考值），每格 n={ruler_n_samples} "
        f"对 {ruler_margin:.1%} SESOI 的未校正功效≈"
        + ((f"{ref_cell.get('power'):.3f}")
           if isinstance(ref_cell.get("power"), float) else "-")
        + (f"；按确认性 α_eff={ruler_alpha:.6g} 仅≈"
           f"{ref_cell_confirm.get('power'):.3f}"
           if isinstance(ref_cell_confirm.get("power"), float) else "")
        + (f"，MDE≈{ref_mde:.2%}"
           + (f"（确认性 {ref_mde_confirm:.2%}）"
              if ref_mde_confirm is not None else "")
           if ref_mde is not None else "")
        + (f"，需每格 {ref_need} 个样例"
           + (f"（确认性 {ref_need_confirm}）"
              if ref_need_confirm is not None else "")
           if ref_need is not None else "")
        + (f"；在 80% 功效下能容忍的最大不一致率≈{ruler_break_even:.4%}"
           + (f"（确认性 {ruler_break_even_confirm:.4%}）"
              if ruler_break_even_confirm is not None else "")
           if ruler_break_even is not None else "")))
    # 参考不一致率下若连 50% 功效都不到，明确告警：这不是“没有效应”，
    # 而是当前每格样本量根本不足以支撑 1pp 级别的实际意义判定。
    ref_power = ref_cell.get("power")
    ref_power_confirm = ref_cell_confirm.get("power")
    if (isinstance(ref_power, float) and ref_power < 0.5
            or isinstance(ref_power_confirm, float)
            and ref_power_confirm < 0.5):
        best_case = (
            f"即使把 {cells} 个格子当作完全独立的 {pooled_n} 个样例"
            f"（对宏观平均最乐观），MDE≈{pooled_mde:.2%}"
            + (f"、网格内最乐观不一致率下仍≈{pooled_min:.2%}"
               if pooled_min is not None else "")
            + "，"
            if pooled_mde is not None else "")
        checks.append(_check(
            "ruler_sample_size", "RULER 每格样本量", "warn",
            f"每格 n={ruler_n_samples} 在参考不一致率={ref_disc:.0%} 下"
            + (f"对 {ruler_margin:.1%} SESOI 的未校正功效仅≈"
               f"{ref_power:.3f}" if isinstance(ref_power, float) else "")
            + (f"；通过多重比较后仅≈{ref_power_confirm:.3f}"
               if isinstance(ref_power_confirm, float) else "")
            + "；"
            + best_case
            + "仍远大于 1pp：RULER 的 SESOI/等价判定只能作描述性证据；"
            "如需确认 1pp 级差异，应按上表增加每格样本数并重新登记"))

    # ---- 分析人群/排除规则必须在开跑前声明（不是事后补的叙述）
    checks.append(_check(
        "analysis_protocol", "分析人群与排除规则", "pass",
        "计划内 (变体, 种子) 全集；只允许协议违规/产物损坏排除，"
        "不得因结果不利或离群排除"))

    if smoke:
        # 冒烟/伪数据本就不在预注册范围；把 fail 降为 warn，避免把
        # “流程连通性测试”误报成正式设计失败，但仍在计划里留痕。
        for c in checks:
            if c["status"] == "fail":
                c["status"] = "warn"
                c["detail"] += "（冒烟运行：仅检查流程，不作确认性设计）"

    endpoint_ppl = {
        "metric": "PPL", "sesoi_relative": ppl_margin,
        "sesoi_source": "predeclared_protocol",
        "planned_seeds": n_seeds,
        "max_paired_sd_for_sesoi": ppl_break_even,
        "max_paired_sd_for_sesoi_confirmatory": ppl_break_even_confirm,
        "multiplicity": dict(ppl_multiplicity),
        "sensitivity": ppl_rows,
    }
    endpoint_ruler = {
        "metric": "RULER平均", "sesoi_pp": ruler_margin,
        "sesoi_source": "predeclared_protocol",
        "samples_per_cell": ruler_n_samples,
        "n_contexts": n_contexts,
        "n_tasks_assumed": _TASKS_FOR_RULER,
        "cells": cells,
        "pooled_samples_best_case": pooled_n,
        "mde_pp_pooled_best_case_at_reference": pooled_mde,
        "mde_pp_pooled_best_case_min_over_grid": pooled_min,
        "max_discordance_for_sesoi": ruler_break_even,
        "max_discordance_for_sesoi_confirmatory":
            ruler_break_even_confirm,
        "multiplicity": dict(ruler_multiplicity),
        "sensitivity": ruler_rows,
    }

    plan = {
        "schema_version": SCHEMA_VERSION,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": "lega.design",
        "inputs": {
            "config_file": config_file,
            "config_sha256": config_sha256,
            "variants": list(variants),
            "seeds": list(seeds),
            "n_seeds": n_seeds,
            "alpha": alpha,
            "alpha_effective": {
                "PPL": ppl_alpha, "RULER": ruler_alpha,
                "AUROC": auroc_alpha, "AP": ap_alpha},
            "multiplicity": mode,
            "power": power,
            "smoke": bool(smoke),
            "registered_hypotheses": [h.get("id") for h in hyps],
            "registration_scope_seeds": scope_seeds_int,
            "analysis_variants": list(analysis_variants),
            "known_variants": sorted(known) if known else None,
        },
        "analysis_protocol": dict(_ANALYSIS_PROTOCOL),
        # 确认性分析计划的只读快照：计划摘要因此同时绑定“登记了哪些
        # 假设、用什么指标/候选/参照”，不只是一个种子/样本量清单。
        "registered_plan": [
            {k: h.get(k) for k in ("id", "metric", "candidate", "reference",
                                   "direction")}
            for h in hyps],
        "assumptions": {
            "ppl_paired_sd_rel": declared_ppl_sd,
            "ruler_discordance": declared_disc,
            "reference_ppl_sd_rel": ref_sd,
            "reference_ruler_discordance": ref_disc,
            "multiplicity": mode,
            "note": (
                "声明假设来自配置 design 块（分析前写死）；参考值是展示用"
                "横轴，不是从当前数据估计的效应量，也不参与任何推断。"
                "确认性功效按最终 Holm 判定阈值规划，不能与未校正功效混用"),
        },
        "multiplicity_plan": {
            "mode": mode,
            "family_sizes": {
                "PPL": family_sizes["ppl_family_size"],
                "RULER": family_sizes["ruler_family_size"],
                "AUROC": family_sizes["auroc_family_size"],
                "AP": family_sizes["ap_family_size"],
                "global_confirmatory": family_sizes["global_family_size"],
            },
            "registered_tests": family_sizes["registered_tests"],
            "planned_comparison_pairs": [list(p) for p in family_sizes["pairs"]],
            "alpha_effective": {
                "PPL": ppl_alpha, "RULER": ruler_alpha,
                "AUROC": auroc_alpha, "AP": ap_alpha},
            "note": (
                "家族大小按计划内全部同指标配对计算（含探索性对比），"
                "全局族按计划内可检验的预注册假设计算；缺失产物只会让"
                "实际家族变小，故这是设计上界/保守口径"),
        },
        "endpoints": {"ppl": endpoint_ppl, "ruler": endpoint_ruler},
        "checks": checks,
        "status": _checklist_status(checks),
        "warnings": [c["detail"] for c in checks
                     if c["status"] in ("warn", "fail")],
    }
    plan["plan_digest"] = plan_digest(plan)
    return plan


