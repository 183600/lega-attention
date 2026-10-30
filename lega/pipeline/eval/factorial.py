"""全因子/推理期消融、预注册假设与门控。"""
from .._shared import _install as _shared_install

_shared_install(globals())

def _detection_diff_tests(runs, key_cand: str, metric: str) -> list:
    """同一 run 内 key_cand(logit) − key_cand(qterm) 的跨种子配对检验。

    “完整门控 logit 比仅查询项更能判断该不该写入”是本方法的核心机制
    主张，但把两个并排的数放在表里不构成证据：它们来自同一批记录、同一
    批样例，必须作为配对差值检验。每个 (变体, 种子) run 的差值当作一次
    观测（H0：差为 0），单一侧缺失的种子整对剔除，n 如实反映有效配对数。
    全家族与 PPL/RULER 一样走 Holm 校正。逐 run 的样例聚类 bootstrap
    区间（analysis JSON 的 auroc_cluster_bootstrap）与本跨种子检验互为
    印证：前者回答“换样例波动多少”，后者回答“换训练种子方向是否稳定”。
    """
    out = []
    runs = runs if isinstance(runs, dict) else {}
    for v in sorted(runs):
        # 机制分析只对 GATED_FOR_ANALYSIS 中的变体生成；补充对照
        # （lega_raw / lega_raw_no_len / lega_detach_conc / null_zero）显式运行时
        # 同样有 analysis JSON，不能按主矩阵 VARIANTS 过滤掉。
        if v not in GATED_FOR_ANALYSIS:
            continue
        m = _variant_seed_map(runs, v)
        seeds = _sorted_seeds(m)
        if not seeds:
            continue
        a, b = [], []
        for s in seeds:
            slot = m.get(s)
            an = slot.get("analysis") if isinstance(slot, dict) else None
            if not isinstance(an, dict):
                continue
            al = _rate_or_none(an.get(key_cand + "_logit"))
            aq = _rate_or_none(an.get(key_cand + "_qterm"))
            if al is None or aq is None:
                continue
            a.append(al)
            b.append(aq)
        if not a:
            continue
        res = paired_t_test(a, b)
        res.update({"candidate": v, "reference": AUROC_DIFF_REFERENCE,
                    "metric": metric, "lower_is_better": False})
        out.append(_enrich_paired(res, a, b))
    return out


# 全因子 2^3 单元格到变体的映射（用于交互效应检验）。
# “none”用 gate_only 而不是 standard：8 个 gated 单元格共享输出门，
# 交互对比必须在同一个门内比较 A/B/C；用无门的 standard 会把门本身
# 重新混进交互项（正是 gate_only 要隔离的混杂）。
# 交互对比式（对第三个因子的两个水平取平均）：
#   I_AB = ½[(AB−A−B+none) + (ABC−AC−BC+C)]
#   I_AC = ½[(AC−A−C+none) + (ABC−AB−BC+B)]
#   I_BC = ½[(BC−B−C+none) + (ABC−AB−AC+A)]


def _stable_factorial_interaction(cells):
    values = [_finite_number(v) for v in cells]
    if any(v is None for v in values):
        return None
    scale = max(abs(v) for v in values)
    if scale == 0.0:
        return 0.0
    total = math.fsum(
        sign * (v / scale)
        for v, sign in zip(values, (1, -1, -1, 1, 1, -1, -1, 1)))
    out = scale * (0.5 * total)
    return out if math.isfinite(out) else None


def _factorial_slots_protocol_ok(slots, metric) -> bool:
    """一组全因子槽位是否共享同一评测协议。"""
    if not slots:
        return False
    if metric == "PPL":
        evals = [s.get("eval") if isinstance(s.get("eval"), dict) else {}
                 for s in slots]
        # 不能把 split 原值直接放进 set：外部/旧版 eval.json 的 split 可能
        # 是数组/对象（不可哈希），会让整份汇总在交互检验处 TypeError。
        # 用与协议核对相同的规范化比较，既保留“缺失=None 与 None 相同”
        # 的旧口径，也能安全处理任意 JSON 值。
        splits = [e.get("split", "val") for e in evals]
        if any(not _same_protocol_value(splits[0], x)
               for x in splits[1:]):
            return False
        protocols = [_ppl_protocol(e) for e in evals]
    else:
        protocols = [_ruler_protocol(
            s.get("ruler") if isinstance(s.get("ruler"), dict) else {})
            for s in slots]
        sigs = [_ruler_cell_signature(
            s.get("ruler") if isinstance(s.get("ruler"), dict) else {})
            for s in slots]
        known = [sig for sig in sigs if sig is not None]
        if len(set(known)) > 1:
            return False
    ref = protocols[0] if protocols else {}
    return all(_same_protocol_fields(ref, p) for p in protocols[1:])


def _factorial_protocol_ok_for_cells(maps, cells, seed, metric) -> bool:
    """指定单元格集合在一个种子上是否齐全且协议一致。"""
    slots = []
    for cell in cells:
        m = maps.get(cell) if isinstance(maps, dict) else None
        slot = m.get(seed) if isinstance(m, dict) else None
        if not isinstance(slot, dict):
            return False
        slots.append(slot)
    return _factorial_slots_protocol_ok(slots, metric)


def _factorial_protocol_ok(maps, seed, metric) -> bool:
    """一个种子的 8 个归一化全因子格是否共享同一评测协议。

    交互对比同时用到全部 8 个格；只要任一格的 split/评测窗口或 RULER
    题面/生成预算/已评测单元不同，整条交互线就混入了非受控差异。判定
    口径与 `_paired_comparisons` 一致：字段集合必须齐全且逐项相同，
    缺失字段不能靠“只比较交集”蒙混过关。
    """
    return _factorial_protocol_ok_for_cells(
        maps, _FACTORIAL_CONTRAST_CELLS, seed, metric)


def _factorial_interaction_tests(runs) -> list:
    """全因子 2^3 设计的交互效应检验（探索性，独立指标家族）。

    组件主效应“有没有用”不等于“效应可加”：A 的收益可能在已有 B 时缩小
    （次可加）或放大（协同）。逐种子计算标准 2^3 交互对比（对第三个因子
    取平均），再对该对比做单样本配对 t 检验（H0：交互=0）。交互检验没有
    预注册方向，因此只报“是否显著偏离可加性”与 Δ 符号，不套用“优/劣”。
    基准格是 gate_only（有门但无 A/B/C），不是无门的 standard。
    """
    out = []
    runs = runs if isinstance(runs, dict) else {}
    for metric, extractor in (
            ("PPL", lambda slot: _ppl_of(slot.get("eval"))),
            ("RULER平均", lambda slot: _ruler_avg_of(slot.get("ruler")))):
        maps = {cell: _variant_seed_map(runs, v)
                for cell, v in _FACTORIAL_CONTRAST_CELLS.items()}
        if any(not m for m in maps.values()):
            continue
        all_seeds = _sorted_seeds(maps["none"])
        seeds = [s for s in all_seeds
                 if _factorial_protocol_ok(maps, s, metric)]
        excluded = len(all_seeds) - len(seeds)
        for pair, (slice_a, slice_b) in _FACTORIAL_CONTRASTS.items():
            vals = []
            for s_ in seeds:
                value = _stable_factorial_interaction(
                    [extractor(maps[cell].get(s_))
                     for cell in slice_a + slice_b])
                if value is not None:
                    vals.append(value)
            if not vals:
                continue
            res = paired_t_test(vals, [0.0] * len(vals))
            res.update({"candidate": f"{pair} 交互", "reference": "可加性（交互=0）",
                        "metric": metric + "交互", "lower_is_better": True,
                        "no_direction": True, "interaction": pair})
            if excluded:
                res["excluded_protocol_mismatches"] = excluded
            out.append(_enrich_paired(res, vals, [0.0] * len(vals)))
    return out


# 全因子 Shapley 分解（探索性）：与交互检验同源，但用全部 8 格把 Full−none
# 的总效应按所有加入顺序平均分摊到 A/B/C，避免单因素边对参照格的依赖。


def _factorial_shapley_tests(runs) -> list:
    """全因子 2^3 的 Shapley 效应分解 + 跨种子检验（探索性）。

    价值函数：PPL 用 v=−log(PPL)（比值尺度上可加，exp(−φ) 即 PPL 几何
    均值比）；RULER 用原始平均 EM（越大越好）。逐种子计算精确 Shapley
    值（3 个因子、8 个格），再对每个因子的 φ 做单样本配对 t 检验
    （H0：φ=0）。同一指标的三因子共用同一 Holm 家族；8 格必须全部运行、
    种子数≥MIN_SEEDS、协议一致，否则如实跳过/标注。

    这不是预注册的确认性检验：Shapley 值依赖价值函数尺度，且把交互
    效应平均分摊后不再对应某个“单因素边”。它回答的是“在完整设计上，
    该因子平均贡献多少”，与去一法/全因子增量并列，不能替代它们。
    """
    out = []
    runs = runs if isinstance(runs, dict) else {}
    for metric, extractor, metric_name in (
            ("PPL", lambda slot: _ppl_of(slot.get("eval")),
             SHAPLEY_PPL_METRIC),
            ("RULER平均", lambda slot: _ruler_avg_of(slot.get("ruler")),
             SHAPLEY_RULER_METRIC)):
        maps = {cell: _variant_seed_map(runs, v)
                for cell, v in _FACTORIAL_CONTRAST_CELLS.items()}
        if any(not m for m in maps.values()):
            continue
        all_seeds = _sorted_seeds(maps["none"])
        seeds = [s for s in all_seeds
                 if _factorial_protocol_ok(maps, s, metric)]
        excluded = len(all_seeds) - len(seeds)
        per_seed = {}
        for s in seeds:
            raw = {}
            ok = True
            for cell in FACTORIAL_CELL_ORDER:
                slot = maps.get(cell, {}).get(s)
                v = extractor(slot) if isinstance(slot, dict) else None
                if v is None:
                    ok = False
                    break
                # PPL=0 是历史夹具的兼容边界值，但 log 分解没有定义；
                # 该种子整条剔除，不用 0/极小值伪造 −log 值。
                if metric == "PPL" and (_finite_number(v) is None
                                       or float(v) <= 0):
                    ok = False
                    break
                raw[cell] = v
            if not ok:
                continue
            try:
                oriented = {c: factorial_oriented_value(metric, raw[c])
                            for c in FACTORIAL_CELL_ORDER}
                dec = factorial_decomposition(oriented)
            except (TypeError, ValueError):
                continue
            per_seed[s] = dec
        if not per_seed:
            continue
        ordered = _sorted_seeds(per_seed)
        totals = [per_seed[s]["total_effect"] for s in ordered]
        total_mean = (float(np.mean(totals)) if totals else None)
        # 占比只在每个种子的总效应都为正时可解释；否则不强行归一化。
        share_ok = bool(totals) and all(float(v) > 1e-12 for v in totals)
        for factor in ("A", "B", "C"):
            vals = [per_seed[s]["phi"].get(factor) for s in ordered]
            vals = [v for v in vals if _finite_number(v) is not None]
            if not vals:
                continue
            res = paired_t_test(vals, [0.0] * len(vals))
            res.update({
                "candidate": f"{factor} 的 Shapley 贡献",
                "reference": "0（无贡献）",
                "metric": metric_name,
                "lower_is_better": False,
                "shapley_factor": factor,
                "shapley_factor_label": SHAPLEY_FACTOR_LABEL[factor],
                "shapley_metric": metric,
                "shapley_total_effect_mean": total_mean,
                "shapley_total_effect_std": (
                    float(np.std(totals, ddof=1)) if len(totals) >= 2 else None),
                "shapley_share": (
                    float(np.mean([per_seed[s]["phi"][factor]
                                   / per_seed[s]["total_effect"]
                                   for s in ordered])) if share_ok else None),
                "shapley_efficiency_gap_max": (
                    max(abs(per_seed[s]["efficiency_gap"])
                        for s in ordered) if ordered else None),
                "excluded_protocol_mismatches": excluded,
                "n_protocol_seeds": len(ordered),
                "exploratory": True,
            })
            if metric == "PPL":
                mean_phi = _finite_number(res.get("mean_diff"))
                res["shapley_ppl_ratio"] = (
                    math.exp(max(-20.0, min(20.0, -float(mean_phi))))
                    if mean_phi is not None else None)
                res["shapley_ppl_ratio_note"] = (
                    "exp(−φ)：<1 表示该因子平均降低 PPL；φ 为 −log PPL "
                    "单位的 Shapley 值")
            out.append(_enrich_paired(res, vals, [0.0] * len(vals)))
    return _apply_holm(out)


def _reparameterization_analysis(runs) -> dict:
    """对比归一化 c 与原始 log max_p 两种参数化下的因子归因。

    该分析需要两套完整 2^3 网格：主矩阵 8 个归一化格，以及
    ``RAW_FACTORIAL_CELLS`` 对应的原始格。两套网格在同一批种子上运行时，
    对每条单因子边和每个 Shapley 分量计算同种子差异
    ``D = raw − normalized``，用于检验“因子效应是否依赖参数化”。

    这不是预注册确认检验，只在显式运行原始对照时输出；默认 10 变体
    矩阵不产生该证据。缺格时返回可机读的 ``missing_variants``，不把
    未运行伪装成“无差异”。
    """
    runs = runs if isinstance(runs, dict) else {}
    normalized = {f"norm:{cell}": v
                  for cell, v in NORMALIZED_FACTORIAL_CELLS.items()}
    raw = {f"raw:{cell}": v
           for cell, v in RAW_FACTORIAL_CELLS.items()}
    all_cells = {**normalized, **raw}
    required = sorted(set(all_cells.values()))
    present = [v for v in required if v in runs]
    missing = [v for v in required if v not in runs]
    ran_raw = sorted(v for v in RAW_SPECIFIC_VARIANTS if v in runs)
    out = {
        "available": False,
        "not_run": not ran_raw,
        "required_variants": required,
        "ran_variants": present,
        "missing_variants": missing,
        "metrics": {},
        "exploratory": True,
    }
    if not ran_raw:
        out["reason"] = ("未运行原始参数化对照（可用 --raw_factorial 追加）")
        return out
    if missing:
        out["reason"] = ("原始/归一化全因子网格不完整，缺少变体："
                         + "、".join(missing))
        return out

    maps = {key: _variant_seed_map(runs, v) for key, v in all_cells.items()}
    seed_sets = [set(m) for m in maps.values()]
    candidates = (sorted(set.intersection(*seed_sets))
                  if seed_sets else [])
    for metric, extractor in (
            ("PPL", lambda slot: _ppl_of(slot.get("eval"))),
            ("RULER平均", lambda slot: _ruler_avg_of(slot.get("ruler")))):
        values = {}
        for variant in required:
            for seed, slot in _variant_seed_map(runs, variant).items():
                value = extractor(slot) if isinstance(slot, dict) else None
                if value is not None:
                    values.setdefault(seed, {})[variant] = value
        protocol_seeds = [
            s for s in candidates
            if _factorial_protocol_ok_for_cells(maps, all_cells, s, metric)]
        usable = {
            s: values[s] for s in protocol_seeds
            if isinstance(values.get(s), dict)
            and all(v in values[s] for v in required)}
        result = reparameterization_analyze(usable, metric)
        result["n_protocol_seeds"] = len(usable)
        result["n_seeds_before_protocol_filter"] = len(candidates)
        result["excluded_missing_or_protocol"] = (
            len(candidates) - len(usable))
        out["metrics"][metric] = result
    out["available"] = any(
        bool(metric.get("contrasts") or metric.get("shapley"))
        for metric in out["metrics"].values()
        if isinstance(metric, dict))
    out["reason"] = (None if out["available"] else
                     "没有可配对的同种子原始/归一化观测")
    return out


# 推理期全因子 Shapley：同一 checkpoint 上干预 2^3 个门控格，而不是重新
# 训练。它回答“训练好的前向里，组件的平均贡献与交互结构是否与训练期
# 全因子消融一致”，不能替代训练期确认性检验。


def _inference_factorial_values(ablations):
    """校验并提取 `lega` 每个种子的推理期 2^3 八格 PPL。

    返回 ``(per_seed, n_excluded)``：``per_seed[seed]`` 只在
    ABC 基线、7 个病灶全部真正施加、且每个病灶与基线共享同一批
    ``window_indices`` 时才建立；缺格/窗口不一致/非正 PPL 的种子整条剔除，
    不按 0 补齐，也不把“没做干预”伪装成某个单元格的观测。非 `lega` 变体
    不计入排除数：它们的 2^3 本来就不完整，不属于“损坏的 lega 证据”。

    推理期八个格子只有在 Full 模型上才同时具备 A/B/C 三个真实因子；
    qgate/gate_only 等变体缺项，清零会得到大量 no-op 格，Shapley/交互值
    退化为“只有存在的因子”，与训练期全因子不可比，故一律不消费。
    """
    per_seed = {}
    excluded = 0
    ablations = ablations if isinstance(ablations, dict) else {}
    for variant in sorted(ablations):
        if variant != "lega":
            continue
        seed_map = ablations.get(variant)
        if not isinstance(seed_map, dict):
            continue
        for seed in _sorted_seeds(seed_map):
            payload = seed_map.get(seed)
            if not isinstance(payload, dict):
                excluded += 1
                continue
            baseline = payload.get("baseline")
            if not isinstance(baseline, dict):
                excluded += 1
                continue
            base_ppl = _finite_number(baseline.get("val_ppl"))
            if base_ppl is None or base_ppl <= 0:
                excluded += 1
                continue
            base_windows = payload.get("window_indices")
            if not isinstance(base_windows, list) or not base_windows:
                excluded += 1
                continue
            recs = payload.get("lesions")
            recs = recs if isinstance(recs, list) else []
            # 外部/旧产物可能把 lesion 写成数组/对象；不可哈希的键会直接让
            # dict 构造失败，必须逐条校验而不是相信 JSON 结构。
            by_id = {}
            for r in recs:
                if not isinstance(r, dict):
                    continue
                lid = r.get("lesion")
                if isinstance(lid, str):
                    by_id[lid] = r
            raw = {"ABC": base_ppl}
            ok = True
            for cell, lesion_id in INFERENCE_FACTORIAL_CELLS.items():
                if lesion_id is None:
                    continue
                rec = by_id.get(lesion_id)
                if not isinstance(rec, dict):
                    ok = False
                    break
                # 只接受真正施加成功、且与基线同窗口的病灶。
                if rec.get("applied") is not True:
                    ok = False
                    break
                if rec.get("window_indices") != base_windows:
                    ok = False
                    break
                ppl = _finite_number(rec.get("val_ppl"))
                if ppl is None or ppl <= 0:
                    ok = False
                    break
                raw[cell] = ppl
            if not ok or len(raw) != len(FACTORIAL_CELL_ORDER):
                excluded += 1
                continue
            per_seed[seed] = raw
    return per_seed, excluded


def _inference_shapley_tests(ablations):
    """由推理期 2^3 病灶格计算逐种子 Shapley 值并做跨种子检验。

    每个种子必须齐全 8 个格（ABC=正常基线，none=A/B/C 全关，其余为
    只保留一个/两个因子的病灶），且每个病灶与基线共享同一批 val 窗口。
    缺格、未真正施加、或窗口不一致的种子整条剔除，不按 0 补齐。

    价值函数与训练期一致：v = −log(PPL)（越大越好），因此 φ>0 表示该
    因子在训练好的前向里降低 PPL；exp(−φ) 为该因子的 PPL 几何均值比。
    """
    out = []
    raw_by_seed, excluded = _inference_factorial_values(ablations)
    per_seed = {}
    for seed in sorted(raw_by_seed):
        raw = raw_by_seed[seed]
        try:
            oriented = {
                c: factorial_oriented_value("PPL", raw[c])
                for c in FACTORIAL_CELL_ORDER}
            per_seed[seed] = factorial_decomposition(oriented)
        except (TypeError, ValueError):
            excluded += 1
    if per_seed:
        ordered = _sorted_seeds(per_seed)
        totals = [per_seed[s]["total_effect"] for s in ordered]
        total_mean = float(np.mean(totals)) if totals else None
        share_ok = bool(totals) and all(float(v) > 1e-12 for v in totals)
        for factor in ("A", "B", "C"):
            vals = [per_seed[s]["phi"].get(factor) for s in ordered]
            vals = [v for v in vals if _finite_number(v) is not None]
            if not vals:
                continue
            res = paired_t_test(vals, [0.0] * len(vals))
            res.update({
                "candidate": f"{factor} 的推理期 Shapley 贡献",
                "reference": "0（无贡献）",
                "metric": INFERENCE_SHAPLEY_METRIC,
                "lower_is_better": False,
                "variant": "lega",
                "shapley_factor": factor,
                "shapley_factor_label": SHAPLEY_FACTOR_LABEL[factor],
                "shapley_metric": "PPL",
                "inference_time": True,
                "shapley_total_effect_mean": total_mean,
                "shapley_total_effect_std": (
                    float(np.std(totals, ddof=1)) if len(totals) >= 2 else None),
                "shapley_share": (
                    float(np.mean([per_seed[s]["phi"][factor]
                                   / per_seed[s]["total_effect"]
                                   for s in ordered])) if share_ok else None),
                "shapley_efficiency_gap_max": (
                    max(abs(per_seed[s]["efficiency_gap"])
                        for s in ordered)),
                "n_protocol_seeds": len(ordered),
                "excluded_protocol_mismatches": excluded,
                "exploratory": True,
            })
            mean_phi = _finite_number(res.get("mean_diff"))
            res["shapley_ppl_ratio"] = (
                math.exp(max(-20.0, min(20.0, -float(mean_phi))))
                if mean_phi is not None else None)
            res["shapley_ppl_ratio_note"] = (
                "exp(−φ)：<1 表示该因子在训练好的前向里平均降低 PPL")
            out.append(_enrich_paired(res, vals, [0.0] * len(vals)))
    return _apply_holm(out)


def _inference_interaction_tests(ablations):
    """推理期 2^3 干预的 A×B/A×C/B×C 交互检验（探索性，独立 Holm 家族）。

    训练期交互检验回答“重新训练时组件收益能否按单组件加总”；这里在同一
    checkpoint 的 8 个推理期格子上回答“训练好的前向里交互结构是否还
    存在”。使用与训练期相同的标准 2^3 对比（对第三个因子取平均）：

        I_AB = ½[(AB−A−B+none) + (ABC−AC−BC+C)]

    数值仍用原始 PPL（与训练期 `PPL交互` 同尺度，便于逐交互对照符号），
    不套用方向判定：交互=0 表示可加，显著非零只说明“不能按单组件加总”。
    与 Shapley 一样只消费 `lega` 的完整 8 格；缺格/未施加/窗口不一致的
    种子整条剔除，不按 0 补齐。
    """
    out = []
    raw_by_seed, excluded = _inference_factorial_values(ablations)
    if not raw_by_seed:
        return out
    ordered = _sorted_seeds(raw_by_seed)
    for pair, (slice_a, slice_b) in _FACTORIAL_CONTRASTS.items():
        # 协议缺格/错窗对每个交互都成立；但某个交互值本身溢出 float
        # 只影响该交互，不能把它的排除数累加到后续交互的 n 上——否则
        # A×C/B×C 会报告一个与自己样本无关的排除数，读者会误以为它们
        # 也丢了同样的种子。
        pair_excluded = excluded
        vals = []
        for seed in ordered:
            raw = raw_by_seed[seed]
            value = _stable_factorial_interaction(
                [raw.get(cell) for cell in slice_a + slice_b])
            if value is not None:
                vals.append(value)
            else:
                # 八格都已通过有限正 PPL 校验；None 只可能是交互值本身
                # 超出 float 表示范围。计入本交互的排除数而不是悄悄缩小 n。
                pair_excluded += 1
        if not vals:
            continue
        res = paired_t_test(vals, [0.0] * len(vals))
        res.update({
            "candidate": f"{pair} 推理期交互",
            "reference": "可加性（交互=0）",
            "metric": INFERENCE_INTERACTION_METRIC,
            "lower_is_better": True,
            "no_direction": True,
            "interaction": pair,
            "variant": "lega",
            "inference_time": True,
            "n_protocol_seeds": len(vals),
            "excluded_protocol_mismatches": pair_excluded,
            "exploratory": True,
        })
        out.append(_enrich_paired(res, vals, [0.0] * len(vals)))
    return _apply_holm(out)


# 因果病灶消融（推理期权重干预）的跨种子指标名与判定口径。


def _ablation_lesion_records(ablations):
    """{变体: {种子: payload}} -> 逐 (变体, 病灶) 的跨种子观测。

    只消费 payload 中 applied=True 的病灶；不适用/失败/窗口协议不一致的
    记录跳过，并在结果里保留计数。每个种子先聚合成一个 ΔNLL（病灶−正常），
    再对种子做配对检验——窗口级 p 只描述单 run，不进入这里的推断。
    """
    out = []
    ablations = ablations if isinstance(ablations, dict) else {}
    for variant in sorted(ablations):
        seed_map = ablations.get(variant)
        if not isinstance(seed_map, dict):
            continue
        for lesion_id in ABLATION_LESION_ORDER:
            spec = ABLATION_LESION_BY_ID.get(lesion_id, {})
            # query_only/conc_only/len_only 是同一 checkpoint 2^3 的“只保留
            # 单因子”格；它们相对 Full 的 ΔNLL 同时去掉另外两个因子，不能
            # 当成单组件因果证据。它们只进入推理期 Shapley/交互分解。
            if spec.get("factorial_cell"):
                continue
            vals, seeds_used, no_op_seeds = [], [], 0
            window_n = 0
            not_applicable = 0
            for seed in _sorted_seeds(seed_map):
                payload = seed_map.get(seed)
                if not isinstance(payload, dict):
                    continue
                recs = payload.get("lesions")
                rec = (next((r for r in recs
                             if isinstance(r, dict)
                             and r.get("lesion") == lesion_id), None)
                       if isinstance(recs, list) else None)
                if not isinstance(rec, dict):
                    continue
                # 只有真正施加成功的病灶才有 ΔNLL 证据。旧/损坏产物若把
                # applied=false（或缺少 applied）的记录连同 delta 一起写进来，
                # 不能让它进入跨种子检验——那会把“没做干预”伪装成效应。
                if (rec.get("applicable") is False
                        or rec.get("applied") is not True):
                    not_applicable += 1
                    continue
                delta = _finite_number(rec.get("delta_loss_mean"))
                if delta is None:
                    continue
                vals.append(delta)
                seeds_used.append(seed)
                if rec.get("no_op"):
                    no_op_seeds += 1
                nw = _integral_value(rec.get("n_windows"))
                if nw and nw > 0:
                    window_n += nw
            if not vals:
                continue
            res = paired_t_test(vals, [0.0] * len(vals))
            mean_delta = _finite_number(res.get("mean_diff"))
            res.update({
                "candidate": variant,
                "reference": f"{variant}:normal",
                "metric": (MATCHED_ABLATION_METRIC
                           if spec.get("matched_control")
                           else CAUSAL_ABLATION_METRIC),
                "lower_is_better": True,
                "lesion": lesion_id,
                "lesion_label": spec.get("label", lesion_id),
                "target": spec.get("target", ""),
                "expected": spec.get("expected"),
                "matched_control": bool(spec.get("matched_control")),
                "matched_of": spec.get("matched_of"),
                "matched_factor": spec.get("matched_factor"),
                "variant": variant,
                "seeds_used": seeds_used,
                "n_no_op_seeds": no_op_seeds,
                "n_not_applicable_seeds": not_applicable,
                "n_windows_total": window_n,
                "mean_ppl_ratio": (
                    math.exp(max(-20.0, min(20.0, float(mean_delta))))
                    if mean_delta is not None else None),
                "mean_delta_ppl_pct": (
                    (math.exp(max(-20.0, min(20.0, float(mean_delta)))) - 1.0)
                    * 100.0 if mean_delta is not None else None),
                "ppl_ratio_clamped": bool(
                    mean_delta is not None
                    and not (-20.0 <= float(mean_delta) <= 20.0)),
            })
            _enrich_paired(res, vals, [0.0] * len(vals))
            # 全部种子 no-op：干预没有真正改变前向，零差不能当“等价证据”。
            if (len(vals) > 0 and no_op_seeds == len(vals)
                    and isinstance(res.get("equivalence"), dict)):
                res["equivalence"].update({
                    "not_applicable": True,
                    "not_applicable_reason": (
                        "不适用：全部种子 no-op（训练后该组件系数已为 0，"
                        "干预未改变前向）"),
                    "status": "not_applicable",
                    "equivalent": False,
                })
            out.append(res)
    # 判定必须在 Holm 校正之后：p_adj 是“显著”的唯一门槛，提前读取会
    # 把全部病灶误判成“未检出”。
    out = _apply_holm(out)
    for res in out:
        # _apply_holm 的“显著劣于”在语义上正确（病灶比正常差），但本表
        # 要回答的是“该机制在推理期是否有因果贡献”，故另给 causal_verdict，
        # 同时保留 Holm 原判供审计。
        res["holm_verdict"] = res.get("verdict")
        mean_delta = _finite_number(res.get("mean_diff"))
        n = _integral_value(res.get("n"))
        no_op_seeds = _integral_value(res.get("n_no_op_seeds")) or 0
        if n is None or n < MIN_SEEDS:
            verdict = f"种子不足（n={n}），不作判定"
        elif no_op_seeds == n:
            verdict = ("全部种子为 no-op：训练后该组件系数已为 0，"
                       "推理期未使用该输入（不是“无效应”的证据）")
        elif res.get("p_adj") is not None and res["p_adj"] < ALPHA:
            if mean_delta is not None and mean_delta > 0:
                verdict = "病灶显著变差：推理期该机制有因果贡献"
            elif mean_delta is not None and mean_delta < 0:
                verdict = "病灶显著变好：该机制在推理期反向有害/冗余"
            else:
                verdict = "显著但均值为 0，无法解释"
            if res.get("within_noise"):
                verdict += "；⚠ 效应落在种子噪声内"
            if res.get("loo_sign_stable") is False:
                verdict += "；⚠ 留一法方向翻转"
        else:
            verdict = ("未检出差异（Holm 校正后）；不等于无因果作用，"
                       "须对照 MDE/窗口功效解读")
        # 部分种子 no-op 时必须在判定里露出，否则读者会把“该组件在部分
        # checkpoint 上根本没被使用”误读成“病灶无效应”。
        if no_op_seeds and no_op_seeds < n:
            verdict = f"部分种子 no-op（{no_op_seeds}/{n}）；" + verdict
        if res.get("matched_control"):
            verdict = "活动匹配对照：" + verdict
        res["causal_verdict"] = verdict
        res["causal_contribution"] = bool(
            n is not None and n >= MIN_SEEDS
            and res.get("p_adj") is not None
            and res["p_adj"] < ALPHA
            and mean_delta is not None and mean_delta > 0
            and not res.get("within_noise")
            and res.get("loo_sign_stable") is not False
            and no_op_seeds < n)
    return out


def _matched_ablation_contrasts(ablations):
    """原始组件病灶 − 活动匹配病灶的跨种子配对差（独立 Holm 家族）。

    原始 `no_query/no_conc/no_len` 清零系数时，既去掉了信息内容，也改变了
    该头/层的平均门控开度（写回强度的乘性因子）。匹配病灶在去掉同一
    组件的同时逐头平移 d，使平均 g 与基线相同。因此逐种子的

        ΔNLL(raw) − ΔNLL(matched)

    与“未匹配版额外改变的平均门控开度（及其伴随的分布变化）”共变的部分。
    差值显著为正不代表原始病灶没有信息特异效应：它只说明原始效应中有一部分
    与门控开度变化共变；信息特异部分应直接看匹配病灶相对正常基线的独立
    检验。
    配对必须同变体、同种子、同窗口；任一记录未成功施加则整对剔除。
    """
    out = []
    ablations = ablations if isinstance(ablations, dict) else {}
    for variant in sorted(ablations):
        seed_map = ablations.get(variant)
        if not isinstance(seed_map, dict):
            continue
        for raw_id, matched_id, factor in MATCHED_LESION_PAIRS:
            vals, seeds_used = [], []
            n_pairs_missing = 0
            n_no_op_pairs = 0
            for seed in _sorted_seeds(seed_map):
                payload = seed_map.get(seed)
                if not isinstance(payload, dict):
                    continue
                recs = payload.get("lesions")
                if not isinstance(recs, list):
                    continue

                def _find(lid):
                    return next((r for r in recs
                                 if isinstance(r, dict)
                                 and r.get("lesion") == lid), None)

                raw = _find(raw_id)
                matched = _find(matched_id)
                if (not isinstance(raw, dict) or not isinstance(matched, dict)
                        or raw.get("applicable") is False
                        or matched.get("applicable") is False
                        or raw.get("applied") is not True
                        or matched.get("applied") is not True):
                    n_pairs_missing += 1
                    continue
                wi_raw = raw.get("window_indices")
                wi_matched = matched.get("window_indices")
                if (isinstance(wi_raw, list) and isinstance(wi_matched, list)
                        and wi_raw != wi_matched):
                    n_pairs_missing += 1
                    continue
                d_raw = _finite_number(raw.get("delta_loss_mean"))
                d_matched = _finite_number(matched.get("delta_loss_mean"))
                if d_raw is None or d_matched is None:
                    n_pairs_missing += 1
                    continue
                diff = d_raw - d_matched
                if not math.isfinite(diff):
                    n_pairs_missing += 1
                    continue
                vals.append(diff)
                seeds_used.append(seed)
                if raw.get("no_op") or matched.get("no_op"):
                    n_no_op_pairs += 1
            if not vals:
                continue
            res = paired_t_test(vals, [0.0] * len(vals))
            res.update({
                "candidate": f"{variant}:{raw_id}",
                "reference": f"{variant}:{matched_id}",
                "metric": MATCHED_ABLATION_CONTRAST_METRIC,
                "no_direction": True,
                "lesion": matched_id,
                "lesion_label": f"{factor} 未匹配−活动匹配",
                "matched_of": raw_id,
                "matched_factor": factor,
                "variant": variant,
                "seeds_used": seeds_used,
                "n_pairs_missing": n_pairs_missing,
                "n_no_op_pairs": n_no_op_pairs,
            })
            out.append(_enrich_paired(res, vals, [0.0] * len(vals)))
    out = _apply_holm(out)
    for res in out:
        mean_diff = _finite_number(res.get("mean_diff"))
        n = _integral_value(res.get("n"))
        if n is None or n < MIN_SEEDS:
            verdict = f"种子不足（n={n}），不作判定"
        elif (res.get("p_adj") is not None and res["p_adj"] < ALPHA
              and mean_diff is not None and mean_diff > 0):
            verdict = ("原始病灶效应显著大于活动匹配版：原始 ΔNLL 中有一部分"
                       "可由平均门控开度变化解释；信息特异效应看匹配版")
        elif (res.get("p_adj") is not None and res["p_adj"] < ALPHA
              and mean_diff is not None and mean_diff < 0):
            verdict = ("活动匹配版反而显著更差：原始效应不能由平均门控开度"
                       "解释，须检查匹配分布差异或该组件的信息作用")
        elif res.get("p_adj") is not None and res["p_adj"] < ALPHA:
            verdict = "显著但均值为 0，无法解释"
        else:
            verdict = ("未检出原始−匹配差异（Holm 校正后）；不能据此断言"
                       "原始效应全部来自信息内容或全部来自平均门控开度")
        if res.get("within_noise"):
            verdict += "；⚠ 效应落在种子噪声内"
        if res.get("loo_sign_stable") is False:
            verdict += "；⚠ 留一法方向翻转"
        if (res.get("n_no_op_pairs") or 0) > 0:
            verdict += f"；⚠ {res['n_no_op_pairs']} 对含 no-op 病灶"
        res["matched_verdict"] = verdict
        # 对比差值没有“候选更好/更差”的方向，不能把 _apply_holm 的通用
        # 方向判词留给机器读取；规范字段直接使用本对比专用判词。
        res["verdict"] = verdict
    return out


def _dose_response_tests(doses):
    """连续剂量-反应的跨种子配对检验与组件级曲线摘要。

    输入 ``{变体: {种子: dose_response payload}}``。每个 (变体, 组件) 先取
    各种子都存在的 λ 交集，避免“不同 λ 用不同种子子集”导致曲线不可比；
    再对每个 λ 的 ΔNLL(λ)=NLL(λ)−NLL(λ=1) 做跨种子单样本配对 t 检验。
    A/B/C 与局部斜率各自按 metric 分 Holm 家族。网格最优 λ、二次顶点与
    改善量只作曲线描述：固定网格挑最优会引入 winner's curse，不能据此
    重选 checkpoint 或改写预注册判定。
    """
    doses = doses if isinstance(doses, dict) else {}
    raw_tests = []
    summaries = []
    excluded_variants = []
    for variant in sorted(doses):
        seed_map = doses.get(variant)
        if not isinstance(seed_map, dict):
            excluded_variants.append({"variant": variant,
                                      "reason": "payload 不是种子映射"})
            continue
        components = set()
        for seed in _sorted_seeds(seed_map):
            payload = seed_map.get(seed)
            if (not isinstance(payload, dict)
                    or payload.get("schema") != 1
                    or not isinstance(payload.get("components"), dict)):
                continue
            components.update(payload["components"].keys())
        for component in DOSE_COMPONENT_ORDER:
            if component not in components:
                continue
            point_rows = []
            not_applicable = 0
            missing = 0
            for seed in _sorted_seeds(seed_map):
                payload = seed_map.get(seed)
                if not isinstance(payload, dict):
                    missing += 1
                    continue
                comp = (payload.get("components") or {}).get(component)
                if not isinstance(comp, dict):
                    missing += 1
                    continue
                if comp.get("applicable") is False:
                    not_applicable += 1
                    continue
                base_idx = payload.get("window_indices")
                points = comp.get("points")
                if not isinstance(points, list):
                    missing += 1
                    continue
                by_factor = {}
                for point in points:
                    if not isinstance(point, dict):
                        continue
                    if point.get("applied") is not True:
                        continue
                    f = _finite_number(point.get("factor"))
                    delta = _finite_number(point.get("delta_loss_mean"))
                    if f is None or delta is None:
                        continue
                    wi = point.get("window_indices")
                    if (isinstance(base_idx, list) and isinstance(wi, list)
                            and wi != base_idx):
                        continue
                    by_factor[f] = {
                        "delta": float(delta),
                        "no_op": bool(point.get("no_op")),
                        "n_windows": _integral_value(
                            point.get("n_windows")) or 0,
                    }
                diag = comp.get("diagnostics")
                if not by_factor or not isinstance(diag, dict):
                    missing += 1
                    continue
                point_rows.append({
                    "seed": seed,
                    "points": by_factor,
                    "diagnostics": diag,
                })
            if not point_rows:
                summaries.append({
                    "variant": variant,
                    "component": component,
                    "label": DOSE_COMPONENT_LABEL.get(component, component),
                    "available": False,
                    "reason": "没有可配对的剂量点",
                    "n_not_applicable_seeds": not_applicable,
                    "n_missing_seeds": missing,
                })
                continue

            common_factors = set(point_rows[0]["points"])
            for row in point_rows[1:]:
                common_factors &= set(row["points"])
            common_factors = sorted(common_factors)
            non_base = [f for f in common_factors
                        if abs(f - 1.0) > 1e-12]
            if not non_base:
                summaries.append({
                    "variant": variant,
                    "component": component,
                    "label": DOSE_COMPONENT_LABEL.get(component, component),
                    "available": False,
                    "reason": "各种子共同剂量只有 λ=1，无法检验剂量效应",
                    "n_not_applicable_seeds": not_applicable,
                    "n_missing_seeds": missing,
                    "common_factors": common_factors,
                })
                continue

            rows = [row for row in point_rows
                    if common_factors and set(common_factors)
                    <= set(row["points"])]
            seeds_used = [row["seed"] for row in rows]
            n_windows_total = sum(
                row["points"][f]["n_windows"]
                for row in rows for f in common_factors)
            component_tests = []
            for f in non_base:
                vals = [row["points"][f]["delta"] for row in rows]
                no_ops = sum(1 for row in rows
                             if row["points"][f]["no_op"])
                res = paired_t_test(vals, [0.0] * len(vals))
                mean_delta = _finite_number(res.get("mean_diff"))
                res.update({
                    "candidate": (f"{variant}:{component} "
                                  f"λ={f:g}"),
                    "reference": f"{variant}:{component} λ=1",
                    "metric": f"{DOSE_RESPONSE_METRIC}[{component}]",
                    "lower_is_better": True,
                    "dose_variant": variant,
                    "dose_component": component,
                    "dose_label": DOSE_COMPONENT_LABEL.get(component,
                                                           component),
                    "dose_factor": float(f),
                    "dose_family": "factor",
                    "replicates_lesion": (
                        DOSE_COMPONENT_SPECS.get(component, {}).get(
                            "training_variant")
                        if abs(f) <= 1e-12 else None),
                    "seeds_used": list(seeds_used),
                    "n_no_op_seeds": int(no_ops),
                    "all_no_op": bool(no_ops == len(rows)),
                    "n_windows_total": int(n_windows_total),
                    "mean_ppl_ratio": (
                        math.exp(max(-20.0, min(20.0, mean_delta)))
                        if mean_delta is not None else None),
                    "ppl_ratio_clamped": bool(
                        mean_delta is not None
                        and not (-20.0 <= mean_delta <= 20.0)),
                    "exploratory": True,
                })
                component_tests.append(
                    _enrich_paired(res, vals, [0.0] * len(vals)))

            slopes = []
            slope_seeds = []
            for row in rows:
                slope = _finite_number(
                    (row["diagnostics"] or {}).get("local_slope"))
                if slope is None:
                    continue
                slopes.append(float(slope))
                slope_seeds.append(row["seed"])
            if slopes:
                sres = paired_t_test(slopes, [0.0] * len(slopes))
                sres.update({
                    "candidate": f"{variant}:{component} λ=1 局部斜率",
                    "reference": "λ=1 处局部斜率=0",
                    "metric": f"{DOSE_SLOPE_METRIC}[{component}]",
                    "lower_is_better": False,
                    "dose_variant": variant,
                    "dose_component": component,
                    "dose_label": DOSE_COMPONENT_LABEL.get(component,
                                                           component),
                    "dose_factor": 1.0,
                    "dose_family": "local_slope",
                    "seeds_used": slope_seeds,
                    "n_windows_total": int(n_windows_total),
                    "exploratory": True,
                })
                component_tests.append(
                    _enrich_paired(sres, slopes, [0.0] * len(slopes)))

            raw_tests.extend(component_tests)
            best_factors = []
            vertices = []
            vertex_in_grid = []
            slopes_all = []
            zero_deltas = []
            flip_deltas = []
            baseline_min = []
            for row in rows:
                diag = row["diagnostics"]
                bf = _finite_number(diag.get("best_factor"))
                if bf is not None:
                    best_factors.append(float(bf))
                vertex = _finite_number(diag.get("quadratic_vertex"))
                if vertex is not None:
                    vertices.append(float(vertex))
                vertex_in_grid.append(bool(
                    diag.get("quadratic_vertex_in_grid")))
                slope = _finite_number(diag.get("local_slope"))
                if slope is not None:
                    slopes_all.append(float(slope))
                zd = _finite_number(diag.get("zero_delta"))
                if zd is not None:
                    zero_deltas.append(float(zd))
                fd = _finite_number(diag.get("sign_flip_delta"))
                if fd is not None:
                    flip_deltas.append(float(fd))
                baseline_min.append(bool(
                    diag.get("baseline_is_grid_minimum")))
            slope_mean, slope_sd = mean_std(slopes_all)
            vertex_mean, vertex_sd = mean_std(vertices)
            zero_mean, zero_sd = mean_std(zero_deltas)
            flip_mean, flip_sd = mean_std(flip_deltas)
            slope_test = next(
                (t for t in component_tests
                 if t.get("dose_family") == "local_slope"), {})
            counts = []
            for f in sorted(set(best_factors)):
                counts.append({"factor": float(f),
                               "count": int(sum(
                                   1 for x in best_factors
                                   if abs(x - f) <= 1e-12))})
            summaries.append({
                "variant": variant,
                "component": component,
                "label": DOSE_COMPONENT_LABEL.get(component, component),
                "available": True,
                "common_factors": common_factors,
                "seeds_used": list(seeds_used),
                "n_seeds": len(rows),
                "n_not_applicable_seeds": int(not_applicable),
                "n_missing_seeds": int(missing),
                "n_windows_total": int(n_windows_total),
                "local_slope_mean": slope_mean,
                "local_slope_sd": slope_sd,
                "local_slope_p_adj": slope_test.get("p_adj"),
                "local_slope_verdict": None,
                "quadratic_vertex_mean": vertex_mean,
                "quadratic_vertex_sd": vertex_sd,
                "quadratic_vertex_in_grid_rate": (
                    sum(vertex_in_grid) / len(vertex_in_grid)
                    if vertex_in_grid else None),
                "best_factor_counts": counts,
                "baseline_is_grid_minimum_rate": (
                    sum(baseline_min) / len(baseline_min)
                    if baseline_min else None),
                "zero_delta_mean": zero_mean,
                "zero_delta_sd": zero_sd,
                "sign_flip_delta_mean": flip_mean,
                "sign_flip_delta_sd": flip_sd,
                "exploratory": True,
            })

    raw_tests = _apply_holm(raw_tests)
    by_component = {}
    for res in raw_tests:
        res["holm_verdict"] = res.get("verdict")
        n = _integral_value(res.get("n"))
        mean_diff = _finite_number(res.get("mean_diff"))
        family = res.get("dose_family")
        if n is None or n < MIN_SEEDS:
            verdict = f"种子不足（n={n}），不作剂量判定"
        elif family == "factor" and res.get("all_no_op"):
            f = res.get("dose_factor")
            verdict = (f"λ={f:g} 全部种子 no-op：该系数训练后已为 0/未改变"
                       "前向，不能把零差当作“该剂量无效应”")
        elif family == "factor" and res.get("p_adj") is not None \
                and res["p_adj"] < ALPHA and mean_diff is not None:
            f = res.get("dose_factor")
            if mean_diff > 0:
                verdict = (f"λ={f:g} 显著更差：当前系数优于该剂量"
                           "（Holm 校正后）")
            elif mean_diff < 0:
                verdict = (f"λ={f:g} 显著更好：当前系数可能未在最优"
                           "（Holm 校正后；描述性）")
            else:
                verdict = "显著但均值为 0，无法解释"
        elif family == "factor":
            f = res.get("dose_factor")
            verdict = (f"未检出 λ={f:g} 与 λ=1 的差异（Holm 校正后）；"
                       "不能据此断言该剂量无效应，须对照 MDE")
        elif family == "local_slope" and res.get("p_adj") is not None \
                and res["p_adj"] < ALPHA and mean_diff is not None:
            if mean_diff < 0:
                verdict = ("λ=1 局部斜率显著为负：增大该系数可降低 NLL，"
                           "当前剂量可能偏小（描述性）")
            elif mean_diff > 0:
                verdict = ("λ=1 局部斜率显著为正：减小该系数可降低 NLL，"
                           "当前剂量可能偏大（描述性）")
            else:
                verdict = "显著但斜率为 0，无法解释"
        elif family == "local_slope":
            verdict = ("未检出 λ=1 局部斜率（Holm 校正后）；不能断言当前"
                       "系数处于最优或非最优")
        else:
            verdict = "无法判定剂量反应"
        if res.get("within_noise"):
            verdict += "；⚠ 效应落在种子噪声内"
        if res.get("below_mde"):
            verdict += "；⚠ 效应小于当前种子数的 MDE"
        if res.get("loo_sign_stable") is False:
            verdict += "；⚠ 留一法方向翻转"
        res["dose_verdict"] = verdict
        res["verdict"] = verdict
        by_component.setdefault(
            (res.get("dose_variant"), res.get("dose_component")),
            []).append(res)

    for summary in summaries:
        key = (summary.get("variant"), summary.get("component"))
        tests = by_component.get(key, [])
        slope = next((t for t in tests
                      if t.get("dose_family") == "local_slope"), {})
        summary["local_slope_p_adj"] = slope.get("p_adj")
        summary["local_slope_verdict"] = slope.get("dose_verdict")
        summary["n_tests"] = len(tests)
        if summary.get("available"):
            rate = summary.get("baseline_is_grid_minimum_rate")
            if slope.get("dose_verdict"):
                summary["verdict"] = slope["dose_verdict"]
            elif rate is not None and rate < 1.0:
                summary["verdict"] = (
                    f"基线仅在 {rate:.0%} 种子上为网格最小点；剂量曲线"
                    "提示当前系数未必最优（描述性）")
            else:
                summary["verdict"] = "剂量曲线不足以判定当前系数是否最优"

    available = any(s.get("available") for s in summaries)
    return {
        "available": bool(available),
        "tests": raw_tests,
        "components": summaries,
        "excluded_variants": excluded_variants,
        "note": ("A/B/C 与局部斜率各自按 metric 分 Holm 家族；λ=1 为训练"
                 "得到的原始系数，λ=0 等价于对应组件的推理期清零。局部"
                 "斜率、二次顶点与网格最优 λ 均为描述性诊断；固定网格上"
                 "挑最优会引入 winner's curse，不据此重选 checkpoint 或"
                 "改写预注册判定。默认只在 val 上做，绝对水平受选型偏差"
                 "影响，只作机制定位。"),
    }


def _auroc_diff_tests(runs) -> list:
    """AUROC(logit)−AUROC(qterm) 配对差检验（H6 的判定依据）。"""
    return _detection_diff_tests(runs, "auroc", AUROC_DIFF_METRIC)


def _ap_diff_tests(runs) -> list:
    """PR-AUC(logit)−PR-AUC(qterm) 配对差检验（不平衡下的主指标口径，H7）。"""
    return _detection_diff_tests(runs, "auprc", AP_DIFF_METRIC)


# 注册假设（confirmatory）：在正式实验前登记的方向性主张；未登记的对比
# 一律视为事后（exploratory）分析，只能作为提出新假设的线索。
# 登记表与防篡改锁链的实现见 lega/hypotheses.py（锁基线 + 修订链）。


def load_registered_hypotheses(path: str = HYPOTHESES_PATH) -> dict:
    """读取预注册假设文件；缺失/损坏时返回空 dict（汇总页显式提示）。"""
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
    except (OSError, ValueError) as e:
        print(f"[summary] 警告：注册假设文件不可读（{e}），跳过判定")
        return {}
    if not isinstance(d, dict):
        print("[summary] 警告：注册假设文件顶层不是 JSON 对象，跳过判定")
        return {}
    if not isinstance(d.get("hypotheses"), list) or not d["hypotheses"]:
        print("[summary] 警告：注册假设文件无 hypotheses 条目，跳过判定")
        return {}
    return d


def evaluate_registered_hypotheses(registered: dict, tests: list,
                                    alpha: float = ALPHA) -> list:
    """把预注册假设逐条对照 Holm 校正后的配对检验，判定 支持/拒绝/未定。

    “拒绝”只在方向相反且校正后显著时给出（假设被证伪）；
    “未定”包括种子不足、对比未运行、校正后不显著——不显著绝不等于
    “假设成立”，也不等于“无效应”，配合 MDE 才能判断是功效不足还是
    真的没有效应。
    """
    hypotheses = _as_dict(registered).get("hypotheses")
    if not isinstance(hypotheses, list) or not hypotheses:
        return []
    out = []
    for h in hypotheses:
        if not isinstance(h, dict):
            continue
        t = _find_test(tests, (h.get("candidate"), h.get("reference")),
                       h.get("metric"))
        item = {"id": h.get("id"), "claim": h.get("claim"),
                "metric": h.get("metric"),
                "candidate": h.get("candidate"), "reference": h.get("reference"),
                "direction": h.get("direction"),
                # p 为家族内校正前的原始配对 p；全局确认族校正需要它，
                # 不能只保存 p_adj（否则无法跨指标家族统一校正）。
                "p": None, "p_adj": None, "p_perm_adj": None,
                "p_adj_confirmatory": None, "confirmatory_family_size": None,
                "global_fwer_pass": None, "status_family": None,
                "mean_diff": None,
                "n": None, "mde": None, "within_noise": None,
                "below_mde": None, "status": None, "detail": None}
        if t is None:
            item["status"] = "未检验"
            item["detail"] = "对应的对比/指标没有检验结果（run 缺失或未评测）"
        else:
            n = _integral_value(t.get("n"))
            if n is None:
                item["status"] = "未定"
                item["detail"] = f"无法检验（样本量缺失或非法 n={t.get('n')!r}）"
            elif n < MIN_SEEDS:
                item["status"] = "未定"
                item["detail"] = f"种子不足（n={n}<{MIN_SEEDS}）"
            elif t.get("p_adj") is None:
                item["status"] = "未定"
                item["detail"] = "无法检验（方差退化或指标缺失）"
            elif t["p_adj"] >= alpha:
                item["status"] = "未定"
                item["detail"] = "Holm 校正后不显著"
                if t.get("mde") is not None and t.get("mean_diff") is not None:
                    item["detail"] += (f"（观测差 {t['mean_diff']:+.4f}，"
                                       f"当前种子数可检测的最小效应 "
                                       f"{t['mde']:+.4f}）")
            else:
                better = (t["mean_diff"] < 0
                          if t.get("lower_is_better") is True
                          else t["mean_diff"] > 0)
                supported = (better if h.get("direction") == "candidate_better"
                             else not better)
                item["status"] = "支持" if supported else "拒绝"
                item["detail"] = ("方向与登记一致且校正后显著" if supported
                                  else "校正后显著但方向与登记相反")
                if (t.get("p_perm_adj") is not None
                        and t["p_perm_adj"] >= alpha):
                    item["detail"] += "；⚠ 置换检验不显著，小样本下需谨慎"
        if t is not None:
            item.update({"p": t.get("p"),
                         "p_adj": t.get("p_adj"),
                         "p_perm_adj": t.get("p_perm_adj"),
                         "mean_diff": t.get("mean_diff"),
                         "n": t.get("n"), "mde": t.get("mde"),
                         "within_noise": t.get("within_noise"),
                         "below_mde": t.get("below_mde")})
            note = _noise_verdict_note(t)
            if note and item.get("detail"):
                item["detail"] += note
            # 预注册判定只回答“方向是否显著”，不等价于“效应有实际意义”。
            # 把 TOST/SESOI 作为补充字段挂上，避免把“显著但 0.1%”写成
            # 重要收益，也避免把“未定”直接读成“无差异”。
            eq = t.get("equivalence")
            if isinstance(eq, dict):
                item.update({
                    "equivalence_status": eq.get("status"),
                    "equivalence_label": eq.get("label"),
                    "equivalence_p_tost_adj": eq.get("p_tost_adj"),
                    "equivalence_margin": eq.get("margin"),
                    "equivalence_margin_label": eq.get("margin_label"),
                })
                eq_status = eq.get("status")
                if item.get("status") in ("支持", "拒绝"):
                    if eq_status == "significant_below_sesoi":
                        item["detail"] += ("；幅度低于 SESOI，实际意义有限"
                                           "（不能只报“显著”就当作重要收益）")
                elif item.get("status") == "未定":
                    if eq_status == "equivalent":
                        item["detail"] += ("；等价性：差异小于 SESOI"
                                           "（不能据此说效应恰为 0）")
                    elif eq_status == "significant_below_sesoi":
                        item["detail"] += "；统计显著但幅度低于 SESOI"
                    elif eq_status == "undetermined":
                        item["detail"] += ("；等价性未定（区间未排除 SESOI，"
                                           "不能宣称无差异）")
        out.append(item)
    return apply_global_fwer(out, alpha=alpha)


def apply_global_fwer(hyp_eval: list, alpha: float = ALPHA) -> list:
    """跨全部可检验预注册假设再做一次 Holm 校正（确认性结论的整体 FWER）。

    指标家族内校正只控制“该指标家族”的 FWER；H1–H7 跨越 PPL/RULER/
    AUROC/AP 多个家族，各自在 5% 下判定时，整体至少一次假阳性的概率仍
    可能超过 5%。这里把所有“已检验且有 n>=MIN_SEEDS”的预注册假设组成
    一个确认族，用原始 p 做 Holm 校正（不能用已按更大探索族校正过的
    p_adj 再校正，否则是重复惩罚）：家族内显著但全局不显著的主张降级为
    “未定”，并保留 status_family 供审计。

    该门只会让判定更严格，不会把原先不支持的主张升级；未检验/种子不足
    的假设不进入族（也不被这条门改变状态）。
    """
    items = [h for h in (hyp_eval if isinstance(hyp_eval, list) else [])
             if isinstance(h, dict)]
    for h in items:
        # 允许重复调用：status_family 固定为第一次进入全局门前的家族判定。
        if h.get("status_family") is None:
            h["status_family"] = h.get("status")
    tested = [h for h in items
              if h.get("p") is not None
              and isinstance(h.get("n"), int)
              and not isinstance(h.get("n"), bool)
              and h["n"] >= MIN_SEEDS
              and h.get("status_family") in ("支持", "拒绝", "未定")]
    pads = holm_bonferroni([h.get("p") for h in tested])
    m = len(tested)
    for h, padj in zip(tested, pads):
        h["p_adj_confirmatory"] = padj
        h["confirmatory_family_size"] = m
        h["global_fwer_pass"] = bool(padj is not None and padj < alpha)
        if (h.get("status_family") in ("支持", "拒绝")
                and h["global_fwer_pass"] is False):
            h["status"] = "未定"
            h["detail"] = ((h.get("detail") or "")
                           + "；跨全部预注册假设的全局 Holm 校正后不显著"
                             "（确认族整体 FWER），不能作为确认性结论")
    return items


def apply_integrity_gate(hyp_eval: list, integrity: dict) -> list:
    """预注册完整性失败时给全部确认性判定加盖“无效”戳。

    预注册的全部意义在于事后不能静默改写；一旦 lega/hypotheses.py 的
    锁链校验发现登记表被删/改/无留痕新增，这些判定就不再具有 confirmatory
    地位——逐条标注 valid 标志与原因，汇总页显式降级为探索性，
    而不是假装边界仍然成立。
    """
    # 只接受真正的 bool true：字符串 "false"/1 经 truthiness 会错误地授予
    # 确认性资格，恰好绕过“完整性失败必须降级”的保护。
    valid = _as_dict(integrity).get("confirmatory_valid") is True
    reason = ("登记表完整性校验通过" if valid else
              "预注册完整性校验未通过（登记表被删/改/无留痕新增，"
              "或锁文件缺失）：判定降级为探索性，不得作为 confirmatory 结论")
    for h in (hyp_eval if isinstance(hyp_eval, list) else []):
        if not isinstance(h, dict):
            continue
        h["confirmatory_valid"] = valid
        h["confirmatory_note"] = reason
    return hyp_eval


# 预注册 scope 核对的配置字段：模型/训练规模（结论可迁移性的前提）。
# lr/weight_decay 刻意排除：按本项目协议，基线调优（--tune_baseline）会在
# 验证集上选定统一超参并注入全部变体，属预注册流程的一部分而非范围外改动。


def registration_scope_check(registered: dict, repro: dict,
                             root: str = ROOT) -> dict:
    """核对“本次 run 是否落在预注册 scope 内”（confirmatory 资格的前提）。

    预注册假设只在登记的实验范围内具有确认性地位：用 tiny 配置、冒烟
    步数、字符伪数据或其它规模跑出的“支持/拒绝”不能冒充预注册结论。
      * smoke 运行 = 明确的范围外；
      * 配置的模型/训练规模字段逐项比对（lr/wd 除外，见上）；
      * scope.data 声明 gpt2 BPE 时核对实际数据分词器（伪数据/换词表会
        改变目标语言建模任务本身，必须显形）；
      * 额外（未登记）种子只列为 warning：种子可交换、不直接推翻判定，
        但偏离登记计划必须显式可见，由读者知情。
    返回 {checked, in_scope, problems, warnings, scope}；无法核对（缺
    repro/config 文件）时 checked=False、in_scope=None，由
    apply_scope_gate 按“不可确认”处理，而不是默认通过。
    """
    registered = _as_dict(registered)
    repro = _as_dict(repro)
    scope = _as_dict(registered.get("scope"))
    out = {"scope": scope, "checked": False, "in_scope": None,
           "problems": [], "warnings": []}
    if not scope:
        out["warnings"].append("登记表未提供 scope 字段，无法核对实验范围")
        return out
    if not repro:
        out["warnings"].append("缺少 repro.json，无法核对实验范围")
        return out

    def _load_json(p):
        if not isinstance(p, (str, os.PathLike)):
            return None
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            return None

    checked_any = False
    unverifiable = []
    if repro.get("smoke") is True:
        out["problems"].append("冒烟运行（--smoke）不在预注册范围（正式配置/全部种子）内")
        checked_any = True

    want_rel = str(scope.get("config") or "")
    want_path = (want_rel if os.path.isabs(want_rel)
                 else os.path.join(root, want_rel))
    got_rel = repro.get("config_file")
    # 与 scope.config 对称：相对路径按项目根解析（否则会相对 cwd 去读另一个
    # 同名文件，悄悄核对错配置）
    got_path = (os.path.join(root, got_rel)
                if isinstance(got_rel, str) and got_rel
                and not os.path.isabs(got_rel) else got_rel)
    want_cfg = _load_json(want_path) if want_path else None
    got_cfg = _load_json(got_path) if got_path else None
    want_cfg = want_cfg if isinstance(want_cfg, dict) else None
    got_cfg = got_cfg if isinstance(got_cfg, dict) else None
    if want_rel:
        if want_cfg is None:
            unverifiable.append("登记 scope.config 不可读")
            out["warnings"].append(
                f"登记 scope.config 不可读（{want_rel or '-'}），无法核对配置规模")
        elif got_cfg is None:
            unverifiable.append("本次配置不可读")
            out["warnings"].append(
                f"本次配置不可读（{got_path or '-'}），无法核对配置规模")
        else:
            diffs = [f"{k}: 登记={want_cfg.get(k)} vs 实际={got_cfg.get(k)}"
                     for k in _SCOPE_CONFIG_KEYS
                     if k in want_cfg and want_cfg.get(k) != got_cfg.get(k)]
            if diffs:
                out["problems"].append(
                    "模型/训练规模与预注册配置不一致（" + "；".join(diffs) + "）")
            checked_any = True

    scope_data = str(scope.get("data") or "")
    actual_tok = ((repro.get("data") or {}).get("tokenizer")
                  if isinstance(repro.get("data"), dict) else None)
    if "gpt2" in scope_data.lower():
        if actual_tok is None:
            unverifiable.append("数据分词器不可读")
            out["warnings"].append(
                "本次数据分词器不可读（repro.json 缺 data.tokenizer），"
                "无法核对 scope.data")
        elif "gpt2" not in str(actual_tok).lower():
            out["problems"].append(
                f"数据分词器与预注册不一致（登记含 gpt2 BPE，实际 {actual_tok}）")
            checked_any = True
        else:
            checked_any = True

    want_seeds = scope.get("seeds")
    got_seeds = repro.get("seeds")
    if isinstance(want_seeds, list):
        if not isinstance(got_seeds, list):
            unverifiable.append("本次种子列表不可读")
            out["warnings"].append(
                "repro.json 缺少可读的 seeds，无法核对预注册种子范围")
        else:
            extra = [s for s in got_seeds if s not in want_seeds]
            if extra:
                out["warnings"].append(
                    f"本次含未登记种子 {extra}：判定基于登记计划外的抽样，证据强度打折")
            checked_any = True
    # 任一已声明范围字段无法核对时，不能因为其它字段通过就默认在范围内；
    # 保持 checked=False / in_scope=None，由 apply_scope_gate 降级。
    out["checked"] = bool(checked_any) and not unverifiable
    if out["problems"]:
        out["in_scope"] = False
    elif out["checked"]:
        out["in_scope"] = True
    return out


def apply_scope_gate(hyp_eval: list, scope_check: dict) -> list:
    """预注册范围核对失败/无法核对时，确认性判定显式降级。

    与完整性门（apply_integrity_gate）同义：登记表没被篡改，但本次 run
    根本不在登记范围内（如 tiny 配置、冒烟步数、伪数据、无法核对），
    此时“支持/拒绝”同样不作数——降级为探索性线索，避免范围外结果
    冒充预注册结论。
    """
    scope_check = _as_dict(scope_check)
    if not scope_check.get("scope"):
        # 登记表未声明 scope（历史/精简登记表）：无从核对，仅由汇总页告警，
        # 不因“没有可核对的字段”本身降级已通过的完整性判定。
        return hyp_eval
    problems = _as_text_list(scope_check.get("problems"))
    # checked 必须是真正的 bool true：字符串 "false"/1 经 truthiness 会被
    # 当成“已核对在范围内”，从而跳过缺 repro/config 时的确认性降级。
    if not problems and scope_check.get("checked") is True:
        return hyp_eval
    reason = ("本次运行超出预注册范围：" + "；".join(problems)
              if problems else
              "无法核对本次运行是否落在预注册范围（缺少 repro/config），"
              "确认性判定不可确认")
    for h in (hyp_eval if isinstance(hyp_eval, list) else []):
        if not isinstance(h, dict):
            continue
        if h.get("confirmatory_valid") is False:
            continue  # 已因完整性失败降级，保留原原因
        h["confirmatory_valid"] = False
        h["confirmatory_note"] = reason
    return hyp_eval


def apply_budget_gate(hyp_eval: list, audit: dict) -> list:
    """早停导致实际训练量不等时，确认性判定显式降级。

    对比实验要求除待验证模块外的训练步数一致；早停按各变体自己的验证曲线
    停在不同步数，实际 tokens_seen 因变体而异。计划步数相同并不能消除
    “有效训练量不同”这一混杂，因此确认性结论降级为探索性，直到用相同
    实际步数重跑。
    """
    audit = _as_dict(audit)
    stopped = _as_text_list(audit.get("early_stop_runs"))
    if not stopped:
        return hyp_eval
    reason = ("早停使实际训练量不一致（" + "；".join(stopped[:4])
              + (" ..." if len(stopped) > 4 else "")
              + "）：确认性判定降级为探索性；请在相同实际步数下重跑")
    for h in (hyp_eval if isinstance(hyp_eval, list) else []):
        if not isinstance(h, dict):
            continue
        if h.get("confirmatory_valid") is False:
            continue
        h["confirmatory_valid"] = False
        h["confirmatory_note"] = reason
    return hyp_eval


def _render_registration_integrity(integrity: dict,
                                   scope_check: dict = None) -> list:
    """预注册完整性小节：删除/静默改写必须在校验中显形。"""
    integrity = _as_dict(integrity)
    st = integrity.get("status")
    if not isinstance(st, str):
        st = None
    icon = {"intact": "✅", "amended": "✅", "violated": "⛔",
            "missing_lock": "⚠", "missing_registry": "⛔"}.get(st, "?")
    desc = {
        "intact": "与锁基线一致（登记后未发生任何修订）",
        "amended": "与修订链一致（既有假设未被改写/删除，"
                   f"链上共 {integrity.get('chain_length')} 环留痕修订）",
        "violated": "完整性校验失败：登记表被删除/改写，或新增假设未留痕",
        "missing_lock": "缺少锁文件，无法验证登记表未被静默改写",
        "missing_registry": "登记表文件缺失，预注册边界不存在",
    }.get(st, "无法判定")
    lines = ["### 预注册完整性（防事后静默改写）", "",
             f"{icon} 状态：`{st}` —— {desc}。"]
    sha = integrity.get("registry_sha256")
    if sha:
        lines.append(f"- 登记表 SHA-256：`{str(sha)[:16]}…`（假设 "
                     f"{integrity.get('n_hypotheses')} 条"
                     + (f"；锁基线 `{str(integrity.get('baseline_document_sha256'))[:16]}…`"
                        if integrity.get("baseline_document_sha256") else "")
                     + "）")
    else:
        lines.append("- 登记表 SHA-256：不可用（文件缺失或不可解析）")
    for p in _as_text_list(integrity.get("problems")):
        lines.append(f"- ⛔ {p}")
    for w in _as_text_list(integrity.get("warnings")):
        lines.append(f"- ⚠ {w}")
    # 预注册范围核对：完整性通过不等于本轮 run 落在登记范围内
    # （tiny 配置/冒烟步数/伪数据跑出的“支持”同样不能冒充确认性结论）。
    sc = _as_dict(scope_check)
    scope = _as_dict(sc.get("scope"))
    scope_failed = False
    if scope or sc:
        # 只有登记表声明了 scope 时才谈得上“范围核对失败”。checked 必须是
        # 真正的 bool true，字符串 "false"/1 不能渲染成“已核对在范围内”。
        scope_failed = bool(scope) and (
            bool(sc.get("problems")) or sc.get("checked") is not True)
        lines.append("")
        lines.append("#### 预注册范围核对（确认性资格的前提）")
        sc_problems = _as_text_list(sc.get("problems"))
        if sc_problems:
            lines.append("- ⛔ 本次运行超出预注册范围：")
            for p in sc_problems:
                lines.append(f"  - {p}")
        elif sc.get("checked") is True:
            cfg_note = (f"模型/训练规模与 `{scope.get('config')}` 一致"
                        if scope.get("config") else "未声明配置范围")
            tok_note = ("；数据分词器与 scope.data 一致"
                        if "gpt2" in str(scope.get("data") or "").lower()
                        else "")
            seeds_note = ("；本次种子均在登记范围内"
                          if isinstance(scope.get("seeds"), list) else "")
            lines.append(f"- ✅ 本次 run 在预注册范围内：{cfg_note}"
                         f"{tok_note}{seeds_note}。")
        else:
            # 无 scope 的登记表不降级（与 apply_scope_gate 的早退一致），
            # 只有“声明了 scope 却核不动”才是不可确认
            lines.append("- ⚠ 无法核对预注册范围（缺少 repro/config 信息）："
                         + ("确认性判定按“不可确认”降级。" if scope else
                            "登记表未声明 scope，完整性判定不受影响。"))
        for w in _as_text_list(sc.get("warnings")):
            lines.append(f"- ⚠ {w}")
    if integrity.get("confirmatory_valid") is True and not scope_failed:
        lines.append("- 修订规则：既有假设只增不改不删；新增假设必须经 "
                     "`python -m lega.hypotheses --amend --reason ...` 留痕，"
                     "否则判违反。")
    else:
        lines.append("- ⚠ **本页全部“注册假设判定”因完整性/范围核对未通过而"
                     "失效（降级为探索性分析）**：排除问题后重跑汇总。")
    lines.append("")
    return lines
