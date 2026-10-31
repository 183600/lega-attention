"""流水线通用工具、单因素审计与协议辅助函数。"""
from ._shared import _install as _shared_install

_shared_install(globals())

def _gate_factor_set(value):
    """把 GATE_FACTORS 的值规范化为因子集合；结构非法时返回 None。

    只接受字符串集合而不是字符串本身：``"ABC"`` 迭代后也是 A/B/C，若不
    排除会被误当成合法因子集合，掩盖“元数据写成了拼接字符串”的漂移。
    """
    if isinstance(value, str) or not isinstance(
            value, (set, frozenset, tuple, list)):
        return None
    try:
        factors = frozenset(value)
    except TypeError:
        return None
    if any(not isinstance(f, str) for f in factors):
        return None
    return factors


_FACTOR_PARAM = {"A": "gate_w", "B": "gate_a", "C": "gate_b"}

_FACTOR_CAPABILITY = {"A": "query", "B": "conc", "C": "len"}

_FACTOR_LOO_VARIANT = {"A": "lega_no_query", "B": "lega_no_conc",
                       "C": "lega_no_len"}


def _factor_cell_set(cell):
    """把全因子格名规范化为因子集合；非法格名返回 None。"""
    if cell == "none":
        return frozenset()
    if (isinstance(cell, str) and cell
            and len(set(cell)) == len(cell)
            and all(ch in "ABC" for ch in cell)):
        return frozenset(cell)
    return None


def _audit_factorial_cells(cells, label, problems):
    """核对 cell -> 变体网格正好覆盖 2^3，并返回变体 -> 因子集合。"""
    mapping = {}
    if not isinstance(cells, dict):
        problems.append(f"{label} 必须是 cell -> 变体映射")
        return mapping
    seen_subsets = set()
    for cell, variant in cells.items():
        factors = _factor_cell_set(cell)
        if factors is None or factors not in _GATE_FACTOR_SUBSETS:
            problems.append(f"{label} 的格名非法：{cell!r}")
            continue
        seen_subsets.add(factors)
        if not isinstance(variant, str):
            problems.append(f"{label} 格 {cell!r} 的变体名非法：{variant!r}")
            continue
        if variant in mapping:
            problems.append(
                f"{label} 的变体 {variant!r} 同时对应多个格")
            continue
        mapping[variant] = factors
    missing = _GATE_FACTOR_SUBSETS - seen_subsets
    extra = seen_subsets - _GATE_FACTOR_SUBSETS
    if missing or extra:
        problems.append(
            f"{label} 未覆盖完整 2^3：缺少 "
            + (", ".join(sorted("".join(sorted(s)) or "none"
                                for s in missing)) or "无")
            + "；多出 "
            + (", ".join(sorted("".join(sorted(s)) or "none"
                                for s in extra)) or "无"))
    return mapping


def _audit_single_factor_edge(label, factor, direction, candidate, reference,
                              factors, problems):
    """核对指定网格中的一条边确实只沿声明方向改变一个因子。"""
    c = factors.get(candidate)
    r = factors.get(reference)
    if c is None or r is None:
        problems.append(
            f"{label} {candidate} vs {reference} 不在对应参数化网格内")
        return False
    diff = frozenset(c.symmetric_difference(r))
    if diff != frozenset({factor}):
        problems.append(
            f"{label} {candidate} vs {reference} 声明为单因素 {factor}，"
            f"实际改变 {sorted(diff) or '无'}")
        return False
    if direction == "remove":
        shape_ok = c < r and (r - c) == frozenset({factor})
    else:
        shape_ok = r < c and (c - r) == frozenset({factor})
    if not shape_ok:
        problems.append(
            f"{label} {candidate} vs {reference} 的 {direction} 方向与"
            f"因子变化不符")
        return False
    return True


def _audit_reparameterization_contrasts(normalized, problems):
    """核对参数化稳健性分析使用的归一化/原始单因子边。

    原始参数化的变体不在 GATE_FACTORS 内，不能复用 changed_gate_factors；
    这里分别按两份 2^3 网格核对。B 的两种参数化表示不同，因此除 A/C 的
    去一法外还必须有 B 的加入/去掉边；A/C 的加入边在两种参数化下是同一
    对变体，无需重复。
    """
    normalized_cells = _audit_factorial_cells(
        globals().get("NORMALIZED_FACTORIAL_CELLS"),
        "NORMALIZED_CELLS", problems)
    raw_cells = _audit_factorial_cells(
        globals().get("RAW_FACTORIAL_CELLS"), "RAW_CELLS", problems)
    for variant, factors in normalized_cells.items():
        if normalized.get(variant) != factors:
            problems.append(
                f"NORMALIZED_CELLS 的变体 {variant!r} 因子集合为 "
                f"{sorted(normalized.get(variant, ()))}，"
                f"应为 {sorted(factors)}")
    if set(normalized_cells) != set(normalized):
        problems.append(
            "NORMALIZED_CELLS 的变体集合与 GATE_FACTORS 不一致："
            f"cells={sorted(normalized_cells)}，"
            f"factors={sorted(normalized)}")
    raw_variants = globals().get("RAW_FACTORIAL_VARIANTS")
    if not isinstance(raw_variants, (tuple, list)):
        problems.append("RAW_FACTORIAL_VARIANTS 必须是变体序列")
    else:
        if len(set(raw_variants)) != len(raw_variants):
            problems.append("RAW_FACTORIAL_VARIANTS 含重复变体")
        if set(raw_variants) != set(raw_cells):
            problems.append(
                "RAW_FACTORIAL_VARIANTS 与 RAW_CELLS 的变体集合不一致")
    # RAW_CELLS 是实验设计元数据；它必须与 raw 变体实现里实际启用的
    # A/B/C 开关一致。否则把 B 格写成归一化 conc 变体（或把 AB 格写成
    # lega_no_len）仍能通过“格名覆盖 2^3”的检查，却不再是原始参数化边。
    raw_gate_factors = globals().get("RAW_GATE_FACTORS")
    raw_impl = {}
    if not isinstance(raw_gate_factors, dict):
        problems.append(
            "RAW_GATE_FACTORS 必须是变体 -> A/B/C 因子集合的映射")
    else:
        for variant, raw_factors in raw_gate_factors.items():
            if not isinstance(variant, str):
                problems.append(
                    f"RAW_GATE_FACTORS 含非字符串变体键：{variant!r}")
                continue
            factors = _gate_factor_set(raw_factors)
            if factors is None:
                problems.append(
                    f"RAW_GATE_FACTORS[{variant!r}] 不是合法的因子集合："
                    f"{raw_factors!r}")
                continue
            unknown = factors - frozenset("ABC")
            if unknown:
                problems.append(
                    f"RAW_GATE_FACTORS[{variant!r}] 含未知因子："
                    f"{sorted(unknown)}")
                continue
            raw_impl[variant] = factors
        if set(raw_impl) != set(raw_cells):
            problems.append(
                "RAW_CELLS 的变体集合与 RAW_GATE_FACTORS 不一致："
                f"cells={sorted(raw_cells)}，"
                f"factors={sorted(raw_impl)}")
        for variant, factors in raw_cells.items():
            impl = raw_impl.get(variant)
            if impl is not None and impl != factors:
                problems.append(
                    f"RAW_CELLS 的变体 {variant!r} 因子集合为 "
                    f"{sorted(factors)}，实现为 {sorted(impl)}")
    contrasts = globals().get("REPARAMETERIZATION_CONTRASTS")
    if (not isinstance(contrasts, (tuple, list))
            or isinstance(contrasts, str)):
        problems.append("REPARAMETERIZATION_CONTRASTS 必须是对比序列")
        return []
    required = {("A", "remove"), ("B", "remove"),
                ("C", "remove"), ("B", "add")}
    seen = set()
    audited = []
    for entry in contrasts:
        if not (isinstance(entry, (tuple, list)) and len(entry) == 5):
            problems.append(f"参数化单因素对比定义结构非法：{entry!r}")
            continue
        factor, direction, norm_pair, raw_pair, _note = entry
        if not all(isinstance(x, str) for x in (factor, direction)):
            problems.append(f"参数化单因素对比含非字符串字段：{entry!r}")
            continue
        if factor not in ("A", "B", "C") or direction not in ("add", "remove"):
            problems.append(
                f"参数化对比存在非 A/B/C 单因素声明：{factor}/{direction}")
            continue
        edge = (factor, direction)
        if edge in seen:
            problems.append(
                f"重复的参数化单因素对比：{factor}/{direction}")
            continue
        seen.add(edge)
        ok = True
        for label, pair, grid in (
                ("归一化参数化", norm_pair, normalized_cells),
                ("原始参数化", raw_pair, raw_cells)):
            if (not isinstance(pair, (tuple, list)) or len(pair) != 2
                    or not all(isinstance(x, str) for x in pair)):
                problems.append(f"{label}对比结构非法：{pair!r}")
                ok = False
                continue
            candidate, reference = pair
            if candidate == reference:
                problems.append(
                    f"{label}单因素对比两侧相同：{candidate} vs {reference}")
                ok = False
                continue
            if not _audit_single_factor_edge(
                    label, factor, direction, candidate, reference,
                    grid, problems):
                ok = False
        sf_pairs = SINGLE_FACTOR_PAIRS if isinstance(
            SINGLE_FACTOR_PAIRS, (tuple, list)) else ()
        normalized_edges = {
            (p[0], p[1]) for p in sf_pairs
            if isinstance(p, (tuple, list)) and len(p) == 2
            and all(isinstance(x, str) for x in p)}
        if (isinstance(norm_pair, (tuple, list)) and len(norm_pair) == 2
                and tuple(norm_pair) not in normalized_edges):
            problems.append(
                f"参数化对比的归一化边 {tuple(norm_pair)!r} 未登记为"
                "单因素对比")
        if ok:
            audited.append({
                "factor": factor, "direction": direction,
                "normalized_candidate": norm_pair[0],
                "normalized_reference": norm_pair[1],
                "raw_candidate": raw_pair[0],
                "raw_reference": raw_pair[1],
            })
    missing = required - seen
    if missing:
        problems.append(
            "缺少原始参数化单因素对比："
            + ", ".join(f"{factor}/{direction}"
                        for factor, direction in sorted(missing)))
    return audited


def _audit_single_factor_pair_table(contrasts, problems):
    """核对派生配对表与已声明的单因素边完全一致。

    ``SINGLE_FACTOR_PAIRS`` 是汇总/报告实际消费的配对集合；它必须由
    ``SINGLE_FACTOR_CONTRASTS`` 逐条派生，不能手工增删。若表里混入
    “B-only vs Full”这类多因子对比，或漏掉一条已审计边，审计必须在
    运行期显形，而不是让组件归因在别处静默失真。
    """
    raw_pairs = globals().get("SINGLE_FACTOR_PAIRS")
    if (not isinstance(raw_pairs, (tuple, list))
            or isinstance(raw_pairs, str)):
        problems.append("SINGLE_FACTOR_PAIRS 必须是对比序列")
        return []
    seen = set()
    pairs = []
    for entry in raw_pairs:
        if (not isinstance(entry, (tuple, list)) or len(entry) != 2
                or not all(isinstance(x, str) and x for x in entry)):
            problems.append(f"单因素对比对结构非法：{entry!r}")
            continue
        candidate, reference = entry
        pair = (candidate, reference)
        if candidate == reference:
            problems.append(
                f"单因素对比对两侧相同：{candidate} vs {reference}")
            continue
        if pair in seen:
            problems.append(
                f"重复的单因素对比对：{candidate} vs {reference}")
            continue
        seen.add(pair)
        if single_factor_change(candidate, reference) is None:
            problems.append(
                f"单因素对比对 {candidate} vs {reference} 实际不改变"
                "恰好一个 A/B/C 因子")
            continue
        pairs.append(pair)
    expected = {(c["candidate"], c["reference"]) for c in contrasts}
    missing = expected - seen
    extra = seen - expected
    if missing:
        problems.append(
            "SINGLE_FACTOR_PAIRS 缺少已声明对比："
            + ", ".join(f"{c} vs {r}" for c, r in sorted(missing)))
    if extra:
        problems.append(
            "SINGLE_FACTOR_PAIRS 含未声明对比："
            + ", ".join(f"{c} vs {r}" for c, r in sorted(extra)))
    return pairs


def _audit_inference_permutation_lesions(problems):
    """核对 A/B/C 的分布匹配置换病灶各自只干预一个证据通道。

    ``query_permute/conc_permute/len_permute`` 是推理期安慰剂：它们不修改
    参数，只把对应证据沿序列维错位。若映射把某个因子指到另一个通道，
    或让同一病灶承担多个因子，报告就会把“位置证据的因果作用”归错组件，
    因此这里按病灶 kind / requires / params 逐项核对。
    """
    entries = globals().get("PERMUTATION_LESION_FACTORS")
    if (not isinstance(entries, (tuple, list))
            or isinstance(entries, str)):
        problems.append("PERMUTATION_LESION_FACTORS 必须是病灶映射序列")
        return []
    lesion_by_id = globals().get("ABLATION_LESION_BY_ID")
    if not isinstance(lesion_by_id, dict):
        problems.append("LESION_BY_ID 必须是病灶映射")
        return []
    expected = {
        "A": ("query_permute", "query"),
        "B": ("conc_permute", "conc"),
        "C": ("len_permute", "len"),
    }
    seen_factors = set()
    seen_ids = set()
    audited = []
    for entry in entries:
        if (not isinstance(entry, (tuple, list)) or len(entry) != 3
                or not all(isinstance(x, str) for x in entry)):
            problems.append(f"单因素置换病灶定义结构非法：{entry!r}")
            continue
        factor, lesion_id, capability = entry
        if factor not in expected:
            problems.append(
                f"单因素置换病灶的因子非法：{factor!r}")
            continue
        if factor in seen_factors:
            problems.append(f"重复的单因素置换因子：{factor}")
        seen_factors.add(factor)
        if lesion_id in seen_ids:
            problems.append(
                f"置换病灶 {lesion_id!r} 承担多个单因子映射")
        seen_ids.add(lesion_id)
        expected_kind, expected_capability = expected[factor]
        if capability != expected_capability:
            problems.append(
                f"置换病灶 {lesion_id!r} 声明为 {factor}，但能力为 "
                f"{capability!r}，应为 {expected_capability!r}")
        spec = lesion_by_id.get(lesion_id)
        if not isinstance(spec, dict):
            problems.append(f"置换病灶 {lesion_id!r} 未定义")
            continue
        if spec.get("kind") != expected_kind:
            problems.append(
                f"置换病灶 {lesion_id!r} 声明为单因素 {factor}，kind 应为 "
                f"{expected_kind!r}，实际 {spec.get('kind')!r}")
        requires = spec.get("requires")
        if (not isinstance(requires, (tuple, list, set, frozenset))
                or not all(isinstance(x, str) for x in requires)):
            problems.append(f"置换病灶 {lesion_id!r} 的 requires 结构非法")
        elif set(requires) != {expected_capability}:
            problems.append(
                f"置换病灶 {lesion_id!r} 依赖 {sorted(set(requires)) or '无'}，"
                f"单因素 {factor} 应只依赖 {expected_capability}")
        params = spec.get("params")
        if (not isinstance(params, (tuple, list, set, frozenset))
                or not all(isinstance(x, str) for x in params)):
            problems.append(f"置换病灶 {lesion_id!r} 的 params 结构非法")
        elif params:
            problems.append(
                f"置换病灶 {lesion_id!r} 不应清零参数，实际为 "
                f"{sorted(set(params))}")
        if spec.get("factorial_cell") is True:
            problems.append(
                f"置换病灶 {lesion_id!r} 不应标记为全因子格")
        audited.append({"factor": factor, "lesion": lesion_id,
                        "capability": capability,
                        "kind": expected_kind})
    missing = set(expected) - seen_factors
    if missing:
        problems.append(
            "缺少单因素置换病灶："
            + ", ".join(sorted(missing)))
    return audited


def _audit_inference_single_factor_design(problems):
    """核对推理期单病灶/活动匹配/2^3 格仍然只对应已声明的单因子。"""
    lesion_by_id = globals().get("ABLATION_LESION_BY_ID")
    if not isinstance(lesion_by_id, dict):
        problems.append("LESION_BY_ID 必须是病灶映射")
        return [], [], []
    pairs = globals().get("MATCHED_LESION_PAIRS")
    if (not isinstance(pairs, (tuple, list)) or isinstance(pairs, str)):
        problems.append("MATCHED_LESION_PAIRS 必须是配对序列")
        pairs = ()
    matched = []
    seen_factors = set()
    seen_ids = set()
    for entry in pairs:
        if (not isinstance(entry, (tuple, list)) or len(entry) != 3
                or not all(isinstance(x, str) for x in entry)):
            problems.append(f"活动匹配病灶配对结构非法：{entry!r}")
            continue
        raw_id, matched_id, factor = entry
        if factor not in ("A", "B", "C"):
            problems.append(
                f"活动匹配病灶 {raw_id} vs {matched_id} 的因子非法："
                f"{factor!r}")
            continue
        if factor in seen_factors:
            problems.append(f"重复的活动匹配因子：{factor}")
        seen_factors.add(factor)
        if raw_id == matched_id:
            problems.append(f"活动匹配病灶两侧相同：{raw_id}")
            continue
        raw_spec = lesion_by_id.get(raw_id)
        matched_spec = lesion_by_id.get(matched_id)
        if not isinstance(raw_spec, dict):
            problems.append(f"原始病灶 {raw_id!r} 未定义")
            continue
        if not isinstance(matched_spec, dict):
            problems.append(f"活动匹配病灶 {matched_id!r} 未定义")
            continue
        expected_params = {_FACTOR_PARAM[factor]}
        expected_capability = {"A": "query", "B": "conc",
                               "C": "len"}[factor]
        for label, lid, spec in (
                ("原始病灶", raw_id, raw_spec),
                ("活动匹配病灶", matched_id, matched_spec)):
            if lid in seen_ids:
                problems.append(f"病灶 {lid!r} 在多条匹配对照中重复使用")
            seen_ids.add(lid)
            params = spec.get("params")
            if (not isinstance(params, (tuple, list, set, frozenset))
                    or not all(isinstance(x, str) for x in params)):
                problems.append(f"{label} {lid!r} 的 params 结构非法")
                continue
            if set(params) != expected_params:
                problems.append(
                    f"{label} {lid!r} 声明为单因素 {factor}，实际清零 "
                    f"{sorted(set(params)) or '无'}")
            requires = spec.get("requires")
            if (not isinstance(requires, (tuple, list, set, frozenset))
                    or not all(isinstance(x, str) for x in requires)):
                problems.append(f"{label} {lid!r} 的 requires 结构非法")
            elif set(requires) != {expected_capability}:
                problems.append(
                    f"{label} {lid!r} 声明为单因素 {factor}，但依赖 "
                    f"{sorted(set(requires)) or '无'}，应为 "
                    f"{expected_capability}")
        if raw_spec.get("kind") != "zero":
            problems.append(f"原始病灶 {raw_id!r} 不是单因子清零病灶")
        if matched_spec.get("kind") != "zero_matched":
            problems.append(
                f"活动匹配病灶 {matched_id!r} 不是 zero_matched 病灶")
        if raw_spec.get("matched_control") is True:
            problems.append(f"原始病灶 {raw_id!r} 不应标为活动匹配对照")
        if matched_spec.get("matched_control") is not True:
            problems.append(
                f"活动匹配病灶 {matched_id!r} 未标记 matched_control")
        if matched_spec.get("matched_of") != raw_id:
            problems.append(
                f"活动匹配病灶 {matched_id!r} 的 matched_of 不是 "
                f"{raw_id!r}")
        if matched_spec.get("matched_factor") != factor:
            problems.append(
                f"活动匹配病灶 {matched_id!r} 的 matched_factor 与 "
                f"{factor} 不一致")
        matched.append({"raw": raw_id, "matched": matched_id,
                        "factor": factor})
    missing_factors = {"A", "B", "C"} - seen_factors
    if missing_factors:
        problems.append(
            "缺少活动匹配单因素病灶："
            + ", ".join(sorted(missing_factors)))
    permutation_lesions = _audit_inference_permutation_lesions(
        problems)
    cells = globals().get("INFERENCE_FACTORIAL_CELLS")
    if not isinstance(cells, dict):
        problems.append("INFERENCE_FACTORIAL_CELLS 必须是 cell -> 病灶映射")
        return matched, permutation_lesions, []
    audited_cells = []
    seen_subsets = set()
    seen_lesions = set()
    for cell, lesion_id in cells.items():
        factors = _factor_cell_set(cell)
        if factors is None or factors not in _GATE_FACTOR_SUBSETS:
            problems.append(f"推理期全因子格的格名非法：{cell!r}")
            continue
        seen_subsets.add(factors)
        if factors == frozenset("ABC"):
            if lesion_id is not None:
                problems.append(
                    f"推理期基线格 {cell!r} 不应映射病灶：{lesion_id!r}")
            else:
                audited_cells.append({"cell": cell, "lesion": None,
                                      "factors": sorted(factors)})
            continue
        if not isinstance(lesion_id, str):
            problems.append(
                f"推理期全因子格 {cell!r} 的病灶名非法：{lesion_id!r}")
            continue
        if lesion_id in seen_lesions:
            problems.append(
                f"推理期病灶 {lesion_id!r} 同时对应多个全因子格")
        seen_lesions.add(lesion_id)
        spec = lesion_by_id.get(lesion_id)
        if not isinstance(spec, dict):
            problems.append(f"推理期全因子格病灶 {lesion_id!r} 未定义")
            continue
        params = spec.get("params")
        if (not isinstance(params, (tuple, list, set, frozenset))
                or not all(isinstance(x, str) for x in params)):
            problems.append(
                f"推理期病灶 {lesion_id!r} 的 params 结构非法")
            continue
        expected_zero = frozenset("ABC") - factors
        expected_params = {_FACTOR_PARAM[f] for f in expected_zero}
        if set(params) != expected_params:
            problems.append(
                f"推理期格 {cell!r} -> {lesion_id!r} 应清零 "
                f"{sorted(expected_params) or '无'}，实际清零 "
                f"{sorted(set(params)) or '无'}")
        expected_kind = "zero_evidence" if not factors else "zero"
        if spec.get("kind") != expected_kind:
            problems.append(
                f"推理期格 {cell!r} -> {lesion_id!r} 的 kind 不是 "
                f"{expected_kind}")
        flagged = spec.get("factorial_cell") is True
        if len(factors) == 1 and not flagged:
            problems.append(
                f"推理期单因子格 {cell!r} 的病灶 {lesion_id!r} 未标记 "
                "factorial_cell，可能混入单病灶因果表")
        if len(factors) != 1 and flagged:
            problems.append(
                f"推理期非单因子格 {cell!r} 的病灶 {lesion_id!r} 不应标记 "
                "factorial_cell")
        audited_cells.append({"cell": cell, "lesion": lesion_id,
                              "factors": sorted(factors)})
    missing = _GATE_FACTOR_SUBSETS - seen_subsets
    extra = seen_subsets - _GATE_FACTOR_SUBSETS
    if missing or extra:
        problems.append(
            "推理期全因子格未覆盖完整 2^3：缺少 "
            + (", ".join(sorted("".join(sorted(s)) or "none"
                                for s in missing)) or "无")
            + "；多出 "
            + (", ".join(sorted("".join(sorted(s)) or "none"
                                for s in extra)) or "无"))
    return matched, permutation_lesions, audited_cells


def _audit_dose_response_components(problems):
    """核对剂量-反应每个组件只缩放一个已登记的单因子参数。

    ``lega.dose_response`` 在同一 checkpoint、同一批 val 窗口上逐组件把
    A（gate_w）、B（gate_a）、C（gate_b）系数乘以 λ。若组件规格把某个
    组件映射到错误的 gate 参数/门控能力，或两个组件共用同一参数，扫描
    就不再是“只改一个因子”的剂量实验，而会变成未声明的联合干预。这里
    同时要求 ``COMPONENT_ORDER``/``COMPONENT_SPECS`` 恰好覆盖 A/B/C，
    且训练期对照仍是该因子的去一法变体。
    """
    specs = globals().get("DOSE_COMPONENT_SPECS")
    order = globals().get("DOSE_COMPONENT_ORDER")
    expected = ("A", "B", "C")
    if not isinstance(specs, dict):
        problems.append("COMPONENT_SPECS 必须是组件 -> 规格映射")
        specs = {}
    if (not isinstance(order, (tuple, list))
            or isinstance(order, str)):
        problems.append("COMPONENT_ORDER 必须是组件序列")
        order_entries = []
    else:
        order_entries = list(order)
    seen_order = set()
    for component in order_entries:
        if not isinstance(component, str) or component not in expected:
            problems.append(f"剂量-反应组件名非法：{component!r}")
            continue
        if component in seen_order:
            problems.append(f"重复的剂量-反应组件：{component}")
        seen_order.add(component)
    missing_order = set(expected) - seen_order
    if missing_order:
        problems.append(
            "缺少剂量-反应组件：" + ", ".join(sorted(missing_order)))
    for component in specs:
        if not isinstance(component, str):
            problems.append(f"剂量-反应规格含非字符串组件键：{component!r}")
        elif component not in expected:
            problems.append(f"存在未登记的剂量-反应组件：{component!r}")
    fields = (
        ("param", _FACTOR_PARAM, "缩放参数"),
        ("capability", _FACTOR_CAPABILITY, "门控能力"),
        ("training_variant", _FACTOR_LOO_VARIANT, "训练期对照"),
    )
    seen_values = {name: {} for name, _mapping, _label in fields}
    audited = []
    for component in expected:
        spec = specs.get(component)
        if not isinstance(spec, dict):
            problems.append(f"缺少剂量-反应组件规格：{component}")
            continue
        row = {"component": component}
        for name, expected_map, label in fields:
            value = spec.get(name)
            if not isinstance(value, str) or not value:
                problems.append(
                    f"剂量-反应组件 {component} 的{label}非法：{value!r}")
                continue
            row[name] = value
            wanted = expected_map[component]
            if value != wanted:
                problems.append(
                    f"剂量-反应组件 {component} 的{label}为 {value!r}，"
                    f"应为 {wanted!r}，否则不再是单因素剂量实验")
            prior = seen_values[name].get(value)
            if prior is not None and prior != component:
                problems.append(
                    f"剂量-反应组件 {component} 与 {prior} 共用{label} "
                    f"{value!r}")
            else:
                seen_values[name][value] = component
        audited.append(row)
    return audited


def changed_gate_factors(candidate, reference):
    """两个门控变体之间激活因子集合的对称差；任一方不在门控内返回 None。

    单因素对比要求对称差恰好一个因子。standard/null 不属于门控内 2^3，
    与它们的对比（如 LEGA vs standard）是整体方法对比，不参与组件归因。
    外部/旧产物可能传入不可哈希键或损坏的因子元数据，此时按“不在门控内”
    处理，而不是让汇总因一次查询崩溃。
    """
    if not isinstance(GATE_FACTORS, dict):
        return None
    try:
        c = _gate_factor_set(GATE_FACTORS.get(candidate))
        r = _gate_factor_set(GATE_FACTORS.get(reference))
    except TypeError:  # candidate/reference 不可哈希
        return None
    if c is None or r is None:
        return None
    known = frozenset("ABC")
    if (c - known) or (r - known):
        return None
    return frozenset(c.symmetric_difference(r))


def single_factor_change(candidate, reference):
    """候选−参照恰好只改一个因子时返回该因子（A/B/C），否则返回 None。"""
    diff = changed_gate_factors(candidate, reference)
    if diff is not None and len(diff) == 1:
        factor = next(iter(diff))
        if factor in ("A", "B", "C"):
            return factor
    return None


def _audit_alt_conc_variants(problems):
    """核对替代集中度统计是 B-only 构念对照，且不混入结构单因素表。

    lega_l2/lega_entropy 与 lega 共享 A（查询）和 C（长度）以及全部参数
    形状，只把 B 的统计量 max_j p_j 换成 L2 质量或熵质量。它们是构念效度
    对照，不是主 2^3 矩阵的另一个 B 格：若误入 GATE_FACTORS，全因子/主效应
    会把“换统计量”和“开关 B”混为一谈；若漏掉 A/C/门控开关，又会静默退化成
    标准注意力。这里同时核对两侧，避免元数据与实现漂移。
    """
    rows = []
    alt = globals().get("ALT_CONC_VARIANTS")
    modes = globals().get("ALT_CONC_STATS")
    if (not isinstance(alt, (tuple, list)) or isinstance(alt, str)):
        problems.append("ALT_CONC_VARIANTS 必须是变体序列")
        return rows
    if not isinstance(modes, dict):
        problems.append("ALT_CONC_STATS 必须是变体 -> 统计量映射")
        return rows
    if len(set(alt)) != len(alt):
        problems.append("ALT_CONC_VARIANTS 含重复变体")
    if set(modes) != set(alt):
        problems.append(
            "ALT_CONC_STATS 与 ALT_CONC_VARIANTS 不一致："
            f"stats={sorted(modes)}，variants={sorted(set(alt))}")
    for variant in alt:
        if not isinstance(variant, str):
            problems.append(f"ALT_CONC_VARIANTS 含非字符串变体：{variant!r}")
            continue
        mode = modes.get(variant)
        if mode not in ("l2", "entropy"):
            problems.append(
                f"替代集中度 {variant!r} 的统计量 {mode!r} 未实现")
            continue
        if variant in CONTROL_VARIANTS:
            problems.append(f"替代集中度 {variant!r} 不能作为参数安慰剂")
        if variant not in RUNNABLE_VARIANTS:
            problems.append(f"替代集中度 {variant!r} 不可训练/评测")
        if variant not in SUPPLEMENTAL_VARIANTS:
            problems.append(f"替代集中度 {variant!r} 未登记为补充对照")
        if variant in GATE_FACTORS:
            problems.append(
                f"替代集中度 {variant!r} 混入了主 A/B/C 因子表；"
                "它只替换 B 的统计量，不能当作另一个结构格")
        if variant in RAW_GATE_FACTORS:
            problems.append(
                f"替代集中度 {variant!r} 混入了原始参数化因子表")
        if variant not in _USE_QUERY or variant not in _USE_CONC \
                or variant not in _USE_LEN:
            problems.append(
                f"替代集中度 {variant!r} 未保持 lega 的 A/B/C 门控结构")
        if variant not in _GATED or variant not in _LEGA_FAMILY:
            problems.append(
                f"替代集中度 {variant!r} 未启用输出门或注意力统计")
        if variant in _USE_RAW_CONC:
            problems.append(
                f"替代集中度 {variant!r} 同时启用了原始 B 参数化")
        rows.append({"variant": variant, "mode": mode,
                     "replaced_factor": "B", "reference": "lega"})
    return rows


def single_factor_audit():
    """核对“需要单因素的消融对比”确实只改变一个门控因子。

    这是运行期自检：GATE_FACTORS 从变体实现派生，SINGLE_FACTOR_CONTRASTS
    是汇总与组件判定的唯一单因素对比表。任何一侧漂移（新增变体、改公式、
    误把 B-only/C-only 与 Full 配对）都会在这里显形，而不是被静默当成
    单组件证据。返回值写入 significance.json 并渲染进汇总页。

    审计刻意不假设输入结构永远正确：GATE_FACTORS/对比表/COMPONENT_TESTS
    都可能被旧版本、手工编辑或上游改动破坏。结构损坏要变成可读的
    ``problems``，而不是让整份汇总在解包/排序时崩溃。
    """
    problems = []
    contrasts = []
    gate_table = GATE_FACTORS if isinstance(GATE_FACTORS, dict) else {}
    if not isinstance(GATE_FACTORS, dict):
        problems.append("GATE_FACTORS 必须是变体 -> A/B/C 因子集合的映射")
    normalized = {}
    for variant, raw_factors in gate_table.items():
        if not isinstance(variant, str):
            problems.append(f"GATE_FACTORS 含非字符串变体键：{variant!r}")
            continue
        factors = _gate_factor_set(raw_factors)
        if factors is None:
            problems.append(
                f"GATE_FACTORS[{variant!r}] 不是合法的因子集合："
                f"{raw_factors!r}")
            continue
        unknown = factors - frozenset("ABC")
        if unknown:
            problems.append(
                f"GATE_FACTORS[{variant!r}] 含未知因子：{sorted(unknown)}")
            continue
        normalized[variant] = factors
    # 2^3 设计必须真的覆盖八个不同因子集合（含空集），而不只是“有 8 个
    # 变体名”。重复集合会让某个组件格实际上不存在，交互/主效应随之失真。
    actual_subsets = set(normalized.values())
    missing_subsets = _GATE_FACTOR_SUBSETS - actual_subsets
    extra_subsets = actual_subsets - _GATE_FACTOR_SUBSETS
    if missing_subsets or extra_subsets:
        problems.append(
            "GATE_FACTORS 未实现完整 2^3 因子设计：缺少 "
            + (", ".join(sorted("".join(sorted(s)) or "none"
                                for s in missing_subsets)) or "无")
            + "；多出 "
            + (", ".join(sorted("".join(sorted(s)) or "none"
                                for s in extra_subsets)) or "无"))
    # 全因子矩阵的变体集合必须与 GATE_FACTORS 完全一致；否则汇总会拿
    # 未登记因子语义的变体当 2^3 的一格。
    factorial_variants = []
    if not isinstance(FACTORIAL_CELLS, (tuple, list)):
        problems.append("FACTORIAL_CELLS 必须是 (名称, 变体) 序列")
    else:
        for entry in FACTORIAL_CELLS:
            if (not isinstance(entry, (tuple, list)) or len(entry) != 2
                    or not isinstance(entry[1], str)):
                problems.append(f"全因子格定义结构非法：{entry!r}")
                continue
            factorial_variants.append(entry[1])
    if (len(set(factorial_variants)) != 8
            or set(factorial_variants) != set(normalized)):
        problems.append(
            "全因子 8 格与 GATE_FACTORS 不一致："
            f"cells={sorted(set(factorial_variants))}，"
            f"factors={sorted(normalized)}")
    # 交互检验使用的 cell -> 变体映射是另一份实验设计元数据，必须与
    # GATE_FACTORS 的因子定义逐格对齐（例如 AB 格必须真的启用 A+B）。
    cell_map = globals().get("_FACTORIAL_CONTRAST_CELLS")
    if cell_map is not None:
        if not isinstance(cell_map, dict):
            problems.append("_FACTORIAL_CONTRAST_CELLS 必须是映射")
        else:
            cell_variants = set()
            for cell, variant in cell_map.items():
                if cell == "none":
                    cell_factors = frozenset()
                elif (isinstance(cell, str) and cell
                        and len(set(cell)) == len(cell)
                        and all(ch in "ABC" for ch in cell)):
                    cell_factors = frozenset(cell)
                else:
                    cell_factors = None
                if (cell_factors is None
                        or cell_factors not in _GATE_FACTOR_SUBSETS):
                    problems.append(f"全因子格名非法：{cell!r}")
                    continue
                if not isinstance(variant, str):
                    problems.append(
                        f"全因子格 {cell!r} 的变体名非法：{variant!r}")
                    continue
                cell_variants.add(variant)
                if normalized.get(variant) != cell_factors:
                    problems.append(
                        f"全因子格 {cell!r} -> {variant!r} 的因子集合为 "
                        f"{sorted(normalized.get(variant, ()))}，"
                        f"应为 {sorted(cell_factors)}")
            if cell_variants != set(normalized):
                problems.append(
                    "交互检验的全因子变体集合与 GATE_FACTORS 不一致："
                    f"cells={sorted(cell_variants)}，"
                    f"factors={sorted(normalized)}")
    # 审计表必须恰好覆盖 A/B/C 的“加入/去掉”六条单因素边。只逐条校验
    # 已列出的对比会漏掉“整条对比被删除/重复”这种漂移：表少一行时仍
    # 全部合法，却不再覆盖某个组件的归因。
    required = {(factor, direction) for factor in ("A", "B", "C")
                for direction in ("add", "remove")}
    seen = set()
    seen_pairs = {}
    if (not isinstance(SINGLE_FACTOR_CONTRASTS, (tuple, list))
            or isinstance(SINGLE_FACTOR_CONTRASTS, str)):
        problems.append("SINGLE_FACTOR_CONTRASTS 必须是对比序列")
        contrast_entries = []
    else:
        contrast_entries = SINGLE_FACTOR_CONTRASTS
    for entry in contrast_entries:
        if not (isinstance(entry, (tuple, list)) and len(entry) == 4):
            problems.append(f"单因素对比定义结构非法：{entry!r}")
            continue
        factor, direction, cand, ref = entry
        if not all(isinstance(x, str) for x in (factor, direction, cand, ref)):
            problems.append(f"单因素对比定义含非字符串字段：{entry!r}")
            continue
        edge = (factor, direction)
        if factor not in ("A", "B", "C") or direction not in ("add", "remove"):
            problems.append(f"存在非 A/B/C 单因素声明：{factor}/{direction}")
            continue
        if edge in seen:
            problems.append(
                f"重复的单因素对比声明：{factor}/{direction}")
        seen.add(edge)
        if cand == ref:
            problems.append(f"单因素对比两侧相同：{cand} vs {ref}")
            continue
        if (cand, ref) in seen_pairs:
            problems.append(
                f"重复的单因素对比对：{cand} vs {ref}"
                f"（已声明为 {seen_pairs[(cand, ref)][0]}/"
                f"{seen_pairs[(cand, ref)][1]}）")
            continue
        seen_pairs[(cand, ref)] = (factor, direction)
        c = normalized.get(cand)
        r = normalized.get(ref)
        if c is None or r is None:
            problems.append(f"{cand} vs {ref} 不在门控变体内，无法做单因素归因")
            continue
        diff = frozenset(c.symmetric_difference(r))
        if diff != frozenset({factor}):
            problems.append(
                f"{cand} vs {ref} 声明为单因素 {factor}，实际改变 "
                f"{sorted(diff) or '无'}")
            continue
        # 方向不只是展示字段：remove 必须从参照中恰好去掉该因子，add 必须
        # 在参照上恰好加入该因子。写反方向会让“加入/去掉”的效应解释颠倒。
        if direction == "remove":
            direction_ok = (c < r and (r - c) == frozenset({factor}))
            expected_shape = f"{ref} 去掉 {factor} 得到 {cand}"
        else:
            direction_ok = (r < c and (c - r) == frozenset({factor}))
            expected_shape = f"{ref} 加入 {factor} 得到 {cand}"
        if not direction_ok:
            problems.append(
                f"{cand} vs {ref} 声明方向 {direction} 与因子变化不符："
                f"应为 {expected_shape}")
            continue
        contrasts.append({"factor": factor, "direction": direction,
                          "candidate": cand, "reference": ref,
                          "changed_factors": sorted(diff)})
    missing = required - seen
    extra = seen - required
    if missing:
        problems.append(
            "缺少单因素对比："
            + ", ".join(f"{factor}/{direction}"
                        for factor, direction in sorted(missing)))
    if extra:
        problems.append(
            "存在非 A/B/C 单因素声明的对比："
            + ", ".join(f"{factor}/{direction}"
                        for factor, direction in sorted(extra)))
    # 汇总/报告实际消费的配对表必须与已声明的 6 条边逐项一致；否则
    # 手工增删 SINGLE_FACTOR_PAIRS 就能让多因子对比混进汇总配对表。
    single_factor_pairs = _audit_single_factor_pair_table(
        contrasts, problems)
    audited_pairs = set(single_factor_pairs)
    # 组件判定实际消费的检验对也必须落在已审计表内；否则改 COMPONENT_TESTS
    # 就能绕开审计去使用多因子对比（运行期而非仅测试期拦截）。
    audited_edges = {(c["candidate"], c["reference"]):
                     (c["factor"], c["direction"]) for c in contrasts}
    component_tests = globals().get("COMPONENT_TESTS")
    if not isinstance(component_tests, (tuple, list)):
        problems.append("COMPONENT_TESTS 必须是组件检验序列")
        component_tests = []
    seen_components = set()
    for entry in component_tests:
        if not (isinstance(entry, (tuple, list)) and len(entry) == 4):
            problems.append(f"组件检验定义结构非法：{entry!r}")
            continue
        name, loo_pair, add_pair, primary = entry
        token = str(name)[:1] if isinstance(name, str) and name else None
        if token not in ("A", "B", "C"):
            problems.append(
                f"组件名 {name!r} 无法映射到 A/B/C，不能作为单组件检验")
            expected = None
        else:
            expected = token
            if expected in seen_components:
                problems.append(f"组件 {expected} 存在重复的检验定义")
            seen_components.add(expected)
        if primary not in _EVIDENCE_METRICS:
            problems.append(
                f"组件 {name!r} 的主指标 {primary!r} 不是已登记指标")
        for label, pair, expected_direction in (
                ("去一法", loo_pair, "remove"),
                ("全因子", add_pair, "add")):
            if (not isinstance(pair, (tuple, list)) or len(pair) != 2
                    or not all(isinstance(x, str) for x in pair)):
                problems.append(
                    f"组件 {name!r} 的{label}对比结构非法：{pair!r}")
                continue
            if tuple(pair) not in audited_pairs:
                problems.append(
                    f"组件 {name!r} 的{label}对比 {pair!r} 不在"
                    " SINGLE_FACTOR_PAIRS 中")
            edge = audited_edges.get(tuple(pair))
            if edge is None:
                problems.append(
                    f"组件 {name!r} 使用了未审计的单因素对比：{pair!r}")
                continue
            factor, direction = edge
            if expected is not None and factor != expected:
                problems.append(
                    f"组件 {name!r} 的对比 {pair!r} 实际单因素为 {factor}，"
                    f"与组件 {expected} 不一致")
            if direction != expected_direction:
                problems.append(
                    f"组件 {name!r} 的{label}对比 {pair!r} 方向为 {direction}，"
                    f"应为 {expected_direction}")
    missing_components = {"A", "B", "C"} - seen_components
    if missing_components:
        problems.append(
            "缺少组件检验：" + ", ".join(sorted(missing_components)))
    reparameterization_contrasts = _audit_reparameterization_contrasts(
        normalized, problems)
    dose_components = _audit_dose_response_components(problems)
    matched_lesions, permutation_lesions, inference_cells = (
        _audit_inference_single_factor_design(problems))
    alt_conc = _audit_alt_conc_variants(problems)
    return {"ok": not problems, "problems": problems,
            "contrasts": contrasts,
            "pairs": [list(p) for p in single_factor_pairs],
            "factors": {v: sorted(f) for v, f in sorted(normalized.items())},
            "reparameterization_contrasts": reparameterization_contrasts,
            "dose_response_components": dose_components,
            "inference_matched_lesions": matched_lesions,
            "inference_permutation_lesions": permutation_lesions,
            "inference_factorial_cells": inference_cells,
            "alt_conc_contrasts": alt_conc,
            "note": ("单因素对比 = 去一法 Full−X vs Full，或全因子增量 "
                     "X-only vs gate_only；整体对比（vs standard、LEGA vs "
                     "qgate）可跨多个因子，不作组件归因。同一审计还核对 "
                     "归一化/原始参数化的单因子边、剂量-反应的 A/B/C 参数"
                     "映射、推理期单病灶、活动匹配、A/B/C 分布匹配置换、"
                     "2^3 格的因子映射，以及替代集中度统计的 B-only 构念"
                     "对照（lega_l2/lega_entropy 不进入主因子表）。")}


def build_comparison_pairs(runs, registered=None):
    """构造汇总用的同种子配对对比表。

    刻意把“单因素消融”和“整体方法对比”分开：
      * 单因素：SINGLE_FACTOR_PAIRS（去一法 3 组 + 全因子增量 3 组）；
      * 整体：各变体 vs standard，以及预注册的其它整体对比（如 H5）；
      * 不再把 lega_conc_only / lega_len_only 与 lega 相比——它们与 Full
        同时差两个因子（A+C / A+B），不是单组件消融。
    """
    runs = runs if isinstance(runs, dict) else {}
    pairs = [(cand, "standard") for cand in ALL_VARIANTS
             if cand != "standard"]
    # 补充对照（raw/detach/null_zero）不在默认矩阵内，但显式运行时同样是
    # “相对 standard”的整体对比；漏掉会让汇总表出现数值却没有任何配对
    # 检验，也与设计层 planned_comparison_pairs（按实际计划变体生成）不一致。
    pairs += [(cand, "standard") for cand in SUPPLEMENTAL_VARIANTS
              if cand in runs]
    # 参数匹配安慰剂只在显式运行时才进入对比，且核心对比是 lega vs placebo
    # （不是 vs standard）：二者门控参数/FLOPs 相同，差别只在证据是否有信息。
    if PARAM_CONTROL_VARIANT in runs:
        pairs.append((PARAM_CONTROL_VARIANT, "standard"))
        if "lega" in runs:
            pairs.append(PARAM_CONTROL_PAIR)
        # 安慰剂 vs qgate：若噪声特征本身与 qgate 等价/更好，说明“多出的
        # 门控容量”单独不伤害优化；若明显更差，则噪声特征可能有害，
        # lega vs placebo 的差异不能单独归因于证据（汇总页会并列提示）。
        if "qgate" in runs:
            pairs.append((PARAM_CONTROL_VARIANT, "qgate"))
    # gate_only 未运行时，三个“只加 X”的全因子增量对无法估计；去一法对
    # 仍保留（缺 lega 时下游会如实报“未运行”，而不是伪造证据）。
    pairs += [(cand, ref) for cand, ref in SINGLE_FACTOR_PAIRS
              if ref != "gate_only" or "gate_only" in runs]
    registered = _as_dict(registered)
    hypotheses = registered.get("hypotheses")
    for h in (hypotheses if isinstance(hypotheses, list) else []):
        if not isinstance(h, dict):
            continue
        c, r = h.get("candidate"), h.get("reference")
        if h.get("metric") in (AUROC_DIFF_METRIC, AP_DIFF_METRIC):
            continue  # 单样本机制族，由 _auroc_diff_tests/_ap_diff_tests 覆盖
        if c in RUNNABLE_VARIANTS and r in RUNNABLE_VARIANTS:
            pairs.append((c, r))
    return list(dict.fromkeys(pairs))


def _ablation_coverage(cells, present, rows_by_var):
    """按“变体存在且种子数≥MIN_SEEDS”判定消融单元格覆盖。

    只检查变体名是否出现会把单种子冒烟格当成完整覆盖；单种子既不能做
    配对检验，也无法区分方法效应与初始化噪声，必须与“未运行”分开报告。
    """
    done, missing, underpowered = [], [], []
    for name, v in cells:
        row = rows_by_var.get(v) if isinstance(rows_by_var, dict) else None
        n_seeds = _integral_value(row.get("n_seeds")) if isinstance(row, dict) else None
        if n_seeds is None or n_seeds < 0:
            n_seeds = 0
        if v not in present:
            missing.append((name, v))
        elif n_seeds < MIN_SEEDS:
            missing.append((name, v))
            underpowered.append((name, v, n_seeds))
        else:
            done.append((name, v))
    return done, missing, underpowered


# 受控变量核对时允许逐 run 不同的字段（方法与复现标识，不属于受控变量）

# 显著性协议：同一指标家族内 Holm 校正后 p<ALPHA 才判定显著；
# 少于 MIN_SEEDS 个种子时只报描述统计，不作显著性/组件有效性判定。
# 窗口级 bootstrap 的最低样本量：窗口只有 2~3 个时，重采样分布本身几乎
# 没有自由度，却可能给出极窄区间和“显著”假象（冒烟配置就会触发）。少于
# 该门槛只报点估计/区间并明确标注证据不足，不生成显著性措辞。



def _finite_number(value):
    """外部 JSON 数值 → 有限 float；bool/缺失/坏类型/NaN/Inf 返回 None。

    bool 是 int 的子类（True==1、False==0），若直接 float() 会把损坏的
    ``"em": true`` / ``"ppl": false`` / ``"val_windows": true`` 伪装成合法
    数值观测，进入跨种子均值、配对检验或 RULER 平均。汇总层读取的是外部/
    旧版产物，必须按缺失处理而不是替上游猜值。
    """
    if value is None or isinstance(value, (bool, np.bool_)):
        return None
    # float("1_0")==10.0：下划线数字串不是规范 JSON 数值，不能伪装成观测。
    if isinstance(value, str) and "_" in value:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) else None


def _test_label_key(value):
    """交互/Shapley 标签 → 可安全作为 dict key 的非空字符串，否则 None。

    汇总层读取外部/旧版产物时，标签可能是 list/dict（不可哈希）或空串；
    直接拿去做 dict key 会让整页汇总因 ``TypeError`` 崩溃。标签只用于与
    训练期结果对齐，畸形值按缺失处理，不影响其余行渲染。
    """
    return value if isinstance(value, str) and value else None


def _rate_or_none(value):
    """外部 JSON 的 AUROC/PR-AUC/F1/基率等比例 → [0,1] 内有限 float。

    这些指标有天然上界 1：坏/手改产物里的 1.5 或 -0.2 若被当成合法观测，
    会直接进入跨种子均值与配对检验，甚至伪造出“机制判别显著更好”的结论。
    越界值按缺失处理，与 analysis/ruler 的取值校验同口径。
    """
    value = _finite_number(value)
    return value if value is not None and 0.0 <= value <= 1.0 else None


def _cmd_env(cmd, env=None):
    """为单条子进程命令构造环境：PYTHONHASHSEED 与该命令 --seed 对齐。

    PYTHONHASHSEED 必须在解释器启动前设置才生效，父进程统一设一个值会让
    所有 (变体, 种子) 子进程共享第一个种子的哈希盐；这里逐命令解析
    --seed，保证每个 run 的 Python 哈希随机化也由它自己的种子唯一确定
    （没有 --seed 的命令如 latency/plots 沿用基准值）。
    """
    e = dict(env if env is not None else os.environ)
    seed_value = None
    for i, arg in enumerate(cmd):
        if arg == "--seed":
            if i + 1 >= len(cmd):
                raise ValueError("--seed 缺少取值")
            seed_value = cmd[i + 1]
            break
        if isinstance(arg, str) and arg.startswith("--seed="):
            seed_value = arg.split("=", 1)[1]
            break
    if seed_value is not None:
        e["PYTHONHASHSEED"] = str(seed_int(seed_value, "--seed"))
    ensure_cublas_workspace_config(e)
    return e


def run(cmd, env=None):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT, env=_cmd_env(cmd, env))


def _resolve_jobs(args) -> int:
    """并发度：显式 --jobs 优先；未指定时 CPU 取 min(4, 核数)，CUDA/--wandb 取 1。

    小核机器（<=4 核）默认取 核数+1：_jobs_env 已把每个子进程压到单 BLAS
    线程，总线程数不随 worker 数增加；大核机器维持 4 并发。显式 --jobs
    完全以用户值为准，不受此默认影响。
    """
    if args.jobs is not None:
        return max(1, int(args.jobs))
    if args.wandb:
        return 1
    if get_device(args.device).startswith("cuda"):
        return 1
    cpu = os.cpu_count() or 1
    return min(4, cpu) if cpu > 4 else cpu + 1


def parse_variants(spec: str) -> list:
    """解析 --variants 逗号列表，保留顺序并按首次出现去重（与 --seeds 同口径）。

    重复变体会让每个阶段生成重复命令（sanity/训练/评测/机制分析同一输出路径），
    并发执行时互相覆盖结果文件——去重后行为与显式去重的种子列表一致。
    """
    return list(dict.fromkeys(v.strip() for v in spec.split(",") if v.strip()))


def _strict_seed_int(value, parser=None, label="种子"):
    """严格种子化：bool/小数/负值/超过 2^32-1 均显式报错。

    True 与 1 在 dict 里同键、1.5 与 1 截断后相等，都会让不同的种子在
    ``dict.fromkeys`` 去重时合并，或让台账/结果目录名与实际播种不一致。
    越界种子会让全局 RNG 只播种一半后才失败，必须在启动子进程前拦截。
    """
    try:
        return seed_int(value, label)
    except ValueError as e:
        if parser is not None:
            parser.error(str(e))
        raise


def _parse_seed_list(spec, parser=None, label="--seeds"):
    """逗号分隔的整数种子列表；整数值浮点/整数字符串合法，其余显式报错。"""
    out = []
    for x in str(spec).split(","):
        x = x.strip()
        if not x:
            continue
        out.append(_strict_seed_int(x, parser, label))
    return out


def _jobs_env(jobs: int):
    """并发时按 jobs 均分每个子进程的 BLAS 线程配额；jobs<=1 返回 None。"""
    if jobs <= 1:
        return None
    env = os.environ.copy()
    n = max(1, (os.cpu_count() or 1) // jobs)
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS"):
        env[k] = str(n)
    return env


def run_many(cmds, jobs: int, env=None) -> None:
    """顺序或并发执行多条独立子进程命令。

    并发只改变多个独立 run 之间的调度，不改变单个 run 的行为；每条命令的
    PYTHONHASHSEED 按自身 --seed 设置（见 _cmd_env）。失败时等待全部命令
    结束后抛出首个（按提交顺序）非零退出码的 CalledProcessError。
    """
    jobs = max(1, int(jobs))
    if jobs == 1 or len(cmds) <= 1:
        for cmd in cmds:
            subprocess.run(cmd, check=True, cwd=ROOT, env=_cmd_env(cmd, env))
        return
    rcs = [None] * len(cmds)

    def _one(i, cmd):
        rcs[i] = subprocess.run(cmd, cwd=ROOT, env=_cmd_env(cmd, env)
                                ).returncode

    with ThreadPoolExecutor(max_workers=min(jobs, len(cmds))) as ex:
        list(ex.map(lambda t: _one(*t), enumerate(cmds)))
    for cmd, rc in zip(cmds, rcs):
        if rc != 0:
            raise subprocess.CalledProcessError(rc, cmd)


def _repo_path(path: str) -> str:
    """相对路径统一解释为仓库根目录；父进程与子进程必须看到同一位置。"""
    return path if os.path.isabs(path) else os.path.abspath(os.path.join(ROOT, path))


def mark_repro_failed(results_dir: str, *, error: str, stage: str,
                      variant: str = None, returncode: int = None) -> bool:
    """把本轮 ``repro.json`` 标成 aborted（失败轮次不能看起来仍在运行）。

    台账已有 aborted 终态，但全局 repro 是人工/下游最先打开的溯源文件：
    只写 ``started`` 内容会让失败轮次与“仍在跑”无法区分。这里保留启动期
    已记录的 git/worktree/seed_audit 等事实，只补失败终态，不伪造 completed
    的 tracking_end。文件不存在/不可解析时静默返回 False——失败路径不能
    因为补写溯源而掩盖原异常。
    """
    if not results_dir:
        return False
    path = os.path.join(results_dir, "repro.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            doc = json.load(f)
    except (OSError, ValueError):
        return False
    if not isinstance(doc, dict):
        return False
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    failure = {"time": ts, "error": error, "stage": stage}
    if variant is not None:
        failure["variant"] = variant
    if returncode is not None:
        failure["returncode"] = int(returncode)
    doc["status"] = "aborted"
    doc["failure"] = failure
    doc["tracking_end"] = dict(failure)
    try:
        save_json(doc, path)
    except OSError as e:
        print(f"[repro] ⚠ 失败终态写入 repro.json 失败（{e}）")
        return False
    return True


def _snapshot_existing_runs(ledger_common: dict) -> list:
    """失败轮次落盘已完成/进行中 run 的实际证据（不伪造未产生的产物）。

    失败发生在训练链或后处理阶段时，已跑完的 run 目录里已有 manifest/
    config/train_log/eval（甚至 checkpoint）。只写一条 aborted 会让这些
    逐 run 种子审计与指标从台账历史中消失。这里按本轮 out_root 扫描
    run_manifest.json，且只接受本轮开始之后写入的清单，避免把上一轮
    残留目录当成本轮证据；单个 run 快照失败不影响其余 run。
    """
    out_root = ledger_common.get("out_root")
    variants = ledger_common.get("variants") or []
    seeds = ledger_common.get("seeds") or []
    if not out_root or not variants or not seeds:
        return []
    n_seeds = len(seeds)
    try:
        start = time.mktime(time.strptime(
            str(ledger_common.get("time_start")), "%Y-%m-%d %H:%M:%S"))
    except (TypeError, ValueError, OverflowError):
        # 无法解析本轮开始时刻就无法证明目录属于本轮；宁可少收证据，
        # 也不能把上一轮残留 manifest 当成本轮快照。
        return []
    runs = []
    for v in variants:
        for seed in seeds:
            vdir = _variant_dir(out_root, v, seed, n_seeds)
            manifest = os.path.join(vdir, "run_manifest.json")
            if not os.path.isfile(manifest):
                continue
            try:
                # 只接受本轮开始之后写入的清单；不设回退宽限（time_start
                # 精确到秒，manifest 在本轮内写入必然 >= start）。
                if os.path.getmtime(manifest) < start:
                    continue
            except OSError:
                continue
            try:
                runs.append(snapshot_train_run(str(v), int(seed), vdir))
            except Exception as e:  # noqa: BLE001 - 失败路径不阻塞主异常
                print(f"[repro] ⚠ run 快照失败 {v}/seed{seed}: {e}")
    return runs


def append_aborted_record(ledger: str, ledger_common: dict, *, error: str,
                          stage: str, variant: str = None,
                          returncode: int = None, sanity: dict = None,
                          runs: list = None) -> None:
    """失败轮次的终态记录：附失败阶段/原因/退出码，失败实验也完整留痕。

    runs（可选）为失败前已产生的逐 run 证据快照，写入 aborted 记录；
    不传时保持旧行为（只有失败原因，不伪造 run 覆盖）。
    """
    extra = {"error": error, "stage": stage}
    if variant is not None:
        extra["variant"] = variant
    if returncode is not None:
        extra["returncode"] = int(returncode)
    append_record(ledger, build_experiment_record(
        status="aborted", time_end=time.strftime("%Y-%m-%d %H:%M:%S"),
        sanity=sanity, runs=runs, extra=extra, **ledger_common))
    # 台账 aborted 与全局 repro 的失败终态必须一起落盘；后者写失败不能
    # 影响前者的完整性，也不能掩盖调用方即将抛出的原始异常。
    try:
        mark_repro_failed(
            ledger_common.get("results_dir"), error=error, stage=stage,
            variant=variant, returncode=returncode)
    except Exception as repro_err:  # noqa: BLE001 - 失败路径只告警
        print(f"[repro] ⚠ 失败终态写入 repro.json 异常（{repro_err}）")


@contextmanager
def abort_ledger_on_failure(ledger: str, ledger_common: dict, stage):
    """任一阶段抛错时为已 started 的轮次补写 aborted 终态。"""
    try:
        yield
    except BaseException as e:
        stage_name = stage() if callable(stage) else stage
        try:
            runs = _snapshot_existing_runs(ledger_common)
            append_aborted_record(
                ledger, ledger_common, error="postrun_stage_failed",
                stage=str(stage_name),
                returncode=getattr(e, "returncode", None),
                runs=runs or None)
        except Exception as ledger_err:
            print(f"[repro] ⚠ aborted 台账写入失败（{ledger_err}）；"
                  f"原异常仍将抛出")
        raise


def record_aborted_sanity(ledger: str, ledger_common: dict, variant: str,
                          returncode: int, results_dir: str,
                          reason: str = None) -> dict:
    """sanity 闸门失败时向台账追写 aborted 记录，返回结果文件摘要。

    只留 dangling started 无法区分“被中止”与“崩溃”，也没有失败原因；
    aborted 记录内嵌失败变体、退出码与该变体的 sanity 结果摘要（含未通过
    项），供事后复现失败现场。reason 用于 stale/variant/seed 等“文件本身
    看起来 PASS 但不属于本轮”的失败：显式写入原因并把摘要 passed 改为
    False，避免 aborted 记录里出现 verdict=fail 却 failed.passed=true。
    """
    failed = sanity_summary(os.path.join(results_dir, f"sanity_{variant}.json"))
    if reason is not None:
        failed["gate_failure_reason"] = reason
        if reason != "not_passed":
            failed["passed"] = False
    append_aborted_record(
        ledger, ledger_common, error="sanity_gate_failed", stage="sanity",
        variant=variant, returncode=returncode,
        sanity={"executed": True, "verdict": {variant: "fail"},
                "reason": reason, "failed": failed})
    return failed


def check_sanity_results(variants, results_dir: str, not_before: float = None,
                         expected_seed: int = None):
    """读回本次 sanity 结果文件，返回 (verdict, failures)。

    子进程退出码 0 不等于闸门通过：结果文件可能缺失/不可解析/仍写着
    passed=false，或包含 malformed 证据。全量训练前逐个复核，失败项带
    variant/reason/summary，供 aborted 记录与排查信息复用。

    ``not_before``（可选，epoch 秒）要求结果文件本次运行后写入：防止
    sanity 子进程异常退出 0 却没写文件时，上一轮的 PASS 被当成本轮证据。
    文件系统 mtime 粒度可能较粗，调用方应传入开始时刻并预留 1 秒余量。
    ``expected_seed``（可选）核对结果内声明的 seed 与 run_all 传入的
    首种子一致；``variant`` 也必须与目标变体一致——把别的变体/种子的
    PASS 文件复制成本文件名时，不能在本轮闸门中复用。
    """
    verdict, failures = {}, []
    for v in variants:
        path = os.path.join(results_dir, f"sanity_{v}.json")
        summary = sanity_summary(path)
        reason = None
        if summary.get("present") is not True:
            reason = "missing_or_unparseable"
        elif summary.get("passed") is not True:
            reason = "not_passed"
        elif summary.get("malformed") is True:
            reason = "malformed"
        elif (summary.get("variant") is not None
              and str(summary.get("variant")) != str(v)):
            reason = "variant_mismatch"
        elif (expected_seed is not None
              and summary.get("seed") is not None):
            try:
                if strict_int(summary.get("seed"),
                              "sanity.seed") != int(expected_seed):
                    reason = "seed_mismatch"
            except (TypeError, ValueError, OverflowError):
                reason = "seed_mismatch"
        if reason is None and not_before is not None:
            try:
                mtime = os.path.getmtime(path)
            except OSError:
                reason = "missing_or_unparseable"
            else:
                if mtime < float(not_before) - 1.0:
                    reason = "stale_result"
        if reason is not None:
            failures.append({"variant": v, "reason": reason,
                             "summary": summary, "path": path})
        else:
            verdict[v] = "pass"
    return verdict, failures


def _seed_tag(seed: int, n_seeds: int) -> str:
    """多种子运行时给结果文件名加 _seed<seed> 后缀；单种子保持旧文件名。"""
    return f"_seed{seed}" if n_seeds > 1 else ""


def _variant_dir(out_root: str, variant: str, seed: int, n_seeds: int) -> str:
    """多种子运行时每个 (变体, 种子) 独立子目录；单种子保持旧布局。"""
    if n_seeds > 1:
        return os.path.join(out_root, variant, f"seed{seed}")
    return os.path.join(out_root, variant)


# 配对/审计必须共享的 PPL 评测协议：同一 split、同一上下文、同一批复用窗口
# （eval_iters/val_windows/val_tokens_evaluated）才是同一批 token 上的 PPL。
# 仅核对训练 config 不够——独立评测可被 --eval_iters 等参数重跑。


def _protocol_value(v):
    """把评测协议字段规范化后比较（"10" 与 10、[256,512] 与元组视为相同）。"""
    if isinstance(v, (bool, np.bool_)):
        return ("bool", bool(v))
    if isinstance(v, (int, np.integer)):
        return ("num", int(v))
    if isinstance(v, str):
        # Python 的 int()/float() 都接受 "1_0"（=10），会把拼写不同的
        # 协议值（以及种子）静默合并；与 strict_int 一样显式拒绝下划线。
        if "_" in v:
            return ("raw", repr(v))
        # 只把普通十进制整数字符串规范化；"10" 与整数 10 视为同一协议值。
        if re.fullmatch(r"[+-]?\d+", v.strip()) is not None:
            return ("num", int(v))
    if isinstance(v, (float, np.floating)) or isinstance(v, str):
        try:
            f = float(v)
        except (TypeError, ValueError, OverflowError):
            return ("raw", repr(v))
        return ("num", f) if math.isfinite(f) else ("raw", repr(v))
    if isinstance(v, (list, tuple)):
        return ("seq", tuple(_protocol_value(x) for x in v))
    return ("raw", repr(v))


def _same_protocol_value(a, b) -> bool:
    return _protocol_value(a) == _protocol_value(b)


def _same_protocol_fields(a, b) -> bool:
    """两侧协议字段集合必须一致，且共同字段逐项相同。

    只比较交集会把“一方是旧产物、缺 eval_ctx/sample_seed”误当成可比：
    缺失字段无法证明两侧用了同一批 token/样例，不能让它静默参与配对。
    """
    if set(a) != set(b):
        return False
    return all(_same_protocol_value(a[k], b[k]) for k in a)


def _integral_value(value):
    """把外部 JSON 数值规范为整数；bool/非有限/小数返回 None。"""
    if isinstance(value, (bool, np.bool_)):
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, str):
        # Python 的 int()/float() 都接受 "1_0"（=10）：下划线写法既不是
        # 规范整数串，也不该与种子 10 视为同一身份（strict_int 同口径拒绝）。
        s = value.strip()
        if not s or "_" in s:
            return None
        if re.fullmatch(r"[+-]?\d+", s) is not None:
            return int(s)
        # 小数/科学计数法字符串必须精确解析：经 float() 往返会让
        # "9007199254740993.0" 变成 9007199254740992，从而伪造种子/步数
        # 身份相等。Decimal 保留十进制精确值，仅在确实为整数时转换。
        if re.fullmatch(
                r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?",
                s) is None:
            return None
        try:
            d = Decimal(s)
        except (InvalidOperation, ValueError):
            return None
        if not d.is_finite() or d != d.to_integral_value():
            return None
        return int(d)
    try:
        f = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(f) or not f.is_integer():
        return None
    return int(f)


def _same_integral_value(value, expected) -> bool:
    """两个值是否表示同一个整数（拒绝 bool/非有限/小数截断）。"""
    iv, ie = _integral_value(value), _integral_value(expected)
    return iv is not None and ie is not None and iv == ie


def _same_seed(value, expected) -> bool:
    return _same_integral_value(value, expected)


def _ppl_protocol(d) -> dict:
    """从 eval.json 提取实际生效的评测条件；缺失字段不伪造。"""
    if not isinstance(d, dict):
        return {}
    return {k: d[k] for k in _PPL_PROTOCOL_KEYS
            if k in d and d[k] is not None}


def _ruler_protocol(d) -> dict:
    """从 ruler payload 提取决定样本集/生成口径的字段；缺失字段不伪造。

    这些字段任一不同，EM 就不是同一批样例/同一生成规则上的配对观测：
    ctx_lens/n_samples/sample_seed 决定题面与抽样，corpus_source 决定
    干扰文本是否见过，max_new/fwe_max_new/vt_* 决定解码预算，
    depth_stratified 决定 needle 位置是分层抽样还是随机抽样；
    samples_manifest_sha256 进一步绑定实际生成的 prompt/答案内容。
    """
    if not isinstance(d, dict):
        return {}
    keys = ("ctx_lens", "n_samples", "sample_seed", "corpus_source",
            "corpus_held_out", "max_new", "fwe_max_new", "vt_chains",
            "vt_depth", "amp", "depth_stratified",
            "samples_manifest_sha256")
    return {k: d[k] for k in keys if k in d and d[k] is not None}


def _ruler_cell_signature(d):
    """RULER 已评测 (任务, 上下文, 样本数, 内容清单) 单元；缺格均不可配对。

    同一 (任务, 上下文) 但 n 不同说明实际参与平均的样例数不同（例如部分
    样例构造失败被跳过），两边的聚合 EM/macro 不是同一批观测上的配对值。
    新产物还带 samples_sha256：同 seed/计数但题面不同会得到不同签名。
    """
    if not isinstance(d, dict):
        return None
    results = d.get("results")
    if not isinstance(results, dict):
        return None
    cells = set()
    for task, by_ctx in results.items():
        if (not isinstance(task, str) or task.startswith("_")
                or not isinstance(by_ctx, dict)):
            continue
        for ctx, cell in by_ctx.items():
            if not isinstance(cell, dict):
                continue
            n = _integral_value(cell.get("n"))
            # 内容清单哈希缺失的旧产物仍是 3 元组；带哈希的新产物与旧产物
            # 不相等，因此“同 seed 但无法证明同题面”不会静默配对。
            if "samples_sha256" in cell:
                digest = _protocol_value(cell.get("samples_sha256"))
                cells.add((task, str(ctx), n, digest))
            else:
                cells.add((task, str(ctx), n))
    return frozenset(cells)
