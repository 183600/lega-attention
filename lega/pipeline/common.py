"""流水线通用工具、单因素审计与协议辅助函数。"""

from ._shared import _install as _shared_install

_shared_install(globals())


def _gate_factor_set(value):
    """把 GATE_FACTORS 的值规范化为因子集合；结构非法时返回 None。

    只接受字符串集合而不是字符串本身：``"ABC"`` 迭代后也是 A/B/C，若不
    排除会被误当成合法因子集合，掩盖“元数据写成了拼接字符串”的漂移。
    """
    if isinstance(value, str) or not isinstance(value, (set, frozenset, tuple, list)):
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

_FACTOR_LOO_VARIANT = {"A": "lega_no_query", "B": "lega_no_conc", "C": "lega_no_len"}


def _factor_cell_set(cell):
    """把全因子格名规范化为因子集合；非法格名返回 None。"""
    if cell == "none":
        return frozenset()
    if (
        isinstance(cell, str)
        and cell
        and len(set(cell)) == len(cell)
        and all(ch in "ABC" for ch in cell)
    ):
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
            problems.append(f"{label} 的变体 {variant!r} 同时对应多个格")
            continue
        mapping[variant] = factors
    missing = _GATE_FACTOR_SUBSETS - seen_subsets
    extra = seen_subsets - _GATE_FACTOR_SUBSETS
    if missing or extra:
        problems.append(
            f"{label} 未覆盖完整 2^3：缺少 "
            + (", ".join(sorted("".join(sorted(s)) or "none" for s in missing)) or "无")
            + "；多出 "
            + (", ".join(sorted("".join(sorted(s)) or "none" for s in extra)) or "无")
        )
    return mapping


def _audit_single_factor_edge(
    label, factor, direction, candidate, reference, factors, problems
):
    """核对指定网格中的一条边确实只沿声明方向改变一个因子。"""
    c = factors.get(candidate)
    r = factors.get(reference)
    if c is None or r is None:
        problems.append(f"{label} {candidate} vs {reference} 不在对应参数化网格内")
        return False
    diff = frozenset(c.symmetric_difference(r))
    if diff != frozenset({factor}):
        problems.append(
            f"{label} {candidate} vs {reference} 声明为单因素 {factor}，"
            f"实际改变 {sorted(diff) or '无'}"
        )
        return False
    if direction == "remove":
        shape_ok = c < r and (r - c) == frozenset({factor})
    else:
        shape_ok = r < c and (c - r) == frozenset({factor})
    if not shape_ok:
        problems.append(
            f"{label} {candidate} vs {reference} 的 {direction} 方向与" f"因子变化不符"
        )
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
        globals().get("NORMALIZED_FACTORIAL_CELLS"), "NORMALIZED_CELLS", problems
    )
    raw_cells = _audit_factorial_cells(
        globals().get("RAW_FACTORIAL_CELLS"), "RAW_CELLS", problems
    )
    for variant, factors in normalized_cells.items():
        if normalized.get(variant) != factors:
            problems.append(
                f"NORMALIZED_CELLS 的变体 {variant!r} 因子集合为 "
                f"{sorted(normalized.get(variant, ()))}，"
                f"应为 {sorted(factors)}"
            )
    if set(normalized_cells) != set(normalized):
        problems.append(
            "NORMALIZED_CELLS 的变体集合与 GATE_FACTORS 不一致："
            f"cells={sorted(normalized_cells)}，"
            f"factors={sorted(normalized)}"
        )
    raw_variants = globals().get("RAW_FACTORIAL_VARIANTS")
    if not isinstance(raw_variants, (tuple, list)):
        problems.append("RAW_FACTORIAL_VARIANTS 必须是变体序列")
    else:
        if len(set(raw_variants)) != len(raw_variants):
            problems.append("RAW_FACTORIAL_VARIANTS 含重复变体")
        if set(raw_variants) != set(raw_cells):
            problems.append("RAW_FACTORIAL_VARIANTS 与 RAW_CELLS 的变体集合不一致")
    # RAW_CELLS 是实验设计元数据；它必须与 raw 变体实现里实际启用的
    # A/B/C 开关一致。否则把 B 格写成归一化 conc 变体（或把 AB 格写成
    # lega_no_len）仍能通过“格名覆盖 2^3”的检查，却不再是原始参数化边。
    raw_gate_factors = globals().get("RAW_GATE_FACTORS")
    raw_impl = {}
    if not isinstance(raw_gate_factors, dict):
        problems.append("RAW_GATE_FACTORS 必须是变体 -> A/B/C 因子集合的映射")
    else:
        for variant, raw_factors in raw_gate_factors.items():
            if not isinstance(variant, str):
                problems.append(f"RAW_GATE_FACTORS 含非字符串变体键：{variant!r}")
                continue
            factors = _gate_factor_set(raw_factors)
            if factors is None:
                problems.append(
                    f"RAW_GATE_FACTORS[{variant!r}] 不是合法的因子集合："
                    f"{raw_factors!r}"
                )
                continue
            unknown = factors - frozenset("ABC")
            if unknown:
                problems.append(
                    f"RAW_GATE_FACTORS[{variant!r}] 含未知因子：" f"{sorted(unknown)}"
                )
                continue
            raw_impl[variant] = factors
        if set(raw_impl) != set(raw_cells):
            problems.append(
                "RAW_CELLS 的变体集合与 RAW_GATE_FACTORS 不一致："
                f"cells={sorted(raw_cells)}，"
                f"factors={sorted(raw_impl)}"
            )
        for variant, factors in raw_cells.items():
            impl = raw_impl.get(variant)
            if impl is not None and impl != factors:
                problems.append(
                    f"RAW_CELLS 的变体 {variant!r} 因子集合为 "
                    f"{sorted(factors)}，实现为 {sorted(impl)}"
                )
    contrasts = globals().get("REPARAMETERIZATION_CONTRASTS")
    if not isinstance(contrasts, (tuple, list)) or isinstance(contrasts, str):
        problems.append("REPARAMETERIZATION_CONTRASTS 必须是对比序列")
        return []
    required = {("A", "remove"), ("B", "remove"), ("C", "remove"), ("B", "add")}
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
            problems.append(f"参数化对比存在非 A/B/C 单因素声明：{factor}/{direction}")
            continue
        edge = (factor, direction)
        if edge in seen:
            problems.append(f"重复的参数化单因素对比：{factor}/{direction}")
            continue
        seen.add(edge)
        ok = True
        for label, pair, grid in (
            ("归一化参数化", norm_pair, normalized_cells),
            ("原始参数化", raw_pair, raw_cells),
        ):
            if (
                not isinstance(pair, (tuple, list))
                or len(pair) != 2
                or not all(isinstance(x, str) for x in pair)
            ):
                problems.append(f"{label}对比结构非法：{pair!r}")
                ok = False
                continue
            candidate, reference = pair
            if candidate == reference:
                problems.append(
                    f"{label}单因素对比两侧相同：{candidate} vs {reference}"
                )
                ok = False
                continue
            if not _audit_single_factor_edge(
                label, factor, direction, candidate, reference, grid, problems
            ):
                ok = False
        sf_pairs = (
            SINGLE_FACTOR_PAIRS
            if isinstance(SINGLE_FACTOR_PAIRS, (tuple, list))
            else ()
        )
        normalized_edges = {
            (p[0], p[1])
            for p in sf_pairs
            if isinstance(p, (tuple, list))
            and len(p) == 2
            and all(isinstance(x, str) for x in p)
        }
        if (
            isinstance(norm_pair, (tuple, list))
            and len(norm_pair) == 2
            and tuple(norm_pair) not in normalized_edges
        ):
            problems.append(
                f"参数化对比的归一化边 {tuple(norm_pair)!r} 未登记为" "单因素对比"
            )
        if ok:
            audited.append(
                {
                    "factor": factor,
                    "direction": direction,
                    "normalized_candidate": norm_pair[0],
                    "normalized_reference": norm_pair[1],
                    "raw_candidate": raw_pair[0],
                    "raw_reference": raw_pair[1],
                }
            )
    missing = required - seen
    if missing:
        problems.append(
            "缺少原始参数化单因素对比："
            + ", ".join(
                f"{factor}/{direction}" for factor, direction in sorted(missing)
            )
        )
    return audited


def _audit_single_factor_pair_table(contrasts, problems):
    """核对派生配对表与已声明的单因素边完全一致。

    ``SINGLE_FACTOR_PAIRS`` 是汇总/报告实际消费的配对集合；它必须由
    ``SINGLE_FACTOR_CONTRASTS`` 逐条派生，不能手工增删。若表里混入
    “B-only vs Full”这类多因子对比，或漏掉一条已审计边，审计必须在
    运行期显形，而不是让组件归因在别处静默失真。
    """
    raw_pairs = globals().get("SINGLE_FACTOR_PAIRS")
    if not isinstance(raw_pairs, (tuple, list)) or isinstance(raw_pairs, str):
        problems.append("SINGLE_FACTOR_PAIRS 必须是对比序列")
        return []
    seen = set()
    pairs = []
    for entry in raw_pairs:
        if (
            not isinstance(entry, (tuple, list))
            or len(entry) != 2
            or not all(isinstance(x, str) and x for x in entry)
        ):
            problems.append(f"单因素对比对结构非法：{entry!r}")
            continue
        candidate, reference = entry
        pair = (candidate, reference)
        if candidate == reference:
            problems.append(f"单因素对比对两侧相同：{candidate} vs {reference}")
            continue
        if pair in seen:
            problems.append(f"重复的单因素对比对：{candidate} vs {reference}")
            continue
        seen.add(pair)
        if single_factor_change(candidate, reference) is None:
            problems.append(
                f"单因素对比对 {candidate} vs {reference} 实际不改变"
                "恰好一个 A/B/C 因子"
            )
            continue
        pairs.append(pair)
    expected = {(c["candidate"], c["reference"]) for c in contrasts}
    missing = expected - seen
    extra = seen - expected
    if missing:
        problems.append(
            "SINGLE_FACTOR_PAIRS 缺少已声明对比："
            + ", ".join(f"{c} vs {r}" for c, r in sorted(missing))
        )
    if extra:
        problems.append(
            "SINGLE_FACTOR_PAIRS 含未声明对比："
            + ", ".join(f"{c} vs {r}" for c, r in sorted(extra))
        )
    return pairs


def _audit_inference_permutation_lesions(problems):
    """核对 A/B/C 的分布匹配置换病灶各自只干预一个证据通道。

    ``query_permute/conc_permute/len_permute`` 是推理期安慰剂：它们不修改
    参数，只把对应证据沿序列维错位。若映射把某个因子指到另一个通道，
    或让同一病灶承担多个因子，报告就会把“位置证据的因果作用”归错组件，
    因此这里按病灶 kind / requires / params 逐项核对。
    """
    entries = globals().get("PERMUTATION_LESION_FACTORS")
    if not isinstance(entries, (tuple, list)) or isinstance(entries, str):
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
        if (
            not isinstance(entry, (tuple, list))
            or len(entry) != 3
            or not all(isinstance(x, str) for x in entry)
        ):
            problems.append(f"单因素置换病灶定义结构非法：{entry!r}")
            continue
        factor, lesion_id, capability = entry
        if factor not in expected:
            problems.append(f"单因素置换病灶的因子非法：{factor!r}")
            continue
        if factor in seen_factors:
            problems.append(f"重复的单因素置换因子：{factor}")
        seen_factors.add(factor)
        if lesion_id in seen_ids:
            problems.append(f"置换病灶 {lesion_id!r} 承担多个单因子映射")
        seen_ids.add(lesion_id)
        expected_kind, expected_capability = expected[factor]
        if capability != expected_capability:
            problems.append(
                f"置换病灶 {lesion_id!r} 声明为 {factor}，但能力为 "
                f"{capability!r}，应为 {expected_capability!r}"
            )
        spec = lesion_by_id.get(lesion_id)
        if not isinstance(spec, dict):
            problems.append(f"置换病灶 {lesion_id!r} 未定义")
            continue
        if spec.get("kind") != expected_kind:
            problems.append(
                f"置换病灶 {lesion_id!r} 声明为单因素 {factor}，kind 应为 "
                f"{expected_kind!r}，实际 {spec.get('kind')!r}"
            )
        requires = spec.get("requires")
        if not isinstance(requires, (tuple, list, set, frozenset)) or not all(
            isinstance(x, str) for x in requires
        ):
            problems.append(f"置换病灶 {lesion_id!r} 的 requires 结构非法")
        elif set(requires) != {expected_capability}:
            problems.append(
                f"置换病灶 {lesion_id!r} 依赖 {sorted(set(requires)) or '无'}，"
                f"单因素 {factor} 应只依赖 {expected_capability}"
            )
        params = spec.get("params")
        if not isinstance(params, (tuple, list, set, frozenset)) or not all(
            isinstance(x, str) for x in params
        ):
            problems.append(f"置换病灶 {lesion_id!r} 的 params 结构非法")
        elif params:
            problems.append(
                f"置换病灶 {lesion_id!r} 不应清零参数，实际为 " f"{sorted(set(params))}"
            )
        if spec.get("factorial_cell") is True:
            problems.append(f"置换病灶 {lesion_id!r} 不应标记为全因子格")
        audited.append(
            {
                "factor": factor,
                "lesion": lesion_id,
                "capability": capability,
                "kind": expected_kind,
            }
        )
    missing = set(expected) - seen_factors
    if missing:
        problems.append("缺少单因素置换病灶：" + ", ".join(sorted(missing)))
    return audited


def _audit_inference_single_factor_design(problems):
    """核对推理期单病灶/活动匹配/2^3 格仍然只对应已声明的单因子。"""
    lesion_by_id = globals().get("ABLATION_LESION_BY_ID")
    if not isinstance(lesion_by_id, dict):
        problems.append("LESION_BY_ID 必须是病灶映射")
        return [], [], []
    pairs = globals().get("MATCHED_LESION_PAIRS")
    if not isinstance(pairs, (tuple, list)) or isinstance(pairs, str):
        problems.append("MATCHED_LESION_PAIRS 必须是配对序列")
        pairs = ()
    matched = []
    seen_factors = set()
    seen_ids = set()
    for entry in pairs:
        if (
            not isinstance(entry, (tuple, list))
            or len(entry) != 3
            or not all(isinstance(x, str) for x in entry)
        ):
            problems.append(f"活动匹配病灶配对结构非法：{entry!r}")
            continue
        raw_id, matched_id, factor = entry
        if factor not in ("A", "B", "C"):
            problems.append(
                f"活动匹配病灶 {raw_id} vs {matched_id} 的因子非法：" f"{factor!r}"
            )
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
        expected_capability = {"A": "query", "B": "conc", "C": "len"}[factor]
        for label, lid, spec in (
            ("原始病灶", raw_id, raw_spec),
            ("活动匹配病灶", matched_id, matched_spec),
        ):
            if lid in seen_ids:
                problems.append(f"病灶 {lid!r} 在多条匹配对照中重复使用")
            seen_ids.add(lid)
            params = spec.get("params")
            if not isinstance(params, (tuple, list, set, frozenset)) or not all(
                isinstance(x, str) for x in params
            ):
                problems.append(f"{label} {lid!r} 的 params 结构非法")
                continue
            if set(params) != expected_params:
                problems.append(
                    f"{label} {lid!r} 声明为单因素 {factor}，实际清零 "
                    f"{sorted(set(params)) or '无'}"
                )
            requires = spec.get("requires")
            if not isinstance(requires, (tuple, list, set, frozenset)) or not all(
                isinstance(x, str) for x in requires
            ):
                problems.append(f"{label} {lid!r} 的 requires 结构非法")
            elif set(requires) != {expected_capability}:
                problems.append(
                    f"{label} {lid!r} 声明为单因素 {factor}，但依赖 "
                    f"{sorted(set(requires)) or '无'}，应为 "
                    f"{expected_capability}"
                )
        if raw_spec.get("kind") != "zero":
            problems.append(f"原始病灶 {raw_id!r} 不是单因子清零病灶")
        if matched_spec.get("kind") != "zero_matched":
            problems.append(f"活动匹配病灶 {matched_id!r} 不是 zero_matched 病灶")
        if raw_spec.get("matched_control") is True:
            problems.append(f"原始病灶 {raw_id!r} 不应标为活动匹配对照")
        if matched_spec.get("matched_control") is not True:
            problems.append(f"活动匹配病灶 {matched_id!r} 未标记 matched_control")
        if matched_spec.get("matched_of") != raw_id:
            problems.append(
                f"活动匹配病灶 {matched_id!r} 的 matched_of 不是 " f"{raw_id!r}"
            )
        if matched_spec.get("matched_factor") != factor:
            problems.append(
                f"活动匹配病灶 {matched_id!r} 的 matched_factor 与 " f"{factor} 不一致"
            )
        matched.append({"raw": raw_id, "matched": matched_id, "factor": factor})
    missing_factors = {"A", "B", "C"} - seen_factors
    if missing_factors:
        problems.append("缺少活动匹配单因素病灶：" + ", ".join(sorted(missing_factors)))
    permutation_lesions = _audit_inference_permutation_lesions(problems)
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
                problems.append(f"推理期基线格 {cell!r} 不应映射病灶：{lesion_id!r}")
            else:
                audited_cells.append(
                    {"cell": cell, "lesion": None, "factors": sorted(factors)}
                )
            continue
        if not isinstance(lesion_id, str):
            problems.append(f"推理期全因子格 {cell!r} 的病灶名非法：{lesion_id!r}")
            continue
        if lesion_id in seen_lesions:
            problems.append(f"推理期病灶 {lesion_id!r} 同时对应多个全因子格")
        seen_lesions.add(lesion_id)
        spec = lesion_by_id.get(lesion_id)
        if not isinstance(spec, dict):
            problems.append(f"推理期全因子格病灶 {lesion_id!r} 未定义")
            continue
        params = spec.get("params")
        if not isinstance(params, (tuple, list, set, frozenset)) or not all(
            isinstance(x, str) for x in params
        ):
            problems.append(f"推理期病灶 {lesion_id!r} 的 params 结构非法")
            continue
        expected_zero = frozenset("ABC") - factors
        expected_params = {_FACTOR_PARAM[f] for f in expected_zero}
        if set(params) != expected_params:
            problems.append(
                f"推理期格 {cell!r} -> {lesion_id!r} 应清零 "
                f"{sorted(expected_params) or '无'}，实际清零 "
                f"{sorted(set(params)) or '无'}"
            )
        expected_kind = "zero_evidence" if not factors else "zero"
        if spec.get("kind") != expected_kind:
            problems.append(
                f"推理期格 {cell!r} -> {lesion_id!r} 的 kind 不是 " f"{expected_kind}"
            )
        flagged = spec.get("factorial_cell") is True
        if len(factors) == 1 and not flagged:
            problems.append(
                f"推理期单因子格 {cell!r} 的病灶 {lesion_id!r} 未标记 "
                "factorial_cell，可能混入单病灶因果表"
            )
        if len(factors) != 1 and flagged:
            problems.append(
                f"推理期非单因子格 {cell!r} 的病灶 {lesion_id!r} 不应标记 "
                "factorial_cell"
            )
        audited_cells.append(
            {"cell": cell, "lesion": lesion_id, "factors": sorted(factors)}
        )
    missing = _GATE_FACTOR_SUBSETS - seen_subsets
    extra = seen_subsets - _GATE_FACTOR_SUBSETS
    if missing or extra:
        problems.append(
            "推理期全因子格未覆盖完整 2^3：缺少 "
            + (", ".join(sorted("".join(sorted(s)) or "none" for s in missing)) or "无")
            + "；多出 "
            + (", ".join(sorted("".join(sorted(s)) or "none" for s in extra)) or "无")
        )
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
    if not isinstance(order, (tuple, list)) or isinstance(order, str):
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
        problems.append("缺少剂量-反应组件：" + ", ".join(sorted(missing_order)))
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
                problems.append(f"剂量-反应组件 {component} 的{label}非法：{value!r}")
                continue
            row[name] = value
            wanted = expected_map[component]
            if value != wanted:
                problems.append(
                    f"剂量-反应组件 {component} 的{label}为 {value!r}，"
                    f"应为 {wanted!r}，否则不再是单因素剂量实验"
                )
            prior = seen_values[name].get(value)
            if prior is not None and prior != component:
                problems.append(
                    f"剂量-反应组件 {component} 与 {prior} 共用{label} " f"{value!r}"
                )
            else:
                seen_values[name][value] = component
        audited.append(row)
    return audited


def _audit_capacity_triple_components(problems, normalized):
    """核对训练期容量匹配三臂仍是逐组件的单因子对照。

    容量分解把同一组件的 Full/Null_X/No_X 配对成两条边：Full−Null_X 只把
    X 的证据置零（参数保留），Null_X−No_X 只去掉 X 的参数（两侧证据都为
    0）。若 ``CAPACITY_NULL_VARIANTS``/``CAPACITY_OFF_VARIANTS`` 把某个
    组件指到别的变体，或 Null_X 实际没有保留全部门控参数、混进了主 2^3
    因子表，分解就会把多因子差异写成 info/capacity。这里核对映射恰好覆盖
    A/B/C、Null_X 与 X 一一对应、No_X 与已审计的去一法变体一致。
    """
    expected = ("A", "B", "C")
    nulls = globals().get("CAPACITY_NULL_VARIANTS")
    offs = globals().get("CAPACITY_OFF_VARIANTS")
    inverse = globals().get("NULL_EVIDENCE_COMPONENT")
    for label, mapping in (
        ("CAPACITY_NULL_VARIANTS", nulls),
        ("CAPACITY_OFF_VARIANTS", offs),
    ):
        if not isinstance(mapping, dict):
            problems.append(f"{label} 必须是组件 -> 变体映射")
            continue
        for component in mapping:
            if not isinstance(component, str):
                problems.append(f"{label} 含非字符串组件键：{component!r}")
            elif component not in expected:
                problems.append(f"{label} 存在未登记组件：{component!r}")
        missing = set(expected) - {c for c in mapping if isinstance(c, str)}
        if missing:
            problems.append(f"{label} 缺少组件：" + ", ".join(sorted(missing)))
    if not isinstance(nulls, dict):
        nulls = {}
    if not isinstance(offs, dict):
        offs = {}
    if not isinstance(inverse, dict):
        problems.append("NULL_EVIDENCE_COMPONENT 必须是变体 -> 组件映射")
        inverse = None
    else:
        reverse = {v: c for c, v in nulls.items() if isinstance(v, str)}
        if inverse != reverse:
            problems.append(
                "NULL_EVIDENCE_COMPONENT 与 CAPACITY_NULL_VARIANTS 不是"
                f"互逆映射：inverse={inverse!r}，reverse={reverse!r}"
            )
    if not isinstance(normalized, dict):
        normalized = {}
    full_factors = normalized.get("lega")
    if full_factors != frozenset(expected):
        problems.append(
            f"容量分解的 Full 变体 lega 因子集合为 "
            f"{sorted(full_factors or ())}，应为 A/B/C"
        )
    rows = []
    seen_null = {}
    seen_off = {}
    for component in expected:
        null_variant = nulls.get(component)
        off_variant = offs.get(component)
        if not isinstance(null_variant, str) or not null_variant:
            problems.append(
                f"容量匹配组件 {component} 的 Null 变体非法：" f"{null_variant!r}"
            )
            null_variant = None
        if not isinstance(off_variant, str) or not off_variant:
            problems.append(
                f"容量匹配组件 {component} 的 No 变体非法：" f"{off_variant!r}"
            )
            off_variant = None
        if null_variant is not None:
            prior = seen_null.get(null_variant)
            if prior is not None:
                problems.append(
                    f"组件 {component} 与 {prior} 共用容量匹配 Null 变体 "
                    f"{null_variant!r}"
                )
            else:
                seen_null[null_variant] = component
            if null_variant == off_variant:
                problems.append(
                    f"组件 {component} 的 Null 与 No 变体相同：" f"{null_variant!r}"
                )
            if inverse is not None and inverse.get(null_variant) != component:
                problems.append(
                    f"容量匹配 Null 变体 {null_variant!r} 对应组件为 "
                    f"{inverse.get(null_variant)!r}，应为 {component!r}"
                )
            for key, collection in (
                ("RUNNABLE_VARIANTS", RUNNABLE_VARIANTS),
                ("CONTROL_VARIANTS", CONTROL_VARIANTS),
                ("NULL_EVIDENCE_VARIANTS", NULL_EVIDENCE_VARIANTS),
                ("_GATED", _GATED),
                ("_LEGA_FAMILY", _LEGA_FAMILY),
            ):
                if null_variant not in collection:
                    problems.append(
                        f"容量匹配 Null 变体 {null_variant!r} 不在 "
                        f"{key} 中，无法保留全部门控参数/计算"
                    )
            if null_variant in GATE_FACTORS:
                problems.append(
                    f"容量匹配 Null 变体 {null_variant!r} 混入了主 A/B/C "
                    "因子表；它只把证据置零，不能当作结构格"
                )
            if isinstance(RAW_GATE_FACTORS, dict) and null_variant in RAW_GATE_FACTORS:
                problems.append(
                    f"容量匹配 Null 变体 {null_variant!r} 混入了原始参数化" "因子表"
                )
        if off_variant is not None:
            prior = seen_off.get(off_variant)
            if prior is not None:
                problems.append(
                    f"组件 {component} 与 {prior} 共用去一法 No 变体 "
                    f"{off_variant!r}"
                )
            else:
                seen_off[off_variant] = component
            wanted = _FACTOR_LOO_VARIANT.get(component)
            if off_variant != wanted:
                problems.append(
                    f"容量匹配组件 {component} 的 No 变体为 "
                    f"{off_variant!r}，应为去一法变体 {wanted!r}"
                )
            wanted_factors = frozenset(expected) - frozenset({component})
            actual = normalized.get(off_variant)
            if actual != wanted_factors:
                problems.append(
                    f"容量匹配组件 {component} 的 No 变体 {off_variant!r} "
                    f"因子集合为 {sorted(actual or ())}，应为 "
                    f"{sorted(wanted_factors)}"
                )
        if null_variant is not None and off_variant is not None:
            rows.append(
                {
                    "component": component,
                    "full_variant": "lega",
                    "null_variant": null_variant,
                    "off_variant": off_variant,
                    "evidence_param": _FACTOR_PARAM[component],
                    "evidence_capability": _FACTOR_CAPABILITY[component],
                }
            )
    overlap = set(seen_null) & set(seen_off)
    if overlap:
        problems.append(
            "容量匹配 Null 变体不能同时作为去一法 No 变体："
            + ", ".join(sorted(overlap))
        )
    return rows


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


def _allowed_multi_factor_pair(candidate, reference):
    """多因子对比是否在显式批准的白名单里（顺序无关）。

    白名单是 ``lega.attention.ALLOWED_MULTI_FACTOR_PAIRS``：网格内只有
    LEGA vs qgate 这类“与另一套已有方法比”的整体对比被批准；其它网格内
    多因子对比（B-only/C-only 与 Full）一律禁止。
    """
    allowed = globals().get("ALLOWED_MULTI_FACTOR_PAIRS")
    if not isinstance(allowed, (tuple, list)):
        return False
    try:
        key = frozenset((candidate, reference))
    except TypeError:
        return False
    for entry in allowed:
        if isinstance(entry, (set, frozenset)) and frozenset(entry) == key:
            return True
    return False


def grid_contrast_info(candidate, reference):
    """对比是否落在门控 2^3 网格内、是否同时改变多个因子、是否被批准。

    “网格内”= 两侧都在 ``GATE_FACTORS``（主归一化 2^3）里。网格内的
    多因子对比（B-only/C-only 与 Full 同时差两个因子）既不是整体方法
    对比，也不能当单组件证据：设计明确禁止把 B-only/C-only 与 Full
    配对，否则会产出一个“看起来像单组件”的确认性判定。只有白名单里
    显式批准的对比才允许作为整体方法对比进入分析。

    返回 dict：
      * ``grid_internal``：两侧是否都在门控网格内；
      * ``changed_factors``：实际改变的因子（网格内才有意义）；
      * ``multi_factor``：是否同时改变 ≥2 个因子；
      * ``allowed``：多因子时是否在白名单内；
      * ``forbidden``：是否是“禁止进入分析/确认性判定”的多因子对比。
    """
    diff = changed_gate_factors(candidate, reference)
    changed = sorted(diff) if diff else []
    multi = bool(diff is not None and len(diff) >= 2)
    allowed = True if not multi else _allowed_multi_factor_pair(
        candidate, reference
    )
    return {
        "candidate": candidate,
        "reference": reference,
        "grid_internal": diff is not None,
        "changed_factors": changed,
        "multi_factor": multi,
        "allowed": allowed,
        "forbidden": bool(multi and not allowed),
    }


# 登记表里"这条假设在消融某个组件"的措辞标记。预注册的正确性不能只靠
# 变体名看起来像：H2–H4 是核心消融假设，若登记表被改成多因子对比
# （如把 lega_no_conc 换成 lega_conc_only vs lega，同时改 A+C），汇总仍会
# 按"支持/拒绝"输出一个看似单组件的确认性判定。
_HYPOTHESIS_SINGLE_FACTOR_MARKERS = ("核心消融", "去一法", "单组件", "单因素", "只去掉")

# 去一法（从参照去掉该因子）期望候选变差；全因子增量（只加该因子）期望
# 候选更好。登记表的 direction 与这条边不符时，优效性判定本身会被写反。
_FACTOR_EDGE_EXPECTED_DIRECTION = {
    "remove": "candidate_worse",
    "add": "candidate_better",
}


def _audit_registered_single_factor(contrasts, problems, implementation_rows=None):
    """核对登记表里自称单组件消融的假设确实只改变一个门控因子。

    ``SINGLE_FACTOR_CONTRASTS`` 管的是流水线内部消费的 6 条边；预注册表
    （``preregistration/hypotheses.json``）里的 H2–H4 才是对外声明的那几个
    "核心消融"主张。两者一旦脱节（例如把候选换成与参照差两个因子的
    B-only 变体），``significance.json`` 仍会给出 支持/拒绝，看起来就像
    单组件的确认性证据。这里按 claim 措辞识别单组件消融假设，逐条核对：
    对比必须落在已审计的单因素表内、声明因子与实际因子一致、优效性方向
    与该边的 add/remove 语义一致；同时把整体方法对比（vs standard、vs
    qgate、机制 AUROC/AP 差）归为 overall，不误报。

    已审计的单因素边还要过第二道关：实现层实测（``implementation_rows``）
    确认两个变体的前向确实只差一个门控项。元数据说“只改一个因子”但实现
    悄悄多改了一项时，确认性判定必须失效。

    登记表不可读时如实记录 ``available=false``：这一层缺证据不等于通过。
    """
    rows = []
    try:
        from lega.hypotheses import load_registration

        registered = load_registration(HYPOTHESES_PATH)
    except Exception as e:  # pragma: no cover - 缺文件/损坏只在极端环境发生
        return [{"available": False, "reason": f"{type(e).__name__}: {e}"}]
    hypotheses = registered.get("hypotheses") if isinstance(registered, dict) else None
    if not isinstance(hypotheses, list):
        return [{"available": False, "reason": "登记表缺少 hypotheses 列表"}]
    audited_edges = {
        (c["candidate"], c["reference"]): (c["factor"], c["direction"])
        for c in contrasts
    }
    audited_pairs = set(audited_edges)
    for entry in hypotheses:
        if not isinstance(entry, dict):
            problems.append(f"登记表条目不是对象：{entry!r}")
            continue
        hyp_id = str(entry.get("id") or "?")
        claim = str(entry.get("claim") or "")
        cand, ref = entry.get("candidate"), entry.get("reference")
        if not (isinstance(cand, str) and isinstance(ref, str) and cand and ref):
            problems.append(f"登记假设 {hyp_id} 的候选/参照缺失或非法")
            continue
        row = {
            "id": hyp_id,
            "candidate": cand,
            "reference": ref,
            "claim": claim,
            "metric": entry.get("metric"),
            # 该对比是否是“禁止进入确认性判定”的网格内多因子对比。
            "forbidden": False,
        }
        requires_single = any(m in claim for m in _HYPOTHESIS_SINGLE_FACTOR_MARKERS)
        row["claims_single_factor"] = bool(requires_single)
        change = single_factor_change(cand, ref)
        if (cand, ref) in audited_pairs:
            factor, direction = audited_edges[(cand, ref)]
            row["contrast_type"] = "single_factor"
            row["factor"] = factor
            row["edge_direction"] = direction
            if change != factor:
                problems.append(
                    f"登记假设 {hyp_id} 的对比 {cand} vs {ref} 声明单因素 "
                    f"{factor}，实际因子差为 {sorted(change) if change else '无'}"
                )
        elif change is not None:
            # 对称差恰好 1 但未登记：不能当单因素归因，只如实标注
            row["contrast_type"] = "unregistered_single_factor"
            row["factor"] = change
            if requires_single:
                problems.append(
                    f"登记假设 {hyp_id} 自称单组件消融，但 {cand} vs {ref} "
                    f"不在已审计的单因素对比表内"
                )
        else:
            diff = changed_gate_factors(cand, ref)
            info = grid_contrast_info(cand, ref)
            row["contrast_type"] = (
                "overall"
                if diff is None
                else ("multi_factor_forbidden" if info["forbidden"] else "multi_factor")
            )
            row["changed_factors"] = sorted(diff) if diff else []
            row["forbidden"] = bool(info["forbidden"])
            if info["forbidden"]:
                # 网格内的多因子对比（B-only/C-only 与 Full 同时改两个因子）
                # 既不是整体方法对比，也不能当单组件证据：登记成确认性
                # 主张会让读者以为某个组件的贡献被单独测出来了。这里不看
                # claim 措辞，直接按因子差判定。
                problems.append(
                    f"登记假设 {hyp_id} 的对比 {cand} vs {ref} 同时改变 "
                    f"{row['changed_factors']}，属于门控网格内的多因子对比："
                    "它不是整体方法对比，也不能用于组件归因，必须改用"
                    "已审计的单因素边（去一法 Full−X vs Full 或 X-only vs "
                    "gate_only）"
                )
            if requires_single:
                problems.append(
                    f"登记假设 {hyp_id} 自称单组件消融，但 {cand} vs {ref} "
                    + (
                        f"改变了多个因子 {row['changed_factors']}"
                        if diff
                        else "不在门控 2^3 因子网格内"
                    )
                )
        if row["contrast_type"] == "single_factor":
            expected_dir = _FACTOR_EDGE_EXPECTED_DIRECTION.get(row["edge_direction"])
            declared_dir = entry.get("direction")
            if expected_dir is not None and declared_dir != expected_dir:
                problems.append(
                    f"登记假设 {hyp_id} 的 direction={declared_dir!r} 与 "
                    f"{row['edge_direction']} 边（{row['factor']}）的期望"
                    f"方向 {expected_dir!r} 不符：优效性判定会被写反"
                )
                row["direction_mismatch"] = True
            else:
                row["direction_mismatch"] = False
            # 第二道关：实现层实测。只改元数据表就能让“实现漂移”的对比
            # 继续拿到确认性判定——这里用实测生效因子把这条边挡住。
            impl_ok, impl_reason = implementation_single_factor_check(
                {"implementation_factors": implementation_rows or []}, cand, ref
            )
            row["implementation_ok"] = impl_ok
            row["implementation_reason"] = impl_reason
            if impl_ok is False:
                problems.append(
                    f"登记假设 {hyp_id} 的对比 {cand} vs {ref} 未通过实现层"
                    f"单因素核对：{impl_reason}"
                )
        rows.append(row)
    return rows


def _audit_implementation_factors(normalized, problems):
    """核对门控因子在**实现层**真正生效的集合与声明的因子表一致。

    ``GATE_FACTORS``/``RAW_GATE_FACTORS`` 都从实现的 ``_USE_*`` 开关派生，
    因此"改公式但忘了改开关"会让元数据与实现一起失真：某个变体名义上是
    ``A+B``，前向却仍然乘 ``b·log n``，所有纯元数据审计都会通过，而单因素
    对比在数据上悄悄变成多因子对比。这里用 :func:`lega.attention.probe_gate_factors`
    做一次微型前向逐参数扰动，把扰动后确实改变门控 logit 的因子与声明集合
    逐项对照；同时核对带门变体没有漏登记、容量匹配对照保留全部三个门控
    参数却只置零自己那一条证据通道。

    核对失败（缺依赖/无设备）也显式记入 ``problems``：不能悄悄跳过这一层，
    让"实现级未核对"看起来像"核对通过"。
    """
    rows = []
    try:
        from lega.attention import (
            _GATED,
            ALT_CONC_VARIANTS,  # noqa: E501
            PLACEBO_VARIANTS,
            SUPPLEMENTAL_GATE_FACTORS,
            probe_gate_factors,
        )
    except Exception as e:  # pragma: no cover - 依赖损坏只在极端环境发生
        problems.append(f"无法执行实现级单因素核对：{type(e).__name__}: {e}")
        return rows
    gate_table = GATE_FACTORS if isinstance(GATE_FACTORS, dict) else {}
    raw_table = (
        globals().get("RAW_GATE_FACTORS")
        if isinstance(globals().get("RAW_GATE_FACTORS"), dict)
        else {}
    )
    supplemental = (
        dict(SUPPLEMENTAL_GATE_FACTORS)
        if isinstance(SUPPLEMENTAL_GATE_FACTORS, dict)
        else {}
    )
    for variant in supplemental:
        if not isinstance(variant, str):
            problems.append(f"SUPPLEMENTAL_GATE_FACTORS 含非字符串变体键：{variant!r}")
    for variant, factors in supplemental.items():
        if isinstance(variant, str) and _gate_factor_set(factors) is None:
            problems.append(
                f"SUPPLEMENTAL_GATE_FACTORS[{variant!r}] 不是合法的因子集合："
                f"{factors!r}"
            )
        if isinstance(variant, str) and variant not in _GATED:
            problems.append(f"带门补充对照 {variant!r} 没有输出门，不应登记因子语义")
    all_params = ["gate_a", "gate_b", "gate_w"]
    for variant in sorted(set(_GATED)):
        probe = probe_gate_factors(variant)
        if probe.get("error"):
            problems.append(f"实现级单因素核对失败（{variant}）：{probe['error']}")
            continue
        if not probe.get("gated"):
            problems.append(
                f"带门变体 {variant!r} 的前向没有门控 logit，无法做单因素归因"
            )
            continue
        # B 通道的**参数化口径**也要实测：元数据（_USE_CONC/_USE_RAW_CONC）
        # 只能说明“打算用哪种口径”。raw 变体若实际仍算 log(n*max_p)，原始
        # 参数化网格会塌陷成归一化网格，单因素边的差异恒为 0。
        conc_probe = probe_conc_parameterization(variant)
        conc_form_measured = conc_probe.get("form")
        conc_form_expected = expected_conc_form(variant)
        if conc_form_measured is None:
            problems.append(
                f"无法实测 {variant!r} 的 B 通道参数化口径："
                f"{conc_probe.get('detail') or '未知原因'}"
            )
        elif conc_form_measured != conc_form_expected:
            drift_note = (
                "原始参数化网格的单因素边会退化成归一化口径，"
                "“参数化稳健性”的差异恒为 0"
                if CONC_FORM_RAW in (conc_form_measured, conc_form_expected)
                else "该变体的单因素对比不再只改变一个因子"
            )
            problems.append(
                f"{variant!r} 的 B 通道参数化口径实测为 "
                f"{conc_form_measured!r}，设计要求 {conc_form_expected!r}："
                f"{drift_note}"
            )
        effective = frozenset(probe.get("effective") or [])
        params = list(probe.get("params") or [])
        declared = {}
        if variant in gate_table:
            got = _gate_factor_set(gate_table.get(variant))
            if got is None:
                problems.append(f"GATE_FACTORS[{variant!r}] 不是合法的因子集合")
            else:
                declared["GATE_FACTORS"] = got
        if variant in raw_table:
            got = _gate_factor_set(raw_table.get(variant))
            if got is None:
                problems.append(f"RAW_GATE_FACTORS[{variant!r}] 不是合法的因子集合")
            else:
                declared["RAW_GATE_FACTORS"] = got
        null_component = probe.get("null_component")
        category = None
        if null_component is not None:
            category = "capacity_null"
            expected = frozenset("ABC") - frozenset({null_component})
            if sorted(params) != sorted(all_params):
                problems.append(
                    f"容量匹配对照 {variant!r} 应保留全部门控参数 "
                    f"{all_params}，实际 {params}，否则不再是参数匹配的三臂"
                )
            if sorted(null_component) not in (sorted("A"), sorted("B"), sorted("C")):
                problems.append(
                    f"容量匹配对照 {variant!r} 的置零组件非法：{null_component!r}"
                )
        elif variant in PLACEBO_VARIANTS:
            category = "placebo"
            expected = frozenset("ABC")
            if sorted(params) != sorted(all_params):
                problems.append(
                    f"参数匹配安慰剂 {variant!r} 应保留全部门控参数 "
                    f"{all_params}，实际 {params}"
                )
        elif variant in ALT_CONC_VARIANTS:
            category = "alt_conc"
            expected = frozenset("ABC")
            if sorted(params) != sorted(all_params):
                problems.append(
                    f"替代集中度 {variant!r} 应保持 lega 的门控参数形状 "
                    f"{all_params}，实际 {params}"
                )
        elif variant in supplemental:
            category = "supplemental_gate"
            expected = supplemental[variant]
            # 这里核对的是“安慰剂表”本身（PLACEBO_VARIANTS），而不是整个
            # CONTROL_VARIANTS 类别：容量匹配对照（Null_X）已由上面的分支
            # 单独归类，而方法设计维度对照（lega_shared/qgate_null）本就属于
            # CONTROL_VARIANTS，它们作为带门补充对照登记 A/B/C 语义是合法的，
            # 不能因为同属对照类别就被判成“混入安慰剂表”。
            if (
                variant in gate_table
                or variant in raw_table
                or variant in PLACEBO_VARIANTS
            ):
                problems.append(
                    f"带门补充对照 {variant!r} 不得混入主/原始因子表或" "安慰剂表"
                )
        elif declared:
            table_label = (
                "GATE_FACTORS" if variant in gate_table else "RAW_GATE_FACTORS"
            )
            category = "gate_factors" if variant in gate_table else "raw_gate_factors"
            expected = declared[table_label]
            for label, factors in declared.items():
                if factors != expected:
                    problems.append(
                        f"{label} 与另一张因子表对 {variant!r} 的声明不一致："
                        f"{sorted(factors)} vs {sorted(expected)}"
                    )
        else:
            problems.append(
                f"带门变体 {variant!r} 未登记到任何因子表"
                "（GATE_FACTORS/RAW_GATE_FACTORS/容量匹配/安慰剂/替代集中度），"
                "不能静默参与单因素归因"
            )
            continue
        if effective != expected:
            problems.append(
                f"{variant!r} 实现层生效因子为 {sorted(effective) or '无'}，"
                f"声明为 {sorted(expected) or '无'}：单因素对比可能实际改变了"
                "多个/过少的门控因子"
            )
        if probe.get("bias_effective") is not True:
            problems.append(
                f"{variant!r} 的门控偏置 gate_d 未生效，输出门可能未接入前向"
            )
        rows.append(
            {
                "variant": variant,
                "category": category,
                "params": params,
                "effective": sorted(effective),
                "declared": sorted(expected),
                "null_component": null_component,
                "use_raw_conc": bool(probe.get("use_raw_conc")),
                "conc_mode": probe.get("conc_mode"),
                "conc_form_measured": conc_form_measured,
                "conc_form_expected": conc_form_expected,
                "ok": bool(
                    effective == expected
                    and probe.get("bias_effective") is True
                    and conc_form_measured is not None
                    and conc_form_measured == conc_form_expected
                ),
            }
        )
    return rows


def implementation_single_factor_check(audit, candidate, reference):
    """按审计的**实现层**实测结果，核对一条对比在数据上是否真的是单因素。

    元数据层（因子表 + 对比表）说“只改一个因子”还不够：变体前向若悄悄
    多乘/少乘了一个门控项，声明与实现会一起失真，任何纯元数据审计都发现
    不了。这里用审计的实测生效因子集合再核对一次：

    * 两侧变体都必须出现在实现层核对里，且该变体的核对本身通过；
    * 两侧**实测**生效因子集合的对称差必须恰好是一个因子。

    返回 ``(ok, reason)``。``ok`` 为 True（核对通过）/ False（核对不通过，
    reason 说明原因）/ None（审计里没有该变体的实现层记录，即“未核对”）。
    “未核对”不等于“违规”，但也不能当成已核对：调用方据此决定是否阻断。
    """
    rows = audit.get("implementation_factors") if isinstance(audit, dict) else None
    if not isinstance(rows, list):
        return None, "单因素审计没有实现层核对记录"
    by_variant = {row.get("variant"): row for row in rows if isinstance(row, dict)}
    pairs = (
        (candidate, by_variant.get(candidate)),
        (reference, by_variant.get(reference)),
    )
    missing = [str(v) for v, row in pairs if row is None]
    if missing:
        return False, "实现层核对缺少变体：" + "、".join(missing)
    failed = [str(v) for v, row in pairs if not row.get("ok")]
    if failed:
        details = []
        for name, row in pairs:
            if str(name) in failed:
                details.append(
                    f"{name}（实测生效因子 {row.get('effective')}，"
                    f"声明 {row.get('declared')}，"
                    f"B 通道口径实测 {row.get('conc_form_measured')}／"
                    f"设计 {row.get('conc_form_expected')}）"
                )
        return False, "实现层核对未通过：" + "、".join(details)
    cand_eff = frozenset(by_variant[candidate].get("effective") or [])
    ref_eff = frozenset(by_variant[reference].get("effective") or [])
    diff = cand_eff.symmetric_difference(ref_eff)
    if len(diff) != 1:
        return False, (f"实测生效因子差为 {sorted(diff) or '无'}，不是恰好一个因子")
    return True, ""


def _audit_lesion_variant_effects(problems):
    """逐变体核对推理期病灶**实际**影响的证据通道，并识别重复干预。

    ``LESION_EXPECTED_CHANNELS`` 是**变体无关**的声明：``no_evidence`` 声明
    影响 A/B/C 三条通道。但变体实际拥有的通道不同（``qgate`` 只有 A，
    ``gate_only`` 一条都没有），病灶只能作用在它实际拥有的通道上。
    :func:`_audit_lesion_implementation` 的通道隔离测量要求变体同时具备
    A/B/C，因此只在 ``lega`` 上核对——其余变体的实际通道集合从未被核对。

    不核对会留下两类真实缺陷：

    * **多通道对照退化成单通道**：``qgate`` 上 ``no_evidence`` 实际只清零
      ``gate_w``（A），却仍标着“去全部证据项 A+B+C”；读者会把单通道差异
      误读成整体证据，或把它当成 A 的第二条单因素证据。
    * **同一个干预被算成两次**：``qgate`` 上 ``no_evidence`` 与
      ``no_query`` 的参数级操作完全相同，ΔNLL 恒等，却被当成两个独立
      检验放进同一个 Holm 家族。

    这里对每个带门变体逐病灶做**参数级**实测（快照 → 施加 → 逐参数比对），
    核对实测通道是声明通道的子集（允许因变体缺通道而更小），并把同一变体
    上签名相同的重复干预显式列出。重复干预本身不是实现错误（它由变体结构
    决定），但必须在结果层被识别并排除出 Holm 家族。
    """
    rows = []
    duplicates = []
    lesion_order = globals().get("ABLATION_LESION_ORDER")
    try:
        from lega.ablate import (
            LESION_BY_ID,
            lesion_intervention_signature,
            probe_lesion_param_effects,
        )
        from lega.attention import _GATED
    except Exception as e:  # pragma: no cover - 依赖损坏只在极端环境发生
        problems.append(f"无法执行推理期病灶逐变体核对：{type(e).__name__}: {e}")
        return rows, duplicates
    if not isinstance(lesion_order, (tuple, list)) or isinstance(lesion_order, str):
        problems.append("LESION_ORDER 必须是病灶序列")
        return rows, duplicates
    if not isinstance(_GATED, (tuple, list)):
        problems.append("_GATED 必须是带门变体序列")
        return rows, duplicates
    for variant in sorted(_GATED):
        seen = {}
        for lesion_id in lesion_order:
            spec = LESION_BY_ID.get(lesion_id)
            if not isinstance(spec, dict):
                problems.append(f"推理期病灶 {lesion_id!r} 未定义")
                continue
            probe = probe_lesion_param_effects(lesion_id, variant=variant)
            if probe.get("error"):
                problems.append(
                    f"推理期病灶 {lesion_id!r} 在变体 {variant!r} 上的参数级核对"
                    f"失败：{probe['error']}"
                )
                continue
            measured = tuple(probe.get("affected") or ())
            declared = tuple(probe.get("declared") or ())
            expected = tuple(probe.get("expected") or ())
            if not set(measured).issubset(set(declared)):
                problems.append(
                    f"推理期病灶 {lesion_id!r} 在变体 {variant!r} 上实际影响通道 "
                    f"{list(measured)}，超出设计声明 {list(declared)}："
                    "单因素推理期实验已被实现漂移破坏"
                )
            elif measured != expected:
                problems.append(
                    f"推理期病灶 {lesion_id!r} 在变体 {variant!r} 上实测影响通道 "
                    f"{list(measured)} 与按变体裁剪的元数据口径 {list(expected)} "
                    "不一致"
                )
            signature = lesion_intervention_signature(
                lesion_id, probe.get("capabilities") or {}
            )
            # 只有**真的被施加**的病灶才是实验；不适用（缺能力）的病灶没
            # 有任何干预，不能与别的病灶构成“重复干预”。2^3 的格
            # （query_only/conc_only/len_only）不进因果表，也不参与重复
            # 判定。
            counted = bool(probe.get("applied")) and not spec.get("factorial_cell")
            first = seen.get(signature) if counted else None
            if counted and first is None:
                seen[signature] = lesion_id
            elif counted:
                duplicates.append(
                    {
                        "variant": variant,
                        "lesion": lesion_id,
                        "duplicate_of": first,
                        "changed_factors": list(measured),
                        "reason": (
                            "同一变体上施加的是同一个干预（参数级操作完全相同），"
                            "ΔNLL 恒等；结果层必须把它排除出 Holm 家族"
                        ),
                    }
                )
            rows.append(
                {
                    "variant": variant,
                    "lesion": lesion_id,
                    "applied": bool(probe.get("applied")),
                    "changed_factors": list(measured),
                    "declared_channels": list(declared),
                    "expected_channels": list(expected),
                    # 变体缺通道时，多通道对照实际只作用在它有的通道上。
                    "degraded": bool(
                        measured and set(measured) != set(declared)
                    ),
                    "ok": bool(
                        set(measured).issubset(set(declared))
                        and measured == expected
                    ),
                }
            )
    return rows, duplicates


def _audit_lesion_implementation(problems):
    """核对推理期病灶在**实现层**只动它声明的证据通道。

    训练期变体已有 :func:`lega.attention.probe_gate_factors` 做实现层自证；
    推理期病灶此前只有元数据核对（``LESION_SPECS`` 的 kind/params/requires）。
    元数据能发现“声明错了”，但发现不了“实现顺手多动了一条通道”：例如把
    ``len_permute`` 写成同时错位 ``c`` 里的 log n、把 ``no_query`` 额外清零
    gate_a，或让 ``query_permute`` 也置换集中度证据。这类漂移会让单因素
    推理期实验悄悄变成多因素实验，而所有纯元数据检查仍然全绿。

    这里用 :func:`lega.ablate.probe_lesion_channels` 做一次确定性微型前向的
    逐通道隔离测量（把另外两条通道的门控系数置零，比较病灶施加前后的
    位置变化），把**实测**影响通道与**声明**通道逐项对照：

    * 单因素病灶（no_query/no_conc/no_len、三个活动匹配对照、三个分布匹配
      置换）必须只动一条通道；
    * 多通道对照（去全部证据、门恒开/恒关、恒为均值、头置换、证据反号）
      必须确实覆盖 A/B/C 三条，不能退化成“只动一条”；
    * 2^3 的单因子格（query_only/conc_only/len_only）必须恰好去掉另外两条
      通道，否则推理期 Shapley/交互分解的格映射就是错的。

    探针失败（缺依赖/无设备）显式记入 ``problems``：不能悄悄跳过这一层，
    让“实现级未核对”看起来像“核对通过”。
    """
    rows = []
    lesion_order = globals().get("ABLATION_LESION_ORDER")
    try:
        from lega.ablate import (
            LESION_EXPECTED_CHANNELS,
            probe_lesion_channels,
        )
    except Exception as e:  # pragma: no cover - 依赖损坏只在极端环境发生
        problems.append(f"无法执行推理期病灶实现层核对：{type(e).__name__}: {e}")
        return rows
    if not isinstance(lesion_order, (tuple, list)) or isinstance(lesion_order, str):
        problems.append("LESION_ORDER 必须是病灶序列")
        return rows
    expected_table = LESION_EXPECTED_CHANNELS
    if not isinstance(expected_table, dict):
        problems.append("LESION_EXPECTED_CHANNELS 必须是病灶 -> 通道映射")
        expected_table = {}
    for lesion_id in lesion_order:
        expected = expected_table.get(lesion_id)
        if not isinstance(expected, (tuple, list)) or not expected:
            problems.append(f"推理期病灶 {lesion_id!r} 没有声明应影响的证据通道")
            continue
        probe = probe_lesion_channels(lesion_id)
        if not isinstance(probe, dict):
            problems.append(f"推理期病灶 {lesion_id!r} 的实现层核对返回非法结构")
            continue
        if probe.get("error"):
            problems.append(
                f"推理期病灶实现层核对失败（{lesion_id}）：{probe['error']}"
            )
            continue
        measured = tuple(probe.get("affected") or ())
        expected = tuple(expected)
        ok = measured == expected
        if not ok:
            problems.append(
                f"推理期病灶 {lesion_id!r} 实现层实际影响通道 "
                f"{list(measured) or '无'}，声明为 {list(expected) or '无'}："
                "单因素推理期实验已被实现漂移破坏"
            )
        rows.append(
            {
                "lesion": lesion_id,
                "expected": list(expected),
                "affected": list(measured),
                "applied": bool(probe.get("applied")),
                "ok": ok,
            }
        )
    return rows


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
    if not isinstance(alt, (tuple, list)) or isinstance(alt, str):
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
            f"stats={sorted(modes)}，variants={sorted(set(alt))}"
        )
    for variant in alt:
        if not isinstance(variant, str):
            problems.append(f"ALT_CONC_VARIANTS 含非字符串变体：{variant!r}")
            continue
        mode = modes.get(variant)
        if mode not in ("l2", "entropy"):
            problems.append(f"替代集中度 {variant!r} 的统计量 {mode!r} 未实现")
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
                "它只替换 B 的统计量，不能当作另一个结构格"
            )
        if variant in RAW_GATE_FACTORS:
            problems.append(f"替代集中度 {variant!r} 混入了原始参数化因子表")
        if (
            variant not in _USE_QUERY
            or variant not in _USE_CONC
            or variant not in _USE_LEN
        ):
            problems.append(f"替代集中度 {variant!r} 未保持 lega 的 A/B/C 门控结构")
        if variant not in _GATED or variant not in _LEGA_FAMILY:
            problems.append(f"替代集中度 {variant!r} 未启用输出门或注意力统计")
        if variant in _USE_RAW_CONC:
            problems.append(f"替代集中度 {variant!r} 同时启用了原始 B 参数化")
        rows.append(
            {
                "variant": variant,
                "mode": mode,
                "replaced_factor": "B",
                "reference": "lega",
            }
        )
    return rows


def _audit_variant_capability_clipping(problems):
    """核对按变体裁剪推理期病灶所用的能力映射与实现层一致。

    结果层（``_ablation_lesion_records``/``_matched_ablation_contrasts``）用
    :func:`lega.attention.variant_gate_capabilities`（按变体名与实现开关推导）
    决定一个病灶在该变体上**实际**影响哪几条证据通道；审计层则用
    :func:`lega.ablate.probe_lesion_param_effects`（快照→施加→逐参数实测）
    核对。两条路径若漂移——例如名字推导多报一条通道——结果层会把
    ``qgate`` 上的 ``no_evidence`` 标成多因素 A+B+C、漏掉它与 ``no_query``
    的重复干预；而探针审计因为走的是另一条路径仍然全绿。这里逐带门变体把
    “名字推导的通道集合”与“实测注册的 gate 参数所对应的通道集合”对齐，
    保证单因素推理期实验的裁剪口径只有一套事实来源。
    """
    rows = []
    try:
        from lega.ablate import _GATE_PARAM_CHANNEL
        from lega.attention import _GATED, probe_gate_factors
    except Exception as e:  # pragma: no cover - 依赖损坏只在极端环境发生
        problems.append(f"无法核对变体能力裁剪映射：{type(e).__name__}: {e}")
        return rows
    name_fn = globals().get("variant_gate_capabilities")
    if not callable(name_fn):
        problems.append("缺少 variant_gate_capabilities，无法核对推理期通道裁剪")
        return rows
    if not isinstance(_GATED, (tuple, list)):
        problems.append("_GATED 必须是带门变体序列")
        return rows
    channel_caps = (("A", "query"), ("B", "conc"), ("C", "len"))
    for variant in sorted(set(_GATED)):
        named_caps = name_fn(variant)
        if not isinstance(named_caps, dict):
            problems.append(
                f"variant_gate_capabilities({variant!r}) 返回非法结构："
                f"{named_caps!r}"
            )
            continue
        named = {c for c, key in channel_caps if named_caps.get(key)}
        probe = probe_gate_factors(variant)
        if probe.get("error"):
            problems.append(
                f"无法实测变体 {variant!r} 的注册门控参数：{probe['error']}"
            )
            continue
        impl = {
            _GATE_PARAM_CHANNEL[name]
            for name in (probe.get("params") or [])
            if name in _GATE_PARAM_CHANNEL
        }
        ok = named == impl
        if not ok:
            problems.append(
                f"变体 {variant!r} 的能力裁剪口径不一致："
                f"variant_gate_capabilities 给出 {sorted(named) or '无'}，"
                f"实测注册参数对应 {sorted(impl) or '无'}："
                "推理期病灶按变体裁剪会失真（多通道对照会被误标成单因素或"
                "漏掉重复干预）"
            )
        rows.append(
            {
                "variant": variant,
                "named_channels": sorted(named),
                "implementation_channels": sorted(impl),
                "ok": ok,
            }
        )
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
                f"GATE_FACTORS[{variant!r}] 不是合法的因子集合：" f"{raw_factors!r}"
            )
            continue
        unknown = factors - frozenset("ABC")
        if unknown:
            problems.append(f"GATE_FACTORS[{variant!r}] 含未知因子：{sorted(unknown)}")
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
            + (
                ", ".join(sorted("".join(sorted(s)) or "none" for s in missing_subsets))
                or "无"
            )
            + "；多出 "
            + (
                ", ".join(sorted("".join(sorted(s)) or "none" for s in extra_subsets))
                or "无"
            )
        )
    # 全因子矩阵的变体集合必须与 GATE_FACTORS 完全一致；否则汇总会拿
    # 未登记因子语义的变体当 2^3 的一格。
    factorial_variants = []
    if not isinstance(FACTORIAL_CELLS, (tuple, list)):
        problems.append("FACTORIAL_CELLS 必须是 (名称, 变体) 序列")
    else:
        for entry in FACTORIAL_CELLS:
            if (
                not isinstance(entry, (tuple, list))
                or len(entry) != 2
                or not isinstance(entry[1], str)
            ):
                problems.append(f"全因子格定义结构非法：{entry!r}")
                continue
            factorial_variants.append(entry[1])
    if len(set(factorial_variants)) != 8 or set(factorial_variants) != set(normalized):
        problems.append(
            "全因子 8 格与 GATE_FACTORS 不一致："
            f"cells={sorted(set(factorial_variants))}，"
            f"factors={sorted(normalized)}"
        )
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
                elif (
                    isinstance(cell, str)
                    and cell
                    and len(set(cell)) == len(cell)
                    and all(ch in "ABC" for ch in cell)
                ):
                    cell_factors = frozenset(cell)
                else:
                    cell_factors = None
                if cell_factors is None or cell_factors not in _GATE_FACTOR_SUBSETS:
                    problems.append(f"全因子格名非法：{cell!r}")
                    continue
                if not isinstance(variant, str):
                    problems.append(f"全因子格 {cell!r} 的变体名非法：{variant!r}")
                    continue
                cell_variants.add(variant)
                if normalized.get(variant) != cell_factors:
                    problems.append(
                        f"全因子格 {cell!r} -> {variant!r} 的因子集合为 "
                        f"{sorted(normalized.get(variant, ()))}，"
                        f"应为 {sorted(cell_factors)}"
                    )
            if cell_variants != set(normalized):
                problems.append(
                    "交互检验的全因子变体集合与 GATE_FACTORS 不一致："
                    f"cells={sorted(cell_variants)}，"
                    f"factors={sorted(normalized)}"
                )
    # 审计表必须恰好覆盖 A/B/C 的“加入/去掉”六条单因素边。只逐条校验
    # 已列出的对比会漏掉“整条对比被删除/重复”这种漂移：表少一行时仍
    # 全部合法，却不再覆盖某个组件的归因。
    required = {
        (factor, direction)
        for factor in ("A", "B", "C")
        for direction in ("add", "remove")
    }
    seen = set()
    seen_pairs = {}
    if not isinstance(SINGLE_FACTOR_CONTRASTS, (tuple, list)) or isinstance(
        SINGLE_FACTOR_CONTRASTS, str
    ):
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
            problems.append(f"重复的单因素对比声明：{factor}/{direction}")
        seen.add(edge)
        if cand == ref:
            problems.append(f"单因素对比两侧相同：{cand} vs {ref}")
            continue
        if (cand, ref) in seen_pairs:
            problems.append(
                f"重复的单因素对比对：{cand} vs {ref}"
                f"（已声明为 {seen_pairs[(cand, ref)][0]}/"
                f"{seen_pairs[(cand, ref)][1]}）"
            )
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
                f"{sorted(diff) or '无'}"
            )
            continue
        # 方向不只是展示字段：remove 必须从参照中恰好去掉该因子，add 必须
        # 在参照上恰好加入该因子。写反方向会让“加入/去掉”的效应解释颠倒。
        if direction == "remove":
            direction_ok = c < r and (r - c) == frozenset({factor})
            expected_shape = f"{ref} 去掉 {factor} 得到 {cand}"
        else:
            direction_ok = r < c and (c - r) == frozenset({factor})
            expected_shape = f"{ref} 加入 {factor} 得到 {cand}"
        if not direction_ok:
            problems.append(
                f"{cand} vs {ref} 声明方向 {direction} 与因子变化不符："
                f"应为 {expected_shape}"
            )
            continue
        contrasts.append(
            {
                "factor": factor,
                "direction": direction,
                "candidate": cand,
                "reference": ref,
                "changed_factors": sorted(diff),
            }
        )
    missing = required - seen
    extra = seen - required
    if missing:
        problems.append(
            "缺少单因素对比："
            + ", ".join(
                f"{factor}/{direction}" for factor, direction in sorted(missing)
            )
        )
    if extra:
        problems.append(
            "存在非 A/B/C 单因素声明的对比："
            + ", ".join(f"{factor}/{direction}" for factor, direction in sorted(extra))
        )
    # 汇总/报告实际消费的配对表必须与已声明的 6 条边逐项一致；否则
    # 手工增删 SINGLE_FACTOR_PAIRS 就能让多因子对比混进汇总配对表。
    single_factor_pairs = _audit_single_factor_pair_table(contrasts, problems)
    audited_pairs = set(single_factor_pairs)
    # 组件判定实际消费的检验对也必须落在已审计表内；否则改 COMPONENT_TESTS
    # 就能绕开审计去使用多因子对比（运行期而非仅测试期拦截）。
    audited_edges = {
        (c["candidate"], c["reference"]): (c["factor"], c["direction"])
        for c in contrasts
    }
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
            problems.append(f"组件名 {name!r} 无法映射到 A/B/C，不能作为单组件检验")
            expected = None
        else:
            expected = token
            if expected in seen_components:
                problems.append(f"组件 {expected} 存在重复的检验定义")
            seen_components.add(expected)
        if primary not in _EVIDENCE_METRICS:
            problems.append(f"组件 {name!r} 的主指标 {primary!r} 不是已登记指标")
        for label, pair, expected_direction in (
            ("去一法", loo_pair, "remove"),
            ("全因子", add_pair, "add"),
        ):
            if (
                not isinstance(pair, (tuple, list))
                or len(pair) != 2
                or not all(isinstance(x, str) for x in pair)
            ):
                problems.append(f"组件 {name!r} 的{label}对比结构非法：{pair!r}")
                continue
            if tuple(pair) not in audited_pairs:
                problems.append(
                    f"组件 {name!r} 的{label}对比 {pair!r} 不在"
                    " SINGLE_FACTOR_PAIRS 中"
                )
            edge = audited_edges.get(tuple(pair))
            if edge is None:
                problems.append(f"组件 {name!r} 使用了未审计的单因素对比：{pair!r}")
                continue
            factor, direction = edge
            if expected is not None and factor != expected:
                problems.append(
                    f"组件 {name!r} 的对比 {pair!r} 实际单因素为 {factor}，"
                    f"与组件 {expected} 不一致"
                )
            if direction != expected_direction:
                problems.append(
                    f"组件 {name!r} 的{label}对比 {pair!r} 方向为 {direction}，"
                    f"应为 {expected_direction}"
                )
    missing_components = {"A", "B", "C"} - seen_components
    if missing_components:
        problems.append("缺少组件检验：" + ", ".join(sorted(missing_components)))
    reparameterization_contrasts = _audit_reparameterization_contrasts(
        normalized, problems
    )
    dose_components = _audit_dose_response_components(problems)
    capacity_components = _audit_capacity_triple_components(problems, normalized)
    matched_lesions, permutation_lesions, inference_cells = (
        _audit_inference_single_factor_design(problems)
    )
    alt_conc = _audit_alt_conc_variants(problems)
    # 实现层核对先算：登记假设的单因素边要拿它做第二道关（元数据说“只改
    # 一个因子”但实现悄悄多改了一项时，确认性判定必须失效）。
    implementation_factors = _audit_implementation_factors(normalized, problems)
    registered_single_factor = _audit_registered_single_factor(
        contrasts, problems, implementation_factors
    )
    variant_capabilities = _audit_variant_capability_clipping(problems)
    lesion_implementation = _audit_lesion_implementation(problems)
    lesion_variant_effects, lesion_duplicates = _audit_lesion_variant_effects(problems)
    forbidden_contrasts = _forbidden_multi_factor_contrasts(
        registered_single_factor, problems
    )
    return {
        "ok": not problems,
        "problems": problems,
        "contrasts": contrasts,
        "pairs": [list(p) for p in single_factor_pairs],
        "factors": {v: sorted(f) for v, f in sorted(normalized.items())},
        "reparameterization_contrasts": reparameterization_contrasts,
        "dose_response_components": dose_components,
        "capacity_components": capacity_components,
        "inference_matched_lesions": matched_lesions,
        "inference_permutation_lesions": permutation_lesions,
        "inference_factorial_cells": inference_cells,
        "alt_conc_contrasts": alt_conc,
        "registered_single_factor": registered_single_factor,
        "implementation_factors": implementation_factors,
        "inference_variant_capabilities": variant_capabilities,
        "inference_lesion_implementation": lesion_implementation,
        "inference_lesion_variant_effects": lesion_variant_effects,
        "inference_lesion_duplicate_interventions": lesion_duplicates,
        "forbidden_multi_factor_contrasts": forbidden_contrasts,
        "note": (
            "单因素对比 = 去一法 Full−X vs Full，或全因子增量 "
            "X-only vs gate_only；整体对比（vs standard、LEGA vs "
            "qgate）可跨多个因子，不作组件归因。同一审计还核对 "
            "归一化/原始参数化的单因素边、剂量-反应的 A/B/C 参数"
            "映射、训练期容量匹配 Full/Null_X/No_X 三臂、推理期"
            "单病灶、活动匹配、A/B/C 分布匹配置换、2^3 格的因子"
            "映射，以及替代集中度统计的 B-only 构念对照"
            "（lega_l2/lega_entropy 不进入主因子表）。训练期变体"
            "与推理期病灶都做实现层自证：变体看逐参数扰动是否真的"
            "改变门控 logit，病灶看逐通道隔离测量是否只动声明的"
            "证据通道。门控网格内的多因子对比（B-only/C-only 与 "
            "Full）默认禁止：它们既不是整体方法对比，也不能用于"
            "组件归因，只有显式白名单（当前仅 LEGA vs qgate）允许。"
        ),
    }


def _forbidden_multi_factor_contrasts(registered_rows, problems):
    """汇总“禁止进入确认性判定”的门控网格内多因子对比。

    登记表里若出现 B-only/C-only 与 Full 这类同时改两个因子的对比，
    审计必须显式点名：这类对比不是整体方法对比，也不能当单组件证据，
    汇总层会把它挡在配对表之外，并把对应假设降级为不可判定。
    """
    rows = []
    for row in registered_rows if isinstance(registered_rows, list) else []:
        if not isinstance(row, dict) or not row.get("forbidden"):
            continue
        rows.append(
            {
                "id": row.get("id"),
                "candidate": row.get("candidate"),
                "reference": row.get("reference"),
                "changed_factors": list(row.get("changed_factors") or []),
                "reason": "门控网格内的多因子对比：不是整体方法对比，也不能用于组件归因",
            }
        )
    return rows



def build_comparison_pairs(runs, registered=None):
    """构造汇总用的同种子配对对比表。

    刻意把“单因素消融”和“整体方法对比”分开：
      * 单因素：SINGLE_FACTOR_PAIRS（去一法 3 组 + 全因子增量 3 组）；
      * 整体：各变体 vs standard，以及预注册的其它整体对比（如 H5）；
      * 不再把 lega_conc_only / lega_len_only 与 lega 相比——它们与 Full
        同时差两个因子（A+C / A+B），不是单组件消融。
    """
    runs = runs if isinstance(runs, dict) else {}
    pairs = [(cand, "standard") for cand in ALL_VARIANTS if cand != "standard"]
    # 补充对照（raw/detach/null_zero）不在默认矩阵内，但显式运行时同样是
    # “相对 standard”的整体对比；漏掉会让汇总表出现数值却没有任何配对
    # 检验，也与设计层 planned_comparison_pairs（按实际计划变体生成）不一致。
    pairs += [(cand, "standard") for cand in SUPPLEMENTAL_VARIANTS if cand in runs]
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
    pairs += [
        (cand, ref)
        for cand, ref in SINGLE_FACTOR_PAIRS
        if ref != "gate_only" or "gate_only" in runs
    ]
    registered = _as_dict(registered)
    hypotheses = registered.get("hypotheses")
    for h in hypotheses if isinstance(hypotheses, list) else []:
        if not isinstance(h, dict):
            continue
        c, r = h.get("candidate"), h.get("reference")
        if h.get("metric") in (AUROC_DIFF_METRIC, AP_DIFF_METRIC):
            continue  # 单样本机制族，由 _auroc_diff_tests/_ap_diff_tests 覆盖
        if c in RUNNABLE_VARIANTS and r in RUNNABLE_VARIANTS:
            # 门控网格内的多因子对比（B-only/C-only 与 Full 同时改两个
            # 因子）不进配对表：它不是整体方法对比，也不能当单组件证据。
            # 登记成假设也不能绕开这条约束——对应假设会被单因素门降级为
            # 不可判定，并在单因素审计里显式点名。
            if grid_contrast_info(c, r)["forbidden"]:
                continue
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
    for k in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
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
        rcs[i] = subprocess.run(cmd, cwd=ROOT, env=_cmd_env(cmd, env)).returncode

    with ThreadPoolExecutor(max_workers=min(jobs, len(cmds))) as ex:
        list(ex.map(lambda t: _one(*t), enumerate(cmds)))
    for cmd, rc in zip(cmds, rcs):
        if rc != 0:
            raise subprocess.CalledProcessError(rc, cmd)


def _repo_path(path: str) -> str:
    """相对路径统一解释为仓库根目录；父进程与子进程必须看到同一位置。"""
    return path if os.path.isabs(path) else os.path.abspath(os.path.join(ROOT, path))


def mark_repro_failed(
    results_dir: str,
    *,
    error: str,
    stage: str,
    variant: str = None,
    returncode: int = None,
) -> bool:
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
        start = time.mktime(
            time.strptime(str(ledger_common.get("time_start")), "%Y-%m-%d %H:%M:%S")
        )
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


def collect_determinism_audit(out_root: str, variants, seeds) -> dict:
    """汇总本轮各 run 的确定性审计结论（档位 + 非确定性算子告警数）。

    逐 run 的 ``run_manifest.json`` 里有训练进程自己的 ``seed_report``：
    确定性档位（strict/warn_only）、本轮实际触发过的“无确定性实现算子”
    次数、以及进程哈希盐是否固定。汇总页必须展示**实测**结论——“开关已
    打开”不等于“本轮没有踩到非确定性 kernel”。

    缺失/损坏的清单只跳过，不让审计本身失败：可复现性报告不应因为某个
    run 目录被手工删掉而整页崩溃。
    """
    variants = list(variants or [])
    seeds = list(seeds or [])
    n_seeds = len(seeds)
    modes = {}
    nondet_total = 0
    unfixed = []
    runs = 0
    for variant in variants:
        for seed in seeds:
            try:
                vdir = _variant_dir(out_root, variant, seed, n_seeds)
            except (TypeError, ValueError):
                continue
            path = os.path.join(vdir, "run_manifest.json")
            try:
                with open(path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
            except (OSError, ValueError):
                continue
            report = manifest.get("seed_report") if isinstance(manifest, dict) else None
            if not isinstance(report, dict):
                continue
            runs += 1
            mode = report.get("deterministic_mode")
            if mode:
                modes[mode] = modes.get(mode, 0) + 1
            count = report.get("non_deterministic_op_warnings")
            if isinstance(count, int) and not isinstance(count, bool):
                nondet_total += count
            if report.get("pythonhashseed_fixed") is False:
                unfixed.append(f"{variant}/seed{seed}")
    return {
        "runs": runs,
        "modes": dict(sorted(modes.items())),
        "non_deterministic_op_warnings": nondet_total,
        "pythonhashseed_unfixed_runs": unfixed,
    }


def append_aborted_record(
    ledger: str,
    ledger_common: dict,
    *,
    error: str,
    stage: str,
    variant: str = None,
    returncode: int = None,
    sanity: dict = None,
    runs: list = None,
) -> None:
    """失败轮次的终态记录：附失败阶段/原因/退出码，失败实验也完整留痕。

    runs（可选）为失败前已产生的逐 run 证据快照，写入 aborted 记录；
    不传时保持旧行为（只有失败原因，不伪造 run 覆盖）。
    """
    extra = {"error": error, "stage": stage}
    if variant is not None:
        extra["variant"] = variant
    if returncode is not None:
        extra["returncode"] = int(returncode)
    append_record(
        ledger,
        build_experiment_record(
            status="aborted",
            time_end=time.strftime("%Y-%m-%d %H:%M:%S"),
            sanity=sanity,
            runs=runs,
            extra=extra,
            **ledger_common,
        ),
    )
    # 台账 aborted 与全局 repro 的失败终态必须一起落盘；后者写失败不能
    # 影响前者的完整性，也不能掩盖调用方即将抛出的原始异常。
    try:
        mark_repro_failed(
            ledger_common.get("results_dir"),
            error=error,
            stage=stage,
            variant=variant,
            returncode=returncode,
        )
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
                ledger,
                ledger_common,
                error="postrun_stage_failed",
                stage=str(stage_name),
                returncode=getattr(e, "returncode", None),
                runs=runs or None,
            )
        except Exception as ledger_err:
            print(f"[repro] ⚠ aborted 台账写入失败（{ledger_err}）；" f"原异常仍将抛出")
        raise


def record_aborted_sanity(
    ledger: str,
    ledger_common: dict,
    variant: str,
    returncode: int,
    results_dir: str,
    reason: str = None,
) -> dict:
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
        ledger,
        ledger_common,
        error="sanity_gate_failed",
        stage="sanity",
        variant=variant,
        returncode=returncode,
        sanity={
            "executed": True,
            "verdict": {variant: "fail"},
            "reason": reason,
            "failed": failed,
        },
    )
    return failed


def check_sanity_results(
    variants, results_dir: str, not_before: float = None, expected_seed: int = None
):
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
        elif summary.get("variant") is not None and str(summary.get("variant")) != str(
            v
        ):
            reason = "variant_mismatch"
        elif expected_seed is not None and summary.get("seed") is not None:
            try:
                if strict_int(summary.get("seed"), "sanity.seed") != int(expected_seed):
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
            failures.append(
                {"variant": v, "reason": reason, "summary": summary, "path": path}
            )
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
        if re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", s) is None:
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
    return {k: d[k] for k in _PPL_PROTOCOL_KEYS if k in d and d[k] is not None}


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
    keys = (
        "ctx_lens",
        "n_samples",
        "sample_seed",
        "corpus_source",
        "corpus_held_out",
        "max_new",
        "fwe_max_new",
        "vt_chains",
        "vt_depth",
        "amp",
        "depth_stratified",
        "samples_manifest_sha256",
    )
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
        if (
            not isinstance(task, str)
            or task.startswith("_")
            or not isinstance(by_ctx, dict)
        ):
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
