"""结果分析质量层：假设诊断、证据强度分级、结论强度总览与分析人群核算。

本模块只负责“读结果”，不引入任何新的显著性判定：结论仍然只以同一指标
家族内 Holm 校正后的 ``p_adj`` 为准。它补的是读结果时最容易被跳过的四件事：

  1. **配对差的分布形态**是否可信。种子数只有 3~5 时正态假设无法检验，
     但偏度、厚尾与离群种子是可以显式记录的；这里给出启发式标记，并明确
     标注“小样本不可评估”，而不是默认“看起来没问题”。
  2. **一条“显著”结论的证据强度**。CI 是否跨 0、效应是否落在逐种子噪声
     内、置换检验是否一致、留一法是否翻转——把这些合成一个可比的等级，
     读者不必自己从几十列里拼装“这条结论有多硬”。
  3. **未显著结论的读法**。“不显著”既可能是“没有效应”，也可能是“功效
     不足”；本层按 MDE 区分这两者，并明确“未检出差异”不等于“差异为 0”。
  4. **整体结论强度**。全部确认性主张合成一个总览等级（强/中/弱/无/无法
     判定），避免结论只能由读者手工拼装。

所有函数都是确定性纯函数：不采样、不改 p 值、不参与 Holm/FWER 校正，
同一输入恒定复现。阈值均为可审计的启发式常数，不产生 p 值。
"""

import math

from .._shared import _install as _shared_install

_shared_install(globals())

# 形状诊断的最低观测数：n<8 时偏度/峰度的抽样误差远大于阈值本身，
# 任何“偏态/厚尾”的判断都没有功效，只能如实标成不可评估。
QA_MIN_SHAPE_N = 8

# 稳健离群点阈值（Iglewicz–Hoaglin 修正 z 分数）：0.6745·(x−median)/MAD。
QA_OUTLIER_Z = 3.5

# 偏度/峰度的启发式阈值（无检验、无 p 值）：|g1|>1 记为偏态，
# |超额峰度|>2 记为厚尾。
QA_SKEW_LIMIT = 1.0
QA_KURT_LIMIT = 2.0

QA_ASSUMPTION_LABELS = {
    "skewed": "配对差明显偏态",
    "heavy_tailed": "配对差厚尾",
    "outlier_driven": "配对差由离群种子主导",
}

QA_STRENGTH_LABELS = {
    "strong": "稳健",
    "marginal": "边缘",
    "fragile": "脆弱",
    "inconclusive": "功效不足，未定",
    "no_evidence": "未检出显著差异",
    "not_assessable": "无法判定",
}

QA_STRENGTH_SCORES = {
    "strong": 3,
    "marginal": 2,
    "fragile": 1,
    "inconclusive": 0,
    "no_evidence": 0,
    "not_assessable": None,
}


def _qa_num(value):
    """JSON 数值安全转换：bool/字符串/NaN/Inf 统一按缺失处理。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return out if math.isfinite(out) else None


def _qa_int(value):
    """整数安全转换：bool/浮点/非法值一律按缺失处理。"""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _qa_median(values):
    """中位数（偶数个取中间两数均值）。"""
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return None
    mid = n // 2
    if n % 2:
        return float(ordered[mid])
    return (float(ordered[mid - 1]) + float(ordered[mid])) / 2.0


def _qa_skew_kurt(values):
    """偏度（Fisher–Pearson 调整）与超额峰度；样本不足时返回 None。

    计算前先按最大绝对值缩放：偏度/峰度是尺度不变的，但同号巨大差值
    直接求三次/四次方会溢出成 inf，进而把形状诊断变成 NaN。
    """
    n = len(values)
    if n < 3:
        return None, None
    scale = max(abs(v) for v in values)
    if scale <= 0.0:
        return None, None
    scaled = [v / scale for v in values]
    mean = sum(scaled) / max(n, 1)
    m2 = sum((v - mean) ** 2 for v in scaled) / max(n, 1)
    if m2 <= 0.0:
        return None, None
    m3 = sum((v - mean) ** 3 for v in scaled) / max(n, 1)
    m4 = sum((v - mean) ** 4 for v in scaled) / max(n, 1)
    skew = None
    try:
        g1 = m3 / (m2 ** 1.5)
        skew = math.sqrt(n * (n - 1)) / max(n - 2, 1) * g1
        if not math.isfinite(skew):
            skew = None
    except (ZeroDivisionError, OverflowError, ValueError):
        skew = None
    kurt = None
    if n >= 4:
        try:
            g2 = m4 / (m2 ** 2) - 3.0
            kurt = ((n + 1) * g2 + 6) * (n - 1) / ((n - 2) * (n - 3))
            if not math.isfinite(kurt):
                kurt = None
        except (ZeroDivisionError, OverflowError, ValueError):
            kurt = None
    return skew, kurt


def _qa_outlier_indices(values):
    """基于中位数/MAD 的稳健离群点下标；MAD=0（多数值相同）时不给离群点。"""
    if len(values) < 4:
        return []
    med = _qa_median(values)
    if med is None:
        return []
    mad = _qa_median([abs(v - med) for v in values])
    if mad is None or mad <= 0.0:
        return []
    out = []
    for i, v in enumerate(values):
        z = 0.6745 * (v - med) / mad
        if math.isfinite(z) and abs(z) > QA_OUTLIER_Z:
            out.append(i)
    return out


def assumption_diagnostics(diffs, min_seeds=MIN_SEEDS) -> dict:
    """配对差的分布形态诊断（假设层，不做显著性判定）。

    返回：样本量、被丢弃的非法值数量、均值/标准差、偏度、超额峰度、离群点
    下标与计数、是否方差退化、形状是否可评估、启发式标记、状态与中文标签。

    状态取值：``no_data`` / ``insufficient_seeds`` / ``degenerate`` /
    ``not_assessable`` / ``plausible`` / ``skewed`` / ``heavy_tailed`` /
    ``outlier_driven``。

    重要限制：这些标记是启发式而非检验，不产生 p 值，不改变任何判定；
    n<8 时偏度/峰度几乎没有功效，必须读成“不可评估”，不能读成“无异常”。
    """
    raw = list(diffs) if isinstance(diffs, (list, tuple)) else []
    values = []
    dropped = 0
    for item in raw:
        num = _qa_num(item)
        if num is None:
            dropped += 1
            continue
        values.append(num)
    n = len(values)
    out = {
        "n": n,
        "n_dropped": dropped,
        "mean": None,
        "std": None,
        "skewness": None,
        "excess_kurtosis": None,
        "n_outliers": 0,
        "outlier_indices": [],
        "degenerate": False,
        "shape_assessable": False,
        "flags": [],
        "status": "no_data",
        "label": "无有效配对差，无法诊断",
        "note": (
            "没有可用于诊断的配对差（非有限值已按缺失剔除）；"
            "本层不产生任何判定。"
        ),
    }
    if n == 0:
        return out
    mean = sum(values) / n
    out["mean"] = mean
    if n >= 2:
        variance = sum((v - mean) ** 2 for v in values) / max(n - 1, 1)
        try:
            std = math.sqrt(variance)
            out["std"] = std if math.isfinite(std) else None
        except (OverflowError, ValueError):
            out["std"] = None
    if out["std"] == 0.0:
        out["degenerate"] = True
    skew, kurt = _qa_skew_kurt(values)
    out["skewness"] = skew
    out["excess_kurtosis"] = kurt
    outliers = _qa_outlier_indices(values)
    out["n_outliers"] = len(outliers)
    out["outlier_indices"] = outliers
    flags = []
    if skew is not None and abs(skew) > QA_SKEW_LIMIT:
        flags.append("skewed")
    if kurt is not None and abs(kurt) > QA_KURT_LIMIT:
        flags.append("heavy_tailed")
    if outliers:
        flags.append("outlier_driven")
    out["flags"] = flags
    assessable = n >= QA_MIN_SHAPE_N
    out["shape_assessable"] = assessable
    if n < min_seeds:
        out["status"] = "insufficient_seeds"
        out["label"] = f"种子不足（n={n}<{min_seeds}），分布形态不可评估"
        out["note"] = (
            f"n={n} 低于最小种子数 {min_seeds}：不产生显著性判定，"
            "分布形态诊断同样没有意义。"
        )
    elif out["degenerate"]:
        out["status"] = "degenerate"
        out["label"] = "配对差无变异（方差退化）"
        out["note"] = "所有配对差相同：t 检验不可计算，形状诊断无意义。"
    elif not assessable:
        out["status"] = "not_assessable"
        out["label"] = f"样本过小（n={n}<{QA_MIN_SHAPE_N}），形状诊断功效不足"
        out["note"] = (
            f"n={n} 时偏度/峰度的抽样误差远大于阈值（{QA_SKEW_LIMIT}/"
            f"{QA_KURT_LIMIT}），任何形状判断都没有功效；此时分布无关的"
            "置换检验与符号检验才是可用的稳健性证据。"
        )
    elif flags:
        out["status"] = flags[0]
        out["label"] = "；".join(
            QA_ASSUMPTION_LABELS.get(f, f) for f in flags
        )
        out["note"] = (
            "启发式标记（无检验、无 p 值）：显著结论应结合置换检验与"
            "留一法一起读；若效应由个别离群种子主导，均值型 p 值不可靠。"
        )
    else:
        out["status"] = "plausible"
        out["label"] = "未见明显形状异常"
        out["note"] = (
            "在启发式阈值下未见明显偏态/厚尾/离群点；这不等于假设被验证，"
            "只是没有可见的反证。"
        )
    return out


def assumption_summary(tests) -> dict:
    """汇总一批检验的假设诊断状态（供 significance.json 机读）。"""
    items = [t for t in (tests if isinstance(tests, list) else []) if isinstance(t, dict)]
    status_counts = {}
    flag_counts = {}
    n_assessable = 0
    n_flagged = 0
    for t in items:
        diag = t.get("assumption_diagnostics")
        if not isinstance(diag, dict):
            continue
        status = diag.get("status") or "unknown"
        status_counts[status] = status_counts.get(status, 0) + 1
        flags = diag.get("flags")
        if isinstance(flags, list):
            for f in flags:
                flag_counts[f] = flag_counts.get(f, 0) + 1
            if flags:
                n_flagged += 1
        if diag.get("shape_assessable"):
            n_assessable += 1
    return {
        "n_tests": len(items),
        "n_with_diagnostics": sum(status_counts.values()),
        "n_shape_assessable": n_assessable,
        "n_flagged": n_flagged,
        "status_counts": status_counts,
        "flag_counts": flag_counts,
        "thresholds": {
            "min_shape_n": QA_MIN_SHAPE_N,
            "outlier_z": QA_OUTLIER_Z,
            "skew_limit": QA_SKEW_LIMIT,
            "kurt_limit": QA_KURT_LIMIT,
        },
        "note": (
            "配对差形状诊断是启发式标记，不产生 p 值、不改变任何判定；"
            "n<8 时形状不可评估（多数种子数下都是如此），此时应读置换检验"
            "与留一法的稳健性证据，而不是把“不可评估”读成“无异常”。"
        ),
    }


def _qa_ci_crosses_zero(ci) -> bool:
    """配对差 bootstrap 区间是否跨 0（无法解析时返回 False）。"""
    if not isinstance(ci, (list, tuple)) or len(ci) != 2:
        return False
    lo, hi = _qa_num(ci[0]), _qa_num(ci[1])
    if lo is None or hi is None:
        return False
    return lo <= 0.0 <= hi


def evidence_strength(test, alpha=ALPHA, min_seeds=MIN_SEEDS) -> dict:
    """把一条配对检验的证据折成可比的强度等级。

    等级：
      * ``strong``：Holm 校正后显著，且没有任何警示（CI 不跨 0、置换检验
        一致、效应不落在逐种子噪声内、不低于 MDE、留一方向稳定、形状无异常）；
      * ``marginal``：显著但只有 1 项警示；
      * ``fragile``：显著但有 2 项及以上警示；
      * ``inconclusive``：未显著且效应低于 MDE → 功效不足，不能断言无效应；
      * ``no_evidence``：未显著且效应不低于 MDE → 未检出显著差异（仍不等于
        差异为 0，须看 TOST/等价性层）；
      * ``not_assessable``：种子不足、方差退化或 p 不可解析 → 不作判定。

    本层不改变 p 值、不参与任何多重比较校正，只解释既有结论的硬度。
    """
    t = test if isinstance(test, dict) else {}
    n = _qa_int(t.get("n"))
    p_adj = _qa_num(t.get("p_adj"))
    diag = t.get("assumption_diagnostics")
    diag = diag if isinstance(diag, dict) else None
    caveats = []
    out = {
        "level": "not_assessable",
        "label": QA_STRENGTH_LABELS["not_assessable"],
        "score": None,
        "significant": None,
        "n": n,
        "p_adj": p_adj,
        "caveats": caveats,
        "assumption_status": diag.get("status") if diag else None,
        "note": (
            "种子不足或检验不可计算（方差退化/指标缺失）：不产生任何"
            "显著性或强度判定。"
        ),
    }
    if n is None or n < min_seeds or p_adj is None:
        out["caveats"].append("样本量不足或检验不可计算，未作判定")
        return out
    significant = p_adj < alpha
    out["significant"] = significant
    p_perm_adj = _qa_num(t.get("p_perm_adj"))
    if significant:
        if p_perm_adj is not None and p_perm_adj >= alpha:
            caveats.append("置换检验未达显著（分布无关口径下证据不足）")
        if _qa_ci_crosses_zero(t.get("mean_diff_ci95")):
            caveats.append("配对差 95% bootstrap CI 跨 0")
        if t.get("within_noise") is True:
            caveats.append("效应落在逐种子噪声内（|Δ|<ΔSD）")
        if t.get("below_mde") is True:
            caveats.append("效应低于当前种子数的 MDE")
        if t.get("loo_sign_stable") is False:
            caveats.append("留一法方向翻转（结论依赖个别种子）")
        if diag is not None:
            for flag, text in QA_ASSUMPTION_LABELS.items():
                if flag in (diag.get("flags") or []):
                    caveats.append(text)
        level = "strong"
        if caveats:
            level = "marginal" if len(caveats) == 1 else "fragile"
        out["level"] = level
        out["score"] = QA_STRENGTH_SCORES[level]
        out["label"] = (
            QA_STRENGTH_LABELS[level]
            if not caveats
            else f"{QA_STRENGTH_LABELS[level]}（{len(caveats)} 项警示）"
        )
        out["note"] = (
            "Holm 校正后显著。"
            + (
                "无附加警示：结论对检验口径与种子选择都稳健。"
                if not caveats
                else "存在附加警示，读结论时必须与警示一并披露。"
            )
        )
        return out
    mde = _qa_num(t.get("mde"))
    mean_diff = _qa_num(t.get("mean_diff"))
    power_limited = bool(t.get("below_mde") is True)
    if not power_limited and mde is not None and mean_diff is not None:
        power_limited = abs(mean_diff) < abs(mde)
    if p_perm_adj is not None and p_perm_adj < alpha:
        caveats.append("仅置换检验显著（t 检验未过校正），属边缘线索")
    if power_limited:
        caveats.append("效应低于 MDE：功效不足，不能断言无效应")
    eq = t.get("equivalence")
    if isinstance(eq, dict) and eq.get("status") == "equivalent":
        caveats.append("TOST：差异小于 SESOI（仍不得据此宣称差异恰为 0）")
    level = "inconclusive" if power_limited else "no_evidence"
    out["level"] = level
    out["score"] = QA_STRENGTH_SCORES[level]
    out["label"] = QA_STRENGTH_LABELS[level]
    out["note"] = (
        "Holm 校正后未达显著：按 MDE 判定为功效不足（只能写“当前种子数"
        "下未检出”，不能写“无效应”）。"
        if power_limited
        else "Holm 校正后未达显著，且观测效应不小于 MDE：未检出显著差异，"
        "但仍不等于差异为 0（需看等价性/TOST 层）。"
    )
    return out


def analysis_population(tests) -> dict:
    """分析人群核算：进入统计的配对观测数与被剔除的配对数（按原因）。

    只统计已登记的剔除原因（split/协议/评测格不一致、指标缺失）。任何新增
    的剔除路径都必须同时登记计数，否则读者无法判断“少掉的样本去了哪里”。
    """
    items = [t for t in (tests if isinstance(tests, list) else []) if isinstance(t, dict)]
    excluded = {
        "split_mismatch": 0,
        "protocol_mismatch": 0,
        "cell_mismatch": 0,
        "missing_value": 0,
    }
    fields = {
        "split_mismatch": "excluded_split_mismatches",
        "protocol_mismatch": "excluded_protocol_mismatches",
        "cell_mismatch": "excluded_cell_mismatches",
        "missing_value": "excluded_missing_values",
    }
    n_pairs = 0
    n_by_size = {}
    for t in items:
        n = _qa_int(t.get("n"))
        if n is not None:
            n_pairs += n
            n_by_size[str(n)] = n_by_size.get(str(n), 0) + 1
        for key, field in fields.items():
            count = _qa_int(t.get(field))
            if count:
                excluded[key] += count
    return {
        "n_tests": len(items),
        "n_paired_observations": n_pairs,
        "n_by_seed_count": n_by_size,
        "excluded_pairs": excluded,
        "total_excluded": sum(excluded.values()),
        "note": (
            "分析人群核算：n_paired_observations 为进入统计的配对观测总数，"
            "excluded_pairs 为按原因剔除的配对数。剔除只允许由已登记的原因"
            "触发；新增剔除路径必须同时登记计数，否则视为分析人群不透明。"
        ),
    }


def conclusion_strength(hyp_eval, component_evidence=None, alpha=ALPHA) -> dict:
    """把全部确认性主张汇成“这次实验的结论强度”总览。

    输出总体等级（strong/moderate/weak/none/inconclusive/not_assessable）、
    逐条主张的强度与警示、以及警示的聚合计数。总体等级只由“确认性主张的
    判定 + 每条主张的证据强度”决定，不引入新的显著性判定。
    """
    items = []
    counts = {}
    caveat_counts = {}
    for h in hyp_eval if isinstance(hyp_eval, list) else []:
        if not isinstance(h, dict):
            continue
        strength = h.get("evidence_strength")
        strength = strength if isinstance(strength, dict) else None
        level = strength.get("level") if strength else None
        caveats = list(strength.get("caveats") or []) if strength else []
        status = h.get("status")
        if status == "未定" and strength and strength.get("significant"):
            caveats.append("跨全部预注册假设的全局 FWER 校正后未达显著")
        for c in caveats:
            caveat_counts[c] = caveat_counts.get(c, 0) + 1
        entry = {
            "id": h.get("id"),
            "claim": h.get("claim"),
            "metric": h.get("metric"),
            "status": status,
            "confirmatory_valid": h.get("confirmatory_valid"),
            "mean_diff": h.get("mean_diff"),
            "mean_diff_ci95": h.get("mean_diff_ci95"),
            "d_z": h.get("d_z"),
            "strength": level,
            "strength_label": strength.get("label") if strength else None,
            "caveats": caveats,
        }
        items.append(entry)
        key = level or "unknown"
        counts[key] = counts.get(key, 0) + 1
    supported = [i for i in items if i.get("status") == "支持"]
    rejected = [i for i in items if i.get("status") == "拒绝"]
    undetermined = [i for i in items if i.get("status") == "未定"]
    untested = [i for i in items if i.get("status") == "未检验"]
    # “不可判定”= 对比不是单因素（归因对象不成立）：与“未定/未检验”不同，
    # 不能算作“主张未被支持”的证据缺失，也不能算作支持。
    undecidable = [i for i in items if i.get("status") == "不可判定"]
    invalid = [i for i in items if i.get("confirmatory_valid") is False]
    strong_supported = [i for i in supported if i.get("strength") == "strong"]
    marginal_supported = [i for i in supported if i.get("strength") == "marginal"]
    fragile_supported = [i for i in supported if i.get("strength") == "fragile"]
    if not items:
        grade = "not_assessable"
    elif invalid:
        grade = "invalid"
    elif strong_supported:
        grade = "strong"
    elif marginal_supported:
        grade = "moderate"
    elif fragile_supported:
        grade = "weak"
    elif supported:
        grade = "weak"
    elif undetermined or untested or undecidable:
        grade = "inconclusive"
    else:
        grade = "none"
    grade_labels = {
        "strong": "强：至少一条确认性主张稳健成立",
        "moderate": "中：确认性主张成立但带 1 项警示",
        "weak": "弱：确认性主张仅边缘/脆弱成立",
        "none": "无：没有确认性主张被支持",
        "inconclusive": "未定：主张未获支持且多为功效不足/未检验/不可判定",
        "invalid": "无效：预注册完整性/范围门未通过，判定不作数",
        "not_assessable": "无法判定：没有可评估的确认性主张",
    }
    comp_items = []
    if isinstance(component_evidence, list):
        for c in component_evidence:
            if not isinstance(c, dict):
                continue
            comp_items.append(
                {
                    "component": c.get("component"),
                    "evidence_status": c.get("evidence_status"),
                    "confirmatory_supported": c.get("confirmatory_supported"),
                    "verdict": c.get("verdict"),
                    "fragile": c.get("fragile"),
                }
            )
    return {
        "grade": grade,
        "label": grade_labels.get(grade, grade),
        "counts": counts,
        "caveat_counts": caveat_counts,
        "n_supported": len(supported),
        "n_rejected": len(rejected),
        "n_undetermined": len(undetermined),
        "n_untested": len(untested),
        "n_undecidable": len(undecidable),
        "n_invalid": len(invalid),
        "items": items,
        "component_evidence": comp_items,
        "note": (
            "结论强度总览只由确认性主张的判定与逐条证据强度合成，不新增"
            "显著性判定：等级为 strong/moderate/weak/none/inconclusive/"
            "invalid。‘支持’必须与强度、CI 与警示一并披露；强度为 fragile "
            "或 marginal 的主张不得单独作为结论。"
        ),
    }
