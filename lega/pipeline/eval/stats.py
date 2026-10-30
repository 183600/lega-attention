"""跨种子统计、RULER 长度/样例统计。"""
from collections import Counter

from .._shared import _install as _shared_install

_shared_install(globals())

def _seed_key(path: str, data: dict):
    m = _SEED_RE.search(os.path.splitext(os.path.basename(path))[0])
    if m:
        return int(m.group(1))
    s = data.get("seed")
    if s is None:
        return None
    # 外部/旧产物里的 seed 可能是对象、字符串、小数或 Infinity；无法解析为
    # 精确整数时按“缺少种子信息”处理，不能截断后冒充另一个种子。
    return _integral_value(s)


def _result_files(results_dir: str, prefix: str, scope_seeds) -> list:
    """按“命名约定优先”的顺序返回 <prefix>_*.json。

    多种子运行的结果带 _seed<seed> 后缀、单种子不带。两种命名可能同时
    存在（如在旧多种子目录上重跑单种子），且经 _seed_key 会落到同一
    (变体, 种子) 槽位；调用方按序赋值、后写覆盖，因此必须让与本次
    约定一致的文件排在最后，否则陈旧的旧命名文件覆盖新结果。
    """
    fps = glob.glob(os.path.join(results_dir, f"{prefix}_*.json"))
    if scope_seeds is None:
        return sorted(fps)
    if not scope_seeds:
        # 空列表表示本次 run 没有任何种子；不能回退成“扫描全部旧结果”。
        return []
    expected_tagged = len(scope_seeds) > 1

    def convention_match(fp: str) -> int:
        stem = os.path.splitext(os.path.basename(fp))[0]
        tagged = _SEED_RE.search(stem) is not None
        return int(tagged == expected_tagged)

    return sorted(fps, key=lambda fp: (convention_match(fp), fp))


def _sorted_seeds(mapping) -> list:
    if not isinstance(mapping, dict):
        return []
    return sorted(mapping.keys(),
                  key=lambda s: (s is None, s if s is not None else -1))


def _variant_seed_map(runs: dict, variant: str) -> dict:
    """取某变体的 {seed: slot}；已有带种子的结果时忽略无种子的文件，
    同一次实验不重复计为额外种子。"""
    m = (runs.get(variant) if isinstance(runs, dict) else None)
    m = m if isinstance(m, dict) else {}
    tagged = {s: slot for s, slot in m.items() if s is not None}
    return tagged if tagged else m


def cross_variant_badcase_map(runs: dict, variants) -> dict:
    """跨变体对齐用：每个变体取“最小种子”的 RULER 错误清单。

    最小种子没有可用的 RULER 错误清单时（例如只跑了其他种子，或产物损坏
    缺 badcases 键）退化到最小的有可用清单的种子；已评测且零错误的变体
    记为 [] 并参与对照。未跑过 RULER 的变体不进结果（无法参与对照）。

    badcases 必须是列表：空列表才表示“已评测且零错误”；缺失/非列表是
    无数据（旧格式或损坏产物），按 [] 处理会把参照变体伪造成全对，进而
    把其他变体的每个失败都误报成 new_vs_reference。
    """
    out = {}
    for v in variants:
        m = _variant_seed_map(runs, v)
        for s in _sorted_seeds(m):
            slot = m[s]
            r = slot.get("ruler") if isinstance(slot, dict) else None
            badcases = r.get("badcases") if isinstance(r, dict) else None
            if isinstance(badcases, list):
                out[v] = badcases
                break
    return out


def _stat(values) -> dict:
    """跨种子聚合：{mean, std, n}；无有效值时 None。

    n 只计有限值，与 mean_std 的口径一致：NaN/Inf 不参与统计也不虚增
    样本量（否则会显示“n=5”但均值实际只来自 4 个种子）。
    """
    vals = []
    for v in (values if isinstance(values, (list, tuple)) else []):
        f = _finite_number(v)
        if f is not None:
            vals.append(f)
    m, s = mean_std(vals)
    if m is None:
        return None
    return {"mean": m, "std": s, "n": len(vals)}


def _ppl_of(d):
    if not isinstance(d, dict):
        return None
    # ppl 键存在但为 null（外部/旧格式）时必须继续回退 val_ppl，不能把
    # 一个缺失字段当成“没有 PPL”。
    v = d.get("ppl")
    if v is None:
        v = d.get("val_ppl")
    v = _finite_number(v)
    # PPL = exp(loss) 不可能为负；负值只可能来自损坏/手改产物，若照常
    # 进入跨种子均值/配对检验会制造“困惑度小于 0”的假证据。0 是若干
    # 历史合成夹具使用的边界值，保留兼容（不参与 log-PPL 族）。
    return v if v is not None and v >= 0 else None


def _log_ppl_of(d):
    """log-PPL：比值尺度的配对分析必须用对数差（=几何均值比）。

    算术 PPL 差由高 PPL 的种子/变体主导；log 差对尺度对称，exp(ΔlogPPL)
    即几何均值比。原始 PPL 仍是预注册主指标，log 分析只作敏感性证据。
    """
    v = _ppl_of(d)
    if v is None or v <= 0:
        return None
    return math.log(v)


def _ppl_ci_rel_pct(d):
    """单次 run 内验证窗口抽样 CI 的相对半宽（占 PPL 点估计的百分比）。

    PPL 的跨种子 std 与此量回答不同问题：前者是训练随机性，后者是
    “只评测有限个验证窗口”的评估随机性。两者都要看，不能互相替代。
    """
    if not isinstance(d, dict):
        return None
    ci = d.get("val_ppl_ci95")
    ppl = _ppl_of(d)
    if (not isinstance(ci, (list, tuple)) or len(ci) != 2
            or ppl is None or ppl <= 0):
        return None
    lo, hi = _finite_number(ci[0]), _finite_number(ci[1])
    if lo is None or hi is None or hi < lo:
        return None
    # 两端都有限不代表差/比值可表示：hi−lo 或除以极小 ppl 仍可能溢出成
    # inf。非有限的相对半宽不能写进汇总/JSON，按缺失处理。
    try:
        half = (hi - lo) / 2.0
        rel = half / ppl * 100.0
    except (OverflowError, ZeroDivisionError):
        return None
    return rel if math.isfinite(rel) and rel >= 0.0 else None


def _eval_scalar(key):
    """从 eval.json 提取标量（缺失/非法值返回 None），用于覆盖率汇总。"""
    def get(d):
        if not isinstance(d, dict):
            return None
        return _finite_number(d.get(key))
    return get


def _ruler_task_means(d, metric: str = "em"):
    """单次 RULER 结果：各任务先在上下文档位求平均。

    默认 `metric="em"`，与预注册假设 H4 的“RULER 平均 EM”逐字一致。
    fwe 的 recall 不进入主平均（recall 不惩罚“把上下文词全列出来”的灌水
    策略），仅由 `_ruler_fwe_recall` 单独报告作诊断。

    所有进入平均的指标（EM/recall/precision/F1）都是 [0,1] 比例：坏/手改
    产物里的 1.5 或 -1 若被当成合法观测，会把 RULER 平均推到 100% 以上或
    负值，直接制造“方法有效/失效”的假结论。越界值按缺失处理，与
    `_ruler_depth_means` 的校验口径一致。

    单元格的 n 是样本数：n=0（或负数/非整数）表示该格没有有效观测，不能
    用一个空洞的 em=0 拉低 macro 平均；这与 `_ruler_avg_micro` /
    `_ruler_depth_means` 只接受正计数的口径一致。旧产物缺 n 字段时保持
    兼容（保留该格），显式 n 非法则按缺失处理。
    """
    if not isinstance(d, dict):
        return {}
    results = d.get("results")
    if not isinstance(results, dict):
        return {}
    out = {}
    for task, by_ctx in results.items():
        if (not isinstance(task, str) or task.startswith("_")
                or not isinstance(by_ctx, dict) or not by_ctx):
            continue
        vals = []
        for m in by_ctx.values():
            if not isinstance(m, dict):
                continue
            if "n" in m and m.get("n") is not None:
                n = _integral_value(m.get("n"))
                if n is None or n <= 0:
                    continue
            v = _finite_number(m.get(metric))
            if v is not None and 0.0 <= v <= 1.0:
                vals.append(v)
        if vals:
            out[task] = float(sum(vals) / len(vals))
    return out


def _ruler_fwe_recall(d):
    """fwe 的召回率（仅诊断，不进入 RULER 主平均）。

    recall 奖励“多答中一个”，对把上下文词全部列出的灌水策略不敏感；主
    指标必须用严格集合 EM（预注册口径）。召回率单列帮助区分“完全没找到
    高频词”与“找对了但多答/少答”。
    """
    tm = _ruler_task_means(d, metric="recall")
    return tm.get("fwe")


def _ruler_avg_of(d):
    """RULER 平均（macro）：任务等权，任务内上下文档位等权，统一用 EM。

    该口径忽略每格样本数 n 与任务/上下文规模差异——n=2 的格与 n=30 的
    格同样重。预注册假设的“RULER平均”固定指这一口径；样本加权口径另见
    `_ruler_avg_micro`，两者差异是“平均值定义依赖”的直接证据。
    fwe 用严格集合 EM（与 H4 的“平均 EM”一致），recall 仅单列诊断。
    """
    tm = _ruler_task_means(d)
    return float(sum(tm.values()) / len(tm)) if tm else None


def _ruler_avg_micro(d):
    """RULER 平均（micro）：按每格样本数 n 加权，所有评测样例等权。

    与 macro 相同统一用各任务 EM（fwe 为严格集合 EM）；不再让每个任务/
    上下文格等权。macro 与 micro 并报，避免“平均”二字的歧义掩盖任务间
    难度与格大小的不均。
    """
    if not isinstance(d, dict):
        return None
    results = d.get("results")
    if not isinstance(results, dict):
        return None
    cells = []
    max_weight = 0.0
    for task, by_ctx in results.items():
        if (not isinstance(task, str) or task.startswith("_")
                or not isinstance(by_ctx, dict)):
            continue
        for cell in by_ctx.values():
            if not isinstance(cell, dict):
                continue
            v, n = _finite_number(cell.get("em")), _integral_value(cell.get("n"))
            # 与 macro 口径同校：EM 是 [0,1] 比例，越界值不能参与按 n 加权。
            if v is None or not 0.0 <= v <= 1.0 or n is None or n <= 0:
                continue
            # n 是样本计数，但外部 JSON 可写入超出 float 范围的合法大整数。
            # 直接 v*n 会 OverflowError；无法表示为有限浮点的计数按损坏元数据
            # 跳过，而不是让整份汇总崩溃。
            try:
                nf = float(n)
            except (OverflowError, ValueError):
                continue
            if not math.isfinite(nf):
                continue
            cells.append((v, nf))
            max_weight = max(max_weight, nf)
    if not cells or max_weight <= 0.0:
        return None
    # 先按最大权重归一化再累加：各 nf 都是有限正数，但两个 1e308 的格
    # 相加就会溢出成 inf（随后 num/total 变成 NaN）。归一化后权重均落在
    # (0,1]，加权和与总权重都不会越过 float 上限，均值保持不变。
    num = total = 0.0
    for v, nf in cells:
        w = nf / max_weight
        num += v * w
        total += w
    return (num / total) if total > 0.0 else None


def _ruler_depth_means(d):
    """NIAH 位置分层的 macro 平均 EM：{early/middle/late: mean}。

    只统计 niah_* 任务的 (任务, 上下文, 位置箱) 单元格，每格等权，避免
    某一上下文长度或任务因样本多而支配分层分数。旧产物没有 `em_by_depth`
    时返回空 dict（不伪造）；空箱不进入结果。分层样本彼此不同，不能把
    三个均值直接当作同批样例做显著性检验——逐格 Wilson CI 见 ruler_*.json。
    """
    if not isinstance(d, dict):
        return {}
    results = d.get("results")
    if not isinstance(results, dict):
        return {}
    vals = {b: [] for b in ("early", "middle", "late")}
    for task, by_ctx in results.items():
        if (not isinstance(task, str) or not task.startswith("niah")
                or not isinstance(by_ctx, dict)):
            continue
        for cell in by_ctx.values():
            if not isinstance(cell, dict):
                continue
            by_depth = cell.get("em_by_depth")
            if not isinstance(by_depth, dict):
                continue
            for b in vals:
                rec = by_depth.get(b)
                if not isinstance(rec, dict):
                    continue
                v = _finite_number(rec.get("em"))
                n = _integral_value(rec.get("n"))
                if (v is None or n is None or n <= 0
                        or not 0.0 <= v <= 1.0):
                    continue
                vals[b].append(v)
    return {b: float(sum(v) / len(v)) for b, v in vals.items() if v}


def _ruler_depth_gap(d):
    """NIAH 位置落差 = early EM − late EM（正=经典“中间/末期更差”方向）。

    仅当两端都有数据时定义；中部单独一列，不用它替代两端落差。
    """
    dm = _ruler_depth_means(d)
    if "early" not in dm or "late" not in dm:
        return None
    return dm["early"] - dm["late"]


def _ruler_min_cell_n(d):
    """单次 RULER 结果里 (任务, 上下文) 单元的最小样本数。

    EM/Wilson CI 的精度直接由每格 n 决定；汇总只报平均分而不报最小 n，
    会把 n=2 的 0.000 与 n=30 的 0.000 当成同样可信的证据。
    """
    if not isinstance(d, dict):
        return None
    results = d.get("results")
    if not isinstance(results, dict):
        return None
    ns = []
    for task, by_ctx in results.items():
        if (not isinstance(task, str) or task.startswith("_")
                or not isinstance(by_ctx, dict)):
            continue
        for cell in by_ctx.values():
            if not isinstance(cell, dict) or cell.get("n") is None:
                continue
            n = _integral_value(cell["n"])
            # n 是评测样本数：负数/0 不是“小样本”，而是损坏计数；若参与
            # min 会让汇总把坏产物报成“最差格 n=-5”。与 _ruler_avg_micro /
            # _ruler_depth_means 同口径，只接受正计数。
            if n is not None and n > 0:
                ns.append(n)
    return min(ns) if ns else None


def _ruler_cell_sample_outcomes(d):
    """RULER payload → {(task, ctx): {sample_id: em}}；损坏时返回 None。

    保留单元格归属，因为预注册 `RULER平均` 是 macro（任务/上下文档位等权），
    而逐样例 McNemar 池化所有样例是 micro（样例等权）：两者估计量不同，
    必须分开报告。旧产物没有 samples 字段时返回 None（不伪造）。
    """
    if not isinstance(d, dict):
        return None
    results = d.get("results")
    if not isinstance(results, dict):
        return None
    out = {}
    seen = set()
    for task, by_ctx in results.items():
        if (not isinstance(task, str) or task.startswith("_")
                or not isinstance(by_ctx, dict)):
            continue
        for ctx, cell in by_ctx.items():
            if not isinstance(cell, dict):
                continue
            samples = cell.get("samples")
            n = _integral_value(cell.get("n"))
            if not isinstance(samples, list) or n is None or n != len(samples):
                return None
            cell_out = {}
            for rec in samples:
                if not isinstance(rec, dict):
                    return None
                sid = rec.get("sample_id")
                em = _num_or_none(rec.get("em"))
                if not isinstance(sid, str) or em not in (0.0, 1.0):
                    return None
                if sid in seen or sid in cell_out:
                    return None
                cell_out[sid] = em
            seen.update(cell_out)
            out[(task, str(ctx))] = cell_out
    return out


def _ruler_cell_depth_outcomes(d):
    """RULER payload → {(task, ctx): {sample_id: {"em", "depth_bin"}}}。

    只保留带 needle 位置信息的 ``niah_*`` 单元格。位置剂量-反应要求
    **同一批 sample_id** 上候选与参照的比较，因此必须消费逐样例的
    ``depth_bin``；只有聚合 ``em_by_depth`` 的旧产物无法做样例配对，返回
    None（下游显式列为“缺少位置信息”，不伪造配对证据）。

    与 ``_ruler_cell_sample_outcomes`` 一样，任一样例 id/EM 损坏、n 与
    samples 长度不符都整份 payload 判为不可用，避免部分损坏的单元格让
    不同位置的估计量基于不同的样本集合。
    """
    cells = _ruler_cell_sample_outcomes(d)
    if cells is None:
        return None
    results = d.get("results") if isinstance(d, dict) else None
    if not isinstance(results, dict):
        return None
    out = {}
    for task, by_ctx in results.items():
        if (not isinstance(task, str) or not task.startswith("niah")
                or not isinstance(by_ctx, dict)):
            continue
        for ctx, cell in by_ctx.items():
            key = (task, str(ctx))
            sid_em = cells.get(key)
            if not isinstance(sid_em, dict) or not isinstance(cell, dict):
                return None
            samples = cell.get("samples")
            if not isinstance(samples, list) or len(samples) != len(sid_em):
                return None
            cell_out = {}
            for rec in samples:
                if not isinstance(rec, dict):
                    return None
                sid = rec.get("sample_id")
                db = rec.get("depth_bin")
                if sid not in sid_em or db not in RULER_DEPTH_BINS:
                    return None
                cell_out[sid] = {"em": sid_em[sid], "depth_bin": db}
            out[key] = cell_out
    return out or None


def _ruler_sample_outcomes(d):
    """RULER payload → {sample_id: em(0/1)}；缺少/损坏/单元 n 不符时返回 None。

    逐样例结果是“固定评测样例”层面配对检验的原材料。旧产物没有 samples
    字段时返回 None（不伪造），下游只跳过该 run 的样例级分析，不影响
    聚合 EM 的既有检验。
    """
    cells = _ruler_cell_sample_outcomes(d)
    if cells is None:
        return None
    out = {}
    for cell_out in cells.values():
        out.update(cell_out)
    return out


def _mcnemar_exact_p(candidate_only: int, reference_only: int) -> float:
    """兼容旧调用点：委托给 lega.utils.mcnemar_exact_p（单一实现）。"""
    return mcnemar_exact_p(candidate_only, reference_only)


def _ruler_pair_protocol_ok(ra, rb) -> bool:
    """两个 RULER payload 是否共享题面/抽样/生成预算与已评测格集合。"""
    pa, pb = _ruler_protocol(ra), _ruler_protocol(rb)
    if not _same_protocol_fields(pa, pb):
        return False
    ca, cb = _ruler_cell_signature(ra), _ruler_cell_signature(rb)
    return not (ca is not None and cb is not None and ca != cb)



# 上下文长度剂量-反应的检验族名。刻意不使用 ``RULER:<task>`` 前缀：
# 逐任务检验是“同一估计量在任务上的分解”，而长度分层是另一个估计量
# （每个 ctx 内的 macro），混进同一 Holm 家族会改变既有逐任务校正的家族
# 大小。三者各自成族，且都标 exploratory。
# 与等价性层的 RULER SESOI 对齐（1.00 pp）。只用于“总平均是否掩盖了
# 实际重要的长上下文效应”的描述性标记，不是新的显著性门槛。


def _length_close(a, b, tol: float = 1e-9) -> bool:
    """两个 [0,1] 量级的 EM 均值是否在数值误差内相等。"""
    if a is None or b is None:
        return False
    try:
        fa, fb = float(a), float(b)
    except (TypeError, ValueError, OverflowError):
        return False
    if not (math.isfinite(fa) and math.isfinite(fb)):
        return False
    return abs(fa - fb) <= tol


def _length_trend_verdict(t: dict, kind: str) -> str:
    """趋势检验的专用结论：不能沿用“候选显著优于参照”的措辞。

    ``_apply_holm`` 的通用 verdict 只区分“更好/更差”，对“优势随长度增大”
    这类交互没有语义；这里在保留 p_adj/置换/噪声信息的前提下改写。
    """
    t = t if isinstance(t, dict) else {}
    n = t.get("n")
    padj = t.get("p_adj")
    if t.get("p") is None or padj is None:
        base = "无法检验（种子不足或配对差方差退化）"
    elif not isinstance(n, int) or isinstance(n, bool) or n < MIN_SEEDS:
        base = f"种子不足（n={n}<{MIN_SEEDS}），无法判定长度趋势"
    elif padj < ALPHA:
        delta = _finite_number(t.get("mean_diff"))
        if delta is None:
            base = "无法检验（斜率/落差不可表示）"
            return base
        delta_pp = delta * 100.0
        # 有限输入乘 100 仍可能溢出（如 1e308），不能写出 +inf。
        pp_text = (f"{delta_pp:+.2f}" if math.isfinite(delta_pp)
                   else "溢出")
        if kind == "slope":
            direction = "增大" if delta > 0 else "减小"
            base = (f"优势随上下文长度显著{direction}"
                    f"（每翻倍 {pp_text} pp，Holm p={padj:.4f}）")
        else:
            direction = "更长上下文更有利" if delta > 0 else "更长上下文更不利"
            base = (f"长−短上下文优势差显著（{direction}，"
                    f"{pp_text} pp，Holm p={padj:.4f}）")
    else:
        base = f"未检出长度趋势（Holm p={padj:.4f}）"
    # 与 _apply_holm 的通用 verdict 一致：t 与符号翻转置换结论冲突时必须
    # 显式披露，不能因为这里重写了措辞就把置换证据的警示丢掉。小样本下
    # 置换 p 的下限可能高于 α，这一句是“显著”结论最重要的稳健性限定。
    perm = _finite_number(t.get("p_perm_adj"))
    if (padj is not None and perm is not None
            and isinstance(n, int) and not isinstance(n, bool)
            and n >= MIN_SEEDS):
        if padj < ALPHA and perm >= ALPHA:
            base += "；⚠ 置换检验不显著（见注）"
        elif padj >= ALPHA and perm < ALPHA:
            base += "；置换检验显著（边缘证据，建议补种子确认）"
    return base + _noise_verdict_note(t)


def _ruler_length_analysis(runs, pairs) -> dict:
    """RULER 上下文长度分层 + 剂量-反应检验（探索性）。

    回答一个只报“RULER 平均”时无法回答的问题：**候选相对参照的优势是否
    随上下文长度变化？** 对每个 (候选, 参照)：

    1. 每个种子在每个公共请求档位（payload 的 ``ctx_lens``；不同任务的
       ``room_ctx`` 因生成预算不同，先按请求档位对齐）上先算任务等权的
       macro EM，再逐 ctx
       做同种子配对 t + 符号翻转置换 + bootstrap CI（家族 ``RULER@ctx``）；
    2. 每个种子对 ``ΔEM ~ log2(ctx)`` 拟合斜率（单位：长度每翻一倍），
       跨种子检验斜率是否偏离 0（家族 ``RULER@length_slope``）；
    3. 每个种子算 ``ΔEM(最长) − ΔEM(最短)``，不假定线性（家族
       ``RULER@length_span``）。

    三条证据互补：斜率回答“是否随长度系统变化”，首末落差回答“端点是否
    同向”，逐 ctx 表回答“哪一档在驱动结论”。任一方向与总体 RULER 平均
    相反时显式标 ``overall_vs_*_sign_flip``，避免用总平均掩盖长上下文失效。

    边界：探索性、未预注册，不改变 H1–H7 的确认性判定；每个 ctx 的估计
    仍受“有限评测样例”影响（Wilson CI 见 ruler_*.json），跨种子 t 检验
    只覆盖训练随机性；``n<MIN_SEEDS`` 只报描述、不判定。
    """
    runs = runs if isinstance(runs, dict) else {}
    ctx_tests, slope_tests, span_tests, pair_rows = [], [], [], []
    for entry in (pairs if isinstance(pairs, list) else []):
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            continue
        cand, ref = entry
        try:
            if cand not in runs or ref not in runs:
                continue
        except TypeError:
            continue
        cm = _variant_seed_map(runs, cand)
        rm = _variant_seed_map(runs, ref)
        seeds = [s for s in _sorted_seeds(cm) if s in rm]
        if not seeds:
            continue
        excluded = {"protocol_mismatch": 0, "missing_ruler": 0,
                    "missing_contexts": 0, "non_finite_overall": 0}
        seed_rows = []
        for s in seeds:
            sa = cm.get(s) if isinstance(cm.get(s), dict) else {}
            sb = rm.get(s) if isinstance(rm.get(s), dict) else {}
            ra = sa.get("ruler") if isinstance(sa.get("ruler"), dict) else None
            rb = sb.get("ruler") if isinstance(sb.get("ruler"), dict) else None
            if ra is None or rb is None:
                excluded["missing_ruler"] += 1
                continue
            if not _ruler_pair_protocol_ok(ra, rb):
                excluded["protocol_mismatch"] += 1
                continue
            ca = context_macro_means(ra, align_requested=True)
            cb = context_macro_means(rb, align_requested=True)
            if not ca or not cb or set(ca) != set(cb):
                excluded["missing_contexts"] += 1
                continue
            ctxs = sorted(ca)
            diffs = {ctx: ca[ctx] - cb[ctx] for ctx in ctxs}
            if not all(math.isfinite(v) for v in diffs.values()):
                excluded["non_finite_overall"] += 1
                continue
            overall = None
            oa, ob = _ruler_avg_of(ra), _ruler_avg_of(rb)
            if oa is not None and ob is not None:
                maybe = oa - ob
                if math.isfinite(maybe):
                    overall = maybe
            seed_rows.append({
                "seed": _integral_value(s),
                "candidate": ca, "reference": cb, "diffs": diffs,
                "overall_diff": overall,
                "ctx_average_diff": (
                    sum(diffs.values()) / len(diffs) if diffs else None),
            })
        if not seed_rows:
            continue
        # 所有被纳入趋势的种子必须共享同一组上下文档位，否则各档的 n 不同，
        # 跨种子的“同 ctx 配对”不再成立。取交集作为参照网格，只保留实际
        # 网格恰好等于交集的种子，并记账；若保留“有额外档位”的种子，
        # overall/ctx-average 会按各 seed 的完整网格计算，与这里报告的
        # 公共 ctx 轴不一致（汇总行的总体 Δ 将无法对应到 contexts）。
        common = set.intersection(*[set(r["diffs"]) for r in seed_rows])
        if not common:
            continue
        kept = [r for r in seed_rows if set(r["diffs"]) == common]
        dropped = len(seed_rows) - len(kept)
        if dropped:
            excluded["missing_contexts"] += dropped
        if not kept:
            continue
        ctxs = sorted(common)
        n_seeds = len(kept)
        per_context = []
        for ctx in ctxs:
            diffs = [r["diffs"][ctx] for r in kept]
            cand_vals = [r["candidate"][ctx] for r in kept]
            ref_vals = [r["reference"][ctx] for r in kept]
            res = paired_t_test(diffs, [0.0] * len(diffs))
            res.update({
                "candidate": cand, "reference": ref,
                "metric": LENGTH_CTX_METRIC,
                "lower_is_better": False,
                "contrast_type": "length_slice",
                "ruler_ctx": ctx,
                "n_contexts": len(ctxs),
                "candidate_mean": (sum(cand_vals) / len(cand_vals)
                                   if cand_vals else None),
                "reference_mean": (sum(ref_vals) / len(ref_vals)
                                   if ref_vals else None),
                "exploratory": True,
                "note": ("同一档上下文内的任务等权 macro EM 配对差（候选−"
                         "参照）；只覆盖训练种子随机性，不覆盖评测样例抽样"
                         "（后者见 ruler_*.json 的 Wilson CI）。"),
            })
            res = _enrich_paired(res, diffs, [0.0] * len(diffs))
            per_context.append(res)
            ctx_tests.append(res)

        # 斜率：每个种子独立拟合，跨种子检验“斜率均值是否为 0”。不能先
        # 把各 ctx 的点估计合并成一条曲线再拟合——那会把种子噪声当成
        # 确定性趋势，失去配对结构。
        slopes = []
        slope_skipped = 0
        for r in kept:
            sl = log2_slope(ctxs, [r["diffs"][c] for c in ctxs])
            if sl is None:
                slope_skipped += 1
            else:
                slopes.append(sl)
        if len(slopes) >= 1:
            res = paired_t_test(slopes, [0.0] * len(slopes))
            res.update({
                "candidate": cand, "reference": ref,
                "metric": LENGTH_SLOPE_METRIC,
                "lower_is_better": False,
                "contrast_type": "length_trend",
                "ruler_length_trend": "slope",
                "contexts": ctxs,
                "unit": "EM per doubling of context length",
                "exploratory": True,
                "note": ("逐种子 OLS 斜率 ΔEM~log2(ctx) 的单样本配对检验；"
                         "正值表示候选相对参照的优势随长度增大。斜率依赖"
                         "评测点间距，仅作趋势方向证据，不替代逐 ctx 表。"),
            })
            res = _enrich_paired(res, slopes, [0.0] * len(slopes))
            if slope_skipped:
                res["excluded_slope_seeds"] = slope_skipped
            slope_tests.append(res)
            slope_row = res
        else:
            slope_row = None

        # 首末落差：不假定线性，作为斜率的稳健对照。
        spans = []
        span_skipped = 0
        for r in kept:
            sp = span_contrast(ctxs, [r["diffs"][c] for c in ctxs])
            if sp is None:
                span_skipped += 1
            else:
                spans.append(sp)
        if len(spans) >= 1:
            res = paired_t_test(spans, [0.0] * len(spans))
            res.update({
                "candidate": cand, "reference": ref,
                "metric": LENGTH_SPAN_METRIC,
                "lower_is_better": False,
                "contrast_type": "length_trend",
                "ruler_length_trend": "span",
                "contexts": ctxs,
                "short_ctx": ctxs[0], "long_ctx": ctxs[-1],
                "unit": "EM(longest) - EM(shortest)",
                "exploratory": True,
                "note": ("逐种子“最长−最短上下文”配对差的单样本检验；"
                         "不假定线性，与斜率方向不一致时说明效应集中在中间"
                         "档位。"),
            })
            res = _enrich_paired(res, spans, [0.0] * len(spans))
            if span_skipped:
                res["excluded_span_seeds"] = span_skipped
            span_tests.append(res)
            span_row = res
        else:
            span_row = None

        overall_vals = [r["overall_diff"] for r in kept
                        if r["overall_diff"] is not None]
        ctx_avg_vals = [r["ctx_average_diff"] for r in kept
                        if r["ctx_average_diff"] is not None]
        overall_mean = (sum(overall_vals) / len(overall_vals)
                        if overall_vals else None)
        ctx_avg_mean = (sum(ctx_avg_vals) / len(ctx_avg_vals)
                        if ctx_avg_vals else None)
        # 平衡性：总体 macro 应等于各档 macro 的等权平均。不等说明任务×
        # 档位有缺失，分层均值不是总体均值的严格分解，必须显式提示。
        balanced = None
        if overall_mean is not None and ctx_avg_mean is not None:
            balanced = _length_close(overall_mean, ctx_avg_mean)
        per_ctx_means = [c.get("mean_diff") for c in per_context]
        pos = sum(1 for v in per_ctx_means if v is not None and v > 0)
        neg = sum(1 for v in per_ctx_means if v is not None and v < 0)
        zero = len(per_ctx_means) - pos - neg
        longest = per_context[-1].get("mean_diff") if per_context else None
        slope_mean = slope_row.get("mean_diff") if slope_row else None
        span_mean = span_row.get("mean_diff") if span_row else None
        # 留一上下文（leave-one-ctx-out）：只有 2~3 档 ctx 时，斜率可能
        # 完全由某一档的偶然波动驱动。逐种子在去掉任一档后重新拟合，
        # 再看跨种子平均斜率符号是否翻转；翻转说明“趋势”不是各档共同的
        # 方向，而是单点造成。与留一法检查种子同构，只作描述性稳健性。
        loo_ctx_means = {}
        if len(ctxs) >= 3:
            for j, ctx in enumerate(ctxs):
                sub = [c for k, c in enumerate(ctxs) if k != j]
                vals = []
                for r in kept:
                    sl = log2_slope(sub, [r["diffs"][c] for c in sub])
                    if sl is not None:
                        vals.append(sl)
                if vals:
                    loo_ctx_means[str(ctx)] = sum(vals) / len(vals)
        slope_loo_stable = None
        if loo_ctx_means and slope_mean is not None:
            if abs(slope_mean) <= 1e-12:
                slope_loo_stable = None
            else:
                sign = slope_mean > 0
                slope_loo_stable = all(
                    (v > 0) == sign for v in loo_ctx_means.values())
        reversal_long = (overall_mean is not None and longest is not None
                         and overall_mean * longest < 0
                         and abs(longest) > 1e-12)
        reversal_slope = (overall_mean is not None and slope_mean is not None
                          and overall_mean * slope_mean < 0
                          and abs(slope_mean) > 1e-12)
        reversal_span = (overall_mean is not None and span_mean is not None
                         and overall_mean * span_mean < 0
                         and abs(span_mean) > 1e-12)
        # 总体在 SESOI 内、最长档却超过 SESOI：总平均“看起来没差别”，
        # 但方法恰在声称有效的长上下文上失效/获益。零与负号不构成严格
        # sign flip，因此单列 masking，避免用“0 组方向相反”掩盖这一情形。
        longest_masking = bool(
            longest is not None and overall_mean is not None
            and abs(longest) >= LENGTH_SESOI
            and abs(overall_mean) < LENGTH_SESOI)
        heterogeneity = None
        finite_means = [v for v in per_ctx_means if v is not None]
        if finite_means:
            heterogeneity = max(finite_means) - min(finite_means)
        pair_rows.append({
            "candidate": cand, "reference": ref,
            "n_seeds": n_seeds,
            "contexts": ctxs,
            "n_contexts": len(ctxs),
            "minimum_contexts_ok": len(ctxs) >= LENGTH_MIN_CONTEXTS,
            "overall_mean_diff": overall_mean,
            "context_average_mean_diff": ctx_avg_mean,
            "balanced_contexts": balanced,
            "per_context": [
                {"ctx": c.get("ruler_ctx"),
                 "candidate_mean": c.get("candidate_mean"),
                 "reference_mean": c.get("reference_mean"),
                 "mean_diff": c.get("mean_diff"),
                 "std_diff": c.get("std_diff"),
                 "mean_diff_ci95": c.get("mean_diff_ci95"),
                 "mde": c.get("mde"),
                 "p": c.get("p"), "p_adj": c.get("p_adj"),
                 "p_perm": c.get("p_perm"),
                 "p_perm_adj": c.get("p_perm_adj"),
                 "n": c.get("n"),
                 "n_seed_wins": c.get("n_seed_wins"),
                 "n_seed_losses": c.get("n_seed_losses"),
                 "n_seed_ties": c.get("n_seed_ties"),
                 "loo_sign_stable": c.get("loo_sign_stable"),
                 "within_noise": c.get("within_noise"),
                 "verdict": c.get("verdict")}
                for c in per_context],
            "slope_per_doubling": (None if slope_row is None else {
                "mean_diff": slope_row.get("mean_diff"),
                "std_diff": slope_row.get("std_diff"),
                "mean_diff_ci95": slope_row.get("mean_diff_ci95"),
                "mde": slope_row.get("mde"),
                "p": slope_row.get("p"), "p_adj": slope_row.get("p_adj"),
                "p_perm": slope_row.get("p_perm"),
                "p_perm_adj": slope_row.get("p_perm_adj"),
                "n": slope_row.get("n"),
                "n_seed_wins": slope_row.get("n_seed_wins"),
                "n_seed_losses": slope_row.get("n_seed_losses"),
                "n_seed_ties": slope_row.get("n_seed_ties"),
                "loo_sign_stable": slope_row.get("loo_sign_stable"),
                "within_noise": slope_row.get("within_noise"),
                "verdict": slope_row.get("verdict")}),
            "long_minus_short": (None if span_row is None else {
                "short_ctx": ctxs[0], "long_ctx": ctxs[-1],
                "mean_diff": span_row.get("mean_diff"),
                "std_diff": span_row.get("std_diff"),
                "mean_diff_ci95": span_row.get("mean_diff_ci95"),
                "mde": span_row.get("mde"),
                "p": span_row.get("p"), "p_adj": span_row.get("p_adj"),
                "p_perm": span_row.get("p_perm"),
                "p_perm_adj": span_row.get("p_perm_adj"),
                "n": span_row.get("n"),
                "n_seed_wins": span_row.get("n_seed_wins"),
                "n_seed_losses": span_row.get("n_seed_losses"),
                "n_seed_ties": span_row.get("n_seed_ties"),
                "loo_sign_stable": span_row.get("loo_sign_stable"),
                "within_noise": span_row.get("within_noise"),
                "verdict": span_row.get("verdict")}),
            "direction_consistency": {"positive": pos, "negative": neg,
                                      "zero": zero},
            "heterogeneity_pp": (None if heterogeneity is None
                                 else heterogeneity * 100.0),
            "overall_vs_longest_sign_flip": bool(reversal_long),
            "overall_vs_slope_sign_flip": bool(reversal_slope),
            "overall_vs_span_sign_flip": bool(reversal_span),
            "long_context_masking": longest_masking,
            "slope_leave_one_ctx_out": loo_ctx_means,
            "slope_loo_sign_stable": slope_loo_stable,
            "per_seed": [
                {"seed": r["seed"],
                 "overall_diff": r["overall_diff"],
                 "context_average_diff": r["ctx_average_diff"],
                 "slope": log2_slope(ctxs, [r["diffs"][c] for c in ctxs]),
                 "long_minus_short": span_contrast(
                     ctxs, [r["diffs"][c] for c in ctxs]),
                 "diffs": {str(c): r["diffs"][c] for c in ctxs}}
                for r in kept],
            "excluded": excluded,
            "exploratory": True,
        })

    ctx_tests = _apply_holm(ctx_tests)
    slope_tests = _apply_holm(slope_tests)
    span_tests = _apply_holm(span_tests)
    for t in slope_tests:
        t["verdict"] = _length_trend_verdict(t, "slope")
    for t in span_tests:
        t["verdict"] = _length_trend_verdict(t, "span")
    # 把校正后的 p/结论回填到 pair 行，避免 summary 与 significance.json
    # 出现两套互相矛盾的判定。
    by_ctx = {(t["candidate"], t["reference"], t["ruler_ctx"]): t
              for t in ctx_tests}
    by_slope = {(t["candidate"], t["reference"]): t for t in slope_tests}
    by_span = {(t["candidate"], t["reference"]): t for t in span_tests}
    for row in pair_rows:
        key = (row["candidate"], row["reference"])
        for cell in row["per_context"]:
            t = by_ctx.get((row["candidate"], row["reference"], cell["ctx"]))
            if t is not None:
                for k in ("p_adj", "p_perm", "p_perm_adj", "verdict"):
                    cell[k] = t.get(k)
        if row["slope_per_doubling"] is not None:
            t = by_slope.get(key)
            if t is not None:
                for k in ("p_adj", "p_perm", "p_perm_adj", "verdict"):
                    row["slope_per_doubling"][k] = t.get(k)
        if row["long_minus_short"] is not None:
            t = by_span.get(key)
            if t is not None:
                for k in ("p_adj", "p_perm", "p_perm_adj", "verdict"):
                    row["long_minus_short"][k] = t.get(k)
    return {
        "pairs": pair_rows,
        "ctx_tests": ctx_tests,
        "slope_tests": slope_tests,
        "span_tests": span_tests,
        "min_seeds": MIN_SEEDS,
        "alpha": ALPHA,
        "min_contexts": LENGTH_MIN_CONTEXTS,
        "note": ("RULER 上下文长度分层与剂量-反应（探索性，未预注册）："
                 "逐 ctx 的 Δmacro EM 配对检验、逐种子斜率（ΔEM~log2 ctx）"
                 "与首末落差；三个独立 Holm 家族。用于回答“优势是否随长度"
                 "变化”，不替代预注册的 RULER平均 确认性判定。"
                 "overall_vs_longest/slope/span_sign_flip 表示总平均与长上下文/"
                 "趋势方向相反，此时只报总平均会掩盖长上下文失效。"),
    }


def _ruler_heterogeneity_analysis(runs, pairs, alpha=ALPHA,
                                  min_seeds=MIN_SEEDS,
                                  n_bootstrap: int = 500,
                                  rng_seed: int = 20261030) -> dict:
    """RULER 任务/上下文层的效应异质性（探索性）。

    对每个 (候选, 参照) 把同种子 RULER 配对差按两条轴分层：

    * ``task``：各任务的 macro EM 差（任务等权，先对上下文档位平均）；
    * ``context``：各请求上下文档位的 macro EM 差（档位内任务等权）。

    每个层先由跨种子样本均值/标准误得到一个效应估计，再做随机效应
    meta 分析（``lega.heterogeneity``）。任务层回答“macro 平均是否掩盖了
    某些任务的反向效应”；上下文层是既有长度剂量-反应分析的补充，用
    I²/Q 与预测区间描述层间方差，不替代斜率/首末落差检验。

    只用所有种子都有定义的层（完整矩阵）；缺失层显式记入 ``dropped_labels``，
    不按 0 补齐。跨层共享种子和评测样例，Q 的 p 值与 I² 只作描述性指标。
    """
    runs = runs if isinstance(runs, dict) else {}
    outcomes = []
    for entry in (pairs if isinstance(pairs, list) else []):
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            continue
        cand, ref = entry
        try:
            if cand not in runs or ref not in runs:
                continue
        except TypeError:
            continue
        cm = _variant_seed_map(runs, cand)
        rm = _variant_seed_map(runs, ref)
        seeds = [s for s in _sorted_seeds(cm) if s in rm]
        if not seeds:
            continue
        excluded = Counter()
        rows = []
        for s in seeds:
            sa = cm.get(s) if isinstance(cm.get(s), dict) else {}
            sb = rm.get(s) if isinstance(rm.get(s), dict) else {}
            ra = sa.get("ruler") if isinstance(sa.get("ruler"), dict) else None
            rb = sb.get("ruler") if isinstance(sb.get("ruler"), dict) else None
            if ra is None or rb is None:
                excluded["missing_ruler"] += 1
                continue
            if not _ruler_pair_protocol_ok(ra, rb):
                excluded["protocol_mismatch"] += 1
                continue
            ta = _ruler_task_means(ra)
            tb = _ruler_task_means(rb)
            ca = context_macro_means(ra, align_requested=True)
            cb = context_macro_means(rb, align_requested=True)
            if not ta or not tb:
                excluded["missing_tasks"] += 1
                continue
            if not ca or not cb:
                excluded["missing_contexts"] += 1
                continue
            rows.append({
                "seed": _integral_value(s),
                "task": (ta, tb),
                "context": (ca, cb),
            })
        if not rows:
            continue
        axes = {}
        for axis, axis_cn, unit in (
                ("task", "RULER 任务", "task macro EM difference"),
                ("context", "RULER 上下文长度", "context macro EM difference")):
            all_a = set.union(*[set(r[axis][0]) for r in rows])
            all_b = set.union(*[set(r[axis][1]) for r in rows])
            common = set.intersection(
                *[set(r[axis][0]) & set(r[axis][1]) for r in rows])
            labels = sorted(common)
            dropped = sorted((all_a | all_b) - common)
            if len(labels) < 2:
                axes[axis] = {
                    "status": "insufficient",
                    "label": "共同层数不足，无法评估异质性",
                    "axis": axis,
                    "axis_label": axis_cn,
                    "unit": unit,
                    "k": len(labels),
                    "n_seeds": len(rows),
                    "labels": labels,
                    "dropped_labels": dropped,
                    "per_stratum": [],
                    "bootstrap": {"available": False},
                    "excluded": dict(excluded),
                    "seed_sufficient": len(rows) >= min_seeds,
                    "exploratory": True,
                }
                continue
            matrix = [
                [r[axis][0][label] - r[axis][1][label] for label in labels]
                for r in rows
            ]
            res = meta_analysis_from_seed_matrix(
                matrix, labels=labels, alpha=alpha, min_k=2,
                n_bootstrap=n_bootstrap, seed=rng_seed)
            res.update({
                "axis": axis,
                "axis_label": axis_cn,
                "unit": unit,
                "n_seeds": len(rows),
                "dropped_labels": dropped,
                "excluded": dict(excluded),
                "seed_sufficient": len(rows) >= min_seeds,
                "exploratory": True,
            })
            if len(rows) < min_seeds:
                res["note"] = (
                    f"种子数 n={len(rows)} < {min_seeds}，仅作描述性展示，"
                    "不据此判定异质性；" + str(res.get("note") or ""))
            axes[axis] = res
        if not axes:
            continue
        statuses = [a.get("status") for a in axes.values()]
        if "heterogeneous" in statuses:
            status = "heterogeneous"
        elif "moderate_heterogeneity" in statuses:
            status = "moderate_heterogeneity"
        elif "no_detected_heterogeneity" in statuses:
            status = "no_detected_heterogeneity"
        else:
            status = "insufficient"
        outcomes.append({
            "candidate": cand,
            "reference": ref,
            "metric": "RULER平均",
            "status": status,
            "axes": axes,
            "n_seeds": len(rows),
            "excluded": dict(excluded),
            "exploratory": True,
        })
    counts = Counter(o["status"] for o in outcomes)
    return {
        "method": ("RULER 任务/上下文长度分层随机效应 meta 分析"
                   "（DerSimonian-Laird；I²/Q/预测区间/留一层/bootseed）"),
        "alpha": alpha,
        "min_seeds": min_seeds,
        "min_strata": 2,
        "n_bootstrap": n_bootstrap,
        "counts": dict(counts),
        "outcomes": outcomes,
        "note": ("探索性、未预注册：任务层与上下文层共享种子及评测样例，"
                 "不是独立研究，Q 的卡方 p 值与 I² 只作描述性异质性指标；"
                 "bootstrap 只重采样训练种子。未检出异质性不等于各层同质，"
                 "I² 高也不表示某一层显著。不改变 H1–H7 的确认性判定，"
                 "也不能替代逐层检验、长度剂量-反应、SESOI/等价性或 MDE。"),
    }


def _ruler_macro_bootstrap_ci(per_seed_cell_diffs, n_boot: int = RULER_MACRO_BOOTSTRAP,
                              rng_seed: int = 20240517):
    """RULER macro 的分层 bootstrap 区间（固定种子，单元格内重采样样例）。

    预注册的 macro 估计量是“任务等权、任务内上下文档位等权”：先对每个
    (任务, 上下文) 单元格内的样例重采样并求均值，再在任务内对单元格等权
    平均，最后对任务等权平均；直接重采样所有样例会退化成 micro，只对
    单元格等权则会在任务间格数不等时偏离预注册口径。固定 rng_seed 保证
    同一输入产生同一区间，便于审计复现。
    """
    n_boot = strict_int(n_boot, "n_boot")
    if n_boot < 1:
        raise ValueError(f"n_boot 必须 >= 1，收到 {n_boot!r}")
    rng_seed = strict_int(rng_seed, "rng_seed")
    if rng_seed < 0:
        raise ValueError(f"rng_seed 必须 >= 0，收到 {rng_seed!r}")
    cells = [c for c in (per_seed_cell_diffs or []) if isinstance(c, dict)]
    by_seed = []
    for c in cells:
        by_task = {}
        for key, vals in c.items():
            arr = np.asarray(vals, dtype=float)
            if arr.size == 0:
                continue
            # cell key 约定为 (task, ctx)；旧/手工输入不是序列时按整键成组。
            task = key[0] if isinstance(key, (tuple, list)) and key else key
            by_task.setdefault(task, []).append(arr)
        if by_task:
            by_seed.append(by_task)
    if not by_seed:
        return None
    rng = np.random.default_rng(rng_seed)
    reps = []
    for _ in range(n_boot):
        seed_means = []
        for by_task in by_seed:
            task_means = []
            for cell_arrays in by_task.values():
                cell_means = []
                for vals in cell_arrays:
                    idx = rng.integers(0, vals.size, size=vals.size)
                    cell_means.append(float(vals[idx].mean()))
                if cell_means:
                    task_means.append(float(np.mean(cell_means)))
            if task_means:
                seed_means.append(float(np.mean(task_means)))
        if seed_means:
            reps.append(float(np.mean(seed_means)))
    if not reps:
        return None
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return [float(lo), float(hi)]


def _ruler_macro_hierarchical_bootstrap(per_seed_cell_diffs,
                                        n_boot: int = RULER_MACRO_BOOTSTRAP,
                                        rng_seed: int = 20240517):
    """RULER macro 的双层 bootstrap：同时重采样种子与评测样例。

    `_ruler_macro_bootstrap_ci` 固定已观测种子，只覆盖“换一批评测样例”
    的抽样误差；跨种子 t 检验只覆盖“换训练种子”的随机性。两者都不能单独
    给出“换种子且换评测样例”时的合并不确定性。本函数先对种子有放回重采样，
    再在每个被选中种子的 (任务, 上下文) 单元格内对样例有放回重采样，按与
    预注册 macro 相同的“任务等权、任务内单元格等权”口径聚合。

    返回 ``{ci95, p, same_sign, observed, n_boot, n_seeds, n_cells,
    n_samples, method}``；``p`` 由中心化 bootstrap 分布计算，``same_sign``
    是重采样均值与观测值同号的比例。小种子数下 bootstrap 分布本身粒度很粗，
    该区间只作探索性稳健性证据，不替代预注册的种子级检验。
    """
    n_boot = strict_int(n_boot, "n_boot")
    if n_boot < 1:
        raise ValueError(f"n_boot 必须 >= 1，收到 {n_boot!r}")
    rng_seed = strict_int(rng_seed, "rng_seed")
    if rng_seed < 0:
        raise ValueError(f"rng_seed 必须 >= 0，收到 {rng_seed!r}")
    cells = [c for c in (per_seed_cell_diffs or []) if isinstance(c, dict)]
    by_seed = []
    structures = []
    n_cells = n_samples = 0
    for c in cells:
        by_task = {}
        structure = []
        payload_cells = payload_samples = 0
        for key, vals in c.items():
            arr = np.asarray(vals, dtype=float)
            if arr.size == 0:
                continue
            task = key[0] if isinstance(key, (tuple, list)) and key else key
            by_task.setdefault(task, []).append(arr)
            structure.append((str(task), str(key), int(arr.size)))
            payload_cells += 1
            payload_samples += int(arr.size)
        if by_task:
            by_seed.append(by_task)
            structures.append(tuple(sorted(structure, key=str)))
            n_cells += payload_cells
            n_samples += payload_samples
    if not by_seed:
        return None
    # 各被选中种子的任务/单元格结构必须一致，否则“先重采样种子再按任务
    # 等权”会把结构差异误当成效应。协议一致的 RULER 产物应满足该条件。
    if any(s != structures[0] for s in structures[1:]):
        return None

    def _macro(by_task, rng=None):
        task_means = []
        for cell_arrays in by_task.values():
            cell_means = []
            for vals in cell_arrays:
                if rng is None:
                    cell_means.append(float(vals.mean()))
                else:
                    idx = rng.integers(0, vals.size, size=vals.size)
                    cell_means.append(float(vals[idx].mean()))
            if cell_means:
                task_means.append(float(np.mean(cell_means)))
        return float(np.mean(task_means)) if task_means else None

    observed_seed_macros = [m for m in (_macro(bt) for bt in by_seed)
                            if m is not None]
    if not observed_seed_macros:
        return None
    observed = float(np.mean(observed_seed_macros))
    rng = np.random.default_rng(rng_seed)
    reps = []
    for _ in range(n_boot):
        seed_idx = rng.integers(0, len(by_seed), size=len(by_seed))
        seed_macros = []
        for si in seed_idx:
            m = _macro(by_seed[int(si)], rng)
            if m is not None:
                seed_macros.append(m)
        if seed_macros:
            reps.append(float(np.mean(seed_macros)))
    if not reps:
        return None
    lo, hi = np.percentile(reps, [2.5, 97.5])
    reps_arr = np.asarray(reps, dtype=float)
    if observed == 0.0:
        p = 1.0
        same_sign = 0.0
    else:
        ge = int(np.sum(np.abs(reps_arr - observed) >= abs(observed)))
        p = (ge + 1) / (len(reps_arr) + 1)
        same_sign = float(np.mean((reps_arr > 0) == (observed > 0)))
    return {
        "ci95": [float(lo), float(hi)],
        "p": float(p),
        "same_sign": same_sign,
        "observed": observed,
        "n_boot": int(n_boot),
        "n_seeds": len(by_seed),
        "n_cells": int(n_cells),
        "n_samples": int(n_samples),
        "method": ("hierarchical bootstrap: resample training seeds, then "
                   "resample paired samples within each (task, ctx) cell; "
                   "task-equal macro"),
    }


def _ruler_macro_variance_components(per_seed_cell_diffs,
                                     per_seed_cell_ids):
    """RULER macro 的种子/样例方差分解（按单元格等权聚合）。

    每个 (任务, 上下文) 单元格内是 seed × sample 的配对差矩阵，可分解为
    种子效应、样例效应与残差；再按预注册 macro 权重（任务等权、任务内
    单元格等权）聚合各单元格的方差占比。只使用所有种子都存在的单元格且
    要求 sample_id 完全对齐；不平衡/缺格不插补，返回 ``None``。
    """
    diffs = [c for c in (per_seed_cell_diffs or []) if isinstance(c, dict)]
    ids = [c for c in (per_seed_cell_ids or []) if isinstance(c, dict)]
    if len(diffs) < 2 or len(ids) != len(diffs):
        return None
    common = set(diffs[0])
    for c in diffs[1:]:
        common &= set(c)
    if not common:
        return None
    comps, weights, tasks = [], [], []
    skipped = 0
    for ck in sorted(common, key=str):
        ref_ids = ids[0].get(ck)
        if not isinstance(ref_ids, (list, tuple)) or not ref_ids:
            skipped += 1
            continue
        if any(tuple(ids[i].get(ck) or ()) != tuple(ref_ids)
               for i in range(1, len(ids))):
            skipped += 1
            continue
        matrix = []
        ok = True
        for c in diffs:
            vals = c.get(ck)
            if (not isinstance(vals, (list, tuple, np.ndarray))
                    or len(vals) != len(ref_ids)):
                ok = False
                break
            matrix.append(list(vals))
        if not ok:
            skipped += 1
            continue
        comp = two_way_variance_components(matrix)
        if comp is None:
            skipped += 1
            continue
        task = ck[0] if isinstance(ck, (tuple, list)) and ck else ck
        comps.append(comp)
        tasks.append(str(task))
    if not comps:
        return None
    counts = Counter(tasks)
    n_tasks = len(counts)
    weights = [1.0 / (n_tasks * counts[task]) for task in tasks]
    agg = aggregate_variance_components(comps, weights)
    if agg is None:
        return None
    agg.update({
        "n_cells": len(comps),
        "n_cells_common": len(common),
        "n_cells_skipped": skipped,
        "n_seeds": len(diffs),
        "scope": "RULER macro cells (task-equal, cell-equal)",
    })
    return agg


def _ruler_macro_absolute_ci(ruler_payloads, n_boot: int = RULER_MACRO_BOOTSTRAP,
                             rng_seed: int = 20240517):
    """每个变体 RULER 绝对 macro EM 的分层 bootstrap 区间。

    点估计 `ruler_avg` 把任务等权、任务内上下文档位等权；区间必须匹配该
    估计量，因此在每个 (任务, 上下文) 单元格内对样例有放回重采样，先算
    单元格均值、再任务内与任务间等权平均。固定已观测种子，只覆盖评测样例
    抽样，不覆盖训练种子随机性（后者见跨种子 Mean ± Std 与配对 t 检验）。
    返回 {ci95, n_seeds, n_cells, n_samples, method}；旧产物缺逐样例记录或
    任一单元格为空时返回 None，不伪造区间。
    """
    n_boot = strict_int(n_boot, "n_boot")
    if n_boot < 1:
        raise ValueError(f"n_boot 必须 >= 1，收到 {n_boot!r}")
    rng_seed = strict_int(rng_seed, "rng_seed")
    if rng_seed < 0:
        raise ValueError(f"rng_seed 必须 >= 0，收到 {rng_seed!r}")
    per_seed = []
    n_cells = n_samples = 0
    for d in ruler_payloads or []:
        cells = _ruler_cell_sample_outcomes(d)
        if not cells:
            # 旧产物/损坏 payload 没有逐样例记录：不能只用其余种子算区间，
            # 否则 n_seeds 与实际观测集不符，静默丢弃一个种子。
            return None
        by_task = {}
        payload_cells = payload_samples = 0
        for (task, _ctx), samples in cells.items():
            vals = np.asarray(list(samples.values()), dtype=float)
            if vals.size == 0:
                # 任一单元格为空时无法保持预注册的“任务内单元格等权”
                # 估计量；按不可用处理，不伪造区间。
                return None
            by_task.setdefault(task, []).append(vals)
            payload_cells += 1
            payload_samples += int(vals.size)
        n_cells += payload_cells
        n_samples += payload_samples
        per_seed.append(by_task)
    if not per_seed:
        return None
    rng = np.random.default_rng(rng_seed)
    reps = []
    for _ in range(n_boot):
        seed_macros = []
        for by_task in per_seed:
            task_means = []
            for arrays in by_task.values():
                cell_means = []
                for vals in arrays:
                    idx = rng.integers(0, vals.size, size=vals.size)
                    cell_means.append(float(vals[idx].mean()))
                task_means.append(float(np.mean(cell_means)))
            seed_macros.append(float(np.mean(task_means)))
        reps.append(float(np.mean(seed_macros)))
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return {
        "ci95": [float(lo), float(hi)],
        "n_seeds": len(per_seed),
        "n_cells": n_cells,
        "n_samples": n_samples,
        "method": ("fixed observed seeds; resample samples within each "
                   "(task, ctx) cell; task-equal macro; average over seeds"),
    }


def _ruler_sample_paired_tests(runs, pairs) -> list:
    """固定评测样例上的配对 McNemar 检验（探索性稳健性证据）。

    跨种子 t 检验把每个种子的聚合 EM 当作一次观测，回答“换训练种子是否
    稳定”；但每格只有 n=20~30 个样例，聚合 EM 的评测抽样误差没有进入
    该检验。变体共享同一 sample_seed 与 sample_id，因此可以在**同一批
    题面**上逐样例配对：候选对、参照错 = 候选胜，反之 = 参照胜，相同
    结果不提供方向信息。McNemar 精确检验只消费 discordant 对，避免把
    “两边都答对/都答错”的样例算成独立证据。

    多种子合并时，每个种子内先按 sample_id 配对，再把各 seed 的
    discordant 计数相加做精确检验（同一样例在不同训练种子下重复出现，
    故该合并口径只作探索性证据；逐种子明细一并落盘供核对）。
    同时报告两种效应量口径：`mean_diff` 为样例等权 micro（与 McNemar
    池化计数一致），`macro_mean_diff` 为任务等权、任务内上下文档位等权
    （与预注册 `RULER平均` 的 macro 估计量一致）；两者不一致说明结论依赖
    “平均值”定义。
    该检验与跨种子 t 检验回答不同问题，不能互相替代：前者是“固定题面
    上的样例级差异”，后者是“训练随机性下的种子级差异”。
    """
    out = []
    runs = runs if isinstance(runs, dict) else {}
    for cand, ref in (pairs if isinstance(pairs, list) else []):
        if cand not in runs or ref not in runs:
            continue
        cm = _variant_seed_map(runs, cand)
        rm = _variant_seed_map(runs, ref)
        seeds = [s for s in _sorted_seeds(cm) if s in rm]
        cand_only = ref_only = both_correct = both_wrong = 0
        n_samples = 0
        per_seed = []
        per_seed_cell_diffs = []
        per_seed_cell_ids = []
        # 按 sample_id 聚合跨种子的配对差：同一样例在多个训练种子下重复
        # 出现，pooled McNemar 把它当成多条独立证据；聚类分析先对每个
        # 样例跨种子取平均，再以样例为独立单位做符号翻转检验。
        cluster_sums = {}
        skipped = 0
        for s in seeds:
            sa = cm.get(s) if isinstance(cm.get(s), dict) else {}
            sb = rm.get(s) if isinstance(rm.get(s), dict) else {}
            ra = sa.get("ruler") if isinstance(sa.get("ruler"), dict) else None
            rb = sb.get("ruler") if isinstance(sb.get("ruler"), dict) else None
            if ra is None or rb is None or not _ruler_pair_protocol_ok(ra, rb):
                skipped += 1
                continue
            ca = _ruler_cell_sample_outcomes(ra)
            cb = _ruler_cell_sample_outcomes(rb)
            if ca is None or cb is None or set(ca) != set(cb):
                skipped += 1
                continue
            cell_diffs = []
            cell_paired = {}
            cell_ids = {}
            cell_ok = True
            for ck in sorted(ca):
                oa, ob = ca[ck], cb[ck]
                if set(oa) != set(ob):
                    cell_ok = False
                    break
                if not oa:
                    continue
                ids = tuple(sorted(oa))
                diffs = [oa[sid] - ob[sid] for sid in ids]
                cell_paired[ck] = diffs
                cell_ids[ck] = ids
                # 预注册 macro = 任务等权、任务内上下文档位等权：先按任务
                # 收集单元格差，最后先任务内平均、再任务间平均。不能直接
                # 对所有单元格等权——任务间格数不等时会偏离预注册估计量。
                task = ck[0] if isinstance(ck, (tuple, list)) and ck else ck
                cell_diffs.append((task, sum(diffs) / len(diffs)))
            if not cell_ok:
                skipped += 1
                continue
            per_seed_cell_diffs.append(cell_paired)
            per_seed_cell_ids.append(cell_ids)
            task_cell_diffs = {}
            for task, diff in cell_diffs:
                task_cell_diffs.setdefault(task, []).append(diff)
            macro_seed = (float(np.mean([np.mean(v) for v in
                                         task_cell_diffs.values()]))
                          if task_cell_diffs else None)
            oa = {sid: em for cell_out in ca.values()
                  for sid, em in cell_out.items()}
            ob = {sid: em for cell_out in cb.values()
                  for sid, em in cell_out.items()}
            for sid in oa:
                cluster_sums.setdefault(sid, []).append(oa[sid] - ob[sid])
            s_cand = s_ref = s_both1 = s_both0 = 0
            for sid in sorted(oa):
                a, b = oa[sid], ob[sid]
                if a == 1.0 and b == 0.0:
                    s_cand += 1
                elif a == 0.0 and b == 1.0:
                    s_ref += 1
                elif a == 1.0 and b == 1.0:
                    s_both1 += 1
                else:
                    s_both0 += 1
            n_s = len(oa)
            per_seed.append({
                "seed": _integral_value(s), "n": n_s,
                "candidate_only": s_cand, "reference_only": s_ref,
                "both_correct": s_both1, "both_wrong": s_both0,
                "mean_diff": ((s_cand - s_ref) / n_s) if n_s else None,
                "macro_mean_diff": macro_seed,
                "n_cells": len(cell_diffs),
                "p": _mcnemar_exact_p(s_cand, s_ref),
            })
            cand_only += s_cand
            ref_only += s_ref
            both_correct += s_both1
            both_wrong += s_both0
            n_samples += n_s
        if not per_seed:
            continue
        wins = sum(1 for r in per_seed
                   if r["mean_diff"] is not None and r["mean_diff"] > 0)
        losses = sum(1 for r in per_seed
                     if r["mean_diff"] is not None and r["mean_diff"] < 0)
        ties = len(per_seed) - wins - losses
        mean_diff = ((cand_only - ref_only) / n_samples) if n_samples else None
        macro_diffs = [r["macro_mean_diff"] for r in per_seed
                       if r.get("macro_mean_diff") is not None]
        macro_mean_diff = (sum(macro_diffs) / len(macro_diffs)
                           if macro_diffs else None)
        # macro 的抽样不确定性：固定已观测种子，在每个 (任务, 上下文) 单元格
        # 内对配对样例重采样，再任务内、任务间等权平均（与预注册 macro 同口径）。
        # 该区间只覆盖评测样例抽样，不覆盖训练种子随机性（后者见种子级 t 检验）。
        macro_ci = _ruler_macro_bootstrap_ci(per_seed_cell_diffs)
        # 双层 bootstrap：种子与样例同时重采样，给出合并不确定性；方差分解
        # 则回答两个来源各自占多少，避免“区间很宽但不知道该补哪一边”。
        macro_hier = _ruler_macro_hierarchical_bootstrap(per_seed_cell_diffs)
        macro_variance = _ruler_macro_variance_components(
            per_seed_cell_diffs, per_seed_cell_ids)
        # 样例聚类（cluster-robust）：每个 sample_id 先跨种子平均，再作为
        # 独立观测做符号翻转检验；只有 1 个样例时不做检验（n<2）。
        cluster_vals = [float(np.mean(v)) for v in cluster_sums.values() if v]
        cluster_test = paired_sign_flip_test(cluster_vals,
                                             [0.0] * len(cluster_vals))
        cluster_ci = bootstrap_mean_ci(cluster_vals)
        n_cells = (per_seed[0].get("n_cells")
                   if per_seed and len({r.get("n_cells") for r in per_seed}) == 1
                   else None)
        res = {
            "candidate": cand, "reference": ref,
            "metric": "RULER样例配对", "lower_is_better": False,
            "sample_level": True, "exploratory": True,
            "n": len(per_seed), "n_seeds": len(per_seed),
            "n_samples": n_samples,
            "candidate_only": cand_only, "reference_only": ref_only,
            "both_correct": both_correct, "both_wrong": both_wrong,
            "mean_diff": mean_diff,
            "macro_mean_diff": macro_mean_diff,
            "macro_mean_diff_ci95": macro_ci,
            "macro_ci_method": (
                "cell-stratified sample bootstrap (seeds fixed): resample "
                "paired samples within each (task, ctx) cell, keep cells "
                "equal-weighted, then average over observed seeds"),
            "macro_mean_diff_ci95_hier": (
                macro_hier.get("ci95") if macro_hier else None),
            "macro_p_hier": (macro_hier.get("p") if macro_hier else None),
            "macro_bootstrap_same_sign_hier": (
                macro_hier.get("same_sign") if macro_hier else None),
            "macro_ci95_hier_method": (
                macro_hier.get("method") if macro_hier else None),
            "variance_components": macro_variance,
            "n_cells": n_cells,
            "p": _mcnemar_exact_p(cand_only, ref_only),
            "p_adj": None,
            "cluster_n": len(cluster_vals),
            "cluster_mean_diff": cluster_test.get("mean_diff"),
            "cluster_mean_diff_ci95": cluster_ci,
            "cluster_p": cluster_test.get("p_perm"),
            "cluster_p_exact": cluster_test.get("exact"),
            "cluster_p_adj": None,
            "cluster_verdict": None,
            "n_seed_wins": wins, "n_seed_losses": losses,
            "n_seed_ties": ties,
            "per_seed": per_seed,
            "verdict": None,
            "note": ("同一批 sample_id 上的逐样例配对；只消费 discordant "
                     "对（候选对/参照错 vs 反之），与跨种子聚合 EM 检验"
                     "回答不同问题，属探索性证据。mean_diff 为样例等权 "
                     "micro，macro_mean_diff 为任务等权、任务内上下文档位"
                     "等权（对应预注册 RULER平均 的 macro 口径）；"
                     "macro_mean_diff_ci95 为"
                     "固定种子、单元格内样例重采样的分层 bootstrap 区间"
                     "（只覆盖评测抽样，不覆盖训练随机性）；"
                     "macro_mean_diff_ci95_hier 同时重采样种子与样例，"
                     "给出合并不确定性；variance_components 把 macro 配对"
                     "差的方差分解为种子/样例/残差三部分。cluster_* 为"
                     "按 sample_id 聚合跨种子后的样例级稳健性检验：每个"
                     "样例先平均、再以样例为独立单位做符号翻转，避免把"
                     "同一样例在多个种子下的重复观测当成独立证据"),
        }
        if skipped:
            res["excluded_runs"] = skipped
        out.append(res)
    pads = holm_bonferroni([t.get("p") for t in out])
    for t, padj in zip(out, pads):
        t["p_adj"] = padj
        if t.get("n_samples", 0) <= 0 or padj is None:
            t["verdict"] = "无法检验（缺少逐样例结果）"
        elif padj < ALPHA:
            t["verdict"] = ("样例级显著优于（McNemar 精确，Holm 校正后）"
                            if t["mean_diff"] > 0 else
                            "样例级显著劣于（McNemar 精确，Holm 校正后）")
        else:
            t["verdict"] = (f"样例级无显著差异（McNemar 精确，Holm 校正后 "
                            f"p={padj:.4f}）")
        md_, mc_ = t.get("mean_diff"), t.get("macro_mean_diff")
        if (md_ is not None and mc_ is not None and md_ != 0
                and (mc_ == 0 or (md_ > 0) != (mc_ > 0))):
            t["estimand_conflict"] = True
            t["verdict"] += ("；⚠ micro/macro 方向不一致（macro 为 0 或反向），"
                             "结论依赖“平均值”定义")
    # 聚类检验独立 Holm 家族：以 sample_id 为独立单位，避免同一样例在
    # 多个训练种子下被重复计数。它只覆盖 micro（样例等权）口径。
    cpads = holm_bonferroni([t.get("cluster_p") for t in out])
    for t, padj in zip(out, cpads):
        t["cluster_p_adj"] = padj
        if t.get("cluster_n", 0) < 2 or padj is None:
            t["cluster_verdict"] = "无法检验（独立样例数不足）"
        elif padj < ALPHA:
            t["cluster_verdict"] = (
                "样例聚类后显著优于（符号翻转，Holm 校正后）"
                if t["cluster_mean_diff"] > 0 else
                "样例聚类后显著劣于（符号翻转，Holm 校正后）")
        else:
            t["cluster_verdict"] = (f"样例聚类后无显著差异（Holm 校正后 "
                                    f"p={padj:.4f}）")
    # 双层 bootstrap 也是独立 Holm 家族：它同时覆盖种子与样例，和上面的
    # McNemar（固定题面）及聚类检验（样例等权 micro）回答不同问题。
    hpads = holm_bonferroni([t.get("macro_p_hier") for t in out])
    for t, padj in zip(out, hpads):
        t["macro_p_hier_adj"] = padj
        delta = t.get("macro_mean_diff")
        if padj is None or delta is None:
            t["macro_hier_verdict"] = "无法检验（双层 bootstrap 自由度不足）"
        elif padj < ALPHA:
            t["macro_hier_verdict"] = (
                "双层 bootstrap 支持候选更优（探索性，Holm 校正后）"
                if delta > 0 else
                "双层 bootstrap 支持候选更差（探索性，Holm 校正后）")
        else:
            t["macro_hier_verdict"] = (
                f"双层 bootstrap 未显示差异（探索性，Holm 校正后 "
                f"p={padj:.4f}）")
    return out
