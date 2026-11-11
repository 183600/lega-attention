"""确认性判定管线的蒙特卡洛校准（simulation-based calibration / 仿真功效）。

背景：确认性判定并不是“一次 t 检验”，而是四层串联的规则：

1. 同一指标家族内的 Holm 校正（探索性配对同样进入家族）；
2. 全部预注册假设的全局 Holm 校正（用未校正的 p）；
3. 最小种子数门（``n < min_seeds`` 不作判定）；
4. test look 预算的保守阈值 ``alpha / planned_looks``。

这套规则是本项目自己写的，“名义 α=0.05”与“实际整体假阳性率”之间没有任何
解析等式；解析功效（正态/t 近似）还额外假设了独立同分布与正态性。因此设计
阶段除了给解析灵敏度曲线，还必须**直接校准这条规则本身**：在全部为真的场景
下模拟整条判定管线，看它到底有多大概率产生至少一个假阳性；在给定备择下
模拟它的实际功效。

模拟口径（与真实分析同源）：

* 推断单位是**种子**：一条检验对应 ``n_seeds`` 个配对差；
* 零假设下配对 t 检验的双侧 p 值精确服从 ``U(0,1)``（连续对称零分布的
  性质），因此零假设检验直接抽均匀随机数，不用近似；
* 备择下按 ``d_i ~ N(effect, 1)`` 抽配对差（``effect`` 以配对差标准差为
  单位），再用 :func:`lega.utils.student_t_two_sided_p` 求双侧 p 值——与
  真实分析走同一条数值路径；
* 效应方向统一以“**该假设所声明的方向**”为正方向：这样“方向是否与登记
  一致”可以由抽样均值的符号直接判定，不需要为每条假设手写符号约定。

本模块只做纯数值模拟：不训练、不读数据、不读测试集，因此可以在开跑前单独
运行核对（见模块底部的 ``main``）。
"""

from __future__ import annotations

import math

import numpy as np

from lega.testreuse import DEFAULT_TEST_LOOK_BUDGET  # noqa: E402
from lega.utils import holm_bonferroni  # noqa: E402

from ._shared import _install as _shared_install

_shared_install(globals())


#: 预注册指标名 → 计划里的家族键。家族键与
#: ``multiplicity_plan.family_sizes`` 的键一一对应。
_METRIC_FAMILY_KEYS = (
    ("PPL", "PPL"),
    ("RULER", "RULER"),
    ("AUROC", "AUROC"),
    ("AP差", "AP"),
    ("PR-AUC", "AP"),
)

DEFAULT_N_SIM = 1000
DEFAULT_CALIBRATION_N_SIM = 1000
DEFAULT_CALIBRATION_SEED = 20260101

#: 蒙特卡洛 95% 区间的双侧正态分位数。
_Z95 = 1.959963984540054


def metric_family_key(metric) -> str:
    """预注册指标名 → 家族键；无法识别时返回 ``"OTHER"``。"""
    if not isinstance(metric, str):
        return "OTHER"
    for prefix, key in _METRIC_FAMILY_KEYS:
        if metric.startswith(prefix):
            return key
    return "OTHER"


def wilson_interval(successes: int, total: int, z: float = _Z95) -> tuple:
    """比例的 Wilson 得分区间（不依赖 scipy）。

    小样本下正态近似区间会越界到 [0,1] 之外并给出 0 宽度，Monte Carlo 的
    比例区间必须用 Wilson：n=1000、p=0.05 时它的覆盖率明显优于正态近似。
    """
    if total <= 0:
        return (0.0, 1.0)
    p = successes / total
    z2 = z * z
    denom = 1.0 + z2 / total
    center = (p + z2 / (2 * total)) / max(denom, 1e-10)
    half = (
        z
        * math.sqrt(p * (1.0 - p) / max(total, 1) + z2 / (4.0 * total * total))
        / denom
    )
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return (lo, hi)


def _plan_families(plan: dict) -> dict:
    """从设计计划解析家族结构：家族大小 + 每条登记假设占据的成员位置。"""
    plan = plan if isinstance(plan, dict) else {}
    mp = plan.get("multiplicity_plan")
    mp = mp if isinstance(mp, dict) else {}
    sizes = mp.get("family_sizes")
    sizes = sizes if isinstance(sizes, dict) else {}
    families = {}
    for key in ("PPL", "RULER", "AUROC", "AP"):
        raw = sizes.get(key)
        try:
            size = int(raw)
        except (TypeError, ValueError):
            size = 0
        families[key] = {"size": max(0, size), "registered": []}
    registered = plan.get("registered_plan")
    registered = registered if isinstance(registered, list) else []
    for item in registered:
        if not isinstance(item, dict):
            continue
        hid = item.get("id")
        if not isinstance(hid, str) or not hid:
            continue
        key = metric_family_key(item.get("metric"))
        fam = families.setdefault(key, {"size": 0, "registered": []})
        fam["registered"].append(hid)
    # 家族大小至少要装得下它自己的登记假设（否则计划本身就不自洽）。
    for key, fam in families.items():
        if fam["size"] < len(fam["registered"]):
            fam["size"] = len(fam["registered"])
        for pos, hid in enumerate(fam["registered"]):
            families[key]["registered"][pos] = (hid, pos)
    return families


def _plan_alpha(plan: dict, alpha=None) -> float:
    if isinstance(alpha, (int, float)) and not isinstance(alpha, bool):
        value = float(alpha)
        if 0.0 < value < 1.0:
            return value
    inputs = plan.get("inputs") if isinstance(plan.get("inputs"), dict) else {}
    raw = inputs.get("alpha")
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        value = float(raw)
        if 0.0 < value < 1.0:
            return value
    return DEFAULT_ALPHA


def _plan_n_seeds(plan: dict) -> int:
    inputs = plan.get("inputs") if isinstance(plan.get("inputs"), dict) else {}
    raw = inputs.get("seeds")
    if isinstance(raw, (list, tuple)):
        return len(raw)
    raw = inputs.get("n_seeds")
    if isinstance(raw, int) and not isinstance(raw, bool):
        return max(0, raw)
    return 0


def _plan_min_seeds(plan: dict, min_seeds=None) -> int:
    if isinstance(min_seeds, int) and not isinstance(min_seeds, bool):
        return max(0, min_seeds)
    return MIN_SEEDS


def _plan_looks(plan: dict) -> int:
    raw = plan.get("test_look_budget")
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return DEFAULT_TEST_LOOK_BUDGET
    return value if value >= 1 else DEFAULT_TEST_LOOK_BUDGET


def _sesoi_effects(plan: dict) -> dict:
    """从计划里推导“效应恰好等于 SESOI”时的标准化效应量（以配对差 SD 为单位）。

    换算需要 SESOI 与配对差 SD：优先用**声明的** SD（配置里显式写死的
    计划假设），没有声明时退回计划里的**参考** SD，并在 ``basis`` 里标出
    口径——参考 SD 只是展示横轴，不是数据估计出来的量，结论里必须照实写。

    返回 ``{家族键: {"effect": 倍数, "basis": "declared"/"reference",
    "sesoi": ..., "sd": ...}}``。
    """
    plan = plan if isinstance(plan, dict) else {}
    endpoints = plan.get("endpoints") if isinstance(plan.get("endpoints"), dict) else {}
    assumptions = (
        plan.get("assumptions") if isinstance(plan.get("assumptions"), dict) else {}
    )
    ppl = endpoints.get("ppl") if isinstance(endpoints.get("ppl"), dict) else {}
    ruler = (
        endpoints.get("ruler") if isinstance(endpoints.get("ruler"), dict) else {}
    )
    ruler_seed = (
        ruler.get("seed_level") if isinstance(ruler.get("seed_level"), dict) else {}
    )
    auroc = endpoints.get("auroc") if isinstance(endpoints.get("auroc"), dict) else {}
    ap = endpoints.get("ap") if isinstance(endpoints.get("ap"), dict) else {}

    def _positive(value):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return None
        fval = float(value)
        if not math.isfinite(fval) or fval <= 0:
            return None
        return fval

    def _entry(margin, declared_sd, reference_sd):
        margin = _positive(margin)
        declared = _positive(declared_sd)
        reference = _positive(reference_sd)
        if margin is None:
            return None
        if declared is not None:
            return {
                "effect": margin / declared,
                "basis": "declared",
                "sesoi": margin,
                "sd": declared,
            }
        if reference is not None:
            return {
                "effect": margin / reference,
                "basis": "reference",
                "sesoi": margin,
                "sd": reference,
            }
        return None

    return {
        "PPL": _entry(
            ppl.get("sesoi_relative"),
            assumptions.get("ppl_paired_sd_rel"),
            DEFAULT_PPL_SD_REF,
        ),
        "RULER": _entry(
            ruler_seed.get("sesoi_pp"),
            assumptions.get("ruler_paired_sd_pp"),
            DEFAULT_RULER_SD_REF,
        ),
        "AUROC": _entry(
            auroc.get("sesoi_absolute"),
            auroc.get("declared_paired_sd"),
            DEFAULT_MECHANISM_SD_REF,
        ),
        "AP": _entry(
            ap.get("sesoi_absolute"),
            ap.get("declared_paired_sd"),
            DEFAULT_MECHANISM_SD_REF,
        ),
    }


def default_scenarios(plan: dict) -> list:
    """默认校准场景：全部为零假设（校准 FWER）+ 效应恰等于 SESOI（仿真功效）。"""
    families = _plan_families(plan)
    all_ids = []
    for fam in families.values():
        for hid, _pos in fam.get("registered", []):
            if hid not in all_ids:
                all_ids.append(hid)
    scenarios = [
        {
            "id": "all_null",
            "label": "全部假设为真（零假设）：校准整条判定管线的整体假阳性率",
            "effects": {hid: 0.0 for hid in all_ids},
        }
    ]
    ratios = _sesoi_effects(plan)
    effects = {}
    missing = []
    basis = {}
    for fam_key, fam in families.items():
        entry = ratios.get(fam_key)
        for hid, _pos in fam.get("registered", []):
            if entry is None:
                missing.append(hid)
                effects[hid] = 0.0
                continue
            effects[hid] = entry["effect"]
            basis[fam_key] = entry["basis"]
    scenarios.append(
        {
            "id": "sesoi",
            "label": (
                "效应恰等于各自 SESOI（换算成配对差 SD 的倍数）：仿真功效"
            ),
            "effects": effects,
            "unavailable": missing,
            "sd_basis": basis,
        }
    )
    return scenarios


def _simulate_once(
    rng,
    families: dict,
    effects: dict,
    n_seeds: int,
    alpha: float,
    min_seeds: int,
    looks: int,
) -> dict:
    """模拟一次完整判定，返回每条登记假设的确认/假阳性标记。

    判定的每一步都与真实汇总层同构：
      * 家族内 Holm（``lega.utils.holm_bonferroni``）；
      * 全局确认族 Holm；
      * 最小种子数门；
      * look 预算阈值 ``alpha / looks``。
    """
    if n_seeds < 2 or n_seeds < min_seeds:
        # 种子数不足：整条管线不作任何判定（与真实汇总的“种子不足→未定”一致）。
        return {"tested": False, "confirmed": [], "false_positive": [], "supported": []}

    # 1) 先按家族抽零假设 p 值（连续对称零分布下双侧 t p 值精确服从 U(0,1)）。
    family_p = {}
    for key, fam in families.items():
        size = int(fam.get("size") or 0)
        if size <= 0:
            family_p[key] = np.zeros(0, dtype=np.float64)
            continue
        pvals = rng.uniform(size=size)
        # 2) 登记假设若非零假设，用备择下的配对差重新算 p 值。
        for hid, pos in fam.get("registered", []):
            effect = effects.get(hid, 0.0)
            try:
                effect = float(effect)
            except (TypeError, ValueError):
                effect = 0.0
            if not math.isfinite(effect) or effect == 0.0:
                continue
            diffs = rng.normal(loc=effect, scale=1.0, size=n_seeds)
            mean = float(diffs.mean())
            sd = float(diffs.std(ddof=1))
            if not (math.isfinite(mean) and math.isfinite(sd)) or sd <= 0.0:
                continue
            t = mean / (sd / math.sqrt(n_seeds))
            p = _student_t_p(t, n_seeds - 1)
            if p is None or not math.isfinite(p):
                continue
            pvals[pos] = p
        family_p[key] = pvals

    # 3) 家族内 Holm。
    family_adj = {}
    for key, pvals in family_p.items():
        family_adj[key] = holm_bonferroni(list(pvals))

    # 4) 全局确认族：只纳入可检验（有 p、种子数够）的登记假设，用未校正 p。
    tested = []
    for key, fam in families.items():
        for hid, pos in fam.get("registered", []):
            pvals = family_p.get(key)
            if pvals is None or pos >= len(pvals):
                continue
            p = float(pvals[pos])
            if not math.isfinite(p):
                continue
            tested.append((hid, key, pos, p))
    if not tested:
        return {"tested": False, "confirmed": [], "false_positive": [], "supported": []}
    global_adj = holm_bonferroni([item[3] for item in tested])
    look_alpha = alpha / looks if looks >= 1 else alpha

    confirmed = []
    false_positive = []
    supported = []
    for (hid, key, pos, p), padj_global in zip(tested, global_adj):
        padj_family = family_adj.get(key)
        if padj_family is None or pos >= len(padj_family):
            continue
        padj_family = padj_family[pos]
        if padj_family is None or padj_global is None:
            continue
        if padj_family >= alpha:
            continue
        if padj_global >= alpha:
            continue
        if padj_global >= look_alpha:
            continue
        confirmed.append(hid)
        if float(effects.get(hid, 0.0) or 0.0) == 0.0:
            false_positive.append(hid)
        else:
            supported.append(hid)
    return {
        "tested": True,
        "confirmed": confirmed,
        "false_positive": false_positive,
        "supported": supported,
    }


def _student_t_p(t: float, df: int):
    """双侧 t p 值；数值不可用时返回 None（不伪造 0 或 1）。"""
    try:
        from lega.utils import student_t_two_sided_p
    except Exception:  # pragma: no cover - 仅在依赖损坏时发生
        return None
    try:
        p = float(student_t_two_sided_p(t, df))
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(p):
        return None
    return min(1.0, max(0.0, p))


def analytic_power_comparison(
    plan: dict, *, n_sim: int = 20000, seed: int = DEFAULT_CALIBRATION_SEED
) -> dict:
    """并列比较三种功效口径：正态近似 / 精确（非中心 t）/ 整条管线仿真。

    计划里的解析功效用的是**正态近似**（历史口径）。在 n=3~5 这种小样本
    下它会显著高估功效：5 个种子、效应=1 个配对差 SD、α=0.003125 时，正态
    近似给 0.236，而非中心 t 的精确值只有约 0.047（约 5 倍差距）。计划如果
    只报前者，就会把“5 个种子够用”说成比实际更乐观的结论。

    这里把三者并列输出，并给出 ``normal_approx_overstates`` 标记：正态近似
    比精确值高出多少。汇总/计划页必须以精确口径为准。
    """
    from .power import paired_t_power, paired_t_power_nct

    plan = plan if isinstance(plan, dict) else {}
    mp = plan.get("multiplicity_plan") if isinstance(plan.get("multiplicity_plan"), dict) else {}
    alpha_eff = mp.get("alpha_effective") if isinstance(mp.get("alpha_effective"), dict) else {}
    n_seeds = _plan_n_seeds(plan)
    ratios = _sesoi_effects(plan)
    by_family = {}
    overstatements = []
    for fam_key, entry in ratios.items():
        if not isinstance(entry, dict):
            continue
        effect = float(entry.get("effect") or 0.0)
        alpha_raw = alpha_eff.get(fam_key)
        try:
            alpha_value = float(alpha_raw)
        except (TypeError, ValueError):
            continue
        if not (0.0 < alpha_value < 1.0):
            continue
        if n_seeds < 2 or effect <= 0.0:
            continue
        normal = paired_t_power(effect, 1.0, n_seeds, alpha_value)
        nct = paired_t_power_nct(
            effect, 1.0, n_seeds, alpha_value, n_sim=n_sim, seed=seed
        )
        exact = nct.get("power")
        over = None
        if isinstance(exact, (int, float)) and not isinstance(exact, bool):
            over = float(normal) - float(exact)
            overstatements.append(over)
        by_family[fam_key] = {
            "effect_sd_units": effect,
            "sd_basis": entry.get("basis"),
            "alpha_effective": alpha_value,
            "power_normal_approx": float(normal),
            "power_noncentral_t": (
                float(exact)
                if isinstance(exact, (int, float)) and not isinstance(exact, bool)
                else None
            ),
            "normal_minus_exact": over,
            "n_sim": nct.get("n_sim"),
            "seed": nct.get("seed"),
        }
    max_over = max(overstatements) if overstatements else None
    return {
        "by_family": by_family,
        "max_overstatement": max_over,
        # 0.10 的绝对门槛：正态近似高出 10 个百分点以上就不是“近似误差”，
        # 而是会改变“种子数够不够”的结论。
        "normal_approx_overstates": bool(max_over is not None and max_over > 0.10),
        "note": (
            "正态近似在 n=3~5 时高估功效；正式结论应以非中心 t 的精确口径"
            "（power_noncentral_t）和整条判定管线的仿真口径为准"
        ),
    }


def calibrate_design(
    plan: dict,
    *,
    n_sim: int = DEFAULT_N_SIM,
    seed: int = DEFAULT_CALIBRATION_SEED,
    scenarios=None,
    alpha=None,
    min_seeds=None,
) -> dict:
    """对设计计划里的确认性判定管线做蒙特卡洛校准。

    返回 dict，含每个场景的：模拟整体假阳性率（FWER）及其 Wilson 区间、
    逐条假设的模拟 I 类错误率/功效、至少一条被确认的概率，以及与解析规划
    阈值 ``alpha_eff`` 的对照。

    ``plan`` 用 :func:`lega.design.build_design_plan` 生成的计划即可；本函数
    不读任何实验结果，因此可以在训练前调用。
    """
    if isinstance(n_sim, bool) or not isinstance(n_sim, int) or n_sim < 1:
        raise ValueError(f"n_sim 必须是 >= 1 的整数，收到 {n_sim!r}")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(f"seed 必须是非负整数，收到 {seed!r}")
    families = _plan_families(plan)
    n_seeds = _plan_n_seeds(plan)
    alpha_eff = _plan_alpha(plan, alpha)
    min_seeds_eff = _plan_min_seeds(plan, min_seeds)
    looks = _plan_looks(plan)
    if scenarios is None:
        scenarios = default_scenarios(plan)
    if not isinstance(scenarios, (list, tuple)) or not scenarios:
        raise ValueError("scenarios 必须是非空列表")

    results = []
    for idx, scenario in enumerate(scenarios):
        if not isinstance(scenario, dict):
            raise ValueError(f"scenarios[{idx}] 必须是对象")
        raw_effects = scenario.get("effects")
        effects = {}
        if isinstance(raw_effects, dict):
            for key, value in raw_effects.items():
                try:
                    fval = float(value)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(fval):
                    effects[str(key)] = fval
        rng = np.random.default_rng(seed + idx)
        any_confirmed = 0
        false_positive_reps = 0
        supported_reps = 0
        tested_reps = 0
        per_hyp = {}
        for _rep in range(n_sim):
            out = _simulate_once(
                rng, families, effects, n_seeds, alpha_eff, min_seeds_eff, looks
            )
            if not out["tested"]:
                continue
            tested_reps += 1
            if out["confirmed"]:
                any_confirmed += 1
            if out["false_positive"]:
                false_positive_reps += 1
            if out["supported"]:
                supported_reps += 1
            for hid in out["confirmed"]:
                rec = per_hyp.setdefault(
                    hid,
                    {"confirmed": 0, "false_positive": 0, "supported": 0},
                )
                rec["confirmed"] += 1
                if hid in out["false_positive"]:
                    rec["false_positive"] += 1
                if hid in out["supported"]:
                    rec["supported"] += 1
        total = tested_reps if tested_reps else n_sim
        fwer = false_positive_reps / max(total, 1) if total else 0.0
        power_any = supported_reps / total if total else 0.0
        confirm_any = any_confirmed / total if total else 0.0
        fwer_lo, fwer_hi = wilson_interval(false_positive_reps, total)
        power_lo, power_hi = wilson_interval(supported_reps, total)
        results.append(
            {
                "id": scenario.get("id"),
                "label": scenario.get("label"),
                "effects": effects,
                "unavailable": list(scenario.get("unavailable") or []),
                "sd_basis": dict(scenario.get("sd_basis") or {}),
                "n_sim": n_sim,
                "n_tested_replicates": tested_reps,
                "fwer": fwer,
                "fwer_ci95": [fwer_lo, fwer_hi],
                "fwer_standard_error": (
                    math.sqrt(fwer * (1.0 - fwer) / max(total, 1)) if total else None
                ),
                "confirm_any": confirm_any,
                "power_any": power_any,
                "power_any_ci95": [power_lo, power_hi],
                "per_hypothesis": {
                    hid: {
                        "confirm_rate": rec["confirmed"] / max(total, 1) if total else 0.0,
                        "false_positive_rate": (
                            rec["false_positive"] / total if total else 0.0
                        ),
                        "support_rate": rec["supported"] / total if total else 0.0,
                    }
                    for hid, rec in sorted(per_hyp.items())
                },
            }
        )

    mp = plan.get("multiplicity_plan") if isinstance(plan.get("multiplicity_plan"), dict) else {}
    analytic = mp.get("alpha_effective") if isinstance(mp.get("alpha_effective"), dict) else {}
    comparison = analytic_power_comparison(plan, seed=seed)
    return {
        "method": (
            "蒙特卡洛直接模拟整条确认性判定管线（家族 Holm → 全局 Holm → "
            "最小种子数门 → alpha/planned_looks），零假设下配对 t 的双侧 p "
            "值精确服从 U(0,1)，备择下按配对差抽样后用同一 t 分布路径求 p"
        ),
        "n_sim": n_sim,
        "seed": seed,
        "alpha": alpha_eff,
        "min_seeds": min_seeds_eff,
        "n_seeds": n_seeds,
        "test_look_budget": looks,
        "families": {
            key: {
                "size": int(fam.get("size") or 0),
                "registered": [hid for hid, _pos in fam.get("registered", [])],
            }
            for key, fam in families.items()
        },
        "analytic_alpha_effective": dict(analytic),
        "analytic_comparison": comparison,
        "scenarios": results,
        "note": (
            "FWER = 至少一条真实为零假设的登记主张被判为“支持/拒绝”的模拟频率；"
            "它应不超过名义 alpha。解析口径 α_eff=alpha/max(家族,全局族) 是"
            "单条假设的保守规划阈值，不是整体假阳性率；模拟给出的是整条管线的"
            "实际水平。仿真基于“配对差近似正态”的模型，真实种子间配对差分布"
            "可能更厚尾，因此模拟结果不能替代真实数据上的敏感性分析。"
        ),
    }


def check_calibration(result: dict, alpha=None) -> dict:
    """把校准结果变成一条设计检查。

    判定口径：只有当模拟 FWER **显著**高于名义 α 时才判 fail（用蒙特卡洛
    标准误做单侧比较，避免把蒙特卡洛噪声误报成设计失败）；FWER 点估计
    高于 α 但在噪声范围内时只告警，并要求在总结页显式写出。
    """
    res = result if isinstance(result, dict) else {}
    scenarios = res.get("scenarios") if isinstance(res.get("scenarios"), list) else []
    null_scenarios = [
        s
        for s in scenarios
        if isinstance(s, dict) and s.get("id") == "all_null"
    ]
    if not null_scenarios:
        return {
            "id": "decision_calibration",
            "title": "确认性判定管线仿真校准",
            "status": "info",
            "detail": "未运行零假设场景：无法校准整条判定管线的整体假阳性率",
        }
    scenario = null_scenarios[0]
    fwer = scenario.get("fwer")
    if not isinstance(fwer, (int, float)) or isinstance(fwer, bool):
        return {
            "id": "decision_calibration",
            "title": "确认性判定管线仿真校准",
            "status": "warn",
            "detail": "零假设场景缺少可解析的 FWER：仿真未产出有效结果",
        }
    alpha_eff = (
        float(alpha)
        if isinstance(alpha, (int, float)) and not isinstance(alpha, bool) and alpha > 0
        else float(res.get("alpha") or DEFAULT_ALPHA)
    )
    se = scenario.get("fwer_standard_error")
    ci = scenario.get("fwer_ci95")
    ci_txt = ""
    if isinstance(ci, (list, tuple)) and len(ci) == 2:
        ci_txt = f"（95% CI {float(ci[0]):.4f}–{float(ci[1]):.4f}）"
    detail = (
        f"全零假设场景下，整条确认性判定管线产生至少一条假阳性的模拟频率 "
        f"FWER≈{float(fwer):.4f}{ci_txt}（名义 α={alpha_eff:.4g}，"
        f"{scenario.get('n_sim')} 次模拟）"
    )
    if isinstance(se, (int, float)) and not isinstance(se, bool):
        if fwer > alpha_eff + _Z95 * se:
            return {
                "id": "decision_calibration",
                "title": "确认性判定管线仿真校准",
                "status": "fail",
                "detail": detail
                + "：显著高于名义 α，确认性判定管线的整体假阳性率未被控制，"
                "必须先修正判定规则再跑正式实验",
            }
        if fwer > alpha_eff:
            return {
                "id": "decision_calibration",
                "title": "确认性判定管线仿真校准",
                "status": "warn",
                "detail": detail
                + "：点估计略高于名义 α（仍在蒙特卡洛噪声范围内），"
                "读结论时须按此口径保守解读",
            }
    detail += "：整体假阳性率受控（与解析规划阈值不同：α_eff 是单条假设的保守阈值）"
    sesoi = [
        s for s in scenarios if isinstance(s, dict) and s.get("id") == "sesoi"
    ]
    if sesoi:
        pw = sesoi[0].get("power_any")
        if isinstance(pw, (int, float)) and not isinstance(pw, bool):
            detail += f"；效应恰等于各自 SESOI 时，至少一条主张被确认的模拟功效≈{float(pw):.3f}"
    comparison = (
        res.get("analytic_comparison")
        if isinstance(res.get("analytic_comparison"), dict)
        else {}
    )
    if comparison.get("normal_approx_overstates"):
        max_over = comparison.get("max_overstatement")
        detail += (
            f"；⚠ 计划里的解析确认性功效用正态近似，在当前种子数下比精确"
            f"（非中心 t）口径高出约 {float(max_over):.3f}"
            if isinstance(max_over, (int, float)) and not isinstance(max_over, bool)
            else "；⚠ 计划里的解析确认性功效用正态近似，在当前种子数下明显高估"
        )
        detail += "，读“种子数够不够”时必须以精确/仿真口径为准"
    return {
        "id": "decision_calibration",
        "title": "确认性判定管线仿真校准",
        "status": "warn" if comparison.get("normal_approx_overstates") else "pass",
        "detail": detail,
    }


def main(argv=None) -> int:
    """命令行入口：``python -m lega.design.calibration --plan design_plan.json``。"""
    import argparse
    import json
    import sys

    p = argparse.ArgumentParser(
        description="对设计计划的确认性判定管线做蒙特卡洛校准"
    )
    p.add_argument("--plan", required=True, help="design_plan.json 路径")
    p.add_argument("--n_sim", type=int, default=DEFAULT_N_SIM)
    p.add_argument("--seed", type=int, default=DEFAULT_CALIBRATION_SEED)
    p.add_argument("--out", default=None, help="把校准结果写成 JSON")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    try:
        with open(args.plan, "r", encoding="utf-8") as f:
            plan = json.load(f)
    except (OSError, ValueError) as e:
        print(f"[calibration] 读取计划失败：{e}", file=sys.stderr)
        return 2
    try:
        result = calibrate_design(
            plan, n_sim=args.n_sim, seed=args.seed
        )
    except ValueError as e:
        print(f"[calibration] 校准失败：{e}", file=sys.stderr)
        return 2
    check = check_calibration(result)
    if args.out:
        try:
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2, sort_keys=True)
        except OSError as e:
            print(f"[calibration] 写入失败：{e}", file=sys.stderr)
            return 2
    if not args.quiet:
        print(f"[calibration] {check['status']}: {check['detail']}")
        for scenario in result["scenarios"]:
            print(
                f"[calibration] 场景 {scenario['id']}: FWER≈"
                f"{scenario['fwer']:.4f}、至少一条功效≈"
                f"{scenario['power_any']:.4f}"
            )
    return 2 if check.get("status") == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
