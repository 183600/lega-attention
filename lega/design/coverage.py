"""预注册覆盖度清点：哪些分析自由度被登记表的锁链覆盖，哪些没有。

研究自由度（researcher degrees of freedom）指的是“研究者可以在不影响
'这是预注册分析'这句话的前提下改变的分析选择”。预注册的价值不在于把
所有旋钮都写进登记表，而在于让**读者能看见**：哪些决定是事先写死的，
哪些是设计阶段自行拍板的。

本项目已经对登记内容做了哈希锁定（``preregistration/hypotheses.lock.json``
锁定 ``scope`` 与 ``decision_rule``），但登记表里的 ``decision_rule`` 只覆盖
了 α、最小种子数与多重比较口径；等价性/非劣性判定用的 SESOI、RULER 每格
样本量、机制端点的 τ、run 顺序、test look 预算等都**不在锁链覆盖范围内**。
它们被 ``design_plan.json`` 的 ``plan_digest`` 覆盖（改了会被发现），但
“设计摘要锁定”与“预注册锁定”不是一回事：前者防漂移，后者防事后选择。

本模块把这条边界显式列成一张表，供计划、汇总页与协议文档共同引用：

* ``covered``：被登记表锁链覆盖的自由度（α、min_seeds、多重比较口径、
  种子集合、配置、数据范围、比较口径）；
* ``uncovered``：只被设计摘要锁定、但没有进入预注册的自由度，每条都带
  一句“若要升级为确认性依据应如何处理”的说明。

它不改变任何判定，只让“确认性/探索性”的边界可核对。
"""

from __future__ import annotations

import math

from ._shared import _install as _shared_install

_shared_install(globals())


def _as_dict(value):
    return value if isinstance(value, dict) else {}


def _as_list(value):
    return value if isinstance(value, list) else []


def _fmt_value(value) -> str:
    if value is None:
        return "未声明"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if not math.isfinite(value):
            return str(value)
        return f"{value:.6g}"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(str(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{k}={v}" for k, v in value.items()) + "}"
    return str(value)


def registration_coverage(
    registered: dict,
    *,
    alpha=None,
    min_seeds=None,
    multiplicity=None,
    seeds=None,
    config_file=None,
    data_scope=None,
    sesoi_ppl=None,
    sesoi_ruler=None,
    sesoi_mechanism=None,
    ruler_n_samples=None,
    run_order=None,
    test_look_budget=None,
    analysis_tau=None,
    analysis_n_examples=None,
    declared_assumptions=None,
) -> dict:
    """清点“被预注册覆盖”与“未被覆盖”的分析自由度。

    参数都是本次设计实际使用的取值（由 :func:`lega.design.build_design_plan`
    填入）。返回值：

    ``covered``
        ``[{"knob": ..., "value": ..., "source": ...}]``——取值能被登记表的
        ``scope``/``decision_rule`` 直接约束。
    ``uncovered``
        ``[{"knob": ..., "value": ..., "note": ...}]``——只被设计摘要
        （``plan_digest``）锁定、不在预注册锁链内的自由度。
    """
    reg = _as_dict(registered)
    scope = _as_dict(reg.get("scope"))
    rule = _as_dict(reg.get("decision_rule"))
    present = bool(reg)

    covered = []
    uncovered = []

    def _cover(knob, value, source, note=None):
        if not present:
            uncovered.append(
                {
                    "knob": knob,
                    "value": _fmt_value(value),
                    "note": note or "没有登记表：无法核对是否被预注册覆盖",
                }
            )
            return
        covered.append({"knob": knob, "value": _fmt_value(value), "source": source})

    def _uncover(knob, value, note):
        uncovered.append(
            {"knob": knob, "value": _fmt_value(value), "note": note}
        )

    _cover(
        "alpha",
        rule.get("alpha", alpha),
        "preregistration/hypotheses.json#decision_rule.alpha",
    )
    _cover(
        "min_seeds",
        rule.get("min_seeds", min_seeds),
        "preregistration/hypotheses.json#decision_rule.min_seeds",
    )
    _cover(
        "multiplicity",
        rule.get("multiple_comparison", multiplicity),
        "preregistration/hypotheses.json#decision_rule.multiple_comparison",
    )
    _cover(
        "confirmatory_seeds",
        scope.get("seeds", seeds),
        "preregistration/hypotheses.json#scope.seeds",
    )
    _cover(
        "scope_config",
        scope.get("config", config_file),
        "preregistration/hypotheses.json#scope.config",
    )
    _cover(
        "scope_data",
        scope.get("data", data_scope),
        "preregistration/hypotheses.json#scope.data",
    )
    _cover(
        "scope_comparisons",
        scope.get("comparisons"),
        "preregistration/hypotheses.json#scope.comparisons",
    )

    _uncover(
        "sesoi_ppl",
        sesoi_ppl,
        "等价性/非劣性判定的实际重要差异阈值未写入 decision_rule："
        "它由 design_plan.json 的 plan_digest 锁定（改动会被发现），"
        "但不属于预注册；若要用它下确认性结论，须先经 lega.hypotheses --amend 登记",
    )
    _uncover(
        "sesoi_ruler",
        sesoi_ruler,
        "RULER 端点的 SESOI 未写入 decision_rule：同上，"
        "只被设计摘要锁定，不属于预注册",
    )
    _uncover(
        "sesoi_mechanism",
        sesoi_mechanism,
        "机制端点（AUROC/AP）的 SESOI 未写入 decision_rule：同上，"
        "只被设计摘要锁定，不属于预注册",
    )
    _uncover(
        "ruler_n_samples",
        ruler_n_samples,
        "RULER 每格评测样本数未登记：改变它会改变评测抽样精度，"
        "但不会被预注册锁链发现",
    )
    _uncover(
        "analysis_tau",
        analysis_tau,
        "机制分析的判别阈值 tau 未登记：它直接决定 AUROC/AP 口径下的正类判定",
    )
    _uncover(
        "analysis_n_examples",
        analysis_n_examples,
        "机制分析样本量未登记：它决定机制端点的估计精度",
    )
    _uncover(
        "run_order",
        run_order,
        "run 顺序（随机化区组）未登记：顺序本身会与时间/硬件漂移纠缠",
    )
    _uncover(
        "test_look_budget",
        test_look_budget,
        "test look 预算未登记：它决定 α/looks 的保守校正强度",
    )
    assumptions = _as_dict(declared_assumptions)
    declared = {k: v for k, v in assumptions.items() if v is not None}
    if declared:
        _uncover(
            "planning_assumptions",
            declared,
            "先验功效所依据的方差/不一致率假设未登记：它们是“展示横轴”"
            "还是“计划假设”必须在跑前写死，否则等于事后挑选假设",
        )

    total = len(covered) + len(uncovered)
    return {
        "registered_present": present,
        "covered": covered,
        "uncovered": uncovered,
        "n_covered": len(covered),
        "n_uncovered": len(uncovered),
        "coverage_ratio": (len(covered) / total) if total else 0.0,
        "note": (
            "covered = 取值被预注册 scope/decision_rule 直接约束（改动会被"
            "hypotheses.lock.json 发现）；uncovered = 只被 design_plan.json 的"
            "plan_digest 锁定（能发现漂移，但不能证明是事先写死的）。"
            "两者都要写进协议文档：读者据此判断每条结论的自由度边界。"
        ),
    }


def check_registration_coverage(coverage: dict) -> dict:
    """把覆盖度清点变成一条设计检查（只告警，不阻断实验）。

    未被预注册覆盖的自由度本身不是错误——它们大多由计划摘要锁定、且在
    跑前写死。检查的作用是让“哪些旋钮不在预注册里”变成一条显式记录，
    避免读者默认“所有分析选择都预注册了”。
    """
    cov = coverage if isinstance(coverage, dict) else {}
    uncovered = cov.get("uncovered") if isinstance(cov.get("uncovered"), list) else []
    n_covered = cov.get("n_covered")
    if not isinstance(n_covered, int) or isinstance(n_covered, bool):
        n_covered = 0
    if not cov.get("registered_present"):
        return {
            "id": "registration_coverage",
            "title": "预注册覆盖度（分析自由度边界）",
            "status": "warn",
            "detail": "未找到登记表：本次没有任何分析自由度能被预注册锁链约束，"
            "全部结论只能按探索性对待",
        }
    if not uncovered:
        return {
            "id": "registration_coverage",
            "title": "预注册覆盖度（分析自由度边界）",
            "status": "pass",
            "detail": f"{n_covered} 项分析自由度全部被预注册锁链覆盖",
        }
    names = [str(item.get("knob")) for item in uncovered if isinstance(item, dict)]
    detail = (
        f"{n_covered} 项分析自由度被预注册锁链覆盖，另有 {len(names)} 项只被"
        f"设计摘要（plan_digest）锁定、不在预注册范围内：{', '.join(names)}。"
        "这些旋钮改动会被计划摘要发现，但不受 hypotheses.lock.json 约束；"
        "要据此下确认性结论，须先经 lega.hypotheses --amend 追加登记"
    )
    return {
        "id": "registration_coverage",
        "title": "预注册覆盖度（分析自由度边界）",
        "status": "warn",
        "detail": detail,
    }
