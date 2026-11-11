"""算力预算预检与取舍阶梯测试。

覆盖：
  * 无实测单步耗时时不编造估算（标注"不可用"）；
  * 实测单步耗时可从 eval_*.json 的 train_step_time_ms 取到；
  * 超预算时给出机械的、结果无关的削减阶梯，且**不删确认性对照**；
  * 阶梯优先去掉非确认性可选臂，再按计划顺序减少种子（下限 3）；
  * 预算策略 warn/fail 的状态区分；
  * 预算计划进入设计计划并生成对应检查条目。
"""

import json
import os
import tempfile

from lega.design import build_design_plan
from lega.design.budget import (
    BUDGET_KIND,
    build_budget_plan,
    budget_check,
    confirmatory_variants,
    estimate_run_hours,
    measured_step_time_ms,
)

REGISTERED = {
    "hypotheses": [
        {
            "id": "H1",
            "metric": "PPL",
            "candidate": "lega",
            "reference": "standard",
            "direction": "candidate_better",
        },
        {
            "id": "H2",
            "metric": "PPL",
            "candidate": "lega_no_query",
            "reference": "lega",
            "direction": "candidate_worse",
        },
    ]
}


def test_estimate_run_hours_requires_step_time():
    """缺少单步耗时时不能编造估算。"""
    out = estimate_run_hours({"steps": 8000}, step_time_ms=None)
    assert out["available"] is False
    assert out["hours_per_run"] is None
    assert out["reason"]

    out2 = estimate_run_hours({"steps": 8000}, step_time_ms=100.0)
    assert out2["available"] is True
    assert out2["hours_per_run"] > 0
    assert out2["hours_per_run_upper"] > out2["hours_per_run"]


def test_measured_step_time_reads_artifacts():
    """实测单步耗时取自 eval_*.json 的 train_step_time_ms（中位数）。"""
    with tempfile.TemporaryDirectory() as tmp:
        for i, step_ms in enumerate([100.0, 200.0, 300.0]):
            with open(
                os.path.join(tmp, f"eval_lega_seed{i}.json"), "w", encoding="utf-8"
            ) as f:
                json.dump(
                    {"variant": "lega", "seed": i, "train_step_time_ms": step_ms}, f
                )
        assert measured_step_time_ms(tmp) == 200.0
    assert measured_step_time_ms("/nonexistent/path") is None


def test_confirmatory_variants_uses_registry():
    """确认性臂 = 预注册对照涉及的变体。"""
    got = confirmatory_variants(REGISTERED, ["lega", "standard", "lega_no_query", "x"])
    assert got == {"lega", "standard", "lega_no_query"}


def test_budget_ladder_never_drops_confirmatory_arms():
    """削减阶梯不得删除确认性对照，只能去掉可选臂并减少种子。"""
    plan = build_budget_plan(
        {"steps": 8000},
        ["lega", "standard", "lega_no_query", "lega_placebo"],
        [1337, 2024, 42, 7, 2025],
        registered=REGISTERED,
        budget_gpu_hours=1.0,
        step_time_ms=100.0,
        estimate_source="declared",
    )
    assert plan["status"] == "warn"
    assert plan["confirmatory_variants"]
    dropped_variants = set()
    dropped_seeds = set()
    for rung in plan["ladder"]:
        dropped_variants.update(rung["dropped_variants"])
        dropped_seeds.update(rung["dropped_seeds"])
    assert "lega" not in dropped_variants
    assert "standard" not in dropped_variants
    assert "lega_no_query" not in dropped_variants
    assert "lega_placebo" in dropped_variants
    # 种子削减是按计划顺序保留前缀
    assert dropped_seeds <= {7, 2025}
    assert all(rung["kept_cells"] > 0 for rung in plan["ladder"])


def test_budget_ladder_stops_at_min_seeds():
    """种子削减不得低于显著性判定下限 3。"""
    plan = build_budget_plan(
        {"steps": 8000},
        ["lega", "standard"],
        [1, 2, 3, 4, 5, 6],
        registered=REGISTERED,
        budget_gpu_hours=0.001,
        step_time_ms=100.0,
    )
    last = plan["ladder"][-1]
    assert last["kept_cells"] >= 2 * 3
    assert len(last["dropped_seeds"]) == 3


def test_budget_recommendation_picks_least_destructive_rung():
    """推荐的是**能装进预算的最小破坏**方案。"""
    plan = build_budget_plan(
        {"steps": 8000},
        ["lega", "standard", "lega_placebo"],
        [1337, 2024, 42],
        registered=REGISTERED,
        budget_gpu_hours=10.0,
        step_time_ms=10.0,
    )
    assert plan["status"] == "pass"
    assert plan["recommendation"]["action"] == "keep_full_plan"


def test_budget_policy_fail_marks_design_failure():
    """预算策略 fail：超预算即设计失败。"""
    plan = build_budget_plan(
        {"steps": 8000},
        ["lega", "standard"],
        [1337, 2024, 42],
        registered=REGISTERED,
        budget_gpu_hours=0.0001,
        step_time_ms=500.0,
        policy="fail",
    )
    assert plan["status"] == "fail"
    check = budget_check(plan)
    assert check["status"] == "fail"
    assert check["id"] == "compute_budget"


def test_budget_check_without_budget_is_info():
    """没声明预算时只提示，不假装做过判定。"""
    plan = build_budget_plan(
        {"steps": 8000}, ["lega"], [1337], registered=REGISTERED, step_time_ms=10.0
    )
    check = budget_check(plan)
    assert check["status"] == "info"


def test_budget_plan_enters_design_plan():
    """预算计划进入设计计划，并生成对应检查条目。"""
    budget = build_budget_plan(
        {"steps": 8000},
        ["lega", "standard"],
        [1337, 2024, 42],
        registered=REGISTERED,
        budget_gpu_hours=100.0,
        step_time_ms=10.0,
        estimate_source="declared",
    )
    assert budget["kind"] == BUDGET_KIND
    plan = build_design_plan(
        {"seeds": [1337, 2024, 42], "steps": 8000, "design": {}},
        [1337, 2024, 42],
        ["lega", "standard"],
        REGISTERED,
        budget_plan=budget,
    )
    checks = {c["id"]: c for c in plan["checks"]}
    assert "compute_budget" in checks
    assert plan["compute_budget"]["budget_gpu_hours"] == 100.0
