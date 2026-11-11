"""试点方差锚定与确认性可行性判定测试。

覆盖：
  * 锚点只接受 val 产物：split=test 的 PPL 被拒并留痕；
  * 试点种子与确认性种子重叠时锚点判 fail（锚点必须独立于确认性推断）；
  * SD 单侧置信上限高于点估计（保守口径），且随试点样本量增大而收紧；
  * 可行性判定只用噪声，不用试点观测到的效应方向；
  * 锚点摘要与内容绑定：改内容后摘要对不上，计划侧据此 fail；
  * 端点口径与登记一致：PPL 用相对差、RULER 用 macro EM、机制端点用变体内差。
"""

import json
import math
import os
import tempfile

from lega.design import build_design_plan
from lega.design.pilot import (
    PILOT_KIND,
    build_pilot_variance,
    load_pilot_anchor,
    pilot_anchor_digest,
    pilot_feasibility,
    pilot_paired_stats,
    pilot_sd_upper,
    registered_pilot_contrasts,
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
            "id": "H4",
            "metric": "RULER平均",
            "candidate": "lega_no_len",
            "reference": "lega",
            "direction": "candidate_worse",
        },
        {
            "id": "H6",
            "metric": "AUROC差(logit−仅查询)",
            "candidate": "lega",
            "reference": "仅查询项基线",
            "direction": "candidate_better",
        },
    ]
}


def _write(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)


def _make_results(root, seeds, *, split="val", ruler_em=0.4, auroc_gap=0.1):
    os.makedirs(root, exist_ok=True)
    for i, seed in enumerate(seeds):
        _write(
            os.path.join(root, f"eval_lega_seed{seed}.json"),
            {
                "variant": "lega",
                "seed": seed,
                "split": split,
                "ppl": 20.0 + 0.01 * i,
            },
        )
        _write(
            os.path.join(root, f"eval_standard_seed{seed}.json"),
            {
                "variant": "standard",
                "seed": seed,
                "split": split,
                "ppl": 21.0 + 0.01 * i,
            },
        )
        _write(
            os.path.join(root, f"ruler_lega_seed{seed}.json"),
            {
                "variant": "lega",
                "seed": seed,
                "results": {"niah_synth": {"256": {"em": ruler_em, "n": 10}}},
            },
        )
        _write(
            os.path.join(root, f"ruler_lega_no_len_seed{seed}.json"),
            {
                "variant": "lega_no_len",
                "seed": seed,
                "results": {"niah_synth": {"256": {"em": ruler_em - 0.05, "n": 10}}},
            },
        )
        _write(
            os.path.join(root, f"analysis_lega_seed{seed}.json"),
            {
                "variant": "lega",
                "seed": seed,
                "auroc_logit": 0.7 + auroc_gap,
                "auroc_qterm": 0.7,
                "auprc_logit": 0.3 + auroc_gap,
                "auprc_qterm": 0.3,
            },
        )
    return root


def test_pilot_anchor_rejects_test_split():
    """试点只允许 val：test 上的读数必须被拒绝并留痕。"""
    with tempfile.TemporaryDirectory() as tmp:
        results = _make_results(tmp, [900001, 900002, 900003], split="test")
        anchor = build_pilot_variance(
            results,
            registered=REGISTERED,
            pilot_seeds=[900001, 900002, 900003],
            confirmatory_seeds=[1337, 2024],
        )
        h1 = anchor["contrasts"]["H1"]
        assert h1["n_pairs"] == 0
        assert h1["n_rejected_artifacts"] >= 1
        assert anchor["status"] == "fail"


def test_pilot_anchor_rejects_seed_overlap():
    """试点种子与确认性种子重叠 → 锚点判 fail（锚点不再独立）。"""
    with tempfile.TemporaryDirectory() as tmp:
        results = _make_results(tmp, [1337, 900002, 900003])
        anchor = build_pilot_variance(
            results,
            registered=REGISTERED,
            pilot_seeds=[1337, 900002, 900003],
            confirmatory_seeds=[1337, 2024],
        )
        assert anchor["status"] == "fail"
        assert any("重叠" in p for p in anchor["problems"])


def test_pilot_anchor_uses_relative_ppl_and_macro_em():
    """PPL 用相对差、RULER 用 macro EM、机制端点用变体内差。"""
    with tempfile.TemporaryDirectory() as tmp:
        results = _make_results(tmp, [900001, 900002, 900003])
        anchor = build_pilot_variance(
            results,
            registered=REGISTERED,
            pilot_seeds=[900001, 900002, 900003],
            confirmatory_seeds=[1337, 2024],
        )
        h1 = anchor["contrasts"]["H1"]
        # 相对差 (PPL_lega − PPL_standard) / PPL_standard ≈ −4.76%
        assert abs(h1["mean_diff"] - (20.0 - 21.0) / 21.0) < 1e-4
        h4 = anchor["contrasts"]["H4"]
        assert h4["n_pairs"] == 3
        h6 = anchor["contrasts"]["H6"]
        assert h6["within_variant"] is True
        assert abs(h6["mean_diff"] - 0.1) < 1e-9


def test_pilot_sd_upper_is_conservative():
    """SD 单侧置信上限必须高于点估计，且随样本量增大而收紧。"""
    sd3 = pilot_sd_upper(0.01, 3)
    sd10 = pilot_sd_upper(0.01, 10)
    assert sd3 is not None and sd10 is not None
    assert sd3 > 0.01
    assert sd10 > 0.01
    assert sd3 > sd10
    assert pilot_sd_upper(0.0, 5) is None
    assert pilot_sd_upper(0.01, 1) is None


def test_pilot_paired_stats_shapes():
    """配对统计给出均值/SD/上界/稳健 SD/bootstrap 区间。"""
    stats = pilot_paired_stats([0.1, 0.2, 0.3], bootstrap=50, seed=7)
    assert stats["n"] == 3
    assert abs(stats["mean_diff"] - 0.2) < 1e-12
    assert stats["sd"] is not None and stats["sd"] > 0
    assert stats["sd_upper"] > stats["sd"]
    assert stats["sd_robust"] is not None
    assert stats["sd_ci95"] is not None and len(stats["sd_ci95"]) == 2


def test_pilot_feasibility_uses_noise_not_direction():
    """判定只用噪声：试点观测到的效应方向不参与功效假设。"""
    with tempfile.TemporaryDirectory() as tmp:
        results = _make_results(tmp, [900001, 900002, 900003])
        anchor = build_pilot_variance(
            results,
            registered=REGISTERED,
            pilot_seeds=[900001, 900002, 900003],
            confirmatory_seeds=[1337, 2024],
        )
        feas = pilot_feasibility(anchor, n_seeds=5, alpha=0.05, power=0.8)
        row = next(r for r in feas["rows"] if r["id"] == "H1")
        # H1 的试点均值差约 -4.8%，远大于 SESOI=1%；但判定必须按"SESOI 处的
        # 功效"给出，而不是按试点均值给功效。
        assert row["power_at_sesoi"] is not None
        assert row["verdict"] in ("proceed", "increase_seeds", "infeasible")
        assert row["required_seeds"] is not None


def test_pilot_anchor_digest_detects_tampering():
    """锚点摘要与内容绑定：改一个数字后摘要必须对不上。"""
    with tempfile.TemporaryDirectory() as tmp:
        results = _make_results(tmp, [900001, 900002, 900003])
        anchor = build_pilot_variance(
            results,
            registered=REGISTERED,
            pilot_seeds=[900001, 900002, 900003],
            confirmatory_seeds=[1337, 2024],
        )
        assert pilot_anchor_digest(anchor) == anchor["anchor_digest"]
        anchor["contrasts"]["H1"]["sd"] = 0.5
        assert pilot_anchor_digest(anchor) != anchor["anchor_digest"]


def test_plan_binds_pilot_anchor_and_flags_tampering():
    """计划绑定锚点；锚点被改写时计划记 fail。"""
    with tempfile.TemporaryDirectory() as tmp:
        results = _make_results(tmp, [900001, 900002, 900003])
        anchor = build_pilot_variance(
            results,
            registered=REGISTERED,
            pilot_seeds=[900001, 900002, 900003],
            confirmatory_seeds=[1337, 2024],
        )
        config = {"seeds": [1337, 2024, 42], "steps": 10, "design": {}}
        plan = build_design_plan(
            config,
            [1337, 2024, 42],
            ["lega", "standard"],
            REGISTERED,
            pilot_anchor=anchor,
        )
        checks = {c["id"]: c for c in plan["checks"]}
        assert "pilot_anchor" in checks
        assert plan["pilot_anchor"]["anchor_digest"] == anchor["anchor_digest"]

        tampered = json.loads(json.dumps(anchor))
        tampered["contrasts"]["H1"]["sd"] = 0.9
        bad_plan = build_design_plan(
            config,
            [1337, 2024, 42],
            ["lega", "standard"],
            REGISTERED,
            pilot_anchor=tampered,
        )
        bad_checks = {c["id"]: c for c in bad_plan["checks"]}
        assert bad_checks["pilot_anchor"]["status"] == "fail"


def test_plan_rejects_non_anchor_payload():
    """不是锚点文件的载荷必须被拒（fail），不能被静默忽略。"""
    config = {"seeds": [1337, 2024, 42], "steps": 10, "design": {}}
    plan = build_design_plan(
        config,
        [1337, 2024, 42],
        ["lega", "standard"],
        REGISTERED,
        pilot_anchor={"foo": "bar"},
    )
    checks = {c["id"]: c for c in plan["checks"]}
    assert checks["pilot_anchor"]["status"] == "fail"


def test_load_pilot_anchor_rejects_foreign_payload():
    """load_pilot_anchor 只接受 pilot_variance_anchor 产物。"""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "pilot_variance.json")
        _write(path, {"kind": "something_else"})
        assert load_pilot_anchor(path) is None
        _write(path, {"kind": PILOT_KIND, "anchor_digest": "x"})
        assert load_pilot_anchor(path)["kind"] == PILOT_KIND
        assert load_pilot_anchor(os.path.join(tmp, "missing.json")) is None


def test_registered_pilot_contrasts_only_supported_metrics():
    """只提取本模块支持的端点口径。"""
    registered = {
        "hypotheses": [
            {"id": "H1", "metric": "PPL", "candidate": "a", "reference": "b"},
            {"id": "HX", "metric": "未知端点", "candidate": "a", "reference": "b"},
        ]
    }
    contrasts = registered_pilot_contrasts(registered)
    assert [c["id"] for c in contrasts] == ["H1"]
    assert contrasts[0]["kind"] == "relative"
    assert contrasts[0]["sesoi"] == 0.01


def test_pilot_missing_results_dir_fails():
    """结果目录不存在时锚点判 fail 并说明原因。"""
    anchor = build_pilot_variance(
        os.path.join(tempfile.gettempdir(), "lega-pilot-does-not-exist"),
        registered=REGISTERED,
        pilot_seeds=[900001],
        confirmatory_seeds=[1337],
    )
    assert anchor["status"] == "fail"
    assert anchor["problems"]
    assert anchor["anchor_digest"]
