"""样本效率（steps-to-target）：区分"终点更好"与"仅收敛更快"。"""

import json
import math
import os
import shutil
import tempfile

from lega.sample_efficiency import (
    analyze_sample_efficiency,
    steps_to_target,
)
from tests.pipeline_helpers import _write_json
from run_all import build_summary
# Reproducibility: Set random seeds
import random
import numpy as np
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
try:
    import torch
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
except ImportError:
    pass


def _strict_json(obj):
    return json.dumps(obj, ensure_ascii=False, allow_nan=False)


def _curve(values):
    return {int(step): float(loss) for step, loss in values}


def test_steps_to_target_first_crossing_and_transient_flag():
    curve = _curve([(100, 2.0), (200, 1.8), (300, 1.9), (400, 1.7)])
    hit = steps_to_target(curve, 1.85)
    assert hit == {"step": 200, "transient": True}  # 300 又升回 1.9 > 1.85
    hit2 = steps_to_target(curve, 1.7)
    assert hit2 == {"step": 400, "transient": False}  # 终点即目标
    # 从未到达：None，不插值外推
    assert steps_to_target(curve, 1.5) is None
    # 坏曲线/坏目标：显式 None
    assert steps_to_target({}, 1.0) is None
    assert steps_to_target({"100": float("nan")}, 1.0) is None
    assert steps_to_target(curve, float("nan")) is None
    _strict_json(hit)
    print("  [ok] steps-to-target：首次穿越 + 瞬态标记，未到达不伪造")


def test_faster_candidate_reaches_reference_endpoint_earlier():
    # 候选在 step 200/300 到达参照各自的终点水平；参照在自己的终点步才到。
    curves = {
        "cand": {
            1: _curve([(100, 2.0), (200, 1.80), (300, 1.75), (400, 1.70)]),
            2: _curve([(100, 2.0), (200, 1.95), (300, 1.80), (400, 1.72)]),
            3: _curve([(100, 1.95), (200, 1.84), (300, 1.78), (400, 1.74)]),
        },
        "ref": {
            1: _curve([(100, 2.1), (200, 2.0), (300, 1.9), (400, 1.85)]),
            2: _curve([(100, 2.1), (200, 2.0), (300, 1.9), (400, 1.86)]),
            3: _curve([(100, 2.1), (200, 2.0), (300, 1.9), (400, 1.87)]),
        },
    }
    out = analyze_sample_efficiency(curves, [("cand", "ref")])
    p = out["pairs"][0]
    assert p["available"] and p["inferential"]
    assert p["n_not_reached"] == 0 and p["n_degenerate_reference_first_step"] == 0
    assert p["frac_reached_earlier"] == 1.0
    assert p["median_ratio"] < 1.0  # 候选更早到达参照终点水平
    # log2 步数比 <0（更快），t 检验 H0：比值=1
    assert p["mean_log2_ratio"] < 0
    assert p["p_adj"] is not None
    # 目标 = 参照逐种子终点 val_loss（两侧同一目标线）
    assert p["target_rule"].startswith("reference")
    for rec in p["per_seed"]:
        assert rec["steps_reference"] == 400  # 参照终点步按构造到达
        assert rec["steps_candidate"] < rec["steps_reference"]
    _strict_json(out)
    print("  [ok] 样本效率：更快候选的中位步数比 <1，逐种子目标=参照终点")


def test_not_reached_is_counted_not_imputed():
    curves = {
        "cand": {s: _curve([(100, 2.5), (200, 2.3), (300, 2.2)]) for s in (1, 2, 3)},
        "ref": {s: _curve([(100, 2.1), (200, 2.0), (300, 1.9)]) for s in (1, 2, 3)},
    }
    p = analyze_sample_efficiency(curves, [("cand", "ref")])["pairs"][0]
    assert p["available"]
    assert p["n_not_reached"] == 3
    assert p["n_comparable"] == 0
    assert p["median_ratio"] is None and p["p"] is None
    assert "未到达参照终点" in p["verdict"]
    _strict_json(p)
    print("  [ok] 样本效率：候选未追平参照终点时计数披露，不插值不算比值")


def test_degenerate_target_and_zero_variance_are_descriptive():
    # 参照在首个评测步即达终点：目标太容易，比值不可解释。
    curves = {
        "cand": {s: _curve([(100, 1.0), (200, 0.9)]) for s in (1, 2, 3)},
        "ref": {s: _curve([(100, 1.0), (200, 1.2)]) for s in (1, 2, 3)},
    }
    p = analyze_sample_efficiency(curves, [("cand", "ref")])["pairs"][0]
    assert p["n_degenerate_reference_first_step"] == 3
    assert p["n_comparable"] == 0
    # 跨种子步数比完全相同 → 无方差，只描述不检验（不伪造 p=0）。
    curves2 = {
        "cand": {s: _curve([(100, 2.0), (200, 1.9), (300, 1.84)]) for s in (1, 2, 3)},
        "ref": {s: _curve([(100, 2.1), (200, 2.0), (300, 1.9), (400, 1.85)]) for s in (1, 2, 3)},
    }
    p2 = analyze_sample_efficiency(curves2, [("cand", "ref")])["pairs"][0]
    assert p2["inferential"] is False and p2["p"] is None
    assert p2["median_ratio"] == 0.75
    assert "无方差" in p2["verdict"]
    _strict_json(analyze_sample_efficiency({**curves, **curves2}, []))
    print("  [ok] 样本效率：退化目标与零方差按描述处理，不伪造显著性")


def test_missing_curves_are_explicit_and_pairs_deduplicated():
    out = analyze_sample_efficiency({}, [("cand", "ref"), ("cand", "ref")])
    assert out["n_pairs"] == 1  # 重复对去重
    p = out["pairs"][0]
    assert p["available"] is False and "曲线" in p["reason"]
    # 一侧缺种子：只有共同种子进入
    curves = {
        "cand": {1: _curve([(100, 2.0), (200, 1.9)]), 2: _curve([(100, 2.0), (200, 1.9)])},
        "ref": {1: _curve([(100, 2.5), (200, 2.1)])},
    }
    p2 = analyze_sample_efficiency(curves, [("cand", "ref")])["pairs"][0]
    assert p2["n_seeds"] == 1 and p2["inferential"] is False
    assert "种子不足" in p2["verdict"]
    # 非法 alpha/min_seeds：显式报错
    for kwargs in ({"alpha": 0.0}, {"alpha": 1.5}, {"min_seeds": 1}):
        try:
            analyze_sample_efficiency({}, [], **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"应拒绝非法参数 {kwargs}")
    _strict_json(out)
    print("  [ok] 样本效率：缺曲线显式 unavailable、重复对去重、参数校验")


def test_summary_persists_sample_efficiency_and_feeds_fdr():
    """样本效率必须写入 summary/significance.json 并进入探索性 FDR 家族。"""
    tmp = tempfile.mkdtemp()
    try:
        seeds = [1, 2, 3]
        _write_json(
            os.path.join(tmp, "repro.json"),
            {
                "variants": ["lega", "standard"],
                "seeds": seeds,
                "smoke": False,
            },
        )
        for v in ("lega", "standard"):
            for s in seeds:
                run_dir = os.path.join(tmp, f"{v}_seed{s}")
                os.makedirs(run_dir, exist_ok=True)
                if v == "lega":
                    # 种子 1/2 在 step 200 到达参照终点水平，种子 3 要到
                    # 终点步才到：穿越步跨种子变化，步数比有方差、可检验。
                    curve = (
                        [(100, 2.0), (200, 1.80), (400, 1.70)]
                        if s in (1, 2)
                        else [(100, 2.0), (200, 1.95), (400, 1.70)]
                    )
                else:
                    curve = [(100, 2.1), (200, 2.0), (400, 1.90 + 0.01 * (s - 1))]
                with open(
                    os.path.join(run_dir, "train_log.jsonl"),
                    "w",
                    encoding="utf-8",
                ) as f:
                    for step, loss in curve:
                        f.write(
                            json.dumps(
                                {"step": step, "val_loss": loss, "final": step == 400}
                            )
                            + "\n"
                        )
                _write_json(
                    os.path.join(tmp, f"eval_{v}_seed{s}.json"),
                    {
                        "variant": v,
                        "seed": s,
                        "split": "val",
                        "val_ppl": math.exp(1.70 if v == "lega" else 1.90),
                        "val_loss": 1.70 if v == "lega" else 1.90,
                        "n_params": 100,
                        "n_gate_extra": 4 if v == "lega" else 0,
                        "ckpt": os.path.join(run_dir, "model.pt"),
                    },
                )
        out = build_summary(tmp)
        se = out["sample_efficiency"]
        assert se is not None and se["n_available"] >= 1
        pair = next(
            p for p in se["pairs"] if p["candidate"] == "lega" and p["reference"] == "standard"
        )
        assert pair["available"] and pair["median_ratio"] < 1.0
        md = out["markdown"]
        assert "样本效率（steps-to-target" in md
        sig = json.load(open(os.path.join(tmp, "significance.json"), encoding="utf-8"))
        assert "sample_efficiency" in sig
        fdr = sig.get("exploratory_fdr", {})
        assert fdr.get("n_by_source", {}).get("sample_efficiency", 0) >= 1, (
            fdr.get("n_by_source")
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] 样本效率：进入 summary/significance.json 与探索性 FDR 家族")
