"""跨文档 vs 文档内窗口分层：交互检验与汇总集成。

覆盖四件事：
1. ``analyze_doc_strata`` 在收益集中于跨文档窗口时给出显著负交互；
2. 收益均匀时交互不显著；缺标志/缺某层/协议不一致时显式不可用；
3. 交互检验进入探索性 FDR 家族（``fdr_source="doc_strata"``）；
4. ``build_summary`` 把分层写进 Markdown、summary/variant 输出与
   ``significance.json``。
"""

import json
import os
import shutil
import tempfile

from lega.doc_strata import (
    DOC_STRATA_INTERACTION_METRIC,
    analyze_doc_strata,
)
from lega.discovery import collect_seed_level_tests, exploratory_fdr
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


def _payload(losses, cross, ctx=4, split="val", with_flags=True):
    d = {
        "split": split,
        "eval_ctx": ctx,
        "window_losses": [float(x) for x in losses],
        "window_indices": list(range(len(losses))),
    }
    if with_flags:
        d["window_crossing_documents"] = [bool(c) for c in cross]
    return d


CROSS = [True, True, False, False, False]


def _runs_gap():
    """候选在跨文档窗口好很多、文档内窗口几乎相同。"""
    runs = {"cand": {}, "ref": {}}
    for s, off in [(1, 0.0), (2, 0.02), (3, -0.01)]:
        base = [1.0 + 0.1 * i for i in range(len(CROSS))]
        runs["ref"][s] = {"eval": _payload([b + off for b in base], CROSS)}
        cand = [
            b + off + (-0.30 if c else -0.02)
            for b, c in zip(base, CROSS)
        ]
        runs["cand"][s] = {"eval": _payload(cand, CROSS)}
    return runs


def test_interaction_detects_benefit_concentrated_in_cross_document():
    out = analyze_doc_strata(_runs_gap(), [("cand", "ref")])
    assert out["available"] and out["n_available"] == 1
    pair = out["pairs"][0]
    assert pair["available"]
    # 跨文档 ΔNLL 明显更负（候选更好）、文档内近似为 0，交互为负。
    assert pair["cross"]["mean_diff"] < pair["aligned"]["mean_diff"]
    assert pair["interaction"]["mean_diff"] < 0
    assert pair["interaction"]["p_adj"] is not None
    assert "跨文档窗口" in pair["verdict"]
    # 严格 JSON：不能出现 NaN/Infinity 字段。
    json.dumps(out, ensure_ascii=False, allow_nan=False)
    print("  [ok] 文档分层：收益集中在跨文档窗口 → 显著负交互")


def test_interaction_null_when_benefit_uniform():
    runs = {"cand": {}, "ref": {}}
    for s in (1, 2, 3):
        # 用可精确表示的整数差值，保证跨文档/文档内两层差值完全相同，
        # 交互精确为 0（否则浮点残差会造成零方差的伪显著）。
        base = [1.0, 2.0, 3.0, 4.0, 5.0]
        runs["ref"][s] = {"eval": _payload(base, CROSS)}
        runs["cand"][s] = {"eval": _payload([b - 0.5 for b in base], CROSS)}
    pair = analyze_doc_strata(runs, [("cand", "ref")])["pairs"][0]
    assert pair["available"]
    # 均匀改善时交互=0；三种子零方差 → p_adj=1（不能伪造成显著）。
    assert pair["interaction"]["mean_diff"] == 0.0
    assert pair["p_adj"] == 1.0
    print("  [ok] 文档分层：均匀改善 → 交互不显著")


def test_missing_flags_and_missing_stratum_are_unavailable():
    runs = _runs_gap()
    # 去掉逐窗口标志 → 不可用，且理由可读。
    for s in runs["cand"]:
        runs["cand"][s]["eval"].pop("window_crossing_documents")
    pair = analyze_doc_strata(runs, [("cand", "ref")])["pairs"][0]
    assert pair["available"] is False and pair["reason"]
    # 只有跨文档窗口（没有文档内窗口）→ 无法做分层，显式不可用。
    runs2 = {"cand": {}, "ref": {}}
    only_cross = [True, True, True, True]
    for s in (1, 2, 3):
        runs2["ref"][s] = {"eval": _payload([1.0, 1.1, 1.2, 1.3], only_cross)}
        runs2["cand"][s] = {"eval": _payload([0.7, 0.8, 0.9, 1.0], only_cross)}
    pair2 = analyze_doc_strata(runs2, [("cand", "ref")])["pairs"][0]
    assert pair2["available"] is False
    print("  [ok] 文档分层：缺标志/缺一层 → 显式不可用，不伪造")


def test_protocol_and_flag_mismatch_skipped():
    runs = _runs_gap()
    # 两侧 eval_ctx 不一致 → 不是同一窗口网格，整对拒绝。
    for s in runs["cand"]:
        runs["cand"][s]["eval"]["eval_ctx"] = 8
    assert analyze_doc_strata(runs, [("cand", "ref")])["pairs"][0]["available"] is False
    # 同一窗口 key 的跨文档标志不一致 → 窗口定义漂移，拒绝。
    runs2 = _runs_gap()
    for s in runs2["cand"]:
        flags = list(runs2["cand"][s]["eval"]["window_crossing_documents"])
        flags[0] = not flags[0]
        runs2["cand"][s]["eval"]["window_crossing_documents"] = flags
    assert analyze_doc_strata(runs2, [("cand", "ref")])["pairs"][0]["available"] is False
    print("  [ok] 文档分层：协议/标志不一致 → 整对拒绝")


def test_interaction_feeds_exploratory_fdr_family():
    out = analyze_doc_strata(_runs_gap(), [("cand", "ref")])
    tests = collect_seed_level_tests(doc_strata=out)
    assert len(tests) == 1
    assert tests[0]["fdr_source"] == "doc_strata"
    assert tests[0]["metric"] == DOC_STRATA_INTERACTION_METRIC
    fdr = exploratory_fdr(tests)
    assert "doc_strata" in fdr["n_by_source"]
    assert fdr["family_size"] == 1
    print("  [ok] 文档分层：交互检验进入探索性 FDR 家族")


def test_build_summary_renders_doc_strata():
    tmp = tempfile.mkdtemp()
    try:
        _write_json(
            os.path.join(tmp, "repro.json"),
            {"variants": ["standard", "lega"], "seeds": [1, 2, 3], "smoke": False},
        )
        seeds = (1, 2, 3)
        for seed in seeds:
            base = [1.0 + 0.1 * i for i in range(len(CROSS))]
            _write_json(
                os.path.join(tmp, f"eval_standard_seed{seed}.json"),
                {
                    "variant": "standard",
                    "seed": seed,
                    "val_ppl": 10.0,
                    "val_loss": 2.3,
                    "n_params": 10,
                    "split": "val",
                    **_payload(base, CROSS),
                },
            )
            cand = [b + (-0.30 if c else -0.02) for b, c in zip(base, CROSS)]
            _write_json(
                os.path.join(tmp, f"eval_lega_seed{seed}.json"),
                {
                    "variant": "lega",
                    "seed": seed,
                    "val_ppl": 8.0,
                    "val_loss": 2.0,
                    "n_params": 10,
                    "split": "val",
                    **_payload(cand, CROSS),
                },
            )
        out = build_summary(tmp)
        assert "跨文档 vs 文档内窗口分层" in out["markdown"]
        assert out["doc_strata"]["available"]
        pair = next(
            p
            for p in out["doc_strata"]["pairs"]
            if p.get("candidate") == "lega" and p.get("reference") == "standard"
        )
        assert pair["available"]
        sig = json.load(
            open(os.path.join(tmp, "significance.json"), encoding="utf-8")
        )
        assert "doc_strata" in sig and "doc_strata_note" in sig
        assert sig["doc_strata"]["n_available"] >= 1
        # 分层交互必须进入探索性 FDR 来源计数。
        assert "doc_strata" in sig["exploratory_fdr"]["n_by_source"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] build_summary 集成：分层进入 Markdown/JSON/FDR")
