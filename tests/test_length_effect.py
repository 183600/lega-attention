"""上下文长度分层与剂量-反应分析的测试。

覆盖三类风险：
  * RULER payload 的损坏字段不能伪造出一个“长度趋势”（越界 EM、非法 n、
    非规范 ctx 键、重复 ctx 键）；
  * 斜率/首末落差只对合法、互异的上下文长度定义，且能还原已知线性关系；
  * 汇总层必须把“总体平均与最长上下文方向相反”“各档任务不等权”显式
    暴露，而不是只用跨长度平均掩盖。
"""
import json
import math
import os
import shutil
import tempfile

from lega.lengtheffect import (context_macro_means, context_task_means,
                               log2_slope, span_contrast)
from run_all import _ruler_avg_of, build_summary


def _payload(results):
    return {"results": results}


def _write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)


def test_context_task_means_skips_invalid_cells():
    """非法 n/越界比例/非规范 ctx 键按缺失或损坏处理，不伪造观测。"""
    payload = _payload({
        "niah_synth": {
            "512": {"em": 0.5, "n": 20},
            "1024": {"em": 1.5, "n": 20},        # EM 越界 → 跳过
            "2048": {"em": 0.25, "n": 0},        # n=0 → 跳过
        },
        "vt": {
            "512": {"em": 0.25, "n": 10},
            "1024": {"em": 0.75},                # 旧产物缺 n：保留
        },
        "_skipped": ["bad"],
        "_badcases": [{"x": 1}],
    })
    out = context_task_means(payload)
    assert set(out) == {512, 1024}, out
    assert out[512] == {"niah_synth": 0.5, "vt": 0.25}
    assert out[1024] == {"vt": 0.75}
    # 任务等权：512 档为 (0.5+0.25)/2，而不是按 n 加权。
    macro = context_macro_means(payload)
    assert abs(macro[512] - 0.375) < 1e-12
    assert abs(macro[1024] - 0.75) < 1e-12


def test_context_task_means_rejects_non_canonical_and_duplicate_keys():
    """'0512' 与 '512' 会合并两个原始键：无法判断是否同档位，整体拒绝。"""
    bad = _payload({"niah_synth": {
        "512": {"em": 0.5, "n": 10},
        "0512": {"em": 0.6, "n": 10},
    }})
    assert context_task_means(bad) == {}
    bad2 = _payload({"niah_synth": {"512.0": {"em": 0.5, "n": 10}}})
    assert context_task_means(bad2) == {}
    bad3 = _payload({"niah_synth": {" 512": {"em": 0.5, "n": 10}}})
    assert context_task_means(bad3) == {}
    # 不同任务在同一 ctx 上的同名键是正常设计，不能误判为重复。
    ok = _payload({
        "niah_synth": {"512": {"em": 0.5, "n": 10}},
        "vt": {"512": {"em": 0.1, "n": 10}},
    })
    assert set(context_task_means(ok)) == {512}


def test_context_task_means_malformed_returns_empty():
    for payload in (None, [], {}, {"results": []},
                    {"results": {"_skipped": ["x"]}}):
        assert context_task_means(payload) == {}
        assert context_macro_means(payload) == {}
    # 全零与“没有观测”必须可区分：前者是有效的 0 分，后者是空。
    zero = _payload({"vt": {"512": {"em": 0.0, "n": 10}}})
    assert context_macro_means(zero) == {512: 0.0}


def test_context_task_means_rejects_numpy_bools_like_run_all():
    """np.bool_ 在 NumPy 2.x 的类型名是 "bool"：必须按 bool 拒绝。"""
    import numpy as np
    from lega.lengtheffect import _finite, _positive_count
    assert _finite(np.True_) is None
    assert _finite(np.False_) is None
    assert _positive_count(np.True_) is None
    assert _positive_count(np.False_) is None
    payload = _payload({"a": {
        "512": {"em": np.True_, "n": 10},
        "1024": {"em": 0.5, "n": 10},
    }})
    # em=True 不能成为 1.0：否则长度分层会多出一格总体 RULER 平均没有的观测。
    assert context_task_means(payload) == {1024: {"a": 0.5}}
    bad_n = _payload({"a": {
        "512": {"em": 1.0, "n": np.True_},
        "1024": {"em": 0.5, "n": 10},
    }})
    assert context_task_means(bad_n) == {1024: {"a": 0.5}}
    assert abs(context_macro_means(bad_n)[1024] - _ruler_avg_of(bad_n)) < 1e-12


def test_context_task_means_accepts_exact_huge_and_numeric_string_counts():
    """合法大整数 n 不能因 float 溢出被丢掉；数值字符串与 run_all 同口径。"""
    huge_n = 10 ** 400
    payload = _payload({
        "niah_synth": {
            "512": {"em": 0.5, "n": huge_n},
            "1024": {"em": 0.75, "n": "20"},
            "2048": {"em": 0.25, "n": 4.0},
        },
        "vt": {
            "512": {"em": 0.1, "n": "2e1"},
            "1024": {"em": 0.25, "n": 5},
            "2048": {"em": 0.75, "n": 5},
        },
    })
    out = context_task_means(payload)
    assert out == {
        512: {"niah_synth": 0.5, "vt": 0.1},
        1024: {"niah_synth": 0.75, "vt": 0.25},
        2048: {"niah_synth": 0.25, "vt": 0.75},
    }
    # 平衡网格下必须与既有 RULER macro 平均严格一致（含大整数/字符串 n）。
    macro = context_macro_means(payload)
    assert abs(sum(macro.values()) / len(macro) - _ruler_avg_of(payload)) < 1e-12
    # 非规范/非整数/非正的字符串计数仍是损坏元数据，按缺失跳过。
    bad = _payload({"vt": {
        "512": {"em": 0.2, "n": "1_0"},
        "1024": {"em": 0.4, "n": "2.5"},
        "2048": {"em": 0.6, "n": "0"},
    }})
    assert context_task_means(bad) == {}


def test_context_task_means_accepts_numeric_string_metrics_like_run_all():
    """数值字符串 EM 不能被长度分层单独丢掉，否则与 RULER macro 不可比。"""
    payload = _payload({
        "a": {"256": {"em": "0.5", "n": 2},
              "512": {"em": "0.25", "n": 2}},
        "b": {"256": {"em": 0.1, "n": "2"},
              "512": {"em": "0.2", "n": 2}},
    })
    out = context_task_means(payload)
    assert out[256] == {"a": 0.5, "b": 0.1}
    assert out[512] == {"a": 0.25, "b": 0.2}
    macro = context_macro_means(payload)
    assert abs(sum(macro.values()) / len(macro) - _ruler_avg_of(payload)) < 1e-12
    # 非数值/下划线数字串仍是缺失，不能伪造成观测。
    bad = _payload({"a": {
        "256": {"em": "bad", "n": 2},
        "512": {"em": "1_0", "n": 2},
    }})
    assert context_task_means(bad) == {}


def test_log2_slope_recovers_known_log_linear_relation():
    ctxs = [512, 1024, 2048, 4096]
    values = [0.30, 0.20, 0.10, 0.00]   # 每翻倍 -0.10
    assert abs(log2_slope(ctxs, values) + 0.10) < 1e-12
    # 输入顺序无关（内部按 ctx 排序）。
    assert abs(log2_slope(list(reversed(ctxs)),
                          list(reversed(values))) + 0.10) < 1e-12
    assert abs(span_contrast(ctxs, values) + 0.30) < 1e-12


def test_log2_slope_and_span_reject_undefined_inputs():
    assert log2_slope([512], [0.1]) is None
    assert span_contrast([512], [0.1]) is None
    # 重复 ctx / 非规范 ctx / 非有限 value / 长度不等均不可定义。
    assert log2_slope([512, 512], [0.1, 0.2]) is None
    assert log2_slope(["0512", "1024"], [0.1, 0.2]) is None
    assert log2_slope([512, 1024], [0.1, float("nan")]) is None
    # 趋势函数只接受已算好的真正数值；数值字符串属于 payload 解析层，
    # 不能绕过类型检查进入 OLS。
    assert log2_slope([512, 1024], ["0.1", "0.2"]) is None
    assert span_contrast([512, 1024], ["0.1", "0.2"]) is None
    assert log2_slope([512, 1024], [0.1]) is None
    assert span_contrast([512, 0], [0.1, 0.2]) is None


def _write_runs(tmp, values_by_variant, ctxs, seeds=(1, 2, 3, 4),
                tasks=("niah_synth", "vt")):
    """values_by_variant[variant][ctx][task] = EM；写出 eval/ruler 产物。"""
    for s in seeds:
        for v, per_ctx in values_by_variant.items():
            _write_json(os.path.join(tmp, f"eval_{v}_seed{s}.json"), {
                "variant": v, "seed": s, "val_ppl": 10.0, "val_loss": 2.3,
                "n_params": 100, "n_gate_extra": 0, "split": "val"})
            results = {
                t: {str(c): {"em": per_ctx[c][t], "n": 20} for c in ctxs}
                for t in tasks
            }
            _write_json(os.path.join(tmp, f"ruler_{v}_seed{s}.json"), {
                "variant": v, "seed": s, "ctx_lens": list(ctxs),
                "n_samples": 20, "sample_seed": 1234,
                "corpus_source": "val", "max_new": 12, "fwe_max_new": 44,
                "vt_chains": 4, "vt_depth": 4, "results": results})


def test_build_summary_length_reversal_is_visible():
    """短上下文涨、长上下文跌：总体平均看似持平，必须显式标方向翻转。"""
    tmp = tempfile.mkdtemp()
    try:
        ctxs = [512, 1024, 2048]
        # 两个任务同向；macro 每档相同，便于手算。
        by_ctx = {
            "standard": {512: 0.60, 1024: 0.55, 2048: 0.50},
            "lega": {512: 0.75, 1024: 0.55, 2048: 0.40},
        }
        _write_runs(tmp, {v: {c: {t: val for t in ("niah_synth", "vt")}
                              for c, val in per_ctx.items()}
                          for v, per_ctx in by_ctx.items()}, ctxs)
        out = build_summary(tmp)
        md = out["markdown"]
        assert "RULER 上下文长度剂量-反应" in md, md[-3000:]
        assert "总平均与最长ctx反向" in md
        assert "优势随上下文长度显著减小" in md

        sig = json.load(open(os.path.join(tmp, "significance.json"),
                             encoding="utf-8"))
        assert "ruler_length_pairs" in sig
        assert "ruler_length_tests" in sig
        pair = next(p for p in sig["ruler_length_pairs"]
                    if p["candidate"] == "lega"
                    and p["reference"] == "standard")
        assert pair["contexts"] == ctxs
        assert pair["n_seeds"] == 4
        assert pair["balanced_contexts"] is True
        assert pair["overall_vs_longest_sign_flip"] is True
        assert pair["overall_vs_slope_sign_flip"] is True
        # 逐档 Δ = +15 / 0 / −10 pp。
        diffs = [c["mean_diff"] for c in pair["per_context"]]
        assert abs(diffs[0] - 0.15) < 1e-12
        assert abs(diffs[1]) < 1e-12
        assert abs(diffs[2] + 0.10) < 1e-12
        assert abs(pair["slope_per_doubling"]["mean_diff"]
                   + 0.125) < 1e-9
        assert abs(pair["long_minus_short"]["mean_diff"] + 0.25) < 1e-12
        assert abs(pair["per_seed"][0]["diffs"]["2048"] + 0.10) < 1e-12
        row = next(r for r in out["rows"] if r["variant"] == "lega")
        assert row["ruler_length_contexts"] == ctxs
        assert abs(row["ruler_length_slope_pp_per_doubling"] + 12.5) < 1e-9
        assert abs(row["ruler_length_longest_delta_pp"] + 10.0) < 1e-9
        assert row["ruler_length_sign_flip"] is True
        assert row["ruler_length_masking"] is False
        assert pair["slope_loo_sign_stable"] is True
        assert pair["slope_per_doubling"]["p_adj"] is not None
        ctx_tests = sig["ruler_length_tests"]["per_context"]
        # 逐对 JSON 必须回填校正后的置换 p，不能只让 tests 数组里有值。
        assert all(c.get("p_perm_adj") is not None
                   for c in pair["per_context"])
        assert (pair["slope_per_doubling"]["p_perm_adj"]
                == next(t["p_perm_adj"] for t in
                        sig["ruler_length_tests"]["slope"]))
        assert (pair["long_minus_short"]["p_perm_adj"]
                == next(t["p_perm_adj"] for t in
                        sig["ruler_length_tests"]["long_minus_short"]))
        assert {t["ruler_ctx"] for t in ctx_tests} == set(ctxs)
        assert all(t.get("exploratory") for t in ctx_tests)
        assert all(t.get("p_adj") is not None for t in ctx_tests)
        # 新检验只进独立的小节/JSON，不污染主配对检验表。
        assert not any(str(t.get("metric", "")).startswith("RULER@")
                       for t in out["tests"])
        main_table = md.split("## 关键对比")[0]
        assert "RULER@ctx" not in main_table
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_build_summary_length_zero_overall_masks_long_context_harm():
    """总体恰好持平、最长档明显更差：零不是“方向相同”，必须标平均掩盖。"""
    tmp = tempfile.mkdtemp()
    try:
        ctxs = [512, 1024, 2048]
        by_ctx = {
            "standard": {512: 0.60, 1024: 0.55, 2048: 0.50},
            "lega": {512: 0.70, 1024: 0.55, 2048: 0.40},
        }
        _write_runs(tmp, {v: {c: {t: val for t in ("niah_synth", "vt")}
                              for c, val in per_ctx.items()}
                          for v, per_ctx in by_ctx.items()}, ctxs)
        out = build_summary(tmp)
        sig = json.load(open(os.path.join(tmp, "significance.json"),
                             encoding="utf-8"))
        pair = next(p for p in sig["ruler_length_pairs"]
                    if p["candidate"] == "lega")
        assert abs(pair["overall_mean_diff"]) < 1e-12
        assert abs(pair["per_context"][-1]["mean_diff"] + 0.10) < 1e-12
        assert pair["overall_vs_longest_sign_flip"] is False
        assert pair["long_context_masking"] is True
        assert "平均掩盖长度效应" in out["markdown"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_build_summary_length_trend_depends_on_single_context_is_flagged():
    """斜率完全由一档驱动时，删掉该档就翻转；必须标趋势不稳健。"""
    tmp = tempfile.mkdtemp()
    try:
        ctxs = [512, 1024, 2048]
        # Δ = +30 / −20 / −10 pp：总体为 0，全斜率 −20 pp/2×；
        # 但删掉 512 档后斜率变为 +10 pp/2×（符号翻转）。
        by_ctx = {
            "standard": {512: 0.5, 1024: 0.5, 2048: 0.5},
            "lega": {512: 0.8, 1024: 0.3, 2048: 0.4},
        }
        _write_runs(tmp, {v: {c: {t: val for t in ("niah_synth", "vt")}
                              for c, val in per_ctx.items()}
                          for v, per_ctx in by_ctx.items()}, ctxs)
        out = build_summary(tmp)
        sig = json.load(open(os.path.join(tmp, "significance.json"),
                             encoding="utf-8"))
        pair = next(p for p in sig["ruler_length_pairs"]
                    if p["candidate"] == "lega")
        assert pair["slope_loo_sign_stable"] is False, pair
        loo = pair["slope_leave_one_ctx_out"]
        assert loo["512"] > 0 > loo["1024"]
        assert "删掉任一单个ctx后斜率符号翻转" in out["markdown"]
        row = next(r for r in out["rows"] if r["variant"] == "lega")
        assert row["ruler_length_loo_sign_stable"] is False
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_build_summary_length_single_context_is_explicitly_unavailable():
    """只有一个公共 ctx 时必须写明“无法检验长度趋势”，不能静默少一张表。"""
    tmp = tempfile.mkdtemp()
    try:
        ctxs = [1024]
        _write_runs(tmp, {
            "standard": {1024: {"niah_synth": 0.4, "vt": 0.4}},
            "lega": {1024: {"niah_synth": 0.6, "vt": 0.6}},
        }, ctxs)
        out = build_summary(tmp)
        md = out["markdown"]
        assert "只有 1 个公共请求档位" in md
        assert "无法检验长度趋势" in md
        sig = json.load(open(os.path.join(tmp, "significance.json"),
                             encoding="utf-8"))
        pair = next(p for p in sig["ruler_length_pairs"]
                    if p["candidate"] == "lega")
        assert pair["n_contexts"] == 1
        assert pair["slope_per_doubling"] is None
        assert pair["long_minus_short"] is None
        assert pair["per_context"][0]["ctx"] == 1024
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_build_summary_length_unalignable_grid_is_not_fabricated():
    """任务缺档时无法把实际 room_ctx 对到请求档位：拒绝分析，不猜映射。"""
    tmp = tempfile.mkdtemp()
    try:
        ctxs = [512, 1024, 2048]
        # niah_synth 缺 1024 档（只有 2 个实际档位），vt 有完整 3 档。
        # 若按实际键顺序硬配，会把“缺格”伪装成一条长度趋势。
        standard = {
            512: {"niah_synth": 0.9, "vt": 0.0},
            1024: {"vt": 0.0},
            2048: {"niah_synth": 0.9, "vt": 0.0},
        }
        lega = {
            512: {"niah_synth": 0.9, "vt": 0.1},
            1024: {"vt": 0.1},
            2048: {"niah_synth": 0.9, "vt": 0.1},
        }
        for s in (1, 2, 3, 4):
            for v, per_ctx in (("standard", standard), ("lega", lega)):
                _write_json(os.path.join(tmp, f"eval_{v}_seed{s}.json"), {
                    "variant": v, "seed": s, "val_ppl": 10.0,
                    "val_loss": 2.3, "n_params": 100, "n_gate_extra": 0,
                    "split": "val"})
                results = {t: {} for t in ("niah_synth", "vt")}
                for c, task_vals in per_ctx.items():
                    for task, em in task_vals.items():
                        results[task][str(c)] = {"em": em, "n": 20}
                _write_json(os.path.join(tmp, f"ruler_{v}_seed{s}.json"), {
                    "variant": v, "seed": s, "ctx_lens": ctxs,
                    "n_samples": 20, "sample_seed": 1,
                    "corpus_source": "val", "max_new": 12,
                    "fwe_max_new": 44, "vt_chains": 4, "vt_depth": 4,
                    "results": results})
        out = build_summary(tmp)
        sig = json.load(open(os.path.join(tmp, "significance.json"),
                             encoding="utf-8"))
        assert sig["ruler_length_pairs"] == []
        assert "无法做上下文长度分层" in out["markdown"]
        row = next(r for r in out["rows"] if r["variant"] == "lega")
        assert "ruler_length_slope_pp_per_doubling" not in row
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_context_macro_means_aligns_requested_context_lengths():
    """不同任务的 room_ctx 因生成预算而不同：按请求档位对齐而非当新档位。"""
    payload = {
        "ctx_lens": [256, 512, 1024],
        "results": {
            # niah 的 1024 请求档被 clamp 到 1011，fwe 被 clamp 到 979。
            "niah_synth": {
                "256": {"em": 0.6, "n": 20},
                "512": {"em": 0.5, "n": 20},
                "1011": {"em": 0.4, "n": 20},
            },
            "fwe": {
                "256": {"em": 0.2, "n": 20},
                "512": {"em": 0.2, "n": 20},
                "979": {"em": 0.2, "n": 20},
            },
        },
    }
    aligned = context_macro_means(payload, align_requested=True)
    assert list(aligned) == [256, 512, 1024], aligned
    assert abs(aligned[1024] - 0.3) < 1e-12
    # 默认按实际 room_ctx：1011 与 979 会成为两个“档位”，任务组成不同。
    actual = context_macro_means(payload)
    assert set(actual) == {256, 512, 979, 1011}
    assert abs(actual[1011] - 0.4) < 1e-12
    assert abs(actual[979] - 0.2) < 1e-12
    # 任务缺一个请求档位（2 个实际键 vs 3 个请求档）→ 拒绝对齐。
    broken = {
        "ctx_lens": [256, 512, 1024],
        "results": {
            "niah_synth": {"256": {"em": 0.6, "n": 20},
                           "512": {"em": 0.5, "n": 20}},
            "fwe": {"256": {"em": 0.2, "n": 20},
                    "512": {"em": 0.2, "n": 20},
                    "979": {"em": 0.2, "n": 20}},
        },
    }
    assert context_macro_means(broken, align_requested=True) == {}
    # 旧产物无 ctx_lens：仅在所有任务共享同一实际档位集合时回退。
    old_ok = _payload({
        "a": {"256": {"em": 0.5, "n": 20}, "512": {"em": 0.4, "n": 20}},
        "b": {"256": {"em": 0.3, "n": 20}, "512": {"em": 0.2, "n": 20}},
    })
    assert set(context_macro_means(old_ok, align_requested=True)) == {256, 512}
    old_bad = _payload({
        "a": {"256": {"em": 0.5, "n": 20}, "512": {"em": 0.4, "n": 20}},
        "b": {"256": {"em": 0.3, "n": 20}},
    })
    assert context_macro_means(old_bad, align_requested=True) == {}


def test_context_macro_means_rejects_impossible_context_alignment():
    """实际 ctx 只能是请求档位向下 clamp：向上/错位网格不能硬配成假趋势。"""
    # 请求 1024，实际却出现 2048（生成预算只会缩短上下文，不会变长）。
    upward = {
        "ctx_lens": [256, 512, 1024],
        "results": {
            "a": {"256": {"em": 0.9, "n": 10},
                  "512": {"em": 0.5, "n": 10},
                  "2048": {"em": 0.1, "n": 10}},
            "b": {"256": {"em": 0.8, "n": 10},
                  "512": {"em": 0.4, "n": 10},
                  "2048": {"em": 0.0, "n": 10}},
        },
    }
    assert context_macro_means(upward, align_requested=True) == {}
    # 实际档位整体向下错位（128/256/512 对 256/512/1024）：每档都跌到
    # 上一请求档位或以下，说明网格与 ctx_lens 不是同一套请求档位。
    shifted = {
        "ctx_lens": [256, 512, 1024],
        "results": {
            "a": {"128": {"em": 0.9, "n": 10},
                  "256": {"em": 0.5, "n": 10},
                  "512": {"em": 0.1, "n": 10}},
        },
    }
    assert context_macro_means(shifted, align_requested=True) == {}
    # 多档时第一档被 clamp 会让后面档位也 clamp 到同一实际值并少档；
    # 能凑满档数却从更小的第一档开始，说明不是同一套请求网格。
    first_clamped = {
        "ctx_lens": [512, 1024],
        "results": {"a": {"256": {"em": 0.9, "n": 10},
                          "1024": {"em": 0.1, "n": 10}}},
    }
    assert context_macro_means(first_clamped, align_requested=True) == {}
    # 单档只有向下 clamp（请求 1024、实际 979）仍应保留描述统计。
    clamped = {
        "ctx_lens": [1024],
        "results": {"a": {"979": {"em": 0.6, "n": 10}}},
    }
    assert context_macro_means(clamped, align_requested=True) == {1024: 0.6}


def test_context_macro_means_matches_ruler_avg_when_balanced():
    """平衡设计下，逐档等权平均应严格等于项目既有的 RULER macro 平均。"""
    from run_all import _ruler_avg_of
    payload = _payload({
        "niah_synth": {
            "512": {"em": 0.8, "n": 20},
            "1024": {"em": 0.6, "n": 20},
            "2048": {"em": 0.4, "n": 20},
        },
        "vt": {
            "512": {"em": 0.2, "n": 20},
            "1024": {"em": 0.3, "n": 20},
            "2048": {"em": 0.4, "n": 20},
        },
    })
    macro = context_macro_means(payload)
    by_ctx = sum(macro.values()) / len(macro)
    assert abs(by_ctx - _ruler_avg_of(payload)) < 1e-12
    # 每档 macro = (0.8+0.2)/2, (0.6+0.3)/2, (0.4+0.4)/2 = 0.5, 0.45, 0.4
    # 每翻一倍变化 -0.05。
    assert math.isclose(log2_slope([512, 1024, 2048],
                                   [macro[c] for c in (512, 1024, 2048)]),
                        -0.05, rel_tol=0.0, abs_tol=1e-12)
