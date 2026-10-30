"""经验长度校准审计的纯统计与固定效应测试。"""
import json
import math
import os
import shutil
import sys
import tempfile

import numpy as np

from lega.lengthcalib import (_ids_start, _window_ids,
                              analyze_calibration_records,
                              calibration_window_starts,
                              collect_calibration_records,
                              parse_growth_modes, parse_lengths)
from lega.model import GPT
from lega.pipeline.report.sections import _render_length_calibration


def _records(slope_by_head, lengths=(32, 64, 128, 256), windows=4, q=2):
    recs = []
    for w in range(windows):
        for offset in range(q):
            for head, slope in slope_by_head.items():
                for i, length in enumerate(lengths):
                    c = slope * i
                    recs.append({
                        "window": w, "query_offset": offset,
                        "length": length, "layer": 0, "head": head,
                        "c": c,
                        "log_max_p": -math.log(length) + (c - math.log(length)),
                        "log_n": math.log(length),
                        "g": 0.1 + 0.01 * i,
                        "gate_logit": 0.2 + 0.02 * i,
                    })
    return recs


def test_parse_lengths_and_growth_modes_reject_ambiguous_inputs():
    assert parse_lengths("128,32,64") == [32, 64, 128]
    assert parse_growth_modes(["suffix", "prefix"]) == ["suffix", "prefix"]
    for bad in ("32", "32,32", "1,2", "32,True", "32,64.5"):
        try:
            parse_lengths(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"应拒绝 {bad!r}")
    for bad in ("suffix,center", "", ["suffix", "suffix"]):
        try:
            parse_growth_modes(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"应拒绝 {bad!r}")


def test_window_starts_are_evenly_spaced_and_aligned_to_max_length():
    starts = calibration_window_starts(1000, 100, 4)
    assert starts == [0, 200, 500, 800]
    try:
        calibration_window_starts(50, 100, 2)
    except ValueError:
        pass
    else:
        raise AssertionError("val 太短时应明确报错")


def test_suffix_growth_uses_a_fixed_window_end():
    val = np.arange(128, dtype=np.uint16)
    assert np.array_equal(_window_ids(val, 0, 8, "suffix", 16),
                          val[8:16])
    assert np.array_equal(_window_ids(val, 0, 16, "suffix", 16),
                          val[0:16])
    assert _ids_start(32, 8, "suffix", 16) == 40
    assert _ids_start(32, 8, "prefix", 16) == 32


def test_analyze_recovers_known_fixed_effect_slopes_and_log_n_sanity():
    recs = _records({0: 0.0, 1: 1.0})
    out = analyze_calibration_records(
        recs, (32, 64, 128, 256), "suffix", tolerance=0.05,
        n_bootstrap=100, seed=7)
    assert abs(out["summary"]["c"]["median"] - 0.5) < 1e-12
    assert out["summary"]["c"]["n_flat"] == 1
    assert out["summary"]["c"]["n_increasing"] == 1
    assert out["summary"]["c"]["fraction_flat"] == 0.5
    assert out["summary"]["log_n"]["available"] is True
    assert abs(out["summary"]["log_n"]["sanity_slope_from_all"]
               - math.log(2.0)) < 1e-12
    by_head = {r["head"]: r for r in out["per_head"]}
    assert abs(by_head[0]["c_slope"]) < 1e-12
    assert abs(by_head[1]["c_slope"] - 1.0) < 1e-12
    assert by_head[1]["c_ci95"][0] > 0.0
    assert by_head[0]["c_class"] == "flat"


def test_bootstrap_ci_is_cluster_level_and_missing_values_are_not_fabricated():
    recs = _records({0: 1.0}, windows=5)
    recs.append({"window": 0, "query_offset": 0, "length": 32,
                 "layer": 0, "head": 0, "c": None})
    out = analyze_calibration_records(
        recs, (32, 64, 128, 256), "suffix", n_bootstrap=100, seed=3)
    row = out["per_head"][0]
    assert row["c_class"] == "increasing"
    assert row["c_ci95"][0] > 0.0
    assert row["c_n_groups"] == 10
    assert row["c_n_obs"] == 40


def test_collect_calibration_records_batches_without_cross_sample_leakage():
    torch = __import__("torch")
    torch.manual_seed(0)
    model = GPT(vocab_size=64, block_size=32, d_model=16, n_layers=1,
                n_heads=2, ffn_mult=2, variant="lega",
                use_sdpa_standard=False)
    model.eval()
    val = np.arange(128, dtype=np.uint16)
    records = collect_calibration_records(
        model, val, [0, 32], [8, 16], query_positions=2,
        growth="suffix", device="cpu", amp=False, batch_windows=2)
    assert len(records) == 2 * 2 * 2 * 1 * 2
    starts = {(r["window"], r["length"]): r["window_start"] for r in records}
    assert starts[(0, 8)] == 8
    assert starts[(0, 16)] == 0
    assert all(math.isfinite(r["c"]) for r in records)


def test_summary_render_reports_observed_slope_and_requested_missing():
    payload = {
        "lengths": [32, 64],
        "window_starts": [0, 32, 64, 96],
        "results": {"suffix": {
            "lengths": [32, 64],
            "per_length": {
                "32": {"c": {"mean": -0.5}},
                "64": {"c": {"mean": -0.4}},
            },
            "summary": {
                "c": {"median": 0.1, "fraction_flat": 0.8, "n_heads": 10},
                "log_max_p": {"median": -0.9},
                "gate_logit": {"median": 0.2},
            },
        }},
    }
    md = "\n".join(_render_length_calibration(
        {"lega": {7: payload}}, requested=True, random_slope=0.3))
    assert "经验长度校准审计" in md
    assert "lega" in md and "suffix" in md and "80%" in md
    assert "0.3" in md
    missing = "\n".join(_render_length_calibration({}, requested=True))
    assert "没有找到可用" in missing


def test_main_records_sys_argv(monkeypatch):
    """模块入口必须记录真实 sys.argv（曾因漏 import sys 在 postrun 崩溃）。"""
    import lega.lengthcalib as lc

    tmp = tempfile.mkdtemp(prefix="lengthcalib_argv_")
    try:
        def fake_run(*args, **kwargs):
            return {"variant": "lega", "lengths": [8, 16],
                    "window_starts": [], "results": {}}

        monkeypatch.setattr(lc, "run_length_calibration", fake_run)
        out = os.path.join(tmp, "length_calibration.json")
        argv = ["-m lega.lengthcalib", "--ckpt", os.path.join(tmp, "model.pt"),
                "--data_dir", tmp, "--lengths", "8,16", "--out", out]
        monkeypatch.setattr(sys, "argv", argv)
        assert lc.main(None) == 0
        with open(out, encoding="utf-8") as f:
            payload = json.load(f)
        assert payload["argv"] == argv
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
