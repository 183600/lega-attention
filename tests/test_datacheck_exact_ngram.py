"""全量 train 扫描的精确 n-gram 泄漏证据测试。"""
import json
import os
import tempfile

import numpy as np

from lega.data import CACHE_SCHEMA, prepare_bins
from lega.datacheck import (CRITICAL_SPLIT_ALERT_CODES, critical_split_alerts,
                            exact_overlap_alerts, ngram_overlap_exact)


def test_ngram_overlap_exact_detects_planted_leak_and_clear_streams():
    rng = np.random.default_rng(0)
    leak = rng.integers(0, 1000, size=64).astype(np.uint16)
    train = np.concatenate(
        [rng.integers(0, 1000, size=5000).astype(np.uint16), leak])
    test = np.concatenate(
        [rng.integers(0, 1000, size=3000).astype(np.uint16), leak])
    res = ngram_overlap_exact(train, test)
    assert res["windows_found_in_reference"] == 1
    assert res["overlap_rate"] > 0
    assert res["eval_sampling"] == "all"
    assert res["screening"] is False
    assert res["exact_scan"] is True

    a = rng.integers(10 ** 6, 2 * 10 ** 6, size=5000).astype(np.uint16)
    b = rng.integers(0, 10 ** 6, size=4000).astype(np.uint16)
    res2 = ngram_overlap_exact(a, b)
    assert res2["windows_found_in_reference"] == 0
    assert res2["overlap_rate"] == 0.0

    dup_train = np.array([1, 2, 3, 4, 5, 6, 7, 8] * 3, dtype=np.uint16)
    dup_eval = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.uint16)
    res3 = ngram_overlap_exact(dup_train, dup_eval, n=4)
    assert res3["windows_found_in_reference"] == 5
    assert res3["overlap_rate"] == 1.0


def test_ngram_overlap_exact_is_chunk_invariant_and_respects_limits():
    rng = np.random.default_rng(1)
    train = rng.integers(0, 50000, size=20000, dtype=np.uint16)
    eval_ids = np.concatenate(
        [rng.integers(0, 50000, size=1000, dtype=np.uint16),
         train[1234:1334]])
    small = ngram_overlap_exact(train, eval_ids, n=16, chunk_tokens=137)
    large = ngram_overlap_exact(train, eval_ids, n=16, chunk_tokens=5000)
    for key in ("windows_found_in_reference", "overlap_rate",
                "eval_windows_checked", "reference_windows_scanned"):
        assert small[key] == large[key]
    assert small["windows_found_in_reference"] > 0

    limited = ngram_overlap_exact(train, eval_ids, n=16,
                                  max_eval_windows=100)
    assert limited["eval_sampling"] == "evenly_spaced"
    assert limited["screening"] is True
    assert limited["eval_windows_checked"] == 100

    unk_train = np.array([0] * 50 + [1] * 50, dtype=np.uint16)
    unk_eval = np.array([0] * 50 + [2] * 50, dtype=np.uint16)
    raw = ngram_overlap_exact(unk_train, unk_eval, n=8)
    assert raw["windows_found_in_reference"] > 0
    ignored = ngram_overlap_exact(unk_train, unk_eval, n=8,
                                  ignore_ids={0})
    assert ignored["windows_found_in_reference"] == 0
    assert ignored["overlap_rate"] == 0.0
    assert ignored["eval_windows_excluded_ignore"] > 0


def test_ngram_overlap_exact_rejects_bad_ids_and_params():
    ok = np.array([0, 1, 2, 3], dtype=np.int64)
    for bad in (np.array([70000], dtype=np.int64),
                np.array([-1], dtype=np.int64),
                np.array([1.5], dtype=np.float64),
                np.array([True, False])):
        try:
            ngram_overlap_exact(bad, ok, n=2)
            raise AssertionError(f"非法 token 流未被拒绝: {bad!r}")
        except ValueError:
            pass
    for kwargs in ({"n": 0}, {"n": -3}, {"n": True},
                   {"n": 2, "max_eval_windows": 0},
                   {"n": 2, "max_eval_windows": True},
                   {"n": 2, "chunk_tokens": 0},
                   {"n": 2, "chunk_tokens": False}):
        try:
            ngram_overlap_exact(ok, ok, **kwargs)
            raise AssertionError(f"非法参数未被拒绝: {kwargs}")
        except ValueError:
            pass
    for bad_ignore in ({1.5}, {True}, {-1}, {65536}, {float("nan")}, 5):
        try:
            ngram_overlap_exact(ok, ok, n=2, ignore_ids=bad_ignore)
            raise AssertionError(f"非法 ignore_ids 未被拒绝: {bad_ignore!r}")
        except ValueError:
            pass
    assert ngram_overlap_exact(ok, ok, n=2, ignore_ids={0.0})[
        "eval_windows_checked"] == 2


def test_exact_overlap_alerts_are_critical_and_scope_visible():
    exact = {
        "train_val": {"n": 64, "overlap_rate": 0.002,
                      "reference_split": "train",
                      "eval_windows_checked": 10, "eval_windows_total": 10,
                      "eval_sampling": "all"},
        "train_test": {"n": 64, "overlap_rate": 0.0,
                       "reference_split": "train",
                       "eval_windows_checked": 10, "eval_windows_total": 10,
                       "eval_sampling": "all"},
        "val_test": {"n": 64, "overlap_rate": 0.5,
                     "reference_split": "val",
                     "eval_windows_checked": 100, "eval_windows_total": 200,
                     "eval_sampling": "evenly_spaced"},
    }
    alerts = exact_overlap_alerts(exact)
    codes = [a["code"] for a in alerts]
    assert codes == ["ngram_overlap_exact_train_val",
                     "ngram_overlap_exact_val_test"]
    assert "全量 eval 窗口" in alerts[0]["message"]
    assert "等间隔抽样 100/200" in alerts[1]["message"]
    assert exact_overlap_alerts({"train_val": {"overlap_rate": None}}) == []
    assert exact_overlap_alerts(None) == []
    assert "ngram_overlap_exact_train_val" in CRITICAL_SPLIT_ALERT_CODES
    picked = critical_split_alerts({"alerts": alerts})
    assert [a["code"] for a in picked] == codes
    high_threshold = exact_overlap_alerts(
        exact, thresholds={"ngram_overlap_rate": 0.01})
    assert [a["code"] for a in high_threshold] == [
        "ngram_overlap_exact_val_test"]


def test_prepare_bins_records_exact_scan_and_upgrades_cache_schema(monkeypatch):
    import lega.data as data_mod
    monkeypatch.setattr(data_mod, "get_gpt2_tokenizer",
                        lambda: (_ for _ in ()).throw(RuntimeError("no tiktoken")))
    with tempfile.TemporaryDirectory() as d:
        segment = "abcdefghij" * 7
        with open(os.path.join(d, "train.txt"), "w", encoding="utf-8") as f:
            f.write("TRAINPREFIX " + segment + " TRAINSUFFIX\n")
            f.write("another unique training line that is long enough for the protocol\n")
        with open(os.path.join(d, "val.txt"), "w", encoding="utf-8") as f:
            f.write(segment + "\n")
        with open(os.path.join(d, "test.txt"), "w", encoding="utf-8") as f:
            f.write("Z" * 80 + "\n")
        meta = prepare_bins(d, kind="auto")
        assert meta["cache_schema"] == CACHE_SCHEMA
        assert CACHE_SCHEMA >= 10
        with open(os.path.join(d, "data_report.json"), encoding="utf-8") as f:
            rep = json.load(f)
        exact = rep["contamination_exact"]
        assert exact["train_val"]["windows_found_in_reference"] > 0
        assert exact["train_val"]["exact_scan"] is True
        assert exact["train_test"]["eval_sampling"] == "all"
        codes = {a["code"] for a in rep["alerts"]}
        assert "ngram_overlap_exact_train_val" in codes
        assert "ngram_overlap_exact_train_test" not in codes
        critical = {a["code"] for a in critical_split_alerts(rep)}
        assert "ngram_overlap_exact_train_val" in critical
