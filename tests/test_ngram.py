"""简单统计基线测试：n-gram 频次/PPL 公式、α 只从验证集选择、test 尘封。

科研口径：基线必须零调参空间可复算（公式逐一核对）；平滑系数只在
验证集上选；“测试集尘封”协议对基线评测同样生效（读取即写访问账本）。
"""
import json
import math
import os
import shutil
import tempfile

import numpy as np

from lega.ngram import (ALPHA_GRID, evaluate, fit_counts, ppl_bigram,
                        ppl_unigram)
from lega.datacheck import TEST_ACCESS_LOG


def _write_bin(path, ids):
    np.asarray(ids, dtype=np.uint16).tofile(path)


def _mk_data(tmp, train_ids, val_ids, vocab_size=4, test_ids=None):
    os.makedirs(tmp, exist_ok=True)
    with open(os.path.join(tmp, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({"tokenizer": "char", "vocab_size": vocab_size}, f)
    _write_bin(os.path.join(tmp, "train.bin"), train_ids)
    _write_bin(os.path.join(tmp, "val.bin"), val_ids)
    if test_ids is not None:
        _write_bin(os.path.join(tmp, "test.bin"), test_ids)
    return tmp


def test_ppl_unigram_matches_closed_form():
    uni = np.array([2.0, 1.0, 0.0])
    # eval=[0,1], α=1：P(0)=3/6=0.5，P(1)=2/6=1/3 → PPL=sqrt(6)
    out = ppl_unigram(np.array([0, 1]), uni, alpha=1.0)
    assert math.isclose(out["ppl"], math.sqrt(6.0), rel_tol=1e-12)
    assert out["n_tokens"] == 2
    # 空评测集：None 而不是崩溃
    assert ppl_unigram(np.array([]), uni, alpha=1.0)["ppl"] is None
    print("  [ok] ppl_unigram（加 α 平滑闭式解）")


def test_ppl_bigram_matches_closed_form():
    uni, keys, counts = fit_counts(np.array([0, 1, 0, 1]), vocab_size=3)
    assert uni[0] == 2 and uni[1] == 2
    # eval=[0,1], α=1, V=3：P(1|0)=(2+1)/(2+3)=0.6 → PPL=1/0.6
    out = ppl_bigram(np.array([0, 1]), uni, keys, counts, alpha=1.0,
                     vocab_size=3)
    assert math.isclose(out["ppl"], 1.0 / 0.6, rel_tol=1e-12)
    # 单 token 训练流：无法形成二元组，仍安全返回（未出现对按 0 计数）
    uni2, keys2, counts2 = fit_counts(np.array([1]), vocab_size=3)
    out2 = ppl_bigram(np.array([0, 1]), uni2, keys2, counts2, alpha=1.0,
                      vocab_size=3)
    assert out2["ppl"] is not None and math.isfinite(out2["ppl"])
    assert ppl_bigram(np.array([1]), uni, keys, counts, 1.0, 3)["ppl"] is None
    print("  [ok] ppl_bigram（条件概率闭式解、单 token 训练流容错）")


def test_evaluate_end_to_end_and_alpha_selected_on_val():
    tmp = tempfile.mkdtemp()
    try:
        train = ([0, 1, 2, 3] * 30)
        val = [0, 1, 2, 3, 0, 1, 2, 0]
        _mk_data(tmp, train, val, vocab_size=4)
        rep = evaluate(tmp, split="val", alphas=(0.01, 1.0))
        assert rep["train_tokens"] == len(train)
        assert rep["eval_tokens"] == len(val)
        assert rep["selected_alpha"]["unigram"] in (0.01, 1.0)
        assert rep["selected_alpha"]["bigram"] in (0.01, 1.0)
        assert rep["ppl"]["unigram"] > 1.0 and rep["ppl"]["bigram"] > 1.0
        assert rep["selection_eval_same_split"] is True
        assert rep["allow_test"] is False
        # 选定 α 的 PPL 必须是该阶数在候选网格中的最优（按 mean NLL）
        for order in ("unigram", "bigram"):
            chosen = str(rep["selected_alpha"][order])
            best_ppl = min(rep["ppl_by_alpha"][order].values())
            assert abs(rep["ppl_by_alpha"][order][chosen] - best_ppl) < 1e-9
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] evaluate（端到端、α 在 val 选优、候选网格完整）")


def test_test_split_sealed_and_access_logged():
    tmp = tempfile.mkdtemp()
    try:
        _mk_data(tmp, [0, 1, 2, 3] * 20, [0, 1, 2, 3], vocab_size=4,
                 test_ids=[0, 1, 2, 3, 0, 1])
        # 未显式开锁：必须拒绝
        try:
            evaluate(tmp, split="test")
            raise AssertionError("split='test' 未 allow_test 应抛 PermissionError")
        except PermissionError:
            pass
        # 显式开锁：可评测且写访问账本；test 只消费选定 α（不扫网格）
        rep = evaluate(tmp, split="test", alphas=(0.01, 1.0), allow_test=True)
        assert rep["allow_test"] is True
        assert rep["ppl"]["unigram"] is not None
        assert rep["ppl_by_alpha"] == {}
        log_path = os.path.join(tmp, TEST_ACCESS_LOG)
        assert os.path.exists(log_path)
        text = open(log_path, encoding="utf-8").read()
        assert "ngram_baseline" in text
        assert ALPHA_GRID[0] > 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] evaluate（test 默认尘封、开锁写账本、test 单次消费）")
