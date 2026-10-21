"""简单统计基线测试：n-gram 频次/PPL 公式、α 只从验证集选择、test 尘封。

科研口径：基线必须零调参空间可复算（公式逐一核对）；平滑系数只在
验证集上选；“测试集尘封”协议对基线评测同样生效（读取即写访问账本）。
"""
import json
import math
import os
import shutil
import sys
import tempfile

import numpy as np

from lega.ngram import (ALPHA_GRID, INTERP_LAMBDA_GRID, _bigram_ppl,
                        _interp_ppl_from_counts, evaluate, fit_counts,
                        ppl_bigram, ppl_interpolated_bigram, ppl_unigram,
                        window_positions)
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


def test_invalid_smoothing_parameters_rejected():
    """α/λ 非有限或越界时显式报错，不能写出 NaN/Inf 报告或静默选参。"""
    uni = np.array([2.0, 1.0, 0.0])
    for bad in (float("nan"), float("inf"), float("-inf"), -0.1, "bad"):
        try:
            ppl_unigram(np.array([0, 1]), uni, bad)
            raise AssertionError(f"非法 α 应报错: {bad!r}")
        except ValueError:
            pass
    _, keys, counts = fit_counts(np.array([0, 1, 0, 1]), vocab_size=3)
    for lam in (float("nan"), float("inf"), -0.1, 1.1):
        try:
            ppl_interpolated_bigram(np.array([0, 1]), uni, keys, counts,
                                    1.0, 1.0, lam, 3)
            raise AssertionError(f"非法 λ 应报错: {lam!r}")
        except ValueError:
            pass
    tmp = tempfile.mkdtemp()
    try:
        _mk_data(tmp, [0, 1, 2, 3] * 5, [0, 1, 2, 0], vocab_size=4)
        for grid in ((), (float("nan"),), (float("inf"),), (-1.0,)):
            try:
                evaluate(tmp, split="val", alphas=grid)
                raise AssertionError(f"非法 α 网格应报错: {grid!r}")
            except ValueError:
                pass
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] n-gram：非法 α/λ 显式拒绝")


def test_zero_smoothing_unseen_tokens_stay_finite():
    """α=0 是合法输入：未登录 token/上下文概率为 0，PPL 必须是有限的
    截断值而不是 NaN/Inf（后者会让 JSON 报告字段退化为 null）。"""
    uni, keys, counts = fit_counts(np.array([0, 1, 0]), vocab_size=3)
    out = ppl_unigram(np.array([2]), uni, alpha=0.0)
    assert math.isfinite(out["mean_nll"]) and math.isfinite(out["ppl"])
    assert out["ppl_clamped"] is True
    assert math.isclose(out["ppl"], math.exp(20.0), rel_tol=1e-12)
    # prev=2 在训练流中从未作为上下文出现：0/0 不能变成 NaN。
    out2 = ppl_bigram(np.array([2, 2]), uni, keys, counts, 0.0, 3)
    assert math.isfinite(out2["mean_nll"]) and math.isfinite(out2["ppl"])
    out3 = ppl_interpolated_bigram(np.array([2, 2]), uni, keys, counts,
                                   0.0, 0.0, 0.5, 3)
    assert math.isfinite(out3["mean_nll"]) and math.isfinite(out3["ppl"])
    print("  [ok] n-gram：α=0 时未登录 token/上下文仍返回有限 PPL")


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
    # 训练流最后一个 token 没有后继，不能作为二元前文计入分母：
    # train=[0,1,2,0] 时 prev=0 只出现 1 次有效上下文（首个 0），
    # 而不是 uni[0]=2；P(1|0)=(1+1)/(1+3)=0.5，故 PPL=2。
    uni3, keys3, counts3 = fit_counts(np.array([0, 1, 2, 0]), vocab_size=3)
    out3 = ppl_bigram(np.array([0, 1]), uni3, keys3, counts3, alpha=1.0,
                      vocab_size=3)
    assert math.isclose(out3["ppl"], 2.0, rel_tol=1e-12)
    print("  [ok] ppl_bigram（条件概率闭式解、单 token 训练流容错、"
          "末位 token 不计上下文）")


def test_ppl_interpolated_bigram_matches_closed_form():
    """插值 bigram 闭式核对：λ·P(w|prev)+(1−λ)·P(w)，λ=1 退化为 bigram。"""
    uni, keys, counts = fit_counts(np.array([0, 1, 0, 1]), vocab_size=3)
    # eval=[0,1]，αu=αb=1，V=3：
    # P(1|0)=(2+1)/(2+3)=0.6；P(1)=(2+1)/(4+3)=3/7
    # λ=0.5 → p=0.5·0.6+0.5·3/7；λ=1 → 纯 bigram 0.6
    half = ppl_interpolated_bigram(np.array([0, 1]), uni, keys, counts,
                                   1.0, 1.0, 0.5, 3)
    exp_p = 0.5 * 0.6 + 0.5 * (3.0 / 7.0)
    assert math.isclose(half["ppl"], 1.0 / exp_p, rel_tol=1e-12)
    full = ppl_interpolated_bigram(np.array([0, 1]), uni, keys, counts,
                                   1.0, 1.0, 1.0, 3)
    assert math.isclose(full["ppl"], 1.0 / 0.6, rel_tol=1e-12)
    # 空/过短：安全返回 None 而不是崩溃
    assert ppl_interpolated_bigram(np.array([1]), uni, keys, counts,
                                   1.0, 1.0, 0.5, 3)["ppl"] is None
    print("  [ok] ppl_interpolated_bigram（闭式解、λ=1 退化、边界）")


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
        # 插值 bigram 由 val 联合选参，且不得比纯 bigram 更差（λ=1 可退化）
        assert rep["ppl"]["interpolated_bigram"] is not None
        si = rep["selected_interpolated"]
        assert si["lambda"] in INTERP_LAMBDA_GRID
        assert si["alpha_unigram"] in (0.01, 1.0)
        assert si["alpha_bigram"] in (0.01, 1.0)
        assert rep["selected_model"] in rep["ppl"]
        assert (rep["val_mean_nll"]["interpolated_bigram"]
                <= rep["val_mean_nll"]["bigram"] + 1e-9)
        assert rep["selected_model_selected_on"].startswith("val")
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


def test_evaluate_non_val_split_still_selects_alpha_on_val():
    """split='train' 只改变报告集，α/λ 仍必须在 val 上选择。

    旧实现把 sel_ids 设为 eval_ids，train 报告会静默用 train 自选参数，
    却仍声明 alpha_selected_on='val'。本数据上 train 自选（bigram α=0.01、
    λ=1.0）与 val 自选（α=1.0、λ=0.5）不同，可直接区分两种口径。
    """
    tmp = tempfile.mkdtemp()
    try:
        train = [3, 2, 2, 1, 1, 0, 0, 0, 0, 3, 2, 3, 2, 2, 3, 2, 2, 2,
                 2, 3, 1, 3, 2, 0, 1, 3, 2, 0, 3, 2, 3, 0, 0, 3, 0, 2,
                 0, 1, 1, 1, 1, 0, 0, 0, 0, 2, 2, 2, 1, 2, 3, 1, 1, 3,
                 3, 3, 1, 2, 3, 2]
        val = [3, 2, 2, 1, 3, 0, 2, 2, 3, 2, 1, 1]
        _mk_data(tmp, train, val, vocab_size=4)
        grid = (0.01, 0.1, 1.0)
        val_rep = evaluate(tmp, split="val", alphas=grid)
        train_rep = evaluate(tmp, split="train", alphas=grid)
        assert train_rep["eval_tokens"] == len(train)
        assert train_rep["selection_eval_same_split"] is False
        assert train_rep["selected_alpha"] == val_rep["selected_alpha"]
        assert (train_rep["selected_interpolated"]
                == val_rep["selected_interpolated"])
        assert train_rep["val_mean_nll"] == val_rep["val_mean_nll"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] evaluate（非 val 报告集仍只用 val 选参）")


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
        from lega.repro import sha256_file
        assert rep["test_access"]["distinct_test_data_versions"] == 1
        assert sha256_file(os.path.join(tmp, "test.bin")) in \
            rep["test_access"]["test_data_fingerprints"]
        assert ALPHA_GRID[0] > 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] evaluate（test 默认尘封、开锁写账本、test 单次消费）")


def test_window_positions_match_neural_eval_rule():
    """窗口评分位置必须与 train._val_eval_indices 的规则逐例一致。

    神经 PPL 的窗口选择规则若与 n-gram 对齐用的规则漂移，两侧又重新
    变成不同 token 集合上的 PPL；这里直接与训练侧实现逐例对照。
    """
    from lega.train import _val_eval_indices
    cases = [(500, 32, 1), (500, 32, 5), (500, 32, 100), (129, 64, 2),
             (65, 16, 3), (33, 32, 1), (1000, 128, 50)]
    for n_tokens, ctx, iters in cases:
        starts, _ = _val_eval_indices(n_tokens, ctx, iters)
        expected = np.concatenate([np.arange(s * ctx + 1, s * ctx + ctx + 1)
                                   for s in starts])
        got = window_positions(n_tokens, ctx, iters)
        assert got is not None and np.array_equal(got, expected), \
            f"不一致: n={n_tokens} ctx={ctx} iters={iters}"
    # 规格非法/流过短：None 而不是崩溃或静默错位
    assert window_positions(10, 16, 2) is None
    assert window_positions(100, 0, 2) is None
    assert window_positions(100, 16, 0) is None
    # bool/小数/非有限值不能被 int() 截断成“看似合法的窗口规格”
    for bad in (True, 100.5, "100.5", float("nan"), float("inf")):
        assert window_positions(bad, 16, 2) is None, bad
        assert window_positions(100, bad, 2) is None, bad
        assert window_positions(100, 16, bad) is None, bad
    assert window_positions("100", "16", "2") is not None
    print("  [ok] window_positions（与 _val_eval_indices 逐例一致、非法规格返回 None）")


def test_evaluate_window_matched_restricts_scoring_positions():
    """window_spec 下额外报告逐位置对齐 PPL，且等于手工按同位置的计算。"""
    tmp = tempfile.mkdtemp()
    try:
        train = [0, 1, 2, 3] * 30
        # val 长度 41：ctx=8 时 n_windows=(41-1)//8=5；iters=2 → 窗口 [0, 4]
        val = list(range(4)) * 10 + [0]
        _mk_data(tmp, train, val, vocab_size=4)
        rep = evaluate(tmp, split="val", alphas=(0.01,),
                       window_spec={"ctx": 8, "iters": 2})
        assert rep["window_spec"] == {"ctx": 8, "iters": 2, "n_windows": 5,
                                      "n_eval": 2, "n_tokens": 16}
        wm = rep["ppl_window_matched"]
        assert wm["n_tokens"] == 16
        # 手工复算：同样的 position 集合、同样的选定 α，必须逐位一致
        uni, keys, counts = fit_counts(np.asarray(train), vocab_size=4)
        pos = window_positions(len(val), 8, 2)
        m_uni = ppl_unigram(np.asarray(val)[pos], uni, 0.01)
        assert math.isclose(wm["unigram"], m_uni["ppl"], rel_tol=1e-12)
        prev = np.asarray(val)[pos - 1]
        keys_pos = prev * 4 + np.asarray(val)[pos]
        c_pair = np.zeros(len(pos))
        lookup = {int(k): float(c) for k, c in zip(keys, counts)}
        for i, k in enumerate(keys_pos):
            c_pair[i] = lookup.get(int(k), 0.0)
        prev_counts = np.bincount(keys // 4, weights=counts, minlength=4)
        m_bi = _bigram_ppl(c_pair, prev_counts[prev], len(pos), 0.01, 4)
        assert math.isclose(wm["bigram"], m_bi["ppl"], rel_tol=1e-12)
        si = rep["selected_interpolated"]
        m_interp = _interp_ppl_from_counts(
            np.asarray(val)[pos], uni, c_pair, prev_counts[prev], len(pos),
            si["alpha_unigram"], si["alpha_bigram"], si["lambda"], 4)
        assert math.isclose(wm["interpolated_bigram"], m_interp["ppl"],
                            rel_tol=1e-12)
        # 一时一用的全 split 口径仍在（保留绝对难度参照）
        assert rep["ppl"]["unigram"] is not None
        # 非法规格：显式报错字段而不是假装对齐
        rep_bad = evaluate(tmp, split="val", alphas=(0.01,),
                           window_spec={"ctx": 999, "iters": 2})
        assert rep_bad["window_spec"] is None
        assert rep_bad["window_spec_error"]
        assert "ppl_window_matched" not in rep_bad
        # bool/小数规格不能被 int() 截断成整数口径：True → 1、8.5 → 8 都会
        # 让"与神经评测逐位置对齐"悄悄失效（对齐的是另一个 token 集合）。
        for bad in ({"ctx": True, "iters": 2}, {"ctx": 8.5, "iters": 2},
                    {"ctx": 8, "iters": True}, {"ctx": float("nan"), "iters": 2}):
            rb = evaluate(tmp, split="val", alphas=(0.01,), window_spec=bad)
            assert rb["window_spec"] is None, bad
            assert rb["window_spec_error"], bad
        # 整数值浮点/整数字符串语义明确，仍走对齐（JSON 既有用法）
        rep_ok = evaluate(tmp, split="val", alphas=(0.01,),
                          window_spec={"ctx": 8.0, "iters": "2"})
        assert rep_ok["window_spec"] == {"ctx": 8, "iters": 2, "n_windows": 5,
                                         "n_eval": 2, "n_tokens": 16}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] evaluate（window_spec 逐位置对齐、非法规格显式报错）")


def test_ngram_cli_runs_data_protocol_preflight():
    """独立 n-gram 评测也必须在已标记泄露的数据前中止。"""
    from lega.ngram import main as ngram_main
    old = sys.argv
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "data_report.json"), "w",
                  encoding="utf-8") as f:
            json.dump({"alerts": [{"level": "warning",
                                   "code": "split_shuffled",
                                   "message": "打乱"}]}, f)
        sys.argv = ["ngram", "--data_dir", d]
        try:
            ngram_main()
            raise AssertionError("n-gram 评测入口应执行数据协议闸门")
        except SystemExit as e:
            assert "数据协议前置检查失败" in str(e)
        finally:
            sys.argv = old
    print("  [ok] n-gram 评测入口执行数据协议闸门（严重告警默认中止）")
