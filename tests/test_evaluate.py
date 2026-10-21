"""语言建模坏案例测试：最差验证窗口转储（预测错误的样本要看得见）。"""
import json
import math
import os
import shutil
import sys
import tempfile

import numpy as np
import torch

from lega.data import CharTokenizer
from lega.evaluate import (_positive_int_or_none, checkpoint_selection_info,
                           collect_worst_windows, json_safe_eval_metrics,
                           load_train_coverage, load_train_peak_mem_mb,
                           load_train_throughput, repeated_test_checkpoint,
                           worst_window_screening, worst_windows_allowed)
from lega.model import GPT


def test_collect_worst_windows():
    """按末位下一 token 损失排序取最差窗口，字段齐全且降序。"""
    torch.manual_seed(0)
    tok = CharTokenizer(sorted("abcdefghij "))
    model = GPT(vocab_size=len(tok.itos), block_size=64, d_model=32, n_layers=1,
                n_heads=2, variant="lega")
    model.eval()
    # 合成验证流：64 token 窗口 × 32 个
    val = np.random.default_rng(0).integers(0, len(tok.itos), size=64 * 33)
    val = val.astype(np.uint16)
    rows = collect_worst_windows(model, val, ctx=64, device="cpu", tok=tok,
                                 n_worst=3, max_windows=32)
    assert len(rows) == 3
    assert all(set(r) >= {"window", "loss", "context_tail", "gold_next",
                          "pred_top5"} for r in rows)
    assert rows[0]["loss"] >= rows[1]["loss"] >= rows[2]["loss"]
    assert all(len(r["pred_top5"]) == 5 for r in rows)
    # top-5 为全词表 softmax 的截断：逐项概率降序、均在 (0,1]，总和不超过 1
    for r in rows:
        probs = [p for _, p in r["pred_top5"]]
        assert all(0.0 < p <= 1.0 for p in probs), probs
        assert sum(probs) <= 1.0 + 1e-9, probs
        assert probs == sorted(probs, reverse=True)
        assert r["context_tail"] and isinstance(r["gold_next"], str)
    # n_worst=0 表示关闭：返回空列表，而不是“至少转储一条”
    assert collect_worst_windows(model, val, ctx=64, device="cpu", tok=tok,
                                 n_worst=0, max_windows=32) == []
    print(f"  [ok] 最差窗口转储（top loss={rows[0]['loss']:.3f}，"
          f"gold={rows[0]['gold_next']!r}）")


def test_collect_worst_windows_forces_eval_and_restores_mode():
    """辅助函数必须自己关掉 dropout（否则“最差窗口”不可复现），并恢复调用方模式。"""
    torch.manual_seed(0)
    tok = CharTokenizer(sorted("abcdefghij "))
    model = GPT(vocab_size=len(tok.itos), block_size=64, d_model=32, n_layers=1,
                n_heads=2, variant="lega", dropout=0.5)
    val = np.random.default_rng(0).integers(0, len(tok.itos), size=64 * 33)
    val = val.astype(np.uint16)
    model.train()
    rows_train = collect_worst_windows(model, val, ctx=64, device="cpu",
                                       tok=tok, n_worst=3, max_windows=32)
    assert model.training, "调用方模式必须恢复"
    model.eval()
    rows_eval = collect_worst_windows(model, val, ctx=64, device="cpu",
                                      tok=tok, n_worst=3, max_windows=32)
    assert [r["window"] for r in rows_train] == [r["window"] for r in rows_eval]
    for a, b in zip(rows_train, rows_eval):
        assert math.isclose(a["loss"], b["loss"], rel_tol=1e-6, abs_tol=1e-6)
    print("  [ok] 最差窗口固定 eval、关闭 dropout 并恢复调用方模式")


def test_worst_window_screening_scope_is_recorded():
    """最差窗口必须连同“候选/总窗口数”一起落盘：不能把子集最差当成全量最差。"""
    val = np.zeros(64 * 34, dtype=np.uint16)
    sc = worst_window_screening(len(val), ctx=64, max_windows=30)
    assert sc["screened"] == 30 and sc["total"] == 33
    assert sc["screen_max"] == 30
    # 候选上限大于总窗口数时全部评估：覆盖率 100%
    sc2 = worst_window_screening(len(val), ctx=64, max_windows=100)
    assert sc2["screened"] == 33 and sc2["total"] == 33
    print("  [ok] 最差窗口筛查范围落盘（候选/总窗口数）")


def test_positive_int_config_value_degrades_to_none():
    """config_used.json 的手改数值不能让评测 int() 崩溃。"""
    assert _positive_int_or_none("bad") is None
    assert _positive_int_or_none(float("inf")) is None
    assert _positive_int_or_none(0) is None
    assert _positive_int_or_none(-3) is None
    assert _positive_int_or_none(8.5) is None
    assert _positive_int_or_none(True) is None
    assert _positive_int_or_none(float("nan")) is None
    assert _positive_int_or_none("128") == 128
    assert _positive_int_or_none(8.0) == 8
    print("  [ok] 非法 train_ctx/eval_iters 配置安全降级")


def test_evaluate_cli_rejects_nonpositive_eval_iters():
    from lega.evaluate import main as evaluate_main
    old = sys.argv
    try:
        for bad in ("0", "-3"):
            sys.argv = ["lega.evaluate", "--ckpt", "x", "--data_dir", "x",
                        "--out", "x", "--eval_iters", bad]
            try:
                evaluate_main()
            except SystemExit as exc:
                assert exc.code not in (0, None)
            else:
                raise AssertionError(f"--eval_iters={bad} 未被拒绝")
    finally:
        sys.argv = old
    print("  [ok] evaluate CLI：拒绝非正 eval_iters")


def test_load_train_throughput_propagates_and_degrades_safely():
    """训练过程记录必须随被评测 checkpoint 传播；缺失/损坏时安全返回 (None, None)。"""
    tmp = tempfile.mkdtemp()
    try:
        assert load_train_throughput(tmp) == (None, None)
        with open(os.path.join(tmp, "eval.json"), "w", encoding="utf-8") as f:
            json.dump({"train_step_time_ms": 12.5,
                       "train_tokens_per_sec": 81920.0}, f)
        assert load_train_throughput(tmp) == (12.5, 81920.0)
        # 旧格式（无该字段）→ None 而非 KeyError
        with open(os.path.join(tmp, "eval.json"), "w", encoding="utf-8") as f:
            json.dump({"val_ppl": 10.0}, f)
        assert load_train_throughput(tmp) == (None, None)
        # 损坏 JSON 不阻塞评测
        with open(os.path.join(tmp, "eval.json"), "w", encoding="utf-8") as f:
            f.write("{broken")
        assert load_train_throughput(tmp) == (None, None)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] 训练过程记录随 checkpoint 传播（缺失/损坏安全降级）")


def test_load_train_peak_mem_propagates_and_degrades_safely():
    """训练进程统计必须随 checkpoint 传播（run_all 汇总列只读评测结果文件）。"""
    tmp = tempfile.mkdtemp()
    try:
        assert load_train_peak_mem_mb(tmp) is None
        with open(os.path.join(tmp, "eval.json"), "w", encoding="utf-8") as f:
            json.dump({"train_peak_mem_mb": 321.5}, f)
        assert load_train_peak_mem_mb(tmp) == 321.5
        # 旧格式（无该字段）/非数值/损坏/结构异常 → None，不阻塞
        with open(os.path.join(tmp, "eval.json"), "w", encoding="utf-8") as f:
            json.dump({"val_ppl": 10.0}, f)
        assert load_train_peak_mem_mb(tmp) is None
        with open(os.path.join(tmp, "eval.json"), "w", encoding="utf-8") as f:
            json.dump({"train_peak_mem_mb": "n/a"}, f)
        assert load_train_peak_mem_mb(tmp) is None
        with open(os.path.join(tmp, "eval.json"), "w", encoding="utf-8") as f:
            json.dump({"train_peak_mem_mb": float("inf")}, f)
        assert load_train_peak_mem_mb(tmp) is None
        with open(os.path.join(tmp, "eval.json"), "w", encoding="utf-8") as f:
            json.dump({"train_peak_mem_mb": 10 ** 400}, f)
        assert load_train_peak_mem_mb(tmp) is None
        with open(os.path.join(tmp, "eval.json"), "w", encoding="utf-8") as f:
            f.write("{broken")
        assert load_train_peak_mem_mb(tmp) is None
        with open(os.path.join(tmp, "eval.json"), "w", encoding="utf-8") as f:
            json.dump([1, 2], f)
        assert load_train_peak_mem_mb(tmp) is None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] 训练进程统计随 checkpoint 传播（缺失/损坏安全降级）")


def test_load_train_coverage_propagates_and_degrades_safely():
    """训练覆盖必须随 checkpoint 传播：<1 epoch 的评测不能被当成收敛结果。"""
    tmp = tempfile.mkdtemp()
    try:
        assert load_train_coverage(tmp) == {}
        with open(os.path.join(tmp, "eval.json"), "w", encoding="utf-8") as f:
            json.dump({"tokens_seen": 20480, "train_tokens": 738388,
                       "epochs_seen": 0.0277}, f)
        cov = load_train_coverage(tmp)
        assert cov == {"train_tokens_seen": 20480, "train_tokens": 738388,
                       "epochs_seen": 0.0277}
        # 旧格式/非法值/损坏/结构异常 → 安全降级，不阻塞评测
        with open(os.path.join(tmp, "eval.json"), "w", encoding="utf-8") as f:
            json.dump({"epochs_seen": float("inf"), "train_tokens": "x"}, f)
        assert load_train_coverage(tmp) == {}
        with open(os.path.join(tmp, "eval.json"), "w", encoding="utf-8") as f:
            json.dump({"epochs_seen": 10 ** 400}, f)
        assert load_train_coverage(tmp) == {}
        with open(os.path.join(tmp, "eval.json"), "w", encoding="utf-8") as f:
            f.write("{broken")
        assert load_train_coverage(tmp) == {}
        with open(os.path.join(tmp, "eval.json"), "w", encoding="utf-8") as f:
            json.dump([1, 2], f)
        assert load_train_coverage(tmp) == {}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] 训练覆盖率随 checkpoint 传播（缺失/损坏安全降级）")


def test_collect_worst_windows_gold_in_context_uses_full_window():
    """gold_in_context 必须按完整窗口判定，而不是只看 160 字符的上下文尾部。"""
    torch.manual_seed(0)
    tok = CharTokenizer(sorted("abcdefghij "))
    model = GPT(vocab_size=len(tok.itos), block_size=64, d_model=32, n_layers=1,
                n_heads=2, variant="lega")
    model.eval()
    val = np.random.default_rng(0).integers(0, len(tok.itos), size=64 * 33)
    val = val.astype(np.uint16)
    rows = collect_worst_windows(model, val, ctx=64, device="cpu", tok=tok,
                                 n_worst=3, max_windows=32)
    for r in rows:
        w = int(r["window"])
        toks = val[w * 64:(w + 1) * 64].tolist()
        full = tok.decode(toks)
        gold_id = int(val[w * 64 + 64])
        gold = tok.decode([gold_id])
        assert r["context_tail"] == full[-160:]
        assert r["gold_in_context"] == (bool(gold) and gold in full), r
        # 出现次数/最后位置/到预测位的距离必须与完整窗口一致：报告里的
        # "gold 距预测位 N tokens" 直接决定该看检索失败还是复制失败。
        pos = [i for i, t in enumerate(toks) if t == gold_id]
        assert r["gold_occurrences"] == len(pos), r
        if pos:
            assert r["gold_last_pos"] == pos[-1], r
            assert r["gold_distance_tokens"] == 64 - pos[-1] - 1, r
            assert r["gold_context_excerpt"], r
        else:
            assert r["gold_last_pos"] is None, r
            assert r["gold_distance_tokens"] is None, r
            assert r["gold_context_excerpt"] is None, r
    # 构造一个 gold 只出现在窗口早段、尾部 160 字符看不到的样本：
    # 摘录必须能让人复核“上下文里到底有没有证据”。
    trend = np.tile(np.arange(len(tok.itos), dtype=np.uint16), 4)[:64]
    val2 = np.concatenate([trend, np.array([0], dtype=np.uint16),
                           trend, np.array([0], dtype=np.uint16)])
    rows2 = collect_worst_windows(model, val2, ctx=64, device="cpu", tok=tok,
                                  n_worst=2, max_windows=2)
    for r in rows2:
        if r["gold_in_context"]:
            assert r["gold_context_excerpt"], r
            assert r["gold_distance_tokens"] >= 0, r
    print("  [ok] 最差窗口 gold_in_context 按完整窗口判定；"
          "位置/距离/摘录可用于人工复核")


def test_checkpoint_selection_info_records_step_and_rejects_garbage():
    """选型步/计划步数必须可审计；旧/坏 checkpoint 字段按缺失处理。"""
    info = checkpoint_selection_info(
        {"step": 250, "steps_planned": 1000, "val_loss": 3.25})
    assert info["ckpt_step"] == 250
    assert info["steps_planned"] == 1000
    assert math.isclose(info["selection_step_fraction"], 0.25)
    assert info["selection_step_exceeds_planned"] is False
    assert math.isclose(info["ckpt_val_loss"], 3.25)
    # 缺 steps_planned 时回退到训练配置的 steps（仅当它是正整数）
    info = checkpoint_selection_info({"step": 40, "val_loss": 2.0}, 100)
    assert info["steps_planned"] == 100
    assert math.isclose(info["selection_step_fraction"], 0.4)
    # bool/小数/非有限值不能冒充整数步数；丢失字段返回 None 而非编造
    for bad in (True, 2.5, float("inf"), "x"):
        info = checkpoint_selection_info(
            {"step": bad, "steps_planned": bad, "val_loss": bad})
        assert info["ckpt_step"] is None
        assert info["steps_planned"] is None
        assert info["selection_step_fraction"] is None
        if bad is True or bad == float("inf") or isinstance(bad, str):
            assert info["ckpt_val_loss"] is None
    # val_loss 本身允许小数；2.5 不是“非法步数”
    assert math.isclose(checkpoint_selection_info(
        {"step": 2.5, "val_loss": 2.5})["ckpt_val_loss"], 2.5)
    # 选型步超过计划预算要显式标记（checkpoint/配置不一致的可审计信号）
    info = checkpoint_selection_info({"step": 120, "steps_planned": 100})
    assert info["selection_step_exceeds_planned"] is True
    assert checkpoint_selection_info(None)["ckpt_step"] is None
    print("  [ok] checkpoint 选型步元数据（缺失/非法字段不伪造）")


def test_json_safe_eval_metrics_handles_nonfinite():
    """非有限验证损失不得写成 NaN 字面量，也不得伪装成 exp(20) 截断值。

    math.exp(min(20, float("nan"))) 恰为 exp(20)（min 对 NaN 返回第一个参数），
    而 bool(nan > 20) 为 False：不做显式处理会把失效 checkpoint 报告为
    "正常但很大"的有限 PPL，且 json.dump 产出非法 JSON 字面量。
    """
    out = json_safe_eval_metrics({"val_loss": float("nan"),
                                  "gate_mean": float("nan")})
    assert out == {"val_ppl": None, "val_loss": None,
                   "val_ppl_clamped": None, "gate_mean": None}
    # allow_nan=False 不抛异常：产物中不存在 NaN/Infinity 字面量
    json.dumps(out, allow_nan=False)
    for bad in (float("inf"), float("-inf")):
        o = json_safe_eval_metrics({"val_loss": bad, "gate_mean": 0.5})
        assert o["val_ppl"] is None and o["val_loss"] is None
        assert o["val_ppl_clamped"] is None and o["gate_mean"] == 0.5
    # 有限值沿用 train.py 的 exp(min(20, loss)) / >20 截断口径
    o = json_safe_eval_metrics({"val_loss": 2.5, "gate_mean": None})
    assert o["val_ppl"] == math.exp(2.5) and o["val_ppl_clamped"] is False
    o = json_safe_eval_metrics({"val_loss": 21.0, "gate_mean": 0.9})
    assert o["val_ppl"] == math.exp(20) and o["val_ppl_clamped"] is True
    assert o["val_loss"] == 21.0 and o["gate_mean"] == 0.9
    print("  [ok] 非有限评测指标归为 null（JSON 安全、不伪装截断值）")


def test_collect_worst_windows_small_vocab_clamps_topk():
    """词表 <5 时 top-5 转储按实际词表截断，不能越界崩溃。"""
    torch.manual_seed(0)
    tok = CharTokenizer(list("abc"))
    model = GPT(vocab_size=3, block_size=16, d_model=8, n_layers=1,
                n_heads=1, variant="standard")
    model.eval()
    val = np.zeros(16 * 33, dtype=np.uint16)
    rows = collect_worst_windows(model, val, ctx=16, device="cpu", tok=tok,
                                 n_worst=2, max_windows=4)
    assert len(rows) == 2
    assert all(1 <= len(r["pred_top5"]) <= 3 for r in rows)
    print("  [ok] 小词表最差窗口：top-k 自动截断")


def test_json_safe_eval_metrics_tolerates_malformed_fields():
    """缺失/非数值 val_loss 与 gate_mean 归为 null，不能抛 TypeError。"""
    empty = {"val_ppl": None, "val_loss": None,
             "val_ppl_clamped": None, "gate_mean": None}
    assert json_safe_eval_metrics({}) == empty
    assert json_safe_eval_metrics(None) == empty
    assert json_safe_eval_metrics({"val_loss": "bad",
                                   "gate_mean": "bad"}) == empty
    out = json_safe_eval_metrics({"val_loss": 2.0, "gate_mean": [1]})
    assert out["val_ppl"] == math.exp(2.0) and out["gate_mean"] is None
    out = json_safe_eval_metrics({"val_loss": 2.0, "gate_mean": "0.5"})
    assert out["gate_mean"] == 0.5
    print("  [ok] 评测指标：畸形 val_loss/gate_mean 安全降级")


def test_repeated_test_checkpoint_gate():
    """同一 checkpoint 重复在 test 上评测必须可被识别（默认拒绝的依据）。"""
    summary = {"evaluations": 2, "distinct_checkpoints": 1,
               "checkpoints_evaluated_multiple_times": ["abc123def4567890"]}
    assert repeated_test_checkpoint(summary, "abc123def4567890" + "0" * 48)
    # 前 16 位不同 → 不判重复
    assert not repeated_test_checkpoint(summary, "ffffffffffffffff" + "0" * 48)
    # 空账本 / 空指纹 / 无重复字段都不得误判
    assert not repeated_test_checkpoint(None, "a" * 64)
    assert not repeated_test_checkpoint({}, "a" * 64)
    assert not repeated_test_checkpoint(summary, None)
    assert not repeated_test_checkpoint(
        {"checkpoints_evaluated_multiple_times": []}, "a" * 64)

    # 精确到 test 数据版本：旧版本重复不阻断新版本评测。
    versioned = {
        "checkpoints_evaluated_multiple_times": ["abc123def4567890"],
        "repeated_evaluations": [
            {"ckpt_prefix": "abc123def4567890", "test_data_fingerprint": "v1",
             "evaluations": 2}],
    }
    ckpt = "abc123def4567890" + "0" * 48
    assert repeated_test_checkpoint(versioned, ckpt, "v1")
    assert not repeated_test_checkpoint(versioned, ckpt, "v2")
    assert repeated_test_checkpoint(versioned, ckpt)  # 旧调用不带版本：保守判定
    print("  [ok] 重复 test 评测检测：16 位指纹 + test 数据版本，空值不误判")


def test_evaluate_runs_data_protocol_preflight_before_loading_model():
    """独立评测也不能在 data_report 标记泄露的数据上产出正式指标。"""
    from lega.evaluate import main as evaluate_main
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "data_report.json"), "w",
                  encoding="utf-8") as f:
            json.dump({"alerts": [{"level": "warning",
                                   "code": "split_shuffled",
                                   "message": "打乱"}]}, f)
        old_argv = sys.argv
        sys.argv = ["evaluate", "--ckpt", os.path.join(d, "missing.pt"),
                    "--data_dir", d, "--out", os.path.join(d, "eval.json"),
                    "--device", "cpu"]
        try:
            evaluate_main()
            raise AssertionError("评测入口应执行数据协议闸门")
        except SystemExit as e:
            assert "数据协议前置检查失败" in str(e)
        finally:
            sys.argv = old_argv
    print("  [ok] 独立评测入口执行数据协议闸门（严重告警默认中止）")


def test_worst_windows_allowed_seals_test():
    """test 上转储最差窗口默认禁止，显式放行才允许；val 不受限。"""
    assert worst_windows_allowed("val", False)
    assert worst_windows_allowed("val", True)
    assert not worst_windows_allowed("test", False)
    assert worst_windows_allowed("test", True)
    print("  [ok] test 最差窗口默认尘封，val 不受限")


def test_evaluate_result_test_access_includes_worst_dump(monkeypatch):
    """结果 JSON 的 test 访问摘要必须包含同次评测的最差窗口转储记录。"""
    from lega.data import prepare_bins
    from lega.datacheck import read_test_access_log
    from lega.evaluate import main as evaluate_main
    from lega.model import GPT, save_checkpoint
    from lega.repro import sha256_file

    torch.manual_seed(0)
    with tempfile.TemporaryDirectory() as d:
        data_dir = os.path.join(d, "data")
        os.makedirs(data_dir)
        with open(os.path.join(data_dir, "train.txt"), "w", encoding="utf-8") as f:
            f.write("the quick brown fox jumps over the lazy dog. " * 4)
        with open(os.path.join(data_dir, "val.txt"), "w", encoding="utf-8") as f:
            f.write("a quick brown fox jumps over the lazy dog. " * 4)
        with open(os.path.join(data_dir, "test.txt"), "w", encoding="utf-8") as f:
            f.write("the quick red fox leaps over the lazy dog. " * 4)
        meta = prepare_bins(data_dir, kind="char", force=True)
        ckpt = os.path.join(d, "run", "model.pt")
        model = GPT(vocab_size=meta["vocab_size"], block_size=16, d_model=16,
                    n_layers=1, n_heads=2, variant="standard")
        save_checkpoint(ckpt, model, extra={"variant": "standard", "seed": 0})
        out = os.path.join(d, "eval.json")
        monkeypatch.setattr(sys, "argv", [
            "evaluate", "--ckpt", ckpt, "--data_dir", data_dir,
            "--out", out, "--device", "cpu", "--worst_n", "1",
            "--worst_on_test", "--allow_repeat_test"])
        evaluate_main()
        with open(out, encoding="utf-8") as f:
            res = json.load(f)
        ledger = read_test_access_log(data_dir)
        assert res["test_access"]["evaluations"] == 2
        assert res["test_access"]["distinct_checkpoints"] == 1
        assert sha256_file(ckpt)[:16] in \
            res["test_access"]["checkpoints_evaluated_multiple_times"]
        assert len(ledger) == 2
        assert [r.get("reason") for r in ledger if r.get("reason")] == \
            ["worst_window_dump_on_test"]
        test_fp = sha256_file(os.path.join(data_dir, "test.bin"))
        assert res["test_access"]["distinct_test_data_versions"] == 1
        assert [r.get("test_data_fingerprint") for r in ledger] == \
            [test_fp, test_fp]
    print("  [ok] test 访问摘要包含 worst-window 转储（不瞒报第二次读取）")


def test_evaluate_warns_when_data_differs_from_training_manifest(capsys):
    """用另一份数据评测 checkpoint 时必须显式告警，而非静默继承训练指纹。"""
    from lega.data import prepare_bins
    from lega.evaluate import main as evaluate_main
    from lega.model import save_checkpoint

    torch.manual_seed(0)
    with tempfile.TemporaryDirectory() as d:
        data_dir = os.path.join(d, "data")
        os.makedirs(data_dir)
        with open(os.path.join(data_dir, "train.txt"), "w",
                  encoding="utf-8") as f:
            f.write("the quick brown fox jumps over the lazy dog. " * 6)
        with open(os.path.join(data_dir, "val.txt"), "w",
                  encoding="utf-8") as f:
            f.write("a slow green turtle sleeps under the old oak tree. " * 6)
        meta = prepare_bins(data_dir, kind="char", force=True)
        run_dir = os.path.join(d, "run")
        ckpt = os.path.join(run_dir, "model.pt")
        model = GPT(vocab_size=meta["vocab_size"], block_size=16, d_model=16,
                    n_layers=1, n_heads=2, variant="standard")
        save_checkpoint(ckpt, model, extra={"variant": "standard", "seed": 0})
        with open(os.path.join(run_dir, "run_manifest.json"), "w",
                  encoding="utf-8") as f:
            json.dump({"data": {"digest": "0" * 64}}, f)
        out = os.path.join(d, "eval.json")
        old_argv = sys.argv
        sys.argv = ["evaluate", "--ckpt", ckpt, "--data_dir", data_dir,
                    "--out", out, "--device", "cpu", "--split", "val"]
        try:
            evaluate_main()
        finally:
            sys.argv = old_argv
        text = capsys.readouterr().out
        assert "数据指纹与 checkpoint 训练清单不一致" in text
        with open(out, encoding="utf-8") as f:
            res = json.load(f)
        assert res["data_fingerprint_matches_training"] is False
        assert res["eval_data_fingerprint"] == res["data_fingerprint"]
        assert res["train_data_fingerprint"] == "0" * 64
    print("  [ok] 评测数据与训练清单不一致时显式告警并分别落盘两侧指纹")
