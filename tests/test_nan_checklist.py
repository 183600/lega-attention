"""NaN 系统排查路径测试：数据 → 代码定位 → 混合精度 → 梯度/学习率 的固定顺序。"""
import json
import os
import shutil
import tempfile

import torch

from lega.model import GPT
from lega.train import _abort_on_nan, run_nan_checklist


def _model():
    torch.manual_seed(0)
    return GPT(vocab_size=64, block_size=64, d_model=32, n_layers=2, n_heads=2,
               ffn_mult=4, variant="lega")


def _batches(vocab=64, bs=2, ctx=16, bad_ids=False):
    x = torch.randint(0, vocab, (bs, ctx))
    y = torch.randint(0, vocab, (bs, ctx))
    if bad_ids:
        y[0, 0] = vocab + 5      # 越界 id
    return [(x, y)]


def test_checklist_data_step_finds_out_of_range_ids():
    m = _model()
    rep = run_nan_checklist(m, _batches(bad_ids=True), "cpu", 42, "train_loss",
                            float("nan"), lr=1e-3, grad_norm_history=[1.0, 1.1],
                            amp_dtype=None, vocab_size=64)
    findings = " ".join(rep["checklist_data"]["findings"])
    assert "越界" in findings, "越界 id 必须在第一步（数据）被发现"
    # 合法整数批不应被误报
    rep2 = run_nan_checklist(m, _batches(), "cpu", 42, "train_loss",
                             float("nan"), lr=1e-3, grad_norm_history=[],
                             amp_dtype=None, vocab_size=64)
    assert not any("越界" in f for f in rep2["checklist_data"]["findings"])
    print("  [ok] 排查步骤 1（数据）：id 越界检查")


def test_checklist_params_step_localizes_nan_layer():
    m = _model()
    with torch.no_grad():
        m.blocks[1].attn.gate_w.zero_()
        m.blocks[1].attn.gate_w[0, 0] = float("nan")
    rep = run_nan_checklist(m, _batches(), "cpu", 7, "train_loss", float("nan"),
                            lr=1e-3, grad_norm_history=[], amp_dtype=None,
                            vocab_size=64)
    tensors = rep["checklist_params"]["nonfinite_tensors"]
    assert any("blocks.1.attn.gate_w" in t for t in tensors), tensors
    assert any("log(0)" in t or "log" in rep["checklist_params"]["note"]
               for t in tensors), "参数定位应附带代码排查指引"
    print("  [ok] 排查步骤 2（代码）：NaN 参数定位到具体层 + log(0)/softmax 指引")


def test_checklist_precision_step_reports_amp_dtype():
    m = _model()
    for amp, expect in ((None, "排除混合精度"), (torch.bfloat16, "bf16"),
                        (torch.float16, "fp16")):
        rep = run_nan_checklist(m, _batches(), "cpu", 7, "train_loss",
                                float("nan"), lr=1e-3, grad_norm_history=[],
                                amp_dtype=amp, vocab_size=64)
        assert expect in rep["checklist_precision"]["note"], rep["checklist_precision"]
    print("  [ok] 排查步骤 3（精度）：bf16/fp16/无 autocast 的差异化结论")


def test_checklist_optimization_step_detects_grad_explosion():
    m = _model()
    hist = [1.0, 1.2, 0.9, 1.1, 800.0]
    rep = run_nan_checklist(m, _batches(), "cpu", 7, "grad_norm", float("inf"),
                            lr=3e-3, grad_norm_history=hist, amp_dtype=None,
                            vocab_size=64)
    findings = " ".join(rep["checklist_optimization"]["findings"])
    assert "梯度爆炸" in findings, "pre-clip 范数尖峰应被判定为疑似梯度爆炸"
    assert rep["checklist_optimization"]["grad_norm_history_tail"] == hist[-20:]
    assert any("学习率" in f for f in rep["checklist_optimization"]["findings"])
    # 无尖峰的历史不应误报爆炸
    rep2 = run_nan_checklist(m, _batches(), "cpu", 7, "train_loss", float("nan"),
                             lr=1e-3, grad_norm_history=[1.0, 1.1, 0.95, 1.05],
                             amp_dtype=None, vocab_size=64)
    assert "未见爆炸性尖峰" in " ".join(rep2["checklist_optimization"]["findings"])
    print("  [ok] 排查步骤 4（优化）：梯度爆炸尖峰检测与学习率报告")


def test_checklist_nonfinite_grad_norm_trigger_reports_explosion():
    """grad_norm 触发的排查：当前范数非有限而历史只保留有限值，
    必须用触发值显式报告“已爆炸”，不能退化成“未见爆炸性尖峰”。"""
    m = _model()
    rep = run_nan_checklist(m, _batches(), "cpu", 7, "grad_norm", float("inf"),
                            lr=3e-3, grad_norm_history=[1.0, 1.1, 0.95, 1.05],
                            amp_dtype=None, vocab_size=64)
    findings = " ".join(rep["checklist_optimization"]["findings"])
    assert "梯度已爆炸" in findings, findings
    # train_loss 触发（非 grad_norm）时不得误报爆炸
    rep2 = run_nan_checklist(m, _batches(), "cpu", 7, "train_loss", float("nan"),
                             lr=1e-3, grad_norm_history=[1.0, 1.1, 0.95, 1.05],
                             amp_dtype=None, vocab_size=64)
    assert "梯度已爆炸" not in " ".join(rep2["checklist_optimization"]["findings"])
    print("  [ok] 排查步骤 4（优化）：grad_norm 非有限的触发值本身被作为爆炸证据")


def test_checklist_action_order_is_fixed():
    m = _model()
    rep = run_nan_checklist(m, _batches(), "cpu", 7, "train_loss", float("nan"),
                            lr=1e-3, grad_norm_history=[1.0, 2.0],
                            amp_dtype=torch.bfloat16, vocab_size=64)
    acts = rep["suggested_actions"]
    assert len(acts) == 4
    assert acts[0].startswith("1) 数据") and "2) 代码" in acts[1] \
        and "3) 精度" in acts[2] and "4) 优化" in acts[3]
    print("  [ok] 建议动作按固定排查顺序输出")


def test_abort_writes_report_and_exits():
    m = _model()
    rep = run_nan_checklist(m, _batches(), "cpu", 123, "train_loss", float("nan"),
                            lr=1e-3, grad_norm_history=[], amp_dtype=None,
                            vocab_size=64)
    tmp = tempfile.mkdtemp()
    try:
        try:
            _abort_on_nan(rep, tmp)
            raise AssertionError("NaN 必须中止训练")
        except SystemExit as e:
            assert "nan_report.json" in str(e), "中止信息应指向排查报告"
        path = os.path.join(tmp, "nan_report.json")
        assert os.path.exists(path), "中止前必须先落盘排查报告"
        saved = json.load(open(path, encoding="utf-8"))
        assert saved["step"] == 123 and saved["where"] == "train_loss"
        assert "suggested_actions" in saved and len(saved["suggested_actions"]) == 4
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  [ok] NaN 中止：先落盘报告再 SystemExit")
