"""NaN 系统排查路径测试：数据 → 代码定位 → 混合精度 → 梯度/学习率 的固定顺序。"""
import json
import os
import shutil
import tempfile

import torch

from lega.model import GPT
from lega.train import _abort_on_nan, audit_activations, run_nan_checklist


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


def test_checklist_prefers_consumed_failing_batch():
    """预取会覆盖 batches：失败步批次必须显式传入并优先检查，否则漏掉罪魁数据。"""
    m = _model()
    rep = run_nan_checklist(m, _batches(), "cpu", 42, "train_loss", float("nan"),
                            lr=1e-3, grad_norm_history=[], amp_dtype=None,
                            vocab_size=64, consumed_batches=_batches(bad_ids=True))
    findings = rep["checklist_data"]["findings"]
    assert findings and findings[0].startswith("失败步批"), findings
    assert any("越界" in f for f in findings), "只查待训练批次会漏掉失败步的越界 id"
    assert rep["checklist_data"]["n_consumed_batches"] == 1
    print("  [ok] 排查步骤 1（数据）：失败步批次优先检查并定位越界 id")


def test_checklist_activation_audit_gets_consumed_batches_first():
    """激活定位也应以失败步批次为主输入（消耗完的批次才是 NaN 现场）。"""
    import lega.train as tr
    captured = {}
    orig = tr.audit_activations

    def fake(model, batches, device, amp_dtype=None):
        captured["batches"] = list(batches)
        return {"checked": True, "origin": None, "records": [],
                "first_nonfinite": None, "forward_error": None,
                "n_modules_checked": 0, "findings": []}
    tr.audit_activations = fake
    try:
        consumed, pending = _batches(), _batches()
        run_nan_checklist(_model(), pending, "cpu", 1, "train_loss",
                          float("nan"), lr=1e-3, grad_norm_history=[],
                          amp_dtype=None, vocab_size=64,
                          consumed_batches=consumed)
        assert captured["batches"][0] is consumed[0], "优先用失败步批次"
        assert captured["batches"][1] is pending[0], "待训练批次保留作补充"
    finally:
        tr.audit_activations = orig
    print("  [ok] 排查步骤 2（代码）：激活定位优先以失败步批次为输入")


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


def test_activation_audit_finds_origin_on_finite_inputs():
    """主动检查：有限参数下由某个算子产生的 NaN，必须定位到该模块（origin）。"""
    m = _model()
    m.ln_f.forward = lambda x: x * float("nan")   # 有限输入 → 非有限输出
    rep = audit_activations(m, _batches(), "cpu")
    assert rep["origin"] is not None, rep
    assert rep["origin"]["module"] == "ln_f", rep["origin"]
    assert rep["origin"]["input_nonfinite"] == 0
    assert rep["origin"]["output_nan"] > 0
    assert any("ln_f" in f for f in rep["findings"])
    # 通过完整排查入口也应包含该定位结果
    full = run_nan_checklist(m, _batches(), "cpu", 7, "train_loss", float("nan"),
                             lr=1e-3, grad_norm_history=[], amp_dtype=None,
                             vocab_size=64)
    assert full["checklist_activations"]["origin"]["module"] == "ln_f"
    print("  [ok] 激活审计：定位首个“输入有限、输出非有限”的模块")


def test_activation_audit_distinguishes_propagation():
    """NaN 参数：origin 定位到注入 NaN 的叶子模块，下游模块标记为传播。"""
    m = _model()
    with torch.no_grad():
        m.blocks[1].ln1.weight[0] = float("nan")
    rep = audit_activations(m, _batches(), "cpu")
    assert rep["origin"]["module"] == "blocks.1.ln1", rep["origin"]
    downstream = [r for r in rep["records"] if r["seq"] > 0]
    assert any(r["input_nonfinite"] > 0 for r in downstream), rep["records"]
    # 健康模型：无任何非有限激活 → 结论应指向损失/聚合环节而非前向算子
    rep2 = audit_activations(_model(), _batches(), "cpu")
    assert rep2["origin"] is None and not rep2["records"]
    assert any("损失" in f for f in rep2["findings"]), rep2["findings"]
    print("  [ok] 激活审计：参数污染定位 + 传播链 + 全有限时指向损失环节")


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
