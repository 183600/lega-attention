"""NaN 系统排查路径测试：数据 → 代码定位 → 混合精度 → 梯度/学习率 的固定顺序。"""
import json
import os
import shutil
import sys
import tempfile

import torch

from lega.data import prepare_bins
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


def test_checklist_labels_interval_candidates_and_first_bad_step():
    """检测点在 log 边界、失败步可能在区间中段：报告必须给出首个失败步，
    并把每条数据检查定位到具体步（而不是笼统的“失败步批”）。"""
    m = _model()
    rep = run_nan_checklist(m, _batches(), "cpu", 40, "train_loss", float("nan"),
                            lr=1e-3, grad_norm_history=[], amp_dtype=None,
                            vocab_size=64,
                            consumed_batches=_batches(bad_ids=True) * 2,
                            consumed_step_labels=[36, 40],
                            first_bad_step=36, interval=(36, 40))
    findings = rep["checklist_data"]["findings"]
    assert findings[0].startswith("失败区间批0@step36"), findings
    assert findings[2].startswith("失败区间批1@step40"), findings
    assert rep["first_bad_step"] == 36
    assert rep["interval"] == [36, 40]
    assert rep["checklist_data"]["first_bad_step"] == 36
    assert "首个非有限损失步=36" in rep["checklist_data"]["note"]
    # 旧调用（无标签）保持“失败步批”前缀，避免破坏既有报告口径
    rep2 = run_nan_checklist(m, _batches(), "cpu", 40, "train_loss",
                             float("nan"), lr=1e-3, grad_norm_history=[],
                             amp_dtype=None, vocab_size=64,
                             consumed_batches=_batches())
    assert rep2["checklist_data"]["findings"][0].startswith("失败步批0")
    assert rep2["first_bad_step"] is None and rep2["interval"] is None
    print("  [ok] 排查步骤 1（数据）：区间候选批次带步标签 + 首个失败步")


def test_activation_audit_iterates_interval_candidates():
    """激活定位必须逐批检查区间候选：NaN 出现在后一批时不能只审计第一批。"""
    m = _model()
    # 只让较长序列（第二批）在 blocks.1.ln1 产生 NaN：首批必须有限
    m.blocks[1].ln1.forward = lambda x: x * (
        float("nan") if x.shape[1] > 16 else 1.0)
    b1 = (torch.randint(0, 64, (2, 16)), torch.randint(0, 64, (2, 16)))
    b2 = (torch.randint(0, 64, (2, 32)), torch.randint(0, 64, (2, 32)))
    rep = audit_activations(m, [b1, b2], "cpu")
    assert rep["n_batches_checked"] == 2, rep
    assert rep["checked_batch_index"] == 1, rep
    assert rep["origin"]["module"] == "blocks.1.ln1", rep["origin"]
    assert any("逐批检查了 2 个候选批次" in f for f in rep["findings"])
    # 全部有限时也要遍历所有候选，并据全部候选下“损失环节”结论
    m2 = _model()
    rep2 = audit_activations(m2, [b1, b2], "cpu")
    assert rep2["n_batches_checked"] == 2 and rep2["origin"] is None
    assert any("已检查 2 批" in f for f in rep2["findings"])
    assert not any("命中第" in f for f in rep2["findings"]), rep2["findings"]
    print("  [ok] 激活审计：逐批检查区间候选并命中失败批次")


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


def test_activation_audit_fp32_recheck_isolates_precision():
    """amp 路径非有限时用同批次 fp32 复测：fp32 有限→判混合精度数值问题；
    fp32 同样非有限→排除精度因素。"""
    import lega.train as tr
    m = _model()
    calls = []

    def fake_forward(model, x, device, amp_dtype, cap=60):
        calls.append(amp_dtype)
        if amp_dtype is not None:
            return ([{"seq": 0, "module": "blocks.0.ln1", "output_nan": 4,
                      "output_inf": 0, "input_nonfinite": 0}], None)
        return ([], None)

    orig = tr._hooked_forward
    tr._hooked_forward = fake_forward
    try:
        rep = tr.audit_activations(m, _batches(), "cuda",
                                   amp_dtype=torch.bfloat16)
        calls.clear()  # 只考察 CPU 审计的调用次数
        rep2 = tr.audit_activations(m, _batches(), "cpu",
                                    amp_dtype=torch.bfloat16)
    finally:
        tr._hooked_forward = orig
    # 第一次：amp 路径 + fp32 隔离复测各跑一次
    assert rep["precision_recheck"]["ran"]
    assert rep["precision_recheck"]["fp32_nonfinite_records"] == 0
    assert "混合精度数值问题" in rep["precision_recheck"]["verdict"], rep
    assert any("混合精度" in f for f in rep["findings"])
    # CPU 上 autocast 未生效（无混合精度可查）：只做主审计、不触发复测
    assert len(calls) == 1 and rep2["precision_recheck"] is None

    # fp32 复测同样非有限 → 明确排除混合精度因素
    def fake_forward2(model, x, device, amp_dtype, cap=60):
        return ([{"seq": 0, "module": "blocks.0.ln1", "output_nan": 1,
                  "output_inf": 0, "input_nonfinite": 0}], None)

    tr._hooked_forward = fake_forward2
    try:
        rep3 = tr.audit_activations(m, _batches(), "cuda",
                                    amp_dtype=torch.float16)
    finally:
        tr._hooked_forward = orig
    assert "不是混合精度" in rep3["precision_recheck"]["verdict"], rep3
    # 完整排查入口：精度步骤的 note 必须带上复测结论
    tr._hooked_forward = fake_forward2
    try:
        full = run_nan_checklist(m, _batches(), "cuda", 7, "train_loss",
                                 float("nan"), lr=1e-3, grad_norm_history=[],
                                 amp_dtype=torch.float16, vocab_size=64)
    finally:
        tr._hooked_forward = orig
    assert "不是混合精度" in full["checklist_precision"]["note"]
    assert any("不是混合精度" in a for a in full["suggested_actions"]), \
        full["suggested_actions"]
    print("  [ok] 排查步骤 3（精度）：fp32 隔离复测（限制在 amp 实际生效时）")


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


def test_checklist_reports_failure_lr_and_schedule_peak_separately():
    """失败步可能仍在 warmup 中：失败时 lr 与调度峰值必须分别显示，
    不能把同一个数值打印两遍。"""
    m = _model()
    rep = run_nan_checklist(m, _batches(), "cpu", 7, "train_loss", float("nan"),
                            lr=5e-4, grad_norm_history=[1.0],
                            amp_dtype=None, vocab_size=64, peak_lr=2e-3)
    findings = " ".join(rep["checklist_optimization"]["findings"])
    assert "5.00e-04" in findings and "2.00e-03" in findings, findings
    print("  [ok] 排查步骤 4（优化）：失败时 lr 与调度峰值分别显示")


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


def test_checklist_reports_nonfinite_grad_and_optimizer_state():
    """NaN 排查不能只看参数：梯度反向溢出（早于参数变 NaN）与 Adam 状态
    溢出（只重置参数无效）都必须被定位到具体张量。"""
    m = _model()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    name, param = next(iter(m.named_parameters()))
    param.grad = torch.full_like(param, float("nan"))
    opt.state[param]["exp_avg"] = torch.full_like(param, float("inf"))
    rep = run_nan_checklist(m, _batches(), "cpu", 7, "train_loss",
                            float("nan"), lr=1e-3, grad_norm_history=[1.0],
                            amp_dtype=None, vocab_size=64, optimizer=opt)
    assert rep["checklist_params"]["nonfinite_grads"], rep["checklist_params"]
    assert any(name in g for g in rep["checklist_params"]["nonfinite_grads"])
    osec = rep["checklist_optimization"]
    assert osec["nonfinite_optimizer_state"], osec
    assert any(name in t for t in osec["nonfinite_optimizer_state"])
    findings = " ".join(osec["findings"])
    assert "优化器状态" in findings and "重建" in findings, findings
    assert "参数梯度" in findings, findings
    # 无优化器时字段仍存在且为空（旧调用兼容）
    rep2 = run_nan_checklist(m, _batches(), "cpu", 7, "train_loss",
                             float("nan"), lr=1e-3, grad_norm_history=[],
                             amp_dtype=None, vocab_size=64)
    assert rep2["checklist_optimization"]["nonfinite_optimizer_state"] == []
    print("  [ok] 排查步骤 4（优化）：非有限梯度与 Adam 状态定位")


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


def test_val_audit_batch_uses_real_validation_window():
    """val_loss 非有限时，排查输入必须是真实验证窗口而非训练批次（回退仅在取窗失败时）。"""
    import numpy as np
    from lega.train import val_audit_batch

    val = np.arange(0, 200, dtype=np.int64)
    fb = _batches()
    batch = val_audit_batch(val, ctx=16, iters=2, device="cpu", fallback=fb)
    assert len(batch) == 1 and batch[0] is not fb[0]
    x, y = batch[0]
    assert x.shape == (1, 16) and y.shape == (1, 16)
    assert torch.equal(x[:, 1:], y[:, :-1]), "y 必须是 x 的右移一位（与训练口径一致）"
    # 验证集过短组不成窗口 → 回退到训练批次而不是崩溃
    short = np.arange(0, 10, dtype=np.int64)
    batch2 = val_audit_batch(short, ctx=16, iters=2, device="cpu", fallback=fb)
    assert len(batch2) == 1 and batch2[0][0] is fb[0][0]
    print("  [ok] val_loss 排查使用真实验证窗口（取窗失败才回退）")


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


def test_train_nan_report_uses_first_bad_step_and_interval_candidates():
    """端到端：第 3 个训练步注入 NaN 损失，log 点在第 4 步才检测到。

    报告必须指出 first_bad_step=3，并检查区间 [3,4] 的候选批次，而不是
    只检查检测点时的“最后一步”批次（那会漏掉真正的罪魁数据）。
    """
    import lega.model as model_mod
    import lega.train as train_mod

    cfg = {
        "d_model": 32, "n_layers": 2, "n_heads": 2, "ffn_mult": 4,
        "block_size": 32, "dropout": 0.0, "attn_dropout": 0.0,
        "gate_bias_init": 2.0, "use_sdpa_standard": True,
        "batch_size": 2, "grad_accum": 1, "steps": 8, "lr": 1e-3,
        "warmup": 1, "weight_decay": 0.0, "grad_clip": 1.0,
        "train_ctx": 32, "amp": False, "seed": 1,
        "log_every": 2, "eval_every": 4, "eval_iters": 2,
    }
    with tempfile.TemporaryDirectory() as tmp:
        data_dir = os.path.join(tmp, "data")
        os.makedirs(data_dir)
        base = "the quick brown fox jumps over the lazy dog. "
        with open(os.path.join(data_dir, "train.txt"), "w",
                  encoding="utf-8") as f:
            f.write(base * 60)
        with open(os.path.join(data_dir, "val.txt"), "w",
                  encoding="utf-8") as f:
            f.write(base * 20)
        prepare_bins(data_dir, kind="char", force=True)
        cfg_path = os.path.join(tmp, "cfg.json")
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f)
        out_dir = os.path.join(tmp, "run")

        orig_forward = model_mod.GPT.forward
        calls = {"n": 0}

        def poisoned_forward(self, idx, targets=None, *a, **kw):
            out = orig_forward(self, idx, targets=targets, *a, **kw)
            # 只污染训练前向（evaluate_val 会切 eval()），第 3 步起 loss=NaN
            if self.training and targets is not None \
                    and out.get("loss") is not None:
                calls["n"] += 1
                if calls["n"] >= 3:
                    out["loss"] = out["loss"] * float("nan")
            return out

        model_mod.GPT.forward = poisoned_forward
        old_argv = sys.argv
        sys.argv = ["lega.train", "--config", cfg_path,
                    "--variant", "standard", "--data_dir", data_dir,
                    "--out_dir", out_dir, "--device", "cpu", "--seed", "1"]
        try:
            try:
                train_mod.main()
                raise AssertionError("NaN 必须中止训练")
            except SystemExit as e:
                assert "nan_report.json" in str(e)
        finally:
            sys.argv = old_argv
            model_mod.GPT.forward = orig_forward

        rep = json.load(open(os.path.join(out_dir, "nan_report.json"),
                             encoding="utf-8"))
        assert rep["where"] == "train_loss", rep
        assert rep["step"] == 4, rep
        assert rep["first_bad_step"] == 3, rep
        assert rep["interval"] == [3, 4], rep
        data = rep["checklist_data"]
        assert data["n_consumed_batches"] == 2, data
        labels = [f for f in data["findings"] if f.startswith("失败区间批")]
        assert any("@step3 " in f for f in labels), labels
        assert any("@step4 " in f for f in labels), labels
        act = rep["checklist_activations"]
        assert act["n_batches_checked"] >= 1, act
    print("  [ok] train 端到端：NaN 报告定位首个失败步并检查区间候选批次")
