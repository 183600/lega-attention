"""模型级测试：KV 缓存一致性、贪心生成确定性、检查点保存与恢复。"""
import os
import tempfile

import numpy as np
import torch

from lega.attention import VARIANTS
from lega.model import GPT, load_model, save_checkpoint
from lega.train import _val_eval_indices, evaluate_val


def _tiny(variant):
    torch.manual_seed(0)
    return GPT(vocab_size=64, block_size=32, d_model=32, n_layers=2, n_heads=2,
               ffn_mult=4, variant=variant)


def test_kv_cache_matches_full_forward():
    """逐 token 增量解码（带缓存）与一次性全量前向的 logits 一致。"""
    for v in VARIANTS:
        model = _tiny(v)
        model.eval()
        idx = torch.randint(0, 64, (1, 12))
        with torch.no_grad():
            full = model(idx)["logits"]
            past = None
            inc = []
            for t in range(12):
                out = model(idx[:, t:t + 1], past_kv=past)
                past = out["past_kv"]
                inc.append(out["logits"])
            inc = torch.cat(inc, dim=1)
        assert torch.allclose(full, inc, atol=1e-4, rtol=1e-4), \
            f"{v} KV 缓存不一致, max diff={(full-inc).abs().max().item()}"
        print(f"  [ok] {v}: KV 缓存一致 (max diff {(full-inc).abs().max().item():.2e})")


def test_chunked_prefill_matches_full_forward():
    """分块预填充（past_kv 非空且 T>1，触发掩码缓存的 past_len>0 分支）与全量前向一致。"""
    for v in VARIANTS:
        model = _tiny(v)
        model.eval()
        idx = torch.randint(0, 64, (2, 15))
        with torch.no_grad():
            full = model(idx)["logits"]
            past = None
            inc = []
            for lo in (0, 5, 10):
                out = model(idx[:, lo:lo + 5], past_kv=past)
                past = out["past_kv"]
                inc.append(out["logits"])
            inc = torch.cat(inc, dim=1)
        assert torch.allclose(full, inc, atol=1e-4, rtol=1e-4), \
            f"{v} 分块预填充不一致, max diff={(full - inc).abs().max().item()}"
        print(f"  [ok] {v}: 分块预填充一致 (max diff {(full - inc).abs().max().item():.2e})")


def test_variants_share_base_weights_same_seed():
    """公平对比前提：同 seed 下所有变体的共享基础参数必须逐元素一致。"""
    def build(v):
        torch.manual_seed(123)
        return GPT(vocab_size=64, block_size=32, d_model=32, n_layers=2,
                   n_heads=2, ffn_mult=4, variant=v)

    ref = build("standard").state_dict()
    for v in VARIANTS:
        sd = build(v).state_dict()
        for k, p in ref.items():
            assert torch.equal(p, sd[k]), f"{v}:{k} 基础权重与 standard 不一致"


def test_greedy_generation_deterministic():
    model = _tiny("lega")
    model.eval()
    idx = torch.randint(0, 64, (1, 5))
    a = model.generate(idx, max_new_tokens=8)
    b = model.generate(idx, max_new_tokens=8)
    assert a.shape == (1, 13)
    assert torch.equal(a, b)


def test_last_logits_only_matches_full_logits():
    """快速路径必须与全量 logits 的最后位置完全一致。"""
    model = _tiny("lega")
    model.eval()
    idx = torch.randint(0, 64, (2, 9))
    with torch.no_grad():
        full = model(idx)["logits"]
        last = model(idx, last_logits_only=True)["logits"]
    assert last.shape == (2, 1, model.vocab_size)
    assert torch.allclose(last[:, -1], full[:, -1], atol=1e-6)


def test_generate_restores_training_mode_on_error():
    """生成中途失败（如超过 block_size）不能把模型永久留在 eval 模式。"""
    model = GPT(vocab_size=64, block_size=8, d_model=32, n_layers=1, n_heads=2,
                ffn_mult=4, variant="lega", dropout=0.2)
    model.train()
    idx = torch.randint(0, 64, (1, 5))
    try:
        model.generate(idx, max_new_tokens=10)
        raise AssertionError("预期超过 block_size 时 forward 抛出异常")
    except AssertionError as exc:
        if str(exc) == "预期超过 block_size 时 forward 抛出异常":
            raise
    assert model.training


def test_checkpoint_roundtrip():
    model = _tiny("lega")
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "model.pt")
        save_checkpoint(path, model, extra={"variant": "lega", "step": 3, "val_ppl": 1.23})
        m2, ck = load_model(path)
        assert ck["step"] == 3 and ck["variant"] == "lega"
        idx = torch.randint(0, 64, (1, 8))
        with torch.no_grad():
            o1 = model(idx)["logits"]
            o2 = m2(idx)["logits"]
        assert torch.allclose(o1, o2, atol=1e-6)


def test_gate_stats_shapes():
    """need_stats=True 时诊断统计形状正确（standard 也应返回集中度统计供分析用）。"""
    for v in ("lega", "qgate", "null", "standard"):
        model = _tiny(v)
        model.eval()
        idx = torch.randint(0, 64, (1, 10))
        with torch.no_grad():
            out = model(idx, need_stats=True)
        assert out["gate_stats"] is not None, v
        st = out["gate_stats"][0]
        assert st["max_p"].shape == (1, 2, 10)
        assert st["argmax"].shape == (1, 2, 10)
        assert (st["max_p"] <= 1.0 + 1e-5).all()
        if v in ("lega", "qgate"):
            assert st["g"] is not None
        else:
            assert st["g"] is None  # standard/null 无门控


def test_compiled_checkpoint_roundtrip():
    """torch.compile 模型的 state_dict 键带 _orig_mod. 前缀，保存时必须剥离，
    否则 load_model 按 GPT 原始键名加载会失败。"""
    model = _tiny("lega")
    try:
        compiled = torch.compile(model)
        compiled_state = compiled.state_dict()
    except Exception as exc:
        print(f"  [skip] torch.compile 不可用: {exc}")
        return
    assert any(k.startswith("_orig_mod.") for k in compiled_state), \
        "环境未产生 _orig_mod 前缀，测试无意义"
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "model.pt")
        save_checkpoint(path, compiled, extra={"variant": "lega"})
        m2, _ = load_model(path)
        idx = torch.randint(0, 64, (1, 8))
        with torch.no_grad():
            assert torch.allclose(model(idx)["logits"], m2(idx)["logits"], atol=1e-6)


def test_val_eval_indices_cover_dataset_without_repeats():
    """验证集评估窗口应覆盖首尾且不重复，避免循环复用造成 PPL 偏置。"""
    idx, n_windows = _val_eval_indices(1001, 10, 3)
    assert n_windows == 100
    assert idx.tolist() == [0, 49, 99]

    idx, n_windows = _val_eval_indices(1001, 10, 500)
    assert n_windows == 100
    assert len(idx) == 100
    assert len(set(idx.tolist())) == 100   # 请求窗口数超过可用数时不循环复用
    assert idx[0] == 0 and idx[-1] == 99

    idx, _ = _val_eval_indices(31, 10, 10)
    assert idx.tolist() == [0, 1, 2]       # 不足 iters 时评估全部可用窗口

    idx, _ = _val_eval_indices(1001, 10, 1)
    assert idx.tolist() == [49]            # 单窗估计取中间窗口，避免偏前


def test_evaluate_val_returns_coverage_and_gate_metrics():
    """val_loss 与 gate_mean 应来自同一组窗口，并记录实际覆盖范围。"""
    torch.manual_seed(0)
    model = _tiny("lega")
    val = (np.arange(100, dtype=np.int64) % 64).astype(np.uint16)
    metrics = evaluate_val(model, val, ctx=10, device="cpu", iters=50, amp=False)
    assert metrics["eval_iters_requested"] == 50
    assert metrics["eval_iters"] == 9      # (100-1)//10 个完整窗口，不循环复用
    assert metrics["val_windows"] == 9
    assert metrics["val_tokens_evaluated"] == 90
    assert metrics["val_loss"] == metrics["val_loss"]  # not NaN
    assert metrics["gate_mean"] is not None
    assert 0.0 < metrics["gate_mean"] < 1.0


def test_inference_mode_then_training_forward_backward():
    """inference_mode 前向产生的缓存常量不能污染随后同形状的训练图。"""
    model = _tiny("lega")
    idx = torch.randint(0, 64, (2, 12))
    with torch.inference_mode():
        model(idx)
    model.train()
    loss = model(idx, targets=idx)["loss"]
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())


def test_targets_ignore_index_forward_backward():
    """targets=-100 应遵守 PyTorch CE 约定：忽略损失且梯度为 0。"""
    torch.manual_seed(0)
    model = _tiny("standard")
    idx = torch.randint(0, 64, (2, 9))
    targets = idx.clone()
    targets[0, :4] = -100
    targets[1, 6:] = -100

    chunked = model(idx, targets=targets)["loss"]
    chunked.backward()
    grads_chunked = {n: p.grad.detach().clone()
                     for n, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)

    logits = model(idx)["logits"]
    ref = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)
    ref.backward()
    assert torch.allclose(chunked, ref, atol=1e-6, rtol=1e-6)
    for n, p in model.named_parameters():
        assert torch.allclose(grads_chunked[n], p.grad, atol=1e-5, rtol=1e-5), n

    model.zero_grad(set_to_none=True)
    ignored = model(idx, targets=torch.full_like(idx, -100))["loss"]
    assert ignored.item() == 0.0
    ignored.backward()
    assert all(torch.all(p.grad == 0) for p in model.parameters() if p.grad is not None)


def test_chunked_ce_preserves_float64_dtype_and_value():
    """fp64 调试路径不能在 loss 处静默降为 fp32。"""
    model = _tiny("standard").double()
    idx = torch.randint(0, 64, (1, 7))
    loss = model(idx, targets=idx)["loss"]
    logits = model(idx)["logits"]
    ref = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)), idx.reshape(-1))
    assert loss.dtype == torch.float64
    assert torch.allclose(loss, ref, atol=1e-12, rtol=1e-12)
