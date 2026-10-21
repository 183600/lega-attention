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


def test_generate_rejects_invalid_length_and_top_k():
    model = _tiny("standard")
    model.eval()
    idx = torch.randint(0, 64, (1, 5))
    for bad in (-1, 1.5, True):
        try:
            model.generate(idx, max_new_tokens=bad)
            raise AssertionError(f"max_new_tokens={bad!r} 应被拒绝")
        except ValueError as exc:
            assert "max_new_tokens" in str(exc)
    for bad in (0, -1, 1.5, True):
        try:
            model.generate(idx, max_new_tokens=2, greedy=False, top_k=bad)
            raise AssertionError(f"top_k={bad!r} 应被拒绝")
        except ValueError as exc:
            assert "top_k" in str(exc)
    assert torch.equal(model.generate(idx, max_new_tokens=0), idx)
    print("  [ok] generate：拒绝非法 max_new_tokens/top_k")


def test_generate_eos_stops_per_row_without_batch_coupling():
    """提供 eos_token_id 时各行的 EOS 判定必须独立：批内某行的命中/未命中
    不能改变其它行的生成内容，也不得把「各行不同步命中」误判成不停。"""
    torch.manual_seed(0)
    m = GPT(vocab_size=12, block_size=32, d_model=16, n_layers=1, n_heads=2,
            variant="standard")
    m.eval()
    T0, max_new, eos = 6, 10, 5
    # 以 prompt 首 token 标识行；值 = 产生 EOS 的生成步下标。行 1 第 1 步、
    # 行 2 第 3 步命中，行 3 从不命中（不同步，无法靠“全体同时命中”停止）。
    hit_at = {1: 1, 2: 3}
    state = {"rows": None}

    def fake_forward(cur, kv_cache=None, cache_pos=0, last_logits_only=True):
        if cur.shape[1] > 1:
            state["rows"] = cur[:, 0].tolist()
        step = cache_pos + cur.shape[1] - T0     # 本次将产生的 token 的步下标
        logits = torch.full((cur.shape[0], 1, m.vocab_size), -5.0)
        for r, key in enumerate(state["rows"]):
            t = eos if hit_at.get(key, -1) == step else (key + 6) % m.vocab_size
            logits[r, 0, t] = 5.0
        return {"logits": logits, "loss": None, "past_kv": None, "gate_stats": None}

    m.forward = fake_forward
    idx = torch.tensor([[k, 7, 7, 7, 7, 7] for k in (1, 2, 3)])
    with torch.no_grad():
        ref = m.generate(idx, max_new_tokens=max_new)
        ab = m.generate(idx[:2], max_new_tokens=max_new, eos_token_id=eos)
        ac = m.generate(idx[::2], max_new_tokens=max_new, eos_token_id=eos)
        single = m.generate(idx[:1], max_new_tokens=max_new, eos_token_id=eos)
    assert ref.shape == (3, T0 + max_new)
    # 所有行都命中后，截到「最后一行首次命中」的步（行 2 第 3 步 → 长度 10）
    assert ab.shape[1] == T0 + 3 + 1
    assert torch.equal(ab, ref[:2, :T0 + 3 + 1]), "停止只能裁剪，不能改变生成内容"
    # 有行从不命中时不得提前停止；行内容不随批内其它行改变
    assert ac.shape == (2, T0 + max_new)
    assert torch.equal(ac, ref[::2]), "行内容与批内其它行耦合"
    # 单行行为保持：首次命中即停（第 1 步 → 长度 8）
    assert torch.equal(single, ref[:1, :T0 + 1 + 1])
    print("  [ok] EOS 逐行独立停止：批量截断到最晚命中步，单行不受批内影响")


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


def test_checkpoint_roundtrip_preserves_attention_path():
    """评测恢复必须沿用训练时的注意力实现，不能被默认值静默改写。"""
    torch.manual_seed(0)
    model = GPT(vocab_size=64, block_size=32, d_model=32, n_layers=1, n_heads=2,
                ffn_mult=4, variant="standard", use_sdpa_standard=False)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "model.pt")
        save_checkpoint(path, model)
        m2, _ = load_model(path)
        assert m2.blocks[0].attn.use_sdpa is False
        # 显式覆盖仍应生效（latency 需要 standard/standard_sdpa 两条路径）
        m3, _ = load_model(path, use_sdpa_standard=True)
        assert m3.blocks[0].attn.use_sdpa is True


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


def test_evaluate_val_reports_window_sampling_uncertainty():
    """有限个验证窗口的点估计必须附带窗口间抽样标准误与 CI。"""
    torch.manual_seed(0)
    model = _tiny("lega")
    val = (np.arange(100, dtype=np.int64) % 64).astype(np.uint16)
    metrics = evaluate_val(model, val, ctx=10, device="cpu", iters=9, amp=False)
    assert metrics["val_loss_se"] is not None and metrics["val_loss_se"] > 0
    lo, hi = metrics["val_loss_ci95"]
    assert lo < metrics["val_loss"] < hi
    ppl_lo, ppl_hi = metrics["val_ppl_ci95"]
    assert 0 < ppl_lo <= ppl_hi
    assert metrics["val_loss_min"] <= metrics["val_loss"] <= metrics["val_loss_max"]
    # 单窗口时无法估计窗口间方差：SE/CI 必须为 None，而不是给出虚假精度。
    one = evaluate_val(model, val, ctx=10, device="cpu", iters=1, amp=False)
    assert one["eval_iters"] == 1
    assert one["val_loss_se"] is None and one["val_loss_ci95"] is None
    assert one["val_loss_min"] == one["val_loss"] == one["val_loss_max"]


def test_per_window_losses_match_individual_forwards():
    """loss_per_window 每行必须等于单独评测该窗口的损失，且均值等于总损失。"""
    torch.manual_seed(0)
    model = _tiny("lega")
    model.eval()
    idx = torch.randint(0, 64, (3, 10))
    y = torch.randint(0, 64, (3, 10))
    with torch.inference_mode():
        out = model(idx, targets=y, return_window_losses=True)
    assert out["loss_per_window"].shape == (3,)
    assert abs(float(out["loss"]) -
               float(out["loss_per_window"].mean())) < 1e-6
    for r in range(3):
        with torch.inference_mode():
            one = model(idx[r:r + 1], targets=y[r:r + 1])
        assert abs(float(one["loss"]) -
                   float(out["loss_per_window"][r])) < 1e-5
    # 训练/需要梯度的路径不得支持该选项（避免悄悄改变训练图）
    model.train()
    try:
        model(idx, targets=y, return_window_losses=True)
    except RuntimeError as e:
        assert "return_window_losses" in str(e)
    else:
        raise AssertionError("梯度路径必须拒绝 return_window_losses")


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


def test_chunked_ce_repeated_backward_does_not_decay_grads():
    """retain_graph 下重复 backward：前向物化的 dx/gw 不得被原地重复缩放，
    否则第二次梯度会被再乘一次 1/n_valid（并不是可累加的等值梯度）。"""
    model = _tiny("standard")
    idx = torch.randint(0, 64, (2, 9))
    loss = model(idx, targets=idx)["loss"]
    loss.backward(retain_graph=True)
    first = {n: p.grad.detach().clone() for n, p in model.named_parameters()
             if p.grad is not None}
    loss.backward()
    for n, p in model.named_parameters():
        if n in first:
            assert torch.allclose(p.grad, 2 * first[n], atol=1e-6, rtol=1e-5), \
                f"{n}: 重复 backward 梯度不可累加"
    print("  [ok] ChunkedLinearCE 重复 backward 梯度可累加（无 1/n 衰减）")


def test_frozen_embedding_training_backward_under_grad():
    """冻结嵌入/detach 上游时梯度模式下不得走残差原地加法路径：
    LayerNorm 反向会保存 x（用于 weight/bias 梯度），原地改写 x 会让
    backward 报“变量被 inplace 修改”而直接失败。"""
    model = _tiny("lega")
    model.tok_emb.weight.requires_grad_(False)   # lm_head 权重与其共享
    model.train()
    idx = torch.randint(0, 64, (2, 10))
    loss = model(idx, targets=idx)["loss"]
    loss.backward()
    assert model.tok_emb.weight.grad is None
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in model.parameters() if p.requires_grad)
    print("  [ok] 冻结嵌入时训练 backward 在梯度模式下正常完成")


def test_evaluate_val_optional_window_loss_detail_for_pairing():
    """独立评测可落盘逐窗口损失/序号，供跨变体窗口级配对；训练默认不落盘。"""
    torch.manual_seed(0)
    model = _tiny("lega")
    val = (np.arange(100, dtype=np.int64) % 64).astype(np.uint16)
    metrics = evaluate_val(model, val, ctx=10, device="cpu", iters=9,
                           amp=False, collect_window_losses=True)
    assert len(metrics["window_losses"]) == metrics["eval_iters"] == 9
    assert len(metrics["window_indices"]) == 9
    assert len(set(metrics["window_indices"])) == 9
    assert metrics["window_indices"] == sorted(metrics["window_indices"])
    assert abs(sum(metrics["window_losses"]) / 9 - metrics["val_loss"]) < 1e-6
    plain = evaluate_val(model, val, ctx=10, device="cpu", iters=9, amp=False)
    assert "window_losses" not in plain and "window_indices" not in plain
    print("  [ok] evaluate_val 逐窗口明细可选落盘，且均值与 val_loss 一致")
