"""过拟合小批量 sanity check 测试：正常模型必须通过、梯度断流必须被拦下。"""
import numpy as np
import torch

from lega.model import GPT
from lega.sanity import (active_regularization, batch_metrics,
                         gradient_coverage, losses_consistent,
                         run_overfit_check)


def _model():
    torch.manual_seed(0)
    return GPT(vocab_size=64, block_size=32, d_model=32, n_layers=1, n_heads=2,
               ffn_mult=4, variant="lega", dropout=0.0, attn_dropout=0.0)


def _batch():
    # 与 get_train_batch 相同的数据契约：y = x 右移一位
    g = torch.Generator().manual_seed(1)
    both = torch.randint(0, 64, (4, 17), generator=g)
    return both[:, :-1], both[:, 1:]


def test_batch_metrics_initial_loss_near_log_vocab():
    import math
    model = _model()
    x, y = _batch()
    m = batch_metrics(model, x, y)
    # 随机初始化下交叉熵应接近 log(V)（这是排查标签错位的第一中间量）
    assert abs(m["loss"] - math.log(64)) < 0.5, m
    assert 0.0 <= m["token_acc"] <= 1.0 and m["n_tokens"] == x.numel()


def test_sanity_main_blocks_on_stale_data_cache(monkeypatch):
    """sanity 入口必须与训练共用数据协议闸门，不能绕过泄露拦截。"""
    import json
    import os
    import sys
    import tempfile

    from lega.data import CACHE_SCHEMA
    from lega.sanity import main as sanity_main
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = os.path.join(root, "configs", "tiny.json")
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "meta.json"), "w", encoding="utf-8") as f:
            json.dump({"cache_schema": CACHE_SCHEMA - 1}, f)
        monkeypatch.setattr(sys, "argv", [
            "sanity", "--config", cfg, "--data_dir", d, "--variant", "lega"])
        try:
            sanity_main()
            raise AssertionError("旧缓存协议应阻断 sanity")
        except SystemExit as e:
            assert "数据协议前置检查失败" in str(e)
    print("  [ok] sanity：与训练共用数据协议闸门，旧缓存协议默认阻断")


def test_overfit_check_passes_on_healthy_model():
    model = _model()
    x, y = _batch()
    res = run_overfit_check(model, x, y, steps=150, lr=5e-3, device="cpu",
                            seed=0, warmup=10, log_every=50, grad_clip=10.0,
                            loss_threshold=0.1, acc_threshold=0.99)
    assert res["passed"], res["checks"]
    assert res["final"]["loss"] < res["initial"]["loss"]
    assert res["final"]["token_acc"] > 0.95
    assert res["first_grad_norm"] and res["first_grad_norm"] > 0
    # first_grad_norm 必须是 clip 前原始范数；后范数只作为被裁剪后的对照
    assert res["first_grad_norm"] == res["first_grad_norm_preclip"]
    assert res["first_grad_norm_postclip"] is not None
    assert res["first_grad_norm_was_clipped"] is False, res["first_grad_norm"]
    # 中间结果核对：训练路径 loss ≡ 直接 logits CE；标签为 next-token 移位
    assert res["checks"]["train_loss_matches_direct_logits"], res["loss_consistency"]
    assert res["checks"]["next_token_shift_consistent"]
    assert not model.training  # 检查结束后恢复 eval 态，避免污染调用方
    print(f"  [ok] 健康模型可过拟合固定批: loss="
          f"{res['initial']['loss']:.3f}->{res['final']['loss']:.4f}, "
          f"acc={res['final']['token_acc']:.3f}")


def test_overfit_check_rejects_degenerate_and_oov_batches():
    """固定批自身必须是可学习数据：常量批/越界 token/形状错误先被拦下。"""
    model = _model()
    base = dict(steps=2, lr=1e-3, device="cpu", seed=0, log_every=1)
    constant = torch.zeros(4, 17, dtype=torch.long)
    res = run_overfit_check(model, constant, constant, **base)
    assert not res["passed"]
    assert res["checks"]["batch_not_constant"] is False
    assert "constant_batch" in res["batch_contract"]["errors"]
    assert res["initial"] is None and res["final"] is None

    oov = torch.full((4, 17), 64, dtype=torch.long)
    res = run_overfit_check(model, oov, oov, **base)
    assert not res["passed"]
    assert res["checks"]["batch_tokens_in_vocab"] is False
    assert res["batch_contract"]["n_out_of_range"] == oov.numel() * 2
    assert res["initial"] is None and res["final"] is None

    x = torch.zeros(4, 17, dtype=torch.long)
    res = run_overfit_check(model, x, x[:, :-1], **base)
    assert not res["passed"]
    assert res["checks"]["batch_same_shape"] is False
    assert "shape_mismatch" in res["batch_contract"]["errors"]

    floats = torch.zeros(4, 17, dtype=torch.float32)
    res = run_overfit_check(model, floats, floats, **base)
    assert not res["passed"]
    assert res["checks"]["batch_integer_tokens"] is False
    assert "non_integer_dtype" in res["batch_contract"]["errors"]
    print("  [ok] sanity 拒绝常量批/越界 token/形状错误/非整数 token")


def test_overfit_check_rejects_invalid_numeric_args():
    """sanity 数值参数不能被 bool/NaN/Inf/非数值静默改写。"""
    import math
    model = _model()
    x, y = _batch()
    base = dict(steps=0, lr=5e-3, device="cpu", seed=0, warmup=0,
                log_every=1)
    for name in ("lr", "grad_clip", "loss_threshold", "acc_threshold",
                 "initial_loss_tol"):
        for bad in (True, np.bool_(True), float("nan"), float("inf"), "bad"):
            kwargs = dict(base)
            kwargs[name] = bad
            try:
                run_overfit_check(model, x, y, **kwargs)
            except ValueError:
                pass
            else:
                raise AssertionError(f"非法 {name} 未被拒绝: {bad!r}")
    for name in ("lr", "grad_clip"):
        for bad in (-1.0,):
            kwargs = dict(base)
            kwargs[name] = bad
            try:
                run_overfit_check(model, x, y, **kwargs)
            except ValueError:
                pass
            else:
                raise AssertionError(f"负 {name} 未被拒绝")
    # 数字字符串仍可解析；initial_loss_tol<0 仍是“显式禁用”
    res = run_overfit_check(model, x, y, steps=0, lr="5e-3", device="cpu",
                            seed=0, warmup=0, log_every=1,
                            loss_threshold="0.05", acc_threshold="0.99",
                            initial_loss_tol="-1.0")
    assert res["checks"]["initial_loss_near_log_vocab"] is True
    print("  [ok] sanity 数值参数拒绝 bool/NaN/Inf/非数值与负 lr/grad_clip")


def test_overfit_check_detects_frozen_params():
    model = _model()
    for p in model.parameters():
        p.requires_grad_(False)
    x, y = _batch()
    res = run_overfit_check(model, x, y, steps=20, lr=5e-3, device="cpu",
                            seed=0, log_every=10)
    assert not res["passed"]
    assert res["checks"]["loss_requires_grad"] is False
    assert res["checks"]["gradient_nonzero_first_step"] is False
    print("  [ok] 冻结参数（梯度断流）被 sanity check 判定为失败")


def test_gradient_coverage_classifies_missing_zero_nonfinite():
    """逐参数覆盖：None/全零/非有限梯度分别归入不同列表。"""
    model = _model()
    x, y = _batch()
    model(x, targets=y)["loss"].backward()
    cov = gradient_coverage(model)
    assert cov["n_trainable"] > 0
    assert cov["missing_grad"] == [] and cov["nonfinite_grad"] == []
    named = [n for n, p in model.named_parameters() if p.requires_grad]
    name0, name1, name2 = named[0], named[1], named[2]
    params = dict(model.named_parameters())
    params[name0].grad = None
    params[name1].grad = torch.zeros_like(params[name1].grad)
    params[name2].grad = torch.full_like(params[name2].grad, float("nan"))
    cov2 = gradient_coverage(model)
    assert cov2["missing_grad"] == [name0]
    assert cov2["zero_grad"] == [name1]
    assert cov2["nonfinite_grad"] == [name2]
    print("  [ok] gradient_coverage 区分 missing/zero/nonfinite 梯度")


def test_overfit_check_detects_detached_submodule():
    """总梯度范数正常、但子模块没接进计算图（detach）也必须判失败。

    这是比“全参数冻结”更隐蔽的问题：loss.requires_grad 与总梯度范数都正常，
    只有逐参数覆盖能定位到断流参数。
    """
    model = _model()
    orig_forward = model.ln_f.forward

    def detached_forward(x):
        return orig_forward(x).detach()

    model.ln_f.forward = detached_forward
    x, y = _batch()
    res = run_overfit_check(model, x, y, steps=5, lr=1e-3, device="cpu",
                            seed=0, log_every=1)
    assert not res["passed"]
    assert res["checks"]["loss_requires_grad"] is True
    assert res["checks"]["all_params_get_grads"] is False
    missing = res["grad_coverage"]["missing_grad"]
    assert any("ln_f" in n for n in missing), missing
    assert any(n.startswith("blocks.") for n in missing), missing
    print(f"  [ok] detach 子模块导致 {len(missing)} 个参数收不到梯度，"
          "被逐参数覆盖检查拦下")


def test_overfit_check_reports_preclip_grad_norm():
    """只有 clip 前范数不足以看出梯度是否被截断：爆炸会被截成 grad_clip。

    保留裁剪后范数与是否触发裁剪，才能避免 grad_clip 越小、报告越“稳定”，
    进而完全掩盖首步梯度爆炸。
    """
    model = _model()
    x, y = _batch()
    res = run_overfit_check(model, x, y, steps=2, lr=0.0, device="cpu",
                            seed=0, log_every=1, grad_clip=1e-6)
    pre = res["first_grad_norm_preclip"]
    post = res["first_grad_norm_postclip"]
    assert pre is not None and post is not None
    assert pre > res["grad_clip"] > 0
    assert post <= res["grad_clip"] + 1e-12, (pre, post)
    assert res["first_grad_norm"] == pre
    assert res["first_grad_norm_was_clipped"] is True
    assert res["checks"]["gradient_nonzero_first_step"] is True
    # 每个日志点都保留 clip 前后两套读数，不能只在首步做样子。
    for h in res["history"]:
        assert h["grad_norm_preclip"] is not None
        assert h["grad_norm_postclip"] is not None
        assert h["grad_norm_was_clipped"] is True
    print(f"  [ok] sanity 区分 clip 前后梯度范数: {pre:.3g} -> {post:.3g}")


def test_overfit_check_detects_all_zero_grad_submodule():
    """子模块收到的是全零梯度（如 detach 后又乘 0）也必须判失败。

    这种断流不会让 grad 变成 None，总梯度范数也可能正常；只有逐参数全零
    检查能发现“参数确实在计算图里，但没有任何有效学习信号”。
    """
    model = _model()

    def zero_ln_f_grad(module, grad_input, grad_output):
        if module.weight.grad is not None:
            module.weight.grad = torch.zeros_like(module.weight)

    handle = model.ln_f.register_full_backward_hook(zero_ln_f_grad)
    try:
        x, y = _batch()
        res = run_overfit_check(model, x, y, steps=5, lr=1e-3, device="cpu",
                                seed=0, log_every=1)
    finally:
        handle.remove()
    assert not res["passed"]
    assert res["checks"]["no_zero_grads_first_step"] is False
    assert "ln_f.weight" in res["grad_coverage"]["zero_grad"], \
        res["grad_coverage"]
    print("  [ok] 首步全零梯度的子模块被 sanity 判为断流")


def test_losses_consistent_tolerance():
    assert losses_consistent(2.0, 2.0 + 1e-6)
    assert not losses_consistent(2.0, 2.0 + 0.1)
    assert not losses_consistent(None, 1.0)
    assert not losses_consistent(float("nan"), 1.0)
    assert not losses_consistent(float("inf"), float("inf"))
    print("  [ok] losses_consistent（容差/None/非有限值判定）")


def test_overfit_check_detects_loss_computation_mismatch():
    """训练路径损失计算错误（如错误缩放/归一化）：仍可能过拟合，但数值对不上。

    这正是“能过拟合 ≠ 损失计算正确”的场景：Adam 对梯度的常数缩放近似
    不敏感，过拟合照样成功；只有训练路径 loss 与直接 logits CE 的
    交叉核对能发现。
    """
    model = _model()
    orig_forward = model.forward

    def scaled_loss_forward(idx, targets=None, **kw):
        out = orig_forward(idx, targets=targets, **kw)
        if targets is not None and out["loss"] is not None:
            out["loss"] = out["loss"] * 2.0
        return out

    model.forward = scaled_loss_forward
    x, y = _batch()
    res = run_overfit_check(model, x, y, steps=40, lr=5e-3, device="cpu",
                            seed=0, log_every=20)
    assert not res["passed"]
    assert res["checks"]["train_loss_matches_direct_logits"] is False
    ratio = (res["loss_consistency"]["train_path"]
             / res["loss_consistency"]["direct_logits"])
    assert abs(ratio - 2.0) < 1e-3, ratio
    print("  [ok] loss 计算错误（训练路径被缩放 2×）被一致性核对拦下")


def test_overfit_check_detects_broken_next_token_shift():
    """标签不是 x 右移一位（错位/换序）必须在消费点被拦下。"""
    model = _model()
    x, y = _batch()
    y_broken = y.flip(1)
    res = run_overfit_check(model, x, y_broken, steps=5, lr=1e-3, device="cpu",
                            seed=0, log_every=1)
    assert not res["passed"]
    assert res["checks"]["next_token_shift_consistent"] is False
    print("  [ok] 标签错位（y ≠ x 右移一位）被消费点校验拦下")


def test_overfit_check_restores_tf32_flags_independently():
    """TF32 开关恢复必须分别处理 matmul 与 cudnn，不能用同一变量互相覆盖。"""
    model = _model()
    x, y = _batch()
    mm = torch.backends.cuda.matmul.allow_tf32
    cudnn = torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True
        run_overfit_check(model, x, y, steps=1, lr=1e-3, device="cpu", seed=0,
                          warmup=1, log_every=1)
        assert torch.backends.cuda.matmul.allow_tf32 is False
        assert torch.backends.cudnn.allow_tf32 is True, "cudnn 开关被 matmul 的旧值覆盖"
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = False
        run_overfit_check(model, x, y, steps=1, lr=1e-3, device="cpu", seed=0,
                          warmup=1, log_every=1)
        assert torch.backends.cuda.matmul.allow_tf32 is True
        assert torch.backends.cudnn.allow_tf32 is False
    finally:
        torch.backends.cuda.matmul.allow_tf32 = mm
        torch.backends.cudnn.allow_tf32 = cudnn
    print("  [ok] TF32 开关分别还原（matmul/cudnn 互不覆盖）")


def test_overfit_check_detects_pathological_initial_logits():
    """初始 loss 远离 log(V)（logits 尺度病态）必须被自动中间结果核对拦下。

    这对应“初始化/标签分布异常”类问题：模型仍可能慢慢过拟合，但初始
    交叉熵不可能接近均匀分布的对数值，应立即报警而不是盲目开训。
    """
    import math
    model = _model()
    with torch.no_grad():
        model.lm_head.weight.mul_(30.0)          # 权重共享：embedding 同步放大
    x, y = _batch()
    res = run_overfit_check(model, x, y, steps=5, lr=1e-3, device="cpu",
                            seed=0, log_every=1)
    assert not res["passed"]
    assert res["checks"]["initial_loss_near_log_vocab"] is False
    assert abs(res["initial_loss_minus_log_vocab"]) > res["thresholds"]["initial_loss_tol"]
    assert math.isfinite(res["initial_loss_minus_log_vocab"])
    # 容差可配置：放宽到足够大时该检查通过（其余检查仍以各自阈值判定）
    res2 = run_overfit_check(model, x, y, steps=1, lr=1e-3, device="cpu",
                             seed=0, log_every=1, initial_loss_tol=1e6)
    assert res2["checks"]["initial_loss_near_log_vocab"] is True
    # 负值显式禁用该检查（偏差仍记录在案）
    res3 = run_overfit_check(model, x, y, steps=1, lr=1e-3, device="cpu",
                             seed=0, log_every=1, initial_loss_tol=-1.0)
    assert res3["checks"]["initial_loss_near_log_vocab"] is True
    assert abs(res3["initial_loss_minus_log_vocab"]) > 1.0
    print(f"  [ok] 病态 logits 初始偏差 "
          f"{res['initial_loss_minus_log_vocab']:+.2f} 被初始 loss≈log(V) 核对拦下")


def test_overfit_check_requires_regularization_disabled():
    """sanity 的“关闭正则化”是前提条件，必须自动核对而不是只写在文档里。

    dropout>0 时固定批也会被随机噪声干扰；若不检查，就会把“配置没关
    dropout”误报成梯度/损失错误（或反之）。
    """
    clean = _model()
    assert active_regularization(clean) == []
    noisy = GPT(vocab_size=64, block_size=32, d_model=32, n_layers=1,
                n_heads=2, ffn_mult=4, variant="lega", dropout=0.2,
                attn_dropout=0.1)
    active = active_regularization(noisy)
    assert any("Dropout" in a for a in active)
    assert any("attn_dropout" in a for a in active)
    x, y = _batch()
    res = run_overfit_check(noisy, x, y, steps=2, lr=1e-3, device="cpu",
                            seed=0, log_every=1)
    assert not res["passed"]
    assert res["checks"]["regularization_disabled"] is False
    assert res["active_regularization"] == active
    print(f"  [ok] 未关闭正则化被 sanity 自动拦下（{len(active)} 处）")
