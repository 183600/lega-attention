"""过拟合小批量 sanity check 测试：正常模型必须通过、梯度断流必须被拦下。"""
import torch

from lega.model import GPT
from lega.sanity import (batch_metrics, gradient_coverage, losses_consistent,
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


def test_overfit_check_passes_on_healthy_model():
    model = _model()
    x, y = _batch()
    res = run_overfit_check(model, x, y, steps=150, lr=5e-3, device="cpu",
                            seed=0, warmup=10, log_every=50,
                            loss_threshold=0.1, acc_threshold=0.99)
    assert res["passed"], res["checks"]
    assert res["final"]["loss"] < res["initial"]["loss"]
    assert res["final"]["token_acc"] > 0.95
    assert res["first_grad_norm"] and res["first_grad_norm"] > 0
    # 中间结果核对：训练路径 loss ≡ 直接 logits CE；标签为 next-token 移位
    assert res["checks"]["train_loss_matches_direct_logits"], res["loss_consistency"]
    assert res["checks"]["next_token_shift_consistent"]
    assert not model.training  # 检查结束后恢复 eval 态，避免污染调用方
    print(f"  [ok] 健康模型可过拟合固定批: loss="
          f"{res['initial']['loss']:.3f}->{res['final']['loss']:.4f}, "
          f"acc={res['final']['token_acc']:.3f}")


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
