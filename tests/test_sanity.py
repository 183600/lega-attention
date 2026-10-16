"""过拟合小批量 sanity check 测试：正常模型必须通过、梯度断流必须被拦下。"""
import torch

from lega.model import GPT
from lega.sanity import batch_metrics, run_overfit_check


def _model():
    torch.manual_seed(0)
    return GPT(vocab_size=64, block_size=32, d_model=32, n_layers=1, n_heads=2,
               ffn_mult=4, variant="lega", dropout=0.0, attn_dropout=0.0)


def _batch():
    g = torch.Generator().manual_seed(1)
    x = torch.randint(0, 64, (4, 16), generator=g)
    y = torch.randint(0, 64, (4, 16), generator=g)
    return x, y


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
