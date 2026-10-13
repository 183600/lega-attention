"""独立困惑度评测：从 checkpoint 计算验证集 PPL。

用法:
  python -m lega.evaluate --ckpt out/tiny/lega/model.pt --data_dir data/dummy \
      --out results/tiny/eval_lega.json
"""
import argparse
import json
import math
import os

from .data import load_bin
from .model import load_model, count_params, gate_extra_params
from .train import evaluate_val
from .utils import get_device, save_json


def main():
    p = argparse.ArgumentParser(description="验证集 PPL 评测")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out", required=True, help="输出 JSON 路径")
    p.add_argument("--eval_iters", type=int, default=None,
                   help="评测窗口数；默认继承训练配置，训练配置缺失时为 100")
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    device = get_device(args.device)
    model, ck = load_model(args.ckpt, device=device)
    val = load_bin(args.data_dir, "val")
    # 与训练时的评估保持一致：优先使用训练配置里的 train_ctx
    cfg_path = os.path.join(os.path.dirname(os.path.abspath(args.ckpt)), "config_used.json")
    train_ctx = None
    use_amp = device.startswith("cuda")
    cfg_used = {}
    if os.path.exists(cfg_path):
        with open(cfg_path, encoding="utf-8") as f:
            cfg_used = json.load(f)
            train_ctx = cfg_used.get("train_ctx")
            # 与训练内评测口径一致：尊重训练配置的 amp 设置
            use_amp = bool(cfg_used.get("amp", True)) and device.startswith("cuda")
    else:
        # 缺少训练配置时评测上下文静默变为 block_size，PPL 口径与训练内评测不一致
        print(f"[eval] 警告：未找到 {cfg_path}，评测上下文退化为 "
              f"model.block_size={model.block_size}，PPL 可能与训练内评测不可比")
    ctx = min(model.block_size, int(train_ctx)) if train_ctx else model.block_size
    eval_iters = (int(args.eval_iters) if args.eval_iters is not None
                  else int(cfg_used.get("eval_iters", 100)))

    # 与训练内评测口径一致：尊重训练配置的 amp/eval_iters 设置（显式参数优先）。
    # evaluate_val 内部会把窗口不重复且均匀铺满整个验证集，避免循环复用样本。
    metrics = evaluate_val(model, val, ctx, device, eval_iters, amp=use_amp)
    vl = metrics["val_loss"]
    ppl = math.exp(min(20, vl))
    gmean = metrics["gate_mean"]

    res = {
        "variant": ck.get("variant", model.variant),
        "ckpt": args.ckpt,
        "val_ppl": ppl,
        "val_loss": vl,
        "val_ppl_clamped": bool(vl > 20),
        "eval_ctx": ctx,
        "gate_mean": gmean,
        "n_params": count_params(model),
        "n_gate_extra": gate_extra_params(model),
        "eval_iters_requested": metrics["eval_iters_requested"],
        "eval_iters": metrics["eval_iters"],
        "val_windows": metrics["val_windows"],
        "val_tokens_evaluated": metrics["val_tokens_evaluated"],
    }
    save_json(res, args.out)
    print(f"[eval] {res['variant']}: val_ppl={ppl:.3f} -> {args.out}")


if __name__ == "__main__":
    main()
