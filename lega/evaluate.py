"""独立困惑度评测：从 checkpoint 计算 PPL（默认 held-out test，缺 test.bin 时回退 val）。

用法:
  python -m lega.evaluate --ckpt out/tiny/lega/model.pt --data_dir data/dummy \
      --out results/tiny/eval_lega.json
"""
import argparse
import contextlib
import json
import math
import os

import torch
import torch.nn.functional as F

from .data import get_val_windows, load_bin, load_tokenizer
from .datacheck import append_test_access
from .model import load_model, count_params, gate_extra_params
from .repro import load_provenance, sha256_file
from .train import evaluate_val, _val_eval_indices
from .utils import enable_tf32, get_device, now_str, save_json, set_seed


@torch.inference_mode()
def collect_worst_windows(model, val, ctx: int, device: str, tok,
                          n_worst: int = 3, amp: bool = False,
                          max_windows: int = 64, batch_windows: int = None,
                          split: str = "val") -> list:
    """语言建模的坏案例：挑出下一 token 预测最差的验证窗口并解码。

    以"窗口末位的下一 token 预测"为坏案例单元（与贪心生成评测口径一致）：
    损失 = 末位交叉熵；同时给出模型的 top-5 预测与真实 token，用于人工区分
    "样本本身不可预测（噪声/生僻 token）"与"模型系统性盲区"。
    """
    indices, _ = _val_eval_indices(len(val), ctx, max_windows)
    if batch_windows is None:
        batch_windows = max(1, min(16384 // max(1, ctx), len(indices)))
    amp_ctx = (torch.autocast("cuda", dtype=torch.bfloat16)
               if (amp and device.startswith("cuda")) else contextlib.nullcontext())
    rows = []
    for s in range(0, len(indices), batch_windows):
        chunk = indices[s:s + batch_windows]
        x, y = get_val_windows(val, ctx, chunk, device)
        with amp_ctx:
            out = model(x, last_logits_only=True)
        logits = out["logits"][:, -1, :].float()
        gold = y[:, -1]
        losses = F.cross_entropy(logits, gold, reduction="none")
        top_p, top_i = torch.topk(torch.softmax(logits, dim=-1), 5, dim=-1)
        # 整表转 Python 标量后再构造行记录。
        loss_l, gold_l = losses.tolist(), gold.tolist()
        top_p_l, top_i_l = top_p.tolist(), top_i.tolist()
        x_l = x.tolist()
        for r in range(len(chunk)):
            ctx_text = tok.decode(x_l[r])
            gold_txt = tok.decode([gold_l[r]])
            rows.append({
                "window": int(chunk[r]),
                "split": str(split),
                "loss": float(loss_l[r]),
                "context_tail": ctx_text[-160:],
                # gold 是否出现在完整上下文窗口内。必须在采集时用全窗口判定：
                # context_tail 只有 160 字符（窗口尾部），早先出现过的 gold 若只
                # 看尾部会被误判为"从未出现"（复制/检索失败被错记成预测失败）。
                "gold_in_context": bool(gold_txt) and gold_txt in ctx_text,
                "gold_next": gold_txt,
                "pred_top5": [(tok.decode([i]), round(float(p), 4))
                              for p, i in zip(top_p_l[r], top_i_l[r])],
            })
    rows.sort(key=lambda r: -r["loss"])
    return rows[:max(1, int(n_worst))]


def json_safe_eval_metrics(metrics: dict) -> dict:
    """把评测指标规范化为 JSON 安全且不误导的字段。

    训练内评测（train.py 的 do_eval）在写入前用 _nan_guard 对非有限 val_loss
    显式中止；独立评测入口没有这道闸门，必须在写盘前显式处理，否则：
      * json.dump 会把 NaN/Inf 序列化为非法 JSON 字面量（NaN/Infinity），
        外部消费者（jq 等）直接解析失败；
      * math.exp(min(20, float("nan"))) 恰为 exp(20)（min 对 NaN 返回第一个
        参数），而 bool(nan > 20) 为 False——失效 checkpoint 会被伪装成
        “正常但很大”的有限 PPL，且不触发任何截断标注。
    非有限值统一记为 None（缺失值语义），由下游按缺失处理；有限值沿用
    exp(min(20, loss)) 与 >20 的截断口径，与 train.py 完全一致。
    """
    vl = float(metrics["val_loss"])
    if math.isfinite(vl):
        ppl = math.exp(min(20, vl))
        loss, clamped = vl, bool(vl > 20)
    else:
        ppl, loss, clamped = None, None, None
    gmean = metrics.get("gate_mean")
    if gmean is not None and not math.isfinite(float(gmean)):
        gmean = None
    return {"val_ppl": ppl, "val_loss": loss, "val_ppl_clamped": clamped,
            "gate_mean": gmean}


def load_train_throughput(run_dir: str):
    """从训练 run 的 eval.json 继承训练步时记录（train_step_time_ms/tokens_per_sec）。

    旧 run 没有该字段时返回 (None, None)，消费方显示 '-' 而不是编造数值。
    文件缺失/损坏同样安全返回 (None, None)，不阻塞评测。
    """
    path = os.path.join(run_dir, "eval.json")
    if not os.path.exists(path):
        return None, None
    try:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
    except (OSError, ValueError):
        return None, None
    return d.get("train_step_time_ms"), d.get("train_tokens_per_sec")


def main():
    p = argparse.ArgumentParser(description="PPL 评测（默认 held-out test，缺 test.bin 时回退 val）")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out", required=True, help="输出 JSON 路径")
    p.add_argument("--eval_iters", type=int, default=None,
                   help="评测窗口数；默认继承训练配置，训练配置缺失时为 100")
    p.add_argument("--worst_n", type=int, default=0,
                   help="转储预测最差的 N 个验证/测试窗口（语言建模坏案例，0=关闭）")
    p.add_argument("--split", default="auto",
                   choices=["auto", "test", "val"],
                   help="默认 auto：存在 test.bin 时用 held-out test（checkpoint 选择在 "
                        "val 上完成，报告指标必须在独立 split 上），否则回退 val")
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    device = get_device(args.device)
    enable_tf32(device)
    model, ck = load_model(args.ckpt, device=device)
    # 评测同样固定随机源（评测本身无随机性，但保证同一 checkpoint 在任何
    # 入口下都从同一 RNG 状态出发）。
    seed = ck.get("seed")
    set_seed(int(seed) if seed is not None else 0)
    split = args.split
    if split == "auto":
        split = "test" if (os.path.exists(os.path.join(args.data_dir, "test.bin"))
                           and os.path.getsize(os.path.join(args.data_dir, "test.bin")) > 0) \
            else "val"
    if split == "val":
        # 与 checkpoint 选择（best on val）共用同一 split：报告指标存在选择偏差，
        # 须明确提示而非静默；正式数据请用 prepare_data.py 生成 held-out test。
        print("[eval] 警告：正在 val split 上评测——这也是 best checkpoint 的选择集，"
              "报告 PPL 偏乐观；可用 prepare_bins 生成 test.bin 后在 held-out test 上评测")
    test_access = None
    if split == "test":
        # 测试集尘封协议：每次在 test 上评测都写入访问账本（data_dir 下），
        # 同一 checkpoint 被反复评测会显式报警——反复据 test 调参会把测试集
        # 变成隐式训练集；账本让这种行为可审计。
        # 只读数据目录等无法写账本的情况不阻塞评测，但必须显式告警。
        try:
            test_access = append_test_access(args.data_dir, {
                "time": now_str(),
                "ckpt": os.path.abspath(args.ckpt),
                "ckpt_sha256": sha256_file(args.ckpt),
                "variant": ck.get("variant", model.variant),
                "seed": ck.get("seed"),
            })
        except OSError as e:
            print(f"[eval] 警告：test 访问账本写入失败（{e}）；"
                  "本次 test 评测未被记录")
        if test_access and test_access["checkpoints_evaluated_multiple_times"]:
            print(f"[eval] 警告：同一 checkpoint 在 test 上被重复评测"
                  f"（累计 {test_access['evaluations']} 次、"
                  f"{test_access['distinct_checkpoints']} 个 checkpoint）；"
                  "测试集只能用于最终一次性评估，反复评测/据此调参视为泄露")
    val = load_bin(args.data_dir, split)
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
    norm = json_safe_eval_metrics(metrics)
    ppl, vl, gmean = norm["val_ppl"], norm["val_loss"], norm["gate_mean"]
    if vl is None:
        print("[eval] 警告：验证损失非有限（NaN/Inf），该 checkpoint 已失效；"
              "val_ppl/val_loss 记为 null（避免写入非法 JSON 字面量、也避免 "
              "min(20, NaN) 恰为 exp(20) 被误读成截断 PPL），"
              "该结果不得与正常 PPL 比较")

    res = {
        "variant": ck.get("variant", model.variant),
        "seed": ck.get("seed"),
        "ckpt": args.ckpt,
        # 被评测产物的内容指纹：结果与 checkpoint 一一绑定，换/改 checkpoint
        # 后旧结果无法冒充（与 test 访问账本、实验台账中的哈希同一口径）
        "ckpt_sha256": sha256_file(args.ckpt),
        "split": split,
        # checkpoint（model_best.pt）在 val 上选出；split=val 时报告指标与选择集同源
        "selection_set_is_eval_set": bool(split == "val"),
        "val_ppl": ppl,
        "val_loss": vl,
        # 通用别名：split="test" 时 val_ppl/val_loss 为兼容键名，消费方应优先用这里
        "ppl": ppl,
        "loss": vl,
        "val_ppl_clamped": norm["val_ppl_clamped"],
        "eval_ctx": ctx,
        "gate_mean": gmean,
        "n_params": count_params(model),
        "n_gate_extra": gate_extra_params(model),
        "eval_iters_requested": metrics["eval_iters_requested"],
        "eval_iters": metrics["eval_iters"],
        "val_windows": metrics["val_windows"],
        "val_tokens_evaluated": metrics["val_tokens_evaluated"],
        # test 访问账本摘要（split=val 时为 None）：审计"测试集被反复评测"
        "test_access": test_access,
    }
    # 坏案例：把预测最差的窗口连同 top-5 预测一起落盘（PPL 只反映平均，
    # 最差窗口揭示模型盲区——生僻 token？不可预测噪声？特定模式崩溃？）
    if args.worst_n > 0:
        try:
            tok = load_tokenizer(args.data_dir)
            res["worst_windows"] = collect_worst_windows(
                model, val, ctx, device, tok, n_worst=int(args.worst_n),
                amp=use_amp, split=split)
            print(f"[eval] 最差 {len(res['worst_windows'])} 个窗口已随结果落盘"
                  "（res['worst_windows']）")
        except Exception as e:  # 坏案例转储失败不应阻塞评测本身
            print(f"[eval] 警告：最差窗口转储失败（{e}）")
    # 继承训练 run 的代码/数据溯源信息（run_manifest.json），保证结果可追溯
    run_dir = os.path.dirname(os.path.abspath(args.ckpt))
    res.update(load_provenance(run_dir, args.data_dir))
    # 训练步时随被评测 checkpoint 一起绑定
    res["train_step_time_ms"], res["train_tokens_per_sec"] = \
        load_train_throughput(run_dir)
    save_json(res, args.out)
    if ppl is None:
        print(f"[eval] {res['variant']}: val_ppl=null（验证损失非有限）-> {args.out}")
    else:
        print(f"[eval] {res['variant']}: val_ppl={ppl:.3f} -> {args.out}")


if __name__ == "__main__":
    main()
