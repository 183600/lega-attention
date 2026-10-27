"""输出级干预：把“写回效用”定义在答案 token 的 NLL/正确概率上。

门控统计（如 ``hit``、集中度）只是注意力分布的性质，不等价于该头/该位置
的写回对语言建模是否真的有帮助。本模块直接在答案 token 上做逐层、逐头的
输出干预：

* 干净前向：对答案 token 计算 NLL、正确 token 概率与 margin；
* 干预：在目标层 attention 输出投影 ``wo`` 的输入中，把指定头在答案位置
  （预测答案 token 的位置）的通道置零或按位置置换，再重新计算同一批答案
  token 的 NLL/margin；
* 输出 ΔNLL（干预 − 干净，>0 表示该头写回有助）与 Δmargin，供逐头
  “写入效用”分析，而不是拿注意力命中率当因果证据。

这是**推理期**干预，不重新训练；只支持单条/同构 batch 的答案跨度。
"""
from __future__ import annotations

import argparse
import json
from numbers import Integral
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

__all__ = ["score_answer", "head_output_intervention_scan", "main"]


def _strict_int(value, name: str, minimum: Optional[int] = None) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} 必须是整数，收到 {value!r}")
    value = int(value)
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} 必须 >= {minimum}，收到 {value}")
    return value


def _answer_positions(seq_len: int, start: int, end: Optional[int]
                      ) -> Tuple[int, int, int, int]:
    start = _strict_int(start, "answer_start", 1)
    end = seq_len if end is None else _strict_int(end, "answer_end", 1)
    if end <= start:
        raise ValueError(f"answer_end({end}) 必须大于 answer_start({start})")
    if end > seq_len:
        raise ValueError(f"answer_end({end}) 超出序列长度 {seq_len}")
    # logits[t] 预测 idx[t+1]；答案 token idx[start:end] 由预测位置
    # [start-1, end-1) 打分。wo 输入位置与预测位置相同。
    pred_start, pred_end = start - 1, end - 1
    return start, end, pred_start, pred_end


@torch.no_grad()
def score_answer(model, idx: torch.Tensor, answer_start: int,
                 answer_end: Optional[int] = None) -> Dict:
    """返回答案 token 的 NLL/正确概率/margin（干净前向）。"""
    if idx.dim() != 2:
        raise ValueError(f"idx 必须是 (B,T)，收到 {tuple(idx.shape)}")
    B, T = idx.shape
    start, end, _, _ = _answer_positions(T, answer_start, answer_end)
    out = model(idx, last_logits_only=False)
    logits = out["logits"]
    if logits is None:
        raise RuntimeError("模型未返回 logits")
    pred = logits[:, start - 1:end - 1, :].float()
    target = idx[:, start:end]
    logp = F.log_softmax(pred, dim=-1)
    nll_tok = -logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    prob_tok = nll_tok.exp()
    # margin = 正确 token logit - 其它 token 最大 logit
    correct = pred.gather(-1, target.unsqueeze(-1))
    other = pred.masked_fill(
        F.one_hot(target, pred.size(-1)).bool(), float("-inf")).amax(dim=-1)
    margin = (correct.squeeze(-1) - other)
    return {
        "answer_start": int(start), "answer_end": int(end),
        "n_answer_tokens": int(end - start),
        "nll": float(nll_tok.mean().item()),
        "prob": float(prob_tok.mean().item()),
        "margin": float(margin.mean().item()),
        "per_token_nll": nll_tok.mean(dim=0).tolist(),
    }


@torch.no_grad()
def _forward_with_head_hook(model, idx: torch.Tensor, layer: int, head: int,
                            positions: Sequence[int], mode: str = "zero"):
    block = model.blocks[layer]
    attn = block.attn
    Dh = int(attn.d_head)
    lo, hi = int(head) * Dh, (int(head) + 1) * Dh
    pos = torch.as_tensor(list(positions), dtype=torch.long,
                          device=idx.device)
    removed = []

    def hook(module, args):
        x = args[0]
        y = x.clone()
        # take 必须只含目标头的通道：mean_abs_removed 是“该头被置零/置换的
        # 激活幅度”，用全部通道求均值会让不同头报告同一个值，无法解释
        # 逐头干预差异。
        take = y.index_select(1, pos)[..., lo:hi]
        if mode == "zero":
            y[:, pos, lo:hi] = 0.0
        elif mode == "permute":
            # 在答案位置之间循环移位；逐值分布不变，只破坏位置对应。
            if take.shape[1] > 1:
                y[:, pos, lo:hi] = torch.roll(take, shifts=1, dims=1)
        else:
            raise ValueError(f"未支持的干预模式: {mode!r}")
        removed.append(float(take.abs().mean().item()))
        return (y,) + tuple(args[1:])

    h = attn.wo.register_forward_pre_hook(hook)
    try:
        out = model(idx, last_logits_only=False)
    finally:
        h.remove()
    return out, (removed[0] if removed else 0.0)


def head_output_intervention_scan(model, idx: torch.Tensor, answer_start: int,
                                  answer_end: Optional[int] = None,
                                  layers: Optional[Iterable[int]] = None,
                                  heads: Optional[Iterable[int]] = None,
                                  scope: str = "answer",
                                  modes: Sequence[str] = ("zero",),
                                  ) -> Dict:
    """逐层/逐头在答案位置干预输出，返回答案 NLL/margin 的配对增量。

    ``scope="answer"`` 只改预测答案 token 的位置；``scope="all"`` 改整条
    序列（用于检查“只改答案位置”是否真的局部化）。每个 (layer, head, mode)
    都相对**同一次干净前向**计算 ΔNLL/Δmargin，因此可直接比较。
    """
    if idx.dim() != 2:
        raise ValueError(f"idx 必须是 (B,T)，收到 {tuple(idx.shape)}")
    B, T = idx.shape
    if B != 1:
        raise ValueError("输出级干预目前只支持 batch=1（同一答案跨度）")
    start, end, pred_start, pred_end = _answer_positions(
        T, answer_start, answer_end)
    n_layers = len(model.blocks)
    if layers is None:
        layers = range(n_layers)
    if heads is None:
        heads = range(int(model.blocks[0].attn.n_heads))
    layers = [_strict_int(x, "layer", 0) for x in layers]
    heads = [_strict_int(x, "head", 0) for x in heads]
    for li in layers:
        if li >= n_layers:
            raise ValueError(f"layer {li} 超出层数 {n_layers}")
    for hi in heads:
        if hi >= int(model.blocks[0].attn.n_heads):
            raise ValueError(f"head {hi} 超出头数 {model.blocks[0].attn.n_heads}")
    if scope == "answer":
        positions = list(range(pred_start, pred_end))
    elif scope == "all":
        positions = list(range(T))
    else:
        raise ValueError(f"scope 必须是 answer/all，收到 {scope!r}")
    modes = tuple(str(m) for m in modes)
    if not modes:
        raise ValueError("modes 不能为空")

    base = score_answer(model, idx, start, end)
    rows = []
    for li in layers:
        for hi in heads:
            for mode in modes:
                out, removed = _forward_with_head_hook(
                    model, idx, li, hi, positions, mode=mode)
                got = score_answer_from_logits(out["logits"], idx, start, end)
                rows.append({
                    "layer": int(li), "head": int(hi), "mode": mode,
                    "scope": scope,
                    "n_positions": len(positions),
                    "mean_abs_removed": float(removed),
                    "nll": got["nll"], "prob": got["prob"],
                    "margin": got["margin"],
                    "delta_nll": float(got["nll"] - base["nll"]),
                    "delta_prob": float(got["prob"] - base["prob"]),
                    "delta_margin": float(got["margin"] - base["margin"]),
                })
    return {"baseline": base, "interventions": rows,
            "answer_start": int(start), "answer_end": int(end),
            "scope": scope}


def score_answer_from_logits(logits: torch.Tensor, idx: torch.Tensor,
                             answer_start: int,
                             answer_end: Optional[int] = None) -> Dict:
    """与 :func:`score_answer` 同口径，但复用已算好的 logits。"""
    B, T = idx.shape
    start, end, _, _ = _answer_positions(T, answer_start, answer_end)
    pred = logits[:, start - 1:end - 1, :].float()
    target = idx[:, start:end]
    logp = F.log_softmax(pred, dim=-1)
    nll_tok = -logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    prob_tok = nll_tok.exp()
    correct = pred.gather(-1, target.unsqueeze(-1))
    other = pred.masked_fill(
        F.one_hot(target, pred.size(-1)).bool(), float("-inf")).amax(dim=-1)
    margin = correct.squeeze(-1) - other
    return {"nll": float(nll_tok.mean().item()),
            "prob": float(prob_tok.mean().item()),
            "margin": float(margin.mean().item()),
            "n_answer_tokens": int(end - start)}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="答案 token 的逐头输出干预")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--idx_json", required=True,
                   help="包含 input_ids (1,T) 的 JSON；可选 answer_start/end")
    p.add_argument("--answer_start", type=int, default=None)
    p.add_argument("--answer_end", type=int, default=None)
    p.add_argument("--layers", default=None, help="逗号分隔，缺省全部层")
    p.add_argument("--heads", default=None, help="逗号分隔，缺省全部头")
    p.add_argument("--scope", default="answer", choices=("answer", "all"))
    p.add_argument("--modes", default="zero,permute")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)
    from .model import load_model
    model, _ = load_model(args.ckpt, args.device)
    model.eval()
    with open(args.idx_json, encoding="utf-8") as f:
        payload = json.load(f)
    idx = torch.as_tensor(payload["input_ids"], dtype=torch.long,
                          device=args.device)
    start = args.answer_start if args.answer_start is not None else payload.get(
        "answer_start")
    end = args.answer_end if args.answer_end is not None else payload.get(
        "answer_end")
    if start is None:
        raise SystemExit("必须提供 answer_start（参数或 JSON）")
    layers = ([int(x) for x in args.layers.split(",") if x.strip()]
              if args.layers else None)
    heads = ([int(x) for x in args.heads.split(",") if x.strip()]
             if args.heads else None)
    modes = [x.strip() for x in args.modes.split(",") if x.strip()]
    rep = head_output_intervention_scan(
        model, idx, int(start), None if end is None else int(end),
        layers=layers, heads=heads, scope=args.scope, modes=modes)
    text = json.dumps(rep, ensure_ascii=False, indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"[output_ablation] -> {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
