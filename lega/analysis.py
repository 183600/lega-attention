"""门控行为机制分析 —— 定量检验核心风险：“注意力很集中 ≠ 找对了信息”。

方法：用带精确 gold 答案跨度的 RULER 样例做单次全量前向（need_stats=True），
对每个 (层, 头, 问题区查询位) 收集：
  max_p   本次注意力最大概率（集中度）
  hit     注意力落点 argmax 是否落在答案数值 token 跨度内（是否找对）
  g/logit 门控值与门控 logit（gated 变体）
  qterm   查询项 w·q（若有）；c = log n + log max_p（LEGA 家族）

关键指标：
  misleading_rate      集中但看错: P(max_p > τ 且 ¬hit)
  p_hit_given_conc     集中时找对率: P(hit | max_p > τ)
  auroc_logit/qterm/c  门控分数区分 hit/¬hit 的 AUROC
                       —— lega 的 auroc_logit vs auroc_qterm 直接回答：
                       “加入长度校准的注意力统计，是否比只看查询更能判断该不该写入”

用法:
  python -m lega.analysis --ckpt out/small/lega/model.pt --data_dir data/wikitext103 \
      --out_dir results/small
"""
import argparse
import csv
import json
import math
import os
from random import Random
from typing import Dict, List, Optional

import numpy as np
import torch

from .attention import VARIANTS, VARIANT_CN
from .data import load_tokenizer
from .model import load_model
from .ruler import (build_niah, build_vt, load_corpus_sents,
                    ContextTooSmall, _stable_hash)
from .utils import auroc, get_device, pearson, save_json

ANALYSIS_TASKS = ("niah_synth", "niah_corpus", "niah_multikey", "vt")  # fwe 无单一 needle 跨度


def _example(task: str, tok, ctx: int, rng: Random, corpus_sents: Optional[List[str]] = None,
             vt_chains: int = 4, vt_depth: int = 4):
    """构造单个带 gold 跨度的分析样例（fwe 无单一 needle，不参与）。"""
    if task == "niah_synth":
        return build_niah(tok, ctx, rng, mode="synth")
    if task == "niah_corpus":
        if not corpus_sents:
            return None
        return build_niah(tok, ctx, rng, mode="corpus", corpus_sents=corpus_sents)
    if task == "niah_multikey":
        return build_niah(tok, ctx, rng, mode="synth", multikey=True)
    if task == "vt":
        return build_vt(tok, ctx, rng, n_chains=vt_chains, depth=vt_depth)
    return None


@torch.inference_mode()
def collect_records(model, tok, device: str, examples: List[Dict], task_of: List[str],
                    ctx_of: List[int]) -> List[Dict]:
    """逐样例全量前向，展平出逐 (层,头,查询位) 记录。"""
    variant = model.variant
    records: List[Dict] = []
    for ex_idx, (ex, task, ctx) in enumerate(zip(examples, task_of, ctx_of)):
        # 先经 NumPy 物化再转 tensor：torch.tensor(list[, device=cuda]) 会逐元素
        # 装箱并做一次同步页式拷贝；asarray+from_numpy 避免一次多余的数据复制。
        ids = torch.from_numpy(np.asarray(ex["ids"], dtype=np.int64)[None])
        ids = ids.to(device, non_blocking=True) if device.startswith("cuda") else ids
        # logits 不参与机制分析；只算最后位置
        out = model(ids, need_stats=True, last_logits_only=True)
        if not out["gate_stats"]:
            continue
        qs, qe = ex["query_span"]
        # hit 以答案数值的精确 token 跨度为准；needle_span（整句/整行）仅作兼容
        # 与人工排查用途，避免把注意力落在模板停用词上误判为“找对”。
        gold_span = ex.get("value_span") or ex.get("needle_span")
        ns, ne = gold_span if gold_span else (None, None)
        width = qe - qs
        qpos = list(range(qs, qe))
        # 只切片回传 query 区间：长上下文下 T 远大于问题区长度，
        # 避免每层把整行 (H, T) 统计搬到 CPU。各层的切片统计 + hit 掩码先在
        # 设备端各拼为一张 (N, H, width) 张量，再把全部层堆叠一次后单次 D2H：
        # 原先每层各同步一次（乃至更早逐字段各拷贝一次），逐样例分析长上下文
        # 时这些同步会串行化整个循环；各层字段完全一致（同模型同变体），堆叠
        # 维度天然对齐。
        per_layer = []
        for st in out["gate_stats"]:
            argmax_d = st["argmax"][0][:, qs:qe]                          # (H, width) 设备端
            packed = [st["max_p"][0][:, qs:qe].float()]
            if ns is not None:
                packed.append(((argmax_d >= 0) & (argmax_d >= ns)
                               & (argmax_d < ne)).float())
            if st.get("g") is not None:
                packed.append(st["g"][0][:, qs:qe].float())
            if st.get("logit") is not None:
                packed.append(st["logit"][0][:, qs:qe].float())
            if st.get("qterm") is not None:
                packed.append(st["qterm"][0][:, qs:qe].float())
            # log_n 为 (width,)：设备端广播进同一张表，避免单独一次拷贝
            packed.append(st["log_n"][qs:qe].float().expand_as(packed[0]))
            per_layer.append(torch.stack(packed))
        merged_all = torch.stack(per_layer).cpu().numpy()                 # (L, N, H, width)
        for st, merged in zip(out["gate_stats"], merged_all):
            layer = st["layer"]
            it = iter(merged)
            max_p = next(it)                                              # (H, width)
            hit_m = next(it) if ns is not None else None
            g = next(it) if st.get("g") is not None else None
            logit = next(it) if st.get("logit") is not None else None
            qterm = next(it) if st.get("qterm") is not None else None
            log_n = next(it)[0]                                           # (width,)
            # hit 掩码已在设备端判好并随合并表一次带回；矩阵整体 tolist()
            # 转 Python 标量，避免逐元素 numpy 标量装箱（每条记录 ~10 次）。
            hit_l = hit_m.tolist() if hit_m is not None else None
            mp_l, ln_l = max_p.tolist(), log_n.tolist()
            g_l = g.tolist() if g is not None else None
            logit_l = logit.tolist() if logit is not None else None
            qterm_l = qterm.tolist() if qterm is not None else None
            use_c_full = variant in ("lega", "lega_no_query")
            use_c_nolen = not use_c_full and variant == "lega_no_len"
            for h in range(model.n_heads):
                mp_h = mp_l[h]
                hit_h = hit_l[h] if hit_l is not None else None
                g_h = g_l[h] if g_l is not None else None
                logit_h = logit_l[h] if logit_l is not None else None
                qterm_h = qterm_l[h] if qterm_l is not None else None
                for i in range(width):
                    mp_i, ln_i = mp_h[i], ln_l[i]
                    rec = {
                        "task": task, "ctx": ctx, "example": ex_idx, "layer": layer,
                        "head": h, "query_pos": qpos[i], "max_p": mp_i,
                        "log_n": ln_i,
                        "hit": hit_h[i] if hit_h is not None else None,
                    }
                    if g_h is not None:
                        rec["g"] = g_h[i]
                    if logit_h is not None:
                        rec["logit"] = logit_h[i]
                    if qterm_h is not None:
                        rec["qterm"] = qterm_h[i]
                    if use_c_full:
                        rec["c"] = ln_i + math.log(max(mp_i, 1e-30))
                    elif use_c_nolen:
                        rec["c"] = math.log(max(mp_i, 1e-30))
                    records.append(rec)
        del out
    return records


def summarize(records: List[Dict], variant: str, tau: float) -> Dict:
    n = len(records)
    if n == 0:
        return {"variant": variant, "n_records": 0}
    hit = np.array([np.nan if r.get("hit") is None else r["hit"] for r in records], dtype=np.float64)
    max_p = np.array([r["max_p"] for r in records])
    conc = (max_p > tau).astype(np.float64)
    g = np.array([np.nan if r.get("g") is None else r["g"] for r in records], dtype=np.float64)
    logit = np.array([np.nan if r.get("logit") is None else r["logit"] for r in records], dtype=np.float64)
    qterm = np.array([np.nan if r.get("qterm") is None else r["qterm"] for r in records], dtype=np.float64)
    c = np.array([np.nan if r.get("c") is None else r["c"] for r in records], dtype=np.float64)
    log_n = np.array([r["log_n"] for r in records])

    n_conc = int(conc.sum())
    # 部分记录可能没有 needle 跨度（hit=None），统计时仅在有效样本上计算
    hv = ~np.isnan(hit)

    def _auroc(scores, labels, mask):
        m = mask & ~np.isnan(scores)
        if not m.any() or len(set(labels[m])) != 2:
            return None
        return auroc(scores[m], labels[m])

    summary = {
        "variant": variant,
        "variant_cn": VARIANT_CN[variant],
        "tau": tau,
        "n_records": n,
        "hit_rate": float(np.nanmean(hit)) if hv.any() else None,
        "conc_rate": float(conc.mean()),
        "p_hit_given_conc": (float(np.nanmean(hit[conc == 1]))
                             if n_conc and hv[conc == 1].any() else None),
        "p_hit_given_notconc": (float(np.nanmean(hit[conc == 0]))
                                if (conc == 0).any() and hv[conc == 0].any() else None),
        "misleading_rate": (float(((max_p[hv] > tau) & (hit[hv] == 0)).mean())
                            if hv.any() else None),
        # 与 misleading_rate 互补的条件风险：P(看错 | 注意力集中)。长上下文下
        # 集中事件本身很少时，前者会趋近 0，单看它会造成虚假的安全印象。
        "misleading_given_conc": ((float((hit[hv & (conc == 1)] == 0).mean())
                                   if int((hv & (conc == 1)).sum()) else None)),
        "mean_g_hit": (float(np.nanmean(g[hit == 1])) if (~np.isnan(g)).any() and (hit == 1).any() else None),
        "mean_g_miss": (float(np.nanmean(g[hit == 0])) if (~np.isnan(g)).any() and (hit == 0).any() else None),
        "auroc_logit": _auroc(logit, hit, hv),
        "auroc_qterm": _auroc(qterm, hit, hv),
        "auroc_c": _auroc(c, hit, hv),
        "auroc_maxp": _auroc(max_p, hit, hv),
        "auroc_logn": _auroc(log_n, hit, hv),
        "corr_c_g": (pearson(c, g) if (~np.isnan(c)).any() and (~np.isnan(g)).any() else None),
        "corr_logn_g": (pearson(log_n, g) if (~np.isnan(g)).any() else None),
        "corr_qterm_g": (pearson(qterm, g) if (~np.isnan(qterm)).any() and (~np.isnan(g)).any() else None),
    }
    # 逐头统计
    per_head = []
    by_lh: Dict[tuple, List[Dict]] = {}
    for r in records:
        by_lh.setdefault((r["layer"], r["head"]), []).append(r)
    for (layer, head), rs in sorted(by_lh.items()):
        gg = np.array([np.nan if r.get("g") is None else r["g"] for r in rs], dtype=np.float64)
        hh = np.array([np.nan if r.get("hit") is None else r["hit"] for r in rs], dtype=np.float64)
        cc = np.array([np.nan if r.get("c") is None else r["c"] for r in rs], dtype=np.float64)
        qq = np.array([np.nan if r.get("qterm") is None else r["qterm"] for r in rs], dtype=np.float64)
        ll = np.array([np.nan if r.get("logit") is None else r["logit"] for r in rs], dtype=np.float64)
        mp = np.array([r["max_p"] for r in rs])
        hv = ~np.isnan(hh)

        def _auroc(scores, labels, mask):
            m = mask & ~np.isnan(scores)
            if not m.any() or len(set(labels[m])) != 2:
                return None
            return auroc(scores[m], labels[m])

        per_head.append({
            "layer": layer, "head": head, "n": len(rs),
            "mean_g": float(np.nanmean(gg)) if (~np.isnan(gg)).any() else None,
            "mean_max_p": float(mp.mean()),
            "hit_rate": float(np.nanmean(hh)) if hv.any() else None,
            "corr_c_g": pearson(cc, gg) if (~np.isnan(cc)).any() and (~np.isnan(gg)).any() else None,
            "corr_qterm_g": pearson(qq, gg) if (~np.isnan(qq)).any() and (~np.isnan(gg)).any() else None,
            "auroc_logit": _auroc(ll, hh, hv),
            "auroc_qterm": _auroc(qq, hh, hv),
        })
    summary["per_head"] = per_head
    return summary


def main():
    p = argparse.ArgumentParser(description="门控行为机制分析")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--n_examples", type=int, default=None)
    p.add_argument("--tau", type=float, default=None)
    p.add_argument("--ctx", type=int, default=None, help="分析用上下文长度，默认取 RULER 最长档")
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    device = get_device(args.device)
    model, ck = load_model(args.ckpt, device=device, use_sdpa_standard=False)
    tok = load_tokenizer(args.data_dir)
    variant = ck.get("variant", model.variant)

    ckpt_cfg = json.load(open(os.path.join(
        os.path.dirname(os.path.abspath(args.ckpt)), "config_used.json"),
        encoding="utf-8")) if os.path.exists(os.path.join(
            os.path.dirname(os.path.abspath(args.ckpt)), "config_used.json")) else {}
    acfg = ckpt_cfg.get("analysis", {})
    rcfg = ckpt_cfg.get("ruler", {})
    n_examples = args.n_examples or int(acfg.get("n_examples", 30))
    tau = args.tau if args.tau is not None else float(acfg.get("tau", 0.5))
    ctx = args.ctx if args.ctx is not None \
        else max(int(x) for x in rcfg.get("ctx_lens", [model.block_size]))
    # run_ruler 为生成预留 max_new 个位置；机制分析必须使用同一个实际 room_ctx，
    # 否则机制统计与 RULER 得分并不来自同一批样本。
    max_new = int(rcfg.get("max_new", 12))
    ctx = min(ctx, model.block_size - max_new - 1)
    ruler_seed = int(rcfg.get("seed", 1234))

    sents = load_corpus_sents(args.data_dir)
    examples, task_of, ctx_of = [], [], []
    skipped_tasks = []
    for i in range(n_examples):
        for task in ANALYSIS_TASKS:
            # 与 lega.ruler.run_ruler 的播种公式完全相同：EM 分数与逐头机制
            # 记录对应到同一批样例，而不是两个独立抽样。
            rng = Random(ruler_seed * 100003 + _stable_hash(task) % 99991 + ctx * 31 + i)
            try:
                ex = _example(task, tok, ctx, rng, corpus_sents=sents,
                              vt_chains=int(rcfg.get("vt_chains", 4)),
                              vt_depth=int(rcfg.get("vt_depth", 4)))
            except ContextTooSmall:
                # 预期的可跳过情况（上下文长度不足）；其余异常说明构造逻辑出错，应暴露。
                continue
            if ex is None:
                if task not in skipped_tasks:
                    skipped_tasks.append(task)
                continue
            if len(ex["ids"]) > ctx:
                continue
            examples.append(ex)
            task_of.append(task)
            ctx_of.append(ctx)

    if not examples:
        print("[analysis] 没有可分析的样例（ctx 过小或缺少语料），跳过")
        save_json({"variant": variant, "n_records": 0, "note": "no examples",
                   "skipped_tasks": skipped_tasks}, os.path.join(
            args.out_dir, f"analysis_{variant}.json"))
        return

    print(f"[analysis] variant={variant} ctx={ctx} 样例数={len(examples)} tau={tau}")
    records = collect_records(model, tok, device, examples, task_of, ctx_of)
    summary = summarize(records, variant, tau)
    if skipped_tasks:
        summary["skipped_tasks"] = skipped_tasks

    os.makedirs(args.out_dir, exist_ok=True)
    # 逐记录 CSV（供画图与复核）
    csv_path = os.path.join(args.out_dir, f"analysis_records_{variant}.csv")
    if records:
        fields = sorted({k for r in records for k in r})
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(records)
    save_json(summary, os.path.join(args.out_dir, f"analysis_{variant}.json"))

    def fmt(x):
        return "n/a" if x is None else f"{x:.3f}"
    print(f"[analysis] hit_rate={fmt(summary.get('hit_rate'))} "
          f"conc_rate={fmt(summary.get('conc_rate'))} "
          f"P(hit|conc)={fmt(summary.get('p_hit_given_conc'))} "
          f"misleading={fmt(summary.get('misleading_rate'))}")
    if summary.get("auroc_logit") is not None:
        print(f"[analysis] AUROC logit={fmt(summary.get('auroc_logit'))} "
              f"qterm={fmt(summary.get('auroc_qterm'))} c={fmt(summary.get('auroc_c'))} "
              f"maxp={fmt(summary.get('auroc_maxp'))}")
    print(f"[analysis] 完成 -> {args.out_dir}")


if __name__ == "__main__":
    main()
