"""门控行为机制分析 —— 定量检验核心风险：“注意力很集中 ≠ 找对了信息”。

方法：用带精确 gold 答案跨度的 RULER 样例做单次全量前向（need_stats=True；
等长样例合并为一批，因果注意力逐行独立，统计与单样例逐位一致），
对每个 (层, 头, 问题区查询位) 收集：
  max_p   本次注意力最大概率（集中度）
  hit     注意力落点 argmax 是否落在答案数值 token 跨度内（是否找对）
  g/logit 门控值与门控 logit（gated 变体）
  qterm   查询项 w·q（若有）；c = log n + log max_p（lega/lega_no_query），
          去长度项或只保留集中度的变体 c = log max_p

关键指标：
  misleading_rate      集中但看错: P(max_p > τ 且 ¬hit)
  p_hit_given_conc     集中时找对率: P(hit | max_p > τ)
  auroc_logit/qterm/c  门控分数区分 hit/¬hit 的 AUROC
                       —— lega 的 auroc_logit vs auroc_qterm 直接回答：
                       “加入长度校准的注意力统计，是否比只看查询更能判断该不该写入”

稳健性协议（防单一阈值/虚假精度）：
  tau_sensitivity          misleading/P(hit|集中) 系指标在多档 τ 上并报，
                           主结论不得只挂在单一任意选定的阈值上；
  auroc_cluster_bootstrap  以样例为聚类单元的 bootstrap 置信区间——同一
                           样例的 (层,头,查询位) 记录共享难度，不是独立
                           观测；逐重采样同步给出 logit−qterm 差的区间，
                           供 run_all 的跨种子配对检验交叉印证。

用法:
  python -m lega.analysis --ckpt out/small/lega/model.pt --data_dir data/wikitext103 \
      --out_dir results/small
"""
import argparse
import contextlib
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
from .repro import load_provenance
from .ruler import (build_niah, build_vt, load_corpus_sents,
                    ContextTooSmall, _stable_hash)
from .utils import (auroc, enable_tf32, get_device, pearson, save_json,
                    set_seed)

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
                    ctx_of: List[int], amp: bool = True,
                    batch_size: int = 8) -> List[Dict]:
    """全量前向，展平出逐 (层,头,查询位) 记录。

    等长样例合并成批前向：因果注意力的每行只依赖自身前缀，批内各行
    统计与单样例前向逐位一致。记录最终按样例原始顺序展平返回。

    CUDA 上默认启用 bf16 autocast（与 train/evaluate 的评测精度口径一致）；
    统计量在设备端升 fp32 后再聚合，不受 autocast 影响。
    """
    variant = model.variant
    records_by_ex: List[List[Dict]] = [[] for _ in examples]
    amp_ctx = (torch.autocast("cuda", dtype=torch.bfloat16)
               if (amp and device.startswith("cuda")) else contextlib.nullcontext())
    use_c_full = variant in ("lega", "lega_no_query")
    use_c_nolen = (not use_c_full
                   and variant in ("lega_no_len", "lega_conc_only"))
    # 按提示长度分组：只有等长样例可堆叠；每组内再按 batch_size 切批。
    order = sorted(range(len(examples)), key=lambda k: len(examples[k]["ids"]))
    bs = max(1, int(batch_size))
    g0 = 0
    while g0 < len(order):
        n_len = len(examples[order[g0]]["ids"])
        g1 = g0 + 1
        while (g1 < len(order) and len(examples[order[g1]]["ids"]) == n_len
               and g1 - g0 < bs):
            g1 += 1
        idxs = order[g0:g1]
        g0 = g1
        # 等长样例分组后可整表转 tensor，再按设备搬移。
        ids = torch.from_numpy(
            np.asarray([examples[k]["ids"] for k in idxs], dtype=np.int64))
        if device.startswith("cuda"):
            ids = ids.to(device, non_blocking=True)
        # logits 不参与机制分析；只算最后位置
        with amp_ctx:
            out = model(ids, need_stats=True, last_logits_only=True)
        if not out["gate_stats"]:
            continue
        for r, ex_idx in enumerate(idxs):
            ex = examples[ex_idx]
            task, ctx = task_of[ex_idx], ctx_of[ex_idx]
            qs, qe = ex["query_span"]
            # hit 以答案数值的精确 token 跨度为准；needle_span（整句/整行）仅作兼容
            # 与人工排查用途，避免把注意力落在模板停用词上误判为“找对”。
            gold_span = ex.get("value_span") or ex.get("needle_span")
            ns, ne = gold_span if gold_span else (None, None)
            width = qe - qs
            qpos = list(range(qs, qe))
            # 只回传 query 区间的切片统计；各层字段完全一致（同模型同变体），
            # 堆叠维度天然对齐。
            per_layer = []
            for st in out["gate_stats"]:
                argmax_d = st["argmax"][r][:, qs:qe]                      # (H, width) 设备端
                packed = [st["max_p"][r][:, qs:qe].float()]
                if ns is not None:
                    packed.append(((argmax_d >= 0) & (argmax_d >= ns)
                                   & (argmax_d < ne)).float())
                if st.get("g") is not None:
                    packed.append(st["g"][r][:, qs:qe].float())
                if st.get("logit") is not None:
                    packed.append(st["logit"][r][:, qs:qe].float())
                if st.get("qterm") is not None:
                    packed.append(st["qterm"][r][:, qs:qe].float())
                # log_n 为 (width,)：设备端广播进同一张表
                packed.append(st["log_n"][qs:qe].float().expand_as(packed[0]))
                per_layer.append(torch.stack(packed))
            merged_all = torch.stack(per_layer).cpu().numpy()             # (L, N, H, width)
            for st, merged in zip(out["gate_stats"], merged_all):
                layer = st["layer"]
                it = iter(merged)
                max_p = next(it)                                          # (H, width)
                hit_m = next(it) if ns is not None else None
                g = next(it) if st.get("g") is not None else None
                logit = next(it) if st.get("logit") is not None else None
                qterm = next(it) if st.get("qterm") is not None else None
                log_n = next(it)[0]                                       # (width,)
                # hit 掩码随合并表一次带回；矩阵整体 tolist() 转 Python 标量。
                hit_l = hit_m.tolist() if hit_m is not None else None
                mp_l, ln_l = max_p.tolist(), log_n.tolist()
                g_l = g.tolist() if g is not None else None
                logit_l = logit.tolist() if logit is not None else None
                qterm_l = qterm.tolist() if qterm is not None else None
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
                        records_by_ex[ex_idx].append(rec)
        del out
    return [rec for rs in records_by_ex for rec in rs]


def _record_arrays(records: List[Dict], key: str) -> np.ndarray:
    """提取记录中标量字段；缺失记 NaN（与 summarize 的 None 口径一致）。"""
    return np.array([np.nan if r.get(key) is None else r[key]
                     for r in records], dtype=np.float64)


def _conc_stats(hit: np.ndarray, max_p: np.ndarray, tau: float) -> Dict:
    """给定集中度阈值 τ 的全部条件统计（misleading 系与 P(hit|集中) 系）。

    hit 含 NaN（无 gold 跨度的记录）时仅在有效样本上计算；
    集中但看错的两口径互补：misleading_rate = P(集中 ∧ 看错) 会随长上下文
    下集中事件变稀而趋 0，造成虚假安全；misleading_given_conc =
    P(看错 | 集中) 才是"集中时有多少比例看错"的条件风险。
    """
    conc = (max_p > tau).astype(np.float64)
    n_conc = int(conc.sum())
    hv = ~np.isnan(hit)
    return {
        "conc_rate": float(conc.mean()),
        "p_hit_given_conc": (float(np.nanmean(hit[conc == 1]))
                             if n_conc and hv[conc == 1].any() else None),
        "p_hit_given_notconc": (float(np.nanmean(hit[conc == 0]))
                                if (conc == 0).any() and hv[conc == 0].any()
                                else None),
        "misleading_rate": (float(((max_p[hv] > tau) & (hit[hv] == 0)).mean())
                            if hv.any() else None),
        "misleading_given_conc": (
            float((hit[hv & (conc == 1)] == 0).mean())
            if int((hv & (conc == 1)).sum()) else None),
    }


def tau_sensitivity(records: List[Dict], taus: List[float]) -> Dict:
    """misleading/P(hit|集中) 系指标对阈值 τ 的敏感性扫描。

    主结论不能建立单个任意选定的 τ 上：若 τ=0.5 时"集中但看错率"很低、
    τ=0.3 时却很高，说明"注意力沉降即可靠"的叙事只在特定阈值下成立。
    逐 τ 输出全部条件统计，供汇总表并列展示与人工审视。
    """
    hit = _record_arrays(records, "hit")
    max_p = _record_arrays(records, "max_p")
    out: Dict = {"taus": [float(t) for t in taus]}
    fields = ("conc_rate", "p_hit_given_conc", "p_hit_given_notconc",
              "misleading_rate", "misleading_given_conc")
    for f in fields:
        out[f] = []
    for tau in out["taus"]:
        s = _conc_stats(hit, max_p, tau)
        for f in fields:
            out[f].append(s[f])
    return out


def auroc_cluster_bootstrap(records: List[Dict], n_resamples: int = 1000,
                            ci: float = 0.95, seed: int = 0) -> Dict:
    """按样例聚类的 bootstrap：给 AUROC 与 logit−qterm 差一个置信区间。

    机制分析的记录不是独立样本：同一样例的全部 (层, 头, 查询位) 记录共享该
    样例的难度、gold 位置与构造随机性，把记录当独立观测会严重虚增 AUROC
    的有效样本量、让区间过窄。这里以样例（example 字段）为聚类单元有放回
    重采样，聚类内相关性完整保留，区间回答的是"换一批评测样例，指标会
    波动多少"。logit−qterm 差的区间直接服务于核心机制主张（完整门控
    logit 是否比仅查询项更能判断该不该写入），逐重采样同步计算避免
    两个边缘区间相减的谬误。
    """
    clusters: Dict[int, List[Dict]] = {}
    for r in records:
        if r.get("hit") is None:
            continue
        clusters.setdefault(r["example"], []).append(r)
    ex_ids = sorted(clusters)
    out = {"n_examples": len(ex_ids), "n_resamples": int(n_resamples),
           "ci_level": float(ci), "ci95": {}, "n_valid_resamples": None}
    if len(ex_ids) < 2:
        return out
    score_keys = [k for k in ("logit", "qterm", "c")
                  if any(k in r for cl in clusters.values() for r in cl)]
    if not score_keys:
        return out
    labels = [np.array([float(r["hit"]) for r in clusters[e]]) for e in ex_ids]
    scores = {k: [np.array([float(r[k]) for r in clusters[e]]) for e in ex_ids]
              for k in score_keys}

    def _auroc_safe(sc, lb):
        if np.isnan(sc).any() or len(set(lb.tolist())) != 2:
            return None
        v = auroc(sc, lb)
        return v if np.isfinite(v) else None

    rng = np.random.default_rng(seed)
    samples: Dict = {k: [] for k in score_keys}
    samples["logit_minus_qterm"] = []
    n_valid = 0
    for _ in range(int(n_resamples)):
        idx = rng.integers(0, len(ex_ids), len(ex_ids))
        lb = np.concatenate([labels[i] for i in idx])
        vals = {}
        for k in score_keys:
            v = _auroc_safe(np.concatenate([scores[k][i] for i in idx]), lb)
            vals[k] = v
            if v is not None:
                samples[k].append(v)
        if "logit" in vals and "qterm" in vals \
                and vals["logit"] is not None and vals["qterm"] is not None:
            samples["logit_minus_qterm"].append(vals["logit"] - vals["qterm"])
            n_valid += 1
    alpha = (1.0 - float(ci)) / 2.0
    for k, vals_list in samples.items():
        if len(vals_list) < 10:
            out["ci95"][k] = None
            continue
        lo, hi = np.quantile(np.asarray(vals_list), [alpha, 1.0 - alpha])
        out["ci95"][k] = [float(lo), float(hi)]
    out["n_valid_resamples"] = n_valid
    return out


def summarize(records: List[Dict], variant: str, tau: float) -> Dict:
    n = len(records)
    if n == 0:
        return {"variant": variant, "n_records": 0}
    hit = np.array([np.nan if r.get("hit") is None else r["hit"] for r in records], dtype=np.float64)
    max_p = np.array([r["max_p"] for r in records])
    g = np.array([np.nan if r.get("g") is None else r["g"] for r in records], dtype=np.float64)
    logit = np.array([np.nan if r.get("logit") is None else r["logit"] for r in records], dtype=np.float64)
    qterm = np.array([np.nan if r.get("qterm") is None else r["qterm"] for r in records], dtype=np.float64)
    c = np.array([np.nan if r.get("c") is None else r["c"] for r in records], dtype=np.float64)
    log_n = np.array([r["log_n"] for r in records])

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
    }
    summary.update(_conc_stats(hit, max_p, tau))
    summary.update({
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
    })
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
    p.add_argument("--tau_grid", default=None,
                   help="逗号分隔的敏感性扫描阈值；默认取配置 analysis.tau_grid "
                        "（[0.3, 0.5, 0.7]），并并入主阈值 tau")
    p.add_argument("--bootstrap_resamples", type=int, default=None,
                   help="按样例聚类的 AUROC bootstrap 重采样次数；"
                        "默认取配置 analysis.bootstrap_resamples（1000），0 关闭")
    p.add_argument("--ctx", type=int, default=None, help="分析用上下文长度，默认取 RULER 最长档")
    p.add_argument("--suffix", default="",
                   help="输出文件名后缀（如 _seed1337），多种子运行避免互相覆盖")
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    device = get_device(args.device)
    enable_tf32(device)
    model, ck = load_model(args.ckpt, device=device, use_sdpa_standard=False)
    tok = load_tokenizer(args.data_dir)
    variant = ck.get("variant", model.variant)
    # 固定全局随机源；样例构造本身使用显式种子的 Random，不受影响
    set_seed(int(ck.get("seed")) if ck.get("seed") is not None else 0)
    provenance = load_provenance(
        os.path.dirname(os.path.abspath(args.ckpt)), args.data_dir)

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
        save_json({"variant": variant, "seed": ck.get("seed"), "n_records": 0,
                   "note": "no examples", "skipped_tasks": skipped_tasks,
                   **provenance},
                  os.path.join(args.out_dir,
                               f"analysis_{variant}{args.suffix}.json"))
        return

    print(f"[analysis] variant={variant} ctx={ctx} 样例数={len(examples)} tau={tau}")
    records = collect_records(model, tok, device, examples, task_of, ctx_of,
                              amp=bool(ckpt_cfg.get("amp", True)))
    summary = summarize(records, variant, tau)
    # 阈值敏感性：misleading/P(hit|集中) 的结论不能只挂在单一 τ 上
    if args.tau_grid is not None:
        grid = [float(x) for x in args.tau_grid.split(",") if x.strip()]
    else:
        grid = [float(x) for x in acfg.get("tau_grid", [0.3, 0.5, 0.7])]
    summary["tau_sensitivity"] = tau_sensitivity(records, sorted(set(grid) | {float(tau)}))
    # 聚类 bootstrap：同一记录池的 AUROC 置信区间（样例为聚类单元）
    n_boot = (int(args.bootstrap_resamples)
              if args.bootstrap_resamples is not None
              else int(acfg.get("bootstrap_resamples", 1000)))
    if n_boot > 0:
        summary["auroc_cluster_bootstrap"] = auroc_cluster_bootstrap(
            records, n_resamples=n_boot,
            seed=int(ck.get("seed")) if ck.get("seed") is not None else 0)
    # 训练种子随分析结果落盘：多种子聚合与配对检验需要它对齐同一批 run
    summary["seed"] = ck.get("seed")
    summary.update(provenance)
    if skipped_tasks:
        summary["skipped_tasks"] = skipped_tasks

    os.makedirs(args.out_dir, exist_ok=True)
    # 逐记录 CSV（供画图与复核）
    csv_path = os.path.join(args.out_dir,
                            f"analysis_records_{variant}{args.suffix}.csv")
    if records:
        fields = sorted({k for r in records for k in r})
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(records)
    save_json(summary,
              os.path.join(args.out_dir, f"analysis_{variant}{args.suffix}.json"))

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
    boot = summary.get("auroc_cluster_bootstrap") or {}
    ci = (boot.get("ci95") or {}).get("logit_minus_qterm")
    if ci:
        print(f"[analysis] AUROC(logit−qterm) 样例聚类 bootstrap 95% CI: "
              f"[{ci[0]:.3f}, {ci[1]:.3f}]"
              f"（n_examples={boot.get('n_examples')}，"
              f"n_resamples={boot.get('n_resamples')}）")
    print(f"[analysis] 完成 -> {args.out_dir}")


if __name__ == "__main__":
    main()
