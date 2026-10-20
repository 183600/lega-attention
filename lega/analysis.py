"""门控行为机制分析 —— 定量检验核心风险：“注意力很集中 ≠ 找对了信息”。

方法：用带精确 gold 答案跨度的 RULER 样例做单次全量前向（need_stats=True；
等长样例合并为一批，因果注意力逐行独立，统计与单样例逐位一致），
对每个 (层, 头, 问题区查询位) 收集：
  max_p   本次注意力最大概率（集中度）
  hit     注意力落点 argmax 是否落在答案数值 token 跨度内（是否找对）
  g/logit 门控值与门控 logit（gated 变体）
  qterm   查询项 w·q（若有）；c = log n + log max_p（含集中度项的全部变体，
          包括去长度项/仅集中度消融——a·c 的长度归一化不能随 b·log n 一起消失）

关键指标：
  misleading_rate      集中但看错: P(max_p > τ 且 ¬hit)
  p_hit_given_conc     集中时找对率: P(hit | max_p > τ)
  auroc_logit/qterm/c  门控分数区分 hit/¬hit 的 AUROC
                       —— lega 的 auroc_logit vs auroc_qterm 直接回答：
                       “加入长度校准的注意力统计，是否比只看查询更能判断该不该写入”
  auprc_*              同分数的 PR-AUC（average precision）：hit 判别通常
                       类别极不平衡（小模型 hit 稀少），AUROC 会被巨大负类
                       基数抬高；正类率 <5% 时以 PR-AUC 为主（其随机基线 =
                       正类率 hit_base_rate），两者必须并报
  gate_f1              g>0.5 决策点的 F1，显式给 macro/micro/weighted 三口径
                       （不平衡数据三者差异大，macro 对稀有类最诚实）
  gate_f1_trivial      恒写/恒不写两个固定规则的 F1（无阈值“稻草人”对照）：
                       不平衡下学习到的门控 F1 必须比多数类规则更有信息量，
                       否则门控的“决策价值”无从谈起
  calibration_g        门控 g 的概率校准：Brier（严格评分规则）、ECE/MCE
                       （等宽分箱）与常数基率 Brier 基线。排序指标（AUROC/
                       PR-AUC）回答“能不能区分”，校准回答“g=0.8 时命中率
                       是否真的约 80%”；g 被当作软权重/概率使用，两者都要看。
  linear_probe         同特征集（log max_p、log n、qterm）的 L2 逻辑回归
                       **简单监督基线**：按样例分组并按标签分层的嵌套 CV，
                       标准化统计只用训练折；λ 在内层 OOF 上按主指标
                       （PR-AUC/AUROC）选择、阈值在内层 OOF 上按 macro-F1
                       选择，最终 OOF 不再参与选参。门控 logit 在语言建模
                       中训练、从未见过 hit 标签；探针直接用 hit 标签监督
                       训练。对照语义：探针显著更好属预期，门控不能显著
                       优于探针时不得把“更能判断该不该写入”归因给门控
                       结构。详见 lega/probe.py

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
from .probe import (DEFAULT_FOLDS as _PROBE_FOLDS,
                    DEFAULT_L2_GRID as _PROBE_L2_GRID,
                    EXTREME_IMBALANCE_THRESHOLD as _EXTREME_IMBALANCE_THRESHOLD,
                    PRIMARY_AP_THRESHOLD as _PRIMARY_AP_THRESHOLD,
                    linear_probe)
from .repro import load_provenance
from .ruler import (build_niah, build_vt, load_corpus_sents_with_source,
                    ContextTooSmall, _stable_hash)
from .utils import (amp_autocast, auroc, average_precision,
                    calibration_metrics, enable_tf32, f1_scores, get_device,
                    pearson, roc_pr_aucs, save_json, set_seed)

ANALYSIS_TASKS = ("niah_synth", "niah_corpus", "niah_multikey", "vt")  # fwe 无单一 needle 跨度

# 命中判别（hit vs miss）正类率低于该值时以 PR-AUC 为主指标：此时负样本
# 基数使 ROC-AUC 过于乐观（FPR 增长缓慢）。阈值集中在 lega.probe 定义
# （线性探针基线复用同一口径），此处导入别名避免两处漂移。
# 门控决策阈值：g 为 sigmoid 输出，0.5 是“是否写入”的自然默认决策点；
# 显式落盘以便复现（F1 依赖阈值，不能隐式）。
_GATE_DECISION_THRESHOLD = 0.5


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
    amp_ctx = amp_autocast(device, bool(amp))
    use_c_full = variant in ("lega", "lega_no_query", "lega_no_len",
                             "lega_conc_only")
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
            ids = ids.to(device)
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
                per_layer.append(torch.stack(packed))
            merged_all = torch.stack(per_layer).cpu().numpy()             # (L, N, H, width)
            # log_n 为 (width,)：逐层扩成 (H, width) 再堆进合并表会白白多拷
            # H 倍数据；单独整表堆叠一次转回设备（单次传输），消费时按层取。
            log_ns_all = torch.stack([st["log_n"][qs:qe].float()
                                      for st in out["gate_stats"]]).cpu().tolist()
            for li, (st, merged) in enumerate(zip(out["gate_stats"], merged_all)):
                layer = st["layer"]
                it = iter(merged)
                max_p = next(it)                                          # (H, width)
                hit_m = next(it) if ns is not None else None
                g = next(it) if st.get("g") is not None else None
                logit = next(it) if st.get("logit") is not None else None
                qterm = next(it) if st.get("qterm") is not None else None
                ln_l = log_ns_all[li]                                     # (width,) 已是标量表
                # hit 掩码随合并表一次带回；矩阵整体 tolist() 转 Python 标量。
                hit_l = hit_m.tolist() if hit_m is not None else None
                mp_l = max_p.tolist()
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
                        records_by_ex[ex_idx].append(rec)
        del out
    return [rec for rs in records_by_ex for rec in rs]


def _finite_float(value):
    """把外部记录中的标量转为有限 float；缺失/类型错误/NaN/Inf 返回 None。"""
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) else None


def _record_arrays(records: List[Dict], key: str) -> np.ndarray:
    """提取记录中标量字段；缺失/非数值/非有限记 NaN（None 口径一致）。"""
    records = records if isinstance(records, (list, tuple)) else []
    vals = []
    for r in records:
        value = _finite_float(r.get(key)) if isinstance(r, dict) else None
        vals.append(np.nan if value is None else value)
    return np.array(vals, dtype=np.float64)


def _conc_stats(hit: np.ndarray, max_p: np.ndarray, tau: float) -> Dict:
    """给定集中度阈值 τ 的全部条件统计（misleading 系与 P(hit|集中) 系）。

    hit 含 NaN（无 gold 跨度的记录）时仅在有效样本上计算；
    集中但看错的两口径互补：misleading_rate = P(集中 ∧ 看错) 会随长上下文
    下集中事件变稀而趋 0，造成虚假安全；misleading_given_conc =
    P(看错 | 集中) 才是"集中时有多少比例看错"的条件风险。
    """
    valid = ~np.isnan(max_p)
    conc = (max_p > tau) & valid
    n_conc = int(conc.sum())
    hv = ~np.isnan(hit) & valid
    return {
        "conc_rate": (float(conc.sum() / valid.sum())
                      if valid.any() else None),
        "p_hit_given_conc": (float(np.nanmean(hit[conc]))
                             if n_conc and hv[conc].any() else None),
        "p_hit_given_notconc": (float(np.nanmean(hit[valid & ~conc]))
                                if (valid & ~conc).any()
                                and hv[valid & ~conc].any() else None),
        "misleading_rate": (float((conc[hv] & (hit[hv] == 0)).mean())
                            if hv.any() else None),
        "misleading_given_conc": (
            float((hit[hv & conc] == 0).mean())
            if int((hv & conc).sum()) else None),
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
    """按样例聚类的 bootstrap：给 AUROC、PR-AUC 及其 logit−qterm 差的置信区间。

    机制分析的记录不是独立样本：同一样例的全部 (层, 头, 查询位) 记录共享该
    样例的难度、gold 位置与构造随机性，把记录当独立观测会严重虚增 AUROC
    的有效样本量、让区间过窄。这里以样例（example 字段）为聚类单元有放回
    重采样，聚类内相关性完整保留，区间回答的是"换一批评测样例，指标会
    波动多少"。logit−qterm 差的区间直接服务于核心机制主张（完整门控
    logit 是否比仅查询项更能判断该不该写入），逐重采样同步计算避免
    两个边缘区间相减的谬误。

    同时给出 PR-AUC（average precision）族：命中判别通常极端不平衡
    （小模型 hit 稀少），此时 AUROC 的乐观偏差不可忽略，`ap_logit` 与
    `ap_logit_minus_qterm` 是机制结论在正确指标口径下的对照证据。
    键名沿用 `ci95`（历史字段），内部为 AUROC 与 AP 两族区间。
    """
    records = records if isinstance(records, (list, tuple)) else []
    clusters: Dict[int, List[Dict]] = {}
    for r in records:
        if not isinstance(r, dict):
            continue
        hit = _finite_float(r.get("hit"))
        example = r.get("example")
        if hit is None or hit not in (0.0, 1.0) or example is None:
            # 与 probe/summarize 同口径：hit 是二值标签，非 0/1 不能当作
            # 正类或负数参与 AUROC/AP，否则指标会越界或方向错误。
            continue
        try:
            hash(example)
        except TypeError:
            continue
        if isinstance(example, float) and not math.isfinite(example):
            continue
        rec = dict(r)
        rec["hit"] = hit
        clusters.setdefault(example, []).append(rec)
    try:
        ex_ids = sorted(clusters)
    except TypeError:
        ex_ids = sorted(clusters, key=lambda e: (type(e).__name__, repr(e)))
    out = {"n_examples": len(ex_ids), "n_resamples": int(n_resamples),
           "ci_level": float(ci), "ci95": {}, "n_valid_resamples": None}
    if len(ex_ids) < 2:
        return out
    score_keys = [k for k in ("logit", "qterm", "c")
                  if any(_finite_float(r.get(k)) is not None
                         for cl in clusters.values() for r in cl)]
    if not score_keys:
        return out
    labels = [np.array([r["hit"] for r in clusters[e]]) for e in ex_ids]
    scores = {}
    for k in score_keys:
        scores[k] = []
        for e in ex_ids:
            vals = [_finite_float(r.get(k)) for r in clusters[e]]
            scores[k].append(np.array(
                [np.nan if v is None else v for v in vals], dtype=np.float64))
    # 逐聚类预计算“是否含正/负类”：重采样时先按聚类标志判定是否可能
    # 恰有两类，再交给 roc_pr_aucs 按分数有限性逐记录过滤（与 summarize
    # 的点估计口径一致）。不能因为某个聚类含 NaN 分数就丢掉整个重采样：
    # 可用分数仍能估计指标，否则 CI 会在仅少量缺失时整体变成 null。
    has_pos = np.array([bool((lb == 1.0).any()) for lb in labels])
    has_neg = np.array([bool((lb == 0.0).any()) for lb in labels])

    def _finite(v):
        return v if (v is not None and np.isfinite(v)) else None

    rng = np.random.default_rng(seed)
    samples: Dict = {k: [] for k in score_keys}
    samples.update({f"ap_{k}": [] for k in score_keys})
    samples["logit_minus_qterm"] = []
    samples["ap_logit_minus_qterm"] = []
    n_valid = 0
    for _ in range(int(n_resamples)):
        idx = rng.integers(0, len(ex_ids), len(ex_ids))
        both = bool(has_pos[idx].any() and has_neg[idx].any())
        lb = None
        vals, aps = {}, {}
        for k in score_keys:
            if not both:
                vals[k] = aps[k] = None
                continue
            if lb is None:
                lb = np.concatenate([labels[i] for i in idx])
            sc = np.concatenate([scores[k][i] for i in idx])
            # 同一份重采样分数上用 roc_pr_aucs 计算逐位等价的 AUROC 与 AP。
            v, ap = roc_pr_aucs(sc, lb)
            v, ap = _finite(v), _finite(ap)
            vals[k] = v
            aps[k] = ap
            if v is not None:
                samples[k].append(v)
            if ap is not None:
                samples[f"ap_{k}"].append(ap)
        if vals.get("logit") is not None and vals.get("qterm") is not None:
            samples["logit_minus_qterm"].append(vals["logit"] - vals["qterm"])
            n_valid += 1
        if aps.get("logit") is not None and aps.get("qterm") is not None:
            samples["ap_logit_minus_qterm"].append(aps["logit"] - aps["qterm"])
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
    records = records if isinstance(records, (list, tuple)) else []
    n = len(records)
    if n == 0:
        return {"variant": variant, "n_records": 0}
    hit = _record_arrays(records, "hit")
    max_p = _record_arrays(records, "max_p")
    g = _record_arrays(records, "g")
    logit = _record_arrays(records, "logit")
    qterm = _record_arrays(records, "qterm")
    c = _record_arrays(records, "c")
    log_n = _record_arrays(records, "log_n")

    # 部分记录可能没有 needle 跨度（hit=None），统计时仅在有效样本上计算。
    # 外部/旧格式记录里的 hit 必须是 0/1；其他有限值（如 1.5、-1）不能
    # 冒充标签，否则 hit 率/条件概率/AUROC 会越界或把非正类当负类。
    hit[(hit != 0.0) & (hit != 1.0)] = np.nan
    hv = ~np.isnan(hit)

    def _auroc(scores, labels, mask):
        m = mask & ~np.isnan(scores)
        if not m.any() or len(set(labels[m])) != 2:
            return None
        return auroc(scores[m], labels[m])

    def _auprc(scores, labels, mask):
        """PR-AUC（average precision）；类别不平衡时的主指标（见模块说明）。"""
        m = mask & ~np.isnan(scores)
        if not m.any() or len(set(labels[m])) != 2:
            return None
        ap = average_precision(scores[m], labels[m])
        return float(ap) if np.isfinite(ap) else None

    hit_pos = int((hit == 1).sum())
    hit_neg = int((hit == 0).sum())
    base_rate = (hit_pos / (hit_pos + hit_neg)) if (hit_pos + hit_neg) else None
    # 命中判别的类别不平衡等级与主指标：AUROC 在负类占优时会过于乐观。
    if base_rate is None:
        imbalance = None
        primary = None
    elif base_rate <= _EXTREME_IMBALANCE_THRESHOLD:
        imbalance, primary = "extreme(≤1:1000)", "PR-AUC"
    elif base_rate < _PRIMARY_AP_THRESHOLD:
        imbalance, primary = "severe(<5%)", "PR-AUC"
    else:
        imbalance, primary = "moderate", "AUROC 与 PR-AUC 并报"
    summary = {
        "variant": variant,
        "variant_cn": VARIANT_CN.get(variant, variant),
        "tau": tau,
        "n_records": n,
        "hit_rate": float(np.nanmean(hit)) if hv.any() else None,
        "n_hit": hit_pos,
        "n_miss": hit_neg,
        # 正类率 = PR-AUC 的“随机猜”基线；低于 5% 时主指标改为 PR-AUC
        "hit_base_rate": base_rate,
        "imbalance_level": imbalance,
        "primary_detection_metric": primary,
        "detection_metric_note": (
            "hit/¬hit 为二分类；正类率即 hit 率。PR-AUC 的随机基线等于正类率。"
            "正类率 <5% 时 AUROC 会因负样本基数大而过于乐观，以 PR-AUC 为主；"
            "F1 三种平均口径并列于 gate_f1（macro/micro/weighted）。"),
    }
    summary.update(_conc_stats(hit, max_p, tau))
    summary.update({
        "mean_g_hit": (float(np.nanmean(g[hit == 1]))
                       if (~np.isnan(g[hit == 1])).any() else None),
        "mean_g_miss": (float(np.nanmean(g[hit == 0]))
                        if (~np.isnan(g[hit == 0])).any() else None),
        "auroc_logit": _auroc(logit, hit, hv),
        "auroc_qterm": _auroc(qterm, hit, hv),
        "auroc_c": _auroc(c, hit, hv),
        "auroc_maxp": _auroc(max_p, hit, hv),
        "auroc_logn": _auroc(log_n, hit, hv),
        "auprc_logit": _auprc(logit, hit, hv),
        "auprc_qterm": _auprc(qterm, hit, hv),
        "auprc_c": _auprc(c, hit, hv),
        "auprc_maxp": _auprc(max_p, hit, hv),
        "auprc_logn": _auprc(log_n, hit, hv),
        "corr_c_g": (pearson(c, g) if (~np.isnan(c)).any() and (~np.isnan(g)).any() else None),
        "corr_logn_g": (pearson(log_n, g) if (~np.isnan(g)).any() else None),
        "corr_qterm_g": (pearson(qterm, g) if (~np.isnan(qterm)).any() and (~np.isnan(g)).any() else None),
    })
    # 门控决策点（g > 0.5 ⇒ 预测“这次写入是对的”）的分类指标。
    # 只对有效（有 hit 标签）且有门控 g 的记录计算；F1 必须注明平均口径。
    gate_f1 = None
    gm = hv & ~np.isnan(g)
    if gm.any() and len(set(hit[gm])) == 2:
        pred = (g[gm] > _GATE_DECISION_THRESHOLD).astype(np.int64)
        f1s = f1_scores(hit[gm].astype(np.int64), pred, labels=[0, 1])
        gate_f1 = {
            "threshold": _GATE_DECISION_THRESHOLD,
            "n": f1s["n"],
            "macro": f1s["macro"],
            "micro": f1s["micro"],
            "weighted": f1s["weighted"],
            "per_class": {str(k): v for k, v in f1s["per_class"].items()},
        }
    summary["gate_f1"] = gate_f1
    # 决策终点的“稻草人”参照：恒写（全判 hit）/恒不写（全判 miss）两个
    # 固定规则的 F1。类别不平衡时学习到的门控 F1 必须对照多数类规则才能
    # 解读——micro/weighted 口径下“比恒写基线还差”的高 F1 并不罕见。
    gate_f1_trivial = None
    if gm.any() and len(set(hit[gm])) == 2:
        y = hit[gm].astype(np.int64)
        gate_f1_trivial = {"n": int(len(y)),
                           "note": "固定规则、无阈值：不平衡下 F1 的稻草人对照"}
        for name, pred in (("always_write", np.ones_like(y)),
                           ("never_write", np.zeros_like(y))):
            f1s_t = f1_scores(y, pred, labels=[0, 1])
            gate_f1_trivial[name] = {
                "macro": f1s_t["macro"], "micro": f1s_t["micro"],
                "weighted": f1s_t["weighted"],
                "per_class": {str(k): v
                              for k, v in f1s_t["per_class"].items()},
            }
    summary["gate_f1_trivial"] = gate_f1_trivial
    # 概率校准：g 是 sigmoid 输出、被当作“该不该写入”的软权重/概率使用。
    # AUROC/PR-AUC 只看排序；Brier/ECE 回答 g 的数值是否可信。常数基率
    # 预测的 Brier 是“无信息基线”，校准后的 Brier 必须对照它解读。
    calibration_g = None
    if gm.any():
        calibration_g = calibration_metrics(g[gm], hit[gm])
        summary["calibration_g"] = calibration_g
        summary["brier_g"] = calibration_g.get("brier")
        summary["ece_g"] = calibration_g.get("ece")
        summary["brier_g_constant"] = calibration_g.get("brier_constant")
        summary["brier_g_skill"] = calibration_g.get("brier_skill")
        summary["calibration_g_n"] = calibration_g.get("n")
    else:
        summary["calibration_g"] = None
        summary["brier_g"] = summary["ece_g"] = None
        summary["brier_g_constant"] = summary["brier_g_skill"] = None
        summary["calibration_g_n"] = None
    # 逐头统计：对整段数组按 (layer, head) 掩码切片。缺字段/非整数的
    # 记录无法归属到具体头，跳过该记录而不是让整份汇总崩溃。
    per_head = []
    layer_arr = _record_arrays(records, "layer")
    head_arr = _record_arrays(records, "head")
    # int64 强制转换对超范围浮点会饱和/回绕，把不同的层/头合并成同一
    # 组；层/头索引远小于 2^53，超出即可安全视为畸形记录。
    limit = float(2 ** 53)
    lh_ok = (np.isfinite(layer_arr) & np.isfinite(head_arr)
             & (layer_arr == np.floor(layer_arr))
             & (head_arr == np.floor(head_arr))
             & (np.abs(layer_arr) <= limit) & (np.abs(head_arr) <= limit))
    if lh_ok.any():
        layer_v = layer_arr[lh_ok].astype(np.int64)
        head_v = head_arr[lh_ok].astype(np.int64)
        # 一次分组得到每行的组号：逐组重扫全表求 (layer, head) 掩码在记录数
        # 大时是 O(N × 组数)；分组顺序仍为字典序，各组记录顺序不变。
        groups, group_idx = np.unique(
            np.stack([layer_v, head_v], axis=1), axis=0,
            return_inverse=True)
        group_idx = group_idx.reshape(-1)
        g_v, hit_v, c_v = g[lh_ok], hit[lh_ok], c[lh_ok]
        qterm_v, logit_v, max_p_v = qterm[lh_ok], logit[lh_ok], max_p[lh_ok]
    for gi, (layer, head) in enumerate(groups if lh_ok.any() else []):
        m = group_idx == gi
        gg, hh, cc = g_v[m], hit_v[m], c_v[m]
        qq, ll, mp = qterm_v[m], logit_v[m], max_p_v[m]
        hv = ~np.isnan(hh)

        def _auroc(scores, labels, mask):
            mk = mask & ~np.isnan(scores)
            if not mk.any() or len(set(labels[mk])) != 2:
                return None
            return auroc(scores[mk], labels[mk])

        def _auprc(scores, labels, mask):
            mk = mask & ~np.isnan(scores)
            if not mk.any() or len(set(labels[mk])) != 2:
                return None
            ap = average_precision(scores[mk], labels[mk])
            return float(ap) if np.isfinite(ap) else None

        per_head.append({
            "layer": int(layer), "head": int(head), "n": int(m.sum()),
            "mean_g": float(np.nanmean(gg)) if (~np.isnan(gg)).any() else None,
            "mean_max_p": (float(np.nanmean(mp))
                           if (~np.isnan(mp)).any() else None),
            "hit_rate": float(np.nanmean(hh)) if hv.any() else None,
            "corr_c_g": pearson(cc, gg) if (~np.isnan(cc)).any() and (~np.isnan(gg)).any() else None,
            "corr_qterm_g": pearson(qq, gg) if (~np.isnan(qq)).any() and (~np.isnan(gg)).any() else None,
            "auroc_logit": _auroc(ll, hh, hv),
            "auroc_qterm": _auroc(qq, hh, hv),
            "auprc_logit": _auprc(ll, hh, hv),
            "auprc_qterm": _auprc(qq, hh, hv),
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
    p.add_argument("--probe_folds", type=int, default=None,
                   help="线性探针的分组交叉验证折数（默认取配置 "
                        "analysis.probe_folds，5）")
    p.add_argument("--probe_l2", default=None,
                   help="逗号分隔的 L2 强度网格；默认取配置 "
                        "analysis.probe_l2_grid（0,0.01,0.1,1）")
    p.add_argument("--no_probe", action="store_true",
                   help="跳过线性探针简单基线（默认运行；跳过时汇总页会"
                        "缺少“门控 vs 最简监督线性模型”的对照）")
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

    ckpt_cfg_path = os.path.join(
        os.path.dirname(os.path.abspath(args.ckpt)), "config_used.json")
    ckpt_cfg = {}
    if os.path.exists(ckpt_cfg_path):
        try:
            with open(ckpt_cfg_path, encoding="utf-8") as f:
                ckpt_cfg = json.load(f)
        except (OSError, ValueError):
            ckpt_cfg = {}
        # 合法 JSON 但非对象（数组/标量）的配置按空配置处理，走默认值。
        if not isinstance(ckpt_cfg, dict):
            ckpt_cfg = {}
    acfg = ckpt_cfg.get("analysis")
    acfg = acfg if isinstance(acfg, dict) else {}
    rcfg = ckpt_cfg.get("ruler")
    rcfg = rcfg if isinstance(rcfg, dict) else {}
    n_examples = args.n_examples or int(acfg.get("n_examples", 30))
    tau = args.tau if args.tau is not None else float(acfg.get("tau", 0.5))
    ctx = args.ctx if args.ctx is not None \
        else max(int(x) for x in rcfg.get("ctx_lens", [model.block_size]))
    # run_ruler 为生成预留 max_new 个位置；机制分析必须使用同一个实际 room_ctx，
    # 否则机制统计与 RULER 得分并不来自同一批样本。
    max_new = int(rcfg.get("max_new", 12))
    ctx = min(ctx, model.block_size - max_new - 1)
    ruler_seed = int(rcfg.get("seed", 1234))

    sents, corpus_source = load_corpus_sents_with_source(args.data_dir)
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

    csv_path = os.path.join(args.out_dir,
                            f"analysis_records_{variant}{args.suffix}.csv")
    if not examples:
        print("[analysis] 没有可分析的样例（ctx 过小或缺少语料），跳过")
        # 空结果必须清掉上一轮的逐记录 CSV：JSON 已声明 n_records=0，
        # 若旧 CSV 还在，plots._read_records 会继续消费过期记录。
        try:
            os.remove(csv_path)
        except FileNotFoundError:
            pass
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
    # 简单监督基线（线性探针）：从同一批记录学“该不该写入”的最简线性模型。
    # 探针有 hit 标签监督，门控 logit 没有；对照语义见 lega/probe.py 与
    # 落盘报告的 note，不得反向解读。
    if not args.no_probe:
        if args.probe_l2:
            probe_l2 = [float(x) for x in args.probe_l2.split(",") if x.strip()]
        else:
            probe_l2 = [float(x) for x in acfg.get(
                "probe_l2_grid", list(_PROBE_L2_GRID))]
        probe_folds = (int(args.probe_folds) if args.probe_folds is not None
                       else int(acfg.get("probe_folds", _PROBE_FOLDS)))
        summary["linear_probe"] = linear_probe(
            records, folds=probe_folds, l2_grid=probe_l2,
            seed=int(ck.get("seed")) if ck.get("seed") is not None else 0)
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
    # 机制分析与 RULER 用同一干扰语料来源：train 回退必须可审计
    summary["corpus_source"] = corpus_source
    summary["corpus_held_out"] = corpus_source in ("val", "test")
    summary.update(provenance)
    if skipped_tasks:
        summary["skipped_tasks"] = skipped_tasks

    os.makedirs(args.out_dir, exist_ok=True)
    # 逐记录 CSV（供画图与复核）
    if records:
        fields = sorted({k for r in records for k in r})
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(records)
    else:
        # 同 early return：本次没有任何记录时不能留下上一轮的 CSV。
        try:
            os.remove(csv_path)
        except FileNotFoundError:
            pass
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
        print(f"[analysis] PR-AUC logit={fmt(summary.get('auprc_logit'))} "
              f"qterm={fmt(summary.get('auprc_qterm'))} "
              f"（hit 基线率={fmt(summary.get('hit_base_rate'))}，"
              f"主指标={summary.get('primary_detection_metric')}）")
    boot = summary.get("auroc_cluster_bootstrap") or {}
    ci = (boot.get("ci95") or {}).get("logit_minus_qterm")
    if ci:
        print(f"[analysis] AUROC(logit−qterm) 样例聚类 bootstrap 95% CI: "
              f"[{ci[0]:.3f}, {ci[1]:.3f}]"
              f"（n_examples={boot.get('n_examples')}，"
              f"n_resamples={boot.get('n_resamples')}）")
    ap_ci = (boot.get("ci95") or {}).get("ap_logit_minus_qterm")
    if ap_ci:
        print(f"[analysis] PR-AUC(logit−qterm) 样例聚类 bootstrap 95% CI: "
              f"[{ap_ci[0]:.3f}, {ap_ci[1]:.3f}]（不平衡下的主指标口径）")
    cal = summary.get("calibration_g") or {}
    if cal.get("brier") is not None:
        # 单一类别（全 hit 或全 miss）时常数基率 Brier=0，skill 无定义；
        # 缺省值必须显式显示 n/a，不能拿 None 去做带符号浮点格式化。
        skill = cal.get("brier_skill")
        skill_txt = "n/a" if skill is None else f"{skill:+.3f}"
        print(f"[analysis] 门控概率校准：Brier={cal['brier']:.4f}"
              f"（常数基率基线={fmt(cal.get('brier_constant'))}，"
              f"skill={skill_txt}）"
              f" ECE={fmt(cal.get('ece'))}（排序好≠概率可信）")
    probe = summary.get("linear_probe") or {}
    pset = (probe.get("feature_sets") or {}).get(probe.get("primary") or "") or {}
    poof = pset.get("oof") or {}
    if poof:
        f1 = poof.get("f1") or {}
        f1t = poof.get("f1_tuned") or {}
        print(f"[analysis] 线性探针简单基线（{pset.get('name')}，"
              f"{probe.get('folds')} 折嵌套分组 CV、"
              f"λ={pset.get('selected_l2')}）："
              f"AUROC={fmt(poof.get('auroc'))} PR-AUC={fmt(poof.get('auprc'))} "
              f"F1(0.5)={fmt(f1.get('macro'))} "
              f"F1(调优阈值)={fmt(f1t.get('macro'))} "
              f"（OOF，有标签监督；门控 logit 无标签监督，务必对照解读）")
    elif probe:
        print(f"[analysis] 线性探针跳过：{probe.get('note', '')[:120]}")
    print(f"[analysis] 完成 -> {args.out_dir}")


if __name__ == "__main__":
    main()
