"""简单学习基线（线性探针）——"该不该写入"判别的最简监督模型对照。

科研方法问题：机制分析把门控 logit 与"仅查询项 qterm""单统计量 max_p /
log n"并排对照，但这些要么是门控自身的组件、要么是未调参的原始统计量，
缺少一个**用同一批特征、有标签监督训练并调过正则**的最简单学习基线。
没有这个对照，"门控 logit 的 AUROC/PR-AUC 很高"可能只说明这些廉价特征
本身线性可分，"复杂门控结构"的贡献无从谈起——即所谓"只打败了稻草人"。

本模块在机制分析的同一批记录上训练 L2 逻辑回归，用**按样例分组的交叉
验证**产生 out-of-fold 分数：

  * 同一 (层,头,查询位) 的记录不独立（同一样例共享难度与 gold 位置），
    按行随机切分会把同一样例的其余记录留给训练折、对留出折"背题"；分组
    CV 保证同一样例只出现在一侧，OOF 指标才是诚实的泛化估计；
  * 标准化均值/方差只用训练折计算，防止信息从留出折泄漏；
  * λ（L2 强度）在网格上用 out-of-fold 平均 log-loss 选择——"简单基线
    也要充分调优"，而不是随手一个 λ；逐 λ 指标一并落盘供敏感性审视；
  * 报告 AUROC 与 PR-AUC（hit/¬hit 常极度不平衡，正类率 <5% 时以
    PR-AUC 为主，其随机基线=正类率），并给 g>0.5 决策点的 F1
    （macro/micro/weighted 三口径，与门控 F1 同阈值，保证同口径可比）。

与门控 logit 的对照语义（必须如实陈述，防止过度解读）：探针是**有监督**
的（用 hit 标签训练），门控是语言建模训练的副产品（训练时从未见过 hit
标签）。因此探针超过门控是预期内的——说明该判别能力本可由线性模型从
廉价统计量学得；门控若不能在 PR-AUC 上显著超过探针，就不能把"更能判断
该不该写入"归因给门控结构。该对照与 n-gram 对比一样属于补充性/探索性
证据，不替代预注册的确认性检验。
"""
import math
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from .utils import auroc, average_precision, f1_scores

# hit 判别正类率低于该值时以 PR-AUC 为主指标（与 lega.analysis 同一口径，
# 集中定义于此并由此导入，避免两处阈值漂移）。
PRIMARY_AP_THRESHOLD = 0.05
EXTREME_IMBALANCE_THRESHOLD = 0.001
# 决策阈值：与门控 F1 的 g>0.5 完全一致，保证探针/门控/固定规则三者同口径。
DECISION_THRESHOLD = 0.5
# max_p 的下界：log(0) 会变 -inf，统计中 0 是数值下溢，用 1e-12 兜底。
_MAXP_FLOOR = 1e-12
DEFAULT_L2_GRID = (0.0, 0.01, 0.1, 1.0)
DEFAULT_FOLDS = 5
# 特征集：base 只含门控自身可见的廉价统计量（集中度、长度）；full 另含
# 查询项 qterm（门控 logit 的线性项之一）。门控 logit 在 full 特征上本身就是
# 线性函数（w·q + a·c + b·log n + d = a·log max_p + (a+b)·log n + w·q + d），
# 所以"同特征集的逻辑回归"是与门控结构最严格的可比简单基线：假设类相同，
# 唯一区别是探针直接用 hit 标签监督训练、门控是语言建模的副产品。
FEATURE_SETS = (
    ("conc_len", "集中度+长度", ("log_maxp", "log_n")),
    ("conc_len_query", "集中度+长度+查询项", ("log_maxp", "log_n", "qterm")),
)


def imbalance_level(base_rate: Optional[float]) -> Tuple[Optional[str], Optional[str]]:
    """正类率 → (不平衡等级, 主指标) 的显式口径（与机制分析一致）。"""
    if base_rate is None:
        return None, None
    if base_rate <= EXTREME_IMBALANCE_THRESHOLD:
        return "extreme(≤1:1000)", "PR-AUC"
    if base_rate < PRIMARY_AP_THRESHOLD:
        return "severe(<5%)", "PR-AUC"
    return "moderate", "AUROC 与 PR-AUC 并报"


def _sigmoid(z: np.ndarray) -> np.ndarray:
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def fit_logreg(X: np.ndarray, y: np.ndarray, l2: float = 0.0,
               max_iter: int = 100, tol: float = 1e-10) -> Dict:
    """L2 逻辑回归（Newton–IRLS，无第三方依赖，确定性）。

    截距不参与 L2 惩罚；Hessian 对角加 1e-8 防完全可分时的奇异。训练集
    只有单一类别时退化为常数先验模型（不抛错，由调用方按指标口径处理）。
    返回 {"coef": (d,), "intercept": float}。
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n, d = X.shape
    if n == 0 or d == 0:
        return {"coef": np.zeros(d), "intercept": 0.0}
    if float(y.min()) == float(y.max()):
        # 单类训练折：最优常数=先验；概率由 _predict 用位置参数给出。
        return {"coef": np.zeros(d),
                "intercept": math.inf if y[0] > 0.5 else -math.inf}
    Xb = np.hstack([X, np.ones((n, 1), dtype=np.float64)])
    pen = float(l2) * np.eye(d + 1)
    pen[-1, -1] = 0.0                      # 截距不惩罚
    pen = pen + 1e-8 * np.eye(d + 1)       # 数值稳定
    w = np.zeros(d + 1)
    for _ in range(int(max_iter)):
        p = _sigmoid(Xb @ w)
        grad = Xb.T @ (p - y)
        grad[:-1] += float(l2) * w[:-1]
        W = np.clip(p * (1.0 - p), 1e-8, None)
        H = Xb.T @ (Xb * W[:, None]) + pen
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            try:
                step = np.linalg.lstsq(H, grad, rcond=None)[0]
            except np.linalg.LinAlgError:
                break
        w = w - step
        if float(np.max(np.abs(step))) < float(tol):
            break
    return {"coef": w[:-1].copy(), "intercept": float(w[-1])}


def predict_logreg(model: Dict, X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.size == 0:
        return np.zeros(0, dtype=np.float64)
    intercept = model.get("intercept", 0.0)
    if math.isinf(intercept):
        return np.full(X.shape[0], 1.0 if intercept > 0 else 0.0)
    return _sigmoid(X @ np.asarray(model["coef"], dtype=np.float64) + intercept)


def _features(records: Sequence[Dict], names: Sequence[str]):
    """构造特征矩阵与有效性掩码；缺失/非有限特征的记录整体剔除。"""
    rows, valid = [], []
    for r in records:
        vals = []
        for name in names:
            if name == "log_maxp":
                mp = r.get("max_p")
                vals.append(math.log(max(float(mp), _MAXP_FLOOR))
                            if mp is not None else float("nan"))
            elif name == "log_n":
                ln = r.get("log_n")
                vals.append(float(ln) if ln is not None else float("nan"))
            elif name == "qterm":
                q = r.get("qterm")
                vals.append(float(q) if q is not None else float("nan"))
            else:
                raise ValueError(f"未知特征: {name}")
        hit = r.get("hit")
        ok = (hit is not None and all(math.isfinite(v) for v in vals)
              and float(hit) in (0.0, 1.0))
        if ok:
            rows.append(vals)
            valid.append(r)
        else:
            rows.append([float("nan")] * len(names))
    return np.asarray(rows, dtype=np.float64), valid


def _group_folds(groups: Sequence, k: int, seed: int) -> np.ndarray:
    """按样例（聚类）分组做分层近似均衡的 k 折分配，样例内绝不跨折。"""
    uniq = sorted(set(groups))
    rng = np.random.default_rng(seed)
    order = list(uniq)
    rng.shuffle(order)
    fold_of = {g: (i % k) for i, g in enumerate(order)}
    return np.asarray([fold_of[g] for g in groups], dtype=np.int64)


def _logloss(y: np.ndarray, p: np.ndarray) -> Optional[float]:
    if y.size == 0:
        return None
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def _brier(y: np.ndarray, p: np.ndarray) -> Optional[float]:
    if y.size == 0:
        return None
    return float(np.mean((p - y) ** 2))


def _safe_auroc(y: np.ndarray, p: np.ndarray) -> Optional[float]:
    if y.size == 0 or len(set(y.tolist())) != 2:
        return None
    v = auroc(p, y)
    return float(v) if np.isfinite(v) else None


def _safe_auprc(y: np.ndarray, p: np.ndarray) -> Optional[float]:
    if y.size == 0 or len(set(y.tolist())) != 2:
        return None
    v = average_precision(p, y)
    return float(v) if np.isfinite(v) else None


def _fold_views(X: np.ndarray, folds: np.ndarray):
    """每折的掩码与标准化后切片。

    返回 [(te, Xtr_n, Xte_n)]；标准化统计（mu/sd）只用训练折拟合，
    防信息从留出折泄漏。
    """
    views = []
    for f in sorted(set(folds.tolist())):
        te = folds == f
        tr = ~te
        if not tr.any():
            continue
        mu = X[tr].mean(axis=0)
        sd = X[tr].std(axis=0)
        sd = np.where(sd > 0, sd, 1.0)
        views.append((te, (X[tr] - mu) / sd, (X[te] - mu) / sd))
    return views


def _oof_scores(X: np.ndarray, y: np.ndarray, folds: np.ndarray,
                l2: float, views=None) -> np.ndarray:
    """分组 CV 的 out-of-fold 概率；标准化统计只用训练折拟合。"""
    oof = np.full(y.shape[0], float("nan"))
    if views is None:
        views = _fold_views(X, folds)
    for te, xtr_n, xte_n in views:
        model = fit_logreg(xtr_n, y[~te], l2=l2)
        oof[te] = predict_logreg(model, xte_n)
    return oof


def _score_block(y: np.ndarray, p: np.ndarray) -> Dict:
    """一组 OOF 分数的全部指标（AUROC/PR-AUC/LogLoss/Brier/F1 三口径）。"""
    m = np.isfinite(p)
    y, p = y[m], p[m]
    out: Dict = {
        "n": int(y.size), "n_pos": int((y == 1).sum()),
        "auroc": _safe_auroc(y, p),
        "auprc": _safe_auprc(y, p),
        "logloss": _logloss(y, p),
        "brier": _brier(y, p),
        "f1": None,
    }
    if y.size and len(set(y.tolist())) == 2:
        pred = (p > DECISION_THRESHOLD).astype(np.int64)
        f1s = f1_scores(y.astype(np.int64), pred, labels=[0, 1])
        out["f1"] = {
            "threshold": DECISION_THRESHOLD,
            "macro": f1s["macro"], "micro": f1s["micro"],
            "weighted": f1s["weighted"],
            "per_class": {str(k): v for k, v in f1s["per_class"].items()},
        }
    return out


def linear_probe(records: Sequence[Dict], folds: int = DEFAULT_FOLDS,
                 l2_grid: Sequence[float] = DEFAULT_L2_GRID,
                 seed: int = 0) -> Dict:
    """在机制分析记录上训练/评估线性探针，返回可直接 JSON 落盘的报告。"""
    report: Dict = {
        "folds": None, "n_records": len(records), "n_valid": 0,
        "n_examples": 0, "n_pos": 0, "n_neg": 0, "base_rate": None,
        "imbalance_level": None, "primary_detection_metric": None,
        "l2_grid": [float(x) for x in l2_grid],
        "selection_metric": "out-of-fold 平均 log-loss（λ 越大越正则；"
                            "λ 网格含 0 以检验正则是否真的必要）",
        "decision_threshold": DECISION_THRESHOLD,
        "grouped_cv": "按样例（example）分组，同一样例的记录绝不跨折；"
                      "标准化统计只用训练折拟合",
        "primary": None, "feature_sets": {},
        "note": ("线性探针为**有标签监督**的简单基线（用 hit 标签训练，OOF "
                 "指标），门控 logit 为语言建模副产品（训练未见 hit 标签）；"
                 "探针优于门控属预期，门控不能显著优于探针时不得把"
                 "“更能判断该不该写入”归因给门控结构。此对照为补充性/"
                 "探索性证据，不替代预注册的确认性检验。"),
    }
    if not report["l2_grid"]:
        # λ 网格为空（如命令行 --probe_l2 "," 过滤后为空、或配置写成空表）
        # 时无从选参：显式返回缺失，而不是在下面 best[1] 处崩溃。
        report["note"] = ("λ 网格为空（至少需一个候选取值），线性探针不可估计。"
                          + report["note"])
        return report
    if not records:
        report["note"] = "无记录，未训练探针。" + report["note"]
        return report
    base_rows, _ = _features(records, ("log_n",))
    valid_all = [r for r, ln in zip(records, base_rows[:, 0].tolist())
                 if r.get("hit") is not None and math.isfinite(float(ln))
                 and float(r.get("hit", 0.0)) in (0.0, 1.0)]
    y_all = np.asarray([float(r["hit"]) for r in valid_all], dtype=np.float64)
    n_pos = int((y_all == 1).sum())
    n_neg = int((y_all == 0).sum())
    base_rate = (n_pos / (n_pos + n_neg)) if (n_pos + n_neg) else None
    level, primary_metric = imbalance_level(base_rate)

    def _ex_key(ex):
        # example 可能缺失或类型混杂（旧/外部记录）：规范化为可排序、可哈希的
        # 键。缺失 example 的记录归入同一"未知"组（分组宁可偏合并也绝不跨折，
        # 泄漏防护不受影响），避免 sorted 在 None 与数值混排时抛 TypeError。
        return ("missing", "") if ex is None else ("example", str(ex))

    ex_keys = [_ex_key(r.get("example")) for r in valid_all]
    report.update({
        "n_valid": int(y_all.size), "n_examples": len(set(ex_keys)),
        "n_pos": n_pos, "n_neg": n_neg, "base_rate": base_rate,
        "imbalance_level": level, "primary_detection_metric": primary_metric,
    })
    examples = sorted(set(ex_keys))
    k = int(folds)
    if len(examples) < 2 or y_all.size == 0 or len(set(y_all.tolist())) != 2:
        report["note"] = ("样例/类别不足（需 ≥2 个样例且两类齐备），"
                          "线性探针不可估计。" + report["note"])
        return report
    k = max(2, min(k, len(examples)))
    report["folds"] = k
    available = []
    feature_matrices = {}
    for key, name, names in FEATURE_SETS:
        X, valid = _features(valid_all, names)
        if len(valid) == len(valid_all) and len(valid) > 0:
            available.append((key, name, names))
            feature_matrices[key] = X
    if not available:
        report["note"] = "没有任何特征集在所有有效记录上齐备。" + report["note"]
        return report
    # 主特征集：优先含 qterm 的 full（与门控 logit 的线性特征最可比），
    # 否则退到廉价统计量集；两个都算、都落盘，体现对基线的"充分调优"。
    report["primary"] = ("conc_len_query" if any(k0 == "conc_len_query"
                                                for k0, _, _ in available)
                         else available[0][0])
    gaps = sorted(set(examples))
    ex_to_idx = {g: i for i, g in enumerate(gaps)}
    groups = np.asarray([ex_to_idx[k] for k in ex_keys], dtype=np.int64)
    fold_arr = _group_folds(groups.tolist(), k, seed)
    for key, name, names in available:
        X = feature_matrices[key]
        views = _fold_views(X, fold_arr)
        set_out: Dict = {"name": name, "features": list(names),
                         "n": int(y_all.size), "n_pos": n_pos,
                         "selected_l2": None, "oof": None,
                         "per_l2": [], "per_fold": []}
        best = None
        for l2 in report["l2_grid"]:
            oof = _oof_scores(X, y_all, fold_arr, float(l2), views=views)
            block = _score_block(y_all, oof)
            block["l2"] = float(l2)
            set_out["per_l2"].append({
                "l2": float(l2), "logloss": block["logloss"],
                "auroc": block["auroc"], "auprc": block["auprc"]})
            # λ 选择：log-loss 越低越好；并列取更小的 λ（更简单的模型）。
            cand = (block["logloss"] if block["logloss"] is not None
                    else float("inf"))
            if best is None or cand < best[0] - 1e-12:
                best = (cand, float(l2), block, oof)
        set_out["selected_l2"] = best[1]
        set_out["oof"] = {k1: v for k1, v in best[2].items() if k1 != "l2"}
        for f in sorted(set(fold_arr.tolist())):
            te = fold_arr == f
            block = _score_block(y_all[te], best[3][te])
            set_out["per_fold"].append({
                "fold": int(f), "n": block["n"], "n_pos": block["n_pos"],
                "auroc": block["auroc"], "auprc": block["auprc"],
                "logloss": block["logloss"]})
        report["feature_sets"][key] = set_out
    return report
