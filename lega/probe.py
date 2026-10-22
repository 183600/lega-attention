"""简单学习基线（线性探针）——"该不该写入"判别的最简监督模型对照。

科研方法问题：机制分析把门控 logit 与"仅查询项 qterm""单统计量 max_p /
log n"并排对照，但这些要么是门控自身的组件、要么是未调参的原始统计量，
缺少一个**用同一批特征、有标签监督训练并调过正则**的最简单学习基线。
没有这个对照，"门控 logit 的 AUROC/PR-AUC 很高"可能只说明这些廉价特征
本身线性可分，"复杂门控结构"的贡献无从谈起——即所谓"只打败了稻草人"。

本模块在机制分析的同一批记录上训练 L2 逻辑回归，用**嵌套的、按样例
分组并按标签均衡的交叉验证**产生 out-of-fold 分数：

  * 同一 (层,头,查询位) 的记录不独立（同一样例共享难度与 gold 位置），
    按行随机切分会把同一样例的其余记录留给训练折、对留出折"背题"；分组
    CV 保证同一样例只出现在一侧，OOF 指标才是诚实的泛化估计；
  * 折分配同时按 hit 标签均衡：正类记录只占少数时，纯轮转分组可能让
    某些折完全没有正类，训练折退化为常数先验、留出折无法计算 AP；分层
    分组把正类样例尽量分散到各折，降低这种“空正类折”造成的基线低估；
  * **嵌套 CV**：λ 与决策阈值只在每个外层训练折内部的 CV 上选择，再在
    外层留出折上评测。若在最终报告的同一份 OOF 上选 λ/阈值，OOF 指标会
    因选择偏差而乐观——简单基线也不能用这种口径“调参”；
  * λ 的选择指标与主检验指标对齐（正类率 <5% 用 PR-AUC，否则 AUROC），
    而不是只看 log-loss；log-loss 仍并报，供概率质量参考；
  * 决策阈值用内层 OOF 的 macro-F1 选择；另保留固定 0.5 阈值的 F1，
    两者并报，既与门控 g>0.5 同口径可比，也避免把“阈值没调”误当成
    “简单基线判别力差”；
  * 报告 AUROC 与 PR-AUC（hit/¬hit 常极度不平衡，正类率 <5% 时以
    PR-AUC 为主，其随机基线=正类率），并给 macro/micro/weighted
    三口径 F1，以及 Brier 分数（概率校准的严格评分规则）。

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

from .utils import (auroc, average_precision, calibration_metrics, f1_scores,
                    seed_int, strict_int)

# hit 判别正类率低于该值时以 PR-AUC 为主指标（与 lega.analysis 同一口径，
# 集中定义于此并由此导入，避免两处阈值漂移）。
PRIMARY_AP_THRESHOLD = 0.05
EXTREME_IMBALANCE_THRESHOLD = 0.001
# 决策阈值：与门控 F1 的 g>0.5 完全一致，保证探针/门控/固定规则三者同口径。
DECISION_THRESHOLD = 0.5
# 阈值调优网格：内层 OOF 上按 macro-F1 选；0.5 在其中，保证“调阈值”不会
# 比固定默认更差（只在有一致收益时移动）。细网格避免只搜索粗档位；
# 极端不平衡时（正类率 <1%）校准后的概率常远低于 0.01，仅有 [0.01,0.99]
# 会让调优永远无法把少量高置信正例判出，因此向低端补几个对数档位。
THRESHOLD_GRID = ((1e-4, 1e-3, 5e-3)
                  + tuple(round(0.01 * i, 2) for i in range(1, 100)))
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


def _finite_nonnegative(value, name: str = "l2") -> float:
    """有限非负浮点；bool/NaN/Inf/负值显式拒绝（负数 L2 会破坏凸性）。"""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} 必须是有限非负数，收到 bool {value!r}")
    if isinstance(value, str) and "_" in value:
        raise ValueError(f"{name} 必须是有限非负数，收到 {value!r}")
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError) as e:
        raise ValueError(f"{name} 必须是有限非负数，收到 {value!r}") from e
    if not math.isfinite(out) or out < 0:
        raise ValueError(f"{name} 必须是有限非负数，收到 {value!r}")
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
    l2 = _finite_nonnegative(l2, "l2")
    n, d = X.shape
    if n == 0 or d == 0:
        return {"coef": np.zeros(d), "intercept": 0.0}
    if float(y.min()) == float(y.max()):
        # 单类训练折：最优常数=先验；概率由 _predict 用位置参数给出。
        return {"coef": np.zeros(d),
                "intercept": math.inf if y[0] > 0.5 else -math.inf}
    Xb = np.hstack([X, np.ones((n, 1), dtype=np.float64)])
    pen = l2 * np.eye(d + 1)
    pen[-1, -1] = 0.0                      # 截距不惩罚
    pen = pen + 1e-8 * np.eye(d + 1)       # 数值稳定
    w = np.zeros(d + 1)
    for _ in range(int(max_iter)):
        p = _sigmoid(Xb @ w)
        grad = Xb.T @ (p - y)
        grad[:-1] += l2 * w[:-1]
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


def _float_or_nan(v) -> float:
    """外部/旧格式记录的单列取值：缺失或非数值一律按缺失（NaN）处理。

    与 analysis/badcase/plots 等模块的有限数值解析同口径：bool 不是数值
    观测，``float("1_0") == 10.0`` 的下划线写法也不是规范 JSON 数值。
    若放行，坏记录里的 "1_0" 会被当成真实特征值 10，线性探针的标准化、
    λ 选择与 OOF 指标都会建立在伪造的强特征上。
    """
    if v is None or isinstance(v, (bool, np.bool_)):
        return float("nan")
    if isinstance(v, str) and "_" in v:
        return float("nan")
    try:
        return float(v)
    except (TypeError, ValueError, OverflowError):
        return float("nan")


def _column_values(records: Sequence[Dict], name: str) -> np.ndarray:
    """单列特征取值（缺失/非数值为 NaN）；语义与旧逐记录实现一致。"""
    if name == "log_maxp":
        vals = []
        for r in records:
            mp = _float_or_nan(r.get("max_p"))
            # max_p 是概率质量：负值或 >1 不是“很小的概率”，而是坏记录。
            # 若只 clamp 下界，max_p=-5 会被当成极强集中度（log 1e-12），
            # 在探针里给出与真实语义相反的信号；按缺失剔除。
            if math.isnan(mp) or not (0.0 <= mp <= 1.0):
                vals.append(float("nan"))
            else:
                vals.append(math.log(max(mp, _MAXP_FLOOR)))
        return np.asarray(vals, dtype=np.float64)
    if name == "log_n":
        return np.asarray([_float_or_nan(r.get("log_n")) for r in records],
                          dtype=np.float64)
    if name == "qterm":
        return np.asarray([_float_or_nan(r.get("qterm")) for r in records],
                          dtype=np.float64)
    raise ValueError(f"未知特征: {name}")


def _hit_values(records: Sequence[Dict]) -> np.ndarray:
    """hit 列（缺失/非数值为 NaN）；非 0/1 值在有效性判定中被整体剔除。"""
    return np.asarray([_float_or_nan(r.get("hit")) for r in records],
                      dtype=np.float64)


def _features(records: Sequence[Dict], names: Sequence[str], cache: dict = None):
    """构造特征矩阵与有效性掩码；缺失/非有限特征的记录整体剔除。

    cache：可选的“列名 -> 数组”缓存，仅在 records 为同一对象时复用；
    由调用方在多次取不同特征子集时传入，避免重复提取同一列。
    """
    n = len(records)
    if cache is None:
        cache = {}
    # 缓存只在同一 records 对象上复用；仅比较长度会把另一批等长记录
    # 的旧列静默喂给新记录，产生错误特征矩阵。保存对象引用而不是 id，
    # 避免 id 回收复用造成误命中。
    if cache.get("_records") is not records:
        cache.clear()
        cache["_records"] = records
    if n == 0:
        # 空输入保持旧实现的形状（np.asarray([]) 为 (0,)），消费方无需分支。
        return np.asarray([], dtype=np.float64), []
    X = np.empty((n, len(names)), dtype=np.float64)
    valid = np.ones(n, dtype=bool)
    for j, name in enumerate(names):
        col = cache.get(name)
        if col is None or col.shape[0] != n:
            col = _column_values(records, name)
            cache[name] = col
        X[:, j] = col
        valid &= np.isfinite(col)
    hit = cache.get("_hit")
    if hit is None or hit.shape[0] != n:
        hit = _hit_values(records)
        cache["_hit"] = hit
    valid &= (hit == 0.0) | (hit == 1.0)
    X[~valid] = np.nan
    valid_list = [r for r, v in zip(records, valid.tolist()) if v]
    return X, valid_list


def _group_folds(groups: Sequence, k: int, seed: int,
                 y: Optional[Sequence] = None) -> np.ndarray:
    """按样例（聚类）分组做 k 折分配，样例内绝不跨折。

    y 给出时按标签做**分层分组**：先把含正类的样例按正类数从多到少
    依次分给当前正类最少的折（平局看总样本数），再把纯负类样例分给
    当前总样本数最少的折。这样稀有正类不会全部落进同一折，避免训练折
    单类退化、留出折 AP 不可计算。y=None 时保持旧的确定性轮转分配。
    """
    uniq = sorted(set(groups))
    rng = np.random.default_rng(seed)
    tie = {g: float(rng.random()) for g in uniq}
    if y is None:
        order = sorted(uniq, key=lambda g: tie[g])
        fold_of = {g: (i % k) for i, g in enumerate(order)}
        return np.asarray([fold_of[g] for g in groups], dtype=np.int64)
    y_arr = np.asarray(list(y), dtype=np.float64)
    if y_arr.shape[0] != len(groups):
        raise ValueError("y 与 groups 长度不一致")
    pos = {g: 0 for g in uniq}
    total = {g: 0 for g in uniq}
    for g, label in zip(groups, y_arr.tolist()):
        total[g] += 1
        if label == 1.0:
            pos[g] += 1
    pos_groups = [g for g in uniq if pos[g] > 0]
    neg_groups = [g for g in uniq if pos[g] == 0]
    pos_groups.sort(key=lambda g: (-pos[g], -total[g], tie[g]))
    neg_groups.sort(key=lambda g: (-total[g], tie[g]))
    fold_pos = [0] * k
    fold_total = [0] * k
    fold_of = {}
    for g in pos_groups:
        f = min(range(k), key=lambda i: (fold_pos[i], fold_total[i], i))
        fold_of[g] = f
        fold_pos[f] += pos[g]
        fold_total[f] += total[g]
    for g in neg_groups:
        f = min(range(k), key=lambda i: (fold_total[i], i))
        fold_of[g] = f
        fold_total[f] += total[g]
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


def _f1_block(y: np.ndarray, p: np.ndarray, thresholds: np.ndarray) -> Optional[Dict]:
    """逐记录阈值下的 F1（macro/micro/weighted + 逐类明细）。"""
    m = np.isfinite(p) & np.isfinite(thresholds)
    y, p, thresholds = y[m], p[m], thresholds[m]
    if y.size == 0 or len(set(y.tolist())) != 2:
        return None
    pred = (p > thresholds).astype(np.int64)
    f1s = f1_scores(y.astype(np.int64), pred, labels=[0, 1])
    return {
        "threshold": ("per-fold inner-CV macro-F1"
                      if len(set(thresholds.tolist())) > 1
                      else float(thresholds[0])),
        "thresholds": [float(x) for x in sorted(set(thresholds.tolist()))],
        "macro": f1s["macro"], "micro": f1s["micro"],
        "weighted": f1s["weighted"],
        "per_class": {str(k): v for k, v in f1s["per_class"].items()},
    }


def _score_block(y: np.ndarray, p: np.ndarray,
                 thresholds: Optional[np.ndarray] = None) -> Dict:
    """一组分数的全部指标（AUROC/PR-AUC/LogLoss/Brier/F1 三口径）。

    默认 F1 在固定 0.5 阈值上计算（与门控 g>0.5 同口径）；传入每条
    记录自己的阈值时追加 `f1_tuned`。两个口径并报，避免把固定 0.5 的
    阈值选择误读成“简单基线判别力差”。
    """
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
    cal = calibration_metrics(p, y)
    out.update({"ece": cal.get("ece"), "mce": cal.get("mce"),
                "brier_constant": cal.get("brier_constant"),
                "brier_skill": cal.get("brier_skill"),
                "calibration_bins": cal.get("bins")})
    fixed = np.full(y.shape[0], DECISION_THRESHOLD, dtype=np.float64)
    out["f1"] = _f1_block(y, p, fixed)
    if thresholds is not None:
        thr = np.asarray(thresholds, dtype=np.float64)[m]
        out["f1_tuned"] = _f1_block(y, p, thr)
    else:
        out["f1_tuned"] = None
    return out


def _fit_predict(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray,
                 l2: float) -> np.ndarray:
    """只用训练折统计量标准化后拟合 L2 逻辑回归并预测留出折。"""
    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    model = fit_logreg((Xtr - mu) / sd, ytr, l2=l2)
    return predict_logreg(model, (Xte - mu) / sd)


def _select_threshold(y: np.ndarray, p: np.ndarray) -> Tuple[float, Optional[float]]:
    """在内层 OOF 上按 macro-F1 选决策阈值（并列取更接近 0.5 的稳定阈值）。"""
    m = np.isfinite(p)
    y, p = y[m], p[m]
    if y.size == 0 or len(set(y.tolist())) != 2:
        return DECISION_THRESHOLD, None
    best = None
    for t in THRESHOLD_GRID:
        pred = (p > t).astype(np.int64)
        f1s = f1_scores(y.astype(np.int64), pred, labels=[0, 1])
        macro = f1s["macro"]
        if macro is None:
            continue
        key = (float(macro), -abs(float(t) - DECISION_THRESHOLD), float(t))
        if best is None or key > best[0]:
            best = (key, float(t), float(macro))
    if best is None:
        return DECISION_THRESHOLD, None
    return best[1], best[2]


def _selection_value(block: Dict, primary_metric: Optional[str]) -> Optional[float]:
    """λ 的选择分数：与主检验指标对齐（AP 或 AUROC），缺失时回退 log-loss。"""
    if primary_metric == "PR-AUC":
        return block.get("auprc")
    if primary_metric and primary_metric.startswith("AUROC"):
        return block.get("auroc")
    return block.get("auprc") if block.get("auprc") is not None else block.get("auroc")


def _selection_name(primary_metric: Optional[str]) -> str:
    if primary_metric == "PR-AUC":
        return "auprc"
    if primary_metric and primary_metric.startswith("AUROC"):
        return "auroc"
    return "auprc/auroc（按可用性）"


def linear_probe(records: Sequence[Dict], folds: int = DEFAULT_FOLDS,
                 l2_grid: Sequence[float] = DEFAULT_L2_GRID,
                 seed: int = 0) -> Dict:
    """在机制分析记录上训练/评估线性探针，返回可直接 JSON 落盘的报告。"""
    # 外部/旧格式 analysis JSON 的记录列表可能混入 null/标量：按坏记录剔除，
    # 否则逐列提取会在 r.get 上崩溃。
    records = [r for r in (records or []) if isinstance(r, dict)]
    folds = strict_int(folds, "folds")
    # 探针的折分配用固定种子：bool/小数/越界种子会让“同一输入可复现”
    # 的承诺失效，必须与 set_seed 同口径（0..2^32-1）。
    seed = seed_int(seed, "seed")
    if folds < 2:
        raise ValueError(f"folds 必须 >= 2，收到 {folds}")
    l2_grid = [_finite_nonnegative(x, "l2_grid 项") for x in l2_grid]
    report: Dict = {
        "folds": None, "n_records": len(records), "n_valid": 0,
        "n_examples": 0, "n_pos": 0, "n_neg": 0, "base_rate": None,
        "imbalance_level": None, "primary_detection_metric": None,
        "l2_grid": l2_grid,
        "selection_metric": ("嵌套分组 CV：λ 在内层 OOF 上按与主检验对齐的"
                             "指标选择（正类率 <5% 用 PR-AUC，否则 AUROC；"
                             "并列按 log-loss 与更小 λ）；最终 OOF 不被用于"
                             "选 λ，避免选择偏差"),
        "decision_threshold": DECISION_THRESHOLD,
        "decision_threshold_tuned": ("每个外层折用内层 OOF 的 macro-F1 选阈值；"
                                     "固定 0.5 与调优阈值两个 F1 口径并报"),
        "grouped_cv": ("按样例（example）分组，同一样例的记录绝不跨折；"
                       "折分配按 hit 标签分层，降低稀有正类折缺失；"
                       "标准化统计只用训练折拟合"),
        "nested_cv": True,
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
    # 逐列提取一次并缓存：log_n/_hit 用于有效性过滤，log_maxp/qterm 供各特征
    # 集按同一份列切片复用（值与逐特征集重复提取逐位一致）。
    col_cache: Dict = {}
    _features(records, ("log_n",), cache=col_cache)
    logn_col = col_cache["log_n"]
    hit_col = col_cache["_hit"]
    valid_mask = np.isfinite(logn_col) & ((hit_col == 0.0) | (hit_col == 1.0))
    valid_all = [r for r, v in zip(records, valid_mask.tolist()) if v]
    y_all = hit_col[valid_mask]
    for nm in ("log_maxp", "qterm"):
        if nm not in col_cache:
            col_cache[nm] = _column_values(records, nm)
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
    k = folds
    if len(examples) < 2 or y_all.size == 0 or len(set(y_all.tolist())) != 2:
        report["note"] = ("样例/类别不足（需 ≥2 个样例且两类齐备），"
                          "线性探针不可估计。" + report["note"])
        return report
    k = min(k, len(examples))
    report["folds"] = k
    available = []
    feature_matrices = {}
    for key, name, names in FEATURE_SETS:
        # 有效记录上逐列切片（列值直接来自缓存，无重复提取）；任一列非有限
        # 即该特征集在全部有效记录上不齐备，与逐记录重建的口径一致。
        cols = [col_cache[n][valid_mask] for n in names]
        if len(valid_all) > 0 and all(np.isfinite(c).all() for c in cols):
            available.append((key, name, names))
            feature_matrices[key] = np.column_stack(cols)
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
    fold_arr = _group_folds(groups.tolist(), k, seed, y=y_all)
    l2s = list(dict.fromkeys(report["l2_grid"]))
    selected_name = _selection_name(primary_metric)
    for key, name, names in available:
        X = feature_matrices[key]
        set_out: Dict = {
            "name": name, "features": list(names),
            "n": int(y_all.size), "n_pos": n_pos,
            "nested_cv": True, "stratified_group_folds": True,
            "selection_metric": selected_name,
            "selected_l2": None, "selected_l2_by_fold": [],
            "selected_l2_counts": {}, "oof": None,
            "per_l2": [], "per_fold": [],
        }
        fixed_oof = {l2: np.full(y_all.shape[0], float("nan")) for l2 in l2s}
        nested_oof = np.full(y_all.shape[0], float("nan"))
        nested_thr = np.full(y_all.shape[0], float("nan"))
        fold_selected = []
        for f in sorted(set(fold_arr.tolist())):
            te = fold_arr == f
            tr = ~te
            Xtr, ytr = X[tr], y_all[tr]
            # 固定 λ 的外层预测：既用于 per_l2 敏感性表，也复用为嵌套
            # 预测（每个外层折只选一个 λ，不重复拟合）。
            for l2 in l2s:
                fixed_oof[l2][te] = _fit_predict(Xtr, ytr, X[te], l2)
            tr_groups = groups[tr].tolist()
            n_tr_groups = len(set(tr_groups))
            inner_k = max(2, min(k, n_tr_groups)) if n_tr_groups >= 2 else 0
            chosen_l2 = 1.0 if 1.0 in l2s else max(l2s)
            chosen_thr = DECISION_THRESHOLD
            inner_metric = None
            inner_best_macro = None
            if inner_k >= 2:
                inner_fold = _group_folds(
                    tr_groups, inner_k, seed + 1009 * (f + 1), y=ytr)
                inner_best = None
                for l2 in l2s:
                    inner_oof = _oof_scores(Xtr, ytr, inner_fold, l2)
                    block = _score_block(ytr, inner_oof)
                    score = _selection_value(block, primary_metric)
                    ll = block.get("logloss")
                    # 主指标优先；并列看 log-loss（越低越好），再取更小 λ。
                    key_l2 = (score if score is not None else float("-inf"),
                              -(ll if ll is not None else float("inf")),
                              -float(l2))
                    if inner_best is None or key_l2 > inner_best[0]:
                        inner_best = (key_l2, l2, inner_oof, score)
                chosen_l2 = float(inner_best[1])
                chosen_thr, inner_best_macro = _select_threshold(
                    ytr, inner_best[2])
                inner_metric = inner_best[3]
            nested_oof[te] = fixed_oof[chosen_l2][te]
            nested_thr[te] = chosen_thr
            fold_selected.append({
                "fold": int(f), "selected_l2": chosen_l2,
                "threshold": float(chosen_thr),
                "inner_folds": int(inner_k) if inner_k >= 2 else None,
                "inner_selection_metric": selected_name,
                "inner_selection_value": inner_metric,
                "inner_best_macro_f1": inner_best_macro,
                "n_train": int(tr.sum()), "n_train_pos": int((ytr == 1).sum()),
                "n_test": int(te.sum()),
                "n_test_pos": int((y_all[te] == 1).sum()),
            })
        selected_counts: Dict[float, int] = {l2: 0 for l2 in l2s}
        for rec in fold_selected:
            selected_counts[rec["selected_l2"]] = (
                selected_counts.get(rec["selected_l2"], 0) + 1)
        # 汇总字段取众数 λ（并列取更小 λ）；实际 nested OOF 用的是逐折 λ，
        # 逐折选择记录在 selected_l2_by_fold / per_fold 中完整保留。
        set_out["selected_l2"] = min(
            l2s, key=lambda x: (-selected_counts.get(x, 0), x))
        set_out["selected_l2_counts"] = {
            str(l2): selected_counts.get(l2, 0) for l2 in l2s}
        set_out["selected_l2_by_fold"] = [
            {"fold": r["fold"], "l2": r["selected_l2"],
             "threshold": r["threshold"]} for r in fold_selected]
        set_out["oof"] = _score_block(y_all, nested_oof, thresholds=nested_thr)
        for l2 in l2s:
            block = _score_block(y_all, fixed_oof[l2])
            f1 = block.get("f1") or {}
            set_out["per_l2"].append({
                "l2": float(l2), "logloss": block.get("logloss"),
                "auroc": block.get("auroc"), "auprc": block.get("auprc"),
                "brier": block.get("brier"),
                "f1_macro_fixed": f1.get("macro"),
                "selected_count": int(selected_counts.get(l2, 0)),
                "selected": bool(l2 == set_out["selected_l2"]),
            })
        for rec in fold_selected:
            f = rec["fold"]
            te = fold_arr == f
            block = _score_block(y_all[te], nested_oof[te],
                                 thresholds=nested_thr[te])
            rec.update({
                "auroc": block.get("auroc"),
                "auprc": block.get("auprc"),
                "logloss": block.get("logloss"),
                "brier": block.get("brier"),
                "f1_fixed": block.get("f1"),
                "f1_tuned": block.get("f1_tuned"),
            })
        set_out["per_fold"] = fold_selected
        report["feature_sets"][key] = set_out
    return report
