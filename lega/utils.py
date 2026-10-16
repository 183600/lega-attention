"""通用工具：随机种子、JSON 读写、指标计算（AUROC/Pearson/统计检验，无 scipy 依赖）。"""
import json
import math
import os
import random
import time

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """固定当前进程内所有随机源，使同一台机器上的同一实验可复现。

    覆盖：Python random、NumPy 全局 RNG、PyTorch（CPU/CUDA）、cuDNN 确定性。
    本项目的“dataloader”是 data.py 中显式接收 ``np.random.default_rng(seed)`` 的
    批采样函数（无 torch DataLoader/worker），因此数据顺序由同一 seed 唯一确定；
    若将来引入 DataLoader，须同时传入该 seed 构造的 ``torch.Generator`` 并设置
    ``worker_init_fn``（参考 PyTorch 官方可复现性说明）。
    TensorFlow 不在依赖内，无需 TF seed。

    PYTHONHASHSEED 仅对之后启动的子进程生效（当前进程的哈希盐在解释器启动时已
    固定）；run_all.py 会在派生子进程前设置。跨硬件架构（不同 GPU 型号、CPU
    指令集、cuDNN 版本）无法保证逐位一致，本函数只保证同机相对稳定。
    """
    os.environ.setdefault("PYTHONHASHSEED", str(int(seed)))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # 训练结论依赖同 seed 的变体间对比；禁用 cudnn 自动调优并启用确定性算法
    # （warn_only 避免因个别无确定实现的 CUDA 算子直接中断，同时给出警告）。
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:  # 兼容没有 warn_only 参数的旧版 PyTorch
        torch.use_deterministic_algorithms(True)


def now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def load_json(path: str, default=None):
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def get_device(device: str = "auto") -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def enable_tf32(device: str) -> None:
    """CUDA 上启用 TF32 矩阵乘/卷积；bf16/AMP 路径不受影响，非 CUDA 设备为空操作。"""
    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def pearson(x, y):
    """Pearson 相关系数；样本不足/零方差/结果非有限时返回 None（调用方以 None 表示不可计算，
    避免 nan 写入 JSON 产生 NaN 字面量——那并非合法 JSON）。"""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m = ~np.isnan(x) & ~np.isnan(y)
    x, y = x[m], y[m]
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return None
    r = float(np.corrcoef(x, y)[0, 1])
    return r if np.isfinite(r) else None


def mean_std(values):
    """样本均值与样本标准差（ddof=1，无偏）。n<2 时 std 为 None。

    统计学约定：跨随机种子的结果必须报告 Mean ± Std（样本标准差），
    单种子结果无法评估方差，调用方应显式标注 n=1。
    """
    a = np.asarray([v for v in values if v is not None], dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return None, None
    if a.size == 1:
        return float(a[0]), None
    return float(a.mean()), float(a.std(ddof=1))


def _betacf(a: float, b: float, x: float) -> float:
    """正则化不完全 Beta 函数的连分式（Lentz 法，Numerical Recipes 口径）。"""
    MAXIT, EPS, FPMIN = 200, 3e-14, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < EPS:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    """正则化不完全 Beta 函数 I_x(a, b)。"""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def student_t_two_sided_p(t: float, df: float) -> float:
    """Student-t 分布双侧 p 值（无 scipy 依赖）。"""
    if not math.isfinite(t):
        return 0.0
    x = df / (df + t * t)
    return float(_betai(df / 2.0, 0.5, x))


def paired_t_test(a, b) -> dict:
    """同种子配对样本的双侧 t 检验：比较 a 与 b 两组一一对应的观测。

    实验协议中所有变体使用同一组随机种子训练，因此变体间比较天然配对：
    d_i = a_i - b_i 消除“种子运气”的公共方差，统计功效高于独立样本检验。
    返回 {n, mean_diff, std_diff, t, df, p, d_z}；n<2 时 t/df/p 为 None。
    d_z = mean_diff / std_diff 为配对 Cohen's d 效应量（方差为 0 时为 None）。
    标准差为 0（所有配对差完全相同）时：均值非 0 记 p=0，否则 p=1。
    """
    av = np.asarray([v for v in a if v is not None], dtype=np.float64)
    bv = np.asarray([v for v in b if v is not None], dtype=np.float64)
    if av.size != bv.size:
        raise ValueError(f"配对样本长度不一致: {av.size} vs {bv.size}")
    d = av - bv
    n = int(d.size)
    if n == 0:
        return {"n": 0, "mean_diff": None, "std_diff": None,
                "t": None, "df": None, "p": None, "d_z": None}
    mean_diff = float(d.mean())
    if n < 2:
        return {"n": n, "mean_diff": mean_diff, "std_diff": None,
                "t": None, "df": None, "p": None, "d_z": None}
    std_diff = float(d.std(ddof=1))
    df = n - 1
    if std_diff == 0.0:
        p = 1.0 if mean_diff == 0.0 else 0.0
        return {"n": n, "mean_diff": mean_diff, "std_diff": 0.0,
                "t": float("inf") if mean_diff else 0.0, "df": df, "p": p,
                "d_z": None}
    t = mean_diff / (std_diff / math.sqrt(n))
    return {"n": n, "mean_diff": mean_diff, "std_diff": std_diff,
            "t": float(t), "df": df, "p": student_t_two_sided_p(t, df),
            "d_z": float(mean_diff / std_diff)}


def holm_bonferroni(pvals) -> list:
    """Holm–Bonferroni 逐步下降法（step-down）校正的单步族错误率 p 值。

    同一次汇总会对多个 (候选, 参照) 配对做显著性检验；不校正时 α=0.05 下
    检验越多、至少一次假阳性的概率越大（family-wise error）。Holm 法在
    任意依赖结构下都控制 FWER，且比 Bonferroni 保守度更低。

    输入为 p 值列表（允许 None，表示无法检验，不占家族大小），返回同序列表：
    p_adj_(i) = max_{j<=i} min(1, (m-j+1) * p_(j))（p 升序的第 j 个观测），
    保证校正后 p 随原始 p 单调不减且 >= 原始 p。
    """
    idx = [i for i, p in enumerate(pvals) if p is not None]
    m = len(idx)
    adj = [None] * len(pvals)
    if m == 0:
        return adj
    running = 0.0
    for rank, i in enumerate(sorted(idx, key=lambda i: pvals[i])):
        val = min(1.0, float(pvals[i]) * (m - rank))
        running = max(running, val)
        adj[i] = running
    return adj


def auroc(scores, labels) -> float:
    """秩和法 AUROC（Mann-Whitney U），含并列取平均秩；labels: 1=正类。

    用于检验“门控分数能否区分这次读取是否找对了信息”。
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels)
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([neg, pos])
    order = np.argsort(allv, kind="mergesort")
    # 并列取平均秩：在有序数组上用 unique 定位等值组，
    # 组内平均秩（1-based）= 首名次（0-based 首个位置）+ (组大小+1)/2
    sorted_v = allv[order]
    first, counts = np.unique(sorted_v, return_index=True, return_counts=True)[1:]
    avg_ranks = first.astype(np.float64) + (counts + 1.0) / 2.0
    ranks = np.empty(len(allv), dtype=np.float64)
    ranks[order] = np.repeat(avg_ranks, counts)
    r_pos = ranks[len(neg):].sum()
    u = r_pos - len(pos) * (len(pos) + 1) / 2.0
    return float(u / (len(pos) * len(neg)))
