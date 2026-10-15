"""通用工具：随机种子、JSON 读写、指标计算（AUROC/Pearson，无 scipy 依赖）。"""
import json
import os
import random
import time

import numpy as np
import torch


def set_seed(seed: int) -> None:
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
    # 组内平均秩 = 首名次 + (组大小+1)/2 - 0.5
    sorted_v = allv[order]
    first, counts = np.unique(sorted_v, return_index=True, return_counts=True)[1:]
    avg_ranks = first.astype(np.float64) + (counts + 1.0) / 2.0
    ranks = np.empty(len(allv), dtype=np.float64)
    ranks[order] = np.repeat(avg_ranks, counts)
    r_pos = ranks[len(neg):].sum()
    u = r_pos - len(pos) * (len(pos) + 1) / 2.0
    return float(u / (len(pos) * len(neg)))
