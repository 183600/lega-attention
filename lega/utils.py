"""通用工具：随机种子、JSON 读写、指标计算（AUROC/Pearson/统计检验，无 scipy 依赖）。

统计检验的设计口径：正式实验的种子数通常只有 3~5，配对 t 检验的正态假设在此
样本量下并不可靠；因此本模块同时提供
  * paired_t_test            —— 参数检验（主判定依据，历史口径保持不变）；
  * paired_sign_flip_test    —— 符号翻转置换检验（H0 下精确/分布无关，n<=16 时
                                全枚举精确，否则固定种子 Monte-Carlo）；
  * bootstrap_mean_ci        —— 配对差的百分位 bootstrap 置信区间；
  * min_detectable_effect    —— 给定种子数与配对差标准差的最小可检测效应，
                                用于区分"没有效应"与"功效不足"；
  * wilson_score_ci          —— 二值指标（RULER EM）的小样本比例置信区间。
汇总同时报告以上全部统计量；t 与置换检验结论冲突时会显式标注，而不是静默
采信任何一方。
"""
import json
import math
import os
import random
import statistics
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


def seed_report(seed: int) -> dict:
    """逐随机源的种子审计记录（写入 run_manifest/实验台账，供事后核对）。

    证明“这个 run 到底固定了哪些种子”，而不是口头声称：set_seed 覆盖的每个
    随机源都以实际传给它的 seed 记录在案；同时记录确定性开关与数据顺序机制。
    字段全部为 JSON 原生类型。``pythonhashseed_env`` 来自环境变量：若与 seed
    不同，说明当前进程的哈希盐在解释器启动时已固定（只能对子进程生效），
    属于已知边界而非静默失败。
    """
    return {
        "seed": int(seed),
        "python_random": int(seed),
        "numpy_global": int(seed),
        "torch_cpu": int(seed),
        "torch_cuda": (int(seed) if torch.cuda.is_available() else None),
        "pythonhashseed_env": os.environ.get("PYTHONHASHSEED"),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "deterministic_algorithms": bool(
            torch.are_deterministic_algorithms_enabled()),
        "data_order": "np.random.default_rng(seed) 显式批采样（无 DataLoader worker）",
    }


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
    """Student-t 分布双侧 p 值（无 scipy 依赖）。

    NaN 输入返回 NaN（不能当作最显著）；±Inf 返回 0。
    """
    if math.isnan(t):
        return float("nan")
    if math.isinf(t):
        return 0.0
    x = df / (df + t * t)
    return float(_betai(df / 2.0, 0.5, x))


def _paired_diffs(a, b) -> np.ndarray:
    """按位置提取配对差 a_i − b_i，供 t 检验/置换检验/bootstrap 共用。

    配对按位置严格一一对应：两侧长度必须一致，None 也必须成对出现，否则抛
    ValueError（None 位置错开意味着种子集合没有对齐，静默过滤会把不同种子
    的观测配到一起）。任一观测非有限（NaN/Inf）的配对整体剔除。
    """
    a_list, b_list = list(a), list(b)
    if len(a_list) != len(b_list):
        raise ValueError(f"配对样本长度不一致: {len(a_list)} vs {len(b_list)}")
    diffs = []
    for x, y in zip(a_list, b_list):
        if (x is None) != (y is None):
            raise ValueError("配对样本 None 位置不一致，无法按位置一一对应")
        if x is None or y is None:
            continue
        xf, yf = float(x), float(y)
        if not (math.isfinite(xf) and math.isfinite(yf)):
            continue
        diffs.append(xf - yf)
    return np.asarray(diffs, dtype=np.float64)


def paired_t_test(a, b) -> dict:
    """同种子配对样本的双侧 t 检验：比较 a 与 b 两组一一对应的观测。

    实验协议中所有变体使用同一组随机种子训练，因此变体间比较天然配对：
    d_i = a_i - b_i 消除“种子运气”的公共方差，统计功效高于独立样本检验。
    返回 {n, mean_diff, std_diff, t, df, p, d_z}；n<2 时 t/df/p 为 None。
    d_z = mean_diff / std_diff 为配对 Cohen's d 效应量（方差为 0 时为 None）。
    标准差为 0（所有配对差完全相同）时：均值非 0 记 p=0，否则 p=1。
    """
    d = _paired_diffs(a, b)
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


# 符号翻转置换检验的精确全枚举上限：2^16 = 65536 次枚举在现代 CPU 上毫秒级。
_SIGNFLIP_EXACT_MAX = 16


def paired_sign_flip_test(a, b, n_resamples: int = 20000,
                          seed: int = 0) -> dict:
    """配对样本的符号翻转置换检验（双侧，分布无关）。

    H0：配对差 d_i 关于 0 对称。检验统计量 T = |mean(d)|；H0 下对任意
    d_i 翻转符号得同样合法的样本，p = P(|mean(翻转样本)| >= |mean(d)|)。
    该检验不依赖正态假设，对 3~5 个种子的小样本推断比 t 检验更稳健：
    n<=16 时 2^n 次全枚举给出**精确** p 值；更大 n 用固定种子 Monte-Carlo
    （同一输入恒定复现）。零差值对任何翻转不变，天然计入分布。

    粒度边界（必须知情）：双侧 p 值的理论下限为 2/2^n——n=3 时 0.25、
    n=5 时 0.0625，n<6 时分布无关证据**本来就不可能**达到 α=0.05。因此
    小种子下"置换不显著"是粒度所限，不能单独推翻 t 检验结论；此时 t 的
    显著性完全依赖正态假设，应报告两者并补种子，而不是静默采信任何一方。

    配对规则与 paired_t_test 完全一致（_paired_diffs）。返回
    {n, mean_diff, p_perm, exact, n_resamples}；n<2 时不检验（p_perm=None，
    与 t 检验口径一致，避免 n=1 的"精确 p=1"被误读为证据）。
    """
    d = _paired_diffs(a, b)
    n = int(d.size)
    out = {"n": n, "mean_diff": None, "p_perm": None,
           "exact": None, "n_resamples": None}
    if n == 0:
        return out
    mean_diff = float(d.mean())
    out["mean_diff"] = mean_diff
    if n < 2:
        return out
    obs = abs(mean_diff)
    if obs == 0.0:
        # 全零差：任何翻转都不改变均值，p=1（无可检效应）。
        out.update({"p_perm": 1.0, "exact": True, "n_resamples": 2 ** n})
        return out
    if n <= _SIGNFLIP_EXACT_MAX:
        # 精确全枚举：2^n 个翻转样本（含观测自身与其全反号）。
        signs = (np.arange(2 ** n, dtype=np.int64)[:, None]
                 >> np.arange(n, dtype=np.int64)[None, :]) & 1
        flipped = d[None, :] * (1.0 - 2.0 * signs)
        ge = int((np.abs(flipped.mean(axis=1)) >= obs - 1e-15).sum())
        out.update({"p_perm": ge / float(2 ** n), "exact": True,
                    "n_resamples": int(2 ** n)})
        return out
    # Monte-Carlo：固定种子可复现。随机翻转每个 d_i 的符号（±1），
    # 双侧口径把观测自身与其全反号一并计入（+2），与精确枚举一致。
    rng = np.random.default_rng(seed)
    flips = rng.integers(0, 2, size=(int(n_resamples), n)) * 2.0 - 1.0
    stat = np.abs((d[None, :] * flips).mean(axis=1))
    ge = int((stat >= obs - 1e-15).sum())
    out.update({"p_perm": (ge + 2) / float(n_resamples + 2),
                "exact": False, "n_resamples": int(n_resamples)})
    return out


def bootstrap_mean_ci(values, n_resamples: int = 10000, ci: float = 0.95,
                      seed: int = 0):
    """样本均值的百分位 bootstrap 置信区间（配对差/任意标量观测通用）。

    None/非有限值剔除后 n<2 时返回 None（单点无区间可言）。
    固定随机种子，同一输入恒定复现。返回 (lo, hi)。
    """
    arr = np.asarray([v for v in values
                      if v is not None and math.isfinite(float(v))],
                     dtype=np.float64)
    if arr.size < 2:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(int(n_resamples), arr.size))
    means = arr[idx].mean(axis=1)
    alpha = (1.0 - float(ci)) / 2.0
    lo, hi = np.quantile(means, [alpha, 1.0 - alpha])
    return float(lo), float(hi)


def min_detectable_effect(std_diff, n: int, alpha: float = 0.05,
                          power: float = 0.8):
    """配对设计在给定 (配对差标准差, 种子数) 下的最小可检测效应（MDE）。

    正态近似（大样本 z 口径）：MDE = (z_{1-α/2} + z_{power}) · std/√n。
    用途：区分“不显著 = 无效应”与“不显著 = 功效不足”。报告的 MDE 远大于
    声称的效应量时，应先补种子而不是下结论；n>=3 时该近似已足够指引决策
    （严格 t 口径的临界值略大，近似值偏乐观，作解释性参考而非判定门槛）。
    标准差不可估计（None/0）或 n<2 时返回 None。
    """
    if std_diff is None or n is None or int(n) < 2:
        return None
    std = float(std_diff)
    if not (math.isfinite(std) and std > 0.0):
        return None
    nd = statistics.NormalDist()
    z = nd.inv_cdf(1.0 - float(alpha) / 2.0) + nd.inv_cdf(float(power))
    return float(z * std / math.sqrt(int(n)))


def wilson_score_ci(k: int, n: int, ci: float = 0.95):
    """二值比例的 Wilson 得分置信区间（小样本下远优于正态近似区间）。

    k 为成功数、n 为试验数；n<=0 返回 None。k=0 时下界恰为 0、k=n 时上界
    恰为 1（不会像 Wald 区间那样越界）。返回 (lo, hi)。
    """
    if n is None or k is None or int(n) <= 0:
        return None
    n = int(n)
    k = int(min(max(k, 0), n))
    z = statistics.NormalDist().inv_cdf(1.0 - (1.0 - float(ci)) / 2.0)
    z2 = z * z
    if k == 0:
        # 端点闭式：p=0 时区间为 [0, z²/(n+z²)]（避免浮点残差 1e-17 进 JSON）
        return 0.0, z2 / (n + z2)
    if k == n:
        # p=1 时区间为 [n/(n+z²), 1]
        return n / (n + z2), 1.0
    p = k / n
    den = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / den
    half = z * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n)) / den
    return max(0.0, center - half), min(1.0, center + half)


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
