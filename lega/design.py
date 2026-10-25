"""实验设计与准备：先验功效/灵敏度、分析人群与设计符合性。

本模块补的是“开跑之前”的科研口径，而不是又一层事后统计：

* **先验灵敏度**：在花掉训练算力之前，回答“计划中的种子数 / RULER 每格
  样本数，能检出多大的效应？”。配对 t 用正态近似（与
  :func:`lega.utils.min_detectable_effect` 同口径）；RULER 的配对二值
  结果用 McNemar 精确功效（小样本）或正态近似（大样本）。输出的是
  **对一系列合理方差/不一致率假设的灵敏度曲线**，不把某一个任意假设
  伪装成数据估计；若配置显式声明了计划假设，则单独核对计划种子数是否
  足够。
* **分析人群**：实验开始前写死“计划内 (变体, 种子) 单元”和排除规则。
  事后只能因协议违规/产物损坏排除，不能因结果不利或离群排除。
* **设计符合性**：汇总时把实际产物与先验计划逐项比对；计划内单元缺失
  时，受影响的确认性假设自动降级为探索性，避免“只保留跑成功的种子”
  被当成预注册证据。
* **实测灵敏度**：用本次观测到的配对差标准差反推 MDE，与 SESOI 比较，
  描述“这次设计实际能分辨多小的效应”（描述性，不作显著性判定）。

产物 ``design_plan.json`` 在训练前写入结果目录，其摘要进入 repro /
汇总页 / significance.json；``design_conformance.json`` 记录计划与实际的
偏差。二者都是可机器读取的证据，而不是文档里的口头约定。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import operator
import os
import statistics
import sys
import time
from numbers import Real

DEFAULT_ALPHA = 0.05
DEFAULT_POWER = 0.8
MIN_SEEDS = 3
MAX_SEED = 2 ** 32 - 1
SCHEMA_VERSION = 1

# 确认性功效必须按“最终判定阈值”规划，而不是按名义 α。run_all 的判定要
# 同时通过同指标家族 Holm 与跨全部预注册假设的全局 Holm；Holm 在最坏情形
# 下把任一假设的阈值收紧到 α/m。这里用 max(家族大小, 全局确认族大小) 作为
# 保守负担，输出一份未校正、一份确认性校正后的灵敏度曲线。
DEFAULT_MULTIPLICITY = "holm_worst_case"
MULTIPLICITY_MODES = ("holm_worst_case", "family_only", "none")

# 与 run_all.SINGLE_FACTOR_PAIRS / CONTROL_VARIANTS 同步的事实来源。设计层
# 不导入 run_all（避免循环依赖与 CLI 拉起整条流水线），因此把“最终会进入
# 同一指标 Holm 家族的配对”显式列在这里，并由测试核对两边集合一致。
_SINGLE_FACTOR_PAIRS = (
    ("lega_no_query", "lega"),
    ("lega_no_conc", "lega"),
    ("lega_no_len", "lega"),
    ("qgate", "gate_only"),
    ("lega_conc_only", "gate_only"),
    ("lega_len_only", "gate_only"),
)
_PARAM_CONTROL_VARIANT = "lega_placebo"
_DEFAULT_GATED_FOR_ANALYSIS = (
    "lega", "qgate", "null", "lega_no_len", "lega_no_conc",
    "lega_no_query", "lega_conc_only", "lega_len_only",
)

# 精确 McNemar 功效的规模上限：每增加一个 n 要枚举不一致数 m 与条件拒绝域，
# 500 以内 O(n^2) 仍在毫秒级；更大时改用正态近似（见 mcnemar_power）。
EXACT_MCNEMAR_MAX_N = 500

# 先验灵敏度网格。它们是“展示设计在多大噪声下仍然够用”的横轴，不是从
# 当前数据估计出来的假设值；调用方可通过配置里的 design 块覆盖。
PPL_SD_GRID = (0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03)
RULER_DISCORDANCE_GRID = (0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5)
DEFAULT_RULER_DISCORDANCE_REF = 0.10
DEFAULT_PPL_SD_REF = 0.01

_TASKS_FOR_RULER = 5  # 与 lega.ruler.TASKS 一致；用于展示格子数与“最乐观
# 合并上界”的说明性计算，不进入任何确认性统计，故不在此导入 torch/ruler。

_ANALYSIS_PROTOCOL = {
    "analysis_population": (
        "全部计划内 (变体, 种子) 单元；每个单元在训练前写入 design_plan.json"),
    "exclusions": [
        "仅允许预先声明的排除：运行在评测前失败、数据/配置/训练预算协议违规、"
        "产物损坏或测试集访问违规",
        "禁止因结果不利、离群、显著性或“看起来不收敛”而事后排除某个种子/变体",
    ],
    "missing_data": (
        "缺失单元必须报告；受影响的确认性假设降级为探索性。可用样本估计"
        "（available-case）只能作为描述性敏感性分析，不能冒充计划内分析"),
    "stopping_rule": (
        "不做基于结果的提前停止；早停只允许按验证集与预注册 patience 执行，"
        "且实际训练量不等会使确认性判定降级（见 budget gate）"),
    "test_sealing": (
        "checkpoint 选择与早停只用 val；test 每次评测写访问账本，重复评测"
        "需要显式放行"),
    "multiplicity": (
        "同一指标家族内 Holm–Bonferroni；预注册确认族再做一次全局 Holm。"
        "先验功效按最坏情形 α/max(家族大小, 全局确认族大小) 规划，"
        "并同时报告未校正口径，避免用名义 α 高估可确认性"),
}


# ------------------------------------------------------------------ 输入校验
def _finite_float(value, name: str, *, positive: bool = False,
                  nonnegative: bool = False, upper: float = None) -> float:
    """严格有限浮点：bool/非数值/NaN/Inf 显式拒绝。"""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} 必须是数值，收到 {value!r}")
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} 无法转换为 float：{value!r}") from None
    if not math.isfinite(out):
        raise ValueError(f"{name} 必须是有限值，收到 {value!r}")
    if positive and not out > 0.0:
        raise ValueError(f"{name} 必须 > 0，收到 {value!r}")
    if nonnegative and out < 0.0:
        raise ValueError(f"{name} 必须 >= 0，收到 {value!r}")
    if upper is not None and out > upper:
        raise ValueError(f"{name} 必须 <= {upper}，收到 {value!r}")
    return out


def _strict_int(value, name: str, minimum: int = 1) -> int:
    """严格整数：bool/小数/非整数值拒绝；支持 Python 大整数。"""
    if isinstance(value, bool):
        raise ValueError(f"{name} 必须是整数，收到 bool {value!r}")
    try:
        out = operator.index(value)
    except TypeError:
        if isinstance(value, Real):
            try:
                f = float(value)
            except (TypeError, ValueError, OverflowError):
                f = None
            if f is not None and math.isfinite(f) and f.is_integer():
                out = int(f)
            else:
                raise ValueError(
                    f"{name} 必须是整数，收到 {value!r}") from None
        else:
            raise ValueError(f"{name} 必须是整数，收到 {value!r}") from None
    if out < minimum:
        raise ValueError(f"{name} 必须 >= {minimum}，收到 {out}")
    return out


def _seed_int(value, name: str = "seed") -> int:
    """设计计划里的种子与 :func:`lega.utils.seed_int` 同口径（0..2^32-1）。"""
    out = _strict_int(value, name, minimum=0)
    if out > MAX_SEED:
        raise ValueError(f"{name} 必须在 0..{MAX_SEED} 内，收到 {out}")
    return out


def _finite_sequence(values, name: str) -> list:
    if isinstance(values, (str, bytes)) or not isinstance(
            values, (list, tuple)):
        raise ValueError(f"{name} 必须是数值列表，收到 {values!r}")
    # 逐项走严格校验：bool/字符串/NaN/Inf 以及超出 float 上限的整数必须在
    # 网格入口拒绝，不能让 float(True)=1.0 之类的隐式转换静默进入设计。
    return [_finite_float(v, name) for v in values]


def _open_unit(value, name: str) -> float:
    out = _finite_float(value, name)
    if not (0.0 < out < 1.0):
        raise ValueError(f"{name} 必须在 (0,1) 内，收到 {out!r}")
    return out


# ------------------------------------------------------- 配对连续指标的功效
def _normal_two_sided_z(alpha: float) -> float:
    """双侧显著性水平 alpha 对应的正态临界值 z_{1-alpha/2}。

    alpha 小到 ``1-alpha/2`` 在 float 中舍入为 1 时（alpha < ~2e-16），
    ``NormalDist.inv_cdf`` 会抛 StatisticsError；此时改用 erfc 的尾概率
    二分求根。alpha 可小到最小次正规数 5e-324，此时 alpha/2 下溢为 0，
    因此直接在 erfc 上比较 alpha 本身。
    """
    nd = statistics.NormalDist()
    upper = 1.0 - alpha / 2.0
    if upper < 1.0:
        return nd.inv_cdf(upper)
    lo, hi = 0.0, 1.0
    while math.erfc(hi / math.sqrt(2.0)) > alpha:
        hi *= 2.0
        if hi > 1e3:
            return math.inf
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if math.erfc(mid / math.sqrt(2.0)) > alpha:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _normal_cdf(x: float) -> float:
    """标准正态 CDF；用 erfc 保住 z<-8 的尾部（NormalDist.cdf 会下溢为 0）。"""
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def _z_sum(alpha: float, power: float) -> float:
    nd = statistics.NormalDist()
    return _normal_two_sided_z(alpha) + nd.inv_cdf(power)


def _sqrt_count(n: int):
    """√n，允许 n 是超出 float 上限的合法大整数；√n 本身不可表示时返回 None。"""
    try:
        root = math.sqrt(n)
    except OverflowError:
        # math.sqrt(int) 会先把 n 转成 float；但 √n 仍可能可表示
        # （如 n=10**400 → 1e200）。用对数求根，超出 float 时如实返回 None。
        try:
            root = math.exp(0.5 * math.log(n))
        except (OverflowError, ValueError):
            return None
    if not (math.isfinite(root) and root > 0.0):
        return None
    return root


def paired_t_power(effect, sd, n, alpha: float = DEFAULT_ALPHA,
                   power: float = None) -> float:
    """配对 t 检验（双侧）的正态近似功效。

    ``effect`` 为待检出的配对差均值（方向无关，取绝对值），``sd`` 为配对差
    标准差，``n`` 为配对数（种子数）。公式：
    ``Φ(|d|/(σ/√n) − z_{1−α/2}) + Φ(−|d|/(σ/√n) − z_{1−α/2})``。

    这是 :func:`lega.utils.min_detectable_effect` 的逆运算；n 只有 3~5 时
    仍是近似值，用于设计灵敏度而不是精确功效承诺。``power`` 参数保留仅为
    与旧调用兼容，不参与计算。
    """
    del power  # 兼容占位；功效由 effect/sd/n/alpha 决定
    effect = _finite_float(effect, "effect", nonnegative=True)
    sd = _finite_float(sd, "sd", positive=True)
    n = _strict_int(n, "n", minimum=2)
    alpha = _open_unit(alpha, "alpha")
    root_n = _sqrt_count(n)
    # √n 超出 float 上限时，非零效应对应的 ncp 必为 +∞（功效 1），零效应
    # 则为 0（功效 α）；不能把合法的大 n 当成非法输入抛错。
    if root_n is None:
        ncp = math.inf if effect > 0.0 else 0.0
    else:
        scaled_sd = sd / root_n
        # sd/√n 可能下溢为 0（极大 n + 次正规 sd）：此时非零效应的 ncp
        # 实际为 +∞，不能执行 0 除法。
        ncp = (0.0 if effect == 0.0
               else (math.inf if scaled_sd == 0.0 else effect / scaled_sd))
    if not math.isfinite(ncp):
        return 1.0 if ncp > 0 else float(alpha)
    z = _normal_two_sided_z(alpha)
    out = _normal_cdf(ncp - z) + _normal_cdf(-ncp - z)
    return float(min(1.0, max(0.0, out)))


def min_detectable_effect(effect_sd, n, alpha: float = DEFAULT_ALPHA,
                          power: float = DEFAULT_POWER):
    """配对设计在给定 (配对差标准差, n) 下的 MDE（正态近似）。"""
    sd = _finite_float(effect_sd, "std_diff", positive=True)
    n = _strict_int(n, "n", minimum=2)
    alpha = _open_unit(alpha, "alpha")
    power = _open_unit(power, "power")
    z = _z_sum(alpha, power)
    root_n = _sqrt_count(n)
    if root_n is None:
        return None
    out = z * sd / root_n
    return float(out) if math.isfinite(out) else None


def required_pairs(effect, sd, alpha: float = DEFAULT_ALPHA,
                   power: float = DEFAULT_POWER, max_n: int = 1_000_000):
    """达到目标功效所需的最少配对数（种子数）；无法在 max_n 内达到返回 None。"""
    effect = _finite_float(effect, "effect", positive=True)
    sd = _finite_float(sd, "sd", positive=True)
    alpha = _open_unit(alpha, "alpha")
    power = _open_unit(power, "power")
    max_n = _strict_int(max_n, "max_n", minimum=2)
    try:
        n0 = math.ceil((_z_sum(alpha, power) * sd / effect) ** 2)
    except (OverflowError, ValueError):
        return None
    if not math.isfinite(n0):
        return None
    n0 = max(2, int(n0))
    if n0 > max_n:
        return None
    # 正态近似下功效单调；把 n0 调到最小的可行值，避免 ceil 的边界误差。
    while n0 > 2 and paired_t_power(effect, sd, n0 - 1, alpha) >= power:
        n0 -= 1
    while paired_t_power(effect, sd, n0, alpha) < power:
        n0 += 1
        if n0 > max_n:
            return None
    return int(n0)


def max_sd_for_effect(effect, n, alpha: float = DEFAULT_ALPHA,
                      power: float = DEFAULT_POWER):
    """在给定 n / 目标功效下，仍能检出 ``effect`` 的最大配对差 SD。

    这是“设计能容忍多大噪声”的反推：观测到的配对差 SD 超过该值，就说明
    当前种子数对 SESOI 的功效不足。不可计算时返回 None。
    """
    effect = _finite_float(effect, "effect", positive=True)
    n = _strict_int(n, "n", minimum=2)
    alpha = _open_unit(alpha, "alpha")
    power = _open_unit(power, "power")
    root_n = _sqrt_count(n)
    if root_n is None:
        return None
    out = effect * root_n / _z_sum(alpha, power)
    return float(out) if math.isfinite(out) else None


# ------------------------------------------------- 配对二值指标的功效（McNemar）
def _paired_binary_rates(discordance, effect):
    """把 (总体不一致率 π, 边际差 Δ) 换成 p10/p01。

    Δ = p10 − p01（候选独对 − 参照独对），π = p10 + p01，因此
    p10=(π+Δ)/2、p01=(π−Δ)/2。要求 |Δ| <= π，否则概率越界。
    """
    pi = _finite_float(discordance, "discordance", positive=True)
    if pi > 1.0:
        raise ValueError(f"discordance 必须 <= 1，收到 {pi!r}")
    d = _finite_float(effect, "effect")
    if abs(d) > pi + 1e-15:
        raise ValueError(
            f"|effect| 不能超过 discordance（{abs(d)} > {pi}）："
            "边际差不可能大于不一致率")
    p10 = (pi + d) / 2.0
    p01 = (pi - d) / 2.0
    # 浮点边界钳制，避免 -1e-17 之类的概率。
    p10 = min(1.0, max(0.0, p10))
    p01 = min(1.0, max(0.0, p01))
    return pi, p10, p01


def _mcnemar_critical_exact(m: int, alpha: float) -> int:
    """m 个不一致对下，精确双侧 McNemar 的拒绝边界 k（b<=k 或 c<=k）。

    H0 下 b~Bin(m, 0.5)，双侧 p=2·P(Bin(m,.5)<=min(b,c))（截断到 1）。
    返回最大的 k 使 2·P(Bin(m,.5)<=k) < alpha；m=0 或不存在时返回 -1。
    """
    if m <= 0:
        return -1
    mode = m // 2
    max_log = (math.lgamma(m + 1) - math.lgamma(mode + 1)
               - math.lgamma(m - mode + 1) - m * math.log(2.0))
    # 以众数 pmf 为尺度求和，避免 0.5^m 下溢。
    total = 0.0
    scaled = []
    for k in range(m + 1):
        logp = (math.lgamma(m + 1) - math.lgamma(k + 1)
                - math.lgamma(m - k + 1) - m * math.log(2.0))
        v = math.exp(logp - max_log)
        scaled.append(v)
        total += v
    if not (math.isfinite(total) and total > 0.0):
        return -1
    cum = 0.0
    last = -1
    for k, v in enumerate(scaled):
        cum += v
        if 2.0 * cum < alpha * total:
            last = k
        else:
            break
    return last


def _mcnemar_critical_normal(m: int, alpha: float) -> int:
    """大 m 下用正态近似得到拒绝边界（连续性校正）。"""
    if m <= 0:
        return -1
    z = _normal_two_sided_z(alpha)
    k = math.floor(m / 2.0 - 0.5 - z * math.sqrt(m) / 2.0)
    return int(max(-1, min(m, k)))


def _binom_cdf_exact(m: int, k: int, p: float) -> float:
    """P(Bin(m,p) <= k)，m<=EXACT_MCNEMAR_MAX_N 时直接求和（无下溢问题）。"""
    if k < 0:
        return 0.0
    if k >= m:
        return 1.0
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 0.0
    terms = []
    for b in range(k + 1):
        try:
            term = (math.comb(m, b) * (p ** b) * ((1.0 - p) ** (m - b)))
        except (OverflowError, ValueError):
            continue
        if math.isfinite(term):
            terms.append(term)
    if not terms:
        return 0.0
    return float(min(1.0, max(0.0, math.fsum(terms))))


def _mcnemar_power_exact(n: int, pi: float, p10: float,
                         alpha: float) -> float:
    """n<=EXACT_MCNEMAR_MAX_N 的精确 McNemar 功效。"""
    q = p10 / pi if pi > 0.0 else 0.0
    total = 0.0
    for m in range(1, n + 1):
        pm = (math.comb(n, m) * (pi ** m) * ((1.0 - pi) ** (n - m)))
        if not math.isfinite(pm) or pm <= 0.0:
            continue
        k = _mcnemar_critical_exact(m, alpha)
        if k < 0:
            continue
        # 拒绝域：b<=k 或 m-b<=k（即 b>=m-k）。b~Bin(m,q)。
        cond = _binom_cdf_exact(m, k, q) + (
            1.0 - _binom_cdf_exact(m, m - k - 1, q))
        total += pm * min(1.0, max(0.0, cond))
    return float(min(1.0, max(0.0, total)))


def _mcnemar_power_normal(n: int, pi: float, effect: float,
                          alpha: float) -> float:
    z = _normal_two_sided_z(alpha)
    root_n = _sqrt_count(n)
    ncp = ((math.inf if effect != 0.0 else 0.0) if root_n is None
           else root_n * abs(effect) / math.sqrt(pi))
    out = _normal_cdf(ncp - z) + _normal_cdf(-ncp - z)
    return float(min(1.0, max(0.0, out)))


def mcnemar_power(n, discordance, effect, alpha: float = DEFAULT_ALPHA) -> dict:
    """McNemar 精确检验（双侧）的功效。

    ``n`` 为配对样例数（RULER 每格样本数），``discordance`` 为总体不一致率
    π，``effect`` 为边际成功率差 Δ（候选−参照）。返回 dict 含 power、
    p10/p01、方法说明与期望不一致对数。

    n<=500 用精确枚举（H0 拒绝域 + 条件二项分布）；更大时用正态近似：
    power ≈ Φ(√n·|Δ|/√π − z_{1−α/2}) + Φ(−√n·|Δ|/√π − z_{1−α/2})。
    """
    n = _strict_int(n, "n", minimum=2)
    pi, p10, p01 = _paired_binary_rates(discordance, effect)
    alpha = _open_unit(alpha, "alpha")
    if n <= EXACT_MCNEMAR_MAX_N:
        pw = _mcnemar_power_exact(n, pi, p10, alpha)
        method = "exact"
    else:
        pw = _mcnemar_power_normal(n, pi, float(effect), alpha)
        method = "normal_approx"
    # n 是合法大整数时 n*π 可能超出 float 上限；期望不一致对数不可表示
    # 就记为缺失，不能让幂计算整体抛 OverflowError。
    try:
        expected = float(n * pi)
    except OverflowError:
        expected = None
    if expected is not None and not math.isfinite(expected):
        expected = None
    return {
        "n": n, "discordance": pi, "effect": float(effect),
        "p10": p10, "p01": p01,
        "expected_discordant": expected,
        "power": pw, "alpha": alpha, "method": method,
    }


def mcnemar_required_samples(effect, discordance,
                             alpha: float = DEFAULT_ALPHA,
                             power: float = DEFAULT_POWER,
                             max_n: int = 1_000_000):
    """达到目标 McNemar 功效所需的最少每格样本数；不可达返回 None。"""
    effect = _finite_float(effect, "effect", positive=True)
    pi, _p10, _p01 = _paired_binary_rates(discordance, effect)
    alpha = _open_unit(alpha, "alpha")
    power = _open_unit(power, "power")
    max_n = _strict_int(max_n, "max_n", minimum=2)
    try:
        n0 = math.ceil((_z_sum(alpha, power) ** 2) * pi / (effect ** 2))
    except (OverflowError, ValueError, ZeroDivisionError):
        # 极小但合法的 effect 在平方后会下溢为 0；所需样本数远超任何
        # 可行预算，按“不可达”处理，而不是让除零异常穿透设计入口。
        return None
    if not math.isfinite(n0):
        return None
    n0 = max(2, int(n0))
    # 正态近似只用于定位起点，可能比精确检验所需样本多；不能仅因起点
    # 超出 max_n 就判不可达——精确功效在 max_n 处可能已经达标。把起点
    # 截到预算上限，再按离散功效向下/向上修正。
    n0 = min(n0, max_n)

    def _pw(k: int) -> float:
        return mcnemar_power(k, pi, effect, alpha)["power"]

    # 离散检验的功效不严格单调（边界跳变），向下修正后向上找第一个可行值。
    while n0 > 2 and _pw(n0 - 1) >= power:
        n0 -= 1
    while _pw(n0) < power:
        n0 += 1
        if n0 > max_n:
            return None
    return int(n0)


def mcnemar_mde(n, discordance, alpha: float = DEFAULT_ALPHA,
                power: float = DEFAULT_POWER):
    """给定每格样本数与不一致率时，McNemar 可检出的最小边际差（正态近似）。"""
    n = _strict_int(n, "n", minimum=2)
    pi = _finite_float(discordance, "discordance", positive=True)
    if pi > 1.0:
        raise ValueError(f"discordance 必须 <= 1，收到 {pi!r}")
    alpha = _open_unit(alpha, "alpha")
    power = _open_unit(power, "power")
    root_n = _sqrt_count(n)
    if root_n is None:
        return None
    out = _z_sum(alpha, power) * math.sqrt(pi) / root_n
    return float(out) if math.isfinite(out) else None


def max_discordance_for_effect(effect, n, alpha: float = DEFAULT_ALPHA,
                               power: float = DEFAULT_POWER):
    """在给定 n / 目标功效下仍能检出 ``effect`` 的最大不一致率。"""
    effect = _finite_float(effect, "effect", positive=True)
    n = _strict_int(n, "n", minimum=2)
    alpha = _open_unit(alpha, "alpha")
    power = _open_unit(power, "power")
    z = _z_sum(alpha, power)
    root_n = _sqrt_count(n)
    if root_n is None:
        # n 大到 √n 不可表示：用对数判断是否已超过上界 1，避免先平方
        # effect 再乘 n 时中间溢出。
        try:
            log_out = (2.0 * math.log(effect) - 2.0 * math.log(z)
                       + math.log(n))
        except (OverflowError, ValueError):
            return None
        if log_out >= 0.0:
            return 1.0
        out = math.exp(log_out)
    else:
        scaled = abs(effect / z) * root_n
        # 结果上界是 1：先比较再平方，避免 ratio 已很大时平方抛
        # OverflowError（有限输入也可能触发）。
        if not math.isfinite(scaled) or scaled >= 1.0:
            return 1.0
        out = scaled * scaled
    if not math.isfinite(out):
        return None
    # 不一致率上界是 1：即使再大的 n 也不可能“容忍 >100% 的不一致”。
    return float(min(1.0, out))


# --------------------------------------------------------------- 计划与校验
def _as_dict(value):
    return value if isinstance(value, dict) else {}


def _as_list(value):
    return value if isinstance(value, list) else []


def _known_variant_names():
    """惰性获取可运行变体名；失败时返回空集合（计划仍可生成）。"""
    try:
        from lega.attention import RUNNABLE_VARIANTS
        return set(RUNNABLE_VARIANTS)
    except Exception:  # pragma: no cover - 导入失败只在极端环境发生
        return set()


def _check(check_id: str, title: str, status: str, detail: str) -> dict:
    if status not in ("pass", "warn", "fail", "info"):
        raise ValueError(f"未知检查状态 {status!r}")
    return {"id": check_id, "title": title, "status": status,
            "detail": detail}


def _checklist_status(checks) -> str:
    statuses = {c.get("status") for c in checks if isinstance(c, dict)}
    if "fail" in statuses:
        return "fail"
    if "warn" in statuses:
        return "warning"
    return "ok"


def planned_comparison_pairs(variants, registered: dict = None) -> list:
    """计划内会进入同一指标 Holm 家族的配对（与 run_all 构造一致）。

    设计层必须知道“最终要做多少次检验”，否则先验功效会按未校正的 5%
    计算，系统性高估确认性设计的灵敏度。这里复刻 run_all 的配对来源：
    各变体 vs standard、六个单因素对比、可选的参数匹配安慰剂对比，以及
    预注册表追加的整体对比；只保留候选和参照都在计划矩阵内的配对。
    """
    planned = [str(v) for v in (variants if isinstance(variants, (list, tuple))
                                else [])]
    planned_set = set(planned)
    pairs = [(v, "standard") for v in planned if v != "standard"]
    if _PARAM_CONTROL_VARIANT in planned_set:
        if "lega" in planned_set:
            pairs.append(("lega", _PARAM_CONTROL_VARIANT))
        if "qgate" in planned_set:
            pairs.append((_PARAM_CONTROL_VARIANT, "qgate"))
    pairs += [(c, r) for c, r in _SINGLE_FACTOR_PAIRS
              if c in planned_set and r in planned_set]
    for h in _as_list(_as_dict(registered).get("hypotheses")):
        if not isinstance(h, dict):
            continue
        metric = h.get("metric")
        # 机制族是“同 run 内 logit − 仅查询”的单样本检验，不进入配对族；
        # run_all 的 build_comparison_pairs 同样跳过它们。
        if isinstance(metric, str) and (
                metric.startswith("AUROC") or metric.startswith("AP差")):
            continue
        cand, ref = h.get("candidate"), h.get("reference")
        if (isinstance(cand, str) and isinstance(ref, str)
                and cand in planned_set and ref in planned_set):
            pairs.append((cand, ref))
    # 顺序稳定（同 run_all 的 dict.fromkeys），便于计划摘要复现。
    return list(dict.fromkeys(pairs))


def planned_family_sizes(variants, registered: dict = None,
                         analysis_variants=None) -> dict:
    """计划内各指标 Holm 家族大小与全局确认族大小。

    PPL/RULER 家族包含全部计划配对（包括探索性对比：run_all 的 Holm 是
    按指标整个家族校正，而不是只校正预注册条目）；AUROC/AP 机制族按每个
    有机制分析的变体各产生一个“完整 logit − 仅查询”检验；全局确认族只
    包含候选变体在计划内的预注册假设。缺失产物会让实际家族变小，因此这里
    是保守的设计上界。
    """
    planned = [str(v) for v in (variants if isinstance(variants, (list, tuple))
                                else [])]
    planned_set = set(planned)
    if analysis_variants is None:
        analysis_set = {v for v in planned if v in _DEFAULT_GATED_FOR_ANALYSIS}
    else:
        analysis_set = {str(v) for v in analysis_variants} & planned_set
    pairs = planned_comparison_pairs(planned, registered)
    hyps = [h for h in _as_list(_as_dict(registered).get("hypotheses"))
            if isinstance(h, dict)]
    # 全局确认族：候选必须真的会被训练/分析，否则该假设不会被检验。
    testable = [
        h for h in hyps
        if isinstance(h.get("candidate"), str)
        and (h.get("candidate") in planned_set
             or h.get("candidate") in analysis_set)
    ]
    return {
        "pairs": pairs,
        "n_pairs": len(pairs),
        "ppl_family_size": max(1, len(pairs)),
        "ruler_family_size": max(1, len(pairs)),
        "auroc_family_size": max(1, len(analysis_set)),
        "ap_family_size": max(1, len(analysis_set)),
        "registered_tests": len(hyps),
        "global_family_size": max(1, len(testable)),
    }


def _metric_family_size(metric, sizes: dict) -> int:
    """指标 → 该指标 Holm 家族的计划大小（未知指标按单检验处理）。"""
    name = metric if isinstance(metric, str) else ""
    if name.startswith("PPL"):
        return int(sizes.get("ppl_family_size") or 1)
    if name.startswith("RULER"):
        return int(sizes.get("ruler_family_size") or 1)
    if name.startswith("AUROC"):
        return int(sizes.get("auroc_family_size") or 1)
    if name.startswith("AP差") or name.startswith("PR-AUC"):
        return int(sizes.get("ap_family_size") or 1)
    return 1


def multiplicity_alpha(alpha: float, family_size: int,
                       global_family_size: int,
                       mode: str = DEFAULT_MULTIPLICITY) -> dict:
    """把名义 α 换算成先验功效规划用的确认性阈值（Holm 最坏情形）。

    ``holm_worst_case``：任一假设要同时通过家族 Holm 与全局 Holm，最坏
    情形下要求 p < α/max(m_family, m_global)；
    ``family_only``：只按家族 Holm；
    ``none``：仅报告未校正口径（不用于正式设计检查）。

    这是**保守下界**，不是精确的 Holm 功效：实际 Holm 阈值取决于同族其他
    p 值的大小。设计阶段不知道那些 p，只能用最坏情形避免高估。
    """
    if mode not in MULTIPLICITY_MODES:
        raise ValueError(
            f"未知 multiplicity={mode!r}；可选 {MULTIPLICITY_MODES}")
    alpha = _open_unit(alpha, "alpha")
    family_size = _strict_int(family_size, "family_size", minimum=1)
    global_family_size = _strict_int(global_family_size,
                                     "global_family_size", minimum=1)
    if mode == "none":
        burden = 1
    elif mode == "family_only":
        burden = family_size
    else:
        burden = max(family_size, global_family_size)
    try:
        effective = alpha / burden
    except OverflowError:
        # 合法但超出 float 可表示范围的家族大小（int 任意精度）；不能
        # 让“规划”因极端配置崩溃，下面按不可表示标记并回退到名义 α。
        effective = 0.0
    if not (math.isfinite(effective) and effective > 0.0):
        # 合法但极端大的家族大小可能让 α/m 下溢；此时不能伪造 0 阈值，
        # 按未校正 α 计算并在字段中如实标注不可表示的负担。
        effective = alpha
        representable = False
    else:
        effective = float(effective)
        representable = True
    return {
        "mode": mode,
        "alpha_raw": alpha,
        "family_size": family_size,
        "global_family_size": global_family_size,
        "burden": burden,
        "alpha_effective": effective,
        "representable": representable,
        "method": (
            "Holm–Bonferroni 最坏情形 α/max(家族, 全局族)；"
            "保守规划下界，不是精确 Holm 功效"),
    }


def build_design_plan(config: dict, seeds, variants, registered: dict = None,
                      *, config_file: str = None, config_sha256: str = None,
                      alpha: float = None, power: float = None,
                      ruler_n_samples: int = None, smoke: bool = False,
                      known_variants=None, analysis_variants=None,
                      multiplicity: str = None) -> dict:
    """根据配置/种子计划/预注册表生成训练前的设计计划。

    不读数据、不训练、不访问 test；只做输入解析与解析式功效计算，因此可以
    在正式训练之前（基线调优等准备步骤之后）生成并落盘。``registered`` 为登记表 dict（可空）；
    ``known_variants`` 用于把假设里的候选/参照映射到变体，缺省时惰性导入。

    返回 dict；``status`` 为 ok/warning/fail。``fail`` 不在此处中止实验，
    由调用方决定是否继续，但会进入汇总页与确认性降级证据。
    """
    config = _as_dict(config)
    if not isinstance(seeds, (list, tuple)) or not seeds:
        raise ValueError("seeds 必须是非空列表")
    seeds = [_seed_int(s, "seed") for s in seeds]
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds 不能重复")
    if not isinstance(variants, (list, tuple)) or not variants:
        raise ValueError("variants 必须是非空列表")
    variants = [str(v) for v in variants]
    if len(set(variants)) != len(variants):
        raise ValueError("variants 不能重复")

    design_cfg = _as_dict(config.get("design"))
    alpha = (_open_unit(alpha, "alpha") if alpha is not None
             else _open_unit(design_cfg.get("alpha", DEFAULT_ALPHA), "alpha"))
    power = (_open_unit(power, "power") if power is not None
             else _open_unit(design_cfg.get("power", DEFAULT_POWER), "power"))
    ppl_grid = design_cfg.get("ppl_paired_sd_rel_grid", list(PPL_SD_GRID))
    ruler_grid = design_cfg.get("ruler_discordance_grid",
                                list(RULER_DISCORDANCE_GRID))
    try:
        ppl_grid = sorted({_finite_float(x, "ppl_paired_sd_rel_grid",
                                         positive=True)
                           for x in _finite_sequence(ppl_grid,
                                                     "ppl_paired_sd_rel_grid")})
        ruler_grid = sorted({
            _finite_float(x, "ruler_discordance_grid", positive=True,
                          upper=1.0)
            for x in _finite_sequence(ruler_grid,
                                      "ruler_discordance_grid")})
    except ValueError as e:
        raise ValueError(f"设计网格非法：{e}") from None
    if not ppl_grid or not ruler_grid:
        raise ValueError("设计网格不能为空")

    ruler_cfg = _as_dict(config.get("ruler"))
    if ruler_n_samples is None:
        ruler_n_samples = ruler_cfg.get("n_samples", 30)
    ruler_n_samples = _strict_int(ruler_n_samples, "ruler n_samples",
                                 minimum=1)
    ctx_lens = ruler_cfg.get("ctx_lens")
    if isinstance(ctx_lens, (list, tuple)) and ctx_lens:
        n_contexts = len(set(str(c) for c in ctx_lens))
    else:
        n_contexts = 1

    registered = _as_dict(registered)
    hyps = [h for h in _as_list(registered.get("hypotheses"))
            if isinstance(h, dict)]
    scope = _as_dict(registered.get("scope"))
    scope_seeds = scope.get("seeds")
    scope_seeds = (scope_seeds if isinstance(scope_seeds, list) else None)
    scope_seeds_int = None
    if scope_seeds is not None:
        try:
            scope_seeds_int = [_seed_int(s, "预注册 scope seed")
                               for s in scope_seeds]
        except ValueError:
            scope_seeds_int = None

    known = (set(known_variants) if known_variants is not None
             else _known_variant_names())
    planned = set(variants)
    analysis_explicit = analysis_variants is not None
    if analysis_variants is None:
        # 默认只把真正会产出机制分析的门控变体列入分析人群；standard
        # 无输出门、不写 analysis_*.json，不能因此被判成“分析缺失”。
        analysis_variants = [v for v in variants
                             if v in _DEFAULT_GATED_FOR_ANALYSIS]
    if not isinstance(analysis_variants, (list, tuple)):
        raise ValueError("analysis_variants 必须是列表")
    analysis_variants = [str(v) for v in analysis_variants]
    unknown_analysis = sorted(set(analysis_variants) - planned)
    if unknown_analysis:
        raise ValueError(
            f"analysis_variants 含计划外变体：{unknown_analysis}")
    n_seeds = len(seeds)

    mode = (multiplicity if multiplicity is not None
            else design_cfg.get("multiplicity", DEFAULT_MULTIPLICITY))
    if not isinstance(mode, str) or mode not in MULTIPLICITY_MODES:
        raise ValueError(
            f"design.multiplicity 必须是 {MULTIPLICITY_MODES} 之一，"
            f"收到 {mode!r}")
    family_sizes = planned_family_sizes(
        variants, registered,
        analysis_variants if analysis_explicit else None)
    ppl_multiplicity = multiplicity_alpha(
        alpha, family_sizes["ppl_family_size"],
        family_sizes["global_family_size"], mode)
    ruler_multiplicity = multiplicity_alpha(
        alpha, family_sizes["ruler_family_size"],
        family_sizes["global_family_size"], mode)
    auroc_multiplicity = multiplicity_alpha(
        alpha, family_sizes["auroc_family_size"],
        family_sizes["global_family_size"], mode)
    ap_multiplicity = multiplicity_alpha(
        alpha, family_sizes["ap_family_size"],
        family_sizes["global_family_size"], mode)
    ppl_alpha = ppl_multiplicity["alpha_effective"]
    ruler_alpha = ruler_multiplicity["alpha_effective"]
    auroc_alpha = auroc_multiplicity["alpha_effective"]
    ap_alpha = ap_multiplicity["alpha_effective"]

    # ---- 检查 1：最小种子数（低于 3 不判定显著性，属协议内但只能探索）
    if n_seeds >= MIN_SEEDS:
        checks = [_check("seed_minimum", "种子数下限", "pass",
                         f"计划 {n_seeds} 个种子（>= {MIN_SEEDS}）")]
    else:
        checks = [_check(
            "seed_minimum", "种子数下限", "warn",
            f"计划只有 {n_seeds} 个种子（< {MIN_SEEDS}）：只能描述，"
            "不能做显著性/组件有效性判定")]

    # ---- 检查 2：与预注册种子计划一致（少跑不等于“范围外”，但确认性
    # 证据不再覆盖登记的全部种子；必须显式降级/告警）
    if scope_seeds_int is None:
        checks.append(_check(
            "seed_scope", "预注册种子计划", "warn",
            "登记表未提供可读的 scope.seeds，无法核对计划种子是否完整；"
            "确认性结论仍受范围门约束"))
    else:
        missing_scope = [s for s in scope_seeds_int if s not in seeds]
        extra_scope = [s for s in seeds if s not in scope_seeds_int]
        if missing_scope:
            checks.append(_check(
                "seed_scope", "预注册种子计划", "fail",
                f"计划缺少预注册种子 {missing_scope}：即使跑完，"
                "登记范围内仍有未观测种子；确认性判定应降级为探索性"))
        elif extra_scope:
            checks.append(_check(
                "seed_scope", "预注册种子计划", "warn",
                f"计划包含未登记种子 {extra_scope}：结论含登记计划外的抽样，"
                "证据强度打折"))
        else:
            checks.append(_check(
                "seed_scope", "预注册种子计划", "pass",
                f"计划种子与登记表一致（{seeds}）"))

    # ---- 检查 3：登记假设的候选/参照变体都必须在计划矩阵内
    missing_hyps = []
    unmapped = []
    for h in hyps:
        hid = h.get("id", "?")
        cand = h.get("candidate")
        ref = h.get("reference")
        for role, name in (("候选", cand), ("参照", ref)):
            if not isinstance(name, str) or not name:
                missing_hyps.append(f"{hid} 的{role}未命名")
                continue
            if name in known and name not in planned:
                missing_hyps.append(f"{hid} 的{role} {name} 不在计划变体中")
            elif name not in known and name not in planned:
                # “仅查询项基线”等分析内对照不是训练变体：只要求候选可映射。
                if role == "参照":
                    unmapped.append(f"{hid} 的参照 {name} 为分析内对照（无对应变体）")
                else:
                    missing_hyps.append(
                        f"{hid} 的候选 {name} 既不是已知变体也不在计划矩阵中")
    if not hyps:
        checks.append(_check("hypothesis_variants", "预注册假设覆盖", "warn",
                             "登记表无假设可核对"))
    elif missing_hyps:
        checks.append(_check(
            "hypothesis_variants", "预注册假设覆盖", "fail",
            "计划矩阵无法检验全部登记假设：" + "；".join(missing_hyps)))
    else:
        detail = f"{len(hyps)} 条登记假设的候选/训练变体均在计划矩阵中"
        if unmapped:
            detail += "；" + "；".join(sorted(set(unmapped))[:3])
        checks.append(_check("hypothesis_variants", "预注册假设覆盖",
                             "pass", detail))

    # ---- 检查 4：先验功效必须按最终确认阈值规划，而非名义 α
    mult_detail = (
        f"模式={mode}；PPL/RULER 计划家族 m="
        f"{family_sizes['ppl_family_size']}，全局确认族 k="
        f"{family_sizes['global_family_size']}；"
        f"PPL α_eff={ppl_alpha:.6g}、RULER α_eff={ruler_alpha:.6g}、"
        f"AUROC α_eff={auroc_alpha:.6g}、AP α_eff={ap_alpha:.6g}"
        f"（名义 α={alpha:.6g}）。按 Holm 最坏情形 α/max(m,k) 规划，"
        "避免用未校正阈值高估确认性功效")
    if not (ppl_multiplicity["representable"]
            and ruler_multiplicity["representable"]):
        mult_detail += "；警告：家族负担过大导致 α/m 下溢，校正阈值不可表示"
        mult_status = "warn"
    else:
        mult_status = "info"
    checks.append(_check("multiplicity_plan", "确认性多重比较规划",
                         mult_status, mult_detail))

    # ---- PPL 先验灵敏度：对合理 SD 网格给 MDE/功效/所需种子数
    ppl_margin = 0.01  # 与 lega.equivalence.metric_margin("PPL") 一致
    try:
        from lega.equivalence import metric_margin
        spec = metric_margin("PPL")
        if isinstance(spec, dict) and spec.get("margin") is not None:
            ppl_margin = _finite_float(spec.get("margin"), "PPL SESOI",
                                       positive=True)
    except Exception:  # pragma: no cover - 保持模块可独立使用
        pass
    ppl_rows = []
    can_power = n_seeds >= 2
    for sd in ppl_grid:
        if can_power:
            mde = min_detectable_effect(sd, n_seeds, alpha, power)
            pw = paired_t_power(ppl_margin, sd, n_seeds, alpha)
            mde_confirm = min_detectable_effect(
                sd, n_seeds, ppl_alpha, power)
            pw_confirm = paired_t_power(ppl_margin, sd, n_seeds, ppl_alpha)
        else:
            mde = pw = mde_confirm = pw_confirm = None
        ppl_rows.append({
            "paired_sd_relative": sd,
            "mde_relative": mde,
            "power_for_sesoi": pw,
            "required_seeds_for_sesoi": required_pairs(
                ppl_margin, sd, alpha, power),
            "mde_relative_confirmatory": mde_confirm,
            "power_for_sesoi_confirmatory": pw_confirm,
            "required_seeds_for_sesoi_confirmatory": required_pairs(
                ppl_margin, sd, ppl_alpha, power),
        })
    ppl_break_even = (max_sd_for_effect(ppl_margin, n_seeds, alpha, power)
                      if can_power else None)
    ppl_break_even_confirm = (
        max_sd_for_effect(ppl_margin, n_seeds, ppl_alpha, power)
        if can_power else None)
    declared_ppl_sd = None
    declared_disc = None
    if "ppl_paired_sd_rel" in design_cfg:
        declared_ppl_sd = design_cfg.get("ppl_paired_sd_rel")
        declared_ppl_sd = _finite_float(declared_ppl_sd,
                                        "design.ppl_paired_sd_rel",
                                        positive=True)
        need = required_pairs(ppl_margin, declared_ppl_sd, alpha, power)
        pw = (paired_t_power(ppl_margin, declared_ppl_sd, n_seeds, alpha)
              if can_power else None)
        ok = (can_power and need is not None and need <= n_seeds)
        checks.append(_check(
            "ppl_declared_power", "PPL 计划功效（声明假设）",
            "pass" if ok else "fail",
            f"声明配对差 SD={declared_ppl_sd:.4g}；{n_seeds} 种子对 "
            f"{ppl_margin:.2%} SESOI 的功效≈"
            + (f"{pw:.3f}" if pw is not None else "无法计算（n<2）")
            + "，"
            + (f"需 {need} 个种子（当前 {n_seeds}）" if need is not None
               else "所需种子数超出可计算范围")
            + ("" if ok else "：设计对 SESOI 功效不足")))
    else:
        checks.append(_check(
            "ppl_declared_power", "PPL 计划功效（声明假设）", "info",
            "配置未声明 design.ppl_paired_sd_rel：不把任一 SD 假设当作已知，"
            "仅报告灵敏度曲线（见 ppl.sensitivity）"))

    if declared_ppl_sd is not None:
        need_c = required_pairs(ppl_margin, declared_ppl_sd,
                                ppl_alpha, power)
        pw_c = (paired_t_power(ppl_margin, declared_ppl_sd,
                              n_seeds, ppl_alpha)
                if can_power else None)
        ok_c = (can_power and need_c is not None and need_c <= n_seeds)
        checks.append(_check(
            "ppl_confirmatory_power", "PPL 确认性计划功效（多重比较校正）",
            "pass" if ok_c else "fail",
            f"声明配对差 SD={declared_ppl_sd:.4g}；按 {mode} 的 "
            f"α_eff={ppl_alpha:.6g}（家族 m="
            f"{family_sizes['ppl_family_size']}，全局 k="
            f"{family_sizes['global_family_size']}），{n_seeds} 种子对 "
            f"{ppl_margin:.2%} SESOI 的功效≈"
            + (f"{pw_c:.3f}" if pw_c is not None else "无法计算（n<2）")
            + "，"
            + (f"需 {need_c} 个种子（当前 {n_seeds}）"
               if need_c is not None else "所需种子数超出可计算范围")
            + ("" if ok_c else "：通过多重比较后对 SESOI 功效不足，"
               "当前设计不能确认该效应")))
    else:
        checks.append(_check(
            "ppl_confirmatory_power", "PPL 确认性计划功效（多重比较校正）",
            "info",
            "未声明 design.ppl_paired_sd_rel；见 sensitivity 中的 "
            "power_for_sesoi_confirmatory"))

    # 参考 SD 不是数据估计，只用于展示“常见噪声水平下”的设计灵敏度。
    ref_sd = DEFAULT_PPL_SD_REF
    ref_pw = (paired_t_power(ppl_margin, ref_sd, n_seeds, alpha)
              if can_power else None)
    ref_pw_confirm = (paired_t_power(ppl_margin, ref_sd, n_seeds, ppl_alpha)
                      if can_power else None)
    ref_need = required_pairs(ppl_margin, ref_sd, alpha, power)
    ref_need_confirm = required_pairs(ppl_margin, ref_sd, ppl_alpha, power)
    ref_detail = (
        f"若配对差 SD={ref_sd:.1%}（参考值），{n_seeds} 种子对 "
        f"{ppl_margin:.1%} SESOI 的未校正功效≈"
        + (f"{ref_pw:.3f}" if ref_pw is not None else "无法计算（n<2）")
        + (f"；按确认性 α_eff={ppl_alpha:.6g} 仅≈{ref_pw_confirm:.3f}"
           if ref_pw_confirm is not None else "")
        + (f"，需 {ref_need} 个种子（校正后 {ref_need_confirm}）"
           if ref_need is not None and ref_need_confirm is not None
           else ""))
    if ppl_break_even is not None:
        ref_detail += f"；未校正 80% 功效下最大 SD≈{ppl_break_even:.3%}"
    if ppl_break_even_confirm is not None:
        ref_detail += f"（确认性 {ppl_break_even_confirm:.3%}）"
    checks.append(_check("ppl_reference_sensitivity",
                         "PPL 参考灵敏度（非数据估计）", "info",
                         ref_detail))
    if (ref_pw_confirm is not None and ref_pw_confirm < power):
        checks.append(_check(
            "ppl_confirmatory_reference", "PPL 确认性参考功效", "warn",
            f"参考 SD={ref_sd:.1%} 时，{n_seeds} 种子对 "
            f"{ppl_margin:.1%} SESOI 的确认性功效仅≈{ref_pw_confirm:.3f}"
            f"（目标 {power:.0%}；α_eff={ppl_alpha:.6g}，"
            f"需约 {ref_need_confirm} 个种子）。{n_seeds} 个种子只能"
            "可靠检出远大于 1% 的 PPL 差异；1% SESOI 结论应标为"
            "描述性/探索性"))

    # ---- 上下文档位设计：至少两档且跨度足够，才谈得上“长度校准”
    ctx_values = []
    if isinstance(ctx_lens, (list, tuple)):
        for c in ctx_lens:
            try:
                v = _finite_float(c, "ctx_lens", positive=True)
            except ValueError:
                continue
            ctx_values.append(v)
    if len(ctx_values) < 2:
        checks.append(_check(
            "ruler_context_span", "RULER 上下文档位", "warn",
            f"计划只有 {len(ctx_values)} 个有效上下文档位：无法检验长度"
            "剂量-反应/长度校准主张，跨长度平均也不能替代分档结果"))
    else:
        ratio = max(ctx_values) / min(ctx_values)
        if ratio < 2.0:
            checks.append(_check(
                "ruler_context_span", "RULER 上下文档位", "warn",
                f"上下文档位跨度仅 {min(ctx_values):g}–{max(ctx_values):g}"
                f"（×{ratio:.2f}）：长度效应可能被限制在过窄范围，"
                "只能作该范围内的结论"))
        else:
            checks.append(_check(
                "ruler_context_span", "RULER 上下文档位", "pass",
                f"上下文档位 {min(ctx_values):g}–{max(ctx_values):g}"
                f"（×{ratio:.2f}，{len(ctx_values)} 档）：可检验长度趋势"))

    # ---- RULER 先验灵敏度：每格样本数决定 McNemar 能检出多大的边际差
    ruler_margin = 0.01  # 与 RULER平均 SESOI 一致
    try:
        from lega.equivalence import metric_margin
        spec = metric_margin("RULER平均")
        if isinstance(spec, dict) and spec.get("margin") is not None:
            ruler_margin = _finite_float(spec.get("margin"), "RULER SESOI",
                                         positive=True)
    except Exception:  # pragma: no cover
        pass
    ruler_rows = []
    for disc in ruler_grid:
        try:
            cell = mcnemar_power(ruler_n_samples, disc, ruler_margin, alpha)
            mde = mcnemar_mde(ruler_n_samples, disc, alpha, power)
            need = mcnemar_required_samples(ruler_margin, disc, alpha, power)
            cell_confirm = mcnemar_power(
                ruler_n_samples, disc, ruler_margin, ruler_alpha)
            mde_confirm = mcnemar_mde(
                ruler_n_samples, disc, ruler_alpha, power)
            need_confirm = mcnemar_required_samples(
                ruler_margin, disc, ruler_alpha, power)
        except ValueError:
            continue
        ruler_rows.append({
            "discordance": disc,
            "power_for_sesoi": cell["power"],
            "mde_pp": mde,
            "required_samples_per_cell_for_sesoi": need,
            "power_for_sesoi_confirmatory": cell_confirm["power"],
            "mde_pp_confirmatory": mde_confirm,
            "required_samples_per_cell_for_sesoi_confirmatory": need_confirm,
        })
    ruler_break_even = max_discordance_for_effect(
        ruler_margin, ruler_n_samples, alpha, power)
    ruler_break_even_confirm = max_discordance_for_effect(
        ruler_margin, ruler_n_samples, ruler_alpha, power)
    declared_disc = design_cfg.get("ruler_discordance")
    if declared_disc is not None:
        declared_disc = _finite_float(declared_disc,
                                      "design.ruler_discordance",
                                      positive=True, upper=1.0)
        try:
            need = mcnemar_required_samples(ruler_margin, declared_disc,
                                            alpha, power)
            pw = mcnemar_power(ruler_n_samples, declared_disc,
                               ruler_margin, alpha)["power"]
        except ValueError as e:
            checks.append(_check(
                "ruler_declared_power", "RULER 计划功效（声明假设）",
                "fail", f"声明的不一致率无法计算：{e}"))
        else:
            ok = need is not None and need <= ruler_n_samples
            checks.append(_check(
                "ruler_declared_power", "RULER 计划功效（声明假设）",
                "pass" if ok else "warn",
                f"声明不一致率={declared_disc:.3g}；每格 "
                f"n={ruler_n_samples} 对 {ruler_margin:.2%} SESOI 的功效"
                f"≈{pw:.3f}"
                + (f"，需每格 {need} 个样例" if need is not None else "")
                + ("" if ok else "：当前 RULER 样本量不足以确认该 SESOI，"
                   "等价/优效判定只能作描述性证据")))
    else:
        checks.append(_check(
            "ruler_declared_power", "RULER 计划功效（声明假设）", "info",
            "配置未声明 design.ruler_discordance：仅报告灵敏度曲线"))

    if declared_disc is not None:
        try:
            need_c = mcnemar_required_samples(
                ruler_margin, declared_disc, ruler_alpha, power)
            pw_c = mcnemar_power(ruler_n_samples, declared_disc,
                                 ruler_margin, ruler_alpha)["power"]
        except ValueError as e:
            checks.append(_check(
                "ruler_confirmatory_power",
                "RULER 确认性计划功效（多重比较校正）", "fail",
                f"声明的 RULER 不一致率在确认性阈值下无法计算：{e}"))
        else:
            ok_c = need_c is not None and need_c <= ruler_n_samples
            checks.append(_check(
                "ruler_confirmatory_power",
                "RULER 确认性计划功效（多重比较校正）",
                "pass" if ok_c else "warn",
                f"声明不一致率={declared_disc:.3g}；按 {mode} 的 "
                f"α_eff={ruler_alpha:.6g}（家族 m="
                f"{family_sizes['ruler_family_size']}，全局 k="
                f"{family_sizes['global_family_size']}），每格 "
                f"n={ruler_n_samples} 对 {ruler_margin:.2%} SESOI 的功效"
                f"≈{pw_c:.3f}"
                + (f"，需每格 {need_c} 个样例" if need_c is not None else "")
                + ("" if ok_c else "：通过多重比较后样本量不足，"
                   "等价/优效判定只能作描述性证据")))
    else:
        checks.append(_check(
            "ruler_confirmatory_power",
            "RULER 确认性计划功效（多重比较校正）", "info",
            "未声明 design.ruler_discordance；见 sensitivity 中的 "
            "power_for_sesoi_confirmatory"))

    ref_disc = DEFAULT_RULER_DISCORDANCE_REF
    try:
        ref_cell = mcnemar_power(ruler_n_samples, ref_disc,
                                 ruler_margin, alpha)
        ref_need = mcnemar_required_samples(ruler_margin, ref_disc,
                                            alpha, power)
        ref_mde = mcnemar_mde(ruler_n_samples, ref_disc, alpha, power)
        ref_cell_confirm = mcnemar_power(
            ruler_n_samples, ref_disc, ruler_margin, ruler_alpha)
        ref_need_confirm = mcnemar_required_samples(
            ruler_margin, ref_disc, ruler_alpha, power)
        ref_mde_confirm = mcnemar_mde(
            ruler_n_samples, ref_disc, ruler_alpha, power)
    except ValueError:
        ref_cell, ref_need, ref_mde = {"power": None}, None, None
        ref_cell_confirm, ref_need_confirm, ref_mde_confirm = (
            {"power": None}, None, None)
    cells = n_contexts * _TASKS_FOR_RULER
    pooled_n = ruler_n_samples * cells
    try:
        pooled_mde = mcnemar_mde(pooled_n, ref_disc, alpha, power)
    except ValueError:
        pooled_mde = None
    pooled_grid = []
    for disc in ruler_grid:
        try:
            pooled_grid.append(mcnemar_mde(pooled_n, disc, alpha, power))
        except ValueError:
            continue
    pooled_min = min((v for v in pooled_grid if v is not None),
                     default=None)
    checks.append(_check(
        "ruler_reference_sensitivity", "RULER 参考灵敏度（非数据估计）",
        "info",
        f"若不一致率={ref_disc:.0%}（参考值），每格 n={ruler_n_samples} "
        f"对 {ruler_margin:.1%} SESOI 的未校正功效≈"
        + ((f"{ref_cell.get('power'):.3f}")
           if isinstance(ref_cell.get("power"), float) else "-")
        + (f"；按确认性 α_eff={ruler_alpha:.6g} 仅≈"
           f"{ref_cell_confirm.get('power'):.3f}"
           if isinstance(ref_cell_confirm.get("power"), float) else "")
        + (f"，MDE≈{ref_mde:.2%}"
           + (f"（确认性 {ref_mde_confirm:.2%}）"
              if ref_mde_confirm is not None else "")
           if ref_mde is not None else "")
        + (f"，需每格 {ref_need} 个样例"
           + (f"（确认性 {ref_need_confirm}）"
              if ref_need_confirm is not None else "")
           if ref_need is not None else "")
        + (f"；在 80% 功效下能容忍的最大不一致率≈{ruler_break_even:.4%}"
           + (f"（确认性 {ruler_break_even_confirm:.4%}）"
              if ruler_break_even_confirm is not None else "")
           if ruler_break_even is not None else "")))
    # 参考不一致率下若连 50% 功效都不到，明确告警：这不是“没有效应”，
    # 而是当前每格样本量根本不足以支撑 1pp 级别的实际意义判定。
    ref_power = ref_cell.get("power")
    ref_power_confirm = ref_cell_confirm.get("power")
    if (isinstance(ref_power, float) and ref_power < 0.5
            or isinstance(ref_power_confirm, float)
            and ref_power_confirm < 0.5):
        best_case = (
            f"即使把 {cells} 个格子当作完全独立的 {pooled_n} 个样例"
            f"（对宏观平均最乐观），MDE≈{pooled_mde:.2%}"
            + (f"、网格内最乐观不一致率下仍≈{pooled_min:.2%}"
               if pooled_min is not None else "")
            + "，"
            if pooled_mde is not None else "")
        checks.append(_check(
            "ruler_sample_size", "RULER 每格样本量", "warn",
            f"每格 n={ruler_n_samples} 在参考不一致率={ref_disc:.0%} 下"
            + (f"对 {ruler_margin:.1%} SESOI 的未校正功效仅≈"
               f"{ref_power:.3f}" if isinstance(ref_power, float) else "")
            + (f"；通过多重比较后仅≈{ref_power_confirm:.3f}"
               if isinstance(ref_power_confirm, float) else "")
            + "；"
            + best_case
            + "仍远大于 1pp：RULER 的 SESOI/等价判定只能作描述性证据；"
            "如需确认 1pp 级差异，应按上表增加每格样本数并重新登记"))

    # ---- 分析人群/排除规则必须在开跑前声明（不是事后补的叙述）
    checks.append(_check(
        "analysis_protocol", "分析人群与排除规则", "pass",
        "计划内 (变体, 种子) 全集；只允许协议违规/产物损坏排除，"
        "不得因结果不利或离群排除"))

    if smoke:
        # 冒烟/伪数据本就不在预注册范围；把 fail 降为 warn，避免把
        # “流程连通性测试”误报成正式设计失败，但仍在计划里留痕。
        for c in checks:
            if c["status"] == "fail":
                c["status"] = "warn"
                c["detail"] += "（冒烟运行：仅检查流程，不作确认性设计）"

    endpoint_ppl = {
        "metric": "PPL", "sesoi_relative": ppl_margin,
        "sesoi_source": "predeclared_protocol",
        "planned_seeds": n_seeds,
        "max_paired_sd_for_sesoi": ppl_break_even,
        "max_paired_sd_for_sesoi_confirmatory": ppl_break_even_confirm,
        "multiplicity": dict(ppl_multiplicity),
        "sensitivity": ppl_rows,
    }
    endpoint_ruler = {
        "metric": "RULER平均", "sesoi_pp": ruler_margin,
        "sesoi_source": "predeclared_protocol",
        "samples_per_cell": ruler_n_samples,
        "n_contexts": n_contexts,
        "n_tasks_assumed": _TASKS_FOR_RULER,
        "cells": cells,
        "pooled_samples_best_case": pooled_n,
        "mde_pp_pooled_best_case_at_reference": pooled_mde,
        "mde_pp_pooled_best_case_min_over_grid": pooled_min,
        "max_discordance_for_sesoi": ruler_break_even,
        "max_discordance_for_sesoi_confirmatory":
            ruler_break_even_confirm,
        "multiplicity": dict(ruler_multiplicity),
        "sensitivity": ruler_rows,
    }

    plan = {
        "schema_version": SCHEMA_VERSION,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": "lega.design",
        "inputs": {
            "config_file": config_file,
            "config_sha256": config_sha256,
            "variants": list(variants),
            "seeds": list(seeds),
            "n_seeds": n_seeds,
            "alpha": alpha,
            "alpha_effective": {
                "PPL": ppl_alpha, "RULER": ruler_alpha,
                "AUROC": auroc_alpha, "AP": ap_alpha},
            "multiplicity": mode,
            "power": power,
            "smoke": bool(smoke),
            "registered_hypotheses": [h.get("id") for h in hyps],
            "registration_scope_seeds": scope_seeds_int,
            "analysis_variants": list(analysis_variants),
            "known_variants": sorted(known) if known else None,
        },
        "analysis_protocol": dict(_ANALYSIS_PROTOCOL),
        # 确认性分析计划的只读快照：计划摘要因此同时绑定“登记了哪些
        # 假设、用什么指标/候选/参照”，不只是一个种子/样本量清单。
        "registered_plan": [
            {k: h.get(k) for k in ("id", "metric", "candidate", "reference",
                                   "direction")}
            for h in hyps],
        "assumptions": {
            "ppl_paired_sd_rel": declared_ppl_sd,
            "ruler_discordance": declared_disc,
            "reference_ppl_sd_rel": ref_sd,
            "reference_ruler_discordance": ref_disc,
            "multiplicity": mode,
            "note": (
                "声明假设来自配置 design 块（分析前写死）；参考值是展示用"
                "横轴，不是从当前数据估计的效应量，也不参与任何推断。"
                "确认性功效按最终 Holm 判定阈值规划，不能与未校正功效混用"),
        },
        "multiplicity_plan": {
            "mode": mode,
            "family_sizes": {
                "PPL": family_sizes["ppl_family_size"],
                "RULER": family_sizes["ruler_family_size"],
                "AUROC": family_sizes["auroc_family_size"],
                "AP": family_sizes["ap_family_size"],
                "global_confirmatory": family_sizes["global_family_size"],
            },
            "registered_tests": family_sizes["registered_tests"],
            "planned_comparison_pairs": [list(p) for p in family_sizes["pairs"]],
            "alpha_effective": {
                "PPL": ppl_alpha, "RULER": ruler_alpha,
                "AUROC": auroc_alpha, "AP": ap_alpha},
            "note": (
                "家族大小按计划内全部同指标配对计算（含探索性对比），"
                "全局族按计划内可检验的预注册假设计算；缺失产物只会让"
                "实际家族变小，故这是设计上界/保守口径"),
        },
        "endpoints": {"ppl": endpoint_ppl, "ruler": endpoint_ruler},
        "checks": checks,
        "status": _checklist_status(checks),
        "warnings": [c["detail"] for c in checks
                     if c["status"] in ("warn", "fail")],
    }
    plan["plan_digest"] = plan_digest(plan)
    return plan


def plan_digest(plan: dict) -> str:
    """计划内容的 SHA-256（排除 created_at/plan_digest 等非科学字段）。"""
    if not isinstance(plan, dict):
        raise ValueError("plan 必须是 dict")
    payload = {k: v for k, v in plan.items()
               if k not in ("created_at", "plan_digest")}
    # 配置文件的绝对路径是机器相关的运行位置，不是科学内容；占位文件系统
    # 位置差异不应让同一设计产生不同摘要（config_sha256 已绑定内容）。
    if isinstance(payload.get("inputs"), dict):
        payload["inputs"] = {k: v for k, v in payload["inputs"].items()
                             if k != "config_file"}
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _history_path(path: str) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(path)),
                        "design_plan_history.jsonl")


def load_design_plan_history(path: str) -> list:
    """读取设计计划历史（``path`` 可以是计划文件或所在目录）。

    每次 ``save_design_plan`` 追加一行（相同 digest 连续写入去重）。历史文件
    让“同一结果目录被重跑/改种子”时，旧计划不会随 ``design_plan.json`` 被
    静默覆盖；损坏行跳过，不影响当前计划。
    """
    if isinstance(path, (str, os.PathLike)):
        p = str(path)
        # 传目录时读目录下的历史；传计划文件时读同目录的历史
        # （不能把 plan JSON 本身当 JSONL 逐行解析）。
        p = (os.path.join(p, "design_plan_history.jsonl")
             if os.path.isdir(p) else _history_path(p))
    else:
        return []
    out = []
    try:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if isinstance(row, dict):
                    out.append(row)
    except OSError:
        return []
    return out


def save_design_plan(plan: dict, path: str) -> None:
    """原子写入计划，并追加到同目录的 ``design_plan_history.jsonl``。

    历史追加在计划写成功之后；连续写入同一 digest 时去重，避免每次
    ``--skip_train`` 汇总都制造重复记录。
    """
    from lega.utils import save_json
    save_json(plan, path)
    hist = _history_path(path)
    entry = {
        "created_at": plan.get("created_at"),
        "plan_digest": plan.get("plan_digest"),
        "status": plan.get("status"),
        "seeds": ((plan.get("inputs") or {}).get("seeds")
                  if isinstance(plan.get("inputs"), dict) else None),
        "variants": ((plan.get("inputs") or {}).get("variants")
                     if isinstance(plan.get("inputs"), dict) else None),
        "config_sha256": ((plan.get("inputs") or {}).get("config_sha256")
                          if isinstance(plan.get("inputs"), dict) else None),
    }
    try:
        prior = load_design_plan_history(hist)
        if prior and prior[-1].get("plan_digest") == entry["plan_digest"]:
            return
        os.makedirs(os.path.dirname(hist) or ".", exist_ok=True)
        with open(hist, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False,
                              separators=(",", ":")) + "\n")
    except OSError:
        # 历史写入失败不应毁掉主计划；主计划已原子落盘。
        pass


def load_design_plan(results_dir: str) -> dict:
    """读取结果目录的 design_plan.json；缺失/损坏/非对象返回 {}。"""
    if not isinstance(results_dir, (str, os.PathLike)):
        return {}
    path = os.path.join(results_dir, "design_plan.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            doc = json.load(f)
    except (OSError, ValueError):
        return {}
    return doc if isinstance(doc, dict) else {}


# ------------------------------------------------------- 分析人群 / 设计符合性
def _cell_set(variants, seeds):
    return [(v, s) for v in variants for s in seeds]


def analysis_population(plan: dict, runs: dict) -> dict:
    """把实际发现的 run 与计划内 (变体, 种子) 单元逐一比对。

    ``runs`` 为 ``{variant: {seed: {"eval":..., "ruler":..., "analysis":...}}}``
    （build_summary 的发现结构）。只按“是否真的产出对应产物”判断，不按
    结果好坏/是否显著筛选。
    """
    plan = _as_dict(plan)
    inputs = _as_dict(plan.get("inputs"))
    variants = [str(v) for v in _as_list(inputs.get("variants"))]
    analysis_variants = inputs.get("analysis_variants")
    analysis_variants = (set(str(v) for v in analysis_variants)
                         if isinstance(analysis_variants, list) else set(variants))
    seeds = [s for s in _as_list(inputs.get("seeds"))
             if isinstance(s, int) and not isinstance(s, bool)]
    runs = _as_dict(runs)
    expected = _cell_set(variants, seeds)
    missing_eval, missing_ruler, missing_analysis = [], [], []
    observed_eval = []
    by_variant = {}
    for v in variants:
        vm = _as_dict(runs.get(v))
        row = {"planned_seeds": list(seeds), "observed_seeds": [],
               "missing_eval": [], "missing_ruler": [],
               "missing_analysis": []}
        for s in seeds:
            slot = _as_dict(vm.get(s))
            label = f"{v}/seed{s}"
            has_eval = isinstance(slot.get("eval"), dict)
            has_ruler = isinstance(slot.get("ruler"), dict)
            has_analysis = isinstance(slot.get("analysis"), dict)
            if has_eval:
                observed_eval.append((v, s))
                row["observed_seeds"].append(s)
            else:
                missing_eval.append(label)
                row["missing_eval"].append(s)
            if not has_ruler:
                missing_ruler.append(label)
                row["missing_ruler"].append(s)
            if v in analysis_variants and not has_analysis:
                missing_analysis.append(label)
                row["missing_analysis"].append(s)
        by_variant[v] = row
    extra_variants = sorted(str(v) for v in runs
                            if str(v) not in set(variants))
    observed_seed_set = set()
    for v in variants:
        observed_seed_set.update(_as_dict(runs.get(v)).keys())
    extra_seeds = sorted((str(s) for s in observed_seed_set
                          if s not in set(seeds)))
    return {
        "expected_cells": len(expected),
        "observed_eval_cells": len(observed_eval),
        "missing_eval": missing_eval,
        "missing_ruler": missing_ruler,
        "missing_analysis": missing_analysis,
        "complete_eval": not missing_eval,
        "complete_ruler": not missing_ruler,
        "complete_analysis": not missing_analysis,
        "extra_variants": extra_variants,
        "extra_seeds": extra_seeds,
        "by_variant": by_variant,
    }


def _stage_enabled(stages, key: str) -> bool:
    if not isinstance(stages, dict):
        return True
    value = stages.get(key)
    # 未记录阶段（旧 repro）按启用处理；显式 false 才是关闭。
    return value is not False


def design_conformance(plan: dict, repro: dict, runs: dict,
                       audit: dict = None, stages: dict = None) -> dict:
    """核对“先验设计计划”与本次实际 run 的一致性。

    返回 ``checked``（计划是否存在且与本次 repro 匹配）、``in_plan``、
    ``problems``/``warnings`` 与 ``population``。``checked=False`` 时不据此
    降级（旧产物/手工汇总保持兼容），但汇总页会显式告警。
    """
    plan = _as_dict(plan)
    repro = _as_dict(repro)
    out = {"checked": False, "in_plan": None, "reason": None,
           "problems": [], "warnings": [], "population": None,
           "plan_digest": plan.get("plan_digest"),
           "plan_status": plan.get("status")}
    if not plan:
        out["reason"] = "missing_plan"
        out["warnings"].append(
            "结果目录缺少 design_plan.json（旧产物或未走统一入口）："
            "无法核对先验设计/分析人群，不据此降级确认性判定")
        return out
    if plan.get("schema_version") != SCHEMA_VERSION:
        out["reason"] = "unsupported_schema"
        out["warnings"].append(
            f"design_plan.json schema={plan.get('schema_version')!r} 不受支持，"
            "忽略该计划")
        return out

    inputs = _as_dict(plan.get("inputs"))
    plan_variants = [str(v) for v in _as_list(inputs.get("variants"))]
    plan_seeds = [s for s in _as_list(inputs.get("seeds"))
                  if isinstance(s, int) and not isinstance(s, bool)]
    repro_variants = [str(v) for v in _as_list(repro.get("variants"))]
    repro_seeds = [s for s in _as_list(repro.get("seeds"))
                   if isinstance(s, int) and not isinstance(s, bool)]
    if repro.get("smoke") is True:
        out["reason"] = "smoke_run"
        out["warnings"].append(
            "本次为冒烟运行：设计计划只检查流程，不用于确认性设计符合性判定")
        return out
    if set(plan_variants) != set(repro_variants):
        out["reason"] = "plan_repro_variant_mismatch"
        out["warnings"].append(
            "design_plan.json 的变体集与 repro.json 不一致（计划可能是上一轮"
            "遗留）：不据此核对/降级")
        return out
    if set(plan_seeds) != set(repro_seeds):
        out["reason"] = "plan_repro_seed_mismatch"
        out["warnings"].append(
            "design_plan.json 的种子集与 repro.json 不一致（计划可能是上一轮"
            "遗留）：不据此核对/降级")
        return out
    plan_sha = inputs.get("config_sha256")
    repro_sha = repro.get("config_sha256")
    if (isinstance(plan_sha, str) and isinstance(repro_sha, str)
            and plan_sha and repro_sha and plan_sha != repro_sha):
        out["reason"] = "plan_repro_config_mismatch"
        out["warnings"].append(
            "design_plan.json 的配置 SHA-256 与 repro.json 不一致：计划可能"
            "对应旧配置，不据此核对/降级")
        return out

    pop = analysis_population(plan, runs)
    out["population"] = pop
    out["checked"] = True
    if not pop["complete_eval"]:
        out["problems"].append(
            f"计划内 eval 单元缺失 {len(pop['missing_eval'])} 个："
            + ", ".join(pop["missing_eval"][:8])
            + (" ..." if len(pop["missing_eval"]) > 8 else ""))
    if _stage_enabled(stages, "ruler") and not pop["complete_ruler"]:
        out["problems"].append(
            f"计划内 RULER 单元缺失 {len(pop['missing_ruler'])} 个："
            + ", ".join(pop["missing_ruler"][:8])
            + (" ..." if len(pop["missing_ruler"]) > 8 else ""))
    if _stage_enabled(stages, "analysis") and not pop["complete_analysis"]:
        out["problems"].append(
            f"计划内机制分析单元缺失 {len(pop['missing_analysis'])} 个："
            + ", ".join(pop["missing_analysis"][:8])
            + (" ..." if len(pop["missing_analysis"]) > 8 else ""))
    if pop["extra_variants"]:
        out["warnings"].append(
            f"实际产物含计划外变体 {pop['extra_variants']}（不进入本计划的"
            "分析人群，也不据此提升证据）")
    if pop["extra_seeds"]:
        out["warnings"].append(
            f"实际产物含计划外种子 {pop['extra_seeds']}（不进入本计划的"
            "分析人群）")
    audit = _as_dict(audit)
    audit_missing = _as_text_list(audit.get("missing_evals"))
    if audit_missing and not out["problems"]:
        # 产物检查与受控变量审计互相印证；审计缺失但 runs 里存在（例如
        # out/ 被清理）时也要留告警。
        out["warnings"].append(
            f"受控变量审计报告缺少 {len(audit_missing)} 个 eval.json："
            + ", ".join(audit_missing[:6]))
    # 计划本身的检查失败也要进入符合性结论：少跑预注册种子时，
    # (变体, 种子) 人口可能“看起来完整”（因为计划就是用少了种子的清单
    # 生成的），但设计已不在登记范围内。seed_scope / hypothesis_variants
    # 是全局设计缺陷，由 apply_design_gate 对所有相关假设降级；声明功效
    # 不足只作告警，不直接推翻统计判定。
    plan_checks = [c for c in _as_list(plan.get("checks"))
                   if isinstance(c, dict)]
    failures = [c for c in plan_checks if c.get("status") == "fail"]
    out["plan_failures"] = [
        {"id": c.get("id"), "detail": c.get("detail")} for c in failures]
    global_ids = {"seed_scope", "hypothesis_variants"}
    out["global_plan_gate"] = any(
        c.get("id") in global_ids for c in failures)
    if out["global_plan_gate"]:
        out["problems"].append(
            "先验设计检查失败（全局）："
            + "；".join(str(c.get("detail")) for c in failures
                       if c.get("id") in global_ids))
    out["in_plan"] = not out["problems"]
    return out


def _as_text_list(value):
    return [str(x) for x in value] if isinstance(value, list) else []


def _metric_endpoint(metric: str) -> str:
    if not isinstance(metric, str):
        return "eval"
    if metric.startswith("RULER"):
        return "ruler"
    if metric.startswith("AUROC") or metric.startswith("AP差") or \
            metric.startswith("PR-AUC"):
        return "analysis"
    return "eval"


def _affected_variants(hyp: dict, plan_variants) -> set:
    """假设涉及的训练变体；分析内对照（如“仅查询项基线”）不算。"""
    out = set()
    planned = set(plan_variants)
    for key in ("candidate", "reference"):
        name = hyp.get(key)
        if isinstance(name, str) and name in planned:
            out.add(name)
    return out


def apply_design_gate(hyp_eval, conformance: dict, plan: dict = None):
    """计划内单元缺失时，受影响假设的确认性判定自动降级。

    只降级“缺失单元涉及该假设候选/参照变体或该指标端点”的条目；未受影响
    的假设保留原判定。计划缺失/与本次运行不匹配时不做降级（向后兼容），
    但写入 ``design_population_ok=None`` 让读者知道人群未被核对。

    与 scope/integrity/budget 门同义：设计上不可核对的“支持/拒绝”不能
    冒充预注册证据；降级只改变证据等级，不改变统计量本身。
    """
    if not isinstance(hyp_eval, list):
        return hyp_eval
    conf = _as_dict(conformance)
    plan = _as_dict(plan) if plan is None else _as_dict(plan)
    pop = _as_dict(conf.get("population"))
    inputs = _as_dict(plan.get("inputs"))
    plan_variants = [str(v) for v in _as_list(inputs.get("variants"))]
    checked = conf.get("checked") is True
    global_fail = (checked and conf.get("global_plan_gate") is True)
    global_reason = (
        "先验设计检查失败："
        + "；".join(str(f.get("detail")) for f in
                   _as_list(conf.get("plan_failures"))
                   if isinstance(f, dict)
                   and f.get("id") in ("seed_scope", "hypothesis_variants"))
        if global_fail else "")
    for h in hyp_eval:
        if not isinstance(h, dict):
            continue
        if not checked:
            h["design_population_ok"] = None
            h["design_note"] = (
                "设计计划缺失或与本次运行不匹配，无法核对分析人群；"
                "未据此降级，但该判定不在先验设计证据之内")
            continue
        if global_fail:
            h["design_population_ok"] = False
            h["design_note"] = global_reason
            if h.get("confirmatory_valid") is not False:
                h["confirmatory_valid"] = False
                h["confirmatory_note"] = global_reason
            continue
        endpoint = _metric_endpoint(h.get("metric"))
        missing = _as_text_list(pop.get({
            "eval": "missing_eval", "ruler": "missing_ruler",
            "analysis": "missing_analysis"}[endpoint]))
        affected = _affected_variants(h, plan_variants)
        missing_variants = set()
        for cell in missing:
            v = cell.split("/seed", 1)[0]
            if v in affected:
                missing_variants.add(v)
        if not missing_variants:
            h["design_population_ok"] = True
            h["design_note"] = (
                f"计划内 {endpoint} 单元完整（{len(affected)} 个相关变体）")
            continue
        h["design_population_ok"] = False
        reason = (
            "分析人群不完整：计划内单元缺失涉及 "
            + ", ".join(sorted(missing_variants))
            + f" 的 {endpoint} 产物；不得只保留跑成功的种子，"
            "该确认性判定降级为探索性")
        h["design_note"] = reason
        if h.get("confirmatory_valid") is False:
            continue  # 已因更早的完整性/范围/预算门降级，保留原原因
        h["confirmatory_valid"] = False
        h["confirmatory_note"] = reason
    return hyp_eval


# --------------------------------------------------------------- 实测灵敏度
def observed_design_sensitivity(tests) -> list:
    """用本次实测配对差 SD 反推每个对比的 MDE，并与 SESOI 比较（描述性）。

    注意这不是“观测功效”（observe power）——它不把 p 值换个说法，而是回答
    “以本次实际方差，这个设计能分辨多小的效应”。与 SESOI 同尺度：
    ``equivalence`` 已把 PPL 换成相对差、RULER/AUROC 换成绝对差。
    """
    rows = []
    for t in (tests if isinstance(tests, list) else []):
        if not isinstance(t, dict):
            continue
        eq = t.get("equivalence")
        if not isinstance(eq, dict) or eq.get("status") == "invalid":
            continue
        sd = eq.get("std_diff")
        n = eq.get("n")
        margin = eq.get("margin")
        try:
            sd_f = float(sd)
            margin_f = float(margin)
        except (TypeError, ValueError, OverflowError):
            continue
        if not isinstance(n, int) or isinstance(n, bool) or n < 2:
            continue
        if not (math.isfinite(sd_f) and sd_f >= 0.0
                and math.isfinite(margin_f) and margin_f > 0.0):
            continue
        mde = (0.0 if sd_f == 0.0
               else min_detectable_effect(sd_f, n, DEFAULT_ALPHA,
                                         DEFAULT_POWER))
        try:
            pw = paired_t_power(margin_f, sd_f, n, DEFAULT_ALPHA) \
                if sd_f > 0.0 else 1.0
        except ValueError:
            pw = None
        rows.append({
            "candidate": t.get("candidate"),
            "reference": t.get("reference"),
            "metric": t.get("metric"),
            "n": n,
            "std_diff_sesoi_units": sd_f,
            "sesoi_margin": margin_f,
            "observed_mde_sesoi_units": mde,
            "power_for_sesoi": pw,
            "adequate_for_sesoi": (None if mde is None
                                   else bool(mde <= margin_f)),
        })
    return rows


# ------------------------------------------------------------------- 渲染
def _fmt(x, nd=4, pct=False, signed=False):
    if x is None:
        return "-"
    try:
        f = float(x)
    except (TypeError, ValueError, OverflowError):
        return "-"
    if not math.isfinite(f):
        return "-"
    if pct:
        return f"{f * 100:+.{nd}f}%" if signed else f"{f * 100:.{nd}f}%"
    return f"{f:+.{nd}f}" if signed else f"{f:.{nd}f}"


def render_design_section(plan: dict, conformance: dict = None,
                          observed: list = None) -> list:
    """把先验计划/设计符合性/实测灵敏度渲染成 summary.md 小节。"""
    md = ["", "## 实验设计与分析人群（先验计划 vs 实际）", ""]
    plan = _as_dict(plan)
    conf = _as_dict(conformance)
    if not plan:
        md.append("- ⚠ 未找到 `design_plan.json`（旧产物或未走统一入口）："
                  "本次无法核对先验设计、分析人群与缺失单元；"
                  "这不等于设计没有问题，只说明没有可核对证据。")
        return md
    inputs = _as_dict(plan.get("inputs"))
    status = plan.get("status")
    icon = {"ok": "✅", "warning": "⚠", "fail": "❌"}.get(status, "•")
    md.append(
        f"- {icon} 计划状态：`{status}`；变体 {len(_as_list(inputs.get('variants')))} "
        f"× 种子 {len(_as_list(inputs.get('seeds')))}；"
        f"alpha={_fmt(inputs.get('alpha'), 3)}、"
        f"power={_fmt(inputs.get('power'), 3)}；"
        f"SHA-256 `{str(plan.get('plan_digest'))[:12]}`（训练前落盘）")
    md.append("- 先验计划只使用配置与预注册范围，不读取任何结果/测试集；"
              "种子数或 RULER 样本量不足时不会在事后被解释成“无效应”。")
    mp = _as_dict(plan.get("multiplicity_plan"))
    if mp:
        sizes = _as_dict(mp.get("family_sizes"))
        md.append(
            f"- 确认性多重比较：`{mp.get('mode')}`；PPL 家族 m="
            f"{sizes.get('PPL')}、RULER 家族 m={sizes.get('RULER')}、"
            f"全局确认族 k={sizes.get('global_confirmatory')}；"
            f"PPL/RULER/AUROC/AP 有效阈值 α_eff="
            f"{_fmt(_as_dict(mp.get('alpha_effective')).get('PPL'), 4)}/"
            f"{_fmt(_as_dict(mp.get('alpha_effective')).get('RULER'), 4)}/"
            f"{_fmt(_as_dict(mp.get('alpha_effective')).get('AUROC'), 4)}/"
            f"{_fmt(_as_dict(mp.get('alpha_effective')).get('AP'), 4)}。"
            "先验功效同时给出未校正与确认性校正两栏；正式判定以后者为准。")

    checks = [c for c in _as_list(plan.get("checks")) if isinstance(c, dict)]
    if checks:
        md += ["", "| 设计检查 | 状态 | 说明 |", "|---|---|---|"]
        for c in checks:
            st = {"pass": "✅ pass", "warn": "⚠ warn",
                  "fail": "❌ fail", "info": "ℹ info"}.get(
                      c.get("status"), str(c.get("status")))
            detail = str(c.get("detail", "")).replace("|", "\\|")
            md.append(f"| {c.get('title')} | {st} | {detail} |")

    ppl = _as_dict(_as_dict(plan.get("endpoints")).get("ppl"))
    if ppl:
        md += ["", f"**PPL 先验灵敏度**（SESOI={_fmt(ppl.get('sesoi_relative'), 2, pct=True)}；"
                   f"计划 {ppl.get('planned_seeds')} 个种子；"
                   f"未校正 80% 功效下可容忍的配对差 SD≈"
                   f"{_fmt(ppl.get('max_paired_sd_for_sesoi'), 3, pct=True)}；"
                   f"确认性校正后≈"
                   f"{_fmt(ppl.get('max_paired_sd_for_sesoi_confirmatory'), 3, pct=True)}）", "",
               "| 配对差SD(相对) | MDE 未校正 | MDE 确认性 | 功效 未校正 | 功效 确认性 | 所需种子 未校正 | 所需种子 确认性 |",
               "|---|---|---|---|---|---|---|"]
        for row in _as_list(ppl.get("sensitivity")):
            if not isinstance(row, dict):
                continue
            md.append("| " + " | ".join([
                _fmt(row.get("paired_sd_relative"), 3, pct=True),
                _fmt(row.get("mde_relative"), 3, pct=True),
                _fmt(row.get("mde_relative_confirmatory"), 3, pct=True),
                _fmt(row.get("power_for_sesoi"), 3),
                _fmt(row.get("power_for_sesoi_confirmatory"), 3),
                _fmt(row.get("required_seeds_for_sesoi"), 0),
                _fmt(row.get("required_seeds_for_sesoi_confirmatory"), 0),
            ]) + " |")

    ruler = _as_dict(_as_dict(plan.get("endpoints")).get("ruler"))
    if ruler:
        md += ["", f"**RULER 先验灵敏度**（SESOI={_fmt(ruler.get('sesoi_pp'), 2, pct=True)}；"
                   f"每格 n={ruler.get('samples_per_cell')}；"
                   f"格子数≈{ruler.get('cells')}；"
                   f"未校正 80% 功效下可容忍的不一致率≈"
                   f"{_fmt(ruler.get('max_discordance_for_sesoi'), 4, pct=True)}；"
                   f"确认性校正后≈"
                   f"{_fmt(ruler.get('max_discordance_for_sesoi_confirmatory'), 4, pct=True)}）", "",
               "| 不一致率 | 每格MDE 未校正 | 每格MDE 确认性 | 功效 未校正 | 功效 确认性 | 所需每格样本 未校正 | 所需每格样本 确认性 |",
               "|---|---|---|---|---|---|---|"]
        for row in _as_list(ruler.get("sensitivity")):
            if not isinstance(row, dict):
                continue
            md.append("| " + " | ".join([
                _fmt(row.get("discordance"), 2, pct=True),
                _fmt(row.get("mde_pp"), 2, pct=True),
                _fmt(row.get("mde_pp_confirmatory"), 2, pct=True),
                _fmt(row.get("power_for_sesoi"), 3),
                _fmt(row.get("power_for_sesoi_confirmatory"), 3),
                _fmt(row.get("required_samples_per_cell_for_sesoi"), 0),
                _fmt(row.get("required_samples_per_cell_for_sesoi_confirmatory"), 0),
            ]) + " |")
        md.append("")
        best_case_note = ""
        if ruler.get("mde_pp_pooled_best_case_at_reference") is not None:
            best_case_note = (
                f"即使把 {ruler.get('cells')} 个格子当作完全独立的 "
                f"{ruler.get('pooled_samples_best_case')} 个样例"
                f"（最乐观上界），参考不一致率下的 MDE 仍≈"
                f"{_fmt(ruler.get('mde_pp_pooled_best_case_at_reference'), 2, pct=True)}")
            if ruler.get("mde_pp_pooled_best_case_min_over_grid") is not None:
                best_case_note += (
                    f"、网格内最乐观仍≈"
                    f"{_fmt(ruler.get('mde_pp_pooled_best_case_min_over_grid'), 2, pct=True)}")
            best_case_note += "。"
        md.append(
            "注：RULER 的配对 McNemar 只使用不一致对；每格样本数远小于"
            " SESOI 所需时，“等价/非劣”只能作为描述性证据。"
            + best_case_note
            + "要把它升级为确认性主张，须先增加每格样本数并重新登记。")

    # ---- 设计符合性/分析人群
    md += ["", "**设计符合性与分析人群**", ""]
    if not conf:
        md.append("- ⚠ 未生成设计符合性报告。")
    elif conf.get("checked") is not True:
        reason = conf.get("reason")
        md.append(f"- ⚠ 未核对（reason=`{reason}`）："
                  + "；".join(_as_text_list(conf.get("warnings"))[:3]))
    else:
        pop = _as_dict(conf.get("population"))
        md.append(
            f"- 计划内单元格 {pop.get('expected_cells')} 个；"
            f"实际有 eval 产物 {pop.get('observed_eval_cells')} 个；"
            f"完整={pop.get('complete_eval')}")
        if conf.get("problems"):
            for p in _as_text_list(conf.get("problems"))[:5]:
                md.append(f"- ❌ {p}")
            md.append(
                "- 结论：受影响假设的确认性判定已自动降级为探索性；"
                "不得只保留跑成功的种子/变体来宣称预注册结果。"
                "若缺失机制与结果相关（例如发散只发生在某变体），"
                "可用样本估计本身也有偏。")
        else:
            md.append("- ✅ 计划内单元齐全：确认性判定基于全部计划种子。")
        for w in _as_text_list(conf.get("warnings"))[:3]:
            md.append(f"- ⚠ {w}")
        by_variant = _as_dict(pop.get("by_variant"))
        missing_rows = [v for v, row in by_variant.items()
                        if isinstance(row, dict) and row.get("missing_eval")]
        if missing_rows:
            md.append("- 逐变体缺失种子："
                      + "；".join(
                          f"{v}: {row.get('missing_eval')}"
                          for v, row in by_variant.items()
                          if isinstance(row, dict) and row.get("missing_eval")))

    # ---- 实测灵敏度（描述性）
    observed = [r for r in (observed or []) if isinstance(r, dict)]
    if observed:
        md += ["", "**本次实测方差下的设计灵敏度（未校正口径，描述性，不作显著性判定）**", "",
               "| 对比 | 指标 | n | 配对差SD | SESOI | 实测MDE | 对SESOI功效 | 是否够用 |",
               "|---|---|---|---|---|---|---|---|"]
        for r in observed[:20]:
            md.append("| " + " | ".join([
                f"{r.get('candidate')} vs {r.get('reference')}",
                str(r.get("metric")), str(r.get("n")),
                _fmt(r.get("std_diff_sesoi_units"), 4),
                _fmt(r.get("sesoi_margin"), 4),
                _fmt(r.get("observed_mde_sesoi_units"), 4),
                _fmt(r.get("power_for_sesoi"), 3),
                ("✅" if r.get("adequate_for_sesoi") is True
                 else "⚠" if r.get("adequate_for_sesoi") is False else "-"),
            ]) + " |")
        md.append("")
        md.append(
            "注：实测 MDE 由本次观测到的配对差 SD 反推，回答“这次设计实际"
            "能分辨多小的效应”；MDE>SESOI 的对比即使显著，也不能证明达到"
            "实际重要差异。它不是把 p 值改写成“观测功效”。")
    return md


# ----------------------------------------------------------------------- CLI
def _parse_seeds(spec):
    if spec is None or spec == "":
        return []
    out = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        out.append(_seed_int(int(part, 0), "seed"))
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="生成训练前设计计划（先验功效/分析人群/设计符合性）")
    p.add_argument("--config", required=True)
    p.add_argument("--seeds", default=None,
                   help="逗号分隔种子；缺省读配置 seeds（再缺省 [seed]）")
    p.add_argument("--variants", default=None,
                   help="逗号分隔变体；缺省全部可运行变体")
    p.add_argument("--out", default="design_plan.json")
    p.add_argument("--registered", default=None,
                   help="预注册表路径（默认 preregistration/hypotheses.json）")
    p.add_argument("--ruler_n_samples", type=int, default=None)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    try:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except (OSError, ValueError) as e:
        print(f"[design] 读取配置失败：{e}", file=sys.stderr)
        return 2
    if not isinstance(cfg, dict):
        print("[design] 配置顶层必须是对象", file=sys.stderr)
        return 2

    if args.variants:
        variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    else:
        variants = sorted(_known_variant_names())
    if not variants:
        print("[design] 没有可用变体（请显式 --variants）", file=sys.stderr)
        return 2

    if args.seeds:
        seeds = _parse_seeds(args.seeds)
    else:
        raw = cfg.get("seeds")
        if raw is None:
            raw = [cfg.get("seed", 1337)]
        seeds = [_seed_int(s, "seed") for s in raw]

    registered = None
    reg_path = args.registered
    if reg_path is None:
        reg_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "preregistration", "hypotheses.json")
    try:
        with open(reg_path, "r", encoding="utf-8") as f:
            registered = json.load(f)
    except (OSError, ValueError):
        registered = None

    try:
        plan = build_design_plan(
            cfg, seeds, variants, registered,
            config_file=args.config,
            ruler_n_samples=args.ruler_n_samples, smoke=args.smoke)
    except ValueError as e:
        print(f"[design] 计划生成失败：{e}", file=sys.stderr)
        return 2
    save_design_plan(plan, args.out)
    if not args.quiet:
        print(f"[design] 计划已写入 {args.out}（status={plan['status']}，"
              f"digest={plan['plan_digest'][:12]}）")
        for c in plan["checks"]:
            if c["status"] in ("warn", "fail"):
                print(f"[design] {c['status']}: {c['title']} — {c['detail']}")
    # 设计缺陷不在此处硬失败（可用 --smoke/探索性运行），但退出码区分
    # fail，便于脚本在正式实验前做闸门。
    return 2 if plan["status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
