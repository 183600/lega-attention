"""先验功效与灵敏度计算。"""
from __future__ import annotations
from ._shared import _install as _shared_install

_shared_install(globals())

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


def _t_critical(df: int, alpha: float) -> float:
    """双侧 Student-t 临界值（二分；仅用于小样本功效仿真）。"""
    from ..utils import student_t_two_sided_p
    lo, hi = 0.0, 1.0
    while student_t_two_sided_p(hi, df) > alpha:
        hi *= 2.0
        if hi > 1e12:
            raise ValueError("无法在数值范围内求 t 临界值")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if student_t_two_sided_p(mid, df) > alpha:
            lo = mid
        else:
            hi = mid
        if hi - lo <= 1e-12 * max(1.0, hi):
            break
    return 0.5 * (lo + hi)


def paired_t_power_nct(effect, sd, n, alpha: float = DEFAULT_ALPHA,
                       n_sim: int = 200000, seed: int = 0) -> dict:
    """配对 t 检验的非中心 t 功效（固定种子仿真，n 小时优先使用）。

    正态近似在 n=3~5 时会高估功效。这里直接仿真
    ``T = (Z + δ) / sqrt(V/(n-1))``，其中 Z~N(0,1)、V~χ²(n-1)，
    δ=|effect|/(sd/√n)，再与双侧 t 临界值比较。返回 ``power`` 与
    ``method="noncentral_t_simulation"``，便于与历史正态口径并列披露。
    """
    effect = _finite_float(effect, "effect", nonnegative=True)
    sd = _finite_float(sd, "sd", positive=True)
    n = _strict_int(n, "n", minimum=2)
    alpha = _open_unit(alpha, "alpha")
    n_sim = _strict_int(n_sim, "n_sim", minimum=1000)
    seed = _strict_int(seed, "seed", minimum=0)
    if effect == 0.0:
        return {"power": float(alpha), "method": "noncentral_t_simulation",
                "n_sim": int(n_sim), "seed": int(seed),
                "ncp": 0.0, "t_critical": _t_critical(n - 1, alpha)}
    import numpy as np
    rng = np.random.default_rng(seed)
    ncp = effect / (sd / math.sqrt(n))
    df = n - 1
    tcrit = _t_critical(df, alpha)
    z = rng.standard_normal(int(n_sim)) + ncp
    v = rng.chisquare(df, size=int(n_sim))
    t = z / np.sqrt(v / df)
    pw = float(np.mean(np.abs(t) >= tcrit))
    return {"power": pw, "method": "noncentral_t_simulation",
            "n_sim": int(n_sim), "seed": int(seed), "ncp": float(ncp),
            "t_critical": float(tcrit)}


def paired_t_power(effect, sd, n, alpha: float = DEFAULT_ALPHA,
                   power: float = None) -> float:
    """配对 t 检验（双侧）的正态近似功效。

    历史接口保留正态近似；n=3~5 的正式规划请同时调用
    :func:`paired_t_power_nct`（非中心 t 仿真），不要把该值当作精确功效。

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
    if z <= 0.0:
        return 0.0
    root_n = _sqrt_count(n)
    if root_n is None:
        return None
    out = sd * (z / root_n)
    return float(out) if math.isfinite(out) else None


def required_pairs(effect, sd, alpha: float = DEFAULT_ALPHA,
                   power: float = DEFAULT_POWER, max_n: int = 1_000_000):
    """达到目标功效所需的最少配对数（种子数）；无法在 max_n 内达到返回 None。"""
    effect = _finite_float(effect, "effect", positive=True)
    sd = _finite_float(sd, "sd", positive=True)
    alpha = _open_unit(alpha, "alpha")
    power = _open_unit(power, "power")
    max_n = _strict_int(max_n, "max_n", minimum=2)
    z = _z_sum(alpha, power)
    if z <= 0.0:
        return 2
    ratio = sd / effect
    if not math.isfinite(ratio):
        return None
    try:
        n0 = math.ceil((ratio * z) ** 2)
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
    z = _z_sum(alpha, power)
    if z <= 0.0:
        return None
    root_n = _sqrt_count(n)
    if root_n is None:
        return None
    out = effect * (root_n / z)
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
    z = _z_sum(alpha, power)
    if z <= 0.0:
        return 0.0
    root_n = _sqrt_count(n)
    if root_n is None:
        return None
    out = z * math.sqrt(pi) / root_n
    return float(out) if math.isfinite(out) else None


def max_discordance_for_effect(effect, n, alpha: float = DEFAULT_ALPHA,
                               power: float = DEFAULT_POWER):
    """在给定 n / 目标功效下仍能检出 ``effect`` 的最大不一致率。"""
    effect = _finite_float(effect, "effect", positive=True)
    n = _strict_int(n, "n", minimum=2)
    alpha = _open_unit(alpha, "alpha")
    power = _open_unit(power, "power")
    z = _z_sum(alpha, power)
    if z <= 0.0:
        return 1.0
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


