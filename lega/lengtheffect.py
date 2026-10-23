"""RULER 上下文长度分层：把“长度校准”从平均分里还原出来。

项目的主方法声称随上下文变长调整门控可信度，但结果分析若只报跨上下文
长度的 RULER 平均，会把两种完全不同的情形混成一个数：

* 短上下文涨、长上下文跌（方法恰好在声称有效的地方失效），与
* 所有长度都小幅涨（平均分同样为正）。

两者对科学结论的意义相反。本模块只做**纯数据整形**（不依赖 torch / 统计
检验），把 RULER payload 变成按上下文长度分层的 macro EM，并提供两个
剂量-反应标量：

``log2_slope``
    以 ``log2(ctx)`` 为自变量的最小二乘斜率，单位是“上下文长度每翻一倍，
    EM 变化多少”。等距的 log 尺度让 1k→2k 与 8k→16k 的斜率可直接比较，
    不会让最长的那一档仅因绝对 token 数大而支配趋势。

``span_contrast``
    ``EM(最长 ctx) − EM(最短 ctx)``。它不假定线性，是趋势检验的稳健对照：
    斜率显著但首末落差≈0 时，差异可能只出现在中间档，不能写成“越长越好”。

估计量口径与 ``run_all._ruler_avg_of`` 保持一致：先在每个上下文内对任务
等权（任务内不再对样本数加权），再对上下文等权。因此“各档上下文等权
平均”严格等于该 run 的 RULER macro 平均——前提是每个任务在所有档位都有
观测（平衡设计）。不平衡时两者相差一个任务×档位权重，调用方必须显式披露，
不能把分层均值当成总体均值的无偏分解。

数据卫生：EM/recall 等比例必须落在 [0,1]；``n`` 显式为负/零/非整数时该格
不是有效观测；上下文键必须是非负整数对应的规范十进制字符串（``"0512"``
与 ``"512"`` 若同时出现会合并两个档位，按损坏输入整体拒绝）。

剂量轴：``context_macro_means(..., align_requested=True)`` 以 payload 声明的
``ctx_lens`` 为请求档位。不同任务的实际 ``room_ctx`` 会因生成预算不同而
不同（如 niah 的 1011 与 fwe 的 979）；只有每个任务恰好覆盖全部请求档位
时才按升序一一对齐，缺档/多档返回空，避免把任务组成变化当成新的长度档。
"""
import math
from decimal import Decimal, InvalidOperation
from numbers import Integral, Real

__all__ = [
    "context_task_means",
    "context_macro_means",
    "log2_slope",
    "span_contrast",
]


def _finite(value):
    """有限实数；bool / 非数值 / NaN / Inf / 不可表示的大整数 → None。

    与项目既有的 ``_finite_number`` 同口径：规范数值字符串（如 ``"0.5"``）
    先转 float 再参与计算。RULER 产物可能来自旧版本/手工整理，数值字符串
    本身没有歧义；若在这里直接拒绝，同一 payload 在 ``context_task_means``
    与 ``_ruler_avg_of`` 中会得到不同的单元格集合，长度分层就与总体 RULER
    平均不可比。带下划线的数字串（``"1_0"``）不是规范 JSON 数值，仍拒绝。
    """
    if value is None or isinstance(value, bool):
        return None
    if type(value).__name__ == "bool_":  # numpy.bool_
        return None
    if isinstance(value, str) and "_" in value:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return f if math.isfinite(f) else None


def _proportion(value):
    """[0,1] 比例；越界值不是合法 EM/recall，返回 None。"""
    f = _finite(value)
    if f is None or not 0.0 <= f <= 1.0:
        return None
    return f


def _positive_count(value):
    """正整数字段（样本数 n）；缺省/非整数/非正 → None。

    与 run_all 的 RULER 汇总同口径：显式 ``n`` 是“该格有多少有效观测”的
    声明，0 或负数不能用一个空洞的 EM=0 拉低平均。整数必须按**精确值**校验：
    ``float(10**400)`` 会 OverflowError，但 10**400 仍是合法的正样本数，
    不能因为无法转成浮点就被当成非法 n 丢掉（否则该格从长度分层中静默
    消失，且与 ``_ruler_avg_of`` 的单元格集合不一致）。整数字符串/整数值
    浮点同样规范化为 int，与 run_all 的 ``_integral_value`` 保持同口径。
    """
    if value is None or isinstance(value, bool):
        return None
    if type(value).__name__ == "bool_":
        return None
    if isinstance(value, str):
        s = value.strip()
        # 下划线数字串不是规范 JSON 数值（与 _integral_value 同口径拒绝）。
        if not s or "_" in s:
            return None
        try:
            d = Decimal(s)
        except (InvalidOperation, ValueError):
            return None
        if not d.is_finite() or d != d.to_integral_value():
            return None
        n = int(d)
    elif isinstance(value, Integral):
        n = int(value)
    elif isinstance(value, Real):
        try:
            f = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if not math.isfinite(f) or not f.is_integer():
            return None
        n = int(f)
    else:
        return None
    return n if n > 0 else None


def _canonical_ctx(key):
    """RULER results 的上下文键 → 正整数；不规范的键返回 None。

    只接受 ``str(int)`` 的规范形式：``"0512"``、``"512.0"``、``" 512"``
    都可能是两个不同键指向同一档位，若静默 ``int()`` 合并，会把同一格
    的重复观测折叠、夸大精度。bool 也不是合法上下文长度。
    """
    if isinstance(key, bool):
        return None
    if isinstance(key, Integral):
        value = int(key)
        return value if value > 0 else None
    if not isinstance(key, str) or not key or key != key.strip():
        return None
    if not key.isdigit():
        return None
    try:
        value = int(key)
    except (TypeError, ValueError):
        return None
    if value <= 0 or str(value) != key:
        return None
    return value


def context_task_means(payload, metric="em"):
    """RULER payload → ``{ctx: {task: mean}}``（只用 (任务, 上下文) 单元格）。

    每格至多一个观测；同一 ``ctx`` 出现两个不同原始键（例如 ``"512"`` 与
    ``"0512"``）时返回 ``{}``，因为无法判断它们是否本应是同一档位，合并
    会伪造精度。缺 ``n`` 的旧产物保持兼容（保留该格）；显式非法 ``n`` 或
    越界比例按缺失处理。

    空 payload / 全损坏时返回 ``{}``（不伪造观测）。调用方据此区分“没有
    长度分档结果”和“每个档位都是 0 分”。
    """
    if not isinstance(payload, dict):
        return {}
    results = payload.get("results")
    if not isinstance(results, dict):
        return {}
    metric = metric if isinstance(metric, str) and metric else "em"
    out = {}
    for task, by_ctx in results.items():
        if (not isinstance(task, str) or task.startswith("_")
                or not isinstance(by_ctx, dict)):
            continue
        # 重复检测必须按任务隔离：不同任务在同一档位有同名的 ctx 键是
        # 正常设计，只有同一任务内部出现 "512"/"0512" 这类并列键才说明
        # 两个原始键会合并到同一档位。
        seen_ctx = set()
        for raw_ctx, cell in by_ctx.items():
            ctx = _canonical_ctx(raw_ctx)
            if ctx is None:
                # 非规范/非法上下文键不能进入趋势分析：它无法与其它档位
                # 可靠排序，也不能对应到 payload 的 ctx_lens 声明。
                return {}
            if ctx in seen_ctx:
                return {}
            seen_ctx.add(ctx)
            if not isinstance(cell, dict):
                continue
            if cell.get("n") is not None:
                if _positive_count(cell.get("n")) is None:
                    continue
            value = _proportion(cell.get(metric))
            if value is None:
                continue
            out.setdefault(ctx, {})[task] = value
    return {ctx: dict(v) for ctx, v in sorted(out.items()) if v}


def _requested_contexts(payload):
    """payload 的 ``ctx_lens`` → 升序唯一请求档位；缺失/非法返回 None。

    请求档位是实验设计变量；实际 ``room_ctx`` 会因每类任务的生成预算
    （``max_new`` vs ``fwe_max_new``）而被 clamp 到不同值（例如 niah 的
    1011 与 fwe 的 979）。把实际值当成不同“长度档”会把任务组成变化误读成
    长度效应，因此对齐必须以请求档位为准。
    """
    if not isinstance(payload, dict):
        return None
    raw = payload.get("ctx_lens")
    if not isinstance(raw, (list, tuple)) or not raw:
        return None
    out, seen = [], set()
    for item in raw:
        ctx = _canonical_ctx(item)
        if ctx is None or ctx in seen:
            return None
        seen.add(ctx)
        out.append(ctx)
    return sorted(out)


def _requested_macro(task_means, requested):
    """把每个任务的实际 room_ctx 按升序对齐到请求档位，再任务等权。

    只有每个任务都恰好有 ``len(requested)`` 个实际档位时才做顺序对齐：
    若少一档（clamp 后两档合并、样例构造失败或旧产物缺格），无法判断
    实际键对应哪个请求档位，返回 ``{}`` 而不是猜测——猜错会把“任务在某
    档缺失”变成一条假趋势。
    """
    per_task = {}
    for ctx, by_task in task_means.items():
        for task, value in by_task.items():
            per_task.setdefault(task, {})[ctx] = value
    if not per_task:
        return {}
    if any(len(v) != len(requested) for v in per_task.values()):
        return {}
    cells = {ctx: [] for ctx in requested}
    for values in per_task.values():
        actual = sorted(values)
        # 实际 room_ctx 只可能由请求档位被生成预算向下 clamp 得到：
        #   actual_i = min(requested_i, cap)，cap 对该任务固定。
        # 因此 actual_i 必须落在上一请求档位与当前请求档位之间；若 actual
        # 反而大于请求值（例如请求 1024 却出现 2048），或某一档跌到上一个
        # 请求档位以下（顺序整体错位），就无法把实际键对应到请求档位。
        # 此时按升序硬配会凭空造出一条不存在的长度趋势，必须拒绝。
        for i, (a, req) in enumerate(zip(actual, requested)):
            if a > req:
                return {}
            if i and a <= requested[i - 1]:
                return {}
        for actual_ctx, req in zip(actual, requested):
            cells[req].append(values[actual_ctx])
    return {req: float(sum(v) / len(v))
            for req, v in cells.items() if v}


def context_macro_means(payload, metric="em", align_requested=False):
    """RULER payload → ``{ctx: macro mean}``（每档上下文内任务等权）。

    默认按实际 ``room_ctx`` 分档。``align_requested=True`` 时改用 payload
    声明的 ``ctx_lens`` 作为剂量轴：先要求每个任务恰好覆盖全部请求档位，
    再按升序一一对齐，避免不同任务的 clamp 差异（如 979/1011）被当成两个
    额外长度档。任一任务缺档/多档时返回 ``{}``（不猜测映射）；旧产物没有
    ``ctx_lens`` 时只在所有任务共享同一实际档位集合时回退。

    与 ``run_all._ruler_avg_of`` 同一估计量顺序：任务内各档等权（这里每个
    任务在该档只有一个格子），任务间等权。任务在某档缺失时只对已有任务
    平均，因此可能偏离“总体 macro 的简单分解”；调用方需核对任务×档位
    是否完整后再解释平衡性。
    """
    task_means = context_task_means(payload, metric=metric)
    if not task_means:
        return {}
    if align_requested:
        raw_lens = payload.get("ctx_lens") if isinstance(payload, dict) else None
        if raw_lens is not None:
            requested = _requested_contexts(payload)
            if requested is None:
                return {}
            return _requested_macro(task_means, requested)
        # 旧产物没有 ctx_lens：仅在所有任务共享同一实际档位集合时回退，
        # 否则不同任务的实际档位无法比较。
        sets = {frozenset(v) for v in task_means.values()}
        if len(sets) != 1:
            return {}
    out = {}
    for ctx, by_task in task_means.items():
        if by_task:
            out[ctx] = float(sum(by_task.values()) / len(by_task))
    return dict(sorted(out.items()))


def _validated_pairs(contexts, values):
    """校验 (ctx, value) 序列并按 ctx 升序返回；不合法返回 None。

    要求：等长、至少 2 个点、ctx 为互不重复的正整数、value 为有限实数。
    只在这里做一次严格校验，两个趋势函数共用同一口径。
    """
    if (not isinstance(contexts, (list, tuple))
            or not isinstance(values, (list, tuple))
            or len(contexts) != len(values) or len(contexts) < 2):
        return None
    pairs = []
    seen = set()
    for ctx, value in zip(contexts, values):
        c = _canonical_ctx(ctx)
        if c is None or c in seen:
            return None
        seen.add(c)
        v = _finite(value)
        if v is None:
            return None
        pairs.append((c, v))
    pairs.sort(key=lambda p: p[0])
    return pairs


def log2_slope(contexts, values):
    """``value`` 对 ``log2(ctx)`` 的 OLS 斜率（每翻倍变化量）；无法定义→None。

    至少需要 2 个不同上下文长度；所有 ctx 相同（方差为 0）时斜率无定义。
    数值用中心化公式，避免大 ctx 的 log 造成精度损失（ctx 本身是整数，
    ``log2`` 的差值仍是良好条件数）。
    """
    pairs = _validated_pairs(contexts, values)
    if pairs is None:
        return None
    xs = [math.log2(c) for c, _ in pairs]
    ys = [v for _, v in pairs]
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    if not math.isfinite(sxx) or sxx <= 0.0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    return float(slope) if math.isfinite(slope) else None


def span_contrast(contexts, values):
    """``value(最长 ctx) − value(最短 ctx)``；无法定义→None。

    不假定线性，作为斜率/单调趋势的稳健对照：首末落差与斜率方向相反时，
    说明效应集中在中间档位，“越长越好/越差”的表述不成立。
    """
    pairs = _validated_pairs(contexts, values)
    if pairs is None:
        return None
    short_v = pairs[0][1]
    long_v = pairs[-1][1]
    out = long_v - short_v
    return float(out) if math.isfinite(out) else None
