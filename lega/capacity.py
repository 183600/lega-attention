"""训练期逐组件容量匹配证据消融的纯计算层。

现有训练期对照回答了两个不同问题：

* ``lega_no_query/no_conc/no_len``（去掉组件）与 ``lega`` 的差回答“重新训练
  时去掉该组件会怎样”，但去掉组件同时减少了门控参数与优化自由度；
* ``lega_placebo``（整体参数匹配、证据换成噪声）回答“多出的门控容量是否有
  优化红利”，但它是整体对照，不能把每个组件的效应拆成信息与容量两部分。

本模块配套的 ``lega_null_query/null_conc/null_len`` 保留 Full 的全部门控参数
与 FLOPs，只把对应证据通道置零（A: ``qterm``、B: ``c``、C: ``log n``）。于是
对每个组件 X 有同一批种子的三臂：

    Full (lega)  —  完整证据与参数
    Null_X       —  参数保留、X 证据置零
    No_X         —  参数与证据都去掉

三个效应在构造上精确相加（同一 seed、同一协议）：

    info_X     = benefit(Full, Null_X)     证据信息贡献
    capacity_X = benefit(Null_X, No_X)     参数/优化容量贡献
    total_X    = benefit(Full, No_X) = info_X + capacity_X

``benefit`` 的方向由指标决定：PPL 越低越好，RULER EM 越高越好；所有行统一
输出“正值 = Full 更好”，便于跨指标阅读。该分解是**定义性恒等式**，不是
统计估计：它把总 LOO 效应机械地分到两臂，不能据此宣称“信息效应显著大于
容量效应”而不做交互/不确定性检验。逐项 p 值（由流水线另加）只检验各效应
是否偏离 0，不检验 info−capacity 之差。

重要边界：Null_X 只把特征置零，不改变训练数据、优化器或门控参数个数；它
不是分布匹配的置换零假设，也不等价于“给 X 注入同分布噪声”。容量项因此
混合了参数容量、初始化/优化红利以及置零后网络重新分配表征的能力，只能
解释为“去掉证据但保留参数后仍存在的那部分差异”。
"""
from __future__ import annotations

import math

COMPONENTS = ("A", "B", "C")

COMPONENT_CN = {
    "A": "查询项 A (w·q)",
    "B": "集中度项 B (a·c)",
    "C": "长度项 C (b·log n)",
}

# 每个组件保留参数、只置零证据的 Full 对照。
CAPACITY_NULL_VARIANTS = {
    "A": "lega_null_query",
    "B": "lega_null_conc",
    "C": "lega_null_len",
}

# 对应的去一法变体：参数和证据都去掉。
CAPACITY_OFF_VARIANTS = {
    "A": "lega_no_query",
    "B": "lega_no_conc",
    "C": "lega_no_len",
}

# 反向索引供 attention/sanity 判断“哪个参数应预期零梯度”。
NULL_EVIDENCE_COMPONENT = {
    variant: component
    for component, variant in CAPACITY_NULL_VARIANTS.items()
}

EFFECT_ORDER = ("info", "capacity", "total")

EFFECT_CN = {
    "info": "信息效应 Full−Null",
    "capacity": "容量效应 Null−No",
    "total": "总效应 Full−No",
}


def _mean(values):
    return sum(values) / len(values) if values else None


def _median(values):
    if not values:
        return None
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return float(ordered[mid])
    return (float(ordered[mid - 1]) + float(ordered[mid])) / 2.0


def _benefit(value: float, baseline: float, lower_better: bool) -> float:
    """value 相对 baseline 的收益；正值恒表示 value 更好。"""
    return (baseline - value) if lower_better else (value - baseline)


def component_effects(full, null, off, lower_better: bool = False) -> dict:
    """把同一批种子的 Full/Null/No 三臂分解为 info/capacity/total。

    输入序列必须逐项对齐（同一 seed 顺序由调用方保证），且只含有限数值。
    ``lower_better=True`` 用于 PPL；``False`` 用于 RULER EM 等越高越好的指标。
    返回未做统计检验的效应值、均值、中位数与恒等式残差；n 不足时各字段为
    None 或空列表，由调用方显式报告“不可计算”。
    """
    full = list(full)
    null = list(null)
    off = list(off)
    if not (len(full) == len(null) == len(off)):
        raise ValueError("Full/Null/No 三臂必须逐 seed 对齐")
    if any(isinstance(v, bool) or not isinstance(v, (int, float))
           or not math.isfinite(float(v)) for arm in (full, null, off)
           for v in arm):
        raise ValueError("Full/Null/No 只接受有限数值")
    full = [float(v) for v in full]
    null = [float(v) for v in null]
    off = [float(v) for v in off]
    info = [_benefit(f, n, lower_better) for f, n in zip(full, null)]
    capacity = [_benefit(n, o, lower_better) for n, o in zip(null, off)]
    total = [_benefit(f, o, lower_better) for f, o in zip(full, off)]
    residual = [t - (i + c) for t, i, c in zip(total, info, capacity)]
    effects = {}
    for name, values in (("info", info), ("capacity", capacity),
                         ("total", total)):
        effects[name] = {
            "values": values,
            "mean": _mean(values),
            "median": _median(values),
        }
    total_mean = effects["total"]["mean"]
    info_mean = effects["info"]["mean"]
    return {
        "n": len(full),
        "lower_is_better": bool(lower_better),
        "direction": "positive_means_full_better",
        "effects": effects,
        "info_share": (info_mean / total_mean
                       if total_mean not in (None, 0.0) else None),
        "identity_residual_max_abs": max((abs(r) for r in residual),
                                         default=None),
    }
