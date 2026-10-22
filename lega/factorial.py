"""全因子 2^3 消融的 Shapley 效应分解（无第三方依赖）。

背景：`run_all` 已实现门控内三因子 A（查询项 w·q）、B（集中度证据
a·c，c=log(n·max_j p_j)）、C（长度项 b·log n）的完整 2^3 矩阵，并以
「去一法 Full−X vs Full」和「全因子增量 X-only vs gate_only」两条
单因素边估计主效应，另做 A×B/A×C/B×C 交互检验。

单因素边回答“在其他因子固定时，删掉/加入 X 会怎样”，但它依赖一个
特定的参照格，且当效应不可加（协同/次可加）时，不同参照格给出的
“X 的贡献”并不相同。本模块用博弈论的 Shapley 值把 Full−none 的总效应
按**所有加入顺序**平均分摊到 A/B/C：

    φ_i = Σ_{S ⊆ N\\{i}} |S|! (n−|S|−1)! / n! · [v(S∪{i}) − v(S)]

其中 N={A,B,C}，v(S) 是只保留因子集合 S 时的性能（越大越好）。性质：

* **效率性**：φ_A+φ_B+φ_C = v(ABC) − v(none)，总效应不会被漏掉或重复计；
* **对称性**：对总效应贡献相同的因子得到相同 φ；
* **零玩家**：在所有联盟中都不改变 v 的因子 φ=0；
* **对交互稳健**：φ 天然把交互效应平均到参与它的因子，而不是把
  “可加性不成立”当成某个参照格下的主效应。

口径与边界：
* 因子定义与 `lega.attention.GATE_FACTORS` 完全一致（B 是 a·c，
  其中 c 始终含 log n 的长度归一化；C 才是额外的 b·log n）；
* PPL 是比值尺度，直接对 PPL 做差会让高 PPL 格主导。这里对有下界的
  PPL 使用 v = −log(PPL)（等价于在 log 似然上做可加分解），因此
  exp(−φ_i) 可解释为“该因子带来的 PPL 几何均值比”（<1 表示降低 PPL）；
* RULER 使用原始 EM（越大越好），φ 的单位是 EM 点数；
* Shapley 分解是**探索性**证据：它没有预注册，不能替代预注册的
  单因素确认性检验；两者方向冲突时并列展示，不挑选有利口径。
"""
import math
from itertools import combinations
from math import factorial
from typing import Dict, Iterable, Mapping

FACTOR_ORDER = ("A", "B", "C")
FACTORS = frozenset(FACTOR_ORDER)

# 与 run_all._FACTORIAL_CONTRAST_CELLS 相同的 8 个门控内单元格。
CELL_ORDER = ("none", "A", "B", "C", "AB", "AC", "BC", "ABC")

_CELL_FACTORS = {
    "none": frozenset(),
    "A": frozenset({"A"}),
    "B": frozenset({"B"}),
    "C": frozenset({"C"}),
    "AB": frozenset({"A", "B"}),
    "AC": frozenset({"A", "C"}),
    "BC": frozenset({"B", "C"}),
    "ABC": frozenset({"A", "B", "C"}),
}


def _finite_float(value, name: str) -> float:
    """严格有限浮点：拒绝 bool/字符串/NaN/Inf。"""
    if isinstance(value, bool):
        raise ValueError(f"{name} 必须是有限数值，收到 bool {value!r}")
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError) as e:
        raise ValueError(f"{name} 必须是有限数值，收到 {value!r}") from e
    if not math.isfinite(out):
        raise ValueError(f"{name} 必须是有限数值，收到 {value!r}")
    return out


def cell_key(factors: Iterable[str]) -> str:
    """因子集合 -> 规范单元格键（A/B/C 顺序固定，便于跨模块对齐）。"""
    fs = frozenset(factors)
    unknown = fs - FACTORS
    if unknown:
        raise ValueError(f"未知因子: {sorted(unknown)}")
    for key, expected in _CELL_FACTORS.items():
        if expected == fs:
            return key
    raise ValueError(f"无法映射因子集合: {sorted(fs)}")


def oriented_value(metric: str, raw) -> float:
    """把原始指标转成“越大越好”的 Shapley 价值函数。

    PPL 必须 >0；返回 −log(PPL)。RULER/EM 等 already-higher-better 的
    指标原样返回。其他指标名显式拒绝，避免把方向搞反后仍产出看似合理的
    “负贡献”。
    """
    value = _finite_float(raw, f"{metric} 原始值")
    if metric == "PPL":
        if value <= 0:
            raise ValueError(f"PPL 必须 > 0 才能做 log 分解，收到 {raw!r}")
        return -math.log(value)
    if metric in ("RULER平均", "RULER", "EM", "AUROC", "PR-AUC"):
        return value
    raise ValueError(
        f"未知指标 {metric!r}：只支持 PPL（转 −log）或 "
        "RULER平均/EM/AUROC/PR-AUC（越大越好）")


def shapley_values(values: Mapping[str, float]) -> Dict[str, float]:
    """由 8 个单元格的**已定向**价值计算精确 Shapley 值。

    values 的键必须是 CELL_ORDER 中的规范字符串，值必须有限。缺格不是
    “按 0 补齐”（那会把未运行伪装成“该组合性能为 0”），而是显式报错。
    """
    if not isinstance(values, Mapping):
        raise ValueError("values 必须是 {单元格: 数值} 映射")
    missing = [c for c in CELL_ORDER if c not in values]
    if missing:
        raise ValueError(f"Shapley 分解缺少单元格: {missing}")
    unknown = [k for k in values if k not in _CELL_FACTORS]
    if unknown:
        raise ValueError(f"Shapley 分解包含未知单元格: {sorted(unknown)}")
    v = {c: _finite_float(values[c], f"单元格 {c}") for c in CELL_ORDER}
    n = len(FACTOR_ORDER)
    phi = {}
    for i in FACTOR_ORDER:
        others = [j for j in FACTOR_ORDER if j != i]
        total = 0.0
        for r in range(len(others) + 1):
            weight = (factorial(r) * factorial(n - r - 1) / factorial(n))
            for subset in combinations(others, r):
                s = frozenset(subset)
                with_i = s | {i}
                total += weight * (v[cell_key(with_i)] - v[cell_key(s)])
        phi[i] = float(total)
    return phi


def decomposition(values: Mapping[str, float]) -> Dict:
    """Shapley 值 + 总效应 + 效率性残差（用于自检/单元测试）。

    total_effect = v(ABC) − v(none)（已定向，越大越好的单位）。
    efficiency_gap = Σφ − total_effect，浮点误差量级（应接近 0）。
    """
    phi = shapley_values(values)
    total = (_finite_float(values["ABC"], "ABC")
             - _finite_float(values["none"], "none"))
    return {"phi": phi, "total_effect": float(total),
            "efficiency_gap": float(sum(phi.values()) - total)}


def shares(phi: Mapping[str, float], total_effect: float) -> Dict[str, float]:
    """各因子占总效应的比例；总效应 ≤ 0 或无意义时返回空字典。

    总效应接近 0（完整方法相对无组件无改善）或为负（更差）时，“贡献
    占比”没有可解释的分母，不能强行归一化。
    """
    total = _finite_float(total_effect, "total_effect")
    if total <= 1e-12:
        return {}
    return {k: float(_finite_float(v, f"phi {k}")) / total
            for k, v in phi.items()}


def ppl_ratio_from_phi(phi: float) -> float:
    """PPL 口径的 φ（−log PPL 单位）-> 该因子带来的 PPL 几何均值比。

    φ>0（降低 PPL）时返回值 <1。指数做上界保护，避免极端病灶把 JSON
    写成 inf；调用方应同时报告截断标记。
    """
    value = _finite_float(phi, "phi")
    return math.exp(max(-20.0, min(20.0, -value)))
