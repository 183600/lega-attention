"""种子空间分区：确认性种子与开发（调优/试点）种子必须互不相交。

为什么这属于“实验设计”而不是工程细节：

同一组种子既用于超参搜索（``--tune_baseline``，只看 val）又用于确认性推断时，
被选中的超参是在“这批种子自己的噪声”上挑出来的（winner's curse）：它在
这些种子上的 val 表现被系统性高估。该超参随后被注入**全部**变体，并在
同一批种子上做确认性配对检验，于是基线（``standard``）在这批种子上获得了
一份**由选择产生的**额外优势。对“LEGA 优于已调优基线”这类主张，这种耦合
既可能把真实优势压向零，也可能制造出方向相反的假象；它属于典型的
“选择偏差”，不能靠事后加一个 Holm 校正来消除。

因此本项目把种子空间显式切成三段：

* ``confirmatory``：只用于确认性推断（预注册 ``scope.seeds`` 或本次计划种子）；
* ``development``：只用于基线调优/超参搜索/流程调试；
* ``pilot``：只用于冒烟与流程连通性预演。

同一颗种子不得同时落在两段里。流水线在未显式指定 ``--tune_seeds`` 时会
**自动分配**与确认性种子不相交的开发种子；显式指定且与确认性种子重叠时，
设计计划把 ``seed_partition`` 检查记为 ``fail``（见
:func:`check_seed_partition`），避免“基线调优红利”被算进方法效果。

本模块只做纯计算（不读磁盘、不训练），因此可以在开跑前单独调用核对。
"""

from __future__ import annotations

import operator
from numbers import Integral, Real

from ._shared import _install as _shared_install

_shared_install(globals())


#: 种子在确认性推断中的角色。
SEED_ROLES = ("confirmatory", "development", "pilot")

#: 自动分配开发种子时的起始基数。该区间与本项目实际使用的确认性种子
#: （1337/2024/42/7/2025/2023/19/2026 等小整数）不重叠，且远小于
#: 2^32-1，因此可以整段留作“开发/试点”专用。
DEFAULT_DEVELOPMENT_SEED_BASE = 10_000_000

#: 允许的分区冲突处理策略。
#: ``fail``：重叠即设计失败（默认，fail-closed）；
#: ``warn``：只告警（用于显式接受风险的探索性运行）。
SEED_OVERLAP_POLICIES = ("fail", "warn")


def _strict_seed(value, name: str) -> int:
    """严格种子：bool/小数/字符串拒绝，取值限定在 0..2^32-1。"""
    if isinstance(value, bool):
        raise ValueError(f"{name} 必须是整数，收到 bool {value!r}")
    try:
        out = operator.index(value)
    except TypeError:
        if isinstance(value, Real):
            f = float(value)
            if not (f.is_integer() and abs(f) < float("inf")):
                raise ValueError(f"{name} 必须是整数，收到 {value!r}") from None
            out = int(f)
        else:
            raise ValueError(f"{name} 必须是整数，收到 {value!r}") from None
    if out < 0 or out > MAX_SEED:
        raise ValueError(f"{name} 必须在 0..{MAX_SEED} 内，收到 {out}")
    return int(out)


def _seed_list(value, name: str) -> list:
    """把可迭代对象规范成去重后的种子列表；``None`` → 空列表。"""
    if value is None:
        return []
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple, set, frozenset)):
        raise ValueError(f"{name} 必须是种子列表，收到 {value!r}")
    # 集合没有稳定顺序：排序后再规范化，保证分区摘要可复算（同一份配置
    # 每次生成的计划摘要一致，否则 plan_digest 会无意义地漂移）。
    items = sorted(value) if isinstance(value, (set, frozenset)) else list(value)
    out = [_strict_seed(v, name) for v in items]
    if len(set(out)) != len(out):
        raise ValueError(f"{name} 不能重复：{out}")
    return out


def derive_development_seeds(
    confirmatory,
    n: int = 1,
    base: int = DEFAULT_DEVELOPMENT_SEED_BASE,
    reserved=None,
) -> list:
    """为调优/调试分配与确认性种子不相交的开发种子。

    分配规则是**确定性**的：从 ``base`` 起依次取整数，跳过任何与确认性
    种子或已保留种子冲突的值。这样同一份配置在多次运行、多个机器上都会
    得到同一组开发种子，计划可复算；分配结果同时写入 ``design_plan.json``
    的 ``seed_partition``，读者能看到“基线调优究竟用了哪几颗种子”。

    参数
    ----
    confirmatory:
        确认性种子（会被用于配对推断的种子）。
    n:
        需要分配的开发种子数量（>= 1）。
    base:
        分配起点；必须是非负整数且 ``base + n`` 不越过 ``MAX_SEED``。
    reserved:
        额外需要避开的种子（例如试点种子）。

    返回
    ----
    list[int]
        与 ``confirmatory``/``reserved`` 均不相交的开发种子。
    """
    if isinstance(n, bool) or not isinstance(n, Integral):
        raise ValueError(f"n 必须是整数，收到 {n!r}")
    n = int(n)
    if n < 1:
        raise ValueError(f"n 必须 >= 1，收到 {n}")
    if isinstance(base, bool) or not isinstance(base, Integral):
        raise ValueError(f"base 必须是整数，收到 {base!r}")
    base = int(base)
    if base < 0 or base > MAX_SEED:
        raise ValueError(f"base 必须在 0..{MAX_SEED} 内，收到 {base}")
    conf = set(_seed_list(confirmatory, "confirmatory"))
    reserved_set = set(_seed_list(reserved, "reserved"))
    out = []
    candidate = base
    while len(out) < n:
        if candidate > MAX_SEED:
            raise ValueError(
                f"无法在 0..{MAX_SEED} 内分配 {n} 个不冲突的开发种子"
                f"（起点 base={base}）"
            )
        if candidate not in conf and candidate not in reserved_set:
            out.append(candidate)
        candidate += 1
    return out


def partition_seeds(confirmatory, development=None, pilot=None) -> dict:
    """把种子空间切成 confirmatory/development/pilot 三段并检出冲突。

    返回 dict::

        {
          "roles": {"confirmatory": [...], "development": [...], "pilot": [...]},
          "overlaps": [{"roles": ["confirmatory", "development"], "seeds": [...]}],
          "has_overlap": bool,
          "unassigned": [...]   # 未声明角色的种子（调用方未提供时为空）
        }

    同一段内重复种子直接报错；跨段重叠会被记录但不抛异常——是否判为设计
    失败由 :func:`check_seed_partition` 按策略决定。
    """
    roles = {
        "confirmatory": _seed_list(confirmatory, "confirmatory"),
        "development": _seed_list(development, "development"),
        "pilot": _seed_list(pilot, "pilot"),
    }
    overlaps = []
    names = list(SEED_ROLES)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            shared = sorted(set(roles[a]) & set(roles[b]))
            if shared:
                overlaps.append({"roles": [a, b], "seeds": shared})
    assigned = set()
    for values in roles.values():
        assigned |= set(values)
    return {
        "roles": roles,
        "overlaps": overlaps,
        "has_overlap": bool(overlaps),
        "unassigned": [],
        "n_assigned": len(assigned),
    }


def check_seed_partition(partition: dict, policy: str = "fail") -> dict:
    """把种子分区冲突变成一条可机读的设计检查。

    ``policy='fail'``：任一角色之间存在共用种子即 ``fail``（默认；
    ``fail`` 会进入 ``design_conformance`` 的全局设计失败集合，使全部
    确认性判定降级为探索性）。``policy='warn'``：只告警，用于显式接受
    风险的探索性运行——但降级原因仍会写进计划摘要，不能被忽略。

    没有开发/试点种子时（最常见的正式配置）返回 ``pass``，不会因为“本来
    就没有调优阶段”而制造噪声。
    """
    if policy not in SEED_OVERLAP_POLICIES:
        raise ValueError(
            f"policy 必须是 {SEED_OVERLAP_POLICIES} 之一，收到 {policy!r}"
        )
    part = partition if isinstance(partition, dict) else {}
    roles = part.get("roles") if isinstance(part.get("roles"), dict) else {}
    overlaps = part.get("overlaps") if isinstance(part.get("overlaps"), list) else []
    conf = list(roles.get("confirmatory") or [])
    dev = list(roles.get("development") or [])
    pilot = list(roles.get("pilot") or [])
    if not overlaps:
        detail = (
            f"确认性种子 {len(conf)} 颗、开发种子 {len(dev)} 颗、"
            f"试点种子 {len(pilot)} 颗，三段互不相交"
            + ("；本次未声明开发/试点种子" if not dev and not pilot else "")
        )
        return {
            "id": "seed_partition",
            "title": "种子空间分区（确认性 vs 开发/试点）",
            "status": "pass",
            "detail": detail,
        }
    pieces = []
    for item in overlaps:
        if not isinstance(item, dict):
            continue
        pair = item.get("roles")
        seeds = item.get("seeds")
        if not isinstance(pair, list) or not isinstance(seeds, list):
            continue
        pieces.append(f"{'∩'.join(str(x) for x in pair)}={list(seeds)}")
    joined = "；".join(pieces) if pieces else "存在未定位的重叠"
    if policy == "fail":
        detail = (
            f"确认性种子与开发/试点种子重叠（{joined}）。"
            "用同一批种子做超参选择与确认性推断会把选择偏差（winner's curse）"
            "带进基线，使“优于已调优基线”的主张不可判定；"
            "请改用与确认性种子不相交的开发种子（可由 "
            "lega.design.derive_development_seeds 自动分配），"
            "或显式声明 design.seed_overlap_policy='warn' 接受该风险"
        )
        status = "fail"
    else:
        detail = (
            f"确认性种子与开发/试点种子重叠（{joined}），"
            "按 design.seed_overlap_policy='warn' 只告警："
            "基线超参是在同一批种子上挑出来的，确认性比较含选择偏差，"
            "结论应按探索性解读"
        )
        status = "warn"
    return {
        "id": "seed_partition",
        "title": "种子空间分区（确认性 vs 开发/试点）",
        "status": status,
        "detail": detail,
    }
