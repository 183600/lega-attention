#!/usr/bin/env python3
"""一键实验流水线：训练全部变体 → PPL → RULER → 机制分析 → 图表 → 汇总表。

变体（相同参数量级与训练步数）:
  standard | gate_only | qgate | null | lega | lega_no_len | lega_no_conc
  | lega_no_query | lega_conc_only | lega_len_only
  可选参数匹配安慰剂：lega_placebo（--param_control；不进入默认矩阵）

对比与消融协议（科研统计口径）:
  * 受控变量：所有变体 × 种子使用同一 config 训练（同一优化器/学习率策略/
    训练步数/数据顺序），训练后自动核对各 run 的 config_used.json 与
    eval.json 溯源（数据指纹/代码 commit/实际训练步数），任何与“除变体外
    全部一致”相悖的差异都会直接报错，避免把调参收益或数据差异误归因给方法；
  * 容量对照：--param_control 加入 lega_placebo——门控参数个数/形状/门控
    FLOPs 与 lega 完全相同，但 B/C 证据替换为确定性伪随机特征；lega vs
    lega_placebo 的 PPL 配对差用于排除“多几个门控参数的容量/优化红利”，
    未运行时汇总页显式提示“容量解释尚未排除”；
  * 多种子：关键指标跨随机种子报告 Mean ± Std（种子一致 → 变体间配对比较），
    配对 t 检验的同指标家族做 Holm–Bonferroni 校正以控制 FWER，并报告
    Cohen's d_z 效应量；默认种子数见 configs（tiny 3 / small 5），可用
    --seeds 覆盖；
  * 小样本稳健性：种子只有 3~5 个时 t 检验的正态假设不可靠，每组配对同时
    报告符号翻转置换检验（分布无关）p_perm（同家族独立 Holm 校正）、配对差
    bootstrap 95% CI 与最小可检测效应 MDE——t 与置换结论冲突显式标 ⚠，
    “不显著”先对照 MDE 区分“无效应”与“功效不足”；
  * 评测抽样不确定性（与训练种子方差分开报告）：RULER 逐样例落盘并在
    共享 sample_id 上做配对 McNemar（micro）并并列 macro 估计量；PPL 落盘
    逐窗口损失，对同序号窗口做“种子×窗口”分层 bootstrap。两者都是固定
    题面/窗口上的探索性证据，低于最低窗口/种子门槛时只报区间不下显著性
    结论，不能替代种子级确认性检验；
  * 机制主张检验：auroc_logit − auroc_qterm 的同种子配对差作为独立指标
    家族正式检验（与逐 run 样例聚类 bootstrap 区间互为印证）；
  * 长度剂量-反应：RULER 不只看跨 ctx 平均——逐 ctx 配对 ΔEM、逐种子
    ΔEM~log2(ctx) 斜率、最长−最短落差分别作为独立探索性 Holm 家族；
    总平均与最长档/趋势方向相反、或总平均在 SESOI 内而最长档超出时显式
    告警，避免“短上下文涨、长上下文跌”被平均成一个看似中性的数；
  * 预注册：preregistration/hypotheses.json 在正式实验前登记方向性主张，
    汇总时逐条判定 支持/拒绝/未定；未登记的对比一律为事后（exploratory）
    分析，只能作为线索（登记表刻意不放 docs/：该目录会被流水线自动清理）；
  * 消融完整性：全因子（无/A/B/C/AB/AC/BC/ABC）与去一法（Full/Full-A/
    Full-B/Full-C）两套组合全部实现；汇总表显式列出消融矩阵，并基于
    Holm 校正后的显著证据逐一判定 A/B/C 组件是否“有用”——组合齐全但证据
    不足时不会下“各组件均有用”的结论；全因子另做 A×B/A×C/B×C 交互效应
    检验（探索性，独立 Holm 家族），区分“主效应可加”与“协同/次可加”。
  * 单因素审计：组件归因只使用每次仅改一个因子的对比（去一法 Full−X vs
    Full；全因子增量 X-only vs gate_only）。B-only/C-only 与 Full 相差
    两个因子，不再进入单因素对比表；审计结果写入 significance.json 与
    汇总页，防止多因子差异被误写成单组件证据。
  * 推理期 2^3：除去一法病灶外，同一 checkpoint 还施加 query_only/
    conc_only/len_only 补齐 8 格，计算推理期 Shapley 并与训练期 Shapley
    逐因子比对符号——区分“重新训练后的组件贡献”与“最终前向实际使用”。

用法:
  # 冒烟测试（tiny 配置 + 伪数据，仅验证全流程可跑通；默认只跑 1 个种子）
  python run_all.py --config configs/tiny.json --data_dir data/dummy --smoke

  # 正式实验（small 配置 + WikiText-103，默认跑配置中的全部种子）
  python scripts/prepare_data.py --out_dir data/wikitext103
  python run_all.py --config configs/small.json --data_dir data/wikitext103

  # 自定义种子列表（例如 3 个种子）
  python run_all.py --config configs/small.json --data_dir data/wikitext103 \
      --seeds 1337,2024,42
"""
import argparse
import csv
import glob
import json
import math
import os
import re
import subprocess
import sys
import time
from contextlib import contextmanager
from decimal import Decimal, InvalidOperation

import numpy as np
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from lega.attention import (CONTROL_VARIANTS, GATE_FACTORS,  # noqa: E402
                            RUNNABLE_VARIANTS, VARIANTS, VARIANT_CN,
                            _USE_CONC)
from lega.ablate import (INFERENCE_FACTORIAL_CELLS,  # noqa: E402
                         LESION_BY_ID as ABLATION_LESION_BY_ID,
                         LESION_ORDER as ABLATION_LESION_ORDER)
from lega.factorial import (CELL_ORDER as FACTORIAL_CELL_ORDER,  # noqa: E402
                            decomposition as factorial_decomposition,
                            oriented_value as factorial_oriented_value)
from lega.lengtheffect import (context_macro_means,  # noqa: E402
                               log2_slope, span_contrast)
from lega.badcase import (CATEGORY_CN as BADCASE_CN,  # noqa: E402
                          WINDOW_CATEGORY_CN, analyze_badcases,
                          analyze_worst_windows, cross_variant_analysis,
                          process_eval_json, process_ruler_json)
from lega.curves import (LOSS_SPIKE_FACTOR, STATUS_ORDER,  # noqa: E402
                         diagnose_run_dir, worst_status)
from lega.datacheck import (read_test_access_log,  # noqa: E402
                            summarize_test_access)
from lega.equivalence import (assess_paired_equivalence,  # noqa: E402
                              interpret_equivalence, metric_margin)
from lega.design import (apply_design_gate,  # noqa: E402
                         build_design_plan, design_conformance,
                         load_design_plan, load_design_plan_history,
                         observed_design_sensitivity, render_design_section,
                         save_design_plan)
from lega.experiment import (append_record, build_experiment_record,  # noqa: E402
                             collect_artifacts, ledger_path, new_run_id,
                             snapshot_train_run)
from lega.hypotheses import (REGISTRY_PATH as HYPOTHESES_PATH,  # noqa: E402
                             fingerprint as registration_fingerprint,
                             load_registration, verify_registration)
from lega.ngram import evaluate as ngram_evaluate  # noqa: E402
from lega.preflight import (ALLOW_ENV,  # noqa: E402
                            allow_protocol_violations,
                            data_cache_schema_issue,
                            data_protocol_preflight,
                            data_report_binding_issue, load_data_report)
from lega.tuning import DEFAULT_TOLERANCE as TUNE_TOLERANCE  # noqa: E402
from lega.tuning import parse_floats, tune as tune_baseline  # noqa: E402
from lega.repro import (data_fingerprint, git_info, provenance_drift,  # noqa: E402
                        runtime_info, sanity_summary, sha256_file,
                        worktree_state, write_worktree_patch)
from lega.utils import (bootstrap_mean_ci, config_bool,
                        ensure_cublas_workspace_config, get_device,
                        holm_bonferroni, mcnemar_exact_p, mean_std,
                        min_detectable_effect,
                        paired_sign_flip_test, paired_sign_test,
                        paired_t_test, save_json,
                        seed_int, seed_plan, strict_int)  # noqa: E402

ALL_VARIANTS = list(VARIANTS)
# 参数匹配安慰剂：不是主实验矩阵的一部分，只有显式 --param_control 或
# --variants 里点名时才训练；它不进入 GATE_FACTORS/全因子 8 格。
PARAM_CONTROL_VARIANT = CONTROL_VARIANTS[0]
PARAM_CONTROL_PAIR = ("lega", PARAM_CONTROL_VARIANT)
# 机制分析不依赖输出门也可计算“集中但看错率/P(hit|集中）”：
# null 变体虽无 g/logit，仍应纳入；standard 无需诊断，保持跳过。
GATED_FOR_ANALYSIS = ("lega", "qgate", "null", "lega_no_len", "lega_no_conc",
                      "lega_no_query", "lega_conc_only", "lega_len_only")

# RULER 逐任务分解的固定任务列表（与 lega.ruler.TASKS 一致，避免导入
# 评测模块只为拿常量）。
RULER_TASKS = ("niah_synth", "niah_corpus", "niah_multikey", "fwe", "vt")

# 全因子消融矩阵（A=查询项 w·q, B=集中度证据 c=log(n*max_j p_j), C=长度项 b*log n）。
# 2^3 设计必须固定其余结构：8 个格子全部带输出门，none 格是 gate_only
# （只有门偏置 d），而不是无门的 standard。否则 A/B/C 的“主效应”会混入
# “加不加门”这一额外因子，表格与交互检验（同样以 gate_only 为基准）也不一致。
FACTORIAL_CELLS = (
    ("无 A/B/C（有门）", "gate_only"),
    ("A 查询", "qgate"),
    ("B 集中度", "lega_conc_only"),
    ("C 长度", "lega_len_only"),
    ("A+B", "lega_no_len"),
    ("A+C", "lega_no_conc"),
    ("B+C", "lega_no_query"),
    ("A+B+C", "lega"),
)
# 全因子范围外的无门控基线：用于 H1（LEGA vs 标准注意力）等整体对比。
# 组件“加入 X”的受控对照是 X-only vs gate_only，不是 vs standard
# （后者把输出门的存在也混进了 A/B/C 效应）。
CONTROL_CELLS = (
    ("无门控基线（全因子范围外）", "standard"),
)
# 去一法（leave-one-out）：Full 及各去掉一个组件的结果。
LOO_CELLS = (
    ("Full", "lega"),
    ("Full−A（去查询）", "lega_no_query"),
    ("Full−B（去集中度）", "lega_no_conc"),
    ("Full−C（去长度）", "lega_no_len"),
)

# 单因素消融对比：每一项只改变一个门控因子。这是“组件是否有用”唯一允许
# 进入归因的对比集合——去一法在每个因子取 1 时删掉它（Full−X vs Full），
# 全因子增量在每个因子取 0 时只加它（X-only vs gate_only）。B-only/C-only
# 与 Full 相比同时改了另外两个因子，不能当作 B/C 的单因素消融证据。
# (因子, 方向, 候选, 参照)；方向仅用于报告，不参与统计方向判定。
SINGLE_FACTOR_CONTRASTS = (
    ("A", "remove", "lega_no_query", "lega"),
    ("B", "remove", "lega_no_conc", "lega"),
    ("C", "remove", "lega_no_len", "lega"),
    ("A", "add", "qgate", "gate_only"),
    ("B", "add", "lega_conc_only", "gate_only"),
    ("C", "add", "lega_len_only", "gate_only"),
)
SINGLE_FACTOR_PAIRS = tuple(
    (cand, ref) for _factor, _direction, cand, ref in SINGLE_FACTOR_CONTRASTS)


# 门控内 A/B/C 的八个合法因子集合。单因素审计不仅核对“表里列出的对比”，
# 还要核对 GATE_FACTORS 本身确实实现了完整的 2^3：若两个变体的因子集合
# 意外相同（例如新变体复制了旧公式），只比较变体名集合是发现不了的。
_GATE_FACTOR_SUBSETS = frozenset({
    frozenset(), frozenset({"A"}), frozenset({"B"}), frozenset({"C"}),
    frozenset({"A", "B"}), frozenset({"A", "C"}), frozenset({"B", "C"}),
    frozenset({"A", "B", "C"}),
})


def _gate_factor_set(value):
    """把 GATE_FACTORS 的值规范化为因子集合；结构非法时返回 None。

    只接受字符串集合而不是字符串本身：``"ABC"`` 迭代后也是 A/B/C，若不
    排除会被误当成合法因子集合，掩盖“元数据写成了拼接字符串”的漂移。
    """
    if isinstance(value, str) or not isinstance(
            value, (set, frozenset, tuple, list)):
        return None
    try:
        factors = frozenset(value)
    except TypeError:
        return None
    if any(not isinstance(f, str) for f in factors):
        return None
    return factors


def changed_gate_factors(candidate, reference):
    """两个门控变体之间激活因子集合的对称差；任一方不在门控内返回 None。

    单因素对比要求对称差恰好一个因子。standard/null 不属于门控内 2^3，
    与它们的对比（如 LEGA vs standard）是整体方法对比，不参与组件归因。
    外部/旧产物可能传入不可哈希键或损坏的因子元数据，此时按“不在门控内”
    处理，而不是让汇总因一次查询崩溃。
    """
    if not isinstance(GATE_FACTORS, dict):
        return None
    try:
        c = _gate_factor_set(GATE_FACTORS.get(candidate))
        r = _gate_factor_set(GATE_FACTORS.get(reference))
    except TypeError:  # candidate/reference 不可哈希
        return None
    if c is None or r is None:
        return None
    known = frozenset("ABC")
    if (c - known) or (r - known):
        return None
    return frozenset(c.symmetric_difference(r))


def single_factor_change(candidate, reference):
    """候选−参照恰好只改一个因子时返回该因子（A/B/C），否则返回 None。"""
    diff = changed_gate_factors(candidate, reference)
    if diff is not None and len(diff) == 1:
        factor = next(iter(diff))
        if factor in ("A", "B", "C"):
            return factor
    return None


def single_factor_audit():
    """核对“需要单因素的消融对比”确实只改变一个门控因子。

    这是运行期自检：GATE_FACTORS 从变体实现派生，SINGLE_FACTOR_CONTRASTS
    是汇总与组件判定的唯一单因素对比表。任何一侧漂移（新增变体、改公式、
    误把 B-only/C-only 与 Full 配对）都会在这里显形，而不是被静默当成
    单组件证据。返回值写入 significance.json 并渲染进汇总页。

    审计刻意不假设输入结构永远正确：GATE_FACTORS/对比表/COMPONENT_TESTS
    都可能被旧版本、手工编辑或上游改动破坏。结构损坏要变成可读的
    ``problems``，而不是让整份汇总在解包/排序时崩溃。
    """
    problems = []
    contrasts = []
    gate_table = GATE_FACTORS if isinstance(GATE_FACTORS, dict) else {}
    if not isinstance(GATE_FACTORS, dict):
        problems.append("GATE_FACTORS 必须是变体 -> A/B/C 因子集合的映射")
    normalized = {}
    for variant, raw_factors in gate_table.items():
        if not isinstance(variant, str):
            problems.append(f"GATE_FACTORS 含非字符串变体键：{variant!r}")
            continue
        factors = _gate_factor_set(raw_factors)
        if factors is None:
            problems.append(
                f"GATE_FACTORS[{variant!r}] 不是合法的因子集合："
                f"{raw_factors!r}")
            continue
        unknown = factors - frozenset("ABC")
        if unknown:
            problems.append(
                f"GATE_FACTORS[{variant!r}] 含未知因子：{sorted(unknown)}")
            continue
        normalized[variant] = factors
    # 2^3 设计必须真的覆盖八个不同因子集合（含空集），而不只是“有 8 个
    # 变体名”。重复集合会让某个组件格实际上不存在，交互/主效应随之失真。
    actual_subsets = set(normalized.values())
    missing_subsets = _GATE_FACTOR_SUBSETS - actual_subsets
    extra_subsets = actual_subsets - _GATE_FACTOR_SUBSETS
    if missing_subsets or extra_subsets:
        problems.append(
            "GATE_FACTORS 未实现完整 2^3 因子设计：缺少 "
            + (", ".join(sorted("".join(sorted(s)) or "none"
                                for s in missing_subsets)) or "无")
            + "；多出 "
            + (", ".join(sorted("".join(sorted(s)) or "none"
                                for s in extra_subsets)) or "无"))
    # 全因子矩阵的变体集合必须与 GATE_FACTORS 完全一致；否则汇总会拿
    # 未登记因子语义的变体当 2^3 的一格。
    factorial_variants = []
    if not isinstance(FACTORIAL_CELLS, (tuple, list)):
        problems.append("FACTORIAL_CELLS 必须是 (名称, 变体) 序列")
    else:
        for entry in FACTORIAL_CELLS:
            if (not isinstance(entry, (tuple, list)) or len(entry) != 2
                    or not isinstance(entry[1], str)):
                problems.append(f"全因子格定义结构非法：{entry!r}")
                continue
            factorial_variants.append(entry[1])
    if (len(set(factorial_variants)) != 8
            or set(factorial_variants) != set(normalized)):
        problems.append(
            "全因子 8 格与 GATE_FACTORS 不一致："
            f"cells={sorted(set(factorial_variants))}，"
            f"factors={sorted(normalized)}")
    # 交互检验使用的 cell -> 变体映射是另一份实验设计元数据，必须与
    # GATE_FACTORS 的因子定义逐格对齐（例如 AB 格必须真的启用 A+B）。
    cell_map = globals().get("_FACTORIAL_CONTRAST_CELLS")
    if cell_map is not None:
        if not isinstance(cell_map, dict):
            problems.append("_FACTORIAL_CONTRAST_CELLS 必须是映射")
        else:
            cell_variants = set()
            for cell, variant in cell_map.items():
                if cell == "none":
                    cell_factors = frozenset()
                elif (isinstance(cell, str) and cell
                        and len(set(cell)) == len(cell)
                        and all(ch in "ABC" for ch in cell)):
                    cell_factors = frozenset(cell)
                else:
                    cell_factors = None
                if (cell_factors is None
                        or cell_factors not in _GATE_FACTOR_SUBSETS):
                    problems.append(f"全因子格名非法：{cell!r}")
                    continue
                if not isinstance(variant, str):
                    problems.append(
                        f"全因子格 {cell!r} 的变体名非法：{variant!r}")
                    continue
                cell_variants.add(variant)
                if normalized.get(variant) != cell_factors:
                    problems.append(
                        f"全因子格 {cell!r} -> {variant!r} 的因子集合为 "
                        f"{sorted(normalized.get(variant, ()))}，"
                        f"应为 {sorted(cell_factors)}")
            if cell_variants != set(normalized):
                problems.append(
                    "交互检验的全因子变体集合与 GATE_FACTORS 不一致："
                    f"cells={sorted(cell_variants)}，"
                    f"factors={sorted(normalized)}")
    # 审计表必须恰好覆盖 A/B/C 的“加入/去掉”六条单因素边。只逐条校验
    # 已列出的对比会漏掉“整条对比被删除/重复”这种漂移：表少一行时仍
    # 全部合法，却不再覆盖某个组件的归因。
    required = {(factor, direction) for factor in ("A", "B", "C")
                for direction in ("add", "remove")}
    seen = set()
    seen_pairs = {}
    if (not isinstance(SINGLE_FACTOR_CONTRASTS, (tuple, list))
            or isinstance(SINGLE_FACTOR_CONTRASTS, str)):
        problems.append("SINGLE_FACTOR_CONTRASTS 必须是对比序列")
        contrast_entries = []
    else:
        contrast_entries = SINGLE_FACTOR_CONTRASTS
    for entry in contrast_entries:
        if not (isinstance(entry, (tuple, list)) and len(entry) == 4):
            problems.append(f"单因素对比定义结构非法：{entry!r}")
            continue
        factor, direction, cand, ref = entry
        if not all(isinstance(x, str) for x in (factor, direction, cand, ref)):
            problems.append(f"单因素对比定义含非字符串字段：{entry!r}")
            continue
        edge = (factor, direction)
        if factor not in ("A", "B", "C") or direction not in ("add", "remove"):
            problems.append(f"存在非 A/B/C 单因素声明：{factor}/{direction}")
            continue
        if edge in seen:
            problems.append(
                f"重复的单因素对比声明：{factor}/{direction}")
        seen.add(edge)
        if cand == ref:
            problems.append(f"单因素对比两侧相同：{cand} vs {ref}")
            continue
        if (cand, ref) in seen_pairs:
            problems.append(
                f"重复的单因素对比对：{cand} vs {ref}"
                f"（已声明为 {seen_pairs[(cand, ref)][0]}/"
                f"{seen_pairs[(cand, ref)][1]}）")
            continue
        seen_pairs[(cand, ref)] = (factor, direction)
        c = normalized.get(cand)
        r = normalized.get(ref)
        if c is None or r is None:
            problems.append(f"{cand} vs {ref} 不在门控变体内，无法做单因素归因")
            continue
        diff = frozenset(c.symmetric_difference(r))
        if diff != frozenset({factor}):
            problems.append(
                f"{cand} vs {ref} 声明为单因素 {factor}，实际改变 "
                f"{sorted(diff) or '无'}")
            continue
        # 方向不只是展示字段：remove 必须从参照中恰好去掉该因子，add 必须
        # 在参照上恰好加入该因子。写反方向会让“加入/去掉”的效应解释颠倒。
        if direction == "remove":
            direction_ok = (c < r and (r - c) == frozenset({factor}))
            expected_shape = f"{ref} 去掉 {factor} 得到 {cand}"
        else:
            direction_ok = (r < c and (c - r) == frozenset({factor}))
            expected_shape = f"{ref} 加入 {factor} 得到 {cand}"
        if not direction_ok:
            problems.append(
                f"{cand} vs {ref} 声明方向 {direction} 与因子变化不符："
                f"应为 {expected_shape}")
            continue
        contrasts.append({"factor": factor, "direction": direction,
                          "candidate": cand, "reference": ref,
                          "changed_factors": sorted(diff)})
    missing = required - seen
    extra = seen - required
    if missing:
        problems.append(
            "缺少单因素对比："
            + ", ".join(f"{factor}/{direction}"
                        for factor, direction in sorted(missing)))
    if extra:
        problems.append(
            "存在非 A/B/C 单因素声明的对比："
            + ", ".join(f"{factor}/{direction}"
                        for factor, direction in sorted(extra)))
    # 组件判定实际消费的检验对也必须落在已审计表内；否则改 COMPONENT_TESTS
    # 就能绕开审计去使用多因子对比（运行期而非仅测试期拦截）。
    audited_edges = {(c["candidate"], c["reference"]):
                     (c["factor"], c["direction"]) for c in contrasts}
    component_tests = globals().get("COMPONENT_TESTS")
    if not isinstance(component_tests, (tuple, list)):
        problems.append("COMPONENT_TESTS 必须是组件检验序列")
        component_tests = []
    seen_components = set()
    for entry in component_tests:
        if not (isinstance(entry, (tuple, list)) and len(entry) == 4):
            problems.append(f"组件检验定义结构非法：{entry!r}")
            continue
        name, loo_pair, add_pair, primary = entry
        token = str(name)[:1] if isinstance(name, str) and name else None
        if token not in ("A", "B", "C"):
            problems.append(
                f"组件名 {name!r} 无法映射到 A/B/C，不能作为单组件检验")
            expected = None
        else:
            expected = token
            if expected in seen_components:
                problems.append(f"组件 {expected} 存在重复的检验定义")
            seen_components.add(expected)
        if primary not in _EVIDENCE_METRICS:
            problems.append(
                f"组件 {name!r} 的主指标 {primary!r} 不是已登记指标")
        for label, pair, expected_direction in (
                ("去一法", loo_pair, "remove"),
                ("全因子", add_pair, "add")):
            if (not isinstance(pair, (tuple, list)) or len(pair) != 2
                    or not all(isinstance(x, str) for x in pair)):
                problems.append(
                    f"组件 {name!r} 的{label}对比结构非法：{pair!r}")
                continue
            edge = audited_edges.get(tuple(pair))
            if edge is None:
                problems.append(
                    f"组件 {name!r} 使用了未审计的单因素对比：{pair!r}")
                continue
            factor, direction = edge
            if expected is not None and factor != expected:
                problems.append(
                    f"组件 {name!r} 的对比 {pair!r} 实际单因素为 {factor}，"
                    f"与组件 {expected} 不一致")
            if direction != expected_direction:
                problems.append(
                    f"组件 {name!r} 的{label}对比 {pair!r} 方向为 {direction}，"
                    f"应为 {expected_direction}")
    missing_components = {"A", "B", "C"} - seen_components
    if missing_components:
        problems.append(
            "缺少组件检验：" + ", ".join(sorted(missing_components)))
    return {"ok": not problems, "problems": problems,
            "contrasts": contrasts,
            "factors": {v: sorted(f) for v, f in sorted(normalized.items())},
            "note": ("单因素对比 = 去一法 Full−X vs Full，或全因子增量 "
                     "X-only vs gate_only；整体对比（vs standard、LEGA vs "
                     "qgate）可跨多个因子，不作组件归因。")}


def build_comparison_pairs(runs, registered=None):
    """构造汇总用的同种子配对对比表。

    刻意把“单因素消融”和“整体方法对比”分开：
      * 单因素：SINGLE_FACTOR_PAIRS（去一法 3 组 + 全因子增量 3 组）；
      * 整体：各变体 vs standard，以及预注册的其它整体对比（如 H5）；
      * 不再把 lega_conc_only / lega_len_only 与 lega 相比——它们与 Full
        同时差两个因子（A+C / A+B），不是单组件消融。
    """
    runs = runs if isinstance(runs, dict) else {}
    pairs = [(cand, "standard") for cand in ALL_VARIANTS
             if cand != "standard"]
    # 参数匹配安慰剂只在显式运行时才进入对比，且核心对比是 lega vs placebo
    # （不是 vs standard）：二者门控参数/FLOPs 相同，差别只在证据是否有信息。
    if PARAM_CONTROL_VARIANT in runs:
        pairs.append((PARAM_CONTROL_VARIANT, "standard"))
        if "lega" in runs:
            pairs.append(PARAM_CONTROL_PAIR)
        # 安慰剂 vs qgate：若噪声特征本身与 qgate 等价/更好，说明“多出的
        # 门控容量”单独不伤害优化；若明显更差，则噪声特征可能有害，
        # lega vs placebo 的差异不能单独归因于证据（汇总页会并列提示）。
        if "qgate" in runs:
            pairs.append((PARAM_CONTROL_VARIANT, "qgate"))
    # gate_only 未运行时，三个“只加 X”的全因子增量对无法估计；去一法对
    # 仍保留（缺 lega 时下游会如实报“未运行”，而不是伪造证据）。
    pairs += [(cand, ref) for cand, ref in SINGLE_FACTOR_PAIRS
              if ref != "gate_only" or "gate_only" in runs]
    registered = _as_dict(registered)
    hypotheses = registered.get("hypotheses")
    for h in (hypotheses if isinstance(hypotheses, list) else []):
        if not isinstance(h, dict):
            continue
        c, r = h.get("candidate"), h.get("reference")
        if h.get("metric") in (AUROC_DIFF_METRIC, AP_DIFF_METRIC):
            continue  # 单样本机制族，由 _auroc_diff_tests/_ap_diff_tests 覆盖
        if c in VARIANTS and r in VARIANTS:
            pairs.append((c, r))
    return list(dict.fromkeys(pairs))


def _ablation_coverage(cells, present, rows_by_var):
    """按“变体存在且种子数≥MIN_SEEDS”判定消融单元格覆盖。

    只检查变体名是否出现会把单种子冒烟格当成完整覆盖；单种子既不能做
    配对检验，也无法区分方法效应与初始化噪声，必须与“未运行”分开报告。
    """
    done, missing, underpowered = [], [], []
    for name, v in cells:
        row = rows_by_var.get(v) if isinstance(rows_by_var, dict) else None
        n_seeds = _integral_value(row.get("n_seeds")) if isinstance(row, dict) else None
        if n_seeds is None or n_seeds < 0:
            n_seeds = 0
        if v not in present:
            missing.append((name, v))
        elif n_seeds < MIN_SEEDS:
            missing.append((name, v))
            underpowered.append((name, v, n_seeds))
        else:
            done.append((name, v))
    return done, missing, underpowered


# 受控变量核对时允许逐 run 不同的字段（方法与复现标识，不属于受控变量）
_AUDIT_EXEMPT_KEYS = {"variant", "seed", "device", "tag"}

# 显著性协议：同一指标家族内 Holm 校正后 p<ALPHA 才判定显著；
# 少于 MIN_SEEDS 个种子时只报描述统计，不作显著性/组件有效性判定。
ALPHA = 0.05
MIN_SEEDS = 3
# 窗口级 bootstrap 的最低样本量：窗口只有 2~3 个时，重采样分布本身几乎
# 没有自由度，却可能给出极窄区间和“显著”假象（冒烟配置就会触发）。少于
# 该门槛只报点估计/区间并明确标注证据不足，不生成显著性措辞。
MIN_PPL_WINDOW_SEEDS = 2
MIN_PPL_WINDOWS = 8

_SEED_RE = re.compile(r"_seed(\d+)$")


def _finite_number(value):
    """外部 JSON 数值 → 有限 float；bool/缺失/坏类型/NaN/Inf 返回 None。

    bool 是 int 的子类（True==1、False==0），若直接 float() 会把损坏的
    ``"em": true`` / ``"ppl": false`` / ``"val_windows": true`` 伪装成合法
    数值观测，进入跨种子均值、配对检验或 RULER 平均。汇总层读取的是外部/
    旧版产物，必须按缺失处理而不是替上游猜值。
    """
    if value is None or isinstance(value, (bool, np.bool_)):
        return None
    # float("1_0")==10.0：下划线数字串不是规范 JSON 数值，不能伪装成观测。
    if isinstance(value, str) and "_" in value:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) else None


def _test_label_key(value):
    """交互/Shapley 标签 → 可安全作为 dict key 的非空字符串，否则 None。

    汇总层读取外部/旧版产物时，标签可能是 list/dict（不可哈希）或空串；
    直接拿去做 dict key 会让整页汇总因 ``TypeError`` 崩溃。标签只用于与
    训练期结果对齐，畸形值按缺失处理，不影响其余行渲染。
    """
    return value if isinstance(value, str) and value else None


def _rate_or_none(value):
    """外部 JSON 的 AUROC/PR-AUC/F1/基率等比例 → [0,1] 内有限 float。

    这些指标有天然上界 1：坏/手改产物里的 1.5 或 -0.2 若被当成合法观测，
    会直接进入跨种子均值与配对检验，甚至伪造出“机制判别显著更好”的结论。
    越界值按缺失处理，与 analysis/ruler 的取值校验同口径。
    """
    value = _finite_number(value)
    return value if value is not None and 0.0 <= value <= 1.0 else None


def _cmd_env(cmd, env=None):
    """为单条子进程命令构造环境：PYTHONHASHSEED 与该命令 --seed 对齐。

    PYTHONHASHSEED 必须在解释器启动前设置才生效，父进程统一设一个值会让
    所有 (变体, 种子) 子进程共享第一个种子的哈希盐；这里逐命令解析
    --seed，保证每个 run 的 Python 哈希随机化也由它自己的种子唯一确定
    （没有 --seed 的命令如 latency/plots 沿用基准值）。
    """
    e = dict(env if env is not None else os.environ)
    seed_value = None
    for i, arg in enumerate(cmd):
        if arg == "--seed":
            if i + 1 >= len(cmd):
                raise ValueError("--seed 缺少取值")
            seed_value = cmd[i + 1]
            break
        if isinstance(arg, str) and arg.startswith("--seed="):
            seed_value = arg.split("=", 1)[1]
            break
    if seed_value is not None:
        e["PYTHONHASHSEED"] = str(seed_int(seed_value, "--seed"))
    ensure_cublas_workspace_config(e)
    return e


def run(cmd, env=None):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT, env=_cmd_env(cmd, env))


def _resolve_jobs(args) -> int:
    """并发度：显式 --jobs 优先；未指定时 CPU 取 min(4, 核数)，CUDA/--wandb 取 1。

    小核机器（<=4 核）默认取 核数+1：_jobs_env 已把每个子进程压到单 BLAS
    线程，总线程数不随 worker 数增加；大核机器维持 4 并发。显式 --jobs
    完全以用户值为准，不受此默认影响。
    """
    if args.jobs is not None:
        return max(1, int(args.jobs))
    if args.wandb:
        return 1
    if get_device(args.device).startswith("cuda"):
        return 1
    cpu = os.cpu_count() or 1
    return min(4, cpu) if cpu > 4 else cpu + 1


def parse_variants(spec: str) -> list:
    """解析 --variants 逗号列表，保留顺序并按首次出现去重（与 --seeds 同口径）。

    重复变体会让每个阶段生成重复命令（sanity/训练/评测/机制分析同一输出路径），
    并发执行时互相覆盖结果文件——去重后行为与显式去重的种子列表一致。
    """
    return list(dict.fromkeys(v.strip() for v in spec.split(",") if v.strip()))


def _strict_seed_int(value, parser=None, label="种子"):
    """严格种子化：bool/小数/负值/超过 2^32-1 均显式报错。

    True 与 1 在 dict 里同键、1.5 与 1 截断后相等，都会让不同的种子在
    ``dict.fromkeys`` 去重时合并，或让台账/结果目录名与实际播种不一致。
    越界种子会让全局 RNG 只播种一半后才失败，必须在启动子进程前拦截。
    """
    try:
        return seed_int(value, label)
    except ValueError as e:
        if parser is not None:
            parser.error(str(e))
        raise


def _parse_seed_list(spec, parser=None, label="--seeds"):
    """逗号分隔的整数种子列表；整数值浮点/整数字符串合法，其余显式报错。"""
    out = []
    for x in str(spec).split(","):
        x = x.strip()
        if not x:
            continue
        out.append(_strict_seed_int(x, parser, label))
    return out


def _jobs_env(jobs: int):
    """并发时按 jobs 均分每个子进程的 BLAS 线程配额；jobs<=1 返回 None。"""
    if jobs <= 1:
        return None
    env = os.environ.copy()
    n = max(1, (os.cpu_count() or 1) // jobs)
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS"):
        env[k] = str(n)
    return env


def run_many(cmds, jobs: int, env=None) -> None:
    """顺序或并发执行多条独立子进程命令。

    并发只改变多个独立 run 之间的调度，不改变单个 run 的行为；每条命令的
    PYTHONHASHSEED 按自身 --seed 设置（见 _cmd_env）。失败时等待全部命令
    结束后抛出首个（按提交顺序）非零退出码的 CalledProcessError。
    """
    jobs = max(1, int(jobs))
    if jobs == 1 or len(cmds) <= 1:
        for cmd in cmds:
            subprocess.run(cmd, check=True, cwd=ROOT, env=_cmd_env(cmd, env))
        return
    rcs = [None] * len(cmds)

    def _one(i, cmd):
        rcs[i] = subprocess.run(cmd, cwd=ROOT, env=_cmd_env(cmd, env)
                                ).returncode

    with ThreadPoolExecutor(max_workers=min(jobs, len(cmds))) as ex:
        list(ex.map(lambda t: _one(*t), enumerate(cmds)))
    for cmd, rc in zip(cmds, rcs):
        if rc != 0:
            raise subprocess.CalledProcessError(rc, cmd)


def _repo_path(path: str) -> str:
    """相对路径统一解释为仓库根目录；父进程与子进程必须看到同一位置。"""
    return path if os.path.isabs(path) else os.path.abspath(os.path.join(ROOT, path))


def mark_repro_failed(results_dir: str, *, error: str, stage: str,
                      variant: str = None, returncode: int = None) -> bool:
    """把本轮 ``repro.json`` 标成 aborted（失败轮次不能看起来仍在运行）。

    台账已有 aborted 终态，但全局 repro 是人工/下游最先打开的溯源文件：
    只写 ``started`` 内容会让失败轮次与“仍在跑”无法区分。这里保留启动期
    已记录的 git/worktree/seed_audit 等事实，只补失败终态，不伪造 completed
    的 tracking_end。文件不存在/不可解析时静默返回 False——失败路径不能
    因为补写溯源而掩盖原异常。
    """
    if not results_dir:
        return False
    path = os.path.join(results_dir, "repro.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            doc = json.load(f)
    except (OSError, ValueError):
        return False
    if not isinstance(doc, dict):
        return False
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    failure = {"time": ts, "error": error, "stage": stage}
    if variant is not None:
        failure["variant"] = variant
    if returncode is not None:
        failure["returncode"] = int(returncode)
    doc["status"] = "aborted"
    doc["failure"] = failure
    doc["tracking_end"] = dict(failure)
    try:
        save_json(doc, path)
    except OSError as e:
        print(f"[repro] ⚠ 失败终态写入 repro.json 失败（{e}）")
        return False
    return True


def _snapshot_existing_runs(ledger_common: dict) -> list:
    """失败轮次落盘已完成/进行中 run 的实际证据（不伪造未产生的产物）。

    失败发生在训练链或后处理阶段时，已跑完的 run 目录里已有 manifest/
    config/train_log/eval（甚至 checkpoint）。只写一条 aborted 会让这些
    逐 run 种子审计与指标从台账历史中消失。这里按本轮 out_root 扫描
    run_manifest.json，且只接受本轮开始之后写入的清单，避免把上一轮
    残留目录当成本轮证据；单个 run 快照失败不影响其余 run。
    """
    out_root = ledger_common.get("out_root")
    variants = ledger_common.get("variants") or []
    seeds = ledger_common.get("seeds") or []
    if not out_root or not variants or not seeds:
        return []
    n_seeds = len(seeds)
    try:
        start = time.mktime(time.strptime(
            str(ledger_common.get("time_start")), "%Y-%m-%d %H:%M:%S"))
    except (TypeError, ValueError, OverflowError):
        # 无法解析本轮开始时刻就无法证明目录属于本轮；宁可少收证据，
        # 也不能把上一轮残留 manifest 当成本轮快照。
        return []
    runs = []
    for v in variants:
        for seed in seeds:
            vdir = _variant_dir(out_root, v, seed, n_seeds)
            manifest = os.path.join(vdir, "run_manifest.json")
            if not os.path.isfile(manifest):
                continue
            try:
                # 只接受本轮开始之后写入的清单；不设回退宽限（time_start
                # 精确到秒，manifest 在本轮内写入必然 >= start）。
                if os.path.getmtime(manifest) < start:
                    continue
            except OSError:
                continue
            try:
                runs.append(snapshot_train_run(str(v), int(seed), vdir))
            except Exception as e:  # noqa: BLE001 - 失败路径不阻塞主异常
                print(f"[repro] ⚠ run 快照失败 {v}/seed{seed}: {e}")
    return runs


def append_aborted_record(ledger: str, ledger_common: dict, *, error: str,
                          stage: str, variant: str = None,
                          returncode: int = None, sanity: dict = None,
                          runs: list = None) -> None:
    """失败轮次的终态记录：附失败阶段/原因/退出码，失败实验也完整留痕。

    runs（可选）为失败前已产生的逐 run 证据快照，写入 aborted 记录；
    不传时保持旧行为（只有失败原因，不伪造 run 覆盖）。
    """
    extra = {"error": error, "stage": stage}
    if variant is not None:
        extra["variant"] = variant
    if returncode is not None:
        extra["returncode"] = int(returncode)
    append_record(ledger, build_experiment_record(
        status="aborted", time_end=time.strftime("%Y-%m-%d %H:%M:%S"),
        sanity=sanity, runs=runs, extra=extra, **ledger_common))
    # 台账 aborted 与全局 repro 的失败终态必须一起落盘；后者写失败不能
    # 影响前者的完整性，也不能掩盖调用方即将抛出的原始异常。
    try:
        mark_repro_failed(
            ledger_common.get("results_dir"), error=error, stage=stage,
            variant=variant, returncode=returncode)
    except Exception as repro_err:  # noqa: BLE001 - 失败路径只告警
        print(f"[repro] ⚠ 失败终态写入 repro.json 异常（{repro_err}）")


@contextmanager
def abort_ledger_on_failure(ledger: str, ledger_common: dict, stage):
    """任一阶段抛错时为已 started 的轮次补写 aborted 终态。"""
    try:
        yield
    except BaseException as e:
        stage_name = stage() if callable(stage) else stage
        try:
            runs = _snapshot_existing_runs(ledger_common)
            append_aborted_record(
                ledger, ledger_common, error="postrun_stage_failed",
                stage=str(stage_name),
                returncode=getattr(e, "returncode", None),
                runs=runs or None)
        except Exception as ledger_err:
            print(f"[repro] ⚠ aborted 台账写入失败（{ledger_err}）；"
                  f"原异常仍将抛出")
        raise


def record_aborted_sanity(ledger: str, ledger_common: dict, variant: str,
                          returncode: int, results_dir: str,
                          reason: str = None) -> dict:
    """sanity 闸门失败时向台账追写 aborted 记录，返回结果文件摘要。

    只留 dangling started 无法区分“被中止”与“崩溃”，也没有失败原因；
    aborted 记录内嵌失败变体、退出码与该变体的 sanity 结果摘要（含未通过
    项），供事后复现失败现场。reason 用于 stale/variant/seed 等“文件本身
    看起来 PASS 但不属于本轮”的失败：显式写入原因并把摘要 passed 改为
    False，避免 aborted 记录里出现 verdict=fail 却 failed.passed=true。
    """
    failed = sanity_summary(os.path.join(results_dir, f"sanity_{variant}.json"))
    if reason is not None:
        failed["gate_failure_reason"] = reason
        if reason != "not_passed":
            failed["passed"] = False
    append_aborted_record(
        ledger, ledger_common, error="sanity_gate_failed", stage="sanity",
        variant=variant, returncode=returncode,
        sanity={"executed": True, "verdict": {variant: "fail"},
                "reason": reason, "failed": failed})
    return failed


def check_sanity_results(variants, results_dir: str, not_before: float = None,
                         expected_seed: int = None):
    """读回本次 sanity 结果文件，返回 (verdict, failures)。

    子进程退出码 0 不等于闸门通过：结果文件可能缺失/不可解析/仍写着
    passed=false，或包含 malformed 证据。全量训练前逐个复核，失败项带
    variant/reason/summary，供 aborted 记录与排查信息复用。

    ``not_before``（可选，epoch 秒）要求结果文件本次运行后写入：防止
    sanity 子进程异常退出 0 却没写文件时，上一轮的 PASS 被当成本轮证据。
    文件系统 mtime 粒度可能较粗，调用方应传入开始时刻并预留 1 秒余量。
    ``expected_seed``（可选）核对结果内声明的 seed 与 run_all 传入的
    首种子一致；``variant`` 也必须与目标变体一致——把别的变体/种子的
    PASS 文件复制成本文件名时，不能在本轮闸门中复用。
    """
    verdict, failures = {}, []
    for v in variants:
        path = os.path.join(results_dir, f"sanity_{v}.json")
        summary = sanity_summary(path)
        reason = None
        if summary.get("present") is not True:
            reason = "missing_or_unparseable"
        elif summary.get("passed") is not True:
            reason = "not_passed"
        elif summary.get("malformed") is True:
            reason = "malformed"
        elif (summary.get("variant") is not None
              and str(summary.get("variant")) != str(v)):
            reason = "variant_mismatch"
        elif (expected_seed is not None
              and summary.get("seed") is not None):
            try:
                if strict_int(summary.get("seed"),
                              "sanity.seed") != int(expected_seed):
                    reason = "seed_mismatch"
            except (TypeError, ValueError, OverflowError):
                reason = "seed_mismatch"
        if reason is None and not_before is not None:
            try:
                mtime = os.path.getmtime(path)
            except OSError:
                reason = "missing_or_unparseable"
            else:
                if mtime < float(not_before) - 1.0:
                    reason = "stale_result"
        if reason is not None:
            failures.append({"variant": v, "reason": reason,
                             "summary": summary, "path": path})
        else:
            verdict[v] = "pass"
    return verdict, failures


def _seed_tag(seed: int, n_seeds: int) -> str:
    """多种子运行时给结果文件名加 _seed<seed> 后缀；单种子保持旧文件名。"""
    return f"_seed{seed}" if n_seeds > 1 else ""


def _variant_dir(out_root: str, variant: str, seed: int, n_seeds: int) -> str:
    """多种子运行时每个 (变体, 种子) 独立子目录；单种子保持旧布局。"""
    if n_seeds > 1:
        return os.path.join(out_root, variant, f"seed{seed}")
    return os.path.join(out_root, variant)


# 配对/审计必须共享的 PPL 评测协议：同一 split、同一上下文、同一批复用窗口
# （eval_iters/val_windows/val_tokens_evaluated）才是同一批 token 上的 PPL。
# 仅核对训练 config 不够——独立评测可被 --eval_iters 等参数重跑。
_PPL_PROTOCOL_KEYS = ("split", "eval_ctx", "eval_iters",
                      "eval_iters_requested", "val_windows",
                      "val_tokens_evaluated", "selection_set_is_eval_set")


def _protocol_value(v):
    """把评测协议字段规范化后比较（"10" 与 10、[256,512] 与元组视为相同）。"""
    if isinstance(v, (bool, np.bool_)):
        return ("bool", bool(v))
    if isinstance(v, (int, np.integer)):
        return ("num", int(v))
    if isinstance(v, str):
        # Python 的 int()/float() 都接受 "1_0"（=10），会把拼写不同的
        # 协议值（以及种子）静默合并；与 strict_int 一样显式拒绝下划线。
        if "_" in v:
            return ("raw", repr(v))
        # 只把普通十进制整数字符串规范化；"10" 与整数 10 视为同一协议值。
        if re.fullmatch(r"[+-]?\d+", v.strip()) is not None:
            return ("num", int(v))
    if isinstance(v, (float, np.floating)) or isinstance(v, str):
        try:
            f = float(v)
        except (TypeError, ValueError, OverflowError):
            return ("raw", repr(v))
        return ("num", f) if math.isfinite(f) else ("raw", repr(v))
    if isinstance(v, (list, tuple)):
        return ("seq", tuple(_protocol_value(x) for x in v))
    return ("raw", repr(v))


def _same_protocol_value(a, b) -> bool:
    return _protocol_value(a) == _protocol_value(b)


def _integral_value(value):
    """把外部 JSON 数值规范为整数；bool/非有限/小数返回 None。"""
    if isinstance(value, (bool, np.bool_)):
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, str):
        # Python 的 int()/float() 都接受 "1_0"（=10）：下划线写法既不是
        # 规范整数串，也不该与种子 10 视为同一身份（strict_int 同口径拒绝）。
        s = value.strip()
        if not s or "_" in s:
            return None
        if re.fullmatch(r"[+-]?\d+", s) is not None:
            return int(s)
        # 小数/科学计数法字符串必须精确解析：经 float() 往返会让
        # "9007199254740993.0" 变成 9007199254740992，从而伪造种子/步数
        # 身份相等。Decimal 保留十进制精确值，仅在确实为整数时转换。
        if re.fullmatch(
                r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?",
                s) is None:
            return None
        try:
            d = Decimal(s)
        except (InvalidOperation, ValueError):
            return None
        if not d.is_finite() or d != d.to_integral_value():
            return None
        return int(d)
    try:
        f = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(f) or not f.is_integer():
        return None
    return int(f)


def _same_integral_value(value, expected) -> bool:
    """两个值是否表示同一个整数（拒绝 bool/非有限/小数截断）。"""
    iv, ie = _integral_value(value), _integral_value(expected)
    return iv is not None and ie is not None and iv == ie


def _same_seed(value, expected) -> bool:
    return _same_integral_value(value, expected)


def _ppl_protocol(d) -> dict:
    """从 eval.json 提取实际生效的评测条件；缺失字段不伪造。"""
    if not isinstance(d, dict):
        return {}
    return {k: d[k] for k in _PPL_PROTOCOL_KEYS
            if k in d and d[k] is not None}


def _ruler_protocol(d) -> dict:
    """从 ruler payload 提取决定样本集/生成口径的字段；缺失字段不伪造。

    这些字段任一不同，EM 就不是同一批样例/同一生成规则上的配对观测：
    ctx_lens/n_samples/sample_seed 决定题面与抽样，corpus_source 决定
    干扰文本是否见过，max_new/fwe_max_new/vt_* 决定解码预算。
    """
    if not isinstance(d, dict):
        return {}
    keys = ("ctx_lens", "n_samples", "sample_seed", "corpus_source",
            "corpus_held_out", "max_new", "fwe_max_new", "vt_chains",
            "vt_depth", "amp")
    return {k: d[k] for k in keys if k in d and d[k] is not None}


def _ruler_cell_signature(d):
    """RULER 已评测 (任务, 上下文, 样本数) 单元；缺格或 n 不同均不可配对。

    同一 (任务, 上下文) 但 n 不同说明实际参与平均的样例数不同（例如部分
    样例构造失败被跳过），两边的聚合 EM/macro 不是同一批观测上的配对值。
    """
    if not isinstance(d, dict):
        return None
    results = d.get("results")
    if not isinstance(results, dict):
        return None
    cells = set()
    for task, by_ctx in results.items():
        if (not isinstance(task, str) or task.startswith("_")
                or not isinstance(by_ctx, dict)):
            continue
        for ctx, cell in by_ctx.items():
            if not isinstance(cell, dict):
                continue
            n = _integral_value(cell.get("n"))
            cells.add((task, str(ctx), n))
    return frozenset(cells)


def verify_controlled_setup(out_root: str, variants, seeds, n_seeds: int,
                            expected_data_digest: str = None) -> dict:
    """核对所有变体 × 种子确实除待验证模块外使用完全相同的训练条件。

    对比实验的公平性前提：优化器/LR 策略/步数/数据顺序等受控变量必须一致，
    且各 run 必须消费同一份数据、同一代码版本、同样的训练步数。该检查把
    “公平对比”从口头约定变成可执行断言，分两层：
     1. config_used.json：除 variant/seed/device/tag 外的字段逐项一致；
     2. eval.json 溯源：数据指纹、代码提交、配置 SHA-256、计划训练步数
        （steps_planned，旧格式回退 steps）跨 run 一致，并与本次流水线的
        数据指纹比对，避免"同一 config 跑了不同数据/代码版本"被静默当成可比。
        实际步数/tokens_seen 默认也必须逐项一致；启用早停（early_stopped）
        时各 run 可以停在不同的步数——这是早停的定义而非条件失控，此时按
        计划步数核对，实际步数逐 run 如实报告、不再要求相同。
     任一层不一致都会报错并列出差异。
    """
    ref = None
    ref_name = None
    mismatches = []
    missing = []
    checked = 0
    meta_keys = ("data_fingerprint", "code_commit", "config_sha256",
                 "steps_planned")
    meta_ref = None
    meta_ref_name = None
    meta_mismatches = []
    meta_missing = []
    meta_checked = 0
    early_stop_runs = []
    actual_meta = []
    eval_protocols = []
    selection_points = []
    eval_protocol_missing = []
    data_order_runs = []
    data_order_missing = []
    attention_backends = {}
    attention_key_seen = False
    attention_missing = []

    def _attention_backend(variant: str, sdpa_on: bool) -> str:
        # 含集中度项的门控必须读取 softmax 概率，无法走 SDPA；其它变体在
        # sdpa_on 时走 SDPA。若配置开启 SDPA，两类变体的注意力实现就不同，
        # PPL 差异不能全部归因于门控。集中度变体集合直接从 attention 的
        # _USE_CONC 派生，避免新增变体（如 lega_placebo）漏进审计时被误判
        # 成 SDPA 路径：lega_placebo 同样 use_conc=True，实际走显式路径。
        needs_probs = variant in _USE_CONC
        return "explicit" if needs_probs or not sdpa_on else "sdpa"
    devices = {}
    device_missing = []
    for v in variants:
        for seed in seeds:
            vdir = _variant_dir(out_root, v, seed, n_seeds)
            cfg_path = os.path.join(vdir, "config_used.json")
            if not os.path.exists(cfg_path):
                missing.append(f"{v}/seed{seed}")
            else:
                try:
                    with open(cfg_path, "r", encoding="utf-8") as f:
                        cfg = json.load(f)
                except (OSError, ValueError):
                    cfg = None
                if not isinstance(cfg, dict):
                    missing.append(f"{v}/seed{seed}(config_used.json 不可解析)")
                else:
                    # 键缺失时沿用默认 seed；显式为 null/非数值时无法证明用了
                    # 预期种子，审计应如实报“不一致”。
                    cfg_seed = cfg.get("seed", seed)
                    seed_mismatch = (cfg_seed is None
                                     or not _same_seed(cfg_seed, seed))
                    if seed_mismatch:
                        mismatches.append(f"{v}/seed{seed}: config seed={cfg.get('seed')}")
                    if "device" in cfg:
                        raw_device = cfg.get("device")
                        if isinstance(raw_device, str) and raw_device:
                            devices[f"{v}/seed{seed}"] = raw_device
                        else:
                            device_missing.append(f"{v}/seed{seed}")
                    else:
                        device_missing.append(f"{v}/seed{seed}")
                    if "use_sdpa_standard" in cfg:
                        attention_key_seen = True
                        try:
                            sdpa_on = config_bool(
                                cfg.get("use_sdpa_standard"),
                                "use_sdpa_standard")
                        except (TypeError, ValueError) as e:
                            # 外部 config_used.json 的坏值不能让审计函数自己
                            # 崩溃（它的契约是报告问题并 SystemExit），也不能
                            # 静默按默认 True 推断。
                            mismatches.append(
                                f"{v}/seed{seed}: use_sdpa_standard 非法"
                                f"（{e}）；无法确认注意力实现路径")
                            attention_backends[v] = "invalid"
                        else:
                            attention_backends[v] = _attention_backend(
                                v, sdpa_on)
                    else:
                        # train.py 的默认值是 True；config_used 未写该键时
                        # 实际仍走 SDPA。先记录缺失，稍后在有任一 run 显式
                        # 记录该键时按默认值补全，否则“部分 run 缺键”会让
                        # 混合后端漏检。
                        attention_missing.append(f"{v}/seed{seed}")
                    ctrl = {k: val for k, val in cfg.items()
                            if k not in _AUDIT_EXEMPT_KEYS}
                    if ref is None:
                        ref, ref_name = ctrl, f"{v}/seed{seed}"
                    elif ctrl != ref:
                        keys = sorted(set(ctrl) | set(ref))
                        diffs = [f"{k}: {ref.get(k)!r} vs {ctrl.get(k)!r}"
                                 for k in keys if ref.get(k) != ctrl.get(k)]
                        mismatches.append(f"{v}/seed{seed} 相对 {ref_name} 的差异: "
                                          + "; ".join(diffs[:8]))
                    checked += 1
            ev_path = os.path.join(vdir, "eval.json")
            if not os.path.exists(ev_path):
                meta_missing.append(f"{v}/seed{seed}")
                continue
            try:
                with open(ev_path, "r", encoding="utf-8") as f:
                    ev = json.load(f)
            except (OSError, ValueError):
                ev = None
            if not isinstance(ev, dict):
                meta_missing.append(f"{v}/seed{seed}(eval.json 不可解析)")
                continue
            meta = {k: ev.get(k) for k in meta_keys}
            if meta["steps_planned"] is None:
                meta["steps_planned"] = ev.get("steps")
            proto = _ppl_protocol(ev)
            if proto:
                eval_protocols.append((f"{v}/seed{seed}", proto))
            else:
                eval_protocol_missing.append(f"{v}/seed{seed}")
            # 早停标志必须是真正的 bool：字符串 "false"/"0" 若按 truthiness
            # 变成 True，会让“实际步数与计划步数不一致”的 run 逃过受控预算
            # 检查。非法值按未早停处理并显式记为元数据不一致。
            early_raw = ev.get("early_stopped")
            if isinstance(early_raw, bool):
                early_stopped = early_raw
            else:
                early_stopped = False
                if early_raw is not None:
                    meta_mismatches.append(
                        f"{v}/seed{seed}: early_stopped={early_raw!r} "
                        "不是布尔值；按未早停核对实际步数")
            actual_meta.append((f"{v}/seed{seed}", ev.get("steps"),
                                ev.get("tokens_seen"), early_stopped,
                                meta["steps_planned"]))
            order_sha = ev.get("data_order_sha256")
            if isinstance(order_sha, str) and order_sha:
                data_order_runs.append((seed, f"{v}/seed{seed}", order_sha,
                                        early_stopped))
            else:
                data_order_missing.append(f"{v}/seed{seed}")
            steps_planned_here = meta["steps_planned"]
            sel_step = _integral_value(ev.get("ckpt_step"))
            if sel_step is not None:
                # 计划步数必须是"可解释为整数"的量：字符串 "10"/10.0 可接受，
                # 但 bool（True 会被 / 当成 1）与小数（10.5）都不能作为分母，
                # 否则选型比例会用错误的预算做除法，且报告里分母与它处不一致。
                planned_int = _integral_value(steps_planned_here)
                sel_frac = None
                if planned_int is not None and planned_int > 0:
                    try:
                        sel_frac = sel_step / planned_int
                    except (OverflowError, ZeroDivisionError):
                        # 外部 JSON 的合法大整数（如 10**400）可精确解析为
                        # 整数，但转 float 做比例会溢出；比例按缺失处理，
                        # 不能让审计自己崩溃。
                        sel_frac = None
                selection_points.append((f"{v}/seed{seed}", sel_step,
                                         steps_planned_here, sel_frac))
            if early_stopped:
                early_stop_runs.append(f"{v}/seed{seed}(step {ev.get('steps')}"
                                       f"/{meta['steps_planned']})")
            here = []
            ev_seed = ev.get("seed", seed)
            ev_seed_bad = (ev_seed is None
                           or not _same_seed(ev_seed, seed))
            if ev_seed_bad:
                here.append(f"eval seed={ev.get('seed')}")
            if (expected_data_digest is not None
                    and meta["data_fingerprint"] != expected_data_digest):
                here.append(f"data_fingerprint={meta['data_fingerprint']!r} "
                            f"!= 本次流水线数据 {expected_data_digest!r}")
            if meta_ref is None:
                meta_ref, meta_ref_name = meta, f"{v}/seed{seed}"
            else:
                diffs = [f"{k}: {meta_ref.get(k)!r} vs {meta.get(k)!r}"
                         for k in meta_keys if meta_ref.get(k) != meta.get(k)]
                if diffs:
                    here.append(f"相对 {meta_ref_name} 的差异: " + "; ".join(diffs))
            if here:
                meta_mismatches.append(f"{v}/seed{seed}: " + "; ".join(here))
            meta_checked += 1
    # 实际步数/tokens_seen：只豁免真正早停的 run。其它 run 的实际步数必须
    # 等于计划步数，且彼此的实际步数/tokens_seen 必须一致；不能因为同批里
    # 有另一个 run 早停，就跳过所有 run 的实际训练量检查。
    def _same_quantity(a, b):
        # 旧产物可能把步数写成 "10"/10.0；语义相同即算一致。但 bool 是 int
        # 的子类（True 会被 float() 当成 1），小数则不能作为步数/tokens_seen：
        # 静默接受会让被污染的外部元数据冒充同一训练预算。
        if a is None and b is None:
            return True
        if isinstance(a, (bool, np.bool_)) or isinstance(b, (bool, np.bool_)):
            return False
        # float("1_0")==10.0：下划线写法不是合法训练量，不能与 10 相等。
        if ((isinstance(a, str) and "_" in a)
                or (isinstance(b, str) and "_" in b)):
            return False
        # 先按精确整数身份比较：>2**53 的合法 JSON 大整数转 float 会丢精度，
        # 10**400 更会直接 OverflowError；两者若表示同一个整数仍应判一致。
        ia, ib = _integral_value(a), _integral_value(b)
        if ia is not None and ib is not None:
            return ia == ib
        try:
            fa, fb = float(a), float(b)
        except (TypeError, ValueError, OverflowError):
            return a == b
        if not (math.isfinite(fa) and math.isfinite(fb)):
            return False
        if not (fa.is_integer() and fb.is_integer()):
            return False
        return fa == fb

    stable_actual = [(name, steps_run, tokens_run, planned)
                     for name, steps_run, tokens_run, early, planned in actual_meta
                     if not early]
    for name, steps_run, _tokens_run, planned in stable_actual:
        if not _same_quantity(steps_run, planned):
            meta_mismatches.append(
                f"{name}: 实际 steps={steps_run!r} != 计划 steps={planned!r}")
    if stable_actual:
        a_ref_name, a_ref_steps, a_ref_tokens = stable_actual[0][:3]
        for name, steps_run, tokens_run, _planned in stable_actual[1:]:
            diffs = []
            if not _same_quantity(a_ref_steps, steps_run):
                diffs.append(f"steps: {a_ref_steps!r} vs {steps_run!r}")
            if not _same_quantity(a_ref_tokens, tokens_run):
                diffs.append(f"tokens_seen: {a_ref_tokens!r} vs {tokens_run!r}")
            if diffs:
                meta_mismatches.append(f"{name} 相对 {a_ref_name} 的差异: "
                                       + "; ".join(diffs))
    # 实际训练数据顺序：同一 seed 下所有变体必须消费完全相同的窗口序列。
    # config/seed 相同只说明“应当”同序；若采样代码路径被改或 RNG 被提前
    # 消耗，实际 batch 序列可能不同，PPL 差异就混入了数据混杂。只比较非早停
    # run（早停会少消费批次，本来就不在同一训练量上）。
    by_seed_order = {}
    for seed_v, name, digest_v, early_v in data_order_runs:
        if not early_v:
            by_seed_order.setdefault(seed_v, []).append((name, digest_v))
    data_order_checked = 0
    for seed_v, items in sorted(by_seed_order.items()):
        if len(items) < 2:
            continue
        ref_name, ref_digest = items[0]
        for name, digest_v in items[1:]:
            data_order_checked += 1
            if digest_v != ref_digest:
                meta_mismatches.append(
                    f"{name}: 训练数据顺序指纹 {digest_v!r} 与 {ref_name} 的 "
                    f"{ref_digest!r} 不同（同 seed 下实际 batch 序列不一致，"
                    "PPL 差异混入数据混杂）")
    # PPL 评测协议：训练配置相同并不保证独立评测用了同一批 token。
    # 只比较双方都记录的字段（旧产物可能缺字段），缺失单独告警；只要共同
    # 字段有差异就直接失败，避免把不同窗口/上下文的 PPL 当成同条件配对。
    protocol_ref = None
    protocol_ref_name = None
    for name, proto in eval_protocols:
        if protocol_ref is None:
            protocol_ref, protocol_ref_name = proto, name
            continue
        common = sorted(set(proto) & set(protocol_ref))
        diffs = [f"{k}: {protocol_ref[k]!r} vs {proto[k]!r}"
                 for k in common
                 if not _same_protocol_value(protocol_ref[k], proto[k])]
        if diffs:
            meta_mismatches.append(
                f"{name} 相对 {protocol_ref_name} 的 PPL 评测协议差异: "
                + "; ".join(diffs)
                + "（不同评测 token 子集/上下文不可直接配对）")
    # 设备类型必须一致：CPU 与 CUDA（或不同精度后端）的数值/内核不同，
    # 若不同变体跑在不同设备上，PPL 差异不能归因于方法。只比较设备类型
    # （cuda:0/cuda:1 视为同一类型），不强制同一张卡。
    device_types = {}
    for name, raw_device in devices.items():
        device_types[name] = str(raw_device).split(":", 1)[0].strip().lower()
    mixed_device = []
    if len(set(device_types.values())) > 1:
        by_type = {}
        for name, dtype in device_types.items():
            by_type.setdefault(dtype, []).append(name)
        mixed_device = [f"{dtype}: {', '.join(sorted(names))}"
                        for dtype, names in sorted(by_type.items())]
        mismatches.append(
            "训练设备类型不一致（" + "；".join(mixed_device)
            + "）：不同设备/精度后端的数值行为不同，正式对比必须同设备类型")
    mixed_attention = []
    attention_inferred = []
    if attention_missing:
        # train.py 对缺键使用默认 True；因此无论是否有 run 显式记录该开关，
        # 缺键 run 都必须按默认值补全后再做混合后端判定。否则一份完全没有
        # 写 use_sdpa_standard 的配置会让 standard/qgate 实际走 SDPA、lega
        # 走显式 softmax，却因“没有可比较的键”被审计静默放行。
        for name in attention_missing:
            v = name.split("/seed", 1)[0]
            if v not in attention_backends:
                attention_backends[v] = _attention_backend(v, True)
                attention_inferred.append(name)
    if len(set(attention_backends.values())) > 1:
        by_backend = {}
        for v, b in attention_backends.items():
            by_backend.setdefault(b, []).append(v)
        mixed_attention = [f"{b}: {', '.join(sorted(vs))}"
                           for b, vs in sorted(by_backend.items())]
    if missing:
        print(f"[audit] 警告：缺少 {len(missing)} 个 run 的 config_used.json，"
              f"无法核对受控变量: {', '.join(missing[:6])}"
              + (" ..." if len(missing) > 6 else ""))
    if meta_missing:
        print(f"[audit] 警告：缺少 {len(meta_missing)} 个 run 的 eval.json，"
              f"无法核对数据指纹/代码版本/训练步数: {', '.join(meta_missing[:6])}"
              + (" ..." if len(meta_missing) > 6 else ""))
    if eval_protocol_missing:
        print(f"[audit] 警告：{len(eval_protocol_missing)} 个 run 的 eval.json "
              "缺少评测协议字段（eval_ctx/eval_iters/val_windows），"
              "无法核对是否在同一批 token 上评测: "
              + ", ".join(eval_protocol_missing[:6])
              + (" ..." if len(eval_protocol_missing) > 6 else ""))
    if attention_missing:
        if attention_key_seen:
            print(f"[audit] 警告：{len(attention_missing)} 个 run 的 "
                  "config_used.json 缺少 use_sdpa_standard，已按 train.py 的"
                  "默认值 True 推断实际注意力后端: "
                  + ", ".join(attention_missing[:6])
                  + (" ..." if len(attention_missing) > 6 else ""))
        else:
            print(f"[audit] 警告：全部 config_used.json 都缺少 "
                  f"use_sdpa_standard（{len(attention_missing)} 个 run），"
                  "已按 train.py 的默认值 True 推断实际注意力后端；"
                  "正式实验建议在配置中显式写入该开关")
    if data_order_missing:
        print(f"[audit] 警告：{len(data_order_missing)} 个 run 的 eval.json "
              "缺少 data_order_sha256，无法证明实际训练批次序列一致: "
              + ", ".join(data_order_missing[:6])
              + (" ..." if len(data_order_missing) > 6 else ""))
    if device_missing:
        print(f"[audit] 警告：{len(device_missing)} 个 run 的 config_used.json "
              "缺少 device，无法核对训练设备类型一致: "
              + ", ".join(device_missing[:6])
              + (" ..." if len(device_missing) > 6 else ""))
    if mixed_attention:
        mismatches.append(
            "注意力实现路径不一致（来自 use_sdpa_standard=true）："
            + "；".join(mixed_attention)
            + "。含集中度项的变体必须走显式 softmax，而其它变体走 SDPA，"
              "两者数值/内核不同；正式对比请设 use_sdpa_standard=false，"
              "让全部变体共用显式路径（延迟评测仍可单独测 standard_sdpa）")
    if mismatches or meta_mismatches:
        msg = ("受控变量自检失败：对比实验除待验证模块外必须完全一致，"
               "否则观测差异不能归因于方法本身。\n  - "
               + "\n  - ".join(mismatches + meta_mismatches))
        print(f"[audit] {msg}")
        raise SystemExit(2)
    if checked:
        print(f"[audit] 受控变量自检通过：{checked} 个 run 的优化器/LR/步数等配置一致")
    if meta_checked:
        print(f"[audit] 训练溯源自检通过：{meta_checked} 个 run 的数据指纹/"
              f"代码版本/计划训练步数一致"
              + (f"（数据指纹 {expected_data_digest}）"
                 if expected_data_digest else ""))
    if data_order_checked:
        print(f"[audit] 训练数据顺序自检通过：同 seed 下 {data_order_checked} "
              "组变体对比的实际 batch 起点序列指纹一致")
    if early_stop_runs:
        print(f"[audit] 早停启用：{len(early_stop_runs)} 个 run 未跑满计划步数"
              f"（{', '.join(early_stop_runs[:6])}"
              + (" ..." if len(early_stop_runs) > 6 else "")
              + "）；实际训练步数不等，预注册判定将按计划步数核对并降级为探索性")
    # checkpoint 选型步分布：val-loss 最优点的训练步是否在变体间差异过大
    # （同一预算下“过早选中 vs 接近末步选中”可能不是公平对照）。只报告，
    # 不作为强制失败，以免污染既有实验流水线；但在汇总中已经有展示列。
    sel_spread = None
    if selection_points:
        fracs = [p[3] for p in selection_points if p[3] is not None]
        if len(fracs) >= 2:
            sel_spread = {
                "n": len(fracs),
                "min": min(fracs), "max": max(fracs),
                "span": max(fracs) - min(fracs),
            }
    return {"ok": not (mismatches or meta_mismatches), "checked": checked,
            "eval_protocol_checked": len(eval_protocols),
            "eval_protocol_missing": eval_protocol_missing,
            "missing_configs": missing, "mismatches": mismatches,
            "attention_backends": attention_backends,
            "mixed_attention": mixed_attention,
            "attention_key_seen": bool(attention_key_seen),
            "attention_missing": attention_missing,
            "attention_inferred": attention_inferred,
            "training_budget_comparable": not early_stop_runs,
            "meta_checked": meta_checked, "missing_evals": meta_missing,
            "meta_mismatches": meta_mismatches,
            "data_order_checked": data_order_checked,
            "data_order_missing": data_order_missing,
            "device_types": device_types,
            "device_missing": device_missing,
            "mixed_device": mixed_device,
            "early_stop_runs": early_stop_runs,
            "selection_step_points": selection_points,
            "selection_step_spread": sel_spread,
            "expected_data_digest": expected_data_digest,
            "exempt_keys": sorted(_AUDIT_EXEMPT_KEYS)}


# ---------------------------------------------------------------- 汇总统计

def _seed_key(path: str, data: dict):
    m = _SEED_RE.search(os.path.splitext(os.path.basename(path))[0])
    if m:
        return int(m.group(1))
    s = data.get("seed")
    if s is None:
        return None
    # 外部/旧产物里的 seed 可能是对象、字符串、小数或 Infinity；无法解析为
    # 精确整数时按“缺少种子信息”处理，不能截断后冒充另一个种子。
    return _integral_value(s)


def _result_files(results_dir: str, prefix: str, scope_seeds) -> list:
    """按“命名约定优先”的顺序返回 <prefix>_*.json。

    多种子运行的结果带 _seed<seed> 后缀、单种子不带。两种命名可能同时
    存在（如在旧多种子目录上重跑单种子），且经 _seed_key 会落到同一
    (变体, 种子) 槽位；调用方按序赋值、后写覆盖，因此必须让与本次
    约定一致的文件排在最后，否则陈旧的旧命名文件覆盖新结果。
    """
    fps = glob.glob(os.path.join(results_dir, f"{prefix}_*.json"))
    if scope_seeds is None:
        return sorted(fps)
    if not scope_seeds:
        # 空列表表示本次 run 没有任何种子；不能回退成“扫描全部旧结果”。
        return []
    expected_tagged = len(scope_seeds) > 1

    def convention_match(fp: str) -> int:
        stem = os.path.splitext(os.path.basename(fp))[0]
        tagged = _SEED_RE.search(stem) is not None
        return int(tagged == expected_tagged)

    return sorted(fps, key=lambda fp: (convention_match(fp), fp))


def _sorted_seeds(mapping) -> list:
    if not isinstance(mapping, dict):
        return []
    return sorted(mapping.keys(),
                  key=lambda s: (s is None, s if s is not None else -1))


def _variant_seed_map(runs: dict, variant: str) -> dict:
    """取某变体的 {seed: slot}；已有带种子的结果时忽略无种子的文件，
    同一次实验不重复计为额外种子。"""
    m = (runs.get(variant) if isinstance(runs, dict) else None)
    m = m if isinstance(m, dict) else {}
    tagged = {s: slot for s, slot in m.items() if s is not None}
    return tagged if tagged else m


def cross_variant_badcase_map(runs: dict, variants) -> dict:
    """跨变体对齐用：每个变体取“最小种子”的 RULER 错误清单。

    最小种子没有可用的 RULER 错误清单时（例如只跑了其他种子，或产物损坏
    缺 badcases 键）退化到最小的有可用清单的种子；已评测且零错误的变体
    记为 [] 并参与对照。未跑过 RULER 的变体不进结果（无法参与对照）。

    badcases 必须是列表：空列表才表示“已评测且零错误”；缺失/非列表是
    无数据（旧格式或损坏产物），按 [] 处理会把参照变体伪造成全对，进而
    把其他变体的每个失败都误报成 new_vs_reference。
    """
    out = {}
    for v in variants:
        m = _variant_seed_map(runs, v)
        for s in _sorted_seeds(m):
            slot = m[s]
            r = slot.get("ruler") if isinstance(slot, dict) else None
            badcases = r.get("badcases") if isinstance(r, dict) else None
            if isinstance(badcases, list):
                out[v] = badcases
                break
    return out


def _stat(values) -> dict:
    """跨种子聚合：{mean, std, n}；无有效值时 None。

    n 只计有限值，与 mean_std 的口径一致：NaN/Inf 不参与统计也不虚增
    样本量（否则会显示“n=5”但均值实际只来自 4 个种子）。
    """
    vals = []
    for v in (values if isinstance(values, (list, tuple)) else []):
        f = _finite_number(v)
        if f is not None:
            vals.append(f)
    m, s = mean_std(vals)
    if m is None:
        return None
    return {"mean": m, "std": s, "n": len(vals)}


def _ppl_of(d):
    if not isinstance(d, dict):
        return None
    # ppl 键存在但为 null（外部/旧格式）时必须继续回退 val_ppl，不能把
    # 一个缺失字段当成“没有 PPL”。
    v = d.get("ppl")
    if v is None:
        v = d.get("val_ppl")
    v = _finite_number(v)
    # PPL = exp(loss) 不可能为负；负值只可能来自损坏/手改产物，若照常
    # 进入跨种子均值/配对检验会制造“困惑度小于 0”的假证据。0 是若干
    # 历史合成夹具使用的边界值，保留兼容（不参与 log-PPL 族）。
    return v if v is not None and v >= 0 else None


def _log_ppl_of(d):
    """log-PPL：比值尺度的配对分析必须用对数差（=几何均值比）。

    算术 PPL 差由高 PPL 的种子/变体主导；log 差对尺度对称，exp(ΔlogPPL)
    即几何均值比。原始 PPL 仍是预注册主指标，log 分析只作敏感性证据。
    """
    v = _ppl_of(d)
    if v is None or v <= 0:
        return None
    return math.log(v)


def _ppl_ci_rel_pct(d):
    """单次 run 内验证窗口抽样 CI 的相对半宽（占 PPL 点估计的百分比）。

    PPL 的跨种子 std 与此量回答不同问题：前者是训练随机性，后者是
    “只评测有限个验证窗口”的评估随机性。两者都要看，不能互相替代。
    """
    if not isinstance(d, dict):
        return None
    ci = d.get("val_ppl_ci95")
    ppl = _ppl_of(d)
    if (not isinstance(ci, (list, tuple)) or len(ci) != 2
            or ppl is None or ppl <= 0):
        return None
    lo, hi = _finite_number(ci[0]), _finite_number(ci[1])
    if lo is None or hi is None or hi < lo:
        return None
    # 两端都有限不代表差/比值可表示：hi−lo 或除以极小 ppl 仍可能溢出成
    # inf。非有限的相对半宽不能写进汇总/JSON，按缺失处理。
    try:
        half = (hi - lo) / 2.0
        rel = half / ppl * 100.0
    except (OverflowError, ZeroDivisionError):
        return None
    return rel if math.isfinite(rel) and rel >= 0.0 else None


def _eval_scalar(key):
    """从 eval.json 提取标量（缺失/非法值返回 None），用于覆盖率汇总。"""
    def get(d):
        if not isinstance(d, dict):
            return None
        return _finite_number(d.get(key))
    return get


def _ruler_task_means(d, metric: str = "em"):
    """单次 RULER 结果：各任务先在上下文档位求平均。

    默认 `metric="em"`，与预注册假设 H4 的“RULER 平均 EM”逐字一致。
    fwe 的 recall 不进入主平均（recall 不惩罚“把上下文词全列出来”的灌水
    策略），仅由 `_ruler_fwe_recall` 单独报告作诊断。

    所有进入平均的指标（EM/recall/precision/F1）都是 [0,1] 比例：坏/手改
    产物里的 1.5 或 -1 若被当成合法观测，会把 RULER 平均推到 100% 以上或
    负值，直接制造“方法有效/失效”的假结论。越界值按缺失处理，与
    `_ruler_depth_means` 的校验口径一致。

    单元格的 n 是样本数：n=0（或负数/非整数）表示该格没有有效观测，不能
    用一个空洞的 em=0 拉低 macro 平均；这与 `_ruler_avg_micro` /
    `_ruler_depth_means` 只接受正计数的口径一致。旧产物缺 n 字段时保持
    兼容（保留该格），显式 n 非法则按缺失处理。
    """
    if not isinstance(d, dict):
        return {}
    results = d.get("results")
    if not isinstance(results, dict):
        return {}
    out = {}
    for task, by_ctx in results.items():
        if (not isinstance(task, str) or task.startswith("_")
                or not isinstance(by_ctx, dict) or not by_ctx):
            continue
        vals = []
        for m in by_ctx.values():
            if not isinstance(m, dict):
                continue
            if "n" in m and m.get("n") is not None:
                n = _integral_value(m.get("n"))
                if n is None or n <= 0:
                    continue
            v = _finite_number(m.get(metric))
            if v is not None and 0.0 <= v <= 1.0:
                vals.append(v)
        if vals:
            out[task] = float(sum(vals) / len(vals))
    return out


def _ruler_fwe_recall(d):
    """fwe 的召回率（仅诊断，不进入 RULER 主平均）。

    recall 奖励“多答中一个”，对把上下文词全部列出的灌水策略不敏感；主
    指标必须用严格集合 EM（预注册口径）。召回率单列帮助区分“完全没找到
    高频词”与“找对了但多答/少答”。
    """
    tm = _ruler_task_means(d, metric="recall")
    return tm.get("fwe")


def _ruler_avg_of(d):
    """RULER 平均（macro）：任务等权，任务内上下文档位等权，统一用 EM。

    该口径忽略每格样本数 n 与任务/上下文规模差异——n=2 的格与 n=30 的
    格同样重。预注册假设的“RULER平均”固定指这一口径；样本加权口径另见
    `_ruler_avg_micro`，两者差异是“平均值定义依赖”的直接证据。
    fwe 用严格集合 EM（与 H4 的“平均 EM”一致），recall 仅单列诊断。
    """
    tm = _ruler_task_means(d)
    return float(sum(tm.values()) / len(tm)) if tm else None


def _ruler_avg_micro(d):
    """RULER 平均（micro）：按每格样本数 n 加权，所有评测样例等权。

    与 macro 相同统一用各任务 EM（fwe 为严格集合 EM）；不再让每个任务/
    上下文格等权。macro 与 micro 并报，避免“平均”二字的歧义掩盖任务间
    难度与格大小的不均。
    """
    if not isinstance(d, dict):
        return None
    results = d.get("results")
    if not isinstance(results, dict):
        return None
    cells = []
    max_weight = 0.0
    for task, by_ctx in results.items():
        if (not isinstance(task, str) or task.startswith("_")
                or not isinstance(by_ctx, dict)):
            continue
        for cell in by_ctx.values():
            if not isinstance(cell, dict):
                continue
            v, n = _finite_number(cell.get("em")), _integral_value(cell.get("n"))
            # 与 macro 口径同校：EM 是 [0,1] 比例，越界值不能参与按 n 加权。
            if v is None or not 0.0 <= v <= 1.0 or n is None or n <= 0:
                continue
            # n 是样本计数，但外部 JSON 可写入超出 float 范围的合法大整数。
            # 直接 v*n 会 OverflowError；无法表示为有限浮点的计数按损坏元数据
            # 跳过，而不是让整份汇总崩溃。
            try:
                nf = float(n)
            except (OverflowError, ValueError):
                continue
            if not math.isfinite(nf):
                continue
            cells.append((v, nf))
            max_weight = max(max_weight, nf)
    if not cells or max_weight <= 0.0:
        return None
    # 先按最大权重归一化再累加：各 nf 都是有限正数，但两个 1e308 的格
    # 相加就会溢出成 inf（随后 num/total 变成 NaN）。归一化后权重均落在
    # (0,1]，加权和与总权重都不会越过 float 上限，均值保持不变。
    num = total = 0.0
    for v, nf in cells:
        w = nf / max_weight
        num += v * w
        total += w
    return (num / total) if total > 0.0 else None


def _ruler_depth_means(d):
    """NIAH 位置分层的 macro 平均 EM：{early/middle/late: mean}。

    只统计 niah_* 任务的 (任务, 上下文, 位置箱) 单元格，每格等权，避免
    某一上下文长度或任务因样本多而支配分层分数。旧产物没有 `em_by_depth`
    时返回空 dict（不伪造）；空箱不进入结果。分层样本彼此不同，不能把
    三个均值直接当作同批样例做显著性检验——逐格 Wilson CI 见 ruler_*.json。
    """
    if not isinstance(d, dict):
        return {}
    results = d.get("results")
    if not isinstance(results, dict):
        return {}
    vals = {b: [] for b in ("early", "middle", "late")}
    for task, by_ctx in results.items():
        if (not isinstance(task, str) or not task.startswith("niah")
                or not isinstance(by_ctx, dict)):
            continue
        for cell in by_ctx.values():
            if not isinstance(cell, dict):
                continue
            by_depth = cell.get("em_by_depth")
            if not isinstance(by_depth, dict):
                continue
            for b in vals:
                rec = by_depth.get(b)
                if not isinstance(rec, dict):
                    continue
                v = _finite_number(rec.get("em"))
                n = _integral_value(rec.get("n"))
                if (v is None or n is None or n <= 0
                        or not 0.0 <= v <= 1.0):
                    continue
                vals[b].append(v)
    return {b: float(sum(v) / len(v)) for b, v in vals.items() if v}


def _ruler_depth_gap(d):
    """NIAH 位置落差 = early EM − late EM（正=经典“中间/末期更差”方向）。

    仅当两端都有数据时定义；中部单独一列，不用它替代两端落差。
    """
    dm = _ruler_depth_means(d)
    if "early" not in dm or "late" not in dm:
        return None
    return dm["early"] - dm["late"]


def _ruler_min_cell_n(d):
    """单次 RULER 结果里 (任务, 上下文) 单元的最小样本数。

    EM/Wilson CI 的精度直接由每格 n 决定；汇总只报平均分而不报最小 n，
    会把 n=2 的 0.000 与 n=30 的 0.000 当成同样可信的证据。
    """
    if not isinstance(d, dict):
        return None
    results = d.get("results")
    if not isinstance(results, dict):
        return None
    ns = []
    for task, by_ctx in results.items():
        if (not isinstance(task, str) or task.startswith("_")
                or not isinstance(by_ctx, dict)):
            continue
        for cell in by_ctx.values():
            if not isinstance(cell, dict) or cell.get("n") is None:
                continue
            n = _integral_value(cell["n"])
            # n 是评测样本数：负数/0 不是“小样本”，而是损坏计数；若参与
            # min 会让汇总把坏产物报成“最差格 n=-5”。与 _ruler_avg_micro /
            # _ruler_depth_means 同口径，只接受正计数。
            if n is not None and n > 0:
                ns.append(n)
    return min(ns) if ns else None


def _ruler_cell_sample_outcomes(d):
    """RULER payload → {(task, ctx): {sample_id: em}}；损坏时返回 None。

    保留单元格归属，因为预注册 `RULER平均` 是 macro（任务/上下文档位等权），
    而逐样例 McNemar 池化所有样例是 micro（样例等权）：两者估计量不同，
    必须分开报告。旧产物没有 samples 字段时返回 None（不伪造）。
    """
    if not isinstance(d, dict):
        return None
    results = d.get("results")
    if not isinstance(results, dict):
        return None
    out = {}
    seen = set()
    for task, by_ctx in results.items():
        if (not isinstance(task, str) or task.startswith("_")
                or not isinstance(by_ctx, dict)):
            continue
        for ctx, cell in by_ctx.items():
            if not isinstance(cell, dict):
                continue
            samples = cell.get("samples")
            n = _integral_value(cell.get("n"))
            if not isinstance(samples, list) or n is None or n != len(samples):
                return None
            cell_out = {}
            for rec in samples:
                if not isinstance(rec, dict):
                    return None
                sid = rec.get("sample_id")
                em = _num_or_none(rec.get("em"))
                if not isinstance(sid, str) or em not in (0.0, 1.0):
                    return None
                if sid in seen or sid in cell_out:
                    return None
                cell_out[sid] = em
            seen.update(cell_out)
            out[(task, str(ctx))] = cell_out
    return out


def _ruler_sample_outcomes(d):
    """RULER payload → {sample_id: em(0/1)}；缺少/损坏/单元 n 不符时返回 None。

    逐样例结果是“固定评测样例”层面配对检验的原材料。旧产物没有 samples
    字段时返回 None（不伪造），下游只跳过该 run 的样例级分析，不影响
    聚合 EM 的既有检验。
    """
    cells = _ruler_cell_sample_outcomes(d)
    if cells is None:
        return None
    out = {}
    for cell_out in cells.values():
        out.update(cell_out)
    return out


def _mcnemar_exact_p(candidate_only: int, reference_only: int) -> float:
    """兼容旧调用点：委托给 lega.utils.mcnemar_exact_p（单一实现）。"""
    return mcnemar_exact_p(candidate_only, reference_only)


def _ruler_pair_protocol_ok(ra, rb) -> bool:
    """两个 RULER payload 是否共享题面/抽样/生成预算与已评测格集合。"""
    pa, pb = _ruler_protocol(ra), _ruler_protocol(rb)
    common = set(pa) & set(pb)
    if any(not _same_protocol_value(pa[k], pb[k]) for k in common):
        return False
    ca, cb = _ruler_cell_signature(ra), _ruler_cell_signature(rb)
    return not (ca is not None and cb is not None and ca != cb)


RULER_MACRO_BOOTSTRAP = 1000

# 上下文长度剂量-反应的检验族名。刻意不使用 ``RULER:<task>`` 前缀：
# 逐任务检验是“同一估计量在任务上的分解”，而长度分层是另一个估计量
# （每个 ctx 内的 macro），混进同一 Holm 家族会改变既有逐任务校正的家族
# 大小。三者各自成族，且都标 exploratory。
LENGTH_CTX_METRIC = "RULER@ctx"
LENGTH_SLOPE_METRIC = "RULER@length_slope"
LENGTH_SPAN_METRIC = "RULER@length_span"
LENGTH_MIN_CONTEXTS = 2
# 与等价性层的 RULER SESOI 对齐（1.00 pp）。只用于“总平均是否掩盖了
# 实际重要的长上下文效应”的描述性标记，不是新的显著性门槛。
LENGTH_SESOI = 0.01


def _length_close(a, b, tol: float = 1e-9) -> bool:
    """两个 [0,1] 量级的 EM 均值是否在数值误差内相等。"""
    if a is None or b is None:
        return False
    try:
        fa, fb = float(a), float(b)
    except (TypeError, ValueError, OverflowError):
        return False
    if not (math.isfinite(fa) and math.isfinite(fb)):
        return False
    return abs(fa - fb) <= tol


def _length_trend_verdict(t: dict, kind: str) -> str:
    """趋势检验的专用结论：不能沿用“候选显著优于参照”的措辞。

    ``_apply_holm`` 的通用 verdict 只区分“更好/更差”，对“优势随长度增大”
    这类交互没有语义；这里在保留 p_adj/置换/噪声信息的前提下改写。
    """
    t = t if isinstance(t, dict) else {}
    n = t.get("n")
    padj = t.get("p_adj")
    if t.get("p") is None or padj is None:
        base = "无法检验（种子不足或配对差方差退化）"
    elif not isinstance(n, int) or isinstance(n, bool) or n < MIN_SEEDS:
        base = f"种子不足（n={n}<{MIN_SEEDS}），无法判定长度趋势"
    elif padj < ALPHA:
        delta = _finite_number(t.get("mean_diff"))
        if delta is None:
            base = "无法检验（斜率/落差不可表示）"
            return base
        delta_pp = delta * 100.0
        # 有限输入乘 100 仍可能溢出（如 1e308），不能写出 +inf。
        pp_text = (f"{delta_pp:+.2f}" if math.isfinite(delta_pp)
                   else "溢出")
        if kind == "slope":
            direction = "增大" if delta > 0 else "减小"
            base = (f"优势随上下文长度显著{direction}"
                    f"（每翻倍 {pp_text} pp，Holm p={padj:.4f}）")
        else:
            direction = "更长上下文更有利" if delta > 0 else "更长上下文更不利"
            base = (f"长−短上下文优势差显著（{direction}，"
                    f"{pp_text} pp，Holm p={padj:.4f}）")
    else:
        base = f"未检出长度趋势（Holm p={padj:.4f}）"
    # 与 _apply_holm 的通用 verdict 一致：t 与符号翻转置换结论冲突时必须
    # 显式披露，不能因为这里重写了措辞就把置换证据的警示丢掉。小样本下
    # 置换 p 的下限可能高于 α，这一句是“显著”结论最重要的稳健性限定。
    perm = _finite_number(t.get("p_perm_adj"))
    if (padj is not None and perm is not None
            and isinstance(n, int) and not isinstance(n, bool)
            and n >= MIN_SEEDS):
        if padj < ALPHA and perm >= ALPHA:
            base += "；⚠ 置换检验不显著（见注）"
        elif padj >= ALPHA and perm < ALPHA:
            base += "；置换检验显著（边缘证据，建议补种子确认）"
    return base + _noise_verdict_note(t)


def _ruler_length_analysis(runs, pairs) -> dict:
    """RULER 上下文长度分层 + 剂量-反应检验（探索性）。

    回答一个只报“RULER 平均”时无法回答的问题：**候选相对参照的优势是否
    随上下文长度变化？** 对每个 (候选, 参照)：

    1. 每个种子在每个公共请求档位（payload 的 ``ctx_lens``；不同任务的
       ``room_ctx`` 因生成预算不同，先按请求档位对齐）上先算任务等权的
       macro EM，再逐 ctx
       做同种子配对 t + 符号翻转置换 + bootstrap CI（家族 ``RULER@ctx``）；
    2. 每个种子对 ``ΔEM ~ log2(ctx)`` 拟合斜率（单位：长度每翻一倍），
       跨种子检验斜率是否偏离 0（家族 ``RULER@length_slope``）；
    3. 每个种子算 ``ΔEM(最长) − ΔEM(最短)``，不假定线性（家族
       ``RULER@length_span``）。

    三条证据互补：斜率回答“是否随长度系统变化”，首末落差回答“端点是否
    同向”，逐 ctx 表回答“哪一档在驱动结论”。任一方向与总体 RULER 平均
    相反时显式标 ``overall_vs_*_sign_flip``，避免用总平均掩盖长上下文失效。

    边界：探索性、未预注册，不改变 H1–H7 的确认性判定；每个 ctx 的估计
    仍受“有限评测样例”影响（Wilson CI 见 ruler_*.json），跨种子 t 检验
    只覆盖训练随机性；``n<MIN_SEEDS`` 只报描述、不判定。
    """
    runs = runs if isinstance(runs, dict) else {}
    ctx_tests, slope_tests, span_tests, pair_rows = [], [], [], []
    for entry in (pairs if isinstance(pairs, list) else []):
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            continue
        cand, ref = entry
        try:
            if cand not in runs or ref not in runs:
                continue
        except TypeError:
            continue
        cm = _variant_seed_map(runs, cand)
        rm = _variant_seed_map(runs, ref)
        seeds = [s for s in _sorted_seeds(cm) if s in rm]
        if not seeds:
            continue
        excluded = {"protocol_mismatch": 0, "missing_ruler": 0,
                    "missing_contexts": 0, "non_finite_overall": 0}
        seed_rows = []
        for s in seeds:
            sa = cm.get(s) if isinstance(cm.get(s), dict) else {}
            sb = rm.get(s) if isinstance(rm.get(s), dict) else {}
            ra = sa.get("ruler") if isinstance(sa.get("ruler"), dict) else None
            rb = sb.get("ruler") if isinstance(sb.get("ruler"), dict) else None
            if ra is None or rb is None:
                excluded["missing_ruler"] += 1
                continue
            if not _ruler_pair_protocol_ok(ra, rb):
                excluded["protocol_mismatch"] += 1
                continue
            ca = context_macro_means(ra, align_requested=True)
            cb = context_macro_means(rb, align_requested=True)
            if not ca or not cb or set(ca) != set(cb):
                excluded["missing_contexts"] += 1
                continue
            ctxs = sorted(ca)
            diffs = {ctx: ca[ctx] - cb[ctx] for ctx in ctxs}
            if not all(math.isfinite(v) for v in diffs.values()):
                excluded["non_finite_overall"] += 1
                continue
            overall = None
            oa, ob = _ruler_avg_of(ra), _ruler_avg_of(rb)
            if oa is not None and ob is not None:
                maybe = oa - ob
                if math.isfinite(maybe):
                    overall = maybe
            seed_rows.append({
                "seed": _integral_value(s),
                "candidate": ca, "reference": cb, "diffs": diffs,
                "overall_diff": overall,
                "ctx_average_diff": (
                    sum(diffs.values()) / len(diffs) if diffs else None),
            })
        if not seed_rows:
            continue
        # 所有被纳入趋势的种子必须共享同一组上下文档位，否则各档的 n 不同，
        # 跨种子的“同 ctx 配对”不再成立。取交集作为参照网格，只保留实际
        # 网格恰好等于交集的种子，并记账；若保留“有额外档位”的种子，
        # overall/ctx-average 会按各 seed 的完整网格计算，与这里报告的
        # 公共 ctx 轴不一致（汇总行的总体 Δ 将无法对应到 contexts）。
        common = set.intersection(*[set(r["diffs"]) for r in seed_rows])
        if not common:
            continue
        kept = [r for r in seed_rows if set(r["diffs"]) == common]
        dropped = len(seed_rows) - len(kept)
        if dropped:
            excluded["missing_contexts"] += dropped
        if not kept:
            continue
        ctxs = sorted(common)
        n_seeds = len(kept)
        per_context = []
        for ctx in ctxs:
            diffs = [r["diffs"][ctx] for r in kept]
            cand_vals = [r["candidate"][ctx] for r in kept]
            ref_vals = [r["reference"][ctx] for r in kept]
            res = paired_t_test(diffs, [0.0] * len(diffs))
            res.update({
                "candidate": cand, "reference": ref,
                "metric": LENGTH_CTX_METRIC,
                "lower_is_better": False,
                "contrast_type": "length_slice",
                "ruler_ctx": ctx,
                "n_contexts": len(ctxs),
                "candidate_mean": (sum(cand_vals) / len(cand_vals)
                                   if cand_vals else None),
                "reference_mean": (sum(ref_vals) / len(ref_vals)
                                   if ref_vals else None),
                "exploratory": True,
                "note": ("同一档上下文内的任务等权 macro EM 配对差（候选−"
                         "参照）；只覆盖训练种子随机性，不覆盖评测样例抽样"
                         "（后者见 ruler_*.json 的 Wilson CI）。"),
            })
            res = _enrich_paired(res, diffs, [0.0] * len(diffs))
            per_context.append(res)
            ctx_tests.append(res)

        # 斜率：每个种子独立拟合，跨种子检验“斜率均值是否为 0”。不能先
        # 把各 ctx 的点估计合并成一条曲线再拟合——那会把种子噪声当成
        # 确定性趋势，失去配对结构。
        slopes = []
        slope_skipped = 0
        for r in kept:
            sl = log2_slope(ctxs, [r["diffs"][c] for c in ctxs])
            if sl is None:
                slope_skipped += 1
            else:
                slopes.append(sl)
        if len(slopes) >= 1:
            res = paired_t_test(slopes, [0.0] * len(slopes))
            res.update({
                "candidate": cand, "reference": ref,
                "metric": LENGTH_SLOPE_METRIC,
                "lower_is_better": False,
                "contrast_type": "length_trend",
                "ruler_length_trend": "slope",
                "contexts": ctxs,
                "unit": "EM per doubling of context length",
                "exploratory": True,
                "note": ("逐种子 OLS 斜率 ΔEM~log2(ctx) 的单样本配对检验；"
                         "正值表示候选相对参照的优势随长度增大。斜率依赖"
                         "评测点间距，仅作趋势方向证据，不替代逐 ctx 表。"),
            })
            res = _enrich_paired(res, slopes, [0.0] * len(slopes))
            if slope_skipped:
                res["excluded_slope_seeds"] = slope_skipped
            slope_tests.append(res)
            slope_row = res
        else:
            slope_row = None

        # 首末落差：不假定线性，作为斜率的稳健对照。
        spans = []
        span_skipped = 0
        for r in kept:
            sp = span_contrast(ctxs, [r["diffs"][c] for c in ctxs])
            if sp is None:
                span_skipped += 1
            else:
                spans.append(sp)
        if len(spans) >= 1:
            res = paired_t_test(spans, [0.0] * len(spans))
            res.update({
                "candidate": cand, "reference": ref,
                "metric": LENGTH_SPAN_METRIC,
                "lower_is_better": False,
                "contrast_type": "length_trend",
                "ruler_length_trend": "span",
                "contexts": ctxs,
                "short_ctx": ctxs[0], "long_ctx": ctxs[-1],
                "unit": "EM(longest) - EM(shortest)",
                "exploratory": True,
                "note": ("逐种子“最长−最短上下文”配对差的单样本检验；"
                         "不假定线性，与斜率方向不一致时说明效应集中在中间"
                         "档位。"),
            })
            res = _enrich_paired(res, spans, [0.0] * len(spans))
            if span_skipped:
                res["excluded_span_seeds"] = span_skipped
            span_tests.append(res)
            span_row = res
        else:
            span_row = None

        overall_vals = [r["overall_diff"] for r in kept
                        if r["overall_diff"] is not None]
        ctx_avg_vals = [r["ctx_average_diff"] for r in kept
                        if r["ctx_average_diff"] is not None]
        overall_mean = (sum(overall_vals) / len(overall_vals)
                        if overall_vals else None)
        ctx_avg_mean = (sum(ctx_avg_vals) / len(ctx_avg_vals)
                        if ctx_avg_vals else None)
        # 平衡性：总体 macro 应等于各档 macro 的等权平均。不等说明任务×
        # 档位有缺失，分层均值不是总体均值的严格分解，必须显式提示。
        balanced = None
        if overall_mean is not None and ctx_avg_mean is not None:
            balanced = _length_close(overall_mean, ctx_avg_mean)
        per_ctx_means = [c.get("mean_diff") for c in per_context]
        pos = sum(1 for v in per_ctx_means if v is not None and v > 0)
        neg = sum(1 for v in per_ctx_means if v is not None and v < 0)
        zero = len(per_ctx_means) - pos - neg
        longest = per_context[-1].get("mean_diff") if per_context else None
        slope_mean = slope_row.get("mean_diff") if slope_row else None
        span_mean = span_row.get("mean_diff") if span_row else None
        # 留一上下文（leave-one-ctx-out）：只有 2~3 档 ctx 时，斜率可能
        # 完全由某一档的偶然波动驱动。逐种子在去掉任一档后重新拟合，
        # 再看跨种子平均斜率符号是否翻转；翻转说明“趋势”不是各档共同的
        # 方向，而是单点造成。与留一法检查种子同构，只作描述性稳健性。
        loo_ctx_means = {}
        if len(ctxs) >= 3:
            for j, ctx in enumerate(ctxs):
                sub = [c for k, c in enumerate(ctxs) if k != j]
                vals = []
                for r in kept:
                    sl = log2_slope(sub, [r["diffs"][c] for c in sub])
                    if sl is not None:
                        vals.append(sl)
                if vals:
                    loo_ctx_means[str(ctx)] = sum(vals) / len(vals)
        slope_loo_stable = None
        if loo_ctx_means and slope_mean is not None:
            if abs(slope_mean) <= 1e-12:
                slope_loo_stable = None
            else:
                sign = slope_mean > 0
                slope_loo_stable = all(
                    (v > 0) == sign for v in loo_ctx_means.values())
        reversal_long = (overall_mean is not None and longest is not None
                         and overall_mean * longest < 0
                         and abs(longest) > 1e-12)
        reversal_slope = (overall_mean is not None and slope_mean is not None
                          and overall_mean * slope_mean < 0
                          and abs(slope_mean) > 1e-12)
        reversal_span = (overall_mean is not None and span_mean is not None
                         and overall_mean * span_mean < 0
                         and abs(span_mean) > 1e-12)
        # 总体在 SESOI 内、最长档却超过 SESOI：总平均“看起来没差别”，
        # 但方法恰在声称有效的长上下文上失效/获益。零与负号不构成严格
        # sign flip，因此单列 masking，避免用“0 组方向相反”掩盖这一情形。
        longest_masking = bool(
            longest is not None and overall_mean is not None
            and abs(longest) >= LENGTH_SESOI
            and abs(overall_mean) < LENGTH_SESOI)
        heterogeneity = None
        finite_means = [v for v in per_ctx_means if v is not None]
        if finite_means:
            heterogeneity = max(finite_means) - min(finite_means)
        pair_rows.append({
            "candidate": cand, "reference": ref,
            "n_seeds": n_seeds,
            "contexts": ctxs,
            "n_contexts": len(ctxs),
            "minimum_contexts_ok": len(ctxs) >= LENGTH_MIN_CONTEXTS,
            "overall_mean_diff": overall_mean,
            "context_average_mean_diff": ctx_avg_mean,
            "balanced_contexts": balanced,
            "per_context": [
                {"ctx": c.get("ruler_ctx"),
                 "candidate_mean": c.get("candidate_mean"),
                 "reference_mean": c.get("reference_mean"),
                 "mean_diff": c.get("mean_diff"),
                 "std_diff": c.get("std_diff"),
                 "mean_diff_ci95": c.get("mean_diff_ci95"),
                 "mde": c.get("mde"),
                 "p": c.get("p"), "p_adj": c.get("p_adj"),
                 "p_perm": c.get("p_perm"),
                 "p_perm_adj": c.get("p_perm_adj"),
                 "n": c.get("n"),
                 "n_seed_wins": c.get("n_seed_wins"),
                 "n_seed_losses": c.get("n_seed_losses"),
                 "n_seed_ties": c.get("n_seed_ties"),
                 "loo_sign_stable": c.get("loo_sign_stable"),
                 "within_noise": c.get("within_noise"),
                 "verdict": c.get("verdict")}
                for c in per_context],
            "slope_per_doubling": (None if slope_row is None else {
                "mean_diff": slope_row.get("mean_diff"),
                "std_diff": slope_row.get("std_diff"),
                "mean_diff_ci95": slope_row.get("mean_diff_ci95"),
                "mde": slope_row.get("mde"),
                "p": slope_row.get("p"), "p_adj": slope_row.get("p_adj"),
                "p_perm": slope_row.get("p_perm"),
                "p_perm_adj": slope_row.get("p_perm_adj"),
                "n": slope_row.get("n"),
                "n_seed_wins": slope_row.get("n_seed_wins"),
                "n_seed_losses": slope_row.get("n_seed_losses"),
                "n_seed_ties": slope_row.get("n_seed_ties"),
                "loo_sign_stable": slope_row.get("loo_sign_stable"),
                "within_noise": slope_row.get("within_noise"),
                "verdict": slope_row.get("verdict")}),
            "long_minus_short": (None if span_row is None else {
                "short_ctx": ctxs[0], "long_ctx": ctxs[-1],
                "mean_diff": span_row.get("mean_diff"),
                "std_diff": span_row.get("std_diff"),
                "mean_diff_ci95": span_row.get("mean_diff_ci95"),
                "mde": span_row.get("mde"),
                "p": span_row.get("p"), "p_adj": span_row.get("p_adj"),
                "p_perm": span_row.get("p_perm"),
                "p_perm_adj": span_row.get("p_perm_adj"),
                "n": span_row.get("n"),
                "n_seed_wins": span_row.get("n_seed_wins"),
                "n_seed_losses": span_row.get("n_seed_losses"),
                "n_seed_ties": span_row.get("n_seed_ties"),
                "loo_sign_stable": span_row.get("loo_sign_stable"),
                "within_noise": span_row.get("within_noise"),
                "verdict": span_row.get("verdict")}),
            "direction_consistency": {"positive": pos, "negative": neg,
                                      "zero": zero},
            "heterogeneity_pp": (None if heterogeneity is None
                                 else heterogeneity * 100.0),
            "overall_vs_longest_sign_flip": bool(reversal_long),
            "overall_vs_slope_sign_flip": bool(reversal_slope),
            "overall_vs_span_sign_flip": bool(reversal_span),
            "long_context_masking": longest_masking,
            "slope_leave_one_ctx_out": loo_ctx_means,
            "slope_loo_sign_stable": slope_loo_stable,
            "per_seed": [
                {"seed": r["seed"],
                 "overall_diff": r["overall_diff"],
                 "context_average_diff": r["ctx_average_diff"],
                 "slope": log2_slope(ctxs, [r["diffs"][c] for c in ctxs]),
                 "long_minus_short": span_contrast(
                     ctxs, [r["diffs"][c] for c in ctxs]),
                 "diffs": {str(c): r["diffs"][c] for c in ctxs}}
                for r in kept],
            "excluded": excluded,
            "exploratory": True,
        })

    ctx_tests = _apply_holm(ctx_tests)
    slope_tests = _apply_holm(slope_tests)
    span_tests = _apply_holm(span_tests)
    for t in slope_tests:
        t["verdict"] = _length_trend_verdict(t, "slope")
    for t in span_tests:
        t["verdict"] = _length_trend_verdict(t, "span")
    # 把校正后的 p/结论回填到 pair 行，避免 summary 与 significance.json
    # 出现两套互相矛盾的判定。
    by_ctx = {(t["candidate"], t["reference"], t["ruler_ctx"]): t
              for t in ctx_tests}
    by_slope = {(t["candidate"], t["reference"]): t for t in slope_tests}
    by_span = {(t["candidate"], t["reference"]): t for t in span_tests}
    for row in pair_rows:
        key = (row["candidate"], row["reference"])
        for cell in row["per_context"]:
            t = by_ctx.get((row["candidate"], row["reference"], cell["ctx"]))
            if t is not None:
                for k in ("p_adj", "p_perm", "p_perm_adj", "verdict"):
                    cell[k] = t.get(k)
        if row["slope_per_doubling"] is not None:
            t = by_slope.get(key)
            if t is not None:
                for k in ("p_adj", "p_perm", "p_perm_adj", "verdict"):
                    row["slope_per_doubling"][k] = t.get(k)
        if row["long_minus_short"] is not None:
            t = by_span.get(key)
            if t is not None:
                for k in ("p_adj", "p_perm", "p_perm_adj", "verdict"):
                    row["long_minus_short"][k] = t.get(k)
    return {
        "pairs": pair_rows,
        "ctx_tests": ctx_tests,
        "slope_tests": slope_tests,
        "span_tests": span_tests,
        "min_seeds": MIN_SEEDS,
        "alpha": ALPHA,
        "min_contexts": LENGTH_MIN_CONTEXTS,
        "note": ("RULER 上下文长度分层与剂量-反应（探索性，未预注册）："
                 "逐 ctx 的 Δmacro EM 配对检验、逐种子斜率（ΔEM~log2 ctx）"
                 "与首末落差；三个独立 Holm 家族。用于回答“优势是否随长度"
                 "变化”，不替代预注册的 RULER平均 确认性判定。"
                 "overall_vs_longest/slope/span_sign_flip 表示总平均与长上下文/"
                 "趋势方向相反，此时只报总平均会掩盖长上下文失效。"),
    }


def _ruler_macro_bootstrap_ci(per_seed_cell_diffs, n_boot: int = RULER_MACRO_BOOTSTRAP,
                              rng_seed: int = 20240517):
    """RULER macro 的分层 bootstrap 区间（固定种子，单元格内重采样样例）。

    预注册的 macro 估计量是“任务等权、任务内上下文档位等权”：先对每个
    (任务, 上下文) 单元格内的样例重采样并求均值，再在任务内对单元格等权
    平均，最后对任务等权平均；直接重采样所有样例会退化成 micro，只对
    单元格等权则会在任务间格数不等时偏离预注册口径。固定 rng_seed 保证
    同一输入产生同一区间，便于审计复现。
    """
    n_boot = strict_int(n_boot, "n_boot")
    if n_boot < 1:
        raise ValueError(f"n_boot 必须 >= 1，收到 {n_boot!r}")
    rng_seed = strict_int(rng_seed, "rng_seed")
    if rng_seed < 0:
        raise ValueError(f"rng_seed 必须 >= 0，收到 {rng_seed!r}")
    cells = [c for c in (per_seed_cell_diffs or []) if isinstance(c, dict)]
    by_seed = []
    for c in cells:
        by_task = {}
        for key, vals in c.items():
            arr = np.asarray(vals, dtype=float)
            if arr.size == 0:
                continue
            # cell key 约定为 (task, ctx)；旧/手工输入不是序列时按整键成组。
            task = key[0] if isinstance(key, (tuple, list)) and key else key
            by_task.setdefault(task, []).append(arr)
        if by_task:
            by_seed.append(by_task)
    if not by_seed:
        return None
    rng = np.random.default_rng(rng_seed)
    reps = []
    for _ in range(n_boot):
        seed_means = []
        for by_task in by_seed:
            task_means = []
            for cell_arrays in by_task.values():
                cell_means = []
                for vals in cell_arrays:
                    idx = rng.integers(0, vals.size, size=vals.size)
                    cell_means.append(float(vals[idx].mean()))
                if cell_means:
                    task_means.append(float(np.mean(cell_means)))
            if task_means:
                seed_means.append(float(np.mean(task_means)))
        if seed_means:
            reps.append(float(np.mean(seed_means)))
    if not reps:
        return None
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return [float(lo), float(hi)]


def _ruler_macro_absolute_ci(ruler_payloads, n_boot: int = RULER_MACRO_BOOTSTRAP,
                             rng_seed: int = 20240517):
    """每个变体 RULER 绝对 macro EM 的分层 bootstrap 区间。

    点估计 `ruler_avg` 把任务等权、任务内上下文档位等权；区间必须匹配该
    估计量，因此在每个 (任务, 上下文) 单元格内对样例有放回重采样，先算
    单元格均值、再任务内与任务间等权平均。固定已观测种子，只覆盖评测样例
    抽样，不覆盖训练种子随机性（后者见跨种子 Mean ± Std 与配对 t 检验）。
    返回 {ci95, n_seeds, n_cells, n_samples, method}；旧产物缺逐样例记录或
    任一单元格为空时返回 None，不伪造区间。
    """
    n_boot = strict_int(n_boot, "n_boot")
    if n_boot < 1:
        raise ValueError(f"n_boot 必须 >= 1，收到 {n_boot!r}")
    rng_seed = strict_int(rng_seed, "rng_seed")
    if rng_seed < 0:
        raise ValueError(f"rng_seed 必须 >= 0，收到 {rng_seed!r}")
    per_seed = []
    n_cells = n_samples = 0
    for d in ruler_payloads or []:
        cells = _ruler_cell_sample_outcomes(d)
        if not cells:
            # 旧产物/损坏 payload 没有逐样例记录：不能只用其余种子算区间，
            # 否则 n_seeds 与实际观测集不符，静默丢弃一个种子。
            return None
        by_task = {}
        payload_cells = payload_samples = 0
        for (task, _ctx), samples in cells.items():
            vals = np.asarray(list(samples.values()), dtype=float)
            if vals.size == 0:
                # 任一单元格为空时无法保持预注册的“任务内单元格等权”
                # 估计量；按不可用处理，不伪造区间。
                return None
            by_task.setdefault(task, []).append(vals)
            payload_cells += 1
            payload_samples += int(vals.size)
        n_cells += payload_cells
        n_samples += payload_samples
        per_seed.append(by_task)
    if not per_seed:
        return None
    rng = np.random.default_rng(rng_seed)
    reps = []
    for _ in range(n_boot):
        seed_macros = []
        for by_task in per_seed:
            task_means = []
            for arrays in by_task.values():
                cell_means = []
                for vals in arrays:
                    idx = rng.integers(0, vals.size, size=vals.size)
                    cell_means.append(float(vals[idx].mean()))
                task_means.append(float(np.mean(cell_means)))
            seed_macros.append(float(np.mean(task_means)))
        reps.append(float(np.mean(seed_macros)))
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return {
        "ci95": [float(lo), float(hi)],
        "n_seeds": len(per_seed),
        "n_cells": n_cells,
        "n_samples": n_samples,
        "method": ("fixed observed seeds; resample samples within each "
                   "(task, ctx) cell; task-equal macro; average over seeds"),
    }


def _ruler_sample_paired_tests(runs, pairs) -> list:
    """固定评测样例上的配对 McNemar 检验（探索性稳健性证据）。

    跨种子 t 检验把每个种子的聚合 EM 当作一次观测，回答“换训练种子是否
    稳定”；但每格只有 n=20~30 个样例，聚合 EM 的评测抽样误差没有进入
    该检验。变体共享同一 sample_seed 与 sample_id，因此可以在**同一批
    题面**上逐样例配对：候选对、参照错 = 候选胜，反之 = 参照胜，相同
    结果不提供方向信息。McNemar 精确检验只消费 discordant 对，避免把
    “两边都答对/都答错”的样例算成独立证据。

    多种子合并时，每个种子内先按 sample_id 配对，再把各 seed 的
    discordant 计数相加做精确检验（同一样例在不同训练种子下重复出现，
    故该合并口径只作探索性证据；逐种子明细一并落盘供核对）。
    同时报告两种效应量口径：`mean_diff` 为样例等权 micro（与 McNemar
    池化计数一致），`macro_mean_diff` 为任务等权、任务内上下文档位等权
    （与预注册 `RULER平均` 的 macro 估计量一致）；两者不一致说明结论依赖
    “平均值”定义。
    该检验与跨种子 t 检验回答不同问题，不能互相替代：前者是“固定题面
    上的样例级差异”，后者是“训练随机性下的种子级差异”。
    """
    out = []
    runs = runs if isinstance(runs, dict) else {}
    for cand, ref in (pairs if isinstance(pairs, list) else []):
        if cand not in runs or ref not in runs:
            continue
        cm = _variant_seed_map(runs, cand)
        rm = _variant_seed_map(runs, ref)
        seeds = [s for s in _sorted_seeds(cm) if s in rm]
        cand_only = ref_only = both_correct = both_wrong = 0
        n_samples = 0
        per_seed = []
        per_seed_cell_diffs = []
        # 按 sample_id 聚合跨种子的配对差：同一样例在多个训练种子下重复
        # 出现，pooled McNemar 把它当成多条独立证据；聚类分析先对每个
        # 样例跨种子取平均，再以样例为独立单位做符号翻转检验。
        cluster_sums = {}
        skipped = 0
        for s in seeds:
            sa = cm.get(s) if isinstance(cm.get(s), dict) else {}
            sb = rm.get(s) if isinstance(rm.get(s), dict) else {}
            ra = sa.get("ruler") if isinstance(sa.get("ruler"), dict) else None
            rb = sb.get("ruler") if isinstance(sb.get("ruler"), dict) else None
            if ra is None or rb is None or not _ruler_pair_protocol_ok(ra, rb):
                skipped += 1
                continue
            ca = _ruler_cell_sample_outcomes(ra)
            cb = _ruler_cell_sample_outcomes(rb)
            if ca is None or cb is None or set(ca) != set(cb):
                skipped += 1
                continue
            cell_diffs = []
            cell_paired = {}
            cell_ok = True
            for ck in sorted(ca):
                oa, ob = ca[ck], cb[ck]
                if set(oa) != set(ob):
                    cell_ok = False
                    break
                if not oa:
                    continue
                diffs = [oa[sid] - ob[sid] for sid in sorted(oa)]
                cell_paired[ck] = diffs
                # 预注册 macro = 任务等权、任务内上下文档位等权：先按任务
                # 收集单元格差，最后先任务内平均、再任务间平均。不能直接
                # 对所有单元格等权——任务间格数不等时会偏离预注册估计量。
                task = ck[0] if isinstance(ck, (tuple, list)) and ck else ck
                cell_diffs.append((task, sum(diffs) / len(diffs)))
            if not cell_ok:
                skipped += 1
                continue
            per_seed_cell_diffs.append(cell_paired)
            task_cell_diffs = {}
            for task, diff in cell_diffs:
                task_cell_diffs.setdefault(task, []).append(diff)
            macro_seed = (float(np.mean([np.mean(v) for v in
                                         task_cell_diffs.values()]))
                          if task_cell_diffs else None)
            oa = {sid: em for cell_out in ca.values()
                  for sid, em in cell_out.items()}
            ob = {sid: em for cell_out in cb.values()
                  for sid, em in cell_out.items()}
            for sid in oa:
                cluster_sums.setdefault(sid, []).append(oa[sid] - ob[sid])
            s_cand = s_ref = s_both1 = s_both0 = 0
            for sid in sorted(oa):
                a, b = oa[sid], ob[sid]
                if a == 1.0 and b == 0.0:
                    s_cand += 1
                elif a == 0.0 and b == 1.0:
                    s_ref += 1
                elif a == 1.0 and b == 1.0:
                    s_both1 += 1
                else:
                    s_both0 += 1
            n_s = len(oa)
            per_seed.append({
                "seed": _integral_value(s), "n": n_s,
                "candidate_only": s_cand, "reference_only": s_ref,
                "both_correct": s_both1, "both_wrong": s_both0,
                "mean_diff": ((s_cand - s_ref) / n_s) if n_s else None,
                "macro_mean_diff": macro_seed,
                "n_cells": len(cell_diffs),
                "p": _mcnemar_exact_p(s_cand, s_ref),
            })
            cand_only += s_cand
            ref_only += s_ref
            both_correct += s_both1
            both_wrong += s_both0
            n_samples += n_s
        if not per_seed:
            continue
        wins = sum(1 for r in per_seed
                   if r["mean_diff"] is not None and r["mean_diff"] > 0)
        losses = sum(1 for r in per_seed
                     if r["mean_diff"] is not None and r["mean_diff"] < 0)
        ties = len(per_seed) - wins - losses
        mean_diff = ((cand_only - ref_only) / n_samples) if n_samples else None
        macro_diffs = [r["macro_mean_diff"] for r in per_seed
                       if r.get("macro_mean_diff") is not None]
        macro_mean_diff = (sum(macro_diffs) / len(macro_diffs)
                           if macro_diffs else None)
        # macro 的抽样不确定性：固定已观测种子，在每个 (任务, 上下文) 单元格
        # 内对配对样例重采样，再任务内、任务间等权平均（与预注册 macro 同口径）。
        # 该区间只覆盖评测样例抽样，不覆盖训练种子随机性（后者见种子级 t 检验）。
        macro_ci = _ruler_macro_bootstrap_ci(per_seed_cell_diffs)
        # 样例聚类（cluster-robust）：每个 sample_id 先跨种子平均，再作为
        # 独立观测做符号翻转检验；只有 1 个样例时不做检验（n<2）。
        cluster_vals = [float(np.mean(v)) for v in cluster_sums.values() if v]
        cluster_test = paired_sign_flip_test(cluster_vals,
                                             [0.0] * len(cluster_vals))
        cluster_ci = bootstrap_mean_ci(cluster_vals)
        n_cells = (per_seed[0].get("n_cells")
                   if per_seed and len({r.get("n_cells") for r in per_seed}) == 1
                   else None)
        res = {
            "candidate": cand, "reference": ref,
            "metric": "RULER样例配对", "lower_is_better": False,
            "sample_level": True, "exploratory": True,
            "n": len(per_seed), "n_seeds": len(per_seed),
            "n_samples": n_samples,
            "candidate_only": cand_only, "reference_only": ref_only,
            "both_correct": both_correct, "both_wrong": both_wrong,
            "mean_diff": mean_diff,
            "macro_mean_diff": macro_mean_diff,
            "macro_mean_diff_ci95": macro_ci,
            "macro_ci_method": (
                "cell-stratified sample bootstrap (seeds fixed): resample "
                "paired samples within each (task, ctx) cell, keep cells "
                "equal-weighted, then average over observed seeds"),
            "n_cells": n_cells,
            "p": _mcnemar_exact_p(cand_only, ref_only),
            "p_adj": None,
            "cluster_n": len(cluster_vals),
            "cluster_mean_diff": cluster_test.get("mean_diff"),
            "cluster_mean_diff_ci95": cluster_ci,
            "cluster_p": cluster_test.get("p_perm"),
            "cluster_p_exact": cluster_test.get("exact"),
            "cluster_p_adj": None,
            "cluster_verdict": None,
            "n_seed_wins": wins, "n_seed_losses": losses,
            "n_seed_ties": ties,
            "per_seed": per_seed,
            "verdict": None,
            "note": ("同一批 sample_id 上的逐样例配对；只消费 discordant "
                     "对（候选对/参照错 vs 反之），与跨种子聚合 EM 检验"
                     "回答不同问题，属探索性证据。mean_diff 为样例等权 "
                     "micro，macro_mean_diff 为任务等权、任务内上下文档位"
                     "等权（对应预注册 RULER平均 的 macro 口径）；"
                     "macro_mean_diff_ci95 为"
                     "固定种子、单元格内样例重采样的分层 bootstrap 区间"
                     "（只覆盖评测抽样，不覆盖训练随机性）。cluster_* 为"
                     "按 sample_id 聚合跨种子后的样例级稳健性检验：每个"
                     "样例先平均、再以样例为独立单位做符号翻转，避免把"
                     "同一样例在多个种子下的重复观测当成独立证据"),
        }
        if skipped:
            res["excluded_runs"] = skipped
        out.append(res)
    pads = holm_bonferroni([t.get("p") for t in out])
    for t, padj in zip(out, pads):
        t["p_adj"] = padj
        if t.get("n_samples", 0) <= 0 or padj is None:
            t["verdict"] = "无法检验（缺少逐样例结果）"
        elif padj < ALPHA:
            t["verdict"] = ("样例级显著优于（McNemar 精确，Holm 校正后）"
                            if t["mean_diff"] > 0 else
                            "样例级显著劣于（McNemar 精确，Holm 校正后）")
        else:
            t["verdict"] = (f"样例级无显著差异（McNemar 精确，Holm 校正后 "
                            f"p={padj:.4f}）")
        md_, mc_ = t.get("mean_diff"), t.get("macro_mean_diff")
        if (md_ is not None and mc_ is not None and md_ != 0
                and (mc_ == 0 or (md_ > 0) != (mc_ > 0))):
            t["estimand_conflict"] = True
            t["verdict"] += ("；⚠ micro/macro 方向不一致（macro 为 0 或反向），"
                             "结论依赖“平均值”定义")
    # 聚类检验独立 Holm 家族：以 sample_id 为独立单位，避免同一样例在
    # 多个训练种子下被重复计数。它只覆盖 micro（样例等权）口径。
    cpads = holm_bonferroni([t.get("cluster_p") for t in out])
    for t, padj in zip(out, cpads):
        t["cluster_p_adj"] = padj
        if t.get("cluster_n", 0) < 2 or padj is None:
            t["cluster_verdict"] = "无法检验（独立样例数不足）"
        elif padj < ALPHA:
            t["cluster_verdict"] = (
                "样例聚类后显著优于（符号翻转，Holm 校正后）"
                if t["cluster_mean_diff"] > 0 else
                "样例聚类后显著劣于（符号翻转，Holm 校正后）")
        else:
            t["cluster_verdict"] = (f"样例聚类后无显著差异（Holm 校正后 "
                                    f"p={padj:.4f}）")
    return out


def _eval_window_losses(d):
    """eval payload → {window_index: loss}；缺失/损坏/重复序号返回 None。

    只接受逐窗口损失与窗口序号一一对应且序号唯一的记录。旧产物没有这些
    字段时返回 None，下游跳过窗口级分析，不影响既有 PPL 配对检验。
    交叉熵损失不可能为负：负值来自坏记录/错误归一化，若参与逐窗口配对会
    污染 avg NLL 差、bootstrap 区间与 PPL 比，必须整表按不可用处理。
    """
    if not isinstance(d, dict):
        return None
    losses = d.get("window_losses")
    indices = d.get("window_indices")
    if (not isinstance(losses, list) or not isinstance(indices, list)
            or not losses or len(losses) != len(indices)):
        return None
    out = {}
    for loss, idx in zip(losses, indices):
        v = _num_or_none(loss)
        i = _integral_value(idx)
        # 窗口序号是 s·ctx 的非负下标；负数/重复都不是同一批评测单元。
        if v is None or v < 0.0 or i is None or i < 0 or i in out:
            return None
        out[i] = v
    return out


def _window_bootstrap_stats(seed_diffs, n_resamples: int = 2000,
                            seed: int = 0) -> dict:
    """逐窗口配对差的分层 bootstrap（先重采样种子、再重采样窗口）。

    输入 seed_diffs 是每个种子的窗口差数组；返回平均差、百分位区间、
    中心化零分布 p 值（P(|mean*−obs| ≥ |obs|)）与同号重采样比例。
    固定 seed 保证可复现。

    同时给出 i.i.d. 窗口重采样与移动块（moving-block）重采样：窗口按时间
    顺序排列时，相邻窗口可能受同一文本片段/上下文漂移影响，把它们当成
    独立观测会低估区间。块长取各序列 ``round(n^(1/3))``（n<4 时退化为 1，
    即与 i.i.d. 相同），块起点在循环序列上均匀抽取；同时报告 lag-1
    自相关均值，便于判断两种区间为何不同。块 bootstrap 仍是近似，只作
    探索性稳健性证据。
    """
    # 重采样次数/种子是循环预算与随机源身份：0/负数会让分位数计算在空数组
    # 上崩溃，bool/小数经 int() 截断后会与调用方声明的次数不一致。
    n_resamples = strict_int(n_resamples, "n_resamples")
    if n_resamples < 1:
        raise ValueError(f"n_resamples 必须 >= 1，收到 {n_resamples!r}")
    seed = strict_int(seed, "seed")
    if seed < 0:
        raise ValueError(f"seed 必须 >= 0，收到 {seed!r}")
    arrays = []
    for d in (seed_diffs or []):
        arr = np.asarray(d, dtype=np.float64).ravel()
        # 窗口差应由调用方保证有限；这里仍独立过滤，避免直接调用方把
        # NaN/Inf 送进均值后让整份探索性证据变成 NaN。
        arr = arr[np.isfinite(arr)]
        if arr.size:
            arrays.append(arr)
    if not arrays:
        return None
    n_seeds = len(arrays)
    # 先按全局最大绝对值缩放再重采样：合法但接近 float 上限的窗口差
    # （如 ±1e308）在朴素求和/平方时会溢出成 inf，使均值、区间和 p 值
    # 全部失真。缩放后各值落在 [-1,1]，统计量按尺度等变还原。
    scale = float(max(np.max(np.abs(d)) for d in arrays))
    if scale > 0.0:
        arrays_scaled = [d / scale for d in arrays]
    else:
        arrays_scaled = [np.zeros_like(d) for d in arrays]
    obs_scaled = float(np.mean([d.mean() for d in arrays_scaled]))
    obs = float(obs_scaled * scale)
    if not math.isfinite(obs):
        return None
    rng = np.random.default_rng(seed)
    reps_scaled = np.empty(int(n_resamples), dtype=np.float64)
    for r in range(int(n_resamples)):
        sidx = rng.integers(0, n_seeds, n_seeds)
        means = []
        for si in sidx:
            d = arrays_scaled[int(si)]
            w = rng.integers(0, d.size, d.size)
            means.append(float(d[w].mean()))
        reps_scaled[r] = float(np.mean(means))
    reps = reps_scaled * scale
    if not np.all(np.isfinite(reps)):
        return None
    alpha = 0.025
    lo_s, hi_s = np.quantile(reps_scaled, [alpha, 1.0 - alpha])
    lo, hi = float(lo_s * scale), float(hi_s * scale)
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return None
    ge = int((np.abs(reps_scaled - obs_scaled) >= abs(obs_scaled)).sum())
    p_boot = (((ge + 1) / (int(n_resamples) + 1))
              if obs_scaled != 0 else 1.0)
    same_sign = (float(np.mean((reps_scaled > 0) == (obs_scaled > 0)))
                 if obs_scaled != 0 else 0.0)

    # 移动块 bootstrap：块内保留窗口顺序/相邻相关，块间独立。块长用
    # n^(1/3) 的经验规则；n<4 时块长 1（等价于 i.i.d. 重采样）。
    def _block_len(n):
        return max(1, int(round(int(n) ** (1.0 / 3.0)))) if n >= 4 else 1
    block_lengths = [_block_len(d.size) for d in arrays_scaled]
    block_reps_scaled = np.empty(int(n_resamples), dtype=np.float64)
    for r in range(int(n_resamples)):
        sidx = rng.integers(0, n_seeds, n_seeds)
        means = []
        for si in sidx:
            d = arrays_scaled[int(si)]
            bl = block_lengths[int(si)]
            if bl <= 1:
                idx = rng.integers(0, d.size, d.size)
            else:
                n_blocks = (d.size + bl - 1) // bl
                starts = rng.integers(0, d.size, n_blocks)
                idx = ((starts[:, None] + np.arange(bl)[None, :]) % d.size)
                idx = idx.reshape(-1)[:d.size]
            means.append(float(d[idx].mean()))
        block_reps_scaled[r] = float(np.mean(means))
    block_reps = block_reps_scaled * scale
    if not np.all(np.isfinite(block_reps)):
        return None
    b_lo_s, b_hi_s = np.quantile(block_reps_scaled, [alpha, 1.0 - alpha])
    b_lo, b_hi = float(b_lo_s * scale), float(b_hi_s * scale)
    if not (math.isfinite(b_lo) and math.isfinite(b_hi)):
        return None
    b_ge = int((np.abs(block_reps_scaled - obs_scaled) >= abs(obs_scaled)).sum())
    p_block = (((b_ge + 1) / (int(n_resamples) + 1))
               if obs_scaled != 0 else 1.0)
    b_same_sign = (float(np.mean((block_reps_scaled > 0) == (obs_scaled > 0)))
                   if obs_scaled != 0 else 0.0)
    lag1 = []
    for d in arrays:
        if d.size >= 3:
            x, y = d[:-1], d[1:]
            # 相关系数对尺度不变；先按本条序列的最大绝对值缩放，避免
            # 1e308 量级的窗口差在 np.std/corrcoef 内平方求和溢出成 inf/inf
            # （只会产生 NaN 与运行时警告，lag-1 信息也被静默丢弃）。
            scale_d = float(np.max(np.abs(d)))
            if scale_d > 0.0:
                x, y = x / scale_d, y / scale_d
            if np.std(x) > 0 and np.std(y) > 0:
                v = float(np.corrcoef(x, y)[0, 1])
                if math.isfinite(v):
                    lag1.append(v)
    return {"mean_diff": obs, "mean_diff_ci95": [lo, hi],
            "p": p_boot, "bootstrap_same_sign": same_sign,
            "block_mean_diff_ci95": [b_lo, b_hi],
            "block_p": p_block, "block_bootstrap_same_sign": b_same_sign,
            "block_length_min": int(min(block_lengths)),
            "block_length_max": int(max(block_lengths)),
            "mean_lag1_autocorr": (float(np.mean(lag1)) if lag1 else None),
            "n_resamples": int(n_resamples),
            "n_seeds": n_seeds,
            "n_windows_min": int(min(d.size for d in arrays)),
            "n_windows_max": int(max(d.size for d in arrays))}


def _ppl_window_paired_tests(runs, pairs, n_resamples: int = 2000,
                             seed: int = 0) -> list:
    """逐窗口配对 + 种子/窗口分层 bootstrap（探索性评估不确定性证据）。

    变体的跨种子 t 检验把“每个种子的 PPL”当作一次观测，只反映训练随机性；
    而 PPL 是在 eval_iters 个等间隔窗口上估的，有限窗口抽样本身也有误差。
    同一 split/同评测协议下各变体共享同一批窗口序号，因此可逐窗口配对：
    d_w = loss_candidate(w) − loss_reference(w)。

    这里对 (种子, 窗口) 做分层 bootstrap：先重采样训练种子，再在每个被选中
    的种子里重采样窗口，得到“换种子 + 换一批评测窗口”下平均 NLL 差的
    不确定性区间。窗口间存在空间相关，按窗口独立重采样的区间是近似值；
    因此只作探索性稳健性证据，不替代确认性种子级检验。
    """
    out = []
    runs = runs if isinstance(runs, dict) else {}
    for cand, ref in (pairs if isinstance(pairs, list) else []):
        if cand not in runs or ref not in runs:
            continue
        cm = _variant_seed_map(runs, cand)
        rm = _variant_seed_map(runs, ref)
        seeds = [s for s in _sorted_seeds(cm) if s in rm]
        seed_diffs = []
        per_seed = []
        skipped = 0
        window_counts = set()
        for s in seeds:
            sa = cm.get(s) if isinstance(cm.get(s), dict) else {}
            sb = rm.get(s) if isinstance(rm.get(s), dict) else {}
            ea = sa.get("eval") if isinstance(sa.get("eval"), dict) else None
            eb = sb.get("eval") if isinstance(sb.get("eval"), dict) else None
            if ea is None or eb is None:
                skipped += 1
                continue
            if ea.get("split", "val") != eb.get("split", "val"):
                skipped += 1
                continue
            pa, pb = _ppl_protocol(ea), _ppl_protocol(eb)
            common = set(pa) & set(pb)
            if any(not _same_protocol_value(pa[k], pb[k]) for k in common):
                skipped += 1
                continue
            wa = _eval_window_losses(ea)
            wb = _eval_window_losses(eb)
            if wa is None or wb is None or set(wa) != set(wb):
                skipped += 1
                continue
            keys = sorted(wa)
            diffs = np.asarray([wa[k] - wb[k] for k in keys], dtype=np.float64)
            if diffs.size == 0 or not np.all(np.isfinite(diffs)):
                skipped += 1
                continue
            seed_diffs.append(diffs)
            window_counts.add(int(diffs.size))
            # diffs 逐元素有限，但同号巨大值相加会溢出；与分层 bootstrap
            # 同口径用缩放后的稳定均值，避免 per_seed 明细出现 inf。
            seed_mean, _ = mean_std(diffs)
            per_seed.append({
                "seed": _integral_value(s),
                "n_windows": int(diffs.size),
                "mean_diff": seed_mean,
                "window_wins": int((diffs < 0).sum()),
                "window_losses": int((diffs > 0).sum()),
                "window_ties": int((diffs == 0).sum()),
            })
        if not seed_diffs:
            continue
        n_seeds = len(seed_diffs)
        stats = _window_bootstrap_stats(seed_diffs, n_resamples, seed)
        if stats is None:
            continue
        obs = stats["mean_diff"]
        enough = (n_seeds >= MIN_PPL_WINDOW_SEEDS
                  and min(window_counts) >= MIN_PPL_WINDOWS)
        res = {
            "candidate": cand, "reference": ref,
            "metric": "PPL窗口配对", "lower_is_better": True,
            "window_level": True, "exploratory": True,
            "n": n_seeds, "n_seeds": n_seeds,
            "n_windows_min": min(window_counts),
            "n_windows_max": max(window_counts),
            "n_window_observations": int(sum(d.size for d in seed_diffs)),
            "enough_windows": bool(enough),
            "min_window_seeds": MIN_PPL_WINDOW_SEEDS,
            "min_windows_per_seed": MIN_PPL_WINDOWS,
            "mean_diff": obs,
            "mean_diff_ci95": stats["mean_diff_ci95"],
            "block_mean_diff_ci95": stats.get("block_mean_diff_ci95"),
            "block_p": stats.get("block_p"),
            "block_bootstrap_same_sign": stats.get("block_bootstrap_same_sign"),
            "block_length_min": stats.get("block_length_min"),
            "block_length_max": stats.get("block_length_max"),
            "mean_lag1_autocorr": stats.get("mean_lag1_autocorr"),
            "ppl_ratio": float(math.exp(min(20.0, obs))),
            "p": stats["p"], "p_adj": None,
            "bootstrap_same_sign": stats["bootstrap_same_sign"],
            "n_resamples": int(n_resamples),
            "per_seed": per_seed,
            "verdict": None,
            "note": ("逐窗口 NLL 差（候选−参照），负值=候选更好；"
                     "分层 bootstrap 同时重采样种子与窗口；i.i.d. 口径忽略"
                     "窗口相关，block_* 为移动块重采样（块长 n^(1/3)）的"
                     "稳健性对照，二者均为探索性证据"),
        }
        if skipped:
            res["excluded_runs"] = skipped
        out.append(res)
    pads = holm_bonferroni([t.get("p") for t in out])
    for t, padj in zip(out, pads):
        t["p_adj"] = padj
        ci = t.get("mean_diff_ci95")
        ci_s = (f"，95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}]" if ci else "")
        if t.get("mean_diff") is None or padj is None:
            t["verdict"] = "无法检验"
        elif not t.get("enough_windows"):
            t["verdict"] = (f"证据不足（种子 {t['n_seeds']}，每种子窗口 "
                            f"{t['n_windows_min']}–{t['n_windows_max']}；"
                            f"最低 {MIN_PPL_WINDOW_SEEDS} 种子×"
                            f"{MIN_PPL_WINDOWS} 窗口），仅报点估计{ci_s}")
        elif padj < ALPHA:
            t["verdict"] = ("窗口级显著优于（Holm 校正后）"
                            if t["mean_diff"] < 0 else
                            "窗口级显著劣于（Holm 校正后）")
        else:
            t["verdict"] = (f"窗口级无显著差异（Holm 校正后 p={padj:.4f}"
                            f"{ci_s}）")
    return out


def _stat_series(seed_map, payload_key, extractor):
    vals = []
    seed_map = seed_map if isinstance(seed_map, dict) else {}
    for s in _sorted_seeds(seed_map):
        slot = seed_map.get(s)
        d = slot.get(payload_key) if isinstance(slot, dict) else None
        if d:
            vals.append(extractor(d))
    return _stat(vals)


def _fmt_stat(st, nd=3) -> str:
    if not st or st.get("mean") is None:
        return "-"
    mean = st["mean"]
    if st.get("n", 1) < 2 or st.get("std") is None:
        return f"{mean:.{nd}f} (n=1)"
    return f"{mean:.{nd}f} ± {st['std']:.{nd}f} (n={st['n']})"


def _enrich_paired(res: dict, a, b) -> dict:
    """为配对检验补充稳健性证据：置换检验 p、配对差 bootstrap CI 与 MDE。

    种子数只有 3~5 时 t 检验的正态假设并不可靠：置换检验给出分布无关的
    对照 p 值（两套结论冲突时必须显式标注，而不是静默采信 t 检验）；
    bootstrap CI 展示效应量的不确定范围；MDE 回答"以当前种子方差，
    多大的效应才检测得到"——区分"没有效应"与"功效不足"。
    均为确定性计算（固定随机种子），同一输入恒定复现。
    """
    # 配对差统一按 _paired_diffs 口径构造：bool/None/非有限值不参与，
    # bootstrap CI 与置换/符号检验使用同一批观测。
    diffs = []
    for x, y in zip(a, b):
        fx, fy = _finite_number(x), _finite_number(y)
        if fx is None or fy is None:
            continue
        d = fx - fy
        if math.isfinite(d):
            diffs.append(d)
    if res.get("n", 0) >= 1 and res.get("mean_diff") is not None:
        perm = paired_sign_flip_test(a, b)
        res["p_perm"] = perm.get("p_perm")
        res["perm_exact"] = perm.get("exact")
        sign = paired_sign_test(a, b)
        res["median_diff"] = sign.get("median_diff")
        res["p_sign"] = sign.get("p_sign")
        res["sign_n_pos"] = sign.get("n_pos")
        res["sign_n_neg"] = sign.get("n_neg")
        res["sign_n_zero"] = sign.get("n_zero")
        res["mean_diff_ci95"] = bootstrap_mean_ci(diffs)
        res["mde"] = min_detectable_effect(res.get("std_diff"), res.get("n"))
    else:
        res["p_perm"] = res["perm_exact"] = None
        res["median_diff"] = res["p_sign"] = None
        res["sign_n_pos"] = res["sign_n_neg"] = res["sign_n_zero"] = None
        res["mean_diff_ci95"] = res["mde"] = None

    # 效应 vs 噪声：同一对比里“显著”与“幅度有意义”是两件事。配对差
    # 的标准差 sd_diff 是种子/评测波动的直接尺度；|Δ|<sd_diff 表示单看
    # 任何一个种子，符号都可能翻转，这类边缘结果必须显式标注。
    diff = res.get("mean_diff")
    sd = res.get("std_diff")
    res["effect_to_noise"] = res["within_noise"] = res["below_mde"] = None
    if diff is not None and sd is not None:
        try:
            diff_f, sd_f = float(diff), float(sd)
        except (TypeError, ValueError, OverflowError):
            diff_f = sd_f = None
        if (diff_f is not None and sd_f is not None
                and math.isfinite(diff_f) and math.isfinite(sd_f)):
            res["within_noise"] = bool(sd_f > 0 and abs(diff_f) < sd_f)
            res["effect_to_noise"] = (abs(diff_f) / sd_f
                                      if sd_f > 0 else None)
            if (res["effect_to_noise"] is not None
                    and not math.isfinite(res["effect_to_noise"])):
                res["effect_to_noise"] = None
    if diff is not None and res.get("mde") is not None:
        try:
            diff_f, mde_f = abs(float(diff)), float(res["mde"])
        except (TypeError, ValueError, OverflowError):
            diff_f = mde_f = None
        if (diff_f is not None and mde_f is not None
                and math.isfinite(diff_f) and math.isfinite(mde_f)):
            res["below_mde"] = bool(abs(diff_f) < abs(mde_f))

    # 留一稳健性：n=5 时一个“幸运/倒霉种子”就可能决定均值符号与显著性。
    # 逐种子胜负计数与 leave-one-seed-out 均值方向给出两条不含正态假设的
    # 稳定性证据：若删掉任一单个种子后均值符号翻转，或逐种子胜负并非
    # 一致，则报道的“显著优于”对个体种子高度敏感，必须显式标注。
    res["n_seed_wins"] = res["n_seed_losses"] = res["n_seed_ties"] = None
    res["loo_sign_stable"] = res["loo_min_abs_mean"] = None
    if len(diffs) >= 2:
        wins = sum(1 for d in diffs if d > 0)
        losses = sum(1 for d in diffs if d < 0)
        res["n_seed_wins"], res["n_seed_losses"] = wins, losses
        res["n_seed_ties"] = len(diffs) - wins - losses
        # 与 _stable_mean_std 同口径：朴素 sum 在多个 1e308 同号配对差上会
        # 溢出成 inf，使“删一种子后的方向”失去意义。mean_std 先缩放再还原。
        loo_means = []
        for i in range(len(diffs)):
            rest = diffs[:i] + diffs[i + 1:]
            rest_mean, _ = mean_std(rest)
            if rest_mean is not None:
                loo_means.append(rest_mean)
        overall_mean, _ = mean_std(diffs)
        if overall_mean is not None and loo_means:
            sign = 1 if overall_mean > 0 else -1
            res["loo_sign_stable"] = bool(
                all((m > 0) == (sign > 0) for m in loo_means))
            res["loo_min_abs_mean"] = float(min(abs(m) for m in loo_means))

    # SESOI 解释层（TOST）：显著性检验不能证明“无差异”。这里为每个有
    # 明确定义 SESOI 的指标附加“差异是否小于实际重要阈值”的证据；
    # 最终 status 会在 _apply_holm 里用同一指标家族内 Holm 校正后的
    # TOST p 重新判定（此处先用原始 p 给独立调用方一个可用值）。
    res["equivalence"] = None
    try:
        eq = assess_paired_equivalence(
            a, b, res.get("metric"),
            lower_is_better=(res.get("lower_is_better") is True),
            alpha=ALPHA, min_seeds=MIN_SEEDS)
    except (TypeError, ValueError, OverflowError) as exc:
        res["equivalence"] = {
            "status": "invalid",
            "label": f"无法计算等价性（{type(exc).__name__}: {exc}）",
            "equivalent": False, "multiplicity_adjusted": False,
            "p_used": None, "reason": str(exc)}
    else:
        if eq is not None:
            eq.update(interpret_equivalence(
                eq, res.get("p_adj"), alpha=ALPHA, min_seeds=MIN_SEEDS))
            res["equivalence"] = eq
    return res


def _fmt_signed(value, spec: str = "+.4f") -> str:
    """有限数值按带符号格式渲染；None/NaN/Inf 等缺失值显示 '-'。

    汇总表直接对 t['mean_diff'] 做格式化的前提是统计检验一定给出了有限
    均值；n=0、方差退化或输入不可表示时该字段为 None。缺少这个保护会让
    一个坏结果文件把整份汇总变成 TypeError。
    """
    f = _finite_number(value)
    return "-" if f is None else format(f, spec)


def _fmt_mean_std(t, key: str = "mean_diff", spec: str = "+.4f") -> str:
    """配对检验的“均值 ± 标准差”渲染；任一缺失时只渲染可用部分。"""
    t = t if isinstance(t, dict) else {}
    mean = _finite_number(t.get(key))
    if mean is None:
        return "-"
    std = _finite_number(t.get("std_diff"))
    return format(mean, spec) + (f" ± {std:.4f}" if std is not None else "")


def _seed_winloss_str(t: dict) -> str:
    """逐种子配对差符号计数渲染为“胜/负/平”（无信息时返回 '-'）。"""
    if not isinstance(t, dict) or t.get("n_seed_wins") is None:
        return "-"
    return (f"{t['n_seed_wins']}/{t.get('n_seed_losses', 0)}/"
            f"{t.get('n_seed_ties', 0)}")


def _loo_stability_str(t: dict) -> str:
    """留一方向稳定性渲染：✅ 稳定 / ❌ 删任一种子即翻转 / '-' 无信息。"""
    if not isinstance(t, dict) or t.get("loo_sign_stable") is None:
        return "-"
    return "✅" if t["loo_sign_stable"] else "❌"


def _fmt_eq_value(value, eq) -> str:
    """把 TOST 尺度上的数值换算为报告单位（相对差/对数比/绝对差）。"""
    f = _finite_number(value)
    if f is None or not isinstance(eq, dict):
        return "-"
    kind = eq.get("kind")
    if kind == "relative":
        v = f * 100.0
        return f"{v:+.2f}%" if math.isfinite(v) else "溢出"
    if kind == "log_ratio":
        try:
            v = math.expm1(f) * 100.0
        except OverflowError:
            return "溢出"
        return f"{v:+.2f}%" if math.isfinite(v) else "溢出"
    scale = _finite_number(eq.get("display_scale"))
    scale = 1.0 if scale is None or scale == 0 else scale
    unit = eq.get("display_unit") or ""
    v = f * scale
    if not math.isfinite(v):
        # 有限输入 × 有限 scale 仍可能溢出（如 1e308 × 100）。
        return "溢出"
    if unit == "pp":
        return f"{v:+.2f} pp"
    if unit == "nat":
        return f"{v:+.4f} nat"
    if unit:
        return f"{v:+.4f} {unit}"
    return f"{v:+.4f}"


def _fmt_eq_margin(eq) -> str:
    """渲染 SESOI 阈值（带 ± 与单位）。"""
    if not isinstance(eq, dict):
        return "-"
    m = _finite_number(eq.get("margin"))
    if m is None:
        return "-"
    if eq.get("kind") == "relative":
        v = m * 100.0
        return f"±{v:.2f}%" if math.isfinite(v) else "±溢出"
    if eq.get("kind") == "log_ratio":
        try:
            v = math.expm1(m) * 100.0
        except OverflowError:
            return "±溢出"
        return f"±{v:.2f}%" if math.isfinite(v) else "±溢出"
    scale = _finite_number(eq.get("display_scale"))
    scale = 1.0 if scale is None or scale == 0 else scale
    unit = eq.get("display_unit") or ""
    v = m * scale
    if not math.isfinite(v):
        # 有限 margin × 有限 scale 仍可能溢出。
        return "±溢出"
    if unit == "pp":
        return f"±{v:.2f} pp"
    if unit == "nat":
        return f"±{v:.4f} nat"
    if unit:
        return f"±{v:.4f} {unit}"
    return f"±{v:.4f}"


def _eq_p_str(eq) -> str:
    """TOST p：优先 Holm 校正值；未校正时用 † 标注。"""
    if not isinstance(eq, dict):
        return "-"
    p = _finite_number(eq.get("p_tost_adj"))
    suffix = ""
    if p is None:
        p = _finite_number(eq.get("p_tost"))
        suffix = "†"
    if p is None:
        return "-"
    return f"{p:.4f}{suffix}"


def _eq_sensitivity_str(eq) -> str:
    """0.5×/1×/2× 网格的等价性标记（每格独立 Holm 校正）。"""
    if not isinstance(eq, dict):
        return "-"
    sens = eq.get("sensitivity")
    if not isinstance(sens, list) or not sens:
        return "-"
    parts = []
    for cell in sens:
        if not isinstance(cell, dict):
            continue
        factor = _finite_number(cell.get("factor"))
        n = cell.get("n")
        n_ok = (isinstance(n, int) and not isinstance(n, bool)
                and n >= MIN_SEEDS)
        if cell.get("status") == "invalid" or not n_ok:
            mark = "–"
        else:
            mark = "✅" if cell.get("equivalent") is True else "❌"
        f_s = f"{factor:g}×" if factor is not None else "?"
        parts.append(f"{f_s}{mark}")
    return " ".join(parts) if parts else "-"


# 主表之外单独做 SESOI 解释的核心指标；交互/Shapley/逐任务等探索性指标
# 若也定义 SESOI 会淹没主线，保留在 significance.json 而不进汇总主段。
# 这里用与 AUROC_DIFF_METRIC/AP_DIFF_METRIC 相同的字面量，避免常量定义
# 顺序造成的导入期 NameError（函数体在调用时才解析该元组）。
_EQ_CORE_METRICS = ("PPL", "RULER平均",
                    "AUROC差(logit−仅查询)", "AP差(logit−仅查询)")


def _equivalence_rows(tests) -> list:
    rows = []
    for t in (tests if isinstance(tests, list) else []):
        if not isinstance(t, dict):
            continue
        if t.get("interaction") or t.get("ruler_task") or t.get("shapley_factor"):
            continue
        eq = t.get("equivalence")
        if not isinstance(eq, dict):
            continue
        if t.get("metric") not in _EQ_CORE_METRICS:
            continue
        if t.get("mean_diff") is None:
            continue
        rows.append(t)
    order = {m: i for i, m in enumerate(_EQ_CORE_METRICS)}
    rows.sort(key=lambda t: (order.get(t.get("metric"), 99),
                             str(t.get("candidate")),
                             str(t.get("reference"))))
    return rows


def _equivalence_overview(tests) -> dict:
    """significance.json 用的等价性计数（只统计核心指标，与汇总表一致）。"""
    rows = _equivalence_rows(tests)
    counts = {}
    for t in rows:
        status = (t.get("equivalence") or {}).get("status")
        counts[status] = counts.get(status, 0) + 1
    margins = {}
    for t in rows:
        eq = t.get("equivalence") or {}
        metric = t.get("metric")
        if metric not in margins and metric is not None:
            margins[metric] = {
                "margin": eq.get("margin"), "kind": eq.get("kind"),
                "label": eq.get("margin_label")}
    return {
        "method": ("双单侧配对 t 检验（TOST）；等价=100(1−2α)% 区间完全落在 "
                   "±SESOI 内；TOST p 在同一指标家族内 Holm 校正"),
        "alpha": ALPHA,
        "n_rows": len(rows),
        "counts": counts,
        "margins": margins,
        "sensitivity_factors": [0.5, 1.0, 2.0],
        "note": ("SESOI 是分析前声明的协议阈值，不由当前数据估计；等价只说明"
                 "差异小于 SESOI，不等于效应为 0；未定不能写成无差异；"
                 "该解释层不改变预注册假设的 支持/拒绝/未定 判定。"),
    }


def _render_equivalence_section(tests) -> list:
    rows = _equivalence_rows(tests)
    md = ["", "### SESOI 与等价性/非劣性（TOST；补充性解释）", ""]
    md.append("显著性检验只能回答“能否拒绝 Δ=0”，不能证明“没有差异”。"
              "本表在预先声明的 SESOI（最小重要差异）上做双单侧 t 检验"
              "（TOST）：**只有 TOST 校正后显著且 90% 区间完全落在 ±SESOI "
              "内，才可写“实际等价”**；区间跨过 0 或 SESOI 边界时只能写"
              "“未定”，不能写成“无差异”。")
    if not rows:
        md.append("")
        md.append("（没有定义了 SESOI 的核心对比；或旧产物缺少 equivalence 字段。）")
        return md
    md.append("")
    md.append("| 对比 | 指标 | n | Δ | SESOI | 90% CI | p_TOST(Holm) | "
              "实际意义判定 | 0.5×/1×/2× |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for t in rows:
        eq = t.get("equivalence") or {}
        ci = eq.get("ci90")
        ci_s = ("-"
                if not (isinstance(ci, (list, tuple)) and len(ci) == 2)
                else f"[{_fmt_eq_value(ci[0], eq)}, {_fmt_eq_value(ci[1], eq)}]")
        n = eq.get("n")
        md.append(
            f"| {t.get('candidate')} vs {t.get('reference')} | "
            f"{t.get('metric')} | {n if n is not None else '-'} | "
            f"{_fmt_eq_value(eq.get('mean_diff'), eq)} | "
            f"{_fmt_eq_margin(eq)} | {ci_s} | {_eq_p_str(eq)} | "
            f"{eq.get('label', '-')} | {_eq_sensitivity_str(eq)} |")
    counts = {}
    for t in rows:
        status = (t.get("equivalence") or {}).get("status")
        counts[status] = counts.get(status, 0) + 1
    md.append("")
    md.append(
        f"本次核心对比 {len(rows)} 组：等价 "
        f"{counts.get('equivalent', 0)}、非劣 "
        f"{counts.get('noninferior', 0)}、显著但低于 SESOI "
        f"{counts.get('significant_below_sesoi', 0)}、显著且实际有意义 "
        f"{counts.get('significant_meaningful_better', 0) + counts.get('significant_meaningful_worse', 0)}、"
        f"未定 {counts.get('undetermined', 0)}、种子不足 "
        f"{counts.get('insufficient_seeds', 0)}、无效 "
        f"{counts.get('invalid', 0)}。")
    md.append("")
    md.append("注: 默认 SESOI 为 PPL 相对差 1%（log-PPL 为几何均值比 1%）、"
              "RULER EM 1.00 pp、AUROC/PR-AUC 0.010、因果 ΔNLL 0.01 nat、"
              "Shapley 0.01（PPL 为 −log 单位，RULER 为 1.00 pp）。这些是"
              "**协议约定**而非从当前数据估计；0.5×/1×/2× 网格各自在同一"
              "指标家族内 Holm 校正，结论若依赖网格点必须并列披露。"
              "p_TOST 后的 † 表示该对比未进入家族校正（通常 n<3 或补充性"
              "基线），只能作描述；等价/非劣不改变预注册的优效性判定，"
              "也不能把“未定”解释成“无差异”。")
    return md


# 因果病灶中“零假设本身有科学含义”的关键对照：安慰剂、阳性对照、
# 门恒开/恒关、证据反号、去全部证据。其余病灶仍以“是否有因果贡献”为主，
# 不强行套等价性解读。
_CAUSAL_EQ_LESIONS = ("head_permute", "gate_open", "gate_closed",
                      "sign_flip", "no_evidence")


def _causal_eq_note(t) -> str:
    """把因果病灶的 TOST 结果翻译成“该对照实际说明了什么”。"""
    if not isinstance(t, dict):
        return "-"
    eq = t.get("equivalence")
    if not isinstance(eq, dict):
        return "-"
    n = _integral_value(t.get("n"))
    no_op = _integral_value(t.get("n_no_op_seeds"))
    if (n is not None and n > 0 and no_op == n):
        return ("全部种子 no-op：训练后该组件系数已为 0，病灶没有真正改变"
                "前向；不作等价性判定（“无差异”不是“无效应”）")
    status = eq.get("status")
    lesion = t.get("lesion")
    if status == "equivalent":
        if lesion == "head_permute":
            return ("安慰剂差异在 SESOI 内：与“容量/参数分布即可解释”一致；"
                    "但不能证明效应恰为 0，也不能据此宣称头—证据对应有因果作用")
        if lesion == "gate_closed":
            return ("⚠ 阳性对照失败：关闭注意力输出仍与正常等价（差异小于 "
                    "SESOI），应先排查实现/评测，而不是解释方法")
        if lesion == "gate_open":
            return ("门控接近恒等：差异小于 SESOI，方法在该 checkpoint "
                    "退化为标准注意力")
        if lesion == "sign_flip":
            return ("门控对证据方向不敏感：反号后的差异小于 SESOI")
        if lesion == "no_evidence":
            return ("去掉全部证据项后的差异小于 SESOI：推理期证据输入的"
                    "可检测贡献有限")
        return "病灶差异小于 SESOI：推理期该机制的可检测贡献有限"
    if status == "significant_meaningful_worse":
        if lesion == "head_permute":
            return ("安慰剂显著变差且超过 SESOI：头—证据对应有因果作用，"
                    "容量/分布解释不足以说明收益")
        if lesion == "gate_closed":
            return "阳性对照通过：关闭注意力输出显著变差且超过 SESOI"
        if lesion == "gate_open":
            return "门控抑制写回：恒开显著变差且超过 SESOI"
        if lesion == "sign_flip":
            return "反号显著变差且超过 SESOI：门控确实在使用证据方向"
        return "病灶显著变差且超过 SESOI：推理期该机制有实际重要贡献"
    if status == "significant_meaningful_better":
        return ("病灶显著改善且超过 SESOI：该机制在推理期反向有害/冗余，"
                "不能写成“机制有用”")
    if status == "significant_below_sesoi":
        return ("统计显著但幅度低于 SESOI：有可重复差异，实际意义有限")
    if status == "noninferior":
        return "非劣：未检出超过 SESOI 的劣化"
    if status == "undetermined":
        return ("未定：置信区间未排除 SESOI，不能宣称安慰剂无效应或该机制"
                "无贡献；需补种子/降方差")
    if status == "insufficient_seeds":
        return "种子不足：不作等价性判定"
    return eq.get("label") or "-"


def _render_causal_equivalence_section(causal_tests) -> list:
    rows = [t for t in (causal_tests if isinstance(causal_tests, list) else [])
            if isinstance(t, dict)
            and t.get("lesion") in _CAUSAL_EQ_LESIONS
            and isinstance(t.get("equivalence"), dict)
            and t.get("mean_diff") is not None]
    md = ["", "#### 关键病灶的 SESOI 解读（安慰剂/阳性对照不能只看 p>0.05）", ""]
    if not rows:
        md.append("（缺少可解释的因果病灶等价性结果。）")
        return md
    md.append("| 变体 | 病灶 | ΔNLL | SESOI | p_TOST | 解读 |")
    md.append("|---|---|---|---|---|---|")
    for t in rows:
        eq = t["equivalence"]
        p_s = "-" if eq.get("not_applicable") is True else _eq_p_str(eq)
        md.append(
            f"| {t.get('variant')} | {t.get('lesion_label', t.get('lesion'))} | "
            f"{_fmt_eq_value(eq.get('mean_diff'), eq)} | "
            f"{_fmt_eq_margin(eq)} | {p_s} | {_causal_eq_note(t)} |")
    md.append("")
    md.append("注: 病灶 ΔNLL 的 SESOI 为 0.01 nat（约 1% PPL），是协议约定。"
              "`head_permute` 是推理期容量/分布安慰剂：**等价**支持“仅靠容量/"
              "分布即可解释”，**显著变差且超过 SESOI** 才支持头—证据对应有"
              "因果作用；`gate_closed` 是阳性对照，若等价则说明实验灵敏度不足，"
              "必须先排查实现/评测。`no_op` 行不参与等价性判定。")
    return md


def _causal_equivalence_overview(causal_tests) -> dict:
    """因果病灶的 SESOI 计数摘要（significance.json）。"""
    rows = [t for t in (causal_tests if isinstance(causal_tests, list) else [])
            if isinstance(t, dict)
            and t.get("lesion") in _CAUSAL_EQ_LESIONS
            and isinstance(t.get("equivalence"), dict)]
    counts = {}
    for t in rows:
        status = (t.get("equivalence") or {}).get("status")
        counts[status] = counts.get(status, 0) + 1
    return {
        "method": "TOST on ΔNLL vs 0，SESOI=0.01 nat；同一指标家族 Holm",
        "n_rows": len(rows), "counts": counts,
        "lesions": list(_CAUSAL_EQ_LESIONS),
        "note": ("no_op 不参与等价性判定；安慰剂等价只支持“容量/分布可解释”，"
                 "不等于效应恰为 0；gate_closed 等价是阳性对照失败信号。"),
    }


def _render_parameter_matched_control(t, capacity_test=None, ran=False) -> list:
    """渲染训练期参数匹配安慰剂对照（lega vs 同参数/同 FLOPs 的噪声证据门）。"""
    md = ["", "### 参数匹配安慰剂对照（训练期：容量/优化红利）", ""]
    if not isinstance(t, dict):
        if ran:
            md.append("⚠ 本轮运行了 `lega_placebo`，但没有与 `lega` 可配对的"
                      "共同种子/一致评测协议，无法形成容量对照；请检查 "
                      "seed 与 eval 协议，不能据此声称容量已排除。")
        else:
            md.append("⚠ 本轮未运行 `lega_placebo`。LEGA vs standard 的差异仍混有"
                      "“多出的门控参数/一次门控运算”这一容量解释；需要排除时应显式"
                      "加 `--param_control`（或在 `--variants` 中点名 "
                      "`lega_placebo`）后重跑。")
        return md
    mean = _finite_number(t.get("mean_diff"))
    std = _finite_number(t.get("std_diff"))
    ci = t.get("mean_diff_ci95")
    ci_s = (f"[{ci[0]:+.4f}, {ci[1]:+.4f}]"
            if isinstance(ci, (list, tuple)) and len(ci) == 2
            and all(_finite_number(x) is not None for x in ci) else "-")
    p_adj = _finite_number(t.get("p_adj"))
    eq = t.get("equivalence") if isinstance(t.get("equivalence"), dict) else {}
    eq_status = eq.get("status", "-")
    # 等价性结论优先于优效性 p：TOST 已判定差异落在 SESOI 内时，即使
    # 5 个种子方向一致、p_adj<alpha，也不能宣称“容量红利不能解释”——
    # 那会与同表“PPL SESOI 等价性=equivalent”自相矛盾。
    if eq_status == "equivalent":
        reading = ("落入 SESOI 等价区：无法区分 LEGA 与噪声证据对照，"
                   "不能把收益归因于 B/C 证据（容量/优化红利未被排除）")
    elif mean is not None and p_adj is not None and p_adj < ALPHA and mean < 0:
        if t.get("within_noise") or t.get("loo_sign_stable") is False:
            reading = "方向有利但效应落在种子噪声内/留一翻转：只能算线索"
        else:
            reading = ("证据特定性成立：同参数/同 FLOPs 下，真实 B/C 证据优于"
                       "噪声证据；容量/优化红利不能单独解释")
    elif mean is not None and mean > 0 and p_adj is not None and p_adj < ALPHA:
        reading = "噪声证据反而更好：B/C 证据项可能有害或训练不稳定"
    else:
        reading = "未定：方向/显著性不足，须对照 MDE 与 TOST 解读，不能写成“无差异”"
    md.append("| 对比 | ΔPPL均值 ± ΔSD | Δ 95% CI | n | 胜/负/平 | p_adj(Holm) | "
              "PPL SESOI 等价性 | 解读 |")
    md.append("|---|---|---|---|---|---|---|---|")
    delta = ("-" if mean is None else
             f"{mean:+.4f}" + (f" ± {std:.4f}" if std is not None else ""))
    md.append(f"| lega vs lega_placebo | {delta} | {ci_s} | {t.get('n', '-')} | "
              f"{_seed_winloss_str(t)} | "
              f"{'-' if p_adj is None else f'{p_adj:.4f}'} | {eq_status} | "
              f"{reading} |")
    if isinstance(capacity_test, dict):
        m2 = _finite_number(capacity_test.get("mean_diff"))
        s2 = _finite_number(capacity_test.get("std_diff"))
        p2 = _finite_number(capacity_test.get("p_adj"))
        ci2 = capacity_test.get("mean_diff_ci95")
        ci2_s = (f"[{ci2[0]:+.4f}, {ci2[1]:+.4f}]"
                 if isinstance(ci2, (list, tuple)) and len(ci2) == 2
                 and all(_finite_number(x) is not None for x in ci2) else "-")
        eq2 = (capacity_test.get("equivalence")
               if isinstance(capacity_test.get("equivalence"), dict) else {})
        if eq2.get("status") == "equivalent":
            read2 = "噪声特征与 qgate 落在等价区：额外容量本身无实质影响"
        elif m2 is not None and p2 is not None and p2 < ALPHA and m2 < 0:
            read2 = ("额外容量/优化自由度单独即可降低 PPL：lega vs placebo 的"
                     "差异必须在该容量红利背景下解读")
        elif m2 is not None and p2 is not None and p2 < ALPHA and m2 > 0:
            read2 = ("噪声特征显著差于 qgate：对照可能“有害”，"
                     "lega > placebo 不能单独证明 B/C 证据有用")
        else:
            read2 = "未定：容量本身的影响方向/显著性不足"
        d2 = ("-" if m2 is None else
              f"{m2:+.4f}" + (f" ± {s2:.4f}" if s2 is not None else ""))
        md.append(f"| lega_placebo vs qgate（容量本身） | {d2} | {ci2_s} | "
                  f"{capacity_test.get('n', '-')} | "
                  f"{_seed_winloss_str(capacity_test)} | "
                  f"{'-' if p2 is None else f'{p2:.4f}'} | "
                  f"{eq2.get('status', '-')} | {read2} |")
    else:
        md.append("| lega_placebo vs qgate（容量本身） | - | - | - | - | - | - | "
                  "未运行/缺少 qgate，无法判断噪声特征本身是中性还是有害 |")
    md.append("")
    md.append("注: `lega_placebo` 的门控形状/参数个数/门控 FLOPs 与 `lega` 完全相同"
              "（w·q + a·u + b·v + d），但 u/v 是逐位置/逐头的确定性伪随机特征，"
              "不含任何查询/注意力/长度信息。二者唯一差别是 B/C 证据是否有信息，"
              "因此该对比把“多几个门控参数的容量/优化红利”从 B/C 证据效应中分离"
              "出来。它是探索性对照（未预注册），不能替代 H2–H5 的确认性检验；"
              "ΔPPL<0 表示 LEGA 更低。")
    return md


def _render_inference_shapley(inference_tests) -> list:
    """渲染同一 checkpoint 内的 2^3 干预 Shapley（与训练期 Shapley 并列）。"""
    rows = [t for t in (inference_tests if isinstance(inference_tests, list)
                        else []) if isinstance(t, dict)]
    md = ["", "### 推理期全因子 Shapley（同一 checkpoint 内干预；探索性）", ""]
    if not rows:
        md.append("⚠ 无法计算推理期 Shapley：需要 `lega` 的 8 个推理期格子"
                  "（正常基线、no_evidence 以及 query_only/conc_only/len_only/"
                  "no_query/no_conc/no_len）全部施加成功、且与基线共享同一批 "
                  "val 窗口。缺格或旧产物缺 `window_indices` 时如实跳过，"
                  "不按 0 补齐。")
        return md
    md.append("训练期全因子消融改变优化轨迹；本表在同一 checkpoint 上只清零门控"
              "项，直接对**训练好的前向**做 2^3 干预并计算 Shapley 值 "
              "`φ_A+φ_B+φ_C=ABC−none`。φ>0 表示该因子在最终前向中降低 PPL，"
              "`exp(−φ)` 为 PPL 几何均值比。与训练期 φ 符号不一致时，说明"
              "“重新训练后的组件贡献结构”没有在最终前向复现，应并列披露。")
    md += ["", "| 因子 | φ(推理期) | PPL几何比 | 占总效应 | n种子 | p_adj(Holm) | "
           "φ(训练期) | 符号一致 | 判定 |",
           "|---|---|---|---|---|---|---|---|---|"]
    for t in rows:
        phi = _finite_number(t.get("mean_diff"))
        ratio = _finite_number(t.get("shapley_ppl_ratio"))
        share = _finite_number(t.get("shapley_share"))
        tr = _finite_number(t.get("training_phi_mean"))
        agree = t.get("training_inference_sign_agree")
        p_adj = _finite_number(t.get("p_adj"))
        p_adj_s = "-" if p_adj is None else f"{p_adj:.4f}"
        md.append(
            f"| {t.get('shapley_factor_label', t.get('shapley_factor'))} | "
            f"{'-' if phi is None else f'{phi:+.4f}'} | "
            f"{'-' if ratio is None else f'{ratio:.4f}'} | "
            f"{'-' if share is None else f'{share*100:.1f}%'} | "
            f"{t.get('n', '-')} | "
            f"{p_adj_s} | "
            f"{'-' if tr is None else f'{tr:+.4f}'} | "
            f"{'-' if agree is None else ('是' if agree else '否 ⚠')} | "
            f"{t.get('verdict', '-')} |")
    md.append("")
    md.append("注: 推理期 φ 由 `query_only/conc_only/len_only`（单因子格）、"
              "`no_query/no_conc/no_len`（两因子格）与 `no_evidence`（无因子格）"
              "同基线共同分解；这些格子只改门控参数、不重新训练，因此是"
              "“最终前向是否使用该组件”的机制证据，不是“重新训练后是否有用”的"
              "训练期证据。推理期与训练期 Shapley 属探索性分析，未预注册。")
    return md


def _render_inference_interactions(inference_tests, training_tests) -> list:
    """渲染同一 checkpoint 内的 2^3 交互（与训练期交互并列）。"""
    rows = [t for t in (inference_tests if isinstance(inference_tests, list)
                        else []) if isinstance(t, dict)]
    md = ["", "### 推理期全因子交互效应（同一 checkpoint 内干预；探索性）", ""]
    if not rows:
        md.append("⚠ 无法计算推理期交互：需要 `lega` 的 8 个推理期格子全部"
                  "施加成功、且与基线共享同一批 val 窗口；缺格或旧产物缺 "
                  "`window_indices` 时如实跳过，不按 0 补齐。")
        return md
    md.append("训练期交互回答“重新训练时组件收益能否按单组件加总”；本表在同一"
              "checkpoint 上对 2^3 格子做同样的 A×B/A×C/B×C 对比，回答"
              "“训练好的前向里交互结构是否仍然存在”。数值为原始 PPL 的"
              "标准交互对比（与训练期 `PPL交互` 同尺度），交互=0 表示可加；"
              "显著非零只说明不能按单组件加总，不套用优/劣方向。")
    train = {}
    for t in (training_tests if isinstance(training_tests, list) else []):
        key = _test_label_key(t.get("interaction")) if isinstance(t, dict) else None
        if key is not None and t.get("metric") == "PPL交互":
            train[key] = t
    md += ["", "| 交互 | ΔPPL均值 ± ΔSD | n种子 | p_adj(Holm) | "
           "训练期ΔPPL | 符号一致 | 判定 |", "|---|---|---|---|---|---|---|"]
    for t in rows:
        tr = _finite_number(t.get("training_interaction_mean"))
        if tr is None:
            key = _test_label_key(t.get("interaction"))
            tr = _finite_number((train.get(key) or {}).get("mean_diff")
                                if key is not None else None)
        ph = _finite_number(t.get("mean_diff"))
        agree = t.get("training_inference_sign_agree")
        if agree is None and tr is not None and ph is not None:
            agree = bool((tr > 0) == (ph > 0) or (tr == 0 and ph == 0))
        p_adj = _finite_number(t.get("p_adj"))
        md.append(
            f"| {t.get('interaction', '-')} | {_fmt_mean_std(t)} | "
            f"{t.get('n', '-')} | "
            f"{'-' if p_adj is None else f'{p_adj:.4f}'} | "
            f"{'-' if tr is None else f'{tr:+.4f}'} | "
            f"{'-' if agree is None else ('是' if agree else '否 ⚠')} | "
            f"{t.get('verdict', '-')} |")
    md.append("")
    md.append("注: 推理期交互与训练期交互都属探索性、未预注册；二者符号不一致"
              "说明“重新训练后是否可加”没有在最终前向复现，应并列披露而不是"
              "只报有利的一期。n<3 或 Holm 校正后不显著只表示未检出，"
              "不等于交互不存在。")
    return md


def _render_ruler_length(analysis) -> list:
    """渲染 RULER 上下文长度剂量-反应（逐 ctx + 斜率 + 首末落差）。"""
    analysis = analysis if isinstance(analysis, dict) else {}
    raw_pairs = analysis.get("pairs")
    rows = ([p for p in raw_pairs if isinstance(p, dict)]
            if isinstance(raw_pairs, list) else [])
    md = ["", "### RULER 上下文长度剂量-反应（探索性；检验“长度校准”是否兑现）", ""]
    if not rows:
        md.append("⚠ 无法做上下文长度分层：缺少同一批种子的 RULER 产物，或"
                  "payload 没有按 ctx 分档的 `results`。只报跨长度平均会把"
                  "“短上下文涨、长上下文跌”与“所有长度小涨”混成同一个数，"
                  "因此不能在没有本表时宣称“长上下文有效”。")
        return md
    md.append("本方法的核心主张是随上下文长度校准门控，因此结果分析必须回答"
              "“优势是否随长度变化”，而不只是报一个跨长度平均。下表把每个"
              "种子的 RULER macro EM 先按**请求的 `ctx_lens` 档位**分档（任务"
              "等权；不同任务的 room_ctx 可能因生成预算被 clamp 到不同值，"
              "按请求档位对齐以免把任务组成变化当成新长度档），再对同一档做"
              "同种子配对差 `Δ=候选−参照`：逐 ctx 表回答哪一档驱动结论；斜率"
              "为逐种子 `ΔEM~log2(ctx)` 的 OLS 系数（长度每翻一倍的 EM 变化，"
              "pp/2×）；长−短为最长与最短档的落差，不假定线性。三者是三个"
              "独立 Holm 家族，全部为探索性、未预注册，不改变 H1–H7 的确认性"
              "判定。")

    def _pp(v):
        # 配对行来自外部/旧版 significance.json：损坏的字符串/NaN/Inf 不能
        # 让整份汇总页在格式化时崩溃，按缺失显示 "-"。
        f = _finite_number(v)
        if f is None:
            return "-"
        scaled = f * 100.0
        # 有限输入乘 100 仍可能溢出（如 1e308），不能写出 +inf。
        return f"{scaled:+.2f}" if math.isfinite(scaled) else "溢出"

    def _padj(t):
        p = _finite_number((t or {}).get("p_adj"))
        return "-" if p is None else f"{p:.4f}"

    def _n_contexts(p):
        n = _integral_value(p.get("n_contexts"))
        return n if n is not None and n >= 0 else 0

    multi = [p for p in rows if _n_contexts(p) >= LENGTH_MIN_CONTEXTS]
    single = [p for p in rows if _n_contexts(p) < LENGTH_MIN_CONTEXTS]
    if single:
        md += ["", f"⚠ {len(single)} 组对比只有 1 个公共请求档位，无法检验"
                   "长度趋势（只能报该档的描述统计）；如需回答长度主张，"
                   "`ruler.ctx_lens` 至少要覆盖 2 档、建议 3 档以上。"]
    if multi:
        md += ["", "| 对比 | 种子 | 请求ctx | 总体Δ(pp) | 最长ctx Δ(pp) | "
                   "长−短(pp) | 斜率(pp/2×) | 方向 正/负/零 | 长度趋势判定 |",
               "|---|---|---|---|---|---|---|---|---|"]
        n_reversal = n_trend = n_unbalanced = n_masking = n_loo = 0
        for p in multi:
            slope = (p.get("slope_per_doubling")
                     if isinstance(p.get("slope_per_doubling"), dict)
                     else {})
            span = (p.get("long_minus_short")
                    if isinstance(p.get("long_minus_short"), dict) else {})
            per_ctx = ([c for c in p.get("per_context")
                        if isinstance(c, dict)]
                       if isinstance(p.get("per_context"), list) else [])
            longest = (per_ctx[-1].get("mean_diff") if per_ctx else None)
            dc = (p.get("direction_consistency")
                  if isinstance(p.get("direction_consistency"), dict) else {})
            flags = []
            if p.get("overall_vs_longest_sign_flip"):
                flags.append("⚠ 总平均与最长ctx反向")
            if p.get("overall_vs_slope_sign_flip"):
                flags.append("⚠ 总平均与斜率反向")
            if p.get("overall_vs_span_sign_flip"):
                flags.append("⚠ 总平均与长−短反向")
            if p.get("long_context_masking"):
                flags.append("⚠ 总平均在SESOI内但最长ctx超出SESOI（平均掩盖长度效应）")
            if p.get("slope_loo_sign_stable") is False:
                flags.append("⚠ 删掉任一单个ctx后斜率符号翻转（趋势依赖个别档位）")
            if p.get("balanced_contexts") is False:
                flags.append("⚠ 各档任务集合不等权（分层均值≠总体分解）")
            n_reversal += int(bool(p.get("overall_vs_longest_sign_flip")))
            n_masking += int(bool(p.get("long_context_masking")))
            n_loo += int(p.get("slope_loo_sign_stable") is False)
            n_unbalanced += int(p.get("balanced_contexts") is False)
            slope_verdict = slope.get("verdict") or "-"
            # 不能按 “显著” 子串计数：``"不显著（Holm ...）"`` 也包含该词。
            # 只把校正后 p<α 的斜率检验算作显著趋势。
            slope_padj = _finite_number(slope.get("p_adj"))
            slope_n = _integral_value(slope.get("n"))
            if (slope_padj is not None and slope_n is not None
                    and slope_n >= MIN_SEEDS and slope_padj < ALPHA):
                n_trend += 1
            verdict = slope_verdict
            if flags:
                verdict += "；" + "；".join(flags)
            contexts = (p.get("contexts")
                        if isinstance(p.get("contexts"), (list, tuple))
                        else [])
            md.append(
                f"| {p.get('candidate')} vs {p.get('reference')} | "
                f"{p.get('n_seeds', '-')} | "
                f"{', '.join(str(c) for c in contexts) if contexts else '-'} | "
                f"{_pp(p.get('overall_mean_diff'))} | {_pp(longest)} | "
                f"{_pp(span.get('mean_diff'))} | "
                f"{_pp(slope.get('mean_diff'))} | "
                f"{dc.get('positive', 0)}/{dc.get('negative', 0)}/"
                f"{dc.get('zero', 0)} | {verdict} |")
        md += ["", f"本次 {len(multi)} 组可分解对比：{n_trend} 组斜率经 Holm 校正"
                   f"后显著，{n_reversal} 组出现“总平均与最长上下文方向相反”，"
                   f"{n_masking} 组“总平均在 ±1pp SESOI 内、最长档却超出”，"
                   f"{n_loo} 组斜率在删掉任一单个 ctx 后符号翻转，"
                   f"{n_unbalanced} 组各档任务集合不等权。方向相反、平均掩盖或"
                   "留一翻转时不能只报总体 RULER 平均；不等权时分层均值只是"
                   "描述，不是总体均值的严格分解。"]

    md += ["", "**逐上下文配对明细（Δ=候选−参照，pp）**", "",
           "| 对比 | ctx | n种子 | 候选EM | 参照EM | Δ(pp) | 95% CI(pp) | "
           "p | p_adj(Holm) | 结论 |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for p in rows:
        per_ctx = ([c for c in p.get("per_context")
                    if isinstance(c, dict)]
                   if isinstance(p.get("per_context"), list) else [])
        for c in per_ctx:
            ci = c.get("mean_diff_ci95")
            if not (isinstance(ci, (list, tuple)) and len(ci) == 2
                    and all(_finite_number(x) is not None for x in ci)):
                ci_s = "-"
            else:
                lo, hi = ci[0] * 100.0, ci[1] * 100.0
                ci_s = (f"[{lo:+.2f}, {hi:+.2f}]"
                        if math.isfinite(lo) and math.isfinite(hi)
                        else "溢出")
            p_raw = _finite_number(c.get("p"))
            md.append(
                f"| {p.get('candidate')} vs {p.get('reference')} | "
                f"{c.get('ctx')} | {c.get('n', '-')} | "
                f"{_fmt4(c.get('candidate_mean'))} | "
                f"{_fmt4(c.get('reference_mean'))} | "
                f"{_pp(c.get('mean_diff'))} | {ci_s} | "
                f"{'-' if p_raw is None else f'{p_raw:.4f}'} | "
                f"{_padj(c)} | {c.get('verdict') or '-'} |")
    md += ["", "注: 逐 ctx 检验只覆盖训练种子随机性；每个格子只有 20~30 个评测"
              "样例，评测抽样误差另见 `ruler_*.json` 的逐格 Wilson CI。斜率/长−短"
              "的替代解释是“长度效应集中在中段”：斜率显著而首末落差≈0、或二者"
              "方向不一致时，不能写成“上下文越长越好/越差”。所有对比的同一"
              f"检验族共用一次 Holm 校正（本次 m={len(analysis.get('ctx_tests') or [])}"
              f"/{len(analysis.get('slope_tests') or [])}"
              f"/{len(analysis.get('span_tests') or [])}，分别对应逐 ctx/斜率/长−短；"
              "保守）；`n<3` 只报描述统计。未检出趋势只说明当前种子数/档位数"
              "下无法检出，不等于“无长度效应”，需对照 JSON 的 `mde`、逐档 "
              "CI 与种子胜负后再解读。完整逐种子"
              "斜率、逐 ctx 差与检验 p 值见 `significance.json` 的 "
              "`ruler_length_pairs` / `ruler_length_tests`。"]
    return md


def _verdict_head(value) -> str:
    """取结论首词（去掉括号解释），用于跨尺度/口径一致性比较。"""
    return str(value if value is not None else "-").split("（")[0]


def _noise_verdict_note(t: dict) -> str:
    """把效应/噪声关系压成一句可渲染的警示（无信息时返回空串）。"""
    if not isinstance(t, dict):
        return ""
    if t.get("within_noise"):
        return "；⚠ |Δ|<ΔSD，效应落在种子/评测噪声内"
    if t.get("below_mde"):
        return "；效应低于当前种子数可检测的 MDE"
    if t.get("loo_sign_stable") is False:
        return "；⚠ 留一法方向翻转：删掉任一单个种子即改变均值符号，结论依赖个别种子"
    return ""


def _paired_comparisons(runs, pairs, metric_name, extractor, lower_better) -> list:
    """对每个 (候选, 参照) 在共同种子上做配对 t 检验（原始 p，未校正）。

    PPL 的 eval.json 可能分别落在 val/test（旧产物或手工重评）上：跨 split
    的“配对差”混入了评测集难度差异，不是同条件对照。此类种子整对剔除，
    并在结果里记录剔除了多少对，避免静默得到虚假的显著性。
    """
    out = []
    for cand, ref in pairs:
        if cand not in runs or ref not in runs:
            continue
        cm = _variant_seed_map(runs, cand)
        rm = _variant_seed_map(runs, ref)
        seeds = [s for s in _sorted_seeds(cm) if s in rm]
        a, b = [], []
        split_skipped = protocol_skipped = cell_skipped = 0
        for s in seeds:
            sa, sb = cm.get(s), rm.get(s)
            if not (isinstance(sa, dict) and isinstance(sb, dict)):
                continue
            if metric_name.startswith("PPL"):
                ea = sa.get("eval") if isinstance(sa.get("eval"), dict) else {}
                eb = sb.get("eval") if isinstance(sb.get("eval"), dict) else {}
                if ea.get("split", "val") != eb.get("split", "val"):
                    split_skipped += 1
                    continue
                pa, pb = _ppl_protocol(ea), _ppl_protocol(eb)
                common = set(pa) & set(pb)
                if any(not _same_protocol_value(pa[k], pb[k])
                       for k in common):
                    protocol_skipped += 1
                    continue
            elif metric_name == "RULER平均" or metric_name.startswith("RULER:"):
                ra = sa.get("ruler") if isinstance(sa.get("ruler"), dict) else {}
                rb = sb.get("ruler") if isinstance(sb.get("ruler"), dict) else {}
                # 同 seed 只保证训练随机性相同；RULER 题面/抽样/生成预算与
                # 干扰语料来源任一不同，EM 就不是同一批样例上的配对观测。
                # 逐任务分解（RULER:<task>）与综合平均必须走同一套可比性门槛，
                # 否则会从不一致的样本数/评测格上得出“逐任务显著”的假结论。
                pa, pb = _ruler_protocol(ra), _ruler_protocol(rb)
                common = set(pa) & set(pb)
                if any(not _same_protocol_value(pa[k], pb[k])
                       for k in common):
                    protocol_skipped += 1
                    continue
                ca = _ruler_cell_signature(ra)
                cb = _ruler_cell_signature(rb)
                if ca is not None and cb is not None and ca != cb:
                    cell_skipped += 1
                    continue
            va, vb = extractor(sa), extractor(sb)
            if va is None or vb is None:
                continue
            a.append(va)
            b.append(vb)
        if not a:
            continue
        res = paired_t_test(a, b)
        change = single_factor_change(cand, ref)
        # 只有审计表登记的六条边才可标成 single_factor。仅凭“对称差=1”会把
        # 未登记的单因子对（例如方向写反、或临时拼出的对比）也标成可归因
        # 证据，绕过 SINGLE_FACTOR_CONTRASTS 这个唯一事实来源。
        audited = (cand, ref) in SINGLE_FACTOR_PAIRS
        is_single = bool(audited and change)
        res.update({"candidate": cand, "reference": ref, "metric": metric_name,
                    "lower_is_better": bool(lower_better),
                    # 单因素标记进入 significance.json：只有单因素对比可用于
                    # 组件归因；vs standard / LEGA vs qgate 等整体对比即使显著
                    # 也不能把差异归给某一个 A/B/C 因子。
                    "contrast_type": ("single_factor" if is_single else "overall"),
                    "changed_factors": ([change] if is_single else [])})
        if split_skipped:
            res["excluded_split_mismatches"] = split_skipped
        if protocol_skipped:
            res["excluded_protocol_mismatches"] = protocol_skipped
        if cell_skipped:
            res["excluded_cell_mismatches"] = cell_skipped
        out.append(_enrich_paired(res, a, b))
    return out


def _adjust_equivalence_family(group, min_seeds=MIN_SEEDS,
                               alpha=ALPHA) -> None:
    """在同一指标家族内对 TOST p 做 Holm 校正（独立于优效性检验族）。

    为什么单独校正：等价性假设（|Δ|<SESOI）与优效性假设（Δ=0）是两类不同
    的推断；把两者混进同一 Holm 家族会双重惩罚。这里只对“有 SESOI 定义且
    n>=min_seeds”的对比计数，n 不足的对比不占家族大小（p 置 None，但仍保留
    原始 TOST p 供描述）。敏感性网格的每个尺度也各自在同一家族内校正，
    否则 0.5×/1×/2× 的“等价/不等价”会受未校正的网格搜索影响。
    """
    eq_tests = [t for t in group if isinstance(t.get("equivalence"), dict)]
    if not eq_tests:
        return

    def _n_ok(obj):
        if not isinstance(obj, dict) or obj.get("not_applicable") is True:
            return False
        n = obj.get("n")
        return (isinstance(n, int) and not isinstance(n, bool)
                and n >= min_seeds)

    pads = holm_bonferroni([
        (t["equivalence"].get("p_tost") if _n_ok(t["equivalence"]) else None)
        for t in eq_tests])
    # 非劣性是单侧假设，必须与 TOST 分开校正：把 TOST 的校正 p 拿来当作
    # 非劣性证据会既漏报（单侧 p 更小）又谎报（校正后不显著仍沿用原始
    # noninferior 标志）。这里对同一家族内的单侧 p 独立做 Holm。
    noninf_pads = holm_bonferroni([
        (t["equivalence"].get("p_noninf_worse")
         if _n_ok(t["equivalence"]) else None)
        for t in eq_tests])
    for t, padj, noninf_adj in zip(eq_tests, pads, noninf_pads):
        eq = t["equivalence"]
        eq["p_tost_adj"] = padj
        eq["p_noninf_worse_adj"] = noninf_adj
        eq["noninferior_adjusted"] = True
        eq["noninferior"] = bool(
            _n_ok(eq) and noninf_adj is not None and noninf_adj < alpha)
        eq["multiplicity_adjusted"] = padj is not None

    n_grid = 0
    for t in eq_tests:
        sens = t["equivalence"].get("sensitivity")
        if isinstance(sens, list):
            n_grid = max(n_grid, len(sens))
    for j in range(n_grid):
        cells = []
        for t in eq_tests:
            sens = t["equivalence"].get("sensitivity")
            if (isinstance(sens, list) and len(sens) > j
                    and isinstance(sens[j], dict)):
                cells.append(sens[j])
        if not cells:
            continue
        adjs = holm_bonferroni([
            (c.get("p_tost") if _n_ok(c) else None) for c in cells])
        for c, padj in zip(cells, adjs):
            c["p_tost_adj"] = padj
            c["multiplicity_adjusted"] = padj is not None
            if c.get("status") == "invalid":
                # 不可表示/损坏的敏感性档（如 2× 绝对 margin 溢出）没有
                # 可检验的 p；不能把它改写成“不等价”，否则会把“无法计算”
                # 冒充成反对等价的证据。
                c["equivalent"] = False
                c["status"] = "invalid"
                continue
            c["equivalent"] = bool(
                _n_ok(c) and padj is not None and padj < alpha)
            c["status"] = ("equivalent" if c["equivalent"]
                           else "insufficient_seeds" if not _n_ok(c)
                           else "not_equivalent")


def _apply_holm(tests, min_seeds=MIN_SEEDS, alpha=ALPHA) -> list:
    """对同一指标家族内的全部配对检验做 Holm–Bonferroni 校正并给出结论。

    汇总同时检验几十组对比；不校正时 FWER 随检验数上升，名义 p<0.05 极易由
    运气产生。结论一律基于同一指标家族内校正后的 p_adj；原始 p 仍保留在
    结果 JSON 中供查阅。

    置换检验的 p_perm 与符号检验的 p_sign 各自在同一指标家族内单独做 Holm
    校正（它们与 t 检验是独立的检验族，合并会双重惩罚）：t 与置换结论是否
    一致被显式记录为 t_perm_agree；t 显著而置换不显著（小样本下 t 的正态
    假设失效的典型症状）时在结论里追加警示，而不是让读者只看到一个 p 值。
    符号检验只作稳健性对照，不参与确认性判定（nz<6 时其双侧精确 p 的
    下限 2/2^nz>0.05，不显著可能只是粒度所限）。
    """
    tests = [t for t in (tests if isinstance(tests, list) else [])
             if isinstance(t, dict)]
    by_metric = {}
    for t in tests:
        by_metric.setdefault(t.get("metric"), []).append(t)
    for group in by_metric.values():
        _adjust_equivalence_family(group, min_seeds=min_seeds, alpha=alpha)
        pads = holm_bonferroni([t.get("p") for t in group])
        ppads = holm_bonferroni([t.get("p_perm") for t in group])
        psads = holm_bonferroni([t.get("p_sign") for t in group])
        for t, padj, ppadj, psadj in zip(group, pads, ppads, psads):
            t["p_adj"] = padj
            t["p_perm_adj"] = ppadj
            t["p_sign_adj"] = psadj
            n = t.get("n")
            n_valid = (isinstance(n, int) and not isinstance(n, bool)
                       and n >= 0)
            p_norm = _finite_number(t.get("p"))
            p_valid = (p_norm is not None
                       and 0.0 <= p_norm <= 1.0)
            if t.get("p") is None:
                t["verdict"] = f"无法检验（n={n} 或方差退化），不作判定"
            elif not p_valid:
                # holm_bonferroni 会把 NaN/Inf/bool/越界/不可比较的 p 归一化
                # 为 None；此时不能顺着“p_adj=None”落到“不显著”，否则损坏
                # 的 p 值会被报告成“没有效应”。
                t["verdict"] = (
                    f"无法检验（p 值非法或不可解析 p={t.get('p')!r}），"
                    "不作判定")
            elif not n_valid:
                # 汇总文件可能来自旧版本/手工编辑：缺 n 不能让整个报告
                # KeyError 崩溃，也不能默认满足最小种子数而给出显著性结论。
                t["verdict"] = f"无法检验（样本量缺失或非法 n={n!r}），不作判定"
            elif n < min_seeds:
                t["verdict"] = f"种子不足（n={n}<{min_seeds}），无法判定"
            elif t.get("no_direction"):
                if t.get("p_adj") is not None and t["p_adj"] < alpha:
                    t["verdict"] = ("显著偏离可加性（Holm 校正后；"
                                    "方向见 Δ 符号）")
                    if ppadj is not None and ppadj >= alpha:
                        t["verdict"] += "；⚠ 置换检验不显著（见注）"
                else:
                    t["verdict"] = ("未检出显著交互（Holm 校正后 p="
                                    f"{t['p_adj']:.4f}）"
                                    if t.get("p_adj") is not None
                                    else "未检出显著交互")
            elif t.get("p_adj") is not None and t["p_adj"] < alpha:
                better = ((t["mean_diff"] < 0) if t["lower_is_better"]
                          else (t["mean_diff"] > 0))
                t["verdict"] = ("显著优于（Holm 校正后）" if better
                                else "显著劣于（Holm 校正后）")
                if ppadj is not None and ppadj >= alpha:
                    t["verdict"] += "；⚠ 置换检验不显著（见注）"
            else:
                t["verdict"] = ("不显著（Holm 校正后 p="
                                f"{t['p_adj']:.4f}）"
                                if t.get("p_adj") is not None else "不显著")
                if ppadj is not None and ppadj < alpha:
                    t["verdict"] += "；置换检验显著（边缘证据，建议补种子确认）"
            t["t_perm_agree"] = (
                None if (t.get("p_adj") is None or ppadj is None
                        or not n_valid or n < min_seeds)
                else (t["p_adj"] < alpha) == (ppadj < alpha))
            note = _noise_verdict_note(t)
            if note and isinstance(t.get("verdict"), str):
                t["verdict"] += note
            # 等价性 status 依赖优效性 p_adj：必须在 p_adj 写入后重算。
            eq = t.get("equivalence")
            if isinstance(eq, dict):
                eq.update(interpret_equivalence(
                    eq, t.get("p_adj"), alpha=alpha, min_seeds=min_seeds))
    return tests


# 组件有效性判定所用的检验对：组件 -> (去一法检验, 全因子检验, 主指标)。
# 去一法: Full−X vs Full；全因子: 仅 X vs gate_only（已有门但无 A/B/C）。
# 主指标按 preregistration/hypotheses.json 的登记口径固定：A/B 在语言建模
# PPL，C 在长上下文 RULER。另一指标只作探索性补充，不能用来“换个指标再
# 判一次”以规避多重比较（两族分别 Holm 后取 OR 会放大假阳性）。
COMPONENT_TESTS = (
    ("A 查询项", ("lega_no_query", "lega"), ("qgate", "gate_only"), "PPL"),
    ("B 集中度证据", ("lega_no_conc", "lega"), ("lega_conc_only", "gate_only"), "PPL"),
    ("C 长度校准", ("lega_no_len", "lega"), ("lega_len_only", "gate_only"), "RULER平均"),
)
_EVIDENCE_METRICS = ("PPL", "RULER平均")


def _find_test(tests, pair, metric):
    for t in (tests if isinstance(tests, list) else []):
        if (isinstance(t, dict)
                and (t.get("candidate"), t.get("reference")) == pair
                and t.get("metric") == metric):
            return t
    return None


def _effect_status(t, removing: bool) -> dict:
    """单条配对检验 -> “组件有用”的证据状态（须方向正确、显著且稳健）。

    removing=True（去一法）：候选=去掉组件的 Full−X，显著变差才算证据；
    removing=False（全因子）：候选=仅含该组件的 X，显著变好才算证据。
    opposite=True 表示检验显著但方向相反——这是“去掉反而更好/加入反而
    更差”的反证，不能被同一组件的另一套消融静默抵消。

    显著不等于稳健：若 |Δ|<ΔSD（效应落在逐种子配对差噪声内）或删掉任一
    单个种子就翻转均值符号，即使 Holm 校正后显著也标 fragile，不算
    “组件有用”的证据，更不能进入 confirmatory_supported。
    """
    if t is None:
        return {"supported": False, "opposite": False, "fragile": False,
                "reason": "无对应检验（变体未运行）", "p_adj": None}
    n = _integral_value(t.get("n"))
    if n is None:
        return {"supported": False, "opposite": False, "fragile": False,
                "reason": f"种子数缺失或非法（n={t.get('n')!r}）",
                "p_adj": t.get("p_adj")}
    if n < MIN_SEEDS:
        return {"supported": False, "opposite": False, "fragile": False,
                "reason": f"种子不足（n={n}）", "p_adj": t.get("p_adj")}
    if (t.get("mean_diff") is None or t.get("p_adj") is None
            or t.get("p_adj") >= ALPHA):
        reason = "Holm 校正后不显著"
        if t.get("within_noise"):
            reason += "（效应在种子噪声内）"
        elif t.get("below_mde"):
            reason += "（效应低于 MDE）"
        return {"supported": False, "opposite": False, "fragile": False,
                "reason": reason, "p_adj": t.get("p_adj")}
    lb = bool(t.get("lower_is_better"))
    if removing:
        evidence = (t["mean_diff"] > 0) if lb else (t["mean_diff"] < 0)
    else:
        evidence = (t["mean_diff"] < 0) if lb else (t["mean_diff"] > 0)
    if not evidence:
        return {"supported": False, "opposite": True, "fragile": False,
                "reason": "方向相反（去掉反而更好/加入反而更差）",
                "p_adj": t.get("p_adj")}
    fragile_reasons = []
    if t.get("within_noise"):
        fragile_reasons.append("|Δ|<ΔSD（种子噪声内）")
    if t.get("loo_sign_stable") is False:
        fragile_reasons.append("留一方向翻转")
    if fragile_reasons:
        return {"supported": False, "opposite": False, "fragile": True,
                "reason": "显著但不稳健（" + "；".join(fragile_reasons) + "）",
                "p_adj": t.get("p_adj")}
    return {"supported": True, "opposite": False, "fragile": False,
            "reason": "显著且稳健", "p_adj": t.get("p_adj")}


def _combine_component_evidence(loo: dict, add: dict) -> dict:
    """合并去一法与全因子两套消融证据，显式处理方向冲突与稳健性。

    规则（与汇总页注释一致）：
      * 两套均稳健支持 -> 有证据支持（一致）；
      * 仅一套稳健支持、另一套不显著 -> 有证据支持（另一套未检出，可能功效不足）；
      * 仅一套稳健支持、另一套显著但不稳健 -> 仍有证据，但标 fragile，
        不能作为确认性结论（confirmatory_supported=False）；
      * 任一套显著反向 -> 证据冲突（即使另一套支持也不能宣称有用）；
      * 都无显著证据；或仅有显著但不稳健的证据 -> 证据不足/不稳健。
    """
    # 任一预注册主指标上的显著反向本身就是反证；另一套“未检出”不能把
    # 它稀释成“证据不足”。只有两套都不显著/都不支持时才是证据不足。
    if loo.get("opposite") or add.get("opposite"):
        conflict = True
    else:
        conflict = False
    if conflict:
        return {"has_evidence": False, "consistent": False, "fragile": False,
                "verdict": "证据冲突（一套消融显著反向）"}
    if loo.get("supported") and add.get("supported"):
        return {"has_evidence": True, "consistent": True, "fragile": False,
                "verdict": "有证据支持（两套消融一致且稳健）"}
    if loo.get("supported"):
        if add.get("fragile"):
            return {"has_evidence": True, "consistent": False,
                    "fragile": True,
                    "verdict": "有证据支持（仅去一法稳健；全因子显著但"
                               "落在噪声内/留一翻转）"}
        return {"has_evidence": True, "consistent": False, "fragile": False,
                "verdict": "有证据支持（仅去一法；全因子未检出）"}
    if add.get("supported"):
        if loo.get("fragile"):
            return {"has_evidence": True, "consistent": False,
                    "fragile": True,
                    "verdict": "有证据支持（仅全因子稳健；去一法显著但"
                               "落在噪声内/留一翻转）"}
        return {"has_evidence": True, "consistent": False, "fragile": False,
                "verdict": "有证据支持（仅全因子；去一法未检出）"}
    if loo.get("fragile") or add.get("fragile"):
        return {"has_evidence": False, "consistent": False, "fragile": True,
                "verdict": "证据不稳健（显著但落在种子噪声内/留一方向翻转）"}
    return {"has_evidence": False, "consistent": False, "fragile": False,
            "verdict": "证据不足"}


def component_evidence(tests, audit: dict = None) -> list:
    """由配对检验结果汇总各组件（A/B/C）是否“有用”的统计证据。

    统计口径：
      * 判定只用预注册主指标（A/B=PPL，C=RULER），另一指标作为探索性
        补充展示——两族分别 Holm 后再取 OR 等于给同一主张两次显著性机会；
      * 去一法（Full−X 显著变差）与全因子（X-only 显著优于 gate_only）
        两套证据合并且检测方向冲突；gate_only 只含门控偏置 d、不含
        A/B/C，避免把“输出门本身”算作组件收益；
      * 任一套显著反向时判“证据冲突”，不再宣称该组件有用；
      * 显著但 |Δ|<ΔSD 或留一方向翻转的证据标为 fragile：最多只能算
        探索性线索，不能升级为“组件有用”的确认性结论。
      * 早停导致实际训练量不等时，统计检验本身仍可计算，但“组件有用”
        不再具备确认性：每条证据同时带 training_budget_comparable /
        confirmatory_supported / evidence_status，供机器消费时不会把
        探索性结果误读成确认结论。
    """
    audit = _as_dict(audit)
    stopped = _as_text_list(audit.get("early_stop_runs"))
    budget_ok = not stopped
    out = []
    for name, loo_pair, add_pair, primary in COMPONENT_TESTS:
        item = {"component": name, "loo_pair": list(loo_pair),
                "add_pair": list(add_pair), "primary_metric": primary,
                "loo": {}, "add": {}, "supported": [],
                "training_budget_comparable": budget_ok,
                "fragile": False,
                "confirmatory_supported": False,
                "evidence_status": "none"}
        if stopped:
            item["budget_note"] = (
                "早停使实际训练量不等（" + "；".join(stopped[:4])
                + (" ..." if len(stopped) > 4 else "")
                + "）：组件证据仅供探索，不能作为确认性结论")
            item["early_stop_runs"] = stopped
        for metric in _EVIDENCE_METRICS:
            item["loo"][metric] = _effect_status(
                _find_test(tests, loo_pair, metric), True)
            item["add"][metric] = _effect_status(
                _find_test(tests, add_pair, metric), False)
        loo_p = item["loo"][primary]
        add_p = item["add"][primary]
        combined = _combine_component_evidence(loo_p, add_p)
        if combined["has_evidence"]:
            item["supported"].append(primary)
        item.update(combined)
        item["confirmatory_supported"] = bool(
            combined["has_evidence"] and budget_ok
            and not combined.get("fragile"))
        if item["confirmatory_supported"]:
            item["evidence_status"] = "confirmatory"
        elif combined["has_evidence"] or combined.get("fragile") or not budget_ok:
            item["evidence_status"] = "exploratory"
        item["evidence"] = ({primary: {
            "direction": ("loo" if loo_p["supported"] else "add"),
            "p_adj": (loo_p if loo_p["supported"] else add_p).get("p_adj")}}
            if combined["has_evidence"] else {})
        item["secondary"] = {
            metric: {"exploratory": True,
                     "loo_supported": item["loo"][metric]["supported"],
                     "add_supported": item["add"][metric]["supported"]}
            for metric in _EVIDENCE_METRICS if metric != primary}
        out.append(item)
    return out


# 机制主张的检验族：门控完整 logit 与仅查询项的 AUROC / PR-AUC 之差
# （越大越好）。两族必须分别检验：hit 判别通常类别极不平衡，AUROC 会因
# 负样本基数巨大而乐观，PR-AUC 是此时更贴近研究目标的口径。
AUROC_DIFF_METRIC = "AUROC差(logit−仅查询)"
AP_DIFF_METRIC = "AP差(logit−仅查询)"
AUROC_DIFF_REFERENCE = "仅查询项基线"


def _detection_diff_tests(runs, key_cand: str, metric: str) -> list:
    """同一 run 内 key_cand(logit) − key_cand(qterm) 的跨种子配对检验。

    “完整门控 logit 比仅查询项更能判断该不该写入”是本方法的核心机制
    主张，但把两个并排的数放在表里不构成证据：它们来自同一批记录、同一
    批样例，必须作为配对差值检验。每个 (变体, 种子) run 的差值当作一次
    观测（H0：差为 0），单一侧缺失的种子整对剔除，n 如实反映有效配对数。
    全家族与 PPL/RULER 一样走 Holm 校正。逐 run 的样例聚类 bootstrap
    区间（analysis JSON 的 auroc_cluster_bootstrap）与本跨种子检验互为
    印证：前者回答“换样例波动多少”，后者回答“换训练种子方向是否稳定”。
    """
    out = []
    runs = runs if isinstance(runs, dict) else {}
    for v in sorted(runs):
        if v not in VARIANTS:
            continue
        m = _variant_seed_map(runs, v)
        seeds = _sorted_seeds(m)
        if not seeds:
            continue
        a, b = [], []
        for s in seeds:
            slot = m.get(s)
            an = slot.get("analysis") if isinstance(slot, dict) else None
            if not isinstance(an, dict):
                continue
            al = _rate_or_none(an.get(key_cand + "_logit"))
            aq = _rate_or_none(an.get(key_cand + "_qterm"))
            if al is None or aq is None:
                continue
            a.append(al)
            b.append(aq)
        if not a:
            continue
        res = paired_t_test(a, b)
        res.update({"candidate": v, "reference": AUROC_DIFF_REFERENCE,
                    "metric": metric, "lower_is_better": False})
        out.append(_enrich_paired(res, a, b))
    return out


# 全因子 2^3 单元格到变体的映射（用于交互效应检验）。
# “none”用 gate_only 而不是 standard：8 个 gated 单元格共享输出门，
# 交互对比必须在同一个门内比较 A/B/C；用无门的 standard 会把门本身
# 重新混进交互项（正是 gate_only 要隔离的混杂）。
_FACTORIAL_CONTRAST_CELLS = {
    "none": "gate_only", "A": "qgate", "B": "lega_conc_only",
    "C": "lega_len_only", "AB": "lega_no_len", "AC": "lega_no_conc",
    "BC": "lega_no_query", "ABC": "lega",
}
# 交互对比式（对第三个因子的两个水平取平均）：
#   I_AB = ½[(AB−A−B+none) + (ABC−AC−BC+C)]
#   I_AC = ½[(AC−A−C+none) + (ABC−AB−BC+B)]
#   I_BC = ½[(BC−B−C+none) + (ABC−AB−AC+A)]
_FACTORIAL_CONTRASTS = {
    "A×B": (("AB", "A", "B", "none"), ("ABC", "AC", "BC", "C")),
    "A×C": (("AC", "A", "C", "none"), ("ABC", "AB", "BC", "B")),
    "B×C": (("BC", "B", "C", "none"), ("ABC", "AB", "AC", "A")),
}


def _stable_factorial_interaction(cells):
    values = [_finite_number(v) for v in cells]
    if any(v is None for v in values):
        return None
    scale = max(abs(v) for v in values)
    if scale == 0.0:
        return 0.0
    total = math.fsum(
        sign * (v / scale)
        for v, sign in zip(values, (1, -1, -1, 1, 1, -1, -1, 1)))
    out = scale * (0.5 * total)
    return out if math.isfinite(out) else None


def _factorial_protocol_ok(maps, seed, metric) -> bool:
    """一个种子的 8 个全因子格是否共享同一评测协议。

    交互对比同时用到全部 8 个格；只要任一格的 split/评测窗口或 RULER
    题面/生成预算/已评测单元不同，整条交互线就混入了非受控差异。判定
    口径与 `_paired_comparisons` 一致：只比较双方都记录的字段，缺失
    字段不伪造。
    """
    slots = []
    for cell in _FACTORIAL_CONTRAST_CELLS:
        m = maps.get(cell) if isinstance(maps, dict) else None
        slot = m.get(seed) if isinstance(m, dict) else None
        if not isinstance(slot, dict):
            return False
        slots.append(slot)
    if metric == "PPL":
        evals = [s.get("eval") if isinstance(s.get("eval"), dict) else {}
                 for s in slots]
        # 不能把 split 原值直接放进 set：外部/旧版 eval.json 的 split 可能
        # 是数组/对象（不可哈希），会让整份汇总在交互检验处 TypeError。
        # 用与协议核对相同的规范化比较，既保留“缺失=None 与 None 相同”
        # 的旧口径，也能安全处理任意 JSON 值。
        splits = [e.get("split", "val") for e in evals]
        if any(not _same_protocol_value(splits[0], x)
               for x in splits[1:]):
            return False
        protocols = [_ppl_protocol(e) for e in evals]
    else:
        protocols = [_ruler_protocol(
            s.get("ruler") if isinstance(s.get("ruler"), dict) else {})
            for s in slots]
        sigs = [_ruler_cell_signature(
            s.get("ruler") if isinstance(s.get("ruler"), dict) else {})
            for s in slots]
        known = [sig for sig in sigs if sig is not None]
        if len(set(known)) > 1:
            return False
    common = (set.intersection(*(set(p) for p in protocols))
              if protocols else set())
    ref = protocols[0] if protocols else {}
    return not any(not _same_protocol_value(ref[k], p[k])
                   for p in protocols[1:] for k in common)


def _factorial_interaction_tests(runs) -> list:
    """全因子 2^3 设计的交互效应检验（探索性，独立指标家族）。

    组件主效应“有没有用”不等于“效应可加”：A 的收益可能在已有 B 时缩小
    （次可加）或放大（协同）。逐种子计算标准 2^3 交互对比（对第三个因子
    取平均），再对该对比做单样本配对 t 检验（H0：交互=0）。交互检验没有
    预注册方向，因此只报“是否显著偏离可加性”与 Δ 符号，不套用“优/劣”。
    基准格是 gate_only（有门但无 A/B/C），不是无门的 standard。
    """
    out = []
    runs = runs if isinstance(runs, dict) else {}
    for metric, extractor in (
            ("PPL", lambda slot: _ppl_of(slot.get("eval"))),
            ("RULER平均", lambda slot: _ruler_avg_of(slot.get("ruler")))):
        maps = {cell: _variant_seed_map(runs, v)
                for cell, v in _FACTORIAL_CONTRAST_CELLS.items()}
        if any(not m for m in maps.values()):
            continue
        all_seeds = _sorted_seeds(maps["none"])
        seeds = [s for s in all_seeds
                 if _factorial_protocol_ok(maps, s, metric)]
        excluded = len(all_seeds) - len(seeds)
        for pair, (slice_a, slice_b) in _FACTORIAL_CONTRASTS.items():
            vals = []
            for s_ in seeds:
                value = _stable_factorial_interaction(
                    [extractor(maps[cell].get(s_))
                     for cell in slice_a + slice_b])
                if value is not None:
                    vals.append(value)
            if not vals:
                continue
            res = paired_t_test(vals, [0.0] * len(vals))
            res.update({"candidate": f"{pair} 交互", "reference": "可加性（交互=0）",
                        "metric": metric + "交互", "lower_is_better": True,
                        "no_direction": True, "interaction": pair})
            if excluded:
                res["excluded_protocol_mismatches"] = excluded
            out.append(_enrich_paired(res, vals, [0.0] * len(vals)))
    return out


# 全因子 Shapley 分解（探索性）：与交互检验同源，但用全部 8 格把 Full−none
# 的总效应按所有加入顺序平均分摊到 A/B/C，避免单因素边对参照格的依赖。
SHAPLEY_PPL_METRIC = "Shapley(PPL)"
SHAPLEY_RULER_METRIC = "Shapley(RULER平均)"
SHAPLEY_FACTOR_LABEL = {"A": "A 查询项", "B": "B 集中度证据",
                        "C": "C 长度校准"}


def _factorial_shapley_tests(runs) -> list:
    """全因子 2^3 的 Shapley 效应分解 + 跨种子检验（探索性）。

    价值函数：PPL 用 v=−log(PPL)（比值尺度上可加，exp(−φ) 即 PPL 几何
    均值比）；RULER 用原始平均 EM（越大越好）。逐种子计算精确 Shapley
    值（3 个因子、8 个格），再对每个因子的 φ 做单样本配对 t 检验
    （H0：φ=0）。同一指标的三因子共用同一 Holm 家族；8 格必须全部运行、
    种子数≥MIN_SEEDS、协议一致，否则如实跳过/标注。

    这不是预注册的确认性检验：Shapley 值依赖价值函数尺度，且把交互
    效应平均分摊后不再对应某个“单因素边”。它回答的是“在完整设计上，
    该因子平均贡献多少”，与去一法/全因子增量并列，不能替代它们。
    """
    out = []
    runs = runs if isinstance(runs, dict) else {}
    for metric, extractor, metric_name in (
            ("PPL", lambda slot: _ppl_of(slot.get("eval")),
             SHAPLEY_PPL_METRIC),
            ("RULER平均", lambda slot: _ruler_avg_of(slot.get("ruler")),
             SHAPLEY_RULER_METRIC)):
        maps = {cell: _variant_seed_map(runs, v)
                for cell, v in _FACTORIAL_CONTRAST_CELLS.items()}
        if any(not m for m in maps.values()):
            continue
        all_seeds = _sorted_seeds(maps["none"])
        seeds = [s for s in all_seeds
                 if _factorial_protocol_ok(maps, s, metric)]
        excluded = len(all_seeds) - len(seeds)
        per_seed = {}
        for s in seeds:
            raw = {}
            ok = True
            for cell in FACTORIAL_CELL_ORDER:
                slot = maps.get(cell, {}).get(s)
                v = extractor(slot) if isinstance(slot, dict) else None
                if v is None:
                    ok = False
                    break
                # PPL=0 是历史夹具的兼容边界值，但 log 分解没有定义；
                # 该种子整条剔除，不用 0/极小值伪造 −log 值。
                if metric == "PPL" and (_finite_number(v) is None
                                       or float(v) <= 0):
                    ok = False
                    break
                raw[cell] = v
            if not ok:
                continue
            try:
                oriented = {c: factorial_oriented_value(metric, raw[c])
                            for c in FACTORIAL_CELL_ORDER}
                dec = factorial_decomposition(oriented)
            except (TypeError, ValueError):
                continue
            per_seed[s] = dec
        if not per_seed:
            continue
        ordered = _sorted_seeds(per_seed)
        totals = [per_seed[s]["total_effect"] for s in ordered]
        total_mean = (float(np.mean(totals)) if totals else None)
        # 占比只在每个种子的总效应都为正时可解释；否则不强行归一化。
        share_ok = bool(totals) and all(float(v) > 1e-12 for v in totals)
        for factor in ("A", "B", "C"):
            vals = [per_seed[s]["phi"].get(factor) for s in ordered]
            vals = [v for v in vals if _finite_number(v) is not None]
            if not vals:
                continue
            res = paired_t_test(vals, [0.0] * len(vals))
            res.update({
                "candidate": f"{factor} 的 Shapley 贡献",
                "reference": "0（无贡献）",
                "metric": metric_name,
                "lower_is_better": False,
                "shapley_factor": factor,
                "shapley_factor_label": SHAPLEY_FACTOR_LABEL[factor],
                "shapley_metric": metric,
                "shapley_total_effect_mean": total_mean,
                "shapley_total_effect_std": (
                    float(np.std(totals, ddof=1)) if len(totals) >= 2 else None),
                "shapley_share": (
                    float(np.mean([per_seed[s]["phi"][factor]
                                   / per_seed[s]["total_effect"]
                                   for s in ordered])) if share_ok else None),
                "shapley_efficiency_gap_max": (
                    max(abs(per_seed[s]["efficiency_gap"])
                        for s in ordered) if ordered else None),
                "excluded_protocol_mismatches": excluded,
                "n_protocol_seeds": len(ordered),
                "exploratory": True,
            })
            if metric == "PPL":
                mean_phi = _finite_number(res.get("mean_diff"))
                res["shapley_ppl_ratio"] = (
                    math.exp(max(-20.0, min(20.0, -float(mean_phi))))
                    if mean_phi is not None else None)
                res["shapley_ppl_ratio_note"] = (
                    "exp(−φ)：<1 表示该因子平均降低 PPL；φ 为 −log PPL "
                    "单位的 Shapley 值")
            out.append(_enrich_paired(res, vals, [0.0] * len(vals)))
    return _apply_holm(out)


# 推理期全因子 Shapley：同一 checkpoint 上干预 2^3 个门控格，而不是重新
# 训练。它回答“训练好的前向里，组件的平均贡献与交互结构是否与训练期
# 全因子消融一致”，不能替代训练期确认性检验。
INFERENCE_SHAPLEY_METRIC = "推理期Shapley(PPL)"
INFERENCE_INTERACTION_METRIC = "推理期交互PPL"


def _inference_factorial_values(ablations):
    """校验并提取 `lega` 每个种子的推理期 2^3 八格 PPL。

    返回 ``(per_seed, n_excluded)``：``per_seed[seed]`` 只在
    ABC 基线、7 个病灶全部真正施加、且每个病灶与基线共享同一批
    ``window_indices`` 时才建立；缺格/窗口不一致/非正 PPL 的种子整条剔除，
    不按 0 补齐，也不把“没做干预”伪装成某个单元格的观测。非 `lega` 变体
    不计入排除数：它们的 2^3 本来就不完整，不属于“损坏的 lega 证据”。

    推理期八个格子只有在 Full 模型上才同时具备 A/B/C 三个真实因子；
    qgate/gate_only 等变体缺项，清零会得到大量 no-op 格，Shapley/交互值
    退化为“只有存在的因子”，与训练期全因子不可比，故一律不消费。
    """
    per_seed = {}
    excluded = 0
    ablations = ablations if isinstance(ablations, dict) else {}
    for variant in sorted(ablations):
        if variant != "lega":
            continue
        seed_map = ablations.get(variant)
        if not isinstance(seed_map, dict):
            continue
        for seed in _sorted_seeds(seed_map):
            payload = seed_map.get(seed)
            if not isinstance(payload, dict):
                excluded += 1
                continue
            baseline = payload.get("baseline")
            if not isinstance(baseline, dict):
                excluded += 1
                continue
            base_ppl = _finite_number(baseline.get("val_ppl"))
            if base_ppl is None or base_ppl <= 0:
                excluded += 1
                continue
            base_windows = payload.get("window_indices")
            if not isinstance(base_windows, list) or not base_windows:
                excluded += 1
                continue
            recs = payload.get("lesions")
            recs = recs if isinstance(recs, list) else []
            # 外部/旧产物可能把 lesion 写成数组/对象；不可哈希的键会直接让
            # dict 构造失败，必须逐条校验而不是相信 JSON 结构。
            by_id = {}
            for r in recs:
                if not isinstance(r, dict):
                    continue
                lid = r.get("lesion")
                if isinstance(lid, str):
                    by_id[lid] = r
            raw = {"ABC": base_ppl}
            ok = True
            for cell, lesion_id in INFERENCE_FACTORIAL_CELLS.items():
                if lesion_id is None:
                    continue
                rec = by_id.get(lesion_id)
                if not isinstance(rec, dict):
                    ok = False
                    break
                # 只接受真正施加成功、且与基线同窗口的病灶。
                if rec.get("applied") is not True:
                    ok = False
                    break
                if rec.get("window_indices") != base_windows:
                    ok = False
                    break
                ppl = _finite_number(rec.get("val_ppl"))
                if ppl is None or ppl <= 0:
                    ok = False
                    break
                raw[cell] = ppl
            if not ok or len(raw) != len(FACTORIAL_CELL_ORDER):
                excluded += 1
                continue
            per_seed[seed] = raw
    return per_seed, excluded


def _inference_shapley_tests(ablations):
    """由推理期 2^3 病灶格计算逐种子 Shapley 值并做跨种子检验。

    每个种子必须齐全 8 个格（ABC=正常基线，none=A/B/C 全关，其余为
    只保留一个/两个因子的病灶），且每个病灶与基线共享同一批 val 窗口。
    缺格、未真正施加、或窗口不一致的种子整条剔除，不按 0 补齐。

    价值函数与训练期一致：v = −log(PPL)（越大越好），因此 φ>0 表示该
    因子在训练好的前向里降低 PPL；exp(−φ) 为该因子的 PPL 几何均值比。
    """
    out = []
    raw_by_seed, excluded = _inference_factorial_values(ablations)
    per_seed = {}
    for seed in sorted(raw_by_seed):
        raw = raw_by_seed[seed]
        try:
            oriented = {
                c: factorial_oriented_value("PPL", raw[c])
                for c in FACTORIAL_CELL_ORDER}
            per_seed[seed] = factorial_decomposition(oriented)
        except (TypeError, ValueError):
            excluded += 1
    if per_seed:
        ordered = _sorted_seeds(per_seed)
        totals = [per_seed[s]["total_effect"] for s in ordered]
        total_mean = float(np.mean(totals)) if totals else None
        share_ok = bool(totals) and all(float(v) > 1e-12 for v in totals)
        for factor in ("A", "B", "C"):
            vals = [per_seed[s]["phi"].get(factor) for s in ordered]
            vals = [v for v in vals if _finite_number(v) is not None]
            if not vals:
                continue
            res = paired_t_test(vals, [0.0] * len(vals))
            res.update({
                "candidate": f"{factor} 的推理期 Shapley 贡献",
                "reference": "0（无贡献）",
                "metric": INFERENCE_SHAPLEY_METRIC,
                "lower_is_better": False,
                "variant": "lega",
                "shapley_factor": factor,
                "shapley_factor_label": SHAPLEY_FACTOR_LABEL[factor],
                "shapley_metric": "PPL",
                "inference_time": True,
                "shapley_total_effect_mean": total_mean,
                "shapley_total_effect_std": (
                    float(np.std(totals, ddof=1)) if len(totals) >= 2 else None),
                "shapley_share": (
                    float(np.mean([per_seed[s]["phi"][factor]
                                   / per_seed[s]["total_effect"]
                                   for s in ordered])) if share_ok else None),
                "shapley_efficiency_gap_max": (
                    max(abs(per_seed[s]["efficiency_gap"])
                        for s in ordered)),
                "n_protocol_seeds": len(ordered),
                "excluded_protocol_mismatches": excluded,
                "exploratory": True,
            })
            mean_phi = _finite_number(res.get("mean_diff"))
            res["shapley_ppl_ratio"] = (
                math.exp(max(-20.0, min(20.0, -float(mean_phi))))
                if mean_phi is not None else None)
            res["shapley_ppl_ratio_note"] = (
                "exp(−φ)：<1 表示该因子在训练好的前向里平均降低 PPL")
            out.append(_enrich_paired(res, vals, [0.0] * len(vals)))
    return _apply_holm(out)


def _inference_interaction_tests(ablations):
    """推理期 2^3 干预的 A×B/A×C/B×C 交互检验（探索性，独立 Holm 家族）。

    训练期交互检验回答“重新训练时组件收益能否按单组件加总”；这里在同一
    checkpoint 的 8 个推理期格子上回答“训练好的前向里交互结构是否还
    存在”。使用与训练期相同的标准 2^3 对比（对第三个因子取平均）：

        I_AB = ½[(AB−A−B+none) + (ABC−AC−BC+C)]

    数值仍用原始 PPL（与训练期 `PPL交互` 同尺度，便于逐交互对照符号），
    不套用方向判定：交互=0 表示可加，显著非零只说明“不能按单组件加总”。
    与 Shapley 一样只消费 `lega` 的完整 8 格；缺格/未施加/窗口不一致的
    种子整条剔除，不按 0 补齐。
    """
    out = []
    raw_by_seed, excluded = _inference_factorial_values(ablations)
    if not raw_by_seed:
        return out
    ordered = _sorted_seeds(raw_by_seed)
    for pair, (slice_a, slice_b) in _FACTORIAL_CONTRASTS.items():
        # 协议缺格/错窗对每个交互都成立；但某个交互值本身溢出 float
        # 只影响该交互，不能把它的排除数累加到后续交互的 n 上——否则
        # A×C/B×C 会报告一个与自己样本无关的排除数，读者会误以为它们
        # 也丢了同样的种子。
        pair_excluded = excluded
        vals = []
        for seed in ordered:
            raw = raw_by_seed[seed]
            value = _stable_factorial_interaction(
                [raw.get(cell) for cell in slice_a + slice_b])
            if value is not None:
                vals.append(value)
            else:
                # 八格都已通过有限正 PPL 校验；None 只可能是交互值本身
                # 超出 float 表示范围。计入本交互的排除数而不是悄悄缩小 n。
                pair_excluded += 1
        if not vals:
            continue
        res = paired_t_test(vals, [0.0] * len(vals))
        res.update({
            "candidate": f"{pair} 推理期交互",
            "reference": "可加性（交互=0）",
            "metric": INFERENCE_INTERACTION_METRIC,
            "lower_is_better": True,
            "no_direction": True,
            "interaction": pair,
            "variant": "lega",
            "inference_time": True,
            "n_protocol_seeds": len(vals),
            "excluded_protocol_mismatches": pair_excluded,
            "exploratory": True,
        })
        out.append(_enrich_paired(res, vals, [0.0] * len(vals)))
    return _apply_holm(out)


# 因果病灶消融（推理期权重干预）的跨种子指标名与判定口径。
CAUSAL_ABLATION_METRIC = "因果消融ΔNLL(lesion−normal)"


def _ablation_lesion_records(ablations):
    """{变体: {种子: payload}} -> 逐 (变体, 病灶) 的跨种子观测。

    只消费 payload 中 applied=True 的病灶；不适用/失败/窗口协议不一致的
    记录跳过，并在结果里保留计数。每个种子先聚合成一个 ΔNLL（病灶−正常），
    再对种子做配对检验——窗口级 p 只描述单 run，不进入这里的推断。
    """
    out = []
    ablations = ablations if isinstance(ablations, dict) else {}
    for variant in sorted(ablations):
        seed_map = ablations.get(variant)
        if not isinstance(seed_map, dict):
            continue
        for lesion_id in ABLATION_LESION_ORDER:
            spec = ABLATION_LESION_BY_ID.get(lesion_id, {})
            # query_only/conc_only/len_only 是同一 checkpoint 2^3 的“只保留
            # 单因子”格；它们相对 Full 的 ΔNLL 同时去掉另外两个因子，不能
            # 当成单组件因果证据。它们只进入推理期 Shapley/交互分解。
            if spec.get("factorial_cell"):
                continue
            vals, seeds_used, no_op_seeds = [], [], 0
            window_n = 0
            not_applicable = 0
            for seed in _sorted_seeds(seed_map):
                payload = seed_map.get(seed)
                if not isinstance(payload, dict):
                    continue
                recs = payload.get("lesions")
                rec = (next((r for r in recs
                             if isinstance(r, dict)
                             and r.get("lesion") == lesion_id), None)
                       if isinstance(recs, list) else None)
                if not isinstance(rec, dict):
                    continue
                # 只有真正施加成功的病灶才有 ΔNLL 证据。旧/损坏产物若把
                # applied=false（或缺少 applied）的记录连同 delta 一起写进来，
                # 不能让它进入跨种子检验——那会把“没做干预”伪装成效应。
                if (rec.get("applicable") is False
                        or rec.get("applied") is not True):
                    not_applicable += 1
                    continue
                delta = _finite_number(rec.get("delta_loss_mean"))
                if delta is None:
                    continue
                vals.append(delta)
                seeds_used.append(seed)
                if rec.get("no_op"):
                    no_op_seeds += 1
                nw = _integral_value(rec.get("n_windows"))
                if nw and nw > 0:
                    window_n += nw
            if not vals:
                continue
            res = paired_t_test(vals, [0.0] * len(vals))
            mean_delta = _finite_number(res.get("mean_diff"))
            res.update({
                "candidate": variant,
                "reference": f"{variant}:normal",
                "metric": CAUSAL_ABLATION_METRIC,
                "lower_is_better": True,
                "lesion": lesion_id,
                "lesion_label": spec.get("label", lesion_id),
                "target": spec.get("target", ""),
                "expected": spec.get("expected"),
                "variant": variant,
                "seeds_used": seeds_used,
                "n_no_op_seeds": no_op_seeds,
                "n_not_applicable_seeds": not_applicable,
                "n_windows_total": window_n,
                "mean_ppl_ratio": (
                    math.exp(max(-20.0, min(20.0, float(mean_delta))))
                    if mean_delta is not None else None),
                "mean_delta_ppl_pct": (
                    (math.exp(max(-20.0, min(20.0, float(mean_delta)))) - 1.0)
                    * 100.0 if mean_delta is not None else None),
                "ppl_ratio_clamped": bool(
                    mean_delta is not None
                    and not (-20.0 <= float(mean_delta) <= 20.0)),
            })
            _enrich_paired(res, vals, [0.0] * len(vals))
            # 全部种子 no-op：干预没有真正改变前向，零差不能当“等价证据”。
            if (len(vals) > 0 and no_op_seeds == len(vals)
                    and isinstance(res.get("equivalence"), dict)):
                res["equivalence"].update({
                    "not_applicable": True,
                    "not_applicable_reason": (
                        "不适用：全部种子 no-op（训练后该组件系数已为 0，"
                        "干预未改变前向）"),
                    "status": "not_applicable",
                    "equivalent": False,
                })
            out.append(res)
    # 判定必须在 Holm 校正之后：p_adj 是“显著”的唯一门槛，提前读取会
    # 把全部病灶误判成“未检出”。
    out = _apply_holm(out)
    for res in out:
        # _apply_holm 的“显著劣于”在语义上正确（病灶比正常差），但本表
        # 要回答的是“该机制在推理期是否有因果贡献”，故另给 causal_verdict，
        # 同时保留 Holm 原判供审计。
        res["holm_verdict"] = res.get("verdict")
        mean_delta = _finite_number(res.get("mean_diff"))
        n = _integral_value(res.get("n"))
        no_op_seeds = _integral_value(res.get("n_no_op_seeds")) or 0
        if n is None or n < MIN_SEEDS:
            verdict = f"种子不足（n={n}），不作判定"
        elif no_op_seeds == n:
            verdict = ("全部种子为 no-op：训练后该组件系数已为 0，"
                       "推理期未使用该输入（不是“无效应”的证据）")
        elif res.get("p_adj") is not None and res["p_adj"] < ALPHA:
            if mean_delta is not None and mean_delta > 0:
                verdict = "病灶显著变差：推理期该机制有因果贡献"
            elif mean_delta is not None and mean_delta < 0:
                verdict = "病灶显著变好：该机制在推理期反向有害/冗余"
            else:
                verdict = "显著但均值为 0，无法解释"
            if res.get("within_noise"):
                verdict += "；⚠ 效应落在种子噪声内"
            if res.get("loo_sign_stable") is False:
                verdict += "；⚠ 留一法方向翻转"
        else:
            verdict = ("未检出差异（Holm 校正后）；不等于无因果作用，"
                       "须对照 MDE/窗口功效解读")
        # 部分种子 no-op 时必须在判定里露出，否则读者会把“该组件在部分
        # checkpoint 上根本没被使用”误读成“病灶无效应”。
        if no_op_seeds and no_op_seeds < n:
            verdict = f"部分种子 no-op（{no_op_seeds}/{n}）；" + verdict
        res["causal_verdict"] = verdict
        res["causal_contribution"] = bool(
            n is not None and n >= MIN_SEEDS
            and res.get("p_adj") is not None
            and res["p_adj"] < ALPHA
            and mean_delta is not None and mean_delta > 0
            and not res.get("within_noise")
            and res.get("loo_sign_stable") is not False
            and no_op_seeds < n)
    return out


def _auroc_diff_tests(runs) -> list:
    """AUROC(logit)−AUROC(qterm) 配对差检验（H6 的判定依据）。"""
    return _detection_diff_tests(runs, "auroc", AUROC_DIFF_METRIC)


def _ap_diff_tests(runs) -> list:
    """PR-AUC(logit)−PR-AUC(qterm) 配对差检验（不平衡下的主指标口径，H7）。"""
    return _detection_diff_tests(runs, "auprc", AP_DIFF_METRIC)


# 注册假设（confirmatory）：在正式实验前登记的方向性主张；未登记的对比
# 一律视为事后（exploratory）分析，只能作为提出新假设的线索。
# 登记表与防篡改锁链的实现见 lega/hypotheses.py（锁基线 + 修订链）。


def load_registered_hypotheses(path: str = HYPOTHESES_PATH) -> dict:
    """读取预注册假设文件；缺失/损坏时返回空 dict（汇总页显式提示）。"""
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
    except (OSError, ValueError) as e:
        print(f"[summary] 警告：注册假设文件不可读（{e}），跳过判定")
        return {}
    if not isinstance(d, dict):
        print("[summary] 警告：注册假设文件顶层不是 JSON 对象，跳过判定")
        return {}
    if not isinstance(d.get("hypotheses"), list) or not d["hypotheses"]:
        print("[summary] 警告：注册假设文件无 hypotheses 条目，跳过判定")
        return {}
    return d


def evaluate_registered_hypotheses(registered: dict, tests: list,
                                    alpha: float = ALPHA) -> list:
    """把预注册假设逐条对照 Holm 校正后的配对检验，判定 支持/拒绝/未定。

    “拒绝”只在方向相反且校正后显著时给出（假设被证伪）；
    “未定”包括种子不足、对比未运行、校正后不显著——不显著绝不等于
    “假设成立”，也不等于“无效应”，配合 MDE 才能判断是功效不足还是
    真的没有效应。
    """
    hypotheses = _as_dict(registered).get("hypotheses")
    if not isinstance(hypotheses, list) or not hypotheses:
        return []
    out = []
    for h in hypotheses:
        if not isinstance(h, dict):
            continue
        t = _find_test(tests, (h.get("candidate"), h.get("reference")),
                       h.get("metric"))
        item = {"id": h.get("id"), "claim": h.get("claim"),
                "metric": h.get("metric"),
                "candidate": h.get("candidate"), "reference": h.get("reference"),
                "direction": h.get("direction"),
                # p 为家族内校正前的原始配对 p；全局确认族校正需要它，
                # 不能只保存 p_adj（否则无法跨指标家族统一校正）。
                "p": None, "p_adj": None, "p_perm_adj": None,
                "p_adj_confirmatory": None, "confirmatory_family_size": None,
                "global_fwer_pass": None, "status_family": None,
                "mean_diff": None,
                "n": None, "mde": None, "within_noise": None,
                "below_mde": None, "status": None, "detail": None}
        if t is None:
            item["status"] = "未检验"
            item["detail"] = "对应的对比/指标没有检验结果（run 缺失或未评测）"
        else:
            n = _integral_value(t.get("n"))
            if n is None:
                item["status"] = "未定"
                item["detail"] = f"无法检验（样本量缺失或非法 n={t.get('n')!r}）"
            elif n < MIN_SEEDS:
                item["status"] = "未定"
                item["detail"] = f"种子不足（n={n}<{MIN_SEEDS}）"
            elif t.get("p_adj") is None:
                item["status"] = "未定"
                item["detail"] = "无法检验（方差退化或指标缺失）"
            elif t["p_adj"] >= alpha:
                item["status"] = "未定"
                item["detail"] = "Holm 校正后不显著"
                if t.get("mde") is not None and t.get("mean_diff") is not None:
                    item["detail"] += (f"（观测差 {t['mean_diff']:+.4f}，"
                                       f"当前种子数可检测的最小效应 "
                                       f"{t['mde']:+.4f}）")
            else:
                better = (t["mean_diff"] < 0 if t["lower_is_better"]
                          else t["mean_diff"] > 0)
                supported = (better if h.get("direction") == "candidate_better"
                             else not better)
                item["status"] = "支持" if supported else "拒绝"
                item["detail"] = ("方向与登记一致且校正后显著" if supported
                                  else "校正后显著但方向与登记相反")
                if (t.get("p_perm_adj") is not None
                        and t["p_perm_adj"] >= alpha):
                    item["detail"] += "；⚠ 置换检验不显著，小样本下需谨慎"
        if t is not None:
            item.update({"p": t.get("p"),
                         "p_adj": t.get("p_adj"),
                         "p_perm_adj": t.get("p_perm_adj"),
                         "mean_diff": t.get("mean_diff"),
                         "n": t.get("n"), "mde": t.get("mde"),
                         "within_noise": t.get("within_noise"),
                         "below_mde": t.get("below_mde")})
            note = _noise_verdict_note(t)
            if note and item.get("detail"):
                item["detail"] += note
            # 预注册判定只回答“方向是否显著”，不等价于“效应有实际意义”。
            # 把 TOST/SESOI 作为补充字段挂上，避免把“显著但 0.1%”写成
            # 重要收益，也避免把“未定”直接读成“无差异”。
            eq = t.get("equivalence")
            if isinstance(eq, dict):
                item.update({
                    "equivalence_status": eq.get("status"),
                    "equivalence_label": eq.get("label"),
                    "equivalence_p_tost_adj": eq.get("p_tost_adj"),
                    "equivalence_margin": eq.get("margin"),
                    "equivalence_margin_label": eq.get("margin_label"),
                })
                eq_status = eq.get("status")
                if item.get("status") in ("支持", "拒绝"):
                    if eq_status == "significant_below_sesoi":
                        item["detail"] += ("；幅度低于 SESOI，实际意义有限"
                                           "（不能只报“显著”就当作重要收益）")
                elif item.get("status") == "未定":
                    if eq_status == "equivalent":
                        item["detail"] += ("；等价性：差异小于 SESOI"
                                           "（不能据此说效应恰为 0）")
                    elif eq_status == "significant_below_sesoi":
                        item["detail"] += "；统计显著但幅度低于 SESOI"
                    elif eq_status == "undetermined":
                        item["detail"] += ("；等价性未定（区间未排除 SESOI，"
                                           "不能宣称无差异）")
        out.append(item)
    return apply_global_fwer(out, alpha=alpha)


def apply_global_fwer(hyp_eval: list, alpha: float = ALPHA) -> list:
    """跨全部可检验预注册假设再做一次 Holm 校正（确认性结论的整体 FWER）。

    指标家族内校正只控制“该指标家族”的 FWER；H1–H7 跨越 PPL/RULER/
    AUROC/AP 多个家族，各自在 5% 下判定时，整体至少一次假阳性的概率仍
    可能超过 5%。这里把所有“已检验且有 n>=MIN_SEEDS”的预注册假设组成
    一个确认族，用原始 p 做 Holm 校正（不能用已按更大探索族校正过的
    p_adj 再校正，否则是重复惩罚）：家族内显著但全局不显著的主张降级为
    “未定”，并保留 status_family 供审计。

    该门只会让判定更严格，不会把原先不支持的主张升级；未检验/种子不足
    的假设不进入族（也不被这条门改变状态）。
    """
    items = [h for h in (hyp_eval if isinstance(hyp_eval, list) else [])
             if isinstance(h, dict)]
    for h in items:
        # 允许重复调用：status_family 固定为第一次进入全局门前的家族判定。
        if h.get("status_family") is None:
            h["status_family"] = h.get("status")
    tested = [h for h in items
              if h.get("p") is not None
              and isinstance(h.get("n"), int)
              and not isinstance(h.get("n"), bool)
              and h["n"] >= MIN_SEEDS
              and h.get("status_family") in ("支持", "拒绝", "未定")]
    pads = holm_bonferroni([h.get("p") for h in tested])
    m = len(tested)
    for h, padj in zip(tested, pads):
        h["p_adj_confirmatory"] = padj
        h["confirmatory_family_size"] = m
        h["global_fwer_pass"] = bool(padj is not None and padj < alpha)
        if (h.get("status_family") in ("支持", "拒绝")
                and h["global_fwer_pass"] is False):
            h["status"] = "未定"
            h["detail"] = ((h.get("detail") or "")
                           + "；跨全部预注册假设的全局 Holm 校正后不显著"
                             "（确认族整体 FWER），不能作为确认性结论")
    return items


def apply_integrity_gate(hyp_eval: list, integrity: dict) -> list:
    """预注册完整性失败时给全部确认性判定加盖“无效”戳。

    预注册的全部意义在于事后不能静默改写；一旦 lega/hypotheses.py 的
    锁链校验发现登记表被删/改/无留痕新增，这些判定就不再具有 confirmatory
    地位——逐条标注 valid 标志与原因，汇总页显式降级为探索性，
    而不是假装边界仍然成立。
    """
    # 只接受真正的 bool true：字符串 "false"/1 经 truthiness 会错误地授予
    # 确认性资格，恰好绕过“完整性失败必须降级”的保护。
    valid = _as_dict(integrity).get("confirmatory_valid") is True
    reason = ("登记表完整性校验通过" if valid else
              "预注册完整性校验未通过（登记表被删/改/无留痕新增，"
              "或锁文件缺失）：判定降级为探索性，不得作为 confirmatory 结论")
    for h in (hyp_eval if isinstance(hyp_eval, list) else []):
        if not isinstance(h, dict):
            continue
        h["confirmatory_valid"] = valid
        h["confirmatory_note"] = reason
    return hyp_eval


# 预注册 scope 核对的配置字段：模型/训练规模（结论可迁移性的前提）。
# lr/weight_decay 刻意排除：按本项目协议，基线调优（--tune_baseline）会在
# 验证集上选定统一超参并注入全部变体，属预注册流程的一部分而非范围外改动。
_SCOPE_CONFIG_KEYS = ("d_model", "n_layers", "n_heads", "ffn_mult",
                      "block_size", "steps", "train_ctx")


def registration_scope_check(registered: dict, repro: dict,
                             root: str = ROOT) -> dict:
    """核对“本次 run 是否落在预注册 scope 内”（confirmatory 资格的前提）。

    预注册假设只在登记的实验范围内具有确认性地位：用 tiny 配置、冒烟
    步数、字符伪数据或其它规模跑出的“支持/拒绝”不能冒充预注册结论。
      * smoke 运行 = 明确的范围外；
      * 配置的模型/训练规模字段逐项比对（lr/wd 除外，见上）；
      * scope.data 声明 gpt2 BPE 时核对实际数据分词器（伪数据/换词表会
        改变目标语言建模任务本身，必须显形）；
      * 额外（未登记）种子只列为 warning：种子可交换、不直接推翻判定，
        但偏离登记计划必须显式可见，由读者知情。
    返回 {checked, in_scope, problems, warnings, scope}；无法核对（缺
    repro/config 文件）时 checked=False、in_scope=None，由
    apply_scope_gate 按“不可确认”处理，而不是默认通过。
    """
    registered = _as_dict(registered)
    repro = _as_dict(repro)
    scope = _as_dict(registered.get("scope"))
    out = {"scope": scope, "checked": False, "in_scope": None,
           "problems": [], "warnings": []}
    if not scope:
        out["warnings"].append("登记表未提供 scope 字段，无法核对实验范围")
        return out
    if not repro:
        out["warnings"].append("缺少 repro.json，无法核对实验范围")
        return out

    def _load_json(p):
        if not isinstance(p, (str, os.PathLike)):
            return None
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            return None

    checked_any = False
    unverifiable = []
    if repro.get("smoke") is True:
        out["problems"].append("冒烟运行（--smoke）不在预注册范围（正式配置/全部种子）内")
        checked_any = True

    want_rel = str(scope.get("config") or "")
    want_path = (want_rel if os.path.isabs(want_rel)
                 else os.path.join(root, want_rel))
    got_rel = repro.get("config_file")
    # 与 scope.config 对称：相对路径按项目根解析（否则会相对 cwd 去读另一个
    # 同名文件，悄悄核对错配置）
    got_path = (os.path.join(root, got_rel)
                if isinstance(got_rel, str) and got_rel
                and not os.path.isabs(got_rel) else got_rel)
    want_cfg = _load_json(want_path) if want_path else None
    got_cfg = _load_json(got_path) if got_path else None
    want_cfg = want_cfg if isinstance(want_cfg, dict) else None
    got_cfg = got_cfg if isinstance(got_cfg, dict) else None
    if want_rel:
        if want_cfg is None:
            unverifiable.append("登记 scope.config 不可读")
            out["warnings"].append(
                f"登记 scope.config 不可读（{want_rel or '-'}），无法核对配置规模")
        elif got_cfg is None:
            unverifiable.append("本次配置不可读")
            out["warnings"].append(
                f"本次配置不可读（{got_path or '-'}），无法核对配置规模")
        else:
            diffs = [f"{k}: 登记={want_cfg.get(k)} vs 实际={got_cfg.get(k)}"
                     for k in _SCOPE_CONFIG_KEYS
                     if k in want_cfg and want_cfg.get(k) != got_cfg.get(k)]
            if diffs:
                out["problems"].append(
                    "模型/训练规模与预注册配置不一致（" + "；".join(diffs) + "）")
            checked_any = True

    scope_data = str(scope.get("data") or "")
    actual_tok = ((repro.get("data") or {}).get("tokenizer")
                  if isinstance(repro.get("data"), dict) else None)
    if "gpt2" in scope_data.lower():
        if actual_tok is None:
            unverifiable.append("数据分词器不可读")
            out["warnings"].append(
                "本次数据分词器不可读（repro.json 缺 data.tokenizer），"
                "无法核对 scope.data")
        elif "gpt2" not in str(actual_tok).lower():
            out["problems"].append(
                f"数据分词器与预注册不一致（登记含 gpt2 BPE，实际 {actual_tok}）")
            checked_any = True
        else:
            checked_any = True

    want_seeds = scope.get("seeds")
    got_seeds = repro.get("seeds")
    if isinstance(want_seeds, list):
        if not isinstance(got_seeds, list):
            unverifiable.append("本次种子列表不可读")
            out["warnings"].append(
                "repro.json 缺少可读的 seeds，无法核对预注册种子范围")
        else:
            extra = [s for s in got_seeds if s not in want_seeds]
            if extra:
                out["warnings"].append(
                    f"本次含未登记种子 {extra}：判定基于登记计划外的抽样，证据强度打折")
            checked_any = True
    # 任一已声明范围字段无法核对时，不能因为其它字段通过就默认在范围内；
    # 保持 checked=False / in_scope=None，由 apply_scope_gate 降级。
    out["checked"] = bool(checked_any) and not unverifiable
    if out["problems"]:
        out["in_scope"] = False
    elif out["checked"]:
        out["in_scope"] = True
    return out


def apply_scope_gate(hyp_eval: list, scope_check: dict) -> list:
    """预注册范围核对失败/无法核对时，确认性判定显式降级。

    与完整性门（apply_integrity_gate）同义：登记表没被篡改，但本次 run
    根本不在登记范围内（如 tiny 配置、冒烟步数、伪数据、无法核对），
    此时“支持/拒绝”同样不作数——降级为探索性线索，避免范围外结果
    冒充预注册结论。
    """
    scope_check = _as_dict(scope_check)
    if not scope_check.get("scope"):
        # 登记表未声明 scope（历史/精简登记表）：无从核对，仅由汇总页告警，
        # 不因“没有可核对的字段”本身降级已通过的完整性判定。
        return hyp_eval
    problems = _as_text_list(scope_check.get("problems"))
    # checked 必须是真正的 bool true：字符串 "false"/1 经 truthiness 会被
    # 当成“已核对在范围内”，从而跳过缺 repro/config 时的确认性降级。
    if not problems and scope_check.get("checked") is True:
        return hyp_eval
    reason = ("本次运行超出预注册范围：" + "；".join(problems)
              if problems else
              "无法核对本次运行是否落在预注册范围（缺少 repro/config），"
              "确认性判定不可确认")
    for h in (hyp_eval if isinstance(hyp_eval, list) else []):
        if not isinstance(h, dict):
            continue
        if h.get("confirmatory_valid") is False:
            continue  # 已因完整性失败降级，保留原原因
        h["confirmatory_valid"] = False
        h["confirmatory_note"] = reason
    return hyp_eval


def apply_budget_gate(hyp_eval: list, audit: dict) -> list:
    """早停导致实际训练量不等时，确认性判定显式降级。

    对比实验要求除待验证模块外的训练步数一致；早停按各变体自己的验证曲线
    停在不同步数，实际 tokens_seen 因变体而异。计划步数相同并不能消除
    “有效训练量不同”这一混杂，因此确认性结论降级为探索性，直到用相同
    实际步数重跑。
    """
    audit = _as_dict(audit)
    stopped = _as_text_list(audit.get("early_stop_runs"))
    if not stopped:
        return hyp_eval
    reason = ("早停使实际训练量不一致（" + "；".join(stopped[:4])
              + (" ..." if len(stopped) > 4 else "")
              + "）：确认性判定降级为探索性；请在相同实际步数下重跑")
    for h in (hyp_eval if isinstance(hyp_eval, list) else []):
        if not isinstance(h, dict):
            continue
        if h.get("confirmatory_valid") is False:
            continue
        h["confirmatory_valid"] = False
        h["confirmatory_note"] = reason
    return hyp_eval


def _render_registration_integrity(integrity: dict,
                                   scope_check: dict = None) -> list:
    """预注册完整性小节：删除/静默改写必须在校验中显形。"""
    integrity = _as_dict(integrity)
    st = integrity.get("status")
    if not isinstance(st, str):
        st = None
    icon = {"intact": "✅", "amended": "✅", "violated": "⛔",
            "missing_lock": "⚠", "missing_registry": "⛔"}.get(st, "?")
    desc = {
        "intact": "与锁基线一致（登记后未发生任何修订）",
        "amended": "与修订链一致（既有假设未被改写/删除，"
                   f"链上共 {integrity.get('chain_length')} 环留痕修订）",
        "violated": "完整性校验失败：登记表被删除/改写，或新增假设未留痕",
        "missing_lock": "缺少锁文件，无法验证登记表未被静默改写",
        "missing_registry": "登记表文件缺失，预注册边界不存在",
    }.get(st, "无法判定")
    lines = ["### 预注册完整性（防事后静默改写）", "",
             f"{icon} 状态：`{st}` —— {desc}。"]
    sha = integrity.get("registry_sha256")
    if sha:
        lines.append(f"- 登记表 SHA-256：`{str(sha)[:16]}…`（假设 "
                     f"{integrity.get('n_hypotheses')} 条"
                     + (f"；锁基线 `{str(integrity.get('baseline_document_sha256'))[:16]}…`"
                        if integrity.get("baseline_document_sha256") else "")
                     + "）")
    else:
        lines.append("- 登记表 SHA-256：不可用（文件缺失或不可解析）")
    for p in _as_text_list(integrity.get("problems")):
        lines.append(f"- ⛔ {p}")
    for w in _as_text_list(integrity.get("warnings")):
        lines.append(f"- ⚠ {w}")
    # 预注册范围核对：完整性通过不等于本轮 run 落在登记范围内
    # （tiny 配置/冒烟步数/伪数据跑出的“支持”同样不能冒充确认性结论）。
    sc = _as_dict(scope_check)
    scope = _as_dict(sc.get("scope"))
    scope_failed = False
    if scope or sc:
        # 只有登记表声明了 scope 时才谈得上“范围核对失败”。checked 必须是
        # 真正的 bool true，字符串 "false"/1 不能渲染成“已核对在范围内”。
        scope_failed = bool(scope) and (
            bool(sc.get("problems")) or sc.get("checked") is not True)
        lines.append("")
        lines.append("#### 预注册范围核对（确认性资格的前提）")
        sc_problems = _as_text_list(sc.get("problems"))
        if sc_problems:
            lines.append("- ⛔ 本次运行超出预注册范围：")
            for p in sc_problems:
                lines.append(f"  - {p}")
        elif sc.get("checked") is True:
            cfg_note = (f"模型/训练规模与 `{scope.get('config')}` 一致"
                        if scope.get("config") else "未声明配置范围")
            tok_note = ("；数据分词器与 scope.data 一致"
                        if "gpt2" in str(scope.get("data") or "").lower()
                        else "")
            seeds_note = ("；本次种子均在登记范围内"
                          if isinstance(scope.get("seeds"), list) else "")
            lines.append(f"- ✅ 本次 run 在预注册范围内：{cfg_note}"
                         f"{tok_note}{seeds_note}。")
        else:
            # 无 scope 的登记表不降级（与 apply_scope_gate 的早退一致），
            # 只有“声明了 scope 却核不动”才是不可确认
            lines.append("- ⚠ 无法核对预注册范围（缺少 repro/config 信息）："
                         + ("确认性判定按“不可确认”降级。" if scope else
                            "登记表未声明 scope，完整性判定不受影响。"))
        for w in _as_text_list(sc.get("warnings")):
            lines.append(f"- ⚠ {w}")
    if integrity.get("confirmatory_valid") is True and not scope_failed:
        lines.append("- 修订规则：既有假设只增不改不删；新增假设必须经 "
                     "`python -m lega.hypotheses --amend --reason ...` 留痕，"
                     "否则判违反。")
    else:
        lines.append("- ⚠ **本页全部“注册假设判定”因完整性/范围核对未通过而"
                     "失效（降级为探索性分析）**：排除问题后重跑汇总。")
    lines.append("")
    return lines


def _as_dict(v) -> dict:
    """外部/旧格式 JSON 的嵌套对象：非 dict 统一按空对象处理。"""
    return v if isinstance(v, dict) else {}


def _as_dict_list(v) -> list:
    """只保留列表中的对象元素，供外部 JSON 的告警/访问记录使用。"""
    return [x for x in v if isinstance(x, dict)] if isinstance(v, list) else []


def _as_text_list(v) -> list:
    return [str(x) for x in v] if isinstance(v, list) else []


def _as_text(v) -> str:
    """外部 JSON 的文本字段：非字符串按可读文本降级（缺失为空串）。"""
    return v if isinstance(v, str) else ("" if v is None else str(v))


def _int_or_zero(v) -> int:
    """外部/旧版 JSON 的计数 → 非负整数；bool/小数/负数/坏类型按 0 处理。"""
    iv = _integral_value(v)
    return iv if iv is not None and iv >= 0 else 0


def _fmt_rate(x) -> str:
    f = _finite_number(x)
    return "-" if f is None else f"{f:.3%}"


def _fmt4(x) -> str:
    """四位小数；None/非数值显式显示 '-'（如空 split 的分布漂移统计为 null）。"""
    f = _finite_number(x)
    return "-" if f is None else f"{f:.4f}"


def _has_test_split(data_dir: str) -> bool:
    """数据目录是否有非空 test.bin（与 lega.evaluate 的 --split auto 同规则）。"""
    p = os.path.join(data_dir, "test.bin")
    return os.path.exists(p) and os.path.getsize(p) > 0


def resolve_ngram_split(data_dir: str, requested: str = "auto") -> str:
    """把 --ngram_split 的 auto 解析为与神经模型评测完全一致的 split。

    口径不一致的陷阱：神经模型的 PPL 默认在 held-out test 上评测
    （存在 test.bin 时），而 n-gram 基线若固定在 val 上算，汇总页给出的
    “神经模型 vs 统计基线”就是 test-vs-val 的错位比较（数据集切片不同、
    难度不同，绝对差值不可解释）。auto 与 lega.evaluate 的 --split auto
    使用同一条判定规则：test.bin 存在且非空 → test，否则回退 val。
    """
    if requested == "auto":
        return "test" if _has_test_split(data_dir) else "val"
    return requested


def _neural_eval_window_spec(cfg: dict, smoke: bool):
    """推导神经评测实际使用的窗口规格 (ctx, iters)，供 n-gram 逐位置对齐。

    口径必须与 train.py/evaluate.py 完全一致：ctx = min(block_size, train_ctx)
    （冒烟模式先应用 train.py 的 SMOKE_OVERRIDES），iters = eval_iters
    （缺省 50）。规格非法时返回 None（不阻塞基线评测）。
    """
    from lega.train import SMOKE_OVERRIDES
    if not isinstance(cfg, dict):
        return None
    try:
        # 严格整数：int(True)/int(2.5) 会把“同窗口”对齐的 ctx/iters 静默改值，
        # 使 n-gram 与神经评测实际上在不同 token 集合上比较。
        ctx = strict_int(cfg.get("train_ctx") or cfg.get("block_size") or 0,
                         "train_ctx")
        iters = strict_int(cfg.get("eval_iters", 50), "eval_iters")
        if smoke:
            if "train_ctx" in SMOKE_OVERRIDES:
                ctx = strict_int(SMOKE_OVERRIDES["train_ctx"], "smoke.train_ctx")
            if "eval_iters" in SMOKE_OVERRIDES:
                iters = strict_int(SMOKE_OVERRIDES["eval_iters"],
                                   "smoke.eval_iters")
        block = strict_int(cfg.get("block_size") or 0, "block_size")
    except ValueError:
        return None
    if block > 0:
        ctx = min(ctx, block)
    if ctx < 1 or iters < 1:
        return None
    return {"ctx": ctx, "iters": iters}


def _eval_matches_window_spec(ev: dict, wspec: dict) -> bool:
    """run 的 eval JSON 是否与 n-gram 窗口规格逐字段一致。

    字段口径：eval_ctx=实际评测上下文、eval_iters=实际窗口数（已按可用
    窗口数截断）、val_windows=评测 split 的可用窗口数。任一项缺失/不同即
    视为未对齐（历史产物、ctx/iters 被覆盖等情况），回退全 split 参考。
    """
    if not isinstance(ev, dict) or not isinstance(wspec, dict) or not wspec:
        return False
    return (_same_integral_value(ev.get("eval_ctx"), wspec.get("ctx"))
            and _same_integral_value(ev.get("eval_iters"),
                                     wspec.get("n_eval"))
            and _same_integral_value(ev.get("val_windows"),
                                     wspec.get("n_windows")))


def ngram_baseline_comparison(runs: dict, results_dir: str) -> dict:
    """神经变体 vs 简单统计基线（同 split、能校验时同 token）的统计对比。

    科研方法问题：“n-gram PPL=xx” 与 “神经 PPL=yy” 并排放着不构成
    “复杂模型比简单基线好”的证据——必须同 split、必须对照方差，而且
    两边必须打在**同一批 token** 上：神经 PPL 只在 eval_iters 个等间隔
    窗口的目标 token 上计算，n-gram 若在整个 split 上打分就是不同 token
    集合上的 PPL（同 split 不等于同 token）。n-gram 对给定 split 是确定值
    （无种子方差），因此每个神经变体的逐种子 PPL 与同一常数配对 = 单样本
    检验；只在两侧 split 一致的种子上比较，split 不一致的 run 显式列出且
    不混入（避免 test-vs-val 错位）。参考值的优先级：全部纳入比较的 run
    的 eval 字段都与 n-gram 的窗口规格一致时，优先用
    `ppl_window_matched_selected`（在同一批窗口 token 上重新选参），旧
    产物才用 `ppl_window_matched`（全 val 选参）；任一不一致（历史产物/
    规格缺省）则整体回退全 split 参考并在结果里显式标注 `reference_source`
    ——同一 Holm 家族必须共用同一个参考值。多个变体同时对比时沿用
    Holm–Bonferroni 校正。该对比不属于预注册假设，标为补充性/探索性证据。
    """
    path = os.path.join(results_dir, "ngram_baseline.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            ng = json.load(f)
    except (OSError, ValueError):
        return {}
    # 合法 JSON 但非对象（如数组）按不可用处理。
    if not isinstance(ng, dict):
        return {"error": "ngram_baseline.json 不是 JSON 对象"}
    # 旧基线必须绑定到本轮的数据身份：results/<tag> 复用且 --skip_ngram 时
    # 目录里的 n-gram PPL 可能来自另一份数据（甚至重建后的 test.bin），
    # 与当前神经 PPL 并排比较会制造“显著优于统计基线”的假结论。新产物
    # 带 data_fingerprint；旧产物缺该字段时保持兼容（无法核对），但只要有
    # 指纹且与 repro.json 不一致就直接标记不可用。
    repro_path = os.path.join(results_dir, "repro.json")
    if os.path.exists(repro_path):
        try:
            with open(repro_path, "r", encoding="utf-8") as f:
                repro_doc = json.load(f)
        except (OSError, ValueError):
            repro_doc = None
        # repro.json 是外部/旧产物：data 合法 JSON 但不是对象（数组/标量）
        # 时按“无法核对指纹”处理，不能在整个汇总页上抛 AttributeError。
        expected_fp = (_as_dict(repro_doc.get("data")).get("digest")
                       if isinstance(repro_doc, dict) else None)
        actual_fp = ng.get("data_fingerprint")
        if (isinstance(expected_fp, str) and expected_fp
                and isinstance(actual_fp, str) and actual_fp
                and actual_fp != expected_fp):
            return {"error": "ngram_baseline.json 的数据指纹与本次 run 的 "
                             "repro.json 不一致：该基线来自另一份数据，"
                             "不能与当前神经 PPL 直接比较；请重跑 n-gram 基线",
                    "split": ng.get("split"),
                    "ngram_data_fingerprint": actual_fp,
                    "run_data_fingerprint": expected_fp}
    ppl_map = ng.get("ppl")
    if ppl_map is not None and not isinstance(ppl_map, dict):
        return {"error": "ngram_baseline.json 的 ppl 不是 JSON 对象"}
    ppl_map = ppl_map or {}
    # 优先用 n-gram 侧按 val 选出的“最强简单基线”，避免固定拿较弱
    # bigram 当靶子；历史产物没有 selected_model 时再按插值→bigram→unigram
    # 的强度顺序回退。
    ref_name = None
    sel = ng.get("selected_model")
    if isinstance(sel, str) and ppl_map.get(sel) is not None:
        ref_name = sel
    if ref_name is None:
        ref_name = next((k for k in ("interpolated_bigram", "bigram",
                                     "unigram")
                         if ppl_map.get(k) is not None), None)
    if ref_name is None:
        return {"error": "ngram_baseline.json 没有可用 PPL"}
    ref_ppl_full = _finite_number(ppl_map[ref_name])
    if ref_ppl_full is None:
        return {"error": "ngram PPL 非数值"}
    if ref_ppl_full <= 0:
        return {"error": "ngram PPL 非有限值或非正数"}
    wspec = ng.get("window_spec")
    wspec = wspec if isinstance(wspec, dict) else {}
    # 窗口参考优先使用“在同一批窗口 token 上重新选参”的结果：神经模型
    # 的 checkpoint 由 eval_iters 个窗口上的 val loss 选出，n-gram 若仍用
    # 完整 val 选出的 α/λ，选参与比较的 token 集就不同。新产物会落盘
    # `window_matched_selection` + `ppl_window_matched_selected`；旧产物
    # 缺少这些字段时回退到全 val 选参的 `ppl_window_matched`（保持兼容）。
    wm_sel = ng.get("window_matched_selection")
    wm_sel = wm_sel if isinstance(wm_sel, dict) else {}
    wm_map_full = ng.get("ppl_window_matched")
    wm_map_full = wm_map_full if isinstance(wm_map_full, dict) else {}
    wm_map_sel = ng.get("ppl_window_matched_selected")
    wm_map_sel = wm_map_sel if isinstance(wm_map_sel, dict) else {}
    wm_model_sel = wm_sel.get("selected_model")
    wm_ref_name = ref_name
    wm_ref = None
    wm_ref_map = wm_map_full
    if (isinstance(wm_model_sel, str)
            and _finite_number(wm_map_sel.get(wm_model_sel)) is not None):
        wm_ref_name = wm_model_sel
        wm_ref = _finite_number(wm_map_sel.get(wm_model_sel))
        wm_ref_map = wm_map_sel
        ng_win_field = "window_losses_matched_selected"
        ng_idx_field = "window_indices_matched_selected"
        ng_win_model = ng.get("window_losses_matched_selected_model")
    else:
        wm_ref = _finite_number(wm_map_full.get(ref_name))
        wm_ref_map = wm_map_full
        ng_win_field = "window_losses_matched"
        ng_idx_field = "window_indices_matched"
        ng_win_model = ng.get("window_losses_matched_model")
    # 选定窗口参考仍不可用/非正时，回到全 val 选参的历史口径。
    if wm_ref is None or wm_ref <= 0:
        wm_ref_name = ref_name
        wm_ref = _finite_number(wm_map_full.get(ref_name))
        wm_ref_map = wm_map_full
        ng_win_field = "window_losses_matched"
        ng_idx_field = "window_indices_matched"
        ng_win_model = ng.get("window_losses_matched_model")
    # n-gram 逐窗口损失（选定模型）用于真正的窗口级配对，而不是只把
    # 神经逐种子 PPL 与一个常数比较。缺字段/模型不匹配时显式降级。
    ng_win = _eval_window_losses({
        "window_losses": ng.get(ng_win_field),
        "window_indices": ng.get(ng_idx_field)})
    if (ng_win is not None and ng_win_model is not None
            and ng_win_model != wm_ref_name):
        ng_win = None
        ng_win_model = None
    ng_split = ng.get("split")
    # 第一遍：收集逐种子 PPL 与该 run 是否与窗口规格逐字段一致。
    staged, incomparable = [], []
    runs = runs if isinstance(runs, dict) else {}
    for v in sorted(runs):
        if v not in VARIANTS:
            continue
        m = _variant_seed_map(runs, v)
        vals, seeds_used, eval_splits, excl_seeds = [], [], set(), []
        matched = []
        win_maps = []
        for s in _sorted_seeds(m):
            slot = m.get(s)
            d = slot.get("eval") if isinstance(slot, dict) else None
            if not d:
                continue
            ev_split = d.get("split", "val")
            if ev_split != ng_split:
                try:
                    eval_splits.add(ev_split)
                except TypeError:
                    eval_splits.add(str(ev_split))
                excl_seeds.append(s)
                continue
            p = _ppl_of(d)
            if p is None or not math.isfinite(p):
                continue
            vals.append(p)
            seeds_used.append(s)
            matched.append(_eval_matches_window_spec(d, wspec))
            win_maps.append(_eval_window_losses(d))
        if eval_splits:
            # 只要存在 split 错位的种子就必须显式列出（哪怕同变体另有可比
            # 种子已被纳入）：静默丢弃部分种子同样会让读者高估证据完整性
            incomparable.append({"variant": v,
                                 "eval_splits": sorted(eval_splits, key=str),
                                 "excluded_seeds": [
                                     _integral_value(s)
                                     for s in excl_seeds]})
        if vals:
            staged.append({"variant": v, "vals": vals,
                           "seeds": [_integral_value(s)
                                     for s in seeds_used],
                           "matched": matched, "win_maps": win_maps})
    # 参考值决策（第一步）：全部纳入比较的 run 都与窗口规格一致，且
    # window-matched PPL 可用 → 用逐位置对齐参考；否则整体回退全 split。
    # 同一 Holm 家族必须共用同一参考值，不能逐行混用。
    if wm_ref is not None and wm_ref <= 0:
        wm_ref = None
    all_matched = bool(staged) and all(all(r["matched"]) for r in staged)
    use_window = bool(wspec) and wm_ref is not None and all_matched
    ref_ppl = wm_ref if use_window else ref_ppl_full
    ref_source = "window_matched" if use_window else "full_split"
    rows = []
    for st in staged:
        v, vals = st["variant"], st["vals"]
        # 窗口级配对：同一批窗口序号上 d_w = loss_neural − loss_ngram。
        # 仅当全体 run 都采用逐位置对齐参考、n-gram 有选定模型的逐窗口
        # 损失且窗口序号完全一致时才计算；否则明确置 None，不按位置强配。
        win_diffs = []
        win_skip_reason = None
        if not use_window:
            win_skip_reason = "参考已回退全 split，无逐窗口对齐"
        elif ng_win is None:
            win_skip_reason = "n-gram 缺少选定模型的逐窗口损失"
        else:
            for wmap in st.get("win_maps") or []:
                if not wmap or set(wmap) != set(ng_win):
                    win_skip_reason = "神经与 n-gram 的窗口序号不一致"
                    win_diffs = []
                    break
                keys = sorted(wmap)
                arr = np.asarray([wmap[k] - ng_win[k] for k in keys],
                                 dtype=np.float64)
                if arr.size == 0 or not np.all(np.isfinite(arr)):
                    win_skip_reason = "逐窗口损失非有限值"
                    win_diffs = []
                    break
                win_diffs.append(arr)
        win_stats = (_window_bootstrap_stats(win_diffs)
                     if win_diffs else None)
        res = paired_t_test(vals, [ref_ppl] * len(vals))
        res.update({"candidate": v, "reference": ref_name,
                    "metric": "PPL vs ngram", "lower_is_better": True})
        res = _enrich_paired(res, vals, [ref_ppl] * len(vals))
        # 与其它汇总统计同口径：朴素 sum 会在多个 1e308 的合法 PPL 上溢出。
        mean_ppl, std_ppl = mean_std(vals)
        ratio = (mean_ppl / ref_ppl
                 if mean_ppl is not None and ref_ppl > 0 else None)
        if ratio is not None and not math.isfinite(ratio):
            ratio = None
        reduction_pct = ((1.0 - ratio) * 100.0
                         if ratio is not None else None)
        if reduction_pct is not None and not math.isfinite(reduction_pct):
            reduction_pct = None
        rows.append({
            "variant": v, "n": len(vals), "seeds": st["seeds"],
            "mean_ppl": mean_ppl,
            "std_ppl": std_ppl,
            "ratio": ratio,
            "reduction_pct": reduction_pct,
            "p": res.get("p"), "p_adj": None,
            "mean_diff": res.get("mean_diff"),
            "mean_diff_ci95": res.get("mean_diff_ci95"),
            "p_perm": res.get("p_perm"), "mde": res.get("mde"),
            "token_matched": bool(all(st["matched"])) and use_window,
            "window_paired_n_seeds": (win_stats["n_seeds"]
                                      if win_stats else None),
            "window_paired_mean_diff": (win_stats["mean_diff"]
                                        if win_stats else None),
            "window_paired_ci95": (win_stats["mean_diff_ci95"]
                                   if win_stats else None),
            "window_paired_block_ci95": (
                win_stats.get("block_mean_diff_ci95") if win_stats else None),
            "window_paired_block_p": (
                win_stats.get("block_p") if win_stats else None),
            "window_paired_block_length_min": (
                win_stats.get("block_length_min") if win_stats else None),
            "window_paired_block_length_max": (
                win_stats.get("block_length_max") if win_stats else None),
            "window_paired_mean_lag1_autocorr": (
                win_stats.get("mean_lag1_autocorr") if win_stats else None),
            "window_paired_ppl_ratio": (
                float(math.exp(min(20.0, win_stats["mean_diff"])))
                if win_stats else None),
            "window_paired_p": (win_stats["p"] if win_stats else None),
            "window_paired_p_adj": None,
            "window_paired_n_windows_min": (
                win_stats["n_windows_min"] if win_stats else None),
            "window_paired_n_windows_max": (
                win_stats["n_windows_max"] if win_stats else None),
            "window_paired_enough": (
                bool(win_stats and win_stats["n_seeds"] >= MIN_PPL_WINDOW_SEEDS
                     and win_stats["n_windows_min"] >= MIN_PPL_WINDOWS)
                if win_stats else None),
            "window_paired_skip_reason": win_skip_reason,
            "window_paired_verdict": None,
            "verdict": None,
        })
    # 回退原因：把“为什么没用窗口对齐参考”显式落盘，避免读者默认已对齐。
    mismatch_variants = [st["variant"] for st in staged
                         if not all(st["matched"])]
    if use_window:
        fallback_reason = None
    elif not wspec:
        fallback_reason = "ngram_baseline.json 缺少 window_spec（历史产物）"
    elif wm_ref is None:
        fallback_reason = "window-matched PPL 缺失或非数值"
    else:
        fallback_reason = ("存在与窗口规格不一致的 run（eval_ctx/eval_iters/"
                           f"val_windows 任一字段不符）：{mismatch_variants}")
    if not rows:
        return {"split": ng_split, "reference_model": ref_name,
                "ngram_ppl": ref_ppl, "rows": [],
                "reference_source": ref_source,
                "ngram_ppl_full_split": ref_ppl_full,
                "ngram_ppl_window_matched": wm_ref,
                "reference_model_window_matched": wm_ref_name,
                "window_reference_values": wm_ref_map,
                "window_matched_selection": wm_sel or None,
                "window_spec": wspec or None,
                "window_paired_reference_model": ng_win_model,
                "window_paired_available": bool(ng_win is not None),
                "fallback_reason": fallback_reason,
                "incomparable": incomparable,
                "note": "没有与 n-gram 同 split 的神经评测结果可比"}
    pads = holm_bonferroni([r.get("p") for r in rows])
    for r, padj in zip(rows, pads):
        r["p_adj"] = padj
        if r["n"] < MIN_SEEDS:
            r["verdict"] = f"种子不足（n={r['n']}<{MIN_SEEDS}）"
        elif padj is None:
            r["verdict"] = "无法检验"
        elif padj < ALPHA and r["mean_ppl"] < ref_ppl:
            r["verdict"] = "显著优于统计基线"
        elif padj < ALPHA:
            r["verdict"] = "显著劣于统计基线（严重）"
        else:
            r["verdict"] = "无显著差异（对照 MDE）"
    # 窗口级配对也是独立检验族：Holm 校正只作用于有逐窗口数据的变体。
    win_rows = [r for r in rows if r.get("window_paired_p") is not None]
    win_pads = holm_bonferroni([r.get("window_paired_p") for r in win_rows])
    for r, padj in zip(win_rows, win_pads):
        r["window_paired_p_adj"] = padj
        if r.get("window_paired_enough") is False:
            r["window_paired_verdict"] = (
                "证据不足（种子或窗口数低于门槛），仅报点估计/区间")
        elif padj is None:
            r["window_paired_verdict"] = "无法检验"
        elif padj < ALPHA and r["window_paired_mean_diff"] < 0:
            r["window_paired_verdict"] = "窗口级显著优于统计基线（Holm 校正后）"
        elif padj < ALPHA:
            r["window_paired_verdict"] = "窗口级显著劣于统计基线（Holm 校正后）"
        else:
            r["window_paired_verdict"] = "窗口级无显著差异（Holm 校正后）"
    note = ("n-gram 无种子方差：逐种子 PPL 与同一常数配对 = 单样本检验；"
            "本对比为补充性/探索性证据，不替代变体间受控对比")
    if use_window:
        note = ("参考值取自与神经评测逐位置对齐的窗口目标 token"
                f"（窗口规格 ctx={wspec.get('ctx')}, iters={wspec.get('iters')}, "
                f"共 {wspec.get('n_tokens')} 个评分 token）；" + note)
    return {"split": ng_split, "reference_model": ref_name,
            "ngram_ppl": ref_ppl, "rows": rows, "incomparable": incomparable,
            "reference_source": ref_source,
            "ngram_ppl_full_split": ref_ppl_full,
            "ngram_ppl_window_matched": wm_ref,
            "reference_model_window_matched": wm_ref_name,
            "window_reference_values": wm_ref_map,
            "window_matched_selection": wm_sel or None,
            "window_spec": wspec or None,
            "window_paired_reference_model": ng_win_model,
            "window_paired_available": bool(ng_win is not None),
            "fallback_reason": fallback_reason,
            # Holm 家族只包含实际参与检验（p 非 None）的行；n<MIN_SEEDS
            # 的变体不占家族大小，渲染时不能把总行数说成校正家族大小。
            "holm_family_size": sum(1 for r in rows
                                    if r.get("p") is not None),
            "n_rows": len(rows),
            "note": note}


def _probe_oof(analysis: dict, key: str, subkey: str = None):
    """从机制分析 JSON 取线性探针主特征集的 out-of-fold 指标。

    subkey 用于嵌套指标（如 oof.f1.macro / oof.f1_tuned.macro）。
    """
    lp = _as_dict(_as_dict(analysis).get("linear_probe"))
    primary = lp.get("primary")
    fs = _as_dict(_as_dict(lp.get("feature_sets")).get(
        primary if isinstance(primary, str) else ""))
    val = _as_dict(fs.get("oof")).get(key)
    if subkey is not None:
        return _as_dict(val).get(subkey)
    return val


# 机制判别的简单学习基线对照（补充性/探索性）：门控 logit 是语言建模的
# 副产品、训练时从未见过 hit 标签；线性探针在同一批记录上用 hit 标签监督
# 训练（嵌套、按样例分组并按标签分层的 CV；λ/阈值只在内层 OOF 选，
# 且 λ 按与主检验对齐的 PR-AUC/AUROC，见 lega/probe.py）。
# Δ = 探针 − 门控：>0 表示简单监督基线更好（门控的“判别优越性”不成立），
# ≈0 表示门控与最简线性模型打平，<0 才是门控在判别上显著更好。
LINEAR_PROBE_AP_METRIC = "PR-AUC差(探针−门控)"
LINEAR_PROBE_AUROC_METRIC = "AUROC差(探针−门控)"


def linear_probe_baseline_comparison(runs: dict) -> dict:
    """门控 logit vs 线性探针简单监督基线的逐种子配对对照。

    没有这个对照时，“门控 logit 的 AUROC/PR-AUC 高”可能只说明廉价特征
    （log max_p、log n、qterm）本身线性可分，复杂门控结构的贡献无从谈起。
    探针是**有标签监督**的简单基线：探针超过门控属预期，门控必须在
    主指标上不被探针显著超越，才有资格谈“更能判断该不该写入”。
    单统计量规则（max_p）一并列出，作为最低档次的规则系统对照。
    """
    rows, tests_ap, tests_aud, missing = [], [], [], []
    runs = runs if isinstance(runs, dict) else {}
    for v in sorted(runs):
        if v not in VARIANTS:
            continue
        m = _variant_seed_map(runs, v)
        per_seed, has_analysis = [], False
        for s in _sorted_seeds(m):
            slot = m.get(s)
            an = slot.get("analysis") if isinstance(slot, dict) else None
            if not isinstance(an, dict):
                continue
            has_analysis = True
            lp = _as_dict(an.get("linear_probe"))
            primary = lp.get("primary")
            fs = _as_dict(_as_dict(lp.get("feature_sets")).get(
                primary if isinstance(primary, str) else ""))
            oof = _as_dict(fs.get("oof"))
            probe_auprc = _rate_or_none(oof.get("auprc"))
            probe_auroc = _rate_or_none(oof.get("auroc"))
            if probe_auprc is None or probe_auroc is None:
                continue
            probe_f1_fixed = _rate_or_none(
                _as_dict(oof.get("f1")).get("macro"))
            probe_f1_tuned = _rate_or_none(
                _as_dict(oof.get("f1_tuned")).get("macro"))
            per_seed.append({
                # 无 seed 字段（_seed_key 返回 None）时按缺种子信息处理。
                # 非整数身份（bool/小数）同样记 None，绝不截断成别的种子号。
                "seed": _integral_value(s),
                "probe_auprc": probe_auprc,
                "probe_auroc": probe_auroc,
                "probe_f1_fixed": probe_f1_fixed,
                "probe_f1_tuned": probe_f1_tuned,
                "gate_auprc": _rate_or_none(an.get("auprc_logit")),
                "gate_auroc": _rate_or_none(an.get("auroc_logit")),
                "rule_auprc": _rate_or_none(an.get("auprc_maxp")),
                "rule_auroc": _rate_or_none(an.get("auroc_maxp")),
                "base_rate": _rate_or_none(an.get("hit_base_rate")),
                "probe_l2": fs.get("selected_l2"),
                "probe_features": fs.get("features"),
                "folds": lp.get("folds"),
                "primary_detection_metric": lp.get("primary_detection_metric"),
                "imbalance_level": lp.get("imbalance_level"),
            })
        if not per_seed:
            if has_analysis:
                missing.append({"variant": v,
                                "reason": "analysis JSON 缺少 linear_probe"
                                          "（旧产物或 --no_probe）"})
            continue

        def _ms(key):
            vals = [r[key] for r in per_seed if r.get(key) is not None]
            return mean_std(vals) if vals else (None, None)

        probe_ap_m, probe_ap_s = _ms("probe_auprc")
        gate_ap_m, gate_ap_s = _ms("gate_auprc")
        probe_auc_m, probe_auc_s = _ms("probe_auroc")
        gate_auc_m, gate_auc_s = _ms("gate_auroc")
        rule_ap_m, _ = _ms("rule_auprc")
        rule_auc_m, _ = _ms("rule_auroc")
        probe_f1_fixed_m, _ = _ms("probe_f1_fixed")
        probe_f1_tuned_m, _ = _ms("probe_f1_tuned")
        base_m, _ = _ms("base_rate")

        def _test(key_p, key_g, metric):
            a, b = [], []
            for r in per_seed:
                if r.get(key_p) is None or r.get(key_g) is None:
                    continue
                a.append(r[key_p])
                b.append(r[key_g])
            if not a:
                return None
            res = paired_t_test(a, b)
            res.update({"candidate": v, "reference": "linear_probe",
                        "metric": metric, "lower_is_better": False})
            return _enrich_paired(res, a, b)

        t_ap = _test("probe_auprc", "gate_auprc", LINEAR_PROBE_AP_METRIC)
        t_aud = _test("probe_auroc", "gate_auroc", LINEAR_PROBE_AUROC_METRIC)
        if t_ap:
            tests_ap.append(t_ap)
        if t_aud:
            tests_aud.append(t_aud)
        d_ap = [r["probe_auprc"] - r["gate_auprc"] for r in per_seed
                if r.get("gate_auprc") is not None]
        d_auc = [r["probe_auroc"] - r["gate_auroc"] for r in per_seed
                 if r.get("gate_auroc") is not None]
        primary = next((r.get("primary_detection_metric") for r in per_seed
                        if r.get("primary_detection_metric")), None)
        rows.append({
            "variant": v, "n_seeds": len(per_seed),
            "seeds": [r["seed"] for r in per_seed],
            "imbalance_level": next((r.get("imbalance_level") for r in per_seed
                                     if r.get("imbalance_level")), None),
            "hit_base_rate": base_m,
            "primary_detection_metric": primary,
            "features": next((r.get("probe_features") for r in per_seed
                              if r.get("probe_features")), None),
            "probe_l2": next((r.get("probe_l2") for r in per_seed
                              if r.get("probe_l2") is not None), None),
            "folds": next((r.get("folds") for r in per_seed
                           if r.get("folds") is not None), None),
            "rule_auprc": rule_ap_m, "probe_auprc": probe_ap_m,
            "probe_auprc_std": probe_ap_s,
            "gate_auprc": gate_ap_m, "gate_auprc_std": gate_ap_s,
            "rule_auroc": rule_auc_m, "probe_auroc": probe_auc_m,
            "probe_auroc_std": probe_auc_s,
            "gate_auroc": gate_auc_m, "gate_auroc_std": gate_auc_s,
            "delta_auprc": (sum(d_ap) / len(d_ap)) if d_ap else None,
            "delta_auroc": (sum(d_auc) / len(d_auc)) if d_auc else None,
            "probe_f1_fixed": probe_f1_fixed_m,
            "probe_f1_tuned": probe_f1_tuned_m,
        })
    tests = _apply_holm(tests_ap + tests_aud)
    for t in tests:
        # _apply_holm 的结论文案面向“候选优于参照”，此处 Δ=探针−门控，
        # 语义相反：Δ>0 是简单监督基线更好，必须改写以免方向误导。
        n = t.get("n") or 0
        if t.get("p") is None:
            t["verdict"] = "无法检验（样本或方差退化），不作判定"
        elif n < MIN_SEEDS:
            t["verdict"] = f"种子不足（n={n}<{MIN_SEEDS}），不作判定"
        elif t.get("p_adj") is not None and t["p_adj"] < ALPHA:
            t["verdict"] = ("⚠ 简单监督基线更好：门控的判别优越性不成立"
                            if t["mean_diff"] > 0 else
                            "门控显著优于最简监督线性基线")
        else:
            t["verdict"] = ("无显著差异（对照 MDE）——打平不等于门控结构"
                            "无用，但也不构成“判别优越性”的证据")
    # 行内证据在 Holm 校正与自定义 verdict 之后回填（校正对象是 tests 里的
    # 同一批 dict；若在循环里先取值会拿到校正前的 None）。
    by_key = {(t.get("candidate"), t.get("metric")): t for t in tests}
    for r in rows:
        t_ap = by_key.get((r["variant"], LINEAR_PROBE_AP_METRIC))
        t_aud = by_key.get((r["variant"], LINEAR_PROBE_AUROC_METRIC))
        r["p_adj_auprc"] = (t_ap or {}).get("p_adj")
        r["p_adj_auroc"] = (t_aud or {}).get("p_adj")
        r["delta_auprc_ci95"] = (t_ap or {}).get("mean_diff_ci95")
        r["delta_auroc_ci95"] = (t_aud or {}).get("mean_diff_ci95")
        primary_detection = r.get("primary_detection_metric")
        if primary_detection == "PR-AUC":
            r["verdict"] = (t_ap or {}).get("verdict")
        elif primary_detection == "AUROC":
            r["verdict"] = (t_aud or {}).get("verdict")
        else:
            # “AUROC 与 PR-AUC 并报”（以及旧产物缺失主指标）：不能默认只
            # 看 AUROC。两个检验可能一个显著、另一个不显著，单取任一个都
            # 会隐藏另一指标上的反证；并列渲染两者的判定。
            parts = []
            if t_ap is not None:
                parts.append("PR-AUC：" + str(t_ap.get("verdict") or "-"))
            if t_aud is not None:
                parts.append("AUROC：" + str(t_aud.get("verdict") or "-"))
            r["verdict"] = "；".join(parts) if parts else "-"
    return {
        "metrics": {"ap": LINEAR_PROBE_AP_METRIC,
                    "auroc": LINEAR_PROBE_AUROC_METRIC},
        "rows": rows, "tests": tests, "missing": missing,
        "note": ("线性探针=有标签监督的简单基线（同批记录、嵌套分组 CV、"
                 "按 hit 标签分层；λ 在内层 OOF 按主指标选、阈值在内层 OOF "
                 "按 macro-F1 选，最终 OOF 不参与选参）；"
                 "门控 logit 为语言建模副产品、训练未见 hit 标签。"
                 "Δ=探针−门控：>0 属预期（判别能力本可由线性模型从廉价"
                 "统计量学得），门控必须在主指标上不被探针显著超越才有"
                 "资格谈“结构优越性”。单统计量规则（max_p）为最低档次"
                 "对照。本对比为补充性/探索性证据，不替代预注册的确认性"
                 "检验。"),
    }


def _render_baseline_sections(results_dir: str, smoke=None,
                              ngram_cmp: dict = None,
                              probe_cmp: dict = None,
                              variants=None) -> list:
    """基线合理性小节：简单统计基线 + 基线超参数调优证据。

    科研方法问题：如果复杂方法只与“随手设一个学习率”的弱基线比较，
    涨点可能全部来自超参数红利。本小节渲染两类证据：
      1. 简单基线（一元/二元/插值二元平滑 n-gram）在同一 held-out split 上的 PPL，
         给出绝对难度参照，并（ngram_cmp 提供时）逐变体做同 split 的
         单样本配对检验，回答“神经模型是否显著优于无梯度基线”；
      2. 基线（standard）在验证集上的学习率/权重衰减网格调优结果与
         “默认是否已接近最优”判定。
    缺少调优证据时显式警告“未调优基线”，而不是让读者默认基线已调好。
    smoke 可由调用方传入；为 None 时自行读取 repro.json。
    """
    if smoke is None:
        smoke = False
        rfp = os.path.join(results_dir, "repro.json")
        if os.path.exists(rfp):
            try:
                with open(rfp, "r", encoding="utf-8") as f:
                    doc = json.load(f)
                # 合法 JSON 但非对象（数组/标量）按缺 repro 处理；smoke 只接受
                # 真正的 bool true，字符串 "false" 不能伪装成冒烟运行。
                smoke = (doc.get("smoke") is True
                         if isinstance(doc, dict) else False)
            except (OSError, ValueError):
                smoke = False
    lines = ["", "## 基线合理性（简单基线 + 调优证据）", ""]
    ngram = None
    nfp = os.path.join(results_dir, "ngram_baseline.json")
    if os.path.exists(nfp):
        try:
            with open(nfp, "r", encoding="utf-8") as f:
                ngram = json.load(f)
            if not isinstance(ngram, dict):
                ngram = None
        except (OSError, ValueError):
            ngram = None
    if ngram:
        ppl_map = ngram.get("ppl")
        ppl_map = ppl_map if isinstance(ppl_map, dict) else {}
        uni = ppl_map.get("unigram")
        bi = ppl_map.get("bigram")
        interp = ppl_map.get("interpolated_bigram")
        split = ngram.get("split")
        lines.append(
            f"- 简单统计基线（train 拟合、{split} 评测；无梯度；α/λ 只在 "
            f"val 上按 mean NLL 选，仍属零训练基线）："
            f"一元平滑 PPL={_fmt_scalar_maybe(uni)}、"
            f"二元平滑 PPL={_fmt_scalar_maybe(bi)}、"
            f"插值二元 PPL={_fmt_scalar_maybe(interp)}、"
            f"val 选出的最强简单模型={ngram.get('selected_model') or '-'}"
            f"（训练 token {_int_or_zero(ngram.get('train_tokens')):,}、"
            f"评测 token {_int_or_zero(ngram.get('eval_tokens')):,}、"
            f"词表 {ngram.get('vocab_size')}）。主科学对比仍是神经变体之间；"
            "该基线只回答“绝对难度有多高、神经模型是否显著优于统计基线”。")
        # 新产物优先展示“在同一批窗口 token 上重新选参”的参考；旧产物
        # 只有全 val 选参的 ppl_window_matched，保持兼容。
        wm_sel = ngram.get("ppl_window_matched_selected")
        wm_sel = wm_sel if isinstance(wm_sel, dict) else {}
        # 比较表实际采用哪张窗口参考值表由 ngram_baseline_comparison 决定
        # （selected 表存在但缺少 selection 元数据时会被忽略）。渲染必须
        # 复用同一选择，否则同一份报告里表格与说明会引用不同基线。
        cmp_window_values = ((ngram_cmp or {}).get("window_reference_values")
                             if isinstance(ngram_cmp, dict) else None)
        if isinstance(cmp_window_values, dict):
            wm = cmp_window_values
        else:
            wm = wm_sel or ngram.get("ppl_window_matched")
        wm = wm if isinstance(wm, dict) else {}
        wsel = ngram.get("window_matched_selection")
        wsel = wsel if isinstance(wsel, dict) else {}
        wspec = ngram.get("window_spec")
        wspec = wspec if isinstance(wspec, dict) else {}
        # 只有比较表确实采用了逐位置对齐参考（全部 run 的窗口规格一致）时，
        # 这些数字才真的打在“神经评测实际使用的窗口 token”上；一旦回退到全
        # split 参考（历史产物/规格不符），同样的数字只是按声明规格算出的
        # 参考值，不能宣称与神经评测同 token——必须显式区分，否则基线证据
        # 被高估。
        token_matched = ((ngram_cmp or {}).get("reference_source")
                         == "window_matched")
        if wm and wspec and token_matched:
            lines.append(
                f"- 逐位置对齐口径（同 split 还不够，两边必须打在同一批 token "
                f"上）：在神经评测实际使用的窗口目标 token 上"
                f"（ctx={wspec.get('ctx')}、iters={wspec.get('iters')}、"
                f"n_eval={wspec.get('n_eval')}、"
                f"共 {_int_or_zero(wspec.get('n_tokens')):,} 个评分 token），"
                f"一元 PPL={_fmt_scalar_maybe(wm.get('unigram'))}、"
                f"二元 PPL={_fmt_scalar_maybe(wm.get('bigram'))}、"
                f"插值二元 PPL={_fmt_scalar_maybe(wm.get('interpolated_bigram'))}"
                f"（{('α/λ 在同一批窗口 token 上重新选择，selected=' + str(wsel.get('selected_model'))) if wm_sel and wsel else '与全 split 口径同一选定 α/λ'}）。")
        elif wm and wspec:
            lines.append(
                f"- 窗口规格参考值（按 n-gram 侧声明规格 ctx={wspec.get('ctx')}、"
                f"iters={wspec.get('iters')}、n_eval={wspec.get('n_eval')}、"
                f"共 {_int_or_zero(wspec.get('n_tokens')):,} 个评分 token）："
                f"一元 PPL={_fmt_scalar_maybe(wm.get('unigram'))}、"
                f"二元 PPL={_fmt_scalar_maybe(wm.get('bigram'))}、"
                f"插值二元 PPL={_fmt_scalar_maybe(wm.get('interpolated_bigram'))}"
                f"（{('α/λ 在同一批窗口 token 上重新选择，selected=' + str(wsel.get('selected_model'))) if wm_sel and wsel else '与全 split 口径同一选定 α/λ'}）。⚠ 本次神经评测未全部满足该"
                "窗口规格，实际比较已回退全 split 参考（见下方比较表），"
                "这些数字不代表神经评测实际使用的 token 集合。")
        elif ngram.get("window_spec_error"):
            lines.append(f"- ⚠ n-gram 窗口对齐不可用："
                         f"{ngram['window_spec_error']}")
    else:
        lines.append("- 简单统计基线：未运行（`python -m lega.ngram`，"
                     "结果写入 `results/<tag>/ngram_baseline.json`）。")
    # 逐变体 vs 简单基线的同 split 配对对比：只并排报两个数不构成
    # “复杂模型优于简单基线”的证据，必须给出比值、校正后 p 与判定。
    cmp = ngram_cmp or {}
    if cmp.get("rows"):
        ref = cmp.get("reference_model")
        matched = cmp.get("reference_source") == "window_matched"
        basis = ("参考值与神经评测逐位置对齐（同一批窗口目标 token）"
                 if matched else
                 "参考值为全 split PPL（未逐位置对齐，见下方说明）")
        lines += [
            "",
            f"- 神经变体 vs 简单统计基线（{ref}，split={cmp.get('split')}；"
            f"{basis}；n-gram 无种子方差 → 逐种子 PPL 与同一常数配对做单样本"
            f"检验，Holm 校正（家族 "
            f"{cmp.get('holm_family_size', len(cmp['rows']))} 个变体）；"
            "属补充性/探索性对比）：",
            "",
            "| 变体 | n | 神经 PPL(mean) | 统计基线 PPL | 比值 | 相对降低 | p_adj | 判定 |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for r in cmp["rows"]:
            ratio = (_fmt4(r.get("ratio"))
                     if r.get("ratio") is not None else "-")
            red = (f"{r['reduction_pct']:.1f}%"
                   if r.get("reduction_pct") is not None else "-")
            padj = (_fmt4(r.get("p_adj"))
                    if r.get("p_adj") is not None else "-")
            lines.append(
                f"| {r.get('variant')} | {r.get('n')} | "
                f"{_fmt_scalar_maybe(r.get('mean_ppl'))} | "
                f"{_fmt_scalar_maybe(cmp.get('ngram_ppl'))} | {ratio} | {red} "
                f"| {padj} | {r.get('verdict')} |")
        lines += ["",
                  "  注：比值 <1 且判定“显著优于统计基线”才说明神经模型真的"
                  "比无梯度基线好；两边必须同 split 且尽量同 token——神经 "
                  "PPL 只在 eval_iters 个窗口的目标 token 上计算，若本表未用"
                  "窗口对齐参考（reference_source=full_split），说明存在"
                  "与窗口规格不一致的 run，比值含 token 集合差异带来的偏移。"
                  "本对比未预注册，只是补充证据，不替代变体间受控配对检验。"]
        if not matched and cmp.get("fallback_reason"):
            lines.append(f"  ⚠ 未使用逐位置对齐参考：{cmp['fallback_reason']}"
                         "（重跑 n-gram 基线或对齐 eval 规格后即可对齐）。")
        win_rows = [r for r in cmp["rows"]
                    if r.get("window_paired_mean_diff") is not None]
        if win_rows:
            lines += [
                "",
                "  **逐窗口配对（探索性；比“与常数做单样本检验”多利用窗口"
                "抽样信息）**",
                "",
                "| 变体 | 种子 | 窗口数 | ΔNLL(神经−n-gram) | 95% CI | "
                "块bootstrap 95% CI | PPL比 | p_adj(Holm) | 判定 |",
                "|---|---|---|---|---|---|---|---|---|",
            ]
            for r in win_rows:
                ci = r.get("window_paired_ci95") or [None, None]
                ci_s = (f"[{ci[0]:+.4f}, {ci[1]:+.4f}]"
                        if ci[0] is not None and ci[1] is not None else "-")
                bci = r.get("window_paired_block_ci95") or [None, None]
                bci_s = (f"[{bci[0]:+.4f}, {bci[1]:+.4f}]"
                         if bci[0] is not None and bci[1] is not None else "-")
                ratio = (f"{r['window_paired_ppl_ratio']:.4f}"
                         if r.get("window_paired_ppl_ratio") is not None else "-")
                padj = (_fmt4(r.get("window_paired_p_adj"))
                        if r.get("window_paired_p_adj") is not None else "-")
                lines.append(
                    f"| {r.get('variant')} | {r.get('window_paired_n_seeds')} | "
                    f"{r.get('window_paired_n_windows_min')}–"
                    f"{r.get('window_paired_n_windows_max')} | "
                    f"{_fmt4(r.get('window_paired_mean_diff'))} | {ci_s} | "
                    f"{bci_s} | {ratio} | {padj} | "
                    f"{r.get('window_paired_verdict')} |")
            lines += ["", "  注: ΔNLL<0 表示神经模型在相同窗口上更好；分层 "
                      "bootstrap 同时重采样训练种子与评测窗口，窗口独立"
                      "重采样忽略空间相关；块bootstrap 用移动块（块长 "
                      "n^(1/3)）保留相邻窗口相关，两者并报，以更宽区间"
                      "解读稳健性，故只作探索性证据。该检验要求"
                      "神经与 n-gram 使用同一批窗口序号；不满足时主表仍"
                      "保留，但逐窗口行不生成。"]
    elif cmp.get("error"):
        lines.append(f"- ⚠ 简单基线对比不可用：{cmp['error']}")
    elif cmp.get("note") and not cmp.get("rows"):
        lines.append(f"- ⚠ {cmp['note']}")
    for item in cmp.get("incomparable") or []:
        excl = item.get("excluded_seeds") or []
        lines.append(
            f"- ⚠ 变体 `{item.get('variant')}` 的评测 split="
            f"{item.get('eval_splits')} 与 n-gram split={cmp.get('split')} "
            "不一致"
            + (f"的种子 {excl} " if excl else "：")
            + "未纳入比较（不同切片的 PPL 不可比；用 "
            "`--ngram_split auto` 或对齐后再比较）。")
    tune = None
    tfp = os.path.join(results_dir, "baseline_tuning.json")
    if os.path.exists(tfp):
        try:
            with open(tfp, "r", encoding="utf-8") as f:
                tune = json.load(f)
        except (OSError, ValueError):
            tune = None
        if not isinstance(tune, dict):
            tune = None
    if tune:
        best = tune.get("selected")
        best = best if isinstance(best, dict) else {}
        dflt = tune.get("default_trial")
        dflt = dflt if isinstance(dflt, dict) else {}
        gap = tune.get("selected_minus_default_val_loss")
        sel_seeds = tune.get("selection_seeds") or [tune.get("seed")]
        sel_seeds = sel_seeds if isinstance(sel_seeds, list) else []
        grid = tune.get("grid") or []
        grid = [c for c in grid if isinstance(c, dict)] if isinstance(grid, list) else []
        def _n_distinct(key):
            vals = set()
            for c in grid:
                v = c.get(key)
                try:
                    hash(v)
                except TypeError:
                    continue
                vals.add(v)
            return len(vals) if grid else "-"

        n_lr, n_wd = _n_distinct("lr"), _n_distinct("weight_decay")
        trials = tune.get("trials")
        n_trials = tune.get("n_trials")
        if n_trials is None:
            n_trials = (len(grid) or (len(trials)
                                      if isinstance(trials, list) else "-"))
        std = best.get("val_loss_best_std")
        lines.append(
            f"- 基线调优（变体 `{tune.get('variant', 'standard')}`，"
            f"网格 {n_lr} 档 lr × {n_wd} 档 wd = "
            f"{n_trials} "
            f"组配置、选择用 {len(sel_seeds)} 个种子、trial 步数 "
            f"{tune.get('steps', '-')}（与正式训练同步数与否以此为准）、"
            f"验证集选择、测试集未使用）：最优 lr={best.get('lr')}"
            + (f"、weight_decay={best.get('weight_decay')}"
               if best.get("weight_decay") is not None else "")
            + f"，最优 val_loss={_fmt4(best.get('val_loss_best', best.get('val_loss')))}"
            + (f" ± {_fmt4(std)}（跨选择种子）" if std is not None else "")
            + f"；默认 lr={dflt.get('lr')} 的 val_loss="
            f"{_fmt4(dflt.get('val_loss_best', dflt.get('val_loss')))}"
            f"（最优−默认={_fmt4(gap)}）。")
        if len(sel_seeds) < 2:
            lines.append("- ⚠ 网格选择只用 1 个种子：val_loss 的种子噪声可能让"
                         "“最优配置”在下一次抽样中翻转（winner's curse）；"
                         "多组接近时应改用 `--tune_seeds`（多带几个种子按均值"
                         "选择）后重跑，不要据此宣称基线已充分调优。")
        diag = tune.get("selection_diagnostics")
        if isinstance(diag, dict) and _int_or_zero(diag.get("n_seeds")) >= 2:
            n_sel = _int_or_zero(diag.get("n_seeds"))
            support = _int_or_zero(diag.get("selected_support_count"))
            frac = _num_or_none(diag.get("selected_support_fraction"))
            if frac is not None:
                lines.append(
                    f"- 选择稳定性：最优配置在 {support}/{n_sel} 个选择种子上"
                    f"实际取得最低 val_loss（支持率 {frac:.0%}）；逐种子赢家与"
                    "完整配对值见 `baseline_tuning.json` 的 "
                    "`selection_diagnostics`。")
                if support < n_sel:
                    lines.append(
                        "- ⚠ 均值最优并非逐种子稳定最优：至少一个种子上另一"
                        "配置更好，说明排序受种子噪声影响；差距接近噪声时"
                        "不能把该配置当作“已验证的最佳超参数”。")
            ru = diag.get("runner_up")
            if isinstance(ru, dict):
                margin = _num_or_none(ru.get("selected_minus_runner_up_mean"))
                dstd = _num_or_none(ru.get("paired_delta_std"))
                dmean = _num_or_none(ru.get("paired_delta_mean"))
                noise = ru.get("within_seed_noise")
                lines.append(
                    "- 最优−次优：次优 lr="
                    f"{ru.get('lr')}"
                    + (f"、weight_decay={ru.get('weight_decay')}"
                       if ru.get("weight_decay") is not None else "")
                    + f"，均值差 {_fmt4(margin)}；逐种子（次优−最优）配对差 "
                    f"mean={_fmt4(dmean)}、SD={_fmt4(dstd)}"
                    + ("——差距在一个种子噪声 SD 内（不稳定）"
                       if noise is True else
                       "——差距大于一个种子噪声 SD"
                       if noise is False else ""))
                if noise is True:
                    lines.append(
                        "- ⚠ 最优与次优的差距未超过逐种子配对波动：更换种子"
                        "即可能翻转选择；此时应优先看两者在正式种子上的"
                        "下游指标，而不是把“网格最优”当成真实优势。")
            loo = diag.get("leave_one_seed_out")
            if isinstance(loo, dict):
                n_folds = _int_or_zero(loo.get("n_folds"))
                agree = _int_or_zero(loo.get("agreement_count"))
                gain = _num_or_none(loo.get("heldout_gain_vs_default_mean"))
                gstd = _num_or_none(loo.get("heldout_gain_vs_default_std"))
                w, t, l = (loo.get("wins_vs_default"),
                           loo.get("ties_vs_default"),
                           loo.get("losses_vs_default"))
                agree_txt = ""
                if n_folds:
                    try:
                        agree_txt = f"（{agree / n_folds:.0%}）"
                    except (OverflowError, ZeroDivisionError):
                        # 外部 JSON 的合法大整数可精确解析，但转 float 做
                        # 比例会溢出；比例按缺失渲染，不能让汇总崩溃。
                        agree_txt = ""
                lines.append(
                    f"- 留一种子复选：{n_folds} 折中 {agree} 折重新选出的配置"
                    "与全量选择一致"
                    + agree_txt
                    + "；在被留出的种子上，LOO 选择相对默认配置的 held-out "
                    f"val_loss 平均增益 {_fmt4(gain)}"
                    + (f" ± {_fmt4(gstd)}" if gstd is not None else "")
                    + (f"（胜/平/负 {w}/{t}/{l}；正=优于默认）"
                       if w is not None else "")
                    + "。LOO 折不独立，仅作稳定性描述。")
                if (n_folds and agree < n_folds
                        and gain is not None and gain <= 0):
                    lines.append(
                        "- ⚠ 留一复选既翻转选择、又未在留出种子上优于默认："
                        "网格调优的泛化收益缺乏稳定性证据，正式结论应以"
                        "新种子复验为准。")
        elif len(sel_seeds) >= 2:
            lines.append(
                "- ⚠ 调优报告缺少 `selection_diagnostics`（旧格式）：无法核对"
                "逐种子支持率、最优−次优噪声差距与留一种子复选；请重跑 "
                "`--tune_baseline` 后再据此宣称基线已稳定调优。")
        if tune.get("default_is_optimal") is True:
            lines.append("- ✅ 默认超参数即为网格最优/在容差内：基线的对比不受"
                         "“选了不利学习率”的质疑。全流水线所有变体使用同一"
                         "（已调优的）配置，受控变量保持一致。")
        elif tune.get("default_is_optimal") is False:
            lines.append("- 默认超参数并非网格最优：全流水线已改用上述最优"
                         "配置训练**所有**变体（受控变量一致），因此与基线的"
                         "对比不是“调好的新方法 vs 没调过的基线”。")
        else:
            lines.append("- ⚠ 默认组合的 trial 失败或缺失，无法判定默认是否"
                         "近优；请核对 `baseline_tuning.json` 的 trials 记录。")
        # 调优只覆盖 tune.variant（默认 standard）；qgate/null 是“已有工作
        # 代表”，若也直接用同一配置，就不能把对比写成“打败了各自最佳调优
        # 的已有方法”。这是设计上的受控对照，但结论范围必须写清楚。
        tuned_variant = tune.get("variant", "standard")
        prior = [v for v in ("qgate", "null")
                 if isinstance(variants, (list, tuple, set)) and v in variants]
        untuned = [v for v in prior if v != tuned_variant]
        if untuned:
            lines.append(
                f"- ⚠ 调优范围：`baseline_tuning.json` 只调优了 "
                f"`{tuned_variant}`；已有工作代表 {', '.join('`'+v+'`' for v in untuned)} "
                "直接继承同一组 lr/weight_decay，未各自做网格调优。因此"
                "“优于已有工作”只能理解为**同超参受控对照**，不是“优于"
                "各自最佳调优版本”；若差距接近 MDE/种子噪声，不能据此宣称"
                "方法优越。需要强结论时应分别调优各基线（并同时调优本方"
                "方法）后在新种子上复验。")
    else:
        lines.append("- 基线调优：未运行。"
                     + ("（冒烟模式不要求。）" if smoke else
                        "⚠ 未提供基线调优证据：若基线（standard）没有经过"
                        "至少一档学习率/权重衰减的网格验证，方法涨点可能只是"
                        "超参数红利。请用 `run_all.py --tune_baseline` 或在"
                        "`results/<tag>/baseline_tuning.json` 提供证据。"))
    # 机制判别的“最简监督线性模型”对照：没有它，“门控 logit 判别力高”
    # 无法排除“廉价特征本身线性可分”。探针 OOF 有标签监督、门控无标签；
    # Δ=探针−门控，>0 属预期，门控不被显著超越才谈得上结构优越性。
    probe = probe_cmp or {}
    if probe.get("rows"):
        lines += [
            "",
            "- 机制判别的简单学习基线（线性探针，OOF；与门控 logit 逐种子"
            "配对、Holm 校正；属补充性/探索性对比）：",
            "",
            "| 变体 | n | 命中基线率 | 特征集 | PR-AUC规则(max_p) | "
            "PR-AUC探针(OOF) | PR-AUC门控(logit) | AUROC探针(OOF) | "
            "AUROC门控(logit) | F1探针(macro,0.5) | F1探针(macro,调优) | "
            "ΔPR-AUC(探针−门控) | p_adj(PR-AUC) | ΔAUROC(探针−门控) | "
            "p_adj(AUROC) | 判定 |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]

        def _delta_ci(r, key):
            delta = r.get(key)
            ci = r.get(key + "_ci95")
            if delta is None:
                return "-"
            text = f"{delta:+.3f}"
            if (isinstance(ci, (list, tuple)) and len(ci) == 2
                    and all(_finite_number(x) is not None for x in ci)):
                text += f" [{_finite_number(ci[0]):+.3f}, "
                text += f"{_finite_number(ci[1]):+.3f}]"
            return text

        for r in probe["rows"]:
            feat = r.get("features")
            feat_s = "+".join(str(x) for x in feat) if feat else "-"
            lines.append(
                f"| {r.get('variant')} | {r.get('n_seeds')} | "
                f"{_fmt4(r.get('hit_base_rate'))} | {feat_s} | "
                f"{_fmt4(r.get('rule_auprc'))} | "
                f"{_fmt4(r.get('probe_auprc'))} | "
                f"{_fmt4(r.get('gate_auprc'))} | "
                f"{_fmt4(r.get('probe_auroc'))} | "
                f"{_fmt4(r.get('gate_auroc'))} | "
                f"{_fmt4(r.get('probe_f1_fixed'))} | "
                f"{_fmt4(r.get('probe_f1_tuned'))} | "
                f"{_delta_ci(r, 'delta_auprc')} | "
                f"{_fmt4(r.get('p_adj_auprc'))} | "
                f"{_delta_ci(r, 'delta_auroc')} | "
                f"{_fmt4(r.get('p_adj_auroc'))} | "
                f"{r.get('verdict') or '-'} |")
        lines += ["", f"  {probe.get('note', '')}"]
        lines.append(
            f"  注：命中基线率=hit 正类率，也是 PR-AUC 的随机猜基线；"
            f"主指标={next((r.get('primary_detection_metric') for r in probe['rows'] if r.get('primary_detection_metric')), 'n/a')}"
            f"（正类率 <5% 时 PR-AUC 为主，AUROC 因负类基数大而乐观）；"
            f"探针 λ 用嵌套分组 CV、在内层 OOF 上按主指标（PR-AUC 或 AUROC）"
            f"选择，阈值在内层 OOF 上按 macro-F1 选择（见 analysis_*.json 的 "
            f"linear_probe.per_fold / per_l2）；固定 0.5 阈值 F1 仅作与门控 "
            f"g>0.5 的同口径对照，调优阈值 F1 才代表简单基线被充分调参后的"
            f"决策能力。PR-AUC 探针/OOF 与门控在同一样本上计算，Δ 的 CI 与 "
            f"p_adj 见 significance.json。")
    if probe.get("missing"):
        names = "、".join(f"{x.get('variant')}（{x.get('reason')}）"
                          for x in probe["missing"])
        lines.append(f"- ⚠ 线性探针简单基线缺失：{names}；"
                     "机制主张缺少“有标签监督的最简线性模型”对照，"
                     "不能用“门控 logit 判别力高”替代。")
    lines.append("")
    return lines


def _fmt_scalar_maybe(x) -> str:
    f = _finite_number(x)
    if f is not None:
        return f"{f:.3f}"
    # 旧产物里可能有 "n/a" 这类说明性字符串，原样保留；bool/容器等
    # 非数值类型不能显示成 1.000/"[...]" 伪观测。
    return x if isinstance(x, str) else "-"


def _dedupe_advice(items, limit=6):
    """按 (轴, 建议) 去重改进方向并截断（不同变体的同一模式只留一条）。"""
    out, seen = [], set()
    for label, r in items:
        key = (r.get("axis"), r.get("suggestion"))
        if key in seen:
            continue
        seen.add(key)
        out.append((label, r))
        if len(out) >= limit:
            break
    return out


def _render_advice(md, advice, title, max_items=6):
    """渲染“坏案例驱动的改进方向”小节（证据 → 建议 → 下一步实验）。"""
    advice = _dedupe_advice(advice, limit=max_items)
    if not advice:
        return
    md += ["", title, ""]
    for label, r in advice:
        md.append(f"- **[{r.get('axis')}·{r.get('priority')}]**（{label}）"
                  f"{r.get('trigger')}")
        md.append(f"  - 建议：{r.get('suggestion')}")
        md.append(f"  - 下一步：{r.get('next_experiment')}")


# ---------------------------------------------------------------- 曲线诊断与权衡

def diagnose_training_curves(chain, data_dir: str, results_dir: str) -> dict:
    """对每个 (变体, 种子) run 做训练曲线诊断并落盘汇总。

    只看最终数值会掩盖训练动态：训练降/验证升=过拟合，两者都不降=欠拟合或
    数据管道问题，NaN=训练失效。诊断结果进入 summary，另存
    curve_diagnosis.json 供逐 run 复核。
    """
    log_vocab = None
    meta_path = os.path.join(data_dir, "meta.json")
    if os.path.exists(meta_path):
        try:
            import math as _math
            with open(meta_path, "r", encoding="utf-8") as f:
                vocab = _integral_value(json.load(f)["vocab_size"])
            if vocab is None or vocab <= 0:
                raise ValueError(f"meta.vocab_size 非法: {vocab!r}")
            log_vocab = _math.log(vocab)
        except (OSError, ValueError, KeyError, TypeError):
            log_vocab = None
    diag = {}
    for v, seed, vdir, _suffix in chain:
        d = diagnose_run_dir(vdir, log_vocab=log_vocab)
        diag.setdefault(v, {})[str(seed)] = d
    if diag:
        save_json(diag, os.path.join(results_dir, "curve_diagnosis.json"))
        n_bad = sum(1 for vs in diag.values() for d in vs.values()
                    if d["status"] not in ("正常收敛", "数据不足"))
        print(f"[curves] 训练曲线诊断完成：{n_bad} 个 run 需要关注"
              f" -> {os.path.join(results_dir, 'curve_diagnosis.json')}")
    return diag


def _num_or_none(v):
    """JSON 数值安全转换：缺失/bool/非数值/非有限值统一返回 None。"""
    return _finite_number(v)


def _eff_cell(entry, key, ctx):
    """latency.json 变体条目里的结构与运行时字段（flops 取 total，mem 取量测值）。"""
    if not isinstance(entry, dict):
        return None
    eff = entry.get("efficiency")
    eff = eff if isinstance(eff, dict) else {}
    if key in ("flops", "total_flops", "gate_flops"):
        by_ctx = eff.get("flops_per_tok")
        if not isinstance(by_ctx, dict):
            return None
        cell = by_ctx.get(str(ctx))
        if not isinstance(cell, dict):
            return None
        field = ("gate_extra_flops_per_tok" if key == "gate_flops"
                 else "total_flops_per_tok")
        return _num_or_none(cell.get(field))
    if key == "mem":
        by_ctx = eff.get("peak_mem_mb_prefill")
        return (_num_or_none(by_ctx.get(str(ctx)))
                if isinstance(by_ctx, dict) else None)
    if key == "params":
        return _num_or_none(eff.get("n_params") or entry.get("n_params"))
    if key == "params_mb":
        return _num_or_none(eff.get("params_mb"))
    if key == "kv_mem":
        # 优先取 latency.json 已按 ctx 落盘的值；旧格式只有 per-token 时
        # 按 ctx 线性换算（KV cache 大小随上下文长度线性增长）。
        by_ctx = eff.get("kv_cache_mb_at_ctx")
        if isinstance(by_ctx, dict):
            v = _num_or_none(by_ctx.get(str(ctx)))
            if v is not None:
                return v
        per_tok = _num_or_none(eff.get("kv_cache_mb_per_tok"))
        if per_tok is not None:
            # ctx 可能来自外部 JSON，是合法但超出 float 上限的大整数；换算
            # 结果也可能溢出成 inf。两者都不能进入汇总/判定，按缺失处理。
            try:
                out = per_tok * float(ctx)
            except (TypeError, ValueError, OverflowError):
                return None
            return out if math.isfinite(out) else None
    return None


def _rel_pct(base, cand):
    """cand 相对 base 的百分比变化；任一缺失/base=0/结果不可表示时返回 None。

    两端都是有限值不代表百分比可表示：base 极小或 cand−base 溢出时会得到
    ±inf，写入汇总行后 JSON 只能把它变成 null，而判定逻辑仍可能按 inf 参与
    max()/比较。这里统一在入口按缺失处理。
    """
    base, cand = _num_or_none(base), _num_or_none(cand)
    if base is None or cand is None or base == 0:
        return None
    try:
        out = (cand - base) / abs(base) * 100.0
    except (OverflowError, ZeroDivisionError):
        return None
    return out if math.isfinite(out) else None


def _lat_noise_pct(entry, metric, ctx, bs="1"):
    """latency.json 中某运行时指标的测量相对散布（spread_pct）；缺失返回 None。

    spread_pct=(p90-p10)/median，由 lega.latency 逐次计时给出。旧格式
    latency.json 没有该字段时返回 None（按无噪声信息处理）。
    """
    if not isinstance(entry, dict):
        return None
    key = ("prefill_ms_spread_pct" if metric == "prefill"
           else "decode_ms_spread_pct")
    by_ctx = entry.get(key)
    if not isinstance(by_ctx, dict):
        return None
    cell = by_ctx.get(str(ctx))
    if metric == "prefill":
        cell = cell.get(str(bs)) if isinstance(cell, dict) else None
    return _num_or_none(cell)


def _noise_floor_pct(base_noise, cand_noise):
    """两端的测量散布按方和根合成（比值差的方差近似为相对方差之和）。

    只有两端都有散布信息时才给出噪声下限：一侧缺失时无法判断"差值是否
    小于测量噪声"，按未过滤的原口径处理，避免用单侧散布低估/高估噪声。
    外部 JSON 里的 bool/字符串/NaN/Inf 先统一按缺失处理，不能直接参与
    ``< 0`` 比较而抛 TypeError。
    """
    base_noise = _num_or_none(base_noise)
    cand_noise = _num_or_none(cand_noise)
    if (base_noise is None or cand_noise is None
            or base_noise < 0.0 or cand_noise < 0.0):
        return None
    try:
        floor = math.sqrt(base_noise * base_noise + cand_noise * cand_noise)
    except (OverflowError, ValueError):
        return None
    return floor if math.isfinite(floor) and floor >= 0.0 else None


# RULER 收益以 EM 绝对百分点计，与 PPL 的相对百分比单位不同：收支比规则
# 不能混算，RULER 单独设阈值（EM 提升通常比 PPL 百分比更难获得，倍数放宽
# 到 10×；下限 0.5 pp 避免收益趋近 0 时比值判定失效）。
RULER_COST_RATIO = 10.0
RULER_RATIO_FLOOR_PP = 0.5


def tradeoff_verdict(benefit_pct, p_adj, cost_pct, alpha=ALPHA,
                     n_seeds=None, ruler_benefit_pp=None, ruler_p_adj=None,
                     ruler_n_seeds=None) -> str:
    """涨点-增幅权衡判定：不看增幅的涨点没有意义；收益必须覆盖两个主指标。

    两个主指标：PPL（benefit_pct=相对基线的百分比改善，正=更好）与 RULER
    平均 EM（ruler_benefit_pp=相对基线的绝对百分点改善，正=更好）。本项目
    预注册主张横跨两者（H1 PPL、H4 长上下文检索），只看 PPL 会把“PPL 持平
    但 RULER 显著提升”的真涨点误判为无效，也会漏掉“PPL 涨、RULER 显著
    退步”的严重信号。

    规则（显式、可审计）：
      * 两个主指标都无数据或增幅缺失 → 数据不足；
      * 参与判定的指标种子不足（给定 n<MIN_SEEDS）→ 无法判定，与全表口径一致；
      * 某一主指标显著劣化：若另一主指标同时显著改善 → 权衡（必须显式
        说明取舍）；否则 → 不可行；
      * 任一主指标显著改善 → 有涨点：综合增幅（各指标相对基线的最大增幅）
        >50% → 需审视；按显著者中最不利的收益套用收支比规则（PPL 5×，
        RULER 单独 10× 且下限 0.5 pp，单位不同不混算）→ 收益-增幅比不利；
        否则 → 可接受，并注明依据哪个指标；
      * 有正向涨点但不显著 → 说服力有限（增幅>10% 判不可行）。
    微小涨点伴随巨大增幅的改进在这里会被明确拒绝。
    """
    # 外部/旧汇总可能把数值写成 bool/字符串/NaN；先在入口统一成有限数值或
    # None，避免 False 被 `False < alpha` 当成显著 p=0，也避免字符串比较
    # 直接抛 TypeError。
    benefit_pct = _num_or_none(benefit_pct)
    cost_pct = _num_or_none(cost_pct)
    ruler_benefit_pp = _num_or_none(ruler_benefit_pp)
    p_adj = _num_or_none(p_adj)
    ruler_p_adj = _num_or_none(ruler_p_adj)
    if (benefit_pct is None and ruler_benefit_pp is None) or cost_pct is None:
        return "数据不足"
    raw_n, raw_ruler_n = n_seeds, ruler_n_seeds
    n_seeds = _integral_value(n_seeds)
    ruler_n_seeds = _integral_value(ruler_n_seeds)
    bad_n = []
    if n_seeds is None and raw_n is not None:
        bad_n.append(f"PPL n={raw_n!r}")
    if ruler_n_seeds is None and raw_ruler_n is not None:
        bad_n.append(f"RULER n={raw_ruler_n!r}")
    if bad_n:
        return f"种子数非法（{'、'.join(bad_n)}），无法判定"
    lacks = []
    if benefit_pct is not None and n_seeds is not None and n_seeds < MIN_SEEDS:
        lacks.append(f"PPL n={n_seeds}")
    if (ruler_benefit_pp is not None and ruler_n_seeds is not None
            and ruler_n_seeds < MIN_SEEDS):
        lacks.append(f"RULER n={ruler_n_seeds}")
    if lacks:
        return f"种子不足（{'、'.join(lacks)}<{MIN_SEEDS}），无法判定"
    ppl_sig = p_adj is not None and p_adj < alpha
    rul_sig = ruler_p_adj is not None and ruler_p_adj < alpha
    ppl_up = benefit_pct is not None and benefit_pct > 0 and ppl_sig
    rul_up = (ruler_benefit_pp is not None and ruler_benefit_pp > 0
              and rul_sig)
    ppl_down = benefit_pct is not None and benefit_pct < 0 and ppl_sig
    rul_down = (ruler_benefit_pp is not None and ruler_benefit_pp < 0
                and rul_sig)
    if ppl_down and rul_up:
        return (f"权衡：PPL 显著退步 {benefit_pct:.1f}% 但 RULER 显著提升 "
                f"+{ruler_benefit_pp:.2f} pp（增幅 {cost_pct:.0f}%，"
                "检索目标改善，须说明 PPL 取舍）")
    if rul_down and ppl_up:
        return (f"权衡：PPL 显著改善 {benefit_pct:.1f}% 但 RULER 显著退步 "
                f"{ruler_benefit_pp:.2f} pp（增幅 {cost_pct:.0f}%，"
                "核心检索目标受损，须说明取舍）")
    if ppl_down:
        extra = (f"，且 RULER 显著退步 {ruler_benefit_pp:+.2f} pp"
                 if rul_down else "")
        return (f"不可行：PPL 显著差于基线（{benefit_pct:.1f}%）"
                f"{extra}")
    if rul_down:
        return f"不可行：RULER 显著差于基线（{ruler_benefit_pp:+.2f} pp）"
    if ppl_up or rul_up:
        if cost_pct > 50:
            return f"需审视：显著但增幅 {cost_pct:.0f}%"
        if ppl_up and cost_pct > 5 * max(benefit_pct, 0.5):
            return (f"收益-增幅比不利：PPL 收益 {benefit_pct:.1f}% vs "
                    f"增幅 {cost_pct:.0f}%")
        if rul_up and cost_pct > RULER_COST_RATIO * max(ruler_benefit_pp,
                                                        RULER_RATIO_FLOOR_PP):
            return (f"收益-增幅比不利：RULER 收益 +{ruler_benefit_pp:.2f} pp vs "
                    f"增幅 {cost_pct:.0f}%")
        basis = "PPL" if ppl_up else "RULER"
        if ppl_up and rul_up:
            basis = "PPL+RULER"
        return "可接受" if basis == "PPL" else f"可接受（依据 {basis} 显著涨点）"
    any_pos = ((benefit_pct is not None and benefit_pct > 0)
               or (ruler_benefit_pp is not None and ruler_benefit_pp > 0))
    if any_pos:
        if cost_pct > 10:
            return f"不可行：涨点不显著且增幅 {cost_pct:.0f}%"
        return "说服力有限：涨点未通过显著性检验"
    if cost_pct > 10:
        return f"不可行：无涨点且增幅 {cost_pct:.0f}%"
    return "说服力有限：无涨点且差异不显著"


def build_summary(results_dir: str, audit: dict = None,
                  data_report: dict = None, preloaded: dict = None) -> dict:
    """汇总 PPL / RULER / 参数量等指标为 Markdown + CSV + JSON。

    多种子口径：每个变体的每个指标先按种子聚合为 Mean ± Std；变体间比较使用
    同一批种子上的配对 t 检验（同种子共享初始化与数据顺序，配对消除公共方差）。
    消融矩阵显式列出全因子 8 个组合的覆盖情况，缺失组合会被标注。
    data_report（可选）来自数据目录的防泄漏自检报告，汇总为独立小节。
    preloaded（可选）为 {results 下结果文件路径: 已解析 JSON} 缓存：坏案例
    阶段已解析过的 eval/ruler 文件在此共用同一份结果。
    """
    audit = audit if isinstance(audit, dict) else None
    data_report = data_report if isinstance(data_report, dict) else None

    def load(p):
        if preloaded and p in preloaded:
            return preloaded[p]
        if not os.path.exists(p):
            return None
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            # 坏/半写入的结果文件按“未运行”处理，不能毁掉整份汇总。
            return None

    # --- “本次 run”范围限定：results/<tag> 会被同一 tag 的多次运行复用
    # （改种子/变体集、--skip_* 等），聚合按 repro.json 记录的本次
    # variants/seeds 过滤，不同实验的观测不混入同一均值/配对检验；
    # 没有 repro.json（直接调用/手工构建目录）时保持全量发现。
    repro_doc = load(os.path.join(results_dir, "repro.json"))
    # 合法 JSON 但非对象（数组/标量）的 repro.json 按“无 repro”处理，
    # 不能让汇总在 .get 上崩溃。
    repro = repro_doc if isinstance(repro_doc, dict) else {}
    scope_variants = repro.get("variants") if isinstance(
        repro.get("variants"), list) else None
    scope_seeds = repro.get("seeds") if isinstance(
        repro.get("seeds"), list) else None

    def in_run_scope(fp, d):
        # 变体名必须是可哈希字符串：外部 JSON 的数组/对象既不能作为 runs
        # 的键，也不是合法的变体标识，按损坏产物跳过。
        if not isinstance(d, dict) or not isinstance(d.get("variant"), str):
            return False
        if scope_variants is not None and d["variant"] not in scope_variants:
            return False
        if scope_seeds is not None and _seed_key(fp, d) not in scope_seeds:
            return False
        return True

    # --- 发现所有 run：（变体, 种子） -> {eval, ruler, analysis}
    runs = {}

    def slot(v, s):
        return runs.setdefault(v, {}).setdefault(
            s, {"eval": None, "ruler": None, "analysis": None})

    for fp in _result_files(results_dir, "eval", scope_seeds):
        d = load(fp)
        if in_run_scope(fp, d):
            slot(d["variant"], _seed_key(fp, d))["eval"] = d
    for fp in _result_files(results_dir, "ruler", scope_seeds):
        d = load(fp)
        if in_run_scope(fp, d):
            slot(d["variant"], _seed_key(fp, d))["ruler"] = d
    for fp in _result_files(results_dir, "analysis", scope_seeds):
        if "records" in os.path.basename(fp):
            continue
        d = load(fp)
        if in_run_scope(fp, d):
            slot(d["variant"], _seed_key(fp, d))["analysis"] = d

    # 推理期因果病灶消融（每 (变体, 种子) 一个 ablation_*.json）；单独存放，
    # 不混入 runs 槽位——它不是训练变体，也不参与受控变量审计。
    ablations = {}
    ablation_scope = repro.get("ablation")
    if isinstance(ablation_scope, dict):
        # 本轮显式关闭消融时，绝不复用旧 ablation 文件冒充本轮证据。
        # executed 必须是真正的 bool true：字符串 "false"/1 经 truthiness
        # 会打开复用，把上一轮的病灶结果算进本轮结论。
        allow_ablation = ablation_scope.get("executed") is True
        ablation_variants = (ablation_scope.get("variants")
                             if isinstance(ablation_scope.get("variants"), list)
                             else [])
    else:
        # 旧 repro 没有该字段：保持向后兼容的全量发现。
        allow_ablation = True
        ablation_variants = None
    for fp in _result_files(results_dir, "ablation", scope_seeds):
        d = load(fp)
        if not in_run_scope(fp, d):
            continue
        if not allow_ablation:
            continue
        if (ablation_variants is not None
                and d["variant"] not in ablation_variants):
            continue
        ablations.setdefault(d["variant"], {})[_seed_key(fp, d)] = d

    variants = sorted(v for v in runs if v in RUNNABLE_VARIANTS)
    eval_splits = set()
    invalid_eval_split = False
    for v in variants:
        for s in _variant_seed_map(runs, v).values():
            if s.get("eval"):
                try:
                    eval_splits.add(s["eval"].get("split", "val"))
                except TypeError:
                    invalid_eval_split = True

    # 汇总运行时指标时必须使用所有变体共有的同一上下文档，不允许各取自己的最大 ctx。
    lat = load(os.path.join(results_dir, "latency.json"))
    if not isinstance(lat, dict):
        lat = {}
    lat_variants = lat.get("variants")
    lat_variants = lat_variants if isinstance(lat_variants, dict) else {}
    lat_entries = {v: (lat_variants.get(v)
                       if isinstance(lat_variants.get(v), dict) else None)
                   for v in variants}

    def _lat_metric_at(e, key, ctx):
        if not isinstance(e, dict) or ctx is None:
            return None
        by_ctx = e.get(key)
        if not isinstance(by_ctx, dict):
            return None
        return _num_or_none(by_ctx.get(str(ctx)))

    def _prefill_at(e, ctx):
        if not isinstance(e, dict) or ctx is None:
            return None
        by_ctx = e.get("prefill_ms")
        if not isinstance(by_ctx, dict):
            return None
        cell = by_ctx.get(str(ctx))
        return _num_or_none(cell.get("1")) if isinstance(cell, dict) else None

    common_ctxs = None
    for v in variants:
        e = lat_entries.get(v)
        if not isinstance(e, dict):
            continue
        prefill = e.get("prefill_ms")
        decode = e.get("decode_ms_per_tok")
        if not isinstance(prefill, dict) or not isinstance(decode, dict):
            continue
        usable = set()
        for c, by_bs in prefill.items():
            if (not isinstance(c, str)
                    or not isinstance(by_bs, dict)
                    or _num_or_none(by_bs.get("1")) is None
                    or _num_or_none(decode.get(c)) is None):
                continue
            # int("1_0")==10、int("040")==40，但这些非规范 key 在
            # _lat_metric_at 里按 str(ctx) 查找时并不存在；若拿来求公共
            # ctx，会选中一个没有真实量测的“最大上下文”。
            ctx = _integral_value(c)
            if ctx is None or c != str(ctx):
                continue
            usable.add(ctx)
        common_ctxs = usable if common_ctxs is None else (common_ctxs & usable)
    common_ctx = max(common_ctxs) if common_ctxs else None

    def _clock_extractor(key):
        return lambda d: d.get(key) if isinstance(d, dict) else None

    metric_specs = [
        ("val_ppl", lambda m: _stat_series(m, "eval", _ppl_of)),
        ("gate_mean", lambda m: _stat_series(m, "eval", _clock_extractor("gate_mean"))),
        ("val_loss", lambda m: _stat_series(m, "eval", _clock_extractor("val_loss"))),
        # 评估不确定性（单 run 内窗口抽样）与训练数据覆盖率：这两个量决定
        # “数值差异是否值得解释”，必须与点估计一起进入 summary.json/csv。
        ("ppl_ci_rel_pct", lambda m: _stat_series(m, "eval", _ppl_ci_rel_pct)),
        ("epochs_seen", lambda m: _stat_series(m, "eval", _eval_scalar("epochs_seen"))),
        # model_best.pt 的选型步：同一最大预算下，各变体可能在差异很大的步
        # 被 val 选中。记录后才能审计“同预算”之外的选型差异。
        ("ckpt_step", lambda m: _stat_series(m, "eval", _eval_scalar("ckpt_step"))),
        ("selection_step_fraction", lambda m: _stat_series(
            m, "eval", _eval_scalar("selection_step_fraction"))),
        ("eval_tokens", lambda m: _stat_series(
            m, "eval", _eval_scalar("val_tokens_evaluated"))),
        ("eval_windows", lambda m: _stat_series(
            m, "eval", _eval_scalar("val_windows"))),
        ("ruler_avg", lambda m: _stat_series(m, "ruler", _ruler_avg_of)),
        ("ruler_avg_micro", lambda m: _stat_series(
            m, "ruler", _ruler_avg_micro)),
        ("ruler_min_n", lambda m: _stat_series(m, "ruler", _ruler_min_cell_n)),
        # NIAH 位置分层：聚合 EM 会把早期高分与中部塌陷互相抵消；分层
        # 指标进入 summary.json/csv，Markdown 汇总另设小节（旧产物缺失）。
        ("ruler_depth_early", lambda m: _stat_series(
            m, "ruler", lambda d: _ruler_depth_means(d).get("early"))),
        ("ruler_depth_middle", lambda m: _stat_series(
            m, "ruler", lambda d: _ruler_depth_means(d).get("middle"))),
        ("ruler_depth_late", lambda m: _stat_series(
            m, "ruler", lambda d: _ruler_depth_means(d).get("late"))),
        ("ruler_depth_early_minus_late", lambda m: _stat_series(
            m, "ruler", _ruler_depth_gap)),
    ]
    ruler_tasks = RULER_TASKS
    for task in ruler_tasks:
        metric_specs.append(
            (f"ruler_{task}",
             (lambda t: lambda m: _stat_series(
                 m, "ruler", lambda d: _ruler_task_means(d).get(t)))(task)))
    # fwe 召回率仅作诊断单列：recall 不惩罚“把上下文词全部列出”的灌水
    # 策略，不能进入预注册的 RULER 平均（主口径为严格集合 EM）。
    metric_specs.append(
        ("ruler_fwe_recall",
         lambda m: _stat_series(m, "ruler", _ruler_fwe_recall)))
    # fwe precision/F1 同为诊断：precision 才惩罚“多答”，与 recall 并排
    # 才能区分“没找到高频词”与“找对但灌水/漏答”；均不进主平均。
    metric_specs.append(
        ("ruler_fwe_precision",
         lambda m: _stat_series(
             m, "ruler", lambda d: _ruler_task_means(d, "precision").get("fwe"))))
    metric_specs.append(
        ("ruler_fwe_f1",
         lambda m: _stat_series(
             m, "ruler", lambda d: _ruler_task_means(d, "f1").get("fwe"))))
    for key in ("auroc_logit", "auroc_qterm", "misleading_rate",
                "misleading_given_conc", "p_hit_given_conc",
                "auprc_logit", "auprc_qterm", "hit_base_rate",
                "brier_g", "ece_g", "ece_g_adaptive",
                "brier_g_constant", "brier_g_skill",
                "brier_g_reliability", "brier_g_resolution",
                "brier_g_uncertainty", "brier_g_within_score",
                "brier_g_within_cov", "brier_g_binning"):
        metric_specs.append(
            (key, (lambda k: lambda m: _stat_series(
                m, "analysis", _clock_extractor(k)))(key)))
    # 门控决策点 F1（g>0.5）：显式只取 macro 进汇总表（不平衡下对稀有类
    # 最诚实的口径）；micro/weighted 与逐类明细在 analysis_*.json 中
    # 的 gate_f1 字段并列完整保留，避免“只写 F1 不注明平均口径”。
    metric_specs.append(
        ("gate_f1_macro", lambda m: _stat_series(
            m, "analysis",
            lambda d: _as_dict(d.get("gate_f1")).get("macro"))))
    # 恒写（全判 hit）规则的 macro F1：门控决策的“稻草人”对照。
    # 不平衡下门控 F1 若不明显高于该固定规则，就没有决策价值。
    metric_specs.append(
        ("gate_f1_macro_always_write", lambda m: _stat_series(
            m, "analysis",
            lambda d: _as_dict(
                _as_dict(d.get("gate_f1_trivial")).get("always_write")
            ).get("macro"))))
    # 线性探针（有标签监督的简单基线）OOF 指标：进入 summary.json/csv，
    # 供“门控 vs 最简线性模型”的补充性对照审计（渲染见基线合理性小节）。
    metric_specs.append(
        ("probe_auprc", lambda m: _stat_series(
            m, "analysis", lambda d: _probe_oof(d, "auprc"))))
    metric_specs.append(
        ("probe_auroc", lambda m: _stat_series(
            m, "analysis", lambda d: _probe_oof(d, "auroc"))))
    metric_specs.append(
        ("probe_f1_fixed", lambda m: _stat_series(
            m, "analysis",
            lambda d: _probe_oof(d, "f1", "macro"))))
    metric_specs.append(
        ("probe_f1_tuned", lambda m: _stat_series(
            m, "analysis",
            lambda d: _probe_oof(d, "f1_tuned", "macro"))))
    metric_specs.append(
        ("probe_brier", lambda m: _stat_series(
            m, "analysis", lambda d: _probe_oof(d, "brier"))))
    metric_specs.append(
        ("probe_ece", lambda m: _stat_series(
            m, "analysis", lambda d: _probe_oof(d, "ece"))))

    rows = []
    for v in variants:
        m = _variant_seed_map(runs, v)
        row = {"variant": v, "variant_cn": VARIANT_CN.get(v, v),
               "n_seeds": len(m)}
        for key, fn in metric_specs:
            st = fn(m)
            if st:
                row[key] = st["mean"]
                row[f"{key}_std"] = st["std"]
                row[f"{key}_n"] = st["n"]
            else:
                row[key] = row[f"{key}_std"] = row[f"{key}_n"] = None
        ev_example = next((s["eval"] for s in m.values() if s.get("eval")), None)
        row["params"] = (ev_example or {}).get("n_params")
        row["gate_extra"] = (ev_example or {}).get("n_gate_extra", 0)
        row["n_gate_per_head"] = (ev_example or {}).get("n_gate_per_head")
        row["n_heads"] = (ev_example or {}).get("n_heads")
        row["n_layers"] = (ev_example or {}).get("n_layers")
        row["d_head"] = (ev_example or {}).get("d_head")
        row["val_ppl_clamped"] = any(
            (s.get("eval") or {}).get("val_ppl_clamped") for s in m.values())
        # 训练/评估覆盖的“最坏种子”兜底值：均值可能掩盖某个种子只跑了
        # 极少数据或只评测了极少窗口的情况，汇总警告以最小值触发。
        evals_v = [s["eval"] for s in m.values() if s.get("eval")]
        eps = [x for x in (_num_or_none(e.get("epochs_seen"))
                           for e in evals_v) if x is not None]
        row["epochs_seen_min"] = min(eps) if eps else None
        toks = [x for x in (_num_or_none(e.get("val_tokens_evaluated"))
                            for e in evals_v) if x is not None]
        row["eval_tokens_min"] = min(toks) if toks else None
        row["val_loss_ci95"] = (ev_example or {}).get("val_loss_ci95")
        row["val_ppl_ci95"] = (ev_example or {}).get("val_ppl_ci95")
        rns = [_ruler_min_cell_n(s.get("ruler")) for s in m.values()
               if s.get("ruler") is not None]
        rns = [n for n in rns if n is not None]
        row["ruler_min_n_min"] = min(rns) if rns else None
        ruler_payloads = [s.get("ruler") for s in m.values()
                          if isinstance(s.get("ruler"), dict)]
        raci = _ruler_macro_absolute_ci(ruler_payloads)
        row["ruler_avg_ci95"] = raci["ci95"] if raci else None
        row["ruler_avg_ci_n_seeds"] = raci["n_seeds"] if raci else None
        # 运行时指标（取全部变体的最大公共 ctx、batch=1）；不受种子影响，测一次
        e = lat_entries.get(v)
        if e and common_ctx is not None:
            row["prefill_ms"] = _prefill_at(e, common_ctx)
            row["decode_ms_per_tok"] = _lat_metric_at(
                e, "decode_ms_per_tok", common_ctx)
            row["decode_prefill_ctx"] = _lat_metric_at(
                e, "decode_prefill_ctx", common_ctx)
            row["latency_ctx"] = common_ctx
        rows.append(row)

    # 运行时基线：优先取 standard_sdpa，缺失时退回显式 standard。
    lat_all = lat_variants
    base_ms, base_name = None, None
    for cand in ("standard_sdpa", "standard"):
        if _prefill_at(lat_all.get(cand), common_ctx) is not None:
            base_ms, base_name = _prefill_at(lat_all.get(cand), common_ctx), cand
            break
    for r in rows:
        if (base_ms and base_ms > 0 and r.get("prefill_ms") is not None
                and r["prefill_ms"] > 0
                and r["variant"] not in ("standard", "standard_sdpa")):
            # 两端有限但比值不可表示时按缺失处理，不能把 inf 写进汇总行。
            rel = _rel_pct(base_ms, r["prefill_ms"])
            if rel is not None:
                r["prefill_rel_pct"] = round(rel, 2)

    # --- 配对显著性检验（同种子）：PPL 与 RULER 平均
    # 单因素消融（去一法 + 全因子增量）与整体对比分开构造；B-only/C-only
    # 不再与 Full 配对（那会同时改变 A+C / A+B，不能归因给单个组件）。
    registered = _as_dict(load_registered_hypotheses())
    sf_audit = single_factor_audit()
    if not sf_audit["ok"]:
        print("[run_all] ⚠ 单因素消融审计未通过："
              + "；".join(sf_audit["problems"]))
    pairs = build_comparison_pairs(runs, registered)
    tests = []
    tests += _paired_comparisons(
        runs, pairs, "PPL", lambda slot: _ppl_of(slot.get("eval")),
        lower_better=True)
    # PPL 是比值尺度：算术均值差会被高 PPL 种子主导。log-PPL 配对给出
    # 几何均值比（exp(Δ)），作为独立敏感性家族；预注册主判定仍是原始 PPL。
    log_tests = _paired_comparisons(
        runs, pairs, "PPL(log)", lambda slot: _log_ppl_of(slot.get("eval")),
        lower_better=True)
    for t in log_tests:
        md = t.get("mean_diff")
        ci = t.get("mean_diff_ci95")
        # log-PPL 差是两个有限 log 值之差，但 exp(Δ) 仍可能超出 float 上限
        # （例如候选 PPL=1e300、参照 PPL=1e-300）。比值不可表示时记为缺失，
        # 不能抛 OverflowError 让整份汇总失败；原始 log 差与 p 值仍保留。
        ratio_overflow = False
        t["ppl_ratio"] = None
        if md is not None and math.isfinite(float(md)):
            try:
                t["ppl_ratio"] = math.exp(float(md))
            except OverflowError:
                ratio_overflow = True
        t["ppl_ratio_ci95"] = None
        if (isinstance(ci, (list, tuple)) and len(ci) == 2
                and all(isinstance(x, (int, float)) and math.isfinite(x)
                        for x in ci)):
            try:
                t["ppl_ratio_ci95"] = [math.exp(float(ci[0])),
                                       math.exp(float(ci[1]))]
            except OverflowError:
                ratio_overflow = True
        t["ppl_ratio_overflow"] = bool(ratio_overflow)
        t["ppl_delta_pct"] = None
        if t["ppl_ratio"] is not None:
            delta_pct = (t["ppl_ratio"] - 1.0) * 100.0
            # exp(Δ) 本身可表示时，再乘 100 仍可能溢出；派生百分比按缺失
            # 处理，不能把 inf 写进 significance.json。
            if math.isfinite(delta_pct):
                t["ppl_delta_pct"] = delta_pct
        t["exploratory"] = True
    tests += log_tests
    tests += _paired_comparisons(
        runs, pairs, "RULER平均", lambda slot: _ruler_avg_of(slot.get("ruler")),
        lower_better=False)
    # RULER 逐任务分解（探索性）：综合平均可能让方向相反的任务互相抵消。
    # 每个任务单独作为指标家族做同种子配对 + Holm；综合平均仍是预注册
    # 主判定，逐任务结果只用于定位“效果来自哪个任务/是否被平均掩盖”。
    for task in RULER_TASKS:
        task_tests = _paired_comparisons(
            runs, pairs, f"RULER:{task}",
            (lambda t: lambda slot: _ruler_task_means(
                slot.get("ruler")).get(t))(task),
            lower_better=False)
        for t in task_tests:
            t["ruler_task"] = task
            t["exploratory"] = True
        tests += task_tests
    # 机制主张的正式检验族：AUROC(logit) − AUROC(qterm) 同种子配对差，
    # 以及不平衡口径下的 PR-AUC 配对差（两族分别 Holm 校正）
    tests += _auroc_diff_tests(runs)
    tests += _ap_diff_tests(runs)
    tests += _factorial_interaction_tests(runs)
    tests += _factorial_shapley_tests(runs)
    tests = _apply_holm(tests)
    # RULER 上下文长度剂量-反应（探索性）：只报跨长度平均会把“短涨长跌”
    # 与“全长度小涨”混成同一个数。独立 Holm 家族，不进入主检验表，也不
    # 改变预注册假设的确认性判定。
    length_analysis = _ruler_length_analysis(runs, pairs)
    # 把 vs standard 的长度剂量-反应摘要并入 summary.json/csv 的变体行，
    # 使“跨长度平均是否掩盖长上下文效应”可被程序消费（完整逐 ctx/逐种子
    # 明细仍在 significance.json）。没有配对（standard 本身/未跑 RULER）时
    # 保持 None，不伪造趋势。
    length_by_pair = {(p.get("candidate"), p.get("reference")): p
                      for p in length_analysis.get("pairs", [])
                      if isinstance(p, dict)}

    def _length_pp(v):
        f = _finite_number(v)
        if f is None:
            return None
        scaled = f * 100.0
        # 外部 significance.json 的有限值乘 100 仍可能溢出；None 比 JSON
        # Infinity（严格 JSON 无法解析）安全。
        return round(scaled, 4) if math.isfinite(scaled) else None

    for row in rows:
        pair = length_by_pair.get((row.get("variant"), "standard"))
        if not isinstance(pair, dict):
            continue
        per_ctx = pair.get("per_context") or []
        slope = pair.get("slope_per_doubling") or {}
        span = pair.get("long_minus_short") or {}
        row["ruler_length_contexts"] = pair.get("contexts")
        row["ruler_length_overall_delta_pp"] = _length_pp(
            pair.get("overall_mean_diff"))
        row["ruler_length_longest_delta_pp"] = _length_pp(
            per_ctx[-1].get("mean_diff") if per_ctx else None)
        row["ruler_length_long_minus_short_pp"] = _length_pp(
            span.get("mean_diff"))
        row["ruler_length_slope_pp_per_doubling"] = _length_pp(
            slope.get("mean_diff"))
        row["ruler_length_slope_p_adj"] = slope.get("p_adj")
        row["ruler_length_span_p_adj"] = span.get("p_adj")
        row["ruler_length_sign_flip"] = bool(
            pair.get("overall_vs_longest_sign_flip")
            or pair.get("overall_vs_slope_sign_flip")
            or pair.get("overall_vs_span_sign_flip"))
        row["ruler_length_masking"] = bool(pair.get("long_context_masking"))
        row["ruler_length_loo_sign_stable"] = pair.get(
            "slope_loo_sign_stable")
        row["ruler_length_balanced"] = pair.get("balanced_contexts")
    shapley_tests = [t for t in tests if t.get("shapley_factor")]
    # 推理期因果病灶消融：与训练期消融回答不同的因果问题（见 lega.ablate
    # 模块文档）。独立指标家族、独立 Holm，不混入上面的变体间配对表。
    causal_tests = _ablation_lesion_records(ablations)
    # 同一 checkpoint 内的 2^3 干预 → 推理期 Shapley；与训练期 Shapley 并列，
    # 用于检验“重新训练后的组件结构”是否也存在于最终前向。探索性证据。
    inference_shapley = _inference_shapley_tests(ablations)
    train_phi = {}
    for t in shapley_tests:
        key = _test_label_key(t.get("shapley_factor"))
        if key is not None and t.get("shapley_metric") == "PPL":
            train_phi[key] = _finite_number(t.get("mean_diff"))
    for t in inference_shapley:
        tr = train_phi.get(_test_label_key(t.get("shapley_factor")))
        ph = _finite_number(t.get("mean_diff"))
        t["training_phi_mean"] = tr
        t["training_inference_sign_agree"] = (
            None if tr is None or ph is None else bool(
                (tr > 0) == (ph > 0) or (tr == 0 and ph == 0)))
    # 推理期交互：与训练期 PPL交互 同尺度并列，检查“训练期交互结构是否在
    # 最终前向复现”。只作探索性证据，不进入确认性判定。
    inference_interactions = _inference_interaction_tests(ablations)
    train_inter = {}
    for t in tests:
        key = _test_label_key(t.get("interaction"))
        if key is not None and t.get("metric") == "PPL交互":
            train_inter[key] = _finite_number(t.get("mean_diff"))
    for t in inference_interactions:
        tr = train_inter.get(_test_label_key(t.get("interaction")))
        ph = _finite_number(t.get("mean_diff"))
        t["training_interaction_mean"] = tr
        t["training_inference_sign_agree"] = (
            None if tr is None or ph is None else bool(
                (tr > 0) == (ph > 0) or (tr == 0 and ph == 0)))
    # 参数匹配安慰剂（lega vs 同参数/同 FLOPs、但证据为噪声的 lega_placebo）。
    # 只有显式运行时才有这条对比；未运行时汇总页显式提示，不默认“容量已排除”。
    param_control_test = next(
        (t for t in tests
         if t.get("metric") == "PPL"
         and t.get("candidate") == PARAM_CONTROL_PAIR[0]
         and t.get("reference") == PARAM_CONTROL_PAIR[1]), None)
    if param_control_test is not None:
        param_control_test["capacity_control"] = True
        param_control_test["exploratory"] = True
    capacity_only_test = next(
        (t for t in tests
         if t.get("metric") == "PPL"
         and t.get("candidate") == PARAM_CONTROL_VARIANT
         and t.get("reference") == "qgate"), None)
    # 样例级配对（McNemar）：固定题面上的评测抽样不确定性，与上面的
    # 跨种子聚合 EM 检验互补；独立 Holm 家族、探索性证据。
    ruler_sample_tests = _ruler_sample_paired_tests(runs, pairs)
    # PPL 的窗口级配对：评估抽样不确定性，与种子级 PPL 检验互补。
    ppl_window_tests = _ppl_window_paired_tests(runs, pairs)
    comp = component_evidence(tests, audit=audit)
    # 单因素审计失败时，任何“组件有用”的统计证据都不能升级为确认性结论：
    # 对比本身不是单因素，归因对象不成立。证据数值保留供排查。
    for c in comp:
        c["single_factor_audit_ok"] = bool(sf_audit["ok"])
        if not sf_audit["ok"]:
            c["confirmatory_supported"] = False
            c["evidence_status"] = "exploratory"
    # 预注册假设判定：confirmatory（登记主张）与 exploratory（其余对比）。
    # 完整性锁链把“事后静默改写登记表”变成可检测事件：校验未通过时
    # 全部确认性判定被显式降级（apply_integrity_gate），不会假装边界仍成立。
    integrity = verify_registration()
    # 范围门：登记表完整还不够——本次 run 必须真的落在登记的实验范围内
    # （config 规模/数据/种子计划），否则同样不能给 confirmatory 判定。
    scope_check = registration_scope_check(registered, repro)
    # 设计符合性门：先验计划里的 (变体, 种子) 单元必须齐全；缺失只影响
    # 涉及该变体/指标端点的假设，不搞“一刀切”降级，也不能把缺失当无效应。
    design_plan = load_design_plan(results_dir)
    design_stages = _as_dict(
        _as_dict(repro.get("design")).get("stages"))
    design_conf = design_conformance(
        design_plan, repro, runs, audit=audit, stages=design_stages)
    design_hist = load_design_plan_history(results_dir)
    if len(design_hist) > 1:
        design_conf["plan_history_count"] = len(design_hist)
        design_conf.setdefault("warnings", []).append(
            f"本结果目录已有 {len(design_hist)} 份设计计划记录"
            "（重跑/改种子/改样本量）：历史见 design_plan_history.jsonl；"
            "当前判定只对应最新计划，不得把旧计划下的产物混入")
    hyp_eval = apply_design_gate(
        apply_budget_gate(
            apply_scope_gate(
                apply_integrity_gate(
                    evaluate_registered_hypotheses(registered, tests),
                    integrity),
                scope_check),
            audit),
        design_conf, design_plan)

    # --- Markdown
    cols = [("variant_cn", "变体"), ("n_seeds", "种子数"), ("params", "参数量"),
            ("gate_extra", "门控额外参数"), ("n_gate_per_head", "门控参数/头"),
            ("val_ppl", "PPL↓"),
            ("ppl_ci_rel_pct", "PPL评估CI±%"),
            ("epochs_seen", "训练覆盖(epoch)"),
            ("ckpt_step", "选型步"),
            ("selection_step_fraction", "选型步/计划"),
            ("ruler_avg", "RULER平均↑"),
            ("ruler_avg_ci95", "RULER评估CI"),
            ("ruler_niah_synth", "niah_synth↑"),
            ("ruler_niah_corpus", "niah_corpus↑"),
            ("ruler_niah_multikey", "multikey↑"),
            ("ruler_fwe", "fwe EM↑"),
            ("ruler_fwe_recall", "fwe recall↑"),
            ("ruler_fwe_precision", "fwe prec↑"),
            ("ruler_fwe_f1", "fwe F1↑"),
            ("ruler_vt", "vt↑"), ("prefill_ms", "Prefill ms↓"),
            ("prefill_rel_pct", "相对基线%"),
            ("decode_ms_per_tok", "Decode ms/tok↓"),
            ("gate_mean", "门控均值"), ("misleading_rate", "集中但看错率"),
            ("misleading_given_conc", "集中后看错率"),
            ("hit_base_rate", "命中基线率"),
            ("auroc_logit", "AUROC(logit)"), ("auroc_qterm", "AUROC(仅查询)"),
            ("auprc_logit", "PR-AUC(logit)↑"), ("auprc_qterm", "PR-AUC(仅查询)↑"),
            ("gate_f1_macro", "门控F1(macro)↑"),
            ("gate_f1_macro_always_write", "恒写基线F1(macro)")]

    def fmt_stat_cell(r, key):
        st = {"mean": r.get(key), "std": r.get(f"{key}_std"),
              "n": r.get(f"{key}_n")}
        # 评估 CI 半宽与 epoch 覆盖用 2 位小数：0.03 epoch 的差异不应被
        # 四舍五入成 0.0 而看起来像“没有训练”；CI ±12.5% 也需要小数位。
        cell = _fmt_stat(st, nd=2 if key in ("ppl_ci_rel_pct", "epochs_seen")
                         else 3)
        # PPL 被钳制在 exp(20) 的 run（发散）必须显式标注：不能与正常 PPL
        # 在同一列表头中静默比较（行级 val_ppl_clamped 由评测写入）。
        if key == "val_ppl" and r.get("val_ppl_clamped"):
            cell = f"{cell} ⚠含截断值"
        return cell

    def _fmt_scalar(x):
        if x is None:
            return "-"
        if isinstance(x, float):
            return f"{x:.3f}" if abs(x) < 10 else f"{x:,.0f}"
        return str(x)

    md = ["# LEGA 实验汇总", "",
          f"> 自动生成于 {time.strftime('%Y-%m-%d %H:%M:%S')}；"
          f"PPL/RULER 为跨随机种子的 Mean ± Std（变体间同种子配对，"
          f"显著性经 Holm–Bonferroni 校正），运行时指标在相同配置下测得。", ""]
    if rows:
        stat_keys = {"val_ppl", "ruler_avg", "ruler_avg_micro",
                     "gate_mean", "misleading_rate",
                     "misleading_given_conc", "auroc_logit", "auroc_qterm",
                     "auprc_logit", "auprc_qterm", "hit_base_rate",
                     "gate_f1_macro", "gate_f1_macro_always_write",
                     "ppl_ci_rel_pct", "epochs_seen", "ckpt_step",
                     "selection_step_fraction", "ruler_fwe_recall",
                     "ruler_fwe_precision", "ruler_fwe_f1"} \
            | {f"ruler_{t}" for t in ruler_tasks}
        header = "| " + " | ".join(c for _, c in cols) + " |"
        sep = "|" + "---|" * len(cols)
        md += [header, sep]
        for r in rows:
            cells = []
            for k, _ in cols:
                if k == "ruler_avg_ci95":
                    ci = r.get(k)
                    cells.append(
                        f"[{ci[0]:.3f}, {ci[1]:.3f}]"
                        if isinstance(ci, (list, tuple)) and len(ci) == 2
                        else "-")
                elif k in stat_keys:
                    cells.append(fmt_stat_cell(r, k))
                else:
                    cells.append(_fmt_scalar(r.get(k)))
            md.append("| " + " | ".join(cells) + " |")
        if invalid_eval_split or len(eval_splits) != 1:
            shown_splits = sorted((str(x) for x in eval_splits))
            split_note = (
                "⚠ PPL 评测 split 缺失、非法或不一致"
                + (f"（{', '.join(shown_splits)}）" if shown_splits else "")
                + "：跨变体数值可能来自不同切片，不能直接比较；请统一用 "
                "held-out test（或在同一 val 上）重跑评测")
        elif "test" in eval_splits:
            split_note = ("PPL 在 held-out test 集上评测（checkpoint 选择在 "
                          "val 上完成）")
        elif "val" in eval_splits:
            split_note = ("⚠ PPL 在 val 集上评测，该集同时用于 best-checkpoint "
                          "选择，数值偏乐观；请用 prepare_data.py 生成 "
                          "held-out test 后重跑评测")
        else:
            only_split = next(iter(eval_splits))
            split_note = (f"⚠ PPL 在 {only_split} 集上评测（非 val/test 口径）："
                          "该切片的绝对 PPL 不能与 val/test 数值直接比较，"
                          "也不能用于 held-out 泛化结论")
        min_n = min((r.get("n_seeds") or 1) for r in rows)
        seed_note = (f"关键指标至少 {min_n} 个随机种子；变体间比较为同种子配对。"
                     if min_n >= 3 else
                     f"⚠ 当前最少只有 {min_n} 个随机种子（<3），"
                     "无法可靠评估方差，结论只能是探索性的；"
                     "正式实验请用 ≥3 个种子（configs 默认已达要求）。")
        cols_notes = [
            "注: 统计单元格为 Mean ± Std (n=种子数)；n=1 时 Std 不可估计。",
            "注: `PPL评估CI` 为单次 run 内验证窗口抽样的 95% 相对半宽"
            "（窗口作为抽样单元，未计窗口间空间相关）；它与跨种子 Mean ± Std "
            "回答不同问题，比较变体时两者都要看；逐 run 区间见 eval.json 的 "
            "`val_ppl_ci95`。",
            "注: `训练覆盖` 为 tokens_seen / train_tokens 的等价 epoch 数；"
            "低于 1 时曲线下降主要反映“还没学完”，不能当作收敛。",
            "注: `选型步`/`选型步/计划` 是 `model_best.pt` 在 val 上被选中时"
            "的训练步数与占计划预算比例（跨种子均值）。所有变体共用同一最大"
            "预算与同一选型规则；选型步差异属 val 波动而非条件失控，但某变体"
            "长期在极早步被选中时，它的“同预算”比较实质是“更少训练”，需结合"
            "训练覆盖与曲线谨慎解读。",
            "注: niah/fwe/vt 为各上下文档位平均后的跨种子均值，"
            "每格样本数见 ruler_*.json（各变体同 seed 使用同一批样本）；",
            "注: `RULER评估CI` 为绝对 macro EM 的分层 bootstrap 95% 区间："
            "固定已观测种子，在每个 (任务, 上下文) 单元格内对样例有放回"
            "重采样，任务等权平均后跨种子平均；它只覆盖评测样例抽样，"
            "不覆盖训练种子随机性（后者看 Mean ± Std 与配对检验）。"
            "旧产物缺逐样例记录时显示 '-'，不伪造区间。",
            "“集中但看错率”= P(max_p>τ 且注意力落点不在答案数值跨度 value_span 内)，"
            "见机制分析。",
            "注: PPL 单元格标注 ⚠含截断值时，该变体至少有 1 个种子验证损失 >20，"
            "PPL 已按 exp(20) 截断；发散的 run 不能与正常 PPL 直接比较。",
            "注（类别不平衡与指标口径）：“命中基线率”= 机制分析中 hit 正类率，"
            "即 PR-AUC 的随机猜基线；正类率 <5% 时 AUROC 会因负样本基数巨大、"
            "FPR 增长缓慢而过于乐观，结论应以 PR-AUC 为主（极度不平衡 <0.1% "
            "时更是如此），两者并报供对照。F1 一律为 **macro**（各类等权，对"
            "稀有类最诚实）；micro/weighted 与逐类明细见 analysis_*.json 的 "
            "`gate_f1`（门控 g>0.5 决策点）。门控 F1 必须对照"
            "“恒写基线F1(macro)”（恒判写入的固定规则）：不高于这个无信息"
            "基线时，门控的决策价值不成立（`gate_f1_trivial` 另含恒不写"
            "规则）。RULER 的 EM 是生成式准确率、无决策阈值，不受此口径"
            "影响；fwe 另报召回率。",
            seed_note, split_note, ""]
        if common_ctx is not None:
            cols_notes.insert(1, f"“相对基线%”以 {base_name or 'standard'} 的 prefill 为基线"
                                 f"（公共 ctx={common_ctx}、batch=1，见 latency.json）；")
        md += [""] + cols_notes
        # 训练覆盖门：不足 1 epoch 的 PPL 下降只是“还没学完”，与“方法有效”
        # 是两回事；在表格外单独给醒目警告，避免读者把可跑通当结论。
        insufficient = [r for r in rows
                        if r.get("epochs_seen_min") is not None
                        and r["epochs_seen_min"] < 1.0]
        missing_cov = [r for r in rows if r.get("epochs_seen_min") is None]
        if insufficient:
            worst_cov = min(r["epochs_seen_min"] for r in insufficient)
            md += ["", f"⚠ 训练覆盖不足：{len(insufficient)}/{len(rows)} 个变体"
                      f"的最少训练覆盖仅约 {worst_cov:.2f} 个 epoch（<1）——"
                      "训练/验证曲线同时下降只说明“还在学”，不能作为收敛证据；"
                      "下表的 PPL/RULER 差异先按“没学完”处理，延长训练后重评。"]
        if missing_cov:
            md += ["", f"注：{len(missing_cov)}/{len(rows)} 个变体的 eval.json "
                      "没有训练覆盖字段（旧结果或手工评测），无法核对训练覆盖。"]
        # 评估噪声门：单 run 的 PPL 窗口抽样 CI 相对半宽若与跨种子差异同量级，
        # 变体间的小 PPL 差异不足以支撑方法结论。
        ci_vals = [r["ppl_ci_rel_pct"] for r in rows
                   if r.get("ppl_ci_rel_pct") is not None]
        if ci_vals and max(ci_vals) > 5.0:
            md += ["", f"注：单 run 内 PPL 评估 95% CI 相对半宽最大 "
                      f"±{max(ci_vals):.1f}%（窗口抽样）；小于该量级的 PPL "
                      "差异不能脱离评估噪声解读。"]
        # RULER 每格样本数门：n=2 与 n=30 的相同 EM 不是同样可信的证据；
        # 汇总层显式提示最小 n，读者需对照 ruler_*.json 的 Wilson CI。
        ruler_ns = [r["ruler_min_n_min"] for r in rows
                    if r.get("ruler_min_n_min") is not None]
        if ruler_ns and min(ruler_ns) < 10:
            md += ["", f"⚠ RULER 每格样本数不足：最小 n={min(ruler_ns)}（<10）——"
                      "EM 的 Wilson 95% CI 在该样本量下很宽（n=2 时接近 [0,0.84]），"
                      "RULER 的变体差异只能作方向性线索；逐格区间见 "
                      "ruler_*.json 的 `em_ci95_wilson`，正式结论需增大 "
                      "`ruler.n_samples` 后重跑。"]
        # RULER 平均口径：macro 与 micro 可能因任务/格大小不均而不同。
        # “平均”不注明定义就等于给读者留了事后择优的空间，这里显式固定
        # 预注册口径（macro）并给出样本加权口径供对照。
        rul_rows = [r for r in rows if r.get("ruler_avg") is not None]
        if rul_rows:
            md += ["", "### RULER 平均口径（macro / micro）", "",
                   "macro = 各任务内先对上下文档位等权平均，再对任务等权平均"
                   "（预注册“RULER平均”固定指此口径）；micro = 按每格样本数 n "
                   "加权（所有评测样例等权）。两者统一用各任务 EM；fwe 用"
                   "严格集合 EM（与预注册 H4 的“RULER 平均 EM”一致），其"
                   "recall 仅作诊断单列、不进入平均——recall 不惩罚“把上下文"
                   "词全部列出”的灌水策略。两者差异大说明任务难度/每格样本数"
                   "不均衡会改变“平均分”的读法，结论应同时对照逐任务列与"
                   "Wilson CI。", "",
                   "| 变体 | RULER平均 macro↑ | RULER平均 micro↑ | "
                   "macro−micro |", "|---|---|---|---|"]
            for r in rul_rows:
                macro, micro = r.get("ruler_avg"), r.get("ruler_avg_micro")
                gap = (macro - micro
                       if macro is not None and micro is not None else None)
                md.append(f"| {r.get('variant_cn') or r.get('variant')} | "
                          f"{_fmt4(macro)} | {_fmt4(micro)} | {_fmt4(gap)} |")
        # NIAH 位置分层：只报总 EM 会让“early/late 高、middle 塌陷”平均成
        # 看似正常的中等分；分层单独成表，并明确样本不可跨箱配对。
        depth_rows = [r for r in rows
                      if any(r.get(k) is not None for k in (
                          "ruler_depth_early", "ruler_depth_middle",
                          "ruler_depth_late"))]
        if depth_rows:
            md += ["", "### RULER NIAH 位置分层（early / middle / late）", "",
                   "按 needle 在上下文中的相对位置三分位分箱（[0,1/3)、"
                   "[1/3,2/3)、[2/3,1]），对 niah_* 任务的 (任务, 上下文) "
                   "单元格等权平均 EM。总 EM 掩盖位置差异：早期与末期的高分"
                   "会抵消中部塌陷（lost-in-the-middle），因此位置分层是"
                   "机制诊断的必报项。三个箱使用**不同评测样例**，不能直接"
                   "当作同批配对样本做显著性检验；每箱样本数与 Wilson 95% "
                   "CI 见 ruler_*.json 的 `em_by_depth`。", "",
                   "| 变体 | early↑ | middle↑ | late↑ | early−late |",
                   "|---|---|---|---|---|"]
            for r in depth_rows:
                e, m, l = (r.get("ruler_depth_early"),
                           r.get("ruler_depth_middle"),
                           r.get("ruler_depth_late"))
                gap = r.get("ruler_depth_early_minus_late")
                md.append(f"| {r.get('variant_cn') or r.get('variant')} | "
                          f"{_fmt4(e)} | {_fmt4(m)} | {_fmt4(l)} | "
                          f"{_fmt4(gap)} |")
            flags = []
            for r in depth_rows:
                e, m, l = (r.get("ruler_depth_early"),
                           r.get("ruler_depth_middle"),
                           r.get("ruler_depth_late"))
                name = r.get("variant_cn") or r.get("variant")
                if None not in (e, m, l):
                    mid_drop = min(e, l) - m
                    if mid_drop >= 0.10:
                        flags.append(
                            f"{name} 的 middle 比两端较低者低 "
                            f"{mid_drop * 100:.1f} pp")
                gap = r.get("ruler_depth_early_minus_late")
                if gap is not None and gap >= 0.10:
                    flags.append(f"{name} 的 early−late 落差 "
                                 f"{gap * 100:.1f} pp")
            if flags:
                md += ["", "⚠ 位置不均衡（描述性提示，非显著性判定）："
                           + "；".join(flags)
                           + "。这会把总 EM 抬到看似可接受的水平；需结合逐箱 "
                           "Wilson CI 与样本文本复核，不能只用 RULER平均 下结论。"]
        else:
            niah_rows = [r for r in rows
                         if any(r.get(f"ruler_{t}") is not None for t in (
                             "niah_synth", "niah_corpus", "niah_multikey"))]
            if niah_rows:
                md += ["", "⚠ 本次 RULER 产物不含 needle 位置分层"
                           "（`em_by_depth`，旧格式）：总 EM 可能掩盖 "
                           "lost-in-the-middle。旧结果不能补算，需用当前版本"
                           "重跑 RULER 后才能给出位置分层结论。"]
        # 上下文长度剂量-反应：位置分层回答“上下文内部的 needle 在哪”，
        # 长度分层回答“上下文本身变长时方法是否更有效”——两者是不同轴，
        # 不能互相替代；方法名为 length-calibrated，这一轴是核心主张。
        md += _render_ruler_length(length_analysis)
    else:
        md.append("（暂无结果，先运行训练评测）")

    # --- 门控概率校准：排序好 ≠ 概率数值可信。g 被当作“该不该写入”的
    #     软权重/概率使用，Brier/ECE 必须与 AUROC/PR-AUC 分开报告。
    cal_rows = [r for r in rows
                if r.get("brier_g") is not None
                or r.get("probe_brier") is not None]
    if cal_rows:
        md += ["", "### 门控概率校准（Brier / ECE）", "",
               "排序指标回答“能不能区分 hit/¬hit”，校准回答“g 的数值是否"
               "可信”。Brier 越小越好；常数基率预测（永远输出 hit_base_rate）"
               "的 Brier 是无信息基线，`Brier skill=1−Brier/Brier_常数` >0 "
               "才说明概率预测优于常数。ECE 为 10 个等宽分箱的加权平均"
               "|置信−准确率|；等频 ECE 按分数排序均分样本，避免分数聚集/"
               "小样本下大量空箱或每个样本独占一箱把 ECE 机械压低。两者"
               "都只作描述性证据；线性探针为有标签监督的 OOF 概率，门控 g "
               "为语言建模副产品，两者校准口径不同，不能只凭 Brier 判定"
               "方法优劣。", "",
               "| 变体 | 命中基线率 | 门控Brier↓ | 常数基率Brier | "
               "Brier skill↑ | 门控ECE↓ | 门控ECE(等频)↓ | "
               "探针Brier↓ | 探针ECE↓ |",
               "|---|---|---|---|---|---|---|---|---|"]
        for r in cal_rows:
            md.append(
                f"| {r.get('variant_cn') or r.get('variant')} | "
                f"{_fmt4(r.get('hit_base_rate'))} | "
                f"{_fmt4(r.get('brier_g'))} | "
                f"{_fmt4(r.get('brier_g_constant'))} | "
                f"{_fmt4(r.get('brier_g_skill'))} | "
                f"{_fmt4(r.get('ece_g'))} | "
                f"{_fmt4(r.get('ece_g_adaptive'))} | "
                f"{_fmt4(r.get('probe_brier'))} | "
                f"{_fmt4(r.get('probe_ece'))} |")
        md += ["",
               "注：门控 Brier/ECE 与固定 0.5 阈值无关；它们评估的是 g 的"
               "概率数值，而不是某个决策点的 F1。逐分箱的置信/准确率、"
               "n 与 gap 见 `analysis_*.json` 的 `calibration_g.bins`；"
               "Brier 的 Murphy 分解（reliability/resolution/uncertainty，"
               "原始分数非箱内常数时另报 brier_binning 箱内项）"
               "与等频分箱明细见 `calibration_g.brier_*` 与 `ece_adaptive`。"]

    # --- 基线合理性（简单统计基线 + 基线调优证据 + 逐变体 vs 统计基线
    #     + 机制判别的线性探针简单监督基线）
    ngram_cmp = ngram_baseline_comparison(runs, results_dir)
    probe_cmp = linear_probe_baseline_comparison(runs)
    md += _render_baseline_sections(results_dir,
                                    smoke=((repro or {}).get("smoke") is True),
                                    ngram_cmp=ngram_cmp,
                                    probe_cmp=probe_cmp,
                                    variants=sorted(runs))

    # --- 先验设计计划与分析人群（训练前声明，汇总时核对）
    md += render_design_section(
        design_plan, design_conf,
        observed=observed_design_sensitivity(tests))

    # --- 受控变量自检
    md += ["", "## 对比实验受控变量自检", ""]
    if audit:
        exempt_keys = _as_text_list(audit.get("exempt_keys"))
        stopped = _as_text_list(audit.get("early_stop_runs"))
        if audit.get("ok"):
            if audit.get("checked"):
                md.append(f"- ✅ 通过：{audit['checked']} 个 run 的优化器、学习率、"
                          f"训练步数、数据顺序等受控设置逐项一致；逐 run 仅 "
                          f"`{'`, `'.join(exempt_keys)}` 不同。")
            else:
                md.append("- ⚠ 未检查受控变量：没有 run 可用于核对"
                          "（全部缺少 config_used.json），不能据此声称一致性。")
            if audit.get("meta_checked"):
                digest = audit.get("expected_data_digest")
                md.append(f"- ✅ 训练溯源一致：{audit['meta_checked']} 个 run 的"
                          f"数据指纹、代码提交、配置 SHA-256、计划训练步数"
                          f"逐项一致"
                          + ("；实际步数差异为早停所致（见下条）"
                             if stopped else "")
                          + (f"（数据指纹 `{digest}`，并与本次流水线数据比对）"
                             if digest else "。"))
            if stopped:
                md.append(
                    f"- ⏹ 早停启用：{len(stopped)} 个 run 未跑满"
                    f"计划步数（{', '.join(stopped[:8])}"
                    + (" …" if len(stopped) > 8 else "")
                    + "）：实际步数/tokens_seen 因验证曲线不同而不同，"
                    "有效训练量不等；受控变量按计划步数核对，但预注册"
                    "确认性判定已自动降级为探索性（见 `significance.json` "
                    "的 registered_hypotheses）。逐 run 实际步数与 tokens_seen "
                    "见各 eval.json。")
        else:
            md.append(f"- ❌ 未通过：{audit.get('mismatches') or audit}")
        missing_configs = _as_text_list(audit.get("missing_configs"))
        if missing_configs:
            md.append(f"- ⚠ 缺少配置文件的 run：{', '.join(missing_configs[:8])}"
                      + (" …" if len(missing_configs) > 8 else ""))
        missing_evals = _as_text_list(audit.get("missing_evals"))
        if missing_evals:
            md.append(f"- ⚠ 缺少 eval.json、无法核对数据/代码/训练步数的 run："
                      f"{', '.join(missing_evals[:8])}"
                      + (" …" if len(missing_evals) > 8 else ""))
    else:
        md.append("- ⚠ 未执行（直接调用 build_summary 或缺少 out 目录）。")

    # --- 数据协议与防泄漏自检（由 lega.data.prepare_bins 生成 data_report.json）
    md += ["", "## 数据协议与防泄漏自检", ""]
    if data_report:
        proto = _as_dict(data_report.get("protocol"))
        md.append(
            f"- 协议：先划分后预处理={proto.get('split_first_then_preprocess')}、"
            f"拟合统计只来自 {proto.get('fit_on')}、"
            f"划分时打乱={proto.get('shuffle_at_split_time')}；"
            "训练阶段 batch 级随机采样（shuffle）不构成泄露。")
        sp = _as_dict(data_report.get("split_protocol"))
        if sp:
            md.append(f"- 划分方式：{sp.get('mode', '-')}；"
                      f"顺序 {sp.get('order', '-')}；"
                      f"文档边界回退={sp.get('fallback_char_cut', '-')}"
                      + ("（⚠ 字符切分可能让同一文档跨 split）"
                         if sp.get("fallback_char_cut") else ""))
            # 声明与文本核对结果：document_disjoint 是上游声明，verified_disjoint
            # 是对实际写出的三份文本做跨 split 长行核对后的结论。
            if "verified_disjoint" in sp:
                vd = sp.get("verified_disjoint")
                md.append(f"- 划分声明核对：无共享长行={vd}"
                          + ("" if vd else
                             "（⚠ 实际文本存在跨 split 共享长行，声明被证伪，"
                             "见 overlap_long_lines）"))
            group = sp.get("group_unit")
            cross = sp.get("documents_cross_cut")
            pcross = sp.get("paragraphs_cross_cut")
            if group or cross is not None or pcross is not None:
                warn = ""
                if cross:
                    warn += "（⚠ 文档群组跨 split，见告警 split_documents_cross_cut）"
                if pcross:
                    warn += "（⚠ 段落群组跨 split，见告警 split_paragraphs_cross_cut）"
                md.append(f"- 群组完整性：群组单位={group or '-'}；"
                          f"标题数={sp.get('n_headings', '-')}；"
                          f"段落边界数={sp.get('n_paragraph_boundaries', '-')}；"
                          f"文档跨切点={cross if cross is not None else '-'}；"
                          f"段落跨切点={pcross if pcross is not None else '-'}"
                          + warn)
            actual = _as_dict(sp.get("actual_fractions"))
            target = _as_dict(sp.get("target_fractions"))
            dev = _finite_number(sp.get("max_abs_fraction_deviation"))
            if actual:
                def _fr(d, k):
                    f = _finite_number(d.get(k))
                    return "-" if f is None else f"{f:.1%}"
                md.append(
                    "- 划分比例：实际 "
                    + " / ".join(f"{k}={_fr(actual, k)}"
                                 for k in ("train", "val", "test"))
                    + ("；目标 " + " / ".join(
                        f"{k}={_fr(target, k)}"
                        for k in ("train", "val", "test")) if target else "")
                    + (f"；最大绝对偏差 {dev:.1%}" if dev is not None else "")
                    + (f"；切点相对目标偏移 {sp.get('cut_deviation_chars')} 字符"
                       if sp.get("cut_deviation_chars") is not None else "")
                    + "。切点吸附到文档/段落边界会改变实际训练/评测规模，"
                      "解读固定比例与检验功效时须按实际值")
        tok = _as_dict(data_report.get("tokenizer"))
        unk = _as_dict(tok.get("unk_rate"))
        unk_s = "、".join(f"{k}={_fmt_rate(v)}" for k, v in unk.items())
        md.append(f"- 分词器：{tok.get('kind')}（vocab={tok.get('vocab_size')}，"
                  f"拟合集={tok.get('fit_on')}）；OOV 字符率：{unk_s or 'N/A'}"
                  "（未知字符映射 <unk>，val/test 不参与词表拟合）")
        splits = _as_dict(data_report.get("splits"))
        md.append("- token 流硬校验：" + "；".join(
            f"{k} {_int_or_zero(_as_dict(v).get('tokens')):,} tokens"
            f"（id∈[{_as_dict(v).get('min_id')},{_as_dict(v).get('max_id')}]，"
            f"NaN={_as_dict(v).get('nan')}，Inf={_as_dict(v).get('inf')}）"
            for k, v in splits.items()))
        dup = _as_dict(data_report.get("duplicates"))
        cont = _as_dict(data_report.get("contamination"))
        doc = _as_dict(dup.get("document_level"))
        if doc:
            md.append(
                f"- 块级去污染：与 val/test 重复的多行块移除 "
                f"{doc.get('train_blocks_dropped_as_duplicates_of_eval', '-')} 个；"
                f"train 内部重复块移除 "
                f"{doc.get('train_intra_split_duplicate_blocks_removed', '-')} 个"
                f"（共 {doc.get('train_lines_dropped', '-')} 行）；"
                f"val↔test 多行块重叠 {doc.get('val_test_overlap_blocks', '-')} 个"
                "（只改 train，val/test 保持尘封）")
        near = _as_dict(dup.get("near_duplicate"))
        if near:
            md.append(
                f"- 近似重复去污染（MinHash/LSH，字符 "
                f"{near.get('shingle_size', '-')}-gram，协议阈值 "
                f"{near.get('threshold', '-')}）：与 val/test 近似块移除 "
                f"{_int_or_zero(near.get('train_blocks_dropped_as_near_duplicates_of_eval')):,} 个、"
                f"train 内部近似重复块移除 "
                f"{_int_or_zero(near.get('train_intra_split_near_duplicates_removed')):,} 个"
                f"（共 {_int_or_zero(near.get('train_lines_dropped')):,} 行；"
                "只改 train，val/test 保持尘封）；"
                f"val↔test 近似对 "
                f"{_int_or_zero(near.get('val_test_near_duplicate_pairs')):,}；"
                f"LSH 理论召回率 {_fmt_rate(near.get('lsh_recall_at_threshold'))}；"
                f"候选对 {_int_or_zero(near.get('candidate_pairs')):,}"
                "（阈值敏感性见 data_report.json）")
            cov = _finite_number(near.get("train_chars_coverage_fraction"))
            if cov is not None:
                md.append(
                    f"- 近似重复扫描覆盖：{cov:.1%} 的 train 字符达到 "
                    f"min_chars={near.get('min_chars', '-')} 且 shingle 数达标；"
                    f"未扫描字符：短块 "
                    f"{_int_or_zero(near.get('train_chars_skipped_short')):,}、"
                    f"少 shingle "
                    f"{_int_or_zero(near.get('train_chars_skipped_few_shingles')):,}、"
                    f"超大块 "
                    f"{_int_or_zero(near.get('train_chars_skipped_too_large')):,}"
                    "（近重复结论只对已覆盖部分成立）")
            removal = _as_dict(near.get("threshold_sensitivity_removal"))
            if removal:
                def _tau_key(x):
                    try:
                        return (0, float(x))
                    except (TypeError, ValueError):
                        return (1, str(x))
                parts = []
                for key in sorted(removal, key=_tau_key):
                    r = _as_dict(removal.get(key))
                    parts.append(
                        f"τ={key}: 删 "
                        f"{_int_or_zero(r.get('train_blocks_removed_total'))} 块/"
                        f"{_int_or_zero(r.get('train_chars_removed')):,} 字符/"
                        f"{_fmt_rate(r.get('removal_fraction_chars'))}")
                md.append("- 近似重复阈值敏感性（实际删除量，不只是判定对数）："
                          + "；".join(parts)
                          + "。阈值是分析前声明的协议参数，不能按下游结果回头挑选")
            if (near.get("candidate_pairs_truncated")
                    or _int_or_zero(near.get("oversized_groups_truncated"))):
                md.append("- ⚠ 近似重复扫描触发候选簇/候选总数上限：删除量是下界，"
                          "须降低语料规模或提高上限后重跑")
        post = _as_dict(dup.get("post_cleanup"))
        if post.get("checked_pairs"):
            clean = post.get("train_eval_clean")
            if clean is True:
                md.append("- 清洗后复核：train 与 val/test 已无共享精确/归一化"
                          "长行或多行块（短行与近似重复另见对应条目）")
            elif clean is False:
                md.append("- ⚠ 清洗后复核：train 仍与 val/test 共享精确/归一化"
                          "长行或多行块，去污染未闭合，须重建数据")
        acct = _as_dict(data_report.get("accounting"))
        if acct:
            md.append(
                "- 语料核算（与分词器无关的分母）：" + "；".join(
                    f"{k} {_int_or_zero(_as_dict(v).get('tokens')):,} tokens / "
                    f"{_int_or_zero(_as_dict(v).get('chars')):,} 字符 / "
                    f"{_int_or_zero(_as_dict(v).get('utf8_bytes')):,} UTF-8 字节"
                    f"（{_fmt4(_as_dict(v).get('bytes_per_token'))} bytes/token）"
                    for k, v in acct.items())
                + "。字符级兜底与 BPE 的 token PPL 不可直接比较，"
                  "规模与分母以字节/字符核算为准")
        md.append(
            f"- 去污染：与 val/test 重复的训练长行移除 "
            f"{dup.get('train_lines_dropped_as_duplicates_of_eval', '-')} 行；"
            f"train 内部重复长行移除 "
            f"{dup.get('train_intra_split_duplicate_lines_removed', '-')} 行"
            f"（保留首次出现）；val↔test 长行重叠 "
            f"{dup.get('val_test_overlap_long_lines', '-')} 条；"
            f"采样 {_as_dict(cont.get('train_val')).get('n', '-')}-gram 重叠率 "
            f"val={_fmt_rate(_as_dict(cont.get('train_val')).get('overlap_rate'))}、"
            f"test={_fmt_rate(_as_dict(cont.get('train_test')).get('overlap_rate'))}、"
            f"val↔test={_fmt_rate(_as_dict(cont.get('val_test')).get('overlap_rate'))}"
            "（采样筛查下界，>0 需人工核查）")
        n_short_tv = _int_or_zero(dup.get("train_val_overlap_short_lines"))
        n_short_tt = _int_or_zero(dup.get("train_test_overlap_short_lines"))
        n_short_vt = _int_or_zero(dup.get("val_test_overlap_short_lines"))
        if n_short_tv or n_short_tt or n_short_vt:
            md.append(
                f"- 短行跨 split 重复（<{dup.get('min_chars', '-')} 字符，"
                "不自动删除以免误删标题/固定用语）："
                f"train↔val {n_short_tv} 条、train↔test {n_short_tt} 条、"
                f"val↔test {n_short_vt} 条"
                f"（空白归一化后额外 "
                f"{_int_or_zero(dup.get('train_val_overlap_normalized_short_lines'))}/"
                f"{_int_or_zero(dup.get('train_test_overlap_normalized_short_lines'))}/"
                f"{_int_or_zero(dup.get('val_test_overlap_normalized_short_lines'))}）；"
                "重复样本会削弱 held-out 独立性，须人工核查")
        cshort = _as_dict(data_report.get("contamination_short_ngram"))
        if cshort:
            md.append(
                f"- 短 n-gram 重叠（{_as_dict(cshort.get('train_val')).get('n', '-')}-gram，"
                "抓 <64 token 整句/模板复用，仅告警）："
                f"val={_fmt_rate(_as_dict(cshort.get('train_val')).get('overlap_rate'))}、"
                f"test={_fmt_rate(_as_dict(cshort.get('train_test')).get('overlap_rate'))}、"
                f"val↔test={_fmt_rate(_as_dict(cshort.get('val_test')).get('overlap_rate'))}"
                "（短 n 易有自然搭配假阳性，不进硬闸门）")
        dist = _as_dict(data_report.get("distribution_shift"))
        dv = _as_dict(dist.get("val_vs_train"))
        dt = _as_dict(dist.get("test_vs_train"))
        dvt = _as_dict(dist.get("test_vs_val"))
        md.append(
            f"- 分布漂移（一元 token，默认精确计数）：val−train TV={_fmt4(dv.get('tv'))}、"
            f"JS={_fmt4(dv.get('js_bits'))} bit；test−train TV={_fmt4(dt.get('tv'))}、"
            f"JS={_fmt4(dt.get('js_bits'))} bit；test−val TV={_fmt4(dvt.get('tv'))}、"
            f"JS={_fmt4(dvt.get('js_bits'))} bit；test OOV token 率="
            f"{_fmt_rate(dt.get('eval_oov_token_rate'))}")
        san = _as_dict(data_report.get("sanitization"))
        noise = sum(_int_or_zero(_as_dict(san.get(s)).get(
            "replacement_char_replaced_with_space")) for s in san)
        ctrl = sum(_int_or_zero(_as_dict(san.get(s)).get(
            "control_chars_deleted")) for s in san)
        md.append(f"- 文本清洗（标签噪声）：U+FFFD 替换 {noise:,} 处、"
                  f"非法控制符删除 {ctrl:,} 处；清洗为逐字符确定性规则，"
                  "不依赖跨样本统计量。")
        ts = _as_dict(data_report.get("text_stats"))
        if ts:
            md.append("- 行长度异常检查：" + "；".join(
                f"{k} 行数={_int_or_zero(_as_dict(v).get('lines')):,}、"
                f"最长行={_int_or_zero(_as_dict(v).get('line_chars_max')):,} 字符、"
                f"超长行(>10万字符)={_int_or_zero(_as_dict(v).get('lines_gt_huge_threshold'))}"
                for k, v in ts.items()))
        alerts = _as_dict_list(data_report.get("alerts"))
        if alerts:
            n_warn = sum(1 for a in alerts if a.get("level") == "warning")
            shown = "；".join(f"[{a.get('code')}] {a.get('message')}"
                              for a in alerts[:5])
            md.append(f"- ⚠ 自检告警（{n_warn} warning / 共 {len(alerts)} 条）："
                      + shown + (" …" if len(alerts) > 5 else "")
                      + "（阈值见 data_report.json 的 alert_thresholds）")
        else:
            md.append("- 自检告警：无（全部指标在 alert_thresholds 阈值内；"
                      "阈值是复核触发线而非显著性检验）")
        ta = _as_dict(data_report.get("test_access"))
        if ta:
            repeated = _as_text_list(
                ta.get("checkpoints_evaluated_multiple_times"))
            warn = ("" if not repeated
                    else "；⚠ 存在被重复评测的 checkpoint："
                         + "、".join(repeated[:4]))
            md.append(f"- 测试集尘封账本：累计 test 评测 {ta.get('evaluations', 0)} 次、"
                      f"不同 checkpoint {ta.get('distinct_checkpoints', 0)} 个"
                      f"{warn}。测试集只用于最终评估；据此调参即视为泄露。")
        else:
            md.append("- 测试集尘封账本：暂无 test 评测记录。")
    else:
        md.append("- ⚠ 未找到 `data_report.json`（旧数据或未走统一数据入口）；"
                  "请用 `scripts/prepare_data.py` / `scripts/make_dummy_data.py` "
                  "重建数据以生成防泄漏自检。")

    # --- 可复现性与 sanity check（代码版本 / 数据指纹 / 过拟合小批量）
    md += ["", "## 可复现性与 sanity check", ""]
    if repro:
        git = _as_dict(repro.get("git"))
        data = _as_dict(repro.get("data"))
        rt = _as_dict(repro.get("runtime"))
        commit = git.get("commit") or "unknown"
        dirty = git.get("dirty")
        md.append(f"- 代码版本：`{str(commit)[:12]}`"
                  + (f"（分支 {git.get('branch')}）" if git.get("branch") else "")
                  + ("；⚠ 工作区 dirty，结果对应未提交状态" if dirty else
                     ("" if dirty is None else "；工作区 clean")))
        files = data.get("files")
        files = [e for e in files if isinstance(e, dict)] \
            if isinstance(files, list) else []
        md.append(f"- 数据指纹：`{data.get('digest')}`"
                  f"（tokenizer={data.get('tokenizer')}，"
                  + "、".join(f"{e.get('name')} "
                              f"sha256:{str(e.get('sha256'))[:12]}…"
                              for e in files)
                  + "）")
        md.append(f"- 配置 SHA-256：`{str(repro.get('config_sha256'))[:16]}…`；"
                  f"种子：{repro.get('seeds')}")
        md.append(f"- 运行环境：python={rt.get('python')} torch={rt.get('torch')} "
                  f"device={rt.get('device')} gpu={rt.get('gpu')} cuda={rt.get('cuda')}")
        # 逐 run 核对评测环境与训练清单：pipeline 内的 eval 通常同环境，但
        # results 目录可能被另一环境单独补跑（--skip_train/手工 evaluate），
        # 仅看全局 repro.json 的 pipeline runtime 无法发现这种错配。
        runtime_drift_rows = []
        for _v, _seed_map in runs.items():
            for _seed, _slots in _seed_map.items():
                _ev = (_slots or {}).get("eval")
                if not isinstance(_ev, dict) or \
                        _ev.get("runtime_matches_training") is not False:
                    continue
                _fields = ",".join(
                    str(m.get("field"))
                    for m in (_ev.get("runtime_mismatches") or [])
                    if isinstance(m, dict) and m.get("field")) or "unknown"
                runtime_drift_rows.append(f"{_v}/seed{_seed}（{_fields}）")
        if runtime_drift_rows:
            md.append("- ⚠ 以下 run 的**评测环境**与训练 `run_manifest.json` "
                      "记录不一致："
                      + "、".join(runtime_drift_rows)
                      + "；同一 checkpoint 在不同 Python/PyTorch/NumPy/CUDA/"
                      "cuDNN/GPU 上可能因浮点内核差异得到不同数值，相关指标不得"
                      "声称与训练环境完全相同（逐项见 `eval_*.json` 的 "
                      "`train_runtime`/`eval_runtime`）。")
        md.append("- 完整记录：`repro.json`（全局）、各 run 目录 `run_manifest.json`"
                  "（代码/数据/环境/命令行）、`config_used.json`（超参数）、"
                  "`train_log.jsonl`（训练曲线）。")
    else:
        md.append("- ⚠ 未找到 `repro.json`（直接调用 build_summary？）；"
                  "无法核对代码版本与数据指纹。")
    sanity_files = sorted(glob.glob(os.path.join(results_dir, "sanity_*.json")))

    def _load_dict(p):
        d = load(p)
        return d if isinstance(d, dict) else None

    # 与 runs 同一“本次 run”口径：sanity 虽无“每变体多档种子”，但
    # run_all 只用首个种子跑 sanity；同一 results 目录被换种子重跑（尤其
    # --skip_sanity）时，旧种子的 PASS 文件不能在本轮汇总里冒充本次证据。
    expected_sanity_seed = (_integral_value(scope_seeds[0])
                            if scope_seeds else None)
    if scope_variants is not None:
        sanity_files = [fp for fp in sanity_files
                        if (_load_dict(fp) or {}).get("variant") in scope_variants]
    if expected_sanity_seed is not None:
        kept = []
        for fp in sanity_files:
            d = _load_dict(fp) or {}
            d_seed = _integral_value(d.get("seed"))
            # 旧 sanity 文件没有 seed 字段时保留（无法核对，不能伪装成
            # “已属于本次 run”）；明确记录为其他 seed 的文件必须剔除。
            if d_seed is not None and d_seed != expected_sanity_seed:
                continue
            kept.append(fp)
        sanity_files = kept
    if sanity_files:
        md.append("")
        md.append("| 变体 | sanity | 初始 loss | 最终 loss | token 准确率 |")
        md.append("|---|---|---|---|---|")
        for fp in sanity_files:
            d = _load_dict(fp)
            if not d:
                continue
            ini = _num_or_none(_as_dict(d.get("initial")).get("loss"))
            fin = _num_or_none(_as_dict(d.get("final")).get("loss"))
            acc = _num_or_none(_as_dict(d.get("final")).get("token_acc"))
            sv = d.get("variant")
            sv = VARIANT_CN.get(sv, sv) if isinstance(sv, str) else "?"
            md.append(f"| {sv} | "
                      f"{'✅ 通过' if d.get('passed') else '❌ 未通过'} | "
                      f"{'-' if ini is None else f'{ini:.4f}'} | "
                      f"{'-' if fin is None else f'{fin:.4f}'} | "
                      f"{'-' if acc is None else f'{acc:.4f}'} |")
        md.append("")
        md.append("注: sanity check 只验证固定小批量的前向/反向/损失/标签链路能被"
                  "优化到接近零损失（必要非充分）；数据增强/特征对齐/标签映射等"
                  "问题仍可能漏网，需配合中间结果检查。")
    else:
        md.append("- ⚠ 未找到 sanity 结果（本次可能跳过了 sanity check）。")
    if os.path.exists(os.path.join(results_dir, "fig_train_curves.png")):
        md.append("- 训练曲线图：`fig_train_curves.png`（各变体训练 loss 与 val PPL，"
                  "多种子取均值）。")

    # --- 训练曲线诊断：结果分析不能只看最终数值，曲线本身先要过一遍体检
    md += ["", "## 训练曲线诊断（过拟合/欠拟合/发散/NaN）", ""]
    curve = load(os.path.join(results_dir, "curve_diagnosis.json"))
    # 合法 JSON 但非对象（数组/标量）的诊断文件按“无诊断”处理。
    if not isinstance(curve, dict):
        curve = None
    if curve:
        md.append("诊断口径：训练降而验证升=过拟合；训练停滞（不再实质下降）而"
                  "验证回升同样判过拟合（平台期泛化先恶化，不按欠拟合处理）；"
                  "训练仍在实质下降而验证无实质变化=过拟合趋势（验证不再随训练"
                  "改善的早期信号，按正则化/早停处理，不据此续训）；"
                  "两者都不降=欠拟合/学习率不当/数据管道问题；"
                  "验证损失显著低于训练损失（倒挂）=泄漏/划分错配/评估口径"
                  "不一致，不能当“验证更好”；best 与次优差距落在评估噪声内="
                  "checkpoint 选型不可靠（winner's curse）；曲线虽在降但"
                  "训练覆盖 <1 epoch=训练不足（还没学完，不能当收敛）；"
                  "出现 NaN=训练失效；"
                  "瞬态损失尖峰（分段均值会掩盖；另查区间单步峰值 loss_max）"
                  "与损坏日志行单独报告；全程首/末之外还比较末段（后 1/3 vs "
                  "中 1/3），晚期平台/回升不被早段大幅改善掩盖。"
                  "逐 run 明细见 `curve_diagnosis.json`。")
        md.append("")
        md.append("| 变体 | 状态（最差种子） | 训练覆盖(epoch) | 训练损失 首→末 "
                  "| 验证损失 首→最优→末 | 早停空间 | 关键建议 |")
        md.append("|---|---|---|---|---|---|---|")
        any_nan = False
        any_grad_spike = False
        any_loss_spike = False
        any_malformed = False
        any_inversion = False
        any_selection_noise = False
        observability = set()
        for v in variants:
            by_seed = curve.get(v)
            if not isinstance(by_seed, dict):
                continue
            diags = [d for d in by_seed.values()
                     if isinstance(d, dict) and isinstance(d.get("status"), str)]
            if not diags:
                continue
            status = worst_status([d["status"] for d in diags])
            worst = next(d for d in diags if d["status"] == status)
            any_nan = any_nan or any(d.get("nonfinite_values") for d in diags)
            spike = any(d.get("has_grad_spike") for d in diags)
            any_grad_spike = any_grad_spike or spike
            lspike = any(d.get("has_loss_spike") for d in diags)
            any_loss_spike = any_loss_spike or lspike
            any_malformed = any_malformed or any(d.get("malformed_records")
                                                 for d in diags)
            any_inversion = any_inversion or any(d.get("val_below_train")
                                                 for d in diags)
            any_selection_noise = any_selection_noise or any(
                d.get("best_checkpoint_within_noise") for d in diags)
            for d in diags:
                for w in (d.get("curve_observability_warnings") or []):
                    if isinstance(w, str) and w.strip():
                        observability.add(w.strip())
            es = [x for x in (_num_or_none(d.get("early_stop_gain_rel"))
                              for d in diags) if x is not None]

            def _f4(x):
                x = _num_or_none(x)
                return "-" if x is None else f"{x:.4f}"

            train_s = (f"{_f4(worst.get('train_loss_first'))}→"
                       f"{_f4(worst.get('train_loss_last'))}")
            val_s = (f"{_f4(worst.get('val_loss_first'))}→"
                     f"{_f4(worst.get('val_loss_best'))}"
                     f"@s{worst.get('val_best_step')}"
                     f"→{_f4(worst.get('val_loss_last'))}")
            es_s = ("-" if not es else
                    f"{sum(es) / len(es):+.1%}"
                    + ("（最终劣于最优，已用 best checkpoint）"
                       if sum(es) / len(es) > 0.02 else ""))
            eps = [x for x in (_num_or_none(d.get("epochs_seen"))
                               for d in diags) if x is not None]
            ep_min = min(eps) if eps else None
            ep_s = ("-" if ep_min is None else
                    f"{ep_min:.3f}" + (" ⚠<1" if ep_min < 1.0 else ""))
            rec = worst.get("recommendations")
            rec = rec[:1] if isinstance(rec, list) else []
            es_flagged = any(d.get("early_stopped") for d in diags)
            # 需关注种子的占比：只看"最差种子"分不清系统性异常与单个种子偶发，
            # 1/5 与 5/5 需要完全不同的处置
            n_attention = sum(1 for d in diags
                              if d.get("status") not in ("正常收敛", "数据不足"))
            attention_s = (f"（{n_attention}/{len(diags)} 种子需关注）"
                           if n_attention else "")
            status_s = status + attention_s \
                + ("　（梯度尖峰）" if spike else "") \
                + ("　（损失尖峰）" if lspike else "") \
                + ("　（早停）" if es_flagged else "")
            md.append(f"| {VARIANT_CN.get(v, v)} | {status_s} | {ep_s} "
                      f"| {train_s} | {val_s} | {es_s} "
                      f"| {rec[0] if rec else '-'} |")
        md.append("")
        if observability:
            shown = sorted(observability)
            more = len(shown) - 2
            md.append("⚠ 曲线采样可观测性受限（配置级，所有种子同因）："
                      + "；".join(shown[:2])
                      + (f"；另有 {more} 条同类告警" if more > 0 else "")
                      + "。上述趋势/早停结论的可靠性受此限制，正式实验请"
                      "按告警调整 `eval_every`/`log_every` 后重跑。")
            md.append("")
        if any_nan:
            md.append("⚠ 存在出现非有限损失的 run：训练已失效，"
                      "对应 run 目录的 `nan_report.json` 给出了"
                      "数据→代码→混合精度→梯度/学习率的系统排查路径"
                      "（含检测点前的整个 log 区间候选批次与区间内首个非有限"
                      "损失步 `first_bad_step`，排查的是失败现场而非最后一步"
                      "批次），处理前该变体的一切指标都不可信。")
        else:
            md.append("注：全部 run 的损失曲线无 NaN/Inf（梯度非有限另见下）；"
                      "有异常状态的变体请按建议处理"
                      "（过拟合/过拟合趋势→正则化/增数据/早停，欠拟合→查学习率"
                      "与数据管道，训练不足→延长训练/增大数据覆盖）后重跑，"
                      "再进入对比结论。")
        if any_inversion:
            md.append("")
            md.append("⚠ 存在验证损失显著低于训练损失的 run（与泛化差距方向"
                      "相反）：先按 datacheck 审计跨 split 重复/n-gram 重叠与"
                      "划分边界，再核对 train/val 的 dropout 状态、精度与损失"
                      "统计口径；排除泄漏/错配前，该 run 的验证指标不得用于"
                      "方法比较。")
        if any_selection_noise:
            md.append("")
            md.append("⚠ 存在 best checkpoint 与次优验证损失差距落在该点 95% CI "
                      "内的 run：最优与次优无法区分（winner's curse），"
                      "不能只报告 best 数字；应同时报告 final/多种子均值，"
                      "并增大 eval_iters 或验证集后再据此选型。")
        if any_grad_spike:
            md.append("")
            md.append("⚠ 检测到 pre-clip 梯度范数非有限或爆炸性尖峰"
                      "（区间峰值/中位数 ≥10×）的 run：这是 NaN/发散的早期信号，"
                      "即使损失曲线看似正常也应先降低学习率/收紧 grad_clip "
                      "后重跑；发生步与逐点清单见 `curve_diagnosis.json` 的 "
                      "`grad_norm_max*`/`grad_spikes` 字段。")
        if any_loss_spike:
            md.append("")
            md.append(f"⚠ 检测到训练损失瞬态尖峰（某点 > 邻近中位数 "
                      f"{LOSS_SPIKE_FACTOR:.0f}×，分段均值会掩盖；"
                      "区间单步峰值 loss_max 另查）：先核对该"
                      "步批次数据与梯度范数，再决定收紧 grad_clip/降低学习率；"
                      "细节见 `curve_diagnosis.json` 的 "
                      "`loss_spikes`/`loss_peak_spikes` 字段。")
        if any_malformed:
            md.append("")
            md.append("⚠ 部分 train_log.jsonl 记录无法解析（已跳过并计数）："
                      "日志可能被中断或损坏，曲线结论只基于其余记录点，"
                      "复核前不要把它当作完整训练历史。")
        any_early = any(
            d.get("early_stopped")
            for vs in curve.values() if isinstance(vs, dict)
            for d in vs.values() if isinstance(d, dict))
        if any_early:
            md.append("")
            md.append("⏹ 上表带「早停」标记的变体触发了早停（验证连续 patience "
                      "次无实质改善即终止，曲线按设计截断，区别于日志缺失/中断）："
                      "下游评测使用的仍是 val 选出的 `model_best.pt`；若其最优"
                      "仍出现在训练早期，应先处理过拟合/学习率问题，而不是"
                      "统一延长所有变体的训练步数。")
    else:
        md.append("- ⚠ 未找到 `curve_diagnosis.json`：只看最终数值会掩盖训练动态，"
                  "请先运行训练（或 `python -m lega.curves --run_dir <run 目录>`）。")

    # --- 门控健康：机制是否真的在起作用，不能只看最终 PPL
    md += ["", "## 门控健康（机制是否真的在起作用）", ""]
    gate_order = {"健康": 0, "不适用（无门控统计）": 0,
                  "门控近似常数（无样本级区分）": 2,
                  "门控剧烈漂移": 3,
                  "门控塌缩（几乎不写入）": 4,
                  "门控饱和（退化为标准注意力）": 4}
    gate_rows = []
    if isinstance(curve, dict):
        for v in variants:
            by_seed = curve.get(v)
            if not isinstance(by_seed, dict):
                continue
            hs = [d.get("gate_health") for d in by_seed.values()
                  if isinstance(d, dict)
                  and isinstance(d.get("gate_health"), dict)
                  and d["gate_health"].get("available")]
            if not hs:
                continue
            worst_g = max(hs, key=lambda h: gate_order.get(h.get("status"), -1))
            status_g = worst_g.get("status") or "-"
            m_last = _num_or_none(worst_g.get("mean_last"))
            s_last = _num_or_none(worst_g.get("std_last"))
            lo = _num_or_none(worst_g.get("frac_low_last"))
            hi = _num_or_none(worst_g.get("frac_high_last"))
            layers = [x for x in (_num_or_none(y) for y in
                                  (worst_g.get("layer_mean_last") or []))
                      if x is not None]
            layer_s = ("-" if not layers else
                       (f"{min(layers):.2f}" if len(layers) == 1
                        else f"{min(layers):.2f}~{max(layers):.2f}"))
            n_bad_gate = sum(1 for h in hs
                             if h.get("status") not in ("健康",))
            status_s = status_g + (f"（{n_bad_gate}/{len(hs)} 种子需关注）"
                                   if n_bad_gate else "")
            rec = (worst_g.get("recommendations") or [""])[0]
            gate_rows.append(
                f"| {VARIANT_CN.get(v, v)} | {status_s} | "
                f"{'-' if m_last is None else f'{m_last:.3f}'} | "
                f"{'-' if s_last is None else f'{s_last:.4f}'} | "
                f"{'-' if lo is None else f'{lo:.0%}'} | "
                f"{'-' if hi is None else f'{hi:.0%}'} | {layer_s} | {rec} |")
    if gate_rows:
        md.append("门控健康检查的是方法机制本身：门控塌缩/饱和/常数化时，"
                  "PPL 的涨跌不能归因于查询、集中度或长度校准项。逐 run 数值"
                  "见 `curve_diagnosis.json` 的 `gate_health`。")
        md.append("")
        md.append("| 变体 | 门控健康（最差种子） | 末点均值 | std | 低饱和比例 "
                  "| 高饱和比例 | 逐层均值范围 | 关键建议 |")
        md.append("|---|---|---|---|---|---|---|---|")
        md += gate_rows
        bad_gate = [r for r in gate_rows if "健康" not in r.split("|")[2]]
        if bad_gate:
            md.append("")
            md.append("⚠ 存在门控塌缩/饱和/常数化/漂移：对应的 PPL/RULER 差异"
                      "可能来自“门关闭/门透明”而不是所声称的机制；先按建议"
                      "修复门控再解释结果，必要时改报“机制未生效”。")
    else:
        md.append("- 暂无门控统计（旧日志或未评测门控变体）；门控均值不能"
                  "替代分布检查，建议用当前版本重跑训练内评测。")

    # --- 坏案例分析：预测错误的样本逐一归因（改进思路的最重要来源）
    md += ["", "## 坏案例分析（预测错误的样本与归因）", ""]
    bad_rows, bad_example_pool = [], []
    # 跨变体对照要求各变体使用同一批评测样本：取每个变体最小种子的错误
    # 清单（RULER 样本由配置里的评测种子构造，与训练种子无关，可逐条对齐）
    per_variant_badcases = cross_variant_badcase_map(runs, variants)
    corpus_sources = set()
    for v in variants:
        for s in _variant_seed_map(runs, v).values():
            r = s.get("ruler") or {}
            if r.get("corpus_source"):
                corpus_sources.add(str(r["corpus_source"]))
    for v in variants:
        m = _variant_seed_map(runs, v)
        cases = []
        for s in m.values():
            r = s.get("ruler")
            if r:
                badcases = r.get("badcases")
                if isinstance(badcases, list):
                    cases += badcases
        if not cases:
            continue
        ana = analyze_badcases(cases, variant=v)
        cats = sorted(ana["category_counts"].items(), key=lambda kv: -kv[1])
        cat_s = "；".join(f"{BADCASE_CN.get(k, k)}×{n}" for k, n in cats[:3])
        bad_rows.append((v, ana, cat_s))
        bad_example_pool += [(v, c) for c in ana["cases"]]
    if corpus_sources:
        src_s = "、".join(f"{x}.txt" for x in sorted(corpus_sources))
        if "train" in corpus_sources:
            md.append(f"⚠ RULER niah_corpus 干扰语料来自 **{src_s}**："
                      "训练文本进入评测上下文会让长上下文检索偏容易，"
                      "该任务的绝对分数与 held-out 语料运行不可比（见 ruler payload "
                      "的 `corpus_source`）。")
        else:
            md.append(f"RULER niah_corpus 干扰语料来源：{src_s}（held-out，"
                      "未使用训练文本）。")
        md.append("")
    if bad_rows:
        md.append("错误样本不是污点而是信息：逐条区分**标注错误**（gold 不在 "
                  "prompt 中→构造错误）、**难度边界样本**（needle 极深）与"
                  "**模型盲区**（误抄干扰值/幻觉数字/无数值回答）。"
                  "逐案例详情见 `badcases_*.md`。")
        md.append("")
        n_seeds_max = max((len(_variant_seed_map(runs, v)) for v in variants),
                          default=0)
        if n_seeds_max > 1:
            md.append(f"注：RULER 样本集跨种子相同（由评测构造种子决定），下表"
                      f"错误样本数为最多 **{n_seeds_max} 个种子的累计频次**"
                      "（同一样本在 k 个种子下错就计 k 次），用于衡量错误的"
                      "稳定性；唯一样本级的对照见下节「跨变体错误对照」"
                      "（取最小种子）。")
            md.append("")
        md.append("| 变体 | 错误样本数（跨种子累计） | 主要归因 | niah 深位(>0.75)失败占比 "
                  "| 标注/构造错误 |")
        md.append("|---|---|---|---|---|")
        for v, ana, cat_s in bad_rows:
            dfr = ana.get("deep_position_fail_ratio")
            md.append(f"| {VARIANT_CN.get(v, v)} | {ana['n_badcases']} | {cat_s} "
                      f"| {'-' if dfr is None else f'{dfr:.0%}'} "
                      f"| {ana['label_errors'] or 0} |")
        md.append("")
        # 摘 3 条代表性个案放进汇总页：错误必须"看得见"，不能只留在文件里
        shown, seen_cat = 0, set()
        for v, c in bad_example_pool:
            if shown >= 3 or c["category"] in seen_cat:
                continue
            seen_cat.add(c["category"])
            shown += 1
            q = _as_text(c.get("question")).strip().replace("\n", " ")[:120]
            pred = _as_text(c.get("pred")).strip().replace("\n", " ")[:80]
            md += [f"- **{VARIANT_CN.get(v, v)} · {c.get('task') or '-'}"
                   f"@ctx{c.get('ctx')}** "
                   f"—— {BADCASE_CN.get(c['category'], c['category'])}",
                   f"  - 问题：{q}…",
                   f"  - 输出：`{pred}` vs gold `{c.get('answer')}`；"
                   f"{c.get('explanation', '')}"]
        md.append("")
        n_label = sum(a["label_errors"] for _, a, _ in bad_rows)
        if n_label:
            md.append(f"⚠ 共 {n_label} 条 gold 不在 prompt 中的样本：这是"
                      "构造/标注错误，不是模型失败，须先处理评测再下结论。")
        # 错误归因的终点是改进方向：把主导错误模式转成下一步可执行的实验，
        # 而不是停在分类表（坏案例是提出新思路的最重要来源）。
        _render_advice(md, [(v, r) for v, a, _ in bad_rows
                            for r in (a.get("recommendations") or [])],
                       "### 坏案例驱动的改进方向（RULER）")
    else:
        md.append("- 暂无错误样本记录（RULER 未运行或全部命中）。"
                  "全部命中时也建议人工抽查若干正确样本确认没有'碰巧命中'。")

    # --- 跨变体错误对照：共识失败（数据/难度）vs 方法特异性盲区
    bad_cross = None
    if len(per_variant_badcases) >= 2:
        bad_cross = cross_variant_analysis(per_variant_badcases,
                                           reference="standard")
        md += ["", "### 跨变体错误对照（共识失败 vs 方法特异性）", ""]
        n_shared = bad_cross["n_shared_all_failures"]
        shared_ci = bad_cross.get("shared_all_ci95")
        shared_ci_s = ""
        if (isinstance(shared_ci, (list, tuple)) and len(shared_ci) == 2
                and all(isinstance(x, (int, float)) for x in shared_ci)):
            shared_ci_s = (f"（Wilson 95% CI {shared_ci[0]:.0%}–"
                           f"{shared_ci[1]:.0%}）")
        md.append(
            f"各变体取最小种子的错误清单（样本集相同，可逐条对齐；RULER 已"
            f"评测但零错误的变体按“全部通过”计入，不会被误算作共识失败）：共 "
            f"{bad_cross['n_unique_failed_items']} 个不同的失败样本，"
            f"{n_shared} 个为全体（已评测）变体共识失败{shared_ci_s}——"
            "共识失败（尤其当 "
            "standard 也错）更可能是数据/构造/内在难度问题，不应算作某一"
            "方法的问题；相对 standard 的新增错误才是该方法的特异性盲区，"
            "优先逐条分析。")
        md += ["", "| 变体 | 错误样本数 | 全体共识失败 | 相对standard新增 "
                   "| 相对standard改对 | 配对p (Holm) |",
                   "|---|---|---|---|---|---|"]
        for v in bad_cross["variants"]:
            pv = (bad_cross["per_variant"] or {}).get(v) or {}

            def _cnt(x):
                return "-" if x is None else str(x)

            p_adj = pv.get("mcnemar_p_holm")
            p_s = (f"{p_adj:.3g}" if isinstance(p_adj, (int, float))
                   else "-")
            md.append(f"| {VARIANT_CN.get(v, v)} | {_cnt(pv.get('n_failed'))} "
                      f"| {_cnt(pv.get('n_shared_all'))} "
                      f"| {_cnt(pv.get('n_new_vs_reference'))} "
                      f"| {_cnt(pv.get('n_fixed_vs_reference'))} "
                      f"| {p_s} |")
        md.append(
            "注: 新增/改对是同一批样本上的配对计数，p 为精确 McNemar 检验"
            "（多候选 Holm 校正）；p≥0.05 时只能当方向性线索，不能据此"
            "断言方法优劣。")
        # 展示至多 2 条特异性新增错误（方法盲区的最直接线索）
        shown = 0
        for v in bad_cross["variants"]:
            if v == "standard" or shown >= 2:
                continue
            for c in (bad_cross["per_variant"][v].get("new_vs_reference") or []):
                if shown >= 2:
                    break
                shown += 1
                ctx_s = f"@ctx{c['ctx']}" if c.get("ctx") is not None else ""
                md.append(
                    f"- **{VARIANT_CN.get(v, v)} · {c.get('task')}{ctx_s} "
                    f"特异性新增错误**：问题「{str(c.get('excerpt') or '')[-100:]}」；"
                    f"输出 `{c.get('pred')}` vs gold `{c.get('answer')}`")
        if n_shared and bad_cross["shared_failures"]:
            c0 = bad_cross["shared_failures"][0]
            ctx_s = f"@ctx{c0['ctx']}" if c0.get("ctx") is not None else ""
            md.append(f"- 共识失败示例：{c0.get('task')}{ctx_s}，问题"
                      f"「{str(c0.get('excerpt') or '')[-100:]}」；gold "
                      f"`{c0.get('answer')}`（所有变体都错，先检查样本本身）")
        md.append("")
        md.append("注: 跨变体对照只使用各变体最小种子的错误清单以保证样本集一致，"
                  "多种子时不同种子的失败集合变化属正常波动，不在此表统计。")
        _render_advice(md, [("跨变体", r) for r in
                            (bad_cross.get("recommendations") or [])],
                       "### 坏案例驱动的改进方向（跨变体）", max_items=4)
    elif per_variant_badcases:
        md += ["", "（仅一个变体有错误样本，无法做跨变体对照）"]

    # --- 语言建模最差窗口（PPL 坏案例）：PPL 只反映平均，错误必须看得见
    md += ["", "## 语言建模最差窗口（预测错误的样本）", ""]
    # 逐变体归因用全部种子的窗口累计（错误模式必须跨种子稳定）；跨变体
    # 对齐仍只用最小种子的同一批窗口（样本集必须一致）。
    per_variant_windows = {}
    per_variant_windows_first = {}
    window_seed_counts = {}
    for v in variants:
        m = _variant_seed_map(runs, v)
        all_keyed, first_keyed = [], None
        for s in _sorted_seeds(m):
            cand = m[s].get("eval") or {}
            wins = cand.get("worst_windows")
            if not isinstance(wins, list) or not wins:
                continue
            # 评测窗口集合由 (val/test 长度, ctx, eval_iters) 决定，与训练
            # 种子无关：给每个窗口写入稳定 key，供跨变体对齐。
            keyed = []
            for w in wins:
                if not isinstance(w, dict):
                    continue
                w = dict(w)
                w.setdefault("split", cand.get("split"))
                w["sample_id"] = f"lm:{w.get('split') or '?'}#w{w.get('window')}"
                keyed.append(w)
            all_keyed.extend(keyed)
            window_seed_counts[v] = window_seed_counts.get(v, 0) + 1
            if first_keyed is None:
                first_keyed = keyed
        if all_keyed:
            per_variant_windows[v] = all_keyed
            per_variant_windows_first[v] = first_keyed
    win_cross = None
    if per_variant_windows:
        md.append("PPL 是所有窗口的平均，看不出「模型究竟错了哪句」。下表按窗口"
                  "末位交叉熵挑出各变体最差的一次 next-token 预测并归因："
                  "高置信错答=top1 概率≥0.5 却错（系统性盲区，含高置信复读）；"
                  "复读吸引子=低置信下首选续写退化为抄上下文已有串；低置信="
                  "模型在多个候选间平均（样本本身多解/噪声，或训练不足）。"
                  "逐案例详情见 `worstcases_*.md`。")
        n_win_seeds = max(window_seed_counts.values()) if window_seed_counts else 0
        if n_win_seeds > 1:
            md.append("")
            md.append(f"注：下表按最多 {n_win_seeds} 个种子累计最差窗口——同一窗口"
                      "在 k 个种子中都差就计 k 次，用于判断错误模式是否跨种子"
                      "稳定；跨变体对照仍取各变体最小种子的同一批窗口。")
        # 最差窗口的“分母”必须可见：只报 top-10 而不报候选/总窗口数，会把
        # 候选子集里的最差误读成全量最差，也会让不同覆盖率的变体不可比。
        scopes = set()
        for v in variants:
            for s in _sorted_seeds(_variant_seed_map(runs, v)):
                cand = (_variant_seed_map(runs, v)[s].get("eval") or {})
                sc = cand.get("worst_window_scope")
                screened = (_integral_value(sc.get("screened"))
                            if isinstance(sc, dict) else None)
                total = (_integral_value(sc.get("total"))
                         if isinstance(sc, dict) else None)
                # 分母缺失/非法时不能写成 0 再按 100% 覆盖率展示：那会把
                # “未知筛查范围”伪装成“已覆盖全量窗口”，并输出 30/0。
                if (screened is not None and total is not None
                        and total > 0 and 0 <= screened <= total):
                    scopes.add((screened, total))
        if scopes:
            md.append("")
            lo_cov = min(scr / tot for scr, tot in scopes)
            shown = "；".join(f"{scr}/{tot}" for scr, tot in sorted(scopes)[:3])
            extra = " 等" if len(scopes) > 3 else ""
            md.append(f"注：最差窗口在等间隔候选窗口子集中排序（本次 "
                      f"{shown}{extra}，最低覆盖率 {lo_cov:.0%}），不是全量窗口"
                      "的严格最差；各变体独立选各自的候选窗口，跨变体的归因"
                      "占比是描述性对照，不是同一批样本的配对比较。")
        md += ["", "| 变体 | 最差窗口数 | 主要归因 | 平均 top-1 概率 "
                   "| gold 在上下文中 | gold 进 top-5 | 平均 loss |",
               "|---|---|---|---|---|---|---|"]
        window_pool, seen_cat, window_advice = [], set(), []
        for v, wins in per_variant_windows.items():
            ana = analyze_worst_windows(wins, variant=v)
            window_advice += [(v, r) for r in (ana.get("recommendations") or [])]
            cats = sorted(ana["category_counts"].items(), key=lambda kv: -kv[1])
            cat_s = "；".join(f"{WINDOW_CATEGORY_CN.get(k, k)}×{n}"
                              for k, n in cats[:3])
            mt1 = ("-" if ana["mean_top1_prob"] is None
                   else f"{ana['mean_top1_prob']:.3f}")
            gic = ("-" if ana["gold_in_context_rate"] is None
                   else f"{ana['gold_in_context_rate']:.0%}")
            g5 = ("-" if ana["gold_in_top5_rate"] is None
                  else f"{ana['gold_in_top5_rate']:.0%}")
            ml = ("-" if ana["mean_loss"] is None else f"{ana['mean_loss']:.3f}")
            md.append(f"| {VARIANT_CN.get(v, v)} | {ana['n_windows']} | {cat_s} "
                      f"| {mt1} | {gic} | {g5} | {ml} |")
            for c in ana["cases"]:
                if c["category"] not in seen_cat:
                    seen_cat.add(c["category"])
                    window_pool.append((v, c))
        # 摘 3 条不同类别的个案进汇总页
        for v, c in window_pool[:3]:
            tail = str(c.get("context_tail") or "").replace("\n", " ")[-100:]
            p1 = c.get("top1_prob")
            p1_s = f"（p={p1:.3f}）" if p1 is not None else ""
            if c.get("gold_in_context") is True:
                tag = "gold 在上下文中"
            elif c.get("gold_in_context") is False:
                tag = "gold 不在上下文中"
            else:
                tag = "gold 是否在上下文中未记录"
            md += [f"- **{VARIANT_CN.get(v, v)} · 窗口 #{c.get('window')}"
                   + (f"（{c['split']}）" if c.get("split") else "")
                   + f"** —— {WINDOW_CATEGORY_CN.get(c['category'], c['category'])}",
                   f"  - 上下文尾部：…{tail}",
                   f"  - 模型 top-1：`{c.get('top1')}`{p1_s}；真实续写："
                   f"`{c.get('gold_next')}`（{tag}）"]
        if len(per_variant_windows_first) >= 2:
            win_cross = cross_variant_analysis(per_variant_windows_first,
                                               reference="standard")
            n_shared_w = win_cross["n_shared_all_failures"]
            md.append("")
            md.append(
                f"跨变体看：{win_cross['n_unique_failed_items']} 个不同的最差窗口，"
                f"其中 {n_shared_w} 个为全体变体共识失败（更可能是数据的局部"
                "不可预测/噪声，而非某一模型的问题）；"
                + (f"⚠ 参照 standard 无最差窗口记录，新增/改对计数不可得。"
                   if not win_cross.get("reference_available") else
                   "相对 standard 的新增窗口才是该方法引入的预测退化。"))
            window_advice += [("跨变体", r) for r in
                              (win_cross.get("recommendations") or [])]
        _render_advice(md, window_advice,
                       "### 坏案例驱动的改进方向（最差窗口）", max_items=4)
    else:
        md.append("- 暂无最差窗口记录（评测未含 --worst_n、评测被跳过，或 test "
                  "split 默认尘封最差窗口转储——需人工迭代请在 val 上做，"
                  "确需 test 错误清单须显式 --worst_on_test 并在报告中披露）。"
                  "PPL 平均分无法替代对错误样本的逐条检查。")

    # --- 消融矩阵（全因子 + 去一法）
    present = set(runs.keys())
    md += ["", "## 消融矩阵（全因子 + 去一法）", ""]
    rows_by_var = {r["variant"]: r for r in rows}
    factorial_done, factorial_missing, factorial_under = _ablation_coverage(
        FACTORIAL_CELLS, present, rows_by_var)
    loo_done, loo_missing, loo_under = _ablation_coverage(
        LOO_CELLS, present, rows_by_var)
    control_done, control_missing, control_under = _ablation_coverage(
        CONTROL_CELLS, present, rows_by_var)
    underpowered_cells = factorial_under + loo_under + control_under

    def _cell_label(name, v):
        r = rows_by_var.get(v)
        if not r:
            return name, f"{v}（未运行）"
        n = _integral_value(r.get("n_seeds"))
        if n is None or n < 0:
            n = 0
        if n < MIN_SEEDS:
            return name, f"{v}（种子不足 n={n}<{MIN_SEEDS}）"
        return name, f"{VARIANT_CN.get(v, v)}（n={n}）"

    md.append("| 组合 | 变体 | PPL↓ | RULER平均↑ |")
    md.append("|---|---|---|---|")
    for name, v in FACTORIAL_CELLS:
        label, cell = _cell_label(name, v)
        r = rows_by_var.get(v)
        if r:
            md.append(f"| {label} | {cell} | "
                      f"{fmt_stat_cell(r, 'val_ppl')} | "
                      f"{fmt_stat_cell(r, 'ruler_avg')} |")
        else:
            md.append(f"| {label} | {cell} | - | - |")
    for name, v in CONTROL_CELLS:
        label, cell = _cell_label(name, v)
        r = rows_by_var.get(v)
        if r:
            md.append(f"| {label} | {cell} | "
                      f"{fmt_stat_cell(r, 'val_ppl')} | "
                      f"{fmt_stat_cell(r, 'ruler_avg')} |")
        else:
            md.append(f"| {label} | {cell} | - | - |")
    if (not factorial_missing and not loo_missing and not control_missing
            and not underpowered_cells):
        md.append("")
        md.append("全因子（门控内 2^3，none=`gate_only`）覆盖 8/8、去一法覆盖完整、"
                  "无门控 `standard` 外部基线已运行，且每格种子数≥"
                  f"{MIN_SEEDS}；但组合覆盖只是必要条件，"
                  "组件是否有用须以下方统计证据为准。`gate_only` 只含可学习门偏置 d，"
                  "是“有门但无 A/B/C”的对照：standard vs gate_only 度量门本身，"
                  "X-only vs gate_only 才是组件 X 的受控效应。")
    elif not factorial_missing and not loo_missing:
        md.append("")
        md.append("全因子（门控内 2^3）8/8、去一法完整；⚠ 缺少无门控 `standard` "
                  "外部基线：H1（LEGA vs 标准注意力）无法估计。组件 A/B/C 的判定"
                  "不依赖 standard，仍以 `gate_only` 为受控对照。")
    elif not factorial_missing:
        md.append("")
        md.append("⚠ 全因子（门控内 2^3）覆盖 8/8，但去一法缺失组合："
                  + "、".join(f"{n}({v})" for n, v in loo_missing)
                  + "；Full−X 的组件结论不完整。")
    elif not loo_missing:
        md.append("")
        md.append("⚠ 已覆盖去一法（Full/Full−A/Full−B/Full−C），但全因子缺失/种子不足："
                  + "、".join(f"{n}({v})" for n, v in factorial_missing)
                  + "；涉及这些组合的单独组件结论不完整。")
    else:
        md.append("")
        md.append("⚠ **消融不完整**：全因子缺失/种子不足 "
                  + "、".join(f"{n}({v})" for n, v in factorial_missing)
                  + "；去一法缺失/种子不足 "
                  + "、".join(f"{n}({v})" for n, v in loo_missing)
                  + ("；无门控 standard 外部基线缺失/种子不足 "
                     + "、".join(v for _, v in control_missing)
                     if control_missing else "")
                  + "。仅凭部分组合无法支撑“每个组件都有用”，请补齐后重跑。")
    if underpowered_cells:
        md.append("")
        md.append("⚠ **消融格种子数不足**（"
                  + f"要求每格≥{MIN_SEEDS} 个种子，单种子无法做配对检验）："
                  + "、".join(f"{n} n={k}" for n, _v, k in underpowered_cells[:8])
                  + (" ..." if len(underpowered_cells) > 8 else "")
                  + "；这些格不参与覆盖完整性宣称。")

    # --- 单因素消融审计：需要单因素的实验必须每次只改一个因子。
    md += ["", "### 单因素消融审计（每次只改变一个因子）", ""]
    if sf_audit["ok"]:
        md.append("✅ 组件归因所用的 6 组消融对比均只改变一个门控因子"
                  "（去一法 Full−X vs Full，全因子增量 X-only vs `gate_only`）；"
                  "不存在“B-only/C-only vs Full”这类同时改两个因子却被当作"
                  "单组件证据的对比。")
    else:
        md.append("⚠ **单因素消融审计未通过**："
                  + "；".join(sf_audit["problems"])
                  + "。这些对比不能用于组件归因，需先修正变体/配对定义。")
    md.append("")
    md.append("| 对比 | 方向 | 改变的因子 | 实际因子差 |")
    md.append("|---|---|---|---|")
    direction_cn = {"remove": "去一法：去掉该因子",
                    "add": "全因子增量：只加该因子"}
    for c in sf_audit["contrasts"]:
        md.append(f"| {c['candidate']} vs {c['reference']} "
                  f"| {direction_cn.get(c['direction'], c['direction'])} "
                  f"| {c['factor']} "
                  f"| {', '.join(c['changed_factors'])} |")
    md.append("")
    md.append("注: 全因子 2^3 本身是多因子设计（用于 A×B/A×C/B×C 交互检验，"
              "见下表，属探索性）；组件主效应只从上述单因素边估计。"
              "各变体 vs `standard` 与 LEGA vs `qgate` 是整体方法对比，"
              "可同时含多个因子，不得把差异归因给某一个 A/B/C 组件。"
              "单因素审计结果同时写入 `significance.json` 的 "
              "`single_factor_audit`。")

    # --- 组件有效性判定：只有 Holm 校正后的显著证据才能支撑“组件有用”
    md += ["", "### 组件有效性判定（同种子配对 t 检验，Holm 校正后 p<0.05 才算证据）", ""]
    md.append("判定只用预注册主指标（A/B=PPL，C=RULER）；另一指标仅作探索性"
              "补充，不参与判定，避免两个指标族分别校正后取 OR 放大假阳性。")
    md.append("")
    md.append("| 组件 | 主指标 | 去一法 Full−X vs Full | 全因子 X vs gate_only | 判定 |")
    md.append("|---|---|---|---|---|")

    def _fmt_evidence(st):
        p = st.get("p_adj")
        ps = f"p_adj={p:.4f}" if p is not None else "p_adj=-"
        return f"{'✅' if st['supported'] else '—'} {ps}（{st['reason']}）"

    for c in comp:
        loo_cells = "；".join(f"{m}: {_fmt_evidence(c['loo'][m])}"
                             for m in _EVIDENCE_METRICS)
        add_cells = "；".join(f"{m}: {_fmt_evidence(c['add'][m])}"
                             for m in _EVIDENCE_METRICS)
        verdict = c["verdict"]
        if not c.get("training_budget_comparable", True):
            verdict += "（探索性：训练预算不等）"
        md.append(f"| {c['component']} | {c['primary_metric']} | {loo_cells} | "
                  f"{add_cells} | {verdict} |")
    supported = [c["component"] for c in comp if c["supported"]]
    unsupported = [c["component"] for c in comp if not c["supported"]]
    fragile = [c["component"] for c in comp if c.get("fragile")]
    inconsistent = [c["component"] for c in comp if not c.get("consistent")]
    sf_ok = bool(sf_audit["ok"])
    if not comp:
        md.append("")
        md.append("⚠ 无法判定组件有效性（无配对检验结果）。")
    elif (not unsupported and not fragile and not factorial_missing
          and not loo_missing and not inconsistent and sf_ok):
        md.append("")
        md.append("✅ 统计证据支持“各组件均有用”：A/B/C 三个组件均在预注册主指标上"
                  "有 Holm 校正后的显著且稳健证据，去一法与全因子两套消融一致，"
                  "且门控内全因子 8/8、去一法完整。")
    elif (not unsupported and not fragile and not factorial_missing
          and not loo_missing and sf_ok):
        md.append("")
        md.append("✅ 统计证据支持各组件有用，但仅部分组件同时得到两套消融支持："
                  + "、".join(inconsistent)
                  + "；另一套未检出可能只是功效不足，报告时不得写成"
                  "“两套消融交叉印证”。")
    elif (not unsupported and not factorial_missing and not loo_missing
          and fragile and sf_ok):
        md.append("")
        md.append("⚠ **不能下“各组件均有用”的稳健结论**："
                  + "、".join(fragile)
                  + " 的证据虽通过 Holm 校正，但 |Δ|<ΔSD（种子噪声内）或"
                  "留一方向翻转；这类显著结果只能作为探索性线索，"
                  "需增加种子/降低方差后重跑才能确认。")
    else:
        md.append("")
        if not sf_ok:
            md.append("⚠ **不能下组件有效性结论**：单因素消融审计未通过，"
                      "上述对比不满足“每次只改一个因子”，不能用于组件归因。")
        if unsupported:
            md.append("⚠ **不能下“各组件均有用”的结论**："
                      + "、".join(unsupported)
                      + " 缺少主指标上 Holm 校正后的显著且稳健证据（幅度不足、"
                      "方向相反、噪声内、证据冲突或种子数不够）；"
                      "组合齐全并不等于组件有效。")
        if fragile:
            md.append("⚠ 不稳健组件：" + "、".join(fragile)
                      + "（显著但效应落在种子噪声内或留一翻转）。")
        if factorial_missing:
            md.append("⚠ 全因子缺失组合："
                      + "、".join(f"{n}({v})" for n, v in factorial_missing)
                      + "，组件单独贡献的覆盖不完整。")
        if loo_missing:
            md.append("⚠ 去一法缺失："
                      + "、".join(f"{n}({v})" for n, v in loo_missing)
                      + "，Full−X 的组件结论不完整。")
    if control_missing:
        md.append("")
        md.append("⚠ 缺少无门控 `standard` 外部基线：H1（LEGA vs 标准注意力）"
                  "无法估计；这不影响门控内 A/B/C 的受控判定。")
    stopped_comp = _as_text_list((audit or {}).get("early_stop_runs"))
    if stopped_comp:
        md.append("")
        md.append(f"⚠ 早停导致实际训练量不等（{len(stopped_comp)} 个 run）："
                  "下方组件证据仍按配对检验数值报告，但属于探索性分析；"
                  "确认性结论需在相同实际步数下重跑。")
    md.append("")
    md.append("注: 去一法证据=去掉该组件的 Full−X 显著差于 Full（去掉有害）；"
              "全因子证据=仅含该组件的 X 显著优于 gate_only（只加它、且已有"
              "门偏置时仍有收益）。判定要求方向正确、Holm 校正后 p<0.05、"
              "种子数≥3，且效应不落在 |Δ|<ΔSD 的种子噪声内、留一法不翻转；"
              "任一套显著反向记“证据冲突”，以证据不足处理。")

    # --- 全因子交互效应（探索性）：主效应有用 ≠ 可加。A 的收益可能在已有
    #     B 时缩小（次可加）或放大（协同）；以 gate_only 为无组件基准，
    #     逐种子算 8 个 gated 单元格的 2^3 交互对比并检验。
    inter = [t for t in tests if t.get("interaction")]
    md += ["", "### 门控内 2^3 交互效应（探索性，独立 Holm 家族）", "",
           "以 `gate_only`（有门但无 A/B/C）为无组件基准，比较 8 个 gated "
           "单元格；交互=0 表示两个组件的效应可加，显著非零说明同时加入"
           "两者时收益不能按单组件加总。该检验未预注册方向，只报是否偏离"
           "可加性与 Δ 符号（PPL 越小越好、RULER 越大越好，符号解释随指标而变）。",
           "", "| 交互 | 指标 | Δ均值 ± Δ标准差 | n | p_adj(Holm) | "
           "结论 |", "|---|---|---|---|---|---|"]
    if inter:
        for t in inter:
            delta = _fmt_mean_std(t)
            padj = (f"{t['p_adj']:.4f}" if t.get("p_adj") is not None else "-")
            md.append(f"| {t['interaction']} | {t['metric']} | {delta} | "
                      f"{t['n']} | {padj} | {t.get('verdict', '-')} |")
    else:
        md.append("（缺少完整 8 格或全部种子不足，无法计算交互对比）")
    md.append("")
    md.append("注: 交互对比以 gate_only 为无组件基准，对第三个因子的两个水平"
              "取平均，逐种子单样本配对检验；n<3 或 Holm 校正后不显著只表示"
              "未检出，不等于交互不存在。")

    # --- 全因子 Shapley 分解（探索性）：单因素边依赖参照格，交互非零时
    #     不同参照格给出不同的“X 的贡献”；Shapley 对全部加入顺序取平均，
    #     且效率性保证 φ_A+φ_B+φ_C 恰等于 Full−none 的总效应。
    md += ["", "### 全因子 Shapley 效应分解（探索性，交互感知）", ""]
    md.append("单因素消融边回答“固定其他因子时删/加 X 会怎样”，但参照格不同"
              "时结论可能不同；Shapley 值把 Full−none 的总效应按所有加入顺序"
              "平均分摊给 A/B/C，天然把 A×B/A×C/B×C 交互平均进参与因子。"
              "PPL 口径在 −log(PPL) 上分解（比值尺度可加），φ>0 表示该因子"
              "降低 PPL，`exp(−φ)` 即其 PPL 几何均值比；RULER 用原始 EM。"
              "占比只在每个种子的总效应均为正时报告，否则留空、不强行归一化。"
              "该分解未预注册，属探索性证据，不替代预注册的单因素确认检验。")
    if shapley_tests:
        md += ["", "| 因子 | 指标 | φ均值 ± SD | 95% CI | 占总效应 | "
               "PPL几何比 | n | p_adj(Holm) | 结论 |",
               "|---|---|---|---|---|---|---|---|---|"]
        for t in shapley_tests:
            factor = t.get("shapley_factor_label") or t.get("shapley_factor")
            delta = _fmt_mean_std(t)
            ci = t.get("mean_diff_ci95")
            ci_s = (f"[{ci[0]:+.4f}, {ci[1]:+.4f}]" if ci else "-")
            share = (f"{t['shapley_share']*100:.1f}%"
                     if t.get("shapley_share") is not None else "-")
            ratio = (f"{t['shapley_ppl_ratio']:.4f}"
                     if t.get("shapley_ppl_ratio") is not None else "-")
            padj = (f"{t['p_adj']:.4f}"
                    if t.get("p_adj") is not None else "-")
            md.append(f"| {factor} | {t.get('shapley_metric')} | {delta} | "
                      f"{ci_s} | {share} | {ratio} | {t['n']} | {padj} | "
                      f"{t.get('verdict', '-')} |")
        md.append("")
        md.append("注: φ 的符号方向=“越大越好”（PPL 已转为 −log）；"
                  "“占总效应”为逐种子 Shapley 份额的均值；总效应为 Full−none，"
                  "效率性残差见 `significance.json` 的 "
                  "`shapley_efficiency_gap_max`（浮点误差量级）。"
                  "Shapley 显著不等于单因素边也显著：若两者方向冲突，"
                  "说明贡献依赖其他因子的水平（交互），必须在报告中并列披露。")
    else:
        md.append("")
        md.append("⚠ 无法计算 Shapley 分解：门控内 8 格未全部运行、种子不足，"
                  "或各格评测协议不一致；此时只能参考上面的单因素边与交互表。")

    # --- 推理期因果病灶消融（探索性）：训练期消融回答“重新训练时去掉
    #     组件会怎样”，本表回答“训练好的模型推理时是否真的使用该组件”。
    md += ["", "### 推理期因果消融（权重病灶，val split；探索性）", ""]
    md.append("训练期消融改变优化轨迹，不能证明最终前向仍在**使用**某组件；"
              "本表在同一 checkpoint、同一批 val 窗口上只改门控参数，"
              "逐窗口配对比较 ΔNLL=病灶−正常（>0 表示该机制在推理期有用）。"
              "`no_query/no_conc/no_len/no_evidence` 是训练期去一法的推理期"
              "对应；`gate_open/closed/constant_mean` 检验门控写回强度与逐位置"
              "变化；`head_permute` 在参数个数、分布、结构完全不变的前提下打乱"
              "头—证据对应，是容量解释的安慰剂对照；`sign_flip` 反转证据方向。"
              "默认只在 val 上做（test 尘封；val 同时是 checkpoint 选择集，"
              "故绝对水平偏乐观，本表只作机制定位）。跨种子 p 经 Holm 校正；"
              "窗口级 p 仅描述单 run，不作为确认性证据。")
    if causal_tests:
        md += ["", "| 变体 | 病灶 | 目标机制 | ΔNLL均值 ± SD | PPL比 | n种子 | "
               "逐种子胜负 | 留一 | p_adj(Holm) | 判定 |",
               "|---|---|---|---|---|---|---|---|---|---|"]
        for t in causal_tests:
            delta = _fmt_mean_std(t)
            ratio = (f"{t['mean_ppl_ratio']:.4f}"
                     + (" ⚠截断" if t.get("ppl_ratio_clamped") else "")
                     if t.get("mean_ppl_ratio") is not None else "-")
            wl = _seed_winloss_str(t)
            loo = _loo_stability_str(t)
            padj = (f"{t['p_adj']:.4f}"
                    if t.get("p_adj") is not None else "-")
            md.append(f"| {t['variant']} | {t['lesion_label']} | "
                      f"{t.get('target', '-')} | {delta} | {ratio} | {t['n']} | "
                      f"{wl} | {loo} | {padj} | {t.get('causal_verdict', '-')} |")
        md.append("")
        md.append("注: `PPL比=exp(ΔNLL均值)`（>1 表示病灶更差）；ΔNLL 与 PPL 比"
                  "只在同一变体/同一 checkpoint 内解释，不跨变体比较绝对值。"
                  "`no_op` 表示该组件训练后系数已为 0（推理期未使用），"
                  "此时“无差异”不是“无效应”的证据；`gate_closed` 是阳性对照，"
                  "若关闭注意力输出仍无差异，应先排查实现/评测而非解释方法。"
                  "训练期消融显著而推理期病灶无差异，说明组件收益可能来自训练"
                  "期的优化/正则化而非推理期必需；推理期显著而训练期不显著，"
                  "则提示训练期消融被再训练补偿。两种证据并列，不互相替代。"
                  "`no_len` 只去 b·log n；c 中的 log n 仍随 a·c 保留，"
                  "因此它检验的是“额外长度系数 C”，不是“完全移除长度依赖”。")
    else:
        md.append("")
        md.append("⚠ 未发现推理期因果消融结果（`--skip_ablation`、"
                  "`--ablate_variants` 为空，或变体不含输出门）；"
                  "组件结论目前只有训练期消融证据，不能据此声称“推理期使用了"
                  "该组件”。")

    # --- 关键病灶的等价性解读（尤其安慰剂/阳性对照）：p>0.05 不能直接
    #     写成“安慰剂无效应”，必须看 TOST 是否把差异限制在 SESOI 内。
    md += _render_causal_equivalence_section(causal_tests)

    # --- 训练期参数匹配安慰剂（容量/优化红利的对照）与同一 checkpoint 内的
    #     2^3 推理期 Shapley（与训练期 Shapley 并列）。
    md += _render_parameter_matched_control(
        param_control_test, capacity_only_test,
        ran=PARAM_CONTROL_VARIANT in runs)
    md += _render_inference_shapley(inference_shapley)
    md += _render_inference_interactions(inference_interactions, tests)

    # --- 配对检验表
    md += ["", "## 关键对比（同种子配对，t 检验 + 置换检验双重印证，Holm 校正）", ""]
    if tests:
        md.append("| 对比 | 类型 | 指标 | Δ均值 ± Δ标准差 | Δ中位数 | Δ 95% CI | d_z | "
                  "n | 胜/负/平 | 留一稳定 | p | p_adj(Holm) | "
                  "p_perm(Holm) | p_sign(Holm) | MDE | 结论 |")
        md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for t in tests:
            if t.get("interaction"):
                continue  # 交互效应单独成表，避免与主效应表重复
            if t.get("ruler_task"):
                continue  # 逐任务分解单独成表，避免主表被 5 个任务淹没
            if t.get("metric") == "PPL(log)":
                continue  # 比值尺度敏感性单独成表，避免与原始 PPL 重复
            if t.get("shapley_factor"):
                continue  # Shapley 分解单独成表，避免与单因素边重复
            if t.get("mean_diff") is None:
                continue
            if t.get("contrast_type") == "single_factor":
                typ = ("单因素 " + "/".join(t.get("changed_factors") or []))
            elif t.get("metric") in (AUROC_DIFF_METRIC, AP_DIFF_METRIC):
                typ = "同 run 机制对比"
            else:
                typ = "整体/多因素"
            std = t.get("std_diff")
            delta = (f"{t['mean_diff']:+.4f}"
                     + (f" ± {std:.4f}" if std is not None else ""))
            med = (f"{t['median_diff']:+.4f}"
                   if t.get("median_diff") is not None else "-")
            ci = t.get("mean_diff_ci95")
            ci_s = (f"[{ci[0]:+.4f}, {ci[1]:+.4f}]"
                    if ci else "-")
            p = f"{t['p']:.4f}" if t.get("p") is not None else "-"
            padj = (f"{t['p_adj']:.4f}" if t.get("p_adj") is not None else "-")
            ppadj = (f"{t['p_perm_adj']:.4f}"
                     if t.get("p_perm_adj") is not None else "-")
            psadj = (f"{t['p_sign_adj']:.4f}"
                     if t.get("p_sign_adj") is not None else "-")
            dz = f"{t['d_z']:+.2f}" if t.get("d_z") is not None else "-"
            mde = (f"{t['mde']:+.4f}" if t.get("mde") is not None else "-")
            wl = _seed_winloss_str(t)
            loo = _loo_stability_str(t)
            md.append(f"| {t['candidate']} vs {t['reference']} | {typ} "
                      f"| {t['metric']} | "
                      f"{delta} | {med} | {ci_s} | {dz} | {t['n']} | {wl} | "
                      f"{loo} | {p} | {padj} | {ppadj} | {psadj} | {mde} | "
                      f"{t.get('verdict', '-')} |")
        md.append("")
        md.append("注: **单因素** = 候选与参照只差一个门控因子（去一法 Full−X "
                  "vs Full，或全因子增量 X-only vs gate_only），只有这类对比"
                  "可用于组件归因；**整体/多因素**（各变体 vs standard、"
                  "LEGA vs qgate 等）回答总体性能，不能把差异归给单个组件。")
        md.append("")
        md.append("注: Δ = 候选 − 参照；PPL 越小越好、RULER/AUROC差 越大越好；"
                  "d_z = Δ均值/Δ标准差为配对 Cohen's d 效应量。"
                  f"p 为双侧配对 t 检验原始值，p_adj 为同一指标家族内 "
                  f"Holm–Bonferroni 校正值（控制多重比较的 FWER），"
                  f"结论一律基于 p_adj<{ALPHA} 且 n≥{MIN_SEEDS}；"
                  f"n<3 只报描述统计、不判定显著性。种子不足时请先补齐种子再下结论。")
        md.append("注: p_perm 为符号翻转置换检验（分布无关；n≤16 全枚举精确）"
                  "在同一指标家族内独立 Holm 校正后的值——种子只有 3~5 个时"
                  "t 检验的正态假设不可靠，置换检验不支持的“显著”会标注"
                  "⚠。**粒度边界**：置换双侧 p 值的理论下限是 2/2^n——"
                  "n=3 时 0.25、n=5 时 0.0625、n=6 才降至 0.03125：种子数 <6 "
                  "时“置换不显著”是粒度所限而非 t 检验一定虚警，此时显著性"
                  "完全依赖 t 的正态假设；n≥6 后两套检验不一致才是真实的"
                  "稳健性警报。Δ中位数为逐种子配对差中位数（对离群种子不敏感）；"
                  "p_sign 为忽略零差后的精确符号检验，在同一指标家族内独立"
                  "Holm 校正，仅作稳健性对照、不参与确认性判定：有效非零"
                  "配对数 nz<6 时其双侧精确 p 的下限 2/2^nz>0.05，"
                  "“不显著”可能只是粒度所限。Δ 95% CI 为配对差的 bootstrap 区间；MDE = 当前"
                  "种子数与方差下 80% 功效可检测的最小效应：观测差远小于 "
                  "MDE 的“不显著”是功效不足，不是无效应。效应/噪声比 = "
                  "|Δ|/ΔSD（ΔSD 为逐种子配对差的标准差）：|Δ|<ΔSD 的结果"
                  "在结论中标注为“种子/评测噪声内”的边缘证据，即使 p_adj "
                  "显著也不能当作稳健收益。")
        md.append("注: “胜/负/平”= 逐种子配对差为正/负/零的种子计数（不含任何"
                  "分布假设）；“留一稳定”= 删掉任一单个种子后均值符号不变，"
                  "❌ 表示结论由个别种子决定，即使 p_adj 显著也不稳健。")
    else:
        md.append("（无可配对的对比；请对多个变体运行同一组种子）")

    # --- SESOI/TOST：把“不显著”与“实际等价”分开。
    md += _render_equivalence_section(tests)

    # --- PPL 比值尺度敏感性（探索性）：log-PPL 几何均值比。
    log_tests = [t for t in tests
                 if t.get("metric") == "PPL(log)"
                 and t.get("mean_diff") is not None]
    if log_tests:
        md += ["", "### PPL 比值尺度敏感性（log-PPL 几何均值比；探索性）", "",
               "PPL 是比值尺度，算术均值差会被高 PPL 种子/变体主导；log-PPL "
               "配对差 Δ=mean(log PPL候选−log PPL参照) 的指数 exp(Δ) 即几何"
               "均值比（<1 表示候选更好），区间由配对差 bootstrap CI 取指数"
               "得到。该表只检验结论是否依赖尺度；预注册主判定仍是原始 PPL，"
               "不能在两种尺度中挑对主张有利的口径。", "",
               "| 对比 | n | ΔlogPPL | PPL几何比 | 比值95% CI | p_adj(Holm) | "
               "SESOI等价性(±1%) | 原始PPL判定 | 尺度结论 |",
               "|---|---|---|---|---|---|---|---|---|"]
        n_mismatch = 0
        for t in log_tests:
            raw = _find_test(tests, (t.get("candidate"), t.get("reference")),
                             "PPL")
            raw_verdict = (_verdict_head(raw.get("verdict"))
                           if isinstance(raw, dict) else "未运行")
            ci = t.get("ppl_ratio_ci95")
            ci_s = (f"[{ci[0]:.4f}, {ci[1]:.4f}]" if ci else "-")
            ratio = (f"{t['ppl_ratio']:.4f}"
                     if t.get("ppl_ratio") is not None else "-")
            padj = (f"{t['p_adj']:.4f}" if t.get("p_adj") is not None else "-")
            log_verdict = _verdict_head(t.get("verdict"))
            mismatch = (raw_verdict != log_verdict
                        and raw_verdict not in ("未运行", "未定")
                        and log_verdict != "未定")
            n_mismatch += int(mismatch)
            mark = ("⚠ 尺度结论不一致" if mismatch else log_verdict)
            eq = t.get("equivalence") if isinstance(t.get("equivalence"), dict) else {}
            eq_p = _eq_p_str(eq) if eq else "-"
            eq_s = ("-" if not eq else
                    f"{eq.get('label') or '-'}"
                    + ("" if eq_p == "-" else f"（p={eq_p}）"))
            md.append(
                f"| {t['candidate']} vs {t['reference']} | {t['n']} | "
                f"{_fmt_signed(t.get('mean_diff'))} | {ratio} | {ci_s} | {padj} | "
                f"{eq_s} | {raw_verdict} | {mark} |")
        md += ["", "注: 几何均值比只改变“平均”的尺度，不改变配对结构；"
               f"原始 PPL 与 log-PPL 的结论{('有' if n_mismatch else '没有')}"
               f"不一致（{n_mismatch} 组），不一致时必须同时报告并解释为"
               "尺度敏感性，不能静默选择其中一个。逐种子 log 差与几何比见 "
               "`significance.json` 的 `ppl_log_pairs`。"]

    # --- PPL 窗口级配对（探索性）：有限窗口抽样的评估不确定性。
    #     与上面的种子级 t 检验回答不同问题，不能互相替代。
    md += ["", "### PPL 窗口级配对（评估抽样不确定性；探索性稳健性）", "",
           "PPL 只在 `eval_iters` 个等间隔窗口上估计，种子级检验只看每个种子的"
           "聚合 PPL。各变体共享同一批窗口序号，以下逐窗口配对 `loss_candidate−"
           "loss_reference`，并用“先重采样种子、再重采样窗口”的分层 bootstrap "
           "给出平均 NLL 差的区间；窗口间空间相关被忽略，故只作探索性证据。"
           "`PPL比`=exp(ΔNLL)，<1 表示候选更好。"
           f"低于 {MIN_PPL_WINDOW_SEEDS} 个种子或每种子 {MIN_PPL_WINDOWS} "
           "个窗口时只报点估计/区间，不生成“显著”措辞（重采样自由度不足）。", ""]
    if ppl_window_tests:
        md += ["| 对比 | 种子 | 窗口数(最小–最大) | ΔNLL均值 | 分层bootstrap 95% CI | "
               "块bootstrap 95% CI | PPL比 | p_adj(Holm) | 种子级判定（供对照） | 结论 |",
               "|---|---|---|---|---|---|---|---|---|---|"]
        for t in ppl_window_tests:
            seed_t = _find_test(tests, (t.get("candidate"), t.get("reference")),
                                "PPL")
            seed_verdict = (seed_t.get("verdict") if isinstance(seed_t, dict)
                            else "未运行")
            ci = t.get("mean_diff_ci95") or [None, None]
            ci_s = (f"[{ci[0]:+.4f}, {ci[1]:+.4f}]"
                    if ci[0] is not None and ci[1] is not None else "-")
            bci = t.get("block_mean_diff_ci95") or [None, None]
            bci_s = (f"[{bci[0]:+.4f}, {bci[1]:+.4f}]"
                     if bci[0] is not None and bci[1] is not None else "-")
            ratio = (f"{t['ppl_ratio']:.4f}"
                     if t.get("ppl_ratio") is not None else "-")
            padj = (f"{t['p_adj']:.4f}"
                    if t.get("p_adj") is not None else "-")
            md.append(
                f"| {t['candidate']} vs {t['reference']} | {t['n_seeds']} | "
                f"{t['n_windows_min']}–{t['n_windows_max']} | "
                f"{_fmt_signed(t.get('mean_diff'))} | {ci_s} | {bci_s} | "
                f"{ratio} | {padj} | "
                f"{str(seed_verdict).split('（')[0]} | {t.get('verdict', '-')} |")
        md += ["", "注: ΔNLL 为负表示候选的窗口平均交叉熵更低（更好）。"
               "分层 bootstrap 的 p 值由中心化重采样分布计算，与种子级 t "
               "检验不是同一个零假设；块bootstrap 用移动块（块长 n^(1/3)）"
               "保留相邻窗口相关，与 i.i.d. 区间并报，以更宽者解读稳健性。"
               "两者冲突时说明“换训练种子”和“换评测窗口”给出不同信息，"
               "均不足以单独升级为确认性结论。逐种子窗口胜负明细见 "
               "`significance.json` 的 `ppl_window_pairs`。"]
    else:
        md.append("（缺少逐窗口损失（旧产物）或没有同 split/同协议的配对 run；"
                  "重新运行独立评测即可启用。）")

    # --- RULER 逐任务分解（探索性）：只渲染预注册 RULER 假设涉及的对比；
    #     全部逐任务检验仍完整保存在 significance.json（tests[*].ruler_task）。
    task_rows = [t for t in tests if t.get("ruler_task")]
    if task_rows:
        reg_ruler_pairs = {
            (h.get("candidate"), h.get("reference"))
            for h in (_as_dict(registered).get("hypotheses") or [])
            if isinstance(h, dict) and h.get("metric") == "RULER平均"}
        if not reg_ruler_pairs:
            reg_ruler_pairs = {("lega", "standard")}
        shown = [t for t in task_rows
                 if (t.get("candidate"), t.get("reference"))
                 in reg_ruler_pairs]
        md += ["", "### RULER 逐任务分解（探索性；综合平均的辅助诊断）", "",
               "综合平均可能把不同任务的方向相反效应相互抵消，因此对预注册"
               "RULER 对比逐任务各做一次同种子配对检验；每个任务是一个独立"
               "Holm 家族（探索性，不替代综合平均的确认性判定）。", ""]
        if shown:
            md += ["| 对比 | 任务 | Δ均值 ± Δ标准差 | 胜/负/平 | 留一稳定 | n | "
                   "p_adj(Holm) | 结论 |",
                   "|---|---|---|---|---|---|---|---|"]
            for t in shown:
                delta = _fmt_mean_std(t)
                padj = (f"{t['p_adj']:.4f}"
                        if t.get("p_adj") is not None else "-")
                wl = _seed_winloss_str(t)
                loo = _loo_stability_str(t)
                md.append(f"| {t['candidate']} vs {t['reference']} | "
                          f"{t['ruler_task']} | {delta} | {wl} | {loo} | "
                          f"{t['n']} | {padj} | {t.get('verdict', '-')} |")
        else:
            md.append("（预注册 RULER 对比未运行，无法分解）")
        md += ["", "注: 全部逐任务检验（含未渲染的对比）见 "
               "`significance.json` 中 metric 以 `RULER:` 开头的条目；这些"
               "检验未预注册方向，只作机制/定位线索，需在新数据上确认。"
               "“胜/负/平”为逐种子配对差符号计数；“留一稳定”= 删掉任一单个"
               "种子后均值符号不变，❌ 表示结论由个别种子决定。"]

    # --- RULER 样例级配对（探索性稳健性证据）：固定题面上的 McNemar。
    #     跨种子 t 检验只看每个种子的聚合 EM，n=20~30 的评测抽样误差
    #     没有进入检验；同一 sample_id 配对能利用“候选对/参照错”的方向
    #     信息，回答不同的统计问题。它不是确认性检验，也不替代种子级判定。
    md += ["", "### RULER 样例级配对（McNemar 精确检验；探索性稳健性）", "",
           "跨种子聚合 EM 检验回答“换训练种子是否稳定”；每格只有 20~30 个"
           "样例，以下用变体间共享的 `sample_id` 做逐样例配对，回答“在这批"
           "固定题面上候选是否更常答对”。只统计结果不同的样对（候选独对 vs "
           "参照独对），相同结果不提供方向信息；多种子按 discordant 计数合并，"
           "独立 Holm 家族。", ""]
    if ruler_sample_tests:
        md += ["| 对比 | 配对种子 | 配对样例 | 候选独对/参照独对 | "
               "ΔEM micro(候选−参照) | ΔEM macro(任务等权) | "
               "ΔEM macro 95% CI | 种子胜/负/平 | p_adj(Holm) | "
               "种子级判定（供对照） | 结论 |",
               "|---|---|---|---|---|---|---|---|---|---|---|"]
        for t in ruler_sample_tests:
            seed_t = _find_test(tests, (t.get("candidate"), t.get("reference")),
                                "RULER平均")
            seed_verdict = (seed_t.get("verdict") if isinstance(seed_t, dict)
                            else "未运行")
            delta = (f"{t['mean_diff']:+.4f}"
                     if t.get("mean_diff") is not None else "-")
            macro_delta = (f"{t['macro_mean_diff']:+.4f}"
                           if t.get("macro_mean_diff") is not None else "-")
            mci = t.get("macro_mean_diff_ci95")
            mci_s = (f"[{mci[0]:+.4f}, {mci[1]:+.4f}]" if mci else "-")
            padj = (f"{t['p_adj']:.4f}"
                    if t.get("p_adj") is not None else "-")
            md.append(
                f"| {t['candidate']} vs {t['reference']} | {t['n_seeds']} | "
                f"{t['n_samples']} | {t['candidate_only']}/{t['reference_only']} | "
                f"{delta} | {macro_delta} | {mci_s} | {t['n_seed_wins']}/"
                f"{t['n_seed_losses']}/{t['n_seed_ties']} | {padj} | "
                f"{str(seed_verdict).split('（')[0]} | {t.get('verdict', '-')} |")
        md += ["", "注: McNemar 只使用 discordant 对，n_samples 是参与配对的总样例"
               "数而非独立证据数；多个训练种子重复使用同一批 sample_id，合并"
               "检验只作探索性证据。`ΔEM micro` 样例等权、与 McNemar 池化计数"
               "对应；`ΔEM macro` 任务等权、任务内上下文档位等权，与预注册 "
               "`RULER平均` 及上表种子级"
               "判定同一估计量。`ΔEM macro 95% CI` 固定已观测种子，在每个"
               "(任务, 上下文) 单元格内重采样配对样例，再任务内、任务间等权平均，只覆盖"
               "评测抽样不确定性；种子数很少时该区间可能很窄，不能替代种子级"
               "检验。两者方向不一致时，结论依赖“平均值”定义，不能"
               "只挑对主张有利的口径。若样例级显著而种子级不显著（或反之），"
               "说明两个不确定性来源给出不同信息：前者多为评测样本量/题目"
               "难度，后者为训练随机性，均不能单独升级为确认性结论。"
               "逐种子明细见 `significance.json` 的 `ruler_sample_pairs`。"]
        cluster_rows = [t for t in ruler_sample_tests
                        if t.get("cluster_p") is not None]
        if cluster_rows:
            md += ["", "**样例聚类稳健性（每个 sample_id 先跨种子平均；"
                   "避免同一样例被多个种子重复计数）**", "",
                   "| 对比 | 独立样例 | 聚类ΔEM | 95% CI | p_adj(Holm) | 判定 |",
                   "|---|---|---|---|---|---|"]
            for t in cluster_rows:
                ci = t.get("cluster_mean_diff_ci95") or [None, None]
                ci_s = (f"[{ci[0]:+.4f}, {ci[1]:+.4f}]"
                        if ci[0] is not None and ci[1] is not None else "-")
                padj = (f"{t['cluster_p_adj']:.4f}"
                        if t.get("cluster_p_adj") is not None else "-")
                md.append(
                    f"| {t['candidate']} vs {t['reference']} | "
                    f"{t.get('cluster_n')} | "
                    f"{_fmt_signed(t.get('cluster_mean_diff'))} | "
                    f"{ci_s} | {padj} | {t.get('cluster_verdict', '-')} |")
            md += ["", "注: 聚类检验先把每个 sample_id 在多个训练种子下的配对差"
                   "取平均，再以样例为独立单位做符号翻转（n≤16 精确，否则固定"
                   "种子 Monte-Carlo）。它只覆盖样例等权 micro 口径，不校正"
                   "macro；与池化 McNemar 的差异本身说明“重复种子被当成独立"
                   "证据”的程度，两者应并列解读。"]
    else:
        md.append("（缺少逐样例 EM 或没有协议一致的配对 run；旧产物需重跑 "
                  "RULER 才能启用该分析。）")

    # --- 注册假设判定：confirmatory 与 exploratory 的边界必须是显式的
    md += ["", "## 注册假设判定（confirmatory；其余对比均为事后探索）", ""]
    md += _render_registration_integrity(integrity, scope_check=scope_check)
    if hyp_eval:
        n_sup = sum(1 for h in hyp_eval if h["status"] == "支持")
        n_ref = sum(1 for h in hyp_eval if h["status"] == "拒绝")
        n_und = sum(1 for h in hyp_eval if h["status"] == "未定")
        n_untested = sum(1 for h in hyp_eval if h["status"] == "未检验")
        counts = f"支持 {n_sup}、拒绝 {n_ref}、未定 {n_und}"
        if n_untested:
            # “未检验”（对比未运行/指标对不上）与“未定”是不同状态，
            # 必须显式计数，否则四项之和小于登记条数、读者会误以为漏算。
            counts += f"、未检验 {n_untested}"
        valid = all(h.get("confirmatory_valid") for h in hyp_eval)
        md.append(f"预注册文件 `preregistration/hypotheses.json`（登记于正式实验前；"
                  f"不放在 docs/ 以免被流水线自动清理删除）："
                  f"{len(hyp_eval)} 条方向性主张，判定 {counts}。"
                  "只有“支持”能写进结论；“拒绝”= 方向相反且校正后"
                  "显著（假设被证伪）；“未定/未检验”不得解释为任何方向的证据。")
        if not valid:
            md.append("")
            md.append("> ⛔ 预注册完整性校验未通过：下表的“支持/拒绝”不作数，"
                      "已按探索性线索处理（见上方完整性小节）。")
        md.append("")
        md.append("| ID | 主张 | Δ均值 | n | p_adj(家族) | p_adj(全体确认族) | "
                  "p_perm | MDE | 判定 |")
        md.append("|---|---|---|---|---|---|---|---|---|")
        n_global_downgraded = 0
        for h in hyp_eval:
            padj = (f"{h['p_adj']:.4f}" if h.get("p_adj") is not None else "-")
            pg = (f"{h['p_adj_confirmatory']:.4f}"
                  if h.get("p_adj_confirmatory") is not None else "-")
            if (h.get("status_family") in ("支持", "拒绝")
                    and h.get("status") == "未定"
                    and h.get("p_adj_confirmatory") is not None):
                n_global_downgraded += 1
            pp = (f"{h['p_perm_adj']:.4f}"
                  if h.get("p_perm_adj") is not None else "-")
            delta = _fmt_signed(h.get("mean_diff"))
            mde = f"{h['mde']:+.4f}" if h.get("mde") is not None else "-"
            mark = {"支持": "✅", "拒绝": "❌"}.get(h["status"], "—")
            if not h.get("confirmatory_valid"):
                mark = "⚠（无效）"
            md.append(f"| {h['id']} | {h['claim']} | {delta} | "
                      f"{h.get('n') if h.get('n') is not None else '-'} | "
                      f"{padj} | {pg} | {pp} | {mde} | {mark} {h['status']}"
                      f"（{h['detail']}） |")
        md.append("")
        if n_global_downgraded:
            md.append(f"注: {n_global_downgraded} 条主张在指标家族内显著，但"
                      "跨全部预注册假设的全局 Holm 校正后不显著，已按确认族"
                      "整体 FWER 降级为“未定”；status_family 仍保留原家族判定"
                      "供审计。")
        md.append("注: p_adj(家族) 为同一指标家族内 Holm；p_adj(全体确认族) "
                  "为全部可检验预注册假设（n≥"
                  f"{MIN_SEEDS}）组成的确认族 Holm，控制“任意一条确认性结论"
                  "为假阳性”的整体 FWER。判定要求两者均 < "
                  f"{ALPHA}：仅家族内显著不足以升级为确认性结论。"
                  "未登记在本文件里的对比（含本汇总表中的其余行）一律属"
                  "事后（exploratory）分析：p 值只受本表家族内校正保护，"
                  "其“显著”只能作为提出新假设的线索；要升级为结论须先登记"
                  "再在新的种子/数据上检验。")
    else:
        md.append("- ⚠ 未找到可用的注册假设文件（`preregistration/hypotheses.json`）。"
                  "本页全部对比均为事后分析：没有预注册边界，任何“显著”都只能"
                  "按探索性线索对待。请登记方向性主张后重跑正式实验。")

    # --- 门控开销：逐头参数与额外计算必须如实核算，不能只报“涨点”。
    # 参数是实际注册参数（gate_extra_params），FLOPs 是 latency.json 的解析
    # 估计（见 estimate_flops_per_token）；两者都取全部变体公共 ctx，避免
    # 拿不同上下文的 FLOPs 比较。
    md += ["", "## 门控开销核算（逐头参数与额外计算）", ""]
    md.append("门控/空注意力的开销按**逐头参数量**与**解析 FLOPs**分列，"
              "与收益（PPL/RULER）分开报告：参数量取自实际注册参数，"
              "FLOPs 为 batch=1 prefill 的解析估计（matmul 2·MACs + "
              "注意力 4·kv_len·d/层 + 门控项），不是实测延迟。")
    md.append("")
    md.append("| 变体 | 门控参数/头 | 门控额外参数 | 占总参数 | "
              "门控 FLOPs/token | 占总 FLOPs | 说明 |")
    md.append("|---|---|---|---|---|---|---|")
    for r in rows:
        v = r.get("variant")
        per_head = _num_or_none(r.get("n_gate_per_head"))
        extra = _num_or_none(r.get("gate_extra"))
        params = _num_or_none(r.get("params"))
        # 参数/FLOPs 来自外部结果文件：负参数、0/极小分母或负开销都会把
        # “占比”算成 inf/-inf。占比只在两端都是非负有限且分母 >0 时定义，
        # 结果非有限则按缺失处理，不能写进汇总表/JSON。
        param_pct = None
        if (extra is not None and extra >= 0.0
                and params is not None and params > 0.0):
            param_pct = extra / params * 100.0
            if not math.isfinite(param_pct):
                param_pct = None
        e = lat_entries.get(v)
        gate_flops = (_eff_cell(e, "gate_flops", common_ctx)
                      if common_ctx is not None else None)
        total_flops = (_eff_cell(e, "flops", common_ctx)
                       if common_ctx is not None else None)
        flop_pct = None
        if (gate_flops is not None and gate_flops >= 0.0
                and total_flops is not None and total_flops > 0.0):
            flop_pct = gate_flops / total_flops * 100.0
            if not math.isfinite(flop_pct):
                flop_pct = None
        r["gate_param_pct"] = (round(param_pct, 4)
                               if param_pct is not None else None)
        r["gate_flops_per_tok"] = gate_flops
        r["gate_flops_pct"] = (round(flop_pct, 4)
                               if flop_pct is not None else None)
        note = ("无门控（外部基线）" if v in ("standard", "standard_sdpa")
                else "逐头 w/a/b/d 或 null 槽" if (extra or 0) > 0 else "-")
        md.append(
            f"| {VARIANT_CN.get(v, v)} "
            f"| {'-' if per_head is None else f'{per_head:g}'} "
            f"| {'-' if extra is None else f'{extra:,.0f}'} "
            f"| {'-' if param_pct is None else f'{param_pct:.4f}%'} "
            f"| {'-' if gate_flops is None else f'{gate_flops:,.0f}'} "
            f"| {'-' if flop_pct is None else f'{flop_pct:.4f}%'} "
            f"| {note} |")
    md.append("")
    md.append("逐头参数 = 总额外参数 / (层数 × 头数)，来自实际注册的 "
              "gate_w/gate_a/gate_b/gate_d 或 k_null/v_null；LEGA 每头 "
              "d_head+3，qgate 每头 d_head+1，null 每头 2·d_head，"
              "gate_only 每头 1。**收益必须对照这两列开销解读**："
              "“门控参数占比很小”不等于“额外计算可忽略”——集中度项要扫描 "
              "每个头的 max_j p_j，其 FLOPs 随上下文长度线性增长。"
              + ("当前公共 ctx=" + str(common_ctx)
                 + "；不同 ctx 的数值见 latency.json。" if common_ctx
                 else "当前缺少公共 ctx 的 latency 数据，FLOPs 列不可得。"))

    # --- 涨点-增幅权衡：评价改进不能只看指标涨点
    md += ["", "## 涨点-增幅权衡（涨点必须对照增幅）", ""]
    # 运行时基线与上表 prefill_rel_pct 同一策略：优先 standard_sdpa（与全部
    # 候选的 SDPA 推理路径可比），缺失时退回显式 standard。
    # 训练步时与进程峰值两列仍取 standard 训练产物（只有它有训练记录）。
    # 注：standard_sdpa 不在 VARIANTS 里，只存在于 latency.json 的变体表。
    # 必须与上表 prefill_rel_pct 的选取口径一致：不仅要存在该变体，还要在
    # 公共 ctx 上真有 prefill 数据，否则会选中一个在该 ctx 无数据的基线，
    # 令 Prefill/Decode 增幅整体缺失、低估开销并可能翻转权衡判定。
    lat_base_name = next((n for n in ("standard_sdpa", "standard")
                          if _prefill_at(lat_variants.get(n),
                                         common_ctx) is not None), None)
    if common_ctx is not None and lat_base_name:
        base_e = lat_variants[lat_base_name]
        base_prefill = _prefill_at(base_e, common_ctx)
        base_decode = _lat_metric_at(base_e, "decode_ms_per_tok", common_ctx)
        base_params = _eff_cell(base_e, "params", common_ctx)
        base_flops = _eff_cell(base_e, "flops", common_ctx)
        base_kv = _eff_cell(base_e, "kv_mem", common_ctx)
        md.append(f"基线={lat_base_name}（公共 ctx={common_ctx}、batch=1）。增幅="
                  "Prefill ms/Decode ms(tok)/训练 ms(step)/训练进程峰值/参数量/"
                  "FLOPs(token)/KV cache/量测分配量 相对基线增幅的最大值（Prefill/Decode "
                  "的差异若不超过两端重复测量的噪声散布则按不可区分处理，标 † "
                  "且不计入增幅）。收益覆盖两个主指标：PPL 相对改善（%）与 "
                  "RULER 平均 EM 改善（绝对百分点 pp）——预注册主张横跨两者"
                  "（H1 PPL、H4 长上下文检索），只看 PPL 会误判长上下文涨点，"
                  "也会漏掉 RULER 显著退步。判定规则：两个主指标都无数据或"
                  "增幅缺失→无法判定；任一参与判定的指标种子不足（n<3）→无法"
                  "判定；某一主指标显著差于基线（Holm 校正后 p<α）→不可行，"
                  "但另一主指标同时显著改善时判“权衡”并显式说明取舍；任一主"
                  "指标显著改善即有涨点（按显著者中最不利的收益套用收支比："
                  "PPL >5×收益、RULER 单独 >10×收益且下限 0.5 pp，单位不同不"
                  "混算），涨点不显著→说服力有限（增幅>10% 判不可行）；显著但"
                  "增幅>50% 或超过上述收支比 →收益-增幅比不利/需审视。微小改善"
                  "伴随巨大增幅的方案会被明确拒绝。")

        def _train_step_ms(v):
            """该变体各种子训练步时的中位数（毫秒/步；旧 eval.json 无此字段→None）。

            同一 run_all 内所有变体在同一台机器、同一批次下训练，该步时是
            同口径的直接可比时序（评测与 checkpoint 写入已从区间计时中
            扣除）；中位数避免首步冷启动与偶发卡顿污染。
            """
            vals = []
            for s in _variant_seed_map(runs, v).values():
                x = _finite_number((s.get("eval") or {}).get(
                    "train_step_time_ms"))
                if x is not None and x >= 0:
                    vals.append(x)
            if not vals:
                return None
            vals.sort()
            n = len(vals)
            if n % 2:
                return vals[n // 2]
            # 合法但接近 float 上限的计时/内存中位数：先加后除会溢出。
            return vals[n // 2 - 1] / 2.0 + vals[n // 2] / 2.0

        def _train_peak_mem_mb(v):
            """该变体各种子训练进程峰值的跨种子中位数（MB；旧 eval.json 无→None）。

            与推理单次前向的分配值分开报告：两者口径不同（进程级高水位
            含优化器状态与激活 vs prefill 单次分配）。
            """
            vals = []
            for s in _variant_seed_map(runs, v).values():
                x = _finite_number((s.get("eval") or {}).get(
                    "train_peak_mem_mb"))
                if x is not None and x >= 0:
                    vals.append(x)
            if not vals:
                return None
            vals.sort()
            n = len(vals)
            if n % 2:
                return vals[n // 2]
            # 合法但接近 float 上限的计时/内存中位数：先加后除会溢出。
            return vals[n // 2 - 1] / 2.0 + vals[n // 2] / 2.0

        md.append("")
        md.append("| 变体 | PPL 相对基线 | p_adj(PPL) | RULER Δ(pp) | p_adj(RULER) "
                  "| Prefill ms | Decode ms/tok "
                  "| 训练 ms/step | 训练峰值 | 参数量 | FLOPs/token | KV cache "
                  "| 量测分配 | 判定 |")
        md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        ppl_test = {t["candidate"]: t for t in tests
                    if t.get("reference") == "standard" and t.get("metric") == "PPL"
                    and t.get("mean_diff") is not None}
        ruler_test = {t["candidate"]: t for t in tests
                      if t.get("reference") == "standard"
                      and t.get("metric") == "RULER平均"
                      and t.get("mean_diff") is not None}
        base_train_ms = _train_step_ms("standard")
        base_train_mem = _train_peak_mem_mb("standard")
        n_metric_within_noise = 0
        for r in rows:
            v = r["variant"]
            # 基线行（standard）不参与涨点-增幅判定；standard_sdpa 不在
            # VARIANTS 里，根本不会出现在 rows 中。
            if v == "standard" or not lat_entries.get(v):
                continue
            e = lat_entries[v]
            ppl = r.get("val_ppl")
            std_row = next((x for x in rows if x["variant"] == "standard"), None)
            std_ppl = (std_row or {}).get("val_ppl")
            ppl_rel = _rel_pct(std_ppl, ppl) if (ppl is not None and std_ppl) else None
            benefit = -ppl_rel if ppl_rel is not None else None
            t = ppl_test.get(v)
            p_adj = t.get("p_adj") if t else None
            n_ppl = t.get("n") if t else None
            # RULER 平均 EM 的配对差是比例（0~1），换算为绝对百分点后进入
            # 权衡判定：与 PPL 的相对百分比单位不同，绝不混算收益比。
            rt = ruler_test.get(v)
            ruler_benefit_pp = (rt["mean_diff"] * 100.0
                                if rt and rt.get("mean_diff") is not None
                                else None)
            ruler_p_adj = rt.get("p_adj") if rt else None
            n_ruler = rt.get("n") if rt else None
            pre_delta = _rel_pct(base_prefill, _prefill_at(e, common_ctx))
            dec_v = _lat_metric_at(e, "decode_ms_per_tok", common_ctx)
            dec_delta = _rel_pct(base_decode, dec_v)
            tr_v = _train_step_ms(v)
            tr_delta = _rel_pct(base_train_ms, tr_v)
            memtr_v = _train_peak_mem_mb(v)
            memtr_delta = _rel_pct(base_train_mem, memtr_v)
            par_delta = _rel_pct(base_params, _eff_cell(e, "params", common_ctx))
            flo_delta = _rel_pct(base_flops, _eff_cell(e, "flops", common_ctx))
            mem_v = _eff_cell(e, "mem", common_ctx)
            mem_b = _eff_cell(base_e, "mem", common_ctx)
            mem_delta = _rel_pct(mem_b, mem_v) if (mem_v is not None and mem_b) else None
            kv_v = _eff_cell(e, "kv_mem", common_ctx)
            kv_delta = _rel_pct(base_kv, kv_v)
            # 运行时差异的测量噪声：|Δ| 不超过两端散布的方和根时不可区分，
            # 按 0 计入增幅（原始值仍保留在 JSON，避免把噪声当真实差异）。
            pre_noise = _noise_floor_pct(
                _lat_noise_pct(base_e, "prefill", common_ctx),
                _lat_noise_pct(e, "prefill", common_ctx))
            dec_noise = _noise_floor_pct(
                _lat_noise_pct(base_e, "decode", common_ctx),
                _lat_noise_pct(e, "decode", common_ctx))
            pre_within = (pre_delta is not None and pre_noise is not None
                          and abs(pre_delta) <= pre_noise)
            dec_within = (dec_delta is not None and dec_noise is not None
                          and abs(dec_delta) <= dec_noise)
            n_metric_within_noise += int(pre_within) + int(dec_within)
            deltas = [d for d in ((0.0 if pre_within else pre_delta),
                                  (0.0 if dec_within else dec_delta),
                                  tr_delta, memtr_delta, par_delta, flo_delta,
                                  kv_delta, mem_delta)
                      if d is not None]
            cost = max(deltas) if deltas else None
            verdict = tradeoff_verdict(benefit, p_adj, cost, n_seeds=n_ppl,
                                       ruler_benefit_pp=ruler_benefit_pp,
                                       ruler_p_adj=ruler_p_adj,
                                       ruler_n_seeds=n_ruler)
            r["eff_cost_max_pct"] = round(cost, 2) if cost is not None else None
            r["eff_flops_per_tok"] = _eff_cell(e, "flops", common_ctx)
            r["eff_prefill_delta_pct"] = (round(pre_delta, 2)
                                          if pre_delta is not None else None)
            r["eff_prefill_noise_pct"] = (round(pre_noise, 2)
                                          if pre_noise is not None else None)
            r["eff_prefill_within_noise"] = pre_within
            r["eff_decode_ms_per_tok"] = dec_v
            r["eff_decode_delta_pct"] = (round(dec_delta, 2)
                                         if dec_delta is not None else None)
            r["eff_decode_noise_pct"] = (round(dec_noise, 2)
                                         if dec_noise is not None else None)
            r["eff_decode_within_noise"] = dec_within
            r["eff_train_step_ms"] = tr_v
            r["eff_train_delta_pct"] = (round(tr_delta, 2)
                                        if tr_delta is not None else None)
            r["eff_train_peak_mem_mb"] = memtr_v
            r["eff_train_mem_delta_pct"] = (round(memtr_delta, 2)
                                            if memtr_delta is not None else None)
            r["eff_kv_cache_mb"] = kv_v
            r["eff_kv_cache_delta_pct"] = (round(kv_delta, 2)
                                           if kv_delta is not None else None)
            r["eff_peak_mem_mb"] = mem_v
            r["eff_benefit_ppl_pct"] = (round(benefit, 2)
                                        if benefit is not None else None)
            r["eff_ruler_p_adj"] = ruler_p_adj
            r["eff_benefit_ruler_pp"] = (round(ruler_benefit_pp, 3)
                                         if ruler_benefit_pp is not None else None)
            r["eff_verdict"] = verdict

            def _d(x, plus=True, within=False):
                if x is None:
                    return "-"
                s = f"{x:+.1f}%" if plus else f"{x:.1f}%"
                return s + "†" if within else s

            def _dpp(x):
                return "-" if x is None else f"{x:+.2f} pp"

            md.append(f"| {VARIANT_CN.get(v, v)} | {_d(benefit)} "
                      f"| {'-' if p_adj is None else f'{p_adj:.4f}'} "
                      f"| {_dpp(ruler_benefit_pp)} "
                      f"| {'-' if ruler_p_adj is None else f'{ruler_p_adj:.4f}'} "
                      f"| {_d(pre_delta, within=pre_within)} "
                      f"| {_d(dec_delta, within=dec_within)} | {_d(tr_delta)} "
                      f"| {_d(memtr_delta)} "
                      f"| {_d(par_delta)} "
                      f"| {_d(flo_delta)} "
                      f"| {_d(kv_delta)} "
                      f"| {'-' if mem_v is None else f'{mem_v:.0f} MB'} "
                      f"| {verdict} |")
        md.append("")
        if n_metric_within_noise:
            md.append(f"† {n_metric_within_noise} 个运行时涨幅落在重复测量的"
                      "噪声区间内（[p10,p90] 相对散布的方和根），按不可区分"
                      "处理、不计入增幅；原始差值仍在 summary.json 中。")
        md.append("注：RULER Δ(pp) 为候选相对 standard 的平均 EM 配对差"
                  "（绝对百分点，越大越好；样本集跨种子相同、同种子配对），"
                  "p_adj(RULER) 取自 RULER 指标家族的 Holm 校正配对检验；"
                  "判定在 PPL 与 RULER 任一显著涨点成立时注明依据"
                  "（可接受（依据 …）），显著退步会单独给出权衡/不可行结论，"
                  "不会因另一个指标涨而被静默忽略。"
                  "FLOPs/token 为解析估计（matmul 2·MACs + 注意力 4·kv_len·d/层 "
                  "+ 门控项，见 latency.json 的 efficiency 字段）；KV cache 为公共 "
                  "ctx 下按推理 dtype 计算的 K/V 缓存总量（随 ctx 线性增长，见 "
                  "efficiency.kv_cache_mb_at_ctx），与参数量/激活峰值分开报告；分配量为"
                  "batch=1 prefill 在 CUDA 上的记录，CPU 环境不适用。"
                  "Decode ms/tok 为带 KV 缓存的逐 token 解码时序，"
                  "训练 ms/step 为纯训练步（区间墙钟已扣除评测与 "
                  "checkpoint 写入）跨种子中位数（见 eval_*.json 的 "
                  "train_step_time_ms），训练峰值为该变体训练进程的进程级"
                  "高水位（含模型/优化器/激活/评测窗口，CUDA，跨种子中位数；"
                  "见 eval_*.json 的 train_peak_mem_mb；CPU 训练为 '-'），"
                  "均为相对基线的百分比变化。"
                  "Prefill/Decode 为逐次计时的中位数，重复测量的相对散布见 "
                  "latency.json 的 *_spread_pct；运行时间指标受硬件与同机负载"
                  "影响，跨机器对比无效。")
    else:
        md.append("- ⚠ 缺少运行时/结构基准（latency.json 或 standard 基线），"
                  "无法给出权衡判定；没有增幅对照的涨点不能作为结论。")

    os.makedirs(results_dir, exist_ok=True)
    md_text = "\n".join(md)
    with open(os.path.join(results_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write(md_text)

    # CSV（统计列以并列的 mean/std/n 展开，便于程序消费）
    if rows:
        all_keys = sorted({k for r in rows for k in r})
        with open(os.path.join(results_dir, "summary.csv"), "w", newline="",
                  encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=all_keys)
            w.writeheader()
            w.writerows(rows)
    with open(os.path.join(results_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    # 设计符合性单独落盘：即使 summary.md 被删改，计划/实际人群的偏差仍可
    # 机读；significance.json 内同时保留一份。
    save_json({"plan_digest": design_plan.get("plan_digest"),
               "plan_status": design_plan.get("status"),
               **design_conf},
              os.path.join(results_dir, "design_conformance.json"))
    with open(os.path.join(results_dir, "significance.json"), "w",
              encoding="utf-8") as f:
        json.dump({"pairs": tests,
                   "single_factor_audit": sf_audit,
                   "note": (f"同种子配对 t 检验；结论基于同一指标家族内 "
                            f"Holm–Bonferroni 校正后的 p_adj（alpha={ALPHA}），"
                            f"n<{MIN_SEEDS} 不判定显著性；"
                            "d_z 为配对 Cohen's d 效应量。p_perm/p_perm_adj 为"
                            "符号翻转置换检验（分布无关，n≤16 精确）的原始/家族内"
                            "Holm 校正值，t 与置换结论冲突时见 t_perm_agree 与"
                            "verdict 中的 ⚠ 标注；mean_diff_ci95 为配对差 "
                            "bootstrap 95% 区间；mde 为当前种子数/方差下 80% "
                            "功效的最小可检测效应（不显著时先对照 MDE 再下结论）"),
                   "equivalence": _equivalence_overview(tests),
                   "component_evidence": comp,
                   "training_budget": {
                       "comparable": not _as_text_list(
                           (audit or {}).get("early_stop_runs")),
                       "early_stop_runs": _as_text_list(
                           (audit or {}).get("early_stop_runs"))},
                   "registered_hypotheses": hyp_eval,
                   "registered_hypotheses_source": (
                       "preregistration/hypotheses.json" if hyp_eval else None),
                   "confirmatory_family": {
                       "method": ("Holm–Bonferroni over all tested registered "
                                  "hypotheses (n>=MIN_SEEDS), using raw p; "
                                  "family p_adj retained separately"),
                       "alpha": ALPHA,
                       "n_tested": sum(
                           1 for h in hyp_eval
                           if h.get("p_adj_confirmatory") is not None),
                       "n_global_downgraded": sum(
                           1 for h in hyp_eval
                           if h.get("status_family") in ("支持", "拒绝")
                           and h.get("status") == "未定"
                           and h.get("p_adj_confirmatory") is not None)},
                   "registration_integrity": integrity,
                   "registration_scope": scope_check,
                   "design_plan": design_plan,
                   "design_conformance": design_conf,
                   "observed_design_sensitivity": (
                       observed_design_sensitivity(tests)),
                   "ngram_comparison": ngram_cmp,
                   "linear_probe_comparison": probe_cmp,
                   "ruler_sample_pairs": ruler_sample_tests,
                   "ruler_length_pairs": length_analysis.get("pairs", []),
                   "ruler_length_tests": {
                       "per_context": length_analysis.get("ctx_tests", []),
                       "slope": length_analysis.get("slope_tests", []),
                       "long_minus_short": length_analysis.get(
                           "span_tests", [])},
                   "ruler_length_note": length_analysis.get("note"),
                   "ppl_window_pairs": ppl_window_tests,
                   "ppl_log_pairs": [
                       t for t in tests if t.get("metric") == "PPL(log)"],
                   "factorial_interactions": [
                       t for t in tests if t.get("interaction")],
                   "factorial_shapley": shapley_tests,
                   "factorial_shapley_note": (
                       "全因子 2^3 的 Shapley 效应分解（探索性）：PPL 在 "
                       "−log(PPL) 上分解（φ>0 降低 PPL，exp(−φ)=PPL 几何"
                       "均值比），RULER 用原始 EM；φ_A+φ_B+φ_C 等于 "
                       "Full−none 的总效应（效率性），交互效应被平均分摊到"
                       "参与因子；不替代预注册的单因素确认检验"),
                   "causal_ablation": causal_tests,
                   "causal_ablation_equivalence": _causal_equivalence_overview(
                       causal_tests),
                   "parameter_matched_control": {
                       "variant": PARAM_CONTROL_VARIANT,
                       "pair": list(PARAM_CONTROL_PAIR),
                       "available": bool(param_control_test),
                       "test": param_control_test,
                       "capacity_only_test": capacity_only_test,
                       "note": ("训练期参数匹配安慰剂：lega_placebo 与 lega 的"
                                "门控参数形状/个数/门控 FLOPs 完全相同，但把 "
                                "B/C 证据替换为确定性伪随机特征；lega vs "
                                "lega_placebo 的 PPL 配对差用于排除“多几个门控"
                                "参数的容量/优化红利”。探索性、未预注册；"
                                "未运行该变体时 available=false，不得默认容量"
                                "已排除。")},
                   "inference_factorial_shapley": inference_shapley,
                   "inference_factorial_shapley_note": (
                       "同一 checkpoint 内的 2^3 门控干预（正常基线、"
                       "query_only/conc_only/len_only、no_query/no_conc/no_len、"
                       "no_evidence）上的 Shapley 分解（探索性）：φ>0 表示该"
                       "因子在训练好的前向里降低 PPL；与训练期 Shapley 并列，"
                       "用于区分“重新训练后的组件贡献”与“最终前向实际使用”。"),
                   "inference_factorial_interactions": inference_interactions,
                   "inference_factorial_interactions_note": (
                       "同一 checkpoint 内的 2^3 干预上的 A×B/A×C/B×C 交互"
                       "检验（探索性、独立 Holm 家族）：与训练期 `PPL交互` "
                       "同尺度并列，检查“重新训练后的交互结构能否在最终前向"
                       "复现”；交互=0 表示可加，显著非零不等于某个组件有用。"),
                   "causal_ablation_note": (
                       "推理期因果病灶消融（lega.ablate，val split）：同一 "
                       "checkpoint/同一批窗口上只改门控参数，逐窗口配对 ΔNLL；"
                       "no_query/no_conc/no_len/no_evidence 是训练期去一法的"
                       "推理期对应，query_only/conc_only/len_only 补齐同一 "
                       "checkpoint 的 2^3 单因子格；head_permute 是容量/分布"
                       "不变的安慰剂对照，gate_open/closed/constant_mean 检验"
                       "写回强度与逐位置变化；跨种子 p 经 Holm 校正，窗口级 p "
                       "仅描述性；equivalence 字段给出 ΔNLL 相对 0.01 nat "
                       "SESOI 的 TOST 结果，no_op 不作等价性判定"),
                   "factorial_interactions_note": (
                       "全因子 2^3 的 A×B/A×C/B×C 交互对比（探索性，独立 "
                       "Holm 家族，基准 gate_only）：交互=0 表示组件效应可加；"
                       "显著非零说明同时加入两个组件时收益不能按单加总"),
                   "auroc_diff_note": (
                       f"{AUROC_DIFF_METRIC} 家族为机制主张的正式检验：每个 "
                       "(变体, 种子) run 的 auroc_logit − auroc_qterm 配对差，"
                       "与逐 run 样例聚类 bootstrap 区间"
                       "（analysis_*.json 的 auroc_cluster_bootstrap）互为印证")},
                  f, ensure_ascii=False, indent=2)
    cross_out = {}
    if bad_cross:
        cross_out["ruler"] = bad_cross
    if win_cross:
        cross_out["lm_windows"] = win_cross
    if cross_out:
        save_json(cross_out, os.path.join(results_dir, "badcase_cross.json"))
    print(f"[summary] 汇总 {len(rows)} 个变体 -> {results_dir}/summary.md "
          f"（配对检验 {len(tests)} 组）")
    return {"rows": rows, "markdown": md_text, "tests": tests,
            "ruler_sample_tests": ruler_sample_tests,
            "ruler_length": length_analysis,
            "ppl_window_tests": ppl_window_tests,
            "factorial_shapley": shapley_tests,
            "inference_factorial_shapley": inference_shapley,
            "inference_interactions": inference_interactions,
            "parameter_matched_control": param_control_test,
            "causal_ablation": causal_tests}


def main():
    p = argparse.ArgumentParser(description="一键实验流水线（多种子 + 受控对比 + 完整消融）")
    p.add_argument("--config", default="configs/tiny.json")
    p.add_argument("--data_dir", default="data/dummy")
    p.add_argument("--out_root", default="out")
    p.add_argument("--results_dir", default=None, help="默认 results/<tag>")
    p.add_argument("--variants", default=",".join(ALL_VARIANTS),
                   help="逗号分隔的变体；默认主实验矩阵。可额外点名 "
                        "lega_placebo（参数匹配安慰剂），或直接用 "
                        "--param_control 追加")
    p.add_argument("--param_control", action="store_true",
                   help="追加训练期参数匹配安慰剂 lega_placebo：门控参数/"
                        "FLOPs 与 lega 相同，但 B/C 证据替换为确定性噪声；"
                        "汇总页给出 lega vs lega_placebo 的容量对照。"
                        "默认关闭（不影响主实验规模）")
    p.add_argument("--seeds", default=None,
                   help="逗号分隔的随机种子；默认取配置 seeds（冒烟模式仅取首个）")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--skip_train", action="store_true")
    p.add_argument("--skip_sanity", action="store_true",
                   help="跳过过拟合小批量 sanity check（默认每变体训练前必跑，"
                        "失败即中止全量训练）")
    p.add_argument("--sanity_steps", type=int, default=None,
                   help="sanity check 的优化步数（默认 300，冒烟 60）")
    p.add_argument("--early_stop_patience", type=int, default=None,
                   help="验证损失连续 N 次评测无实质改善则早停（透传 lega.train；"
                        "默认读配置 early_stop_patience，缺省关闭）。启用后各 run "
                        "实际步数可不同，受控变量审计按计划步数核对并在汇总页"
                        "如实标记")
    p.add_argument("--worst_n", type=int, default=10,
                   help="每个 run 转储的最差语言建模窗口数（坏案例归因用；"
                        "默认 10，少于 3 条的模式分析不具统计意义）")
    p.add_argument("--worst_on_test", action="store_true",
                   help="允许在 test split 上转储最差窗口（默认关闭：test 只能"
                        "最终评估一次，把 test 错误清单交给人工迭代会让测试集"
                        "变成隐式训练集；开启后写访问账本并要求报告披露）")
    p.add_argument("--allow_repeat_test", action="store_true",
                   help="允许同一 checkpoint 在 test 上重复评测（默认拒绝，这是"
                        "防止“测试集变隐式训练集”的闸门；--skip_train 重跑会命中，"
                        "确需重测时显式放行，账本仍留痕）")
    p.add_argument("--skip_ruler", action="store_true")
    p.add_argument("--skip_latency", action="store_true")
    p.add_argument("--skip_analysis", action="store_true")
    p.add_argument("--skip_ablation", action="store_true",
                   help="跳过推理期因果病灶消融（默认对 --ablate_variants "
                        "指定的门控变体，在同一 checkpoint/同一批 val 窗口上"
                        "做逐窗口配对干预；不访问 test）")
    p.add_argument("--ablate_variants", default="lega,qgate",
                   help="做推理期病灶消融的变体（逗号分隔；默认 lega,qgate，"
                        "让核心对比的已有门控基线也接受同一套病灶；"
                        "可再加 gate_only 作对照；standard/null 无输出门，"
                        "会被拒绝）")
    p.add_argument("--skip_plots", action="store_true")
    p.add_argument("--tune_baseline", action="store_true",
                   help="训练前对基线（standard）做 lr/权重衰减网格调优（仅验证集）；"
                        "选定组合注入配置并统一用于全部变体，证据写入 "
                        "results/<tag>/baseline_tuning.json")
    p.add_argument("--tune_steps", type=int, default=None,
                   help="调优 trial 的训练步数（默认与正式训练相同；显式缩小步数时"
                        "汇总页会如实记录，不能当作与正式同步数的调优）")
    p.add_argument("--tune_lr_mults", default="0.5,1,2",
                   help="调优网格的 lr 乘子（逗号分隔，默认 0.5,1,2）")
    p.add_argument("--tune_wd_list", default=None,
                   help="调优网格的 weight_decay 列表（逗号分隔；默认仅配置值）")
    p.add_argument("--tune_seeds", default=None,
                   help="调优选择用的种子（逗号分隔；默认只用首个正式种子）。"
                        "单种子选择可能被 val_loss 噪声左右（winner's curse），"
                        "正式实验建议给 ≥3 个（网格配置数 × 种子数 次训练）")
    p.add_argument("--skip_ngram", action="store_true",
                   help="跳过简单统计基线（一元/二元/插值二元平滑 n-gram PPL）")
    p.add_argument("--ngram_split", default="auto",
                   choices=["auto", "val", "test"],
                   help="n-gram 基线评测 split；auto（默认）与 lega.evaluate 的 "
                        "--split auto 同规则：有非空 test.bin 用 held-out test，"
                        "否则回退 val——保证与神经模型 PPL 同 split 可比；"
                        "test 会写访问账本")
    p.add_argument("--allow_data_protocol_violations", action="store_true",
                   help="即使 data_report 检测到打乱/字符切分/文档跨 split 也继续"
                        "训练（默认拒绝；仅用于复现历史实验，不推荐）")
    p.add_argument("--jobs", type=int, default=None,
                   help="sanity/训练/机制分析阶段的并发子进程数；默认 CPU 上 "
                        "min(4, 核数)（<=4 核时为核数+1），CUDA 或 --wandb 下为 1")
    args = p.parse_args()
    if args.allow_data_protocol_violations:
        # 训练/调优/sanity 子进程继承该环境变量；否则 run_all 放行了、
        # lega.train 仍会按默认策略拒绝同一份数据。
        os.environ[ALLOW_ENV] = "1"

    args.config = _repo_path(args.config)
    args.data_dir = _repo_path(args.data_dir)
    py = sys.executable
    tag = os.path.splitext(os.path.basename(args.config))[0] + ("_smoke" if args.smoke else "")
    out_root = os.path.join(_repo_path(args.out_root), tag)
    results_dir = _repo_path(args.results_dir) if args.results_dir else os.path.join(ROOT, "results", tag)
    os.makedirs(results_dir, exist_ok=True)
    variants = parse_variants(args.variants)
    if not variants:
        # 空 --variants 会让流水线“成功”跑完但没有任何 run，下游汇总为空；
        # 这是显式配置错误，不能当作合法实验。
        p.error("--variants 不能为空：至少指定一个变体（可选："
                + ", ".join(RUNNABLE_VARIANTS) + "）")
    if args.param_control and PARAM_CONTROL_VARIANT not in variants:
        variants.append(PARAM_CONTROL_VARIANT)
    invalid = sorted(set(variants) - set(RUNNABLE_VARIANTS))
    if invalid:
        p.error(f"未知变体: {', '.join(invalid)}；可选: "
                f"{', '.join(RUNNABLE_VARIANTS)}")
    ablate_variants = ([] if args.skip_ablation
                       else parse_variants(args.ablate_variants))
    invalid_ablate = sorted(set(ablate_variants) - set(GATE_FACTORS))
    if invalid_ablate:
        p.error("--ablate_variants 含无输出门/未知变体: "
                + ", ".join(invalid_ablate)
                + "；可选: " + ", ".join(sorted(GATE_FACTORS)))
    ablate_variants = [v for v in ablate_variants if v in variants]
    if not args.skip_ablation and not ablate_variants:
        print("[run_all] 警告：--ablate_variants 与本次 --variants 无交集，"
              "跳过推理期病灶消融")

    if not os.path.exists(args.config):
        p.error(f"配置文件不存在: {args.config}")
    with open(args.config, "r", encoding="utf-8") as cf:
        cfg = json.load(cf)
    if not isinstance(cfg, dict):
        p.error(f"配置文件顶层必须是 JSON 对象，收到 {type(cfg).__name__}: "
                f"{args.config}")

    # 多种子协议：显式 --seeds 优先；否则用配置 seeds；冒烟模式只跑首个种子。
    # 种子一律严格解析：int("1.5") 会直接抛错，而 int(1.5)/int(True) 会把
    # 1.5、true 静默截断成 1——结果目录与台账都会用错种子，且两个"不同"
    # 种子经 dict.fromkeys 去重后只剩一个，N 个种子的结论凭空变少。
    if args.seeds is not None:
        seeds = _parse_seed_list(args.seeds, p, "--seeds")
    else:
        # 只有“未提供 seeds”才回退到单个 seed；显式 seeds=[]（或 --seeds ""）
        # 是“没有种子”的配置错误，不能被 `or` 静默替换成默认种子。
        cfg_seeds = cfg.get("seeds")
        if cfg_seeds is None:
            cfg_seeds = [cfg.get("seed", 1337)]
        if not isinstance(cfg_seeds, (list, tuple)):
            p.error(f"配置 seeds 需要整数列表，收到 {cfg_seeds!r}")
        if args.smoke and cfg_seeds:
            seeds = [_strict_seed_int(cfg_seeds[0], p, "配置 seed")]
        else:
            seeds = [_strict_seed_int(s, p, "配置 seeds") for s in cfg_seeds]
    seeds = list(dict.fromkeys(seeds))
    if not seeds:
        p.error("种子列表为空")
    n_seeds = len(seeds)
    print(f"[run_all] 变体={variants}")
    print(f"[run_all] 种子={seeds}（{n_seeds} 个）tag={tag}")
    jobs = _resolve_jobs(args)
    if jobs > 1:
        print(f"[run_all] jobs={jobs}（sanity/训练/评测/机制分析并发；--jobs 可覆盖）")

    # 数据协议前置闸门：报告缺失只告警（兼容旧数据），但一旦报告声明的划分
    # 协议违反“时序/群组不跨侧”，默认直接中止——已知泄露的数据不应进入
    # 全量训练；报告与 meta 的哈希绑定也必须一致，避免陈旧/被改报告冒充
    # 当前自检结果。
    protocol_override = allow_protocol_violations(
        args.allow_data_protocol_violations)
    data_preflight = data_protocol_preflight(
        args.data_dir, allow_override=protocol_override)
    critical_data_alerts = data_preflight["critical_violations"]
    if critical_data_alerts:
        detail = "；".join(
            f"[{a.get('code')}] {a.get('message')}"
            for a in critical_data_alerts[:5])
        print(f"[run_all] ⚠ 数据协议自检发现 {len(critical_data_alerts)} 个"
              f"严重问题，但 --allow_data_protocol_violations 已放行：{detail}")
    elif not data_preflight["report_present"]:
        print("[run_all] 警告：数据目录缺少 data_report.json，"
              "无法做训练前防泄漏闸门（旧数据或未走统一入口）")

    # 子进程随机性/确定性环境：PYTHONHASHSEED 必须在解释器启动前设置；
    # 这里给出基准值（无 --seed 的命令用），带 --seed 的命令由 _cmd_env 逐条
    # 覆盖为自身种子。CUBLAS 工作区配置须在任何 CUDA 初始化前生效。
    os.environ["PYTHONHASHSEED"] = str(seeds[0])
    ensure_cublas_workspace_config()

    # 基线调优（可选）：先调 standard 的 lr/权重衰减（只用验证集，test 不加载），
    # 把最优组合注入配置并统一用于所有变体——保证对比是“调过的基线 vs 新方法”，
    # 而不是把超参数红利误归因给模块。证据落盘 results/<tag>/baseline_tuning.json。
    if args.tune_baseline and args.skip_train:
        print("[tuning] --skip_train：跳过基线调优（没有要训练的新 run）")
    elif args.tune_baseline:
        tune_root = os.path.join(out_root, "_baseline_tuning")
        tune_seeds = (_parse_seed_list(args.tune_seeds, p, "--tune_seeds")
                      if args.tune_seeds else None)
        tune_report = tune_baseline(
            args.config, args.data_dir, tune_root, variant="standard",
            seed=seeds[0], seeds=tune_seeds, steps=args.tune_steps,
            lr_mults=parse_floats(args.tune_lr_mults),
            wd_list=(parse_floats(args.tune_wd_list)
                     if args.tune_wd_list else None),
            device=args.device, tolerance=TUNE_TOLERANCE)
        save_json(tune_report,
                  os.path.join(results_dir, "baseline_tuning.json"))
        best = tune_report["selected"]
        cfg["lr"] = float(best["lr"])
        cfg["weight_decay"] = float(best["weight_decay"])
        tuned_path = os.path.join(results_dir, "config_tuned.json")
        save_json(cfg, tuned_path)
        args.config = tuned_path
        print(f"[tuning] 全部变体改用调优后配置: lr={cfg['lr']:.3g} "
              f"wd={cfg['weight_decay']:.3g} -> {tuned_path}")

    # 简单统计基线：一元/二元/插值二元平滑 n-gram 的 held-out PPL，给出
    # “数据绝对难度”参照（零训练；平滑系数/插值权重只在 val 上选）。split
    # 默认 auto：与神经
    # 模型的评测 split 完全一致（test.bin 存在且非空 → held-out test），
    # 否则汇总页的“神经 vs 统计基线”会变成 test-vs-val 的错位比较。
    if not args.skip_ngram:
        ngram_split = resolve_ngram_split(args.data_dir, args.ngram_split)
        # 窗口规格来自本次配置（含冒烟覆盖）：同 split 还不够，必须同 token——
        # n-gram 额外在神经评测窗口的目标 token 上打分，供汇总做逐位置对比。
        window_spec = _neural_eval_window_spec(cfg, args.smoke)
        try:
            ng = ngram_evaluate(args.data_dir, split=ngram_split,
                                allow_test=(ngram_split == "test"),
                                window_spec=window_spec)
            save_json(ng, os.path.join(results_dir, "ngram_baseline.json"))
            wm = ng.get("ppl_window_matched") or {}
            print(f"[ngram] 简单基线 {ngram_split} PPL: "
                  f"unigram={ng['ppl'].get('unigram')} "
                  f"bigram={ng['ppl'].get('bigram')} "
                  f"interp={ng['ppl'].get('interpolated_bigram')} "
                  f"(α={ng['selected_alpha']}, "
                  f"interp={ng.get('selected_interpolated')}, "
                  f"selected={ng.get('selected_model')})；"
                  f"窗口对齐 PPL: unigram={wm.get('unigram')} "
                  f"bigram={wm.get('bigram')} "
                  f"interp={wm.get('interpolated_bigram')}"
                  f"（spec={ng.get('window_spec')}）")
        except (OSError, ValueError) as e:
            print(f"[run_all] ⚠ n-gram 基线跳过：{e}")

    # 全局溯源清单：本次流水线的代码版本、数据指纹、运行环境与命令。
    git = git_info()
    config_sha256 = sha256_file(args.config)
    # 预注册完整性：把登记表指纹与锁链校验结论写进 repro 与实验台账，
    # 使“本轮实验对应哪份登记表、是否发生静默改写”可事后审计。
    reg_doc = load_registration()
    reg_integrity = verify_registration()
    reg_info = {
        "path": os.path.relpath(HYPOTHESES_PATH, ROOT),
        "present": reg_doc is not None,
        "fingerprint": (registration_fingerprint(reg_doc)
                        if reg_doc is not None else None),
        "integrity": reg_integrity,
    }
    if not reg_integrity.get("confirmatory_valid"):
        print(f"[repro] ⚠ 预注册完整性未通过（status="
              f"{reg_integrity.get('status')}）：本轮全部对比按探索性对待")
    else:
        print(f"[repro] 预注册完整性 {reg_integrity.get('status')}"
              f"（假设 {reg_integrity.get('n_hypotheses')} 条，"
              f"修订链 {reg_integrity.get('chain_length')} 环）")

    # 先验设计计划：在正式训练/评测之前落盘“计划内 (变体, 种子) 单元、
    # 种子数/RULER 每格样本量对 SESOI 的灵敏度、分析人群与排除规则”。
    # 它不读取任何结果或 test，故可作为“开跑前声明”的证据；汇总时据此
    # 核对实际分析人群，计划内单元缺失会使受影响确认性判定降级。
    try:
        design_plan = build_design_plan(
            cfg, seeds, variants, reg_doc,
            config_file=args.config, config_sha256=config_sha256,
            ruler_n_samples=(2 if args.smoke else None),
            smoke=bool(args.smoke), known_variants=set(RUNNABLE_VARIANTS),
            analysis_variants=[v for v in variants
                               if v in GATED_FOR_ANALYSIS])
    except ValueError as e:
        p.error(f"设计计划生成失败（检查配置的 design/ruler 字段）：{e}")
    design_plan_path = os.path.join(results_dir, "design_plan.json")
    save_design_plan(design_plan, design_plan_path)
    print(f"[design] 先验计划 status={design_plan.get('status')} "
          f"digest={str(design_plan.get('plan_digest'))[:12]} -> "
          f"{design_plan_path}")
    _design_hist = load_design_plan_history(results_dir)
    if len(_design_hist) > 1:
        print(f"[design] ⚠ 本结果目录已有 {len(_design_hist)} 份设计计划记录"
              "（重跑/改种子/改样本量）；历史见 design_plan_history.jsonl，"
              "当前计划以最新 digest 为准")
    for _c in design_plan.get("checks", []):
        if isinstance(_c, dict) and _c.get("status") in ("warn", "fail"):
            print(f"[design] {_c['status']}: {_c.get('title')} — "
                  f"{_c.get('detail')}")

    try:
        data_fp = data_fingerprint(args.data_dir)
        data_digest = data_fp["digest"]
    except FileNotFoundError as e:
        # 数据 bin 尚未生成（如 --skip_train 复用历史产物）时不阻塞流水线，
        # 但在计数指纹处显式标注缺失，避免汇总页假装数据版本已知。
        data_fp, data_digest = None, None
        print(f"[repro] 警告：无法计算数据指纹（{e}）")
    # dirty 工作区：commit 之外还需要 diff 指纹与可重放补丁，否则结果无法
    # 仅凭 commit 复原。补丁写入 results_dir 并纳入 completed 台账的产物哈希。
    worktree = worktree_state()
    if worktree.get("dirty") and worktree.get("diff_bytes"):
        patch = write_worktree_patch(
            os.path.join(results_dir, "worktree.patch"))
        if patch.get("patch_file"):
            patch["patch_file"] = os.path.relpath(patch["patch_file"], ROOT)
            if patch.get("patch_sha256") != worktree.get("diff_sha256"):
                patch["differs_from_manifest"] = True
            worktree["patch"] = patch
    repro = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tag": tag,
        "argv": sys.argv,
        "config_file": args.config,
        "config_sha256": config_sha256,
        "seeds": seeds,
        # 逐 seed 的声明式审计：各随机源将使用的 seed 与可独立复算的数据
        # 顺序 RNG 初始指纹。实际生效证据在各 run_manifest.seed_report。
        "seed_audit": {str(s): seed_plan(s) for s in seeds},
        "variants": variants,
        # 推理期病灶消融是否真的在本轮执行、覆盖哪些变体。汇总据此过滤
        # 旧 ablation_*.json：--skip_ablation 时不能把上一轮的病灶证据
        # 静默当成本轮结果（旧 repro 无此字段时保持向后兼容的全量发现）。
        "ablation": {
            "executed": bool(not args.skip_ablation and ablate_variants),
            "variants": list(ablate_variants),
            "split": "val",
        },
        "smoke": bool(args.smoke),
        "git": git,
        "worktree": worktree,
        "data": data_fp,
        "runtime": runtime_info(args.device),
        "registration": reg_info,
        # 先验设计计划摘要 + 各阶段是否实际执行：汇总据此区分“计划内但
        # 被 --skip_* 关闭”与“产物缺失”，不把跳过的阶段当成完整分析人群。
        "design": {
            "plan_path": os.path.relpath(design_plan_path, ROOT),
            "plan_digest": design_plan.get("plan_digest"),
            "status": design_plan.get("status"),
            "checks": [
                {"id": c.get("id"), "status": c.get("status")}
                for c in design_plan.get("checks", [])
                if isinstance(c, dict)],
            "stages": {
                "ruler": bool(not args.skip_ruler),
                "analysis": bool(not args.skip_analysis),
                "latency": bool(not args.skip_latency),
                "ablation": bool(not args.skip_ablation and ablate_variants),
            },
        },
        "data_protocol_preflight": data_preflight,
    }
    save_json(repro, os.path.join(results_dir, "repro.json"))
    print(f"[repro] code={git.get('commit_short')} dirty={git.get('dirty')} "
          f"data={data_digest} -> {os.path.join(results_dir, 'repro.json')}")

    # 实验台账：本轮 started 先落盘（崩溃轮次也留痕），结束时追加同一 run_id
    # 的 completed 记录（含最终指标与产物哈希），跨轮次可追溯，见 lega/experiment.py。
    ledger = ledger_path()
    run_id = new_run_id(tag, config_sha256)
    ledger_common = dict(
        run_id=run_id, tag=tag, config_path=args.config,
        config_sha256=config_sha256,
        # 快照而不是引用：started/completed 两条记录必须对应同一份配置
        config=json.loads(json.dumps(cfg)), variants=variants,
        seeds=seeds,
        # started 就落盘种子计划：即使进程随后崩溃、逐 run manifest 尚未
        # 生成，台账仍能回答“准备用哪些 seed、数据顺序 RNG 应是什么”。
        seed_audit=repro["seed_audit"],
        smoke=bool(args.smoke), argv=sys.argv, git=git,
        worktree=worktree,
        data_fingerprint=data_digest, runtime=repro["runtime"],
        out_root=out_root, results_dir=results_dir,
        time_start=repro["time"], registration=reg_info)
    append_record(ledger, build_experiment_record(status="started",
                                                  **ledger_common))
    print(f"[repro] 实验台账 run_id={run_id} -> {ledger}")

    # 冒烟：少量样例，ctx 取配置 block_size（离线字符分词下 needle+问题
    # 模板约 211 token，128 会让 RULER 全部任务与机制分析被跳过）
    smoke_ctx = (min(256, strict_int(cfg.get("block_size", 256),
                                    "block_size")) if args.smoke else 0)

    # 过拟合小批量 sanity check：全量训练前的代码正确性闸门（梯度/损失/标签）。
    # 每个变体跑一次（用首个种子），失败立即中止——连小批量都无法过拟合时不应
    # 继续全量训练。通过不代表数据增强/对齐/映射等问题不存在。
    sanity = {}
    if args.skip_sanity or args.skip_train:
        print("[run_all] sanity check 已跳过（--skip_sanity/--skip_train）")
    else:
        cmds = []
        for v in variants:
            out_json = os.path.join(results_dir, f"sanity_{v}.json")
            cmd = [py, "-m", "lega.sanity", "--config", args.config,
                   "--data_dir", args.data_dir, "--variant", v,
                   "--seed", str(seeds[0]), "--out", out_json,
                   "--device", args.device]
            if args.sanity_steps is not None:
                cmd += ["--steps", str(args.sanity_steps)]
            if args.smoke:
                cmd += ["--smoke"]
            print("+", " ".join(cmd), flush=True)
            cmds.append(cmd)
        sanity_started = time.time()
        try:
            run_many(cmds, jobs, _jobs_env(jobs))
        except subprocess.CalledProcessError as e:
            v = variants[cmds.index(e.cmd)]
            # 失败实验也要完整留痕：aborted 记录内嵌 sanity 判定与结果文件摘要
            record_aborted_sanity(ledger, ledger_common, v, e.returncode,
                                  results_dir)
            print(f"[repro] 实验台账已记录 aborted（run_id={run_id}，"
                  f"sanity 闸门失败）-> {ledger}")
            raise SystemExit(
                f"sanity check 失败（变体 {v}，退出码 {e.returncode}）：无法过拟合小批量，"
                "说明梯度反传/损失计算/标签对齐存在严重问题；请先排查再开全量"
                f"训练。结果见 {os.path.join(results_dir, f'sanity_{v}.json')}。")
        except BaseException as e:
            # 非“子进程非零退出”的异常（OSError/ValueError/KeyboardInterrupt
            # 等）同样会把已 started 的轮次留在台账里；补 aborted 后再抛。
            append_aborted_record(
                ledger, ledger_common, error="sanity_stage_failed",
                stage="sanity", returncode=getattr(e, "returncode", None))
            raise
        # 不能只信子进程退出码：逐个读回本次 sanity 文件，确认它真的
        # present=true 且 passed=true（旧/空/被手改的文件签不出 PASS），
        # 否则在全量训练前就中止，而不是等 train --sanity_file 再报错。
        sanity_verdict, sanity_failures = check_sanity_results(
            variants, results_dir, not_before=sanity_started,
            expected_seed=seeds[0])
        if sanity_failures:
            first = sanity_failures[0]
            v = first["variant"]
            record_aborted_sanity(ledger, ledger_common, v, 0, results_dir,
                                  reason=first["reason"])
            summary = first.get("summary") or {}
            raise SystemExit(
                f"sanity check 退出码为 0，但结果文件未通过复核："
                f"{first.get('path')}（reason={first['reason']}, "
                f"passed={summary.get('passed')!r}, "
                f"failed={summary.get('failed_checks')!r}）。"
                "不能把没有证据的退出码当作闸门通过。")
        sanity = sanity_verdict
        print(f"[run_all] sanity check 全部通过: {', '.join(sanity.keys())}")

    # 每个 (变体, 种子) 是一个独立 run：训练 → 独立评测 → RULER → 机制分析。
    # 四个阶段由 lega.postrun 在同一进程内顺序执行：单 run 的行为与逐条独立
    # 进程完全相同（各阶段 main() 自行 set_seed/加载模型/写自己的产物，评测/
    # RULER/分析只依赖本 run 的 checkpoint）。evaluate 追加 test 访问账本，
    # append_test_access 已用进程间文件锁保证不丢记录。
    chain = []
    chain_cmds = []
    for v in variants:
        for seed in seeds:
            vdir = _variant_dir(out_root, v, seed, n_seeds)
            suffix = _seed_tag(seed, n_seeds)
            chain.append((v, seed, vdir, suffix))
            cmd = [py, "-m", "lega.postrun",
                   "--config", args.config, "--variant", v, "--seed", str(seed),
                   "--data_dir", args.data_dir, "--out_dir", vdir,
                   "--results_dir", results_dir, "--suffix", suffix,
                   "--device", args.device]
            if args.skip_train:
                cmd += ["--skip_train"]
            if args.wandb:
                cmd += ["--wandb"]
            if sanity:
                # 把开训前过拟合闸门的判定与哈希绑定进 run_manifest；
                # 跳过 sanity 时不透传，避免旧结果文件被当作本次证据。
                cmd += ["--sanity_file",
                        os.path.join(results_dir, f"sanity_{v}.json")]
            if args.early_stop_patience is not None:
                cmd += ["--early_stop_patience", str(args.early_stop_patience)]
            if args.worst_n is not None:
                cmd += ["--worst_n", str(args.worst_n)]
            if args.worst_on_test:
                cmd += ["--worst_on_test"]
            if args.allow_repeat_test:
                cmd += ["--allow_repeat_test"]
            if args.smoke:
                cmd += ["--smoke", "--ruler_ctx_lens", str(smoke_ctx),
                        "--ruler_n_samples", "2",
                        "--analysis_n_examples", "2",
                        "--analysis_ctx", str(smoke_ctx)]
            if args.skip_ruler:
                cmd += ["--skip_ruler"]
            if args.skip_analysis or v not in GATED_FOR_ANALYSIS:
                cmd += ["--skip_analysis"]
            print("+", " ".join(cmd), flush=True)
            chain_cmds.append(cmd)
    if chain_cmds:
        try:
            run_many(chain_cmds, jobs, _jobs_env(jobs))
        except subprocess.CalledProcessError as e:
            # 失败可能出现在链的任一阶段（postrun stderr 已标明阶段名与等价
            # 命令）；训练中止最常见的原因是 NaN 防护触发（train.py 会先落盘
            # nan_report.json 再退出），把排查入口直接指给用户
            v0 = variants[chain_cmds.index(e.cmd) // max(1, len(seeds))]
            append_aborted_record(
                ledger, ledger_common, error="run_chain_failed",
                stage="train/eval/ruler/analysis", variant=v0,
                returncode=e.returncode,
                runs=_snapshot_existing_runs(ledger_common) or None)
            print(f"[repro] 实验台账已记录 aborted（run_id={run_id}，"
                  f"run 链失败）-> {ledger}")
            raise SystemExit(
                f"run 链失败（变体 {v0}，退出码 {e.returncode}）：训练/评测/"
                "机制分析任一阶段失败即中止；若为训练 NaN 中止，对应 run 目录"
                "下已生成 nan_report.json（数据→代码→混合精度→梯度/学习率的"
                "系统排查路径；报告含检测点前的整个 log 区间候选批次与区间内"
                "首个非有限损失步 first_bad_step，数据/激活排查针对失败现场而"
                f"非最后一步批次），排查后重跑；命令: {' '.join(e.cmd)}")
        except BaseException as e:
            # 非“子进程非零退出”的异常（OSError/ValueError/KeyboardInterrupt）
            # 不能让已 started 的轮次永久悬空：补 aborted 终态后再抛。
            append_aborted_record(
                ledger, ledger_common, error="run_chain_failed",
                stage="train/eval/ruler/analysis",
                returncode=getattr(e, "returncode", None),
                runs=_snapshot_existing_runs(ledger_common) or None)
            raise

    # 坏案例报告：把每个变体预测错误的样本逐一归因成 Markdown；
    # 错误样本是改进思路的最重要来源，不能只留聚合分数。
    # 解析结果进 preloaded 缓存：build_summary 汇总时使用同一份 payload。
    preloaded = {}
    if not args.skip_ruler:
        for fp in sorted(glob.glob(os.path.join(results_dir, "ruler_*.json"))):
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                ana = process_ruler_json(fp, payload=payload)
            except (OSError, ValueError, KeyError) as e:
                print(f"[badcase] 跳过 {fp}: {e}")
                continue
            preloaded[fp] = payload
            if ana["n_badcases"]:
                top = ", ".join(f"{BADCASE_CN.get(k, k)}×{n}"
                                for k, n in sorted(ana["category_counts"].items(),
                                                   key=lambda kv: -kv[1])[:3])
                print(f"[badcase] {os.path.basename(fp)}: "
                      f"{ana['n_badcases']} 条错误样本 —— {top}"
                      f" -> badcases_*.md")

    # 语言建模坏案例：把 PPL 最差窗口（next-token 预测错误）逐一归因成
    # Markdown，与 RULER 坏案例同等对待——PPL 平均分看不出模型错在哪句话
    for fp in sorted(glob.glob(os.path.join(results_dir, "eval_*.json"))):
        try:
            with open(fp, "r", encoding="utf-8") as f:
                payload = json.load(f)
            ana = process_eval_json(fp, payload=payload)
        except (OSError, ValueError, KeyError) as e:
            print(f"[badcase] 跳过 {fp}: {e}")
            continue
        preloaded[fp] = payload
        if not ana["n_windows"]:
            continue
        top = ", ".join(f"{WINDOW_CATEGORY_CN.get(k, k)}×{n}"
                        for k, n in sorted(ana["category_counts"].items(),
                                           key=lambda kv: -kv[1])[:3])
        print(f"[badcase] {os.path.basename(fp)}: 最差窗口 {ana['n_windows']} 个"
              f" —— {top} -> worstcases_*.md")

    with abort_ledger_on_failure(ledger, ledger_common, lambda: post_stage):
        # 推理期因果病灶消融：必须在汇总前完成，结果由 build_summary 读取。
        post_stage = "ablation"
        if not args.skip_ablation and ablate_variants:
            ablate_cmds = []
            for v, seed, vdir, suffix in chain:
                if v not in ablate_variants:
                    continue
                ckpt = os.path.join(vdir, "model_best.pt")
                if not os.path.exists(ckpt):
                    ckpt = os.path.join(vdir, "model.pt")
                if not os.path.exists(ckpt):
                    print(f"[ablate] 跳过 {v}/seed{seed}：找不到 checkpoint")
                    continue
                out = os.path.join(results_dir, f"ablation_{v}{suffix}.json")
                cmd = [py, "-m", "lega.ablate", "--ckpt", ckpt,
                       "--data_dir", args.data_dir, "--out", out,
                       "--seed", str(seed), "--device", args.device]
                if args.smoke:
                    # 冒烟时训练配置已被覆盖为小 ctx/少窗口；显式对齐，
                    # 避免病灶分析退回默认 50 窗口拖慢冒烟。
                    cmd += ["--ctx", str(smoke_ctx), "--eval_iters", "2"]
                print("+", " ".join(cmd), flush=True)
                ablate_cmds.append(cmd)
            if ablate_cmds:
                run_many(ablate_cmds, jobs, _jobs_env(jobs))

        post_stage = "latency"
        if not args.skip_latency:
            # 显式给出基准的变体清单：lega.latency 的默认值不含消融变体，
            # 不传会让汇总表的消融变体对应列恒为 "-"。多种子时按各变体首个
            # 种子的 checkpoint 测量（该指标由架构决定，与训练种子无关）。
            lat_variants = ",".join(dict.fromkeys(["standard", "standard_sdpa"] + variants))
            run([py, "-m", "lega.latency", "--config", args.config,
                 "--data_dir", args.data_dir, "--ckpt_dir", out_root,
                 "--out", os.path.join(results_dir, "latency.json"),
                 "--variants", lat_variants, "--seed", str(seeds[0]),
                 "--device", args.device] + (["--smoke"] if args.smoke else []))

        post_stage = "plots"
        if not args.skip_plots:
            run([py, "-m", "lega.plots", "--results_dir", results_dir,
                 "--train_root", out_root])

        # 训练曲线诊断：曲线体检（过拟合/欠拟合/发散/NaN）在汇总前完成，
        # 结果落盘 curve_diagnosis.json 并进入 summary.md
        post_stage = "curve_diagnosis"
        diagnose_training_curves(chain, args.data_dir, results_dir)

        # 数据协议防泄漏报告（prepare_bins 生成）：在所有评测结束后重新读取，连同
        # 实时测试集访问账本复制进 results，供汇总与审计（测试账本已在本轮更新）。
        post_stage = "data_report"
        data_report = load_data_report(args.data_dir)
        if data_report is None:
            print("[repro] 警告：数据目录缺少可解析的 data_report.json"
                  "（旧数据或未走统一入口），汇总将标注防泄漏自检缺失")
        if data_report is not None:
            data_report["test_access"] = summarize_test_access(
                read_test_access_log(args.data_dir))
            save_json(data_report, os.path.join(results_dir, "data_report.json"))
            print(f"[repro] 防泄漏报告 -> {os.path.join(results_dir, 'data_report.json')}")
        else:
            print("[repro] 警告：数据目录缺少 data_report.json（旧数据或未走统一入口），"
                  "汇总将标注防泄漏自检缺失")

        post_stage = "audit"
        audit = verify_controlled_setup(out_root, variants, seeds, n_seeds,
                                        expected_data_digest=data_digest)
        post_stage = "summary"
        build_summary(results_dir, audit=audit, data_report=data_report,
                      preloaded=preloaded)

        # 收尾台账：最终指标（逐 run 的 eval.json）+ 全部结果文件的 SHA-256。
        # 曲线/结果文件后续被覆盖或手改，都能通过哈希与当时记录比对发现。
        # evaluate 阶段已对本 run 的 model_best.pt 做过全量哈希（写在
        # results_dir/eval_<variant><suffix>.json 的 ckpt/ckpt_sha256）；
        # 路径匹配时台账使用该值，否则（未评测/旧评测文件）重算。
        post_stage = "ledger_finalize"
        known_ckpt = {}
        for v, seed, _vdir, suffix in chain:
            try:
                with open(os.path.join(results_dir, f"eval_{v}{suffix}.json"),
                          "r", encoding="utf-8") as f:
                    rec = json.load(f)
            except (OSError, ValueError):
                rec = None
            if isinstance(rec, dict) and rec.get("ckpt_sha256") and rec.get("ckpt"):
                known_ckpt[(v, seed)] = (str(rec["ckpt_sha256"]), str(rec["ckpt"]))
        run_snapshots = []
        for v, seed, vdir, suffix in chain:
            sha, ckpt = known_ckpt.get((v, seed), (None, None))
            run_snapshots.append(snapshot_train_run(
                v, seed, vdir, known_ckpt_sha256=sha, known_ckpt_path=ckpt,
                heldout_path=os.path.join(results_dir, f"eval_{v}{suffix}.json")))
        # 收尾身份核对：长实验期间代码/数据若被改动，开始时记录的 commit/指纹
        # 已不足以代表整轮实验。把结束快照与漂移结论一并写入 completed 记录，
        # 让“全程同一版本”成为可核对的事实而不是假设。
        end_git = git_info()
        end_worktree = worktree_state()
        try:
            end_data_digest = data_fingerprint(args.data_dir)["digest"]
        except FileNotFoundError:
            end_data_digest = None
        drift = provenance_drift(
            {"git": git, "worktree": worktree,
             "data_fingerprint": data_digest},
            {"git": end_git, "worktree": end_worktree,
             "data_fingerprint": end_data_digest})
        tracking_end = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "git": end_git,
            "worktree": end_worktree,
            "data_fingerprint": end_data_digest,
            "drift": drift,
        }
        # 把逐 run 的**实际**种子审计回填到 repro.json：启动时只有 seed_plan
        # 声明，这里补上每个 (变体, 种子) run_manifest.seed_report 的真实 RNG
        # 状态指纹。repro.json 因此即使脱离 out/ 目录也能自证种子覆盖；失败
        # 轮次不会走到这里，仍保留启动期的计划与台账 aborted 记录。
        repro["seed_reports"] = {
            f"{v}/seed{seed}": ((snap.get("provenance") or {}).get("seed_report"))
            for (v, seed, _vdir, _suffix), snap
            in zip(chain, run_snapshots)}
        repro["tracking_end"] = tracking_end
        repro["status"] = ("completed_with_drift" if drift["changed"]
                           else "completed")
        save_json(repro, os.path.join(results_dir, "repro.json"))
        print(f"[repro] 逐 run 种子审计已回填 -> "
              f"{os.path.join(results_dir, 'repro.json')}")
        # repro.json 必须在最终哈希前写入：completed 台账的产物哈希要覆盖
        # 回填后的版本，否则重算会显示“repro 被改写”。
        result_files = [p for p in glob.glob(os.path.join(results_dir, "*"))
                        if os.path.isfile(p)]
        if drift["changed"]:
            print("[repro] ⚠ 实验期间代码或数据身份发生变化："
                  + "; ".join(f"{m['field']} {m['start']} -> {m['end']}"
                              for m in drift["mismatches"])
                  + "；该轮 completed 记录已显式标注，结果不得声称对应单一版本")
        append_record(ledger, build_experiment_record(
            status="completed", time_end=time.strftime("%Y-%m-%d %H:%M:%S"),
            sanity={"executed": bool(sanity), "verdict": sanity},
            runs=run_snapshots,
            artifacts=collect_artifacts(result_files, rel_to=ROOT),
            extra={"tracking_end": tracking_end},
            **ledger_common))
        print(f"[repro] 实验台账已记录 completed（run_id={run_id}）-> {ledger}")
    print(f"\n[run_all] 全部完成；结果目录: {results_dir}")


if __name__ == "__main__":
    main()
