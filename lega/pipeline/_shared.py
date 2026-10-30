"""共享导入与常量（由 run_all.py 拆分而来，保持原定义与语义）。"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import argparse
import csv
import glob
import json
import math
import re
import subprocess
import time
from contextlib import contextmanager
from decimal import Decimal, InvalidOperation
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from lega.attention import (CONTROL_VARIANTS, GATE_FACTORS,  # noqa: E402
                            RUNNABLE_VARIANTS, SUPPLEMENTAL_VARIANTS,
                            VARIANTS, VARIANT_CN,
                            _USE_CONC, _USE_RAW_CONC)
from lega.ablate import (INFERENCE_FACTORIAL_CELLS,  # noqa: E402
                         LESION_BY_ID as ABLATION_LESION_BY_ID,
                         LESION_ORDER as ABLATION_LESION_ORDER,
                         MATCHED_LESION_PAIRS, PERMUTATION_LESION_FACTORS)
from lega.factorial import (CELL_ORDER as FACTORIAL_CELL_ORDER,  # noqa: E402
                            decomposition as factorial_decomposition,
                            oriented_value as factorial_oriented_value)
from lega.reparameterization import (  # noqa: E402
    CONTRASTS as REPARAMETERIZATION_CONTRASTS,
    NORMALIZED_CELLS as NORMALIZED_FACTORIAL_CELLS,
    RAW_CELLS as RAW_FACTORIAL_CELLS,
    RAW_FACTORIAL_VARIANTS, RAW_SPECIFIC_VARIANTS,
    analyze as reparameterization_analyze)
from lega.lengtheffect import (context_macro_means,  # noqa: E402
                               log2_slope, span_contrast)
from lega.budget_sensitivity import (  # noqa: E402
    analyze_budget_sensitivity, curves_from_runs, load_train_curve)
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
from lega.replication import replication_forecast  # noqa: E402
from lega.design import (apply_design_gate,  # noqa: E402
                         build_design_plan, build_run_order,
                         design_conformance, load_design_plan,
                         load_design_plan_history, observed_design_sensitivity,
                         render_design_section, save_design_plan,
                         validate_run_order)
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
from lega.testreuse import (DEFAULT_TEST_LOOK_BUDGET,  # noqa: E402
                            apply_test_reuse_gate, repro_test_data_fingerprint,
                            test_dependent_metrics, test_reuse_report)
from lega.tuning import DEFAULT_TOLERANCE as TUNE_TOLERANCE  # noqa: E402
from lega.tuning import parse_floats, tune as tune_baseline  # noqa: E402
from lega.repro import (PIPELINE_RUN_ID_ENV, data_fingerprint,  # noqa: E402
                        environment_lock_report, git_info, provenance_drift,
                        runtime_info, sanity_summary, sha256_file,
                        source_tree_fingerprint, worktree_state,
                        write_worktree_patch)
from lega.ruler import DEPTH_BINS as RULER_DEPTH_BINS  # noqa: E402
from lega.uncertainty import (aggregate_variance_components,  # noqa: E402
                              two_way_variance_components)
from lega.utils import (bootstrap_mean_ci, config_bool,
                        config_finite_float,
                        ensure_cublas_workspace_config, get_device,
                        holm_bonferroni, mcnemar_exact_p, mean_std,
                        min_detectable_effect,
                        paired_sign_flip_test, paired_sign_test,
                        paired_t_test, save_json, save_text,
                        seed_int, seed_plan, strict_int)  # noqa: E402

ALL_VARIANTS = list(VARIANTS)

# 参数匹配安慰剂：不是主实验矩阵的一部分，只有显式 --param_control 或
# --variants 里点名时才训练；它不进入 GATE_FACTORS/全因子 8 格。
PARAM_CONTROL_VARIANT = CONTROL_VARIANTS[0]

PARAM_CONTROL_PAIR = ("lega", PARAM_CONTROL_VARIANT)

# 机制分析不依赖输出门也可计算“集中但看错率/P(hit|集中）”：
# null 变体虽无 g/logit，仍应纳入；standard 无需诊断，保持跳过。
GATED_FOR_ANALYSIS = ("lega", "qgate", "null", "lega_no_len", "lega_no_conc",
                      "lega_no_query", "lega_conc_only", "lega_len_only",
                      "lega_raw", "lega_raw_no_len", "lega_raw_conc_only",
                      "lega_raw_no_query", "lega_detach_conc", "null_zero")

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

# 配对/审计必须共享的 PPL 评测协议：同一 split、同一上下文、同一批复用窗口
# （eval_iters/val_windows/val_tokens_evaluated）才是同一批 token 上的 PPL。
# 仅核对训练 config 不够——独立评测可被 --eval_iters 等参数重跑。
_PPL_PROTOCOL_KEYS = ("split", "eval_ctx", "eval_iters",
                      "eval_iters_requested", "val_windows",
                      "val_tokens_evaluated", "selection_set_is_eval_set")

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

# 主表之外单独做 SESOI 解释的核心指标；交互/Shapley/逐任务等探索性指标
# 若也定义 SESOI 会淹没主线，保留在 significance.json 而不进汇总主段。
# 这里用与 AUROC_DIFF_METRIC/AP_DIFF_METRIC 相同的字面量，避免常量定义
# 顺序造成的导入期 NameError（函数体在调用时才解析该元组）。
_EQ_CORE_METRICS = ("PPL", "RULER平均",
                    "AUROC差(logit−仅查询)", "AP差(logit−仅查询)")

# 因果病灶中“零假设本身有科学含义”的关键对照：分布匹配置换/头置换安慰剂、
# 阳性对照、门恒开/恒关、证据反号、去全部证据。其余病灶仍以“是否有因果
# 贡献”为主，不强行套等价性解读。
_CAUSAL_EQ_LESIONS = ("head_permute", "conc_permute", "query_permute",
                      "len_permute", "gate_open", "gate_closed",
                      "sign_flip", "no_evidence")

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

# 机制主张的检验族：门控完整 logit 与仅查询项的 AUROC / PR-AUC 之差
# （越大越好）。两族必须分别检验：hit 判别通常类别极不平衡，AUROC 会因
# 负样本基数巨大而乐观，PR-AUC 是此时更贴近研究目标的口径。
AUROC_DIFF_METRIC = "AUROC差(logit−仅查询)"

AP_DIFF_METRIC = "AP差(logit−仅查询)"

AUROC_DIFF_REFERENCE = "仅查询项基线"

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

# 全因子 Shapley 分解（探索性）：与交互检验同源，但用全部 8 格把 Full−none
# 的总效应按所有加入顺序平均分摊到 A/B/C，避免单因素边对参照格的依赖。
SHAPLEY_PPL_METRIC = "Shapley(PPL)"

SHAPLEY_RULER_METRIC = "Shapley(RULER平均)"

SHAPLEY_FACTOR_LABEL = {"A": "A 查询项", "B": "B 集中度证据",
                        "C": "C 长度校准"}

# 推理期全因子 Shapley：同一 checkpoint 上干预 2^3 个门控格，而不是重新
# 训练。它回答“训练好的前向里，组件的平均贡献与交互结构是否与训练期
# 全因子消融一致”，不能替代训练期确认性检验。
INFERENCE_SHAPLEY_METRIC = "推理期Shapley(PPL)"

INFERENCE_INTERACTION_METRIC = "推理期交互PPL"

# 因果病灶消融（推理期权重干预）的跨种子指标名与判定口径。
# 原始病灶与活动匹配病灶分成两个 Holm 家族：前者回答“去掉该通道（含平均
# 门控水平变化）会怎样”，后者回答“在相同平均写回强度下，去掉信息内容会
# 怎样”。把二者混在同一家族会既改变问题又改变多重比较惩罚。
CAUSAL_ABLATION_METRIC = "因果消融ΔNLL(lesion−normal)"
MATCHED_ABLATION_METRIC = "因果消融ΔNLL(活动匹配−normal)"
MATCHED_ABLATION_CONTRAST_METRIC = "因果消融ΔNLL(未匹配−活动匹配)"

# 预注册 scope 核对的配置字段：模型/训练规模（结论可迁移性的前提）。
# lr/weight_decay 刻意排除：按本项目协议，基线调优（--tune_baseline）会在
# 验证集上选定统一超参并注入全部变体，属预注册流程的一部分而非范围外改动。
_SCOPE_CONFIG_KEYS = ("d_model", "n_layers", "n_heads", "ffn_mult",
                      "block_size", "steps", "train_ctx")

# 机制判别的简单学习基线对照（补充性/探索性）：门控 logit 是语言建模的
# 副产品、训练时从未见过 hit 标签；线性探针在同一批记录上用 hit 标签监督
# 训练（嵌套、按样例分组并按标签分层的 CV；λ/阈值只在内层 OOF 选，
# 且 λ 按与主检验对齐的 PR-AUC/AUROC，见 lega/probe.py）。
# Δ = 探针 − 门控：>0 表示简单监督基线更好（门控的“判别优越性”不成立），
# ≈0 表示门控与最简线性模型打平，<0 才是门控在判别上显著更好。
LINEAR_PROBE_AP_METRIC = "PR-AUC差(探针−门控)"

LINEAR_PROBE_AUROC_METRIC = "AUROC差(探针−门控)"

# RULER 收益以 EM 绝对百分点计，与 PPL 的相对百分比单位不同：收支比规则
# 不能混算，RULER 单独设阈值（EM 提升通常比 PPL 百分比更难获得，倍数放宽
# 到 10×；下限 0.5 pp 避免收益趋近 0 时比值判定失效）。
RULER_COST_RATIO = 10.0

RULER_RATIO_FLOOR_PP = 0.5


def _install(namespace):
    """把共享命名空间注入模块，避免每个拆分模块重复一长串导入。"""
    for key, value in globals().items():
        if key != "_install" and not key.startswith("__"):
            namespace.setdefault(key, value)

