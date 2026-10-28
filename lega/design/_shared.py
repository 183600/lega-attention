from __future__ import annotations
"""实验设计与准备（拆分模块，对外接口与命令行保持不变）。"""
"""实验设计与准备：先验功效/灵敏度、分析人群与设计符合性。

本模块补的是“开跑之前”的科研口径，而不是又一层事后统计：

* **先验灵敏度**：在花掉训练算力之前，回答“计划中的种子数 / RULER 每格
  样本数，能检出多大的效应？”。配对 t 的历史函数仍用正态近似（与
  :func:`lega.utils.min_detectable_effect` 同口径）；但 3~5 个种子时
  正态近似偏乐观，正式规划应同时看 :func:`paired_t_power_nct`
  （非中心 t 的固定种子仿真）。RULER 的配对二值结果用 McNemar 精确
  功效（小样本）或正态近似（大样本）。输出的是
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
    "lega_raw", "lega_raw_no_len", "lega_detach_conc", "null_zero",
)

# 精确 McNemar 功效的规模上限：每增加一个 n 要枚举不一致数 m 与条件拒绝域，
# 500 以内 O(n^2) 仍在毫秒级；更大时改用正态近似（见 mcnemar_power）。
EXACT_MCNEMAR_MAX_N = 500

# 先验灵敏度网格。它们是“展示设计在多大噪声下仍然够用”的横轴，不是从
# 当前数据估计出来的假设值；调用方可通过配置里的 design 块覆盖。
PPL_SD_GRID = (0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03)

# 机制端点（AUROC/AP 的 logit−仅查询配对差）是绝对尺度。它们是跨种子配对
# 差的标准差网格，不是单 run 内有限样例带来的测量误差；后者由机制分析里的
# 样例聚类 bootstrap 负责，两者不能互相替代。
MECHANISM_SD_GRID = (0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.05)

DEFAULT_MECHANISM_SD_REF = 0.01

RULER_DISCORDANCE_GRID = (0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5)

DEFAULT_RULER_DISCORDANCE_REF = 0.10

DEFAULT_PPL_SD_REF = 0.01

_TASKS_FOR_RULER = 5  # 与 lega.ruler.TASKS 一致；用于展示格子数与“最乐观

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
    "run_order": (
        "默认随机化区组：种子为区组，每个区块内变体顺序由跑前写死的 order_seed "
        "确定并记录在 design_plan.json；不得在看到结果后改顺序。显式选择 "
        "variant_major（旧行为）会把变体与时间/硬件漂移混淆，正式实验应避免"),
    "test_sealing": (
        "checkpoint 选择与早停只用 val；test 每次评测写访问账本，重复评测"
        "需要显式放行"),
    "multiplicity": (
        "同一指标家族内 Holm–Bonferroni；预注册确认族再做一次全局 Holm。"
        "先验功效按最坏情形 α/max(家族大小, 全局确认族大小) 规划，"
        "并同时报告未校正口径，避免用名义 α 高估可确认性"),
}


def _install(namespace):
    """把共享命名空间注入拆分模块。"""
    for key, value in globals().items():
        if key != "_install" and not key.startswith("__"):
            namespace.setdefault(key, value)

