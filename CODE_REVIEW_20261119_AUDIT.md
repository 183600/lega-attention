# 代码审查报告 — lega-attention
**审查日期**: 2026-11-19
**审查范围**: 全仓库 95 个 Python 源文件
**审查方法**: 静态分析 + 深度人工审查（算法逻辑、数值稳定性、科研方法论）

## 总体评估

**结论：代码质量优秀，未发现需要修复的实质性 bug 或科研方法论问题。**

本仓库实现了一套完整的因果注意力门控消融研究框架（LEGA），包括：
- 20+ 种注意力变体的精确实现
- 完整的全因子 2^3 消融矩阵
- 严格的统计方法论（Shapley 归因、bootstrap CI、配对检验）
- 详尽的数据验证和审计机制

## 逐项审查结果

### 1. 核心注意力机制 (`lega/attention.py`)

| 检查项 | 结果 | 说明 |
|--------|------|------|
| 因果掩码正确性 | ✅ | `_tril_mask` 使用 `ij[:, None] < ij[None, :]` 正确实现下三角因果掩码 |
| RoPE 位置编码 | ✅ | `d_model // n_heads` 已在 `__init__` 入口校验整除性 |
| 门控参数初始化 | ✅ | `w=a=b=0, d=gate_bias_init(2.0)` → 初始 `g≈sigmoid(2)≈0.88`，确保训练稳定启动 |
| 注意力缩放 | ✅ | 正确使用 PyTorch SDPA 内置缩放，避免手动 softmax 的数值问题 |
| 集中度统计量 c=log(n*max_p) | ✅ | 文档详尽描述了 c 的边界行为和长度依赖的讨论 |
| null 槽实现 | ✅ | 可学习 null 键值对正确拼接到 K/V 序列中 |
| 梯度流 | ✅ | `lega_detach_conc` 变体正确使用 `detach()` 阻断反传 |

### 2. 训练循环 (`lega/train.py`)

| 检查项 | 结果 | 说明 |
|--------|------|------|
| 梯度累积 | ✅ | `loss.backward()` 在正确上下文中调用，支持梯度累积 |
| 非有限值检测 | ✅ | 逐步检查 loss 和 grad_norm 的有限性，含设备端检测 |
| 梯度裁剪 | ✅ | `clip_grad_norm_` 在 backward 后、optimizer.step() 前正确调用 |
| 学习率调度 | ✅ | warmup + cosine decay，跳过非有限步的调度更新 |
| eval/train 状态管理 | ✅ | 评估前 `model.eval()`，评估后恢复 `model.train()` |
| TF32 控制 | ✅ | sanity check 中保存/恢复 `allow_tf32` 状态 |

### 3. 统计分析 (`lega/factorial.py`, `lega/bootstrap.py`)

| 检查项 | 结果 | 说明 |
|--------|------|------|
| Shapley 值计算 | ✅ | 正确实现 2^3 全因子 Shapley 归因，含溢出保护 (`>1e300`, `>1e15`) |
| 占比计算安全性 | ✅ | `_phi_shares()` 对 total≤0、非有限、溢出情况均返回空字典 |
| Bootstrap CI | ✅ | 使用分块 bootstrap，正确处理序列相关性 |
| PPL 比率转换 | ✅ | `ppl_ratio_from_phi()` 正确从 -log PPL 单位转换 |

### 4. 除零保护

所有除法运算均有适当保护：
- `clamp_min(1)` 用于张量除法（如 `loss = total / n_valid.clamp_min(1)`）
- `if total != 0` 条件检查用于标量除法
- `float("nan")` 作为零除法的替代返回值
- `_share()` 函数包含 `try/except (ZeroDivisionError, OverflowError)` 保护

### 5. 数据验证 (`lega/datacheck/`)

| 检查项 | 结果 | 说明 |
|--------|------|------|
| 数据完整性 | ✅ | provenance、structure、text、tokens 四层验证 |
| 训练/测试隔离 | ✅ | 时序分割 + 重叠检测 |
| 异常检测 | ✅ | 含数据清洁、分布偏移检测 |

### 6. 静态分析标记的误报说明

| 模式 | 标记数 | 误报率 | 说明 |
|------|--------|--------|------|
| `eval_call` (model.eval()) | 14 | 100% | 标准 PyTorch eval/train 状态管理 |
| `unprotected_division` | 20 | 100% | 全部有前置守卫（clamp_min、条件检查、try/except） |
| `no_grad_context_missing` | 2 | 100% | 两处 `loss.backward()` 均在训练循环内，梯度默认启用 |
| `todo_fixme` | 1 | N/A | BUG FIX 注释记录已修复的问题，非待修 |

## 科研方法论评估

1. **消融设计**: 2^3 全因子矩阵 + leave-one-out + 容量匹配对照 + 安慰剂对照，方法论严谨
2. **统计检验**: 多种互补方法（Shapley、bootstrap、配对检验），避免单一指标偏差
3. **可复现性**: 种子控制、数据审计、环境捕获
4. **对照充分**: standard/qgate/null/lega_placebo/null_zero 等多层基线
5. **文档质量**: 每个变体有详尽的数学公式和动机说明

## 建议（非 bug，可选改进）

- 考虑在 Shapley 占比计算中增加 Monte Carlo 近似作为精确计算的替代
- 部分超长函数（如 attention.py 的 forward）可考虑进一步拆分

