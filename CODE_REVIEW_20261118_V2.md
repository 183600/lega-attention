# 代码审查报告 (2026-11-18 第二轮)

## 审查概要

| 项目 | 内容 |
|------|------|
| 仓库 | lega-attention |
| 审查日期 | 2026-11-18 (第二轮独立审查) |
| 审查方法 | 多模式深度分析：AST 分析 + 数值稳定性 + 梯度正确性 + 科研方法学 + 注意力机制数学验证 |
| 审查范围 | 全部 95 个 Python 源文件 |

## 总体评价

代码质量**优秀**。经过独立深度审查，确认此前审查结论。仅发现一处防御性编码不一致并已修复。

## 审查维度与结论

### 1. 核心注意力机制数学正确性

逐一验证了 `lega/attention.py` 中所有 22 种变体的门控公式实现：

- **LEGA 门控** `g = sigmoid(w·q + a·c + b·log n + d)`：实现正确。`c = log(n * max_j p_j)` 的计算通过 `_softmax_reuse` → `torch.max` → `log_n` 链路与公式一致。
- **Null 变体**的解析合并 `p_null = sigmoid(a_n - logsumexp(att))`：与"拼接 null 槽后做联合 softmax"在数学上逐项等价，注释清晰。
- **`_softmax_reuse` 的原地/非原地分支**：当 `grad_enabled() and requires_grad` 时使用 `F.softmax`（保留计算图），否则使用 `out=att` 原地操作。`att` 在被原地修改后不再被引用（`del att`），安全。
- **分块注意力** (`_attend_q_chunk`)：对角带掩码 `band_mask` 的平移不变性与整段计算的因果掩码一致，逐块 max_p / argmax / conc_stat 的汇总正确。

### 2. 除零保护完整性

审查了静态分析标记的 23 处 `unprotected_division`：

| 文件 | 行号 | 判定 |
|------|------|------|
| `positional.py:163` | `total_nll / total_valid` | 安全：第 157 行有 `total_valid <= 0.0` 的提前返回 |
| `factorial.py:218-231` | Shapley 占比计算 | 安全：有 `total <= 1e-12` 检查和 `max_phi / total > 1e300` 溢出保护 |
| `uncertainty.py:54` | `value / total` | 安全：有 `total <= 0.0` 和 try/except |
| `missingness.py:155` | `(n*d + m*x) / (n+m)` | 安全：有 `total <= 0` 的提前返回和 `max(total, 1)` 保护 |
| `utils.py:1969` | `(b+1) / n_bins` | **已修复**：与 `lo: b / max(n_bins, 1)` 保持一致 |
| `model.py:790` | `total / n_valid.clamp_min(1)` | 安全：已有 clamp |
| `train.py:959` | `loss_acc / total_w` | 安全：有 `if total_w` 条件 |
| 其余 | 各类 `// n_heads` 等 | 安全：注释标明前置校验 |

### 3. 梯度流正确性

- `sanity.py:617` 的 `loss.backward()`：位于训练态 sanity check 循环中，前后有 `zero_grad` 和 `opt.step()`，正确。
- `train.py:1929` 的 `loss.backward()`：位于梯度累积循环内，循环外有 `opt.zero_grad(set_to_none=True)` (line 1664) 和累积起点的 `zero_grad` (line 1884)，正确。
- `_softmax_reuse` 的 `out=att` 原地操作：仅在 `requires_grad=False` 或 `grad_enabled=False` 时执行，不影响 autograd 图。

### 4. 科研方法学

- **数据泄漏防护**：`data.py` 的 train/val/test 划分使用独立文本源，`datacheck/` 模块提供完整性验证。
- **多重比较校正**：`badcase.py:1675` 使用 Holm-Bonferroni 校正；`heterogeneity.py` 报告 Q 统计量的置信区间。
- **随机种子管理**：使用 `torch.random.fork_rng` 隔离随机状态，确保可重复性。
- **Shapley 归因**：`factorial.py` 实现了完整的 2^3 因子消融矩阵，包含 leave-one-out 验证。

### 5. 安全性

- 无 `eval()` / `exec()` 调用（排除静态分析误报）。
- `torch.load()` 调用均带有 `weights_only=True`。

## 修复清单

| # | 文件 | 行号 | 问题 | 修复 |
|---|------|------|------|------|
| 1 | `lega/utils.py` | 1969 | ECE 等宽分箱中 `lo` 使用 `max(n_bins, 1)` 而 `hi` 使用裸 `n_bins`，虽因第 1944 行的提前返回（`n_bins < 1` 时不执行）而不会触发除零，但风格不一致降低了防御性 | 将 `hi: (b + 1) / n_bins` 改为 `hi: (b + 1) / max(n_bins, 1)` |

## 审查结论

lega-attention 是一个高质量的科研代码库，体现了严谨的实验设计和完善的防御性编程。唯一发现的代码不一致已在本次提交中修复。
