# Code Review Report - lega-attention (2026-11-13)

## 审查结果

| 类别 | 结果 |
|---|---|
| **发现 Bug** | 1 个中等严重度（已修复） |
| **科研方法问题** | 无 |
| **代码质量** | 优秀 (9.5/10) |

## 修复的 Bug

### 修复：`max(tensor, int)` 返回 Python int 导致 `.clamp_min()` 崩溃

**位置：** `lega/model.py:773`

**问题：** 当 `n_valid` 为 `tensor(0)` 时（所有目标均为 ignore_index），`max(n_valid, 1)` 返回 Python `int` 类型 `1` 而非 tensor，后续调用 `.clamp_min(1)` 抛出 `AttributeError: 'int' object has no attribute 'clamp_min'`。

**修复：** `max(n_valid, 1).clamp_min(1)` → `n_valid.clamp_min(1)`

**影响范围：** 仅在训练时某窗口内所有目标均被 ignore 的极端情况触发，正常训练不受影响。

## 预执行扫描误报核实

前次自动扫描报告的 371 个问题经逐条核实，绝大多数为误报：

- **eval() 安全风险（30 项高严重度）：** 全部为 `model.eval()`（PyTorch 模型评估模式）或自定义函数名（如 `_compact_eval`、`_read_run_eval`），无 Python 内置 `eval()` 调用。
- **除零错误：** 均有 `if n == 0`、`clamp_min(1)` 或 `max(n, 1)` 保护。
- **数据泄漏风险：** shuffle 操作仅在训练数据上进行，验证/测试集未受影响；实验设计中的 blocked scheduling 不属于数据分割。

## 代码质量评估

- **数值稳定性：** 分块线性+交叉熵使用 fp32 累加，log(max_p) 使用 clamp_min(1e-30) 保护
- **统计方法：** Holm-Bonferroni 多重检验校正，Bootstrap 置信区间，数据协议预检
- **可复现性：** 完整的种子管理、运行清单、数据指纹、检查点选择日志
- **科研设计：** 全因子 2^3 消融矩阵，容量匹配对照，安慰剂变体
