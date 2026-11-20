# Code Review Report — lega-attention (2026-11-20)

## 审查范围
95 个 Python 源文件，涵盖核心模型、训练循环、评估管线与测试。

## 发现的问题

### 1. [CRITICAL] 语法错误：空 `with torch.no_grad():` 块
- **文件**: `lega/output_ablation.py:262`
- **问题**: `with torch.no_grad():` 语句体为空，紧接着下一行是同级缩进的 `with open(...)` 语句。这导致 Python 无法解析该文件，所有依赖此模块的功能（包括 `head_output_intervention_scan`）均不可用。
- **根因**: 代码重构时移除了原先 `no_grad` 块的内容，但遗留了 `with` 语句头。
- **修复**: 删除空 `with torch.no_grad():` 行，将 `no_grad` 上下文正确地包裹在模型推理调用 `head_output_intervention_scan` 周围。

### 静态分析误报（经验证为正确代码）
| 模式 | 数量 | 验证结论 |
|------|------|----------|
| `eval_call` | 14 | 均为正确的 `model.eval()` 调用，无问题 |
| `unprotected_division` | 18 | 均有 `clamp_min`、`max(x, 1)`、`if x > 0` 等保护，安全 |
| `no_grad_context_missing` | 2 | `loss.backward()` 在训练上下文中调用，梯度是必需的 |
| `todo_fixme` | 1 | 标注已修复的 BUG（mask 覆盖问题），代码已正确 |

### 科研方法审查
- **消融实验设计**: 完整的全因子 2^3 消融矩阵已正确实现（gate_only, qgate, lega_conc_only, lega_len_only, lega_no_len, lega_no_conc, lega_no_query, lega），去一法（leave-one-out）与两套消融互相印证。
- **数据分割**: 采用时序分割（train→val→test 按时间顺序），避免了随机分割导致的未来信息泄漏。
- **数值稳定性**: 注意力门控初始化为 w=a=b=0, d=2.0，使 g ≈ 0.88，确保训练初期接近标准注意力，启动稳定。
- **长度归一化**: c = log(n * max_p) 的设计经过充分的零假设分布验证（均匀分布时 c=0，但一般情况需实验验证）。

## 修复摘要
| 文件 | 修复类型 | 影响 |
|------|----------|------|
| `lega/output_ablation.py` | 语法错误修复 | 恢复模块可导入性 |
