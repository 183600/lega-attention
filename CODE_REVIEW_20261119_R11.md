# Code Review Report — lega-attention (Round 11)

**日期**: 2026-11-19
**审查方法**: 深度逐行审查 + 模式扫描 + 跨仓库对比分析
**审查范围**: 全部 95 个 Python 源文件

## 审查结论

发现并修复了 **8 处真实 bug**，均为同一类型：`model.eval()` 后紧跟 `model.train()` 使评估模式成为空操作。

## 发现的问题

### Bug: `model.eval()` 后立即 `model.train()` 导致 eval 模式失效

**严重程度**: 中等（影响评估/测量的正确性）

**机制**: 代码先调用 `model.eval()` 意图切换到评估模式（关闭 dropout、固定 batch norm 等），但紧接着调用 `model.train()` 将模型切回训练模式，使 eval() 成为空操作。后续的前向传播在训练模式下执行，导致：
- Dropout 在评估/验证期间保持活跃，测量结果带噪声
- BatchNorm 使用批次统计量而非运行统计量
- 延迟测量不准确（训练模式可能有额外开销）

**修复**: 移除紧跟 `model.eval()` 之后的 `model.train()` 调用。

| 文件 | 行号 | 影响 |
|---|---|---|
| `lega/train.py` | 158–159 | `triage_nonfinite()`: 诊断前向在 train 模式下执行，dropout 干扰诊断结果 |
| `lega/train.py` | 832–833 | `validate()`: 验证评估在 train 模式下执行，dropout 导致验证 loss 偏高 |
| `lega/model.py` | 1184–1185 | `load_model()`: 加载的模型被错误地置于 train 模式而非 eval 模式 |
| `lega/ablate.py` | 1272–1273 | 消融评估: gate 统计在 train 模式下收集，dropout 噪声影响统计精度 |
| `lega/ablate.py` | 1445–1446 | 消融评估: 同上，另一评估函数 |
| `lega/latency.py` | 600–601 | 延迟测量: 在 train 模式下测量，结果包含训练模式的额外开销 |
| `lega/output_ablation.py` | 260–261 | 输出消融: 评估在 train 模式下执行 |
| `lega/sanity.py` | 648–650 | finally 块: `model.eval()` + `model.train()` 无意义配对 |

## 静态分析标志复核

| 标志模式 | 数量 | 复核结果 |
|---|---|---|
| `eval_call` | 14 | 全部为 `model.eval()` 标准 PyTorch 用法，无安全问题 |
| `unprotected_division` | 19 | 全部有 `clamp_min`、三元表达式或前置验证保护 |
| `todo_fixme` | 1 | 文档注释，非代码 bug |
| `no_grad_context_missing` | 2 | 均在 `torch.enable_grad()` 上下文中 |

## 科研方法论审查

- 注意力机制（LEGA 门控）: 数学正确，全因子消融矩阵完整
- 统计方法: Bootstrap CI + 置换检验 + p 值校正均正确
- 数据管线: 训练/评估隔离完整，无数据泄漏
- 数值稳定性: 全面的非有限值检测和诊断

## 结论

代码整体质量优秀。本次发现的 8 处 `eval()/train()` 排序 bug 属于编码疏忽，已在本次审查中全部修复。核心科研实现（注意力机制、统计方法、数据管线）无问题。
