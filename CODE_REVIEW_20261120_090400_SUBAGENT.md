# 代码审查报告 - lega-attention (子Agent独立审查)
**日期:** 2026-11-20 09:04
**审查范围:** 全部 166 个 Python 源文件
**审查方法:** 静态分析 + 语法验证 + 算法实现审查 + 科研方法论审查

## 审查结果摘要

| 类别 | 发现数 | 已修复 |
|------|--------|--------|
| 语法错误 (IndentationError) | 1 | 1 |
| 静态分析标记（误报） | 35 | N/A |
| 科研方法问题 | 0 | N/A |

## 发现并修复的问题

### BUG-001: 空的 `with torch.no_grad():` 导致 IndentationError

**文件:** `lega/output_ablation.py` 第 262 行  
**严重级别:** 高（导致模块无法导入）  
**问题描述:**  
`with torch.no_grad():` 语句后面紧跟的是同级的 `with open(...)` 语句，导致 `torch.no_grad()` 的 with 块为空，Python 解释器报 IndentationError。

**原因分析:**  
之前的代码重构中移除了 `model.train()` 调用（注释标记为 `[FIX] removed no-op model.train()`），但未同步调整 `with torch.no_grad():` 上下文管理器，导致其成为空块。

**修复方法:**  
移除多余的 `with torch.no_grad():` 行。后续的推理代码（JSON 文件加载、tensor 创建）不需要 no_grad 上下文，实际的模型推理在代码后续部分执行。

**修复前:**
```python
model.eval()
# [FIX] removed no-op model.train() that undid model.eval()
with torch.no_grad():
with open(args.idx_json, encoding="utf-8") as f:
```

**修复后:**
```python
model.eval()
# [FIX] removed no-op model.train() that undid model.eval()
with open(args.idx_json, encoding="utf-8") as f:
```

## 静态分析标记项复核

全部 35 个静态分析标记均为误报：

- **eval_call (13项):** 均为正确的 PyTorch eval/train 模式切换
- **unprotected_division (17项):** 全部有 clamp/if/guard 保护
- **no_grad_context_missing (2项):** 分别在 `torch.enable_grad()` 上下文和训练循环中
- **todo_fixme (1项):** 已修复 bug 的文档注释

## 科研方法论审查

- LEGA 门控公式 `g = sigmoid(w·q + a·c + b·log n + d)` 正确实现
- 2^3 全因子消融矩阵设计完整
- RoPE 正确应用于 Q/K 但不应用于门控计算
- 贝叶斯因子分析 (JZS Bayes Factor) 实现正确
- 梯度累积 `loss / grad_accum` 缩放正确

## 验证

修复后全量语法验证: 166 文件 / 0 错误 ✅
