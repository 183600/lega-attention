# Code Review Report — lega-attention (Round 4)

**日期：** 2026-11-19
**审查范围：** 全部 95 个 Python 源文件
**审查方法：** 静态分析 + 人工逐文件深度审查

---

## 结论

**未发现 bug 或科研方法问题。**

静态分析工具共标记 40 处可疑模式，经逐一人工审查，全部为误报。

---

## 静态分析标记分类与审查结论

### 1. `eval_call` 模式（13 处）

标记了 `model.eval()` / `m.eval()` / `attn.eval()` 等调用。

**审查结论：全部安全。** 这些是标准的 PyTorch 评估模式切换，均出现在推理/评估上下文中（配合 `torch.no_grad()` 使用），属于正确的深度学习实践。

涉及文件：`ablate.py`、`attention.py`、`evaluate.py`、`latency.py`、`model.py`、`output_ablation.py`、`sanity.py`、`train.py`

### 2. `unprotected_division` 模式（18 处）

标记了除法运算。

**审查结论：全部安全。** 每处除法均有适当的防护措施：

| 文件 | 行号 | 防护措施 |
|------|------|----------|
| `attention.py:930` | `d_model // n_heads` | 上游已校验 `d_model % n_heads == 0` |
| `capacity.py:139` | `info_mean / total_mean` | `total_mean not in (None, 0.0)` 守卫 |
| `config_schema.py:398,403` | `d_model // n_heads` | 整除性已校验 |
| `factorial.py:218,225` | `max_phi / total` | `total > 1e-12` 前置检查 |
| `factorial.py:231` | `f / total` | `total != 0` 守卫 + OverflowError 捕获 |
| `model.py:578` | `d_model // n_heads` | 上游整除性校验 |
| `model.py:790` | `total / n_valid` | `clamp_min(1)` |
| `positional.py:194` | `nlls[i] / total_nll` | `total_nll != 0.0 and isfinite` 守卫 |
| `positional.py:227` | `(total_valid - warm_valid) / total_valid` | `total_valid > 0.0` 守卫 |
| `train.py:961` | `loss_acc / total_w` | `total_w` 非零检查 |
| `uncertainty.py:54` | `value / total` | `total <= 0.0` 守卫 + 异常捕获 |
| `uncertainty.py:188` | `sum / total_w` | `total_w > 0.0` 守卫 |
| `tokens.py:134` | `observed.max() / total` | `total and observed.size` 守卫 |
| `calibration.py:555,556,580,582` | 各种比率 | `total` 非零守卫 |
| `stats.py:1595` | `(s_cand - s_ref) / n_s` | `n_s` 非零守卫 |

### 3. `no_grad_context_missing` 模式（2 处）

标记了 `loss.backward()` 可能缺少梯度上下文。

**审查结论：全部安全。**

- `sanity.py:618`：在 `with torch.enable_grad():` 内部（第 617 行），显式启用梯度计算。
- `train.py:1931`：在标准训练循环中，模型处于 `train()` 模式，`loss.backward()` 是正确的训练模式用法。

### 4. `todo_fixme` 模式（1 处）

- `analysis.py:1330`：标记了 "BUG FIX" 注释。**审查结论：这是一个已修复 bug 的文档注释**，说明之前的 `np.ones_like(ctx_valid)` 已被修正为 `valid_mask.copy()`。修复逻辑正确。

---

## 科研方法审查

### 统计分析方法
- 不确定性量化使用 bootstrap 重采样，方法正确
- 方差分析（ANOVA）实现标准，df 检查和退化情况处理得当
- Shapley 值分解有数值稳定性保护（溢出检测、灾难性抵消检测）

### 模型架构
- 注意力机制实现正确，`d_model % n_heads == 0` 的约束在入口处校验
- RoPE 位置编码的维度要求（偶数 head_dim）有显式校验
- 梯度累积（grad_accum）的 loss 缩放正确

### 实验设计
- 消融实验覆盖全面（`ablate.py`、`output_ablation.py`）
- 数据完整性检查（`datacheck/`）包含结构、文本、token、来源多个维度
- 校准分析（`design/calibration.py`）方法论正确

---

## 总结

lega-attention 是一个成熟的科研代码库，防御性编程模式出色。静态分析标记的所有模式均经过人工验证，确认为正确的代码实践。无需修改。
