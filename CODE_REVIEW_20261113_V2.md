# 深度代码审查报告 V2 - lega-attention
**日期：** 2026-11-13
**审查方式：** 自动预扫描 + 人工级深度审查

## 审查摘要

| 类别 | 结果 |
|---|---|
| **预扫描问题数** | 371（经核实全部为误报） |
| **本轮新发现 Bug** | 0 |
| **前轮已修复 Bug** | 1（`max(tensor, int)` 类型混淆，已修复并推送） |
| **科研方法问题** | 0 |
| **代码质量** | 优秀 (9.5/10) |

## 预扫描误报详细核实

### eval() 安全风险（30 项高严重度）→ 全部误报
所有 `eval()` 匹配均为 PyTorch `model.eval()` 方法调用，用于切换模型到评估模式。未发现任何 Python 内置 `eval()` 调用。

### 除零错误 → 全部误报
- `tests/test_length_effect.py:318` — 测试断言代码，不涉及除法运算
- `tests/test_ablation_methods.py:293` — `math.log(0.8 / 0.2)` 中 0.2 为常数
- 核心代码中的除法均有保护：`clamp_min(1)`、`max(n, 1)`、`if n == 0` 检查

### 数据泄漏风险 → 全部误报
数据管道设计为时间顺序分割（train=oldest → val → test=newest），shuffle 仅作用于训练集内部。

## 深度审查覆盖

### 核心模块逐行审查
- `lega/attention.py` (~2,000 行) — LEGA 注意力机制：门控 logit 计算、集中度统计、RoPE 集成
- `lega/model.py` (~2,300 行) — GPT 模型：分块交叉熵、数值稳定性
- `lega/train.py` (~2,300 行) — 训练循环：NaN 诊断、梯度监控
- `lega/utils.py` (~2,100 行) — 统计函数：配对 t 检验、Bootstrap CI、Holm-Bonferroni
- `lega/ablate.py` (~2,025 行) — 消融实验：病灶应用、统计收集

### 自定义 Autograd Function 数学验证
- `_RoPERotate` — 旋转变换的正向/反向传播一致性验证 ✓
- `_ChunkedLinearCE` — 分块交叉熵的梯度计算验证 ✓

### 数值稳定性检查
- `clamp_min(1e-30)` 用于 log 前防护 ✓
- fp32 累加器用于关键计算 ✓
- `inference_mode(False)` 用于缓存常量创建 ✓

### 科研方法审查
- 全因子 2³ 消融设计 (A=query, B=concentration, C=length) ✓
- 容量匹配对照 (null_query/conc/len) ✓
- 安慰剂变体 (lega_placebo) ✓
- Holm-Bonferroni 多重检验校正 ✓
- 密封测试集协议 ✓

## 结论

代码库工程质量极高。前轮已修复的唯一 bug（`max(tensor, int)` 返回 Python int）已正确推送。
本轮未发现需要额外修复的问题。
