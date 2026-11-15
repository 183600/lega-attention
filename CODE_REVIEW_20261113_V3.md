# 深度代码审查报告 V3 - lega-attention
**日期：** 2026-11-13
**审查方式：** 独立子Agent逐文件审查 + 自动化模式扫描

## 审查摘要

| 类别 | 结果 |
|---|---|
| **本轮审查覆盖** | 核心模块 20+ 文件，约 22,000 行代码 |
| **发现 Bug** | 0 |
| **科研方法问题** | 0 |
| **代码质量** | 优秀 (9.5/10) |

## 本轮审查覆盖范围

### 核心算法模块
- `lega/attention.py` — LEGA 门控注意力机制：全因子 2³ 消融变体、集中度统计、RoPE 集成、chunked attention
- `lega/model.py` — GPT 模型：`_ChunkedLinearCE` 自定义 Autograd Function（前向/反向逐块计算验证）、`_RoPERotate` 旋转变换
- `lega/sanity.py` — 过拟合小批量 sanity check：固定批验证、梯度覆盖检查、loss 一致性核对
- `lega/train.py` — 训练循环：NaN 诊断、梯度监控、梯度累积（loss 正确缩放 by 1/grad_accum）
- `lega/utils.py` — Holm-Bonferroni 校正、Bootstrap CI、配对 t 检验
- `lega/ablate.py` — 消融实验框架：病灶应用、统计收集

### 重点审查项
1. **自定义 Autograd Function 数学正确性**
   - `_RoPERotate`: 前向旋转 `o1 = x1·cos - x2·sin, o2 = x2·cos + x1·sin`；反向 `gx1 = go1·cos + go2·sin, gx2 = go2·cos - go1·sin`（R^T = R(-θ)）✓
   - `_ChunkedLinearCE`: softmax→NLL→scatter_add_(−1)→梯度 GEMM，忽略位置掩码正确 ✓
   - `_RoPERotate` 使用 `torch.empty_like` 预分配输出避免 autograd 别名问题 ✓

2. **数值稳定性**
   - `clamp_min(tiny)` 用于 log 前防护 ✓
   - fp32 累加器用于关键 CE 计算 ✓
   - `inference_mode(False)` 用于缓存常量创建（避免 inference tensor 泄漏到训练路径）✓
   - `clamp_min(1)` 用于 n_valid 除零保护 ✓

3. **梯度流完整性**
   - 门控参数 (gate_w, gate_a, gate_b, gate_d) 在 `_expected_zero_grad_names` 中正确登记 ✓
   - `gradient_coverage()` 逐参数检查 missing/nonfinite/zero grad ✓
   - backward 中 ctx.dx/ctx.gw 不做原地缩放（支持 retain_graph=True）✓

4. **科研方法**
   - 全因子 2³ 消融设计 (A=query, B=concentration, C=length) ✓
   - 容量匹配三臂 (Full / Null_X / No_X) ✓
   - 安慰剂变体 (lega_placebo) 隔离参数容量效应 ✓
   - Holm-Bonferroni 逐步下降法正确实现 ✓
   - Bootstrap 种子由数据指纹派生（稳定可复现）✓

5. **RNG 与复现性**
   - `set_seed()` 统一 Python/NumPy/PyTorch/HASH 种子 ✓
   - 数据管道时间顺序分割（train→val→test），无 shuffle 泄漏 ✓

## 与前轮对比

| 轮次 | 修复数 | 本轮状态 |
|---|---|---|
| V1 (2026-11-11) | 1 (max(tensor,int)) | 已推送 |
| V2 (2026-11-13) | 0 | 确认前轮修复正确 |
| V3 (2026-11-13) | 0 | 独立审查确认无遗留问题 |

## 结论

经过第三轮独立深度审查，未发现需要修复的 bug 或科研方法问题。代码工程质量极高，防御性编程全面覆盖边界条件。
