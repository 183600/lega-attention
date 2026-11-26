# Code Review — lega-attention (2026-11-26 Agent)

## 审查范围
- 核心注意力机制 (`attention.py`) — 20+ 变体实现
- 模型架构 (`model.py`) — GPT 风格解码器, RoPE, 分块交叉熵
- 评测管线 (`evaluate.py`) — PPL 评测, 窗口相位敏感性, 文档对齐
- 统计方法 — Cohen's d, Hedges' g, bootstrap CI, 多重比较校正

## 审查结果：未发现 bug 或科研方法问题

### 注意力机制
- 全部 20+ 变体（standard/qgate/null/lega 及消融）的数学实现正确
- `probe_gate_factors` 微型前向验证：元数据声明（GATE_FACTORS）与实际前向一致
- `probe_controlled_dimensions` 验证：受控单维度对照确实只改变一个设计维度
- `probe_conc_parameterization` 验证：B 通道的参数化口径（normalized/raw/disabled/custom）与声明一致
- 数值稳定性：clamp_min 防护、log 空间计算、softmax 数值处理均正确

### 模型架构
- 分块交叉熵（`_ChunkedLinearCE`）：闭式梯度与 `F.cross_entropy` 数值等价
- RoPE：rotate_half 实现正确，满足平移等变性
- KV 缓存：动态拼接与静态就地写入两条路径均正确
- 权重共享（tok_emb / lm_head）正确处理

### 评测管线
- 测试集尘封协议：同一 checkpoint 重复评测会显式报警
- 窗口相位敏感性：重叠窗口正确标注为非独立样本
- 文档结构审计：跨文档窗口比例和文档内 PPL 口径正确分离
- 数据协议预检：防泄漏闸门在评测前执行

### 统计方法
- Cohen's d / Hedges' g：pooled_std < 1e-12 防护正确
- Bootstrap CI：percentile + BCa + sign-flip 方法均正确实现
- 多重比较：Holm, BH, Bonferroni 辅助函数完备

## 结论
代码质量优秀，实验设计严谨（2^3 因子消融 + 容量匹配对照 + 参数匹配安慰剂），统计方法完备。未发现需要修复的 bug 或方法论问题。
