# 独立代码审查报告 (2026-11-27)

## 审查范围
- `lega/attention.py` - LEGA 因果自注意力门控机制，全因子消融矩阵
- `lega/model.py` - GPT 模型架构，RoPE，分块交叉熵
- `lega/data.py` - 数据管线，防泄漏，bin 缓存
- `lega/evaluate.py` - 独立 PPL 评测，窗口相位敏感性

## 审查结论

**未发现需要修复的 bug 或科研方法问题。**

### 代码质量亮点

1. **门控机制设计严谨**：2^3 全因子消融矩阵从实现开关派生，`probe_gate_factors` 用微型前向实测验证因子集合，防止元数据与实现漂移
2. **数值稳定性充分**：`_ChunkedLinearCE` 的闭式梯度处理 AMP/非 AMP 路径，ignore_index 掩码正确，空输入安全返回
3. **缓存管理健全**：掩码缓存有 LRU 驱逐和字节上限，log_n 前缀表按需倍增，`inference_mode(False)` 防止缓存被推理态污染
4. **防泄漏措施完善**：数据管线先划分后编码，字符级词表只在 train 上拟合，64-gram 精确扫描覆盖全量

### 已审查的关键逻辑点
- [x] 门控公式：c=log(n*max_j p_j) 的均匀注意力边界、重参数化等价性
- [x] RoPE 旋转：forward/backward 正确，推理态融合路径安全
- [x] 分块交叉熵：闭式梯度 dL/dx = softmax - onehot，ignore 位置零梯度
- [x] 窗口相位偏移：均匀覆盖 [0, ctx-1]，重叠窗口不是独立样本的标注
- [x] checkpoint 选择：记录选型步/步比例，审计 same-budget 可比性
