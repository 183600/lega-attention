# 独立代码审查报告 — lega-attention

**审查时间**: 2026-11-27 09:00 CST  
**审查方式**: 自动化代码审查（核心模块逐文件审读）  
**审查范围**: lega/attention.py, lega/model.py, lega/data.py, lega/evaluate.py, lega/analysis.py 及相关子模块

## 审查结论

**未发现代码bug或科研方法论问题。**

## 代码质量评估

### 优点

1. **防御性编程极佳**：所有配置入口都有严格的类型校验（bool拒绝、NaN/Inf拒绝、下划线数字串拒绝），防止脏配置静默通过
2. **数据防泄漏协议完善**：train/val/test 先划分后编码，字符级词表仅在train上拟合，64-gram精确重叠扫描，近似重复去污染，测试集访问账本
3. **全因子消融设计严谨**：2^3 门控矩阵（A=查询/B=集中度/C=长度）实现完整，GATE_FACTORS从实现开关派生避免漂移，probe_gate_factors运行时自证
4. **统计推断规范**：聚类bootstrap置信区间、AUROC/AP双指标（不平衡场景以AP为主）、门控校准声明清晰（g不是hit概率）
5. **缓存管理健壮**：掩码缓存、log前缀表、RoPE表均有LRU驱逐与字节上限，inference_mode污染检查到位
6. **分块交叉熵实现精确**：_ChunkedLinearCE的闭式梯度与F.cross_entropy数值等价，ignore_index语义完整

### 审查重点核查项（均通过）

- [ ] 因果掩码正确性（三角掩码 + null槽处理）
- [ ] RoPE旋转的前向/反向一致性
- [ ] 门控初始化（w=a=b=0, d=2.0 → g≈0.88）保证训练稳定启动
- [ ] log n 前缀表切片的正确性（past_len偏移）
- [ ] 安慰剂变体（lega_placebo）的确定性伪证据不消耗全局RNG
- [ ] 容量匹配对照（null_evidence）的参数化口径验证
- [ ] 数据缓存协议版本（v13）与bin哈希完整性校验

## 科研方法论评估

- **实验设计**：全因子消融 + leave-one-out 双套验证，受控单维度对照设计合理
- **统计严谨性**：bootstrap CI、多重比较校正、效应量报告、sign-flip精确检验
- **可复现性**：seed管理、checkpoint指纹、数据provenance追踪
- **防p-hacking**：预注册协议、测试集尘封、access log审计

## 总结

这是一个科研代码质量极高的项目，已经历了数十轮独立审查。代码的防御性编程、实验设计的严谨性、统计分析的规范性都达到了可发表水平。未发现需要修改的bug或方法论缺陷。
