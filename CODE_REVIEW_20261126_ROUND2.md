# Code Review — lega-attention (2026-11-26 第二轮独立审查)

## 审查范围
- `lega/attention.py` — 20+ 变体的因果多头自注意力实现（含门控、空注意力、消融矩阵）
- `lega/model.py` — GPT 解码器、RoPE、分块交叉熵 (`_ChunkedLinearCE`)、KV 缓存
- `lega/train.py` — 训练循环、NaN 排查、梯度监控
- `lega/data.py` — 数据管线、防泄漏协议、文档结构审计

## 审查方法
独立逐文件审查，重点检查：
1. PyTorch autograd 安全性（原地操作、视图别名、inference tensor 泄漏）
2. 数值稳定性（log(0)、除零、fp16/bf16 上溢/下溢）
3. 门控公式的数学正确性
4. 消融实验设计的严谨性（单因子控制、容量匹配）
5. 训练循环的边界处理（空序列、非有限损失、梯度爆炸）

## 审查结果：未发现 bug 或科研方法问题

### 注意力机制 (`attention.py`)
- 全部变体（standard/qgate/null/lega 及 2^3 消融矩阵）实现数学正确
- 分块注意力路径 (`_attend_chunked`) 的因果掩码、null 槽解析合并、log_n 前缀表均正确
- `_gate_logit` 的原地累加 (`addcmul_`) 在 autograd 下安全：`logit` 是本次前向独占的新张量
- `o.mul_(g.to(o.dtype).unsqueeze(-1))` 原地门控安全：`o` 的全部反向依赖 (p, v) 已在前序计算中保存
- `_mask_cache` / `_LOGN_TABLE` / `_TRI_CACHE` 的 LRU 驱逐逻辑正确，无内存泄漏
- `_placebo_evidence` 的确定性伪随机特征正确使用绝对位置，不消耗全局 RNG
- `probe_gate_factors` / `probe_controlled_dimensions` / `probe_conc_parameterization` 的微型前向验证机制完善

### 模型架构 (`model.py`)
- `_ChunkedLinearCE` 的闭式梯度与 `F.cross_entropy(reduction="none")` 数值等价
- `save_checkpoint` 的原子写（临时文件 + fsync + os.replace）正确防止半截 checkpoint
- RoPE 的 `_RoPERotate` autograd.Function 前向/反向实现正确
- KV 缓存：动态拼接 (`past_kv`) 与静态就地写入 (`kv_cache`) 两条路径均正确
- `generate` 的 EOS 判定：逐行独立标志，批内全行命中后截断到「最后一行首次命中步」

### 训练循环 (`train.py`)
- NaN 排查路径：前向钩子按执行顺序记录非有限激活，定位首个"输入有限、输出非有限"的模块
- `_nonfinite_stats` 递归统计 NaN/Inf，正确处理 tuple/list/dict 混合结构
- 混合精度路径出现非有限时，fp32 复测区分精度问题与数据/代码问题

### 数据管线 (`data.py`)
- bin 缓存协议版本化（v6-v8），缓存身份与内容哈希核对
- 防泄漏：train/val/test 先划分后编码，字符级词表只在 train 上拟合
- 跨文档窗口审计：token 流按文档群组分段编码，边界精确可知

## 结论
代码质量优秀，PyTorch autograd 安全性、数值稳定性和实验设计均无问题。
