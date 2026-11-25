# Code Review Report - 2026-11-26

## 审查范围

核心模块：`lega/attention.py`、`lega/model.py`、`lega/train.py`

## 审查结果

### 总体评价：代码质量优秀，未发现关键性 bug 或科研方法问题

本次审查聚焦于：
1. 注意力门控公式的数值稳定性
2. 反向传播梯度路径的正确性
3. 消融实验设计的科学严谨性
4. 数据泄漏与缓存一致性
5. KV 缓存与增量解码的正确性

### 详细发现

#### 1. 门控公式实现 — 正确 ✓

LEGA 门控 `g = σ(w·q + a·c + b·log(n) + d)` 在 `_gate_logit` 中正确实现：
- `c = log(n · max_j p_j)` 使用 `clamp_min(1e-30)` 防止 log(0)
- 各消融变体（lega_no_len, lega_no_conc, lega_no_query 等）通过 `_USE_QUERY/_USE_CONC/_USE_LEN` 元组严格控制
- 初始化 `w=a=b=0, d=2.0` → `g ≈ σ(2) ≈ 0.88`，训练稳定启动
- placebo 对照（确定性伪随机特征）正确隔离了"参数容量"与"真实证据"

#### 2. 反向传播 — 正确 ✓

- `_ChunkedLinearCE` 的闭式梯度（softmax - onehot）正确处理 ignore_index
- RoPE 旋转 `_RoPERotate` 的前向/反向一致（旋转矩阵正交，反向即转置）
- 门控查询项 w·q 使用 RoPE 旋转前的 q_content，避免位置混淆
- 分块路径的 `torch.utils.checkpoint` 正确设置 `use_reentrant=False`

#### 3. 掩码与缓存 — 正确 ✓

- `_MASK_CACHE` 使用 LRU 驱逐，字节上限 `_MASK_CACHE_MAX_BYTES=128MB`
- 因果掩码以 `inference_mode(False)` 创建，避免 inference tensor 被 autograd 拒绝
- `_log_n_table` 设备级前缀表：`table[n] = log(n+1)`，任意 (past_len, T) 切片正确
- KV 缓存静态/动态两条路径互斥使用，无混用风险

#### 4. 消融实验方法学 — 严谨 ✓

- 全因子 2^3 矩阵完整实现（standard/gate_only 作为外部基线，不计入 2^3）
- 去一法（Full/Full-A/Full-B/Full-C）与全因子互相印证
- `GATE_FACTORS` 从实现开关派生，`probe_gate_factors` 运行时实测验证
- `CONTROLLED_DIMENSION_CONTRASTS` 受控单维度对照（集中度统计量、detach、共享粒度、null 可学性）

#### 5. 轻微建议（非 bug）

- `_ce_chunk_size` 中的 `except Exception` 吞掉了 GPU 属性查询异常。虽然这是故意的容错设计（降级为默认分块大小），但建议在 warning 中记录实际异常信息以便调试。
- `save_checkpoint` 的 `os.fsync` 异常静默处理合理（网络文件系统可能不支持），但建议增加可选的 verbose 模式。

## 结论

代码实现与论文方法一致，实验设计科学严谨，未发现需要修复的问题。
