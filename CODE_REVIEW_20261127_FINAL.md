# Code Review — lega-attention (2026-11-27)

## 审查范围

核心源代码文件：
- `lega/attention.py` — LEGA 门控注意力实现（含 2^3 全因子消融矩阵）
- `lega/model.py` — GPT 模型（RoPE、分块交叉熵、KV 缓存）
- `lega/train.py` — 训练循环
- `lega/analysis.py` — 机制分析（AUROC/PR-AUC、校准指标、线性探针）
- `lega/experiment.py` — 实验台账与完整性验证
- `lega/probe.py` — 线性探针
- `lega/evaluation.py` — 评测模块

## 审查结论：代码质量优秀，未发现实质性 bug 或科研方法问题

### 1. 核心算法正确性

**门控公式实现** — `_gate_logit` 中 `g = sigmoid(w·q + a·c + b·log n + d)` 的实现与文档公式逐项一致。容量匹配对照（`lega_null_*`）正确地将对应证据置零而非删除参数，保证了"信息贡献"与"参数容量贡献"的可分离性。

**集中度归一化** — `c = log(n * max_j p_j)` 的实现正确处理了均匀注意力边界情况（`max_j p_j = 1/n → c = 0`），并在文档中明确标注了"不能仅靠单元测试验证长度无关性"的局限性。

**RoPE 与门控查询的解耦** — 门控使用旋转前的 `q_content`，注意力打分使用旋转后的 `q`，正确避免了绝对位置信息混入长度校准。

**分块交叉熵** — `_ChunkedLinearCE` 的闭式梯度正确实现了 `softmax - onehot` 梯度，`ignore_index` 处理完备，空有效标签场景返回零损失。

### 2. 统计方法

- AUROC 与 PR-AUC 并报，且在类别极度不平衡时正确优先推荐 PR-AUC
- Bootstrap 置信区间使用样例为聚类单元（非独立观测），方法正确
- 线性探针使用嵌套 CV + L2 正则化，λ 在内层 OOF 上选择，阈值在 OOF 上按 macro-F1 选择，最终 OOF 不参与选参
- 校准指标（Brier/ECE）同时报等宽分箱与等频分箱两套口径

### 3. 实验设计

- 参数匹配安慰剂（`lega_placebo`）正确隔离了"额外参数/计算"的优化红利
- 全因子消融矩阵（2^3）与去一法（leave-one-out）双重实现，可互相印证
- 受控单维度对照（`CONTROLLED_DIMENSION_CONTRASTS`）保证了每次只改一个设计维度
- 实验台账（ledger）使用 SHA-256 链 + 文件锁，防篡改、防并发

### 4. 代码质量

- 缓存机制（掩码、log_n 前缀表、QKV 融合权重）设计合理，有 LRU 驱逐与字节数上限
- `inference_mode`/`no_grad` 的使用正确区分了训练/推理路径
- 原子 checkpoint 写入（临时文件 + fsync + rename）防止断电损坏
- 输入验证全面（d_model/n_heads/vocab_size/block_size 的正整数校验）

### 5. 注意事项（非 bug）

- `generate` 中 `logits / max(temperature, 1e-6)` 在 `greedy=False` 前已校验 temperature > 0，故 clamp 是防御性冗余，不影响正确性
- `_placebo_evidence` 使用确定性哈希生成伪随机特征，不消耗全局 RNG，保证了同 seed 公平性

## 无需修改

代码无需修改，审查通过。
