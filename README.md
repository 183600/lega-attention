# LEGA：长度校准的证据门控 Attention

**中文** | [English](README_en.md)


**Length-calibrated Evidence-Gated Attention** —— 研究一个注意力头在当前上下文中**找不到可靠信息**时，能否学会**减少向残差流写入无关内容**。

> 核心创新不是"再加一个门"：已有工作要么在注意力输出后加门控（只看查询向量），要么让注意力头选择"不读取"（显式空注意力）。LEGA 让**同一个门同时参考当前查询、本次注意力分布的集中程度、以及上下文长度**。

---

## 1. 方法

### 1.1 公式

对每个查询位置 $t$、每个头 $h$，标准注意力输出 $o = \sum_j p_j v_j$。LEGA 在写回残差流之前计算：

$$
c = \log\!\bigl(n \cdot \max_j p_j\bigr), \qquad
g = \sigma\bigl(w^\top q + a\,c + b\log n + d\bigr), \qquad
o' = g \cdot o
$$

其中 $p_j$ 是因果注意力权重，$n$ 是该查询可读取的真实 token 数（即位置 $t+1$），$w\in\mathbb{R}^{d_{head}}$ 与标量 $a, b, d$ 均为**逐头**参数。

### 1.2 设计直觉

- **c 的性质**：均匀注意力下 $\max_j p_j = 1/n$，故 $c = \log(n\cdot 1/n) = 0$，**与上下文长度无关**。因此 $c$ 度量的是"超出均匀分布的证据集中度"，把"读得集中"和"读得长"这两个混淆因素解耦了。
- **b·log n 是长度校准项**：即使集中度 $c$ 相同，长上下文中一次"集中读取"的可信度与短上下文不同。$b$ 让每个头自己学出这条长度校准曲线。
- **w·q 保留内容语义**：门不是纯统计规则，查询向量仍然参与判断。
- **初始化**：$w=a=b=0,\ d=2.0$，初始 $g\approx\sigma(2)\approx 0.88$，接近标准注意力，训练稳定启动。

### 1.3 与已有工作的关系

| 方法 | 门控依据 | 机制 | 与 LEGA 的区别 |
|---|---|---|---|
| 输出门控（qgate，已有） | 仅查询 $w^\top q$ | $o'=g\cdot o$ | 不感知"这次读得集中吗/上下文多长" |
| 显式空注意力（null，已有） | 可学习 null 键值槽 | 注意力可整体不读取 | "不读取"与"读了但不写入"是两种不同自由度 |
| **LEGA（ours）** | 查询 + 集中度 + 长度 | $o'=g\cdot o$ | 三因素联合决策"该不该写入" |

### 1.4 变体一览（代码中全部实现）

| 变体 | 门控公式 | 作用 |
|---|---|---|
| `standard` | 无 | 基线 |
| `qgate` | $g=\sigma(w^\top q+d)$ | 已有工作代表：只看查询的输出门控 |
| `null` | 无（额外 null KV 槽） | 已有工作代表：显式"选择不读取" |
| `lega` | $g=\sigma(w^\top q+a\,c+b\log n+d)$ | **本文方法** |
| `lega_no_len` | $g=\sigma(w^\top q+a\log\max_j p_j+d)$ | 消融：去掉长度校准 |
| `lega_no_conc` | $g=\sigma(w^\top q+b\log n+d)$ | 消融：去掉集中度证据 |
| `lega_no_query` | $g=\sigma(a\,c+b\log n+d)$ | 消融：去掉查询项 |

参数量差异：门控仅引入**每头 $d_{head}+3$ 个参数**（LEGA），相对模型总量 <0.02%（`null` 为每头 $2d_{head}$）。脚本会在训练开始时打印精确数值。

---

## 2. 实验设计

- **公平对比**：所有变体使用相同参数量级、相同训练步数/数据（同一 config、同一 seed）。
- **三个评测维度**：
  1. **困惑度**（WikiText-103 验证集 PPL）；评估窗口不重复，并在整个验证集上等间隔抽样（窗口数不足时评估全部）；`gate_mean` 与 PPL 使用同一批窗口、同一精度口径；
  2. **RULER 长上下文检索**（简化复现 5 任务）：`niah_synth`（合成干扰单键检索）、`niah_corpus`（自然文本干扰）、`niah_multikey`（多键干扰）、`fwe`（高频词提取）、`vt`（变量追踪）；多档上下文长度 + 贪心解码；数值任务 EM **只承认生成内容中的第一个数值回答**，避免"先答错后蒙中"被计分；
  3. **实际推理延迟**：生成推理口径的 prefill（只算最后位置 logits）与带 KV 缓存的逐 token decode，对比 SDPA 快速路径。
- **核心风险检验（不能只看"注意力沉降减少"）**：`lega/analysis.py` 用带 gold 跨度的 RULER 样例做机制分析：
  - **misleading_rate**：注意力很集中（$\max_j p_j>\tau$）但落点不在 needle 内的比率 —— "集中 ≠ 找对"的直接证据；
  - **P(hit | 集中)** vs **P(hit | 不集中)**；
  - **AUROC**：门控分数区分"找对/没找对"的能力，且 **LEGA 的完整 logit vs 仅查询项 $w^\top q$ 的 AUROC 之差**，直接回答"结合长度校准后的注意力统计是否比只看查询更能判断该不该写入"。

---

## 3. 快速开始

### 3.1 安装

```bash
pip install -r requirements.txt          # torch / numpy / tiktoken / matplotlib
# 可选: pip install pyarrow wandb
```

### 3.2 冒烟测试（推荐先跑，CPU 几分钟）

```bash
bash scripts/run_smoke.sh
# 等价于: 单元测试 -> 生成伪数据(data/dummy) -> run_all --smoke
```

冒烟模式自动缩减：20 步/变体、ctx=min(256, block_size)、每任务 2 个样例，用于验证代码正确性而非科学结论。

### 3.3 正式实验（WikiText-103 + small 配置）

```bash
# 1) 数据（约 300MB 下载；无网络时可 --local_txt 指向本地文本）
python scripts/prepare_data.py --out_dir data/wikitext103
# 可选 --max_tokens 30000000 控制训练规模

# 2) 全流水线：7 变体训练 → PPL → RULER → 延迟 → 机制分析 → 图表 → 汇总表
python run_all.py --config configs/small.json --data_dir data/wikitext103

# 可选：W&B 日志
wandb login
python run_all.py --config configs/small.json --data_dir data/wikitext103 --wandb
```

结果输出：

| 路径 | 内容 |
|---|---|
| `out/<tag>/<variant>/model.pt` | 检查点 + 训练配置 |
| `out/<tag>/<variant>/train_log.jsonl` | 训练/验证曲线数据 |
| `results/<tag>/summary.md` | **汇总表（可直接贴入周报/论文初稿）** |
| `results/<tag>/eval_*.json` | 各变体验证集 PPL |
| `results/<tag>/ruler_*.json` | RULER 5 任务 × 多档上下文 |
| `results/<tag>/latency.json` | prefill / decode 延迟 |
| `results/<tag>/analysis_*.csv|json` | 逐 (层，头，查询位) 门控记录与汇总 |
| `results/<tag>/fig_*.png` | 全部图表（中文标注） |

### 3.4 单独运行各环节

```bash
python -m lega.train     --config configs/small.json --variant lega --data_dir data/wikitext103
python -m lega.evaluate  --ckpt out/small/lega/model.pt --data_dir data/wikitext103 --out results/small/eval_lega.json
python -m lega.ruler     --ckpt out/small/lega/model.pt --data_dir data/wikitext103 --out results/small/ruler_lega.json
python -m lega.latency   --config configs/small.json --data_dir data/wikitext103 --ckpt_dir out/small --out results/small/latency.json
python -m lega.analysis  --ckpt out/small/lega/model.pt --data_dir data/wikitext103 --out_dir results/small
python -m lega.plots     --results_dir results/small
python -m tests.run_tests   # 单元测试（不依赖 pytest）
```

---

## 4. 项目结构

```
lega-attention/
├── run_all.py                 # 一键流水线 + 汇总表生成
├── configs/
│   ├── tiny.json              # ~6.8M 参数冒烟配置（CPU 可跑）
│   └── small.json             # ~30M 参数正式配置（单卡数小时/变体）
├── lega/
│   ├── attention.py           # 核心：7 个变体的因果注意力（LEGA 门控公式在此）
│   ├── model.py               # GPT 主干 + RoPE + KV 缓存生成 + 参数量工具
│   ├── data.py                # 分词（GPT-2 BPE/字符级兜底）+ bin 缓存 + 批采样
│   ├── train.py               # 训练循环（AdamW、cosine、bf16、W&B 可选）
│   ├── evaluate.py            # 验证集 PPL
│   ├── ruler.py               # 简化 RULER：5 任务，token 级构造，保留 gold 跨度
│   ├── latency.py             # prefill / decode 延迟基准
│   ├── analysis.py            # 机制分析：集中≠找对、门控校准、AUROC
│   ├── plots.py               # 中文标注图表（PNG）
│   └── utils.py               # 种子/JSON/AUROC/Pearson（无 scipy 依赖）
├── scripts/
│   ├── prepare_data.py        # WikiText-103 下载（parquet→zip→rows 多级回退）
│   ├── make_dummy_data.py     # 离线伪语料（冒烟用）
│   └── run_smoke.sh           # 一键冒烟
├── tests/                     # 单元测试（数学性质/KV 缓存/RULER 构造）
└── results/                   # 实验输出（git 忽略，.gitkeep 占位）
```

## 5. 实现细节与注意事项

- **PPL 的展示边界**：JSON 同时保存精确的 `val_loss`；为避免发散模型的 `exp(loss)` 溢出/淹没表格，`val_ppl` 在 `loss>20` 时记为 `exp(20)`，并显式写入 `val_ppl_clamped: true`。出现这种情况时应比较 `val_loss`。
- **c 的数值稳定性**：`max_p = exp(zmax − LSE)`，恒有界 (0,1]；`log` 前 clamp 到 `1e-30`。均匀注意力→`c=0` 有单元测试覆盖（`tests/test_attention.py`）。
- **n 的定义**：查询位置 $t$ 的可读 token 数为 $t+1$（因果），null 槽不计入 $n$；`null` 变体的 null 键不施加 RoPE（位置无关槽）。
- **门控查询项的定位**：门控里的 $q$ 使用 **RoPE 旋转前**的查询内容向量；注意力打分仍使用旋转后的 $q,k$。这样长度/绝对位置只通过显式的 $\log n$ 进入门控，避免方法主张中的因果混淆。
- **可复现性**：`set_seed` 在 CUDA 下同时关闭 cudnn benchmark 并启用确定性算法（`warn_only`）；同 seed 重跑用于变体间公平对比。
- **torch.compile**：配置 `"compile": true` 可在 CUDA 上启用 torch.compile；保存检查点时统一使用剥离包装前缀后的参数键名，`load_model`（evaluate/ruler/latency/analysis）可直接加载。
- **标准注意力双路径**：SDPA 快速路径与显式路径数值一致（有单测），显式路径用于需要注意力统计的变体与延迟公平对比（`standard` vs `standard_sdpa` 两者都测）。
- **KV 缓存**：gated 变体在增量解码时逐 token 计算门控，缓存等价性有单测（全量前向 vs 逐 token）。
- **分词**：默认 GPT-2 BPE（tiktoken，首用需联网缓存）；无网络时 `--tokenizer auto` 退化为字符级（PPL 不可与文献直接对比，但变体间对比有效）。
- **设备**：`small.json` 建议 GPU + bf16；CPU 仅适合 tiny 冒烟。正式实验建议先跑 `lega_no_len` 与 `lega` 的对比（长度校准是核心主张）。
- **已知限制**：30M 模型的 RULER 绝对准确率不高（小模型检索能力有限），关注点是**变体间相对差异**与**机制分析指标**，而非绝对分数；`fwe` 的 top-10 在词频并列时存在微小评估噪声（同 seed 下所有变体一致，不影响对比）。

## 6. License

MIT
