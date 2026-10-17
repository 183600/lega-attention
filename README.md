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
| `lega` | $g=\sigma(w^\top q+a\,c+b\log n+d)$ | **本文方法**（全因子 A+B+C） |
| `lega_no_len` | $g=\sigma(w^\top q+a\log\max_j p_j+d)$ | 去一法：去掉长度校准（A+B） |
| `lega_no_conc` | $g=\sigma(w^\top q+b\log n+d)$ | 去一法：去掉集中度证据（A+C） |
| `lega_no_query` | $g=\sigma(a\,c+b\log n+d)$ | 去一法：去掉查询项（B+C） |
| `lega_conc_only` | $g=\sigma(a\log\max_j p_j+d)$ | 全因子：仅集中度项（B） |
| `lega_len_only` | $g=\sigma(b\log n+d)$ | 全因子：仅长度项（C） |

**消融完整性**：设查询项=A、集中度项=B、长度项=C，则
全因子 8 格（无/`standard`、A/`qgate`、B/`lega_conc_only`、C/`lega_len_only`、
AB/`lega_no_len`、AC/`lega_no_conc`、BC/`lega_no_query`、ABC/`lega`）与
去一法（Full/Full−A/Full−B/Full−C）全部实现；汇总表会自动列出矩阵并标注缺失组合。
组件是否有用不靠“组合跑全”自证：汇总表基于 Holm 校正后的显著配对检验逐一判定
A/B/C（去一法：Full−X 显著变差；全因子：仅 X 显著优于无门控），
组合齐全但证据不足或方向相反时，明确拒绝“各组件均有用”的结论。

参数量差异：门控仅引入**每头 $d_{head}+3$ 个参数**（LEGA），相对模型总量 <0.02%（`null` 为每头 $2d_{head}$）。脚本会在训练开始时打印精确数值。

---

## 2. 实验设计

- **数据协议（防泄漏，全部可审计）**：
  1. **先划分后预处理**：先产出 train/val/test 文本再编码；字符级词表（特征工程）只在 train 上拟合，val/test 只做 transform，未见字符统一映射 `<unk>`（GPT-2 BPE 是公开预训练词表，不依赖本项目数据）；
  2. **时序划分**：`--local_txt` 按时间先后切分——最早 80% 训练、其后 10% 验证、最新 10% 测试；切点对齐文档边界（标题行 > 空行段落 > 换行），划分时绝不随机打乱（训练阶段对训练样本的 batch 级随机采样不构成泄露）；WikiText-103 官方 split 按文档划分，天然 held-out；
  3. **群组不跨侧**：以文档/段落为整体单位，同一文档只落在一个 split；与 val/test 精确重复的训练长行自动去污染（val/test 保持尘封不改写），并用采样 64-gram 哈希筛查跨 split 的逐 token 复制；
  4. **清洗与异常检查**：坏字符（U+FFFD/NUL/非法控制符）确定性清除；token 流做整数/有限/id 越界硬校验；重复样本、超长行、train 与 val/test 的一元分布漂移（TV/JS、OOV）全部写入 `data_report.json`；
  5. **测试集尘封**：checkpoint 选择与早停只用 val，test 只在最终评估时访问；每次 test 评测写入 `data_dir/.test_access_log.jsonl` 账本，汇总表报告累计次数与被重复评测的 checkpoint。
- **公平对比（受控变量）**：所有变体 × 种子使用同一 config 训练（同一优化器/学习率策略/训练步数/数据顺序），训练后双层自动核对：`config_used.json`（超参数逐项一致）与 `eval.json` 溯源（数据指纹、代码 commit、配置 SHA-256、实际训练步数与 tokens_seen 一致，并与本次流水线数据比对），任何差异都会直接报错；模型初始化完成后统一重放随机种子，使各变体的训练期随机流（dropout 等）也完全一致——把“公平对比”从口头约定变成可执行断言。
- **统计可靠性（多种子）**：关键指标跨随机种子报告 **Mean ± Std** 与配对 Cohen's d 效应量（`configs/small.json` 默认 5 个种子，`--seeds` 可覆盖）；变体间同种子配对的双侧 t 检验在同一指标家族内做 **Holm–Bonferroni 校正**（控制多重比较 FWER），结论一律基于校正后 p 值（`results/<tag>/significance.json`）。n<3 时只报描述统计、不判定显著性，避免“提升 0.2%、方差 0.5%”被当成有效增益。
- **预注册与防事后改写（confirmatory vs exploratory）**：正式实验前把方向性主张登记在 `docs/registered_hypotheses.json`，汇总时逐条对照 Holm 校正后的同种子配对检验，判定 支持 / 拒绝（方向相反且显著=被证伪）/ 未定（未定≠无效应，须对照 MDE 解读），结果进 `summary.md` 与 `significance.json`；未登记的对比一律属事后（exploratory），只能作为线索。登记表由 `docs/registered_hypotheses.lock.json` 防篡改：锁基线记录整文档 SHA-256 与**逐假设科学内容摘要**，修订必须经 `python -m lega.hypotheses --amend --reason ...` 在**修订链**上留痕（既有假设只增不改不删）。删除登记表、改写 claim、调换变体对、删条目或未留痕新增，都会被 `--verify`／汇总页判为 `violated`，并**自动把全部确认性判定降级为探索性**；登记表的意外删除与恢复事件本身也必须在 amendments 中留痕（本项目已发生过一次清理误删，见 `docs/design_notes.md`）。
- **三个评测维度**：
  1. **困惑度**（WikiText-103 PPL：checkpoint 选择在 val 上完成，PPL 在 held-out test 上评测，无 test 时回退 val）；评估窗口不重复，并在整个评测集上等间隔抽样（窗口数不足时评估全部）；`gate_mean` 与 PPL 使用同一批窗口、同一精度口径；
  2. **RULER 长上下文检索**（简化复现 5 任务）：`niah_synth`（合成干扰单键检索）、`niah_corpus`（自然文本干扰）、`niah_multikey`（多键干扰）、`fwe`（高频词提取）、`vt`（变量追踪）；多档上下文长度 + 贪心解码；数值任务 EM **只承认生成内容中的第一个数值回答**，避免"先答错后蒙中"被计分；
  3. **实际推理延迟**：生成推理口径的 prefill（只算最后位置 logits）与带 KV 缓存的逐 token decode，对比 SDPA 快速路径。
- **核心风险检验（不能只看"注意力沉降减少"）**：`lega/analysis.py` 用带 gold 跨度的 RULER 样例做机制分析：
  - **misleading_rate**：注意力很集中（$\max_j p_j>\tau$）但落点不在答案数值跨度（value_span）内的比率 —— hit 判定以答案数值的精确 token 跨度为准（而非整条 needle 句），"集中 ≠ 找对"的直接证据；
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

冒烟模式自动缩减：20 步/变体、仅 1 个种子、ctx=min(256, block_size)、每任务 2 个样例，并自动执行过拟合小批量 sanity check（60 步），用于验证代码正确性而非科学结论。

### 3.3 正式实验（WikiText-103 + small 配置）

```bash
# 1) 数据（约 300MB 下载；无网络时可 --local_txt 指向本地文本）
python scripts/prepare_data.py --out_dir data/wikitext103
# 可选 --max_tokens 30000000 控制训练规模

# 2) 全流水线：9 变体 × 多种子（默认 5 个）训练 → PPL → RULER → 延迟
#    → 机制分析 → 图表 → 汇总表（Mean ± Std + 配对显著性检验）
python run_all.py --config configs/small.json --data_dir data/wikitext103

# 可选：自定义种子（如 3 个）或 W&B 日志
python run_all.py --config configs/small.json --data_dir data/wikitext103 --seeds 1337,2024,42
wandb login
python run_all.py --config configs/small.json --data_dir data/wikitext103 --wandb
```

结果输出：

| 路径 | 内容 |
|---|---|
| `out/<tag>/<variant>[/seed<seed>]/model.pt` | 检查点 + 训练配置 + 代码/数据溯源字段（多种子时按种子分子目录） |
| `out/<tag>/<variant>[/seed<seed>]/train_log.jsonl` | 训练/验证曲线数据 |
| `out/<tag>/<variant>[/seed<seed>]/run_manifest.json` | **溯源清单：git commit/dirty、配置 SHA-256、数据指纹、运行环境、命令行** |
| `results/<tag>/summary.md` | **汇总表（Mean ± Std + 消融矩阵 + 受控变量自检 + 可复现性/sanity）** |
| `results/<tag>/repro.json` | 流水线全局溯源（代码版本、数据指纹、种子、运行环境） |
| `results/experiments.jsonl` | **追加式实验台账：每轮 run 的 run_id、代码版本、配置 SHA-256、种子、数据指纹、sanity 结论、逐 run 最终指标与产物哈希（started/completed 生命周期）** |
| `data/<data_dir>/data_report.json` | 数据防泄漏自检（划分协议、词表拟合集、去污染、64-gram 重叠、分布漂移、超长行、访问账本） |
| `results/<tag>/sanity_<variant>.json` | 过拟合小批量 sanity check 结果与阈值 |
| `results/<tag>/significance.json` | 同种子配对 t 检验（Δ均值 ± Δ标准差、配对 Cohen's d_z、原始 p、Holm 校正 p_adj、组件有效性证据）；含 `registration_integrity`（预注册锁链校验）与 `registered_hypotheses`（逐条判定） |
| `results/<tag>/curve_diagnosis.json` | 逐 run 训练曲线诊断（过拟合（含训练停滞+验证回升的平台期）/过拟合趋势（训练降而验证停滞）/欠拟合/发散/NaN + 梯度范数尖峰/非有限 + 处理建议） |
| `results/<tag>/badcases_*.md` | RULER 错误样本归因（标注错误/误抄干扰值/幻觉/无数值回答/fwe） |
| `results/<tag>/worstcases_*.md` | 语言建模最差窗口归因（复读吸引子/高置信错答/低置信；gold 是否在上下文中出现） |
| `results/<tag>/badcase_cross.json` | 跨变体错误对照（全体共识失败 vs 相对 standard 的特异性新增/修复） |
| `results/<tag>/eval_*.json` | 各变体 held-out test PPL（无 test.bin 时回退 val），含 worst_windows 坏案例与训练吞吐（`train_step_time_ms`/`train_tokens_per_sec`，供效率-精度权衡） |
| `results/<tag>/ruler_*.json` | RULER 5 任务 × 多档上下文，含 badcases 错误样本 |
| `results/<tag>/latency.json` | prefill / decode 延迟（逐次计时的中位数 + `*_spread_pct=(p90-p10)/median` 测量散布，供判断变体差异是否超出噪声） |
| `results/<tag>/analysis_*.csv|json` | 逐 (层，头，查询位) 门控记录与汇总 |
| `results/<tag>/fig_*.png` | 全部图表（中文标注，含训练曲线 `fig_train_curves.png`） |

### 3.4 可复现性、sanity check 与实验追踪

- **随机种子全覆盖**：`lega.utils.set_seed` 固定 Python `random`、NumPy、PyTorch（CPU/CUDA）、cuDNN 确定性算法（`warn_only`）与 `PYTHONHASHSEED`（子进程）；本项目的批采样使用显式 `np.random.default_rng(seed)`（无 torch DataLoader/worker），数据顺序由 seed 唯一确定，单测覆盖（`tests/test_repro.py`）。`run_all` 还会逐命令把子进程的 `PYTHONHASHSEED` 对齐到该命令自己的 `--seed`（而非统一用第一个种子），每个 run 的 `run_manifest.json` 内嵌 `seed_report` 审计记录：逐随机源的实际种子、**各 RNG 真实状态的短指纹**（同 seed 可被第三方用 `set_seed(seed)+rng_state_fingerprint()` 复算验证，异 seed 必然不同，当前状态不属于该 seed 时为 `null` 而非张冠李戴）、cuDNN/确定性开关、数据顺序机制，并如实区分 `PYTHONHASHSEED` 的“解释器启动时取值 vs 仅对子进程生效的当前 env”（启动时未设置时 `pythonhashseed_effective=false`，不再让字段被 `set_seed` 自身写成 seed 而自证为真），杜绝“声称固定了其实没有”。跨硬件架构（不同 GPU/CPU/cuDNN 版本）无法保证逐位一致，但同一台机器上重跑结果稳定。
- **过拟合小批量 sanity check（训练前闸门）**：`lega.sanity` 抽取固定小批量（默认 8 条 × 128 token），关闭 dropout/weight decay 与全部数据扰动，在**同一批数据**上优化，检查损失能否迅速趋近 0、下一 token 准确率能否接近 100%（阈值可用 `--loss_threshold/--acc_threshold` 或配置 `sanity` 块调整）。`run_all.py` 在全量训练前对每个变体自动执行，失败立即中止（`--skip_sanity` 跳过，`--sanity_steps` 调节步数）。这是**必要非充分**检查：连小批量都无法过拟合说明梯度反传/损失/标签存在严重 bug；通过也不代表数据增强、特征对齐、标签映射等 bug 不存在，sanity JSON 同时记录首步 loss（应≈log 词表）、首步梯度范数、**首步逐参数梯度覆盖**（`missing/zero/nonfinite`，可定位“总梯度范数正常但某个子模块没接进计算图”的断流 bug）与完整损失曲线，供中间结果复核。
- **实验追踪/可追溯**：每次训练自动写 `run_manifest.json`（git commit/dirty、配置 SHA-256、数据指纹、种子审计、Python/PyTorch/CUDA 环境、完整命令行），超参数在 `config_used.json`，训练曲线在 `train_log.jsonl`（并汇总为 `fig_train_curves.png`）；`run_all` 写全局 `results/<tag>/repro.json`，并在固定路径 `results/experiments.jsonl` **追加**一轮实验台账记录（开始 `started` + 结束 `completed`，同一 `run_id`）：代码版本、配置哈希、种子、数据指纹、sanity 结论、逐 run 最终指标与 `train_log.jsonl`/被评测 checkpoint/全部结果文件的 SHA-256（`eval_*.json` 内亦记录 `ckpt_sha256`，结果与产出的权重一一绑定）。重跑不再丢失上一轮历史；只有 `started` 没有 `completed` 的记录即该轮被中止/崩溃，失败实验同样留痕。评测结果（eval/ruler/analysis/latency）均内嵌同一份代码/数据溯源字段，不依赖手工记录。

```bash
# 单独运行 sanity check（退出码 0=通过，1=未通过）
python -m lega.sanity --config configs/tiny.json --data_dir data/dummy --variant lega \
    --out results/tiny_sanity/lega.json
```

### 3.5 单独运行各环节

```bash
python -m lega.train     --config configs/small.json --variant lega --data_dir data/wikitext103
python -m lega.evaluate  --ckpt out/small/lega/model.pt --data_dir data/wikitext103 --out results/small/eval_lega.json
python -m lega.ruler     --ckpt out/small/lega/model.pt --data_dir data/wikitext103 --out results/small/ruler_lega.json
python -m lega.latency   --config configs/small.json --data_dir data/wikitext103 --ckpt_dir out/small --out results/small/latency.json
python -m lega.analysis  --ckpt out/small/lega/model.pt --data_dir data/wikitext103 --out_dir results/small
python -m lega.plots     --results_dir results/small --train_root out/small
python -m lega.sanity    --config configs/small.json --data_dir data/wikitext103 --variant lega --out results/sanity_lega.json
python -m lega.curves    --run_dir out/small/lega/seed1337   # 训练曲线诊断（过拟合/欠拟合/发散/NaN）
python -m lega.badcase   --results_dir results/small         # 错误样本归因（RULER + 语言建模最差窗口）
python -m lega.hypotheses --verify                           # 预注册完整性校验（0=可确认；1=被删/改/未留痕）
python -m tests.run_tests   # 单元测试（不依赖 pytest）
```

结果分析口径（详见 `summary.md` 对应小节）：训练诊断看曲线而非只看最终值，
除过拟合/欠拟合/发散外还独立报告 **pre-clip 梯度范数尖峰/非有限**（NaN 与发散的
早期信号，损失曲线可能尚未反映）；训练损失停滞而验证回升同样判为过拟合
（平台期泛化先恶化，不因"训练没降"误判成欠拟合），训练仍在下降而验证已停滞
则单独判 **过拟合趋势**（按正则化/早停处理，不据此续训）；NaN 按数据→代码
（log(0)/除零/softmax 溢出，前向钩子给出首个"输入有限、输出非有限"的模块即
`origin`）→混合精度下溢→梯度爆炸/学习率的顺序排查（`nan_report.json`），
数据与激活检查均优先分析失败步批次（预取会覆盖待训练批次），val 损失非有限时
直接改用真实验证窗口（训练批次仅作取窗失败的回退），混合精度路径出现非有限时
还会对同一批次做 **fp32 隔离复测**（fp32 有限→判混合精度数值问题；fp32 同样
非有限→排除精度因素）；错误样本逐条归因，并用
跨变体对照区分"全体共识失败（数据/难度）"与"方法特异性盲区"；效率-精度权衡
显式判定，增幅取 **Prefill/Decode 推理延迟、训练 ms/step、参数量、FLOPs、
显存峰值** 相对基线的最大者，其中延迟为逐次计时的中位数，**差异不超过两端
测量散布（[p10,p90] 方和根）时按不可区分处理、不计入增幅（表中标 †）**，
涨点未过显著性检验或增幅远大于收益时明确标注
（见 §2 的 `tradeoff_verdict` 规则），"涨 0.5% 但训练/推理贵 10 倍"会被直接拒绝。

---

## 4. 项目结构

```
lega-attention/
├── run_all.py                 # 一键流水线 + 汇总表生成
├── configs/
│   ├── tiny.json              # ~6.8M 参数冒烟配置（CPU 可跑）
│   └── small.json             # ~30M 参数正式配置（单卡数小时/变体）
├── lega/
│   ├── attention.py           # 核心：9 个变体的因果注意力（LEGA 门控公式与消融矩阵在此）
│   ├── model.py               # GPT 主干 + RoPE + KV 缓存生成 + 参数量工具
│   ├── data.py                # 分词（GPT-2 BPE/字符级兜底，词表只在 train 拟合）+ bin 缓存 + 批采样
│   ├── datacheck.py           # 防泄漏：时序/文档划分、去污染、token 硬校验、漂移筛查、test 访问账本
│   ├── train.py               # 训练循环（AdamW、cosine、bf16、W&B 可选）+ NaN 系统排查（激活钩子定位）+ 溯源清单
│   ├── evaluate.py            # held-out test PPL（缺 test.bin 时回退 val）
│   ├── ruler.py               # 简化 RULER：5 任务，token 级构造，保留 gold 跨度
│   ├── latency.py             # prefill / decode 延迟基准
│   ├── analysis.py            # 机制分析：集中≠找对、门控校准、AUROC
│   ├── curves.py              # 训练曲线诊断（过拟合/欠拟合/发散/NaN + 梯度尖峰）+ 处理建议
│   ├── badcase.py             # 错误样本归因（RULER + 语言建模最差窗口）+ 跨变体对照
│   ├── sanity.py              # 过拟合小批量 sanity check（训练前闸门）
│   ├── repro.py               # 代码版本/数据指纹/运行环境/种子审计/溯源继承
│   ├── experiment.py          # 追加式实验台账（跨轮次记录代码/配置/种子/数据/结果哈希）
│   ├── hypotheses.py          # 预注册完整性：锁基线 + 修订链 + 防篡改校验（CLI --verify/--init/--amend）
│   ├── plots.py               # 中文标注图表（PNG，含训练曲线）
│   └── utils.py               # 种子/JSON/AUROC/Pearson（无 scipy 依赖）
├── docs/
│   ├── design_notes.md        # 设计笔记、统计推断协议与结果分析协议
│   ├── registered_hypotheses.json       # 预注册登记表（禁止删除；修订须留痕）
│   └── registered_hypotheses.lock.json  # 锁基线 + 修订链（防事后静默改写/删除）
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
- **可复现性**：`set_seed` 固定 Python/NumPy/PyTorch（CPU/CUDA）种子，CUDA 下同时关闭 cudnn benchmark 并启用确定性算法（`warn_only`），并为子进程设置 `PYTHONHASHSEED`；批采样由显式 `np.random.default_rng(seed)` 驱动（无 DataLoader worker），数据顺序可复现且有单测。同一组随机种子训练全部变体，变体间用同种子配对统计（`run_all.py --seeds` / configs 中的 `seeds`）。跨硬件架构无法保证逐位一致，同机重跑稳定；每次训练另写 `run_manifest.json`（git/数据指纹/环境/命令行），评测 JSON 内嵌同一溯源字段。训练前默认执行过拟合小批量 sanity check（`--skip_sanity` 可跳过）。
- **torch.compile**：配置 `"compile": true` 可在 CUDA 上启用 torch.compile；保存检查点时统一使用剥离包装前缀后的参数键名，`load_model`（evaluate/ruler/latency/analysis）可直接加载。
- **标准注意力双路径**：SDPA 快速路径与显式路径数值一致（有单测），显式路径用于需要注意力统计的变体与延迟公平对比（`standard` vs `standard_sdpa` 两者都测）。
- **KV 缓存**：gated 变体在增量解码时逐 token 计算门控，缓存等价性有单测（全量前向 vs 逐 token）。
- **分词**：默认 GPT-2 BPE（tiktoken，首用需联网缓存，公开预训练词表不依赖本项目数据）；无网络时 `--tokenizer auto` 退化为字符级，词表只在 train 上拟合、未见字符映射 `<unk>`（PPL 不可与文献直接对比，但变体间对比有效）。数据进入训练前统一经过 `lega.data.prepare_bins`：坏字符清洗 → 跨 split 重复行去污染 → 拟合词表 → token 流硬校验 → 写 `data_report.json`。
- **设备**：`small.json` 建议 GPU + bf16；CPU 仅适合 tiny 冒烟。正式实验建议先跑 `lega_no_len` 与 `lega` 的对比（长度校准是核心主张）。
- **已知限制**：30M 模型的 RULER 绝对准确率不高（小模型检索能力有限），关注点是**变体间相对差异**与**机制分析指标**，而非绝对分数；`fwe` 的高频词以 k≥3 次、其它词以 2 次出现，答案按词集合评分，不存在词频并列歧义；ctx 过小时个别 `fwe`/`vt` 样例会因长度空间不足被跳过（JSON `_skipped` 字段如实记录，变体间跳过集合一致）。

## 6. License

MIT
