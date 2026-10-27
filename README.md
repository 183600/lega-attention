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

- **c 的性质**：均匀注意力下 $\max_j p_j = 1/n$，故 $c = \log(n\cdot 1/n) = 0$。但这只是**均匀分布特例**：一般注意力下 $\max_j p_j$ 未必按 $1/n$ 收缩，$c$ 仍可能随长度增长。因此 $c$ 不能被宣称为"已与长度解耦的充分统计量"；`lega/length_null.py` 给出随机/重复/均匀上下文的零假设诊断，正式主张必须同时报告 $c\sim\log_2 n$ 的斜率，不能只引用均匀情形的单元测试。
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
| `standard` | 无 | 基线（无门控） |
| `gate_only` | $g=\sigma(d)$ | 门控对照：只有可学习偏置，无 A/B/C（隔离“门本身”效应） |
| `qgate` | $g=\sigma(w^\top q+d)$ | 已有工作代表：只看查询的输出门控 |
| `null` | 无（额外 null KV 槽） | 已有工作代表：显式"选择不读取" |
| `lega` | $g=\sigma(w^\top q+a\,c+b\log n+d)$ | **本文方法**（全因子 A+B+C） |
| `lega_no_len` | $g=\sigma(w^\top q+a\,c+d)$ | 去一法：去掉长度项 $b\log n$（A+B） |
| `lega_no_conc` | $g=\sigma(w^\top q+b\log n+d)$ | 去一法：去掉集中度证据（A+C） |
| `lega_no_query` | $g=\sigma(a\,c+b\log n+d)$ | 去一法：去掉查询项（B+C） |
| `lega_conc_only` | $g=\sigma(a\,c+d)$ | 全因子：仅集中度项（B，$c=\log(n\max_j p_j)$） |
| `lega_len_only` | $g=\sigma(b\log n+d)$ | 全因子：仅长度项（C） |

**消融完整性**：设查询项=A、集中度项=B、长度项=C，则
全因子 8 格（门控内 2^3：无/`gate_only`、A/`qgate`、B/`lega_conc_only`、
C/`lega_len_only`、AB/`lega_no_len`、AC/`lega_no_conc`、BC/`lega_no_query`、
ABC/`lega`）与去一法（Full/Full−A/Full−B/Full−C）全部实现；无门的
`standard` 作为全因子范围外的外部基线（用于“LEGA vs 标准注意力”），
不当作 2^3 的 none 格，避免把“加门”本身误算成 A/B/C 的贡献。
汇总表会自动列出矩阵并标注缺失组合。组件是否有用不靠“组合跑全”自证：
汇总表基于 Holm 校正后的显著配对检验逐一判定 A/B/C（去一法：Full−X 显著变差；
全因子增量：X vs `gate_only` 显著变好），主指标按预注册：A/B 用 PPL、C 用 RULER；
另一指标只作探索性敏感性分析。全部变体统一走显式注意力路径
（`use_sdpa_standard=false`），避免 SDPA 与显式 softmax 混用引入实现差异；
审计会拒绝混用后端的对比。组合齐全但证据不足或方向相反时，
明确拒绝“各组件均有用”的结论。全因子矩阵还做 A×B/A×C/B×C 交互效应检验
（探索性、独立 Holm 家族，以 `gate_only` 为无组件基准），检验组件收益能否按单组件加总。

参数量差异：门控仅引入**每头 $d_{head}+3$ 个参数**（LEGA），相对模型总量 <0.02%（`null` 为每头 $2d_{head}$）。脚本会在训练开始时打印精确数值；汇总页另有“门控开销核算”小节，逐变体列出**实际逐头参数量、额外参数占比、解析门控 FLOPs/token 及其占总 FLOPs 比例**，并明确提示集中度项 `max_j p_j` 的计算随上下文长度线性增长——收益必须对照这两类开销解读，不能只报涨点。

---

## 2. 实验设计

- **数据协议（防泄漏，全部可审计）**：
  1. **先划分后预处理**：先产出 train/val/test 文本再编码；字符级词表（特征工程）只在 train 上拟合，val/test 只做 transform，未见字符统一映射 `<unk>`（GPT-2 BPE 是公开预训练词表，不依赖本项目数据）；
  2. **划分协议**：`--local_txt` 才按输入顺序做“时序式”切分——最早 80% 训练、其后 10% 验证、最新 10% 测试；切点对齐文档边界（标题行 > 空行段落 > 换行），绝不随机打乱。默认 WikiText-103 官方 split 是**按文档划分**，只保证文档不跨 split，**不保证时间先后**，不能写成“时序外推”或“未来数据”证据；若来源本身含基准测试文本，官方 split 也不能替代下游基准去污染。所有模式都必须通过跨 split 长行/近似重复核对，失败时拒绝写出（`--allow_leaky_split` 仅供诊断）。
  3. **群组不跨侧**：以文档/段落为整体单位，同一文档只落在一个 split；与 val/test 精确/NFKC-空白归一化重复的训练行、多行块自动去污染，并进一步用**近似重复筛查**（UTF-8 字节 5-gram MinHash + LSH 找候选，候选对精确算 Jaccard/containment；跨 split 取 max(Jaccard, containment)，train 内部只用 Jaccard；协议阈值 0.80，报告同时给 0.70/0.80/0.90 三档敏感性与 LSH 理论召回率）只删 train 侧副本（val/test 保持尘封不改写）；另用采样 64-gram 哈希筛查跨 split 的逐 token 复制；
  4. **清洗与异常检查**：坏字符（U+FFFD/NUL/非法控制符）确定性清除；token 流做整数/有限/id 越界硬校验；重复样本、超长行、train 与 val/test 的一元分布漂移（TV/JS、OOV）全部写入 `data_report.json`；
  5. **测试集尘封**：checkpoint 选择与早停只用 val，test 只在最终评估时访问；每次 test 评测写入 `data_dir/.test_access_log.jsonl` 账本，汇总表报告累计次数与被重复评测的 checkpoint。
- **公平对比（受控变量）**：所有变体 × 种子使用同一 config 训练（同一优化器/学习率策略/训练步数/数据顺序），训练后双层自动核对：`config_used.json`（超参数逐项一致）与 `eval.json` 溯源（数据指纹、代码 commit、配置 SHA-256、实际训练步数与 tokens_seen 一致，并与本次流水线数据比对），以及**实际评测协议**（PPL 的 `split/eval_ctx/eval_iters/val_windows/val_tokens_evaluated` 必须同批 token；RULER 的 `ctx_lens/n_samples/sample_seed/corpus_source/max_new` 等与已评测格集合必须一致，否则 EM 不是同题面配对），任何差异都会直接报错；模型初始化完成后统一重放随机种子，使各变体的训练期随机流（dropout 等）也完全一致——把“公平对比”从口头约定变成可执行断言。
- **调过的基线（防“超参数红利”误归因）**：加 `--tune_baseline` 时先对**基线（standard）**做 lr × weight-decay 小网格搜索——只用验证集选择（`lega/tuning.py`，test 从不加载），选定组合**统一注入全部变体**训练；输出 `results/<tag>/baseline_tuning.json`（含 `default_is_optimal`：默认超参数与网格最优的 val_loss 差是否在 0.5% 容差内）并在汇总页渲染。没有这条证据时，汇总页显式警告“未调优基线”，而不是默认基线已调好。选择本身也有噪声：`--tune_seeds` 让每个网格组合在多个种子上重复训练、**按平均 val_loss 选择**，报告 `selection_seeds`/逐种子明细；不指定时只用单个种子（首个正式种子），汇总页会显式警告 winner's curse（“幸运种子”可能把更差的配置选成最优），正式实验建议给 ≥3 个。**简单统计基线**：`lega/ngram.py` 给出与神经模型同 split 的一元/二元/**插值二元**平滑 n-gram PPL（α 与插值权重 λ 只在 val 上联合选，`selected_model` 取 val 最强简单模型），回答“数据绝对难度有多高、神经模型是否显著优于充分调优的无参数基线”。
- **对简单基线的显著性（补充性对比）**：`--ngram_split` 默认 `auto`，与神经评测同一规则（有非空 `test.bin` → test，否则 val），从根上避免 val 评测的神经模型与 test 基线的错位比较；**同 split 还不够，必须同 token**——神经 PPL 只在 `eval_iters` 个等间隔窗口的目标 token 上计算，n-gram 因此同时报告逐位置对齐口径 `ppl_window_matched`（`lega.ngram.window_positions` 与 `train._val_eval_indices` 同规则，评分位置为每个窗口的 `s·ctx+1..s·ctx+ctx`），汇总优先用对齐参考；任一 run 的 `eval_ctx/eval_iters/val_windows` 与窗口规格不符（历史产物等）时整体回退全 split 参考并在汇总页显式披露 `fallback_reason`（同一 Holm 家族共用同一参考值）。汇总页把**每个变体逐种子 PPL 与同一常数基线做单样本配对检验**（n-gram 无种子方差），按变体家族 Holm 校正，输出比值/相对降低/p_adj/判定；split 不一致的变体单列为“未纳入比较”，绝不混进表里。此对比未预注册，标注为探索性证据，不替代变体间受控配对检验。
- **统计可靠性（多种子）**：关键指标跨随机种子报告 **Mean ± Std** 与配对 Cohen's d 效应量（`configs/small.json` 默认 5 个种子，`--seeds` 可覆盖）；变体间同种子配对的双侧 t 检验在同一指标家族内做 **Holm–Bonferroni 校正**（控制多重比较 FWER），结论一律基于校正后 p 值（`results/<tag>/significance.json`）。n<3 时只报描述统计、不判定显著性，避免“提升 0.2%、方差 0.5%”被当成有效增益；每项配对同时给出 MDE 与效应/噪声比（|Δ|/ΔSD），**|Δ| 小于逐种子配对差标准差时显式标注为“噪声内”的边缘证据**，即使校正后显著也不当作稳健收益。
- **实际意义与等价性（SESOI/TOST）**：显著性检验只回答“能否拒绝 Δ=0”；p>0.05 既可能是“没有实际重要差异”，也可能只是功效不足。`lega/equivalence.py` 在协议 SESOI（PPL 逐种子相对差 1%、RULER EM 1.00 pp、AUROC/PR-AUC 差 0.010、因果 ΔNLL 0.01 nat）上做配对 TOST：只有 90% 区间完全落在 ±SESOI 内才判“等价”，单侧支持才判“非劣”；统计显著但幅度低于 SESOI 标为“实际意义有限”；区间跨过 0 或 SESOI 只能写“未定”，不能写成“无差异”。TOST p 在同一指标家族内单独 Holm 校正，0.5×/1×/2× margin 网格各自校正并并列披露；**SESOI 是分析前声明的协议约定而非数据估计**，等价性判定不改变预注册的优效性判定；该分析当前未写入预注册锁链，按补充性/描述性证据解读，升级为确认性主张须先登记 margin 与对比并在新数据上检验。`summary.md` 新增“SESOI 与等价性/非劣性”和“关键病灶的 SESOI 解读”两节，`significance.json` 的 `pairs[*].equivalence` 与顶层 `equivalence` 可机器读取；`head_permute` 安慰剂只有在 TOST 判等价时才能说“差异小于 SESOI”，p>0.05 不能直接写成“安慰剂无效应”。
- **推理期因果消融（训练后病灶，回答“训练好的模型是否真的在用该组件”）**：训练期全因子/去一法消融改变优化轨迹，不能证明最终前向仍在使用某组件。`lega/ablate.py` 在**同一 checkpoint、同一批 val 窗口**上只改门控参数并逐窗口配对：`no_query/no_conc/no_len/no_evidence` 是训练期去一法的推理期对应；`gate_open/gate_closed` 检验门控写回强度（后者是“关闭注意力输出必须显著变差”的阳性对照）；`gate_constant_mean` 保留每个头/层的平均写入强度、去掉逐位置/逐查询变化；`head_permute` 在参数个数/分布/结构完全不变的前提下打乱头—证据对应，是容量解释的**安慰剂对照**；`conc_permute` 把逐位置 `c` 沿序列维循环移位，保持 c 的逐值分布/均值/方差/长度分布不变，只破坏位置对应，回答“收益是否来自容量/尺度”而不是证据位置；`sign_flip` 保留幅度、反转证据方向。病灶后逐位恢复，`no_op`（系数训练后已为 0）如实标注——“无差异”不是“无效应”的证据。窗口级 p 仅描述单 run；汇总先按种子聚合 ΔNLL，再跨种子配对 t + Holm（`significance.json.causal_ablation`）。默认只在 val 上做（test 尘封，val 是选型集，只作机制定位）。训练期显著而推理期病灶无差异，说明收益可能来自训练期优化/正则化；推理期显著而训练期不显著，则提示训练期消融被再训练补偿——两种证据并列，不能互相替代。边界：`head_permute` 是**推理期**容量/分布安慰剂，不是训练期参数量匹配对照；`no_evidence` 保留全部参数、只关证据输入，能隔离“推理期是否使用证据”，但不能消除“训练期多出的门控参数是否带来优化红利”这一混淆，需与参数开销、单因素消融合读。
- **全因子 Shapley 分解（交互感知的补充归因，探索性）**：单因素边依赖参照格，交互非零时“加入 X 的收益”在 `gate_only` 与 `Full` 基准下并不相同。`lega/factorial.py` 对门控内 8 格做精确 Shapley 分解：PPL 在 `−log(PPL)` 上分解（φ>0 表示降低 PPL，`exp(−φ)` 即 PPL 几何均值比），RULER 用原始 EM；效率性保证 `φ_A+φ_B+φ_C=Full−none`，交互被平均分摊到参与因子。逐种子算 φ，再同种子配对 t + 同指标 Holm；占比只在总效应为正时报告，不强行归一化。该分解未预注册，不能替代预注册的单因素确认检验；与单因素边方向冲突时并列披露。
- **预注册与防事后改写（confirmatory vs exploratory）**：正式实验前把方向性主张登记在 `preregistration/hypotheses.json`（**刻意不放 `docs/`**：流水线清理会删除该目录下除 `design_notes.md` 外的文件，该登记表曾被自动清理提交多次误删），汇总时逐条对照 Holm 校正后的同种子配对检验，判定 支持 / 拒绝（方向相反且显著=被证伪）/ 未定（未定≠无效应，须对照 MDE 解读），结果进 `summary.md` 与 `significance.json`；未登记的对比一律属事后（exploratory），只能作为线索。登记表由 `preregistration/hypotheses.lock.json` 防篡改：锁基线记录整文档 SHA-256 与**逐假设科学内容摘要**，修订必须经 `python -m lega.hypotheses --amend --reason ...` 在**修订链**上留痕（既有假设只增不改不删）。删除登记表、改写 claim、调换变体对、删条目或未留痕新增，都会被 `--verify`／汇总页判为 `violated`，并**自动把全部确认性判定降级为探索性**；登记表的意外删除与恢复/迁移事件本身也必须在 amendments 中留痕（本项目已发生过一次清理误删，见 `docs/design_notes.md`）。锁链完整还不够：汇总页还做**预注册范围核对**——登记表里的 `scope`（配置规模、数据/分词器、种子计划）必须与本次实跑（`repro.json` 的 `config_file`/数据指纹/种子）对上；tiny/冒烟/伪数据/非同登记分词器都会判为范围外，确认性判定同样自动降级（调优注入 lr/权重衰减属协议内，不算越界；未登记种子只警告）。
- **三个评测维度**：
  1. **困惑度**（WikiText-103 PPL：checkpoint 选择在 val 上完成，PPL 在 held-out test 上评测，无 test 时回退 val）；评估窗口不重复，并在整个评测集上等间隔抽样（窗口数不足时评估全部）；`gate_mean` 与 PPL 使用同一批窗口、同一精度口径；**一元/二元/插值二元平滑 n-gram PPL** 作为简单统计基线并报（`ngram_baseline.json`，α/λ 只在 val 上选，参照取 val 最强简单模型），默认与神经评测同一 split（`--ngram_split auto`，避免错位比较），避免“绝对难度未知”下解读神经模型分数；
  2. **RULER 长上下文检索**（简化复现 5 任务）：`niah_synth`（合成干扰单键检索）、`niah_corpus`（自然文本干扰）、`niah_multikey`（多键干扰）、`fwe`（高频词提取）、`vt`（变量追踪）；多档上下文长度 + 贪心解码；数值任务 EM **只承认生成内容中的第一个数值回答**，避免"先答错后蒙中"被计分；
  3. **实际推理延迟**：生成推理口径的 prefill（只算最后位置 logits）与带 KV 缓存的逐 token decode，对比 SDPA 快速路径。
- **核心风险检验（不能只看"注意力沉降减少"）**：`lega/analysis.py` 用带 gold 跨度的 RULER 样例做机制分析：
  - **misleading_rate**：注意力很集中（$\max_j p_j>\tau$）但落点不在答案数值跨度（value_span）内的比率 —— hit 判定以答案数值的精确 token 跨度为准（而非整条 needle 句），"集中 ≠ 找对"的直接证据；
  - **P(hit | 集中)** vs **P(hit | 不集中)**；
  - **AUROC 与 PR-AUC（不平衡口径并报）**：门控分数区分"找对/没找对"的能力，且 **LEGA 的完整 logit vs 仅查询项 $w^\top q$ 的 AUROC 之差 / PR-AUC（average precision）之差**，直接回答"结合长度校准后的注意力统计是否比只看查询更能判断该不该写入"。hit/¬hit 是类别极不平衡的二分类（正类率即 `hit_base_rate`），正类率 <5% 时 AUROC 会因负样本基数巨大而乐观，**以 PR-AUC 为主指标**（其随机基线恰为正类率，便于对照）；正类率 <0.1% 时尤其如此。**门控决策点 F1**（$g>0.5$）同时给出 **macro / micro / weighted 三口径**（`analysis_*.json` 的 `gate_f1`，汇总表列 macro）——不平衡数据下三者差异巨大，任何只写"F1"而不注明平均口径的表格都不完整；同时并报**固定规则稻草人对照 `gate_f1_trivial`**（恒写/恒不写），门控若打不过无信息基线则谈不上决策价值（汇总表列“恒写基线F1(macro)”）；RULER 的 EM 是生成式指标、无阈值，不受此影响（fwe 另报召回率）。**概率校准**：门控 g 被当作软权重/概率使用，故另报 Brier/ECE 并与常数基率 Brier 基线对照（`analysis_*.json.calibration_g`）——排序指标好不等于 g 的数值可信；线性探针同样报 Brier/ECE。**RULER 平均口径**：汇总表同时给 macro（任务等权、任务内上下文档位等权，预注册口径）与 micro（按每格样本数加权），避免“平均”定义不清掩盖任务/格大小不均。

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
# 可选 --max_tokens 30000000 控制训练规模；
# 近似去污染阈值可显式设置：--near_dup_threshold 0.8 --near_dup_min_chars 200

# 2) 全流水线：10 变体 × 多种子（默认 5 个）训练 → PPL → RULER → 延迟
#    → 机制分析 → 图表 → 汇总表（Mean ± Std + 配对显著性检验）
python run_all.py --config configs/small.json --data_dir data/wikitext103

# 可选：自定义种子（如 3 个）或 W&B 日志
python run_all.py --config configs/small.json --data_dir data/wikitext103 --seeds 1337,2024,42
wandb login
python run_all.py --config configs/small.json --data_dir data/wikitext103 --wandb

# 可选（推荐）：先调基线再全量训练——standard 的 lr/权重衰减网格（仅用 val），
# 每组合跑 --tune_seeds 指定的多种子并按平均 val_loss 选择（防幸运种子，
# 默认沿用 --seeds），选定组合统一用于所有变体，证据写入
# results/<tag>/baseline_tuning.json
python run_all.py --config configs/small.json --data_dir data/wikitext103 \
  --tune_baseline --tune_seeds 1337,2024,42

# 可选：推理期病灶消融默认对 lega,qgate 做（核心对比双方同一套病灶），
# 可再加 gate_only 作对照；同一 checkpoint/同一批 val 窗口，只改门控参数；
# 不访问 test
python run_all.py --config configs/small.json --data_dir data/wikitext103 \
  --ablate_variants lega,qgate,gate_only

# 可选：早停——验证连续 N 次评测无实质改善即终止（默认关闭；启用后各变体
# 实际步数可不同，受控变量审计按计划步数核对，汇总页会标注「早停」）
python run_all.py --config configs/small.json --data_dir data/wikitext103 --early_stop_patience 5

# 简单统计基线可单独运行（默认 val；test 需显式选择并写访问账本）。
# 全流水线里 --ngram_split auto（默认）会与神经评测同 split；神经变体
# 逐种子 PPL 与基线做单样本配对检验 + Holm 校正并渲染进汇总页
python -m lega.ngram --data_dir data/wikitext103 --split val --out results/ngram_baseline.json
```

结果输出：

| 路径 | 内容 |
|---|---|
| `out/<tag>/<variant>[/seed<seed>]/model.pt` | 检查点 + 训练配置 + 代码/数据溯源字段（多种子时按种子分子目录） |
| `out/<tag>/<variant>[/seed<seed>]/train_log.jsonl` | 训练/验证曲线数据 |
| `out/<tag>/<variant>[/seed<seed>]/run_manifest.json` | **溯源清单：git commit/dirty、dirty 工作区 diff 指纹/文件清单（含未跟踪文件，`worktree.patch` 可重放、截断标记）、配置 SHA-256、数据指纹、运行环境（含 TF32/线程数/CUDA 可见性与确定性算法严格/warn_only 模式）、命令行、开训前 sanity 闸门结论/哈希/身份核对** |
| `results/<tag>/summary.md` | **汇总表（Mean ± Std + 消融矩阵 + 受控变量自检 + 可复现性/sanity）** |
| `results/<tag>/repro.json` | 流水线全局溯源（代码版本、数据指纹、种子、运行环境） |
| `results/experiments.jsonl` | **追加式实验台账：每轮 run 的 run_id、代码版本/工作区 diff、解析后的完整超参数、逐 run 种子审计与随机源指纹、数据指纹、sanity 结论与身份、最终指标与产物哈希（started/completed/aborted 生命周期，失败含原因）；`python -m lega.experiment --check_artifacts` 可核验链与哈希** |
| `data/<data_dir>/data_report.json` | 数据防泄漏自检（划分协议、词表拟合集、精确/近似去污染、64-gram 重叠、分布漂移、超长行、访问账本） |
| `results/<tag>/sanity_<variant>.json` | 过拟合小批量 sanity check 结果与阈值 |
| `results/<tag>/baseline_tuning.json` | 基线（standard）lr/权重衰减调优证据（仅验证集；含 `default_is_optimal` 近优判定、`selection_seeds`/逐种子 val_loss 明细与逐 trial 训练溯源） |
| `results/<tag>/ngram_baseline.json` | 简单统计基线：一元/二元/插值二元平滑 n-gram 的 held-out PPL（α/λ 在 val 上联合选，`selected_model` 取 val 最强；test 需显式开锁；流水线默认与神经评测同 split） |
| `results/<tag>/significance.json` | 同种子配对 t 检验（Δ均值 ± Δ标准差、配对 Cohen's d_z、原始 p、Holm 校正 p_adj、组件有效性证据）；含 `registration_integrity`（预注册锁链校验）、`registration_scope`（范围核对：配置规模/分词器/种子计划）、`registered_hypotheses`（逐条判定，早停/完整性/范围未过则降级）与 `factorial_interactions`（全因子交互检验，含 `training_budget` 早停可比性标记）；另含 `factorial_shapley`（交互感知的全因子 Shapley 分解）、`causal_ablation`（推理期病灶的跨种子配对检验）与 `equivalence`/`pairs[*].equivalence`（SESOI/TOST 等价性、非劣性、90%/95% 区间、0.5×/1×/2× margin 敏感性及 Holm 校正）；另含 `ngram_comparison`（逐变体 vs 简单基线单样本配对检验） |
| `results/<tag>/ablation_*.json` | **推理期因果病灶消融**（默认 `lega,qgate`，`--ablate_variants` 可扩展）：同一 checkpoint/同一批 val 窗口上的 `no_query/no_conc/no_len/no_evidence`、`gate_open/closed/constant_mean`、`head_permute`（头参数安慰剂）、`conc_permute`（c 分布匹配置换）、`sign_flip` 干预；含逐窗口 ΔNLL、PPL 比、门控统计、`no_op` 标记与溯源 |
| `results/<tag>/curve_diagnosis.json` | 逐 run 训练曲线诊断（过拟合（含训练停滞+验证回升的平台期）/过拟合趋势（训练降而验证停滞）/欠拟合/发散/NaN + 梯度范数尖峰/非有限 + 处理建议） |
| `results/<tag>/badcases_*.md` | RULER 错误样本归因（标注错误/误抄干扰值/幻觉/无数值回答/fwe）+ 粗归因轴计数 + **改进方向**（触发证据→建议→下一步实验） |
| `results/<tag>/worstcases_*.md` | 语言建模最差窗口归因（复读吸引子/高置信错答/低置信；gold 是否在上下文中出现）+ **改进方向** |
| `results/<tag>/badcase_cross.json` | 跨变体错误对照（全体共识失败 vs 相对 standard 的特异性新增/修复；含逐变体新增错误类别计数与跨变体改进方向） |
| `results/<tag>/eval_*.json` | 各变体 held-out test PPL（无 test.bin 时回退 val），含 worst_windows 坏案例与训练吞吐/成本（`train_step_time_ms`/`train_tokens_per_sec`/`train_peak_mem_mb`，供效率-精度权衡）与早停记录（`steps_planned`/`steps`/`early_stopped`） |
| `results/<tag>/ruler_*.json` | RULER 5 任务 × 多档上下文，含 badcases 错误样本 |
| `results/<tag>/latency.json` | prefill / decode 延迟（逐次计时的中位数 + `*_spread_pct=(p90-p10)/median` 测量散布，供判断变体差异是否超出噪声） |
| `results/<tag>/analysis_*.csv|json` | 逐 (层，头，查询位) 门控记录与汇总 |
| `results/<tag>/fig_*.png` | 全部图表（中文标注，含训练曲线 `fig_train_curves.png`） |

### 3.4 可复现性、sanity check 与实验追踪

- **随机种子全覆盖**：`lega.utils.set_seed` 固定 Python `random`、NumPy、PyTorch（CPU/CUDA）、cuDNN 确定性算法（`warn_only`）与 `PYTHONHASHSEED`（子进程）；本项目的批采样使用显式 `np.random.default_rng(seed)`（无 torch DataLoader/worker），数据顺序由 seed 唯一确定，单测覆盖（`tests/test_repro.py`）。`run_all` 还会逐命令把子进程的 `PYTHONHASHSEED` 对齐到该命令自己的 `--seed`（而非统一用第一个种子），每个 run 的 `run_manifest.json` 内嵌 `seed_report` 审计记录：逐随机源的实际种子、**各 RNG 真实状态的短指纹**（同 seed 可被第三方用 `set_seed(seed)+rng_state_fingerprint()` 复算验证，异 seed 必然不同，当前状态不属于该 seed 时为 `null` 而非张冠李戴）、cuDNN/确定性开关（**如实标注严格模式还是 `warn_only`，避免把“只警告”写成“严格确定性”**）、数据顺序机制，并如实区分 `PYTHONHASHSEED` 的“解释器启动时取值 vs 仅对子进程生效的当前 env”（启动时未设置时 `pythonhashseed_effective=false`，不再让字段被 `set_seed` 自身写成 seed 而自证为真），杜绝“声称固定了其实没有”。运行环境快照同时记录 TF32（matmul/cudnn）、torch 线程数与 `CUDA_VISIBLE_DEVICES`；训练清单在 `enable_tf32` **之后**生成，记录的是实际生效值而非默认值。跨硬件架构（不同 GPU/CPU/cuDNN 版本）无法保证逐位一致，但同一台机器上重跑结果稳定。
- **过拟合小批量 sanity check（训练前闸门）**：`lega.sanity` 抽取固定小批量（默认 8 条 × 128 token），关闭 dropout/weight decay 与全部数据扰动，在**同一批数据**上优化，检查损失能否迅速趋近 0、下一 token 准确率能否接近 100%（阈值可用 `--loss_threshold/--acc_threshold/--initial_loss_tol` 或配置 `sanity` 块调整）。`run_all.py` 在全量训练前对每个变体自动执行，失败立即中止并写入台账 `aborted` 记录（`--skip_sanity` 跳过，`--sanity_steps` 调节步数），通过时把 sanity 结果摘要（路径/SHA-256/判定/未通过项）绑定进该 run 的 `run_manifest.json`，checkpoint 可自证“开训前过了代码正确性闸门”。这是**必要非充分**检查：连小批量都无法过拟合说明梯度反传/损失/标签存在严重 bug；通过也不代表数据增强、特征对齐、标签映射等 bug 不存在。sanity JSON 除损失曲线与首步梯度范数外，还**自动核对**：首步训练路径 loss ≡ 直接 logits CE、标签 y=x 右移一位、初始 loss≈log(V)（默认容差 1.0 nat，负值可禁用），并做**首步逐参数梯度覆盖**（`missing/zero/nonfinite`，可定位“总梯度范数正常但某个子模块没接进计算图”的断流 bug）；同时强制核对 dropout/attn_dropout 已关闭（未关闭即失败，避免把配置错误误判为代码 bug）。sanity 证据绑定到 run 时会核对 variant/config/data/git（含 dirty diff）身份：旧配置/旧数据/旧代码的 PASS 不会被当成本 run 的通过证据。
- **实验追踪/可追溯**：每次训练自动写 `run_manifest.json`（git commit/dirty、配置 SHA-256、数据指纹、种子审计、含实际精度开关的 Python/PyTorch/CUDA 环境、完整命令行、sanity 闸门证据），超参数在 `config_used.json`，训练曲线在 `train_log.jsonl`（并汇总为 `fig_train_curves.png`）；`run_all` 写全局 `results/<tag>/repro.json`，并在固定路径 `results/experiments.jsonl` **追加**一轮实验台账记录（开始 `started` + 结束 `completed`，失败为 `aborted`，同一 `run_id`）：代码版本、**解析后的完整超参数（config 文件被改写/删除后仍可还原）**、逐 run 的 `config_used.json` 内容、种子、数据指纹、sanity 结论（含未通过项）、逐 run 最终指标与 `train_log.jsonl`/被评测 checkpoint/全部结果文件的 SHA-256（`eval_*.json` 内亦记录 `ckpt_sha256`，结果与产出的权重一一绑定）。重跑不再丢失上一轮历史；失败轮次（sanity 闸门或训练/评测链）以 `aborted` 记录失败阶段、原因与退出码，失败实验同样完整留痕。台账快照还内嵌逐 run 的 `seed_report`/运行环境/sanity 身份，manifest 被删除后仍可自证；dirty 工作区的补丁与指纹随 run 保存（已跟踪 diff + 未跟踪文件，预算 8 MiB 并显式标记截断），仅凭 commit 无法复原代码时仍可重放。评测结果（eval/ruler/analysis/latency）均内嵌同一份代码/数据溯源字段，不依赖手工记录。

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
python -m lega.ablate    --ckpt out/small/lega/seed1337/model_best.pt --data_dir data/wikitext103 --out results/small/ablation_lega_seed1337.json
python -m lega.plots     --results_dir results/small --train_root out/small
python -m lega.sanity    --config configs/small.json --data_dir data/wikitext103 --variant lega --out results/sanity_lega.json
python -m lega.experiment --check_artifacts                  # 台账完整性/产物哈希核验（0=通过；1=异常）
python -m lega.curves    --run_dir out/small/lega/seed1337   # 训练曲线诊断（过拟合/欠拟合/发散/NaN）
python -m lega.badcase   --results_dir results/small         # 错误样本归因（RULER + 语言建模最差窗口）+ 改进方向合成
python -m lega.hypotheses --verify                           # 预注册完整性校验（0=可确认；1=被删/改/未留痕）
python -m tests.run_tests   # 单元测试（不依赖 pytest）
```

结果分析口径（详见 `summary.md` 对应小节）：训练诊断看曲线而非只看最终值，
除过拟合/欠拟合/发散外还独立报告 **pre-clip 梯度范数尖峰/非有限**（NaN 与发散的
早期信号，损失曲线可能尚未反映；训练日志按 log 区间记录 `grad_norm_max` 与发生步，
避免只看区间最后一步漏掉中段爆炸）和**单步损失峰值尖峰**（`loss_max` 防止一步
异常批次被区间均值稀释）；训练损失停滞而验证回升同样判为过拟合
（平台期泛化先恶化，不因"训练没降"误判成欠拟合），训练仍在下降而验证已停滞
则单独判 **过拟合趋势**（按正则化/早停处理，不据此续训）；全程首/末分段之外
还比较末段（后 1/3 vs 中 1/3），避免"早先大幅下降、末段才平台/回升"被全程均值掩盖；NaN 按数据→代码
（log(0)/除零/softmax 溢出，前向钩子给出首个"输入有限、输出非有限"的模块即
`origin`）→混合精度下溢→梯度爆炸/学习率的顺序排查（`nan_report.json`），
数据与激活检查均优先分析失败步批次（预取会覆盖待训练批次），val 损失非有限时
直接改用真实验证窗口（训练批次仅作取窗失败的回退），混合精度路径出现非有限时
还会对同一批次做 **fp32 隔离复测**（fp32 有限→判混合精度数值问题；fp32 同样
非有限→排除精度因素）；过拟合建议的"早停"是**可选的实际机制**（
`--early_stop_patience N`，或配置 `early_stop_patience`/`early_stop_min_delta`：
验证连续 N 次评测无实质改善即终止，默认关闭不影响受控对比；启用后各 run 实际
步数可不同，`eval.json` 同时记录计划/实际步数，受控变量审计按计划步数核对并在
汇总页如实标注"早停"与截断原因；由于实际训练量（tokens_seen）不等，预注册
确认性判定会自动降级为探索性（`apply_budget_gate`），需在相同实际步数下重跑
才能恢复 confirmatory 地位，`significance.json` 的 `component_evidence` 也同步写入 `training_budget_comparable=false` / `confirmatory_supported=false` / `evidence_status="exploratory"`，不会在 JSON 里继续冒充确认性证据；下游仍使用 val 选出的 `model_best.pt`）；
错误样本逐条归因（每个案例附粗归因轴：数据/构造、难度边界、模型/方法），并用
跨变体对照区分"全体共识失败（数据/难度）"与"方法特异性盲区"；归因最终**合成为
可执行的改进方向**（触发证据→建议→下一步实验，确定性阈值见 `lega/badcase.py`），
同时进入逐案例 md、`badcase_summary.json` 与 `summary.md`——坏案例分析的落点不是
分类表，而是提出新改进思路的最重要来源；效率-精度权衡
显式判定，增幅取 **Prefill/Decode 推理延迟、训练 ms/step、训练峰值显存、
参数量、FLOPs、推理峰值分配** 相对基线的最大者，其中延迟为逐次计时的中位数，**差异不超过两端
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
│   ├── attention.py           # 核心：10 个主变体 + 5 个补充/对照的因果注意力（LEGA 门控公式与消融矩阵在此）
│   ├── model.py               # GPT 主干 + RoPE + KV 缓存生成 + 参数量工具
│   ├── data.py                # 分词（GPT-2 BPE/字符级兜底，词表只在 train 拟合）+ bin 缓存 + 批采样
│   ├── datacheck.py           # 防泄漏：时序/文档划分、精确+近似去污染、token 硬校验、漂移筛查、test 访问账本
│   ├── train.py               # 训练循环（AdamW、cosine、bf16、W&B 可选）+ NaN 系统排查（激活钩子定位）+ 溯源清单
│   ├── evaluate.py            # held-out test PPL（缺 test.bin 时回退 val）
│   ├── ruler.py               # 简化 RULER：5 任务，token 级构造，保留 gold 跨度
│   ├── latency.py             # prefill / decode 延迟基准
│   ├── analysis.py            # 机制分析：集中≠找对、门控 Brier/ECE 校准、AUROC + PR-AUC（不平衡主指标）+ 门控 F1（三口径 + 恒写/恒不写稻草人对照）
│   ├── ablate.py              # 推理期因果病灶消融（训练后权重干预，与训练期消融互补）
│   ├── length_null.py         # c 的长度零假设诊断（随机/重复/均匀上下文 + 斜率）
│   ├── output_ablation.py     # 答案 token 的逐层/逐头输出干预（ΔNLL/Δmargin 写回效用）
│   ├── factorial.py           # 全因子 2^3 的精确 Shapley 分解（交互感知的消融归因）
│   ├── equivalence.py         # SESOI + TOST 等价性/非劣性（Holm 校正、0.5×/1×/2× 敏感性）
│   ├── ngram.py               # 简单统计基线：一元/二元/插值二元平滑 n-gram PPL（α/λ 只在 val 联合选；test 需开锁并记账；auto 与神经评测同 split）
│   ├── curves.py              # 训练曲线诊断（过拟合/欠拟合/发散/NaN + 梯度尖峰）+ 处理建议
│   ├── badcase.py             # 错误样本归因（RULER + 语言建模最差窗口）+ 跨变体对照 + 改进方向合成
│   ├── sanity.py              # 过拟合小批量 sanity check（训练前闸门）
│   ├── repro.py               # 代码版本/数据指纹/运行环境/种子审计/溯源继承
│   ├── experiment.py          # 追加式实验台账（跨轮次记录代码/配置/种子/数据/结果哈希）
│   ├── hypotheses.py          # 预注册完整性：锁基线 + 修订链 + 防篡改校验（CLI --verify/--init/--amend）
│   ├── tuning.py              # 基线（standard）lr/权重衰减网格调优（仅验证集；多种子按均值选择，default_is_optimal 判定）
│   ├── plots.py               # 中文标注图表（PNG，含训练曲线）
│   └── utils.py               # 种子/JSON/AUROC/PR-AUC/F1/Pearson（无 scipy 依赖）
├── preregistration/           # 预注册（刻意不放 docs/：流水线清理会删除 docs/ 下非白名单文件）
│   ├── hypotheses.json        # 预注册登记表（禁止删除/移动；修订须留痕）
│   └── hypotheses.lock.json   # 锁基线 + 修订链（防事后静默改写/删除）
├── docs/
│   └── design_notes.md        # 设计笔记、统计推断协议与结果分析协议
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
- **门控查询项的定位**：门控里的 $q$ 使用 **RoPE 旋转前**的查询内容向量；注意力打分仍使用旋转后的 $q,k$。显式位置编码不进入 $q$，但 token 内容本身仍可能与位置/长度相关，不能据此宣称已消除全部位置混淆；$c$ 中也含 $\log n$，需与 $b\log n$ 一并解读。
- **可复现性**：`set_seed` 固定 Python/NumPy/PyTorch（CPU/CUDA）种子，CUDA 下同时关闭 cudnn benchmark 并启用确定性算法（`warn_only`），并为子进程设置 `PYTHONHASHSEED`；批采样由显式 `np.random.default_rng(seed)` 驱动（无 DataLoader worker），数据顺序可复现且有单测。同一组随机种子训练全部变体，变体间用同种子配对统计（`run_all.py --seeds` / configs 中的 `seeds`）。跨硬件架构无法保证逐位一致，同机重跑稳定；每次训练另写 `run_manifest.json`（git/数据指纹/环境/命令行），评测 JSON 内嵌同一溯源字段。训练前默认执行过拟合小批量 sanity check（`--skip_sanity` 可跳过）；基线调优的每个 trial 子进程也把 `PYTHONHASHSEED` 对齐到自己的 seed。
- **torch.compile**：配置 `"compile": true` 可在 CUDA 上启用 torch.compile；保存检查点时统一使用剥离包装前缀后的参数键名，`load_model`（evaluate/ruler/latency/analysis）可直接加载。
- **标准注意力双路径**：SDPA 快速路径与显式路径数值一致（有单测），显式路径用于需要注意力统计的变体与延迟公平对比（`standard` vs `standard_sdpa` 两者都测）。
- **KV 缓存**：gated 变体在增量解码时逐 token 计算门控，缓存等价性有单测（全量前向 vs 逐 token）。
- **分词**：默认 GPT-2 BPE（tiktoken，首用需联网缓存，公开预训练词表不依赖本项目数据）；无网络时 `--tokenizer auto` 退化为字符级，词表只在 train 上拟合、未见字符映射 `<unk>`（PPL 不可与文献直接对比，但变体间对比有效）。数据进入训练前统一经过 `lega.data.prepare_bins`：坏字符清洗 → 跨 split 精确/近似去污染 → 拟合词表 → token 流硬校验 → 写 `data_report.json`。
- **设备**：`small.json` 建议 GPU + bf16；CPU 仅适合 tiny 冒烟。正式实验建议先跑 `lega_no_len` 与 `lega` 的对比（长度校准是核心主张）。`eval_iters=50` 只是逐窗口抽样，不是完整 val；summary 必须按窗口抽样误差解读，正式 PPL 结论需补全量 val/多起点滑窗。
- **输出级写回效用**：`lega/output_ablation.py` 在答案 token 的预测位置对 `wo` 输入逐层/逐头置零或置换，报告同一批答案 token 的 ΔNLL/Δmargin；`hit`/集中度只是注意力分布代理，不能替代该因果干预。
- **无证据条件**：`lega/ruler.py` 提供可选 `evidence_absent/evidence_wrong/evidence_conflict`（显式要求不可靠时输出 `NONE`，`NONE` 才记 EM=1）。它们不进入默认 5 任务主表，需在调用 `run_ruler(tasks=...)` 时显式加入；用于检验门控是否在无可靠证据时抑制写回。
- **长度边界**：训练与评测都受 `block_size`（small=1024）限制；RULER 的 256/512/1024 是**域内**长度扫掠，不是长度外推。任何超过 `block_size` 的位置都应被拒绝，不能把 benchmark 的更长上下文结果当作本模型能力。
- **补充对照**：`lega_raw`（原始 $\log\max_j p_j$，不并入 $\log n$）、`lega_raw_no_len`（原始集中度且无 $b\log n$）、`lega_detach_conc`（集中度统计不回传 Q/K）、`null_zero`（固定零 value、null 键仍可学）不属于主 2^3 因子设计，默认不训练；它们分别诊断“归一化/重参数化”“反传路径”“空槽写入”替代解释。
- **已知限制**：30M 模型的 RULER 绝对准确率不高（小模型检索能力有限），关注点是**变体间相对差异**与**机制分析指标**，而非绝对分数；`fwe` 的高频词以 k≥3 次、其它词以 2 次出现，答案按词集合评分，不存在词频并列歧义；ctx 过小时个别 `fwe`/`vt` 样例会因长度空间不足被跳过（JSON `_skipped` 字段如实记录，变体间跳过集合一致）。

## 6. License

MIT
