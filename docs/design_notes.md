# 设计笔记

## LEGA（Length-calibrated Evidence-Gated Attention）

注意力门控同时参考三个因素：查询向量 $q$、注意力集中度 $c=\log(n\cdot\max_j p_j)$、上下文长度的对数 $\log n$：

$$g=\sigma(w^\top q+a\,c+b\log n+d),\qquad o'=g\cdot o$$

- $c$ 在均匀注意力下恒为 0，但这只是一特例；一般情形 $c$ 仍可能随长度增长，不能据此声称已把"读得集中"和"读得长"解耦。`lega/length_null.py` 提供随机/重复/均匀上下文的零假设诊断；任何"长度解耦/长度校准"主张必须同时报告 $c\sim\log_2 n$ 斜率。统一流水线现在会在训练前自动生成 `results/<tag>/length_null.json`（`lengths`/`modes`/`score_sd`/`n_trials`/`seed` 由配置 `length_null` 块控制），写入 `pipeline_run_id` 与 `config_sha256`，并在汇总页新增“长度零假设诊断（c ~ log2 n）”小节；缺失或不属于本轮 run 的诊断会被标为证据不足，不能用均匀特例单元测试替代；
- $b\log n$ 是长度校准项，让每个头自行学习长上下文中"集中读取"可信度的变化；
- 初始化 $w=a=b=0,\ d=2.0$，接近标准注意力，训练可稳定启动。

## 变体

`standard`、`gate_only`、`qgate`、`null`、`lega` 及五个消融（`lega_no_len`、`lega_no_conc`、`lega_no_query`、`lega_conc_only`、`lega_len_only`），覆盖全因子与去一法消融矩阵，并以 `gate_only` 隔离门控本身效应，详见 README。另有可选的训练期**参数匹配安慰剂** `lega_placebo`（门控参数/FLOPs 与 `lega` 相同，B/C 证据替换为确定性噪声；默认不进入 10 变体矩阵，用 `--param_control` 开启），用于排除“多几个门控参数的容量/优化红利”这一替代解释；其伪证据仍由绝对位置哈希得到，不能当作分布匹配的置换对照，正式结论需结合推理期 `conc_permute`/`head_permute`。补充训练对照 `lega_raw`、`lega_raw_no_len`、`lega_raw_conc_only`、`lega_raw_no_query`、`lega_detach_conc`、`null_zero` 默认不运行，分别检验原始集中度/无长度项、原始参数化全因子格、集中度 detach 反传、固定零 value null；它们不进入 `GATE_FACTORS` 的归一化 2^3 矩阵。

消融口径按组件严格定义：B 为 $a\,c$（$c=\log(n\max_j p_j)$ 始终保留长度归一化），C 为 $b\log n$；因此 Full−C（`lega_no_len`）只去掉 $b\log n$，不会连带删掉 $a\log n$（否则不再是单组件消融）。汇总同时以 `gate_only` 为无组件基准给出 A×B/A×C/B×C 交互检验，区分“主效应可加”与“协同/次可加”。

单因素审计：组件归因只允许使用每次仅改变一个因子的对比——去一法 Full−X vs Full，以及全因子增量 X-only vs `gate_only`，共 6 组。`B-only`（`lega_conc_only`）与 `C-only`（`lega_len_only`）相对 Full 同时改变两个因子（A+C / A+B），因此不再与 Full 配对；它们与 Full 的差异也不得写成单组件消融结论。因子表 `GATE_FACTORS` 直接从实现里的 `use_query/use_conc/use_len` 派生，运行期审计会核对这 6 组对比的因子差恰好为 1，并把结果写入 `significance.json` 的 `single_factor_audit` 与汇总页。同一审计还核对参数化稳健性使用的 4 条原始/归一化单因子边，以及推理期单病灶、活动匹配对照和 2^3 单因子格的因子映射；原始参数化变体不在 `GATE_FACTORS` 内，因此按 `NORMALIZED_CELLS`/`RAW_CELLS` 两份网格分别校验。各变体 vs `standard`、LEGA vs `qgate` 属于整体方法对比，可跨多个因子，不参与组件归因。

### 参数化稳健性：归一化 c vs 原始 log max_p

默认 B 因子是 $a\,c$，而 $c=\log(n\max_j p_j)$ 可展开为
$\log\max_j p_j+\log n$。因此 Full−B（`lega_no_conc` vs `lega`）同时
去掉了 $a\log\max_j p_j$ 与 $a\log n$，Full−C（`lega_no_len` vs `lega`）
只去掉显式的 $b\log n$；B 与 C 在参数化上并不正交。为把“集中度”与
“c 中的长度分量”分开，补充网格把 B 改成原始 $a\log\max_j p_j$：

| 格 | 归一化参数化 | 原始参数化 |
|---|---|---|
| none | `gate_only` | `gate_only` |
| A | `qgate` | `qgate` |
| B | `lega_conc_only`（$a(\log\max p+\log n)$） | `lega_raw_conc_only`（$a\log\max p$） |
| C | `lega_len_only` | `lega_len_only` |
| A+B | `lega_no_len` | `lega_raw_no_len` |
| A+C | `lega_no_conc` | `lega_no_conc` |
| B+C | `lega_no_query` | `lega_raw_no_query` |
| A+B+C | `lega` | `lega_raw` |

两种参数化属于同一线性门控函数族，差异是参数化/优化几何与解释，不是
表达能力。`--raw_factorial` 会追加原始网格缺的 4 个变体，与默认 10 变体
一起训练；两套网格必须在同一批种子上完整运行，且逐格评测协议一致，才进入
`lega/reparameterization.py` 的分析。分析对每条单因子边和每个 Shapley
分量计算同种子差异 $D=\text{raw}-\text{normalized}$，再做配对 t 检验
（同一指标内 Holm）：PPL 的效应先定向为 $-\log$ PPL（正值=候选降低 PPL
更多），RULER 用原始 EM 差。结果写入 `significance.json` 的
`reparameterization` 并在 `summary.md` 单列。

解释规则：

- $D$ 显著非零或方向翻转：因子归因**依赖参数化**，不能把归一化 c 的
  效应直接写成“纯集中度证据”的贡献；必须同时报告原始网格；
- $D$ 不显著：只能写“未检出参数化差异”，不能写“稳健等价”；需对照
  该检验的 MDE 与种子数；
- 原始网格缺格/协议不一致：`not_run`/`missing_variants` 显式标记，不得
  把缺失伪装成“无差异”或默认稳健。

该分析属探索性，不替代预注册的单因素确认检验；它的作用是给 B/C 归因
加上一个可证伪的参数化敏感性边界。

### 全因子 Shapley 分解（交互感知的补充归因）

单因素边依赖参照格：当 A×B/A×C/B×C 交互不为零时，“加入 X 的收益”在
`gate_only` 基准与 `Full` 基准下并不相同，只报一条边会掩盖这种依赖。
`lega/factorial.py` 对门控内 8 格做精确 Shapley 分解：

```
φ_i = Σ_{S ⊆ N\{i}} |S|!(n−|S|−1)!/n! · [v(S∪{i}) − v(S)]
```

其中 `N={A,B,C}`，`v(S)` 是只保留因子集合 S 时的性能。PPL 用
`v=−log(PPL)`（比值尺度上可加，`exp(−φ)` 即该因子的 PPL 几何均值比），
RULER 用原始 EM。性质与口径：

- 效率性 `φ_A+φ_B+φ_C = v(ABC)−v(none)`，总效应不会被漏算或重复计；
- 交互效应被平均分摊到参与因子，而不是归给某个参照格；
- 零玩家（所有联盟中都不改变 v）的 φ=0；
- 占比只在每个种子的总效应均为正时报告，否则留空，不强行归一化；
- 逐种子算 φ，再对 φ 做同种子单样本配对 t 检验 + 同一指标内 Holm；
- **探索性**：Shapley 依赖价值函数尺度且未预注册，不能替代预注册的
  单因素确认性检验；与单因素边方向冲突时并列披露“贡献依赖其他因子”。

### 训练预算敏感性（终点结论是否覆盖整个训练过程）

训练期消融只比较最终 checkpoint，但“某个组件有用”可能只是优化早期更快、
后段被其他通路追上，或早期有害、后期才转正。`lega/budget_sensitivity.py`
读取各 run 已落盘的 `train_log.jsonl`，在**同种子、同评测 step** 上对齐
候选与参照的 `val_loss`，逐预算计算配对差轨迹：

- 只取精确 step 交集，不插值；轨迹使用所有共同种子的共同 step，避免
  “不同预算对应不同种子组成”。缺日志/共同 step 时 `available=false`，
  不伪造一条全量 split 曲线；
- 输出初始/终点 Δ(logPPL)、逐 step 均值与区间、符号翻转次数、与终点
  符号一致率、最终方向首次稳定 step，以及 `stable` / `transient` /
  `reversal_nonsignificant` / `budget_sensitive` 轨迹模式；
- **探索性**：逐 step p 值高度相关且未做跨 step 校正，`budget_sensitive`
  只表示至少两个 step 以相反方向达到未校正 p<α，不能当作确认性结论；
- **不重新选 checkpoint**：轨迹只用于诊断“终点是否代表训练过程”，
  不能挑一个有利 step 改写主结果；终点仍使用预注册的 held-out 指标；
- val 同时是 checkpoint 选择集，故轨迹诊断不能替代 test 终点，也不能
  与跨种子终点检验互相替换；两者冲突时在汇总页并列披露。

## 输出级写回效用（不是 hit 代理）

`lega/output_ablation.py` 直接干预答案 token 的预测位置：把某层某头在
`wo` 输入中的通道置零/置换，重新计算同一批答案 token 的 NLL、正确概率与
margin，输出 ΔNLL/Δmargin。它回答“该头在该位置的写回是否真的帮助语言建模”，
而不是“注意力 argmax 是否落在答案跨度”。逐头 hit/AUROC 只能作为机制描述，
不能替代该干预；输出级干预仍是推理期反事实，不能证明训练期因果。

## 无证据/错误证据/冲突证据（可选条件）

`lega/ruler.py` 的 `EVIDENCE_TASKS` 提供三种 abstention 条件：
`evidence_absent`（目标键完全缺失）、`evidence_wrong`（只出现错键值）、
`evidence_conflict`（同一键两个不同值）。问题明确要求无可靠答案时输出
`NONE`，只有预测 `NONE` 才记 EM=1。它们默认不进入主表，需显式传入
`run_ruler(tasks=...)`；这是“门控在无可靠证据时是否减少写回”的直接检验，
不能拿普通 NIAH 的 hit 率代替。

## 推理期因果消融（训练后病灶）

训练期消融回答的是“**重新训练**时去掉组件会怎样”，它改变优化轨迹；
不能据此断言最终前向仍在**使用**该组件。`lega/ablate.py` 在同一个
checkpoint、同一批 val 窗口上只改门控参数，逐窗口配对计算
`ΔNLL = 病灶 − 正常`（>0 表示该机制在推理期有用）：

| 病灶 | 干预 | 回答的问题 |
|---|---|---|
| `no_query` | `gate_w := 0` | 推理期去 A（对照训练期 `lega_no_query`） |
| `no_conc` | `gate_a := 0` | 推理期去 B |
| `no_len` | `gate_b := 0` | 推理期去 C（只去 b·log n；`c` 里的 log n 仍随 `a·c` 保留，与训练期单因素定义一致） |
| `no_query_matched` | `gate_w := 0`，逐头平移 d 匹配基线 E[g] | 去 A，但保持平均门控开度；分离“信息内容”与“门控开度” |
| `no_conc_matched` | `gate_a := 0`，逐头平移 d 匹配基线 E[g] | 去 B 的活动匹配对照 |
| `no_len_matched` | `gate_b := 0`，逐头平移 d 匹配基线 E[g] | 去 C 的活动匹配对照 |
| `no_evidence` | `gate_w/a/b := 0` | 推理期退化为 `gate_only`（只留学习到的 d） |
| `query_only` | `gate_a/b := 0` | 推理期只留 A（`w·q + d`），补齐 2^3 的 A 格 |
| `conc_only` | `gate_w/b := 0` | 推理期只留 B（`a·c + d`），补齐 B 格 |
| `len_only` | `gate_w/a := 0` | 推理期只留 C（`b·log n + d`），补齐 C 格 |
| `gate_open` | 证据项 0，`d := +30` | 门恒开：门控是否在抑制某些写回 |
| `gate_closed` | 证据项 0，`d := −30` | 阳性对照：关闭注意力输出必须显著变差 |
| `gate_constant_mean` | 证据项 0，`d_h := logit(E[g_h])` | 去掉逐位置/逐查询变化，保留每个头/层的平均写入强度 |
| `head_permute` | 门参数在头间置换 | 容量/分布/结构不变，只破坏头—证据对应（安慰剂） |
| `conc_permute` | 逐位置 `c` 沿序列维循环移位 | c 的逐值分布/均值/方差/长度分布不变，只破坏位置对应；排除容量/尺度解释 |
| `sign_flip` | 证据项反号 | 保留幅度、反转证据方向 |

要点：

- **只改门控参数或前向临时属性**，不动注意力/主干；`conc_permute` 通过
  `conc_permute_shift` 临时属性实现，`restore_gate_params` 会同时清除它，
  保证病灶不叠加；
- 与基线使用同一 `ctx/eval_iters/AMP` 口径、同一 `window_indices`，否则拒绝配对；
- 默认只在 **val** 上做（test 尘封；val 是 checkpoint 选择集，绝对水平
  偏乐观，只作机制定位）；不写 test 访问账本；
- 记录 `no_op`：若训练后该系数已为 0，清零不改变前向——“无差异”不是
  “无效应”的证据；`head_permute` 在全同头时也如实标为 no-op；
- 窗口级 p 仅描述单 run（相邻窗口可能相关）；正式汇总先按种子聚合
  ΔNLL，再做跨种子配对 t + Holm（`significance.json.causal_ablation`）；
- **活动匹配病灶（消融混杂控制）**：直接清零 `gate_w/a/b` 同时改变
  “使用哪类信息”和“门平均开多大”。`no_*_matched` 在去掉同一组件后，用
  一次完整统计前向反推 `base_z = logit − d`，减去待去项得到去掉组件后的
  logit，再逐层逐头二分平移 `d'`，使 `mean(sigmoid(z + d'))` 等于该头
  基线 `mean(g)`。因此匹配版保持平均门控开度（写回强度的乘性因子），只去掉信息
  内容；它使用独立 Holm 家族
  `significance.json.causal_ablation` 中的 `因果消融ΔNLL(活动匹配−normal)`，
  不与原始病灶混合。跨种子配对差
  `ΔNLL(raw) − ΔNLL(matched)` 写入
  `significance.json.matched_ablation_contrasts`（独立家族
  `因果消融ΔNLL(未匹配−活动匹配)`）：差值显著为正，说明原始效应中有一部分
  可由平均门控开度变化解释；不显著不能反证“全部来自信息内容”，仍受种子
  数与 MDE 限制。匹配只控制一阶矩，不保持门控方差、饱和比例或跨层联合
  分布；匹配偏移又在同一批 val 窗口上估计（每头/层仅一个标量），属于事后
  均值匹配而非留出样本上的确认性控制，故不能写成“机制完全匹配”，也不能
  把差值当作精确的“信息效应 + 写回效应”或形式化中介分解；信息特异效应
  仍应直接阅读匹配病灶相对正常基线的独立检验；
- **长度边界**：训练/评测都受 `block_size` 限制（small=1024）；RULER 的
  256/512/1024 是域内扫掠，不是长度外推。超过 `block_size` 的位置必须拒绝，
  不能把其他模型的长上下文结果归因到本模型。
- **配对嵌套上下文**：当前 VT 已用中性填充精确补到请求长度并记录
  `evidence_distance_tokens/actual_tokens`，但尚未实现“同一证据在嵌套上下文
  中固定位置”的正式配对实验；长度趋势只能作为观测性证据，不能写成严格
  嵌套配对因果结论。
- **与训练期消融并列**：训练期显著而推理期病灶无差异 → 收益可能来自
  训练期优化/正则化而非推理期必需；推理期显著而训练期不显著 → 训练期
  消融可能被再训练补偿。两者不能互相替代。
- **边界（必须如实声明）**：`head_permute` 是**推理期**的容量/分布安慰剂，
  `conc_permute` 是 c 的分布匹配置换，两者都不是训练期的参数量匹配对照；
  训练期 `lega_placebo` 的伪证据仍含绝对位置哈希，不能替代置换零假设；`no_evidence` 在推理期保留全部参数、只关掉
  证据输入，能隔离“推理期是否使用证据”，但不能消除“训练期多出的门控参数
  是否带来优化红利”这一混淆。后者由下面的训练期 `lega_placebo` 对照回答；
  推理期 Shapley 只回答“最终前向的组件结构”，不能替代训练期因果问题。

### 同一 checkpoint 内的 2^3 全因子与 Shapley

`query_only/conc_only/len_only`、`no_query/no_conc/no_len`、`no_evidence`
与正常基线 `ABC` 共同构成**同一 checkpoint 内的 8 格干预设计**（`lega/ablate.py`
的 `INFERENCE_FACTORIAL_CELLS`）。它与训练期全因子消融的关键区别是：
不重新训练，只清零训练好的门控系数，因此没有“再训练补偿”或优化轨迹改变。
`run_all` 对齐全 8 格、且与基线共享同一批 `window_indices` 的种子计算
精确 Shapley 值（PPL 用 `−log` 价值函数），再跨种子配对 t + Holm
（`significance.json.inference_factorial_shapley`），并与训练期 Shapley
逐因子并列比较符号。三个单因子格（`query_only/conc_only/len_only`）相对
Full 同时去掉另外两个因子，因此**不进入**单病灶 ΔNLL 因果表，只用于
2^3 的全因子/Shapley 分解；否则会把“去掉 B+C”误写成“A 的因果贡献”。
这样可以把两种冲突显式化：

- 训练期 φ 显著、推理期 φ≈0：收益可能来自训练期优化/正则化，最终前向
  并未真正使用该组件；
- 训练期 φ≈0、推理期 φ 显著：训练期消融可能被再训练补偿；
- 两期符号相反：组件贡献依赖训练轨迹或参照格，不能只报有利的一期。

该分析仍是**探索性**的：干预的是已训练系数，不是重新拟合的反事实；
推理期 φ 依赖基线/病灶的零点定义，且未预注册。

**推理期交互效应**：Shapley 把交互平均分摊到参与因子，本身不告诉我们是
哪一对因子不可加。`run_all._inference_interaction_tests` 因此在同一组
8 格上做与训练期相同的标准 2^3 交互对比（对第三个因子取平均）：

```
I_AB = ½[(AB−A−B+none) + (ABC−AC−BC+C)]
I_AC = ½[(AC−A−C+none) + (ABC−AB−BC+B)]
I_BC = ½[(BC−B−C+none) + (ABC−AB−AC+A)]
```

数值用原始 PPL（与训练期 `PPL交互` 同尺度，便于逐交互对照符号），逐种子
计算后做单样本配对 t + 独立 Holm 家族，结果写入
`significance.json.inference_factorial_interactions` 与汇总页“推理期全因子
交互效应”表。训练期与推理期交互符号不一致时显式标注 ⚠：这说明“重新训练后
的交互结构”没有在最终前向复现，不能只报其中一期。该检验未预注册、无方向
主张；交互=0 表示可加，显著非零只说明不能按单组件加总，**不等于**某个
组件有用。缺格、未真正施加、窗口不一致或 PPL 非正的种子整条剔除，不按 0
补齐。

### 参数匹配安慰剂（训练期容量/优化红利对照）

`lega_placebo` 与 `lega` 的门控形状、参数个数、门控 FLOPs、初始化与
优化器分组完全相同：

```
lega        : g = sigmoid(w·q + a·c + b·log n + d)
lega_placebo: g = sigmoid(w·q + a·u + b·v + d)
```

其中 `u/v` 是由当前前向内的 token 位置与头索引确定性哈希得到的伪随机特征（取值
[-1,1]，不消耗全局 RNG，训练/推理/逐 token 解码一致），不含任何查询、
注意力集中度或长度信息。二者唯一的系统差别是 B/C 证据是否有信息，因此
`lega vs lega_placebo` 的 PPL 配对差用于回答：

- 若 LEGA 显著更低 PPL 且效应不在种子噪声内：B/C 证据的增量不能由
  “多几个门控参数/一次门控运算”解释；
- 若二者 TOST 等价：不能把 LEGA 的收益归因于 B/C 证据，容量/优化红利
  尚未被排除；
- 若 placebo 反而更好：B/C 证据项可能有害或训练不稳定。

同时报告 `lega_placebo vs qgate`：若噪声特征显著差于 qgate，说明对照本身
可能“有害”，此时 `lega > placebo` 不能单独证明 B/C 证据有用；若二者等价，
则额外容量本身无实质影响；若 placebo 反而优于 qgate，则容量/优化红利确实
存在，`lega vs placebo` 的差异必须在这个背景下解读。

运行方式：`python run_all.py ... --param_control`（或直接在 `--variants`
中点名 `lega_placebo`）。该变体**不进入**默认 10 变体矩阵、不进入
`GATE_FACTORS`/全因子 8 格，也不参与组件归因；汇总页在未运行时显式提示
“容量解释尚未排除”，不会默认已控制。该对照属探索性分析，未预注册，
不能替代 H2–H5。


命令：

```bash
python -m lega.ablate --ckpt out/small/lega/seed1337/model_best.pt \
    --data_dir data/wikitext103 --out results/small/ablation_lega_seed1337.json
# 全流水线默认对 lega,qgate 做（--ablate_variants 可加 gate_only 作对照）
python run_all.py --config configs/small.json --data_dir data/wikitext103 \
    --ablate_variants lega,qgate
# 追加训练期参数匹配安慰剂（lega vs lega_placebo 容量对照）
python run_all.py --config configs/small.json --data_dir data/wikitext103 \
    --param_control
# 追加原始集中度参数化的完整 2^3 网格（参数化稳健性）
python run_all.py --config configs/small.json --data_dir data/wikitext103 \
    --raw_factorial
```

输出 `ablation_<variant>[_seed<seed>].json`：基线/各病灶的 val_loss、PPL、
门控均值/标准差/饱和比例、逐窗口 ΔNLL、窗口级 t/符号检验、`no_op` 标记与
溯源字段；匹配病灶另含 `matched_control`（逐头 `gate_d` 偏移、目标/实际
平均 g、最大残差、使用位置数），便于审计“平均门控确实匹配”。`--skip_ablation`
可关闭；`standard`/`null` 无输出门，不能做该分析。

## 数据卫生：精确去重之后的近似重复

精确行/块去重只覆盖逐字节（以及 NFKC/空白归一化后）相同的复制。真实语料中
“改了几个词、删了一段”的近似复制更常见：它同样会让 train 记住 val/test 的
内容，也会在 train 内部放大同一段文本的权重。`lega/datacheck/` 的
`decontaminate_near_duplicates` 在精确去重之后补一层近似重复筛查：

- **表示**：NFKC + casefold + 空白折叠后取 UTF-8 字节 5-gram，集合去重；
  不依赖分词器，英文/CJK/符号走同一确定性路径；
- **候选**：18 个 MinHash 签名（6 band × 3 row）做 LSH；对阈值 0.80，至少一个
  band 相同的理论概率为 `1-(1-0.80³)^6≈98.7%`，漏检只发生在低相似度一侧；
- **核实**：候选对再精确计算 Jaccard 与 containment。跨 split 用
  `max(Jaccard, containment)`——val 的一小段被整段抄进更长的 train 文档时，
  Jaccard 会被长度稀释，containment 才能发现；**train 内部只用 Jaccard**，
  避免“短段被长文档包含”造成过度删除；
- **作用域**：只从 train 删除；val/test 逐字节不动。阈值 0.80 是分析前声明的
  协议参数（`scripts/prepare_data.py --near_dup_threshold/--near_dup_min_chars` 可显式设置），不是从当前数据估计出来的；报告同时落盘 0.70/0.80/0.90 三档
  **判定对数与真实删除量**（块/行/字符、占 train 字符比例），避免把“判定对
  数”误当成“删除规模”，也让阈值依赖在汇总页可见；
- **覆盖率与复核**：只有达到 `min_chars` 且 shingle 数达标的块会进入扫描。
  `data_report` 因此同时报告 `train_chars_coverage_fraction` 及被短块/少
  shingle/超大块跳过的字符数；覆盖率低于 80% 时告警，近重复结论只对已覆盖
  部分成立。清洗完成后 `post_cleanup_residual` 再独立核对最终 train 与
  val/test 是否仍共享精确/归一化长行或多行块（不改写任何 split），把“删过
  N 个”升级为“剩余为零”，`train_eval_clean=False` 会进入 alerts，并作为
  `post_cleanup_residual_overlap` 严重违规在训练前拦截；预检还会独立读取
  `duplicates.post_cleanup`，所以清空 alerts 不能绕过该结论；
- **证据位置**：`data_report.json` 的 `duplicates.near_duplicate` 与
  `duplicates.post_cleanup`，并生成
  `near_duplicate_decontamination_applied` / `eval_splits_near_duplicates` /
  `near_duplicate_scan_truncated` / `near_duplicate_low_coverage` /
  `post_cleanup_residual_overlap` 等告警；bin 缓存协议升至 v8，旧缓存必须
  重建，避免报告声称已做近重复处理而实际 token 流未处理。

### 规模核算与划分比例（让“80/10/10”可核对）

`data_report.accounting` 逐 split 记录 token 数、字符数、UTF-8 字节数与
`bytes/token`。PPL 的分母是 token；字符级兜底与 GPT-2 BPE 的 token 粒度
不同，token 级 PPL 不能直接跨分词器比较，但字节/字符分母可以。该核算也让
“清洗后 token 数正常但字节数异常”的分词事故可见。

时序划分把切点吸附到文档/段落边界后，实际比例可能偏离声明的 80/10/10。
`temporal_split_text` 同时返回目标/实际比例、切点相对目标偏移与
`max_abs_fraction_deviation`；绝对偏差超过 5 个百分点时生成
`split_fraction_skew` 告警。该告警不阻断（群组完整性优先），但读者必须按
实际训练/评测规模解读“80/10/10”与检验功效。

**局限**：LSH 是近似检索，不是精确全对比；containment 对模板化文本/固定
用语可能偏激进，因此只在跨 split 去污染时启用；三档敏感性只说明“删除量对
阈值有依赖”，不证明下游 PPL/RULER 对阈值不敏感。`post_cleanup_residual`
只覆盖精确/归一化行与块，不覆盖低于 `min_chars` 的短行或低相似度近似副本；
覆盖率字段给出该结论的适用边界。正式对比必须固定在同一预处理版本（同一
`data_report`/bin 哈希）上，不能按测试结果回头挑阈值。

## 评测

- WikiText-103 PPL（有 held-out test 时评 test，缺 test.bin 时回退验证集并提示选择偏差；窗口不重复、等间隔抽样）。默认只评 `eval_iters` 个抽样窗口，`eval.json` 因此必须落盘 `eval_iters`/`val_windows`/`full_eval`/`eval_window_fraction`，汇总对覆盖率 <1 的 PPL 单独告警；正式 PPL 结论应给 `run_all.py`/`lega.postrun` 加 `--full_eval`，评测全部不重叠窗口（checkpoint 选择仍只用 val）；
- **窗口边界相位敏感性**（`lega/evaluate.py --window_offsets N` 或 `run_all.py --window_offsets N`，默认关闭）：默认窗口网格固定在 offset=0，窗口边界落在哪里是数据侧任意选择。开启后在 `[0, ctx-1]` 取 N 个均匀相位起点（含首尾），用所有相位都能提供的**公共窗口预算**分别评测，`eval.json.window_offset_sensitivity` 落盘逐相位 `val_loss`/`val_ppl`、范围、离散度、最大相对 offset=0 的偏移与 `val_ppl_ratio`；主指标仍只取 offset=0，训练/选型与 n-gram 对齐口径不变。不同相位的窗口共享大量 token、不是独立重复，因此该字段只作“结论是否依赖窗口切分”的描述性稳健性诊断，不计算 p 值/置信区间，不能替代预注册的单起点或全量窗口分析；
- 简化 RULER 五任务（niah_synth / niah_corpus / niah_multikey / fwe / vt），多档上下文、贪心解码；
  每格 em 附带 Wilson 95% 区间（`em_ci95_wilson`）：n_samples 小时点估计的抽样误差不可忽略；
  niah_corpus 的干扰语句优先取 held-out 的 val/test.txt（train 仅在无 held-out
  文本时回退并显式告警），来源写入 payload/summary 的 `corpus_source`——评测
  上下文与训练语料重叠会把“检索”变成“回忆”，绝对分数不可比；
- 机制分析（`lega/analysis.py`）：misleading_rate、P(hit|集中) 对比、门控分数 AUROC 与 PR-AUC、门控决策点 F1（macro/micro/weighted + **恒写/恒不写稻草人对照** `gate_f1_trivial`——门控 F1 不高于无信息固定规则时没有决策价值）。
- 简单统计基线（`lega/ngram.py`）：与神经模型同 split 的一元/二元/**插值二元**平滑 n-gram PPL。插值模型 `P = λ·P_addα(w|prev) + (1−λ)·P_addα(w)` 让未见上下文回退到一元分布，而不是只回退到近似均匀；α 与 λ 只在 val 上按 mean NLL **联合**选优，`selected_model` 取 val 上最强的简单模型（λ=1 可退化为纯 bigram，因此最强基线不会低于它），正式报告只消费选定参数一次；test split 需显式 `--allow_test` 并写访问账本。流水线 `--ngram_split auto`（有非空 `test.bin` → test，否则 val）与神经评测同一规则，从源头避免 val 评测的模型对比 test 基线这类错位。

### RULER 上下文长度剂量-反应（只报跨长度平均会掩盖长度效应）

方法名与核心主张都是“随上下文长度校准门控”，但只报跨 `ctx_lens` 的
`RULER平均` 回答不了“优势是否随长度变化”：短上下文涨、长上下文跌，与
所有长度都小幅涨，可以得到同一个平均分，科学含义却相反。`lega/lengtheffect.py`
负责把 payload 整形为按上下文分层的 macro EM，`run_all._ruler_length_analysis`
做三层探索性检验并渲染进汇总页：

- **剂量轴**：以 payload 声明的 `ctx_lens` 为请求档位，而不是每类任务
  被 clamp 后的 `room_ctx`。不同任务因生成预算不同，同一请求档位可能落到
  不同实际值（如 niah 1011 vs fwe 979）；若把实际值当档位，会凭空多出
  “只含 fwe”和“只含 niah/vt”的两档，把任务组成变化误读成长度效应。
  模块要求每个任务恰好覆盖全部请求档位后按升序对齐；任务缺档/多档时
  拒绝该对比的分层分析（`ruler_length_pairs` 不含它），不猜映射。旧产物
  无 `ctx_lens` 时只在所有任务共享同一实际档位集合时回退。
- **逐 ctx**：每个种子在每个公共请求档位上先算任务等权 macro EM，再
  对同一档做同种子配对 `Δ=候选−参照`（t + 符号翻转置换 + 配对差
  bootstrap CI + MDE）；所有对比的同一档位检验合并为一个 Holm 家族
  （`RULER@ctx`，保守）。逐档表回答“哪一档在驱动结论”，不把跨档平均
  当成每一档都成立。
- **斜率**：每个种子独立对 `ΔEM ~ log2(ctx)` 拟合 OLS 斜率（单位
  `pp/2×`，长度每翻一倍的变化），再跨种子对斜率做单样本配对检验
  （独立 Holm 家族 `RULER@length_slope`）。先合并各档点估计再拟合会把
  种子噪声当成确定性趋势，因此坚持逐种子拟合、再做配对。
- **首末落差**：`ΔEM(最长) − ΔEM(最短)`，不假定线性（独立家族
  `RULER@length_span`）。斜率显著而首末落差≈0、或二者方向不一致时，效应
  可能集中在中间档，不能写成“越长越好/越差”。
- **留一上下文**：只有 2~3 档 ctx 时，斜率可能由某一档的偶然波动驱动。
  逐种子在删掉任一单个 ctx 后重新拟合斜率，再看跨种子平均斜率符号是否
  翻转（`slope_loo_sign_stable`）；翻转则标“趋势依赖个别档位”，与留一
  法检查种子的逻辑同构。
- **平衡性**：分析只在“每个任务恰好覆盖全部请求档位”的完整网格上进行；
  此时逐档等权平均严格等于既有 `RULER平均`。运行期仍核对总体 macro 与
  逐档平均是否一致（`balanced_contexts`），不一致说明对齐/数据异常，
  分层均值只作描述并在汇总页提示，不冒充总体均值的严格分解。
- **方向翻转与平均掩盖**：总平均与最长 ctx、斜率或首末落差符号相反时写
  `overall_vs_longest/slope/span_sign_flip=true`；总平均落在 ±1pp SESOI
  内、最长档却超出 SESOI 时写 `long_context_masking=true`（零与负号不构成
  严格翻转，但同样说明总平均不代表长上下文）。汇总页对两者都标
  “总平均与最长ctx反向 / 平均掩盖长度效应”。此时只报总平均会把
  “长上下文失效”掩盖成“平均无差异/小涨”，是长度主张最需要防止的读法。
- **边界**：全部为探索性、未预注册，不改变 H1–H7 的确认性判定；逐 ctx
  检验只覆盖训练种子随机性，每格 20~30 个样例的评测抽样误差仍由
  `ruler_*.json` 的 Wilson CI 单独报告；`n<3` 只报描述、不判定；只有 1 个
  公共上下文档位时明确写“无法检验长度趋势”，不伪造斜率。
- **产物**：`summary.md` 的“RULER 上下文长度剂量-反应”小节；
  `summary.json`/`summary.csv` 逐变体增加 `ruler_length_*` 字段（vs
  `standard`：总体 Δ、最长档 Δ、长−短、斜率、p_adj、方向翻转/掩盖标记、
  平衡性），便于程序扫描；`significance.json` 的 `ruler_length_pairs`
  （逐对逐 ctx 的候选/参照均值、Δ、CI、p/p_adj、逐种子斜率与首末落差）与
  `ruler_length_tests`（`per_context` / `slope` / `long_minus_short`
  三族完整检验）。
- **输入卫生**：非规范或重复的 ctx 键（如 `"0512"` 与 `"512"` 同时出现）、
  越界 EM（>1 或 <0）、非法/非正 `n` 都按缺失或损坏处理；缺 `n` 的旧产物
  保持兼容。空 payload 与“每档都是 0 分”由 `{}` 与全零值区分，不伪造
  趋势。

## 基线与指标口径（先测最简单的，再谈复杂模型）

- **简单基线不可省**：没有无参数基线时，“PPL=xx”无法区分“数据容易”与“模型好”。n-gram 基线与神经变体用同一 train 拟合、同一 held-out split 评测，给出绝对难度参照。
- **与简单基线的差异也要检验**：n-gram 是确定性基线（无种子方差），逐种子神经 PPL 与同一常数做**单样本配对检验**，按变体家族 Holm 校正，报告比值/相对降低/p_adj/判定（`ngram_baseline_comparison` → `significance.json.ngram_comparison`）。split 不一致的变体单列“未纳入比较”，绝不混表。**同 split 还不够、必须同 token**：神经 PPL 只在 `eval_iters` 个等间隔窗口的目标 token 上计算，n-gram 因此额外报告逐位置对齐口径 `ppl_window_matched`（`lega.ngram.window_positions` 与 `train._val_eval_indices` 同规则，评分位置为每个窗口的 `s·ctx+1..s·ctx+ctx`）；仅同 token 还不够——神经 checkpoint 由窗口 val loss 选择，n-gram 若仍用完整 val 选 α/λ，选参 token 集依旧不同；新产物另在**同一批窗口目标 token** 上独立选参（`window_matched_selection` / `ppl_window_matched_selected`），汇总优先用这套参考，旧产物才回退 `ppl_window_matched`。汇总在全部纳入比较的 run 的 `eval_ctx/eval_iters/val_windows` 与 `window_spec` 一致时优先使用窗口参考，任一不一致即整体回退全 split 并写入 `fallback_reason`（同一 Holm 家族共用同一参考值）。该对比未预注册，标为补充性/探索性证据，不替代变体间受控配对检验。**逐窗口配对**：n-gram 侧同时落盘选定模型的 `window_losses_matched` 与神经评测同规则的 `window_indices_matched`；汇总在逐位置对齐成立时，对同一窗口序号做 `loss_neural − loss_ngram`，并用“先重采样种子、再重采样窗口”的分层 bootstrap 报告 ΔNLL 区间/PPL 比。窗口序号不一致或 n-gram 缺字段时显式跳过并写明原因，不按位置强配。
- **基线必须调过，调优选择本身也要防噪声**：新方法的涨点可能只是超参数红利。`lega/tuning.py` 对基线（standard）做 lr × weight-decay 小网格（默认 lr×{0.5,1,2}），只用验证集选择（`train_log.jsonl` 全评测点最小值，与 `model_best.pt` 同口径；test 从不加载）；`default_is_optimal` 在默认与网格最优的相对差 ≤0.5% 时为真。单一种子上的网格最大值可能是“幸运种子”选出的假最优（winner's curse）：`--tune_seeds` 让每个组合在多个种子上重复训练、按**平均 val_loss** 选择，报告 `selection_seeds` 与逐种子明细；不指定时只用单一种子（汇总页显式警告），正式实验建议 ≥3 个。`run_all --tune_baseline` 把最优组合注入**全部**变体（受控变量仅整体平移），证据写入 `baseline_tuning.json` 并在汇总页渲染；缺证据时汇总页显式警告“未调优基线”。**调优范围必须披露**：该流程只对 `standard` 做网格；`qgate`/`null` 等“已有工作代表”继承同一组 lr/weight_decay，未各自调优，因此“优于已有工作”只能按**同超参受控对照**解读，不能写成“优于各自最佳调优版本”；差距接近 MDE/种子噪声时更不足以支持优越性结论。
- **机制判别也要有最简监督基线（线性探针）**：门控 logit 是语言建模的副产品、训练时从未见过 hit 标签；“门控 AUROC/PR-AUC 高”不能排除“`log max_p`、`log n`、`qterm` 这些廉价特征本身线性可分，复杂门控结构没有贡献”。`lega/probe.py` 在同一批记录上训练 L2 逻辑回归，并对简单基线同样做**嵌套、按样例分组、按 hit 标签分层**的交叉验证：同一 (层,头,查询位) 记录不独立，按行切分会“背题”；稀有正类可能让某些折没有正类，分层分组降低空正类折造成的基线低估。λ 与决策阈值只在每个外层训练折的内层 CV 上选择（λ 按与主检验对齐的 PR-AUC/AUROC，阈值按内层 OOF macro-F1），最终 OOF 不再参与选参，避免选择偏差；标准化统计只用训练折拟合。报告 OOF 的 AUROC/PR-AUC、Brier/ECE、以及固定 0.5 与调优阈值两套 F1（macro/micro/weighted 三口径）。`run_all.linear_probe_baseline_comparison` 把逐种子 Δ=探针−门控 做配对检验 + Holm 校正并渲染进基线小节：探针超过门控属预期（判别能力本可由线性模型学得），门控不能在主指标上被探针显著超越，才有资格谈“结构优越性”；单统计量规则（max_p）一并列出作最低档次对照；旧产物缺探针时显式警告而不是当作已有对照。该对比与 n-gram 对比一样属补充性/探索性证据，不替代预注册的确认性检验。
- **不平衡指标必须并报且注明口径**：hit/¬hit 判别是类别极不平衡的二分类，`hit_base_rate`（正类率）即 PR-AUC 的随机基线。正类率 <5% 时 AUROC 因负样本基数巨大、FPR 增长缓慢而乐观，**以 PR-AUC（average precision）为主指标**（<0.1% 尤甚）；正类率 ≥5% 时两者并报。F1 一律注明平均口径（`gate_f1` 同时给 macro/micro/weighted 与逐类明细；macro 对稀有类最诚实，汇总表列 macro）。**任何学习指标都要有无信息对照**：门控 F1 并报恒写/恒不写固定规则的 `gate_f1_trivial`，打不过固定规则就谈不上“门控学会判断”。RULER 的 EM 为生成式指标、无决策阈值，不受此影响（fwe 另报召回率、精确率与 F1 作诊断）。**排序好不等于概率可信**：门控 g 是 sigmoid 输出且被当作软权重/概率使用，必须另报 Brier 与 ECE（与常数基率 Brier 基线对照；`analysis_*.json` 的 `calibration_g`），AUROC/PR-AUC 不能替代校准证据。ECE 同时给等宽与等频两套箱：等宽箱在分数聚集/小样本时可能大量为空，单个样本独占一箱还会把 ECE 机械压低；等频箱按分数排序均分样本，作为对分箱选择不敏感的第二组证据。Brier 另做 Murphy 分解：等频箱内用箱均值预测时 `Brier_bin = reliability − resolution + uncertainty`；原始分数在箱内不恒定时还须加箱内项 `brier_binning = within_score − 2·within_cov`（`brier = reliability − resolution + uncertainty + brier_binning`），否则三项之和并不等于原始 Brier。该分解把“箱内校准误差”“箱间区分度”“标签固有不确定性”与“分箱引入的箱内方差/协方差”分开；这些仍只是描述性证据，不替代正式检验。RULER 平均同时报 **macro**（任务等权、任务内上下文档位等权，预注册口径）与 **micro**（按每格样本数加权），避免“平均”未注明定义而掩盖任务/格大小不均。两者统一用各任务 **EM**（fwe 为严格集合 EM，与 H4 的“RULER 平均 EM”一致）；fwe 的 recall 仅单列诊断、不进入平均——recall 不惩罚“把上下文词全部列出”的灌水策略；precision/F1 一并单列，才能区分“没找到高频词”与“找对但灌水/漏答”。

训练预算轨迹作为补充证据写入 `significance.json` 的
`budget_sensitivity`，并在汇总页“训练预算敏感性”表中列出发生方向翻转/
预算敏感的对比；它不进入确认性判定，也不改变任何预注册假设的状态。

## 统计推断协议（小样本科研口径）

**两种不确定性分开报告**：跨种子检验只看每个种子的聚合指标（训练随机性），
但 RULER 每格只有 20~30 个样例、PPL 只评测有限个窗口，评测抽样误差不会
自动进入种子级 t 检验。因此：

- RULER 结果逐单元格落盘 `samples`（`sample_id` 跨变体稳定对齐），汇总时对
  同一批题面做**样例级配对 McNemar 精确检验**（只消费 discordant 对：候选
  独对 vs 参照独对），并给出逐种子 discordant 明细；多种子按计数合并只作
  探索性证据。**样例聚类稳健性**：同一样例在多个训练种子下重复出现，池化
  McNemar 会把它当成多条独立证据；因此另按 `sample_id` 先跨种子平均配对差，
  再以样例为独立单位做符号翻转检验（`cluster_*`，独立 Holm 家族），与池化
  结果并列报告。
- RULER 同时报告样例等权 micro 与单元格等权 macro 效应量：McNemar 的
  discordant 计数对应 micro，预注册 `RULER平均` 是 macro，两者方向不一致
  时结论依赖“平均值”定义，必须并列展示。macro 的抽样区间固定已观测种子，
  在 (任务, 上下文) 单元格内对配对样例重采样、再保持单元格等权平均
  （`_ruler_macro_bootstrap_ci`）；它只覆盖评测样例抽样，不覆盖训练种子
  随机性，不能替代种子级检验。
- 每个变体的 **RULER 绝对 macro EM** 也报分层 bootstrap 95% 区间
  （`_ruler_macro_absolute_ci`）：在每个单元格内对样例重采样，按与
  `ruler_avg` 相同的“任务等权、任务内上下文档位等权”顺序聚合，再对观测
  种子取平均；同一结果写入汇总表 `RULER评估CI` 与 summary JSON 的
  `ruler_avg_ci95`。它回答“这个绝对水平在评测抽样下有多不确定”，与
  “候选−参照差”的区间及跨种子 Mean ± Std 三者不可互相替代；旧产物缺
  逐样例记录时如实留空。
- 独立评测落盘 `window_losses`/`window_indices`，汇总时对同序号窗口逐点
  相减，并用“先重采样种子、再重采样窗口”的分层 bootstrap 给出平均 NLL 差
  区间与 PPL 比。窗口按时间顺序排列时相邻窗口可能相关，i.i.d. 重采样会
  低估区间，因此同时报告移动块 bootstrap（块长 `n^(1/3)`，保留块内相邻
  相关）与 lag-1 自相关；以更宽区间解读稳健性，两者都仍是近似。每个种子
  不足 8 个窗口或少于 2 个种子时只报点估计/区间，不生成“显著”措辞——
  小窗口重采样分布的自由度不足以支持显著性声明。
- **种子与评测单元的方差分解**（`lega/uncertainty.py`）：把同一
  (种子, 评测单元) 单元格内的配对差视为 `seed × unit` 二维矩阵，用两因素
  随机效应 ANOVA 矩估计把方差分为种子、评测单元与残差三部分
  （`two_way_variance_components`），并按预注册 macro 权重聚合多单元格
  （`aggregate_variance_components`）。同时报告**总方差占比**与**均值方差
  占比**：前者描述单次观测的波动，后者已按 `1/S`、`1/U` 收缩，才是“补训练
  种子还是补评测窗口/样例更有效”的依据；汇总页的“不确定性来源分解”表把
  PPL 窗口与 RULER macro 并列，并在种子/单元均值方差占比 ≥65% 时给出
  探索性资源分配提示。RULER 另增**双层 bootstrap**：先重采样训练种子，再在
  被选中种子的每个 (任务, 上下文) 单元格内重采样配对样例，输出
  `macro_mean_diff_ci95_hier` / `macro_p_hier` / `macro_p_hier_adj` /
  `macro_bootstrap_same_sign_hier`，补上“固定种子区间”与“种子级 t 检验”
  各自缺失的合并不确定性（独立 Holm 家族，仍为探索性）；PPL 窗口
  检验也在 `variance_components` 中报告同一分解，且只在各窗口序号对齐时计算。
  **边界**：方差分量是矩估计而非最大似然，小样本下可能为负（按 0 截断并
  标记 `negative_*_component`）；PPL 窗口按时间顺序排列、相邻窗口可能相关，
  把窗口当独立单元的方差分量只是近似（区间稳健性仍看移动块 bootstrap）；
  不平衡/缺格/`sample_id` 未对齐时不插补、不按位置强配，只使用所有种子
  共有的单元格；65% 阈值只是资源分配提示，不参与确认性判定，也不改变
  预注册的种子级 Holm 结论。
- 两种证据与种子级检验回答不同问题（固定题面/窗口上的差异 vs 换训练种子
  的稳定性），冲突时在汇总页并列展示，任何一方都不单独升级为确认性结论；
  确认性判定仍只走预注册的种子级 Holm 校正检验。

正式实验的种子数只有 3~5，任何单点统计声明都站不住。协议分四层：

1. **主判定**：同种子配对 t 检验，同一指标家族内 Holm–Bonferroni 校正
   （控制家族 FWER），配对 Cohen's d_z 效应量；n<3 只报描述统计。
   **确认族整体校正**：预注册的 H1–H7 跨 PPL/RULER/AUROC/AP 多个指标
   家族，“每个家族各自 5%”并不控制全部确认性结论的整体 FWER；汇总页
   因此再对全部可检验预注册假设（n≥3）的**原始 p** 做一次 Holm 校正
   （`apply_global_fwer`）。判定要求家族内 p_adj 与确认族 p_adj 均 < α；
   仅家族内显著者降级为“未定”，同时保留 `status_family` 供审计。
   该门只收紧、不放松判定，且不改变既有假设的科学内容。
2. **稳健性印证**（每组配对同时报告，`significance.json`）：
   - **符号翻转置换检验** p_perm（分布无关；n≤16 全枚举精确）：t 检验在
     小样本下依赖正态假设，两套结论冲突时在结论文本中显式标 ⚠ 而非静默
     采信任何一方。**粒度边界必须知情**：置换双侧 p 的理论下限是 2/2^n
     （n=3→0.25、n=5→0.0625、n=6→0.03125），种子 <6 时"置换不显著"是
     粒度所限，此时 t 的显著性完全依赖正态假设；n≥6 后不一致才是真实警报；
   - **中位数配对差与精确符号检验** p_sign（独立 Holm 家族）：均值/标准差
     会被单个离群种子主导；Δ中位数与只数方向的符号检验提供对离群值稳健的
     对照。零差按经典口径从符号检验中剔除（nz=正/负配对数），nz=0 时 p=1；
     其双侧精确 p 下限同样是 2/2^nz（nz<6 时 >0.05），只作稳健性对照，
     不参与确认性判定；
   - **配对差 bootstrap 95% CI**：效应量的不确定范围，防止把点估计当精确值；
   - **PPL 比值尺度敏感性**（探索性，`PPL(log)` 独立家族）：预注册主指标
     仍是原始 PPL 的配对差；同时报告 log-PPL 配对差与几何均值比
     exp(ΔlogPPL)（及其 bootstrap 区间）。算术 PPL 差会被高 PPL 种子主导，
     两种尺度方向不一致时汇总页显式标为“尺度结论不一致”，不得挑选对主张
     有利的口径；
   - **MDE（最小可检测效应）**：(z_{1-α/2}+z_{power})·std/√n。"不显著"
     必须对照 MDE 解读：观测差远小于 MDE → 功效不足（先补种子），而不是
     "证明了没有效应"；
    - **效应/噪声比**：|Δ|/ΔSD（ΔSD 为逐种子配对差的标准差）。|Δ|<ΔSD
      表示任一种子的符号都可能翻转，结论中显式标注“种子/评测噪声内”的
      边缘证据；即使 p_adj 显著也不得写成稳健收益。
    - **逐种子胜负与留一方向稳定性**（`n_seed_wins/losses/ties`、
      `loo_sign_stable`）：n 只有 3~5 时一个“幸运/倒霉种子”就可能决定均值
      符号。逐种子配对差符号计数不含任何分布假设；留一法检查删掉任一单个
      种子后均值符号是否翻转，翻转即在结论中标注 ⚠（结论由个别种子决定），
      即使 p_adj 显著也不算稳健。`component_evidence` 据此把“显著但
      |Δ|<ΔSD 或留一翻转”的组件证据标为 `fragile`：最多算探索性线索，
      `confirmatory_supported` 必须为假，不能据此宣称“各组件均有用”。
    - **RULER 逐任务分解**（探索性，`RULER:<task>` 指标家族）：综合平均会
      把方向相反的任务效应互相抵消，故对预注册 RULER 对比逐任务各做一次
      同种子配对检验；每个任务一个独立 Holm 家族，仅作定位/机制线索，
      不替代预注册的“RULER 平均”确认性判定。

3. **机制主张的正式检验（两种指标口径）**：AUROC(完整 logit) − AUROC(仅查询项)
   （H6）与 PR-AUC(完整 logit) − PR-AUC(仅查询项)（H7，不平衡主指标口径）
   各作为独立指标家族做同种子配对检验（`run_all._auroc_diff_tests` /
   `_ap_diff_tests`）——两个分数指标来自同一批记录，并排放数字不构成证据。
   与逐 run 的**样例聚类 bootstrap** 区间（`analysis.auroc_cluster_bootstrap`，
   同时给 AUROC 与 AP 两族及各自差值）互为印证：后者以样例为重采样单元
   （同一样例的层×头×查询位记录共享难度，不是独立观测），回答"换样例波动
   多少"；前者回答"换训练种子方向是否稳定"。
4. **阈值敏感性**：misleading/P(hit|集中) 系指标在多档 τ 上并报
   （`analysis.tau_sensitivity`，默认 [0.3, 0.5, 0.7] 并入主 τ），
   机制结论不得只挂在单一任意选定的阈值上。

## 预注册（confirmatory vs exploratory）

登记表与锁文件位于 `preregistration/`（`hypotheses.json` + `hypotheses.lock.json`）——
**刻意不放 `docs/`**：流水线自动清理会删除 docs/ 下除 `design_notes.md` 外的
全部文件。当前登记 H1–H7（H7 为 PR-AUC 口径的机制主张，与 H6 的 AUROC 口径互为印证）。

`run_all` 支持在正式实验前登记方向性主张，汇总时逐条对照
Holm 校正后的配对检验判定 **支持 / 拒绝（方向相反且显著=被证伪）/ 未定**
（未定≠无效应，须对照 MDE 解读），结果进 `summary.md` 与
`significance.json`。未登记的对比一律属事后（exploratory）分析：其 p 值
只受所在家族内校正保护，"显著"只能作为提出新假设的线索；要升级为结论
须先登记再在新的种子/数据上检验——把"哪些对比是事先声称的"从记忆变成
可执行的边界。

**完整性锁链（防事后静默改写/删除）**：登记表配套锁文件记录锁基线（整文档
SHA-256 + **逐假设科学内容摘要**，只覆盖 id/claim/metric/candidate/reference/
direction，注释类字段不参与）与**修订链**。既有假设只增不改不删；新增假设
必须经 `python -m lega.hypotheses --amend --reason ...` 留痕，否则判违反。
`verify_registration` 的状态为 intact/amended（可确认）与 violated/missing_lock/
missing_registry（不可确认）；run_all 把登记表指纹与完整性报告写入 `repro.json`、
实验台账与 `significance.json`，汇总页显式渲染；**violated 时全部确认性判定
自动降级为探索性**（`apply_integrity_gate`），不假装边界仍成立。局限如实声明：
本机制防御的是静默/意外改写，无法对抗同时改写登记表、锁文件与 git 历史的
攻击者——那需要外部审计。

**范围核对（锁链完整 ≠ 实验落在范围内）**：登记表声明 `scope`（正式配置的
模型规模、数据/分词器、种子计划）。汇总时 `registration_scope_check` 把它与
本次实跑的 `repro.json`（`config_file` 的规模字段、数据指纹的 tokenizer、种子）
逐项比对：tiny/冒烟/伪数据/非同登记分词器都判范围外，`apply_scope_gate` 与
完整性门串联，确认性判定同样降级并写明原因；无法核对（缺 repro/config）也不
默认通过。调优按协议注入的 lr/weight_decay 改变不算越界（本来是 val 选择）；
未登记种子只警告不降级。登记表未声明 `scope` 的历史登记表则只警告，不因
“没有可核对字段”本身推翻已通过的完整性判定。

## 可复现性与 sanity check

- **种子**：`set_seed` 覆盖 Python/NumPy/PyTorch（CPU/CUDA）、cuDNN 确定性算法与子进程 `PYTHONHASHSEED`；批采样由显式 `np.random.default_rng(seed)` 驱动（无 DataLoader worker）。`run_all` 逐命令把 `PYTHONHASHSEED` 对齐到该命令自己的 `--seed`，基线调优的每个 trial 子进程也按 trial seed 设置该变量，`run_manifest.json` 内嵌 `seed_report`（逐随机源的种子、**RNG 真实状态短指纹**、确定性开关、数据顺序机制）供事后审计：指纹可被第三方用 `set_seed(seed)+rng_state_fingerprint()` 复算，当前状态不属于所请求 seed 时为 `null`；`PYTHONHASHSEED` 分“解释器启动值/子进程 env/是否真的对当前进程生效”三字段记录，避免字段被 set_seed 自身写值而自证。确定性算法如实标注严格/`warn_only`；环境快照记录 TF32（matmul/cudnn）、线程数与 `CUDA_VISIBLE_DEVICES`，训练清单在 `enable_tf32` 之后生成以记录实际生效值。跨硬件架构不保证逐位一致，同机重跑稳定。
- **过拟合小批量**（`lega/sanity.py`）：训练前在固定小批量、关闭正则化与数据扰动的条件下检查损失能否趋近 0、下一 token 准确率能否接近 100%；首步还会做**逐参数梯度覆盖**（missing/zero/nonfinite），可定位“总梯度范数正常但门控等子模块没接进计算图”的断流问题，并强制要求 dropout/attn_dropout 已关闭（未关闭直接失败，避免配置错误被误判为代码错误），同时自动核对三条中间结果：训练路径 loss（lm_head 分块 + 交叉熵）与直接完整 logits 的 `F.cross_entropy` 数值一致（捕捉分块/归一化/ignore 处理错误）、消费点标签满足 `y = x 右移一位`（捕捉错位/换序）、初始 loss ≈ log(V)（默认容差 1.0 nat，负值禁用，捕捉初始化/logits 尺度病态）。不能过拟合或核对失败说明梯度/损失/标签链路存在严重问题，`run_all` 会直接中止全量训练并向台账写 `aborted`；通过时 sanity 摘要（路径/SHA-256/判定/身份）绑定进该 run 的 `run_manifest.json`；绑定前核对 variant/config/data/git（含 dirty diff）身份，旧配置/旧数据/旧代码的 PASS 不会被当作本 run 的通过证据。通过是必要非充分条件，需继续配合中间结果检查。
- **追踪**：训练写 `run_manifest.json`（git commit/dirty、配置 SHA-256、数据指纹、种子审计、环境、命令行、sanity 证据）与 `config_used.json`/`train_log.jsonl`；dirty 工作区额外记录 diff 指纹/文件清单并保存 `worktree.patch`（已跟踪 diff + 未跟踪文件，上限 8 MiB、截断显式标记；仅凭 commit 无法复原代码时可重放）。`run_all` 写全局 `repro.json`，并向固定路径 `results/experiments.jsonl` 追加实验台账（started/completed/aborted 生命周期，记录解析后的完整超参数与逐 run `config_used.json` 内容，config 文件被改写/删除后仍可还原；逐 run 最终指标与曲线/被评测 checkpoint/结果文件 SHA-256；`eval_*.json` 内嵌 `ckpt_sha256` 绑定权重），失败轮次带阶段/原因/退出码，重跑不丢历史，评测 JSON 内嵌同一溯源字段。台账快照还内嵌逐 run 的 `seed_report`/环境/sanity 身份，manifest 丢失后仍可自证；`python -m lega.experiment --check_artifacts` 可核验 started/completed 链、重复终态、悬挂 run 与产物哈希。本轮 `run_id` 在训练前写入 `repro.json`，并经 `LEGA_PIPELINE_RUN_ID` 传给调优/训练/评测/汇总；结果文件内嵌同一 ID，汇总只纳入同 `run_id`（及数据指纹一致）的产物，避免复用 `results/<tag>` 时把 `--skip_*` 留下的旧文件混入本轮统计。训练清单与评测结果另记录**源码树指纹**（`lega/`、`scripts/`、`configs/`、`preregistration/`、`run_all.py`、requirements 的内容哈希，忽略 `__pycache__`/`.pyc`），在无 git 的源码包中仍可复算；评测源码树与训练清单不一致时汇总显式告警。JSON/CSV/Markdown 报告统一以 fsync + `os.replace` 原子落盘，崩溃不会留下半截结果。
- **环境锁与单 run 核验**：`requirements-lock.txt` 固定正式实验的直接运行时依赖（`torch==2.14.1`、`numpy==2.5.4`、`tiktoken==0.11.0`、`matplotlib==3.11.2`）与 Python 主次版本；`requirements.txt` 保留开发用下限，`requirements-ci.txt` 另固定测试工具；lock 与 CI 固定版本由测试交叉核对。`python -m lega.repro --check-env` 对照 lock 与实际 `importlib.metadata` 版本，忽略 `+cpu` 等本地构建后缀；`run_manifest.json`、`repro.json`、实验台账与下游 `load_provenance` 都记录 lock SHA-256、Python/包匹配结果及训练/评测两端环境锁。`python -m lega.repro --verify-run out/<tag>/<variant>/seed<seed> --data_dir data` 逐项核验 completion、`config_used.json`、artifacts 哈希/字节数、checkpoint、seed 状态指纹、sanity 身份、数据指纹、源码树与环境锁；显式矛盾/哈希不匹配是 issue，旧清单缺字段只告警，`--strict` 把告警也当失败。lock 只覆盖直接运行时依赖，不证明 OS/驱动/传递依赖完全一致；源码树不一致对历史 run 是预期告警，重放前需 checkout 对应版本。

## 结果分析协议

- **训练曲线而非最终值**（`lega/curves.py` → `curve_diagnosis.json`）：训练降/验证升=过拟合（正则化/增数据/早停），**训练停滞而验证回升同样判过拟合**（平台期泛化先于训练损失恶化，不能误判成欠拟合并建议续训/扩容），**训练仍在实质下降而验证无实质变化=过拟合趋势**（验证不再随训练改善的早期信号：按正则化/早停处理，不据此续训；验证仍同步下降才是正常收敛），两者都不降=欠拟合或数据管道/学习率问题，NaN=训练失效；另独立报告 **pre-clip 梯度范数尖峰/非有限**（最大/中位数 ≥10×，NaN/发散的早期信号，可能先于损失曲线异常出现；训练日志按 log 区间记录 `grad_norm_max` 与发生步，避免只采区间末步而漏掉中段爆炸，并给出尖峰 step 清单）与**瞬态损失尖峰**（单步损失 > 邻近 5 点中位数 8×：分段均值会把它平均掉，常对应异常批次或一次梯度爆炸；同时记录区间 `loss_max`/发生步，防止一步异常被均值稀释），并对损坏日志行跳过计数。末段趋势另比较后 1/3 vs 中 1/3，避免“早先大幅改善、末段才平台/回升”被全程首末均值掩盖。训练留档图（`fig_train_curves.png`）除训练 loss 与 val PPL 两个面板外，还有训练(实线)/验证(虚线)损失同轴对照面板，泛化差距与过拟合形态一眼可见。诊断不是事后才能看：每次评测记录 `val_vs_best_rel`（当前 val 与历史最优的差距），连续 3 次无实质改善且未开早停时训练中打印过一次拟合/平台告警（`live_overfit_hint`，改善后计数归零、下轮连击会再次提示），避免“训练 loss 仍在降”的过程信号被忽略。
- **早停是实际机制而非仅建议**（`train.py` 的 `early_stop_update` + `--early_stop_patience`/配置 `early_stop_min_delta`）：验证连续 patience 次评测无实质改善（相对阈值，不随损失尺度漂移）即终止。默认关闭（0）保证受控对比不受影响；启用时 `eval.json` 同时记录计划/实际步数与 `early_stopped`，train_log 写入 `early_stop` 终标记，曲线诊断据此把截断标注为设计行为（不再建议"增加步数"或"延长训练"），受控变量审计改为按计划步数核对并把逐 run 实际步数如实列出——不让早停把"实际步数不同"伪装成"受控条件一致"。模型选择口径不变：始终用 val 选出的 `model_best.pt`。审计同时核对实际评测协议：PPL 的 `split/eval_ctx/eval_iters/val_windows/val_tokens_evaluated` 必须同批 token，RULER 的 `ctx_lens/n_samples/sample_seed/corpus_source/max_new` 与已评测 (任务, 上下文) 格集合必须一致；配对检验还会逐种子整对剔除协议/格不一致的观测，避免独立重评或不同题面混入“同种子配对”。确认性判定由 `apply_budget_gate` 降级；`component_evidence` 的每条证据也带 `training_budget_comparable`/`fragile`/`confirmatory_supported`/`evidence_status`，机器读取时不会把统计上显著但预算不等或效应不稳健的组件证据当成确认性结论。除声明配置外，审计还核对**实际训练批次序列**：`get_train_batch` 把每批窗口起点写入 SHA-256，`eval.json` 记录 `data_order_sha256`，同 seed 下各变体必须消费完全相同的序列（只比较非早停 run）；并核对 `config_used.json` 的设备类型，CPU/CUDA 混跑直接判失败。消融矩阵的“覆盖完整”同时要求每格种子数 ≥ `MIN_SEEDS`，单种子格显式标为“种子不足”，不再因变体名出现就显示 8/8。
- **NaN 系统排查**（`train.run_nan_checklist` → `nan_report.json`）：固定顺序 数据（**优先检查失败所在 log 区间的全部候选批次**——NaN 在 log 点才由区间累计损失发现，首个失败步可能早于检测点、预取已覆盖 `batches`，训练循环按区间保存逐步批次引用（≤ log_every 步）并以设备标量标记 `first_bad_step`，传入 `consumed_batches`/`consumed_step_labels` 才不漏罪魁数据；*val 损失非有限时改用真实验证窗口，训练批次仅作取窗失败回退*；NaN/Inf/越界 id）→ 代码（非有限参数定位 + **前向钩子逐批检查候选输入、定位首个“输入有限、输出非有限”的模块**，即 `origin`，附 log(0)/除零/softmax 溢出的排查指向）→ 混合精度（bf16/fp16 下溢-上溢风险；amp 实际生效且出现非有限时，**对命中批次做 fp32 隔离复测**：fp32 全部有限=混合精度数值问题，fp32 同样非有限=排除精度因素）→ 优化（梯度范数尖峰与学习率）。NaN 即中止训练并落盘报告，避免污染曲线；报告显式给出 `checklist_order`、结构化 `bad_batches` 与 `first_bad_data_step`。门控开销另在汇总页单列：实际逐头参数量、额外参数占比、解析门控 FLOPs/token 与总 FLOPs 占比，收益必须对照开销解读。
- **坏案例**（`lega/badcase.py`）：RULER 错误样本按标注错误/误抄干扰值/幻觉/无数值回答归因，语言建模按最差窗口归因；跨变体对照区分共识失败与特异性盲区，逐案例 Markdown 可读。分析不止步于分类：每个案例附**粗归因轴**（数据/构造、校准、难度边界、评测/导出、模型/方法；排位对但置信低的 top1_correct 归“校准”，先于深位难度判定），主导错误模式再按确定性阈值（占比 1/3、深位失败 1/2、最小样本 3 等）合成为**改进方向**（触发证据→建议→下一步实验，按高/中/低优先级），跨变体侧统计每个变体相对 standard 的新增错误类别，把"共识失败→先修数据、特异性新增→方法盲区"落成可执行条目。改进方向进入逐案例 md、`badcase_summary.json`（带 `source` 标注）与 `summary.md`——坏案例分析是提出新改进思路的最重要来源，不能止于分类表。

### SESOI 与等价性/非劣性（TOST；把“不显著”与“无差异”分开）

显著性检验只回答“能否拒绝 Δ=0”。本项目种子数只有 3~5 个，`p>0.05` 既可能
是“真的没有实际重要差异”，也可能只是功效不足；**不能把两者混写成“无差异”**。
`lega/equivalence.py` 因此增加一层 SESOI（smallest effect size of interest，
最小重要差异）解释：

- **SESOI 是分析前声明的协议阈值**，不是从当前数据估计出来的。默认值：
  PPL 逐种子相对差 **1%**（log-PPL 为几何均值比 1%）、RULER EM **1.00 pp**、
  AUROC / PR-AUC 差 **0.010**、因果病灶 ΔNLL **0.01 nat**（约 1% PPL）、
  Shapley 的 PPL 值 **0.01**（−log PPL 单位）与 RULER 值 **1.00 pp**。
- **等价（TOST）**：对配对差做双单侧 t 检验，H0_low: μ≤−SESOI、
  H0_high: μ≥+SESOI；只有两个单侧检验都在 α 下显著（等价于
  100(1−2α)% 区间完全落在 ±SESOI 内）才判“等价”。
- **非劣**：只检验“候选比参照差不超过 SESOI”这一侧；用于“新方法没有
  明显更差”的主张，方向随指标好坏翻转（PPL 越小越好，AUROC/RULER 越大越好）。
- **统计显著但幅度低于 SESOI**：差异可重复，但实际意义有限；不能只报
  “显著优于”就当作重要收益。
- **未定**：区间既不能排除 0，也不能排除超过 SESOI 的差异；不能写成
  “无差异/无效应”，只能写“未定，需补种子或降方差”。

TOST 的 p 与优效性 t 检验是**不同的检验族**：`run_all` 在同一指标家族内对
TOST 原始 p 单独做 Holm 校正（`_adjust_equivalence_family`），n<3 的对比不占
家族大小、也不判等价。0.5×/1×/2× 的 SESOI 敏感性网格各自再做一次 Holm；
若结论随网格翻转，必须在报告中并列披露，不能只挑对主张有利的阈值。等价性
判定**不改变**预注册假设的 支持/拒绝/未定 状态，只作为“实际意义”补充；
`significance.json` 的 `pairs[*].equivalence` 保留 `p_tost`、`p_tost_adj`、
90%/95% 区间、方向、`beyond_margin`、`noninferior`、`sensitivity` 与最终
`status`；顶层 `equivalence` 给出核心指标的计数与 margin 清单；预注册假设
另带 `equivalence_status`/`equivalence_label`。

**因果病灶中的零假设**尤其需要这一层：`head_permute` 是推理期容量/分布
安慰剂，“p>0.05”不能直接写成“安慰剂无效应”。只有 TOST 判等价，才能说
“差异小于 SESOI，与容量/分布解释一致”；若显著变差且超过 SESOI，才支持
“头—证据对应有因果作用”。`gate_closed` 是阳性对照，若它与正常等价，
说明实验灵敏度不足，必须先排查实现/评测；`no_op`（训练后系数已为 0）不参与
等价性判定，因为干预根本没有改变前向，“无差异”不是“无效应”的证据。
`summary.md` 的“SESOI 与等价性/非劣性”表与“关键病灶的 SESOI 解读”表分别
呈现这两类证据。

边界必须如实声明：TOST 仍是 t 分布近似，小样本下与置换检验/MDE 合读；
SESOI 是协议约定，换一个 margin 结论可能改变，所以敏感性网格是必报项；
“等价”只说明差异小于某个实际重要阈值，不说明效应恰好为 0，也不替代
优效性检验或预注册判定。**目前的 SESOI 未写入
`preregistration/hypotheses.json` 的锁链**，因此所有等价性判定一律按
补充性/描述性证据处理；若要把某条“等价/非劣”升级为确认性主张，必须先把
margin 与对比登记进预注册表并走 `--amend` 修订链，再在新的种子/数据上检验。

### 新种子复现预测（均值 CI ≠ 单次复现区间）

跨种子 t 检验与均值置信区间回答的是“当前这组种子的**均值**能否拒绝 0”；
它们不回答“再抽一个新训练种子，候选是否仍会赢”。两者在小样本下差别很大：
均值的标准误按 `1/√n` 收缩，而单个新种子的预测方差还要加上种子自身方差，
宽度按 `√(1+1/n)` 放大。把 95% CI 当作“下一个种子有 95% 概率落在其中”
会系统性过度自信。`lega/replication.py` 因此为每组配对差
`d_i = 候选_i − 参照_i` 增加一层**描述性**预测：

- **t 预测区间**：`mean ± t_{n−1,1−α/2} · sd · √(1+1/n)`，
  回答“同配置/同数据/同评测协议下换一个种子，配对差可能落在哪”；
- **同号概率**：`F_t(|mean| / (sd·√(1+1/n)); n−1)`，即新种子与已观测
  均值同号的模型预测概率（≥0.95 标“方向稳定可复现”，≥0.8 标“方向
  可复现”，否则“未定”）；
- 若该指标有 SESOI，再给三个互斥且穷尽的预测概率：新种子**同向超过
  SESOI**、**落在 ±SESOI 等价带内**、**反向超过 SESOI**；另有“反号概率”
  作为更宽的口径（反向但仍在等价带内也计入）；
- **留一方向一致率**：删掉任一种子后用其余种子均值预测该种子符号的一致
  比例（零差/留一均值为 0 记 0.5），不含分布假设，只作 t 预测的朴素对照。

集成方式：`_enrich_paired` 在原始指标尺度写入 `tests[*].replication`；
`assess_paired_equivalence` 在**指标尺度**（PPL 相对差、log 比、RULER 绝对差
等）写入 `tests[*].equivalence.replication`，避免把原始 PPL 单位的预测
区间与相对 SESOI 混读。`significance.json.equivalence.replication` 汇总
状态计数；`summary.md` 的 SESOI 小节后新增“新种子复现预测”表。

边界必须如实声明：这是**预测/描述性**证据，不是新的确认性检验，不改变
预注册判定；模型假设种子效应独立、可交换且近似正态，3~5 个种子无法检验
该假设，重尾或种子×数据交互会让概率失真；零方差（所有配对差完全相同）
单列为 `degenerate_zero_variance`，不能当作强复现证据；n<3 与项目其余
协议一致不作概率判定；预测只适用于同配置/同数据/同评测协议下的新训练
种子，不外推到新数据、新超参数、新硬件或更长上下文。

## 实验设计与准备（先验功效、分析人群与设计符合性）

前面几节大多在**结果产生之后**做校正；但“这个实验值不值得跑、能回答什么”
必须在开跑前决定。`lega/design/` 因此把设计阶段的四件事变成可执行产物：

1. **先验灵敏度**：在花训练算力之前，回答“计划种子数 / RULER 每格样本数
   能检出多大的效应”。配对 t（PPL/AUROC 的种子级比较）用与
   `min_detectable_effect` 同一正态近似：
   `power = Φ(|Δ|/(σ/√n) − z_{1−α/2}) + Φ(−|Δ|/(σ/√n) − z_{1−α/2})`；
   `required_pairs` 是它的逆运算。RULER 的配对二值结果用 McNemar：
   n≤500 时精确枚举 H0 拒绝域与条件二项分布，更大时用
   `power ≈ Φ(√n|Δ|/√π − z) + Φ(−√n|Δ|/√π − z)`（π 为不一致率）。
   计划输出的是**对一系列合理 SD/不一致率的灵敏度曲线**，不把某一个
   任意假设伪装成数据估计；配置 `design.ppl_paired_sd_rel` /
   `design.ruler_discordance` 若显式声明，则额外核对计划样本量是否达到
   目标功效（声明值必须写死在跑之前，不能事后挑）。
2. **分析人群**：训练前把“计划内 (变体, 种子) 全集”写进
   `design_plan.json`，并声明排除规则——只允许运行在评测前失败、数据/
   配置/训练预算协议违规、产物损坏、测试集访问违规等**与结果无关**的
   排除；禁止因结果不利、离群或不显著而事后删种子/变体。缺失单元必须
   报告；可用样本估计（available-case）只能作描述性敏感性分析。
3. **设计符合性**：汇总时用 `analysis_population` 把实际产物与计划逐格
   比对。计划内单元缺失时，`apply_design_gate` 只降级**受影响假设**的
   确认性判定（候选/参照变体或该指标端点涉及缺失单元），未受影响的
   假设保留原判定；但“少跑预注册种子/计划矩阵缺登记变体”是全局设计
   违规，全部确认性判定降级——前者使登记范围内仍有未观测种子，后者使
   部分假设根本无法检验。计划缺失/与本次 `repro.json` 的变体、种子或
   配置 SHA-256 不一致时（上一轮遗留计划）不据此降级，但显式告警
   “无法核对分析人群”，避免用旧计划冤枉新实验或反过来放行。
4. **分析代码冻结与运行期漂移**：计划落盘时把训练/评测代码、配置、
   预注册表与依赖声明的源码树指纹写入 `inputs.analysis_code`
   （`lega.repro.source_tree_fingerprint`，逐文件 SHA-256 + 总 digest）。
   汇总时重新计算；digest 不一致即 `code_freeze_gate=True`，确认性判定
   全部降级为探索性，并列出变更/新增/删除文件示例——防止“先写计划、
   跑完再改分析代码/阈值、然后只报显著结果”。若 `repro.json` 的
   `tracking_end.drift.changed` 为真（实验期间 git/工作区/源码树/数据
   身份变化），同样触发 `provenance_drift_gate=True`，因为该轮不能
   声称全程对应单一版本。另外，`repro.json` 的 `design.plan_digest` 是
   启动时的计划摘要；汇总时磁盘上的 `design_plan.json` 与之不一致，说明
   计划在结果产生后被改写，触发 `plan_integrity_gate=True`，避免用新
   计划“追认”旧结果。旧计划没有代码指纹、旧产物没有 `tracking_end`
   时显式标记“未核对”并告警，但不据此降级；**未记录不能当成未变化**。

### 执行顺序：种子为区组、区块内随机化

同一台机器上的顺序不是中性的：GPU 温度/降频、文件缓存、驱动状态和后台
任务都会随轮次漂移。旧的 `for variant: for seed` 让某个变体永远排在固定
时段，方法效应与时间漂移完全混淆。新计划默认使用**随机化区组**：

- 每个训练种子是一个区组，区内包含全部计划变体各一次；
- 区内顺序由跑前确定并写入 `design_plan.json` 的 `run_order`，随机流只
  依赖变体/种子/模式（或显式 `design.run_order_seed`），不读任何结果；
- `run_all` 按 `run_order.cells` 提交 run，失败归因也按该单元取变体，不再
  假设变体主序；`design.run_order` 可显式设为 `seed_major` /
  `variant_major`（旧行为），但后者会在设计检查中告警“可能与时间漂移
  混淆”；
- `run_order` 摘要进入 `repro.json` 的 `design` 字段，顺序本身可复算。

**并发限制**：`--jobs>1` 时随机化的是**提交顺序**，多个 run 仍可能重叠
执行，因此不能把区组随机化当作完全消除漂移。对时间/硬件敏感或需要严格
区组解释的正式实验，应设 `--jobs 1` 顺序执行；默认 CPU 并发只减少
“变体主序”这种系统性混淆。

### 机制端点：AUROC/AP 也要先验规划

预注册的 H6/H7 是机制端点（AUROC、AP 的候选−参照配对差），但旧计划只给
PPL/RULER 算功效，等于对机制主张“先跑再看”。现在 `endpoints.auroc` /
`endpoints.ap` 与 PPL/RULER 并列给出：

- SESOI 默认 0.01（绝对 AUROC/AP 差），显式配置
  `design.mechanism_sesoi` 可改；
- 对 `MECHANISM_SD_GRID` 中每个跨种子配对差 SD 给出未校正与确认性 MDE、
  功效、所需种子数；
- 声明 SD 用 `design.auroc_paired_sd` / `design.ap_paired_sd`；声明值在
  `α_eff` 下功效不足时，`auroc_declared_power` / `ap_declared_power`
  fail，而不是等到汇总才说“机制结果不显著”；
- 机制端点的候选变体必须在分析子集内（参照“仅查询项”由同一
  `analysis_*.json` 内部计算，不是独立变体）；计划阶段
  `mechanism_endpoint_variants` 会直接检查，避免把“没有分析产物”误当成
  “无效应”。

这里的 SD 是**换训练种子**时配对差的标准差，不包含单次评测有限样例的
抽样误差；后者由逐 run 的样例聚类 bootstrap 回答，两者不能互相替代。

### 确认性功效：按最终判定阈值规划（不是名义 α）

先验功效最容易犯的错误，是用名义 `α=0.05` 计算“5 个种子能检出 1% SESOI”，
但最终判定并不是单次检验。`run_all.py` 先在同一指标家族内做 Holm 校正
（`_apply_holm`：PPL、RULER、AUROC/AP 各自成族），再对全部预注册假设做
一次全局 Holm（`apply_global_fwer`）。Holm 在**最坏情形**下把任一假设的
阈值收紧到 `α/m`，因此设计阶段必须用

```
α_eff = α / max(m_family, k_global)
```

规划，而不是拿 `α` 直接算。这里 `m_family` 是该指标家族计划内全部配对
（包括探索性对比；Holm 不会因为对比没预注册就不惩罚），`k_global` 是
计划内可检验的预注册假设数。`design.multiplicity` 可选
`holm_worst_case`（默认，取两者最大负担）、`family_only`（只按家族）或
`none`（仅兼容/展示未校正口径）；正式设计必须用前两者之一。

家族大小由**计划配对矩阵**推导，不靠手填：9 个变体 vs `standard`、
6 个 `SINGLE_FACTOR_PAIRS`、预注册表追加的整体对比（例如 H5 的
`lega vs qgate`），当前正式矩阵为 16 个 PPL/RULER 检验；全局确认族为
7 条预注册假设。设计层因此输出两套并列字段，而不是覆盖旧字段：

- 未校正：`mde_relative`、`power_for_sesoi`、`required_seeds_for_sesoi`
  （旧口径，保留用于对照/向后兼容）；
- 确认性：`mde_relative_confirmatory`、
  `power_for_sesoi_confirmatory`、
  `required_seeds_for_sesoi_confirmatory`；RULER 对应
  `mde_pp_confirmatory` 等；端点级还有
  `max_paired_sd_for_sesoi_confirmatory` /
  `max_discordance_for_sesoi_confirmatory` 与 `multiplicity` 元数据。

`multiplicity_plan` 会把模式、各家族大小、全局族大小、计划配对清单和
`α_eff` 一并写进计划摘要，并新增 `multiplicity_plan` /
`ppl_confirmatory_reference` 等检查；声明假设在 `α_eff` 下功效不足时，
`ppl_confirmatory_power` 直接 fail（冒烟运行降为 warn）。这样“5 个种子
够不够”与“5 个种子在最终 Holm 判定下够不够”不会混成同一个结论。

**这是保守规划下界，不是精确 Holm 功效**：实际 Holm 阈值取决于同族其他
p 值的大小，设计阶段不知道它们，只能用最坏情形避免高估；如果其他假设
很强，真实功效可能高于此下界。反过来，把探索性配对排除出家族大小会
系统性低估多重比较负担，因此这里不这么做。

### 计划里必须有什么

`design_plan.json` 在正式训练开始前落盘（基线调优等准备步骤之后；只消费配置与预注册表，不消费任何结果或 test 产物），字段包括：

- `inputs`：配置 SHA-256、变体、种子、α/功效、`multiplicity` 模式与
  `alpha_effective`、登记假设 id、预注册 `scope.seeds`、可产出机制分析
  的变体子集 `analysis_variants`（例如 `standard` 无输出门、不产出
  `analysis_*.json`，不能被误算成缺失）；记录 `run_order_mode` /
  `run_order_seed` / `run_order_seed_source`、机制端点分析样例数与是否
  传入 `data_report`；
- `run_order`：模式、顺序种子及来源、种子区块数与 run 数、逐 run 的
  `position/block/variant/seed`；`run_all` 直接消费该清单；
- `scope_readiness`：预注册 `scope.config` 与当前配置的模型/训练规模差异、
  `scope.data` 声明与 `data_report.tokenizer.kind` 的核对结果；只比较
  `d_model/n_layers/n_heads/ffn_mult/block_size/steps/train_ctx`，不把
  按协议统一调优的 lr/weight_decay 误判为范围外；
- `data_budget`：来自 `data_report.accounting` 的 train/val/test token 数、
  计划每种子 token 触碰次数（steps × batch × grad_accum × train_ctx）、
  触碰次数相对 train token 数的倍数、若随机重采样的预期唯一 token 覆盖
  比例，以及 val/test 在每个 eval_ctx 下的可用窗口数；不读取原始文本，
  也不访问 test；
- `assumptions`：显式声明的计划假设与“仅作展示横轴”的参考值，二者分开；
  另记录 `multiplicity` 模式、`run_order` 模式与机制端点声明 SD；
- `multiplicity_plan`：模式、PPL/RULER/AUROC/AP 家族大小、全局确认族
  大小、计划配对清单与四类指标的 `α_eff`，让摘要绑定“最终要做多少次
  检验”；AUROC/AP 机制族按每个有分析产物的变体计一个检验；
- `endpoints.ppl`：SESOI（默认 1%）、各 SD 下的未校正与确认性 MDE/
  功效/所需种子数、两种口径下可容忍的最大配对差 SD；
- `endpoints.ruler`：每格样本数、格子数、各不一致率下的未校正与确认性
  McNemar 功效/MDE/所需每格样本数、两种口径下可容忍的最大不一致率；
- `endpoints.auroc` / `endpoints.ap`：机制端点的绝对 SESOI、各跨种子
  配对差 SD 下的未校正/确认性 MDE、功效、所需种子数，以及声明 SD；
- `analysis_protocol`：分析人群、排除规则、缺失数据处理、停止规则、
  test 封存、多重比较策略；
- `test_look_budget`：同一 `test.bin` 版本允许的独立 `pipeline_run_id`
  数（默认 1）。汇总按访问账本核对实际 look；超预算时把依赖 test 的
  PPL/RULER 确认性判定降级为探索性；预先声明多次 look 时以
  `alpha/planned_looks` 作为保守判定阈值，并单列无法归属 `run_id` 的旧
  账本记录，避免把“无法核对”当成“只访问过一次”；
- `registered_plan`：登记假设的 id/指标/候选/参照/方向的只读快照，
  使计划摘要同时绑定确认性分析计划本身；
- `inputs.analysis_code`：计划落盘时的分析代码/配置/预注册/依赖源码树
  指纹（总体 digest + 逐文件 SHA-256）；汇总时不一致即代码冻结失败，
  不是重新生成一个计划就能掩盖；
- `checks`：种子数下限、是否覆盖预注册种子、预注册配置规模/数据分词器
  是否一致、run 顺序是否为随机化区组、登记假设的候选/参照变体是否都在
  计划矩阵内、机制端点是否有可分析产物、确认性多重比较规划与 `α_eff`、
  上下文档位是否为合法正整数且至少两档/跨度足够、RULER 最长上下文是否
  超过 `block_size`/`train_ctx`、`length_null` 是否覆盖最长档位、声明假设
  下的未校正与确认性功效、训练 token 触碰次数与 val 窗口是否足够、分析
  规则是否声明；
- `plan_digest`：排除时间戳/机器路径后的内容 SHA-256，供 reproducible 审计。

每次 `save_design_plan` 还会向同目录的 `design_plan_history.jsonl` 追加一行
（同一 digest 连续写入去重）；重跑/改种子/改样本量时旧计划不会被
`design_plan.json` 静默覆盖。汇总页在历史多于一条时显式告警，读者必须
按最新计划核对当前产物，不得把旧计划下的结果混入。

**关键设计事实（正式 small 配置）**：计划 PPL/RULER 家族各有 16 个检验，
全局确认族 7 条，因此 `α_eff=0.05/16=0.003125`。在参考配对差 SD=1% 时，
5 个种子对 1% SESOI 的未校正功效约 0.61、所需约 8 个种子；按确认性
`α_eff` 只有约 0.24、需约 15 个种子。80% 功效下可容忍的配对差 SD 从
未校正约 0.80% 收紧到确认性约 0.59%。RULER 每格 30 个样例对 1pp SESOI
的 McNemar 功效本来极低（参考不一致率 10% 时约 0.002，需每格约 7800 个
样例），校正后更低；因此 RULER 的 1pp 等价/非劣判定只能作描述性证据。
若要升级为确认性主张，必须先增加每格样本数/种子数并重新登记，而不是把
“未检出”写成“无差异”。这也意味着：当前 5 种子设计适合检出明显大于 1%
的 PPL 差异，不足以确认 1% SESOI；配置里声明 `design.ppl_paired_sd_rel`
时计划会直接标出这一点。

### 端点级设计失败

设计检查不再只有“全局降级/不影响”两档。`ruler_context_values`、
`ruler_context_model`、`length_null_coverage` 失败只影响 RULER 端点，
`eval_budget` 失败只影响 eval 端点；`seed_scope`、`hypothesis_variants`、
`scope_config`、`scope_data`、空训练集（`data_budget` fail）与不可执行的
`model_context` 仍按全局失败处理。`design_conformance.json` 新增 `endpoint_plan_failures`；
`apply_design_gate` 只把对应指标的假设降级为探索性，同一轮里未受影响的
PPL 对比不会被 RULER 上下文超容量连带推翻。训练前发现这类问题应优先修正
配置并重新生成计划，而不是等汇总时解释。

### 实测灵敏度（描述性）

汇总页另用本次观测到的配对差 SD 反推 MDE，并与同尺度的 SESOI 比较
（`observed_design_sensitivity`）。它回答“这次设计实际能分辨多小的效应”，
不是把 p 值改写成“观测功效”，也不改变任何显著性判定；MDE>SESOI 的对比
即使显著，也不能证明达到实际重要差异。

### 命令与产物

```bash
# 训练前单独生成/检查设计计划（退出码 2 表示设计 fail）
python -m lega.design --config configs/small.json \
    --seeds 1337,2024,42,7,2025 \
    --variants standard,gate_only,qgate,null,lega,lega_no_len,lega_no_conc,lega_no_query,lega_conc_only,lega_len_only \
    --data_dir data \
    --out results/small/design_plan.json
```

`configs/small.json` / `configs/tiny.json` 的 `design` 块显式写死
`alpha`、`power`、`multiplicity`（默认 `holm_worst_case`）与
`run_order`（默认 `randomized_blocked`）；修改这些字段等同于改设计前提，
必须重新生成计划而不是只重跑汇总。`run_all.py`
在训练前自动生成 `results/<tag>/design_plan.json`，并把
`data_report.json` 的语料规模核算一并传给设计计划（摘要进
`repro.json` 的 `design` 字段，其中包含 `run_order` 模式、顺序种子与逐
run 清单），汇总时写 `design_conformance.json`，并在 `significance.json`
增加 `design_plan` / `design_conformance` /
`observed_design_sensitivity`。`summary.md` 新增“实验设计与分析人群（先验
计划 vs 实际）”小节，逐项列出检查状态、run 顺序、PPL/RULER/AUROC/AP
灵敏度表、计划内单元缺失、分析代码冻结/运行期漂移状态与确认性降级原因。

### 边界

- 功效公式是正态/t 近似；n=3~5 时它用于设计灵敏度，不是精确功效承诺。
  McNemar 精确功效的边界在 n≤500 时精确枚举，更大时近似。
- 确认性功效使用 Holm **最坏情形** `α/max(m_family,k_global)`，是保守
  下界而非精确 Holm 功效；实际阈值依赖同族其他 p 值。家族大小按计划内
  全部同指标配对计算（含探索性对比），缺失产物只会让实际家族变小，因此
  这个口径不会系统性低估多重比较负担。
- 计划假设（SD、不一致率）若来自历史数据，必须在跑之前声明来源；来自
  当前结果再回填就是事后选择，计划摘要会让这种改写可见。
- “缺失单元”与结果可能相关（例如难训练变体更易 NaN），此时
  available-case 估计本身有偏；设计门只保证不把它冒充确认性证据，不能
  修复缺失机制。
- 计划 fail 不硬中止流水线（允许明确标注的探索性/冒烟运行），但会进入
  汇总页与确认性降级证据；正式实验应把 `status=ok` 当作开跑前的检查项。
