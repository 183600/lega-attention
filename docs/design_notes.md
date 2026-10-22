# 设计笔记

## LEGA（Length-calibrated Evidence-Gated Attention）

注意力门控同时参考三个因素：查询向量 $q$、注意力集中度 $c=\log(n\cdot\max_j p_j)$、上下文长度的对数 $\log n$：

$$g=\sigma(w^\top q+a\,c+b\log n+d),\qquad o'=g\cdot o$$

- $c$ 在均匀注意力下恒为 0，与上下文长度无关，因此可以把"读得集中"和"读得长"解耦；
- $b\log n$ 是长度校准项，让每个头自行学习长上下文中"集中读取"可信度的变化；
- 初始化 $w=a=b=0,\ d=2.0$，接近标准注意力，训练可稳定启动。

## 变体

`standard`、`gate_only`、`qgate`、`null`、`lega` 及五个消融（`lega_no_len`、`lega_no_conc`、`lega_no_query`、`lega_conc_only`、`lega_len_only`），覆盖全因子与去一法消融矩阵，并以 `gate_only` 隔离门控本身效应，详见 README。

消融口径按组件严格定义：B 为 $a\,c$（$c=\log(n\max_j p_j)$ 始终保留长度归一化），C 为 $b\log n$；因此 Full−C（`lega_no_len`）只去掉 $b\log n$，不会连带删掉 $a\log n$（否则不再是单组件消融）。汇总同时以 `gate_only` 为无组件基准给出 A×B/A×C/B×C 交互检验，区分“主效应可加”与“协同/次可加”。

## 评测

- WikiText-103 PPL（有 held-out test 时评 test，缺 test.bin 时回退验证集并提示选择偏差；窗口不重复、等间隔抽样）；
- 简化 RULER 五任务（niah_synth / niah_corpus / niah_multikey / fwe / vt），多档上下文、贪心解码；
  每格 em 附带 Wilson 95% 区间（`em_ci95_wilson`）：n_samples 小时点估计的抽样误差不可忽略；
  niah_corpus 的干扰语句优先取 held-out 的 val/test.txt（train 仅在无 held-out
  文本时回退并显式告警），来源写入 payload/summary 的 `corpus_source`——评测
  上下文与训练语料重叠会把“检索”变成“回忆”，绝对分数不可比；
- 机制分析（`lega/analysis.py`）：misleading_rate、P(hit|集中) 对比、门控分数 AUROC 与 PR-AUC、门控决策点 F1（macro/micro/weighted + **恒写/恒不写稻草人对照** `gate_f1_trivial`——门控 F1 不高于无信息固定规则时没有决策价值）。
- 简单统计基线（`lega/ngram.py`）：与神经模型同 split 的一元/二元/**插值二元**平滑 n-gram PPL。插值模型 `P = λ·P_addα(w|prev) + (1−λ)·P_addα(w)` 让未见上下文回退到一元分布，而不是只回退到近似均匀；α 与 λ 只在 val 上按 mean NLL **联合**选优，`selected_model` 取 val 上最强的简单模型（λ=1 可退化为纯 bigram，因此最强基线不会低于它），正式报告只消费选定参数一次；test split 需显式 `--allow_test` 并写访问账本。流水线 `--ngram_split auto`（有非空 `test.bin` → test，否则 val）与神经评测同一规则，从源头避免 val 评测的模型对比 test 基线这类错位。

## 基线与指标口径（先测最简单的，再谈复杂模型）

- **简单基线不可省**：没有无参数基线时，“PPL=xx”无法区分“数据容易”与“模型好”。n-gram 基线与神经变体用同一 train 拟合、同一 held-out split 评测，给出绝对难度参照。
- **与简单基线的差异也要检验**：n-gram 是确定性基线（无种子方差），逐种子神经 PPL 与同一常数做**单样本配对检验**，按变体家族 Holm 校正，报告比值/相对降低/p_adj/判定（`ngram_baseline_comparison` → `significance.json.ngram_comparison`）。split 不一致的变体单列“未纳入比较”，绝不混表。**同 split 还不够、必须同 token**：神经 PPL 只在 `eval_iters` 个等间隔窗口的目标 token 上计算，n-gram 因此额外报告逐位置对齐口径 `ppl_window_matched`（`lega.ngram.window_positions` 与 `train._val_eval_indices` 同规则，评分位置为每个窗口的 `s·ctx+1..s·ctx+ctx`）；仅同 token 还不够——神经 checkpoint 由窗口 val loss 选择，n-gram 若仍用完整 val 选 α/λ，选参 token 集依旧不同；新产物另在**同一批窗口目标 token** 上独立选参（`window_matched_selection` / `ppl_window_matched_selected`），汇总优先用这套参考，旧产物才回退 `ppl_window_matched`。汇总在全部纳入比较的 run 的 `eval_ctx/eval_iters/val_windows` 与 `window_spec` 一致时优先使用窗口参考，任一不一致即整体回退全 split 并写入 `fallback_reason`（同一 Holm 家族共用同一参考值）。该对比未预注册，标为补充性/探索性证据，不替代变体间受控配对检验。**逐窗口配对**：n-gram 侧同时落盘选定模型的 `window_losses_matched` 与神经评测同规则的 `window_indices_matched`；汇总在逐位置对齐成立时，对同一窗口序号做 `loss_neural − loss_ngram`，并用“先重采样种子、再重采样窗口”的分层 bootstrap 报告 ΔNLL 区间/PPL 比。窗口序号不一致或 n-gram 缺字段时显式跳过并写明原因，不按位置强配。
- **基线必须调过，调优选择本身也要防噪声**：新方法的涨点可能只是超参数红利。`lega/tuning.py` 对基线（standard）做 lr × weight-decay 小网格（默认 lr×{0.5,1,2}），只用验证集选择（`train_log.jsonl` 全评测点最小值，与 `model_best.pt` 同口径；test 从不加载）；`default_is_optimal` 在默认与网格最优的相对差 ≤0.5% 时为真。单一种子上的网格最大值可能是“幸运种子”选出的假最优（winner's curse）：`--tune_seeds` 让每个组合在多个种子上重复训练、按**平均 val_loss** 选择，报告 `selection_seeds` 与逐种子明细；不指定时只用单一种子（汇总页显式警告），正式实验建议 ≥3 个。`run_all --tune_baseline` 把最优组合注入**全部**变体（受控变量仅整体平移），证据写入 `baseline_tuning.json` 并在汇总页渲染；缺证据时汇总页显式警告“未调优基线”。**调优范围必须披露**：该流程只对 `standard` 做网格；`qgate`/`null` 等“已有工作代表”继承同一组 lr/weight_decay，未各自调优，因此“优于已有工作”只能按**同超参受控对照**解读，不能写成“优于各自最佳调优版本”；差距接近 MDE/种子噪声时更不足以支持优越性结论。
- **机制判别也要有最简监督基线（线性探针）**：门控 logit 是语言建模的副产品、训练时从未见过 hit 标签；“门控 AUROC/PR-AUC 高”不能排除“`log max_p`、`log n`、`qterm` 这些廉价特征本身线性可分，复杂门控结构没有贡献”。`lega/probe.py` 在同一批记录上训练 L2 逻辑回归，并对简单基线同样做**嵌套、按样例分组、按 hit 标签分层**的交叉验证：同一 (层,头,查询位) 记录不独立，按行切分会“背题”；稀有正类可能让某些折没有正类，分层分组降低空正类折造成的基线低估。λ 与决策阈值只在每个外层训练折的内层 CV 上选择（λ 按与主检验对齐的 PR-AUC/AUROC，阈值按内层 OOF macro-F1），最终 OOF 不再参与选参，避免选择偏差；标准化统计只用训练折拟合。报告 OOF 的 AUROC/PR-AUC、Brier/ECE、以及固定 0.5 与调优阈值两套 F1（macro/micro/weighted 三口径）。`run_all.linear_probe_baseline_comparison` 把逐种子 Δ=探针−门控 做配对检验 + Holm 校正并渲染进基线小节：探针超过门控属预期（判别能力本可由线性模型学得），门控不能在主指标上被探针显著超越，才有资格谈“结构优越性”；单统计量规则（max_p）一并列出作最低档次对照；旧产物缺探针时显式警告而不是当作已有对照。该对比与 n-gram 对比一样属补充性/探索性证据，不替代预注册的确认性检验。
- **不平衡指标必须并报且注明口径**：hit/¬hit 判别是类别极不平衡的二分类，`hit_base_rate`（正类率）即 PR-AUC 的随机基线。正类率 <5% 时 AUROC 因负样本基数巨大、FPR 增长缓慢而乐观，**以 PR-AUC（average precision）为主指标**（<0.1% 尤甚）；正类率 ≥5% 时两者并报。F1 一律注明平均口径（`gate_f1` 同时给 macro/micro/weighted 与逐类明细；macro 对稀有类最诚实，汇总表列 macro）。**任何学习指标都要有无信息对照**：门控 F1 并报恒写/恒不写固定规则的 `gate_f1_trivial`，打不过固定规则就谈不上“门控学会判断”。RULER 的 EM 为生成式指标、无决策阈值，不受此影响（fwe 另报召回率、精确率与 F1 作诊断）。**排序好不等于概率可信**：门控 g 是 sigmoid 输出且被当作软权重/概率使用，必须另报 Brier 与 ECE（与常数基率 Brier 基线对照；`analysis_*.json` 的 `calibration_g`），AUROC/PR-AUC 不能替代校准证据。ECE 同时给等宽与等频两套箱：等宽箱在分数聚集/小样本时可能大量为空，单个样本独占一箱还会把 ECE 机械压低；等频箱按分数排序均分样本，作为对分箱选择不敏感的第二组证据。Brier 另做 Murphy 分解：等频箱内用箱均值预测时 `Brier_bin = reliability − resolution + uncertainty`；原始分数在箱内不恒定时还须加箱内项 `brier_binning = within_score − 2·within_cov`（`brier = reliability − resolution + uncertainty + brier_binning`），否则三项之和并不等于原始 Brier。该分解把“箱内校准误差”“箱间区分度”“标签固有不确定性”与“分箱引入的箱内方差/协方差”分开；这些仍只是描述性证据，不替代正式检验。RULER 平均同时报 **macro**（任务等权、任务内上下文档位等权，预注册口径）与 **micro**（按每格样本数加权），避免“平均”未注明定义而掩盖任务/格大小不均。两者统一用各任务 **EM**（fwe 为严格集合 EM，与 H4 的“RULER 平均 EM”一致）；fwe 的 recall 仅单列诊断、不进入平均——recall 不惩罚“把上下文词全部列出”的灌水策略；precision/F1 一并单列，才能区分“没找到高频词”与“找对但灌水/漏答”。

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
- **追踪**：训练写 `run_manifest.json`（git commit/dirty、配置 SHA-256、数据指纹、种子审计、环境、命令行、sanity 证据）与 `config_used.json`/`train_log.jsonl`；dirty 工作区额外记录 diff 指纹/文件清单并保存 `worktree.patch`（已跟踪 diff + 未跟踪文件，上限 8 MiB、截断显式标记；仅凭 commit 无法复原代码时可重放）。`run_all` 写全局 `repro.json`，并向固定路径 `results/experiments.jsonl` 追加实验台账（started/completed/aborted 生命周期，记录解析后的完整超参数与逐 run `config_used.json` 内容，config 文件被改写/删除后仍可还原；逐 run 最终指标与曲线/被评测 checkpoint/结果文件 SHA-256；`eval_*.json` 内嵌 `ckpt_sha256` 绑定权重），失败轮次带阶段/原因/退出码，重跑不丢历史，评测 JSON 内嵌同一溯源字段。台账快照还内嵌逐 run 的 `seed_report`/环境/sanity 身份，manifest 丢失后仍可自证；`python -m lega.experiment --check_artifacts` 可核验 started/completed 链、重复终态、悬挂 run 与产物哈希。

## 结果分析协议

- **训练曲线而非最终值**（`lega/curves.py` → `curve_diagnosis.json`）：训练降/验证升=过拟合（正则化/增数据/早停），**训练停滞而验证回升同样判过拟合**（平台期泛化先于训练损失恶化，不能误判成欠拟合并建议续训/扩容），**训练仍在实质下降而验证无实质变化=过拟合趋势**（验证不再随训练改善的早期信号：按正则化/早停处理，不据此续训；验证仍同步下降才是正常收敛），两者都不降=欠拟合或数据管道/学习率问题，NaN=训练失效；另独立报告 **pre-clip 梯度范数尖峰/非有限**（最大/中位数 ≥10×，NaN/发散的早期信号，可能先于损失曲线异常出现；训练日志按 log 区间记录 `grad_norm_max` 与发生步，避免只采区间末步而漏掉中段爆炸，并给出尖峰 step 清单）与**瞬态损失尖峰**（单步损失 > 邻近 5 点中位数 8×：分段均值会把它平均掉，常对应异常批次或一次梯度爆炸；同时记录区间 `loss_max`/发生步，防止一步异常被均值稀释），并对损坏日志行跳过计数。末段趋势另比较后 1/3 vs 中 1/3，避免“早先大幅改善、末段才平台/回升”被全程首末均值掩盖。训练留档图（`fig_train_curves.png`）除训练 loss 与 val PPL 两个面板外，还有训练(实线)/验证(虚线)损失同轴对照面板，泛化差距与过拟合形态一眼可见。诊断不是事后才能看：每次评测记录 `val_vs_best_rel`（当前 val 与历史最优的差距），连续 3 次无实质改善且未开早停时训练中打印过一次拟合/平台告警（`live_overfit_hint`，改善后计数归零、下轮连击会再次提示），避免“训练 loss 仍在降”的过程信号被忽略。
- **早停是实际机制而非仅建议**（`train.py` 的 `early_stop_update` + `--early_stop_patience`/配置 `early_stop_min_delta`）：验证连续 patience 次评测无实质改善（相对阈值，不随损失尺度漂移）即终止。默认关闭（0）保证受控对比不受影响；启用时 `eval.json` 同时记录计划/实际步数与 `early_stopped`，train_log 写入 `early_stop` 终标记，曲线诊断据此把截断标注为设计行为（不再建议"增加步数"或"延长训练"），受控变量审计改为按计划步数核对并把逐 run 实际步数如实列出——不让早停把"实际步数不同"伪装成"受控条件一致"。模型选择口径不变：始终用 val 选出的 `model_best.pt`。审计同时核对实际评测协议：PPL 的 `split/eval_ctx/eval_iters/val_windows/val_tokens_evaluated` 必须同批 token，RULER 的 `ctx_lens/n_samples/sample_seed/corpus_source/max_new` 与已评测 (任务, 上下文) 格集合必须一致；配对检验还会逐种子整对剔除协议/格不一致的观测，避免独立重评或不同题面混入“同种子配对”。确认性判定由 `apply_budget_gate` 降级；`component_evidence` 的每条证据也带 `training_budget_comparable`/`fragile`/`confirmatory_supported`/`evidence_status`，机器读取时不会把统计上显著但预算不等或效应不稳健的组件证据当成确认性结论。除声明配置外，审计还核对**实际训练批次序列**：`get_train_batch` 把每批窗口起点写入 SHA-256，`eval.json` 记录 `data_order_sha256`，同 seed 下各变体必须消费完全相同的序列（只比较非早停 run）；并核对 `config_used.json` 的设备类型，CPU/CUDA 混跑直接判失败。消融矩阵的“覆盖完整”同时要求每格种子数 ≥ `MIN_SEEDS`，单种子格显式标为“种子不足”，不再因变体名出现就显示 8/8。
- **NaN 系统排查**（`train.run_nan_checklist` → `nan_report.json`）：固定顺序 数据（**优先检查失败所在 log 区间的全部候选批次**——NaN 在 log 点才由区间累计损失发现，首个失败步可能早于检测点、预取已覆盖 `batches`，训练循环按区间保存逐步批次引用（≤ log_every 步）并以设备标量标记 `first_bad_step`，传入 `consumed_batches`/`consumed_step_labels` 才不漏罪魁数据；*val 损失非有限时改用真实验证窗口，训练批次仅作取窗失败回退*；NaN/Inf/越界 id）→ 代码（非有限参数定位 + **前向钩子逐批检查候选输入、定位首个“输入有限、输出非有限”的模块**，即 `origin`，附 log(0)/除零/softmax 溢出的排查指向）→ 混合精度（bf16/fp16 下溢-上溢风险；amp 实际生效且出现非有限时，**对命中批次做 fp32 隔离复测**：fp32 全部有限=混合精度数值问题，fp32 同样非有限=排除精度因素）→ 优化（梯度范数尖峰与学习率）。NaN 即中止训练并落盘报告，避免污染曲线；报告显式给出 `checklist_order`、结构化 `bad_batches` 与 `first_bad_data_step`。门控开销另在汇总页单列：实际逐头参数量、额外参数占比、解析门控 FLOPs/token 与总 FLOPs 占比，收益必须对照开销解读。
- **坏案例**（`lega/badcase.py`）：RULER 错误样本按标注错误/误抄干扰值/幻觉/无数值回答归因，语言建模按最差窗口归因；跨变体对照区分共识失败与特异性盲区，逐案例 Markdown 可读。分析不止步于分类：每个案例附**粗归因轴**（数据/构造、校准、难度边界、评测/导出、模型/方法；排位对但置信低的 top1_correct 归“校准”，先于深位难度判定），主导错误模式再按确定性阈值（占比 1/3、深位失败 1/2、最小样本 3 等）合成为**改进方向**（触发证据→建议→下一步实验，按高/中/低优先级），跨变体侧统计每个变体相对 standard 的新增错误类别，把"共识失败→先修数据、特异性新增→方法盲区"落成可执行条目。改进方向进入逐案例 md、`badcase_summary.json`（带 `source` 标注）与 `summary.md`——坏案例分析是提出新改进思路的最重要来源，不能止于分类表。
