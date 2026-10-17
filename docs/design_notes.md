# 设计笔记

## LEGA（Length-calibrated Evidence-Gated Attention）

注意力门控同时参考三个因素：查询向量 $q$、注意力集中度 $c=\log(n\cdot\max_j p_j)$、上下文长度的对数 $\log n$：

$$g=\sigma(w^\top q+a\,c+b\log n+d),\qquad o'=g\cdot o$$

- $c$ 在均匀注意力下恒为 0，与上下文长度无关，因此可以把"读得集中"和"读得长"解耦；
- $b\log n$ 是长度校准项，让每个头自行学习长上下文中"集中读取"可信度的变化；
- 初始化 $w=a=b=0,\ d=2.0$，接近标准注意力，训练可稳定启动。

## 变体

`standard`、`qgate`、`null`、`lega` 及五个消融（`lega_no_len`、`lega_no_conc`、`lega_no_query`、`lega_conc_only`、`lega_len_only`），覆盖全因子与去一法消融矩阵，详见 README。

## 评测

- WikiText-103 PPL（有 held-out test 时评 test，缺 test.bin 时回退验证集并提示选择偏差；窗口不重复、等间隔抽样）；
- 简化 RULER 五任务（niah_synth / niah_corpus / niah_multikey / fwe / vt），多档上下文、贪心解码；
  每格 em 附带 Wilson 95% 区间（`em_ci95_wilson`）：n_samples 小时点估计的抽样误差不可忽略；
- 机制分析（`lega/analysis.py`）：misleading_rate、P(hit|集中) 对比、门控分数 AUROC。

## 统计推断协议（小样本科研口径）

正式实验的种子数只有 3~5，任何单点统计声明都站不住。协议分四层：

1. **主判定**：同种子配对 t 检验，同一指标家族内 Holm–Bonferroni 校正
   （控制 FWER），配对 Cohen's d_z 效应量；n<3 只报描述统计。
2. **稳健性印证**（每组配对同时报告，`significance.json`）：
   - **符号翻转置换检验** p_perm（分布无关；n≤16 全枚举精确）：t 检验在
     小样本下依赖正态假设，两套结论冲突时在结论文本中显式标 ⚠ 而非静默
     采信任何一方。**粒度边界必须知情**：置换双侧 p 的理论下限是 2/2^n
     （n=3→0.25、n=5→0.0625、n=6→0.03125），种子 <6 时"置换不显著"是
     粒度所限，此时 t 的显著性完全依赖正态假设；n≥6 后不一致才是真实警报；
   - **配对差 bootstrap 95% CI**：效应量的不确定范围，防止把点估计当精确值；
   - **MDE（最小可检测效应）**：(z_{1-α/2}+z_{power})·std/√n。"不显著"
     必须对照 MDE 解读：观测差远小于 MDE → 功效不足（先补种子），而不是
     "证明了没有效应"。
3. **机制主张的正式检验**：AUROC(完整 logit) − AUROC(仅查询项) 作为独立
   指标家族做同种子配对检验（`run_all._auroc_diff_tests`）——两个 AUROC
   来自同一批记录，并排放数字不构成证据。与逐 run 的**样例聚类 bootstrap**
   区间（`analysis.auroc_cluster_bootstrap`）互为印证：后者以样例为重采样
   单元（同一样例的层×头×查询位记录共享难度，不是独立观测），回答"换样例
   波动多少"；前者回答"换训练种子方向是否稳定"。
4. **阈值敏感性**：misleading/P(hit|集中) 系指标在多档 τ 上并报
   （`analysis.tau_sensitivity`，默认 [0.3, 0.5, 0.7] 并入主 τ），
   机制结论不得只挂在单一任意选定的阈值上。

## 预注册（confirmatory vs exploratory）

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

## 可复现性与 sanity check

- **种子**：`set_seed` 覆盖 Python/NumPy/PyTorch（CPU/CUDA）、cuDNN 确定性算法与子进程 `PYTHONHASHSEED`；批采样由显式 `np.random.default_rng(seed)` 驱动（无 DataLoader worker）。`run_all` 逐命令把 `PYTHONHASHSEED` 对齐到该命令自己的 `--seed`，`run_manifest.json` 内嵌 `seed_report`（逐随机源的种子、**RNG 真实状态短指纹**、确定性开关、数据顺序机制）供事后审计：指纹可被第三方用 `set_seed(seed)+rng_state_fingerprint()` 复算，当前状态不属于所请求 seed 时为 `null`；`PYTHONHASHSEED` 分“解释器启动值/子进程 env/是否真的对当前进程生效”三字段记录，避免字段被 set_seed 自身写值而自证。跨硬件架构不保证逐位一致，同机重跑稳定。
- **过拟合小批量**（`lega/sanity.py`）：训练前在固定小批量、关闭正则化与数据扰动的条件下检查损失能否趋近 0、下一 token 准确率能否接近 100%；首步还会做**逐参数梯度覆盖**（missing/zero/nonfinite），可定位“总梯度范数正常但门控等子模块没接进计算图”的断流问题，并自动核对两条中间结果：训练路径 loss（lm_head 分块 + 交叉熵）与直接完整 logits 的 `F.cross_entropy` 数值一致（捕捉分块/归一化/ignore 处理错误），以及消费点标签满足 `y = x 右移一位`（捕捉错位/换序）。不能过拟合或核对失败说明梯度/损失/标签链路存在严重问题，`run_all` 会直接中止全量训练。通过是必要非充分条件，需继续配合中间结果检查。
- **追踪**：训练写 `run_manifest.json`（git commit/dirty、配置 SHA-256、数据指纹、种子审计、环境、命令行）与 `config_used.json`/`train_log.jsonl`；`run_all` 写全局 `repro.json`，并向固定路径 `results/experiments.jsonl` 追加实验台账（started/completed 生命周期，逐 run 最终指标与曲线/被评测 checkpoint/结果文件 SHA-256；`eval_*.json` 内嵌 `ckpt_sha256` 绑定权重），重跑不丢历史，评测 JSON 内嵌同一溯源字段。

## 结果分析协议

- **训练曲线而非最终值**（`lega/curves.py` → `curve_diagnosis.json`）：训练降/验证升=过拟合（正则化/增数据/早停），**训练停滞而验证回升同样判过拟合**（平台期泛化先于训练损失恶化，不能误判成欠拟合并建议续训/扩容），**训练仍在实质下降而验证无实质变化=过拟合趋势**（验证不再随训练改善的早期信号：按正则化/早停处理，不据此续训；验证仍同步下降才是正常收敛），两者都不降=欠拟合或数据管道/学习率问题，NaN=训练失效；另独立报告 **pre-clip 梯度范数尖峰/非有限**（最大/中位数 ≥10×），它是 NaN/发散的早期信号，可能先于损失曲线异常出现。训练留档图（`fig_train_curves.png`）除训练 loss 与 val PPL 两个面板外，还有训练(实线)/验证(虚线)损失同轴对照面板，泛化差距与过拟合形态一眼可见。
- **NaN 系统排查**（`train.run_nan_checklist` → `nan_report.json`）：固定顺序 数据（**优先检查失败步批次**——预取已覆盖 `batches`，传入 `consumed_batches` 才不漏罪魁数据；*val 损失非有限时改用真实验证窗口，训练批次仅作取窗失败回退*；NaN/Inf/越界 id）→ 代码（非有限参数定位 + **前向钩子定位首个“输入有限、输出非有限”的模块**，即 `origin`，同样以失败输入为主，附 log(0)/除零/softmax 溢出的排查指向）→ 混合精度（bf16/fp16 下溢-上溢风险；amp 实际生效且出现非有限时，**对同一批次做 fp32 隔离复测**：fp32 全部有限=混合精度数值问题，fp32 同样非有限=排除精度因素）→ 优化（梯度范数尖峰与学习率）。NaN 即中止训练并落盘报告，避免污染曲线。
- **坏案例**（`lega/badcase.py`）：RULER 错误样本按标注错误/误抄干扰值/幻觉/无数值回答归因，语言建模按最差窗口归因；跨变体对照区分共识失败与特异性盲区，逐案例 Markdown 可读。
