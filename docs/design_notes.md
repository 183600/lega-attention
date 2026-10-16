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
- 机制分析（`lega/analysis.py`）：misleading_rate、P(hit|集中) 对比、门控分数 AUROC。

## 可复现性与 sanity check

- **种子**：`set_seed` 覆盖 Python/NumPy/PyTorch（CPU/CUDA）、cuDNN 确定性算法与子进程 `PYTHONHASHSEED`；批采样由显式 `np.random.default_rng(seed)` 驱动（无 DataLoader worker）。`run_all` 逐命令把 `PYTHONHASHSEED` 对齐到该命令自己的 `--seed`，`run_manifest.json` 内嵌 `seed_report`（逐随机源的种子、确定性开关、数据顺序机制）供事后审计。跨硬件架构不保证逐位一致，同机重跑稳定。
- **过拟合小批量**（`lega/sanity.py`）：训练前在固定小批量、关闭正则化与数据扰动的条件下检查损失能否趋近 0、下一 token 准确率能否接近 100%；首步还会做**逐参数梯度覆盖**（missing/zero/nonfinite），可定位“总梯度范数正常但门控等子模块没接进计算图”的断流问题。不能过拟合说明梯度/损失/标签链路存在严重问题，`run_all` 会直接中止全量训练。通过是必要非充分条件，需配合中间结果检查。
- **追踪**：训练写 `run_manifest.json`（git commit/dirty、配置 SHA-256、数据指纹、种子审计、环境、命令行）与 `config_used.json`/`train_log.jsonl`；`run_all` 写全局 `repro.json`，并向固定路径 `results/experiments.jsonl` 追加实验台账（started/completed 生命周期，逐 run 最终指标与曲线/结果文件 SHA-256），重跑不丢历史，评测 JSON 内嵌同一溯源字段。
