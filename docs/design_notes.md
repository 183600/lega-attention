# 设计笔记

## LEGA（Length-calibrated Evidence-Gated Attention）

注意力门控同时参考三个因素：查询向量 $q$、注意力集中度 $c=\log(n\cdot\max_j p_j)$、上下文长度的对数 $\log n$：

$$g=\sigma(w^\top q+a\,c+b\log n+d),\qquad o'=g\cdot o$$

- $c$ 在均匀注意力下恒为 0，与上下文长度无关，因此可以把"读得集中"和"读得长"解耦；
- $b\log n$ 是长度校准项，让每个头自行学习长上下文中"集中读取"可信度的变化；
- 初始化 $w=a=b=0,\ d=2.0$，接近标准注意力，训练可稳定启动。

## 变体

`standard`、`qgate`、`null`、`lega` 及三个消融（`lega_no_len`、`lega_no_conc`、`lega_no_query`），详见 README。

## 评测

- WikiText-103 验证集 PPL（窗口不重复、等间隔抽样）；
- 简化 RULER 五任务（niah_synth / niah_corpus / niah_multikey / fwe / vt），多档上下文、贪心解码；
- 机制分析（`lega/analysis.py`）：misleading_rate、P(hit|集中) 对比、门控分数 AUROC。
