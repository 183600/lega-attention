# 对比实验与消融的科研方法改进

本文件说明在不修改 `.github` 与 `README*.md` 的前提下，对 LEGA 项目科研方法在对比实验与消融角度的增强。

## 1. 现有基础
项目已实现完整的 2^3 全因子门控矩阵、去一法消融、推理期因果病灶、剂量-反应、Shapley 分解、参数匹配安慰剂、容量匹配证据置零等，统计上采用多种子配对检验 + Holm 校正、SESOI/TOST 等价性、FDR 控制等。

## 2. 改进点

### 2.1 统计功效与可重复性
- **种子数提升**：`configs/comprehensive_small.json` 将种子从 5 个提升至 8 个（1337,2024,42,7,2025,2023,19,2026），降低 MDE，提升对小效应检测的稳定性。
- **基线调优统一注入**：`--tune_baseline --tune_seeds` 对 `standard` 做 lr/weight-decay 网格搜索，选优后统一注入全部变体，避免“超参数红利”误归因。
- **完整评估口径**：`--full_eval` 使用全部非重叠窗口评估 PPL，而非抽样子集；`--tune_baseline` 证据写入 `baseline_tuning.json`。

### 2.2 参数与容量对照
- **参数匹配安慰剂** `--param_control`：加入 `lega_placebo`，门控结构与 LEGA 完全相同但证据替换为确定性噪声，参数量/FLOPs/初始化一致，用于隔离“多几个门控参数带来的优化红利”。
- **容量匹配证据置零** `--capacity_ablation`：训练 `lega_null_query/lega_null_conc/lega_null_len`，保留 Full 的全部参数与计算，仅把对应证据通道置零。与 `lega_no_*` 构成 Full / Null / No 三臂，可精确拆分 `info_X = Full−Null` 与 `capacity_X = Null−No`。
- **原始参数化对照** `--raw_factorial`：用 `a·log max_p` 替代归一化 `a·log(n·max_p)` 构成同构 2^3 网格，检验 A/B/C 主效应是否依赖 c 的归一化选择，验证重参数化稳健性。

### 2.3 构念效度与统计量稳健性
- **替代集中度统计量**：`lega_l2` 与 `lega_entropy` 在门控结构、参数个数上与 LEGA 完全一致，仅把 B 证据由 `max_j p_j` 换成 L2 质量/熵质量，检验结论是否绑定特定统计量。
- **推理期因果病灶**：对 `lega,qgate,gate_only` 做 `no_query/no_conc/no_len/no_evidence/gate_open/closed/constant_mean/head_permute/conc_permute/sign_flip` 干预，同一 checkpoint/同一批 val 窗口配对，区分训练期优化效应与前向使用效应。
- **连续剂量-反应**：对 A/B/C 系数做 λ∈{0,0.5,0.75,1,1.25,1.5,2} 缩放，检验训练得到的系数是否处在合理剂量且效应单调，避免二值病灶只能回答“有无”。

### 2.4 对比矩阵完整性
默认主矩阵 10 变体 + 通过标志位可扩展：
- 参数匹配安慰剂：`lega_placebo`
- 容量匹配：`lega_null_query,lega_null_conc,lega_null_len`
- 原始参数化：`lega_raw,lega_raw_no_len,lega_raw_conc_only,lega_raw_no_query`
- 构念效度：`lega_l2,lega_entropy`
- 补充对照：`lega_detach_conc,null_zero`

汇总表自动标注矩阵缺失格，确保任何“各组件均有用”的结论必须同时满足去一法显著变差、全因子增量显著变好、且推理期病灶复现。

### 2.5 长度校准专项对比
- `length_null` 诊断强制报告 `c~log2 n` 斜率，随机/重复/均匀上下文下不同斜率用于检验“c 已与长度解耦”主张。
- `length_calibration` 与 `length_ppl` 做长度分层 PPL 与校准曲线，RULER 按上下文长度分层，报告 macro/micro 均值，避免平均分掩盖长上下文效应。
- `length_stratification` 在设计中开启，确保变体间长度子集评测窗口一致。

## 3. 使用方式
```bash
bash scripts/run_comprehensive_ablation.sh configs/comprehensive_small.json data/wikitext103
```
或手动：
```bash
python run_all.py --config configs/comprehensive_small.json --data_dir data/wikitext103 \
  --param_control --capacity_ablation --raw_factorial \
  --tune_baseline --tune_seeds 1337,2024,42 \
  --ablate_variants lega,qgate,gate_only \
  --dose_response --full_eval
```

## 4. 输出与判定
- `results/<tag>/summary.md`：主表 + 消融矩阵 + 参数/容量对照 + 重参数化稳健性 + SESOI/TOST 等价性。
- `results/<tag>/significance.json`：同种子配对检验、组件证据、推理期消融、剂量-反应、Shapley、FDR 控制。
- `results/<tag>/ablation_*.json`、`dose_response_*.json`、`factorial` 相关文件提供逐窗口可审计证据。

## 5. 不可修改项
本改进未触碰 `.github/` 任何文件，亦未删除或移动任何 `README*.md`，符合项目约束。
