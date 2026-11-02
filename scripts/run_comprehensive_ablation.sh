#!/usr/bin/env bash
# 综合对比实验与消融流水线
# 使用 comprehensive_small 配置，启用参数匹配安慰剂、容量匹配消融、原始参数化对照
# 以及推理期剂量-反应与因果病灶
set -euo pipefail

CONFIG="${1:-configs/comprehensive_small.json}"
DATA_DIR="${2:-data/wikitext103}"

echo "=== 综合对比实验与消融 ==="
echo "配置: $CONFIG"
echo "数据: $DATA_DIR"

# 1) 数据准备
python scripts/prepare_data.py --out_dir "$DATA_DIR"

# 2) 全流水线：启用参数匹配安慰剂、容量匹配、原始参数化对照
python run_all.py \
  --config "$CONFIG" \
  --data_dir "$DATA_DIR" \
  --param_control \
  --capacity_ablation \
  --raw_factorial \
  --tune_baseline \
  --tune_seeds 1337,2024,42 \
  --ablate_variants lega,qgate,gate_only \
  --dose_response \
  --dose_variants lega \
  --full_eval

echo "完成。结果在 results/ 下，汇总表 results/*/summary.md"
