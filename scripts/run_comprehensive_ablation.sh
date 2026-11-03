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
# 调优种子必须与确认性种子不相交：否则基线超参是在同一批种子上挑出来的
# （winner's curse），基线会获得一份并非方法带来的优势，“优于已调优基线”
# 的主张不可判定。这里用 10000000 起的开发种子段（与配置里的 8 个确认性
# 种子完全不相交）；设计计划会把三段种子写进 seed_partition 供审计。
python run_all.py \
  --config "$CONFIG" \
  --data_dir "$DATA_DIR" \
  --param_control \
  --capacity_ablation \
  --raw_factorial \
  --tune_baseline \
  --tune_seeds 10000000,10000001,10000002 \
  --ablate_variants lega,qgate,gate_only \
  --dose_response \
  --dose_variants lega \
  --full_eval

echo "完成。结果在 results/ 下，汇总表 results/*/summary.md"
