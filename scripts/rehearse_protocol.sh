#!/usr/bin/env bash
# 协议预演（rehearsal）：在动用正式算力之前，把"正式实验的每一步"先在真实
# 数据上以最小规模走一遍，并把该走的闸门全部走一遍。
#
# 正式实验要花 100~300 GPU·小时；如果数据协议、设计计划、run 顺序、评测口径
# 或报告渲染里有任何一个问题，等到跑完几十个 run 才发现，代价是整轮算力。
# 预演把这条失败路径提前到几分钟的 CPU 时间里。
#
# 四个阶段（任一阶段失败即整体失败）：
#   1) 数据准备：真实数据目录（下载/切分/去重/分词/协议报告）；
#   2) 设计计划：正式配置 + 真实数据 → 先验设计计划（功效、分析人群、种子
#      分区、run 顺序、测试集尘封、数据身份绑定）；退出码 2 表示设计 fail；
#   3) 端到端流水线：tiny 配置 + 真实数据 + smoke，验证训练→评测→RULER→
#      机制分析→汇总全链路；
#   4) 算力预算预检：用实测单步耗时估算正式网格的总开销，并与声明预算比较。
#
# 用法:
#   bash scripts/rehearse_protocol.sh [DATA_DIR] [CONFIG]
#   DATA_DIR 默认 data/wikitext103；CONFIG 默认 configs/small.json
set -euo pipefail
cd "$(dirname "$0")/.."
export MPLBACKEND=Agg
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$(nproc 2>/dev/null || echo 4)}"

DATA_DIR="${1:-data/wikitext103}"
CONFIG="${2:-configs/small.json}"
REHEARSAL_DIR="results/rehearsal"

echo "=== 协议预演 ==="
echo "数据目录: $DATA_DIR"
echo "正式配置: $CONFIG"
echo "产物目录: $REHEARSAL_DIR"

echo
echo "=== [1/4] 数据准备（真实数据，幂等） ==="
python scripts/prepare_data.py --out_dir "$DATA_DIR"

echo
echo "=== [2/4] 先验设计计划（正式配置 + 真实数据） ==="
mkdir -p "$REHEARSAL_DIR"
# 退出码 2 = 设计 fail（闸门不通过）；设计不过就不该开跑，这里直接失败。
python -m lega.design \
  --config "$CONFIG" \
  --data_dir "$DATA_DIR" \
  --out "$REHEARSAL_DIR/design_plan.json"

echo
echo "=== [3/4] 端到端流水线（tiny + 真实数据 + smoke） ==="
python run_all.py \
  --config configs/tiny.json \
  --data_dir "$DATA_DIR" \
  --smoke \
  --results_dir "$REHEARSAL_DIR/pipeline"

echo
echo "=== [4/4] 算力预算预检（用预演实测的单步耗时） ==="
# 单步耗时取自预演产物里的 train_step_time_ms（实测值，优于任何 FLOPs 估算）。
python -m lega.design.budget \
  --config "$CONFIG" \
  --results_dir "$REHEARSAL_DIR/pipeline" \
  --out "$REHEARSAL_DIR/compute_budget.json"

echo
echo "=== 预演完成 ==="
echo "设计计划: $REHEARSAL_DIR/design_plan.json"
echo "流水线产物: $REHEARSAL_DIR/pipeline/summary.md"
echo "预算预检: $REHEARSAL_DIR/compute_budget.json"
echo
echo "开跑前请人工核对："
echo "  1) 设计计划 status 不是 fail；warn 项是否可接受；"
echo "  2) 预算预检的估算与实际申请到的算力是否匹配；"
echo "  3) 数据协议报告 data_report.json 没有严重问题；"
echo "  4) 确认性种子与调优/试点种子互不相交。"
