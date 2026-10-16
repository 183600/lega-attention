#!/usr/bin/env bash
# 一键冒烟测试：单元测试 → 伪数据 → 全流水线（tiny 配置，每变体 20 步，CPU 可跑）
# 用法: bash scripts/run_smoke.sh [--wandb]
set -euo pipefail
cd "$(dirname "$0")/.."
export MPLBACKEND=Agg
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$(nproc 2>/dev/null || echo 4)}"

echo "=== [1/3] 单元测试 ==="
python -m tests.run_tests

echo "=== [2/3] 生成伪数据 ==="
python scripts/make_dummy_data.py --out_dir data/dummy

echo "=== [3/3] 全流水线冒烟（tiny, 9 变体 x 20 步） ==="
python run_all.py --config configs/tiny.json --data_dir data/dummy --smoke "$@"

echo "冒烟测试完成: results/tiny_smoke/summary.md"
