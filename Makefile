# 统一的工程入口：把“提交前必须过的检查”收敛到一处，避免每个人各跑一套。
# 用法：make help 查看全部目标。

PYTHON ?= python
RESULTS_DIR ?= results
REHEARSAL_DATA_DIR ?= data/wikitext103
REHEARSAL_CONFIG ?= configs/small.json

.PHONY: help test test-portable smoke lint format lock-check configs-check verify env-check rehearse pilot-anchor clean-results

help: ## 显示可用目标
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-16s %s\n", $$1, $$2}'

test: ## 完整测试（pytest，CI 使用同一入口）
	$(PYTHON) -m pytest tests/ -q

test-portable: ## 无 pytest 环境的测试入口（tests/run_tests.py）
	$(PYTHON) -m tests.run_tests

smoke: ## 冒烟：单测 + 伪数据 + tiny 全流水线
	bash scripts/run_smoke.sh

lint: ## 静态检查与格式检查（ruff / pre-commit）
	pre-commit run --all-files

format: ## 自动格式化（ruff）
	pre-commit run ruff-format --all-files
	pre-commit run ruff --all-files

lock-check: ## 依赖声明一致性（requirements / lock / CI / pyproject）
	$(PYTHON) scripts/check_requirements_lock.py

configs-check: ## 配置契约校验（configs/*.json 全量）
	$(PYTHON) scripts/validate_configs.py

env-check: ## 当前环境与锁定环境的差异
	$(PYTHON) -m lega.repro --check-env

rehearse: ## 协议预演：真实数据上跑通数据准备/设计计划/端到端流水线/预算预检
	bash scripts/rehearse_protocol.sh "$(REHEARSAL_DATA_DIR)" "$(REHEARSAL_CONFIG)"

pilot-anchor: ## 试点方差锚点（需先有试点结果目录）：PILOT_RESULTS=... make pilot-anchor
	$(PYTHON) -m lega.design.pilot --results_dir "$(PILOT_RESULTS)" --config "$(REHEARSAL_CONFIG)" --out "$(RESULTS_DIR)/pilot_variance.json"

verify: lock-check configs-check test ## 提交前完整校验

clean-results: ## 清空本地产物目录（不触碰仓库内任何受保护文件）
	find $(RESULTS_DIR) -mindepth 1 -not -name '.gitkeep' -delete
