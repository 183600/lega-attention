# Code Review Report - lega-attention
**审查日期**: 2026-11-26
**审查者**: AI Sub-Agent #1 (Automated Static Analysis)


## 1. 静态分析发现的Bug

未发现明显的静态分析bug

## 2. 科研方法论检查

- ✅ 包含多重比较校正 (Bonferroni/FDR)
- ✅ 包含 Bootstrap 方法
- ✅ 包含统计功效分析
- ✅ 包含随机种子控制

## 3. 深度代码审查

- **ablate.py:1355**: 函数 `_match_gate_bias` 缺少返回类型注解
- **ablate.py:1943**: 函数 `main` 缺少返回类型注解
- **analysis.py:881**: 函数 `main` 过长 (87 条语句)
- **analysis.py:881**: 函数 `main` 缺少返回类型注解
- **artifacts.py:574**: 函数 `main` 缺少返回类型注解
- **attention.py:909**: 函数 `__init__` 缺少返回类型注解
- **attention.py:1206**: 函数 `_attend_q_chunk` 缺少返回类型注解
- **attention.py:1329**: 函数 `_attend_chunked` 缺少返回类型注解
- **attention.py:1516**: 函数 `forward` 缺少返回类型注解
- **badcase.py:1742**: 函数 `main` 缺少返回类型注解
- **bayes.py:160**: 函数 `_grid` 缺少返回类型注解
- **bayes.py:341**: 函数 `bayes_paired_evidence` 缺少返回类型注解
- **curves.py:824**: 函数 `diagnose_curve` 过长 (98 条语句)
- **data.py:171**: 函数 `_cache_integrity` 缺少返回类型注解
- **data.py:374**: 函数 `prepare_bins` 过长 (142 条语句)
- **data.py:1388**: 函数 `get_train_batch_doc_aligned` 缺少返回类型注解
- **discovery.py:225**: 函数 `exploratory_fdr` 缺少返回类型注解
- **discovery.py:361**: 函数 `collect_seed_level_tests` 缺少返回类型注解
- **dose_response.py:317**: 函数 `_quadratic_fit` 缺少返回类型注解
- **dose_response.py:708**: 函数 `main` 缺少返回类型注解

*... 及其他 69 个发现*

## 4. 总结

共发现 0 个潜在代码问题，89 个代码结构建议。

---
*审查完成于 2026-11-26 22:46:46.468876*
