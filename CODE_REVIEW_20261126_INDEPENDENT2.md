# Code Review Report — 2026-11-26 (Independent Review)

**Reviewer**: QwenGo! Code Review Agent  
**Repository**: lega-attention  
**Scope**: Full code review of all Python modules (attention, model, analysis, pipeline, tests)

## Summary

**Verdict: No new bugs or scientific methodology issues found.**

This is an independent review conducted after Round 6. All previously identified fixes (clamp_min artifact removal, tolerance adjustments) are confirmed in place.

## Areas Reviewed

### 1. Core Attention Mechanism (`lega/attention.py`)
- **Gate logit computation**: All variant formulas (standard, qgate, null, lega, ablation variants) are mathematically correct and consistent with the docstring specifications.
- **In-place operations**: Verified that all in-place ops (`mul_`, `add_`, `masked_fill_`) are applied only to tensors that don't participate in autograd backward where mutation would be unsafe. The `logit.sigmoid_()` call is safe because `logit` is a newly allocated tensor in the gated path.
- **Mask cache correctness**: The `_MASK_CACHE` LRU eviction and `_LOGN_TABLE` doubling strategy are correct. Cache invalidation keys include device, past_len, T, S to prevent cross-context stale reads.
- **Null attention variant**: The analytic merge (`p_null = sigmoid(a_n - lse)`) is mathematically equivalent to the explicit `S+1` column softmax, confirmed by the `_attend_q_chunk` implementation.

### 2. Analysis Module (`lega/analysis.py`)
- **Statistical methodology**: AUROC/PR-AUC computation, bootstrap CI, Brier score calibration, and linear probe baselines all follow standard statistical practices.
- **No data leakage**: The analysis pipeline correctly separates train/val/test data. Hit labels are derived from gold answer spans, not from model predictions.

### 3. Pipeline & Training
- **Cross-entropy loss**: Correctly implemented with ignore_index masking.
- **Gradient checkpointing**: Properly configured with `use_reentrant=False` and `preserve_rng_state` based on dropout presence.

### 4. Recent Fixes Verified
- `.clamp_min(1e-10)` artifact removal from comments/strings: ✅ Correctly applied
- Floating-point tolerance relaxation in uniform attention test: ✅ Appropriate

## Conclusion

The codebase is well-engineered with extensive defensive programming, thorough documentation, and comprehensive test coverage. No new issues identified.
