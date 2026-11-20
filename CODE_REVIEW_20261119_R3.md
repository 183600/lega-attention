# Deep Code Review Report - lega-attention (Round 3)

**Date:** 2026-11-19  
**Reviewer:** Automated Deep Review Agent (Round 3 - Independent Sub-Agent)  
**Scope:** Full codebase (95 Python files) + scientific methodology audit  

## Executive Summary

This is the third independent review pass of the lega-attention codebase. The review focused on:
1. Custom autograd function correctness (gradient flow, numerical stability)
2. Attention mechanism implementation (causal masking, gating logic, RoPE)
3. Statistical methods (bootstrap CI, p-value correction, division safety)
4. Training loop correctness (gradient accumulation, loss normalization, eval/train separation)
5. Edge case handling (empty sequences, NaN/Inf, degenerate distributions)

## Results

| Category | Count |
|---|---|
| Files scanned | 95 |
| Static analysis flags | 40 |
| Previously fixed bugs | 2 |
| New bugs found | 0 |
| False positives | 38 |
| Methodology concerns | 0 |

## Code-Audit Details

### 1. Attention Mechanism (`lega/attention.py`, 1993 lines)

**Causal masking**: Correctly implemented with `torch.ones(...).triu_(1)` upper-triangular mask. Verified that null slot handling properly includes the null key in the softmax denominator via `lse(a·w_null)`.

**Gating logic**: The evidence-gated output mechanism `g = sigmoid(w·q + a*c + b*log n + d)` is correctly implemented:
- Query content `q_content` uses pre-RoPE query vectors (correct: avoids position confound)
- Concentration `c = log(n * max_j p_j)` with proper length normalization
- Initial bias `d=2.0` gives `g ≈ 0.88` for stable training startup
- All 18 ablation variants properly implement their declared factor structure

**RoPE**: Proper rotary position embedding with dtype-aware caching and version-tracked invalidation.

**KV cache**: Static cache path correctly uses `kv_cache[:, :, cache_pos:end]` with `index_copy_` semantics.

### 2. Custom Autograd Functions

**Chunked cross-entropy (`_ChunkedCrossEntropy`)**: Verified gradient correctness:
- Forward: `total / count.clamp(min=1)` — safe division
- Backward: `grad = (softmax - one_hot) / count` for valid positions, `0` for ignored
- `probs[~valid[start:end]] = 0.0` correctly zeroes ignored-position gradients

**Fused linear-cross-entropy (`_FusedLinearCrossEntropy`)**: 
- Weight gradient correctly accumulates via `addmm_` across chunks
- Mixed-dtype handling (bf16/fp16 → fp32 accumulator) is correct
- `ignore_index` masking properly zeroed in both hidden and weight gradients

### 3. Training Loop (`lega/train.py`, 2294 lines)

**Loss normalization**: `val_loss = float(loss_acc.item()) / total_w` with `total_w` guarded — correct.

**Window-based CI**: Uses t-distribution critical values for small-sample confidence intervals (appropriate for `eval_iters=20` windows).

**Gate statistics**: Welford-style accumulation for mean/variance, with proper `gate_w_total` denominator guard.

### 4. Statistical Methods

**Bootstrap**: `_window_bootstrap_stats()` with percentile CI at [2.5, 97.5] — standard 95% CI.

**P-value correction**: G-test with permutation-based p-values using `(1 + exceed) / (n_sim + 1)` — correct +1 correction for conservative estimation.

**Division safety**: All divisions use `clamp_min(1)`, `max(total, 1)`, ternary guards, or validated denominators. No unprotected divisions found.

### 5. Previously Fixed Issues (Confirmed Still Fixed)

1. **sanity.py:617** — `loss.backward()` properly wrapped in `with torch.enable_grad()`
2. **analysis.py:1330** — Mask covers all records (`valid_mask.copy()`), not just valid subset
3. **missingness.py** — Consistent `max(total, 1)` pattern in `_convex()`

### 6. Edge Cases Verified

- **Empty sequence (T=0)**: Returns empty tensors without executing projection/RoPE/softmax
- **All targets ignored**: Returns `loss.sum() * 0.0` (zero loss, no NaN)
- **Single position (T=1)**: Skips mask construction (all keys visible by causality)
- **NaN/Inf handling**: Comprehensive `isfinite()` checks with triage diagnostics

## Conclusion

The lega-attention codebase demonstrates exceptional engineering quality. The evidence-gated attention mechanism is correctly implemented with proper ablation variant coverage. All statistical methods follow rigorous methodology. No new bugs or scientific methodology issues were found in this third independent review pass.

**Recommendation:** Codebase is production-ready. No changes required.
