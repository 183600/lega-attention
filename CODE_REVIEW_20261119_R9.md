# Code Review Round 9 — lega-attention

**Date:** 2026-11-19  
**Reviewer:** Independent deep code review agent  
**Scope:** Full repository (255 Python files)

## Summary

**Result: No bugs found.**

This is the 9th independent review round. After thorough examination of all static analysis flags and manual deep review of core scientific code, no actionable bugs were identified.

## Static Analysis Review

40 issues were flagged by static analysis across 95 scanned files. Each category was manually verified:

### 1. eval_call (14 flags) — ALL FALSE POSITIVES
- All instances are `model.eval()` calls in standard PyTorch evaluation patterns
- Properly paired with `torch.no_grad()` or `torch.inference_mode()` contexts
- No instances of evaluation without proper gradient disabling

### 2. unprotected_division (16 flags) — ALL FALSE POSITIVES
- Every flagged division has proper guards:
  - `clamp_min(1)` for tensor divisions
  - `if total else` / `if total > 0 else` for scalar divisions
  - `try/except ZeroDivisionError` where appropriate
- Examples verified: `uncertainty.py:54` has full guard chain, `capacity.py:139` checks for None and zero

### 3. no_grad_context_missing (2 flags) — ALL FALSE POSITIVES
- `sanity.py:618`: `loss.backward()` is inside `torch.enable_grad()` context — correct
- `train.py:1931`: `loss.backward()` is in the training loop where gradients are expected — correct

### 4. todo_fixme (1 flag) — NOT A BUG
- `analysis.py:1330`: Comment documents a previously fixed bug (mask covering all records). The fix is already in place using `valid_mask.copy()`.

## Deep Manual Review

### Core Attention Mechanism (attention.py)
- **Gate formulas**: All 20+ variants (lega, lega_no_len, lega_no_conc, etc.) are mathematically consistent
- **Concentration metric**: `c = log(n * max_j p_j)` correctly normalizes for sequence length
- **Causal masking**: Uses `triu_(1)` correctly for autoregressive masking
- **Fused weight caching**: Proper invalidation based on parameter identity, _version, data_ptr, device, dtype

### RoPE Implementation (model.py)
- Rotate-half style rotation is mathematically correct
- Forward: `o1 = x1·cos - x2·sin`, `o2 = x2·cos + x1·sin` ✓
- Backward: Proper adjoint computation ✓
- Custom autograd Function with correct gradient flow

### Training Pipeline (train.py)
- Gradient accumulation: loss properly divided by `grad_accum` before backward
- Mixed precision: Proper AMP context management
- Gradient clipping: Pre-clip norm tracked for diagnostics
- Non-finite detection: Interval-based tracking with first-bad-step recording

### Evaluation Pipeline (evaluate.py)
- Proper `model.eval()` + `torch.no_grad()` for all evaluation
- Checkpoint selection metadata properly recorded for auditability

### Data Integrity (data.py)
- N-gram overlap contamination checks between train/test splits
- Multiple contamination metrics (ngram, short_ngram, exact)

### Statistical Methods
- Wilson confidence intervals for proportions
- Bootstrap uncertainty estimation with window-level resampling
- Proper multiple testing correction (Benjamini-Hochberg)

## Conclusion

The codebase demonstrates high engineering quality with:
- Comprehensive numerical guards on all divisions
- Proper gradient context management
- Sound scientific methodology
- Extensive error handling and diagnostic instrumentation

No code changes required.
