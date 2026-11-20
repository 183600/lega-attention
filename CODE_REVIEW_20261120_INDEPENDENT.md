# Independent Code Review Report — lega-attention

**Date:** 2026-11-20 03:05 (Asia/Shanghai)  
**Reviewer:** Independent Code Review Agent (Qwen)  
**Scope:** Full codebase (95 Python source files), focusing on bugs and scientific methodology

## Executive Summary

This independent review confirms the codebase is of **high quality** with no remaining bugs or methodological issues requiring code changes. Previous 20+ review rounds have been thorough and accurate.

## Review Methodology

1. **Static analysis**: Scanned all 95 Python files with pattern-based detection for division-by-zero, gradient issues, eval/no_grad misuse, exec calls, hardcoded seeds
2. **Deep code review**: Manual inspection of core algorithms (attention mechanism, RoPE, training loop, loss computation, KV cache)
3. **Methodology audit**: Verified scientific correctness of experimental design, statistical methods, and reproducibility measures

## Findings

### Attention Mechanism (lega/attention.py) — ✅ Correct

- Scaled dot-product attention: scale factor `1/sqrt(d_head)` correctly computed
- Gate mechanism: sigmoid gating with proper numerical handling
- All variant implementations (lega, qgate, null, standard, ablation variants) follow consistent patterns
- Factorial design matrix (_USE_QUERY/_USE_CONC/_USE_LEN) is self-consistent

### RoPE Implementation (lega/model.py) — ✅ Correct

- Custom autograd function `_RoPERotate`: forward and backward passes are mathematically consistent
  - Forward: o1 = x1·cos - x2·sin, o2 = x2·cos + x1·sin
  - Backward: gx1 = go1·cos + go2·sin, gx2 = go2·cos - go1·sin ✅
- Half-width rotation: correctly handles head_dim/2 split
- Max length check prevents out-of-bounds position encoding
- Device/dtype migration cache properly invalidates

### Training Loop (lega/train.py) — ✅ Correct

- Gradient accumulation: `loss = out["loss"] / grad_accum` before backward — standard and correct
- Gradient clipping: pre-clip norm preserved for diagnostics
- Loss accumulation uses `clamp_min(1)` to prevent division by zero
- eval()/no_grad() usage is correct throughout

### Loss Computation (lega/model.py) — ✅ Correct

- `_ChunkedLinearCE`: chunked cross-entropy with closed-form gradient
- Empty target handling: returns zero loss (consistent with ignore_index semantics)
- `n_valid.clamp_min(1)` prevents division by zero
- Window losses computed independently for validation diagnostics

### Statistical Methods — ✅ Sound

- Bootstrap confidence intervals at correct confidence levels
- Multiple comparison correction (Holm-Bonferroni, BH-FDR) implemented
- Position-wise NLL analysis with proper normalization
- Uncertainty quantification with weighted aggregation and proper guards

### Data Handling — ✅ No Leakage

- No train/validation data mixing detected
- Proper separation of training and evaluation code paths
- Seed management uses explicit Generator objects (not global state)

### Division Operations — All Guarded

All 35 flagged division operations were verified:
- Integer divisions (`//`): validated by upstream `d_model % n_heads == 0` checks
- Float divisions: protected by `clamp_min`, `if total`, or `if n > 0` guards
- No unprotected division found that could produce NaN/Inf in normal operation

## Conclusion

**No bugs found. No code changes required.** The codebase demonstrates excellent defensive programming, sound scientific methodology, and thorough documentation. The factorial ablation design (2^3 matrix + leave-one-out) is particularly well-executed for causal attribution.
