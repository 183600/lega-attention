# Independent Code Review Report — lega-attention
**Date:** 2026-11-20 13:07 CST
**Reviewer:** Independent Sub-Agent Review (V3)
**Scope:** Full codebase — 95 Python files — bug detection + research methodology audit

## Executive Summary

| Metric | Value |
|--------|-------|
| Status | **PASS** |
| Real bugs found | 0 |
| False positives from static analysis | 35 |
| Code changes required | 0 |
| Code quality rating | Excellent |

## Analysis Performed

### 1. Custom Autograd Functions
- `_ChunkedCrossEntropy`: Forward/backward mathematically correct. `ignore_index` properly handled in both directions. `count.clamp(min=1)` prevents division by zero when all targets are ignored. Ignored positions receive zero gradient.
- `_FusedLinearCrossEntropy`: Correct gradient computation via GEMM operations. Proper dtype promotion through autocast. Weight gradient accumulation handles mixed precision correctly.

### 2. Attention Mechanism (LEGA)
- Length-calibrated evidence gating: `c = log(n * max_j p_j)` correctly separates RoPE-applied q (for attention scoring) from pre-RoPE q_content (for gate computation). This prevents position information from leaking into the length calibration term.
- Causal masking: Position-based mask `qpos < key_pos` correctly implements causal constraint. Null slot always visible (ones column appended).
- KV cache: Static cache with in-place write and prefix view read. Dynamic cache via torch.cat. Both paths correctly track past_len for RoPE offset.

### 3. Statistical Methodology
- **JZS Bayes Factor** (Rouder et al. 2009): Cauchy prior on standardized effect, Jeffreys prior on variance. Grid integration with log-sum-exp for numerical stability. Zero mass at delta=0 split symmetrically.
- **Bootstrap CI**: Percentile bootstrap with +1 smoothing for two-sided p-values. Non-finite resample detection. BCa intervals computed as companion.
- **Sensitivity analysis**: Prior scale varied by factors to assess robustness.

### 4. Data Integrity
- Train/val/test split with document boundary tracking
- Vocabulary fitted only on training split
- 64-gram contamination scanning between train and eval
- Group-level split exclusivity audit (group/paragraph/line-prefix)

### 5. Training Pipeline
- Gradient accumulation with `loss / grad_accum` scaling before backward
- `zero_grad(set_to_none=True)` before accumulation cycles
- Non-finite loss/gradient detection with ordered NaN triage
- Gradient clipping after accumulation, before optimizer step

### 6. Static Analysis False Positives (35 total)
All 35 flagged issues verified as false positives:
- `eval_call` (13): Standard PyTorch eval mode switching, correctly placed
- `unprotected_division` (17): All have proper guards (clamp, conditional, prior validation)
- `no_grad_context_missing` (2): Inside explicit `enable_grad()` or training loop
- `todo_fixme` (1): Comment documenting already-applied fix
- `exec_call` (2): Intentional experiment composition from local files

## Conclusion
The lega-attention codebase is well-engineered research software. No bugs or methodology issues found. No code changes required.
