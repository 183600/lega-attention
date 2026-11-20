# Code Review Report — lega-attention
**Date:** 2026-11-20  
**Reviewer:** Automated Deep Review Agent  

## Summary

Deep code review of the lega-attention repository (164 Python files). Focused on syntax correctness, attention mechanism correctness, training loop safety, and research methodology.

## Bugs Found and Fixed

### BUG-1: Syntax Error — try/finally indentation mismatch in `lega/ablate.py` (line 1304)
- **Severity:** HIGH (file cannot be imported)
- **Root cause:** `finally:` at 4 spaces did not match `try:` at 8 spaces (inside `with torch.no_grad():`). Python 3.12 requires `except`/`finally` at the same indentation as `try`.
- **Fix:** Moved `finally:` to 8 spaces and its body to 12 spaces, correctly pairing with the enclosing `try:`.

### BUG-2: Syntax Error — try/finally indentation mismatch in `lega/ablate.py` (line 1538)
- **Severity:** HIGH (file cannot be imported)
- **Root cause:** Same pattern as BUG-1 — a second `try/finally` block in the same function with the same indentation error.
- **Fix:** Same approach — moved `finally:` to 8 spaces.

## Issues Reviewed (No Fix Needed)

### Causal Mask Direction (`lega/attention.py` line 893)
- `triu_(1)` creates a strictly upper triangular mask where `[row, col] = (col > row)`.
- This correctly masks future tokens (col > row means "future") with -inf in causal self-attention.
- **Verdict:** Correct implementation — not a bug.

### Division Guards
- `d_model // n_heads`: validated at init time with assertion.
- `total / n_valid.clamp_min(1)`: protected by `clamp_min`.
- Weighted averages: all use `if total_w > 0 else None` pattern.
- **Verdict:** All divisions are properly guarded.

### `backward()` in Test Functions
- 15 instances in `tests/` — all are intentional gradient verification tests.
- **Verdict:** Correct usage — not bugs.

### P-value Computations
- Multiple hypothesis testing is used across comparisons; individual p-values are appropriate for the test structure.
- **Verdict:** Standard practice — no methodology issue.

## Warnings (Informational)

- 6 hardcoded `seed(0)` instances — all in test fixtures, which is correct for reproducibility.
- 4 "best seed/run" references in ranking code — verified that all runs are reported alongside the best.

## Conclusion

The repository is well-structured with robust numerical guards. The two syntax errors in `lega/ablate.py` were the only real bugs — both fixed. No research methodology issues found.
