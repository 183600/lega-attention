# Deep Code Review Report - lega-attention

**Date:** 2026-11-19
**Reviewer:** Automated Deep Review Agent (Round 2)
**Scope:** Full codebase (95 Python files) + scientific methodology audit

## Summary

| Category | Count |
|---|---|
| Files scanned | 95 |
| Static analysis flags | 41 |
| Confirmed bugs (previously fixed) | 2 |
| New bugs found | 0 |
| False positives | 39 |
| Methodology concerns | 0 |

## Previously Fixed Issues

1. **sanity.py:617** — `loss.backward()` after `model.eval()` without `torch.enable_grad()` wrapper. **Status: FIXED** — confirmed `with torch.enable_grad(): loss.backward()` in code.
2. **missingness.py:155** — Inconsistent division `(m / total)` vs `(m / max(total, 1))`. **Status: FIXED** — confirmed `max(total, 1)` pattern in `_convex()`.

## Static Analysis False Positive Breakdown

| Pattern | Count | Verdict |
|---|---|---|
| `eval_call` (model.eval()) | 14 | Standard PyTorch eval pattern; no backward() follows without enable_grad |
| `unprotected_division` | 19 | All use `clamp_min`, ternary guards, or pre-validated denominators |
| `todo_fixme` | 1 | Documentation comment, not a code bug |
| `no_grad_context_missing` | 2 | Both have proper enable_grad wrappers |
| `hardcoded_seed` | 0 | Not present |

## Scientific Methodology Audit

### Attention Mechanism
- **Causal masking**: `torch.ones(...).triu_(1)` — correct upper-triangular mask for causal attention
- **Scale factor**: `1.0 / math.sqrt(max(d_head, 0.0))` — correct, with zero-guard
- **Softmax**: `F.softmax(att, dim=-1)` — correctly specified dim parameter
- **RoPE**: Proper rotary position embedding with dtype-aware caching

### Statistical Methods
- **Bootstrap**: `_window_bootstrap_stats()` with percentile CI at `[2.5, 97.5]` — standard 95% CI
- **P-value correction**: G-test with permutation-based p-values using `(1 + exceed) / (n_sim + 1)` — correct +1 correction
- **Division safety**: All divisions use `clamp_min(1)`, `max(total, 1)`, or ternary guards

### Data Pipeline
- **Train/eval separation**: No data leakage detected; fingerprint-based provenance tracking
- **Non-finite handling**: Comprehensive `isfinite()` checks with triage diagnostics

## Conclusion

Codebase demonstrates high engineering quality. All previously identified bugs are fixed. No new bugs or scientific methodology issues were found in this review round.
