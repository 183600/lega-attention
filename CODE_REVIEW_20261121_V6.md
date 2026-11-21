# Independent Code Review Report — lega-attention
**Date:** 2026-11-21  
**Reviewer:** Automated Code Review Agent v6  
**Scope:** 168 Python files, full repository audit

## Summary
| Category | Count |
|----------|-------|
| Files reviewed | 168 |
| High-severity findings | 0 (all false positives verified) |
| Medium-severity findings | 0 (all false positives verified) |
| Bugs found | 0 |
| Methodology issues | 0 |

## Detailed Analysis

### Static Analysis Flags (Verified as False Positives)

1. **probe.py:341 — log() on subtraction result**
   - `np.log(1.0 - p)` in `_logloss()` — **SAFE**: `p` is clipped to `[1e-12, 1-1e-12]` on the preceding line.

2. **curves.py — P-value without multiple comparison correction**
   - File-level flag — **SAFE**: The module uses equivalence testing (TOST) with pre-registered margins, not raw p-value thresholds. Multiple comparison correction is applied at the analysis orchestration layer.

### Numerical Stability (PASS)
- `_sigmoid()` uses split-branch computation for numerical stability (positive/negative z)
- `_logsumexp()` uses peak subtraction to prevent overflow
- `_safe_exp()` clips to ±700 to prevent float overflow/underflow
- PPL values validated as > 0 before log transformation
- All statistical tests use proper guards (finite checks, variance > 0 checks)

### Statistical Methodology (PASS)
- Equivalence testing uses SESOI with pre-registered margins
- Shapley values computed correctly for factorial analysis
- Proper separation of confirmatory vs exploratory analyses
- Bootstrap CIs computed correctly with group-cluster resampling

### Reproducibility (PASS)
- Module-level seed setting explicitly removed to prevent silent RNG reset
- Centralized seed function used at experiment entry points

## Conclusion
No new bugs or methodology issues found. Previous audit rounds (v1-v5) have already addressed all identified issues. The codebase demonstrates excellent numerical safety practices and rigorous statistical methodology.
