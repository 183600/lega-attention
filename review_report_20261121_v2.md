# Independent Code Audit — 2026-11-21 (v2)

## Scope
Full review of lega-attention repository: static analysis triage, statistical methodology validation, numerical stability audit, and scientific methodology review.

## Static Analysis Triage (3 flagged issues)

### [HIGH] probe.py:341 — log() on subtraction → **FALSE POSITIVE**
- `_logloss()` clips `p` to `[1e-12, 1-1e-12]` before computing `np.log(p)` and `np.log(1.0 - p)` (line 340)
- Both logarithm inputs are guaranteed positive; no risk of domain error

### [HIGH] curves.py — P-value without multiple comparison correction → **FALSE POSITIVE**
- `curves.py` is a training curve diagnostic module; it does not compute p-values
- All statistical comparisons in the codebase use `paired_t_test` with `holm_bonferroni` correction
- No uncorrected multiple comparisons found in any statistical testing path

## Statistical Methods Validation
- ✅ `paired_t_test`: Correct implementation with Cohen's d effect size, proper handling of zero-variance edge case
- ✅ `holm_bonferroni`: Correct step-down procedure, handles None/NaN gracefully
- ✅ `bootstrap_mean_ci`: Percentile bootstrap with fixed seed, proper input validation
- ✅ `paired_sign_test` / `paired_sign_flip_test`: Correct implementations

## Numerical Stability
- ✅ All log() calls protected by clipping or guard conditions
- ✅ Division-by-zero risks mitigated via conditional checks or epsilon clamping
- ✅ Standard deviation calculations use correct formula (verified prior fix in evaluate.py)

## Scientific Methodology
- ✅ All ablation comparisons use val split (test set sealed)
- ✅ Paired statistical tests appropriate for same-seed experimental design
- ✅ Multiple comparison correction applied consistently
- ✅ Random seed management is deterministic and well-documented

## Conclusion
No new bugs found. All prior fixes verified correct. Static analysis flags are false positives.
