# Code Review Report - lega-attention (V4)
**Date:** 2026-11-18  
**Reviewer:** Automated Deep Review Agent

## Summary
Deep review of 95 Python source files focusing on bug detection and scientific methodology.

## Fixes Applied

### 1. lega/ablate.py (Line 1311)
- **Issue:** `sums[li] / counts[li]` - potential division by zero when counts is zero
- **Fix:** Changed to `sums[li] / max(counts[li], 1e-10)` for defensive programming
- **Context:** While outer `if` already checks `counts[li] <= 0`, added explicit guard for robustness

### 2. lega/positional.py (Line 163)
- **Issue:** `total_nll / total_valid` - potential division by zero
- **Fix:** Changed to `total_nll / max(total_valid, 1e-10)` for defensive programming
- **Context:** While outer `if total_valid <= 0.0` already guards, explicit guard adds safety

### 3. lega/factorial.py (Line 231)
- **Issue:** `share = f / total` in loop - relies on try/except for ZeroDivisionError
- **Fix:** Changed to `share = f / total if total != 0 else 0.0` - avoids exception-based control flow
- **Improvement:** Better practice than catching ZeroDivisionError

## False Positives from Static Analysis (Not Bugs)
- `model.eval()` calls: Standard PyTorch eval mode - correct usage
- `d_model // n_heads`: Validated by config_schema.py assertions
- Most `/total` divisions: Already have proper guards (clamp_min, if-checks)
- `uncertainty.py:54`: Already guarded by `if total <= 0.0: return None` at line 52

## Scientific Methodology Review
- Cross-validation: Some instances without explicit shuffle/stratification (doc_strata.py)
- Statistical tests: Present; recommend adding multiple comparison corrections
- Data splitting: Properly handled with seed management
- Ablation study: Comprehensive and well-designed
- Uncertainty quantification: Properly implemented

## Recommendations
1. Add shuffle=True to cross-validation in doc_strata.py
2. Consider Bonferroni/FDR correction for multiple statistical tests
3. Document significance threshold choices
