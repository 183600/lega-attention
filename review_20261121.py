"""Independent deep code review — 2026-11-21

Review scope:
  - Code correctness (bugs, edge cases, numerical safety)
  - Research methodology (statistical validity, data leakage, multiple comparisons)
  - Model state management (train/eval transitions, gradient flow)

Method:
  1. pycodestyle E9/F static scan
  2. Automated pattern search for common bug classes
     (unguarded log/exp/sqrt, division by zero, index errors,
      unchecked array access, model state leaks)
  3. Manual review of flagged items and core algorithmic code
  4. Research methodology audit (Wilson CI, SD calculation,
     multiple comparison correction, data leakage, p-value handling)

Results:
  - Static scan: 2 priority items flagged, both verified as FALSE POSITIVES
    * probe.py:341 logloss — protected by np.clip(p, 1e-12, 1-1e-12) on line 340
    * curves.py P-value — design-level documentation note, not a code bug
  - Wilson CI (wilson_score_ci): Correct formula, proper edge case handling
  - SD calculation (evaluate.py:446): Uses math.fsum for precision, correct formula
  - Model state (ablate.py:1273,1447): eval() properly set, stale train() removed
  - Multiple comparison: Holm-Bonferroni correction implemented in badcase.py
  - Data leakage: train/val/test isolation enforced in datacheck module
  - Numerical safety: All log/exp/sqrt operations properly guarded with
    clip/maximum/epsilon throughout the codebase
  - Gradient flow: Proper use of detach(), no_grad(), inference_mode

Conclusion: No new bugs or research methodology issues found.
All previously identified issues (SD max→sum, np.finfo TypeError,
model state restoration) have been correctly fixed in prior commits.
"""
