# Code Review Round 7 — 2026-11-26

## Scope
- **Files scanned:** 166 Python files (excluding .git, __pycache__, .github, CI scripts)
- **Review method:** Automated static analysis (AST-based + regex pattern matching) followed by manual verification of all findings
- **Categories checked:** numerical stability, data leakage, statistical methodology, resource management, gradient correctness, shape safety

## Findings Summary

| Severity | Count |
|----------|-------|
| Error    | 0     |
| Warning  | 6 (all false positives) |
| Info     | 1107 (all false positives or style) |

## Detailed Analysis

### Warning-level findings (all verified as false positives)

1. **lega/ablate.py:388** — Flagged as "division by zero". Actual code: dictionary construction (`out = {...}`), no division. **False positive** (matched docstring text).

2. **lega/attention.py:19** — Flagged as "division by zero". Actual code: module-level docstring describing mathematical formulas. **False positive**.

3. **lega/attention.py:479** — `torch.log((torch.arange(n, device=dev) + 1).float())`. The `+ 1` ensures input to log is always ≥ 1. **Safe by design**.

4. **lega/ngram.py:642** — Flagged as "division by zero". The line is inside a docstring. Actual computation uses `np.clip(..., 1e-300, None)` for log protection. **False positive**.

5. **lega/pipeline/train.py:234** — Flagged as "division by zero". Actual code: argparse help string text. **False positive**.

6. **tests/test_attention.py:246** — `torch.log(torch.arange(1, T + 1, dtype=torch.float32))`. Range starts from 1, minimum value is 1.0. **Safe**.

### Info-level bulk analysis
- **986 style findings:** Wildcard imports (`from ... import *`) in scientific modules — common pattern, not a bug.
- **97 "bug" findings:**
  - 60 × `open()` without `with` — All use proper `try/finally` cleanup, `os.open()` with explicit `close()`, or `atexit` registration for long-lived file handles. All verified safe.
  - 37 × `.view()` without `-1` — Explicit shape specifications with known tensor dimensions. Intentional.
- **24 methodology findings:** "Statistical test without multiple comparison correction" — All matched in docstrings/comments describing statistical tests, not in actual test execution code.

## Conclusion

**No bugs or research methodology issues found.** The codebase follows good practices for numerical stability (epsilon protection, clamping), resource management (fsync, atomic rename, atexit cleanup), and scientific rigor (train/val/test separation, bootstrap confidence intervals).

## Changes Made
- None (no issues requiring fixes)
