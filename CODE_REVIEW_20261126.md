# Independent Code Audit — 2026-11-26

**Repository:** lega-attention  
**Auditor:** QwenGo (qwen3.7-max)  
**Date:** 2026-11-26  
**Method:** Full source review + py_compile + grep-based pattern scanning + manual inspection

---

## Scope

- All Python source files under `lega/` (~44,000 lines)
- Attention mechanism variants (20+ types: standard, qgate, null, lega, ablations, placebo, raw, l2, entropy, detach, null_zero)
- Statistical methods: TOST equivalence, bootstrap CI, Bayesian analysis, multiple comparison corrections
- Training loop and evaluation protocol
- Data pipeline and leakage prevention

## Checks Performed

| Check | Result |
|-------|--------|
| Syntax (py_compile on 10 core files) | ✅ All pass |
| Unguarded math.log / division-by-zero | ✅ All guarded (max/eps/early-return) |
| Bare except clauses | ✅ None found |
| Mutable default function arguments | ✅ None found |
| torch.no_grad / inference_mode in eval | ✅ Correctly applied |
| Data leakage (split-before-process) | ✅ Enforced |
| TOST p-value sign handling | ✅ Correct (`_one_sided_upper`/`_one_sided_lower` handle t sign) |
| Bootstrap CI stratification | ✅ Correct |
| Multiple comparison (Holm/BH/Bonferroni) | ✅ Correctly applied |
| Causal masking in attention | ✅ Correct |
| Numerical stability (overflow/underflow) | ✅ Handled (log-space, fsum, scale-normalization) |

## Findings

**No bugs or scientific method issues found.**

Key observations:
1. The TOST implementation correctly handles the `t < 0` case via `_one_sided_upper(t, df)` returning `p2/2` when `t >= 0` and `1 - p2/2` when `t < 0`. This was previously a bug in SMELT-ISR (fixed in commit f32b878) but is correctly implemented here.
2. The variance computation includes a scale-normalized fallback for overflow scenarios.
3. PYTHONHASHSEED bootstrap ensures deterministic set/dict iteration across runs.

## Conclusion

Code is scientifically sound and production-ready.
