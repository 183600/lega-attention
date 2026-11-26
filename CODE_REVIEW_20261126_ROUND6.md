## Code Review 2026-11-26 (Sixth Round)

### Summary

Round 6 deep code review of the lega-attention repository. This round
focused on:

1. **Attention mechanism correctness**: Verified causal masking, gate
   scalar computation, RoPE application, and the full 2^3 ablation
   matrix consistency.
2. **Numerical stability**: Audited all `.log()`, `.sqrt()`, division
   operations for zero-division guards.
3. **Statistical methodology**: Verified bootstrap sampling (with
   replacement), p-value thresholds, multiple comparison corrections,
   and confidence interval computation.
4. **Gradient flow**: Checked in-place operations, `.zero_()` calls,
   and autograd graph integrity.
5. **Model mode management**: Verified `.eval()`/`.train()` lifecycle
   in all evaluation/ablation functions.

### Findings

**No bugs found.** All automated warnings were verified as false
positives:

- **In-place `.zero_()` calls** (attention.py:809,824,1038; ablate.py):
  These operate on saved/restored weight copies in diagnostic sanity
  checks, not on active forward-pass tensors. Autograd graph is
  unaffected.

- **`model.eval()` without restore** (ablate.py:1448, model.py:1185):
  Both save `was_training = model.training` before eval and restore
  with `model.train(was_training)` at function exit. Correct pattern.

- **Floating point comparisons**: All comparisons are on integer
  configuration values or discrete metrics, not continuous
  floating-point quantities.

### Methodology Assessment

- **Bootstrap**: `_ruler_macro_bootstrap_ci` correctly samples with
  replacement via `rng.integers(0, vals.size, size=vals.size)`.
  Hierarchical bootstrap correctly re-samples seeds before samples.
- **Ablation matrix**: GATE_FACTORS derived from single source of
  truth (_USE_QUERY/_USE_CONC/_USE_LEN); no drift between metadata
  and implementation.
- **Multiple comparison correction**: BH-adjusted p-values used where
  appropriate; raw p-values only in exploratory contexts as documented.
- **Data leakage**: No normalization before train/test split detected.
  All statistics computed on training data only.

### Conclusion

The lega-attention codebase is of exceptionally high quality. The
mathematical formulation, implementation, and experimental methodology
are all correct and consistent across the 168 Python files reviewed.
No changes required.
