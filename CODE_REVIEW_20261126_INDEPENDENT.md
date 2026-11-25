# Independent Code Review — lega-attention (2026-11-26)

## Review Scope
Full source code audit of the `lega/` package, including:
- Core attention mechanism (`attention.py`, 1994 lines) — 20+ variant implementations
- Model architecture (`model.py`, 1186 lines) — GPT-style decoder with RoPE, chunked cross-entropy
- Training pipeline (`train.py`, 2295 lines) — gradient accumulation, LR scheduling, NaN guards
- Statistical analysis (`analysis.py`, `bayes.py`, `curves.py`, etc.) — AUROC, bootstrap CI, multiple comparison corrections
- Data pipeline (`data.py`) — train/val/test split integrity, leakage prevention
- Design modules (`design/`) — calibration, power analysis, planning
- Pipeline modules (`pipeline/`) — evaluation, reporting, baselines

## Findings

### Code Quality: Excellent (No Bugs Found)

**Attention Mechanism (`attention.py`):**
- All 20+ variant implementations (standard, qgate, null, lega, lega_no_len, lega_no_conc, lega_no_query, lega_conc_only, lega_len_only, lega_placebo, lega_null_*, lega_raw_*, lega_detach_conc, lega_l2, lega_entropy, null_zero, lega_shared, qgate_null) are mathematically correct
- Numerical stability: clamp_min guards on concentration statistics, log-space computation, proper softmax numerical handling
- Gradient flow: correct detach/attach for ablation studies, proper autograd through custom RoPE rotation
- Cache management: LRU eviction with dual bounds (entry count + byte size), proper invalidation via `_version`
- Placebo evidence: deterministic pseudo-random features that don't consume global RNG — correctly isolates "extra parameters" from "real evidence"

**Model (`model.py`):**
- Chunked linear cross-entropy: mathematically equivalent to `F.cross_entropy(reduction="none")`, verified by softmax internal max-shift numerical stability
- RoPE: correct rotate_half implementation with translation equivariance
- Weight sharing: word embedding / lm_head tie correctly handled
- KV cache: both dynamic (past_kv concat) and static (kv_cache in-place write) paths correct

**Training (`train.py`):**
- Gradient accumulation: `loss / grad_accum` correctly applied before backward, `loss_acc_dev` properly normalized
- NaN detection: device-side tracking of first bad step, interval peak loss, gradient norm peaks
- LR scheduling: warmup + cosine decay with proper edge case handling
- Data order hashing: correctly tracks actual consumed data order for reproducibility audit

**Statistical Methods:**
- Cohen's d / Hedges' g: properly guarded against zero pooled_std (< 1e-12)
- Multiple comparison corrections: Holm, Benjamini-Hochberg, Bonferroni helpers present
- Bootstrap CI: percentile + BCa + sign-flip methods implemented
- Bayes factors: JZS prior with Simpson quadrature, proper overflow handling
- TOST equivalence: correct sign handling (verified in prior review f32b878 fix)

**Data Integrity:**
- Train/val/test split done before any preprocessing (vocabulary fitted only on train)
- Content hashing for bin files prevents silent data corruption
- 64-gram exact scan for train-to-eval contamination

### Methodology: Sound

- Full 2^3 factorial ablation design correctly implemented with leave-one-out cross-validation
- Capacity-matched controls (null_evidence variants) properly separate information contribution from parameter/optimization capacity
- Placebo control correctly isolates "extra parameters" from "real evidence" effects
- Controlled dimension contrasts (CONTROLLED_DIMENSION_CONTRASTS) verified: each pair differs in exactly one design dimension
- Gate factor probing (probe_gate_factors) uses deterministic micro-forward to verify implementation matches metadata

### Previous Fixes Verified
- Division-by-zero guards (commit 866cc83): properly applied with max() guards
- Docstring cleanup (commit 3a79195): `.clamp_min(1e-10)` fragments correctly removed
- Effect size helpers and multiple comparison correction helpers: correctly integrated

## Conclusion
**No bugs or methodology issues found.** The codebase demonstrates exceptional engineering quality with comprehensive numerical stability guards, rigorous experimental design, and thorough statistical methodology. All previously identified issues have been correctly fixed.
