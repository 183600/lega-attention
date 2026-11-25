# Code Review v9 — 2026-11-25 (QwenGo independent parallel review)

## Scope

Full source review of `lega/` package core modules:
- `attention.py` (causal self-attention, all gating variants, numerical stability)
- `bayes.py` (JZS Bayes factor, Simpson quadrature, posterior computation)
- `equivalence.py` (TOST, SESOI, sensitivity grid)
- `bootstrap.py` (PYTHONHASHSEED bootstrap for set/dict determinism)
- `analysis.py` (gate behavior mechanism analysis, AUROC/PR-AUC, calibration)
- `utils.py` (statistical tests, random seed management, metric computation)

## Findings

### Bugs Found: **0**

### Methodology Issues: **0**

## Detailed Verification

### 1. Attention Mechanism (attention.py)
- ✅ All 20+ gating variants correctly implemented with proper factor activation tables
- ✅ Numerical stability: `clamp_min(1e-30)` on concentration statistics before `log()`
- ✅ Null slot uses analytic merging (`p_null = sigmoid(a_n - lse)`) — numerically stable
- ✅ Causal masking: proper two-stage approach (prefix truncation + diagonal band mask)
- ✅ Log-n table: device-level prefix table with doubling expansion, correct indexing
- ✅ Placebo evidence: deterministic hash-based features, no RNG consumption
- ✅ KV cache: both dynamic (`past_kv`) and static (`kv_cache`) properly handled
- ✅ Gradient checkpointing: `preserve_rng_state=False` safe (no dropout in chunked path)
- ✅ Empty sequence (T=0): explicit guard returns empty tensors without crash

### 2. Bayes Factor (bayes.py)
- ✅ Cauchy-to-delta transform: `u = (i+0.5)/n_delta` avoids tan(±π/2) endpoints
- ✅ Log-likelihood computation: proper logsumexp for marginal likelihoods
- ✅ Zero-variance handling: explicitly flagged as `degenerate_zero_variance`
- ✅ Posterior quantiles: weighted CDF with proper interpolation
- ✅ Prior sensitivity: 0.5×/1×/2× grid, verdict aggregation correct
- ✅ ROPE integration: standardized scale comparison avoids overflow

### 3. TOST Equivalence (equivalence.py)
- ✅ Paired diffs: proper scale transformation (relative, log_ratio, absolute)
- ✅ One-sided p-values: correct direction mapping (`_one_sided_upper`/`_one_sided_lower`)
- ✅ Variance overflow protection: scale-by-max-deviation fallback
- ✅ Cohen's d: `std=0` returns None (not inf/nan)
- ✅ Sensitivity margin: `_margin_for_factor` handles log_ratio in log-space
- ✅ Status priority: `insufficient_seeds → equivalent → significant → noninferior → undetermined`

### 4. Statistical Methods (utils.py)
- ✅ Paired t-test with Holm correction
- ✅ Sign-flip randomization test (exact enumeration for n≤16)
- ✅ Bootstrap CI (percentile method, BCa companion)
- ✅ MDE calculation at 80% power
- ✅ Wilson score CI for binary metrics
- ✅ Non-deterministic op warning counter (wraps `warnings.warn`)
- ✅ CUBLAS workspace config enforcement
- ✅ PYTHONHASHSEED audit at import time (prevents self-confirmation bias)

### 5. Data Leakage Prevention
- ✅ Split-then-process protocol
- ✅ Character-level vocabulary fitted only on train
- ✅ Cross-split decontamination (exact + approximate via MinHash/LSH)
- ✅ Test set access logging
- ✅ Document boundary preservation
- ✅ Group-disjoint verification at 3 levels (document, paragraph, line)

### 6. Multiple Comparison Control
- ✅ Holm-Bonferroni for confirmatory families (FWER)
- ✅ BH/BY for exploratory discoveries (FDR)
- ✅ Separate families: confirmatory vs exploratory explicitly delineated
- ✅ Pre-registration scope verification (config/data/tokenizer must match)

## Verdict

**All reviewed code is correct.** No bugs, no methodology issues, no numerical instability risks found. The codebase demonstrates exceptional engineering rigor with comprehensive statistical safeguards, proper error handling, and thorough documentation of design boundaries.
