# Code Review Report - lega-attention
## Pass 6: Independent Deep Source Code Audit

**Date:** 2026-11-14 05:00 UTC
**Reviewer:** QwenGo Independent Review System (Pass 6)
**Scope:** Full independent source code audit — focus on attention computation,
gradient correctness, data pipeline integrity, and statistical methodology.

---

## Summary

### Verdict: CODE QUALITY CONFIRMED — No new issues found

This pass performed direct source-code reading of all core modules, with
particular focus on:
- `lega/attention.py` (1993 lines) — full forward pass, gate computation,
  mask caching, RoPE, chunked attention, gradient checkpointing
- `lega/data.py` — train/eval batch sampling, document-aligned sampling
- `lega/analysis.py` — statistical analysis
- Previous fix verification (Pass 1: empty-input guard, Pass 5: confirmed)

---

## Detailed Checks Performed

### 1. Attention Forward Pass (attention.py:1515+)
**Status:** ✅ CORRECT

- **Q/K/V projection:** Training uses three independent `F.linear` calls for
  correct autograd; inference uses a fused weight with proper staleness
  detection (identity + `_version` + `data_ptr` + `device` + `dtype`).
- **q_content extraction:** Correctly captured *before* RoPE rotation,
  preventing position information from leaking into the length-calibration
  gate.
- **KV cache:** Both dynamic (`past_kv` with `torch.cat`) and static
  (`kv_cache` with in-place write) paths correctly handled.
- **Empty input guard:** `T == 0` returns empty tensors without computing
  attention — prevents IndexError from max/amax on empty sequences.

### 2. Gate Logit Computation (attention.py:1062-1161)
**Status:** ✅ CORRECT

- **Concentration evidence:** `clamp_min(1e-30)` prevents `log(0)` with
  proper branching for detach/requires_grad paths. In-place `clamp_min_`
  only used when the tensor is not shared (not in computation graph).
- **Null evidence components:** Evidence zeroing uses `* 0.0` (not
  `zeros_like`) to preserve autograd graph for sanity checks.
- **Placebo variants:** Deterministic pseudo-random evidence preserves
  parameter count and computation while destroying information content.
- **addcmul_ accumulation:** In-place `logit.addcmul_(a, c)` correctly
  accumulates gate terms; the initial `logit` is cloned when
  `keep_terms=True` to avoid mutating the original `qterm`.

### 3. Mask Caching System (attention.py:350-480)
**Status:** ✅ CORRECT

- **LRU eviction:** Dual bounds on entry count (256) and total bytes
  (128 MiB for masks, 64 MiB for triangular cache).
- **Inference-mode safety:** Cache constants created with
  `inference_mode(False)` to prevent autograd rejection when reused in
  training forward passes.
- **Device isolation:** Cache keys include device, so cross-device
  sharing is impossible by construction.

### 4. Data Pipeline (data.py:1351-1450)
**Status:** ✅ NO LEAKAGE

- **Train batch sampling:** `rng.integers(0, len(train) - ctx)` with
  proper bounds checking (`len(train) >= ctx + 1`).
- **Document-aligned sampling:** Window placement ensures windows do not
  cross document boundaries; document selection weighted by token count
  to preserve the original mixture ratio.
- **Order hashing:** SHA-256 digest of actual sampling indices enables
  cross-run verification that all variants consumed identical data.
- **Evaluation isolation:** Fixed eval batches, no fit-on-test patterns.

### 5. Previous Fix Verification
**Status:** ✅ CONFIRMED

- Pass 1 commit (cd4454a): Empty-input guard to `_brute_signflip_p` and
  removed inconsistent `(n + 1e-8)` epsilon — verified correct.
- Pass 5 report (85b9cb9): All false positives properly identified and
  documented.

## Conclusion

The lega-attention codebase maintains exceptional quality. The attention
mechanism correctly implements 22+ variants with shared infrastructure,
careful numerical stability guards, and proper gradient flow. The data
pipeline enforces strict train/eval separation. No bugs or methodology
errors were found. No code changes were necessary.
