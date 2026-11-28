# Independent Code Review — LEGA-Attention

**Reviewer**: QwenGo Code Audit  
**Date**: 2026-11-28  
**Scope**: attention.py, model.py, train.py, analysis.py, ablate.py, equivalence.py, factorial.py, data.py, evaluate.py  

## Verdict: ✅ No Critical Bugs Found

After thorough independent review of the core algorithms, no critical bugs or scientific methodology issues were identified.

### Mathematical Correctness Verified

1. **Gate formula** `g = σ(w·q + a·c + b·log(n) + d)` correctly implemented. The concentration statistic `c = log(n · max_j p_j)` is decomposed as `log(max_p) + log_n` in the code, which is mathematically equivalent.

2. **Null attention analytic marginalization**: `p_null = sigmoid(a_n - lse)` where `lse = logsumexp(scores)` correctly avoids the explicit concatenation of the null slot. The scaling factor `(1 - p_null)` applied to real-key outputs is consistent with the joint softmax interpretation.

3. **Causal masking**: Two-phase approach (truncate unseen keys + diagonal band mask) is mathematically equivalent to a full causal mask while being more memory-efficient.

4. **RoPE application**: RoPE correctly applied to Q/K for attention scoring but the gate query term `w·q` uses pre-RoPE content vectors, which is the intended design (gate depends on semantic content, not position-modulated features).

5. **Gradient checkpointing**: `_ChunkedLinearCE` backward correctly recomputes the forward and applies proper `ignore_index` masking.

### Ablation Design Integrity

- `GATE_FACTORS` derived from `_USE_QUERY/_USE_CONC/_USE_LEN` ensures metadata cannot drift from implementation.
- Leave-one-out contrasts (Full vs No_X) correctly change exactly one factor.
- Capacity-matched controls (`lega_null_*`) preserve parameter count while zeroing evidence.
- Placebo variant uses deterministic hash-based features independent of global RNG.

### Statistical Methodology Assessment

- Holm-Bonferroni correction properly applied within indicator families.
- SESOI/TOST correctly uses pre-registered margins, not data-estimated.
- Missing data sensitivity analysis distinguishes MCAR from MAR appropriately.
- Exploratory FDR control (BH/BY) properly separates confirmatory from exploratory findings.

### Minor Observations (Non-blocking)

1. `_hooked_forward` has a redundant inner `torch.no_grad()` context manager (harmless).
2. Mask cache LRU eviction correctly uses identity checks to guard against stale entries.

### Conclusion

The codebase demonstrates exceptional engineering rigor with comprehensive controls, proper statistical methodology, and correct mathematical implementations. No changes required.
