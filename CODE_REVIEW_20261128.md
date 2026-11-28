# Code Review — 2026-11-28

## Summary
Independent review of lega-attention core modules (attention.py, model.py, train.py).

## Findings

### No Critical Bugs Found

The codebase has undergone extensive prior review cycles (30+ review files present). Core modules are well-structured with:

1. **Attention mechanism (attention.py)**: LEGA gate formula correctly implemented with all ablation variants. The `c = log(n * max_j p_j)` normalization is mathematically sound. RoPE correctly applied before gate computation. Null attention slot correctly uses analytic marginalization via `p_null = sigmoid(a_n - lse)`.

2. **Model (model.py)**: `_ChunkedLinearCE` correctly handles `ignore_index=-100` with proper masking in both forward and backward. RoPE autograd function correctly implements transpose rotation for backward. Weight sharing between `tok_emb` and `lm_head` properly maintained.

3. **Training (train.py)**: NaN detection pipeline is comprehensive with data→code→precision→optimization ordering. Early stopping correctly uses CI-aware thresholds. Gradient accumulation properly divides loss.

### Minor Observations

1. **`_hooked_forward` nested context managers**: Has `with torch.no_grad(): model.eval()` then another `with torch.no_grad(), amp_ctx:` inside — the inner `torch.no_grad()` is redundant but harmless.

2. **Mask cache memory management**: `_MASK_CACHE` uses LRU with byte budget. The `_mask_cache_grow` function correctly guards against stale entries via identity check.

3. **Placebo evidence**: `_placebo_evidence` uses deterministic hash-based features independent of global RNG — correct for fair comparison.

## Methodology Assessment

- **Factorial design integrity**: `GATE_FACTORS` derived from `_USE_QUERY/_USE_CONC/_USE_LEN` ensures metadata cannot drift from implementation.
- **Single-factor ablation**: Leave-one-out contrasts (Full vs No_X) correctly change exactly one factor.
- **Capacity-matched controls**: `lega_null_*` variants preserve parameter count while zeroing evidence — sound methodology.
- **Reparameterization controls**: `lega_raw*` variants correctly test whether conclusions depend on the normalization choice.

## Verdict
✅ No code changes required. Codebase is in good shape.
