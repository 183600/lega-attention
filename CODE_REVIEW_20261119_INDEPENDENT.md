# Independent Deep Code Review - lega-attention (Round 8)

**Date:** 2026-11-19 15:05 CST  
**Reviewer:** Fresh Independent Agent Review  
**Scope:** Full codebase (95 Python files) — bug detection + scientific methodology audit

## Executive Summary

| Category | Result |
|---|---|
| Files scanned | 95 |
| Static analysis flags | 40 |
| **True bugs found** | **0** |
| **Scientific methodology issues** | **0** |
| **Verdict** | **PASS — Codebase is clean** |

## Detailed Verification

### 1. Causal Mask Correctness (attention.py:1694)

```python
not_allowed = qpos.view(1, 1, T, 1) < torch.arange(n_real, device=dev).view(1, 1, 1, n_real)
```

**Verified correct.** Query position `past_len + t` cannot attend to key position `k` when `past_len + t < k`. With `past_len` offset for incremental decoding, the causal constraint is properly maintained. No off-by-one error. Single-query-step path correctly skips mask construction (all keys visible).

### 2. LEGA Gate Formula (attention.py:1-60, 1100-1170)

The gate `g = sigmoid(w·q + a*c + b*log(n) + d)` where `c = log(n * max_j p_j)`:
- **Concentration term:** `c = log(n) + log(max_p)` correctly computed; when `use_conc=True`, `log_n` is added to the raw `log(max_p)`.
- **Null attention:** Analytic marginalization `p_null = sigmoid(a_n - logsumexp(att))` is mathematically equivalent to concatenating null slot to K/V — verified term by term.
- **Ablation variants:** `lega_no_len` correctly zeroes only `b*log_n` while preserving `a*c` (which still contains length normalization via `log(n)`). `lega_no_conc` zeroes only `a*c`. All 2^3 factorial cells properly implemented.
- **Placebo variant:** Uses deterministic pseudo-random features (no RNG consumption), correctly parameter-matched.
- **Detach variant:** `conc_stat.detach()` correctly blocks gradient flow through concentration while preserving the parameter's gradient path.

### 3. RoPE Implementation (attention.py, positional encoding)

`_log_n_table` uses `torch.inference_mode(False)` to prevent inference tensor contamination — correct. Table sliced as `table[past_len:past_len+T]` for proper position indexing.

### 4. Division Safety (17 flagged)

All verified safe:
- `d_model // n_heads`: Validated at `__init__` entry (`d_model % n_heads != 0` → ValueError)
- `loss / total`: Protected by `clamp_min(1)` or conditional guards
- `share = value / total`: Guarded by `if total > 0` or ternary expressions
- Statistical computations: All have proper edge-case handling

### 5. Gradient Flow

All `loss.backward()` calls verified in correct contexts:
- `sanity.py:618`: Wrapped in `torch.enable_grad()` within the sanity check probe
- `train.py:1931`: Standard training loop with `model.train()` active
- No detached computation graphs or frozen parameters where training is expected

### 6. Scientific Methodology

- **Full 2^3 factorial design** with capacity-matched controls (Full / Null_X / No_X three-arm decomposition)
- **Holm-Bonferroni correction** for multiple testing
- **Bootstrap confidence intervals** with proper percentile method
- **Data leakage prevention** via fingerprint-based overlap detection
- **Sealed test-set protocol** with access logging
- **Shapley value decomposition** with numerical stability guards (rejects catastrophic cancellation)

### 7. Static Analysis Flag Resolution

| Pattern | Count | Verdict |
|---|---|---|
| `eval_call` | 14 | Standard PyTorch `model.eval()` lifecycle |
| `unprotected_division` | 17 | All guarded (clamp_min, ternary, pre-validation) |
| `todo_fixme` | 1 | Documentation note about previously fixed bug |
| `no_grad_context_missing` | 2 | Both in `torch.enable_grad()` contexts |

### 8. Previously Fixed Bug (analysis.py:1330)

The `# BUG FIX` comment documents a previously fixed issue: mask now correctly covers all records (length `n_records`) rather than just the valid subset. The fix is correct — `valid_mask.copy()` produces a full-length boolean array.

**No code changes required.**
