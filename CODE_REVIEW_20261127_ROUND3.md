# Code Review Report - 2026-11-27 (Round 3 - Independent Deep Review)

## Reviewer: QwenGo Automated Review
## Repository: lega-attention
## Date: 2026-11-27 05:00 CST

## Scope

Comprehensive independent review of core modules:
- `lega/attention.py` (~2000 lines): CausalSelfAttention with 25+ variants
- `lega/model.py` (~1200 lines): GPT model, RoPE, ChunkedLinearCE
- `lega/train.py` (~2300 lines): Training pipeline, NaN diagnostics
- `lega/data.py` (~1600 lines): Data pipeline with anti-leakage measures
- `lega/utils.py` (~2100 lines): Statistical utilities, seed management

## Findings

### Code Quality: EXCELLENT

#### 1. Attention Mechanism (attention.py)
- **Gate formula correctness**: Verified `c = log(n * max_j p_j)` implementation with `clamp_min(1e-30)` on max_p prevents log(0). The mathematical equivalence between the documented formula and the code is exact.
- **Mask caching**: LRU eviction with dual bounds (entry count + byte size) is correctly implemented. Global `_MASK_CACHE` and `_TRI_CACHE` use OrderedDict with proper move_to_end/popitem semantics.
- **Log-n table**: Device-side prefix table with doubling expansion is correct; `_log_n(past_len, T, dev)` returns the correct slice `[past_len, past_len+T)`.
- **Placebo evidence**: Deterministic hash-based pseudo-features don't consume global RNG and are position-aware (using absolute positions with `pos_offset`), preventing cross-mode inconsistencies.
- **Factor probes**: `probe_gate_factors()` and `probe_controlled_dimensions()` use `torch.random.fork_rng()` to avoid polluting caller's RNG state — correct.

#### 2. Model (model.py)
- **RoPE backward**: Verified the inverse rotation formula: `gx1 = go1·cos + go2·sin`, `gx2 = go2·cos - go1·sin` is the correct adjoint of the forward rotation.
- **ChunkedLinearCE**: Handles `ignore_index=-100` correctly; `_ce_nll` with `has_ignore=False` fast-path is appropriate for LM training where all positions typically have valid targets.
- **RoPE inference fusion**: The dual-path logic (fused buffer in eval, separate autograd nodes in training) correctly avoids view aliasing issues with in-place scaling.

#### 3. Training (train.py)
- **NaN audit path**: `_hooked_forward` properly records per-module NaN/Inf counts in execution order. The `forward_error` exception capture ensures the diagnostic path never crashes.
- **Minor cosmetic**: Double `torch.no_grad()` in `_hooked_forward` (outer wraps inner) is harmless but redundant. Not a functional bug.

#### 4. Data Pipeline (data.py)
- **Anti-leakage**: Cache schema v13 with doc_boundaries.json, 64-gram overlap scanning, near-duplicate detection, and source provenance tracking is thorough.
- **Cache integrity**: `_cache_integrity` checks source text hash + report hash + bin content hash + cache schema version. The `_bin_hashes_ok` function prevents silent bin substitution.

#### 5. Statistical Methods (utils.py)
- **Test suite**: Provides paired t-test, sign-flip permutation test, sign test, bootstrap CI, and Wilson score CI. The multiple-test approach with conflict flagging is methodologically sound.
- **Non-deterministic warning counter**: Monkey-patches `warnings.warn` to count "no deterministic implementation" warnings — correctly increments stacklevel to maintain warning attribution.

### Bugs Found: NONE (functional)
### Methodology Issues Found: NONE

### Minor Observation (cosmetic, not committed)
- `_hooked_forward` in train.py has a redundant nested `torch.no_grad()` context. Harmless, not worth a commit.

## Recommendation: No changes needed. Code is production-ready.
