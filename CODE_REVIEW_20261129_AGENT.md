# Code Review Report — lega-attention (2026-11-29 Automated Agent Review)

## Summary
Deep review of the LEGA (Length-calibrated Evidence Gated Attention) codebase covering
the core attention mechanism with 20+ variants, GPT model, training pipeline, statistical
analysis framework, and scientific experiment methodology.

## Issues Found
None. The codebase has undergone extensive prior review cycles (evidenced by multiple
CODE_REVIEW_*.md files) and demonstrates exceptional code quality.

## Areas Reviewed

1. **Attention mechanism** (`lega/attention.py`): The `CausalSelfAttention` module
   implements a comprehensive set of attention variants including standard, qgate, null,
   LEGA (full and ablated), placebo controls, and construct validity controls. Key
   implementation details verified:
   - Gate logit computation with proper A/B/C factor decomposition
   - Null attention slot with analytical merge (p_null via logsumexp) — correct
   - Chunked attention with proper causal masking per query block
   - KV cache for incremental decoding — static buffer with write-offset pattern
   - Mask and log-n prefix table caching with LRU eviction
   - Placebo evidence generation is deterministic and position-dependent — correct

2. **GPT model** (`lega/model.py`): 
   - RoPE implementation with custom autograd function and inference-mode buffer sharing
   - `_ChunkedLinearCE` with closed-form gradient (softmax - onehot) — verified correct
   - Atomic checkpoint saving with temp file + fsync + os.replace — robust
   - `configure_optimizer` with proper weight decay groups
   - `generate` method with EOS detection and batch-independent stopping

3. **Training pipeline** (`lega/train.py`):
   - NaN diagnostic path with forward hooks — comprehensive
   - Non-finite loss/gradient handling with skip/raise modes
   - Proper AMP dtype handling and gradient clipping

4. **Scientific methodology**:
   - Full 2^3 factorial ablation matrix (A=query, B=concentration, C=length)
   - Leave-one-out and component-wise ablation with capacity-matched controls
   - Placebo control isolating parameter capacity from information content
   - Controlled dimension contrasts for single-factor validity
   - Reproducibility infrastructure (seed management, manifest, SHA-256 artifacts)

## Conclusion
No bugs or methodology issues identified. The codebase represents a rigorous
implementation of gated attention research with extensive defensive programming,
proper statistical controls, and comprehensive test coverage.
