# Code Review — 2026-11-27 (Independent Review)

**Reviewer**: Independent Code Audit  
**Scope**: Core attention mechanism (attention.py), model architecture (model.py), experiment pipeline (experiment.py), evaluation (evaluate.py), analysis module (analysis.py)

## Summary

**No bugs or methodology issues found.**

This is a mature research codebase that has undergone extensive multi-round review. The core LEGA (Length-calibrated Evidence Gated Attention) mechanism is correctly implemented with thorough ablation support.

## Detailed Findings

### 1. Attention Mechanism (`lega/attention.py`)

- **Gate formula correctness**: All 23+ gating variants correctly derive from `_USE_QUERY`/`_USE_CONC`/`_USE_LEN` flags. The `GATE_FACTORS` and `RAW_GATE_FACTORS` tables are consistently derived from the same source of truth.
- **RoPE implementation**: The `_RoPERotate` autograd function correctly implements rotate-half style RoPE with proper gradient computation. The inference path correctly uses pre-allocated buffers.
- **Causal masking**: The `_MASK_CACHE` and `_LOGN_TABLE` correctly handle both training (full sequence) and inference (KV cache with `past_len`) modes.
- **Placebo evidence**: `_placebo_evidence()` correctly produces deterministic position/head-dependent features that do not consume global RNG state.
- **Probe functions**: `probe_gate_factors()` and `probe_controlled_dimensions()` use `torch.random.fork_rng()` to avoid disturbing the caller's RNG state.

### 2. Model Architecture (`lega/model.py`)

- **ChunkedLinearCE**: The custom cross-entropy implementation correctly handles `ignore_index=-100` semantics. The CUDA path conservatively uses mask-based computation even when no ignored positions exist (`tgt_has_ignore = True`), which is a safe performance trade-off.
- **AMP handling**: Correctly detects autocast context and promotes to fp32 for loss computation. The `grad_ctx` properly disables autocast for gradient GEMM when input is fp32/fp64.
- **Gradient correctness**: The closed-form `softmax - onehot` gradient is computed correctly, with ignored positions receiving zero gradient.

### 3. Experiment Pipeline (`lega/experiment.py`)

- **Ledger integrity**: The append-only experiment ledger uses SHA-256 chaining (`chain_record`/`verify_chain`) for tamper detection.
- **Seed verification**: `_seed_evidence_issues()` thoroughly validates that seed reports match declared seeds, including RNG state fingerprints.
- **Concurrent safety**: File locking via `fcntl.flock()` protects concurrent writes.

### 4. Evaluation (`lega/evaluate.py`)

- **Test set protection**: `worst_windows_allowed()` correctly prevents dumping worst-case windows on test data by default.
- **Checkpoint selection**: `checkpoint_selection_info()` correctly tracks the step and validation loss used for model selection.

### 5. Methodology Assessment

- **Ablation design**: The 2^3 full factorial design (A=query, B=concentration, C=length) is correctly implemented with leave-one-out and single-factor decompositions.
- **Placebo control**: `lega_placebo` correctly isolates "extra parameters/optimization freedom" from "meaningful evidence channels".
- **Reparameterization awareness**: The codebase correctly distinguishes between normalized (`c = log(n*q)`) and raw (`c = log q`) parameterizations, with `probe_conc_parameterization()` verifying the actual functional form.

## Conclusion

No actionable issues found. The codebase demonstrates exceptional attention to methodological rigor and reproducibility.
