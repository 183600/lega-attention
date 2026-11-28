# Code Review Report — lega-attention
**Date:** 2026-11-29 03:00 UTC+8  
**Reviewer:** Automated Agent  
**Branch:** local_main (db1ac1d)  
**Scope:** Full codebase audit (attention, model, train, evaluate, data, bootstrap modules)

## Summary

Comprehensive review of the lega-attention codebase found **no unfixed bugs or scientific methodology issues**. All previously identified issues have been properly addressed in recent commits.

## Previously Fixed Issues (Verified)

1. **torch.load misplaced parenthesis** (commit 38a15a7): `map_location="cpu"` was erroneously passed as an argument to `os.path.join()` instead of `torch.load()`. Fixed correctly in `tests/test_early_stop.py` (5 occurrences).

2. **Model train/eval mode switching**: All `model.eval()` / `model.train()` transitions are properly balanced throughout the codebase.

## Code Quality Assessment

### Attention Module (`lega/attention.py`)
- **RoPE implementation**: Forward and backward rotation signs verified correct. `_RoPERotate` autograd function properly handles the rotate_half convention.
- **Chunked Linear CE**: Both `_ChunkedLinearCE` and `_ChunkedCrossEntropy` backward passes properly handle `ignore_index` masking, including gradient zeroing for ignored positions.
- **Gate variants**: The full 2^3 factorial ablation matrix (A=query, B=concentration, C=length) is correctly implemented with proper factor activation tables (`GATE_FACTORS`, `RAW_GATE_FACTORS`).
- **Cache management**: LRU mask cache, log-n prefix table, and triangular cache all have proper eviction policies and byte-size bounds.

### Model Module (`lega/model.py`)
- **RoPE rotation**: Custom `_RoPERotate` autograd function is numerically correct with proper gradient computation.
- **Checkpoint save/load**: Proper NaN/Inf checking before serialization.

### Training Module (`lega/train.py`)
- **NaN diagnostics**: Multi-stage triage with activation auditing, precision rechecking, and module-level origin detection.
- **Early stopping**: Properly restores best model state when patience is exhausted.

### Data Pipeline (`lega/data.py`)
- **Leakage prevention**: Cache schema v13 with document-boundary encoding, near-duplicate decontamination, 64-gram exact overlap scanning.
- **Split integrity**: Train/val/test disjointness enforced at group, paragraph, and line-prefix levels.

## Scientific Methodology Assessment

- **Ablation design**: Full factorial 2^3 matrix with leave-one-out and single-factor controls properly implemented.
- **Placebo control**: `lega_placebo` variant correctly isolates capacity/optimization effects from information effects.
- **Null evidence controls**: Capacity-matched null-evidence variants properly separate information from parameter count.
- **Reparameterization analysis**: `lega_raw` variants correctly test whether conclusions depend on the normalized `c=log(n*q)` parameterization.

## Verdict

**No action required.** The codebase is well-maintained with thorough defensive programming, extensive validation, and sound scientific methodology.
