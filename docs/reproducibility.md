# Reproducibility Protocols

This document describes the reproducibility guarantees, protocols, and tooling provided by the LEGA Attention project. It serves as both a reference for users and a specification for maintainers.

---

## 1. Overview

The project implements **multi-layer reproducibility** covering:

| Layer | Scope | Mechanism |
|-------|-------|-----------|
| **Environment** | Python, PyTorch, CUDA, system libs | `requirements-lock.txt`, Docker, `seed_report()` |
| **Randomness** | All RNG sources (Python, NumPy, PyTorch CPU/CUDA) | `set_seed()`, `rng_state_fingerprint()`, `seed_plan()` |
| **Data** | Source provenance, preprocessing, tokenization | `source_manifest.json`, `data_report.json`, cache versioning |
| **Configuration** | Experiment params, schema validation | `config_schema.py`, canonical SHA-256 |
| **Code** | Git commit, dirty state | `code_commit`, `code_dirty` in artifacts |
| **Artifacts** | Model checkpoints, results | `lega.artifacts` module with content-addressing |
| **Statistics** | Seeds, pairing, corrections | Pre-registration, Holm-Bonferroni, TOST, MDE |

---

## 2. Random Seed Management

### 2.1 Setting Seeds

```python
from lega.utils import set_seed, seed_report, seed_plan

# Set seed for current process (all RNGs)
set_seed(1337)

# Get audit report of what was seeded
report = seed_report(1337)
# Contains: seed_verified, rng_fingerprints, rng_fingerprints_expected,
#           rng_fingerprints_match_expected, pythonhashseed_*, deterministic_*

# Get plan BEFORE setting (for pipeline startup audit)
plan = seed_plan(1337)
# Contains: expected RNG fingerprints, data_order_rng_fingerprint, is_plan=True
```

### 2.2 RNG Fingerprinting

Every `set_seed()` call records **actual RNG state fingerprints** (not just the seed value):

```python
from lega.utils import rng_state_fingerprint, expected_rng_state_fingerprint

# Actual state after set_seed()
actual = rng_state_fingerprint()
# {"python_random": "...", "numpy_global": "...", "torch_cpu": "...", "torch_cuda_all": "..."}

# Expected state computed independently (without touching global RNG)
expected = expected_rng_state_fingerprint(1337)
# Third parties can verify by calling this function
```

**Key property**: `rng_fingerprints_match_expected` in `seed_report()` is `True` iff the fingerprints genuinely come from the declared seed.

### 2.3 Data Order RNG

Training data sampling uses a dedicated `np.random.Generator` per seed:

```python
from lega.utils import data_rng, data_rng_state_fingerprint

rng = data_rng(1337)  # Used by get_train_batch()
fingerprint = data_rng_state_fingerprint(1337)  # Verifiable by third parties
```

This ensures data order is **independent** of model initialization and evaluation RNG consumption.

---

## 3. Data Provenance & Preprocessing

### 3.1 Source Manifest

`scripts/prepare_data.py` produces `source_manifest.json` containing:

- Original URLs / revision / file sizes / SHA-256
- Cleaned split text hashes
- Changes vs previous run (`changed`, `previous_sha256`)

**Fail-closed**: By default, if source files change (SHA-256 mismatch), preparation fails unless `--allow_source_change` is passed.

### 3.2 Data Report

`data_report.json` contains complete preprocessing audit:

- Sanitization stats (bad chars removed)
- Exact & near-duplicate decontamination (train only; val/test sealed)
- Cross-split n-gram overlap (64-gram exact + sampling + full scan)
- Distribution shift (TV/JS/OOV between splits)
- Token stream validation (int/finite/id-range)
- Test access log (prep-time reads are logged but not for selection)

### 3.3 Cache Versioning

`prepare_bins()` uses `CACHE_SCHEMA` (currently v11) in `meta.json`. Cache hits require:

1. Source text SHA-256 matches
2. `data_report.json` SHA-256 matches
3. Bin content SHA-256 matches (`bin_sha256` in meta)
4. `cache_schema` matches current version
5. Near-duplicate parameters match
6. Tokenizer + fit_on declaration matches
7. Source provenance digest matches (if provided)

**Any mismatch → forced rebuild**. Old caches with missing fields are rejected.

### 3.4 Test Set Sealing

`load_bin(split="test")` requires explicit `allow_test=True`:

```python
from lega.data import load_bin

# Default: PermissionError
load_bin(data_dir, "test")

# Must explicitly declare test access
load_bin(data_dir, "test", allow_test=True)
```

`lega.evaluate` writes to `.test_access_log.jsonl`; summary reports cumulative access count.

---

## 4. Configuration Validation

### 4.1 Schema Audit

All pipeline entry points call `load_config_json(path, validate=True)` which:

- Rejects duplicate JSON keys (silent overwrite in stdlib)
- Validates all known keys against `TOP_LEVEL` + `NESTED` specs
- Enforces strict types: `bool` ≠ `0/1`, `int` ≠ `1.5`, `float` ≠ `NaN/Inf`
- Cross-field checks: `d_model % n_heads == 0`, `train_ctx ≤ block_size`, RULER ctx_lens ≤ block_size
- Returns canonical SHA-256 (`config_canonical_sha256`) independent of formatting

### 4.2 Running Validation Manually

```bash
# CLI
python -m lega.config_schema --config configs/small.json

# Python
from lega.config_schema import audit_config, assert_config_valid
from lega.utils import load_config_json

cfg = load_config_json("configs/small.json", validate=False)
report = audit_config(cfg)  # or assert_config_valid(cfg) to raise on fail
```

### 4.3 Config Fingerprinting

```python
from lega.config_schema import config_canonical_sha256, canonical_config_json

# Stable across key order, whitespace, comments
fp = config_canonical_sha256(cfg)
json_str = canonical_config_json(cfg)
```

Two configs with same semantic content → same fingerprint.

---

## 5. Experiment Artifacts

### 5.1 Model Checkpoints

```bash
# Register a checkpoint
python -m lega.artifacts register-checkpoint \
  results/exp/lega_seed1337_step8000.pt \
  --experiment-tag small \
  --seed 1337 \
  --variant lega \
  --step 8000 \
  --config configs/small.json \
  --data-fingerprint <sha256-of-data>

# Verify a checkpoint
python -m lega.artifacts verify \
  --name lega-small-seed1337 \
  --version 1.0.0 \
  --artifact-path results/exp/lega_seed1337_step8000.pt
```

### 5.2 Experiment Results

```bash
# Register entire result directory
python -m lega.artifacts register-result \
  results/small \
  --experiment-tag small \
  --config configs/small.json \
  --data-fingerprint <sha256-of-data>

# List all artifacts for an experiment
python -m lega.artifacts list --experiment-tag small
```

### 5.3 Artifact Metadata

Each artifact records:

| Field | Description |
|-------|-------------|
| `artifact_type` | `model_checkpoint` or `experiment_result` |
| `content_sha256` | Content hash (file or directory tree) |
| `code_commit` | Git commit at creation time |
| `code_dirty` | Whether working tree had uncommitted changes |
| `config_sha256` | Canonical config fingerprint |
| `data_fingerprint` | Source data fingerprint |
| `python_version`, `torch_version`, `cuda_version` | Environment |
| `experiment_tag`, `seed`, `variant`, `step` | Experiment context |

---

## 6. Statistical Rigor

### 6.1 Pre-registration

Confirmatory hypotheses are registered in `preregistration/hypotheses.json` **before** formal experiments run. The lock file `hypotheses.lock.json` provides:

- Baseline document SHA-256 + per-hypothesis content hashes
- Append-only amendment chain with timestamps and reasons
- Scope lock (config, seeds, data, comparisons) to prevent scope creep

**Any violation (deletion, modification, undeclared addition) → all confirmatory judgments downgraded to exploratory.**

### 6.2 Multiple Comparison Correction

- **Primary**: Holm–Bonferroni within each metric family (controls FWER)
- **Exploratory FDR**: Benjamini–Hochberg across exploratory families (with BY sensitivity)
- **Equivalence (TOST)**: Separate Holm families per SESOI margin (0.5×/1×/2×)

### 6.3 Small-Sample Protections

For n_seeds < 6:

- Permutation test granularity limit explicitly reported (n=3 → min p=0.25)
- t-test + permutation test both reported; conflicts flagged
- MDE (Minimum Detectable Effect) reported per comparison
- |Δ| < paired_diff_std → flagged as "within noise"

### 6.4 Missing Data Sensitivity

`lega/missingness.py` reports:

- Completer vs planned-seed analyses
- Tipping-point / ±SESOI boundary analysis
- Sharp Manski bounds (RULER EM) & sign-score bounds (all metrics)
- **Does not assume MAR/MCAR**; bounds are assumption-free

---

## 7. Running Reproducible Experiments

### 7.1 Using Locked Dependencies

```bash
# Exact versions used in reference environment
pip install -r requirements-lock.txt
```

### 7.2 Using Docker (Full Environment)

```bash
# CPU
docker compose run --rm lega-cpu

# GPU
docker compose run --rm lega-gpu

# Smoke test in container
docker compose run --rm smoke-test

# Full small experiment
docker compose run --rm run-small
```

### 7.3 Pipeline Execution

```bash
# Smoke test (CI verification)
bash scripts/run_smoke.sh

# Formal experiment (5 seeds, small config)
python run_all.py --config configs/small.json --data_dir data/wikitext103

# With baseline tuning (recommended)
python run_all.py --config configs/small.json --data_dir data/wikitext103 \
  --tune_baseline --tune_seeds 1337,2024,42

# With inference-time ablations
python run_all.py --config configs/small.json --data_dir data/wikitext103 \
  --ablate_variants lega,qgate,gate_only

# With dose-response
python run_all.py --config configs/small.json --data_dir data/wikitext103 \
  --dose_response --dose_variants lega
```

### 7.4 Result Verification

After each run, the pipeline produces:

```
results/<tag>/
├── config_used.json          # Exact config (validated)
├── repro.json                # Code commit, config SHA, data fingerprint
├── significance.json         # All statistical tests (Holm, TOST, FDR, etc.)
├── summary.md                # Human-readable summary
├── length_null.json          # c ~ log2(n) slopes (required for length claims)
└── artifact_registry.jsonl   # Registered artifacts (if using lega.artifacts)
```

**Verify reproducibility** by checking:
1. `repro.json.config_sha256` matches `config_canonical_sha256` of your config
2. `repro.json.data_fingerprint` matches your data
3. `repro.json.code_commit` matches your git HEAD
4. `significance.json` contains expected tests with consistent p-values

---

## 8. CI/CD Reproducibility

### 8.1 Standard CI (`.github/workflows/ci.yml`)

Runs on every push/PR:
- `ruff` lint + format check
- `mypy --strict` type checking
- `pytest` on Python 3.10/3.11/3.12
- Smoke test
- Config validation

### 8.2 Pre-commit Hooks

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

Checks: ruff, mypy, trailing whitespace, YAML/JSON/TOML syntax, large files, requirements-lock consistency, config validation.

### 8.3 Continuous Loop (AI Agent)

Separate workflow (`.github/workflows/continuous-loop.yml`) for automated code improvement. **Protected**: `.github/` and `README*.md` are read-only for the agent.

---

## 9. Troubleshooting Reproducibility Failures

| Symptom | Likely Cause | Resolution |
|---------|--------------|------------|
| `seed_report.rng_fingerprints_match_expected: false` | RNG consumed before `seed_report()` call | Call `seed_report()` immediately after `set_seed()` |
| Cache rebuild on every run | Source text modified or `data_report.json` changed | Check `source_manifest.json` for `changed: true` |
| `PermissionError: test set sealed` | Code calls `load_bin("test")` without `allow_test=True` | Add `allow_test=True` only in final evaluation |
| Config validation fails on unknown key | Typo in config or stale key from old version | Fix typo or remove key; use `--allow_unknown` only for migration |
| Permutation test p=1.0 with n=3 | Granularity limit (min p=0.25) | Increase seeds; report granularity limitation |
| TOST "equivalent" but t-test significant | Effect within SESOI but statistically detectable | Report both; SESOI is practical significance threshold |

---

## 10. Extending Reproducibility

### Adding New Config Keys

1. Add to `TOP_LEVEL` or `NESTED` in `config_schema.py`
2. Define type spec (use existing kinds or add to `_check_scalar`)
3. Add cross-field validation if needed
4. Test with `python -m lega.config_schema --config configs/small.json`

### Adding New Artifact Types

1. Extend `ArtifactMetadata.artifact_type` values
2. Add creation function in `lega/artifacts.py`
3. Update registry schema version if structure changes

### Adding New Statistical Tests

1. Implement in `lega/utils.py` following existing patterns
2. Return dict with standard keys (`n`, `p`, `effect_size`, etc.)
3. Register in appropriate family in pipeline summary
4. Add to exploratory FDR if exploratory

---

## 11. Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-11 | Initial reproducibility protocol documentation |

---

*This document is part of the LEGA Attention project's commitment to reproducible research. For questions or improvements, see the issue tracker.*