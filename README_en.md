# LEGA: Length-calibrated Evidence-Gated Attention

**Length-calibrated Evidence-Gated Attention** — investigating whether an attention head can learn to **write less irrelevant content into the residual stream** when it **cannot find reliable information** in the current context.

> The core novelty is not "adding yet another gate": prior work either gates the attention output (looking only at the query vector) or lets the head choose "not to read" (explicit null attention). LEGA lets **the same gate jointly consider the current query, the concentration of the attention distribution, and the context length**.

[中文说明](README.md) | **English**

---

## 1. Method

### 1.1 Formulation

For each query position $t$ and head $h$, the standard attention output is $o = \sum_j p_j v_j$. LEGA computes, before writing back to the residual stream:

$$
c = \log\!\bigl(n \cdot \max_j p_j\bigr), \qquad
g = \sigma\bigl(w^\top q + a\,c + b\log n + d\bigr), \qquad
o' = g \cdot o
$$

where $p_j$ are the causal attention weights, $n$ is the number of real tokens readable by the query (i.e., position $t+1$), and $w\in\mathbb{R}^{d_{head}}$ and the scalars $a, b, d$ are all **per-head** parameters.

### 1.2 Design intuition

- **Property of $c$**: under uniform attention $\max_j p_j = 1/n$, so $c = \log(n \cdot 1/n) = 0$ — **independent of context length**. Hence $c$ measures "evidence concentration beyond uniform", decoupling "reading attentively" from "reading a long context".
- **$b\log n$ is the length-calibration term**: even at equal concentration $c$, a "focused read" in a long context has different credibility than in a short one. $b$ lets each head learn its own length-calibration curve.
- **$w \cdot q$ preserves content semantics**: the gate is not a pure statistical rule; the query vector still participates.
- **Initialization**: $w=a=b=0,\ d=2.0$, giving $g\approx\sigma(2)\approx 0.88$ initially — close to standard attention, so training starts stably.

### 1.3 Relation to prior work

| Method | Gating evidence | Mechanism | Difference from LEGA |
|---|---|---|---|
| Output gating (qgate, prior) | query only, $w^\top q$ | $o'=g\cdot o$ | Unaware of "was this read concentrated / how long is the context" |
| Explicit null attention (null, prior) | learnable null KV slots | attention may skip reading entirely | "Not reading" and "reading but not writing" are two different degrees of freedom |
| **LEGA (ours)** | query + concentration + length | $o'=g\cdot o$ | Jointly decides "should we write" from three factors |

### 1.4 Variants (all implemented in code)

| Variant | Gate formula | Purpose |
|---|---|---|
| `standard` | none | Baseline |
| `qgate` | $g=\sigma(w^\top q+d)$ | Prior-work representative: query-only output gating |
| `null` | none (extra null KV slots) | Prior-work representative: explicit "choose not to read" |
| `lega` | $g=\sigma(w^\top q+a\,c+b\log n+d)$ | **Ours** (full factorial A+B+C) |
| `lega_no_len` | $g=\sigma(w^\top q+a\log\max_j p_j+d)$ | Leave-one-out: no length calibration (A+B) |
| `lega_no_conc` | $g=\sigma(w^\top q+b\log n+d)$ | Leave-one-out: no concentration evidence (A+C) |
| `lega_no_query` | $g=\sigma(a\,c+b\log n+d)$ | Leave-one-out: no query term (B+C) |
| `lega_conc_only` | $g=\sigma(a\log\max_j p_j+d)$ | Factorial: concentration term only (B) |
| `lega_len_only` | $g=\sigma(b\log n+d)$ | Factorial: length term only (C) |

**Ablation completeness**: with query=A, concentration=B, length=C, both the full
factorial (none/`standard`, A/`qgate`, B/`lega_conc_only`, C/`lega_len_only`,
AB/`lega_no_len`, AC/`lega_no_conc`, BC/`lega_no_query`, ABC/`lega`) and the
leave-one-out set (Full/Full−A/Full−B/Full−C) are implemented; the summary lists
the matrix and flags missing cells. Component usefulness is never inferred from
coverage alone: the summary judges each of A/B/C with Holm-corrected significant
paired tests (leave-one-out: Full−X significantly worse than Full; factorial:
X alone significantly better than no gate), and explicitly rejects
"every component helps" when the evidence is missing or points the other way.

Parameter overhead: gating adds only **$d_{head}+3$ parameters per head** (LEGA), <0.02% of the model (`null` is $2d_{head}$ per head). Scripts print the exact numbers at the start of training.

---

## 2. Experimental design

- **Data protocol (leakage-proof, fully auditable)**:
  1. **Split first, preprocess second**: train/val/test text is produced before encoding; the character-level vocabulary (feature engineering) is fit on train only, val/test are only transformed, and unseen characters map to `<unk>` (GPT-2 BPE is a public pretrained vocabulary, independent of this project's data);
  2. **Temporal split**: with `--local_txt`, the earliest 80% is train, the next 10% is val and the latest 10% is test; cut points align to document boundaries (heading lines > blank-line paragraphs > newlines) and are never shuffled at split time (batch-level random sampling of training samples during training is allowed and not leakage); the official WikiText-103 split is document-disjoint and inherently held out;
  3. **Groups never cross sides**: documents/paragraphs are the group unit and land entirely in one split; training lines that exactly duplicate val/test lines are decontaminated (val/test stay sealed and untouched), and sampled 64-gram hashing screens for token-level copies across splits;
  4. **Cleaning & anomaly checks**: corrupt characters (U+FFFD/NUL/illegal control chars) are deterministically removed; token streams are hard-validated (integer, finite, id range); duplicate samples, overlong lines and train-vs-val/test unigram shift (TV/JS, OOV) are all written to `data_report.json`;
  5. **Test seal**: checkpoint selection and early stopping use val only; test is accessed for final evaluation only, and every test evaluation appends to the `data_dir/.test_access_log.jsonl` ledger; the summary reports the total count and any repeatedly evaluated checkpoint.
- **Fair comparison (controlled variables)**: all variants × seeds are trained with the same config (same optimizer / LR schedule / steps / data order); after training, two layers are cross-checked automatically: `config_used.json` (hyperparameters item by item) and the `eval.json` provenance (data fingerprint, code commit, config SHA-256, actual steps and tokens_seen, also compared against the pipeline's data fingerprint). Any difference fails the pipeline; the random seed is replayed after model construction so training-time randomness (dropout etc.) is also identical across variants — fairness is an executable assertion, not a verbal promise.
- **A tuned baseline (avoid misattributing hyperparameter gains)**: with `--tune_baseline`, a small lr × weight-decay grid search is run on the **baseline (`standard`) first**, selected on the validation split only (`lega/tuning.py`; test is never loaded). The winning combination is injected into **all** variants, and the evidence (including `default_is_optimal`, whether the default hyperparameters are within a 0.5% val_loss tolerance of the grid optimum) is written to `results/<tag>/baseline_tuning.json` and rendered in the summary. Without this evidence the summary explicitly warns about an untuned baseline instead of assuming one. The selection itself is noisy: `--tune_seeds` trains every grid combination on multiple seeds and selects by **mean val_loss**, reporting `selection_seeds` and per-seed details; without it only a single seed (the first formal seed) is used and the summary explicitly warns about the winner's curse (a lucky seed can crown a worse configuration) — give ≥3 seeds for formal runs. **Simple statistical baseline**: `lega/ngram.py` reports unigram/bigram smoothed n-gram PPL on the same split as the neural models (α selected on val), answering how hard the data is in absolute terms and whether the neural models actually beat a parameter-free baseline.
- **Significance against the simple baseline (supplementary)**: `--ngram_split` defaults to `auto` — the same rule as neural evaluation (non-empty `test.bin` → test, else val), so val-evaluated models can never be silently compared against a test-split baseline. The summary runs a **one-sample paired test of each variant's per-seed PPL against the same constant baseline** (n-gram has no seed variance), Holm-corrected within the variant family, reporting ratio / relative reduction / p_adj / verdict; variants evaluated on a mismatched split are listed separately as "not compared" rather than mixed into the table. This comparison is not pre-registered and is labelled exploratory evidence; it does not replace the controlled same-seed paired tests between variants.
- **Statistical reliability (multi-seed)**: key metrics are reported as **Mean ± Std** plus the paired Cohen's d effect size across random seeds (`configs/small.json` defaults to 5 seeds; override with `--seeds`). Variants are compared on identical seeds with a two-sided paired t-test, and p-values within each metric family are corrected with **Holm–Bonferroni** (controlling the family-wise error rate); verdicts are based on corrected p-values (`results/<tag>/significance.json`). With n<3 only descriptive statistics are reported and no significance verdict is given, so a "0.2% gain" cannot be mistaken for real when the variance is 0.5%.
- **Pre-registration and tamper-evidence (confirmatory vs exploratory)**: directional claims are registered before the formal runs in `preregistration/hypotheses.json` (**deliberately not under `docs/`**: the pipeline cleanup deletes everything there except `design_notes.md`, and repeated auto-clean commits once deleted the registry); at summarization each claim is judged against Holm-corrected same-seed paired tests as supported / rejected (significant in the opposite direction = falsified) / undetermined (undetermined ≠ no effect; read it against the MDE). Results go to `summary.md` and `significance.json`; unregistered comparisons are exploratory leads only. The registry is protected by `preregistration/hypotheses.lock.json`: the lock stores the whole-document SHA-256 and **per-hypothesis scientific-content digests**, and any amendment must be traced on an **amendment chain** via `python -m lega.hypotheses --amend --reason ...` (existing hypotheses may only be added to, never edited or removed). Deleting the registry, rewriting a claim, swapping a variant pair, dropping an entry or adding an unrecorded hypothesis is flagged as `violated` by `--verify`/the summary page, and **all confirmatory verdicts are automatically downgraded to exploratory**. Even the accidental deletion/relocation and restoration of the registry (see `docs/design_notes.md`) must be recorded in `amendments`. An intact chain is not enough: the summary also performs a **pre-registration scope check** — the registry's `scope` (config scale, data/tokenizer, seed plan) must match the actual run (`repro.json`'s `config_file` / data fingerprint / seeds); tiny/smoke/dummy-data/unregistered-tokenizer runs are out of scope and confirmatory verdicts are downgraded as well (tuned lr/weight-decay injected by the tuning step are within protocol and do not count; unregistered seeds only warn).
- **Three evaluation axes**:
  1. **Perplexity** (WikiText-103 PPL: checkpoint selection on val, PPL reported on the held-out test split, falling back to val when test.bin is absent); windows are disjoint and evenly sampled across the whole evaluation set (all windows used when too few); `gate_mean` and PPL use the same windows and precision; a **unigram/bigram smoothed n-gram PPL** is reported alongside as a simple statistical baseline (`ngram_baseline.json`), by default on the same split as the neural evaluation (`--ngram_split auto`, preventing mismatched comparisons), so neural scores are never interpreted without the absolute difficulty reference;
  2. **RULER long-context retrieval** (simplified reimplementation, 5 tasks): `niah_synth` (synthetic-distractor single-key retrieval), `niah_corpus` (natural-text distractors), `niah_multikey` (multi-key distractors), `fwe` (frequent-word extraction), `vt` (variable tracking); multiple context lengths + greedy decoding; for numeric tasks the EM **only accepts the first numeric answer in the generated text**, so "wrong answer first, lucky guess later" is never scored;
  3. **Actual inference latency**: prefill (last-position logits only) and token-by-token decode with KV cache, compared against the SDPA fast path.
- **Core-risk checks (not just "attention sink reduced")**: `lega/analysis.py` performs mechanistic analysis on RULER samples with gold spans:
  - **misleading_rate**: fraction of highly concentrated attention ($\max_j p_j>\tau$) whose landing point is outside the answer's numeric span (value_span); the hit test uses the exact answer-number token span rather than the whole needle sentence — direct evidence that "concentrated ≠ correct";
  - **P(hit | concentrated)** vs **P(hit | not concentrated)**;
  - **AUROC and PR-AUC (both reported under imbalance)**: ability of the gate score to separate "hit / miss", plus **the difference between LEGA's full logit and the query-only term $w^\top q$** in AUROC and in PR-AUC (average precision), directly answering whether length-calibrated attention statistics judge "should we write" better than the query alone. hit/¬hit is a heavily imbalanced binary task (the positive rate is `hit_base_rate`); below 5% positive rate AUROC becomes optimistic because of the huge negative base, so **PR-AUC is the primary metric** (its chance baseline equals the positive rate), especially below 0.1%. The **gate decision-point F1** ($g>0.5$) is reported with **macro / micro / weighted** averages (`gate_f1` in `analysis_*.json`; macro in the summary table) — under imbalance these differ widely, and any table that writes "F1" without the averaging scheme is incomplete; a **fixed-rule straw-man comparison `gate_f1_trivial`** (always-write / never-write) is reported next to it, since a gate that cannot beat the uninformative baseline has no decision value (summary column "always-write baseline F1(macro)"); RULER EM is a generative metric without a threshold (fwe additionally reports recall).

---

## 3. Quick start

### 3.1 Install

```bash
pip install -r requirements.txt          # torch / numpy / tiktoken / matplotlib
# optional: pip install pyarrow wandb
```

### 3.2 Smoke test (recommended first; a few minutes on CPU)

```bash
bash scripts/run_smoke.sh
# equivalent to: unit tests -> generate dummy data (data/dummy) -> run_all --smoke
```

Smoke mode automatically shrinks: 20 steps/variant, a single seed, ctx=min(256, block_size), 2 examples per task, and runs the overfit-small-batch sanity check (60 steps) — for verifying code correctness, not scientific conclusions.

### 3.3 Full experiments (WikiText-103 + small config)

```bash
# 1) Data (~300MB download; offline: point --local_txt at local text)
python scripts/prepare_data.py --out_dir data/wikitext103
# optional --max_tokens 30000000 to control training size

# 2) Full pipeline: 9 variants x multiple seeds (5 by default) -> PPL -> RULER
#    -> latency -> analysis -> plots -> summary (Mean ± Std + paired tests)
python run_all.py --config configs/small.json --data_dir data/wikitext103

# optional: custom seeds (e.g. 3) or W&B logging
python run_all.py --config configs/small.json --data_dir data/wikitext103 --seeds 1337,2024,42
wandb login
python run_all.py --config configs/small.json --data_dir data/wikitext103 --wandb

# optional (recommended): tune the baseline first — lr/weight-decay grid on
# `standard` (val only); each combination is trained on the --tune_seeds seeds
# (defaults to --seeds) and selected by mean val_loss (anti lucky-seed); the
# winner is injected into all variants and evidence goes to
# results/<tag>/baseline_tuning.json
python run_all.py --config configs/small.json --data_dir data/wikitext103 \
  --tune_baseline --tune_seeds 1337,2024,42

# optional: early stopping — stop when val loss shows no material improvement for
# N consecutive evals (off by default; actual steps may differ across variants,
# the audit then compares planned steps and the summary marks early-stopped runs)
python run_all.py --config configs/small.json --data_dir data/wikitext103 --early_stop_patience 5

# the simple statistical baseline can also be run standalone (val by default;
# test requires an explicit opt-in and writes the access ledger). In the full
# pipeline --ngram_split auto (default) matches the neural eval split and each
# variant's per-seed PPL is tested against the baseline (one-sample paired,
# Holm-corrected) in the summary.
python -m lega.ngram --data_dir data/wikitext103 --split val --out results/ngram_baseline.json
```

Outputs:

| Path | Content |
|---|---|
| `out/<tag>/<variant>[/seed<seed>]/model.pt` | Checkpoint + training config + code/data provenance fields (seeded subdirs for multi-seed runs) |
| `out/<tag>/<variant>[/seed<seed>]/train_log.jsonl` | Training/validation curves |
| `out/<tag>/<variant>[/seed<seed>]/run_manifest.json` | **Provenance: git commit/dirty, config SHA-256, data fingerprint, runtime, command line** |
| `results/<tag>/summary.md` | **Summary (Mean ± Std + ablation matrix + controlled-variable audit + reproducibility/sanity)** |
| `results/<tag>/repro.json` | Pipeline-level provenance (code version, data fingerprint, seeds, runtime) |
| `results/experiments.jsonl` | **Append-only experiment ledger: per-run run_id, code version, config SHA-256, seeds, data fingerprint, sanity verdict, per-run final metrics and artifact hashes (started/completed lifecycle)** |
| `data/<data_dir>/data_report.json` | leakage audit (split protocol, vocabulary fit set, decontamination, 64-gram overlap, distribution shift, overlong lines, test-access ledger) |
| `results/<tag>/sanity_<variant>.json` | Overfit-small-batch sanity check result and thresholds |
| `results/<tag>/baseline_tuning.json` | Baseline (`standard`) lr/weight-decay tuning evidence (val only; includes the `default_is_optimal` verdict, `selection_seeds`/per-seed val_loss details, and per-trial provenance) |
| `results/<tag>/ngram_baseline.json` | Simple statistical baseline: unigram/bigram smoothed n-gram held-out PPL (α selected on val; test needs an explicit opt-in; the pipeline matches the neural eval split by default) |
| `results/<tag>/significance.json` | Same-seed paired t-tests (Δ mean ± Δ std, paired Cohen's d_z, raw p, Holm-corrected p_adj, component evidence); includes `registration_integrity` (pre-registration lock-chain check), `registration_scope` (config-scale/tokenizer/seed-plan scope check) and `registered_hypotheses` (per-claim verdicts); also `ngram_comparison` (per-variant one-sample paired tests vs the simple baseline) |
| `results/<tag>/curve_diagnosis.json` | Per-run training-curve diagnosis (overfitting incl. plateaued-train/rising-val, overfitting trend (train still falling, validation stalled), underfitting/divergence/NaN + grad-norm spikes/non-finite + recommendations; early-stopped runs are flagged as intentionally truncated) |
| `results/<tag>/badcases_*.md` | RULER error-sample attribution (label error / distractor copy / hallucinated / no numeric answer / fwe) + coarse attribution-axis counts + **improvement directions** (trigger evidence → suggestion → next experiment) |
| `results/<tag>/worstcases_*.md` | LM worst-window attribution (repetition attractor / confident bias / low confidence; whether gold appeared in context) + **improvement directions** |
| `results/<tag>/badcase_cross.json` | Cross-variant error comparison (shared failures vs method-specific new/fixed errors relative to standard; includes per-variant new-error category counts and cross-variant improvement directions) |
| `results/<tag>/eval_*.json` | Held-out test PPL per variant (val fallback when test.bin is absent), with worst_windows bad cases and training throughput/cost (`train_step_time_ms`/`train_tokens_per_sec`/`train_peak_mem_mb`, feeding the efficiency-precision trade-off) plus early-stop records (`steps_planned`/`steps`/`early_stopped`) |
| `results/<tag>/ruler_*.json` | RULER 5 tasks × context lengths, with badcases error samples |
| `results/<tag>/latency.json` | prefill / decode latency (per-iteration median + `*_spread_pct=(p90-p10)/median` measurement spread, for telling real differences from noise) |
| `results/<tag>/analysis_*.csv|json` | Per (layer, head, query position) gate records and aggregates |
| `results/<tag>/fig_*.png` | All plots (including training curves `fig_train_curves.png`) |

### 3.4 Reproducibility, sanity check and experiment tracking

- **Full seed coverage**: `lega.utils.set_seed` fixes Python `random`, NumPy, PyTorch (CPU/CUDA), cuDNN determinism (`warn_only`) and `PYTHONHASHSEED` (for child processes); batch sampling uses an explicit `np.random.default_rng(seed)` (no torch DataLoader/workers), so data order is uniquely determined by the seed — unit-tested in `tests/test_repro.py`. `run_all` additionally aligns each child command's `PYTHONHASHSEED` with that command's own `--seed` (instead of one shared value), and every `run_manifest.json` embeds a `seed_report` recording the actual seed per random source, **short fingerprints of the real RNG states** (a third party can recompute them with `set_seed(seed)+rng_state_fingerprint()`; identical for the same seed, different across seeds, and `null` whenever the current state does not belong to the requested seed), cuDNN/determinism switches and the data-order mechanism. The report also separates the `PYTHONHASHSEED` seen at interpreter start from the current env value (which only affects child processes) and sets `pythonhashseed_effective=false` when the running process is not actually fixed — the field can no longer be made self-fulfilling by `set_seed` itself. Bit-exact reproducibility across hardware (different GPU/CPU/cuDNN) is not guaranteed, but reruns on the same machine are stable.
- **Overfit-small-batch sanity check (pre-training gate)**: `lega.sanity` draws a fixed small batch (8 × 128 tokens by default), disables dropout/weight decay and all data perturbation, and optimizes the **same batch**, checking whether the loss can be driven toward 0 and next-token accuracy toward 100% (thresholds adjustable via `--loss_threshold/--acc_threshold` or a `sanity` config block). `run_all.py` runs it per variant before full training and aborts on failure (`--skip_sanity` to skip, `--sanity_steps` to change the budget). This is a **necessary but not sufficient** check: failing it means gradients/loss/labels have a serious bug; passing does not rule out data-augmentation, feature-alignment or label-mapping bugs. The sanity JSON also records the first-step loss (should be ≈log V), the first-step gradient norm, the **first-step per-parameter gradient coverage** (`missing/zero/nonfinite`, which localizes "total grad norm is fine but a submodule never entered the graph" disconnections) and the full loss curve for intermediate inspection.
- **Experiment tracking / traceability**: every training run writes `run_manifest.json` (git commit/dirty, config SHA-256, data fingerprint, seed audit, Python/PyTorch/CUDA environment, exact command line), hyperparameters live in `config_used.json`, curves in `train_log.jsonl` (aggregated into `fig_train_curves.png`); `run_all` writes a global `results/<tag>/repro.json` and **appends** one ledger record per experiment at the fixed path `results/experiments.jsonl` (`started` at launch + `completed` at the end, same `run_id`): code version, config hash, seeds, data fingerprint, sanity verdict, per-run final metrics and SHA-256 of `train_log.jsonl`, the evaluated checkpoint and all result files (`eval_*.json` also embeds `ckpt_sha256`, binding results to the exact weights). Reruns no longer erase the previous run's history; a record with only `started` means that run was aborted/crashed — failed experiments leave a trace too. All evaluation outputs (eval/ruler/analysis/latency) embed the same code/data provenance fields — no manual bookkeeping.

```bash
# Run the sanity check standalone (exit code 0 = pass, 1 = fail)
python -m lega.sanity --config configs/tiny.json --data_dir data/dummy --variant lega \
    --out results/tiny_sanity/lega.json
```

### 3.5 Running each stage separately

```bash
python -m lega.train     --config configs/small.json --variant lega --data_dir data/wikitext103
python -m lega.evaluate  --ckpt out/small/lega/model.pt --data_dir data/wikitext103 --out results/small/eval_lega.json
python -m lega.ruler     --ckpt out/small/lega/model.pt --data_dir data/wikitext103 --out results/small/ruler_lega.json
python -m lega.latency   --config configs/small.json --data_dir data/wikitext103 --ckpt_dir out/small --out results/small/latency.json
python -m lega.analysis  --ckpt out/small/lega/model.pt --data_dir data/wikitext103 --out_dir results/small
python -m lega.plots     --results_dir results/small --train_root out/small
python -m lega.sanity    --config configs/small.json --data_dir data/wikitext103 --variant lega --out results/sanity_lega.json
python -m lega.curves    --run_dir out/small/lega/seed1337   # training-curve diagnosis (overfit/underfit/divergence/NaN)
python -m lega.badcase   --results_dir results/small         # error attribution (RULER + LM worst windows) + improvement synthesis
python -m lega.hypotheses --verify                           # pre-registration integrity (0=confirmatory-valid; 1=deleted/edited/unrecorded)
python -m tests.run_tests   # unit tests (no pytest dependency)
```

Result-analysis protocol (see the matching sections in `summary.md`): diagnose the
training curves rather than final values only — besides overfitting/underfitting/
divergence, pre-clip **gradient-norm spikes/non-finite values** are reported
independently (an early signal of NaN/divergence that the loss curve may not show
yet); a plateaued training loss with rising validation loss is also diagnosed as
overfitting (generalization degrades before the training loss does, so it is not
misread as underfitting), while a training loss that is still falling with a
stalled validation loss is diagnosed separately as an **overfitting trend**
(regularize/early-stop; do not extend training on that basis); for NaN follow the
checklist order data → code (log(0)/
division by zero/softmax overflow; forward hooks localize the first module whose
finite input yields a non-finite output, the `origin`) → mixed-precision underflow →
gradient explosion/LR (see `nan_report.json`); both the data and the activation
checks inspect the failing step's batches first (prefetching overwrites the pending
ones), and a non-finite validation loss switches to the real validation windows
(training batches are only a fallback when windowing fails); when the mixed-precision
path produces non-finite values, the same batch is **re-run in fp32 to isolate the
precision factor** (fp32 finite → mixed-precision numerics; fp32 also non-finite →
precision ruled out). The "early stopping" recommended for overfitting is an
**optional, actually implemented mechanism** (`--early_stop_patience N`, or config
`early_stop_patience`/`early_stop_min_delta`: stop when val loss shows no material
improvement for N consecutive evals; off by default so controlled comparisons are
unaffected by it; when enabled, actual steps may differ across runs, `eval.json`
records both planned and actual steps, the controlled-variable audit compares
planned steps and the summary marks early-stopped runs — downstream evaluation
still uses the val-selected `model_best.pt`). Attribute
every error case (each case also gets a coarse attribution axis: data/construction,
difficulty boundary, or model/method), and use the cross-variant
comparison to separate "shared failures (data/difficulty)" from "method-specific
blind spots"; the attribution is then **synthesized into actionable improvement
directions** (trigger evidence → suggestion → next experiment, deterministic
thresholds in `lega/badcase.py`) that appear in the per-case reports, in
`badcase_summary.json` and in `summary.md` — error analysis must not stop at the
category table, since hard cases are the most important source of new ideas; the accuracy-efficiency trade-off is judged explicitly, where the cost
is the maximum relative increase over the baseline among **prefill/decode inference
latency, training ms/step, training peak GPU memory, parameters, FLOPs and
inference peak allocation**. Latency is the median
of per-iteration timings and **differences within the combined measurement spread
([p10,p90] added in quadrature) are treated as indistinguishable and excluded from
the cost (marked †)**; a gain without
significance, or a cost far exceeding the gain, is flagged — see the
`tradeoff_verdict` rules in §2. A "0.5% gain at 10× training/inference cost" is
rejected outright.

---

## 4. Repository structure

```
lega-attention/
├── run_all.py                 # one-command pipeline + summary table
├── configs/
│   ├── tiny.json              # ~6.8M smoke config (CPU-friendly)
│   └── small.json             # ~30M full config (hours/variant on one GPU)
├── lega/
│   ├── attention.py           # core: causal attention for 9 variants (LEGA gate + ablation matrix)
│   ├── model.py               # GPT backbone + RoPE + KV-cache generation + param tools
│   ├── data.py                # tokenization (GPT-2 BPE / char fallback, vocab fit on train only) + bin cache + batch sampling
│   ├── datacheck.py           # leakage-proofing: temporal/document split, decontamination, token hard checks, shift screening, test-access ledger
│   ├── train.py               # training loop (AdamW, cosine, bf16, optional W&B) + systematic NaN audit (activation hooks) + provenance manifest
│   ├── evaluate.py            # held-out test PPL (val fallback)
│   ├── ruler.py               # simplified RULER: 5 tasks, token-level construction, gold spans kept
│   ├── latency.py             # prefill / decode latency benchmark
│   ├── analysis.py            # mechanism analysis: concentrated≠correct, gate calibration, AUROC + PR-AUC (primary under imbalance) + gate F1 (three averages + always/never-write straw men)
│   ├── ngram.py               # simple statistical baseline: unigram/bigram smoothed n-gram PPL (α on val; test opt-in + ledger; pipeline matches the neural eval split by default)
│   ├── curves.py              # training-curve diagnosis (overfit/underfit/divergence/NaN + grad spikes) + recommendations
│   ├── badcase.py             # error attribution (RULER + LM worst windows) + cross-variant comparison + improvement synthesis
│   ├── sanity.py              # overfit-small-batch sanity check (pre-training gate)
│   ├── repro.py               # code version / data fingerprint / runtime / seed audit / provenance
│   ├── experiment.py          # append-only experiment ledger (code/config/seed/data/result hashes per run)
│   ├── hypotheses.py          # pre-registration integrity: lock baseline + amendment chain + tamper check (CLI --verify/--init/--amend)
│   ├── tuning.py              # baseline (`standard`) lr/weight-decay grid (val only; multi-seed mean selection, default_is_optimal verdict)
│   ├── plots.py               # plots (PNG, including training curves)
│   └── utils.py               # seed/JSON/AUROC/PR-AUC/F1/Pearson (no scipy)
├── preregistration/           # deliberately not under docs/ (pipeline cleanup deletes non-whitelisted files there)
│   ├── hypotheses.json        # pre-registration registry (never delete/move; amendments must be traced)
│   └── hypotheses.lock.json   # lock baseline + amendment chain (anti silent rewrite/deletion)
├── docs/
│   └── design_notes.md        # design notes, statistical-inference and result-analysis protocols
├── scripts/
│   ├── prepare_data.py        # WikiText-103 download (parquet→zip→rows fallbacks)
│   ├── make_dummy_data.py     # offline dummy corpus (smoke)
│   └── run_smoke.sh           # one-command smoke
├── tests/                     # unit tests (math properties / KV cache / RULER construction)
└── results/                   # experiment outputs (gitignored, .gitkeep placeholder)
```

## 5. Implementation details and caveats

- **PPL display bound**: exact `val_loss` is always saved in JSON; to avoid `exp(loss)` overflow drowning the tables for divergent models, `val_ppl` is recorded as `exp(20)` when `loss>20` with `val_ppl_clamped: true` — compare `val_loss` in that case.
- **Numerical stability of $c$**: `max_p = exp(zmax − LSE)` is always bounded in (0,1]; clamped to `1e-30` before `log`. Uniform attention → `c=0` is unit-tested (`tests/test_attention.py`).
- **Definition of $n$**: the number of readable tokens for query position $t$ is $t+1$ (causal); null slots don't count; `null`-variant null keys get no RoPE (position-independent slots).
- **Gate query term**: the gate's $q$ uses the **pre-RoPE** query content vector; attention scoring still uses rotated $q,k$. Length/absolute position thus enters the gate only via the explicit $\log n$, avoiding the causal confounding in the method's claims.
- **Reproducibility**: `set_seed` fixes Python/NumPy/PyTorch (CPU/CUDA) seeds, disables cudnn benchmark and enables deterministic algorithms (`warn_only`) on CUDA, and sets `PYTHONHASHSEED` for child processes; batch sampling is driven by an explicit `np.random.default_rng(seed)` (no DataLoader workers) so data order is reproducible and unit-tested. All variants are trained on the same seed set and compared with same-seed paired statistics (`run_all.py --seeds` / `seeds` in the configs). Bit-exact reproducibility across hardware is not guaranteed, but same-machine reruns are stable; every training run also writes `run_manifest.json` (git/data fingerprint/environment/command) and evaluation JSONs embed the same provenance. An overfit-small-batch sanity check runs by default before training (`--skip_sanity` to skip).
- **torch.compile**: `"compile": true` enables torch.compile on CUDA; checkpoints are saved with wrapper-stripped parameter keys so `load_model` (evaluate/ruler/latency/analysis) loads them directly.
- **Dual paths for standard attention**: SDPA fast path and explicit path are numerically identical (unit-tested); the explicit path is used for variants needing attention statistics and for fair latency comparison (`standard` vs `standard_sdpa` both measured).
- **KV cache**: gated variants compute the gate token-by-token during incremental decoding; cache equivalence is unit-tested (full forward vs token-by-token).
- **Tokenization**: default GPT-2 BPE (tiktoken, needs network on first use); offline, `--tokenizer auto` falls back to character level (PPL not directly comparable to literature, but variant comparisons remain valid).
- **Device**: `small.json` expects GPU + bf16; CPU is only for tiny smoke runs. For full experiments, start with the `lega_no_len` vs `lega` comparison (length calibration is the core claim).
- **Known limitations**: the 30M model's absolute RULER accuracy is low (small models have limited retrieval ability); the focus is **relative differences between variants** and **mechanistic metrics**, not absolute scores; `fwe` gives high-frequency words k≥3 occurrences vs 2 for the rest and scores the answer as a word set, so frequency ties are structurally impossible; at very small contexts individual `fwe`/`vt` samples are skipped for lack of room (recorded in the JSON `_skipped` field; the skipped set is identical across variants).

## 6. License

MIT
