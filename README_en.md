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
| `lega` | $g=\sigma(w^\top q+a\,c+b\log n+d)$ | **Ours** |
| `lega_no_len` | $g=\sigma(w^\top q+a\log\max_j p_j+d)$ | Ablation: no length calibration |
| `lega_no_conc` | $g=\sigma(w^\top q+b\log n+d)$ | Ablation: no concentration evidence |
| `lega_no_query` | $g=\sigma(a\,c+b\log n+d)$ | Ablation: no query term |

Parameter overhead: gating adds only **$d_{head}+3$ parameters per head** (LEGA), <0.02% of the model (`null` is $2d_{head}$ per head). Scripts print the exact numbers at the start of training.

---

## 2. Experimental design

- **Fair comparison**: all variants share the same parameter budget and the same training steps/data (same config, same seed).
- **Three evaluation axes**:
  1. **Perplexity** (WikiText-103 PPL: checkpoint selection on val, PPL reported on the held-out test split, falling back to val when test.bin is absent); windows are disjoint and evenly sampled across the whole evaluation set (all windows used when too few); `gate_mean` and PPL use the same windows and precision;
  2. **RULER long-context retrieval** (simplified reimplementation, 5 tasks): `niah_synth` (synthetic-distractor single-key retrieval), `niah_corpus` (natural-text distractors), `niah_multikey` (multi-key distractors), `fwe` (frequent-word extraction), `vt` (variable tracking); multiple context lengths + greedy decoding; for numeric tasks the EM **only accepts the first numeric answer in the generated text**, so "wrong answer first, lucky guess later" is never scored;
  3. **Actual inference latency**: prefill (last-position logits only) and token-by-token decode with KV cache, compared against the SDPA fast path.
- **Core-risk checks (not just "attention sink reduced")**: `lega/analysis.py` performs mechanistic analysis on RULER samples with gold spans:
  - **misleading_rate**: fraction of highly concentrated attention ($\max_j p_j>\tau$) whose landing point is outside the answer's numeric span (value_span); the hit test uses the exact answer-number token span rather than the whole needle sentence — direct evidence that "concentrated ≠ correct";
  - **P(hit | concentrated)** vs **P(hit | not concentrated)**;
  - **AUROC**: ability of the gate score to separate "hit / miss", plus **the difference between LEGA's full logit and the query-only term $w^\top q$**, directly answering whether length-calibrated attention statistics judge "should we write" better than the query alone.

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

Smoke mode automatically shrinks: 20 steps/variant, ctx=min(256, block_size), 2 examples per task — for verifying code correctness, not scientific conclusions.

### 3.3 Full experiments (WikiText-103 + small config)

```bash
# 1) Data (~300MB download; offline: point --local_txt at local text)
python scripts/prepare_data.py --out_dir data/wikitext103
# optional --max_tokens 30000000 to control training size

# 2) Full pipeline: train 7 variants -> PPL -> RULER -> latency -> analysis -> plots -> summary table
python run_all.py --config configs/small.json --data_dir data/wikitext103

# optional: W&B logging
wandb login
python run_all.py --config configs/small.json --data_dir data/wikitext103 --wandb
```

Outputs:

| Path | Content |
|---|---|
| `out/<tag>/<variant>/model.pt` | Checkpoint + training config |
| `out/<tag>/<variant>/train_log.jsonl` | Training/validation curves |
| `results/<tag>/summary.md` | **Summary table (paste into a report/paper draft)** |
| `results/<tag>/eval_*.json` | Held-out test PPL per variant (val fallback when test.bin is absent) |
| `results/<tag>/ruler_*.json` | RULER 5 tasks × context lengths |
| `results/<tag>/latency.json` | prefill / decode latency |
| `results/<tag>/analysis_*.csv|json` | Per (layer, head, query position) gate records and aggregates |
| `results/<tag>/fig_*.png` | All plots |

### 3.4 Running each stage separately

```bash
python -m lega.train     --config configs/small.json --variant lega --data_dir data/wikitext103
python -m lega.evaluate  --ckpt out/small/lega/model.pt --data_dir data/wikitext103 --out results/small/eval_lega.json
python -m lega.ruler     --ckpt out/small/lega/model.pt --data_dir data/wikitext103 --out results/small/ruler_lega.json
python -m lega.latency   --config configs/small.json --data_dir data/wikitext103 --ckpt_dir out/small --out results/small/latency.json
python -m lega.analysis  --ckpt out/small/lega/model.pt --data_dir data/wikitext103 --out_dir results/small
python -m lega.plots     --results_dir results/small
python -m tests.run_tests   # unit tests (no pytest dependency)
```

---

## 4. Repository structure

```
lega-attention/
├── run_all.py                 # one-command pipeline + summary table
├── configs/
│   ├── tiny.json              # ~6.8M smoke config (CPU-friendly)
│   └── small.json             # ~30M full config (hours/variant on one GPU)
├── lega/
│   ├── attention.py           # core: causal attention for 7 variants (LEGA gate lives here)
│   ├── model.py               # GPT backbone + RoPE + KV-cache generation + param tools
│   ├── data.py                # tokenization (GPT-2 BPE / char fallback) + bin cache + batch sampling
│   ├── train.py               # training loop (AdamW, cosine, bf16, optional W&B)
│   ├── evaluate.py            # held-out test PPL (val fallback)
│   ├── ruler.py               # simplified RULER: 5 tasks, token-level construction, gold spans kept
│   ├── latency.py             # prefill / decode latency benchmark
│   ├── analysis.py            # mechanism analysis: concentrated≠correct, gate calibration, AUROC
│   ├── plots.py               # plots (PNG)
│   └── utils.py               # seed/JSON/AUROC/Pearson (no scipy)
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
- **Reproducibility**: `set_seed` disables cudnn benchmark and enables deterministic algorithms (`warn_only`) on CUDA; same-seed reruns are used for fair variant comparison.
- **torch.compile**: `"compile": true` enables torch.compile on CUDA; checkpoints are saved with wrapper-stripped parameter keys so `load_model` (evaluate/ruler/latency/analysis) loads them directly.
- **Dual paths for standard attention**: SDPA fast path and explicit path are numerically identical (unit-tested); the explicit path is used for variants needing attention statistics and for fair latency comparison (`standard` vs `standard_sdpa` both measured).
- **KV cache**: gated variants compute the gate token-by-token during incremental decoding; cache equivalence is unit-tested (full forward vs token-by-token).
- **Tokenization**: default GPT-2 BPE (tiktoken, needs network on first use); offline, `--tokenizer auto` falls back to character level (PPL not directly comparable to literature, but variant comparisons remain valid).
- **Device**: `small.json` expects GPU + bf16; CPU is only for tiny smoke runs. For full experiments, start with the `lega_no_len` vs `lega` comparison (length calibration is the core claim).
- **Known limitations**: the 30M model's absolute RULER accuracy is low (small models have limited retrieval ability); the focus is **relative differences between variants** and **mechanistic metrics**, not absolute scores; `fwe` gives high-frequency words k≥3 occurrences vs 2 for the rest and scores the answer as a word set, so frequency ties are structurally impossible; at very small contexts individual `fwe`/`vt` samples are skipped for lack of room (recorded in the JSON `_skipped` field; the skipped set is identical across variants).

## 6. License

MIT
