Below is the English translation of the experimental plan from the uploaded file. 

# Experimental Plan: Baseline Ablation Study

## Objective

Evaluate three optimization directions on top of the baseline architecture:

1. Sequence length
2. Tokenizer vocabulary size
3. GPTQ quantization vs. int8+zlib compression

Only one variable is changed per experiment. After each training run, both **int8+zlib** and **GPTQ+pergroup** compressed models are generated for comparison.

---

## Experiment Matrix

| Experiment ID | Tokenizer | Seq Len | Variable   | Description                  |
| ------------- | --------- | ------- | ---------- | ---------------------------- |
| E0            | SP1024    | 1024    | ?          | GPTQ baseline calibration    |
| A1            | SP1024    | 2048    | seq_len ?  | Pure sequence-length gain    |
| A2            | SP1024    | 4096    | seq_len ?? | Longer sequence length       |
| B1            | SP8192    | 1024    | vocab ?    | Pure vocabulary-size gain    |
| B2            | SP8192    | 2048    | Combined   | Vocabulary + longer sequence |

**Execution Order:** E0 ? A1 ? A2 ? B1 ? B2

---

## Fixed Configuration (Shared by All Experiments)

### Model Architecture

| Parameter      | Value |
| -------------- | ----- |
| NUM_LAYERS     | 9     |
| MODEL_DIM      | 512   |
| NUM_HEADS      | 8     |
| NUM_KV_HEADS   | 4     |
| MLP_MULT       | 2     |
| TIE_EMBEDDINGS | 1     |

### Training Parameters

| Parameter             | Value  |
| --------------------- | ------ |
| MAX_WALLCLOCK_SECONDS | 5100   |
| WARMUP_STEPS          | 20     |
| ITERATIONS            | 20000  |
| VAL_LOSS_EVERY        | 200    |
| TRAIN_LOG_EVERY       | 50     |
| SEED                  | 1337   |
| TRAIN_BATCH_TOKENS    | 524288 |

### Optimizer Parameters

| Parameter                  | Value |
| -------------------------- | ----- |
| MATRIX_LR                  | 0.04  |
| TIED_EMBED_LR              | 0.05  |
| SCALAR_LR                  | 0.04  |
| MUON_MOMENTUM              | 0.95  |
| MUON_MOMENTUM_WARMUP_START | 0.85  |
| MUON_MOMENTUM_WARMUP_STEPS | 500   |

### GPTQ Compression Parameters

| Parameter                | Value    |
| ------------------------ | -------- |
| COMPRESSOR               | pergroup |
| MATRIX_BITS              | 6        |
| EMBED_BITS               | 8        |
| MATRIX_CLIP_SIGMAS       | 12.85    |
| ATTN_CLIP_SIGMAS         | 13.0     |
| MLP_CLIP_SIGMAS          | 11.5     |
| EMBED_CLIP_SIGMAS        | 3.0      |
| GPTQ_CALIBRATION_BATCHES | 32       |
| LQER_ENABLED             | 1        |
| LQER_RANK                | 4        |
| LQER_TOP_K               | 3        |
| LQER_FACTOR_BITS         | 4        |
| LQER_ASYM_ENABLED        | 1        |
| LQER_ASYM_GROUP          | 32       |

---

## Dataset Paths

| Experiment | DATA_PATH                                                                                                         | TOKENIZER_PATH                                                               | VOCAB_SIZE |
| ---------- | ----------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- | ---------- |
| E0/A1/A2   | `./data/datasets/fineweb10B_sp1024/`                                                                              | `./data/tokenizers/fineweb_1024_bpe.model`                                   | 1024       |
| B1/B2      | `./data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved` | `./data/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model` | 8192       |

---

## Command Templates

### E0 ? SP1024 / seq1024 / GPTQ Baseline Calibration

```bash
cd /root/data1/llm_hw_cc/parameter-golf
NCCL_IB_DISABLE=1 \
CUDA_VISIBLE_DEVICES=0,1 \
RUN_ID=exp_e0_sp1024_seq1024 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
TRAIN_SEQ_LEN=1024 \
MAX_WALLCLOCK_SECONDS=5100 \
VAL_LOSS_EVERY=200 \
TRAIN_LOG_EVERY=50 \
COMPRESSOR=pergroup \
MATRIX_BITS=6 EMBED_BITS=8 \
MATRIX_CLIP_SIGMAS=12.85 ATTN_CLIP_SIGMAS=13.0 MLP_CLIP_SIGMAS=11.5 \
EMBED_CLIP_SIGMAS=3.0 \
GPTQ_CALIBRATION_BATCHES=32 \
LQER_ENABLED=1 LQER_RANK=4 LQER_TOP_K=3 \
LQER_FACTOR_BITS=4 LQER_ASYM_ENABLED=1 LQER_ASYM_GROUP=32 \
torchrun --standalone --nproc_per_node=2 train_gpt.py
```

### A1 ? SP1024 / seq2048

```bash
# Same as above, only change TRAIN_SEQ_LEN and RUN_ID
TRAIN_SEQ_LEN=2048 \
RUN_ID=exp_a1_sp1024_seq2048 \
```

### A2 ? SP1024 / seq4096

```bash
# Same as above, only change TRAIN_SEQ_LEN and RUN_ID
TRAIN_SEQ_LEN=4096 \
RUN_ID=exp_a2_sp1024_seq4096 \
```

### B1 ? SP8192 / seq1024

```bash
cd /root/data1/llm_hw_cc/parameter-golf
NCCL_IB_DISABLE=1 \
CUDA_VISIBLE_DEVICES=0,1 \
RUN_ID=exp_b1_sp8192_seq1024 \
DATA_PATH=./data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved \
TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model \
VOCAB_SIZE=8192 \
TRAIN_SEQ_LEN=1024 \
MAX_WALLCLOCK_SECONDS=5100 \
VAL_LOSS_EVERY=200 \
TRAIN_LOG_EVERY=50 \
COMPRESSOR=pergroup \
MATRIX_BITS=6 EMBED_BITS=8 \
MATRIX_CLIP_SIGMAS=12.85 ATTN_CLIP_SIGMAS=13.0 MLP_CLIP_SIGMAS=11.5 \
EMBED_CLIP_SIGMAS=3.0 \
GPTQ_CALIBRATION_BATCHES=32 \
LQER_ENABLED=1 LQER_RANK=4 LQER_TOP_K=3 \
LQER_FACTOR_BITS=4 LQER_ASYM_ENABLED=1 LQER_ASYM_GROUP=32 \
torchrun --standalone --nproc_per_node=2 train_gpt.py
```

### B2 ? SP8192 / seq2048

```bash
# Same as B1, only change TRAIN_SEQ_LEN and RUN_ID
TRAIN_SEQ_LEN=2048 \
RUN_ID=exp_b2_sp8192_seq2048 \
```

---

## Expected Output Directory Structure

```text
artifacts/exp_e0_sp1024_seq1024/
??? final_model.pt                       # Original FP32 weights
??? final_model.int8.ptz                 # int8+zlib compressed artifact
??? final_model.pergroup.ptz             # GPTQ+pergroup compressed artifact

logs/exp_e0_sp1024_seq1024.txt           # Training metrics log
logs/exp_e0_sp1024_seq1024_console.log   # Console output
```

---

## Experiment Report Template

For each completed experiment, create:

```text
experiments/exp_<id>_<tokenizer>_seq<len>.md
```

````markdown
# Experiment <ID>: <Description>

## Command

```bash
...full command...
````

## Training Results

| Metric            | Value  |
| ----------------- | ------ |
| Stopping Step     | xxx    |
| Training Time     | xxx ms |
| Pre-quant val_bpb | x.xxxx |

## Quantization Comparison

| Method                  |   BPB  | Artifact Size | ?16MB |
| ----------------------- | :----: | :-----------: | :---: |
| int8+zlib roundtrip     | x.xxxx |    xx.x MB    |  ?/?  |
| GPTQ+pergroup roundtrip | x.xxxx |    xx.x MB    |  ?/?  |

## Conclusion

...

```

---

## Final Summary Matrix (Phase 1 — Completed)

| ID | Tokenizer | Seq | int8+zlib BPB | GPTQ BPB | int8 Size | Variable |
|----|-----------|-----|:---:|:---:|:---:|----------|
| E0 | SP1024 | 1024 | 1.2320 | N/A | 15.87 MB ✓ | baseline |
| A1 | SP1024 | 2048 | 1.2155 | N/A | 15.87 MB ✓ | seq_len |
| A2 | SP1024 | 4096 | 1.2113 | N/A | 15.85 MB ✓ | seq_len |
| B1 | SP8192 | 1024 | 1.1940 | N/A | 19.38 MB ✗ | vocab |
| B2 | SP8192 | 2048 | 1.1815 | N/A | 19.34 MB ✗ | combined |

\* E0 from `baseline_sp1024_1xh100_1h_seed1337_torch26.summary.json`; A1–B2 from `experiments/rootbaseline_ablation_summary.md`. GPTQ not implemented in root script.

### Comparative Analysis

- **Sequence-length gain:** E0 ? A1 (1024?2048), E0 ? A2 (1024?4096)
- **Vocabulary-size gain:** E0 ? B1 (SP1024?SP8192, both seq1024)
- **Combined effect:** E0 ? B2 (overall improvement)
- **GPTQ vs. int8+zlib:** Compare both compression methods within each experiment.

---

## Phase 1 Status (Completed)

Phase 1 ablation (E0/A1/A2/B1/B2) is complete. Summary: `experiments/rootbaseline_ablation_summary.md`.

| ID | int8+zlib BPB | Total Size | <=16MB |
| --- | ---: | ---: | :---: |
| E0 (SP1024 seq1024) | 1.2320 | 15.87 MB | yes |
| A2 (SP1024 seq4096) | 1.2113 | 15.85 MB | yes |
| B1 (SP8192 seq1024) | 1.1940 | 19.38 MB | no |
| B2 (SP8192 seq2048) | 1.1815 | 19.34 MB | no |

**Phase 1 conclusion:** SP8192 (especially B2 seq2048) gives the best pre-quant quality, but default int8+zlib overshoots the 16 MB cap by ~3.3 MB. Phase 2 focuses on **quantization/compression**, not architecture or training length changes.

**Reference runs outside Phase 1** (same root `train_gpt.py`, different wallclock/GPU):

| Run | Config | Roundtrip BPB | Total Size |
| --- | --- | ---: | ---: |
| `sp8192_int8_2xh100_30m` | int8 default | 1.1934 | 19.38 MB |
| `sp8192_clipsearch_int6int7_2xh100_30m` | matrix int6, embed int7, clip-search | 1.2191 | 15.81 MB |

RTN int6/int7 fits under 16 MB but loses ~0.026 BPB vs int8; records SOTA closes this gap with **Hessian GPTQ + SDClip + LQER**.

---

## Quantization Strategy Survey (`results/` + `records/`)

Survey of mainstream compression paths relevant to SP8192 under the 16 MB cap. Ordered from **already in root `train_gpt.py`** to **records SOTA**.

### Tier 0 — Current root baseline (`train_gpt.py`)

| Strategy | Mechanism | Typical use | SP8192 note |
| --- | --- | --- | --- |
| **int8 per-row + zlib** | Per-row clip at 99.99984th percentile; small tensors (<65536) kept fp16; control params fp16/fp32; zlib L9 | Default export | B2: 1.1815 BPB, **19.34 MB** — quality good, size bad |
| **Packed intN RTN** | `MATRIX_QUANT_BITS` / `EMBED_QUANT_BITS` (2–8), bit-packing | Lower bitwidth without GPTQ | int6/int7: **15.81 MB**, BPB **1.219** — size good, quality bad |
| **Clip-search RTN** | `QUANT_CLIP_SEARCH=1` sweeps clip percentiles by MSE | Tune RTN without GPTQ | Helps int6/int7; still not Hessian-aware |

**Important:** Root packed low-bit is **post-training round-to-nearest (RTN) + clip-search**, not full GPTQ. Do not label it "GPTQ" in run IDs or reports.

### Tier 1 — Lightweight records tricks (moderate code / env only)

| Strategy | Source | Idea | Expected effect |
| --- | --- | --- | --- |
| **Asymmetric matrix/embed bits** | `ref_config.md`, PR #1586 | int6 matrices + int7/int8 embed | Embed is ~25% of params; protect it while shrinking attn/MLP |
| **Per-layer mixed precision** | `2026-03-19_10L_MixedPrecision` | Middle layers int6, early/late int8 | ~1.6 MB saved on 10L SP1024 with 0.002 BPB loss; adapt layer groups for 9L |
| **Magnitude pruning pre-quant** | `2026-03-20_10L_Int5MLP`, PR #609 | Zero smallest ~3% weights before quant | Lowers entropy → better zlib; used with int6 export |
| **Alternate compressors** | Multiple records | brotli, zstd L22, lzma preset 9 | Root only has zlib; brotli/zstd often beat zlib on quant payloads |

### Tier 2 — GPTQ family (records SOTA backbone)

| Strategy | Source | Idea | Typical config |
| --- | --- | --- | --- |
| **SDClip GPTQ** | PR #1394 (`2026-04-05_SP8192_GPTQ-Embeddings`) | Clip threshold = `sigma * CLIP_SIGMAS / clip_range` per row; no quantile search | `MATRIX_CLIP_SIGMAS=12.85`, `ATTN=13.0`, `MLP=11.5`, `EMBED=3.0–15.0` |
| **Full Hessian GPTQ** | PR #535, #609, #1797 | Block-wise GPTQ with Hessian error compensation; embed GPTQ | `MATRIX_BITS=6`, `EMBED_BITS=7/8`, `GPTQ_CALIBRATION_BATCHES=16–32` |
| **AR self-gen calibration** | `2026-03-25_ValCalib_GPTQ_XSA` | Model generates calibration tokens post-train (legal under strict 600s rules) | Less critical for 1×H100 1h, but good default for compliance |
| **LQER asymmetric** | PR #1797 | Rank-4 int4 correction on top-3 highest-error tensors | `LQER_RANK=4`, `LQER_TOP_K=3`, `LQER_FACTOR_BITS=4`, `LQER_ASYM_GROUP=32–64` |
| **int8-per-row gates** | PR #1736 | Special path for small gated-attn tensors | Only if gated attn is added |

SOTA example (`2026-04-27_SP8192_LQER_...`, 1.061 BPB): **GPTQ int6 + int7 embed + LQER + pergroup compression** — far beyond root architecture, but the **quant stack** is the template for closing the B2 gap.

### Tier 3 — Compression beyond quant bitwidth

| Strategy | Source | Idea |
| --- | --- | --- |
| **pergroup + lrzip ZPAQ** | PR #1851 / `COMPRESSOR=pergroup` | Bucket tensors by role; L1 row similarity sort; lrzip on hot groups, brotli on rest |
| **Entropy-aware bitwidth** | PR #1394 README | Wider clip can yield *smaller* compressed int5 than narrow int4 — optimize **compressed size**, not raw bits |

### Recommended port order into root `train_gpt.py`

1. **Now (env-only):** Q1–Q3 below on B2 training backbone.
2. **Next code milestone:** SDClip GPTQ + Hessian collection (from `records/track_10min_16mb/2026-04-29_SmearGateBOSFix_3Seed_1.06141/train_gpt.py` or PR #1797 subset — no architecture changes).
3. **Then:** LQER + optional brotli/pergroup if still over budget.

---

## Phase 2: SP8192 Quantization Experiments

### Objective

Starting from the **B2 training configuration** (SP8192, seq2048, 1×H100, 3600s), find a compression path that satisfies:

- `Total submission size int8+zlib` (or named equivalent) **<= 16,777,216 bytes**
- Minimize `final_*_roundtrip_exact val_bpb`
- Target: approach B2 pre-quant **1.1744** or int8 roundtrip **1.1815**, not RTN int6/int7 **1.219**

### Fixed training backbone (all Q experiments)

Same as B2 / Phase 1 SP8192 runs:

| Parameter | Value |
| --- | --- |
| DATA_PATH | `/base/datasets/SP8192/datasets/fineweb10B_sp8192` |
| TOKENIZER_PATH | `/base/datasets/SP8192/tokenizers/fineweb_8192_bpe.model` |
| VOCAB_SIZE | 8192 |
| TRAIN_SEQ_LEN | 2048 |
| MAX_WALLCLOCK_SECONDS | 3600 |
| SEED | 1337 |
| GPU | 1× H100 (`torchrun --nproc_per_node=1`) |
| Architecture | Default 9L×512 (unchanged) |
| VAL_LOSS_EVERY | 0 (final roundtrip eval only) |
| TRAIN_LOG_EVERY | 200 |

Only **quantization/export settings** change between Q1–Q4.

### Experiment matrix

| ID | Variable | Matrix bits | Embed bits | Extra quant settings | Code change | Hypothesis |
| --- | --- | ---: | ---: | --- | --- | --- |
| **Q1** | Symmetric RTN + clip-search | 7 | 7 | `QUANT_CLIP_SEARCH=1` | None | Middle ground between int8 (oversize) and int6/int7 (1.219 BPB) |
| **Q2** | Asymmetric RTN + clip-search | 6 | 8 | `QUANT_CLIP_SEARCH=1` | None | Shrink attn/MLP aggressively; keep embed at int8 per PR #1586 / `ref_config.md` |
| **Q3** | Known asymmetric RTN baseline | 6 | 7 | `QUANT_CLIP_SEARCH=1` | None | Reproduce `sp8192_clipsearch_int6int7` on **B2 1h** weights for apples-to-apples vs B2 int8 |
| **Q4** | GPTQ + SDClip + LQER | 6 | 7 | See GPTQ table below | **Port from records** | Records path to ~15.9 MB with ~0.01–0.04 BPB quant gap vs pre-quant |

**Execution order:** Q3 (sanity) → Q1 → Q2 → Q4 (after GPTQ port).

### Q4 GPTQ parameters (from records, applied to root 9L model)

| Parameter | Value | Notes |
| --- | --- | --- |
| COMPRESSOR | pergroup (stretch) or brotli (minimal port) | pergroup needs lrzip; start with brotli/zstd if simpler |
| MATRIX_BITS | 6 | |
| EMBED_BITS | 7 | int7 embed is SOTA default for SP8192 |
| MATRIX_CLIP_SIGMAS | 12.85 | |
| ATTN_CLIP_SIGMAS | 13.0 | |
| MLP_CLIP_SIGMAS | 11.5 | |
| EMBED_CLIP_SIGMAS | 14.0 | SOTA uses 14.0 (tighter than PLAN Phase 1 `3.0`) |
| GPTQ_CALIBRATION_BATCHES | 32 | Reduce to 16 if export time is tight on 1h job |
| LQER_ENABLED | 1 | |
| LQER_RANK | 4 | |
| LQER_TOP_K | 3 | |
| LQER_FACTOR_BITS | 4 | |
| LQER_ASYM_ENABLED | 1 | |
| LQER_ASYM_GROUP | 64 | |

Reference implementation: `records/track_10min_16mb/2026-04-29_SmearGateBOSFix_3Seed_1.06141/train_gpt.py` (GPTQ/LQER/export blocks only — strip architecture extras).

### Command templates (container paths)

Shared prefix:

```bash
cd /base/project/parameter-golf/results/experiments/<RUN_ID>
CUDA_VISIBLE_DEVICES=6 \
RUN_ID=<RUN_ID> \
SEED=1337 \
DATA_PATH=/base/datasets/SP8192/datasets/fineweb10B_sp8192 \
TOKENIZER_PATH=/base/datasets/SP8192/tokenizers/fineweb_8192_bpe.model \
VOCAB_SIZE=8192 \
TRAIN_SEQ_LEN=2048 \
MAX_WALLCLOCK_SECONDS=3600 \
WARMUP_STEPS=20 \
ITERATIONS=20000 \
VAL_LOSS_EVERY=0 \
TRAIN_LOG_EVERY=200 \
TRAIN_BATCH_TOKENS=524288 \
/opt/conda/bin/torchrun --standalone --nproc_per_node=1 /base/project/parameter-golf/train_gpt.py
```

**Q1 — int7/int7 clip-search**

```bash
RUN_ID=exp_q1_sp8192_seq2048_int7int7_clipsearch \
MATRIX_QUANT_BITS=7 \
EMBED_QUANT_BITS=7 \
QUANT_CLIP_SEARCH=1
```

**Q2 — int6 matrix / int8 embed clip-search**

```bash
RUN_ID=exp_q2_sp8192_seq2048_int6m_int8e_clipsearch \
MATRIX_QUANT_BITS=6 \
EMBED_QUANT_BITS=8 \
QUANT_CLIP_SEARCH=1
```

**Q3 — int6/int7 clip-search (B2 backbone repro)**

```bash
RUN_ID=exp_q3_sp8192_seq2048_int6int7_clipsearch \
MATRIX_QUANT_BITS=6 \
EMBED_QUANT_BITS=7 \
QUANT_CLIP_SEARCH=1
```

**Q4 — GPTQ + LQER (after code port)**

```bash
RUN_ID=exp_q4_sp8192_seq2048_gptq_lqer \
COMPRESSOR=pergroup \
MATRIX_BITS=6 EMBED_BITS=7 \
MATRIX_CLIP_SIGMAS=12.85 ATTN_CLIP_SIGMAS=13.0 MLP_CLIP_SIGMAS=11.5 \
EMBED_CLIP_SIGMAS=14.0 \
GPTQ_CALIBRATION_BATCHES=32 \
LQER_ENABLED=1 LQER_RANK=4 LQER_TOP_K=3 \
LQER_FACTOR_BITS=4 LQER_ASYM_ENABLED=1 LQER_ASYM_GROUP=64
```

### Smoke protocol (before each 1h run)

1. Reuse a finished B2 checkpoint **or** run `MAX_WALLCLOCK_SECONDS=300` with the same quant env vars.
2. Check log lines: `Serialized model int8+zlib` / `Total submission size int8+zlib` and `final_int8_zlib_roundtrip_exact val_bpb`.
3. Proceed to 3600s only if size <= 16 MB (or Q4 equivalent artifact).

### Phase 2 report template

Save to `experiments/exp_<id>_....md`:

| Metric | Value |
| --- | --- |
| Pre-quant val_bpb | (from B2 or same run) |
| Packed RTN / GPTQ roundtrip BPB | |
| Total submission bytes | |
| <=16MB | yes/no |
| Quant gap (roundtrip − pre-quant) | |

### Phase 2 success criteria

| Priority | Criterion |
| --- | --- |
| P0 | Total size <= 16 MB |
| P1 | Roundtrip BPB < 1.20 (beat RTN int6/int7 1.219) |
| P2 | Roundtrip BPB <= 1.185 (within ~0.004 of B2 int8 1.1815) |
| P3 | Match or beat E0 SP1024 compliant baseline 1.232 while keeping SP8192 quality advantage |

### Phase 2 summary matrix (to fill)

| ID | Matrix | Embed | Method | Roundtrip BPB | Total bytes | <=16MB |
| --- | ---: | ---: | --- | ---: | ---: | :---: |
| B2 ref | 8 | 8 | int8+zlib RTN | 1.1815 | 19,343,563 | no |
| Q1 | 7 | 7 | RTN + clip-search | | | |
| Q2 | 6 | 8 | RTN + clip-search | | | |
| Q3 | 6 | 7 | RTN + clip-search | | | |
| Q4 | 6 | 7 | GPTQ + LQER | | | |
