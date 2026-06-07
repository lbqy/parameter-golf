# Phase 3 Milestone: Q43

Date: 2026-06-07

Constraint: single H100, 1 hour training per run, final submission <= 16 MB.

## Best Artifact

| Field | Value |
| --- | --- |
| RUN_ID | `exp_s3_q43_r15_clip_top4_rank4_export` |
| Artifact | `results/experiments/exp_s3_q43_r15_clip_top4_rank4_export/final_model.gptq.ptz` |
| Roundtrip BPB | `1.16735413` |
| Total bytes | `15,753,494` |
| Training checkpoint | `exp_s3_r15_b3_qkgain5_hparam_tied004_muon097_recur_l3_5_start030_1xh100/final_model.pt` |

## Winning Recipe

Training:

```text
SP8192
TRAIN_SEQ_LEN=4096
MAX_WALLCLOCK_SECONDS=3600
QK_GAIN_INIT=5.0
BETA2=0.99
GRAD_CLIP_NORM=0.3
MLP_CLIP_SIGMAS=12
EMBED_CLIP_SIGMAS=15
TIED_EMBED_LR=0.04
MUON_MOMENTUM=0.97
RECURRENCE_EXTRA_PASSES=1
RECURRENCE_START_LAYER=3
RECURRENCE_END_LAYER=5
RECURRENCE_START_FRAC=0.30
```

Export:

```text
QUANT_MODE=gptq
COMPRESSOR=brotli
FRESH_MODEL_AFTER_QUANT=1
GPTQ_SCALE_MODE=quantile
GPTQ_SCALE_FLOOR=1
GPTQ_ERROR_SCALE=1
MATRIX_BITS=6
EMBED_BITS=8
GPTQ_CALIBRATION_BATCHES=16
LQER_ENABLED=1
LQER_RANK=4
LQER_TOP_K=4
LQER_ASYM_ENABLED=1
LQER_ASYM_GROUP=64
RECURRENCE_ACTIVE=1
```

## Evolution Path

| Step | Best | Roundtrip BPB | Main gain |
| --- | --- | ---: | --- |
| B3 GPTQ/LQER baseline | `exp_gptq_b3_fresh_i6e8_quantile_err1_lqer` | 1.17661956 | Correct packed GPTQ + LQER |
| QK/hparam stack | `exp_s3_h11_b3_qkgain5_hparam_tiedembedlr004_1xh100` | 1.17163042 | `QK_GAIN_INIT=5.0`, clip stack, `TIED_EMBED_LR=0.04` |
| Recurrence | `exp_s3_r4_b3_qkgain5_hparam_tied004_recur_l3_5_start035_1xh100` | 1.17006027 | L3-L5 one extra pass after train frac 0.35 |
| R4 quant sweep | `exp_s3_q26_r4_clip_top4_rank8_calib64_export` | 1.16955543 | R4 export tuning |
| Muon + recurrence | `exp_s3_r8_b3_qkgain5_hparam_tied004_muon097_recur_l3_5_start035_1xh100` | 1.16835283 | `MUON_MOMENTUM=0.97` stacks with recurrence |
| Start-frac tuning | `exp_s3_r15_b3_qkgain5_hparam_tied004_muon097_recur_l3_5_start030_1xh100` | 1.16744173 | recurrence start 0.30 |
| Final export sweep | `exp_s3_q43_r15_clip_top4_rank4_export` | 1.16735413 | R15 + LQER top4 rank4 |

## Negative Results

- Coprime loader was strongly negative in current implementation.
- Warmdown/min-lr was negative.
- Score-first lm-head LoRA TTT MVP was strongly negative; richer phased TTT remains untested.
- R15 export tuning with more calibration, rank8, top5, or GPTQ error scale sweeps did not beat Q43.
- `MUON_MOMENTUM=0.96/0.98`, `TIED_EMBED_LR=0.035/0.045`, and recurrence starts 0.25/0.28/0.32/0.40/0.45/0.50/0.60 did not beat Q43 after roundtrip quantization.
- CaseOps was blocked by missing raw doc stream / sidecars in the current environment.

## SOTA Gap Hypotheses

The current best is a strong local optimum for the present root script, but it is still far from records-level SOTA because:

- The data/tokenization path is still plain SP8192 FineWeb, not CaseOps or comparable byte-sidecar-aware preprocessing.
- The model still has a conventional compact GPT block; records-level runs often combine recurrence with gates, residual lanes, SmearGate/BOS-safe document handling, or phased adaptation.
- The TTT implementation tested here was only a minimal lm-head LoRA eval MVP, not the richer legal/phased TTT used in stronger records.
- The compression stack uses brotli + packed GPTQ/LQER, but not per-group layout, similarity sorting, lrzip, or more specialized submission packing.
- The search is constrained to one-hour single-H100 runs, so deeper recurrence and slower but better late-training recipes lose too much step budget unless the architecture compensates.
