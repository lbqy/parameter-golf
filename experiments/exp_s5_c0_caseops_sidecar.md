# S5-C0 CaseOps Full Prepare and Sidecar Gate

Date: 2026-06-08

This file tracks the Phase 5 CaseOps data gate. Smoke runs are recorded only as environment/compliance checks, not as optimization experiments.

## Goal

Generate the full CaseOps dataset and make root `train_gpt.py` compute BPB from `fineweb_val_bytes_*.bin` when that sidecar exists.

## Code Fixes

- `load_validation_tokens()` now excludes `fineweb_val_bytes_*.bin`; the previous `fineweb_val_*.bin` glob also matched byte sidecars.
- `eval_val()` now accepts an optional validation byte sidecar and sums bytes aligned to the shifted `y` targets.
- Logs now distinguish normal tokenizer-LUT BPB from sidecar BPB:

```text
val_bpb:enabled tokenizer_kind=sentencepiece ...
val_bpb:byte_sidecar:enabled ...
```

## Compliance Check

`/tmp/caseops_smoke` 1-step run completed after the fix. This is not an experiment result.

Key log lines:

```text
val_bpb:byte_sidecar:enabled files:1 bytes_sum:72159 bos_count:16 bad_bos_bytes:0
val_loader:shards pattern=/tmp/caseops_smoke/out/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/fineweb_val_*.bin token_files:1 tokens:22272
final_int8+zlib_roundtrip_exact val_loss:10.31520040 val_bpb:4.59325885
```

The important checks are `byte_sidecar:enabled`, `bad_bos_bytes:0`, and `token_files:1`.

## Full Prepare

Initial full-doc prepare was stopped after discovering the raw manifest contains 15,368,808 docs and would take many hours. For the 1xH100/1h experiments, the comparable SP8192 baseline uses 80 train shards, so `prepare_caseops_data.py` was patched with `--train-shards` to stop after a fixed number of full train shards and then flush validation sidecars.

Started limited prepare in container `lbqy0`:

```text
PID: 3865037
Log: /base/datasets/CaseOps/logs/prepare_caseops_80shards.log
Output dataset: /base/datasets/CaseOps/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved
Tokenizer: /base/datasets/CaseOps/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model
Target train shards: 80
```

Command:

```bash
cd /base/project/parameter-golf
CASEOPS_ROOT=/base/datasets/CaseOps
CASEOPS_REC=records/track_10min_16mb/2026-04-27_SP8192_LQER_SparseGate_BOSSmearFix_9HpStack_1.0611
nohup /opt/conda/bin/python "$CASEOPS_REC/prepare_caseops_data.py" \
  --docs "$CASEOPS_ROOT/raw/docs_selected.jsonl" \
  --out "$CASEOPS_ROOT" \
  --sp "$CASEOPS_ROOT/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model" \
  --val-docs 10000 \
  --train-shards 80 \
  > "$CASEOPS_ROOT/logs/prepare_caseops_80shards.log" 2>&1 &
```

Latest observed progress:

```text
loaded sp: vocab=8192
stopping after requested train_shards=80
done. docs=828995 train_shards=80 val_shards=1
```

Verification:

```text
counts 80 1 1
val_tokens 9662502 val_bytes_len 9662502 byte_sum 29950979
bos_count 10000 bad_bos_bytes 0 max_byte 16
CASEOPS_80SHARD_VERIFY_OK
```

## Next Gates

- Wait for full prepare to finish and record train/val/val_bytes shard counts.
- Verify every `fineweb_val_*.bin` token shard has a matching `fineweb_val_bytes_*.bin` sidecar with equal length.
- Verify BOS sidecar byte count remains zero on full val.
- Only then run S5-C3/C4 real CaseOps training experiments.

## Capacity Guard Note

While full prepare was running, three non-experiment capacity guards were briefly started for `11L`, `MLP3`, and `11L+MLP3` on `/tmp/caseops_smoke`. They reached GPTQ Hessian collection but then became CPU-bound with near-zero GPU utilization, competing with full CaseOps prepare. They were terminated and are not counted as experiment results.

Observed parameter counts before termination:

| Guard | Params | Status |
| --- | ---: | --- |
| `s5_guard_11l` | 24,404,568 | aborted during GPTQ/export |
| `s5_guard_mlp3` | 25,448,520 | aborted during GPTQ/export |
| `s5_guard_11l_mlp3` | 30,171,736 | aborted during GPTQ/export |

Decision: do not spend 1h training on these larger static structures until CaseOps baseline and a lighter capacity audit justify them.
