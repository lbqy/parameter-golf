# Root Baseline Ablation Summary

## Scope

E0 is the existing root-baseline run from `baseline_sp1024_1xh100_1h_seed1337_torch26.summary.json`. A1/A2/B1/B2 were run with the root `train_gpt.py` on single H100 GPUs for 3600 seconds each. The root baseline currently exports int8+zlib or packed low-bit variants, but does not implement GPTQ/pergroup, so GPTQ columns are N/A for this batch.

The SP8192 CaseOps paths in PLAN.md were missing in the container. B1/B2 therefore use the available regular SP8192 dataset and tokenizer:

```text
/base/datasets/SP8192/datasets/fineweb10B_sp8192
/base/datasets/SP8192/tokenizers/fineweb_8192_bpe.model
```

## Final Matrix

| ID | Tokenizer | Seq | Steps | Pre-quant BPB | int8+zlib BPB | int8+zlib Total | <=16MB | GPTQ BPB | Variable |
| --- | --- | ---: | ---: | ---: | ---: | ---: | :---: | ---: | --- |
| E0 | SP1024 | 1024 | 10378 | 1.2250 | 1.23203866 | 15,869,303 | yes | N/A | baseline |
| A1 | SP1024 | 2048 | 8966 | 1.2092 | 1.21549379 | 15,866,171 | yes | N/A | seq_len |
| A2 | SP1024 | 4096 | 6893 | 1.2058 | 1.21126584 | 15,849,795 | yes | N/A | seq_len |
| B1 | SP8192 | 1024 | 9574 | 1.1865 | 1.19400986 | 19,376,008 | no | N/A | vocab |
| B2 | SP8192 | 2048 | 8287 | 1.1744 | 1.18147653 | 19,343,563 | no | N/A | combined |

## Observations

- Sequence length gain on SP1024 is real and compliant: E0 -> A1 improves roundtrip BPB by 0.01654487, and E0 -> A2 improves by 0.02077282.
- A2 beats A1 despite fewer steps, so seq4096 is the better SP1024 setting in this batch.
- SP8192 gives a much larger quality gain, but default int8+zlib is over the 16MB limit by more than 3.3MB.
- B2 is the best quality run overall, but it needs stronger compression before it can be a compliant submission.

