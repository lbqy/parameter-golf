# 第四阶段实验状态

日期：2026-06-07

约束：每个完整训练实验使用 1xH100，`MAX_WALLCLOCK_SECONDS=3600`，最终以 GPTQ/LQER roundtrip BPB 和总字节为准。

## 当前基线

| 项目 | 值 |
| --- | --- |
| 本地最佳 | `exp_s3_q43_r15_clip_top4_rank4_export` |
| Roundtrip BPB | `1.16735413` |
| 总字节 | `15,753,494` |
| 默认训练栈 | Q43/R15 recipe：SP8192 seq4096, QK5, beta2=0.99, grad clip 0.3, tied embed lr 0.04, Muon momentum 0.97, L3-L5 recurrence start 0.30 |
| 默认导出栈 | GPTQ int6/embed8 + brotli + LQER rank4 top4, `FRESH_MODEL_AFTER_QUANT=1`, `RECURRENCE_ACTIVE=1` |

## 首批并行实验

目的：先验证不依赖 CaseOps 数据的三个低风险 S4 开关及二阶组合，给后续 CaseOps/TTT/结构栈提供新 baseline。

| ID | RUN_ID | GPU | 改动 | 状态 | Pre BPB | Roundtrip BPB | 总字节 | 结论 |
| --- | --- | ---: | --- | --- | ---: | ---: | ---: | --- |
| S4-G1 | `exp_s4_g1_leakyrelu2_q43_1xh100` | 0 | `MLP_LEAKY_RELU_SLOPE=0.5` | running | | | | |
| S4-RoPE1 | `exp_s4_rope1_rotary16_q43_1xh100` | 3 | `ROTARY_DIM=16` | running | | | | |
| S4-K3 | `exp_s4_k3_polarns_q43_1xh100` | 4 | `MUON_NS_MODE=polar` | running | | | | |
| S4-G1K3 | `exp_s4_g1k3_leaky_polarns_q43_1xh100` | 5 | `MLP_LEAKY_RELU_SLOPE=0.5 MUON_NS_MODE=polar` | running | | | | |
| S4-G1RoPE | `exp_s4_g1rope16_leaky_rotary16_q43_1xh100` | 6 | `MLP_LEAKY_RELU_SLOPE=0.5 ROTARY_DIM=16` | running | | | | |
| S4-K3RoPE | `exp_s4_k3rope16_polarns_rotary16_q43_1xh100` | 7 | `MUON_NS_MODE=polar ROTARY_DIM=16` | running | | | | |

启动检查：

- 6 个实验已完成 `warmup_step:20/20` 并进入训练循环。
- 早期 `step_avg` 约 706-730ms，GPU 0/3/4/5/6/7 利用率约 99-100%。
- 每个实验目录下保存 `console.log`、`train.pid` 和脚本日志 `logs/<RUN_ID>.txt`。

### 公共命令环境

```bash
DATA_PATH=/base/datasets/SP8192/datasets/fineweb10B_sp8192
TOKENIZER_PATH=/base/datasets/SP8192/tokenizers/fineweb_8192_bpe.model
VOCAB_SIZE=8192
TRAIN_SEQ_LEN=4096
MAX_WALLCLOCK_SECONDS=3600
WARMUP_STEPS=20
ITERATIONS=20000
VAL_LOSS_EVERY=0
TRAIN_LOG_EVERY=200
TRAIN_BATCH_TOKENS=524288
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

## CaseOps 状态

当前容器只发现 `/base/datasets/SP8192`，未发现现成 CaseOps dataset、`fineweb_val_bytes_*.bin` 或 `docs_selected.jsonl`。S4-C0 需要先恢复数据链路；在 byte sidecar 可用前，不启动 S4-C1/C2 完整训练。

## 记录要求

每个实验完成后记录：

- `step`、`step_avg`、训练 wallclock；
- pre-quant `val_bpb`；
- `final_gptq+brotli_roundtrip_exact val_bpb`；
- `Total submission size gptq+brotli`；
- 是否优于 Q43，是否值得进入组合实验。
