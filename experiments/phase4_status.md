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
| S4-G1 | `exp_s4_g1_leakyrelu2_q43_1xh100` | 0 | `MLP_LEAKY_RELU_SLOPE=0.5` | completed | 1.1681 | 1.17263005 | 15,747,565 | 负于 Q43；单独 LeakyReLU^2 不保留 |
| S4-RoPE1 | `exp_s4_rope1_rotary16_q43_1xh100` | 3 | `ROTARY_DIM=16` | completed | 1.1710 | 1.17619727 | 15,663,208 | 负收益；partial RoPE 单项淘汰 |
| S4-K3 | `exp_s4_k3_polarns_q43_1xh100` | 4 | `MUON_NS_MODE=polar` | completed | 1.1696 | 1.17343709 | 15,714,973 | 负于 Q43；速度更快但质量未保住 |
| S4-G1K3 | `exp_s4_g1k3_leaky_polarns_q43_1xh100` | 5 | `MLP_LEAKY_RELU_SLOPE=0.5 MUON_NS_MODE=polar` | completed | 1.1656 | 1.16956051 | 15,763,857 | 首批最佳；pre 强但量化后仍负于 Q43，进入 export-only 细扫候选 |
| S4-G1RoPE | `exp_s4_g1rope16_leaky_rotary16_q43_1xh100` | 6 | `MLP_LEAKY_RELU_SLOPE=0.5 ROTARY_DIM=16` | completed | 1.1677 | 1.17309561 | 15,731,332 | 负于 Q43；RoPE16 组合不继续 |
| S4-K3RoPE | `exp_s4_k3rope16_polarns_rotary16_q43_1xh100` | 7 | `MUON_NS_MODE=polar ROTARY_DIM=16` | completed | 1.1681 | 1.17323532 | 15,703,692 | 负于 Q43；RoPE16 组合不继续 |

启动检查：

- 6 个实验已完成 `warmup_step:20/20` 并进入训练循环。
- 早期 `step_avg` 约 706-730ms，GPU 0/3/4/5/6/7 利用率约 99-100%。
- 每个实验目录下保存 `console.log`、`train.pid` 和脚本日志 `logs/<RUN_ID>.txt`。

运行中检查：

- 约 14-17 分钟时，6 个实验均仍在运行，GPU 0/3/4/5/6/7 利用率约 100%。
- `exp_s4_g1_leakyrelu2_q43_1xh100`：step 1400，step_avg 718.75ms。
- `exp_s4_rope1_rotary16_q43_1xh100`：step 1200，step_avg 717.74ms。
- `exp_s4_k3_polarns_q43_1xh100`：step 1400，step_avg 707.93ms。
- `exp_s4_g1k3_leaky_polarns_q43_1xh100`：step 1400，step_avg 715.44ms。
- `exp_s4_g1rope16_leaky_rotary16_q43_1xh100`：step 1200，step_avg 720.95ms。
- `exp_s4_k3rope16_polarns_rotary16_q43_1xh100`：step 1200，step_avg 720.23ms。

中段检查：

- 约 26 分钟时，6 个实验均到 step 2200，GPU 0/3/4/5/6/7 仍约 100% 利用率。
- `exp_s4_g1_leakyrelu2_q43_1xh100`：step_avg 719.44ms。
- `exp_s4_rope1_rotary16_q43_1xh100`：step_avg 717.06ms。
- `exp_s4_k3_polarns_q43_1xh100`：step_avg 708.18ms。
- `exp_s4_g1k3_leaky_polarns_q43_1xh100`：step_avg 715.43ms。
- `exp_s4_g1rope16_leaky_rotary16_q43_1xh100`：step_avg 721.45ms。
- `exp_s4_k3rope16_polarns_rotary16_q43_1xh100`：step_avg 720.53ms。

收尾前检查：

- 约 52 分钟时，6 个实验均到 step 4400，GPU 0/3/4/5/6/7 仍约 100% 利用率。
- `exp_s4_g1_leakyrelu2_q43_1xh100`：step_avg 719.08ms。
- `exp_s4_rope1_rotary16_q43_1xh100`：step_avg 716.55ms。
- `exp_s4_k3_polarns_q43_1xh100`：step_avg 707.43ms。
- `exp_s4_g1k3_leaky_polarns_q43_1xh100`：step_avg 714.72ms。
- `exp_s4_g1rope16_leaky_rotary16_q43_1xh100`：step_avg 721.12ms。
- `exp_s4_k3rope16_polarns_rotary16_q43_1xh100`：step_avg 720.24ms。

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

## SmearGate 实现状态

- 已在根目录 `train_gpt.py` 增加默认关闭的 `SMEAR_GATE_ENABLED` / `SMEAR_GATE_WIDTH` / `BOS_ID` 开关。
- SmearGate 放在 token embedding + RMSNorm 后、Transformer blocks 前；当前 token 为 BOS 时屏蔽前 token smear，避免跨文档泄露。
- `smear_gate` 和 `smear_lambda` 已纳入控制 tensor / scalar Adam 路径；默认零初始化，关闭时不影响 Q43 复现。
- `smoke_s4_smear_1step` 已通过 train/export/roundtrip 路径。

## SmearGate 完整训练批次

| ID | RUN_ID | GPU | 改动 | 状态 | Pre BPB | Roundtrip BPB | 总字节 | 结论 |
| --- | --- | ---: | --- | --- | ---: | ---: | ---: | --- |
| S4-G2 | `exp_s4_g2_smear_w12_q43_1xh100` | 0 | `SMEAR_GATE_ENABLED=1 SMEAR_GATE_WIDTH=12` | running | | | | |
| S4-G2W24 | `exp_s4_g2_smear_w24_q43_1xh100` | 3 | `SMEAR_GATE_ENABLED=1 SMEAR_GATE_WIDTH=24` | running | | | | |
| S4-G2G1 | `exp_s4_g2g1_smear_w12_leaky_q43_1xh100` | 4 | Smear w12 + `MLP_LEAKY_RELU_SLOPE=0.5` | running | | | | |
| S4-G2K3 | `exp_s4_g2k3_smear_w12_polarns_q43_1xh100` | 5 | Smear w12 + `MUON_NS_MODE=polar` | running | | | | |
| S4-G2G1K3 | `exp_s4_g2g1k3_smear_w12_leaky_polarns_q43_1xh100` | 6 | Smear w12 + LeakyReLU^2 + Polar NS | running | | | | |
| S4-G2G1W24 | `exp_s4_g2g1_smear_w24_leaky_q43_1xh100` | 7 | Smear w24 + `MLP_LEAKY_RELU_SLOPE=0.5` | running | | | | |

启动检查：6 个 SmearGate 实验已完成 `warmup_step:20/20` 并进入训练循环；早期 `step_avg` 约 720-731ms，GPU 0/3/4/5/6/7 约 95-100% 利用率。

中段检查：约 14 分钟时，6 个 SmearGate 实验均到 step 1200，GPU 0/3/4/5/6/7 约 91-100% 利用率。

- `exp_s4_g2_smear_w12_q43_1xh100`：step_avg 717.26ms。
- `exp_s4_g2_smear_w24_q43_1xh100`：step_avg 710.86ms。
- `exp_s4_g2g1_smear_w12_leaky_q43_1xh100`：step_avg 713.56ms。
- `exp_s4_g2k3_smear_w12_polarns_q43_1xh100`：step_avg 713.11ms。
- `exp_s4_g2g1k3_smear_w12_leaky_polarns_q43_1xh100`：step_avg 713.76ms。
- `exp_s4_g2g1_smear_w24_leaky_q43_1xh100`：step_avg 718.43ms。

收尾前检查：约 45 分钟时，6 个 SmearGate 实验均到 step 3800，GPU 0/3/4/5/6/7 约 79-100% 利用率。

- `exp_s4_g2_smear_w12_q43_1xh100`：step_avg 716.41ms。
- `exp_s4_g2_smear_w24_q43_1xh100`：step_avg 709.92ms。
- `exp_s4_g2g1_smear_w12_leaky_q43_1xh100`：step_avg 712.14ms。
- `exp_s4_g2k3_smear_w12_polarns_q43_1xh100`：step_avg 711.89ms。
- `exp_s4_g2g1k3_smear_w12_leaky_polarns_q43_1xh100`：step_avg 712.93ms。
- `exp_s4_g2g1_smear_w24_leaky_q43_1xh100`：step_avg 716.54ms。

## 首批结论

- 首批 6 个 1h 实验均合规完成，所有 artifact 总字节均小于 16,000,000。
- 没有实验超过 Q43 的 `1.16735413`。
- `S4-G1K3` 的 pre-quant BPB `1.1656` 是首批最强训练结果，但 roundtrip BPB `1.16956051` 被量化损失吃掉；下一步优先对该 checkpoint 做 export-only GPTQ/LQER 细扫，而不是直接淘汰 LeakyReLU^2 + Polar NS。
- `ROTARY_DIM=16` 在单项和组合中都偏负，暂不进入下一轮完整训练。
- `MUON_NS_MODE=polar` 单项速度较快，step 数 5091，高于其他组；但 roundtrip 未优于 Q43，后续只作为与 LeakyReLU^2 或新结构组合的候选。

## G1K3 export-only 量化细扫

基座 checkpoint：`results/experiments/exp_s4_g1k3_leaky_polarns_q43_1xh100/final_model.pt`。

| ID | RUN_ID | GPU | 改动 | 状态 | Roundtrip BPB | 总字节 | 结论 |
| --- | --- | ---: | --- | --- | ---: | ---: | --- |
| S4-QG1 | `exp_s4_qg1_g1k3_top4_rank4_calib32_export` | 0 | `LQER_TOP_K=4 LQER_RANK=4 GPTQ_CALIBRATION_BATCHES=32` | completed | 1.16990228 | 15,765,056 | 未优于 QG5 |
| S4-QG2 | `exp_s4_qg2_g1k3_top4_rank8_calib32_export` | 3 | `LQER_TOP_K=4 LQER_RANK=8 GPTQ_CALIBRATION_BATCHES=32` | completed | 1.16990251 | 15,768,927 | rank8 无收益 |
| S4-QG3 | `exp_s4_qg3_g1k3_top5_rank4_calib32_export` | 4 | `LQER_TOP_K=5 LQER_RANK=4 GPTQ_CALIBRATION_BATCHES=32` | completed | 1.16990255 | 15,765,516 | top5 无收益 |
| S4-QG4 | `exp_s4_qg4_g1k3_top4_rank4_calib64_export` | 5 | `LQER_TOP_K=4 LQER_RANK=4 GPTQ_CALIBRATION_BATCHES=64` | completed | 1.17028975 | 15,764,969 | calib64 负收益 |
| S4-QG5 | `exp_s4_qg5_g1k3_top4_rank4_err075_export` | 6 | `LQER_TOP_K=4 LQER_RANK=4 GPTQ_ERROR_SCALE=0.75` | completed | **1.16810703** | 15,763,516 | G1K3 当前最佳；继续扫低 error_scale |
| S4-QG6 | `exp_s4_qg6_g1k3_top4_rank4_err125_export` | 7 | `LQER_TOP_K=4 LQER_RANK=4 GPTQ_ERROR_SCALE=1.25` | completed | 1.17247441 | 15,764,980 | 明显负收益 |

结论：G1K3 的量化最敏感旋钮是 `GPTQ_ERROR_SCALE`。`0.75` 明显优于默认 `1.0`，但仍未超过 Q43；第二轮围绕 `0.50-0.85` 和 `err075 + rank/top/calib` 继续 export-only。

## G1K3 export-only 第二轮

| ID | RUN_ID | GPU | 改动 | 状态 | Roundtrip BPB | 总字节 | 结论 |
| --- | --- | ---: | --- | --- | ---: | ---: | --- |
| S4-QG7 | `exp_s4_qg7_g1k3_top4_rank4_err050_export` | 0 | `GPTQ_ERROR_SCALE=0.50` | completed | 1.16881739 | 15,763,291 | error too low，负于 QG9 |
| S4-QG8 | `exp_s4_qg8_g1k3_top4_rank4_err060_export` | 3 | `GPTQ_ERROR_SCALE=0.60` | completed | 1.16830020 | 15,762,650 | 负于 QG9 |
| S4-QG9 | `exp_s4_qg9_g1k3_top4_rank4_err070_export` | 4 | `GPTQ_ERROR_SCALE=0.70` | completed | **1.16809560** | 15,763,250 | G1K3 export-only 当前最佳，但仍负于 Q43 |
| S4-QG10 | `exp_s4_qg10_g1k3_top4_rank4_err085_export` | 5 | `GPTQ_ERROR_SCALE=0.85` | completed | 1.16836381 | 15,763,651 | 负于 QG9 |
| S4-QG11 | `exp_s4_qg11_g1k3_top4_rank8_err075_export` | 6 | `GPTQ_ERROR_SCALE=0.75 LQER_RANK=8` | completed | 1.16810487 | 15,768,973 | 与 QG9 持平但更大 |
| S4-QG12 | `exp_s4_qg12_g1k3_top4_rank4_err075_calib32_export` | 7 | `GPTQ_ERROR_SCALE=0.75 GPTQ_CALIBRATION_BATCHES=32` | completed | 1.16832362 | 15,763,159 | calib32 无收益 |

结论：G1K3 的最佳 export-only 为 `GPTQ_ERROR_SCALE=0.70`，roundtrip `1.16809560`，仍未超过 Q43 `1.16735413`。该路线保留为“pre-quant 强但量化不够鲁棒”的参考，不再继续做细粒度 error_scale 微扫。

## 记录要求

每个实验完成后记录：

- `step`、`step_avg`、训练 wallclock；
- pre-quant `val_bpb`；
- `final_gptq+brotli_roundtrip_exact val_bpb`；
- `Total submission size gptq+brotli`；
- 是否优于 Q43，是否值得进入组合实验。
