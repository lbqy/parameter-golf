# Parameter Golf 实验计划

**约束**：1×H100 / 1h 训练，提交物 ≤ 16 MB，主指标为 roundtrip `val_bpb`。

**脚本**：根目录 `train_gpt.py` + `gptq_export.py`（GPTQ 导出）。

---

## 第一阶段：Baseline 消融（已完成）

### 目标

固定 9L×512 架构，每次只改一个变量：序列长度、词表大小、或两者组合。

### 结果汇总

| ID | Tokenizer | Seq | Pre-quant BPB | int8+zlib BPB | 总大小 | ≤16MB |
| --- | --- | ---: | ---: | ---: | ---: | :---: |
| E0 | SP1024 | 1024 | 1.2250 | 1.2320 | 15.87 MB | ✓ |
| A1 | SP1024 | 2048 | 1.2092 | 1.2155 | 15.87 MB | ✓ |
| A2 | SP1024 | 4096 | 1.2058 | 1.2113 | 15.85 MB | ✓ |
| B1 | SP8192 | 1024 | 1.1865 | 1.1940 | 19.38 MB | ✗ |
| B2 | SP8192 | 2048 | 1.1744 | 1.1815 | 19.34 MB | ✗ |

详细报告：`experiments/rootbaseline_ablation_summary.md`

### 结论

- SP1024 拉长序列有效且合规；A2 为最佳 SP1024 配置。
- **B2 质量最优**，但 int8+zlib 超 cap **~3.3 MB**。
- 第二阶段不再改架构/训练时长，专注 **重做 GPTQ 压缩**。现有低比特 RTN/GPTQ 代码仅作为反例保留，不再作为主路线。

### B2 训练 backbone（第二阶段固定）

```text
SP8192, TRAIN_SEQ_LEN=2048, 1×H100, MAX_WALLCLOCK_SECONDS=3600, SEED=1337
DATA_PATH=/base/datasets/SP8192/datasets/fineweb10B_sp8192
TOKENIZER_PATH=/base/datasets/SP8192/tokenizers/fineweb_8192_bpe.model
VOCAB_SIZE=8192
VAL_LOSS_EVERY=0, TRAIN_LOG_EVERY=200, TRAIN_BATCH_TOKENS=524288
```

---

## 第二阶段：重做 GPTQ 量化压缩（已完成）

### 背景

| 参考 | Roundtrip BPB | 大小 | 说明 |
| --- | ---: | ---: | --- |
| B2 int8+zlib | 1.1815 | 19.34 MB | 质量目标，超限 |
| RTN int6/int7 + clip-search | 1.2191 | 15.81 MB | 合规但损失大，**不再作为主路线** |
| records GPTQ + LQER | ~1.06 | ~15.9 MB | SOTA 参考 |

根脚本导出模式：

| `QUANT_MODE` | 产物 | 说明 |
| --- | --- | --- |
| `rtn`（默认） | `final_model.int8.ptz` | int8/intN RTN + zlib |
| `gptq` | `final_model.gptq.ptz` | Hessian GPTQ + bit-packed intN + SDClip + brotli/zlib |

> RTN 仅保留 `ref_config.md` 中已有对照结果，不再安排新 RTN 实验。

### records 调研结论

先进 16MB 提交的量化共同点：

| 来源 | 可复用点 | 当前根脚本落地 |
| --- | --- | --- |
| `2026-04-23_SP8192_CaseOps...PolarNS...` | Hessian GPTQ，matrix int6、embed int7/int8，MLP/attn/embed 分别调 clip sigma，`GPTQ_CALIBRATION_BATCHES=16` 即可 | 作为核心 PTQ 路线 |
| `2026-04-27_SP8192_LQER...` | LQER asymmetric int4 rank-4 修正 top-3 quant-error tensors | 作为 Q3/Q4 消融 |
| 同上 | per-group / similarity-sort / lrzip 可再省约 280KB，但实现复杂且依赖系统 `lrzip` | 暂列 Q5，只有前 4 组还差容量时启用 |
| 早期 mixed quant 记录 | embedding/head 对量化最敏感；大 tokenizer 下 embed bits 需要单独 sweep | Q1-Q4 专门 sweep `EMBED_BITS` |

现有根目录量化问题：

- “int6/int7” 实际存成 `torch.int8`，容量依赖 brotli 压缩高位，不能稳定逼近 16MB。
- GPTQ artifact 缺少 size guard，容易过小或超限，不能系统利用 16MB 预算。
- LQER 可用但没有先完成正确 bit packing，导致容量/收益判断失真。

### 成功标准

| 优先级 | 标准 |
| --- | --- |
| P0 | 总提交大小 ≤ 16 MB |
| P1 | roundtrip BPB < 1.20 |
| P2 | roundtrip BPB ≤ 1.185（接近 B2 int8） |

### 渐进消融矩阵

早期用 B2 历史 `final_model.pt` 做 `EXPORT_ONLY=1` 快速量化消融，但该 checkpoint 与当前代码存在 eval 不一致；后续以 current-code checkpoint 为准，优先在 R9-full 与 B3-full 上做 fresh compiled roundtrip 验证。

| ID | 目的 | 方法 | Matrix | Embed | LQER | Calib | 预期 |
| --- | --- | --- | ---: | ---: | --- | ---: | --- |
| **Q1** | 正确 packing 基线 | GPTQ + bit-packed SDClip | 6 | 8 | off | 16 | 验证容量和 roundtrip，预计接近上限 |
| **Q2** | embed 容量换质量 | GPTQ + bit-packed SDClip | 6 | 7 | off | 16 | 更小但 embed 量化损失更大 |
| **Q3** | 误差校正收益 | Q2 + LQER asym int4 | 6 | 7 | rank4 top3 | 16 | 用少量字节追回 BPB |
| **Q4** | 高质量/容量边界 | GPTQ + bit-packed SDClip | 7 | 8 或 7 | off | 16 | 若 ≤16MB，优先追近 B2 int8 |
| **Q5** | 压缩兜底 | Q3/Q4 + per-group/simsort 或 brotli 参数调优 | 6/7 | 7/8 | best | 16 | 只在 Q4 略超或 Q3 过小但质量不足时做 |

并行策略：Q1-Q3 可各占 1 张 H100 同时 export；Q4 根据 Q1-Q3 的 size 结果决定 bits；完整训练只跑最有希望的 1-2 组。

### GPTQ 环境变量

| 变量 | Q1 | Q2 | 说明 |
| --- | --- | --- | --- |
| `QUANT_MODE` | gptq | gptq | 启用 GPTQ 导出 |
| `COMPRESSOR` | brotli | brotli | 无 brotli 时改 `zlib` |
| `MATRIX_BITS` | 6 | 6 | 矩阵 bitwidth，Q4 sweep 7 |
| `EMBED_BITS` | 8 | 7 | embedding bitwidth |
| `MATRIX_CLIP_SIGMAS` | 12.85 | 12.85 | 默认矩阵 clip |
| `ATTN_CLIP_SIGMAS` | 13.0 | 13.0 | attention clip |
| `MLP_CLIP_SIGMAS` | 11.5 | 11.5 | MLP clip |
| `EMBED_CLIP_SIGMAS` | 14.0 | 14.0 | embedding clip |
| `GPTQ_CALIBRATION_BATCHES` | 16 | 16 | Hessian 校准 batch 数 |

**Q3** — `RUN_ID=exp_q3_b2_export_gptq_pack_i6e7_lqer`，`EMBED_BITS=7`，并追加：

```bash
LQER_ENABLED=1 LQER_RANK=4 LQER_TOP_K=3 \
LQER_FACTOR_BITS=4 LQER_ASYM_ENABLED=1 LQER_ASYM_GROUP=64
```

### 命令模板

```bash
cd /base/project/parameter-golf/results/experiments/<RUN_ID>
CUDA_VISIBLE_DEVICES=<GPU> \
RUN_ID=<RUN_ID> \
SEED=1337 \
QUANT_MODE=gptq \
COMPRESSOR=brotli \
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
MATRIX_BITS=6 EMBED_BITS=8 \
MATRIX_CLIP_SIGMAS=12.85 ATTN_CLIP_SIGMAS=13.0 MLP_CLIP_SIGMAS=11.5 \
EMBED_CLIP_SIGMAS=14.0 \
GPTQ_CALIBRATION_BATCHES=16 \
/opt/conda/bin/torchrun --standalone --nproc_per_node=1 /base/project/parameter-golf/train_gpt.py
```

**Q1** — `RUN_ID=exp_q1_b2_export_gptq_pack_i6e8`，`MATRIX_BITS=6 EMBED_BITS=8`

**Q2** — `RUN_ID=exp_q2_b2_export_gptq_pack_i6e7`，`MATRIX_BITS=6 EMBED_BITS=7`

**Q4** — 根据 Q1-Q3 结果选择：

```bash
MATRIX_BITS=7 EMBED_BITS=8
# 若超 16MB，则降为 MATRIX_BITS=7 EMBED_BITS=7 或 MATRIX_BITS=6 EMBED_BITS=8 + LQER_RANK/TOP_K sweep
```

### 结果记录

每个实验写入 `experiments/exp_<id>.md`，包含：完整命令、pre-quant BPB、`final_gptq+brotli_roundtrip_exact val_bpb`、总字节、是否 ≤16MB。

### 结果矩阵（待填）

| ID | Embed | 方法 | Roundtrip BPB | 总字节 | ≤16MB |
| --- | ---: | --- | ---: | ---: | :---: |
| B2 ref | 8 | int8+zlib | 1.1815 | 19,343,563 | ✗ |
| Q1 | 8 | packed GPTQ int6/8 | | | |
| Q2 | 7 | packed GPTQ int6/7 | | | |
| Q3 | 7 | packed GPTQ int6/7 + LQER | | | |
| Q4 | 7/8 | packed GPTQ int7/x | | | |

### 当前 export-only 结果

B2 checkpoint：`results/experiments/exp_b2_sp8192_seq2048_1xh100_rootbaseline/final_model.pt`

| ID | 方法 | 关键配置 | Roundtrip BPB | 总字节 | ≤16MB | 结论 |
| --- | --- | --- | ---: | ---: | :---: | --- |
| Q1 | packed GPTQ | matrix6/embed8, sigma scale | 1.2888 | 11,197,834 | ✓ | 旧诊断；B2 checkpoint 与当前代码存在 eval 不一致 |
| Q2 | packed GPTQ | matrix6/embed7, sigma scale | 1.5161 | 11,313,032 | ✓ | embed7 损失过大 |
| Q3 | packed GPTQ + LQER | matrix6/embed7, rank4 top3 | 1.2671 | 11,322,900 | ✓ | LQER 有效但 GPTQ 基底太差 |
| Q4 | packed GPTQ | matrix7/embed8, sigma scale | 1.6046 | 14,605,212 | ✓ | sigma scale 明显不适配根模型 |
| Q6 | packed GPTQ | matrix8/embed8, sigma scale | 1.6023 | 14,174,063 | ✓ | int8 仍坏，定位为 GPTQ 算法/scale 问题 |
| Q7 | packed GPTQ | matrix8/embed8, max scale | 1.5499 | 19,207,764 | ✗ | 旧诊断；后续改用 current checkpoint + fresh eval |
| R1 | packed RTN | matrix6/embed8, zlib | 1.1892 | 16,169,244 | ✗ | 只超约 169KB |
| R4 | packed RTN | matrix6/embed8, brotli | 1.1892 | 15,839,698 | ✓ | 当前可靠基线 |
| R7 | packed RTN override | R4 + `blocks.8.mlp.fc/proj` 7bit | 1.1888 | 15,967,918 | ✓ | 合规，小幅优于 R4 |
| R8 | packed RTN override | R4 + `blocks.2-8.attn.c_v` 7bit | **1.1876** | **15,992,174** | ✓ | 当前最佳，接近 16MB |
| R9 | packed RTN override | R4 + `blocks.3-8.attn.c_v` 7bit | 1.1878 | 15,968,899 | ✓ | 安全余量备选 |

完整训练确认：

| ID | RUN_ID | 状态 |
| --- | --- | --- |
| R8-full | `exp_r8_full_sp8192_seq2048_rtn_i6e8_brotli_cv7_l2_8` | completed: pre 1.1748, roundtrip 1.18727933, total 15,991,001 |
| R9-full | `exp_r9_full_sp8192_seq2048_rtn_i6e8_brotli_cv7_l3_8` | completed: pre 1.1743, roundtrip 1.18714581, total 15,972,698 |

### GPTQ 重新对齐结果（current checkpoint）

后续 GPTQ 以 R9-full 的 `final_model.pt` 为 current-code checkpoint，并启用 `FRESH_MODEL_AFTER_QUANT=1`，即量化后新建模型、加载 roundtrip 权重、重新 compile 后验证。原因：同一模型在 eager 与 compiled eval 下 BPB 差异很大，且 GPTQ Hessian collection 后复用同一 model 对象会污染 roundtrip eval；fresh compiled eval 更接近真实 artifact 加载。

| ID | 方法 | 关键配置 | Roundtrip BPB | 总字节 | ≤16MB | 结论 |
| --- | --- | --- | ---: | ---: | :---: | --- |
| R9 ref | packed RTN | matrix6/embed8 + `blocks.3-8.attn.c_v` 7bit | 1.18714581 | 15,972,698 | ✓ | 当前 RTN baseline |
| G-R9-0 | packed GPTQ | matrix6/embed8, quantile scale, `error_scale=0` | 1.18778834 | 15,795,917 | ✓ | 等价 RTN-ish 基座 |
| G-R9-025 | packed GPTQ | matrix6/embed8, quantile scale, `error_scale=0.25` | 1.18444411 | 15,799,338 | ✓ | Hessian correction 有效 |
| G-R9-05 | packed GPTQ | matrix6/embed8, quantile scale, `error_scale=0.5` | 1.18212195 | 15,796,816 | ✓ | 继续提升 |
| G-R9-075 | packed GPTQ | matrix6/embed8, quantile scale, `error_scale=0.75` | 1.18080509 | 15,799,756 | ✓ | 接近 int8 上限 |
| G-R9-1 | packed GPTQ | matrix6/embed8, quantile scale, `error_scale=1.0` | 1.18032229 | 15,798,310 | ✓ | 当前无 LQER 最佳 |
| G-R9-025-LQER | packed GPTQ + LQER | `error_scale=0.25`, LQER rank4 top3 | **1.17978818** | **15,803,202** | ✓ | R9 当前最佳，优于 B2 int8 参考 |

GPTQ 当前推荐命令核心：

```bash
QUANT_MODE=gptq COMPRESSOR=brotli FRESH_MODEL_AFTER_QUANT=1 \
GPTQ_SCALE_MODE=quantile GPTQ_SCALE_FLOOR=1 GPTQ_ERROR_SCALE=1 \
MATRIX_BITS=6 EMBED_BITS=8 GPTQ_CALIBRATION_BATCHES=16
```

B3 状态：

| ID | RUN_ID | 状态 |
| --- | --- | --- |
| B3 | `exp_b3_sp8192_seq4096_1xh100_rootbaseline` | completed: step 6485, pre 1.1745, int8+zlib roundtrip 1.18068959, total 19,287,254（超 16MB） |

B3 GPTQ 对齐结果：

| ID | 方法 | 关键配置 | Roundtrip BPB | 总字节 | ≤16MB | 结论 |
| --- | --- | --- | ---: | ---: | :---: | --- |
| G-B3-1 | packed GPTQ | matrix6/embed8, quantile scale, `error_scale=1.0` | 1.18001764 | 15,731,454 | ✓ | B3 GPTQ 基线，接近 R9+LQER |
| G-B3-025-LQER | packed GPTQ + LQER | `error_scale=0.25`, LQER rank4 top3 | 1.18007382 | 15,740,005 | ✓ | B3 上低 error_scale + LQER 未优于纯 GPTQ |
| G-B3-1-LQER | packed GPTQ + LQER | `error_scale=1.0`, LQER rank4 top3 | **1.17661956** | **15,738,799** | ✓ | 当前全局最佳；B3 可替换 R9 作为后续量化 baseline |

第二阶段里程碑归档：`experiments/phase2_quantization_milestone.md`

---

## 第三阶段：深入细节优化消融（规划启动）

### 当前基线

第三阶段默认以 B3 作为训练基座，并沿用第二阶段已验证的 GPTQ+LQER roundtrip：

```text
SP8192, TRAIN_SEQ_LEN=4096, 1xH100, MAX_WALLCLOCK_SECONDS=3600
QUANT_MODE=gptq, COMPRESSOR=brotli, FRESH_MODEL_AFTER_QUANT=1
GPTQ_SCALE_MODE=quantile, GPTQ_SCALE_FLOOR=1, GPTQ_ERROR_SCALE=1
MATRIX_BITS=6, EMBED_BITS=8, GPTQ_CALIBRATION_BATCHES=16
LQER_ENABLED=1, LQER_RANK=4, LQER_TOP_K=3, LQER_ASYM_ENABLED=1
baseline roundtrip BPB = 1.17661956, total bytes = 15,738,799
```

第三阶段成功标准：

| 优先级 | 标准 |
| --- | --- |
| P0 | 所有候选最终 roundtrip 总字节 <= 16 MB |
| P1 | 优于 B3-GPTQ-LQER：roundtrip BPB < 1.17662 |
| P2 | 首个阶段目标：roundtrip BPB <= 1.170 |
| P3 | 中期目标：roundtrip BPB <= 1.160 |

### records 再调研：可迁移优化池

| 方向 | records 证据 | 预期收益 | 成本/风险 | 第三阶段处理 |
| --- | --- | ---: | --- | --- |
| GPTQ/LQER 细扫 | `2026-04-27_SP8192_LQER...`：LQER top3 rank4 + clip stack；`2026-04-29` 确认 GPTQ reserve 合规 | 小到中 | 低，已实现主框架 | 先跑 export-only，作为 S3-Q 组 |
| 训练超参 stack | `2026-04-27` 9 hparam stack：`BETA2=0.99`、`WARMDOWN_FRAC=0.85`、`MIN_LR=0.10`、`GRAD_CLIP_NORM=0.3`、clip retune | 小到中 | 低到中，当前缺 `MIN_LR/WARMDOWN_FRAC` | S3-H 组，先加 env 再跑 |
| Coprime-stride loader | `2026-03-29_Loader_FullGPTQ...`：多 shard coprime stride 提升 batch 多样性，零 step overhead | 小到中 | 中，需替换 loader 但不改模型 | S3-L 组 |
| 深度递归 | `2026-04-09_SP8192_3LayerRecur...`、`2026-04-06_ProgressiveRecurrence`、`2026-04-27`：layers 3-5 loop 2x/3x，常在 frac 0.35 后启用 | 中到大 | 中，增加 step time，1xH100 需平衡步数 | S3-R 组，优先轻量递归 |
| Parallel residual / decoder lane | `2026-04-14`、`2026-04-27`：parallel residuals 与递归叠加稳定 | 中 | 中高，当前仅 U-Net skip，未实现 lane mix | S3-P 组，放在递归后 |
| TTT / Phased TTT | `2026-04-06_QK5_LegalTTT` 约 -0.0028 BPB；`2026-04-14`、`2026-04-23`、`2026-04-27` 常见 -0.012 BPB | 大 | 高，需严格 score-before-update，eval 时间增加 | S3-T 组，优先 eval-only LoRA TTT |
| SmearGate + BOS fix | `2026-04-16`、`2026-04-27`：位置混合门控，BOS mask 修复跨文档泄露 | 中 | 中，需正确识别 BOS；packed stream 有边界风险 | S3-G 组，在 loader/doc 边界清楚后做 |
| Sparse/attention output gate | `2026-04-16`、`2026-04-23`、`2026-04-27`：小参数 head-output gate，量化 gate int8 | 小到中 | 中，增加控制参数和量化路径 | S3-G 组 |
| CaseOps tokenizer/data | `2026-04-18` 以后 SOTA 基础；lossless caps caseops SP8192 | 大 | 高，需要 dataset/tokenizer 准备与 BPB byte sidecar 合规核查 | S3-C 组，单独分支 |
| Per-group compression | `2026-04-27`：lrzip+similarity-sort 约省 280KB | 容量收益 | 中高，依赖 `lrzip` | 暂不优先，除非新结构超 16MB |

### 第三阶段消融矩阵

先做低成本、高信息量实验；每组最终都以 `final_gptq+brotli_roundtrip_exact` 为准。

| ID | 目的 | 改动 | 代码需求 | 运行方式 | 通过标准 |
| --- | --- | --- | --- | --- | --- |
| S3-Q1 | B3 量化余量细扫 | `LQER_TOP_K=4/5`、`LQER_RANK=2/4/8`、`GPTQ_ERROR_SCALE=0.75/1.0/1.25` | 无或很小 | B3 checkpoint export-only 并行 | 找到 <1.17662 且 <=16MB |
| S3-Q2 | embedding bit/clip 边界 | `EMBED_BITS=7/8`，`EMBED_CLIP_SIGMAS=12/14/15`，必要时 matrix override | 无 | B3 checkpoint export-only | 若 embed7 质量不掉，可释放容量给结构参数 |
| S3-H1 | records 超参栈最小移植 | `BETA2=0.99`、`GRAD_CLIP_NORM=0.3`、`MLP_CLIP_SIGMAS=12`、`EMBED_CLIP_SIGMAS=15` | 无 | 完整 1h train + GPTQ | roundtrip 优于 B3 |
| S3-H2 | warmdown/min-lr | 增加 `WARMDOWN_FRAC`、`MIN_LR`，测试 `0.85/0.10` | 小 | 完整 1h train + GPTQ | pre 和 roundtrip 同时改善 |
| S3-L1 | coprime loader | 多 shard coprime stride batch 采样 | 中 | 完整 1h train + GPTQ | step time 基本不变，roundtrip 改善 |
| S3-R1 | 轻量深度递归 | layers 3-5 在训练 frac>=0.35 后额外跑 1 次，权重共享 | 中 | 完整 1h train + GPTQ | 改善抵消 step 变慢 |
| S3-R2 | 递归强度 sweep | loop 2x/3x，start frac 0.25/0.35/0.50 | 中 | 只保留 R1 正信号后跑 | 找最佳质量/步数折中 |
| S3-T1 | eval-only legal TTT MVP | LoRA on q/k/v/o 或 mlp，score-first chunk update，单 phase | 高 | 使用 B3-GPTQ artifact eval-only | TTT 后 BPB 至少 -0.002 |
| S3-T2 | phased TTT | 3 phase prefix docs，LoRA rank/lr/WD sweep | 高 | T1 正信号后跑 | TTT gain 接近 records 级别 |
| S3-G1 | SmearGate BOS-safe | 加位置混合 gate，BOS mask 防跨文档泄露 | 中 | 完整 1h train + GPTQ | pre/roundtrip 改善，无边界泄露 |
| S3-G2 | attention output/sparse gate | head-output gate window=12，gate scale sweep | 中 | G1 后或并行分支 | 小参数收益为正 |
| S3-C1 | CaseOps feasibility | 检查/准备 CaseOps SP8192 tokenizer+dataset，复核 BPB byte 统计 | 高 | smoke + 短训 | 数据链路合规再进入完整实验 |

### 首批执行顺序

1. **S3-Q1/Q2**：不改训练，直接用 B3 checkpoint export-only 并行扫；目标是把当前 1.17662 再压一点，同时摸清容量余量。
2. **S3-H1/H2**：小代码/无代码训练超参组，最适合先占卡完整跑，给后续结构改动提供新 baseline。
3. **S3-L1**：loader 改动相对独立；若正收益，所有后续训练组默认继承。
4. **S3-R1**：深度递归 MVP，只做共享权重、轻量 loop；若 step time 损失太大则停止 R2。
5. **S3-T1**：单独开发 eval-only legal TTT，不和训练结构同时混，先证明 score-first LoRA TTT 在当前 B3 artifact 上有收益。

### 第三阶段实验记录格式

每个实验新增 `experiments/exp_s3_<id>.md`，至少记录：

- 基座 checkpoint / 是否完整训练；
- 完整 env 命令；
- step 数、step_avg、pre-quant BPB；
- GPTQ/LQER 配置、artifact bytes、total bytes；
- `final_gptq+brotli_roundtrip_exact val_bpb`；
- 若启用 TTT，另记 pre-TTT、post-TTT、eval_time、score-before-update 合规说明。
