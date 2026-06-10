# Parameter Golf 实验计划

**约束**：1×H100 / 1h 训练，提交物 ≤ 16 MB，主指标为 roundtrip `val_bpb`。

**脚本**：根目录 `train_gpt.py` + `gptq_export.py`（GPTQ 导出）。

## 当前状态快照（2026-06-10）

当前有效最佳已经从根目录 QS28/Phase 4 线转移到 records 04-27 stack 的 CaseOps + `lrzip` + legal phased TTT 主线。最新候选为 R8-pg291 上的 partial RoPE 低频置零变体：

| 项目 | 值 |
| --- | --- |
| 当前最佳训练 run | `exp_rope_zlf2_r8_pg291_fa3_smear_seed42_nottt` |
| TTT eval run | `exp_rope_zlf2_r8_pg291_fa3_smear_seed42_ttt_eval` |
| artifact | `results/experiments/exp_rope_zlf2_r8_pg291_fa3_smear_seed42_nottt/final_model.int6.ptz` |
| no-TTT / roundtrip BPB | **1.08972570** |
| post-TTT BPB | **1.07586081** |
| total bytes | **15,917,703** |
| 关键开关 | `ROPE_DIMS=16 ROPE_ZERO_LOW_FREQS=2 REQUIRE_FA3=1 REQUIRE_TENSOR_DESCRIPTOR=1 CASEOPS_ENABLED=1 SMEAR_GATE_ENABLED=1 COMPRESSOR=pergroup` |

当前判断：

- 旧 R8 fallback 最佳为 `1.09647097 -> 1.08179851`；它证明 records 主线有效，但不是最终环境。
- 修复 `pg291` 环境后，R8 严格复跑使用 FA3 `flash_attn_interface`、`triton.tools.tensor_descriptor`、DocumentPackingLoader 和 real `lrzip`，刷新到 `1.09043610 -> 1.07615418`。
- partial RoPE 小扫中，`ROPE_ZERO_LOW_FREQS=2` 进一步刷新到 `1.08972570 -> 1.07586081`；`zlf1` 只小幅改善 no-TTT，`ROPE_DIMS=32 zlf1` 退化。
- 下一步不再围绕旧 fallback artifact 做 TTT/export 微扫；应在 R8-pg291/zlf2 基础上做很小的训练动态矩阵，优先 seed、warmdown、batch/loop 时机和 RoPE 低频置零邻域。

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

---

## 第四阶段：面向 SOTA 差距的深度优化实验（规划）

### 阶段三复盘与不要误判的负结果

第三阶段已经把根目录脚本推进到一个清晰的局部最优：

| 项目 | 值 |
| --- | --- |
| 当前最佳 | `exp_s3_q43_r15_clip_top4_rank4_export` |
| Roundtrip BPB | **1.16735413** |
| 总字节 | **15,753,494** |
| 训练 checkpoint | `exp_s3_r15_b3_qkgain5_hparam_tied004_muon097_recur_l3_5_start030_1xh100/final_model.pt` |

对照 records 中 `2026-04-27_SP8192_LQER_SparseGate_BOSSmearFix_9HpStack_1.0611` 的 3-seed mean **1.06108**，当前差距已经不是 Q43 周围继续微扫 `LQER_TOP_K`、`GPTQ_ERROR_SCALE` 或 `RECURRENCE_START_FRAC` 能解决的。第三阶段负结果应这样理解：

- `coprime loader` 和 `warmdown/min-lr` 在当前根脚本中负收益，不代表 records 对应思想在 CaseOps/doc-boundary/TTT 栈里无效。
- `lm-head LoRA TTT MVP` 负收益，只证明当前简化实现不可用，不能淘汰 records 中普遍有效的 phased/legal TTT。
- 递归已经有正收益，但 L3-L5、start 0.30 附近出现平台期；继续单独扫 start frac 的收益很低。
- R15/Q43 的量化细扫基本收敛；下一阶段要把重点转向数据链路、TTT、结构组件和压缩余量。

### 为什么 TTT 没做好

当前 `ttt_eval.py` 与 records 中有效 TTT 的差距很大：

| 维度 | 当前 MVP | records 中有效方案 |
| --- | --- | --- |
| Adapter 位置 | 只在 lm head 上加 LoRA | Q/K/V/O、MLP、lm head 多处 adapter |
| 更新单位 | 固定连续 token block | document boundary / prefix docs |
| 调度 | 单阶段边 score 边更新 | multi-phase global SGD，score-first 后再更新 |
| 状态策略 | 一个全局 adapter 连续更新 | per-doc reset、warm-start A、phase 累积更新 |
| 合规性验证 | token block score-before-update | doc-level score-before-update、single pass、no rescoring |
| forward 镜像 | 只复用 hidden + head | TTT forward 必须镜像门控、递归、parallel residual、SmearGate |

因此 S3-T1 的失败结论应写成：**lm-head-only TTT 替代品不可用**。第四阶段仍应把真正的 phased/legal TTT 作为高优先级主线，目标不是“再调当前脚本几个 TTT LR”，而是重做 doc-boundary TTT 路径。

### records 共性拆解

| 差距方向 | records 共性 | 当前根脚本状态 | 第四阶段判断 |
| --- | --- | --- | --- |
| 数据/tokenizer | CaseOps SP8192 + validation byte sidecar + BOS boundary | 普通 SP8192，BPB 来自 tokenizer LUT | 最高优先级，先打通数据和合规 BPB |
| TTT | phased TTT、multi-phase global SGD、doc-level score-first | lm-head LoRA MVP | 最高优先级，需重做而非微调 |
| Attention kernel | FA3 / varlen attention 支撑 doc-boundary 和 TTT | torch SDPA flash backend 已启用，固定 seq | 固定 seq 足够；做 doc/varlen/TTT 时再引入 FA3 |
| RoPE | partial RoPE、YaRN/LN scale 常见；可试 FoPE 变体 | full RoPE | 中优先级，先小矩阵短训 |
| MLP activation | LeakyReLU(0.5)^2 + fused MLP 常见 | ReLU^2，MLP_MULT=2 | 高优先级低风险，先只换 activation |
| 结构 | 11L、MLP4x、XSA、parallel decoder lane、depth recurrence | 9L、MLP2x、U-Net skip、轻量 recurrence | 分阶段移植，避免一次性增参/降速 |
| Gate | Sparse attention gate、SmearGate BOS-fix、QuantGate | 未实现 | 高优先级，单独验证 |
| 优化器/内核 | Polar-Express NS、fused softcapped CE、fused MLP | 标准 Muon NS、eager CE/MLP | 作为吞吐/质量支撑线 |
| 压缩 | per-group、similarity-sort、lrzip | brotli + packed GPTQ/LQER | 用于释放容量，而不是单独追 BPB |

### 第四阶段成功标准

| 优先级 | 标准 |
| --- | --- |
| P0 | 所有最终候选 artifact 总字节 <= 16,000,000 |
| P1 | 任何完整 1h 训练候选必须报告 pre-quant、roundtrip、step 数、step_avg |
| P2 | 单项结构/训练改动 roundtrip BPB 优于 Q43 或带来可证明的容量/吞吐余量 |
| P3 | 第四阶段短期目标：roundtrip BPB <= 1.160 |
| P4 | 若启用 TTT，post-TTT 至少比同 artifact pre-TTT 改善 0.002 BPB，且满足 score-before-update、single pass、doc boundary、no rescoring |

### 第四阶段实验矩阵

每个实验仍以 `final_gptq+brotli_roundtrip_exact` 或明确的 post-TTT BPB 为准。除非特别说明，训练基线使用 Q43 winning recipe：

```text
SP8192, TRAIN_SEQ_LEN=4096, 1xH100, MAX_WALLCLOCK_SECONDS=3600
QK_GAIN_INIT=5.0, BETA2=0.99, GRAD_CLIP_NORM=0.3
MLP_CLIP_SIGMAS=12, EMBED_CLIP_SIGMAS=15
TIED_EMBED_LR=0.04, MUON_MOMENTUM=0.97
RECURRENCE_EXTRA_PASSES=1, RECURRENCE_START_LAYER=3, RECURRENCE_END_LAYER=5
RECURRENCE_START_FRAC=0.30
QUANT_MODE=gptq, COMPRESSOR=brotli, FRESH_MODEL_AFTER_QUANT=1
GPTQ_SCALE_MODE=quantile, GPTQ_SCALE_FLOOR=1, GPTQ_ERROR_SCALE=1
MATRIX_BITS=6, EMBED_BITS=8, GPTQ_CALIBRATION_BATCHES=16
LQER_ENABLED=1, LQER_RANK=4, LQER_TOP_K=4, LQER_ASYM_ENABLED=1
RECURRENCE_ACTIVE=1
```

| ID | 目的 | 改动 | 代码需求 | 运行方式 | 通过标准 |
| --- | --- | --- | --- | --- | --- |
| S4-C0 | CaseOps 数据链路可行性 | 下载/恢复 CaseOps tokenizer、train shards、`fineweb_val_bytes_*.bin` | 小到中 | 数据 smoke + eval smoke | 日志出现 `val_bpb:byte_sidecar:enabled`，BOS sidecar byte=0，token/byte 对齐 |
| S4-C1 | CaseOps 短训 | plain root/Q43 结构切到 CaseOps SP8192 | 中 | 10-15min smoke train + GPTQ | pre BPB 不异常，roundtrip 合规，确认可进入 1h |
| S4-C2 | CaseOps 完整基线 | Q43 recipe + CaseOps | 中 | 1h train + GPTQ/LQER | roundtrip 优于 Q43，若 TTT 未启用也记录 pre/post quant |
| S4-T0 | TTT 合规数据切分 | 从 val tokens 找 BOS doc boundary，构造 prefix phase 切分 | 中 | artifact eval-only smoke | 每个 token 只 score 一次，phase 更新只用已 score docs |
| S4-T1 | 真 phased TTT MVP | LoRA on Q/K/V/O + MLP + head，3 phase global SGD | 高 | Q43 artifact eval-only | post-TTT 比 Q43 artifact pre-TTT 至少 -0.002 BPB |
| S4-T2 | TTT 参数扫 | rank 32/64/80，alpha、weight decay、warm-start A、prefix docs 1500/2500 | 高 | T1 正信号后 eval-only | 找到稳定收益且 eval time 可控 |
| S4-G1 | LeakyReLU^2 | `ReLU(x)^2` 改为 `LeakyReLU(x, 0.5)^2`，MLP_MULT 保持 2 | 小 | 完整 1h train + GPTQ | roundtrip 优于 Q43 或 pre 明显改善且量化损失不增 |
| S4-G2 | SmearGate BOS-safe | position-mixing gate，当前 token 为 BOS 时屏蔽前 token smear | 中 | 完整 1h train + GPTQ | pre/roundtrip 改善，无跨 doc 泄露 |
| S4-G3 | Sparse attention output gate | per-head narrow gate，`gate_window=12`，gate int8/float16 量化路径 | 中 | 完整 1h train + GPTQ | roundtrip 改善，artifact 仍 <=16MB |
| S4-RoPE1 | Partial RoPE | 只对 head_dim 前 16/64 dims 做 RoPE，剩余 dims no-RoPE | 小 | 短训后 1h | 短训 pre 改善约 0.001 后再完整跑 |
| S4-RoPE2 | FoPE 小矩阵 | `pure RoPE -> nomix+floor0.25 -> mix+zero -> mix+nozero` | 中 | 10-15min 短训筛选 | 只有优于 partial RoPE 时进入完整 1h |
| S4-P1 | Parallel decoder lane | 后段 decoder lane + learned lane mix，先不改层数 | 中高 | 完整 1h train + GPTQ | step 下降可接受，roundtrip 优于 Q43 |
| S4-P2 | 11L/MLP4x 受控移植 | 先 11L+MLP2，再 9L+MLP3/4，最后组合 | 高 | 0-step size guard + 短训 + 1h | artifact 容量可控，吞吐不抵消质量收益 |
| S4-K1 | FA3/varlen attention | 保留 torch SDPA flash 作为默认；doc/TTT 路径启用 FA3/varlen | 高 | kernel smoke + TTT eval | 固定 seq 不退化，doc-boundary eval/TTT 更快或可运行 |
| S4-K2 | fused CE / fused MLP | fused softcapped CE；若 G1 正收益则 fused LeakyReLU^2 MLP | 高 | microbench + 1h train | step time 改善或持平，BPB 不退化 |
| S4-K3 | Polar-Express NS | 5-step per-iteration minimax NS tuple 替换固定 Muon tuple | 小 | 完整 1h train + GPTQ | 与 `MUON_MOMENTUM=0.97` 叠加后 roundtrip 改善 |
| S4-Z1 | per-group compression | hot tensor 分组、similarity-sort、lrzip ZPAQ，其余 brotli | 中高 | export-only | 至少释放 200KB，roundtrip 权重一致 |
| S4-Z2 | 容量再分配 | 用 Z1 容量试 embed7/8、gate int8、LQER top/rank、MLP/11L | 中 | export-only + 1h train | 证明新增容量换来 BPB，而不是只变小 |

### 首批执行顺序

1. **S4-C0/C1：CaseOps 数据链路 smoke**。先恢复 tokenizer、dataset、byte sidecar 和 BOS 边界；没有 byte sidecar 之前不进入完整 1h。
2. **S4-T0/T1：正版 phased TTT eval-only**。先在 Q43 artifact 上验证 doc-level score-before-update TTT 是否有至少 `-0.002 BPB`，避免继续调 lm-head MVP。
3. **S4-G1/G2/G3：LeakyReLU^2、SmearGate、SparseGate 单项完整训练**。先单独验证，不和 parallel lane/11L 同时叠加。
4. **S4-K1/K2：内核支撑**。固定 seq 训练继续用 torch SDPA flash；只有当 TTT/doc-boundary 或 MLP/CE 成为瓶颈时引入 FA3/varlen/fused kernel。
5. **S4-Z1：压缩余量**。若 gate、11L、MLP4x、TTT adapter 元数据导致容量紧张，再做 per-group/lrzip；否则不把压缩当作独立 BPB 主线。
6. **组合实验**。只组合已经单项为正的组件，优先顺序为 `CaseOps + TTT`，再叠 `LeakyReLU^2 + SparseGate/SmearGate`，最后考虑 `parallel lane / 11L / MLP4x`。

### 记录格式

每个第四阶段实验新增 `experiments/exp_s4_<id>.md`，至少记录：

- 基座：Q43/R15、CaseOps 或新结构 checkpoint；
- 完整 env 命令、代码 diff 摘要、是否启用 FA3/fused kernel；
- step 数、step_avg、peak memory、train time；
- pre-quant BPB、roundtrip BPB、artifact bytes、total bytes；
- GPTQ/LQER/压缩配置，是否使用 per-group/lrzip；
- 若启用 CaseOps：tokenizer 路径、dataset 路径、byte sidecar 路径、`val_bpb:byte_sidecar:enabled` 日志；
- 若启用 TTT：pre-TTT、post-TTT、TTT gain、eval_time、prefix docs、phase 数、adapter 位置、rank/lr/wd、score-before-update 合规说明；
- 结论必须明确写为：继续叠加、需要 retune、或淘汰。

---

## 第五阶段：CaseOps 合规链路与 SOTA 主线重启（规划）

### 2026-06-08 实时修正

S5-C0/C1/C2 已完成：CaseOps 80 train shards、1 val shard、1 byte sidecar shard 已生成，full verifier 通过；根目录 `train_gpt.py` 已接入 `fineweb_val_bytes_*.bin`，并修复 `fineweb_val_*.bin` 误匹配 byte sidecar 的问题。真实训练日志已出现 `val_bpb:byte_sidecar:enabled`。

S5-C4/C5/C4b 首批 1xH100/1h 结果显示：root 静态模型直接切 CaseOps 并没有刷新 Phase 4 最佳。最佳完成结果是 S5-C5 `1.17380957`，明显负于普通 SP8192 QS28 的 `1.16522320`。因此后续不再做宽泛的 CaseOps root sweep；CaseOps 只保留三类目标实验：

1. schedule 诊断：S5-C6 用 CaseOps + records-like LR/warmdown 判断 C4/C5 是否只是调度不匹配。
2. records 栈验证：S5-R0 跑 2026-04-27 advanced stack 的单卡 fallback，判断 CaseOps 是否依赖 doc-boundary/TTT/结构联合设计。
3. true legal TTT：优先实现 score-before-update、single pass、no rescoring 的 doc-level eval path，而不是继续复用第三阶段 lm-head-only TTT。

空闲 GPU 的主线临时回到普通 SP8192 QS28，跑 S5-O1/S5-O2 检查训练动态是否还能突破平台期：records-like LR/warmdown 与更大 `TRAIN_BATCH_TOKENS=786432`。若这两条仍未超过 QS28，下一步应转向 TTT/结构联合实现，而不是继续做导出微扫。

S5-O1/O2/C6 首批结果已回收：O2 `TRAIN_BATCH_TOKENS=786432` 刷新到 `1.16474115`，是 Phase 5 第一个真实改善；O1 records-like LR/warmdown 量化后退化到 `1.16649737`，C6 CaseOps schedule 仍为 `1.17440583`。O2 checkpoint 的 export-only 窄扫中，Q1 `GPTQ_ERROR_SCALE=0.95` 进一步刷新到 `1.16465057`。O6 继续把 batch 上探到 `1,048,576` 后退化到 `1.16711896`，说明 O2 附近不是“越大越好”。O7-O10 的 bracket 最好是 O8 `TRAIN_BATCH_TOKENS=917504` 的 `1.16357912`，O9 later recurrence 基本持平 `1.16359641`；root QS28 仍能挤出小收益，但已被 records+lrzip+TTT 主线大幅超过。后续普通 SP8192 路线降为备份，不再消耗主要 GPU 配额。

结构/容量线出现强信号：S5-O5 `NUM_LAYERS=10` + int5/embed8 有效刷新到 `1.16432394`，但更重要的是 S5-O4 `MLP_MULT=3` + int5/embed8 取得 `1.15317521`，总字节 `16,806,331`，超 16MB 约 806KB。O4 不算有效提交，但它证明更宽 MLP 的 pre/roundtrip 质量大幅优于 QS28；后续优先做 O4 checkpoint 的容量修复（embed7、LQER top/rank、按层 bit、per-group/lrzip/brotli），而不是继续只追 `1e-4` 级 batch 微调。

O4 容量修复阶段性结论：P3 `embed7` 去 LQER 后仍为 `1.16400331` / `16,531,775` bytes，P4 `embed6 + LQER top3 + err0.95` 虽压到 `15,898,791` bytes，但 BPB 退化到 `1.16583102`，不如 S5-O5。因此 O4 只靠现有 bit/LQER 旋钮已经不值得继续；除非加入更强 per-group/simsort/packing，否则 GPU 优先级下调。

records 04-27 fallback 线成为新的主线：R1 checkpoint 在补齐 FA3/SDPA/TensorDescriptor/per-group fallback 后，R1Q2 已经完成量化验证，`val_bpb=1.09951341`，但 fallback-compressed 提交大小 `16,081,772` bytes，按 16,000,000-byte 上限超约 82KB。R1Q3 `EMBED_BITS=6 + LQER_TOP_K=3` 压到 `15,557,072` bytes，BPB `1.10542246`，成为第一条有效 near-1.10；R1Q4 `EMBED_BITS=7 + LQER_TOP_K=2` 几乎不损质量（`1.09951899`）但仍为 `16,079,398` bytes，说明 top_k 不是主要容量来源。自适应 brotli/lzma fallback（R1Q5/R1Q6）没有解决容量问题；安装真实外部 `lrzip` 后，R1Q7 用同一 embed7/top3 权重压到 `15,925,658` bytes，诊断 BPB 保持 `1.09951341`。R1Q2 的 3-phase TTT 长跑完成，post-TTT `1.08464298`，说明 TTT 仍有约 `0.0149 BPB` 收益；R1Q3T 的 embed6 artifact 也从 `1.10542246` 降到 `1.09020482`，证明 TTT 不是 embed7 特例，但 embed7/top3 更优。R1Q7T 已用合法 `lrzip` artifact 直接复现 3-phase TTT：post-TTT `1.08464351`，总字节 `15,925,658`，成为当前 Phase 5 有效最好。最终候选必须明确依赖真实 `lrzip` 可用，不能退回 fallback 压缩；若要继续逼近 `1.06`，下一步应围绕 R1Q7T 做 TTT 超参/phase/doc-prefix 优化或结构压缩联合，而不是再做 root QS28 微调。

与 2026-04-27 records 对照后，当前主要 gap 不在 TTT 是否有效，而在训练质量/吞吐：R1Q7T 的 TTT 收益约 `0.0149 BPB`，与 records phased TTT 的量级相近；但 R1Q7 的 no-TTT 起点 `1.09951341` 明显差于 records post-quant 约 `1.073-1.075`。因此当前 Phase 5 主线拆成两条并行分支：一是 T3-T8 固定 R1Q7 合法 artifact 扫 `prefix_docs`、phase 数和 `GLOBAL_TTT_LR`，寻找是否能低于 `1.08464351`；二是 R2/R3 复跑 records 04-27 同栈 seed42/seed0 的 1xH100 legal-lrzip 训练，判断单卡 fallback 下是否存在 seed 方差或更好训练轨迹。若 T3-T8 不能刷新且 R2/R3 no-TTT 仍在 `1.09+`，下一步应优先处理 FA3/fused MLP/训练吞吐 blocker，而不是继续增加 TTT 小扫。

T3-T8 已回收：最佳为 T5 `PHASED_TTT_PREFIX_DOCS=2500`、`PHASED_TTT_NUM_PHASES=4`，post-TTT `1.08461728`，只比 R1Q7T 默认好 `0.00002623 BPB`。这是一个真实刷新，但收益太小，不能改变主判断：TTT 近邻已接近平台。T9-T14 只做一次 4-phase 邻域确认（prefix 2000/2250/2750、5 phases、lr 0.0013/0.0015）；若仍只有 `1e-5` 级收益，停止 TTT 小扫，把 GPU 转给 records 训练质量/吞吐 blocker。

T9-T14 已确认平台：最佳 T12 `4 phases + GLOBAL_TTT_LR=0.0015` 到 `1.08458960`，相对 R1Q7T 只改善约 `0.000054 BPB`。因此 TTT 小扫停止，当前合法最好暂记 T12，但 Phase 5 后续主线转向训练起点。R2/R3/R9 覆盖 records seeds 42/0/1234；R4-R6 延后 layer loop 到 0.50/0.65/0.90，R7 测 917k batch，R8 测 `WARMDOWN_FRAC=0.95`。这些实验的判断标准是 no-TTT roundtrip 是否明显低于 R1Q7 的 `1.09951341`；若仍无改善，则 blocker 是单卡 fallback 吞吐/内核而不是超参。

R2/R3 已回收首个训练质量正信号：seed42 R2 训练 3188 steps，no-TTT roundtrip `1.09686294`，默认 3-phase TTT 后 `1.08218120`，刷新 Phase 5 合法最好并略优于 2026-04-06 记录 `1.08279`；seed0 R3 为 `1.09803468 -> 1.08311378`，也优于 R1Q7 但不如 R2。R2T12/R3T12 正在把 T12 的 `4 phases + GLOBAL_TTT_LR=0.0015` 迁移到 R2/R3 artifact；若能保持 T12 的小收益，R2 可能再降到约 `1.08213`。R4-R9 继续验证延后 loop、batch917k、warmdown095、seed1234 是否能进一步降低 no-TTT 起点。

R2T12/R3T12 已回收：R2T12 `1.08217886` 只比 R2 默认 TTT 好 `0.00000234 BPB`，R3T12 为 `1.08306801`。这确认 TTT 近邻不是主要杠杆；当前有效最好仍是 R2 系。R4-R6 中段显示延后 loop 能显著保持吞吐，尤其 R6 `loop090` 到 3500 step 仍约 `882k tok/s`，但 train loss 是否转化为更低 BPB 仍需最终 eval。基于这个中段信号，R10/R11 已启动：`loop080` 与 `loop090 + warmdown095`。

R4-R9 已把训练动态方向收窄：R8 `WARMDOWN_FRAC=0.95` 成为新的有效最好，no-TTT `1.09647097`、post-TTT `1.08179851`、总字节 `15,924,510`。延后 loop 单独不成立：R4/R5 虽增加步数但 post-TTT 只到 `1.08375/1.08320`，R6 `loop090` 直接退化到 `1.10134`；batch917k 也退化到 `1.08561`。因此后续主线从“loop/吞吐”改为“warmdown 邻域与跨 seed”：R8T12 迁移 T12 TTT 设置，R12/R13 探 `WARMDOWN_FRAC=0.90/0.98`，R14/R15 验证 seed0/1234 上 warmdown095 是否普适，R16 只作为 early-loop+warmdown 的交互诊断。

R8T12 已回收为 `1.08180270`，略差于 R8 默认 TTT，说明 R8 artifact 上不需要继续 TTT 微调。R17 追加 `WARMDOWN_FRAC=0.95 + MIN_LR=0.2`，用于判断 R8 的收益是否来自更晚/更弱的学习率衰减。

R10/R11 已关闭延后 loop 分支：`loop080` post-TTT `1.08734365`，`loop090 + warmdown095` `1.10325317`，均明显负于 R8。R18/R19 补充 warmdown/min_lr 邻域（0.95+0.15、0.98+0.2），当前第五阶段活线只剩 warmdown/min_lr 与跨 seed 验证。

R17 `MIN_LR=0.2` 的 no-TTT 已明显负向，说明不能简单提高最低学习率；R20 补充 `WARMDOWN_FRAC=0.94`，在 R8 的 0.95 附近做最后的窄扫。

R12-R17 收口：warmdown 0.90/0.98 都没有超过 R8 的 0.95；warmdown095 在 seed0/1234 上也不如 seed42；early-loop+warmdown 与 `MIN_LR=0.2` 均负向。因此训练侧当前唯一仍未收口的是 R20 `WARMDOWN_FRAC=0.94` 和 R18/R19 的 min_lr 邻域。并行启动 R8 checkpoint 的 export-only 量化微扫（R8Q1-Q5：err0.95、calib32/64、err1.05、top4），目标是在 R8 no-TTT `1.09647097` 基础上不重训再降一点；若 export sweep 无法改善，则 Phase 5 最好候选保持 R8 `1.08179851`。

R18/R19/R20 与 R8Q1-Q5 已全部回收且未刷新。旧环境下 Phase 5 有效最好保持 R8：`exp_s5_r8_seed42_warmdown095_records0427_caseops_lrzip_1xh100`，no-TTT `1.09647097`，post-TTT `1.08179851`，总字节 `15,924,510`。这条线的结论是：在单 H100/1h + fallback 环境下，records 04-27 栈可通过 seed42、真实 `lrzip`、warmdown095、默认 phased TTT 从 R1Q7T `1.08464` 推到 `1.08180`，已经略优于 2026-04-06 记录，但距离 2026-04-27 `~1.061` 仍主要差在训练吞吐/内核与 8 卡训练预算，而不是 TTT 微参、导出微参或 loop 延后。

2026-06-09/10 环境修复后，R8 不再接受静默 fallback。`pg291` 环境中 `torch 2.9.1+cu128`、FA3 `flash_attn_interface`、`triton.tools.tensor_descriptor` 和 real `lrzip` 均可用；训练命令加入 `REQUIRE_FA3=1 REQUIRE_TENSOR_DESCRIPTOR=1`，依赖缺失时直接失败。严格 R8-pg291 复跑日志确认 `train_loader:DocumentPackingLoader flash_attn_interface:True`，不再退到 fixed-sequence/SDPA/eager fallback；结果为 no-TTT `1.09043610`、post-TTT `1.07615418`、总字节 `15,917,614`，说明 FA3/doc-packing/fused MLP 是实质质量因素。

随后在 R8-pg291 上做 partial RoPE 低频置零小扫。代码新增 `ROPE_ZERO_LOW_FREQS`，默认 `0` 保持旧行为；`ROPE_DIMS=16 ROPE_ZERO_LOW_FREQS=2` 会把 8 个 RoPE `inv_freq` 中最低频的两个尾部元素置零。

| 变体 | steps | pre-EMA post-train BPB | no-TTT / roundtrip BPB | post-TTT BPB | bytes | 结论 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| R8-pg291 baseline | 3893 | 1.08176476 | 1.09043610 | 1.07615418 | 15,917,614 | 严格 R8 基线 |
| `ROPE_DIMS=16 ROPE_ZERO_LOW_FREQS=1` | 3909 | 1.08151851 | 1.09014656 | | 15,917,111 | no-TTT 小幅好于 baseline |
| `ROPE_DIMS=16 ROPE_ZERO_LOW_FREQS=2` | 3928 | **1.08117850** | **1.08972570** | **1.07586081** | 15,917,703 | 当前最佳 |
| `ROPE_DIMS=32 ROPE_ZERO_LOW_FREQS=1` | 3906 | 1.08258273 | 1.09116293 | | 15,917,904 | 退化 |

当前 Phase 5 的真实优化结论更新为：CaseOps 合规链路、records 04-27 结构栈、真实 `lrzip`、legal phased TTT、`pg291` kernel 环境和 partial RoPE zlf2 共同形成当前最佳 `1.07586081`。剩余 gap 仍主要在 no-TTT 起点：zlf2 no-TTT `1.08972570` 仍高于 04-27 records post-quant `1.073-1.075` 区间，因此主线应继续改善训练后 artifact，而不是继续做 TTT 或 export-only 微扫。

### 2026-06-10 下一步队列

以 `exp_rope_zlf2_r8_pg291_fa3_smear_seed42_nottt` / R8-pg291 配置作为新锚点，短期只做小矩阵：

| 优先级 | 方向 | 建议实验 | 判断标准 |
| --- | --- | --- | --- |
| P0 | 固化复现 | 记录 `pg291`、CaseOps 路径、real `lrzip`、`REQUIRE_*` 硬闸、`ROPE_ZERO_LOW_FREQS=2`、TTT_EVAL_ONLY 命令 | 可稳定复现 `1.08972570 -> 1.07586081` |
| P1 | RoPE 低频邻域 | `ROPE_ZERO_LOW_FREQS=3`、`ROPE_ZERO_LOW_FREQS=2 + ROPE_BASE` 小扰动，最多 2-3 组 | no-TTT 明显低于 `1.08972570` 才跑 TTT |
| P1 | seed 复验 | 在 `pg291 + zlf2` 下补 seed0/1234 或最少 seed0 | 判断 zlf2 是否只吃 seed42 噪声 |
| P2 | warmdown 复验 | 在 `pg291 + zlf2` 下扫 `WARMDOWN_FRAC=0.94/0.96`，避免重复旧 fallback 大矩阵 | no-TTT 改善 ≥ `0.0003` 才进入 TTT |
| P2 | batch/loop 时机 | 小规模复验 `TRAIN_BATCH_TOKENS=917504` 或 loop start，因 doc-packing 后旧负结果未必完全迁移 | 只保留 step 数和 no-TTT 同时改善的分支 |
| P3 | TTT/export 微扫 | 暂停 | 只有出现更强 no-TTT artifact 后再重启 |

### 阶段四收口

第四阶段实际完成了两类工作：

| 方向 | 当前状态 | 结论 |
| --- | --- | --- |
| 静态结构局部移植 | `SparseGate + LeakyReLU^2 + Polar NS` 经 QS28 导出达到 `1.16522320` | 比 Q43 改善约 `0.00213 BPB`，但远不足以缩小 SOTA gap |
| CaseOps 数据恢复 | 容器 `b5e2809a5863` 内 raw docs 已下载，`/tmp/caseops_smoke` 小样本 prepare/sidecar/root-loader smoke 通过 | S4-C0 从“缺数据”推进到“raw docs 与 prepare smoke 可用”，但完整 shards 与 byte-sidecar BPB 仍未完成 |

因此第四阶段的真实结论是：**Q43/QS28 附近继续做单开关或导出微扫已经进入平台期；第五阶段必须把主线转向 CaseOps 合规 BPB、doc-boundary TTT、结构/压缩联合设计。**

### 当前锚点

| 项目 | 值 |
| --- | --- |
| 本地最佳 | `exp_s4_qs28_g3g1k3_rerun_calib32_err0975_export` |
| Roundtrip BPB | **1.16522320** |
| 总字节 | **15,755,553** |
| Q43 对照 | `exp_s3_q43_r15_clip_top4_rank4_export`：`1.16735413` |
| records 参考 | `2026-04-27...1.0611`，3-seed mean 约 `1.06108` |
| CaseOps raw docs | `/base/datasets/CaseOps/raw/docs_selected.jsonl`，约 44GB |
| CaseOps smoke 记录 | `experiments/caseops_handoff.md` |

### 第五阶段核心原则

1. 不再把 `QS28 + export-only` 作为主要优化面。除非为新结构释放容量，否则不继续大量扫 `LQER_TOP_K`、`LQER_RANK`、`GPTQ_ERROR_SCALE`、`GPTQ_CALIBRATION_BATCHES`。
2. CaseOps 必须先合规再训练。没有 `val_bpb:byte_sidecar:enabled`、BOS sidecar byte=0、token/byte 长度对齐之前，不启动完整 1h CaseOps 实验。
3. TTT 必须依赖 doc boundary。不能复用第三阶段 lm-head LoRA MVP 作为判断依据；第五阶段只接受 score-before-update、single pass、no rescoring 的 doc-level TTT。
4. 结构改动必须和压缩容量一起设计。parallel lane、11L/MLP4x、gate、adapter 不能只看 pre BPB，必须从一开始检查 artifact bytes 和 roundtrip。
5. 内核优化服务于更复杂路线。固定 seq 训练继续使用 torch SDPA flash；FA3/varlen/fused CE/fused MLP 只有在 CaseOps/doc-boundary/TTT 或更复杂结构成为吞吐瓶颈时作为支撑线。

### 第五阶段成功标准

| 优先级 | 标准 |
| --- | --- |
| P0 | 完整 CaseOps dataset 生成完成，含 `fineweb_train_*.bin`、`fineweb_val_*.bin`、`fineweb_val_bytes_*.bin` |
| P1 | 根脚本 eval 日志出现 `val_bpb:byte_sidecar:enabled`，BOS sidecar byte=0，token/byte sidecar 长度一致 |
| P2 | CaseOps Q43/QS28 结构短训 smoke 可完成 train/eval/GPTQ roundtrip，BPB 分母使用原始 bytes |
| P3 | CaseOps 1h baseline roundtrip 优于普通 SP8192 Q43，或明确证明当前 CaseOps 接入存在可修问题 |
| P4 | 真 phased TTT eval-only 在同一 artifact 上至少 `-0.002 BPB`，且满足 score-before-update、single pass、doc boundary、no rescoring |
| P5 | 阶段五组合候选目标：roundtrip BPB <= `1.155`；若 TTT 可用，post-TTT 目标 <= `1.150` |

### 实验矩阵

| ID | 目的 | 基座 | 关键工作 | 通过 / 淘汰标准 |
| --- | --- | --- | --- | --- |
| S5-C0 | 完整 CaseOps 生成 | raw docs + records tokenizer | 运行 `prepare_caseops_data.py`，生成 full train/val/val_bytes shards | shard header 正常，val token/byte 长度一致，BOS byte=0 |
| S5-C1 | byte-sidecar BPB 接入 | 根目录 `train_gpt.py` | eval 加载 `fineweb_val_bytes_*.bin`，按 y token 对齐 sidecar byte count | smoke 日志出现 `val_bpb:byte_sidecar:enabled`；普通 SP8192 路径不退化 |
| S5-C2 | CaseOps loader/eval smoke | `/tmp/caseops_smoke` 与 full CaseOps | 极小模型 1-2 step + roundtrip，验证 tokenizer/sidecar/eval | train/eval/export 均跑通，BPB 使用 sidecar |
| S5-C3 | CaseOps 短训 | Q43 recipe 缩短版 | 10-15min train + GPTQ/LQER | pre/roundtrip 不异常，step_avg 可接受 |
| S5-C4 | CaseOps 1h baseline | Q43 recipe | 1xH100 1h，先不叠新 TTT/parallel lane | roundtrip 优于 Q43；若负收益，先排查 byte 对齐和 tokenizer |
| S5-C5 | CaseOps + QS28 结构 | SparseGate + LeakyReLU^2 + Polar NS | 在 CaseOps 上复测阶段四唯一正组合 | 必须优于 S5-C4 或显示可量化 pre 收益 |
| S5-T0 | doc boundary 切分 smoke | CaseOps val tokens | 从 BOS 构造 docs/prefix phase，记录每 token score 次数 | 每个 token 只 score 一次，phase update 只使用已 score docs |
| S5-T1 | true legal TTT MVP | Q43/QS28 artifact，优先 CaseOps eval | Q/K/V/O + MLP + head adapters，score-before-update | post-TTT 至少 `-0.002 BPB`，无 rescoring |
| S5-T2 | phased TTT sweep | S5-T1 正信号 | rank/lr/wd/phase/prefix docs/warm-start A | 收益稳定且 eval time 可接受 |
| S5-P0 | 容量 guard | 当前 GPTQ/LQER artifact | 估算 lane/11L/MLP4x/gate/adapters 字节预算 | 明确需要 Z 线释放多少容量 |
| S5-Z0 | per-group/simsort/lrzip smoke | QS28 artifact | export-only 验证权重重排与压缩恢复一致 | 至少释放 150-250KB，roundtrip 不变 |
| S5-P1 | parallel residual/lane MVP | CaseOps baseline 或 QS28 | 后段 lane + learned mix，小心控制参数量 | roundtrip 优于对应基座，step 损失可接受 |
| S5-P2 | 11L/MLP4x 受控移植 | S5-P1 或 CaseOps baseline | 先 11L/MLP2，再 9L/MLP3/4，最后组合 | artifact <=16MB，pre 收益能保到 roundtrip |
| S5-K0 | kernel microbench | S5-C/T/P 触发时 | FA3/varlen、fused CE、fused LeakyReLU^2 MLP | 吞吐改善或支撑 TTT/doc-boundary，BPB 不退化 |

### 首批执行顺序

1. **S5-C0：完整 CaseOps prepare**。在容器 `b5e2809a5863` 内从 `/base/datasets/CaseOps/raw/docs_selected.jsonl` 生成 full shards。该步骤可 CPU 长跑，完成后记录 shard 数、总 token 数、val byte sum。
2. **S5-C1/C2：byte-sidecar BPB 接入与 smoke**。先用 `/tmp/caseops_smoke` 做 1 step 验证，再切 full CaseOps val。目标日志必须包含 `val_bpb:byte_sidecar:enabled`。
3. **S5-C3/C4：CaseOps 短训与 1h baseline**。只用 Q43 recipe，不叠 TTT/parallel lane，先量化 CaseOps 本身收益。
4. **S5-T0/T1：doc-boundary TTT eval-only**。CaseOps 合规 eval 后再做 TTT；若 TTT 不产生至少 `-0.002 BPB`，先修合规/adapter path，不急着跑组合训练。
5. **S5-Z0/P0：压缩容量审计**。若 CaseOps+TTT 有正信号，再为 lane/11L/MLP4x/adapters 释放容量。
6. **S5-C5/P1/P2 组合实验**。组合顺序优先 `CaseOps + TTT`，再叠 `SparseGate + LeakyReLU^2 + Polar NS`，最后叠 parallel lane/11L/MLP4x。

### 记录格式

第五阶段每个实验新增 `experiments/exp_s5_<id>.md`，至少记录：

- 数据路径：`DATA_PATH`、`TOKENIZER_PATH`、`fineweb_val_bytes_*.bin` pattern；
- CaseOps 合规项：sidecar header、token/byte 长度、BOS count、BOS byte=0、byte sum；
- 完整 env 命令和代码 diff 摘要；
- step 数、step_avg、peak memory、train/eval/export time；
- pre BPB、roundtrip BPB、若启用 TTT 则记录 pre-TTT/post-TTT；
- artifact bytes、total bytes、GPTQ/LQER/per-group/lrzip 配置；
- TTT 合规项：doc 数、phase 数、prefix docs、adapter 位置、rank/lr/wd、score-before-update、single pass、no rescoring；
- 结论必须写明：继续叠加、修实现、降优先级、或淘汰。
