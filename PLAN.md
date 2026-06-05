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
- 第二阶段不再改架构/训练时长，专注 **GPTQ 压缩**。

### B2 训练 backbone（第二阶段固定）

```text
SP8192, TRAIN_SEQ_LEN=2048, 1×H100, MAX_WALLCLOCK_SECONDS=3600, SEED=1337
DATA_PATH=/base/datasets/SP8192/datasets/fineweb10B_sp8192
TOKENIZER_PATH=/base/datasets/SP8192/tokenizers/fineweb_8192_bpe.model
VOCAB_SIZE=8192
VAL_LOSS_EVERY=0, TRAIN_LOG_EVERY=200, TRAIN_BATCH_TOKENS=524288
```

---

## 第二阶段：GPTQ 量化压缩（进行中）

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
| `gptq` | `final_model.gptq.ptz` | Hessian GPTQ + SDClip + brotli/zlib |

> RTN 仅保留 `ref_config.md` 中已有对照结果，不再安排新 RTN 实验。

### 成功标准

| 优先级 | 标准 |
| --- | --- |
| P0 | 总提交大小 ≤ 16 MB |
| P1 | roundtrip BPB < 1.20 |
| P2 | roundtrip BPB ≤ 1.185（接近 B2 int8） |

### 实验矩阵

| ID | 方法 | Matrix | Embed | 压缩 | 需改代码 |
| --- | --- | ---: | ---: | --- | :---: |
| **G1** | GPTQ + SDClip | 6 | 8 | brotli | 已移植 |
| **G2** | GPTQ + SDClip | 6 | 7 | brotli | 已移植 |
| **G3** | GPTQ + SDClip + LQER | 6 | 7 | brotli | 已移植 |

**执行顺序**：G1 → G2 → G3

### GPTQ 环境变量

| 变量 | G1 | G2 | 说明 |
| --- | --- | --- | --- |
| `QUANT_MODE` | gptq | gptq | 启用 GPTQ 导出 |
| `COMPRESSOR` | brotli | brotli | 无 brotli 时改 `zlib` |
| `MATRIX_BITS` | 6 | 6 | 矩阵 bitwidth |
| `EMBED_BITS` | 8 | 7 | embedding bitwidth |
| `MATRIX_CLIP_SIGMAS` | 12.85 | 12.85 | 默认矩阵 clip |
| `ATTN_CLIP_SIGMAS` | 13.0 | 13.0 | attention clip |
| `MLP_CLIP_SIGMAS` | 11.5 | 11.5 | MLP clip |
| `EMBED_CLIP_SIGMAS` | 14.0 | 14.0 | embedding clip |
| `GPTQ_CALIBRATION_BATCHES` | 32 | 32 | Hessian 校准 batch 数 |

**G3** — `RUN_ID=exp_g3_sp8192_seq2048_gptq_lqer_1xh100`，`EMBED_BITS=7`，并追加：

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
GPTQ_CALIBRATION_BATCHES=32 \
/opt/conda/bin/torchrun --standalone --nproc_per_node=1 /base/project/parameter-golf/train_gpt.py
```

**G1** — `RUN_ID=exp_g1_sp8192_seq2048_gptq_int6e8`，`EMBED_BITS=8`

**G2** — `RUN_ID=exp_g2_sp8192_seq2048_gptq_int6e7`，`EMBED_BITS=7`

### 结果记录

每个实验写入 `experiments/exp_<id>.md`，包含：完整命令、pre-quant BPB、`final_gptq+brotli_roundtrip_exact val_bpb`、总字节、是否 ≤16MB。

### 结果矩阵（待填）

| ID | Embed | 方法 | Roundtrip BPB | 总字节 | ≤16MB |
| --- | ---: | --- | ---: | ---: | :---: |
| B2 ref | 8 | int8+zlib | 1.1815 | 19,343,563 | ✗ |
| G1 | 8 | GPTQ int6/8 | | | |
| G2 | 7 | GPTQ int6/7 | | | |
| G3 | 7 | GPTQ + LQER | | | |
