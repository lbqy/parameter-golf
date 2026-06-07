# 第二阶段量化里程碑

日期：2026-06-06

约束：1xH100 / 1h 训练，提交物 <= 16 MB，主指标为 roundtrip `val_bpb`。

## 结论

第二阶段完成：GPTQ + bit packing + brotli + fresh compiled roundtrip 验证已正常工作，LQER asymmetric 修正可用。当前最佳合规提交候选为 B3 checkpoint 上的 GPTQ + LQER：

| ID | RUN_ID | 基座 | 方法 | Roundtrip BPB | 总字节 | 结论 |
| --- | --- | --- | --- | ---: | ---: | --- |
| G-B3-1-LQER | `exp_gptq_b3_fresh_i6e8_quantile_err1_lqer` | SP8192 + Seq4096 | GPTQ int6/embed8 + LQER rank4 top3 | **1.17661956** | **15,738,799** | 当前最佳，合规 |
| G-B3-1 | `exp_gptq_b3_fresh_i6e8_quantile_err1` | SP8192 + Seq4096 | GPTQ int6/embed8 | 1.18001764 | 15,731,454 | B3 GPTQ 基线 |
| G-B3-025-LQER | `exp_gptq_b3_fresh_i6e8_quantile_err025_lqer` | SP8192 + Seq4096 | GPTQ int6/embed8 + LQER rank4 top3 | 1.18007382 | 15,740,005 | 低 error_scale 不优 |
| G-R9-025-LQER | `exp_gptq_r9_fresh_i6e8_quantile_err025_lqer` | SP8192 + Seq2048 | GPTQ int6/embed8 + LQER rank4 top3 | 1.17978818 | 15,803,202 | R9 最佳 |
| R9-full | `exp_r9_full_sp8192_seq2048_rtn_i6e8_brotli_cv7_l3_8` | SP8192 + Seq2048 | RTN override | 1.18714581 | 15,972,698 | 第二阶段 RTN 对照 |

## B3 基座

| RUN_ID | Seq | Step | Pre-quant BPB | int8+zlib BPB | int8+zlib 总字节 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `exp_b3_sp8192_seq4096_1xh100_rootbaseline` | 4096 | 6485 | 1.1745 | 1.18068959 | 19,287,254 |

B3 原始 int8 仍超 16MB，但配合 GPTQ+LQER 后明显优于 R9，后续第三阶段以 B3 作为默认 baseline。

## 关键修复

- 新增 `gptq_export.py`：Hessian GPTQ、signed bit packing、mixed bitwidth、LQER asymmetric int4。
- `train_gpt.py` 支持 `QUANT_MODE=gptq`、`COMPRESSOR=brotli|zlib|lzma`、`GPTQ_SCALE_MODE=quantile`、`GPTQ_ERROR_SCALE`、`GPTQ_SCALE_FLOOR`、`LQER_*`。
- 启用 `FRESH_MODEL_AFTER_QUANT=1`：量化后新建模型、加载 roundtrip 权重、重新 compile 后验证，避免 Hessian collection 污染 eval。
- 旧 B2 checkpoint 的 export-only GPTQ 结果仅保留为诊断，不再作为 current-code 结论。

## 推荐复现实验核心

```bash
QUANT_MODE=gptq COMPRESSOR=brotli FRESH_MODEL_AFTER_QUANT=1 \
GPTQ_SCALE_MODE=quantile GPTQ_SCALE_FLOOR=1 GPTQ_ERROR_SCALE=1 \
MATRIX_BITS=6 EMBED_BITS=8 GPTQ_CALIBRATION_BATCHES=16 \
LQER_ENABLED=1 LQER_RANK=4 LQER_TOP_K=3 \
LQER_FACTOR_BITS=4 LQER_ASYM_ENABLED=1 LQER_ASYM_GROUP=64
```

## 日志路径

- B3 基座：`results/experiments/exp_b3_sp8192_seq4096_1xh100_rootbaseline/logs/exp_b3_sp8192_seq4096_1xh100_rootbaseline.txt`
- B3 最佳 GPTQ：`results/experiments/exp_gptq_b3_fresh_i6e8_quantile_err1_lqer/logs/exp_gptq_b3_fresh_i6e8_quantile_err1_lqer.txt`
- R9 最佳 GPTQ：`results/experiments/exp_gptq_r9_fresh_i6e8_quantile_err025_lqer/logs/exp_gptq_r9_fresh_i6e8_quantile_err025_lqer.txt`
