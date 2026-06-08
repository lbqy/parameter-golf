# 第五阶段实验状态

日期：2026-06-08

约束：完整训练实验使用 1xH100，`MAX_WALLCLOCK_SECONDS=3600`。Smoke 只作为环境/合规检查，不计入实验矩阵。

## CaseOps 数据与评估闸门

已完成：

| 项目 | 结果 |
| --- | --- |
| CaseOps train shards | 80 |
| CaseOps val shards | 1 |
| CaseOps val byte sidecar shards | 1 |
| val tokens | 9,662,502 |
| val byte sum | 29,950,979 |
| BOS count | 10,000 |
| bad BOS bytes | 0 |

根脚本修复：

- `fineweb_val_*.bin` 验证 glob 不再误读 `fineweb_val_bytes_*.bin`。
- 若 sidecar 存在，`eval_val()` 按 shifted `y` target 对齐 sidecar byte count 计算 BPB。
- CaseOps smoke 日志已出现 `val_bpb:byte_sidecar:enabled`，且 full dataset verifier 通过。

## 1h 实验队列

| ID | RUN_ID | GPU | 基座 | 关键改动 | 状态 |
| --- | --- | ---: | --- | --- | --- |
| S5-C4 | `exp_s5_c4_caseops_q43_seq4096_1xh100` | 3 | root Q43 | CaseOps, seq4096 | completed |
| S5-C5 | `exp_s5_c5_caseops_qs28_sparse_leaky_polarns_seq4096_1xh100` | 4 | root QS28 | CaseOps + SparseGate + LeakyReLU^2 + PolarNS, seq4096 | completed |
| S5-C4b | `exp_s5_c4b_caseops_q43_seq2048_1xh100` | 5 | root Q43 | CaseOps, seq2048 | completed |
| S5-R0 | `exp_s5_r0_records0427_caseops_advanced_1xh100` | 6 | records 2026-04-27 | advanced stack with FA3/TensorDescriptor fallbacks | failed at eval compile fallback |
| S5-C5b | `exp_s5_c5b_caseops_qs28_sparse_leaky_polarns_seq2048_1xh100` | 7 | root QS28 | CaseOps + SparseGate + LeakyReLU^2 + PolarNS, seq2048 | completed |
| S5-R0b | `exp_s5_r0b_records0427_caseops_evalfallback_prequant5m` | 6 | records 2026-04-27 | prequant-only eval fallback verification | failed at SDPA backend fallback |
| S5-R0c | `exp_s5_r0c_records0427_caseops_sdpafallback_prequant90s` | 6 | records 2026-04-27 | prequant-only SDPA fallback verification | failed at compiled training fallback |
| S5-R0d | `exp_s5_r0d_records0427_caseops_varleneager_prequant90s` | 6 | records 2026-04-27 | prequant-only varlen-only eager SDPA fallback | completed |
| S5-R1 | `exp_s5_r1_records0427_caseops_advanced_fallback_1xh100` | 6 | records 2026-04-27 | advanced stack after eval/SDPA fallback fixes | failed at missing `lrzip` |
| S5-O1 | `exp_s5_o1_sp8192_qs28_recordslr_warmdown_1xh100` | 3 | root QS28 | ordinary SP8192, records-like LR/warmdown | completed |
| S5-O2 | `exp_s5_o2_sp8192_qs28_batch786k_1xh100` | 4 | root QS28 | ordinary SP8192, `TRAIN_BATCH_TOKENS=786432` | completed |
| S5-C6 | `exp_s5_c6_caseops_qs28_recordslr_warmdown_1xh100` | 5 | root QS28 | CaseOps, records-like LR/warmdown | completed |
| S5-O3 | `exp_s5_o3_sp8192_qs28_seq8192_1xh100` | 7 | root QS28 | ordinary SP8192, `TRAIN_SEQ_LEN=8192` | completed |
| S5-O4 | `exp_s5_o4_sp8192_qs28_mlp3_i5e8_1xh100` | 1 | root QS28 | ordinary SP8192, `MLP_MULT=3`, matrix int5/embed8 | completed-over-cap |
| S5-O5 | `exp_s5_o5_sp8192_qs28_10l_i5e8_1xh100` | 2 | root QS28 | ordinary SP8192, `NUM_LAYERS=10`, matrix int5/embed8 | completed |
| S5-O6 | `exp_s5_o6_sp8192_qs28_batch1048k_1xh100` | 0 | root QS28 | ordinary SP8192, `TRAIN_BATCH_TOKENS=1048576` | completed |
| S5-Q1 | `exp_s5_q1_o2_export_err095` | 3 | O2 checkpoint | export-only, `GPTQ_ERROR_SCALE=0.95` | completed |
| S5-Q2 | `exp_s5_q2_o2_export_err100` | 4 | O2 checkpoint | export-only, `GPTQ_ERROR_SCALE=1.0` | completed |
| S5-Q3 | `exp_s5_q3_o2_export_calib64_err0975` | 5 | O2 checkpoint | export-only, `GPTQ_CALIBRATION_BATCHES=64` | completed |
| S5-O7 | `exp_s5_o7_sp8192_qs28_batch655k_err095_1xh100` | 3 | root QS28 | ordinary SP8192, batch 655k, export err0.95 | completed |
| S5-O8 | `exp_s5_o8_sp8192_qs28_batch917k_err095_1xh100` | 4 | root QS28 | ordinary SP8192, batch 917k, export err0.95 | completed |
| S5-O9 | `exp_s5_o9_sp8192_qs28_batch786k_recur040_err095_1xh100` | 5 | root QS28 | ordinary SP8192, batch 786k, recurrence at 0.40, export err0.95 | completed |
| S5-O10 | `exp_s5_o10_sp8192_qs28_batch786k_recur020_err095_1xh100` | 7 | root QS28 | ordinary SP8192, batch 786k, recurrence at 0.20, export err0.95 | completed |
| S5-P1 | `exp_s5_p1_o4_mlp3_i5e7_top4_export` | 1 | O4 checkpoint | export-only capacity fix, embed7 top4 | completed-over-cap |
| S5-P2 | `exp_s5_p2_o4_mlp3_i5e7_top3_err095_export` | 2 | O4 checkpoint | export-only capacity fix, embed7 top3 err0.95 | completed-over-cap |
| S5-R1Q | `exp_s5_r1q_records0427_caseops_exportfallback_ttt` | 6 | R1 checkpoint | records export-only after per-group brotli fallback patch | superseded |
| S5-P3 | `exp_s5_p3_o4_mlp3_i5e7_nolqer_export` | 1 | O4 checkpoint | export-only, embed7 without LQER | completed-over-cap |
| S5-P4 | `exp_s5_p4_o4_mlp3_i5e6_top3_err095_export` | 2 | O4 checkpoint | export-only, embed6/top3 err0.95 | completed |
| S5-R1Q2 | `exp_s5_r1q2_records0427_caseops_exportfallback_ttt` | 6 | R1 checkpoint | records export-only after BOS/per-group fixes, with phased TTT | completed-over-cap |
| S5-R1Q3 | `exp_s5_r1q3_records0427_caseops_embed6_top3_nottt` | 1 | R1 checkpoint | records shrink probe, embed6/top3, no TTT | completed |
| S5-R1Q4 | `exp_s5_r1q4_records0427_caseops_embed7_top2_nottt` | 2 | R1 checkpoint | records shrink probe, embed7/top2, no TTT | completed-over-cap |
| S5-R1Q5 | `exp_s5_r1q5_records0427_caseops_embed7_top3_adaptzip_nottt` | 1 | R1 checkpoint | records shrink probe, embed7/top3, adaptive brotli/lzma per-group fallback, no TTT | completed-over-cap |
| S5-R1Q6 | `exp_s5_r1q6_records0427_caseops_embed7_top2_adaptzip_nottt` | 2 | R1 checkpoint | records shrink probe, embed7/top2, adaptive brotli/lzma per-group fallback, no TTT | failed |
| S5-R1Q7 | `exp_s5_r1q7_records0427_caseops_embed7_top3_lrzip_nottt` | 1 | R1 checkpoint | records export-only with real external `lrzip`, embed7/top3, no TTT | completed |
| S5-R1Q7T | `exp_s5_r1q7t_records0427_caseops_embed7_top3_lrzip_ttt` | 1 | R1Q7 artifact | legal artifact TTT eval-only, embed7/top3, 3 phases | completed |
| S5-R1Q3T | `exp_s5_r1q3t_records0427_caseops_embed6_top3_ttt_eval` | 2 | R1Q3 artifact | legal fallback artifact TTT eval-only, embed6/top3, 1 phase | completed |
| S5-T3 | `exp_s5_t3_r1q7_p2000_ph3_ttt` | 0 | R1Q7 artifact | TTT eval-only, prefix docs 2000, 3 phases | completed |
| S5-T4 | `exp_s5_t4_r1q7_p3000_ph3_ttt` | 1 | R1Q7 artifact | TTT eval-only, prefix docs 3000, 3 phases | completed |
| S5-T5 | `exp_s5_t5_r1q7_p2500_ph4_ttt` | 2 | R1Q7 artifact | TTT eval-only, prefix docs 2500, 4 phases | completed |
| S5-T6 | `exp_s5_t6_r1q7_p2500_ph2_ttt` | 3 | R1Q7 artifact | TTT eval-only, prefix docs 2500, 2 phases | completed |
| S5-T7 | `exp_s5_t7_r1q7_p2500_ph3_glr0007_ttt` | 4 | R1Q7 artifact | TTT eval-only, prefix docs 2500, 3 phases, global lr 0.0007 | completed |
| S5-T8 | `exp_s5_t8_r1q7_p2500_ph3_glr0013_ttt` | 5 | R1Q7 artifact | TTT eval-only, prefix docs 2500, 3 phases, global lr 0.0013 | completed |
| S5-R2 | `exp_s5_r2_seed42_records0427_caseops_lrzip_1xh100` | 6 | records 2026-04-27 | full 1xH100 seed 42 rerun, legal `lrzip`, same TTT config | completed |
| S5-R3 | `exp_s5_r3_seed0_records0427_caseops_lrzip_1xh100` | 7 | records 2026-04-27 | full 1xH100 seed 0 rerun, legal `lrzip`, same TTT config | completed |
| S5-T9 | `exp_s5_t9_r1q7_p2000_ph4_ttt` | 0 | R1Q7 artifact | TTT eval-only, prefix docs 2000, 4 phases | completed |
| S5-T10 | `exp_s5_t10_r1q7_p2500_ph5_ttt` | 1 | R1Q7 artifact | TTT eval-only, prefix docs 2500, 5 phases | completed |
| S5-T11 | `exp_s5_t11_r1q7_p2500_ph4_glr0013_ttt` | 2 | R1Q7 artifact | TTT eval-only, prefix docs 2500, 4 phases, global lr 0.0013 | completed |
| S5-T12 | `exp_s5_t12_r1q7_p2500_ph4_glr0015_ttt` | 3 | R1Q7 artifact | TTT eval-only, prefix docs 2500, 4 phases, global lr 0.0015 | completed |
| S5-T13 | `exp_s5_t13_r1q7_p2250_ph4_ttt` | 4 | R1Q7 artifact | TTT eval-only, prefix docs 2250, 4 phases | completed |
| S5-T14 | `exp_s5_t14_r1q7_p2750_ph4_ttt` | 5 | R1Q7 artifact | TTT eval-only, prefix docs 2750, 4 phases | completed |
| S5-R4 | `exp_s5_r4_seed42_loop050_records0427_caseops_lrzip_1xh100` | 0 | records 2026-04-27 | seed42 full rerun, delay layer loop to 0.50 | completed |
| S5-R5 | `exp_s5_r5_seed42_loop065_records0427_caseops_lrzip_1xh100` | 1 | records 2026-04-27 | seed42 full rerun, delay layer loop to 0.65 | completed |
| S5-R6 | `exp_s5_r6_seed42_loop090_records0427_caseops_lrzip_1xh100` | 2 | records 2026-04-27 | seed42 full rerun, effectively late/no layer loop at 0.90 | completed |
| S5-R7 | `exp_s5_r7_seed42_batch917k_records0427_caseops_lrzip_1xh100` | 3 | records 2026-04-27 | seed42 full rerun, `TRAIN_BATCH_TOKENS=917504` | completed |
| S5-R8 | `exp_s5_r8_seed42_warmdown095_records0427_caseops_lrzip_1xh100` | 4 | records 2026-04-27 | seed42 full rerun, `WARMDOWN_FRAC=0.95` | completed |
| S5-R9 | `exp_s5_r9_seed1234_records0427_caseops_lrzip_1xh100` | 5 | records 2026-04-27 | full 1xH100 seed 1234 rerun, legal `lrzip` | completed |
| S5-R2T12 | `exp_s5_r2t12_seed42_ph4_glr0015_ttt` | 6 | R2 artifact | eval-only, T12 TTT settings: 4 phases, global lr 0.0015 | completed |
| S5-R3T12 | `exp_s5_r3t12_seed0_ph4_glr0015_ttt` | 7 | R3 artifact | eval-only, T12 TTT settings: 4 phases, global lr 0.0015 | completed |
| S5-R10 | `exp_s5_r10_seed42_loop080_records0427_caseops_lrzip_1xh100` | 6 | records 2026-04-27 | seed42 full rerun, delay layer loop to 0.80 | completed |
| S5-R11 | `exp_s5_r11_seed42_loop090_warmdown095_records0427_caseops_lrzip_1xh100` | 7 | records 2026-04-27 | seed42 full rerun, loop 0.90 plus warmdown 0.95 | completed |
| S5-R8T12 | `exp_s5_r8t12_warmdown095_ph4_glr0015_ttt` | 0 | R8 artifact | eval-only, T12 TTT settings on warmdown095 artifact | completed |
| S5-R12 | `exp_s5_r12_seed42_warmdown090_records0427_caseops_lrzip_1xh100` | 1 | records 2026-04-27 | seed42 full rerun, `WARMDOWN_FRAC=0.90` | completed |
| S5-R13 | `exp_s5_r13_seed42_warmdown098_records0427_caseops_lrzip_1xh100` | 2 | records 2026-04-27 | seed42 full rerun, `WARMDOWN_FRAC=0.98` | completed |
| S5-R14 | `exp_s5_r14_seed0_warmdown095_records0427_caseops_lrzip_1xh100` | 3 | records 2026-04-27 | seed0 full rerun, `WARMDOWN_FRAC=0.95` | completed |
| S5-R15 | `exp_s5_r15_seed1234_warmdown095_records0427_caseops_lrzip_1xh100` | 4 | records 2026-04-27 | seed1234 full rerun, `WARMDOWN_FRAC=0.95` | completed |
| S5-R16 | `exp_s5_r16_seed42_loop025_warmdown095_records0427_caseops_lrzip_1xh100` | 5 | records 2026-04-27 | seed42 full rerun, loop 0.25 plus warmdown 0.95 | completed |
| S5-R17 | `exp_s5_r17_seed42_warmdown095_minlr020_records0427_caseops_lrzip_1xh100` | 0 | records 2026-04-27 | seed42 full rerun, warmdown 0.95 plus `MIN_LR=0.2` | failed-after-quant |
| S5-R18 | `exp_s5_r18_seed42_warmdown095_minlr015_records0427_caseops_lrzip_1xh100` | 6 | records 2026-04-27 | seed42 full rerun, warmdown 0.95 plus `MIN_LR=0.15` | completed |
| S5-R19 | `exp_s5_r19_seed42_warmdown098_minlr020_records0427_caseops_lrzip_1xh100` | 7 | records 2026-04-27 | seed42 full rerun, warmdown 0.98 plus `MIN_LR=0.2` | completed |
| S5-R20 | `exp_s5_r20_seed42_warmdown094_records0427_caseops_lrzip_1xh100` | 0 | records 2026-04-27 | seed42 full rerun, `WARMDOWN_FRAC=0.94` | completed |
| S5-R8Q1 | `exp_s5_r8q1_export_err095` | 1 | R8 checkpoint | export-only, `GPTQ_ERROR_SCALE=0.95`, no TTT | completed |
| S5-R8Q2 | `exp_s5_r8q2_export_calib32` | 2 | R8 checkpoint | export-only, `GPTQ_CALIBRATION_BATCHES=32`, no TTT | completed |
| S5-R8Q3 | `exp_s5_r8q3_export_calib64` | 3 | R8 checkpoint | export-only, `GPTQ_CALIBRATION_BATCHES=64`, no TTT | completed |
| S5-R8Q4 | `exp_s5_r8q4_export_err105` | 4 | R8 checkpoint | export-only, `GPTQ_ERROR_SCALE=1.05`, no TTT | completed |
| S5-R8Q5 | `exp_s5_r8q5_export_top4` | 5 | R8 checkpoint | export-only, `LQER_TOP_K=4`, no TTT | completed |

## Startup Checks

Root runs:

```text
val_bpb:byte_sidecar:enabled files:1 bytes_sum:29950821 bos_count:10000 bad_bos_bytes:0
train_loader:dataset:fineweb10B_sp8192_lossless_caps_caseops_v1_reserved train_shards:80
```

Records run required environment fallbacks:

| Issue | Fix |
| --- | --- |
| missing `flash_attn_interface` | fallback to torch SDPA |
| missing `triton.tools.tensor_descriptor` | fallback eager LeakyReLU^2 MLP path |
| varlen fallback incompatible with `torch.compile` | `FixedSequenceTrainLoader` when FA3 is unavailable |
| fallback loader did not set `BOS_ID` | initialize global `BOS_ID=1` |
| cu bucket warmup assumed doc packing | skip cu bucket warmup for fixed sequence loader |

Current records log confirms:

```text
train_loader:FixedSequenceTrainLoader flash_attn_interface:False
warmup_cu_buckets:skipped fixed_sequence_loader
```

## Results Matrix

| ID | Steps | Step Avg | Pre BPB | Roundtrip BPB | Post-TTT BPB | Total Bytes | Conclusion |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| S5-C4 | 5536 | 650.37ms | 1.1710 | 1.17385239 | | 15,776,719 | CaseOps root Q43 负于 Phase 4 QS28；seq4096 明显好于 seq2048 |
| S5-C5 | 5337 | 674.60ms | 1.1683 | 1.17380957 | | 15,770,457 | CaseOps + QS28 结构 pre 有小幅收益，但量化后与 C4 持平 |
| S5-C4b | 7022 | 512.71ms | 1.1780 | 1.18091384 | | 15,776,916 | CaseOps seq2048 明显差；不继续该长度 |
| S5-R0 | 3000 | ~706k tok/s | | | | | failed at final prequant eval because torch.compile could not guard varlen fallback slices without FA3 |
| S5-C5b | 6791 | 530.15ms | 1.1738 | 1.18037367 | | 15,759,522 | CaseOps QS28 seq2048 clearly worse than seq4096; do not continue seq2048 CaseOps |
| S5-R0b | ~119 | | | | | | eager eval reached fallback attention but failed with `RuntimeError: No available kernel` under global flash-only SDPA |
| S5-R0c | warmup | | | | | | failed because `sdp_kernel` context was placed inside the compile-time fixed-seq fallback path |
| S5-R0d | 80 | | 3.04137398 | | | | 90s prequant-only verification succeeded; fallback eval path is now runnable without FA3 |
| S5-R1 | trained | | | | | | trained but failed during per-group compression because `lrzip` binary is missing |
| S5-O1 | 5386 | 668.45ms | 1.1648 | 1.16649737 | | 15,633,997 | records-like LR/warmdown improves pre over Phase 4 slightly but quantized result is worse; do not continue this schedule alone |
| S5-O2 | 3662 | 983.23ms | 1.1627 | **1.16474115** | | 15,766,242 | new best; larger batch token throughput is a real positive signal |
| S5-C6 | 5356 | 672.27ms | 1.1723 | 1.17440583 | | 15,634,526 | CaseOps remains negative even with records-like schedule; stop root CaseOps static training |
| S5-O3 | 3727 | 966.21ms | 1.1704 | 1.17215222 | | 15,774,996 | seq8192 is clearly worse; do not continue long-context root QS28 |
| S5-O4 | 5055 | 712.26ms | 1.1470 | 1.15317521 | | 16,806,331 | very strong quality but over 16MB; prioritize capacity repair from checkpoint |
| S5-O5 | 4844 | 743.23ms | 1.1582 | 1.16432394 | | 15,116,709 | valid and new best, but much weaker than over-cap MLP3 |
| S5-O6 | 2734 | 1316.96ms | 1.1653 | 1.16711896 | | 15,747,478 | larger 1,048k batch is worse than O2/Q1; stop upward batch-size direction |
| S5-Q1 | export-only | | | **1.16465057** | | 15,766,671 | new best; O2 checkpoint prefers `GPTQ_ERROR_SCALE=0.95` |
| S5-Q2 | export-only | | | 1.16476925 | | 15,766,427 | worse than O2/Q1 |
| S5-Q3 | export-only | | | 1.16466107 | | 15,766,871 | close to Q1 but worse/larger |
| S5-O7 | 4350 | 827.65ms | 1.1627 | 1.16460123 | | 15,760,713 | lower batch bracket roughly matches Q1, but does not beat root best |
| S5-O8 | 3168 | 1136.43ms | 1.1616 | **1.16357912** | | 15,761,143 | best root QS28 result so far; still dwarfed by records line |
| S5-O9 | 3768 | 955.63ms | 1.1618 | 1.16359641 | | 15,766,873 | later recurrence start is essentially tied with O8 |
| S5-O10 | 3564 | 1010.25ms | 1.1623 | 1.16423352 | | 15,762,765 | earlier recurrence start is worse than O8/O9 |
| S5-P1 | export-only | | | 1.15421412 | | 16,540,848 | over cap; embed7/top4 does not recover enough capacity |
| S5-P2 | export-only | | | 1.15432447 | | 16,538,418 | over cap; embed7/top3/err0.95 remains too large |
| S5-R1Q | export-only | | | | | | superseded by R1Q2/R1Q7 after BOS/per-group and real `lrzip` fixes |
| S5-P3 | export-only | | | 1.16400331 | | 16,531,775 | O4 MLP3 quality remains decent but is still 531KB over cap; no-LQER is not enough |
| S5-P4 | export-only | | | 1.16583102 | | 15,898,791 | valid but worse than S5-O5; embed6/top3 loses too much quality |
| S5-R1Q2 | export-only | | | 1.09951341 | 1.08464298 | 16,081,772 | fallback-compressed artifact is invalid, but 3-phase TTT signal is strong and transfers to the same decompressed weights as legal R1Q7 |
| S5-R1Q3 | export-only | | | 1.10542246 | | 15,557,072 | valid fallback result with embed6; superseded by R1Q7 once real `lrzip` is available |
| S5-R1Q4 | export-only | | | 1.09951899 | | 16,079,398 | still ~79KB over cap under fallback compression; reducing LQER top_k is almost size-neutral |
| S5-R1Q5 | export-only | | | 1.09951341 | | 16,081,823 | adaptive brotli/lzma fallback did not improve size; larger code made it slightly worse |
| S5-R1Q6 | export-only | | | | | | failed transient checkpoint read while probing adaptive top2; deprioritized because R1Q5 already proved fallback compression is not enough |
| S5-R1Q7 | export-only | | | **1.09951341** | | **15,925,658** | current valid best; real external `lrzip` recovers ~156KB and makes embed7/top3 legal |
| S5-R1Q7T | eval-only | | | 1.09951341 | **1.08464351** | **15,925,658** | current best valid Phase 5 result; legal lrzip artifact plus 3-phase TTT |
| S5-R1Q3T | eval-only | | | 1.10542246 | 1.09020482 | 15,557,072 | embed6 TTT is valid and strong, but worse than embed7/top3 TTT signal |
| S5-T3 | eval-only | | | 1.09951341 | 1.08462841 | 15,925,658 | prefix docs 2000 / 3 phases improves slightly over default, but worse than T5 |
| S5-T4 | eval-only | | | 1.09951341 | 1.08466509 | 15,925,658 | prefix docs 3000 / 3 phases is worse |
| S5-T5 | eval-only | | | 1.09951341 | **1.08461728** | 15,925,658 | new best valid Phase 5 result; 4 phases beats 2/3 phases by a tiny margin |
| S5-T6 | eval-only | | | 1.09951341 | 1.08469031 | 15,925,658 | 2 phases is worse; do not continue lower phase count |
| S5-T7 | eval-only | | | 1.09951341 | 1.08466617 | 15,925,658 | lower global lr 0.0007 is worse |
| S5-T8 | eval-only | | | 1.09951341 | 1.08462086 | 15,925,658 | higher global lr 0.0013 helps versus default 3-phase, close to T5 |
| S5-R2 | 3188 | ~704k tok/s final | 1.08868633 | 1.09686294 | **1.08218120** | 15,925,323 | new best Phase 5 result; seed42 improves no-TTT by `0.00265 BPB` over R1Q7 |
| S5-R3 | 3181 | ~702k tok/s final | 1.08968277 | 1.09803468 | 1.08311378 | 15,923,614 | valid improvement over R1Q7 but worse than seed42 |
| S5-T9 | eval-only | | | 1.09951341 | 1.08460962 | 15,925,658 | shorter prefix plus 4 phases helps, but less than higher lr |
| S5-T10 | eval-only | | | 1.09951341 | 1.08461694 | 15,925,658 | 5 phases does not improve over 4 phases |
| S5-T11 | eval-only | | | 1.09951341 | 1.08459633 | 15,925,658 | 4 phases plus lr 0.0013 improves slightly |
| S5-T12 | eval-only | | | 1.09951341 | **1.08458960** | 15,925,658 | current best valid Phase 5 result; improvement remains only `~5e-5 BPB` over R1Q7T |
| S5-T13 | eval-only | | | 1.09951341 | 1.08461777 | 15,925,658 | prefix docs 2250 is essentially tied with T5 |
| S5-T14 | eval-only | | | 1.09951341 | 1.08463606 | 15,925,658 | prefix docs 2750 is worse |
| S5-R4 | 3339 | ~751k tok/s final | 1.09042535 | 1.09860931 | 1.08374669 | 15,923,435 | loop 0.50 gives more steps than default but worse quality than R2/R8 |
| S5-R5 | 3601 | ~794k tok/s final | 1.08965953 | 1.09811073 | 1.08320086 | 15,920,845 | loop 0.65 improves throughput but still loses to default seed42 |
| S5-R6 | 3902 | ~882k tok/s pre-loop | 1.11167701 | 1.12131178 | 1.10134207 | 15,922,988 | loop 0.90 collapses quality; do not continue late/no-loop alone |
| S5-R7 | 2759 | ~716k tok/s final | 1.09222282 | 1.10034453 | 1.08560789 | 15,927,679 | batch 917k is worse in records stack |
| S5-R8 | 3202 | ~708k tok/s final | **1.08798009** | **1.09647097** | **1.08179851** | 15,924,510 | current best valid Phase 5 result; later warmdown is a real positive signal |
| S5-R9 | 3183 | ~703k tok/s final | 1.08976095 | 1.09812598 | 1.08336520 | 15,925,854 | seed1234 roughly matches seed0; seed42 remains best |
| S5-R2T12 | eval-only | | | 1.09686294 | **1.08217886** | 15,925,323 | current best valid Phase 5 result; T12 setting transfers but only by `0.00000234 BPB` over R2 |
| S5-R3T12 | eval-only | | | 1.09803468 | 1.08306801 | 15,923,614 | improves R3 by `0.00004577 BPB`, still worse than R2 |
| S5-R10 | 3808 | ~862k tok/s near loop | 1.09405982 | 1.10276361 | 1.08734365 | 15,920,673 | loop 0.80 is much worse despite more steps; stop delayed-loop branch |
| S5-R11 | 3931 | ~888k tok/s pre-loop | 1.11314137 | 1.12342617 | 1.10325317 | 15,922,541 | loop 0.90 + warmdown095 collapses; stop late-loop interactions |
| S5-R8T12 | eval-only | | | 1.09647097 | 1.08180270 | 15,924,510 | T12 TTT setting is slightly worse than R8 default; keep R8 as best |
| S5-R12 | 3191 | ~705k tok/s final | 1.08810526 | 1.09652668 | 1.08186396 | 15,923,320 | warmdown 0.90 is close but worse than R8 0.95 |
| S5-R13 | 3160 | ~697k tok/s final | 1.08842136 | 1.09696300 | 1.08225069 | 15,924,424 | warmdown 0.98 is worse; 0.95 remains local best |
| S5-R14 | 3198 | ~707k tok/s final | 1.08969782 | 1.09845654 | 1.08363383 | 15,923,476 | seed0 does not benefit enough from warmdown095 to beat seed42 |
| S5-R15 | 3200 | ~707k tok/s final | 1.08963697 | 1.09833407 | 1.08366228 | 15,930,493 | seed1234 with warmdown095 also worse than seed42 |
| S5-R16 | 3044 | ~666k tok/s final | 1.08911578 | 1.09762581 | 1.08296624 | 15,927,702 | early loop 0.25 with warmdown095 is worse than R8 |
| S5-R17 | 3147 | ~693k tok/s final | 1.09122964 | 1.09881552 | | 15,923,313 | min_lr 0.2 is negative and later TTT crashed; no need to rerun |
| S5-R18 | 3185 | ~703k tok/s final | 1.08867969 | 1.09689729 | 1.08217238 | 15,925,313 | min_lr 0.15 is close but worse than R8 |
| S5-R19 | 3180 | ~702k tok/s final | 1.09062094 | 1.09838193 | 1.08343385 | 15,921,752 | warmdown098 + min_lr0.2 is negative |
| S5-R20 | 3150 | ~694k tok/s final | 1.08903288 | 1.09769210 | 1.08289760 | 15,927,912 | warmdown 0.94 is worse than R8 0.95 |
| S5-R8Q1 | export-only | | | 1.09734050 | | 15,926,021 | err0.95 is worse than R8 quantization |
| S5-R8Q2 | export-only | | | 1.09722868 | | 15,925,437 | calib32 is worse |
| S5-R8Q3 | export-only | | | 1.09726753 | | 15,924,537 | calib64 is worse |
| S5-R8Q4 | export-only | | | 1.09734050 | | 15,926,021 | err1.05 equals the worse err path |
| S5-R8Q5 | export-only | | | 1.09733811 | | 15,928,275 | top4 is worse and larger |

## Interim Phase 5 Conclusions

- CaseOps data/sidecar path is now technically valid: full 80-shard dataset exists, sidecar length/BOS checks pass, and root logs use `val_bpb:byte_sidecar:enabled`.
- First true root CaseOps runs did not improve the Phase 4 best. Best completed CaseOps root result is S5-C5 at `1.17380957`, versus Phase 4 QS28 `1.16522320`.
- The negative CaseOps signal is not yet enough to reject CaseOps globally, because the records 04-27 stack uses doc-boundary/TTT/structure pieces that root C4/C5 do not reproduce.
- Do not continue broad CaseOps root sweeps. Only run targeted diagnostics: schedule mismatch, records stack/fallback result, and true doc-boundary/TTT.
- Empty GPUs should prioritize ordinary SP8192 QS28 training-dynamic tests and legal TTT implementation rather than more export-only QS28 micro-sweeps.
- S5-O2 was the first Phase 5 improvement over Phase 4: ordinary SP8192 QS28 with `TRAIN_BATCH_TOKENS=786432` reached `1.16474115`, and S5-Q1 improved that checkpoint to `1.16465057`. The follow-up bracket peaked at S5-O8 `1.16357912`; this is a real but small root-line gain, now secondary to records+lrzip+TTT.
- S5-O4 shows the largest quality signal so far: MLP3 + int5 reached roundtrip `1.15317521`, but total bytes are `16,806,331`, over the 16MB cap. This is not a valid submission result yet; it is a capacity-repair target.
- O4 capacity repair has not yet produced a valid improvement: P3 no-LQER remains over cap, while P4 fits but degrades to `1.16583102`, worse than valid O5. Do not spend more GPU on O4 export-only bit fiddling unless adding a materially better compressor/packing path.
- The records 04-27 fallback path is now the primary Phase 5 line. Adaptive brotli/lzma fallback did not shrink the R1Q2 artifact, but installing and using real external `lrzip` made the same embed7/top3 weights legal: R1Q7 reached `1.09951341` / `15,925,658`. R1Q7T then validated the legal artifact with 3-phase TTT at `1.08464351`, the current valid Phase 5 best.
- Compared with the 04-27 record (`~1.073` post-quant, `~1.061` post-TTT), the remaining gap is mostly training quality/throughput rather than quantization or TTT mechanics: R1Q7T gains about `0.0149 BPB` from TTT, similar to records, but starts from a much worse no-TTT BPB. T3-T14 found only tiny TTT improvements, with T12 `2500 docs / 4 phases / lr 0.0015` reaching `1.08458960`; stop TTT micro-sweeps unless a better trained artifact appears. R2/R3/R9 now cover record seeds, and R4-R8 test single-card training dynamics: delayed loop, larger batch, and later warmdown.
- R2 confirms seed/training variance is meaningful under the 1xH100 fallback: seed42 improves no-TTT from R1Q7 `1.09951341` to `1.09686294`, then default TTT reaches `1.08218120`. R2T12 inches this to `1.08217886`, the current valid best and close to the 2026-04-06 record (`1.08279`), but still far from 2026-04-27 mean (`~1.061`). The TTT transfer gain is negligible, so the live branch remains training quality: R4-R11 test delayed loop, later warmdown, and batch effects.
- R8 supersedes R2: `WARMDOWN_FRAC=0.95` reaches no-TTT `1.09647097` and post-TTT `1.08179851`, the current valid best. Loop delay by itself is not the answer: R5/R4 are worse and R6 collapses, despite more steps. Batch 917k is also worse. The live branch is now a narrow warmdown line: R8T12 checks TTT transfer on R8, while R12/R13 sweep warmdown 0.90/0.98 and R14/R15 test warmdown095 across seeds.
- R10/R11 close the delayed-loop branch: both are much worse, so more steps via late loop do not translate to better BPB. R17/R18/R19 show higher `MIN_LR` is not the mechanism behind R8; R20 shows warmdown 0.94 is worse than 0.95. R8Q1-Q5 export-only probes also fail to beat the original R8 quantization. Phase 5 best remains R8 `1.08179851`.

## Records Fallback Issue

S5-R0 trained for 3000 steps under the records 04-27 stack but failed at the final prequant eval. Root cause:

```text
torch._dynamo.exc.UserError: Could not guard on data-dependent expression u0 < 0
Caused by: outputs.append(flash_attn_3_func(q[start:end][None], ...))
```

This is an environment/fallback issue, not a model-quality result: `flash_attn_interface` is unavailable, so the patched varlen attention fallback loops over `cu_seqlens` slices; records eval compiled `forward_logits`, and Dynamo cannot prove those dynamic slice bounds. The records script now uses eager eval logits when FA3 is unavailable while preserving the compiled path when FA3 exists.

S5-R0b confirmed the Dynamo fix but exposed the next fallback issue: records globally enables flash-only SDPA, and some varlen fallback segments have no available flash kernel. S5-R0c put a local SDPA backend override in the generic fallback, but that path is also used by compiled fixed-seq training and Dynamo cannot trace the context manager. The fallback is now split: fixed-seq fallback remains compile-friendly, while varlen eager fallback locally permits flash, math, and memory-efficient SDPA backends. S5-R0d verified the fix with a 90-second `PREQUANT_ONLY=1` run and reached `diagnostic pre-quantization post-ema` without crashing.

## Next Decision Rules

- If CaseOps root runs are worse than Q43/QS28, first audit tokenizer/sidecar/eval alignment before rejecting CaseOps.
- If records S5-R0 is much better despite fallback, prioritize porting its structure/TTT pieces into root.
- If records S5-R0 is bottlenecked by fallback speed, treat FA3/varlen as a real blocker for near-1.1 and plan kernel installation or a root-native doc-boundary path.
