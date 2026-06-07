# 第三阶段实验状态

日期：2026-06-06

约束：1xH100 / 1h 训练，提交物 <= 16 MB。默认基座为 B3 Seq4096 + GPTQ int6/embed8 + LQER rank4 top3。

## 当前最佳

| ID | RUN_ID | 改动 | Pre BPB | Roundtrip BPB | 总字节 | 结论 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| S3-H4 | `exp_s3_h4_b3_qkgain5_1xh100` | `QK_GAIN_INIT=5.0` | 1.1743 | 1.17610282 | 15,754,272 | 第一批正收益，已被 H7 超过 |
| S3-H7 | `exp_s3_h7_b3_qkgain5_hparam_stack_1xh100` | `QK_GAIN_INIT=5.0` + H1 hparam stack | 1.1716 | 1.17479819 | 15,750,114 | 明显优于 H4，已被 R1 超过 |
| S3-R1 | `exp_s3_r1_b3_qkgain5_recur_l3_5_start035_1xh100` | `QK_GAIN_INIT=5.0` + recurrence L3-L5 start 0.35 | 1.1728 | **1.17445595** | 15,735,360 | 新最佳，recurrence 有正信号 |
| S3-Q7e | `exp_s3_q7e_h7_lqer_top4_rank4_eval` | H7 ckpt, top4 rank4 artifact 补评 | n/a | **1.17377474** | 15,753,342 | 新最佳，H7 量化细扫正收益 |
| S3-H11 | `exp_s3_h11_b3_qkgain5_hparam_tiedembedlr004_1xh100` | H7 stack + `TIED_EMBED_LR=0.04` | 1.1699 | **1.17163042** | 15,745,727 | 新最佳，显著优于 Q7e |
| S3-R4 | `exp_s3_r4_b3_qkgain5_hparam_tied004_recur_l3_5_start035_1xh100` | H11 idea + recurrence L3-L5 start 0.35 | 1.1679 | **1.17006027** | 15,736,208 | 当前最佳，递归与 tied embed LR 可叠加 |
| S3-Q17 | `exp_s3_q17_r4_clip_top4_rank4_export` | R4 ckpt, recurrence-active export, top4 rank4 | n/a | **1.16964365** | 15,737,085 | 当前最佳，R4 量化细扫正收益 |
| S3-Q20 | `exp_s3_q20_r4_clip_top4_rank8_export` | R4 ckpt, recurrence-active export, top4 rank8 | n/a | **1.16964352** | 15,741,769 | 当前最佳，较 Q17 仅微小提升 |
| S3-Q24 | `exp_s3_q24_r4_clip_top4_rank8_calib32_export` | R4 ckpt, recurrence-active export, top4 rank8, 32 calibration batches | n/a | **1.16955557** | 15,741,500 | 当前最佳，calibration 扩大有效 |
| S3-Q26 | `exp_s3_q26_r4_clip_top4_rank8_calib64_export` | R4 ckpt, recurrence-active export, top4 rank8, 64 calibration batches | n/a | **1.16955543** | 15,741,429 | 当前最佳，较 Q24 仅微小提升 |
| S3-Q30 | `exp_s3_q30_r6_clip_top4_rank4_calib32_export` | R6 ckpt, recurrence-active export, top4 rank4, 32 calibration batches | n/a | **1.16942948** | 15,749,135 | R6 细扫刷新，但已被 R8 超过 |
| S3-R8 | `exp_s3_r8_b3_qkgain5_hparam_tied004_muon097_recur_l3_5_start035_1xh100` | H11 idea + `MUON_MOMENTUM=0.97` + recurrence L3-L5 start 0.35 | 1.1666 | **1.16835283** | 15,752,498 | 当前最佳，muon + recurrence 可叠加 |
| S3-R15 | `exp_s3_r15_b3_qkgain5_hparam_tied004_muon097_recur_l3_5_start030_1xh100` | H11 idea + `MUON_MOMENTUM=0.97` + recurrence L3-L5 start 0.30 | 1.1657 | **1.16744173** | 15,754,050 | 当前最佳，start 0.30 保住更多 pre 收益 |
| S3-Q43 | `exp_s3_q43_r15_clip_top4_rank4_export` | R15 ckpt, recurrence-active export, top4 rank4 | n/a | **1.16735413** | 15,753,494 | 当前最佳，R15 上 top4 rank4 有小收益 |

## 已完成批次

| ID | RUN_ID | 类型 | 关键配置 | Pre BPB | Roundtrip BPB | 总字节 | 判断 |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| S3-H1 | `exp_s3_h1_b3_hparam_stack_1xh100` | 1h train | `BETA2=0.99 GRAD_CLIP_NORM=0.3 MLP_CLIP_SIGMAS=12 EMBED_CLIP_SIGMAS=15` | 1.1712 | 1.17983113 | 15,749,175 | 量化后负收益 |
| S3-H2 | `exp_s3_h2_b3_warmfrac085_minlr010_1xh100` | 1h train | `WARMDOWN_FRAC=0.85 MIN_LR=0.10` | 1.1798 | 1.18171780 | 15,693,951 | 负收益 |
| S3-H3 | `exp_s3_h3_b3_hparam_warmfrac_1xh100` | 1h train | H1 + H2 | 1.1760 | 1.17874598 | 15,694,384 | 负收益 |
| S3-H4 | `exp_s3_h4_b3_qkgain5_1xh100` | 1h train | `QK_GAIN_INIT=5.0` | 1.1743 | 1.17610282 | 15,754,272 | 正收益 |
| S3-Q1 | `exp_s3_q1_b3_lqer_top4_rank4_err1_export` | export-only | B3 ckpt, `LQER_TOP_K=4 LQER_RANK=4` | n/a | 1.17676263 | 15,742,382 | 未优于 baseline |
| S3-Q2 | `exp_s3_q2_b3_lqer_top5_rank4_err125_export` | export-only | B3 ckpt, top5 rank4 `GPTQ_ERROR_SCALE=1.25` | n/a | 1.17841072 | 15,744,787 | 负收益 |
| S3-Q3 | `exp_s3_q3_b3_lqer_top3_rank8_err1_export` | export-only | B3 ckpt, top3 rank8 | n/a | 1.17677578 | 15,743,441 | 未优于 baseline |
| S3-Q4 | `exp_s3_q4_h4_lqer_top4_rank4_export` | export-only | H4 ckpt, top4 rank4 | n/a | 1.17609196 | 15,755,654 | 微弱优于 H4 |
| S3-Q5 | `exp_s3_q5_h4_lqer_top3_rank8_export` | export-only | H4 ckpt, top3 rank8 | n/a | 1.17609259 | 15,757,785 | 微弱优于 H4 |
| S3-Q6 | `exp_s3_q6_h4_clipretune_export` | export-only | H4 ckpt, `MLP_CLIP_SIGMAS=12 EMBED_CLIP_SIGMAS=15` | n/a | 1.17609257 | 15,754,562 | 微弱优于 H4 |
| S3-H5 | `exp_s3_h5_b3_qkgain4_1xh100` | 1h train | `QK_GAIN_INIT=4.0` | 1.1742 | 1.17606936 | 15,751,352 | 微弱优于 H4，不如 H7 |
| S3-H6 | `exp_s3_h6_b3_qkgain6_1xh100` | 1h train | `QK_GAIN_INIT=6.0` | 1.1741 | 1.17599881 | 15,745,831 | 微弱优于 H4，不如 H7 |
| S3-H7 | `exp_s3_h7_b3_qkgain5_hparam_stack_1xh100` | 1h train | QK5 + H1 hparam stack | 1.1716 | 1.17479819 | 15,750,114 | 新最佳 |
| S3-H8 | `exp_s3_h8_b3_qkgain5_warmfrac_1xh100` | 1h train | QK5 + warmdown/min-lr | 1.1795 | 1.18307589 | 15,701,191 | 负收益 |
| S3-L1 | `exp_s3_l1_b3_coprime_1xh100` | 1h train | `LOADER_MODE=coprime` | 1.1847 | 1.18650699 | 15,688,528 | 明显负收益，当前 coprime loader 淘汰 |
| S3-L2 | `exp_s3_l2_b3_qkgain5_coprime_1xh100` | 1h train | QK5 + `LOADER_MODE=coprime` | 1.1846 | 1.18634911 | 15,698,473 | 明显负收益，不继续扩展 coprime |
| S3-R1 | `exp_s3_r1_b3_qkgain5_recur_l3_5_start035_1xh100` | 1h train | QK5 + recurrence L3-L5 start 0.35 | 1.1728 | 1.17445595 | 15,735,360 | 新最佳，继续 recurrence sweep |
| S3-Q7 | `exp_s3_q7_h7_lqer_top4_rank4_export` | export-only | H7 ckpt, top4 rank4 | n/a | n/a | n/a | 生成 `ptz` 后无 roundtrip 输出，记为异常待复跑/重评 |
| S3-Q8 | `exp_s3_q8_h7_clipretune_export` | export-only | H7 ckpt, clip retune | n/a | n/a | n/a | 生成 `ptz` 后无 roundtrip 输出，记为异常待复跑/重评 |
| S3-H9 | `exp_s3_h9_b3_qkgain5_hparam_matrixlr026_1xh100` | 1h train | H7 stack + `MATRIX_LR=0.026` | 1.1727 | 1.17600320 | 15,674,282 | 正收益但不如 R1 |
| S3-H10 | `exp_s3_h10_b3_qkgain5_hparam_matrixlr03_1xh100` | 1h train | H7 stack + `MATRIX_LR=0.03` | 1.1715 | 1.17551445 | 15,714,585 | 正收益但不如 R1 |
| S3-Q7e | `exp_s3_q7e_h7_lqer_top4_rank4_eval` | eval-only | H7 ckpt, top4 rank4 artifact | n/a | 1.17377474 | 15,753,342 | 新最佳 |
| S3-Q8e | `exp_s3_q8e_h7_clipretune_eval` | eval-only | H7 ckpt, clip retune artifact | n/a | 1.17378505 | 15,751,596 | 与 Q7e 基本持平 |
| S3-Q9 | `exp_s3_q9_h10_lqer_top4_rank4_export` | export-only | H10 ckpt, top4 rank4 | n/a | 1.17446827 | 15,716,463 | 正收益，不如 Q7e |
| S3-Q10 | `exp_s3_q10_h7_lqer_top5_rank4_export` | export-only | H7 ckpt, top5 rank4 | n/a | 1.17377726 | 15,755,583 | 几乎持平 Q7e，略差 |
| S3-R2 | `exp_s3_r2_b3_qkgain5_hparam_recur_l3_5_start035_1xh100` | 1h train | H7 stack + recurrence start 0.35 | 1.1709 | 1.17404803 | 15,724,849 | 正收益，不如 H11 |
| S3-H11 | `exp_s3_h11_b3_qkgain5_hparam_tiedembedlr004_1xh100` | 1h train | H7 stack + `TIED_EMBED_LR=0.04` | 1.1699 | 1.17163042 | 15,745,727 | 新最佳 |
| S3-T1 | `exp_s3_t1_q7e_lmhead_lora_full` | eval-only TTT | Q7 artifact lm-head LoRA score-first | n/a | 1.61284870 | n/a | 明显负收益，淘汰该 MVP |
| S3-Q11 | `exp_s3_q11_h11_lqer_top4_rank4_export` | export-only | H11 ckpt, top4 rank4 | n/a | 1.17166463 | 15,747,570 | 几乎持平 H11，略差 |
| S3-R3 | `exp_s3_r3_b3_qkgain5_hparam_recur_l3_5_start025_1xh100` | 1h train | H7 stack + recurrence start 0.25 | 1.1710 | 1.17296940 | 15,720,724 | 正收益，不如 H11 |
| S3-H12 | `exp_s3_h12_b3_qkgain5_hparam_tiedembedlr006_1xh100` | 1h train | H7 stack + `TIED_EMBED_LR=0.06` | 1.1727 | 1.17588022 | 15,753,401 | 负于 H11 |
| S3-H13 | `exp_s3_h13_b3_qkgain5_hparam_matrixlr035_1xh100` | 1h train | H7 stack + `MATRIX_LR=0.035` | 1.1726 | 1.17668257 | 15,746,040 | 负收益 |
| S3-H14 | `exp_s3_h14_b3_qkgain5_hparam_muonmom097_1xh100` | 1h train | H7 stack + `MUON_MOMENTUM=0.97` | 1.1702 | 1.17292917 | 15,765,373 | 正收益，不如 H11 |
| S3-H15 | `exp_s3_h15_b3_qkgain5_hparam_tied004_matrixlr03_1xh100` | 1h train | H11 idea + `MATRIX_LR=0.03` | 1.1706 | 1.17252850 | 15,699,434 | 正收益，不如 H11 |
| S3-H16 | `exp_s3_h16_b3_qkgain5_hparam_tiedembedlr003_1xh100` | 1h train | H7 stack + `TIED_EMBED_LR=0.03` | 1.1701 | 1.17377910 | 15,777,801 | 量化后不如 H11 |
| S3-Q12 | `exp_s3_q12_h11_err075_lqer_top3_export` | export-only | H11 ckpt, `GPTQ_ERROR_SCALE=0.75` | n/a | 1.17279557 | 15,746,516 | 负于 H11 |
| S3-Q13 | `exp_s3_q13_h11_err125_lqer_top3_export` | export-only | H11 ckpt, `GPTQ_ERROR_SCALE=1.25` | n/a | 1.17217624 | 15,747,736 | 负于 H11 |
| S3-H17 | `exp_s3_h17_b3_qkgain5_hparam_tiedembedlr0035_1xh100` | 1h train | H7 stack + `TIED_EMBED_LR=0.035` | 1.1718 | 1.17813295 | 15,752,100 | 负于 H11 |
| S3-H18 | `exp_s3_h18_b3_qkgain5_hparam_tiedembedlr0045_1xh100` | 1h train | H7 stack + `TIED_EMBED_LR=0.045` | 1.1700 | 1.17519083 | 15,738,154 | 负于 H11 |
| S3-R4 | `exp_s3_r4_b3_qkgain5_hparam_tied004_recur_l3_5_start035_1xh100` | 1h train | H11 idea + recurrence L3-L5 start 0.35 | 1.1679 | 1.17006027 | 15,736,208 | 当前最佳 |
| S3-H19 | `exp_s3_h19_b3_qkgain5_hparam_tied004_matrixlr026_1xh100` | 1h train | H11 idea + `MATRIX_LR=0.026` | 1.1698 | 1.17223982 | 15,655,308 | 正收益，不如 H11 |
| S3-Q14 | `exp_s3_q14_h11_clip_top4_rank4_export` | export-only | H11 ckpt, clip stack + top4 rank4 | n/a | 1.17166463 | 15,747,570 | 基本复现 Q11，略差于 H11 |
| S3-Q15 | `exp_s3_q15_h11_clip_top5_rank4_export` | export-only | H11 ckpt, clip stack + top5 rank4 | n/a | 1.17166530 | 15,749,314 | 与 Q14 持平，略差于 H11 |
| S3-Q16 | `exp_s3_q16_h11_clip_err125_top3_export` | export-only | H11 ckpt, clip stack + `GPTQ_ERROR_SCALE=1.25` | n/a | 1.17217624 | 15,747,736 | 负于 H11 |
| S3-Q17 | `exp_s3_q17_r4_clip_top4_rank4_export` | export-only | R4 ckpt, recurrence-active export, top4 rank4 | n/a | 1.16964365 | 15,737,085 | 新最佳 |
| S3-Q18 | `exp_s3_q18_r4_clip_top3_err125_export` | export-only | R4 ckpt, recurrence-active export, `GPTQ_ERROR_SCALE=1.25` | n/a | 1.17058656 | 15,735,290 | 负于 R4/Q17 |
| S3-Q19 | `exp_s3_q19_r4_clip_top5_rank4_export` | export-only | R4 ckpt, recurrence-active export, top5 rank4 | n/a | 1.16964491 | 15,738,543 | 与 Q17 持平，略差且更大 |
| S3-Q20 | `exp_s3_q20_r4_clip_top4_rank8_export` | export-only | R4 ckpt, recurrence-active export, top4 rank8 | n/a | 1.16964352 | 15,741,769 | 当前最佳，但只微幅优于 Q17 |
| S3-Q21 | `exp_s3_q21_r4_clip_top6_rank4_export` | export-only | R4 ckpt, recurrence-active export, top6 rank4 | n/a | 1.16964628 | 15,739,442 | 与 Q17/Q20 持平，略差 |
| S3-Q22 | `exp_s3_q22_r4_clip_top4_err075_export` | export-only | R4 ckpt, recurrence-active export, top4 rank4, `GPTQ_ERROR_SCALE=0.75` | n/a | 1.17009150 | 15,736,573 | 负于 Q17/Q20 |
| S3-H20 | `exp_s3_h20_b3_qkgain4_hparam_tied004_1xh100` | 1h train | QK4 + H11 idea | 1.1702 | 1.17215628 | 15,747,280 | 负于 H11/R4 |
| S3-H21 | `exp_s3_h21_b3_qkgain6_hparam_tied004_1xh100` | 1h train | QK6 + H11 idea | 1.1698 | 1.17220424 | 15,733,848 | 负于 H11/R4 |
| S3-Q23 | `exp_s3_q23_r4_clip_top4_rank4_calib32_export` | export-only | R4 ckpt, recurrence-active export, top4 rank4, 32 calibration batches | n/a | 1.16955601 | 15,738,579 | 新最佳，略差于 Q24 |
| S3-Q24 | `exp_s3_q24_r4_clip_top4_rank8_calib32_export` | export-only | R4 ckpt, recurrence-active export, top4 rank8, 32 calibration batches | n/a | 1.16955557 | 15,741,500 | 当前最佳 |
| S3-Q25 | `exp_s3_q25_r4_clip_top4_rank4_damp005_export` | export-only | R4 ckpt, recurrence-active export, top4 rank4, `GPTQ_DAMP=0.005` | n/a | 1.17113668 | 15,740,348 | 明显负收益 |
| S3-R5 | `exp_s3_r5_b3_qkgain5_hparam_tied004_recur_l3_5_start025_1xh100` | 1h train | H11 idea + recurrence L3-L5 start 0.25 | 1.1688 | 1.17055401 | 15,738,914 | 正收益，不如 Q24/Q26 |
| S3-R6 | `exp_s3_r6_b3_qkgain5_hparam_tied004_recur_l3_5_start050_1xh100` | 1h train | H11 idea + recurrence L3-L5 start 0.50 | 1.1677 | 1.16957564 | 15,747,462 | 接近当前最佳，继续量化细扫 |
| S3-R7 | `exp_s3_r7_b3_qkgain5_hparam_tied004_recur_l4_5_start035_1xh100` | 1h train | H11 idea + recurrence L4-L5 start 0.35 | 1.1701 | 1.17295937 | 15,740,811 | 负收益 |
| S3-Q26 | `exp_s3_q26_r4_clip_top4_rank8_calib64_export` | export-only | R4 ckpt, recurrence-active export, top4 rank8, 64 calibration batches | n/a | 1.16955543 | 15,741,429 | 当前最佳，64 calibration 微幅优于 32 |
| S3-Q27 | `exp_s3_q27_r4_clip_top4_rank4_calib64_export` | export-only | R4 ckpt, recurrence-active export, top4 rank4, 64 calibration batches | n/a | 1.16955647 | 15,738,261 | 略差于 Q26 |
| S3-Q28 | `exp_s3_q28_r4_clip_top5_rank8_calib32_export` | export-only | R4 ckpt, recurrence-active export, top5 rank8, 32 calibration batches | n/a | 1.16955627 | 15,744,193 | 略差于 Q26 |
| S3-Q29 | `exp_s3_q29_r6_clip_top4_rank8_calib32_export` | export-only | R6 ckpt, recurrence-active export, top4 rank8, 32 calibration batches | n/a | 1.16943265 | 15,753,320 | 刷新 R6 默认，不如 R8 |
| S3-Q30 | `exp_s3_q30_r6_clip_top4_rank4_calib32_export` | export-only | R6 ckpt, recurrence-active export, top4 rank4, 32 calibration batches | n/a | 1.16942948 | 15,749,135 | R6 最佳，不如 R8 |
| S3-Q31 | `exp_s3_q31_r6_clip_top4_rank8_calib64_export` | export-only | R6 ckpt, recurrence-active export, top4 rank8, 64 calibration batches | n/a | 1.16943292 | 15,753,206 | 不如 Q30 |
| S3-Q32 | `exp_s3_q32_r6_clip_top5_rank8_calib32_export` | export-only | R6 ckpt, recurrence-active export, top5 rank8, 32 calibration batches | n/a | 1.16943057 | 15,756,171 | 不如 Q30 且更大 |
| S3-R8 | `exp_s3_r8_b3_qkgain5_hparam_tied004_muon097_recur_l3_5_start035_1xh100` | 1h train | H11 idea + `MUON_MOMENTUM=0.97` + recurrence L3-L5 start 0.35 | 1.1666 | 1.16835283 | 15,752,498 | 当前最佳，继续量化细扫 |
| S3-R9 | `exp_s3_r9_b3_qkgain5_hparam_tied004_recur_l3_6_start035_1xh100` | 1h train | H11 idea + recurrence L3-L6 start 0.35 | 1.1709 | 1.17466984 | 15,743,186 | 明显负收益 |
| S3-Q33 | `exp_s3_q33_r8_clip_top4_rank8_calib32_export` | export-only | R8 ckpt, top4 rank8, 32 calibration batches | n/a | 1.16849556 | 15,758,037 | 负于 R8 默认 |
| S3-Q34 | `exp_s3_q34_r8_clip_top4_rank4_calib32_export` | export-only | R8 ckpt, top4 rank4, 32 calibration batches | n/a | 1.16849556 | 15,753,513 | 负于 R8 默认 |
| S3-Q35 | `exp_s3_q35_r8_clip_top4_rank8_calib64_export` | export-only | R8 ckpt, top4 rank8, 64 calibration batches | n/a | 1.16846709 | 15,758,914 | 负于 R8 默认 |
| S3-Q36 | `exp_s3_q36_r8_clip_top5_rank8_calib32_export` | export-only | R8 ckpt, top5 rank8, 32 calibration batches | n/a | 1.16849460 | 15,760,370 | 负于 R8 默认且更大 |
| S3-Q37 | `exp_s3_q37_r8_clip_top3_rank4_calib32_export` | export-only | R8 ckpt, top3 rank4, 32 calibration batches | n/a | 1.16849321 | 15,752,248 | 负于 R8 默认 |
| S3-Q38 | `exp_s3_q38_r8_clip_top3_rank4_calib64_export` | export-only | R8 ckpt, top3 rank4, 64 calibration batches | n/a | 1.16846951 | 15,752,575 | 负于 R8 默认 |
| S3-Q39 | `exp_s3_q39_r8_clip_top3_rank4_err075_export` | export-only | R8 ckpt, top3 rank4, `GPTQ_ERROR_SCALE=0.75` | n/a | 1.16935322 | 15,752,606 | 负收益 |
| S3-Q40 | `exp_s3_q40_r8_clip_top3_rank4_err125_export` | export-only | R8 ckpt, top3 rank4, `GPTQ_ERROR_SCALE=1.25` | n/a | 1.16867803 | 15,753,565 | 负于 R8 默认 |
| S3-R10 | `exp_s3_r10_b3_qkgain5_hparam_tied004_recur_l3_5_start045_1xh100` | 1h train | H11 idea + recurrence L3-L5 start 0.45 | 1.1692 | 1.17266607 | 15,739,711 | 负收益 |
| S3-R11 | `exp_s3_r11_b3_qkgain5_hparam_tied004_recur_l3_5_start060_1xh100` | 1h train | H11 idea + recurrence L3-L5 start 0.60 | 1.1688 | 1.17057736 | 15,746,217 | 不如 R8 |
| S3-R12 | `exp_s3_r12_b3_qkgain5_hparam_tied004_muon097_recur_l3_5_start045_1xh100` | 1h train | H11 idea + `MUON_MOMENTUM=0.97` + recurrence L3-L5 start 0.45 | 1.1668 | 1.16918619 | 15,745,861 | pre 接近 R8，量化后不如 |
| S3-R13 | `exp_s3_r13_b3_qkgain5_hparam_tied004_muon097_recur_l3_5_start050_1xh100` | 1h train | H11 idea + `MUON_MOMENTUM=0.97` + recurrence L3-L5 start 0.50 | 1.1674 | 1.16917815 | 15,742,644 | pre/量化后均不如 R8 |
| S3-R14 | `exp_s3_r14_b3_qkgain5_hparam_tied004_muon097_recur_l3_5_start025_1xh100` | 1h train | H11 idea + `MUON_MOMENTUM=0.97` + recurrence L3-L5 start 0.25 | 1.1671 | 1.17104210 | 15,744,983 | 量化后负于 R8/R15 |
| S3-R15 | `exp_s3_r15_b3_qkgain5_hparam_tied004_muon097_recur_l3_5_start030_1xh100` | 1h train | H11 idea + `MUON_MOMENTUM=0.97` + recurrence L3-L5 start 0.30 | 1.1657 | 1.16744173 | 15,754,050 | 当前最佳，继续量化细扫 |
| S3-R16 | `exp_s3_r16_b3_qkgain5_hparam_tied004_muon097_recur_l3_5_start040_1xh100` | 1h train | H11 idea + `MUON_MOMENTUM=0.97` + recurrence L3-L5 start 0.40 | 1.1662 | 1.16798998 | 15,750,745 | 正收益，不如 R15 |
| S3-R17 | `exp_s3_r17_b3_qkgain5_hparam_tied004_muon097_recur_l3_5_start060_1xh100` | 1h train | H11 idea + `MUON_MOMENTUM=0.97` + recurrence L3-L5 start 0.60 | 1.1668 | 1.16863127 | 15,756,871 | 正收益，不如 R15 |
| S3-R18 | `exp_s3_r18_b3_qkgain5_hparam_tied0035_muon097_recur_l3_5_start035_1xh100` | 1h train | `TIED_EMBED_LR=0.035 MUON_MOMENTUM=0.97` + recurrence L3-L5 start 0.35 | 1.1685 | 1.17145238 | 15,765,592 | 负于 R15 |
| S3-R19 | `exp_s3_r19_b3_qkgain5_hparam_tied0045_muon097_recur_l3_5_start035_1xh100` | 1h train | `TIED_EMBED_LR=0.045 MUON_MOMENTUM=0.97` + recurrence L3-L5 start 0.35 | 1.1670 | 1.16904270 | 15,730,644 | 不如 R15 |
| S3-R20 | `exp_s3_r20_b3_qkgain5_hparam_tied004_muon096_recur_l3_5_start035_1xh100` | 1h train | H11 idea + `MUON_MOMENTUM=0.96` + recurrence L3-L5 start 0.35 | 1.1674 | 1.17050827 | 15,757,430 | 负于 R15 |
| S3-R21 | `exp_s3_r21_b3_qkgain5_hparam_tied004_muon098_recur_l3_5_start035_1xh100` | 1h train | H11 idea + `MUON_MOMENTUM=0.98` + recurrence L3-L5 start 0.35 | 1.1659 | 1.16792408 | 15,750,894 | 正收益，不如 R15 |
| S3-Q41 | `exp_s3_q41_r15_clip_top3_rank4_calib32_export` | export-only | R15 ckpt, top3 rank4, 32 calibration batches | n/a | 1.16747029 | 15,753,017 | 负于 R15/Q43 |
| S3-Q42 | `exp_s3_q42_r15_clip_top3_rank4_calib64_export` | export-only | R15 ckpt, top3 rank4, 64 calibration batches | n/a | 1.16743155 | 15,753,369 | 略优于 R15，负于 Q43 |
| S3-Q43 | `exp_s3_q43_r15_clip_top4_rank4_export` | export-only | R15 ckpt, top4 rank4 | n/a | 1.16735413 | 15,753,494 | 当前最佳 |
| S3-Q44 | `exp_s3_q44_r15_clip_top4_rank8_export` | export-only | R15 ckpt, top4 rank8 | n/a | 1.16735712 | 15,758,520 | 与 Q43 持平，略差且更大 |
| S3-Q45 | `exp_s3_q45_r15_clip_top5_rank4_export` | export-only | R15 ckpt, top5 rank4 | n/a | 1.16735526 | 15,755,306 | 与 Q43 基本持平，略差且更大 |
| S3-Q46 | `exp_s3_q46_r15_clip_top4_rank4_calib32_export` | export-only | R15 ckpt, top4 rank4, 32 calibration batches | n/a | 1.16747303 | 15,755,786 | 负于 Q43 |
| S3-Q47 | `exp_s3_q47_r15_clip_top4_rank4_calib64_export` | export-only | R15 ckpt, top4 rank4, 64 calibration batches | n/a | 1.16743122 | 15,755,677 | 负于 Q43 |
| S3-Q48 | `exp_s3_q48_r15_clip_top4_rank4_err075_export` | export-only | R15 ckpt, top4 rank4, `GPTQ_ERROR_SCALE=0.75` | n/a | 1.16810816 | 15,753,505 | 负收益 |
| S3-Q49 | `exp_s3_q49_r15_clip_top4_rank4_err125_export` | export-only | R15 ckpt, top4 rank4, `GPTQ_ERROR_SCALE=1.25` | n/a | 1.16792230 | 15,755,260 | 负于 Q43 |
| S3-Q50 | `exp_s3_q50_r15_clip_top3_rank8_export` | export-only | R15 ckpt, top3 rank8 | n/a | 1.16735861 | 15,757,796 | 与 Q43 持平，略差且更大 |
| S3-R22 | `exp_s3_r22_b3_qkgain5_hparam_tied004_muon097_recur_l3_5_start028_1xh100` | 1h train | H11 idea + `MUON_MOMENTUM=0.97` + recurrence L3-L5 start 0.28 | 1.1666 | 1.16887275 | 15,761,170 | pre 强，量化后不如 Q43 |
| S3-R23 | `exp_s3_r23_b3_qkgain5_hparam_tied004_muon097_recur_l3_5_start032_1xh100` | 1h train | H11 idea + `MUON_MOMENTUM=0.97` + recurrence L3-L5 start 0.32 | 1.1672 | 1.16900601 | 15,751,531 | 不如 Q43 |

## 正在运行

| RUN_ID | 类型 | GPU | 目的 |
| --- | --- | ---: | --- |
| 无 | - | - | 阶段 3 当前实验队列已跑完 |

## 代码入口

已新增第三阶段实验开关，默认关闭：

| 方向 | Env |
| --- | --- |
| S3-L1 coprime loader | `LOADER_MODE=coprime COPRIME_FILE_STRIDE=17 COPRIME_TOKEN_STRIDE=1048573` |
| S3-R1 轻量递归 | `RECURRENCE_EXTRA_PASSES=1 RECURRENCE_START_LAYER=3 RECURRENCE_END_LAYER=5 RECURRENCE_START_FRAC=0.35` |
| S3-H2 warmdown/min-lr | `WARMDOWN_FRAC`, `MIN_LR` |
| GPTQ artifact 重评 | `EVAL_GPTQ_ARTIFACT_ONLY=1 EVAL_GPTQ_ARTIFACT=final_model.gptq.ptz` |
| 递归 checkpoint export-only | `RECURRENCE_ACTIVE=1` |
| S3-T1 eval-only TTT MVP | `ttt_eval.py`，score-first lm-head LoRA，先记 pre-update score 再用同一 chunk 更新 adapter |

验证：`python -m py_compile train_gpt.py gptq_export.py` 通过；coprime loader CPU smoke 输出 `(16, 4096)` batch。

## S3-C1 CaseOps 可行性

- records 内有 CaseOps tokenizer 与准备脚本，例如 `records/track_10min_16mb/2026-04-23_SP8192_CaseOps_SparseGate_QuantGate_Loop45_PhasedTTT_PolarNS_MinLR_FusedCE/`。
- 当前 `/base/datasets` 与项目 `data/` 下未发现现成 CaseOps shard、`fineweb_val_bytes_*` sidecar 或 `docs_selected.jsonl`。
- 因此 CaseOps 不能直接进入 1h 训练；需要先恢复/生成原始 doc stream，再按 records 的 `prepare_caseops_data.py` 生成 token shard 和 byte sidecar，并把根脚本 BPB 统计改为优先读 sidecar。

## 下一步

- 当前最佳为 S3-Q43：R15 checkpoint + top4 rank4 GPTQ/LQER，`final_gptq+brotli_roundtrip_exact val_bpb=1.16735413`，总字节 15,753,494。
- 阶段 3 主线收敛：coprime/warmdown/TTT MVP 淘汰；有效组合为 `TIED_EMBED_LR=0.04 MUON_MOMENTUM=0.97 RECURRENCE_START_FRAC=0.30`，量化侧为 top4 rank4。
