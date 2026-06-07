# Parameter Golf 1xH100/1h Milestone Report

日期：2026-06-07

## 摘要

本项目在 OpenAI parameter-golf 思路下，约束改为单张 H100、每组训练 1 小时、最终提交物不超过 16 MB，主指标为 roundtrip `val_bpb`。当前最佳合规 artifact 为：

| 项目 | 值 |
| --- | --- |
| RUN_ID | `exp_s3_q43_r15_clip_top4_rank4_export` |
| Artifact | `results/experiments/exp_s3_q43_r15_clip_top4_rank4_export/final_model.gptq.ptz` |
| Roundtrip BPB | **1.16735413** |
| 总字节 | **15,753,494** |
| 训练 checkpoint | `exp_s3_r15_b3_qkgain5_hparam_tied004_muon097_recur_l3_5_start030_1xh100/final_model.pt` |

优化路线已经从“找一个质量好的训练基座”推进到“在 16 MB 内最大化可保留质量”。最有效的组合是：SP8192 + Seq4096 训练基座，GPTQ int6/embed8 + LQER 压缩，训练侧叠加 `QK_GAIN_INIT=5.0`、更适合量化的 hparam stack、`TIED_EMBED_LR=0.04`、`MUON_MOMENTUM=0.97`，并在训练后 30% 开启 L3-L5 轻量递归。

## 约束与核心矛盾

这个任务不是单纯追求最低 pre-quant BPB，而是在固定训练预算和固定提交大小内追求最低 roundtrip BPB。因此每个改动都必须同时回答两个问题：

1. 训练质量是否提高。
2. 量化、打包、压缩后是否仍能把质量保留下来。

第一阶段已经证明 SP8192 明显优于 SP1024，但普通 int8+zlib 会超过 16 MB。第二阶段的核心矛盾因此变成：如何把 SP8192 的质量压进 16 MB。第三阶段则进一步变成：在 GPTQ/LQER 已可用的基础上，哪些训练或结构改动既能降低 pre BPB，又不会被量化过程吃掉。

## 阶段一：先确定质量基座

第一阶段固定 9L x 512 架构，只扫 tokenizer 和序列长度。实验逻辑是先判断容量应该花在更长上下文还是更大词表上。

| 配置 | 结果 | 判断 |
| --- | ---: | --- |
| SP1024, Seq1024 | int8+zlib BPB 1.2320 | 合规但质量弱 |
| SP1024, Seq4096 | int8+zlib BPB 1.2113 | 长序列有效，但仍弱于 SP8192 |
| SP8192, Seq2048 | int8+zlib BPB 1.1815 | 质量强，但约 19.34 MB，超 16 MB |
| SP8192, Seq4096 B3 | int8+zlib BPB 1.18068959 | 质量更稳，仍超 16 MB |

阶段一结论是：SP8192 是正确方向，Seq4096 成为后续默认训练基座；但容量约束迫使项目必须重做低比特量化，而不是继续在 int8 路线上微调。

## 阶段二：把质量压进 16 MB

第二阶段先修正量化链路。早期 RTN 能合规但损失较大，说明“压缩得下”不等于“质量可用”。随后实现 GPTQ、signed bit packing、mixed bitwidth、brotli 压缩、LQER asymmetric int4 修正，并引入 `FRESH_MODEL_AFTER_QUANT=1`，避免 Hessian collection 后复用模型对象污染 roundtrip eval。

关键转折是用 current-code checkpoint 做 fresh compiled roundtrip 验证后，GPTQ 才成为可信主线：

| 方案 | Roundtrip BPB | 总字节 | 判断 |
| --- | ---: | ---: | --- |
| R9 RTN override | 1.18714581 | 15,972,698 | 合规 RTN 对照 |
| R9 GPTQ + LQER | 1.17978818 | 15,803,202 | GPTQ 明显优于 RTN |
| B3 GPTQ int6/embed8 | 1.18001764 | 15,731,454 | B3 GPTQ 基线 |
| B3 GPTQ + LQER top3 rank4 | **1.17661956** | **15,738,799** | 第二阶段最佳 |

阶段二的优化逻辑是：先让量化评估可信，再比较压缩策略。最终证明 B3 + GPTQ/LQER 比 Seq2048 RTN/GPTQ 分支更适合作为第三阶段 baseline。

## 阶段三：围绕可量化收益做局部爬山

第三阶段没有盲目叠加复杂 records 技术，而是把可迁移优化分成低成本量化细扫、训练超参、loader、递归、TTT、CaseOps 等方向逐步验证。实验结果显示，真正有效的是一条“训练动态改善 + 轻量递归 + 小幅量化细扫”的路径。

### 最佳方案演进

| 步骤 | 最佳 RUN_ID | Roundtrip BPB | 主要收益来源 |
| --- | --- | ---: | --- |
| 阶段二 baseline | `exp_gptq_b3_fresh_i6e8_quantile_err1_lqer` | 1.17661956 | 正确 GPTQ + LQER |
| QK gain | `exp_s3_h4_b3_qkgain5_1xh100` | 1.17610282 | `QK_GAIN_INIT=5.0` 首个正信号 |
| hparam stack | `exp_s3_h7_b3_qkgain5_hparam_stack_1xh100` | 1.17479819 | `BETA2=0.99`、clip/grad stack |
| tied embed LR | `exp_s3_h11_b3_qkgain5_hparam_tiedembedlr004_1xh100` | 1.17163042 | `TIED_EMBED_LR=0.04` 显著改善 |
| 轻量递归 | `exp_s3_r4_b3_qkgain5_hparam_tied004_recur_l3_5_start035_1xh100` | 1.17006027 | L3-L5 extra pass，可与 H11 叠加 |
| R4 export sweep | `exp_s3_q26_r4_clip_top4_rank8_calib64_export` | 1.16955543 | 量化校准和 LQER 细扫 |
| muon + recurrence | `exp_s3_r8_b3_qkgain5_hparam_tied004_muon097_recur_l3_5_start035_1xh100` | 1.16835283 | `MUON_MOMENTUM=0.97` 与递归叠加 |
| recurrence start | `exp_s3_r15_b3_qkgain5_hparam_tied004_muon097_recur_l3_5_start030_1xh100` | 1.16744173 | start frac 0.30 更好保留 pre 收益 |
| final export | `exp_s3_q43_r15_clip_top4_rank4_export` | **1.16735413** | R15 + LQER top4 rank4 |

从 B3 GPTQ/LQER 到 Q43，roundtrip BPB 下降约 0.00927。这个收益不是来自单个技巧，而是来自多个经验证的小正收益叠加：QK gain 改善注意力尺度，hparam stack 提高训练稳定性，tied embedding LR 让大词表参数更好适配 1h 预算，轻量递归增加有效深度，muon momentum 与 recurrence 共同改善后段训练，最后用 LQER top-k/rank 细扫吃掉剩余量化误差。

### 被证伪或暂缓的方向

| 方向 | 结果 | 结论 |
| --- | --- | --- |
| warmdown/min-lr | 多组 roundtrip 负收益 | 当前 1h 预算下不适合作为主线 |
| coprime loader | S3-L1/L2 明显负收益 | 当前实现破坏训练分布或批次局部性，不继续扩展 |
| lm-head LoRA TTT MVP | BPB 退化到 1.61284870 | 简化 TTT 不成立，不能代表 records 中的 phased/legal TTT |
| 更激进 LQER rank/top-k | 多数只持平或更大 | 当前瓶颈不再是简单增加 LQER 容量 |
| recurrence 更早/更晚/更深 | start 0.25/0.28/0.32/0.40/0.45/0.50/0.60、L3-L6 等均未超过 Q43 | 递归存在窄窗口，过强或时机不对会损失 step 效率或量化鲁棒性 |
| CaseOps | 当前环境缺 raw doc stream/byte sidecar | 潜在大收益，但需要先补数据链路 |

这些负结果很重要：它们说明当前最优不是“把 records 里的所有组件堆上去”，而是在 1h 单卡约束下寻找训练步数、有效深度、量化鲁棒性之间的平衡点。

## 当前最优组合的解释

Q43 的训练侧可以理解为三个层次的配合：

1. `QK_GAIN_INIT=5.0` 和 clip stack 先把基础训练动态调到更适合 SP8192/Seq4096 的区域。
2. `TIED_EMBED_LR=0.04` 重点照顾大词表 embedding/head 相关参数，这类参数对 BPB 和量化都敏感。
3. `MUON_MOMENTUM=0.97` 加 L3-L5 轻量递归提高有效建模能力，但只在训练后 30% 开启，避免全程递归拖慢太多 step 或放大早期不稳定。

导出侧选择 GPTQ int6/embed8 + LQER top4 rank4，是因为 matrix int6 提供主要容量节省，embed8 避免大词表质量断崖，LQER 用少量字节修正最大量化误差张量。R15 上 top4 rank4 优于 top3、top5、rank8 和更大 calibration sweep，说明当前 export 已接近局部平台期。

## 为什么距离 SOTA 仍然较远

当前结果是根脚本内的强局部最优，但距离 records 级 SOTA 仍有明显差距，核心原因有五类：

1. 数据和 tokenizer 仍然是普通 SP8192 FineWeb 路线。records 中后期强结果往往依赖 CaseOps、byte sidecar、doc 边界处理等数据链路优化；这类优化直接降低可建模熵，通常不是训练超参能完全补回的。
2. 模型结构仍偏传统 compact GPT。当前只加入了轻量共享递归，没有实现 parallel residual lane、SparseGate/QuantGate、SmearGate、BOS-safe document handling 等 records 中常见的结构收益。
3. TTT 仍未真正落地。已测试的只是 lm-head LoRA MVP，且结果明显负收益；records 里的收益来自更严格的 score-before-update、phased TTT、adapter 位置和更新日程设计，不能用当前 MVP 代表上限。
4. 压缩还有工程余量。当前使用 brotli + packed GPTQ/LQER，但没有 per-group layout、similarity sort、lrzip、专门的 weight ordering 或更细粒度 bit allocation。这些主要提供容量余量，进而允许更有价值的结构参数或修正项进入 16 MB。
5. 搜索预算仍是局部搜索。单组 1h、单 seed、本地网格扫法对小于 1e-4 的差异很敏感；很多 records 级组合需要更完整的联动调参，而不是单独移植一个开关。

## 下一步建议

短期不建议继续在 R15/Q43 周围做大量 top-k、rank、calibration 微扫，因为 Q43 附近已经出现平台期。更值得投入的是三条更高杠杆路径：

1. CaseOps/data sidecar 可行性：先恢复 raw doc stream 和 byte sidecar，验证 BPB 统计合规，再做短训 smoke。
2. 真正的 legal/phased TTT：不要沿用 lm-head MVP，改为按 records 设计 adapter 位置、phase、score-before-update 和更新预算。
3. 结构性小参数收益：优先尝试 parallel residual lane 或 gate 类结构，但必须同步设计量化/打包方式，避免 pre BPB 改善被 roundtrip 吃掉。

当前 Q43 是一个清晰里程碑：它证明 GPTQ/LQER 主线、tied embed LR、muon momentum 和轻量递归在 1xH100/1h 约束下可以稳定叠加；同时也说明，下一轮要接近 SOTA，主要瓶颈已经从普通超参搜索转向数据链路、推理时自适应和结构/压缩联合设计。
