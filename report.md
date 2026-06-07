# Parameter Golf 1xH100/1h Milestone Report

日期：2026-06-07

## 摘要

本项目在 OpenAI parameter-golf 思路下，约束改为单张 H100、每组训练 1 小时、最终提交物不超过 16 MB，主指标为 roundtrip `val_bpb`。当前最佳合规 artifact 为：

| 项目 | 值 |
| --- | --- |
| RUN_ID | `exp_s4_qs28_g3g1k3_rerun_calib32_err0975_export` |
| Artifact | `results/experiments/exp_s4_qs28_g3g1k3_rerun_calib32_err0975_export/final_model.gptq.ptz` |
| Roundtrip BPB | **1.16522320** |
| 总字节 | **15,755,553** |
| 训练 checkpoint | `results/experiments/exp_s4_g3g1k3_sparse_w12_leaky_polarns_q43_rerun_ckpt_1xh100/final_model.pt` |
| Q43 对照 | `exp_s3_q43_r15_clip_top4_rank4_export`：1.16735413 / 15,753,494 bytes |
| records 参考 | `2026-04-27...1.0611`，3-seed mean 约 1.06108 |

优化路线已经从“找一个质量好的训练基座”推进到“在 16 MB 内最大化可保留质量”。第三阶段的 Q43 是根脚本局部最优：SP8192 + Seq4096 训练基座，GPTQ int6/embed8 + LQER 压缩，训练侧叠加 `QK_GAIN_INIT=5.0`、更适合量化的 hparam stack、`TIED_EMBED_LR=0.04`、`MUON_MOMENTUM=0.97`，并在训练后 30% 开启 L3-L5 轻量递归。第四阶段在此基础上加入 SparseGate + LeakyReLU^2 + Polar NS，并通过 `calib32 + GPTQ_ERROR_SCALE=0.975` 导出刷新到 1.16522320；相对 Q43 只改善约 0.00213 BPB，距离 records mean 仍有约 0.10414 BPB 的巨大差距。

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

## 阶段四：records 组件移植与局部收益

第四阶段尝试把 records 中常见但不依赖新数据链路的组件先移植到 Q43 栈上，重点看它们是否能在 1h 和 16 MB 约束下转化为 roundtrip 收益。结果比较明确：SparseGate + LeakyReLU^2 + Polar NS 是唯一打破 Q43 的组合，但提升幅度远小于 SOTA gap。

### 阶段四最佳路径

| 步骤 | RUN_ID | Roundtrip BPB | 总字节 | 判断 |
| --- | --- | ---: | ---: | --- |
| Q43 对照 | `exp_s3_q43_r15_clip_top4_rank4_export` | 1.16735413 | 15,753,494 | 第三阶段最佳 |
| SparseGate + LeakyReLU^2 + Polar NS | `exp_s4_g3g1k3_sparse_w12_leaky_polarns_q43_1xh100` | 1.16622225 | 15,756,176 | 首次超过 Q43，但原并行 cwd 覆盖 checkpoint |
| checkpoint-safe rerun | `exp_s4_g3g1k3_sparse_w12_leaky_polarns_q43_rerun_ckpt_1xh100` | 1.16529603 | 15,755,792 | 可靠 checkpoint，复现并超过原 run |
| final export | `exp_s4_qs28_g3g1k3_rerun_calib32_err0975_export` | **1.16522320** | **15,755,553** | 当前本地最佳 |

阶段四的净收益约为 `1.16735413 - 1.16522320 = 0.00213093 BPB`。这说明 gate/activation/optimizer 组合确实有信号，但这个信号只相当于 SOTA gap 的约 2%，不能改变整体落后格局。

### 阶段四负结果

| 方向 | 代表结果 | 结论 |
| --- | --- | --- |
| LeakyReLU^2 单项 | `S4-G1` roundtrip 1.17263005 | 单独替换激活不可用 |
| partial RoPE / `ROTARY_DIM=16` | `S4-RoPE1` roundtrip 1.17619727 | 当前 partial RoPE 明显负收益 |
| Polar NS 单项 | `S4-K3` roundtrip 1.17343709 | 速度信号不足以抵消质量损失 |
| LeakyReLU^2 + Polar NS | `S4-G1K3` pre 1.1656，roundtrip 1.16956051 | pre 强但量化损失吃掉收益 |
| SmearGate | 最好 `S4-G2G1W24` roundtrip 1.17304535 | 单独接入 Q43 不成立，不代表 records 组合无效 |
| SparseGate 单项 | `S4-G3` roundtrip 1.16893548 | 单项仍负于 Q43 |
| SparseGate + Polar NS | `S4-G3K3` roundtrip 1.16774440 | 接近 Q43 但不够 |
| SparseGate + LeakyReLU^2 + Polar NS | QS28 roundtrip 1.16522320 | 当前唯一有效组合，但收益有限 |
| export-only 细扫 | QS26-QS30 均在 1.16522 附近 | 已进入约 `1e-5 BPB` 级平台期 |

还有一个工程复盘：第一次 SparseGate 完整批次从项目根目录并行启动，`final_model.pt` 被多个训练进程覆盖，导致根目录 checkpoint 不能可靠归属到最佳 run。后续通过 run-dir cwd 的 checkpoint-safe rerun 修正，并以 `results/experiments/exp_s4_g3g1k3_sparse_w12_leaky_polarns_q43_rerun_ckpt_1xh100/final_model.pt` 作为唯一可信导出基座。

### 阶段四结论

第四阶段没有证明 records 技术无效，只证明“把少数组件单独移植到 Q43 静态训练栈上”收益很小。SmearGate、SparseGate、LeakyReLU^2、Polar NS 这类组件在 records 中大概率不是孤立增益源，而是与 CaseOps/doc boundary、TTT、parallel residual/lane、压缩布局共同 co-adapt。当前做法缺少这些配套机制，因此只能得到局部小收益或被量化损失吞掉。

## 当前最优组合的解释

当前最佳 QS28 是 Q43 训练栈上继续叠加 SparseGate + LeakyReLU^2 + Polar NS 后，再做一轮窄幅导出细扫得到的结果。它可以理解为四个层次的配合：

1. `QK_GAIN_INIT=5.0` 和 clip stack 先把基础训练动态调到更适合 SP8192/Seq4096 的区域。
2. `TIED_EMBED_LR=0.04` 重点照顾大词表 embedding/head 相关参数，这类参数对 BPB 和量化都敏感。
3. `MUON_MOMENTUM=0.97` 加 L3-L5 轻量递归提高有效建模能力，但只在训练后 30% 开启，避免全程递归拖慢太多 step 或放大早期不稳定。
4. SparseGate + LeakyReLU^2 + Polar NS 在训练质量上进一步降低 pre BPB；最终通过 `GPTQ_CALIBRATION_BATCHES=32` 和 `GPTQ_ERROR_SCALE=0.975` 保住其中一部分收益。

导出侧仍选择 GPTQ int6/embed8 + LQER top4 rank4，是因为 matrix int6 提供主要容量节省，embed8 避免大词表质量断崖，LQER 用少量字节修正最大量化误差张量。QS28 之后的 export-only 收益已经进入 `1e-5 BPB` 级别，说明当前瓶颈不再是普通 GPTQ/LQER 参数细扫。

## 为什么距离 SOTA 仍然较远

当前结果是根脚本内的强局部最优，但距离 records 级 SOTA 仍有约 0.10414 BPB 差距。这个 gap 太大，不可能靠 Q43/QS28 周围继续扫 `top_k`、`rank`、`calib`、`error_scale` 或单个小结构开关关闭。更本质的问题是：我们仍在优化一个“普通 token-stream 静态模型”，而 records 级方案很可能已经把数据表示、推理时自适应、结构栈和压缩布局一起改了。

1. 数据链路缺陷是第一层硬伤。当前仍是普通 SP8192 token stream，缺 CaseOps、byte sidecar、doc boundary 和 BOS sidecar 合规评估。records 路线不是单纯让同一个 token 分布训练得更好，而是降低待预测序列的有效熵，并把 byte 级信息放进合规 sidecar 中。这个收益通常是架构和超参补不回来的。
2. TTT 还没有真正实现。第三阶段失败的 `ttt_eval.py` 只是 lm-head LoRA、全局连续 token block 更新，不能代表 records 中的 legal/phased TTT。真正可能有收益的版本需要 doc-boundary、score-before-update、单 pass 无 rescoring、multi-phase global SGD、Q/K/V/O/MLP/head adapters、per-doc reset 或 warm-start，并且 forward path 必须镜像训练/导出模型。我们现在缺的是完整机制，不是差一个 LoRA rank。
3. 结构栈是零散移植，不是共同设计。QS28 只接入了 SparseGate + LeakyReLU^2 + Polar NS；缺 parallel residual / decoder lane、XSA、11L/MLP4x、QuantGate/SmearGate 的正确组合、doc-safe gate 路由等 records 常见组件。单项 ablation 负收益不意外，因为这些组件可能依赖数据链路、TTT 和压缩容量共同适配。
4. 压缩链路仍在“保质量”，没有“创造容量”。GPTQ/LQER/brotli 已经能把当前模型压进 16 MB，但没有 per-group layout、simsort、lrzip、权重重排、按层 bit allocation 或面向 gate/lane/adapters 的专门编码。没有新的压缩余量，就很难同时放入 embed7/更大结构/adapter/LQER 修正并维持 roundtrip。
5. 1h 静态训练的有效计算量不够。当前 H100 1h 约 5k optimizer steps，靠静态小模型一次性学完所有分布。records 级方案很可能通过 CaseOps 降低分布难度，通过 TTT 在验证/测试文档上追加合规自适应计算，通过结构栈提高参数效率。我们主要还在做离线训练内的局部优化。
6. 搜索方式仍是局部爬山。第四阶段大多数实验是“Q43 + 一个开关或两个开关”，这对发现小收益有效，但不适合复现 records 级组合。SOTA gap 是系统性 gap，不是单开关 gap。

所以，本项目当前最本质的缺陷不是 FlashAttention、RoPE、ReLU^2 或某个 kernel 是否还差一点，而是缺少 records 方案里改变问题形态的三件事：数据/sidecar、合法 TTT、结构/压缩联合设计。内核优化仍有价值，但它主要买训练步数或让更复杂结构跑满 1h；如果目标分布和推理机制不变，单靠 kernel 很难把 0.104 BPB 的 gap 打穿。

## 下一步建议

短期不建议继续在 QS28 周围做大量 top-k、rank、calibration、error-scale 微扫，因为导出侧已经出现平台期。若目标是逼近 SOTA，下一阶段需要停止以单开关 ablation 为主线，改成三条高杠杆路径：

1. CaseOps/data sidecar 可行性：先恢复 raw doc stream 和 byte sidecar，验证 `val_bpb:byte_sidecar:enabled`、BOS sidecar 和 doc boundary 合规，再做短训 smoke。
2. 真正的 legal/phased TTT：不要沿用 lm-head MVP，改为按 records 设计 adapter 位置、phase、score-before-update、单 pass、无 rescoring、per-doc reset/warm-start 和更新预算；先 eval-only 接到 Q43/QS28 artifact 上，看是否至少有 `-0.002 BPB`。
3. 结构/压缩联合设计：parallel residual / decoder lane、11L/MLP4x、XSA、gate 类结构必须和 per-group compression、simsort/lrzip、专门 bit allocation 一起设计，避免 pre BPB 改善继续被 roundtrip 吃掉。

当前 QS28 是一个清晰但也令人不舒服的里程碑：它证明 Q43 主线仍能被 SparseGate + LeakyReLU^2 + Polar NS 小幅推进；同时也说明，继续沿着“静态训练 + 普通 token stream + 导出微扫”的路线，只会得到 `1e-3` 到 `1e-5` 级收益，无法逼近 records 级 SOTA。下一轮的成败点必须转向数据链路、推理时自适应和结构/压缩联合设计。
