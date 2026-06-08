# Parameter Golf 1xH100/1h Milestone Report

日期：2026-06-08

## 摘要

本项目在 OpenAI parameter-golf 思路下，约束改为单张 H100、每组训练 1 小时、最终提交物不超过 16 MB，主指标为 roundtrip `val_bpb`。当前最佳合规 artifact 为：

| 项目 | 值 |
| --- | --- |
| RUN_ID | `exp_s5_r8_seed42_warmdown095_records0427_caseops_lrzip_1xh100` |
| Artifact | `results/experiments/exp_s5_r8_seed42_warmdown095_records0427_caseops_lrzip_1xh100/final_model.int6.ptz` |
| Roundtrip BPB | **1.09647097** |
| Post-TTT BPB | **1.08179851** |
| 总字节 | **15,924,510** |
| 训练 checkpoint | `results/experiments/exp_s5_r8_seed42_warmdown095_records0427_caseops_lrzip_1xh100/final_model.pt` |
| Phase 4 对照 | `exp_s4_qs28_g3g1k3_rerun_calib32_err0975_export`：1.16522320 / 15,755,553 bytes |
| records 参考 | `2026-04-27...1.0611`，3-seed mean 约 1.06108 |

优化路线已经从“找一个质量好的训练基座”推进到“复现并改造 records 级系统栈”。第三阶段的 Q43 是根脚本局部最优；第四阶段通过 SparseGate + LeakyReLU^2 + Polar NS 将根脚本 best 推到 1.16522320；第五阶段补齐 CaseOps byte-sidecar 评估、records 2026-04-27 advanced stack fallback、真实外部 `lrzip` 和 legal phased TTT，最终由 seed42 + `WARMDOWN_FRAC=0.95` 的 R8 刷新到 post-TTT 1.08179851。相对 Phase 4 best，Phase 5 下降 `0.08342469 BPB`，已经略优于 2026-04-06 记录，但距离 2026-04-27 mean 仍有约 `0.02072 BPB`。

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
| final export | `exp_s4_qs28_g3g1k3_rerun_calib32_err0975_export` | **1.16522320** | **15,755,553** | 阶段四最佳 |

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
| SparseGate + LeakyReLU^2 + Polar NS | QS28 roundtrip 1.16522320 | 阶段四唯一有效组合，但收益有限 |
| export-only 细扫 | QS26-QS30 均在 1.16522 附近 | 已进入约 `1e-5 BPB` 级平台期 |

还有一个工程复盘：第一次 SparseGate 完整批次从项目根目录并行启动，`final_model.pt` 被多个训练进程覆盖，导致根目录 checkpoint 不能可靠归属到最佳 run。后续通过 run-dir cwd 的 checkpoint-safe rerun 修正，并以 `results/experiments/exp_s4_g3g1k3_sparse_w12_leaky_polarns_q43_rerun_ckpt_1xh100/final_model.pt` 作为唯一可信导出基座。

### 阶段四结论

第四阶段没有证明 records 技术无效，只证明“把少数组件单独移植到 Q43 静态训练栈上”收益很小。SmearGate、SparseGate、LeakyReLU^2、Polar NS 这类组件在 records 中大概率不是孤立增益源，而是与 CaseOps/doc boundary、TTT、parallel residual/lane、压缩布局共同 co-adapt。当前做法缺少这些配套机制，因此只能得到局部小收益或被量化损失吞掉。

### CaseOps 数据链路补充

2026-06-08 已补上阶段四最关键的数据前置工作：容器 `b5e2809a5863` / `lbqy0` 内的 canonical raw docs 已下载完成，路径为 `/base/datasets/CaseOps/raw/docs_selected.jsonl`，大小约 44GB；`docs_selected.source_manifest.json` 和 `manifest.json` 也已就位。详细交接记录见 `experiments/caseops_handoff.md`。

目前完成的 smoke：

| 检查项 | 结果 | 说明 |
| --- | --- | --- |
| raw docs 读取 | 通过 | 首行是 JSON object，含 `text` 字段 |
| CaseOps tokenizer | 可用 | records 内 `fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model` 可加载，vocab=8192 |
| 小样本 prepare | 通过 | 前 64 篇 doc 生成 train/val/val_bytes shard |
| byte sidecar 对齐 | 通过 | val token shard 与 `fineweb_val_bytes_000000.bin` 长度一致 |
| BOS sidecar | 通过 | 16 个验证 doc 的 BOS byte 全为 0 |
| 根脚本 loader | 通过 | 极小模型 1 step + int8 roundtrip 可读取 CaseOps token shard |

这个补充改变了阶段四的问题定位：CaseOps 已不再是“缺 raw docs 无法开始”，而是“raw docs 和 prepare smoke 可用，但完整 token shards 尚未生成，根脚本尚未按 `fineweb_val_bytes_*.bin` 计算原始 byte BPB”。当前 `train_gpt.py` smoke 日志仍是 `val_bpb:enabled tokenizer_kind=sentencepiece`，还没有出现 `val_bpb:byte_sidecar:enabled`。因此，CaseOps 在阶段四末尾还不能进入完整 1h 训练基线；第五阶段必须先完成 full prepare 和 byte-sidecar BPB 接入。

## 阶段五：CaseOps 合规链路与 records 主线

第五阶段的目标从“继续榨 QS28”切换为“让 records 04-27 主线在 1xH100/1h 环境下真实跑通并优化”。这一步先补齐数据和评估闸门：CaseOps full prepare 生成 80 个 train shards、1 个 val shard、1 个 val byte sidecar shard；验证集 token 数为 9,662,502，byte sidecar 合计 29,950,979 bytes，BOS 数 10,000，bad BOS bytes 为 0。根目录 `train_gpt.py` 同时修复了 `fineweb_val_*.bin` 误匹配 `fineweb_val_bytes_*.bin` 的 glob 问题，并在 sidecar 存在时按 shifted `y` target 对齐 byte count 计算 BPB，日志已出现 `val_bpb:byte_sidecar:enabled`。

第一批 root CaseOps 结果是负信号：C5 的 CaseOps + QS28 结构为 `1.17380957`，明显差于 Phase 4 QS28 的 `1.16522320`；C4/C4b/C5b/C6 也没有证明 root 静态 CaseOps 可直接获益。普通 SP8192 root 线仍能挤出小幅收益，O8 达到 `1.16357912`，O4 的 MLP3 甚至到 `1.15317521`，但 O4 总字节为 `16,806,331`，超过 16 MB；后续 P1-P4 容量修复没有产出有效更优结果。因此 root 线从主线降级为备份。

真正的转折来自 records 2026-04-27 advanced stack。当前环境缺 `flash_attn_interface`、`triton.tools.tensor_descriptor` 和部分 varlen compile 条件，阶段五先补了 torch SDPA fallback、eager MLP fallback、无 FA3 时固定序列 loader、BOS 初始化和 cu bucket warmup 跳过逻辑。R1 完成训练后卡在缺少 `lrzip`；安装真实外部 `lrzip` 后，R1Q7 将 embed7/top3 artifact 压到 `15,925,658` bytes，no-TTT BPB 为 `1.09951341`，首次形成有效 near-1.10 结果。R1Q7T 的 legal phased TTT 进一步降到 `1.08464351`，证明 TTT 收益约 `0.0149 BPB`，且不是非法重评分或 fallback artifact 的偶然结果。

随后第五阶段重点转向训练起点，而不是继续扫 TTT 微参。T3-T14 只把 R1Q7T 从 `1.08464351` 微调到 T12 的 `1.08458960`，收益约 `5e-5 BPB`；R2/R3/R9 显示 seed42 明显更好，其中 R2 为 `1.09686294 -> 1.08218120`。最终 R8 在 seed42 上把 `WARMDOWN_FRAC` 调到 0.95，得到 no-TTT `1.09647097`、post-TTT `1.08179851`、总字节 `15,924,510`，成为当前有效最好。R12/R13/R20 的 warmdown 0.90/0.98/0.94、R14/R15 的跨 seed、R18/R19 的 min_lr、R8Q1-Q5 的导出微扫均未刷新 R8。

| 分支 | 最佳结果 | no-TTT / roundtrip | post-TTT | bytes | 结论 |
| --- | --- | ---: | ---: | ---: | --- |
| Phase 4 root QS28 | `exp_s4_qs28_g3g1k3_rerun_calib32_err0975_export` | 1.16522320 | | 15,755,553 | 旧本地最佳 |
| Phase 5 root QS28 | S5-O8 | 1.16357912 | | 15,761,143 | 有小收益，但已非主线 |
| records R1Q7 | S5-R1Q7/T | 1.09951341 | 1.08464351 | 15,925,658 | 真实 `lrzip` 让 records artifact 合规 |
| records seed42 | S5-R2 | 1.09686294 | 1.08218120 | 15,925,323 | seed/training 质量是主要杠杆 |
| records warmdown095 | S5-R8 | **1.09647097** | **1.08179851** | **15,924,510** | 当前最终候选 |

第五阶段的真实结论是：在单 H100/1h + 当前 fallback 环境下，records 04-27 栈已经大幅优于 root QS28；有效收益来自 CaseOps 合规链路、records 结构栈、真实 `lrzip`、seed42、warmdown095 和默认 phased TTT 的组合。继续做 TTT 小扫、导出小扫、root CaseOps 静态训练或延后 loop 已经进入低收益区。

## 当前最优组合的解释

当前最佳 R8 不是 QS28 的局部导出结果，而是 records 04-27 advanced stack 在本地 fallback 环境下跑通后的真实优化结果。它可以理解为四个层次的配合：

1. CaseOps + byte sidecar 先改变评估和数据表示，让 BPB 按原始 byte 对齐，并保留 doc/BOS 信息供 records stack 和 TTT 使用。
2. records 04-27 结构栈提供主要质量跃迁，包括更深/更宽的 advanced 配置、gate/smear/quant gate、recurrence、parallel decoder/lane、LQER 和 int7 embedding 等；本地因缺 FA3/fused 组件走了 SDPA/eager/fixed-loader fallback。
3. 真实外部 `lrzip` 是合规关键。fallback brotli/lzma 压缩下 embed7/top3 artifact 仍超过 16 MB；`lrzip` 将同一权重压到 15,925,658 bytes，并保留 `1.09951341` 的 no-TTT 质量。
4. seed42 + `WARMDOWN_FRAC=0.95` 改善了单卡低步数训练起点，R8 no-TTT 从 R1Q7 的 `1.09951341` 降到 `1.09647097`；默认 phased TTT 再提供约 `0.01467 BPB`，到最终 `1.08179851`。

因此当前瓶颈已经不在普通 GPTQ/LQER 微参。R8Q1-Q5 的 error scale、calib batch、top-k 导出小扫均退化；R8T12 也说明 R1Q7 上的 TTT 近邻设置迁移到 R8 后不再改善。主杠杆是更好的训练吞吐/内核、records 栈完整度和可能的多 seed/多预算策略。

## 为什么距离 SOTA 仍然较远

当前结果已经不是根脚本局部最优，而是本地 records fallback 主线的有效结果。距离 2026-04-27 records mean 约 `1.06108` 仍有 `1.08179851 - 1.06108 ~= 0.02072 BPB`。这个 gap 已经比 Phase 4 小很多，但仍不是靠 export-only 或 TTT 微扫能关闭的量级。

1. no-TTT 起点仍偏高。R8 no-TTT 为 `1.09647097`，而 04-27 records post-quant 约在 `1.073-1.075` 区间；R8 的 TTT 收益约 `0.01467 BPB`，量级已经接近 records，因此主要差距在训练后 artifact 本身。
2. 当前 records 路线运行在 fallback 环境。缺 FA3、TensorDescriptor/fused MLP 和完整 varlen compile path 后，训练退到 fixed-sequence/SDPA/eager fallback；R8 只有约 3202 steps，吞吐约 708k tok/s。04-27 参考在 8xH100/600s 环境下约 4931 steps、约 121.7ms/step，训练预算和 kernel 条件都更强。
3. 结构栈虽然跑通，但不是完整高吞吐形态。fallback 让 doc-boundary/varlen 优势、融合 MLP 和部分 layout 优化不能完全发挥；这会同时影响训练步数、batch 组成和后段收敛质量。
4. TTT 近邻已经平台。R1Q7 上 T3-T14 最多只改善约 `5e-5 BPB`，R8T12 反而略差于默认 TTT；继续扫 prefix docs、phase 数、global lr 的预期收益很低。
5. 压缩不是当前主瓶颈，但仍是 hard constraint。真实 `lrzip` 已经让 embed7/top3 合规；R8Q1-Q5 说明当前量化参数附近没有可吃的大收益。未来若想放更大结构或 adapter，仍需要更强 packing/per-group/layout，而不是现有 LQER top-k 微调。

所以，Phase 5 后的核心判断变了：项目已经补齐了数据/sidecar、合法 TTT 和 records 结构主线，不再是“缺机制”；剩余 gap 更像是单卡 1h fallback 训练质量与 records 参考环境之间的计算/内核/吞吐差距。下一轮应优先让同一 records 栈跑得更接近原始高吞吐条件，而不是继续围绕 R8 artifact 做小半径微调。

## 下一步建议

短期应把 R8 固化为当前候选，并停止低收益分支：root CaseOps 静态训练、QS28/export-only 大量微扫、R1Q7/R8 TTT 近邻、延后 loop、batch917k、min_lr 抬高都已经有明确负结果。

1. 固化 R8 复现链路：记录真实外部 `lrzip` 依赖、CaseOps shards/sidecar、records fallback 开关、默认 TTT 配置和最终 artifact bytes，确保最终提交脚本能稳定复现 `1.08179851`。
2. 优先解决 FA3/fused MLP/TensorDescriptor 环境 blocker。目标不是单纯加速，而是让 records 04-27 栈恢复 varlen/doc-boundary 高吞吐训练形态；这是目前最可能继续降低 no-TTT 起点的方向。
3. 环境修好后重跑最小矩阵：seed42 + `WARMDOWN_FRAC=0.95` 作为第一优先级，再补 seed0/1234 或一个 warmdown 近邻；判断是否能把 no-TTT 从 `1.09647` 推向 records 参考的 `1.073-1.075`。
4. 若 kernel 环境短期不可解，备选是 root-native doc-boundary 高吞吐实现：尽量保留 CaseOps sidecar、records TTT 合规语义和 R8 的 warmdown/seed 经验，但减少 fallback 带来的 compile/loader 损失。
5. 只有在出现更强 no-TTT artifact 后，再重启 TTT 或 export 小扫；否则这些微扫大概率只贡献 `1e-5` 到 `1e-4 BPB`。

当前里程碑比 Phase 4 清楚得多：项目已经从 `1.16522` 跳到 `1.08180`，证明 records 主线在单卡 1h 约束下也能真实工作；下一步的胜负手不是“再找一个小开关”，而是把同一主线的训练质量和运行环境继续往 04-27 reference 靠近。
