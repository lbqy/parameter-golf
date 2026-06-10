# Parameter Golf 单卡 H100 1 小时约束下的实验报告

日期：2026-06-09

## 摘要

本项目参考 OpenAI Parameter Golf 的 16MB 模型压缩挑战，但将训练资源约束改为 **单张 H100、每组训练 1 小时**。项目目标不是单纯训练一个验证 loss 更低的语言模型，而是在固定训练时间和固定提交大小下，得到最低的 roundtrip `val_bpb`。所谓 roundtrip，指模型经过量化、打包、压缩、解压并重新加载后再评估；因此最终分数同时受训练质量、量化鲁棒性、压缩容量和评估策略影响。

实验从根目录 baseline 出发，依次经历了 tokenizer/上下文长度探索、低比特量化链路修复、训练动态局部优化、records 组件移植，以及最终的 CaseOps + records 04-27 stack 主线复现与优化。当前最佳有效结果为：

| 项目 | 结果 |
| --- | --- |
| RUN_ID | `exp_rope_zlf2_r8_pg291_fa3_smear_seed42_nottt` + `exp_rope_zlf2_r8_pg291_fa3_smear_seed42_ttt_eval` |
| 当前最佳方案 | CaseOps + records 04-27 stack + `pg291` FA3/TensorDescriptor 环境 + seed42 + warmdown 0.95 + partial RoPE low-frequency zeroing + real `lrzip` + legal phased TTT |
| no-TTT / roundtrip BPB | **1.08972570** |
| post-TTT BPB | **1.07586081** |
| artifact total bytes | **15,917,703** |
| 第四阶段最佳对照 | `1.16522320` |
| 改善幅度 | `0.08936239 BPB` |

最终结果说明：在单卡 1 小时约束下，单纯在根目录训练脚本的局部最优附近继续微调已经进入平台期；真正的大幅提升来自把数据链路、模型结构、量化压缩、运行环境和 legal TTT 作为一个系统整体优化。

## 1. 实验背景与核心问题

Parameter Golf 的特殊之处在于，它把语言模型训练问题变成了一个“质量-容量-时间”的三方折中问题。普通大模型训练通常只关心验证集 loss 或 perplexity；本项目还必须满足：

1. 最终提交物小于 16MB。
2. 训练预算固定为单张 H100 1 小时。
3. 评估时看到的是压缩后重新加载的模型。

因此，实验过程中我逐渐认识到：一个优化方法不能只看 pre-quant BPB。它必须同时回答三个问题：

| 问题 | 对应指标 | 含义 |
| --- | --- | --- |
| 模型学得好不好 | pre-quant BPB | 浮点权重本身的质量 |
| 压缩后还剩多少质量 | roundtrip BPB | 量化、打包、压缩再加载后的质量 |
| 是否能提交 | total bytes | 是否小于 16MB |

这也决定了本项目的探索路线不是简单扩大模型，而是不断寻找“能被压缩保留下来的有效能力”。

## 2. 实验方法演进

### 2.1 第一阶段：确定 tokenizer 和上下文长度基座

最初的 baseline 使用 SP1024 tokenizer、9 层 512 维 GPT-like decoder、int8+zlib 导出。第一阶段主要验证两个基础问题：

1. 更长上下文是否有用。
2. 更大词表是否值得付出 embedding 体积代价。

实验结果如下：

| 配置 | Roundtrip BPB | artifact 状态 | 结论 |
| --- | ---: | --- | --- |
| SP1024, seq1024 | 1.23203866 | 合规 | baseline |
| SP1024, seq2048 | 1.21549379 | 合规 | 长上下文有效 |
| SP1024, seq4096 | 1.21126584 | 合规 | seq4096 更优 |
| SP8192, seq2048, int8 | 1.18147653 | 超 16MB | 质量强但装不下 |
| SP8192, seq4096, int8 | 1.18068959 | 超 16MB | 后补做的更强 SP8192 训练基座 |

这一阶段的关键发现是：SP8192 明显优于 SP1024，seq4096 又比 seq2048 略好，因此后续以 **SP8192 + seq4096** 作为主要质量基座。这个基座虽然是后续补做才确认的，但从方法逻辑上属于第一阶段：它回答的是“训练基座应该长什么样”。问题在于，普通 int8 压缩仍无法满足 16MB，因此后续实验的核心不再是“要不要用大词表”，而是“如何把 SP8192 + seq4096 的质量压进 16MB”。

### 2.2 第二阶段：修复低比特量化与 roundtrip 评估

第二阶段的重点是量化。早期 RTN 量化可以压缩模型，但 BPB 损失明显。随后引入 GPTQ、mixed bitwidth、signed bit packing、LQER 和 fresh roundtrip eval。

| 方法 | Roundtrip BPB | bytes | 判断 |
| --- | ---: | ---: | --- |
| RTN override | 1.18714581 | 15,972,698 | 合规但损失大 |
| GPTQ + LQER | 1.17978818 | 15,803,202 | 明显优于 RTN |
| SP8192 seq4096 + GPTQ int6/embed8 | 1.18001764 | 15,731,454 | 可用低比特基线 |
| SP8192 seq4096 + GPTQ/LQER top3 rank4 | **1.17661956** | **15,738,799** | 第二阶段最佳 |

这一阶段让我形成了一个重要判断：量化不是训练结束后的附属步骤，而是整个实验的核心。很多训练收益如果不能被量化保留下来，对最终提交没有意义。

### 2.3 第三阶段：围绕 SP8192 低比特基座做局部爬山

在 GPTQ/LQER 链路稳定后，第三阶段开始优化训练动态和小结构。实验采用局部爬山方式，每次只改变少数变量，观察 roundtrip BPB 是否真正改善。

关键演进如下：

| 步骤 | 最佳 BPB | 主要方法 | 理解 |
| --- | ---: | --- | --- |
| 低比特压缩基座 | 1.17661956 | GPTQ + LQER | 合规质量基座 |
| QK gain | 1.17610282 | `QK_GAIN_INIT=5.0` | 改善 attention 初始尺度 |
| hparam stack | 1.17479819 | beta/clip/LR stack | 稳定短训 |
| tied embed LR | 1.17163042 | `TIED_EMBED_LR=0.04` | 大词表 embedding 学习率很敏感 |
| recurrence | 1.17006027 | L3-L5 extra pass | 用少量额外计算提高有效深度 |
| Muon + recurrence | 1.16835283 | `MUON_MOMENTUM=0.97` | 矩阵更新动态更好 |
| 训练动态综合方案 | **1.16735413** | recurrence start 0.30 + LQER top4 rank4 | 第三阶段最佳 |

第三阶段的收获是：小模型短训中，attention 尺度、embedding 学习率、后段 recurrence 和 optimizer momentum 都会产生可见影响。但这些收益是逐步叠加的，每个单项通常只贡献 `1e-3` 到 `1e-2` BPB。

同时，一些看似合理的方向被证伪：coprime loader 明显负收益；简化版 lm-head LoRA TTT 退化严重；更激进的 LQER top-k/rank 没有带来持续收益。这些负结果帮助我认识到，不能把 records 里的概念做一个简化版本就认为等价。

### 2.4 第四阶段：尝试移植 records 组件

第四阶段开始从历史 records 中移植结构组件，重点测试 SparseGate、SmearGate、LeakyReLU²、Polar NS、partial RoPE 等方法能否在第三阶段训练动态综合方案上直接生效。

| 方向 | 代表结果 | 结论 |
| --- | ---: | --- |
| LeakyReLU² 单项 | 1.17263005 | 负收益 |
| partial RoPE | 1.17619727 | 明显负收益 |
| Polar NS 单项 | 1.17343709 | 负收益 |
| SmearGate 单项 | 1.17304535 | 不成立 |
| SparseGate 单项 | 1.16893548 | 接近但未超过第三阶段最佳 |
| SparseGate + LeakyReLU² + Polar NS | **1.16522320** | 阶段四最佳 |

这一阶段虽然刷新了本地根目录脚本最佳结果，但改善幅度只有：

```text
1.16735413 - 1.16522320 = 0.00213093 BPB
```

这说明 records 组件不是简单的单开关增益。它们很可能依赖 CaseOps、doc boundary、TTT、parallel lane、压缩布局等其他机制共同作用。这个认识直接推动了第五阶段的方向转变：不再只在 root 脚本里移植单个组件，而是跑通 records 04-27 的系统组合。

### 2.5 第五阶段：CaseOps 与 records 04-27 stack 主线

第五阶段首先补齐 CaseOps 数据链路：

| 项目 | 结果 |
| --- | --- |
| train shards | 80 |
| val shards | 1 |
| val byte sidecar shards | 1 |
| val tokens | 9,662,502 |
| val byte sum | 29,950,979 |
| BOS count | 10,000 |
| bad BOS bytes | 0 |

根目录 `train_gpt.py` 同时修复了 `fineweb_val_*.bin` 误匹配 byte sidecar 的问题，并实现按 shifted target 对齐 byte sidecar 计算 BPB。

但第一批 root CaseOps 训练并没有改善：

| 方案 | 配置 | Roundtrip BPB | 结论 |
| --- | --- | ---: | --- |
| CaseOps 静态结构方案 | CaseOps + 第四阶段结构组合 + seq4096 | 1.17380957 | 明显差于第四阶段最佳 |
| CaseOps + records式学习率方案 | CaseOps + records-like schedule | 1.17440583 | 仅改 schedule 不能解决 |

这说明 CaseOps 不是“换数据就自动变好”，它需要和 records stack 中的结构、压缩和 TTT 配套使用。

随后转向 records 04-27 advanced stack。这里遇到多个环境问题：缺 FA3、缺 `triton.tools.tensor_descriptor`、varlen fallback 与 `torch.compile` 不兼容、缺 `lrzip`。通过 SDPA fallback、eager MLP fallback、fixed sequence loader、BOS 初始化、cu bucket warmup 跳过和安装 real `lrzip`，最终跑通了可用主线。

关键结果如下：

| 分支 | no-TTT BPB | post-TTT BPB | bytes | 结论 |
| --- | ---: | ---: | ---: | --- |
| 初始合规 records 方案 | 1.09951341 | 1.08464351 | 15,925,658 | real `lrzip` 让 embed7/top3 合规 |
| seed42 records 重训方案 | 1.09686294 | 1.08218120 | 15,925,323 | seed42 明显更好 |
| warmdown 0.95 fallback 方案 | 1.09647097 | 1.08179851 | 15,924,510 | 旧环境最佳 |
| warmdown 0.95 FA3/TensorDescriptor 复跑 | 1.09043610 | 1.07615418 | 15,917,614 | 严格 R8 基线 |
| partial RoPE zlf2 | **1.08972570** | **1.07586081** | **15,917,703** | 当前最佳 |

围绕最终方案的后续 sweep 表明：

- TTT 微参已经平台，迁移来的 4-phase 高学习率设置反而略差于默认 TTT。
- warmdown 0.90/0.98/0.94 都不如 0.95。
- seed0/1234 不如 seed42。
- delayed loop、batch917k、min_lr 抬高均未刷新。
- 最终方案 checkpoint 的 export-only 微扫也未超过原始导出配置。

因此 Phase 5 的判断先是：主要提升来自 records 系统栈，而旧环境下的瓶颈主要是单卡 fallback 环境中的训练质量和吞吐，不是 TTT 或导出微参。随后补齐 FA3 和 TensorDescriptor 后，这个判断得到直接验证：严格 R8 复跑从 post-TTT `1.08179851` 进一步降到 `1.07615418`；partial RoPE zlf2 又小幅刷新到 `1.07586081`。

## 3. 大量扫参探索与淘汰路线

为了避免报告呈现出“事后上帝视角”，这里单独整理探索过程中被尝试、比较、淘汰的分支。它们大多没有成为最终方案，但每一类都帮助回答了一个问题：当前瓶颈到底在训练、结构、量化、压缩、TTT，还是系统环境。

### 3.1 量化压缩扫参：先确认“压得下且保质量”

第二阶段并不是一开始就确定 GPTQ/LQER 是正确答案。最初的低比特路线经历了多轮比较：

| 探索方向 | 观察结果 | 得到的判断 |
| --- | --- | --- |
| 朴素 RTN 量化 | 可以合规，但 BPB 明显弱于 GPTQ | 只做四舍五入不够，需要考虑量化误差对输出的影响 |
| matrix int6 / embed8 | 大矩阵压缩明显，embedding 质量还能保住 | embedding 比普通矩阵更敏感，不能随意降 bit |
| embed7 尝试 | 早期质量损失明显，后续需依赖更强压缩和 records stack 才可用 | embed bit 是容量和质量的关键旋钮 |
| LQER top-k/rank | top3/rank4 明显有效；继续加 top-k/rank 多数收益变小或变大 | 误差修正有甜点，不能无限加 |
| GPTQ calibration / error scale | 小范围能带来微小刷新；后期收益进入 `1e-5` 到 `1e-4` BPB | 导出微扫适合收尾，不适合关闭大 gap |
| fresh roundtrip eval | 修正了量化评估污染风险 | 评估链路可信比单次分数更重要 |

这条线的意义是建立了一个可靠基座：SP8192 + seq4096 可以通过 GPTQ/LQER 压进 16MB，并且 roundtrip BPB 确实优于普通小词表 baseline。没有这个基座，后面的训练动态和结构实验都无法公平比较。

更具体地说，第二阶段经历了三次明显的判断修正。

第一次修正是：**早期 GPTQ 结果不能直接相信**。当时直接拿较早 checkpoint 做 export-only 量化，出现了非常反常的结果：matrix6/embed8 的 packed GPTQ 只有 `1.2888`，embed7 甚至退化到 `1.5161`，matrix7/embed8 和 matrix8/embed8 也都在 `1.60` 左右。这些分数明显不符合“更高 bit 应该更稳”的直觉。后续排查发现，旧 checkpoint 与当前代码存在 eval 不一致，且 Hessian collection 后复用同一个模型对象会污染 roundtrip eval。因此后来引入 fresh compiled roundtrip：量化后重新创建模型、重新加载 artifact、重新 compile，再评估。这个修正比某个量化参数本身更重要，因为它让后续所有比较重新可信。

第二次修正是：**RTN 作为兜底可用，但上限不够**。在 GPTQ 还不可信时，packed RTN + brotli 给出了合规结果，roundtrip 约 `1.1892`；再对少数敏感矩阵保留更高 bit，可以小幅推到 `1.1876-1.1871`。这条线证明了 bit packing 和 mixed bitwidth 方向是对的，但 RTN 的质量上限明显低于目标。

第三次修正是：**GPTQ 的收益需要在 current-code checkpoint 上逐步打开**。在 fresh eval 修正后，Hessian error scale 从 0 到 1 的 sweep 呈现出合理趋势：从接近 RTN 的 `1.1878`，逐步降到 `1.1844`、`1.1821`、`1.1808`、`1.1803`。再加入 LQER 后，旧 seq2048 基座达到 `1.1798`。随后补做的 SP8192 seq4096 基座进一步验证：纯 GPTQ 达到 `1.1800`，GPTQ + LQER top3 rank4 达到 `1.1766`。这才真正建立了第二阶段的可信里程碑。

### 3.2 训练动态扫参：小收益叠加，而不是单个神奇参数

第三阶段尝试了很多训练动态参数。最终有效路径是 QK gain、hparam stack、tied embedding LR、recurrence、Muon momentum 的叠加，但这不是一开始就知道的。

| 探索方向 | 观察结果 | 是否保留 |
| --- | --- | --- |
| QK gain | `QK_GAIN_INIT=5.0` 首次带来稳定小正收益 | 保留 |
| Adam beta / grad clip / LR stack | 改善短训稳定性 | 保留 |
| tied embedding LR | `TIED_EMBED_LR=0.04` 带来显著收益 | 保留 |
| Muon momentum | `0.97` 与 recurrence 组合效果更好 | 保留 |
| recurrence start | 训练后段开启有效，过早/过晚都不如 start 0.30 左右 | 保留窄窗口 |
| 更深/更早 recurrence | step 效率和量化鲁棒性下降 | 淘汰 |
| coprime loader | 明显负收益 | 淘汰 |
| warmdown/min_lr 早期尝试 | 在根目录训练栈中多组负收益 | 暂停，等换 stack 后再看 |
| 简化 lm-head LoRA TTT | 退化严重 | 淘汰，不能代表 legal phased TTT |

这一阶段最重要的体会是：训练动态优化像“调乐器”，不是一个参数解决所有问题。每个有效改动都只推进一点，但组合后形成第三阶段的可靠局部最优。

`PLAN.md` 里第三阶段最初并不只规划训练超参，而是分成多条并行探索线：

| 探索线 | 原始问题 | 实验后的认识 |
| --- | --- | --- |
| 量化余量细扫 | LQER top-k、rank、error scale 是否还能继续压低 BPB | 有少量收尾收益，但很快平台，不能作为主线 |
| embedding bit/clip | embed7 是否可以释放容量给结构 | 大词表 embedding 太敏感，过早降 bit 会损失质量 |
| 训练超参栈 | records 的 beta、clip、LR 是否可迁移 | 部分可迁移，尤其 tied embedding LR 和稳定性参数 |
| warmdown/min_lr | records 的后段学习率策略是否可直接用 | 在当时根目录栈中负收益，先暂停 |
| coprime loader | 更复杂的数据步幅是否提高 batch 多样性 | 实测负收益，说明 loader 改动容易破坏局部上下文 |
| 深度递归 | 共享权重重复执行能否提升有效深度 | 有效，但只在 L3-L5、后段开启的窄窗口有效 |
| eval-only TTT | 能否先用简化 TTT 验证收益 | lm-head-only 版本失败，说明必须重做 legal/phased TTT |
| CaseOps feasibility | 是否应提前切 CaseOps | 当时数据链路不完整，延后到第五阶段 |

这个矩阵让第三阶段不是简单“调几个超参”，而是逐步区分：哪些是低成本可迁移收益，哪些必须等数据/TTT/records stack 完整后再判断。

### 3.3 结构组件扫参：单项移植大多失败

第四阶段表面上只得到一个小幅刷新，但它的探索价值很大：它证明 records 里的结构组件不能机械拆开。

| 探索方向 | 观察结果 | 结论 |
| --- | --- | --- |
| LeakyReLU² 单项 | 明显变差 | 激活函数不能孤立替换 |
| partial RoPE | 明显变差 | 当前训练栈不适配 |
| Polar NS 单项 | 质量损失大于稳定性收益 | 需要和结构共同调 |
| SmearGate 单项 | 没有超过第三阶段最佳 | 单独平滑局部表示不够 |
| SparseGate 单项 | 接近但不够 | 有信号，但不是完整答案 |
| SparseGate + LeakyReLU² + Polar NS | 小幅刷新到 `1.16522320` | 组合有局部收益 |
| 导出微扫 | 多数组合只在 `1e-5` BPB 量级波动 | 第四阶段附近已经平台 |

如果只看最终结果，这一阶段似乎“收获不大”；但它实际上给第五阶段提供了非常重要的判断：records 方法必须以系统 stack 形式复现，不能只移植几个看起来有名的组件。

第四阶段在计划中其实包含更多方向，不只是最后跑出来的 gate/activation/optimizer 组合。它的原始目标是把 SOTA gap 拆成几类机制逐个验证：

| 方向 | 当时计划 | 实际收获或暂停原因 |
| --- | --- | --- |
| CaseOps 数据链路 | 先恢复 tokenizer、raw docs、byte sidecar、BOS 边界 | 第四阶段末完成 raw docs 和 smoke，但 full shards 与 byte-sidecar BPB 到第五阶段才补齐 |
| 真 phased TTT | doc-level score-before-update，Q/K/V/O/MLP/head adapters | 简化 TTT 已失败，正版 TTT 依赖 CaseOps/doc boundary，因此延后 |
| LeakyReLU² / SmearGate / SparseGate | 单项完整 1h 训练，确认哪些能叠加 | 单项大多失败，组合小幅有效 |
| partial RoPE / FoPE | 低成本位置编码变体 | partial RoPE 明显负收益，未继续扩展 |
| parallel lane / 11L / MLP4x | 更接近 records 结构，但容量和吞吐风险高 | 在 root 栈上直接上风险太大，等待 CaseOps/压缩主线 |
| per-group / `lrzip` 压缩 | 只有新结构超 16MB 时才作为容量工具 | 第四阶段还不需要；第五阶段 records artifact 超限后变成关键 |
| FA3 / fused kernel | 服务于 doc-boundary/TTT/复杂结构 | 固定 seq 训练暂时够用，真正 blocker 到第五阶段暴露 |

因此，第四阶段不是“只做了 SparseGate”。更准确地说，它先尝试单组件移植，发现收益太小；同时完成 CaseOps 数据恢复的前置工作，并把 TTT、parallel lane、per-group compression 这些需要系统条件的方向推迟到第五阶段。

### 3.4 第五阶段前半：根目录备份线和容量线

第五阶段一开始并没有直接锁定最终 records 方案。为了判断根目录训练路线是否还有潜力，我们并行探索了普通 SP8192 的训练动态、结构容量和 CaseOps 静态训练。

| 分支 | 代表结果 | 判断 |
| --- | ---: | --- |
| 普通 SP8192 + records-like 学习率/warmdown | 1.16649737 | pre 有小信号，但 roundtrip 不如第四阶段 |
| 普通 SP8192 + 786k batch | 1.16474115，导出微调到 1.16465057 | 第五阶段第一个小刷新，但仍是根目录路线的小收益 |
| 普通 SP8192 + 917k batch | 1.16357912 | 根目录路线最好的小刷新，但已被 records 主线远超 |
| seq8192 | 1.17215222 | 长上下文过长，step 损失大于收益 |
| 10 层模型 + int5/embed8 | 1.16432394 | 合规但收益有限 |
| MLP3 加宽模型 | 1.15317521，但 16,806,331 bytes | 质量强，容量不合规 |
| MLP3 容量修复 | embed7/no-LQER 仍超限，embed6 合规但退化到 1.16583102 | 只靠 bit/LQER 旋钮修不回来 |
| 根目录 CaseOps 静态训练 | 1.17380957 / 1.17440583 | CaseOps 不能脱离 records 结构和 TTT 单独生效 |

这一大组实验让逻辑更完整：我们不是因为预先知道 records 线强才放弃根目录路线，而是因为根目录路线在 batch、seq、层数、MLP、CaseOps、导出细扫上都只剩小收益或容量冲突，才把主要资源转向 records stack。

### 3.5 records 合规化探索：真正卡住的是压缩和环境

records 04-27 stack 不是一跑就成功。它先暴露出多个工程和合规问题：

| 问题 | 处理 | 得到的判断 |
| --- | --- | --- |
| 缺 FA3 | 用 torch SDPA fallback | 能跑，但吞吐和 varlen 能力受损 |
| varlen eval 与 compile 冲突 | eval fallback 改 eager，训练退 fixed sequence loader | 先保证正确性，再讨论速度 |
| 缺 TensorDescriptor/fused MLP | LeakyReLU² MLP 走 eager fallback | 质量可评估，但不是理想环境 |
| 缺 `lrzip` | 安装 real external `lrzip` | 这是 embed7/top3 合规的关键 |
| fallback brotli/lzma 压缩 | embed7/top3 超 16MB | fallback 压缩不足以支撑最佳 artifact |
| embed6 缩小 artifact | 合规但 BPB 变差 | 不能简单牺牲 embedding bit |
| embed7 + top2 / adaptive brotli-lzma | 仍超限或无改善 | LQER top-k 不是主要容量来源 |

这组探索说明，Phase 5 的“真实优化”首先是让高分 stack 合规、可运行、可评估。real `lrzip` 并不改变模型数学能力，但它让更高质量的 embed7/top3 artifact 装进 16MB，因此成为关键拐点。

后续又专门补了环境修复复跑，以确认 fallback 不是被“偷偷接受”的长期方案。新建 `pg291` 环境后，`torch 2.9.1+cu128`、FA3 `flash_attn_interface`、`triton.tools.tensor_descriptor` 和 `lrzip` 均可用；训练命令加入 `REQUIRE_FA3=1 REQUIRE_TENSOR_DESCRIPTOR=1`，确保依赖缺失时直接失败。严格 R8 复跑日志确认 `train_loader:DocumentPackingLoader flash_attn_interface:True`，不再退到 fixed sequence loader。

| 复跑项 | 旧 R8 fallback | 新 R8-pg291 |
| --- | ---: | ---: |
| 训练步数 | 3202 | 3893 |
| pre-EMA post-train BPB | 1.08798009 | 1.08176476 |
| no-TTT / roundtrip BPB | 1.09647097 | **1.09043610** |
| post-TTT BPB | 1.08179851 | **1.07615418** |
| bytes | 15,924,510 | **15,917,614** |

这个复跑说明，FA3/doc-packing/fused MLP 不是单纯加速开关，而是通过增加 1 小时内的有效训练步数和恢复 records 数据组织方式，实质改善最终 artifact。

在 R8-pg291 基础上又补了一组 partial RoPE 小扫。脚本新增 `ROPE_ZERO_LOW_FREQS` 开关，默认 `0` 完全保留旧行为；当 `ROPE_DIMS=16` 时，`ROPE_ZERO_LOW_FREQS=2` 会把 8 个 RoPE `inv_freq` 中最低频的两个尾部元素置为 `0`。

| 变体 | steps | pre-EMA post-train BPB | no-TTT / roundtrip BPB | post-TTT BPB | bytes | 结论 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `ROPE_DIMS=16 ROPE_ZERO_LOW_FREQS=1` | 3909 | 1.08151851 | 1.09014656 | | 15,917,111 | 小幅好于 R8-pg291 no-TTT |
| `ROPE_DIMS=16 ROPE_ZERO_LOW_FREQS=2` | 3928 | **1.08117850** | **1.08972570** | **1.07586081** | 15,917,703 | 当前最佳 |
| `ROPE_DIMS=32 ROPE_ZERO_LOW_FREQS=1` | 3906 | 1.08258273 | 1.09116293 | | 15,917,904 | 变差 |

`zlf2` 相对 R8-pg291 no-TTT 改善约 `0.00071 BPB`，TTT 后仍保留约 `0.00029 BPB` 的净收益。这个收益很小，但它是在 `REQUIRE_FA3=1 REQUIRE_TENSOR_DESCRIPTOR=1`、DocumentPackingLoader、SmearGate 和同一 CaseOps/quantization 配置下得到的真实刷新。

### 3.6 TTT 微扫：确认收益存在，也确认平台存在

初始合规 records 方案跑通后，TTT 带来约 `0.0149 BPB` 的大收益。接下来很自然会问：能不能靠 TTT 参数继续下降？因此做了 prefix docs、phase 数、global lr 的邻域扫参。

| TTT 方向 | 观察结果 | 判断 |
| --- | --- | --- |
| 2 phases vs 3 phases vs 4 phases | 4 phases 略好，2 phases 较差 | phase 数有影响，但幅度很小 |
| prefix docs 2000/2250/2500/2750/3000 | 2500 附近较好，3000 变差 | prefix 太长不一定好 |
| global lr 0.0007/0.0013/0.0015 | 0.0013/0.0015 略好 | 学习率有微小甜点 |
| 5 phases | 没有超过 4 phases | 更多 phase 不自动更好 |
| 最佳 TTT 邻域 | 只比默认 TTT 好约 `5e-5 BPB` | TTT 大框架有效，微参已平台 |
| 迁移到更强训练 artifact | 几乎无收益，甚至略差 | 更强 artifact 上不需要继续 TTT 小扫 |

这个结果改变了后续优先级：TTT 已经贡献了主要收益，但继续扫 TTT 微参不再是高杠杆。下一步应该改善 no-TTT artifact，也就是训练起点。

### 3.7 records 训练质量扫参：从 seed 到 warmdown

当 TTT 小扫确认平台后，实验转向 records artifact 本身的训练质量。这里做了 seed、layer loop、batch、warmdown、min_lr、导出配置等多组验证。

| 探索方向 | 观察结果 | 判断 |
| --- | --- | --- |
| seed42 / seed0 / seed1234 | seed42 明显最好 | 随机初始化和训练轨迹在短训中很重要 |
| 延后 layer loop 到 0.50/0.65 | 步数变多，但质量不如 seed42 默认路线 | 更多 step 不等于更低 BPB |
| 延后 layer loop 到 0.80/0.90 | 明显变差，0.90 附近甚至崩塌 | 太晚启用 loop 会破坏建模质量 |
| batch 917k | post-TTT 变差 | records stack 不复用根目录路线的 batch 结论 |
| warmdown 0.95 | no-TTT 和 post-TTT 都刷新 | 当前训练侧最大正信号 |
| warmdown 0.90 / 0.98 / 0.94 | 都不如 0.95 | 0.95 是窄甜点 |
| warmdown 0.95 跨 seed | seed0/1234 仍不如 seed42 | 最终方案依赖 seed42 轨迹 |
| 提高 min_lr | 没有超过 0.95 默认方案 | warmdown 0.95 的收益不是来自简单保持高学习率 |
| 最终 checkpoint 导出微扫 | error scale、calib batch、top-k 都变差 | 原导出已经接近局部最优 |

这组实验特别重要，因为它避免了误判：中途延后 loop 看起来提升吞吐，但最终 BPB 更差；TTT 迁移看起来应该有收益，但实际几乎没有；min_lr 看起来可能解释 warmdown 收益，但实验否定了它。最终保留下来的不是“看起来合理”的参数，而是实际 roundtrip/post-TTT 都最好的 warmdown 0.95 + seed42。

### 3.8 这些淘汰实验如何补全逻辑链

整体看，扫参和淘汰实验承担了四个作用：

1. 证明 SP8192 + seq4096 是值得压缩的强基座，而不是盲目扩大模型。
2. 证明根目录路线仍能小幅改善，但无法解释和 records 的巨大差距。
3. 证明 CaseOps、TTT、records 结构必须配套，不能单独简化。
4. 证明最终方案的关键不是导出微参或 TTT 微参，而是 records 训练质量、真实压缩器和运行环境。

因此，最终路线并不是一开始就知道的答案，而是在大量负结果逐步排除之后留下来的路径。

## 4. 关键方法对比与作用机制

为了更清晰地总结探索过程，可以把有效方法分为四类。

### 4.1 数据表示方法

SP8192 和 CaseOps 都属于数据表示层面的改变。SP8192 通过更大词表降低 token 序列难度，但增加 embedding 体积；CaseOps 则进一步引入 byte sidecar 和 doc boundary，使 BPB 计算与 TTT 更合规。

本项目的经验是：SP8192 单独就有明显质量收益，但必须靠 GPTQ/LQER 等方法压缩；CaseOps 单独用于 root 静态训练并不好，但和 records stack 配合后成为必要基础。

### 4.2 模型结构方法

QK gain、recurrence、SparseGate、LeakyReLU²、parallel lane、MLP4 等都改变模型 forward 过程。它们的共同目标是提高参数效率，让 16MB 内的模型拥有更强表达能力。

不过实验也说明，结构方法往往强依赖训练动态和量化方式。比如 LeakyReLU²、Polar NS、SparseGate 单项都不强，但组合后在 Phase 4 有小收益；records stack 中这些组件与 CaseOps、TTT 和压缩布局共同作用，才产生大幅跃迁。

### 4.3 训练动态方法

Muon momentum、tied embedding LR、warmdown、min_lr、batch tokens、seed 都属于训练动态。它们不一定改变模型参数量，却会改变模型在 1 小时内走到哪个解。

最典型的例子是：

- 第三阶段中 `TIED_EMBED_LR=0.04` 是根目录训练路线的大收益点。
- 第五阶段中 seed42 明显优于 seed0/1234。
- `WARMDOWN_FRAC=0.95` 在 records 线刷新到最终方案，但根目录脚本早期 warmdown 并不是主线。

这说明训练动态没有脱离模型结构的“通用最优值”，必须在具体 stack 上重新验证。

### 4.4 量化压缩与 TTT

GPTQ、LQER、mixed bit、bit packing、`lrzip` 决定了模型能否在 16MB 内保留质量。TTT 则在评估阶段继续利用当前文档信息进行合法适配。

本项目中：

- GPTQ/LQER 将 SP8192 从“质量好但超限”变成可提交主线。
- real `lrzip` 让初始合规 records 方案的 embed7/top3 artifact 从超限变为合规。
- legal phased TTT 给 warmdown 0.95 最终方案带来约 `0.01467 BPB` 的收益。

但这些方法也有平台期：最终方案 checkpoint 的导出微扫没有改善，围绕 phased TTT 的 prefix、phase 数和学习率微扫也几乎没有继续收益。

## 5. 当前最佳结果分析

partial RoPE zlf2 最终方案的结果可以拆成两部分理解：

```text
no-TTT:   1.08972570
post-TTT: 1.07586081
gain:     0.01386489 BPB
```

no-TTT 代表压缩 artifact 本身的质量；post-TTT 代表 legal TTT 后的最终质量。最终方案的 TTT 收益已经接近 records 中 TTT 的量级，因此当前差距主要不在 TTT 是否有效，而在 no-TTT 起点。

与 04-27 records mean `~1.06108` 相比，最终方案仍落后约：

```text
1.07586081 - 1.06108 ~= 0.01478 BPB
```

造成这个 gap 的主要原因包括：

1. no-TTT 起点仍高于 04-27 reference。partial RoPE zlf2 no-TTT 是 `1.08972570`，而 04-27 records post-quant 约在 `1.073-1.075` 区间。
2. 环境 blocker 已修复第一层，但计算预算仍不同。04-27 record 是 8xH100/600s，约 4931 steps；R8-pg291 是 1xH100/1h，约 3893 steps。
3. 新环境改变了吞吐和 batch/doc-packing 形态，旧 fallback 下的部分训练动态结论需要在新环境上重新验证。
4. TTT 和导出微参已经接近平台，继续小扫预期收益很低。

## 6. 思考与体会

### 6.1 不能只看单点指标，要看完整链路

刚开始很容易把实验理解成“哪个模型结构 BPB 更低”。但随着实验推进，我发现真正的目标是完整链路最优：训练、量化、压缩、加载、TTT 每一步都可能改变最终分数。MLP3 加宽方案是最典型例子：它质量很强，但超 16MB，因此不能作为最终答案。

### 6.2 负结果同样重要

Phase 3/4/5 中有大量负结果，例如 coprime loader、partial RoPE、LeakyReLU² 单项、SmearGate 单项、delayed loop、min_lr 抬高、TTT 微扫等。它们看起来没有刷新分数，但实际上帮助缩小搜索空间，避免在低收益方向继续消耗 GPU。

### 6.3 records 方法不能机械拆开

第四阶段给我的最大教训是：records 中的组件不是孤立存在的。一个方法在 record stack 中有效，不代表单独移植到根目录训练栈就有效。SparseGate、LeakyReLU²、Polar NS 的组合只带来很小收益，而完整 records stack 才带来从 `1.16` 到 `1.08` 的跃迁。

### 6.4 工程环境也是算法的一部分

Phase 5 中很多时间花在 FA3、SDPA fallback、TensorDescriptor、`lrzip`、byte sidecar 对齐等问题上。这让我意识到，高分方案往往不是论文式的单个公式，而是算法、kernel、压缩器和数据格式共同构成的系统。缺少某个环境组件时，同样的代码可能只能跑 fallback，最终训练质量也会下降。环境修复复跑把这个判断从推测变成了实证：同一 R8 配置从 `1.08179851` 下降到 `1.07615418`；后续 partial RoPE zlf2 再下降到 `1.07586081`。

### 6.5 局部微调有边界，系统切换才带来跃迁

第三、四阶段的局部优化很有价值，因为它建立了可靠 baseline，也让实验链路可信。但当第四阶段结构组合附近的 export-only 收益进入 `1e-5` 到 `1e-4` BPB 级别时，继续微扫意义不大。第五阶段切换到 CaseOps + records stack 后，才出现主线级提升。

## 7. 后续工作

基于当前结果，我认为下一步优先级应当是：

1. 固化 R8-pg291 最终方案的复现链路，确保 `pg291` 环境、CaseOps sidecar、`REQUIRE_FA3=1 REQUIRE_TENSOR_DESCRIPTOR=1`、real `lrzip`、默认 TTT 和 artifact bytes 可稳定复现。
2. 在新环境上补最小矩阵：seed0/1234、warmdown 0.90/0.98/0.94，判断旧 fallback 下的 seed/warmdown 排序是否保持。
3. 重新验证 batch 和 layer-loop 时机。新环境下 step 数、doc-packing 和吞吐形态已改变，旧的 batch917k/延后 loop 负结果值得小规模复验。
4. 在得到更强 no-TTT artifact 之前，不再大量投入 TTT 微扫或 export-only 微扫。

## 8. 结论

本项目从 baseline 出发，经历了从局部调参到系统复现的探索过程。前四阶段证明了 SP8192、大上下文、GPTQ/LQER、QK gain、tied embedding LR、recurrence、SparseGate/LeakyReLU²/Polar NS 等方法的局部价值，但也暴露了根目录局部优化路线的平台期。第五阶段补齐 CaseOps 合规链路并跑通 records 04-27 stack 后，先将 best 从第四阶段的 `1.16522320` 推进到 fallback R8 的 post-TTT `1.08179851`；环境修复后严格 R8 进一步刷新到 `1.07615418`；partial RoPE zlf2 最终刷新到 `1.07586081`。

这一结果说明，在 16MB 小模型挑战中，真正有效的优化不是单点技巧，而是数据表示、模型结构、训练动态、量化压缩、运行环境和测试时自适应的联合设计。当前剩余差距主要来自单卡训练预算、新环境下训练动态尚未重扫，以及与 04-27 reference 的并行形态差异，而不是某个简单超参尚未扫到。
