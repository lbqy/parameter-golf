# Parameter Golf 优化方法详解

日期：2026-06-08

本文面向“略懂大模型、但还不熟悉 parameter-golf 工程细节”的读者。目标不是复述所有实验流水账，而是解释：每个优化方法是什么、作用在训练/导出/评估的哪个阶段、为什么会影响 BPB，以及本项目实际怎么用、结果如何。

说明：报告里有时会看到 `BPB`、`val_bpb`、`roundtrip BPB`。如果你写的是 `PBP`，这里统一按 `BPB` 理解。BPB 是 bits per byte，越低越好。

## 1. 先理解这个比赛在优化什么

这个项目不是普通语言模型训练。普通训练常问：“验证集 loss 能不能更低？”Parameter Golf 还多两个硬约束：

1. 最终提交物必须小于 16 MB。
2. 评估看的是压缩 artifact 解包加载后的模型，而不是训练内存里的原始模型。

所以一个方法只有在下面三关都成立时才算真的有用：

| 阶段 | 看什么 | 为什么重要 |
| --- | --- | --- |
| 训练阶段 | pre-quant BPB / val loss | 原始模型学得好不好。 |
| 量化导出阶段 | roundtrip BPB | 低比特量化、打包、压缩、重新加载后，质量还剩多少。 |
| 合规阶段 | artifact bytes | 文件能不能压进 16 MB。 |

本项目当前最好的结果是 Phase 5 R8：

| 项目 | 值 |
| --- | --- |
| RUN_ID | `exp_s5_r8_seed42_warmdown095_records0427_caseops_lrzip_1xh100` |
| no-TTT / roundtrip BPB | `1.09647097` |
| post-TTT BPB | `1.08179851` |
| total bytes | `15,924,510` |
| 主要来源 | CaseOps + records 04-27 stack + real `lrzip` + seed42 + `WARMDOWN_FRAC=0.95` + phased TTT |

一个很重要的直觉是：BPB 不是“参数越多越低”。参数多可能降低 pre-quant BPB，但也会让 artifact 超过 16 MB；低比特量化能压小文件，但可能破坏模型质量，让 roundtrip BPB 变差。这个项目的难点就在这里。

## 2. BPB、pre-quant、roundtrip、post-TTT 是什么

### 2.1 BPB

语言模型训练通常最小化 token-level cross entropy。可是不同 tokenizer 的 token 长度不一样，用 token loss 比较 SP1024、SP8192、CaseOps 不公平。BPB 把模型在验证集上的负对数似然换算成“预测每个原始 byte 需要多少 bit”。

直觉上：

- BPB 低，说明模型对原始文本 byte 的压缩预测更好。
- 同样一段文本，如果 tokenizer 变了，token 数可能变，但原始 byte 数不变，所以 BPB 更适合作为跨 tokenizer 指标。
- 本项目最终主指标是 `val_bpb`。

### 2.2 pre-quant BPB

pre-quant BPB 是训练结束后，直接用训练出来的浮点/bfloat16 权重评估得到的 BPB。

它回答的是：“模型本身学得怎么样？”

但它不是最终分数，因为提交不能带完整浮点权重。一个 pre-quant 很好的大模型，如果压不进 16 MB，或者量化后严重退化，就不是好方案。

### 2.3 roundtrip BPB

roundtrip BPB 是最终 artifact 经过“量化 -> 打包 -> 压缩 -> 解压 -> 反量化/加载 -> 重新评估”之后的 BPB。

它回答的是：“提交评测真正看到的模型怎么样？”

本项目绝大多数比较都以 roundtrip BPB 为准。比如 Phase 4 一些配置 pre BPB 看起来更好，但量化后收益被吃掉，roundtrip 反而差。

### 2.4 post-TTT BPB

TTT 是 test-time training，也就是评估时允许模型在合法规则下对当前验证/测试文档做少量在线适配。post-TTT BPB 是执行这套适配后的 BPB。

它回答的是：“如果评估阶段也使用合法自适应，最终能到多少？”

Phase 5 R8 的 no-TTT 是 `1.09647097`，post-TTT 是 `1.08179851`，TTT 大约贡献 `0.01467 BPB`。

## 3. 整体流程：方法分别插在哪里

可以把一次实验看成这条流水线：

```text
raw docs
  -> tokenizer / token shards / byte sidecar
  -> train loader 产生 x,y
  -> model forward 预测下一个 token
  -> loss backward
  -> optimizer 更新参数
  -> LR schedule 控制训练节奏
  -> pre-quant eval
  -> quantization / LQER / packing
  -> compression
  -> roundtrip eval
  -> optional legal TTT
  -> post-TTT eval
```

不同优化方法插入的位置不同：

| 类别 | 作用阶段 | 代表方法 |
| --- | --- | --- |
| 数据和 tokenizer | 训练前、验证 BPB 计算 | SP1024/SP8192、CaseOps、byte sidecar、doc boundary |
| 模型结构 | forward | 层数、MLP_MULT、GQA、RoPE、QK gain、SparseGate、recurrence |
| 优化器和学习率 | backward/update | Muon、Adam 参数组、tied embedding LR、warmdown、min_lr、grad clip |
| 训练吞吐 | loader/forward/compile | seq len、batch tokens、FA3、SDPA fallback、fused MLP |
| 量化和压缩 | 导出阶段 | RTN、GPTQ、mixed bit、LQER、bit packing、brotli/lzma/lrzip |
| 测试时自适应 | eval 阶段 | legal/phased TTT |

下面逐个解释。

### 3.1 `records stack` 到底是什么

我在前文多次写到 `records stack`，这不是一个正式算法名，也不是单个开关。它的意思是：`records/track_10min_16mb/` 里历史高分提交积累下来的一整套“组合拳”。其中我们 Phase 5 主要参考的是：

```text
records/track_10min_16mb/
  2026-04-27_SP8192_LQER_SparseGate_BOSSmearFix_9HpStack_1.0611/
```

这个 04-27 record 的官方式描述是：

```text
11L XSA + LQER + SparseAttnGate + SmearGate (BOS-fixed)
+ PolarNS Muon + 9-hparam stack
```

如果翻译成人话，它不是“用了 SparseGate 所以强”，也不是“用了 TTT 所以强”，而是把下面六层东西一起调过：

| 层 | 包含什么 | 解决什么问题 |
| --- | --- | --- |
| 数据层 | SP8192 CaseOps tokenizer、lossless caps transform、byte sidecar、BOS/doc boundary | 让 BPB 按原始 byte 合规计算，并给 doc-level TTT 提供边界。 |
| 模型结构层 | 11 层 512d、8 heads/4 KV GQA、MLP 4x、LeakyReLU²、XSA all layers、U-Net skip、partial RoPE/YaRN、parallel decoder lane、depth recurrence | 用更高参数效率和更多有效计算降低 pre-quant BPB。 |
| gate/信息流层 | Sparse attention gate、SmearGate、BOS leak fix、skip gates、parallel lane mix | 控制信息在哪些 head/通道/位置之间流动，避免容量浪费和跨文档泄漏。 |
| 训练动态层 | Polar-Express Muon、Adam hparam stack、warmdown、min_lr、grad clip、EMA、QK gain | 让 10min/1h 的短训更快更稳地收敛。 |
| 导出压缩层 | GPTQ int6、int7 embedding、int8 gate、LQER asymmetric int4 rank4 top3、per-group row reorder、`lrzip`/brotli | 把更强模型压进 16 MB，同时尽量保住 roundtrip BPB。 |
| 评估自适应层 | legal phased TTT、doc-boundary score-before-update、LoRA adapters、multi-phase global SGD | 在测试/验证阶段利用当前文档已看过的部分继续适配，降低 post-TTT BPB。 |

这就是 `stack` 这个词的含义：每一层单独看都可能只有小收益，甚至单独移植会变差；但它们一起设计时，训练分布、模型结构、量化误差、压缩布局和 TTT 行为会互相适配。

#### 3.1.1 为什么不能把 records stack 理解成“records 模型结构”

因为 04-27 record 里至少有三类东西不属于模型结构：

1. CaseOps/byte sidecar 是数据与评估链路。
2. GPTQ/LQER/`lrzip` 是导出压缩链路。
3. phased TTT 是评估阶段的在线自适应。

如果只把其中一个结构组件搬进 root 脚本，比如只开 SparseGate 或只换 LeakyReLU²，通常不会得到 record 级收益。Phase 4 已经验证过这一点：SparseGate 单项、SmearGate 单项、Polar NS 单项都没有超过 Q43；只有 SparseGate + LeakyReLU² + Polar NS 的小组合带来 `~0.00213 BPB` 的局部收益。

#### 3.1.2 04-27 records stack 的关键组件逐层解释

数据层：

- `SP8192 CaseOps tokenizer`：仍是 8192 词表大 tokenizer，但配合 CaseOps 的 lossless caps 预处理，让文本大小写/特殊形式更适合挑战。
- `byte sidecar`：每个 val token 对应原始 byte 数，BPB 直接按 sidecar 求分母。
- `BOS/doc boundary`：知道文档从哪里开始，TTT 和 SmearGate 都不能跨文档泄漏信息。

模型层：

- `11L 512d`：比 root 9 层更深，但宽度仍控制在 512。
- `MLP 4x + LeakyReLU²`：增大 FFN 容量，同时用 records 里验证过的激活。
- `XSA all layers`：一种 extra/extended skip attention 风格的结构增强，核心直觉是让每层能更好利用早期表示，不只是线性堆 block。
- `U-Net skips`：前半层的表示跳连到后半层，像 U-Net 一样保留浅层信息。
- `parallel decoder lane`：后面几层拆成两个并行 lane，attention/MLP 输出用可学习系数混合，增加表示路径。
- `depth recurrence / loop layers 3-5`：训练到一定进度后，把部分层重复执行，增加有效深度但不增加对应权重存储。

gate/信息流层：

- `Sparse attention gate`：对 attention head 输出做窄门控，用很少参数控制哪些 head 输出更该通过。
- `SmearGate`：让当前位置表示可以混入前一个 token 的少量信息，像一种可学习的局部位置平滑。
- `BOS-fixed SmearGate`：关键合规修复。普通 SmearGate 如果在 packed stream 中直接用前一个 token，会把上一个文档的最后 token 混进下一个文档 BOS 后的位置，形成跨文档泄漏；04-27 record 在 BOS 位置用 mask 阻断这个路径。

训练动态层：

- `Polar-Express Muon`：矩阵参数用更强的 Muon/NS 更新，改善短训稳定性。
- `QK_GAIN_INIT=5.0`：让 attention 初始尺度更适合这个小模型。
- `warmdown/min_lr/grad clip/EMA`：控制后段收敛、梯度稳定和最终权重平滑。
- `9-hparam stack`：04-27 README 里列了 9 个经 greedy forward-selection 验证的超参，包括 clip sigmas、`BETA2=0.99`、TTT beta/weight decay/rank、SparseGate scale、TTT prefix docs、warmdown frac 等。

导出压缩层：

- `GPTQ int6`：大矩阵 6 bit Hessian-aware 量化。
- `int7 embedding`：embedding 比普通矩阵更敏感，所以保留 7 bit。
- `LQER asymmetric int4 rank4 top3`：挑量化误差最大的 3 个张量，用低秩 int4 因子补回来。
- `per-group compression`：把相似类型的量化权重分组，热点 2D tensor 做行相似排序，再用 `lrzip` ZPAQ 压缩；剩余部分用 brotli。

TTT 层：

- `phased TTT`：把测试时适配分成多个 phase。
- `prefix docs`：先用前若干文档或文档前缀做合法更新。
- `LoRA adapters`：不是直接大改全部模型权重，而是在 Q/K/V/O/MLP/head 等位置用轻量可训练 adapter。
- `score-before-update`：先评分当前 token/chunk，再用它更新，避免偷看答案。

#### 3.1.3 本项目 Phase 5 实际用了哪些，没完整用哪些

Phase 5 不是在原样复现 8xH100/600s 的 04-27 record。我们是在单 H100/1h 环境下尽量跑通这套 stack，因此有可用部分，也有 fallback 部分。

已经实际用到并形成 R8 的部分：

- CaseOps full data + byte sidecar。
- records 04-27 脚本里的 advanced 模型/量化/TTT 主线。
- GPTQ int6 + embed7/top3 类 artifact。
- LQER/per-group 序列化逻辑。
- real external `lrzip`。
- legal phased TTT。
- seed42。
- `WARMDOWN_FRAC=0.95` 这个本地重新调出来的 warmdown。

没有完整恢复、以 fallback 形式运行的部分：

- 缺 FA3，所以 varlen attention/高吞吐 attention 没按原 record 环境跑。
- 缺 `triton.tools.tensor_descriptor`，fused LeakyReLU² MLP 走 eager fallback。
- varlen/doc-boundary compile 有问题，因此训练退到 `FixedSequenceTrainLoader`。
- 04-27 record 是 8xH100/600s，约 4931 steps、约 121.7ms/step；本项目 R8 是 1xH100/1h，约 3202 steps、约 708k tok/s final，训练质量起点明显弱。

所以，当报告里说“records stack 是主线”，准确意思是：

```text
不是继续在 root QS28 上加一个小组件，
而是改用 04-27 高分提交那套数据 + 结构 + 训练 + 量化 + 压缩 + TTT 的系统组合，
再针对本地 1xH100/1h 和 fallback 环境做修补与重调。
```

#### 3.1.4 为什么它能把结果从 1.16 拉到 1.08

Phase 4 QS28 的 best 是 `1.16522320`，它只是在 root Q43 上局部加了 SparseGate + LeakyReLU² + Polar NS，再做导出细扫。这个路线仍然缺：

- CaseOps byte sidecar/doc boundary；
- records advanced 11L/MLP4/parallel lane/depth recurrence；
- real `lrzip` 支撑的 embed7/top3 合规 artifact；
- legal phased TTT。

Phase 5 R1Q7/R8 补上这些系统部件后：

- R1Q7 no-TTT 到 `1.09951341`，说明模型/量化/压缩主线已经远强于 root QS28。
- R1Q7T 到 `1.08464351`，说明 legal phased TTT 带来约 `0.0149 BPB` 收益。
- R8 通过 seed42 + `WARMDOWN_FRAC=0.95` 把 no-TTT 推到 `1.09647097`，post-TTT 到 `1.08179851`。

这就是 `records stack` 的实质：它通过多层共同作用产生跃迁，而不是靠单个组件贡献全部收益。

## 4. 数据与 tokenizer 类优化

### 4.1 SP1024 与 SP8192 tokenizer

概念：tokenizer 把文本切成 token。`SP1024` 是 1024 词表，`SP8192` 是 8192 词表。词表越大，常见片段可以用更少 token 表示；但 embedding 矩阵也越大。

作用阶段：训练前的数据准备、模型 embedding/head 参数规模、验证 BPB 计算。

影响 BPB 的机制：

- 大词表通常让序列更短，模型预测更高层的文本片段，BPB 可能明显下降。
- 但 embedding 参数变多，在 16 MB 下更难压缩。
- 如果 embedding 被量化太狠，质量会掉很多。

本项目如何使用：

- Phase 1 证明 SP8192 明显优于 SP1024。
- SP8192 + seq2048 的质量明显更好，但 int8+zlib artifact 约 19.34 MB，超过 16 MB。
- 因此后续主线变成“保留 SP8192 质量，同时用更强量化/压缩压进 16 MB”。

实验结论：

| 配置 | roundtrip / 状态 | 结论 |
| --- | ---: | --- |
| SP1024 seq4096 | `1.21126584` | 合规但质量有限 |
| SP8192 seq2048 int8 | `1.18147653` | 质量强但超 16 MB |
| SP8192 + GPTQ/LQER | `1.17661956` | 成为 Phase 2 主线 |

### 4.2 训练序列长度 `TRAIN_SEQ_LEN`

概念：每个训练样本里模型能看到多长的上下文。seq1024 表示最多看 1024 token，seq4096 表示最多看 4096 token。

作用阶段：训练 loader、attention 计算、验证分块。

影响 BPB 的机制：

- 长上下文让模型利用更远的信息，通常能降低 BPB。
- 但 attention 计算更贵，step 更慢，同样 1 小时内 optimizer step 变少。
- 太长还可能因为 batch/显存限制改变训练动态。

本项目如何使用：

- SP1024 从 seq1024 到 seq4096 有稳定收益。
- SP8192 上 seq4096 成为 Phase 2/3 的重要基座。
- Phase 5 root CaseOps seq2048 明显差于 seq4096。
- Phase 5 O3 的 seq8192 反而变差，说明更长不是无脑更好。

结论：seq4096 是根脚本阶段比较稳的甜点；seq8192 在本地 1xH100/1h 下不划算。

### 4.3 CaseOps 数据链路

概念：CaseOps 是 records 中使用的一套数据/tokenizer/sidecar 链路。它不只是换 tokenizer，还保留了验证文档的 byte sidecar 和 BOS 信息，用于更准确地按原始 byte 计算 BPB，也为 doc-level TTT 提供边界信息。

作用阶段：训练前数据准备、验证 BPB 计算、TTT 文档边界。

影响 BPB 的机制：

- byte sidecar 让 BPB denominator 按原始 byte 对齐，而不是简单从 tokenizer 估算。
- doc boundary 让 TTT 可以按文档做合法更新，避免把未来 token 或未来文档信息泄漏到当前评分。
- CaseOps 可能降低有效预测难度，但必须和 records 结构/TTT 配套。

本项目如何使用：

- Phase 5 生成了 80 个 train shards、1 个 val shard、1 个 val byte sidecar shard。
- 验证集 token 数为 `9,662,502`，byte sidecar 合计 `29,950,979`，BOS 数 `10,000`，bad BOS bytes 为 0。
- 根目录 `train_gpt.py` 修复了 `fineweb_val_*.bin` 误匹配 `fineweb_val_bytes_*.bin` 的问题。
- sidecar 存在时，`eval_val()` 按 shifted `y` target 对齐 byte count 计算 BPB。

实验结论：

- root 静态模型直接换 CaseOps 是负收益，C5 为 `1.17380957`，差于 Phase 4 QS28 的 `1.16522320`。
- 但 records stack + CaseOps + TTT 大幅超过 root QS28，说明 CaseOps 需要和 doc-boundary/TTT/结构栈共同使用。

### 4.4 byte sidecar 与 shifted y 对齐

概念：训练时输入 `x=tokens[:-1]`，目标是 `y=tokens[1:]`。BPB 应该统计目标 token 对应的原始 byte，而不是输入 token 的 byte。

作用阶段：验证评估。

影响 BPB 的机制：

- 如果 byte count 对齐错了，BPB 分母会偏，分数不可信。
- 特别是 BOS token 通常没有真实 byte，必须保证它不污染 byte 统计。

本项目如何使用：

- Phase 5 修复 `eval_val()`，使用 shifted target 对齐 sidecar byte count。
- full verifier 检查 BOS byte 为 0。

结论：这是合规和可信评估的基础，不是直接降低模型 loss 的优化，但没有它，CaseOps 分数不能相信。

## 5. 模型结构类优化

### 5.1 模型层数 `NUM_LAYERS`

概念：Transformer block 的数量。层数越多，模型可以做更多非线性变换。

作用阶段：forward/backward、artifact 参数规模。

影响 BPB 的机制：

- 更多层通常提高建模能力，降低 pre-quant BPB。
- 但会增加计算时间，同样 1 小时内 step 变少。
- 也会增加参数，可能压缩后超过 16 MB。

本项目如何使用：

- root baseline 主要从 9 层出发。
- Phase 5 O5 试过 `NUM_LAYERS=10` + int5/embed8，结果 `1.16432394`，有效但弱于 records 线。
- records 04-27 stack 使用更 advanced 的深度/结构组合，而不是简单加一层。

结论：加层有用，但单独加层不如 records 整体结构栈；还必须配合量化容量。

### 5.2 MLP 宽度 `MLP_MULT`

概念：Transformer block 里的 MLP/FFN 通常是 `hidden_dim -> mlp_mult * hidden_dim -> hidden_dim`。`MLP_MULT=3` 比 `2` 有更多中间维度。

作用阶段：forward/backward、artifact 参数规模。

影响 BPB 的机制：

- MLP 是语言模型容量的大头，变宽通常明显降低 pre-quant BPB。
- 但 MLP 参数很多，文件体积和计算都会上升。
- 低比特量化 MLP 后，质量可能有较大损失。

本项目如何使用：

- Phase 5 O4 用 `MLP_MULT=3` + matrix int5/embed8，roundtrip 到 `1.15317521`，这是 root 线很强的质量信号。
- 但总字节 `16,806,331`，超过 16 MB。
- 后续 P1-P4 尝试 embed7、embed6、减少 LQER 等容量修复，没有得到有效更优结果。

结论：MLP3 证明“更大模型”有质量潜力，但当前压缩方式装不下。它是容量修复目标，不是当前最终提交。

### 5.3 GQA：`NUM_HEADS` 与 `NUM_KV_HEADS`

概念：Grouped Query Attention 让多个 query head 共享较少的 key/value head。例如 8 个 Q heads、4 个 KV heads。

作用阶段：attention forward/backward、参数规模和速度。

影响 BPB 的机制：

- 相比普通多头注意力，GQA 减少 K/V 参数和计算。
- 在小模型和 16 MB 限制下，它是一种用较小容量损失换更好参数效率的方法。

本项目如何使用：

- 根目录 baseline 默认 `NUM_HEADS=8`、`NUM_KV_HEADS=4`。
- 后续实验主要沿用，没有作为单独大 sweep 主线。

结论：GQA 是基础架构省参数设计，不是 Phase 5 的主要新增收益来源。

### 5.4 RoPE 与 partial RoPE

概念：RoPE 是旋转位置编码，把位置信息注入 Q/K。partial RoPE 只对一部分维度加 RoPE。

作用阶段：attention forward。

影响 BPB 的机制：

- 位置编码影响模型理解 token 顺序和长距离关系。
- partial RoPE 可能减少位置编码对某些通道的约束，但也可能损害位置泛化。

本项目如何使用：

- Phase 4 试过 `ROTARY_DIM=16` 的 partial RoPE。
- 结果 roundtrip `1.17619727`，明显负收益。

结论：本项目没有继续 partial RoPE。它不是当前路线。

### 5.5 `QK_GAIN_INIT`

概念：attention 里 Q/K 点积会决定注意力分布有多尖锐。`QK_GAIN_INIT` 是可学习 attention gain 的初值，相当于调 attention logits 的尺度。

作用阶段：model forward，尤其训练早期的 attention 动态。

影响 BPB 的机制：

- gain 太小，注意力可能过平，模型难以快速聚焦重要 token。
- gain 太大，注意力可能过尖，训练不稳定。
- 合适的初值能让 1 小时短训更快进入有效区域。

本项目如何使用：

- Phase 3 中 `QK_GAIN_INIT=5.0` 是首个明确正信号。
- QK gain 从 B3 baseline `1.17661956` 推到 `1.17610282`。

结论：这是小收益但稳定的训练动态优化，适合 root Q43 路线。

### 5.6 tied embedding 与 `TIED_EMBED_LR`

概念：tied embedding 指输入 token embedding 和输出 logits head 共用同一个矩阵。`TIED_EMBED_LR` 是这个矩阵的学习率。

作用阶段：forward logits、optimizer update、artifact 参数规模。

影响 BPB 的机制：

- 绑定 embedding 可以省掉一整个 output head，在 16 MB 下非常重要。
- 但 embedding/head 同时承担“读 token”和“预测 token”的任务，学习率很敏感。
- 大词表时 embedding 参数多，对 BPB 和量化都很敏感。

本项目如何使用：

- 根脚本多数有效路线保持 tied embedding。
- Phase 3 将 `TIED_EMBED_LR=0.04` 后有显著收益，从 H7 `1.17479819` 到 H11 `1.17163042`。

结论：这是 Phase 3 root 路线的大收益点之一。它影响的是训练更新，不改变模型结构大小。

### 5.7 ReLU squared 与 LeakyReLU squared

概念：普通 MLP 常用 GELU/SwiGLU。baseline 用 ReLU 后平方，即 `ReLU(x)^2`。LeakyReLU squared 则允许负半轴保留一个小斜率，再平方或进入类似结构。

作用阶段：MLP forward。

影响 BPB 的机制：

- ReLU squared 提供更强非线性，对小模型可能参数效率高。
- Leaky 版本让负激活不完全归零，可能改善梯度流。
- 但激活分布变化会影响量化鲁棒性。

本项目如何使用：

- Phase 4 单独 LeakyReLU^2 是负收益，S4-G1 roundtrip `1.17263005`。
- LeakyReLU^2 + Polar NS pre BPB 看起来好，但 roundtrip 变差到 `1.16956051`。
- SparseGate + LeakyReLU^2 + Polar NS 组合最终在 Phase 4 有效，QS28 到 `1.16522320`。

结论：LeakyReLU^2 不是单独有效开关，需要和 gate/optimizer 动态配合。

### 5.8 SparseGate、SmearGate、QuantGate

概念：gate 是在网络里控制信息流的机制。可以理解成模型学会“哪些通道/分支该多用，哪些少用”。SparseGate 偏稀疏选择，SmearGate 偏把信号平滑扩散，QuantGate 更考虑量化后的门控行为。

作用阶段：model forward，也间接影响量化分布。

影响 BPB 的机制：

- gate 可以提高参数效率，让模型把有限容量用在更重要的特征上。
- 稀疏或平滑的激活分布可能更容易压缩，也可能更难量化。
- 如果只把 gate 单独塞进旧结构，训练动态可能不匹配。

本项目如何使用：

- Phase 4 单项 SmearGate、SparseGate 都没有超过 Q43。
- SparseGate + LeakyReLU^2 + Polar NS 组合有效，Phase 4 best 为 `1.16522320`。
- records 04-27 stack 中 gate 类结构和 CaseOps/TTT/压缩布局共同出现，是 Phase 5 大幅提升的一部分。

结论：gate 类方法更像“系统栈组件”，不是即插即用的小魔法。

### 5.9 recurrence / layer loop

概念：recurrence 是让某些层在训练后段或 forward 中重复执行，相当于用相同参数增加有效深度。layer loop 是 records 里更强的类似思想。

作用阶段：model forward，训练后段或特定阶段启用。

影响 BPB 的机制：

- 参数不增加太多，但有效计算深度增加，可能降低 BPB。
- 代价是每 step 更慢。
- 如果太早启用，早期训练会变慢或不稳定；太晚启用，模型来不及适应。

本项目如何使用：

- Phase 3 轻量递归 L3-L5 在训练后 30% 左右开启，有效降低 BPB。
- R15 的 recurrence start 0.30 比更早/更晚更好。
- Phase 5 records delayed loop 分支 R4-R6/R10/R11 结果不好：有些配置步数更多，但质量更差，R6/R11 甚至崩得很明显。

结论：root Q43 中轻量 recurrence 有窄窗口收益；records stack 中延后 loop 不是当前单卡 fallback 的答案。

### 5.10 parallel residual / decoder lane

概念：parallel residual 是让 attention、MLP 或额外 decoder 分支并行贡献残差，而不是完全串行堆叠。decoder lane 可以理解成额外的信息处理通道。

作用阶段：model forward、参数布局、压缩布局。

影响 BPB 的机制：

- 并行分支增加表示能力，可能比单纯加层更高效。
- 但会增加参数和计算，也需要专门压缩策略。

本项目如何使用：

- 根目录 Phase 4 没有完整复现这类 records 组件。
- Phase 5 records 04-27 stack 中包含相关 advanced 结构，是 R1Q7/R8 大幅超过 root QS28 的一部分。

结论：这是 records 主线的重要组件，但本项目目前主要通过 records 脚本使用，而不是在 root 脚本里逐个 ablation。

## 6. 优化器与训练动态

### 6.1 Adam 参数组

概念：Adam 是常见自适应优化器。根脚本把 embedding、小标量参数、head 等分到不同学习率。

作用阶段：backward 之后的参数更新。

影响 BPB 的机制：

- 不同参数类型的梯度尺度和学习难度不同。
- embedding/head 对 token 预测很直接，学习率过大可能不稳，过小学不动。
- scalar/gain/skip 这类控制参数少但影响全局动态。

本项目如何使用：

- Phase 3 的 hparam stack 调整了 Adam beta、clip 等训练动态。
- `TIED_EMBED_LR=0.04` 是重要正收益。

结论：在小模型短训里，优化器参数组常常比大模型训练里更敏感。

### 6.2 Muon optimizer

概念：Muon 是一种面向矩阵参数的优化器，会对更新方向做类似正交化/归一化的处理。直觉上，它希望大矩阵权重的更新更稳定、更“形状友好”。

作用阶段：backward 之后，主要更新 2D 矩阵参数，比如 attention 和 MLP 权重。

影响 BPB 的机制：

- 让矩阵更新方向更规整，可能提高短时间训练效率。
- momentum 和 Newton-Schulz 步数会影响稳定性和速度。
- 过强或过慢会损失 wallclock 内可完成 step。

本项目如何使用：

- baseline 已经使用 Muon 更新 transformer 2D 矩阵。
- Phase 3 将 `MUON_MOMENTUM=0.97` 与 recurrence 叠加，进一步刷新到 `1.16835283`。
- Phase 4 的 Polar NS 与 Muon/矩阵更新动态有关。

结论：Muon 是根脚本强 baseline 的底层训练方法；momentum 微调带来小但真实收益。

### 6.3 Polar Newton-Schulz / Polar NS

概念：Newton-Schulz 是一种迭代近似矩阵正交化/极分解的方法。Polar NS 在优化器或矩阵更新里鼓励更新方向更接近“旋转/正交”结构。

作用阶段：optimizer update。

影响 BPB 的机制：

- 对大矩阵更新做几何约束，可能提升训练稳定性和泛化。
- 但计算有成本，如果 step 变慢太多，1 小时内总训练量下降。
- 它也可能改变权重分布，影响量化。

本项目如何使用：

- Phase 4 Polar NS 单项是负收益，S4-K3 `1.17343709`。
- LeakyReLU^2 + Polar NS pre BPB 强，但量化后变差。
- SparseGate + LeakyReLU^2 + Polar NS 组合有效，成为 Phase 4 best。

结论：Polar NS 需要和结构/激活配合；单独开不一定好。

### 6.4 warmup

概念：训练开始前或开始阶段用小步预热，避免刚编译/刚初始化时的异常耗时或不稳定更新。根脚本还会做 compile warmup，然后恢复初始权重，不把 warmup 算作正式训练。

作用阶段：训练前、训练早期。

影响 BPB 的机制：

- 主要减少计时噪声和初期不稳定。
- 对最终 BPB 的直接影响通常小，但对实验可比性重要。

本项目如何使用：

- baseline 默认有 `WARMUP_STEPS`。
- Phase 5 records fallback 也处理了 cu bucket warmup 在 fixed loader 下的问题。

结论：warmup 是工程稳定性措施，不是主要收益来源。

### 6.5 warmdown 与 `WARMDOWN_FRAC`

概念：warmdown 是训练后段逐渐降低学习率，让模型从“快速学习”进入“细致收敛”。`WARMDOWN_FRAC=0.95` 可以理解成很晚才进入或调整衰减节奏，具体取决于 records 脚本实现。

作用阶段：训练后段 LR schedule。

影响 BPB 的机制：

- 后段学习率太高，模型可能在好区域附近抖动。
- 后段学习率太低或衰减太早，又可能浪费短训时间。
- 1xH100/1h 的 step 数有限，衰减时机尤其敏感。

本项目如何使用：

- Phase 3 root 线里 warmdown/min_lr 多组负收益。
- Phase 5 records 线中，R8 的 `WARMDOWN_FRAC=0.95` 是最终关键正信号。
- R12/R13/R20 的 0.90/0.98/0.94 都不如 0.95。
- R18/R19 提高 min_lr 也没有超过 R8。

结论：warmdown 是否有效强依赖训练栈。root Q43 里不是主线；records 04-27 stack 中 0.95 是当前最优。

### 6.6 `MIN_LR`

概念：学习率衰减时不降到 0，而是保留一个最低学习率比例。

作用阶段：训练后段 LR schedule。

影响 BPB 的机制：

- 保留 min_lr 可以继续探索，避免过早停住。
- 但如果模型已经需要收敛，min_lr 太高会让权重抖动，BPB 变差。

本项目如何使用：

- Phase 5 R17/R18/R19 测试了 `MIN_LR=0.2/0.15` 等邻域。
- 结果均不如 R8，说明 R8 的收益不是简单来自更高最低学习率。

结论：当前 records 单卡训练不需要继续抬高 min_lr。

### 6.7 gradient clipping 与 beta2

概念：gradient clipping 限制梯度范数，避免单步更新过大。Adam 的 `BETA2` 控制二阶矩估计平滑程度。

作用阶段：backward/update。

影响 BPB 的机制：

- clipping 可以提高稳定性，尤其在较大学习率或复杂结构下。
- `BETA2` 较低更快响应梯度变化，较高更平滑。
- 小模型短训中，响应速度和稳定性需要平衡。

本项目如何使用：

- Phase 3 hparam stack 包含 `BETA2=0.99`、clip/grad stack 等。
- 该 stack 从 QK gain 后继续降低 BPB。

结论：这些不是醒目的结构创新，但它们让短训更稳，是 Phase 3 小收益叠加的一部分。

### 6.8 seed

概念：seed 控制初始化、数据顺序等随机性。

作用阶段：训练初始化和训练轨迹。

影响 BPB 的机制：

- 在 1 小时短训里，模型可能还没有充分平均掉随机性。
- 不同 seed 会进入不同局部区域，最终 BPB 差异可见。

本项目如何使用：

- Phase 5 R2 seed42 明显优于 R3 seed0 和 R9 seed1234。
- R14/R15 说明 warmdown095 在 seed0/1234 上仍不如 seed42。

结论：seed42 是当前最好的训练轨迹；最终候选依赖这个 seed。

## 7. batch、loader 与吞吐

### 7.1 `TRAIN_BATCH_TOKENS`

概念：每个 optimizer step 看到多少 token。单卡时通常用 gradient accumulation 模拟大 batch。

作用阶段：loader、backward、optimizer step。

影响 BPB 的机制：

- 大 batch 梯度更稳定，但每 step 更慢，1 小时内 step 更少。
- 小 batch step 更多，但梯度噪声更大。
- 最优点取决于模型、数据和硬件吞吐。

本项目如何使用：

- Phase 5 root QS28 O2 的 `TRAIN_BATCH_TOKENS=786432` 是第一个 Phase 5 小刷新。
- O8 的 917k batch 在 root 线达到 `1.16357912`。
- records 线 R7 的 917k batch 变差到 post-TTT `1.08560789`，不如 R8。

结论：batch 最优点依赖训练栈。root 线大 batch 有小收益；records 线 917k 不是答案。

### 7.2 coprime loader

概念：改变数据读取步幅，让不同 step 看到的数据位置更“错开”，减少周期性重复。

作用阶段：train loader。

影响 BPB 的机制：

- 理论上可能改善数据覆盖和随机性。
- 但如果破坏局部上下文、doc 边界或 batch 连续性，反而变差。

本项目如何使用：

- Phase 3 S3-L1/L2 明显负收益。

结论：当前实现不继续扩展。

### 7.3 fixed sequence loader 与 varlen/doc-boundary loader

概念：fixed sequence loader 把 token stream 切成固定长度块；varlen/doc-boundary loader 按文档边界或可变长度组织样本。

作用阶段：train loader、attention kernel、TTT 合规。

影响 BPB 的机制：

- fixed sequence 简单、容易 compile，吞吐稳定。
- varlen/doc-boundary 更贴近文档结构，适合 CaseOps 和 TTT，但 kernel/compile 更复杂。

本项目如何使用：

- Phase 5 records 因缺 FA3，在 fallback 下使用 `FixedSequenceTrainLoader`。
- 这让 records stack 能跑通，但可能损失原始 04-27 高吞吐/doc-boundary 训练优势。

结论：当前 R8 成功跑通，但剩余 SOTA gap 很可能和 fallback loader/内核有关。

## 8. 量化、打包与压缩

### 8.1 为什么必须量化

概念：训练权重通常是 fp32/bf16，直接保存会远超 16 MB。量化把权重变成 int8/int7/int6/int5 等低比特表示。

作用阶段：训练结束后的导出。

影响 BPB 的机制：

- bit 数越低，artifact 越小。
- 但权重误差越大，模型预测变差，roundtrip BPB 上升。
- 好的量化方法要在大小和质量之间取平衡。

本项目如何使用：

- Phase 1 SP8192 int8 质量好但超 16 MB。
- Phase 2 开始围绕 GPTQ、mixed bit、LQER 等解决“压得下但不掉太多质量”的问题。

结论：量化是这个项目的核心，不是附属步骤。

### 8.2 RTN：round-to-nearest

概念：最朴素的量化，把浮点权重按 scale 映射到整数，四舍五入。

作用阶段：导出。

影响 BPB 的机制：

- 简单、快、容易实现。
- 不考虑权重量化后对模型输出的影响，所以低比特下质量损失较大。

本项目如何使用：

- Phase 2 R9 RTN override 合规但 BPB `1.18714581`。
- GPTQ + LQER 后同类路线明显更好。

结论：RTN 是 baseline 对照，不是最终主线。

### 8.3 GPTQ

概念：GPTQ 是 Hessian-aware post-training quantization。它量化一层权重时，会用校准数据估计“哪些权重误差对输出影响更大”，尽量让量化误差对模型输出的损害更小。

作用阶段：导出阶段，需要校准 batch。

影响 BPB 的机制：

- 同样 bit 数下，GPTQ 通常比 RTN 更保质量。
- 校准 batch 越充分，估计可能越准，但导出更慢、可能稍影响文件。

本项目如何使用：

- Phase 2 引入 GPTQ 后，B3 GPTQ int6/embed8 到 `1.18001764`。
- GPTQ + LQER top3 rank4 到 `1.17661956`，成为 Phase 2 best。
- Phase 4 QS28 的 final export 使用 `GPTQ_CALIBRATION_BATCHES=32` 和 `GPTQ_ERROR_SCALE=0.975`。
- Phase 5 R8Q1-Q5 的 GPTQ/export 微扫没有超过原 R8。

结论：GPTQ 是 root 线能压进 16 MB 且保持质量的关键技术；但到 R8 后，普通 GPTQ 微参不是主瓶颈。

### 8.4 mixed bitwidth：matrix bits 与 embed bits

概念：不同权重用不同 bit 数。例如 transformer 矩阵用 int6，embedding 用 int8 或 int7。

作用阶段：导出。

影响 BPB 的机制：

- embedding/head 对词预测非常敏感，bit 太低会明显伤 BPB。
- 大矩阵参数多，用低 bit 可以省很多空间。
- mixed bit 的目标是把高 bit 留给敏感参数，把低 bit 用在更能承受误差的参数上。

本项目如何使用：

- Phase 2/3 root 线常用 matrix int6 + embed8。
- Phase 5 records R1Q7 使用 embed7/top3，并依赖 real `lrzip` 合规。
- O4/P 系列尝试 MLP3 + int5/embed8、embed7/embed6 容量修复。

结论：embedding bit 是质量和大小的关键旋钮。R1Q7 证明 embed7/top3 在 real `lrzip` 下是合规甜点。

### 8.5 signed bit packing

概念：低比特整数如果还按 int8 一个 byte 存储，会浪费空间。bit packing 把多个 int6/int7/int5 权重紧密塞进 byte stream。

作用阶段：导出打包。

影响 BPB 的机制：

- 主要降低 artifact bytes，不直接改善模型质量。
- 但节省出来的空间可以换更高 bit、更大 embedding、更多 LQER 修正。

本项目如何使用：

- Phase 2 实现 signed bit packing 后，mixed low-bit artifact 才真正有意义。

结论：这是合规基础设施。没有 packing，低比特量化节省不了足够空间。

### 8.6 clip search 与 `GPTQ_ERROR_SCALE`

概念：量化时通常要决定权重最大范围。clip 太少会截断大权重，clip 太多会让大多数小权重精度变粗。`GPTQ_ERROR_SCALE` 则调整 GPTQ 误差补偿强度。

作用阶段：导出。

影响 BPB 的机制：

- 合适 clipping 可以减少量化 MSE。
- 但最佳点依赖 checkpoint 和层分布。
- 过度微扫通常收益很小。

本项目如何使用：

- 早期 clip-search 帮助低比特路线建立 baseline。
- Phase 4 QS28 通过 `GPTQ_ERROR_SCALE=0.975` 小幅刷新。
- Phase 5 R8Q1/R8Q4 的 err0.95/err1.05 都不如原 R8。

结论：clip/error scale 是导出细调工具，不是 Phase 5 之后的大杠杆。

### 8.7 LQER

概念：LQER 可以理解成“量化后低秩误差修正”。先做主权重量化，再用少量额外参数近似补回最大、最重要的量化误差。

作用阶段：导出，量化之后、压缩之前。

影响 BPB 的机制：

- 增加少量 bytes，换取 roundtrip BPB 降低。
- top-k/rank 控制修正多少张量、每个修正的秩多大。
- 太多 LQER 会超大小，太少修不回来。

本项目如何使用：

- Phase 2 B3 + LQER top3 rank4 从 `1.18001764` 到 `1.17661956`。
- Phase 3 Q43 使用 LQER top4 rank4。
- Phase 5 records embed7/top3 使用 LQER/类似 per-group 修正与 `lrzip` 配合。
- R8Q5 top4 反而更差且更大。

结论：LQER 是 Phase 2/3 的重要收益来源，但在 R8 附近已经平台。

### 8.8 brotli、lzma、zlib、lrzip

概念：量化后的权重和元数据还要用通用压缩器压缩。不同压缩器对权重 byte pattern 的压缩率不同。

作用阶段：最终 artifact 压缩。

影响 BPB 的机制：

- 压缩器本身不改变模型输出，所以不直接改变 BPB。
- 但压得更小可以让更高 bit、更多 LQER、更大 embedding 合规，间接保住 BPB。

本项目如何使用：

- baseline 用 int8+zlib。
- Phase 2/3 使用 brotli 等压缩增强。
- Phase 5 fallback brotli/lzma 无法把 R1Q2 embed7/top3 压到 16 MB。
- 安装真实外部 `lrzip` 后，同一权重 R1Q7 压到 `15,925,658` bytes，no-TTT 仍是 `1.09951341`。

结论：real `lrzip` 是 Phase 5 合规关键。它不让模型更聪明，但让聪明一点的模型装得下。

### 8.9 `FRESH_MODEL_AFTER_QUANT`

概念：量化校准可能会在模型对象上挂状态或改变某些 buffer。导出后重新创建干净模型再加载量化权重，可以避免评估污染。

作用阶段：roundtrip eval。

影响 BPB 的机制：

- 不应改变真实模型能力。
- 但能避免“评估的不是提交会加载的模型”这种 bug。

本项目如何使用：

- Phase 2 引入 `FRESH_MODEL_AFTER_QUANT=1` 后，GPTQ roundtrip 评估才可信。

结论：这是评估正确性修复，不是模型能力优化。

## 9. TTT：test-time training

### 9.1 TTT 的基本思想

概念：普通评估时模型权重固定。TTT 允许模型在测试时读取当前文档的前缀或已评分过的部分，做少量梯度更新，然后继续预测后面的 token。

作用阶段：评估阶段，不是训练主循环。

影响 BPB 的机制：

- 模型可以适应当前文档的主题、格式、重复实体和局部规律。
- 如果合规实现，不能用未来 token 给当前 token 打分，也不能重复评分后挑最好结果。
- 合法 TTT 的收益通常体现在 post-TTT BPB。

### 9.2 legal / score-before-update

概念：score-before-update 指每个 token 或 chunk 必须先用当前模型打分，再用它更新模型。不能先看答案更新，再回来给同一个 token 打分。

作用阶段：TTT eval。

影响 BPB 的机制：

- 保证没有信息泄漏。
- 让 TTT 分数可以被认为是合法提交策略。

本项目如何使用：

- Phase 3 的 lm-head LoRA TTT MVP 不是真正 records TTT，结果退化到 `1.61284870`，不能代表 TTT。
- Phase 5 使用 records legal phased TTT，R1Q7T 从 `1.09951341` 降到 `1.08464351`。
- R8 默认 TTT 从 `1.09647097` 降到 `1.08179851`。

结论：TTT 是 Phase 5 的大收益来源之一，但必须是 legal/phased/doc-aware 版本。

### 9.3 phased TTT

概念：phased TTT 把测试时更新分成多个 phase，用不同学习率、adapter 或更新范围逐步适配。

作用阶段：TTT eval。

影响 BPB 的机制：

- 多阶段更新比一次性更新更稳定。
- prefix docs、phase 数、global lr 会影响适配强度。
- 过强会过拟合或破坏已有能力，过弱收益不足。

本项目如何使用：

- R1Q7T 默认 3-phase 到 `1.08464351`。
- T3-T14 扫 prefix docs、phase 数、global lr，最好 T12 为 `1.08458960`，只改善约 `5e-5 BPB`。
- R8T12 迁移 T12 设置后为 `1.08180270`，略差于 R8 默认 TTT。

结论：TTT 的大框架有效，但 R8 附近的 TTT 微参已经平台。下一步要先改善 no-TTT artifact。

## 10. 系统与 kernel 优化

### 10.1 `torch.compile`

概念：PyTorch 2 的编译器把 Python/PyTorch 代码编译成更高效的图执行。

作用阶段：训练 forward/backward、eval。

影响 BPB 的机制：

- 不直接改变模型数学结果。
- 提高吞吐，让 1 小时内看到更多 token、完成更多 optimizer step，间接降低 BPB。
- 如果遇到动态 shape 或 data-dependent guard，可能编译失败。

本项目如何使用：

- 根脚本使用 `torch.compile(..., dynamic=False, fullgraph=True)`。
- Phase 5 records fallback 遇到 varlen attention 动态 slice 导致 Dynamo guard 失败，后来改成无 FA3 时 eager eval/fixed loader。

结论：compile 是吞吐工具，但和 varlen/doc-boundary 结构有冲突，需要小心处理。

### 10.2 FA3 与 SDPA fallback

概念：FA3 是 FlashAttention 3，高效 attention kernel。SDPA 是 PyTorch 的 scaled dot-product attention 接口，可选择 flash/math/mem-efficient 后端。

作用阶段：attention forward/backward。

影响 BPB 的机制：

- kernel 不改变理论模型，但影响 step time。
- 在固定 1 小时预算下，step 越快，训练越充分，BPB 越可能下降。
- varlen/doc-boundary attention 依赖高效 kernel，否则 fallback 很慢或难 compile。

本项目如何使用：

- 当前环境缺 `flash_attn_interface`。
- Phase 5 records 线补了 torch SDPA fallback。
- fixed-seq fallback 能跑通 R8，但可能比原 records 04-27 高吞吐环境弱。

结论：FA3/fused kernel 是下一阶段最重要的环境 blocker，不是因为它让模型结构变了，而是因为它决定 1 小时内能训练到什么质量。

### 10.3 TensorDescriptor / fused MLP fallback

概念：records 脚本里可能用 Triton TensorDescriptor 或 fused MLP kernel 加速 MLP/LeakyReLU^2 等操作。

作用阶段：MLP forward/backward。

影响 BPB 的机制：

- 和 attention kernel 类似，主要影响吞吐。
- MLP 是 transformer 计算大头之一，fused MLP 可以显著减少 overhead。

本项目如何使用：

- 当前环境缺 `triton.tools.tensor_descriptor`。
- Phase 5 records fallback 使用 eager LeakyReLU^2 MLP path。

结论：能跑通，但不是最佳训练环境。剩余 SOTA gap 很可能部分来自这里。

## 11. 本项目各阶段的优化主线

### Phase 1：确定质量基座

核心问题：应该用小词表省参数，还是大词表降 BPB？

结论：SP8192 明显更好，但 int8 artifact 超 16 MB。因此后续必须强化量化压缩。

### Phase 2：把 SP8192 压进 16 MB

核心问题：如何让大词表模型合规，同时不损失太多质量？

有效方法：

- GPTQ 替代 RTN。
- mixed bitwidth。
- signed bit packing。
- LQER 修正量化误差。
- fresh roundtrip eval 保证评估可信。

结果：B3 GPTQ + LQER 到 `1.17661956`。

### Phase 3：root Q43 局部爬山

核心问题：在 GPTQ/LQER 已可用的基础上，哪些训练动态能继续降低 roundtrip BPB？

有效方法：

- `QK_GAIN_INIT=5.0`
- hparam stack
- `TIED_EMBED_LR=0.04`
- 轻量 recurrence
- `MUON_MOMENTUM=0.97`
- LQER top/rank 小扫

结果：Q43 到 `1.16735413`。

### Phase 4：records 组件局部移植

核心问题：records 里的结构组件单独搬到 root Q43 是否有效？

有效方法：

- SparseGate + LeakyReLU^2 + Polar NS 组合。

负结果：

- LeakyReLU^2 单项、Polar NS 单项、partial RoPE、SmearGate 单项都不成立。

结果：QS28 到 `1.16522320`，但提升很小。

### Phase 5：CaseOps 与 records 主线

核心问题：不再局部模仿 records，而是让 records 04-27 advanced stack 在 1xH100/1h 下真实跑通。

有效方法：

- CaseOps full data + byte sidecar。
- records FA3/SDPA/TensorDescriptor fallback。
- real external `lrzip` 让 embed7/top3 artifact 合规。
- legal phased TTT。
- seed42。
- `WARMDOWN_FRAC=0.95`。

负结果：

- root CaseOps 静态训练。
- TTT 近邻微扫。
- delayed loop。
- records batch917k。
- min_lr 抬高。
- R8 export-only 小扫。

结果：R8 no-TTT `1.09647097`，post-TTT `1.08179851`。

## 12. 如何读一个新实验结果

读实验时建议按这个顺序：

1. 先看是否合规：`total bytes <= 16,000,000`。
2. 再看 roundtrip BPB，而不是只看 pre-quant BPB。
3. 如果有 TTT，看 no-TTT 和 post-TTT 的差值。
4. 看它改的是数据、结构、训练动态、量化，还是纯压缩。
5. 看收益是否大于噪声：`1e-5` 级通常只是导出微扫；`1e-3` 级是小但真实；`1e-2` 级是主线级收益。
6. 看有没有副作用：step 变少、bytes 变大、需要外部依赖、是否依赖特定 seed。

一个例子：

```text
R8:
no-TTT 1.09647097
post-TTT 1.08179851
bytes 15,924,510
```

这说明：

- artifact 合规。
- 训练后模型本身比 R1Q7 更好。
- TTT 有大约 `0.01467 BPB` 收益。
- 它依赖 real `lrzip` 和 records fallback 栈。
- 后续如果想继续提升，应该优先降低 no-TTT 起点，而不是再扫 TTT 小参。

## 13. 当前最重要的经验教训

1. 只降低 pre-quant BPB 不够。O4 MLP3 很强，但超 16 MB，不能直接提交。
2. 只压小 artifact 也不够。embed6 能合规，但 BPB 会变差。
3. root QS28 已经平台。Phase 4/5 的 root 线最多是 `1e-3` 级小收益。
4. records 主线是真正跃迁来源。R1Q7/R8 把结果从 `1.16` 区间带到 `1.08` 区间。
5. TTT 有用，但 R8 附近 TTT 微调已经平台。
6. real `lrzip` 是合规关键依赖，不能在最终复现链路里忘掉。
7. 剩余 gap 主要是训练质量/吞吐 gap。当前 R8 no-TTT 仍明显弱于 04-27 records post-quant，优先级应是 FA3/fused MLP/TensorDescriptor 或 root-native 高吞吐 doc-boundary 路线。

## 14. 方法速查表

| 方法 | 阶段 | 主要影响 | 本项目结论 |
| --- | --- | --- | --- |
| SP8192 | 数据/tokenizer | 降低 token 化难度，提高质量，增加 embedding 体积 | 有效，必须配强压缩 |
| seq4096 | loader/context | 更长上下文，step 更慢 | 有效甜点 |
| CaseOps | 数据/评估/TTT | byte sidecar、doc boundary | root 静态负收益，records 配套有效 |
| QK gain | forward | 调 attention 尺度 | Phase 3 小正收益 |
| tied embed LR | optimizer | 改善 embedding/head 学习 | Phase 3 大正收益 |
| Muon momentum | optimizer | 矩阵更新更稳 | Phase 3 小正收益 |
| recurrence | forward | 增加有效深度 | root 窄窗口有效，records delayed loop 负向 |
| SparseGate | forward | 提高通道/分支选择效率 | 单项不强，组合有效 |
| LeakyReLU^2 | forward | 改激活分布 | 单项负，组合有效 |
| Polar NS | optimizer | 改矩阵更新几何 | 单项负，组合有效 |
| MLP3 | forward/size | 增加 FFN 容量 | 质量强但超 16 MB |
| GPTQ | quant | 降低低比特量化损失 | Phase 2/3 核心 |
| mixed bit | quant | 敏感参数高 bit，大矩阵低 bit | 核心容量策略 |
| LQER | quant | 补回量化误差 | Phase 2/3 有效，R8 平台 |
| bit packing | export | 降 artifact bytes | 合规基础 |
| lrzip | compression | 更强最终压缩 | Phase 5 合规关键 |
| legal TTT | eval | 测试时文档适配 | 大收益，约 `0.014-0.015 BPB` |
| TTT micro sweep | eval | 调 phase/prefix/lr | R8 附近平台 |
| warmdown095 | LR schedule | 改后段收敛 | Phase 5 R8 关键 |
| min_lr up | LR schedule | 保持后段学习率 | Phase 5 负向 |
| FA3/fused MLP | kernel | 提高吞吐、恢复 records 形态 | 下一阶段 blocker |
