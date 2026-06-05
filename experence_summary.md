# Parameter Golf 历史优化经验总结

本文总结 `records/` 历史提交中的常见优化手段，重点面向当前根目录 `train_gpt.py` 的后续改造。历史记录大多针对官方主赛道 `8xH100 / 10min / 16MB`，但其中很多方法也适用于当前 `1xH100 / 1h / 16MB` 的实验。

## 1. 总体演进脉络

历史成绩大致经历了几条主线叠加：

| 阶段 | 代表方向 | 核心收益 |
| --- | --- | --- |
| baseline 调参 | 更长 seq、LR/warmdown、fp16/bf16 embedding、Muon 参数 | 低风险，改善训练稳定性和收敛 |
| 结构增强 | 10/11 层、MLP 3x/4x、LeakyReLU²、U-Net skip、XSA、partial RoPE | 用接近 16MB 的容量换更低 loss |
| 量化压缩 | int6/int7/int8、GPTQ/GPTQ-lite、std clip、QAT、EMA/SWA | 在 16MB 内容纳更大模型，减少 post-quant 损失 |
| tokenizer/数据 | SP4096、SP8192、CaseOps、byte sidecar | 改善 BPB 度量下的 tokenizer 效率 |
| 推理/评估增强 | sliding eval、legal score-first TTT、phased TTT、LoRA TTT | 不改训练权重或少改 artifact，显著降低最终 BPB |
| 系统优化 | FlashAttention 3、fused MLP、fused CE、parallel Muon、压缩管线 | 用更少时间跑更多有效 step，或腾出 eval/quant 时间 |

一个重要经验是：后期 SOTA 不是单个技巧的胜利，而是多个 0.001-0.01 BPB 级别改进的堆叠，同时严格控制 artifact 大小和合法性。

## 2. 模型结构类优化

### 2.1 增加有效容量

常见做法：

- `NUM_LAYERS` 从 9 增到 10/11。
- `MLP_MULT` 从 2 提到 3/4。
- 保持 `MODEL_DIM=512`，因为宽度扩大对参数和压缩体积压力更大。
- 保持 GQA，例如 8 query heads / 4 KV heads。

经验：

- 11 层、512d、8H/4KV 是很多强提交的基础形态。
- 4x MLP 提升容量明显，但需要配合更强压缩和更高 weight decay。
- 结构增大前必须持续检查 `Total submission size`，最终看 `final roundtrip` 而不是 pre-quant。

### 2.2 LeakyReLU² 替代 ReLU²

历史记录中，`LeakyReLU(0.5)^2` 是一个低复杂度、高性价比改动。它保留 squared activation 的形式，同时让负半轴也有梯度。

典型替换：

```python
# baseline
x = torch.relu(self.fc(x))
return self.proj(x.square())

# historical
x = F.leaky_relu(self.fc(x), negative_slope=0.5)
return self.proj(x.square())
```

经验：

- 曾有记录报告单独贡献约 `-0.002 ~ -0.003 BPB`。
- 改动很小，适合当前 baseline 优先尝试。
- 需要重新跑量化后 roundtrip，因为 activation 改动会改变权重分布和压缩表现。

### 2.3 XSA / extra skip / value residual

历史提交中多次使用：

- XSA last-N 或 XSA-all。
- U-Net skip connections。
- sigmoid-gated skip。
- value embedding / value residual。

经验：

- XSA 从 last 4 layers 扩到 all layers 曾经带来收益，但实现复杂度高于简单 MLP 激活替换。
- value residual、hash embedding、smear gate 等在某些大 vocab 配置下被移除，说明它们不是无条件有效。
- 这类改动需要从对应 record 迁移完整 forward/eval/quant 路径，半迁移容易产生 silent mismatch。

### 2.4 深度复用 / depth recurrence

高分记录反复出现“复用中间层”的思路：

- 循环 layers 4-5 或 3-5。
- 共享大部分参数，只 untie 重复 MLP 或少量分支。
- 延迟开启 recurrence，例如训练到一定 step 或 fraction 后再启用。

经验：

- 有效区域通常在 U-Net hinge 附近的中间层。
- 一直开启 recurrence 会降低 step 数；延迟开启能兼顾早期吞吐和后期深度。
- 重复太多层可能被 step-time penalty 抵消。

对当前 baseline：

- 这是中等风险改动。单卡 1h 中 step 数本来有限，延迟开启尤其重要。
- 需要记录“有效虚拟层数增加”与“step_avg 变慢”的 tradeoff。

### 2.5 Parallel residuals

并行残差把 attention lane 和 MLP lane 分开，深层开始让 attention/MLP 从不同 residual stream 读取，并学习写回混合。

经验：

- 在一些记录中是叠加在 depth recurrence 上的显著收益项。
- 学到的路由往往不对称，说明 attention/MLP stream 的分工确实不同。
- 实现复杂度较高，影响 forward、TTT forward、量化命名和小参数保存。

## 3. Tokenizer 与数据表示

### 3.1 SP4096 / SP8192

从 sp1024 baseline 往后，历史强提交大量转向更大 vocab：

- SP4096：embedding/tokenizer 体积增加不大，但同样 seq_len 能覆盖更多原始文本。
- SP8192：进一步提升 tokenization efficiency，是后期主流强栈基础。

经验：

- 大 vocab 增加 embedding 参数，但减少 token 数和 BPB 压力。
- 必须重新下载或导出匹配 tokenizer 的数据 shard。
- `VOCAB_SIZE`、`TOKENIZER_PATH`、`DATA_PATH` 必须严格一致。

对当前 baseline：

- SP4096 是比较自然的下一阶段。
- SP8192 更接近历史强栈，但会立即触发更强量化/压缩需求。

### 3.2 CaseOps / lossless caps

后期 SOTA 使用 CaseOps：对大小写做可逆变换，并配合 byte sidecar 正确计算原始 bytes 上的 BPB。

经验：

- 对 BPB 很关键，因为它改变了 tokenizer 对大小写和文本模式的建模负担。
- 合法性要求高：必须证明变换可逆，验证 BPB 统计必须基于原始 bytes。
- 历史中还出现过 BOS/document boundary bug 修复，说明数据边界处理很容易影响结果。

对当前 baseline：

- 不建议作为第一个改动。
- 若迁移，应直接从高分 record 复制完整 `prepare_caseops_data.py`、`lossless_caps.py`、tokenizer 和 byte sidecar 路径，并单独写正确性测试。

### 3.3 Loader 改进

历史中出现过：

- coprime-stride loader，避免连续 minibatch 来自相近文档区域。
- shuffled sequence loader。
- 简化 loader，减少复杂系统风险。

经验：

- 数据顺序会影响短时间训练表现。
- loader 改动收益通常小于 tokenizer/结构/量化，但成本也相对可控。
- 需要保持训练集和验证集隔离，不能把 val 信息泄漏到训练。

## 4. 优化器与训练日程

### 4.1 Muon 系列

历史记录中 Muon 是核心优化器，改进方向包括：

- Muon momentum 从 0.95 提高到 0.97/0.99。
- momentum warmup 起点和长度调优。
- row-normalized Muon / MuonEq-R。
- Polar Express Newton-Schulz 系数，替换固定 `(a,b,c)`。
- parallel Muon / parameter banking 提升系统效率。

经验：

- Muon 细节对矩阵参数训练质量影响很大。
- Newton-Schulz 迭代质量和速度之间有 tradeoff。
- 在当前 baseline 中，`MATRIX_LR`、`MUON_MOMENTUM`、`MUON_BACKEND_STEPS` 是优先 sweep 对象。

### 4.2 LR、warmdown、MIN_LR

历史记录中反复调：

- 更长 warmdown，例如 `WARMDOWN_ITERS=3500` 或按 fraction 设置。
- warmdown 不降到 0，而是保留 `MIN_LR=0.1` 之类的 LR floor。
- 大模型配更低 `MATRIX_LR`，如 0.026 左右。
- Adam `BETA2` 提高到 0.99 稳定 embedding/scalar。

经验：

- 在严格 wallclock 下，过早把 LR 降到接近 0 会浪费最后阶段 step。
- LR floor 在后期记录中是一个低复杂度有效技巧。
- 单卡 1h 下 warmdown 的绝对 step 数要重新估计，不能直接照搬 8 卡 10 分钟。

### 4.3 EMA / SWA

历史中常见：

- EMA decay 约 0.9965/0.997。
- Tight SWA 在 warmdown 后期每隔固定 step 平均。

经验：

- EMA/SWA 可以降低量化前验证损失，也可能让量化更稳。
- 会增加代码和状态保存复杂度。
- 需要明确最终量化的是当前权重、EMA 权重还是 SWA 权重。

## 5. 量化与压缩

### 5.1 后训练量化是核心战场

baseline 使用 int8 per-row + zlib，历史强提交使用过：

- int6 matrix。
- int7 embedding。
- int8 gate/control。
- GPTQ / GPTQ-lite。
- AR self-generated GPTQ calibration。
- AWQ-lite / mixed GPTQ。
- LQER asymmetric quant-error correction。

经验：

- pre-quant BPB 和 final BPB 常有明显差距。
- 强量化能塞下更大模型，但校准和压缩时间也要计入训练或 eval 合规预算。
- GPTQ calibration 如果访问训练数据，需要预留训练 wallclock。

### 5.2 clip 策略影响压缩，不只是误差

历史记录强调：压缩后大小不只取决于 bitwidth，还取决于量化值熵。

关键经验：

- int4 未必比 int5/int6 压得显著更小，因为 histogram 更平会增加熵。
- row std-based clip 比搜索多个 percentile 更快、更可控。
- 增大 clip sigma 有时能降低压缩后大小，因为更多值集中在中心 bins。
- embedding 和 MLP/attention 的 clip 应分开调。

### 5.3 压缩器与序列化布局

历史提交使用过：

- zlib baseline。
- zstd level 22。
- brotli。
- lzma。
- per-group compression。
- 对热点矩阵做相似行排序再压缩。
- lrzip/zpaq 处理长程重复。

经验：

- 接近 16MB 时，几十 KB 都很重要。
- 压缩器速度也重要：压缩可以慢一些，但 eval 解压必须在预算内。
- 复杂压缩会增加外部二进制依赖和复现成本。

对当前 baseline：

- 当前 1h baseline artifact 只剩约 `130KB` headroom，已经很紧。
- 任何模型增大都必须先引入更强量化或压缩。

## 6. 评估与 Test-Time Training

### 6.1 Sliding window eval

早期记录使用 sliding window eval，通过更密集上下文评分降低 BPB。

经验：

- 可以不改训练，只改 evaluation。
- 需要避免重复计数或边界 bug。
- eval 时间预算会增加。

### 6.2 Legal score-first TTT

TTT 的合法原则：

1. 先 score 当前 chunk。
2. 再用已经 score 过的 token 更新模型或 LoRA。
3. 未来 chunk 可以受过去 chunk 更新影响。
4. 不能在评分前使用当前或未来验证 token 更新。

历史方法：

- chunk-level SGD TTT。
- LoRA TTT。
- per-doc reset。
- phased TTT。
- warm-start LoRA A，只 reset B。
- multi-phase global SGD。

经验：

- 后期 phased TTT 能带来非常大的 post-eval BPB 改善，常见量级可达 `~0.01 BPB`。
- TTT 是 eval-time 改进，不一定改善 pre-quant 或 no-TTT 分数。
- TTT forward 必须和普通 forward 完全对齐；历史中曾因为 gate 没同步到 TTT path 导致灾难性结果。

对当前 baseline：

- 如果目标只是先提升训练 baseline，TTT 可以后置。
- 如果目标追排行榜式最终 BPB，TTT 是绕不开的高收益方向。

## 7. 系统优化

### 7.1 Attention kernel

历史强提交常用 FlashAttention 3，尤其是 H100/Hopper。

经验：

- 当前根目录 baseline 用 PyTorch `F.scaled_dot_product_attention`，并开启 flash SDP backend；它已经不是普通 math attention。
- 外部 FA3 可能更快，但收益取决于 shape、PyTorch 版本、GQA 和编译状态。
- 单纯替换 attention kernel 未必是最大收益点，因为 step time 还包含 MLP、optimizer、grad accumulation、CE、量化等。

### 7.2 Fused MLP / fused CE

历史记录中出现：

- fused LeakyReLU² MLP Triton kernel。
- fused softcapped cross entropy。

经验：

- 对 10 分钟限制非常重要，能把节省的时间换成更多 step。
- 实现和维护成本较高。
- 对当前单卡 1h，若 profiler 显示 MLP/CE 占比高，值得考虑。

### 7.3 Parameter banking / parallel Muon

做法：

- 把多个 Linear weight 合并成 3D parameter bank。
- batched Newton-Schulz。
- 减少 DDP/optimizer overhead。

经验：

- 早期记录中更多是系统提速，直接 BPB 改善不一定明显。
- 对多卡更重要；单卡收益可能较小。

## 8. 高收益但高风险的细节修复

历史记录里有不少“不是新模型，但分数很关键”的修复：

- SmearGate BOS/document boundary leak。
- CaseOps BOS prep bug。
- sliding window 重复计数 bug。
- TTT path 和 normal forward 不一致。
- GPTQ reserve 时间不足导致合规风险。

经验：

- Parameter Golf 里指标很敏感，边界处理错误可能制造虚假收益或严重回退。
- 每次引入 tokenizer、TTT、sliding eval、packed docs，都要单独检查 BOS、doc boundary、byte count、score-first 顺序。

## 9. 对当前 `train_gpt.py` 的建议优先级

### 第一优先级：低风险 sweep

这些不大改代码，适合先做单卡 1h sweep：

- `QK_GAIN_INIT=3.0/5.0/5.25`
- `MUON_MOMENTUM=0.97/0.99`
- `MATRIX_LR=0.03/0.04/0.05`
- `TIED_EMBED_LR=0.03/0.05/0.07`
- `WARMDOWN_ITERS=2000/3500/5000`
- `BETA2=0.95/0.99`
- `TRAIN_SEQ_LEN=1024/2048`

### 第二优先级：小代码改动

- LeakyReLU(0.5)²。
- MIN_LR floor。
- EMA 权重平均。
- 更细的 int8 clip 或 std-based clip。
- 更紧凑的 summary/logging，便于比较实验。

### 第三优先级：中等改造

- SP4096 tokenizer + 数据。
- 10/11 层 + MLP 3x/4x。
- GPTQ-lite 或 std-clip quantization。
- delayed mini depth recurrence。
- sliding window eval。

### 第四优先级：强栈迁移

- SP8192 + GPTQ embeddings。
- CaseOps lossless caps。
- phased LoRA TTT。
- sparse attention gate / SmearGate BOS-fix。
- fused CE / fused MLP。
- per-group compression。

这类改动收益大，但牵涉数据、评估、量化、压缩和合法性，建议一次只迁移一个完整子系统，并保留可复现实验 JSON。

## 10. 实验记录模板

建议每次实验都写一个 `.summary.json`，至少包含：

- `run_id`
- git diff 或代码版本
- GPU、torch、环境
- `DATA_PATH`、`TOKENIZER_PATH`、`VOCAB_SIZE`
- 模型结构超参
- optimizer/LR/warmdown
- stop step、step_avg、tokens_seen
- pre-quant `val_bpb`
- final roundtrip `val_bpb`
- artifact bytes 和 bytes_under_limit
- 是否使用 sliding eval / TTT / GPTQ / CaseOps

最终排序应以 `final_roundtrip_eval.val_bpb` 为准；历史经验已经多次说明，pre-quant 分数和最终 artifact 分数之间可能差出一个足以改变结论的量级。
