# Parameter Golf 项目结构与 baseline 训练分析

本文档面向当前仓库的 `train_gpt.py` 开发，约束假设为：最终提交物不超过 16MB，训练资源为 `1xH100`，训练时间约 `1h`。原始 OpenAI Parameter Golf leaderboard 的主赛道约束是 `8xH100 / 10min / 16MB`，因此本项目在单卡一小时场景下，核心目标可以理解为：在同样 16MB artifact 约束下，用更长的单卡时间补足训练步数，并优先改进单文件训练脚本。

## 0. 当前运行环境速查

当前实验主要在 H100 容器中运行。为了方便会话迁移，完整速查表见 `ref_config.md`。

| 项目 | 当前值 |
| --- | --- |
| 宿主项目目录 | `/public/home/lvbqy/project/parameter-golf` |
| 容器名 | `lbqy0` |
| 容器项目目录 | `/base/project/parameter-golf` |
| 推荐训练 Python | `/opt/conda/bin/python` |
| 推荐训练 torch | `2.6.0+cu126` |
| `gf` 环境 | torch `2.12.0+cu130`，曾在默认 full batch 下触发 `torch.compile/Inductor` backward shape 错误 |
| GPU | 8 x NVIDIA H100 80GB HBM3 |
| 常用空闲卡 | 先查 `nvidia-smi`，最近实验使用 `CUDA_VISIBLE_DEVICES=6,7` |

当前数据位置：

| 数据 | 路径 | 状态 |
| --- | --- | --- |
| SP1024 dataset | `/base/project/parameter-golf/data/datasets/fineweb10B_sp1024` | 80 train shards + full val |
| SP1024 tokenizer | `/base/project/parameter-golf/data/tokenizers/fineweb_1024_bpe.model` | 可用 |
| SP8192 dataset | `/base/datasets/SP8192/datasets/fineweb10B_sp8192` | 80 train shards + full val |
| SP8192 tokenizer | `/base/datasets/SP8192/tokenizers/fineweb_8192_bpe.model` | 可用 |

SP8192 是通过用户代理下载的。成功路线是设置 `HTTP_PROXY/HTTPS_PROXY/ALL_PROXY=http://59.66.143.200:7897`，并 `unset HF_ENDPOINT` 后访问官方 Hugging Face；`HF_ENDPOINT=https://hf-mirror.com` 在当前环境里会触发 `huggingface_hub` metadata 校验问题。

当前 `train_gpt.py` 相比原 baseline 新增了可选 packed low-bit 导出：

- `MATRIX_QUANT_BITS` / `QUANT_BITS`：矩阵量化 bitwidth，默认 `8`。
- `EMBED_QUANT_BITS` / `EMBED_BITS`：`tok_emb.weight` 量化 bitwidth，默认同矩阵。
- `QUANT_CLIP_SEARCH`：是否启用多个 clip percentile 的 MSE 搜索，默认 `0`。
- `QUANT_CLIP_SEARCH_PCTS`：clip-search 候选分位。

注意：当前实现是 packed int6/int7 后训练量化和 clip-search，不是完整 Hessian-aware GPTQ。

## 1. 项目结构

仓库整体是一个“可运行 baseline + 数据工具 + 历史优秀提交记录”的结构。

| 路径 | 作用 |
| --- | --- |
| `train_gpt.py` | CUDA/PyTorch baseline 主脚本。包含超参、数据加载、模型定义、Muon 优化器、训练循环、验证、int8+zlib 导出。当前主要开发入口。 |
| `train_gpt_mlx.py` | Apple Silicon/MLX 版本 baseline，用于本地小规模试跑，不是当前 H100 主路径。 |
| `requirements.txt` | Python 依赖参考，包括 `torch`、`sentencepiece`、`datasets`、`huggingface-hub` 等。 |
| `data/` | 数据下载、重分词、tokenizer 说明与脚本。默认数据来自 FineWeb 缓存导出。 |
| `data/cached_challenge_fineweb.py` | 下载官方缓存 FineWeb shard 与 tokenizer。 |
| `data/download_hf_docs_and_tokenize.py` | 基于固定文档集合重新训练/导出 tokenizer 和 shard。 |
| `data/tokenizer_specs.json` | tokenizer 规格配置。 |
| `records/track_10min_16mb/` | 已记录的 16MB/10min 主赛道提交，包含每次提交的 `train_gpt.py`、README、日志和 `submission.json`。是后续借鉴改法的主要资料库。 |
| `records/track_non_record_16mb/` | 非主榜或超出标准计算约束但仍满足 16MB 的实验性提交。 |
| `results/`、`scripts/` | 复现实验、rerun 或辅助脚本。 |
| `paper/` | 相关实验/论文材料。 |
| `ref_config.md` | 当前会话迁移速查，包含容器环境、数据路径、已完成 run 和常用命令。 |
| `*.summary.json` | 精简实验记录，避免从完整 log 中手工提取配置和结果。 |

从开发角度看，根目录 `train_gpt.py` 是“新手友好 baseline”，不是当前 leaderboard SOTA。历史 SOTA 思路主要沉淀在 `records/` 中，例如更大 tokenizer、长上下文、深度复用、并行残差、TTT、GPTQ/AWQ、CaseOps tokenizer 等。

## 2. 当前 baseline 训练方法

### 2.1 默认配置概览

`train_gpt.py` 的默认 Simple Baseline：

| 类别 | 默认值 |
| --- | --- |
| tokenizer / vocab | SentencePiece BPE，`VOCAB_SIZE=1024` |
| 数据 | `./data/datasets/fineweb10B_sp1024`，训练 shard 为 `fineweb_train_*.bin`，验证 shard 为 `fineweb_val_*.bin` |
| 模型宽度 | `MODEL_DIM=512` |
| 层数 | `NUM_LAYERS=9` |
| attention | `NUM_HEADS=8`，`NUM_KV_HEADS=4`，即 GQA |
| MLP | `MLP_MULT=2`，ReLU squared MLP |
| context | `TRAIN_SEQ_LEN=1024` |
| batch | `TRAIN_BATCH_TOKENS=524288` 全局 token/step |
| 训练步数 | `ITERATIONS=20000`，但默认 `MAX_WALLCLOCK_SECONDS=600` 会提前停止 |
| embedding | 默认输入输出权重绑定，`TIE_EMBEDDINGS=1` |
| 优化器 | embedding/小参数用 Adam，矩阵参数用 Muon |
| 精度 | 模型主体 bf16，线性层权重保留 fp32，计算时 cast 到 bf16 |
| 导出 | 训练后保存 `final_model.pt`，并生成 `final_model.int8.ptz` |

历史 naive baseline 记录显示，在 `8xH100 / 10min` 下会因 wallclock cap 停在约 `13780/20000` step，最终 int8+zlib roundtrip 的 `val_bpb` 约 `1.2244`，压缩后模型约 `15.8MB`，加代码后低于 16MB。

### 2.2 数据与验证

训练数据是预先分词的 uint16 binary shard。每个 shard 有固定 header，`load_data_shard()` 会校验 magic、版本、token 数和文件大小。

训练加载器：

- `TokenStream` 按文件名排序顺序读取 shard，读完后循环。
- `DistributedTokenLoader` 每步从连续 token stream 中切出每个 rank 的不重叠片段。
- 每个 batch 使用 shift 方式构造 `x = tokens[:-1]`、`y = tokens[1:]`。
- 当前 loader 没有随机采样、worker 或复杂 shuffle，复现实验较简单。

验证指标：

- `val_loss` 是 token-level cross entropy。
- `val_bpb` 是 tokenizer-agnostic bits-per-byte，是挑战实际关注指标。
- 脚本会基于 SentencePiece 构建 token 到 byte 数的 LUT，处理 leading space，并用验证集总 byte 数换算 BPB。
- 验证默认使用完整 `fineweb_val_*` split。

注意：如果改 tokenizer，必须保证 `VOCAB_SIZE` 与 tokenizer `.model` 匹配，并重新确认 BPB 计算没有偏差。历史规则对 tokenizer 改动审查更严。

### 2.3 模型结构

当前 `GPT` 是一个小型 GPT-like decoder，但带有若干 parameter-golf 风格设计：

- `tok_emb`: token embedding，默认同时作为 output head 使用。
- `RMSNorm`: embedding 后、每个 attention/MLP 前、输出前使用 RMSNorm。
- `CausalSelfAttention`: Q/K/V/proj 无 bias，Q/K 做 RMSNorm，RoPE 位置编码，Flash SDPA，支持 GQA。
- `q_gain`: 每个 attention head 一个可学习缩放，默认初始化 `QK_GAIN_INIT=1.5`。
- `MLP`: `Linear(dim, mlp_mult*dim) -> ReLU -> square -> Linear(hidden, dim)`。
- `Block`: attention 和 MLP 残差分支各有可学习 scale；还有 `resid_mix` 将当前状态与初始 embedding 状态 `x0` 混合。
- U-Net-like skip：前半层保存 skip，后半层按反序加回，并有 `skip_weights`。
- logits 使用 `LOGIT_SOFTCAP * tanh(logits / LOGIT_SOFTCAP)` 做 softcap。

这套结构的参数主要花在：

- token embedding / tied output embedding；
- 每层 attention 的 Q/K/V/O 矩阵；
- 每层 MLP 的 fc/proj 矩阵；
- 少量 scale、gain、skip、mix 控制参数。

由于提交物按压缩后的权重加代码计入 16MB，不能只看训练时参数量；还要看 int8/zlib 后大小是否留有余量。

### 2.4 优化与训练循环

训练主路径：

1. 初始化分布式环境。即使单卡，也用 `torchrun --nproc_per_node=1` 更贴近默认用法。
2. 要求 CUDA，开启 TF32 matmul，SDPA 只启用 Flash backend。
3. 固定 Python/NumPy/PyTorch seed。
4. 加载 tokenizer 与验证 tokens，构建 BPB 统计 LUT。
5. 构造模型，转换到 bf16；`CastedLinear` 权重保持 fp32；低维控制参数恢复 fp32。
6. `torch.compile(base_model, dynamic=False, fullgraph=True)` 编译模型。
7. 按参数类型拆分优化器：
   - tied embedding：Adam，`TIED_EMBED_LR=0.05`；
   - untied head：Adam，`HEAD_LR=0.008`；
   - transformer 2D 矩阵：Muon，`MATRIX_LR=0.04`；
   - scalar/vector/control 参数：Adam，`SCALAR_LR=0.04`。
8. 做 `WARMUP_STEPS=20` 编译/优化器 warmup，然后恢复初始模型与优化器状态，不把 warmup 算入正式训练。
9. 主循环中按 `grad_accum_steps = 8 // WORLD_SIZE` 累积梯度。单卡时 `WORLD_SIZE=1`，因此默认累积 8 个 micro step；8 卡时累积 1 个 micro step。
10. 学习率使用 warmdown：当剩余 wallclock 不足 `WARMDOWN_ITERS` 个 step 的估算时间时，按剩余时间线性降到 0。
11. 达到 `ITERATIONS` 或 `MAX_WALLCLOCK_SECONDS` 后停止。
12. 训练后保存原始 fp/bf 权重，再量化为 int8+zlib，重新加载 roundtrip 权重并输出最终 `val_bpb`。

## 3. 可调超参

所有主要超参都通过环境变量覆盖，集中在 `Hyperparameters` 类中。

### 3.1 数据与运行

| 环境变量 | 默认值 | 作用 | 调参建议 |
| --- | ---: | --- | --- |
| `DATA_PATH` | `./data/datasets/fineweb10B_sp1024` | 训练/验证数据目录 | 换 tokenizer 或数据导出时一起改。 |
| `TOKENIZER_PATH` | `./data/tokenizers/fineweb_1024_bpe.model` | SentencePiece tokenizer | 必须与 `VOCAB_SIZE` 匹配。 |
| `VOCAB_SIZE` | `1024` | 模型词表大小 | 大 vocab 降低序列长度压力但增加 embedding 参数和 artifact 体积。历史强配置常用 4096/8192，但需要配套压缩。 |
| `RUN_ID` | UUID | 日志文件名 | 建议包含关键超参和 seed。 |
| `SEED` | `1337` | 随机种子 | 最后要多 seed 验证稳定性。 |

### 3.2 训练长度与 batch

| 环境变量 | 默认值 | 作用 | 调参建议 |
| --- | ---: | --- | --- |
| `ITERATIONS` | `20000` | 最大训练步数 | 单卡 1h 可设较大，让 wallclock cap 决定停止。 |
| `MAX_WALLCLOCK_SECONDS` | `600` | 训练 wallclock 上限，0 表示不限 | 单卡 1h 场景设 `3600`。 |
| `WARMUP_STEPS` | `20` | 编译/路径预热步数，不计入正式训练 | 单卡也保留，能减少计时噪声。 |
| `WARMDOWN_ITERS` | `1200` | 最后阶段 LR 衰减长度 | 1h 训练可扩大，例如 2000-5000，取决于实际 step 数。 |
| `TRAIN_BATCH_TOKENS` | `524288` | 全局 token batch | 单卡默认会被 8 次累积实现。显存够时可试增大；若 step 太慢可降到 262144。 |
| `TRAIN_SEQ_LEN` | `1024` | 训练和验证分块长度 | 2048/4096 可能改善上下文，但显存和 step time 增加。 |
| `VAL_BATCH_SIZE` | `524288` | 验证 batch tokens | 只影响验证速度/显存。 |
| `VAL_LOSS_EVERY` | `1000` | 周期验证间隔 | 1h 实验建议 1000-3000；调参短跑可设 0 或较大。 |
| `TRAIN_LOG_EVERY` | `200` | 训练日志间隔 | 对性能影响小，保持 100-500 均可。 |

### 3.3 模型结构

| 环境变量 | 默认值 | 作用 | 调参建议 |
| --- | ---: | --- | --- |
| `NUM_LAYERS` | `9` | transformer block 数 | 增层通常提升能力但增加时间和体积；10/11 是早期常见方向。 |
| `MODEL_DIM` | `512` | hidden width | 体积敏感，增大需配合更强量化或小 vocab。 |
| `NUM_HEADS` | `8` | attention heads | 需整除 `MODEL_DIM`，head_dim 需为偶数。 |
| `NUM_KV_HEADS` | `4` | KV heads | GQA 压缩 KV 参数和计算；必须整除 `NUM_HEADS`。 |
| `MLP_MULT` | `2` | MLP hidden 倍率 | 3/4 常能增容量，但 MLP 参数占比大，artifact 更紧。 |
| `TIE_EMBEDDINGS` | `1` | 输入输出 embedding 绑定 | 16MB 下通常保持绑定。关闭会增加 head 参数。 |
| `TIED_EMBED_INIT_STD` | `0.005` | tied embedding 初始化 std | 与 embedding LR、vocab 改动相关。 |
| `ROPE_BASE` | `10000` | RoPE base | 长上下文时可调。 |
| `QK_GAIN_INIT` | `1.5` | attention Q gain 初值 | 历史记录中更大值如 5 左右曾有效，值得 sweep。 |
| `LOGIT_SOFTCAP` | `30.0` | logits softcap | 可影响稳定性和校准，通常小范围调。 |

### 3.4 优化器与学习率

| 环境变量 | 默认值 | 作用 | 调参建议 |
| --- | ---: | --- | --- |
| `TIED_EMBED_LR` | `0.05` | tied embedding Adam LR | 对小 vocab baseline 很敏感。 |
| `EMBED_LR` | `0.6` | 非 tied input embedding LR | 仅 `TIE_EMBEDDINGS=0` 时主要使用。 |
| `HEAD_LR` | `0.008` | 非 tied output head LR | 仅 untied 时使用。 |
| `MATRIX_LR` | `0.04` | Muon 矩阵参数 LR | 核心超参之一。 |
| `SCALAR_LR` | `0.04` | scale/gain/skip 等小参数 Adam LR | 对控制参数稳定性有影响。 |
| `MUON_MOMENTUM` | `0.95` | Muon momentum 目标值 | 历史记录常试 0.95-0.99。 |
| `MUON_BACKEND_STEPS` | `5` | Newton-Schulz 正交化迭代步数 | 更高可能更稳但更慢。 |
| `MUON_MOMENTUM_WARMUP_START` | `0.85` | Muon momentum warmup 起点 | 影响早期训练。 |
| `MUON_MOMENTUM_WARMUP_STEPS` | `500` | Muon momentum warmup 步数 | 长训练可适度增加。 |
| `BETA1` | `0.9` | Adam beta1 | embedding/小参数相关。 |
| `BETA2` | `0.95` | Adam beta2 | 小模型常用较低 beta2。 |
| `ADAM_EPS` | `1e-8` | Adam epsilon | 通常少动。 |
| `GRAD_CLIP_NORM` | `0.0` | 全局梯度裁剪，0 为关闭 | 若加大 LR/模型后不稳，可开启如 1.0。 |

### 3.5 量化与 artifact

脚本中还有一组导出相关常量，默认不是通过 `Hyperparameters` 管理：

| 名称 | 默认值 | 作用 |
| --- | ---: | --- |
| `INT8_KEEP_FLOAT_MAX_NUMEL` | `65536` | 小 tensor 保持浮点，避免量化元数据开销。 |
| `INT8_KEEP_FLOAT_STORE_DTYPE` | `float16` | 小浮点 tensor 存储 dtype。 |
| `INT8_PER_ROW_SCALE_DTYPE` | `float16` | 2D tensor per-row scale dtype。 |
| `INT8_CLIP_PERCENTILE` | `99.99984` | int8 clipping 分位数。 |
| `CONTROL_TENSOR_NAME_PATTERNS` | 多个 scale/gain/skip/mix 名称 | 控制参数保持高精度或特殊处理。 |

当前 baseline 的 int8+zlib artifact 非常接近 16MB 上限。任何增加 `VOCAB_SIZE`、`MODEL_DIM`、`NUM_LAYERS`、`MLP_MULT` 或关闭 tied embedding 的改动，都需要同步观察：

- `Serialized model int8+zlib`
- `Code size`
- `Total submission size int8+zlib`
- `final_int8_zlib_roundtrip val_bpb`

## 4. 单卡 H100 一小时的实验建议

### 4.1 首次复现命令

建议先用官方 sp1024 baseline 在单卡上跑通，并把 wallclock 改为 1h：

```bash
RUN_ID=baseline_sp1024_1xh100_1h \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MAX_WALLCLOCK_SECONDS=3600 \
VAL_LOSS_EVERY=2000 \
TRAIN_LOG_EVERY=200 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

单卡下脚本会自动设置 `grad_accum_steps=8`，因此默认全局 batch 仍是 `524288` tokens/step，但每个 optimizer step 要跑 8 个 microbatch。和 8 卡 10 分钟相比，单卡一小时的总吞吐大致是 `1/8 * 6 = 75%`，所以默认配置可能仍无法完成 8 卡 10 分钟 baseline 的全部训练 token 量。

### 4.2 优先 sweep 的低风险超参

在不大改结构的前提下，优先试：

| 方向 | 示例 |
| --- | --- |
| 训练时长适配 | `MAX_WALLCLOCK_SECONDS=3600`，`ITERATIONS=50000`，让 wallclock 决定停点。 |
| warmdown | `WARMDOWN_ITERS=2000/3500/5000`。一小时训练中，过短 warmdown 可能收尾粗糙。 |
| Muon momentum | `MUON_MOMENTUM=0.95/0.97/0.99`，同时观察稳定性。 |
| LR | `MATRIX_LR=0.03/0.04/0.05`，`TIED_EMBED_LR=0.03/0.05/0.07`，`SCALAR_LR=0.02/0.04/0.06`。 |
| QK gain | `QK_GAIN_INIT=1.5/3.0/5.0/5.25`。 |
| seq len | `TRAIN_SEQ_LEN=1024/2048`。若显存足够且 step time 可接受，再试 4096。 |
| batch tokens | `TRAIN_BATCH_TOKENS=262144/524288/1048576`。小 batch 步数更多，大 batch 梯度更稳；最终看 BPB。 |

### 4.3 中等风险结构改动

这些改动可能带来收益，但更容易触碰 16MB 或训练速度：

- `NUM_LAYERS=10/11`：早期记录显示增加层数有效，但体积和计算都会上升。
- `MLP_MULT=3/4`：增加 FFN 容量，参数增长明显；需看压缩后大小。
- `VOCAB_SIZE=4096/8192`：历史强提交大量使用更大 tokenizer，但需要重新下载/导出对应数据和 tokenizer，并改进量化压缩。
- `TRAIN_SEQ_LEN=4096+`：长上下文常有收益，但单卡 step time 压力较大。
- 参考 `records/` 引入 depth recurrence、parallel residual、TTT、GPTQ/AWQ 等技巧。这些通常不只是超参，需要代码改动和合法性检查。

### 4.4 实验记录建议

每次实验至少记录：

- 完整命令和 git diff；
- `RUN_ID`、seed、数据/tokenizer 版本；
- stop step、train time、step_avg；
- pre-quant `val_bpb`；
- `final_int8_zlib_roundtrip_exact val_bpb`；
- int8+zlib 总 submission size；
- peak memory。

最终比较应以 `final_int8_zlib_roundtrip_exact val_bpb` 为准，因为提交评估看到的是压缩再加载后的模型，而不是训练内存中的原始权重。

## 5. 当前 baseline 的主要瓶颈

1. 16MB 约束非常紧：默认 baseline 已接近上限，直接扩大模型空间有限。
2. sp1024 tokenizer 序列较长：小 vocab 节省 embedding 参数，但 BPB 可能吃亏。
3. 单卡一小时吞吐未必超过 8 卡 10 分钟：需要关注 step_avg 和实际 tokens seen。
4. 后训练 int8 量化会带来明显 BPB 损失：结构/训练收益若小于量化损失，最终 artifact 分数不会改善。
5. 当前根目录脚本追求简洁，缺少历史 SOTA 中的 TTT、GPTQ、CaseOps、深度复用等高级机制。

因此，推荐路线是：先复现单卡一小时 baseline；再做 LR/warmdown/QK gain/seq len 的小 sweep；确认收益后，再从 `records/track_10min_16mb/` 里挑一个成熟技巧移植到根目录 `train_gpt.py`。
