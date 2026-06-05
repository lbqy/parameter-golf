# 当前会话迁移速查

本文记录当前 workspace、容器、数据、代码改动和已完成 run，方便新会话继续实验。

## 运行环境

| 项目 | 当前值 |
| --- | --- |
| 宿主项目目录 | `/public/home/lvbqy/project/parameter-golf` |
| 容器名 | `lbqy0` |
| 容器项目目录 | `/base/project/parameter-golf` |
| 主要训练 Python | `/opt/conda/bin/python` |
| 主要训练 torch | `2.6.0+cu126` |
| `gf` 环境 torch | `2.12.0+cu130` |
| 推荐训练环境 | 容器 base 环境，即直接用 `/opt/conda/bin/torchrun` |
| GPU | 8 x NVIDIA H100 80GB HBM3 |
| 常用空闲卡 | 先查 `nvidia-smi`；最近实验用 `CUDA_VISIBLE_DEVICES=6,7` |

注意：`gf` 环境在默认 full batch 下曾触发 `torch.compile/Inductor` backward shape 错误；base 环境的 torch `2.6.0+cu126` 已通过 full batch、2 卡训练。

## 数据位置

| 数据 | 路径 | 状态 |
| --- | --- | --- |
| SP1024 dataset | `/base/project/parameter-golf/data/datasets/fineweb10B_sp1024` | 80 train shards + full val |
| SP1024 tokenizer | `/base/project/parameter-golf/data/tokenizers/fineweb_1024_bpe.model` | 可用 |
| SP8192 dataset | `/base/datasets/SP8192/datasets/fineweb10B_sp8192` | 80 train shards + full val |
| SP8192 tokenizer | `/base/datasets/SP8192/tokenizers/fineweb_8192_bpe.model` | 可用 |

SP8192 下载使用用户代理：

```bash
export HTTP_PROXY=http://59.66.143.200:7897
export HTTPS_PROXY=http://59.66.143.200:7897
export ALL_PROXY=http://59.66.143.200:7897
unset HF_ENDPOINT
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf \
python3 cached_challenge_fineweb.py --variant sp8192 --train-shards 80
```

`HF_ENDPOINT=https://hf-mirror.com` 在当前 `huggingface_hub` 版本下触发 metadata 校验问题；成功路线是用代理访问官方 `huggingface.co`。

## 当前 `train_gpt.py` 改动

在 baseline 量化导出基础上新增了可选 mixed low-bit packed quantization：

| 环境变量 | 默认 | 作用 |
| --- | ---: | --- |
| `MATRIX_QUANT_BITS` / `QUANT_BITS` | `8` | 大 tensor 默认矩阵量化 bitwidth |
| `EMBED_QUANT_BITS` / `EMBED_BITS` | 同 `MATRIX_QUANT_BITS` | `tok_emb.weight` 量化 bitwidth |
| `QUANT_CLIP_SEARCH` | `0` | 是否启用多个 clip percentile 的 MSE 搜索 |
| `QUANT_CLIP_SEARCH_PCTS` | `0.999,0.9995,0.9999,0.99999,1.0` | clip-search 候选分位 |

重要说明：当前实现是 **packed int6/int7 后训练量化 + clip-search**，还不是完整 Hessian-aware GPTQ。文档和 run id 中应避免把它误称为 full GPTQ。

## 已完成 Run

| Run | 配置 | 资源/时长 | final BPB | artifact | 结论 |
| --- | --- | --- | ---: | ---: | --- |
| `baseline_sp1024_1xh100_1h_seed1337_torch26` | SP1024, int8 zlib | 1 x H100, 1h | `1.23203866` | `15,869,303` | 合规 baseline |
| `sp8192_clipsearch_int6int7_2xh100_30m_seed1337` | SP8192, matrix int6, embed int7, clip-search | 2 x H100, 30m | `1.21914623` | `15,811,722` | 合规，优于 SP1024 baseline，但量化损失大 |
| `sp8192_int8_2xh100_30m_seed1337` | SP8192, default int8 | 2 x H100, 30m | `1.19337027` | `19,379,256` | 质量好但超 16MB，作为上限参考 |

对应 summary：

- `baseline_sp1024_1xh100_1h_seed1337_torch26.summary.json`
- `sp8192_clipsearch_int6int7_2xh100_30m_seed1337.summary.json`
- `sp8192_int8_2xh100_30m_seed1337.summary.json`

## 常用命令

查 GPU：

```bash
docker exec lbqy0 bash -lc \
'nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits; \
 echo PROCS; \
 nvidia-smi --query-compute-apps=gpu_bus_id,pid,used_memory --format=csv,noheader,nounits 2>/dev/null || true'
```

SP8192 mixed int6/int7 2 卡 30 分钟：

```bash
docker exec lbqy0 bash -lc '
cd /base/project/parameter-golf
CUDA_VISIBLE_DEVICES=6,7 \
MAX_WALLCLOCK_SECONDS=1800 \
RUN_ID=sp8192_clipsearch_int6int7_2xh100_30m_seed1337 \
DATA_PATH=/base/datasets/SP8192/datasets/fineweb10B_sp8192 \
TOKENIZER_PATH=/base/datasets/SP8192/tokenizers/fineweb_8192_bpe.model \
VOCAB_SIZE=8192 \
MATRIX_QUANT_BITS=6 \
EMBED_QUANT_BITS=7 \
QUANT_CLIP_SEARCH=1 \
VAL_LOSS_EVERY=0 \
TRAIN_LOG_EVERY=200 \
/opt/conda/bin/torchrun --standalone --nproc_per_node=2 train_gpt.py
'
```

SP8192 int8 上限参考 2 卡 30 分钟：

```bash
docker exec lbqy0 bash -lc '
cd /base/project/parameter-golf
CUDA_VISIBLE_DEVICES=6,7 \
MAX_WALLCLOCK_SECONDS=1800 \
RUN_ID=sp8192_int8_2xh100_30m_seed1337 \
DATA_PATH=/base/datasets/SP8192/datasets/fineweb10B_sp8192 \
TOKENIZER_PATH=/base/datasets/SP8192/tokenizers/fineweb_8192_bpe.model \
VOCAB_SIZE=8192 \
VAL_LOSS_EVERY=0 \
TRAIN_LOG_EVERY=200 \
/opt/conda/bin/torchrun --standalone --nproc_per_node=2 train_gpt.py
'
```

## 下一步建议

当前最明显瓶颈是 SP8192 低比特量化损失：

```text
SP8192 int8 final BPB:       1.19337027, 但 artifact 超 16MB
SP8192 int6/int7 final BPB:  1.21914623, 合规
量化目标：尽量接近 1.193，同时压回 16MB 内
```

建议优先尝试：

- `MATRIX_QUANT_BITS=7 EMBED_QUANT_BITS=7` 或 `MATRIX_QUANT_BITS=6 EMBED_QUANT_BITS=8`，先用 0/1 step 冒烟看 payload，再跑 30m。
- 迁移历史记录中的 Hessian-aware GPTQ，降低当前 int6/int7 的 roundtrip gap。
- 不建议先扩大模型；SP8192 int8 已经超限，当前问题是压缩质量而不是模型容量。
