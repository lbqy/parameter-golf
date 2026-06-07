# CaseOps 数据交接记录

日期：2026-06-08

目的：记录容器内 CaseOps raw docs、tokenizer、smoke 数据和后续使用方式，便于跨会话恢复。

## 容器与路径

| 项目 | 路径 / 值 |
| --- | --- |
| 容器 | `b5e2809a5863` / `lbqy0` |
| 容器内项目目录 | `/base/project/parameter-golf` |
| CaseOps 根目录 | `/base/datasets/CaseOps` |
| raw docs | `/base/datasets/CaseOps/raw/docs_selected.jsonl` |
| raw manifest | `/base/datasets/CaseOps/raw/docs_selected.source_manifest.json` |
| HF manifest | `/base/datasets/CaseOps/raw/manifest.json` |
| CaseOps tokenizer 源 | `/base/project/parameter-golf/records/track_10min_16mb/2026-04-27_SP8192_LQER_SparseGate_BOSSmearFix_9HpStack_1.0611/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model` |
| prepare 脚本 | `/base/project/parameter-golf/records/track_10min_16mb/2026-04-27_SP8192_LQER_SparseGate_BOSSmearFix_9HpStack_1.0611/prepare_caseops_data.py` |

## 当前数据状态

`docs_selected.jsonl` 已下载完成，大小约 44GB：

```text
/base/datasets/CaseOps/raw/manifest.json                         1,925 bytes
/base/datasets/CaseOps/raw/docs_selected.source_manifest.json       481 bytes
/base/datasets/CaseOps/raw/docs_selected.jsonl             46,231,122,868 bytes
```

raw docs 读取检查通过：首行是 JSON object，包含 `text` 字段。

当前尚未生成完整 CaseOps token dataset。完整训练前仍需运行 `prepare_caseops_data.py`，生成：

```text
/base/datasets/CaseOps/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/
  fineweb_train_*.bin
  fineweb_val_*.bin
  fineweb_val_bytes_*.bin
```

## Smoke 结果

已用前 64 篇文档生成小样本 smoke 数据：

```text
/tmp/caseops_smoke/raw/docs_64.jsonl
/tmp/caseops_smoke/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model
/tmp/caseops_smoke/out/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/
  fineweb_train_000000.bin
  fineweb_val_000000.bin
  fineweb_val_bytes_000000.bin
```

prepare smoke 通过：

```text
loaded sp: vocab=8192
done. docs=64 train_shards=1 val_shards=1
```

sidecar 合规检查通过：

```text
fineweb_val_000000.bin       magic=20240520 version=1 tokens=22397
fineweb_val_bytes_000000.bin magic=20240520 version=1 tokens=22397
fineweb_train_000000.bin     magic=20240520 version=1 tokens=34965
bos_count=16
bos_byte_values=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
CASEOPS_SMOKE_OK
```

根脚本 loader smoke 通过。命令使用 `/tmp/caseops_smoke` 数据、GPU 1、极小模型、1 step；完成 train/eval/int8 roundtrip：

```text
train_loader:dataset:fineweb10B_sp8192_lossless_caps_caseops_v1_reserved train_shards:1
val_loader:shards pattern=/tmp/caseops_smoke/out/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/fineweb_val_*.bin tokens:44672
final_int8+zlib_roundtrip_exact val_loss:9.43411015 val_bpb:5.85013344
```

注意：当前根目录 `train_gpt.py` 仍只显示 `val_bpb:enabled tokenizer_kind=sentencepiece`，尚未读取 `fineweb_val_bytes_*.bin`。因此该 loader smoke 只能证明 CaseOps token shards 可被当前训练路径读取，不能证明最终 CaseOps BPB 已按原始 bytes 合规计分。

## 完整 CaseOps 生成命令

在容器内运行：

```bash
cd /base/project/parameter-golf

export CASEOPS_ROOT=/base/datasets/CaseOps
export CASEOPS_REC=records/track_10min_16mb/2026-04-27_SP8192_LQER_SparseGate_BOSSmearFix_9HpStack_1.0611

mkdir -p "$CASEOPS_ROOT/tokenizers"
cp "$CASEOPS_REC/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model" \
  "$CASEOPS_ROOT/tokenizers/"

python3 "$CASEOPS_REC/prepare_caseops_data.py" \
  --docs "$CASEOPS_ROOT/raw/docs_selected.jsonl" \
  --out "$CASEOPS_ROOT" \
  --sp "$CASEOPS_ROOT/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model" \
  --val-docs 10000
```

完整生成后，训练/评估路径应设为：

```bash
DATA_PATH=/base/datasets/CaseOps/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved
TOKENIZER_PATH=/base/datasets/CaseOps/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model
VOCAB_SIZE=8192
BOS_ID=1
```

## 下一步

1. 运行完整 `prepare_caseops_data.py`，生成 full train/val token shards 和 `fineweb_val_bytes_*.bin`。
2. 在根目录 `train_gpt.py` 接入 CaseOps byte sidecar BPB：eval 时若存在 `fineweb_val_bytes_*.bin`，分母使用 sidecar 的原始 byte count，而不是 tokenizer LUT。
3. smoke 日志必须出现类似 `val_bpb:byte_sidecar:enabled`，并检查 BOS sidecar byte 为 0。
4. 通过合规 BPB smoke 后，再启动 S4-C1 短训和 S4-C2 完整 1h CaseOps 基线。
