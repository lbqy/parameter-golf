"""
The `train_gpt.py` and `train_gpt_mlx.py` scripts are intended as good launching-off points for new participants, not SOTA configs. We'll accept PRs that tune, improve, or simplify these scripts without significantly increasing complexity, but competitive submissions should stay in the `/records` folder.

Hard stop: To keep readable for newcomers, let's make sure `train_gpt.py` and `train_gpt_mlx.py` never are longer than 1500 lines.
"""

from __future__ import annotations

import copy
import glob
import io
import math
import os
import random
import subprocess
import sys
import time
import uuid
import zlib
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

# -----------------------------
# HYPERPARAMETERS
# -----------------------------
# Default Simple Baseline run:
# - 9 transformer blocks at width 512
# - 8 attention heads with 4 KV heads (GQA) and 2x MLP expansion
# - vocab size 1024, sequence length 1024, tied embeddings
# - 524,288 train tokens per step for 20,000 iterations with a ~10 minute cap

class Hyperparameters:
    # Data paths are shard globs produced by the existing preprocessing pipeline.
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))

    # Validation cadence and batch size. Validation always uses the full fineweb_val split.
    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 1000))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 200))

    # Training length.
    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 1200))
    warmdown_frac = float(os.environ.get("WARMDOWN_FRAC", 0.0))
    min_lr = float(os.environ.get("MIN_LR", 0.0))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))

    # Model shape.
    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers = int(os.environ.get("NUM_LAYERS", 9))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult = int(os.environ.get("MLP_MULT", 2))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))

    # Optimizer hyperparameters.
    embed_lr = float(os.environ.get("EMBED_LR", 0.6))
    head_lr = float(os.environ.get("HEAD_LR", 0.008))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.05))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.04))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.04))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.95))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 500))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.0))

    # Export / quantization mode: rtn (default int8+zlib) or gptq (Hessian SDClip).
    quant_mode = os.environ.get("QUANT_MODE", "rtn").lower()
    compressor = os.environ.get("COMPRESSOR", "zlib" if quant_mode == "rtn" else "brotli")
    gptq_calibration_batches = int(os.environ.get("GPTQ_CALIBRATION_BATCHES", 32))
    matrix_bits = int(os.environ.get("MATRIX_BITS", 6))
    embed_bits = int(os.environ.get("EMBED_BITS", 8))
    matrix_clip_sigmas = float(os.environ.get("MATRIX_CLIP_SIGMAS", 12.85))
    attn_clip_sigmas = float(os.environ.get("ATTN_CLIP_SIGMAS", 13.0))
    mlp_clip_sigmas = float(os.environ.get("MLP_CLIP_SIGMAS", 11.5))
    embed_clip_sigmas = float(os.environ.get("EMBED_CLIP_SIGMAS", 14.0))
    gptq_scale_mode = os.environ.get("GPTQ_SCALE_MODE", "sigma").lower()
    gptq_clip_quantile = float(os.environ.get("GPTQ_CLIP_QUANTILE", 0.9999984))
    gptq_damp = float(os.environ.get("GPTQ_DAMP", 0.01))
    gptq_act_order = bool(int(os.environ.get("GPTQ_ACT_ORDER", "1")))
    gptq_error_scale = float(os.environ.get("GPTQ_ERROR_SCALE", 1.0))
    gptq_scale_floor = float(os.environ.get("GPTQ_SCALE_FLOOR", 1.0))
    lqer_enabled = bool(int(os.environ.get("LQER_ENABLED", "0")))
    lqer_rank = int(os.environ.get("LQER_RANK", 4))
    lqer_top_k = int(os.environ.get("LQER_TOP_K", 3))
    lqer_factor_bits = int(os.environ.get("LQER_FACTOR_BITS", 4))
    lqer_asym_enabled = bool(int(os.environ.get("LQER_ASYM_ENABLED", "1")))
    lqer_asym_group = int(os.environ.get("LQER_ASYM_GROUP", 64))

    # Stage-3 experiment switches.
    loader_mode = os.environ.get("LOADER_MODE", "sequential").lower()
    coprime_file_stride = int(os.environ.get("COPRIME_FILE_STRIDE", 17))
    coprime_token_stride = int(os.environ.get("COPRIME_TOKEN_STRIDE", 1_048_573))
    recurrence_extra_passes = int(os.environ.get("RECURRENCE_EXTRA_PASSES", 0))
    recurrence_start_layer = int(os.environ.get("RECURRENCE_START_LAYER", 3))
    recurrence_end_layer = int(os.environ.get("RECURRENCE_END_LAYER", 5))
    recurrence_start_frac = float(os.environ.get("RECURRENCE_START_FRAC", 0.35))

# -----------------------------
# MUON OPTIMIZER 
# -----------------------------
# 
# As borrowed from modded-nanogpt
# Background on Muon: https://kellerjordan.github.io/posts/muon/

def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    # Orthogonalize a 2D update matrix with a fast Newton-Schulz iteration.
    # Muon uses this to normalize matrix-shaped gradients before applying them.
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float, momentum: float, backend_steps: int, nesterov: bool = True):
        super().__init__(
            params,
            dict(lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov),
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0

        for group in self.param_groups:
            params = group["params"]
            if not params:
                continue
            lr = group["lr"]
            momentum = group["momentum"]
            backend_steps = group["backend_steps"]
            nesterov = group["nesterov"]

            total_params = sum(int(p.numel()) for p in params)
            updates_flat = torch.zeros(total_params, device=params[0].device, dtype=torch.bfloat16)

            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g = p.grad
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    if nesterov:
                        g = g.add(buf, alpha=momentum)
                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                    # Scale correction from Muon reference implementations.
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr : curr + p.numel()] = g.reshape(-1)
                curr += p.numel()

            if distributed:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            curr = 0
            for p in params:
                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
                p.add_(g, alpha=-lr)
                curr += p.numel()

        return loss


# -----------------------------
# TOKENIZER-AGNOSTIC EVALUATION SETUP 
# -----------------------------
#
# It's common for small models have a large fraction of their parameters be embeddings, since the 2 * d_model * d_vocab vectors can be gigantic.
# Instead of locking the tokenizer, we let you bring your own and calculate our validation metrics on the average compression of the validation set.
# We calculate BPB (bits-per-byte) instead of validation loss, so we need methods to count the number of bits per token in the tokenizer.
# Note: Submissions that edit the tokenizer will be examined more carefully, since screwing this up might unjustly improve your score.

def build_sentencepiece_luts(
    sp: spm.SentencePieceProcessor, vocab_size: int, device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_np[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_np[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("▁"):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
    )


def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    # The export pipeline writes the fixed first-50k-doc validation set to fineweb_val_*.
    tokens = torch.cat([load_data_shard(file) for file in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split is too short for TRAIN_SEQ_LEN={seq_len}")
    return tokens[: usable + 1]


def eval_val(
    args: Hyperparameters,
    model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    grad_accum_steps: int,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
) -> tuple[float, float]:
    # Validation computes two metrics:
    # - val_loss: token cross-entropy (natural log)
    # - val_bpb: tokenizer-agnostic compression metric used by the challenge
    local_batch_tokens = args.val_batch_size // (world_size * grad_accum_steps)
    if local_batch_tokens < args.train_seq_len:
        raise ValueError(
            "VAL_BATCH_SIZE must provide at least one sequence per rank; "
            f"got VAL_BATCH_SIZE={args.val_batch_size}, WORLD_SIZE={world_size}, "
            f"GRAD_ACCUM_STEPS={grad_accum_steps}, TRAIN_SEQ_LEN={args.train_seq_len}"
        )
    local_batch_seqs = local_batch_tokens // args.train_seq_len
    total_seqs = (val_tokens.numel() - 1) // args.train_seq_len
    seq_start = (total_seqs * rank) // world_size
    seq_end = (total_seqs * (rank + 1)) // world_size
    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)

    model.eval()
    with torch.inference_mode():
        for batch_seq_start in range(seq_start, seq_end, local_batch_seqs):
            batch_seq_end = min(batch_seq_start + local_batch_seqs, seq_end)
            raw_start = batch_seq_start * args.train_seq_len
            raw_end = batch_seq_end * args.train_seq_len + 1
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, args.train_seq_len)
            y = local[1:].reshape(-1, args.train_seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                batch_loss = model(x, y).detach()
            batch_token_count = float(y.numel())
            val_loss_sum += batch_loss.to(torch.float64) * batch_token_count
            val_token_count += batch_token_count
            prev_ids = x.reshape(-1)
            tgt_ids = y.reshape(-1)
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
            val_byte_count += token_bytes.to(torch.float64).sum()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)

    val_loss = val_loss_sum / val_token_count
    bits_per_token = val_loss.item() / math.log(2.0)
    tokens_per_byte = val_token_count.item() / val_byte_count.item()
    model.train()
    return float(val_loss.item()), float(bits_per_token * tokens_per_byte)

# -----------------------------
# POST-TRAINING QUANTIZATION
# -----------------------------
#
# It's silly to export our model, which is trained in bf16 and fp32, at that same precision.
# Instead, we get approximately the same model (with a small hit) by quantizing the model to int8 & zlib compressing.
# We can then decompress the model and run in higher precision for evaluation, after closing in under the size limit.

CONTROL_TENSOR_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights",
    ).split(",")
    if pattern
)
INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "INT8_KEEP_FLOAT_FP32_NAME_PATTERNS",
        ",".join(CONTROL_TENSOR_NAME_PATTERNS),
    ).split(",")
    if pattern
)
INT8_KEEP_FLOAT_MAX_NUMEL = 65_536
INT8_KEEP_FLOAT_STORE_DTYPE = torch.float16
INT8_PER_ROW_SCALE_DTYPE = torch.float16
INT8_CLIP_PERCENTILE = 99.99984
INT8_CLIP_Q = INT8_CLIP_PERCENTILE / 100.0
MATRIX_QUANT_BITS = int(os.environ.get("MATRIX_QUANT_BITS", os.environ.get("QUANT_BITS", "8")))
EMBED_QUANT_BITS = int(os.environ.get("EMBED_QUANT_BITS", os.environ.get("EMBED_BITS", str(MATRIX_QUANT_BITS))))
QUANT_CLIP_SEARCH = bool(int(os.environ.get("QUANT_CLIP_SEARCH", "0")))
QUANT_CLIP_SEARCH_PCTS = tuple(
    float(x)
    for x in os.environ.get("QUANT_CLIP_SEARCH_PCTS", "0.999,0.9995,0.9999,0.99999,1.0").split(",")
    if x
)
QUANT_BITS_OVERRIDES: list[tuple[str, int]] = []
for item in os.environ.get("QUANT_BITS_OVERRIDES", "").split(","):
    if not item:
        continue
    pattern, _, bits_s = item.partition(":")
    if not pattern or not bits_s:
        raise ValueError(f"Bad QUANT_BITS_OVERRIDES item: {item!r}")
    QUANT_BITS_OVERRIDES.append((pattern, int(bits_s)))

if not (2 <= MATRIX_QUANT_BITS <= 8 and 2 <= EMBED_QUANT_BITS <= 8):
    raise ValueError("MATRIX_QUANT_BITS and EMBED_QUANT_BITS must be in [2, 8]")
for pattern, bits in QUANT_BITS_OVERRIDES:
    if not (2 <= bits <= 8):
        raise ValueError(f"Override bits for {pattern!r} must be in [2, 8], got {bits}")

def tensor_nbytes(t: Tensor) -> int:
    return int(t.numel()) * int(t.element_size())


def compress_quant_bytes(data: bytes, compressor: str) -> bytes:
    if compressor == "zlib":
        return zlib.compress(data, level=9)
    if compressor == "brotli":
        import brotli

        return brotli.compress(data, quality=11)
    if compressor == "lzma":
        import lzma

        return lzma.compress(data, preset=6)
    raise ValueError(f"Unknown compressor: {compressor!r}")


def decompress_quant_bytes(data: bytes, compressor: str) -> bytes:
    if compressor == "zlib":
        return zlib.decompress(data)
    if compressor == "brotli":
        import brotli

        return brotli.decompress(data)
    if compressor == "lzma":
        import lzma

        return lzma.decompress(data)
    raise ValueError(f"Unknown compressor: {compressor!r}")

def keep_float_tensor(name: str, t: Tensor, passthrough_orig_dtypes: dict[str, str]) -> Tensor:
    if any(pattern in name for pattern in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS):
        return t.float().contiguous()
    if t.dtype in {torch.float32, torch.bfloat16}:
        passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
        return t.to(dtype=INT8_KEEP_FLOAT_STORE_DTYPE).contiguous()
    return t

def pack_unsigned_values(values: np.ndarray, bits: int) -> Tensor:
    flat = np.asarray(values, dtype=np.uint8).reshape(-1)
    if bits == 8:
        return torch.from_numpy(flat.copy()).contiguous()
    bit_shifts = np.arange(bits, dtype=np.uint8)
    bit_matrix = ((flat[:, None] >> bit_shifts[None, :]) & 1).astype(np.uint8, copy=False)
    packed = np.packbits(bit_matrix.reshape(-1), bitorder="little")
    return torch.from_numpy(packed.copy()).contiguous()


def unpack_unsigned_values(packed: Tensor, bits: int, numel: int) -> np.ndarray:
    packed_np = packed.detach().cpu().numpy().astype(np.uint8, copy=False)
    if bits == 8:
        return packed_np[:numel].copy()
    raw_bits = np.unpackbits(packed_np, bitorder="little")[: numel * bits].reshape(numel, bits)
    shifts = (1 << np.arange(bits, dtype=np.uint16))
    return (raw_bits.astype(np.uint16, copy=False) * shifts[None, :]).sum(axis=1).astype(np.uint8)


def quant_bits_for_tensor(name: str, t: Tensor) -> int:
    for pattern, bits in QUANT_BITS_OVERRIDES:
        if pattern in name:
            return bits
    if t.ndim == 2 and name == "tok_emb.weight":
        return EMBED_QUANT_BITS
    return MATRIX_QUANT_BITS


def pack_signed_quant(q: Tensor, bits: int) -> Tensor:
    qmin = -(1 << (bits - 1))
    if bits == 8:
        return q.to(torch.int8).contiguous()
    unsigned = (q.cpu().numpy().astype(np.int16, copy=False) - qmin).astype(np.uint8)
    return pack_unsigned_values(unsigned, bits)


def quantize_float_tensor(t: Tensor, bits: int = 8) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    qmax = (1 << (bits - 1)) - 1
    qmin = -(1 << (bits - 1))
    if t32.ndim == 2:
        # Matrices get one scale per row, which usually tracks output-channel
        # ranges much better than a single tensor-wide scale.
        best_q: Tensor | None = None
        best_scale: Tensor | None = None
        best_err = float("inf")
        pcts = QUANT_CLIP_SEARCH_PCTS if QUANT_CLIP_SEARCH else (INT8_CLIP_Q,)
        for pct in pcts:
            if not t32.numel():
                clip_abs = torch.empty((t32.shape[0],), dtype=torch.float32)
            elif pct >= 1.0:
                clip_abs = t32.abs().amax(dim=1)
            else:
                clip_abs = torch.quantile(t32.abs(), pct, dim=1)
            clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
            scale = (clip_abs / max(qmax, 1)).clamp_min(1.0 / max(qmax, 1))
            q = torch.clamp(torch.round(clipped / scale[:, None]), qmin, qmax).to(torch.int16).contiguous()
            if not QUANT_CLIP_SEARCH:
                best_q, best_scale = q, scale
                break
            recon = q.float() * scale[:, None]
            err = (t32 - recon).pow(2).mean().item()
            if err < best_err:
                best_q, best_scale, best_err = q, scale, err
        if best_q is None or best_scale is None:
            raise RuntimeError("quantization failed to select a scale")
        return pack_signed_quant(best_q, bits), best_scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()

    # Vectors / scalars use a simpler per-tensor scale.
    clip_abs = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / max(qmax, 1) if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), qmin, qmax).to(torch.int16).contiguous()
    return pack_signed_quant(q, bits), scale

def quantize_state_dict_int8(state_dict: dict[str, Tensor]):
    # Single supported clean-script export format:
    # - per-row int8 for 2D float tensors
    # - per-tensor int8 for other float tensors
    # - exact passthrough for non-floats
    # - passthrough for small float tensors, stored as fp16 to save bytes
    quantized: dict[str, Tensor] = {}
    scales: dict[str, Tensor] = {}
    dtypes: dict[str, str] = {}
    passthrough: dict[str, Tensor] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    qmeta: dict[str, dict[str, object]] = {}
    stats = dict.fromkeys(
        ("param_count", "num_tensors", "num_float_tensors", "num_nonfloat_tensors", "baseline_tensor_bytes", "int8_payload_bytes"),
        0,
    )

    for name, tensor in state_dict.items():
        t = tensor.detach().to("cpu").contiguous()
        stats["param_count"] += int(t.numel())
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += tensor_nbytes(t)

        if not t.is_floating_point():
            stats["num_nonfloat_tensors"] += 1
            passthrough[name] = t
            stats["int8_payload_bytes"] += tensor_nbytes(t)
            continue

        # Small float tensors are cheap enough to keep directly. We still downcast
        # fp32/bf16 passthrough tensors to fp16 so metadata does not dominate size.
        if t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL:
            kept = keep_float_tensor(name, t, passthrough_orig_dtypes)
            passthrough[name] = kept
            stats["int8_payload_bytes"] += tensor_nbytes(kept)
            continue

        stats["num_float_tensors"] += 1
        bits = quant_bits_for_tensor(name, t)
        q, s = quantize_float_tensor(t, bits=bits)
        if s.ndim > 0:
            qmeta[name] = {"scheme": "per_row", "axis": 0}
        if bits != 8:
            qmeta.setdefault(name, {}).update({"bits": bits, "shape": tuple(t.shape), "packed": True})
        quantized[name] = q
        scales[name] = s
        dtypes[name] = str(t.dtype).removeprefix("torch.")
        stats["int8_payload_bytes"] += tensor_nbytes(q) + tensor_nbytes(s)

    obj: dict[str, object] = {
        "__quant_format__": "int8_clean_per_row_v1",
        "quantized": quantized,
        "scales": scales,
        "dtypes": dtypes,
        "passthrough": passthrough,
    }
    if qmeta:
        obj["qmeta"] = qmeta
    if passthrough_orig_dtypes:
        obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    return obj, stats

def dequantize_state_dict_int8(obj: dict[str, object]) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    qmeta = obj.get("qmeta", {})
    passthrough_orig_dtypes = obj.get("passthrough_orig_dtypes", {})
    for name, q in obj["quantized"].items():
        dtype = getattr(torch, obj["dtypes"][name])
        s = obj["scales"][name]
        meta = qmeta.get(name, {})
        bits = int(meta.get("bits", 8))
        if meta.get("packed"):
            shape = tuple(int(x) for x in meta["shape"])
            unsigned = unpack_unsigned_values(q, bits, math.prod(shape))
            qmin = -(1 << (bits - 1))
            q = torch.from_numpy(unsigned.astype(np.int16) + qmin).reshape(shape)
        if qmeta.get(name, {}).get("scheme") == "per_row" or s.ndim > 0:
            s = s.to(dtype=torch.float32)
            # Broadcast the saved row scale back across trailing dimensions.
            out[name] = (q.float() * s.view(q.shape[0], *([1] * (q.ndim - 1)))).to(dtype=dtype).contiguous()
        else:
            scale = float(s.item())
            out[name] = (q.float() * scale).to(dtype=dtype).contiguous()
    for name, t in obj["passthrough"].items():
        # Restore small tensors, undoing the temporary fp16 storage cast if needed.
        out_t = t.detach().to("cpu").contiguous()
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            out_t = out_t.to(dtype=getattr(torch, orig_dtype)).contiguous()
        out[name] = out_t
    return out


# -----------------------------
# DATA LOADING 
# -----------------------------

def load_data_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    # SHARD HEADER INTS & SHARD_MAGIC
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * token_bytes
    if file.stat().st_size != expected_size:
        raise ValueError(f"Shard size mismatch for {file}: expected {expected_size} bytes")
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens_np.size != num_tokens:
        raise ValueError(f"Short read for {file}")
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


class TokenStream:
    # Reads shards sequentially and wraps around forever. The training loop therefore
    # has deterministic, simple streaming behavior with no sampling or workers.
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def _advance_file(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n: int) -> Tensor:
        chunks: list[Tensor] = []
        remaining = n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance_file()
                continue
            k = min(remaining, avail)
            chunks.append(self.tokens[self.pos : self.pos + k])
            self.pos += k
            remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)


class DistributedTokenLoader:
    # Each call consumes a contiguous chunk from the shared token stream, then slices out
    # one disjoint span per rank. The extra "+1" token lets us build (x, y) by shifting.
    def __init__(
        self,
        pattern: str,
        rank: int,
        world_size: int,
        device: torch.device,
        mode: str = "sequential",
        coprime_file_stride: int = 17,
        coprime_token_stride: int = 1_048_573,
    ):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.mode = mode
        self.step = 0
        self.coprime_file_stride = coprime_file_stride
        self.coprime_token_stride = coprime_token_stride
        if mode == "sequential":
            self.stream = TokenStream(pattern)
            self.files: list[Path] = []
            self.cached_file_idx = -1
            self.cached_tokens: Tensor | None = None
        elif mode == "coprime":
            self.stream = None
            self.files = [Path(p) for p in sorted(glob.glob(pattern))]
            if not self.files:
                raise FileNotFoundError(f"No files found for pattern: {pattern}")
            if math.gcd(coprime_file_stride, len(self.files)) != 1:
                raise ValueError(
                    f"COPRIME_FILE_STRIDE={coprime_file_stride} must be coprime with shard count {len(self.files)}"
                )
            self.cached_file_idx = -1
            self.cached_tokens = None
        else:
            raise ValueError(f"Unknown LOADER_MODE: {mode!r}")

    def _load_coprime_file(self, file_idx: int) -> Tensor:
        if file_idx != self.cached_file_idx or self.cached_tokens is None:
            self.cached_tokens = load_data_shard(self.files[file_idx])
            self.cached_file_idx = file_idx
        return self.cached_tokens

    def next_batch(self, global_tokens: int, seq_len: int, grad_accum_steps: int) -> tuple[Tensor, Tensor]:
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        per_rank_span = local_tokens + 1
        if self.mode == "sequential":
            assert self.stream is not None
            chunk = self.stream.take(per_rank_span * self.world_size)
            start = self.rank * per_rank_span
            local = chunk[start : start + per_rank_span].to(dtype=torch.int64)
        else:
            file_idx = (self.step * self.coprime_file_stride + self.rank) % len(self.files)
            tokens = self._load_coprime_file(file_idx)
            max_start = max(tokens.numel() - per_rank_span - 1, 1)
            offset = (self.step * self.coprime_token_stride + self.rank * per_rank_span) % max_start
            local = tokens[offset : offset + per_rank_span].to(dtype=torch.int64)
            self.step += 1
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

# -----------------------------
# TRANSFORMER MODULES
# -----------------------------

class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class CastedLinear(nn.Linear):
    # Keep weights in fp32 for optimizer/state quality, cast at matmul time for bf16 compute.
    def forward(self, x: Tensor) -> Tensor:
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, self.weight.to(x.dtype), bias)


def restore_low_dim_params_to_fp32(module: nn.Module) -> None:
    # Keep small/control parameters in fp32 even when the model body runs in bf16.
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (param.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)) and param.dtype != torch.float32:
                param.data = param.data.float()


class Rotary(nn.Module):
    # Caches cos/sin tables per sequence length on the current device.
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: Tensor | None = None
        self._sin_cached: Tensor | None = None

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        if (
            self._cos_cached is None
            or self._sin_cached is None
            or self._seq_len_cached != seq_len
            or self._cos_cached.device != device
        ):
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cos_cached = freqs.cos()[None, None, :, :]
            self._sin_cached = freqs.sin()[None, None, :, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float,
        qk_gain_init: float,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        kv_dim = self.num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(self.head_dim, base=rope_base)

    def forward(self, x: Tensor) -> Tensor:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            is_causal=True,
            enable_gqa=(self.num_kv_heads != self.num_heads),
        )
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.proj(y)


class MLP(nn.Module):
    # relu^2 MLP from the original modded-nanogpt setup
    def __init__(self, dim: int, mlp_mult: int):
        super().__init__()
        hidden = mlp_mult * dim
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        x = torch.relu(self.fc(x))
        return self.proj(x.square())


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        rope_base: float,
        qk_gain_init: float,
    ):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())

    def forward(self, x: Tensor, x0: Tensor) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(self.attn_norm(x))
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        model_dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        tie_embeddings: bool,
        tied_embed_init_std: float,
        logit_softcap: float,
        rope_base: float,
        qk_gain_init: float,
        recurrence_extra_passes: int = 0,
        recurrence_start_layer: int = 3,
        recurrence_end_layer: int = 5,
        recurrence_active: bool = False,
    ):
        super().__init__()
        if logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {logit_softcap}")
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap
        self.recurrence_extra_passes = recurrence_extra_passes
        self.recurrence_start_layer = recurrence_start_layer
        self.recurrence_end_layer = recurrence_end_layer
        self.recurrence_active = recurrence_active
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))
        self.blocks = nn.ModuleList(
            [
                Block(
                    model_dim,
                    num_heads,
                    num_kv_heads,
                    mlp_mult,
                    rope_base,
                    qk_gain_init,
                )
                for i in range(num_layers)
            ]
        )
        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        self._init_weights()

    def _init_weights(self) -> None:
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        for module in self.modules():
            if isinstance(module, nn.Linear) and getattr(module, "_zero_init", False):
                nn.init.zeros_(module.weight)

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        skips: list[Tensor] = []

        # First half stores skips; second half reuses them in reverse order.
        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0)
            if self.recurrence_active and self.recurrence_start_layer <= i <= self.recurrence_end_layer:
                for _ in range(self.recurrence_extra_passes):
                    x = self.blocks[i](x, x0)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            layer_idx = self.num_encoder_layers + i
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            x = self.blocks[layer_idx](x, x0)
            if self.recurrence_active and self.recurrence_start_layer <= layer_idx <= self.recurrence_end_layer:
                for _ in range(self.recurrence_extra_passes):
                    x = self.blocks[layer_idx](x, x0)

        x = self.final_norm(x).reshape(-1, x.size(-1))
        targets = target_ids.reshape(-1)
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            if self.lm_head is None:
                raise RuntimeError("lm_head is required when tie_embeddings=False")
            logits_proj = self.lm_head(x)
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        return F.cross_entropy(logits.float(), targets, reduction="mean")


# -----------------------------
# TRAINING
# -----------------------------

def main() -> None:
    global zeropower_via_newtonschulz5

    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)

    # -----------------------------
    # DISTRIBUTED + CUDA SETUP
    # -----------------------------

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 0:
        raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")
    if 8 % world_size != 0:
        raise ValueError(f"WORLD_SIZE={world_size} must divide 8 so grad_accum_steps stays integral")
    grad_accum_steps = 8 // world_size
    grad_scale = 1.0 / grad_accum_steps
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master_process = rank == 0

    # Fast math knobs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp

    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)

    logfile = None
    if master_process:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{args.run_id}.txt"
        print(logfile)

    def log0(msg: str, console: bool = True) -> None:
        if not master_process:
            return
        if console:
            print(msg)
        if logfile is not None:
            with open(logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)

    log0(code, console=False)
    log0("=" * 100, console=False)
    log0(f"Running Python {sys.version}", console=False)
    log0(f"Running PyTorch {torch.__version__}", console=False)
    log0(
        subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False).stdout,
        console=False,
    )
    log0("=" * 100, console=False)

    # -----------------------------
    # TOKENIZER + VALIDATION METRIC SETUP
    # -----------------------------

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if not args.tokenizer_path.endswith(".model"):
        raise ValueError(f"Script only setup for SentencePiece .model file: {args.tokenizer_path}")
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(
            f"VOCAB_SIZE={args.vocab_size} does not match tokenizer vocab_size={int(sp.vocab_size())}"
        )
    dataset_dir = Path(args.data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )
    log0(f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={args.tokenizer_path}")
    log0(f"train_loader:dataset:{dataset_dir.name} train_shards:{actual_train_files}")
    log0(
        f"loader_mode:{args.loader_mode} coprime_file_stride:{args.coprime_file_stride} "
        f"coprime_token_stride:{args.coprime_token_stride}"
    )
    log0(f"val_loader:shards pattern={args.val_files} tokens:{val_tokens.numel() - 1}")

    # -----------------------------
    # MODEL + OPTIMIZER SETUP
    # -----------------------------
    initial_recurrence_active = bool(int(os.environ.get("RECURRENCE_ACTIVE", "0"))) or (args.recurrence_extra_passes > 0 and args.recurrence_start_frac <= 0.0)
    base_model = GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
        recurrence_extra_passes=args.recurrence_extra_passes,
        recurrence_start_layer=args.recurrence_start_layer,
        recurrence_end_layer=args.recurrence_end_layer,
        recurrence_active=initial_recurrence_active,
    ).to(device).bfloat16()
    for module in base_model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(base_model)
    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)
    model: nn.Module = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False) if distributed else compiled_model

    # Optimizer split:
    # - token embedding (Adam) uses EMBED_LR
    # - untied lm_head (Adam) uses HEAD_LR
    # - matrix params in transformer blocks use MATRIX_LR via Muon
    # - vectors/scalars use SCALAR_LR via Adam
    block_named_params = list(base_model.blocks.named_parameters())
    matrix_params = [
        p
        for name, p in block_named_params
        if p.ndim == 2 and not any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    scalar_params = [
        p
        for name, p in block_named_params
        if p.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    if base_model.skip_weights.numel() > 0:
        scalar_params.append(base_model.skip_weights)
    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    optimizer_tok = torch.optim.Adam(
        [{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        fused=True,
    )
    optimizer_muon = Muon(
        matrix_params,
        lr=args.matrix_lr,
        momentum=args.muon_momentum,
        backend_steps=args.muon_backend_steps,
    )
    for group in optimizer_muon.param_groups:
        group["base_lr"] = args.matrix_lr
    optimizer_scalar = torch.optim.Adam(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        fused=True,
    )
    optimizers: list[torch.optim.Optimizer] = [optimizer_tok, optimizer_muon, optimizer_scalar]
    if base_model.lm_head is not None:
        optimizer_head = torch.optim.Adam(
            [{"params": [base_model.lm_head.weight], "lr": args.head_lr, "base_lr": args.head_lr}],
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            fused=True,
        )
        optimizers.insert(1, optimizer_head)

    n_params = sum(p.numel() for p in base_model.parameters())
    log0(f"model_params:{n_params}")
    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")
    log0("sdp_backends:cudnn=False flash=True mem_efficient=False math=False")
    log0(f"attention_mode:gqa num_heads:{args.num_heads} num_kv_heads:{args.num_kv_heads}")
    log0(
        f"recurrence_extra_passes:{args.recurrence_extra_passes} "
        f"recurrence_layers:{args.recurrence_start_layer}-{args.recurrence_end_layer} "
        f"recurrence_start_frac:{args.recurrence_start_frac}"
    )
    log0(
        f"tie_embeddings:{args.tie_embeddings} embed_lr:{token_lr} "
        f"head_lr:{args.head_lr if base_model.lm_head is not None else 0.0} "
        f"matrix_lr:{args.matrix_lr} scalar_lr:{args.scalar_lr}"
    )
    log0(
        f"train_batch_tokens:{args.train_batch_tokens} train_seq_len:{args.train_seq_len} "
        f"iterations:{args.iterations} warmup_steps:{args.warmup_steps} "
        f"warmdown_iters:{args.warmdown_iters} warmdown_frac:{args.warmdown_frac} "
        f"min_lr:{args.min_lr} "
        f"max_wallclock_seconds:{args.max_wallclock_seconds:.3f}"
    )
    log0(f"seed:{args.seed}")
    log0(
        f"quant_mode:{args.quant_mode} compressor:{args.compressor} "
        f"matrix_bits:{args.matrix_bits} embed_bits:{args.embed_bits} "
        f"gptq_scale_mode:{args.gptq_scale_mode} gptq_damp:{args.gptq_damp} "
        f"gptq_act_order:{args.gptq_act_order} gptq_error_scale:{args.gptq_error_scale} "
        f"gptq_scale_floor:{args.gptq_scale_floor} "
        f"gptq_calibration_batches:{args.gptq_calibration_batches} "
        f"lqer_enabled:{args.lqer_enabled}"
    )
    if args.quant_mode == "rtn":
        log0(
            f"rtn_quant_bits:matrix={MATRIX_QUANT_BITS} embed={EMBED_QUANT_BITS} "
            f"clip_search={QUANT_CLIP_SEARCH} overrides={QUANT_BITS_OVERRIDES}"
        )

    # -----------------------------
    # DATA LOADER & MODEL WARMUP
    # -----------------------------

    train_loader = DistributedTokenLoader(
        args.train_files,
        rank,
        world_size,
        device,
        mode=args.loader_mode,
        coprime_file_stride=args.coprime_file_stride,
        coprime_token_stride=args.coprime_token_stride,
    )

    def zero_grad_all() -> None:
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None

    def lr_mul(step: int, elapsed_ms: float) -> float:
        if args.warmdown_frac > 0:
            if max_wallclock_ms is None:
                progress = step / max(args.iterations, 1)
            else:
                progress = min(elapsed_ms / max(max_wallclock_ms, 1e-9), 1.0)
            start = max(1.0 - args.warmdown_frac, 0.0)
            if progress <= start:
                return 1.0
            frac = (1.0 - progress) / max(args.warmdown_frac, 1e-9)
            return max(args.min_lr, min(frac, 1.0))
        if args.warmdown_iters <= 0:
            return 1.0
        if max_wallclock_ms is None:
            warmdown_start = max(args.iterations - args.warmdown_iters, 0)
            if warmdown_start <= step < args.iterations:
                return max(args.min_lr, max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0))
            return 1.0
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = args.warmdown_iters * step_ms
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        if remaining_ms <= warmdown_ms:
            return max(args.min_lr, remaining_ms / max(warmdown_ms, 1e-9))
        return 1.0

    eval_gptq_artifact_only = bool(int(os.environ.get("EVAL_GPTQ_ARTIFACT_ONLY", "0")))
    # Warmup primes the compiled forward/backward/optimizer paths, then we restore the
    # initial weights/optimizer state so measured training starts from the true init.
    if args.warmup_steps > 0 and not eval_gptq_artifact_only:
        initial_model_state = {name: tensor.detach().cpu().clone() for name, tensor in base_model.state_dict().items()}
        initial_optimizer_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        for warmup_step in range(args.warmup_steps):
            zero_grad_all()
            for micro_step in range(grad_accum_steps):
                if distributed:
                    model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    warmup_loss = model(x, y)
                (warmup_loss * grad_scale).backward()
            for opt in optimizers:
                opt.step()
            zero_grad_all()
            if args.warmup_steps <= 20 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == args.warmup_steps:
                log0(f"warmup_step:{warmup_step + 1}/{args.warmup_steps}")
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states, strict=True):
            opt.load_state_dict(state)
        zero_grad_all()
        if distributed:
            model.require_backward_grad_sync = True
        train_loader = DistributedTokenLoader(
            args.train_files,
            rank,
            world_size,
            device,
            mode=args.loader_mode,
            coprime_file_stride=args.coprime_file_stride,
            coprime_token_stride=args.coprime_token_stride,
        )

    export_only = bool(int(os.environ.get("EXPORT_ONLY", "0")))

    # -----------------------------
    # MAIN TRAINING LOOP
    # -----------------------------

    training_time_ms = 0.0
    stop_after_step: int | None = None
    step = 0
    recurrence_activated = bool(getattr(base_model, "recurrence_active", False))

    if export_only:
        if not os.path.isfile("final_model.pt"):
            raise FileNotFoundError("EXPORT_ONLY=1 requires final_model.pt in cwd")
        base_model.load_state_dict(torch.load("final_model.pt", map_location="cpu"), strict=True)
        log0("export_only:loaded final_model.pt, skipping training")
    else:
        torch.cuda.synchronize()
        t0 = time.perf_counter()

    while not export_only and not eval_gptq_artifact_only:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)

        should_validate = last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)
        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss, val_bpb = eval_val(
                args,
                model,
                rank,
                world_size,
                device,
                grad_accum_steps,
                val_tokens,
                base_bytes_lut,
                has_leading_space_lut,
                is_boundary_token_lut,
            )
            log0(
                f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms"
            )
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(
                    f"stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms "
                    f"step:{step}/{args.iterations}"
                )
            break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)
        if args.recurrence_extra_passes > 0 and not recurrence_activated:
            if max_wallclock_ms is None:
                train_frac = step / max(args.iterations, 1)
            else:
                train_frac = min(elapsed_ms / max(max_wallclock_ms, 1e-9), 1.0)
            if train_frac >= args.recurrence_start_frac:
                base_model.recurrence_active = True
                recurrence_activated = True
                log0(f"recurrence:activated step:{step} train_frac:{train_frac:.4f}")
        zero_grad_all()
        train_loss = torch.zeros((), device=device)
        for micro_step in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x, y)
            train_loss += loss.detach()
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps

        frac = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
        muon_momentum = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for group in optimizer_muon.param_groups:
            group["momentum"] = muon_momentum

        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * scale

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        for opt in optimizers:
            opt.step()
        zero_grad_all()

        step += 1
        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        should_log_train = (
            args.train_log_every > 0
            and (step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None)
        )
        if should_log_train:
            log0(
                f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                f"train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / step:.2f}ms"
            )

        # Needed to sync whether we've reached the wallclock cap.
        reached_cap = max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms
        if distributed and max_wallclock_ms is not None:
            reached_cap_tensor = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(reached_cap_tensor, op=dist.ReduceOp.MAX)
            reached_cap = bool(reached_cap_tensor.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step

    if eval_gptq_artifact_only:
        log0("eval_gptq_artifact_only:loading existing GPTQ artifact")
    elif not export_only:
        log0(
            f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
            f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB"
        )
    else:
        log0("export_only:starting GPTQ export")

    code_bytes = len(code.encode("utf-8"))
    if master_process and not export_only and not eval_gptq_artifact_only:
        torch.save(base_model.state_dict(), "final_model.pt")
        model_bytes = os.path.getsize("final_model.pt")
        log0(f"Serialized model: {model_bytes} bytes")
        log0(f"Code size: {code_bytes} bytes")
        log0(f"Total submission size: {model_bytes + code_bytes} bytes")
    elif export_only:
        log0(f"export_only:code size: {code_bytes} bytes")
    elif eval_gptq_artifact_only:
        log0(f"eval_gptq_artifact_only:code size: {code_bytes} bytes")
    if args.quant_mode == "gptq":
        from gptq_export import GptqConfig, dequantize_mixed, export_gptq_artifact, load_gptq_state_dict

        gptq_cfg = GptqConfig(
            matrix_bits=args.matrix_bits,
            embed_bits=args.embed_bits,
            matrix_clip_sigmas=args.matrix_clip_sigmas,
            attn_clip_sigmas=args.attn_clip_sigmas,
            mlp_clip_sigmas=args.mlp_clip_sigmas,
            embed_clip_sigmas=args.embed_clip_sigmas,
            scale_mode=args.gptq_scale_mode,
            clip_quantile=args.gptq_clip_quantile,
            damp=args.gptq_damp,
            act_order=args.gptq_act_order,
            error_scale=args.gptq_error_scale,
            scale_floor=args.gptq_scale_floor,
            calibration_batches=args.gptq_calibration_batches,
            compressor=args.compressor,
            lqer_enabled=args.lqer_enabled,
            lqer_rank=args.lqer_rank,
            lqer_top_k=args.lqer_top_k,
            lqer_factor_bits=args.lqer_factor_bits,
            lqer_asym_enabled=args.lqer_asym_enabled,
            lqer_asym_group=args.lqer_asym_group,
        )
        quant_artifact = "final_model.gptq.ptz"
        sd_cpu = {k: v.detach().cpu() for k, v in base_model.state_dict().items()}
        if eval_gptq_artifact_only:
            quant_artifact = os.environ.get("EVAL_GPTQ_ARTIFACT", quant_artifact)
            if not os.path.isfile(quant_artifact): raise FileNotFoundError(f"EVAL_GPTQ_ARTIFACT_ONLY=1 requires {quant_artifact!r}")
            quant_file_bytes = os.path.getsize(quant_artifact)
            log0(f"Loaded existing model gptq+{args.compressor}: {quant_file_bytes} bytes")
            log0(f"Total submission size gptq+{args.compressor}: {quant_file_bytes + code_bytes} bytes")
        else:
            t_gptq = time.perf_counter()
            quant_blob, _quant_meta = export_gptq_artifact(
                sd_cpu, model, train_loader, args, device, grad_accum_steps, gptq_cfg, log0
            )
            log0(f"GPTQ:export finished in {time.perf_counter() - t_gptq:.1f}s")
            if master_process:
                with open(quant_artifact, "wb") as f:
                    f.write(quant_blob)
                quant_file_bytes = len(quant_blob)
                log0(f"Serialized model gptq+{args.compressor}: {quant_file_bytes} bytes")
                log0(f"Total submission size gptq+{args.compressor}: {quant_file_bytes + code_bytes} bytes")
            if distributed:
                dist.barrier()
        quant_state = load_gptq_state_dict(open(quant_artifact, "rb").read(), args.compressor)
        deq_state = dequantize_mixed(quant_state["w"], quant_state["m"], sd_cpu)
        base_model.load_state_dict(deq_state, strict=True)
        roundtrip_state = deq_state
        roundtrip_tag = f"gptq+{args.compressor}"
    else:
        quant_obj, quant_stats = quantize_state_dict_int8(base_model.state_dict())
        quant_buf = io.BytesIO()
        torch.save(quant_obj, quant_buf)
        quant_raw = quant_buf.getvalue()
        quant_blob = compress_quant_bytes(quant_raw, args.compressor)
        quant_raw_bytes = len(quant_raw)
        quant_artifact = f"final_model.int8.{args.compressor}.ptz" if args.compressor != "zlib" else "final_model.int8.ptz"
        if master_process:
            with open(quant_artifact, "wb") as f:
                f.write(quant_blob)
            quant_file_bytes = os.path.getsize(quant_artifact)
            ratio = quant_stats["baseline_tensor_bytes"] / max(quant_stats["int8_payload_bytes"], 1)
            log0(
                f"Serialized model int8+{args.compressor}: {quant_file_bytes} bytes "
                f"(payload:{quant_stats['int8_payload_bytes']} raw_torch:{quant_raw_bytes} payload_ratio:{ratio:.2f}x)"
            )
            log0(f"Total submission size int8+{args.compressor}: {quant_file_bytes + code_bytes} bytes")
        if distributed:
            dist.barrier()
        quant_state = torch.load(io.BytesIO(decompress_quant_bytes(open(quant_artifact, "rb").read(), args.compressor)), map_location="cpu")
        roundtrip_state = dequantize_state_dict_int8(quant_state)
        base_model.load_state_dict(roundtrip_state, strict=True)
        roundtrip_tag = f"int8+{args.compressor}"

    if bool(int(os.environ.get("FRESH_MODEL_AFTER_QUANT", "0"))):
        log0("fresh_model_after_quant:enabled")
        base_model = GPT(
            vocab_size=args.vocab_size,
            num_layers=args.num_layers,
            model_dim=args.model_dim,
            num_heads=args.num_heads,
            num_kv_heads=args.num_kv_heads,
            mlp_mult=args.mlp_mult,
            tie_embeddings=args.tie_embeddings,
            tied_embed_init_std=args.tied_embed_init_std,
            logit_softcap=args.logit_softcap,
            rope_base=args.rope_base,
            qk_gain_init=args.qk_gain_init,
            recurrence_extra_passes=args.recurrence_extra_passes,
            recurrence_start_layer=args.recurrence_start_layer,
            recurrence_end_layer=args.recurrence_end_layer,
            recurrence_active=recurrence_activated,
        ).to(device).bfloat16()
        for module in base_model.modules():
            if isinstance(module, CastedLinear):
                module.float()
        restore_low_dim_params_to_fp32(base_model)
        base_model.load_state_dict(roundtrip_state, strict=True)
        compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)
        model = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False) if distributed else compiled_model

    if bool(int(os.environ.get("RECOMPILE_AFTER_QUANT", "0"))):
        log0("recompile_after_quant:enabled")
        compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)
        model = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False) if distributed else compiled_model

    torch.cuda.synchronize()
    t_qeval = time.perf_counter()
    q_val_loss, q_val_bpb = eval_val(
        args,
        model,
        rank,
        world_size,
        device,
        grad_accum_steps,
        val_tokens,
        base_bytes_lut,
        has_leading_space_lut,
        is_boundary_token_lut,
    )
    torch.cuda.synchronize()
    log0(
        f"final_{roundtrip_tag}_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} "
        f"eval_time:{1000.0 * (time.perf_counter() - t_qeval):.0f}ms"
    )
    log0(f"final_{roundtrip_tag}_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
