"""GPTQ export helpers for root train_gpt.py (SDClip, brotli/zlib)."""

from __future__ import annotations

import io
import lzma
import zlib
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

KEEP_FLOAT_MAX_NUMEL = 65_536
CONTROL_TENSOR_NAME_PATTERNS = tuple(
    pattern
    for pattern in "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights".split(",")
    if pattern
)


def _normalize_param_name(name: str) -> str:
    while name.startswith("module."):
        name = name[len("module.") :]
    while name.startswith("_orig_mod."):
        name = name[len("_orig_mod.") :]
    return name


def _calib_model(model: nn.Module) -> nn.Module:
    m = model.module if hasattr(model, "module") else model
    return getattr(m, "_orig_mod", m)


def _keep_float_tensor(name: str, t: Tensor) -> Tensor:
    if any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS):
        return t.float().contiguous()
    return t.to(torch.float16).contiguous()


@dataclass
class GptqConfig:
    matrix_bits: int = 6
    embed_bits: int = 8
    matrix_clip_sigmas: float = 12.85
    attn_clip_sigmas: float = 13.0
    mlp_clip_sigmas: float = 11.5
    embed_clip_sigmas: float = 14.0
    scale_mode: str = "sigma"
    clip_quantile: float = 0.9999984
    damp: float = 0.01
    act_order: bool = True
    error_scale: float = 1.0
    scale_floor: float = 1.0
    calibration_batches: int = 32
    compressor: str = "brotli"
    lqer_enabled: bool = False
    lqer_rank: int = 4
    lqer_top_k: int = 3
    lqer_factor_bits: int = 4
    lqer_asym_enabled: bool = True
    lqer_asym_group: int = 64


def _clip_sigmas_for_name(name: str, cfg: GptqConfig) -> tuple[float, int]:
    if "tok_emb" in name:
        return cfg.embed_clip_sigmas, cfg.embed_bits
    if ".attn." in name:
        return cfg.attn_clip_sigmas, cfg.matrix_bits
    if ".mlp." in name:
        return cfg.mlp_clip_sigmas, cfg.matrix_bits
    return cfg.matrix_clip_sigmas, cfg.matrix_bits


def _pack_unsigned_values(values, bits: int) -> Tensor:
    import numpy as np

    flat = values.reshape(-1).astype(np.uint8, copy=False)
    if bits == 8:
        return torch.from_numpy(flat.copy()).contiguous()
    bit_shifts = np.arange(bits, dtype=np.uint8)
    bit_matrix = ((flat[:, None] >> bit_shifts[None, :]) & 1).astype(np.uint8, copy=False)
    packed = np.packbits(bit_matrix.reshape(-1), bitorder="little")
    return torch.from_numpy(packed.copy()).contiguous()


def _unpack_unsigned_values(packed: Tensor, bits: int, numel: int):
    import numpy as np

    packed_np = packed.detach().cpu().numpy().astype(np.uint8, copy=False)
    if bits == 8:
        return packed_np[:numel].copy()
    raw_bits = np.unpackbits(packed_np, bitorder="little")[: numel * bits].reshape(numel, bits)
    shifts = (1 << np.arange(bits, dtype=np.uint16))
    return (raw_bits.astype(np.uint16, copy=False) * shifts[None, :]).sum(axis=1).astype(np.uint8)


def _pack_signed_quant(q: Tensor, bits: int) -> Tensor:
    qmin = -(1 << (bits - 1))
    unsigned = (q.detach().cpu().to(torch.int16).numpy() - qmin).astype("uint8", copy=False)
    return _pack_unsigned_values(unsigned, bits)


def _unpack_signed_quant(packed: Tensor, bits: int, shape: tuple[int, ...]) -> Tensor:
    import math
    import numpy as np

    qmin = -(1 << (bits - 1))
    unsigned = _unpack_unsigned_values(packed, bits, math.prod(shape)).astype(np.int16, copy=False)
    signed = (unsigned + qmin).astype(np.int8, copy=False).reshape(shape)
    return torch.from_numpy(signed.copy()).contiguous()


def collect_hessians(
    model: nn.Module,
    train_loader,
    args,
    device: torch.device,
    grad_accum_steps: int,
    n_calibration_batches: int,
    log,
) -> dict[str, Tensor]:
    hessians: dict[str, Tensor] = {}
    hooks: list[torch.utils.hooks.RemovableHandle] = []
    calib = _calib_model(model)

    def make_hook(param_name: str):
        def hook_fn(_module, inp, _out):
            x = inp[0].detach().float()
            if x.ndim == 3:
                x = x.reshape(-1, x.shape[-1])
            if param_name not in hessians:
                hessians[param_name] = torch.zeros(x.shape[1], x.shape[1], dtype=torch.float32, device=device)
            hessians[param_name].addmm_(x.T, x)

        return hook_fn

    for name, module in calib.named_modules():
        if isinstance(module, nn.Linear) and module.weight.numel() > KEEP_FLOAT_MAX_NUMEL:
            param_name = _normalize_param_name(f"{name}.weight")
            hooks.append(module.register_forward_hook(make_hook(param_name)))

    if getattr(calib, "tie_embeddings", False):

        def embed_hook(_module, _inp, out):
            x = out.detach().float().reshape(-1, out.shape[-1])
            key = "tok_emb.weight"
            if key not in hessians:
                hessians[key] = torch.zeros(x.shape[1], x.shape[1], dtype=torch.float32, device=device)
            hessians[key].addmm_(x.T, x)

        hooks.append(calib.final_norm.register_forward_hook(embed_hook))

    was_training = calib.training
    calib.eval()
    with torch.no_grad():
        for _ in range(n_calibration_batches):
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            calib(x, y)
    for hook in hooks:
        hook.remove()
    if was_training:
        calib.train()

    for name in hessians:
        hessians[name] = hessians[name].cpu() / n_calibration_batches
    log(f"GPTQ:collected {len(hessians)} Hessians")
    return hessians


def gptq_quantize_weight(
    w: Tensor,
    H: Tensor,
    clip_sigmas: float = 12.85,
    clip_range: int = 63,
    scale_mode: str = "sigma",
    clip_quantile: float = 0.9999984,
    damp_ratio: float = 0.01,
    act_order: bool = True,
    error_scale: float = 1.0,
    scale_floor: float = 1.0,
    block_size: int = 128,
) -> tuple[Tensor, Tensor]:
    W_orig = w.float().clone()
    rows, cols = W_orig.shape
    H = H.float().clone()

    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    damp = damp_ratio * H.diag().mean()
    H.diagonal().add_(damp)

    perm = torch.argsort(H.diag(), descending=True) if act_order else torch.arange(cols)
    invperm = torch.argsort(perm)
    W_perm = W_orig[:, perm].clone()
    W_perm[:, dead[perm]] = 0
    H = H[perm][:, perm]

    Hinv = torch.cholesky_inverse(torch.linalg.cholesky(H))
    Hinv = torch.linalg.cholesky(Hinv, upper=True)

    if scale_mode == "max":
        clip_abs = W_orig.abs().amax(dim=1)
    elif scale_mode == "quantile":
        clip_abs = torch.quantile(W_orig.abs(), clip_quantile, dim=1)
    elif scale_mode == "sigma":
        clip_abs = clip_sigmas * W_orig.std(dim=1)
    else:
        raise ValueError(f"Unknown GPTQ scale_mode: {scale_mode!r}")
    s = (clip_abs / clip_range).clamp_min(scale_floor / max(clip_range, 1)).to(torch.float16)
    sf = s.float()

    Q = torch.zeros(rows, cols, dtype=torch.int8)
    W_work = W_perm.clone()
    for i1 in range(0, cols, block_size):
        i2 = min(i1 + block_size, cols)
        W_block = W_work[:, i1:i2].clone()
        Hinv_block = Hinv[i1:i2, i1:i2]
        err_acc = torch.zeros(rows, i2 - i1)
        for j in range(i2 - i1):
            w_col = W_block[:, j]
            d = Hinv_block[j, j]
            q_col = torch.clamp(torch.round(w_col / sf), -clip_range, clip_range)
            Q[:, i1 + j] = q_col.to(torch.int8)
            err = (w_col - q_col.float() * sf) / d
            err_acc[:, j] = err
            if error_scale != 0.0 and j + 1 < i2 - i1:
                W_block[:, j + 1 :] -= error_scale * err.unsqueeze(1) * Hinv_block[j, j + 1 :].unsqueeze(0)
        if i2 < cols:
            if error_scale != 0.0:
                W_work[:, i2:] -= error_scale * err_acc @ Hinv[i1:i2, i2:]

    return Q[:, invperm], s


def _lqer_pack(A: Tensor, B: Tensor, bits: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    rng = (1 << (bits - 1)) - 1
    sA = (A.abs().amax(dim=1).clamp_min(1e-10) / rng).to(torch.float16)
    sB = (B.abs().amax(dim=1).clamp_min(1e-10) / rng).to(torch.float16)
    qA = torch.clamp(torch.round(A / sA.float().view(-1, 1)), -rng, rng).to(torch.int8)
    qB = torch.clamp(torch.round(B / sB.float().view(-1, 1)), -rng, rng).to(torch.int8)
    return qA, sA, qB, sB


def _lqer_pack_asym(A: Tensor, B: Tensor, group: int = 64) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    sA = (A.abs().amax().clamp_min(1e-10) / 1.5).to(torch.float16)
    qA = torch.clamp(torch.round(A / sA.float()), -2, 1).to(torch.int8)
    bf = B.reshape(-1, group)
    bmax = bf.abs().amax(dim=-1, keepdim=True).clamp_min(1e-10)
    sB = (bmax / 7.5).to(torch.float16).reshape(-1)
    qB = torch.clamp(torch.round(bf / sB.float().reshape(-1, 1)), -8, 7).to(torch.int8).reshape(B.shape)
    return qA, sA, qB, sB


def gptq_mixed_quantize(
    state_dict: dict[str, Tensor],
    hessians: dict[str, Tensor],
    cfg: GptqConfig,
    log,
) -> tuple[dict[str, Tensor], dict[str, object]]:
    result: dict[str, Tensor] = {}
    meta: dict[str, object] = {}
    lqer_cands: dict[str, tuple[Tensor, float]] = {}

    for name, tensor in state_dict.items():
        t = tensor.detach().cpu().contiguous()
        if not t.is_floating_point() or t.numel() <= KEEP_FLOAT_MAX_NUMEL:
            result[name] = _keep_float_tensor(name, t) if t.is_floating_point() else t
            meta[name] = "passthrough (float16)"
            continue
        if name not in hessians:
            raise KeyError(f"Missing Hessian for {name}")
        clip_sigmas, bits = _clip_sigmas_for_name(name, cfg)
        clip_range = (1 << (bits - 1)) - 1
        q, s = gptq_quantize_weight(
            t,
            hessians[name],
            clip_sigmas=clip_sigmas,
            clip_range=clip_range,
            scale_mode=cfg.scale_mode,
            clip_quantile=cfg.clip_quantile,
            damp_ratio=cfg.damp,
            act_order=cfg.act_order,
            error_scale=cfg.error_scale,
            scale_floor=cfg.scale_floor,
        )
        result[f"{name}.q"] = _pack_signed_quant(q, bits)
        result[f"{name}.scale"] = s
        meta[name] = {"kind": "gptq", "bits": bits, "shape": tuple(q.shape), "packed": True}
        if cfg.lqer_enabled:
            w_q = q.float() * s.float().view(-1, 1)
            err = t.float() - w_q
            lqer_cands[name] = (err, float(err.norm()))

    if cfg.lqer_enabled and lqer_cands:
        top = sorted(lqer_cands.items(), key=lambda kv: -kv[1][1])[: cfg.lqer_top_k]
        log(f"LQER:top_k={cfg.lqer_top_k} rank={cfg.lqer_rank} asym={cfg.lqer_asym_enabled}")
        for name, (err, err_norm) in top:
            u, svals, vh = torch.linalg.svd(err, full_matrices=False)
            rank = min(cfg.lqer_rank, svals.numel())
            a = (u[:, :rank] * svals[:rank]).contiguous()
            b = vh[:rank, :].contiguous()
            if cfg.lqer_asym_enabled and b.numel() % cfg.lqer_asym_group == 0:
                qA, sA, qB, sB = _lqer_pack_asym(a, b, cfg.lqer_asym_group)
                result[f"{name}.lqA_a"] = qA
                result[f"{name}.lqAs_a"] = sA
                result[f"{name}.lqB_a"] = qB
                result[f"{name}.lqBs_a"] = sB
                meta[name]["lqer"] = "asym"
            else:
                qA, sA, qB, sB = _lqer_pack(a, b, cfg.lqer_factor_bits)
                result[f"{name}.lqA"] = qA
                result[f"{name}.lqAs"] = sA
                result[f"{name}.lqB"] = qB
                result[f"{name}.lqBs"] = sB
                meta[name]["lqer"] = "sym"
            log(f"LQER:{name} err_norm={err_norm:.4f} rank={rank}")

    counts: dict[str, int] = {}
    for info in meta.values():
        if isinstance(info, dict) and info.get("kind") == "gptq":
            label = f"gptq (int{info['bits']})"
            if "lqer" in info:
                label += f"+lqer_{info['lqer']}"
            counts[label] = counts.get(label, 0) + 1
    for kind, count in sorted(counts.items()):
        log(f"GPTQ:{kind} tensors:{count}")
    return result, meta


def dequantize_mixed(result: dict[str, Tensor], meta: dict[str, object], template_sd: dict[str, Tensor]) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    for name, orig in template_sd.items():
        info = meta.get(name)
        if info is None:
            continue
        orig_dtype = orig.dtype
        if isinstance(info, str) and "passthrough" in info:
            t = result[name]
            if t.dtype == torch.float16 and orig_dtype in (torch.float32, torch.bfloat16):
                t = t.to(orig_dtype)
            out[name] = t
            continue
        if isinstance(info, dict):
            q = _unpack_signed_quant(result[f"{name}.q"], int(info["bits"]), tuple(info["shape"]))
            lqer_kind = info.get("lqer")
        else:
            q = result[f"{name}.q"]
            lqer_kind = "asym" if isinstance(info, str) and "lqer_asym" in info else "sym" if isinstance(info, str) and "+lqer" in info else None
        s = result[f"{name}.scale"]
        if s.ndim > 0:
            w = q.float() * s.float().view(q.shape[0], *([1] * (q.ndim - 1)))
        else:
            w = q.float() * float(s.item())
        if lqer_kind == "asym":
            qA = result[f"{name}.lqA_a"].float() * float(result[f"{name}.lqAs_a"])
            qB_t = result[f"{name}.lqB_a"]
            sB_t = result[f"{name}.lqBs_a"]
            group = qB_t.numel() // sB_t.numel()
            qB = (qB_t.reshape(-1, group).float() * sB_t.float().view(-1, 1)).reshape(qB_t.shape)
            w = w + qA @ qB
        elif lqer_kind == "sym":
            qA = result[f"{name}.lqA"].float() * result[f"{name}.lqAs"].float().view(-1, 1)
            qB = result[f"{name}.lqB"].float() * result[f"{name}.lqBs"].float().view(-1, 1)
            w = w + qA @ qB
        out[name] = w.to(orig_dtype)
    return out


def compress_bytes(data: bytes, compressor: str) -> bytes:
    if compressor == "zlib":
        return zlib.compress(data, level=9)
    if compressor == "lzma":
        return lzma.compress(data, preset=6)
    if compressor == "brotli":
        import brotli

        return brotli.compress(data, quality=11)
    raise ValueError(f"Unknown compressor: {compressor!r}")


def decompress_bytes(data: bytes, compressor: str) -> bytes:
    if compressor == "zlib":
        return zlib.decompress(data)
    if compressor == "lzma":
        return lzma.decompress(data)
    if compressor == "brotli":
        import brotli

        return brotli.decompress(data)
    raise ValueError(f"Unknown compressor: {compressor!r}")


def export_gptq_artifact(
    state_dict: dict[str, Tensor],
    model: nn.Module,
    train_loader,
    args,
    device: torch.device,
    grad_accum_steps: int,
    cfg: GptqConfig,
    log,
) -> tuple[bytes, dict[str, object]]:
    hessians = collect_hessians(model, train_loader, args, device, grad_accum_steps, cfg.calibration_batches, log)
    quant_result, quant_meta = gptq_mixed_quantize(state_dict, hessians, cfg, log)
    buf = io.BytesIO()
    torch.save({"w": quant_result, "m": quant_meta, "compressor": cfg.compressor}, buf)
    return compress_bytes(buf.getvalue(), cfg.compressor), quant_meta


def load_gptq_state_dict(blob: bytes, compressor: str) -> dict[str, object]:
    raw = decompress_bytes(blob, compressor)
    return torch.load(io.BytesIO(raw), map_location="cpu")
