from __future__ import annotations

import math
import os
import time
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.nn.functional as F
from torch import Tensor

from gptq_export import dequantize_mixed, load_gptq_state_dict
from train_gpt import (
    GPT,
    Hyperparameters,
    build_sentencepiece_luts,
    load_validation_tokens,
    restore_low_dim_params_to_fp32,
)


def forward_hidden(model: GPT, input_ids: Tensor) -> Tensor:
    x = model.tok_emb(input_ids)
    x = F.rms_norm(x, (x.size(-1),))
    x0 = x
    skips: list[Tensor] = []
    for i in range(model.num_encoder_layers):
        x = model.blocks[i](x, x0)
        if model.recurrence_active and model.recurrence_start_layer <= i <= model.recurrence_end_layer:
            for _ in range(model.recurrence_extra_passes):
                x = model.blocks[i](x, x0)
        skips.append(x)
    for i in range(model.num_decoder_layers):
        layer_idx = model.num_encoder_layers + i
        if skips:
            x = x + model.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
        x = model.blocks[layer_idx](x, x0)
        if model.recurrence_active and model.recurrence_start_layer <= layer_idx <= model.recurrence_end_layer:
            for _ in range(model.recurrence_extra_passes):
                x = model.blocks[layer_idx](x, x0)
    return model.final_norm(x).reshape(-1, x.size(-1))


def loss_with_lora(model: GPT, hidden: Tensor, targets: Tensor, lora_a: Tensor, lora_b: Tensor, alpha: float) -> Tensor:
    logits = F.linear(hidden, model.tok_emb.weight)
    logits = logits + ((hidden @ lora_a.t()) @ lora_b.t()) * (alpha / max(lora_a.size(0), 1))
    logits = model.logit_softcap * torch.tanh(logits / model.logit_softcap)
    return F.cross_entropy(logits, targets.reshape(-1))


def main() -> None:
    args = Hyperparameters()
    device = torch.device("cuda")
    torch.manual_seed(int(os.environ.get("TTT_SEED", args.seed)))
    np.random.seed(int(os.environ.get("TTT_SEED", args.seed)))

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(sp, args.vocab_size, device)

    recurrence_active = bool(int(os.environ.get("RECURRENCE_ACTIVE", "0")))
    model = GPT(
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
        recurrence_active=recurrence_active,
    ).to(device).bfloat16()
    restore_low_dim_params_to_fp32(model)
    template_sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    artifact = os.environ.get("TTT_ARTIFACT", "final_model.gptq.ptz")
    quant_state = load_gptq_state_dict(Path(artifact).read_bytes(), args.compressor)
    model.load_state_dict(dequantize_mixed(quant_state["w"], quant_state["m"], template_sd), strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    rank = int(os.environ.get("TTT_LORA_RANK", "8"))
    lr = float(os.environ.get("TTT_LR", "0.05"))
    wd = float(os.environ.get("TTT_WEIGHT_DECAY", "0.0"))
    alpha = float(os.environ.get("TTT_LORA_ALPHA", str(rank)))
    batch_tokens = int(os.environ.get("TTT_BATCH_TOKENS", "65536"))
    max_batches = int(os.environ.get("TTT_MAX_BATCHES", "0"))
    local_batch_seqs = max(batch_tokens // args.train_seq_len, 1)
    total_seqs = (val_tokens.numel() - 1) // args.train_seq_len

    lora_a = torch.randn(rank, args.model_dim, device=device, dtype=torch.float32) * 0.01
    lora_b = torch.zeros(args.vocab_size, rank, device=device, dtype=torch.float32)
    lora_a.requires_grad_(True)
    lora_b.requires_grad_(True)
    opt = torch.optim.AdamW([lora_a, lora_b], lr=lr, weight_decay=wd)

    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)
    t0 = time.perf_counter()
    batches = 0

    for batch_seq_start in range(0, total_seqs, local_batch_seqs):
        if max_batches > 0 and batches >= max_batches:
            break
        batch_seq_end = min(batch_seq_start + local_batch_seqs, total_seqs)
        raw_start = batch_seq_start * args.train_seq_len
        raw_end = batch_seq_end * args.train_seq_len + 1
        local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
        x = local[:-1].reshape(-1, args.train_seq_len)
        y = local[1:].reshape(-1, args.train_seq_len)
        targets = y.reshape(-1)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            hidden = forward_hidden(model, x)
            scored_loss = loss_with_lora(model, hidden, targets, lora_a, lora_b, alpha)
        hidden = hidden.detach().clone()
        batch_token_count = float(targets.numel())
        loss_sum += scored_loss.to(torch.float64) * batch_token_count
        token_count += batch_token_count
        prev_ids = x.reshape(-1)
        token_bytes = base_bytes_lut[targets].to(dtype=torch.int16)
        token_bytes += (has_leading_space_lut[targets] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
        byte_count += token_bytes.to(torch.float64).sum()

        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            adapt_loss = loss_with_lora(model, hidden, targets, lora_a, lora_b, alpha)
        adapt_loss.backward()
        torch.nn.utils.clip_grad_norm_([lora_a, lora_b], float(os.environ.get("TTT_GRAD_CLIP", "1.0")))
        opt.step()
        batches += 1
        if batches <= 3 or batches % 20 == 0:
            val_loss = loss_sum.item() / token_count.item()
            val_bpb = (val_loss / math.log(2.0)) * (token_count.item() / byte_count.item())
            print(f"ttt_batch:{batches} val_loss:{val_loss:.6f} val_bpb:{val_bpb:.6f}", flush=True)

    val_loss = loss_sum.item() / token_count.item()
    val_bpb = (val_loss / math.log(2.0)) * (token_count.item() / byte_count.item())
    print(
        f"final_ttt_lmhead_lora val_loss:{val_loss:.8f} val_bpb:{val_bpb:.8f} "
        f"batches:{batches} eval_time:{1000.0 * (time.perf_counter() - t0):.0f}ms",
        flush=True,
    )


if __name__ == "__main__":
    main()
