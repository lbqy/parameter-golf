#!/usr/bin/env python3
"""Check environment requirements for the records 2026-04-27 stack."""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys


def module_origin(name: str) -> str | None:
    spec = importlib.util.find_spec(name)
    return spec.origin if spec else None


def run(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True).strip()
    except Exception as exc:  # pragma: no cover - diagnostic script
        return f"ERROR: {exc}"


def main() -> int:
    import torch

    required: list[tuple[str, bool, str]] = []
    warnings: list[tuple[str, bool, str]] = []

    torch_version = torch.__version__
    torch_cuda = torch.version.cuda or "none"
    required.append(("torch>=2.9 for official FA3 wheels", torch_version >= "2.9", torch_version))
    warnings.append(("torch CUDA build is cu128/cu130", torch_cuda.startswith(("12.8", "13.0")), torch_cuda))

    cuda_ok = torch.cuda.is_available()
    required.append(("CUDA available", cuda_ok, str(cuda_ok)))
    if cuda_ok:
        cap = torch.cuda.get_device_capability(0)
        required.append(("Hopper SM90 GPU", cap >= (9, 0), f"capability={cap}, name={torch.cuda.get_device_name(0)}"))

    fa3_origin = module_origin("flash_attn_interface")
    required.append(("flash_attn_interface available", fa3_origin is not None, str(fa3_origin)))

    fa2_origin = module_origin("flash_attn")
    warnings.append(("flash_attn package available (not required by 04-27 script)", fa2_origin is not None, str(fa2_origin)))

    td_origin = module_origin("triton.tools.tensor_descriptor")
    required.append(("legacy triton.tools.tensor_descriptor available", td_origin is not None, str(td_origin)))

    lrzip_path = shutil.which("lrzip")
    required.append(("lrzip binary available", lrzip_path is not None, str(lrzip_path)))
    if lrzip_path:
        required.append(("lrzip runs", "lrzip version" in run(["lrzip", "-V"]).lower(), run(["lrzip", "-V"]).splitlines()[0]))

    print("records0427 environment check")
    print(f"python: {sys.version.split()[0]}")
    print(f"torch: {torch_version} cuda:{torch_cuda}")
    try:
        import triton

        print(f"triton: {triton.__version__}")
    except Exception as exc:
        print(f"triton: ERROR {exc}")
    print()

    failed = 0
    for label, ok, detail in required:
        status = "OK" if ok else "MISSING"
        print(f"{status:7} {label}: {detail}")
        failed += 0 if ok else 1
    for label, ok, detail in warnings:
        status = "OK" if ok else "WARN"
        print(f"{status:7} {label}: {detail}")

    print()
    if fa3_origin is None:
        print("Fallback impact: attention uses torch SDPA and training uses FixedSequenceTrainLoader instead of DocumentPackingLoader.")
    if td_origin is None:
        print("Fallback impact: LeakyReLU^2 MLP uses eager PyTorch path instead of the TensorDescriptor Triton kernel.")
    if lrzip_path is None:
        print("Fallback impact: per-group compressor cannot reproduce the current best legal artifact.")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
