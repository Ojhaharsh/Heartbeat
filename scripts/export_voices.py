#!/usr/bin/env python3
"""
Export Kokoro voice .pt packs to a simple binary format for C++ runtime.

Input tensors are expected to be shaped [N, 1, D] or [N, D].
Output format (.hvp):
  - 4 bytes magic: b"HVPK"
  - uint32 rows (N)
  - uint32 dim  (D)
  - float32 payload, row-major [N, D]
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import torch


def export_voice_file(src: Path, dst: Path) -> tuple[int, int]:
    tensor = torch.load(src, map_location="cpu", weights_only=True)
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{src}: expected Tensor, got {type(tensor)}")

    if tensor.ndim == 3 and tensor.shape[1] == 1:
        tensor = tensor[:, 0, :]
    elif tensor.ndim != 2:
        raise ValueError(f"{src}: unsupported shape {tuple(tensor.shape)}")

    tensor = tensor.detach().float().contiguous()
    rows, dim = int(tensor.shape[0]), int(tensor.shape[1])

    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("wb") as f:
        f.write(b"HVPK")
        f.write(struct.pack("<II", rows, dim))
        f.write(tensor.numpy().tobytes(order="C"))

    return rows, dim


def main() -> int:
    parser = argparse.ArgumentParser(description="Export Kokoro voice packs to .hvp format")
    parser.add_argument("--input-dir", default="models/voices", help="Directory containing .pt voice packs")
    parser.add_argument("--output-dir", default="models/voices", help="Directory for .hvp outputs")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    files = sorted(input_dir.glob("*.pt"))
    if not files:
        print(f"[WARN] No .pt files found in {input_dir}")
        return 0

    print(f"[INFO] Exporting {len(files)} voice pack(s)...")
    for src in files:
        dst = output_dir / (src.stem + ".hvp")
        rows, dim = export_voice_file(src, dst)
        print(f"  {src.name} -> {dst.name} [{rows} x {dim}]")

    print("[SUCCESS] Voice export complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

