"""
generate_weights_c.py
=====================
Converts the Kaggle-exported binary/hex weight files into a single
nir_weights.c file that can be compiled into the F28379D CCS project.

Run this on your PC (not the MCU) after downloading the Kaggle output files.

Input files (all in the same folder as this script, or set paths below):
  nir_backbone_weights.bin   — Q8 int8 + Q15 int16 CNN weights (raw bytes)
  svm_weights.hex            — Q15 int16 SVM weight matrix (one hex value/line)
  svm_biases.hex             — Q15 int16 SVM biases
  scaler_mean.hex            — Q15 int16 StandardScaler means
  scaler_std.hex             — Q15 int16 StandardScaler stds

Output file:
  nir_weights.c              — add to your CCS project

Usage:
  python generate_weights_c.py
  python generate_weights_c.py --input-dir output --out nir_weights.c
"""

import argparse
import struct
import io
import numpy as np
from pathlib import Path

# ── Layer sequence exactly as written by the v6 training script ──────────────
# Each entry: (name, type, shape)
#   type 'i8'  → int8,  written raw as signed bytes in .bin
#   type 'i16' → int16, written raw as signed 16-bit LE in .bin
BACKBONE_LAYERS = [
    ("bb0_weight",        "i8",  (64, 1, 25)),       # Conv stem weights
    ("bb1_weight",        "i16", (64,)),              # BN1 gamma
    ("bb1_bias",          "i16", (64,)),              # BN1 beta
    ("bb1_running_mean",  "i16", (64,)),              # BN1 running mean
    ("bb1_running_var",   "i16", (64,)),              # BN1 running var
    ("bb3_weight",        "i8",  (128, 64, 5)),       # Conv block1 weights
    ("bb4_weight",        "i16", (128,)),             # BN2 gamma
    ("bb4_bias",          "i16", (128,)),             # BN2 beta
    ("bb4_running_mean",  "i16", (128,)),             # BN2 running mean
    ("bb4_running_var",   "i16", (128,)),             # BN2 running var
    ("bb7_weight",        "i8",  (128, 128, 3)),      # Conv block2 weights
    ("bb8_weight",        "i16", (128,)),             # BN3 gamma
    ("bb8_bias",          "i16", (128,)),             # BN3 beta
    ("bb8_running_mean",  "i16", (128,)),             # BN3 running mean
    ("bb8_running_var",   "i16", (128,)),             # BN3 running var
    # head weights follow in the .bin but we skip them for inference
    ("head1_weight",      "i16", (64, 128)),          # SKIP
    ("head1_bias",        "i16", (64,)),              # SKIP
    ("head3_weight",      "i16", (5, 64)),            # SKIP
    ("head3_bias",        "i16", (5,)),               # SKIP
]
SKIP_LAYERS = {"head1_weight", "head1_bias", "head3_weight", "head3_bias"}


def read_hex_file(path: Path) -> np.ndarray:
    """Read a .hex file (one 4-hex-digit signed int16 per line) → int16 array."""
    vals = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        raw = int(line, 16)
        # Interpret as signed 16-bit
        if raw > 32767:
            raw -= 65536
        vals.append(raw)
    return np.array(vals, dtype=np.int16)


def fmt_i16_row(vals):
    return "    " + ", ".join(f"{int(v):6d}" for v in vals) + ","


def format_c_array_i16_plain(name: str, data: np.ndarray, section: str = "") -> str:
    """Emit a single const int16_t array, optionally with a section pragma."""
    flat = data.flatten().astype(np.int16)
    n    = len(flat)
    ROW  = 12
    lines = []
    if section:
        lines.append(f'#pragma DATA_SECTION({name}, "{section}")')
    lines.append(f"const int16_t {name}[{n}] = {{")
    for i in range(0, n, ROW):
        lines.append(fmt_i16_row(flat[i:i+ROW]))
    lines.append("};\n")
    return "\n".join(lines)


def format_bb3_split(data: np.ndarray) -> str:
    """
    bb3_weight: (128, 64, 5) = 40960 elements.
    Split into:
      g_bb3_weight_lo[20480]  section .nir_bb3_lo  (first half)
      g_bb3_weight_hi[20480]  section .nir_bb3_hi  (second half)
    Upcast int8 -> int16 (values stay in -128..127).
    """
    flat = data.flatten().astype(np.int16)   # upcast int8 -> int16
    assert len(flat) == 40960, f"Expected 40960, got {len(flat)}"
    lo = flat[:20480]
    hi = flat[20480:]
    out  = format_c_array_i16_plain("g_bb3_weight_lo", lo, ".nir_bb3_lo")
    out += format_c_array_i16_plain("g_bb3_weight_hi", hi, ".nir_bb3_hi")
    return out


def format_bb7_split(data: np.ndarray) -> str:
    """
    bb7_weight: (128, 128, 3) = 49152 elements.
    Split into:
      g_bb7_weight_a[16384]  section .nir_bb7_a
      g_bb7_weight_b[16384]  section .nir_bb7_b
      g_bb7_weight_c[16384]  section .nir_bb7_c
    Upcast int8 -> int16.
    """
    flat = data.flatten().astype(np.int16)   # upcast int8 -> int16
    assert len(flat) == 49152, f"Expected 49152, got {len(flat)}"
    a = flat[:16384]
    b = flat[16384:32768]
    c = flat[32768:]
    out  = format_c_array_i16_plain("g_bb7_weight_a", a, ".nir_bb7_a")
    out += format_c_array_i16_plain("g_bb7_weight_b", b, ".nir_bb7_b")
    out += format_c_array_i16_plain("g_bb7_weight_c", c, ".nir_bb7_c")
    return out


def format_c_array_i16(name: str, data: np.ndarray) -> str:
    """Wrapper kept for BN / SVM / scaler int16 arrays (no section needed)."""
    return format_c_array_i16_plain(f"g_{name}", data)



def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", default=".", help="Directory with Kaggle output files")
    p.add_argument("--out",       default="nir_weights.c", help="Output .c file path")
    return p.parse_args()


def main():
    args   = parse_args()
    indir  = Path(args.input_dir)
    outpath = Path(args.out)

    # ── Read backbone binary ─────────────────────────────────────────────────
    bin_path = indir / "nir_backbone_weights.bin"
    print(f"Reading {bin_path}  ({bin_path.stat().st_size} bytes)")
    raw_bytes = bin_path.read_bytes()
    buf = io.BytesIO(raw_bytes)

    arrays = {}
    for name, dtype, shape in BACKBONE_LAYERS:
        n_elem = 1
        for d in shape:
            n_elem *= d
        if dtype == "i8":
            data = np.frombuffer(buf.read(n_elem), dtype=np.int8).reshape(shape)
        else:  # i16
            data = np.frombuffer(buf.read(n_elem * 2), dtype=np.int16).reshape(shape)
        arrays[name] = data
        print(f"  {name:30s}  shape={shape}  dtype={dtype}")

    remaining = len(buf.read())
    if remaining > 0:
        print(f"  WARNING: {remaining} bytes remaining in .bin — check layer list")

    # ── Read hex files ────────────────────────────────────────────────────────
    svm_w  = read_hex_file(indir / "svm_weights.hex")
    svm_b  = read_hex_file(indir / "svm_biases.hex")
    sc_mu  = read_hex_file(indir / "scaler_mean.hex")
    sc_std = read_hex_file(indir / "scaler_std.hex")

    NUM_CLASSES  = 5
    CNN_FEAT_DIM = 128

    print(f"  svm_weights:  {svm_w.shape}  (expect {NUM_CLASSES * CNN_FEAT_DIM})")
    print(f"  svm_biases:   {svm_b.shape}  (expect {NUM_CLASSES})")
    print(f"  scaler_mean:  {sc_mu.shape}  (expect {CNN_FEAT_DIM})")
    print(f"  scaler_std:   {sc_std.shape}  (expect {CNN_FEAT_DIM})")

    assert len(svm_w)  == NUM_CLASSES * CNN_FEAT_DIM, "svm_weights size mismatch"
    assert len(svm_b)  == NUM_CLASSES,                "svm_biases size mismatch"
    assert len(sc_mu)  == CNN_FEAT_DIM,               "scaler_mean size mismatch"
    assert len(sc_std) == CNN_FEAT_DIM,               "scaler_std size mismatch"

    svm_w = svm_w.reshape(NUM_CLASSES, CNN_FEAT_DIM)

    # ── Emit nir_weights.c ────────────────────────────────────────────────────
    out_lines = [
        "/*",
        " * nir_weights.c — AUTO-GENERATED by generate_weights_c.py",
        " * Do not edit by hand.",
        " *",
        " * Conv weights (bb0, bb3, bb7): stored as int16_t (widened from int8/Q8).",
        " * BN weights (bb1, bb4, bb8):  int16_t Q15.",
        " * SVM + scaler weights:         int16_t Q15.",
        " * bb3 split into _lo/_hi (flash sectors F/G).",
        " * bb7 split into _a/_b/_c (flash sectors H/I/J).",
        " * Dequantisation scales in nir_model.h",
        " */",
        "",
        '#include <stdint.h>',
        '#include "nir_weights.h"',
        "",
    ]

    # bb0: stem conv, (64,1,25) int8 -> int16_t plain array
    out_lines.append(format_c_array_i16_plain("g_bb0_weight",
                                               arrays["bb0_weight"].astype(np.int16)))

    # bb1 BN (int16, no section)
    for layer_name in ["bb1_weight", "bb1_bias", "bb1_running_mean", "bb1_running_var"]:
        out_lines.append(format_c_array_i16_plain(f"g_{layer_name}", arrays[layer_name]))

    # bb3: block1 conv, split into _lo / _hi
    out_lines.append(format_bb3_split(arrays["bb3_weight"]))

    # bb4 BN (int16, no section)
    for layer_name in ["bb4_weight", "bb4_bias", "bb4_running_mean", "bb4_running_var"]:
        out_lines.append(format_c_array_i16_plain(f"g_{layer_name}", arrays[layer_name]))

    # bb7: block2 conv, split into _a / _b / _c
    out_lines.append(format_bb7_split(arrays["bb7_weight"]))

    # bb8 BN (int16, no section)
    for layer_name in ["bb8_weight", "bb8_bias", "bb8_running_mean", "bb8_running_var"]:
        out_lines.append(format_c_array_i16_plain(f"g_{layer_name}", arrays[layer_name]))

    # SVM + scaler (all int16, no section)
    out_lines.append(format_c_array_i16_plain("g_svm_weights", svm_w))
    out_lines.append(format_c_array_i16_plain("g_svm_bias",    svm_b))
    out_lines.append(format_c_array_i16_plain("g_scaler_mean", sc_mu))
    out_lines.append(format_c_array_i16_plain("g_scaler_std",  sc_std))

    outpath.write_text("\n".join(out_lines))
    print(f"\nWrote {outpath}  ({outpath.stat().st_size / 1024:.1f} KB)")
    print("Array layout:")
    print("  g_bb0_weight     [1600]  int16_t")
    print("  g_bb1_weight/bias/mean/var [64 each]  int16_t")
    print("  g_bb3_weight_lo  [20480]  int16_t  section .nir_bb3_lo")
    print("  g_bb3_weight_hi  [20480]  int16_t  section .nir_bb3_hi")
    print("  g_bb4_weight/bias/mean/var [128 each]  int16_t")
    print("  g_bb7_weight_a   [16384]  int16_t  section .nir_bb7_a")
    print("  g_bb7_weight_b   [16384]  int16_t  section .nir_bb7_b")
    print("  g_bb7_weight_c   [16384]  int16_t  section .nir_bb7_c")
    print("  g_bb8_weight/bias/mean/var [128 each]  int16_t")
    print("  g_svm_weights/bias/scaler_mean/std")


if __name__ == "__main__":
    main()
