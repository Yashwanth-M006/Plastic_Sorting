"""
nir_host_eval.py
================
Host-side sensor emulator for NIR plastic classifier.
Streams preprocessed SNV samples from a CSV over UART to the TI F28379D,
collects predicted class IDs, and prints a final accuracy report.

Hardware:  F28379D  ←UART→  XDS110 onboard USB-UART  ←USB→  this PC
Protocol:
  Host → MCU :  0xAA  +  224 × float32 LE  =  897 bytes per sample
  MCU  → Host:  0xBB  +  1 byte class (1–5)  =  2 bytes per sample

Usage:
  python nir_host_eval.py --port COM17 --csv train_preprocessed_snv.csv
  python nir_host_eval.py --port /dev/ttyUSB0 --csv train_preprocessed_snv.csv
"""

import argparse
import struct
import sys
import time

import numpy as np
import pandas as pd
import serial
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ── Protocol constants (must match MCU firmware) ──────────────────────────────
START_BYTE   = 0xAA          # host sends this to begin a sample transfer
ACK_BYTE     = 0xBB          # MCU prefixes every response with this
NUM_CHANNELS = 224           # SNV float32 channels per sample
BAUD_RATE    = 115200        # match SCI_B init in firmware
TIMEOUT_S    = 30.0           # per-sample receive timeout (seconds)


def parse_args():
    p = argparse.ArgumentParser(description="NIR MCU inference evaluator")
    p.add_argument("--port", required=True,
                   help="Serial port, e.g. COM3 or /dev/ttyUSB0")
    p.add_argument("--csv",  required=True,
                   help="Preprocessed SNV CSV (class_id, ch.1 … ch.224)")
    p.add_argument("--baud", type=int, default=BAUD_RATE,
                   help=f"Baud rate (default {BAUD_RATE})")
    p.add_argument("--delay", type=float, default=0.02,
                   help="Inter-sample delay in seconds (default 0.02)")
    p.add_argument("--max-samples", type=int, default=None,
                   help="Limit number of samples to send (default: all)")
    return p.parse_args()


def load_csv(path: str, max_samples):
    """Load preprocessed SNV CSV. Returns (X float32, y int32)."""
    df = pd.read_csv(path)

    # Flexible column detection: class_id may be col 0 or named
    if "class_id" in df.columns:
        y = df["class_id"].values.astype(np.int32)
        # Channel columns: everything after class_id
        # Original train CSV has two extra cols before channels (col index 3+)
        # preprocessed SNV CSV should only have class_id + 224 channels
        ch_cols = [c for c in df.columns if c != "class_id"]
        X = df[ch_cols].values.astype(np.float32)
    else:
        # Fallback: assume first col = label, rest = channels
        y = df.iloc[:, 0].values.astype(np.int32)
        X = df.iloc[:, 1:].values.astype(np.float32)

    assert X.shape[1] == NUM_CHANNELS, (
        f"Expected {NUM_CHANNELS} channels, got {X.shape[1]}. "
        "Did you pass the raw CSV instead of the preprocessed SNV CSV?"
    )

    if max_samples is not None:
        X, y = X[:max_samples], y[:max_samples]

    print(f"Loaded {len(X)} samples  |  classes present: {sorted(np.unique(y))}")
    return X, y


def send_sample(ser: serial.Serial, x: np.ndarray) -> int | None:
    """
    Send one SNV sample to the MCU and return the predicted class ID (1–5),
    or None on timeout / framing error.

    Wire format:
      → [0xAA] [float32×224 little-endian]   (897 bytes)
      ← [0xBB] [uint8 class_id]              (2 bytes)
    """
    # Pack payload: start byte + 224 little-endian float32s
    payload = bytes([START_BYTE]) + struct.pack(f"<{NUM_CHANNELS}f", *x)
    ser.write(payload)

    # Wait for 2-byte response: ACK + class
    deadline = time.monotonic() + TIMEOUT_S
    buf = bytearray()
    while len(buf) < 2:
        if time.monotonic() > deadline:
            return None
        chunk = ser.read(2 - len(buf))
        buf.extend(chunk)

    if buf[0] != ACK_BYTE:
        # Framing error — drain and resync
        ser.reset_input_buffer()
        return None

    return int(buf[1])


def run_evaluation(port: str, baud: int, X: np.ndarray, y: np.ndarray,
                   inter_sample_delay: float):
    y_pred  = []
    errors  = []
    n       = len(X)

    print(f"\nOpening {port} @ {baud} baud …")
    with serial.Serial(port, baud, timeout=TIMEOUT_S) as ser:
        # Give MCU time to finish any startup prints
        time.sleep(0.5)
        ser.reset_input_buffer()

        print(f"Sending {n} samples …\n")
        t_start = time.perf_counter()

        for i, (x, label) in enumerate(zip(X, y)):
            pred = send_sample(ser, x)

            if pred is None:
                errors.append(i)
                # Use sentinel so accuracy calc skips this sample
                y_pred.append(-1)
                status = "TIMEOUT"
            else:
                y_pred.append(pred)
                status = f"pred={pred}  true={label}  {'OK' if pred == label else 'WRONG'}"

            # Simple inline progress (no tqdm dependency)
            pct = (i + 1) / n * 100
            print(f"  [{i+1:>5}/{n}]  {pct:5.1f}%  {status}", end="\r", flush=True)

            if inter_sample_delay > 0:
                time.sleep(inter_sample_delay)

        elapsed = time.perf_counter() - t_start

    print()   # newline after \r progress
    return np.array(y_pred, dtype=np.int32), errors, elapsed


def print_report(y_true, y_pred_raw, errors, elapsed):
    """Filter timeout samples and print accuracy + sklearn report."""
    # Remove timeout samples (pred == -1)
    valid_mask = y_pred_raw != -1
    y_true_v   = y_true[valid_mask]
    y_pred_v   = y_pred_raw[valid_mask]
    n_total    = len(y_true)
    n_valid    = valid_mask.sum()
    n_errors   = len(errors)

    print("=" * 60)
    print("  NIR MCU INFERENCE — EVALUATION REPORT")
    print("=" * 60)
    print(f"  Total samples     : {n_total}")
    print(f"  Received (valid)  : {n_valid}")
    print(f"  Timeouts/errors   : {n_errors}  "
          f"(indices: {errors[:10]}{'…' if len(errors) > 10 else ''})"
          if n_errors else f"  Timeouts/errors   : {n_errors}")
    print(f"  Wall time         : {elapsed:.1f}s  "
          f"({elapsed/n_total*1000:.1f} ms/sample)")
    print()

    if n_valid == 0:
        print("  No valid responses received. Check serial connection and firmware.")
        return

    acc = accuracy_score(y_true_v, y_pred_v)
    print(f"  Accuracy          : {acc*100:.4f}%  ({int(acc*n_valid)}/{n_valid})")
    print()
    print("  Confusion Matrix (rows=true, cols=pred):")
    labels = sorted(np.unique(np.concatenate([y_true_v, y_pred_v])))
    cm = confusion_matrix(y_true_v, y_pred_v, labels=labels)
    # Header row
    hdr = "       " + "  ".join(f"P{l}" for l in labels)
    print(f"  {hdr}")
    for row_label, row in zip(labels, cm):
        print(f"  T{row_label}  " + "   ".join(f"{v:4d}" for v in row))
    print()
    print("  Classification Report:")
    report = classification_report(y_true_v, y_pred_v,
                                   target_names=[f"Class {l}" for l in labels],
                                   digits=4)
    for line in report.splitlines():
        print(f"    {line}")
    print("=" * 60)


def main():
    args = parse_args()

    try:
        X, y = load_csv(args.csv, args.max_samples)
    except Exception as e:
        print(f"[ERROR] Could not load CSV: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        y_pred, errors, elapsed = run_evaluation(
            args.port, args.baud, X, y, args.delay
        )
    except serial.SerialException as e:
        print(f"[ERROR] Serial port error: {e}", file=sys.stderr)
        print("  • Check --port value (e.g. COM3 on Windows, /dev/ttyUSB0 on Linux)")
        print("  • Ensure XDS110 USB driver is installed and no other app holds the port")
        sys.exit(1)

    print_report(y, y_pred, errors, elapsed)


if __name__ == "__main__":
    main()
