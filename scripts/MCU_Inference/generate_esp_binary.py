# generate_esp_binary.py
# Run this on PC to create the binary file to flash to ESP32 SPIFFS
import numpy as np
import pandas as pd
import struct

CSV_PATH   = "train_preprocessed_snv.csv"
OUT_PATH   = "nir_samples.bin"
N_PER_CLASS = 100    # 100 per class × 5 classes = 500 samples total
NUM_CHANNELS = 224

df = pd.read_csv(CSV_PATH)

if "class_id" in df.columns:
    y_col = "class_id"
    x_cols = [c for c in df.columns if c != "class_id"]
else:
    y_col = df.columns[0]
    x_cols = df.columns[1:]

# Balanced sample: N_PER_CLASS from each class
sampled = df.groupby(y_col).apply(
    lambda g: g.sample(min(N_PER_CLASS, len(g)), random_state=42)
).reset_index(drop=True).sample(frac=1, random_state=42)  # shuffle

X = sampled[x_cols].values.astype(np.float32)
y = sampled[y_col].values.astype(np.uint8)
n = len(X)

# Binary format:
#   [0..3]   : uint32 num_samples
#   [4..7]   : uint32 num_channels
#   [8..]    : for each sample:
#                1 byte  label (1-5)
#                224 × float32 LE  (896 bytes)
#              = 897 bytes per sample

with open(OUT_PATH, "wb") as f:
    f.write(struct.pack("<II", n, NUM_CHANNELS))
    for i in range(n):
        f.write(struct.pack("B", int(y[i])))
        f.write(X[i].tobytes())   # 224 × float32 LE

size_kb = (8 + n * 897) / 1024
print(f"Written {n} samples to {OUT_PATH}")
print(f"File size: {size_kb:.1f} KB  ({size_kb/1024:.2f} MB)")
print(f"Class distribution:\n{sampled[y_col].value_counts().sort_index()}")