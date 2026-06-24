"""
retrain_svm.py
==============
Re-trains the StandardScaler + LinearSVC using features extracted
from the CURRENT nir_cnn_backbone (1).pt model.

This fixes the mismatch between:
  - nir_cnn_backbone (1).pt  (current backbone)
  - nir_svm_pipeline (1).pkl (SVM from a DIFFERENT training run)

Run this once, then run generate_weights_c.py and generate_weights_bn_float.py
to regenerate all weight files.

Usage:
    python retrain_svm.py --csv train_preprocessed_snv.csv
    python retrain_svm.py --csv train_preprocessed_snv.csv --C 0.5

Output:
    nir_svm_pipeline_new.pkl   -- new matched pipeline
    svm_weights_new.hex        -- for generate_weights_c.py
    svm_biases_new.hex         -- for generate_weights_c.py
    scaler_mean_new.hex        -- for generate_weights_c.py
    scaler_std_new.hex         -- for generate_weights_c.py
"""
import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

V6_DIR    = r"E:\CSIR\medley\nir_v6"
MODEL_PT  = f"{V6_DIR}/nir_cnn_backbone (1).pt"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv",      required=True,   help="Preprocessed SNV CSV")
    p.add_argument("--C",        type=float, default=0.5, help="SVM regularisation C")
    p.add_argument("--max-iter", type=int, default=50000)
    p.add_argument("--out-pkl",  default="nir_svm_pipeline_new.pkl")
    p.add_argument("--hex-dir",  default=".",      help="Where to write hex files")
    return p.parse_args()


# ── Build backbone ─────────────────────────────────────────────────────────────
class NIRBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv1d(1, 64,  25, stride=8, padding=12, bias=False)
        self.bn1   = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, 5, stride=1, padding=2, bias=False)
        self.bn4   = nn.BatchNorm1d(128)
        self.pool  = nn.MaxPool1d(2, 2)
        self.conv7 = nn.Conv1d(128, 128, 3, stride=1, padding=1, bias=False)
        self.bn8   = nn.BatchNorm1d(128)
        self.gelu  = nn.GELU()

    def forward(self, x):  # x: [B, 1, 224]
        x = self.gelu(self.bn1(self.conv0(x)))   # [B, 64, 28]
        x = self.gelu(self.bn4(self.conv3(x)))   # [B, 128, 28]
        x = self.pool(x)                          # [B, 128, 14]
        x = self.gelu(self.bn8(self.conv7(x)))   # [B, 128, 14]
        return x.mean(dim=-1)                     # [B, 128]


def load_backbone(model_pt: str) -> NIRBackbone:
    data = torch.load(model_pt, map_location="cpu", weights_only=False)
    sd   = data["cnn_state"] if isinstance(data, dict) else data.state_dict()

    net = NIRBackbone()
    net.conv0.weight.data = sd["backbone.0.weight"]
    net.bn1.weight.data   = sd["backbone.1.weight"]
    net.bn1.bias.data     = sd["backbone.1.bias"]
    net.bn1.running_mean  = sd["backbone.1.running_mean"].clone()
    net.bn1.running_var   = sd["backbone.1.running_var"].clone()
    net.conv3.weight.data = sd["backbone.3.weight"]
    net.bn4.weight.data   = sd["backbone.4.weight"]
    net.bn4.bias.data     = sd["backbone.4.bias"]
    net.bn4.running_mean  = sd["backbone.4.running_mean"].clone()
    net.bn4.running_var   = sd["backbone.4.running_var"].clone()
    net.conv7.weight.data = sd["backbone.7.weight"]
    net.bn8.weight.data   = sd["backbone.8.weight"]
    net.bn8.bias.data     = sd["backbone.8.bias"]
    net.bn8.running_mean  = sd["backbone.8.running_mean"].clone()
    net.bn8.running_var   = sd["backbone.8.running_var"].clone()
    net.eval()
    return net


def float_to_hex16(val: float, scale: float) -> str:
    q = int(round(val / scale))
    q = max(-32767, min(32767, q))
    if q < 0:
        q += 65536
    return f"{q:04X}"


def write_hex_file(path: str, arr: np.ndarray, scale: float):
    with open(path, "w") as f:
        for v in arr.flatten():
            f.write(float_to_hex16(float(v), scale) + "\n")
    print(f"  Written: {path}  ({len(arr.flatten())} values, scale={scale:.8f})")


def main():
    args = parse_args()

    print(f"Loading backbone from {MODEL_PT} ...")
    net = load_backbone(MODEL_PT)

    print(f"Loading dataset from {args.csv} ...")
    df = pd.read_csv(args.csv)
    if "class_id" in df.columns:
        y = df["class_id"].values.astype(np.int32)
        X = df.drop(columns=["class_id"]).values.astype(np.float32)
    else:
        y = df.iloc[:, 0].values.astype(np.int32)
        X = df.iloc[:, 1:].values.astype(np.float32)

    print(f"  Samples: {len(X)}  Classes: {sorted(np.unique(y))}")

    print("Extracting features (backbone forward pass) ...")
    BATCH = 256
    features_list = []
    with torch.no_grad():
        for start in range(0, len(X), BATCH):
            batch = torch.tensor(X[start:start+BATCH]).unsqueeze(1)  # [B,1,224]
            feat  = net(batch).numpy()  # [B, 128]
            features_list.append(feat)
            if (start // BATCH) % 10 == 0:
                print(f"  {start+len(feat)}/{len(X)} samples processed ...", end="\r")
    features = np.concatenate(features_list, axis=0)
    print(f"\n  Features shape: {features.shape}")
    print(f"  Features min={features.min():.4f}  max={features.max():.4f}  mean={features.mean():.4f}")

    print()
    print("Fitting StandardScaler ...")
    scaler = StandardScaler()
    feat_scaled = scaler.fit_transform(features)
    print(f"  Scaler mean range: [{scaler.mean_.min():.4f}, {scaler.mean_.max():.4f}]")
    print(f"  Scaler std  range: [{scaler.scale_.min():.6f}, {scaler.scale_.max():.4f}]")

    print()
    print(f"Training LinearSVC (C={args.C}, max_iter={args.max_iter}) ...")
    svm = LinearSVC(C=args.C, max_iter=args.max_iter, dual=False, random_state=42)
    svm.fit(feat_scaled, y)
    preds = svm.predict(feat_scaled)
    acc   = accuracy_score(y, preds)
    print(f"  Training accuracy: {acc*100:.2f}%")
    print(f"  SVM intercept: {svm.intercept_}")
    print(f"  Classes: {svm.classes_}")

    # Quick per-class check
    print()
    print("  Per-class accuracy:")
    for cls in sorted(np.unique(y)):
        mask = y == cls
        cls_acc = accuracy_score(y[mask], preds[mask])
        print(f"    Class {cls}: {cls_acc*100:.1f}%  ({mask.sum()} samples)")

    print()
    print(f"Saving pipeline to {args.out_pkl} ...")
    old_pipeline = joblib.load(f"{V6_DIR}/nir_svm_pipeline (1).pkl")
    new_pipeline = {
        "svm":        svm,
        "scaler":     scaler,
        "dark_ref":   old_pipeline.get("dark_ref"),
        "n_channels": 224,
    }
    joblib.dump(new_pipeline, args.out_pkl)
    print(f"  Saved!")

    print()
    print("Writing hex files for generate_weights_c.py ...")
    from pathlib import Path
    hex_dir = Path(args.hex_dir)

    # Compute scales (same approach as generate_weights_c.py)
    svm_w   = svm.coef_          # [5, 128]
    svm_b   = svm.intercept_     # [5]
    sc_mean = scaler.mean_       # [128]
    sc_std  = scaler.scale_      # [128]

    scale_w = np.abs(svm_w).max() / 32767.0
    scale_b = np.abs(svm_b).max() / 32767.0
    scale_m = np.abs(sc_mean).max() / 32767.0
    scale_s = np.abs(sc_std).max() / 32767.0

    write_hex_file(str(hex_dir / "svm_weights_new.hex"),  svm_w,   scale_w)
    write_hex_file(str(hex_dir / "svm_biases_new.hex"),   svm_b,   scale_b)
    write_hex_file(str(hex_dir / "scaler_mean_new.hex"),  sc_mean, scale_m)
    write_hex_file(str(hex_dir / "scaler_std_new.hex"),   sc_std,  scale_s)

    print()
    print("NEW SCALES for nir_model.h (update these manually):")
    print(f"  #define SVM_W_SCALE       {scale_w:.8f}f")
    print(f"  #define SVM_B_SCALE       {scale_b:.8f}f")
    print(f"  #define SCALER_MEAN_SCALE {scale_m:.8f}f")
    print(f"  #define SCALER_STD_SCALE  {scale_s:.8f}f")

    print()
    print("NEXT STEPS:")
    print("  1. Copy svm_*_new.hex and scaler_*_new.hex -> nir_v6/  (rename, removing '_new')")
    print("  2. Update scales in nir_model.h with the values above")
    print("  3. Run: python generate_weights_c.py --input-dir . --out nir_weights.c")
    print("  4. Copy nir_weights.c to MCU_Firmware/")
    print("  5. Run: python generate_weights_bn_float.py --model 'nir_cnn_backbone (1).pt'")
    print("  6. Copy nir_bn_weights.h to MCU_Firmware/")
    print("  7. Rebuild and flash in CCS")


if __name__ == "__main__":
    main()
