"""
feature_debug.py
================
Investigates WHY the CNN produces features that make SVM always predict class 3.

Hypothesis: The selftest samples in nir_selftest_data.h might be
  - Already SNV-normalized -> correct
  - Raw spectra that need dark_ref subtraction first -> WRONG
  - Or some other preprocessing mismatch

Checks:
  1. What preprocessing the nir_v6 training pipeline expects
  2. What the selftest .h file actually contains
  3. Simulate the full pipeline with correct preprocessing
"""
import torch
import torch.nn as nn
import numpy as np
import joblib, re
from sklearn.preprocessing import StandardScaler

V6_DIR  = r"E:\CSIR\medley\nir_v6"
H_STEST = r"E:\CSIR\medley\MCU\MCU_Firmware\nir_selftest_data.h"
H_GEN   = r"E:\CSIR\medley\MCU_Inference\generate_selftest_h.py"

model_data = torch.load(f"{V6_DIR}/nir_cnn_backbone (1).pt", map_location="cpu", weights_only=False)
sd         = model_data["cnn_state"]
pipeline   = joblib.load(f"{V6_DIR}/nir_svm_pipeline (1).pkl")
svm        = pipeline["svm"]
scaler_svm = pipeline["scaler"]
dark_ref   = pipeline.get("dark_ref", None)
n_ch       = pipeline.get("n_channels", 224)

print("=" * 70)
print("PART 1: TRAINING PIPELINE PREPROCESSING")
print("=" * 70)
print(f"dark_ref present: {dark_ref is not None}")
if dark_ref is not None:
    print(f"  dark_ref shape={dark_ref.shape}  min={dark_ref.min():.2f}  max={dark_ref.max():.2f}")
    print(f"  dark_ref mean={dark_ref.mean():.2f}")
print(f"n_channels: {n_ch}")

# Check generate_selftest_h.py for preprocessing logic
print()
print("=" * 70)
print("PART 2: SELFTEST GENERATION SCRIPT PREPROCESSING LOGIC")
print("=" * 70)
with open(H_GEN) as f:
    gen_text = f.read()
print(gen_text[:3000])

# Parse selftest samples
print()
print("=" * 70)
print("PART 3: SELFTEST SAMPLE PROPERTIES")
print("=" * 70)
with open(H_STEST) as f:
    txt = f.read()

labels = list(map(int, re.findall(r'g_selftest_labels\[10\]\s*=\s*\{([^}]+)\}', txt)[0].split(',')))
sample_blocks = re.split(r'/\* sample \d+ [^*]*\*/', txt)
samples = []
for block in sample_blocks[1:]:
    nums = re.findall(r'[-+]?\d+\.\d+e?[+-]?\d*f', block)
    vals = [float(n.rstrip('f')) for n in nums]
    if len(vals) >= 224:
        samples.append(vals[:224])

X = np.array(samples[:10], dtype=np.float32)
print(f"Selftest samples: {X.shape}")
for i in range(min(3, len(X))):
    print(f"  Sample {i} (class {labels[i]}): min={X[i].min():.4f}  max={X[i].max():.4f}  "
          f"mean={X[i].mean():.4f}  std={X[i].std():.4f}")

print()
print("SNV check (mean should be ~0, std should be ~1 if SNV-normalized):")
for i in range(min(3, len(X))):
    print(f"  Sample {i}: mean={X[i].mean():.4f}  std={X[i].std():.4f}  "
          f"{'SNV OK' if abs(X[i].mean()) < 0.01 and abs(X[i].std()-1.0) < 0.05 else 'NOT SNV!'}")

# ── Load actual training data to compare ──────────────────────────────────────
print()
print("=" * 70)
print("PART 4: RAW TRAINING CSV vs SELFTEST PREPROCESSING")
print("=" * 70)
import pandas as pd
CSV = r"E:\CSIR\medley\MCU_Inference\train_preprocessed_snv.csv"
try:
    df = pd.read_csv(CSV, nrows=50)
    print(f"CSV columns: {list(df.columns[:5])} ...")
    if "class_id" in df.columns:
        X_csv = df.drop(columns=["class_id"]).values.astype(np.float32)
        y_csv = df["class_id"].values
    else:
        X_csv = df.iloc[:, 1:].values.astype(np.float32)
        y_csv = df.iloc[:, 0].values

    print(f"CSV shape: {X_csv.shape}")
    print(f"CSV sample 0: min={X_csv[0].min():.4f}  max={X_csv[0].max():.4f}  mean={X_csv[0].mean():.4f}")

    # Run the CORRECT forward pass on CSV samples
    class NIRBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv0 = nn.Conv1d(1, 64, 25, stride=8, padding=12, bias=False)
            self.bn1   = nn.BatchNorm1d(64)
            self.conv3 = nn.Conv1d(64, 128, 5, stride=1, padding=2, bias=False)
            self.bn4   = nn.BatchNorm1d(128)
            self.pool  = nn.MaxPool1d(2, 2)
            self.conv7 = nn.Conv1d(128, 128, 3, stride=1, padding=1, bias=False)
            self.bn8   = nn.BatchNorm1d(128)
            self.gelu  = nn.GELU()
        def forward(self, x):
            x = self.gelu(self.bn1(self.conv0(x)))
            x = self.gelu(self.bn4(self.conv3(x)))
            x = self.pool(x)
            x = self.gelu(self.bn8(self.conv7(x)))
            return x.mean(dim=-1)

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

    print()
    print("Inference on CSV samples (ground truth data):")
    with torch.no_grad():
        for i in range(min(10, len(X_csv))):
            x  = torch.tensor(X_csv[i]).unsqueeze(0).unsqueeze(0)
            ft = net(x).numpy()[0]
            fs = scaler_svm.transform(ft.reshape(1,-1))[0]
            sc = svm.coef_ @ fs + svm.intercept_
            p  = svm.classes_[np.argmax(sc)]
            print(f"  CSV sample {i}  true={y_csv[i]}  pred={p}  "
                  f"scores={np.round(sc,2)}  feat_mean={ft.mean():.3f}")

    print()
    print("Inference on SELFTEST samples:")
    with torch.no_grad():
        for i in range(len(X)):
            x  = torch.tensor(X[i]).unsqueeze(0).unsqueeze(0)
            ft = net(x).numpy()[0]
            fs = scaler_svm.transform(ft.reshape(1,-1))[0]
            sc = svm.coef_ @ fs + svm.intercept_
            p  = svm.classes_[np.argmax(sc)]
            print(f"  ST  sample {i}  true={labels[i]}  pred={p}  "
                  f"scores={np.round(sc,2)}  feat_mean={ft.mean():.3f}")

    print()
    print("Feature comparison (CSV vs Selftest):")
    with torch.no_grad():
        x_csv0 = torch.tensor(X_csv[0]).unsqueeze(0).unsqueeze(0)
        ft_csv = net(x_csv0).numpy()[0]
        x_st0  = torch.tensor(X[0]).unsqueeze(0).unsqueeze(0)
        ft_st  = net(x_st0).numpy()[0]
    print(f"  CSV sample 0 features: mean={ft_csv.mean():.4f}  min={ft_csv.min():.4f}  max={ft_csv.max():.4f}")
    print(f"  ST  sample 0 features: mean={ft_st.mean():.4f}   min={ft_st.min():.4f}  max={ft_st.max():.4f}")
    print(f"  Feature max_diff = {np.abs(ft_csv - ft_st).max():.4f}")

except Exception as e:
    import traceback
    print(f"Error loading CSV: {e}")
    traceback.print_exc()
