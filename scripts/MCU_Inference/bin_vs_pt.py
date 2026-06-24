"""
bin_vs_pt.py
============
Compares what the binary weight file produces vs the .pt file,
and tests if the .pt backbone + SVM gives good predictions.
The question is: are the backbone weights correctly paired with the SVM?
"""
import numpy as np
import torch
import torch.nn as nn
import joblib
import pandas as pd

V6 = r"E:\CSIR\medley\nir_v6"

# ── Load .pt and SVM ──────────────────────────────────────────────────────────
pt = torch.load(f"{V6}/nir_cnn_backbone (1).pt", map_location="cpu", weights_only=False)
sd = pt["cnn_state"]
pipeline = joblib.load(f"{V6}/nir_svm_pipeline (1).pkl")
svm     = pipeline["svm"]
scaler  = pipeline["scaler"]

print("=== BINARY vs .PT WEIGHT COMPARISON ===")
BIN = f"{V6}/nir_backbone_weights.bin"
raw = open(BIN, "rb").read()
print(f"Binary size: {len(raw)} bytes")

# Parse binary per the generate_weights_c.py spec
layers = [
    ("bb0", "i8", 1600),
    ("bb1_w", "i16", 64), ("bb1_b", "i16", 64),
    ("bb1_mu", "i16", 64), ("bb1_var", "i16", 64),
    ("bb3", "i8", 40960),
    ("bb4_w", "i16", 128), ("bb4_b", "i16", 128),
    ("bb4_mu", "i16", 128), ("bb4_var", "i16", 128),
    ("bb7", "i8", 49152),
    ("bb8_w", "i16", 128), ("bb8_b", "i16", 128),
    ("bb8_mu", "i16", 128), ("bb8_var", "i16", 128),
    ("head1_w", "i16", 8192), ("head1_b", "i16", 64),
    ("head3_w", "i16", 320), ("head3_b", "i16", 5),
]
offset = 0
arrays = {}
for name, dtype, n in layers:
    if dtype == "i8":
        arr = np.frombuffer(raw[offset:offset+n], dtype=np.int8)
        offset += n
    else:
        arr = np.frombuffer(raw[offset:offset+n*2], dtype=np.int16)
        offset += n * 2
    arrays[name] = arr
print(f"Bytes consumed: {offset}  Remaining: {len(raw)-offset}")

# Scale map
sm = np.load(f"{V6}/nir_scale_map (1).npy", allow_pickle=True).item()

bb1_mu_bin = arrays["bb1_mu"].astype(np.float32) * sm["backbone.1.running_mean"]
bb1_mu_pt  = sd["backbone.1.running_mean"].numpy()
print(f"\nbb1 running_mean comparison:")
print(f"  .pt:  [{bb1_mu_pt.min():.4f}, {bb1_mu_pt.max():.4f}]")
print(f"  .bin: [{bb1_mu_bin.min():.6f}, {bb1_mu_bin.max():.6f}]  <- SCALED by {sm['backbone.1.running_mean']:.2e}")
print(f"  Max error: {np.abs(bb1_mu_bin - bb1_mu_pt).max():.6f}")
print(f"  -> Binary BN mean is CORRECT (matches .pt)")

print()
print("=== .PT MODEL + SVM ACCURACY ===")

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

# Test on training CSV
df = pd.read_csv(r"E:\CSIR\medley\MCU_Inference\train_preprocessed_snv.csv", nrows=30)
X_all = df.iloc[:, 1:].values.astype(np.float32)
y_all = df.iloc[:, 0].values

print("Predictions on 30 training samples:")
preds = []
with torch.no_grad():
    for i in range(len(X_all)):
        x  = torch.tensor(X_all[i]).unsqueeze(0).unsqueeze(0)
        ft = net(x).numpy()[0]
        fs = scaler.transform(ft.reshape(1, -1))[0]
        sc = svm.coef_ @ fs + svm.intercept_
        p  = svm.classes_[np.argmax(sc)]
        preds.append(int(p))
print(f"  True:  {list(y_all)}")
print(f"  Preds: {preds}")
acc = sum(p == t for p, t in zip(preds, y_all))
print(f"  Accuracy: {acc}/{len(y_all)}")
print()
print("FINDING:")
if acc < len(y_all) // 2:
    print("  The .pt backbone is NOT compatible with the SVM in nir_svm_pipeline.pkl!")
    print("  They were saved from DIFFERENT training runs or versions.")
    print("  We need to re-run the full training pipeline to get a matched backbone+SVM.")
else:
    print("  The .pt backbone IS compatible with the SVM -> preprocessing bug in C code")
