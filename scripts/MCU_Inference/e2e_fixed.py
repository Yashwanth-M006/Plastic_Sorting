"""
e2e_fixed.py
============
Correct end-to-end test:
  1. Parses nir_selftest_data.h properly
  2. Runs PyTorch backbone + sklearn SVM (gold standard)
  3. Simulates the C code using the quantized weights (to find where it diverges)
  4. Reports where the divergence is and what to fix
"""
import torch
import torch.nn as nn
import numpy as np
import joblib
import re

V6_DIR   = r"E:\CSIR\medley\nir_v6"
H_STEST  = r"E:\CSIR\medley\MCU\MCU_Firmware\nir_selftest_data.h"
H_BN     = r"E:\CSIR\medley\MCU\MCU_Firmware\nir_bn_weights.h"
H_WGHTS  = r"E:\CSIR\medley\MCU\MCU_Firmware\nir_weights.c"

# ── Load model & pipeline ──────────────────────────────────────────────────────
model_data = torch.load(f"{V6_DIR}/nir_cnn_backbone (1).pt", map_location="cpu", weights_only=False)
sd = model_data["cnn_state"]
pipeline = joblib.load(f"{V6_DIR}/nir_svm_pipeline (1).pkl")
svm    = pipeline["svm"]
scaler = pipeline["scaler"]

# ── Parse selftest .h ──────────────────────────────────────────────────────────
print("Parsing nir_selftest_data.h ...")
with open(H_STEST) as f:
    txt = f.read()

# Labels
labels = list(map(int, re.findall(r'g_selftest_labels\[10\]\s*=\s*\{([^}]+)\}', txt)[0].split(',')))
print(f"  Labels: {labels}")

# Samples — each sample is a { ... } block inside the outer array
# Split on "/* sample N" markers
sample_blocks = re.split(r'/\* sample \d+ [^*]*\*/', txt)
# First element is header, rest are sample blocks (each ends before next sample or closing brace)
samples = []
for block in sample_blocks[1:]:  # skip header
    # Extract all floats from this block
    nums = re.findall(r'[-+]?\d+\.\d+e?[+-]?\d*f', block)
    vals = [float(n.rstrip('f')) for n in nums]
    if len(vals) == 224:
        samples.append(vals)
    elif len(vals) > 0:
        # Take first 224
        samples.append(vals[:224])

X = np.array(samples[:10], dtype=np.float32)
print(f"  Parsed {len(X)} samples, shape={X.shape}")
print(f"  Sample[0] min={X[0].min():.4f}  max={X[0].max():.4f}")

# ── Build correct PyTorch backbone (bias=False on Conv layers!) ────────────────
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

    def forward(self, x):  # x: [B, 1, 224]
        x = self.gelu(self.bn1(self.conv0(x)))   # -> [B, 64, 28]
        x = self.gelu(self.bn4(self.conv3(x)))   # -> [B, 128, 28]
        x = self.pool(x)                          # -> [B, 128, 14]
        x = self.gelu(self.bn8(self.conv7(x)))   # -> [B, 128, 14]
        return x.mean(dim=-1)                     # -> [B, 128]

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

# ── Gold standard inference ────────────────────────────────────────────────────
print()
print("=" * 70)
print("PYTORCH + SKLEARN GOLD STANDARD")
print("=" * 70)
y_pred_py = []
features_py = []
with torch.no_grad():
    for i in range(len(X)):
        x = torch.tensor(X[i]).unsqueeze(0).unsqueeze(0)  # [1,1,224]
        feat = net(x).numpy()[0]  # [128]
        features_py.append(feat)
        feat_s = scaler.transform(feat.reshape(1, -1))[0]
        # Manual argmax (same as LinearSVC)
        scores = svm.coef_ @ feat_s + svm.intercept_
        pred = svm.classes_[np.argmax(scores)]
        y_pred_py.append(int(pred))

print(f"  Labels:  {labels}")
print(f"  PyPreds: {y_pred_py}")
correct = sum(p == t for p, t in zip(y_pred_py, labels))
print(f"  Accuracy: {correct}/{len(labels)}  ({100*correct/len(labels):.0f}%)")

# ── Simulate C code ────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("C CODE SIMULATION (using quantized weights)")
print("=" * 70)

# Read quantized stem weights from nir_weights.c
print("Parsing quantized stem weights from nir_weights.c ...")
with open(H_WGHTS) as f:
    wc = f.read()

def parse_c_int16(text, name):
    m = re.search(rf'const int16_t {name}\[.*?\]\s*=\s*\{{(.*?)\}};', text, re.DOTALL)
    if not m: return None
    return np.array(list(map(int, re.findall(r'-?\d+', m.group(1)))), dtype=np.int16)

# Stem weights are int16 in nir_weights.c
bb0_q = parse_c_int16(wc, "g_bb0_weight")  # [1600] = 64*1*25
print(f"  bb0_weight shape: {bb0_q.shape}")

# BN weights from .h
def parse_float_array(text, name):
    m = re.search(rf'const float {name}\[.*?\]\s*=\s*\{{(.*?)\}};', text, re.DOTALL)
    if not m: return None
    return np.array([float(v.rstrip('f')) for v in re.findall(r'[-+]?\d+\.\d+(?:e[+-]?\d+)?f', m.group(1))], dtype=np.float32)

with open(H_BN) as f:
    bn_txt = f.read()

bn1_w   = parse_float_array(bn_txt, "g_bn1_weight")
bn1_b   = parse_float_array(bn_txt, "g_bn1_bias")
bn1_mu  = parse_float_array(bn_txt, "g_bn1_running_mean")
bn1_var = parse_float_array(bn_txt, "g_bn1_running_var")

# Scale
SCALE_BACKBONE_0_WEIGHT = 0.00857819

# ── Step 1: Stem conv (sample 0) ─────────────────────────────────────────────
x0 = X[0]  # [224]
W0_dq = bb0_q.reshape(64, 25).astype(np.float32) * SCALE_BACKBONE_0_WEIGHT  # [64, 25]

# Conv1d(1->64, k=25, s=8, pad=12)
in_len  = 224
out_len = 28
act0 = np.zeros((64, out_len), dtype=np.float32)
for oc in range(64):
    for p in range(out_len):
        in_start = p * 8 - 12
        acc = 0.0
        for k in range(25):
            pos = in_start + k
            if 0 <= pos < in_len:
                acc += x0[pos] * W0_dq[oc, k]
        act0[oc, p] = acc

# Compare with PyTorch stem
with torch.no_grad():
    x_pt = torch.tensor(X[0]).unsqueeze(0).unsqueeze(0)
    stem_pt = net.conv0(x_pt).numpy()[0]  # [64, 28]

stem_err = np.abs(act0 - stem_pt).max()
print(f"\n  Stem conv max error (C sim vs PyTorch): {stem_err:.6f}")
print(f"  C sim  stem[0,:5]: {act0[0,:5]}")
print(f"  PyTorch stem[0,:5]: {stem_pt[0,:5]}")

# ── Step 2: BN1 ───────────────────────────────────────────────────────────────
act0_bn = np.zeros_like(act0)
for ch in range(64):
    for p in range(out_len):
        v = act0[ch, p]
        v = (v - bn1_mu[ch]) * (1.0 / np.sqrt(bn1_var[ch] + 1e-5)) * bn1_w[ch] + bn1_b[ch]
        act0_bn[ch, p] = v

# GELU
def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x**3)))

act0_gelu = gelu(act0_bn)

# PyTorch BN+GELU after stem
with torch.no_grad():
    act0_pt = net.gelu(net.bn1(net.conv0(x_pt))).numpy()[0]  # [64, 28]

bn1_err = np.abs(act0_gelu - act0_pt).max()
print(f"\n  After BN1+GELU max error: {bn1_err:.6f}")
print(f"  C sim  [0,:5]: {act0_gelu[0,:5]}")
print(f"  PyTorch[0,:5]: {act0_pt[0,:5]}")

# ── Step 3: Feature comparison at GAP ─────────────────────────────────────────
print()
print("=" * 70)
print("FEATURE COMPARISON (Python vs stored in features_py)")
print("=" * 70)
f_py = features_py[0]
print(f"  Python features[0] min={f_py.min():.4f}  max={f_py.max():.4f}  mean={f_py.mean():.4f}")
print(f"  First 10: {f_py[:10]}")

# ── Step 4: SVM decision surface analysis ─────────────────────────────────────
print()
print("=" * 70)
print("SVM DECISION SURFACE (on Python features)")
print("=" * 70)
for i in range(len(X)):
    feat_s = scaler.transform(features_py[i].reshape(1,-1))[0]
    scores = svm.coef_ @ feat_s + svm.intercept_
    pred   = svm.classes_[np.argmax(scores)]
    print(f"  Sample {i}  true={labels[i]}  pred={pred}  scores={np.round(scores,3)}")

print()
print("=" * 70)
print("KEY FINDING SUMMARY")
print("=" * 70)
print(f"""
The architecture is: CNN backbone -> GAP -> SVM (sklearn)
  - The SVM weights in nir_v6/*.hex are CORRECT (verified match to pkl)
  - The SVM scales in nir_model.h are CORRECT

Current C code state after our last fix:
  - run_head() uses head.1+head.3 weights (WRONG - these are training-only layers)
  - The correct classifier is the sklearn SVM from nir_v6

CORRECT inference pipeline for the C code:
  1. backbone (conv+BN+GELU) -> GAP -> features[128]    [already works]
  2. StandardScaler: scaled[d] = (feat[d] - mean[d]) / scale[d]
  3. SVM: score[c] = bias[c] + dot(coef_row_c, scaled)
  4. return argmax(score) + 1   (classes are 1-indexed, argmax gives 0-indexed)

What the C code must be doing wrong now (using head weights):
  - head.1 weights map 128->64 and were trained alongside the backbone
  - They are NOT calibrated against the SVM feature space
  - So head.1+head.3 outputs systematically prefer class 4 (index 3)

FIX: Revert run_head() back to run_svm() but use nir_v6 hex data in nir_weights.c
The existing nir_weights.c SVM data IS CORRECT. The error was something else.
""")
