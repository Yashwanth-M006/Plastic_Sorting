"""
e2e_test.py
===========
Runs a complete end-to-end inference using the SAME weights as the C code
to figure out where the CNN feature output diverges from expected.

Tests:
1. Full PyTorch forward pass (gold standard)
2. Simulates C code step-by-step using the quantized weights
3. Compares features at each stage to locate the discrepancy
"""
import torch
import torch.nn as nn
import numpy as np
import joblib

V6_DIR    = r"E:\CSIR\medley\nir_v6"
SELFTEST  = r"E:\CSIR\medley\MCU\MCU_Firmware\nir_selftest_data.h"
BN_H      = r"E:\CSIR\medley\MCU\MCU_Firmware\nir_bn_weights.h"

print("Loading model and SVM pipeline...")
model_data = torch.load(f"{V6_DIR}/nir_cnn_backbone (1).pt", map_location="cpu", weights_only=False)
sd         = model_data["cnn_state"]
pipeline   = joblib.load(f"{V6_DIR}/nir_svm_pipeline (1).pkl")
svm        = pipeline["svm"]
scaler     = pipeline["scaler"]

# ── Build PyTorch model ────────────────────────────────────────────────────────
class NIRBackbone(nn.Module):
    def __init__(self, sd):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=25, stride=8, padding=12),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),
        )
    def forward(self, x):
        x = self.backbone(x)
        return x.mean(dim=-1)  # GAP

net = NIRBackbone(sd)
net.load_state_dict({
    "backbone.0.weight": sd["backbone.0.weight"],
    "backbone.0.bias":   torch.zeros(64),
    "backbone.1.weight": sd["backbone.1.weight"],
    "backbone.1.bias":   sd["backbone.1.bias"],
    "backbone.1.running_mean": sd["backbone.1.running_mean"],
    "backbone.1.running_var":  sd["backbone.1.running_var"],
    "backbone.1.num_batches_tracked": sd["backbone.1.num_batches_tracked"],
    "backbone.3.weight": sd["backbone.3.weight"],
    "backbone.3.bias":   torch.zeros(128),
    "backbone.4.weight": sd["backbone.4.weight"],
    "backbone.4.bias":   sd["backbone.4.bias"],
    "backbone.4.running_mean": sd["backbone.4.running_mean"],
    "backbone.4.running_var":  sd["backbone.4.running_var"],
    "backbone.4.num_batches_tracked": sd["backbone.4.num_batches_tracked"],
    "backbone.6.weight": torch.ones(128),   # MaxPool dummy (no weight)
    "backbone.7.weight": sd["backbone.7.weight"],
    "backbone.7.bias":   torch.zeros(128),
    "backbone.8.weight": sd["backbone.8.weight"],
    "backbone.8.bias":   sd["backbone.8.bias"],
    "backbone.8.running_mean": sd["backbone.8.running_mean"],
    "backbone.8.running_var":  sd["backbone.8.running_var"],
    "backbone.8.num_batches_tracked": sd["backbone.8.num_batches_tracked"],
}, strict=False)

# Correct load without MaxPool dummy
net2 = nn.Sequential(
    nn.Conv1d(1, 64, kernel_size=25, stride=8, padding=12, bias=False),
    nn.BatchNorm1d(64),
    nn.GELU(),
    nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2, bias=False),
    nn.BatchNorm1d(128),
    nn.GELU(),
    nn.MaxPool1d(2, 2),
    nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
    nn.BatchNorm1d(128),
    nn.GELU(),
)

# Load weights manually
net2[0].weight.data = sd["backbone.0.weight"]
net2[1].weight.data = sd["backbone.1.weight"]
net2[1].bias.data   = sd["backbone.1.bias"]
net2[1].running_mean = sd["backbone.1.running_mean"]
net2[1].running_var  = sd["backbone.1.running_var"]
net2[3].weight.data = sd["backbone.3.weight"]
net2[4].weight.data = sd["backbone.4.weight"]
net2[4].bias.data   = sd["backbone.4.bias"]
net2[4].running_mean = sd["backbone.4.running_mean"]
net2[4].running_var  = sd["backbone.4.running_var"]
net2[7].weight.data = sd["backbone.7.weight"]
net2[8].weight.data = sd["backbone.8.weight"]
net2[8].bias.data   = sd["backbone.8.bias"]
net2[8].running_mean = sd["backbone.8.running_mean"]
net2[8].running_var  = sd["backbone.8.running_var"]
net2.eval()

# ── Pull a few self-test samples from the .h file ──────────────────────────────
print("Parsing selftest samples from nir_selftest_data.h ...")
import re
with open(SELFTEST) as f:
    text = f.read()

# Find all float arrays for samples (g_selftest_X)
label_matches = re.findall(r'g_selftest_labels\s*\[.*?\]\s*=\s*\{([^}]+)\}', text, re.DOTALL)
labels_raw = label_matches[0] if label_matches else ""
labels = [int(x) for x in re.findall(r'\b\d+\b', labels_raw)]

sample_matches = re.findall(r'g_selftest_X\s*\[.*?\]\s*\[.*?\]\s*=\s*\{(.*?)\};', text, re.DOTALL)
if not sample_matches:
    print("Could not parse selftest samples -- trying alternate format")
    # Try flat array
    flat_matches = re.findall(r'[-+]?\d*\.\d+(?:e[+-]?\d+)?f', text)
    if flat_matches:
        vals = [float(v.rstrip('f')) for v in flat_matches]
        n_samp = len(labels)
        n_ch   = 224
        X = np.array(vals[:n_samp*n_ch], dtype=np.float32).reshape(n_samp, n_ch)
    else:
        print("ERROR: cannot parse samples")
        X = np.random.randn(10, 224).astype(np.float32)
else:
    all_vals = []
    for m in sample_matches:
        vals = [float(v.rstrip('f')) for v in re.findall(r'[-+]?\d*\.\d+(?:e[+-]?\d+)?f', m)]
        all_vals.append(vals)
    X = np.array(all_vals, dtype=np.float32)

print(f"  Got {len(X)} samples, {len(labels)} labels")
print(f"  Sample[0] shape: {X[0].shape}  min={X[0].min():.4f}  max={X[0].max():.4f}")

# ── Run PyTorch inference ──────────────────────────────────────────────────────
print()
print("=" * 70)
print("PYTORCH GOLD STANDARD")
print("=" * 70)
y_pred_python = []
with torch.no_grad():
    for i in range(len(X)):
        x = torch.tensor(X[i]).unsqueeze(0).unsqueeze(0)  # [1, 1, 224]
        feat = net2(x).mean(dim=-1).numpy()[0]             # [128]
        # SVM predict
        feat_scaled = scaler.transform(feat.reshape(1, -1))
        pred = svm.predict(feat_scaled)[0]
        y_pred_python.append(pred)

print(f"  Labels: {labels}")
print(f"  Python preds: {y_pred_python}")
correct = sum(p == t for p, t in zip(y_pred_python, labels))
print(f"  Accuracy: {correct}/{len(labels)}")

# ── Check backbone.0 (stem) PADDING ───────────────────────────────────────────
print()
print("=" * 70)
print("STEM CONV PADDING CHECK")
print("=" * 70)
x = torch.tensor(X[0]).unsqueeze(0).unsqueeze(0)
with torch.no_grad():
    after_stem = net2[0](x)
print(f"  Input shape: {x.shape}")
print(f"  After Conv1d(k=25,s=8,p=12): {after_stem.shape}  <- should be [1,64,28]")
# The C code uses STEM_OUT_LEN=28, but let's confirm:
# out = floor((224 + 2*12 - 25)/8) + 1 = floor(223/8)+1 = 27+1 = 28
print(f"  C code STEM_OUT_LEN = 28.  PyTorch output len = {after_stem.shape[2]}  MATCH: {after_stem.shape[2]==28}")

# Check block1 after maxpool
with torch.no_grad():
    after_bn1  = net2[:3](x)        # stem + BN + GELU
    after_bl1  = net2[:6](x)        # + block1 BN GELU
    after_mp   = net2[:7](x)        # + maxpool
    after_bl2  = net2[:10](x)       # + block2
print(f"  After stem+BN+GELU:    {after_bn1.shape}  len={after_bn1.shape[2]}")
print(f"  After block1+BN+GELU:  {after_bl1.shape}  len={after_bl1.shape[2]}")
print(f"  After MaxPool(2,2):    {after_mp.shape}   len={after_mp.shape[2]}")
print(f"  After block2+BN+GELU:  {after_bl2.shape}  len={after_bl2.shape[2]}")
print(f"  C code BLK1_OUT_LEN=14. PyTorch pool output = {after_mp.shape[2]}  MATCH: {after_mp.shape[2]==14}")

# ── Verify bn_weights.h values match the .pt ──────────────────────────────────
print()
print("=" * 70)
print("BN WEIGHTS VERIFICATION (nir_bn_weights.h vs .pt)")
print("=" * 70)

# Parse a few values from the .h file
with open(BN_H) as f:
    bn_text = f.read()

def parse_c_array(text, name):
    m = re.search(rf'const float {name}\[.*?\]\s*=\s*\{{(.*?)\}};', text, re.DOTALL)
    if not m:
        return None
    vals = re.findall(r'[-+]?\d*\.\d+(?:e[+-]?\d+)?f', m.group(1))
    return np.array([float(v.rstrip('f')) for v in vals], dtype=np.float32)

bn1_w   = parse_c_array(bn_text, "g_bn1_weight")
bn1_mu  = parse_c_array(bn_text, "g_bn1_running_mean")
bn1_var = parse_c_array(bn_text, "g_bn1_running_var")

pt_w   = sd["backbone.1.weight"].float().numpy()
pt_mu  = sd["backbone.1.running_mean"].float().numpy()
pt_var = sd["backbone.1.running_var"].float().numpy()

if bn1_w is not None:
    print(f"  BN1 weight:  H max_err={np.abs(bn1_w-pt_w).max():.6f}")
    print(f"  BN1 mean:    H max_err={np.abs(bn1_mu-pt_mu).max():.6f}")
    print(f"  BN1 var:     H max_err={np.abs(bn1_var-pt_var).max():.6f}")
    print(f"  First 5 BN1 mean  .h: {bn1_mu[:5]}  .pt: {pt_mu[:5]}")
    print(f"  First 5 BN1 var   .h: {bn1_var[:5]}  .pt: {pt_var[:5]}")
else:
    print("  Could not parse nir_bn_weights.h")
