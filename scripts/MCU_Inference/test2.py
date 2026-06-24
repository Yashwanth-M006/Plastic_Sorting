import torch, torch.nn as nn, numpy as np, joblib, re

V6 = r"E:\CSIR\medley\nir_v6"
H  = r"E:\CSIR\medley\MCU\MCU_Firmware\nir_selftest_data.h"

new_pipe = joblib.load(r"E:\CSIR\medley\MCU_Inference\nir_svm_pipeline_new.pkl")
svm, scaler = new_pipe["svm"], new_pipe["scaler"]

pt = torch.load(f"{V6}/nir_cnn_backbone (1).pt", map_location="cpu", weights_only=False)
sd = pt["cnn_state"]

bb0_W = sd["backbone.0.weight"].numpy()
bb3_W = sd["backbone.3.weight"].numpy()
bb7_W = sd["backbone.7.weight"].numpy()

bn1_w = sd["backbone.1.weight"].numpy()
bn1_b = sd["backbone.1.bias"].numpy()
bn1_m = sd["backbone.1.running_mean"].numpy()
bn1_v = sd["backbone.1.running_var"].numpy()

bn4_w = sd["backbone.4.weight"].numpy()
bn4_b = sd["backbone.4.bias"].numpy()
bn4_m = sd["backbone.4.running_mean"].numpy()
bn4_v = sd["backbone.4.running_var"].numpy()

bn8_w = sd["backbone.8.weight"].numpy()
bn8_b = sd["backbone.8.bias"].numpy()
bn8_m = sd["backbone.8.running_mean"].numpy()
bn8_v = sd["backbone.8.running_var"].numpy()

with open(H) as f:
    txt = f.read()

labels = list(map(int, re.findall(r"g_selftest_labels\[10\]\s*=\s*\{([^}]+)\}", txt)[0].split(",")))
blks = re.split(r"/\* sample \d+ [^*]*\*/", txt)
samples = []
for blk in blks[1:]:
    nums = re.findall(r"[-+]?\d+\.\d+e?[+-]?\d*f", blk)
    vals = [float(n.rstrip("f")) for n in nums]
    if len(vals) >= 224:
        samples.append(vals[:224])

X = np.array(samples[:10], dtype=np.float32)

def gelu(x):
    x3 = x * x * x
    return 0.5 * x * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x3)))

# Run on Sample 0 (Sample 1 in C)
in_val = X[0]
print("Input first 3:", in_val[:3])

# CNN
act0 = np.zeros((64, 28), dtype=np.float32)
for oc in range(64):
    for p in range(28):
        acc = 0.0
        in_start = p * 8 - 12
        for k in range(25):
            pos = in_start + k
            if 0 <= pos < 224:
                acc += in_val[pos] * bb0_W[oc, 0, k]
        act0[oc, p] = acc
        
for oc in range(64):
    for p in range(28):
        val = act0[oc, p]
        val = (val - bn1_m[oc]) / np.sqrt(bn1_v[oc] + 1e-5) * bn1_w[oc] + bn1_b[oc]
        act0[oc, p] = gelu(val)
        
act1 = np.zeros((128, 28), dtype=np.float32)
for oc in range(128):
    for p in range(28):
        acc = 0.0
        in_start = p - 2
        for ic in range(64):
            for k in range(5):
                pos = in_start + k
                if 0 <= pos < 28:
                    acc += act0[ic, pos] * bb3_W[oc, ic, k]
        act1[oc, p] = acc
        
for oc in range(128):
    for p in range(28):
        val = act1[oc, p]
        val = (val - bn4_m[oc]) / np.sqrt(bn4_v[oc] + 1e-5) * bn4_w[oc] + bn4_b[oc]
        act1[oc, p] = gelu(val)
        
act1_pooled = np.zeros((128, 28), dtype=np.float32)
for oc in range(128):
    for p in range(14):
        act1_pooled[oc, p] = max(act1[oc, 2 * p], act1[oc, 2 * p + 1])
        
act2 = np.zeros((128, 14), dtype=np.float32)
act1_flat = act1_pooled.flatten()

for oc in range(128):
    for p in range(14):
        acc = 0.0
        in_start = p - 1
        for ic in range(128):
            for k in range(3):
                pos = in_start + k
                if 0 <= pos < 14:
                    flat_idx = ic * 28 + pos
                    acc += act1_flat[flat_idx] * bb7_W[oc, ic, k]
        act2[oc, p] = acc
        
for oc in range(128):
    for p in range(14):
        val = act2[oc, p]
        val = (val - bn8_m[oc]) / np.sqrt(bn8_v[oc] + 1e-5) * bn8_w[oc] + bn8_b[oc]
        act2[oc, p] = gelu(val)
        
features = np.zeros(128, dtype=np.float32)
for oc in range(128):
    features[oc] = np.mean(act2[oc])

print("Features first 3:", features[:3])

fs = scaler.transform(features.reshape(1, -1))[0]
scores = svm.coef_ @ fs + svm.intercept_
print("Expected Scores (Python float):", scores)