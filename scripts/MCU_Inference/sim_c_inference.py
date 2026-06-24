"""
sim_c_inference.py - Simulate the C inference code behavior in Python.
"""
import torch, torch.nn as nn, numpy as np, joblib, re

V6 = r"E:\CSIR\medley\nir_v6"
H  = r"E:\CSIR\medley\MCU\MCU_Firmware\nir_selftest_data.h"

# Load new SVM pipeline
new_pipe = joblib.load(r"E:\CSIR\medley\MCU_Inference\nir_svm_pipeline_new.pkl")
svm, scaler = new_pipe["svm"], new_pipe["scaler"]

# Load backbone weights
pt = torch.load(f"{V6}/nir_cnn_backbone (1).pt", map_location="cpu", weights_only=False)
sd = pt["cnn_state"]

# Extract weights from PyTorch state dict
bb0_W = sd["backbone.0.weight"].numpy() # (64, 1, 25)
bb3_W = sd["backbone.3.weight"].numpy() # (128, 64, 5)
bb7_W = sd["backbone.7.weight"].numpy() # (128, 128, 3)

# BatchNorm parameters (float32)
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

# Read selftest data
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
    # C GELU approximation: 0.5 * x * (1.0 + tanh(0.7978845608 * (x + 0.044715 * x^3)))
    x3 = x * x * x
    return 0.5 * x * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x3)))

def run_sim(bugged=False):
    correct = 0
    predictions = []
    
    for i in range(len(X)):
        # 1. Stem Conv: in_ch=1, out_ch=64, ksize=25, stride=8, pad=12
        # Input spectrum is 224 floats. Output length is 28.
        in_val = X[i]
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
        
        # 2. BN1 + GELU
        for oc in range(64):
            for p in range(28):
                val = act0[oc, p]
                val = (val - bn1_m[oc]) / np.sqrt(bn1_v[oc] + 1e-5) * bn1_w[oc] + bn1_b[oc]
                act0[oc, p] = gelu(val)
                
        # 3. Conv1d Block1: in_ch=64, out_ch=128, ksize=5, stride=1, pad=2
        # Output length is 28.
        act1 = np.zeros((128, 28), dtype=np.float32)
        for oc in range(128):
            for p in range(28):
                acc = 0.0
                in_start = p * 1 - 2
                for ic in range(64):
                    for k in range(5):
                        pos = in_start + k
                        if 0 <= pos < 28:
                            acc += act0[ic, pos] * bb3_W[oc, ic, k]
                act1[oc, p] = acc
                
        # 4. BN4 + GELU
        for oc in range(128):
            for p in range(28):
                val = act1[oc, p]
                val = (val - bn4_m[oc]) / np.sqrt(bn4_v[oc] + 1e-5) * bn4_w[oc] + bn4_b[oc]
                act1[oc, p] = gelu(val)
                
        # 5. MaxPool1D (stride 2) in-place:
        # Each channel of act1 (28 floats) is pooled to 14 floats.
        # But in memory it resides in the same 28-float row.
        # So physical stride between rows is 28.
        # If bugged=True, the next layer reads with stride 14.
        # If bugged=False, the next layer reads with stride 28.
        act1_pooled = np.zeros((128, 28), dtype=np.float32)
        for oc in range(128):
            for p in range(14):
                a = act1[oc, 2 * p]
                b = act1[oc, 2 * p + 1]
                act1_pooled[oc, p] = max(a, b)
            # act1_pooled[oc, 14:28] remains 0
            
        # 6. Conv1d Block2: in_ch=128, out_ch=128, ksize=3, stride=1, pad=1
        # Output length is 14.
        act2 = np.zeros((128, 14), dtype=np.float32)
        
        # Flattened buffer representation
        act1_flat = act1_pooled.flatten() # 128 * 28 = 3584 elements
        stride_val = 14 if bugged else 28
        
        for oc in range(128):
            for p in range(14):
                acc = 0.0
                in_start = p * 1 - 1
                for ic in range(128):
                    for k in range(3):
                        pos = in_start + k
                        if 0 <= pos < 14: # boundary check on active len (14)
                            # Access flat buffer
                            flat_idx = ic * stride_val + pos
                            acc += act1_flat[flat_idx] * bb7_W[oc, ic, k]
                act2[oc, p] = acc
                
        # 7. BN8 + GELU
        for oc in range(128):
            for p in range(14):
                val = act2[oc, p]
                val = (val - bn8_m[oc]) / np.sqrt(bn8_v[oc] + 1e-5) * bn8_w[oc] + bn8_b[oc]
                act2[oc, p] = gelu(val)
                
        # 8. GAP
        features = np.zeros(128, dtype=np.float32)
        for oc in range(128):
            features[oc] = np.mean(act2[oc])
            
        # 9. StandardScaler + SVM
        fs = scaler.transform(features.reshape(1, -1))[0]
        scores = svm.coef_ @ fs + svm.intercept_
        p = int(svm.classes_[np.argmax(scores)])
        
        ok = (p == labels[i])
        if ok:
            correct += 1
        predictions.append(p)
        
    return correct, predictions

print("Running bugged C-sim (stride 14)...")
c_bug, p_bug = run_sim(bugged=True)
print(f"  Accuracy: {c_bug}/10")
print(f"  Predictions: {p_bug}")

print("\nRunning fixed C-sim (stride 28)...")
c_fix, p_fix = run_sim(bugged=False)
print(f"  Accuracy: {c_fix}/10")
print(f"  Predictions: {p_fix}")
