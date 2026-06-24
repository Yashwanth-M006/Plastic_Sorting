"""
quantized_verify.py - Quantize the model weights and evaluate accuracy on the entire train_preprocessed_snv.csv dataset locally.
"""
import torch, torch.nn as nn, numpy as np, joblib, pandas as pd

# Files
V6_DIR = r"E:\CSIR\medley\nir_v6"
MODEL_PT = f"{V6_DIR}/nir_cnn_backbone (1).pt"
SVM_PKL = r"E:\CSIR\medley\MCU_Inference\nir_svm_pipeline_new.pkl"
CSV_PATH = r"E:\CSIR\medley\MCU_Inference\train_preprocessed_snv.csv"

# 1. Load PyTorch Backbone
pt = torch.load(MODEL_PT, map_location="cpu", weights_only=False)
sd = pt["cnn_state"]

# Quantization Scales (from nir_model.h)
SCALE_BB0_W = 0.00857819
SCALE_BB3_W = 0.00558934
SCALE_BB7_W = 0.00380100

# Function to quantize floats to int16 (matching C generation)
def quantize_to_i16(w, scale):
    q = np.round(w / scale).astype(np.int32)
    q = np.clip(q, -32767, 32767).astype(np.int16)
    return q

# Quantize Conv Weights
bb0_W_q = quantize_to_i16(sd["backbone.0.weight"].numpy(), SCALE_BB0_W)
bb3_W_q = quantize_to_i16(sd["backbone.3.weight"].numpy(), SCALE_BB3_W)
bb7_W_q = quantize_to_i16(sd["backbone.7.weight"].numpy(), SCALE_BB7_W)

# Dequantized weights for simulation (simulates C math: float = (float)wq * W_scale)
bb0_W = bb0_W_q.astype(np.float32) * SCALE_BB0_W
bb3_W = bb3_W_q.astype(np.float32) * SCALE_BB3_W
bb7_W = bb7_W_q.astype(np.float32) * SCALE_BB7_W

# BatchNorm parameters (kept in float32)
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

# 2. Load SVM + Scaler
new_pipe = joblib.load(SVM_PKL)
svm, scaler = new_pipe["svm"], new_pipe["scaler"]

# SVM Quantization Scales
SVM_W_SCALE = 0.00006215
SVM_B_SCALE = 0.00009003
SCALER_MEAN_SCALE = 0.00033705
SCALER_STD_SCALE = 0.00002776

# Quantize SVM weights/biases and Scaler mean/std to simulate C execution
svm_w_q = quantize_to_i16(svm.coef_, SVM_W_SCALE)
svm_b_q = quantize_to_i16(svm.intercept_, SVM_B_SCALE)
scaler_mean_q = quantize_to_i16(scaler.mean_, SCALER_MEAN_SCALE)
scaler_std_q = quantize_to_i16(scaler.scale_, SCALER_STD_SCALE)

# Dequantized classifier params
svm_w = svm_w_q.astype(np.float32) * SVM_W_SCALE
svm_b = svm_b_q.astype(np.float32) * SVM_B_SCALE
scaler_mean = scaler_mean_q.astype(np.float32) * SCALER_MEAN_SCALE
scaler_std = scaler_std_q.astype(np.float32) * SCALER_STD_SCALE

# 3. Load dataset
print("Loading dataset...")
df = pd.read_csv(CSV_PATH)
if "class_id" in df.columns:
    y_true = df["class_id"].values.astype(np.int32)
    X = df.drop(columns=["class_id"]).values.astype(np.float32)
else:
    y_true = df.iloc[:, 0].values.astype(np.int32)
    X = df.iloc[:, 1:].values.astype(np.float32)

print(f"Loaded {len(X)} samples.")

# GELU activation
def gelu(x):
    x3 = x * x * x
    return 0.5 * x * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x3)))

# Run locally mimicking the fixed C code
print("Running quantized local inference...")
correct = 0
BATCH_SIZE = 128
predictions = []

for i in range(len(X)):
    in_val = X[i]
    
    # 1. Stem Conv: in_ch=1, out_ch=64, ksize=25, stride=8, pad=12
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
            
    # 4. BN4 + GELU
    for oc in range(128):
        for p in range(28):
            val = act1[oc, p]
            val = (val - bn4_m[oc]) / np.sqrt(bn4_v[oc] + 1e-5) * bn4_w[oc] + bn4_b[oc]
            act1[oc, p] = gelu(val)
            
    # 5. MaxPool1D (in-place stride 2, physical stride 28)
    act1_pooled = np.zeros((128, 28), dtype=np.float32)
    for oc in range(128):
        for p in range(14):
            act1_pooled[oc, p] = max(act1[oc, 2 * p], act1[oc, 2 * p + 1])
            
    # 6. Conv1d Block2: in_ch=128, out_ch=128, ksize=3, stride=1, pad=1
    act2 = np.zeros((128, 14), dtype=np.float32)
    act1_flat = act1_pooled.flatten() # stride is 28
    
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
        
    # 9. StandardScaler + SVM (quantized)
    scaled = np.zeros(128, dtype=np.float32)
    for d in range(128):
        std_val = scaler_std[d]
        if std_val < 1e-8:
            std_val = 1e-8
        scaled[d] = (features[d] - scaler_mean[d]) / std_val
        
    best_score = -3.4e38
    best_c = 0
    for c in range(5):
        score = svm_b[c]
        for d in range(128):
            score += svm_w[c, d] * scaled[d]
        if score > best_score:
            best_score = score
            best_c = c
            
    pred = best_c + 1 # 1-indexed
    predictions.append(pred)
    if pred == y_true[i]:
        correct += 1
        
    if (i + 1) % 500 == 0 or (i + 1) == len(X):
        print(f"Processed {i+1}/{len(X)} samples... Accuracy: {100 * correct / (i+1):.2f}%")

final_accuracy = 100.0 * correct / len(X)
print(f"\nFinal Quantized Model Accuracy on train dataset: {final_accuracy:.4f}%")
