"""
inspect_v6.py
=============
Inspects the nir_v6 output directory to understand:
1. The SVM pipeline structure (sklearn pickle)
2. The scale map (class label mapping)
3. What the CNN backbone outputs vs what the SVM expects
4. Why we get class 4 with the current head weights
5. The correct inference path
"""
import sys
sys.path.insert(0, ".")

import torch
import numpy as np
import pickle

V6_DIR = r"E:\CSIR\medley\nir_v6"

print("=" * 70)
print("PART 1: SVM PIPELINE from nir_svm_pipeline (1).pkl")
print("=" * 70)

with open(f"{V6_DIR}/nir_svm_pipeline (1).pkl", "rb") as f:
    pipeline = pickle.load(f)

print(f"Pipeline type: {type(pipeline)}")
print(f"Pipeline steps:" if hasattr(pipeline, 'steps') else "")

if hasattr(pipeline, 'steps'):
    for name, step in pipeline.steps:
        print(f"  Step '{name}': {type(step).__name__}")
        if hasattr(step, 'mean_'):
            print(f"    mean_  shape={step.mean_.shape}  min={step.mean_.min():.4f}  max={step.mean_.max():.4f}")
        if hasattr(step, 'scale_'):
            print(f"    scale_ shape={step.scale_.shape}  min={step.scale_.min():.6f}  max={step.scale_.max():.4f}")
        if hasattr(step, 'coef_'):
            print(f"    coef_  shape={step.coef_.shape}  min={step.coef_.min():.4f}  max={step.coef_.max():.4f}")
        if hasattr(step, 'intercept_'):
            print(f"    intercept_ shape={step.intercept_.shape}")
            print(f"    intercept_ values: {step.intercept_}")
        if hasattr(step, 'classes_'):
            print(f"    classes_: {step.classes_}")
else:
    print(f"  Object: {pipeline}")
    if hasattr(pipeline, '__dict__'):
        for k, v in pipeline.__dict__.items():
            print(f"    {k}: {type(v)} = {v if not hasattr(v, 'shape') else v.shape}")

print()
print("=" * 70)
print("PART 2: SCALE MAP from nir_scale_map (1).npy")
print("=" * 70)
scale_map = np.load(f"{V6_DIR}/nir_scale_map (1).npy", allow_pickle=True)
print(f"Scale map type: {type(scale_map)}")
print(f"Scale map: {scale_map}")

print()
print("=" * 70)
print("PART 3: CNN BACKBONE (the .pt is ONLY the feature extractor)")
print("=" * 70)
model_data = torch.load(f"{V6_DIR}/nir_cnn_backbone (1).pt", map_location="cpu", weights_only=False)
print(f"Type: {type(model_data)}")
if isinstance(model_data, dict):
    print("Keys:", list(model_data.keys()))
    sd = model_data.get("cnn_state", model_data)
    print("CNN state keys:")
    for k, v in sd.items():
        arr = v.float().numpy() if isinstance(v, torch.Tensor) else np.array(v)
        if hasattr(arr, 'shape') and arr.ndim > 0:
            print(f"  {k:40s}  shape={str(list(arr.shape)):20s}  min={arr.min():.4f}  max={arr.max():.4f}")

print()
print("=" * 70)
print("PART 4: HEX FILE VALIDATION (what's in the actual weight files)")
print("=" * 70)

def read_hex_file(path):
    vals = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = int(line, 16)
            if raw > 32767:
                raw -= 65536
            vals.append(raw)
    return np.array(vals, dtype=np.int16)

svm_b_int16 = read_hex_file(f"{V6_DIR}/svm_biases.hex")
svm_w_int16 = read_hex_file(f"{V6_DIR}/svm_weights.hex")
sc_mu_int16  = read_hex_file(f"{V6_DIR}/scaler_mean.hex")
sc_std_int16 = read_hex_file(f"{V6_DIR}/scaler_std.hex")

print(f"svm_biases.hex:  {svm_b_int16.shape}  values: {svm_b_int16}")
print(f"svm_weights.hex: {svm_w_int16.shape}")
print(f"scaler_mean.hex: {sc_mu_int16.shape}  first5: {sc_mu_int16[:5]}")
print(f"scaler_std.hex:  {sc_std_int16.shape}  first5: {sc_std_int16[:5]}")

print()
print("=" * 70)
print("PART 5: REVERSE-ENGINEER QUANTISATION SCALES")
print("=" * 70)

# Get the true SVM parameters from sklearn
if hasattr(pipeline, 'steps'):
    scaler = dict(pipeline.steps).get('scaler', None) or dict(pipeline.steps).get('standardscaler', None)
    svm    = dict(pipeline.steps).get('svm', None) or dict(pipeline.steps).get('linearsvc', None)
    # Try all step names
    for name, step in pipeline.steps:
        if hasattr(step, 'mean_'):
            scaler = step
        if hasattr(step, 'coef_'):
            svm = step

    if scaler and svm:
        true_mean  = scaler.mean_
        true_scale = scaler.scale_
        true_coef  = svm.coef_   # shape [n_classes, n_features]
        true_bias  = svm.intercept_

        print(f"True scaler mean  shape={true_mean.shape}  min={true_mean.min():.4f}  max={true_mean.max():.4f}")
        print(f"True scaler scale shape={true_scale.shape}  min={true_scale.min():.6f}  max={true_scale.max():.4f}")
        print(f"True SVM coef     shape={true_coef.shape}  min={true_coef.min():.4f}  max={true_coef.max():.4f}")
        print(f"True SVM bias     shape={true_bias.shape}  values: {true_bias}")
        print(f"True SVM classes: {svm.classes_}")

        print()
        print("Quantisation verification (hex -> float via nir_model.h scales):")
        SVM_B_SCALE       = 0.00007590
        SCALER_MEAN_SCALE = 0.00006030
        SCALER_STD_SCALE  = 0.00004164
        SVM_W_SCALE       = 0.00001627

        recon_bias = svm_b_int16.astype(float) * SVM_B_SCALE
        recon_mean = sc_mu_int16.astype(float) * SCALER_MEAN_SCALE
        recon_std  = sc_std_int16.astype(float) * SCALER_STD_SCALE

        print(f"\nReconstructed SVM bias:  {recon_bias}")
        print(f"True SVM bias:           {true_bias}")
        bias_ok = np.allclose(recon_bias, true_bias, atol=1e-3)
        print(f"  Match: {bias_ok}")

        print(f"\nReconstructed scaler mean (first 5): {recon_mean[:5]}")
        print(f"True scaler mean (first 5):          {true_mean[:5]}")
        mean_ok = np.allclose(recon_mean, true_mean, atol=1e-3)
        print(f"  Match: {mean_ok}")

        print(f"\nReconstructed scaler std (first 5): {recon_std[:5]}")
        print(f"True scaler std (first 5):          {true_scale[:5]}")
        std_ok = np.allclose(recon_std, true_scale, atol=1e-3)
        print(f"  Match: {std_ok}")

        print()
        print("=" * 70)
        print("PART 6: IDEAL SCALES FOR C CODE")
        print("=" * 70)
        ideal_bias_scale = np.abs(true_bias).max() / 32767.0
        ideal_mean_scale = np.abs(true_mean).max() / 32767.0
        ideal_std_scale  = np.abs(true_scale).max() / 32767.0
        ideal_w_scale    = np.abs(true_coef).max() / 32767.0

        print(f"  Ideal SVM_B_SCALE       = {ideal_bias_scale:.8f}f  (used: {SVM_B_SCALE:.8f})")
        print(f"  Ideal SCALER_MEAN_SCALE = {ideal_mean_scale:.8f}f  (used: {SCALER_MEAN_SCALE:.8f})")
        print(f"  Ideal SCALER_STD_SCALE  = {ideal_std_scale:.8f}f  (used: {SCALER_STD_SCALE:.8f})")
        print(f"  Ideal SVM_W_SCALE       = {ideal_w_scale:.8f}f  (used: {SVM_W_SCALE:.8f})")

        print()
        print("=" * 70)
        print("PART 7: WHAT ARE THE CNN BACKBONE FEATURES (no head.1/head.3)?")
        print("=" * 70)
        print("The BACKBONE produces a 128-dim feature vector via GAP.")
        print("This feature vector feeds into the SKLEARN SVM pipeline, NOT head.1/head.3.")
        print("head.1 and head.3 are the TRAINING HEAD used during supervised pre-training only.")
        print("head.1 and head.3 should NOT be used for inference — use the SVM pipeline!")
        print()
        print("The correct inference chain is:")
        print("  raw_spectrum -> dark_ref_subtraction -> SNV -> CNN_backbone -> GAP -> 128-dim features")
        print("  -> StandardScaler.transform(features) -> LinearSVC.predict(scaled_features)")
        print()
        print("The C code must implement:")
        print("  1. CNN backbone (already done correctly with BN float32 fix)")
        print("  2. StandardScaler: scaled[d] = (features[d] - mean[d]) / scale[d]")
        print("  3. LinearSVC argmax: score[c] = bias[c] + dot(coef[c], scaled)")
        print("  4. Return argmax(scores) + 1  (or map via scale_map)")
