"""
debug_model.py
==============
Inspects the full saved model structure, prints all top-level keys,
SVM coefficients, biases, and scaler parameters to identify
the class 5 prediction bias issue.
"""
import torch
import numpy as np

MODEL_PATH = "nir_cnn_backbone (1).pt"

print("=" * 70)
print("Loading model:", MODEL_PATH)
model = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
print("Type:", type(model))

# ── Case 1: Full model object ─────────────────────────────────────────
if hasattr(model, "state_dict"):
    print("\n[Case 1] Has state_dict()")
    sd = model.state_dict()
    print("State dict keys:")
    for k, v in sd.items():
        if hasattr(v, 'shape'):
            arr = v.float().numpy()
            print(f"  {k:50s}  shape={list(v.shape)}  min={arr.min():.4f}  max={arr.max():.4f}")
        else:
            print(f"  {k}: {v}")

# ── Case 2: Plain dict ────────────────────────────────────────────────
elif isinstance(model, dict):
    print("\n[Case 2] Dict with top-level keys:")
    for k, v in model.items():
        if hasattr(v, 'shape'):
            arr = v.float().numpy() if isinstance(v, torch.Tensor) else np.array(v, dtype=float)
            print(f"  {k:50s}  shape={list(arr.shape)}  min={arr.min():.4f}  max={arr.max():.4f}")
        elif hasattr(v, '__len__'):
            print(f"  {k:50s}  len={len(v)}")
        else:
            print(f"  {k}: {type(v)} = {v}")

    # Dig into cnn_state if present
    if "cnn_state" in model:
        sd = model["cnn_state"]
        print("\n  [cnn_state] sub-keys:")
        for k, v in sd.items():
            if hasattr(v, 'shape'):
                arr = v.float().numpy()
                print(f"    {k:48s}  shape={list(v.shape)}  min={arr.min():.4f}  max={arr.max():.4f}")

print("\n" + "=" * 70)
print("SVM / Scaler investigation")
print("=" * 70)

# Try to extract SVM data from whatever structure
svm_coef = None
svm_intercept = None
scaler_mean = None
scaler_std = None

def try_get(obj, *keys):
    for k in keys:
        if isinstance(obj, dict) and k in obj:
            v = obj[k]
            if isinstance(v, torch.Tensor):
                return v.float().numpy()
            return np.array(v, dtype=float)
    return None

top = model if isinstance(model, dict) else {}

svm_coef      = try_get(top, "svm_coef", "svm.coef_", "classifier.weight")
svm_intercept = try_get(top, "svm_intercept", "svm.intercept_", "classifier.bias", "svm_bias")
scaler_mean   = try_get(top, "scaler_mean", "scaler.mean_")
scaler_std    = try_get(top, "scaler_scale", "scaler.scale_", "scaler_std")

if svm_intercept is not None:
    print(f"\nSVM intercept (bias) raw: {svm_intercept.flatten()}")
    print(f"  shape={svm_intercept.shape}")
    print(f"  class 5 bias vs others: {svm_intercept.flatten()}")
else:
    print("SVM intercept NOT found in top-level dict!")

if svm_coef is not None:
    print(f"\nSVM coef shape: {svm_coef.shape}")
    print(f"  per-class L2 norm: {np.linalg.norm(svm_coef, axis=-1)}")
else:
    print("SVM coef NOT found!")

if scaler_mean is not None:
    print(f"\nScaler mean: shape={scaler_mean.shape}  min={scaler_mean.min():.4f}  max={scaler_mean.max():.4f}")
    print(f"  first 10: {scaler_mean.flatten()[:10]}")

if scaler_std is not None:
    print(f"\nScaler std:  shape={scaler_std.shape}  min={scaler_std.min():.6f}  max={scaler_std.max():.4f}")
    print(f"  first 10: {scaler_std.flatten()[:10]}")
    near_zero = np.sum(scaler_std.flatten() < 1e-6)
    print(f"  near-zero (<1e-6) count: {near_zero}")

print("\n" + "=" * 70)
print("Quantisation scale sanity check (from nir_model.h)")
print("=" * 70)

# These are the scales defined in nir_model.h
SVM_B_SCALE       = 0.00007590
SCALER_MEAN_SCALE = 0.00006030
SCALER_STD_SCALE  = 0.00004164

if svm_intercept is not None:
    # Check what the C code will compute vs raw float
    biases_int16 = np.array([-10826, -27186, -19798, -12021, -32767], dtype=np.float64)
    c_biases = biases_int16 * SVM_B_SCALE
    print(f"\n[C code] SVM biases after scale = {c_biases}")
    raw_biases = svm_intercept.flatten()
    print(f"[Python] SVM biases raw          = {raw_biases}")
    print(f"  Max int16 range = +/-32767. Scale = {SVM_B_SCALE:.8f}")
    expected_scale = raw_biases.max() / 32767.0
    print(f"  Ideal scale for max bias      = {expected_scale:.8f}")
    print(f"  Ratio (ideal/used) for bias   = {expected_scale / SVM_B_SCALE:.2f}x")

if scaler_mean is not None and scaler_std is not None:
    sc_mean_int16 = np.array([4970, 4608, 6769, 6107, 5221], dtype=np.float64)
    c_means = sc_mean_int16 * SCALER_MEAN_SCALE
    print(f"\n[C code] scaler means (first 5) = {c_means}")
    print(f"[Python] scaler means (first 5)  = {scaler_mean.flatten()[:5]}")
    expected_scale_m = scaler_mean.flatten().max() / 32767.0
    print(f"  Ideal scale for scaler_mean   = {expected_scale_m:.8f}")
    print(f"  Used SCALER_MEAN_SCALE        = {SCALER_MEAN_SCALE:.8f}")
    print(f"  Ratio (ideal/used)            = {expected_scale_m / SCALER_MEAN_SCALE:.2f}x")

    sc_std_int16 = np.array([5206, 6558, 5679, 4204, 6414], dtype=np.float64)
    c_stds = sc_std_int16 * SCALER_STD_SCALE
    print(f"\n[C code] scaler stds (first 5)  = {c_stds}")
    print(f"[Python] scaler stds (first 5)   = {scaler_std.flatten()[:5]}")
    expected_scale_s = scaler_std.flatten().max() / 32767.0
    print(f"  Ideal scale for scaler_std    = {expected_scale_s:.8f}")
    print(f"  Used SCALER_STD_SCALE         = {SCALER_STD_SCALE:.8f}")
    print(f"  Ratio (ideal/used)            = {expected_scale_s / SCALER_STD_SCALE:.2f}x")
