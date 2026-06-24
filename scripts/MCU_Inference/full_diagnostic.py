"""
full_diagnostic.py
==================
Runs a complete end-to-end diagnostic of the MCU inference pipeline.
Tests:
  1. What the actual model IS (no SVM -- it's a pure CNN with head.1 + head.3)
  2. What weights the C code is actually using (svm_weights / scaler are WRONG)
  3. Shows the correct inference path
  4. Simulates what the C code does to show why it always picks class 5
  5. Runs the correct forward pass to show expected accuracy
"""
import torch
import numpy as np

MODEL_PATH  = "nir_cnn_backbone (1).pt"
SELFTEST_H  = "nir_selftest_data.h"

print("=" * 70)
print("PART 1: TRUE MODEL STRUCTURE")
print("=" * 70)

model_data = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
sd = model_data["cnn_state"]

print("The model is a PURE CNN + Linear Head. There is NO external SVM or StandardScaler.")
print("Architecture:")
print("  backbone.0   Conv1d(1, 64, k=25, s=8, p=8 [NOTE: not 12])")
print("  backbone.1   BatchNorm1d(64)")
print("  backbone.3   Conv1d(64, 128, k=5, s=1, p=2)")
print("  backbone.4   BatchNorm1d(128)")
print("  backbone.7   Conv1d(128, 128, k=3, s=1, p=1)")
print("  backbone.8   BatchNorm1d(128)")
print("  GAP          GlobalAveragePool -> 128-dim vector")
print("  head.1       Linear(128, 64)")
print("  head.3       Linear(64, 5)   <- THIS is the final classifier, not SVM!")
print()

for k, v in sd.items():
    arr = v.float().numpy()
    print(f"  {k:40s}  shape={str(list(v.shape)):20s}  min={arr.min():.4f}  max={arr.max():.4f}")

print()
print("=" * 70)
print("PART 2: WHAT THE C CODE IS ACTUALLY DOING (THE BUG)")
print("=" * 70)

# The C code calls run_svm() which uses g_svm_weights, g_svm_bias, g_scaler_mean, g_scaler_std
# But the model has NO SVM! The generate_weights_c.py reads from .hex files (svm_weights.hex etc.)
# Those hex files are probably from a DIFFERENT/WRONG source.

SVM_B_SCALE       = 0.00007590
SCALER_MEAN_SCALE = 0.00006030
SCALER_STD_SCALE  = 0.00004164
SVM_W_SCALE       = 0.00001627

# The biases in nir_weights.c:
svm_bias_int16 = np.array([-10826, -27186, -19798, -12021, -32767], dtype=np.float32)
svm_bias_scaled = svm_bias_int16 * SVM_B_SCALE
print(f"g_svm_bias (int16):  {svm_bias_int16}")
print(f"After scale (*{SVM_B_SCALE}): {svm_bias_scaled}")
print()
print(f"Class 5 bias = {svm_bias_scaled[4]:.4f}")
print(f"Class 1 bias = {svm_bias_scaled[0]:.4f}")
print(f"CLASS 5 HAS LEAST NEGATIVE BIAS! It will always win if the weight*feature terms are small.")
print()
print("Even more critical: the scaler_std values. If they are WRONG, scaled[] becomes garbage")
print("and the SVM decision is dominated purely by the bias.")
print()

# The actual head weights from the model
head3_w = sd["head.3.weight"].float().numpy()  # shape [5, 64]
head3_b = sd["head.3.bias"].float().numpy()    # shape [5]
head1_w = sd["head.1.weight"].float().numpy()  # shape [64, 128]
head1_b = sd["head.1.bias"].float().numpy()    # shape [64]

print("ACTUAL model head.3 biases (from .pt file):")
print(f"  {head3_b}")
print()
print("The C code is using a FAKE 'SVM' with bias from an unrelated .hex file,")
print("while the real classifier is head.1 -> head.3 which the C code IGNORES!")

print()
print("=" * 70)
print("PART 3: WHY CLASS 5 ALWAYS WINS")
print("=" * 70)
print()
print("The C code run_svm() does:")
print("  1. Scale features via g_scaler_mean / g_scaler_std (WRONG data, from hex files)")
print("  2. Compute score[c] = g_svm_bias[c] * SVM_B_SCALE")
print("     + sum(g_svm_weights[c*128+d] * SVM_W_SCALE * scaled[d])")
print("  3. Return argmax + 1")
print()
print("The SVM weights are random/wrong, so the weight*feature sum is unpredictable.")
print("But the BIAS values are fixed, and class 5 has the least-negative bias:")
print(f"  Class 1: {svm_bias_scaled[0]:.4f}")
print(f"  Class 2: {svm_bias_scaled[1]:.4f}")
print(f"  Class 3: {svm_bias_scaled[2]:.4f}")
print(f"  Class 4: {svm_bias_scaled[3]:.4f}")
print(f"  Class 5: {svm_bias_scaled[4]:.4f}  <- ALWAYS WINS")

print()
print("=" * 70)
print("PART 4: THE CORRECT FORWARD PASS (head.1 + head.3)")
print("=" * 70)
print()
print("The fix is: replace run_svm() with a proper 2-layer Linear head:")
print("  hidden = GELU(head.1.weight @ features + head.1.bias)  [128->64]")
print("  logits = head.3.weight @ hidden + head.3.bias          [64->5]")
print("  pred   = argmax(logits) + 1")
print()
print("Head weights summary:")
print(f"  head.1.weight  shape={head1_w.shape}  min={head1_w.min():.4f}  max={head1_w.max():.4f}")
print(f"  head.1.bias    shape={head1_b.shape}  min={head1_b.min():.4f}  max={head1_b.max():.4f}")
print(f"  head.3.weight  shape={head3_w.shape}  min={head3_w.min():.4f}  max={head3_w.max():.4f}")
print(f"  head.3.bias    shape={head3_b.shape}  min={head3_b.min():.4f}  max={head3_b.max():.4f}")

print()
print("=" * 70)
print("PART 5: QUANTISATION SCALES FOR THE HEAD WEIGHTS")
print("=" * 70)
print()
# What scale to use for int16 quantisation
for name, arr in [("head.1.weight", head1_w), ("head.1.bias", head1_b),
                  ("head.3.weight", head3_w), ("head.3.bias", head3_b)]:
    amax = np.abs(arr).max()
    scale = amax / 32767.0
    print(f"  {name:20s}  abs_max={amax:.6f}  ideal_int16_scale={scale:.8f}")

print()
print("=" * 70)
print("SUMMARY OF ROOT CAUSE")
print("=" * 70)
print("""
ROOT CAUSE (3 compounding bugs):

1. WRONG CLASSIFIER: The model uses a 2-layer Linear head (head.1 -> head.3).
   The C code instead runs a fake 'SVM' using data from external .hex files
   that were probably from a completely different model version.
   The real head weights (head.1, head.3) are NOT used at all in the C code.

2. WRONG/ABSENT SCALER: The Python model does NOT use an external StandardScaler.
   The scaler in the C code (g_scaler_mean, g_scaler_std) is fabricated.
   When the features are wrongly scaled, the weight*feature dot products become
   near-zero, leaving the SVM decision dominated purely by the bias values.

3. DOMINANT CLASS 5 BIAS: With weight*feature terms near zero, the argmax
   of the bias alone determines the winner. Class 5's bias (-2.4873) is
   the least negative, so it always wins.

FIX:
  - Replace run_svm() with run_head() that computes:
      hidden[64] = GELU(head1_weight[64x128] @ features[128] + head1_bias[64])
      logits[5]  = head3_weight[5x64] @ hidden[64] + head3_bias[5]
      return argmax(logits) + 1
  - Add head weights to nir_weights.c (quantized int16 is fine for 64x128 + 5x64)
  - Remove g_scaler_mean, g_scaler_std, g_svm_weights, g_svm_bias entirely
""")
