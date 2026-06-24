# Python Scripts & MCU Inference Verification

This directory contains utility scripts to verify preprocessing, compare outputs, generate test vectors, and export PyTorch models to C-compatible header arrays.

## Files
- `compare.py`: Utility script to compare binary array outputs against expected floating-point values.
- `verify_preprocessing.py`: Script to validate the SNV normalization and Dark Current subtraction math.
- `MCU_Inference/`: Contains comprehensive Python toolset used for:
  - Converting `.pt` / `.pkl` models to C-Headers (`generate_weights_c.py`, `generate_head_weights.py`).
  - End-to-end simulations on PC (`e2e_test.py`, `sim_c_inference.py`).
  - Diagnostic and debugging scripts (`debug_model.py`, `feature_debug.py`).
  
## How to Run
Ensure you have `numpy`, `pandas`, `scikit-learn`, `torch`, and `matplotlib` installed.
```bash
# Example: Simulating the inference pipeline locally
python MCU_Inference/sim_c_inference.py
```
