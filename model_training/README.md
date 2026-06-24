# Model Training and Weights

This directory holds the training scripts, exported PyTorch models, and trained parameters for the CNN backbone and SVM classifier.

## Folders
- `nir_v6/`: Contains intermediate scaled arrays, binary weights (`.bin`, `.hex`), pickled SVM pipeline, and the PyTorch CNN Backbone model (`.pt`) for the fully connected/CNN approach.
- `n_7/`: Contains alternative models, PCA parameter scales, and `.hex` exported parameters for quantized testing or alternate model architectures.

## Usage
These files are primarily loaded by the scripts in the `scripts/` folder to convert weights into C-compatible header files (`.h`), which are then imported by the C2000 firmware. 
You can use standard Python libraries (`torch`, `joblib`, `pickle`, `numpy`) to load the `.pt` and `.pkl` files and perform inference or re-training.
