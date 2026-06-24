# Plastic Sorting using NIR Spectroscopy & Machine Learning on C2000 Microcontroller

This repository contains all the code, firmware, models, and datasets for the Plastic Sorting Project. The project uses Near-Infrared (NIR) Spectroscopy data to classify plastics into five categories using a CNN Backbone and an SVM Classifier, deployed on a Texas Instruments C2000 Microcontroller (TMS320F28379D).

## Repository Structure

- `datasets/`: Contains the raw and preprocessed NIR spectrum CSV datasets (train, test, dark current, wavelength data).
- `docs/`: Contains the comprehensive Project Report (PDF/Markdown) and timing optimization plans.
- `mcu_firmware/`: Contains Code Composer Studio (CCS) projects for the TMS320F28379D microcontroller (preprocessing, single-core inference, dual-core optimized).
- `model_training/`: Contains pre-trained PyTorch models, weights, scalers, and pickled pipelines used in the project.
- `scripts/`: Python utility scripts for verification, C header generation, PC-based simulation, and testing.

## Getting Started

### 1. Clone the Repository
```bash
git clone <repository_url>
cd plastic_sorting_repo
```

### 2. Environment Setup (for Python scripts)
It's recommended to create a Python virtual environment to run the scripts.
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install numpy pandas scikit-learn torch matplotlib pyserial
```

### 3. Running Firmware
To run the firmware on a TI C2000 MCU:
1. Open Code Composer Studio (CCS).
2. Go to **File -> Import -> C/C++ -> CCS Projects**.
3. Browse to `mcu_firmware/` and select the desired project (e.g., `CPU1_MCU_Optimized` or `MCU_Firmware`).
4. Build the project and flash it to the board using a compatible XDS110 debug probe.
5. Open a Serial Terminal at 115200 baud to view the inference output.

### 4. Running PC Scripts
You can simulate the inference on your PC or verify preprocessing:
```bash
cd scripts
python verify_preprocessing.py
python MCU_Inference/sim_c_inference.py
```

Please refer to the `docs/Project_Report.pdf` for an in-depth understanding of the architecture, data pipeline, and performance metrics.
