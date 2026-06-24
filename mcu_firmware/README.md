# MCU Firmware (C2000)

This folder contains the C projects intended to be compiled and flashed to the Texas Instruments C2000 Microcontroller (TMS320F28379D).

## Projects
- `Preprocessing/`: A dedicated project purely used to benchmark the C implementation of Dark Current Subtraction and SNV Normalization.
- `MCU_Firmware/`: The fully integrated single-core inference pipeline. It runs the entire CNN backbone + SVM classifier and outputs predictions via UART.
- `CPU1_MCU_Optimized/`: The dual-core optimized version of the firmware (CPU1 side). CPU1 handles data acquisition, preprocessing, UART output, and IPC (Inter-Processor Communication).
- `CPU2_MCU_Optimized/`: The dual-core optimized version of the firmware (CPU2 side). CPU2 handles heavy CNN/SVM inference tasks.

## Build Instructions
1. Open Code Composer Studio (CCS).
2. Go to **File -> Import -> C/C++ -> CCS Projects**.
3. Point the directory to any of the subfolders (e.g. `MCU_Firmware`).
4. Ensure the `ti-cgt-c2000` compiler is selected.
5. Hit **Build**.
6. Flash via XDS110 to the C2000 controlCARD or LaunchPad.
