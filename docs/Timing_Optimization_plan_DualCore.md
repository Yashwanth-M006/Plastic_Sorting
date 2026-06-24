# Goal Description

The objective is to drastically reduce the MCU inference time from the current 400 ms (0.4 seconds) down to a range closer to PC-level performance (single-digit milliseconds), by fully utilizing the unique hardware accelerators and multicore architecture of the Texas Instruments TMS320F28379D.

## User Review Required

> [!IMPORTANT]
> Achieving sub-millisecond inference on a 200 MHz MCU compared to a multi-GHz PC is mathematically extremely challenging. However, we can likely reduce the latency from 400ms down to the **5 - 15 ms range** by throwing all hardware resources at it. Please review the proposed parallelization and memory strategies below and approve if you are comfortable adding this level of complexity (IPC, DMA, Dual-Core) to the firmware.

## Open Questions

1. **Memory Allocation**: The F28379D has roughly 82K words (164 KB) of RAM. Our weights total around 90K words. Can we dedicate almost all available MCU RAM exclusively to this inference task, or do other system processes (like control loops or communications) require significant RAM?
2. **Model Accuracy Trade-offs**: Are you open to replacing the mathematically heavy GELU activation (`tanh`) with a computationally cheaper alternative like ReLU, or an aggressive fast-math approximation, to save thousands of clock cycles?

## Proposed Changes

To maximize utilization, we will transform the strictly sequential C-code pipeline into a highly parallelized, hardware-accelerated architecture.

### 1. Dual-Core Parallelization (CPU1 & CPU2)
The F28379D has two identical 200 MHz C28x cores. We will split the most computationally expensive layers (Block 1 and Block 2 Conv1D) across both cores.
- **Data Sharing**: We will place the input and intermediate activation buffers (`g_act0`, `g_act1`) in the **Global Shared RAM (GSRAM)**, accessible by both CPUs.
- **Workload Split**: When executing `conv1d_Q16_split3`, CPU1 will compute output channels `0` to `63`, while CPU2 concurrently computes output channels `64` to `127`.
- **Synchronization**: We will use Inter-Processor Communication (IPC) flags to sync the start and completion of each layer.
- **Expected Speedup**: ~1.9x

### 2. Control Law Accelerator (CLA) Co-processing
Each CPU features an independent 32-bit floating-point math coprocessor (CLA1 and CLA2), giving us 4 effective processing units.
- **Offloading**: We will offload the Batch Normalization and GELU activation loops to the CLAs. While the CPUs compute the convolutions for the *next* block, the CLAs will normalize and activate the *previous* block's output in parallel.
- **Expected Speedup**: ~1.3x

### 3. Flash Wait-State Mitigation via DMA & RAM
Currently, the `int16_t` weights are stored in Flash memory. At 200 MHz, reading from Flash incurs 3 wait states (taking 4 clock cycles per read). RAM operates at 0 wait states (1 cycle).
- **RAM Mapping**: We will map the smaller weights (Stem layer, SVM weights, Batch Norm parameters) directly into fast Local Shared RAM (LSRAM) at boot.
- **DMA Ping-Pong Buffering**: For the large Conv1D weights (which won't entirely fit in RAM), we will configure the **DMA controller** to pre-fetch chunks of weights from Flash into a small RAM buffer in the background. The CPU will compute using Buffer A (in fast RAM) while the DMA loads Buffer B from Flash, completely hiding the Flash latency.
- **Expected Speedup**: ~3.0x - 4.0x

### 4. DSP Library & FPU Optimization
We will replace our standard C nested loops with highly optimized assembly macros.
- **TI DSPLib**: We will utilize the C28x FPU DSP Library (`C28x_FPU_Library`), specifically the optimized FIR filter and vector dot-product functions (`mpy_DP_mac`). These use the `MACF32` instruction to perform a parallel multiply and accumulate in a single clock cycle.
- **TMU Acceleration**: We will enable the Trigonometric Math Unit (TMU) to accelerate the polynomial division and transcendental functions required by the GELU activation.

## Verification Plan

### Automated Tests
- The Python simulation (`sim_c_inference.py`) will remain our ground truth for mathematical equivalence.
- The dual-core execution will be simulated and verified against the current single-core output to ensure race conditions or IPC memory sharing do not corrupt the features.

### Manual Verification
- We will deploy the split architecture to both CPU1 and CPU2 via Code Composer Studio.
- The UART telemetry will be updated to output independent timing profiles for CPU1, CPU2, and total pipeline latency, ensuring we hit our target speed bracket.
