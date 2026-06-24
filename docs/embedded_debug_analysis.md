# Embedded Systems Debugging Analysis: Memory Stride & Alignment Mismatch

This report documents the root-cause analysis of the classification issue where the on-device inference systematically predicts **Class 5** for all samples, despite the offline Python models and mathematical validations being fully correct.

---

## 1. Executive Summary & Core Bug

The primary cause of the failure is a **memory stride mismatch** in the second convolution layer (`conv1d_Q16_split3`). 

* **The Problem**: The intermediate activation buffer `g_act1` is allocated as a 2D array with a physical channel width of `STEM_OUT_LEN = 28` floats. After max-pooling, the active length of each channel is reduced to `BLK1_OUT_LEN = 14` floats, leaving the remaining 14 elements as zeros. The array is modified *in-place*. 
* **The Bug**: When calling `conv1d_Q16_split3`, the active length of `14` is passed as the input length parameter `in_len`. The convolution function indexes the input buffer using:
  $$\text{flat\_index} = ic \times in\_len + pos$$
  Because `in_len = 14` is passed, the function assumes that consecutive channels start every **14 floats**. However, in memory, the rows of `g_act1` still start every **28 floats**!
* **The Impact**: 
  * For $ic = 0$ (channel 0): reads indices `0..13` (correctly maps to `g_act1[0][0..13]`).
  * For $ic = 1$ (channel 1): reads indices `14..27` (incorrectly maps to the zeroed-out second half of `g_act1[0]`).
  * For $ic = 2$ (channel 2): reads indices `28..41` (incorrectly maps to `g_act1[1][0..13]`, which is channel 1's data).
  * For $ic = 3$ (channel 3): reads indices `42..55` (incorrectly maps to the zeroed-out second half of `g_act1[1]`).
  
  This scrambles all input channels to the second convolution layer: **half the channels are read as absolute zero, and the other half read the wrong channel's data**. The resulting feature vector is garbage, causing the StandardScaler and SVM to default to the class with the least negative bias (**Class 5**).

---

## 2. Detailed Technical Breakdown

### A. Memory Layout of `g_act1`
In C, a multidimensional array:
```c
static float g_act1[128][28];
```
is stored contiguously in row-major order:
$$\text{Addr}(g\_act1[ic][pos]) = \text{base} + (ic \times 28) + pos$$

### B. In-place Max Pooling
The max-pooling function updates the first half of each row in-place and clears the rest:
```c
static void maxpool1d_2(float *buf, uint32_t in_len, uint32_t channels) {
    uint32_t out_len = in_len >> 1u, c, p;
    for (c = 0; c < channels; c++) {
        float *row = buf + c * in_len; // stride is still in_len (28)
        for (p = 0; p < out_len; p++) {
            row[p] = max(row[2p], row[2p+1]);
        }
        memset(row + out_len, 0, out_len * sizeof(float)); // set elements 14..27 to 0
    }
}
```
At the end of this function, channel $c$'s data occupies `g_act1[c][0..13]`. However, the physical memory stride between `g_act1[c]` and `g_act1[c+1]` remains **28 words**.

### C. Mismatched Indexing in `conv1d_Q16_split3`
Inside `conv1d_Q16_split3`, the indexing is:
```c
float in_val = in[ic * in_len + (uint32_t)in_pos];
```
When called as:
```c
conv1d_Q16_split3((float*)g_act1, BLK1_OUT_LEN, ...); // BLK1_OUT_LEN is 14
```
The stride calculation becomes $ic \times 14$. This is off by a factor of 2!

---

## 3. Other Potential Embedded Hardware/Toolchain Errors to Watch

While the memory stride mismatch is the direct mathematical cause of the bug, any embedded C28x developer must also verify these system-level factors:

### A. Word Addressing (16-bit Char/Int/Float sizes)
On the TI C28x DSP, the minimum addressable unit is a **16-bit word**.
* `sizeof(char) == 1` (but represents 16 bits)
* `sizeof(int16_t) == 1` (16 bits)
* `sizeof(float) == 2` (32 bits, occupies 2 words)
* `sizeof(double) == 2` (32 bits on F2837xD, same as float)

**Risk**: Using standard byte-count assumptions in `memcpy` or custom pointer arithmetic (e.g., adding raw byte offsets to pointers) will result in double-offsetting or reading out-of-bounds memory.
* *Status in main.c*: Safe. All pointer offsets are calculated in terms of the underlying type (e.g. `sizeof(float)` which correctly returns `2` words).

### B. stack Overflow & RAM Allocation Limits
The system uses significant global memory for model activations.
* The `.stack` is placed in `RAMM1` (1024 words / 2048 bytes).
* The stack contains local variables such as `float scaled[128]` (256 words / 512 bytes) inside `run_svm()`.
* If local allocations in stack-frame exceed 1024 words, it will silently corrupt the variables above/below the stack (often `RAMM0` or `.ebss`).

**Fix/Mitigation**: Make sure stack size is set to at least `0x400` in the linker properties, and keep stack-allocated local arrays minimal.

### C. Flash Wait States and Pipeline Alignment
Reading constants (weights) from flash memory at 200 MHz requires wait states. If the flash configuration registers do not match the system clock frequency:
* Flash reads will return corrupted or stale data.
* System may trigger an illegal instruction interrupt (`ILGMSG`).

**Fix/Mitigation**: Ensure `Device_init()` initializes the Flash controller and wait states before running the neural network.

---

## 4. Proposed Fixes

To fix the stride mismatch, we must decoupling the **active spatial length** (`in_len`) from the **physical channel memory stride** (`in_stride`).

### Fix A: Modify Convolution Signatures to support Stride
Update `conv1d_Q16_split3` (and optionally `conv1d_Q16_split`) to accept an explicit `in_stride` parameter:

```diff
 static void conv1d_Q16_split3(
-    const float    *in,       uint32_t in_len,
+    const float    *in,       uint32_t in_len, uint32_t in_stride,
     const int16_t  *W_a,      uint32_t a_elems,
     const int16_t  *W_b,      uint32_t b_elems,
     const int16_t  *W_c,      float W_scale,
     float          *out,      uint32_t out_len,
     uint32_t        in_ch,    uint32_t out_ch,
     uint32_t        ksize,    uint32_t stride, uint32_t padding)
 {
     ...
     for (ic = 0; ic < in_ch; ic++)
         for (k = 0; k < ksize; k++)
         {
             int32_t in_pos = in_start + (int32_t)k;
             if (in_pos >= 0 && in_pos < (int32_t)in_len)
             {
-                float    in_val = in[ic * in_len + (uint32_t)in_pos];
+                float    in_val = in[ic * in_stride + (uint32_t)in_pos];
                 ...
```

Then, call it in `run_cnn()` passing `STEM_OUT_LEN` (28) as the physical memory stride:
```diff
     conv1d_Q16_split3(
-        (float*)g_act1, BLK1_OUT_LEN,
+        (float*)g_act1, BLK1_OUT_LEN, STEM_OUT_LEN,
         g_bb7_weight_a, 16384u,
         g_bb7_weight_b, 16384u,
         g_bb7_weight_c,
         SCALE_BACKBONE_7_WEIGHT,
         (float*)g_act2, BLK1_OUT_LEN,
         128u, CNN_FEAT_DIM, 3u, 1u, 1u);
```

### Fix B: Copy MaxPool Output to a Packed Stride Buffer
Alternatively, we can copy the pooled values from `g_act1` (stride 28) into a packed temporary buffer of stride 14 before calling the convolution. However, this incurs additional memory copies and consumes more RAM. **Fix A is highly preferred as it has zero memory overhead and zero performance penalty.**
