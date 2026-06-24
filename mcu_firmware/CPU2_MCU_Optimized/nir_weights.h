/*
 * nir_weights.h  (C28x / C2000Ware driverlib)
 * ============================================
 * g_bb3_weight and g_bb7_weight are physically split across flash sectors.
 * Each sub-array has its own named section placed in one 64KB flash bank.
 * The inference code reconstructs a contiguous pointer view via a wrapper.
 */

#ifndef NIR_WEIGHTS_H
#define NIR_WEIGHTS_H

#include "driverlib.h"
#include "device.h"
#include "nir_model.h"

/* Backbone stem: Conv1d(1->64, k=25)  Q8 as int16_t  -- .const  */
extern const int16_t g_bb0_weight       [1600];

/* BatchNorm after stem (64 ch)  Q15  -- .const */
extern const int16_t g_bb1_weight       [64];
extern const int16_t g_bb1_bias         [64];
extern const int16_t g_bb1_running_mean [64];
extern const int16_t g_bb1_running_var  [64];

/* Block1 Conv1d(64->128,k=5)  Q8 as int16_t
 * Split: lo[20480] in FLASHF,  hi[20480] in FLASHG               */
extern const int16_t g_bb3_weight_lo    [20480];  /* section .nir_bb3_lo */
extern const int16_t g_bb3_weight_hi    [20480];  /* section .nir_bb3_hi */

/* BatchNorm after block 1 (128 ch)  Q15  -- .const */
extern const int16_t g_bb4_weight       [128];
extern const int16_t g_bb4_bias         [128];
extern const int16_t g_bb4_running_mean [128];
extern const int16_t g_bb4_running_var  [128];

/* Block2 Conv1d(128->128,k=3)  Q8 as int16_t
 * Split: a[16384] in FLASHH, b[16384] in FLASHI, c[16384] in FLASHJ */
extern const int16_t g_bb7_weight_a     [16384];  /* section .nir_bb7_a */
extern const int16_t g_bb7_weight_b     [16384];  /* section .nir_bb7_b */
extern const int16_t g_bb7_weight_c     [16384];  /* section .nir_bb7_c */

/* BatchNorm after block 2 (128 ch)  Q15  -- .const */
extern const int16_t g_bb8_weight       [128];
extern const int16_t g_bb8_bias         [128];
extern const int16_t g_bb8_running_mean [128];
extern const int16_t g_bb8_running_var  [128];

/* SVM + scaler  Q15  -- .const */
extern const int16_t g_svm_weights      [640];
extern const int16_t g_svm_bias         [5];
extern const int16_t g_scaler_mean      [128];
extern const int16_t g_scaler_std       [128];

#endif /* NIR_WEIGHTS_H */