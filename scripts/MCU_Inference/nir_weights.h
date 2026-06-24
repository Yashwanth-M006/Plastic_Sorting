/*
 * nir_weights.h
 * =============
 * Extern declarations for all quantised CNN + SVM weight arrays.
 *
 * The actual definitions live in nir_weights.c, which you generate
 * on your training machine by running:
 *
 *   python generate_weights_c.py
 *
 * That script reads nir_backbone_weights.bin, svm_weights.hex,
 * svm_biases.hex, scaler_mean.hex, scaler_std.hex and emits
 * nir_weights.c with all arrays in C28x-compatible format.
 *
 * Array layout (matches PyTorch weight dump in v6 training script)
 * ─────────────────────────────────────────────────────────────────
 *   g_bb0_weight        int8_t  [64 * 1 * 25]       backbone.0.weight  Q8
 *   g_bb1_weight        int16_t [64]                 backbone.1.weight  Q15
 *   g_bb1_bias          int16_t [64]                 backbone.1.bias    Q15
 *   g_bb1_running_mean  int16_t [64]                 backbone.1.rm      Q15
 *   g_bb1_running_var   int16_t [64]                 backbone.1.rv      Q15
 *   g_bb3_weight        int8_t  [128 * 64 * 5]       backbone.3.weight  Q8
 *   g_bb4_weight        int16_t [128]                backbone.4.weight  Q15
 *   g_bb4_bias          int16_t [128]                backbone.4.bias    Q15
 *   g_bb4_running_mean  int16_t [128]                backbone.4.rm      Q15
 *   g_bb4_running_var   int16_t [128]                backbone.4.rv      Q15
 *   g_bb7_weight        int8_t  [128 * 128 * 3]      backbone.7.weight  Q8
 *   g_bb8_weight        int16_t [128]                backbone.8.weight  Q15
 *   g_bb8_bias          int16_t [128]                backbone.8.bias    Q15
 *   g_bb8_running_mean  int16_t [128]                backbone.8.rm      Q15
 *   g_bb8_running_var   int16_t [128]                backbone.8.rv      Q15
 *   g_svm_weights       int16_t [NUM_CLASSES * SVM_DIM]  Q15
 *   g_svm_bias          int16_t [NUM_CLASSES]             Q15
 *   g_scaler_mean       int16_t [CNN_FEAT_DIM]            Q15
 *   g_scaler_std        int16_t [CNN_FEAT_DIM]            Q15
 *
 * Place all of these in flash (.const section) — they are read-only.
 */

#ifndef NIR_WEIGHTS_H
#define NIR_WEIGHTS_H

#include <stdint.h>
#include "nir_model.h"

/* ── Backbone stem: Conv1d(1→64, k=25) ─────────────────────────────────── */
extern const int8_t  g_bb0_weight       [64 * 1 * 25];

/* ── BatchNorm after stem (64 channels) ─────────────────────────────────── */
extern const int16_t g_bb1_weight       [64];
extern const int16_t g_bb1_bias         [64];
extern const int16_t g_bb1_running_mean [64];
extern const int16_t g_bb1_running_var  [64];

/* ── Block 1: Conv1d(64→128, k=5) ───────────────────────────────────────── */
extern const int8_t  g_bb3_weight       [128 * 64 * 5];

/* ── BatchNorm after block 1 (128 channels) ─────────────────────────────── */
extern const int16_t g_bb4_weight       [128];
extern const int16_t g_bb4_bias         [128];
extern const int16_t g_bb4_running_mean [128];
extern const int16_t g_bb4_running_var  [128];

/* ── Block 2: Conv1d(128→128, k=3) ──────────────────────────────────────── */
extern const int8_t  g_bb7_weight       [128 * 128 * 3];

/* ── BatchNorm after block 2 (128 channels) ─────────────────────────────── */
extern const int16_t g_bb8_weight       [128];
extern const int16_t g_bb8_bias         [128];
extern const int16_t g_bb8_running_mean [128];
extern const int16_t g_bb8_running_var  [128];

/* ── SVM (LinearSVC, 5 classes × 128 features) ───────────────────────────── */
extern const int16_t g_svm_weights      [NUM_CLASSES * SVM_DIM];
extern const int16_t g_svm_bias         [NUM_CLASSES];

/* ── StandardScaler (128-dim) ────────────────────────────────────────────── */
extern const int16_t g_scaler_mean      [CNN_FEAT_DIM];
extern const int16_t g_scaler_std       [CNN_FEAT_DIM];

#endif /* NIR_WEIGHTS_H */
