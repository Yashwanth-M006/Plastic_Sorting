/*
 * nir_inference_main.c
 * ====================
 * TI F28379D (C28x) — NIR Hybrid CNN+SVM Inference over SCI-B UART
 *
 * Protocol
 * --------
 *   Host → MCU : 0xAA  +  224 × float32 LE  (897 bytes total)
 *   MCU  → Host: 0xBB  +  uint8 class (1–5)  (2 bytes total)
 *
 * Build requirements
 * ------------------
 *   • CCS project targeting F28379D (CPU1)
 *   • Add this file + nir_model.h + nir_weights.c to project
 *   • SCI-B pins:  GPIO19 = SCIRXDB,  GPIO18 = SCITXDB
 *     (these are the XDS110-connected UART lines on the F28379D LaunchPad)
 *   • FPU enabled in linker + compiler (-mfpu=fpu32 or equivalent)
 *
 * Memory budget (flash)
 * ---------------------
 *   backbone weights (bin):  ~93K params × ~1.2 bytes avg  ≈  112 KB
 *   SVM weights (128×5 int16):  1280 bytes
 *   Activations (stack):  224×4 + 64×4 + 14×128×4 ≈ ~9 KB  → fits in SARAM
 */

#include "F28x_Project.h"   /* TI DriverLib / driverlib headers */
#include <string.h>
#include <math.h>
#include "nir_model.h"
#include "nir_weights.h"    /* generated: extern arrays for all quantised weights */

/* ── Protocol ────────────────────────────────────────────────────────────── */
#define START_BYTE  0xAAu
#define ACK_BYTE    0xBBu

/* ── Derived dimensions ───────────────────────────────────────────────────── */
/* Stem: Conv1d(1→64, k=25, stride=8, pad=12): output length = (224+24-25)/8+1 = 28 */
#define STEM_OUT_LEN   28u
/* Block1 MaxPool2→ 14 */
#define BLK1_OUT_LEN   14u
/* Block2 + AdaptiveAvgPool1d(1) → 1 → Flatten → 128 */

/* ── Static activation buffers (SARAM — no heap needed) ─────────────────── */
#pragma DATA_SECTION(g_input,   ".ebss")
#pragma DATA_SECTION(g_act0,    ".ebss")
#pragma DATA_SECTION(g_act1,    ".ebss")
#pragma DATA_SECTION(g_act2,    ".ebss")
#pragma DATA_SECTION(g_features,".ebss")

static float g_input   [NUM_CHANNELS];           /* received SNV spectrum   */
static float g_act0    [64][STEM_OUT_LEN];        /* after stem BN+GELU      */
static float g_act1    [128][BLK1_OUT_LEN];       /* after block1 pool+GELU  */
static float g_act2    [CNN_FEAT_DIM][1];         /* after block2 GELU       */
static float g_features[CNN_FEAT_DIM];            /* after GAP + flatten     */

/* ── Forward declarations ────────────────────────────────────────────────── */
static void     sci_init(void);
static uint16_t sci_rx_byte(void);
static void     sci_tx_byte(uint16_t byte);
static void     receive_sample(void);
static void     run_cnn(void);
static uint8_t  run_svm(void);
static void     apply_bn(float *inout, uint32_t len, uint32_t layer_idx);
static float    gelu_approx(float x);
static void     conv1d_stride(const float *in,  uint32_t in_len,
                               const int8_t *W,  float    W_scale,
                               float        *out, uint32_t out_len,
                               uint32_t in_ch,   uint32_t out_ch,
                               uint32_t kernel,  uint32_t stride,
                               uint32_t padding);
static void     maxpool1d_2(float *inout, uint32_t in_len, uint32_t channels);

/* ═══════════════════════════════════════════════════════════════════════════
 * MAIN
 * ═══════════════════════════════════════════════════════════════════════════ */
void main(void)
{
    InitSysCtrl();
    InitGpio();
    DINT;
    InitPieCtrl();
    IER = 0x0000;
    IFR = 0x0000;
    InitPieVectTable();
    EINT;
    ERTM;

    sci_init();

    /* Infinite inference loop */
    for (;;)
    {
        receive_sample();       /* blocks until 897 bytes received */
        run_cnn();              /* backbone → g_features[128]      */
        uint8_t pred = run_svm();  /* scaler → SVM → class 1–5    */

        sci_tx_byte(ACK_BYTE);
        sci_tx_byte((uint16_t)pred);
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * SCI-B INITIALISATION  (XDS110 UART on LaunchPad: GPIO18/19)
 * ═══════════════════════════════════════════════════════════════════════════ */
static void sci_init(void)
{
    EALLOW;

    /* GPIO mux: GPIO18 = SCITXDB (mux=2), GPIO19 = SCIRXDB (mux=2) */
    GpioCtrlRegs.GPAMUX2.bit.GPIO18 = 2;
    GpioCtrlRegs.GPAMUX2.bit.GPIO19 = 2;
    GpioCtrlRegs.GPADIR.bit.GPIO18  = 1;   /* TX = output */
    GpioCtrlRegs.GPADIR.bit.GPIO19  = 0;   /* RX = input  */
    GpioCtrlRegs.GPAPUD.bit.GPIO18  = 0;   /* pull-up on TX */
    GpioCtrlRegs.GPAPUD.bit.GPIO19  = 0;   /* pull-up on RX */
    EDIS;

    /* Release SCI-B from reset */
    ScibRegs.SCICCR.all  = 0x0007;     /* 8-bit, no parity, 1 stop, async */
    ScibRegs.SCICTL1.all = 0x0003;     /* TX+RX enable */
    ScibRegs.SCICTL2.all = 0x0000;     /* no interrupts */

    /*
     * Baud = LSPCLK / (BRR+1) / 8
     * LSPCLK = 50 MHz (SYSCLK=200 MHz, /4)
     * BRR for 115200: 50e6 / (115200 * 8) - 1 = 53.25 → 53
     */
    ScibRegs.SCIHBAUD.all = 0x0000;
    ScibRegs.SCILBAUD.all = 53;

    ScibRegs.SCICTL1.all = 0x0023;     /* SW reset release */
}

/* ── Blocking byte receive (polls RXRDY) ─────────────────────────────────── */
static uint16_t sci_rx_byte(void)
{
    while (ScibRegs.SCIRXST.bit.RXRDY == 0) { }
    return ScibRegs.SCIRXBUF.all & 0x00FFu;
}

/* ── Blocking byte transmit ──────────────────────────────────────────────── */
static void sci_tx_byte(uint16_t byte)
{
    while (ScibRegs.SCICTL2.bit.TXRDY == 0) { }
    ScibRegs.SCITXBUF.all = byte & 0x00FFu;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * RECEIVE SAMPLE
 * Wait for 0xAA start byte, then read 224×4 = 896 bytes into g_input[].
 * ═══════════════════════════════════════════════════════════════════════════ */
static void receive_sample(void)
{
    uint16_t b;

    /* Sync: wait for start byte, discard everything else */
    do {
        b = sci_rx_byte();
    } while (b != START_BYTE);

    /* Read 224 float32s, little-endian */
    uint8_t raw[4];
    uint32_t i;
    for (i = 0; i < NUM_CHANNELS; i++)
    {
        raw[0] = (uint8_t)sci_rx_byte();
        raw[1] = (uint8_t)sci_rx_byte();
        raw[2] = (uint8_t)sci_rx_byte();
        raw[3] = (uint8_t)sci_rx_byte();
        /* Reinterpret bytes as IEEE-754 float (C28x is LE for float) */
        memcpy(&g_input[i], raw, 4);
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * GELU APPROXIMATION  (tanh form — accurate to <0.01% for |x|<4)
 * GELU(x) ≈ 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))
 * ═══════════════════════════════════════════════════════════════════════════ */
#define GELU_C1   0.7978845608f   /* sqrt(2/pi) */
#define GELU_C2   0.044715f

static float gelu_approx(float x)
{
    float x3  = x * x * x;
    float arg = GELU_C1 * (x + GELU_C2 * x3);
    return 0.5f * x * (1.0f + tanhf(arg));
}

/* ═══════════════════════════════════════════════════════════════════════════
 * BATCH-NORM (inference: fused into scale+shift)
 * y = (x - running_mean) / sqrt(running_var + eps) * weight + bias
 *
 * layer_idx: 0 = backbone.1 (64 ch), 1 = backbone.4 (128 ch), 2 = backbone.8 (128 ch)
 * Weights are in Q15; we dequantise on the fly.
 * ═══════════════════════════════════════════════════════════════════════════ */

/* Scale constants (from nir_model.h) for BN dequantisation */
static const float kBnWScale[3]   = { SCALE_BACKBONE_1_WEIGHT,
                                      SCALE_BACKBONE_4_WEIGHT,
                                      SCALE_BACKBONE_8_WEIGHT };
static const float kBnBScale[3]   = { SCALE_BACKBONE_1_BIAS,
                                      SCALE_BACKBONE_4_BIAS,
                                      SCALE_BACKBONE_8_BIAS };
static const float kBnMuScale[3]  = { SCALE_BACKBONE_1_RUNNING_MEAN,
                                      SCALE_BACKBONE_4_RUNNING_MEAN,
                                      SCALE_BACKBONE_8_RUNNING_MEAN };
static const float kBnVarScale[3] = { SCALE_BACKBONE_1_RUNNING_VAR,
                                      SCALE_BACKBONE_4_RUNNING_VAR,
                                      SCALE_BACKBONE_8_RUNNING_VAR };

static void apply_bn_channel(float *val, uint32_t ch, uint32_t layer_idx,
                              const int16_t *bn_w, const int16_t *bn_b,
                              const int16_t *bn_mu, const int16_t *bn_var)
{
    float eps    = 1e-5f;
    float w      = bn_w [ch] * kBnWScale  [layer_idx];
    float b      = bn_b [ch] * kBnBScale  [layer_idx];
    float mu     = bn_mu[ch] * kBnMuScale [layer_idx];
    float var    = bn_var[ch]* kBnVarScale [layer_idx];
    float inv_std = 1.0f / sqrtf(var + eps);
    *val = (*val - mu) * inv_std * w + b;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * CONV1D with stride (general — used for stem with stride=8)
 *
 *   in      : [in_ch × in_len]  (row-major: in[ch * in_len + pos])
 *   W       : Q8  [out_ch × in_ch × kernel]  (PyTorch layout)
 *   out     : [out_ch × out_len]
 * ═══════════════════════════════════════════════════════════════════════════ */
static void conv1d_Q8(const float   *in,      /* [in_ch × in_len]        */
                      uint32_t       in_len,
                      const int8_t  *W,        /* [out_ch × in_ch × ksize]*/
                      float          W_scale,
                      float         *out,      /* [out_ch × out_len]      */
                      uint32_t       out_len,
                      uint32_t       in_ch,
                      uint32_t       out_ch,
                      uint32_t       ksize,
                      uint32_t       stride,
                      uint32_t       padding)
{
    uint32_t oc, ic, p, k;
    for (oc = 0; oc < out_ch; oc++)
    {
        for (p = 0; p < out_len; p++)
        {
            float acc = 0.0f;
            int32_t in_start = (int32_t)(p * stride) - (int32_t)padding;
            for (ic = 0; ic < in_ch; ic++)
            {
                for (k = 0; k < ksize; k++)
                {
                    int32_t in_pos = in_start + (int32_t)k;
                    if (in_pos >= 0 && in_pos < (int32_t)in_len)
                    {
                        float in_val = in[ic * in_len + in_pos];
                        float w_val  = W[(oc * in_ch + ic) * ksize + k] * W_scale;
                        acc += in_val * w_val;
                    }
                }
            }
            out[oc * out_len + p] = acc;
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * MAXPOOL1D stride=2 (in-place, per channel)
 * ═══════════════════════════════════════════════════════════════════════════ */
static void maxpool1d_2(float *buf, uint32_t in_len, uint32_t channels)
{
    uint32_t out_len = in_len / 2;
    uint32_t ch, p;
    for (ch = 0; ch < channels; ch++)
    {
        float *row = buf + ch * in_len;
        for (p = 0; p < out_len; p++)
        {
            float a = row[p * 2];
            float b = row[p * 2 + 1];
            row[p]  = (a > b) ? a : b;  /* write result back in-place */
        }
        /* Zero out the upper half so we don't accidentally read it */
        memset(row + out_len, 0, out_len * sizeof(float));
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * RUN CNN BACKBONE  →  populates g_features[CNN_FEAT_DIM]
 *
 * Layer sequence (mirrors UltraFastCNN.backbone):
 *   [0] Conv1d(1→64, k=25, stride=8, pad=12) → 28     Q8
 *   [1] BN1d(64)                                        Q15
 *       GELU
 *   [3] Conv1d(64→128, k=5, pad=2)            → 28     Q8
 *   [4] BN1d(128)                                       Q15
 *       GELU
 *       MaxPool1d(2)                           → 14
 *   [7] Conv1d(128→128, k=3, pad=1)            → 14     Q8
 *   [8] BN1d(128)                                       Q15
 *       GELU
 *       AdaptiveAvgPool1d(1) → mean over 14
 *       Flatten → g_features[128]
 * ═══════════════════════════════════════════════════════════════════════════ */
static void run_cnn(void)
{
    uint32_t ch, p;

    /* ── Stem: Conv1d(1→64, k=25, stride=8, pad=12) ─────────────────────── */
    /* g_input is [1 × 224], treat as flat array since in_ch=1 */
    conv1d_Q8(g_input, NUM_CHANNELS,
              g_bb0_weight, SCALE_BACKBONE_0_WEIGHT,
              (float*)g_act0, STEM_OUT_LEN,
              1, 64, 25, 8, 12);

    /* BN + GELU on act0 [64 × 28] */
    for (ch = 0; ch < 64; ch++)
        for (p = 0; p < STEM_OUT_LEN; p++)
        {
            float *v = &g_act0[ch][p];
            apply_bn_channel(v, ch, 0,
                             g_bb1_weight, g_bb1_bias,
                             g_bb1_running_mean, g_bb1_running_var);
            *v = gelu_approx(*v);
        }

    /* ── Block 1: Conv1d(64→128, k=5, pad=2) ────────────────────────────── */
    conv1d_Q8((float*)g_act0, STEM_OUT_LEN,
              g_bb3_weight, SCALE_BACKBONE_3_WEIGHT,
              (float*)g_act1, STEM_OUT_LEN,   /* still 28 before pool */
              64, 128, 5, 1, 2);

    /* BN + GELU on act1 [128 × 28] */
    for (ch = 0; ch < 128; ch++)
        for (p = 0; p < STEM_OUT_LEN; p++)
        {
            float *v = &g_act1[ch][p];
            apply_bn_channel(v, ch, 1,
                             g_bb4_weight, g_bb4_bias,
                             g_bb4_running_mean, g_bb4_running_var);
            *v = gelu_approx(*v);
        }

    /* MaxPool1d(2): 28 → 14 */
    maxpool1d_2((float*)g_act1, STEM_OUT_LEN, 128);

    /* ── Block 2: Conv1d(128→128, k=3, pad=1) ───────────────────────────── */
    conv1d_Q8((float*)g_act1, BLK1_OUT_LEN,
              g_bb7_weight, SCALE_BACKBONE_7_WEIGHT,
              (float*)g_act2, BLK1_OUT_LEN,
              128, CNN_FEAT_DIM, 3, 1, 1);

    /* BN + GELU on act2 [128 × 14] */
    for (ch = 0; ch < CNN_FEAT_DIM; ch++)
        for (p = 0; p < BLK1_OUT_LEN; p++)
        {
            float *v = &g_act2[ch][p];
            apply_bn_channel(v, ch, 2,
                             g_bb8_weight, g_bb8_bias,
                             g_bb8_running_mean, g_bb8_running_var);
            *v = gelu_approx(*v);
        }

    /* ── AdaptiveAvgPool1d(1): mean over spatial dim → [128 × 1] ────────── */
    float inv_len = 1.0f / (float)BLK1_OUT_LEN;
    for (ch = 0; ch < CNN_FEAT_DIM; ch++)
    {
        float s = 0.0f;
        for (p = 0; p < BLK1_OUT_LEN; p++)
            s += g_act2[ch][p];
        g_features[ch] = s * inv_len;
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * RUN SVM
 *   1. StandardScaler: x = (x - mean) / std
 *   2. LinearSVC: score_c = dot(W_c, x) + b_c  for c in 0..4
 *   3. argmax → class 1–5
 *
 * All weights Q15 — dequantised with their respective scales.
 * ═══════════════════════════════════════════════════════════════════════════ */
static uint8_t run_svm(void)
{
    uint32_t d, c;

    /* StandardScaler (Q15 mean and std) */
    float scaled[CNN_FEAT_DIM];
    for (d = 0; d < CNN_FEAT_DIM; d++)
    {
        float mu  = g_scaler_mean [d] * SCALER_MEAN_SCALE;
        float std = g_scaler_std  [d] * SCALER_STD_SCALE;
        if (std < 1e-8f) std = 1e-8f;
        scaled[d] = (g_features[d] - mu) / std;
    }

    /* LinearSVC: score[c] = W[c,:] · scaled + b[c] */
    float best_score = -1e30f;
    uint8_t best_c   = 0;

    for (c = 0; c < NUM_CLASSES; c++)
    {
        float score = g_svm_bias[c] * SVM_B_SCALE;
        for (d = 0; d < SVM_DIM; d++)
            score += g_svm_weights[c * SVM_DIM + d] * SVM_W_SCALE * scaled[d];

        if (score > best_score)
        {
            best_score = score;
            best_c     = (uint8_t)c;
        }
    }

    return best_c + 1u;   /* return class label 1–5 */
}
