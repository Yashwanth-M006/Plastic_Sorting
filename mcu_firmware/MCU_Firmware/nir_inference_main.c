/*
 * nir_inference_main.c
 * ====================
 * TI F28379D CPU1 -- NIR Hybrid CNN+SVM  SELF-TEST MODE
 *
 * Output channels:
 *   1. LED GPIO31  -- blinks predicted class per sample, final score
 *   2. SCI-A TX    -- GPIO42 -> XDS110 USB COM Port, 115200 baud
 *                     Computer receives and pretty-prints to its Serial Monitor
 *
 * Serial protocol (plain ASCII lines, \r\n terminated):
 *   "BOOT\r\n"
 *   "SAMPLE,<i>,<total>,<true>,<pred>,<OK|WRONG>\r\n"  (one per sample)
 *   "RESULT,<correct>,<total>,<accuracy_x10>\r\n"       (e.g. 90 = 9.0%)
 *   "CM,<r>,<c1>,<c2>,<c3>,<c4>,<c5>\r\n"              (5 rows, r=1..5)
 *   "DONE\r\n"
 *
 * LED sequence:
 *   3 slow blinks              = boot OK
 *   5 slow blinks              = engine ready
 *   Per sample: N fast blinks  = predicted class N (1-5)
 *   Final: K long blinks       = K correct out of 10 (repeats)
 */

#include "driverlib.h"
#include "device.h"
#include <string.h>
#include <math.h>
#include "nir_model.h"
#include "nir_weights.h"
#include "nir_bn_weights.h"    /* float32 BN params — fixes running_mean scale bug */
#include "nir_head_weights.h"   /* correct Linear head weights: head.1 + head.3       */
#include "nir_selftest_data.h"

/* ── CNN dimensions ───────────────────────────────────────────────── */
#define STEM_OUT_LEN   28u
#define BLK1_OUT_LEN   14u

/* ── SCI-A TX to XDS110 (USB COM Port) ────────────────────────────── */
#define LOG_SCI_BASE   SCIA_BASE

/* ── Activation buffers (.ebss -> RAMGS_NIR) ─────────────────────── */
#pragma DATA_SECTION(g_input,    ".ebss")
#pragma DATA_SECTION(g_act0,     ".ebss")
#pragma DATA_SECTION(g_act1,     ".ebss")
#pragma DATA_SECTION(g_act2,     ".ebss")
#pragma DATA_SECTION(g_features, ".ebss")

/* ── Execute hot code from RAM for zero wait states (200 MHz) ─────── */
#pragma CODE_SECTION(conv1d_stem, ".TI.ramfunc")
#pragma CODE_SECTION(conv1d_Q16_split, ".TI.ramfunc")
#pragma CODE_SECTION(conv1d_Q16_split3, ".TI.ramfunc")
#pragma CODE_SECTION(run_cnn, ".TI.ramfunc")
#pragma CODE_SECTION(apply_bn_ch, ".TI.ramfunc")
#pragma CODE_SECTION(gelu_approx, ".TI.ramfunc")

static float g_input   [NUM_CHANNELS];
static float g_act0    [64][STEM_OUT_LEN];
static float g_act1    [128][STEM_OUT_LEN];
static float g_act2    [CNN_FEAT_DIM][BLK1_OUT_LEN];
static float g_features[CNN_FEAT_DIM];

/* ── Forward declarations ─────────────────────────────────────────── */
static void    log_init(void);
static void    log_char(char c);
static void    log_str(const char *s);
static void    log_uint(uint32_t v);
static void    log_nl(void);
static void    log_float(float f);
static void    led_blink(uint32_t n, uint32_t on_us, uint32_t off_us);
static void    run_cnn(void);
static uint8_t run_svm(void);
static float   gelu_approx(float x);
static void    apply_bn_ch(float *v, uint16_t ch, uint16_t layer,
                            const int16_t *W, const int16_t *B,
                            const int16_t *MU, const int16_t *VAR);
                            /* W/B/MU/VAR unused — float BN used instead */
static void    maxpool1d_2(float *buf, uint32_t in_len, uint32_t ch);
static void    conv1d_Q16_split(
                   const float *in, uint32_t in_len,
                   const int16_t *W_lo, uint32_t lo_elems,
                   const int16_t *W_hi, float W_scale,
                   float *out, uint32_t out_len,
                   uint32_t in_ch, uint32_t out_ch,
                   uint32_t ksize, uint32_t stride, uint32_t padding);
static void    conv1d_Q16_split3(
                   const float *in, uint32_t in_len, uint32_t in_stride,
                   const int16_t *W_a, uint32_t a_elems,
                   const int16_t *W_b, uint32_t b_elems,
                   const int16_t *W_c, float W_scale,
                   float *out, uint32_t out_len,
                   uint32_t in_ch, uint32_t out_ch,
                   uint32_t ksize, uint32_t stride, uint32_t padding);
static void    conv1d_stem(
                   const float *in, const int16_t *W, float W_scale,
                   float *out, uint32_t in_len, uint32_t out_len,
                   uint32_t ksize, uint32_t stride, uint32_t padding);

/* ── LED macros ───────────────────────────────────────────────────── */
#define BLINK_SLOW(n)  led_blink((n), 400000u, 300000u)
#define BLINK_FAST(n)  led_blink((n), 100000u, 100000u)
#define BLINK_LONG(n)  led_blink((n), 600000u, 400000u)
#define PAUSE(us)      DEVICE_DELAY_US(us)

/* ═══════════════════════════════════════════════════════════════════
 * SCI-A TX logger — GPIO42 output only, 115200 baud
 * ═══════════════════════════════════════════════════════════════════ */
static void log_init(void)
{
    SysCtl_enablePeripheral(SYSCTL_PERIPH_CLK_SCIA);
    DEVICE_DELAY_US(100);

    GPIO_setPinConfig(GPIO_42_SCITXDA);
    GPIO_setDirectionMode(42, GPIO_DIR_MODE_OUT);
    GPIO_setPadConfig(42, GPIO_PIN_TYPE_STD);
    /* GPIO43 RX not used — TX only */

    SCI_performSoftwareReset(LOG_SCI_BASE);
    SCI_setConfig(LOG_SCI_BASE, DEVICE_LSPCLK_FREQ, 115200UL,
                  SCI_CONFIG_WLEN_8 | SCI_CONFIG_STOP_ONE | SCI_CONFIG_PAR_NONE);
    SCI_resetChannels(LOG_SCI_BASE);
    SCI_clearInterruptStatus(LOG_SCI_BASE, SCI_INT_RXFF | SCI_INT_TXFF);
    SCI_enableModule(LOG_SCI_BASE);
    SCI_performSoftwareReset(LOG_SCI_BASE);
}

static void log_char(char c)
{
    SCI_writeCharBlockingNonFIFO(LOG_SCI_BASE, (uint16_t)(uint8_t)c);
}

static void log_str(const char *s)
{
    while (*s) log_char(*s++);
}

static void log_uint(uint32_t v)
{
    char    buf[12];
    int8_t  i = 0;
    if (v == 0u) { log_char('0'); return; }
    while (v > 0u) { buf[i++] = (char)('0' + (v % 10u)); v /= 10u; }
    while (--i >= 0) log_char(buf[i]);
}

static void log_nl(void)
{
    log_char('\r');
    log_char('\n');
}

static void log_float(float f)
{
    if (f < 0.0f)
    {
        log_char('-');
        f = -f;
    }
    uint32_t integer_part = (uint32_t)f;
    float fractional_part = f - (float)integer_part;
    uint32_t fraction_val = (uint32_t)(fractional_part * 1000.0f + 0.5f);
    
    log_uint(integer_part);
    log_char('.');
    if (fraction_val < 100u) log_char('0');
    if (fraction_val < 10u)  log_char('0');
    log_uint(fraction_val);
}

/* ═══════════════════════════════════════════════════════════════════
 * LED helper — GPIO31 blue LED
 * ═══════════════════════════════════════════════════════════════════ */
static void led_blink(uint32_t n, uint32_t on_us, uint32_t off_us)
{
    uint32_t j;
    for (j = 0; j < n; j++)
    {
        GPIO_writePin(31, 1);
        DEVICE_DELAY_US(on_us);
        GPIO_writePin(31, 0);
        DEVICE_DELAY_US(off_us);
    }
}

/* ═══════════════════════════════════════════════════════════════════
 * MAIN
 * ═══════════════════════════════════════════════════════════════════ */
void main(void)
{
    Device_init();
    Device_initGPIO();

    GPIO_setPinConfig(GPIO_31_GPIO31);
    GPIO_setDirectionMode(31, GPIO_DIR_MODE_OUT);
    GPIO_writePin(31, 0);

    Interrupt_initModule();
    Interrupt_initVectorTable();
    log_init();

    // Initialize CPU Timer 0 for timing analysis
    CPUTimer_stopTimer(CPUTIMER0_BASE);
    CPUTimer_setPeriod(CPUTIMER0_BASE, 0xFFFFFFFFUL);
    CPUTimer_setPreScaler(CPUTIMER0_BASE, 199u); // 200 cycles per tick = 1 us at 200MHz
    CPUTimer_reloadTimerCounter(CPUTIMER0_BASE);

    /* Boot signal */
    BLINK_SLOW(3);
    log_str("BOOT"); log_nl();
    PAUSE(1000000u);

    BLINK_SLOW(5);
    log_str("START,");
    log_uint(SELFTEST_N_SAMPLES);
    log_char(',');
    log_uint(NUM_CLASSES);
    log_nl();
    PAUSE(500000u);

    /* ── Run all embedded samples ────────────────────────────────── */
    uint32_t correct = 0u;
    uint32_t confusion[6][6];
    uint32_t i, r, c;
    uint32_t total_time_us = 0u;
    memset(confusion, 0, sizeof(confusion));

    for (i = 0u; i < SELFTEST_N_SAMPLES; i++)
    {
        memcpy(g_input, g_selftest_X[i], NUM_CHANNELS * sizeof(float));
        
        CPUTimer_startTimer(CPUTIMER0_BASE);
        run_cnn();
        uint8_t pred  = run_svm();
        CPUTimer_stopTimer(CPUTIMER0_BASE);
        
        uint32_t elapsed_us = 0xFFFFFFFFUL - CPUTimer_getTimerCount(CPUTIMER0_BASE);
        CPUTimer_reloadTimerCounter(CPUTIMER0_BASE);
        total_time_us += elapsed_us;
        
        uint8_t truth = g_selftest_labels[i];
        uint8_t ok    = (pred == truth) ? 1u : 0u;
        if (ok) correct++;
        if (truth >= 1u && truth <= 5u && pred >= 1u && pred <= 5u)
            confusion[truth][pred]++;

        /* TX: "SAMPLE,<i+1>,<total>,<true>,<pred>,<OK|WRONG>,<time_us>us" */
        log_str("SAMPLE,");
        log_uint(i + 1u);  log_char(',');
        log_uint(SELFTEST_N_SAMPLES); log_char(',');
        log_uint(truth);   log_char(',');
        log_uint(pred);    log_char(',');
        log_str(ok ? "OK" : "WRONG");  log_char(',');
        log_uint(elapsed_us); log_str("us");
        log_nl();

        /* LED: N fast blinks = predicted class */
        PAUSE(300000u);
        BLINK_FAST(pred);
        PAUSE(700000u);
    }

    /* accuracy_x10: e.g. 9/10 = 90, send as integer to avoid floats */
    uint32_t acc_x10 = (correct * 1000u) / SELFTEST_N_SAMPLES;

    /* TX: "RESULT,<correct>,<total>,<acc_x10>" */
    log_str("RESULT,");
    log_uint(correct);      log_char(',');
    log_uint(SELFTEST_N_SAMPLES); log_char(',');
    log_uint(acc_x10);
    log_nl();

    /* TX: confusion matrix, one row per line */
    for (r = 1u; r <= 5u; r++)
    {
        log_str("CM,");
        log_uint(r);
        for (c = 1u; c <= 5u; c++)
        {
            log_char(',');
            log_uint(confusion[r][c]);
        }
        log_nl();
    }

    // Print timing and accuracy analysis
    uint32_t avg_time_us = total_time_us / SELFTEST_N_SAMPLES;
    float pct = ((float)correct * 100.0f) / (float)SELFTEST_N_SAMPLES;

    log_str("AVG_INFERENCE_TIME: ");
    log_uint(avg_time_us);
    log_str(" us");
    log_nl();

    log_str("ACCURACY: ");
    log_uint(correct);
    log_char('/');
    log_uint(SELFTEST_N_SAMPLES);
    log_str(" (");
    log_float(pct);
    log_str("%)");
    log_nl();

    log_str("DONE"); log_nl();

    /* Final LED score loop — K long blinks = K correct, repeats */
    for (;;)
    {
        PAUSE(2000000u);
        if (correct > 0u)
            BLINK_LONG(correct);
        else
            PAUSE(1000000u);
        PAUSE(3000000u);
    }
}

/* ═══════════════════════════════════════════════════════════════════
 * conv1d: two flash banks
 * ═══════════════════════════════════════════════════════════════════ */
static void conv1d_Q16_split(
    const float    *in,      uint32_t in_len,
    const int16_t  *W_lo,    uint32_t lo_elems,
    const int16_t  *W_hi,    float W_scale,
    float          *out,     uint32_t out_len,
    uint32_t        in_ch,   uint32_t out_ch,
    uint32_t        ksize,   uint32_t stride, uint32_t padding)
{
    uint32_t oc, ic, p, k;
    float w_local[320]; // 64 ch * 5 kernel = 320 max local cache size
    for (oc = 0; oc < out_ch; oc++)
    {
        for (ic = 0; ic < in_ch; ic++)
            for (k = 0; k < ksize; k++)
            {
                uint32_t widx = (oc * in_ch + ic) * ksize + k;
                int16_t  wq   = (widx < lo_elems) ? W_lo[widx]
                                                   : W_hi[widx - lo_elems];
                w_local[ic * ksize + k] = (float)wq * W_scale;
            }

        for (p = 0; p < out_len; p++)
        {
            float   acc      = 0.0f;
            int32_t in_start = (int32_t)(p * stride) - (int32_t)padding;
            for (ic = 0; ic < in_ch; ic++)
                for (k = 0; k < ksize; k++)
                {
                    int32_t in_pos = in_start + (int32_t)k;
                    if (in_pos >= 0 && in_pos < (int32_t)in_len)
                    {
                        acc += in[ic * in_len + (uint32_t)in_pos] * w_local[ic * ksize + k];
                    }
                }
            out[oc * out_len + p] = acc;
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════
 * conv1d: three flash banks
 * ═══════════════════════════════════════════════════════════════════ */
static void conv1d_Q16_split3(
    const float    *in,       uint32_t in_len,  uint32_t in_stride,
    const int16_t  *W_a,      uint32_t a_elems,
    const int16_t  *W_b,      uint32_t b_elems,
    const int16_t  *W_c,      float W_scale,
    float          *out,      uint32_t out_len,
    uint32_t        in_ch,    uint32_t out_ch,
    uint32_t        ksize,    uint32_t stride, uint32_t padding)
{
    uint32_t oc, ic, p, k;
    uint32_t ab_elems = a_elems + b_elems;
    float w_local[384]; // 128 ch * 3 kernel = 384 local cache size
    for (oc = 0; oc < out_ch; oc++)
    {
        for (ic = 0; ic < in_ch; ic++)
            for (k = 0; k < ksize; k++)
            {
                uint32_t widx = (oc * in_ch + ic) * ksize + k;
                int16_t  wq;
                if      (widx < a_elems)  wq = W_a[widx];
                else if (widx < ab_elems) wq = W_b[widx - a_elems];
                else                      wq = W_c[widx - ab_elems];
                w_local[ic * ksize + k] = (float)wq * W_scale;
            }

        for (p = 0; p < out_len; p++)
        {
            float   acc      = 0.0f;
            int32_t in_start = (int32_t)(p * stride) - (int32_t)padding;
            for (ic = 0; ic < in_ch; ic++)
                for (k = 0; k < ksize; k++)
                {
                    int32_t in_pos = in_start + (int32_t)k;
                    if (in_pos >= 0 && in_pos < (int32_t)in_len)
                    {
                        acc += in[ic * in_stride + (uint32_t)in_pos] * w_local[ic * ksize + k];
                    }
                }
            out[oc * out_len + p] = acc;
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════
 * Stem conv (1->64, k=25, stride=8, pad=12)
 * ═══════════════════════════════════════════════════════════════════ */
static void conv1d_stem(
    const float    *in,
    const int16_t  *W,      float W_scale,
    float          *out,
    uint32_t in_len, uint32_t out_len,
    uint32_t ksize,  uint32_t stride, uint32_t padding)
{
    uint32_t oc, p, k;
    float w_local[25]; // 1 ch * 25 kernel = 25 max cache size
    for (oc = 0; oc < 64u; oc++)
    {
        for (k = 0; k < ksize; k++)
            w_local[k] = (float)W[oc * ksize + k] * W_scale;

        for (p = 0; p < out_len; p++)
        {
            float   acc      = 0.0f;
            int32_t in_start = (int32_t)(p * stride) - (int32_t)padding;
            for (k = 0; k < ksize; k++)
            {
                int32_t pos = in_start + (int32_t)k;
                if (pos >= 0 && pos < (int32_t)in_len)
                    acc += in[(uint32_t)pos] * w_local[k];
            }
            out[oc * out_len + p] = acc;
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════
 * GELU approximation
 * ═══════════════════════════════════════════════════════════════════ */
#define GELU_C1 0.7978845608f
#define GELU_C2 0.044715f
static float gelu_approx(float x)
{
    float x3 = x * x * x;
    return 0.5f * x * (1.0f + tanhf(GELU_C1 * (x + GELU_C2 * x3)));
}

/* ═══════════════════════════════════════════════════════════════════
 * BatchNorm (inference fused)
 * ═══════════════════════════════════════════════════════════════════ */
/*
 * apply_bn_ch — now uses raw float32 BN params from nir_bn_weights.h.
 * The int16 BN arrays in nir_weights.c are no longer used for BN.
 * This fixes the running_mean scale underflow bug where bb1 running_mean
 * was quantized with a scale 70x too small, zeroing all mean values.
 *
 * W, B, MU, VAR params kept in signature for compatibility but now
 * ignored — the float arrays are indexed directly by layer and ch.
 */
static void apply_bn_ch(float *v, uint16_t ch, uint16_t layer,
                         const int16_t *W, const int16_t *B,
                         const int16_t *MU, const int16_t *VAR)
{
    float w, b, mu, var;
    (void)W; (void)B; (void)MU; (void)VAR;   /* suppress unused warnings */

    switch (layer)
    {
        case 0:  /* backbone.1 — after stem, 64 channels */
            w   = g_bn1_weight      [ch];
            b   = g_bn1_bias        [ch];
            mu  = g_bn1_running_mean[ch];
            var = g_bn1_running_var [ch];
            break;
        case 1:  /* backbone.4 — after block1, 128 channels */
            w   = g_bn4_weight      [ch];
            b   = g_bn4_bias        [ch];
            mu  = g_bn4_running_mean[ch];
            var = g_bn4_running_var [ch];
            break;
        case 2:  /* backbone.8 — after block2, 128 channels */
        default:
            w   = g_bn8_weight      [ch];
            b   = g_bn8_bias        [ch];
            mu  = g_bn8_running_mean[ch];
            var = g_bn8_running_var [ch];
            break;
    }

    *v = (*v - mu) * (1.0f / sqrtf(var + 1e-5f)) * w + b;
}

/* ═══════════════════════════════════════════════════════════════════
 * MaxPool1d stride=2, in-place
 * ═══════════════════════════════════════════════════════════════════ */
static void maxpool1d_2(float *buf, uint32_t in_len, uint32_t channels)
{
    uint32_t out_len = in_len >> 1u, c, p;
    for (c = 0; c < channels; c++)
    {
        float *row = buf + c * in_len;
        for (p = 0; p < out_len; p++)
        {
            float a = row[p*2u], b2 = row[p*2u+1u];
            row[p] = (a > b2) ? a : b2;
        }
        memset(row + out_len, 0, out_len * sizeof(float));
    }
}

/* ═══════════════════════════════════════════════════════════════════
 * CNN backbone
 * ═══════════════════════════════════════════════════════════════════ */
static void run_cnn(void)
{
    uint32_t ch, p;

    conv1d_stem(g_input, g_bb0_weight, SCALE_BACKBONE_0_WEIGHT,
                (float*)g_act0, NUM_CHANNELS, STEM_OUT_LEN, 25u, 8u, 12u);

    for (ch = 0; ch < 64u; ch++)
        for (p = 0; p < STEM_OUT_LEN; p++)
        {
            float *v = &g_act0[ch][p];
            apply_bn_ch(v, (uint16_t)ch, 0,
                        g_bb1_weight, g_bb1_bias,
                        g_bb1_running_mean, g_bb1_running_var);
            *v = gelu_approx(*v);
        }

    conv1d_Q16_split(
        (float*)g_act0, STEM_OUT_LEN,
        g_bb3_weight_lo, 20480u, g_bb3_weight_hi,
        SCALE_BACKBONE_3_WEIGHT,
        (float*)g_act1, STEM_OUT_LEN,
        64u, 128u, 5u, 1u, 2u);

    for (ch = 0; ch < 128u; ch++)
        for (p = 0; p < STEM_OUT_LEN; p++)
        {
            float *v = &g_act1[ch][p];
            apply_bn_ch(v, (uint16_t)ch, 1,
                        g_bb4_weight, g_bb4_bias,
                        g_bb4_running_mean, g_bb4_running_var);
            *v = gelu_approx(*v);
        }

    maxpool1d_2((float*)g_act1, STEM_OUT_LEN, 128u);

    conv1d_Q16_split3(
        (float*)g_act1, BLK1_OUT_LEN, STEM_OUT_LEN,
        g_bb7_weight_a, 16384u,
        g_bb7_weight_b, 16384u,
        g_bb7_weight_c,
        SCALE_BACKBONE_7_WEIGHT,
        (float*)g_act2, BLK1_OUT_LEN,
        128u, CNN_FEAT_DIM, 3u, 1u, 1u);

    for (ch = 0; ch < CNN_FEAT_DIM; ch++)
        for (p = 0; p < BLK1_OUT_LEN; p++)
        {
            float *v = &g_act2[ch][p];
            apply_bn_ch(v, (uint16_t)ch, 2,
                        g_bb8_weight, g_bb8_bias,
                        g_bb8_running_mean, g_bb8_running_var);
            *v = gelu_approx(*v);
        }

    float inv = 1.0f / (float)BLK1_OUT_LEN;
    for (ch = 0; ch < CNN_FEAT_DIM; ch++)
    {
        float s = 0.0f;
        for (p = 0; p < BLK1_OUT_LEN; p++) s += g_act2[ch][p];
        g_features[ch] = s * inv;
    }
}

/* ═══════════════════════════════════════════════════════════════════
 * run_svm: StandardScaler + LinearSVC argmax
 *
 * Uses the sklearn SVM pipeline from nir_v6 (nir_svm_pipeline.pkl).
 * Weights (g_svm_weights, g_svm_bias, g_scaler_mean, g_scaler_std) must
 * be generated from the SAME training run as the CNN backbone in
 * nir_cnn_backbone.pt. See root_cause_analysis.md for details.
 *
 * NOTE: nir_head_weights.h (run_head) is NOT used — head.1/head.3 are
 * training-only layers; the production classifier is the SVM.
 * ═══════════════════════════════════════════════════════════════════ */
static uint8_t run_svm(void)
{
    uint32_t d, c;
    float scaled[CNN_FEAT_DIM];

    /* StandardScaler: scaled[d] = (features[d] - mean[d]) / std[d] */
    for (d = 0u; d < CNN_FEAT_DIM; d++)
    {
        float mu  = (float)g_scaler_mean[d] * SCALER_MEAN_SCALE;
        float std = (float)g_scaler_std [d] * SCALER_STD_SCALE;
        if (std < 1e-8f) std = 1e-8f;
        scaled[d] = (g_features[d] - mu) / std;
    }

    /* LinearSVC argmax */
    float   best   = -3.4e38f;
    uint8_t best_c = 0u;
    for (c = 0u; c < (uint32_t)NUM_CLASSES; c++)
    {
        float score = (float)g_svm_bias[c] * SVM_B_SCALE;
        for (d = 0u; d < SVM_DIM; d++)
            score += (float)g_svm_weights[c * SVM_DIM + d] * SVM_W_SCALE * scaled[d];
        if (score > best) { best = score; best_c = (uint8_t)c; }
    }
    return best_c + 1u;   /* SVM classes_ = [1,2,3,4,5] -> offset 0 = class 1 */
}
