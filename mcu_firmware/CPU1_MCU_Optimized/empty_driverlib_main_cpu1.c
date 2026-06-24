#include "driverlib.h"
#include "device.h"
#include <string.h>
#include <math.h>
#include "nir_model.h"
#include "nir_weights.h"
#include "nir_bn_weights.h"
#include "nir_head_weights.h"
#include "nir_selftest_data.h"
#include "nir_shared_ipc.h"

#pragma CODE_SECTION(conv1d_stem_cpu1, ".TI.ramfunc")
#pragma CODE_SECTION(conv1d_Q16_split_cpu1, ".TI.ramfunc")
#pragma CODE_SECTION(conv1d_Q16_split3_cpu1, ".TI.ramfunc")
#pragma CODE_SECTION(run_cnn_cpu1, ".TI.ramfunc")
#pragma CODE_SECTION(apply_bn_ch, ".TI.ramfunc")
#pragma CODE_SECTION(gelu_approx, ".TI.ramfunc")


#define STEM_OUT_LEN 28u
#define BLK1_OUT_LEN 14u

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

#define GELU_C1 0.7978845608f
#define GELU_C2 0.044715f
static float gelu_approx(float x)
{
    float x3 = x * x * x;
    return 0.5f * x * (1.0f + tanhf(GELU_C1 * (x + GELU_C2 * x3)));
}

// Split array functions
static void apply_bn_ch(float *v, uint16_t ch, uint16_t layer,
                         const int16_t *W, const int16_t *B,
                         const int16_t *MU, const int16_t *VAR)
{
    float w, b, mu, var;
    (void)W; (void)B; (void)MU; (void)VAR;

    switch (layer)
    {
        case 0:
            w   = g_bn1_weight      [ch];
            b   = g_bn1_bias        [ch];
            mu  = g_bn1_running_mean[ch];
            var = g_bn1_running_var [ch];
            break;
        case 1:
            w   = g_bn4_weight      [ch];
            b   = g_bn4_bias        [ch];
            mu  = g_bn4_running_mean[ch];
            var = g_bn4_running_var [ch];
            break;
        case 2:
        default:
            w   = g_bn8_weight      [ch];
            b   = g_bn8_bias        [ch];
            mu  = g_bn8_running_mean[ch];
            var = g_bn8_running_var [ch];
            break;
    }

    *v = (*v - mu) * (1.0f / sqrtf(var + 1e-5f)) * w + b;
}

// Memory mapping definition in C file
float g_input[NUM_CHANNELS];
float g_act0_cpu1[32][28];
float g_act1_cpu1[64][28];
float g_act2_cpu1[64][14];
float g_features_cpu1[64];

float g_act0_cpu2[32][28];
float g_act1_cpu2[64][28];
float g_act2_cpu2[64][14];
float g_features_cpu2[64];

// Utility: get row pointer seamlessly across both CPU buffers
static float* get_act0_row(uint32_t ch) {
    if (ch < 32u) return g_act0_cpu1[ch];
    return g_act0_cpu2[ch - 32u];
}

static float* get_act1_row(uint32_t ch) {
    if (ch < 64u) return g_act1_cpu1[ch];
    return g_act1_cpu2[ch - 64u];
}

static float* get_act2_row(uint32_t ch) {
    if (ch < 64u) return g_act2_cpu1[ch];
    return g_act2_cpu2[ch - 64u];
}

static float get_act0_val(uint32_t ch, uint32_t p) {
    return get_act0_row(ch)[p];
}

static float get_act1_val(uint32_t ch, uint32_t p) {
    return get_act1_row(ch)[p];
}



/* ── SCI-A TX to XDS110 (USB COM Port) ── */
#define LOG_SCI_BASE SCIA_BASE

static void log_init(void) {
    SysCtl_enablePeripheral(SYSCTL_PERIPH_CLK_SCIA);
    DEVICE_DELAY_US(100);
    GPIO_setPinConfig(GPIO_42_SCITXDA);
    GPIO_setDirectionMode(42, GPIO_DIR_MODE_OUT);
    GPIO_setPadConfig(42, GPIO_PIN_TYPE_STD);
    SCI_performSoftwareReset(LOG_SCI_BASE);
    SCI_setConfig(LOG_SCI_BASE, DEVICE_LSPCLK_FREQ, 115200UL, SCI_CONFIG_WLEN_8 | SCI_CONFIG_STOP_ONE | SCI_CONFIG_PAR_NONE);
    SCI_resetChannels(LOG_SCI_BASE);
    SCI_clearInterruptStatus(LOG_SCI_BASE, SCI_INT_RXFF | SCI_INT_TXFF);
    SCI_enableModule(LOG_SCI_BASE);
    SCI_performSoftwareReset(LOG_SCI_BASE);
}
static void log_char(char c) { SCI_writeCharBlockingNonFIFO(LOG_SCI_BASE, (uint16_t)(uint8_t)c); }
static void log_str(const char *s) { while (*s) log_char(*s++); }
static void log_uint(uint32_t v) { char buf[12]; int8_t i = 0; if (v == 0u) { log_char('0'); return; } while (v > 0u) { buf[i++] = (char)('0' + (v % 10u)); v /= 10u; } while (--i >= 0) log_char(buf[i]); }
static void log_nl(void) { log_char('\r'); log_char('\n'); }
static void log_float(float f) { if (f < 0.0f) { log_char('-'); f = -f; } uint32_t integer_part = (uint32_t)f; float fractional_part = f - (float)integer_part; uint32_t fraction_val = (uint32_t)(fractional_part * 1000.0f + 0.5f); log_uint(integer_part); log_char('.'); if (fraction_val < 100u) log_char('0'); if (fraction_val < 10u)  log_char('0'); log_uint(fraction_val); }

static void led_blink(uint32_t n, uint32_t on_us, uint32_t off_us) {
    uint32_t j;
    for (j = 0; j < n; j++) { GPIO_writePin(31, 1); DEVICE_DELAY_US(on_us); GPIO_writePin(31, 0); DEVICE_DELAY_US(off_us); }
}
#define BLINK_SLOW(n) led_blink((n), 400000u, 300000u)
#define BLINK_FAST(n) led_blink((n), 100000u, 100000u)
#define BLINK_LONG(n) led_blink((n), 600000u, 400000u)
#define PAUSE(us) DEVICE_DELAY_US(us)

// Convolution definitions for CPU1 (Handles first half of channels)
static void conv1d_stem_cpu1() {
    uint32_t oc, p, k;
    float w_local[25];
    for (oc = 0; oc < 32u; oc++) { // CPU1 does 0..31
        for (k = 0; k < 25u; k++) w_local[k] = (float)g_bb0_weight[oc * 25u + k] * SCALE_BACKBONE_0_WEIGHT;
        for (p = 0; p < STEM_OUT_LEN; p++) {
            float acc = 0.0f;
            int32_t in_start = (int32_t)(p * 8u) - 12;
            #pragma MUST_ITERATE(1)
            for (k = 0; k < 25u; k++) {
                int32_t pos = in_start + (int32_t)k;
                if (pos >= 0 && pos < (int32_t)NUM_CHANNELS) acc += g_input[pos] * w_local[k];
            }
            g_act0_cpu1[oc][p] = acc;
        }
    }
}

static void conv1d_Q16_split_cpu1() {
    uint32_t oc, ic, p, k;
    float w_local[320]; // 64 ch * 5
    for (oc = 0; oc < 64u; oc++) { // CPU1 does 0..63
        for (ic = 0; ic < 64u; ic++) {
            for (k = 0; k < 5u; k++) {
                uint32_t widx = (oc * 64u + ic) * 5u + k;
                int16_t wq = (widx < 20480u) ? g_bb3_weight_lo[widx] : g_bb3_weight_hi[widx - 20480u];
                w_local[ic * 5u + k] = (float)wq * SCALE_BACKBONE_3_WEIGHT;
            }
        }
        for (p = 0; p < STEM_OUT_LEN; p++) {
            float acc = 0.0f;
            int32_t in_start = (int32_t)(p * 1u) - 2;
            for (ic = 0; ic < 64u; ic++) {
                #pragma MUST_ITERATE(1)
                for (k = 0; k < 5u; k++) {
                    int32_t in_pos = in_start + (int32_t)k;
                    if (in_pos >= 0 && in_pos < (int32_t)STEM_OUT_LEN) {
                        acc += get_act0_val(ic, (uint32_t)in_pos) * w_local[ic * 5u + k];
                    }
                }
            }
            g_act1_cpu1[oc][p] = acc;
        }
    }
}

static void conv1d_Q16_split3_cpu1() {
    uint32_t oc, ic, p, k;
    float w_local[384]; // 128 ch * 3
    for (oc = 0; oc < 64u; oc++) { // CPU1 does 0..63
        for (ic = 0; ic < 128u; ic++) {
            for (k = 0; k < 3u; k++) {
                uint32_t widx = (oc * 128u + ic) * 3u + k;
                int16_t wq;
                if (widx < 16384u) wq = g_bb7_weight_a[widx];
                else if (widx < 32768u) wq = g_bb7_weight_b[widx - 16384u];
                else wq = g_bb7_weight_c[widx - 32768u];
                w_local[ic * 3u + k] = (float)wq * SCALE_BACKBONE_7_WEIGHT;
            }
        }
        for (p = 0; p < BLK1_OUT_LEN; p++) {
            float acc = 0.0f;
            int32_t in_start = (int32_t)(p * 1u) - 1;
            for (ic = 0; ic < 128u; ic++) {
                #pragma MUST_ITERATE(1)
                for (k = 0; k < 3u; k++) {
                    int32_t in_pos = in_start + (int32_t)k;
                    if (in_pos >= 0 && in_pos < (int32_t)STEM_OUT_LEN) { // STEM_OUT_LEN stride logic
                        acc += get_act1_val(ic, (uint32_t)in_pos) * w_local[ic * 3u + k];
                    }
                }
            }
            g_act2_cpu1[oc][p] = acc;
        }
    }
}

static void run_cnn_cpu1() {
    uint32_t ch, p;
    // Signal CPU2 to start
    IPC_setFlagLtoR(IPC_CPU1_L_CPU2_R, IPC_FLAG_START_STEM);
    
    // CPU1 Stem (ch 0..31)
    conv1d_stem_cpu1();
    for (ch = 0; ch < 32u; ch++) {
        for (p = 0; p < STEM_OUT_LEN; p++) {
            float *v = &g_act0_cpu1[ch][p];
            apply_bn_ch(v, ch, 0, g_bb1_weight, g_bb1_bias, g_bb1_running_mean, g_bb1_running_var);
            *v = gelu_approx(*v);
        }
    }
    
    IPC_setFlagLtoR(IPC_CPU1_L_CPU2_R, IPC_FLAG_DONE_STEM);
    IPC_waitForFlag(IPC_CPU1_L_CPU2_R, IPC_FLAG_DONE_STEM);
    IPC_ackFlagRtoL(IPC_CPU1_L_CPU2_R, IPC_FLAG_DONE_STEM);

    // CPU1 Block 1 (ch 0..63)
    IPC_setFlagLtoR(IPC_CPU1_L_CPU2_R, IPC_FLAG_START_BLK1);
    conv1d_Q16_split_cpu1();
    for (ch = 0; ch < 64u; ch++) {
        for (p = 0; p < STEM_OUT_LEN; p++) {
            float *v = &g_act1_cpu1[ch][p];
            apply_bn_ch(v, ch, 1, g_bb4_weight, g_bb4_bias, g_bb4_running_mean, g_bb4_running_var);
            *v = gelu_approx(*v);
        }
    }
    maxpool1d_2((float*)g_act1_cpu1, STEM_OUT_LEN, 64u);
    
    IPC_setFlagLtoR(IPC_CPU1_L_CPU2_R, IPC_FLAG_DONE_BLK1);
    IPC_waitForFlag(IPC_CPU1_L_CPU2_R, IPC_FLAG_DONE_BLK1);
    IPC_ackFlagRtoL(IPC_CPU1_L_CPU2_R, IPC_FLAG_DONE_BLK1);

    // CPU1 Block 2 (ch 0..63)
    IPC_setFlagLtoR(IPC_CPU1_L_CPU2_R, IPC_FLAG_START_BLK2);
    conv1d_Q16_split3_cpu1();
    for (ch = 0; ch < 64u; ch++) {
        for (p = 0; p < BLK1_OUT_LEN; p++) {
            float *v = &g_act2_cpu1[ch][p];
            apply_bn_ch(v, ch, 2, g_bb8_weight, g_bb8_bias, g_bb8_running_mean, g_bb8_running_var);
            *v = gelu_approx(*v);
        }
    }
    
    float inv = 1.0f / (float)BLK1_OUT_LEN;
    for (ch = 0; ch < 64u; ch++) {
        float s = 0.0f;
        for (p = 0; p < BLK1_OUT_LEN; p++) s += g_act2_cpu1[ch][p];
        g_features_cpu1[ch] = s * inv;
    }
    
    IPC_setFlagLtoR(IPC_CPU1_L_CPU2_R, IPC_FLAG_DONE_BLK2);
    IPC_waitForFlag(IPC_CPU1_L_CPU2_R, IPC_FLAG_DONE_BLK2);
    IPC_ackFlagRtoL(IPC_CPU1_L_CPU2_R, IPC_FLAG_DONE_BLK2);
}

static uint8_t run_svm() {
    uint32_t d, c;
    float scaled[CNN_FEAT_DIM];
    
    // Merge features logic inside loops
    for (d = 0u; d < CNN_FEAT_DIM; d++) {
        float mu = (float)g_scaler_mean[d] * SCALER_MEAN_SCALE;
        float std = (float)g_scaler_std[d] * SCALER_STD_SCALE;
        if (std < 1e-8f) std = 1e-8f;
        float feat = (d < 64u) ? g_features_cpu1[d] : g_features_cpu2[d - 64u];
        scaled[d] = (feat - mu) / std;
    }

    float best = -3.4e38f;
    uint8_t best_c = 0u;
    for (c = 0u; c < (uint32_t)NUM_CLASSES; c++) {
        float score = (float)g_svm_bias[c] * SVM_B_SCALE;
        for (d = 0u; d < SVM_DIM; d++) {
            score += (float)g_svm_weights[c * SVM_DIM + d] * SVM_W_SCALE * scaled[d];
        }
        if (score > best) { best = score; best_c = (uint8_t)c; }
    }
    return best_c + 1u;
}

void main(void) {
    uint32_t i;
    uint32_t correct = 0u;
    uint32_t total_time_us = 0u;

    Device_init();
    Device_initGPIO();
    
    // Assign RAMGS2, RAMGS4, RAMGS6, RAMGS8 to CPU2 using DriverLib API
    MemCfg_setGSRAMControllerSel(MEMCFG_SECT_GS2, MEMCFG_GSRAMCONTROLLER_CPU2);
    MemCfg_setGSRAMControllerSel(MEMCFG_SECT_GS4, MEMCFG_GSRAMCONTROLLER_CPU2);
    MemCfg_setGSRAMControllerSel(MEMCFG_SECT_GS6, MEMCFG_GSRAMCONTROLLER_CPU2);
    MemCfg_setGSRAMControllerSel(MEMCFG_SECT_GS8, MEMCFG_GSRAMCONTROLLER_CPU2);
    
    // Boot CPU2
#ifdef _FLASH
    Device_bootCPU2(C1C2_BROM_BOOTMODE_BOOT_FROM_FLASH);
#else
    Device_bootCPU2(C1C2_BROM_BOOTMODE_BOOT_FROM_RAM);
#endif

    IPC_clearFlagLtoR(IPC_CPU1_L_CPU2_R, IPC_FLAG_ALL);
    
    GPIO_setPinConfig(GPIO_31_GPIO31);
    GPIO_setDirectionMode(31, GPIO_DIR_MODE_OUT);
    GPIO_writePin(31, 0);

    Interrupt_initModule();
    Interrupt_initVectorTable();
    log_init();

    CPUTimer_stopTimer(CPUTIMER0_BASE);
    CPUTimer_setPeriod(CPUTIMER0_BASE, 0xFFFFFFFFUL);
    CPUTimer_setPreScaler(CPUTIMER0_BASE, 199u);
    CPUTimer_reloadTimerCounter(CPUTIMER0_BASE);

    BLINK_SLOW(3);
    log_str("CPU1 BOOT OK\r\n");
    
    // Wait for CPU2 to be ready
    IPC_waitForFlag(IPC_CPU1_L_CPU2_R, IPC_FLAG_CPU2_READY);
    IPC_ackFlagRtoL(IPC_CPU1_L_CPU2_R, IPC_FLAG_CPU2_READY);
    log_str("CPU2 READY\r\n");
    PAUSE(1000000u);

    BLINK_SLOW(5);
    log_str("START,"); log_uint(SELFTEST_N_SAMPLES); log_char(','); log_uint(NUM_CLASSES); log_nl();
    
    for (i = 0u; i < SELFTEST_N_SAMPLES; i++) {
        memcpy(g_input, g_selftest_X[i], NUM_CHANNELS * sizeof(float));
        
        CPUTimer_startTimer(CPUTIMER0_BASE);
        run_cnn_cpu1();
        uint8_t pred = run_svm();
        CPUTimer_stopTimer(CPUTIMER0_BASE);
        
        uint32_t elapsed_us = 0xFFFFFFFFUL - CPUTimer_getTimerCount(CPUTIMER0_BASE);
        CPUTimer_reloadTimerCounter(CPUTIMER0_BASE);
        total_time_us += elapsed_us;
        
        uint8_t truth = g_selftest_labels[i];
        if (pred == truth) correct++;

        log_str("SAMPLE,"); log_uint(i + 1u); log_char(','); log_uint(SELFTEST_N_SAMPLES); log_char(',');
        log_uint(truth); log_char(','); log_uint(pred); log_char(','); log_str(pred == truth ? "OK" : "WRONG");
        log_char(','); log_uint(elapsed_us); log_str("us\r\n");
        PAUSE(300000u); BLINK_FAST(pred); PAUSE(700000u);
    }

    uint32_t avg_time_us = total_time_us / SELFTEST_N_SAMPLES;
    log_str("AVG_INFERENCE_TIME: "); log_uint(avg_time_us); log_str(" us\r\n");
    log_str("ACCURACY: "); log_uint(correct); log_str("/10\r\n");
    log_str("DONE\r\n");
    for (;;) { PAUSE(2000000u); if (correct > 0u) BLINK_LONG(correct); else PAUSE(1000000u); PAUSE(3000000u); }
}
