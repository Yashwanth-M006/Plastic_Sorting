#include "driverlib.h"
#include "device.h"
#include <string.h>
#include <math.h>
#include "nir_model.h"
#include "nir_weights.h"
#include "nir_bn_weights.h"
#include "nir_shared_ipc.h"

#pragma CODE_SECTION(conv1d_stem_cpu2, ".TI.ramfunc")
#pragma CODE_SECTION(conv1d_Q16_split_cpu2, ".TI.ramfunc")
#pragma CODE_SECTION(conv1d_Q16_split3_cpu2, ".TI.ramfunc")
#pragma CODE_SECTION(run_cnn_cpu2, ".TI.ramfunc")
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



static void conv1d_stem_cpu2() {
    uint32_t oc, p, k;
    float w_local[25];
    for (oc = 32u; oc < 64u; oc++) { // CPU2 does 32..63
        for (k = 0; k < 25u; k++) w_local[k] = (float)g_bb0_weight[oc * 25u + k] * SCALE_BACKBONE_0_WEIGHT;
        for (p = 0; p < STEM_OUT_LEN; p++) {
            float acc = 0.0f;
            int32_t in_start = (int32_t)(p * 8u) - 12;
            #pragma MUST_ITERATE(1)
            for (k = 0; k < 25u; k++) {
                int32_t pos = in_start + (int32_t)k;
                if (pos >= 0 && pos < (int32_t)NUM_CHANNELS) acc += g_input[pos] * w_local[k];
            }
            g_act0_cpu2[oc - 32u][p] = acc;
        }
    }
}

static void conv1d_Q16_split_cpu2() {
    uint32_t oc, ic, p, k;
    float w_local[320];
    for (oc = 64u; oc < 128u; oc++) { // CPU2 does 64..127
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
            g_act1_cpu2[oc - 64u][p] = acc;
        }
    }
}

static void conv1d_Q16_split3_cpu2() {
    uint32_t oc, ic, p, k;
    float w_local[384];
    for (oc = 64u; oc < 128u; oc++) { // CPU2 does 64..127
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
                    if (in_pos >= 0 && in_pos < (int32_t)STEM_OUT_LEN) {
                        acc += get_act1_val(ic, (uint32_t)in_pos) * w_local[ic * 3u + k];
                    }
                }
            }
            g_act2_cpu2[oc - 64u][p] = acc;
        }
    }
}

static void run_cnn_cpu2() {
    uint32_t ch, p;
    while(1) {
        // Wait for CPU1 to signal STEM start
        IPC_waitForFlag(IPC_CPU2_L_CPU1_R, IPC_FLAG_START_STEM);
        IPC_ackFlagRtoL(IPC_CPU2_L_CPU1_R, IPC_FLAG_START_STEM);
        
        conv1d_stem_cpu2();
        for (ch = 32u; ch < 64u; ch++) {
            for (p = 0; p < STEM_OUT_LEN; p++) {
                float *v = &g_act0_cpu2[ch - 32u][p];
                apply_bn_ch(v, ch, 0, g_bb1_weight, g_bb1_bias, g_bb1_running_mean, g_bb1_running_var);
                *v = gelu_approx(*v);
            }
        }
        
        IPC_setFlagLtoR(IPC_CPU2_L_CPU1_R, IPC_FLAG_DONE_STEM);
        
        // Block 1
        IPC_waitForFlag(IPC_CPU2_L_CPU1_R, IPC_FLAG_START_BLK1);
        IPC_ackFlagRtoL(IPC_CPU2_L_CPU1_R, IPC_FLAG_START_BLK1);
        
        conv1d_Q16_split_cpu2();
        for (ch = 64u; ch < 128u; ch++) {
            for (p = 0; p < STEM_OUT_LEN; p++) {
                float *v = &g_act1_cpu2[ch - 64u][p];
                apply_bn_ch(v, ch, 1, g_bb4_weight, g_bb4_bias, g_bb4_running_mean, g_bb4_running_var);
                *v = gelu_approx(*v);
            }
        }
        maxpool1d_2((float*)g_act1_cpu2, STEM_OUT_LEN, 64u);
        
        IPC_setFlagLtoR(IPC_CPU2_L_CPU1_R, IPC_FLAG_DONE_BLK1);

        // Block 2
        IPC_waitForFlag(IPC_CPU2_L_CPU1_R, IPC_FLAG_START_BLK2);
        IPC_ackFlagRtoL(IPC_CPU2_L_CPU1_R, IPC_FLAG_START_BLK2);
        
        conv1d_Q16_split3_cpu2();
        for (ch = 64u; ch < 128u; ch++) {
            for (p = 0; p < BLK1_OUT_LEN; p++) {
                float *v = &g_act2_cpu2[ch - 64u][p];
                apply_bn_ch(v, ch, 2, g_bb8_weight, g_bb8_bias, g_bb8_running_mean, g_bb8_running_var);
                *v = gelu_approx(*v);
            }
        }
        
        float inv = 1.0f / (float)BLK1_OUT_LEN;
        for (ch = 64u; ch < 128u; ch++) {
            float s = 0.0f;
            for (p = 0; p < BLK1_OUT_LEN; p++) s += g_act2_cpu2[ch - 64u][p];
            g_features_cpu2[ch - 64u] = s * inv;
        }
        
        IPC_setFlagLtoR(IPC_CPU2_L_CPU1_R, IPC_FLAG_DONE_BLK2);
    }
}

void main(void) {
    Device_init();
    Interrupt_initModule();
    Interrupt_initVectorTable();

    IPC_clearFlagLtoR(IPC_CPU2_L_CPU1_R, IPC_FLAG_ALL);
    
    // Signal to CPU1 that we are booted and ready
    IPC_setFlagLtoR(IPC_CPU2_L_CPU1_R, IPC_FLAG_CPU2_READY);
    
    run_cnn_cpu2(); // Enters infinite loop handling inference requests
}
