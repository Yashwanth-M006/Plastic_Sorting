//#############################################################################
//
// FILE:   empty_driverlib_main.c
//
// TITLE:  Empty Project
//
// Empty Project Example
//
// This example is an empty project setup for Driverlib development.
//
//#############################################################################
//
//
// $Copyright:
// Copyright (C) 2013-2026 Texas Instruments Incorporated - http://www.ti.com/
//
// Redistribution and use in source and binary forms, with or without 
// modification, are permitted provided that the following conditions 
// are met:
// 
//   Redistributions of source code must retain the above copyright 
//   notice, this list of conditions and the following disclaimer.
// 
//   Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the following disclaimer in the 
//   documentation and/or other materials provided with the   
//   distribution.
// 
//   Neither the name of Texas Instruments Incorporated nor the names of
//   its contributors may be used to endorse or promote products derived
//   from this software without specific prior written permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS 
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT 
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT 
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, 
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT 
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT 
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE 
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// $
//#############################################################################

//
// Included Files
//
#include "driverlib.h"
#include "device.h"
#include "board.h"
#include "c2000ware_libraries.h"
#include <math.h>
#include "dark_current_mean.h"

//
// Preprocessing pipeline
//
void apply_preprocessing(float* spectrum, int len)
{
    float temp1[NUM_CHANNELS];
    float temp2[NUM_CHANNELS];
    int i;
    
    // 1. Subtract Dark Current
    for (i = 0; i < len; i++) {
        temp1[i] = spectrum[i] - g_dark_current_mean[i];
    }
    
    // 2. Median Filter (3-tap)
    for (i = 0; i < len; i++) {
        float a = (i == 0) ? temp1[i] : temp1[i-1];
        float b = temp1[i];
        float c = (i == len - 1) ? temp1[i] : temp1[i+1];
        
        float min_ab = (a < b) ? a : b;
        float max_ab = (a > b) ? a : b;
        float min_max_ab_c = (max_ab < c) ? max_ab : c;
        temp2[i] = (min_ab > min_max_ab_c) ? min_ab : min_max_ab_c;
    }
    
    // 3. Savitzky-Golay Filter (5-tap)
    for (i = 0; i < len; i++) {
        float a = (i < 2) ? temp2[0] : temp2[i-2];
        float b = (i < 1) ? temp2[0] : temp2[i-1];
        float c = temp2[i];
        float d = (i > len - 2) ? temp2[len-1] : temp2[i+1];
        float e = (i > len - 3) ? temp2[len-1] : temp2[i+2];
        
        spectrum[i] = (-22.0f * a + 88.0f * b + 124.0f * c + 88.0f * d - 22.0f * e) / 256.0f;
    }
    
    // 4. SNV (Standard Normal Variate)
    float sum = 0.0f;
    for (i = 0; i < len; i++) {
        sum += spectrum[i];
    }
    float mean = sum / len;
    
    float sq_sum = 0.0f;
    for (i = 0; i < len; i++) {
        float diff = spectrum[i] - mean;
        sq_sum += diff * diff;
    }
    float std = sqrtf(sq_sum / len); // Population std (ddof=0) matching Python
    if (std == 0.0f) std = 1.0f; // Prevent div by zero
    
    float inv_std = 1.0f / std;
    for (i = 0; i < len; i++) {
        spectrum[i] = (spectrum[i] - mean) * inv_std;
    }
}

/* ── SCI-A TX to XDS110 (USB COM Port) ────────────────────────────── */
#define LOG_SCI_BASE   SCIA_BASE

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

//
// Main
//
void main(void)
{
    //
    // Initialize device clock and peripherals
    //
    Device_init();

    //
    // Disable pin locks and enable internal pull-ups.
    //
    Device_initGPIO();

    //
    // Initialize PIE and clear PIE registers. Disables CPU interrupts.
    //
    Interrupt_initModule();

    //
    // Initialize the PIE vector table with pointers to the shell Interrupt
    // Service Routines (ISR).
    //
    Interrupt_initVectorTable();

    //
    // PinMux and Peripheral Initialization
    //
    Board_init();

    //
    // C2000Ware Library initialization
    //
    C2000Ware_libraries_init();

    //
    // Enable Global Interrupt (INTM) and real time interrupt (DBGM)
    //
    EINT;
    ERTM;

    // Initialize SCI-A TX for logging
    log_init();

    // Initialize CPU Timer 0 for timing analysis
    CPUTimer_stopTimer(CPUTIMER0_BASE);
    CPUTimer_setPeriod(CPUTIMER0_BASE, 0xFFFFFFFFUL);
    CPUTimer_setPreScaler(CPUTIMER0_BASE, 199u); // 200 cycles per tick = 1 us at 200MHz
    CPUTimer_reloadTimerCounter(CPUTIMER0_BASE);

    // Boot message
    log_str("--- C2000 Preprocessing Test ---"); log_nl();

    // Test preprocessing on a dummy array
    float test_spectrum[NUM_CHANNELS];
    int i;
    for(i=0; i<NUM_CHANNELS; i++) {
        test_spectrum[i] = g_dark_current_mean[i] + 10.0f; // dummy signal
    }
    
    log_str("Running preprocessing..."); log_nl();

    CPUTimer_startTimer(CPUTIMER0_BASE);
    apply_preprocessing(test_spectrum, NUM_CHANNELS);
    CPUTimer_stopTimer(CPUTIMER0_BASE);

    uint32_t elapsed_us = 0xFFFFFFFFUL - CPUTimer_getTimerCount(CPUTIMER0_BASE);

    log_str("Preprocessing Time: ");
    log_uint(elapsed_us);
    log_str(" us");
    log_nl();

    log_str("Preprocessed Spectrum:"); log_nl();
    for(i=0; i<NUM_CHANNELS; i++) {
        log_float(test_spectrum[i]);
        if(i < NUM_CHANNELS - 1) {
            log_str(", ");
        }
        if((i + 1) % 10 == 0) {
            log_nl();
        }
    }
    log_nl();
    log_str("--- End of Test ---"); log_nl();

    while(1)
    {
        
    }
}

//
// End of File
//

