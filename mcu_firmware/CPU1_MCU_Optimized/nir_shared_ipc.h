
#ifndef NIR_SHARED_IPC_H
#define NIR_SHARED_IPC_H

#include "driverlib.h"
#include "device.h"
#include "nir_model.h"

// ── Shared Memory for Dual-Core inference ──
// CPU1 writes to these, CPU2 can read
#pragma DATA_SECTION(g_input, "SHARERAMGS0")
#pragma DATA_SECTION(g_act0_cpu1, "SHARERAMGS1")
#pragma DATA_SECTION(g_act1_cpu1, "SHARERAMGS3")
#pragma DATA_SECTION(g_act2_cpu1, "SHARERAMGS5")
#pragma DATA_SECTION(g_features_cpu1, "SHARERAMGS7")

extern float g_input[NUM_CHANNELS];
extern float g_act0_cpu1[32][28];
extern float g_act1_cpu1[64][28];
extern float g_act2_cpu1[64][14];
extern float g_features_cpu1[64];

// CPU2 writes to these, CPU1 can read
#pragma DATA_SECTION(g_act0_cpu2, "SHARERAMGS2")
#pragma DATA_SECTION(g_act1_cpu2, "SHARERAMGS4")
#pragma DATA_SECTION(g_act2_cpu2, "SHARERAMGS6")
#pragma DATA_SECTION(g_features_cpu2, "SHARERAMGS8")

extern float g_act0_cpu2[32][28];
extern float g_act1_cpu2[64][28];
extern float g_act2_cpu2[64][14];
extern float g_features_cpu2[64];

// IPC Synchronization Flags
#define IPC_FLAG_CPU2_READY    IPC_FLAG10
#define IPC_FLAG_START_STEM    IPC_FLAG11
#define IPC_FLAG_DONE_STEM     IPC_FLAG12
#define IPC_FLAG_START_BLK1    IPC_FLAG13
#define IPC_FLAG_DONE_BLK1     IPC_FLAG14
#define IPC_FLAG_START_BLK2    IPC_FLAG15
#define IPC_FLAG_DONE_BLK2     IPC_FLAG16
#define IPC_FLAG_START_FEAT    IPC_FLAG17
#define IPC_FLAG_DONE_FEAT     IPC_FLAG18

#endif // NIR_SHARED_IPC_H
