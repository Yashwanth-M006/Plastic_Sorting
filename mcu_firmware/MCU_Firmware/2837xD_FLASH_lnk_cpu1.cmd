/*
 * 2837xD_FLASH_lnk_cpu1.cmd  -- NIR inference firmware
 * ======================================================
 * Fix log vs. original:
 *   FIX-1 : RAMLS0 (and RAMLS1 as overflow) added to PAGE 0 so the
 *            .TI.ramfunc RUN target is in program-address space.
 *            RAMLS0/1 are unified RAM — physically the same words —
 *            so declaring them on PAGE 0 costs nothing; the linker
 *            just uses the program-space alias (0x008000 on PAGE 0
 *            == 0x008000 on PAGE 1 on this device).
 *   FIX-2 : .init_array placed (already present, kept as-is).
 *   FIX-3 : EABI symbol names have NO leading underscore.
 *            _RamfuncsLoadStart -> RamfuncsLoadStart  (etc.)
 *   FIX-4 : ALIGN(8) on .TI.ramfunc eliminates trampoline warnings
 *            that arise with FPU32 when the section starts on an
 *            odd 32-bit boundary.
 */

MEMORY
{
PAGE 0:  /* -- Program (flash + code-RAM) -- */

   BEGIN      : origin = 0x080000, length = 0x000002
   FLASHA     : origin = 0x080002, length = 0x001FFE

   FLASHB     : origin = 0x082000, length = 0x002000
   FLASHC     : origin = 0x084000, length = 0x002000
   FLASHD     : origin = 0x086000, length = 0x002000
   FLASHE     : origin = 0x088000, length = 0x008000
   FLASHF     : origin = 0x090000, length = 0x008000
   FLASHG     : origin = 0x098000, length = 0x008000
   FLASHH     : origin = 0x0A0000, length = 0x008000
   FLASHI     : origin = 0x0A8000, length = 0x008000
   FLASHJ     : origin = 0x0B0000, length = 0x008000
   FLASHK     : origin = 0x0B8000, length = 0x008000
   FLASHL     : origin = 0x0C0000, length = 0x008000
   FLASHM     : origin = 0x0C8000, length = 0x008000
   FLASHN     : origin = 0x0D0000, length = 0x008000

   /* FIX-1: RAMLS0/1 declared on PAGE 0 so .TI.ramfunc RUN= is legal.
    * These are the same physical words as the PAGE 1 entries below;
    * C2000 unified RAM is accessible from both address spaces.        */
   RAMLS0_P0  : origin = 0x008000, length = 0x000800   /* 2 KB, code-run target  */
   RAMLS1_P0  : origin = 0x008800, length = 0x000800   /* 2 KB, overflow if needed */

   RESET      : origin = 0x3FFFC0, length = 0x000002

PAGE 1:  /* -- Data (RAM) -- */

   BOOT_RSVD  : origin = 0x000002, length = 0x000121
   RAMM0      : origin = 0x000123, length = 0x0002DD
   RAMM1      : origin = 0x000400, length = 0x000400

   /* RAMLS0/1 kept here too for any data-space users (e.g. CLA).
    * Linker will not double-allocate as long as no section targets
    * both the PAGE 0 and PAGE 1 aliases simultaneously.              */
   RAMLS0     : origin = 0x008000, length = 0x000800
   RAMLS1     : origin = 0x008800, length = 0x000800
   RAMLS2     : origin = 0x009000, length = 0x000800
   RAMLS3     : origin = 0x009800, length = 0x000800
   RAMLS4     : origin = 0x00A000, length = 0x000800
   RAMLS5     : origin = 0x00A800, length = 0x000800

   RAMGS0     : origin = 0x00C000, length = 0x001000
   RAMGS1     : origin = 0x00D000, length = 0x001000
   RAMGS2     : origin = 0x00E000, length = 0x001000
   RAMGS3     : origin = 0x00F000, length = 0x001000
   RAMGS4     : origin = 0x010000, length = 0x001000
   RAMGS5     : origin = 0x011000, length = 0x001000

   /* RAMGS6-9 merged as one 32 KB region for .ebss activation buffers */
   RAMGS_NIR  : origin = 0x012000, length = 0x004000

   RAMGS10    : origin = 0x016000, length = 0x001000
   RAMGS11    : origin = 0x017000, length = 0x001000
   RAMGS12    : origin = 0x018000, length = 0x001000
   RAMGS13    : origin = 0x019000, length = 0x001000
   RAMGS14    : origin = 0x01A000, length = 0x001000
   RAMGS15    : origin = 0x01B000, length = 0x001000

   CPU2TOCPU1RAM : origin = 0x03F800, length = 0x000400
   CPU1TOCPU2RAM : origin = 0x03FC00, length = 0x000400
}

SECTIONS
{
   /* ── Vectors & entry ───────────────────────────────────────────── */
   codestart    : > BEGIN,     PAGE = 0
   .reset       : > RESET,     PAGE = 0, TYPE = DSECT

   /* ── Code ──────────────────────────────────────────────────────── */
   .text        : >> FLASHB | FLASHC | FLASHD | FLASHE,  PAGE = 0
   .cinit       : > FLASHD,    PAGE = 0
   .switch      : > FLASHD,    PAGE = 0
   .pinit       : > FLASHD,    PAGE = 0
   .init_array  : > FLASHD,    PAGE = 0   /* FIX-2: C++ global constructors */

   /* ── Small read-only data (BN weights, SVM, scaler, strings) ───── */
   .const       : > FLASHE,    PAGE = 0

   /* ── Large conv weight arrays: ONE section per flash bank ─────── */
   .nir_bb3_lo  : > FLASHF,    PAGE = 0   /* g_bb3_weight_lo[20480] = 40 KB */
   .nir_bb3_hi  : > FLASHG,    PAGE = 0   /* g_bb3_weight_hi[20480] = 40 KB */
   .nir_bb7_a   : > FLASHH,    PAGE = 0   /* g_bb7_weight_a [16384] = 32 KB */
   .nir_bb7_b   : > FLASHI,    PAGE = 0   /* g_bb7_weight_b [16384] = 32 KB */
   .nir_bb7_c   : > FLASHJ,    PAGE = 0   /* g_bb7_weight_c [16384] = 32 KB */

   /* ── Flash API copied to RAM ────────────────────────────────────── 
    * FIX-1: RUN target is RAMLS0_P0 (PAGE 0 alias of RAMLS0).
    * FIX-3: No leading underscore on symbols (EABI convention).
    * FIX-4: ALIGN(8) prevents FPU32 trampoline warning #10099.       */
   .TI.ramfunc  : LOAD = FLASHD,
                  RUN  = RAMLS0_P0,
                  ALIGN(8),
                  LOAD_START(RamfuncsLoadStart),
                  LOAD_SIZE(RamfuncsLoadSize),
                  LOAD_END(RamfuncsLoadEnd),
                  RUN_START(RamfuncsRunStart),
                  RUN_SIZE(RamfuncsRunSize),
                  RUN_END(RamfuncsRunEnd),
                  PAGE = 0

   /* ── Activation buffers (29.4 KB) -> 32 KB contiguous region ──── */
   .ebss        : > RAMGS_NIR, PAGE = 1

   /* ── Stack ─────────────────────────────────────────────────────── */
   .stack       : > RAMM1,     PAGE = 1

   /* ── General RAM ───────────────────────────────────────────────── */
   .data        : > RAMGS0,    PAGE = 1
   .bss         : > RAMGS0,    PAGE = 1
   .sysmem      : > RAMGS1,    PAGE = 1

   /* ── IPC ───────────────────────────────────────────────────────── */
   MSGRAM_CPU1_TO_CPU2 : > CPU1TOCPU2RAM, PAGE = 1
   MSGRAM_CPU2_TO_CPU1 : > CPU2TOCPU1RAM, PAGE = 1
}