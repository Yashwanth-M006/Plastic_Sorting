# bin_to_header.py  — run on PC
BIN_FILE = "nir_samples.bin"
OUT_FILE = "nir_data.h"
VAR_NAME = "g_nir_data"

with open(BIN_FILE, "rb") as f:
    data = f.read()

with open(OUT_FILE, "w") as f:
    f.write("#pragma once\n")
    f.write("#include <stdint.h>\n\n")
    f.write(f"// Auto-generated from {BIN_FILE}\n")
    f.write(f"// {len(data)} bytes total\n\n")
    f.write(f"const uint8_t {VAR_NAME}[] PROGMEM = {{\n")
    for i, b in enumerate(data):
        if i % 16 == 0:
            f.write("    ")
        f.write(f"0x{b:02X},")
        if i % 16 == 15:
            f.write("\n")
        else:
            f.write(" ")
    f.write("\n};\n")
    f.write(f"const uint32_t {VAR_NAME}_len = {len(data)};\n")

print(f"Written {len(data)} bytes → {OUT_FILE}")
print(f"Array size in flash: {len(data)/1024:.1f} KB")