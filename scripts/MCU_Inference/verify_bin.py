import struct

with open("nir_samples.bin", "rb") as f:
    raw = f.read(8)
    n, ch = struct.unpack("<II", raw)
    print(f"Header: num_samples={n}, num_channels={ch}")
    total = 8 + n * (1 + ch * 4)
    import os
    actual = os.path.getsize("nir_samples.bin")
    print(f"Expected size: {total} bytes")
    print(f"Actual size:   {actual} bytes")
    print("OK" if total == actual else "MISMATCH — file is corrupt")