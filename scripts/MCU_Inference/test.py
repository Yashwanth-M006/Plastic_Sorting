# uart_ping.py  — paste and run standalone
import serial, time, struct

PORT  = "COM17"
BAUD  = 115200

print(f"Opening {PORT}...")
with serial.Serial(PORT, BAUD, timeout=3) as s:
    time.sleep(0.6)
    s.reset_input_buffer()

    # Step 1: send just the start byte
    print("Sending 0xAA...")
    s.write(bytes([0xAA]))
    time.sleep(0.1)

    # Step 2: send the 896 float bytes (all zeros)
    dummy = struct.pack(f"<{224}f", *([0.0] * 224))
    s.write(dummy)
    print(f"Sent {1 + len(dummy)} bytes total")

    # Step 3: wait for response
    resp = s.read(2)
    if len(resp) == 0:
        print("ERROR: no response — MCU never received 0xAA")
        print("  → XDS100 backchannel RX is not reaching GPIO28")
    elif resp[0] != 0xBB:
        print(f"ERROR: bad ACK byte 0x{resp[0]:02X}, expected 0xBB")
    else:
        print(f"OK: got 0xBB + class={resp[1]}")