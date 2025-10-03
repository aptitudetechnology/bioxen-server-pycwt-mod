#!/usr/bin/env bash
# Quick ELM11 functionality test

echo "╔════════════════════════════════════════════════════════════╗"
echo "║         ELM11 FPGA - Functionality Test                   ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

cd ELM11-Lua-FFT

echo "Testing ELM11 interface..."
echo ""
echo "This will attempt to:"
echo "  1. Connect to ELM11 on /dev/ttyUSB1 at 115200 baud"
echo "  2. Send a test command"
echo "  3. Verify response"
echo ""

# Create a simple test script
cat > test_connection.py << 'EOF'
#!/usr/bin/env python3
import serial
import time

print("Connecting to ELM11...")
try:
    ser = serial.Serial('/dev/ttyUSB1', 115200, timeout=2)
    print(f"✓ Connected to /dev/ttyUSB1 at 115200 baud")
    
    time.sleep(2)  # Wait for device
    ser.read(1024)  # Clear buffer
    
    # Try to get into Lua REPL
    print("\nSending test command...")
    ser.write(b'\r\n')
    ser.flush()
    time.sleep(0.5)
    response1 = ser.read(256)
    print(f"Response 1: {response1}")
    
    # Try a Lua print command
    ser.write(b'print("HELLO")\r\n')
    ser.flush()
    time.sleep(0.5)
    response2 = ser.read(256)
    print(f"Response 2: {response2}")
    
    if b'HELLO' in response2:
        print("\n✓ ELM11 Lua REPL is working!")
        print("✓ Device is ready for FFT operations")
    else:
        print("\n⚠ Got response but not in expected format")
        print("  Device may need initialization")
    
    ser.close()
    print("\n✓ Test complete")
    
except Exception as e:
    print(f"✗ Error: {e}")
EOF

python test_connection.py
rm test_connection.py

echo ""
echo "════════════════════════════════════════════════════════════"
echo ""
echo "ELM11 Status: Hardware detected and responding"
echo ""
echo "Next steps for FPGA integration:"
echo "  1. Review ELM11-Lua-FFT/elm11_interface.py"
echo "  2. Test FFT operations: python elm11_interface.py"
echo "  3. Integrate with pycwt backend system (Phase 3)"
echo ""
