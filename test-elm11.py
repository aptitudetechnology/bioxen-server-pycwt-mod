#!/usr/bin/env python3
"""Quick test script to detect and verify ELM11 FPGA accelerator."""

import sys
import os
import glob
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Colors
GREEN = '\033[0;32m'
RED = '\033[0;31m'
YELLOW = '\033[1;33m'
BLUE = '\033[0;34m'
NC = '\033[0m'

print(f"{BLUE}")
print("╔════════════════════════════════════════════════════════════╗")
print("║         ELM11 FPGA Accelerator - Detection Test           ║")
print("╚════════════════════════════════════════════════════════════╝")
print(f"{NC}\n")

# Step 1: Check for USB serial devices
print(f"{BLUE}[1/5] Checking for USB serial devices...{NC}")
usb_devices = glob.glob('/dev/ttyUSB*') + glob.glob('/dev/ttyACM*')

if usb_devices:
    print(f"{GREEN}✓ Found {len(usb_devices)} USB serial device(s):{NC}")
    for device in usb_devices:
        print(f"  - {device}")
else:
    print(f"{RED}✗ No USB serial devices found{NC}")
    print(f"{YELLOW}  Make sure ELM11 is connected and drivers are installed{NC}")
    sys.exit(1)

# Step 2: Check if pyserial is installed
print(f"\n{BLUE}[2/5] Checking for pyserial library...{NC}")
try:
    import serial
    print(f"{GREEN}✓ pyserial is installed (version {serial.__version__}){NC}")
except ImportError:
    print(f"{RED}✗ pyserial not installed{NC}")
    print(f"{YELLOW}  Install with: pip install pyserial{NC}")
    sys.exit(1)

# Step 3: Check for ELM11 interface
print(f"\n{BLUE}[3/5] Checking for ELM11 interface...{NC}")
elm11_interface_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ELM11-Lua-FFT', 'elm11_interface.py')
if os.path.exists(elm11_interface_path):
    print(f"{GREEN}✓ ELM11 interface found at:{NC}")
    print(f"  {elm11_interface_path}")
else:
    print(f"{YELLOW}⚠ ELM11 interface not found at expected location{NC}")
    print(f"  Looking for: {elm11_interface_path}")
    # Continue anyway - we can still test connection
    print(f"  {BLUE}Continuing with connection test...{NC}")

# Step 4: Try to connect to ELM11
print(f"\n{BLUE}[4/5] Attempting to connect to ELM11...{NC}")
print(f"  Testing each device...")

connected = False
for device in usb_devices:
    print(f"\n  Trying {device}...")
    for baud in [115200, 9600, 19200, 38400, 57600]:
        try:
            print(f"    Baud rate: {baud}...", end='')
            ser = serial.Serial(device, baud, timeout=2)
            time.sleep(2)  # Wait for device to be ready
            
            # Try to clear and get prompt
            ser.read(1024)  # Clear buffer
            ser.write(b'\r\n')
            ser.flush()
            time.sleep(0.5)
            response = ser.read(256)
            
            if response:
                print(f" {GREEN}✓ Response received{NC}")
                print(f"      Device: {device}")
                print(f"      Baud: {baud}")
                print(f"      Response: {response[:50]}...")
                connected = True
                
                # Try to identify as ELM11
                ser.write(b'print("ELM11")\r\n')
                ser.flush()
                time.sleep(0.5)
                identify = ser.read(256)
                
                if b'ELM11' in identify:
                    print(f"      {GREEN}✓ ELM11 identified!{NC}")
                
                ser.close()
                break
            else:
                print(f" {YELLOW}No response{NC}")
                ser.close()
        except Exception as e:
            print(f" {RED}Error: {e}{NC}")
    
    if connected:
        break

if not connected:
    print(f"\n{YELLOW}⚠ Could not establish communication with ELM11{NC}")
    print(f"{YELLOW}  Device is detected but not responding{NC}")
    print(f"{YELLOW}  Check:${NC}")
    print(f"{YELLOW}    - USB cable connection${NC}")
    print(f"{YELLOW}    - Device permissions (may need: sudo chmod 666 {usb_devices[0]}){NC}")
    print(f"{YELLOW}    - Correct baud rate${NC}")

# Step 5: Check for FPGA backend integration
print(f"\n{BLUE}[5/5] Checking for FPGA backend integration...{NC}")
try:
    from pycwt_mod.backends import list_backends, get_backend
    backends = list_backends()
    
    if 'fpga' in backends or 'elm11' in backends:
        print(f"{GREEN}✓ FPGA backend is registered{NC}")
        backend_name = 'fpga' if 'fpga' in backends else 'elm11'
        backend = get_backend(backend_name)
        print(f"  Backend: {backend_name}")
        print(f"  Available: {backend.is_available()}")
    else:
        print(f"{YELLOW}⚠ FPGA backend not yet integrated{NC}")
        print(f"  Available backends: {', '.join(backends)}")
        print(f"  {BLUE}Note: This is expected - FPGA backend integration is Phase 3{NC}")
except Exception as e:
    print(f"{YELLOW}⚠ Could not check backends: {e}{NC}")

# Summary
print(f"\n{BLUE}{'='*60}{NC}")
print(f"{BLUE}Summary{NC}")
print(f"{BLUE}{'='*60}{NC}")

if usb_devices and connected:
    print(f"{GREEN}✓ ELM11 hardware detected and responding{NC}")
    print(f"\n{BLUE}Next steps:{NC}")
    print(f"  1. Test ELM11 interface:")
    print(f"     cd ELM11-Lua-FFT")
    print(f"     python elm11_interface.py")
    print(f"\n  2. Ready for FPGA backend integration (Phase 3)")
elif usb_devices:
    print(f"{YELLOW}⚠ ELM11 hardware detected but not responding{NC}")
    print(f"\n{BLUE}Troubleshooting:{NC}")
    print(f"  sudo chmod 666 {usb_devices[0]}")
    print(f"  python test-elm11.py")
else:
    print(f"{RED}✗ ELM11 hardware not detected{NC}")

print()
