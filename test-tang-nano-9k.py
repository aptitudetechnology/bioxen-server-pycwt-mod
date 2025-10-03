#!/usr/bin/env python3
"""
Tang Nano 9K FPGA - Detection and Integration Test

This script tests the Tang Nano 9K FPGA board detection, serial communication,
and integration with the PyCWT backend system.
"""

import sys
import os

# ANSI color codes
RED = '\033[0;31m'
GREEN = '\033[0;32m'
YELLOW = '\033[1;33m'
BLUE = '\033[0;34m'
NC = '\033[0m'  # No Color

def print_header(text):
    """Print a formatted header."""
    print(f"\n{BLUE}╔{'═' * 60}╗{NC}")
    print(f"{BLUE}║{text:^60}║{NC}")
    print(f"{BLUE}╚{'═' * 60}╝{NC}\n")

def print_section(text):
    """Print a section header."""
    print(f"\n{BLUE}[{text}]{NC}")

print_header("Tang Nano 9K FPGA - Detection Test")

# Step 1: Check for USB serial devices
print_section("1/6] Checking for USB serial devices...")
try:
    import serial.tools.list_ports
    ports = list(serial.tools.list_ports.comports())
    usb_devices = [port.device for port in ports]
    
    if usb_devices:
        print(f"{GREEN}✓ Found {len(usb_devices)} USB serial device(s):{NC}")
        for device in usb_devices:
            port_info = next(p for p in ports if p.device == device)
            print(f"  - {device}")
            print(f"    Description: {port_info.description}")
            print(f"    Manufacturer: {port_info.manufacturer if port_info.manufacturer else 'Unknown'}")
    else:
        print(f"{YELLOW}⚠ No USB serial devices found{NC}")
except ImportError:
    print(f"{RED}✗ pyserial library not installed{NC}")
    print(f"  Install with: pip install pyserial")
    sys.exit(1)
except Exception as e:
    print(f"{RED}✗ Error checking devices: {e}{NC}")
    sys.exit(1)

# Step 2: Check for pyserial library
print_section("2/6] Checking for pyserial library...")
try:
    import serial
    print(f"{GREEN}✓ pyserial is installed (version {serial.VERSION}){NC}")
except ImportError:
    print(f"{RED}✗ pyserial is not installed{NC}")
    print(f"  Install with: pip install pyserial")
    sys.exit(1)

# Step 3: Look for Tang Nano 9K specific devices
print_section("3/6] Looking for Tang Nano 9K devices...")
tang_nano_keywords = ['tang nano', 'tangnano', 'gowin', 'fpga', 'sipeed', 'jtag debugger']
tang_nano_devices = []

for port in ports:
    desc_lower = port.description.lower()
    mfr_lower = (port.manufacturer or '').lower()
    # Check both description and manufacturer
    if any(keyword in desc_lower for keyword in tang_nano_keywords) or \
       any(keyword in mfr_lower for keyword in tang_nano_keywords):
        tang_nano_devices.append(port)
        print(f"{GREEN}✓ Found Tang Nano 9K device:{NC}")
        print(f"  Device: {port.device}")
        print(f"  Description: {port.description}")
        if port.manufacturer:
            print(f"  Manufacturer: {port.manufacturer}")

if not tang_nano_devices:
    print(f"{YELLOW}⚠ No Tang Nano 9K devices found by description{NC}")
    print(f"  Will attempt to test all USB serial devices...")

# Step 4: Attempt to connect to devices
print_section("4/6] Attempting to connect to devices...")
connected = False
connected_device = None
connected_baud = None

test_devices = tang_nano_devices if tang_nano_devices else ports

if not test_devices:
    print(f"{YELLOW}⚠ No devices to test{NC}")
else:
    print(f"  Testing {len(test_devices)} device(s)...\n")
    
    for port in test_devices:
        print(f"  Trying {port.device}...")
        
        # Try common baudrates
        for baudrate in [115200, 9600, 19200, 38400, 57600]:
            try:
                print(f"    Baud rate: {baudrate}...", end=' ')
                ser = serial.Serial(port.device, baudrate, timeout=1)
                
                # Try to communicate
                ser.write(b'\n')  # Send newline
                ser.flush()
                
                # Read response
                response = ser.read(100)
                ser.close()
                
                if response:
                    print(f"{GREEN}✓ Response received{NC}")
                    print(f"      Device: {port.device}")
                    print(f"      Baud: {baudrate}")
                    print(f"      Response: {response[:50]}...")
                    
                    # Check if it looks like a REPL or FPGA
                    if any(marker in response for marker in [b'$', b'>', b'#', b'OK']):
                        print(f"      {GREEN}✓ Interactive interface detected!{NC}")
                        connected = True
                        connected_device = port.device
                        connected_baud = baudrate
                        break
                    else:
                        print(f"      {YELLOW}Response received but no interactive interface{NC}")
                else:
                    print(f"{YELLOW}No response{NC}")
                    
            except serial.SerialException as e:
                if "Permission denied" in str(e):
                    print(f"{YELLOW}Permission denied{NC}")
                else:
                    print(f"{YELLOW}Error: {e}{NC}")
            except Exception as e:
                print(f"{YELLOW}Error: {e}{NC}")
        
        if connected:
            break

# Step 5: Check for FPGA backend integration
print_section("5/6] Checking for FPGA backend integration...")
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
    from pycwt_mod.backends import list_backends, get_backend
    
    backends = list_backends()
    
    if 'elm11' in backends:
        print(f"{GREEN}✓ FPGA backend is registered{NC}")
        backend = get_backend('elm11')
        print(f"  Backend: elm11")
        
        available = backend.is_available()
        print(f"  Available: {available}")
        
        if available:
            print(f"{GREEN}✓ FPGA backend is fully operational{NC}")
            
            # Show capabilities
            capabilities = backend.get_capabilities()
            print(f"  Capabilities: {', '.join(capabilities)}")
        else:
            print(f"{YELLOW}⚠ FPGA backend registered but not available{NC}")
            if not connected:
                print(f"  {YELLOW}Reason: No hardware detected{NC}")
            else:
                print(f"  {YELLOW}Reason: Check permissions on {connected_device}{NC}")
    else:
        print(f"{YELLOW}⚠ FPGA backend not registered{NC}")
        print(f"  Available backends: {', '.join(backends)}")
        
except Exception as e:
    print(f"{YELLOW}⚠ Could not check backends: {e}{NC}")

# Step 6: Run quick backend test
print_section("6/6] Running backend test...")
if 'elm11' in backends:
    try:
        backend = get_backend('elm11')
        
        if backend.is_available():
            print(f"  Testing Monte Carlo execution...")
            
            import numpy as np
            
            def test_worker(seed):
                rng = np.random.default_rng(seed)
                return rng.normal()
            
            results = backend.run_monte_carlo(
                test_worker,
                n_simulations=5,
                seed=42,
                verbose=False
            )
            
            print(f"{GREEN}✓ Backend test passed{NC}")
            print(f"  Executed 5 simulations")
            print(f"  Mean: {np.mean(results):.3f}")
            print(f"  Std: {np.std(results):.3f}")
        else:
            print(f"{YELLOW}⚠ Backend not available, skipping execution test{NC}")
            
    except Exception as e:
        print(f"{YELLOW}⚠ Backend test failed: {e}{NC}")
else:
    print(f"{YELLOW}⚠ Backend not registered, skipping test{NC}")

# Summary
print(f"\n{BLUE}{'='*60}{NC}")
print(f"{BLUE}Summary{NC}")
print(f"{BLUE}{'='*60}{NC}")

if usb_devices:
    print(f"{GREEN}✓ USB devices detected: {len(usb_devices)}{NC}")
else:
    print(f"{RED}✗ No USB devices detected{NC}")

if tang_nano_devices:
    print(f"{GREEN}✓ Tang Nano 9K devices found: {len(tang_nano_devices)}{NC}")
else:
    print(f"{YELLOW}⚠ No Tang Nano 9K devices identified by description{NC}")

if connected:
    print(f"{GREEN}✓ Device communication successful{NC}")
    print(f"  Device: {connected_device}")
    print(f"  Baudrate: {connected_baud}")
else:
    print(f"{YELLOW}⚠ No interactive device communication established{NC}")

if 'elm11' in backends:
    backend = get_backend('elm11')
    if backend.is_available():
        print(f"{GREEN}✓ FPGA backend fully operational{NC}")
    else:
        print(f"{YELLOW}⚠ FPGA backend registered but not operational{NC}")
else:
    print(f"{YELLOW}⚠ FPGA backend not registered{NC}")

print(f"\n{BLUE}Next steps:{NC}")

if not usb_devices:
    print(f"  1. {RED}Connect Tang Nano 9K via USB{NC}")
    print(f"  2. Re-run this test")
elif not connected:
    print(f"  1. {YELLOW}Fix USB permissions:{NC}")
    print(f"     sudo chmod 666 {usb_devices[0]}")
    print(f"     # OR for permanent access:")
    print(f"     sudo usermod -a -G dialout $USER")
    print(f"     # (then logout/login)")
    print(f"  2. Re-run this test")
elif 'elm11' not in backends or not backend.is_available():
    print(f"  1. {YELLOW}Ensure package is installed:{NC}")
    print(f"     pip install -e .")
    print(f"  2. Re-run this test")
else:
    print(f"  1. {GREEN}✓ Test wavelet analysis with FPGA:{NC}")
    print(f"     python3 -c \"from pycwt_mod import wct_significance; \\")
    print(f"              sig = wct_significance(0.72, 0.72, 0.25, 0.25, 0.5, 5, \\")
    print(f"                                      mc_count=50, backend='elm11')\"")
    print(f"  2. {GREEN}Run full test suite:{NC}")
    print(f"     pytest src/pycwt_mod/tests/backends/test_elm11.py -v")

print()
