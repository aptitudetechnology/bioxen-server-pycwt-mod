# Tang Nano 9K Testing Guide

## Overview

This guide covers testing the Tang Nano 9K FPGA board integration with the PyCWT backend system.

## Test Files

### 1. Unit Tests (`test_elm11.py`)

Location: `/src/pycwt_mod/tests/backends/test_elm11.py`

Comprehensive pytest test suite covering:

- **Initialization Tests**: Backend creation, configuration, capabilities
- **Hardware Detection**: Serial port detection, Tang Nano 9K identification
- **Monte Carlo Execution**: Simulation execution, determinism, error handling
- **Backend Registration**: Registry integration, retrieval
- **Tang Nano 9K Specific**: Keyword recognition, multiple device handling
- **Integration Tests**: WCT significance, parameter passing
- **Performance Tests**: Timing comparisons, speedup measurements

**Run the tests:**

```bash
# Run all ELM11/Tang Nano 9K tests
pytest src/pycwt_mod/tests/backends/test_elm11.py -v

# Run specific test class
pytest src/pycwt_mod/tests/backends/test_elm11.py::TestTangNano9KSpecific -v

# Run with output
pytest src/pycwt_mod/tests/backends/test_elm11.py -v -s

# Skip slow tests
pytest src/pycwt_mod/tests/backends/test_elm11.py -v -m "not slow"
```

### 2. Hardware Detection Script (`test-tang-nano-9k.py`)

Location: `/test-tang-nano-9k.py`

Interactive script that:

1. Checks for USB serial devices
2. Validates pyserial installation
3. Identifies Tang Nano 9K devices
4. Tests serial communication
5. Verifies FPGA backend integration
6. Runs basic backend test

**Run the script:**

```bash
python3 test-tang-nano-9k.py
```

## Test Coverage

### Hardware Detection Tests

```python
# Test Tang Nano 9K keyword recognition
def test_tang_nano_9k_keywords()

# Test handling of multiple USB devices
def test_multiple_usb_devices()

# Test common port patterns
def test_common_port_patterns()
```

### Backend Functionality Tests

```python
# Test basic Monte Carlo execution
def test_elm11_basic_execution()

# Test deterministic behavior
def test_elm11_determinism()

# Test with arguments and kwargs
def test_elm11_with_args_kwargs()

# Test error handling
def test_elm11_error_handling()
```

### Integration Tests

```python
# Test with wct_significance
def test_elm11_with_wct_significance()

# Test backend parameter passing
def test_elm11_backend_parameter_passing()
```

### Performance Tests

```python
# Compare ELM11 vs sequential timing
@pytest.mark.slow
def test_elm11_vs_sequential_timing()
```

## Expected Behavior

### When Tang Nano 9K is Connected

```bash
$ python3 test-tang-nano-9k.py

╔════════════════════════════════════════════════════════════╗
║       Tang Nano 9K FPGA - Detection Test                  ║
╚════════════════════════════════════════════════════════════╝

[1/6] Checking for USB serial devices...
✓ Found 2 USB serial device(s):
  - /dev/ttyUSB2
    Description: USB Serial Port
    Manufacturer: Gowin
  - /dev/ttyUSB3
    Description: USB Serial Port
    Manufacturer: Gowin

[2/6] Checking for pyserial library...
✓ pyserial is installed (version 3.5)

[3/6] Looking for Tang Nano 9K devices...
✓ Found Tang Nano 9K device:
  Device: /dev/ttyUSB2
  Description: USB Serial Port (Gowin)

[4/6] Attempting to connect to devices...
  Trying /dev/ttyUSB2...
    Baud rate: 115200... ✓ Response received
      Device: /dev/ttyUSB2
      Baud: 115200
      ✓ Interactive interface detected!

[5/6] Checking for FPGA backend integration...
✓ FPGA backend is registered
  Backend: elm11
  Available: True
✓ FPGA backend is fully operational
  Capabilities: fpga_acceleration, fft_hardware, deterministic, available

[6/6] Running backend test...
  Testing Monte Carlo execution...
✓ Backend test passed
  Executed 5 simulations
  Mean: 0.123
  Std: 0.987

============================================================
Summary
============================================================
✓ USB devices detected: 2
✓ Tang Nano 9K devices found: 1
✓ Device communication successful
  Device: /dev/ttyUSB2
  Baudrate: 115200
✓ FPGA backend fully operational
```

### When Tang Nano 9K is Not Connected

```bash
[1/6] Checking for USB serial devices...
⚠ No USB serial devices found

...

Next steps:
  1. Connect Tang Nano 9K via USB
  2. Re-run this test
```

### When Permissions are Needed

```bash
[4/6] Attempting to connect to devices...
  Trying /dev/ttyUSB2...
    Baud rate: 115200... Permission denied

...

Next steps:
  1. Fix USB permissions:
     sudo chmod 666 /dev/ttyUSB2
     # OR for permanent access:
     sudo usermod -a -G dialout $USER
     # (then logout/login)
  2. Re-run this test
```

## Tang Nano 9K Specific Features

### Detected Keywords

The backend automatically detects Tang Nano 9K devices using these keywords:

- `'tang nano'`
- `'tangnano'`
- `'gowin'` (FPGA manufacturer)
- `'fpga'`

### Common Serial Ports

The backend checks these common port locations:

- Linux: `/dev/ttyUSB0-3`, `/dev/ttyACM0-1`
- Windows: `COM3`, `COM4`

### Supported Baudrates

The detection script tries these baudrates:

- 115200 (default)
- 9600
- 19200
- 38400
- 57600

## Troubleshooting

### No Devices Found

**Issue**: `⚠ No USB serial devices found`

**Solutions**:
1. Check USB cable connection
2. Verify Tang Nano 9K power LED is on
3. Try a different USB port
4. Check if device appears in system:
   ```bash
   ls -la /dev/ttyUSB*
   dmesg | tail  # Check recent USB events
   ```

### Permission Denied

**Issue**: `Permission denied: '/dev/ttyUSB2'`

**Solutions**:
1. Temporary fix:
   ```bash
   sudo chmod 666 /dev/ttyUSB2
   ```

2. Permanent fix:
   ```bash
   sudo usermod -a -G dialout $USER
   # Then logout and login
   ```

3. Verify group membership:
   ```bash
   groups  # Should show 'dialout'
   ```

### Backend Not Available

**Issue**: `⚠ FPGA backend registered but not available`

**Solutions**:
1. Check hardware connection (Step 1)
2. Fix permissions (Step 2)
3. Reinstall package:
   ```bash
   pip install -e .
   ```
4. Verify import:
   ```bash
   python3 -c "from pycwt_mod.backends import get_backend; print(get_backend('elm11'))"
   ```

### Tests Skipped

**Issue**: `SKIPPED [1] test_elm11.py:123: ELM11 hardware not available`

**Expected**: Tests that require hardware will be skipped when Tang Nano 9K is not connected.

**Solutions**:
- Connect hardware to run hardware-dependent tests
- Tests will pass/skip automatically based on availability

## Performance Testing

### Running Performance Tests

```bash
# Run performance tests (marked as slow)
pytest src/pycwt_mod/tests/backends/test_elm11.py -v -m slow

# Run with detailed output
pytest src/pycwt_mod/tests/backends/test_elm11.py::TestELM11PerformanceCharacteristics -v -s
```

### Expected Performance Output

```
test_elm11_vs_sequential_timing ...
Performance comparison:
  Sequential: 2.45s
  ELM11: 0.87s
  Speedup: 2.82×
PASSED
```

## Integration with Main Test Suite

### Add to Phase 2 Tests

The ELM11/Tang Nano 9K tests are automatically included when you run:

```bash
python3 run_phase2_tests.py
```

### Manual Backend Test Run

```bash
# Run all backend tests including ELM11
pytest src/pycwt_mod/tests/backends/ -v

# Run only ELM11 tests
pytest src/pycwt_mod/tests/backends/test_elm11.py -v
```

## Continuous Integration

### Skip Hardware Tests in CI

Hardware-dependent tests will automatically skip in CI environments:

```python
@pytest.mark.skipif(not SERIAL_AVAILABLE, reason="pyserial not installed")
def test_hardware_detection():
    # Will skip in CI without hardware
    pass
```

### Mark Hardware-Dependent Tests

```python
@pytest.mark.hardware  # Custom marker for hardware tests
def test_tang_nano_9k_communication():
    # Only run when --hardware flag is passed
    pass
```

Run with hardware tests:
```bash
pytest -v -m hardware
```

Skip hardware tests:
```bash
pytest -v -m "not hardware"
```

## Test Statistics

- **Total test file**: 450+ lines
- **Test classes**: 8
- **Test functions**: 30+
- **Coverage areas**:
  - Initialization: 5 tests
  - Hardware detection: 5 tests
  - Execution: 4 tests
  - Registration: 3 tests
  - Tang Nano 9K specific: 3 tests
  - Integration: 2 tests
  - Performance: 1 test

All tests include proper error handling, skip conditions, and descriptive assertions.
