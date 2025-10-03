# ELM11 FPGA Backend

This directory contains the Lua FFT implementation for the ELM11 FPGA microcontroller.

## Files

- `fourier_main.lua` - Main FFT implementation with DFT, FFT, and spectral analysis functions
- `init.lua` - Module initialization and global function exports

## Usage

The FFT functions can be loaded onto the ELM11 microcontroller via serial connection:

```python
import serial
ser = serial.Serial('/dev/ttyUSB1', 115200, timeout=2)

# Load FFT code
with open('server/elm11/fourier_main.lua', 'r') as f:
    fft_code = f.read()
ser.write(fft_code.encode('utf-8'))

# Use FFT functions
ser.write(b'local spectrum = fft({1, 0, -1, 0})\n')
```

## Available Functions

- `fft(signal)` - Fast Fourier Transform
- `dft(signal)` - Discrete Fourier Transform  
- `ifft(spectrum)` - Inverse FFT
- `magnitude_spectrum(spectrum)` - Magnitude spectrum
- `phase_spectrum(spectrum)` - Phase spectrum
- `power_spectrum(spectrum)` - Power spectrum
