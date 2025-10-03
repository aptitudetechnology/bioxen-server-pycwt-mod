-- Minimal FFT Implementation for ELM11
-- Simplified version for microcontroller constraints

-- Complex number operations
local function complex_mult(a, b)
    return {a[1]*b[1] - a[2]*b[2], a[1]*b[2] + a[2]*b[1]}
end

local function complex_add(a, b)
    return {a[1] + b[1], a[2] + b[2]}
end

-- Discrete Fourier Transform (DFT) - most reliable for small signals
function dft(signal)
    local N = #signal
    local result = {}

    for k = 1, N do
        local real, imag = 0, 0
        for n = 1, N do
            local angle = -2 * math.pi * (k-1) * (n-1) / N
            real = real + signal[n] * math.cos(angle)
            imag = imag + signal[n] * math.sin(angle)
        end
        result[k] = {real, imag}
    end

    return result
end

-- Simplified FFT for powers of 2 (Cooley-Tukey)
function fft(signal)
    local N = #signal

    -- For small arrays, just use DFT
    if N <= 8 then
        return dft(signal)
    end

    -- Full FFT for larger arrays
    if N <= 1 then
        return {{signal[1] or 0, 0}}
    end

    -- Split into even and odd
    local even = {}
    local odd = {}
    for i = 1, N, 2 do
        table.insert(even, signal[i])
        if i + 1 <= N then
            table.insert(odd, signal[i + 1])
        end
    end

    -- Recursive FFT
    local fft_even = fft(even)
    local fft_odd = fft(odd)

    -- Combine
    local result = {}
    for k = 1, N/2 do
        local angle = -2 * math.pi * (k-1) / N
        local twiddle = {math.cos(angle), math.sin(angle)}
        local t = complex_mult(twiddle, fft_odd[k])
        result[k] = complex_add(fft_even[k], t)
        result[k + N/2] = complex_add(fft_even[k], {-t[1], -t[2]})
    end

    return result
end

-- Magnitude spectrum
function magnitude_spectrum(spectrum)
    local result = {}
    for i = 1, #spectrum do
        local real, imag = spectrum[i][1], spectrum[i][2]
        result[i] = math.sqrt(real*real + imag*imag)
    end
    return result
end

-- Test function
function test_fft()
    print("ELM11 FFT Test")

    -- Test signal
    local signal = {}
    local N = 4  -- Small size for testing
    for i = 1, N do
        signal[i] = math.sin(2 * math.pi * (i-1) / N)
    end

    print("Input signal:")
    for i, v in ipairs(signal) do
        print(string.format("%.3f", v))
    end

    -- Compute FFT
    local spectrum = fft(signal)

    print("\\nFFT Spectrum:")
    for i, bin in ipairs(spectrum) do
        print(string.format("Bin %d: %.3f + %.3fj", i, bin[1], bin[2]))
    end

    print("FFT test completed")
end

-- Export functions
_G.dft = dft
_G.fft = fft
_G.magnitude_spectrum = magnitude_spectrum
_G.test_fft = test_fft

print("Minimal FFT loaded successfully")