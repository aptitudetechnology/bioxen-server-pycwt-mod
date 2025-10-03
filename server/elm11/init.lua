-- ELM11 FFT System Initialization
-- Common constants and utilities for FFT analysis

-- Configuration constants
SAMPLE_RATE = 48000
BUFFER_SIZE = 1024
FFT_SIZE = 512

-- Signal generation functions
function generate_sine(freq, amp, sample_rate, buffer_size)
    local signal = {}
    for i = 1, buffer_size do
        local t = (i-1) / sample_rate
        signal[i] = amp * math.sin(2 * math.pi * freq * t)
    end
    return signal
end

function generate_square(freq, amp, sample_rate, buffer_size)
    local signal = {}
    for i = 1, buffer_size do
        local t = (i-1) / sample_rate
        signal[i] = amp * (math.sin(2 * math.pi * freq * t) > 0 and 1 or -1)
    end
    return signal
end

function generate_sawtooth(freq, amp, sample_rate, buffer_size)
    local signal = {}
    for i = 1, buffer_size do
        local t = (i-1) / sample_rate
        signal[i] = amp * (2 * (freq * t - math.floor(freq * t + 0.5)))
    end
    return signal
end

function generate_triangle(freq, amp, sample_rate, buffer_size)
    local signal = {}
    for i = 1, buffer_size do
        local t = (i-1) / sample_rate
        signal[i] = amp * (2 * math.abs(2 * (freq * t - math.floor(freq * t + 0.5))) - 1)
    end
    return signal
end

-- FFT computation (simplified)
function compute_fft(signal)
    -- This would be replaced with a proper FFT implementation
    -- For now, return a mock frequency domain representation
    local fft_result = {}
    for i = 1, FFT_SIZE/2 do
        fft_result[i] = {real = 0, imag = 0}
    end
    -- Add some mock peaks for testing
    fft_result[math.floor(440 * FFT_SIZE / SAMPLE_RATE)] = {real = BUFFER_SIZE/2, imag = 0}
    return fft_result
end

-- Fourier series analysis
function get_fourier_series(fft_result, n_harmonics)
    local coeffs = {a0 = 0, a_n = {}, b_n = {}}

    -- Simplified Fourier coefficient extraction
    for n = 1, n_harmonics do
        local idx = n * math.floor(440 * FFT_SIZE / SAMPLE_RATE)
        if idx <= #fft_result then
            coeffs.a_n[n] = fft_result[idx].real / FFT_SIZE * 2
            coeffs.b_n[n] = fft_result[idx].imag / FFT_SIZE * 2
        else
            coeffs.a_n[n] = 0
            coeffs.b_n[n] = 0
        end
    end

    return coeffs
end

-- Signal reconstruction from Fourier coefficients
function reconstruct_signal(n_harmonics)
    local reconstructed = {}
    local fundamental_freq = 440  -- Hz

    for i = 1, BUFFER_SIZE do
        local t = (i-1) / SAMPLE_RATE
        reconstructed[i] = fourier_coeffs.a0 / 2

        for n = 1, n_harmonics do
            reconstructed[i] = reconstructed[i] +
                fourier_coeffs.a_n[n] * math.cos(2 * math.pi * n * fundamental_freq * t) +
                fourier_coeffs.b_n[n] * math.sin(2 * math.pi * n * fundamental_freq * t)
        end
    end

    return reconstructed
end

-- Global state variables (will be initialized by main script)
current_signal = {}
fft_result = {}
fourier_coeffs = {}

print("ELM11 FFT System initialized")