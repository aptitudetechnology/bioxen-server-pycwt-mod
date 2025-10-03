-- ELM11 FFT Interface
-- Lua script for FFT analysis and visualization on ELM11 microcontroller
-- Runs directly on ELM11 hardware with LÖVE2D framework

-- Load required modules
local fft = require("fft")
local signal = require("signal.generator")
local visualization = require("visualization")

-- Configuration
local SAMPLE_RATE = 48000
local BUFFER_SIZE = 1024
local FFT_SIZE = 512

-- Global state
local current_signal = {}
local fft_result = {}
local fourier_coeffs = {}
local display_mode = "time"  -- "time", "freq", "waterfall", "fourier"

-- Initialize hardware
function love.load()
    -- Set up display
    love.window.setTitle("ELM11 FFT Analyzer")
    love.window.setMode(800, 600, {resizable=true})

    -- Initialize GPIO for sensors (if available)
    if pcall(require, "gpio") then
        local gpio = require("gpio")
        gpio.pin_mode(1, gpio.INPUT)  -- ADC input
    end

    -- Generate initial test signal
    current_signal = signal.sine(440, 1.0, SAMPLE_RATE, BUFFER_SIZE)

    -- Initialize visualization
    visualization.init()
end

-- Main update loop
function love.update(dt)
    -- Read sensor data if available
    if pcall(require, "adc") then
        local adc = require("adc")
        -- Read ADC values into current_signal buffer
        for i = 1, BUFFER_SIZE do
            current_signal[i] = adc.read(1) / 4096.0 * 2.0 - 1.0  -- Normalize to -1..1
            love.timer.sleep(1/SAMPLE_RATE)  -- Maintain sample rate
        end
    end

    -- Perform FFT analysis
    fft_result = fft.analyze(current_signal)

    -- Extract Fourier series coefficients
    fourier_coeffs = fft.get_fourier_series(fft_result, 10)
end

-- Main draw function
function love.draw()
    love.graphics.clear(0.1, 0.1, 0.1)  -- Dark background

    if display_mode == "time" then
        draw_time_domain()
    elseif display_mode == "freq" then
        draw_frequency_domain()
    elseif display_mode == "waterfall" then
        draw_waterfall()
    elseif display_mode == "fourier" then
        draw_fourier_series()
    end

    -- Draw UI controls
    draw_ui()

    -- Display performance info
    love.graphics.setColor(1, 1, 1)
    love.graphics.print(string.format("FPS: %.1f", love.timer.getFPS()), 10, 10)
    love.graphics.print(string.format("Mode: %s", display_mode), 10, 30)
end

-- Draw time domain signal
function draw_time_domain()
    love.graphics.setColor(0, 1, 0)  -- Green waveform
    local width, height = love.graphics.getDimensions()
    local plot_height = height * 0.7
    local plot_y = height * 0.1

    for i = 1, #current_signal - 1 do
        local x1 = (i-1) / (#current_signal-1) * width
        local y1 = plot_y + plot_height/2 - current_signal[i] * plot_height/2
        local x2 = i / (#current_signal-1) * width
        local y2 = plot_y + plot_height/2 - current_signal[i+1] * plot_height/2
        love.graphics.line(x1, y1, x2, y2)
    end

    -- Draw grid
    love.graphics.setColor(0.3, 0.3, 0.3)
    for i = 0, 10 do
        local x = i * width / 10
        love.graphics.line(x, plot_y, x, plot_y + plot_height)
    end
    for i = 0, 4 do
        local y = plot_y + i * plot_height / 4
        love.graphics.line(0, y, width, y)
    end
end

-- Draw frequency domain spectrum
function draw_frequency_domain()
    love.graphics.setColor(0, 0, 1)  -- Blue spectrum
    local width, height = love.graphics.getDimensions()
    local plot_height = height * 0.7
    local plot_y = height * 0.1

    -- Calculate magnitudes
    local magnitudes = {}
    for i = 1, #fft_result/2 do
        local real = fft_result[i*2-1]
        local imag = fft_result[i*2]
        magnitudes[i] = math.sqrt(real*real + imag*imag)
    end

    -- Normalize
    local max_mag = 0
    for _, mag in ipairs(magnitudes) do
        if mag > max_mag then max_mag = mag end
    end

    -- Draw spectrum
    for i = 1, #magnitudes - 1 do
        local x1 = (i-1) / (#magnitudes-1) * width
        local y1 = plot_y + plot_height - (magnitudes[i] / max_mag) * plot_height
        local x2 = i / (#magnitudes-1) * width
        local y2 = plot_y + plot_height - (magnitudes[i+1] / max_mag) * plot_height
        love.graphics.line(x1, y1, x2, y2)
    end
end

-- Draw waterfall spectrogram
function draw_waterfall()
    -- Simplified waterfall - in full implementation would maintain history
    love.graphics.setColor(1, 0, 0, 0.5)
    local width, height = love.graphics.getDimensions()
    love.graphics.rectangle("fill", 0, height*0.8, width, height*0.2)
end

-- Draw Fourier series reconstruction
function draw_fourier_series()
    love.graphics.setColor(1, 0, 1)  -- Magenta
    local width, height = love.graphics.getDimensions()
    local plot_height = height * 0.7
    local plot_y = height * 0.1

    -- Reconstruct signal from coefficients
    local reconstructed = {}
    for t = 1, BUFFER_SIZE do
        reconstructed[t] = fourier_coeffs.a0 / 2
        for n = 1, #fourier_coeffs.a_n do
            local freq = n * 440 * 2 * math.pi / SAMPLE_RATE
            reconstructed[t] = reconstructed[t] +
                fourier_coeffs.a_n[n] * math.cos(n * freq * t) +
                fourier_coeffs.b_n[n] * math.sin(n * freq * t)
        end
    end

    -- Draw reconstructed signal
    for i = 1, #reconstructed - 1 do
        local x1 = (i-1) / (#reconstructed-1) * width
        local y1 = plot_y + plot_height/2 - reconstructed[i] * plot_height/2
        local x2 = i / (#reconstructed-1) * width
        local y2 = plot_y + plot_height/2 - reconstructed[i+1] * plot_height/2
        love.graphics.line(x1, y1, x2, y2)
    end
end

-- Draw UI controls
function draw_ui()
    local width, height = love.graphics.getDimensions()
    love.graphics.setColor(0.8, 0.8, 0.8)
    love.graphics.print("Controls:", 10, height - 80)
    love.graphics.print("1: Time Domain", 10, height - 60)
    love.graphics.print("2: Frequency Domain", 10, height - 40)
    love.graphics.print("3: Waterfall", 10, height - 20)
    love.graphics.print("4: Fourier Series", width - 150, height - 20)
end

-- Input handling
function love.keypressed(key)
    if key == "1" then
        display_mode = "time"
    elseif key == "2" then
        display_mode = "freq"
    elseif key == "3" then
        display_mode = "waterfall"
    elseif key == "4" then
        display_mode = "fourier"
    elseif key == "escape" then
        love.event.quit()
    end
end

-- Signal generation functions (for testing)
function generate_sine(freq, amp, sample_rate, samples)
    local signal = {}
    for i = 1, samples do
        signal[i] = amp * math.sin(2 * math.pi * freq * (i-1) / sample_rate)
    end
    return signal
end

function generate_square(freq, amp, sample_rate, samples)
    local signal = {}
    local period = sample_rate / freq
    for i = 1, samples do
        signal[i] = amp * (math.floor(2 * (i-1) / period) % 2 == 0 and 1 or -1)
    end
    return signal
end

-- Export functions for external use
return {
    generate_sine = generate_sine,
    generate_square = generate_square,
    set_display_mode = function(mode) display_mode = mode end,
    get_current_signal = function() return current_signal end,
    get_fft_result = function() return fft_result end,
    get_fourier_coeffs = function() return fourier_coeffs end
}