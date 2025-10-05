"""
Test module 1: Sensor Hardware Integration

Tests for BME280 and LTR-559 sensor detection, calibration, and data acquisition.

Status: 🔮 Wishful - These tests define ideal sensor APIs
"""

import pytest


@pytest.mark.wishful
@pytest.mark.sensor
class TestBME280Hardware:
    """Test BME280 environmental sensor integration."""
    
    def test_detect_bme280_sensor(self, test_client):
        """GET /api/v1/sensors/bme280/detect"""
        response = test_client.get("/api/v1/sensors/bme280/detect")
        assert response.status_code == 200
        result = response.json()
        assert "sensor_detected" in result
        assert result["sensor_detected"] is True
        assert "i2c_address" in result
        assert result["i2c_address"] in ["0x76", "0x77"]
        assert "chip_id" in result
    
    def test_read_temperature(self, test_client):
        """GET /api/v1/sensors/bme280/temperature"""
        response = test_client.get("/api/v1/sensors/bme280/temperature")
        assert response.status_code == 200
        result = response.json()
        assert "temperature_celsius" in result
        assert -40 <= result["temperature_celsius"] <= 85
        assert "timestamp" in result
    
    def test_read_humidity(self, test_client):
        """GET /api/v1/sensors/bme280/humidity"""
        response = test_client.get("/api/v1/sensors/bme280/humidity")
        assert response.status_code == 200
        result = response.json()
        assert "humidity_percent" in result
        assert 0 <= result["humidity_percent"] <= 100
        assert "timestamp" in result
    
    def test_read_pressure(self, test_client):
        """GET /api/v1/sensors/bme280/pressure"""
        response = test_client.get("/api/v1/sensors/bme280/pressure")
        assert response.status_code == 200
        result = response.json()
        assert "pressure_hpa" in result
        assert 300 <= result["pressure_hpa"] <= 1100
        assert "altitude_meters" in result
    
    def test_read_all_environmental(self, test_client, sample_bme280_reading):
        """GET /api/v1/sensors/bme280/all"""
        response = test_client.get("/api/v1/sensors/bme280/all")
        assert response.status_code == 200
        result = response.json()
        
        # Should contain all measurements
        assert "temperature_celsius" in result
        assert "humidity_percent" in result
        assert "pressure_hpa" in result
        assert "timestamp" in result
        
        # Validate ranges
        assert -40 <= result["temperature_celsius"] <= 85
        assert 0 <= result["humidity_percent"] <= 100
        assert 300 <= result["pressure_hpa"] <= 1100
    
    def test_configure_sampling_rate(self, test_client):
        """POST /api/v1/sensors/bme280/configure"""
        response = test_client.post("/api/v1/sensors/bme280/configure", json={
            "sampling_rate_hz": 1.0,  # 1 Hz
            "oversampling_temp": 2,
            "oversampling_pressure": 16,
            "oversampling_humidity": 1
        })
        assert response.status_code == 200
    
    def test_sensor_not_detected_error(self, test_client):
        """Test error handling when sensor not present"""
        # This would test error path when sensor is disconnected
        # Expected: 503 Service Unavailable or 404 Not Found
        pass


@pytest.mark.wishful
@pytest.mark.sensor
class TestLTR559Hardware:
    """Test LTR-559 light and proximity sensor integration."""
    
    def test_detect_ltr559_sensor(self, test_client):
        """GET /api/v1/sensors/ltr559/detect"""
        response = test_client.get("/api/v1/sensors/ltr559/detect")
        assert response.status_code == 200
        result = response.json()
        assert result["sensor_detected"] is True
        assert "part_id" in result
        assert result["part_id"] == "0x09"
        assert "i2c_address" in result
        assert result["i2c_address"] == "0x23"
    
    def test_read_light_intensity(self, test_client, sample_ltr559_reading):
        """GET /api/v1/sensors/ltr559/light"""
        response = test_client.get("/api/v1/sensors/ltr559/light")
        assert response.status_code == 200
        result = response.json()
        
        assert "lux" in result
        assert 0.01 <= result["lux"] <= 64000
        assert "ch0" in result  # Visible + IR
        assert "ch1" in result  # IR only
        assert "timestamp" in result
        
        # Ch0 should be >= Ch1 (since Ch0 includes IR)
        assert result["ch0"] >= result["ch1"]
    
    def test_read_proximity(self, test_client):
        """GET /api/v1/sensors/ltr559/proximity"""
        response = test_client.get("/api/v1/sensors/ltr559/proximity")
        assert response.status_code == 200
        result = response.json()
        
        assert "proximity" in result
        assert 0 <= result["proximity"] <= 2047
        assert "distance_cm" in result
        assert "timestamp" in result
    
    def test_configure_light_gain(self, test_client):
        """POST /api/v1/sensors/ltr559/configure"""
        response = test_client.post("/api/v1/sensors/ltr559/configure", json={
            "als_gain": 4,  # 1, 2, 4, 8, 48, 96
            "integration_time_ms": 100,
            "measurement_rate_ms": 500
        })
        assert response.status_code == 200
        result = response.json()
        assert result["configured"] is True
    
    def test_auto_gain_adjustment(self, test_client):
        """POST /api/v1/sensors/ltr559/auto-gain"""
        response = test_client.post("/api/v1/sensors/ltr559/auto-gain", json={
            "enable": True,
            "target_range_percent": 80.0
        })
        assert response.status_code == 200
    
    def test_read_spectrum(self, test_client):
        """GET /api/v1/sensors/ltr559/spectrum"""
        response = test_client.get("/api/v1/sensors/ltr559/spectrum")
        assert response.status_code == 200
        result = response.json()
        
        # Should include visible and IR components
        assert "visible_light_lux" in result
        assert "infrared_light_lux" in result
        assert "ratio" in result  # Ch0/Ch1 ratio


@pytest.mark.wishful
@pytest.mark.sensor
class TestSensorCalibration:
    """Test sensor calibration procedures."""
    
    def test_calibrate_light_sensor_dark(self, test_client):
        """POST /api/v1/sensors/ltr559/calibrate/dark"""
        response = test_client.post("/api/v1/sensors/ltr559/calibrate/dark", json={
            "duration_seconds": 10
        })
        assert response.status_code == 200
        result = response.json()
        
        assert "dark_baseline_lux" in result
        assert result["dark_baseline_lux"] < 1.0
        assert "calibration_timestamp" in result
    
    def test_calibrate_light_sensor_reference(self, test_client):
        """POST /api/v1/sensors/ltr559/calibrate/reference"""
        response = test_client.post("/api/v1/sensors/ltr559/calibrate/reference", json={
            "reference_lux": 1000.0,
            "measurement_duration_seconds": 10
        })
        assert response.status_code == 200
        result = response.json()
        
        assert "calibration_factor" in result
        assert 0.5 <= result["calibration_factor"] <= 2.0
        assert "calibration_applied" in result
    
    def test_verify_temperature_accuracy(self, test_client):
        """POST /api/v1/sensors/bme280/calibrate/verify"""
        response = test_client.post("/api/v1/sensors/bme280/calibrate/verify", json={
            "reference_temperature_celsius": 25.0,
            "tolerance_celsius": 1.0,
            "measurement_duration_seconds": 60
        })
        assert response.status_code == 200
        result = response.json()
        
        assert "within_tolerance" in result
        assert "measured_temperature" in result
        assert "deviation_celsius" in result
    
    def test_calibrate_altitude(self, test_client):
        """POST /api/v1/sensors/bme280/calibrate/altitude"""
        response = test_client.post("/api/v1/sensors/bme280/calibrate/altitude", json={
            "reference_altitude_meters": 0.0,  # Sea level
            "reference_pressure_hpa": 1013.25
        })
        assert response.status_code == 200
        result = response.json()
        assert "calibration_applied" in result
    
    def test_factory_reset_calibration(self, test_client):
        """POST /api/v1/sensors/calibrate/reset"""
        response = test_client.post("/api/v1/sensors/calibrate/reset", json={
            "sensor": "all"
        })
        assert response.status_code == 200


@pytest.mark.wishful
@pytest.mark.sensor
class TestSensorDataQuality:
    """Test sensor data quality and validation."""
    
    def test_validate_sensor_readings_consistency(self, test_client):
        """Check that multiple readings are consistent"""
        readings = []
        for _ in range(10):
            response = test_client.get("/api/v1/sensors/bme280/temperature")
            readings.append(response.json()["temperature_celsius"])
        
        # Temperature shouldn't vary wildly in 10 readings
        import statistics
        std_dev = statistics.stdev(readings)
        assert std_dev < 1.0  # Less than 1°C variation
    
    def test_check_sensor_noise_levels(self, test_client):
        """GET /api/v1/sensors/quality/noise"""
        response = test_client.get("/api/v1/sensors/quality/noise", params={
            "sensor": "bme280",
            "metric": "temperature",
            "samples": 100
        })
        assert response.status_code == 200
        result = response.json()
        
        assert "signal_to_noise_ratio" in result
        assert result["signal_to_noise_ratio"] > 20  # Good SNR
    
    def test_detect_sensor_drift(self, test_client):
        """GET /api/v1/sensors/quality/drift"""
        response = test_client.get("/api/v1/sensors/quality/drift", params={
            "sensor": "ltr559",
            "duration_minutes": 60
        })
        assert response.status_code == 200
        result = response.json()
        
        assert "drift_detected" in result
        assert "drift_rate" in result


@pytest.mark.wishful
@pytest.mark.sensor
class TestMultiSensorIntegration:
    """Test integration of multiple sensors."""
    
    def test_read_all_sensors(self, test_client):
        """GET /api/v1/sensors/all"""
        response = test_client.get("/api/v1/sensors/all")
        assert response.status_code == 200
        result = response.json()
        
        assert "bme280" in result
        assert "ltr559" in result
        assert "timestamp" in result
        
        # BME280 data
        assert "temperature_celsius" in result["bme280"]
        assert "humidity_percent" in result["bme280"]
        assert "pressure_hpa" in result["bme280"]
        
        # LTR-559 data
        assert "lux" in result["ltr559"]
        assert "proximity" in result["ltr559"]
    
    def test_synchronized_sensor_reading(self, test_client):
        """POST /api/v1/sensors/read-synchronized"""
        response = test_client.post("/api/v1/sensors/read-synchronized", json={
            "sensors": ["bme280", "ltr559"],
            "samples": 10,
            "interval_ms": 100
        })
        assert response.status_code == 200
        result = response.json()
        
        assert "synchronized_readings" in result
        assert len(result["synchronized_readings"]) == 10
        
        # All readings should have same timestamp
        for reading in result["synchronized_readings"]:
            assert "timestamp" in reading
            assert "bme280" in reading
            assert "ltr559" in reading
