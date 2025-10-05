# BioXen Environmental Sensor Integration API Tests (Wishful Thinking)

**Purpose:** Define comprehensive test suite for BioXen's future environmental sensing APIs that would provide real-time environmental data to biological VMs for circadian entrainment, environmental response validation, and model parameter tuning.

**Status:** 🔮 Aspirational - These are the tests we WOULD run if we had hardware sensor integration  
**Context:** Addresses Claude's critique about circadian clocks being EMERGENT (not imposed) - these sensors would provide environmental INPUT for organisms that DO have circadian systems

---

## 🌡️ Hardware Sensors We Would Integrate

### 1. **BME280 Environmental Sensor**
- **Temperature:** -40°C to +85°C (±1°C accuracy)
- **Pressure:** 300-1100 hPa (±1 hPa accuracy)
- **Humidity:** 0-100% RH (±3% accuracy)
- **Use Case:** Temperature compensation studies, environmental stress responses

### 2. **LTR-559 Light and Proximity Sensor**
- **Light Intensity:** 0.01 to 64,000 lux
- **Spectral Response:** 400-700nm (visible light)
- **Proximity Detection:** 0-10cm range
- **Use Case:** Circadian entrainment, photoperiod studies, light-dark cycle experiments

---

## 🎯 Why These Sensors Matter for Circadian Biology

### Responding to Claude's Critique

**Claude's Key Point:**
> "Circadian clocks are EMERGENT from transcriptional-translational feedback loops. They're the OUTPUT of biochemical reactions, not an INPUT you impose."

**Our Response:**
✅ **Correct!** That's exactly why we need environmental sensors:

1. **Light Input (LTR-559):** 
   - Real circadian systems ENTRAIN to light-dark cycles
   - Light is an environmental INPUT that synchronizes the EMERGENT oscillator
   - This is how biology actually works (cryptochromes, photoreceptors)

2. **Temperature Input (BME280):**
   - Circadian clocks exhibit temperature compensation (Q10 ≈ 1)
   - Testing if our models maintain ~24h periods across 15°C-35°C is validation
   - Temperature changes can phase-shift circadian clocks

3. **Humidity/Pressure (BME280):**
   - Some organisms respond to barometric pressure changes
   - Humidity affects metabolic rates in microorganisms
   - Environmental stress responses can be validated

### What We Would Actually Test

**NOT testing:** "Does the sensor create a circadian clock?" (No - the genes do)

**YES testing:**
- Does light input entrain an EMERGENT clock to 24h?
- Does temperature change affect the period (or stay compensated)?
- Can we validate model predictions against real environmental data?
- Do environmental stress responses match experimental observations?

---

## 📋 Test Structure (5 Modules, 75+ Tests)

### Module 1: `test_sensor_hardware.py`
**Purpose:** Basic hardware detection, calibration, data acquisition

```python
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
    
    def test_read_pressure(self, test_client):
        """GET /api/v1/sensors/bme280/pressure"""
        response = test_client.get("/api/v1/sensors/bme280/pressure")
        assert response.status_code == 200
        result = response.json()
        assert "pressure_hpa" in result
        assert 300 <= result["pressure_hpa"] <= 1100
    
    def test_read_all_environmental(self, test_client):
        """GET /api/v1/sensors/bme280/all"""
        response = test_client.get("/api/v1/sensors/bme280/all")
        assert response.status_code == 200
        result = response.json()
        assert "temperature_celsius" in result
        assert "humidity_percent" in result
        assert "pressure_hpa" in result
        assert "timestamp" in result

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
    
    def test_read_light_intensity(self, test_client):
        """GET /api/v1/sensors/ltr559/light"""
        response = test_client.get("/api/v1/sensors/ltr559/light")
        assert response.status_code == 200
        result = response.json()
        assert "lux" in result
        assert 0.01 <= result["lux"] <= 64000
        assert "ch0" in result  # Visible + IR
        assert "ch1" in result  # IR only
        assert "timestamp" in result
    
    def test_read_proximity(self, test_client):
        """GET /api/v1/sensors/ltr559/proximity"""
        response = test_client.get("/api/v1/sensors/ltr559/proximity")
        assert response.status_code == 200
        result = response.json()
        assert "proximity" in result
        assert 0 <= result["proximity"] <= 2047
    
    def test_configure_light_gain(self, test_client):
        """POST /api/v1/sensors/ltr559/configure"""
        response = test_client.post("/api/v1/sensors/ltr559/configure", json={
            "als_gain": 4,  # 1, 2, 4, 8, 48, 96
            "integration_time": 100  # milliseconds
        })
        assert response.status_code == 200

class TestSensorCalibration:
    """Test sensor calibration procedures."""
    
    def test_calibrate_light_sensor_dark(self, test_client):
        """POST /api/v1/sensors/ltr559/calibrate/dark"""
        response = test_client.post("/api/v1/sensors/ltr559/calibrate/dark")
        assert response.status_code == 200
        result = response.json()
        assert "dark_baseline_lux" in result
        assert result["dark_baseline_lux"] < 1.0
    
    def test_calibrate_light_sensor_reference(self, test_client):
        """POST /api/v1/sensors/ltr559/calibrate/reference"""
        response = test_client.post("/api/v1/sensors/ltr559/calibrate/reference", json={
            "reference_lux": 1000.0
        })
        assert response.status_code == 200
        result = response.json()
        assert "calibration_factor" in result
    
    def test_verify_temperature_accuracy(self, test_client):
        """POST /api/v1/sensors/bme280/calibrate/verify"""
        response = test_client.post("/api/v1/sensors/bme280/calibrate/verify", json={
            "reference_temperature_celsius": 25.0,
            "tolerance_celsius": 1.0
        })
        assert response.status_code == 200
        result = response.json()
        assert "within_tolerance" in result
```

---

### Module 2: `test_environmental_streaming.py`
**Purpose:** Continuous environmental data streaming to VMs

```python
class TestEnvironmentalDataStreaming:
    """Test real-time environmental data streaming."""
    
    def test_start_environmental_monitoring(self, test_client):
        """POST /api/v1/sensors/monitor/start"""
        response = test_client.post("/api/v1/sensors/monitor/start", json={
            "sensors": ["bme280", "ltr559"],
            "sampling_interval_seconds": 60,
            "duration_hours": 24
        })
        assert response.status_code == 202
        result = response.json()
        assert "monitoring_id" in result
        assert result["status"] == "started"
    
    def test_get_environmental_history(self, test_client):
        """GET /api/v1/sensors/monitor/{monitor_id}/history"""
        response = test_client.get("/api/v1/sensors/monitor/mon_001/history")
        assert response.status_code == 200
        result = response.json()
        assert "timestamps" in result
        assert "temperature_celsius" in result
        assert "light_lux" in result
        assert "humidity_percent" in result
        assert len(result["timestamps"]) > 0
    
    def test_stream_to_vm(self, test_client):
        """POST /api/v1/vms/{vm_id}/sensors/attach"""
        response = test_client.post("/api/v1/vms/yeast_001/sensors/attach", json={
            "sensors": ["bme280", "ltr559"],
            "update_interval_seconds": 300  # Update VM every 5 min
        })
        assert response.status_code == 200
        result = response.json()
        assert result["sensors_attached"] is True
    
    def test_get_vm_environmental_state(self, test_client):
        """GET /api/v1/vms/{vm_id}/environment"""
        response = test_client.get("/api/v1/vms/yeast_001/environment")
        assert response.status_code == 200
        result = response.json()
        assert "current_temperature_celsius" in result
        assert "current_light_lux" in result
        assert "current_humidity_percent" in result
        assert "last_update_timestamp" in result

class TestEnvironmentalAlerts:
    """Test environmental condition alerts."""
    
    def test_configure_temperature_alert(self, test_client):
        """POST /api/v1/sensors/alerts/configure"""
        response = test_client.post("/api/v1/sensors/alerts/configure", json={
            "alert_type": "temperature_out_of_range",
            "min_celsius": 20.0,
            "max_celsius": 30.0,
            "vm_id": "yeast_001"
        })
        assert response.status_code == 200
    
    def test_configure_light_dark_transition_alert(self, test_client):
        """Alert when light-dark transition occurs"""
        response = test_client.post("/api/v1/sensors/alerts/configure", json={
            "alert_type": "light_dark_transition",
            "threshold_lux": 10.0,
            "transition_direction": "dark_to_light"
        })
        assert response.status_code == 200
    
    def test_get_active_environmental_alerts(self, test_client):
        """GET /api/v1/sensors/alerts/active"""
        response = test_client.get("/api/v1/sensors/alerts/active")
        assert response.status_code == 200
        alerts = response.json()
        assert isinstance(alerts, list)
```

---

### Module 3: `test_circadian_entrainment.py`
**Purpose:** Light-dark cycle entrainment for organisms with circadian systems

```python
class TestLightDarkCycles:
    """Test light-dark cycle generation and control."""
    
    def test_configure_light_dark_cycle(self, test_client):
        """POST /api/v1/environment/light-cycle/configure"""
        response = test_client.post("/api/v1/environment/light-cycle/configure", json={
            "cycle_type": "12L:12D",  # 12h light, 12h dark
            "light_intensity_lux": 1000,
            "dark_intensity_lux": 0.1,
            "start_time": "08:00",
            "timezone": "UTC"
        })
        assert response.status_code == 200
        result = response.json()
        assert result["cycle_configured"] is True
    
    def test_simulate_natural_dawn_dusk(self, test_client):
        """Gradual light transitions (not instantaneous)"""
        response = test_client.post("/api/v1/environment/light-cycle/configure", json={
            "cycle_type": "natural",
            "dawn_duration_minutes": 30,
            "dusk_duration_minutes": 30,
            "peak_lux": 5000,
            "night_lux": 0.01
        })
        assert response.status_code == 200
    
    def test_apply_light_cycle_to_vm(self, test_client):
        """POST /api/v1/vms/{vm_id}/environment/light-cycle"""
        response = test_client.post("/api/v1/vms/yeast_001/environment/light-cycle", json={
            "cycle_id": "cycle_12L12D",
            "duration_hours": 72  # 3 days
        })
        assert response.status_code == 202
        assert "simulation_id" in response.json()

class TestCircadianEntrainment:
    """Test circadian clock entrainment to light cycles."""
    
    def test_yeast_entrainment_to_12L12D(self, test_client):
        """Test yeast VM entrains to 12:12 light-dark cycle"""
        # This is for organisms that HAVE circadian genes!
        
        # Step 1: Create yeast VM with circadian genes
        vm_response = test_client.post("/api/v1/vms", json={
            "vm_id": "yeast_circadian",
            "biological_type": "yeast",
            "vm_type": "circadian_capable",
            "genes": ["FRQ", "WC-1", "WC-2"]  # Neurospora homologs
        })
        assert vm_response.status_code == 201
        
        # Step 2: Apply light cycle
        cycle_response = test_client.post(
            "/api/v1/vms/yeast_circadian/environment/light-cycle",
            json={"cycle_type": "12L:12D", "duration_hours": 72}
        )
        assert cycle_response.status_code == 202
        
        # Step 3: Validate entrainment after 3 days
        validation = test_client.post(
            "/api/v1/vms/yeast_circadian/validate/circadian-entrainment",
            json={
                "expected_period_hours": 24.0,
                "expected_phase_hours": 12.0,  # Peak at midday
                "tolerance_hours": 1.0
            }
        )
        assert validation.status_code == 200
        result = validation.json()
        assert result["entrained"] is True
        assert abs(result["detected_period_hours"] - 24.0) < 1.0
    
    def test_phase_shift_response_to_light_pulse(self, test_client):
        """Test phase response curve (PRC) - classic circadian experiment"""
        # Run VM in constant darkness for 2 days
        test_client.post("/api/v1/vms/yeast_circadian/environment/constant-dark", json={
            "duration_hours": 48
        })
        
        # Apply 1-hour light pulse at CT12 (circadian time 12)
        pulse_response = test_client.post(
            "/api/v1/vms/yeast_circadian/environment/light-pulse",
            json={
                "circadian_time": 12.0,
                "pulse_duration_minutes": 60,
                "pulse_intensity_lux": 1000
            }
        )
        assert pulse_response.status_code == 200
        
        # Measure phase shift
        validation = test_client.get(
            "/api/v1/vms/yeast_circadian/validate/phase-shift"
        )
        result = validation.json()
        assert "phase_shift_hours" in result
        # Should see delay (phase shift) at CT12
    
    def test_free_running_period_in_constant_conditions(self, test_client):
        """Measure free-running period (tau) in constant darkness"""
        # Run VM in DD (constant darkness)
        test_client.post("/api/v1/vms/yeast_circadian/environment/constant-dark", json={
            "duration_hours": 120  # 5 days
        })
        
        # Analyze free-running period
        fourier_response = test_client.post(
            "/api/v1/vms/yeast_circadian/analysis/circadian-period"
        )
        assert fourier_response.status_code == 200
        result = fourier_response.json()
        assert "free_running_period_hours" in result
        # Should be close to 24h but not exactly (tau ≠ 24.0)
        assert 22.0 <= result["free_running_period_hours"] <= 26.0

class TestPhotoperiodExperiments:
    """Test photoperiod manipulation experiments."""
    
    def test_short_day_photoperiod(self, test_client):
        """8L:16D (short day) - winter simulation"""
        response = test_client.post("/api/v1/environment/light-cycle/configure", json={
            "cycle_type": "8L:16D",
            "light_intensity_lux": 1000
        })
        assert response.status_code == 200
    
    def test_long_day_photoperiod(self, test_client):
        """16L:8D (long day) - summer simulation"""
        response = test_client.post("/api/v1/environment/light-cycle/configure", json={
            "cycle_type": "16L:8D",
            "light_intensity_lux": 1000
        })
        assert response.status_code == 200
    
    def test_skeleton_photoperiod(self, test_client):
        """Skeleton photoperiod: 1h light, 10h dark, 1h light, 12h dark"""
        response = test_client.post("/api/v1/environment/light-cycle/configure", json={
            "cycle_type": "skeleton",
            "light_pulses": [
                {"start_hour": 0, "duration_hours": 1, "intensity_lux": 1000},
                {"start_hour": 11, "duration_hours": 1, "intensity_lux": 1000}
            ]
        })
        assert response.status_code == 200
```

---

### Module 4: `test_temperature_compensation.py`
**Purpose:** Temperature compensation validation (Q10 studies)

```python
class TestTemperatureCompensation:
    """Test circadian period temperature compensation."""
    
    def test_period_at_15_celsius(self, test_client):
        """Measure circadian period at 15°C"""
        # Set temperature
        test_client.post("/api/v1/vms/yeast_circadian/environment/temperature", json={
            "temperature_celsius": 15.0,
            "duration_hours": 72
        })
        
        # Measure period
        period_response = test_client.get(
            "/api/v1/vms/yeast_circadian/analysis/circadian-period"
        )
        result = period_response.json()
        period_15c = result["period_hours"]
        assert 22.0 <= period_15c <= 26.0
    
    def test_period_at_25_celsius(self, test_client):
        """Measure circadian period at 25°C"""
        test_client.post("/api/v1/vms/yeast_circadian/environment/temperature", json={
            "temperature_celsius": 25.0,
            "duration_hours": 72
        })
        
        period_response = test_client.get(
            "/api/v1/vms/yeast_circadian/analysis/circadian-period"
        )
        result = period_response.json()
        period_25c = result["period_hours"]
        assert 22.0 <= period_25c <= 26.0
    
    def test_period_at_35_celsius(self, test_client):
        """Measure circadian period at 35°C"""
        test_client.post("/api/v1/vms/yeast_circadian/environment/temperature", json={
            "temperature_celsius": 35.0,
            "duration_hours": 72
        })
        
        period_response = test_client.get(
            "/api/v1/vms/yeast_circadian/analysis/circadian-period"
        )
        result = period_response.json()
        period_35c = result["period_hours"]
        assert 22.0 <= period_35c <= 26.0
    
    def test_calculate_q10_coefficient(self, test_client):
        """Calculate Q10 for circadian period - should be ~1.0"""
        response = test_client.post("/api/v1/vms/yeast_circadian/validate/temperature-compensation", json={
            "temperatures_celsius": [15, 25, 35],
            "expected_q10": 1.0,
            "tolerance": 0.2
        })
        assert response.status_code == 200
        result = response.json()
        assert "q10" in result
        assert 0.8 <= result["q10"] <= 1.2  # Temperature compensated
        assert result["is_temperature_compensated"] is True

class TestTemperatureStressResponses:
    """Test heat shock and cold shock responses."""
    
    def test_heat_shock_response(self, test_client):
        """Sudden temperature increase from 25°C to 42°C"""
        # Apply heat shock
        shock_response = test_client.post("/api/v1/vms/ecoli_001/environment/temperature-shock", json={
            "baseline_celsius": 25.0,
            "shock_celsius": 42.0,
            "shock_duration_minutes": 30
        })
        assert shock_response.status_code == 200
        
        # Validate heat shock protein upregulation
        validation = test_client.post("/api/v1/vms/ecoli_001/validate/gene-expression", json={
            "genes": ["dnaK", "dnaJ", "groEL"],  # Heat shock proteins
            "expected_fold_change": 10.0,
            "tolerance": 5.0
        })
        assert validation.status_code == 200
        result = validation.json()
        assert result["genes_upregulated"] is True
    
    def test_cold_shock_response(self, test_client):
        """Sudden temperature decrease from 37°C to 15°C"""
        shock_response = test_client.post("/api/v1/vms/ecoli_001/environment/temperature-shock", json={
            "baseline_celsius": 37.0,
            "shock_celsius": 15.0,
            "shock_duration_minutes": 60
        })
        assert shock_response.status_code == 200
        
        # Validate cold shock protein expression
        validation = test_client.post("/api/v1/vms/ecoli_001/validate/gene-expression", json={
            "genes": ["cspA", "cspB"],  # Cold shock proteins
            "expected_fold_change": 50.0
        })
        result = validation.json()
        assert result["genes_upregulated"] is True

class TestTemperatureCycles:
    """Test circadian temperature cycles (zeitgebers)."""
    
    def test_temperature_cycle_entrainment(self, test_client):
        """Test entrainment to 12h warm:12h cool cycles"""
        response = test_client.post("/api/v1/vms/yeast_circadian/environment/temperature-cycle", json={
            "cycle_type": "12W:12C",
            "warm_celsius": 28.0,
            "cool_celsius": 22.0,
            "duration_hours": 72
        })
        assert response.status_code == 202
        
        # Validate entrainment
        validation = test_client.post(
            "/api/v1/vms/yeast_circadian/validate/circadian-entrainment",
            json={"zeitgeber": "temperature", "expected_period_hours": 24.0}
        )
        result = validation.json()
        assert result["entrained"] is True
```

---

### Module 5: `test_environmental_model_validation.py`
**Purpose:** Validate model predictions against real environmental data

```python
class TestRealWorldDataValidation:
    """Test VM predictions against actual sensor data."""
    
    def test_validate_atp_levels_vs_temperature(self, test_client):
        """Validate ATP prediction matches temperature-dependent metabolism"""
        # Get actual temperature from BME280
        temp_data = test_client.get("/api/v1/sensors/bme280/temperature").json()
        actual_temp = temp_data["temperature_celsius"]
        
        # Run VM at that temperature
        test_client.post("/api/v1/vms/ecoli_001/environment/temperature", json={
            "temperature_celsius": actual_temp,
            "duration_hours": 2
        })
        
        # Validate ATP levels are reasonable for that temperature
        validation = test_client.post("/api/v1/vms/ecoli_001/validate/metabolite-levels", json={
            "metabolite": "ATP",
            "reference_data": {"temperature": actual_temp},
            "expected_range": [80, 120]
        })
        result = validation.json()
        assert result["within_expected_range"] is True
    
    def test_validate_growth_rate_vs_light_intensity(self, test_client):
        """For photosynthetic organisms (e.g., cyanobacteria)"""
        # Get actual light from LTR-559
        light_data = test_client.get("/api/v1/sensors/ltr559/light").json()
        actual_lux = light_data["lux"]
        
        # Simulate cyanobacteria at that light intensity
        test_client.post("/api/v1/vms/cyano_001/environment/light", json={
            "light_intensity_lux": actual_lux,
            "duration_hours": 6
        })
        
        # Validate growth rate
        validation = test_client.get("/api/v1/vms/cyano_001/validate/growth-rate")
        result = validation.json()
        assert "growth_rate_per_hour" in result
        # Light-dependent growth should correlate with lux
    
    def test_compare_simulated_vs_real_environmental_response(self, test_client):
        """Compare VM response to actual organism response (if available)"""
        # Replay historical environmental data
        historical_data = {
            "timestamps": [0, 3600, 7200, 10800],
            "temperature": [25.0, 26.5, 28.0, 27.0],
            "light_lux": [1000, 1200, 800, 600]
        }
        
        response = test_client.post("/api/v1/vms/yeast_001/simulate/replay-environment", json={
            "environmental_data": historical_data,
            "track_metabolites": ["ATP", "glucose"]
        })
        assert response.status_code == 202
        
        # Compare to experimental reference data (if available)
        validation = test_client.post("/api/v1/vms/yeast_001/validate/against-experimental", json={
            "experimental_atp": [100, 95, 90, 92],
            "tolerance_percent": 15
        })
        result = validation.json()
        assert "rmse" in result
        assert "correlation" in result

class TestEnvironmentalParameterTuning:
    """Test parameter tuning based on environmental responses."""
    
    def test_tune_temperature_sensitivity_parameters(self, test_client):
        """Adjust model parameters based on temperature response data"""
        response = test_client.post("/api/v1/vms/ecoli_001/tune/temperature-sensitivity", json={
            "observed_growth_rates": [
                {"temperature_celsius": 20, "growth_rate": 0.3},
                {"temperature_celsius": 30, "growth_rate": 0.8},
                {"temperature_celsius": 37, "growth_rate": 1.2},
                {"temperature_celsius": 42, "growth_rate": 0.4}
            ],
            "optimize": "activation_energy"
        })
        assert response.status_code == 200
        result = response.json()
        assert "suggested_activation_energy" in result
        assert "optimized_parameters" in result
    
    def test_tune_light_response_parameters(self, test_client):
        """Adjust photosynthetic parameters based on light response"""
        response = test_client.post("/api/v1/vms/cyano_001/tune/light-response", json={
            "observed_photosynthesis_rates": [
                {"light_lux": 100, "ps_rate": 5},
                {"light_lux": 1000, "ps_rate": 50},
                {"light_lux": 10000, "ps_rate": 80},
                {"light_lux": 50000, "ps_rate": 85}  # Saturation
            ],
            "optimize": "light_saturation_curve"
        })
        result = response.json()
        assert "suggested_vmax" in result
        assert "suggested_km" in result

class TestLongTermEnvironmentalMonitoring:
    """Test multi-day environmental monitoring and validation."""
    
    def test_30_day_circadian_tracking(self, test_client):
        """Track circadian period stability over 30 days"""
        response = test_client.post("/api/v1/vms/yeast_circadian/monitor/long-term", json={
            "duration_days": 30,
            "validation_interval_hours": 24,
            "track_metrics": ["circadian_period", "amplitude", "phase"]
        })
        assert response.status_code == 202
        
        # Get summary after 30 days
        summary = test_client.get("/api/v1/vms/yeast_circadian/monitor/long-term/summary")
        result = summary.json()
        assert "mean_period_hours" in result
        assert "period_stability_cv" in result  # Coefficient of variation
        assert result["period_stability_cv"] < 0.05  # Stable <5% variation
    
    def test_seasonal_simulation(self, test_client):
        """Simulate seasonal photoperiod changes"""
        response = test_client.post("/api/v1/environment/seasonal-simulation", json={
            "latitude": 40.7,  # NYC
            "start_date": "2025-03-20",  # Spring equinox
            "duration_days": 365,
            "time_compression": 365  # 1 year in 1 day
        })
        assert response.status_code == 202
```

---

## 🎯 Test Coverage Goals

| API Category | Test Count | Coverage Target |
|-------------|-----------|-----------------|
| Sensor Hardware | 15 tests | 100% sensor APIs |
| Environmental Streaming | 10 tests | 90% data flow |
| Circadian Entrainment | 15 tests | 100% light protocols |
| Temperature Compensation | 12 tests | 95% temp studies |
| Environmental Validation | 13 tests | 90% model validation |
| **Total** | **75 tests** | **95% overall** |

---

## 🔬 Addressing Claude's Circadian Critique

### What We're NOT Claiming:
- ❌ Sensors "create" circadian rhythms
- ❌ VMs have clock signals like computers
- ❌ All organisms need circadian clocks

### What We ARE Testing:
- ✅ **Entrainment:** For organisms with clock genes, light input synchronizes EMERGENT oscillators
- ✅ **Temperature compensation:** Q10 ≈ 1 for circadian systems (not metabolic reactions)
- ✅ **Model validation:** Do simulated responses match biological reality?
- ✅ **Parameter tuning:** Adjust kinetic constants based on environmental response data

### Biologically Sound Use Cases:

**1. Neurospora crassa (fungus) - HAS circadian genes:**
```python
vm = create_bio_vm('neurospora', genes=['frq', 'wc-1', 'wc-2'])
# Apply light-dark cycles via LTR-559
# Validate: Does FRQ protein oscillate with ~22h period?
# Test: Does light pulse at CT12 cause phase delay?
```

**2. Cyanobacteria - SIMPLEST circadian system:**
```python
vm = create_bio_vm('synechococcus', genes=['kaiA', 'kaiB', 'kaiC'])
# KaiC phosphorylation oscillates with 24h period
# Validate against temperature changes (Q10 test)
```

**3. E. coli - NO circadian clock:**
```python
vm = create_bio_vm('ecoli')  # No clock genes
# But: metabolic oscillations CAN occur
# Use sensors for temperature stress responses (heat shock)
# NOT circadian validation
```

**4. Syn3A - NO circadian clock, but temperature matters:**
```python
vm = create_bio_vm('syn3a')
# No circadian tests
# But: BME280 validates temperature-dependent growth rates
# Validate model matches Arrhenius kinetics
```

---

## 📊 Hardware Setup (If We Built This)

```bash
# Raspberry Pi with sensors

# Install sensor libraries
pip install smbus2 bme280 ltr559

# I2C Configuration
sudo raspi-config  # Enable I2C
sudo i2cdetect -y 1

# Expected addresses:
# BME280: 0x76 or 0x77
# LTR-559: 0x23

# Test sensors
python3 -c "import bme280; print(bme280.sample())"
python3 -c "import ltr559; print(ltr559.get_lux())"
```

---

## 🚀 When Would We Implement This?

**Phase 6+:** After Phase 5 REST API implementation:
1. Add hardware sensor support to BioXen server
2. Implement environmental data streaming
3. Build light-dark cycle controllers
4. Run these tests for hardware integration validation

**Current Reality:**
- ❌ No hardware sensor integration
- ❌ No environmental control systems
- ❌ VMs don't respond to real-time environmental data

**Future Vision:**
- ✅ Real sensors provide INPUT for biological validation
- ✅ Test entrainment for organisms that HAVE clock genes
- ✅ Validate temperature compensation (Q10 studies)
- ✅ Compare model predictions to real environmental responses

---

## 📚 Scientific References

**Circadian Biology (The Right Way):**
- Dunlap JC (1999) "Molecular Bases for Circadian Clocks" - *Cell*
- Takahashi JS (2017) "Transcriptional architecture of the mammalian circadian clock" - *Nat Rev Genet*
- Nakajima M et al. (2005) "Reconstitution of Circadian Oscillation of Cyanobacterial KaiC" - *Science*

**Temperature Compensation:**
- Ruoff P et al. (2005) "The Goodwin Oscillator and Temperature Compensation" - *J Theor Biol*

**Phase Response Curves (PRC):**
- Johnson CH (1999) "Forty years of PRCs" - *Chronobiol Int*

---

## ✅ Key Takeaway

These tests assume organisms **have circadian genes** and use sensors to provide **environmental INPUT** that affects **EMERGENT oscillations**. This is biologically sound and addresses Claude's critique:

> "Circadian rhythms are EMERGENT from biochemical feedback loops. Light is an INPUT that synchronizes the EMERGENT oscillator."

**That's exactly what these tests validate.**

---

**Status:** 🔮 Wishful thinking  
**Next:** Build actual sensor integration (Phase 6+)  
**References:** `circadian-clock-claude-thoughts.md`, `REFRAMING_COMPLETE.md`
