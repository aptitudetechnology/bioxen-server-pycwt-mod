"""
Test module 2: Circadian Entrainment

Tests for light-dark cycle generation, circadian entrainment validation,
and phase response curves for organisms with circadian clock genes.

Status: 🔮 Wishful - Defines ideal circadian validation APIs
"""

import pytest


@pytest.mark.wishful
@pytest.mark.circadian
class TestLightDarkCycles:
    """Test light-dark cycle generation and control."""
    
    def test_configure_12L_12D_cycle(self, test_client, light_dark_cycle_12_12):
        """POST /api/v1/environment/light-cycle/configure"""
        response = test_client.post(
            "/api/v1/environment/light-cycle/configure",
            json=light_dark_cycle_12_12
        )
        assert response.status_code == 200
        result = response.json()
        
        assert result["cycle_configured"] is True
        assert "cycle_id" in result
        assert result["cycle_type"] == "12L:12D"
    
    def test_configure_16L_8D_cycle(self, test_client):
        """Long day photoperiod (summer simulation)"""
        response = test_client.post("/api/v1/environment/light-cycle/configure", json={
            "cycle_type": "16L:8D",
            "light_intensity_lux": 1000,
            "dark_intensity_lux": 0.1,
            "start_time": "06:00",
            "timezone": "UTC"
        })
        assert response.status_code == 200
        result = response.json()
        assert result["cycle_type"] == "16L:8D"
    
    def test_configure_8L_16D_cycle(self, test_client):
        """Short day photoperiod (winter simulation)"""
        response = test_client.post("/api/v1/environment/light-cycle/configure", json={
            "cycle_type": "8L:16D",
            "light_intensity_lux": 1000,
            "dark_intensity_lux": 0.1
        })
        assert response.status_code == 200
    
    def test_simulate_natural_dawn_dusk(self, test_client):
        """Gradual light transitions (not instantaneous)"""
        response = test_client.post("/api/v1/environment/light-cycle/configure", json={
            "cycle_type": "natural",
            "dawn_duration_minutes": 30,
            "dusk_duration_minutes": 30,
            "peak_lux": 5000,
            "night_lux": 0.01,
            "dawn_start": "06:00",
            "dusk_start": "18:00"
        })
        assert response.status_code == 200
        result = response.json()
        assert result["cycle_type"] == "natural"
    
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
    
    def test_apply_light_cycle_to_vm(self, test_client, yeast_circadian_vm_config):
        """POST /api/v1/vms/{vm_id}/environment/light-cycle"""
        # First create VM
        vm_response = test_client.post("/api/v1/vms", json=yeast_circadian_vm_config)
        vm_id = vm_response.json()["vm_id"]
        
        # Apply light cycle
        response = test_client.post(
            f"/api/v1/vms/{vm_id}/environment/light-cycle",
            json={
                "cycle_id": "cycle_12L12D",
                "duration_hours": 72  # 3 days
            }
        )
        assert response.status_code == 202  # Accepted (async)
        assert "simulation_id" in response.json()
    
    def test_get_current_light_state(self, test_client):
        """GET /api/v1/environment/light-cycle/current"""
        response = test_client.get("/api/v1/environment/light-cycle/current")
        assert response.status_code == 200
        result = response.json()
        
        assert "current_lux" in result
        assert "phase" in result  # "light" or "dark"
        assert "cycle_time" in result
        assert "time_until_transition" in result


@pytest.mark.wishful
@pytest.mark.circadian
class TestCircadianEntrainment:
    """Test circadian clock entrainment to light cycles."""
    
    def test_yeast_entrainment_to_12L_12D(self, test_client, yeast_circadian_vm_config):
        """
        Test yeast VM entrains to 12:12 light-dark cycle.
        Only works for organisms WITH circadian genes!
        """
        # Step 1: Create yeast VM with circadian capability
        vm_response = test_client.post("/api/v1/vms", json=yeast_circadian_vm_config)
        assert vm_response.status_code == 201
        vm_id = vm_response.json()["vm_id"]
        
        # Step 2: Apply light cycle for 3 days
        cycle_response = test_client.post(
            f"/api/v1/vms/{vm_id}/environment/light-cycle",
            json={"cycle_type": "12L:12D", "duration_hours": 72}
        )
        assert cycle_response.status_code == 202
        
        # Step 3: Validate entrainment
        validation = test_client.post(
            f"/api/v1/vms/{vm_id}/validate/circadian-entrainment",
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
        assert "phase_coherence" in result
    
    def test_cyanobacteria_kai_oscillation(self, test_client, cyanobacteria_vm_config):
        """Test Kai protein oscillation in cyanobacteria"""
        vm_response = test_client.post("/api/v1/vms", json=cyanobacteria_vm_config)
        vm_id = vm_response.json()["vm_id"]
        
        # Run in 12L:12D for 3 days
        test_client.post(
            f"/api/v1/vms/{vm_id}/environment/light-cycle",
            json={"cycle_type": "12L:12D", "duration_hours": 72}
        )
        
        # Validate KaiC phosphorylation oscillates
        validation = test_client.post(
            f"/api/v1/vms/{vm_id}/validate/protein-oscillation",
            json={
                "protein": "KaiC",
                "modification": "phosphorylation",
                "expected_period_hours": 24.0
            }
        )
        result = validation.json()
        assert result["oscillation_detected"] is True
    
    def test_entrainment_time_course(self, test_client, yeast_circadian_vm_config):
        """Test how long it takes to entrain from random initial phase"""
        vm_response = test_client.post("/api/v1/vms", json=yeast_circadian_vm_config)
        vm_id = vm_response.json()["vm_id"]
        
        # Start with random phase in constant darkness
        test_client.post(
            f"/api/v1/vms/{vm_id}/environment/constant-dark",
            json={"duration_hours": 48}
        )
        
        # Apply light cycle
        test_client.post(
            f"/api/v1/vms/{vm_id}/environment/light-cycle",
            json={"cycle_type": "12L:12D", "duration_hours": 120}  # 5 days
        )
        
        # Measure entrainment time
        analysis = test_client.get(f"/api/v1/vms/{vm_id}/analysis/entrainment-time")
        result = analysis.json()
        
        assert "entrainment_time_hours" in result
        assert 24 <= result["entrainment_time_hours"] <= 96  # 1-4 days typical


@pytest.mark.wishful
@pytest.mark.circadian
class TestPhaseResponseCurves:
    """Test phase response curves (PRC) - classic circadian experiments."""
    
    def test_phase_shift_light_pulse_at_CT0(self, test_client, yeast_circadian_vm_config):
        """Light pulse at subjective dawn (CT0) - minimal phase shift"""
        vm_response = test_client.post("/api/v1/vms", json=yeast_circadian_vm_config)
        vm_id = vm_response.json()["vm_id"]
        
        # Run in constant darkness to establish free-running rhythm
        test_client.post(
            f"/api/v1/vms/{vm_id}/environment/constant-dark",
            json={"duration_hours": 48}
        )
        
        # Apply light pulse at CT0 (circadian time 0)
        pulse_response = test_client.post(
            f"/api/v1/vms/{vm_id}/environment/light-pulse",
            json={
                "circadian_time": 0.0,
                "pulse_duration_minutes": 60,
                "pulse_intensity_lux": 1000
            }
        )
        assert pulse_response.status_code == 200
        
        # Measure phase shift
        validation = test_client.get(f"/api/v1/vms/{vm_id}/validate/phase-shift")
        result = validation.json()
        
        assert "phase_shift_hours" in result
        assert abs(result["phase_shift_hours"]) < 1.0  # Minimal shift at CT0
    
    def test_phase_delay_light_pulse_at_CT14(self, test_client, yeast_circadian_vm_config):
        """Light pulse at CT14 (early night) - phase delay"""
        vm_response = test_client.post("/api/v1/vms", json=yeast_circadian_vm_config)
        vm_id = vm_response.json()["vm_id"]
        
        test_client.post(
            f"/api/v1/vms/{vm_id}/environment/constant-dark",
            json={"duration_hours": 48}
        )
        
        # Light pulse at CT14
        test_client.post(
            f"/api/v1/vms/{vm_id}/environment/light-pulse",
            json={"circadian_time": 14.0, "pulse_duration_minutes": 60, "pulse_intensity_lux": 1000}
        )
        
        validation = test_client.get(f"/api/v1/vms/{vm_id}/validate/phase-shift")
        result = validation.json()
        
        # Should see phase delay (negative shift)
        assert result["phase_shift_hours"] < -0.5
    
    def test_phase_advance_light_pulse_at_CT22(self, test_client, yeast_circadian_vm_config):
        """Light pulse at CT22 (late night) - phase advance"""
        vm_response = test_client.post("/api/v1/vms", json=yeast_circadian_vm_config)
        vm_id = vm_response.json()["vm_id"]
        
        test_client.post(
            f"/api/v1/vms/{vm_id}/environment/constant-dark",
            json={"duration_hours": 48}
        )
        
        # Light pulse at CT22
        test_client.post(
            f"/api/v1/vms/{vm_id}/environment/light-pulse",
            json={"circadian_time": 22.0, "pulse_duration_minutes": 60, "pulse_intensity_lux": 1000}
        )
        
        validation = test_client.get(f"/api/v1/vms/{vm_id}/validate/phase-shift")
        result = validation.json()
        
        # Should see phase advance (positive shift)
        assert result["phase_shift_hours"] > 0.5
    
    def test_generate_full_prc(self, test_client, yeast_circadian_vm_config):
        """Generate full phase response curve (PRC) across all CTs"""
        response = test_client.post("/api/v1/experiments/phase-response-curve", json={
            "vm_config": yeast_circadian_vm_config,
            "circadian_times": list(range(0, 24, 2)),  # Every 2 hours
            "pulse_duration_minutes": 60,
            "pulse_intensity_lux": 1000,
            "replicate s": 3
        })
        assert response.status_code == 202  # Async experiment
        
        # Get results
        experiment_id = response.json()["experiment_id"]
        results = test_client.get(f"/api/v1/experiments/{experiment_id}/results")
        prc_data = results.json()
        
        assert "circadian_times" in prc_data
        assert "phase_shifts" in prc_data
        assert len(prc_data["circadian_times"]) == 12


@pytest.mark.wishful
@pytest.mark.circadian
class TestFreeRunningPeriod:
    """Test free-running period (tau) in constant conditions."""
    
    def test_measure_tau_in_constant_darkness(self, test_client, yeast_circadian_vm_config):
        """Measure free-running period in DD (constant darkness)"""
        vm_response = test_client.post("/api/v1/vms", json=yeast_circadian_vm_config)
        vm_id = vm_response.json()["vm_id"]
        
        # Run in DD for 5 days
        test_client.post(
            f"/api/v1/vms/{vm_id}/environment/constant-dark",
            json={"duration_hours": 120}
        )
        
        # Analyze free-running period using Fourier
        fourier_response = test_client.post(
            f"/api/v1/vms/{vm_id}/analysis/circadian-period",
            json={"analysis_method": "fourier"}
        )
        assert fourier_response.status_code == 200
        result = fourier_response.json()
        
        assert "free_running_period_hours" in result
        # Should be close to but not exactly 24h
        assert 22.0 <= result["free_running_period_hours"] <= 26.0
        assert result["free_running_period_hours"] != 24.0  # tau ≠ 24
    
    def test_measure_tau_in_constant_light(self, test_client, yeast_circadian_vm_config):
        """Measure free-running period in LL (constant light)"""
        vm_response = test_client.post("/api/v1/vms", json=yeast_circadian_vm_config)
        vm_id = vm_response.json()["vm_id"]
        
        # Run in LL
        test_client.post(
            f"/api/v1/vms/{vm_id}/environment/constant-light",
            json={"duration_hours": 120, "intensity_lux": 100}
        )
        
        analysis = test_client.post(f"/api/v1/vms/{vm_id}/analysis/circadian-period")
        result = analysis.json()
        
        assert "free_running_period_hours" in result
        # Period may be different in LL vs DD
    
    def test_compare_tau_across_genotypes(self, test_client):
        """Compare free-running periods for different clock gene mutations"""
        genotypes = [
            {"name": "wildtype", "genes": ["FRQ", "WC-1", "WC-2"]},
            {"name": "frq_short", "genes": ["FRQ_short", "WC-1", "WC-2"]},
            {"name": "frq_long", "genes": ["FRQ_long", "WC-1", "WC-2"]}
        ]
        
        results = {}
        for genotype in genotypes:
            vm_config = {
                "vm_id": f"yeast_{genotype['name']}",
                "biological_type": "yeast",
                "vm_type": "circadian_capable",
                "genes": genotype["genes"]
            }
            
            vm_response = test_client.post("/api/v1/vms", json=vm_config)
            vm_id = vm_response.json()["vm_id"]
            
            test_client.post(
                f"/api/v1/vms/{vm_id}/environment/constant-dark",
                json={"duration_hours": 120}
            )
            
            analysis = test_client.post(f"/api/v1/vms/{vm_id}/analysis/circadian-period")
            results[genotype["name"]] = analysis.json()["free_running_period_hours"]
        
        # Verify mutations affect period
        assert results["frq_short"] < results["wildtype"]
        assert results["frq_long"] > results["wildtype"]


@pytest.mark.wishful
@pytest.mark.circadian
class TestPhotoperiodExperiments:
    """Test photoperiod manipulation experiments."""
    
    def test_photoperiodic_time_measurement(self, test_client, yeast_circadian_vm_config):
        """Test how organism measures day length"""
        vm_response = test_client.post("/api/v1/vms", json=yeast_circadian_vm_config)
        vm_id = vm_response.json()["vm_id"]
        
        # Test response to different photoperiods
        photoperiods = ["8L:16D", "12L:12D", "16L:8D"]
        
        for photoperiod in photoperiods:
            test_client.post(
                f"/api/v1/vms/{vm_id}/environment/light-cycle",
                json={"cycle_type": photoperiod, "duration_hours": 72}
            )
            
            # Measure output gene expression
            response = test_client.get(f"/api/v1/vms/{vm_id}/genes/CCG-1/expression")
            result = response.json()
            
            assert "expression_level" in result
            # Expression should vary with photoperiod
    
    def test_critical_photoperiod_response(self, test_client):
        """Test critical photoperiod for developmental response"""
        # This would test photoperiodic responses like flowering
        # or seasonal metabolic changes
        pass
