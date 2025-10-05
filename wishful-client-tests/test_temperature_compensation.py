"""
Test module 3: Temperature Compensation

Tests for circadian period temperature compensation (Q10 ≈ 1),
heat shock responses, and temperature cycle entrainment.

Status: 🔮 Wishful - Defines ideal temperature validation APIs
"""

import pytest


@pytest.mark.wishful
@pytest.mark.circadian
class TestTemperatureCompensation:
    """Test circadian period temperature compensation (Q10 studies)."""
    
    def test_measure_period_at_15_celsius(self, test_client, yeast_circadian_vm_config):
        """Measure circadian period at 15°C"""
        vm_response = test_client.post("/api/v1/vms", json=yeast_circadian_vm_config)
        vm_id = vm_response.json()["vm_id"]
        
        # Set temperature to 15°C
        test_client.post(
            f"/api/v1/vms/{vm_id}/environment/temperature",
            json={"temperature_celsius": 15.0, "duration_hours": 72}
        )
        
        # Measure period
        period_response = test_client.get(
            f"/api/v1/vms/{vm_id}/analysis/circadian-period"
        )
        result = period_response.json()
        
        assert "period_hours" in result
        period_15c = result["period_hours"]
        assert 22.0 <= period_15c <= 26.0
        
        return period_15c
    
    def test_measure_period_at_25_celsius(self, test_client, yeast_circadian_vm_config):
        """Measure circadian period at 25°C"""
        vm_response = test_client.post("/api/v1/vms", json=yeast_circadian_vm_config)
        vm_id = vm_response.json()["vm_id"]
        
        test_client.post(
            f"/api/v1/vms/{vm_id}/environment/temperature",
            json={"temperature_celsius": 25.0, "duration_hours": 72}
        )
        
        period_response = test_client.get(f"/api/v1/vms/{vm_id}/analysis/circadian-period")
        result = period_response.json()
        period_25c = result["period_hours"]
        assert 22.0 <= period_25c <= 26.0
        
        return period_25c
    
    def test_measure_period_at_35_celsius(self, test_client, yeast_circadian_vm_config):
        """Measure circadian period at 35°C"""
        vm_response = test_client.post("/api/v1/vms", json=yeast_circadian_vm_config)
        vm_id = vm_response.json()["vm_id"]
        
        test_client.post(
            f"/api/v1/vms/{vm_id}/environment/temperature",
            json={"temperature_celsius": 35.0, "duration_hours": 72}
        )
        
        period_response = test_client.get(f"/api/v1/vms/{vm_id}/analysis/circadian-period")
        result = period_response.json()
        period_35c = result["period_hours"]
        assert 22.0 <= period_35c <= 26.0
        
        return period_35c
    
    def test_calculate_q10_coefficient(self, test_client, yeast_circadian_vm_config):
        """
        Calculate Q10 for circadian period across temperature range.
        Q10 should be ~1.0 for temperature-compensated circadian clocks.
        """
        response = test_client.post(
            "/api/v1/experiments/temperature-compensation",
            json={
                "vm_config": yeast_circadian_vm_config,
                "temperatures_celsius": [15, 25, 35],
                "duration_per_temp_hours": 72,
                "expected_q10": 1.0,
                "tolerance": 0.2
            }
        )
        assert response.status_code == 202  # Async experiment
        
        experiment_id = response.json()["experiment_id"]
        results = test_client.get(f"/api/v1/experiments/{experiment_id}/results")
        result = results.json()
        
        assert "q10" in result
        assert "temperatures" in result
        assert "periods" in result
        
        # Q10 should be close to 1.0 (temperature compensated)
        assert 0.8 <= result["q10"] <= 1.2
        assert result["is_temperature_compensated"] is True
    
    def test_compare_metabolic_vs_circadian_q10(self, test_client, yeast_circadian_vm_config):
        """
        Compare Q10 of metabolic reactions (~2-3) vs circadian period (~1).
        Shows temperature compensation is specific to circadian system.
        """
        vm_response = test_client.post("/api/v1/vms", json=yeast_circadian_vm_config)
        vm_id = vm_response.json()["vm_id"]
        
        response = test_client.post(
            f"/api/v1/vms/{vm_id}/analysis/q10-comparison",
            json={
                "temperatures_celsius": [15, 25, 35],
                "metrics": ["circadian_period", "metabolic_rate", "enzyme_activity"]
            }
        )
        assert response.status_code == 202
        
        result = test_client.get(f"/api/v1/vms/{vm_id}/analysis/q10-comparison/results").json()
        
        # Circadian Q10 ~ 1, metabolic Q10 ~ 2-3
        assert 0.8 <= result["q10"]["circadian_period"] <= 1.2
        assert 2.0 <= result["q10"]["metabolic_rate"] <= 3.5
    
    def test_temperature_compensation_mechanism_validation(self, test_client):
        """Test proposed temperature compensation mechanisms"""
        # This would test if model implements known compensation mechanisms
        # e.g., opposing temperature effects on synthesis vs degradation
        pass


@pytest.mark.wishful
class TestHeatShockResponse:
    """Test heat shock and stress responses."""
    
    def test_heat_shock_37_to_42_celsius(self, test_client, ecoli_vm_config):
        """Sudden temperature increase from 37°C to 42°C"""
        vm_response = test_client.post("/api/v1/vms", json=ecoli_vm_config)
        vm_id = vm_response.json()["vm_id"]
        
        # Apply heat shock
        shock_response = test_client.post(
            f"/api/v1/vms/{vm_id}/environment/temperature-shock",
            json={
                "baseline_celsius": 37.0,
                "shock_celsius": 42.0,
                "shock_duration_minutes": 30
            }
        )
        assert shock_response.status_code == 200
        
        # Validate heat shock protein upregulation
        validation = test_client.post(
            f"/api/v1/vms/{vm_id}/validate/gene-expression",
            json={
                "genes": ["dnaK", "dnaJ", "groEL", "groES"],
                "timepoint_minutes": 15,  # Peak response
                "expected_fold_change": 10.0,
                "tolerance": 5.0
            }
        )
        assert validation.status_code == 200
        result = validation.json()
        
        assert result["genes_upregulated"] is True
        for gene in ["dnaK", "dnaJ", "groEL"]:
            assert result["fold_changes"][gene] > 5.0
    
    def test_extreme_heat_shock_lethal(self, test_client, ecoli_vm_config):
        """Test extreme heat (50°C) causes cell death"""
        vm_response = test_client.post("/api/v1/vms", json=ecoli_vm_config)
        vm_id = vm_response.json()["vm_id"]
        
        test_client.post(
            f"/api/v1/vms/{vm_id}/environment/temperature-shock",
            json={"baseline_celsius": 37.0, "shock_celsius": 50.0, "shock_duration_minutes": 10}
        )
        
        # Check cell viability
        status = test_client.get(f"/api/v1/vms/{vm_id}/status").json()
        assert status["viable"] is False
    
    def test_cold_shock_37_to_15_celsius(self, test_client, ecoli_vm_config):
        """Sudden temperature decrease from 37°C to 15°C"""
        vm_response = test_client.post("/api/v1/vms", json=ecoli_vm_config)
        vm_id = vm_response.json()["vm_id"]
        
        shock_response = test_client.post(
            f"/api/v1/vms/{vm_id}/environment/temperature-shock",
            json={"baseline_celsius": 37.0, "shock_celsius": 15.0, "shock_duration_minutes": 60}
        )
        assert shock_response.status_code == 200
        
        # Validate cold shock protein expression
        validation = test_client.post(
            f"/api/v1/vms/{vm_id}/validate/gene-expression",
            json={
                "genes": ["cspA", "cspB", "cspG"],
                "timepoint_minutes": 30,
                "expected_fold_change": 50.0
            }
        )
        result = validation.json()
        
        assert result["genes_upregulated"] is True
        # cspA especially should be highly induced
        assert result["fold_changes"]["cspA"] > 20.0
    
    def test_heat_shock_recovery_dynamics(self, test_client, ecoli_vm_config, temperature_shock_response):
        """Test time course of heat shock response and recovery"""
        vm_response = test_client.post("/api/v1/vms", json=ecoli_vm_config)
        vm_id = vm_response.json()["vm_id"]
        
        # Apply shock and track recovery
        test_client.post(
            f"/api/v1/vms/{vm_id}/environment/temperature-shock",
            json={"baseline_celsius": 25.0, "shock_celsius": 42.0, "shock_duration_minutes": 30}
        )
        
        # Get time series of HSP expression
        timeseries = test_client.get(
            f"/api/v1/vms/{vm_id}/genes/dnaK/expression-timeseries"
        ).json()
        
        assert "timestamps" in timeseries
        assert "expression_levels" in timeseries
        
        # Should see peak around shock_start + 15 min
        # Then gradual return to baseline


@pytest.mark.wishful
@pytest.mark.circadian
class TestTemperatureCycles:
    """Test circadian temperature cycles as zeitgebers."""
    
    def test_temperature_cycle_entrainment(self, test_client, yeast_circadian_vm_config):
        """Test entrainment to 12h warm:12h cool cycles"""
        vm_response = test_client.post("/api/v1/vms", json=yeast_circadian_vm_config)
        vm_id = vm_response.json()["vm_id"]
        
        response = test_client.post(
            f"/api/v1/vms/{vm_id}/environment/temperature-cycle",
            json={
                "cycle_type": "12W:12C",
                "warm_celsius": 28.0,
                "cool_celsius": 22.0,
                "duration_hours": 72
            }
        )
        assert response.status_code == 202
        
        # Validate entrainment
        validation = test_client.post(
            f"/api/v1/vms/{vm_id}/validate/circadian-entrainment",
            json={"zeitgeber": "temperature", "expected_period_hours": 24.0}
        )
        result = validation.json()
        
        assert result["entrained"] is True
        assert abs(result["detected_period_hours"] - 24.0) < 1.0
    
    def test_conflicting_light_and_temperature_cycles(self, test_client, yeast_circadian_vm_config):
        """Test response when light and temperature cycles conflict"""
        vm_response = test_client.post("/api/v1/vms", json=yeast_circadian_vm_config)
        vm_id = vm_response.json()["vm_id"]
        
        # Light cycle: 12L:12D
        # Temperature cycle: 10W:14C (out of phase)
        response = test_client.post(
            f"/api/v1/vms/{vm_id}/environment/conflicting-cycles",
            json={
                "light_cycle": {"type": "12L:12D", "light_lux": 1000},
                "temperature_cycle": {"type": "10W:14C", "warm_c": 28, "cool_c": 22},
                "duration_hours": 120
            }
        )
        assert response.status_code == 202
        
        # Analyze which zeitgeber dominates
        analysis = test_client.get(f"/api/v1/vms/{vm_id}/analysis/zeitgeber-hierarchy")
        result = analysis.json()
        
        assert "dominant_zeitgeber" in result
        # Light usually dominates over temperature
        assert result["dominant_zeitgeber"] == "light"


@pytest.mark.wishful
class TestTemperatureGradients:
    """Test temperature gradient responses."""
    
    def test_spatial_temperature_gradient(self, test_client):
        """Test if organisms sense and respond to temperature gradients"""
        # This would be relevant for multicellular or spatial models
        pass
    
    def test_optimal_temperature_determination(self, test_client, ecoli_vm_config):
        """Determine optimal growth temperature empirically"""
        vm_response = test_client.post("/api/v1/vms", json=ecoli_vm_config)
        vm_id = vm_response.json()["vm_id"]
        
        response = test_client.post(
            f"/api/v1/vms/{vm_id}/experiments/optimal-temperature",
            json={
                "temperature_range": [20, 45],
                "temperature_step": 2.5,
                "duration_per_temp_hours": 6,
                "metric": "growth_rate"
            }
        )
        assert response.status_code == 202
        
        experiment_id = response.json()["experiment_id"]
        results = test_client.get(f"/api/v1/experiments/{experiment_id}/results").json()
        
        assert "optimal_temperature_celsius" in results
        assert 35 <= results["optimal_temperature_celsius"] <= 40  # E. coli optimum


@pytest.mark.wishful
class TestArrheniusKinetics:
    """Test Arrhenius temperature dependence of metabolic rates."""
    
    def test_validate_arrhenius_behavior(self, test_client, ecoli_vm_config):
        """Validate that metabolic rates follow Arrhenius equation"""
        vm_response = test_client.post("/api/v1/vms", json=ecoli_vm_config)
        vm_id = vm_response.json()["vm_id"]
        
        response = test_client.post(
            f"/api/v1/vms/{vm_id}/validate/arrhenius",
            json={
                "temperatures_celsius": [15, 20, 25, 30, 35, 40],
                "reaction": "ATP_synthesis",
                "expected_activation_energy_kj_mol": 50.0,
                "tolerance": 10.0
            }
        )
        assert response.status_code == 202
        
        result = test_client.get(f"/api/v1/vms/{vm_id}/validate/arrhenius/results").json()
        
        assert "activation_energy_kj_mol" in result
        assert "fits_arrhenius" in result
        assert result["fits_arrhenius"] is True
    
    def test_tune_activation_energy_parameters(self, test_client, ecoli_vm_config):
        """Tune model activation energies based on temperature response"""
        vm_response = test_client.post("/api/v1/vms", json=ecoli_vm_config)
        vm_id = vm_response.json()["vm_id"]
        
        response = test_client.post(
            f"/api/v1/vms/{vm_id}/tune/temperature-sensitivity",
            json={
                "observed_growth_rates": [
                    {"temperature_celsius": 20, "growth_rate": 0.3},
                    {"temperature_celsius": 30, "growth_rate": 0.8},
                    {"temperature_celsius": 37, "growth_rate": 1.2},
                    {"temperature_celsius": 42, "growth_rate": 0.4}
                ],
                "optimize": "activation_energy"
            }
        )
        assert response.status_code == 200
        result = response.json()
        
        assert "suggested_activation_energy" in result
        assert "optimized_parameters" in result
        assert "confidence" in result
