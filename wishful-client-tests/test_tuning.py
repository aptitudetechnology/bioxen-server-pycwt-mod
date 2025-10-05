"""
Unit tests for BioXen parameter tuning endpoints.

Tests:
- Rate constant optimization
- Timestep tuning
- Parameter sweeps
- Sensitivity analysis
- Multi-objective optimization
"""

import pytest
import numpy as np


class TestRateConstantTuning:
    """Test suite for POST /api/v1/tune/rate-constants"""
    
    def test_basic_rate_tuning(self, test_client, api_base_url, metabolic_time_series):
        """Test basic rate constant tuning against observed data."""
        request = {
            "observed_data": {
                "timestamps": metabolic_time_series["timestamps"],
                "atp": metabolic_time_series["atp"],
                "nadh": metabolic_time_series["nadh"]
            },
            "initial_rates": {
                "k_glycolysis": 0.5,
                "k_tca": 0.3,
                "k_oxphos": 0.7
            },
            "bounds": {
                "k_glycolysis": [0.1, 2.0],
                "k_tca": [0.05, 1.0],
                "k_oxphos": [0.2, 2.0]
            }
        }
        
        response = test_client.post(
            f"{api_base_url}/tune/rate-constants",
            json=request
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "optimized_rates" in data
        assert "k_glycolysis" in data["optimized_rates"]
        assert "fit_quality" in data
        assert 0 <= data["fit_quality"] <= 1
    
    def test_circadian_gene_tuning(self, test_client, api_base_url, gene_expression_series):
        """Tune circadian gene expression rates."""
        request = {
            "observed_data": {
                "timestamps": gene_expression_series["timestamps"],
                "gene_a": gene_expression_series["values"]["gene_a"],
                "gene_r": gene_expression_series["values"]["gene_r"]
            },
            "model_type": "circadian",
            "initial_rates": {
                "transcription_rate": 1.0,
                "translation_rate": 0.5,
                "degradation_rate": 0.1
            },
            "biological_constraints": {
                "organism": "ecoli",
                "max_transcription_rate": 10.0,
                "min_protein_lifetime": 0.5
            }
        }
        
        response = test_client.post(
            f"{api_base_url}/tune/rate-constants",
            json=request
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Check biological constraints are respected
        assert data["optimized_rates"]["transcription_rate"] <= 10.0
        assert 1.0 / data["optimized_rates"]["degradation_rate"] >= 0.5
    
    def test_multi_objective_tuning(self, test_client, api_base_url, metabolic_time_series):
        """Test multi-objective optimization (fit + parsimony)."""
        request = {
            "observed_data": {
                "timestamps": metabolic_time_series["timestamps"],
                "atp": metabolic_time_series["atp"]
            },
            "initial_rates": {
                "k1": 0.5,
                "k2": 0.3,
                "k3": 0.2
            },
            "objectives": [
                {"type": "fit_quality", "weight": 0.7},
                {"type": "parsimony", "weight": 0.3}
            ]
        }
        
        response = test_client.post(
            f"{api_base_url}/tune/rate-constants",
            json=request
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "optimized_rates" in data
        assert "pareto_front" in data


class TestTimestepOptimization:
    """Test suite for POST /api/v1/tune/timestep"""
    
    def test_basic_timestep_tuning(self, test_client, api_base_url, circadian_time_series):
        """Test adaptive timestep optimization."""
        request = {
            "observed_data": {
                "timestamps": circadian_time_series["timestamps"],
                "values": circadian_time_series["values"]
            },
            "initial_timestep": 0.1,
            "target_accuracy": 1e-6
        }
        
        response = test_client.post(
            f"{api_base_url}/tune/timestep",
            json=request
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "optimal_timestep" in data
        assert "estimated_accuracy" in data
        assert data["estimated_accuracy"] <= request["target_accuracy"] * 10
    
    def test_stiff_system_timestep(self, test_client, api_base_url):
        """Test timestep optimization for stiff system."""
        # Create stiff system data (fast and slow dynamics)
        t = np.linspace(0, 10, 100)
        values = np.exp(-100*t) + np.sin(t)
        
        request = {
            "observed_data": {
                "timestamps": t.tolist(),
                "values": values.tolist()
            },
            "system_type": "stiff",
            "solver_hint": "implicit"
        }
        
        response = test_client.post(
            f"{api_base_url}/tune/timestep",
            json=request
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Should recommend small timestep for stiff system
        assert data["optimal_timestep"] < 0.01
        assert "stiffness_ratio" in data
    
    def test_adaptive_timestep_schedule(self, test_client, api_base_url, circadian_time_series):
        """Test adaptive timestep schedule generation."""
        request = {
            "observed_data": {
                "timestamps": circadian_time_series["timestamps"],
                "values": circadian_time_series["values"]
            },
            "adaptive": True,
            "min_timestep": 0.01,
            "max_timestep": 1.0
        }
        
        response = test_client.post(
            f"{api_base_url}/tune/timestep-schedule",
            json=request
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "timestep_schedule" in data
        assert isinstance(data["timestep_schedule"], list)
        
        # All timesteps should be within bounds
        for dt in data["timestep_schedule"]:
            assert 0.01 <= dt <= 1.0


class TestParameterSweeps:
    """Test suite for POST /api/v1/tune/sweep"""
    
    def test_1d_parameter_sweep(self, test_client, api_base_url, circadian_time_series):
        """Test 1D parameter sweep."""
        request = {
            "observed_data": {
                "timestamps": circadian_time_series["timestamps"],
                "values": circadian_time_series["values"]
            },
            "parameter": "transcription_rate",
            "range": [0.1, 2.0],
            "num_points": 20
        }
        
        response = test_client.post(
            f"{api_base_url}/tune/sweep",
            json=request
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "parameter_values" in data
        assert "fit_scores" in data
        assert len(data["parameter_values"]) == 20
        assert len(data["fit_scores"]) == 20
    
    def test_2d_parameter_sweep(self, test_client, api_base_url, metabolic_time_series):
        """Test 2D parameter sweep (grid search)."""
        request = {
            "observed_data": {
                "timestamps": metabolic_time_series["timestamps"],
                "atp": metabolic_time_series["atp"]
            },
            "parameters": {
                "k1": {"range": [0.1, 1.0], "num_points": 10},
                "k2": {"range": [0.2, 2.0], "num_points": 10}
            }
        }
        
        response = test_client.post(
            f"{api_base_url}/tune/sweep-2d",
            json=request
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "parameter_grid" in data
        assert "fit_scores" in data
        assert len(data["fit_scores"]) == 100  # 10x10 grid
    
    def test_sweep_with_constraints(self, test_client, api_base_url, gene_expression_series):
        """Test parameter sweep with biological constraints."""
        request = {
            "observed_data": {
                "timestamps": gene_expression_series["timestamps"],
                "gene_a": gene_expression_series["values"]["gene_a"]
            },
            "parameter": "degradation_rate",
            "range": [0.01, 1.0],
            "num_points": 50,
            "constraints": {
                "organism": "ecoli",
                "min_protein_lifetime": 1.0,
                "max_mrna_lifetime": 10.0
            }
        }
        
        response = test_client.post(
            f"{api_base_url}/tune/sweep",
            json=request
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # All parameter values should respect constraints
        for deg_rate in data["parameter_values"]:
            lifetime = 1.0 / deg_rate
            assert lifetime >= 1.0


class TestSensitivityAnalysis:
    """Test suite for POST /api/v1/tune/sensitivity"""
    
    def test_local_sensitivity(self, test_client, api_base_url, circadian_time_series):
        """Test local sensitivity analysis."""
        request = {
            "observed_data": {
                "timestamps": circadian_time_series["timestamps"],
                "values": circadian_time_series["values"]
            },
            "parameters": {
                "k1": 0.5,
                "k2": 0.3,
                "k3": 0.2
            },
            "perturbation": 0.01
        }
        
        response = test_client.post(
            f"{api_base_url}/tune/sensitivity",
            json=request
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "sensitivities" in data
        assert "k1" in data["sensitivities"]
        assert "k2" in data["sensitivities"]
        assert "k3" in data["sensitivities"]
        
        # Should identify most sensitive parameter
        assert "most_sensitive" in data
    
    def test_global_sensitivity(self, test_client, api_base_url, metabolic_time_series):
        """Test global sensitivity analysis (Sobol indices)."""
        request = {
            "observed_data": {
                "timestamps": metabolic_time_series["timestamps"],
                "atp": metabolic_time_series["atp"]
            },
            "parameters": {
                "k_glycolysis": {"range": [0.1, 2.0]},
                "k_tca": {"range": [0.05, 1.0]},
                "k_oxphos": {"range": [0.2, 2.0]}
            },
            "method": "sobol",
            "num_samples": 1000
        }
        
        response = test_client.post(
            f"{api_base_url}/tune/sensitivity-global",
            json=request
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "first_order_indices" in data
        assert "total_order_indices" in data
        
        # Indices should sum to ~1.0
        total = sum(data["first_order_indices"].values())
        assert 0.5 <= total <= 1.5
    
    def test_time_varying_sensitivity(self, test_client, api_base_url, long_circadian_series):
        """Test time-varying sensitivity analysis."""
        request = {
            "observed_data": {
                "timestamps": long_circadian_series["timestamps"],
                "values": long_circadian_series["values"]
            },
            "parameters": {
                "transcription_rate": 1.0,
                "degradation_rate": 0.1
            },
            "time_windows": [
                {"start": 0, "end": 24},
                {"start": 24, "end": 48},
                {"start": 48, "end": 72}
            ]
        }
        
        response = test_client.post(
            f"{api_base_url}/tune/sensitivity-time-varying",
            json=request
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "time_windows" in data
        assert len(data["time_windows"]) == 3
        
        # Each window should have sensitivity data
        for window in data["time_windows"]:
            assert "sensitivities" in window


class TestOptimizationAlgorithms:
    """Test suite for POST /api/v1/tune/optimize"""
    
    def test_gradient_descent(self, test_client, api_base_url, circadian_time_series):
        """Test gradient descent optimization."""
        request = {
            "observed_data": {
                "timestamps": circadian_time_series["timestamps"],
                "values": circadian_time_series["values"]
            },
            "initial_parameters": {
                "k1": 0.5,
                "k2": 0.3
            },
            "method": "gradient_descent",
            "learning_rate": 0.01,
            "max_iterations": 1000
        }
        
        response = test_client.post(
            f"{api_base_url}/tune/optimize",
            json=request
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "optimized_parameters" in data
        assert "convergence_history" in data
        assert "final_loss" in data
    
    def test_genetic_algorithm(self, test_client, api_base_url, metabolic_time_series):
        """Test genetic algorithm optimization."""
        request = {
            "observed_data": {
                "timestamps": metabolic_time_series["timestamps"],
                "atp": metabolic_time_series["atp"]
            },
            "parameter_bounds": {
                "k1": [0.1, 2.0],
                "k2": [0.05, 1.0],
                "k3": [0.1, 1.5]
            },
            "method": "genetic",
            "population_size": 50,
            "num_generations": 100
        }
        
        response = test_client.post(
            f"{api_base_url}/tune/optimize",
            json=request
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "optimized_parameters" in data
        assert "best_fitness" in data
        assert "generation_history" in data
    
    def test_bayesian_optimization(self, test_client, api_base_url, circadian_time_series):
        """Test Bayesian optimization."""
        request = {
            "observed_data": {
                "timestamps": circadian_time_series["timestamps"],
                "values": circadian_time_series["values"]
            },
            "parameter_bounds": {
                "transcription_rate": [0.1, 5.0],
                "degradation_rate": [0.01, 0.5]
            },
            "method": "bayesian",
            "num_iterations": 50,
            "acquisition_function": "expected_improvement"
        }
        
        response = test_client.post(
            f"{api_base_url}/tune/optimize",
            json=request
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "optimized_parameters" in data
        assert "posterior_mean" in data
        assert "posterior_variance" in data
    
    @pytest.mark.slow
    def test_parallel_optimization(self, test_client, api_base_url, metabolic_time_series):
        """Test parallel optimization across multiple starting points."""
        request = {
            "observed_data": {
                "timestamps": metabolic_time_series["timestamps"],
                "atp": metabolic_time_series["atp"]
            },
            "parameter_bounds": {
                "k1": [0.1, 2.0],
                "k2": [0.1, 2.0]
            },
            "method": "multi_start",
            "num_starts": 10,
            "base_method": "lbfgs"
        }
        
        response = test_client.post(
            f"{api_base_url}/tune/optimize",
            json=request
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "best_parameters" in data
        assert "all_solutions" in data
        assert len(data["all_solutions"]) <= 10


class TestBiologicalConstraints:
    """Test suite for POST /api/v1/tune/constrained"""
    
    def test_ecoli_constraints(self, test_client, api_base_url, metabolic_time_series):
        """Test E. coli biological constraints."""
        request = {
            "observed_data": {
                "timestamps": metabolic_time_series["timestamps"],
                "atp": metabolic_time_series["atp"]
            },
            "organism": "ecoli",
            "parameters_to_tune": ["growth_rate", "protein_production"],
            "auto_constrain": True
        }
        
        response = test_client.post(
            f"{api_base_url}/tune/constrained",
            json=request
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # E. coli doubling time should be realistic
        if "growth_rate" in data["optimized_parameters"]:
            doubling_time = np.log(2) / data["optimized_parameters"]["growth_rate"]
            assert 15 <= doubling_time <= 120  # minutes
    
    def test_thermodynamic_constraints(self, test_client, api_base_url, metabolic_time_series):
        """Test thermodynamic constraints (Gibbs free energy)."""
        request = {
            "observed_data": {
                "timestamps": metabolic_time_series["timestamps"],
                "atp": metabolic_time_series["atp"]
            },
            "reactions": [
                {"name": "glycolysis", "delta_g": -50.0},
                {"name": "tca", "delta_g": -30.0}
            ],
            "enforce_thermodynamics": True
        }
        
        response = test_client.post(
            f"{api_base_url}/tune/constrained",
            json=request
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Should respect thermodynamic feasibility
        assert "thermodynamic_feasible" in data
        assert data["thermodynamic_feasible"] is True
