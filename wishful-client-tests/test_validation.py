"""
Unit tests for BioXen model validation endpoints.

Tests:
- Oscillation period validation
- Amplitude stability checks
- Numerical stability analysis
- Quality scoring
"""

import pytest
import numpy as np


class TestOscillationValidation:
    """Test suite for POST /api/v1/validate/oscillation"""
    
    def test_validate_circadian_period_basic(self, test_client, api_base_url, circadian_time_series):
        """Test basic circadian period validation."""
        request = {
            "timestamps": circadian_time_series["timestamps"],
            "values": circadian_time_series["values"],
            "expected_period_hours": 24.0,
            "tolerance_hours": 2.0
        }
        
        response = test_client.post(
            f"{api_base_url}/validate/oscillation",
            json=request
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Check required fields
        assert "measured_period_hours" in data
        assert "validation_passed" in data
        assert "deviation_hours" in data
        assert isinstance(data["validation_passed"], bool)
    
    def test_validate_amplitude_stability(self, test_client, api_base_url, oscillating_signal):
        """Test amplitude decay detection."""
        request = {
            "timestamps": oscillating_signal["timestamps"],
            "values": oscillating_signal["values"],
            "max_decay_percent": 10.0
        }
        
        response = test_client.post(
            f"{api_base_url}/validate/amplitude",
            json=request
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "decay_detected" in data
        assert "decay_percent" in data
        assert "validation_passed" in data
    
    def test_validate_missing_timestamps(self, test_client, api_base_url):
        """Test validation with missing timestamps."""
        invalid_request = {
            "values": [1, 2, 3, 4, 5],
            "expected_period_hours": 24.0
            # Missing timestamps
        }
        
        response = test_client.post(
            f"{api_base_url}/validate/oscillation",
            json=invalid_request
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_validate_empty_data(self, test_client, api_base_url):
        """Test validation with empty data."""
        invalid_request = {
            "timestamps": [],
            "values": [],
            "expected_period_hours": 24.0
        }
        
        response = test_client.post(
            f"{api_base_url}/validate/oscillation",
            json=invalid_request
        )
        
        assert response.status_code in [400, 422]
    
    def test_validate_mismatched_lengths(self, test_client, api_base_url):
        """Test validation with mismatched timestamp/value lengths."""
        invalid_request = {
            "timestamps": [0, 1, 2, 3],
            "values": [1, 2],  # Different length
            "expected_period_hours": 24.0
        }
        
        response = test_client.post(
            f"{api_base_url}/validate/oscillation",
            json=invalid_request
        )
        
        assert response.status_code in [400, 422]
    
    @pytest.mark.slow
    def test_validate_long_time_series(self, test_client, api_base_url, long_circadian_series):
        """Test validation with 96-hour time series."""
        request = {
            "timestamps": long_circadian_series["timestamps"],
            "values": long_circadian_series["values"],
            "expected_period_hours": 24.0,
            "tolerance_hours": 1.0
        }
        
        response = test_client.post(
            f"{api_base_url}/validate/oscillation",
            json=request
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["validation_passed"] is True


class TestNumericalStability:
    """Test suite for POST /api/v1/validate/stability"""
    
    def test_laplace_stability_check_basic(self, test_client, api_base_url, stable_time_series):
        """Test Laplace-based stability check."""
        request = {
            "timestamps": stable_time_series["timestamps"],
            "values": stable_time_series["values"]
        }
        
        response = test_client.post(
            f"{api_base_url}/validate/stability",
            json=request
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "stable" in data
        assert "poles" in data
        assert isinstance(data["stable"], bool)
        
        # All poles should have negative real parts for stability
        if data["stable"]:
            for pole in data["poles"]:
                assert pole["real"] < 0
    
    def test_stability_unstable_system(self, test_client, api_base_url, unstable_time_series):
        """Test stability check on unstable system."""
        request = {
            "timestamps": unstable_time_series["timestamps"],
            "values": unstable_time_series["values"]
        }
        
        response = test_client.post(
            f"{api_base_url}/validate/stability",
            json=request
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["stable"] is False
        assert len(data["poles"]) > 0
    
    def test_detect_numerical_instability(self, test_client, api_base_url):
        """Test detection of NaN/Inf values."""
        request = {
            "timestamps": [0, 1, 2, 3, 4],
            "values": [1.0, 2.0, float('nan'), 4.0, 5.0]
        }
        
        response = test_client.post(
            f"{api_base_url}/validate/check-instability",
            json=request
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["nan_detected"] is True


class TestQualityScoring:
    """Test suite for POST /api/v1/validate/quality-score"""
    
    def test_compute_quality_score_basic(self, test_client, api_base_url, circadian_time_series):
        """Test quality score computation."""
        request = {
            "timestamps": circadian_time_series["timestamps"],
            "values": circadian_time_series["values"],
            "expected_period_hours": 24.0
        }
        
        response = test_client.post(
            f"{api_base_url}/validate/quality-score",
            json=request
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "overall_score" in data
        assert "components" in data
        assert 0 <= data["overall_score"] <= 100
        
        # Check score components
        components = data["components"]
        assert "numerical_stability" in components
        assert "oscillation_quality" in components
    
    def test_quality_score_with_reference_data(self, test_client, api_base_url):
        """Test quality score comparing to reference data."""
        request = {
            "timestamps": [0, 6, 12, 18, 24],
            "values": [100, 110, 105, 108, 102],
            "reference_data": {
                "timestamps": [0, 6, 12, 18, 24],
                "values": [100, 112, 107, 109, 103]
            }
        }
        
        response = test_client.post(
            f"{api_base_url}/validate/quality-score",
            json=request
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "rmse" in data
        assert "max_deviation_percent" in data


class TestDeviationDetection:
    """Test suite for POST /api/v1/validate/detect-deviations"""
    
    def test_detect_period_drift(self, test_client, api_base_url, drifting_oscillation):
        """Test detection of period drift over time."""
        request = {
            "timestamps": drifting_oscillation["timestamps"],
            "values": drifting_oscillation["values"],
            "expected_period_hours": 24.0,
            "max_drift_hours_per_day": 0.5
        }
        
        response = test_client.post(
            f"{api_base_url}/validate/detect-deviations",
            json=request
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "drift_detected" in data
        assert "drift_hours_per_day" in data
    
    def test_detect_amplitude_decay(self, test_client, api_base_url, decaying_oscillation):
        """Test detection of amplitude decay."""
        request = {
            "timestamps": decaying_oscillation["timestamps"],
            "values": decaying_oscillation["values"],
            "max_decay_percent": 10.0
        }
        
        response = test_client.post(
            f"{api_base_url}/validate/amplitude-decay",
            json=request
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "decay_detected" in data
        assert "decay_percent" in data


class TestBatchValidation:
    """Test suite for POST /api/v1/validate/batch"""
    
    def test_batch_validate_multiple_signals(self, test_client, api_base_url):
        """Test batch validation of multiple time series."""
        request = {
            "signals": [
                {
                    "id": "signal_1",
                    "timestamps": [0, 1, 2, 3, 4],
                    "values": [1, 2, 3, 2, 1]
                },
                {
                    "id": "signal_2",
                    "timestamps": [0, 1, 2, 3, 4],
                    "values": [2, 3, 4, 3, 2]
                }
            ],
            "validations": ["oscillation", "stability", "quality-score"]
        }
        
        response = test_client.post(
            f"{api_base_url}/validate/batch",
            json=request
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert len(data["results"]) == 2
        for result in data["results"]:
            assert "signal_id" in result
            assert "validations" in result
