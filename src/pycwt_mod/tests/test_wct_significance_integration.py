"""
Phase 2 Validation Tests: Integration of backend system with wct_significance.

These tests validate that the backend integration works correctly and produces
consistent, reproducible results across different backends.
"""

import pytest
import numpy as np
import sys
import os

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from pycwt_mod import wct_significance
from pycwt_mod.backends import list_backends, get_backend


class TestWCTSignificanceBackendIntegration:
    """Test wct_significance integration with backend system."""
    
    def test_default_backend_selection(self):
        """Test that default backend is automatically selected."""
        # Should run without specifying backend
        sig95 = wct_significance(
            al1=0.72,
            al2=0.72,
            dt=0.25,
            dj=0.25,
            s0=0.5,
            J=10,
            mc_count=10,
            progress=False,
            cache=False
        )
        
        assert sig95 is not None
        assert isinstance(sig95, np.ndarray)
        assert sig95.shape[0] == 11  # J+1 scales
    
    def test_explicit_sequential_backend(self):
        """Test explicit sequential backend selection."""
        sig95 = wct_significance(
            al1=0.72,
            al2=0.72,
            dt=0.25,
            dj=0.25,
            s0=0.5,
            J=10,
            mc_count=10,
            backend='sequential',
            progress=False,
            cache=False
        )
        
        assert sig95 is not None
        assert isinstance(sig95, np.ndarray)
        assert sig95.shape[0] == 11
    
    @pytest.mark.skipif(
        not get_backend('joblib').is_available(),
        reason="Joblib backend not available"
    )
    def test_explicit_joblib_backend(self):
        """Test explicit joblib backend selection."""
        sig95 = wct_significance(
            al1=0.72,
            al2=0.72,
            dt=0.25,
            dj=0.25,
            s0=0.5,
            J=10,
            mc_count=10,
            backend='joblib',
            n_jobs=2,
            progress=False,
            cache=False
        )
        
        assert sig95 is not None
        assert isinstance(sig95, np.ndarray)
        assert sig95.shape[0] == 11
    
    def test_backward_compatibility_positional_args(self):
        """Test backward compatibility with positional arguments."""
        # Old-style function call (no backend parameter)
        sig95 = wct_significance(
            0.72,  # al1
            0.72,  # al2
            dt=0.25,
            dj=0.25,
            s0=0.5,
            J=10,
            mc_count=10,
            progress=False,
            cache=False
        )
        
        assert sig95 is not None
        assert isinstance(sig95, np.ndarray)
    
    def test_invalid_backend_raises_error(self):
        """Test that invalid backend name raises appropriate error."""
        with pytest.raises(ValueError, match="not found"):
            wct_significance(
                al1=0.72,
                al2=0.72,
                dt=0.25,
                dj=0.25,
                s0=0.5,
                J=10,
                mc_count=10,
                backend='nonexistent',
                progress=False,
                cache=False
            )
    
    def test_progress_parameter_works(self):
        """Test that progress parameter doesn't cause errors."""
        # Should work with progress=True (though output not tested)
        sig95 = wct_significance(
            al1=0.72,
            al2=0.72,
            dt=0.25,
            dj=0.25,
            s0=0.5,
            J=10,
            mc_count=5,
            backend='sequential',
            progress=True,
            cache=False
        )
        
        assert sig95 is not None
    
    def test_results_are_probabilistic(self):
        """Test that results are in valid probability range."""
        sig95 = wct_significance(
            al1=0.72,
            al2=0.72,
            dt=0.25,
            dj=0.25,
            s0=0.5,
            J=10,
            mc_count=20,
            backend='sequential',
            progress=False,
            cache=False
        )
        
        # Significance values should be between 0 and 1
        valid_values = sig95[~np.isnan(sig95)]
        assert np.all(valid_values >= 0.0)
        assert np.all(valid_values <= 1.0)
    
    def test_different_wavelet_types(self):
        """Test integration works with different wavelet types."""
        from pycwt_mod.mothers import Morlet, Paul, DOG
        
        wavelets = [Morlet(), Paul(), DOG()]
        
        for wavelet in wavelets:
            sig95 = wct_significance(
                al1=0.72,
                al2=0.72,
                dt=0.25,
                dj=0.25,
                s0=0.5,
                J=10,
                mc_count=10,
                wavelet=wavelet,
                backend='sequential',
                progress=False,
                cache=False
            )
            
            assert sig95 is not None
            assert not np.all(np.isnan(sig95))


class TestBackendEquivalence:
    """Test that different backends produce equivalent results."""
    
    def test_sequential_reproducibility(self):
        """Test sequential backend produces reproducible results."""
        # Run twice - should get similar results (stochastic but bounded)
        sig95_1 = wct_significance(
            al1=0.72,
            al2=0.72,
            dt=0.25,
            dj=0.25,
            s0=0.5,
            J=10,
            mc_count=50,
            backend='sequential',
            progress=False,
            cache=False
        )
        
        sig95_2 = wct_significance(
            al1=0.72,
            al2=0.72,
            dt=0.25,
            dj=0.25,
            s0=0.5,
            J=10,
            mc_count=50,
            backend='sequential',
            progress=False,
            cache=False
        )
        
        # Results should be similar (Monte Carlo variation expected)
        # but not wildly different
        valid_1 = sig95_1[~np.isnan(sig95_1)]
        valid_2 = sig95_2[~np.isnan(sig95_2)]
        
        # Check that results are in similar range
        assert len(valid_1) == len(valid_2)
        correlation = np.corrcoef(valid_1, valid_2)[0, 1]
        assert correlation > 0.8, f"Results poorly correlated: {correlation}"
    
    @pytest.mark.skipif(
        not get_backend('joblib').is_available(),
        reason="Joblib backend not available"
    )
    def test_sequential_vs_joblib_equivalence(self):
        """Test that joblib and sequential backends produce similar results."""
        # Sequential
        sig95_seq = wct_significance(
            al1=0.72,
            al2=0.72,
            dt=0.25,
            dj=0.25,
            s0=0.5,
            J=10,
            mc_count=100,
            backend='sequential',
            progress=False,
            cache=False
        )
        
        # Joblib parallel
        sig95_par = wct_significance(
            al1=0.72,
            al2=0.72,
            dt=0.25,
            dj=0.25,
            s0=0.5,
            J=10,
            mc_count=100,
            backend='joblib',
            n_jobs=2,
            progress=False,
            cache=False
        )
        
        # Results should be similar (Monte Carlo variation expected)
        valid_seq = sig95_seq[~np.isnan(sig95_seq)]
        valid_par = sig95_par[~np.isnan(sig95_par)]
        
        assert len(valid_seq) == len(valid_par)
        
        # Results should be highly correlated
        correlation = np.corrcoef(valid_seq, valid_par)[0, 1]
        assert correlation > 0.8, \
            f"Sequential and parallel results poorly correlated: {correlation}"
        
        # Mean should be within reasonable range
        mean_diff = np.abs(np.mean(valid_seq) - np.mean(valid_par))
        assert mean_diff < 0.1, \
            f"Mean difference too large: {mean_diff}"


class TestEdgeCases:
    """Test edge cases and robustness."""
    
    def test_small_mc_count(self):
        """Test with very small Monte Carlo count."""
        sig95 = wct_significance(
            al1=0.72,
            al2=0.72,
            dt=0.25,
            dj=0.25,
            s0=0.5,
            J=10,
            mc_count=5,  # Very small
            backend='sequential',
            progress=False,
            cache=False
        )
        
        assert sig95 is not None
        assert sig95.shape[0] == 11
    
    def test_large_mc_count(self):
        """Test with large Monte Carlo count."""
        sig95 = wct_significance(
            al1=0.72,
            al2=0.72,
            dt=0.25,
            dj=0.25,
            s0=0.5,
            J=5,  # Smaller J for speed
            mc_count=200,  # Larger mc_count
            backend='sequential',
            progress=False,
            cache=False
        )
        
        assert sig95 is not None
        assert sig95.shape[0] == 6
    
    def test_extreme_ar1_coefficients(self):
        """Test with extreme AR(1) coefficients."""
        # Near white noise
        sig95_white = wct_significance(
            al1=0.01,
            al2=0.01,
            dt=0.25,
            dj=0.25,
            s0=0.5,
            J=10,
            mc_count=20,
            backend='sequential',
            progress=False,
            cache=False
        )
        
        # Near perfect autocorrelation
        sig95_red = wct_significance(
            al1=0.95,
            al2=0.95,
            dt=0.25,
            dj=0.25,
            s0=0.5,
            J=10,
            mc_count=20,
            backend='sequential',
            progress=False,
            cache=False
        )
        
        assert sig95_white is not None
        assert sig95_red is not None
        assert not np.all(np.isnan(sig95_white))
        assert not np.all(np.isnan(sig95_red))
    
    @pytest.mark.skipif(
        not get_backend('joblib').is_available(),
        reason="Joblib backend not available"
    )
    def test_different_n_jobs_values(self):
        """Test with different n_jobs values."""
        for n_jobs in [1, 2, -1]:  # -1 means all CPUs
            sig95 = wct_significance(
                al1=0.72,
                al2=0.72,
                dt=0.25,
                dj=0.25,
                s0=0.5,
                J=10,
                mc_count=20,
                backend='joblib',
                n_jobs=n_jobs,
                progress=False,
                cache=False
            )
            
            assert sig95 is not None
            assert sig95.shape[0] == 11


class TestCacheCompatibility:
    """Test that backend system works with caching."""
    
    def test_cache_bypasses_backend(self):
        """Test that cached results bypass backend system."""
        # First run (creates cache)
        sig95_1 = wct_significance(
            al1=0.72,
            al2=0.72,
            dt=0.25,
            dj=0.25,
            s0=0.5,
            J=10,
            mc_count=20,
            backend='sequential',
            progress=False,
            cache=True  # Enable cache
        )
        
        # Second run (should load from cache)
        sig95_2 = wct_significance(
            al1=0.72,
            al2=0.72,
            dt=0.25,
            dj=0.25,
            s0=0.5,
            J=10,
            mc_count=20,
            backend='joblib',  # Different backend
            progress=False,
            cache=True
        )
        
        # Results should be identical when cached
        np.testing.assert_array_equal(sig95_1, sig95_2)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
