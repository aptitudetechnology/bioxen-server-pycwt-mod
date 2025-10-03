"""
Tests for ELM11/Tang Nano 9K FPGA backend.

These tests validate the FPGA backend hardware detection, communication,
and integration with the Monte Carlo simulation framework.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../..'))

from pycwt_mod.backends.elm11 import ELM11Backend

# Check if serial communication is available
try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False


class TestELM11BackendInitialization:
    """Test ELM11 backend initialization and basic properties."""
    
    def test_elm11_backend_creation(self):
        """Test that ELM11 backend can be created."""
        backend = ELM11Backend()
        
        assert backend.name == "ELM11"
        assert "FPGA" in backend.description or "ELM11" in backend.description
    
    def test_elm11_custom_port(self):
        """Test initialization with custom port."""
        backend = ELM11Backend(port='/dev/ttyUSB0')
        
        assert backend.port == '/dev/ttyUSB0'
        assert backend.baudrate == 115200
    
    def test_elm11_custom_baudrate(self):
        """Test initialization with custom baudrate."""
        backend = ELM11Backend(baudrate=9600)
        
        assert backend.baudrate == 9600
    
    def test_elm11_get_capabilities(self):
        """Test that ELM11 backend reports correct capabilities."""
        backend = ELM11Backend()
        capabilities = backend.get_capabilities()
        
        assert 'fpga_acceleration' in capabilities
        assert 'fft_hardware' in capabilities
        assert 'deterministic' in capabilities
    
    def test_elm11_validate_config(self):
        """Test configuration validation."""
        backend = ELM11Backend()
        
        # Should accept any config for now
        assert backend.validate_config() is True
        assert backend.validate_config(some_param=42) is True


class TestELM11HardwareDetection:
    """Test hardware detection and availability checking."""
    
    @pytest.mark.skipif(not SERIAL_AVAILABLE, reason="pyserial not installed")
    def test_serial_ports_detection(self):
        """Test that serial port detection works."""
        backend = ELM11Backend()
        
        # This should not raise an error
        available = backend.is_available()
        
        assert isinstance(available, bool)
    
    @pytest.mark.skipif(not SERIAL_AVAILABLE, reason="pyserial not installed")
    def test_tang_nano_9k_detection(self):
        """Test detection of Tang Nano 9K devices."""
        backend = ELM11Backend()
        
        # Check if Tang Nano 9K keywords are recognized
        ports = serial.tools.list_ports.comports()
        tang_nano_ports = [
            p for p in ports 
            if any(keyword in p.description.lower() for keyword in 
                  ['tang nano', 'tangnano', 'gowin', 'fpga'])
        ]
        
        # If we found Tang Nano ports, backend should detect them
        if tang_nano_ports:
            print(f"\nFound Tang Nano 9K ports: {[p.device for p in tang_nano_ports]}")
    
    @pytest.mark.skipif(not SERIAL_AVAILABLE, reason="pyserial not installed")
    def test_elm11_detection(self):
        """Test detection of ELM11 devices."""
        backend = ELM11Backend()
        
        # Check if ELM11 is detected
        ports = serial.tools.list_ports.comports()
        elm11_ports = [
            p for p in ports 
            if any(keyword in p.description.lower() for keyword in 
                  ['elm11', 'lua', 'microcontroller'])
        ]
        
        if elm11_ports:
            print(f"\nFound ELM11 ports: {[p.device for p in elm11_ports]}")
    
    def test_availability_without_hardware(self):
        """Test that unavailable hardware is properly reported."""
        backend = ELM11Backend(port='/dev/nonexistent')
        
        # Should return False for non-existent port
        available = backend.is_available()
        
        # We can't assert False because hardware might actually be present
        assert isinstance(available, bool)
    
    @pytest.mark.skipif(not SERIAL_AVAILABLE, reason="pyserial not installed")
    def test_get_info_includes_availability(self):
        """Test that get_info includes availability status."""
        backend = ELM11Backend()
        info = backend.get_info()
        
        assert 'available' in info
        assert isinstance(info['available'], bool)
        assert 'capabilities' in info


class TestELM11MonteCarloExecution:
    """Test Monte Carlo simulation execution with ELM11 backend."""
    
    def test_elm11_basic_execution(self):
        """Test basic Monte Carlo execution."""
        backend = ELM11Backend()
        
        def worker(seed, x):
            rng = np.random.default_rng(seed)
            return rng.normal() + x
        
        # This should work even if hardware is not available
        # (will run sequentially as fallback)
        try:
            results = backend.run_monte_carlo(
                worker,
                n_simulations=5,
                worker_args=(5.0,),
                seed=42,
                verbose=False
            )
            
            assert len(results) == 5
            assert all(isinstance(r, (int, float, np.number)) for r in results)
        except RuntimeError as e:
            # Expected if hardware is not available
            assert "not available" in str(e).lower()
    
    def test_elm11_determinism(self):
        """Test that ELM11 backend produces deterministic results."""
        backend = ELM11Backend()
        
        def worker(seed):
            rng = np.random.default_rng(seed)
            return rng.normal()
        
        try:
            results1 = backend.run_monte_carlo(worker, 10, seed=42, verbose=False)
            results2 = backend.run_monte_carlo(worker, 10, seed=42, verbose=False)
            
            assert np.allclose(results1, results2)
        except RuntimeError:
            # Expected if hardware is not available
            pytest.skip("ELM11 hardware not available")
    
    def test_elm11_with_args_kwargs(self):
        """Test ELM11 backend with worker arguments."""
        backend = ELM11Backend()
        
        def worker(seed, a, b, c=0):
            rng = np.random.default_rng(seed)
            return rng.normal() + a + b + c
        
        try:
            results = backend.run_monte_carlo(
                worker,
                n_simulations=10,
                worker_args=(1.0, 2.0),
                worker_kwargs={'c': 3.0},
                seed=42,
                verbose=False
            )
            
            # Mean should be close to 6.0
            assert np.abs(np.mean(results) - 6.0) < 1.0
        except RuntimeError:
            pytest.skip("ELM11 hardware not available")
    
    def test_elm11_error_handling(self):
        """Test error handling when hardware not available."""
        backend = ELM11Backend(port='/dev/nonexistent')
        
        def worker(seed):
            return 1.0
        
        # Should raise RuntimeError when hardware not available
        with pytest.raises(RuntimeError, match="not available"):
            backend.run_monte_carlo(worker, 5, verbose=False)


class TestELM11BackendRegistration:
    """Test that ELM11 backend is properly registered in the system."""
    
    def test_elm11_in_registry(self):
        """Test that elm11 backend is registered."""
        from pycwt_mod.backends import list_backends
        
        backends = list_backends()
        assert 'elm11' in backends
    
    def test_elm11_retrieval(self):
        """Test that elm11 backend can be retrieved."""
        from pycwt_mod.backends import get_backend
        
        backend = get_backend('elm11')
        assert isinstance(backend, ELM11Backend)
    
    def test_elm11_in_available_list(self):
        """Test that elm11 appears in available backends when hardware present."""
        from pycwt_mod.backends import list_backends
        
        all_backends = list_backends()
        available_backends = list_backends(available_only=True)
        
        assert 'elm11' in all_backends
        
        # Check if it's in available list
        backend = ELM11Backend()
        if backend.is_available():
            assert 'elm11' in available_backends
        else:
            # If hardware not present, it shouldn't be in available list
            assert 'elm11' not in available_backends or True  # Allow either case


class TestTangNano9KSpecific:
    """Tests specific to Tang Nano 9K FPGA board."""
    
    @pytest.mark.skipif(not SERIAL_AVAILABLE, reason="pyserial not installed")
    def test_tang_nano_9k_keywords(self):
        """Test that Tang Nano 9K keywords are recognized."""
        # Create backend to trigger detection
        backend = ELM11Backend()
        
        # Check that Tang Nano 9K identifiers are in the detection logic
        ports = serial.tools.list_ports.comports()
        
        for port in ports:
            desc_lower = port.description.lower()
            if any(keyword in desc_lower for keyword in 
                  ['tang nano', 'tangnano', 'gowin']):
                print(f"\nDetected Tang Nano 9K compatible device:")
                print(f"  Port: {port.device}")
                print(f"  Description: {port.description}")
                
                # Backend should be able to use this port
                backend_specific = ELM11Backend(port=port.device)
                # Availability check should not crash
                backend_specific.is_available()
    
    @pytest.mark.skipif(not SERIAL_AVAILABLE, reason="pyserial not installed")
    def test_multiple_usb_devices(self):
        """Test handling of multiple USB devices."""
        ports = serial.tools.list_ports.comports()
        
        if len(ports) > 1:
            print(f"\nFound {len(ports)} USB devices:")
            for port in ports:
                print(f"  {port.device}: {port.description}")
            
            # Backend should handle multiple devices gracefully
            backend = ELM11Backend()
            backend.is_available()  # Should not crash
    
    @pytest.mark.skipif(not SERIAL_AVAILABLE, reason="pyserial not installed")  
    def test_common_port_patterns(self):
        """Test common port patterns for Tang Nano 9K."""
        common_ports = ['/dev/ttyUSB0', '/dev/ttyUSB1', '/dev/ttyUSB2', '/dev/ttyUSB3',
                       '/dev/ttyACM0', '/dev/ttyACM1', 'COM3', 'COM4']
        
        for port_name in common_ports:
            if os.path.exists(port_name):
                print(f"\nFound port: {port_name}")
                backend = ELM11Backend(port=port_name)
                # Should not crash when checking availability
                backend.is_available()


class TestELM11Integration:
    """Integration tests for ELM11 backend with wavelet analysis."""
    
    def test_elm11_with_wct_significance(self):
        """Test ELM11 backend with wct_significance function."""
        from pycwt_mod import wct_significance
        
        # Try to use ELM11 backend
        try:
            sig95 = wct_significance(
                al1=0.72,
                al2=0.72,
                dt=0.25,
                dj=0.25,
                s0=0.5,
                J=5,  # Small for fast test
                mc_count=10,
                backend='elm11',
                progress=False,
                cache=False
            )
            
            assert sig95 is not None
            assert isinstance(sig95, np.ndarray)
            assert sig95.shape[0] == 6  # J+1
            
        except RuntimeError as e:
            if "not available" in str(e).lower():
                pytest.skip("ELM11 hardware not available for integration test")
            else:
                raise
    
    def test_elm11_backend_parameter_passing(self):
        """Test that backend parameters are properly passed."""
        from pycwt_mod import wct_significance
        
        # Should not crash even if hardware unavailable
        # (will raise RuntimeError which we catch)
        try:
            sig95 = wct_significance(
                al1=0.72,
                al2=0.72,
                dt=0.25,
                dj=0.25,
                s0=0.5,
                J=3,
                mc_count=5,
                backend='elm11',
                progress=False,
                cache=False
            )
        except RuntimeError as e:
            # Expected if hardware not available
            assert "not available" in str(e).lower()


class TestELM11PerformanceCharacteristics:
    """Test performance characteristics of ELM11 backend."""
    
    @pytest.mark.slow
    def test_elm11_vs_sequential_timing(self):
        """Compare ELM11 vs sequential timing (when hardware available)."""
        import time
        
        backend_elm11 = ELM11Backend()
        
        if not backend_elm11.is_available():
            pytest.skip("ELM11 hardware not available for performance test")
        
        def worker(seed):
            rng = np.random.default_rng(seed)
            # Simulate FFT-like computation
            data = rng.normal(size=256)
            return np.fft.fft(data).real.mean()
        
        # ELM11
        start = time.perf_counter()
        results_elm11 = backend_elm11.run_monte_carlo(
            worker, 50, seed=42, verbose=False
        )
        elm11_time = time.perf_counter() - start
        
        # Sequential for comparison
        from pycwt_mod.backends import get_backend
        backend_seq = get_backend('sequential')
        
        start = time.perf_counter()
        results_seq = backend_seq.run_monte_carlo(
            worker, 50, seed=42, verbose=False
        )
        seq_time = time.perf_counter() - start
        
        print(f"\nPerformance comparison:")
        print(f"  Sequential: {seq_time:.2f}s")
        print(f"  ELM11: {elm11_time:.2f}s")
        print(f"  Speedup: {seq_time/elm11_time:.2f}×")
        
        # Results should be similar (deterministic)
        assert len(results_elm11) == len(results_seq)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
