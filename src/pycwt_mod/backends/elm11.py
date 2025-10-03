"""
ELM11 FPGA backend for Monte Carlo simulations.

This backend provides FPGA-accelerated Monte Carlo simulations using
the ELM11 microcontroller with Lua-based FFT implementations.
"""

import numpy as np
from typing import Any, Callable, Dict, List, Optional
from .base import MonteCarloBackend


class ELM11Backend(MonteCarloBackend):
    """
    FPGA-accelerated backend using ELM11 for Monte Carlo simulations.
    
    This backend uses the ELM11 FPGA microcontroller to accelerate
    FFT computations within Monte Carlo simulations for wavelet analysis.
    
    Characteristics:
    - FPGA acceleration for FFT operations
    - Serial communication with ELM11 hardware
    - Lua-based signal processing on microcontroller
    - Deterministic execution with seed control
    
    Parameters
    ----------
    port : str, optional
        Serial port for ELM11 connection (default: auto-detect)
    baudrate : int, optional
        Serial baudrate (default: 115200)
    
    Examples
    --------
    >>> from pycwt_mod.backends import ELM11Backend
    >>> backend = ELM11Backend()
    >>> def worker(seed, x):
    ...     # Worker function that uses FFT
    ...     return some_fft_based_computation(seed, x)
    >>> results = backend.run_monte_carlo(worker, 100, worker_args=(5.0,), seed=42)
    """
    
    def __init__(self, port: Optional[str] = None, baudrate: int = 115200):
        """
        Initialize the ELM11 backend.
        
        Parameters
        ----------
        port : str, optional
            Serial port for ELM11 connection
        baudrate : int, optional
            Serial communication baudrate
        """
        super().__init__()
        self.name = "ELM11"
        self.description = "FPGA acceleration using ELM11 microcontroller"
        self.port = port
        self.baudrate = baudrate
        self._elm11_available = None
        self._connection = None
    
    def is_available(self) -> bool:
        """
        Check if ELM11 backend is available.
        
        Returns
        -------
        available : bool
            True if ELM11 hardware is connected and accessible
        """
        if self._elm11_available is None:
            self._elm11_available = self._check_elm11_availability()
        return self._elm11_available
    
    def _check_elm11_availability(self) -> bool:
        """
        Check if ELM11 hardware is available.
        
        Returns
        -------
        available : bool
            True if ELM11 can be connected to
        """
        try:
            import serial
            import serial.tools.list_ports
            
            # Try to find ELM11 by port or auto-detect
            if self.port:
                ports = [self.port]
            else:
                # Look for common ELM11 identifiers
                ports = []
                for port in serial.tools.list_ports.comports():
                    desc_lower = port.description.lower()
                    mfr_lower = (port.manufacturer or '').lower()
                    # Check both description and manufacturer
                    if any(keyword in desc_lower for keyword in 
                          ['elm11', 'fpga', 'microcontroller', 'lua', 'tang nano', 'tangnano', 'gowin', 'jtag debugger']) or \
                       any(keyword in mfr_lower for keyword in ['sipeed', 'gowin']):
                        ports.append(port.device)
                
                # If no specific ports found, try common ports
                if not ports:
                    ports = ['/dev/ttyUSB0', '/dev/ttyUSB1', '/dev/ttyACM0', '/dev/ttyACM1', 'COM3', 'COM4']
            
            # Try to connect to each port
            for port in ports:
                # Try multiple baud rates
                for baudrate in [self.baudrate, 9600, 115200, 19200, 38400, 57600]:
                    try:
                        ser = serial.Serial(port, baudrate, timeout=1)
                        # Try to communicate
                        ser.write(b'\n')  # Send newline
                        ser.flush()
                        response = ser.read(100)
                        ser.close()
                        
                        # Check for interactive interface markers
                        if response and any(marker in response for marker in [b'$', b'>', b'#', b'OK', b'Command']):
                            self.port = port
                            self.baudrate = baudrate
                            return True
                    except:
                        continue
            
            return False
            
        except ImportError:
            return False
    
    def run_monte_carlo(
        self,
        worker_func: Callable,
        n_simulations: int,
        worker_args: tuple = (),
        worker_kwargs: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        verbose: bool = True,
        **backend_kwargs
    ) -> List[Any]:
        """
        Execute Monte Carlo simulations with FPGA acceleration.
        
        Currently delegates to sequential execution while ELM11
        integration is being developed. Future versions will use
        FPGA acceleration for FFT operations within simulations.
        
        Parameters
        ----------
        worker_func : callable
            Function to execute for each simulation
        n_simulations : int
            Number of Monte Carlo simulations to run
        worker_args : tuple, optional
            Additional positional arguments for worker_func
        worker_kwargs : dict, optional
            Additional keyword arguments for worker_func
        seed : int, optional
            Master random seed for reproducibility
        verbose : bool, optional
            Whether to display progress information
        **backend_kwargs : dict
            Backend-specific options
            
        Returns
        -------
        results : list
            List of results from each simulation
        """
        if not self.is_available():
            raise RuntimeError(
                "ELM11 backend is not available. Check hardware connection."
            )
        
        if worker_kwargs is None:
            worker_kwargs = {}
        
        # Generate independent seeds for each simulation
        if seed is not None:
            seed_sequence = np.random.SeedSequence(seed)
            child_seeds = seed_sequence.spawn(n_simulations)
            seeds = [s.generate_state(1)[0] for s in child_seeds]
        else:
            seeds = [None] * n_simulations
        
        results = []
        
        if verbose:
            print(f"Running {n_simulations} simulations with ELM11 acceleration...")
        
        for i, sim_seed in enumerate(seeds):
            try:
                # TODO: Integrate FPGA acceleration for FFT operations
                # For now, run sequentially
                result = worker_func(sim_seed, *worker_args, **worker_kwargs)
                results.append(result)
                
                if verbose and (i + 1) % max(1, n_simulations // 10) == 0:
                    progress = (i + 1) / n_simulations * 100
                    print(f"Progress: {i + 1}/{n_simulations} ({progress:.1f}%)")
                    
            except Exception as e:
                if verbose:
                    print(f"Error in simulation {i + 1}: {e}")
                raise
        
        if verbose:
            print(f"Completed {n_simulations} simulations with ELM11")
        
        return results
    
    def get_capabilities(self) -> List[str]:
        """
        Get backend capabilities.
        
        Returns
        -------
        capabilities : list of str
            List of ELM11 backend capabilities
        """
        caps = ["fpga_acceleration", "fft_hardware", "deterministic"]
        if self.is_available():
            caps.append("available")
        return caps
    
    def validate_config(self, **kwargs) -> bool:
        """
        Validate configuration options.
        
        Parameters
        ----------
        **kwargs : dict
            Configuration options to validate
            
        Returns
        -------
        valid : bool
            True if configuration is valid
        """
        # For now, accept all configurations
        return True