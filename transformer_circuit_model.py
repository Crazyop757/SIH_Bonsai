# Create the complete FRA synthetic data generation framework
# File 1: transformer_circuit_model.py

transformer_circuit_model_code = '''
"""
Transformer Circuit Model for FRA Simulation
Physics-based equivalent circuit modeling of power transformers
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.optimize import minimize
import pandas as pd
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

class TransformerCircuitModel:
    """
    Physics-based transformer equivalent circuit model for FRA simulation
    """
    
    def __init__(self, n_sections: int = 10, voltage_level: float = 132.0, 
                 power_rating: float = 50.0):
        """
        Initialize transformer circuit model
        
        Args:
            n_sections: Number of winding sections to model
            voltage_level: Transformer voltage level in kV
            power_rating: Transformer power rating in MVA
        """
        self.n_sections = n_sections
        self.voltage_level = voltage_level  # kV
        self.power_rating = power_rating    # MVA
        self.frequency_range = np.logspace(1, 6, 1000)  # 10 Hz to 1 MHz
        
        # Initialize baseline parameters
        self.baseline_params = self._initialize_baseline_parameters()
        
    def _initialize_baseline_parameters(self) -> Dict:
        """Initialize baseline RLC parameters based on transformer specifications"""
        
        # Scale parameters based on voltage and power rating
        voltage_scale = self.voltage_level / 132.0  # Normalized to 132kV
        power_scale = self.power_rating / 50.0      # Normalized to 50MVA
        
        # Baseline parameters (typical values for power transformers)
        params = {
            'series_resistance': np.linspace(0.1, 2.0, self.n_sections) * voltage_scale,
            'series_inductance': np.linspace(0.001, 0.1, self.n_sections) * power_scale,
            'shunt_capacitance_ground': np.linspace(100e-12, 1e-9, self.n_sections) / voltage_scale,
            'series_capacitance': np.linspace(10e-12, 100e-12, self.n_sections) / voltage_scale,
            'core_inductance': 10.0 * power_scale,  # Core magnetizing inductance
            'core_resistance': 1000.0 * voltage_scale**2  # Core loss resistance
        }
        
        return params
    
    def calculate_frequency_response(self, params: Dict = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate frequency response of the transformer circuit
        
        Args:
            params: Circuit parameters (uses baseline if None)
            
        Returns:
            frequencies, magnitude_db, phase_degrees
        """
        if params is None:
            params = self.baseline_params
            
        frequencies = self.frequency_range
        omega = 2 * np.pi * frequencies
        
        # Initialize impedance calculation
        z_total = np.zeros(len(frequencies), dtype=complex)
        
        for i, freq in enumerate(frequencies):
            # Calculate section impedances
            z_series_r = params['series_resistance']
            z_series_l = 1j * omega[i] * params['series_inductance']
            z_series_c = 1 / (1j * omega[i] * params['series_capacitance'])
            z_shunt_c = 1 / (1j * omega[i] * params['shunt_capacitance_ground'])
            
            # Series impedance per section
            z_series = z_series_r + z_series_l + z_series_c
            
            # Calculate total impedance using ladder network
            z_section = z_series[0]
            for j in range(1, self.n_sections):
                # Parallel combination with shunt capacitance
                z_parallel = 1 / (1/z_shunt_c[j-1] + 1/z_section)
                # Add next series section
                z_section = z_parallel + z_series[j]
            
            # Add core impedance (parallel combination)
            z_core = params['core_resistance'] * (1j * omega[i] * params['core_inductance']) / \\
                     (params['core_resistance'] + 1j * omega[i] * params['core_inductance'])
            
            z_total[i] = 1 / (1/z_section + 1/z_core)
        
        # Convert to magnitude and phase
        magnitude_db = 20 * np.log10(np.abs(z_total))
        phase_degrees = np.angle(z_total) * 180 / np.pi
        
        return frequencies, magnitude_db, phase_degrees
    
    def add_measurement_noise(self, magnitude_db: np.ndarray, phase_degrees: np.ndarray, 
                            noise_level: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Add realistic measurement noise to FRA data
        
        Args:
            magnitude_db: Magnitude in dB
            phase_degrees: Phase in degrees
            noise_level: Noise standard deviation
            
        Returns:
            Noisy magnitude and phase
        """
        # Add Gaussian noise
        mag_noise = np.random.normal(0, noise_level, len(magnitude_db))
        phase_noise = np.random.normal(0, noise_level * 2, len(phase_degrees))
        
        # Add frequency-dependent noise (higher at high frequencies)
        freq_factor = np.log10(self.frequency_range / 1000)
        freq_factor = np.clip(freq_factor, 0, 2)
        
        magnitude_noisy = magnitude_db + mag_noise * (1 + freq_factor)
        phase_noisy = phase_degrees + phase_noise * (1 + freq_factor)
        
        return magnitude_noisy, phase_noisy
    
    def plot_frequency_response(self, frequencies: np.ndarray, magnitude_db: np.ndarray, 
                              phase_degrees: np.ndarray, title: str = "FRA Response"):
        """Plot frequency response"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Magnitude plot
        ax1.semilogx(frequencies, magnitude_db, 'b-', linewidth=2)
        ax1.set_ylabel('Magnitude (dB)')
        ax1.set_title(title)
        ax1.grid(True, alpha=0.3)
        
        # Phase plot
        ax2.semilogx(frequencies, phase_degrees, 'r-', linewidth=2)
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Phase (degrees)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
'''

print("Created transformer_circuit_model.py")