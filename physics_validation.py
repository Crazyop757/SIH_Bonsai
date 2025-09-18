# File 4: physics_validation.py

physics_validation_code = '''
"""
Physics Validation Module
Ensures synthetic FRA data adheres to physical constraints and causality
"""

import numpy as np
from scipy import signal
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

class PhysicsValidator:
    """
    Validates synthetic FRA data against physical constraints
    """
    
    def __init__(self, tolerance: float = 1e-3):
        """
        Initialize physics validator
        
        Args:
            tolerance: Numerical tolerance for validation checks
        """
        self.tolerance = tolerance
        
    def validate_causality(self, frequencies: np.ndarray, 
                          magnitude_db: np.ndarray, 
                          phase_degrees: np.ndarray) -> Dict:
        """
        Validate causality using Kramers-Kronig relations
        
        Args:
            frequencies: Frequency array in Hz
            magnitude_db: Magnitude in dB
            phase_degrees: Phase in degrees
            
        Returns:
            Validation results dictionary
        """
        # Convert to linear scale and radians
        magnitude_linear = 10**(magnitude_db / 20)
        phase_rad = phase_degrees * np.pi / 180
        
        # Create complex response
        complex_response = magnitude_linear * np.exp(1j * phase_rad)
        
        # Check causality using Kramers-Kronig
        omega = 2 * np.pi * frequencies
        
        # Compute real part from imaginary part (simplified KK check)
        real_computed = self._kramers_kronig_real_from_imag(omega, complex_response.imag)
        real_actual = complex_response.real
        
        # Calculate error
        causality_error = np.mean(np.abs(real_computed - real_actual) / 
                                 (np.abs(real_actual) + self.tolerance))
        
        # Check if causal (error should be small)
        is_causal = causality_error < 0.1  # 10% tolerance
        
        return {
            'is_causal': is_causal,
            'causality_error': float(causality_error),
            'max_error': float(np.max(np.abs(real_computed - real_actual))),
            'mean_error': float(np.mean(np.abs(real_computed - real_actual)))
        }
    
    def _kramers_kronig_real_from_imag(self, omega: np.ndarray, 
                                      imag_part: np.ndarray) -> np.ndarray:
        """
        Compute real part from imaginary part using Kramers-Kronig relations
        (Simplified numerical implementation)
        """
        real_part = np.zeros_like(omega)
        
        for i, w in enumerate(omega):
            # Avoid singularity at w = omega[i]
            mask = np.abs(omega - w) > self.tolerance
            if np.any(mask):
                integrand = (omega[mask] * imag_part[mask]) / (omega[mask]**2 - w**2)
                # Simple trapezoidal integration
                real_part[i] = -(2/np.pi) * np.trapz(integrand, omega[mask])
        
        return real_part
    
    def validate_passivity(self, frequencies: np.ndarray,
                          complex_impedance: np.ndarray) -> Dict:
        """
        Validate that the circuit remains passive (no energy generation)
        
        Args:
            frequencies: Frequency array
            complex_impedance: Complex impedance array
            
        Returns:
            Passivity validation results
        """
        # For passive circuits, real part of impedance should be non-negative
        real_impedance = complex_impedance.real
        
        # Check for negative resistance (non-passive behavior)
        negative_resistance_points = np.sum(real_impedance < -self.tolerance)
        min_resistance = np.min(real_impedance)
        
        is_passive = negative_resistance_points == 0
        
        return {
            'is_passive': is_passive,
            'negative_resistance_points': int(negative_resistance_points),
            'min_resistance': float(min_resistance),
            'passivity_margin': float(min_resistance) if is_passive else float(min_resistance)
        }
    
    def validate_stability(self, circuit_params: Dict) -> Dict:
        """
        Validate circuit stability by checking pole locations
        
        Args:
            circuit_params: Circuit parameter dictionary
            
        Returns:
            Stability validation results
        """
        try:
            # Create state-space representation for stability analysis
            # Simplified approach: check if all time constants are positive
            
            # Extract time constants from RLC parameters
            R = circuit_params.get('series_resistance', [1])
            L = circuit_params.get('series_inductance', [1e-3])
            C = circuit_params.get('series_capacitance', [1e-9])
            
            # Ensure arrays
            R = np.atleast_1d(R)
            L = np.atleast_1d(L)
            C = np.atleast_1d(C)
            
            # Calculate time constants
            tau_RL = L / (R + self.tolerance)  # Avoid division by zero
            tau_RC = R * C
            
            # All time constants should be positive for stability
            is_stable = np.all(tau_RL > 0) and np.all(tau_RC > 0)
            
            # Check for reasonable ranges
            reasonable_tau_RL = np.all(tau_RL < 1.0)  # Less than 1 second
            reasonable_tau_RC = np.all(tau_RC < 1.0)  # Less than 1 second
            
            return {
                'is_stable': is_stable,
                'reasonable_time_constants': reasonable_tau_RL and reasonable_tau_RC,
                'max_tau_RL': float(np.max(tau_RL)),
                'max_tau_RC': float(np.max(tau_RC)),
                'min_tau_RL': float(np.min(tau_RL)),
                'min_tau_RC': float(np.min(tau_RC))
            }
            
        except Exception as e:
            return {
                'is_stable': False,
                'error': str(e),
                'reasonable_time_constants': False
            }
    
    def validate_frequency_response_characteristics(self, frequencies: np.ndarray,
                                                  magnitude_db: np.ndarray) -> Dict:
        """
        Validate that frequency response exhibits expected transformer characteristics
        
        Args:
            frequencies: Frequency array
            magnitude_db: Magnitude response in dB
            
        Returns:
            Characteristic validation results
        """
        log_freq = np.log10(frequencies)
        
        # Expected characteristics for transformer FRA:
        # 1. Low frequency: relatively flat or decreasing
        # 2. Mid frequency: may have resonances/anti-resonances
        # 3. High frequency: generally increasing due to capacitive effects
        
        # Analyze different frequency bands
        low_freq_mask = frequencies < 1000  # Below 1 kHz
        mid_freq_mask = (frequencies >= 1000) & (frequencies < 100000)  # 1-100 kHz
        high_freq_mask = frequencies >= 100000  # Above 100 kHz
        
        # Calculate slopes in each band
        low_freq_slope = self._calculate_slope(log_freq[low_freq_mask], 
                                             magnitude_db[low_freq_mask])
        high_freq_slope = self._calculate_slope(log_freq[high_freq_mask], 
                                              magnitude_db[high_freq_mask])
        
        # Check for reasonable dynamic range
        magnitude_range = np.max(magnitude_db) - np.min(magnitude_db)
        has_reasonable_range = 10 < magnitude_range < 200  # 10-200 dB range
        
        # Check for excessive noise/oscillations
        magnitude_diff = np.diff(magnitude_db)
        noise_level = np.std(magnitude_diff)
        low_noise = noise_level < 5.0  # Less than 5 dB std deviation
        
        return {
            'low_freq_slope': float(low_freq_slope),
            'high_freq_slope': float(high_freq_slope),
            'magnitude_range_db': float(magnitude_range),
            'has_reasonable_range': has_reasonable_range,
            'noise_level': float(noise_level),
            'low_noise': low_noise,
            'physically_reasonable': (has_reasonable_range and low_noise and 
                                    abs(low_freq_slope) < 40 and high_freq_slope > -40)
        }
    
    def _calculate_slope(self, x: np.ndarray, y: np.ndarray) -> float:
        """Calculate slope using linear regression"""
        if len(x) < 2:
            return 0.0
        
        # Simple linear regression
        A = np.vstack([x, np.ones(len(x))]).T
        try:
            slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]
            return slope
        except:
            return 0.0
    
    def validate_parameter_ranges(self, circuit_params: Dict) -> Dict:
        """
        Validate that circuit parameters are within realistic ranges
        
        Args:
            circuit_params: Circuit parameter dictionary
            
        Returns:
            Parameter range validation results
        """
        # Define realistic parameter ranges for power transformers
        realistic_ranges = {
            'series_resistance': (0.001, 1000.0),      # 1 mΩ to 1 kΩ
            'series_inductance': (1e-6, 100.0),        # 1 μH to 100 H
            'series_capacitance': (1e-15, 1e-6),       # 1 fF to 1 μF
            'shunt_capacitance_ground': (1e-15, 1e-6), # 1 fF to 1 μF
            'core_inductance': (0.01, 1000.0),         # 10 mH to 1 kH
            'core_resistance': (1.0, 1e6)              # 1 Ω to 1 MΩ
        }
        
        validation_results = {}
        overall_valid = True
        
        for param_name, (min_val, max_val) in realistic_ranges.items():
            if param_name in circuit_params:
                param_values = np.atleast_1d(circuit_params[param_name])
                
                within_range = np.all((param_values >= min_val) & 
                                    (param_values <= max_val))
                
                validation_results[param_name] = {
                    'within_range': within_range,
                    'min_value': float(np.min(param_values)),
                    'max_value': float(np.max(param_values)),
                    'expected_range': (min_val, max_val)
                }
                
                overall_valid = overall_valid and within_range
        
        validation_results['overall_valid'] = overall_valid
        return validation_results
    
    def comprehensive_validation(self, frequencies: np.ndarray,
                                magnitude_db: np.ndarray,
                                phase_degrees: np.ndarray,
                                circuit_params: Dict) -> Dict:
        """
        Perform comprehensive physics validation
        
        Returns:
            Complete validation report
        """
        # Convert to complex impedance for some tests
        magnitude_linear = 10**(magnitude_db / 20)
        phase_rad = phase_degrees * np.pi / 180
        complex_impedance = magnitude_linear * np.exp(1j * phase_rad)
        
        # Run all validation tests
        causality_results = self.validate_causality(frequencies, magnitude_db, phase_degrees)
        passivity_results = self.validate_passivity(frequencies, complex_impedance)
        stability_results = self.validate_stability(circuit_params)
        characteristic_results = self.validate_frequency_response_characteristics(frequencies, magnitude_db)
        parameter_results = self.validate_parameter_ranges(circuit_params)
        
        # Overall validation score
        validation_checks = [
            causality_results['is_causal'],
            passivity_results['is_passive'],
            stability_results['is_stable'],
            characteristic_results['physically_reasonable'],
            parameter_results['overall_valid']
        ]
        
        validation_score = sum(validation_checks) / len(validation_checks)
        
        return {
            'validation_score': validation_score,
            'overall_valid': validation_score >= 0.8,  # 80% of tests must pass
            'causality': causality_results,
            'passivity': passivity_results,
            'stability': stability_results,
            'characteristics': characteristic_results,
            'parameters': parameter_results,
            'summary': {
                'total_checks': len(validation_checks),
                'passed_checks': sum(validation_checks),
                'failed_checks': len(validation_checks) - sum(validation_checks)
            }
        }

# Example usage functions
def validate_sample_data():
    """Example function to validate a sample FRA response"""
    # Create synthetic test data
    frequencies = np.logspace(1, 6, 1000)  # 10 Hz to 1 MHz
    
    # Simple RLC response for testing
    R, L, C = 1.0, 1e-3, 1e-9
    omega = 2 * np.pi * frequencies
    Z = R + 1j * omega * L + 1 / (1j * omega * C)
    
    magnitude_db = 20 * np.log10(np.abs(Z))
    phase_degrees = np.angle(Z) * 180 / np.pi
    
    circuit_params = {
        'series_resistance': [R],
        'series_inductance': [L],
        'series_capacitance': [C],
        'shunt_capacitance_ground': [1e-12],
        'core_inductance': 1.0,
        'core_resistance': 1000.0
    }
    
    # Validate
    validator = PhysicsValidator()
    results = validator.comprehensive_validation(
        frequencies, magnitude_db, phase_degrees, circuit_params
    )
    
    print("Physics Validation Results:")
    print(f"Overall Valid: {results['overall_valid']}")
    print(f"Validation Score: {results['validation_score']:.3f}")
    print(f"Passed Checks: {results['summary']['passed_checks']}/{results['summary']['total_checks']}")
    
    return results
'''

print("Created physics_validation.py")