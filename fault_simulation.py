# File 2: fault_simulation.py

"""
Fault Simulation Module for FRA Synthetic Data Generation
Implements various transformer fault types and their parameter modifications
"""

import numpy as np
from typing import Dict, Tuple, List
from enum import Enum

class FaultType(Enum):
    """Enumeration of transformer fault types"""
    HEALTHY = 0
    AXIAL_DISPLACEMENT = 1
    RADIAL_DEFORMATION = 2
    TURN_TO_TURN_SHORT = 3
    CORE_FAULT = 4
    WINDING_CONNECTION = 5
    INSULATION_DEGRADATION = 6

class FaultSimulator:
    """
    Simulates various transformer faults by modifying circuit parameters
    """
    
    def __init__(self, random_seed: int = 42):
        """Initialize fault simulator"""
        np.random.seed(random_seed)
        
        # Define fault parameter ranges
        self.fault_ranges = {
            FaultType.AXIAL_DISPLACEMENT: {
                'inductance_change': (-0.3, 0.3),      # ±30% inductance change
                'capacitance_change': (-0.2, 0.2),     # ±20% capacitance change
                'frequency_bands': [(20, 2000), (2000, 20000)]  # Most affected bands
            },
            FaultType.RADIAL_DEFORMATION: {
                'inductance_change': (-0.25, 0.25),    # ±25% inductance change
                'capacitance_change': (-0.4, 0.4),     # ±40% capacitance change
                'frequency_bands': [(2000, 200000)]     # Medium to high frequency
            },
            FaultType.TURN_TO_TURN_SHORT: {
                'resistance_change': (-0.8, -0.1),     # 10-80% resistance reduction
                'inductance_change': (-0.6, -0.1),     # 10-60% inductance reduction
                'frequency_bands': [(20, 20000)]        # Low to medium frequency
            },
            FaultType.CORE_FAULT: {
                'core_resistance_change': (-0.7, 0.5),  # Core loss changes
                'core_inductance_change': (-0.4, 0.2),  # Magnetizing inductance
                'frequency_bands': [(20, 1000)]          # Low frequency dominated
            },
            FaultType.WINDING_CONNECTION: {
                'resistance_change': (0.1, 2.0),        # Increased resistance
                'inductance_change': (-0.3, 0.5),       # Variable inductance
                'frequency_bands': [(1000, 100000)]      # Medium frequency
            },
            FaultType.INSULATION_DEGRADATION: {
                'capacitance_change': (0.2, 1.5),       # Increased capacitance
                'resistance_change': (-0.5, 0.3),       # Variable resistance
                'frequency_bands': [(10000, 1000000)]    # High frequency
            }
        }
    
    def apply_fault(self, base_params: Dict, fault_type: FaultType, 
                   severity: float, affected_sections: List[int] = None) -> Dict:
        """
        Apply fault simulation to transformer parameters
        
        Args:
            base_params: Baseline circuit parameters
            fault_type: Type of fault to simulate
            severity: Fault severity (0-100%)
            affected_sections: Which winding sections are affected
            
        Returns:
            Modified parameters with fault effects
        """
        if fault_type == FaultType.HEALTHY:
            return base_params.copy()
        
        # Copy parameters to avoid modifying original
        faulty_params = {key: val.copy() if isinstance(val, np.ndarray) 
                        else val for key, val in base_params.items()}
        
        # Determine affected sections
        if affected_sections is None:
            n_sections = len(base_params['series_resistance'])
            # Randomly select 20-80% of sections to be affected
            n_affected = int(np.random.uniform(0.2, 0.8) * n_sections)
            affected_sections = np.random.choice(n_sections, n_affected, replace=False)
        
        # Apply fault-specific modifications
        if fault_type == FaultType.AXIAL_DISPLACEMENT:
            faulty_params = self._simulate_axial_displacement(
                faulty_params, severity, affected_sections)
                
        elif fault_type == FaultType.RADIAL_DEFORMATION:
            faulty_params = self._simulate_radial_deformation(
                faulty_params, severity, affected_sections)
                
        elif fault_type == FaultType.TURN_TO_TURN_SHORT:
            faulty_params = self._simulate_turn_short(
                faulty_params, severity, affected_sections)
                
        elif fault_type == FaultType.CORE_FAULT:
            faulty_params = self._simulate_core_fault(
                faulty_params, severity)
                
        elif fault_type == FaultType.WINDING_CONNECTION:
            faulty_params = self._simulate_connection_fault(
                faulty_params, severity, affected_sections)
                
        elif fault_type == FaultType.INSULATION_DEGRADATION:
            faulty_params = self._simulate_insulation_fault(
                faulty_params, severity, affected_sections)
        
        return faulty_params
    
    def _simulate_axial_displacement(self, params: Dict, severity: float, 
                                   sections: List[int]) -> Dict:
        """Simulate axial displacement effects"""
        severity_factor = severity / 100.0
        
        # Axial displacement affects mutual inductance between sections
        inductance_change = np.random.uniform(*self.fault_ranges[FaultType.AXIAL_DISPLACEMENT]['inductance_change'])
        capacitance_change = np.random.uniform(*self.fault_ranges[FaultType.AXIAL_DISPLACEMENT]['capacitance_change'])
        
        for section in sections:
            # Modify inductance (coupling changes)
            params['series_inductance'][section] *= (1 + inductance_change * severity_factor)
            
            # Modify capacitance (spacing changes)
            params['series_capacitance'][section] *= (1 + capacitance_change * severity_factor)
            params['shunt_capacitance_ground'][section] *= (1 + capacitance_change * severity_factor * 0.5)
        
        return params
    
    def _simulate_radial_deformation(self, params: Dict, severity: float, 
                                   sections: List[int]) -> Dict:
        """Simulate radial deformation effects"""
        severity_factor = severity / 100.0
        
        inductance_change = np.random.uniform(*self.fault_ranges[FaultType.RADIAL_DEFORMATION]['inductance_change'])
        capacitance_change = np.random.uniform(*self.fault_ranges[FaultType.RADIAL_DEFORMATION]['capacitance_change'])
        
        for section in sections:
            # Radial deformation primarily affects turn-to-turn spacing
            params['series_capacitance'][section] *= (1 + capacitance_change * severity_factor)
            params['series_inductance'][section] *= (1 + inductance_change * severity_factor)
            
            # Ground capacitance less affected
            params['shunt_capacitance_ground'][section] *= (1 + capacitance_change * severity_factor * 0.3)
        
        return params
    
    def _simulate_turn_short(self, params: Dict, severity: float, 
                           sections: List[int]) -> Dict:
        """Simulate turn-to-turn short circuit"""
        severity_factor = severity / 100.0
        
        resistance_change = np.random.uniform(*self.fault_ranges[FaultType.TURN_TO_TURN_SHORT]['resistance_change'])
        inductance_change = np.random.uniform(*self.fault_ranges[FaultType.TURN_TO_TURN_SHORT]['inductance_change'])
        
        for section in sections:
            # Short circuit reduces resistance and inductance
            params['series_resistance'][section] *= (1 + resistance_change * severity_factor)
            params['series_inductance'][section] *= (1 + inductance_change * severity_factor)
            
            # May increase local capacitance due to conductor proximity
            params['series_capacitance'][section] *= (1 + 0.2 * severity_factor)
        
        return params
    
    def _simulate_core_fault(self, params: Dict, severity: float) -> Dict:
        """Simulate core-related faults"""
        severity_factor = severity / 100.0
        
        core_r_change = np.random.uniform(*self.fault_ranges[FaultType.CORE_FAULT]['core_resistance_change'])
        core_l_change = np.random.uniform(*self.fault_ranges[FaultType.CORE_FAULT]['core_inductance_change'])
        
        # Modify core parameters
        params['core_resistance'] *= (1 + core_r_change * severity_factor)
        params['core_inductance'] *= (1 + core_l_change * severity_factor)
        
        return params
    
    def _simulate_connection_fault(self, params: Dict, severity: float, 
                                 sections: List[int]) -> Dict:
        """Simulate winding connection issues"""
        severity_factor = severity / 100.0
        
        resistance_change = np.random.uniform(*self.fault_ranges[FaultType.WINDING_CONNECTION]['resistance_change'])
        inductance_change = np.random.uniform(*self.fault_ranges[FaultType.WINDING_CONNECTION]['inductance_change'])
        
        # Usually affects only a few sections (connection points)
        affected_connections = sections[:max(1, len(sections)//3)]
        
        for section in affected_connections:
            params['series_resistance'][section] *= (1 + resistance_change * severity_factor)
            params['series_inductance'][section] *= (1 + inductance_change * severity_factor)
        
        return params
    
    def _simulate_insulation_fault(self, params: Dict, severity: float, 
                                 sections: List[int]) -> Dict:
        """Simulate insulation degradation"""
        severity_factor = severity / 100.0
        
        capacitance_change = np.random.uniform(*self.fault_ranges[FaultType.INSULATION_DEGRADATION]['capacitance_change'])
        resistance_change = np.random.uniform(*self.fault_ranges[FaultType.INSULATION_DEGRADATION]['resistance_change'])
        
        for section in sections:
            # Insulation degradation increases capacitance
            params['shunt_capacitance_ground'][section] *= (1 + capacitance_change * severity_factor)
            params['series_capacitance'][section] *= (1 + capacitance_change * severity_factor * 0.7)
            
            # May affect resistance due to tracking/carbonization
            params['series_resistance'][section] *= (1 + resistance_change * severity_factor)
        
        return params
    
    def generate_random_fault_scenario(self) -> Tuple[FaultType, float, List[int]]:
        """Generate a random fault scenario for Monte Carlo simulation"""
        
        # Select random fault type (with healthy bias)
        fault_types = list(FaultType)
        weights = [0.3, 0.15, 0.15, 0.1, 0.1, 0.1, 0.1]  # Bias towards healthy
        fault_type = np.random.choice(fault_types, p=weights)
        
        # Generate random severity
        if fault_type == FaultType.HEALTHY:
            severity = 0.0
        else:
            # Use beta distribution for realistic severity distribution
            severity = np.random.beta(2, 5) * 100  # Bias towards lower severities
        
        # Random affected sections (if applicable)
        n_sections = 10  # Default assumption
        if fault_type not in [FaultType.HEALTHY, FaultType.CORE_FAULT]:
            n_affected = int(np.random.uniform(0.1, 0.6) * n_sections)
            affected_sections = list(np.random.choice(n_sections, n_affected, replace=False))
        else:
            affected_sections = []
        
        return fault_type, severity, affected_sections


print("Created fault_simulation.py")