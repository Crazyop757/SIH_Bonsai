# File 3: synthetic_data_generator.py

synthetic_data_generator_code = '''
"""
Synthetic FRA Data Generator
Main class for generating large-scale synthetic FRA datasets
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import h5py
import json
from pathlib import Path
from tqdm import tqdm
import logging

# Import our custom modules (assuming they're in the same directory)
# from transformer_circuit_model import TransformerCircuitModel
# from fault_simulation import FaultSimulator, FaultType

class SyntheticFRADataGenerator:
    """
    Main class for generating synthetic FRA datasets for machine learning
    """
    
    def __init__(self, output_dir: str = "./synthetic_fra_data", random_seed: int = 42):
        """
        Initialize the synthetic data generator
        
        Args:
            output_dir: Directory to save generated datasets
            random_seed: Random seed for reproducibility
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        np.random.seed(random_seed)
        
        # Initialize components
        # self.circuit_model = TransformerCircuitModel()
        # self.fault_simulator = FaultSimulator(random_seed)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Dataset configuration
        self.dataset_config = {
            'frequency_points': 1000,
            'voltage_levels': [11, 33, 66, 132, 220, 400],  # kV
            'power_ratings': [1, 5, 10, 25, 50, 100, 200],  # MVA
            'noise_levels': [0.05, 0.1, 0.15, 0.2],  # Measurement noise
            'temperature_variations': [-20, -10, 0, 10, 20, 40]  # Celsius
        }
    
    def generate_transformer_population(self, n_transformers: int = 1000) -> List[Dict]:
        """
        Generate a diverse population of transformer specifications
        
        Args:
            n_transformers: Number of different transformer designs
            
        Returns:
            List of transformer specification dictionaries
        """
        transformers = []
        
        for i in range(n_transformers):
            # Random transformer specifications
            voltage = np.random.choice(self.dataset_config['voltage_levels'])
            power = np.random.choice(self.dataset_config['power_ratings'])
            
            # Winding configuration
            winding_type = np.random.choice(['layer', 'disc', 'helical'])
            n_sections = np.random.randint(5, 20)  # Number of winding sections
            
            # Operating conditions
            temperature = np.random.choice(self.dataset_config['temperature_variations'])
            age = np.random.uniform(0, 40)  # Years in service
            
            transformer_spec = {
                'id': f'T_{i:04d}',
                'voltage_level': voltage,
                'power_rating': power,
                'winding_type': winding_type,
                'n_sections': n_sections,
                'operating_temperature': temperature,
                'age_years': age
            }
            
            transformers.append(transformer_spec)
        
        return transformers
    
    def generate_single_fra_sample(self, transformer_spec: Dict) -> Dict:
        """
        Generate a single FRA measurement sample
        
        Args:
            transformer_spec: Transformer specifications
            
        Returns:
            Dictionary containing FRA data and metadata
        """
        # Initialize circuit model with transformer specs
        circuit_model = TransformerCircuitModel(
            n_sections=transformer_spec['n_sections'],
            voltage_level=transformer_spec['voltage_level'],
            power_rating=transformer_spec['power_rating']
        )
        
        # Generate random fault scenario
        fault_simulator = FaultSimulator()
        fault_type, severity, affected_sections = fault_simulator.generate_random_fault_scenario()
        
        # Apply fault to parameters
        base_params = circuit_model.baseline_params
        faulty_params = fault_simulator.apply_fault(
            base_params, fault_type, severity, affected_sections
        )
        
        # Calculate frequency response
        frequencies, magnitude_db, phase_degrees = circuit_model.calculate_frequency_response(faulty_params)
        
        # Add measurement noise
        noise_level = np.random.choice(self.dataset_config['noise_levels'])
        magnitude_noisy, phase_noisy = circuit_model.add_measurement_noise(
            magnitude_db, phase_degrees, noise_level
        )
        
        # Add temperature effects (simplified model)
        temp_factor = 1 + (transformer_spec['operating_temperature'] / 100) * 0.02
        magnitude_noisy *= temp_factor
        
        # Add aging effects (simplified model)
        aging_factor = 1 + (transformer_spec['age_years'] / 40) * 0.05
        phase_noisy *= aging_factor
        
        # Compile sample data
        sample = {
            'transformer_id': transformer_spec['id'],
            'frequencies': frequencies,
            'magnitude_db': magnitude_noisy,
            'phase_degrees': phase_noisy,
            'fault_type': fault_type.value,
            'fault_severity': severity,
            'affected_sections': affected_sections,
            'transformer_specs': transformer_spec,
            'noise_level': noise_level,
            'circuit_parameters': faulty_params
        }
        
        return sample
    
    def generate_dataset(self, n_samples: int = 10000, 
                        validation_split: float = 0.2,
                        test_split: float = 0.1) -> Dict:
        """
        Generate complete synthetic FRA dataset
        
        Args:
            n_samples: Total number of samples to generate
            validation_split: Fraction for validation set
            test_split: Fraction for test set
            
        Returns:
            Dictionary with train/validation/test splits
        """
        self.logger.info(f"Generating {n_samples} synthetic FRA samples...")
        
        # Generate transformer population
        n_transformers = min(1000, n_samples // 5)  # Each transformer gets ~5 samples
        transformer_population = self.generate_transformer_population(n_transformers)
        
        # Generate samples
        samples = []
        for i in tqdm(range(n_samples), desc="Generating samples"):
            # Select random transformer from population
            transformer = np.random.choice(transformer_population)
            
            # Generate sample
            try:
                sample = self.generate_single_fra_sample(transformer)
                samples.append(sample)
            except Exception as e:
                self.logger.warning(f"Failed to generate sample {i}: {str(e)}")
                continue
        
        # Split dataset
        np.random.shuffle(samples)
        n_train = int(len(samples) * (1 - validation_split - test_split))
        n_val = int(len(samples) * validation_split)
        
        dataset = {
            'train': samples[:n_train],
            'validation': samples[n_train:n_train + n_val],
            'test': samples[n_train + n_val:],
            'metadata': {
                'total_samples': len(samples),
                'train_samples': len(samples[:n_train]),
                'validation_samples': len(samples[n_train:n_train + n_val]),
                'test_samples': len(samples[n_train + n_val:]),
                'generation_config': self.dataset_config
            }
        }
        
        self.logger.info(f"Generated {len(samples)} samples successfully")
        return dataset
    
    def export_to_csv(self, dataset: Dict, filename: str = "synthetic_fra_dataset.csv"):
        """Export dataset to CSV format"""
        all_samples = dataset['train'] + dataset['validation'] + dataset['test']
        
        # Flatten data for CSV export
        csv_data = []
        for sample in all_samples:
            # Create features from frequency response
            mag_features = {f'mag_{i}': val for i, val in enumerate(sample['magnitude_db'])}
            phase_features = {f'phase_{i}': val for i, val in enumerate(sample['phase_degrees'])}
            
            # Metadata
            meta_data = {
                'transformer_id': sample['transformer_id'],
                'fault_type': sample['fault_type'],
                'fault_severity': sample['fault_severity'],
                'voltage_level': sample['transformer_specs']['voltage_level'],
                'power_rating': sample['transformer_specs']['power_rating'],
                'noise_level': sample['noise_level']
            }
            
            # Combine all features
            row_data = {**meta_data, **mag_features, **phase_features}
            csv_data.append(row_data)
        
        # Save to CSV
        df = pd.DataFrame(csv_data)
        csv_path = self.output_dir / filename
        df.to_csv(csv_path, index=False)
        self.logger.info(f"Exported CSV dataset to {csv_path}")
        
        return csv_path
    
    def export_to_hdf5(self, dataset: Dict, filename: str = "synthetic_fra_dataset.h5"):
        """Export dataset to HDF5 format (recommended for large datasets)"""
        hdf5_path = self.output_dir / filename
        
        with h5py.File(hdf5_path, 'w') as f:
            for split_name in ['train', 'validation', 'test']:
                if split_name not in dataset:
                    continue
                    
                split_data = dataset[split_name]
                if not split_data:
                    continue
                
                # Create group for this split
                split_group = f.create_group(split_name)
                
                # Convert to numpy arrays
                n_samples = len(split_data)
                n_freq = len(split_data[0]['frequencies'])
                
                # Initialize arrays
                frequencies = np.array(split_data[0]['frequencies'])
                magnitudes = np.zeros((n_samples, n_freq))
                phases = np.zeros((n_samples, n_freq))
                fault_types = np.zeros(n_samples, dtype=np.int32)
                fault_severities = np.zeros(n_samples)
                transformer_ids = []
                
                # Fill arrays
                for i, sample in enumerate(split_data):
                    magnitudes[i] = sample['magnitude_db']
                    phases[i] = sample['phase_degrees']
                    fault_types[i] = sample['fault_type']
                    fault_severities[i] = sample['fault_severity']
                    transformer_ids.append(sample['transformer_id'])
                
                # Save to HDF5
                split_group.create_dataset('frequencies', data=frequencies)
                split_group.create_dataset('magnitudes', data=magnitudes)
                split_group.create_dataset('phases', data=phases)
                split_group.create_dataset('fault_types', data=fault_types)
                split_group.create_dataset('fault_severities', data=fault_severities)
                split_group.create_dataset('transformer_ids', 
                                         data=[s.encode() for s in transformer_ids])
            
            # Save metadata
            f.attrs['metadata'] = json.dumps(dataset['metadata'])
        
        self.logger.info(f"Exported HDF5 dataset to {hdf5_path}")
        return hdf5_path
    
    def validate_dataset_quality(self, dataset: Dict) -> Dict:
        """
        Perform quality validation on the generated dataset
        
        Returns:
            Validation report dictionary
        """
        all_samples = dataset['train'] + dataset['validation'] + dataset['test']
        
        # Fault type distribution
        fault_counts = {}
        severities = []
        
        for sample in all_samples:
            fault_type = sample['fault_type']
            fault_counts[fault_type] = fault_counts.get(fault_type, 0) + 1
            if sample['fault_severity'] > 0:
                severities.append(sample['fault_severity'])
        
        # Statistical analysis
        validation_report = {
            'total_samples': len(all_samples),
            'fault_distribution': fault_counts,
            'severity_statistics': {
                'mean': np.mean(severities) if severities else 0,
                'std': np.std(severities) if severities else 0,
                'min': np.min(severities) if severities else 0,
                'max': np.max(severities) if severities else 0
            },
            'frequency_range': {
                'min': float(all_samples[0]['frequencies'][0]),
                'max': float(all_samples[0]['frequencies'][-1]),
                'n_points': len(all_samples[0]['frequencies'])
            },
            'transformer_diversity': len(set(s['transformer_id'] for s in all_samples))
        }
        
        return validation_report

# Example usage and testing functions
def create_small_test_dataset():
    """Create a small test dataset for validation"""
    generator = SyntheticFRADataGenerator()
    dataset = generator.generate_dataset(n_samples=100)
    
    # Export in different formats
    csv_path = generator.export_to_csv(dataset, "test_dataset.csv")
    hdf5_path = generator.export_to_hdf5(dataset, "test_dataset.h5")
    
    # Validate quality
    validation_report = generator.validate_dataset_quality(dataset)
    print("Dataset Validation Report:")
    print(json.dumps(validation_report, indent=2))
    
    return dataset, csv_path, hdf5_path
'''

print("Created synthetic_data_generator.py")