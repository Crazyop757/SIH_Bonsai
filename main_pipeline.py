# File 6: main_pipeline.py - Main execution script


"""
Main Pipeline for FRA Synthetic Data Generation
Complete workflow for generating, validating, and exporting synthetic FRA data
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
import logging
from typing import Dict, List, Tuple
import argparse
from tqdm import tqdm

# Import our custom modules
# Uncomment these when running with actual files
# from transformer_circuit_model import TransformerCircuitModel
# from fault_simulation import FaultSimulator, FaultType
# from synthetic_data_generator import SyntheticFRADataGenerator
# from physics_validation import PhysicsValidator
# from data_export_utils import FRADataExporter

class FRAPipeline:
    """
    Main pipeline class for FRA synthetic data generation
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the FRA data generation pipeline
        
        Args:
            config: Configuration dictionary for pipeline parameters
        """
        # Default configuration
        self.config = {
            'n_samples': 10000,
            'validation_split': 0.2,
            'test_split': 0.1,
            'output_dir': './fra_synthetic_data',
            'random_seed': 42,
            'validate_physics': True,
            'export_formats': ['hdf5', 'csv', 'numpy'],
            'vendor_formats': ['omicron', 'megger'],
            'quality_threshold': 0.8
        }
        
        # Update with user config
        if config:
            self.config.update(config)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.data_generator = None
        self.validator = None
        self.exporter = None
        
        # Results storage
        self.dataset = None
        self.validation_results = None
    
    def setup_pipeline(self):
        """Initialize all pipeline components"""
        self.logger.info("Setting up FRA synthetic data generation pipeline...")
        
        # Initialize components
        # self.data_generator = SyntheticFRADataGenerator(
        #     output_dir=self.config['output_dir'],
        #     random_seed=self.config['random_seed']
        # )
        # self.validator = PhysicsValidator()
        # self.exporter = FRADataExporter(
        #     output_dir=Path(self.config['output_dir']) / 'exports'
        # )
        
        self.logger.info("Pipeline setup complete")
    
    def generate_dataset(self) -> Dict:
        """
        Generate synthetic FRA dataset
        
        Returns:
            Generated dataset dictionary
        """
        self.logger.info(f"Generating {self.config['n_samples']} synthetic FRA samples...")
        
        # Generate dataset
        # self.dataset = self.data_generator.generate_dataset(
        #     n_samples=self.config['n_samples'],
        #     validation_split=self.config['validation_split'],
        #     test_split=self.config['test_split']
        # )
        
        # For demonstration, create a mock dataset structure
        self.dataset = self._create_mock_dataset()
        
        self.logger.info("Dataset generation complete")
        return self.dataset
    
    def _create_mock_dataset(self) -> Dict:
        """Create mock dataset for demonstration"""
        n_samples = self.config['n_samples']
        n_freq = 1000
        
        # Generate mock data
        frequencies = np.logspace(1, 6, n_freq)
        
        samples = []
        for i in range(n_samples):
            # Simple mock FRA response
            omega = 2 * np.pi * frequencies
            R, L, C = 1.0 + np.random.normal(0, 0.1), 1e-3 + np.random.normal(0, 1e-4), 1e-9
            Z = R + 1j * omega * L + 1 / (1j * omega * C)
            
            magnitude_db = 20 * np.log10(np.abs(Z))
            phase_degrees = np.angle(Z) * 180 / np.pi
            
            # Add noise
            magnitude_db += np.random.normal(0, 0.1, len(magnitude_db))
            phase_degrees += np.random.normal(0, 1, len(phase_degrees))
            
            sample = {
                'transformer_id': f'T_{i:04d}',
                'frequencies': frequencies,
                'magnitude_db': magnitude_db,
                'phase_degrees': phase_degrees,
                'fault_type': np.random.randint(0, 7),
                'fault_severity': np.random.uniform(0, 100),
                'transformer_specs': {
                    'voltage_level': np.random.choice([11, 33, 132]),
                    'power_rating': np.random.choice([10, 50, 100])
                }
            }
            samples.append(sample)
        
        # Split dataset
        np.random.shuffle(samples)
        n_train = int(len(samples) * (1 - self.config['validation_split'] - self.config['test_split']))
        n_val = int(len(samples) * self.config['validation_split'])
        
        return {
            'train': samples[:n_train],
            'validation': samples[n_train:n_train + n_val],
            'test': samples[n_train + n_val:],
            'metadata': {
                'total_samples': len(samples),
                'train_samples': n_train,
                'validation_samples': n_val,
                'test_samples': len(samples) - n_train - n_val
            }
        }
    
    def validate_dataset_physics(self) -> Dict:
        """
        Validate dataset against physics constraints
        
        Returns:
            Validation results dictionary
        """
        if not self.config['validate_physics']:
            self.logger.info("Physics validation disabled")
            return {'validation_enabled': False}
        
        self.logger.info("Validating dataset physics...")
        
        # Sample validation on subset of data
        validation_samples = self.dataset['train'][:100]  # Validate first 100 samples
        
        validation_results = {
            'total_samples_validated': len(validation_samples),
            'passed_samples': 0,
            'failed_samples': 0,
            'average_score': 0.0,
            'individual_results': []
        }
        
        scores = []
        
        for i, sample in enumerate(tqdm(validation_samples, desc="Validating physics")):
            try:
                # Create mock circuit parameters for validation
                circuit_params = {
                    'series_resistance': [1.0],
                    'series_inductance': [1e-3],
                    'series_capacitance': [1e-9],
                    'shunt_capacitance_ground': [1e-12],
                    'core_inductance': 1.0,
                    'core_resistance': 1000.0
                }
                
                # Validate using mock validator
                # result = self.validator.comprehensive_validation(
                #     sample['frequencies'],
                #     sample['magnitude_db'],
                #     sample['phase_degrees'],
                #     circuit_params
                # )
                
                # Mock validation result
                score = 0.85 + np.random.normal(0, 0.1)
                score = np.clip(score, 0, 1)
                
                result = {
                    'validation_score': score,
                    'overall_valid': score >= self.config['quality_threshold']
                }
                
                scores.append(score)
                
                if result['overall_valid']:
                    validation_results['passed_samples'] += 1
                else:
                    validation_results['failed_samples'] += 1
                
                validation_results['individual_results'].append({
                    'sample_id': i,
                    'transformer_id': sample['transformer_id'],
                    'validation_score': result['validation_score'],
                    'valid': result['overall_valid']
                })
                
            except Exception as e:
                self.logger.warning(f"Validation failed for sample {i}: {str(e)}")
                validation_results['failed_samples'] += 1
        
        validation_results['average_score'] = float(np.mean(scores)) if scores else 0.0
        validation_results['pass_rate'] = validation_results['passed_samples'] / len(validation_samples)
        
        self.validation_results = validation_results
        
        self.logger.info(f"Physics validation complete. Pass rate: {validation_results['pass_rate']:.2%}")
        
        return validation_results
    
    def export_datasets(self) -> Dict[str, str]:
        """
        Export datasets in multiple formats (CSV, NumPy, HDF5)
        Returns:
            Dictionary of export paths
        """
        import os
        import numpy as np
        import pandas as pd
        try:
            import h5py
        except ImportError:
            h5py = None

        self.logger.info("Exporting datasets...")
        export_paths = {}
        export_dir = os.path.join(os.getcwd(), "exports")
        os.makedirs(export_dir, exist_ok=True)

        # Helper to extract features for CSV (statistical summaries instead of full arrays)
        def flatten_sample(sample):
            mag = sample['magnitude_db']
            phase = sample['phase_degrees']
            freq = sample['frequencies']
            
            return {
                'transformer_id': sample['transformer_id'],
                'fault_type': sample['fault_type'],
                'fault_severity': sample['fault_severity'],
                'voltage_level': sample['transformer_specs']['voltage_level'],
                'power_rating': sample['transformer_specs']['power_rating'],
                # Frequency range features
                'freq_min_hz': float(freq.min()),
                'freq_max_hz': float(freq.max()),
                # Magnitude statistical features
                'mag_mean_db': float(mag.mean()),
                'mag_std_db': float(mag.std()),
                'mag_min_db': float(mag.min()),
                'mag_max_db': float(mag.max()),
                'mag_range_db': float(mag.max() - mag.min()),
                # Phase statistical features
                'phase_mean_deg': float(phase.mean()),
                'phase_std_deg': float(phase.std()),
                'phase_min_deg': float(phase.min()),
                'phase_max_deg': float(phase.max()),
                'phase_range_deg': float(phase.max() - phase.min()),
                # Frequency band features (low, mid, high frequency characteristics)
                'mag_low_freq_mean': float(mag[:83].mean()),   # First 1/3 of frequency range
                'mag_mid_freq_mean': float(mag[83:167].mean()), # Middle 1/3
                'mag_high_freq_mean': float(mag[167:].mean()),  # Last 1/3
                'phase_low_freq_mean': float(phase[:83].mean()),
                'phase_mid_freq_mean': float(phase[83:167].mean()),
                'phase_high_freq_mean': float(phase[167:].mean()),
            }

        # CSV Export
        if 'csv' in self.config['export_formats']:
            try:
                all_samples = self.dataset['train'] + self.dataset['validation'] + self.dataset['test']
                df = pd.DataFrame([flatten_sample(s) for s in all_samples])
                csv_path = os.path.join(export_dir, 'synthetic_fra_dataset.csv')
                df.to_csv(csv_path, index=False)
                export_paths['csv'] = csv_path
                self.logger.info(f"Exported CSV format to {csv_path}")
            except Exception as e:
                self.logger.error(f"Failed to export CSV format: {str(e)}")

        # NumPy Export
        if 'numpy' in self.config['export_formats']:
            try:
                npy_path = os.path.join(export_dir, 'synthetic_fra_dataset.npy')
                # Save train/validation/test splits as dict
                np.save(npy_path, self._convert_to_json_compatible(self.dataset))
                export_paths['numpy'] = npy_path
                self.logger.info(f"Exported NumPy format to {npy_path}")
            except Exception as e:
                self.logger.error(f"Failed to export NumPy format: {str(e)}")

        # HDF5 Export
        if 'hdf5' in self.config['export_formats']:
            if h5py is None:
                self.logger.error("h5py not installed, skipping HDF5 export.")
            else:
                try:
                    hdf5_path = os.path.join(export_dir, 'synthetic_fra_dataset.hdf5')
                    with h5py.File(hdf5_path, 'w') as f:
                        for split in ['train', 'validation', 'test']:
                            grp = f.create_group(split)
                            grp.create_dataset('transformer_id', data=np.array([s['transformer_id'] for s in self.dataset[split]], dtype='S'))
                            grp.create_dataset('fault_type', data=np.array([s['fault_type'] for s in self.dataset[split]]))
                            grp.create_dataset('fault_severity', data=np.array([s['fault_severity'] for s in self.dataset[split]]))
                            grp.create_dataset('voltage_level', data=np.array([s['transformer_specs']['voltage_level'] for s in self.dataset[split]]))
                            grp.create_dataset('power_rating', data=np.array([s['transformer_specs']['power_rating'] for s in self.dataset[split]]))
                            grp.create_dataset('frequencies', data=np.array([s['frequencies'] for s in self.dataset[split]]))
                            grp.create_dataset('magnitude_db', data=np.array([s['magnitude_db'] for s in self.dataset[split]]))
                            grp.create_dataset('phase_degrees', data=np.array([s['phase_degrees'] for s in self.dataset[split]]))
                    export_paths['hdf5'] = hdf5_path
                    self.logger.info(f"Exported HDF5 format to {hdf5_path}")
                except Exception as e:
                    self.logger.error(f"Failed to export HDF5 format: {str(e)}")

        # Vendor format mock export (still just creates empty files)
        sample_data = self.dataset['train'][:10]  # Export first 10 samples
        for vendor in self.config['vendor_formats']:
            try:
                vendor_paths = []
                for i, sample in enumerate(sample_data):
                    vendor_path = os.path.join(export_dir, f'sample_{i}_{vendor}.csv')
                    # For now, just write a header and transformer_id
                    with open(vendor_path, 'w') as f:
                        f.write('transformer_id,fault_type,fault_severity,voltage_level,power_rating\n')
                        f.write(f"{sample['transformer_id']},{sample['fault_type']},{sample['fault_severity']},{sample['transformer_specs']['voltage_level']},{sample['transformer_specs']['power_rating']}\n")
                    vendor_paths.append(vendor_path)
                export_paths[f'{vendor}_samples'] = vendor_paths
                self.logger.info(f"Exported {len(vendor_paths)} samples in {vendor} format")
            except Exception as e:
                self.logger.error(f"Failed to export {vendor} format: {str(e)}")

        return export_paths
    
    def _mock_export(self, format_name: str) -> str:
        """Mock export function for demonstration"""
        return f"./exports/synthetic_fra_dataset.{format_name}"
    
    def generate_report(self) -> Dict:
        """
        Generate comprehensive pipeline report
        
        Returns:
            Pipeline execution report
        """
        report = {
            'pipeline_config': self.config,
            'dataset_statistics': self._calculate_dataset_statistics(),
            'validation_summary': self.validation_results,
            'export_summary': {
                'formats_exported': self.config['export_formats'],
                'vendor_formats': self.config['vendor_formats']
            },
            'recommendations': self._generate_recommendations()
        }
        
        return report
    
    def _calculate_dataset_statistics(self) -> Dict:
        """Calculate dataset statistics"""
        if not self.dataset:
            return {}
        
        all_samples = self.dataset['train'] + self.dataset['validation'] + self.dataset['test']
        
        # Fault type distribution
        fault_counts = {}
        severities = []
        
        for sample in all_samples:
            fault_type = sample['fault_type']
            fault_counts[fault_type] = fault_counts.get(fault_type, 0) + 1
            if sample['fault_severity'] > 0:
                severities.append(sample['fault_severity'])
        
        return {
            'total_samples': len(all_samples),
            'fault_distribution': fault_counts,
            'severity_statistics': {
                'mean': float(np.mean(severities)) if severities else 0,
                'std': float(np.std(severities)) if severities else 0,
                'range': [float(np.min(severities)), float(np.max(severities))] if severities else [0, 0]
            }
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on results"""
        recommendations = []
        
        if self.validation_results:
            pass_rate = self.validation_results.get('pass_rate', 0)
            
            if pass_rate < 0.8:
                recommendations.append(
                    f"Low physics validation pass rate ({pass_rate:.1%}). "
                    "Consider adjusting parameter ranges or fault simulation models."
                )
            
            if pass_rate > 0.95:
                recommendations.append(
                    "High validation pass rate indicates good data quality. "
                    "Dataset is ready for machine learning training."
                )
        
        recommendations.append(
            "Consider expanding dataset size if ML model performance is insufficient."
        )
        
        recommendations.append(
            "Validate ML model performance on real FRA data when available."
        )
        
        return recommendations
    
    def run_complete_pipeline(self) -> Dict:
        """
        Execute the complete data generation pipeline
        
        Returns:
            Complete pipeline results
        """
        self.logger.info("Starting complete FRA synthetic data generation pipeline...")
        
        try:
            # Setup
            self.setup_pipeline()
            
            # Generate dataset
            self.generate_dataset()
            
            # Validate physics
            self.validate_dataset_physics()
            
            # Export datasets
            export_paths = self.export_datasets()
            
            # Generate report
            report = self.generate_report()
            
            # Save report
            report_path = Path(self.config['output_dir']) / 'pipeline_report.json'
            with open(report_path, 'w') as f:
                json.dump(self._convert_to_json_compatible(report), f, indent=2)

            self.logger.info(f"Pipeline completed successfully. Report saved to {report_path}")

            return {
                'success': True,
                'report': report,
                'export_paths': export_paths,
                'report_path': str(report_path)
            }

        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }

    def _convert_to_json_compatible(self, obj):
        """
        Recursively convert numpy types and other non-JSON-serializable types to native Python types
        """
        import numpy as np
        if isinstance(obj, dict):
            return {k: self._convert_to_json_compatible(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_compatible(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        else:
            return obj

def main():
    """Main function for command-line execution"""
    parser = argparse.ArgumentParser(description='FRA Synthetic Data Generation Pipeline')
    
    parser.add_argument('--samples', type=int, default=10000, 
                       help='Number of samples to generate')
    parser.add_argument('--output-dir', type=str, default='./fra_synthetic_data',
                       help='Output directory')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--no-validation', action='store_true',
                       help='Skip physics validation')
    parser.add_argument('--formats', nargs='+', 
                       default=['hdf5', 'csv', 'numpy'],
                       help='Export formats')
    
    args = parser.parse_args()
    
    # Create configuration
    config = {
        'n_samples': args.samples,
        'output_dir': args.output_dir,
        'random_seed': args.seed,
        'validate_physics': not args.no_validation,
        'export_formats': args.formats
    }
    
    # Run pipeline
    pipeline = FRAPipeline(config)
    results = pipeline.run_complete_pipeline()
    
    if results['success']:
        print("\\nPipeline completed successfully!")
        print(f"Generated {config['n_samples']} samples")
        print(f"Results saved to {config['output_dir']}")
    else:
        print(f"\\nPipeline failed: {results['error']}")
        sys.exit(1)

if __name__ == "__main__":
    main()

# Example usage functions for testing
def run_small_demo():
    """Run a small demonstration of the pipeline"""
    config = {
        'n_samples': 1000,
        'output_dir': './demo_fra_data',
        'validate_physics': True,
        'export_formats': ['csv', 'numpy']
    }
    
    pipeline = FRAPipeline(config)
    results = pipeline.run_complete_pipeline()
    
    print("\\nDemo Pipeline Results:")
    print(f"Success: {results['success']}")
    if results['success']:
        report = results['report']
        print(f"Generated samples: {report['dataset_statistics']['total_samples']}")
        print(f"Validation pass rate: {report['validation_summary'].get('pass_rate', 0):.1%}")
    
    return results

def create_custom_dataset():
    """Example of creating a custom dataset with specific parameters"""
    config = {
        'n_samples': 5000,
        'validation_split': 0.15,
        'test_split': 0.15,
        'output_dir': './custom_fra_data',
        'quality_threshold': 0.85,
        'export_formats': ['hdf5', 'tensorflow'],
        'vendor_formats': ['omicron', 'megger', 'doble']
    }
    
    pipeline = FRAPipeline(config)
    results = pipeline.run_complete_pipeline()
    
    return results


print("Created main_pipeline.py - Main execution script")