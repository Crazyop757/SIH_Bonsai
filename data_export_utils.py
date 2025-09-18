# File 5: data_export_utils.py

"""
Data Export Utilities
Various export formats and data processing utilities for FRA datasets
"""

import numpy as np
import pandas as pd
import h5py
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import csv
import pickle
from datetime import datetime

class FRADataExporter:
    """
    Utility class for exporting FRA data in various formats
    """
    
    def __init__(self, output_dir: str = "./exports"):
        """Initialize exporter with output directory"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def export_to_vendor_format(self, fra_data: Dict, vendor: str, filename: str) -> str:
        """
        Export FRA data in vendor-specific formats
        
        Args:
            fra_data: FRA measurement data dictionary
            vendor: Vendor name ('omicron', 'megger', 'doble')
            filename: Output filename
            
        Returns:
            Path to exported file
        """
        if vendor.lower() == 'omicron':
            return self._export_omicron_format(fra_data, filename)
        elif vendor.lower() == 'megger':
            return self._export_megger_format(fra_data, filename)
        elif vendor.lower() == 'doble':
            return self._export_doble_format(fra_data, filename)
        else:
            raise ValueError(f"Unsupported vendor format: {vendor}")
    
    def _export_omicron_format(self, fra_data: Dict, filename: str) -> str:
        """Export in Omicron FRAnalyzer format (CSV-based)"""
        output_path = self.output_dir / f"{filename}_omicron.csv"
        
        # Omicron format: Frequency, Magnitude, Phase columns
        export_data = {
            'Frequency [Hz]': fra_data['frequencies'],
            'Magnitude [dB]': fra_data['magnitude_db'],
            'Phase [deg]': fra_data['phase_degrees']
        }
        
        df = pd.DataFrame(export_data)
        
        # Add header information
        header_lines = [
            f"# FRA Measurement Data - Generated {datetime.now()}",
            f"# Transformer ID: {fra_data.get('transformer_id', 'Unknown')}",
            f"# Fault Type: {fra_data.get('fault_type', 'Unknown')}",
            f"# Severity: {fra_data.get('fault_severity', 0):.1f}%",
            "#"
        ]
        
        with open(output_path, 'w', newline='') as f:
            # Write header
            for line in header_lines:
                f.write(line + '\\n')
            
            # Write data
            df.to_csv(f, index=False)
        
        return str(output_path)
    
    def _export_megger_format(self, fra_data: Dict, filename: str) -> str:
        """Export in Megger FRAX format (XML-based)"""
        output_path = self.output_dir / f"{filename}_megger.xml"
        
        # Create XML structure
        root = ET.Element("FRAMeasurement")
        
        # Header information
        header = ET.SubElement(root, "Header")
        ET.SubElement(header, "Timestamp").text = datetime.now().isoformat()
        ET.SubElement(header, "TransformerID").text = str(fra_data.get('transformer_id', 'Unknown'))
        ET.SubElement(header, "TestType").text = "Synthetic FRA"
        
        # Measurement data
        measurements = ET.SubElement(root, "Measurements")
        
        for i, freq in enumerate(fra_data['frequencies']):
            measurement = ET.SubElement(measurements, "DataPoint")
            ET.SubElement(measurement, "Frequency").text = f"{freq:.6e}"
            ET.SubElement(measurement, "Magnitude").text = f"{fra_data['magnitude_db'][i]:.6f}"
            ET.SubElement(measurement, "Phase").text = f"{fra_data['phase_degrees'][i]:.6f}"
        
        # Write XML file
        tree = ET.ElementTree(root)
        tree.write(output_path, encoding='utf-8', xml_declaration=True)
        
        return str(output_path)
    
    def _export_doble_format(self, fra_data: Dict, filename: str) -> str:
        """Export in Doble M4000 format (proprietary text format)"""
        output_path = self.output_dir / f"{filename}_doble.txt"
        
        with open(output_path, 'w') as f:
            # Doble header format
            f.write("DOBLE FRA DATA FILE\\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n")
            f.write(f"Transformer: {fra_data.get('transformer_id', 'Unknown')}\\n")
            f.write("Units: Hz, dB, degrees\\n")
            f.write("Data:\\n")
            
            # Write measurement data
            for i, freq in enumerate(fra_data['frequencies']):
                f.write(f"{freq:12.6e}\\t{fra_data['magnitude_db'][i]:10.6f}\\t"
                       f"{fra_data['phase_degrees'][i]:10.6f}\\n")
        
        return str(output_path)
    
    def export_for_machine_learning(self, dataset: Dict, format_type: str = 'numpy') -> str:
        """
        Export dataset optimized for machine learning workflows
        
        Args:
            dataset: Complete dataset with train/val/test splits
            format_type: 'numpy', 'tensorflow', 'pytorch', 'sklearn'
            
        Returns:
            Path to exported files
        """
        if format_type == 'numpy':
            return self._export_numpy_format(dataset)
        elif format_type == 'tensorflow':
            return self._export_tensorflow_format(dataset)
        elif format_type == 'pytorch':
            return self._export_pytorch_format(dataset)
        elif format_type == 'sklearn':
            return self._export_sklearn_format(dataset)
        else:
            raise ValueError(f"Unsupported ML format: {format_type}")
    
    def _export_numpy_format(self, dataset: Dict) -> str:
        """Export in NumPy format (.npz files)"""
        for split_name in ['train', 'validation', 'test']:
            if split_name not in dataset:
                continue
            
            split_data = dataset[split_name]
            if not split_data:
                continue
            
            # Convert to numpy arrays
            n_samples = len(split_data)
            n_freq = 250  # Downsampled length

            X_magnitude = np.zeros((n_samples, n_freq), dtype='float16')
            X_phase = np.zeros((n_samples, n_freq), dtype='float16')
            y_fault_type = np.zeros(n_samples, dtype=np.int32)
            y_severity = np.zeros(n_samples)
            frequencies = np.array(split_data[0]['frequencies'])
            
            for i, sample in enumerate(split_data):
                X_magnitude[i] = sample['magnitude_db'].astype('float16')
                X_phase[i] = sample['phase_degrees'].astype('float16')
                y_fault_type[i] = sample['fault_type']
                y_severity[i] = sample['fault_severity']


            
            # Save split
            output_path = self.output_dir / f"{split_name}_data.npz"
            np.savez(output_path,
                    X_magnitude=X_magnitude,
                    X_phase=X_phase,
                    y_fault_type=y_fault_type,
                    y_severity=y_severity,
                    frequencies=frequencies)
        
        return str(self.output_dir)
    
    def _export_tensorflow_format(self, dataset: Dict) -> str:
        """Export in TensorFlow-compatible format"""
        try:
            import tensorflow as tf
        except ImportError:
            raise ImportError("TensorFlow not installed")
        
        for split_name in ['train', 'validation', 'test']:
            if split_name not in dataset:
                continue
            
            split_data = dataset[split_name]
            if not split_data:
                continue
            
            # Convert to tensorflow format
            features = []
            labels = []
            
            for sample in split_data:
                # Combine magnitude and phase as features
                feature_vector = np.concatenate([
                    sample['magnitude_db'], 
                    sample['phase_degrees']
                ])
                features.append(feature_vector)
                labels.append(sample['fault_type'])
            
            # Create TensorFlow dataset
            features_tensor = tf.constant(np.array(features), dtype=tf.float32)
            labels_tensor = tf.constant(np.array(labels), dtype=tf.int32)
            
            tf_dataset = tf.data.Dataset.from_tensor_slices((features_tensor, labels_tensor))
            
            # Save as TFRecord
            output_path = self.output_dir / f"{split_name}_data.tfrecord"
            
            def serialize_example(features, label):
                feature = {
                    'features': tf.train.Feature(float_list=tf.train.FloatList(value=features.numpy().flatten())),
                    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label.numpy()]))
                }
                example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
                return example_proto.SerializeToString()
            
            with tf.io.TFRecordWriter(str(output_path)) as writer:
                for features, label in tf_dataset:
                    example = serialize_example(features, label)
                    writer.write(example)
        
        return str(self.output_dir)
    
    def _export_pytorch_format(self, dataset: Dict) -> str:
        """Export in PyTorch-compatible format"""
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch not installed")
        
        for split_name in ['train', 'validation', 'test']:
            if split_name not in dataset:
                continue
            
            split_data = dataset[split_name]
            if not split_data:
                continue
            
            # Convert to PyTorch tensors
            features = []
            labels = []
            
            for sample in split_data:
                feature_vector = np.concatenate([
                    sample['magnitude_db'],
                    sample['phase_degrees']
                ])
                features.append(feature_vector)
                labels.append(sample['fault_type'])
            
            features_tensor = torch.FloatTensor(np.array(features))
            labels_tensor = torch.LongTensor(np.array(labels))
            
            # Save tensors
            torch.save({
                'features': features_tensor,
                'labels': labels_tensor
            }, self.output_dir / f"{split_name}_data.pt")
        
        return str(self.output_dir)
    
    def _export_sklearn_format(self, dataset: Dict) -> str:
        """Export in scikit-learn compatible format"""
        # Combine all splits into train/test
        train_data = dataset.get('train', [])
        val_data = dataset.get('validation', [])
        test_data = dataset.get('test', [])
        
        # Combine train and validation for sklearn
        all_train_data = train_data + val_data
        
        def process_split(split_data, split_name):
            if not split_data:
                return
            
            features = []
            labels = []
            
            for sample in split_data:
                feature_vector = np.concatenate([
                    sample['magnitude_db'],
                    sample['phase_degrees']
                ])
                features.append(feature_vector)
                labels.append(sample['fault_type'])
            
            X = np.array(features)
            y = np.array(labels)
            
            # Save as pickle files (sklearn standard)
            with open(self.output_dir / f"X_{split_name}.pkl", 'wb') as f:
                pickle.dump(X, f)
            
            with open(self.output_dir / f"y_{split_name}.pkl", 'wb') as f:
                pickle.dump(y, f)
        
        process_split(all_train_data, 'train')
        process_split(test_data, 'test')
        
        return str(self.output_dir)
    
    def create_data_manifest(self, dataset: Dict, filename: str = "data_manifest.json") -> str:
        """Create a manifest file describing the dataset structure"""
        manifest = {
            'dataset_info': {
                'creation_date': datetime.now().isoformat(),
                'total_samples': sum(len(dataset.get(split, [])) for split in ['train', 'validation', 'test']),
                'splits': {
                    split: len(dataset.get(split, []))
                    for split in ['train', 'validation', 'test']
                }
            },
            'data_format': {
                'frequency_points': len(dataset['train'][0]['frequencies']) if dataset.get('train') else 0,
                'frequency_range': {
                    'min_hz': float(dataset['train'][0]['frequencies'][0]) if dataset.get('train') else 0,
                    'max_hz': float(dataset['train'][0]['frequencies'][-1]) if dataset.get('train') else 0
                },
                'features': ['magnitude_db', 'phase_degrees'],
                'labels': ['fault_type', 'fault_severity']
            },
            'fault_types': {
                0: 'HEALTHY',
                1: 'AXIAL_DISPLACEMENT',
                2: 'RADIAL_DEFORMATION', 
                3: 'TURN_TO_TURN_SHORT',
                4: 'CORE_FAULT',
                5: 'WINDING_CONNECTION',
                6: 'INSULATION_DEGRADATION'
            }
        }
        
        # Add statistics if available
        if 'metadata' in dataset:
            manifest['generation_config'] = dataset['metadata']
        
        manifest_path = self.output_dir / filename
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        return str(manifest_path)
    
    def export_visualization_data(self, sample_data: List[Dict], 
                                filename: str = "visualization_samples.json") -> str:
        """Export representative samples for visualization purposes"""
        
        # Select representative samples from each fault type
        viz_samples = {}
        
        for sample in sample_data:
            fault_type = sample['fault_type']
            if fault_type not in viz_samples:
                viz_samples[fault_type] = []
            
            if len(viz_samples[fault_type]) < 3:  # Keep max 3 samples per fault type
                viz_sample = {
                    'fault_type': fault_type,
                    'fault_severity': sample['fault_severity'],
                    'frequencies': sample['frequencies'].tolist(),
                    'magnitude_db': sample['magnitude_db'].tolist(),
                    'phase_degrees': sample['phase_degrees'].tolist(),
                    'transformer_id': sample.get('transformer_id', 'Unknown')
                }
                viz_samples[fault_type].append(viz_sample)
        
        # Export to JSON
        viz_path = self.output_dir / filename
        with open(viz_path, 'w') as f:
            json.dump(viz_samples, f, indent=2)
        
        return str(viz_path)

# Utility functions for data processing
def combine_magnitude_phase_features(magnitude_db: np.ndarray, 
                                    phase_degrees: np.ndarray) -> np.ndarray:
    """Combine magnitude and phase into feature vector"""
    return np.concatenate([magnitude_db, phase_degrees])

def normalize_frequency_features(frequencies: np.ndarray, 
                                magnitude_db: np.ndarray, 
                                phase_degrees: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Normalize features for machine learning"""
    # Log-scale frequencies for better ML performance
    log_freq = np.log10(frequencies)
    
    # Normalize magnitude to 0-1 range
    mag_normalized = (magnitude_db - np.min(magnitude_db)) / (np.max(magnitude_db) - np.min(magnitude_db))
    
    # Normalize phase to -1 to 1 range
    phase_normalized = phase_degrees / 180.0
    
    return mag_normalized, phase_normalized

def create_frequency_bands_features(frequencies: np.ndarray, 
                                  magnitude_db: np.ndarray) -> Dict[str, float]:
    """Extract statistical features from different frequency bands"""
    
    bands = {
        'low': (20, 2000),
        'medium': (2000, 20000),
        'high': (20000, 200000),
        'very_high': (200000, 2000000)
    }
    
    features = {}
    
    for band_name, (f_min, f_max) in bands.items():
        mask = (frequencies >= f_min) & (frequencies <= f_max)
        if np.any(mask):
            band_magnitude = magnitude_db[mask]
            features.update({
                f'{band_name}_mean': float(np.mean(band_magnitude)),
                f'{band_name}_std': float(np.std(band_magnitude)),
                f'{band_name}_max': float(np.max(band_magnitude)),
                f'{band_name}_min': float(np.min(band_magnitude))
            })
    
    return features


print("Created data_export_utils.py")