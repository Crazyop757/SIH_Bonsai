
# Usage Examples for FRA Synthetic Data Generator

## 1. Basic Usage
```python
from src.main_pipeline import FRAPipeline

# Create pipeline with default configuration
pipeline = FRAPipeline()

# Generate dataset
results = pipeline.run_complete_pipeline()

print(f"Generated {results['report']['dataset_statistics']['total_samples']} samples")
```

## 2. Custom Configuration
```python
config = {
    'n_samples': 50000,
    'validation_split': 0.2,
    'test_split': 0.1,
    'output_dir': './my_fra_data',
    'export_formats': ['hdf5', 'tensorflow', 'pytorch'],
    'vendor_formats': ['omicron', 'megger', 'doble'],
    'quality_threshold': 0.9
}

pipeline = FRAPipeline(config)
results = pipeline.run_complete_pipeline()
```

## 3. Individual Component Usage
```python
from src.transformer_circuit_model import TransformerCircuitModel
from src.fault_simulation import FaultSimulator, FaultType

# Create transformer model
transformer = TransformerCircuitModel(
    n_sections=15, 
    voltage_level=132.0, 
    power_rating=50.0
)

# Create fault simulator
fault_sim = FaultSimulator()

# Generate healthy response
freq, mag, phase = transformer.calculate_frequency_response()

# Apply fault
faulty_params = fault_sim.apply_fault(
    transformer.baseline_params, 
    FaultType.AXIAL_DISPLACEMENT, 
    severity=35.0
)

# Calculate faulty response
freq_f, mag_f, phase_f = transformer.calculate_frequency_response(faulty_params)
```

## 4. Machine Learning Integration
```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import h5py

# Load dataset
with h5py.File('synthetic_fra_dataset.h5', 'r') as f:
    X_train = f['train/magnitudes'][:]
    y_train = f['train/fault_types'][:]
    X_test = f['test/magnitudes'][:]
    y_test = f['test/fault_types'][:]

# Train classifier
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)

# Evaluate
accuracy = clf.score(X_test, y_test)
print(f"Fault detection accuracy: {accuracy:.3f}")
```

## 5. Command Line Usage
```bash
# Basic generation
python src/main_pipeline.py --samples 10000

# Advanced options
python src/main_pipeline.py     --samples 50000     --output-dir ./large_dataset     --formats hdf5 numpy tensorflow     --seed 123

# Skip validation for faster generation
python src/main_pipeline.py --samples 100000 --no-validation
```
