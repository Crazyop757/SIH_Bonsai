# FRA Synthetic Data Generator

A comprehensive Python framework for generating physics-based synthetic Frequency Response Analysis (FRA) data for power transformer fault detection and machine learning applications.

## Overview

This project addresses the critical need for standardized, high-quality FRA training data by providing:

- **Physics-Based Modeling**: Accurate transformer equivalent circuit models
- **Fault Simulation**: Comprehensive fault injection for all major transformer fault types
- **Multi-Format Export**: Support for vendor formats (Omicron, Megger, Doble) and ML frameworks
- **Physics Validation**: Automated validation against causality, passivity, and stability constraints
- **Scalable Generation**: Efficient Monte Carlo simulation for large-scale datasets

## Key Features

### 🔬 Physics-Based Circuit Modeling
- Multi-section RLC ladder networks
- Frequency-dependent parameters
- Realistic transformer specifications (voltage levels, power ratings)
- Temperature and aging effects simulation

### ⚡ Comprehensive Fault Simulation
- **Axial Displacement**: Winding movement effects
- **Radial Deformation**: Turn-to-turn spacing changes  
- **Short Circuits**: Inter-turn and layer faults
- **Core Faults**: Magnetic circuit issues
- **Connection Problems**: Terminal and tap changer faults
- **Insulation Degradation**: Capacitance changes

### 📊 Advanced Data Export
- **Vendor Formats**: Omicron CSV, Megger XML, Doble proprietary
- **ML Frameworks**: NumPy, TensorFlow, PyTorch, scikit-learn
- **Standard Formats**: HDF5, CSV, JSON
- **Visualization Ready**: Sample exports for plotting and analysis

### ✅ Rigorous Validation
- Kramers-Kronig causality checks
- Passivity and stability validation
- Statistical parameter distribution verification
- Physics constraint enforcement

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/fra-synthetic-data-generator.git
cd fra-synthetic-data-generator

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Basic Usage

```python
from src.main_pipeline import FRAPipeline

# Generate 10,000 synthetic FRA samples
pipeline = FRAPipeline({
    'n_samples': 10000,
    'output_dir': './my_fra_data',
    'export_formats': ['hdf5', 'csv', 'numpy']
})

results = pipeline.run_complete_pipeline()
print(f"Generated {results['report']['dataset_statistics']['total_samples']} samples")
```

### Command Line Usage

```bash
# Generate dataset with default settings
python src/main_pipeline.py --samples 10000

# Advanced configuration
python src/main_pipeline.py \
    --samples 50000 \
    --output-dir ./large_dataset \
    --formats hdf5 numpy tensorflow \
    --seed 123
```

## Architecture

### Core Components

1. **TransformerCircuitModel**: Physics-based equivalent circuit modeling
2. **FaultSimulator**: Systematic fault injection and parameter modification  
3. **SyntheticFRADataGenerator**: Large-scale dataset generation with Monte Carlo
4. **PhysicsValidator**: Comprehensive validation against physical constraints
5. **FRADataExporter**: Multi-format data export utilities

### Workflow

1. **Model Generation**: Create diverse transformer population
2. **Fault Injection**: Apply realistic fault scenarios
3. **Response Calculation**: Compute frequency responses using circuit analysis
4. **Noise Addition**: Add realistic measurement noise
5. **Physics Validation**: Verify data quality and constraints
6. **Export**: Generate outputs in multiple formats

## Configuration

### Dataset Parameters

```python
config = {
    'n_samples': 10000,              # Total samples to generate
    'validation_split': 0.2,         # Validation set fraction
    'test_split': 0.1,              # Test set fraction
    'random_seed': 42,              # Reproducibility seed
    'quality_threshold': 0.8,       # Physics validation threshold
    'export_formats': ['hdf5', 'csv', 'numpy'],
    'vendor_formats': ['omicron', 'megger']
}
```

### Transformer Specifications

- **Voltage Levels**: 11kV - 800kV
- **Power Ratings**: 1MVA - 500MVA  
- **Winding Types**: Layer, disc, helical
- **Operating Conditions**: Temperature, aging, loading

## Validation and Quality Assurance

### Physics Constraints
- **Causality**: Kramers-Kronig relations validation
- **Passivity**: No negative resistance verification  
- **Stability**: Pole location analysis
- **Parameter Ranges**: Realistic component value bounds

### Statistical Validation
- Parameter distribution matching
- Fault severity distributions
- Frequency response characteristics
- Cross-correlation analysis

## Output Formats

### Machine Learning Ready
```python
# NumPy format
X_train, y_train = np.load('train_data.npz')['X'], np.load('train_data.npz')['y']

# TensorFlow format  
dataset = tf.data.TFRecordDataset('train_data.tfrecord')

# PyTorch format
data = torch.load('train_data.pt')
```

### Vendor Compatibility
```csv
# Omicron FRAnalyzer format
Frequency [Hz],Magnitude [dB],Phase [deg]
10.0,-45.2,89.1
20.0,-42.1,87.3
```

## Examples and Tutorials

### Jupyter Notebooks
- **Circuit Model Exploration**: Understanding the physics model
- **Fault Analysis**: Analyzing different fault signatures  
- **Dataset Generation**: Complete workflow demonstration
- **ML Integration**: Training fault detection models

### Script Examples
- Basic dataset generation
- Custom fault scenario creation
- Vendor format export
- Physics validation analysis

## Performance and Scalability

- **Generation Speed**: ~100 samples/second on standard hardware
- **Memory Efficiency**: Streaming generation for large datasets
- **Parallel Processing**: Multi-core support for validation
- **Storage Optimization**: HDF5 compression for large datasets

## Research Applications

### Academic Use Cases
- Transformer condition monitoring research
- FRA interpretation algorithm development
- Machine learning benchmark datasets
- Physics-informed AI training

### Industry Applications  
- AI-powered diagnostic system development
- Training data for commercial FRA software
- Fault detection algorithm validation
- Asset management system integration

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Check code quality
flake8 src/
black src/
```

## Citation

If you use this software in your research, please cite:

```bibtex
@software{fra_synthetic_generator,
  title={FRA Synthetic Data Generator: Physics-Based Training Data for Transformer Fault Detection},
  author={Your Name},
  year={2025},
  url={https://github.com/your-repo/fra-synthetic-data-generator}
}
```

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## Support and Documentation

- 📖 [User Guide](docs/user_guide.md)
- 🔧 [API Reference](docs/api_reference.md)  
- 🧬 [Physics Background](docs/physics_background.md)
- 💡 [Examples](docs/examples/)
- 🐛 [Issue Tracker](https://github.com/your-repo/fra-synthetic-data-generator/issues)

## Acknowledgments

- Power system research community for FRA methodology
- IEEE C57.149 standard for FRA guidelines
- Open source scientific Python ecosystem

---

**Disclaimer**: This synthetic data generator is for research and development purposes. Always validate results with real FRA measurements before making operational decisions.