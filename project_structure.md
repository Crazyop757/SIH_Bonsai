
# FRA Synthetic Data Generation Project Structure

fra_synthetic_data_generator/
│
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
├── setup.py                          # Package installation script
│
├── src/                              # Source code directory
│   ├── __init__.py
│   ├── transformer_circuit_model.py  # Physics-based circuit modeling
│   ├── fault_simulation.py          # Fault injection and simulation
│   ├── synthetic_data_generator.py   # Main data generation class
│   ├── physics_validation.py        # Physics constraint validation
│   ├── data_export_utils.py         # Multi-format data export utilities
│   └── main_pipeline.py             # Complete pipeline execution
│
├── config/                           # Configuration files
│   ├── default_config.yaml          # Default pipeline configuration
│   ├── transformer_specs.json       # Transformer specification templates
│   └── fault_parameters.json        # Fault simulation parameters
│
├── data/                            # Generated data storage
│   ├── raw/                         # Raw generated datasets
│   ├── processed/                   # Processed/validated datasets
│   ├── exports/                     # Vendor format exports
│   └── samples/                     # Sample data for visualization
│
├── tests/                           # Unit tests
│   ├── __init__.py
│   ├── test_circuit_model.py
│   ├── test_fault_simulation.py
│   ├── test_data_generation.py
│   ├── test_physics_validation.py
│   └── test_export_utils.py
│
├── notebooks/                       # Jupyter notebooks for analysis
│   ├── 01_circuit_model_exploration.ipynb
│   ├── 02_fault_simulation_analysis.ipynb
│   ├── 03_data_generation_demo.ipynb
│   ├── 04_physics_validation_study.ipynb
│   └── 05_dataset_analysis.ipynb
│
├── scripts/                         # Utility scripts
│   ├── generate_dataset.py          # Simple dataset generation script
│   ├── validate_data.py            # Standalone validation script
│   ├── export_vendor_formats.py    # Vendor format export script
│   └── visualize_samples.py        # Data visualization script
│
├── docs/                            # Documentation
│   ├── user_guide.md               # User guide and tutorials
│   ├── api_reference.md            # API documentation
│   ├── physics_background.md       # Physics theory background
│   └── examples/                    # Usage examples
│       ├── basic_usage.py
│       ├── custom_configuration.py
│       └── ml_integration.py
│
└── outputs/                         # Default output directory
    ├── datasets/                    # Generated datasets
    ├── reports/                     # Validation reports
    ├── plots/                       # Generated plots
    └── logs/                        # Pipeline logs
