# Example Commands for Running the Pipeline

This document provides example commands to run the pipeline with various configurations.

## Basic Usage

Run the pipeline with default settings:

```powershell
python pipeline.py
```

This will:

- Use auto mode (trying both FRA and SCADA models)
- Generate 1000 synthetic samples
- Save results in ./pipeline_results/

## Specific Model Selection

### Using only the FRA model:

```powershell
python pipeline.py --mode fra --samples 500
```

### Using only the SCADA model:

```powershell
python pipeline.py --mode scada --samples 500
```

## Using Pre-trained Models

If you have pre-trained models:

```powershell
python pipeline.py --fra-model "path/to/fra_model.h5" --scada-model "path/to/scada_model.h5"
```

## Loading Data from Files

Use existing data instead of generating synthetic data:

```powershell
python pipeline.py --data-source file --fra-file "data/fra_data.csv" --scada-file "data/scada_data.csv"
```

## Custom Output Location

Save results to a custom directory:

```powershell
python pipeline.py --output-dir "my_results"
```

## Full Example with Multiple Options

```powershell
python pipeline.py --mode auto --samples 2000 --output-dir "transformer_analysis_results" --scada-model "models/scada_model.h5"
```

This will:

- Use both models (auto mode) with preference for SCADA
- Generate 2000 synthetic samples
- Save results in ./transformer_analysis_results/
- Use a pre-trained SCADA model
