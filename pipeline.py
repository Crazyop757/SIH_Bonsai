"""
SIH_Bonsai Transformer Analysis Pipeline

This module implements a complete data processing and prediction pipeline
for transformer fault analysis using either FRA or SCADA data.

The pipeline handles:
1. Data loading/generation
2. Feature extraction
3. Model selection
4. Prediction
5. Result formatting
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
from pathlib import Path

# Import both FRA and SCADA components
try:
    # FRA components (from root directory)
    from data import EnhancedFRAGenerator as RootFRAGenerator
    import model as fra_model
except ImportError:
    print("Warning: Root FRA components not found")

try:
    # SCADA components
    from scada.data_pipeline import SCADARealisticGenerator
    from scada.features import SCADAFeatureExtractor
    from scada.enhanced_fra_generator import EnhancedFRAGenerator as ScadaFRAGenerator
    import scada.model as scada_model
except ImportError:
    print("Warning: SCADA components not found")


class TransformerAnalysisPipeline:
    """
    Complete pipeline for transformer fault analysis with support for
    both FRA-direct and SCADA-based approaches.
    """
    
    def __init__(self, config=None):
        """
        Initialize the pipeline with configuration parameters.
        
        Args:
            config (dict): Configuration parameters for the pipeline
        """
        self.config = config or {}
        self.pipeline_mode = self.config.get('mode', 'auto')
        self.output_dir = self.config.get('output_dir', 'pipeline_results')
        self.data_source = self.config.get('data_source', 'generate')
        self.n_samples = self.config.get('n_samples', 1000)
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize models
        self.fra_model = None
        self.scada_model = None
        
        # Initialize data components
        self.fra_generator = None
        self.scada_generator = None
        
        # Setup logging
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(self.output_dir, f"pipeline_log_{self.run_id}.txt")
        
        # Load feature extractors
        self.scada_feature_extractor = SCADAFeatureExtractor()

    def log(self, message):
        """Log a message to the log file and print to console."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] {message}"
        print(log_msg)
        with open(self.log_file, 'a') as f:
            f.write(log_msg + '\n')

    def initialize_components(self):
        """Initialize all required components for the pipeline."""
        self.log("Initializing pipeline components...")
        
        # Initialize generators
        if self.pipeline_mode in ['fra', 'auto']:
            self.fra_generator = RootFRAGenerator()
        
        if self.pipeline_mode in ['scada', 'auto']:
            try:
                # Initialize SCADA with public data if available
                public_data = None
                self.scada_generator = SCADARealisticGenerator(real_baseline=public_data)
            except Exception as e:
                self.log(f"Error initializing SCADA generator: {e}")

        # Load or initialize models
        self.load_models()
        
        self.log("Pipeline components initialized successfully")
        return True

    def load_models(self):
        """Load pre-trained models or initialize new ones."""
        self.log("Loading models...")
        
        # FRA model (from root model.py)
        if self.pipeline_mode in ['fra', 'auto']:
            try:
                # Check if we have a pre-trained FRA model
                fra_model_path = self.config.get('fra_model_path')
                if fra_model_path and os.path.exists(fra_model_path):
                    self.log(f"Loading pre-trained FRA model from {fra_model_path}")
                    # For TF models
                    if fra_model_path.endswith('.h5'):
                        self.fra_model = tf.keras.models.load_model(fra_model_path)
                else:
                    self.log("No pre-trained FRA model found. Will train or use on-the-fly.")
            except Exception as e:
                self.log(f"Error loading FRA model: {e}")
        
        # SCADA model (from scada/model.py)
        if self.pipeline_mode in ['scada', 'auto']:
            try:
                # Check if we have a pre-trained SCADA model
                scada_model_path = self.config.get('scada_model_path')
                if scada_model_path and os.path.exists(scada_model_path):
                    self.log(f"Loading pre-trained SCADA model from {scada_model_path}")
                    # For TF models
                    if scada_model_path.endswith('.h5'):
                        self.scada_model = tf.keras.models.load_model(
                            scada_model_path,
                            custom_objects={'FocalLoss': scada_model.FocalLoss}
                        )
                else:
                    self.log("No pre-trained SCADA model found. Will train or use on-the-fly.")
            except Exception as e:
                self.log(f"Error loading SCADA model: {e}")

    def generate_or_load_data(self):
        """Generate synthetic data or load from files based on config."""
        self.log(f"Data source: {self.data_source}")
        
        fra_data = None
        scada_data = None
        
        if self.data_source == 'generate':
            # Generate synthetic data
            if self.pipeline_mode in ['fra', 'auto'] and self.fra_generator:
                self.log(f"Generating {self.n_samples} synthetic FRA samples...")
                fra_data = self.fra_generator.generate_dataset(n_samples=self.n_samples)
                self.log(f"Generated {len(fra_data)} FRA samples")
            
            if self.pipeline_mode in ['scada', 'auto'] and self.scada_generator:
                self.log(f"Generating {self.n_samples} synthetic SCADA samples...")
                scada_data = self.scada_generator.generate(n_samples=self.n_samples)
                self.log(f"Generated {len(scada_data)} SCADA samples")
        
        elif self.data_source == 'file':
            # Load data from files
            fra_file = self.config.get('fra_data_file')
            scada_file = self.config.get('scada_data_file')
            input_file = self.config.get('input_file')  # Generic input file for production
            
            # If a generic input file is provided (e.g., from frontend)
            if input_file and os.path.exists(input_file):
                self.log(f"Production mode: Processing input file {input_file}")
                file_type = self._detect_file_type(input_file)
                
                if file_type == 'fra':
                    self.log(f"Detected FRA data file format")
                    fra_file = input_file
                elif file_type == 'scada':
                    self.log(f"Detected SCADA data file format")
                    scada_file = input_file
                else:
                    self.log(f"Could not determine file type for {input_file}. Attempting to process as both types.")
                    # Try loading as both types in auto mode
                    if self.pipeline_mode == 'auto':
                        try:
                            fra_data = self._try_load_fra_data(input_file)
                        except:
                            self.log("Failed to load as FRA data")
                        
                        try:
                            scada_data = self._try_load_scada_data(input_file)
                        except:
                            self.log("Failed to load as SCADA data")
            
            # Process specific FRA file if provided or detected
            if fra_file and os.path.exists(fra_file) and fra_data is None:
                self.log(f"Loading FRA data from {fra_file}")
                fra_data = self._try_load_fra_data(fra_file)
            
            # Process specific SCADA file if provided or detected
            if scada_file and os.path.exists(scada_file) and scada_data is None:
                self.log(f"Loading SCADA data from {scada_file}")
                scada_data = self._try_load_scada_data(scada_file)
        
        return fra_data, scada_data
        
    def _detect_file_type(self, file_path):
        """Detect if a file contains FRA or SCADA data based on its content and format."""
        try:
            # Check file extension first
            if file_path.endswith('.npz'):
                # NPZ files are typically used for FRA data with raw traces
                return 'fra'
            
            # For CSV files, we need to look at the content
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path, nrows=5)  # Read just a few rows
                
                # Check for typical FRA columns
                fra_indicators = ['frequency', 'magnitude_db', 'phase_deg', 'fault_type']
                fra_match = sum(1 for col in fra_indicators if col in df.columns)
                
                # Check for typical SCADA columns
                scada_indicators = ['timestamp', 'transformer_id', 'voltage_l1', 'current_l1', 
                                   'oil_temp_top', 'thd_voltage', 'thd_current']
                scada_match = sum(1 for col in scada_indicators if col in df.columns)
                
                if fra_match > scada_match:
                    return 'fra'
                elif scada_match > 0:
                    return 'scada'
            
            # For JSON files
            if file_path.endswith('.json'):
                with open(file_path, 'r') as f:
                    import json
                    data = json.load(f)
                    # Check first record for typical fields
                    if isinstance(data, list) and len(data) > 0:
                        first_record = data[0]
                        if any(key in first_record for key in ['frequency', 'magnitude_db', 'phase_deg']):
                            return 'fra'
                        if any(key in first_record for key in ['timestamp', 'transformer_id', 'voltage_l1']):
                            return 'scada'
            
            # Could not determine type
            return 'unknown'
        except Exception as e:
            self.log(f"Error detecting file type: {e}")
            return 'unknown'
    
    def _try_load_fra_data(self, file_path):
        """Attempt to load a file as FRA data."""
        try:
            if file_path.endswith('.csv'):
                fra_data = pd.read_csv(file_path)
                self.log(f"Successfully loaded FRA data from CSV: {len(fra_data)} rows")
                return fra_data
            elif file_path.endswith('.npz'):
                fra_data = np.load(file_path, allow_pickle=True)
                self.log(f"Successfully loaded FRA data from NPZ file")
                return fra_data
            elif file_path.endswith('.json'):
                with open(file_path, 'r') as f:
                    import json
                    fra_data = pd.DataFrame(json.load(f))
                    self.log(f"Successfully loaded FRA data from JSON: {len(fra_data)} rows")
                    return fra_data
            else:
                self.log(f"Unsupported file format for FRA data: {file_path}")
                return None
        except Exception as e:
            self.log(f"Error loading FRA data from {file_path}: {e}")
            return None
    
    def _try_load_scada_data(self, file_path):
        """Attempt to load a file as SCADA data."""
        try:
            if file_path.endswith('.csv'):
                scada_data = pd.read_csv(file_path)
                self.log(f"Successfully loaded SCADA data from CSV: {len(scada_data)} rows")
                return scada_data
            elif file_path.endswith('.json'):
                with open(file_path, 'r') as f:
                    import json
                    scada_data = pd.DataFrame(json.load(f))
                    self.log(f"Successfully loaded SCADA data from JSON: {len(scada_data)} rows")
                    return scada_data
            else:
                self.log(f"Unsupported file format for SCADA data: {file_path}")
                return None
        except Exception as e:
            self.log(f"Error loading SCADA data from {file_path}: {e}")
            return None

    def extract_features(self, fra_data=None, scada_data=None):
        """Extract features from data sources."""
        fra_features = None
        scada_features = None
        
        # Extract FRA features if needed
        if fra_data is not None and self.pipeline_mode in ['fra', 'auto']:
            self.log("Extracting FRA features...")
            # If fra_data is from generate_dataset, features are already included
            # This is for raw FRA data that needs feature extraction
            if isinstance(fra_data, list) and 'features' in fra_data[0]:
                fra_features = [sample['features'] for sample in fra_data]
            else:
                self.log("No feature extraction needed for provided FRA data")
                fra_features = fra_data
        
        # Extract SCADA features
        if scada_data is not None and self.pipeline_mode in ['scada', 'auto']:
            self.log("Extracting SCADA features...")
            if isinstance(scada_data, pd.DataFrame):
                scada_features = []
                for _, row in scada_data.iterrows():
                    features = self.scada_feature_extractor.extract(row)
                    scada_features.append(features)
                self.log(f"Extracted features from {len(scada_features)} SCADA samples")
        
        return fra_features, scada_features

    def run_fra_model(self, fra_features):
        """Run the FRA-based model for transformer fault prediction."""
        if not self.fra_model:
            self.log("FRA model not loaded. Attempting to create...")
            # Here you would implement on-the-fly model creation
            # using code from the root model.py file
            self.log("On-the-fly FRA model creation not implemented")
            return None
        
        self.log("Running predictions with FRA model...")
        # Format data for model input
        # This would depend on the specific format expected by your model
        
        predictions = self.fra_model.predict(fra_features)
        self.log("FRA model predictions complete")
        
        return predictions

    def run_scada_model(self, scada_features):
        """Run the SCADA-based model for transformer fault prediction."""
        if not self.scada_model:
            self.log("SCADA model not loaded. Attempting to create...")
            # Here you would implement on-the-fly model creation
            # using code from scada/model.py
            self.log("On-the-fly SCADA model creation not implemented")
            return None
        
        self.log("Running predictions with SCADA model...")
        # Format data for model input
        # This would depend on the specific format expected by your model
        
        predictions = self.scada_model.predict(scada_features)
        self.log("SCADA model predictions complete")
        
        return predictions

    def select_model_and_predict(self, fra_features=None, scada_features=None):
        """Select which model to use and run predictions."""
        results = {
            'fra_predictions': None,
            'scada_predictions': None,
            'selected_predictions': None,
            'model_used': None
        }
        
        # Run available models based on pipeline mode and available data
        if self.pipeline_mode in ['fra', 'auto'] and fra_features is not None:
            results['fra_predictions'] = self.run_fra_model(fra_features)
        
        if self.pipeline_mode in ['scada', 'auto'] and scada_features is not None:
            results['scada_predictions'] = self.run_scada_model(scada_features)
        
        # Auto-select the best model based on confidence or predefined rules
        if self.pipeline_mode == 'auto':
            # Implement model selection logic here
            # For now, prefer SCADA if available, otherwise use FRA
            if results['scada_predictions'] is not None:
                results['selected_predictions'] = results['scada_predictions']
                results['model_used'] = 'scada'
                self.log("Auto-selected SCADA model")
            elif results['fra_predictions'] is not None:
                results['selected_predictions'] = results['fra_predictions']
                results['model_used'] = 'fra'
                self.log("Auto-selected FRA model")
            else:
                self.log("No predictions available from either model")
        elif self.pipeline_mode == 'fra':
            results['selected_predictions'] = results['fra_predictions']
            results['model_used'] = 'fra'
        elif self.pipeline_mode == 'scada':
            results['selected_predictions'] = results['scada_predictions']
            results['model_used'] = 'scada'
        
        return results

    def format_results(self, predictions, data_source=None):
        """Format predictions into a useful structure."""
        formatted_results = []
        
        if predictions is None:
            self.log("No predictions to format")
            return formatted_results
        
        # This formatting would depend on your specific models
        # Here's a generic example assuming standard outputs
        
        if isinstance(predictions, list) and len(predictions) == 5:
            # Assuming 5 outputs: anomaly, fault_type, severity, criticality, fault_probability
            anomaly_preds, fault_preds, severity_preds, criticality_preds, fault_prob_preds = predictions
            
            # Assuming batched outputs
            for i in range(len(anomaly_preds)):
                result = {
                    'anomaly': float(anomaly_preds[i][0]),
                    'fault_type_probs': fault_preds[i].tolist(),
                    'fault_type': int(np.argmax(fault_preds[i])),
                    'severity': float(severity_preds[i][0]),
                    'criticality': float(criticality_preds[i][0]),
                    'fault_probability': float(fault_prob_preds[i][0])
                }
                
                # Add source data if available
                if data_source is not None:
                    if isinstance(data_source, pd.DataFrame) and i < len(data_source):
                        # Add key data fields from source
                        for key in ['id', 'transformer_type', 'vendor']:
                            if key in data_source.columns:
                                result[key] = data_source.iloc[i][key]
                
                formatted_results.append(result)
        else:
            self.log("Unexpected prediction format. Custom formatting required.")
        
        return formatted_results

    def save_results(self, results, prefix='pipeline_results'):
        """Save the results to files."""
        if not results:
            self.log("No results to save")
            return
        
        # Save as CSV
        result_file = os.path.join(self.output_dir, f"{prefix}_{self.run_id}.csv")
        pd.DataFrame(results).to_csv(result_file, index=False)
        self.log(f"Results saved to {result_file}")
        
        # Also save as JSON for easy parsing
        json_file = os.path.join(self.output_dir, f"{prefix}_{self.run_id}.json")
        pd.DataFrame(results).to_json(json_file, orient='records')
        self.log(f"Results also saved to {json_file}")
        
        return result_file

    def run_pipeline(self):
        """Run the complete pipeline end-to-end."""
        self.log(f"Starting pipeline with mode: {self.pipeline_mode}")
        
        # Step 1: Initialize components
        if not self.initialize_components():
            self.log("Failed to initialize pipeline components")
            return False
        
        # Step 2: Generate or load data
        fra_data, scada_data = self.generate_or_load_data()
        
        # Step 3: Extract features
        fra_features, scada_features = self.extract_features(fra_data, scada_data)
        
        # Step 4: Select model and run predictions
        prediction_results = self.select_model_and_predict(fra_features, scada_features)
        
        # Step 5: Format the results
        data_source = scada_data if prediction_results['model_used'] == 'scada' else fra_data
        formatted_results = self.format_results(prediction_results['selected_predictions'], data_source)
        
        # Step 6: Save the results
        if formatted_results:
            self.save_results(formatted_results, f"transformer_analysis_{prediction_results['model_used']}")
        
        self.log("Pipeline execution completed")
        return {
            'success': True,
            'model_used': prediction_results['model_used'],
            'results_count': len(formatted_results) if formatted_results else 0,
            'run_id': self.run_id
        }


def main():
    """Main function to run the pipeline with command line arguments."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run the transformer analysis pipeline')
    parser.add_argument('--mode', choices=['auto', 'fra', 'scada'], default='auto',
                      help='Pipeline mode: auto (use both models), fra (use FRA model), or scada (use SCADA model)')
    parser.add_argument('--samples', type=int, default=1000, 
                      help='Number of samples to generate when using synthetic data')
    parser.add_argument('--output-dir', default='pipeline_results',
                      help='Directory to save the results')
    parser.add_argument('--data-source', choices=['generate', 'file'], default='generate',
                      help='Source of data: generate (synthetic) or file (load from files)')
    parser.add_argument('--input-file', help='Path to input file (auto-detection of type for production use)')
    parser.add_argument('--fra-file', help='Path to FRA data file when using --data-source=file')
    parser.add_argument('--scada-file', help='Path to SCADA data file when using --data-source=file')
    parser.add_argument('--fra-model', help='Path to pre-trained FRA model')
    parser.add_argument('--scada-model', help='Path to pre-trained SCADA model')
    
    args = parser.parse_args()
    
    # Configure the pipeline based on arguments
    config = {
        'mode': args.mode,
        'n_samples': args.samples,
        'output_dir': args.output_dir,
        'data_source': args.data_source,
        'input_file': args.input_file,
        'fra_data_file': args.fra_file,
        'scada_data_file': args.scada_file,
        'fra_model_path': args.fra_model,
        'scada_model_path': args.scada_model
    }
    
    # Create and run the pipeline
    pipeline = TransformerAnalysisPipeline(config)
    results = pipeline.run_pipeline()
    
    print(f"\nPipeline completed {'successfully' if results['success'] else 'with errors'}")
    print(f"Model used: {results['model_used']}")
    print(f"Results count: {results['results_count']}")
    print(f"Output directory: {args.output_dir}")
    print(f"Run ID: {results['run_id']}")


if __name__ == "__main__":
    main()