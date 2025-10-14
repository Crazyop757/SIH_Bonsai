#!/usr/bin/env python
# SCADA Transformer Monitoring API
# REST API for SCADA fault prediction

from flask import Flask, request, jsonify
import os
import sys
import json
from datetime import datetime
import pandas as pd

# Import the pipeline
from scada_pipeline import SCADATransformerPipeline

app = Flask(__name__)

# Initialize pipeline
pipeline = SCADATransformerPipeline(
    model_path='best_scada_model.h5',
    scaler_path='scada_scaler.pkl',
    encoder_path='scada_fault_encoder.pkl'
)

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint to make predictions on SCADA data
    
    Accepts:
    - JSON data with SCADA measurements
    - CSV file upload
    
    Returns:
    - Prediction results with recommendations
    """
    try:
        # Check if the request has a file part
        if 'file' in request.files:
            file = request.files['file']
            # Save temporarily
            temp_path = f"temp_upload_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv"
            file.save(temp_path)
            
            # Process with pipeline
            results = pipeline.run_pipeline(
                data=temp_path,
                format_type='csv',
                use_tta=True
            )
            
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
        # If JSON data
        elif request.json:
            # Process JSON data
            if isinstance(request.json, list):
                # List of measurements
                df = pd.DataFrame(request.json)
                results = pipeline.run_pipeline(
                    data=df,
                    format_type='dataframe',
                    use_tta=True
                )
            else:
                # Single measurement
                results = pipeline.run_pipeline(
                    data=request.json,
                    format_type='dict',
                    use_tta=True
                )
        else:
            return jsonify({"error": "No data provided. Send either a CSV file or JSON data"}), 400
            
        return jsonify(results)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """
    Endpoint for batch prediction on multiple SCADA measurements
    
    Accepts:
    - JSON array of SCADA measurements
    - CSV file upload with multiple rows
    
    Returns:
    - Batch prediction results with recommendations
    """
    try:
        # Process similar to predict endpoint but optimized for batch
        if 'file' in request.files:
            file = request.files['file']
            temp_path = f"temp_batch_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv"
            file.save(temp_path)
            
            results = pipeline.run_pipeline(
                data=temp_path,
                format_type='csv',
                use_tta=True
            )
            
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
        elif request.json and isinstance(request.json, list):
            df = pd.DataFrame(request.json)
            results = pipeline.run_pipeline(
                data=df,
                format_type='dataframe',
                use_tta=True
            )
        else:
            return jsonify({"error": "Invalid data format. Send either a CSV file or a JSON array"}), 400
            
        return jsonify(results)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)