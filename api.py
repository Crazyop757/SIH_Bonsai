"""
Production API for the Transformer Analysis Pipeline
This module provides a Flask API for the transformer analysis pipeline,
allowing it to be used in production environments.
"""

from flask import Flask, request, jsonify
import os
import json
import tempfile
import pandas as pd
from datetime import datetime
from pipeline import TransformerAnalysisPipeline

app = Flask(__name__)

# Configuration
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', 'pipeline_results')
FRA_MODEL_PATH = os.environ.get('FRA_MODEL_PATH', None)
SCADA_MODEL_PATH = os.environ.get('SCADA_MODEL_PATH', None)

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat()
    }), 200


@app.route('/process', methods=['POST'])
def process_data():
    """
    Process transformer data (FRA or SCADA) and return predictions.
    
    Expected JSON format for request:
    {
        "mode": "auto",  # Optional: "auto", "fra", or "scada"
        "data": [...]  # Array of data records
    }
    
    Or with file upload:
    - data_file: CSV or JSON file
    - mode: auto, fra, or scada (optional)
    """
    try:
        # Check if this is a file upload or JSON payload
        if 'data_file' in request.files:
            # Handle file upload
            data_file = request.files['data_file']
            mode = request.form.get('mode', 'auto')
            
            # Save uploaded file to temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix='.' + data_file.filename.split('.')[-1]) as temp:
                data_file.save(temp.name)
                temp_file_path = temp.name
            
            # Configure pipeline
            config = {
                'mode': mode,
                'output_dir': OUTPUT_DIR,
                'data_source': 'file',
                'input_file': temp_file_path,
                'fra_model_path': FRA_MODEL_PATH,
                'scada_model_path': SCADA_MODEL_PATH
            }
            
        else:
            # Handle JSON payload
            payload = request.json
            if not payload:
                return jsonify({
                    'error': 'No data provided. Send JSON data or upload a file.'
                }), 400
                
            data = payload.get('data', [])
            mode = payload.get('mode', 'auto')
            
            # Save JSON data to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as temp:
                json.dump(data, temp)
                temp_file_path = temp.name
                
            # Configure pipeline
            config = {
                'mode': mode,
                'output_dir': OUTPUT_DIR,
                'data_source': 'file',
                'input_file': temp_file_path,
                'fra_model_path': FRA_MODEL_PATH,
                'scada_model_path': SCADA_MODEL_PATH
            }
        
        # Create and run pipeline
        pipeline = TransformerAnalysisPipeline(config)
        result = pipeline.run_pipeline()
        
        # Check result
        if not result or not result.get('success'):
            return jsonify({
                'error': 'Pipeline execution failed',
                'details': result
            }), 500
            
        # Get results from saved file
        result_file = os.path.join(
            OUTPUT_DIR, 
            f"transformer_analysis_{result['model_used']}_{result['run_id']}.json"
        )
        
        if os.path.exists(result_file):
            with open(result_file, 'r') as f:
                predictions = json.load(f)
        else:
            predictions = []
            
        # Clean up temporary file
        try:
            os.unlink(temp_file_path)
        except:
            pass
            
        # Return results
        return jsonify({
            'success': True,
            'model_used': result['model_used'],
            'predictions': predictions,
            'count': result['results_count']
        })
        
    except Exception as e:
        app.logger.error(f"Error processing request: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500


if __name__ == '__main__':
    # For development only - use proper WSGI server in production
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)