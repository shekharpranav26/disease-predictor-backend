from flask import Flask, request, jsonify
from flask_cors import CORS
from config import Config

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import joblib
import os
from ml_models import DiseasePredictor
from data_preprocessing import DataPreprocessor
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

app.config.from_object(Config)

# Initialize components
preprocessor = DataPreprocessor()
predictor = DiseasePredictor()

# Global variables to store model performance
model_performance = {}

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'Disease Detection API is running'
    }), 200

@app.route('/train-models', methods=['POST'])
def train_models():
    """Train all ML models and return performance metrics"""
    try:
        logger.info("Starting model training...")
        
        # Load and preprocess data
        X, y, symptom_columns = preprocessor.load_and_preprocess_data()
        
        # Train models
        models, performance = predictor.train_models(X, y)
        
        # Store performance globally
        global model_performance
        model_performance = performance
        
        logger.info("Model training completed successfully")
        
        return jsonify({
            'status': 'success',
            'message': 'Models trained successfully',
            'performance': performance,
            'total_symptoms': len(symptom_columns),
            'total_diseases': len(np.unique(y))
        }), 200
        
    except Exception as e:
        logger.error(f"Error training models: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Error training models: {str(e)}'
        }), 500

@app.route('/predict', methods=['POST'])
def predict_disease():
    """Predict disease based on symptoms"""
    global model_performance
    if not predictor._load_models() and not model_performance:
        return jsonify({'status':'error','message':'Models not trained yet; call /train-models first'}), 400

    try:
        data = request.get_json(force=True, silent=True)
        if not data:
            return jsonify({'status':'error','message':'No JSON payload received'}), 400

        # Accept both {"symptoms": ["a","b"]} OR {"a":1,"b":0,...}
        if isinstance(data, dict) and 'symptoms' in data and isinstance(data['symptoms'], list):
            symptoms = data['symptoms']
        elif isinstance(data, dict):
            # convert truthy keys to symptoms list
            # ignore model / metadata keys if present
            symptoms = [k for k, v in data.items() if k.lower() != 'symptoms' and str(v).lower() not in ('0','false','None','none','') and bool(v)]
        elif isinstance(data, list):
            symptoms = data
        else:
            return jsonify({'status':'error','message':'Unexpected payload format'}), 400

        if not isinstance(symptoms, list) or len(symptoms) == 0:
            return jsonify({'status':'error','message':'Symptoms must be a non-empty list'}), 400

        # normalize symptom strings to the same format used during training
        predictions = predictor.predict(symptoms)

        # Merge performance metrics into predictions (if available)
        predictions_with_metrics = {}
        for model_name, info in predictions.items():
            metrics = model_performance.get(model_name, {})  # accuracy/precision/recall/f1
        # round nicely and merge
            merged = dict(info)
        for k, v in metrics.items():
            if isinstance(v, float):
                merged[k] = round(v, 4)
            else:
                merged[k] = v
        predictions_with_metrics[model_name] = merged

        predictions = predictions_with_metrics  # replace the original


        if not predictions:
            return jsonify({'status':'error', 'message':'No matching symptoms or models unavailable'}), 400

        # pick the best model by confidence
        best_model = max(predictions, key=lambda m: predictions[m]['confidence'])
        predicted_disease = predictions[best_model]['disease']

        # fetch precautions for the DISEASE (not the model name)
        precautions = preprocessor.get_precautions(predicted_disease)

        return jsonify({
            'status': 'success',
            'predictions': predictions,       # still include all models
            'top_model': best_model,          # which model won
            'most_likely_disease': predicted_disease,
            'predicted_disease': predicted_disease,
            'confidence': predictions[best_model]['confidence'],
            'precautions': precautions,
            'input_symptoms': symptoms
        }), 200
    
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        return jsonify({'status':'error','message':f'Error making prediction: {str(e)}'}), 500

@app.route('/available-symptoms', methods=['GET'])
def get_available_symptoms():
    """Get list of all available symptoms"""
    try:
        symptoms = preprocessor.get_available_symptoms()
        
        return jsonify({
            'status': 'success',
            'symptoms': symptoms,
            'total_count': len(symptoms)
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting symptoms: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Error getting symptoms: {str(e)}'
        }), 500

@app.route('/model-performance', methods=['GET'])
def get_model_performance():
    """Get performance metrics of trained models"""
    try:
        if not model_performance:
            return jsonify({
                'status': 'error',
                'message': 'Models not trained yet. Please train models first.'
            }), 400
        
        return jsonify({
            'status': 'success',
            'performance': model_performance
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting model performance: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Error getting model performance: {str(e)}'
        }), 500

@app.route('/diseases', methods=['GET'])
def get_diseases():
    """Get list of all diseases that can be predicted"""
    try:
        diseases = preprocessor.get_available_diseases()
        
        return jsonify({
            'status': 'success',
            'diseases': diseases,
            'total_count': len(diseases)
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting diseases: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Error getting diseases: {str(e)}'
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'status': 'error',
        'message': 'Endpoint not found'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'status': 'error',
        'message': 'Internal server error'
    }), 500

if __name__ == '__main__':
    # Auto-train models on startup if data exists
    try:
        if os.path.exists('data/DiseaseAndSymptoms.csv'):
            logger.info("Auto-training models on startup...")
            X, y, symptom_columns = preprocessor.load_and_preprocess_data()
            models, performance = predictor.train_models(X, y)
            model_performance = performance
            logger.info("Auto-training completed successfully")
    except Exception as e:
        logger.warning(f"Auto-training failed: {str(e)}")

    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=5000)