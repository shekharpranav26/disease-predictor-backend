import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any
import re

logger = logging.getLogger(__name__)

def validate_symptoms(symptoms: List[str]) -> Dict[str, Any]:
    """Validate user input symptoms"""
    validation_result = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'cleaned_symptoms': []
    }
    
    if not symptoms:
        validation_result['is_valid'] = False
        validation_result['errors'].append('No symptoms provided')
        return validation_result
    
    if not isinstance(symptoms, list):
        validation_result['is_valid'] = False
        validation_result['errors'].append('Symptoms must be provided as a list')
        return validation_result
    
    # Clean and validate each symptom
    cleaned_symptoms = []
    for symptom in symptoms:
        if not isinstance(symptom, str):
            validation_result['warnings'].append(f'Non-string symptom ignored: {symptom}')
            continue
        
        # Clean the symptom
        cleaned_symptom = clean_symptom_text(symptom)
        
        if not cleaned_symptom:
            validation_result['warnings'].append(f'Empty symptom ignored: {symptom}')
            continue
        
        if len(cleaned_symptom) < 2:
            validation_result['warnings'].append(f'Very short symptom: {cleaned_symptom}')
        
        cleaned_symptoms.append(cleaned_symptom)
    
    if not cleaned_symptoms:
        validation_result['is_valid'] = False
        validation_result['errors'].append('No valid symptoms found after cleaning')
        return validation_result
    
    validation_result['cleaned_symptoms'] = cleaned_symptoms
    return validation_result

def clean_symptom_text(symptom: str) -> str:
    """Clean and normalize symptom text"""
    if not isinstance(symptom, str):
        return ""
    
    # Convert to lowercase and strip whitespace
    cleaned = symptom.lower().strip()
    
    # Remove extra whitespace and special characters
    cleaned = re.sub(r'[^\w\s]', ' ', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned)
    
    return cleaned.strip()

def calculate_symptom_similarity(symptom1: str, symptom2: str) -> float:
    """Calculate similarity between two symptoms using simple word matching"""
    words1 = set(symptom1.lower().split())
    words2 = set(symptom2.lower().split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union) if union else 0.0

def get_top_symptoms_by_importance(feature_importance: Dict[str, float], top_n: int = 10) -> List[Dict[str, Any]]:
    """Get top N symptoms by feature importance"""
    sorted_symptoms = sorted(
        feature_importance.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    return [
        {
            'symptom': symptom,
            'importance': importance,
            'rank': i + 1
        }
        for i, (symptom, importance) in enumerate(sorted_symptoms[:top_n])
    ]

def format_disease_name(disease: str) -> str:
    """Format disease name for better readability"""
    if not disease:
        return "Unknown"
    
    # Handle special cases
    disease = disease.strip()
    
    # Capitalize first letter of each word
    formatted = ' '.join(word.capitalize() for word in disease.split())
    
    # Handle specific formatting cases
    replacements = {
        'Gerd': 'GERD',
        'Aids': 'AIDS',
        'Hiv': 'HIV',
        'Copd': 'COPD'
    }
    
    for old, new in replacements.items():
        formatted = formatted.replace(old, new)
    
    return formatted

def calculate_confidence_level(confidence: float) -> str:
    """Convert numerical confidence to descriptive level"""
    if confidence >= 0.9:
        return "Very High"
    elif confidence >= 0.7:
        return "High"
    elif confidence >= 0.5:
        return "Medium"
    elif confidence >= 0.3:
        return "Low"
    else:
        return "Very Low"

def generate_health_disclaimer() -> str:
    """Generate health disclaimer text"""
    return (
        "IMPORTANT DISCLAIMER: This prediction is for informational purposes only "
        "and should not be considered as professional medical advice, diagnosis, or treatment. "
        "Always consult with a qualified healthcare provider for proper medical evaluation "
        "and treatment of any health concerns."
    )

def log_prediction_request(symptoms: List[str], predictions: Dict[str, Any], user_ip: str = None):
    """Log prediction request for monitoring and analytics"""
    try:
        log_data = {
            'symptoms_count': len(symptoms),
            'symptoms': symptoms[:5],  # Log only first 5 symptoms for privacy
            'predictions_count': len(predictions),
            'user_ip': user_ip[:8] + '***' if user_ip else 'unknown'  # Partial IP for privacy
        }
        
        logger.info(f"Prediction request: {log_data}")
        
    except Exception as e:
        logger.error(f"Error logging prediction request: {str(e)}")

def validate_model_performance(performance: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    """Validate model performance metrics"""
    validation_result = {
        'is_valid': True,
        'warnings': [],
        'best_model': None,
        'worst_model': None
    }
    
    if not performance:
        validation_result['is_valid'] = False
        return validation_result
    
    # Find best and worst performing models based on F1 score
    f1_scores = {}
    for model_name, metrics in performance.items():
        if 'f1_score' in metrics:
            f1_scores[model_name] = metrics['f1_score']
    
    if f1_scores:
        validation_result['best_model'] = max(f1_scores.items(), key=lambda x: x[1])
        validation_result['worst_model'] = min(f1_scores.items(), key=lambda x: x[1])
        
        # Check for poor performance
        if validation_result['best_model'][1] < 0.5:
            validation_result['warnings'].append('Best model has low F1 score (< 0.5)')
        
        if validation_result['worst_model'][1] < 0.3:
            validation_result['warnings'].append('Some models have very low F1 score (< 0.3)')
    
    return validation_result

def create_prediction_summary(predictions: Dict[str, Any]) -> Dict[str, Any]:
    """Create a summary of predictions from all models"""
    if not predictions:
        return {}
    
    # Count disease predictions
    disease_counts = {}
    confidence_sum = {}
    
    for model_name, pred in predictions.items():
        disease = pred.get('disease', 'Unknown')
        confidence = pred.get('confidence', 0.0)
        
        if disease not in disease_counts:
            disease_counts[disease] = 0
            confidence_sum[disease] = 0.0
        
        disease_counts[disease] += 1
        confidence_sum[disease] += confidence
    
    # Calculate consensus
    total_models = len(predictions)
    consensus_disease = max(disease_counts.items(), key=lambda x: x[1])
    
    summary = {
        'consensus_disease': consensus_disease[0],
        'consensus_votes': consensus_disease[1],
        'consensus_percentage': round((consensus_disease[1] / total_models) * 100, 1),
        'average_confidence': round(confidence_sum[consensus_disease[0]] / consensus_disease[1], 4),
        'total_models': total_models,
        'unique_predictions': len(disease_counts),
        'all_predictions': dict(disease_counts)
    }
    
    return summary