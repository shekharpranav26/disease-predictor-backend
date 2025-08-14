import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib
import os
import logging
from data_preprocessing import DataPreprocessor

logger = logging.getLogger(__name__)

class DiseasePredictor:
    def __init__(self):
        self.models = {}
        self.preprocessor = DataPreprocessor()
        self.is_trained = False
        
    def train_models(self, X, y):
        """Train multiple ML models and evaluate their performance"""
        try:
            logger.info("Starting model training...")
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            logger.info(f"Training set size: {X_train.shape[0]}")
            logger.info(f"Test set size: {X_test.shape[0]}")
            
            # Define models to train
            model_configs = {
                "Random Forest": RandomForestClassifier(
                    n_estimators=400, max_features='sqrt', class_weight='balanced',
                    random_state=42, n_jobs=-1
                ),
                "SVM": SVC(
                    C=10, kernel='rbf', gamma='scale', probability=True,
                    class_weight='balanced', random_state=42
                ),
                "Logistic Regression": LogisticRegression(
                    max_iter=2000, solver='lbfgs', multi_class='auto', n_jobs=-1
                ),
                "Naive Bayes": GaussianNB()
            }         
            performance_metrics = {}
            
            # Train and evaluate each model
            for model_name, model in model_configs.items():
                logger.info(f"Training {model_name}...")
                
                # Train the model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                y_pred_proba = None
                
                if hasattr(model, 'predict_proba'):
                    y_pred_proba = model.predict_proba(X_test)
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                
                # Store model and metrics
                self.models[model_name] = model
                performance_metrics[model_name] = {
                    'accuracy': round(accuracy, 4),
                    'precision': round(precision, 4),
                    'recall': round(recall, 4),
                    'f1_score': round(f1, 4),

                }
                
                logger.info(f"{model_name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
            
            # Save models
            self._save_models()
            
            self.is_trained = True
            logger.info("Model training completed successfully")
            
            return self.models, performance_metrics
            
        except Exception as e:
            logger.error(f"Error training models: {str(e)}")
            raise
    
    def predict(self, user_symptoms):
        """Make predictions using all trained models"""
        try:
            if not self.is_trained and not self._load_models():
                logger.error("Models not trained and cannot be loaded")
                return {}
            
            # Preprocess user symptoms
            symptom_vector, matched_symptoms = self.preprocessor.preprocess_user_symptoms(user_symptoms)
            
            if np.sum(symptom_vector) == 0:
                logger.warning("No symptoms matched with available symptoms")
                return {}
            
            predictions = {}
            
            # Make predictions with each model
            for model_name, model in self.models.items():
                try:
                    # Get prediction
                    prediction = model.predict(symptom_vector)[0]
                    disease_name = self.preprocessor.decode_prediction(prediction)
                    
                    # Get prediction probability if available
                    confidence = 0.0
                    if hasattr(model, 'predict_proba'):
                        probabilities = model.predict_proba(symptom_vector)[0]
                        confidence = float(np.max(probabilities))
                    
                    predictions[model_name] = {
                        'disease': disease_name,
                        'confidence': round(confidence, 4),
                        'matched_symptoms': matched_symptoms
                    }
                    
                except Exception as e:
                    logger.error(f"Error making prediction with {model_name}: {str(e)}")
                    continue
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            raise
    
    def get_ensemble_prediction(self, user_symptoms):
        """Get ensemble prediction by combining all models"""
        try:
            predictions = self.predict(user_symptoms)
            
            if not predictions:
                return None
            
            # Count votes for each disease
            disease_votes = {}
            confidence_sum = {}
            
            for model_name, pred in predictions.items():
                disease = pred['disease']
                confidence = pred['confidence']
                
                if disease not in disease_votes:
                    disease_votes[disease] = 0
                    confidence_sum[disease] = 0
                
                disease_votes[disease] += 1
                confidence_sum[disease] += confidence
            
            # Find the disease with most votes
            best_disease = max(disease_votes.items(), key=lambda x: x[1])
            avg_confidence = confidence_sum[best_disease[0]] / disease_votes[best_disease[0]]
            
            return {
                'disease': best_disease[0],
                'votes': best_disease[1],
                'total_models': len(predictions),
                'average_confidence': round(avg_confidence, 4),
                'individual_predictions': predictions
            }
            
        except Exception as e:
            logger.error(f"Error getting ensemble prediction: {str(e)}")
            return None
    
    def _save_models(self):
        """Save trained models to disk"""
        try:
            os.makedirs('models', exist_ok=True)
            
            for model_name, model in self.models.items():
                filename = f"models/{model_name.lower().replace(' ', '_')}.joblib"
                joblib.dump(model, filename)
                logger.info(f"Saved {model_name} to {filename}")
            
            # Save preprocessor
            joblib.dump(self.preprocessor, 'models/preprocessor.joblib')
            
        except Exception as e:
            logger.error(f"Error saving models: {str(e)}")
    
    def _load_models(self):
        """Load trained models from disk"""
        try:
            models_dir = 'models'
            if not os.path.exists(models_dir):
                return False
            
            # Load preprocessor
            preprocessor_path = 'models/preprocessor.joblib'
            if os.path.exists(preprocessor_path):
                self.preprocessor = joblib.load(preprocessor_path)
            
            # Load models
            model_files = {
                'Random Forest': 'random_forest.joblib',
                'SVM': 'svm.joblib',
                'Naive Bayes': 'naive_bayes.joblib',
                'Logistic Regression': 'logistic_regression.joblib'
            }
            
            loaded_models = {}
            for model_name, filename in model_files.items():
                filepath = os.path.join(models_dir, filename)
                if os.path.exists(filepath):
                    loaded_models[model_name] = joblib.load(filepath)
                    logger.info(f"Loaded {model_name} from {filepath}")
            
            if loaded_models:
                self.models = loaded_models
                self.is_trained = True
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            return False
    
    def get_feature_importance(self, model_name='Random Forest'):
        """Get feature importance for tree-based models"""
        try:
            if model_name not in self.models:
                return {}
            
            model = self.models[model_name]
            
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                feature_importance = {}
                
                for i, importance in enumerate(importances):
                    if i < len(self.preprocessor.symptom_columns):
                        symptom = self.preprocessor.symptom_columns[i]
                        feature_importance[symptom] = round(importance, 4)
                
                # Sort by importance
                sorted_importance = dict(sorted(
                    feature_importance.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                ))
                
                return sorted_importance
            
            return {}
            
        except Exception as e:
            logger.error(f"Error getting feature importance: {str(e)}")
            return {}