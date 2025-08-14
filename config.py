import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    # Flask configuration
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    DEBUG = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'
    
    # Data paths
    SYMPTOMS_DATA_PATH = os.environ.get('SYMPTOMS_DATA_PATH', 'data/DiseaseAndSymptoms.csv')
    PRECAUTIONS_DATA_PATH = os.environ.get('PRECAUTIONS_DATA_PATH', 'data/Disease precaution.csv')
    
    # Model configuration
    MODELS_DIR = os.environ.get('MODELS_DIR', 'models')
    AUTO_TRAIN_ON_STARTUP = os.environ.get('AUTO_TRAIN_ON_STARTUP', 'True').lower() == 'true'
    
    # API configuration
    MAX_SYMPTOMS_PER_REQUEST = int(os.environ.get('MAX_SYMPTOMS_PER_REQUEST', '20'))
    
    # Cross-validation settings
    CV_FOLDS = int(os.environ.get('CV_FOLDS', '5'))
    TEST_SIZE = float(os.environ.get('TEST_SIZE', '0.2'))
    RANDOM_STATE = int(os.environ.get('RANDOM_STATE', '42'))

class DevelopmentConfig(Config):
    DEBUG = True

class ProductionConfig(Config):
    DEBUG = False

class TestingConfig(Config):
    TESTING = True
    DEBUG = True

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}