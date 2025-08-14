# 🏥 Disease Prediction API - Backend

*An intelligent machine learning API that predicts diseases based on symptoms using multiple ML algorithms*

[Features](#-features) -  [Quick Start](#-quick-start) -  [API Documentation](#-api-documentation) -  [Deployment](#-deployment)



## 📋 Table of Contents

- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Quick Start](#-quick-start)
- [API Documentation](#-api-documentation)
- [Machine Learning Models](#-machine-learning-models)
- [Dataset Information](#-dataset-information)
- [Environment Variables](#-environment-variables)
- [Deployment](#-deployment)
- [Testing](#-testing)
- [Performance Metrics](#-performance-metrics)
- [Contributing](#-contributing)
- [License](#-license)

## ✨ Features

🧠 **Advanced ML Pipeline**
- 6 different machine learning algorithms (Random Forest, SVM, Naive Bayes, etc.)
- Ensemble prediction with confidence scoring
- Automatic model training on startup

🔄 **Intelligent Preprocessing** 
- Smart symptom matching and normalization
- Handles variations in symptom descriptions
- Input validation and error handling

📊 **Performance Analytics**
- Comprehensive model metrics (Accuracy, Precision, Recall, F1-Score)
- 5-fold cross-validation scores
- Model comparison and selection

🛡️ **Production Ready**
- RESTful API with proper error handling
- CORS support for frontend integration
- Health check endpoints
- Comprehensive logging

## 🛠 Tech Stack

- **Framework:** Flask 2.3+
- **ML Library:** Scikit-learn 1.3+
- **Data Processing:** Pandas, NumPy
- **API:** RESTful with JSON responses
- **Deployment:** Render
- **Environment:** Python 3.11+

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/disease-predictor-backend.git
cd disease-predictor-backend
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env with your configuration
```

4. **Run the application**
```bash
python app.py
```

The API will be available at `http://localhost:5000`

### Quick Test
```bash
# Health check
curl http://localhost:5000/health

# Make a prediction
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"symptoms": ["fever", "cough", "headache"]}'
```

## 📚 API Documentation

### Base URL
- **Production:** `https://disease-predictor-api-lryq.onrender.com`
- **Local:** `http://localhost:5000`

### Endpoints

#### 🏠 Root Information
```
GET /
```
Returns API information and available endpoints.

#### ❤️ Health Check
```
GET /health
```
Returns API health status and model training information.

#### 🔮 Predict Disease
```
POST /predict
Content-Type: application/json

{
    "symptoms": ["fever", "cough", "headache", "fatigue"]
}
```

**Response:**
```json
{
    "status": "success",
    "most_likely_disease": "Common Cold",
    "confidence": 0.85,
    "predictions": {
        "random_forest": {
            "disease": "Common Cold",
            "confidence": 0.85,
            "accuracy": 0.92
        }
    },
    "precautions": [
        "Get plenty of rest",
        "Drink lots of fluids",
        "Take over-the-counter pain relievers"
    ],
    "input_symptoms": ["fever", "cough", "headache", "fatigue"]
}
```

#### 🎯 Train Models
```
POST /train-models
```
Manually trigger model training and return performance metrics.

#### 📝 Available Symptoms
```
GET /available-symptoms
```
Get list of all available symptoms in the dataset.

#### 🦠 Available Diseases
```
GET /diseases
```
Get list of all diseases that can be predicted.

#### 📊 Model Performance
```
GET /model-performance
```
Get detailed performance metrics for all trained models.

## 🤖 Machine Learning Models

Our ensemble approach uses 6 different algorithms:

| Model | Purpose | Strengths |
|-------|---------|-----------|
| **Random Forest** | Primary predictor | High accuracy, handles overfitting |
| **SVM (Support Vector Machine)** | Pattern recognition | Effective in high dimensions |
| **Naive Bayes** | Probabilistic prediction | Fast, works well with small datasets |
| **Logistic Regression** | Linear relationships | Interpretable, fast training |
| **Gradient Boosting** | Ensemble method | High performance, reduces bias |
| **K-Nearest Neighbors** | Similarity matching | Simple, effective for certain patterns |

### Model Selection
The API automatically selects the best prediction based on confidence scores and combines insights from multiple models for robust disease prediction.

## 📊 Dataset Information

### Disease & Symptoms Dataset
- **File:** `data/DiseaseAndSymptoms.csv`
- **Records:** 4000+ symptom-disease combinations
- **Diseases:** 40+ different conditions
- **Symptoms:** 130+ unique symptoms

### Precautions Dataset
- **File:** `data/Disease precaution.csv`
- **Content:** 4 precautionary measures for each disease
- **Format:** Disease name with corresponding preventive actions

## 🔧 Environment Variables

Create a `.env` file in the root directory:

```env
# Flask Configuration
FLASK_DEBUG=False
SECRET_KEY=your-super-secret-key-here

# Data Paths
SYMPTOMS_DATA_PATH=data/DiseaseAndSymptoms.csv
PRECAUTIONS_DATA_PATH=data/Disease precaution.csv

# ML Configuration
AUTO_TRAIN_ON_STARTUP=True
MAX_SYMPTOMS_PER_REQUEST=20
CV_FOLDS=5
TEST_SIZE=0.2
RANDOM_STATE=42

# Deployment
PORT=5000
```

## 🚀 Deployment

### Deploy to Render

1. **Connect your GitHub repository** to Render

2. **Configure build settings:**
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app`

3. **Set environment variables** in Render dashboard

4. **Deploy!** Your API will be live with automatic HTTPS

### Docker Deployment
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 5000

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000"]
```

## 🧪 Testing

### Run Test Suite
```bash
python test_api.py
```

### Manual Testing Examples

**Test with different symptoms:**
```bash
# Respiratory symptoms
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"symptoms": ["continuous_sneezing", "chills", "fatigue"]}'

# Digestive symptoms  
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"symptoms": ["stomach_pain", "vomiting", "nausea"]}'
```

## 📈 Performance Metrics

Our models achieve excellent performance across key metrics:

- **Overall Accuracy:** 90%+ across all models
- **Cross-Validation Score:** 88-94% range
- **Precision:** High precision for common diseases
- **Recall:** Balanced recall across disease categories
- **F1-Score:** Optimized for medical prediction accuracy

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run linting
flake8 .

# Run tests
pytest
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Medical Disclaimer

**Important:** This application is for educational and informational purposes only. It should **never** be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare providers for medical concerns.

## 🙏 Acknowledgments

- Scikit-learn community for excellent ML tools
- Flask team for the lightweight web framework
- Medical dataset contributors
- Open source community

***

[Report Bug](https://github.com/yourusername/disease-predictor-backend/issues) -  [Request Feature](https://github.com/yourusername/disease-predictor-backend/issues)
