import requests
import json
import time

# API base URL
BASE_URL = "http://localhost:5000"

def test_health_check():
    """Test health check endpoint"""
    print("Testing health check...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print("-" * 50)

def test_train_models():
    """Test model training endpoint"""
    print("Testing model training...")
    response = requests.post(f"{BASE_URL}/train-models")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print("-" * 50)

def test_get_symptoms():
    """Test get available symptoms endpoint"""
    print("Testing get available symptoms...")
    response = requests.get(f"{BASE_URL}/available-symptoms")
    print(f"Status: {response.status_code}")
    data = response.json()
    if data['status'] == 'success':
        print(f"Total symptoms: {data['total_count']}")
        print(f"First 10 symptoms: {data['symptoms'][:10]}")
    else:
        print(f"Error: {data}")
    print("-" * 50)

def test_get_diseases():
    """Test get available diseases endpoint"""
    print("Testing get available diseases...")
    response = requests.get(f"{BASE_URL}/diseases")
    print(f"Status: {response.status_code}")
    data = response.json()
    if data['status'] == 'success':
        print(f"Total diseases: {data['total_count']}")
        print(f"First 10 diseases: {data['diseases'][:10]}")
    else:
        print(f"Error: {data}")
    print("-" * 50)

def test_prediction():
    """Test disease prediction endpoint"""
    print("Testing disease prediction...")
    
    # Test symptoms
    test_symptoms = [
        "fever",
        "cough",
        "headache",
        "fatigue",
        "body ache"
    ]
    
    payload = {
        "symptoms": test_symptoms
    }
    
    response = requests.post(
        f"{BASE_URL}/predict",
        json=payload,
        headers={'Content-Type': 'application/json'}
    )
    
    print(f"Status: {response.status_code}")
    data = response.json()
    
    if data['status'] == 'success':
        print(f"Input symptoms: {data['input_symptoms']}")
        print(f"Most likely disease: {data['most_likely_disease']}")
        print(f"Confidence: {data['confidence']}")
        print(f"Precautions: {data['precautions']}")
        print("\nAll predictions:")
        for model, pred in data['predictions'].items():
            print(f"  {model}: {pred['disease']} (confidence: {pred['confidence']})")
    else:
        print(f"Error: {data}")
    print("-" * 50)

def test_model_performance():
    """Test model performance endpoint"""
    print("Testing model performance...")
    response = requests.get(f"{BASE_URL}/model-performance")
    print(f"Status: {response.status_code}")
    data = response.json()
    
    if data['status'] == 'success':
        print("Model Performance:")
        for model, metrics in data['performance'].items():
            print(f"  {model}:")
            for metric, value in metrics.items():
                print(f"    {metric}: {value}")
    else:
        print(f"Error: {data}")
    print("-" * 50)

def run_all_tests():
    """Run all API tests"""
    print("Starting API tests...\n")
    
    # Test health check
    test_health_check()
    
    # Test model training (this might take a while)
    test_train_models()
    
    # Wait a bit for training to complete
    time.sleep(2)
    
    # Test other endpoints
    test_get_symptoms()
    test_get_diseases()
    test_model_performance()
    test_prediction()
    
    print("All tests completed!")

if __name__ == "__main__":
    try:
        run_all_tests()
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the API. Make sure the server is running on http://localhost:5000")
    except Exception as e:
        print(f"Error running tests: {str(e)}")