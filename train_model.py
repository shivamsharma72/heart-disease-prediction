import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

def predict_heart_disease(data):
    """
    Make heart disease predictions using the trained model
    
    Parameters:
    data (dict): Dictionary containing patient information with the following keys:
        - Age: int (29-77)
        - Sex: int (0=female, 1=male)
        - ChestPainType: int (0-3)
        - RestingBP: int (94-200 mmHg)
        - Cholesterol: int (126-564 mg/dl)
        - FastingBS: int (0=<=120mg/dl, 1=>120mg/dl)
        - RestingECG: int (0-2)
        - MaxHR: int (71-202 bpm)
        - ExerciseAngina: int (0=no, 1=yes)
        - Oldpeak: float (0.0-6.2)
        - ST_Slope: int (0-2)
        - NumMajorVessels: int (0-4)
        - Thal: int (0-3)
    
    Returns:
    dict: Prediction results containing:
        - prediction: 0 (no heart disease) or 1 (heart disease)
        - probability: Probability of heart disease
    """
    
    # Validate input data
    required_features = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 
                        'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 
                        'Oldpeak', 'ST_Slope', 'NumMajorVessels', 'Thal']
    
    # Check if all required features are present
    missing_features = [f for f in required_features if f not in data]
    if missing_features:
        return {"error": f"Missing required features: {', '.join(missing_features)}"}
    
    # Validate value ranges
    validations = {
        'Age': (29, 77),
        'Sex': (0, 1),
        'ChestPainType': (0, 3),
        'RestingBP': (94, 200),
        'Cholesterol': (126, 564),
        'FastingBS': (0, 1),
        'RestingECG': (0, 2),
        'MaxHR': (71, 202),
        'ExerciseAngina': (0, 1),
        'Oldpeak': (0.0, 6.2),
        'ST_Slope': (0, 2),
        'NumMajorVessels': (0, 4),
        'Thal': (0, 3)
    }
    
    for feature, (min_val, max_val) in validations.items():
        value = data[feature]
        if not min_val <= value <= max_val:
            return {"error": f"{feature} value {value} is outside valid range ({min_val}-{max_val})"}
    
    # Load the trained model
    try:
        model = joblib.load('heart_disease_classification_model.pkl')
        scaler = joblib.load('scaler.pkl')  # You'll need to save this during training
    except FileNotFoundError:
        return {"error": "Model files not found. Please ensure the model is trained and saved."}
    
    # Convert input data to DataFrame
    input_df = pd.DataFrame([data])
    
    # Scale the features
    scaled_features = scaler.transform(input_df)
    
    # Make prediction
    prediction = model.predict(scaled_features)[0]
    probability = model.predict_proba(scaled_features)[0][1]
    
    return {
        "prediction": int(prediction),
        "probability": float(probability)
    }

# Example usage
if __name__ == "__main__":
    # Example test case
    test_data = {
        "Age": 58,
        "Sex": 0,
        "ChestPainType": 1,
        "RestingBP": 120,
        "Cholesterol": 244,
        "FastingBS": 0,
        "RestingECG": 1,
        "MaxHR": 162,
        "ExerciseAngina": 0,
        "Oldpeak": 1.1,
        "ST_Slope": 2,
        "NumMajorVessels": 0,
        "Thal": 2
    }
    
    # Make prediction
    result = predict_heart_disease(test_data)
    
    # Print results
    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print("\nPrediction Results:")
        print(f"Heart Disease: {'Yes' if result['prediction'] == 1 else 'No'}")
        print(f"Probability: {result['probability']:.2%}") 