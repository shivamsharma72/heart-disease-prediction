import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

def create_and_save_scaler(data_path='heart.csv'):
    """
    Create and save a StandardScaler fitted on the training data
    
    Parameters:
    data_path (str): Path to the heart disease dataset CSV file
    
    Returns:
    tuple: (scaler, dataframe) - The fitted scaler and the dataframe used to fit it
    """
    # Load the dataset
    df = pd.read_csv(data_path)
    
    # Rename columns for consistency
    df.columns = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 
                 'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 
                 'Oldpeak', 'ST_Slope', 'NumMajorVessels', 'Thal', 'Target']
    
    # Separate features from target
    X = df.drop('Target', axis=1)
    
    # Create and fit the scaler
    scaler = StandardScaler()
    scaler.fit(X)
    
    # Save the scaler
    joblib.dump(scaler, 'scaler.pkl')
    print("Scaler has been created and saved as 'scaler.pkl'")
    
    # Return both the scaler and dataframe
    return scaler, df

if __name__ == "__main__":
    # Create and save the scaler
    scaler, df = create_and_save_scaler()
    
    # Print some basic statistics to verify the scaler
    print("\nScaler mean values:")
    for feature, mean in zip(df.columns[:-1], scaler.mean_):
        print(f"{feature}: {mean:.2f}")