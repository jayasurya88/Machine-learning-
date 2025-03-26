import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pickle
import os
import sys
import glob

def find_dataset_file():
    """Search for the dataset file in various locations"""
    possible_locations = [
        "fuel_consumption_vs_distance.csv",
        "data/fuel_consumption_vs_distance.csv",
        "datasets/fuel_consumption_vs_distance.csv",
        "ml_models/fuel_consumption_vs_distance.csv",
        "core/data/fuel_consumption_vs_distance.csv"
    ]
    
    # Get the project root directory
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # Search in all possible locations
    for location in possible_locations:
        full_path = os.path.join(project_root, location)
        if os.path.exists(full_path):
            return full_path
    
    # If not found, try to find any CSV file with similar name
    csv_files = glob.glob(os.path.join(project_root, "**/*.csv"), recursive=True)
    for file in csv_files:
        if "fuel" in file.lower() or "consumption" in file.lower():
            return file
    
    return None

def train_and_save_model():
    try:
        # Get the absolute path to the project root directory
        project_root = os.path.dirname(os.path.abspath(__file__))
        
        # Create models directory if it doesn't exist
        models_dir = os.path.join(project_root, 'core', 'models')
        os.makedirs(models_dir, exist_ok=True)
        print(f"Models directory: {models_dir}")

        # Find the dataset file
        file_path = find_dataset_file()
        if not file_path:
            print("\nError: Dataset file not found!")
            print("\nPlease ensure your dataset file (fuel_consumption_vs_distance.csv) is in one of these locations:")
            print("1. Project root directory (same as manage.py)")
            print("2. data/ directory")
            print("3. datasets/ directory")
            print("4. ml_models/ directory")
            print("5. core/data/ directory")
            print("\nOr place it in the project root directory with the name 'fuel_consumption_vs_distance.csv'")
            return False

        print(f"\nFound dataset at: {file_path}")
        print("Loading dataset...")
        df = pd.read_csv(file_path)
        
        # Print dataset info
        print("\nDataset Info:")
        print(f"Number of rows: {len(df)}")
        print(f"Columns: {', '.join(df.columns)}")
        print("\nFirst few rows:")
        print(df.head())

        # Extract features and target
        X = df[['Distance_Driven_km']]
        y = df['Fuel_Consumed_Liters']

        # Handle missing values using SimpleImputer
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        X = imputer.fit_transform(X)
        y = imputer.fit_transform(y.values.reshape(-1, 1)).ravel()

        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train Simple Linear Regression model
        print("\nTraining model...")
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predict on test set
        y_pred = model.predict(X_test)

        # Calculate accuracy
        accuracy = r2_score(y_test, y_pred)
        print(f"\nModel Accuracy: {accuracy:.2%}")

        # Save trained model
        model_path = os.path.join(models_dir, 'slr_model.pkl')
        with open(model_path, "wb") as file:
            pickle.dump(model, file)

        print(f"\nModel successfully saved to: {model_path}")
        
        # Verify the file exists
        if os.path.exists(model_path):
            print("Model file verified successfully")
            return True
        else:
            print("Error: Model file was not created successfully")
            return False

    except Exception as e:
        print(f"\nError during model training and saving: {str(e)}")
        return False

if __name__ == "__main__":
    print("Starting model training process...")
    success = train_and_save_model()
    if not success:
        print("\nFailed to train and save the model")
        sys.exit(1) 