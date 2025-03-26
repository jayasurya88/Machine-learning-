import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error
import pickle
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_sample_dataset():
    """Create a sample dataset with multiple features"""
    np.random.seed(42)
    n_samples = 1000
    
    # Generate random data
    distance = np.random.uniform(10, 500, n_samples)
    speed = np.random.uniform(30, 120, n_samples)
    vehicle_weight = np.random.uniform(1000, 3000, n_samples)
    temperature = np.random.uniform(5, 35, n_samples)
    traffic_density = np.random.choice(['low', 'medium', 'high'], n_samples)
    
    # Create target variable with some noise
    base_consumption = (
        0.1 * distance +  # Base consumption based on distance
        0.05 * speed +   # Additional consumption based on speed
        0.001 * vehicle_weight +  # Weight impact
        0.01 * temperature +  # Temperature impact
        np.random.normal(0, 2, n_samples)  # Random noise
    )
    
    # Ensure positive values
    fuel_consumption = np.maximum(base_consumption, 0)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Distance_Driven_km': distance,
        'Average_Speed_kmh': speed,
        'Vehicle_Weight_kg': vehicle_weight,
        'Temperature_C': temperature,
        'Traffic_Density': traffic_density,
        'Fuel_Consumed_Liters': fuel_consumption
    })
    
    return df

def train_and_save_model():
    try:
        # Get the absolute path to the project root directory
        project_root = os.path.dirname(os.path.abspath(__file__))
        
        # Create models directory if it doesn't exist
        models_dir = os.path.join(project_root, 'core', 'models')
        os.makedirs(models_dir, exist_ok=True)
        logger.info(f"Models directory: {models_dir}")

        # Create or load dataset
        data_path = os.path.join(project_root, 'fuel_consumption_mlr.csv')
        if not os.path.exists(data_path):
            logger.info("Creating sample dataset...")
            df = create_sample_dataset()
            df.to_csv(data_path, index=False)
        else:
            logger.info("Loading existing dataset...")
            df = pd.read_csv(data_path)
        
        # Print dataset info
        logger.info("\nDataset Info:")
        logger.info(f"Number of rows: {len(df)}")
        logger.info(f"Columns: {', '.join(df.columns)}")
        logger.info("\nFirst few rows:")
        logger.info(df.head())

        # Prepare features and target
        X = df[['Distance_Driven_km', 'Average_Speed_kmh', 'Vehicle_Weight_kg', 'Temperature_C', 'Traffic_Density']]
        y = df['Fuel_Consumed_Liters']

        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Prepare numerical and categorical features
        numerical_features = ['Distance_Driven_km', 'Average_Speed_kmh', 'Vehicle_Weight_kg', 'Temperature_C']
        categorical_features = ['Traffic_Density']

        # Scale numerical features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train[numerical_features])
        X_test_scaled = scaler.transform(X_test[numerical_features])

        # Encode categorical features
        label_encoder = LabelEncoder()
        X_train_traffic = label_encoder.fit_transform(X_train['Traffic_Density'])
        X_test_traffic = label_encoder.transform(X_test['Traffic_Density'])

        # Combine features
        X_train_processed = np.hstack([X_train_scaled, X_train_traffic.reshape(-1, 1)])
        X_test_processed = np.hstack([X_test_scaled, X_test_traffic.reshape(-1, 1)])

        # Train Multiple Linear Regression model
        logger.info("\nTraining model...")
        model = LinearRegression()
        model.fit(X_train_processed, y_train)

        # Make predictions
        y_pred = model.predict(X_test_processed)

        # Calculate metrics
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)

        logger.info(f"\nModel Performance:")
        logger.info(f"R² Score: {r2:.4f}")
        logger.info(f"RMSE: {rmse:.4f}")

        # Save model and preprocessors
        model_path = os.path.join(models_dir, 'mlr_model.pkl')
        scaler_path = os.path.join(models_dir, 'mlr_scaler.pkl')
        label_encoder_path = os.path.join(models_dir, 'mlr_label_encoder.pkl')

        with open(model_path, 'wb') as file:
            pickle.dump(model, file)
        with open(scaler_path, 'wb') as file:
            pickle.dump(scaler, file)
        with open(label_encoder_path, 'wb') as file:
            pickle.dump(label_encoder, file)

        logger.info(f"\nModel and preprocessors saved to:")
        logger.info(f"Model: {model_path}")
        logger.info(f"Scaler: {scaler_path}")
        logger.info(f"Label Encoder: {label_encoder_path}")

        # Print feature importance
        feature_importance = dict(zip(
            numerical_features + ['Traffic_Density'],
            model.coef_
        ))
        logger.info("\nFeature Importance:")
        for feature, importance in feature_importance.items():
            logger.info(f"{feature}: {importance:.4f}")

        # Test prediction with sample input
        sample_input = {
            'Distance_Driven_km': 100,
            'Average_Speed_kmh': 60,
            'Vehicle_Weight_kg': 1500,
            'Temperature_C': 25,
            'Traffic_Density': 'medium'
        }
        
        # Prepare sample input
        sample_numerical = np.array([[sample_input['Distance_Driven_km'],
                                    sample_input['Average_Speed_kmh'],
                                    sample_input['Vehicle_Weight_kg'],
                                    sample_input['Temperature_C']]])
        sample_scaled = scaler.transform(sample_numerical)
        sample_traffic = label_encoder.transform([sample_input['Traffic_Density']])
        sample_processed = np.hstack([sample_scaled, sample_traffic.reshape(-1, 1)])
        
        # Make prediction
        sample_prediction = model.predict(sample_processed)[0]
        
        logger.info("\nSample Prediction Test:")
        logger.info(f"Input: {sample_input}")
        logger.info(f"Predicted Fuel Consumption: {sample_prediction:.2f} liters")

        return True

    except Exception as e:
        logger.error(f"\nError during model training and saving: {str(e)}")
        return False

if __name__ == "__main__":
    logger.info("Starting MLR model training process...")
    success = train_and_save_model()
    if not success:
        logger.error("\nFailed to train and save the model")
        exit(1) 