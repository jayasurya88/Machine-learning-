import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import pickle
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_sample_dataset():
    """Create a synthetic dataset for car performance modeling"""
    np.random.seed(42)
    n_samples = 1000
    
    # Generate car specifications data
    weights = np.random.normal(1500, 300, n_samples)  # Weight in kg
    powers = np.random.normal(150, 50, n_samples)     # Power in hp
    displacements = np.random.normal(2.0, 0.5, n_samples)  # Displacement in L
    drag_coefficients = np.random.normal(0.3, 0.05, n_samples)  # Drag coefficient
    
    # Calculate power-to-weight ratio
    power_to_weight = powers / (weights / 1000)  # hp/ton
    
    # Generate speed array (0 to 200 km/h)
    speeds = np.linspace(0, 200, 201)
    
    # Create a dataframe to store all samples
    data = []
    
    for i in range(n_samples):
        weight = weights[i]
        power = powers[i]
        displacement = displacements[i]
        drag = drag_coefficients[i]
        ptw = power_to_weight[i]
        
        for speed in speeds:
            # Skip some speeds for sparser dataset
            if speed % 5 != 0 and np.random.rand() > 0.1:
                continue
                
            # Add some gradient values
            gradient = np.random.uniform(-10, 10)
            
            # Calculate fuel consumption using polynomial relationship with speed
            # Lower consumption at moderate speeds, higher at low and high speeds
            base_consumption = 5 + 0.015 * (speed - 80)**2 / 400
            
            # Adjust for weight (heavier cars use more fuel)
            weight_factor = weight / 1500
            
            # Adjust for power (more powerful cars use more fuel)
            power_factor = power / 150
            
            # Adjust for displacement
            displacement_factor = displacement / 2.0
            
            # Adjust for drag (higher drag means more fuel at high speeds)
            drag_factor = 1 + (drag - 0.3) * (speed / 100)**2
            
            # Adjust for gradient (uphill uses more fuel)
            gradient_factor = 1 + 0.01 * gradient * (weight / 1500)
            
            # Calculate fuel consumption
            fuel_consumption = (
                base_consumption * 
                weight_factor * 
                power_factor * 
                displacement_factor * 
                drag_factor * 
                gradient_factor
            )
            
            # Add some noise
            fuel_consumption += np.random.normal(0, 0.5)
            fuel_consumption = max(0, fuel_consumption)
            
            # Calculate acceleration capacity (simplified model)
            # Cars with better power-to-weight accelerate faster
            acceleration_0_100 = 20 - 0.8 * ptw
            acceleration_0_100 = max(2, min(20, acceleration_0_100))
            
            # Calculate theoretical top speed (simplified model)
            # Based on power, weight, and drag
            top_speed = 80 + 60 * (power / weight)**0.5 / (drag**0.3)
            top_speed = min(300, top_speed)
            
            # Add to data
            data.append({
                'Weight': weight,
                'Power': power,
                'Displacement': displacement,
                'Drag_Coefficient': drag,
                'Speed': speed,
                'Gradient': gradient,
                'Fuel_Consumption': fuel_consumption,
                'Acceleration_0_100': acceleration_0_100,
                'Top_Speed': top_speed,
                'Power_to_Weight': ptw
            })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    return df

def train_polynomial_model(degree=3):
    """Train a polynomial regression model for car performance prediction"""
    try:
        # Create or load dataset
        data_path = 'car_performance_data.csv'
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
            logger.info(f"Loaded existing dataset from {data_path}")
        else:
            df = create_sample_dataset()
            df.to_csv(data_path, index=False)
            logger.info(f"Created new dataset and saved to {data_path}")
        
        logger.info(f"Dataset shape: {df.shape}")
        
        # Train fuel consumption model
        logger.info(f"Training fuel consumption model with polynomial degree {degree}...")
        
        # Features for fuel consumption prediction
        X_fuel = df[['Weight', 'Power', 'Displacement', 'Drag_Coefficient', 'Speed', 'Gradient']]
        y_fuel = df['Fuel_Consumption']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_fuel, y_fuel, test_size=0.2, random_state=42)
        
        # Create polynomial model
        fuel_model = Pipeline([
            ('poly', PolynomialFeatures(degree=degree)),
            ('linear', LinearRegression())
        ])
        
        # Train model
        fuel_model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = fuel_model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        logger.info(f"Fuel Consumption Model - R² score: {r2:.4f}, RMSE: {rmse:.4f}")
        
        # Train acceleration model
        logger.info("Training acceleration model...")
        
        # Features for acceleration prediction
        X_accel = df[['Weight', 'Power', 'Displacement']].drop_duplicates()
        y_accel = X_accel.apply(lambda row: 20 - 0.8 * (row['Power'] / (row['Weight'] / 1000)), axis=1)
        
        # Split data
        X_accel_train, X_accel_test, y_accel_train, y_accel_test = train_test_split(
            X_accel, y_accel, test_size=0.2, random_state=42
        )
        
        # Create polynomial model
        accel_model = Pipeline([
            ('poly', PolynomialFeatures(degree=2)),
            ('linear', LinearRegression())
        ])
        
        # Train model
        accel_model.fit(X_accel_train, y_accel_train)
        
        # Evaluate model
        y_accel_pred = accel_model.predict(X_accel_test)
        r2_accel = r2_score(y_accel_test, y_accel_pred)
        
        logger.info(f"Acceleration Model - R² score: {r2_accel:.4f}")
        
        # Save models
        models_dir = 'core/models'
        os.makedirs(models_dir, exist_ok=True)
        
        with open(os.path.join(models_dir, 'polynomial_fuel_model.pkl'), 'wb') as f:
            pickle.dump(fuel_model, f)
            
        with open(os.path.join(models_dir, 'polynomial_accel_model.pkl'), 'wb') as f:
            pickle.dump(accel_model, f)
            
        logger.info("Models saved successfully")
        
        # Generate and save plot of fuel consumption vs. speed for a typical car
        typical_car = pd.DataFrame({
            'Weight': [1500] * 201,
            'Power': [150] * 201,
            'Displacement': [2.0] * 201,
            'Drag_Coefficient': [0.3] * 201,
            'Speed': np.linspace(0, 200, 201),
            'Gradient': [0] * 201
        })
        
        typical_consumption = fuel_model.predict(typical_car)
        
        plt.figure(figsize=(10, 6))
        plt.plot(typical_car['Speed'], typical_consumption)
        plt.title('Fuel Consumption vs. Speed for Typical Car')
        plt.xlabel('Speed (km/h)')
        plt.ylabel('Fuel Consumption (L/100km)')
        plt.grid(True)
        plt.savefig(os.path.join(models_dir, 'fuel_consumption_curve.png'))
        logger.info("Saved fuel consumption curve plot")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in training process: {str(e)}")
        return False

def test_model_predictions():
    """Test polynomial model predictions with sample inputs"""
    try:
        # Load models
        models_dir = 'core/models'
        
        with open(os.path.join(models_dir, 'polynomial_fuel_model.pkl'), 'rb') as f:
            fuel_model = pickle.load(f)
            
        with open(os.path.join(models_dir, 'polynomial_accel_model.pkl'), 'rb') as f:
            accel_model = pickle.load(f)
        
        # Test car specifications
        test_cars = [
            {
                'name': 'Economy Car',
                'weight': 1200,
                'power': 90,
                'displacement': 1.4,
                'drag': 0.32
            },
            {
                'name': 'Family Sedan',
                'weight': 1600,
                'power': 160,
                'displacement': 2.0,
                'drag': 0.30
            },
            {
                'name': 'Sports Car',
                'weight': 1400,
                'power': 300,
                'displacement': 3.0,
                'drag': 0.28
            },
            {
                'name': 'SUV',
                'weight': 2200,
                'power': 220,
                'displacement': 2.5,
                'drag': 0.35
            }
        ]
        
        for car in test_cars:
            logger.info(f"\nTesting {car['name']}:")
            
            # Create speed range
            speeds = np.linspace(0, 200, 41)
            
            # Create input dataframe for fuel consumption prediction
            inputs = pd.DataFrame({
                'Weight': [car['weight']] * len(speeds),
                'Power': [car['power']] * len(speeds),
                'Displacement': [car['displacement']] * len(speeds),
                'Drag_Coefficient': [car['drag']] * len(speeds),
                'Speed': speeds,
                'Gradient': [0] * len(speeds)
            })
            
            # Predict fuel consumption
            consumption = fuel_model.predict(inputs)
            
            # Find optimal speed (lowest consumption)
            optimal_speed = speeds[np.argmin(consumption)]
            min_consumption = np.min(consumption)
            
            logger.info(f"Optimal speed: {optimal_speed:.1f} km/h")
            logger.info(f"Minimum consumption: {min_consumption:.2f} L/100km")
            
            # Predict acceleration
            accel_input = pd.DataFrame({
                'Weight': [car['weight']],
                'Power': [car['power']],
                'Displacement': [car['displacement']]
            })
            
            acceleration = accel_model.predict(accel_input)[0]
            
            logger.info(f"0-100 km/h acceleration: {acceleration:.2f} seconds")
            
            # Calculate power-to-weight ratio
            ptw = car['power'] / (car['weight'] / 1000)
            logger.info(f"Power-to-weight ratio: {ptw:.2f} hp/ton")
            
            # Calculate estimated top speed
            top_speed = 80 + 60 * (car['power'] / car['weight'])**0.5 / (car['drag']**0.3)
            logger.info(f"Estimated top speed: {top_speed:.1f} km/h")
    
    except Exception as e:
        logger.error(f"Error testing model: {str(e)}")

if __name__ == "__main__":
    logger.info("Training polynomial regression models for car performance...")
    success = train_polynomial_model(degree=3)
    
    if success:
        logger.info("Testing model predictions...")
        test_model_predictions()
    else:
        logger.error("Model training failed. Cannot proceed with testing.") 