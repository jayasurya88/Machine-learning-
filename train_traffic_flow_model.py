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
import shutil

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_sample_dataset():
    """Create a synthetic dataset for traffic flow prediction"""
    np.random.seed(42)
    n_samples = 1000
    
    # Features
    time_of_day = np.random.randint(0, 24, n_samples)  # Hour of day (0-23)
    day_of_week = np.random.randint(1, 8, n_samples)   # Day of week (1-7, 1=Monday)
    weather_condition = np.random.randint(1, 5, n_samples)  # 1=Clear, 2=Light Rain, 3=Heavy Rain, 4=Snow
    road_type = np.random.randint(1, 4, n_samples)  # 1=Highway, 2=Main Road, 3=Residential
    
    # Generate traffic volume based on features
    # Base volume
    base_volume = 500  # Base vehicles per hour
    
    # Time of day effect: peak at morning and evening rush hours
    time_effect = np.zeros(n_samples)
    for i, time in enumerate(time_of_day):
        if 6 <= time <= 9:  # Morning rush: 6-9 AM
            time_effect[i] = 800 + 200 * np.sin((time - 6) * np.pi / 3)
        elif 16 <= time <= 19:  # Evening rush: 4-7 PM
            time_effect[i] = 900 + 100 * np.sin((time - 16) * np.pi / 3)
        elif 10 <= time <= 15:  # Midday
            time_effect[i] = 500
        else:  # Night
            time_effect[i] = 200
    
    # Day of week effect
    day_effect = np.zeros(n_samples)
    for i, day in enumerate(day_of_week):
        if day <= 5:  # Weekday
            day_effect[i] = 1.0
        else:  # Weekend
            day_effect[i] = 0.7
    
    # Weather effect
    weather_effect = np.zeros(n_samples)
    for i, weather in enumerate(weather_condition):
        if weather == 1:  # Clear
            weather_effect[i] = 1.0
        elif weather == 2:  # Light Rain
            weather_effect[i] = 0.9
        elif weather == 3:  # Heavy Rain
            weather_effect[i] = 0.7
        else:  # Snow
            weather_effect[i] = 0.5
    
    # Road type effect
    road_effect = np.zeros(n_samples)
    for i, road in enumerate(road_type):
        if road == 1:  # Highway
            road_effect[i] = 2.0
        elif road == 2:  # Main Road
            road_effect[i] = 1.5
        else:  # Residential
            road_effect[i] = 0.7
    
    # Calculate traffic volume
    traffic_volume = (base_volume + time_effect) * day_effect * weather_effect * road_effect
    
    # Add some random noise
    traffic_volume += np.random.normal(0, 100, n_samples)
    traffic_volume = np.maximum(0, traffic_volume)  # Ensure no negative values
    
    # Create DataFrame
    df = pd.DataFrame({
        'Time_of_Day': time_of_day,
        'Day_of_Week': day_of_week,
        'Weather_Condition': weather_condition,
        'Road_Type': road_type,
        'Traffic_Volume': traffic_volume
    })
    
    return df

def train_traffic_model(degree=3):
    """Train a polynomial regression model for traffic flow prediction"""
    try:
        # Create or load dataset
        data_path = 'traffic_flow_data.csv'
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
            logger.info(f"Loaded existing dataset from {data_path}")
        else:
            df = create_sample_dataset()
            df.to_csv(data_path, index=False)
            logger.info(f"Created new dataset and saved to {data_path}")
        
        logger.info(f"Dataset shape: {df.shape}")
        logger.info(f"Dataset columns: {df.columns.tolist()}")
        logger.info(f"Dataset sample: \n{df.head()}")
        
        # Features and target
        X = df[['Time_of_Day', 'Day_of_Week', 'Weather_Condition', 'Road_Type']]
        y = df['Traffic_Volume']
        
        logger.info(f"Feature matrix shape: {X.shape}")
        logger.info(f"Target vector shape: {y.shape}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create polynomial model
        model = Pipeline([
            ('poly', PolynomialFeatures(degree=degree)),
            ('linear', LinearRegression())
        ])
        
        # Train model
        logger.info("Training model...")
        model.fit(X_train, y_train)
        logger.info("Model training completed")
        
        # Evaluate model
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        logger.info(f"Traffic Flow Model - R² score: {r2:.4f}, RMSE: {rmse:.4f}")
        
        # Make sure model directory exists
        models_dir = 'core/models'
        os.makedirs(models_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(models_dir, 'traffic_flow_model.pkl')
        
        # Backup existing model if it exists
        if os.path.exists(model_path):
            backup_path = model_path + '.bak'
            logger.info(f"Backing up existing model to {backup_path}")
            shutil.copy2(model_path, backup_path)
        
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Verify saved model
        if os.path.exists(model_path):
            model_size = os.path.getsize(model_path)
            logger.info(f"Model saved successfully at {model_path} (size: {model_size} bytes)")
            if model_size < 100:
                logger.warning(f"Model file is suspiciously small: {model_size} bytes")
        else:
            logger.error(f"Failed to save model at {model_path}")
            
        # Generate and save traffic pattern plot for visualization
        plt.figure(figsize=(10, 6))
        
        # Group by time of day and plot average traffic volume
        time_traffic = df.groupby('Time_of_Day')['Traffic_Volume'].mean()
        plt.plot(time_traffic.index, time_traffic.values)
        plt.title('Average Traffic Volume by Time of Day')
        plt.xlabel('Hour of Day')
        plt.ylabel('Traffic Volume (vehicles/hour)')
        plt.grid(True)
        plt.xticks(range(0, 24, 2))
        plt.savefig(os.path.join(models_dir, 'traffic_pattern.png'))
        
        logger.info("Saved traffic pattern plot")
        
        # Test model with sample inputs
        test_traffic_prediction()
        
        return True
        
    except Exception as e:
        import traceback
        logger.error(f"Error in training process: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def test_traffic_prediction():
    """Test traffic flow prediction with sample inputs"""
    try:
        # Load model
        models_dir = 'core/models'
        model_path = os.path.join(models_dir, 'traffic_flow_model.pkl')
        
        if not os.path.exists(model_path):
            logger.error(f"Model file not found at {model_path}")
            return False
        
        logger.info(f"Loading model from {model_path} (size: {os.path.getsize(model_path)} bytes)")
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        logger.info(f"Model loaded successfully: {type(model)}")
        
        # Test scenarios
        test_scenarios = [
            {
                'name': 'Morning Rush Hour (Monday, Clear, Highway)',
                'time': 8,
                'day': 1,
                'weather': 1,
                'road': 1
            },
            {
                'name': 'Weekend Midday (Saturday, Light Rain, Main Road)',
                'time': 12,
                'day': 6,
                'weather': 2,
                'road': 2
            },
            {
                'name': 'Evening Rush Hour (Friday, Clear, Main Road)',
                'time': 17,
                'day': 5,
                'weather': 1,
                'road': 2
            },
            {
                'name': 'Night (Wednesday, Heavy Rain, Residential)',
                'time': 22,
                'day': 3,
                'weather': 3,
                'road': 3
            }
        ]
        
        logger.info("Testing traffic predictions with sample scenarios:")
        
        for scenario in test_scenarios:
            # Prepare input data
            test_input = pd.DataFrame({
                'Time_of_Day': [scenario['time']],
                'Day_of_Week': [scenario['day']],
                'Weather_Condition': [scenario['weather']],
                'Road_Type': [scenario['road']]
            })
            
            logger.info(f"Test input for {scenario['name']}: {test_input.to_dict()}")
            
            # Make prediction
            prediction = model.predict(test_input)[0]
            
            # Define congestion levels
            congestion_level = "Unknown"
            if prediction < 500:
                congestion_level = "Light"
            elif prediction < 1000:
                congestion_level = "Moderate"
            elif prediction < 1500:
                congestion_level = "Heavy"
            else:
                congestion_level = "Severe"
            
            # Calculate travel time increase
            base_travel_time = 20  # minutes
            if congestion_level == "Light":
                time_increase = "0-10%"
            elif congestion_level == "Moderate":
                time_increase = "10-30%"
            elif congestion_level == "Heavy":
                time_increase = "30-60%"
            else:
                time_increase = "60-100%+"
            
            logger.info(f"\n{scenario['name']}:")
            logger.info(f"Predicted Traffic Volume: {prediction:.2f} vehicles/hour")
            logger.info(f"Congestion Level: {congestion_level}")
            logger.info(f"Estimated Travel Time Increase: {time_increase}")
            
        return True
        
    except Exception as e:
        import traceback
        logger.error(f"Error in test prediction: {str(e)}")
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    logger.info("Training traffic flow prediction model...")
    
    # Force fresh dataset and model
    data_path = 'traffic_flow_data.csv'
    if os.path.exists(data_path):
        logger.info(f"Removing existing dataset at {data_path}")
        os.remove(data_path)
    
    model_path = 'core/models/traffic_flow_model.pkl'
    if os.path.exists(model_path):
        logger.info(f"Removing existing model at {model_path}")
        os.remove(model_path)
    
    success = train_traffic_model(degree=3)
    
    if success:
        logger.info("Traffic flow model training completed successfully")
    else:
        logger.error("Traffic flow model training failed") 