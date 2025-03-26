import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import os
import logging
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_sample_dataset():
    """Create a synthetic dataset for house price prediction"""
    np.random.seed(42)
    n_samples = 1000
    
    # Generate features
    square_footage = np.random.normal(1800, 600, n_samples).astype(int)
    square_footage = np.clip(square_footage, 600, 5000)
    
    bedrooms = np.random.randint(1, 7, n_samples)
    bathrooms = np.random.choice([1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5], n_samples)
    
    current_year = 2023
    age = np.random.exponential(30, n_samples).astype(int)
    age = np.clip(age, 0, 120)
    year_built = current_year - age
    
    # Create zip codes with price correlation
    zip_codes = []
    zip_base = ['9021', '9001', '9010', '9030', '9040']
    for i in range(n_samples):
        # Higher square footage tends to be in more expensive zip codes
        zip_idx = min(int(square_footage[i] / 1200), len(zip_base) - 1)
        zip_codes.append(f"{zip_base[zip_idx]}0")
    
    neighborhood_score = np.random.randint(1, 11, n_samples)
    
    # Generate house prices based on features
    base_price = 200000
    sqft_factor = square_footage * 150
    bedroom_factor = bedrooms * 20000
    bathroom_factor = bathrooms * 25000
    age_factor = -age * 1000
    neighborhood_factor = neighborhood_score * 15000
    
    # Add zip code effect
    zip_factor = np.zeros(n_samples)
    for i, zip_code in enumerate(zip_codes):
        if zip_code == '90210':
            zip_factor[i] = 300000
        elif zip_code == '90010':
            zip_factor[i] = 100000
        elif zip_code == '90100':
            zip_factor[i] = 200000
        elif zip_code == '90300':
            zip_factor[i] = 150000
        else:
            zip_factor[i] = 50000
    
    # Calculate prices
    house_prices = (
        base_price + 
        sqft_factor + 
        bedroom_factor + 
        bathroom_factor + 
        age_factor + 
        zip_factor + 
        neighborhood_factor
    )
    
    # Add random variation
    noise = np.random.normal(0, 50000, n_samples)
    house_prices += noise
    house_prices = np.maximum(house_prices, 50000)  # Ensure no negative prices
    
    # Create DataFrame
    df = pd.DataFrame({
        'square_footage': square_footage,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'year_built': year_built,
        'zip_code': zip_codes,
        'neighborhood_score': neighborhood_score,
        'price': house_prices.astype(int)
    })
    
    return df

def train_and_save_knn_model():
    """Train a KNN model for house price prediction and save it"""
    try:
        # Create or load dataset
        data_path = 'house_price_data.csv'
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
            logger.info(f"Loaded existing dataset from {data_path}")
        else:
            df = create_sample_dataset()
            df.to_csv(data_path, index=False)
            logger.info(f"Created new dataset and saved to {data_path}")
        
        logger.info(f"Dataset shape: {df.shape}")
        logger.info(f"Dataset sample:\n{df.head()}")
        
        # Prepare data
        # Create a label encoder for zip codes
        zip_codes = df['zip_code'].unique()
        zip_dict = {zip_code: i for i, zip_code in enumerate(zip_codes)}
        
        # Convert zip codes to numerical
        df['zip_code_encoded'] = df['zip_code'].map(zip_dict)
        
        # Features and target
        X = df[['square_footage', 'bedrooms', 'bathrooms', 'year_built', 
                'zip_code_encoded', 'neighborhood_score']]
        y = df['price']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Find optimal k
        k_values = range(1, 21)
        test_rmse = []
        
        for k in k_values:
            knn = KNeighborsRegressor(n_neighbors=k)
            knn.fit(X_train_scaled, y_train)
            y_pred = knn.predict(X_test_scaled)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            test_rmse.append(rmse)
            logger.info(f"k={k}, RMSE=${rmse:.2f}")
        
        # Find the best k (balance between underfitting and overfitting)
        best_k = k_values[np.argmin(test_rmse)]
        logger.info(f"Best k: {best_k}")
        
        # Train final model with best k
        knn = KNeighborsRegressor(n_neighbors=best_k)
        knn.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = knn.predict(X_test_scaled)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        logger.info(f"Final Model - RMSE: ${rmse:.2f}, R²: {r2:.4f}")
        
        # Plot RMSE vs k
        plt.figure(figsize=(10, 6))
        plt.plot(k_values, test_rmse, marker='o')
        plt.xlabel('Number of Neighbors (k)')
        plt.ylabel('RMSE')
        plt.title('RMSE vs k for KNN')
        plt.grid(True)
        
        # Save plot
        models_dir = 'core/models'
        os.makedirs(models_dir, exist_ok=True)
        plt.savefig(os.path.join(models_dir, 'knn_k_selection.png'))
        
        # Save the model and scaler
        model_path = os.path.join(models_dir, 'knn_house_price_model.pkl')
        scaler_path = os.path.join(models_dir, 'knn_scaler.pkl')
        zip_dict_path = os.path.join(models_dir, 'knn_zip_dict.pkl')
        
        with open(model_path, 'wb') as f:
            pickle.dump(knn, f)
        
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        
        with open(zip_dict_path, 'wb') as f:
            pickle.dump(zip_dict, f)
        
        logger.info(f"Model and preprocessing objects saved successfully")
        
        # Test model with sample input
        test_knn_prediction()
        
        return True
    
    except Exception as e:
        import traceback
        logger.error(f"Error in training process: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def test_knn_prediction():
    """Test the trained KNN model with sample inputs"""
    try:
        # Load model and preprocessing objects
        models_dir = 'core/models'
        model_path = os.path.join(models_dir, 'knn_house_price_model.pkl')
        scaler_path = os.path.join(models_dir, 'knn_scaler.pkl')
        zip_dict_path = os.path.join(models_dir, 'knn_zip_dict.pkl')
        
        with open(model_path, 'rb') as f:
            knn = pickle.load(f)
        
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        with open(zip_dict_path, 'rb') as f:
            zip_dict = pickle.load(f)
        
        # Test scenarios
        test_houses = [
            {
                'name': 'Small Starter Home',
                'square_footage': 1200,
                'bedrooms': 2,
                'bathrooms': 1,
                'year_built': 1980,
                'zip_code': '90010',
                'neighborhood_score': 6
            },
            {
                'name': 'Luxury Estate',
                'square_footage': 4000,
                'bedrooms': 5,
                'bathrooms': 4.5,
                'year_built': 2015,
                'zip_code': '90210',
                'neighborhood_score': 10
            },
            {
                'name': 'Family Home',
                'square_footage': 2200,
                'bedrooms': 4,
                'bathrooms': 2.5,
                'year_built': 2000,
                'zip_code': '90100',
                'neighborhood_score': 8
            }
        ]
        
        logger.info("Testing house price predictions with sample houses:")
        
        for house in test_houses:
            # Prepare input
            try:
                zip_encoded = zip_dict.get(house['zip_code'], 0)  # Default to 0 if zip not found
            except:
                zip_encoded = 0
                
            input_data = np.array([
                [
                    house['square_footage'],
                    house['bedrooms'],
                    house['bathrooms'],
                    house['year_built'],
                    zip_encoded,
                    house['neighborhood_score']
                ]
            ])
            
            # Scale input
            input_scaled = scaler.transform(input_data)
            
            # Make prediction
            prediction = knn.predict(input_scaled)[0]
            
            # Log results
            logger.info(f"\n{house['name']}:")
            logger.info(f"Square Footage: {house['square_footage']}")
            logger.info(f"Bedrooms: {house['bedrooms']}")
            logger.info(f"Bathrooms: {house['bathrooms']}")
            logger.info(f"Year Built: {house['year_built']}")
            logger.info(f"ZIP Code: {house['zip_code']}")
            logger.info(f"Neighborhood Score: {house['neighborhood_score']}")
            logger.info(f"Predicted Price: ${prediction:,.2f}")
            logger.info(f"Price per Square Foot: ${prediction/house['square_footage']:,.2f}")
        
        return True
    
    except Exception as e:
        logger.error(f"Error testing predictions: {str(e)}")
        return False

if __name__ == "__main__":
    logger.info("Training KNN house price prediction model...")
    success = train_and_save_knn_model()
    if success:
        logger.info("KNN house price model training completed successfully")
    else:
        logger.error("KNN house price model training failed") 