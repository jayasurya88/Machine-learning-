import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_sample_dataset():
    """Create a sample dataset with more realistic relationships and interactions"""
    np.random.seed(42)
    n_samples = 5000
    
    # Generate features with more realistic distributions
    attendance = np.random.normal(85, 8, n_samples)
    attendance = np.clip(attendance, 0, 100)
    
    previous_grades = np.random.normal(75, 12, n_samples)
    previous_grades = np.clip(previous_grades, 0, 100)
    
    study_hours = np.random.normal(25, 6, n_samples)
    study_hours = np.clip(study_hours, 0, 40)
    
    # Categorical features with weighted probabilities
    family_background = np.random.choice(['low', 'medium', 'high'], n_samples, p=[0.2, 0.5, 0.3])
    extracurricular_activities = np.random.choice(['none', 'low', 'medium', 'high'], n_samples, p=[0.1, 0.3, 0.4, 0.2])
    parent_education = np.random.choice(['high_school', 'bachelors', 'masters', 'phd'], n_samples, p=[0.2, 0.4, 0.3, 0.1])
    
    # Create feature interactions
    attendance_grade_interaction = (attendance * previous_grades / 100) ** 1.2
    study_attendance_interaction = (study_hours * attendance / 100) ** 1.1
    
    # Calculate base probability with balanced weights
    base_prob = (
        0.25 * (attendance / 100) +
        0.25 * (previous_grades / 100) +
        0.15 * (study_hours / 40) +
        0.15 * attendance_grade_interaction +
        0.1 * study_attendance_interaction
    )
    
    # Add categorical feature effects with reduced impact
    for i in range(n_samples):
        if family_background[i] == 'high':
            base_prob[i] += 0.1
        elif family_background[i] == 'medium':
            base_prob[i] += 0.05
            
        if extracurricular_activities[i] == 'high':
            base_prob[i] += 0.05
        elif extracurricular_activities[i] == 'medium':
            base_prob[i] += 0.03
            
        if parent_education[i] in ['masters', 'phd']:
            base_prob[i] += 0.1
        elif parent_education[i] == 'bachelors':
            base_prob[i] += 0.05
    
    # Clip probabilities to [0, 1]
    base_prob = np.clip(base_prob, 0, 1)
    
    # Add noise to create more balanced distribution
    noise = np.random.normal(0, 0.1, n_samples)
    success_prob = np.clip(base_prob + noise, 0, 1)
    
    # Generate binary outcomes with a threshold
    success = (success_prob > 0.5).astype(int)
    
    # Ensure we have both classes in the dataset
    success_rate = np.mean(success)
    if success_rate > 0.8 or success_rate < 0.2:
        # Adjust threshold to get more balanced classes
        threshold = np.percentile(success_prob, 50)
        success = (success_prob > threshold).astype(int)
    
    # Create DataFrame
    data = pd.DataFrame({
        'Attendance': attendance,
        'Previous_Grades': previous_grades,
        'Study_Hours': study_hours,
        'Family_Background': family_background,
        'Extracurricular_Activities': extracurricular_activities,
        'Parent_Education': parent_education,
        'Success': success
    })
    
    return data

def train_and_save_model():
    """Train and save the logistic regression model with optimized parameters"""
    try:
        # Create or load dataset
        data = create_sample_dataset()
        
        # Log dataset information
        logger.info(f"Dataset shape: {data.shape}")
        logger.info(f"Success rate: {data['Success'].mean():.2%}")
        
        # Prepare features and target
        X = data.drop('Success', axis=1)
        y = data['Success']
        
        # Split the data with stratification
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Scale numerical features
        scaler = StandardScaler()
        numerical_features = ['Attendance', 'Previous_Grades', 'Study_Hours']
        X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
        X_test[numerical_features] = scaler.transform(X_test[numerical_features])
        
        # Encode categorical features
        label_encoders = {}
        categorical_features = ['Family_Background', 'Extracurricular_Activities', 'Parent_Education']
        for feature in categorical_features:
            label_encoders[feature] = LabelEncoder()
            X_train[feature] = label_encoders[feature].fit_transform(X_train[feature])
            X_test[feature] = label_encoders[feature].transform(X_test[feature])
        
        # Train model with optimized parameters
        model = LogisticRegression(
            C=0.01,  # Stronger regularization for more confident predictions
            class_weight='balanced',  # Handle class imbalance
            max_iter=3000,  # More iterations for better convergence
            random_state=42,
            solver='lbfgs',  # Use L-BFGS solver for better convergence
            multi_class='auto',  # Automatically handle binary classification
            penalty='l2',  # L2 regularization for better generalization
            tol=1e-5  # Tighter convergence tolerance
        )
        
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate and log metrics
        accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"Model Accuracy: {accuracy:.2%}")
        logger.info("\nClassification Report:")
        logger.info(classification_report(y_test, y_pred))
        
        # Save the model and preprocessors
        project_root = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(project_root, 'core', 'models')
        os.makedirs(models_dir, exist_ok=True)
        
        model_path = os.path.join(models_dir, 'logistic_model.pkl')
        scaler_path = os.path.join(models_dir, 'logistic_scaler.pkl')
        label_encoders_path = os.path.join(models_dir, 'logistic_label_encoders.pkl')
        
        with open(model_path, 'wb') as file:
            pickle.dump(model, file)
        with open(scaler_path, 'wb') as file:
            pickle.dump(scaler, file)
        with open(label_encoders_path, 'wb') as file:
            pickle.dump(label_encoders, file)
        
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Scaler saved to {scaler_path}")
        logger.info(f"Label encoders saved to {label_encoders_path}")
        
        # Log feature importance
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': abs(model.coef_[0])
        }).sort_values('Importance', ascending=False)
        
        logger.info("\nFeature Importance:")
        logger.info(feature_importance)
        
        # Test prediction with a sample input
        sample_input = pd.DataFrame([{
            'Attendance': 75,
            'Previous_Grades': 65,
            'Study_Hours': 15,
            'Family_Background': 'low',
            'Extracurricular_Activities': 'none',
            'Parent_Education': 'high_school'
        }])
        
        # Scale numerical features
        sample_input[numerical_features] = scaler.transform(sample_input[numerical_features])
        
        # Encode categorical features
        for feature in categorical_features:
            sample_input[feature] = label_encoders[feature].transform(sample_input[feature])
        
        # Make prediction
        prediction = model.predict(sample_input)[0]
        probability = model.predict_proba(sample_input)[0][1]
        
        logger.info("\nSample Prediction Test:")
        logger.info(f"Input: {sample_input.iloc[0].to_dict()}")
        logger.info(f"Prediction: {'Success' if prediction == 1 else 'Failure'}")
        logger.info(f"Success Probability: {probability:.2%}")
        
    except Exception as e:
        logger.error(f"Error in training process: {str(e)}")
        raise

if __name__ == "__main__":
    train_and_save_model() 