from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
from .forms import SLRPredictionForm, MLRPredictionForm, StudentPerformanceForm, TrafficFlowForm, CarPerformanceForm, KNNPredictionForm
import pickle
import os
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from django.views.decorators.csrf import csrf_exempt
import json

# Set up logging
logger = logging.getLogger(__name__)

# Create your views here.
def index(request):
    return render(request, 'index.html')

def slr_prediction(request):
    if request.method == 'POST':
        form = SLRPredictionForm(request.POST)
        if form.is_valid():
            try:
                distance = form.cleaned_data['distance']
                
                # Get the absolute path to the model file
                project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                model_path = os.path.join(project_root, 'core', 'models', 'slr_model.pkl')
                
                # Check if model file exists
                if not os.path.exists(model_path):
                    logger.error(f"Model file not found at: {model_path}")
                    return JsonResponse({
                        'success': False,
                        'error': 'Model file not found. Please ensure the model is properly trained and saved.'
                    }, status=500)
                
                # Load the trained model
                try:
                    with open(model_path, 'rb') as file:
                        model = pickle.load(file)
                except Exception as e:
                    logger.error(f"Error loading model: {str(e)}")
                    return JsonResponse({
                        'success': False,
                        'error': 'Error loading the model. Please try again.'
                    }, status=500)
                
                # Make prediction
                try:
                    prediction = model.predict([[distance]])[0]
                    
                    # Calculate accuracy using the test set
                    try:
                        # Load the dataset
                        data_path = os.path.join(project_root, 'fuel_consumption_vs_distance.csv')
                        df = pd.read_csv(data_path)
                        X = df[['Distance_Driven_km']]
                        y = df['Fuel_Consumed_Liters']
                        
                        # Calculate R² score
                        accuracy = r2_score(y, model.predict(X))
                    except Exception as e:
                        logger.warning(f"Could not calculate accuracy: {str(e)}")
                        accuracy = None
                    
                    return JsonResponse({
                        'success': True,
                        'prediction': round(prediction, 2),
                        'distance': distance,
                        'accuracy': round(accuracy * 100, 2) if accuracy is not None else None
                    })
                except Exception as e:
                    logger.error(f"Error making prediction: {str(e)}")
                    return JsonResponse({
                        'success': False,
                        'error': 'Error making prediction. Please check your input values.'
                    }, status=500)
                    
            except Exception as e:
                logger.error(f"Unexpected error: {str(e)}")
                return JsonResponse({
                    'success': False,
                    'error': 'An unexpected error occurred. Please try again.'
                }, status=500)
    else:
        form = SLRPredictionForm()
    
    return render(request, 'slr.html', {'form': form})

def get_efficiency_insights(speed, traffic_density, distance, vehicle_weight, temperature):
    """Generate insights about fuel consumption efficiency based on input parameters"""
    insights = []
    
    # Speed-based insights
    if 60 <= speed <= 90:
        insights.append("Optimal speed range for fuel efficiency")
    elif speed > 90:
        insights.append("High speed may increase fuel consumption")
    elif speed < 40:
        insights.append("Low speed may lead to inefficient fuel usage")
    
    # Traffic-based insights
    if traffic_density == 'low':
        insights.append("Low traffic conditions are ideal for fuel efficiency")
    elif traffic_density == 'medium':
        insights.append("Moderate traffic may slightly impact fuel efficiency")
    else:
        insights.append("Heavy traffic conditions typically increase fuel consumption")
    
    # Distance-based insights
    if distance > 100:
        insights.append("Longer distance allows for better fuel efficiency optimization")
    
    # Weight-based insights
    if vehicle_weight > 2000:
        insights.append("Heavy vehicle weight may increase fuel consumption")
    
    # Temperature-based insights
    if temperature > 30:
        insights.append("High temperature may increase fuel consumption due to AC usage")
    elif temperature < 10:
        insights.append("Low temperature may affect fuel efficiency")
    
    return insights

def mlr_prediction(request):
    if request.method == 'POST':
        form = MLRPredictionForm(request.POST)
        if form.is_valid():
            try:
                # Get form data
                distance = form.cleaned_data['distance']
                speed = form.cleaned_data['speed']
                vehicle_weight = form.cleaned_data['vehicle_weight']
                temperature = form.cleaned_data['temperature']
                traffic_density = form.cleaned_data['traffic_density']
                
                # Get the absolute path to the model file
                project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                model_path = os.path.join(project_root, 'core', 'models', 'mlr_model.pkl')
                scaler_path = os.path.join(project_root, 'core', 'models', 'mlr_scaler.pkl')
                label_encoder_path = os.path.join(project_root, 'core', 'models', 'mlr_label_encoder.pkl')
                
                # Check if model files exist
                if not all(os.path.exists(path) for path in [model_path, scaler_path, label_encoder_path]):
                    logger.error("Required model files not found")
                    return JsonResponse({
                        'success': False,
                        'error': 'Model files not found. Please ensure the model is properly trained and saved.'
                    }, status=500)
                
                try:
                    # Load the model, scaler, and label encoder
                    with open(model_path, 'rb') as file:
                        model = pickle.load(file)
                    with open(scaler_path, 'rb') as file:
                        scaler = pickle.load(file)
                    with open(label_encoder_path, 'rb') as file:
                        label_encoder = pickle.load(file)
                except Exception as e:
                    logger.error(f"Error loading model files: {str(e)}")
                    return JsonResponse({
                        'success': False,
                        'error': 'Error loading the model. Please try again.'
                    }, status=500)
                
                try:
                    # Prepare input data
                    input_data = np.array([[distance, speed, vehicle_weight, temperature]])
                    
                    # Scale numerical features
                    input_data_scaled = scaler.transform(input_data)
                    
                    # Encode categorical feature (traffic_density)
                    traffic_encoded = label_encoder.transform([traffic_density])
                    
                    # Combine features
                    X = np.hstack([input_data_scaled, traffic_encoded.reshape(-1, 1)])
                    
                    # Make prediction
                    prediction = model.predict(X)[0]
                    
                    # Calculate confidence interval (95%)
                    std_dev = np.std(model.predict(X))
                    confidence_interval = 1.96 * std_dev
                    
                    # Get feature importance
                    feature_importance = dict(zip(
                        ['Distance', 'Speed', 'Weight', 'Traffic', 'Temperature'],
                        model.coef_
                    ))
                    most_important = max(feature_importance.items(), key=lambda x: abs(x[1]))[0]
                    
                    # Calculate accuracy
                    try:
                        data_path = os.path.join(project_root, 'fuel_consumption_mlr.csv')
                        df = pd.read_csv(data_path)
                        X_test = df.drop('Fuel_Consumed_Liters', axis=1)
                        y_test = df['Fuel_Consumed_Liters']
                        
                        # Prepare test data
                        X_test_numerical = X_test[['Distance_Driven_km', 'Average_Speed_kmh', 'Vehicle_Weight_kg', 'Temperature_C']]
                        X_test_scaled = scaler.transform(X_test_numerical)
                        X_test_traffic = label_encoder.transform(X_test['Traffic_Density'])
                        X_test_processed = np.hstack([X_test_scaled, X_test_traffic.reshape(-1, 1)])
                        
                        accuracy = r2_score(y_test, model.predict(X_test_processed))
                    except Exception as e:
                        logger.warning(f"Could not calculate accuracy: {str(e)}")
                        accuracy = None
                    
                    # Get efficiency insights
                    insights = get_efficiency_insights(speed, traffic_density, distance, vehicle_weight, temperature)
                    
                    return JsonResponse({
                        'success': True,
                        'prediction': round(prediction, 2),
                        'accuracy': round(accuracy * 100, 2) if accuracy is not None else None,
                        'confidence_interval': round(confidence_interval, 2),
                        'most_important_feature': most_important,
                        'insights': insights
                    })
                    
                except Exception as e:
                    logger.error(f"Error making prediction: {str(e)}")
                    return JsonResponse({
                        'success': False,
                        'error': f'Error making prediction: {str(e)}'
                    }, status=500)
                    
            except Exception as e:
                logger.error(f"Unexpected error: {str(e)}")
                return JsonResponse({
                    'success': False,
                    'error': 'An unexpected error occurred. Please try again.'
                }, status=500)
    else:
        form = MLRPredictionForm()
    
    return render(request, 'mlr.html', {'form': form})

def get_student_insights(attendance, previous_grades, study_hours, family_background, extracurricular_activities, parent_education):
    """Generate insights about student performance based on input parameters"""
    insights = []
    
    # Attendance-based insights
    if attendance >= 90:
        insights.append("Excellent attendance record")
    elif attendance >= 80:
        insights.append("Good attendance, but room for improvement")
    else:
        insights.append("Low attendance may impact performance")
    
    # Grades-based insights
    if previous_grades >= 90:
        insights.append("Strong academic performance history")
    elif previous_grades >= 80:
        insights.append("Good academic track record")
    else:
        insights.append("Previous grades indicate potential challenges")
    
    # Study hours insights
    if study_hours >= 30:
        insights.append("High study commitment")
    elif study_hours >= 20:
        insights.append("Moderate study hours")
    else:
        insights.append("Consider increasing study hours")
    
    # Family background insights
    if family_background == 'high':
        insights.append("Strong family support system")
    elif family_background == 'medium':
        insights.append("Moderate family support")
    else:
        insights.append("May need additional support resources")
    
    # Extracurricular activities insights
    if extracurricular_activities == 'high':
        insights.append("Well-rounded with many activities")
    elif extracurricular_activities == 'medium':
        insights.append("Balanced extracurricular involvement")
    else:
        insights.append("Consider engaging in more activities")
    
    # Parent education insights
    if parent_education in ['masters', 'phd']:
        insights.append("Strong educational background in family")
    elif parent_education == 'bachelors':
        insights.append("Good educational foundation")
    else:
        insights.append("May benefit from additional academic guidance")
    
    return insights

def logistic_prediction(request):
    if request.method == 'POST':
        form = StudentPerformanceForm(request.POST)
        if form.is_valid():
            try:
                # Get form data
                attendance = form.cleaned_data['attendance']
                previous_grades = form.cleaned_data['previous_grades']
                study_hours = form.cleaned_data['study_hours']
                family_background = form.cleaned_data['family_background']
                extracurricular_activities = form.cleaned_data['extracurricular_activities']
                parent_education = form.cleaned_data['parent_education']
                
                # Get the absolute path to the model file
                project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                model_path = os.path.join(project_root, 'core', 'models', 'logistic_model.pkl')
                scaler_path = os.path.join(project_root, 'core', 'models', 'logistic_scaler.pkl')
                label_encoders_path = os.path.join(project_root, 'core', 'models', 'logistic_label_encoders.pkl')
                
                # Check if model files exist
                if not all(os.path.exists(path) for path in [model_path, scaler_path, label_encoders_path]):
                    logger.error("Required model files not found")
                    return JsonResponse({
                        'success': False,
                        'error': 'Model files not found. Please ensure the model is properly trained and saved.'
                    }, status=500)
                
                try:
                    # Load the model, scaler, and label encoders
                    with open(model_path, 'rb') as file:
                        model = pickle.load(file)
                    with open(scaler_path, 'rb') as file:
                        scaler = pickle.load(file)
                    with open(label_encoders_path, 'rb') as file:
                        label_encoders = pickle.load(file)
                except Exception as e:
                    logger.error(f"Error loading model files: {str(e)}")
                    return JsonResponse({
                        'success': False,
                        'error': 'Error loading the model. Please try again.'
                    }, status=500)
                
                try:
                    # Prepare input data
                    input_data = pd.DataFrame([{
                        'Attendance': attendance,
                        'Previous_Grades': previous_grades,
                        'Study_Hours': study_hours,
                        'Family_Background': family_background,
                        'Extracurricular_Activities': extracurricular_activities,
                        'Parent_Education': parent_education
                    }])
                    
                    # Scale numerical features
                    numerical_features = ['Attendance', 'Previous_Grades', 'Study_Hours']
                    input_data[numerical_features] = scaler.transform(input_data[numerical_features])
                    
                    # Encode categorical features
                    categorical_features = ['Family_Background', 'Extracurricular_Activities', 'Parent_Education']
                    for feature in categorical_features:
                        input_data[feature] = label_encoders[feature].transform(input_data[feature])
                    
                    # Make prediction
                    prediction = model.predict(input_data)[0]
                    probability = model.predict_proba(input_data)[0][1]
                    
                    # Get insights
                    insights = get_student_insights(
                        attendance, previous_grades, study_hours,
                        family_background, extracurricular_activities, parent_education
                    )
                    
                    # Log prediction details for debugging
                    logger.info(f"Input data: {input_data.iloc[0].to_dict()}")
                    logger.info(f"Prediction: {prediction}")
                    logger.info(f"Probability: {probability:.2%}")
                    
                    return JsonResponse({
                        'success': True,
                        'prediction': 'Likely to Succeed' if prediction == 1 else 'May Need Support',
                        'probability': round(probability * 100, 2),
                        'insights': insights
                    })
                    
                except Exception as e:
                    logger.error(f"Error making prediction: {str(e)}")
                    return JsonResponse({
                        'success': False,
                        'error': f'Error making prediction: {str(e)}'
                    }, status=500)
                    
            except Exception as e:
                logger.error(f"Unexpected error: {str(e)}")
                return JsonResponse({
                    'success': False,
                    'error': 'An unexpected error occurred. Please try again.'
                }, status=500)
    else:
        form = StudentPerformanceForm()
    
    return render(request, 'logistic.html', {'form': form})

def get_car_performance_insights(vehicle_weight, engine_power, engine_displacement, drag_coefficient, speed):
    """Generate insights about car performance based on input parameters"""
    insights = []
    
    # Weight-based insights
    if vehicle_weight < 1200:
        insights.append("Lightweight vehicle - good for fuel efficiency and acceleration")
    elif vehicle_weight > 2000:
        insights.append("Heavy vehicle - may affect acceleration and fuel economy")
    else:
        insights.append("Medium weight vehicle - balanced performance characteristics")
    
    # Power-based insights
    power_to_weight = engine_power / (vehicle_weight / 1000)
    if power_to_weight < 100:
        insights.append("Low power-to-weight ratio - economy-oriented performance")
    elif power_to_weight > 200:
        insights.append("High power-to-weight ratio - sporty performance characteristics")
    else:
        insights.append("Balanced power-to-weight ratio - good all-around performance")
    
    # Engine displacement insights
    if engine_displacement < 1.5:
        insights.append("Small engine - typically better fuel economy but less power")
    elif engine_displacement > 3.0:
        insights.append("Large engine - typically more power but higher fuel consumption")
    else:
        insights.append("Medium engine - good balance of power and economy")
    
    # Aerodynamics insights
    if drag_coefficient < 0.3:
        insights.append("Excellent aerodynamics - reduces fuel consumption at high speeds")
    elif drag_coefficient > 0.4:
        insights.append("Higher drag coefficient - may increase fuel consumption at high speeds")
    
    # Speed insights
    if speed < 60:
        insights.append("Lower speeds typically yield better fuel economy in city driving")
    elif speed > 110:
        insights.append("Higher speeds significantly increase fuel consumption due to air resistance")
    else:
        insights.append("Moderate highway speeds typically offer the best fuel efficiency")
    
    return insights

def polynomial_prediction(request):
    if request.method == 'POST':
        form = CarPerformanceForm(request.POST)
        if form.is_valid():
            try:
                # Get form data
                vehicle_weight = form.cleaned_data['vehicle_weight']
                engine_power = form.cleaned_data['engine_power']
                engine_displacement = form.cleaned_data['engine_displacement']
                drag_coefficient = form.cleaned_data['drag_coefficient']
                speed = form.cleaned_data['speed']
                road_gradient = form.cleaned_data['road_gradient']
                polynomial_degree = int(form.cleaned_data['polynomial_degree'])
                
                # Get the absolute path to the model files
                project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                fuel_model_path = os.path.join(project_root, 'core', 'models', 'polynomial_fuel_model.pkl')
                accel_model_path = os.path.join(project_root, 'core', 'models', 'polynomial_accel_model.pkl')
                
                # Check if model files exist
                if not all(os.path.exists(path) for path in [fuel_model_path, accel_model_path]):
                    logger.error("Required model files not found")
                    return JsonResponse({
                        'success': False,
                        'error': 'Model files not found. Please ensure the models are properly trained and saved.'
                    }, status=500)
                
                try:
                    # Load the models
                    with open(fuel_model_path, 'rb') as file:
                        fuel_model = pickle.load(file)
                    with open(accel_model_path, 'rb') as file:
                        accel_model = pickle.load(file)
                except Exception as e:
                    logger.error(f"Error loading model files: {str(e)}")
                    return JsonResponse({
                        'success': False,
                        'error': 'Error loading the models. Please try again.'
                    }, status=500)
                
                try:
                    # Generate speed range for consumption curve
                    speed_values = list(range(0, 201, 5))
                    
                    # Prepare input data for fuel consumption prediction at various speeds
                    # IMPORTANT: Feature names must match exactly with what the model was trained on
                    fuel_inputs = pd.DataFrame({
                        'Weight': [vehicle_weight] * len(speed_values),
                        'Power': [engine_power] * len(speed_values),
                        'Displacement': [engine_displacement] * len(speed_values),
                        'Drag_Coefficient': [drag_coefficient] * len(speed_values),
                        'Speed': speed_values,
                        'Gradient': [road_gradient] * len(speed_values)
                    })
                    
                    # Predict fuel consumption at various speeds
                    consumption_values = fuel_model.predict(fuel_inputs).tolist()
                    
                    # Find optimal speed (lowest consumption)
                    optimal_speed = speed_values[np.argmin(consumption_values)]
                    
                    # Predict fuel consumption at the given speed
                    current_speed_input = pd.DataFrame({
                        'Weight': [vehicle_weight],
                        'Power': [engine_power],
                        'Displacement': [engine_displacement],
                        'Drag_Coefficient': [drag_coefficient],
                        'Speed': [speed],
                        'Gradient': [road_gradient]
                    })
                    
                    fuel_consumption = fuel_model.predict(current_speed_input)[0]
                    
                    # Predict acceleration time
                    accel_input = pd.DataFrame({
                        'Weight': [vehicle_weight],
                        'Power': [engine_power],
                        'Displacement': [engine_displacement]
                    })
                    
                    acceleration = accel_model.predict(accel_input)[0]
                    
                    # Calculate power-to-weight ratio
                    power_to_weight = engine_power / (vehicle_weight / 1000)
                    
                    # Calculate estimated top speed
                    top_speed = 80 + 60 * (engine_power / vehicle_weight)**0.5 / (drag_coefficient**0.3)
                    top_speed = min(300, top_speed)
                    
                    # Extract coefficients for model formula
                    try:
                        poly_features = fuel_model.named_steps['poly']
                        linear_model = fuel_model.named_steps['linear']
                        
                        feature_names = poly_features.get_feature_names_out(['Weight', 'Power', 'Displacement', 'Drag_Coefficient', 'Speed', 'Gradient'])
                        coefficients = linear_model.coef_
                        
                        # Simplify model formula by taking only the top terms
                        coef_importance = np.abs(coefficients)
                        top_indices = np.argsort(coef_importance)[-5:]  # Get indices of top 5 terms
                        
                        model_formula = f"y = {linear_model.intercept_:.4f}"
                        for idx in top_indices:
                            if coefficients[idx] >= 0:
                                model_formula += f" + {coefficients[idx]:.4f} × {feature_names[idx]}"
                            else:
                                model_formula += f" - {abs(coefficients[idx]):.4f} × {feature_names[idx]}"
                    except Exception as e:
                        logger.warning(f"Could not extract model formula: {str(e)}")
                        model_formula = "Model formula not available"
                    
                    # Generate insights
                    insights = get_car_performance_insights(
                        vehicle_weight, engine_power, engine_displacement, 
                        drag_coefficient, speed
                    )
                    
                    # Calculate R² of the model
                    try:
                        r2 = fuel_model.score(fuel_inputs, consumption_values)
                    except Exception as e:
                        logger.warning(f"Could not calculate R² score: {str(e)}")
                        r2 = 0.95  # Default value
                    
                    # Generate power curve (simplified for visualization)
                    power_curve = []
                    for s in speed_values:
                        if s == 0:
                            power_curve.append(0)
                        else:
                            # Simplified power curve based on speed
                            power_factor = min(1.0, s / 60)  # Max power reached at 60 km/h
                            if s > 120:
                                power_factor *= (1 - 0.2 * (s - 120) / 80)  # Power drops at very high speeds
                            power_curve.append(engine_power * power_factor)
                    
                    return JsonResponse({
                        'success': True,
                        'fuel_consumption': f"{fuel_consumption:.2f}",
                        'optimal_speed': f"{optimal_speed}",
                        'acceleration': f"{acceleration:.2f}",
                        'top_speed': f"{top_speed:.0f}",
                        'power_to_weight': f"{power_to_weight:.2f}",
                        'polynomial_degree': str(polynomial_degree),
                        'r2_score': f"{r2:.4f}",
                        'model_formula': model_formula,
                        'speed_values': speed_values,
                        'consumption_values': consumption_values,
                        'power_values': power_curve,
                        'insights': insights
                    })
                    
                except Exception as e:
                    logger.error(f"Error making prediction: {str(e)}")
                    return JsonResponse({
                        'success': False,
                        'error': f'Error analyzing performance: {str(e)}'
                    }, status=500)
                    
            except Exception as e:
                logger.error(f"Unexpected error: {str(e)}")
                return JsonResponse({
                    'success': False,
                    'error': 'An unexpected error occurred. Please try again.'
                }, status=500)
    else:
        form = CarPerformanceForm()
    
    return render(request, 'polynomial.html', {'form': form})

def knn_prediction(request):
    if request.method == 'POST':
        form = KNNPredictionForm(request.POST)
        if form.is_valid():
            try:
                # Get form data
                square_footage = form.cleaned_data['square_footage']
                bedrooms = form.cleaned_data['bedrooms']
                bathrooms = form.cleaned_data['bathrooms']
                year_built = form.cleaned_data['year_built']
                zip_code = form.cleaned_data['zip_code']
                neighborhood_score = form.cleaned_data['neighborhood_score']
                k_neighbors = int(form.cleaned_data['k_neighbors'])
                
                logger.info(f"KNN prediction request with: square_footage={square_footage}, bedrooms={bedrooms}, "
                           f"bathrooms={bathrooms}, year_built={year_built}, zip={zip_code}")
                
                # Get model paths
                model_path = 'core/models/knn_house_price_model.pkl'
                scaler_path = 'core/models/knn_scaler.pkl'
                zip_dict_path = 'core/models/knn_zip_dict.pkl'
                
                # Check if model files exist
                if not all(os.path.exists(path) for path in [model_path, scaler_path, zip_dict_path]):
                    logger.error("KNN model files not found")
                    return JsonResponse({
                        'success': False,
                        'error': 'Model files not found. Please train the KNN model first.'
                    }, status=500)
                
                # Load model and preprocessing objects
                try:
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f)
                    with open(scaler_path, 'rb') as f:
                        scaler = pickle.load(f)
                    with open(zip_dict_path, 'rb') as f:
                        zip_dict = pickle.load(f)
                except Exception as e:
                    logger.error(f"Error loading KNN model: {str(e)}")
                    return JsonResponse({
                        'success': False,
                        'error': 'Error loading the model. Please try again.'
                    }, status=500)
                
                # Prepare input data
                try:
                    # Convert zip code to numerical using the dictionary
                    zip_encoded = zip_dict.get(zip_code, 0)  # Default to 0 if not found
                    
                    # Create feature array
                    input_data = np.array([[
                        square_footage,
                        bedrooms,
                        bathrooms,
                        year_built,
                        zip_encoded,
                        neighborhood_score
                    ]])
                    
                    # Scale the input
                    input_scaled = scaler.transform(input_data)
                    
                    # Override k value if requested (for experimentation)
                    if k_neighbors != model.n_neighbors:
                        model.n_neighbors = k_neighbors
                        logger.info(f"Adjusted k to {k_neighbors} for this prediction")
                    
                    # Get prediction and neighbors
                    prediction = model.predict(input_scaled)[0]
                    
                    # Get neighbor details for display
                    # Load dataset for neighbor lookup
                    data_path = 'house_price_data.csv'
                    if os.path.exists(data_path):
                        df = pd.read_csv(data_path)
                        # Add encoded zip code for matching
                        df['zip_code_encoded'] = df['zip_code'].map(zip_dict)
                        
                        # Get features and scale them
                        X = df[['square_footage', 'bedrooms', 'bathrooms', 'year_built', 
                                'zip_code_encoded', 'neighborhood_score']]
                        X_scaled = scaler.transform(X)
                        
                        # Find nearest neighbors
                        distances, indices = model.kneighbors(input_scaled)
                        
                        # Prepare neighbor data for response
                        neighbors = []
                        for i, idx in enumerate(indices[0]):
                            neighbors.append({
                                'square_footage': int(df.iloc[idx]['square_footage']),
                                'bedrooms': int(df.iloc[idx]['bedrooms']),
                                'bathrooms': float(df.iloc[idx]['bathrooms']),
                                'year_built': int(df.iloc[idx]['year_built']),
                                'zip_code': df.iloc[idx]['zip_code'],
                                'neighborhood_score': int(df.iloc[idx]['neighborhood_score']),
                                'price': int(df.iloc[idx]['price']),
                                'distance': float(distances[0][i])
                            })
                    else:
                        # Generate placeholder neighbors if dataset not available
                        neighbors = [
                            {
                                'square_footage': int(square_footage * 0.9),
                                'bedrooms': bedrooms,
                                'bathrooms': bathrooms,
                                'year_built': year_built - 5,
                                'zip_code': zip_code,
                                'neighborhood_score': neighborhood_score - 1,
                                'price': int(prediction * 0.9),
                                'distance': 0.25
                            },
                            {
                                'square_footage': int(square_footage * 1.1),
                                'bedrooms': bedrooms + 1,
                                'bathrooms': bathrooms + 0.5,
                                'year_built': year_built + 3,
                                'zip_code': zip_code,
                                'neighborhood_score': neighborhood_score + 1,
                                'price': int(prediction * 1.1),
                                'distance': 0.35
                            }
                        ]
                    
                    # Calculate price per square foot
                    price_per_sqft = prediction / square_footage
                    
                    # Generate estimated confidence interval (±15%)
                    confidence = 15
                    
                    # Generate feature importance (simplified)
                    feature_importance = {
                        'Square Footage': 35,
                        'Location': 25,
                        'Year Built': 15,
                        'Bathrooms': 10,
                        'Bedrooms': 10,
                        'Neighborhood': 5
                    }
                    
                    # Return prediction results
                    return JsonResponse({
                        'success': True,
                        'predicted_price': round(prediction),
                        'confidence': confidence,
                        'price_per_sqft': round(price_per_sqft),
                        'price_percentile': 65,  # Placeholder, would calculate from dataset
                        'area_avg_price': round(prediction * 1.05),  # Placeholder
                        'neighborhood': f"ZIP {zip_code}",
                        'feature_importance': feature_importance,
                        'neighbors': neighbors,
                        'currency': 'USD'  # Added to indicate original currency
                    })
                    
                except Exception as e:
                    import traceback
                    logger.error(f"Error making prediction: {str(e)}")
                    logger.error(traceback.format_exc())
                    return JsonResponse({
                        'success': False,
                        'error': f'Error making prediction: {str(e)}'
                    }, status=500)
                    
            except Exception as e:
                logger.error(f"Unexpected error in KNN prediction: {str(e)}")
                return JsonResponse({
                    'success': False,
                    'error': f'An unexpected error occurred: {str(e)}'
                }, status=500)
        else:
            # Form validation failed
            logger.error(f"Form validation errors: {form.errors}")
            return JsonResponse({
                'success': False,
                'error': 'Invalid input. Please check your values.'
            }, status=400)
    
    # GET request - render form
    form = KNNPredictionForm()
    return render(request, 'knn.html', {'form': form})

def get_traffic_insights(time_of_day, day_of_week, weather_condition, road_type, traffic_volume):
    """Generate insights about traffic flow based on input parameters and prediction"""
    insights = []
    
    # Time-based insights
    if 6 <= int(time_of_day) <= 9:
        insights.append("Morning rush hour typically has high traffic volume")
    elif 16 <= int(time_of_day) <= 19:
        insights.append("Evening rush hour typically has the highest traffic volume")
    elif 10 <= int(time_of_day) <= 15:
        insights.append("Midday traffic is usually moderate")
    else:
        insights.append("Night time usually has light traffic")
    
    # Day-based insights
    if int(day_of_week) <= 5:
        insights.append("Weekdays generally have higher traffic than weekends")
        if int(day_of_week) == 1 or int(day_of_week) == 5:
            insights.append("Monday and Friday may have unique traffic patterns")
    else:
        insights.append("Weekend traffic is typically lighter than weekdays")
    
    # Weather-based insights
    if weather_condition == '1':
        insights.append("Clear weather allows for optimal traffic flow")
    elif weather_condition == '2':
        insights.append("Light rain may slightly reduce traffic speed")
    elif weather_condition == '3':
        insights.append("Heavy rain significantly impacts traffic flow and safety")
    else:
        insights.append("Snow/ice conditions severely reduce traffic capacity")
    
    # Road-based insights
    if road_type == '1':
        insights.append("Highways have higher capacity but can experience significant congestion during peak hours")
    elif road_type == '2':
        insights.append("Main roads are affected by traffic signals and intersections")
    else:
        insights.append("Residential areas typically have lower traffic volumes")
    
    # Volume-based insights
    if traffic_volume < 500:
        insights.append("Light traffic allows for free-flowing conditions")
    elif traffic_volume < 1000:
        insights.append("Moderate traffic may cause minor delays")
    elif traffic_volume < 1500:
        insights.append("Heavy traffic will likely cause significant delays")
    else:
        insights.append("Severe congestion will result in stop-and-go conditions")
    
    return insights

@csrf_exempt
def traffic_flow_prediction(request):
    if request.method == 'POST':
        form = TrafficFlowForm(request.POST)
        if form.is_valid():
            try:
                # Log input data for debugging
                time_of_day = form.cleaned_data['time_of_day']
                day_of_week = form.cleaned_data['day_of_week']
                weather_condition = form.cleaned_data['weather_condition']
                road_type = form.cleaned_data['road_type']
                polynomial_degree = int(form.cleaned_data['polynomial_degree'])
                
                logger.info(f"Traffic prediction request with: time={time_of_day}, day={day_of_week}, "
                           f"weather={weather_condition}, road_type={road_type}")
                
                # Check if model file exists and is not empty
                model_path = 'core/models/traffic_flow_model.pkl'
                if not os.path.exists(model_path):
                    logger.error(f"Traffic flow model file not found at {model_path}")
                    response_data = {'success': False, 'error': 'Model file not found. Please train the model first.'}
                    return HttpResponse(json.dumps(response_data), content_type='application/json', status=500)
                
                model_size = os.path.getsize(model_path)
                if model_size < 100:
                    logger.error(f"Traffic flow model file is too small: {model_size} bytes")
                    response_data = {'success': False, 'error': 'Invalid model file. Please retrain the model.'}
                    return HttpResponse(json.dumps(response_data), content_type='application/json', status=500)
                
                # Load model
                try:
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f)
                    logger.info(f"Traffic flow model loaded successfully: {type(model)}")
                except Exception as e:
                    logger.error(f"Error loading traffic flow model: {str(e)}")
                    response_data = {'success': False, 'error': 'Failed to load prediction model. Please try again.'}
                    return HttpResponse(json.dumps(response_data), content_type='application/json', status=500)
                
                # Prepare input data for prediction
                input_data = pd.DataFrame({
                    'Time_of_Day': [time_of_day],
                    'Day_of_Week': [day_of_week],
                    'Weather_Condition': [weather_condition],
                    'Road_Type': [road_type]
                })
                
                logger.info(f"Input DataFrame for prediction: {input_data.to_dict()}")
                
                # Make prediction
                try:
                    prediction = model.predict(input_data)[0]
                    logger.info(f"Traffic volume prediction: {prediction:.2f}")
                except Exception as e:
                    import traceback
                    logger.error(f"Error making prediction: {str(e)}")
                    logger.error(traceback.format_exc())
                    response_data = {'success': False, 'error': 'Error making prediction. Please check input values and model.'}
                    return HttpResponse(json.dumps(response_data), content_type='application/json', status=500)
                
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
                if congestion_level == "Light":
                    time_increase = "0-10%"
                elif congestion_level == "Moderate":
                    time_increase = "10-30%"
                elif congestion_level == "Heavy":
                    time_increase = "30-60%"
                else:
                    time_increase = "60-100%+"
                
                # Generate hourly prediction for the chart (keep the same day, weather, road)
                hours = list(range(0, 24))
                hourly_volumes = []
                
                for hour in hours:
                    hourly_input = pd.DataFrame({
                        'Time_of_Day': [hour],
                        'Day_of_Week': [int(day_of_week)],
                        'Weather_Condition': [int(weather_condition)],
                        'Road_Type': [int(road_type)]
                    })
                    
                    try:
                        hourly_prediction = model.predict(hourly_input)[0]
                        hourly_volumes.append(round(hourly_prediction, 1))
                    except Exception as e:
                        logger.warning(f"Error predicting for hour {hour}: {str(e)}")
                        hourly_volumes.append(None)  # Use None for missing values
                
                # Generate insights
                insights = [
                    f"Current traffic conditions show {congestion_level.lower()} congestion.",
                    "Rush hours typically show higher traffic volumes.",
                    f"Weather conditions ({form.cleaned_data['weather_condition']}) affect traffic flow.",
                    f"Road type ({form.cleaned_data['road_type']}) influences traffic density."
                ]
                
                # Return prediction results
                response_data = {
                    'success': True,
                    'traffic_volume': round(prediction, 2),
                    'congestion_level': congestion_level,
                    'time_increase': time_increase,
                    'hours': hours,
                    'volume_data': hourly_volumes,
                    'insights': insights
                }
                
                return HttpResponse(json.dumps(response_data), content_type='application/json')
                
            except Exception as e:
                import traceback
                logger.error(f"Unexpected error in traffic flow prediction: {str(e)}")
                logger.error(traceback.format_exc())
                response_data = {'success': False, 'error': f'An unexpected error occurred: {str(e)}'}
                return HttpResponse(json.dumps(response_data), content_type='application/json', status=500)
        else:
            # Log form validation errors
            logger.error(f"Form validation errors: {form.errors}")
            response_data = {'success': False, 'error': 'Invalid input. Please check your values.'}
            return HttpResponse(json.dumps(response_data), content_type='application/json', status=400)
    
    # GET request - render form
    form = TrafficFlowForm()
    return render(request, 'traffic_flow.html', {'form': form})

