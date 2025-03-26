from flask import Flask, render_template, request, jsonify
import numpy as np
from slr import SimpleLinearRegression
from mlr import MultipleLinearRegression
from logistic import LogisticRegression
from polynomial import PolynomialRegression
from knn import KNN
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Load and train models
def load_and_train_models():
    # Load datasets
    slr_data = pd.read_csv('slr_data.csv')
    mlr_data = pd.read_csv('mlr_data.csv')
    log_data = pd.read_csv('log_data.csv')
    poly_data = pd.read_csv('poly_data.csv')
    knn_data = pd.read_csv('knn_data.csv')

    # Train SLR
    slr_model = SimpleLinearRegression()
    slr_model.fit(slr_data['X'].values, slr_data['y'].values)

    # Train MLR
    mlr_model = MultipleLinearRegression()
    mlr_model.fit(mlr_data[['X1', 'X2']].values, mlr_data['y'].values)

    # Train Logistic Regression
    log_model = LogisticRegression()
    log_model.fit(log_data[['X1', 'X2']].values, log_data['y'].values)

    # Train Polynomial Regression
    poly_model = PolynomialRegression(degree=2)
    poly_model.fit(poly_data['X'].values.reshape(-1, 1), poly_data['y'].values)

    # Train KNN
    knn_model = KNN(k=3)
    knn_model.fit(knn_data[['X1', 'X2']].values, knn_data['y'].values)

    return slr_model, mlr_model, log_model, poly_model, knn_model

# Initialize models
slr_model, mlr_model, log_model, poly_model, knn_model = load_and_train_models()

def generate_plot(fig):
    img = io.BytesIO()
    fig.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict/slr', methods=['POST'])
def predict_slr():
    data = request.get_json()
    x = float(data['x'])
    
    # Make prediction
    prediction = slr_model.predict(x)
    
    # Generate plot
    plt.figure(figsize=(10, 6))
    plt.scatter(slr_model.X_train, slr_model.y_train, color='blue', label='Training Data')
    plt.plot([x], [prediction], 'ro', label='Prediction')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Simple Linear Regression')
    plt.legend()
    
    plot_data = generate_plot(plt)
    plt.close()
    
    return jsonify({
        'prediction': float(prediction),
        'plot': plot_data
    })

@app.route('/predict/mlr', methods=['POST'])
def predict_mlr():
    data = request.get_json()
    x1 = float(data['x1'])
    x2 = float(data['x2'])
    
    # Make prediction
    prediction = mlr_model.predict(np.array([[x1, x2]]))
    
    # Generate 3D plot
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(mlr_model.X_train[:, 0], mlr_model.X_train[:, 1], mlr_model.y_train, 
              c='blue', label='Training Data')
    ax.scatter([x1], [x2], [prediction], c='red', label='Prediction')
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('y')
    ax.set_title('Multiple Linear Regression')
    ax.legend()
    
    plot_data = generate_plot(plt)
    plt.close()
    
    return jsonify({
        'prediction': float(prediction),
        'plot': plot_data
    })

@app.route('/predict/logistic', methods=['POST'])
def predict_logistic():
    data = request.get_json()
    x1 = float(data['x1'])
    x2 = float(data['x2'])
    
    # Make prediction
    prediction = log_model.predict(np.array([[x1, x2]]))
    probability = log_model.sigmoid(np.dot([x1, x2], log_model.coefficients) + log_model.intercept)
    
    # Generate plot
    plt.figure(figsize=(10, 6))
    plt.scatter(log_model.X_train[log_model.y_train == 0][:, 0], 
                log_model.X_train[log_model.y_train == 0][:, 1], 
                c='blue', label='Class 0')
    plt.scatter(log_model.X_train[log_model.y_train == 1][:, 0], 
                log_model.X_train[log_model.y_train == 1][:, 1], 
                c='red', label='Class 1')
    plt.scatter([x1], [x2], c='green', marker='*', s=200, label='Prediction')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Logistic Regression')
    plt.legend()
    
    plot_data = generate_plot(plt)
    plt.close()
    
    return jsonify({
        'prediction': int(prediction[0]),
        'probability': float(probability),
        'plot': plot_data
    })

@app.route('/predict/polynomial', methods=['POST'])
def predict_polynomial():
    data = request.get_json()
    x = float(data['x'])
    degree = int(data['degree'])
    
    # Update model degree if needed
    if degree != poly_model.degree:
        poly_model.degree = degree
        poly_model.poly_features = PolynomialFeatures(degree=degree)
        poly_model.fit(poly_model.X_train, poly_model.y_train)
    
    # Make prediction
    prediction = poly_model.predict(np.array([[x]]))
    
    # Generate plot
    plt.figure(figsize=(10, 6))
    plt.scatter(poly_model.X_train, poly_model.y_train, color='blue', label='Training Data')
    plt.plot([x], [prediction], 'ro', label='Prediction')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title(f'Polynomial Regression (degree={degree})')
    plt.legend()
    
    plot_data = generate_plot(plt)
    plt.close()
    
    return jsonify({
        'prediction': float(prediction),
        'plot': plot_data
    })

@app.route('/predict/knn', methods=['POST'])
def predict_knn():
    data = request.get_json()
    x1 = float(data['x1'])
    x2 = float(data['x2'])
    k = int(data['k'])
    
    # Update k if needed
    if k != knn_model.k:
        knn_model.k = k
    
    # Make prediction
    prediction = knn_model.predict(np.array([[x1, x2]]))
    
    # Generate plot
    plt.figure(figsize=(10, 6))
    plt.scatter(knn_model.X_train[knn_model.y_train == 0][:, 0], 
                knn_model.X_train[knn_model.y_train == 0][:, 1], 
                c='blue', label='Class 0')
    plt.scatter(knn_model.X_train[knn_model.y_train == 1][:, 0], 
                knn_model.X_train[knn_model.y_train == 1][:, 1], 
                c='red', label='Class 1')
    plt.scatter([x1], [x2], c='green', marker='*', s=200, label='Prediction')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title(f'K-Nearest Neighbors (k={k})')
    plt.legend()
    
    plot_data = generate_plot(plt)
    plt.close()
    
    return jsonify({
        'prediction': int(prediction[0]),
        'plot': plot_data
    })

if __name__ == '__main__':
    app.run(debug=True) 