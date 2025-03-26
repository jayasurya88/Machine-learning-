import numpy as np
import pandas as pd

# Generate sample data for various models
np.random.seed(42)

# For Simple Linear Regression (SLR)
n_samples = 100
X_slr = np.random.rand(n_samples) * 10
y_slr = 2 * X_slr + 1 + np.random.randn(n_samples) * 0.5
slr_data = pd.DataFrame({'X': X_slr, 'y': y_slr})

# For Multiple Linear Regression (MLR)
X1_mlr = np.random.rand(n_samples) * 10
X2_mlr = np.random.rand(n_samples) * 5
y_mlr = 2 * X1_mlr + 1.5 * X2_mlr + 1 + np.random.randn(n_samples) * 0.5
mlr_data = pd.DataFrame({'X1': X1_mlr, 'X2': X2_mlr, 'y': y_mlr})

# For Logistic Regression
X1_log = np.random.randn(n_samples) * 2
X2_log = np.random.randn(n_samples) * 2
y_log = ((X1_log + X2_log) > 0).astype(int)
log_data = pd.DataFrame({'X1': X1_log, 'X2': X2_log, 'y': y_log})

# For Polynomial Regression
X_poly = np.random.rand(n_samples) * 10
y_poly = 0.5 * X_poly**2 + 2 * X_poly + 1 + np.random.randn(n_samples) * 0.5
poly_data = pd.DataFrame({'X': X_poly, 'y': y_poly})

# For KNN
X1_knn = np.random.randn(n_samples) * 2
X2_knn = np.random.randn(n_samples) * 2
y_knn = ((X1_knn**2 + X2_knn**2) < 2).astype(int)
knn_data = pd.DataFrame({'X1': X1_knn, 'X2': X2_knn, 'y': y_knn})

# Save datasets to CSV files
slr_data.to_csv('slr_data.csv', index=False)
mlr_data.to_csv('mlr_data.csv', index=False)
log_data.to_csv('log_data.csv', index=False)
poly_data.to_csv('poly_data.csv', index=False)
knn_data.to_csv('knn_data.csv', index=False)

print("Datasets have been created and saved to CSV files.") 