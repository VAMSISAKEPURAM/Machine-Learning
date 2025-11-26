# 1. Project Initialization
# 1.1 Import Dependencies

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

# Configure visualization settings
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# 2. Data Management
# 2.1 Data Loading


print("Loading dataset...")
df = pd.read_csv('data.csv')
print("Dataset loaded successfully!")
print("\nFirst 5 rows:")
print(df.head())

# 2.2 Data Inspection
# - Check shape, data types
# - View rows
# - Identify missing values

print(f"\nDataset shape: {df.shape}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nMissing values:\n{df.isna().sum()}")

# 2.3 Data Quality Assessment
# - Detect outliers, duplicates
# - Validate consistency
# - Assess completeness

print(f"\nDuplicate rows: {df.duplicated().sum()}")

# Visualize outliers
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
sns.boxplot(data=df, y='y')
plt.title('Boxplot of y')

plt.subplot(1, 2, 2)
sns.histplot(df['y'], kde=True)
plt.title('Distribution of y')
plt.tight_layout()
plt.show()

# 3. Exploratory Data Analysis
# 3.1 Statistical Summary
# - Descriptive statistics
# - Correlations
# - Distribution analysis
# - Skewness identification

print("\nDescriptive Statistics:")
print(df.describe())

# Correlation analysis
corr = df.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
plt.title('Correlation Matrix')
plt.show()

# 3.2 Visualization & Insights
# - Scatter, histograms, boxplots
# - Identify patterns/trends
# - Generate insights

# Scatter plot to understand relationship between x and y
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='x', y='y')
plt.title('Scatter Plot: x vs y')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# 3.3 Feature-Target Relationship
# - Analyze correlation
# - Identify influential features
# - Select relevant features

print(f"\nCorrelation between x and y: {df['x'].corr(df['y']):.3f}")

# 4. Model Implementation
# 4.1 Data Preparation
# - Split into train/test sets
# - Feature scaling
# - Handle categorical variables
# - Create polynomial features

print("\nPreparing data for modeling...")
X = df[['x']]
y = df['y']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# 4.2 Model Selection & Training
# - Choose algorithms
# - Set up pipelines
# - Train models
# - Hyperparameter tuning

print("\nTraining models...")

# Linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Polynomial Regression (degree=2)
poly_model = make_pipeline(
    PolynomialFeatures(degree=2),
    StandardScaler(),
    LinearRegression()
)
poly_model.fit(X_train, y_train)

# Polynomial Regression (degree=3)
poly_model_3 = make_pipeline(
    PolynomialFeatures(degree=3),
    StandardScaler(),
    LinearRegression()
)
poly_model_3.fit(X_train, y_train)

print("Models trained successfully!")

# 4.3 Model Evaluation
# - Cross-validation
# - Performance metrics
# - Compare models
# - Select best model

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """Evaluate model performance"""
    # Training predictions
    y_train_pred = model.predict(X_train)
    
    # Test predictions
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    
    print(f"\n{model_name} Performance:")
    print(f"Train MSE: {train_mse:.4f}")
    print(f"Test MSE: {test_mse:.4f}")
    print(f"Train R²: {train_r2:.4f}")
    print(f"Test R²: {test_r2:.4f}")
    print(f"Cross-validation R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    return {
        'model_name': model_name,
        'train_mse': train_mse,
        'test_mse': test_mse,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std()
    }

# Evaluate all models
results = []
results.append(evaluate_model(linear_model, X_train, X_test, y_train, y_test, "Linear Regression"))
results.append(evaluate_model(poly_model, X_train, X_test, y_train, y_test, "Polynomial Regression (deg=2)"))
results.append(evaluate_model(poly_model_3, X_train, X_test, y_train, y_test, "Polynomial Regression (deg=3)"))

# Convert results to DataFrame for better visualization
results_df = pd.DataFrame(results)
print("\nModel Comparison:")
print(results_df[['model_name', 'test_r2', 'test_mse', 'cv_mean']])

# 4.4 Visualization of Results
# - Plot predictions vs actual
# - Residual analysis
# - Learning curves

# Plot predictions vs actual for the best model
best_model_idx = results_df['test_r2'].idxmax()
best_model_name = results_df.loc[best_model_idx, 'model_name']

if best_model_name == "Linear Regression":
    best_model = linear_model
elif best_model_name == "Polynomial Regression (deg=2)":
    best_model = poly_model
else:
    best_model = poly_model_3

# Generate predictions
y_pred = best_model.predict(X_test)

# Plot actual vs predicted
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title(f'Actual vs Predicted - {best_model_name}')

# Plot residuals
residuals = y_test - y_pred
plt.subplot(1, 2, 2)
plt.scatter(y_pred, residuals, alpha=0.7)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')

plt.tight_layout()
plt.show()

# 5. Model Deployment Preparation
# 5.1 Final Model Selection
# - Choose best performing model
# - Retrain on full dataset
# - Save model

print(f"\nBest model: {best_model_name}")

# Retrain best model on full dataset
print("Retraining best model on full dataset...")
if best_model_name == "Linear Regression":
    final_model = LinearRegression()
elif best_model_name == "Polynomial Regression (deg=2)":
    final_model = make_pipeline(
        PolynomialFeatures(degree=2),
        StandardScaler(),
        LinearRegression()
    )
else:
    final_model = make_pipeline(
        PolynomialFeatures(degree=3),
        StandardScaler(),
        LinearRegression()
    )

final_model.fit(X, y)

# Make final predictions on full dataset
y_final_pred = final_model.predict(X)
final_r2 = r2_score(y, y_final_pred)
final_mse = mean_squared_error(y, y_final_pred)

print(f"Final model performance on full dataset:")
print(f"R²: {final_r2:.4f}")
print(f"MSE: {final_mse:.4f}")

# 5.2 Model Serialization
# - Save model to file
# - Create prediction function
# - Prepare for deployment

# Example prediction function
def predict_value(x_value, model=final_model):
    """Make prediction using the trained model"""
    if isinstance(x_value, (int, float)):
        x_value = [[x_value]]
    elif isinstance(x_value, list):
        x_value = np.array(x_value).reshape(-1, 1)
    
    prediction = model.predict(x_value)
    return prediction[0] if len(prediction) == 1 else prediction

# Test the prediction function
test_value = 2.5
prediction = predict_value(test_value)
print(f"\nExample prediction for x={test_value}: y={prediction:.4f}")

# 6. Summary and Next Steps
print("\n" + "="*50)
print("PROJECT SUMMARY")
print("="*50)
print(f"Dataset: {df.shape[0]} samples, {df.shape[1]} features")
print(f"Best Model: {best_model_name}")
print(f"Final R² Score: {final_r2:.4f}")
print(f"Final MSE: {final_mse:.4f}")
print("\nNext Steps:")
print("1. Further hyperparameter tuning")
print("2. Try other algorithms (Random Forest, SVM, etc.)")
print("3. Feature engineering")
print("4. Model deployment")
print("="*50)

# Create a comprehensive results visualization
plt.figure(figsize=(15, 5))

# Plot 1: Original data with fitted curve
plt.subplot(1, 3, 1)
x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_range_pred = final_model.predict(x_range)

plt.scatter(X, y, alpha=0.5, label='Original Data')
plt.plot(x_range, y_range_pred, 'r-', label='Fitted Curve', linewidth=2)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Data with Fitted Model')
plt.legend()

# Plot 2: Model comparison
plt.subplot(1, 3, 2)
models = results_df['model_name']
test_r2_scores = results_df['test_r2']
plt.bar(models, test_r2_scores, color=['skyblue', 'lightgreen', 'lightcoral'])
plt.xticks(rotation=45)
plt.ylabel('R² Score')
plt.title('Model Comparison (Test R²)')
plt.ylim(0, 1)

# Plot 3: Residual distribution
plt.subplot(1, 3, 3)
sns.histplot(residuals, kde=True)
plt.xlabel('Residuals')
plt.title('Residual Distribution')
plt.axvline(x=0, color='r', linestyle='--')

plt.tight_layout()
plt.show()

print("\nProject execution completed successfully!")