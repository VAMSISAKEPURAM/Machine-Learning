import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

def load_and_prepare_data(file_path):
    """
    Load dataset and prepare features/target variables
    
    Parameters:
    file_path (str): Path to the CSV file
    
    Returns:
    tuple: (X, y) - Feature matrix and target vector
    """
    df = pd.read_csv(file_path)
    X = df[['Attendance_Hours']]
    y = df['Final_Marks']
    return X, y

def train_linear_regression(X_train, y_train):
    """Train a Linear Regression model with feature scaling."""
    # Initialize and fit scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    return model, scaler

def evaluate_model(model, scaler, X_test, y_test):
    """Evaluate the model on test data."""
    # Scale test data
    X_test_scaled = scaler.transform(X_test)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return {
        "MSE": mse,
        "R2": r2,
        "predictions": y_pred,
        "model": model,
        "scaler": scaler
    }

def predict_final_marks(model, scaler, attendance_hours):
    """
    Predict final marks for new attendance hours
    
    Parameters:
    model: Trained LinearRegression model
    scaler: Fitted StandardScaler
    attendance_hours (array-like): Attendance hours to predict
    
    Returns:
    array: Predicted final marks
    """
    attendance_array = np.array(attendance_hours).reshape(-1, 1)
    attendance_scaled = scaler.transform(attendance_array)
    return model.predict(attendance_scaled)

# Example usage
if __name__ == "__main__":
    # Load data
    X, y = load_and_prepare_data('Study_vs_Score_data.csv')
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model, scaler = train_linear_regression(X_train, y_train)
    
    # Evaluate model
    results = evaluate_model(model, scaler, X_test, y_test)
    
    print(f"Model Performance:")
    print(f"MSE: {results['MSE']:.2f}")
    print(f"R² Score: {results['R2']:.2f}")
    
    # Example prediction
    sample_hours = [[30], [40], [50]]
    predictions = predict_final_marks(model, scaler, sample_hours)
    
    print("\nSample Predictions:")
    for hours, pred in zip(sample_hours, predictions):
        print(f"{hours[0]} attendance hours → {pred:.1f} predicted marks")