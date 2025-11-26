import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
import seaborn as sns
import matplotlib.pyplot as plt

class HousePricePredictor:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = None
        self.features = [
            'OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 
            'TotalBsmtSF', '1stFlrSF', 'FullBath', 'TotRmsAbvGrd', 
            'YearBuilt', 'YearRemodAdd', 'GarageFinish_Unf', 
            'KitchenQual_TA', 'ExterQual_TA'
        ]
        self.target = 'SalePrice'
        
    def load_data(self, file_path):
        """Load and return the housing dataset"""
        self.df = pd.read_csv(file_path)
        return self.df.head()
    
    def preprocess_data(self):
        """Handle missing values and encode categorical variables"""
        # Fill missing values
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                self.df[col].fillna(self.df[col].mode()[0], inplace=True)
            else:
                self.df[col].fillna(self.df[col].median(), inplace=True)
        
        # One-hot encoding
        self.df = pd.get_dummies(self.df, drop_first=True, dtype=float)
        
        # Select most correlated features
        correlations = self.df.corr()[self.target].sort_values(ascending=False).drop(self.target).to_dict()
        high_corr_features = {k: v for k, v in correlations.items() if abs(v) > 0.5}
        self.features = list(high_corr_features.keys())
        
        # Filter dataset to selected features
        self.df = self.df[self.features + [self.target]]
        
        return self.df.describe()
    
    def explore_data(self):
        """Perform exploratory data analysis"""
        print("Dataset Info:")
        print(self.df.info())
        print("\nStatistical Summary:")
        print(self.df.describe())
        print(f"\nSelected Features: {len(self.features)}")
        print(self.features)
        
        # Correlation heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(self.df.corr(), annot=True, cmap='coolwarm', center=0)
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        plt.show()
        
        return self.df.corr()[self.target].sort_values(ascending=False)
    
    def build_model(self, alpha=1.0, degree=2):
        """Build the Lasso regression model with polynomial features"""
        self.model = make_pipeline(
            StandardScaler(),
            PolynomialFeatures(degree=degree, include_bias=False),
            Ridge(alpha=alpha, random_state=self.random_state)
        )
        return self.model
    
    def train_model(self, test_size=0.2):
        """Train the model and evaluate performance"""
        X = self.df[self.features]
        y = self.df[self.target]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Model Performance:")
        print(f"Mean Squared Error: {mse:.2f}")
        print(f"R² Score: {r2:.4f}")
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X, y, cv=5, scoring='r2')
        print(f"Cross-Validation R² Scores: {cv_scores}")
        print(f"Mean CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return {
            'mse': mse,
            'r2': r2,
            'cv_scores': cv_scores,
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred
        }
    
    def plot_results(self, results):
        """Plot actual vs predicted values"""
        plt.figure(figsize=(10, 6))
        plt.scatter(results['y_test'], results['y_pred'], alpha=0.7)
        plt.plot([results['y_test'].min(), results['y_test'].max()], 
                [results['y_test'].min(), results['y_test'].max()], 'r--', lw=2)
        plt.xlabel('Actual Prices')
        plt.ylabel('Predicted Prices')
        plt.title('Actual vs Predicted House Prices')
        plt.tight_layout()
        plt.show()
        
        # Residual plot
        residuals = results['y_test'] - results['y_pred']
        plt.figure(figsize=(10, 6))
        plt.scatter(results['y_pred'], residuals, alpha=0.7)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Prices')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        plt.tight_layout()
        plt.show()

def main():
    # Initialize the predictor
    predictor = HousePricePredictor()
    
    # Load data
    print("Loading data...")
    predictor.load_data('train.csv')
    
    # Preprocess data
    print("\nPreprocessing data...")
    predictor.preprocess_data()
    
    # Explore data
    print("\nExploring data...")
    correlations = predictor.explore_data()
    
    # Build model
    print("\nBuilding model...")
    predictor.build_model(alpha=0.1, degree=2)
    
    # Train and evaluate model
    print("\nTraining model...")
    results = predictor.train_model()
    
    # Plot results
    print("\nGenerating plots...")
    predictor.plot_results(results)
    
    print("\nModel training completed successfully!")

if __name__ == "__main__":
    main()