import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error

class PolynomialLasso:
    """
    Polynomial Regression with L1 regularization and validation-based alpha selection.
    """
    
    def __init__(self, degree=2, alphas=[0.1, 1.0, 10.0]):
        self.degree = degree
        self.alphas = alphas
        self.best_alpha = None
        self.best_model = None
        self.min_val_error = float('inf')
        self.poly = None
        self.scaler = None
        
    def _transform_features(self, X):
        """
        Transforms input features X into polynomial features and scales them.
        """
        # TODO: Implement this function
        # 1. Initialize self.poly (if None) and transform X
        # 2. Initialize self.scaler (if None) and scale the poly features
        # Hint: Check if self.poly/self.scaler are None to decide whether to fit_transform or just transform
        # X_scaled = ...
        return X_scaled

    def _split_data(self, X, y, val_ratio):
        """
        Splits the data into training and validation sets using the LAST n rows logic.
        """
        # TODO: Implement this function
        # Calculate split index: n_val = total_samples * val_ratio
        # X_train_sub = ...
        # y_train_sub = ...
        # X_val = ...
        # y_val = ...
        
        return X_train_sub, y_train_sub, X_val, y_val

    def fit(self, X, y, val_ratio=0.2):
        """
        Fits the model with hyperparameter tuning.
        """
        # Reset transformers for new training
        self.poly = None
        self.scaler = None
        
        # 1. Transform features (will fit transformers)
        # X_poly = self._transform_features(X)
        
        # 2. Split data
        # X_sub, y_sub, X_val, y_val = ...

        best_alpha = None
        best_mse = float('inf')
        
        # 3. Hyperparameter search
        # for alpha in self.alphas:
            # Initialize Lasso
            # Increase max_iter to 100000 to ensure convergence with Coordinate Descent.
            # lasso = Lasso(..., max_iter=100000)
            
            # Train on sub-training set
            # Validate on validation set
            # Update best_alpha if mse is lower
        
        # self.best_alpha = ...
        # self.min_val_error = ...
        
        # 4. Refit on ALL data with best alpha
        # self.best_model = ...
        # self.best_model.fit(X_poly, y)
        
    def predict(self, X):
        """
        Predicts using the best fitted model.
        """
        if self.best_model is None:
            raise RuntimeError("Model is not fitted yet.")
            
        # 5. Transform features (will use fitted transformers)
        # X_poly = self._transform_features(X)
        # return self.best_model.predict(X_poly)
