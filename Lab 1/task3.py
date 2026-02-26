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
        if self.poly is None:
            self.poly = PolynomialFeatures(degree=self.degree, include_bias=False)
            X_poly = self.poly.fit_transform(X)
        else:
            X_poly = self.poly.transform(X)
        if self.scaler is None:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X_poly)
        else:
            X_scaled = self.scaler.transform(X_poly)
        return X_scaled

    def _split_data(self, X, y, val_ratio):
        """
        Splits the data into training and validation sets using the LAST n rows logic.
        """
        n_val = int(len(X) * val_ratio)
        X_train_sub = X[:-n_val]
        y_train_sub = y[:-n_val]
        X_val = X[-n_val:]
        y_val = y[-n_val:]
        return X_train_sub, y_train_sub, X_val, y_val

    def fit(self, X, y, val_ratio=0.2):
        """
        Fits the model with hyperparameter tuning.
        """
        # Reset transformers for new training
        self.poly = None
        self.scaler = None

        X_poly = self._transform_features(X)
        X_sub, y_sub, X_val, y_val = self._split_data(X_poly, y, val_ratio)

        best_alpha = None
        best_mse = float('inf')

        for alpha in self.alphas:
            lasso = Lasso(alpha=alpha, max_iter=100000)
            lasso.fit(X_sub, y_sub)
            mse = mean_squared_error(y_val, lasso.predict(X_val))
            if mse < best_mse:
                best_mse = mse
                best_alpha = alpha

        self.best_alpha = best_alpha
        self.min_val_error = best_mse
        self.best_model = Lasso(alpha=self.best_alpha, max_iter=100000)
        self.best_model.fit(X_poly, y)
        
    def predict(self, X):
        """
        Predicts using the best fitted model.
        """
        if self.best_model is None:
            raise RuntimeError("Model is not fitted yet.")
        X_poly = self._transform_features(X)
        return self.best_model.predict(X_poly)
