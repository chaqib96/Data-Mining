import numpy as np
from sklearn.tree import DecisionTreeClassifier

class AdaBoost:
    """
    AdaBoost (Adaptive Boosting) classifier for binary classification.
    """
    
    def __init__(self, n_estimators=50, seed=None):
        self.n_estimators = n_estimators
        self.seed = seed
        self.alphas = []
        self.models = []
        
    def _compute_error(self, y, y_pred, w):
        """
        Calculates the weighted training error.
        """
        # TODO: Implement this function
        # Calculate the weighted sum of incorrect predictions
        # error = ...
        return error
    
    def _compute_alpha(self, error):
        """
        Computes the weight (alpha) of the current weak learner.
        """
        epsilon = 1e-10  # Tiny constant to avoid division by zero
        # TODO: Implement this function
        # Calculate alpha based on the error
        # alpha = ...
        return alpha
    
    def _update_weights(self, w, alpha, y, y_pred):
        """
        Updates and normalizes sample weights.
        """
        # TODO: Implement this function
        # Step 1: Update weights (increase weight for misclassified samples)
        # Hint: y * y_pred is 1 if correct, -1 if incorrect
        # w = ...
        
        # Step 2: Normalize weights so they sum to 1
        # w = ...
        
        return w
        
    def fit(self, X, y):
        """
        Trains the AdaBoost model.
        """
        n_samples, n_features = X.shape
        
        # Clear previous models
        self.models = []
        self.alphas = []
        
        # IMPORTANT: Convert labels to {-1, 1} if they are {0, 1}
        if set(np.unique(y)) == {0, 1}:
            y = np.where(y == 0, -1, 1)
            
        # Initialize weights uniformly: w_i = 1/n
        w = np.full(n_samples, 1.0 / n_samples)
        
        if self.seed is not None:
            np.random.seed(self.seed)
            
        for t in range(self.n_estimators):
            # a) Fit a weak learner on weighted data
            stump = DecisionTreeClassifier(max_depth=1, random_state=self.seed)
            stump.fit(X, y, sample_weight=w)
            
            # Predict on training data to compute error
            y_pred = stump.predict(X)
            
            # TODO: Implement the boosting steps
            # b) call _compute_error
            # error = ...
            
            # c) call _compute_alpha
            # alpha = ...
            
            # d) call _update_weights
            # w = ...
            
            # Store the trained model and its weight
            self.models.append(stump)
            self.alphas.append(alpha)
            
    def predict(self, X):
        """
        Predicts class labels for X.
        """
        # Collect predictions from all weak learners
        weak_preds = np.array([stump.predict(X) for stump in self.models])
        
        # Weighted sum of predictions
        weighted_sum = np.dot(self.alphas, weak_preds)
        
        # Return sign of the sum
        final_preds = np.sign(weighted_sum)
        
        # Map 0s to 1s (rare edge case where sum is exactly 0)
        final_preds[final_preds == 0] = 1
        
        return final_preds