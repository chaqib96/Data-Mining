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
        incorrect = (y != y_pred)
        error = np.sum(w[incorrect])
        return error
    
    def _compute_alpha(self, error):
        """
        Computes the weight (alpha) of the current weak learner.
        """
        epsilon = 1e-10  # Tiny constant to avoid division by zero
        error = np.clip(error, epsilon, 1 - epsilon)
        alpha = 0.5 * np.log((1 - error) / error)
        return alpha
    
    def _update_weights(self, w, alpha, y, y_pred):
        """
        Updates and normalizes sample weights.
        """
        w = w * np.exp(-alpha * y * y_pred)
        w = w / np.sum(w)
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
            
            error = self._compute_error(y, y_pred, w)
            alpha = self._compute_alpha(error)
            w = self._update_weights(w, alpha, y, y_pred)

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