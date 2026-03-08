import numpy as np
from subtask1 import rbf_kernel
from subtask2 import kmeans

class RBFNetwork:
    """RBF network for binary classification.
    
    Parameters
    ----------
    n_centres : int
        Number of RBF centres (K).
    gamma : float
        Width parameter for the Gaussian basis functions.
    random_state : int or None
        Random seed for K-means initialisation.
    """
    
    def __init__(self, n_centres=10, gamma=1.0, random_state=None):
        self.n_centres = n_centres
        self.gamma = gamma
        self.random_state = random_state
    
    def fit(self, X, y):
        """Fit the RBF network.
    
        1. Run K-means on X to find the centres.
        2. Build the design matrix Phi using the RBF kernel.
        3. Solve for the weights using the pseudo-inverse.
    
        Parameters
        ----------
        X : ndarray of shape (n, d)
        y : ndarray of shape (n,), values in {-1, +1}
        """
        # Step 1: Find centres with K-means (use your implementation)
        # Hint: store the centers. By convention scikit-learn uses self.variable_name_
        # for estimator state. The tests expect the centroids to be self.centres_.
        centres, _ = kmeans(X, self.n_centres, random_state=self.random_state)
        self.centres_ = centres

        # Step 2: Build design matrix Phi of shape (n, K+1)
        #   - First K columns: RBF kernel values between X and the centres
        #   - Last column: ones (bias term)
        # Hint: Use np.hstack (or the more general np.concatenate) to add a column
        # to the matrix.
        Phi = rbf_kernel(X, self.centres_, gamma=self.gamma)
        Phi = np.hstack([Phi, np.ones((X.shape[0], 1))])

        # Step 3: Compute weights via pseudo-inverse
        # Hint: you need to store the weight matrix as an instance variable.
        # The tests expect this variable to be self.weights_
        self.weights_ = np.linalg.lstsq(Phi, y, rcond=None)[0]

        # By convention scikit-learn returns self. This allows for chaining invokations.
        # Example: y_hat = RBFNetwork().fit(X, y).predict(new_X)
        return self 
    
    def decision_function(self, X):
        """Compute the raw output (before sign) for each sample.
    
        Parameters
        ----------
        X : ndarray of shape (n, d)
    
        Returns
        -------
        scores : ndarray of shape (n,)
        """
        # Step 1: Build the design matrix Phi of shape (n, K+1)
        # using the already fitted centers.
        # Hint: Remember to set the same hyper-parameters for the RBF-kernel.
        # Hint: Remember to construct the design matrix to mirror the the one
        # used during fitting.
        #   - First K columns: the kernel values
        #   - Last column: ones (the bias term)
        Phi = rbf_kernel(X, self.centres_, gamma=self.gamma)
        Phi = np.hstack([Phi, np.ones((X.shape[0], 1))])

        # Step 3: Return the raw scores (n, ). Positive scores predict +1, negative scores -1
        # The predict function uses these scores and turn them into labels.
        return Phi @ self.weights_ 
    
    def predict(self, X):
        """Predict class labels in {-1, +1}.
    
        Parameters
        ----------
        X : ndarray of shape (n, d)
    
        Returns
        -------
        y_pred : ndarray of shape (n,)
        """
        # We turn the raw scores into a prediction
        return np.sign(self.decision_function(X))
