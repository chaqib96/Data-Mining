import numpy as np

class EigenfacesSVD:
    """
    A class to perform SVD on image data (faces) to extract eigenfaces
    and reconstruct images.
    """
    def __init__(self):
        self.mean_face = None
        self.U = None
        self.S = None
        self.Vt = None

    def fit(self, X):
        """
        Computes the SVD of the mean-centered data.
        Assumes X is of shape (n_features, n_samples).
        """
        # --- YOUR CODE HERE ---
        # 1. Compute the mean face (mean along the sample axis)
        self.mean_face = ...

        # 2. Mean-center the data
        X_centered = ...

        # 3. Compute SVD of the centered data
        # Hint: Use np.linalg.svd with full_matrices=False
        self.U, self.S, self.Vt = ...

        # --- END YOUR CODE ---

        print("Shape of U:", self.U.shape)
        print("Shape of S:", self.S.shape)
        print("Shape of Vt:", self.Vt.shape)


    def rank_k_approximation(self, k):
        """
        Generates the rank-k approximation of X using the top k eigenfaces.
        Note: This specific reconstruction is meant for the training data where U, S, Vt
        are already computed. We reconstruct using the outer product of the first k components.
        """

        # --- STUDENT CODE HERE ---
        # 1. Initialize numpy matrix matching the shape of X filled with zeros (read the shape of X from U and V)
        X_reconstructed = ...

        # 2. Iterate up to k, adding the outer product of the i-th column of U and i-th row of Vt,
        # scaled by the i-th singular value.
        for i in range(k):
            X_reconstructed ...

        # 3. Add the mean face back
        X_reconstructed ...
        # --- END STUDENT CODE ---

        return X_reconstructed

    def reconstruct_dataset(self, X, k):
        """
        Reconstructs a given dataset of faces using the top k eigenfaces by projecting it to the eigenfaces base.
        """

        # 1. Mean-center the input data
        X_centered = X - self.mean_face

        # 2. Extract the top k eigenfaces (first k columns of U)
        U_k = self.U[:, :k]

        # 3. Project the centered data onto the top k eigenfaces to get the weights
        weights = U_k.T @ X_centered

        # 4. Reconstruct the faces using the weights and the top k eigenfaces, then add the mean face back
        X_reconstructed = (U_k @ weights) + self.mean_face

        return X_reconstructed

    def cumulative_explained_variance(self):
        """
        Computes the cumulative explained variance of the singular values.
        """
        # --- YOUR CODE HERE ---
        # 1. Compute the variance explained by each singular value (stored in self.S), which is proportional to the square of its value
        variance = ...

        # 2. Calculate the cumulative sum of the variance using np.cumsum(). The value in the position k of this vector is proportional to the total variance explained by the rank k approximation.
        cumulative_variance = ...

        # 3. Normalize dividing by the total variance
        explained_variance_ratio = ...
        # --- END YOUR CODE ---

        return explained_variance_ratio

    def reconstruction_error(self, k):
        """
        Computes the Mean Squared Error (MSE) of the rank-k approximation.
        """
        # --- YOUR CODE HERE ---
        # 1. Compute the original dataset form the U S and Vt matrices.
        # To match the n x m shape of X, we should take only the first m columns of U:
        m = self.Vt.shape[0]
        U_m = self.U[:,:m]
        # Observation: Notice that self.S is a vector, not a diagonal matrix.
        X_centered = ...

        # Add the mean face
        X_full = ...

        # 2. Reconstruct X using k components
        X_approx = ...

        # 3. Calculate the Mean Squared Error between original X and X_approx
        error = ...
        # --- END YOUR CODE ---

        return error
