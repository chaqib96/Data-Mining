import numpy as np

def kmeans(X, K, max_iter=100, random_state=None):
    """Run Lloyd's K-means algorithm.

    Parameters
    ----------
    X : ndarray of shape (n, d)
        Data matrix.
    K : int
        Number of clusters.
    max_iter : int
        Maximum number of iterations.
    random_state : int or None
        Seed for reproducibility.

    Returns
    -------
    centres : ndarray of shape (K, d)
        Final cluster centres.
    labels : ndarray of shape (n,)
        Cluster assignment for each point (values in {0, ..., K-1}).
    """
    rng = np.random.default_rng(random_state)

    # Step 1: Initialise centres by picking K random data points
    n = X.shape[0]
    indices = rng.choice(n, size=K, replace=False)
    centres = X[indices].copy()

    labels = np.zeros(n, dtype=int)

    for iteration in range(max_iter):
        # Step 2: Assign each point to the nearest centre
        # Compute squared distances from each point to each centre: (n, K)
        # Using ||x - mu||^2 = ||x||^2 - 2 x^T mu + ||mu||^2
        X_sq = np.sum(X ** 2, axis=1, keepdims=True)
        centres_sq = np.sum(centres ** 2, axis=1)
        sq_dists = X_sq - 2 * (X @ centres.T) + centres_sq

        # Assign new labels for each point.
        new_labels = np.argmin(sq_dists, axis=1)

        # Step 3: Update centres
        for k in range(K):
            mask = new_labels == k
            if np.any(mask):
                centres[k] = X[mask].mean(axis=0)
            else:
                # Re-initialize empty cluster to a random data point
                centres[k] = X[rng.integers(0, n)]

        # Step 4: Check for convergence and stop the algorithm
        # if the new labels are unchanged
        if np.all(labels == new_labels):
            labels = new_labels
            break
        labels = new_labels

    return centres, labels
