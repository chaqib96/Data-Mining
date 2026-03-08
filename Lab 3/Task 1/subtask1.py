import numpy as np

def linear_kernel(X, Y):
    """Compute the linear kernel matrix.

    Parameters
    ----------
    X : ndarray of shape (n, d)
    Y : ndarray of shape (m, d)

    Returns
    -------
    K : ndarray of shape (n, m)
    """
    return X @ Y.T


def polynomial_kernel(X, Y, gamma=1.0, coef0=1.0, degree=3):
    """Compute the polynomial kernel matrix.

    Parameters
    ----------
    X : ndarray of shape (n, d)
    Y : ndarray of shape (m, d)
    gamma : float
    coef0 : float
    degree : int

    Returns
    -------
    K : ndarray of shape (n, m)
    """
    return (gamma * (X @ Y.T) + coef0) ** degree


def rbf_kernel(X, Y, gamma=1.0):
    """Compute the RBF (Gaussian) kernel matrix.

    Parameters
    ----------
    X : ndarray of shape (n, d)
    Y : ndarray of shape (m, d)
    gamma : float

    Returns
    -------
    K : ndarray of shape (n, m)

    Hint
    ----
    Use the identity ||x - y||^2 = ||x||^2 - 2 x^T y + ||y||^2
    to avoid explicit loops.
    """
    X_sq = np.sum(X ** 2, axis=1, keepdims=True)
    Y_sq = np.sum(Y ** 2, axis=1)
    pairwise_sq_dist = X_sq - 2 * (X @ Y.T) + Y_sq
    return np.exp(-gamma * pairwise_sq_dist)


def sigmoid_kernel(X, Y, gamma=1.0, coef0=0.0):
    """Compute the sigmoid (hyperbolic tangent) kernel matrix.

    Parameters
    ----------
    X : ndarray of shape (n, d)
    Y : ndarray of shape (m, d)
    gamma : float
    coef0 : float

    Returns
    -------
    K : ndarray of shape (n, m)
    """
    return np.tanh(gamma * (X @ Y.T) + coef0)
