import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

def pseudoinverse_normal_eq(A: np.ndarray) -> np.ndarray:
    """
    TODO: Compute the Moore–Penrose pseudoinverse using the normal-equation formula.

    When A has full column rank:
        A^+ = (A^T A)^{-1} A^T

    Parameters
    ----------
    A : np.ndarray, shape (n, d)

    Returns
    -------
    A_pinv : np.ndarray, shape (d, n)
        The pseudoinverse of A.
    """
    # TODO: implement
    # A_pinv = ...
    return A_pinv

# --------------------------
# Task 3: Gradient descent solver (TODO)
# --------------------------

def least_squares_grad(A: np.ndarray, b: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """
    Gradient of J(theta) = (1/(2n)) ||A theta - b||_2^2.

    Parameters
    ----------
    A : (n, d') design matrix
    b : (n,) target vector
    theta : (d',) parameter vector

    Returns
    -------
    grad : (d',) gradient vector
    """
    n = A.shape[0]
    # TODO (a): implement grad
    # grad = ...
    return grad


def choose_learning_rate(A: np.ndarray) -> float:
    """
    Choose a learning rate using a Lipschitz constant bound.

    For J(theta) = (1/(2n))||A theta - b||^2, one can use:
        L = (1/n) * ||A||_2^2
    A conservative stable choice is eta = 1/L.

    Returns
    -------
    eta : float
    """
    n = A.shape[0]
    # TODO (b): compute L and return eta = 1/L
    # Hint: np.linalg.svd(A, compute_uv=False)[0] is the largest singular value
    # sigma_max = ...
    # L =
    # eta =
    return eta


def solve_least_squares_gd(
    A: np.ndarray,
    b: np.ndarray,
    n_steps: int = 5000,
    tol: float = 1e-8,
) -> np.ndarray:
    """
    Solve least squares with gradient descent on J(theta)=(1/(2n))||A theta - b||^2.

    Parameters
    ----------
    A : (n, d') design matrix
    b : (n,) target vector
    n_steps : number of GD iterations
    tol : stopping tolerance on gradient norm

    Returns
    -------
    theta_gd : (d',) approximate minimizer
    """
    b = np.asarray(b).reshape(-1)
    dprime = A.shape[1]
    theta = np.zeros(dprime)

    eta = choose_learning_rate(A)

    for t in range(n_steps):
        g = least_squares_grad(A, b, theta)

        # TODO (c): gradient descent update
        # theta = ...

        # Optional early stopping (keep)
        if np.linalg.norm(g) < tol:
            break

    return theta

