import numpy as np

# --------------------------
# SubTask 1: Objectives
# --------------------------

def f1(x: float) -> float:
    """
    Implement objective f1.
    f1(x) = (x^2 - 1)^2 + x
    """
    return (x ** 2 - 1) ** 2 + x


def f2(x: float) -> float:
    """
    Implement objective f2.
    f2(x) = 10 (x - 2)^2
    """
    return 10.0 * (x - 2) ** 2

# --------------------------
# SubTask 2: Gradients
# --------------------------

def grad_f1(x: float) -> float:
    """
    Derivative of:
        f1(x) = (x^2 - 1)^2 + x
    """
    return 4 * x * (x ** 2 - 1) + 1


def grad_f2(x: float) -> float:
    """
    Derivative of:
        f2(x) = 10 (x - 2)^2
    """
    return 20.0 * (x - 2)

# --------------------------
# SubTask 3: Gradient Descent
# --------------------------

def gradient_descent(
    grad_fn,
    f_fn,
    x0: float,
    lr: float,
    n_steps: int = 200,
    tol: float = 1e-6,
    divergence_bound: float = 1e6,
):
    """
    Runs gradient descent on a 1D objective.

    Returns
    -------
    x_final : float
        Final iterate
    history : np.ndarray
        All iterates x_0, x_1, ..., x_T
    status : str
        One of: "converged", "max_steps", "diverged"
    """
    x = float(x0)
    history = [x]

    for t in range(n_steps):
        g = grad_fn(x)
        x = x - lr * g
        history.append(x)
        if abs(grad_fn(x)) < tol:
            return x, np.array(history), "converged"
        if (not np.isfinite(x)) or (abs(x) > divergence_bound) or (not np.isfinite(f_fn(x))):
            return x, np.array(history), "diverged"

    return x, np.array(history), "max_steps"
