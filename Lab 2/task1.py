import numpy as np

# --------------------------
# SubTask 1: Objectives
# --------------------------

def f1(x: float) -> float:
    """
    Implement objective f1.

    """
    #TODO
    #return ...


def f2(x: float) -> float:
    """
    Implement objective f2.

    """
    #TODO
    #return ...

# --------------------------
# SubTask 2: Gradients
# --------------------------

def grad_f1(x: float) -> float:
    """
    Derivative of:
        f1(x) = (x^2 - 1)^2 + x
    """
    #TODO
    #return ...


def grad_f2(x: float) -> float:
    """
    Derivative of:
        f2(x) = 10 (x - 2)^2
    """
    #TODO
    #return ...

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

        # (3a) Gradient descent update
        #TODO
        #x = ...

        # (3b) Track history
        #TODO: Uncomment
        #history.append(x)

        # (3c) Convergence check
        #TODO: Uncomment
        #if abs(grad_fn(x)) < tol:
        #    return x, np.array(history), "converged"

        # (3d) Divergence detection
        #TODO: Uncomment
        #if (not np.isfinite(x)) or (abs(x) > divergence_bound) or (not np.isfinite(f_fn(x))):
        #    return x, np.array(history), "diverged"

    return x, np.array(history), "max_steps"
