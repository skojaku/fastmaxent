"""
Fast Maximum Entropy Network Parameter Inference

This module implements fast parameter inference algorithms for maximum entropy
network models using Monte Carlo gradient estimation with Numba acceleration.

The main functionality includes:
- Fast gradient computation using Monte Carlo sampling
- Parameter optimization using Adam optimizer
- Convergence diagnostics and parameter initialization
"""

import numpy as np
from numba import njit, jit, prange
from . import sampling
from .optimizer import AdamOptimizer, clip_grad_norm


def calc_grad(alpha, deg, n_samples=30):
    """
    Fast computation of gradients using optimized batch degree sampling.

    This function uses the optimized batch degree computation with Numba acceleration
    for maximum performance in parameter inference. The batch sampling generates
    multiple networks in a single call, reducing function call overhead.

    Parameters
    ----------
    alpha : numpy.ndarray
        Current parameter estimates.
    deg : numpy.ndarray
        Observed degree sequence.
    n_samples : int, default=30
        Number of Monte Carlo samples.

    Returns
    -------
    numpy.ndarray
        Gradient vector (observed - expected degrees).
    """
    # Use optimized batch degree computation directly from the sampler
    n_nodes = len(alpha)

    edges = sampling(alpha=alpha, n_samples=n_samples, weighted=False, flatten=True)

    expected_deg = (
        np.bincount(np.array(edges)[:2].reshape(-1), minlength=n_nodes) / n_samples
    )

    grad = deg - expected_deg
    grad /= np.maximum(deg, 1)
    return grad


@njit(fastmath=True, cache=True)
def compute_loss(grad):
    """
    Compute loss function (sum of squared gradients).

    Parameters
    ----------
    grad : numpy.ndarray
        Gradient vector.

    Returns
    -------
    float
        Loss value.
    """
    return np.sum(grad * grad)


@njit(fastmath=True, cache=True)
def compute_relative_loss(grad, deg):
    """
    Compute relative loss function (relative to observed degrees).

    Parameters
    ----------
    grad : numpy.ndarray
        Gradient vector.
    deg : numpy.ndarray
        Observed degree sequence.

    Returns
    -------
    float
        Relative loss value.
    """
    relative_grad = grad / np.maximum(deg, 1e-8)
    return np.sum(relative_grad * relative_grad)


@njit(fastmath=True, cache=True)
def check_convergence(loss_history, patience=10, min_delta=1e-6):
    """
    Check if optimization has converged based on loss history.

    Parameters
    ----------
    loss_history : numpy.ndarray
        Array of recent loss values.
    patience : int, default=10
        Number of iterations to wait for improvement.
    min_delta : float, default=1e-6
        Minimum change in loss to be considered improvement.

    Returns
    -------
    bool
        True if converged, False otherwise.
    """
    if len(loss_history) < patience + 1:
        return False

    recent_loss = loss_history[-1]
    best_loss = np.min(loss_history[:-patience])

    return (best_loss - recent_loss) < min_delta


def initialize_parameters(deg, method="degree_based"):
    """
    Initialize parameters for optimization.

    Parameters
    ----------
    deg : numpy.ndarray
        Observed degree sequence.
    method : str, default='degree_based'
        Initialization method ('degree_based', 'random', 'uniform').

    Returns
    -------
    numpy.ndarray
        Initial parameter values.
    """
    n_nodes = len(deg)

    if method == "degree_based":
        # Initialize based on observed degrees
        m = np.sum(deg) // 2  # Number of edges
        if m > 0:
            alpha = -np.log(deg / np.sqrt(2 * m))
            # Handle zero degrees
            alpha[deg == 0] = np.max(alpha[deg > 0]) + 1.0
        else:
            alpha = np.ones(n_nodes)

    elif method == "random":
        # Random initialization
        alpha = np.random.normal(0, 1, n_nodes)

    elif method == "uniform":
        # Uniform initialization
        alpha = np.ones(n_nodes)

    else:
        raise ValueError(f"Unknown initialization method: {method}")

    return alpha.astype(np.float64)


def estimate_parameters(
    deg,
    n_iter=200,
    lr=0.01,
    n_samples=30,
    init_method="degree_based",
    patience=20,
    grad_clip=None,
    verbose=True,
):
    """
    Estimate UBCM parameters using Adam optimization.

    Parameters
    ----------
    deg : numpy.ndarray
        Observed degree sequence.
    n_iter : int, default=200
        Maximum number of optimization iterations.
    lr : float, default=0.01
        Learning rate for Adam optimizer.
    n_samples : int, default=30
        Number of Monte Carlo samples for gradient estimation.
    init_method : str, default='degree_based'
        Parameter initialization method.
    patience : int, default=20
        Early stopping patience.
    grad_clip : float, optional
        Maximum gradient norm for clipping.
    verbose : bool, default=True
        Whether to print progress.

    Returns
    -------
    dict
        Dictionary containing:
        - 'alpha': estimated parameters
        - 'losses': loss history
        - 'converged': whether optimization converged
        - 'n_iter': number of iterations performed
    """
    # Initialize parameters
    alpha = initialize_parameters(deg, method=init_method)

    # Initialize optimizer
    optimizer = AdamOptimizer(lr=lr, beta1=0.9, beta2=0.999)

    # Tracking variables
    losses = []
    loss_history = np.zeros(min(patience + 1, n_iter), dtype=np.float64)

    if verbose:
        print(f"Starting parameter estimation with {n_iter} max iterations...")

    for iteration in range(n_iter):
        grad = calc_grad(alpha, deg, n_samples=n_samples)

        # Apply gradient clipping if specified
        if grad_clip is not None:
            grad = clip_grad_norm(grad, grad_clip)

        # Compute loss
        loss = compute_loss(grad)
        losses.append(loss)

        # Update loss history for convergence check
        if iteration < len(loss_history):
            loss_history[iteration] = loss
        else:
            loss_history[:-1] = loss_history[1:]
            loss_history[-1] = loss

        # Update parameters
        alpha = optimizer.step(alpha, grad)

        # Print progress
        if verbose and iteration % 10 == 0:
            rel_loss = compute_relative_loss(grad, deg)
            print(
                f"Iteration {iteration:4d}, Loss: {loss:.6f}, Rel Loss: {rel_loss:.6f}"
            )

        # Check convergence
        if iteration > patience and check_convergence(loss_history, patience):
            if verbose:
                print(f"Converged after {iteration + 1} iterations")
            break

    return {
        "alpha": alpha,
        "losses": np.array(losses),
        "converged": iteration < n_iter - 1,
        "n_iter": iteration + 1,
    }
