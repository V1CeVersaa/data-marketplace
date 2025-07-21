"""
Implements three main components of the mechanism: allocate_features (AF*), revenue_function (RF*) and shapley_approx / robustify_shapley (PD*)
"""

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from .model import MLModel, gain


def allocate_features(
    X: NDArray[np.float64],
    p: float,
    bid: float,
    sigma: float = 1.0,
    rng: np.random.Generator | None = None,
) -> NDArray[np.float64]:
    """
    This allocation is monotone in bid: a larger bid implies less noise,
    which in turn leads to a higher expected gain.
    """
    if rng is None:
        rng = np.random.default_rng()
    noise_scale = max(0.0, p - bid)
    if noise_scale == 0.0:
        return X.copy()
    noise = rng.normal(loc=0.0, scale=sigma * noise_scale, size=X.shape)
    return X + noise


def revenue_function(
    p: float,
    bid: float,
    y_true: NDArray[np.float64],
    model: MLModel,
    X_alloc_func: Callable[[float], NDArray[np.float64]],
    integral_grid: int = 15,
) -> tuple[float, float]:
    """
    Compute payment (Eq. 3) with Myerson's payment function rule.
    Returns (revenue r_n, realised_gain g_n).

    However, this function is **computationally expensive** as it performs a numerical
    integral that requires training the model 'integral_grid' times.
    """
    # Calculate gain G(b) at the reported bid
    Xb = X_alloc_func(bid)
    y_test, y_test_hat = model.fit_predict(Xb, y_true)
    g_bid = gain(y_test, y_test_hat)

    # Numerical integral to compute int_0^bid G(z) dz
    zs = np.linspace(0.0, bid, integral_grid)
    gs = []
    for z in zs:
        Xz = X_alloc_func(z)
        y_test_z, y_test_hat_z = model.fit_predict(Xz, y_true)
        gs.append(gain(y_test_z, y_test_hat_z))

    integral = float(np.trapezoid(gs, zs))
    rev = bid * g_bid - integral
    rev = max(0.0, rev)  # Ensure non-negative revenue (individual rationality)
    return rev, g_bid


def shapley_approx(
    y_true: NDArray[np.float64],
    X: NDArray[np.float64],
    model: MLModel,
    num_samples: int = 200,
    rng: np.random.Generator | None = None,
) -> NDArray[np.float64]:
    """
    Algorithm 2: Monte-Carlo approximation of Shapley value for M features.
    Returns psi_hat in R^M, a vector of contribution shares that sum to ~1.

    NOTE: This function is **computationally expensive**, especially with many
    features, as it may require training the model on many different subsets
    of features. The use of caching is a critical optimization.
    """
    if rng is None:
        rng = np.random.default_rng()
    M = X.shape[0]
    contrib = np.zeros(M)
    indices = np.arange(M)

    base_gain_cache: dict[tuple[int, ...], float] = {}

    def marginal_gain(prefix: tuple[int, ...], idx: int) -> float:
        """Calculates the marginal gain of adding feature 'idx' to 'prefix'."""
        prefix = tuple(sorted(prefix))
        if prefix not in base_gain_cache:
            X_prefix = X[list(prefix), :] if prefix else np.empty((0, X.shape[1]))
            y_test, y_test_hat = model.fit_predict(X_prefix, y_true)
            base_gain_cache[prefix] = gain(y_test, y_test_hat)
        g_prefix = base_gain_cache[prefix]

        new_set = tuple(sorted(prefix + (idx,)))
        if new_set not in base_gain_cache:
            X_new = X[list(new_set), :]
            y_test_new, y_test_hat_new = model.fit_predict(X_new, y_true)
            base_gain_cache[new_set] = gain(y_test_new, y_test_hat_new)
        g_new = base_gain_cache[new_set]

        return g_new - g_prefix

    for _ in range(num_samples):
        perm = rng.permutation(indices)
        prefix_indices: tuple[int, ...] = tuple()
        for j in perm:
            mg = marginal_gain(prefix_indices, int(j))
            contrib[j] += mg
            prefix_indices += (int(j),)

    contrib /= num_samples
    total = contrib.sum()
    if total > 0:
        contrib /= total  # Normalise to sum to 1
    return contrib


def robustify_shapley(
    psi: NDArray[np.float64],
    X: NDArray[np.float64],
    lam: float = np.log(2.0),
) -> NDArray[np.float64]:
    """
    Algorithm 3: Apply an anti-replication penalty to Shapley values
    based on the cosine similarity between features.
    """
    M = X.shape[0]
    if M == 0:
        return psi

    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    Xn = X / norms
    sim = np.abs(Xn @ Xn.T)  # Cosine similarity

    # Penalty is higher for features that are similar to many other features
    penalty = np.exp(-lam * (sim.sum(axis=1) - 1.0))  # Subtract self-similarity of 1
    psi_penalised = psi * penalty

    total_psi_penalised = psi_penalised.sum()
    if total_psi_penalised > 0:
        psi_penalised /= total_psi_penalised  # Re-normalise

    return psi_penalised
