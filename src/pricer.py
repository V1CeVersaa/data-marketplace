"""
Implements the Price Update Function (PF) from Algorithm 1, which uses the multiplicative weights update rule, dynamically updating the price.
We implement the machanism as a class MWUPricer with method sample_price() and update(gains).
"""

import numpy as np
from numpy.typing import NDArray


class MWUPricer:
    """
    Implements Algorithm 1 (price update via Multiplicative Weights) a.k.a. PF
    """

    def __init__(
        self,
        b_min: float,
        b_max: float,
        N: int,
        L_lipschitz: float = 1.0,
        rng: np.random.Generator | None = None,
    ):
        self.rng = rng or np.random.default_rng()
        self.eps = 1.0 / (L_lipschitz * np.sqrt(max(N, 1)))  # epsilon-net granularity suggested by theory
        self.grid: NDArray[np.float64] = np.arange(b_min, b_max + 1e-9, self.eps, dtype=np.float64)
        self.num_exp = len(self.grid)
        self.delta = np.sqrt(np.log(self.num_exp) / max(N, 1))
        self.weights: NDArray[np.float64] = np.ones(self.num_exp, dtype=np.float64)  # Initialise w_i = 1

    def sample_price(self) -> float:
        """Sample a price from the current distribution over the price grid."""
        prob = self.weights / self.weights.sum()
        idx = self.rng.choice(self.num_exp, p=prob)
        return float(self.grid[idx])

    def update(self, gains: NDArray[np.float64]):
        """
        Update weights using the multiplicative rule based on evaluated gains.
        'gains' should be a numpy array of the same size as self.grid,
        containing the normalised revenue for each candidate price.
        """
        if gains.shape != self.weights.shape:
            raise ValueError("Gains array shape must match weights shape.")
        self.weights *= 1.0 + self.delta * gains
