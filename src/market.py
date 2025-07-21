"""
Implements the Marketplace class, which glues all components together to simulate the market, and the generate_synthetic_market function, which generates a synthetic market with correlated features and buyers with sparse tasks.
"""

import logging
from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

from .mechanism import allocate_features, revenue_function, robustify_shapley, shapley_approx
from .model import MLModel
from .participants import Buyer, Seller
from .pricer import MWUPricer


class Marketplace:
    """Glues all components together to simulate the market."""

    def __init__(
        self,
        sellers: Sequence[Seller],
        pricer: MWUPricer,
        model_factory: Callable[[], MLModel],
        noise_sigma: float = 1.0,
        shapley_samples: int = 200,
        robust_shapley: bool = True,
        rng: np.random.Generator | None = None,
    ):
        self.sellers: list[Seller] = list(sellers)
        self.M = len(sellers)
        self.pricer = pricer
        self.model_factory = model_factory
        self.noise_sigma = noise_sigma
        self.shapley_samples = shapley_samples
        self.robust_shapley = robust_shapley
        self.rng = rng or np.random.default_rng()
        self.X_full = np.stack([s.feature for s in self.sellers], axis=0)

    def transact(self, buyer: Buyer) -> dict[str, float]:
        """Runs the complete 7-step transaction for a single buyer."""

        # Step 1: Price sampling
        p_n = self.pricer.sample_price()

        # Steps 2, 3: Buyer makes a truthful bid
        bid = buyer.choose_bid()

        # Step 4: Allocation based on price and bid
        def alloc_func(_bid: float) -> NDArray[np.float64]:
            return allocate_features(self.X_full, p=p_n, bid=_bid, sigma=self.noise_sigma, rng=self.rng)

        # Steps 5, 6: Revenue calculation based on the realised gain
        model = self.model_factory()
        revenue, g_bid = revenue_function(
            p=p_n,
            bid=bid,
            y_true=buyer.y,
            model=model,
            X_alloc_func=alloc_func,
        )

        # Step 7: Payment division via (Robust) Shapley values
        X_allocated = alloc_func(bid)  # Use the actually allocated features for division
        psi = shapley_approx(
            buyer.y,
            X_allocated,
            model=model,
            num_samples=self.shapley_samples,
            rng=self.rng,
        )

        if self.robust_shapley:  # use robust Shapley approximation to penalise sellers offering redundant information
            psi = robustify_shapley(psi, X_allocated)

        for m, s in enumerate(self.sellers):
            s.revenue += revenue * psi[m]

        # MWU Update: Evaluate "what-if" revenue for each candidate price in the grid
        B_max = self.pricer.grid[-1]

        def hypothetical_revenue(candidate_price: float) -> float:
            """Calculates the revenue if the posted price were candidate_price."""
            alloc_func_hypothetical = lambda b: allocate_features(self.X_full, p=candidate_price, bid=b, sigma=self.noise_sigma, rng=self.rng)
            r, _ = revenue_function(
                p=candidate_price,
                bid=bid,
                y_true=buyer.y,
                model=self.model_factory(),
                X_alloc_func=alloc_func_hypothetical,
                integral_grid=10,  # Use a coarser grid for faster MWU update
            )
            return r

        hypothetical_revenues = np.array([hypothetical_revenue(c) for c in self.pricer.grid])
        normalised_gains = np.clip(hypothetical_revenues / max(B_max, 1e-8), 0.0, 1.0)

        self.pricer.update(normalised_gains)

        return {"p": p_n, "bid": bid, "gain": g_bid, "revenue": revenue}


def generate_synthetic_market(
    M: int = 20,
    T: int = 400,
    N_buyers: int = 120,
    rng: np.random.Generator | None = None,
) -> tuple[list[Seller], list[Buyer]]:
    """
    Generates sellers with correlated features and buyers with sparse tasks, simulating a market.
    Returns a tuple of (sellers, buyers).
    """
    logging.info(f"Generating synthetic market with M={M} sellers, T={T} time steps, N_buyers={N_buyers}")

    rng = rng or np.random.default_rng(12345)
    latent = rng.normal(size=(5, T))
    Xs: list[NDArray[np.float64]] = []  # List to hold features for each seller
    for _ in range(M):
        weights = rng.normal(size=5)
        feature: NDArray[np.float64] = weights @ latent + 0.05 * rng.normal(size=T)
        Xs.append(feature)
    sellers = [Seller(idx=i, feature=Xs[i]) for i in range(M)]

    buyers: list[Buyer] = []
    X_stack = np.stack(Xs)
    for n in range(N_buyers):
        coef = rng.normal(size=M)
        coef[rng.random(size=M) < 0.7] = 0.0  # Sparse task
        y = coef @ X_stack + 0.5 * rng.normal(size=T)
        y = y.astype(np.float64)
        mu = float(rng.uniform(0.5, 2.0))  # Private value for unit gain
        buyers.append(Buyer(idx=n, y=y, mu=mu))

    logging.info(f"Generated {len(sellers)} sellers and {len(buyers)} buyers")
    for i in range(min(3, len(sellers))):
        logging.info(f"Seller {i}: feature[:3]={sellers[i].feature[:3]}")
    for i in range(min(3, len(buyers))):
        logging.info(f"Buyer {i}: mu={buyers[i].mu:.2f}, y[:3]={buyers[i].y[:3]}")

    return sellers, buyers
