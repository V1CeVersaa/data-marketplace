"""
Defines Participants in the marketplace: Seller and Buyer.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class Seller:
    """
    Data class representing a seller in the auction.
    Attributes:
        idx (int): Index of the seller.
        feature (NDArray[np.float64]): Feature vector of the seller.
        revenue (float): Accumulated monetary compensation received from the marketplace.
    """

    idx: int
    feature: NDArray[np.float64]
    revenue: float = 0.0  # accumulated


@dataclass
class Buyer:
    """
    Data Class representing a buyer / prediction task in the marketplace.
    Attributes:
        idx (int): Index of the buyer.
        y (NDArray[np.float64]): Target vector this buyer wants to predict.
        mu (float): Private valuation or unit gain of the buyer.
    """

    idx: int
    y: NDArray[np.float64]
    mu: float  # private value / unit gain

    def choose_bid(self) -> float:
        """
        In a truthful mechanism, the optimal strategy is to bid the true value, b = mu.
        """
        return self.mu
