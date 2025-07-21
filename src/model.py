"""
Implements the MLModel class, which is a wrapper for the learning algorithm M.
"""

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def rmse(y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
    """Root-mean-square error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def gain(y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
    """
    Normalised prediction gain G in [0,1].
    Definition 2.1 (1 - RMSE / (y_max - y_min)).
    The gain is computed based on the provided y_true subset (e.g., a test set).
    """
    y_max, y_min = float(np.max(y_true)), float(np.min(y_true))
    normaliser = max(y_max - y_min, 1e-8)  # avoid 0-division
    return max(0.0, 1.0 - rmse(y_true, y_pred) / normaliser)


class MLModel:
    """
    Fixed learning algorithm M.
    Ridge regression wrapped in a scikit-learn pipeline for normalisation.
    """

    def __init__(self):
        self._model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))

    def fit_predict(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Use 70% as train, 30% as test split for simplicity.
        Returns the true and predicted values ONLY for the test set (y_test, y_test_hat).
        """
        # A check for edge cases where X might be empty
        if X.shape[0] == 0:
            T = len(y)
            cutoff = int(0.7 * T)
            y_test = y[cutoff:]
            return y_test, np.zeros_like(y_test)

        T = X.shape[1]
        cutoff = int(0.7 * T)
        X_train, X_test = X[:, :cutoff].T, X[:, cutoff:].T
        y_train, y_test = y[:cutoff], y[cutoff:]

        self._model.fit(X_train, y_train)
        y_test_hat = self._model.predict(X_test)

        return y_test, y_test_hat
