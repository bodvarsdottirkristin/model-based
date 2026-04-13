"""
Bayesian model definition.

This module defines the :class:`BayesianModel` class, which wraps a
PyMC-based Bayesian linear regression model.  The class exposes a
scikit-learn-style API (``fit`` / ``predict``) together with a
``sample_posterior`` method for full posterior inference.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
import pymc as pm
import arviz as az

if TYPE_CHECKING:
    import pandas as pd


class BayesianModel:
    """Bayesian linear regression model implemented with PyMC.

    The generative model is::

        alpha  ~ Normal(0, 10)          # intercept
        beta   ~ Normal(0, 1)           # coefficients (one per feature)
        sigma  ~ HalfNormal(1)          # observation noise
        mu     = alpha + X @ beta
        y      ~ Normal(mu, sigma)

    Parameters
    ----------
    num_samples : int, optional
        Number of posterior samples to draw during MCMC (default 1000).
    tune : int, optional
        Number of tuning (warm-up) steps for NUTS (default 500).
    chains : int, optional
        Number of independent Markov chains (default 2).
    random_seed : int, optional
        Random seed for reproducibility (default 42).
    """

    def __init__(
        self,
        num_samples: int = 1000,
        tune: int = 500,
        chains: int = 2,
        random_seed: int = 42,
    ) -> None:
        self.num_samples = num_samples
        self.tune = tune
        self.chains = chains
        self.random_seed = random_seed

        self._model: Optional[pm.Model] = None
        self._trace: Optional[az.InferenceData] = None
        self._X_train: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BayesianModel":
        """Build the PyMC model and draw posterior samples via NUTS.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training feature matrix.
        y : np.ndarray of shape (n_samples,)
            Training target vector.

        Returns
        -------
        self : BayesianModel
            The fitted model (for method chaining).
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n_features = X.shape[1] if X.ndim > 1 else 1
        self._X_train = X

        with pm.Model() as self._model:
            # Data containers
            X_data = pm.Data("X", X)

            # Priors
            alpha = pm.Normal("alpha", mu=0, sigma=10)
            beta = pm.Normal("beta", mu=0, sigma=1, shape=n_features)
            sigma = pm.HalfNormal("sigma", sigma=1)

            # Likelihood
            mu = alpha + pm.math.dot(X_data, beta)
            pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)

            # Sample
            self._trace = pm.sample(
                draws=self.num_samples,
                tune=self.tune,
                chains=self.chains,
                random_seed=self.random_seed,
                progressbar=False,
            )

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return posterior predictive mean for new inputs.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Feature matrix for which to generate predictions.

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            Predicted mean values.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called yet.
        """
        if self._model is None or self._trace is None:
            raise RuntimeError("Model must be fitted before calling predict().")

        X = np.asarray(X, dtype=float)

        with self._model:
            pm.set_data({"X": X})
            ppc = pm.sample_posterior_predictive(
                self._trace,
                var_names=["y_obs"],
                random_seed=self.random_seed,
                progressbar=False,
            )

        # Average over posterior samples and chains
        y_pred = ppc.posterior_predictive["y_obs"].values.mean(axis=(0, 1))
        return y_pred

    def sample_posterior(self) -> az.InferenceData:
        """Return the full posterior trace as an ArviZ InferenceData object.

        Returns
        -------
        trace : az.InferenceData
            The posterior trace containing samples for all model parameters.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called yet.
        """
        if self._trace is None:
            raise RuntimeError("Model must be fitted before calling sample_posterior().")
        return self._trace

    # ------------------------------------------------------------------
    # Diagnostics helpers
    # ------------------------------------------------------------------

    def summary(self) -> "pd.DataFrame":
        """Print and return an ArviZ summary of the posterior.

        Returns
        -------
        summary_df : pd.DataFrame
            Summary statistics (mean, sd, HDI, R-hat, ESS) for each parameter.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called yet.
        """
        if self._trace is None:
            raise RuntimeError("Model must be fitted before calling summary().")
        return az.summary(self._trace)
