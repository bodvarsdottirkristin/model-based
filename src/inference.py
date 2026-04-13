"""
Posterior inference utilities.

This module provides convenience wrappers around PyMC sampling routines,
supporting both MCMC (NUTS) and mean-field Variational Inference (ADVI).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pymc as pm
import arviz as az


def run_mcmc(
    model: pm.Model,
    draws: int = 1000,
    tune: int = 500,
    chains: int = 2,
    target_accept: float = 0.9,
    random_seed: int = 42,
) -> az.InferenceData:
    """Run NUTS MCMC sampling on a compiled PyMC model.

    Parameters
    ----------
    model : pm.Model
        A compiled (context-entered) PyMC model.
    draws : int, optional
        Number of posterior draws per chain (default 1000).
    tune : int, optional
        Number of tuning steps per chain (default 500).
    chains : int, optional
        Number of parallel chains (default 2).
    target_accept : float, optional
        Target Metropolis acceptance probability for the NUTS step-size
        adaptation (default 0.9).
    random_seed : int, optional
        Random seed for reproducibility (default 42).

    Returns
    -------
    trace : az.InferenceData
        ArviZ InferenceData object containing posterior samples.
    """
    with model:
        trace = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            nuts_sampler="pymc",
            target_accept=target_accept,
            random_seed=random_seed,
            progressbar=False,
        )
    return trace


def run_vi(
    model: pm.Model,
    n_iterations: int = 30_000,
    random_seed: int = 42,
) -> az.InferenceData:
    """Run mean-field ADVI variational inference on a compiled PyMC model.

    Parameters
    ----------
    model : pm.Model
        A compiled (context-entered) PyMC model.
    n_iterations : int, optional
        Maximum number of optimisation iterations (default 30 000).
    random_seed : int, optional
        Random seed for reproducibility (default 42).

    Returns
    -------
    trace : az.InferenceData
        ArviZ InferenceData object constructed from 2000 draws from the
        fitted variational approximation.
    """
    with model:
        approx = pm.fit(
            n=n_iterations,
            method="advi",
            random_seed=random_seed,
            progressbar=False,
        )
        trace = approx.sample(2000, random_seed=random_seed)
    return trace


def compute_diagnostics(trace: az.InferenceData) -> dict:
    """Compute common MCMC convergence diagnostics.

    Parameters
    ----------
    trace : az.InferenceData
        Posterior trace returned by :func:`run_mcmc` or :func:`run_vi`.

    Returns
    -------
    diagnostics : dict
        Dictionary with keys:

        - ``'summary'`` – ArviZ summary DataFrame.
        - ``'rhat_max'`` – Maximum R-hat value across all parameters.
        - ``'ess_bulk_min'`` – Minimum bulk ESS across all parameters.
        - ``'divergences'`` – Total number of divergent transitions
          (``None`` if the trace has no sampler statistics).
    """
    summary = az.summary(trace)
    diagnostics: dict = {
        "summary": summary,
        "rhat_max": float(summary["r_hat"].max()) if "r_hat" in summary.columns else None,
        "ess_bulk_min": (
            float(summary["ess_bulk"].min()) if "ess_bulk" in summary.columns else None
        ),
        "divergences": None,
    }

    # Count divergent transitions if sampler stats are present
    try:
        divergences = int(
            trace.sample_stats["diverging"].values.sum()
        )
        diagnostics["divergences"] = divergences
    except (AttributeError, KeyError):
        pass

    return diagnostics


def sample_posterior_predictive(
    model: pm.Model,
    trace: az.InferenceData,
    X_new: np.ndarray,
    var_names: Optional[list] = None,
    random_seed: int = 42,
) -> az.InferenceData:
    """Draw posterior predictive samples for new inputs.

    Parameters
    ----------
    model : pm.Model
        The compiled PyMC model (must have an ``"X"`` Data node).
    trace : az.InferenceData
        Posterior trace from MCMC or VI.
    X_new : np.ndarray of shape (n_samples, n_features)
        New feature matrix for which to generate predictions.
    var_names : list of str, optional
        Variable names to include in the PPC.  Defaults to all observed
        variables.
    random_seed : int, optional
        Random seed (default 42).

    Returns
    -------
    ppc : az.InferenceData
        Posterior predictive samples.
    """
    with model:
        pm.set_data({"X": np.asarray(X_new, dtype=float)})
        ppc = pm.sample_posterior_predictive(
            trace,
            var_names=var_names,
            random_seed=random_seed,
            progressbar=False,
        )
    return ppc
