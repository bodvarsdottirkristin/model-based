"""
Metrics and model-comparison utilities.

Functions for evaluating predictive performance (RMSE, MAE, R²) and for
comparing models using information criteria (WAIC, LOO-CV) via ArviZ.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import arviz as az
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

if TYPE_CHECKING:
    import pandas as pd


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Root Mean Squared Error.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground-truth target values.
    y_pred : array-like of shape (n_samples,)
        Predicted target values.

    Returns
    -------
    float
        RMSE value.
    """
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Mean Absolute Error.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground-truth target values.
    y_pred : array-like of shape (n_samples,)
        Predicted target values.

    Returns
    -------
    float
        MAE value.
    """
    return float(mean_absolute_error(y_true, y_pred))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the coefficient of determination R².

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground-truth target values.
    y_pred : array-like of shape (n_samples,)
        Predicted target values.

    Returns
    -------
    float
        R² score (1.0 is perfect prediction).
    """
    return float(r2_score(y_true, y_pred))


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute RMSE, MAE, and R² in a single call.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground-truth target values.
    y_pred : array-like of shape (n_samples,)
        Predicted target values.

    Returns
    -------
    metrics : dict
        Dictionary with keys ``'rmse'``, ``'mae'``, and ``'r2'``.
    """
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "r2": r2(y_true, y_pred),
    }


def compute_waic(trace: az.InferenceData) -> az.ELPDData:
    """Compute the Widely Applicable Information Criterion (WAIC).

    Parameters
    ----------
    trace : az.InferenceData
        Posterior trace that contains log-likelihood values.

    Returns
    -------
    az.ELPDData
        WAIC estimate with pointwise log-likelihood contributions.
    """
    return az.waic(trace)


def compute_loo(trace: az.InferenceData) -> az.ELPDData:
    """Compute Pareto-smoothed importance-sampling LOO cross-validation (LOO-CV).

    Parameters
    ----------
    trace : az.InferenceData
        Posterior trace that contains log-likelihood values.

    Returns
    -------
    az.ELPDData
        LOO estimate with Pareto-k diagnostic values.
    """
    return az.loo(trace)


def compare_models(traces: dict) -> "pd.DataFrame":
    """Compare multiple models using LOO-CV.

    Parameters
    ----------
    traces : dict
        Mapping of model name → :class:`az.InferenceData` object.
        Each trace must contain log-likelihood values.

    Returns
    -------
    comparison_df : pd.DataFrame
        ArviZ model-comparison DataFrame sorted by LOO score.
    """
    return az.compare(traces)
