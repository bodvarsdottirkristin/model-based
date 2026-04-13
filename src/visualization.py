"""
Plotting helpers for exploratory data analysis and model evaluation.

All functions return the active :class:`matplotlib.figure.Figure` so that
callers can either display them in a notebook or save them to disk.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
import pandas as pd


# ---------------------------------------------------------------------------
# EDA helpers
# ---------------------------------------------------------------------------


def plot_distribution(
    data: pd.Series,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot a histogram with a KDE overlay for a single variable.

    Parameters
    ----------
    data : pd.Series
        The data to plot.
    title : str, optional
        Figure title.
    save_path : str, optional
        If provided, the figure is saved to this path.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(data, kde=True, ax=ax)
    ax.set_xlabel(data.name if data.name else "Value")
    ax.set_ylabel("Count")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_correlation_heatmap(
    df: pd.DataFrame,
    title: str = "Correlation Matrix",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot a heatmap of pairwise Pearson correlations.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame whose numeric columns are used.
    title : str, optional
        Figure title (default ``'Correlation Matrix'``).
    save_path : str, optional
        If provided, the figure is saved to this path.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    corr = df.select_dtypes(include=[np.number]).corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_pairplot(
    df: pd.DataFrame,
    hue: Optional[str] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Create a seaborn pairplot for exploratory analysis.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to plot.
    hue : str, optional
        Column name to use for colour encoding.
    save_path : str, optional
        If provided, the figure is saved to this path.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    pair_grid = sns.pairplot(df, hue=hue)
    fig = pair_grid.figure
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


# ---------------------------------------------------------------------------
# Prediction / residual helpers
# ---------------------------------------------------------------------------


def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Predicted vs. Actual",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Scatter plot of predicted versus actual values.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground-truth targets.
    y_pred : array-like of shape (n_samples,)
        Model predictions.
    title : str, optional
        Figure title.
    save_path : str, optional
        If provided, the figure is saved to this path.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.6, edgecolors="k", linewidths=0.3)
    ax.plot(lims, lims, "r--", label="Perfect prediction")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Residuals",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot residuals (y_true - y_pred) as a function of predicted values.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground-truth targets.
    y_pred : array-like of shape (n_samples,)
        Model predictions.
    title : str, optional
        Figure title.
    save_path : str, optional
        If provided, the figure is saved to this path.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    residuals = y_true - y_pred

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(y_pred, residuals, alpha=0.6, edgecolors="k", linewidths=0.3)
    ax.axhline(0, color="r", linestyle="--")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual")
    ax.set_title(title)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


# ---------------------------------------------------------------------------
# Posterior / trace helpers
# ---------------------------------------------------------------------------


def plot_posterior(
    trace: az.InferenceData,
    var_names: Optional[Sequence[str]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot marginal posterior distributions using ArviZ.

    Parameters
    ----------
    trace : az.InferenceData
        Posterior trace.
    var_names : sequence of str, optional
        Variables to include.  If ``None``, all parameters are shown.
    save_path : str, optional
        If provided, the figure is saved to this path.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    axes = az.plot_posterior(trace, var_names=var_names)
    fig = axes.ravel()[0].get_figure()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_trace(
    trace: az.InferenceData,
    var_names: Optional[Sequence[str]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot MCMC trace plots (sample values over iterations) using ArviZ.

    Parameters
    ----------
    trace : az.InferenceData
        Posterior trace.
    var_names : sequence of str, optional
        Variables to include.  If ``None``, all parameters are shown.
    save_path : str, optional
        If provided, the figure is saved to this path.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    axes = az.plot_trace(trace, var_names=var_names)
    fig = axes.ravel()[0].get_figure()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def save_figure(fig: plt.Figure, path: str, dpi: int = 150) -> None:
    """Save a matplotlib figure to disk, creating parent directories as needed.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save.
    path : str
        Destination file path (PNG, PDF, SVG, etc.).
    dpi : int, optional
        Resolution in dots per inch (default 150).
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
