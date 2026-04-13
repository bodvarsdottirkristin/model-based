"""
Data loading and preprocessing utilities.

This module provides helper functions for reading raw data from disk,
splitting it into train/test sets, and applying common preprocessing
transformations (scaling, encoding, missing-value imputation).
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_csv(filepath: str) -> pd.DataFrame:
    """Load a CSV file into a pandas DataFrame.

    Parameters
    ----------
    filepath : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        The loaded DataFrame.
    """
    return pd.read_csv(filepath)


def split_features_target(df: pd.DataFrame, target_col: str):
    """Split a DataFrame into feature matrix X and target vector y.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing both features and the target column.
    target_col : str
        Name of the column to use as the prediction target.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix (all columns except *target_col*).
    y : pd.Series
        Target vector.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def train_test_split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Split features and target into train/test sets.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target vector.
    test_size : float, optional
        Proportion of data to use for the test set (default 0.2).
    random_state : int, optional
        Random seed for reproducibility (default 42).

    Returns
    -------
    X_train, X_test, y_train, y_test : tuple
        Train and test splits of features and target.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def scale_features(X_train: np.ndarray, X_test: np.ndarray):
    """Standardise features using statistics computed on the training set.

    Parameters
    ----------
    X_train : array-like of shape (n_train, n_features)
        Training feature matrix.
    X_test : array-like of shape (n_test, n_features)
        Test feature matrix.

    Returns
    -------
    X_train_scaled : np.ndarray
        Scaled training features.
    X_test_scaled : np.ndarray
        Scaled test features (using training-set statistics).
    scaler : StandardScaler
        Fitted scaler object (can be used to inverse-transform predictions).
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def handle_missing_values(df: pd.DataFrame, strategy: str = "mean") -> pd.DataFrame:
    """Impute missing values in numeric columns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame that may contain NaN values.
    strategy : str, optional
        Imputation strategy – ``'mean'``, ``'median'``, or ``'zero'``
        (default ``'mean'``).

    Returns
    -------
    pd.DataFrame
        DataFrame with missing values filled in.

    Raises
    ------
    ValueError
        If *strategy* is not one of the supported options.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if strategy == "mean":
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    elif strategy == "median":
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    elif strategy == "zero":
        df[numeric_cols] = df[numeric_cols].fillna(0)
    else:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose 'mean', 'median', or 'zero'.")
    return df
