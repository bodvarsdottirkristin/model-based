"""
Basic unit tests for the Bayesian model and utility modules.

These tests use lightweight synthetic data and deliberately keep the
MCMC sampler settings small (few draws, 1 chain) so that the suite runs
quickly in CI.
"""

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_regression_data():
    """Return a small, noise-free synthetic regression dataset.

    Returns
    -------
    X_train, X_test, y_train, y_test : tuple of np.ndarray
    """
    rng = np.random.default_rng(0)
    n_train, n_test, n_features = 30, 10, 2
    X_train = rng.standard_normal((n_train, n_features))
    true_beta = np.array([2.0, -1.5])
    y_train = X_train @ true_beta + rng.normal(0, 0.5, n_train)
    X_test = rng.standard_normal((n_test, n_features))
    y_test = X_test @ true_beta + rng.normal(0, 0.5, n_test)
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# data_loader tests
# ---------------------------------------------------------------------------


def test_handle_missing_values_mean():
    """handle_missing_values with strategy='mean' fills NaNs with column mean."""
    import pandas as pd
    from src.data_loader import handle_missing_values

    df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [np.nan, 2.0, 4.0]})
    result = handle_missing_values(df.copy(), strategy="mean")
    assert result.isnull().sum().sum() == 0
    assert result["a"].iloc[1] == pytest.approx(2.0)


def test_handle_missing_values_unknown_strategy():
    """handle_missing_values raises ValueError for an unknown strategy."""
    import pandas as pd
    from src.data_loader import handle_missing_values

    df = pd.DataFrame({"x": [1.0, np.nan]})
    with pytest.raises(ValueError, match="Unknown strategy"):
        handle_missing_values(df, strategy="mode")


def test_scale_features_shape():
    """scale_features returns arrays of the same shape as inputs."""
    from src.data_loader import scale_features

    rng = np.random.default_rng(1)
    X_train = rng.standard_normal((20, 3))
    X_test = rng.standard_normal((5, 3))
    X_tr_s, X_te_s, scaler = scale_features(X_train, X_test)
    assert X_tr_s.shape == X_train.shape
    assert X_te_s.shape == X_test.shape


def test_scale_features_zero_mean():
    """Scaled training features should have approximately zero mean."""
    from src.data_loader import scale_features

    rng = np.random.default_rng(2)
    X_train = rng.standard_normal((100, 4)) * 5 + 10
    X_test = rng.standard_normal((20, 4)) * 5 + 10
    X_tr_s, _, _ = scale_features(X_train, X_test)
    np.testing.assert_allclose(X_tr_s.mean(axis=0), 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# evaluation tests
# ---------------------------------------------------------------------------


def test_evaluate_predictions_perfect():
    """evaluate_predictions returns RMSE=0 and R²=1 for perfect predictions."""
    from src.evaluation import evaluate_predictions

    y = np.array([1.0, 2.0, 3.0, 4.0])
    metrics = evaluate_predictions(y, y)
    assert metrics["rmse"] == pytest.approx(0.0)
    assert metrics["mae"] == pytest.approx(0.0)
    assert metrics["r2"] == pytest.approx(1.0)


def test_rmse_known_value():
    """rmse returns the expected value for a known input."""
    from src.evaluation import rmse

    y_true = np.array([0.0, 0.0, 0.0, 0.0])
    y_pred = np.array([1.0, 1.0, 1.0, 1.0])
    assert rmse(y_true, y_pred) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# BayesianModel tests  (lightweight – small draws to keep runtime short)
# ---------------------------------------------------------------------------


def test_bayesian_model_fit_predict(synthetic_regression_data):
    """BayesianModel.fit() and predict() complete without error."""
    from src.model import BayesianModel

    X_train, X_test, y_train, _ = synthetic_regression_data
    model = BayesianModel(num_samples=50, tune=50, chains=1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    assert y_pred.shape == (X_test.shape[0],)


def test_bayesian_model_sample_posterior(synthetic_regression_data):
    """sample_posterior() returns an ArviZ InferenceData object."""
    import arviz as az
    from src.model import BayesianModel

    X_train, _, y_train, _ = synthetic_regression_data
    model = BayesianModel(num_samples=50, tune=50, chains=1)
    model.fit(X_train, y_train)
    trace = model.sample_posterior()
    assert isinstance(trace, az.InferenceData)


def test_bayesian_model_predict_before_fit():
    """predict() raises RuntimeError if called before fit()."""
    from src.model import BayesianModel

    model = BayesianModel()
    with pytest.raises(RuntimeError, match="fitted"):
        model.predict(np.zeros((5, 2)))


def test_bayesian_model_sample_posterior_before_fit():
    """sample_posterior() raises RuntimeError if called before fit()."""
    from src.model import BayesianModel

    model = BayesianModel()
    with pytest.raises(RuntimeError, match="fitted"):
        model.sample_posterior()
