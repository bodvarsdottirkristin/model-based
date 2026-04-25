"""
Model 2 — Add time of day

Non-homogeneous Poisson Process where λ(t) varies by hour of day. This captures the intra-day rhythm your instructor mentioned.
"""
import numpy as np
import torch
import pyro
import pyro.distributions as dist

from pyro.infer import MCMC, NUTS, HMC, SVI, Trace_ELBO
from pyro.contrib.autoguide import AutoDiagonalNormal, AutoMultivariateNormal
from pyro.optim import Adam, ClippedAdam

def hourly_poisson_model(hour_idx, y=None):
    n = len(hour_idx)

    # One log-rate per hour: 24 parameters
    log_rate_hour = pyro.sample(
        "log_rate_hour",
        dist.Normal(torch.zeros(24), 2.0 * torch.ones(24)).to_event(1)
    )

    rate = torch.exp(log_rate_hour[hour_idx])

    with pyro.plate("data", n):
        pyro.sample("obs", dist.Poisson(rate), obs=y)

def train_model(hour_train_torch, y_train_torch, n_steps=3000, lr=0.005):
    pyro.clear_param_store()

    guide = AutoDiagonalNormal(hourly_poisson_model)
    optimizer = ClippedAdam({"lr": lr})
    svi = SVI(hourly_poisson_model, guide, optimizer, loss=Trace_ELBO())

    for step in range(n_steps):
        loss = svi.step(hour_train_torch, y_train_torch)

        if step % 500 == 0:
            print(f"[{step}] ELBO loss: {loss:.2f}")

    return guide

def predict(guide, hour_test_torch, num_samples=1000):
    predictive = Predictive(
        hourly_poisson_model,
        guide=guide,
        num_samples=num_samples,
        return_sites=("log_rate_hour",)
    )

    samples = predictive(hour_test_torch, None)

    log_rate_hour_samples = samples["log_rate_hour"]   # shape: [S, 24]
    rate_hour_samples = torch.exp(log_rate_hour_samples)

    rate_pred_samples = rate_hour_samples[:, hour_test_torch]  # [S, N_test]
    y_pred = rate_pred_samples.mean(dim=0)

    return y_pred.detach().numpy()

def compute_error(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    denominator = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - np.sum((y_true - y_pred) ** 2) / denominator

    if np.isclose(np.std(y_pred), 0) or np.isclose(np.std(y_true), 0):
        corr = np.nan
    else:
        corr = np.corrcoef(y_true, y_pred)[0, 1]

    return corr, mae, rmse, r2

def hourly_model():
    y_train = np.load("data/processed/y_train.npy")
    y_test = np.load("data/processed/y_test.npy")

    # This must be an integer array with values 0,...,23
    hour_train = np.load("data/processed/hour_train.npy")
    hour_test = np.load("data/processed/hour_test.npy")

    y_train_torch = torch.tensor(y_train, dtype=torch.float32)
    hour_train_torch = torch.tensor(hour_train, dtype=torch.long)
    hour_test_torch = torch.tensor(hour_test, dtype=torch.long)

    y_test_np = y_test.astype(float)

    guide = train_model(hour_train_torch, y_train_torch)

    preds = predict(guide, hour_test_torch)

    corr, mae, rmse, r2 = compute_error(y_test_np, preds)

    print("\nHourly Poisson model")
    print(f"CorrCoef: {corr:.3f}")
    print(f"MAE: {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R2: {r2:.3f}")
