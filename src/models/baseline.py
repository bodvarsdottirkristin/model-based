"""
Model 1 — Baseline (simplest possible)

Homogeneous Poisson Process — crime arrives at a constant rate λ everywhere, all the time. This is your "dumb" baseline that everything else should beat.
"""
import numpy as np
import torch
import pyro
import pyro.distributions as dist

from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.contrib.autoguide import AutoDiagonalNormal
from pyro.optim import ClippedAdam


def constant_poisson_model(y=None):
    n = len(y)
    log_rate = pyro.sample("log_rate", dist.Normal(0., 1.))
    rate = torch.exp(log_rate)
    
    with pyro.plate("data", n):
        pyro.sample("obs", dist.Poisson(rate), obs=y)
        

def train_model(y_train_torch, n_steps=3000, lr=0.005):
    pyro.clear_param_store()
    
    guide = AutoDiagonalNormal(constant_poisson_model)
    optimizer = ClippedAdam({"lr": lr})
    svi = SVI(constant_poisson_model, guide, optimizer, loss=Trace_ELBO())
    
    for step in range(n_steps):
        loss = svi.step(y_train_torch)
        
        if step % 500 == 0:
            print({f"[{step}] ELBO loss: {loss:.2f}"})
            
    return guide

def predict(guide, n_test, num_samples=1000):
    predictive = Predictive(
        constant_poisson_model,
        guide=guide,
        num_samples=num_samples,
        return_sites=("log_rate",)
    )

    dummy_y = torch.zeros(n_test)
    samples = predictive(dummy_y)

    log_rate_samples = samples["log_rate"]
    rate_samples = torch.exp(log_rate_samples)

    y_pred = rate_samples.mean().repeat(n_test)

    return y_pred.detach().numpy()


def compute_error(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    denominator = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - np.sum((y_true - y_pred) ** 2) / denominator

    return mae, rmse, r2


def baseline_model():
    y_train = np.load("data/processed/y_train.npy")
    y_test = np.load("data/processed/y_test.npy")

    y_train_torch = torch.tensor(y_train, dtype=torch.float32)
    y_test_np = y_test.astype(float)

    guide = train_model(y_train_torch)

    preds = predict(guide, n_test=len(y_test_np))

    mae, rmse, r2 = compute_error(y_test_np, preds)

    print("\nBaseline Poisson model")
    print(f"Estimated constant rate prediction: {preds[0]:.3f}")
    print(f"CorrCoef: undefined for constant baseline")
    print(f"MAE: {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R2: {r2:.3f}")