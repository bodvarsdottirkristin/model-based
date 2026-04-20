"""
Model 1 — Baseline (simplest possible)

Homogeneous Poisson Process — crime arrives at a constant rate λ everywhere, all the time. This is your "dumb" baseline that everything else should beat.
"""
import pyro
import pyro.distributions as dist

def model(T, y_obs=None):
    # Prior on crime rate (crimes per hour)
    lam = pyro.sample("lambda", dist.Gamma(2.0, 0.1))
    
    # Likelihood
    with pyro.plate("data", len(y_obs)):
        pyro.sample("y", dist.Poisson(lam * T), obs=y_obs)
        
