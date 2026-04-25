"""
Generates the project notebook: bayesian_demand_forecasting.ipynb
Run with: uv run python create_notebook.py
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(src): return nbf.v4.new_markdown_cell(src)
def code(src): return nbf.v4.new_code_cell(src)

# ─────────────────────────────────────────────────────────────
# SECTION 0 — Title
# ─────────────────────────────────────────────────────────────
cells.append(md("""\
# Bayesian Demand Forecasting & Uncertainty-Aware Inventory Optimisation
## Model-based Machine Learning — DTU Spring 2026

---

### Motivation

Retailers must decide how much stock to order **before** knowing actual demand.
Classical forecasting (e.g. ARIMA, moving averages) returns a *point estimate* of
demand — but inventory decisions depend on the **full probability distribution**.

* **Understocking** → lost sales, unhappy customers
* **Overstocking** → waste, holding costs, markdown losses

A Bayesian approach gives us the complete posterior predictive distribution of
demand, which we can feed directly into the classic **Newsvendor optimisation**
to derive the provably optimal order quantity for any cost structure.

### What we build

| Step | Method | Purpose |
|---|---|---|
| 1 | Hierarchical Bayesian NegBinomial | Learn demand distributions with partial pooling |
| 2 | Variational Inference (SVI) | Fast approximate posterior |
| 3 | MCMC (NUTS) | Gold-standard posterior — used for comparison |
| 4 | Newsvendor optimisation | Turn posteriors into optimal order quantities |
| 5 | Cost analysis | Quantify savings vs. naive mean forecast |

### Project type
*Problem-driven* — the PGM is designed around the supply-chain inventory problem.
"""))

# ─────────────────────────────────────────────────────────────
# SECTION 1 — Imports
# ─────────────────────────────────────────────────────────────
cells.append(md("## 1. Setup"))

cells.append(code("""\
# !pip install pyro-ppl  # uncomment if needed

import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO, MCMC, NUTS, Predictive
from pyro.infer.autoguide import AutoNormal
from pyro.optim import Adam

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

torch.manual_seed(42)
pyro.set_rng_seed(42)
np.random.seed(42)

plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (13, 4)
plt.rcParams['font.size'] = 11

print(f"Pyro {pyro.__version__} | PyTorch {torch.__version__}")
"""))

# ─────────────────────────────────────────────────────────────
# SECTION 2 — Data Generation
# ─────────────────────────────────────────────────────────────
cells.append(md("""\
## 2. Synthetic Data

We generate daily sales data for a retail store with three product categories
and five products per category (15 products total).

Using **synthetic data** serves two purposes:
1. We know the **true parameters**, so we can validate our inference later.
2. We have full control over the data-generating process, making the PGM
   assumptions transparent.

### Data-generating process

$$y_{i} \\sim \\text{NegBinomial}\\!\\left(r,\\; \\text{logits}=\\mu_{j(i)} + s_{d(i)} - \\log r\\right)$$

where $j(i)$ is the product and $d(i)$ is the day-of-week for observation $i$,
$\\mu_j$ is the product log-demand, $s_d$ the day-of-week seasonal effect,
and $r$ the overdispersion parameter.

The **mean** of this NegBinomial is $\\exp(\\mu_{j(i)} + s_{d(i)})$, so everything
is on the log scale — convenient for a hierarchical prior.
"""))

cells.append(code("""\
# ── Configuration ────────────────────────────────────────────
N_CATEGORIES  = 3
N_PER_CAT     = 5
N_PRODUCTS    = N_CATEGORIES * N_PER_CAT   # 15
N_TRAIN       = 365
N_TEST        = 14
N_TOTAL       = N_TRAIN + N_TEST

CATEGORY_NAMES = ['Food', 'Clothing', 'Electronics']
PRODUCT_NAMES  = [f'{CATEGORY_NAMES[c]} #{i+1}'
                  for c in range(N_CATEGORIES) for i in range(N_PER_CAT)]

# ── True (latent) parameters ─────────────────────────────────
torch.manual_seed(0)

TRUE_CAT_MU    = torch.tensor([4.0, 3.5, 3.8])   # log-demand by category
TRUE_CAT_SIGMA = torch.tensor([0.3, 0.4, 0.3])

# product → category mapping
category_ids = torch.tensor([c for c in range(N_CATEGORIES)
                              for _ in range(N_PER_CAT)])

# product log-demands drawn from category priors
TRUE_PROD_MU = dist.Normal(TRUE_CAT_MU[category_ids],
                            TRUE_CAT_SIGMA[category_ids]).sample()

# weekly seasonality (Mon=0 … Sun=6), centred on 0 in log-space
TRUE_SEASON_RAW = torch.tensor([-0.15, -0.10, 0.00, 0.05, 0.10, 0.25, 0.20])
TRUE_SEASON     = TRUE_SEASON_RAW - TRUE_SEASON_RAW.mean()

TRUE_R = torch.tensor(5.0)   # overdispersion

# ── Generate observations ─────────────────────────────────────
product_ids_all = torch.arange(N_PRODUCTS).repeat_interleave(N_TOTAL)
day_ids_all     = torch.arange(N_TOTAL).repeat(N_PRODUCTS)
dow_ids_all     = day_ids_all % 7

log_rate_all = TRUE_PROD_MU[product_ids_all] + TRUE_SEASON[dow_ids_all]
sales_all    = dist.NegativeBinomial(TRUE_R,
                logits=log_rate_all - TRUE_R.log()).sample()

sales_matrix = sales_all.reshape(N_PRODUCTS, N_TOTAL)
sales_train  = sales_matrix[:, :N_TRAIN]
sales_test   = sales_matrix[:, N_TRAIN:]

# Flat training tensors (used by Pyro models)
product_ids_train = torch.arange(N_PRODUCTS).repeat_interleave(N_TRAIN)
dow_ids_train     = torch.arange(N_TRAIN).repeat(N_PRODUCTS) % 7
sales_train_flat  = sales_train.reshape(-1).float()

print(f'Sales matrix: {sales_matrix.shape}  '
      f'(products × days)')
print(f'Mean daily demand per category:')
for c, name in enumerate(CATEGORY_NAMES):
    idx = slice(c*N_PER_CAT, (c+1)*N_PER_CAT)
    print(f'  {name:12s}  {sales_train[idx].float().mean():.1f} units/day')
print(f'\\nTrue seasonal effects: {TRUE_SEASON.numpy().round(2)}')
"""))

# EDA
cells.append(code("""\
fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharey=False)

for c, (name, ax) in enumerate(zip(CATEGORY_NAMES, axes)):
    idx = range(c*N_PER_CAT, (c+1)*N_PER_CAT)
    for p in idx:
        ax.plot(sales_matrix[p, :N_TRAIN].numpy(), alpha=0.6, lw=0.9)
    ax.set_title(f'{name}', fontsize=13)
    ax.set_xlabel('Day')
    ax.set_ylabel('Units sold')

plt.suptitle('Daily Sales by Category (Training Data, 365 days)', fontsize=14, y=1.01)
plt.tight_layout()
plt.show()

# ── Day-of-week pattern ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
DOW = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

for c, name in enumerate(CATEGORY_NAMES):
    idx = range(c*N_PER_CAT, (c+1)*N_PER_CAT)
    weekly = [sales_train[list(idx), d::7].float().mean().item() for d in range(7)]
    ax.plot(weekly, marker='o', label=name, lw=2)

ax.set_xticks(range(7)); ax.set_xticklabels(DOW)
ax.set_title('Average Sales by Day of Week'); ax.set_ylabel('Mean units/day')
ax.legend(); plt.tight_layout(); plt.show()
"""))

# ─────────────────────────────────────────────────────────────
# SECTION 3 — PGM Description
# ─────────────────────────────────────────────────────────────
cells.append(md("""\
## 3. Probabilistic Graphical Model

### Generative process

We build a three-level hierarchy:

$$
\\begin{aligned}
&\\textbf{Global}\\\\
&\\quad \\mu_\\text{global} \\sim \\mathcal{N}(3.5,\\, 1.0)\\\\
&\\quad \\sigma_\\text{global} \\sim \\text{HalfNormal}(1.0)\\\\[4pt]
&\\textbf{Category level} \\quad k = 1 \\dots K\\\\
&\\quad \\mu_k^{\\text{cat}} \\sim \\mathcal{N}(\\mu_\\text{global},\\, \\sigma_\\text{global})\\\\
&\\quad \\sigma_k^{\\text{cat}} \\sim \\text{HalfNormal}(0.5)\\\\[4pt]
&\\textbf{Product level} \\quad j = 1 \\dots J\\\\
&\\quad \\mu_j^{\\text{prod}} \\sim \\mathcal{N}(\\mu_{k(j)}^{\\text{cat}},\\, \\sigma_{k(j)}^{\\text{cat}})\\\\[4pt]
&\\textbf{Seasonality}\\\\
&\\quad s_d^{\\text{raw}} \\sim \\mathcal{N}(0, 1) \\quad d=0\\dots 6\\\\
&\\quad s_d = s_d^{\\text{raw}} - \\bar{s}^{\\text{raw}} \\quad\\text{(identifiability)}\\\\[4pt]
&\\textbf{Overdispersion}\\\\
&\\quad r \\sim \\text{Gamma}(2, 0.5)\\quad\\text{(mean = 4)}\\\\[4pt]
&\\textbf{Likelihood} \\quad i = 1 \\dots N\\\\
&\\quad y_i \\sim \\text{NegBinomial}\\!\\left(r,\\; \\text{logits}=\\mu_{j(i)}^{\\text{prod}} + s_{d(i)} - \\log r\\right)
\\end{aligned}
$$

### Why NegBinomial?

Real retail demand is **count data** (non-negative integers) with
**overdispersion** — variance exceeds the mean. The NegBinomial handles
both via the parameter $r$: as $r \\to \\infty$, it converges to Poisson.

### Why hierarchical?

**Partial pooling** balances two extremes:
* *No pooling*: each product is estimated independently → high variance for
  slow movers with few observations.
* *Complete pooling*: all products share one estimate → ignores product
  heterogeneity.

With partial pooling, products *borrow statistical strength* from their
category while still having individual estimates.

### Why NegBinomial logits parameterisation?

In PyTorch/Pyro, `NegBinomial(total_count=r, logits=l)` has:

$$\\text{mean} = r \\cdot \\exp(l)$$

So setting $l = \\mu_j^{\\text{prod}} + s_d - \\log r$ gives $\\text{mean} = e^{\\mu_j^{\\text{prod}}+s_d}$,
which is exactly what we want — the log-scale additive structure.
"""))

# ─────────────────────────────────────────────────────────────
# SECTION 4 — Models
# ─────────────────────────────────────────────────────────────
cells.append(md("## 4. Model Definitions"))

cells.append(code("""\
# ── Model 1: Flat (no hierarchy, no partial pooling) ─────────
#
# We use .to_event(1) for multi-dimensional parameters so that
# AutoNormal creates one variational mean/scale per element while
# avoiding nested-plate dimension conflicts.

def model_flat(product_ids, dow_ids, n_products, sales_flat=None):
    \"\"\"
    Baseline Bayesian model: each product gets its own independent
    log-demand parameter. No pooling across products or categories.
    \"\"\"
    N = len(product_ids)

    # Independent product log-demands  shape: (n_products,)
    prod_mu = pyro.sample('prod_mu',
                          dist.Normal(3.5 * torch.ones(n_products),
                                      torch.ones(n_products)).to_event(1))

    # Shared weekly seasonality  shape: (7,)
    season_raw = pyro.sample('season_raw',
                             dist.Normal(torch.zeros(7),
                                        torch.ones(7)).to_event(1))
    season = season_raw - season_raw.mean()  # identifiability

    # Shared overdispersion  shape: scalar
    r = pyro.sample('r', dist.Gamma(torch.tensor(2.0), torch.tensor(0.5)))

    # Likelihood  shape: (N,)
    log_rate = prod_mu[product_ids] + season[dow_ids]
    with pyro.plate('observations', N):
        pyro.sample('sales',
                    dist.NegativeBinomial(r, logits=log_rate - r.log()),
                    obs=sales_flat)


# ── Model 2: Hierarchical (our main model) ───────────────────

def model_hierarchical(product_ids, category_ids, dow_ids,
                        n_products, n_categories, sales_flat=None):
    \"\"\"
    Hierarchical Bayesian NegBinomial model with three levels:
    global  →  category  →  product.

    Products within the same category share a common prior,
    enabling partial pooling and improving estimates.

    All multi-dim parameters use .to_event(1) so AutoNormal creates
    independent variational parameters per element without nested plates.
    \"\"\"
    N = len(product_ids)

    # ── Level 1: Global hyperpriors  shape: scalar ───────────
    global_mu    = pyro.sample('global_mu',
                               dist.Normal(torch.tensor(3.5),
                                           torch.tensor(1.0)))
    global_sigma = pyro.sample('global_sigma',
                               dist.HalfNormal(torch.tensor(1.0)))

    # ── Level 2: Category-level  shape: (n_categories,) ──────
    cat_mu = pyro.sample('cat_mu',
                         dist.Normal(global_mu  * torch.ones(n_categories),
                                     global_sigma * torch.ones(n_categories)
                                     ).to_event(1))
    cat_sigma = pyro.sample('cat_sigma',
                            dist.HalfNormal(0.5 * torch.ones(n_categories)
                                            ).to_event(1))

    # ── Level 3: Product-level  shape: (n_products,) ─────────
    # Each product j draws its log-demand from its category prior
    prod_mu = pyro.sample('prod_mu',
                          dist.Normal(cat_mu[category_ids],
                                      cat_sigma[category_ids]).to_event(1))

    # ── Seasonality  shape: (7,) ─────────────────────────────
    season_raw = pyro.sample('season_raw',
                             dist.Normal(torch.zeros(7),
                                        torch.ones(7)).to_event(1))
    season = season_raw - season_raw.mean()   # zero-sum constraint

    # ── Overdispersion  shape: scalar ────────────────────────
    r = pyro.sample('r', dist.Gamma(torch.tensor(2.0), torch.tensor(0.5)))

    # ── Likelihood  shape: (N,) ───────────────────────────────
    # Mean of NegBinomial = exp(log_rate)
    # NegBinomial(r, logits=l) has mean = r * exp(l)
    # so logits = log_rate - log(r)  gives  mean = exp(log_rate)
    log_rate = prod_mu[product_ids] + season[dow_ids]
    with pyro.plate('observations', N):
        pyro.sample('sales',
                    dist.NegativeBinomial(r, logits=log_rate - r.log()),
                    obs=sales_flat)


print('Models defined.')
"""))

# ─────────────────────────────────────────────────────────────
# SECTION 5 — Ancestral Sampling Validation
# ─────────────────────────────────────────────────────────────
cells.append(md("""\
## 5. Model Validation via Ancestral Sampling

Before fitting to real data we **validate our model implementation**:

1. Fix the true parameters and generate synthetic observations from the model.
2. Run inference on those observations.
3. Check that the inferred posteriors recover the true parameters.

If inference cannot recover known parameters, the model is mis-specified
or the inference algorithm is not working correctly.
"""))

cells.append(code("""\
# ── Step 1: generate small dataset from known parameters ─────
N_VAL_PRODUCTS  = 6    # 2 per category
N_VAL_CATS      = 3
N_VAL_DAYS      = 200

torch.manual_seed(99)
val_cat_ids   = torch.tensor([0, 0, 1, 1, 2, 2])
val_true_mu   = dist.Normal(TRUE_CAT_MU[val_cat_ids],
                             TRUE_CAT_SIGMA[val_cat_ids]).sample()
val_true_r    = torch.tensor(4.0)
val_true_sea  = TRUE_SEASON

val_prod_ids  = torch.arange(N_VAL_PRODUCTS).repeat_interleave(N_VAL_DAYS)
val_dow_ids   = torch.arange(N_VAL_DAYS).repeat(N_VAL_PRODUCTS) % 7
val_log_rate  = val_true_mu[val_prod_ids] + val_true_sea[val_dow_ids]
val_sales     = dist.NegativeBinomial(val_true_r,
                logits=val_log_rate - val_true_r.log()).sample().float()

# ── Step 2: run SVI on the small dataset ─────────────────────
pyro.clear_param_store()
pyro.set_rng_seed(7)

val_guide = AutoNormal(model_hierarchical)
val_svi   = SVI(model_hierarchical, val_guide,
                Adam({'lr': 0.02}), loss=Trace_ELBO())

for step in range(2000):
    val_svi.step(val_prod_ids, val_cat_ids, val_dow_ids,
                 N_VAL_PRODUCTS, N_VAL_CATS, sales_flat=val_sales)

# ── Step 3: compare inferred prod_mu to true values ──────────
val_predictive = Predictive(model_hierarchical, guide=val_guide,
                            num_samples=1000,
                            return_sites=['prod_mu', 'r'])
val_samples = val_predictive(val_prod_ids, val_cat_ids, val_dow_ids,
                             N_VAL_PRODUCTS, N_VAL_CATS)

# reshape to 2-D: (n_samples, n_products) regardless of extra dims
inferred_mu = val_samples['prod_mu'].reshape(1000, N_VAL_PRODUCTS)
inferred_r  = val_samples['r'].reshape(-1)

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

# prod_mu recovery
ax = axes[0]
means = inferred_mu.mean(dim=0).detach().numpy().flatten()
lo    = inferred_mu.quantile(0.05, dim=0).detach().numpy().flatten()
hi    = inferred_mu.quantile(0.95, dim=0).detach().numpy().flatten()
x     = range(N_VAL_PRODUCTS)

ax.errorbar(x, means, yerr=[means-lo, hi-means], fmt='o',
            capsize=4, label='Posterior median ± 90% CI', color='steelblue')
ax.scatter(x, val_true_mu.numpy(), color='red', zorder=5,
           label='True value', marker='*', s=120)
ax.set_xlabel('Product'); ax.set_ylabel('log-demand (μ)')
ax.set_title('Parameter Recovery: Product Log-Demand')
ax.legend()

# r recovery
ax = axes[1]
ax.hist(inferred_r.detach().numpy(), bins=40, color='steelblue',
        alpha=0.7, density=True, label='Posterior of r')
ax.axvline(val_true_r.item(), color='red', lw=2,
           label=f'True r = {val_true_r:.1f}')
ax.set_xlabel('Overdispersion r'); ax.set_ylabel('Density')
ax.set_title('Parameter Recovery: Overdispersion')
ax.legend()

plt.tight_layout()
plt.suptitle('Ancestral Sampling Validation', y=1.02, fontsize=13)
plt.show()

print('True  prod_mu:', val_true_mu.numpy().round(2))
print('Inferred mean:', means.round(2))
"""))

# ─────────────────────────────────────────────────────────────
# SECTION 6 — SVI Inference
# ─────────────────────────────────────────────────────────────
cells.append(md("""\
## 6. Variational Inference (SVI)

We use **Stochastic Variational Inference (SVI)** with an `AutoNormal` guide,
which approximates the posterior with a mean-field Normal distribution
(a fully-factored Gaussian — one per latent variable).

**Why VI here?** With 15 products × 365 days = 5475 observations and ~30
latent variables, VI converges in seconds. We later compare to MCMC (NUTS)
to validate the approximation.
"""))

cells.append(code("""\
def run_svi(model, model_args, n_steps=2000, lr=0.01, seed=42):
    \"\"\"Run SVI and return (guide, loss_history).\"\"\"
    pyro.clear_param_store()
    pyro.set_rng_seed(seed)
    guide = AutoNormal(model)
    svi   = SVI(model, guide, Adam({'lr': lr}), loss=Trace_ELBO())
    losses = []
    for step in range(n_steps):
        loss = svi.step(*model_args)
        losses.append(loss)
        if step % 500 == 0:
            print(f'  step {step:4d}  ELBO = {-loss:,.0f}')
    return guide, losses


# ── Flat model ───────────────────────────────────────────────
print('Training Model 1 (flat):')
flat_args  = (product_ids_train, dow_ids_train, N_PRODUCTS, sales_train_flat)
guide_flat, losses_flat = run_svi(model_flat, flat_args)

# ── Hierarchical model ───────────────────────────────────────
print('\\nTraining Model 2 (hierarchical):')
hier_args  = (product_ids_train, category_ids, dow_ids_train,
              N_PRODUCTS, N_CATEGORIES, sales_train_flat)
guide_hier, losses_hier = run_svi(model_hierarchical, hier_args)
"""))

cells.append(code("""\
# ── ELBO convergence plot ────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(losses_flat,  alpha=0.8, label='Model 1 (flat)')
ax.plot(losses_hier, alpha=0.8, label='Model 2 (hierarchical)')
ax.set_xlabel('SVI step')
ax.set_ylabel('ELBO loss  (lower = better)')
ax.set_title('SVI Convergence')
ax.legend()
plt.tight_layout()
plt.show()
"""))

# ─────────────────────────────────────────────────────────────
# SECTION 7 — Posterior Analysis
# ─────────────────────────────────────────────────────────────
cells.append(md("""\
## 7. Posterior Analysis

We draw samples from the posterior and compare the inferred latent
variables to the **known true values** (available because we generated
the data synthetically).

Key things to inspect:
* Do the posterior credible intervals **cover the true values**?
* Does the hierarchical model produce **tighter** intervals (lower variance)
  than the flat model for products within the same category?
"""))

cells.append(code("""\
N_POSTERIOR = 2000

# ── Posterior samples for both models ───────────────────────
post_flat = Predictive(model_flat, guide=guide_flat,
                       num_samples=N_POSTERIOR,
                       return_sites=['prod_mu', 'r', 'season_raw'])
samples_flat = post_flat(product_ids_train, dow_ids_train, N_PRODUCTS)

post_hier = Predictive(model_hierarchical, guide=guide_hier,
                       num_samples=N_POSTERIOR,
                       return_sites=['prod_mu', 'r', 'season_raw',
                                     'cat_mu', 'global_mu'])
samples_hier = post_hier(product_ids_train, category_ids, dow_ids_train,
                         N_PRODUCTS, N_CATEGORIES)

# ── Plot: product log-demands ────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=True)

for ax, samples, title in zip(
        axes,
        [samples_flat, samples_hier],
        ['Model 1 — Flat (no pooling)', 'Model 2 — Hierarchical (partial pooling)']):

    mu_post  = samples['prod_mu'].reshape(N_POSTERIOR, N_PRODUCTS)
    means    = mu_post.mean(dim=0).detach().numpy().flatten()
    lo       = mu_post.quantile(0.05, dim=0).detach().numpy().flatten()
    hi       = mu_post.quantile(0.95, dim=0).detach().numpy().flatten()

    colors   = ['tab:green']*5 + ['tab:orange']*5 + ['tab:blue']*5
    x        = range(N_PRODUCTS)

    ax.errorbar(x, means, yerr=[means-lo, hi-means],
                fmt='none', ecolor='grey', capsize=3, alpha=0.7)
    for xi, m, c in zip(x, means, colors):
        ax.scatter(xi, m, color=c, zorder=5, s=60)
    ax.scatter(x, TRUE_PROD_MU.numpy(), color='red',
               marker='*', s=130, zorder=6, label='True value')
    ax.set_xlabel('Product (0-4: Food, 5-9: Clothing, 10-14: Electronics)')
    ax.set_ylabel('log-demand μ')
    ax.set_title(title)
    ax.legend()

    # category background shading
    for c, col in enumerate(['tab:green','tab:orange','tab:blue']):
        ax.axvspan(c*5-0.5, c*5+4.5, alpha=0.05, color=col)

plt.suptitle('Posterior of Product Log-Demands', fontsize=13, y=1.01)
plt.tight_layout()
plt.show()
"""))

cells.append(code("""\
# ── Posterior interval width comparison (partial pooling benefit) ─
mu_flat = samples_flat['prod_mu'].reshape(N_POSTERIOR, N_PRODUCTS)
mu_hier = samples_hier['prod_mu'].reshape(N_POSTERIOR, N_PRODUCTS)

width_flat = (mu_flat.quantile(0.95, dim=0) - mu_flat.quantile(0.05, dim=0)).detach().flatten()
width_hier = (mu_hier.quantile(0.95, dim=0) - mu_hier.quantile(0.05, dim=0)).detach().flatten()

fig, ax = plt.subplots(figsize=(10, 4))
x = range(N_PRODUCTS)
ax.bar([xi-0.2 for xi in x], width_flat.numpy(), width=0.35,
       alpha=0.8, label='Flat model', color='tab:blue')
ax.bar([xi+0.2 for xi in x], width_hier.numpy(), width=0.35,
       alpha=0.8, label='Hierarchical model', color='tab:orange')
ax.set_xlabel('Product'); ax.set_ylabel('90% CI width (log-scale)')
ax.set_title('Posterior Uncertainty: Flat vs Hierarchical\\n'
             '(narrower = more precise estimate)')
ax.legend(); plt.tight_layout(); plt.show()

reduction = (width_flat - width_hier) / width_flat * 100
print(f'Average CI width reduction from partial pooling: '
      f'{reduction.mean():.1f}%')
"""))

cells.append(code("""\
# ── Posterior of seasonal effects ────────────────────────────
DOW = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']

season_post = samples_hier['season_raw']    # (N_POSTERIOR, 7)
# centre each sample for identifiability
season_post = season_post - season_post.mean(-1, keepdim=True)

season_mean = season_post.mean(0).detach().numpy()
season_lo   = season_post.quantile(0.05, dim=0).detach().numpy()
season_hi   = season_post.quantile(0.95, dim=0).detach().numpy()

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(DOW, TRUE_SEASON.numpy(), 'r--o', lw=2, label='True season')
ax.errorbar(DOW, season_mean, yerr=[season_mean-season_lo, season_hi-season_mean],
            fmt='b-o', capsize=4, label='Posterior mean ± 90% CI')
ax.axhline(0, color='grey', lw=0.8, ls='--')
ax.set_title('Recovered Weekly Seasonality (log-scale)')
ax.set_ylabel('Seasonal effect'); ax.legend()
plt.tight_layout(); plt.show()
"""))

# ─────────────────────────────────────────────────────────────
# SECTION 8 — VI vs MCMC
# ─────────────────────────────────────────────────────────────
cells.append(md("""\
## 8. VI vs MCMC (NUTS)

Variational Inference is fast but approximate — it minimises KL divergence
to a factored Gaussian family, which may underestimate posterior variance
(a known limitation of mean-field VI).

**NUTS (No-U-Turn Sampler)** is an adaptive Hamiltonian Monte Carlo method
that produces asymptotically exact posterior samples.

We compare the two on the hierarchical model to validate that VI is a
reasonable approximation here.

> ⏱ MCMC is slower than VI. We use 200 warm-up + 400 sample iterations
> for demonstration. For a production run, increase `num_samples` to 2000+.
"""))

cells.append(code("""\
pyro.set_rng_seed(42)

nuts_kernel = NUTS(model_hierarchical, adapt_step_size=True,
                   jit_compile=False)
mcmc = MCMC(nuts_kernel, num_samples=400, warmup_steps=200,
            disable_progbar=False)

mcmc.run(product_ids_train, category_ids, dow_ids_train,
         N_PRODUCTS, N_CATEGORIES, sales_train_flat)

mcmc_samples = mcmc.get_samples()
print('MCMC finished. Samples collected:', {k: v.shape
      for k, v in mcmc_samples.items()})
"""))

cells.append(code("""\
# ── Compare VI vs MCMC posteriors for prod_mu ────────────────
fig, axes = plt.subplots(3, 5, figsize=(16, 8))

for p, ax in enumerate(axes.flat):
    vi_post   = samples_hier['prod_mu'].reshape(N_POSTERIOR, N_PRODUCTS)[:, p].detach().numpy()
    mcmc_post = mcmc_samples['prod_mu'].reshape(-1, N_PRODUCTS)[:, p].detach().numpy()

    ax.hist(vi_post,   bins=30, alpha=0.5, density=True,
            color='steelblue', label='VI')
    ax.hist(mcmc_post, bins=30, alpha=0.5, density=True,
            color='darkorange', label='MCMC')
    ax.axvline(TRUE_PROD_MU[p].item(), color='red', lw=1.5,
               ls='--', label='True')
    ax.set_title(PRODUCT_NAMES[p], fontsize=8)
    ax.set_xlabel('μ', fontsize=8); ax.set_yticks([])
    if p == 0: ax.legend(fontsize=7)

plt.suptitle('Posterior of Product Log-Demands: VI (blue) vs MCMC (orange)',
             fontsize=13, y=1.01)
plt.tight_layout(); plt.show()
"""))

cells.append(code("""\
# ── Numerical comparison: posterior mean and std ─────────────
vi_means   = samples_hier['prod_mu'].reshape(N_POSTERIOR, N_PRODUCTS).mean(dim=0).detach().numpy().flatten()
mcmc_means = mcmc_samples['prod_mu'].reshape(-1, N_PRODUCTS).mean(dim=0).detach().numpy().flatten()
vi_std     = samples_hier['prod_mu'].reshape(N_POSTERIOR, N_PRODUCTS).std(dim=0).detach().numpy().flatten()
mcmc_std   = mcmc_samples['prod_mu'].reshape(-1, N_PRODUCTS).std(dim=0).detach().numpy().flatten()

df_compare = pd.DataFrame({
    'Product'    : PRODUCT_NAMES,
    'True μ'     : TRUE_PROD_MU.numpy().round(3),
    'VI mean'    : vi_means.round(3),
    'MCMC mean'  : mcmc_means.round(3),
    'VI std'     : vi_std.round(3),
    'MCMC std'   : mcmc_std.round(3),
})
print(df_compare.to_string(index=False))

# Mean absolute error
vi_mae   = np.abs(vi_means   - TRUE_PROD_MU.numpy()).mean()
mcmc_mae = np.abs(mcmc_means - TRUE_PROD_MU.numpy()).mean()
print(f'\\nMAE vs true — VI: {vi_mae:.3f}  |  MCMC: {mcmc_mae:.3f}')
print('(Both should be small if inference is working correctly)')
"""))

# ─────────────────────────────────────────────────────────────
# SECTION 9 — Newsvendor Optimisation
# ─────────────────────────────────────────────────────────────
cells.append(md("""\
## 9. The Newsvendor Problem

### Theory

The **Newsvendor Problem** is a fundamental model in inventory theory:

> A retailer must order $q$ units *before* demand $D$ is realised.
> * If $q < D$: underage cost $c_u$ per unmet unit (lost profit).
> * If $q > D$: overage cost $c_o$ per excess unit (waste/holding).

The optimal order quantity is:

$$q^* = F_D^{-1}\\!\\left(\\frac{c_u}{c_u + c_o}\\right)$$

where $F_D$ is the CDF of demand. This is the **critical ratio** — the
quantile of the demand distribution that balances the two cost types.

### Bayesian twist

Instead of using a parametric demand distribution with point-estimated
parameters, we use the **posterior predictive distribution**:

$$p(y^* \\mid \\text{data}) = \\int p(y^* \\mid \\theta)\\, p(\\theta \\mid \\text{data})\\, d\\theta$$

This propagates **parameter uncertainty** into the order decision —
making it more robust to estimation error, especially for new products
or those with limited data.

### Cost structures

We assign different cost structures per category, reflecting real-world
business logic:

| Category    | $c_u$ | $c_o$ | Critical ratio | Rationale |
|-------------|-------|-------|----------------|-----------|
| Food        |   5   |  15   |   0.25         | Perishable — overstock is expensive |
| Clothing    |  12   |   8   |   0.60         | Seasonal — understock hurts revenue |
| Electronics |  20   |   4   |   0.83         | High margin — never run out |
"""))

cells.append(code("""\
# ── Cost parameters ──────────────────────────────────────────
COST_U = [5,  5,  12, 12, 12, 20, 20, 20, 20, 20,  5,  5,  5,  5,  5]
COST_O = [15, 15,  8,  8,  8,  4,  4,  4,  4,  4, 15, 15, 15, 15, 15]

# Re-order: Food(0-4), Clothing(5-9), Electronics(10-14)
# Matches our product ordering: Food 0-4, Clothing 5-9, Electronics 10-14
COST_U = ([5]*5) + ([12]*5) + ([20]*5)
COST_O = ([15]*5) + ([8]*5) + ([4]*5)

def newsvendor_order(demand_samples, cu, co):
    \"\"\"
    Optimal order from the Newsvendor formula.
    q* = F^{-1}(cu / (cu + co))
    Returns (q_star, q_naive) where q_naive = posterior mean.
    \"\"\"
    cr   = cu / (cu + co)                       # critical ratio
    q_star  = float(demand_samples.float().quantile(cr))
    q_naive = float(demand_samples.float().mean())
    return round(q_star), round(q_naive)

def expected_cost(order_qty, demand_samples, cu, co):
    \"\"\"Expected cost under a given order quantity.\"\"\"
    d         = demand_samples.float()
    underage  = torch.clamp(d - order_qty, min=0)
    overage   = torch.clamp(order_qty - d, min=0)
    return float((cu * underage + co * overage).mean())


print('Cost structure:')
for c, name in enumerate(CATEGORY_NAMES):
    print(f'  {name:12s}  cu={COST_U[c*5]}  co={COST_O[c*5]}  '
          f'CR={COST_U[c*5]/(COST_U[c*5]+COST_O[c*5]):.2f}')
"""))

cells.append(code("""\
# ── Posterior predictive for test period ─────────────────────
# Build index arrays for the test window
product_ids_test = torch.arange(N_PRODUCTS).repeat_interleave(N_TEST)
dow_ids_test     = torch.arange(N_TEST).repeat(N_PRODUCTS) % 7

post_pred = Predictive(model_hierarchical, guide=guide_hier,
                       num_samples=4000,
                       return_sites=['sales'])

pred_samples = post_pred(product_ids_test, category_ids, dow_ids_test,
                         N_PRODUCTS, N_CATEGORIES)

# pred_sales: (4000, N_PRODUCTS*N_TEST) — reshape to (4000, N_PRODUCTS, N_TEST)
n_pred = pred_samples['sales'].shape[0]
pred_sales = pred_samples['sales'].reshape(n_pred, N_PRODUCTS, N_TEST)
print(f'Posterior predictive shape: {pred_sales.shape}')
print('(samples, products, days)')
"""))

cells.append(code("""\
# ── Compute optimal orders for each product × day ────────────
results = []

for p in range(N_PRODUCTS):
    cu, co = COST_U[p], COST_O[p]
    for day in range(N_TEST):
        demand_post = pred_sales[:, p, day]           # (4000,)
        true_demand = sales_test[p, day].item()

        q_star, q_naive = newsvendor_order(demand_post, cu, co)

        cost_star  = (cu * max(true_demand - q_star,  0)
                    + co * max(q_star  - true_demand, 0))
        cost_naive = (cu * max(true_demand - q_naive, 0)
                    + co * max(q_naive - true_demand, 0))

        results.append({
            'product'     : PRODUCT_NAMES[p],
            'category'    : CATEGORY_NAMES[p // N_PER_CAT],
            'day'         : day,
            'true_demand' : true_demand,
            'q_star'      : q_star,
            'q_naive'     : q_naive,
            'cost_star'   : cost_star,
            'cost_naive'  : cost_naive,
            'cu': cu, 'co': co,
        })

df = pd.DataFrame(results)
print(df.head(10).to_string(index=False))
"""))

# ─────────────────────────────────────────────────────────────
# SECTION 10 — Results
# ─────────────────────────────────────────────────────────────
cells.append(md("## 10. Results & Cost Comparison"))

cells.append(code("""\
# ── Aggregate cost by category ───────────────────────────────
summary = (df.groupby('category')[['cost_star','cost_naive']]
             .sum()
             .rename(columns={'cost_star':'Bayesian optimal',
                               'cost_naive':'Naive (mean forecast)'}))
summary['Savings (%)'] = ((summary['Naive (mean forecast)']
                           - summary['Bayesian optimal'])
                           / summary['Naive (mean forecast)'] * 100).round(1)
print('Total cost over 14-day test window:')
print(summary.to_string())

total_star  = df['cost_star'].sum()
total_naive = df['cost_naive'].sum()
print(f'\\nOVERALL — Bayesian: {total_star:,.0f}  |  '
      f'Naive: {total_naive:,.0f}  |  '
      f'Savings: {(total_naive-total_star)/total_naive*100:.1f}%')
"""))

cells.append(code("""\
# ── Bar chart: total cost by category ────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Per-category totals
ax = axes[0]
cats    = summary.index.tolist()
x       = np.arange(len(cats))
w       = 0.35
ax.bar(x - w/2, summary['Naive (mean forecast)'], w,
       label='Naive (mean forecast)', color='tab:blue',   alpha=0.8)
ax.bar(x + w/2, summary['Bayesian optimal'],      w,
       label='Bayesian optimal',      color='tab:orange', alpha=0.8)
ax.set_xticks(x); ax.set_xticklabels(cats)
ax.set_ylabel('Total cost (14-day window)')
ax.set_title('Inventory Cost by Category')
ax.legend()

# Per-product breakdown
ax = axes[1]
prod_sum = (df.groupby('product')[['cost_star','cost_naive']].sum()
              .reset_index())
x2  = np.arange(N_PRODUCTS)
ax.bar(x2 - 0.2, prod_sum['cost_naive'].values, 0.35,
       color='tab:blue',   alpha=0.8, label='Naive')
ax.bar(x2 + 0.2, prod_sum['cost_star'].values,  0.35,
       color='tab:orange', alpha=0.8, label='Bayesian')
ax.set_xticks(x2)
ax.set_xticklabels([f'P{i}' for i in range(N_PRODUCTS)], rotation=45)
ax.set_ylabel('Total cost')
ax.set_title('Inventory Cost per Product')
ax.legend()

plt.tight_layout(); plt.show()
"""))

cells.append(code("""\
# ── Posterior predictive vs actuals for 2 example products ───
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, p in zip(axes, [0, 10]):   # Food #1 and Electronics #1
    name = PRODUCT_NAMES[p]
    cu, co = COST_U[p], COST_O[p]
    cr = cu / (cu + co)

    for day in range(N_TEST):
        post = pred_sales[:, p, day].float()
        lo   = float(post.quantile(0.1))
        mid  = float(post.quantile(0.5))
        hi   = float(post.quantile(0.9))
        q_s, q_n = newsvendor_order(post, cu, co)

        ax.vlines(day, lo, hi, color='steelblue', alpha=0.4, lw=4)
        ax.scatter(day, mid,          color='steelblue',  s=30, zorder=4)
        ax.scatter(day, sales_test[p, day].item(),
                   color='black', s=50, marker='D', zorder=5)
        ax.scatter(day, q_s,  color='tab:orange', s=40,
                   marker='^', zorder=6)

    handles = [
        mpatches.Patch(color='steelblue', alpha=0.5,
                       label='80% predictive interval'),
        plt.Line2D([0],[0], marker='D', color='w',
                   markerfacecolor='black', ms=8, label='True demand'),
        plt.Line2D([0],[0], marker='^', color='w',
                   markerfacecolor='tab:orange', ms=8,
                   label=f'Bayesian q* (CR={cr:.2f})'),
    ]
    ax.legend(handles=handles, fontsize=9)
    ax.set_title(f'{name}  |  cu={cu}, co={co}')
    ax.set_xlabel('Test day'); ax.set_ylabel('Units')

plt.suptitle('Posterior Predictive Intervals & Optimal Order Quantities',
             fontsize=13, y=1.01)
plt.tight_layout(); plt.show()
"""))

# ─────────────────────────────────────────────────────────────
# SECTION 11 — Conclusions
# ─────────────────────────────────────────────────────────────
cells.append(md("""\
## 11. Discussion & Conclusions

### Summary of findings

| Model | What it captures | Inference |
|---|---|---|
| Flat Bayesian NegBinomial | Independent product demands + seasonality | VI |
| **Hierarchical Bayesian NegBinomial** | Partial pooling across products & categories | VI + MCMC |

**Partial pooling** reduces posterior uncertainty (credible interval width)
by ~X% on average compared to the flat model — products with little data
borrow strength from their category.

**VI and MCMC** agree closely on posterior means, validating that the
`AutoNormal` guide is a reasonable approximation for this model.

**The Newsvendor layer** converts parameter uncertainty into a decision:
the Bayesian optimal policy achieves lower expected cost than the naive
mean-forecast policy, especially for products with asymmetric cost structures
(e.g., high-margin electronics with CR = 0.83).

### Limitations & extensions

1. **Real data** — the next step is to run this on the
   [M5 Walmart dataset](https://www.kaggle.com/competitions/m5-forecasting-accuracy)
   with 42 000+ time series.

2. **Price elasticity** — add a product-level price coefficient
   $\\beta_j$ and include daily price covariates.

3. **Trend** — extend the model with a slow-moving trend component
   (e.g., local linear trend via a state-space model, as in Week 7).

4. **Dynamic replenishment** — extend the static newsvendor to a
   multi-period inventory policy (e.g., base-stock policy with
   Bayesian updating as demand is observed).

5. **Non-parametric demand** — replace the NegBinomial with a
   Dirichlet process mixture for heavier tails.

### What this means for supply chain

Bayesian demand modelling is directly applicable to:
* **Safety stock optimisation** under uncertain demand
* **Service-level guarantees** (\"97.5% probability we don't stock out\")
* **New product launches** — cold-start via category-level priors
* **ABC analysis** — posterior variance identifies which products need
  more data collection
"""))

# ─────────────────────────────────────────────────────────────
# Write notebook
# ─────────────────────────────────────────────────────────────
nb.cells = cells

path = 'bayesian_demand_forecasting.ipynb'
with open(path, 'w') as f:
    nbf.write(nb, f)

print(f'Notebook written to {path}  ({len(cells)} cells)')
