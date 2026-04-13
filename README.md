# Model-Based Machine Learning – Group Project

## Project Description

This repository contains the code, notebooks, and report for our university
group project on **Model-Based Machine Learning**.

The project covers the full pipeline of a Bayesian modelling workflow:

1. **Exploratory Data Analysis (EDA)** – understanding the dataset through
   summary statistics and visualisations.
2. **Preprocessing** – cleaning raw data, handling missing values, and
   feature engineering.
3. **Bayesian Model** – building and training a probabilistic model (Bayesian
   linear regression implemented with [PyMC](https://www.pymc.io/)).
4. **Posterior Inference** – full posterior inference via NUTS MCMC, with
   optional mean-field variational inference (ADVI) as a faster alternative.
5. **Evaluation & Visualisation** – predictive metrics (RMSE, MAE, R²),
   convergence diagnostics, and publication-ready plots.

---

## Setup

### Prerequisites

- Python ≥ 3.10
- [pip](https://pip.pypa.io/) or [conda](https://docs.conda.io/)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/bodvarsdottirkristin/model-based.git
cd model-based

# 2. (Recommended) create an isolated environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch JupyterLab
jupyter lab
```

---

## Repo Structure

```
project-root/
├── data/
│   ├── raw/                  # Original, unmodified data files
│   └── processed/            # Cleaned / preprocessed data
├── notebooks/
│   ├── 01_eda.ipynb          # Exploratory data analysis
│   ├── 02_preprocessing.ipynb
│   ├── 03_model.ipynb        # Model definition and training
│   └── 04_evaluation.ipynb
├── src/
│   ├── __init__.py
│   ├── data_loader.py        # Data loading and preprocessing utilities
│   ├── model.py              # Bayesian model definition (BayesianModel class)
│   ├── inference.py          # Posterior inference (MCMC / VI)
│   ├── evaluation.py         # Metrics and model-comparison utilities
│   └── visualization.py     # Plotting helpers
├── tests/
│   └── test_model.py         # Basic unit tests
├── results/
│   └── figures/              # Saved plots and outputs
├── report/                   # LaTeX or PDF report files
├── requirements.txt          # Python dependencies
├── .gitignore
└── README.md
```

---

## Team Members
_Helga María Magnúsdóttir_

_Kolbrún Védís Jónsdóttir_

_Kristín Böðvarsdóttir_
