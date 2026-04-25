from dataclasses import dataclass

@dataclass
class Config:
    train_size = 0.66
    random_seed = 42
    n_steps = 3000
    lr = 0.005
    posterior_samples = 1000

cfg = Config()