from __future__ import annotations
import numpy as np
from typing import Tuple

def fit_standardizer(X_train: np.ndarray):
    # X_train : (n_traj, time_steps, feature_dim)
    eps: float = 1e-8
    
    # for each feature calculate the mean and std over all trajectories and time steps
    # sum of all values / (n_traj * time_steps) (for each feature)
    mean = X_train.mean(axis=(0, 1), keepdims=True)  
    std  = X_train.std(axis=(0, 1), keepdims=True) + eps

    # mean.shape == (1, 1, feature_dim)
    # std.shape == (1, 1, feature_dim)
    return mean, std

def apply_standardizer(A: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (A - mean) / std

def inverse_standardizer(A_std: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    # de-standardize: A = A_std * std + mean
    return A_std * std + mean